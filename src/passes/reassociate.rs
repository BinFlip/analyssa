//! Reassociation pass — reorders nested associative operations to enable
//! constant folding.
//!
//! # Algorithm
//!
//! Detects and rewrites nested binary operations where both the inner
//! and outer operations have a constant operand:
//!
//! - `(x + c1) + c2` → `x + (c1 + c2)` (and Sub/Mul/And/Or/Xor variants)
//! - `(x << c1) << c2` → `x << (c1 + c2)` (and the Shr variant)
//!
//! After rewriting, the two constant operands are combined into a single
//! constant value that a later constant-propagation pass can fold.
//!
//! # Details
//!
//! 1. Find all constant variables via [`SsaFunction::find_constants`].
//! 2. For each instruction, check if its defining op matches the outer
//!    pattern: one operand is a constant and the other is defined by the
//!    same op kind with a constant on the corresponding side.
//! 3. If the inner result has only one use, the rewrite is valid: the
//!    inner constant is replaced with the combined value, the inner op
//!    is rewritten as `x ∘ c1`, and the outer op is rewritten as a `Copy`
//!    of the inner result.
//! 4. The `combine` step uses `ptr_size` so wraparound on `NativeInt`
//!    constants matches the host's pointer width.
//!
//! # Supported Operations
//!
//! | Outer kind | Inner kind | Combine op | Example |
//! |-----------|-----------|-----------|---------|
//! | Add | Add | Add | `(x + 3) + 7 → x + 10` |
//! | Sub | Sub | Add | `(x - 3) - 2 → x - 5` |
//! | Mul | Mul | Mul | `(x * 3) * 5 → x * 15` |
//! | And | And | And | `(x & 0xF0) & 0x0F → x & 0` |
//! | Or | Or | Or | `(x \| 0xF0) \| 0x0F → x \| 0xFF` |
//! | Xor | Xor | Xor | `(x ^ 0xF0) ^ 0x0F → x ^ 0xFF` |
//! | Shl | Shl | Add | `(x << 2) << 3 → x << 5` |
//! | Shr | Shr | Add | `(x >> 2) >> 3 → x >> 5` |

use std::collections::{BTreeMap, HashSet};

use crate::{
    analysis::DefUseIndex,
    events::{EventKind, EventListener},
    ir::{
        function::{SsaEditOptions, SsaFunction},
        ops::SsaOp,
        value::ConstValue,
        variable::SsaVarId,
    },
    pointer::PointerSize,
    target::Target,
};

/// Run the reassociation pass on `ssa`.
///
/// Rewrites nested associative operations `(x op c1) op c2` into
/// `x op (c1 combined c2)` so the constants can be folded.
///
/// # Arguments
///
/// * `ssa` — The SSA function to transform in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::ConstantFolded`] events.
/// * `ptr_size` — Host pointer width (`Bit32` or `Bit64`). Affects
///   wraparound semantics for `NativeInt` constant combination.
///
/// # Returns
///
/// `true` if any operation was rewritten.
pub fn run<T, L>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
    ptr_size: PointerSize,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let constants = ssa.find_constants();
    let index = DefUseIndex::<T>::build_with_ops(ssa);
    let uses = ssa.count_uses();
    let candidates = find_candidates(ssa, &constants, &index, &uses);
    apply_reassociations(ssa, candidates, method, events, ptr_size)
}

/// A detected reassociation opportunity: `(base_var op const1) op const2`.
#[derive(Debug)]
struct ReassociationCandidate<T: Target> {
    /// Block containing the outer instruction.
    block_idx: usize,
    /// Index of the outer instruction within its block.
    instr_idx: usize,
    /// Destination variable of the outer operation's result.
    dest: SsaVarId,
    /// The non-constant base variable in the inner operation.
    base_var: SsaVarId,
    /// Variable holding the first constant `c1`.
    const1_var: SsaVarId,
    /// Variable holding the second constant `c2`.
    #[allow(dead_code)]
    const2_var: SsaVarId,
    /// Value of the first constant.
    const1_value: ConstValue<T>,
    /// Value of the second constant.
    const2_value: ConstValue<T>,
    /// Block containing the inner instruction.
    inner_block: usize,
    /// Index of the inner instruction within its block.
    inner_instr: usize,
    /// Destination variable of the inner operation's result.
    inner_dest: SsaVarId,
    /// The operation kind (must match for both inner and outer).
    op_kind: OpKind,
}

/// The kind of binary operation involved in a reassociation pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpKind {
    /// Integer addition (`Add`).
    Add,
    /// Integer subtraction (`Sub`).
    Sub,
    /// Integer multiplication (`Mul`).
    Mul,
    /// Bitwise AND (`And`).
    And,
    /// Bitwise OR (`Or`).
    Or,
    /// Bitwise XOR (`Xor`).
    Xor,
    /// Left shift (`Shl`).
    Shl,
    /// Right shift (`Shr`), with an `unsigned` flag.
    Shr { unsigned: bool },
}

impl OpKind {
    fn combine<T: Target>(
        self,
        c1: &ConstValue<T>,
        c2: &ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        match self {
            OpKind::Add | OpKind::Sub | OpKind::Shl | OpKind::Shr { .. } => c1.add(c2, ptr_size),
            OpKind::Mul => c1.mul(c2, ptr_size),
            OpKind::And => c1.bitwise_and(c2, ptr_size),
            OpKind::Or => c1.bitwise_or(c2, ptr_size),
            OpKind::Xor => c1.bitwise_xor(c2, ptr_size),
        }
    }

    fn name(self) -> &'static str {
        match self {
            OpKind::Add => "add",
            OpKind::Sub => "sub",
            OpKind::Mul => "mul",
            OpKind::And => "and",
            OpKind::Or => "or",
            OpKind::Xor => "xor",
            OpKind::Shl => "shl",
            OpKind::Shr { unsigned: false } => "shr",
            OpKind::Shr { unsigned: true } => "shr.un",
        }
    }

    fn combine_name(self) -> &'static str {
        match self {
            OpKind::Add | OpKind::Mul | OpKind::And | OpKind::Or | OpKind::Xor => self.name(),
            OpKind::Sub | OpKind::Shl | OpKind::Shr { .. } => "add",
        }
    }

    const fn is_commutative(self) -> bool {
        match self {
            OpKind::Add | OpKind::Mul | OpKind::And | OpKind::Or | OpKind::Xor => true,
            OpKind::Sub | OpKind::Shl | OpKind::Shr { .. } => false,
        }
    }
}

fn get_op_kind<T: Target>(op: &SsaOp<T>) -> Option<(OpKind, SsaVarId, SsaVarId, SsaVarId)> {
    match op {
        SsaOp::Add {
            dest, left, right, ..
        } => Some((OpKind::Add, *dest, *left, *right)),
        SsaOp::Sub {
            dest, left, right, ..
        } => Some((OpKind::Sub, *dest, *left, *right)),
        SsaOp::Mul {
            dest, left, right, ..
        } => Some((OpKind::Mul, *dest, *left, *right)),
        SsaOp::And {
            dest, left, right, ..
        } => Some((OpKind::And, *dest, *left, *right)),
        SsaOp::Or {
            dest, left, right, ..
        } => Some((OpKind::Or, *dest, *left, *right)),
        SsaOp::Xor {
            dest, left, right, ..
        } => Some((OpKind::Xor, *dest, *left, *right)),
        SsaOp::Shl {
            dest,
            value,
            amount,
            ..
        } => Some((OpKind::Shl, *dest, *value, *amount)),
        SsaOp::Shr {
            dest,
            value,
            amount,
            unsigned,
            ..
        } => Some((
            OpKind::Shr {
                unsigned: *unsigned,
            },
            *dest,
            *value,
            *amount,
        )),
        _ => None,
    }
}

fn make_op<T: Target>(kind: OpKind, dest: SsaVarId, left: SsaVarId, right: SsaVarId) -> SsaOp<T> {
    match kind {
        OpKind::Add => SsaOp::Add {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::Sub => SsaOp::Sub {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::Mul => SsaOp::Mul {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::And => SsaOp::And {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::Or => SsaOp::Or {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::Xor => SsaOp::Xor {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::Shl => SsaOp::Shl {
            dest,
            value: left,
            amount: right,
            flags: None,
        },
        OpKind::Shr { unsigned } => SsaOp::Shr {
            dest,
            value: left,
            amount: right,
            unsigned,
            flags: None,
        },
    }
}

fn find_candidates<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    index: &DefUseIndex<T>,
    uses: &BTreeMap<SsaVarId, usize>,
) -> Vec<ReassociationCandidate<T>> {
    let mut candidates = Vec::new();
    for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
        if let Some(candidate) =
            check_reassociation(instr.op(), block_idx, instr_idx, constants, index, uses)
        {
            candidates.push(candidate);
        }
    }
    candidates
}

fn check_reassociation<T: Target>(
    op: &SsaOp<T>,
    block_idx: usize,
    instr_idx: usize,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    index: &DefUseIndex<T>,
    uses: &BTreeMap<SsaVarId, usize>,
) -> Option<ReassociationCandidate<T>> {
    let (outer_kind, dest, outer_left, outer_right) = get_op_kind(op)?;
    let c2_value = constants.get(&outer_right)?;

    let (inner_block, inner_instr, inner_op) = index.full_definition(outer_left)?;
    let (inner_kind, inner_dest, inner_left, inner_right) = get_op_kind(inner_op)?;

    if inner_kind != outer_kind {
        return None;
    }
    let inner_uses = uses.get(&inner_dest).copied().unwrap_or(0);
    if inner_uses > 1 {
        return None;
    }

    if let Some(c1_value) = constants.get(&inner_right) {
        return Some(ReassociationCandidate {
            block_idx,
            instr_idx,
            dest,
            base_var: inner_left,
            const1_var: inner_right,
            const2_var: outer_right,
            const1_value: c1_value.clone(),
            const2_value: c2_value.clone(),
            inner_block,
            inner_instr,
            inner_dest,
            op_kind: outer_kind,
        });
    }

    if outer_kind.is_commutative() {
        if let Some(c1_value) = constants.get(&inner_left) {
            return Some(ReassociationCandidate {
                block_idx,
                instr_idx,
                dest,
                base_var: inner_right,
                const1_var: inner_left,
                const2_var: outer_right,
                const1_value: c1_value.clone(),
                const2_value: c2_value.clone(),
                inner_block,
                inner_instr,
                inner_dest,
                op_kind: outer_kind,
            });
        }
    }

    None
}

fn apply_reassociations<T, L>(
    ssa: &mut SsaFunction<T>,
    candidates: Vec<ReassociationCandidate<T>>,
    method: &T::MethodRef,
    events: &L,
    ptr_size: PointerSize,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut modified: HashSet<(usize, usize)> = HashSet::new();
    let mut changed = false;

    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        for candidate in candidates {
            // Skip if either position has already been rewritten by a prior
            // overlapping candidate. This avoids ambiguous chain rewrites.
            if modified.contains(&(candidate.inner_block, candidate.inner_instr))
                || modified.contains(&(candidate.block_idx, candidate.instr_idx))
            {
                continue;
            }

            let Some(combined) = candidate.op_kind.combine(
                &candidate.const1_value,
                &candidate.const2_value,
                ptr_size,
            ) else {
                continue;
            };

            let Some(const_instr_idx) =
                editor
                    .function()
                    .block(candidate.inner_block)
                    .and_then(|block| {
                        block.instructions().iter().position(|instr| {
                            matches!(
                                instr.op(),
                                SsaOp::Const { dest, .. } if *dest == candidate.const1_var
                            )
                        })
                    })
            else {
                continue;
            };

            let inner_exists = editor
                .function()
                .block(candidate.inner_block)
                .and_then(|block| block.instruction(candidate.inner_instr))
                .is_some();
            let outer_exists = editor
                .function()
                .block(candidate.block_idx)
                .and_then(|block| block.instruction(candidate.instr_idx))
                .is_some();
            if !inner_exists || !outer_exists {
                continue;
            }

            editor.replace_instruction_op(
                candidate.inner_block,
                const_instr_idx,
                SsaOp::Const {
                    dest: candidate.const1_var,
                    value: combined.clone(),
                },
            )?;
            editor.replace_instruction_op(
                candidate.inner_block,
                candidate.inner_instr,
                make_op(
                    candidate.op_kind,
                    candidate.inner_dest,
                    candidate.base_var,
                    candidate.const1_var,
                ),
            )?;
            editor.replace_instruction_op(
                candidate.block_idx,
                candidate.instr_idx,
                SsaOp::Copy {
                    dest: candidate.dest,
                    src: candidate.inner_dest,
                },
            )?;

            modified.insert((candidate.inner_block, candidate.inner_instr));
            modified.insert((candidate.block_idx, candidate.instr_idx));

            let event = crate::events::Event {
                kind: EventKind::ConstantFolded,
                method: Some(method.clone()),
                location: Some(candidate.instr_idx),
                message: format!(
                    "reassociate: (x {} c1) {} c2 → x {} (c1 {} c2)",
                    candidate.op_kind.name(),
                    candidate.op_kind.name(),
                    candidate.op_kind.name(),
                    candidate.op_kind.combine_name()
                ),
                pass: None,
            };
            events.push(event);
            changed = true;
        }
        Ok(())
    });

    if result.is_err() {
        return false;
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{run_mock_pass_boundary, MockTarget, MockType},
    };

    fn add(
        c1: ConstValue<MockTarget>,
        c2: ConstValue<MockTarget>,
    ) -> Option<ConstValue<MockTarget>> {
        OpKind::Add.combine(&c1, &c2, PointerSize::Bit64)
    }

    #[test]
    fn op_kind_combine_add() {
        assert_eq!(
            add(ConstValue::I32(5), ConstValue::I32(3)),
            Some(ConstValue::I32(8))
        );
    }

    #[test]
    fn op_kind_combine_xor() {
        let r = OpKind::Xor.combine(
            &ConstValue::<MockTarget>::I32(0xF0),
            &ConstValue::<MockTarget>::I32(0x0F),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(0xFF)));
    }

    #[test]
    fn op_kind_combine_mul() {
        let r = OpKind::Mul.combine(
            &ConstValue::<MockTarget>::I32(7),
            &ConstValue::<MockTarget>::I32(11),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(77)));
    }

    #[test]
    fn op_kind_combine_and() {
        let r = OpKind::And.combine(
            &ConstValue::<MockTarget>::I32(0xF0),
            &ConstValue::<MockTarget>::I32(0x33),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(0x30)));
    }

    #[test]
    fn op_kind_combine_or() {
        let r = OpKind::Or.combine(
            &ConstValue::<MockTarget>::I32(0xF0),
            &ConstValue::<MockTarget>::I32(0x0F),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(0xFF)));
    }

    /// Sub uses Add internally for combining (`(x - c1) - c2` → `x - (c1 + c2)`).
    #[test]
    fn op_kind_combine_sub() {
        let r = OpKind::Sub.combine(
            &ConstValue::<MockTarget>::I32(5),
            &ConstValue::<MockTarget>::I32(3),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(8)));
    }

    /// Shl uses Add for shift-amount combining (`(x << c1) << c2` → `x << (c1 + c2)`).
    #[test]
    fn op_kind_combine_shl() {
        let r = OpKind::Shl.combine(
            &ConstValue::<MockTarget>::I32(2),
            &ConstValue::<MockTarget>::I32(3),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(5)));
    }

    #[test]
    fn op_kind_combine_shr() {
        let r = OpKind::Shr { unsigned: false }.combine(
            &ConstValue::<MockTarget>::I32(2),
            &ConstValue::<MockTarget>::I32(3),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(5)));
    }

    #[test]
    fn op_kind_combine_shr_unsigned() {
        let r = OpKind::Shr { unsigned: true }.combine(
            &ConstValue::<MockTarget>::I32(1),
            &ConstValue::<MockTarget>::I32(2),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(3)));
    }

    #[test]
    fn op_kind_is_commutative() {
        assert!(OpKind::Add.is_commutative());
        assert!(OpKind::Mul.is_commutative());
        assert!(OpKind::And.is_commutative());
        assert!(OpKind::Or.is_commutative());
        assert!(OpKind::Xor.is_commutative());
        assert!(!OpKind::Sub.is_commutative());
        assert!(!OpKind::Shl.is_commutative());
        assert!(!OpKind::Shr { unsigned: false }.is_commutative());
        assert!(!OpKind::Shr { unsigned: true }.is_commutative());
    }

    #[test]
    fn op_kind_combine_name_associative() {
        // Associative ops report their own name as combine_name.
        assert_eq!(OpKind::Add.combine_name(), "add");
        assert_eq!(OpKind::Mul.combine_name(), "mul");
        assert_eq!(OpKind::And.combine_name(), "and");
        assert_eq!(OpKind::Or.combine_name(), "or");
        assert_eq!(OpKind::Xor.combine_name(), "xor");
    }

    #[test]
    fn op_kind_combine_name_non_associative() {
        // Non-associative ops fall back to add for the constant fold.
        assert_eq!(OpKind::Sub.combine_name(), "add");
        assert_eq!(OpKind::Shl.combine_name(), "add");
        assert_eq!(OpKind::Shr { unsigned: false }.combine_name(), "add");
        assert_eq!(OpKind::Shr { unsigned: true }.combine_name(), "add");
    }

    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
    }

    fn local_at(
        ssa: &mut SsaFunction<MockTarget>,
        idx: u16,
        block: usize,
        instr: usize,
    ) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::instruction(block, instr),
            MockType::I32,
        )
    }

    fn run_reassociate(
        ssa: &mut SsaFunction<MockTarget>,
        label: &str,
        log: &EventLog<MockTarget>,
    ) -> bool {
        run_mock_pass_boundary(ssa, label, |ssa| run(ssa, &0u32, log, PointerSize::Bit64))
    }

    #[test]
    fn reassociate_add_combines_constants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(7),
        }));
        // (x + 3) + 7 → x + (3 + 7) = x + 10
        block.add_instruction(instr(SsaOp::Add {
            dest: inner,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(outer) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "add reassociation", &log);
        assert!(changed, "constant add reassociation should fire");
        assert!(log.has(EventKind::ConstantFolded));
    }

    #[test]
    fn reassociate_mul_combines_constants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(5),
        }));
        // (x * 3) * 5 → x * (3 * 5) = x * 15
        block.add_instruction(instr(SsaOp::Mul {
            dest: inner,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(outer) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "mul reassociation", &log);
        assert!(changed, "constant mul reassociation should fire");
    }

    #[test]
    fn reassociate_and_combines_constants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(0xFF),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(0xF0),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(0x0F),
        }));
        // (x & 0xF0) & 0x0F → x & (0xF0 & 0x0F) = x & 0
        block.add_instruction(instr(SsaOp::And {
            dest: inner,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::And {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(outer) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "and reassociation", &log);
        assert!(changed, "constant and reassociation should fire");
    }

    #[test]
    fn reassociate_or_combines_constants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(0x00),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(0xF0),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(0x0F),
        }));
        // (x | 0xF0) | 0x0F → x | (0xF0 | 0x0F) = x | 0xFF
        block.add_instruction(instr(SsaOp::Or {
            dest: inner,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Or {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(outer) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "or reassociation", &log);
        assert!(changed, "constant or reassociation should fire");
    }

    #[test]
    fn no_candidates_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let y = local_at(&mut ssa, 1, 0, 1);
        let z = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: y,
            value: ConstValue::I32(2),
        }));
        // No nested expression with constants
        block.add_instruction(instr(SsaOp::Add {
            dest: z,
            left: x,
            right: y,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(z) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "no-candidate reassociation", &log);
        assert!(
            !changed,
            "no nested constant pattern should not trigger reassociation"
        );
    }

    #[test]
    fn reassociate_commutative_constant_on_left() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(7),
        }));
        // (3 + x) + 7 = x + (3 + 7) = x + 10
        block.add_instruction(instr(SsaOp::Add {
            dest: inner,
            left: c1,
            right: x,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(outer) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "commutative-left reassociation", &log);
        assert!(
            changed,
            "commutative reassociation with constant on left should fire"
        );
    }

    #[test]
    fn reassociate_shift_combines_amounts() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(3),
        }));
        // (x << 2) << 3 → x << (2 + 3) = x << 5
        block.add_instruction(instr(SsaOp::Shl {
            dest: inner,
            value: x,
            amount: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Shl {
            dest: outer,
            value: inner,
            amount: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(outer) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "shift reassociation", &log);
        assert!(changed, "shift reassociation should fire");
    }

    #[test]
    fn reassociate_sub_combines_constants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(2),
        }));
        // (x - 3) - 2 → x - (3 + 2) = x - 5
        block.add_instruction(instr(SsaOp::Sub {
            dest: inner,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Sub {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(outer) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "sub reassociation", &log);
        assert!(changed, "sub reassociation should fire");
    }

    #[test]
    fn reassociate_empty_function() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "empty reassociation", &log);
        assert!(!changed);
    }
}
