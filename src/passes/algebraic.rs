//! Algebraic simplification pass — transforms redundant operations using
//! the identities catalogued in [`crate::analysis::algebraic`].
//!
//! Pure-SSA and target-agnostic. Hosts can call [`run`] directly or use
//! [`crate::passes::AlgebraicSimplificationPass`] with the scheduler.
//!
//! # Algorithm
//!
//! 1. Find all constant-valued variables via [`SsaFunction::find_constants`].
//! 2. For each instruction, call [`simplify_op`] which checks identities:
//!    - `x ^ x` → 0, `x - x` → 0, `x & x` → `x`, `x | x` → `x`
//!    - `x + 0` → `x`, `x * 1` → `x`, `x * 0` → 0
//!    - `x << 0` → `x`, `x >> 0` → `x`
//!    - `ceq(x, x)` → 1, `clt(x, x)` → 0, `cgt(x, x)` → 0
//!    - `div(x, 1)` → `x`, `rem(x, 1)` → 0
//! 3. Replace matching operations with `Const` or `Copy` ops.
//! 4. Report one [`EventKind::ConstantFolded`] event per replacement.
//!
//! # Complexity
//!
//! O(n) in the number of instructions — a single linear scan plus
//! constant-table lookups.

use std::collections::BTreeMap;

use crate::{
    analysis::algebraic::{simplify_op, SimplifyResult},
    events::{EventKind, EventListener},
    ir::{
        function::{SsaEditOptions, SsaFunction},
        ops::SsaOp,
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// Run algebraic simplification on `ssa`.
///
/// Scans all instructions, applies algebraic identities, and replaces
/// redundant computations with constants or copy operations.
///
/// # Arguments
///
/// * `ssa` — The SSA function to simplify in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink receiving one [`EventKind::ConstantFolded`]
///   event per simplification. Pass [`crate::NullListener`] to discard.
///
/// # Returns
///
/// `true` if any operation was rewritten, `false` otherwise.
pub fn run<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let constants = ssa.find_constants();
    let candidates = find_candidates(ssa, &constants);
    apply_simplifications(ssa, candidates, method, events)
}

/// The type of simplification applied to a redundant operation.
#[derive(Debug, Clone)]
enum Simplification<T: Target> {
    /// Replace the operation's result with a known constant value.
    Constant(ConstValue<T>),
    /// Replace the operation's result with a copy of another variable.
    Copy(SsaVarId),
}

/// A single instruction identified as a candidate for algebraic simplification.
#[derive(Debug)]
struct SimplificationCandidate<T: Target> {
    /// Index of the block containing the instruction.
    block_idx: usize,
    /// Index of the instruction within the block.
    instr_idx: usize,
    /// Destination variable of the instruction's result.
    dest: SsaVarId,
    /// The simplification to apply (constant or copy).
    simplification: Simplification<T>,
    /// Human-readable description of the applied identity.
    description: &'static str,
}

fn find_candidates<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Vec<SimplificationCandidate<T>> {
    let mut candidates = Vec::new();
    for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
        let op = instr.op();
        if let Some(candidate) = check_simplification(op, block_idx, instr_idx, constants) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn check_simplification<T: Target>(
    op: &SsaOp<T>,
    block_idx: usize,
    instr_idx: usize,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Option<SimplificationCandidate<T>> {
    if op.defs().count() != 1 {
        return None;
    }
    let dest = op.dest()?;
    match simplify_op(op, constants) {
        SimplifyResult::Constant(value) => Some(SimplificationCandidate {
            block_idx,
            instr_idx,
            dest,
            simplification: Simplification::Constant(value),
            description: "algebraic → const",
        }),
        SimplifyResult::Copy(src) => Some(SimplificationCandidate {
            block_idx,
            instr_idx,
            dest,
            simplification: Simplification::Copy(src),
            description: "algebraic → copy",
        }),
        SimplifyResult::None => None,
    }
}

fn apply_simplifications<T, L>(
    ssa: &mut SsaFunction<T>,
    candidates: Vec<SimplificationCandidate<T>>,
    method: &T::MethodRef,
    events: &L,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut changed = false;
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        for candidate in candidates {
            let new_op = match candidate.simplification {
                Simplification::Constant(value) => SsaOp::Const {
                    dest: candidate.dest,
                    value,
                },
                Simplification::Copy(src) => SsaOp::Copy {
                    dest: candidate.dest,
                    src,
                },
            };
            editor.replace_instruction_op(candidate.block_idx, candidate.instr_idx, new_op)?;
            changed = true;

            let event = crate::events::Event {
                kind: EventKind::ConstantFolded,
                method: Some(method.clone()),
                location: Some(candidate.instr_idx),
                message: candidate.description.to_string(),
                pass: None,
            };
            events.push(event);
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
            variable::{DefSite, VariableOrigin},
        },
        testing::{mock_op_at, run_mock_pass_boundary, MockTarget, MockType},
        NullListener,
    };

    fn build_function() -> SsaFunction<MockTarget> {
        SsaFunction::<MockTarget>::new(0, 0)
    }

    /// Build a function with a single block that performs `dest = left ^ right`.
    fn build_xor(
        left_var: SsaVarId,
        right_var: SsaVarId,
        dest_var: SsaVarId,
    ) -> SsaFunction<MockTarget> {
        let mut f = build_function();
        f.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        f.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        f.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );

        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        // Const 5 -> left
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: left_var,
            value: ConstValue::I32(5),
        }));
        // Const 5 -> right
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: right_var,
            value: ConstValue::I32(5),
        }));
        // dest = left ^ right (which equals left ^ left's value -> 0 if left == right)
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Xor {
            dest: dest_var,
            left: left_var,
            right: left_var,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(dest_var),
        }));
        f.add_block(block);
        f.recompute_uses();
        f
    }

    #[test]
    fn rewrites_xor_self_to_const_zero() {
        let left = SsaVarId::from_index(0);
        let right = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let mut ssa = build_xor(left, right, dest);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0xABu32;

        let changed = run_mock_pass_boundary(&mut ssa, "xor-self algebraic rewrite", |ssa| {
            run(ssa, &method, &log)
        });
        assert!(changed);
        assert_eq!(log.count_kind(EventKind::ConstantFolded), 1);

        // The third instruction should now be a Const, not Xor.
        match mock_op_at(&ssa, 0, 2) {
            SsaOp::Const {
                value: ConstValue::I32(0),
                dest: d,
            } => assert_eq!(*d, dest),
            other => panic!("expected Const I32(0), got {:?}", other),
        }
    }

    #[test]
    fn null_listener_runs_silently() {
        let left = SsaVarId::from_index(0);
        let right = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let mut ssa = build_xor(left, right, dest);
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "null-listener algebraic rewrite", |ssa| {
            run(ssa, &method, &NullListener)
        });
        assert!(changed);
    }

    #[test]
    fn no_op_returns_false_and_records_nothing() {
        // Build an empty function with no instructions to simplify.
        let mut ssa = build_function();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;

        let changed = run_mock_pass_boundary(&mut ssa, "empty algebraic pass", |ssa| {
            run(ssa, &method, &log)
        });
        assert!(!changed);
        assert!(log.is_empty());
    }

    // ============================================================================
    // Exercises the private `check_simplification` helper directly. These tests cover
    // div-by-1, rem-by-1, and self-comparison patterns that the public
    // `run()` would also catch but at lower granularity.
    // ============================================================================

    #[test]
    fn check_simplification_div_by_one() {
        let left = SsaVarId::from_index(0);
        let right = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(right, ConstValue::I32(1))].into();
        let op: SsaOp<MockTarget> = SsaOp::Div {
            dest,
            left,
            right,
            unsigned: false,
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("div-by-1 should simplify");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == left));
    }

    #[test]
    fn check_simplification_rem_by_one() {
        let left = SsaVarId::from_index(0);
        let right = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(right, ConstValue::I32(1))].into();
        let op: SsaOp<MockTarget> = SsaOp::Rem {
            dest,
            left,
            right,
            unsigned: false,
            flags: None,
        };
        let candidate = check_simplification(&op, 0, 0, &constants).expect("rem-by-1 simplifies");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(0))
        ));
    }

    #[test]
    fn check_simplification_ceq_same_var() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Ceq {
            dest,
            left: x,
            right: x,
        };
        let candidate = check_simplification(&op, 0, 0, &constants).expect("ceq(x,x) simplifies");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(1))
        ));
    }

    #[test]
    fn check_simplification_clt_same_var() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Clt {
            dest,
            left: x,
            right: x,
            unsigned: false,
        };
        let candidate = check_simplification(&op, 0, 0, &constants).expect("clt(x,x) simplifies");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(0))
        ));
    }

    #[test]
    fn check_simplification_cgt_same_var() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Cgt {
            dest,
            left: x,
            right: x,
            unsigned: false,
        };
        let candidate = check_simplification(&op, 0, 0, &constants).expect("cgt(x,x) simplifies");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(0))
        ));
    }

    #[test]
    fn check_simplification_add_identity() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Add {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("x + 0 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    #[test]
    fn check_simplification_mul_identity() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(1))].into();
        let op: SsaOp<MockTarget> = SsaOp::Mul {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("x * 1 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    #[test]
    fn check_simplification_mul_zero() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Mul {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("x * 0 should simplify to 0");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(0))
        ));
    }

    #[test]
    fn check_simplification_sub_self() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Sub {
            dest,
            left: x,
            right: x,
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("x - x should simplify to 0");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(0))
        ));
    }

    #[test]
    fn check_simplification_and_self() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::And {
            dest,
            left: x,
            right: x,
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("x & x should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    #[test]
    fn check_simplification_or_self() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Or {
            dest,
            left: x,
            right: x,
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("x | x should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    #[test]
    fn check_simplification_and_with_all_ones() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(-1))].into();
        let op: SsaOp<MockTarget> = SsaOp::And {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("x & -1 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    #[test]
    fn check_simplification_or_with_zero() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Or {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("x | 0 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    #[test]
    fn check_simplification_xor_with_zero() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Xor {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("x ^ 0 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    #[test]
    fn check_simplification_shl_by_zero() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Shl {
            dest,
            value: x,
            amount: SsaVarId::from_index(1),
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("x << 0 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    #[test]
    fn check_simplification_shr_by_zero() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Shr {
            dest,
            value: x,
            amount: SsaVarId::from_index(1),
            unsigned: false,
            flags: None,
        };
        let result = check_simplification(&op, 0, 0, &constants);
        let candidate = result.expect("x >> 0 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    #[test]
    fn check_simplification_no_match_for_unknown_op() {
        let x = SsaVarId::from_index(0);
        let y = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Add {
            dest,
            left: x,
            right: y,
            flags: None,
        };
        assert!(check_simplification(&op, 0, 0, &constants).is_none());
    }

    #[test]
    fn run_simplifies_multiple_ops_in_single_function() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);

        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );
        ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(0, 3),
            MockType::I32,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(0),
        }));
        // x + 0 -> x
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        // x - x -> 0
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Sub {
            dest: v3,
            left: v0,
            right: v0,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v3) }));
        ssa.add_block(block);

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        ssa.recompute_uses();
        let changed =
            run_mock_pass_boundary(&mut ssa, "multiple algebraic simplifications", |ssa| {
                run(ssa, &method, &log)
            });
        assert!(changed);
        // Should have simplified both ops
        assert!(log.count_kind(EventKind::ConstantFolded) >= 2);
    }

    #[test]
    fn run_with_no_simplifications_on_constant_only_block() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(block);

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        ssa.recompute_uses();
        let changed = run_mock_pass_boundary(&mut ssa, "constant-only algebraic no-op", |ssa| {
            run(ssa, &method, &log)
        });
        assert!(!changed);
        assert!(log.is_empty());
    }
}
