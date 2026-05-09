//! Global Value Numbering (GVN) pass — eliminates redundant computations
//! across basic blocks.
//!
//! Detects when the same expression is computed multiple times with
//! identical operands and replaces later uses with the earlier result.
//!
//! # Algorithm
//!
//! 1. **Key construction**: For each pure operation (binary, unary,
//!    `LoadArg`), build a hashable value key from its opcode and
//!    operands. Commutative operations are normalized so that
//!    `a + b` and `b + a` produce the same key. Overflow-checked
//!    operations (`AddOvf`, `SubOvf`, `MulOvf`) and `Ckfinite` are
//!    excluded (they may throw).
//! 2. **Hash-consing**: If the same value key was seen before, the
//!    later definition is redundant — queue it for replacement.
//! 3. **Replacement**: For each redundant result, call
//!    [`SsaFunction::replace_uses_including_phis`] to forward all uses
//!    (including phi operands) to the original result. Then nop-out the
//!    redundant instruction so a subsequent DCE run removes it. Nopping
//!    (rather than leaving the instruction live) prevents ping-ponging
//!    with DCE on the next normalization iteration.
//!
//! # Scope
//!
//! GVN operates across all blocks in a single function (global within the
//! function). It does not perform interprocedural value numbering.
//!
//! # Complexity
//!
//! O(n) in the number of instructions — a single linear scan plus hash
//! map lookups.

use std::collections::HashMap;

use crate::{
    events::{EventKind, EventListener},
    ir::{
        function::SsaFunction,
        ops::{BinaryOpKind, SsaOp, UnaryOpKind},
        variable::SsaVarId,
    },
    target::Target,
};

/// Run Global Value Numbering on `ssa`.
///
/// Scans all pure instructions, builds hash-consed value keys, and
/// replaces redundant computations with references to the original result.
///
/// # Arguments
///
/// * `ssa` — The SSA function to optimize in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink receiving one [`EventKind::ConstantFolded`]
///   event per eliminated expression.
///
/// # Returns
///
/// `true` if any redundant computation was eliminated.
pub fn run<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    run_gvn(ssa, method, events) > 0
}

/// A hashable key representing an operation's value for value numbering.
///
/// Captures the semantics of an expression — the operation kind and
/// operands, not the destination. Two operations with the same key compute
/// the same value and are candidates for elimination.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ValueKey {
    /// A binary operation.
    ///
    /// Fields: `(kind, unsigned_flag, left_operand, right_operand)`.
    /// The unsigned flag is included for operations where signedness affects
    /// semantics (`Div`, `Rem`, `Shr`, `Clt`, `Cgt`). For other operations
    /// it is normalized to `false`. Commutative operations are normalized
    /// so that `(Add, false, a, b)` and `(Add, false, b, a)` produce the
    /// same hash.
    Binary(BinaryOpKind, bool, SsaVarId, SsaVarId),
    /// A unary operation.
    ///
    /// Fields: `(kind, operand)`. Not normalized for commutativity (unary
    /// operations are non-commutative by definition).
    Unary(UnaryOpKind, SsaVarId),
    /// Load of a method argument.
    ///
    /// Loading the same argument always produces the same value within a
    /// single function invocation. The field is the zero-based argument
    /// index.
    LoadArg(usize),
}

impl ValueKey {
    /// Builds a normalized value key from an SSA operation. Returns `None`
    /// for operations that should not be value-numbered (impure operations,
    /// constants, control flow, etc.).
    fn from_op<T: Target>(op: &SsaOp<T>) -> Option<(Self, SsaVarId)> {
        if let Some(info) = op.as_binary_op() {
            // Skip overflow-checked operations (they may throw).
            if matches!(
                info.kind,
                BinaryOpKind::AddOvf | BinaryOpKind::SubOvf | BinaryOpKind::MulOvf
            ) {
                return None;
            }
            let normalized = info.normalized();
            let (kind, unsigned, left, right) = normalized.value_key();
            return Some((Self::Binary(kind, unsigned, left, right), normalized.dest));
        }

        if let Some(info) = op.as_unary_op() {
            // Skip Ckfinite (it may throw).
            if info.kind == UnaryOpKind::Ckfinite {
                return None;
            }
            return Some((Self::Unary(info.kind, info.operand), info.dest));
        }

        if let SsaOp::LoadArg { dest, arg_index } = op {
            return Some((Self::LoadArg(*arg_index as usize), *dest));
        }

        None
    }
}

/// Internal GVN driver. Returns the number of uses replaced (used by tests).
fn run_gvn<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut value_map: HashMap<ValueKey, SsaVarId> = HashMap::new();
    let mut redundant: Vec<(SsaVarId, SsaVarId, usize, usize)> = Vec::new();

    for block in ssa.blocks() {
        let block_idx = block.id();
        for (instr_idx, instr) in block.instructions().iter().enumerate() {
            if let Some((key, dest)) = ValueKey::from_op(instr.op()) {
                if let Some(&original) = value_map.get(&key) {
                    redundant.push((dest, original, block_idx, instr_idx));
                } else {
                    value_map.insert(key, dest);
                }
            }
        }
    }

    let mut total_replaced: usize = 0;
    for (redundant_var, original_var, block_idx, instr_idx) in &redundant {
        let result = ssa.replace_uses_including_phis(*redundant_var, *original_var);
        if result.replaced > 0 {
            let event = crate::events::Event {
                kind: EventKind::ConstantFolded,
                method: Some(method.clone()),
                location: None,
                message: format!(
                    "GVN: {redundant_var} → {original_var} ({} uses)",
                    result.replaced
                ),
                pass: None,
            };
            events.push(event);
            total_replaced = total_replaced.saturating_add(result.replaced);
        }
        ssa.remove_instruction(*block_idx, *instr_idx);
    }

    total_replaced
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
            variable::{DefSite, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    #[test]
    fn value_key_binary_commutative() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);

        let add_op1: SsaOp<MockTarget> = SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        };
        let add_op2: SsaOp<MockTarget> = SsaOp::Add {
            dest: v3,
            left: v1,
            right: v0,
            flags: None,
        };
        let (k1, _) = ValueKey::from_op(&add_op1).unwrap();
        let (k2, _) = ValueKey::from_op(&add_op2).unwrap();
        assert_eq!(k1, k2, "Add should be commutative");

        let sub_op1: SsaOp<MockTarget> = SsaOp::Sub {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        };
        let sub_op2: SsaOp<MockTarget> = SsaOp::Sub {
            dest: v3,
            left: v1,
            right: v0,
            flags: None,
        };
        let (k3, _) = ValueKey::from_op(&sub_op1).unwrap();
        let (k4, _) = ValueKey::from_op(&sub_op2).unwrap();
        assert_ne!(k3, k4, "Sub should NOT be commutative");
    }

    #[test]
    fn value_key_recognizes_unary_and_skips_const() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);

        let add_op: SsaOp<MockTarget> = SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        };
        let (key, dest) = ValueKey::from_op(&add_op).unwrap();
        assert_eq!(dest, v2);
        assert!(matches!(key, ValueKey::Binary(BinaryOpKind::Add, _, _, _)));

        let neg_op: SsaOp<MockTarget> = SsaOp::Neg {
            dest: v1,
            operand: v0,
            flags: None,
        };
        let (key, dest) = ValueKey::from_op(&neg_op).unwrap();
        assert_eq!(dest, v1);
        assert!(matches!(key, ValueKey::Unary(UnaryOpKind::Neg, _)));

        let const_op: SsaOp<MockTarget> = SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        };
        assert!(ValueKey::from_op(&const_op).is_none());
    }

    /// End-to-end smoke test: identical Add expressions should collapse.
    /// Builds the IR by hand because `SsaFunctionBuilder` is CIL-pinned.
    #[test]
    fn eliminates_identical_binop() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        // 5 vars: v0, v1 (consts), v2 (add), v3 (redundant add), v4 (mul)
        for i in 0..5 {
            ssa.create_variable(
                VariableOrigin::Local(i),
                0,
                DefSite::instruction(0, i as usize),
                MockType::I32,
            );
        }
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);
        let v4 = SsaVarId::from_index(4);

        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(20),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v3,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Mul {
            dest: v4,
            left: v2,
            right: v3,
            flags: None,
        }));
        ssa.add_block(block);

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0xABu32;
        let replaced = run_gvn(&mut ssa, &method, &log);
        assert!(replaced > 0);
        assert!(!log.is_empty());

        // mul should now reference v2 twice instead of (v2, v3).
        let mul_instr = ssa.block(0).unwrap().instructions().last().unwrap();
        match mul_instr.op() {
            SsaOp::Mul { left, right, .. } => {
                assert_eq!(*left, v2);
                assert_eq!(*right, v2);
            }
            other => panic!("expected Mul, got {:?}", other),
        }
    }

    #[test]
    fn same_expression_across_blocks_eliminated() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        for i in 0..5 {
            ssa.create_variable(
                VariableOrigin::Local(i as u16),
                0,
                DefSite::instruction(0, i as usize),
                MockType::I32,
            );
        }
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);
        let v4 = SsaVarId::from_index(4);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(2),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v3,
            left: v0,
            right: v1,
            flags: None,
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v4,
            left: v2,
            right: v3,
            flags: None,
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v4) }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0xABu32;
        let changed = run(&mut ssa, &method, &log);
        assert!(
            changed,
            "duplicate expression across blocks should be eliminated"
        );
    }

    #[test]
    fn non_commutative_ops_not_confused() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        for i in 0..4 {
            ssa.create_variable(
                VariableOrigin::Local(i as u16),
                0,
                DefSite::instruction(0, i as usize),
                MockType::I32,
            );
        }
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(20),
        }));
        // a - b and b - a are different
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Sub {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Sub {
            dest: v3,
            left: v1,
            right: v0,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        // v2 and v3 are different computations — GVN should NOT eliminate either
        let _ = run(&mut ssa, &method, &log);
        // v3's Sub should still be present (not Nop'd)
        let third_instr = ssa.block(0).unwrap().instruction(3).unwrap();
        assert!(
            matches!(third_instr.op(), SsaOp::Sub { .. }),
            "non-commutative swapped sub should NOT be nop'd"
        );
    }

    #[test]
    fn gvn_skips_impure_operations() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        for i in 0..3 {
            ssa.create_variable(
                VariableOrigin::Local(i as u16),
                0,
                DefSite::instruction(0, i as usize),
                MockType::I32,
            );
        }
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(4),
        }));
        // Two AddOvf ops with same operands — should NOT be eliminated (may throw)
        block.add_instruction(SsaInstruction::synthetic(SsaOp::AddOvf {
            dest: v2,
            left: v0,
            right: v1,
            unsigned: false,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::AddOvf {
            dest: SsaVarId::from_index(3),
            left: v0,
            right: v1,
            unsigned: false,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let _ = run(&mut ssa, &method, &log);
        // Neither AddOvf should be nop'd
        assert!(matches!(
            ssa.block(0).unwrap().instruction(2).unwrap().op(),
            SsaOp::AddOvf { .. }
        ));
        assert!(matches!(
            ssa.block(0).unwrap().instruction(3).unwrap().op(),
            SsaOp::AddOvf { .. }
        ));
    }

    #[test]
    fn no_duplicates_returns_false() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        for i in 0..3 {
            ssa.create_variable(
                VariableOrigin::Local(i as u16),
                0,
                DefSite::instruction(0, i as usize),
                MockType::I32,
            );
        }
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log);
        assert!(!changed, "no duplicates should return false");
    }

    #[test]
    fn value_key_skips_const_and_return() {
        let v0 = SsaVarId::from_index(0);
        let const_op: SsaOp<MockTarget> = SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(0),
        };
        assert!(
            ValueKey::from_op(&const_op).is_none(),
            "Const should not generate a key"
        );

        let ret_op: SsaOp<MockTarget> = SsaOp::Return { value: None };
        assert!(
            ValueKey::from_op(&ret_op).is_none(),
            "Return should not generate a key"
        );
    }

    #[test]
    fn identical_loadargs_are_value_numbered() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(2, 2);
        let v0 = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let v1 = ssa.create_variable(
            VariableOrigin::Argument(1),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let v2 = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let _v3 = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: v0,
            arg_index: 0,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: v1,
            arg_index: 0,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log);
        assert!(changed, "identical LoadArgs should be value numbered");
    }

    #[test]
    fn gvn_with_mul_level_expression() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 6);
        for i in 0..7 {
            ssa.create_variable(
                VariableOrigin::Local(i as u16),
                0,
                DefSite::instruction(0, i as usize),
                MockType::I32,
            );
        }
        let a = SsaVarId::from_index(0);
        let b = SsaVarId::from_index(1);
        let c = SsaVarId::from_index(2);
        let expr1 = SsaVarId::from_index(3);
        let expr2 = SsaVarId::from_index(4);
        let result1 = SsaVarId::from_index(5);
        let result2 = SsaVarId::from_index(6);

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: b,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: c,
            value: ConstValue::I32(4),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: expr1,
            left: a,
            right: b,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Mul {
            dest: result1,
            left: expr1,
            right: c,
            flags: None,
        }));
        // Duplicate of (a+b)*c
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: expr2,
            left: a,
            right: b,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Mul {
            dest: result2,
            left: expr2,
            right: c,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(result2),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log);
        assert!(changed, "multi-level duplicates should be eliminated");
    }
}
