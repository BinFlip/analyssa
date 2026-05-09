//! Focused SSA verifier failure cases.

#![allow(clippy::unwrap_used)]

use analyssa::{
    analysis::{SsaVerifier, VerifierError, VerifyLevel},
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    testing::{MockTarget, MockType},
};

fn local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::instruction(block, instr),
        MockType::I32,
    )
}

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

#[test]
fn verifier_rejects_duplicate_instruction_definitions() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = local(&mut ssa, 0, 0, 0);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(errors
        .iter()
        .any(|err| matches!(err, VerifierError::DuplicateDefinition { var, .. } if *var == v0)));
}

#[test]
fn verifier_rejects_missing_phi_operand_for_predecessor() {
    let mut ssa = SsaFunction::new(0, 3);
    let from_left = local(&mut ssa, 0, 1, 0);
    let from_right = local(&mut ssa, 1, 2, 0);
    let merged = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(3), MockType::I32);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Branch {
        condition: from_left,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    let mut left = SsaBlock::new(1);
    left.add_instruction(instr(SsaOp::Const {
        dest: from_left,
        value: ConstValue::I32(1),
    }));
    left.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(left);

    let mut right = SsaBlock::new(2);
    right.add_instruction(instr(SsaOp::Const {
        dest: from_right,
        value: ConstValue::I32(2),
    }));
    right.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(right);

    let mut join = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(from_left, 1));
    join.add_phi(phi);
    join.add_instruction(instr(SsaOp::Return {
        value: Some(merged),
    }));
    ssa.add_block(join);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::MissingPhiOperand {
            block: 3,
            phi_idx: 0,
            missing_pred: 2,
        }
    )));
}

#[test]
fn verifier_rejects_terminator_before_block_end() {
    let mut ssa = SsaFunction::new(0, 2);
    let v0 = local(&mut ssa, 0, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Jump { target: 1 }));
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    ssa.add_block(block);

    let mut exit = SsaBlock::new(1);
    exit.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
    ssa.add_block(exit);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::TerminatorNotLast {
            block: 0,
            instr_idx: 0,
            instr_count: 2,
        }
    )));
}

#[test]
fn verifier_rejects_self_referential_instruction() {
    let mut ssa = SsaFunction::new(0, 2);
    let v0 = local(&mut ssa, 0, 0, 0);
    let v1 = local(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: v1,
        left: v0,
        right: v1,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(v1) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::SelfReferentialInstruction { var, .. } if *var == v1
    )));
}

#[test]
fn verifier_rejects_extra_phi_operand_for_non_predecessor() {
    let mut ssa = SsaFunction::new(0, 3);
    let left_value = local(&mut ssa, 0, 1, 0);
    let stale_value = local(&mut ssa, 1, 2, 0);
    let merged = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(3), MockType::I32);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(entry);

    let mut left = SsaBlock::new(1);
    left.add_instruction(instr(SsaOp::Const {
        dest: left_value,
        value: ConstValue::I32(1),
    }));
    left.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(left);

    let mut disconnected = SsaBlock::new(2);
    disconnected.add_instruction(instr(SsaOp::Const {
        dest: stale_value,
        value: ConstValue::I32(2),
    }));
    disconnected.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(disconnected);

    let mut join = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(left_value, 1));
    phi.add_operand(PhiOperand::new(stale_value, 2));
    join.add_phi(phi);
    join.add_instruction(instr(SsaOp::Return {
        value: Some(merged),
    }));
    ssa.add_block(join);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::ExtraPhiOperand {
            block: 3,
            phi_idx: 0,
            extra_pred: 2,
        }
    )));
}

#[test]
fn verifier_rejects_dominance_violation_across_branches() {
    let mut ssa = SsaFunction::new(0, 2);
    let condition = local(&mut ssa, 0, 0, 0);
    let branch_only = local(&mut ssa, 1, 1, 0);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: condition,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    let mut left = SsaBlock::new(1);
    left.add_instruction(instr(SsaOp::Const {
        dest: branch_only,
        value: ConstValue::I32(42),
    }));
    left.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(left);

    let mut right = SsaBlock::new(2);
    right.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(right);

    let mut join = SsaBlock::new(3);
    join.add_instruction(instr(SsaOp::Return {
        value: Some(branch_only),
    }));
    ssa.add_block(join);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::DominanceViolation {
            var,
            def_block: 1,
            use_block: 3,
        } if *var == branch_only
    )));
}

#[test]
fn verifier_rejects_undefined_use_in_instruction() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = SsaVarId::from_index(0);
    let v1 = SsaVarId::from_index(1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: v1,
        left: v0,
        right: SsaVarId::from_index(99),
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(v1) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors
        .iter()
        .any(|err| matches!(err, VerifierError::UndefinedUse { .. })));
}

#[test]
fn verifier_rejects_orphan_variable() {
    let mut ssa = SsaFunction::new(0, 1);
    let orphan = local(&mut ssa, 0, 0, 0);

    let mut block = SsaBlock::new(0);
    // Don't use `orphan` — it stays in the variables vec but is never defined in a block
    block.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::OrphanVariable { var } if *var == orphan
    )));
}

#[test]
fn verifier_rejects_unregistered_variable() {
    let mut ssa = SsaFunction::new(0, 0);
    let ghost = SsaVarId::from_index(42);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: ghost,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(ghost) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::UnregisteredVariable { var } if *var == ghost
    )));
}

#[test]
fn verifier_rejects_phi_in_entry_block() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = local(&mut ssa, 0, 0, 0);
    let phi_var = ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(0), MockType::I32);

    let mut entry = SsaBlock::new(0);
    let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(1));
    phi.add_operand(PhiOperand::new(v0, 0));
    entry.add_phi(phi);
    entry.add_instruction(instr(SsaOp::Return {
        value: Some(phi_var),
    }));
    ssa.add_block(entry);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors
        .iter()
        .any(|err| matches!(err, VerifierError::PhiInEntryBlock { block: 0, .. })));
}

#[test]
fn verifier_rejects_intra_block_cycle() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = local(&mut ssa, 0, 0, 0);
    let v1 = local(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Add {
        dest: v0,
        left: SsaVarId::from_index(2),
        right: SsaVarId::from_index(2),
        flags: None,
    }));
    let v2_instr = SsaInstruction::synthetic(SsaOp::Const {
        dest: SsaVarId::from_index(2),
        value: ConstValue::I32(1),
    });
    block.add_instruction(v2_instr);
    block.add_instruction(instr(SsaOp::Return { value: Some(v1) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(errors
        .iter()
        .any(|err| matches!(err, VerifierError::IntraBlockCycle { .. })));
}

#[test]
fn verifier_rejects_placeholder_variable_in_instruction() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = local(&mut ssa, 0, 0, 0);
    let sentinel = SsaVarId::from_index(usize::MAX);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: sentinel,
        left: v0,
        right: sentinel,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(errors
        .iter()
        .any(|err| matches!(err, VerifierError::PlaceholderVariable { .. })));
}

#[test]
fn verifier_all_levels_accept_well_formed_minimal_function() {
    let mut ssa = SsaFunction::new(0, 0);
    let v0 = local(&mut ssa, 0, 0, 0);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Quick).is_empty());
    assert!(SsaVerifier::new(&ssa)
        .verify(VerifyLevel::Standard)
        .is_empty());
    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
}

#[test]
fn verifier_all_levels_accept_diamond_with_phis() {
    let mut ssa = SsaFunction::new(0, 4);
    let cond = local(&mut ssa, 0, 0, 0);
    let left_val = local(&mut ssa, 1, 1, 0);
    let right_val = local(&mut ssa, 2, 2, 0);
    let phi_var = ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(3), MockType::I32);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    let mut left = SsaBlock::new(1);
    left.add_instruction(instr(SsaOp::Const {
        dest: left_val,
        value: ConstValue::I32(10),
    }));
    left.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(left);

    let mut right = SsaBlock::new(2);
    right.add_instruction(instr(SsaOp::Const {
        dest: right_val,
        value: ConstValue::I32(20),
    }));
    right.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(right);

    let mut join = SsaBlock::new(3);
    let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(3));
    phi.add_operand(PhiOperand::new(left_val, 1));
    phi.add_operand(PhiOperand::new(right_val, 2));
    join.add_phi(phi);
    join.add_instruction(instr(SsaOp::Return {
        value: Some(phi_var),
    }));
    ssa.add_block(join);
    ssa.recompute_uses();

    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Quick).is_empty());
    assert!(SsaVerifier::new(&ssa)
        .verify(VerifyLevel::Standard)
        .is_empty());
    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
}

#[test]
fn verifier_reports_multiple_error_kinds_at_once() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = local(&mut ssa, 0, 0, 0);
    let v1 = local(&mut ssa, 1, 0, 1);
    let undefined = SsaVarId::from_index(99);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: v1,
        left: v0,
        right: v1,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Copy {
        dest: v1,
        src: undefined,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors.len() >= 3,
        "expected at least 3 errors, got {}",
        errors.len()
    );
    let has_duplicate = errors
        .iter()
        .any(|e| matches!(e, VerifierError::DuplicateDefinition { .. }));
    let has_self_ref = errors
        .iter()
        .any(|e| matches!(e, VerifierError::SelfReferentialInstruction { .. }));
    let has_undef = errors
        .iter()
        .any(|e| matches!(e, VerifierError::UndefinedUse { .. }));
    assert!(has_duplicate, "missing DuplicateDefinition");
    assert!(has_self_ref, "missing SelfReferentialInstruction");
    assert!(has_undef, "missing UndefinedUse");
}
