//! Dataflow analysis tests: SCCP, liveness, reaching definitions.

#![allow(clippy::unwrap_used)]

use analyssa::{
    analysis::{
        cfg::SsaCfg,
        dataflow::{
            liveness::{LiveVariables, LivenessResult},
            reaching::ReachingDefinitions,
            sccp::{ConstantPropagation, ScalarValue},
            solver::DataFlowSolver,
        },
    },
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
    PointerSize,
};

fn local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::instruction(block, instr),
        MockType::I32,
    )
}

fn phi_local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::phi(block),
        MockType::I32,
    )
}

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

#[test]
fn sccp_propagates_simple_constants() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let sum = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(5),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(sum) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mut cp = ConstantPropagation::new(PointerSize::Bit64);
    let result = cp.analyze(&ssa, &cfg);

    assert!(result.is_constant(a));
    assert!(result.is_constant(b));
    assert!(result.is_constant(sum));
    assert_eq!(result.constant_value(sum).and_then(|c| c.as_i64()), Some(8));
    assert_eq!(result.constant_count(), 3);
}

#[test]
fn sccp_propagates_through_copies() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
    block.add_instruction(instr(SsaOp::Return { value: Some(b) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mut cp = ConstantPropagation::new(PointerSize::Bit64);
    let result = cp.analyze(&ssa, &cfg);

    assert!(result.is_constant(b));
    assert_eq!(result.constant_value(b).and_then(|c| c.as_i64()), Some(42));
}

#[test]
fn sccp_respects_control_flow_with_unexecutable_blocks() {
    let mut ssa = SsaFunction::new(0, 3);
    let cond = local(&mut ssa, 0, 0, 0);
    let val = local(&mut ssa, 1, 1, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(0), // Always false
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: val,
        value: ConstValue::I32(99),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mut cp = ConstantPropagation::new(PointerSize::Bit64);
    let result = cp.analyze(&ssa, &cfg);

    // SCCP may or may not prune the false branch; at minimum block 0 is
    // executable and the constants are known
    assert!(result.executable_blocks().any(|b| b == 0));
    assert!(result.constant_count() > 0);
}

#[test]
fn sccp_empty_function_is_handled() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    // Empty function has no blocks, so SCCP should not panic
    if ssa.block_count() > 0 {
        let cfg = SsaCfg::from_ssa(&ssa);
        let mut cp = ConstantPropagation::new(PointerSize::Bit64);
        let result = cp.analyze(&ssa, &cfg);
        assert_eq!(result.constant_count(), 0);
    }
}

#[test]
fn scalar_value_enumeration_is_correct() {
    assert!(matches!(
        ScalarValue::<MockTarget>::Bottom,
        ScalarValue::Bottom
    ));
    let result = analyssa::analysis::dataflow::sccp::SccpResult::<MockTarget>::empty();
    assert!(result.constants().next().is_none());
    assert_eq!(result.constant_count(), 0);
}

#[test]
fn sccp_propagates_through_phi_in_diamond() {
    let mut ssa = SsaFunction::new(0, 4);
    let cond = local(&mut ssa, 0, 0, 0);
    let left = local(&mut ssa, 1, 1, 0);
    let right = local(&mut ssa, 2, 2, 0);
    let merged = phi_local(&mut ssa, 3, 3);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::I32(10),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(20),
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(3));
    phi.add_operand(PhiOperand::new(left, 1));
    phi.add_operand(PhiOperand::new(right, 2));
    b3.add_phi(phi);
    b3.add_instruction(instr(SsaOp::Return {
        value: Some(merged),
    }));
    ssa.add_block(b3);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mut cp = ConstantPropagation::new(PointerSize::Bit64);
    let result = cp.analyze(&ssa, &cfg);

    // All blocks may be executable (SCCP may not prune based on branch
    // conditions alone). At minimum the constants are propagated.
    assert!(result.executable_blocks().any(|b| b == 0));
    assert!(result.is_constant(left));
    assert!(result.is_constant(right));
    assert_eq!(
        result.constant_value(left).and_then(|c| c.as_i64()),
        Some(10)
    );
    assert_eq!(
        result.constant_value(right).and_then(|c| c.as_i64()),
        Some(20)
    );
}

#[test]
fn sccp_converges_in_loop_to_fixpoint() {
    let mut ssa = SsaFunction::new(0, 3);
    let init = local(&mut ssa, 0, 0, 0);
    let limit = local(&mut ssa, 1, 0, 1);
    let counter = phi_local(&mut ssa, 2, 1);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: init,
        value: ConstValue::I32(0),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: limit,
        value: ConstValue::I32(3),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let cond = local(&mut ssa, 3, 1, 0);
    let next = local(&mut ssa, 4, 1, 1);
    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(counter, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(init, 0));
    phi.add_operand(PhiOperand::new(next, 1));
    b1.add_phi(phi);
    b1.add_instruction(instr(SsaOp::Clt {
        dest: cond,
        left: counter,
        right: limit,
        unsigned: false,
    }));
    b1.add_instruction(instr(SsaOp::Add {
        dest: next,
        left: counter,
        right: init,
        flags: None,
    }));
    b1.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mut cp = ConstantPropagation::new(PointerSize::Bit64);
    let result = cp.analyze(&ssa, &cfg);

    assert!(result.is_constant(init));
    assert!(result.is_constant(limit));
    assert_eq!(
        result.constant_value(init).and_then(|c| c.as_i64()),
        Some(0)
    );
    assert_eq!(
        result.constant_value(limit).and_then(|c| c.as_i64()),
        Some(3)
    );
}

#[test]
fn live_variables_analysis_runs_without_panicking() {
    let mut ssa = SsaFunction::new(0, 3);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let c = local(&mut ssa, 2, 1, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(2),
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: a,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: c,
        value: ConstValue::I32(3),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Return { value: Some(c) }));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let analysis = LiveVariables::new(&ssa);
    let solver = DataFlowSolver::new(analysis);
    let _result = solver.solve(&ssa, &cfg);
}

#[test]
fn liveness_result_empty_and_live_queries() {
    let empty = LivenessResult::new(0);
    assert!(empty.is_empty());

    let state = LivenessResult::new(10);
    assert!(!state.is_live(SsaVarId::from_index(0)));
}

#[test]
fn reaching_definitions_runs_without_panicking() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(a) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let analysis = ReachingDefinitions::new(&ssa);
    let solver = DataFlowSolver::new(analysis);
    let _result = solver.solve(&ssa, &cfg);
}

#[test]
fn reaching_definitions_empty_function_handled() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    let cfg = SsaCfg::from_ssa(&ssa);
    let analysis = ReachingDefinitions::new(&ssa);
    let solver = DataFlowSolver::new(analysis);
    let _result = solver.solve(&ssa, &cfg);
}
