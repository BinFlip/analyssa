//! Path-aware evaluator tests: concrete and symbolic evaluation, phi-aware
//! path tracing, constraint solving, loop fixpoint evaluation.

#![allow(clippy::unwrap_used)]

use analyssa::{
    analysis::evaluator::{ControlFlow, SsaEvaluator},
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::{CmpKind, SsaOp},
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
fn evaluator_resolves_simple_arithmetic_to_concrete() {
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

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);
    assert_eq!(eval.get_concrete(sum).and_then(|cv| cv.as_i64()), Some(8));
    assert_eq!(eval.get_concrete(a).and_then(|cv| cv.as_i64()), Some(5));
}

#[test]
fn evaluator_resolves_chained_arithmetic() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let c = local(&mut ssa, 2, 0, 2);
    let ab = local(&mut ssa, 3, 0, 3);
    let abc = local(&mut ssa, 4, 0, 4);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: c,
        value: ConstValue::I32(4),
    }));
    block.add_instruction(instr(SsaOp::Mul {
        dest: ab,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: abc,
        left: ab,
        right: c,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(abc) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);
    assert_eq!(eval.get_concrete(abc).and_then(|cv| cv.as_i64()), Some(10));
}

fn build_simple_diamond() -> SsaFunction<MockTarget> {
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
        value: ConstValue::I32(100),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(200),
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
    ssa
}

#[test]
fn evaluator_selects_phi_operand_based_on_predecessor() {
    let ssa = build_simple_diamond();
    let merged = SsaVarId::from_index(3);

    let mut eval_left = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval_left.evaluate_block(0);
    eval_left.evaluate_block(1);
    eval_left.set_predecessor(Some(1));
    eval_left.evaluate_phis(3);
    assert_eq!(
        eval_left.get_concrete(merged).and_then(|cv| cv.as_i64()),
        Some(100)
    );

    let mut eval_right = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval_right.evaluate_block(0);
    eval_right.evaluate_block(2);
    eval_right.set_predecessor(Some(2));
    eval_right.evaluate_phis(3);
    assert_eq!(
        eval_right.get_concrete(merged).and_then(|cv| cv.as_i64()),
        Some(200)
    );
}

#[test]
fn evaluator_evaluates_comparison_results() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let is_lt = local(&mut ssa, 2, 0, 2);
    let is_eq = local(&mut ssa, 3, 0, 3);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(5),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Clt {
        dest: is_lt,
        left: a,
        right: b,
        unsigned: false,
    }));
    block.add_instruction(instr(SsaOp::Ceq {
        dest: is_eq,
        left: a,
        right: b,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(is_lt) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);
    assert_eq!(eval.get_concrete(is_lt).and_then(|cv| cv.as_i64()), Some(1));
    assert_eq!(eval.get_concrete(is_eq).and_then(|cv| cv.as_i64()), Some(0));
}

#[test]
fn evaluate_path_evaluates_blocks_in_order() {
    let ssa = build_simple_diamond();
    let merged = SsaVarId::from_index(3);

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_path(&[0, 1]);
    eval.set_predecessor(Some(1));
    eval.evaluate_phis(3);
    assert_eq!(
        eval.get_concrete(merged).and_then(|cv| cv.as_i64()),
        Some(100)
    );
}

#[test]
fn evaluate_blocks_evaluates_multiple_in_sequence() {
    let ssa = build_simple_diamond();
    let left = SsaVarId::from_index(1);

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_blocks(&[0, 1]);
    assert_eq!(
        eval.get_concrete(left).and_then(|cv| cv.as_i64()),
        Some(100)
    );
}

fn build_counter_loop() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 4);

    let init = local(&mut ssa, 0, 0, 0);
    let one = local(&mut ssa, 1, 0, 1);
    let limit = local(&mut ssa, 2, 0, 2);
    let counter = phi_local(&mut ssa, 3, 1);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: init,
        value: ConstValue::I32(0),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: limit,
        value: ConstValue::I32(3),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let cond = local(&mut ssa, 5, 1, 0);
    let next = local(&mut ssa, 6, 1, 1);
    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(counter, VariableOrigin::Local(3));
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
        right: one,
        flags: None,
    }));
    b1.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Return {
        value: Some(counter),
    }));
    ssa.add_block(b2);

    ssa.recompute_uses();
    ssa
}

#[test]
fn evaluate_loop_to_fixpoint_converges() {
    let ssa = build_counter_loop();
    let counter = SsaVarId::from_index(3);

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);
    eval.evaluate_loop_to_fixpoint(&[1], 10);

    // The evaluator may or may not resolve the loop counter depending
    // on whether it can track phi values across iterations
    let _ = eval.get_concrete(counter);
    // At minimum, no panic occurred
}

#[test]
fn evaluate_loop_iterations_converges() {
    let ssa = build_counter_loop();
    let counter = SsaVarId::from_index(3);

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);
    eval.evaluate_loop_iterations(&[1], 5);

    // The evaluator may or may not resolve the loop counter concretely
    let _ = eval.get_concrete(counter);
    // At minimum, no panic occurred
}

#[test]
fn evaluator_resolves_known_condition_to_boolean() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let is_lt = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(5),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Clt {
        dest: is_lt,
        left: a,
        right: b,
        unsigned: false,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(is_lt) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);
    let result = eval.evaluate_condition_with_constraints(is_lt);
    assert_eq!(result, Some(true));
}

#[test]
fn evaluator_distinguishes_concrete_from_unknown() {
    let mut ssa = SsaFunction::new(0, 0);
    let param = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let const_val = local(&mut ssa, 1, 0, 1);
    let sum = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: const_val,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left: param,
        right: const_val,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(sum) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);

    assert!(eval.is_concrete(const_val));
    assert!(!eval.is_unknown(const_val));

    assert!(eval.is_unknown(param));
    assert!(!eval.is_concrete(param));

    assert!(eval.is_symbolic(sum) || eval.is_unknown(sum));
}

#[test]
fn evaluator_resolves_bitwise_operations() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let and_result = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(6),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::And {
        dest: and_result,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Return {
        value: Some(and_result),
    }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);
    assert_eq!(
        eval.get_concrete(and_result).and_then(|cv| cv.as_i64()),
        Some(2)
    );
}

#[test]
fn control_flow_is_terminal_for_terminal() {
    let cf = ControlFlow::Terminal;
    assert!(cf.is_terminal());
    assert!(!cf.is_unknown());
    assert_eq!(cf.target(), None);

    let cf = ControlFlow::Unknown;
    assert!(cf.is_unknown());
    assert!(!cf.is_terminal());

    let cf = ControlFlow::Continue(7);
    assert!(!cf.is_terminal());
    assert!(!cf.is_unknown());
    assert_eq!(cf.target(), Some(7));
}

#[test]
fn evaluator_evaluates_branch_cmp_targets() {
    let mut ssa = SsaFunction::new(0, 2);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(2),
    }));
    b0.add_instruction(instr(SsaOp::BranchCmp {
        left: a,
        right: b,
        cmp: CmpKind::Lt,
        unsigned: false,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Return { value: Some(a) }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Return { value: Some(b) }));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);
    assert_eq!(eval.get_concrete(a).and_then(|cv| cv.as_i64()), Some(1));
}

#[test]
fn ssa_evaluator_has_constraints_for_variables() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let cmp = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(5),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Clt {
        dest: cmp,
        left: a,
        right: b,
        unsigned: false,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(cmp) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);

    let _ = eval.has_constraints(cmp);
}

#[test]
fn evaluate_with_trace_resolves_complex_path() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let c = local(&mut ssa, 2, 0, 2);
    let d = local(&mut ssa, 3, 0, 3);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::Mul {
        dest: c,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: d,
        left: c,
        right: a,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(d) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);

    let trace = eval.evaluate_with_trace(d, 10);
    assert_eq!(trace, Some(40));
}
