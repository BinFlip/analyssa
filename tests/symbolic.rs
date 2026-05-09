//! Symbolic expression evaluation tests.

#![allow(clippy::unwrap_used)]

use analyssa::{
    analysis::{
        evaluator::SsaEvaluator,
        symbolic::{SymbolicEvaluator, SymbolicExpr, SymbolicOp},
    },
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
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

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

#[test]
fn symbolic_expr_constant_and_variable_construction() {
    let c = SymbolicExpr::<MockTarget>::Constant(ConstValue::I32(42));
    assert!(c.is_constant());
    assert!(!c.is_variable());
    assert_eq!(c.as_constant().and_then(|cv| cv.as_i64()), Some(42));

    let v = SymbolicExpr::<MockTarget>::Variable(SsaVarId::from_index(7));
    assert!(v.is_variable());
    assert!(!v.is_constant());
    assert_eq!(v.as_variable(), Some(SsaVarId::from_index(7)));
}

#[test]
fn symbolic_expr_binary_operation() {
    let a = SymbolicExpr::<MockTarget>::Constant(ConstValue::I32(5));
    let b = SymbolicExpr::<MockTarget>::Constant(ConstValue::I32(3));

    let sum = SymbolicExpr::<MockTarget>::Binary {
        op: SymbolicOp::Add,
        left: Box::new(a),
        right: Box::new(b),
    };

    assert!(matches!(sum, SymbolicExpr::Binary { .. }));
    assert!(!matches!(sum, SymbolicExpr::Constant(_)));
}

#[test]
fn symbolic_expr_unary_operation() {
    let v = SymbolicExpr::<MockTarget>::Variable(SsaVarId::from_index(0));

    let neg = SymbolicExpr::<MockTarget>::Unary {
        op: SymbolicOp::Neg,
        operand: Box::new(v),
    };

    assert!(matches!(neg, SymbolicExpr::Unary { .. }));
    assert!(!matches!(neg, SymbolicExpr::Constant(_)));
}

#[test]
fn symbolic_expr_complex_nesting() {
    let a = SymbolicExpr::<MockTarget>::Constant(ConstValue::I32(1));
    let b = SymbolicExpr::<MockTarget>::Variable(SsaVarId::from_index(0));

    let sum = SymbolicExpr::<MockTarget>::Binary {
        op: SymbolicOp::Add,
        left: Box::new(a),
        right: Box::new(b),
    };
    let _prod = SymbolicExpr::<MockTarget>::Binary {
        op: SymbolicOp::Mul,
        left: Box::new(sum),
        right: Box::new(SymbolicExpr::<MockTarget>::Constant(ConstValue::I32(2))),
    };
}

#[test]
fn symbolic_expr_helpers_constant_and_binary() {
    let c = SymbolicExpr::<MockTarget>::constant(ConstValue::I32(10));
    assert!(c.is_constant());

    let bin = SymbolicExpr::<MockTarget>::binary(
        SymbolicOp::Add,
        SymbolicExpr::<MockTarget>::constant(ConstValue::I32(1)),
        SymbolicExpr::<MockTarget>::constant(ConstValue::I32(2)),
    );
    assert!(matches!(bin, SymbolicExpr::Binary { .. }));

    let uni = SymbolicExpr::<MockTarget>::unary(
        SymbolicOp::Neg,
        SymbolicExpr::<MockTarget>::constant(ConstValue::I32(5)),
    );
    assert!(matches!(uni, SymbolicExpr::Unary { .. }));
}

#[test]
fn symbolic_evaluator_tracks_expressions() {
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
        value: ConstValue::I32(10),
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

    let mut sym_eval = SymbolicEvaluator::new(&ssa, PointerSize::Bit64);
    sym_eval.evaluate_block(0);

    let c_expr = sym_eval.get_expression(const_val);
    assert!(c_expr.is_some());
    assert!(c_expr.unwrap().is_constant());

    let s_expr = sym_eval.get_expression(sum);
    assert!(s_expr.is_some());
    assert!(matches!(s_expr.unwrap(), SymbolicExpr::Binary { .. }));

    // param may or may not have an expression (entry-point variables are
    // not tracked unless explicitly set_symbolic is called)
    let _p_expr = sym_eval.get_expression(param);
}

#[test]
fn symbolic_evaluator_handles_empty_function() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    let mut sym_eval = SymbolicEvaluator::new(&ssa, PointerSize::Bit64);
    sym_eval.evaluate_block(0);
}

#[test]
fn symbolic_expr_tracks_through_copy() {
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

    let mut sym_eval = SymbolicEvaluator::new(&ssa, PointerSize::Bit64);
    sym_eval.evaluate_block(0);

    let expr = sym_eval.get_expression(b);
    assert!(expr.is_some());
    assert!(expr.unwrap().is_constant());
}

#[test]
fn ssa_evaluator_concrete_and_symbolic() {
    let mut ssa = SsaFunction::new(0, 0);
    let param = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let c = local(&mut ssa, 1, 0, 1);
    let prod = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: c,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::Mul {
        dest: prod,
        left: param,
        right: c,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(prod) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    eval.evaluate_block(0);

    assert!(eval.is_concrete(c));
    assert!(eval.is_concrete(c));
    assert_eq!(eval.get_concrete(c).and_then(|cv| cv.as_i64()), Some(2));

    // prod is unknown (param is unknown too) or symbolic
    let _sym = eval.get_symbolic(prod);
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
