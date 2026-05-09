//! Control-flow and block utility SSA patterns.

#![allow(clippy::unwrap_used)]

use analyssa::{
    analysis::{SsaCfg, SsaVerifier, VerifyLevel},
    ir::{
        block::SsaBlock,
        exception::{native_is_filter_handler, NativeExceptionKind, SsaExceptionHandler},
        function::{FunctionKind, SsaFunction},
        instruction::SsaInstruction,
        ops::{CmpKind, SsaOp},
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
fn switch_and_branchcmp_successors_are_indexed_by_cfg() {
    let mut ssa = SsaFunction::new(0, 3);
    let selector = local(&mut ssa, 0, 0, 0);
    let left = local(&mut ssa, 1, 1, 0);
    let right = local(&mut ssa, 2, 1, 1);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: selector,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Switch {
        value: selector,
        targets: vec![1, 2, 3],
        default: 4,
    }));
    ssa.add_block(entry);

    let mut compare = SsaBlock::new(1);
    compare.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::I32(7),
    }));
    compare.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(9),
    }));
    compare.add_instruction(instr(SsaOp::BranchCmp {
        left,
        right,
        cmp: CmpKind::Lt,
        unsigned: false,
        true_target: 5,
        false_target: 6,
    }));
    ssa.add_block(compare);

    for block_idx in 2..=6 {
        let mut block = SsaBlock::new(block_idx);
        block.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(block);
    }
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    assert_eq!(cfg.block_successors(0), &[1, 2, 3, 4]);
    assert_eq!(cfg.block_successors(1), &[5, 6]);
    assert_eq!(cfg.block_predecessors(5), &[1]);
    assert_eq!(cfg.exits().len(), 5);

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.is_empty(), "verifier errors: {errors:?}");
}

#[test]
fn exception_handler_mapping_adds_synthetic_cfg_edge() {
    let mut ssa = SsaFunction::new(0, 1);
    let value = local(&mut ssa, 0, 0, 0);

    let mut try_entry = SsaBlock::new(0);
    try_entry.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(1),
    }));
    try_entry.add_instruction(instr(SsaOp::Return { value: Some(value) }));
    ssa.add_block(try_entry);

    let mut handler = SsaBlock::new(1);
    handler.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(handler);

    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 1,
        handler_offset: 1,
        handler_length: 1,
        class_token_or_filter: 0,
        try_start_block: Some(0),
        try_end_block: Some(1),
        handler_start_block: Some(1),
        handler_end_block: Some(2),
        filter_start_block: None,
    }]);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    assert_eq!(cfg.block_successors(0), &[1]);
    assert_eq!(cfg.block_predecessors(1), &[0]);
}

#[test]
fn block_utilities_detect_trampolines_and_reorder_local_dependencies() {
    let mut trampoline = SsaBlock::<MockTarget>::new(0);
    trampoline.add_instruction(instr(SsaOp::Jump { target: 9 }));
    assert_eq!(trampoline.is_trampoline(), Some(9));

    let v0 = SsaVarId::from_index(0);
    let v1 = SsaVarId::from_index(1);
    let v2 = SsaVarId::from_index(2);
    let mut block = SsaBlock::<MockTarget>::new(1);
    block.add_instruction(instr(SsaOp::Add {
        dest: v2,
        left: v1,
        right: v0,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: v1,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(v2) }));

    assert!(block.sort_instructions_topologically());
    assert!(matches!(block.instruction(0).unwrap().op(), SsaOp::Const { dest, .. } if *dest == v0));
    assert!(matches!(block.instruction(1).unwrap().op(), SsaOp::Const { dest, .. } if *dest == v1));
    assert!(matches!(block.instruction(2).unwrap().op(), SsaOp::Add { dest, .. } if *dest == v2));
    assert!(block.instruction(3).unwrap().is_terminator());
}

// ---------------------------------------------------------------------------
// Interrupt / ISR function integration tests
// ---------------------------------------------------------------------------

#[test]
fn interrupt_handler_function_kind_defaults_to_normal() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    assert_eq!(ssa.kind(), FunctionKind::Normal);
    assert!(ssa.kind().is_normal());
    assert!(!ssa.kind().is_interrupt_handler());
}

#[test]
fn interrupt_handler_function_kind_can_be_set() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    ssa.set_kind(FunctionKind::InterruptHandler);
    assert_eq!(ssa.kind(), FunctionKind::InterruptHandler);
    assert!(ssa.kind().is_interrupt_handler());
    assert!(!ssa.kind().is_normal());
}

#[test]
fn interrupt_handler_with_interrupt_return_terminator() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    ssa.set_kind(FunctionKind::InterruptHandler);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::InterruptReturn));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert_eq!(ssa.kind(), FunctionKind::InterruptHandler);
    assert!(ssa.has_interrupt_return());
    assert!(ssa
        .block(0)
        .unwrap()
        .terminator_op()
        .is_some_and(|op| { matches!(op, SsaOp::InterruptReturn) }));
}

#[test]
fn interrupt_return_is_terminal_no_successors() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::InterruptReturn));
    ssa.add_block(block);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    assert!(cfg.block_successors(0).is_empty());
}

#[test]
fn has_interrupt_return_returns_false_when_no_interrupt_return() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    assert!(!ssa.has_interrupt_return());
}

#[test]
fn has_interrupt_return_returns_true_when_interrupt_return_present() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::InterruptReturn));
    ssa.add_block(block);
    ssa.recompute_uses();
    assert!(ssa.has_interrupt_return());
}

#[test]
fn interrupt_return_survives_canonicalization() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    ssa.set_kind(FunctionKind::InterruptHandler);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::InterruptReturn));
    ssa.add_block(block);
    ssa.recompute_uses();

    ssa.canonicalize();
    assert_eq!(ssa.block_count(), 1);
    assert!(ssa.has_interrupt_return());
    assert_eq!(ssa.kind(), FunctionKind::InterruptHandler);
}

#[test]
fn normal_function_without_interrupt_return_remains_normal_after_canonicalize() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(block);
    ssa.recompute_uses();
    ssa.canonicalize();

    assert_eq!(ssa.kind(), FunctionKind::Normal);
    assert!(!ssa.has_interrupt_return());
}

// ---------------------------------------------------------------------------
// SEH / NativeExceptionKind integration tests
// ---------------------------------------------------------------------------

#[test]
fn native_exception_kind_is_filter_handler_correctly_identifies_filter() {
    assert!(native_is_filter_handler(&NativeExceptionKind::Filter));
    assert!(!native_is_filter_handler(&NativeExceptionKind::Catch));
    assert!(!native_is_filter_handler(&NativeExceptionKind::Finally));
    assert!(!native_is_filter_handler(&NativeExceptionKind::Fault));
}

#[test]
fn native_exception_handler_with_catch_flags_uses_mock_target() {
    // MockTarget uses u32 as ExceptionKind, so we test the concept
    // using u32 flags matching the Catch/Filter/Finally/Fault convention:
    // 0=Catch, 1=Filter, 2=Finally, 3=Fault
    let handler = SsaExceptionHandler::<MockTarget> {
        flags: 0, // Catch
        try_offset: 0,
        try_length: 10,
        handler_offset: 10,
        handler_length: 20,
        class_token_or_filter: 42,
        try_start_block: Some(0),
        try_end_block: Some(1),
        handler_start_block: Some(1),
        handler_end_block: Some(2),
        filter_start_block: None,
    };

    // MockTarget::is_filter_handler always returns false
    assert!(handler.filter_offset().is_none());
    assert_eq!(handler.class_token_or_filter, 42);
}

#[test]
fn native_exception_handler_block_mapping_works_with_u32_flags() {
    let kinds = [0u32, 1, 2, 3]; // Catch, Filter, Finally, Fault

    for flags in &kinds {
        let mut handler = SsaExceptionHandler::<MockTarget> {
            flags: *flags,
            try_offset: 0,
            try_length: 10,
            handler_offset: 10,
            handler_length: 20,
            class_token_or_filter: 0,
            try_start_block: Some(0),
            try_end_block: Some(2),
            handler_start_block: Some(3),
            handler_end_block: Some(5),
            filter_start_block: Some(4),
        };

        assert!(handler.has_block_mapping());
        handler.remap_block_indices(&[Some(0), None, None, Some(1), Some(2), Some(3)]);
        assert_eq!(handler.try_start_block, Some(0));
        assert!(handler.handler_start_block.is_some());
    }
}

#[test]
fn isr_debug_shows_function_kind() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    ssa.set_kind(FunctionKind::InterruptHandler);
    let debug = format!("{ssa:?}");
    assert!(debug.contains("kind: InterruptHandler"));

    let normal = SsaFunction::<MockTarget>::new(0, 0);
    let normal_debug = format!("{normal:?}");
    assert!(normal_debug.contains("kind: Normal"));
}
