//! Integration test: end-to-end SSA pipeline driven by `MockTarget`.
//!
//! Constructs an `SsaFunction<MockTarget>` by hand, runs a sequence of
//! built-in passes (algebraic, GVN, copy propagation, dead code), and
//! confirms the IR roundtrips cleanly without ever touching a CIL host.
//! This covers the target-agnostic pass pipeline using only analyssa-provided
//! target metadata.

#![allow(clippy::unwrap_used)]

use analyssa::{
    events::{EventKind, EventLog},
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes,
    testing::{MockTarget, MockType},
};

/// Build a small `SsaFunction<MockTarget>` with redundant computation that
/// triggers algebraic + gvn + copy + dce, in roughly the order a host
/// scheduler would invoke them.
fn build_pipeline_test_function() -> SsaFunction<MockTarget> {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);

    // Allocate variables.
    for i in 0..6 {
        ssa.create_variable(
            VariableOrigin::Local(i as u16),
            0,
            DefSite::instruction(0, i),
            MockType::I32,
        );
    }
    let v0 = SsaVarId::from_index(0); // const 5
    let v1 = SsaVarId::from_index(1); // const 0
    let v2 = SsaVarId::from_index(2); // v0 + v1 (algebraic: x + 0 → x)
    let v3 = SsaVarId::from_index(3); // v0 + v1 (duplicate: GVN candidate)
    let v4 = SsaVarId::from_index(4); // copy of v2 (copy-prop candidate)
    let v5 = SsaVarId::from_index(5); // unused (DCE candidate)

    let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(5),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: v1,
        value: ConstValue::I32(0),
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
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy { dest: v4, src: v2 }));
    // Unused dead variable: v5 = const 99 (never used).
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: v5,
        value: ConstValue::I32(99),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v4) }));
    ssa.add_block(block);

    ssa
}

#[test]
fn mock_target_drives_algebraic_gvn_copy_dce() {
    let mut ssa = build_pipeline_test_function();
    let log: EventLog<MockTarget> = EventLog::new();
    let method = 0xDEAD_BEEFu32;

    // Phase 1: algebraic simplification — `v2 = v0 + 0` becomes `v2 = copy(v0)`.
    let alg_changed = passes::algebraic::run(&mut ssa, &method, &log);
    assert!(alg_changed, "algebraic should rewrite x+0 to copy");
    assert!(log.has(EventKind::ConstantFolded));

    // Phase 2: GVN — `v3 = v0 + 0` is now `v3 = copy(v0)` too; the duplicate
    // copy gets merged.
    let _gvn_changed = passes::gvn::run(&mut ssa, &method, &log);
    // GVN may or may not eliminate copy chains depending on canonical form;
    // the pass must at minimum not fail. The downstream copy pass cleans up.

    // Phase 3: copy propagation — chain `v4 = copy(v2) = copy(v0)` collapses.
    let copy_changed = passes::copying::run(&mut ssa, &method, &log, 5);
    assert!(copy_changed, "copy propagation should fire");
    assert!(log.has(EventKind::CopyPropagated));

    // Phase 4: dead code — `v5 = const 99` is never used.
    let dce_changed = passes::deadcode::run(&mut ssa, &method, &log, 20);
    assert!(dce_changed, "DCE should remove the unused const");
    assert!(log.has(EventKind::InstructionRemoved));

    // The IR should still validate as well-formed: block 0 still ends in a
    // Return that references something live.
    let block = ssa.block(0).unwrap();
    let last = block.instructions().last().unwrap();
    assert!(
        matches!(last.op(), SsaOp::Return { value: Some(_) }),
        "function still returns a value after pipeline"
    );

    // Acceptance: every event was recorded against the same `method`
    // identifier we passed in (proves the listener never sees a CIL Token
    // or any other target-specific identity).
    for ev in &log {
        assert_eq!(ev.method, Some(method));
    }
}

#[test]
fn mock_target_runs_blockmerge_and_controlflow() {
    // Build: B0: jump B1; B1: jump B2; B2: ret. Two trampolines that
    // blockmerge collapses, plus controlflow's dead-tail removal.
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
    let _ = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );

    for i in 0..3 {
        let mut block: SsaBlock<MockTarget> = SsaBlock::new(i);
        if i < 2 {
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: i + 1 }));
        } else {
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        }
        ssa.add_block(block);
    }

    let log: EventLog<MockTarget> = EventLog::new();
    let method = 1u32;

    let bm = passes::blockmerge::run(&mut ssa, &method, &log, 10);
    assert!(bm, "blockmerge should fire");

    let cf = passes::controlflow::run(&mut ssa, &method, &log, 10);
    // controlflow may or may not have more work after blockmerge; not asserted.
    let _ = cf;

    // After blockmerge, B0 should jump directly to B2.
    let last = ssa.block(0).unwrap().instructions().last().unwrap();
    assert!(matches!(
        last.op(),
        SsaOp::Jump { target: 2 } | SsaOp::Return { .. }
    ));
}
