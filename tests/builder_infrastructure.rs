//! Builder-backed fixture infrastructure tests.

use analyssa::{
    analysis::{
        memory::MemorySsa, DefUseIndex, LoopAnalyzer, SsaCfg, SsaVerifier, VerifierError,
        VerifyLevel,
    },
    events::EventLog,
    graph::NodeId,
    ir::{
        function::{SsaDefSpec, SsaFunctionBuilder},
        ops::{SsaEffectKind, SsaOp},
        SsaVarId,
    },
    passes,
    testing::{
        diamond_phi_fixture, loop_counter_fixture, memory_effect_fixture, mock_i32, mock_mask4,
        mock_ptr, mock_ref, mock_v4i32, native_effect_fixture, scalar_rewrite_fixture,
        vector_simd_fixture, MockTarget, MockType,
    },
    PointerSize,
};

fn assert_valid(ssa: &analyssa::ir::SsaFunction<MockTarget>, level: VerifyLevel) {
    let errors = SsaVerifier::new(ssa).verify(level);
    assert!(errors.is_empty(), "verifier errors: {errors:?}");
}

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| std::process::abort())
}

fn result_or_abort<T>(result: analyssa::Result<T>) -> T {
    result.unwrap_or_else(|_| std::process::abort())
}

#[test]
fn mock_def_specs_cover_common_fixture_types() {
    assert_eq!(mock_i32().var_type, MockType::I32);
    assert_eq!(mock_ptr().var_type, MockType::Ptr);
    assert_eq!(mock_ref().var_type, MockType::Ref);
    assert_eq!(mock_v4i32().var_type, MockType::V4I32);
    assert_eq!(mock_mask4().var_type, MockType::Mask4);
}

#[test]
fn all_builder_fixtures_verify_at_full_level() {
    let fixtures = [
        scalar_rewrite_fixture(),
        diamond_phi_fixture(),
        loop_counter_fixture(),
        memory_effect_fixture(),
        native_effect_fixture(),
        vector_simd_fixture(),
    ];

    for fixture in fixtures {
        assert_valid(&fixture, VerifyLevel::Full);
    }
}

#[test]
fn diamond_fixture_has_precise_cfg_and_phi_uses() {
    let ssa = diamond_phi_fixture();
    assert_valid(&ssa, VerifyLevel::Full);

    let cfg = SsaCfg::from_ssa(&ssa);
    assert_eq!(cfg.block_successors(0), &[1, 2]);
    assert_eq!(cfg.block_predecessors(3), &[1, 2]);

    let merge = some_or_abort(ssa.block(3));
    let phi = some_or_abort(merge.phi_nodes().first());
    assert_eq!(phi.operand_count(), 2);
    for operand in phi.operands() {
        let uses = some_or_abort(ssa.variable(operand.value())).uses();
        assert!(uses.iter().any(|use_site| use_site.is_phi_operand));
    }
}

#[test]
fn loop_fixture_exposes_natural_loop_analysis() {
    let ssa = loop_counter_fixture();
    assert_valid(&ssa, VerifyLevel::Full);

    let forest = LoopAnalyzer::new(&ssa).analyze();
    assert_eq!(forest.len(), 1);
    assert!(forest.loop_for_header(NodeId::new(1)).is_some());
    assert!(forest.is_in_loop(NodeId::new(1)));
    assert!(!forest.is_in_loop(NodeId::new(2)));
}

#[test]
fn memory_fixture_feeds_memory_ssa_and_effect_queries() {
    let ssa = memory_effect_fixture();
    assert_valid(&ssa, VerifyLevel::Full);

    let cfg = SsaCfg::from_ssa(&ssa);
    let memory = MemorySsa::build(&ssa, &cfg);
    let stats = memory.stats();
    assert!(
        stats.store_count >= 2,
        "expected ordinary and atomic stores"
    );
    assert!(stats.load_count >= 1, "expected field load");
    assert!(stats.location_count >= 1);

    let mut saw_atomic = false;
    let mut saw_fence = false;
    for (_, _, instr) in ssa.iter_instructions() {
        let effects = instr.op().effects();
        saw_atomic |= matches!(effects.kind, SsaEffectKind::Atomic);
        saw_fence |= matches!(effects.kind, SsaEffectKind::Fence);
    }
    assert!(saw_atomic);
    assert!(saw_fence);
}

#[test]
fn native_fixture_indexes_secondary_defs_and_preserves_opaque_effects() {
    let ssa = native_effect_fixture();
    assert_valid(&ssa, VerifyLevel::Full);

    let index = DefUseIndex::build_with_ops(&ssa);
    let mut secondary_defs = 0usize;
    let mut native_outputs = 0usize;
    let mut saw_opaque = false;

    for (_, _, instr) in ssa.iter_instructions() {
        match instr.op() {
            SsaOp::Add {
                flags: Some(flags), ..
            } => {
                secondary_defs += 1;
                assert!(index.def_op(*flags).is_some());
            }
            SsaOp::WideMul { high, .. } => {
                secondary_defs += 1;
                assert!(index.def_op(*high).is_some());
            }
            SsaOp::NativeOpaque(data) => {
                saw_opaque = true;
                native_outputs = data.outputs.len();
                assert!(matches!(data.effects.kind, SsaEffectKind::Opaque));
                for output in &data.outputs {
                    assert!(index.def_op(*output).is_some());
                }
            }
            _ => {}
        }
    }

    assert!(secondary_defs >= 2);
    assert_eq!(native_outputs, 2);
    assert!(saw_opaque);
}

#[test]
fn vector_fixture_exercises_modern_simd_shapes() {
    let ssa = vector_simd_fixture();
    assert_valid(&ssa, VerifyLevel::Full);

    let mut saw_compare = false;
    let mut saw_faulting_load = false;
    let mut saw_segment_load = false;
    let mut saw_segment_store = false;

    for (_, _, instr) in ssa.iter_instructions() {
        match instr.op() {
            SsaOp::VectorCompare { dest, .. } => {
                saw_compare = true;
                assert_eq!(
                    some_or_abort(ssa.variable(*dest)).var_type(),
                    &MockType::Mask4
                );
            }
            SsaOp::VectorFaultingLoad {
                fault: Some(fault), ..
            } => {
                saw_faulting_load = true;
                assert_eq!(
                    some_or_abort(ssa.variable(*fault)).var_type(),
                    &MockType::Mask4
                );
            }
            SsaOp::VectorSegmentLoad { dests, .. } => {
                saw_segment_load = true;
                assert_eq!(dests.len(), 2);
            }
            SsaOp::VectorSegmentStore { values, .. } => {
                saw_segment_store = true;
                assert_eq!(values.len(), 2);
            }
            _ => {}
        }
    }

    assert!(saw_compare);
    assert!(saw_faulting_load);
    assert!(saw_segment_load);
    assert!(saw_segment_store);
}

#[test]
fn scalar_fixture_survives_rewrite_pass_stack() {
    let mut ssa = scalar_rewrite_fixture();
    assert_valid(&ssa, VerifyLevel::Full);

    let method = 0u32;
    let events = EventLog::<MockTarget>::new();
    let mut changed = false;
    changed |= passes::algebraic::run(&mut ssa, &method, &events);
    changed |= passes::gvn::run(&mut ssa, &method, &events);
    changed |= passes::copying::run(&mut ssa, &method, &events, 8);
    changed |= passes::strength::run(&mut ssa, &method, &events, &|_| false);
    changed |= passes::reassociate::run(&mut ssa, &method, &events, PointerSize::Bit64);
    changed |= passes::ranges::run(&mut ssa, &method, &events, 12);
    changed |= passes::deadcode::run(&mut ssa, &method, &events, 20);

    assert!(changed, "fixture should expose at least one rewrite");
    assert_valid(&ssa, VerifyLevel::Full);
}

#[test]
fn cfg_fixtures_feed_threading_and_pass_verification() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 1);
    builder.ensure_block(3);
    let condition = result_or_abort(builder.in_block(0, |block| {
        let condition = block.const_bool(SsaDefSpec::tmp(MockType::I32), true)?;
        block.jump(1)?;
        Ok(condition)
    }));
    result_or_abort(builder.in_block(1, |block| {
        block.branch(condition, 2, 3)?;
        Ok(())
    }));
    result_or_abort(builder.in_block(2, |block| {
        block.ret(None)?;
        Ok(())
    }));
    result_or_abort(builder.in_block(3, |block| {
        block.ret(None)?;
        Ok(())
    }));
    let mut ssa = result_or_abort(builder.finish());
    assert_valid(&ssa, VerifyLevel::Full);

    let method = 0u32;
    let events = EventLog::<MockTarget>::new();
    let changed = passes::threading::run(&mut ssa, &method, &events, PointerSize::Bit64);

    assert!(changed);
    assert_valid(&ssa, VerifyLevel::Full);
}

#[test]
fn verifier_reports_builder_fixture_corruption_clearly() {
    let mut ssa = diamond_phi_fixture();
    let block = some_or_abort(ssa.block_mut(1));
    let terminator = some_or_abort(block.instructions_mut().last_mut());
    terminator.set_op(SsaOp::Return {
        value: Some(SsaVarId::PLACEHOLDER),
    });

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(errors
        .iter()
        .any(|err| matches!(err, VerifierError::PlaceholderVariable { .. })));
}
