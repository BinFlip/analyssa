//! Memory SSA and alias analysis tests.

#![allow(clippy::unwrap_used)]

use analyssa::{
    analysis::memory::{
        analyze_alias, AliasResult, ArrayIndex, MemoryLocation, MemorySsa, MemorySsaStats,
    },
    analysis::SsaCfg,
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
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
fn memory_location_equality_and_hash() {
    let obj = SsaVarId::from_index(0);
    let loc1 = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);
    let loc2 = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);
    let loc3 = MemoryLocation::<MockTarget>::InstanceField(obj, 2u32);

    assert_eq!(loc1, loc2);
    assert_ne!(loc1, loc3);

    let static_loc = MemoryLocation::<MockTarget>::StaticField(1u32);
    assert_ne!(loc1, static_loc);
}

#[test]
fn alias_analysis_same_location_is_must_alias() {
    let obj = SsaVarId::from_index(0);
    let loc1 = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);
    let loc2 = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);

    assert_eq!(analyze_alias(&loc1, &loc2), AliasResult::MustAlias);
}

#[test]
fn alias_analysis_different_fields_diff_object() {
    let obj = SsaVarId::from_index(0);
    let loc1 = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);
    let loc2 = MemoryLocation::<MockTarget>::InstanceField(obj, 2u32);

    // Different fields on the same object: the analysis may return NoAlias or
    // MayAlias depending on whether it can prove non-overlap.
    let result = analyze_alias(&loc1, &loc2);
    assert!(result != AliasResult::MustAlias);

    let loc3 = MemoryLocation::<MockTarget>::InstanceField(SsaVarId::from_index(1), 1u32);
    // Same field but different known objects: can be NoAlias if objects are
    // provably distinct
    let _ = analyze_alias(&loc1, &loc3);
}

#[test]
fn alias_analysis_static_vs_instance() {
    let static_loc = MemoryLocation::<MockTarget>::StaticField(42u32);
    let instance_loc = MemoryLocation::<MockTarget>::InstanceField(SsaVarId::from_index(0), 42u32);

    assert_eq!(
        analyze_alias(&static_loc, &instance_loc),
        AliasResult::NoAlias
    );
}

#[test]
fn memory_location_array_and_indirect() {
    let array_loc = MemoryLocation::<MockTarget>::ArrayElement(
        SsaVarId::from_index(0),
        ArrayIndex::Constant(0),
    );
    let array_loc2 =
        MemoryLocation::<MockTarget>::ArrayElement(SsaVarId::from_index(0), ArrayIndex::Unknown);

    assert_eq!(
        analyze_alias(&array_loc, &array_loc2),
        AliasResult::MayAlias
    );
}

#[test]
fn memory_location_indirect_and_unknown() {
    let indirect = MemoryLocation::<MockTarget>::Indirect(SsaVarId::from_index(1));
    let unknown = MemoryLocation::<MockTarget>::Unknown;

    let _ = analyze_alias(&indirect, &unknown);
}

#[test]
fn memory_location_debug_output() {
    let loc = MemoryLocation::<MockTarget>::InstanceField(SsaVarId::from_index(5), 3u32);
    let debug_str = format!("{loc:?}");
    assert!(!debug_str.is_empty());
}

#[test]
fn alias_result_equality() {
    assert_eq!(AliasResult::MustAlias, AliasResult::MustAlias);
    assert_ne!(AliasResult::MustAlias, AliasResult::NoAlias);
    assert_ne!(AliasResult::MayAlias, AliasResult::NoAlias);
}

#[test]
fn memory_ssa_empty_function_stats() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg);

    let stats: MemorySsaStats = mem_ssa.stats();
    assert_eq!(stats.store_count, 0);
    assert_eq!(stats.load_count, 0);
    assert_eq!(stats.location_count, 0);
}

#[test]
fn memory_ssa_with_field_loads_and_stores() {
    let mut ssa = SsaFunction::new(0, 2);
    let obj = local(&mut ssa, 0, 0, 0);
    let val = local(&mut ssa, 1, 0, 1);
    let loaded = local(&mut ssa, 2, 0, 2);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: obj,
        value: ConstValue::I32(42),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: val,
        value: ConstValue::I32(100),
    }));
    b0.add_instruction(instr(SsaOp::StoreField {
        object: obj,
        field: 1u32,
        value: val,
    }));
    b0.add_instruction(instr(SsaOp::LoadField {
        dest: loaded,
        object: obj,
        field: 1u32,
    }));
    b0.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
    ssa.add_block(b0);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg);

    let stats: MemorySsaStats = mem_ssa.stats();
    assert_eq!(stats.store_count, 1);
    assert_eq!(stats.load_count, 1);
}

#[test]
fn memory_ssa_new_is_empty() {
    let mem_ssa: MemorySsa<MockTarget> = MemorySsa::new();
    let stats = mem_ssa.stats();
    assert_eq!(stats.store_count, 0);
    assert_eq!(stats.load_count, 0);
    assert_eq!(stats.memory_phi_count, 0);
    assert_eq!(stats.version_count, 0);
}

#[test]
fn memory_ssa_handles_store_in_branch() {
    let mut ssa = SsaFunction::new(0, 3);
    let obj = local(&mut ssa, 0, 0, 0);
    let val = local(&mut ssa, 1, 0, 1);
    let cond = local(&mut ssa, 2, 0, 2);
    let loaded = local(&mut ssa, 3, 2, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: obj,
        value: ConstValue::I32(42),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: val,
        value: ConstValue::I32(100),
    }));
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
    b1.add_instruction(instr(SsaOp::StoreField {
        object: obj,
        field: 5u32,
        value: val,
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::LoadField {
        dest: loaded,
        object: obj,
        field: 5u32,
    }));
    b2.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg);

    let stats: MemorySsaStats = mem_ssa.stats();
    assert!(
        stats.store_count >= 1,
        "expected at least 1 store, got {}",
        stats.store_count
    );
    assert!(
        stats.load_count >= 1,
        "expected at least 1 load, got {}",
        stats.load_count
    );
    assert!(
        stats.location_count >= 1,
        "expected at least 1 location, got {}",
        stats.location_count
    );
}
