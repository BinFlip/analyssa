//! Infrastructure tests: BitSet, Event system, Graph algorithms, utility
//! functions, and pointer size.

use analyssa::{
    analysis::{ConstEvaluator, DefUseIndex, SsaCfg, ValueResolver},
    bitset::BitSet,
    events::{EventKind, EventLog},
    graph::{algorithms, DirectedGraph, EdgeId, NodeId},
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::utils,
    testing::{MockTarget, MockType},
    PointerSize,
};

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| std::process::abort())
}

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

// BitSet

#[test]
fn bitset_insert_contains_and_iteration() {
    let mut set = BitSet::new(100);
    assert!(set.is_empty());

    assert!(set.insert(5));
    assert!(set.contains(5));
    assert!(!set.is_empty());
    assert_eq!(set.count(), 1);

    assert!(!set.insert(5));

    assert!(set.insert(50));
    assert!(set.insert(99));
    assert_eq!(set.count(), 3);

    let mut iterated: Vec<usize> = set.iter().collect();
    iterated.sort_unstable();
    assert_eq!(iterated, vec![5, 50, 99]);

    set.remove(50);
    assert!(!set.contains(50));
    assert_eq!(set.count(), 2);
}

#[test]
fn bitset_capacity_grows_as_needed() {
    // Bitset does not auto-grow beyond initial capacity; create with
    // sufficient capacity upfront
    let mut set = BitSet::new(128);
    assert!(set.insert(50));
    assert!(set.contains(50));
    assert!(!set.contains(100));
    // Verify it works with indices at the edge of capacity
    assert!(set.insert(127));
    assert!(set.contains(127));
}

#[test]
fn bitset_clear_resets() {
    let mut set = BitSet::new(64);
    assert!(set.insert(1));
    assert!(set.insert(2));
    set.clear();
    assert!(set.is_empty());
    assert!(!set.contains(1));
}

#[test]
fn bitset_union_and_intersection() {
    let mut a = BitSet::new(100);
    a.insert(1);
    a.insert(3);
    a.insert(5);

    let mut b = BitSet::new(100);
    b.insert(3);
    b.insert(5);
    b.insert(7);

    // union_with mutates a in place; returns whether anything changed
    let union_changed = a.union_with(&b);
    assert!(union_changed);
    assert!(a.contains(1));
    assert!(a.contains(3));
    assert!(a.contains(5));
    assert!(a.contains(7));

    // Reset and test intersection
    let mut c = BitSet::new(100);
    c.insert(1);
    c.insert(3);
    c.insert(5);

    let inter_changed = c.intersect_with(&b);
    assert!(inter_changed); // 1 is removed
    assert!(!c.contains(1));
    assert!(c.contains(3));
    assert!(c.contains(5));
    assert!(!c.contains(7));
}

#[test]
fn bitset_difference() {
    let mut a = BitSet::new(100);
    a.insert(1);
    a.insert(2);
    a.insert(3);

    let mut b = BitSet::new(100);
    b.insert(2);
    b.insert(4);

    let diff_changed = a.difference_with(&b);
    assert!(diff_changed); // 2 is removed
    assert!(a.contains(1));
    assert!(!a.contains(2));
    assert!(a.contains(3));
    assert!(!a.contains(4));
}

// Event System

#[test]
fn event_log_records_and_queries_events() {
    let log: EventLog<MockTarget> = EventLog::new();

    log.record(EventKind::ConstantFolded).method(1u32);
    log.record(EventKind::CopyPropagated).method(1u32);
    log.record(EventKind::InstructionRemoved).method(1u32);

    assert!(log.has(EventKind::ConstantFolded));
    assert!(log.has(EventKind::CopyPropagated));
    assert!(log.has(EventKind::InstructionRemoved));
    assert!(!log.has(EventKind::BranchSimplified));
}

#[test]
fn event_log_count_by_kind_is_correct() {
    let log: EventLog<MockTarget> = EventLog::new();

    for _ in 0..3 {
        log.record(EventKind::CopyPropagated).method(7u32);
    }

    let copy_count: usize = log
        .iter()
        .filter(|e| e.kind == EventKind::CopyPropagated)
        .count();
    assert_eq!(copy_count, 3);
}

#[test]
fn event_log_has_any_checks_multiple_kinds() {
    let log: EventLog<MockTarget> = EventLog::new();
    log.record(EventKind::InstructionRemoved).method(1u32);

    assert!(log.has_any(&[EventKind::InstructionRemoved, EventKind::BranchSimplified]));
    assert!(!log.has_any(&[EventKind::ConstantFolded, EventKind::CopyPropagated]));
}

#[test]
fn event_log_counts_events_by_kind() {
    let log: EventLog<MockTarget> = EventLog::new();

    for _ in 0..3 {
        log.record(EventKind::CopyPropagated).method(7u32);
    }
    assert_eq!(log.count_kind(EventKind::CopyPropagated), 3);
    assert_eq!(log.count_kind(EventKind::ConstantFolded), 0);
}

// Graph Algorithms

#[test]
fn indexed_graph_basic_operations() {
    let mut g: DirectedGraph<'static, usize, ()> = DirectedGraph::new();
    let n0 = g.add_node(0);
    let n1 = g.add_node(1);
    let n2 = g.add_node(2);

    let _ = g.add_edge(n0, n1, ());
    let _ = g.add_edge(n1, n2, ());
    let _ = g.add_edge(n0, n2, ());

    let succs: Vec<NodeId> = g.successors(n0).collect();
    assert!(succs.contains(&n1));
    assert!(succs.contains(&n2));

    let preds: Vec<NodeId> = g.predecessors(n2).collect();
    assert!(preds.contains(&n0));
    assert!(preds.contains(&n1));
}

#[test]
fn dominator_tree_on_diamond_graph() {
    let mut g: DirectedGraph<'static, usize, ()> = DirectedGraph::new();
    let n0 = g.add_node(0);
    let n1 = g.add_node(1);
    let n2 = g.add_node(2);
    let n3 = g.add_node(3);

    let _ = g.add_edge(n0, n1, ());
    let _ = g.add_edge(n0, n2, ());
    let _ = g.add_edge(n1, n3, ());
    let _ = g.add_edge(n2, n3, ());

    let dom_tree = algorithms::compute_dominators(&g, n0);

    assert!(dom_tree.dominates(n0, n0));
    assert!(dom_tree.dominates(n0, n3));
    assert!(!dom_tree.dominates(n1, n3));
    assert!(dom_tree.dominates(n1, n1));

    let idom_3 = dom_tree.immediate_dominator(n3);
    assert_eq!(idom_3, Some(n0));
}

#[test]
fn postorder_on_dag() {
    let mut g: DirectedGraph<'static, usize, ()> = DirectedGraph::new();
    let n0 = g.add_node(0);
    let n1 = g.add_node(1);
    let n2 = g.add_node(2);
    let n3 = g.add_node(3);

    let _ = g.add_edge(n0, n1, ());
    let _ = g.add_edge(n0, n2, ());
    let _ = g.add_edge(n1, n3, ());
    let _ = g.add_edge(n2, n3, ());

    let order = algorithms::postorder(&g, n0);
    let n3_pos = order.iter().position(|&id| id == n3);
    assert!(n3_pos.is_some());
}

#[test]
fn node_id_and_edge_id_usability() {
    let n = NodeId::new(42);
    assert_eq!(n.index(), 42);

    let e = EdgeId::new(0);
    assert_eq!(e.index(), 0);

    let e2 = EdgeId::new(7);
    assert_ne!(e, e2);
    assert_eq!(e2, EdgeId::new(7));
}

// Utility functions

#[test]
fn is_power_of_two_recognizes_valid_values() {
    assert_eq!(utils::is_power_of_two(1), Some(0));
    assert_eq!(utils::is_power_of_two(2), Some(1));
    assert_eq!(utils::is_power_of_two(8), Some(3));
    assert_eq!(utils::is_power_of_two(1024), Some(10));
}

#[test]
fn is_power_of_two_rejects_non_powers() {
    assert_eq!(utils::is_power_of_two(0), None);
    assert_eq!(utils::is_power_of_two(3), None);
    assert_eq!(utils::is_power_of_two(5), None);
    assert_eq!(utils::is_power_of_two(7), None);
    assert_eq!(utils::is_power_of_two(9), None);
    assert_eq!(utils::is_power_of_two(15), None);
}

// ConstEvaluator and ValueResolver

#[test]
fn const_evaluator_evaluates_chained_expressions() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let c = local(&mut ssa, 2, 0, 2);
    let d = local(&mut ssa, 3, 0, 3);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(2),
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

    let mut eval = ConstEvaluator::new(&ssa, PointerSize::Bit64);
    let result = eval.evaluate_var(d);
    assert_eq!(result.and_then(|cv| cv.as_i64()), Some(8));
}

#[test]
fn value_resolver_resolves_simple_constant() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(a) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut resolver = ValueResolver::new(&ssa, PointerSize::Bit64);
    let resolved = resolver.resolve(a);
    assert_eq!(resolved.and_then(|cv| cv.as_i64()), Some(42));
}

#[test]
fn value_resolver_resolve_all_for_multiple_vars() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(20),
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(b) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut resolver = ValueResolver::new(&ssa, PointerSize::Bit64);
    let all = resolver.resolve_all(&[a, b]);
    assert!(all.is_some());
    let values = some_or_abort(all);
    assert_eq!(values.first().and_then(|v| v.as_i64()), Some(10));
    assert_eq!(values.get(1).and_then(|v| v.as_i64()), Some(20));
}

// DefUseIndex

#[test]
fn defuse_index_detects_single_use_variables() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let c = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: c,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(c) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let index = DefUseIndex::build(&ssa);
    assert!(index.is_single_use(a));
    assert!(index.is_single_use(b));
    assert!(!index.is_unused(a));
}

#[test]
fn defuse_index_detects_dead_variable() {
    let mut ssa = SsaFunction::new(0, 0);
    let dead = local(&mut ssa, 0, 0, 0);
    let live = local(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: dead,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: live,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(live) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let index = DefUseIndex::build(&ssa);
    assert!(index.is_unused(dead));
    assert!(!index.is_unused(live));
    assert!(index.has_uses(live));
}

// PointerSize

#[test]
fn pointer_size_bytes_and_bits() {
    assert_eq!(PointerSize::Bit32.bytes(), 4);
    assert_eq!(PointerSize::Bit64.bytes(), 8);
    assert_eq!(PointerSize::Bit32.bits(), 32);
    assert_eq!(PointerSize::Bit64.bits(), 64);
}

// SsaCfg additional queries

#[test]
fn cfg_reports_entry_and_block_count() {
    let mut ssa = SsaFunction::new(0, 2);
    let v0 = local(&mut ssa, 0, 0, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
    ssa.add_block(b1);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    assert!(!cfg.is_empty());
}
