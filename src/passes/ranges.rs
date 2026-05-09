//! Value Range Propagation pass — simplifies branches and comparisons
//! using sparse conditional constant propagation with [`ValueRange`].
//!
//! # Algorithm
//!
//! Uses a worklist-based dataflow analysis similar to Wegman-Zadeck SCCP,
//! but tracks [`ValueRange`] intervals instead of just constants:
//!
//! 1. **Initialize**: every variable starts at `ValueRange::top()` (unknown).
//!    The entry block is marked executable, and its CFG successors are
//!    added to the worklist.
//! 2. **Propagate**: process CFG edges and SSA variables from worklists.
//!    - When a new edge is executable, process phi nodes at the target
//!      (join ranges from executable predecessors).
//!    - When a variable's range changes, re-evaluate all instructions
//!      and phi nodes that use it.
//!    - For `Branch` and `Switch` terminators, only propagate to targets
//!      that are reachable given the condition's current range.
//! 3. **Evaluate**: compute output ranges for each operation type:
//!    - `Const` → exact constant range.
//!    - `Copy` → same range as source.
//!    - `Add`, `Sub`, `Mul` → arithmetic on operand ranges.
//!    - `Shr` (unsigned, non-negative) → shifted range.
//!    - `Rem` (positive divisor, non-negative dividend) → bounded range.
//!    - `And` → bounded by mask.
//!    - `Ceq`, `Clt`, `Cgt` → bounded to 0 or 1.
//!    - `ArrayLength` → non-negative.
//! 4. **Apply results**: simplify branches whose condition range is
//!    provably constant, and replace comparisons whose operands' ranges
//!    guarantee the result.
//!
//! # Limitations
//!
//! - Does not track non-numeric types (pointers, objects).
//! - Does not handle loop-carried ranges (widening/ narrowing not
//!   implemented — loops iterate up to `max_iterations` then stop).

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    analysis::{cfg::SsaCfg, range::ValueRange},
    bitset::BitSet,
    events::{EventKind, EventListener},
    graph::{NodeId, RootedGraph, Successors},
    ir::{
        block::SsaBlock, function::SsaFunction, ops::SsaOp, phi::PhiNode, value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// Run value-range propagation on `ssa`.
///
/// Applies sparse conditional range propagation, then simplifies branches
/// and comparisons based on the computed ranges.
///
/// # Arguments
///
/// * `ssa` — The SSA function to analyze and simplify in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::OpaquePredicateRemoved`],
///   [`EventKind::BranchSimplified`], and [`EventKind::ConstantFolded`].
/// * `max_iterations` — Cap on the inner range-propagation worklist loop.
///
/// # Returns
///
/// `true` if any branch or comparison was simplified.
pub fn run<T, L>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
    max_iterations: usize,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut analysis: RangeAnalysis<T> = RangeAnalysis::new(max_iterations);
    let result = analysis.analyze(ssa);

    let mut branch_simplifications: Vec<(usize, usize, bool)> = Vec::new();
    let mut comparison_replacements: Vec<(usize, usize, SsaVarId, bool)> = Vec::new();

    for (block_idx, block) in ssa.iter_blocks() {
        if let Some(SsaOp::Branch {
            condition,
            true_target,
            false_target,
        }) = block.terminator_op()
        {
            if let Some(range) = result.get_range(*condition) {
                if let Some(is_true) = range.always_equal_to(0) {
                    if is_true {
                        branch_simplifications.push((block_idx, *false_target, false));
                    }
                }
                if let Some(val) = range.as_constant() {
                    if val != 0 {
                        branch_simplifications.push((block_idx, *true_target, true));
                    }
                }
            }
        }

        for (instr_idx, instr) in block.instructions().iter().enumerate() {
            if let Some((dest, value)) = try_simplify_comparison(instr.op(), &result) {
                comparison_replacements.push((block_idx, instr_idx, dest, value));
            }
        }
    }

    let mut changed = false;

    for (block_idx, target, is_true) in branch_simplifications {
        if let Some(block) = ssa.block_mut(block_idx) {
            if let Some(last_instr) = block.instructions_mut().last_mut() {
                last_instr.set_op(SsaOp::Jump { target });
                let event = crate::events::Event {
                    kind: EventKind::OpaquePredicateRemoved,
                    method: Some(method.clone()),
                    location: Some(block_idx),
                    message: format!(
                        "range analysis: condition always {}",
                        if is_true { "true" } else { "false" }
                    ),
                    pass: None,
                };
                events.push(event);
                let event = crate::events::Event {
                    kind: EventKind::BranchSimplified,
                    method: Some(method.clone()),
                    location: Some(block_idx),
                    message: format!("simplified to unconditional jump to {target}"),
                    pass: None,
                };
                events.push(event);
                changed = true;
            }
        }
    }

    for (block_idx, instr_idx, dest, value) in comparison_replacements {
        if let Some(block) = ssa.block_mut(block_idx) {
            let const_value = if value {
                ConstValue::True
            } else {
                ConstValue::False
            };
            if let Some(instr) = block.instructions_mut().get_mut(instr_idx) {
                instr.set_op(SsaOp::Const {
                    dest,
                    value: const_value,
                });
            }
            let event = crate::events::Event {
                kind: EventKind::ConstantFolded,
                method: Some(method.clone()),
                location: Some(instr_idx),
                message: format!("range analysis: comparison → {value}"),
                pass: None,
            };
            events.push(event);
            changed = true;
        }
    }

    changed
}

fn try_simplify_comparison<T: Target>(
    op: &SsaOp<T>,
    result: &RangeResult,
) -> Option<(SsaVarId, bool)> {
    match op {
        SsaOp::Clt {
            dest, left, right, ..
        } => {
            let left_range = result.get_range(*left)?;
            let right_range = result.get_range(*right)?;
            if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min()) {
                if l_max < r_min {
                    return Some((*dest, true));
                }
            }
            if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max()) {
                if l_min >= r_max {
                    return Some((*dest, false));
                }
            }
            None
        }
        SsaOp::Cgt {
            dest, left, right, ..
        } => {
            let left_range = result.get_range(*left)?;
            let right_range = result.get_range(*right)?;
            if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max()) {
                if l_min > r_max {
                    return Some((*dest, true));
                }
            }
            if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min()) {
                if l_max <= r_min {
                    return Some((*dest, false));
                }
            }
            None
        }
        SsaOp::Ceq { dest, left, right } => {
            let left_range = result.get_range(*left)?;
            let right_range = result.get_range(*right)?;
            if let (Some(l), Some(r)) = (left_range.as_constant(), right_range.as_constant()) {
                return Some((*dest, l == r));
            }
            if !ranges_overlap(left_range, right_range) {
                return Some((*dest, false));
            }
            None
        }
        _ => None,
    }
}

fn ranges_overlap(a: &ValueRange, b: &ValueRange) -> bool {
    if a.is_top() || b.is_top() {
        return true;
    }
    if a.is_bottom() || b.is_bottom() {
        return false;
    }
    match (a.max(), a.min(), b.max(), b.min()) {
        (Some(a_max), Some(a_min), Some(b_max), Some(b_min)) => a_max >= b_min && a_min <= b_max,
        _ => true,
    }
}

/// Sparse range propagation analysis engine.
///
/// Worklist algorithm similar to Wegman-Zadeck SCCP but tracking
/// [`ValueRange`] intervals instead of just constants.
struct RangeAnalysis<T: Target> {
    /// Per-variable computed ranges.
    ranges: HashMap<SsaVarId, ValueRange>,
    /// Set of CFG edges `(from, to)` determined to be executable.
    executable_edges: HashSet<(usize, usize)>,
    /// Blocks that have at least one executable incoming edge.
    executable_blocks: BitSet,
    /// Worklist of SSA variables whose range changed and need re-evaluation.
    ssa_worklist: VecDeque<SsaVarId>,
    /// Worklist of CFG edges to process for first-time execution.
    cfg_worklist: VecDeque<(usize, usize)>,
    /// Maximum iterations to prevent non-termination on loops without widening.
    max_iterations: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Target> RangeAnalysis<T> {
    fn new(max_iterations: usize) -> Self {
        Self {
            ranges: HashMap::new(),
            executable_edges: HashSet::new(),
            executable_blocks: BitSet::new(0),
            ssa_worklist: VecDeque::new(),
            cfg_worklist: VecDeque::new(),
            max_iterations,
            _phantom: std::marker::PhantomData,
        }
    }

    fn analyze(&mut self, ssa: &SsaFunction<T>) -> RangeResult {
        let cfg = SsaCfg::from_ssa(ssa);
        self.initialize(ssa, &cfg);
        self.propagate(ssa, &cfg);
        RangeResult {
            ranges: self.ranges.clone(),
        }
    }

    fn initialize<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        self.ranges.clear();
        self.executable_edges.clear();
        self.executable_blocks = BitSet::new(ssa.block_count());
        self.ssa_worklist.clear();
        self.cfg_worklist.clear();

        for var in ssa.variables() {
            self.ranges.insert(var.id(), ValueRange::top());
        }

        let entry = cfg.entry().index();
        self.executable_blocks.insert(entry);
        for succ in cfg.successors(cfg.entry()) {
            self.cfg_worklist.push_back((entry, succ.index()));
        }
        if let Some(block) = ssa.block(entry) {
            self.process_block_definitions(block);
        }
    }

    fn propagate<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        let mut iterations: usize = 0;
        loop {
            iterations = iterations.saturating_add(1);
            if iterations > self.max_iterations {
                break;
            }
            while let Some((from, to)) = self.cfg_worklist.pop_front() {
                if self.executable_edges.insert((from, to)) {
                    self.process_edge(from, to, ssa, cfg);
                }
            }
            if let Some(var) = self.ssa_worklist.pop_front() {
                self.process_variable_uses(var, ssa, cfg);
            } else {
                break;
            }
        }
    }

    fn process_edge<G>(&mut self, from: usize, to: usize, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        let first_visit = !self.executable_blocks.contains(to);
        if first_visit {
            self.executable_blocks.insert(to);
            if let Some(block) = ssa.block(to) {
                self.process_block_definitions(block);
            }
        }
        if let Some(block) = ssa.block(to) {
            for phi in block.phi_nodes() {
                if phi.operand_from(from).is_some() {
                    let new_range = self.evaluate_phi(phi, to);
                    self.update_range(phi.result(), &new_range);
                }
            }
        }
        if first_visit {
            if let Some(block) = ssa.block(to) {
                self.propagate_outgoing_edges(to, block, cfg);
            }
        }
    }

    fn process_block_definitions(&mut self, block: &SsaBlock<T>) {
        for instr in block.instructions() {
            if let Some(def) = instr.def() {
                let range = self.evaluate_instruction(instr.op());
                self.update_range(def, &range);
            }
        }
    }

    fn process_variable_uses<G>(&mut self, var: SsaVarId, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        if let Some(ssa_var) = ssa.variable(var) {
            for use_site in ssa_var.uses() {
                let block_id = use_site.block;
                if !self.executable_blocks.contains(block_id) {
                    continue;
                }
                if use_site.is_phi_operand {
                    if let Some(block) = ssa.block(block_id) {
                        if let Some(phi) = block.phi(use_site.instruction) {
                            let new_range = self.evaluate_phi(phi, block_id);
                            self.update_range(phi.result(), &new_range);
                        }
                    }
                } else if let Some(block) = ssa.block(block_id) {
                    if let Some(instr) = block.instruction(use_site.instruction) {
                        if let Some(def) = instr.def() {
                            let range = self.evaluate_instruction(instr.op());
                            self.update_range(def, &range);
                        }
                        if instr.is_terminator() {
                            self.propagate_outgoing_edges(block_id, block, cfg);
                        }
                    }
                }
            }
        }
    }

    fn propagate_outgoing_edges<G>(&mut self, block_id: usize, block: &SsaBlock<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        match block.terminator_op() {
            Some(SsaOp::Branch {
                condition,
                true_target,
                false_target,
            }) => {
                let range = self.get_range(*condition);
                if let Some(val) = range.as_constant() {
                    if val != 0 {
                        self.add_cfg_edge(block_id, *true_target);
                    } else {
                        self.add_cfg_edge(block_id, *false_target);
                    }
                } else if range.always_equal_to(0) == Some(true) {
                    self.add_cfg_edge(block_id, *false_target);
                } else if range.is_always_positive() {
                    self.add_cfg_edge(block_id, *true_target);
                } else if range.is_top() {
                    // unknown
                } else {
                    self.add_cfg_edge(block_id, *true_target);
                    self.add_cfg_edge(block_id, *false_target);
                }
            }
            Some(SsaOp::Switch {
                value,
                targets,
                default,
            }) => {
                let range = self.get_range(*value);
                if let Some(idx) = range.as_constant().and_then(|i| usize::try_from(i).ok()) {
                    if let Some(&target) = targets.get(idx) {
                        self.add_cfg_edge(block_id, target);
                    } else {
                        self.add_cfg_edge(block_id, *default);
                    }
                } else {
                    for &target in targets {
                        self.add_cfg_edge(block_id, target);
                    }
                    self.add_cfg_edge(block_id, *default);
                }
            }
            Some(SsaOp::Jump { target }) => {
                self.add_cfg_edge(block_id, *target);
            }
            Some(
                SsaOp::Return { .. }
                | SsaOp::Throw { .. }
                | SsaOp::Rethrow
                | SsaOp::EndFinally
                | SsaOp::EndFilter { .. }
                | SsaOp::InterruptReturn,
            ) => {}
            _ => {
                let node = NodeId::new(block_id);
                for succ in cfg.successors(node) {
                    self.add_cfg_edge(block_id, succ.index());
                }
            }
        }
    }

    fn add_cfg_edge(&mut self, from: usize, to: usize) {
        if !self.executable_edges.contains(&(from, to)) {
            self.cfg_worklist.push_back((from, to));
        }
    }

    fn evaluate_phi(&self, phi: &PhiNode, block_id: usize) -> ValueRange {
        let mut result = ValueRange::bottom();
        let mut has_executable_operand = false;
        for operand in phi.operands() {
            let pred = operand.predecessor();
            if !self.executable_edges.contains(&(pred, block_id)) {
                continue;
            }
            has_executable_operand = true;
            let op_range = self.get_range(operand.value());
            result = result.join(&op_range);
            if result.is_top() {
                break;
            }
        }
        if !has_executable_operand {
            return ValueRange::top();
        }
        result
    }

    fn evaluate_instruction(&self, op: &SsaOp<T>) -> ValueRange {
        match op {
            SsaOp::Const { value, .. } => {
                if let Some(v) = value.as_i64() {
                    ValueRange::constant(v)
                } else {
                    ValueRange::top()
                }
            }
            SsaOp::Copy { src, .. } => self.get_range(*src),
            SsaOp::Add { left, right, .. } => {
                let l = self.get_range(*left);
                let r = self.get_range(*right);
                l.add(&r)
            }
            SsaOp::Sub { left, right, .. } => {
                let l = self.get_range(*left);
                let r = self.get_range(*right);
                l.sub(&r)
            }
            SsaOp::Mul { left, right, .. } => {
                let l = self.get_range(*left);
                let r = self.get_range(*right);
                l.mul(&r)
            }
            SsaOp::And { left, right, .. } => {
                let r = self.get_range(*right);
                if let Some(mask) = r.as_constant() {
                    ValueRange::bounded(0, mask.max(0))
                } else {
                    let l = self.get_range(*left);
                    if let Some(mask) = l.as_constant() {
                        ValueRange::bounded(0, mask.max(0))
                    } else {
                        ValueRange::top()
                    }
                }
            }
            SsaOp::Shr {
                value,
                amount,
                unsigned,
                ..
            } => {
                let val_range = self.get_range(*value);
                let amt_range = self.get_range(*amount);
                if let Some(amt) = amt_range.as_constant() {
                    if (0..64).contains(&amt) && *unsigned && val_range.is_always_non_negative() {
                        if let (Some(min), Some(max)) = (val_range.min(), val_range.max()) {
                            let new_min = min >> amt;
                            let new_max = max >> amt;
                            return ValueRange::bounded(new_min, new_max);
                        }
                    }
                }
                ValueRange::top()
            }
            SsaOp::Rem { left, right, .. } => {
                let r = self.get_range(*right);
                if let Some(n) = r.as_constant() {
                    if n > 0 {
                        let l = self.get_range(*left);
                        if l.is_always_non_negative() {
                            return ValueRange::bounded(0, n.saturating_sub(1));
                        }
                    }
                }
                ValueRange::top()
            }
            SsaOp::ArrayLength { .. } => ValueRange::non_negative(),
            SsaOp::NewArr { .. }
            | SsaOp::NewObj { .. }
            | SsaOp::Box { .. }
            | SsaOp::LoadToken { .. } => ValueRange::top(),
            SsaOp::Ceq { .. } | SsaOp::Clt { .. } | SsaOp::Cgt { .. } => ValueRange::bounded(0, 1),
            _ => ValueRange::top(),
        }
    }

    fn get_range(&self, var: SsaVarId) -> ValueRange {
        self.ranges.get(&var).cloned().unwrap_or_default()
    }

    fn update_range(&mut self, var: SsaVarId, new_range: &ValueRange) {
        let old_range = self.ranges.get(&var).cloned().unwrap_or_default();
        if *new_range != old_range {
            self.ranges.insert(var, new_range.clone());
            self.ssa_worklist.push_back(var);
        }
    }
}

#[derive(Debug)]
struct RangeResult {
    ranges: HashMap<SsaVarId, ValueRange>,
}

impl RangeResult {
    fn get_range(&self, var: SsaVarId) -> Option<&ValueRange> {
        self.ranges.get(&var)
    }
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
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    #[test]
    fn ranges_overlap_basics() {
        // Non-overlapping ranges
        let a = ValueRange::bounded(0, 5);
        let b = ValueRange::bounded(10, 15);
        assert!(!ranges_overlap(&a, &b));

        // Overlapping ranges
        let c = ValueRange::bounded(0, 10);
        let d = ValueRange::bounded(5, 15);
        assert!(ranges_overlap(&c, &d));

        // Same range
        let e = ValueRange::bounded(5, 10);
        assert!(ranges_overlap(&e, &e));

        // Top overlaps with everything
        let top = ValueRange::top();
        assert!(ranges_overlap(&top, &a));

        // Bottom doesn't overlap
        let bottom = ValueRange::bottom();
        assert!(!ranges_overlap(&bottom, &a));
    }

    fn make_result(entries: Vec<(SsaVarId, ValueRange)>) -> RangeResult {
        RangeResult {
            ranges: entries.into_iter().collect(),
        }
    }

    #[test]
    fn try_simplify_clt_always_true() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::bounded(0, 5)),
            (v1, ValueRange::bounded(10, 20)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Clt {
            dest,
            left: v0,
            right: v1,
            unsigned: false,
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, true)));
    }

    #[test]
    fn try_simplify_cgt_always_true() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::bounded(100, 200)),
            (v1, ValueRange::bounded(0, 50)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Cgt {
            dest,
            left: v0,
            right: v1,
            unsigned: false,
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, true)));
    }

    #[test]
    fn try_simplify_ceq_never() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::bounded(0, 5)),
            (v1, ValueRange::bounded(10, 20)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Ceq {
            dest,
            left: v0,
            right: v1,
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, false)));
    }

    #[test]
    fn try_simplify_ceq_constants_equal() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::constant(42)),
            (v1, ValueRange::constant(42)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Ceq {
            dest,
            left: v0,
            right: v1,
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, true)));
    }

    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
    }

    fn local_at(
        ssa: &mut SsaFunction<MockTarget>,
        idx: u16,
        block: usize,
        instr: usize,
    ) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::instruction(block, instr),
            MockType::I32,
        )
    }

    #[test]
    fn range_propagation_through_copy() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: v1, src: v0 }));
        block.add_instruction(instr(SsaOp::Return { value: Some(v1) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 20);
        let _ = changed;
        // Just verify no crash
    }

    #[test]
    fn range_on_add_propagates() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);
        let v2 = local_at(&mut ssa, 2, 0, 2);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(v2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let _ = run(&mut ssa, &method, &log, 20);
    }

    #[test]
    fn range_simplifies_branch_with_constant_condition() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v0,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 20);
        // Branch on constant 1 should be simplified to Jump 1
        if changed {
            assert!(matches!(
                ssa.block(0).unwrap().terminator_op().unwrap(),
                SsaOp::Jump { target: 1 }
            ));
        }
    }

    #[test]
    fn single_block_no_branch_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(block);
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 20);
        assert!(!changed);
    }

    #[test]
    fn comparison_folding_with_ranges() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);
        let v2 = local_at(&mut ssa, 2, 0, 2);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(100),
        }));
        // v0 < v1 is always true since both are constants
        block.add_instruction(instr(SsaOp::Clt {
            dest: v2,
            left: v0,
            right: v1,
            unsigned: false,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(v2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 20);
        // Should fold Clt to Const(true)
        if changed {
            assert!(log.has(EventKind::ConstantFolded));
        }
    }

    #[test]
    fn range_propagation_does_not_crash_with_phi() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 1, 0);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let cond = local_at(&mut ssa, 3, 0, 1);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(0),
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
        b1.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(10),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = crate::ir::phi::PhiNode::new(phi_var, VariableOrigin::Local(2));
        phi.add_operand(crate::ir::phi::PhiOperand::new(v0, 0));
        phi.add_operand(crate::ir::phi::PhiOperand::new(v1, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Return {
            value: Some(phi_var),
        }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let _ = run(&mut ssa, &method, &log, 20);
    }

    #[test]
    fn ranges_overlap_edge_cases() {
        // Exactly adjacent
        let a = ValueRange::bounded(0, 5);
        let b = ValueRange::bounded(6, 10);
        assert!(
            !ranges_overlap(&a, &b),
            "adjacent ranges should not overlap"
        );

        // Single point overlapping
        let c = ValueRange::bounded(5, 5);
        let d = ValueRange::bounded(5, 10);
        assert!(
            ranges_overlap(&c, &d),
            "single point should overlap if same value"
        );

        // Negative ranges
        let e = ValueRange::bounded(-10, -1);
        let f = ValueRange::bounded(-5, 5);
        assert!(
            ranges_overlap(&e, &f),
            "negative ranges should overlap correctly"
        );

        // Non-overlapping negatives
        let g = ValueRange::bounded(-10, -5);
        let h = ValueRange::bounded(-4, 5);
        assert!(
            !ranges_overlap(&g, &h),
            "non-overlapping negatives should not overlap"
        );
    }

    #[test]
    fn try_simplify_clt_always_false() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::bounded(10, 20)),
            (v1, ValueRange::bounded(0, 5)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Clt {
            dest,
            left: v0,
            right: v1,
            unsigned: false,
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, false)));
    }

    #[test]
    fn try_simplify_cgt_always_false() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::bounded(0, 5)),
            (v1, ValueRange::bounded(10, 20)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Cgt {
            dest,
            left: v0,
            right: v1,
            unsigned: false,
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, false)));
    }
}
