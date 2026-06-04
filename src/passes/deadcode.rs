//! Dead code elimination passes — both per-method and interprocedural.
//!
//! # Per-method DCE ([`run`] / [`run_iteration`])
//!
//! Performs the following steps each iteration until convergence:
//!
//! 1. **Reachable blocks** — BFS from the entry block (plus exception
//!    handler roots) to determine which blocks are live.
//! 2. **Clear unreachable blocks** — instructions and phis in blocks not
//!    reachable from entry or any handler are removed.
//! 3. **Remove op-less instructions** — `Nop` instructions and other
//!    stack-simulation artefacts are stripped.
//! 4. **Remove Nops** — simplifies the CFG for subsequent block merging.
//! 5. **Prune phi operands** — remove phi operands from unreachable
//!    predecessors.
//! 6. **Simplify trivial phis** — phis where all operands are the same
//!    variable (or self-referential) are replaced with a `Copy` or removed.
//! 7. **Recompute reachability** after phi changes.
//! 8. **Liveness via reverse dataflow** — compute which variables are
//!    live at each point using an RPO traversal and a worklist.
//! 9. **Remove dead phis** — phis whose results are not live.
//! 10. **Remove dead pure definitions** — instructions with no side effects
//!     whose results are not live.
//! 11. **Clean up Nops** created by dead-def removal.
//!
//! Each mutating step runs through the checked SSA editor and performs
//! boundary repair before the next analysis step.
//!
//! # Global DCE ([`run_global`])
//!
//! Interprocedural dead-method detection. Traverses the call graph from
//! [`World::entry_points`] and marks every method that is not transitively
//! reachable as dead. The host's [`World::callees`] decides how call edges
//! are resolved.
//!
//! # Shared utilities
//!
//! - [`find_dead_tails`] — identifies unreachable code after a terminator
//!   (shared with the control flow pass).

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use crate::{
    analysis::{cfg::SsaCfg, phis::PhiAnalyzer},
    bitset::BitSet,
    events::{EventKind, EventListener},
    graph::{algorithms, NodeId},
    ir::{
        function::{SsaEditOptions, SsaFunction},
        ops::SsaOp,
        phi::PhiNode,
        variable::{SsaVarId, VariableOrigin},
    },
    target::Target,
    world::World,
};

/// Identify blocks with unreachable code after a terminator.
///
/// Scans all blocks for instructions that follow a terminator (return,
/// throw, etc.) within the same block. These are unreachable and should
/// be removed.
///
/// # Arguments
///
/// * `ssa` — The SSA function to scan.
///
/// # Returns
///
/// A vector of `(block_idx, first_dead_instr_idx)` pairs describing the
/// unreachable instruction spans.
#[must_use]
pub fn find_dead_tails<T: Target>(ssa: &SsaFunction<T>) -> Vec<(usize, usize)> {
    ssa.iter_blocks()
        .filter_map(|(block_idx, block)| {
            let last_idx = block.instruction_count().checked_sub(1)?;
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if instr.op().is_terminator() && instr_idx < last_idx {
                    return Some((block_idx, instr_idx.saturating_add(1)));
                }
            }
            None
        })
        .collect()
}

/// Run per-method dead code elimination on `ssa`.
///
/// Iterates the multi-step DCE algorithm (reachability, liveness, dead
/// removal) until convergence. Each checked edit boundary performs the
/// required SSA repair before the next analysis step.
///
/// # Arguments
///
/// * `ssa` — The SSA function to clean up in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::InstructionRemoved`],
///   [`EventKind::BlockRemoved`], and [`EventKind::PhiSimplified`] events.
/// * `max_iterations` — Cap on the outer fixpoint loop.
///
/// # Returns
///
/// `true` if any instruction, phi node, or block was removed.
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
    let mut changed = false;
    for _ in 0..max_iterations {
        if run_iteration(ssa, method, events) == 0 {
            break;
        }
        changed = true;
    }
    changed
}

/// Single iteration of per-method DCE.
///
/// Performs reachability analysis, removes unreachable blocks, strips
/// op-less instructions and nops, prunes phi operands, simplifies trivial
/// phis, computes liveness, and removes dead phis and definitions.
///
/// # Arguments
///
/// * `ssa` — The SSA function to clean up in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for transformation events.
///
/// # Returns
///
/// The total number of changes made this iteration. Zero means the
/// algorithm has converged.
pub fn run_iteration<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut total_changes: usize = 0;

    // Step 1: reachable blocks (entry + exception handlers + fallback).
    let reachable = find_reachable_blocks(ssa);

    // Step 2: clear unreachable blocks.
    total_changes =
        total_changes.saturating_add(clear_unreachable_blocks(ssa, &reachable, method, events));

    // Step 3: remove op-less instructions (stack-simulation artifacts).
    let opless = find_opless_instructions(ssa, &reachable);
    total_changes =
        total_changes.saturating_add(remove_opless_instructions(ssa, &opless, method, events));

    // Step 4: remove Nop instructions (simplifies CFG for block merging).
    total_changes =
        total_changes.saturating_add(remove_nop_instructions(ssa, &reachable, method, events));

    // Step 5: prune phi operands from unreachable predecessors.
    let mut pruned = 0usize;
    let _ = ssa.edit(SsaEditOptions::new(), |editor| {
        pruned = editor.prune_phi_operands(&reachable);
        Ok(())
    });
    total_changes = total_changes.saturating_add(pruned);
    let reachable_set: BTreeSet<usize> = reachable.iter().collect();

    // Step 6: simplify trivial phis (purely structural).
    let trivial_phis = PhiAnalyzer::new(ssa).find_all_trivial(&reachable_set);
    total_changes =
        total_changes.saturating_add(simplify_trivial_phis(ssa, &trivial_phis, method, events));

    // Step 7+8: recompute reachability after phi simplification, then compute
    // liveness. Steps 1-6 do not modify any terminator, so a single CFG built
    // here serves both the reachability sweep and the RPO traversal.
    let cfg = SsaCfg::from_ssa(ssa);
    let reachable = find_reachable_blocks_with_cfg(ssa, &cfg);
    let rpo = compute_reverse_postorder(ssa, &reachable, &cfg);
    let live = compute_live_variables(ssa, &reachable, &rpo);

    // Step 9: remove dead phis.
    let dead_phis = find_dead_phis(ssa, &reachable, &live);
    let mut dead_phi_results = BitSet::new(ssa.var_id_capacity());
    for &(block_idx, phi_idx) in &dead_phis {
        if let Some(result) = ssa
            .block(block_idx)
            .and_then(|b| b.phi_nodes().get(phi_idx))
            .map(PhiNode::result)
        {
            dead_phi_results.insert(result.index());
        }
    }
    remove_phis(ssa, &dead_phis, method, events);
    total_changes = total_changes.saturating_add(dead_phis.len());

    // Step 10: dead pure definitions.
    let dead_defs = find_dead_definitions(ssa, &reachable, &live, &dead_phi_results);
    let c10 = dead_defs.len();
    remove_instructions(ssa, &dead_defs, method, events);
    total_changes = total_changes.saturating_add(c10);

    total_changes
}

fn find_reachable_blocks<T: Target>(ssa: &SsaFunction<T>) -> BitSet {
    if ssa.block_count() == 0 {
        return BitSet::new(0);
    }
    let cfg = SsaCfg::from_ssa(ssa);
    find_reachable_blocks_with_cfg(ssa, &cfg)
}

/// Reachability using a prebuilt CFG, so callers that already have one (and
/// that have not changed any terminator) avoid rebuilding it.
fn find_reachable_blocks_with_cfg<T: Target>(ssa: &SsaFunction<T>, cfg: &SsaCfg<T>) -> BitSet {
    if ssa.block_count() == 0 {
        return BitSet::new(0);
    }

    let mut reachable = BitSet::new(ssa.block_count());
    for n in algorithms::bfs(cfg, NodeId::new(0)) {
        let n: NodeId = n;
        reachable.insert(n.index());
    }

    let mut exception_roots = BitSet::new(ssa.block_count());
    for handler in ssa.exception_handlers() {
        if let Some(handler_block) = handler.handler_start_block {
            if !reachable.contains(handler_block) {
                exception_roots.insert(handler_block);
            }
        }
        if let Some(filter_block) = handler.filter_start_block {
            if !reachable.contains(filter_block) {
                exception_roots.insert(filter_block);
            }
        }
    }

    // Fallback: handler blocks recognized by their first instruction.
    for (block_idx, block) in ssa.iter_blocks() {
        if reachable.contains(block_idx) || exception_roots.contains(block_idx) {
            continue;
        }
        if let Some(first_instr) = block.instructions().first() {
            if matches!(first_instr.op(), SsaOp::EndFinally | SsaOp::Rethrow) {
                exception_roots.insert(block_idx);
            }
        }
    }

    for root in exception_roots.iter() {
        for node in algorithms::bfs(cfg, NodeId::new(root)) {
            let node: NodeId = node;
            reachable.insert(node.index());
        }
    }

    reachable
}

fn compute_reverse_postorder<T: Target>(
    ssa: &SsaFunction<T>,
    reachable: &BitSet,
    cfg: &SsaCfg<T>,
) -> Vec<usize> {
    if ssa.block_count() == 0 || reachable.is_empty() {
        return Vec::new();
    }

    let mut rpo: Vec<usize> = algorithms::reverse_postorder(cfg, NodeId::new(0))
        .into_iter()
        .map(|n: NodeId| n.index())
        .filter(|idx| reachable.contains(*idx))
        .collect();

    let mut in_rpo = BitSet::new(ssa.block_count());
    for &idx in &rpo {
        in_rpo.insert(idx);
    }
    let mut additional: Vec<usize> = reachable
        .iter()
        .filter(|idx| !in_rpo.contains(*idx))
        .collect();
    additional.sort_unstable();

    for &root in &additional {
        // Maintain `in_rpo` incrementally so membership is O(1); a linear
        // `rpo.contains` here would make the handler sweep O(blocks^2).
        for node in algorithms::reverse_postorder(cfg, NodeId::new(root)) {
            let idx = node.index();
            if reachable.contains(idx) && !in_rpo.contains(idx) {
                in_rpo.insert(idx);
                rpo.push(idx);
            }
        }
    }

    rpo
}

fn compute_live_variables<T: Target>(
    ssa: &SsaFunction<T>,
    reachable: &BitSet,
    rpo: &[usize],
) -> BitSet {
    let mut live = BitSet::new(ssa.var_id_capacity());
    let mut worklist = VecDeque::new();

    // Phase 1: roots — operands of side-effectful ops, return values, throws.
    for &block_idx in rpo {
        if !reachable.contains(block_idx) {
            continue;
        }
        if let Some(block) = ssa.block(block_idx) {
            for instr in block.instructions() {
                let op = instr.op();
                if !op.effects().is_pure() {
                    op.for_each_use(|var| {
                        if live.insert(var.index()) {
                            worklist.push_back(var);
                        }
                    });
                }
                if let SsaOp::Return { value: Some(v) } = op {
                    if live.insert(v.index()) {
                        worklist.push_back(*v);
                    }
                }
                if let SsaOp::Throw { exception } = op {
                    if live.insert(exception.index()) {
                        worklist.push_back(*exception);
                    }
                }
            }
        }
    }

    // Phase 2: backward propagation.
    let mut def_uses: BTreeMap<SsaVarId, Vec<SsaVarId>> = BTreeMap::new();

    let mut origin_defs: BTreeMap<VariableOrigin, Vec<SsaVarId>> = BTreeMap::new();
    let mut load_local_info: Vec<(SsaVarId, VariableOrigin)> = Vec::new();

    for &block_idx in rpo {
        if !reachable.contains(block_idx) {
            continue;
        }
        if let Some(block) = ssa.block(block_idx) {
            for phi in block.phi_nodes() {
                let def = phi.result();
                for operand in phi.operands() {
                    def_uses.entry(def).or_default().push(operand.value());
                }
                let origin = phi.origin();
                if matches!(
                    origin,
                    VariableOrigin::Local(_) | VariableOrigin::Argument(_)
                ) {
                    origin_defs.entry(origin).or_default().push(def);
                }
            }

            for instr in block.instructions() {
                let op = instr.op();
                let defs: Vec<SsaVarId> = op.defs().collect();
                for &def in &defs {
                    op.for_each_use(|use_var| {
                        def_uses.entry(def).or_default().push(use_var);
                    });
                    if let Some(var) = ssa.variable(def) {
                        let origin = var.origin();
                        if matches!(
                            origin,
                            VariableOrigin::Local(_) | VariableOrigin::Argument(_)
                        ) {
                            origin_defs.entry(origin).or_default().push(def);
                        }
                    }
                }

                match op {
                    SsaOp::LoadLocal { dest, local_index } => {
                        load_local_info.push((*dest, VariableOrigin::Local(*local_index)));
                    }
                    SsaOp::LoadLocalAddr { dest, local_index } => {
                        load_local_info.push((*dest, VariableOrigin::Local(*local_index)));
                    }
                    SsaOp::LoadArg { dest, arg_index } => {
                        load_local_info.push((*dest, VariableOrigin::Argument(*arg_index)));
                    }
                    _ => {}
                }
            }
        }
    }

    while let Some(var) = worklist.pop_front() {
        if let Some(uses) = def_uses.get(&var) {
            for &use_var in uses {
                if live.insert(use_var.index()) {
                    worklist.push_back(use_var);
                }
            }
        }
    }

    // Phase 3: bridge LoadLocal/LoadArg → defs of corresponding origin. Loop
    // because re-propagation can keep newly-live LoadLocals alive (e.g. a
    // Copy of a LoadLocal's dest going live transitively).
    loop {
        let mut newly_live = false;
        for (dest, origin) in &load_local_info {
            if !live.contains(dest.index()) {
                continue;
            }
            if let Some(defs) = origin_defs.get(origin) {
                for &def_var in defs {
                    if live.insert(def_var.index()) {
                        worklist.push_back(def_var);
                        newly_live = true;
                    }
                }
            }
        }
        while let Some(var) = worklist.pop_front() {
            if let Some(uses) = def_uses.get(&var) {
                for &use_var in uses {
                    if live.insert(use_var.index()) {
                        worklist.push_back(use_var);
                    }
                }
            }
        }
        if !newly_live {
            break;
        }
    }

    live
}

fn find_dead_definitions<T: Target>(
    ssa: &SsaFunction<T>,
    reachable: &BitSet,
    live: &BitSet,
    dead_phi_results: &BitSet,
) -> Vec<(usize, usize)> {
    let mut dead_vars = BitSet::new(ssa.var_id_capacity());
    let mut dead = Vec::new();

    for block_idx in reachable.iter() {
        if let Some(block) = ssa.block(block_idx) {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let op = instr.op();
                if !op.effects().removable_when_unused() {
                    continue;
                }
                if matches!(op, SsaOp::Pop { .. }) {
                    continue;
                }
                let defs: Vec<SsaVarId> = op.defs().collect();
                if defs.is_empty() {
                    dead.push((block_idx, instr_idx));
                } else if defs.iter().all(|def| !live.contains(def.index())) {
                    dead.push((block_idx, instr_idx));
                    for def in defs {
                        dead_vars.insert(def.index());
                    }
                }
            }
        }
    }

    // Dead Pop: operand's definer is being removed in this iteration.
    for block_idx in reachable.iter() {
        if let Some(block) = ssa.block(block_idx) {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if let SsaOp::Pop { value } = instr.op() {
                    let instr_definer_dead = dead_vars.contains(value.index());
                    let phi_definer_dead = dead_phi_results.contains(value.index());
                    if instr_definer_dead || phi_definer_dead {
                        dead.push((block_idx, instr_idx));
                    }
                }
            }
        }
    }

    dead
}

fn find_dead_phis<T: Target>(
    ssa: &SsaFunction<T>,
    reachable: &BitSet,
    live: &BitSet,
) -> Vec<(usize, usize)> {
    let mut dead = Vec::new();
    for block_idx in reachable.iter() {
        if let Some(block) = ssa.block(block_idx) {
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                if !live.contains(phi.result().index()) {
                    dead.push((block_idx, phi_idx));
                }
            }
        }
    }
    dead
}

fn remove_instructions<T, L>(
    ssa: &mut SsaFunction<T>,
    dead_defs: &[(usize, usize)],
    method: &T::MethodRef,
    events: &L,
) where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut by_block: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(block_idx, instr_idx) in dead_defs {
        by_block.entry(block_idx).or_default().push(instr_idx);
    }

    let _ = ssa.edit(SsaEditOptions::new(), |editor| {
        for (block_idx, mut indices) in by_block {
            indices.sort_by(|a, b| b.cmp(a));
            for instr_idx in indices {
                let Some(instr) = editor
                    .function()
                    .block(block_idx)
                    .and_then(|block| block.instructions().get(instr_idx))
                else {
                    continue;
                };

                let defs: Vec<SsaVarId> = instr.op().defs().collect();
                let message = if defs.is_empty() {
                    format!("dead {}", instr.mnemonic())
                } else {
                    format!("dead definition(s) {defs:?}")
                };

                editor.nop_instruction(block_idx, instr_idx)?;
                let location = block_idx.saturating_mul(1000).saturating_add(instr_idx);
                push_event(
                    events,
                    EventKind::InstructionRemoved,
                    method,
                    Some(location),
                    message,
                );
            }
        }
        Ok(())
    });
}

fn remove_phis<T, L>(
    ssa: &mut SsaFunction<T>,
    dead_phis: &[(usize, usize)],
    method: &T::MethodRef,
    events: &L,
) where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut by_block: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(block_idx, phi_idx) in dead_phis {
        by_block.entry(block_idx).or_default().push(phi_idx);
    }

    let _ = ssa.edit(SsaEditOptions::new(), |editor| {
        for (block_idx, mut indices) in by_block {
            indices.sort_by(|a, b| b.cmp(a));
            for phi_idx in indices {
                if editor.remove_phi(block_idx, phi_idx).is_ok() {
                    push_event(
                        events,
                        EventKind::PhiSimplified,
                        method,
                        Some(block_idx),
                        "removed dead phi node".to_string(),
                    );
                }
            }
        }
        Ok(())
    });
}

fn simplify_trivial_phis<T, L>(
    ssa: &mut SsaFunction<T>,
    trivial_phis: &[(usize, usize, Option<SsaVarId>)],
    method: &T::MethodRef,
    events: &L,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut simplified: usize = 0;

    let mut by_block: BTreeMap<usize, Vec<(usize, Option<SsaVarId>)>> = BTreeMap::new();
    for &(block_idx, phi_idx, replacement) in trivial_phis {
        by_block
            .entry(block_idx)
            .or_default()
            .push((phi_idx, replacement));
    }

    let _ = ssa.edit(SsaEditOptions::new(), |editor| {
        for (block_idx, mut phis) in by_block {
            phis.sort_by_key(|p| std::cmp::Reverse(p.0));
            for (phi_idx, replacement) in phis {
                if let Some(replacement_var) = replacement {
                    if editor
                        .simplify_phi_to_copy(block_idx, phi_idx, replacement_var)
                        .is_ok()
                    {
                        push_event(
                            events,
                            EventKind::PhiSimplified,
                            method,
                            Some(block_idx),
                            format!("replaced with {replacement_var}"),
                        );
                        simplified = simplified.saturating_add(1);
                    }
                } else if editor.remove_phi(block_idx, phi_idx).is_ok() {
                    push_event(
                        events,
                        EventKind::PhiSimplified,
                        method,
                        Some(block_idx),
                        "removed self-referential phi".to_string(),
                    );
                    simplified = simplified.saturating_add(1);
                }
            }
        }
        Ok(())
    });

    simplified
}

fn clear_unreachable_blocks<T, L>(
    ssa: &mut SsaFunction<T>,
    reachable: &BitSet,
    method: &T::MethodRef,
    events: &L,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut cleared: usize = 0;
    let total_blocks = ssa.block_count();
    let _ = ssa.edit(SsaEditOptions::new(), |editor| {
        for block_idx in 0..total_blocks {
            if !reachable.contains(block_idx) && editor.clear_block(block_idx).unwrap_or(false) {
                push_event(
                    events,
                    EventKind::BlockRemoved,
                    method,
                    Some(block_idx),
                    format!("removed unreachable block {block_idx}"),
                );
                cleared = cleared.saturating_add(1);
            }
        }
        Ok(())
    });
    cleared
}

fn find_opless_instructions<T: Target>(
    ssa: &SsaFunction<T>,
    reachable: &BitSet,
) -> Vec<(usize, usize)> {
    let mut opless = Vec::new();
    for block_idx in reachable.iter() {
        if let Some(block) = ssa.block(block_idx) {
            let instr_count = block.instructions().len();
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let is_last = instr_idx == instr_count.saturating_sub(1);
                if matches!(instr.op(), SsaOp::Nop) && (!is_last || instr_count > 1) {
                    opless.push((block_idx, instr_idx));
                }
            }
        }
    }
    opless
}

fn remove_opless_instructions<T, L>(
    ssa: &mut SsaFunction<T>,
    opless: &[(usize, usize)],
    method: &T::MethodRef,
    events: &L,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    if opless.is_empty() {
        return 0;
    }
    let mut by_block: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(block_idx, instr_idx) in opless {
        by_block.entry(block_idx).or_default().push(instr_idx);
    }
    let mut removed: usize = 0;
    let _ = ssa.edit(SsaEditOptions::new(), |editor| {
        for (block_idx, mut indices) in by_block {
            indices.sort_by(|a, b| b.cmp(a));
            for instr_idx in indices {
                if let Some(instr) = editor
                    .function()
                    .block(block_idx)
                    .and_then(|block| block.instructions().get(instr_idx))
                {
                    let mnemonic_owned = instr.mnemonic().to_string();
                    editor.nop_instruction(block_idx, instr_idx)?;
                    let location = block_idx.saturating_mul(1000).saturating_add(instr_idx);
                    push_event(
                        events,
                        EventKind::InstructionRemoved,
                        method,
                        Some(location),
                        format!("removed op-less instruction: {mnemonic_owned}"),
                    );
                    removed = removed.saturating_add(1);
                }
            }
        }
        Ok(())
    });
    removed
}

fn remove_nop_instructions<T, L>(
    ssa: &mut SsaFunction<T>,
    reachable: &BitSet,
    method: &T::MethodRef,
    events: &L,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut nops = Vec::new();
    for block_idx in reachable.iter() {
        if let Some(block) = ssa.block(block_idx) {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if matches!(instr.op(), SsaOp::Nop) {
                    nops.push((block_idx, instr_idx));
                }
            }
        }
    }

    if nops.is_empty() {
        return 0;
    }

    let mut removed = 0usize;
    let _ = ssa.edit(SsaEditOptions::new(), |editor| {
        for (block_idx, instr_idx) in &nops {
            if editor.nop_instruction(*block_idx, *instr_idx).is_ok() {
                removed = removed.saturating_add(1);
            }
        }
        Ok(())
    });

    if removed > 0 {
        let mut per_block: BTreeMap<usize, usize> = BTreeMap::new();
        for (block_idx, _) in nops {
            let entry = per_block.entry(block_idx).or_insert(0);
            *entry = entry.saturating_add(1);
        }
        for (block_idx, count) in per_block {
            push_event(
                events,
                EventKind::InstructionRemoved,
                method,
                Some(block_idx),
                format!("removed {count} Nop instructions"),
            );
        }
    }

    removed
}

fn push_event<T, L>(
    events: &L,
    kind: EventKind,
    method: &T::MethodRef,
    location: Option<usize>,
    message: String,
) where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let event = crate::events::Event {
        kind,
        method: Some(method.clone()),
        location,
        message,
        pass: None,
    };
    events.push(event);
}

/// Run interprocedural dead-method elimination.
///
/// Traverses the call graph from [`World::entry_points`], marking every
/// method that is not transitively reachable as dead via
/// [`World::mark_dead`].
///
/// # Arguments
///
/// * `world` — The host's [`World<T>`] providing entry points, call edges,
///   and the full method list.
/// * `events` — Event sink for [`EventKind::MethodMarkedDead`] events.
///
/// # Returns
///
/// `true` if any method was newly marked dead.
pub fn run_global<T, L, W>(world: &W, events: &L) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
    W: World<T> + ?Sized,
{
    let entries = world.entry_points();
    let mut live: HashSet<T::MethodRef> = entries.iter().cloned().collect();
    let mut worklist: VecDeque<T::MethodRef> = entries.into_iter().collect();

    while let Some(method) = worklist.pop_front() {
        for callee in world.callees(&method) {
            if live.insert(callee.clone()) {
                worklist.push_back(callee);
            }
        }
    }

    let mut changed = false;
    for method in world.all_methods() {
        if !live.contains(&method) && !world.is_dead(&method) {
            world.mark_dead(&method);
            push_event(
                events,
                EventKind::MethodMarkedDead,
                &method,
                None,
                "method has no live callers".to_string(),
            );
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{
            assert_mock_valid_full, run_mock_malformed_cleanup_boundary, run_mock_pass_boundary,
            MockTarget, MockType, MockWorld,
        },
    };

    #[test]
    fn find_dead_tails_after_terminator() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Nop));
        ssa.add_block(block);

        let dead = find_dead_tails(&ssa);
        assert_eq!(dead, vec![(0, 1)]);
    }

    #[test]
    fn global_dce_marks_unreachable_methods() {
        // Methods 1, 2, 3, 4. Entry: 1. Edges: 1→2, 2→3. Method 4 is
        // unreachable.
        let world = MockWorld::new([1u32, 2, 3, 4], [1u32], [(1u32, 2), (2u32, 3)]);
        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_global(&world, &log);
        assert!(changed);
        assert!(world.is_dead(&4));
        assert!(!world.is_dead(&1));
        assert!(!world.is_dead(&2));
        assert!(!world.is_dead(&3));
        assert_eq!(log.count_kind(EventKind::MethodMarkedDead), 1);
    }

    #[test]
    fn global_dce_no_op_when_all_reachable() {
        let world = MockWorld::new([1u32, 2], [1u32], [(1u32, 2)]);
        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_global(&world, &log);
        assert!(!changed);
    }

    /// Verifies `find_dead_tails` finds a single dead-tail span when one
    /// `Const` follows a `Return`. This builds the invalid block manually so
    /// the dead-tail scanner can inspect instructions after a terminator.
    #[test]
    fn find_dead_tails_with_dead_code() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let mut block0: SsaBlock<MockTarget> = SsaBlock::new(0);
        block0.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        block0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: SsaVarId::from_index(0),
            value: ConstValue::I32(42),
        }));
        ssa.add_block(block0);

        let dead = find_dead_tails(&ssa);
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0], (0, 1));
    }

    /// Multiple `Const` instructions after a `Return` should still report a
    /// single dead-tail span starting at index 1.
    #[test]
    fn find_dead_tails_multiple_dead_instructions() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let mut block0: SsaBlock<MockTarget> = SsaBlock::new(0);
        block0.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        for (i, v) in [1, 2, 3].iter().enumerate() {
            block0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest: SsaVarId::from_index(i),
                value: ConstValue::I32(*v),
            }));
        }
        ssa.add_block(block0);

        let dead = find_dead_tails(&ssa);
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0], (0, 1));
    }

    /// Exercises `SsaOp::successors` for every terminator shape DCE recognizes
    /// during reachability.
    #[test]
    fn successor_extraction() {
        let op: SsaOp<MockTarget> = SsaOp::Jump { target: 5 };
        assert_eq!(op.successors(), vec![5]);

        let cond = SsaVarId::from_index(0);
        let op: SsaOp<MockTarget> = SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        };
        assert_eq!(op.successors(), vec![1, 2]);

        let val = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> = SsaOp::Switch {
            value: val,
            targets: vec![1, 2, 3],
            default: 4,
        };
        assert_eq!(op.successors(), vec![1, 2, 3, 4]);

        let op: SsaOp<MockTarget> = SsaOp::Return { value: None };
        assert!(op.successors().is_empty());

        let op: SsaOp<MockTarget> = SsaOp::Leave { target: 3 };
        assert_eq!(op.successors(), vec![3]);
    }

    /// A function with no blocks should be a no-op for DCE.
    #[test]
    fn empty_function() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "empty DCE", |ssa| run(ssa, &method, &log, 20));
        assert!(!changed);
    }

    /// A self-referential phi `v0 = phi(v0)` is trivial; DCE should remove it
    /// via `simplify_trivial_phis`.
    #[test]
    fn self_referential_phi_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let phi_var = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(0), MockType::I32);

        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(phi_var, 0)); // self-ref
        block.add_phi(phi);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        ssa.add_block(block);

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_malformed_cleanup_boundary(&mut ssa, "self-referential phi DCE", |ssa| {
                run(ssa, &method, &log, 20)
            });
        assert!(changed, "self-referential phi should be removed");
        // The trivial phi should have been removed.
        assert!(ssa
            .block(0)
            .expect("entry block should remain")
            .phi_nodes()
            .is_empty());
    }

    #[test]
    fn removes_unreachable_block() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(99),
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v1) }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "unreachable-block DCE", |ssa| {
            run(ssa, &method, &log, 20)
        });
        assert!(changed, "unreachable block should be removed");
        assert!(log.has(EventKind::BlockRemoved));
    }

    #[test]
    fn unused_pure_instruction_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        // v1 is computed but never used
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Neg {
            dest: v1,
            operand: v0,
            flags: None,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "unused pure instruction DCE", |ssa| {
            run(ssa, &method, &log, 20)
        });
        assert!(changed, "unused pure instruction should be removed");
        assert!(log.has(EventKind::InstructionRemoved));
    }

    #[test]
    fn dead_phi_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let phi_var = SsaVarId::from_index(2);
        ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(2),
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(v1, 1));
        b2.add_phi(phi);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "dead phi DCE", |ssa| run(ssa, &method, &log, 20));
        // phi_var is never used after the phi, so the phi should be removed
        assert!(changed);
        assert!(log.has(EventKind::PhiSimplified));
    }

    #[test]
    fn nop_instructions_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Nop));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Nop));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "nop DCE", |ssa| run(ssa, &method, &log, 20));
        assert!(changed, "Nop instructions should be removed");
        assert!(log.has(EventKind::InstructionRemoved));
        // Only the Return should remain
        assert_eq!(
            ssa.block(0)
                .expect("entry block should remain")
                .instructions()
                .len(),
            1
        );
    }

    #[test]
    fn side_effect_instruction_not_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let v2 = SsaVarId::from_index(2);
        ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(20),
        }));
        // Use an overflow-checked arithmetic op (impure — may throw)
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::AddOvf {
            dest: v2,
            left: v0,
            right: v1,
            unsigned: false,
            flags: None,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        // We don't expect specific changes; just verify the function stays valid
        let changed = run_mock_pass_boundary(&mut ssa, "side-effect DCE", |ssa| {
            run(ssa, &method, &log, 20)
        });
        assert!(!changed, "side-effecting unused instruction must remain");
    }

    #[test]
    fn all_blocks_reachable_nothing_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(2, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: v0,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(2),
        }));
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v1) }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "reachable-block DCE", |ssa| {
            run(ssa, &method, &log, 20)
        });
        // v1 is used, both blocks reachable — no dead code
        assert!(!changed, "reachable live blocks should not be removed");
    }

    #[test]
    fn run_iteration_returns_zero_on_no_work() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert_mock_valid_full(&ssa, "DCE iteration no-work before");
        let count = run_iteration(&mut ssa, &0u32, &log);
        assert_mock_valid_full(&ssa, "DCE iteration no-work after");
        assert_eq!(count, 0);
    }

    #[test]
    fn compact_variables_after_dce() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(99),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "variable compaction DCE", |ssa| {
            run(ssa, &method, &log, 20)
        });
        assert!(changed);
        // v1 should be gone after compaction
        assert!(log.has(EventKind::InstructionRemoved));
    }
}
