//! Block merging pass — eliminates trampoline blocks and coalesces
//! single-edge block pairs in the SSA CFG.
//!
//! # Transformations
//!
//! 1. **Trampoline elimination** — removes blocks containing only an
//!    unconditional jump by redirecting predecessors to the ultimate target.
//!    Phi operands that referenced the trampoline are updated to reference
//!    the correct predecessor.
//! 2. **Block coalescing** — merges a block into its sole predecessor when
//!    the predecessor's only successor is that block. Phi nodes in the
//!    successor are converted to `Copy` instructions because they have
//!    exactly one incoming edge.
//!
//! Entry block (B0) is handled specially: when it's a trampoline, the
//! target block is inlined into B0 (if safe — single predecessor, no phis)
//! or the method is marked for code regeneration.
//!
//! # Algorithm
//!
//! Phase 1 iterates at most `max_iterations` times, each pass identifying
//! trampolines (via [`SsaFunction::find_trampoline_blocks`]), redirecting
//! all predecessors through [`redirect_target`](SsaOp::redirect_target),
//! then clearing the trampolines.
//!
//! Phase 2 handles the entry block specially (it has no predecessor, so
//! phase 1 cannot redirect through it).
//!
//! Phase 3 coalesces blocks: it computes predecessor counts, identifies
//! single-edge (A → B) pairs where B has exactly one predecessor, converts
//! B's phis to copies, appends B's instructions to A, and redirects phi
//! operands from B to A. Exception-handler boundary blocks are excluded
//! from coalescing.
//!
//! # Complexity
//!
//! O(n * max_iterations) where n is the number of blocks.

use std::collections::{BTreeMap, VecDeque};

use crate::{
    bitset::BitSet,
    events::{EventKind, EventListener},
    ir::{
        function::{SsaEditOptions, SsaEditor, SsaFunction},
        ops::SsaOp,
    },
    passes::utils::resolve_chain,
    target::Target,
};

/// Run block merging on `ssa`.
///
/// Executes three phases: trampoline elimination, entry-trampoline
/// simplification, and block coalescing. Each inner loop is capped by
/// `max_iterations`.
///
/// # Arguments
///
/// * `ssa` — The SSA function to simplify in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::BranchSimplified`] and
///   [`EventKind::BlockRemoved`] events.
/// * `max_iterations` — Cap on the inner fixpoint loops for both
///   trampoline elimination and block coalescing.
///
/// # Returns
///
/// `true` if any block was merged or any branch redirected.
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

    // Phase 1: eliminate trampoline blocks.
    for _ in 0..max_iterations {
        let iteration_changes = run_trampoline_iteration(ssa, method, events);
        if iteration_changes == 0 {
            break;
        }
        changed = true;
    }

    // Phase 2: handle entry trampoline (B0 has no predecessors so phase 1
    // can't redirect them — instead inline the target if safe, otherwise
    // mark for codegen regeneration).
    if simplify_entry_trampoline(ssa, method, events) {
        changed = true;
    }

    // Phase 3: coalesce non-trivial blocks connected by a single edge.
    if coalesce_blocks(ssa, method, events, max_iterations) > 0 {
        changed = true;
    }

    changed
}

fn run_trampoline_iteration<T, L>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let trampolines = ssa.find_trampoline_blocks(true);
    if trampolines.is_empty() {
        return 0;
    }
    let mut redirected = 0usize;
    let mut cleared = 0usize;
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        redirected = redirect_to_ultimate_targets(editor, &trampolines, method, events);
        cleared = clear_trampolines(editor, &trampolines, method, events);
        Ok(())
    });
    if result.is_err() {
        return 0;
    }
    redirected.saturating_add(cleared)
}

fn redirect_to_ultimate_targets<T, L>(
    editor: &mut SsaEditor<T>,
    trampolines: &BTreeMap<usize, usize>,
    method: &T::MethodRef,
    events: &L,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    if trampolines.is_empty() {
        return 0;
    }

    let ultimate_targets: BTreeMap<usize, usize> = trampolines
        .keys()
        .map(|&t| (t, resolve_chain(trampolines, t)))
        .collect();

    // Maps (trampoline, ultimate_target) → predecessors that redirected
    // through it; needed to fix up phi operands at the ultimate target.
    let mut redirected_preds: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    let mut redirected: usize = 0;

    let block_count = editor.function().block_count();
    for block_idx in 0..block_count {
        let Some(old_targets) = editor
            .function()
            .block(block_idx)
            .and_then(|block| block.terminator_op())
            .map(SsaOp::successors)
        else {
            continue;
        };

        // Only this block's actual successors can be trampolines, so look each
        // one up directly instead of trying every trampoline (which made this
        // O(blocks * trampolines)).
        let mut changed = false;
        for &target in &old_targets {
            let Some(&ultimate) = ultimate_targets.get(&target) else {
                continue;
            };
            if editor
                .redirect_terminator_target(block_idx, target, ultimate)
                .unwrap_or(false)
            {
                redirected_preds
                    .entry((target, ultimate))
                    .or_default()
                    .push(block_idx);
                changed = true;
            }
        }

        if changed {
            let new_targets = editor
                .function()
                .block(block_idx)
                .and_then(|block| block.terminator_op())
                .map(SsaOp::successors)
                .unwrap_or_default();
            let event = crate::events::Event {
                kind: EventKind::BranchSimplified,
                method: Some(method.clone()),
                location: Some(block_idx),
                message: format!(
                    "redirected through trampoline: {old_targets:?} -> {new_targets:?}"
                ),
                pass: None,
            };
            events.push(event);
            redirected = redirected.saturating_add(1);
        }
    }

    // Update phi operands at ultimate target blocks.
    for (&(trampoline, ultimate), preds) in &redirected_preds {
        let _ = editor.expand_phi_predecessor(ultimate, trampoline, preds);
    }

    redirected
}

fn clear_trampolines<T, L>(
    editor: &mut SsaEditor<T>,
    trampolines: &BTreeMap<usize, usize>,
    method: &T::MethodRef,
    events: &L,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut cleared: usize = 0;
    for &block_idx in trampolines.keys() {
        if editor.clear_block(block_idx).unwrap_or(false) {
            let event = crate::events::Event {
                kind: EventKind::BlockRemoved,
                method: Some(method.clone()),
                location: Some(block_idx),
                message: format!("cleared trampoline block B{block_idx}"),
                pass: None,
            };
            events.push(event);
            cleared = cleared.saturating_add(1);
        }
    }
    cleared
}

/// Inline B0's target when B0 is a trampoline. Non-entry trampolines are
/// handled by `run_trampoline_iteration`; B0 has no predecessors so that
/// approach can't reach it.
fn simplify_entry_trampoline<T, L>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let target = match ssa.block(0).and_then(|b| b.is_trampoline()) {
        Some(t) => t,
        None => return false,
    };

    let preds = ssa.block_predecessors(target);
    let target_has_phis = ssa.block(target).is_none_or(|b| !b.phi_nodes().is_empty());

    if preds.len() == 1 && preds.first().copied() == Some(0) && !target_has_phis {
        // Safe to inline: the target's only external predecessor is B0 and it
        // has no phis. Move target's instructions into B0, then redirect any
        // self-references (B_target had a back-edge to itself) to B0.
        let target_instrs = ssa
            .block(target)
            .map(|b| b.instructions().to_vec())
            .unwrap_or_default();

        let result = ssa.edit(SsaEditOptions::new(), |editor| {
            editor.remove_instruction_tail(0, 0)?;
            for (instr_idx, instr) in target_instrs.iter().cloned().enumerate() {
                editor.insert_instruction(0, instr_idx, instr)?;
            }
            let entry_len = editor
                .function()
                .block(0)
                .map(|block| block.instructions().len())
                .unwrap_or(0);
            for instr_idx in 0..entry_len {
                let Some(mut op) = editor
                    .function()
                    .block(0)
                    .and_then(|block| block.instruction(instr_idx))
                    .map(|instr| instr.op().clone())
                else {
                    continue;
                };
                if op.redirect_target(target, 0) {
                    editor.replace_instruction_op(0, instr_idx, op)?;
                }
            }
            editor.clear_block(target)?;
            Ok(())
        });
        if result.is_err() {
            return false;
        }
        let event = crate::events::Event {
            kind: EventKind::BlockRemoved,
            method: Some(method.clone()),
            location: Some(0),
            message: format!("inlined entry trampoline: B0 jump to B{target} merged into B0"),
            pass: None,
        };
        events.push(event);
        true
    } else {
        // Can't inline (multiple predecessors or phis); just mark as modified
        // so codegen regenerates clean IL without original junk bytes.
        let event = crate::events::Event {
            kind: EventKind::BranchSimplified,
            method: Some(method.clone()),
            location: Some(0),
            message: format!("entry block is trampoline to B{target} (regenerating clean IL)"),
            pass: None,
        };
        events.push(event);
        true
    }
}

/// Merge each block A into its sole predecessor when A is the only successor.
fn coalesce_blocks<T, L>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
    max_iterations: usize,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut merged: usize = 0;

    // Collect exception-handler boundary blocks.
    //
    // - Region *start* blocks must not be the merge target — absorbing a
    //   predecessor outside the region would pull non-region code in.
    // - Region *end* blocks must not be the merge source — absorbing a
    //   successor outside the region would extend the region.
    let mut no_merge_into = BitSet::new(ssa.block_count());
    let mut no_merge_from = BitSet::new(ssa.block_count());
    for handler in ssa.exception_handlers() {
        if let Some(b) = handler.try_start_block {
            no_merge_into.insert(b);
        }
        if let Some(b) = handler.try_end_block {
            no_merge_from.insert(b);
        }
        if let Some(b) = handler.handler_start_block {
            no_merge_into.insert(b);
        }
        if let Some(b) = handler.handler_end_block {
            no_merge_from.insert(b);
        }
        if let Some(b) = handler.filter_start_block {
            no_merge_into.insert(b);
        }
    }

    for _ in 0..max_iterations {
        let mut iteration_merges: usize = 0;

        let block_count = ssa.block_count();
        let mut pred_counts: Vec<usize> = vec![0; block_count];
        let mut pred_of: Vec<Option<usize>> = vec![None; block_count];
        for idx in 0..block_count {
            let successors = ssa
                .block(idx)
                .and_then(|b| b.terminator_op())
                .map(SsaOp::successors)
                .unwrap_or_default();
            for succ in successors {
                if succ < block_count {
                    if let Some(c) = pred_counts.get_mut(succ) {
                        *c = c.saturating_add(1);
                    }
                    if let Some(p) = pred_of.get_mut(succ) {
                        *p = Some(idx);
                    }
                }
            }
        }
        if let Some(c) = pred_counts.get_mut(0) {
            *c = c.saturating_add(1);
        }

        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut consumed = BitSet::new(block_count);
        for a_idx in 0..block_count {
            if consumed.contains(a_idx) {
                continue;
            }
            let b_idx = match ssa.block(a_idx).and_then(|b| b.terminator_op()) {
                Some(SsaOp::Jump { target }) => *target,
                _ => continue,
            };
            if b_idx >= block_count || b_idx == a_idx {
                continue;
            }
            if pred_counts.get(b_idx).copied().unwrap_or(0) != 1 {
                continue;
            }
            if no_merge_from.contains(a_idx) || no_merge_into.contains(b_idx) {
                continue;
            }
            if block_reaches(ssa, b_idx, a_idx) {
                continue;
            }
            let b_empty = ssa.block(b_idx).is_none_or(|b| b.instructions().is_empty());
            if b_empty {
                continue;
            }
            pairs.push((a_idx, b_idx));
            consumed.insert(a_idx);
            consumed.insert(b_idx);
        }

        if !pairs.is_empty() {
            let result = ssa.edit(SsaEditOptions::new(), |editor| {
                for &(a_idx, b_idx) in &pairs {
                    if editor.coalesce_unconditional_successor(a_idx, b_idx)? {
                        let event = crate::events::Event {
                            kind: EventKind::BlockRemoved,
                            method: Some(method.clone()),
                            location: Some(b_idx),
                            message: format!("coalesced B{b_idx} into B{a_idx}"),
                            pass: None,
                        };
                        events.push(event);
                        iteration_merges = iteration_merges.saturating_add(1);
                    }
                }
                Ok(())
            });
            if result.is_err() {
                iteration_merges = 0;
            }
        }

        merged = merged.saturating_add(iteration_merges);
        if iteration_merges == 0 {
            break;
        }
    }

    merged
}

fn block_reaches<T: Target>(ssa: &SsaFunction<T>, start: usize, target: usize) -> bool {
    if start >= ssa.block_count() || target >= ssa.block_count() {
        return false;
    }

    let mut visited = BitSet::new(ssa.block_count());
    let mut worklist = VecDeque::new();
    worklist.push_back(start);
    visited.insert(start);

    while let Some(block_idx) = worklist.pop_front() {
        let successors = ssa
            .block(block_idx)
            .and_then(|block| block.terminator_op())
            .map(SsaOp::successors)
            .unwrap_or_default();
        for successor in successors {
            if successor == target {
                return true;
            }
            if successor < ssa.block_count() && visited.insert(successor) {
                worklist.push_back(successor);
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            ops::SsaOp,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{run_mock_pass_boundary, MockTarget, MockType},
    };

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
    fn simple_trampoline_elimination() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "simple block merge", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(changed);
        // B1 trampoline should be eliminated
        assert!(log.has(EventKind::BranchSimplified) || log.has(EventKind::BlockRemoved));
    }

    #[test]
    fn chain_of_trampolines() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        for i in 1..4 {
            let mut b = SsaBlock::new(i);
            b.add_instruction(instr(SsaOp::Jump { target: i + 1 }));
            ssa.add_block(b);
        }
        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b4);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "trampoline chain block merge", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(changed, "chain of trampolines should be eliminated");
    }

    #[test]
    fn coalesce_sequential_blocks() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 1, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
        b1.add_instruction(instr(SsaOp::Return { value: Some(b) }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "sequential block coalescing", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(changed, "sequential blocks should coalesce");
    }

    #[test]
    fn coalesce_with_phi_operand() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 1, 0);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(10),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(v1, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Return {
            value: Some(phi_var),
        }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "phi operand block coalescing", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(changed, "coalescing should preserve phi operands");
    }

    #[test]
    fn entry_trampoline_is_handled() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 1, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(7),
        }));
        b1.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "entry trampoline block merge", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(changed, "entry trampoline should be handled");
    }

    #[test]
    fn empty_function_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty block merge", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(!changed);
    }

    #[test]
    fn no_trampoline_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "no-trampoline block merge", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(!changed, "no trampolines should mean no changes");
    }

    #[test]
    fn trampoline_with_phi_successor() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 2, 0);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(v0, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(20),
        }));
        b2.add_instruction(instr(SsaOp::Return {
            value: Some(phi_var),
        }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "phi successor trampoline merge", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(changed, "trampoline with phi successor should be handled");
    }
}
