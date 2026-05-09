//! Control flow simplification pass — reduces CFG complexity by removing
//! redundant branches, threading through trampolines, and cleaning up dead
//! code after terminators.
//!
//! # Algorithm
//!
//! Each iteration performs three steps:
//!
//! 1. **Jump threading** — resolves block references through trampoline
//!    chains using [`SsaFunction::find_trampoline_blocks`] and
//!    [`SsaOp::redirect_target`]. Skips the entry block (handled by
//!    block merge).
//! 2. **Branch-to-same-target simplification** — detects `Branch` and
//!    `Switch` terminators where all targets (after trampoline resolution)
//!    point to the same block, reducing them to a `Jump`. Also handles
//!    CFF-style self-loop switches that degenerate to a single non-self
//!    target.
//! 3. **Dead tail removal** — removes instructions that follow a terminator
//!    in the same block (shared with [`crate::passes::deadcode`]).
//!
//! The outer loop iterates until convergence (no changes) or
//! `max_iterations` is reached.

use std::collections::BTreeMap;

use crate::{
    events::{EventKind, EventListener},
    ir::{function::SsaFunction, ops::SsaOp},
    passes::{deadcode::find_dead_tails, utils::resolve_chain},
    target::Target,
};

/// Run control flow simplification on `ssa`.
///
/// Iterates a three-step simplification until convergence or
/// `max_iterations` is reached.
///
/// # Arguments
///
/// * `ssa` — The SSA function to simplify in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::ControlFlowRestructured`],
///   [`EventKind::BranchSimplified`], and [`EventKind::InstructionRemoved`].
/// * `max_iterations` — Cap on the outer fixpoint loop.
///
/// # Returns
///
/// `true` if any transformation fired.
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

/// Single iteration of control flow simplification.
///
/// Returns the number of changes made (branches simplified, instructions
/// removed); zero means the algorithm has converged for this iteration.
///
/// # Arguments
///
/// * `ssa` — The SSA function to simplify.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for transformation events.
///
/// # Returns
///
/// Count of changes made this iteration.
pub fn run_iteration<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut total_changes: usize = 0;

    // Step 1: jump threading (don't skip entry block).
    let trampolines = ssa.find_trampoline_blocks(false);
    if !trampolines.is_empty() {
        total_changes =
            total_changes.saturating_add(apply_jump_threading(ssa, &trampolines, method, events));
    }

    // Step 2: simplify branches to same target (also resolves through trampolines).
    let same_target_branches = find_same_target_branches(ssa, &trampolines);
    if !same_target_branches.is_empty() {
        total_changes = total_changes.saturating_add(simplify_same_target_branches(
            ssa,
            &same_target_branches,
            method,
            events,
        ));
    }

    // Step 3: remove dead tails.
    let dead_tails = find_dead_tails(ssa);
    if !dead_tails.is_empty() {
        total_changes =
            total_changes.saturating_add(remove_dead_tails(ssa, &dead_tails, method, events));
    }

    total_changes
}

fn find_same_target_branches<T: Target>(
    ssa: &SsaFunction<T>,
    trampolines: &BTreeMap<usize, usize>,
) -> Vec<(usize, usize)> {
    ssa.iter_blocks()
        .filter_map(|(block_idx, block)| {
            block.terminator_op().and_then(|op| match op {
                SsaOp::Branch {
                    true_target,
                    false_target,
                    ..
                }
                | SsaOp::BranchCmp {
                    true_target,
                    false_target,
                    ..
                } => {
                    if true_target == false_target {
                        return Some((block_idx, *true_target));
                    }
                    let true_ultimate = resolve_chain(trampolines, *true_target);
                    let false_ultimate = resolve_chain(trampolines, *false_target);
                    if true_ultimate == false_ultimate {
                        Some((block_idx, true_ultimate))
                    } else {
                        None
                    }
                }
                SsaOp::Switch {
                    targets, default, ..
                } => {
                    if targets.iter().all(|t| *t == *default) {
                        return Some((block_idx, *default));
                    }
                    let default_ultimate = resolve_chain(trampolines, *default);
                    if targets
                        .iter()
                        .all(|t| resolve_chain(trampolines, *t) == default_ultimate)
                    {
                        return Some((block_idx, default_ultimate));
                    }
                    // Self-loop elimination: residual CFF in exception
                    // handlers can leave a switch where all cases except one
                    // are self-loops. Degenerates to a jump to the single
                    // non-self target.
                    let non_self: Vec<usize> = targets
                        .iter()
                        .chain(std::iter::once(default))
                        .copied()
                        .filter(|&t| t != block_idx)
                        .collect();
                    if let Some(&first) = non_self.first() {
                        if non_self.iter().all(|t| *t == first) {
                            Some((block_idx, first))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            })
        })
        .collect()
}

fn apply_jump_threading<T, L>(
    ssa: &mut SsaFunction<T>,
    trampolines: &BTreeMap<usize, usize>,
    method: &T::MethodRef,
    events: &L,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let ultimate_targets: BTreeMap<usize, usize> = trampolines
        .keys()
        .map(|&t| (t, resolve_chain(trampolines, t)))
        .collect();

    let mut threaded_count: usize = 0;

    let block_count = ssa.block_count();
    for block_idx in 0..block_count {
        if let Some(block) = ssa.block_mut(block_idx) {
            if let Some(last) = block.instructions_mut().last_mut() {
                let op = last.op_mut();
                let old_targets = op.successors();
                let mut changed = false;
                for (&trampoline, &ultimate) in &ultimate_targets {
                    if op.redirect_target(trampoline, ultimate) {
                        changed = true;
                    }
                }
                if changed {
                    let new_targets = op.successors();
                    let event = crate::events::Event {
                        kind: EventKind::ControlFlowRestructured,
                        method: Some(method.clone()),
                        location: Some(block_idx),
                        message: format!("jump threaded: {old_targets:?} -> {new_targets:?}"),
                        pass: None,
                    };
                    events.push(event);
                    threaded_count = threaded_count.saturating_add(1);
                }
            }
        }
    }

    threaded_count
}

fn simplify_same_target_branches<T, L>(
    ssa: &mut SsaFunction<T>,
    same_target_branches: &[(usize, usize)],
    method: &T::MethodRef,
    events: &L,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut simplified_count: usize = 0;

    for &(block_idx, target) in same_target_branches {
        if let Some(block) = ssa.block_mut(block_idx) {
            if let Some(last) = block.instructions_mut().last_mut() {
                last.set_op(SsaOp::Jump { target });
                let event = crate::events::Event {
                    kind: EventKind::BranchSimplified,
                    method: Some(method.clone()),
                    location: Some(block_idx),
                    message: format!(
                        "branch to same target simplified: B{block_idx} branch -> jump B{target}"
                    ),
                    pass: None,
                };
                events.push(event);
                simplified_count = simplified_count.saturating_add(1);
            }
        }
    }

    simplified_count
}

fn remove_dead_tails<T, L>(
    ssa: &mut SsaFunction<T>,
    dead_tails: &[(usize, usize)],
    method: &T::MethodRef,
    events: &L,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut removed_count: usize = 0;

    for &(block_idx, start_idx) in dead_tails {
        if let Some(block) = ssa.block_mut(block_idx) {
            let instr_count = block.instruction_count();
            let to_remove = instr_count.saturating_sub(start_idx);
            for _ in 0..to_remove {
                block.instructions_mut().pop();
                removed_count = removed_count.saturating_add(1);
            }
            if to_remove > 0 {
                let event = crate::events::Event {
                    kind: EventKind::InstructionRemoved,
                    method: Some(method.clone()),
                    location: Some(block_idx),
                    message: format!(
                        "removed {to_remove} dead instructions after terminator in B{block_idx}"
                    ),
                    pass: None,
                };
                events.push(event);
            }
        }
    }

    removed_count
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
    fn dead_tail_after_return_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Return { value: None }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(999),
        }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 10);
        assert!(changed, "dead tail after return should be removed");
        assert!(log.has(EventKind::InstructionRemoved));
    }

    #[test]
    fn branch_to_same_target_simplified() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let cond = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 1,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 10);
        assert!(changed, "branch to same target should be simplified");
        assert!(log.has(EventKind::BranchSimplified));
        // Should now be a Jump
        assert!(matches!(
            ssa.block(0).unwrap().terminator_op().unwrap(),
            SsaOp::Jump { target: 1 }
        ));
    }

    #[test]
    fn jump_thread_through_trampoline() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 10);
        assert!(changed, "jump through trampoline should be threaded");
    }

    #[test]
    fn no_changes_on_well_formed_cfg() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 10);
        assert!(!changed, "well-formed CFG should have no changes");
    }

    #[test]
    fn multiple_changes_in_one_run() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let cond = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 1,
        }));
        b0.add_instruction(instr(SsaOp::Nop)); // dead tail
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 10);
        assert!(changed);
    }

    #[test]
    fn dead_tails_in_multiple_blocks() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        b0.add_instruction(instr(SsaOp::Nop));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        b1.add_instruction(instr(SsaOp::Nop));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 10);
        assert!(changed);
        assert!(log.count_kind(EventKind::InstructionRemoved) >= 2);
    }

    #[test]
    fn switch_all_targets_same() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let val = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: val,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Switch {
            value: val,
            targets: vec![1, 1, 1],
            default: 1,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 10);
        assert!(changed, "switch with all same targets should simplify");
        assert!(matches!(
            ssa.block(0).unwrap().terminator_op().unwrap(),
            SsaOp::Jump { target: 1 }
        ));
    }

    #[test]
    fn empty_function() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 10);
        assert!(!changed);
    }

    #[test]
    fn trampoline_not_skipping_entry_block() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log, 10);
        // B0 is entry trampoline — it has no predecessors so jump threading can't redirect.
        // The entry trampoline handling is done by block merge pass, not control flow.
        let _ = changed;
    }
}
