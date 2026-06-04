//! Jump threading pass — redirects branches through predecessors when the
//! branch condition is known from the incoming path.
//!
//! # Algorithm
//!
//! For each block ending with a `Branch { condition, true_target, false_target }`:
//!
//! 1. **Evaluate predecessor**: use [`SsaEvaluator`] to compute concrete
//!    values for all variables at the end of the predecessor block.
//! 2. **Resolve phis**: set the evaluator's predecessor context so phi
//!    nodes in the branch block are resolved to the value coming from that
//!    specific predecessor.
//! 3. **Evaluate condition**: if the condition variable resolves to a
//!    concrete constant (via [`SsaEvaluator::get_concrete`] or
//!    [`SsaEvaluator::resolve_with_trace`]), the branch target is known.
//! 4. **Redirect**: change the predecessor's terminator to jump directly
//!    to the proven target, bypassing the branch. If the predecessor had a
//!    `Branch`, it becomes a `Jump`. If it had a `Jump` or `Leave` to the
//!    branch block, the target is updated.
//!
//! # Scope
//!
//! This pass handles only `Branch` terminators at the threading target.
//! `Jump` and `Switch` terminators are not threaded. Trampoline blocks
//! (blocks containing only a `Jump`) are handled by the block merging
//! and control flow simplification passes.

use std::collections::BTreeSet;

use crate::{
    analysis::{cfg::SsaCfg, defuse::DefUseIndex, evaluator::SsaEvaluator},
    events::{EventKind, EventListener},
    ir::{
        function::{SsaEditOptions, SsaEditor, SsaFunction},
        ops::SsaOp,
        value::ConstValue,
        variable::SsaVarId,
    },
    pointer::PointerSize,
    target::Target,
};

/// Run jump threading on `ssa`.
///
/// For each predecessor of each branch block, evaluates the path from the
/// predecessor through the branch using [`SsaEvaluator`] and redirects
/// the predecessor's terminator when the condition is provably constant.
///
/// # Arguments
///
/// * `ssa` — The SSA function to transform in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::ControlFlowRestructured`]
///   and [`EventKind::BranchSimplified`] events.
/// * `ptr_size` — Host pointer width, passed to [`SsaEvaluator`].
///
/// # Returns
///
/// `true` if any predecessor was rerouted.
pub fn run<T, L>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
    ptr_size: PointerSize,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    if ssa.is_empty() {
        return false;
    }

    let cfg = SsaCfg::from_ssa(ssa);
    // One def-use index for the whole collection phase (ssa is not mutated until
    // the edit below), so the per-branch safety check is O(local defs) instead
    // of an O(N) full-function rescan.
    let defuse = DefUseIndex::build(ssa);

    // Collect threading opportunities first to avoid borrow conflicts.
    let mut threadings: Vec<(usize, usize, usize)> = Vec::new();

    for (block_idx, block) in ssa.iter_blocks() {
        let Some(SsaOp::Branch {
            condition,
            true_target,
            false_target,
        }) = block.terminator_op()
        else {
            continue;
        };

        if !is_safe_to_bypass_branch_block(ssa, block_idx, &defuse) {
            continue;
        }

        for pred_idx in cfg.block_predecessors(block_idx) {
            if let Some(target) = try_thread(
                ssa,
                *pred_idx,
                block_idx,
                *condition,
                *true_target,
                *false_target,
                ptr_size,
            ) {
                let pred_target = ssa.block(*pred_idx).and_then(|b| {
                    b.terminator_op().and_then(|op| match op {
                        SsaOp::Jump { target } | SsaOp::Leave { target } => Some(*target),
                        _ => None,
                    })
                });
                if pred_target != Some(target) {
                    threadings.push((*pred_idx, block_idx, target));
                }
            }
        }
    }

    let mut changed = false;
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        for (pred_block, branch_block, new_target) in threadings {
            if apply_threading(editor, pred_block, branch_block, new_target, method, events) {
                changed = true;
            }
        }
        Ok(())
    });
    if result.is_err() {
        return false;
    }
    changed
}

fn is_safe_to_bypass_branch_block<T: Target>(
    ssa: &SsaFunction<T>,
    branch_block: usize,
    defuse: &DefUseIndex<T>,
) -> bool {
    let Some(block) = ssa.block(branch_block) else {
        return false;
    };
    let Some(terminator_idx) = block.instructions().len().checked_sub(1) else {
        return false;
    };

    if !matches!(block.terminator_op(), Some(SsaOp::Branch { .. })) {
        return false;
    }

    let mut local_defs = BTreeSet::new();
    for phi in block.phi_nodes() {
        local_defs.insert(phi.result());
    }

    for (instr_idx, instr) in block.instructions().iter().enumerate() {
        if instr_idx == terminator_idx {
            continue;
        }
        if !matches!(instr.op(), SsaOp::Nop) {
            return false;
        }
        local_defs.extend(instr.defs());
    }

    if local_defs.is_empty() {
        return true;
    }

    // A local def may be referenced only by this block's own branch terminator.
    // Any phi-operand use, or any instruction use elsewhere, makes bypass unsafe.
    // Querying the prebuilt def-use index touches only each local def's own use
    // sites rather than rescanning the whole function per branch block.
    for local in &local_defs {
        if let Some(uses) = defuse.uses_of(*local) {
            for use_site in uses {
                let is_branch_terminator_use = !use_site.is_phi_operand
                    && use_site.block == branch_block
                    && use_site.instruction == terminator_idx;
                if !is_branch_terminator_use {
                    return false;
                }
            }
        }
    }

    true
}

fn try_thread<T: Target>(
    ssa: &SsaFunction<T>,
    pred_block: usize,
    branch_block: usize,
    condition: SsaVarId,
    true_target: usize,
    false_target: usize,
    ptr_size: PointerSize,
) -> Option<usize> {
    let mut eval = SsaEvaluator::new(ssa, ptr_size);

    eval.evaluate_block(pred_block);
    eval.set_predecessor(Some(pred_block));
    eval.evaluate_phis(branch_block);

    let cond_value = eval
        .get_concrete(condition)
        .and_then(ConstValue::as_i64)
        .or_else(|| {
            eval.resolve_with_trace(condition, 10)
                .and_then(|e| e.as_i64())
        })?;

    let _ = branch_block;
    if cond_value != 0 {
        Some(true_target)
    } else {
        Some(false_target)
    }
}

fn apply_threading<T, L>(
    editor: &mut SsaEditor<T>,
    pred_block: usize,
    _branch_block: usize,
    new_target: usize,
    method: &T::MethodRef,
    events: &L,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let Some(op) = editor
        .function()
        .block(pred_block)
        .and_then(|block| block.instructions().last())
        .map(|instr| instr.op().clone())
    else {
        return false;
    };

    match op {
        SsaOp::Jump { target } if target != new_target => {
            if editor
                .replace_terminator_op(pred_block, SsaOp::Jump { target: new_target })
                .is_err()
            {
                return false;
            }
            push(
                events,
                EventKind::ControlFlowRestructured,
                method,
                pred_block,
                format!("jump threaded: B{pred_block} now jumps to B{new_target} (was B{target})"),
            );
            true
        }
        SsaOp::Branch {
            condition,
            true_target,
            false_target,
        } => {
            let old_target = if new_target == true_target {
                false_target
            } else {
                true_target
            };
            if editor
                .replace_terminator_op(pred_block, SsaOp::Jump { target: new_target })
                .is_err()
            {
                return false;
            }
            push(events, EventKind::BranchSimplified, method, pred_block, format!(
                "branch threaded: B{pred_block} condition on {condition:?} resolved to B{new_target} (eliminated B{old_target})"
            ));
            true
        }
        SsaOp::Leave { target } if target != new_target => {
            if editor
                .replace_terminator_op(pred_block, SsaOp::Leave { target: new_target })
                .is_err()
            {
                return false;
            }
            push(
                events,
                EventKind::ControlFlowRestructured,
                method,
                pred_block,
                format!(
                    "leave threaded: B{pred_block} now leaves to B{new_target} (was B{target})"
                ),
            );
            true
        }
        _ => false,
    }
}

fn push<T, L>(events: &L, kind: EventKind, method: &T::MethodRef, location: usize, message: String)
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let event = crate::events::Event {
        kind,
        method: Some(method.clone()),
        location: Some(location),
        message,
        pass: None,
    };
    events.push(event);
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
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{run_mock_pass_boundary, MockTarget, MockType},
        PointerSize,
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
    fn thread_constant_condition_to_true_branch() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let true_val = local_at(&mut ssa, 0, 0, 0);
        let cond = local_at(&mut ssa, 1, 0, 1);

        // B0 is a predecessor that jumps to B1 (where the branch lives)
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: true_val,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        // B1 has the branch — it has a predecessor (B0) so threading can work
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        // B2 true target
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return {
            value: Some(true_val),
        }));
        ssa.add_block(b2);

        // B3 false target
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "constant branch threading", |ssa| {
            run(ssa, &method, &log, PointerSize::Bit64)
        });
        assert!(changed, "constant true condition should be threaded");
        assert!(
            log.has(EventKind::BranchSimplified) || log.has(EventKind::ControlFlowRestructured)
        );
    }

    #[test]
    fn thread_with_phi_value_resolution() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let true_cond = local_at(&mut ssa, 0, 0, 0);
        let false_cond = local_at(&mut ssa, 1, 1, 0);
        let merged_cond =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: true_cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: false_cond,
            value: ConstValue::I32(0),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(merged_cond, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(true_cond, 0));
        phi.add_operand(PhiOperand::new(false_cond, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Branch {
            condition: merged_cond,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b4);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "phi value branch threading", |ssa| {
            run(ssa, &method, &log, PointerSize::Bit64)
        });
        assert!(changed, "phi-based threading should work");
    }

    #[test]
    fn no_threading_when_bypassed_phi_value_is_live_after_branch() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let true_cond = local_at(&mut ssa, 0, 0, 0);
        let false_cond = local_at(&mut ssa, 1, 1, 0);
        let merged_cond =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: true_cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: false_cond,
            value: ConstValue::I32(0),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(merged_cond, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(true_cond, 0));
        phi.add_operand(PhiOperand::new(false_cond, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Branch {
            condition: merged_cond,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Return {
            value: Some(merged_cond),
        }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b4);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "live phi branch threading", |ssa| {
            run(ssa, &method, &log, PointerSize::Bit64)
        });
        assert!(
            !changed,
            "threading must not bypass a phi value used after the branch"
        );
    }

    #[test]
    fn no_threading_when_all_unknown() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let cond = local_at(&mut ssa, 0, 0, 0);

        let mut b0 = SsaBlock::new(0);
        // cond comes from LoadArg, not a constant — can't thread
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: cond,
            arg_index: 0,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "unknown condition threading", |ssa| {
            run(ssa, &method, &log, PointerSize::Bit64)
        });
        assert!(
            !changed,
            "no threading should occur when condition cannot be resolved"
        );
    }

    #[test]
    fn empty_function_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty threading", |ssa| {
            run(ssa, &method, &log, PointerSize::Bit64)
        });
        assert!(!changed);
    }

    #[test]
    fn threading_with_copy_chain() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let src = local_at(&mut ssa, 0, 0, 0);
        let mid = local_at(&mut ssa, 1, 0, 1);
        let cond = local_at(&mut ssa, 2, 0, 2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: src,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Copy { dest: mid, src }));
        b0.add_instruction(instr(SsaOp::Copy {
            dest: cond,
            src: mid,
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "copy-chain branch threading", |ssa| {
            run(ssa, &method, &log, PointerSize::Bit64)
        });
        // The evaluator should trace through Copy chain to find constant
        assert!(changed, "copy-chain condition should be threaded");
    }

    #[test]
    fn no_branch_terminator_no_threading() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "non-branch threading", |ssa| {
            run(ssa, &method, &log, PointerSize::Bit64)
        });
        assert!(!changed, "no branch means no threading");
    }

    #[test]
    fn jump_not_branch_not_threaded() {
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
        let changed = run_mock_pass_boundary(&mut ssa, "jump terminator threading", |ssa| {
            run(ssa, &method, &log, PointerSize::Bit64)
        });
        assert!(!changed, "Jump should not be threaded by this pass");
    }
}
