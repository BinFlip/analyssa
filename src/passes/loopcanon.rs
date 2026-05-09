//! Loop canonicalization pass — transforms loops into well-formed
//! canonical form with single preheaders and single latches.
//!
//! # Transformations
//!
//! 1. **Preheader insertion**: When a loop header has multiple non-loop
//!    predecessors, a fresh preheader block is created. All non-loop
//!    predecessors are redirected to it, and its terminator is a `Jump`
//!    to the header. Phi nodes in the header are updated: multiple non-loop
//!    operands are merged through a new phi in the preheader; a single
//!    non-loop operand is forwarded directly.
//!
//! 2. **Latch unification**: When a loop has multiple back-edges (latches),
//!    a single unified latch block is created. All original latches are
//!    redirected to it, and its terminator is a `Jump` to the header.
//!    Phi nodes in the header are rewritten: multiple latch operands are
//!    merged through a new phi in the unified latch.
//!
//! # Impact
//!
//! Canonical loop structure is required by subsequent analyses:
//! - LICM needs a single preheader to insert hoisted code.
//! - Induction variable detection assumes a single latch.
//! - Loop interchange and unrolling benefit from canonical form.
//!
//! # Algorithm
//!
//! Repeatedly analyzes loop structure via [`SsaFunction::analyze_loops`],
//! processing loops innermost-first. For each non-canonical loop, either
//! inserts a preheader or unifies latches (one transformation per iteration
//! to keep phi management simple). After all loops are canonical,
//! [`SsaFunction::canonicalize`] is called.

use std::collections::HashMap;

use crate::{
    analysis::{loop_analyzer::SsaLoopAnalysis, loops::LoopInfo},
    events::{EventKind, EventListener},
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Target,
};

/// Run loop canonicalization on `ssa`.
///
/// Inserts missing preheaders and unifies multiple latches so every loop
/// has well-formed canonical structure. After convergence,
/// [`SsaFunction::canonicalize`] is called on the function.
///
/// # Arguments
///
/// * `ssa` — The SSA function to canonicalize in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::ControlFlowRestructured`] events.
///
/// # Returns
///
/// `true` if any preheader was inserted or any latch unified.
pub fn run<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    if ssa.block_count() < 2 {
        return false;
    }

    let modified = canonicalize_loops(ssa, method, events);
    if modified > 0 {
        ssa.canonicalize();
        true
    } else {
        false
    }
}

fn canonicalize_loops<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut total_modified: usize = 0;

    loop {
        let forest = ssa.analyze_loops();
        if forest.is_empty() {
            break;
        }

        let mut modified_this_iteration: usize = 0;

        for loop_info in forest.by_depth_descending() {
            if !loop_info.has_preheader() {
                let non_loop_preds = get_non_loop_predecessors(ssa, loop_info);
                if non_loop_preds.len() > 1 {
                    insert_preheader(ssa, loop_info, &non_loop_preds, method, events);
                    modified_this_iteration = modified_this_iteration.saturating_add(1);
                    break;
                }
            }
            if !loop_info.has_single_latch() && loop_info.latches.len() > 1 {
                unify_latches(ssa, loop_info, method, events);
                modified_this_iteration = modified_this_iteration.saturating_add(1);
                break;
            }
        }

        total_modified = total_modified.saturating_add(modified_this_iteration);
        if modified_this_iteration == 0 {
            break;
        }
    }

    total_modified
}

fn get_non_loop_predecessors<T: Target>(ssa: &SsaFunction<T>, loop_info: &LoopInfo) -> Vec<usize> {
    let header_idx = loop_info.header.index();
    let mut non_loop_preds = Vec::new();
    for (block_idx, block) in ssa.iter_blocks() {
        if let Some(op) = block.terminator_op() {
            let targets = get_targets(op);
            if targets.contains(&header_idx) && !loop_info.body.contains(block_idx) {
                non_loop_preds.push(block_idx);
            }
        }
    }
    non_loop_preds
}

/// Extract the set of target block indices from a terminator operation.
fn get_targets<T: Target>(op: &SsaOp<T>) -> Vec<usize> {
    match op {
        SsaOp::Jump { target } | SsaOp::Leave { target } => vec![*target],
        SsaOp::Branch {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        SsaOp::Switch {
            targets, default, ..
        } => {
            let mut all = targets.clone();
            all.push(*default);
            all
        }
        _ => vec![],
    }
}

fn insert_preheader<T, L>(
    ssa: &mut SsaFunction<T>,
    loop_info: &LoopInfo,
    non_loop_preds: &[usize],
    method: &T::MethodRef,
    events: &L,
) where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let header_idx = loop_info.header.index();
    let preheader_idx = ssa.block_count();

    let mut preheader: SsaBlock<T> = SsaBlock::new(preheader_idx);
    preheader.add_instruction(SsaInstruction::synthetic(SsaOp::Jump {
        target: header_idx,
    }));

    let phi_info: Vec<(VariableOrigin, Vec<PhiOperand>)> = ssa
        .block(header_idx)
        .map(|header| {
            header
                .phi_nodes()
                .iter()
                .filter_map(|phi| {
                    let non_loop_operands: Vec<_> = phi
                        .operands()
                        .iter()
                        .filter(|op| non_loop_preds.contains(&op.predecessor()))
                        .copied()
                        .collect();
                    if non_loop_operands.len() > 1 {
                        Some((phi.origin(), non_loop_operands))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    for (origin, operands) in &phi_info {
        let new_var = ssa.create_variable_for_origin(*origin, 0, DefSite::phi(preheader_idx));
        let mut preheader_phi = PhiNode::new(new_var, *origin);
        for op in operands {
            preheader_phi.add_operand(*op);
        }
        preheader.phi_nodes_mut().push(preheader_phi);
    }

    ssa.add_block(preheader);

    for &pred_idx in non_loop_preds {
        redirect_targets(ssa, pred_idx, header_idx, preheader_idx);
    }

    let preheader_phi_map: HashMap<VariableOrigin, SsaVarId> = ssa
        .block(preheader_idx)
        .map(|b| {
            b.phi_nodes()
                .iter()
                .map(|p| (p.origin(), p.result()))
                .collect()
        })
        .unwrap_or_default();

    if let Some(header) = ssa.block_mut(header_idx) {
        for phi in header.phi_nodes_mut() {
            let origin = phi.origin();
            let operands = phi.operands_mut();
            let mut loop_operands: Vec<PhiOperand> = Vec::new();
            let mut non_loop_values: Vec<PhiOperand> = Vec::new();

            for op in operands.drain(..) {
                if non_loop_preds.contains(&op.predecessor()) {
                    non_loop_values.push(op);
                } else {
                    loop_operands.push(op);
                }
            }

            operands.extend(loop_operands);

            if !non_loop_values.is_empty() {
                if let [single] = non_loop_values.as_slice() {
                    operands.push(PhiOperand::new(single.value(), preheader_idx));
                } else if let Some(&preheader_var) = preheader_phi_map.get(&origin) {
                    operands.push(PhiOperand::new(preheader_var, preheader_idx));
                }
            }
        }
    }

    let event = crate::events::Event {
        kind: EventKind::ControlFlowRestructured,
        method: Some(method.clone()),
        location: Some(preheader_idx),
        message: format!("Inserted preheader B{preheader_idx} for loop at B{header_idx}"),
        pass: None,
    };
    events.push(event);
}

fn unify_latches<T, L>(
    ssa: &mut SsaFunction<T>,
    loop_info: &LoopInfo,
    method: &T::MethodRef,
    events: &L,
) where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let header_idx = loop_info.header.index();
    let latches: Vec<usize> = loop_info.latches.iter().map(|n| n.index()).collect();
    let unified_latch_idx = ssa.block_count();

    let mut unified_latch: SsaBlock<T> = SsaBlock::new(unified_latch_idx);
    unified_latch.add_instruction(SsaInstruction::synthetic(SsaOp::Jump {
        target: header_idx,
    }));

    let mut latch_phi_vars: HashMap<VariableOrigin, SsaVarId> = HashMap::new();
    let phi_info: Vec<(VariableOrigin, Vec<PhiOperand>)> = ssa
        .block(header_idx)
        .map(|header| {
            header
                .phi_nodes()
                .iter()
                .map(|phi| {
                    let latch_operands: Vec<_> = phi
                        .operands()
                        .iter()
                        .filter(|op| latches.contains(&op.predecessor()))
                        .copied()
                        .collect();
                    (phi.origin(), latch_operands)
                })
                .collect()
        })
        .unwrap_or_default();

    for (origin, latch_operands) in &phi_info {
        if latch_operands.len() > 1 {
            let new_var =
                ssa.create_variable_for_origin(*origin, 0, DefSite::phi(unified_latch_idx));
            let mut latch_phi = PhiNode::new(new_var, *origin);
            for op in latch_operands {
                latch_phi.add_operand(*op);
            }
            latch_phi_vars.insert(*origin, new_var);
            unified_latch.phi_nodes_mut().push(latch_phi);
        } else if let [single] = latch_operands.as_slice() {
            latch_phi_vars.insert(*origin, single.value());
        }
    }

    ssa.add_block(unified_latch);

    for &latch_idx in &latches {
        redirect_targets(ssa, latch_idx, header_idx, unified_latch_idx);
    }

    if let Some(header) = ssa.block_mut(header_idx) {
        for phi in header.phi_nodes_mut() {
            let origin = phi.origin();
            let operands = phi.operands_mut();
            operands.retain(|op| !latches.contains(&op.predecessor()));
            if let Some(&var) = latch_phi_vars.get(&origin) {
                operands.push(PhiOperand::new(var, unified_latch_idx));
            }
        }
    }

    let event = crate::events::Event {
        kind: EventKind::ControlFlowRestructured,
        method: Some(method.clone()),
        location: Some(unified_latch_idx),
        message: format!(
            "Unified {} latches into B{} for loop at B{}",
            latches.len(),
            unified_latch_idx,
            header_idx
        ),
        pass: None,
    };
    events.push(event);
}

fn redirect_targets<T: Target>(
    ssa: &mut SsaFunction<T>,
    block_idx: usize,
    old_target: usize,
    new_target: usize,
) {
    if let Some(block) = ssa.block_mut(block_idx) {
        if let Some(last) = block.instructions_mut().last_mut() {
            let new_op = match last.op() {
                SsaOp::Jump { target } if *target == old_target => {
                    Some(SsaOp::Jump { target: new_target })
                }
                SsaOp::Leave { target } if *target == old_target => {
                    Some(SsaOp::Leave { target: new_target })
                }
                SsaOp::Branch {
                    condition,
                    true_target,
                    false_target,
                } => {
                    let new_true = if *true_target == old_target {
                        new_target
                    } else {
                        *true_target
                    };
                    let new_false = if *false_target == old_target {
                        new_target
                    } else {
                        *false_target
                    };
                    if new_true != *true_target || new_false != *false_target {
                        Some(SsaOp::Branch {
                            condition: *condition,
                            true_target: new_true,
                            false_target: new_false,
                        })
                    } else {
                        None
                    }
                }
                SsaOp::Switch {
                    value,
                    targets,
                    default,
                } => {
                    let new_targets: Vec<_> = targets
                        .iter()
                        .map(|&t| if t == old_target { new_target } else { t })
                        .collect();
                    let new_default = if *default == old_target {
                        new_target
                    } else {
                        *default
                    };
                    if new_targets != *targets || new_default != *default {
                        Some(SsaOp::Switch {
                            value: *value,
                            targets: new_targets,
                            default: new_default,
                        })
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(new_op) = new_op {
                last.set_op(new_op);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::SsaVerifier,
        analysis::VerifyLevel,
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            phi::{PhiNode, PhiOperand},
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

    fn phi_local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::phi(block),
            MockType::I32,
        )
    }

    fn build_loop_with_multiple_latches() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 6);
        let init = local_at(&mut ssa, 0, 0, 0);
        let one = local_at(&mut ssa, 1, 0, 1);
        let limit = local_at(&mut ssa, 2, 0, 2);
        let i_phi = phi_local(&mut ssa, 3, 1);
        let cond = local_at(&mut ssa, 4, 1, 0);
        let next = local_at(&mut ssa, 5, 2, 0);
        let alt_next = local_at(&mut ssa, 6, 3, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: init,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: limit,
            value: ConstValue::I32(100),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(i_phi, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(init, 0));
        phi.add_operand(PhiOperand::new(next, 2));
        phi.add_operand(PhiOperand::new(alt_next, 3));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Clt {
            dest: cond,
            left: i_phi,
            right: limit,
            unsigned: false,
        }));
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 4,
        }));
        ssa.add_block(b1);

        // Latch 1 (B2)
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: next,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        // Latch 2 (B3)
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Add {
            dest: alt_next,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b3.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b3);

        // Exit (B4)
        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Return { value: Some(i_phi) }));
        ssa.add_block(b4);

        ssa.recompute_uses();
        ssa
    }

    #[test]
    fn multiple_latches_unified() {
        let mut ssa = build_loop_with_multiple_latches();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;

        let original_blocks = ssa.block_count();
        let changed = run(&mut ssa, &method, &log);
        assert!(changed, "multiple latches should be unified");
        assert!(
            ssa.block_count() > original_blocks,
            "should add a unified latch block"
        );

        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
        assert!(errors.is_empty(), "verifier errors: {errors:?}");
    }

    #[test]
    fn single_latch_already_canonical() {
        let mut ssa = SsaFunction::new(0, 5);
        let init = local_at(&mut ssa, 0, 0, 0);
        let one = local_at(&mut ssa, 1, 0, 1);
        let limit = local_at(&mut ssa, 2, 0, 2);
        let i_phi = phi_local(&mut ssa, 3, 1);
        let cond = local_at(&mut ssa, 4, 1, 0);
        let next = local_at(&mut ssa, 5, 2, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: init,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: limit,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(i_phi, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(init, 0));
        phi.add_operand(PhiOperand::new(next, 2));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Clt {
            dest: cond,
            left: i_phi,
            right: limit,
            unsigned: false,
        }));
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: next,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Return { value: Some(i_phi) }));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log);
        assert!(!changed, "single latch should already be canonical");
    }

    #[test]
    fn no_loops_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log);
        assert!(!changed);
    }

    #[test]
    fn single_block_no_loop() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        b0.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log);
        assert!(!changed, "single block cannot have a loop");
    }

    #[test]
    fn idempotent_after_canonicalization() {
        let mut ssa = build_loop_with_multiple_latches();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;

        let _ = run(&mut ssa, &method, &log);
        let after_first = ssa.block_count();

        // Second run should be idempotent
        let changed = run(&mut ssa, &method, &log);
        assert!(!changed, "second canonicalization should be a no-op");
        assert_eq!(ssa.block_count(), after_first);
    }

    #[test]
    fn loop_with_preheader_insertion_and_latch_unification() {
        // Build a loop with BOTH: no preheader (multiple entries) AND multiple latches
        let mut ssa = SsaFunction::new(0, 6);
        let init_a = local_at(&mut ssa, 0, 0, 0);
        let init_b = local_at(&mut ssa, 1, 1, 0);
        let one = local_at(&mut ssa, 2, 0, 1);
        let limit = local_at(&mut ssa, 3, 0, 2);
        let i_phi = phi_local(&mut ssa, 4, 2);
        let cond = local_at(&mut ssa, 5, 2, 0);
        let next1 = local_at(&mut ssa, 6, 3, 0);
        let next2 = local_at(&mut ssa, 7, 4, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: init_a,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: limit,
            value: ConstValue::I32(50),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: init_b,
            value: ConstValue::I32(10),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(i_phi, VariableOrigin::Local(4));
        phi.add_operand(PhiOperand::new(init_a, 0));
        phi.add_operand(PhiOperand::new(init_b, 1));
        phi.add_operand(PhiOperand::new(next1, 3));
        phi.add_operand(PhiOperand::new(next2, 4));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Clt {
            dest: cond,
            left: i_phi,
            right: limit,
            unsigned: false,
        }));
        b2.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 3,
            false_target: 5,
        }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Add {
            dest: next1,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b3.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Add {
            dest: next2,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b4.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b4);

        let mut b5 = SsaBlock::new(5);
        b5.add_instruction(instr(SsaOp::Return { value: Some(i_phi) }));
        ssa.add_block(b5);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run(&mut ssa, &method, &log);
        assert!(changed, "loop with both issues should be canonicalized");

        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
        assert!(errors.is_empty(), "verifier errors: {errors:?}");
    }
}
