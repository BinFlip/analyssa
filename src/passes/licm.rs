//! Loop Invariant Code Motion (LICM) pass — moves loop-invariant
//! computations into the loop preheader.
//!
//! # Algorithm
//!
//! 1. Analyze the loop structure using [`LoopAnalyzer`].
//! 2. Process loops innermost-first so hoisted values are available to
//!    outer loops in subsequent passes.
//! 3. For each loop with a valid preheader:
//!    a. Find invariant instructions — those whose operands are all
//!    defined outside the loop or by other invariants, excluding
//!    header phi defs.
//!    b. Filter hoistable candidates: reject instructions that feed a phi
//!    operand on an intra-loop back-edge (would collapse per-edge
//!    attribution), and reject instructions whose removal would leave
//!    a trampoline block whose successor has phis.
//!    c. Insert invariants just before the preheader's terminator,
//!    preserving original definition order.
//!    d. Replace original instructions with `Nop`.
//!    e. If a source block is now a trampoline, redirect successor phi
//!    operands from the source block to the preheader.
//!
//! # Conservative Guards
//!
//! - Requires a loop preheader that is a CFG predecessor of the header.
//! - Skips loops whose header terminator is a `Switch` (CFF dispatcher
//!   pattern — hoisting corrupts SSA rebuild).
//! - Rejects instructions whose result transitively feeds a phi on an
//!   intra-loop edge (would erase per-edge value distinctions).
//! - Avoids creating trampoline blocks whose successor has phis (phi
//!   stability across subsequent block merging).

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    analysis::{loop_analyzer::LoopAnalyzer, loops::LoopInfo},
    bitset::BitSet,
    events::{EventKind, EventListener},
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        instruction::SsaInstruction,
        ops::SsaOp,
        variable::SsaVarId,
    },
    target::Target,
};

/// Run Loop Invariant Code Motion on `ssa`.
///
/// Hoists pure, invariant instructions from loop bodies into their
/// preheaders. Processes loops innermost-first.
///
/// # Arguments
///
/// * `ssa` — The SSA function to optimize in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::InstructionRemoved`] events.
///
/// # Returns
///
/// `true` if any instruction was hoisted.
pub fn run<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let forest = LoopAnalyzer::new(ssa).analyze();
    if forest.is_empty() {
        return false;
    }

    let mut total_hoisted: usize = 0;

    for loop_info in forest.by_depth_descending() {
        let Some(preheader) = loop_info.preheader else {
            continue;
        };

        let header_idx = loop_info.header.index();
        let preheader_is_pred = ssa
            .block(preheader.index())
            .map(|b| {
                b.instructions()
                    .last()
                    .map(|i| i.op().has_successor(header_idx))
                    .unwrap_or(false)
            })
            .unwrap_or(false);
        if !preheader_is_pred {
            continue;
        }

        let header_has_switch = ssa
            .block(header_idx)
            .and_then(|b| b.terminator_op())
            .is_some_and(|op| matches!(op, SsaOp::Switch { .. }));
        if header_has_switch {
            continue;
        }

        let invariants = find_loop_invariants(ssa, loop_info);
        if invariants.is_empty() {
            continue;
        }

        let phi_back_edge_operands = phi_back_edge_operands(ssa, loop_info);
        let mut hoistable: Vec<_> = invariants
            .into_iter()
            .filter(|(block_idx, instr_idx)| {
                can_hoist(
                    ssa,
                    loop_info,
                    &phi_back_edge_operands,
                    *block_idx,
                    *instr_idx,
                )
            })
            .collect();

        let preheader_idx = preheader.index();
        let insert_base = if let Some(preheader_block) = ssa.block(preheader_idx) {
            let instrs = preheader_block.instructions();
            if instrs.is_empty() {
                0
            } else if instrs.last().is_some_and(SsaInstruction::is_terminator) {
                instrs.len().saturating_sub(1)
            } else {
                instrs.len()
            }
        } else {
            0
        };

        // Convergent filter: drop instructions whose operands are not already
        // defined outside the loop. Dependent invariants can be hoisted by a
        // later LICM invocation after their producer has been moved to the
        // preheader. This avoids inserting a hoisted use before a hoisted def.
        let mut outside_defs = BitSet::new(ssa.var_id_capacity());
        for v in ssa.variables() {
            let site = v.def_site();
            let preheader_def_after_insert = site.block == preheader_idx
                && site
                    .instruction
                    .is_some_and(|instr_idx| instr_idx >= insert_base);
            if !loop_info.body.contains(site.block) && !preheader_def_after_insert {
                outside_defs.insert(v.id().index());
            }
        }

        loop {
            let before = hoistable.len();
            hoistable.retain(|(block_idx, instr_idx)| {
                let Some(block) = ssa.block(*block_idx) else {
                    return false;
                };
                let Some(instr) = block.instruction(*instr_idx) else {
                    return false;
                };
                let def_used_before_insert = instr.op().defs().any(|def| {
                    ssa.variable(def).is_some_and(|var| {
                        var.uses().iter().any(|site| {
                            site.block == preheader_idx
                                && !site.is_phi_operand
                                && site.instruction < insert_base
                        })
                    })
                });
                if def_used_before_insert {
                    return false;
                }
                let mut operands_are_available = true;
                instr.op().for_each_use(|operand| {
                    operands_are_available &= outside_defs.contains(operand.index());
                });
                operands_are_available
            });
            if hoistable.len() == before {
                break;
            }
        }

        // Skip blocks that would become trampolines whose successor has phis.
        {
            let mut hoist_count_per_block: HashMap<usize, usize> = HashMap::new();
            for (block_idx, _) in &hoistable {
                let entry = hoist_count_per_block.entry(*block_idx).or_insert(0);
                *entry = entry.saturating_add(1);
            }
            let mut trampoline_blocks = BitSet::new(ssa.block_count());
            for (&block_idx, &hoist_count) in &hoist_count_per_block {
                if let Some(block) = ssa.block(block_idx) {
                    let non_term = block
                        .instructions()
                        .iter()
                        .filter(|i| !i.is_terminator() && !matches!(i.op(), SsaOp::Nop))
                        .count();
                    if hoist_count >= non_term {
                        if let Some(term) = block.terminator_op() {
                            term.for_each_successor(|succ| {
                                if let Some(succ_block) = ssa.block(succ) {
                                    if !succ_block.phi_nodes().is_empty() {
                                        trampoline_blocks.insert(block_idx);
                                    }
                                }
                            });
                        }
                    }
                }
            }
            if !trampoline_blocks.is_empty() {
                hoistable.retain(|(block_idx, _)| !trampoline_blocks.contains(*block_idx));
            }
        }

        if hoistable.is_empty() {
            continue;
        }

        let mut to_hoist: Vec<(usize, usize, SsaOp<T>)> = Vec::new();
        for (block_idx, instr_idx) in &hoistable {
            if let Some(block) = ssa.block(*block_idx) {
                if let Some(instr) = block.instruction(*instr_idx) {
                    to_hoist.push((*block_idx, *instr_idx, instr.op().clone()));
                }
            }
        }

        // Preserve original definition order so dependencies stay valid.
        to_hoist.sort_by_key(|(block_idx, instr_idx, _)| (*block_idx, *instr_idx));

        let mut hoisted_this_loop = 0usize;
        let result = ssa.edit(
            SsaEditOptions::new()
                .with_verify(true)
                .with_rollback(SsaRollbackPolicy::OnFailure),
            |editor| {
                let mut hoisted_from = BitSet::new(editor.function().block_count());

                for (i, (block_idx, instr_idx, op)) in to_hoist.iter().enumerate() {
                    hoisted_from.insert(*block_idx);

                    editor.insert_instruction(
                        preheader_idx,
                        insert_base.saturating_add(i),
                        SsaInstruction::synthetic(op.clone()),
                    )?;
                    editor.nop_instruction(*block_idx, *instr_idx)?;
                    hoisted_this_loop = hoisted_this_loop.saturating_add(1);
                }

                // If a source block is now a trampoline, redirect successor phis
                // from the source block to the preheader where the hoisted defs live.
                for source_block in hoisted_from.iter() {
                    let is_trampoline = editor.function().block(source_block).is_some_and(|b| {
                        b.instructions()
                            .iter()
                            .all(|i| i.is_terminator() || matches!(i.op(), SsaOp::Nop))
                    });
                    if !is_trampoline {
                        continue;
                    }
                    let successors: Vec<usize> = editor
                        .function()
                        .block(source_block)
                        .map(|b| {
                            b.instructions()
                                .last()
                                .map(|i| i.op().successors())
                                .unwrap_or_default()
                        })
                        .unwrap_or_default();
                    for succ in successors {
                        editor.replace_phi_predecessor(succ, source_block, preheader_idx)?;
                    }
                }

                Ok(())
            },
        );
        if result.is_ok() {
            total_hoisted = total_hoisted.saturating_add(hoisted_this_loop);
        }
    }

    if total_hoisted > 0 {
        let event = crate::events::Event {
            kind: EventKind::InstructionRemoved,
            method: Some(method.clone()),
            location: Some(0),
            message: format!("LICM: hoisted {total_hoisted} loop-invariant instructions"),
            pass: None,
        };
        events.push(event);
    }

    total_hoisted > 0
}

fn find_loop_invariants<T: Target>(
    ssa: &SsaFunction<T>,
    loop_info: &LoopInfo,
) -> Vec<(usize, usize)> {
    let mut invariants: HashSet<(usize, usize)> = HashSet::new();
    let mut invariant_defs = BitSet::new(ssa.var_id_capacity());

    let mut header_phi_defs = BitSet::new(ssa.var_id_capacity());
    if let Some(header_block) = ssa.block(loop_info.header.index()) {
        for phi in header_block.phi_nodes() {
            header_phi_defs.insert(phi.result().index());
        }
    }

    let mut outside_defs = BitSet::new(ssa.var_id_capacity());
    for var in ssa.variables() {
        let def_site = var.def_site();
        if !loop_info.body.contains(def_site.block) {
            outside_defs.insert(var.id().index());
        }
    }

    // Worklist over loop-body instructions: seed with all of them, and when an
    // instruction becomes invariant, only re-examine the instructions that use
    // its definitions (via a scoped def→users map). This replaces the previous
    // `while changed { rescan whole loop body }` fixpoint, which was O(loop^2).
    let mut users: HashMap<SsaVarId, Vec<(usize, usize)>> = HashMap::new();
    let mut worklist: VecDeque<(usize, usize)> = VecDeque::new();
    let mut queued: HashSet<(usize, usize)> = HashSet::new();
    for block_idx in loop_info.body.iter() {
        if let Some(block) = ssa.block(block_idx) {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if instr.is_terminator() || matches!(instr.op(), SsaOp::Nop) {
                    continue;
                }
                instr.op().for_each_use(|u| {
                    users.entry(u).or_default().push((block_idx, instr_idx));
                });
                worklist.push_back((block_idx, instr_idx));
                queued.insert((block_idx, instr_idx));
            }
        }
    }

    while let Some((block_idx, instr_idx)) = worklist.pop_front() {
        queued.remove(&(block_idx, instr_idx));
        if invariants.contains(&(block_idx, instr_idx)) {
            continue;
        }
        let Some(instr) = ssa
            .block(block_idx)
            .and_then(|b| b.instructions().get(instr_idx))
        else {
            continue;
        };
        if is_instruction_invariant(instr, &outside_defs, &invariant_defs, &header_phi_defs) {
            invariants.insert((block_idx, instr_idx));
            for def in instr.defs() {
                invariant_defs.insert(def.index());
                if let Some(dependents) = users.get(&def) {
                    for &(ub, ui) in dependents {
                        if !invariants.contains(&(ub, ui)) && queued.insert((ub, ui)) {
                            worklist.push_back((ub, ui));
                        }
                    }
                }
            }
        }
    }

    invariants.into_iter().collect()
}

fn is_instruction_invariant<T: Target>(
    instr: &SsaInstruction<T>,
    outside_defs: &BitSet,
    invariant_defs: &BitSet,
    header_phi_defs: &BitSet,
) -> bool {
    let mut invariant = true;
    instr.op().for_each_use(|operand| {
        if header_phi_defs.contains(operand.index()) {
            invariant = false;
        }
        if !outside_defs.contains(operand.index()) && !invariant_defs.contains(operand.index()) {
            invariant = false;
        }
    });
    invariant
}

fn can_hoist<T: Target>(
    ssa: &SsaFunction<T>,
    loop_info: &LoopInfo,
    phi_back_edge_operands: &BitSet,
    block_idx: usize,
    instr_idx: usize,
) -> bool {
    let Some(block) = ssa.block(block_idx) else {
        return false;
    };
    let Some(instr) = block.instruction(instr_idx) else {
        return false;
    };
    if !instr.has_def() {
        return false;
    }
    if !instr.op().effects().is_pure() {
        return false;
    }
    if loop_info.preheader.is_none() {
        return false;
    }
    for dest in instr.defs() {
        if feeds_phi_back_edge(ssa, loop_info, phi_back_edge_operands, dest) {
            return false;
        }
    }
    true
}

fn phi_back_edge_operands<T: Target>(ssa: &SsaFunction<T>, loop_info: &LoopInfo) -> BitSet {
    let mut operands = BitSet::new(ssa.var_id_capacity());
    for phi_block_idx in loop_info.body.iter() {
        let Some(phi_block) = ssa.block(phi_block_idx) else {
            continue;
        };
        for phi in phi_block.phi_nodes() {
            for operand in phi.operands() {
                if loop_info.body.contains(operand.predecessor()) {
                    operands.insert(operand.value().index());
                }
            }
        }
    }
    operands
}

fn feeds_phi_back_edge<T: Target>(
    ssa: &SsaFunction<T>,
    loop_info: &LoopInfo,
    phi_back_edge_operands: &BitSet,
    var: SsaVarId,
) -> bool {
    let mut worklist: VecDeque<SsaVarId> = VecDeque::new();
    let mut visited = BitSet::new(ssa.var_id_capacity());
    worklist.push_back(var);
    visited.insert(var.index());

    while let Some(current) = worklist.pop_front() {
        if phi_back_edge_operands.contains(current.index()) {
            return true;
        }

        let Some(current_var) = ssa.variable(current) else {
            continue;
        };
        for site in current_var.uses() {
            if site.is_phi_operand || !loop_info.body.contains(site.block) {
                continue;
            }
            let Some(instr) = ssa
                .block(site.block)
                .and_then(|block| block.instruction(site.instruction))
            else {
                continue;
            };
            for dest in instr.defs() {
                if visited.insert(dest.index()) {
                    worklist.push_back(dest);
                }
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

    /// Builds: preheader (B0) → header (B1: branch cond → body or exit)
    /// Body (B2) contains invariant expression, latches back to header
    fn build_simple_loop() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 5);
        let base = local_at(&mut ssa, 0, 0, 0);
        let one = local_at(&mut ssa, 1, 0, 1);
        let cond = local_at(&mut ssa, 2, 0, 2);
        let invariant = ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(2, 0),
            MockType::I32,
        );
        let iv = local_at(&mut ssa, 4, 1, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: base,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: iv,
            value: ConstValue::I32(0),
        }));
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: invariant,
            left: base,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b3);

        ssa.recompute_uses();
        ssa
    }

    #[test]
    fn licm_hoists_invariant_to_preheader() {
        let mut ssa = build_simple_loop();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "simple LICM hoist", |ssa| run(ssa, &method, &log));
        assert!(changed, "LICM should hoist invariant expression");
        assert!(log.has(EventKind::InstructionRemoved));
    }

    #[test]
    fn no_loops_nothing_hoisted() {
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
        let changed =
            run_mock_pass_boundary(&mut ssa, "no-loop LICM", |ssa| run(ssa, &method, &log));
        assert!(!changed, "no loops should mean nothing to hoist");
    }

    #[test]
    fn invariant_with_loop_variant_operand_not_hoisted() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 5);
        let base = local_at(&mut ssa, 0, 0, 0);
        let cond = local_at(&mut ssa, 1, 0, 1);
        let iv = local_at(&mut ssa, 2, 0, 2);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(1), MockType::I32);
        let result = local_at(&mut ssa, 4, 2, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: base,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: iv,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        // phi_var is loop-variant (defined by phi in header, fed by back-edge)
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(iv, 0));
        phi.add_operand(PhiOperand::new(result, 2));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        // result depends on phi_var which is loop-variant -> not invariant
        b2.add_instruction(instr(SsaOp::Add {
            dest: result,
            left: base,
            right: phi_var,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "variant-operand LICM", |ssa| {
            run(ssa, &method, &log)
        });
        // phi_var is defined by header phi which is part of loop body, so Add should NOT be hoisted
        assert!(
            !changed,
            "loop-variant operand (header phi) should prevent hoisting"
        );
    }

    #[test]
    fn empty_function_is_noop() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty LICM", |ssa| run(ssa, &method, &log));
        assert!(!changed);
    }

    #[test]
    fn loop_without_preheader_not_hoisted() {
        // A self-loop where the only block branches to itself — no preheader
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let cond = local_at(&mut ssa, 1, 0, 1);
        let sum = local_at(&mut ssa, 2, 0, 2);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Add {
            dest: sum,
            left: v0,
            right: v0,
            flags: None,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 0,
            false_target: 1,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "no-preheader LICM", |ssa| run(ssa, &method, &log));
        assert!(!changed, "loop without preheader should not hoist");
    }

    #[test]
    fn hoist_multiple_invariants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 5);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let cond = local_at(&mut ssa, 2, 0, 2);
        let inv1 = local_at(&mut ssa, 3, 2, 0);
        let inv2 = local_at(&mut ssa, 4, 2, 1);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(3),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: b,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
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
        b2.add_instruction(instr(SsaOp::Add {
            dest: inv1,
            left: a,
            right: b,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Mul {
            dest: inv2,
            left: a,
            right: b,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "multiple-invariant LICM", |ssa| {
            run(ssa, &method, &log)
        });
        assert!(changed, "multiple invariants should be hoisted");
    }
}
