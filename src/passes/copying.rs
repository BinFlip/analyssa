//! Copy propagation pass — eliminates redundant copy operations and trivial
//! phi nodes.
//!
//! # Algorithm
//!
//! 1. **Collect copies**: gather all `Copy` instructions and trivial phi
//!    nodes via [`crate::analysis::PhiAnalyzer::collect_all_copies`].
//! 2. **Protect sole local defs**: exclude copies that are the only
//!    instruction-based definition of a local or argument variable whose
//!    source is in a different rename group. This prevents information loss
//!    when the variable's value cannot be recovered from a phi node or
//!    address-taken location.
//! 3. **Protect ordering**: exclude copies whose replacement source is
//!    defined later in the same block than an existing destination use.
//! 4. **Resolve chains**: replace `dest → src` pairs with
//!    `dest → ultimate_src` using [`resolve_chain`].
//! 5. **Propagate**: call [`SsaFunction::propagate_copies`] to rewrite all
//!    uses of dest variables to their ultimate sources.
//! 6. **Nop out and repair**: replace fully propagated copy-defining
//!    instructions with `Nop`, strip those nops, compact the variable table,
//!    and recompute use lists so the pass does not leave orphan variables.
//! 7. **Validate**: if the repaired function is not verifier-clean, restore
//!    the pre-iteration function and report no progress for that iteration.
//! 8. **Repeat** until no more changes (converges in 1–3 iterations).
//!
//! # Host hook
//!
//! Some hosts need to perform extra work between resolving the copy chains
//! and applying the propagation, such as recovering local-variable type
//! information. [`run_with_hook`] takes a closure
//! invoked once per iteration after `protect_sole_local_defs` and
//! `resolve_chains`, before `propagate_copies` rewrites uses. Hosts that
//! don't need a hook call [`run`] (no-op closure).
//!
//! # Complexity
//!
//! - Time: O(n * m) where n is the number of variables and m is iterations.
//! - Space: O(n) for the copy map.

use std::collections::BTreeMap;

use crate::{
    analysis::phis::PhiAnalyzer,
    bitset::BitSet,
    events::{EventKind, EventListener},
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        ops::SsaOp,
        variable::SsaVarId,
    },
    passes::utils::resolve_chain,
    target::Target,
};

/// Run copy propagation on `ssa`.
///
/// Equivalent to `run_with_hook(ssa, method, events, max_iterations, |_, _| {})`.
///
/// # Arguments
///
/// * `ssa` — The SSA function to transform in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::CopyPropagated`] events.
/// * `max_iterations` — Cap on the outer fixpoint loop.
///
/// # Returns
///
/// `true` if any copy was propagated.
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
    run_with_hook(ssa, method, events, max_iterations, |_, _| {})
}

/// Run copy propagation with a host-specified hook.
///
/// Invokes `on_resolved` once per iteration after the resolved-copies map
/// is built and before `propagate_copies` rewrites uses. The host can use
/// the hook to perform extra per-iteration work, such as recovering type
/// information.
///
/// # Arguments
///
/// * `ssa` — The SSA function to transform in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::CopyPropagated`] events.
/// * `max_iterations` — Cap on the outer fixpoint loop.
/// * `on_resolved` — Closure invoked with the resolved copy map before
///   propagation. Receives `(&mut SsaFunction<T>, &BTreeMap<SsaVarId, SsaVarId>)`.
///
/// # Returns
///
/// `true` if any copy was propagated.
pub fn run_with_hook<T, L, F>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
    max_iterations: usize,
    mut on_resolved: F,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
    F: FnMut(&mut SsaFunction<T>, &BTreeMap<SsaVarId, SsaVarId>),
{
    let mut changed = false;
    for _ in 0..max_iterations {
        let replaced = run_iteration(ssa, method, events, &mut on_resolved);
        if replaced == 0 {
            break;
        }
        changed = true;
    }
    changed
}

/// Run a single iteration of copy propagation.
///
/// Exposed for hosts that want to drive their own fixpoint loop.
///
/// # Arguments
///
/// * `ssa` — The SSA function to transform in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::CopyPropagated`] events.
/// * `on_resolved` — Closure invoked with the resolved copy map before
///   propagation.
///
/// # Returns
///
/// The number of uses replaced this iteration.
pub fn run_iteration<T, L, F>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
    mut on_resolved: F,
) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
    F: FnMut(&mut SsaFunction<T>, &BTreeMap<SsaVarId, SsaVarId>),
{
    let mut copies = PhiAnalyzer::new(ssa).collect_all_copies();
    if copies.is_empty() {
        return 0;
    }

    protect_sole_local_defs(ssa, &mut copies);
    // `source_defs` depends only on `ssa`, so build it once and reuse it for
    // both ordering-protection passes below instead of rebuilding per call.
    let source_defs = build_source_defs(ssa);
    protect_same_block_ordering_with(ssa, &source_defs, &mut copies);

    let resolved: BTreeMap<SsaVarId, SsaVarId> = copies
        .iter()
        .map(|(&dest, &src)| (dest, resolve_chain(&copies, src)))
        .collect();
    let mut resolved = resolved;
    protect_same_block_ordering_with(ssa, &source_defs, &mut resolved);

    on_resolved(ssa, &resolved);

    let mut result = None;
    let edit_result = ssa.edit(
        SsaEditOptions::new()
            .with_verify(true)
            .with_rollback(SsaRollbackPolicy::OnFailure),
        |editor| {
            let propagation = editor.propagate_copies(&resolved);
            for dest_idx in propagation.fully_propagated.iter() {
                editor.nop_copy_defining(SsaVarId::from_index(dest_idx));
            }
            result = Some(propagation);
            Ok(())
        },
    );
    if edit_result.is_err() {
        return 0;
    }
    let Some(result) = result else {
        return 0;
    };

    let event_pairs: Vec<(SsaVarId, SsaVarId)> = result
        .fully_propagated
        .iter()
        .chain(result.partially_propagated.iter())
        .filter_map(|dest_idx| {
            let dest = SsaVarId::from_index(dest_idx);
            resolved.get(&dest).copied().map(|src| (dest, src))
        })
        .collect();

    for (dest, src) in event_pairs {
        let event = crate::events::Event {
            kind: EventKind::CopyPropagated,
            method: Some(method.clone()),
            location: None,
            message: format!("{dest} → {src}"),
            pass: None,
        };
        events.push(event);
    }

    result.total_replaced
}

/// Remove copies from the candidate set when they would lose information.
///
/// A copy is protected (excluded from propagation) when all of the
/// following hold:
/// - The destination is in a local or argument rename group.
/// - The source is in a different rename group.
/// - The destination's group has only one instruction-based definition
///   and that group appears in phi operands, or the group is address-taken.
///
/// This prevents information loss when the variable's value cannot be
/// recovered from phi nodes or address-taken locations.
///
/// # Arguments
///
/// * `ssa` — The SSA function providing variable and rename group info.
/// * `copies` — Mutable map of `dest → src` copies to filter in place.
pub fn protect_sole_local_defs<T: Target>(
    ssa: &SsaFunction<T>,
    copies: &mut BTreeMap<SsaVarId, SsaVarId>,
) {
    let real_local_limit = (ssa.num_args() as u32).saturating_add(ssa.num_locals() as u32);

    let group_bound = ssa
        .num_locals()
        .saturating_add(ssa.num_args())
        .saturating_add(1);
    // Single traversal collecting all three facts: per-group definition counts,
    // groups referenced by phi nodes, and address-taken groups.
    let mut group_def_count: BTreeMap<u32, usize> = BTreeMap::new();
    let mut groups_in_phis = BitSet::new(group_bound);
    let mut address_taken_groups = BitSet::new(group_bound);
    for block in ssa.blocks() {
        for instr in block.instructions() {
            let op = instr.op();
            for dest in op.defs() {
                let group = ssa.rename_group(dest);
                if group < real_local_limit {
                    let counter = group_def_count.entry(group).or_insert(0);
                    *counter = counter.saturating_add(1);
                }
            }
            if let SsaOp::LoadLocalAddr { local_index, .. } = op {
                let group = (ssa.num_args() as u32).saturating_add(*local_index as u32);
                if group < real_local_limit {
                    address_taken_groups.insert(group as usize);
                }
            }
        }
        for phi in block.phi_nodes() {
            for operand in phi.operands() {
                let group = ssa.rename_group(operand.value());
                if group < real_local_limit {
                    groups_in_phis.insert(group as usize);
                }
            }
            let result_group = ssa.rename_group(phi.result());
            if result_group < real_local_limit {
                groups_in_phis.insert(result_group as usize);
            }
        }
    }

    let mut protected = BitSet::new(ssa.var_id_capacity());
    for (&dest, &src) in copies.iter() {
        let dest_group = ssa.rename_group(dest);
        if dest_group >= real_local_limit {
            continue;
        }
        if ssa.rename_group(src) == dest_group {
            continue;
        }
        let def_count = group_def_count.get(&dest_group).copied().unwrap_or(0);
        if address_taken_groups.contains(dest_group as usize)
            || (def_count <= 1 && groups_in_phis.contains(dest_group as usize))
        {
            protected.insert(dest.index());
        }
    }

    if !protected.is_empty() {
        copies.retain(|dest, _| !protected.contains(dest.index()));
    }
}

/// Removes copies that could create same-block use-before-def ordering.
///
/// A replacement `dest -> src` is unsafe when `src` is defined by an
/// instruction in block `B` and `dest` is used by an earlier instruction in
/// the same block. Rewriting that use would place `src` before its
/// definition, violating SSA instruction order.
///
/// # Arguments
///
/// * `ssa` - The SSA function providing definitions and uses.
/// * `copies` - Mutable map of copy replacements to filter in place.
pub fn protect_same_block_ordering<T: Target>(
    ssa: &SsaFunction<T>,
    copies: &mut BTreeMap<SsaVarId, SsaVarId>,
) {
    let source_defs = build_source_defs(ssa);
    protect_same_block_ordering_with(ssa, &source_defs, copies);
}

/// Builds a `def-variable -> (block, instruction)` index in a single pass.
///
/// The result depends only on `ssa`, so callers that filter multiple copy
/// maps for the same function can build it once and reuse it across
/// [`protect_same_block_ordering_with`] calls.
fn build_source_defs<T: Target>(ssa: &SsaFunction<T>) -> BTreeMap<SsaVarId, (usize, usize)> {
    let mut source_defs: BTreeMap<SsaVarId, (usize, usize)> = BTreeMap::new();
    for (block_idx, block) in ssa.blocks().iter().enumerate() {
        for (instr_idx, instr) in block.instructions().iter().enumerate() {
            for dest in instr.op().defs() {
                source_defs.insert(dest, (block_idx, instr_idx));
            }
        }
    }
    source_defs
}

/// Like [`protect_same_block_ordering`] but reuses a prebuilt `source_defs`
/// index (see [`build_source_defs`]) instead of recomputing it.
fn protect_same_block_ordering_with<T: Target>(
    ssa: &SsaFunction<T>,
    source_defs: &BTreeMap<SsaVarId, (usize, usize)>,
    copies: &mut BTreeMap<SsaVarId, SsaVarId>,
) {
    copies.retain(|dest, src| {
        let Some((src_block, src_instr)) = source_defs.get(src).copied() else {
            return true;
        };
        let Some(block) = ssa.block(src_block) else {
            return true;
        };

        for (instr_idx, instr) in block.instructions().iter().enumerate() {
            if instr_idx >= src_instr {
                break;
            }
            if instr.op().uses_var(*dest) {
                return false;
            }
        }

        true
    });
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
        testing::{
            assert_mock_valid_full, mock_terminator_at, run_mock_pass_boundary, MockTarget,
            MockType,
        },
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
    fn simple_copy_eliminated() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let src = local_at(&mut ssa, 0, 0, 0);
        let dst = local_at(&mut ssa, 1, 0, 1);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: src,
            value: ConstValue::I32(42),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: dst, src }));
        block.add_instruction(instr(SsaOp::Return { value: Some(dst) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "simple copy propagation", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(changed, "simple copy should be eliminated");
        assert!(log.has(EventKind::CopyPropagated));
    }

    #[test]
    fn three_element_chain_collapsed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let c = local_at(&mut ssa, 2, 0, 2);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
        block.add_instruction(instr(SsaOp::Copy { dest: c, src: b }));
        block.add_instruction(instr(SsaOp::Return { value: Some(c) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "copy chain propagation", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(changed, "three-element chain should be collapsed");
    }

    #[test]
    fn no_copies_nothing_changed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(7),
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "no-copy propagation", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(!changed, "no copies should mean no changes");
    }

    #[test]
    fn empty_function_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty copy propagation", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(!changed);
    }

    #[test]
    fn copy_with_zero_iterations_no_change() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let src = local_at(&mut ssa, 0, 0, 0);
        let dst = local_at(&mut ssa, 1, 0, 1);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: src,
            value: ConstValue::I32(99),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: dst, src }));
        block.add_instruction(instr(SsaOp::Return { value: Some(dst) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "zero-iteration copy propagation", |ssa| {
            run(ssa, &method, &log, 0)
        });
        assert!(!changed, "zero iterations should make no changes");
    }

    #[test]
    fn copy_propagation_uses_existing_value() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let c = local_at(&mut ssa, 2, 0, 2);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
        b0.add_instruction(instr(SsaOp::Copy { dest: c, src: a }));
        b0.add_instruction(instr(SsaOp::Return { value: Some(c) }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "existing-value copy propagation", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(changed, "copy propagation should rewrite existing values");
        // The return should reference v0 (a) or v2 (c) — propagation happens
        let ret = mock_terminator_at(&ssa, 0);
        assert!(matches!(ret, SsaOp::Return { .. }));
    }

    #[test]
    fn run_iteration_returns_count() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let src = local_at(&mut ssa, 0, 0, 0);
        let dst = local_at(&mut ssa, 1, 0, 1);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: src,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: dst, src }));
        block.add_instruction(instr(SsaOp::Return { value: Some(dst) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert_mock_valid_full(&ssa, "copy propagation iteration before");
        let count = run_iteration(&mut ssa, &0u32, &log, |_, _| {});
        assert_mock_valid_full(&ssa, "copy propagation iteration after");
        assert!(count > 0, "should have replaced at least one use");
    }

    #[test]
    fn multiple_copies_in_different_blocks() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let c = local_at(&mut ssa, 2, 1, 0);
        let d = local_at(&mut ssa, 3, 2, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Copy { dest: c, src: b }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Copy { dest: d, src: c }));
        b2.add_instruction(instr(SsaOp::Return { value: Some(d) }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "multi-block copy propagation", |ssa| {
            run(ssa, &method, &log, 10)
        });
        assert!(changed, "chain across blocks should be collapsed");
    }

    #[test]
    fn protect_same_block_ordering_removes_late_source_replacement() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let dest = local_at(&mut ssa, 0, 0, 0);
        let use_before_src = local_at(&mut ssa, 1, 0, 1);
        let src = local_at(&mut ssa, 2, 0, 2);
        let one = local_at(&mut ssa, 3, 0, 3);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: use_before_src,
            left: dest,
            right: dest,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: src,
            value: ConstValue::I32(20),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Return {
            value: Some(use_before_src),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();
        assert_mock_valid_full(&ssa, "same-block ordering protection fixture");

        let mut copies = BTreeMap::from([(dest, src)]);
        protect_same_block_ordering(&ssa, &mut copies);

        assert!(copies.is_empty());
    }

    #[test]
    fn propagated_copies_do_not_leave_orphan_variables() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let c = local_at(&mut ssa, 2, 0, 2);
        let out = local_at(&mut ssa, 3, 0, 3);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
        block.add_instruction(instr(SsaOp::Copy { dest: c, src: b }));
        block.add_instruction(instr(SsaOp::Copy { dest: out, src: c }));
        block.add_instruction(instr(SsaOp::Return { value: Some(out) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        assert!(run_mock_pass_boundary(
            &mut ssa,
            "orphan-free copy propagation",
            |ssa| run(ssa, &method, &log, 10)
        ));
        assert!(!ssa
            .iter_instructions()
            .any(|(_, _, instr)| matches!(instr.op(), SsaOp::Copy { .. })));
    }
}
