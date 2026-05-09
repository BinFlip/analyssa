//! Diamond and loop-shaped SSA fixtures for infrastructure coverage.

#![allow(clippy::unwrap_used)]

use analyssa::{
    analysis::{
        loop_analyzer::SsaLoopAnalysis, LoopForest, LoopType, SsaCfg, SsaEvaluator, SsaVerifier,
        VerifyLevel,
    },
    graph::NodeId,
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    testing::{MockTarget, MockType},
    PointerSize,
};

fn local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize) -> SsaVarId {
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

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

fn build_diamond_loop() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 9);

    let seed = local(&mut ssa, 0, 0, 0);
    let one = local(&mut ssa, 1, 0, 1);
    let limit = local(&mut ssa, 2, 0, 2);
    let dead = local(&mut ssa, 3, 0, 3);
    let left = local(&mut ssa, 4, 1, 0);
    let right = local(&mut ssa, 5, 2, 0);
    let merged = phi_local(&mut ssa, 6, 3);
    let cmp = local(&mut ssa, 7, 3, 0);
    let next = local(&mut ssa, 8, 4, 0);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: seed,
        value: ConstValue::I32(10),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: limit,
        value: ConstValue::I32(20),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: dead,
        value: ConstValue::I32(1234),
    }));
    entry.add_instruction(instr(SsaOp::Branch {
        condition: seed,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    let mut then_block = SsaBlock::new(1);
    then_block.add_instruction(instr(SsaOp::Add {
        dest: left,
        left: seed,
        right: one,
        flags: None,
    }));
    then_block.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(then_block);

    let mut else_block = SsaBlock::new(2);
    else_block.add_instruction(instr(SsaOp::Sub {
        dest: right,
        left: seed,
        right: one,
        flags: None,
    }));
    else_block.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(else_block);

    let mut header = SsaBlock::new(3);
    let mut merge_phi = PhiNode::new(merged, VariableOrigin::Local(6));
    merge_phi.add_operand(PhiOperand::new(left, 1));
    merge_phi.add_operand(PhiOperand::new(right, 2));
    merge_phi.add_operand(PhiOperand::new(next, 4));
    header.add_phi(merge_phi);
    header.add_instruction(instr(SsaOp::Clt {
        dest: cmp,
        left: merged,
        right: limit,
        unsigned: false,
    }));
    header.add_instruction(instr(SsaOp::Branch {
        condition: cmp,
        true_target: 4,
        false_target: 5,
    }));
    ssa.add_block(header);

    let mut latch = SsaBlock::new(4);
    latch.add_instruction(instr(SsaOp::Add {
        dest: next,
        left: merged,
        right: one,
        flags: None,
    }));
    latch.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(latch);

    let mut exit = SsaBlock::new(5);
    exit.add_instruction(instr(SsaOp::Return {
        value: Some(merged),
    }));
    ssa.add_block(exit);

    ssa.recompute_uses();
    ssa
}

#[test]
fn diamond_loop_validates_cfg_verifier_and_def_use_index() {
    let ssa = build_diamond_loop();
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "verifier errors: {errors:?}");

    let cfg = SsaCfg::from_ssa(&ssa);
    assert_eq!(cfg.block_successors(0), &[1, 2]);
    assert_eq!(cfg.block_predecessors(3), &[1, 2, 4]);
    assert_eq!(cfg.block_successors(3), &[4, 5]);
    assert_eq!(cfg.exits().len(), 1);

    let index = analyssa::analysis::DefUseIndex::build_with_ops(&ssa);
    let seed = SsaVarId::from_index(0);
    let dead = SsaVarId::from_index(3);
    let merged = SsaVarId::from_index(6);
    let cmp = SsaVarId::from_index(7);

    assert!(index.has_ops());
    assert_eq!(index.variable_count(), ssa.variable_count());
    assert_eq!(index.use_count(seed), 3);
    assert_eq!(index.use_count(merged), 3);
    assert!(index.is_unused(dead));
    assert!(index.is_phi_def(merged));
    assert!(matches!(index.def_op(cmp), Some(SsaOp::Clt { .. })));
    assert_eq!(index.defs_at(3, 0), &[cmp]);

    let header_uses = index.uses_at(3, 0);
    assert!(header_uses.contains(&merged));
    assert!(header_uses.contains(&SsaVarId::from_index(2)));
    assert!(index.defs_in_block(3).contains(&merged));
}

#[test]
fn diamond_loop_analysis_finds_header_latch_exit_and_noncanonical_preheader() {
    let ssa = build_diamond_loop();
    let forest = ssa.analyze_loops();

    assert_eq!(forest.len(), 1);
    assert!(forest.is_in_loop(analyssa::graph::NodeId::new(3)));
    assert!(forest.is_in_loop(analyssa::graph::NodeId::new(4)));
    assert!(!forest.is_in_loop(analyssa::graph::NodeId::new(5)));

    let loop_info = forest
        .loop_for_header(analyssa::graph::NodeId::new(3))
        .unwrap();
    assert_eq!(loop_info.header.index(), 3);
    assert_eq!(loop_info.size(), 2);
    assert_eq!(loop_info.loop_type, LoopType::PreTested);
    assert_eq!(loop_info.single_latch().unwrap().index(), 4);
    assert_eq!(loop_info.preheader, None);
    assert_eq!(loop_info.exits.len(), 1);
    assert_eq!(
        loop_info.exits.first().map(|e| e.exiting_block.index()),
        Some(3)
    );
    assert_eq!(
        loop_info.exits.first().map(|e| e.exit_block.index()),
        Some(5)
    );
}

#[test]
fn diamond_path_aware_evaluator_selects_phi_operand_by_predecessor() {
    let ssa = build_diamond_loop();
    let merged = SsaVarId::from_index(6);

    let mut from_then = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    from_then.evaluate_block(0);
    from_then.evaluate_block(1);
    from_then.set_predecessor(Some(1));
    from_then.evaluate_phis(3);
    assert_eq!(
        from_then.get_concrete(merged).and_then(ConstValue::as_i64),
        Some(11)
    );

    let mut from_else = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    from_else.evaluate_block(0);
    from_else.evaluate_block(2);
    from_else.set_predecessor(Some(2));
    from_else.evaluate_phis(3);
    assert_eq!(
        from_else.get_concrete(merged).and_then(ConstValue::as_i64),
        Some(9)
    );
}

#[test]
fn replace_uses_reports_self_reference_skips_without_touching_phis() {
    let mut ssa = build_diamond_loop();
    let seed = SsaVarId::from_index(0);
    let left = SsaVarId::from_index(4);
    let merged = SsaVarId::from_index(6);

    let result = ssa.replace_uses(seed, left);
    assert_eq!(result.skipped, 1);
    assert!(result.replaced > 0);
    assert!(!result.is_complete());

    let header_phi = &ssa.block(3).unwrap().phi_nodes().first().unwrap();
    assert_eq!(header_phi.operands().first().map(|o| o.value()), Some(left));
    assert_eq!(
        header_phi.operands().get(2).map(|o| o.value()),
        Some(SsaVarId::from_index(8))
    );

    let latch_add = ssa.block(4).unwrap().instruction(0).unwrap();
    assert!(matches!(
        latch_add.op(),
        SsaOp::Add {
            left: l,
            right: r,
            ..
        } if *l == merged && *r == SsaVarId::from_index(1)
    ));
}
/// Builds: outer preheader → outer header → inner header → inner body → inner
/// latch → inner header; inner exit → outer body → outer latch → outer header;
/// outer exit → ret.
///
/// ```text
/// B0: preheader (const i=0, const n=10, jump B1)
/// B1: outer header (phi i, clt i < n, branch B2/B5)
/// B2: inner header (phi j, clt j < n, branch B3/B4)
/// B3: inner body (add j, jump B2)
/// B4: inner exit (add i, jump B1)
/// B5: exit (ret i)
/// ```
fn build_nested_loops() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);

    let zero = local(&mut ssa, 0, 0, 0);
    let n = local(&mut ssa, 1, 0, 1);
    let i_init = local(&mut ssa, 2, 0, 2);
    let i_phi = phi_local(&mut ssa, 3, 1);
    let i_lt = local(&mut ssa, 4, 1, 0);
    let j_phi = phi_local(&mut ssa, 5, 2);
    let j_lt = local(&mut ssa, 6, 2, 0);
    let one = local(&mut ssa, 7, 0, 3);
    let i_next = local(&mut ssa, 8, 4, 0);
    let j_next = local(&mut ssa, 9, 3, 0);

    let mut block0 = SsaBlock::new(0);
    block0.add_instruction(instr(SsaOp::Const {
        dest: zero,
        value: ConstValue::I32(0),
    }));
    block0.add_instruction(instr(SsaOp::Const {
        dest: n,
        value: ConstValue::I32(10),
    }));
    block0.add_instruction(instr(SsaOp::Copy {
        dest: i_init,
        src: zero,
    }));
    block0.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    block0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(block0);

    // Outer header (B1)
    let mut block1 = SsaBlock::new(1);
    let mut phi_i = PhiNode::new(i_phi, VariableOrigin::Local(3));
    phi_i.add_operand(PhiOperand::new(i_init, 0));
    phi_i.add_operand(PhiOperand::new(i_next, 4));
    block1.add_phi(phi_i);
    block1.add_instruction(instr(SsaOp::Clt {
        dest: i_lt,
        left: i_phi,
        right: n,
        unsigned: false,
    }));
    block1.add_instruction(instr(SsaOp::Branch {
        condition: i_lt,
        true_target: 2,
        false_target: 5,
    }));
    ssa.add_block(block1);

    // Inner header (B2)
    let mut block2 = SsaBlock::new(2);
    let mut phi_j = PhiNode::new(j_phi, VariableOrigin::Local(5));
    phi_j.add_operand(PhiOperand::new(zero, 1));
    phi_j.add_operand(PhiOperand::new(j_next, 3));
    block2.add_phi(phi_j);
    block2.add_instruction(instr(SsaOp::Clt {
        dest: j_lt,
        left: j_phi,
        right: n,
        unsigned: false,
    }));
    block2.add_instruction(instr(SsaOp::Branch {
        condition: j_lt,
        true_target: 3,
        false_target: 4,
    }));
    ssa.add_block(block2);

    // Inner body (B3)
    let mut block3 = SsaBlock::new(3);
    block3.add_instruction(instr(SsaOp::Add {
        dest: j_next,
        left: j_phi,
        right: one,
        flags: None,
    }));
    block3.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(block3);

    // Inner exit / outer body (B4)
    let mut block4 = SsaBlock::new(4);
    block4.add_instruction(instr(SsaOp::Add {
        dest: i_next,
        left: i_phi,
        right: one,
        flags: None,
    }));
    block4.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(block4);

    // Outer exit (B5)
    let mut block5 = SsaBlock::new(5);
    block5.add_instruction(instr(SsaOp::Return { value: Some(i_phi) }));
    ssa.add_block(block5);

    ssa.recompute_uses();
    ssa
}

#[test]
fn nested_loops_are_detected_with_correct_depth_and_sizes() {
    let ssa = build_nested_loops();
    let forest: LoopForest = ssa.analyze_loops();

    assert_eq!(forest.len(), 2, "expected 2 loops (inner + outer)");

    // Outer loop header = B1
    let outer = forest.loop_for_header(NodeId::new(1));
    assert!(outer.is_some(), "outer loop not found");
    let outer = outer.unwrap();
    assert_eq!(outer.loop_type, LoopType::PreTested);
    // Outer contains B1, B2, B3, B4
    assert!(outer.size() >= 3);
    assert!(forest.is_in_loop(NodeId::new(1)));
    assert!(forest.is_in_loop(NodeId::new(2)));

    // Inner loop header = B2
    let inner = forest.loop_for_header(NodeId::new(2));
    assert!(inner.is_some(), "inner loop not found");
    let inner = inner.unwrap();
    assert_eq!(inner.loop_type, LoopType::PreTested);

    // B5 is not in any loop
    assert!(!forest.is_in_loop(NodeId::new(5)));

    // Inner is innermost; outer might or might not be depending on nesting tree
    assert!(inner.is_innermost());
    assert!(forest.innermost_loop(NodeId::new(2)).is_some());
    assert!(forest.is_in_loop(NodeId::new(1)));
    assert!(forest.is_in_loop(NodeId::new(2)));
}

#[test]
fn nested_loops_cfg_is_valid() {
    let ssa = build_nested_loops();
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.is_empty(), "verifier errors: {errors:?}");

    let cfg = SsaCfg::from_ssa(&ssa);
    assert_eq!(cfg.block_successors(0), &[1]);
    assert_eq!(cfg.block_successors(1), &[2, 5]);
    assert_eq!(cfg.block_successors(2), &[3, 4]);
    assert_eq!(cfg.block_successors(3), &[2]);
    assert_eq!(cfg.block_successors(4), &[1]);
    assert!(cfg.block_predecessors(1).len() == 2);
    assert!(cfg.block_predecessors(2).len() == 2);
    assert_eq!(cfg.exits().len(), 1);
}

/// Builds: preheader B0 → header B1 → body B2 → condition B3 (branch back to
/// B2 or exit B4)
fn build_post_tested_loop() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 5);

    let init = local(&mut ssa, 0, 0, 0);
    let one = local(&mut ssa, 1, 0, 1);
    let phi_i = phi_local(&mut ssa, 2, 2);
    let i_add = local(&mut ssa, 3, 2, 0);
    let cond = local(&mut ssa, 4, 3, 0);

    // Preheader
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: init,
        value: ConstValue::I32(0),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b0);

    // Header (used as phi entry but post-tested)
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    // Body (B2)
    let mut b2 = SsaBlock::new(2);
    let mut phi = PhiNode::new(phi_i, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(init, 0));
    phi.add_operand(PhiOperand::new(init, 1));
    phi.add_operand(PhiOperand::new(i_add, 3));
    b2.add_phi(phi);
    b2.add_instruction(instr(SsaOp::Add {
        dest: i_add,
        left: phi_i,
        right: one,
        flags: None,
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b2);

    // Condition
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::Clt {
        dest: cond,
        left: i_add,
        right: SsaVarId::from_index(0),
        unsigned: false,
    }));
    b3.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 2,
        false_target: 4,
    }));
    ssa.add_block(b3);

    // Exit
    let mut b4 = SsaBlock::new(4);
    b4.add_instruction(instr(SsaOp::Return { value: Some(i_add) }));
    ssa.add_block(b4);

    ssa.recompute_uses();
    ssa
}

#[test]
fn post_tested_loop_detected_in_nonstandard_layout() {
    let ssa = build_post_tested_loop();
    let forest: LoopForest = ssa.analyze_loops();

    // The dominance-based analyzer should find loops based on back edges.
    // B2 is the header (dominator of B3 which has a back edge to B2).
    assert!(forest.is_in_loop(NodeId::new(2)) || forest.is_in_loop(NodeId::new(3)));
    assert_eq!(forest.len(), 1);
}

/// Builds: entry B0 branches to header B1 or B2. B1 adds x and jumps to B3.
/// B2 adds y and jumps to B3. B3 (latch) branches back to header or exit.
fn build_multi_latch_loop() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 6);

    let cond = local(&mut ssa, 0, 0, 0);
    let phi_x = phi_local(&mut ssa, 1, 1);
    let x_val = local(&mut ssa, 2, 2, 0);
    let y_val = local(&mut ssa, 3, 3, 0);
    let latch_cond = local(&mut ssa, 4, 4, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    // Header (B1)
    let _one = SsaVarId::from_index(10);
    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(phi_x, VariableOrigin::Local(1));
    phi.add_operand(PhiOperand::new(cond, 0));
    phi.add_operand(PhiOperand::new(x_val, 2));
    phi.add_operand(PhiOperand::new(y_val, 3));
    b1.add_phi(phi);
    // Need to use existing variables for operands
    b1.add_instruction(instr(SsaOp::Copy {
        dest: SsaVarId::from_index(10),
        src: phi_x,
    }));
    b1.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 2,
        false_target: 3,
    }));
    ssa.add_block(b1);

    // Latch path 1 (B2)
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Add {
        dest: x_val,
        left: phi_x,
        right: SsaVarId::from_index(10),
        flags: None,
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 4 }));
    ssa.add_block(b2);

    // Latch path 2 (B3)
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::Sub {
        dest: y_val,
        left: phi_x,
        right: SsaVarId::from_index(10),
        flags: None,
    }));
    b3.add_instruction(instr(SsaOp::Jump { target: 4 }));
    ssa.add_block(b3);

    // Latch (B4) — branches back to header
    let mut b4 = SsaBlock::new(4);
    b4.add_instruction(instr(SsaOp::Clt {
        dest: latch_cond,
        left: phi_x,
        right: SsaVarId::from_index(10),
        unsigned: false,
    }));
    b4.add_instruction(instr(SsaOp::Branch {
        condition: latch_cond,
        true_target: 1,
        false_target: 5,
    }));
    ssa.add_block(b4);

    // Exit (B5)
    let mut b5 = SsaBlock::new(5);
    b5.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(b5);

    ssa.recompute_uses();
    ssa
}

#[test]
fn multi_latch_loop_detected_as_single_loop() {
    let ssa = build_multi_latch_loop();
    let forest: LoopForest = ssa.analyze_loops();

    assert_eq!(forest.len(), 1);
    // The loop should contain blocks B1, B2, B3, B4
    assert!(forest.is_in_loop(NodeId::new(1)));
    assert!(forest.is_in_loop(NodeId::new(4)));
    assert!(!forest.is_in_loop(NodeId::new(5)));
}

/// Builds: header with conditional branch — one goes to simple latch that
/// jumps back, other goes to body that does some work then jumps to different
/// latch that also jumps back.
fn build_double_back_edge_loop() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 5);

    let init = local(&mut ssa, 0, 0, 0);
    let cond = local(&mut ssa, 1, 0, 1);
    let phi = phi_local(&mut ssa, 2, 1);
    let body_val = local(&mut ssa, 3, 2, 0);
    let latch_val = local(&mut ssa, 4, 3, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: init,
        value: ConstValue::I32(0),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    // Header (B1)
    let mut b1 = SsaBlock::new(1);
    let mut p = PhiNode::new(phi, VariableOrigin::Local(2));
    p.add_operand(PhiOperand::new(init, 0));
    p.add_operand(PhiOperand::new(body_val, 2));
    p.add_operand(PhiOperand::new(latch_val, 3));
    b1.add_phi(p);
    b1.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 2,
        false_target: 3,
    }));
    ssa.add_block(b1);

    // Body (B2) — back edge 1
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Add {
        dest: body_val,
        left: phi,
        right: init,
        flags: None,
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b2);

    // Latch (B3) — back edge 2
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::Sub {
        dest: latch_val,
        left: phi,
        right: init,
        flags: None,
    }));
    b3.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b3);

    ssa.recompute_uses();
    ssa
}

#[test]
fn double_back_edge_loop_detected_and_has_two_latches() {
    let ssa = build_double_back_edge_loop();
    let forest: LoopForest = ssa.analyze_loops();

    assert_eq!(forest.len(), 1);
    let loop_info = forest.loop_for_header(NodeId::new(1)).unwrap();
    assert!(!loop_info.has_single_latch(), "expected multiple latches");
    assert_eq!(loop_info.latches.len(), 2);
    // Header is canonical only with single latch
    assert!(!loop_info.is_canonical());
}

/// Builds irreducible: B0 branches to B1 and B2, both jump to B3, B3 branches
/// to B1/B2 (mutual dependency without single back edge). No loop should be
/// detected via natural loop analysis.
fn build_irreducible_cfg() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 4);

    let cond = local(&mut ssa, 0, 0, 0);
    let split_cond = local(&mut ssa, 1, 3, 0);

    let mut b0 = SsaBlock::new(0);
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
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::Clt {
        dest: split_cond,
        left: cond,
        right: cond,
        unsigned: false,
    }));
    b3.add_instruction(instr(SsaOp::Branch {
        condition: split_cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b3);

    ssa.recompute_uses();
    ssa
}

#[test]
fn irreducible_cfg_produces_no_natural_loops() {
    let ssa = build_irreducible_cfg();
    let forest: LoopForest = ssa.analyze_loops();

    // Natural loop detection uses dominator-based back edges; irreducible
    // loops don't have a single dominating header, so none are found.
    assert_eq!(
        forest.len(),
        0,
        "irreducible cfg should produce no natural loops"
    );
}

#[test]
fn empty_function_has_no_loops() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    let forest: LoopForest = ssa.analyze_loops();
    assert!(forest.is_empty());
}

#[test]
fn single_block_function_has_no_loops() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    let v = local(&mut ssa, 0, 0, 0);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(v) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let forest: LoopForest = ssa.analyze_loops();
    assert!(forest.is_empty());
    assert_eq!(forest.len(), 0);
}

#[test]
fn pre_tested_loop_identified_vs_post_tested() {
    let ssa = build_nested_loops();
    let forest: LoopForest = ssa.analyze_loops();

    let outer = forest.loop_for_header(NodeId::new(1)).unwrap();
    // Outer loop condition is tested at the top of the loop (PreTested)
    assert_eq!(outer.loop_type, LoopType::PreTested);
}

#[test]
fn ssa_loop_analysis_trait_method_works() {
    let ssa = build_nested_loops();
    let forest = SsaFunction::analyze_loops(&ssa);
    assert_eq!(forest.len(), 2);
}
