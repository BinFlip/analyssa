//! Sparse Conditional Constant Propagation (SCCP).
//!
//! SCCP is a powerful constant propagation algorithm that combines sparse
//! analysis (working on SSA def-use chains directly) with conditional
//! propagation (using constant branch conditions to prune unreachable paths).
//!
//! # Algorithm (Wegman & Zadeck 1991)
//!
//! SCCP maintains two lattice structures:
//!
//! 1. **Value lattice** ([`ScalarValue`]): Three-level lattice per variable
//!    - `Top`: No information yet (value might be anything)
//!    - `Constant(c)`: Known compile-time constant
//!    - `Bottom`: Not constant (multiple possible values or varying)
//!
//! 2. **CFG reachability**: Tracks which edges `(from, to)` are executable.
//!    Only executable edges contribute to phi evaluation.
//!
//! ## Worklists
//!
//! The algorithm uses dual worklists:
//! - **SSA worklist**: Variables whose lattice value changed → triggers
//!   re-evaluation of all uses of that variable
//! - **CFG worklist**: Edges that became executable → triggers block discovery
//!   and phi re-evaluation
//!
//! ## Edge-Based Phi Evaluation (Key Insight)
//!
//! Phi nodes are evaluated based on which **edges** are executable, not which
//! blocks are reachable. Consider:
//!
//! ```text
//!        B0
//!       /  \
//!      B1  B2
//!       \  /
//!        B3: y = phi(x from B1, x from B2)
//! ```
//!
//! If the branch in B0 evaluates to a constant `true`, only edge B0→B1 is
//! executable. Even though B3 is reachable via B1→B3, the phi should only
//! consider the operand from B1. This gives `y = 5` (if B1 assigns 5) instead
//! of `y = Bottom` (which would result from merging B1's 5 with B2's unknown).
//!
//! ## Back Edge Handling
//!
//! For loop back edges, phi operands are treated as `Bottom` (unknown/varying).
//! This prevents incorrect constant propagation where a phi appears constant
//! using the first-iteration value but actually varies across iterations
//! (e.g., a Fibonacci counter). Without this, SCCP would incorrectly conclude
//! that `b = phi(1, temp)` is always `1` when `temp = 1` in the first iteration.
//!
//! ## Argument Initialization
//!
//! Argument variables (version 0, defined at function entry) start as `Bottom`
//! rather than `Top`. This is critical because arguments are external inputs
//! that could be anything, and without this distinction, branch conditions
//! depending on arguments would stay at `Top` forever (since no instruction
//! defines them), preventing the branch from ever adding edges.
//!
//! # Differences from Generic Solver
//!
//! Unlike the generic framework solver, SCCP doesn't use block-level transfer
//! functions. Instead, it processes individual SSA instructions and phi nodes
//! directly, which is more efficient for sparse analyses and enables the
//! edge-based precision.
//!
//! # Reference
//!
//! Wegman & Zadeck, "Constant Propagation with Conditional Branches",
//! ACM TOPLAS 1991.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::{
    analysis::{consts::evaluate_const_op, dataflow::lattice::MeetSemiLattice},
    bitset::BitSet,
    graph::{NodeId, RootedGraph, Successors},
    ir::{
        block::SsaBlock, function::SsaFunction, instruction::SsaInstruction, ops::SsaOp,
        phi::PhiNode, value::ConstValue, variable::SsaVarId,
    },
    pointer::PointerSize,
    target::Target,
};

/// Sparse Conditional Constant Propagation analysis.
///
/// This analysis computes which SSA variables have constant values,
/// taking into account that some branches may never be taken.
///
/// # Example
///
/// ```rust
/// use analyssa::{
///     analysis::{
///         dataflow::{ConstantPropagation, ScalarValue},
///         SsaCfg,
///     },
///     ir::SsaVarId,
///     testing, PointerSize,
/// };
///
/// // `const_i32_return(42)` is `v0 = 42; return v0`.
/// let ssa = testing::const_i32_return(42);
/// let graph = SsaCfg::from_ssa(&ssa);
///
/// let mut sccp = ConstantPropagation::new(PointerSize::Bit64);
/// let results = sccp.analyze(&ssa, &graph);
///
/// // Check if a variable is constant.
/// let var_id = SsaVarId::from_index(0);
/// assert!(matches!(
///     results.get_value(var_id),
///     Some(ScalarValue::Constant(_))
/// ));
/// ```
pub struct ConstantPropagation<T: Target> {
    /// Current value for each SSA variable.
    values: BTreeMap<SsaVarId, ScalarValue<T>>,
    /// Executable CFG edges.
    executable_edges: BTreeSet<(usize, usize)>,
    /// Blocks that have been marked executable.
    executable_blocks: BitSet,
    /// SSA worklist: variables whose values have changed.
    ssa_worklist: VecDeque<SsaVarId>,
    /// CFG worklist: edges that have become executable.
    cfg_worklist: VecDeque<(usize, usize)>,
    /// Back edges: edges where the target was already executable when the edge was added.
    /// These represent loop back edges and their values should be treated as unknown.
    back_edges: BTreeSet<(usize, usize)>,
    /// Target pointer size for native int/uint masking.
    pointer_size: PointerSize,
}

impl<T: Target> ConstantPropagation<T> {
    /// Creates a new constant propagation analysis.
    ///
    /// # Arguments
    ///
    /// * `ptr_size` - Target pointer size for native int/uint masking.
    #[must_use]
    pub fn new(ptr_size: PointerSize) -> Self {
        Self {
            values: BTreeMap::new(),
            executable_edges: BTreeSet::new(),
            executable_blocks: BitSet::new(0),
            ssa_worklist: VecDeque::new(),
            cfg_worklist: VecDeque::new(),
            back_edges: BTreeSet::new(),
            pointer_size: ptr_size,
        }
    }

    /// Runs the SCCP algorithm on the given SSA function.
    ///
    /// The CFG parameter can be any type that implements the required graph traits:
    /// - `RootedGraph` for the entry point
    /// - `Successors` for traversing outgoing edges
    ///
    /// This allows using both `ControlFlowGraph` (from CIL blocks) and `SsaCfg`
    /// (from SSA function terminators).
    ///
    /// Returns the analysis results containing the value for each variable.
    pub fn analyze<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G) -> SccpResult<T>
    where
        G: RootedGraph + Successors,
    {
        self.initialize(ssa, cfg);
        self.propagate(ssa, cfg);

        SccpResult {
            values: std::mem::take(&mut self.values),
            executable_blocks: std::mem::take(&mut self.executable_blocks),
        }
    }

    /// Initializes the analysis state.
    fn initialize<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        self.values.clear();
        self.executable_edges.clear();
        self.executable_blocks = BitSet::new(ssa.block_count());
        self.ssa_worklist.clear();
        self.cfg_worklist.clear();
        self.back_edges.clear();

        // Initialize variable values:
        // - Argument variables (version 0, defined at entry) start as Bottom (unknown input)
        // - All other variables start as Top (no information yet)
        //
        // This distinction is critical: arguments are external inputs that could be anything,
        // while other variables are defined by instructions that SCCP will evaluate.
        // Without this, branch conditions depending on arguments stay at Top forever
        // (since no instruction defines them), causing the branch to never add edges.
        for var in ssa.variables() {
            let initial_value = if var.origin().is_argument()
                && var.version() == 0
                && var.def_site().instruction.is_none()
            {
                // This is the initial definition of an argument - it's an unknown input
                ScalarValue::Bottom
            } else {
                // Regular variable - will be evaluated by instructions
                ScalarValue::Top
            };
            self.values.insert(var.id(), initial_value);
        }

        // Mark entry block as executable
        let entry = cfg.entry().index();
        self.mark_block_executable(entry);

        // Add entry block's outgoing edges to CFG worklist
        // For unconditional edges or first visit, add all successors
        for succ in cfg.successors(cfg.entry()) {
            self.cfg_worklist.push_back((entry, succ.index()));
        }

        // Process entry block definitions immediately
        if let Some(block) = ssa.block(entry) {
            self.process_block_definitions(block);
        }
    }

    /// Main propagation loop.
    fn propagate<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        // Process until both worklists are empty
        loop {
            // Process CFG worklist first (to discover new blocks)
            while let Some((from, to)) = self.cfg_worklist.pop_front() {
                if self.executable_edges.insert((from, to)) {
                    // Detect back edges: if the target block was already executable
                    // when this edge is being added, it's a back edge (loop).
                    // PHI operands from back edges represent values that change
                    // across loop iterations and should be treated as unknown.
                    if self.is_block_executable(to) {
                        self.back_edges.insert((from, to));
                    }
                    // This edge became executable
                    self.process_edge(from, to, ssa, cfg);
                }
            }

            // Process SSA worklist
            if let Some(var) = self.ssa_worklist.pop_front() {
                self.process_variable_uses(var, ssa, cfg);
            } else {
                // Both worklists empty
                break;
            }
        }
    }

    /// Processes a newly executable CFG edge.
    ///
    /// When an edge `(from, to)` becomes executable:
    /// 1. If this is the first edge reaching `to`, mark the block executable and
    ///    process all its definitions
    /// 2. Re-evaluate all phi nodes in `to` since they may now have a new operand
    ///    from the `from` block
    /// 3. If first visit, propagate outgoing edges based on the terminator
    fn process_edge<G>(&mut self, from: usize, to: usize, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        let first_visit = !self.is_block_executable(to);

        if first_visit {
            self.mark_block_executable(to);

            // Process all definitions in the block
            if let Some(block) = ssa.block(to) {
                self.process_block_definitions(block);
            }
        }

        // Re-evaluate phi nodes in the target block.
        // The new edge (from, to) may contribute a new operand value.
        if let Some(block) = ssa.block(to) {
            for phi in block.phi_nodes() {
                // Only re-evaluate if this phi has an operand from the `from` block
                if phi.operand_from(from).is_some() {
                    let new_value = self.evaluate_phi(phi, to);
                    self.update_value(phi.result(), &new_value);
                }
            }
        }

        // If first visit, propagate outgoing edges based on terminator
        if first_visit {
            if let Some(block) = ssa.block(to) {
                self.propagate_outgoing_edges(to, block, cfg);
            }
        }
    }

    /// Processes all definitions in a block (non-phi instructions).
    ///
    /// This evaluates each instruction and updates the value lattice for any
    /// variables defined by the instruction.
    fn process_block_definitions(&mut self, block: &SsaBlock<T>) {
        for instr in block.instructions() {
            self.update_instruction_defs(instr);
        }
    }

    /// Processes uses of a variable whose value changed.
    fn process_variable_uses<G>(&mut self, var: SsaVarId, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        // Find all uses of this variable
        if let Some(ssa_var) = ssa.variable(var) {
            for use_site in ssa_var.uses() {
                let block_id = use_site.block;

                // Skip if block is not executable
                if !self.is_block_executable(block_id) {
                    continue;
                }

                if use_site.is_phi_operand {
                    // Re-evaluate the phi node
                    if let Some(block) = ssa.block(block_id) {
                        if let Some(phi) = block.phi(use_site.instruction) {
                            let new_value = self.evaluate_phi(phi, block_id);
                            self.update_value(phi.result(), &new_value);
                        }
                    }
                } else {
                    // Re-evaluate the instruction
                    if let Some(block) = ssa.block(block_id) {
                        if let Some(instr) = block.instruction(use_site.instruction) {
                            self.update_instruction_defs(instr);

                            // Check if this is a branch instruction
                            if instr.is_terminator() {
                                self.propagate_outgoing_edges(block_id, block, cfg);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Propagates outgoing edges from a block based on terminator.
    fn propagate_outgoing_edges<G>(&mut self, block_id: usize, block: &SsaBlock<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        // Find the terminator instruction
        match block.terminator_op() {
            Some(SsaOp::Branch {
                condition,
                true_target,
                false_target,
            }) => {
                // Conditional branch - check if condition is constant
                match self.get_value(*condition) {
                    ScalarValue::Constant(c) => {
                        // Known branch direction
                        let target = if c.as_bool() == Some(true) {
                            *true_target
                        } else {
                            *false_target
                        };
                        self.add_cfg_edge(block_id, target);
                    }
                    ScalarValue::Top => {
                        // Unknown - don't add edges yet
                    }
                    ScalarValue::Bottom => {
                        // Could go either way - add both edges
                        self.add_cfg_edge(block_id, *true_target);
                        self.add_cfg_edge(block_id, *false_target);
                    }
                }
            }
            Some(SsaOp::Switch {
                value,
                targets,
                default,
            }) => {
                // Switch statement
                match self.get_value(*value) {
                    ScalarValue::Constant(c) => {
                        // Known switch value - use checked conversion to handle negative values
                        if let Some(idx) = c.as_i32().and_then(|i| usize::try_from(i).ok()) {
                            if let Some(target) = targets.get(idx) {
                                self.add_cfg_edge(block_id, *target);
                            } else {
                                self.add_cfg_edge(block_id, *default);
                            }
                        } else {
                            self.add_cfg_edge(block_id, *default);
                        }
                    }
                    ScalarValue::Top | ScalarValue::Bottom => {
                        // Unknown or could be anything - conservatively add all edges.
                        // This is critical for control flow obfuscation where the switch
                        // value is computed dynamically and cannot be statically determined.
                        for &target in targets {
                            self.add_cfg_edge(block_id, target);
                        }
                        self.add_cfg_edge(block_id, *default);
                    }
                }
            }
            Some(SsaOp::Jump { target }) => {
                // Unconditional jump
                self.add_cfg_edge(block_id, *target);
            }
            Some(
                SsaOp::Return { .. }
                | SsaOp::Throw { .. }
                | SsaOp::Rethrow
                | SsaOp::EndFinally
                | SsaOp::EndFilter { .. }
                | SsaOp::InterruptReturn
                | SsaOp::Unreachable,
            ) => {
                // No successors
            }
            _ => {
                // Fall through or unknown terminator - add all CFG successors
                let node = NodeId::new(block_id);
                for succ in cfg.successors(node) {
                    self.add_cfg_edge(block_id, succ.index());
                }
            }
        }
    }

    /// Adds a CFG edge to the worklist if not already executable.
    fn add_cfg_edge(&mut self, from: usize, to: usize) {
        if !self.executable_edges.contains(&(from, to)) {
            self.cfg_worklist.push_back((from, to));
        }
    }

    /// Returns `true` if `block` is currently marked executable.
    ///
    /// Block indices flow in from terminator targets, which the IR permits to
    /// be out of range — a terminator may reference a block that was never
    /// recovered (common in stripped/obfuscated binaries), and the
    /// [`crate::analysis::verifier`] explicitly tolerates such dangling
    /// successors. The `executable_blocks` bitset is sized to exactly
    /// `block_count`, so an out-of-range target is by definition unreachable:
    /// report it `false` instead of indexing past the bitset and panicking.
    fn is_block_executable(&self, block: usize) -> bool {
        self.executable_blocks.contains_checked(block)
    }

    /// Marks `block` executable, ignoring out-of-range indices (see
    /// [`Self::is_block_executable`]).
    fn mark_block_executable(&mut self, block: usize) {
        self.executable_blocks.insert_checked(block);
    }

    /// Evaluates a phi node to get its current value.
    ///
    /// This is the key to SCCP's precision: we only consider operands from
    /// **executable edges**, not just reachable blocks. This allows us to
    /// propagate constants through conditional branches more precisely.
    ///
    /// For example, if we have:
    /// ```text
    /// B0: if (true) goto B1 else goto B2
    /// B1: x = 5; goto B3
    /// B2: x = 10; goto B3
    /// B3: y = phi(x from B1, x from B2)
    /// ```
    /// Even though B3 is reachable, only the edge B1→B3 is executable (because
    /// the branch condition is constant true). So y = 5, not bottom.
    ///
    /// # Arguments
    ///
    /// * `phi` - The phi node to evaluate
    /// * `block_id` - The block containing this phi node (needed to check edge executability)
    fn evaluate_phi(&self, phi: &PhiNode, block_id: usize) -> ScalarValue<T> {
        let mut result = ScalarValue::Top;
        let mut has_executable_operand = false;

        for operand in phi.operands() {
            let pred = operand.predecessor();

            // The key SCCP insight: only consider this operand if the specific
            // edge (pred -> block_id) is executable, not just if pred is reachable.
            if !self.executable_edges.contains(&(pred, block_id)) {
                continue;
            }

            has_executable_operand = true;

            // For back edges (loop edges), treat the operand value as Bottom.
            // Back edge values represent loop-carried dependencies that change
            // across iterations. Using the first-iteration value would incorrectly
            // mark the PHI as constant when it's actually varying.
            //
            // Example: Fibonacci loop where b = phi(1, temp)
            // - First iteration: temp = 0 + 1 = 1, so b = phi(1, 1) looks constant
            // - But iteration 2: temp = 1 + 1 = 2, so b should be 2
            // Without this check, SCCP would incorrectly conclude b is always 1.
            let op_value = if self.back_edges.contains(&(pred, block_id)) {
                ScalarValue::Bottom
            } else {
                self.get_value(operand.value())
            };
            result = result.meet(&op_value);

            // Early exit if already bottom
            if result.is_bottom() {
                break;
            }
        }

        // If no operands were from executable edges, return Top (no information yet)
        if !has_executable_operand {
            return ScalarValue::Top;
        }

        result
    }

    /// Evaluates an SSA instruction to get its result value.
    ///
    /// This performs abstract interpretation of the instruction, computing
    /// what value the result would have given the current lattice values
    /// of the operands. Delegates to [`evaluate_const_op`] for arithmetic
    /// dispatch, while handling lattice Top/Bottom propagation locally.
    fn evaluate_instruction(&self, op: &SsaOp<T>) -> ScalarValue<T> {
        // Copy propagates the source's lattice value directly.
        if let SsaOp::Copy { src, .. } = op {
            return self.get_value(*src);
        }

        // Delegate arithmetic to the shared constant evaluator.
        // Track whether any operand was Top (unknown) vs Bottom (varying)
        // so the lattice result is correct.
        let mut saw_top = false;
        let ptr_size = self.pointer_size;
        let result = evaluate_const_op(
            op,
            |var| match self.get_value(var) {
                ScalarValue::Constant(c) => Some(c),
                ScalarValue::Top => {
                    saw_top = true;
                    None
                }
                ScalarValue::Bottom => None,
            },
            ptr_size,
        );

        match result {
            Some(c) => ScalarValue::Constant(c),
            // If the shared evaluator returned None but an operand was Top,
            // the result is still unknown (Top), not varying (Bottom).
            None if saw_top => ScalarValue::Top,
            None => ScalarValue::Bottom,
        }
    }

    /// Updates all definitions produced by an instruction.
    fn update_instruction_defs(&mut self, instr: &SsaInstruction<T>) {
        let primary = instr.op().dest();
        let value = self.evaluate_instruction(instr.op());
        for def in instr.defs() {
            if Some(def) == primary {
                self.update_value(def, &value);
            } else {
                self.update_value(def, &ScalarValue::Bottom);
            }
        }
    }

    /// Gets the current value of a variable.
    fn get_value(&self, var: SsaVarId) -> ScalarValue<T> {
        self.values.get(&var).cloned().unwrap_or_default()
    }

    /// Updates a variable's value and adds it to the worklist if changed.
    fn update_value(&mut self, var: SsaVarId, new_value: &ScalarValue<T>) {
        let old_value = self.values.get(&var).cloned().unwrap_or_default();

        // Apply meet to move down the lattice (values can only decrease)
        let final_value = old_value.meet(new_value);

        if final_value != old_value {
            self.values.insert(var, final_value);
            self.ssa_worklist.push_back(var);
        }
    }
}

/// Scalar value in the SCCP lattice.
///
/// This forms a simple three-level lattice:
/// - Top: No information (might be any value)
/// - Constant: Known compile-time constant
/// - Bottom: Not a constant (multiple possible values)
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ScalarValue<T: Target> {
    /// No information yet (top of lattice).
    #[default]
    Top,
    /// Known constant value.
    Constant(ConstValue<T>),
    /// Multiple possible values (bottom of lattice).
    Bottom,
}

impl<T: Target> ScalarValue<T> {
    /// Returns `true` if this is the top element.
    #[must_use]
    pub const fn is_top(&self) -> bool {
        matches!(self, Self::Top)
    }

    /// Returns `true` if this is the bottom element.
    #[must_use]
    pub const fn is_bottom(&self) -> bool {
        matches!(self, Self::Bottom)
    }

    /// Returns `true` if this is a known constant.
    #[must_use]
    pub const fn is_constant(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    /// Returns the constant value if this is a constant.
    #[must_use]
    pub const fn as_constant(&self) -> Option<&ConstValue<T>> {
        match self {
            Self::Constant(c) => Some(c),
            _ => None,
        }
    }
}

impl<T: Target> MeetSemiLattice for ScalarValue<T> {
    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            // Top meets anything yields the other
            (Self::Top, x) | (x, Self::Top) => x.clone(),

            // Same constants stay constant
            (Self::Constant(a), Self::Constant(b)) if a == b => Self::Constant(a.clone()),

            // Different constants or anything with bottom yields bottom
            _ => Self::Bottom,
        }
    }

    fn is_bottom(&self) -> bool {
        matches!(self, Self::Bottom)
    }
}

/// Results of SCCP analysis.
#[derive(Debug, Clone)]
pub struct SccpResult<T: Target> {
    /// Value for each SSA variable.
    values: BTreeMap<SsaVarId, ScalarValue<T>>,
    /// Blocks determined to be executable.
    executable_blocks: BitSet,
}

impl<T: Target> SccpResult<T> {
    /// Creates an empty SCCP result.
    ///
    /// This is useful for testing or when no analysis has been performed.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            values: BTreeMap::new(),
            executable_blocks: BitSet::new(0),
        }
    }

    /// Gets the value of an SSA variable.
    #[must_use]
    pub fn get_value(&self, var: SsaVarId) -> Option<&ScalarValue<T>> {
        self.values.get(&var)
    }

    /// Returns `true` if a variable is known to be constant.
    #[must_use]
    pub fn is_constant(&self, var: SsaVarId) -> bool {
        self.values
            .get(&var)
            .is_some_and(|v| matches!(v, ScalarValue::Constant(_)))
    }

    /// Returns the constant value of a variable if known.
    #[must_use]
    pub fn constant_value(&self, var: SsaVarId) -> Option<&ConstValue<T>> {
        self.values.get(&var).and_then(|v| match v {
            ScalarValue::Constant(c) => Some(c),
            _ => None,
        })
    }

    /// Returns `true` if a block is executable (reachable).
    ///
    /// An out-of-range `block` index (past the analyzed `block_count`) is
    /// reported `false` rather than panicking — terminator targets may be
    /// dangling and callers should treat unknown blocks as unreachable.
    #[must_use]
    pub fn is_block_executable(&self, block: usize) -> bool {
        block < self.executable_blocks.len() && self.executable_blocks.contains(block)
    }

    /// Returns an iterator over all constant variables.
    pub fn constants(&self) -> impl Iterator<Item = (SsaVarId, &ConstValue<T>)> {
        self.values.iter().filter_map(|(var, val)| match val {
            ScalarValue::Constant(c) => Some((*var, c)),
            _ => None,
        })
    }

    /// Returns an iterator over all executable blocks.
    pub fn executable_blocks(&self) -> impl Iterator<Item = usize> + '_ {
        self.executable_blocks.iter()
    }

    /// Returns the number of variables found to be constant.
    #[must_use]
    pub fn constant_count(&self) -> usize {
        self.values
            .values()
            .filter(|v| matches!(v, ScalarValue::Constant(_)))
            .count()
    }

    /// Returns the number of executable blocks.
    #[must_use]
    pub fn executable_block_count(&self) -> usize {
        self.executable_blocks.count()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    use crate::{
        analysis::{cfg::SsaCfg, dataflow::lattice::MeetSemiLattice},
        ir::{
            block::SsaBlock,
            function::SsaFunction,
            instruction::SsaInstruction,
            ops::SsaOp,
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MockTarget, MockType},
        PointerSize,
    };

    type Sv = ScalarValue<MockTarget>;
    type Cv = ConstValue<MockTarget>;

    #[test]
    fn test_scalar_value_meet() {
        // Top meets anything yields the other
        assert_eq!(
            Sv::Top.meet(&Sv::Constant(Cv::I32(5))),
            Sv::Constant(Cv::I32(5))
        );

        // Same constants stay constant
        assert_eq!(
            Sv::Constant(Cv::I32(5)).meet(&Sv::Constant(Cv::I32(5))),
            Sv::Constant(Cv::I32(5))
        );

        // Different constants become bottom
        assert_eq!(
            Sv::Constant(Cv::I32(5)).meet(&Sv::Constant(Cv::I32(10))),
            Sv::Bottom
        );

        // Bottom meets anything yields bottom
        assert_eq!(Sv::Bottom.meet(&Sv::Constant(Cv::I32(5))), Sv::Bottom);
    }

    #[test]
    fn test_scalar_value_accessors() {
        let top: Sv = Sv::Top;
        let const_val: Sv = Sv::Constant(Cv::I32(42));
        let bottom: Sv = Sv::Bottom;

        assert!(top.is_top());
        assert!(!top.is_constant());
        assert!(!top.is_bottom());

        assert!(!const_val.is_top());
        assert!(const_val.is_constant());
        assert!(!const_val.is_bottom());
        assert_eq!(const_val.as_constant(), Some(&ConstValue::I32(42)));

        assert!(!bottom.is_top());
        assert!(!bottom.is_constant());
        assert!(bottom.is_bottom());
    }

    #[test]
    fn test_sccp_result() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);

        let mut values: BTreeMap<SsaVarId, Sv> = BTreeMap::new();
        values.insert(v0, Sv::Constant(Cv::I32(42)));
        values.insert(v1, Sv::Bottom);
        values.insert(v2, Sv::Top);

        let mut executable_blocks = BitSet::new(3);
        executable_blocks.insert(0);
        executable_blocks.insert(1);

        let result = SccpResult {
            values,
            executable_blocks,
        };

        assert!(result.is_constant(v0));
        assert!(!result.is_constant(v1));
        assert!(!result.is_constant(v2));

        assert_eq!(result.constant_value(v0), Some(&ConstValue::I32(42)));
        assert_eq!(result.constant_value(v1), None);

        assert!(result.is_block_executable(0));
        assert!(result.is_block_executable(1));
        assert!(!result.is_block_executable(2));
        // Out-of-range index is reported unreachable, never a panic.
        assert!(!result.is_block_executable(9_999));

        assert_eq!(result.constant_count(), 1);
        assert_eq!(result.executable_block_count(), 2);
    }

    #[test]
    fn out_of_range_branch_targets_do_not_panic() {
        // A terminator may reference a block that was never recovered. SCCP's
        // `executable_blocks` bitset is sized to `block_count`, so an
        // out-of-range target index used to panic in `BitSet::contains`. The
        // analysis must instead treat it as unreachable. The condition is an
        // unconstrained argument (`Bottom`), so both edges — including the
        // dangling one — are explored.
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let cond = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            // Block 99 does not exist — only blocks 0 and 1 are present.
            false_target: 99,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let cfg = SsaCfg::from_ssa(&ssa);
        let mut sccp: ConstantPropagation<MockTarget> =
            ConstantPropagation::new(PointerSize::Bit64);
        let result = sccp.analyze(&ssa, &cfg);

        // Real blocks reachable, the dangling target never marked executable.
        assert!(result.is_block_executable(0));
        assert!(result.is_block_executable(1));
        assert!(!result.is_block_executable(99));
    }
}
