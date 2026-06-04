//! PHI node analysis and placement utilities.
//!
//! This module provides utilities for PHI nodes in SSA form, including analysis
//! of trivial PHIs, uniform constant detection, and pruned phi placement.
//!
//! # Analyses
//!
//! **Trivial PHI detection**: A PHI is trivial if all non-self-referencing operands
//! point to the same source variable. Such PHIs can be replaced with a simple copy
//! operation. A PHI that is fully self-referential (all operands are the result
//! itself) indicates unreachable code and can be removed entirely.
//!
//! **Uniform constant detection**: A PHI is uniform if all its operands resolve
//! to the same constant value. Such PHIs can be replaced with a constant assignment.
//!
//! # Pruned Phi Placement Algorithm
//!
//! The `place_pruned_phis` function implements the standard iterated dominance
//! frontier (IDF) algorithm from Cytron et al., with liveness-based pruning:
//!
//! 1. **Iterated dominance frontier**: For each variable group, compute the IDF
//!    of its definition blocks by iteratively adding frontier blocks until a
//!    fixed point is reached. This identifies all blocks where a phi may be needed.
//!
//! 2. **Liveness pruning**: For each candidate phi block, check if the variable
//!    is live-in to that block (using pre-computed `live_in` sets). If not live,
//!    the phi is omitted (pruned SSA). This avoids dead-on-arrival phi nodes.
//!
//! 3. **Phi insertion**: For each surviving candidate, create a `PhiNode` with
//!    a placeholder result ID and the appropriate `VariableOrigin`.
//!
//! # Exception Handler Support
//!
//! For exception handler blocks, Leave targets are also considered as phi placement
//! points (handler exits merge with normal flow). The `leave_target_fn` callback
//! provides Leave target resolution specific to the host CIL format.
//!
//! # Complexity
//!
//! Phi analysis: O(P) per phi where P is the number of operands.
//! Phi placement: O(G * F) where G is the number of variable groups and F is the
//! iterated dominance frontier size. Liveness pruning adds O(G * B) overhead.
//!
//! # Example
//!
//! ```rust,ignore
//! use analyssa::analysis::{ConstEvaluator, PhiAnalyzer};
//!
//! let analyzer = PhiAnalyzer::new(&ssa);
//!
//! // Check if a PHI is trivial (has single unique non-self source)
//! if let Some(source) = analyzer.is_trivial(phi) {
//!     println!("PHI can be replaced with copy from {:?}", source);
//! }
//!
//! // Check if all PHI operands resolve to the same constant
//! let mut evaluator = ConstEvaluator::new(&ssa, PointerSize::Bit64);
//! if let Some(value) = analyzer.uniform_constant(phi, &mut evaluator) {
//!     println!("PHI always produces: {:?}", value);
//! }
//! ```

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    analysis::consts::ConstEvaluator,
    graph::NodeId,
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{SsaVarId, VariableOrigin},
    },
    target::Target,
    BitSet,
};

/// Analyzes PHI nodes for various patterns.
///
/// This struct provides methods for common PHI node analysis tasks:
/// - Detecting trivial PHIs that can be replaced with copies
/// - Finding PHIs where all operands resolve to the same constant
/// - Looking up PHI operands by predecessor block
/// - Finding the PHI node that defines a variable
pub struct PhiAnalyzer<'a, T: Target> {
    /// Reference to the SSA function being analyzed.
    ssa: &'a SsaFunction<T>,
}

impl<'a, T: Target> PhiAnalyzer<'a, T> {
    /// Creates a new PHI analyzer for the given SSA function.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to analyze.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>) -> Self {
        Self { ssa }
    }

    /// Returns a reference to the SSA function being analyzed.
    #[must_use]
    pub fn ssa(&self) -> &SsaFunction<T> {
        self.ssa
    }

    /// Checks if a PHI is trivial (has a single unique non-self source).
    ///
    /// A trivial PHI can be replaced with a simple copy operation.
    /// This occurs when all non-self-referential operands point to the
    /// same source variable.
    ///
    /// # Arguments
    ///
    /// * `phi` - The PHI node to analyze.
    ///
    /// # Returns
    ///
    /// `Some(source)` if the PHI has exactly one unique non-self source,
    /// `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```text
    /// // Trivial PHI (can be replaced with: result = v1)
    /// result = phi(v1, v1, result)  // Returns Some(v1)
    ///
    /// // Non-trivial PHI (multiple different sources)
    /// result = phi(v1, v2)  // Returns None
    ///
    /// // Non-trivial PHI (only self-references, unreachable)
    /// result = phi(result, result)  // Returns None
    /// ```
    #[must_use]
    pub fn is_trivial(&self, phi: &PhiNode) -> Option<SsaVarId> {
        let result = phi.result();

        // Collect non-self-referential operands
        let unique_sources: BTreeSet<SsaVarId> = phi
            .operands()
            .iter()
            .map(PhiOperand::value)
            .filter(|&v| v != result)
            .collect();

        // Trivial if exactly one unique non-self source
        if unique_sources.len() == 1 {
            let source = unique_sources.into_iter().next()?;

            // Check if replacing result with source would create a self-referential instruction.
            if self.ssa.would_create_self_reference(source, result) {
                return None;
            }

            Some(source)
        } else {
            None
        }
    }

    /// Checks if a PHI is fully self-referential (all operands reference the PHI's result).
    ///
    /// A fully self-referential PHI indicates unreachable code or undefined behavior,
    /// since there's no external value entering the PHI. Such PHIs can be safely removed.
    ///
    /// # Arguments
    ///
    /// * `phi` - The PHI node to analyze.
    ///
    /// # Returns
    ///
    /// `true` if all operands reference the PHI's own result variable, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```text
    /// // Fully self-referential (returns true)
    /// result = phi(result, result)
    ///
    /// // Not fully self-referential (returns false)
    /// result = phi(v1, result)
    /// result = phi(v1, v2)
    /// ```
    #[must_use]
    pub fn is_fully_self_referential(&self, phi: &PhiNode) -> bool {
        let result = phi.result();
        !phi.operands().is_empty() && phi.operands().iter().all(|op| op.value() == result)
    }

    /// Analyzes a PHI to determine its trivial status.
    ///
    /// This is the comprehensive analysis method that distinguishes between:
    /// - Trivial PHIs with a single replacement value
    /// - Fully self-referential PHIs that should be removed
    /// - Non-trivial PHIs that must be kept
    ///
    /// # Arguments
    ///
    /// * `phi` - The PHI node to analyze.
    ///
    /// # Returns
    ///
    /// - `Some(Some(var))` - PHI is trivial, can be replaced with `var`
    /// - `Some(None)` - PHI is fully self-referential, can be removed
    /// - `None` - PHI is not trivial, must be kept
    #[must_use]
    pub fn analyze_trivial(&self, phi: &PhiNode) -> Option<Option<SsaVarId>> {
        // Check if trivial with a replacement value
        if let Some(source) = self.is_trivial(phi) {
            return Some(Some(source));
        }

        // Check if fully self-referential (can be removed)
        if self.is_fully_self_referential(phi) {
            return Some(None);
        }

        // Not trivial
        None
    }

    /// Finds all trivial PHI nodes in the SSA function.
    ///
    /// Scans all reachable blocks for PHI nodes that are either:
    /// - Trivial with a single replacement value
    /// - Fully self-referential and can be removed
    ///
    /// # Arguments
    ///
    /// * `reachable` - Set of reachable block indices to scan.
    ///
    /// # Returns
    ///
    /// A vector of `(block_idx, phi_idx, replacement)` tuples where:
    /// - `replacement = Some(var)` - PHI can be replaced with `var`
    /// - `replacement = None` - PHI is fully self-referential and can be removed
    #[must_use]
    pub fn find_all_trivial(
        &self,
        reachable: &BTreeSet<usize>,
    ) -> Vec<(usize, usize, Option<SsaVarId>)> {
        let mut trivial = Vec::new();

        for &block_idx in reachable {
            if let Some(block) = self.ssa.block(block_idx) {
                for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                    if let Some(replacement) = self.analyze_trivial(phi) {
                        trivial.push((block_idx, phi_idx, replacement));
                    }
                }
            }
        }

        trivial
    }

    /// Collects all copy-like operations in the SSA function.
    ///
    /// This method identifies all operations that are effectively copies:
    /// - Explicit `Copy` instructions: `dest = copy src`
    /// - Trivial phi nodes: `dest = phi(src, src, ...)` where all non-self operands are identical
    ///
    /// This is the unified entry point for copy detection, used by copy propagation
    /// and other optimizations that need to identify copy relationships.
    ///
    /// # Returns
    ///
    /// A map from each copy destination to its immediate source.
    ///
    /// # Example
    ///
    /// ```text
    /// // Given:
    /// v1 = copy v0           // Explicit copy
    /// v2 = phi(v0, v0)       // Trivial phi (all same source)
    /// v3 = phi(v0, v3)       // Trivial phi (self-ref excluded)
    /// v4 = phi(v0, v1)       // Non-trivial (different sources)
    ///
    /// // Returns: {v1 → v0, v2 → v0, v3 → v0}
    /// ```
    #[must_use]
    pub fn collect_all_copies(&self) -> BTreeMap<SsaVarId, SsaVarId> {
        let mut copies = BTreeMap::new();

        for block in self.ssa.blocks() {
            // Collect explicit copy instructions
            for instr in block.instructions() {
                if let SsaOp::Copy { dest, src } = instr.op() {
                    copies.insert(*dest, *src);
                }
            }

            // Collect trivial phi nodes (effectively copies)
            for phi in block.phi_nodes() {
                if let Some(source) = self.is_trivial(phi) {
                    copies.insert(phi.result(), source);
                }
            }
        }

        copies
    }

    /// Checks if all PHI operands resolve to the same constant.
    ///
    /// This is useful for detecting PHIs that always produce the same value,
    /// which can be replaced with a constant assignment.
    ///
    /// # Arguments
    ///
    /// * `phi` - The PHI node to analyze.
    /// * `evaluator` - A constant evaluator for resolving operand values.
    ///
    /// # Returns
    ///
    /// `Some(value)` if all operands evaluate to the same constant,
    /// `None` if operands differ, cannot be evaluated, or PHI is empty.
    ///
    /// # Examples
    ///
    /// ```text
    /// // Given: v1 = 42, v2 = 42
    /// result = phi(v1, v2)  // Returns Some(42)
    ///
    /// // Given: v1 = 42, v2 = 99
    /// result = phi(v1, v2)  // Returns None (values differ)
    ///
    /// // Given: v1 = 42, v2 = unknown
    /// result = phi(v1, v2)  // Returns None (v2 not constant)
    /// ```
    pub fn uniform_constant(
        &self,
        phi: &PhiNode,
        evaluator: &mut ConstEvaluator<'_, T>,
    ) -> Option<ConstValue<T>> {
        let operands = phi.operands();

        // Empty PHI has no uniform value
        if operands.is_empty() {
            return None;
        }

        // Get the first operand's constant value
        let first_value = evaluator.evaluate_var(operands.first()?.value())?;

        // Check that all other operands have the same value
        for operand in operands.iter().skip(1) {
            let value = evaluator.evaluate_var(operand.value())?;
            if value != first_value {
                return None;
            }
        }

        Some(first_value)
    }

    /// Finds the PHI node that defines a variable.
    ///
    /// This delegates to [`SsaFunction::find_phi_defining`] for the actual lookup,
    /// which uses O(1) lookup via the variable's definition site when available.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable ID to find the defining PHI for.
    ///
    /// # Returns
    ///
    /// `Some((block_idx, &PhiNode))` if the variable is defined by a PHI node,
    /// `None` if the variable is not defined by a PHI or doesn't exist.
    #[must_use]
    pub fn find_phi_defining(&self, var: SsaVarId) -> Option<(usize, &PhiNode)> {
        self.ssa.find_phi_defining(var)
    }
}

/// Callback type for resolving Leave targets in exception handler blocks.
pub(crate) type LeaveTargetFn<'a, T> = dyn Fn(usize, &[SsaBlock<T>]) -> Option<usize> + 'a;

/// Input sets and callbacks used when placing pruned phi nodes.
pub struct PhiPlacementConfig<'a, T: Target> {
    /// Definition sites for each rename group.
    pub defs: &'a BTreeMap<u32, BitSet>,
    /// Live-in sets for each rename group.
    pub live_in: &'a BTreeMap<u32, BitSet>,
    /// Dominance frontier sets indexed by block.
    pub dominance_frontiers: &'a [BitSet],
    /// Reachable block set, or `None` to treat every block as reachable.
    pub reachable: Option<&'a BitSet>,
    /// Predicate that selects rename groups to process.
    pub group_filter: &'a dyn Fn(u32) -> bool,
    /// Maps a rename group to the origin used for inserted phi nodes.
    pub group_to_origin: &'a dyn Fn(u32) -> VariableOrigin,
    /// Optional callback that maps `leave` blocks to exceptional exit targets.
    pub leave_target_fn: Option<&'a LeaveTargetFn<'a, T>>,
}

/// Places phi nodes at iterated dominance frontier blocks, pruned by liveness.
///
/// This implements the standard IDF phi placement algorithm from Cytron et al.,
/// with pruning based on liveness analysis (only placing phis where the variable
/// is live-in).
///
/// Data structures are keyed by `u32` group IDs. The `group_to_origin` mapping
/// translates group IDs to `VariableOrigin` values for the created phi nodes.
///
/// # Returns
///
/// A list of `(block_idx, group)` pairs for each phi node placed, in the order
/// they were added to each block. This allows callers to associate phi nodes
/// with their rename groups during the rename phase.
pub fn place_pruned_phis<T: Target>(
    blocks: &mut [SsaBlock<T>],
    config: PhiPlacementConfig<'_, T>,
) -> Vec<(usize, u32)> {
    let PhiPlacementConfig {
        defs,
        live_in,
        dominance_frontiers,
        reachable,
        group_filter,
        group_to_origin,
        leave_target_fn,
    } = config;
    let block_count = blocks.len();
    let mut placements: Vec<(usize, u32)> = Vec::new();

    for (&group, def_blocks) in defs {
        if !group_filter(group) {
            continue;
        }

        // Compute iterated dominance frontier
        let mut phi_blocks = BitSet::new(block_count);
        let mut worklist: Vec<usize> = def_blocks.iter().collect();

        while let Some(block_idx) = worklist.pop() {
            let node_id = NodeId::new(block_idx);
            if let Some(frontier) = dominance_frontiers.get(node_id.index()) {
                for frontier_idx in frontier.iter() {
                    let is_reachable = reachable.is_none_or(|r| r.contains(frontier_idx));
                    if frontier_idx < block_count && is_reachable && phi_blocks.insert(frontier_idx)
                    {
                        worklist.push(frontier_idx);
                    }
                }
            }

            // For exception handler blocks, use Leave targets as phi placement points
            if let Some(leave_fn) = leave_target_fn {
                if let Some(target) = leave_fn(block_idx, blocks) {
                    let is_reachable = reachable.is_none_or(|r| r.contains(target));
                    if target < block_count && is_reachable && phi_blocks.insert(target) {
                        worklist.push(target);
                    }
                }
            }
        }

        // Pruned SSA: only place phi if variable is live at the frontier block.
        // If no liveness data for this group, place unconditionally (used by
        // converter for Stack origins that don't have liveness tracking).
        let group_live_in = live_in.get(&group);

        for phi_block_idx in phi_blocks.iter() {
            if let Some(live_set) = group_live_in {
                if !live_set.contains(phi_block_idx) {
                    continue;
                }
            }

            if let Some(block) = blocks.get_mut(phi_block_idx) {
                let origin = group_to_origin(group);
                // Phi result IDs are temporary placeholders; they will be replaced
                // during the rename phase with properly allocated variable IDs.
                let phi = PhiNode::new(SsaVarId::PLACEHOLDER, origin);
                block.add_phi(phi);
                placements.push((phi_block_idx, group));
            }
        }
    }

    placements
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        ir::{
            function::SsaFunction,
            phi::{PhiNode, PhiOperand},
            variable::{SsaVarId, VariableOrigin},
        },
        testing::MockTarget,
    };

    #[test]
    fn trivial_phi_single_operand() {
        let ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let analyzer = PhiAnalyzer::new(&ssa);
        let result = SsaVarId::from_index(0);
        let v0 = SsaVarId::from_index(1);
        let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(v0, 0));
        assert_eq!(analyzer.is_trivial(&phi), Some(v0));
    }

    #[test]
    fn trivial_phi_all_same_operands() {
        let ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let analyzer = PhiAnalyzer::new(&ssa);
        let result = SsaVarId::from_index(0);
        let v0 = SsaVarId::from_index(1);
        let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(v0, 0));
        phi.add_operand(PhiOperand::new(v0, 1));
        phi.add_operand(PhiOperand::new(v0, 2));
        assert_eq!(analyzer.is_trivial(&phi), Some(v0));
    }

    #[test]
    fn trivial_phi_with_self_references() {
        // phi(self, v0, self) — self-references ignored, v0 is the trivial source.
        let ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let analyzer = PhiAnalyzer::new(&ssa);
        let result = SsaVarId::from_index(0);
        let v0 = SsaVarId::from_index(1);
        let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(result, 0)); // self-ref
        phi.add_operand(PhiOperand::new(v0, 1));
        phi.add_operand(PhiOperand::new(result, 2)); // self-ref
        assert_eq!(analyzer.is_trivial(&phi), Some(v0));
    }

    #[test]
    fn non_trivial_phi_different_operands() {
        let ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let analyzer = PhiAnalyzer::new(&ssa);
        let result = SsaVarId::from_index(0);
        let v0 = SsaVarId::from_index(1);
        let v1 = SsaVarId::from_index(2);
        let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(v0, 0));
        phi.add_operand(PhiOperand::new(v1, 1));
        assert_eq!(analyzer.is_trivial(&phi), None);
    }

    #[test]
    fn trivial_phi_all_self_references() {
        // phi(self, self) → undefined / unreachable; reported via analyze_trivial as Some(None).
        let ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let analyzer = PhiAnalyzer::new(&ssa);
        let result = SsaVarId::from_index(0);
        let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(result, 0));
        phi.add_operand(PhiOperand::new(result, 1));
        assert!(analyzer.is_trivial(&phi).is_none());
        assert_eq!(analyzer.analyze_trivial(&phi), Some(None));
    }
}
