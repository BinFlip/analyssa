//! Generic taint analysis for SSA functions.
//!
//! This module provides a reusable taint analysis framework that propagates
//! taint information through SSA variables and instructions. It supports
//! forward propagation (input taint flows to outputs), backward propagation
//! (output taint flows to inputs), and configurable PHI handling.
//!
//! # Algorithm
//!
//! Taint propagation runs iteratively to a fixpoint:
//!
//! 1. **Initialization**: Taint sources are added via `add_tainted_var`,
//!    `add_tainted_instr`, or `add_tainted_phi`.
//!
//! 2. **PHI propagation**: For each PHI node, taint flows according to the
//!    configured [`PhiTaintMode`]:
//!    - `TaintAllOperands`: If PHI result is tainted, ALL operands become tainted
//!    - `TaintIfAnyOperand`: If ANY operand is tainted, the result becomes tainted
//!    - `TaintFromPredecessors`: Only taint operands from specific predecessors
//!    - `SelectivePhi`: For CFF analysis, selectively traces through PHI chains
//!      matching an origin filter
//!    - `NoPropagation`: PHI nodes act as taint barriers
//!
//! 3. **Instruction propagation**:
//!    - Forward: If any USE is tainted, the DEF becomes tainted
//!    - Backward: If the DEF is tainted, all USEs become tainted
//!    - Array-aware: If an array is tainted, `StoreElement` operations to that
//!      array are also tainted (critical for cleanup neutralization)
//!
//! 4. **Termination**: Repeats until no changes or `max_iterations` is reached.
//!
//! # PHI Taint Modes for CFF Analysis
//!
//! For control flow flattening analysis, the `SelectivePhi` mode is specifically
//! designed to trace state values back through dispatcher PHI chains:
//! - Only follows PHIs whose `VariableOrigin` matches the state variable origin
//! - Only taints operands from blocks that jump to the dispatcher
//! - This prevents over-tainting loop counters and other non-state variables
//!
//! # Complexity
//!
//! O(I * (V + P)) where I is iterations to fixpoint (bounded by max_iterations),
//! V is the number of variables, and P is the number of PHI nodes.
//!
//! # Use Cases
//!
//! - **CFF Unflattening**: Track state variables to identify dispatcher machinery
//! - **Cleanup Neutralization**: Identify instructions dependent on removed tokens
//! - **Security Analysis**: Track data flow from untrusted sources
//!
//! # Example
//!
//! ```rust,ignore
//! use analyssa::analysis::taint::{PhiTaintMode, TaintAnalysis, TaintConfig};
//! use analyssa::{ir::SsaFunction, MockTarget};
//!
//! let ssa: SsaFunction<MockTarget> = /* ... */;
//!
//! let config = TaintConfig {
//!     forward: true,
//!     backward: true,
//!     phi_mode: PhiTaintMode::TaintAllOperands,
//!     max_iterations: 100,
//! };
//!
//! let mut taint = TaintAnalysis::new(config);
//! taint.add_tainted_var(some_var_id);
//! taint.propagate(&ssa);
//!
//! // Check what's tainted
//! if taint.is_var_tainted(other_var_id) {
//!     println!("Variable is tainted!");
//! }
//! ```

use std::collections::HashSet;

use crate::{
    ir::{
        function::SsaFunction,
        ops::SsaOp,
        variable::{SsaVarId, VariableOrigin},
    },
    target::Target,
};

/// How to handle PHI nodes during taint propagation.
///
/// PHI nodes are control flow merge points where values from different
/// predecessors come together. The taint mode determines how taint
/// flows through these merge points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhiTaintMode {
    /// If the PHI result is tainted, all operands become tainted.
    ///
    /// Use this for backward analysis where you need to find all sources
    /// that could contribute to a tainted value.
    TaintAllOperands,

    /// If any operand is tainted, the PHI result becomes tainted.
    ///
    /// Use this for forward analysis where taint should flow from any
    /// predecessor path.
    TaintIfAnyOperand,

    /// Only taint operands from specific predecessor blocks.
    ///
    /// Use this for path-sensitive analysis where only certain control
    /// flow paths should propagate taint.
    TaintFromPredecessors(HashSet<usize>),

    /// Selective backward taint through PHI chains for CFF analysis.
    ///
    /// This mode is specifically designed for control flow flattening (CFF)
    /// analysis where we need to trace state values back through PHI chains.
    ///
    /// When a PHI result is tainted:
    /// - Check if the PHI's origin matches (if origin filter is Some)
    /// - Only taint operands from predecessors in the set
    /// - Recursively trace through intermediate PHIs with the same origin
    SelectivePhi {
        /// Set of predecessor blocks whose operands should be tainted.
        /// For CFF, this is typically the set of blocks that jump to the dispatcher.
        predecessors: HashSet<usize>,
        /// Optional `VariableOrigin` to filter PHI chains.
        /// Only PHIs with matching origin will be traversed.
        origin_filter: Option<VariableOrigin>,
    },

    /// Don't propagate taint through PHI nodes.
    ///
    /// Use this when PHIs represent control flow merge points that
    /// should act as taint barriers.
    NoPropagation,
}

/// Configuration for taint analysis.
///
/// Controls how taint propagates through the SSA graph.
#[derive(Debug, Clone)]
pub struct TaintConfig {
    /// Propagate forward (input tainted → output tainted).
    ///
    /// When enabled, if an instruction uses a tainted variable, its
    /// defined variable (if any) becomes tainted.
    pub forward: bool,

    /// Propagate backward (output tainted → inputs tainted).
    ///
    /// When enabled, if an instruction's defined variable is tainted,
    /// all variables it uses become tainted.
    pub backward: bool,

    /// How to handle PHI nodes.
    pub phi_mode: PhiTaintMode,

    /// Maximum iterations for fixpoint computation.
    ///
    /// Prevents infinite loops in pathological cases.
    pub max_iterations: usize,
}

impl Default for TaintConfig {
    fn default() -> Self {
        Self {
            forward: true,
            backward: false,
            phi_mode: PhiTaintMode::TaintIfAnyOperand,
            max_iterations: 100,
        }
    }
}

impl TaintConfig {
    /// Creates a config for forward-only propagation.
    ///
    /// Suitable for tracking what variables depend on a taint source.
    #[must_use]
    pub fn forward_only() -> Self {
        Self {
            forward: true,
            backward: false,
            phi_mode: PhiTaintMode::TaintIfAnyOperand,
            max_iterations: 100,
        }
    }

    /// Creates a config for bidirectional propagation.
    ///
    /// Suitable for cleanup neutralization where we need to find all
    /// instructions connected to removed tokens.
    #[must_use]
    pub fn bidirectional() -> Self {
        Self {
            forward: true,
            backward: true,
            phi_mode: PhiTaintMode::TaintAllOperands,
            max_iterations: 100,
        }
    }
}

/// Statistics about taint analysis execution.
///
/// Reports the number of iterations required to reach a fixpoint and the
/// counts of tainted variables, instructions, and PHI nodes discovered.
/// These statistics help tune the analysis parameters (particularly
/// `max_iterations`) and assess the scope of taint propagation.
#[derive(Debug, Clone, Default)]
pub struct TaintStats {
    /// Number of fixpoint iterations performed (bounded by `TaintConfig::max_iterations`).
    pub iterations: usize,
    /// Number of SSA variables found to be tainted by the analysis.
    pub tainted_vars: usize,
    /// Number of instructions found to be tainted by the analysis.
    pub tainted_instrs: usize,
    /// Number of PHI nodes found to be tainted by the analysis.
    pub tainted_phis: usize,
}

/// Generic taint analysis for SSA functions.
///
/// This struct tracks which variables and instructions are "tainted" - meaning
/// they are connected to some set of taint sources through data flow.
///
/// The analysis runs to a fixpoint, propagating taint through the SSA graph
/// according to the configuration.
#[derive(Debug, Clone)]
pub struct TaintAnalysis {
    /// Tainted SSA variables.
    tainted_vars: HashSet<SsaVarId>,

    /// Tainted instructions: (block_idx, instr_idx).
    tainted_instrs: HashSet<(usize, usize)>,

    /// Tainted PHI nodes: (block_idx, phi_idx).
    tainted_phis: HashSet<(usize, usize)>,

    /// Configuration.
    config: TaintConfig,

    /// Statistics from the last propagation.
    stats: TaintStats,
}

impl TaintAnalysis {
    /// Creates a new taint analysis with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration controlling propagation behavior.
    ///
    /// # Returns
    ///
    /// A new `TaintAnalysis` with empty taint sets.
    #[must_use]
    pub fn new(config: TaintConfig) -> Self {
        Self {
            tainted_vars: HashSet::new(),
            tainted_instrs: HashSet::new(),
            tainted_phis: HashSet::new(),
            config,
            stats: TaintStats::default(),
        }
    }

    /// Creates a taint analysis with default forward-only configuration.
    #[must_use]
    pub fn forward_only() -> Self {
        Self::new(TaintConfig::forward_only())
    }

    /// Creates a taint analysis with bidirectional configuration.
    #[must_use]
    pub fn bidirectional() -> Self {
        Self::new(TaintConfig::bidirectional())
    }

    /// Adds a variable as a taint source.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to mark as tainted.
    pub fn add_tainted_var(&mut self, var: SsaVarId) {
        self.tainted_vars.insert(var);
    }

    /// Adds multiple variables as taint sources.
    ///
    /// # Arguments
    ///
    /// * `vars` - Iterator of variable IDs to mark as tainted.
    pub fn add_tainted_vars(&mut self, vars: impl IntoIterator<Item = SsaVarId>) {
        self.tainted_vars.extend(vars);
    }

    /// Adds an instruction as a taint source.
    ///
    /// Also taints the instruction's defined variable (if any) and its uses
    /// (for backward propagation from instructions without defs like stores).
    ///
    /// # Arguments
    ///
    /// * `block` - Block index containing the instruction.
    /// * `instr` - Instruction index within the block.
    /// * `ssa` - The SSA function for looking up the instruction's def/uses.
    pub fn add_tainted_instr<T: Target>(
        &mut self,
        block: usize,
        instr: usize,
        ssa: &SsaFunction<T>,
    ) {
        self.tainted_instrs.insert((block, instr));

        if let Some(block_data) = ssa.block(block) {
            if let Some(instruction) = block_data.instructions().get(instr) {
                // Taint the instruction's defined variable (for forward propagation)
                for def in instruction.defs() {
                    self.tainted_vars.insert(def);
                }

                // Also taint the instruction's uses (for backward propagation).
                // This is critical for instructions like StoreStaticField that have
                // no def - we need to taint what feeds into them.
                if self.config.backward {
                    instruction.op().for_each_use(|use_var| {
                        self.tainted_vars.insert(use_var);
                    });
                }
            }
        }
    }

    /// Adds a PHI node as a taint source.
    ///
    /// Also taints the PHI's result variable.
    ///
    /// # Arguments
    ///
    /// * `block` - Block index containing the PHI.
    /// * `phi_idx` - PHI index within the block.
    /// * `ssa` - The SSA function for looking up the PHI's result.
    pub fn add_tainted_phi<T: Target>(
        &mut self,
        block: usize,
        phi_idx: usize,
        ssa: &SsaFunction<T>,
    ) {
        self.tainted_phis.insert((block, phi_idx));

        // Also taint the PHI's result variable
        if let Some(block_data) = ssa.block(block) {
            if let Some(phi) = block_data.phi_nodes().get(phi_idx) {
                self.tainted_vars.insert(phi.result());
            }
        }
    }

    /// Runs taint propagation to fixpoint.
    ///
    /// Iteratively propagates taint through the SSA graph until no more
    /// changes occur or the iteration limit is reached.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to analyze.
    pub fn propagate<T: Target>(&mut self, ssa: &SsaFunction<T>) {
        let mut iterations: usize = 0;

        loop {
            if iterations >= self.config.max_iterations {
                break;
            }
            iterations = iterations.saturating_add(1);

            let mut changed = false;

            // Process PHI nodes first
            changed |= self.propagate_phis(ssa);

            // Process instructions
            changed |= self.propagate_instructions(ssa);

            if !changed {
                break;
            }
        }

        // Update statistics
        self.stats = TaintStats {
            iterations,
            tainted_vars: self.tainted_vars.len(),
            tainted_instrs: self.tainted_instrs.len(),
            tainted_phis: self.tainted_phis.len(),
        };
    }

    /// Propagates taint through PHI nodes.
    ///
    /// Returns `true` if any changes were made.
    fn propagate_phis<T: Target>(&mut self, ssa: &SsaFunction<T>) -> bool {
        let mut changed = false;

        for (block_idx, block) in ssa.blocks().iter().enumerate() {
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                let result = phi.result();
                let result_tainted = self.tainted_vars.contains(&result);

                match &self.config.phi_mode {
                    PhiTaintMode::TaintAllOperands => {
                        // If result is tainted, all operands become tainted
                        if result_tainted {
                            for operand in phi.operands() {
                                if self.tainted_vars.insert(operand.value()) {
                                    changed = true;
                                }
                            }
                            if self.tainted_phis.insert((block_idx, phi_idx)) {
                                changed = true;
                            }
                        }
                    }
                    PhiTaintMode::TaintIfAnyOperand => {
                        // If any operand is tainted, result becomes tainted
                        let any_operand_tainted = phi
                            .operands()
                            .iter()
                            .any(|op| self.tainted_vars.contains(&op.value()));

                        if any_operand_tainted {
                            if self.tainted_vars.insert(result) {
                                changed = true;
                            }
                            if self.tainted_phis.insert((block_idx, phi_idx)) {
                                changed = true;
                            }
                        }
                    }
                    PhiTaintMode::TaintFromPredecessors(preds) => {
                        // Only taint operands from specific predecessors
                        if result_tainted {
                            for operand in phi.operands() {
                                if preds.contains(&operand.predecessor())
                                    && self.tainted_vars.insert(operand.value())
                                {
                                    changed = true;
                                }
                            }
                            if self.tainted_phis.insert((block_idx, phi_idx)) {
                                changed = true;
                            }
                        }
                    }
                    PhiTaintMode::SelectivePhi {
                        predecessors,
                        origin_filter,
                    } => {
                        // Selective backward taint for CFF analysis
                        if result_tainted {
                            // Check if this PHI's origin matches the filter
                            let should_follow = origin_filter
                                .as_ref()
                                .is_none_or(|filter| phi.origin() == *filter);

                            if should_follow {
                                for operand in phi.operands() {
                                    if predecessors.contains(&operand.predecessor())
                                        && self.tainted_vars.insert(operand.value())
                                    {
                                        changed = true;
                                    }
                                }
                                if self.tainted_phis.insert((block_idx, phi_idx)) {
                                    changed = true;
                                }
                            }
                        }
                    }
                    PhiTaintMode::NoPropagation => {
                        // Don't propagate through PHIs
                    }
                }
            }
        }

        changed
    }

    /// Propagates taint through instructions.
    ///
    /// Returns `true` if any changes were made.
    fn propagate_instructions<T: Target>(&mut self, ssa: &SsaFunction<T>) -> bool {
        let mut changed = false;

        // Reused across instructions so each instruction does not allocate fresh
        // def/use vectors (these are read several times per instruction below).
        let mut defs: Vec<SsaVarId> = Vec::new();
        let mut uses: Vec<SsaVarId> = Vec::new();

        for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
            defs.clear();
            defs.extend(instr.op().defs());
            uses.clear();
            instr.op().for_each_use(|u| uses.push(u));

            // Forward propagation: if any USE is tainted, DEF becomes tainted
            if self.config.forward {
                for def_var in &defs {
                    let uses_tainted = uses.iter().any(|u| self.tainted_vars.contains(u));
                    if uses_tainted {
                        if self.tainted_vars.insert(*def_var) {
                            changed = true;
                        }
                        if self.tainted_instrs.insert((block_idx, instr_idx)) {
                            changed = true;
                        }
                    }
                }
            }

            // Backward propagation: if DEF is tainted, all USEs become tainted
            if self.config.backward {
                let def_tainted = defs.iter().any(|d| self.tainted_vars.contains(d));
                if def_tainted {
                    for use_var in &uses {
                        if self.tainted_vars.insert(*use_var) {
                            changed = true;
                        }
                    }
                    if self.tainted_instrs.insert((block_idx, instr_idx)) {
                        changed = true;
                    }
                }
            }

            // Array-aware propagation: if an array is tainted, all StoreElement
            // operations to that array are also tainted (they're preparing dead data).
            // This is critical for cleanup neutralization where protection code fills
            // arrays that are passed to removed methods.
            if self.config.backward {
                if let SsaOp::StoreElement { array, .. } = instr.op() {
                    if self.tainted_vars.contains(array)
                        && self.tainted_instrs.insert((block_idx, instr_idx))
                    {
                        changed = true;
                        // Also taint the value and index being stored - they feed into dead code
                        for use_var in &uses {
                            if self.tainted_vars.insert(*use_var) {
                                changed = true;
                            }
                        }
                    }
                }
            }

            // Mark instruction as tainted if it uses tainted vars (even without def)
            let uses_tainted = uses.iter().any(|u| self.tainted_vars.contains(u));
            if uses_tainted && self.tainted_instrs.insert((block_idx, instr_idx)) {
                changed = true;
            }
        }

        changed
    }

    /// Checks if a variable is tainted.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to check.
    ///
    /// # Returns
    ///
    /// `true` if the variable is tainted.
    #[must_use]
    pub fn is_var_tainted(&self, var: SsaVarId) -> bool {
        self.tainted_vars.contains(&var)
    }

    /// Checks if an instruction is tainted.
    ///
    /// # Arguments
    ///
    /// * `block` - Block index.
    /// * `instr` - Instruction index within the block.
    ///
    /// # Returns
    ///
    /// `true` if the instruction is tainted.
    #[must_use]
    pub fn is_instr_tainted(&self, block: usize, instr: usize) -> bool {
        self.tainted_instrs.contains(&(block, instr))
    }

    /// Checks if a PHI node is tainted.
    ///
    /// # Arguments
    ///
    /// * `block` - Block index.
    /// * `phi_idx` - PHI index within the block.
    ///
    /// # Returns
    ///
    /// `true` if the PHI is tainted.
    #[must_use]
    pub fn is_phi_tainted(&self, block: usize, phi_idx: usize) -> bool {
        self.tainted_phis.contains(&(block, phi_idx))
    }

    /// Returns all tainted variables.
    #[must_use]
    pub fn tainted_variables(&self) -> &HashSet<SsaVarId> {
        &self.tainted_vars
    }

    /// Returns all tainted instructions.
    #[must_use]
    pub fn tainted_instructions(&self) -> &HashSet<(usize, usize)> {
        &self.tainted_instrs
    }

    /// Returns all tainted PHI nodes.
    #[must_use]
    pub fn tainted_phis(&self) -> &HashSet<(usize, usize)> {
        &self.tainted_phis
    }

    /// Returns statistics from the last propagation.
    #[must_use]
    pub fn stats(&self) -> &TaintStats {
        &self.stats
    }

    /// Returns the number of tainted variables.
    #[must_use]
    pub fn tainted_var_count(&self) -> usize {
        self.tainted_vars.len()
    }

    /// Returns the number of tainted instructions.
    #[must_use]
    pub fn tainted_instr_count(&self) -> usize {
        self.tainted_instrs.len()
    }

    /// Clears all taint information.
    pub fn clear(&mut self) {
        self.tainted_vars.clear();
        self.tainted_instrs.clear();
        self.tainted_phis.clear();
        self.stats = TaintStats::default();
    }
}

/// Finds all blocks that have a direct jump/branch to the target block.
///
/// This is useful for CFF analysis where we need to identify which blocks
/// set the state variable (those that jump back to the dispatcher).
///
/// # Arguments
///
/// * `ssa` - The SSA function to analyze.
/// * `target` - The target block index to find jumpers to.
///
/// # Returns
///
/// A set of block indices that have a control flow edge to `target`.
#[must_use]
pub fn find_blocks_jumping_to<T: Target>(ssa: &SsaFunction<T>, target: usize) -> HashSet<usize> {
    let mut jumpers = HashSet::new();

    for block in ssa.blocks() {
        if let Some(terminator) = block.instructions().last() {
            let jumps_to_target = match terminator.op() {
                SsaOp::Jump { target: t } | SsaOp::Leave { target: t } => *t == target,
                SsaOp::Branch {
                    true_target,
                    false_target,
                    ..
                }
                | SsaOp::BranchCmp {
                    true_target,
                    false_target,
                    ..
                } => *true_target == target || *false_target == target,
                SsaOp::Switch {
                    targets, default, ..
                } => *default == target || targets.contains(&target),
                _ => false,
            };

            if jumps_to_target {
                jumpers.insert(block.id());
            }
        }
    }

    jumpers
}

/// Creates a CFF-specific taint configuration for state variable analysis.
///
/// This configuration is designed for control flow flattening analysis where:
/// - Forward propagation is enabled (derived values from state are tainted)
/// - Backward propagation is disabled (too aggressive, taints loop counters)
/// - PHI taint uses selective mode (only from blocks jumping to dispatcher)
///
/// # Arguments
///
/// * `ssa` - The SSA function being analyzed.
/// * `dispatcher_block` - The block index of the CFF dispatcher.
/// * `state_origin` - Optional `VariableOrigin` to filter PHI chains.
///
/// # Returns
///
/// A `TaintConfig` configured for CFF state tracking.
#[must_use]
pub fn cff_taint_config<T: Target>(
    ssa: &SsaFunction<T>,
    dispatcher_block: usize,
    state_origin: Option<VariableOrigin>,
) -> TaintConfig {
    let predecessors = find_blocks_jumping_to(ssa, dispatcher_block);

    TaintConfig {
        forward: true,
        backward: false,
        phi_mode: PhiTaintMode::SelectivePhi {
            predecessors,
            origin_filter: state_origin,
        },
        max_iterations: 100,
    }
}

// Token-specific taint helpers belong in host crates because they rely on
// host-specific call targets and metadata token types.
