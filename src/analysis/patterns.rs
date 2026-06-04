//! Generic obfuscation pattern detection for SSA-based analysis.
//!
//! This module provides pattern detection for common obfuscation constructs
//! without being tied to any specific obfuscator (ConfuserEx, etc.). The patterns
//! detected include dispatchers (switch-based state machines for control flow
//! flattening), source blocks (that feed state values to dispatchers), and opaque
//! predicates (conditionals with statically determinable outcomes).
//!
//! # Dispatcher Detection Algorithm
//!
//! A dispatcher is characterized by:
//! 1. A switch instruction with multiple targets
//! 2. At least one target that eventually loops back to the dispatcher
//! 3. A computed switch index derived from state variables
//!
//! Detection works in phases:
//!
//! **Phase 1: Switch Identification**: Scan all blocks for `SsaOp::Switch` terminators.
//!
//! **Phase 2: Loop-back Check**: For each potential dispatcher, use BFS (depth-limited
//! to 50) to verify that at least one switch target reaches back to the dispatcher block.
//!
//! **Phase 3: Dispatch Expression**: Use `SsaEvaluator` to build a symbolic expression
//! for how the switch index is computed from phi node results at the dispatcher block.
//! This produces the `dispatch_expr` (e.g., `(state XOR C1) % N`).
//!
//! # Source Block Detection Algorithm
//!
//! Source blocks set the state value that determines which dispatcher case executes next.
//!
//! 1. **Reachability**: Compute all blocks that can reach the dispatcher (reverse BFS).
//! 2. **Source analysis**: For each non-dispatcher block reaching the dispatcher, check
//!    if its terminator targets the dispatcher. If so, evaluate the block using
//!    `SsaEvaluator` to determine what state value it produces.
//! 3. **Target case**: If the state value is known, evaluate the dispatch expression
//!    to determine which case index this source block triggers.
//!
//! # Opaque Predicate Detection
//!
//! An opaque predicate is a conditional branch where the condition can be statically
//! determined to always be true or always false:
//!
//! 1. For each block with an `SsaOp::Branch` terminator, evaluate the block using
//!    `SsaEvaluator` with phi nodes set as symbolic.
//! 2. If the condition variable evaluates to a concrete non-zero value, the predicate
//!    is `AlwaysTrue`.
//! 3. If it evaluates to a concrete zero value, the predicate is `AlwaysFalse`.
//! 4. Otherwise the predicate is `Symbolic` (depends on inputs) or `Unknown`.
//!
//! # Complexity
//!
//! Dispatcher detection: O(B * (B + E)) worst case due to BFS for each potential
//! dispatcher. Source detection: O(B + E) per dispatcher for reachability + evaluation.
//! Opaque predicate detection: O(B * n) where n is instructions per block.
//!
//! # Example
//!
//! ```rust,ignore
//! use analyssa::{analysis::PatternDetector, ir::SsaFunction, MockTarget, PointerSize};
//!
//! let ssa: SsaFunction<MockTarget> = /* ... */;
//!
//! let detector = PatternDetector::new(&ssa, PointerSize::Bit32);
//!
//! // Find dispatcher patterns
//! let dispatchers = detector.find_dispatchers();
//!
//! for dispatcher in &dispatchers {
//!     println!("Dispatcher at block {}", dispatcher.block);
//!
//!     // Find all source blocks
//!     let sources = detector.find_sources(dispatcher);
//!     for source in &sources {
//!         println!("  Source block {} -> case {}", source.block, source.target_case);
//!     }
//! }
//! ```

use std::collections::HashMap;

use crate::{
    analysis::{evaluator::SsaEvaluator, symbolic::SymbolicExpr},
    ir::{function::SsaFunction, ops::SsaOp, value::ConstValue, variable::SsaVarId},
    target::Target,
    BitSet, PointerSize,
};

/// Detects common obfuscation patterns in SSA form.
///
/// The detector analyzes the SSA function to identify structural patterns
/// commonly used in obfuscation, such as:
///
/// - Control flow flattening dispatchers
/// - Opaque predicates
/// - Dead code regions
#[derive(Debug)]
pub struct PatternDetector<'a, T: Target> {
    /// Reference to the SSA function being analyzed for patterns.
    ssa: &'a SsaFunction<T>,
    /// Target pointer size for native int/uint constant evaluation.
    pointer_size: PointerSize,
}

impl<'a, T: Target> PatternDetector<'a, T> {
    /// Creates a new pattern detector for the given SSA function.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>, pointer_size: PointerSize) -> Self {
        Self { ssa, pointer_size }
    }

    /// Returns the underlying SSA function.
    #[must_use]
    pub fn ssa(&self) -> &SsaFunction<T> {
        self.ssa
    }

    // Dispatcher Detection

    /// Finds all potential dispatcher patterns in the function.
    ///
    /// A dispatcher is characterized by:
    /// - A switch instruction with multiple targets
    /// - Targets that eventually loop back to the dispatcher
    /// - A computed switch index (not just a simple variable)
    ///
    /// # Returns
    ///
    /// A vector of detected dispatcher patterns, sorted by block index.
    #[must_use]
    pub fn find_dispatchers(&self) -> Vec<DispatcherPattern<T>> {
        let mut dispatchers: Vec<_> = (0..self.ssa.block_count())
            .filter_map(|block_idx| self.analyze_potential_dispatcher(block_idx))
            .collect();
        dispatchers.sort_by_key(|d| d.block);
        dispatchers
    }

    /// Analyzes a block to determine if it's a dispatcher.
    fn analyze_potential_dispatcher(&self, block_idx: usize) -> Option<DispatcherPattern<T>> {
        let block = self.ssa.block(block_idx)?;

        // Look for Switch instruction at the end
        let terminator = block.terminator()?;
        let (switch_var, targets, default) = match terminator.op() {
            SsaOp::Switch {
                value,
                targets,
                default,
            } => (*value, targets.clone(), *default),
            _ => return None,
        };

        // Must have multiple targets to be a dispatcher
        if targets.len() < 2 {
            return None;
        }

        // Check if any targets loop back to this block
        let has_loopback = targets
            .iter()
            .any(|&target| self.reaches_block(target, block_idx))
            || self.reaches_block(default, block_idx);

        if !has_loopback {
            return None;
        }

        // Try to build the dispatch expression
        let dispatch_expr = self.build_dispatch_expression(block_idx, switch_var);

        // Identify state variables (inputs to the dispatch computation)
        let state_vars = dispatch_expr
            .as_ref()
            .map(|e| {
                // Collect referenced variables without allocating a HashSet;
                // dispatch expressions are small so the linear dedup is cheap.
                let mut vars: Vec<SsaVarId> = Vec::new();
                e.for_each_variable(|v| {
                    if !vars.contains(&v) {
                        vars.push(v);
                    }
                });
                vars
            })
            .unwrap_or_default();

        Some(DispatcherPattern {
            block: block_idx,
            switch_var,
            targets,
            default,
            dispatch_expr,
            state_vars,
        })
    }

    /// Checks if there's a path from `from_block` that reaches `target_block`.
    ///
    /// Uses BFS with a depth limit to avoid infinite loops.
    fn reaches_block(&self, from_block: usize, target_block: usize) -> bool {
        let block_count = self.ssa.block_count().max(1);
        let mut visited = BitSet::new(block_count);
        let mut queue = vec![from_block];
        let max_depth: u32 = 50; // Prevent infinite loops
        let mut depth: u32 = 0;

        while !queue.is_empty() && depth < max_depth {
            let mut next_queue = Vec::new();

            for block_idx in queue {
                if block_idx == target_block {
                    return true;
                }

                if block_idx >= block_count || !visited.insert(block_idx) {
                    continue;
                }

                // Get successors
                if let Some(successors) = self.block_successors(block_idx) {
                    next_queue.extend(successors);
                }
            }

            queue = next_queue;
            depth = depth.saturating_add(1);
        }

        false
    }

    /// Gets the successor blocks of a given block.
    fn block_successors(&self, block_idx: usize) -> Option<Vec<usize>> {
        let block = self.ssa.block(block_idx)?;
        block.terminator()?;
        Some(block.successors())
    }

    /// Builds a symbolic expression for how the switch index is computed.
    fn build_dispatch_expression(
        &self,
        block_idx: usize,
        switch_var: SsaVarId,
    ) -> Option<SymbolicExpr<T>> {
        let mut eval = SsaEvaluator::new(self.ssa, self.pointer_size);

        // Mark all phi results as symbolic (they come from different paths)
        if let Some(block) = self.ssa.block(block_idx) {
            for phi in block.phi_nodes() {
                let name = format!("phi_{}", phi.result().index());
                eval.set_symbolic(phi.result(), name);
            }
        }

        // Evaluate the block
        eval.evaluate_block(block_idx);

        // Get the switch variable's value as a symbolic expression
        eval.get(switch_var).cloned()
    }

    // Source Block Detection

    /// Finds all source blocks for a given dispatcher.
    ///
    /// A source block is one that:
    /// - Sets a state value that determines which case is taken
    /// - Eventually reaches the dispatcher (directly or through intermediates)
    ///
    /// # Arguments
    ///
    /// * `dispatcher` - The dispatcher pattern to find sources for.
    ///
    /// # Returns
    ///
    /// A vector of source blocks with their analyzed state values.
    #[must_use]
    pub fn find_sources(&self, dispatcher: &DispatcherPattern<T>) -> Vec<SourceBlock<T>> {
        // Find all blocks that reach the dispatcher
        let reaching_blocks = self.find_reaching_blocks(dispatcher.block);

        reaching_blocks
            .iter()
            .filter(|&block_idx| block_idx != dispatcher.block)
            .filter_map(|block_idx| self.analyze_source_block(block_idx, dispatcher))
            .collect()
    }

    /// Finds all blocks that can reach the dispatcher block.
    fn find_reaching_blocks(&self, dispatcher_block: usize) -> BitSet {
        let block_count = self.ssa.block_count().max(1);
        let mut reaching = BitSet::new(block_count);

        // Build reverse CFG (predecessors)
        let mut predecessors: HashMap<usize, Vec<usize>> = HashMap::new();
        for block_idx in 0..self.ssa.block_count() {
            if let Some(succs) = self.block_successors(block_idx) {
                for succ in succs {
                    predecessors.entry(succ).or_default().push(block_idx);
                }
            }
        }

        // BFS backwards from dispatcher
        let mut queue = vec![dispatcher_block];
        while let Some(block_idx) = queue.pop() {
            if block_idx >= block_count || !reaching.insert(block_idx) {
                continue;
            }

            if let Some(preds) = predecessors.get(&block_idx) {
                queue.extend(preds.iter().copied());
            }
        }

        reaching
    }

    /// Analyzes a block to determine if it's a source for the dispatcher.
    fn analyze_source_block(
        &self,
        block_idx: usize,
        dispatcher: &DispatcherPattern<T>,
    ) -> Option<SourceBlock<T>> {
        let block = self.ssa.block(block_idx)?;

        // Check if this block has a jump or branch that leads to dispatcher
        let terminator = block.terminator()?;
        let (leads_to_dispatcher, is_conditional) = match terminator.op() {
            SsaOp::Jump { target } => (*target == dispatcher.block, false),
            SsaOp::Branch {
                true_target,
                false_target,
                ..
            } => {
                let leads = *true_target == dispatcher.block || *false_target == dispatcher.block;
                (leads, true)
            }
            _ => return None,
        };

        if !leads_to_dispatcher {
            return None;
        }

        // Try to determine what state value this block sets
        let state_value = self.compute_state_value(block_idx, dispatcher);

        // Try to determine which case this leads to
        let target_case = self.compute_target_case(state_value.as_ref(), dispatcher);

        Some(SourceBlock {
            block: block_idx,
            state_value,
            target_case,
            is_conditional,
        })
    }

    /// Computes the state value set by a block.
    fn compute_state_value(
        &self,
        block_idx: usize,
        dispatcher: &DispatcherPattern<T>,
    ) -> Option<SymbolicExpr<T>> {
        let mut eval = SsaEvaluator::new(self.ssa, self.pointer_size);

        // Set phi nodes in this block as symbolic
        if let Some(block) = self.ssa.block(block_idx) {
            for phi in block.phi_nodes() {
                let name = format!("phi_{}", phi.result().index());
                eval.set_symbolic(phi.result(), name);
            }
        }

        // Evaluate the block
        eval.evaluate_block(block_idx);

        // Get the value of the state variable
        // We need to find which variable in this block contributes to the dispatcher's state
        if let Some(state_var) = dispatcher.state_vars.first() {
            // The state var in dispatcher comes from a phi, we need to find
            // what value this block provides to that phi
            if let Some(disp_block) = self.ssa.block(dispatcher.block) {
                for phi in disp_block.phi_nodes() {
                    if phi.result() == *state_var {
                        // Find operand from our block
                        for operand in phi.operands() {
                            if operand.predecessor() == block_idx {
                                return eval.get(operand.value()).cloned();
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Computes which case a state value leads to.
    fn compute_target_case(
        &self,
        state_value: Option<&SymbolicExpr<T>>,
        dispatcher: &DispatcherPattern<T>,
    ) -> Option<usize> {
        // If we have a concrete state value and a dispatch expression,
        // we can compute the target case
        let concrete_state = state_value.and_then(SymbolicExpr::as_constant)?;
        let dispatch_expr = dispatcher.dispatch_expr.as_ref()?;

        // Build bindings for evaluation
        // First, create owned names so we can borrow them
        let state_var_names: Vec<String> = dispatcher
            .state_vars
            .iter()
            .map(|v| format!("phi_{}", v.index()))
            .collect();

        let mut bindings: HashMap<&str, ConstValue<T>> = HashMap::new();
        for name in &state_var_names {
            bindings.insert(name.as_str(), concrete_state.clone());
        }

        // Also try with just "state" as a generic name
        bindings.insert("state", concrete_state.clone());

        // Evaluate the dispatch expression
        let case_idx = dispatch_expr.evaluate_named(&bindings, self.pointer_size)?;

        // Convert to usize and check bounds
        let idx = case_idx.as_i64().and_then(|v| usize::try_from(v).ok())?;
        if idx < dispatcher.targets.len() {
            Some(idx)
        } else {
            None // Out of bounds -> default case
        }
    }

    // Opaque Predicate Detection

    /// Finds potential opaque predicates in the function.
    ///
    /// An opaque predicate is a conditional branch where the condition
    /// can be statically determined to always be true or always false.
    #[must_use]
    pub fn find_opaque_predicates(&self) -> Vec<OpaquePredicatePattern<T>> {
        (0..self.ssa.block_count())
            .filter_map(|block_idx| self.analyze_opaque_predicate(block_idx))
            .collect()
    }

    /// Analyzes a block to see if it contains an opaque predicate.
    fn analyze_opaque_predicate(&self, block_idx: usize) -> Option<OpaquePredicatePattern<T>> {
        let block = self.ssa.block(block_idx)?;

        // Look for Branch instruction
        let terminator = block.terminator()?;
        let (condition_var, true_target, false_target) = match terminator.op() {
            SsaOp::Branch {
                condition,
                true_target,
                false_target,
            } => (*condition, *true_target, *false_target),
            _ => return None,
        };

        // Evaluate the block to see if condition is determinable
        let mut eval = SsaEvaluator::new(self.ssa, self.pointer_size);

        // Set phi nodes as symbolic
        for phi in block.phi_nodes() {
            let name = format!("phi_{}", phi.result().index());
            eval.set_symbolic(phi.result(), name);
        }

        eval.evaluate_block(block_idx);

        let condition_value = eval.get(condition_var);

        let resolution = match condition_value {
            Some(expr) if expr.is_constant() => {
                if expr.as_constant().is_some_and(ConstValue::is_zero) {
                    PredicateResolution::AlwaysFalse
                } else {
                    PredicateResolution::AlwaysTrue
                }
            }
            Some(expr) => PredicateResolution::Symbolic(expr.clone()),
            None => PredicateResolution::Unknown,
        };

        // Only report if it's always true or always false (actual opaque predicate)
        if matches!(
            resolution,
            PredicateResolution::AlwaysTrue | PredicateResolution::AlwaysFalse
        ) {
            Some(OpaquePredicatePattern {
                block: block_idx,
                condition_var,
                true_target,
                false_target,
                resolution,
            })
        } else {
            None
        }
    }
}

// Pattern Types

/// A detected dispatcher pattern (switch-based state machine).
///
/// Dispatchers are the core of control flow flattening. They use a computed
/// switch index to dispatch to different case blocks, with state variables
/// controlling the execution flow.
#[derive(Debug, Clone)]
pub struct DispatcherPattern<T: Target> {
    /// Block index containing the switch instruction.
    pub block: usize,

    /// The SSA variable used as the switch condition.
    pub switch_var: SsaVarId,

    /// Target blocks for each switch case (indexed by case value).
    pub targets: Vec<usize>,

    /// Default target when case is out of range.
    pub default: usize,

    /// The symbolic expression computing the switch index, if determinable.
    /// This is typically something like `(state ^ const) % num_cases`.
    pub dispatch_expr: Option<SymbolicExpr<T>>,

    /// State variables that control the dispatch (phi node results that
    /// feed into the dispatch expression).
    pub state_vars: Vec<SsaVarId>,
}

impl<T: Target> DispatcherPattern<T> {
    /// Returns the number of cases in this dispatcher.
    #[must_use]
    pub fn case_count(&self) -> usize {
        self.targets.len()
    }

    /// Gets the target block for a specific case index.
    #[must_use]
    pub fn target_for_case(&self, case_idx: usize) -> usize {
        self.targets.get(case_idx).copied().unwrap_or(self.default)
    }
}

/// A source block that feeds into a dispatcher.
///
/// Source blocks set state values that determine which case the dispatcher
/// will execute next.
#[derive(Debug, Clone)]
pub struct SourceBlock<T: Target> {
    /// Block index.
    pub block: usize,

    /// The state value this block sets.
    /// `None` means the value could not be determined (unknown).
    pub state_value: Option<SymbolicExpr<T>>,

    /// The target case this state value leads to, if determinable.
    pub target_case: Option<usize>,

    /// Whether this is a conditional source (branch vs jump).
    pub is_conditional: bool,
}

/// A detected opaque predicate.
///
/// Opaque predicates are conditionals that always evaluate to the same result,
/// used to confuse analysis and add fake branches.
#[derive(Debug, Clone)]
pub struct OpaquePredicatePattern<T: Target> {
    /// Block index containing the branch.
    pub block: usize,

    /// The SSA variable holding the condition.
    pub condition_var: SsaVarId,

    /// Target if condition is true.
    pub true_target: usize,

    /// Target if condition is false.
    pub false_target: usize,

    /// How the predicate resolves.
    pub resolution: PredicateResolution<T>,
}

impl<T: Target> OpaquePredicatePattern<T> {
    /// Returns the target that will always be taken.
    #[must_use]
    pub fn actual_target(&self) -> Option<usize> {
        match self.resolution {
            PredicateResolution::AlwaysTrue => Some(self.true_target),
            PredicateResolution::AlwaysFalse => Some(self.false_target),
            _ => None,
        }
    }

    /// Returns the target that will never be taken.
    #[must_use]
    pub fn dead_target(&self) -> Option<usize> {
        match self.resolution {
            PredicateResolution::AlwaysTrue => Some(self.false_target),
            PredicateResolution::AlwaysFalse => Some(self.true_target),
            _ => None,
        }
    }
}

/// How an opaque predicate resolves after analysis.
///
/// Indicates whether a conditional branch's outcome can be statically determined
/// (the branch is an opaque predicate) or depends on external inputs.
#[derive(Debug, Clone)]
pub enum PredicateResolution<T: Target> {
    /// The condition always evaluates to true. The false branch is dead code.
    AlwaysTrue,
    /// The condition always evaluates to false. The true branch is dead code.
    AlwaysFalse,
    /// The outcome depends on symbolic values that were not fully resolved.
    /// The branch is NOT a true opaque predicate but may become one with
    /// additional analysis or constant propagation.
    Symbolic(SymbolicExpr<T>),
    /// The condition could not be evaluated at all. The branch direction is
    /// completely unknown (e.g., depends on external input or side effects).
    Unknown,
}
