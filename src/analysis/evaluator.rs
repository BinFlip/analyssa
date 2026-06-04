//! Hybrid concrete/symbolic SSA evaluator for computing values from operations.
//!
//! This module provides [`SsaEvaluator`], an interpreter for SSA operations that
//! can evaluate arithmetic and logical operations given known input values. It
//! serves as the core computation engine for control flow unflattening, opaque
//! predicate detection, and symbolic execution.
//!
//! # Value Representation
//!
//! Values use a three-tier representation as [`SymbolicExpr`]:
//!
//! | Category | Representation | Meaning |
//! |----------|---------------|---------|
//! | Concrete | `SymbolicExpr::Constant(v)` | Known integer constant |
//! | Symbolic | Other `SymbolicExpr` variants | Expression tree depending on unknown inputs |
//! | Unknown | `None` (absent from value map) | Cannot be determined statically |
//!
//! # Algorithm
//!
//! ## Instruction Evaluation
//!
//! Each SSA operation is evaluated by:
//! 1. Looking up operand values in the tracked values map
//! 2. If all operands are concrete, computing a concrete result via `ConstValue` ops
//! 3. If any operand is symbolic, building a `SymbolicExpr` tree preserving the operation
//! 4. Applying simplification (constant folding, algebraic identities) to the result
//! 5. Storing the result for use by subsequent operations
//!
//! ## Path-Aware Evaluation
//!
//! Phi nodes merge values from different predecessors. For path-aware evaluation,
//! callers set the predecessor block via `set_predecessor()` before evaluating a
//! block. The evaluator then selects the phi operand from that predecessor,
//! enabling precise tracking along a specific execution path.
//!
//! Phi evaluation uses two-phase semantics ("simultaneous execution"):
//! 1. Read all source values into a temporary buffer
//! 2. Write all results from the buffer
//!
//! This ensures correct swap semantics (e.g., `v1 = phi(v2); v2 = phi(v1)`).
//!
//! ## Loop Evaluation
//!
//! Two strategies are available:
//! - **Fixed-point iteration**: `evaluate_loop_to_fixpoint` iterates until values
//!   stabilize, then widens unstable values to Unknown. Useful when loop bounds
//!   are unknown.
//! - **Bounded iteration**: `evaluate_loop_iterations` runs a known number of
//!   iterations. Useful when the loop count is a known constant.
//!
//! ## Trace Execution
//!
//! The `execute` method provides full CFG traversal: starting from a block, it
//! evaluates each block, follows control flow decisions based on computed values,
//! and records the execution trace. This is the primary tool for CFF deobfuscation
//! where the goal is to record all state transitions through the dispatcher.
//!
//! # Constraint Tracking
//!
//! When taking a branch, `apply_branch_constraint()` derives facts about the
//! condition variable (e.g., `x == 5` on the true branch of `ceq(x, 5)`).
//! These constraints can be used to detect infeasible paths and dead code.
//!
//! # Local/Argument State Tracking
//!
//! The evaluator maintains `local_state` and `arg_state` maps that mirror CIL
//! local variable and argument slots. When a variable with `VariableOrigin::Local(N)`
//! receives a value, the corresponding local slot is updated. `LoadLocal`/`LoadArg`
//! instructions read from these maps, enabling tracking across SSA variable versions.
//!
//! Address-of operations (`LoadLocalAddr`, `LoadArgAddr`) invalidate the tracked
//! state for the corresponding local/arg, preventing incorrect reuse of
//! pre-address-taken values (critical for patterns like `Monitor.Enter`).
//!
//! # CIL Semantics
//!
//! All arithmetic operations use 32-bit wrapping semantics as per ECMA-335.
//! Values are stored as i64 for convenience, but operations intentionally
//! truncate to i32/u32 to match CLR behavior.
//!
//! # Complexity
//!
//! Instruction evaluation: O(d) where d is the expression tree depth
//! (bounded by simplification). Block evaluation: O(n * d) for n instructions.
//! Loop fixed-point: O(i * b * n * d) where i is iterations (bounded by config).
//!
//! # Usage
//!
//! ```rust,ignore
//! use analyssa::{analysis::{SsaEvaluator, SymbolicExpr}, PointerSize};
//!
//! let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
//!
//! // Set known concrete values
//! eval.set_concrete(state_var, initial_state);
//!
//! // Or mark as symbolic
//! eval.set_symbolic(arg_var, "arg0");
//!
//! // Evaluate a block's instructions
//! eval.evaluate_block(block_idx);
//!
//! // Get computed result
//! match eval.get(result_var) {
//!     Some(expr) if expr.is_constant() => println!("Known: {}", expr.as_constant().unwrap()),
//!     Some(expr) => println!("Symbolic: {}", expr),
//!     None => println!("Cannot determine"),
//! }
//!
//! // Or use convenience method for concrete values
//! if let Some(next_state) = eval.get_concrete(result_var) {
//!     println!("Next state: {}", next_state);
//! }
//! ```

use std::collections::BTreeMap;

use crate::{
    analysis::{
        constraints::Constraint,
        memory::{MemoryLocation, MemoryState},
        symbolic::{SymbolicExpr, SymbolicOp},
    },
    ir::{
        function::SsaFunction,
        ops::{CmpKind, SsaOp},
        value::ConstValue,
        variable::{SsaVarId, VariableOrigin},
    },
    target::Target,
    PointerSize,
};

/// Result of evaluating a control flow decision.
///
/// This represents the outcome of analyzing a terminator instruction to
/// determine the next block to execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlFlow {
    /// Continue to the specified block.
    Continue(usize),
    /// Terminal instruction - execution ends here (return, throw, etc.).
    Terminal,
    /// Cannot determine the next block - condition is unknown or symbolic.
    Unknown,
}

impl ControlFlow {
    /// Returns the target block if this is a `Continue` result.
    #[must_use]
    pub fn target(&self) -> Option<usize> {
        match self {
            Self::Continue(block) => Some(*block),
            _ => None,
        }
    }

    /// Returns `true` if this is a terminal result.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal)
    }

    /// Returns `true` if the control flow cannot be determined.
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }
}

/// Configuration for SSA evaluators.
///
/// This struct controls the behavior of [`SsaEvaluator`], allowing it to be
/// configured for different use cases like general evaluation, path-aware
/// analysis, or CFF deobfuscation.
#[derive(Debug, Clone, Default)]
pub struct EvaluatorConfig {
    /// Track the execution path (sequence of visited blocks).
    pub track_path: bool,
    /// Track memory state (field loads/stores).
    pub track_memory: bool,
    /// Require predecessor for phi evaluation (strict path-aware mode).
    pub strict_phi: bool,
}

impl EvaluatorConfig {
    /// Creates a new default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a configuration for path-aware analysis.
    ///
    /// Enables path tracking, memory tracking, and strict phi evaluation.
    #[must_use]
    pub fn path_aware() -> Self {
        Self {
            track_path: true,
            track_memory: true,
            strict_phi: true,
        }
    }

    /// Creates a configuration with memory tracking only.
    #[must_use]
    pub fn with_memory() -> Self {
        Self {
            track_path: false,
            track_memory: true,
            strict_phi: false,
        }
    }

    /// Enables path tracking.
    #[must_use]
    pub fn with_path_tracking(mut self) -> Self {
        self.track_path = true;
        self
    }

    /// Enables memory state tracking.
    #[must_use]
    pub fn with_memory_tracking(mut self) -> Self {
        self.track_memory = true;
        self
    }

    /// Enables strict phi evaluation (requires predecessor).
    #[must_use]
    pub fn with_strict_phi(mut self) -> Self {
        self.strict_phi = true;
        self
    }
}

/// Records the execution trace of SSA evaluation.
///
/// This struct tracks the sequence of blocks visited during SSA evaluation,
/// along with optional state values at each step. This is essential for
/// CFF (Control Flow Flattening) deobfuscation, where we need to record
/// the dispatcher state transitions to reconstruct the original control flow.
#[derive(Debug, Clone)]
pub struct ExecutionTrace<T: Target> {
    /// Sequence of block indices visited.
    blocks: Vec<usize>,
    /// Optional state values captured at each block (for state machines).
    states: Vec<Option<ConstValue<T>>>,
    /// Whether execution completed normally (reached terminal).
    completed: bool,
    /// Maximum blocks to trace before stopping (prevents infinite loops).
    limit: usize,
}

impl<T: Target> ExecutionTrace<T> {
    /// Creates a new execution trace with the given block limit.
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            blocks: Vec::new(),
            states: Vec::new(),
            completed: false,
            limit,
        }
    }

    /// Returns the blocks visited during execution.
    #[must_use]
    pub fn blocks(&self) -> &[usize] {
        &self.blocks
    }

    /// Returns the state values captured during execution.
    #[must_use]
    pub fn states(&self) -> &[Option<ConstValue<T>>] {
        &self.states
    }

    /// Returns `true` if execution completed (reached a terminal instruction).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.completed
    }

    /// Returns the number of blocks visited.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Returns `true` if no blocks were visited.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Returns the last visited block, if any.
    #[must_use]
    pub fn last_block(&self) -> Option<usize> {
        self.blocks.last().copied()
    }

    /// Returns `true` if the trace reached the block limit.
    #[must_use]
    pub fn hit_limit(&self) -> bool {
        self.blocks.len() >= self.limit
    }

    /// Records a block visit.
    fn record_block(&mut self, block_idx: usize, state: Option<ConstValue<T>>) {
        self.blocks.push(block_idx);
        self.states.push(state);
    }

    /// Marks execution as complete.
    fn mark_complete(&mut self) {
        self.completed = true;
    }
}

/// SSA evaluator with hybrid concrete/symbolic value tracking.
///
/// This evaluator interprets SSA operations to compute values without needing
/// full CIL emulation. Values are represented as [`SymbolicExpr`]:
///
/// - **Concrete**: `SymbolicExpr::Constant(v)` - Known integer values
/// - **Symbolic**: Other `SymbolicExpr` variants - Expressions depending on unknown inputs
/// - **Unknown**: `None` (not in the values map) - Values that cannot be determined
///
/// # Value Representation
///
/// Values are represented as `i64` internally to accommodate both 32-bit and 64-bit
/// integer operations. For 32-bit operations, the evaluator applies appropriate
/// wrapping/truncation semantics.
#[derive(Debug, Clone)]
pub struct SsaEvaluator<'a, T: Target> {
    /// Reference to the SSA function being evaluated.
    ssa: &'a SsaFunction<T>,
    /// Tracked values for variables. Missing entries represent unknown values.
    values: BTreeMap<SsaVarId, SymbolicExpr<T>>,
    /// Current predecessor block for path-aware phi evaluation.
    /// When set, phi nodes will select the operand from this predecessor.
    predecessor: Option<usize>,
    /// Constraints on variable values derived from branch conditions.
    /// Used to detect dead code and propagate information after branches.
    constraints: BTreeMap<SsaVarId, Vec<Constraint<T>>>,
    /// Evaluator configuration controlling behavior.
    config: EvaluatorConfig,
    /// Execution path (sequence of visited blocks). Only populated if `config.track_path`.
    path: Vec<usize>,
    /// Memory state tracking for fields. Only used if `config.track_memory`.
    memory: MemoryState<T>,
    /// Target pointer size for native int/uint masking.
    pointer_size: PointerSize,
    /// Current value of CIL local variables, indexed by local_index.
    /// Updated whenever a variable with `Local(N)` origin receives a value,
    /// and read by `LoadLocal` instructions.
    local_state: BTreeMap<u16, SymbolicExpr<T>>,
    /// Current value of CIL arguments, indexed by arg_index.
    /// Updated whenever a variable with `Argument(N)` origin receives a value,
    /// and read by `LoadArg` instructions.
    arg_state: BTreeMap<u16, SymbolicExpr<T>>,
}

impl<'a, T: Target> SsaEvaluator<'a, T> {
    /// Creates a new evaluator for the given SSA function.
    ///
    /// The evaluator starts with no known values. Use the `set_*` methods
    /// to provide initial values for input variables.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to evaluate.
    /// * `ptr_size` - Target pointer size for native int/uint masking.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>, ptr_size: PointerSize) -> Self {
        Self::with_config(ssa, EvaluatorConfig::default(), ptr_size)
    }

    /// Creates an evaluator with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to evaluate.
    /// * `config` - Configuration controlling evaluator behavior.
    /// * `ptr_size` - Target pointer size for native int/uint masking.
    #[must_use]
    pub fn with_config(
        ssa: &'a SsaFunction<T>,
        config: EvaluatorConfig,
        ptr_size: PointerSize,
    ) -> Self {
        Self {
            ssa,
            values: BTreeMap::new(),
            predecessor: None,
            constraints: BTreeMap::new(),
            config,
            path: Vec::new(),
            memory: MemoryState::new(),
            pointer_size: ptr_size,
            local_state: BTreeMap::new(),
            arg_state: BTreeMap::new(),
        }
    }

    /// Creates a path-aware evaluator with memory tracking.
    ///
    /// This is equivalent to `PathAwareEvaluator::with_memory_tracking()` and is
    /// the recommended configuration for CFF deobfuscation.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to evaluate.
    /// * `ptr_size` - Target pointer size for native int/uint masking.
    #[must_use]
    pub fn path_aware(ssa: &'a SsaFunction<T>, ptr_size: PointerSize) -> Self {
        Self::with_config(ssa, EvaluatorConfig::path_aware(), ptr_size)
    }

    /// Creates an evaluator with pre-populated concrete values.
    ///
    /// Useful when you already have a set of known constants from SCCP or
    /// other analyses.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to evaluate.
    /// * `values` - Pre-populated concrete values.
    /// * `ptr_size` - Target pointer size for native int/uint masking.
    #[must_use]
    pub fn with_values(
        ssa: &'a SsaFunction<T>,
        values: BTreeMap<SsaVarId, ConstValue<T>>,
        ptr_size: PointerSize,
    ) -> Self {
        let exprs = values
            .into_iter()
            .map(|(k, v)| (k, SymbolicExpr::constant(v)))
            .collect();
        Self {
            ssa,
            values: exprs,
            predecessor: None,
            constraints: BTreeMap::new(),
            config: EvaluatorConfig::default(),
            path: Vec::new(),
            memory: MemoryState::new(),
            pointer_size: ptr_size,
            local_state: BTreeMap::new(),
            arg_state: BTreeMap::new(),
        }
    }

    /// Returns the target pointer size.
    #[must_use]
    pub fn pointer_size(&self) -> PointerSize {
        self.pointer_size
    }

    /// Returns a reference to the underlying SSA function.
    #[must_use]
    pub fn ssa(&self) -> &SsaFunction<T> {
        self.ssa
    }

    /// Returns a reference to the evaluator configuration.
    #[must_use]
    pub fn config(&self) -> &EvaluatorConfig {
        &self.config
    }

    /// Returns the execution path if path tracking is enabled.
    #[must_use]
    pub fn path(&self) -> &[usize] {
        &self.path
    }

    /// Clears the recorded execution path.
    pub fn clear_path(&mut self) {
        self.path.clear();
    }

    /// Returns whether memory tracking is enabled.
    #[must_use]
    pub fn memory_tracking_enabled(&self) -> bool {
        self.config.track_memory
    }

    // Value Setting

    /// Sets a concrete (known) value for a variable.
    ///
    /// The caller is responsible for providing the correct `ConstValue` type
    /// that matches the variable's type in the SSA function.
    pub fn set_concrete(&mut self, var: SsaVarId, value: ConstValue<T>) {
        let expr = SymbolicExpr::constant(value);
        self.values.insert(var, expr.clone());
        self.track_origin_state(var, &expr);
    }

    /// Sets a symbolic value for a variable using a named expression.
    ///
    /// This is useful for marking method arguments or other external inputs
    /// as symbolic with descriptive names.
    pub fn set_symbolic(&mut self, var: SsaVarId, name: impl Into<String>) {
        let expr = SymbolicExpr::named(name);
        self.values.insert(var, expr.clone());
        self.track_origin_state(var, &expr);
    }

    /// Sets a symbolic value for a variable using an expression.
    pub fn set_symbolic_expr(&mut self, var: SsaVarId, expr: SymbolicExpr<T>) {
        self.values.insert(var, expr.clone());
        self.track_origin_state(var, &expr);
    }

    /// Sets a variable as unknown by removing it from the values map.
    pub fn set_unknown(&mut self, var: SsaVarId) {
        self.values.remove(&var);
    }

    /// Sets an expression for a variable.
    pub fn set(&mut self, var: SsaVarId, value: SymbolicExpr<T>) {
        self.values.insert(var, value.clone());
        self.track_origin_state(var, &value);
    }

    // Value Getting

    /// Gets the expression for a variable.
    ///
    /// Returns `None` if the variable hasn't been assigned a value (unknown).
    #[must_use]
    pub fn get(&self, var: SsaVarId) -> Option<&SymbolicExpr<T>> {
        self.values.get(&var)
    }

    /// Gets the typed constant value for a variable, if it's a constant.
    ///
    /// Returns `None` if the variable is symbolic, unknown, or not set.
    /// Use [`ConstValue`] methods to extract specific types (e.g., `as_i64()`, `as_i32()`).
    #[must_use]
    pub fn get_concrete(&self, var: SsaVarId) -> Option<&ConstValue<T>> {
        self.values.get(&var).and_then(SymbolicExpr::as_constant)
    }

    /// Gets the symbolic expression for a variable, if it's not a constant.
    #[must_use]
    pub fn get_symbolic(&self, var: SsaVarId) -> Option<&SymbolicExpr<T>> {
        self.values.get(&var).filter(|e| !e.is_constant())
    }

    /// Checks if a variable has a concrete (constant) value.
    #[must_use]
    pub fn is_concrete(&self, var: SsaVarId) -> bool {
        self.values.get(&var).is_some_and(SymbolicExpr::is_constant)
    }

    /// Checks if a variable has a symbolic (non-constant) value.
    #[must_use]
    pub fn is_symbolic(&self, var: SsaVarId) -> bool {
        self.values.get(&var).is_some_and(|e| !e.is_constant())
    }

    /// Checks if a variable is unknown (not in the values map).
    #[must_use]
    pub fn is_unknown(&self, var: SsaVarId) -> bool {
        !self.values.contains_key(&var)
    }

    /// Returns all tracked values as expressions.
    #[must_use]
    pub fn values(&self) -> &BTreeMap<SsaVarId, SymbolicExpr<T>> {
        &self.values
    }

    /// Returns all concrete values as a map of i64 values.
    ///
    /// This is useful for compatibility with code that expects `HashMap<SsaVarId, i64>`.
    /// Values that can't be converted to i64 are skipped.
    #[must_use]
    pub fn concrete_values(&self) -> BTreeMap<SsaVarId, i64> {
        self.values
            .iter()
            .filter_map(|(k, v)| v.as_i64().map(|c| (*k, c)))
            .collect()
    }

    /// Returns all concrete values as typed `ConstValue`.
    #[must_use]
    pub fn const_values(&self) -> BTreeMap<SsaVarId, ConstValue<T>> {
        self.values
            .iter()
            .filter_map(|(k, v)| v.as_constant().map(|c| (*k, c.clone())))
            .collect()
    }

    /// Clears all tracked values.
    pub fn clear(&mut self) {
        self.values.clear();
        self.predecessor = None;
        self.constraints.clear();
    }

    // Constraint Management

    /// Adds a constraint on a variable.
    ///
    /// If the constraint is an equality constraint, also sets the variable's value
    /// to concrete. This allows constraint propagation to directly affect evaluation.
    pub fn add_constraint(&mut self, var: SsaVarId, constraint: Constraint<T>) {
        // If it's an equality constraint, we can directly set the value
        if let Constraint::Equal(ref v) = constraint {
            let expr = SymbolicExpr::constant(v.clone());
            self.values.insert(var, expr.clone());
            self.track_origin_state(var, &expr);
        }

        self.constraints.entry(var).or_default().push(constraint);
    }

    /// Gets all constraints on a variable.
    #[must_use]
    pub fn constraints(&self, var: SsaVarId) -> &[Constraint<T>] {
        self.constraints.get(&var).map_or(&[], |v| v.as_slice())
    }

    /// Checks if a variable has any constraints.
    #[must_use]
    pub fn has_constraints(&self, var: SsaVarId) -> bool {
        self.constraints.get(&var).is_some_and(|v| !v.is_empty())
    }

    /// Clears constraints for a specific variable.
    pub fn clear_constraints(&mut self, var: SsaVarId) {
        self.constraints.remove(&var);
    }

    /// Applies constraints derived from taking a specific branch.
    ///
    /// When we know which branch was taken, we can derive facts about the condition
    /// variable. For example, if we took the true branch of `if (ceq x, 5)`, we know x == 5.
    ///
    /// # Arguments
    ///
    /// * `condition` - The variable used as the branch condition
    /// * `took_true_branch` - Whether we followed the true or false branch
    ///
    /// # Returns
    ///
    /// `true` if constraints were successfully derived, `false` otherwise.
    pub fn apply_branch_constraint(&mut self, condition: SsaVarId, took_true_branch: bool) -> bool {
        // Find the definition of the condition variable to understand what comparison it represents
        let Some(ssa_var) = self.ssa.variable(condition) else {
            return false;
        };

        let def_site = ssa_var.def_site();
        let Some(block) = self.ssa.block(def_site.block) else {
            return false;
        };

        let Some(instr_idx) = def_site.instruction else {
            return false;
        };

        let Some(instr) = block.instruction(instr_idx) else {
            return false;
        };

        self.derive_constraints_from_comparison(instr.op(), took_true_branch)
    }

    /// Derives constraints from a comparison operation.
    fn derive_constraints_from_comparison(
        &mut self,
        op: &SsaOp<T>,
        took_true_branch: bool,
    ) -> bool {
        match op {
            SsaOp::Ceq { left, right, .. } => {
                // ceq: true branch means left == right, false means left != right
                let left_val = self.get(*left).cloned();
                let right_val = self.get(*right).cloned();

                if took_true_branch {
                    // left == right
                    match (&left_val, &right_val) {
                        (Some(l), None) => {
                            if let Some(v) = l.as_constant() {
                                self.add_constraint(*right, Constraint::Equal(v.clone()));
                                true
                            } else {
                                false
                            }
                        }
                        (None, Some(r)) => {
                            if let Some(v) = r.as_constant() {
                                self.add_constraint(*left, Constraint::Equal(v.clone()));
                                true
                            } else {
                                false
                            }
                        }
                        (Some(l), Some(r)) if l.as_constant() == r.as_constant() => {
                            // Both concrete and equal - constraint is satisfied
                            true
                        }
                        _ => false,
                    }
                } else {
                    // left != right
                    match (&left_val, &right_val) {
                        (Some(l), None) => {
                            if let Some(v) = l.as_constant() {
                                self.add_constraint(*right, Constraint::NotEqual(v.clone()));
                                true
                            } else {
                                false
                            }
                        }
                        (None, Some(r)) => {
                            if let Some(v) = r.as_constant() {
                                self.add_constraint(*left, Constraint::NotEqual(v.clone()));
                                true
                            } else {
                                false
                            }
                        }
                        _ => false,
                    }
                }
            }

            SsaOp::Cgt {
                left,
                right,
                unsigned,
                ..
            } => {
                // cgt: true branch means left > right
                let right_val = self.get(*right).and_then(|e| e.as_constant().cloned());

                if took_true_branch {
                    // left > right
                    if let Some(v) = right_val {
                        if *unsigned {
                            self.add_constraint(*left, Constraint::GreaterThanUnsigned(v));
                        } else {
                            self.add_constraint(*left, Constraint::GreaterThan(v));
                        }
                        return true;
                    }
                } else {
                    // left <= right
                    if let Some(v) = right_val {
                        self.add_constraint(*left, Constraint::LessOrEqual(v));
                        return true;
                    }
                }
                false
            }

            SsaOp::Clt {
                left,
                right,
                unsigned,
                ..
            } => {
                // clt: true branch means left < right
                let right_val = self.get(*right).and_then(|e| e.as_constant().cloned());

                if took_true_branch {
                    // left < right
                    if let Some(v) = right_val {
                        if *unsigned {
                            self.add_constraint(*left, Constraint::LessThanUnsigned(v));
                        } else {
                            self.add_constraint(*left, Constraint::LessThan(v));
                        }
                        return true;
                    }
                } else {
                    // left >= right
                    if let Some(v) = right_val {
                        self.add_constraint(*left, Constraint::GreaterOrEqual(v));
                        return true;
                    }
                }
                false
            }

            _ => false,
        }
    }

    /// Checks if the current constraints imply that a condition is always true or false.
    ///
    /// This is useful for detecting dead code after branch conditions.
    ///
    /// # Returns
    ///
    /// - `Some(true)` if the condition is always true given current constraints
    /// - `Some(false)` if the condition is always false given current constraints
    /// - `None` if the condition cannot be determined
    #[must_use]
    pub fn evaluate_condition_with_constraints(&self, condition: SsaVarId) -> Option<bool> {
        if let Some(v) = self.get_concrete(condition) {
            return Some(!v.is_zero());
        }

        // Check if constraints imply a value
        // For now, we handle the case where we have conflicting constraints
        // which would indicate dead code
        let ssa_var = self.ssa.variable(condition)?;
        let def_site = ssa_var.def_site();
        let block = self.ssa.block(def_site.block)?;
        let instr_idx = def_site.instruction?;
        let instr = block.instruction(instr_idx)?;
        self.check_condition_against_constraints(instr.op())
    }

    /// Checks if a comparison's result can be determined from constraints.
    fn check_condition_against_constraints(&self, op: &SsaOp<T>) -> Option<bool> {
        match op {
            SsaOp::Ceq { left, right, .. } => {
                // Check if we know both operands are equal or not equal
                let left_constraints = self.constraints(*left);
                let right_val = self.get_concrete(*right)?;

                for constraint in left_constraints {
                    match constraint {
                        Constraint::Equal(v) => {
                            // v == right_val means ceq is true
                            return Some(v.ceq(right_val).is_some_and(|r| !r.is_zero()));
                        }
                        Constraint::NotEqual(v)
                            // If v == right_val, then left != right_val, so ceq is false
                            if v.ceq(right_val).is_some_and(|r| !r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::GreaterThan(v)
                            // left > v, so if right_val <= v, then left != right_val
                            if right_val.cgt(v).is_none_or(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::LessThan(v)
                            // left < v, so if right_val >= v, then left != right_val
                            if right_val.clt(v).is_none_or(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        _ => {}
                    }
                }
                None
            }

            SsaOp::Cgt { left, right, .. } => {
                let left_constraints = self.constraints(*left);
                let right_val = self.get_concrete(*right)?;

                for constraint in left_constraints {
                    match constraint {
                        Constraint::GreaterThan(v)
                            // left > v, so if v >= right_val, then left > right_val
                            if v.clt(right_val).is_none_or(|r| r.is_zero()) =>
                        {
                            return Some(true);
                        }
                        Constraint::LessOrEqual(v)
                            // left <= v, so if v <= right_val, then left <= right_val
                            if v.cgt(right_val).is_none_or(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::LessThan(v)
                            // left < v, so if v <= right_val, then left < right_val <= right_val
                            if v.cgt(right_val).is_none_or(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::Equal(v) => {
                            // left == v, so return v > right_val
                            return Some(v.cgt(right_val).is_some_and(|r| !r.is_zero()));
                        }
                        _ => {}
                    }
                }
                None
            }

            SsaOp::Clt { left, right, .. } => {
                let left_constraints = self.constraints(*left);
                let right_val = self.get_concrete(*right)?;

                for constraint in left_constraints {
                    match constraint {
                        Constraint::LessThan(v)
                            // left < v, so if v <= right_val, then left < right_val
                            if v.cgt(right_val).is_none_or(|r| r.is_zero()) =>
                        {
                            return Some(true);
                        }
                        Constraint::GreaterOrEqual(v)
                            // left >= v, so if v >= right_val, then left >= right_val
                            if v.clt(right_val).is_none_or(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::GreaterThan(v)
                            // left > v, so if v >= right_val, then left > right_val >= right_val
                            if v.clt(right_val).is_none_or(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::Equal(v) => {
                            // left == v, so return v < right_val
                            return Some(v.clt(right_val).is_some_and(|r| !r.is_zero()));
                        }
                        _ => {}
                    }
                }
                None
            }

            _ => None,
        }
    }

    // Path-Aware Evaluation

    /// Sets the predecessor block for path-aware phi evaluation.
    ///
    /// When evaluating a block, phi nodes will select the operand that
    /// corresponds to this predecessor. This enables accurate evaluation
    /// when following a specific path through the CFG.
    ///
    /// # Arguments
    ///
    /// * `pred` - The predecessor block index, or `None` to clear.
    pub fn set_predecessor(&mut self, pred: Option<usize>) {
        self.predecessor = pred;
    }

    /// Gets the current predecessor for phi evaluation.
    #[must_use]
    pub fn predecessor(&self) -> Option<usize> {
        self.predecessor
    }

    // Block Evaluation

    /// Evaluates all phi nodes in a block.
    ///
    /// **REQUIRES** a predecessor to be set via [`set_predecessor`](Self::set_predecessor).
    /// If no predecessor is set, phi results will be `None` (removed from value map).
    ///
    /// # Phi Node Semantics
    ///
    /// Phi nodes execute "simultaneously" - all source values are read BEFORE any
    /// results are written. This is critical for correct swap semantics:
    ///
    /// ```text
    /// v1 = phi(v2 from pred)
    /// v2 = phi(v1 from pred)
    /// ```
    ///
    /// This swaps v1 and v2. If we wrote v1 before reading for v2, we'd get the
    /// wrong value. The implementation uses a two-phase approach:
    /// 1. Read all source values into a temporary buffer
    /// 2. Write all results from the buffer
    pub fn evaluate_phis(&mut self, block_idx: usize) {
        let Some(block) = self.ssa.block(block_idx) else {
            return;
        };

        // Phase 1: Read all phi source values BEFORE any writes
        // This ensures correct "simultaneous" phi semantics (no swap problem)
        let phi_results: Vec<(SsaVarId, Option<SymbolicExpr<T>>)> = block
            .phi_nodes()
            .iter()
            .map(|phi| {
                let result = phi.result();
                // REQUIRE predecessor - no fallback, no merging
                let value = self.predecessor.and_then(|pred| {
                    phi.operands()
                        .iter()
                        .find(|op| op.predecessor() == pred)
                        .and_then(|op| self.values.get(&op.value()).cloned())
                });
                (result, value)
            })
            .collect();

        // Phase 2: Write all results
        for (result, value) in phi_results {
            if let Some(v) = value {
                self.values.insert(result, v.clone());
                self.track_origin_state(result, &v);
            } else {
                // No predecessor or no operand from predecessor = no value
                self.values.remove(&result);
            }
        }
    }

    /// Evaluates all instructions in a block, updating tracked values.
    ///
    /// This evaluates phi nodes first (if predecessor is set), then
    /// evaluates all other instructions in order.
    pub fn evaluate_block(&mut self, block_idx: usize) {
        // Record path if tracking is enabled
        if self.config.track_path {
            self.path.push(block_idx);
        }

        // First evaluate phi nodes
        self.evaluate_phis(block_idx);

        // Then evaluate instructions
        let Some(block) = self.ssa.block(block_idx) else {
            return;
        };

        for instr in block.instructions() {
            self.evaluate_op(instr.op());
        }
    }

    /// Evaluates a sequence of blocks in order.
    ///
    /// This is useful for evaluating a path through the CFG.
    /// Note: This does not set predecessors automatically.
    pub fn evaluate_blocks(&mut self, block_indices: &[usize]) {
        for &block_idx in block_indices {
            self.evaluate_block(block_idx);
        }
    }

    /// Evaluates a sequence of blocks along a path.
    ///
    /// For each block after the first, sets the predecessor to the previous
    /// block before evaluation. This enables accurate phi node evaluation.
    pub fn evaluate_path(&mut self, path: &[usize]) {
        for (i, &block_idx) in path.iter().enumerate() {
            if i > 0 {
                if let Some(&prev) = path.get(i.saturating_sub(1)) {
                    self.set_predecessor(Some(prev));
                }
            }
            self.evaluate_block(block_idx);
        }
    }

    // Fixed-Point Iteration for Loops

    /// Evaluates a loop until values reach a fixed point.
    ///
    /// This is useful for analyzing loops where variable values may change each
    /// iteration until they stabilize. The method iterates up to `max_iterations`
    /// times, or until all tracked values stop changing.
    ///
    /// # Arguments
    ///
    /// * `loop_blocks` - The blocks that form the loop body (in execution order)
    /// * `max_iterations` - Maximum number of iterations before giving up
    ///
    /// # Returns
    ///
    /// The number of iterations performed before reaching fixed point (or max).
    pub fn evaluate_loop_to_fixpoint(
        &mut self,
        loop_blocks: &[usize],
        max_iterations: usize,
    ) -> usize {
        if loop_blocks.is_empty() {
            return 0;
        }

        // Only variables defined inside the loop can change between iterations;
        // values defined outside are invariant during loop evaluation. Snapshot
        // and compare just those instead of deep-cloning the entire value map
        // every iteration.
        let mut loop_vars: Vec<SsaVarId> = Vec::new();
        for &block_idx in loop_blocks {
            if let Some(block) = self.ssa.block(block_idx) {
                for phi in block.phi_nodes() {
                    loop_vars.push(phi.result());
                }
                for instr in block.instructions() {
                    for dest in instr.op().defs() {
                        loop_vars.push(dest);
                    }
                }
            }
        }
        loop_vars.sort_unstable();
        loop_vars.dedup();

        for iteration in 0..max_iterations {
            // Snapshot only the loop-defined values.
            let snapshot: Vec<Option<SymbolicExpr<T>>> = loop_vars
                .iter()
                .map(|v| self.values.get(v).cloned())
                .collect();

            // Evaluate all loop blocks
            for (i, &block_idx) in loop_blocks.iter().enumerate() {
                if i > 0 {
                    if let Some(&prev) = loop_blocks.get(i.saturating_sub(1)) {
                        self.set_predecessor(Some(prev));
                    }
                } else if loop_blocks.len() > 1 {
                    // First block - predecessor is the last block (loop back edge)
                    if let Some(&last) = loop_blocks.last() {
                        self.set_predecessor(Some(last));
                    }
                }
                self.evaluate_block(block_idx);
            }

            // Fixed point reached when no loop-defined value changed.
            let stable = loop_vars
                .iter()
                .zip(snapshot.iter())
                .all(|(v, old)| self.values.get(v) == old.as_ref());
            if stable {
                return iteration.saturating_add(1);
            }
        }

        // Didn't reach fixed point - mark variables that changed as widened
        self.widen_unstable_values(loop_blocks);
        max_iterations
    }

    /// Widens values that didn't stabilize in a loop to Unknown.
    ///
    /// This is called when fixed-point iteration doesn't converge. Variables
    /// defined in loop blocks that still have different values are marked Unknown.
    fn widen_unstable_values(&mut self, loop_blocks: &[usize]) {
        // Find all variables defined in the loop
        for &block_idx in loop_blocks {
            let Some(block) = self.ssa.block(block_idx) else {
                continue;
            };

            // Mark phi results as unknown (they depend on loop iteration)
            for phi in block.phi_nodes() {
                self.values.remove(&phi.result());
            }

            // Check instructions for variables that might not have stabilized
            for instr in block.instructions() {
                // If this op defines variables, consider widening them
                for dest in instr.op().defs() {
                    // Keep concrete values if they're stable, widen symbolic to unknown
                    if let Some(expr) = self.values.get(&dest) {
                        if !expr.is_constant() {
                            // Symbolic values that didn't stabilize become unknown
                            self.values.remove(&dest);
                        }
                    }
                }
            }
        }
    }

    /// Evaluates a loop with a specific iteration count.
    ///
    /// This is useful when you know exactly how many times a loop should run
    /// (e.g., from a constant loop bound).
    pub fn evaluate_loop_iterations(&mut self, loop_blocks: &[usize], iterations: usize) {
        for _ in 0..iterations {
            for (i, &block_idx) in loop_blocks.iter().enumerate() {
                if i > 0 {
                    if let Some(&prev) = loop_blocks.get(i.saturating_sub(1)) {
                        self.set_predecessor(Some(prev));
                    }
                }
                self.evaluate_block(block_idx);
            }
        }
    }

    /// Evaluates a single SSA operation, updating tracked values.
    ///
    /// Returns the computed expression for operations that produce a result,
    /// or `None` for operations without results (stores, branches, etc.) or
    /// when the result is unknown.
    pub fn evaluate_op(&mut self, op: &SsaOp<T>) -> Option<SymbolicExpr<T>> {
        match op {
            SsaOp::Const { dest, value } => {
                let expr = SymbolicExpr::constant(value.clone());
                self.values.insert(*dest, expr.clone());
                self.track_origin_state(*dest, &expr);
                Some(expr)
            }

            SsaOp::Copy { dest, src } => {
                let value = self.values.get(src).cloned();
                if let Some(v) = value {
                    self.values.insert(*dest, v.clone());
                    self.track_origin_state(*dest, &v);
                    Some(v)
                } else {
                    self.values.remove(dest);
                    None
                }
            }

            SsaOp::LoadLocal { dest, local_index } => {
                let value = self.local_state.get(local_index).cloned();
                if let Some(v) = value {
                    self.values.insert(*dest, v.clone());
                    Some(v)
                } else {
                    self.values.remove(dest);
                    None
                }
            }

            SsaOp::LoadArg { dest, arg_index } => {
                let value = self.arg_state.get(arg_index).cloned();
                if let Some(v) = value {
                    self.values.insert(*dest, v.clone());
                    Some(v)
                } else {
                    self.values.remove(dest);
                    None
                }
            }

            SsaOp::Add {
                dest, left, right, ..
            } => self.eval_binary_op(*dest, *left, *right, SymbolicOp::Add),

            SsaOp::Sub {
                dest, left, right, ..
            } => self.eval_binary_op(*dest, *left, *right, SymbolicOp::Sub),

            SsaOp::Mul {
                dest, left, right, ..
            } => self.eval_binary_op(*dest, *left, *right, SymbolicOp::Mul),

            SsaOp::Div {
                dest,
                left,
                right,
                unsigned,
                ..
            } => {
                let op = if *unsigned {
                    SymbolicOp::DivU
                } else {
                    SymbolicOp::DivS
                };
                self.eval_binary_op(*dest, *left, *right, op)
            }

            SsaOp::Rem {
                dest,
                left,
                right,
                unsigned,
                ..
            } => {
                let op = if *unsigned {
                    SymbolicOp::RemU
                } else {
                    SymbolicOp::RemS
                };
                self.eval_binary_op(*dest, *left, *right, op)
            }

            SsaOp::Xor {
                dest, left, right, ..
            } => self.eval_binary_op(*dest, *left, *right, SymbolicOp::Xor),

            SsaOp::And {
                dest, left, right, ..
            } => self.eval_binary_op(*dest, *left, *right, SymbolicOp::And),

            SsaOp::Or {
                dest, left, right, ..
            } => self.eval_binary_op(*dest, *left, *right, SymbolicOp::Or),

            SsaOp::Shl {
                dest,
                value,
                amount,
                ..
            } => self.eval_binary_op(*dest, *value, *amount, SymbolicOp::Shl),

            SsaOp::Shr {
                dest,
                value,
                amount,
                unsigned,
                ..
            } => {
                let op = if *unsigned {
                    SymbolicOp::ShrU
                } else {
                    SymbolicOp::ShrS
                };
                self.eval_binary_op(*dest, *value, *amount, op)
            }

            SsaOp::Neg { dest, operand, .. } => {
                self.eval_unary_op(*dest, *operand, SymbolicOp::Neg)
            }

            SsaOp::Not { dest, operand, .. } => {
                self.eval_unary_op(*dest, *operand, SymbolicOp::Not)
            }

            // Rotations (like shifts - pure, two operands)
            SsaOp::Rol {
                dest,
                value,
                amount,
            } => {
                let v = self.values.get(value).and_then(|e| e.as_i64());
                let a = self.values.get(amount).and_then(|e| e.as_i64());
                match (v, a) {
                    (Some(v), Some(a)) => {
                        let shift = (a & 0x1f) as u32;
                        let v32 = v as u32;
                        let result = v32.rotate_left(shift);
                        let expr = SymbolicExpr::constant(ConstValue::I32(result as i32));
                        self.values.insert(*dest, expr.clone());
                        self.track_origin_state(*dest, &expr);
                        Some(expr)
                    }
                    _ => {
                        self.values.remove(dest);
                        None
                    }
                }
            }

            SsaOp::Ror {
                dest,
                value,
                amount,
            } => {
                let v = self.values.get(value).and_then(|e| e.as_i64());
                let a = self.values.get(amount).and_then(|e| e.as_i64());
                match (v, a) {
                    (Some(v), Some(a)) => {
                        let shift = (a & 0x1f) as u32;
                        let v32 = v as u32;
                        let result = v32.rotate_right(shift);
                        let expr = SymbolicExpr::constant(ConstValue::I32(result as i32));
                        self.values.insert(*dest, expr.clone());
                        self.track_origin_state(*dest, &expr);
                        Some(expr)
                    }
                    _ => {
                        self.values.remove(dest);
                        None
                    }
                }
            }

            SsaOp::Rcl {
                dest,
                value,
                amount,
            } => {
                let v = self.values.get(value).and_then(|e| e.as_i64());
                let a = self.values.get(amount).and_then(|e| e.as_i64());
                match (v, a) {
                    (Some(v), Some(a)) => {
                        let shift = (a & 0x1f) as u32;
                        let v32 = v as u32;
                        let result = v32.rotate_left(shift);
                        let expr = SymbolicExpr::constant(ConstValue::I32(result as i32));
                        self.values.insert(*dest, expr.clone());
                        self.track_origin_state(*dest, &expr);
                        Some(expr)
                    }
                    _ => {
                        self.values.remove(dest);
                        None
                    }
                }
            }

            SsaOp::Rcr {
                dest,
                value,
                amount,
            } => {
                let v = self.values.get(value).and_then(|e| e.as_i64());
                let a = self.values.get(amount).and_then(|e| e.as_i64());
                match (v, a) {
                    (Some(v), Some(a)) => {
                        let shift = (a & 0x1f) as u32;
                        let v32 = v as u32;
                        let result = v32.rotate_right(shift);
                        let expr = SymbolicExpr::constant(ConstValue::I32(result as i32));
                        self.values.insert(*dest, expr.clone());
                        self.track_origin_state(*dest, &expr);
                        Some(expr)
                    }
                    _ => {
                        self.values.remove(dest);
                        None
                    }
                }
            }

            // Bit manipulation (like unary ops - pure, one operand)
            SsaOp::BSwap { dest, src } => {
                let v = self.values.get(src).and_then(|e| e.as_i64());
                match v {
                    Some(v) => {
                        let result = (v as u32).swap_bytes();
                        let expr = SymbolicExpr::constant(ConstValue::I32(result as i32));
                        self.values.insert(*dest, expr.clone());
                        self.track_origin_state(*dest, &expr);
                        Some(expr)
                    }
                    _ => {
                        self.values.remove(dest);
                        None
                    }
                }
            }

            SsaOp::BRev { dest, src } => {
                let v = self.values.get(src).and_then(|e| e.as_i64());
                match v {
                    Some(v) => {
                        let result = (v as u32).reverse_bits();
                        let expr = SymbolicExpr::constant(ConstValue::I32(result as i32));
                        self.values.insert(*dest, expr.clone());
                        self.track_origin_state(*dest, &expr);
                        Some(expr)
                    }
                    _ => {
                        self.values.remove(dest);
                        None
                    }
                }
            }

            SsaOp::BitScanForward { dest, src } => {
                let v = self.values.get(src).and_then(|e| e.as_i64());
                match v {
                    Some(v) => {
                        let v32 = v as u32;
                        let result = if v32 == 0 {
                            32
                        } else {
                            v32.trailing_zeros() as i32
                        };
                        let expr = SymbolicExpr::constant(ConstValue::I32(result));
                        self.values.insert(*dest, expr.clone());
                        self.track_origin_state(*dest, &expr);
                        Some(expr)
                    }
                    _ => {
                        self.values.remove(dest);
                        None
                    }
                }
            }

            SsaOp::BitScanReverse { dest, src } => {
                let v = self.values.get(src).and_then(|e| e.as_i64());
                match v {
                    Some(v) => {
                        let v32 = v as u32;
                        let result = if v32 == 0 {
                            32
                        } else {
                            31u32.saturating_sub(v32.leading_zeros()) as i32
                        };
                        let expr = SymbolicExpr::constant(ConstValue::I32(result));
                        self.values.insert(*dest, expr.clone());
                        self.track_origin_state(*dest, &expr);
                        Some(expr)
                    }
                    _ => {
                        self.values.remove(dest);
                        None
                    }
                }
            }

            SsaOp::Popcount { dest, src } => {
                let v = self.values.get(src).and_then(|e| e.as_i64());
                match v {
                    Some(v) => {
                        let result = (v as u32).count_ones() as i32;
                        let expr = SymbolicExpr::constant(ConstValue::I32(result));
                        self.values.insert(*dest, expr.clone());
                        self.track_origin_state(*dest, &expr);
                        Some(expr)
                    }
                    _ => {
                        self.values.remove(dest);
                        None
                    }
                }
            }

            SsaOp::Parity { dest, src } => {
                let v = self.values.get(src).and_then(|e| e.as_i64());
                match v {
                    Some(v) => {
                        let result = if (v as u32).count_ones() % 2 == 1 {
                            1
                        } else {
                            0
                        };
                        let expr = SymbolicExpr::constant(ConstValue::I32(result));
                        self.values.insert(*dest, expr.clone());
                        self.track_origin_state(*dest, &expr);
                        Some(expr)
                    }
                    _ => {
                        self.values.remove(dest);
                        None
                    }
                }
            }

            // Conditional select (pure, conditional move)
            SsaOp::Select {
                dest,
                condition,
                true_val,
                false_val,
            } => {
                let cond = self.values.get(condition).and_then(|e| e.as_i64());
                match cond {
                    Some(c) if c != 0 => {
                        if let Some(expr) = self.values.get(true_val).cloned() {
                            self.values.insert(*dest, expr.clone());
                            self.track_origin_state(*dest, &expr);
                            Some(expr)
                        } else {
                            self.values.remove(dest);
                            None
                        }
                    }
                    Some(_) => {
                        if let Some(expr) = self.values.get(false_val).cloned() {
                            self.values.insert(*dest, expr.clone());
                            self.track_origin_state(*dest, &expr);
                            Some(expr)
                        } else {
                            self.values.remove(dest);
                            None
                        }
                    }
                    None => {
                        self.values.remove(dest);
                        None
                    }
                }
            }

            SsaOp::Ceq { dest, left, right } => {
                self.eval_binary_op(*dest, *left, *right, SymbolicOp::Eq)
            }

            SsaOp::Cgt {
                dest,
                left,
                right,
                unsigned,
            } => {
                let op = if *unsigned {
                    SymbolicOp::GtU
                } else {
                    SymbolicOp::GtS
                };
                self.eval_binary_op(*dest, *left, *right, op)
            }

            SsaOp::Clt {
                dest,
                left,
                right,
                unsigned,
            } => {
                let op = if *unsigned {
                    SymbolicOp::LtU
                } else {
                    SymbolicOp::LtS
                };
                self.eval_binary_op(*dest, *left, *right, op)
            }

            SsaOp::Conv {
                dest,
                operand,
                target,
                unsigned,
                ..
            } => {
                let value = self.values.get(operand).cloned();
                if let Some(expr) = value {
                    if let Some(v) = expr.as_i64() {
                        // Route apply_conversion through Target so the SsaType
                        // pattern-matching stays in the host. Defaults to
                        // wrapping the raw i64 (sufficient for hosts without
                        // CIL-specific integer-to-typed-constant conversion).
                        let ptr_bytes = self.pointer_size.bytes() as u32;
                        let converted = T::evaluate_int_conv(v, target, *unsigned, ptr_bytes)
                            .unwrap_or(ConstValue::I64(v));
                        let result = SymbolicExpr::constant(converted);
                        self.values.insert(*dest, result.clone());
                        self.track_origin_state(*dest, &result);
                        Some(result)
                    } else {
                        // Symbolic/Unknown pass through (conversions don't change symbolic structure)
                        self.values.insert(*dest, expr.clone());
                        self.track_origin_state(*dest, &expr);
                        Some(expr)
                    }
                } else {
                    self.values.remove(dest);
                    None
                }
            }

            // Operations with Option<SsaVarId> dest that produce unknown results
            SsaOp::Call { dest, .. }
            | SsaOp::CallVirt { dest, .. }
            | SsaOp::CallIndirect { dest, .. } => {
                if let Some(d) = dest {
                    self.values.remove(d);
                }
                None
            }

            // Memory operations (when tracking is enabled)
            SsaOp::LoadStaticField { dest, field } => {
                if self.config.track_memory {
                    let location = MemoryLocation::StaticField(field.clone());
                    if let Some(stored_var) = self.memory.load(&location) {
                        // Propagate the stored value
                        if let Some(expr) = self.values.get(&stored_var).cloned() {
                            self.values.insert(*dest, expr.clone());
                            return Some(expr);
                        }
                        self.values
                            .insert(*dest, SymbolicExpr::variable(stored_var));
                        return Some(SymbolicExpr::variable(stored_var));
                    }
                }
                self.values.remove(dest);
                None
            }

            SsaOp::StoreStaticField { value, field } => {
                if self.config.track_memory {
                    let location = MemoryLocation::StaticField(field.clone());
                    // Use 0 as version for simple tracking (version not critical for evaluation)
                    self.memory.store(location, *value, 0);
                }
                None
            }

            SsaOp::LoadField {
                dest,
                object,
                field,
            } => {
                if self.config.track_memory {
                    let location = MemoryLocation::InstanceField(*object, field.clone());
                    if let Some(stored_var) = self.memory.load(&location) {
                        if let Some(expr) = self.values.get(&stored_var).cloned() {
                            self.values.insert(*dest, expr.clone());
                            return Some(expr);
                        }
                        self.values
                            .insert(*dest, SymbolicExpr::variable(stored_var));
                        return Some(SymbolicExpr::variable(stored_var));
                    }
                }
                self.values.remove(dest);
                None
            }

            SsaOp::StoreField {
                object,
                field,
                value,
            } => {
                if self.config.track_memory {
                    let location = MemoryLocation::InstanceField(*object, field.clone());
                    self.memory.store(location, *value, 0);
                }
                None
            }

            // Address-of-local/arg: the address is taken, meaning external code
            // can write to this local/arg through the pointer. Invalidate the
            // tracked state so subsequent LoadLocal/LoadArg returns Unknown
            // instead of the stale pre-address-taken value.
            //
            // This is critical for patterns like `Monitor.Enter(obj, ref bool)`
            // where the CLR writes `true` to the bool via the by-reference
            // parameter. Without invalidation, the evaluator would see the
            // initial value (false/0) and incorrectly fold branches that check
            // the lock flag.
            SsaOp::LoadLocalAddr {
                dest, local_index, ..
            } => {
                self.values.remove(dest);
                self.local_state.remove(local_index);
                None
            }
            SsaOp::LoadArgAddr {
                dest, arg_index, ..
            } => {
                self.values.remove(dest);
                self.arg_state.remove(arg_index);
                None
            }

            // Operations with SsaVarId dest that produce unknown results
            SsaOp::NewObj { dest, .. }
            | SsaOp::NewArr { dest, .. }
            | SsaOp::LoadElement { dest, .. }
            | SsaOp::LoadIndirect { dest, .. }
            | SsaOp::Box { dest, .. }
            | SsaOp::Unbox { dest, .. }
            | SsaOp::UnboxAny { dest, .. }
            | SsaOp::CastClass { dest, .. }
            | SsaOp::IsInst { dest, .. }
            | SsaOp::ArrayLength { dest, .. }
            | SsaOp::LoadToken { dest, .. }
            | SsaOp::SizeOf { dest, .. }
            | SsaOp::Ckfinite { dest, .. }
            | SsaOp::LocalAlloc { dest, .. }
            | SsaOp::LoadFunctionPtr { dest, .. }
            | SsaOp::LoadVirtFunctionPtr { dest, .. }
            | SsaOp::LoadFieldAddr { dest, .. }
            | SsaOp::LoadStaticFieldAddr { dest, .. }
            | SsaOp::LoadElementAddr { dest, .. }
            | SsaOp::LoadObj { dest, .. } => {
                self.values.remove(dest);
                None
            }

            // Operations without results (stores, branches, etc.)
            _ => None,
        }
    }

    /// Helper to evaluate a binary operation.
    fn eval_binary_op(
        &mut self,
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        op: SymbolicOp,
    ) -> Option<SymbolicExpr<T>> {
        let left_expr = self.values.get(&left)?;
        let right_expr = self.values.get(&right)?;
        if left_expr.reaches_recursive_depth_limit() || right_expr.reaches_recursive_depth_limit() {
            let result = SymbolicExpr::variable(dest);
            self.values.insert(dest, result.clone());
            self.track_origin_state(dest, &result);
            return Some(result);
        }

        // Build expression and simplify (handles constant folding automatically)
        let result = SymbolicExpr::binary(op, left_expr.clone(), right_expr.clone())
            .simplify(self.pointer_size);

        // Mask native int/uint results to target pointer width
        let result = self.mask_symbolic_native(result);

        self.values.insert(dest, result.clone());
        self.track_origin_state(dest, &result);
        Some(result)
    }

    /// Helper to evaluate a unary operation.
    fn eval_unary_op(
        &mut self,
        dest: SsaVarId,
        operand: SsaVarId,
        op: SymbolicOp,
    ) -> Option<SymbolicExpr<T>> {
        let operand_expr = self.values.get(&operand)?;
        if operand_expr.reaches_recursive_depth_limit() {
            let result = SymbolicExpr::variable(dest);
            self.values.insert(dest, result.clone());
            self.track_origin_state(dest, &result);
            return Some(result);
        }

        // Build expression and simplify
        let result = SymbolicExpr::unary(op, operand_expr.clone()).simplify(self.pointer_size);

        // Mask native int/uint results to target pointer width
        let result = self.mask_symbolic_native(result);

        self.values.insert(dest, result.clone());
        self.track_origin_state(dest, &result);
        Some(result)
    }

    /// Updates local/arg state when a variable with Local(N) or Argument(N)
    /// origin receives a value. This enables `LoadLocal`/`LoadArg` to read
    /// the most recently written value for the corresponding CIL local/arg.
    fn track_origin_state(&mut self, var: SsaVarId, value: &SymbolicExpr<T>) {
        if let Some(ssa_var) = self.ssa.variable(var) {
            match ssa_var.origin() {
                VariableOrigin::Local(idx) => {
                    self.local_state.insert(idx, value.clone());
                }
                VariableOrigin::Argument(idx) => {
                    self.arg_state.insert(idx, value.clone());
                }
                _ => {}
            }
        }
    }

    /// Masks a `SymbolicExpr` constant to the target pointer width if it contains
    /// a `NativeInt` or `NativeUInt` value.
    fn mask_symbolic_native(&self, expr: SymbolicExpr<T>) -> SymbolicExpr<T> {
        if let Some(cv) = expr.as_constant() {
            match cv {
                ConstValue::NativeInt(_) | ConstValue::NativeUInt(_) => {
                    SymbolicExpr::constant(cv.clone().mask_native(self.pointer_size))
                }
                _ => expr,
            }
        } else {
            expr
        }
    }

    /// Tries to resolve a variable's value by tracing back through its definition.
    ///
    /// This is useful when a variable's value depends on earlier computations
    /// that haven't been evaluated yet. It recursively evaluates dependencies.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to resolve
    /// * `max_depth` - Maximum recursion depth to prevent infinite loops
    pub fn resolve_with_trace(
        &mut self,
        var: SsaVarId,
        max_depth: usize,
    ) -> Option<SymbolicExpr<T>> {
        // Already known?
        if let Some(v) = self.values.get(&var) {
            return Some(v.clone());
        }

        if max_depth == 0 {
            return None;
        }

        // Find the definition of this variable
        let ssa_var = self.ssa.variable(var)?;
        let def_site = ssa_var.def_site();
        let block = self.ssa.block(def_site.block)?;
        // Is it defined by a phi node? Without path context, it's unknown
        let instr_idx = def_site.instruction?;
        let instr = block.instruction(instr_idx)?;
        let op = instr.op();

        // Recursively resolve operands first
        for operand in op.uses() {
            if !self.values.contains_key(&operand) {
                if let Some(resolved) =
                    self.resolve_with_trace(operand, max_depth.saturating_sub(1))
                {
                    self.values.insert(operand, resolved);
                }
            }
        }

        // Now evaluate this operation
        self.evaluate_op(op)
    }

    /// Tries to evaluate a variable by tracing back through its definition.
    ///
    /// Alias for [`resolve_with_trace`](Self::resolve_with_trace) that returns
    /// `Option<i64>` for API compatibility.
    pub fn evaluate_with_trace(&mut self, var: SsaVarId, max_depth: usize) -> Option<i64> {
        self.resolve_with_trace(var, max_depth)
            .and_then(|e| e.as_i64())
    }

    /// Determines the next block to execute based on the terminator of the given block.
    ///
    /// This is the core method for control flow analysis. It evaluates the terminating
    /// instruction of a block and determines which block(s) execution should continue to.
    ///
    /// # Returns
    ///
    /// - `ControlFlow::Continue(block)` - Continue to the specified block
    /// - `ControlFlow::Terminal` - No successor (return, throw, etc.)
    /// - `ControlFlow::Unknown` - Cannot determine (condition is unknown/symbolic)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    /// eval.set_concrete(state_var, initial_state);
    /// eval.evaluate_block(0);
    ///
    /// match eval.next_block(0) {
    ///     ControlFlow::Continue(next) => { /* continue to next */ }
    ///     ControlFlow::Terminal => { /* execution ends */ }
    ///     ControlFlow::Unknown => { /* cannot determine */ }
    /// }
    /// ```
    #[must_use]
    pub fn next_block(&self, block_idx: usize) -> ControlFlow {
        let Some(block) = self.ssa.block(block_idx) else {
            return ControlFlow::Unknown;
        };

        // Find the terminating instruction
        let terminator = block
            .instructions()
            .iter()
            .rev()
            .find(|instr| instr.op().is_terminator());

        let Some(instr) = terminator else {
            // No terminator - fall through to next block if it exists
            let next_idx = block_idx.saturating_add(1);
            if next_idx < self.ssa.block_count() {
                return ControlFlow::Continue(next_idx);
            }
            return ControlFlow::Unknown;
        };

        self.evaluate_control_flow(instr.op())
    }

    /// Evaluates a control flow operation to determine the next block.
    ///
    /// Uses typed `ConstValue` operations for comparisons and truthiness checks.
    fn evaluate_control_flow(&self, op: &SsaOp<T>) -> ControlFlow {
        match op {
            // Unconditional jumps
            SsaOp::Jump { target } | SsaOp::Leave { target } => ControlFlow::Continue(*target),

            // Conditional branch (bool condition)
            SsaOp::Branch {
                condition,
                true_target,
                false_target,
            } => match self.get_concrete(*condition) {
                Some(v) => {
                    // Non-zero is true in CIL
                    if v.is_zero() {
                        ControlFlow::Continue(*false_target)
                    } else {
                        ControlFlow::Continue(*true_target)
                    }
                }
                None => ControlFlow::Unknown,
            },

            // Compare and branch
            SsaOp::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                true_target,
                false_target,
            } => {
                let left_val = self.get_concrete(*left);
                let right_val = self.get_concrete(*right);

                match (left_val, right_val) {
                    (Some(l), Some(r)) => {
                        let result = Self::evaluate_comparison(l, r, *cmp, *unsigned);
                        if result {
                            ControlFlow::Continue(*true_target)
                        } else {
                            ControlFlow::Continue(*false_target)
                        }
                    }
                    _ => ControlFlow::Unknown,
                }
            }

            // Switch - needs a non-negative integer index
            SsaOp::Switch {
                value,
                targets,
                default,
            } => match self.get_concrete(*value).and_then(ConstValue::as_u64) {
                Some(v) => {
                    #[allow(clippy::cast_possible_truncation)]
                    let idx = v as usize;
                    if let Some(&target) = targets.get(idx) {
                        ControlFlow::Continue(target)
                    } else {
                        ControlFlow::Continue(*default)
                    }
                }
                None => ControlFlow::Unknown,
            },

            // Terminal instructions
            SsaOp::Return { .. }
            | SsaOp::Throw { .. }
            | SsaOp::Rethrow
            | SsaOp::EndFinally
            | SsaOp::EndFilter { .. }
            | SsaOp::InterruptReturn
            | SsaOp::Unreachable => ControlFlow::Terminal,

            // Not a control flow operation
            _ => ControlFlow::Unknown,
        }
    }

    /// Evaluates a comparison between two typed constant values.
    ///
    /// Uses the typed comparison methods on `ConstValue` which properly
    /// handle signedness based on the operand types.
    pub fn evaluate_comparison(
        left: &ConstValue<T>,
        right: &ConstValue<T>,
        cmp: CmpKind,
        unsigned: bool,
    ) -> bool {
        match cmp {
            CmpKind::Eq => left.ceq(right).is_some_and(|v| !v.is_zero()),
            CmpKind::Ne => left.ceq(right).is_some_and(|v| v.is_zero()),
            CmpKind::Lt => if unsigned {
                left.clt_un(right)
            } else {
                left.clt(right)
            }
            .is_some_and(|v| !v.is_zero()),
            CmpKind::Le => {
                // x <= y is !(x > y)
                if unsigned {
                    left.cgt_un(right)
                } else {
                    left.cgt(right)
                }
                .is_some_and(|v| v.is_zero())
            }
            CmpKind::Gt => if unsigned {
                left.cgt_un(right)
            } else {
                left.cgt(right)
            }
            .is_some_and(|v| !v.is_zero()),
            CmpKind::Ge => {
                // x >= y is !(x < y)
                if unsigned {
                    left.clt_un(right)
                } else {
                    left.clt(right)
                }
                .is_some_and(|v| v.is_zero())
            }
        }
    }

    /// Executes the SSA function starting from a given block and records the trace.
    ///
    /// This method steps through the SSA, evaluating each block and following
    /// control flow decisions based on computed values. It records the sequence
    /// of blocks visited and optionally captures state values at each step.
    ///
    /// # Arguments
    ///
    /// * `start_block` - The block to start execution from
    /// * `state_var` - Optional variable to capture state values (for CFF analysis)
    /// * `max_steps` - Maximum number of blocks to visit (prevents infinite loops)
    ///
    /// # Returns
    ///
    /// An [`ExecutionTrace`] containing the visited blocks and state values.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut eval = SsaEvaluator::new(&ssa, PointerSize::Bit64);
    /// eval.set_concrete(state_var, initial_state);
    ///
    /// let trace = eval.execute(0, Some(state_var), 1000);
    /// for (block, state) in trace.blocks().iter().zip(trace.states()) {
    ///     println!("Block {}: state = {:?}", block, state);
    /// }
    /// ```
    pub fn execute(
        &mut self,
        start_block: usize,
        state_var: Option<SsaVarId>,
        max_steps: usize,
    ) -> ExecutionTrace<T> {
        let mut trace = ExecutionTrace::new(max_steps);
        let mut current_block = start_block;

        loop {
            // Check if we've hit the limit
            if trace.hit_limit() {
                break;
            }

            // Record the current state before evaluation
            let state = state_var.and_then(|v| self.get_concrete(v).cloned());
            trace.record_block(current_block, state);

            // Set predecessor for phi evaluation
            if let Some(prev) = trace.blocks().iter().rev().nth(1) {
                self.set_predecessor(Some(*prev));
            }

            // Evaluate the block
            self.evaluate_block(current_block);

            // Determine next block
            match self.next_block(current_block) {
                ControlFlow::Continue(next) => {
                    current_block = next;
                }
                ControlFlow::Terminal => {
                    trace.mark_complete();
                    break;
                }
                ControlFlow::Unknown => {
                    // Can't determine next block - stop execution
                    break;
                }
            }
        }

        trace
    }

    /// Executes starting from block 0 with default settings.
    ///
    /// This is a convenience method for simple cases where you want to execute
    /// from the entry block without state tracking.
    pub fn execute_from_entry(&mut self, max_steps: usize) -> ExecutionTrace<T> {
        self.execute(0, None, max_steps)
    }
}
