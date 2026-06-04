//! Opaque predicate detection and removal pass.
//!
//! Opaque predicates are conditional expressions that always evaluate to the same
//! value at runtime, but appear complex to static analysis. Obfuscators use them
//! to confuse decompilers and analysis tools.
//!
//! # Detection Strategies
//!
//! ## Basic Patterns
//! - **Self-comparison**: `x == x`, `x != x`, `x < x`, `x > x`
//! - **Identity operations**: `x ^ x == 0`, `x - x == 0`
//! - **Zero operations**: `x * 0`, `x & 0`, `x % 1`
//!
//! ## Number-Theoretic Predicates
//! - **Consecutive integers**: `(x * (x + 1)) % 2 == 0` (always true)
//! - **Square properties**: `x² >= 0` (always true for integers)
//! - **Modular arithmetic**: `(x² - x) % 2 == 0` (always true)
//!
//! ## Type-Based Predicates
//! - **Null checks**: `obj != null` after `newobj` (always true)
//! - **Array length**: `arr.Length >= 0` (always true)
//!
//! ## Range-Based Predicates
//! - **Unsigned bounds**: `unsigned_x >= 0` (always true)
//! - **Correlated conditions**: `if (x > 5) { if (x < 3) { dead } }`
//!
//! # Example
//!
//! Before:
//! ```text
//! v0 = 5
//! v1 = ceq v0, v0    // Always true
//! branch v1, B1, B2  // Always goes to B1
//! ```
//!
//! After:
//! ```text
//! v0 = 5
//! v1 = true
//! jump B1
//! ```

use std::collections::BTreeMap;

use crate::{
    analysis::{defuse::DefUseIndex, evaluator::SsaEvaluator, range::ValueRange},
    bitset::BitSet,
    events::{EventKind, EventListener},
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        instruction::SsaInstruction,
        ops::SsaOp,
        value::ConstValue,
        variable::SsaVarId,
    },
    pointer::PointerSize,
    target::Target,
};

/// The result of analyzing a potential opaque predicate expression.
///
/// This tri-state result captures whether a comparison or branch condition
/// can be statically resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateResult {
    /// The predicate always evaluates to a non-zero (true) value at runtime.
    AlwaysTrue,
    /// The predicate always evaluates to zero (false) at runtime.
    AlwaysFalse,
    /// The predicate's value cannot be statically determined.
    Unknown,
}

impl PredicateResult {
    /// Converts to an optional boolean.
    ///
    /// # Returns
    ///
    /// `Some(true)` for `AlwaysTrue`, `Some(false)` for `AlwaysFalse`, `None` for `Unknown`.
    #[must_use]
    pub fn as_bool(self) -> Option<bool> {
        match self {
            Self::AlwaysTrue => Some(true),
            Self::AlwaysFalse => Some(false),
            Self::Unknown => None,
        }
    }

    /// Negates the predicate result.
    ///
    /// `AlwaysTrue` becomes `AlwaysFalse` and vice versa. `Unknown` stays `Unknown`.
    #[must_use]
    pub fn negate(self) -> Self {
        match self {
            Self::AlwaysTrue => Self::AlwaysFalse,
            Self::AlwaysFalse => Self::AlwaysTrue,
            Self::Unknown => Self::Unknown,
        }
    }
}

/// Result of analyzing a comparison for algebraic simplification.
///
/// Unlike `PredicateResult` which determines if a comparison is always true/false,
/// this enum represents transformations that simplify comparisons while preserving
/// their runtime behavior.
#[derive(Debug, Clone)]
enum ComparisonSimplification<T: Target> {
    /// Replace with a simpler comparison operation.
    SimplerOp {
        new_op: SsaOp<T>,
        reason: &'static str,
    },
    /// Replace with a copy of another variable (e.g., `(cmp) == 1` → `cmp`).
    Copy {
        dest: SsaVarId,
        src: SsaVarId,
        reason: &'static str,
    },
}

/// Cached definition information for efficient predicate analysis.
///
/// Wraps [`DefUseIndex`] for basic definition lookups and augments it with
/// specialized tracking: phi-defined variables, non-null provenance, and
/// computed [`ValueRange`]s.
struct DefinitionCache<T: Target> {
    /// Index mapping variables to their defining block, instruction, and op.
    index: DefUseIndex<T>,
    /// Bitset of variables defined by phi nodes (not covered by `DefUseIndex`).
    phi_defs: BitSet,
    /// Variables known to be non-null (produced by `NewObj`, `NewArr`, `Box`, or `LoadToken`).
    non_null_vars: BitSet,
    /// Computed [`ValueRange`]s for each variable. Constants get exact
    /// ranges, array lengths get non-negative ranges.
    ranges: BTreeMap<SsaVarId, ValueRange>,
}

impl<T: Target> DefinitionCache<T> {
    /// Builds the definition cache from an SSA function.
    ///
    /// Performs a single pass over all blocks to populate:
    /// - `index`: delegated to [`DefUseIndex::build_with_ops`] for var-to-op mapping.
    /// - `phi_defs`: bitset of variables defined by phi nodes (not tracked by `DefUseIndex`).
    /// - `non_null_vars`: bitset of variables produced by `NewObj`, `NewArr`, `Box`, or `LoadToken`.
    /// - `array_length_vars`: bitset of variables from `ArrayLength` ops.
    /// - `ranges`: [`ValueRange::constant`] for `Const` ops, [`ValueRange::non_negative`] for
    ///   `ArrayLength` ops.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to build the cache from.
    ///
    /// # Returns
    ///
    /// A fully populated `DefinitionCache` with definition index, phi-def tracking,
    /// non-null provenance, array-length provenance, and value ranges.
    fn build(ssa: &SsaFunction<T>) -> Self {
        // Use DefUseIndex for basic definition tracking
        let index = DefUseIndex::build_with_ops(ssa);

        let var_count = ssa.variable_count();
        let mut phi_defs = BitSet::new(var_count);
        let mut non_null_vars = BitSet::new(var_count);
        let mut ranges = BTreeMap::new();

        for (_block_idx, block) in ssa.iter_blocks() {
            // Process phi nodes (not covered by DefUseIndex)
            for phi in block.phi_nodes() {
                phi_defs.insert(phi.result().index());
            }

            // Process instructions for specialized tracking
            for instr in block.instructions() {
                let op = instr.op();
                if let Some(dest) = op.dest() {
                    // Track non-null producing operations and value ranges
                    match op {
                        SsaOp::NewObj { .. }
                        | SsaOp::NewArr { .. }
                        | SsaOp::Box { .. }
                        | SsaOp::LoadToken { .. } => {
                            // Non-null tracked separately (not a numeric range)
                            non_null_vars.insert(dest.index());
                        }
                        SsaOp::ArrayLength { .. } => {
                            ranges.insert(dest, ValueRange::non_negative());
                        }
                        SsaOp::Const { value, .. } => {
                            if let Some(v) = value.as_i64() {
                                ranges.insert(dest, ValueRange::constant(v));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Self {
            index,
            phi_defs,
            non_null_vars,
            ranges,
        }
    }

    /// Gets the defining operation for a variable.
    fn get_definition(&self, var: SsaVarId) -> Option<&SsaOp<T>> {
        self.index.def_op(var)
    }

    /// Checks if a variable is defined by a phi node.
    fn is_phi_defined(&self, var: SsaVarId) -> bool {
        self.phi_defs.contains(var.index())
    }

    /// Checks if a variable is known to be non-null.
    fn is_non_null(&self, var: SsaVarId) -> bool {
        self.non_null_vars.contains(var.index())
    }

    /// Gets the value range for a variable.
    fn get_range(&self, var: SsaVarId) -> Option<&ValueRange> {
        self.ranges.get(&var)
    }
}

/// Opaque predicate detection and removal pass.
///
/// Detects conditional expressions that always evaluate to the same value
/// (always true or always false) and simplifies branches, comparisons,
/// and phi nodes accordingly. Handles self-comparison, identity operations,
/// number-theoretic predicates, null checks, range-based predicates, and
/// nested predicate chains.
///
/// Generic over [`Target`]; the inner `PhantomData` lets the same struct
/// host all analysis methods without holding runtime state.
pub struct OpaquePredicatePass<T: Target>(std::marker::PhantomData<T>);

impl<T: Target> Default for OpaquePredicatePass<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Target> OpaquePredicatePass<T> {
    /// Creates a new opaque predicate pass.
    #[must_use]
    pub fn new() -> Self {
        Self(std::marker::PhantomData)
    }

    /// Maximum recursion depth for nested predicate analysis.
    ///
    /// Each level corresponds to one SSA instruction defining a comparison result
    /// that feeds into another comparison. 16 levels handles deeply nested opaque
    /// predicates from advanced obfuscators (e.g., PureLogs multi-level chains)
    /// while preventing stack overflow on pathological inputs.
    const MAX_PREDICATE_DEPTH: usize = 16;

    /// Analyzes a predicate operation with full context, dispatching by `SsaOp` kind.
    ///
    /// Pattern-matching cascade:
    /// 1. **Self-comparison** (`Ceq`/`Clt`/`Cgt` where `left == right`): immediate result.
    /// 2. **Equality analysis** (`Ceq`): delegates to [`analyze_equality`](Self::analyze_equality)
    ///    for XOR==0, SUB==0, MUL*0==0, AND&0==0, number-theoretic, constant, null, and nested patterns.
    /// 3. **Less-than / greater-than** (`Clt`/`Cgt`): delegates to range-based and constant analysis.
    /// 4. **Zero-producing ops** (`Xor`/`Sub` with `left==right`): returns `Unknown` since the
    ///    result only becomes meaningful when used in a comparison (handled at that level).
    /// 5. **Remainder / Multiplication / And**: delegates to specialized analyzers.
    ///
    /// Supports recursion up to [`MAX_PREDICATE_DEPTH`](Self::MAX_PREDICATE_DEPTH) for nested
    /// predicates (e.g., `ceq(ceq(x, x), 1)`).
    ///
    /// # Arguments
    ///
    /// * `op` - The SSA operation to analyze (typically a comparison or arithmetic op).
    /// * `cache` - Pre-built definition cache for efficient variable resolution.
    /// * `depth` - Current recursion depth (0 at the top level).
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] or [`AlwaysFalse`](PredicateResult::AlwaysFalse) if the
    /// predicate can be statically determined, [`Unknown`](PredicateResult::Unknown) otherwise.
    fn analyze_predicate_with_cache(
        op: &SsaOp<T>,
        cache: &DefinitionCache<T>,
        depth: usize,
    ) -> PredicateResult {
        if depth > Self::MAX_PREDICATE_DEPTH {
            return PredicateResult::Unknown;
        }

        match op {
            // Self-comparison patterns
            SsaOp::Ceq { left, right, .. } => {
                if left == right {
                    return PredicateResult::AlwaysTrue;
                }
                Self::analyze_equality(*left, *right, cache, depth)
            }

            SsaOp::Clt {
                left,
                right,
                unsigned,
                ..
            } => {
                if left == right {
                    return PredicateResult::AlwaysFalse;
                }
                Self::analyze_less_than(*left, *right, *unsigned, cache, depth)
            }

            SsaOp::Cgt {
                left,
                right,
                unsigned,
                ..
            } => {
                if left == right {
                    return PredicateResult::AlwaysFalse;
                }
                Self::analyze_greater_than(*left, *right, *unsigned, cache, depth)
            }

            // Operations that produce zero
            SsaOp::Xor { left, right, .. } if left == right => {
                // x ^ x = 0, handled when used in comparison
                PredicateResult::Unknown
            }

            SsaOp::Sub { left, right, .. } if left == right => {
                // x - x = 0, handled when used in comparison
                PredicateResult::Unknown
            }

            SsaOp::Rem { left, right, .. } => Self::analyze_remainder(*left, *right, cache, depth),

            SsaOp::Mul { left, right, .. } => {
                Self::analyze_multiplication(*left, *right, cache, depth)
            }

            SsaOp::And { left, right, .. } => Self::analyze_and(*left, *right, cache, depth),

            _ => PredicateResult::Unknown,
        }
    }

    /// Analyzes an equality comparison (`Ceq`) for opaque predicate patterns.
    ///
    /// Checks the following patterns (each with symmetric left/right variants):
    /// - `(x ^ x) == 0` -- XOR self-cancellation, always true.
    /// - `(x - x) == 0` -- subtraction self-cancellation, always true.
    /// - `(x * 0) == 0` or `(0 * x) == 0` -- zero-producing multiplication, always true.
    /// - `(x & 0) == 0` or `(0 & x) == 0` -- zero-producing AND, always true.
    /// - Number-theoretic: `(x*(x+1)) % 2 == 0` and factored forms, always true.
    /// - Constant equality: both sides are constants with known values.
    /// - Null checks: non-null variable (from `NewObj` etc.) compared to null, always false.
    /// - **Nested analysis fallback**: if the left operand is itself a predicate (comparison),
    ///   recursively analyzes it. If the result is known and compared to 1, returns that result;
    ///   if compared to 0, returns the negation.
    ///
    /// # Arguments
    ///
    /// * `left` - Left operand of the `Ceq`.
    /// * `right` - Right operand of the `Ceq`.
    /// * `cache` - Definition cache for resolving variable definitions.
    /// * `depth` - Current recursion depth for nested predicate analysis.
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] or [`AlwaysFalse`](PredicateResult::AlwaysFalse) if the
    /// equality can be statically determined, [`Unknown`](PredicateResult::Unknown) otherwise.
    fn analyze_equality(
        left: SsaVarId,
        right: SsaVarId,
        cache: &DefinitionCache<T>,
        depth: usize,
    ) -> PredicateResult {
        let left_def = cache.get_definition(left);
        let right_def = cache.get_definition(right);

        // Check for (x ^ x) == 0 pattern
        if let Some(SsaOp::Xor {
            left: xl,
            right: xr,
            ..
        }) = left_def
        {
            if xl == xr {
                if let Some(r) = right_def {
                    if Self::is_zero_constant(r) {
                        return PredicateResult::AlwaysTrue;
                    }
                }
            }
        }

        // Symmetric check
        if let Some(SsaOp::Xor {
            left: xl,
            right: xr,
            ..
        }) = right_def
        {
            if xl == xr {
                if let Some(l) = left_def {
                    if Self::is_zero_constant(l) {
                        return PredicateResult::AlwaysTrue;
                    }
                }
            }
        }

        // Check for (x - x) == 0 pattern
        if let Some(SsaOp::Sub {
            left: sl,
            right: sr,
            ..
        }) = left_def
        {
            if sl == sr {
                if let Some(r) = right_def {
                    if Self::is_zero_constant(r) {
                        return PredicateResult::AlwaysTrue;
                    }
                }
            }
        }

        // Symmetric check
        if let Some(SsaOp::Sub {
            left: sl,
            right: sr,
            ..
        }) = right_def
        {
            if sl == sr {
                if let Some(l) = left_def {
                    if Self::is_zero_constant(l) {
                        return PredicateResult::AlwaysTrue;
                    }
                }
            }
        }

        // Check for (x * 0) == 0 pattern
        if Self::is_zero_producing_mul(left_def, cache) {
            if let Some(r) = right_def {
                if Self::is_zero_constant(r) {
                    return PredicateResult::AlwaysTrue;
                }
            }
        }

        // Symmetric check
        if Self::is_zero_producing_mul(right_def, cache) {
            if let Some(l) = left_def {
                if Self::is_zero_constant(l) {
                    return PredicateResult::AlwaysTrue;
                }
            }
        }

        // Check for (x & 0) == 0 pattern
        if Self::is_zero_producing_and(left_def, cache) {
            if let Some(r) = right_def {
                if Self::is_zero_constant(r) {
                    return PredicateResult::AlwaysTrue;
                }
            }
        }

        // Symmetric check
        if Self::is_zero_producing_and(right_def, cache) {
            if let Some(l) = left_def {
                if Self::is_zero_constant(l) {
                    return PredicateResult::AlwaysTrue;
                }
            }
        }

        // Check for number-theoretic predicates that always evaluate to zero:
        //   (x * (x + 1)) % 2 == 0    — consecutive integer product is always even
        //   (x * x - x) % 2 == 0      — x²-x = x(x-1), consecutive product factored
        if Self::is_always_even_expression(left_def, cache) {
            if let Some(r) = right_def {
                if Self::is_zero_constant(r) {
                    return PredicateResult::AlwaysTrue;
                }
            }
        }

        // Check constant equality
        if let (Some(SsaOp::Const { value: lval, .. }), Some(SsaOp::Const { value: rval, .. })) =
            (left_def, right_def)
        {
            if let (Some(l), Some(r)) = (lval.as_i64(), rval.as_i64()) {
                return if l == r {
                    PredicateResult::AlwaysTrue
                } else {
                    PredicateResult::AlwaysFalse
                };
            }
        }

        // Check non-null equality with null
        if cache.is_non_null(left) {
            if let Some(r) = right_def {
                if Self::is_null_constant(r) {
                    return PredicateResult::AlwaysFalse;
                }
            }
        }

        if cache.is_non_null(right) {
            if let Some(l) = left_def {
                if Self::is_null_constant(l) {
                    return PredicateResult::AlwaysFalse;
                }
            }
        }

        // Nested analysis
        if let Some(left_op) = left_def {
            let left_result =
                Self::analyze_predicate_with_cache(left_op, cache, depth.saturating_add(1));
            if left_result != PredicateResult::Unknown {
                if let Some(r) = right_def {
                    if Self::is_one_constant(r) {
                        return left_result;
                    }
                    if Self::is_zero_constant(r) {
                        return left_result.negate();
                    }
                }
            }
        }

        PredicateResult::Unknown
    }

    /// Analyzes a less-than comparison (`Clt`) for opaque predicate patterns.
    ///
    /// Checks in order:
    /// 1. **Constant comparison**: both operands are constants, evaluate directly (signed or unsigned).
    /// 2. **Range-based**: if both operands have known [`ValueRange`]s, checks whether
    ///    `left.max < right.min` (always true) or `left.min >= right.max` (always false).
    /// 3. **Left range vs. constant right**: uses [`ValueRange::always_less_than`].
    /// 4. **Unsigned bounds**: `x <.un 0` is always false (no unsigned value is less than zero).
    /// 5. **Non-negative check**: if `left` is known non-negative (e.g., `ArrayLength`),
    ///    then `left < 0` is always false.
    ///
    /// # Arguments
    ///
    /// * `left` - Left operand of the comparison.
    /// * `right` - Right operand of the comparison.
    /// * `unsigned` - Whether this is an unsigned comparison (`clt.un`).
    /// * `cache` - Definition cache for resolving variable definitions and ranges.
    /// * `_depth` - Unused (less-than analysis is non-recursive).
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] or [`AlwaysFalse`](PredicateResult::AlwaysFalse) if the
    /// less-than comparison can be statically determined,
    /// [`Unknown`](PredicateResult::Unknown) otherwise.
    fn analyze_less_than(
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
        cache: &DefinitionCache<T>,
        _depth: usize,
    ) -> PredicateResult {
        let left_def = cache.get_definition(left);
        let right_def = cache.get_definition(right);

        // Constant comparison
        if let (Some(SsaOp::Const { value: lval, .. }), Some(SsaOp::Const { value: rval, .. })) =
            (left_def, right_def)
        {
            if unsigned {
                if let (Some(l), Some(r)) = (lval.as_u64(), rval.as_u64()) {
                    return if l < r {
                        PredicateResult::AlwaysTrue
                    } else {
                        PredicateResult::AlwaysFalse
                    };
                }
            } else if let (Some(l), Some(r)) = (lval.as_i64(), rval.as_i64()) {
                return if l < r {
                    PredicateResult::AlwaysTrue
                } else {
                    PredicateResult::AlwaysFalse
                };
            }
        }

        // Range-based analysis
        if let Some(left_range) = cache.get_range(left) {
            if let Some(right_range) = cache.get_range(right) {
                // left.max < right.min => always true
                if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min()) {
                    if l_max < r_min {
                        return PredicateResult::AlwaysTrue;
                    }
                }
                // left.min >= right.max => always false
                if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max()) {
                    if l_min >= r_max {
                        return PredicateResult::AlwaysFalse;
                    }
                }
            }

            // Check if left < constant
            if let Some(SsaOp::Const { value: rval, .. }) = right_def {
                if let Some(r) = rval.as_i64() {
                    if let Some(result) = left_range.always_less_than(r) {
                        return if result {
                            PredicateResult::AlwaysTrue
                        } else {
                            PredicateResult::AlwaysFalse
                        };
                    }
                }
            }
        }

        // Unsigned comparison: x < 0 is always false
        if unsigned {
            if let Some(SsaOp::Const { value: rval, .. }) = right_def {
                if rval.as_u64() == Some(0) {
                    return PredicateResult::AlwaysFalse;
                }
            }
        }

        // Non-negative < 0 is always false
        if let Some(left_range) = cache.get_range(left) {
            if left_range.is_always_non_negative() {
                if let Some(SsaOp::Const { value: rval, .. }) = right_def {
                    if rval.as_i64() == Some(0) {
                        return PredicateResult::AlwaysFalse;
                    }
                }
            }
        }

        PredicateResult::Unknown
    }

    /// Analyzes a greater-than comparison (`Cgt`) for opaque predicate patterns.
    ///
    /// Checks in order:
    /// 1. **Constant comparison**: both operands are constants, evaluate directly (signed or unsigned).
    /// 2. **Range-based**: if both operands have known [`ValueRange`]s, checks whether
    ///    `left.min > right.max` (always true) or `left.max <= right.min` (always false).
    /// 3. **Left range vs. constant right**: uses [`ValueRange::always_greater_than`].
    /// 4. **Unsigned bounds**: `0 >.un x` is always false (zero is never greater than any unsigned value).
    /// 5. **Non-negative vs. negative**: if `left` is known non-negative and `right` is a
    ///    negative constant, returns always true.
    ///
    /// # Arguments
    ///
    /// * `left` - Left operand of the comparison.
    /// * `right` - Right operand of the comparison.
    /// * `unsigned` - Whether this is an unsigned comparison (`cgt.un`).
    /// * `cache` - Definition cache for resolving variable definitions and ranges.
    /// * `_depth` - Unused (greater-than analysis is non-recursive).
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] or [`AlwaysFalse`](PredicateResult::AlwaysFalse) if the
    /// greater-than comparison can be statically determined,
    /// [`Unknown`](PredicateResult::Unknown) otherwise.
    fn analyze_greater_than(
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
        cache: &DefinitionCache<T>,
        _depth: usize,
    ) -> PredicateResult {
        let left_def = cache.get_definition(left);
        let right_def = cache.get_definition(right);

        // Constant comparison
        if let (Some(SsaOp::Const { value: lval, .. }), Some(SsaOp::Const { value: rval, .. })) =
            (left_def, right_def)
        {
            if unsigned {
                if let (Some(l), Some(r)) = (lval.as_u64(), rval.as_u64()) {
                    return if l > r {
                        PredicateResult::AlwaysTrue
                    } else {
                        PredicateResult::AlwaysFalse
                    };
                }
            } else if let (Some(l), Some(r)) = (lval.as_i64(), rval.as_i64()) {
                return if l > r {
                    PredicateResult::AlwaysTrue
                } else {
                    PredicateResult::AlwaysFalse
                };
            }
        }

        // Range-based analysis
        if let Some(left_range) = cache.get_range(left) {
            if let Some(right_range) = cache.get_range(right) {
                // left.min > right.max => always true
                if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max()) {
                    if l_min > r_max {
                        return PredicateResult::AlwaysTrue;
                    }
                }
                // left.max <= right.min => always false
                if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min()) {
                    if l_max <= r_min {
                        return PredicateResult::AlwaysFalse;
                    }
                }
            }

            // Check if left > constant
            if let Some(SsaOp::Const { value: rval, .. }) = right_def {
                if let Some(r) = rval.as_i64() {
                    if let Some(result) = left_range.always_greater_than(r) {
                        return if result {
                            PredicateResult::AlwaysTrue
                        } else {
                            PredicateResult::AlwaysFalse
                        };
                    }
                }
            }
        }

        // Unsigned: 0 > x is always false
        if unsigned {
            if let Some(SsaOp::Const { value: lval, .. }) = left_def {
                if lval.as_u64() == Some(0) {
                    return PredicateResult::AlwaysFalse;
                }
            }
        }

        // Non-negative value >= 0 is always true (x > -1 equivalent)
        if let Some(left_range) = cache.get_range(left) {
            if left_range.is_always_non_negative() {
                if let Some(SsaOp::Const { value: rval, .. }) = right_def {
                    if rval.as_i64().is_some_and(|r| r < 0) {
                        return PredicateResult::AlwaysTrue;
                    }
                }
            }
        }

        PredicateResult::Unknown
    }

    /// Analyzes a remainder operation (`Rem`) for `x % 1` which always produces zero.
    ///
    /// Returns `Unknown` rather than `AlwaysTrue`/`AlwaysFalse` because the zero result
    /// is only meaningful when subsequently compared (handled by the equality analysis).
    ///
    /// # Arguments
    ///
    /// * `_left` - Left operand of the remainder (unused; any value mod 1 is zero).
    /// * `right` - Right operand of the remainder (checked for constant 1).
    /// * `cache` - Definition cache for resolving variable definitions.
    /// * `_depth` - Unused (remainder analysis is non-recursive).
    ///
    /// # Returns
    ///
    /// Always returns [`PredicateResult::Unknown`] because the zero result is only
    /// meaningful when used in a subsequent comparison.
    fn analyze_remainder(
        _left: SsaVarId,
        right: SsaVarId,
        cache: &DefinitionCache<T>,
        _depth: usize,
    ) -> PredicateResult {
        // x % 1 == 0 is always true
        if let Some(SsaOp::Const { value: rval, .. }) = cache.get_definition(right) {
            if rval.as_i64() == Some(1) {
                // Result is always 0
                return PredicateResult::Unknown; // Handled when compared to 0
            }
        }
        PredicateResult::Unknown
    }

    /// Analyzes a multiplication (`Mul`) for zero-producing patterns (`x * 0` or `0 * x`).
    ///
    /// Returns `Unknown` because the zero result is only meaningful when subsequently
    /// compared (handled by the equality analysis at the comparison level).
    ///
    /// # Arguments
    ///
    /// * `left` - Left operand of the multiplication.
    /// * `right` - Right operand of the multiplication.
    /// * `cache` - Definition cache for resolving variable definitions.
    /// * `_depth` - Unused (multiplication analysis is non-recursive).
    ///
    /// # Returns
    ///
    /// Always returns [`PredicateResult::Unknown`] because the zero result is only
    /// meaningful when used in a subsequent comparison.
    fn analyze_multiplication(
        left: SsaVarId,
        right: SsaVarId,
        cache: &DefinitionCache<T>,
        _depth: usize,
    ) -> PredicateResult {
        // x * 0 = 0
        if let Some(SsaOp::Const { value: lval, .. }) = cache.get_definition(left) {
            if lval.is_zero() {
                return PredicateResult::Unknown; // Result is 0
            }
        }
        if let Some(SsaOp::Const { value: rval, .. }) = cache.get_definition(right) {
            if rval.is_zero() {
                return PredicateResult::Unknown; // Result is 0
            }
        }
        PredicateResult::Unknown
    }

    /// Analyzes a bitwise AND (`And`) for zero-producing patterns (`x & 0` or `0 & x`).
    ///
    /// Returns `Unknown` because the zero result is only meaningful when subsequently
    /// compared (handled by the equality analysis at the comparison level).
    ///
    /// # Arguments
    ///
    /// * `left` - Left operand of the AND.
    /// * `right` - Right operand of the AND.
    /// * `cache` - Definition cache for resolving variable definitions.
    /// * `_depth` - Unused (AND analysis is non-recursive).
    ///
    /// # Returns
    ///
    /// Always returns [`PredicateResult::Unknown`] because the zero result is only
    /// meaningful when used in a subsequent comparison.
    fn analyze_and(
        left: SsaVarId,
        right: SsaVarId,
        cache: &DefinitionCache<T>,
        _depth: usize,
    ) -> PredicateResult {
        // x & 0 = 0
        if let Some(SsaOp::Const { value: lval, .. }) = cache.get_definition(left) {
            if lval.is_zero() {
                return PredicateResult::Unknown;
            }
        }
        if let Some(SsaOp::Const { value: rval, .. }) = cache.get_definition(right) {
            if rval.is_zero() {
                return PredicateResult::Unknown;
            }
        }
        PredicateResult::Unknown
    }

    /// Checks if an operation produces a constant zero.
    fn is_zero_constant(op: &SsaOp<T>) -> bool {
        matches!(op, SsaOp::Const { value, .. } if value.is_zero())
    }

    /// Checks if an operation produces a constant one.
    fn is_one_constant(op: &SsaOp<T>) -> bool {
        matches!(op, SsaOp::Const { value, .. } if value.is_one())
    }

    /// Checks if an operation produces a null constant.
    fn is_null_constant(op: &SsaOp<T>) -> bool {
        matches!(op, SsaOp::Const { value, .. } if value.is_null())
    }

    /// Checks if an operation produces a constant -1.
    fn is_minus_one_constant(op: &SsaOp<T>) -> bool {
        matches!(op, SsaOp::Const { value, .. } if value.is_minus_one())
    }

    /// Returns `true` if the operation is a `Mul` where either operand is a constant zero.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to check, or `None` if the variable has no definition.
    /// * `cache` - Definition cache for resolving the multiplication operands.
    ///
    /// # Returns
    ///
    /// `true` if `op` is a `Mul` with at least one constant-zero operand, `false` otherwise.
    fn is_zero_producing_mul(op: Option<&SsaOp<T>>, cache: &DefinitionCache<T>) -> bool {
        if let Some(SsaOp::Mul { left, right, .. }) = op {
            if let Some(l) = cache.get_definition(*left) {
                if Self::is_zero_constant(l) {
                    return true;
                }
            }
            if let Some(r) = cache.get_definition(*right) {
                if Self::is_zero_constant(r) {
                    return true;
                }
            }
        }
        false
    }

    /// Returns `true` if the operation is an `And` where either operand is a constant zero.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to check, or `None` if the variable has no definition.
    /// * `cache` - Definition cache for resolving the AND operands.
    ///
    /// # Returns
    ///
    /// `true` if `op` is an `And` with at least one constant-zero operand, `false` otherwise.
    fn is_zero_producing_and(op: Option<&SsaOp<T>>, cache: &DefinitionCache<T>) -> bool {
        if let Some(SsaOp::And { left, right, .. }) = op {
            if let Some(l) = cache.get_definition(*left) {
                if Self::is_zero_constant(l) {
                    return true;
                }
            }
            if let Some(r) = cache.get_definition(*right) {
                if Self::is_zero_constant(r) {
                    return true;
                }
            }
        }
        false
    }

    /// Checks if an operation is an expression modulo 2 that always evaluates to 0.
    ///
    /// Detects number-theoretic opaque predicates based on the mathematical
    /// property that the product of two consecutive integers is always even:
    ///
    /// - `(x * (x + 1)) % 2` — direct consecutive product
    /// - `(x * (x - 1)) % 2` — reversed consecutive product
    /// - `(x * x - x) % 2` — factored form: x^2-x = x(x-1)
    /// - `(x * x + x) % 2` — factored form: x^2+x = x(x+1)
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to check, or `None` if the variable has no definition.
    /// * `cache` - Definition cache for resolving operand definitions.
    ///
    /// # Returns
    ///
    /// `true` if the expression is a `Rem` by 2 whose dividend is always even, `false` otherwise.
    fn is_always_even_expression(op: Option<&SsaOp<T>>, cache: &DefinitionCache<T>) -> bool {
        let Some(SsaOp::Rem {
            left: rem_left,
            right: rem_right,
            ..
        }) = op
        else {
            return false;
        };

        // Divisor must be 2
        let is_mod2 = cache
            .get_definition(*rem_right)
            .is_some_and(|d| matches!(d, SsaOp::Const { value, .. } if value.as_i64() == Some(2)));
        if !is_mod2 {
            return false;
        }

        let dividend_def = cache.get_definition(*rem_left);

        // Pattern 1: x * (x +/- 1) — consecutive product
        if let Some(SsaOp::Mul {
            left: mul_left,
            right: mul_right,
            ..
        }) = dividend_def
        {
            if Self::is_consecutive_pair(*mul_left, *mul_right, cache) {
                return true;
            }
        }

        // Pattern 2: (x * x) -/+ x — factored consecutive product
        // x^2-x = x(x-1), x^2+x = x(x+1), both always even
        if let Some(SsaOp::Sub {
            left: op_left,
            right: op_right,
            ..
        })
        | Some(SsaOp::Add {
            left: op_left,
            right: op_right,
            ..
        }) = dividend_def
        {
            if Self::is_self_square(*op_left, *op_right, cache)
                || Self::is_self_square(*op_right, *op_left, cache)
            {
                return true;
            }
        }

        false
    }

    /// Checks if `square_var` is defined as `other * other` (i.e., `other^2`).
    ///
    /// # Arguments
    ///
    /// * `square_var` - The variable suspected to be a square.
    /// * `other` - The variable that should appear as both operands of the multiplication.
    /// * `cache` - Definition cache for resolving the definition of `square_var`.
    ///
    /// # Returns
    ///
    /// `true` if `square_var` is defined as `Mul { left: other, right: other }`, `false` otherwise.
    fn is_self_square(square_var: SsaVarId, other: SsaVarId, cache: &DefinitionCache<T>) -> bool {
        matches!(
            cache.get_definition(square_var),
            Some(SsaOp::Mul { left, right, .. }) if *left == other && *right == other
        )
    }

    /// Checks if two variables form a consecutive integer pair (`n` and `n+1`).
    ///
    /// Performs three symmetric checks:
    /// - `b = a + 1` (either operand order of the `Add`).
    /// - `a = b + 1` (symmetric: `a` is the incremented one).
    /// - `b = a - (-1)` (subtraction of -1 is equivalent to adding 1).
    ///
    /// # Arguments
    ///
    /// * `a` - First variable of the potential consecutive pair.
    /// * `b` - Second variable of the potential consecutive pair.
    /// * `cache` - Definition cache for resolving variable definitions.
    ///
    /// # Returns
    ///
    /// `true` if one variable is defined as the other plus one, `false` otherwise.
    fn is_consecutive_pair(a: SsaVarId, b: SsaVarId, cache: &DefinitionCache<T>) -> bool {
        // Check if b = a + 1
        if let Some(SsaOp::Add {
            left: add_left,
            right: add_right,
            ..
        }) = cache.get_definition(b)
        {
            if *add_left == a {
                if let Some(SsaOp::Const { value: rval, .. }) = cache.get_definition(*add_right) {
                    if rval.as_i64() == Some(1) {
                        return true;
                    }
                }
            }
            if *add_right == a {
                if let Some(SsaOp::Const { value: lval, .. }) = cache.get_definition(*add_left) {
                    if lval.as_i64() == Some(1) {
                        return true;
                    }
                }
            }
        }

        // Check if a = b + 1 (symmetric)
        if let Some(SsaOp::Add {
            left: add_left,
            right: add_right,
            ..
        }) = cache.get_definition(a)
        {
            if *add_left == b {
                if let Some(SsaOp::Const { value: rval, .. }) = cache.get_definition(*add_right) {
                    if rval.as_i64() == Some(1) {
                        return true;
                    }
                }
            }
            if *add_right == b {
                if let Some(SsaOp::Const { value: lval, .. }) = cache.get_definition(*add_left) {
                    if lval.as_i64() == Some(1) {
                        return true;
                    }
                }
            }
        }

        // Check if b = a - (-1) which is also a + 1
        if let Some(SsaOp::Sub {
            left: sub_left,
            right: sub_right,
            ..
        }) = cache.get_definition(b)
        {
            if *sub_left == a {
                if let Some(SsaOp::Const { value: rval, .. }) = cache.get_definition(*sub_right) {
                    if rval.as_i64() == Some(-1) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Analyzes a branch condition variable to determine if it is an opaque predicate.
    ///
    /// Follows `Copy` chains iteratively with a [`BitSet`]-based cycle detector to handle
    /// SSA copies from phi nodes or obfuscated control flow. At each step:
    /// 1. If the variable has no definition in `DefUseIndex` but is phi-defined, checks
    ///    range info for a constant-zero equivalence (all-zero phi = always false).
    /// 2. Delegates to [`analyze_predicate_with_cache`](Self::analyze_predicate_with_cache) for
    ///    comparison operations.
    /// 3. For `Copy` ops, advances to the source variable and continues the loop.
    /// 4. For all other operations, falls back to [`analyze_branch_op`](Self::analyze_branch_op)
    ///    which checks direct truthiness of arithmetic and constant operations.
    ///
    /// # Arguments
    ///
    /// * `condition` - The SSA variable used as the branch condition.
    /// * `cache` - Definition cache for resolving the condition's definition chain.
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] if the branch always takes the true path,
    /// [`AlwaysFalse`](PredicateResult::AlwaysFalse) if it always takes the false path,
    /// [`Unknown`](PredicateResult::Unknown) if the condition cannot be statically resolved.
    fn analyze_branch(condition: SsaVarId, cache: &DefinitionCache<T>) -> PredicateResult {
        // Follow Copy chain iteratively with cycle detection to prevent infinite recursion.
        // This is needed because SSA can have Copy cycles (e.g., from phi nodes or
        // obfuscated control flow patterns).
        let mut current = condition;
        let mut visited = BitSet::new(cache.phi_defs.len());

        loop {
            // Cycle detection: if we've seen this variable before, bail out
            if !visited.insert(current.index()) {
                return PredicateResult::Unknown;
            }

            let Some(cond_op) = cache.get_definition(current) else {
                // Check if it's a phi node - analyze all operands
                if cache.is_phi_defined(current) {
                    // For phi nodes, we'd need to check if all operands lead to the same result
                    // This is complex, so we return Unknown for now unless we have range info
                    if let Some(range) = cache.get_range(current) {
                        if let Some(result) = range.always_equal_to(0) {
                            return if result {
                                PredicateResult::AlwaysFalse
                            } else {
                                PredicateResult::AlwaysTrue
                            };
                        }
                    }
                }
                return PredicateResult::Unknown;
            };

            // First, check if it's a direct comparison predicate
            let predicate_result = Self::analyze_predicate_with_cache(cond_op, cache, 0);
            if predicate_result != PredicateResult::Unknown {
                return predicate_result;
            }

            // Check if the condition is a Copy - trace through to the source iteratively
            if let SsaOp::Copy { src, .. } = cond_op {
                current = *src;
                continue;
            }

            // Not a Copy, break out and analyze the operation
            return Self::analyze_branch_op(cond_op, cache);
        }
    }

    /// Analyzes a non-Copy, non-comparison operation for direct truthiness in a branch.
    ///
    /// Called after the `Copy` chain has been resolved by [`analyze_branch`](Self::analyze_branch).
    /// Checks:
    /// - `x ^ x` = 0 (always false in `brtrue`).
    /// - `x - x` = 0 (always false).
    /// - `x & 0` or `x * 0` = 0 (always false).
    /// - `x | -1` = all-bits-set (always true).
    /// - `Const`: zero/null/false is always false; non-zero numeric or `true` is always true.
    ///
    /// # Arguments
    ///
    /// * `cond_op` - The resolved operation producing the branch condition value.
    /// * `cache` - Definition cache for resolving operands of the condition operation.
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] if the operation always produces a non-zero value,
    /// [`AlwaysFalse`](PredicateResult::AlwaysFalse) if it always produces zero,
    /// [`Unknown`](PredicateResult::Unknown) if the truthiness cannot be determined.
    fn analyze_branch_op(cond_op: &SsaOp<T>, cache: &DefinitionCache<T>) -> PredicateResult {
        // Check operations that produce known zero values
        match cond_op {
            // x ^ x = 0, so brtrue on this result never jumps
            SsaOp::Xor { left, right, .. } if left == right => PredicateResult::AlwaysFalse,

            // x - x = 0, so brtrue on this result never jumps
            SsaOp::Sub { left, right, .. } if left == right => PredicateResult::AlwaysFalse,

            // x & 0 = 0, x * 0 = 0
            SsaOp::And { left, right, .. } | SsaOp::Mul { left, right, .. } => {
                let is_left_zero = cache
                    .get_definition(*left)
                    .is_some_and(Self::is_zero_constant);
                let is_right_zero = cache
                    .get_definition(*right)
                    .is_some_and(Self::is_zero_constant);

                if is_left_zero || is_right_zero {
                    PredicateResult::AlwaysFalse
                } else {
                    PredicateResult::Unknown
                }
            }

            // x | -1 = -1 (all bits set), so brtrue always jumps
            SsaOp::Or { left, right, .. } => {
                let is_left_minus_one = cache
                    .get_definition(*left)
                    .is_some_and(Self::is_minus_one_constant);
                let is_right_minus_one = cache
                    .get_definition(*right)
                    .is_some_and(Self::is_minus_one_constant);

                if is_left_minus_one || is_right_minus_one {
                    PredicateResult::AlwaysTrue
                } else {
                    PredicateResult::Unknown
                }
            }

            // Constant values: 0/null/false is always false, non-zero is always true
            SsaOp::Const { value, .. } => {
                if value.is_zero() || value.is_null() {
                    PredicateResult::AlwaysFalse
                } else if value.as_i64().is_some() || value.as_bool().is_some() {
                    // Non-zero numeric or true boolean
                    PredicateResult::AlwaysTrue
                } else {
                    PredicateResult::Unknown
                }
            }

            // All other operations have unknown truthiness
            // Note: ArrayLength is always >= 0, but we can't prove non-empty
            _ => PredicateResult::Unknown,
        }
    }

    /// Analyzes a comparison operation for algebraic simplification opportunities.
    ///
    /// This checks for patterns like:
    /// - `(x - y) == 0` → `x == y`
    /// - `(x - y) < 0` → `x < y`
    /// - `(x - y) > 0` → `x > y`
    /// - `(x ^ y) == 0` → `x == y`
    /// - `(cmp) == 1` → `cmp`
    ///
    /// # Arguments
    ///
    /// * `op` - The SSA comparison operation to analyze (`Ceq`, `Clt`, or `Cgt`).
    /// * `cache` - Definition cache for resolving operand definitions.
    ///
    /// # Returns
    ///
    /// `Some(ComparisonSimplification)` if the comparison can be algebraically simplified,
    /// `None` if no simplification applies or the operation is not a comparison.
    fn analyze_comparison_simplification(
        op: &SsaOp<T>,
        cache: &DefinitionCache<T>,
    ) -> Option<ComparisonSimplification<T>> {
        match op {
            SsaOp::Ceq { dest, left, right } => {
                Self::analyze_ceq_simplification(*dest, *left, *right, cache)
            }
            SsaOp::Clt {
                dest,
                left,
                right,
                unsigned,
            } => Self::analyze_clt_simplification(*dest, *left, *right, *unsigned, cache),
            SsaOp::Cgt {
                dest,
                left,
                right,
                unsigned,
            } => Self::analyze_cgt_simplification(*dest, *left, *right, *unsigned, cache),
            _ => None,
        }
    }

    /// Checks if a variable is defined as a constant zero.
    fn is_zero_var(var: SsaVarId, cache: &DefinitionCache<T>) -> bool {
        cache
            .get_definition(var)
            .is_some_and(Self::is_zero_constant)
    }

    /// Checks if a variable is defined as a constant with value 1.
    fn is_one_var(var: SsaVarId, cache: &DefinitionCache<T>) -> bool {
        cache.get_definition(var).is_some_and(Self::is_one_constant)
    }

    /// Analyzes a `Ceq` operation for algebraic simplification.
    ///
    /// Detects three patterns:
    /// - `(x - y) == 0` simplifies to `x == y` (subtraction-zero, skip self-subtraction).
    /// - `(x ^ y) == 0` simplifies to `x == y` (XOR-zero, skip self-XOR).
    /// - `(cmp) == 1` simplifies to `Copy(cmp)` when the other operand is a `Ceq`/`Clt`/`Cgt`
    ///   result, since CIL comparisons already produce 0 or 1.
    ///
    /// # Arguments
    ///
    /// * `dest` - Destination variable of the `Ceq` (preserved in the simplified op).
    /// * `left` - Left operand of the `Ceq`.
    /// * `right` - Right operand of the `Ceq`.
    /// * `cache` - Definition cache for resolving operand definitions.
    ///
    /// # Returns
    ///
    /// `Some(ComparisonSimplification)` if a simplification pattern matches, `None` otherwise.
    fn analyze_ceq_simplification(
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        cache: &DefinitionCache<T>,
    ) -> Option<ComparisonSimplification<T>> {
        // Check if comparing to zero
        let (other_var, is_comparing_to_zero) = if Self::is_zero_var(right, cache) {
            (left, true)
        } else if Self::is_zero_var(left, cache) {
            (right, true)
        } else {
            (left, false)
        };

        if is_comparing_to_zero {
            if let Some(def_op) = cache.get_definition(other_var) {
                // Pattern: (x - y) == 0 → x == y
                if let SsaOp::Sub {
                    left: sub_left,
                    right: sub_right,
                    ..
                } = def_op
                {
                    // Skip self-subtraction - that's handled by PredicateResult (always true)
                    if sub_left != sub_right {
                        return Some(ComparisonSimplification::SimplerOp {
                            new_op: SsaOp::Ceq {
                                dest,
                                left: *sub_left,
                                right: *sub_right,
                            },
                            reason: "(x - y) == 0 simplified to x == y",
                        });
                    }
                }

                // Pattern: (x ^ y) == 0 → x == y
                if let SsaOp::Xor {
                    left: xor_left,
                    right: xor_right,
                    ..
                } = def_op
                {
                    // Skip self-XOR - that's handled by PredicateResult (always true)
                    if xor_left != xor_right {
                        return Some(ComparisonSimplification::SimplerOp {
                            new_op: SsaOp::Ceq {
                                dest,
                                left: *xor_left,
                                right: *xor_right,
                            },
                            reason: "(x ^ y) == 0 simplified to x == y",
                        });
                    }
                }
            }
        }

        // Check if comparing to one (true in CIL)
        let (other_var, is_comparing_to_one) = if Self::is_one_var(right, cache) {
            (left, true)
        } else if Self::is_one_var(left, cache) {
            (right, true)
        } else {
            (left, false)
        };

        if is_comparing_to_one {
            if let Some(def_op) = cache.get_definition(other_var) {
                // Pattern: (cmp) == 1 → copy cmp
                if matches!(
                    def_op,
                    SsaOp::Ceq { .. } | SsaOp::Clt { .. } | SsaOp::Cgt { .. }
                ) {
                    return Some(ComparisonSimplification::Copy {
                        dest,
                        src: other_var,
                        reason: "(cmp) == 1 simplified to cmp",
                    });
                }
            }
        }

        None
    }

    /// Analyzes a `Clt` operation for algebraic simplification.
    ///
    /// Detects `(x - y) < 0` and simplifies to `x < y` (signed only; unsigned subtraction
    /// has different overflow semantics). Self-subtraction is skipped since it is handled
    /// as a constant predicate (`AlwaysFalse`).
    ///
    /// # Arguments
    ///
    /// * `dest` - Destination variable of the `Clt` (preserved in the simplified op).
    /// * `left` - Left operand of the `Clt`.
    /// * `right` - Right operand of the `Clt`.
    /// * `unsigned` - Whether this is an unsigned comparison; if `true`, no simplification is attempted.
    /// * `cache` - Definition cache for resolving operand definitions.
    ///
    /// # Returns
    ///
    /// `Some(ComparisonSimplification::SimplerOp)` if `(x - y) < 0` is detected, `None` otherwise.
    fn analyze_clt_simplification(
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
        cache: &DefinitionCache<T>,
    ) -> Option<ComparisonSimplification<T>> {
        // Only handle signed comparisons for subtraction patterns
        // (unsigned subtraction has different overflow semantics)
        if unsigned {
            return None;
        }

        // Pattern: (x - y) < 0 → x < y
        if Self::is_zero_var(right, cache) {
            if let Some(SsaOp::Sub {
                left: sub_left,
                right: sub_right,
                ..
            }) = cache.get_definition(left)
            {
                // Skip self-subtraction - that's handled by PredicateResult (always false)
                if sub_left != sub_right {
                    return Some(ComparisonSimplification::SimplerOp {
                        new_op: SsaOp::Clt {
                            dest,
                            left: *sub_left,
                            right: *sub_right,
                            unsigned,
                        },
                        reason: "(x - y) < 0 simplified to x < y",
                    });
                }
            }
        }

        None
    }

    /// Analyzes a `Cgt` operation for algebraic simplification.
    ///
    /// Detects `(x - y) > 0` and simplifies to `x > y` (signed only; unsigned subtraction
    /// has different overflow semantics). Self-subtraction is skipped since it is handled
    /// as a constant predicate (`AlwaysFalse`).
    ///
    /// # Arguments
    ///
    /// * `dest` - Destination variable of the `Cgt` (preserved in the simplified op).
    /// * `left` - Left operand of the `Cgt`.
    /// * `right` - Right operand of the `Cgt`.
    /// * `unsigned` - Whether this is an unsigned comparison; if `true`, no simplification is attempted.
    /// * `cache` - Definition cache for resolving operand definitions.
    ///
    /// # Returns
    ///
    /// `Some(ComparisonSimplification::SimplerOp)` if `(x - y) > 0` is detected, `None` otherwise.
    fn analyze_cgt_simplification(
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
        cache: &DefinitionCache<T>,
    ) -> Option<ComparisonSimplification<T>> {
        // Only handle signed comparisons for subtraction patterns
        if unsigned {
            return None;
        }

        // Pattern: (x - y) > 0 → x > y
        if Self::is_zero_var(right, cache) {
            if let Some(SsaOp::Sub {
                left: sub_left,
                right: sub_right,
                ..
            }) = cache.get_definition(left)
            {
                // Skip self-subtraction - that's handled by PredicateResult (always false)
                if sub_left != sub_right {
                    return Some(ComparisonSimplification::SimplerOp {
                        new_op: SsaOp::Cgt {
                            dest,
                            left: *sub_left,
                            right: *sub_right,
                            unsigned,
                        },
                        reason: "(x - y) > 0 simplified to x > y",
                    });
                }
            }
        }

        None
    }

    /// Detects phi nodes where every operand resolves to the same constant value.
    ///
    /// Iterates over all phi nodes in all blocks. For each phi, looks up each operand's
    /// defining operation: if all are `Const` with identical values, records the mapping
    /// from the phi result variable to that constant. These entries are later used to
    /// replace the phi with a `Const` instruction and to resolve branch conditions that
    /// depend on phi-defined variables.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function whose phi nodes are analyzed.
    ///
    /// # Returns
    ///
    /// A map from phi result variable to the constant value that all operands agree on.
    /// Empty if no phi nodes have all-constant, all-identical operands.
    fn analyze_phi_constants(ssa: &SsaFunction<T>) -> BTreeMap<SsaVarId, ConstValue<T>> {
        let mut phi_constants = BTreeMap::new();

        // Index defining ops once (O(1) lookups) rather than calling
        // `get_definition` per phi operand, whose slow path is a full-function
        // scan when def-sites are stale.
        let index = DefUseIndex::build_with_ops(ssa);

        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                let operands: Vec<_> = phi.operands().iter().collect();
                let Some(first_operand) = operands.first() else {
                    continue;
                };

                // Check if all operands come from the same constant
                let first_val = first_operand.value();
                let mut all_same_const = true;
                let mut const_value = None;

                for operand in &operands {
                    let var = operand.value();
                    // Look up the defining operation via the index.
                    if let Some(op) = index.def_op(var) {
                        if let SsaOp::Const { value, .. } = op {
                            if const_value.is_none() {
                                const_value = Some(value.clone());
                            } else if const_value.as_ref() != Some(value) {
                                all_same_const = false;
                                break;
                            }
                        } else {
                            all_same_const = false;
                            break;
                        }
                    } else if var != first_val {
                        all_same_const = false;
                        break;
                    }
                }

                if all_same_const {
                    if let Some(value) = const_value {
                        phi_constants.insert(phi.result(), value);
                    }
                }
            }
        }

        phi_constants
    }
}

/// Run the opaque-predicate pass on `ssa`.
///
/// Returns `true` only if a structural change occurred (branches
/// simplified, comparisons resolved). Phi-only constant folding still
/// happens but reports `false` so the host scheduler does not trigger
/// an immediate SSA rebuild that would re-create the phi.
///
/// # Arguments
///
/// * `ssa` — The SSA function to analyze and simplify in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::OpaquePredicateRemoved`],
///   [`EventKind::BranchSimplified`], and [`EventKind::ConstantFolded`].
/// * `ptr_size` — Pointer width of the target runtime (affects
///   evaluator behavior for native-size constants).
///
/// # Returns
///
/// `true` if any structural change was made (branch simplified or
/// comparison resolved). Returns `false` if only phi-constant folding
/// occurred.
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
    OpaquePredicatePass::<T>::new().run(ssa, method, events, ptr_size)
}

impl<T: Target> OpaquePredicatePass<T> {
    /// Per-method entry point. See [`run`] for the free-function shape that
    /// host wrappers usually call.
    pub fn run<L>(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        events: &L,
        ptr_size: PointerSize,
    ) -> bool
    where
        L: EventListener<T> + ?Sized,
    {
        // Per-pass internal event collector. Flushed to `events` at the end
        // so the caller still gets every record in invocation order.
        let changes: crate::events::EventLog<T> = crate::events::EventLog::new();
        let method_token = method;

        // Build definition cache for efficient lookup
        let cache = DefinitionCache::build(ssa);

        // Analyze phi nodes for constant values
        let phi_constants = Self::analyze_phi_constants(ssa);

        // Collect branches to simplify
        let mut branch_simplifications: Vec<(usize, usize, bool)> = Vec::new();

        // Collect comparison replacements (opaque predicates that become constant true/false)
        let mut comparison_replacements: Vec<(usize, usize, SsaVarId, bool)> = Vec::new();

        // Collect comparison simplifications (algebraic simplifications like (x-y)==0 → x==y)
        let mut comparison_simplifications: Vec<(usize, usize, ComparisonSimplification<T>)> =
            Vec::new();

        // Collect phi replacements
        let mut phi_replacements: Vec<(usize, usize, SsaVarId, ConstValue<T>)> = Vec::new();

        // Symbolic evaluator for branch conditions, built lazily and evaluated
        // over every block exactly once. Previously each unresolved branch built
        // a fresh evaluator and re-evaluated blocks `0..=block_idx`, which was
        // O(branches * blocks); a single forward pass is sufficient because SSA
        // values are single-assignment (evaluating later blocks never changes an
        // earlier branch condition's value).
        let mut branch_evaluator = None;

        // Analyze each block
        for (block_idx, block) in ssa.iter_blocks() {
            // Analyze branch terminators
            if let Some(SsaOp::Branch {
                condition,
                true_target,
                false_target,
            }) = block.terminator_op()
            {
                // Check phi constants first
                if let Some(const_val) = phi_constants.get(condition) {
                    let is_true = const_val.as_bool().unwrap_or(false)
                        || const_val.as_i64().is_some_and(|v| v != 0);
                    if is_true {
                        branch_simplifications.push((block_idx, *true_target, true));
                    } else {
                        branch_simplifications.push((block_idx, *false_target, false));
                    }
                    // Can't use continue with iter_blocks in a for loop, collect the data
                } else {
                    let mut result = Self::analyze_branch(*condition, &cache);

                    // If pattern matching couldn't determine the result, try the
                    // shared symbolic evaluator (built and fully evaluated once).
                    if result == PredicateResult::Unknown {
                        let evaluator = branch_evaluator.get_or_insert_with(|| {
                            let mut e = SsaEvaluator::new(ssa, ptr_size);
                            for idx in 0..ssa.block_count() {
                                e.evaluate_block(idx);
                            }
                            e
                        });
                        result = match evaluator.get(*condition) {
                            Some(expr) if expr.is_constant() => {
                                if expr.as_constant().is_some_and(ConstValue::is_zero) {
                                    PredicateResult::AlwaysFalse
                                } else {
                                    PredicateResult::AlwaysTrue
                                }
                            }
                            Some(_) | None => PredicateResult::Unknown,
                        };
                    }

                    match result {
                        PredicateResult::AlwaysTrue => {
                            branch_simplifications.push((block_idx, *true_target, true));
                        }
                        PredicateResult::AlwaysFalse => {
                            branch_simplifications.push((block_idx, *false_target, false));
                        }
                        PredicateResult::Unknown => {}
                    }
                }
            }

            // Analyze comparison instructions
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let op = instr.op();
                // First check for opaque predicates (constant true/false)
                let result = Self::analyze_predicate_with_cache(op, &cache, 0);
                if let Some(value) = result.as_bool() {
                    if let Some(dest) = op.dest() {
                        comparison_replacements.push((block_idx, instr_idx, dest, value));
                        continue; // Don't also check for simplification
                    }
                }

                // Then check for algebraic simplifications
                if let Some(simplification) = Self::analyze_comparison_simplification(op, &cache) {
                    comparison_simplifications.push((block_idx, instr_idx, simplification));
                }
            }

            // Check for phi nodes that can be replaced with constants
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                if let Some(const_val) = phi_constants.get(&phi.result()) {
                    phi_replacements.push((block_idx, phi_idx, phi.result(), const_val.clone()));
                }
            }
        }

        // CFG-changing branch simplifications invalidate instruction and phi indices
        // collected from the old graph. Apply them alone; the scheduler's next
        // pass iteration will re-analyze comparisons and phis on the rebuilt SSA.
        if !branch_simplifications.is_empty() {
            comparison_replacements.clear();
            comparison_simplifications.clear();
            phi_replacements.clear();
        }

        // Track structural changes (branches, comparisons) vs phi-only changes.
        // Phi constant replacement doesn't warrant rebuild_ssa (see comment below).
        let has_structural = !branch_simplifications.is_empty()
            || !comparison_replacements.is_empty()
            || !comparison_simplifications.is_empty();

        let edit_result = ssa.edit(
            SsaEditOptions::new()
                .with_verify(true)
                .with_rollback(SsaRollbackPolicy::OnFailure),
            |editor| {
                // Apply branch simplifications
                for (block_idx, target, is_true) in branch_simplifications {
                    let Some(last_idx) = editor
                        .function()
                        .block(block_idx)
                        .and_then(|block| block.instructions().len().checked_sub(1))
                    else {
                        continue;
                    };
                    editor.replace_instruction_op(block_idx, last_idx, SsaOp::Jump { target })?;
                    editor.mark_cfg_changed();
                    changes
                        .record(EventKind::OpaquePredicateRemoved)
                        .at(method_token.clone(), block_idx)
                        .message(format!(
                            "removed opaque predicate (always {})",
                            if is_true { "true" } else { "false" }
                        ));
                    changes
                        .record(EventKind::BranchSimplified)
                        .at(method_token.clone(), block_idx)
                        .message(format!("simplified to unconditional branch to {target}"));
                }

                // Apply comparison replacements (opaque predicates → constant true/false)
                for (block_idx, instr_idx, dest, value) in comparison_replacements {
                    if editor
                        .function()
                        .block(block_idx)
                        .and_then(|block| block.instruction(instr_idx))
                        .is_none()
                    {
                        continue;
                    }
                    let const_value = if value {
                        ConstValue::True
                    } else {
                        ConstValue::False
                    };
                    editor.replace_instruction_op(
                        block_idx,
                        instr_idx,
                        SsaOp::Const {
                            dest,
                            value: const_value,
                        },
                    )?;
                    changes
                        .record(EventKind::ConstantFolded)
                        .at(method_token.clone(), instr_idx)
                        .message(format!("opaque predicate → {value}"));
                }

                // Apply comparison simplifications (algebraic transformations)
                for (block_idx, instr_idx, simplification) in comparison_simplifications {
                    if editor
                        .function()
                        .block(block_idx)
                        .and_then(|block| block.instruction(instr_idx))
                        .is_none()
                    {
                        continue;
                    }
                    match simplification {
                        ComparisonSimplification::SimplerOp { new_op, reason } => {
                            editor.replace_instruction_op(block_idx, instr_idx, new_op)?;
                            changes
                                .record(EventKind::ConstantFolded)
                                .at(method_token.clone(), instr_idx)
                                .message(reason);
                        }
                        ComparisonSimplification::Copy { dest, src, reason } => {
                            editor.replace_instruction_op(
                                block_idx,
                                instr_idx,
                                SsaOp::Copy { dest, src },
                            )?;
                            changes
                                .record(EventKind::ConstantFolded)
                                .at(method_token.clone(), instr_idx)
                                .message(reason);
                        }
                    }
                }

                // Apply phi replacements: PHIs where all operands are the same constant.
                let mut phi_removals: Vec<(usize, usize)> = Vec::new();
                for (block_idx, phi_idx, phi_result, const_value) in phi_replacements {
                    let const_instr = SsaInstruction::synthetic(SsaOp::Const {
                        dest: phi_result,
                        value: const_value.clone(),
                    });
                    editor.insert_instruction(block_idx, 0, const_instr)?;
                    phi_removals.push((block_idx, phi_idx));

                    changes
                        .record(EventKind::ConstantFolded)
                        .at(method_token.clone(), block_idx)
                        .message(format!("phi with constant operands → {const_value:?}"));
                }

                phi_removals.sort_by(|a, b| b.cmp(a));
                for (block_idx, phi_idx) in phi_removals {
                    if editor
                        .function()
                        .block(block_idx)
                        .and_then(|block| block.phi_nodes().get(phi_idx))
                        .is_some()
                    {
                        editor.remove_phi(block_idx, phi_idx)?;
                    }
                }

                Ok(())
            },
        );

        if edit_result.is_err() {
            return false;
        }

        if !changes.is_empty() {
            for ev in &changes {
                events.push(ev.clone());
            }
        }

        // Report only structural changes so the host scheduler doesn't
        // immediately trigger a rebuild that would re-create the folded
        // phi from the still-live original variable definitions. Once DCE
        // sweeps those defs and a structural change does fire rebuild, the
        // phi won't come back.
        has_structural
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies predicate tri-state conversions and negation.
    #[test]
    fn predicate_result_negate_and_as_bool() {
        assert_eq!(PredicateResult::AlwaysTrue.as_bool(), Some(true));
        assert_eq!(PredicateResult::AlwaysFalse.as_bool(), Some(false));
        assert_eq!(PredicateResult::Unknown.as_bool(), None);

        assert_eq!(
            PredicateResult::AlwaysTrue.negate(),
            PredicateResult::AlwaysFalse
        );
        assert_eq!(
            PredicateResult::AlwaysFalse.negate(),
            PredicateResult::AlwaysTrue
        );
        assert_eq!(PredicateResult::Unknown.negate(), PredicateResult::Unknown);
    }

    use crate::{
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        pointer::PointerSize,
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

    fn run_predicates(
        ssa: &mut SsaFunction<MockTarget>,
        label: &str,
        method: &u32,
        log: &EventLog<MockTarget>,
    ) -> bool {
        run_mock_pass_boundary(ssa, label, |ssa| run(ssa, method, log, PointerSize::Bit64))
    }

    fn build_branch_with_opaque_predicate() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 4);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);

        // Need B1 and B2 as branch targets
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: v1,
            left: v0,
            right: v0,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v1,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        ssa.recompute_uses();
        ssa
    }

    #[test]
    fn opaque_predicate_self_equality_detected() {
        let mut ssa = build_branch_with_opaque_predicate();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 3u32;
        let changed = run_predicates(&mut ssa, "self-equality predicate", &method, &log);
        // The Ceq(v0, v0) should be detected as always true
        assert!(changed, "self-equality opaque predicate should be detected");
        assert!(log.has(EventKind::OpaquePredicateRemoved) || log.has(EventKind::BranchSimplified));
    }

    #[test]
    fn xor_self_always_zero_in_predicate() {
        let mut ssa = SsaFunction::new(0, 4);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v_xor = local_at(&mut ssa, 1, 0, 1);
        let v_zero = local_at(&mut ssa, 2, 0, 2);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::Xor {
            dest: v_xor,
            left: v0,
            right: v0,
            flags: None,
        }));
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: v_zero,
            left: v_xor,
            right: v0,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_zero,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "xor-self predicate", &method, &log);
        assert!(changed, "xor-self pattern should be detected");
    }

    #[test]
    fn sub_self_always_zero_in_predicate() {
        let mut ssa = SsaFunction::new(0, 4);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v_sub = local_at(&mut ssa, 1, 0, 1);
        let v_ceq = local_at(&mut ssa, 2, 0, 2);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Sub {
            dest: v_sub,
            left: v0,
            right: v0,
            flags: None,
        }));
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: v_ceq,
            left: v_sub,
            right: v0,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_ceq,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "sub-self predicate", &method, &log);
        assert!(changed, "sub-self comparison should be detected");
    }

    #[test]
    fn constant_comparison_true() {
        let mut ssa = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);
        let v_ceq = local_at(&mut ssa, 2, 0, 2);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(5),
        }));
        // 5 == 5 → always true
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: v_ceq,
            left: v0,
            right: v1,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_ceq,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "constant-true predicate", &method, &log);
        assert!(changed, "constant equality should be detected");
    }

    #[test]
    fn constant_comparison_false() {
        let mut ssa = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);
        let v_ceq = local_at(&mut ssa, 2, 0, 2);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(3),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(7),
        }));
        // 3 == 7 → always false
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: v_ceq,
            left: v0,
            right: v1,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_ceq,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "constant-false predicate", &method, &log);
        assert!(changed, "constant inequality should be detected");
    }

    #[test]
    fn clt_same_var_always_false() {
        let mut ssa = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v_clt = local_at(&mut ssa, 1, 0, 1);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Clt {
            dest: v_clt,
            left: v0,
            right: v0,
            unsigned: false,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_clt,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "same-var less-than predicate", &method, &log);
        assert!(changed, "x < x should be always false");
    }

    #[test]
    fn unsigned_less_than_zero_always_false() {
        let mut ssa = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let zero = local_at(&mut ssa, 1, 0, 1);
        let v_clt = local_at(&mut ssa, 2, 0, 2);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::U32(10),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::U32(0),
        }));
        // unsigned x < 0 is always false
        b0.add_instruction(instr(SsaOp::Clt {
            dest: v_clt,
            left: v0,
            right: zero,
            unsigned: true,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_clt,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "unsigned-less-than-zero predicate", &method, &log);
        assert!(changed, "unsigned x < 0 should be always false");
    }

    #[test]
    fn analyze_phi_constants_all_same() {
        let mut ssa = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 1, 0);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(42),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(v0, 0));
        phi.add_operand(PhiOperand::new(v1, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Return {
            value: Some(phi_var),
        }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let constants = OpaquePredicatePass::<MockTarget>::analyze_phi_constants(&ssa);
        // Both operands are constant I32(42) — phi should be resolved
        assert!(
            constants.contains_key(&phi_var),
            "phi with same constants should be resolved"
        );
    }

    #[test]
    fn branch_always_true_opaque_predicate() {
        let mut ssa = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        // Branch on constant 1 — always true
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v0,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "constant-branch predicate", &method, &log);
        // The branch on constant 1 should be simplified
        assert!(changed, "constant branch should be simplified");
    }

    #[test]
    fn empty_function_noop() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "empty predicates", &method, &log);
        assert!(!changed);
    }
}
