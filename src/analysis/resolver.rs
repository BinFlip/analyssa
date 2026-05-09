//! Unified constant resolution for SSA variables.
//!
//! The [`ValueResolver`] composes [`ConstEvaluator`], [`PhiAnalyzer`], and optionally
//! [`SsaEvaluator`] into a single reusable entry point for demand-driven constant
//! resolution.
//!
//! # Three-Tier Fallback Strategy
//!
//! 1. **ConstEvaluator** — Handles all instruction-defined ops (arithmetic, bitwise,
//!    comparisons, conversions) with caching and cycle detection. This is the primary
//!    tier and handles most cases.
//!
//! 2. **PhiAnalyzer** — Checks whether all PHI operands resolve to the same constant
//!    value. This catches cases where control flow merges values from different paths
//!    that happen to be the same constant.
//!
//! 3. **SsaEvaluator** (path-aware fallback, optional) — Uses `resolve_with_trace`
//!    for instruction-defined variables that `ConstEvaluator` couldn't fold
//!    (e.g., XOR where one operand comes from a Call instruction). This is disabled
//!    by default and must be enabled via `with_path_aware_fallback()`.
//!
//! # Complexity
//!
//! Resolution is O(d) where d is the definition chain depth (bounded by
//! ConstEvaluator's max_depth). Results are cached, so repeated resolution of
//! the same variable is O(1).
//!
//! # Example
//!
//! ```rust,ignore
//! use analyssa::{analysis::ValueResolver, ir::SsaFunction, MockTarget, PointerSize};
//!
//! let ssa: SsaFunction<MockTarget> = /* ... */;
//!
//! let mut resolver = ValueResolver::new(&ssa, PointerSize::Bit64)
//!     .with_path_aware_fallback();
//! if let Some(value) = resolver.resolve(some_var) {
//!     println!("Resolved to: {:?}", value);
//! }
//! ```

use crate::{
    analysis::{consts::ConstEvaluator, evaluator::SsaEvaluator, phis::PhiAnalyzer},
    ir::{function::SsaFunction, value::ConstValue, variable::SsaVarId},
    target::Target,
    PointerSize,
};

/// Demand-driven constant resolver composing multiple analysis components.
///
/// Provides a unified API for resolving SSA variables to constant values,
/// combining the strengths of [`ConstEvaluator`] (instruction folding),
/// [`PhiAnalyzer`] (uniform PHI detection), and optionally [`SsaEvaluator`]
/// (path-aware tracing for variables that pure constant folding can't handle).
pub struct ValueResolver<'a, T: Target> {
    /// Reference to the SSA function being analyzed.
    ssa: &'a SsaFunction<T>,
    /// Primary constant evaluator for instruction-defined operations.
    evaluator: ConstEvaluator<'a, T>,
    /// PHI analyzer for detecting uniform constants across merge points.
    phi: PhiAnalyzer<'a, T>,
    /// Whether PHI uniform constant detection is enabled (default: true).
    resolve_phis: bool,
    /// Whether path-aware fallback via SsaEvaluator is enabled (default: false).
    path_aware_fallback: bool,
    /// Target pointer size for native int/uint masking in constant evaluation.
    ptr_size: PointerSize,
}

impl<'a, T: Target> ValueResolver<'a, T> {
    /// Creates a new resolver with PHI resolution enabled and path-aware fallback disabled.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>, ptr_size: PointerSize) -> Self {
        Self {
            ssa,
            evaluator: ConstEvaluator::new(ssa, ptr_size),
            phi: PhiAnalyzer::new(ssa),
            resolve_phis: true,
            path_aware_fallback: false,
            ptr_size,
        }
    }

    /// Enables the path-aware fallback via [`SsaEvaluator`].
    ///
    /// When enabled, variables that `ConstEvaluator` can't fold (e.g., XOR where
    /// one operand comes from a Call) will be attempted via `resolve_with_trace`.
    #[must_use]
    pub fn with_path_aware_fallback(mut self) -> Self {
        self.path_aware_fallback = true;
        self
    }

    /// Injects a single known value into the resolver.
    pub fn set_known(&mut self, var: SsaVarId, value: ConstValue<T>) {
        self.evaluator.set_known(var, value);
    }

    /// Resolves a variable to a constant using a three-tier fallback strategy.
    ///
    /// 1. Try [`ConstEvaluator`] (all instruction-defined ops).
    /// 2. If PHI-defined, check for uniform constant via [`PhiAnalyzer`].
    /// 3. If path-aware fallback is enabled, try [`SsaEvaluator::resolve_with_trace`].
    pub fn resolve(&mut self, var: SsaVarId) -> Option<ConstValue<T>> {
        // 1. Try ConstEvaluator (handles Const, arithmetic, bitwise, etc. with caching)
        if let Some(val) = self.evaluator.evaluate_var(var) {
            return Some(val);
        }

        // 2. PHI uniform constant check
        if self.resolve_phis {
            if let Some((_, phi)) = self.ssa.find_phi_defining(var) {
                let result = self.phi.uniform_constant(phi, &mut self.evaluator);
                if result.is_some() {
                    return result;
                }
            }
        }

        // 3. Path-aware fallback via SsaEvaluator (for instruction-defined vars
        //    that ConstEvaluator couldn't fold, e.g. XOR with a Call operand)
        if self.path_aware_fallback && self.ssa.get_definition(var).is_some() {
            let mut eval = SsaEvaluator::new(self.ssa, self.ptr_size);
            if let Some(resolved) = eval.resolve_with_trace(var, 15) {
                if let Some(c) = resolved.as_constant() {
                    self.evaluator.set_known(var, c.clone());
                    return Some(c.clone());
                }
            }
        }

        None
    }

    /// Resolves all variables to constants. Returns `None` if any variable can't be resolved.
    pub fn resolve_all(&mut self, vars: &[SsaVarId]) -> Option<Vec<ConstValue<T>>> {
        let mut result = Vec::with_capacity(vars.len());
        for &var in vars {
            result.push(self.resolve(var)?);
        }
        Some(result)
    }
}
