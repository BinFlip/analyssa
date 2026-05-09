//! Concrete [`SsaPass`] trait implementations for
//! every built-in analyssa pass.
//!
//! Each pass has a target-agnostic body in its own submodule
//! (e.g. [`super::algebraic::run`]); this module provides ready-to-register
//! pass structs that wrap those bodies in the [`SsaPass`] trait.
//! Hosts that want richer behavior (e.g. CIL copy propagation with
//! type-aware post-processing) can keep their own custom pass struct
//! that calls into the pass body directly.
//!
//! # Default Layer Conventions
//!
//! The fallback layer numbers follow the deobfuscation-pipeline convention:
//!
//! - **Layer 0**: Structural passes (control flow simplification, jump
//!   threading, loop canonicalization, block merging).
//! - **Layer 1**: Value-level passes (algebraic simplification, strength
//!   reduction, reassociation, value range propagation, opaque predicates,
//!   LICM).
//!
//! Normalize passes (DCE, GVN, copy propagation) are registered via
//! [`PassScheduler::add_normalize`](crate::scheduling::PassScheduler::add_normalize)
//! and run between every layer's fixpoint iterations (their layer number
//! is irrelevant).
//!
//! # Pass Inventory
//!
//! | Struct | Layer | Body module | Scope |
//! |--------|-------|-------------|-------|
//! | [`AlgebraicSimplificationPass`] | 1 | [`super::algebraic`] | InstructionsOnly |
//! | [`BlockMergingPass`] | 0 | [`super::blockmerge`] | CfgModifying |
//! | [`ControlFlowSimplificationPass`] | 0 | [`super::controlflow`] | CfgModifying |
//! | [`CopyPropagationPass`] | normalize | [`super::copying`] | InstructionsOnly |
//! | [`DeadCodeEliminationPass`] | normalize | [`super::deadcode`] | InstructionsOnly |
//! | [`DeadMethodEliminationPass`] | global | [`super::deadcode`] | global |
//! | [`GlobalValueNumberingPass`] | normalize | [`super::gvn`] | UsesOnly |
//! | [`JumpThreadingPass`] | 0 | [`super::threading`] | CfgModifying |
//! | [`LicmPass`] | 1 | [`super::licm`] | CfgModifying |
//! | [`LoopCanonicalizationPass`] | 0 | [`super::loopcanon`] | CfgModifying |
//! | [`OpaquePredicatePass<T>`] | 1 | [`super::predicates`] | CfgModifying |
//! | [`ReassociationPass`] | 1 | [`super::reassociate`] | InstructionsOnly |
//! | [`StrengthReductionPass`] | 1 | [`super::strength`] | InstructionsOnly |
//! | [`ValueRangePropagationPass`] | 1 | [`super::ranges`] | InstructionsOnly |

use crate::{
    error::Result,
    ir::{function::SsaFunction, variable::SsaVarId},
    passes,
    scheduling::{ModificationScope, SsaPass, SsaPassHost},
    target::Target,
};

/// Default cap on inner-loop iterations for passes with fixpoint loops.
///
/// Applied by default to [`BlockMergingPass`], [`ControlFlowSimplificationPass`],
/// [`CopyPropagationPass`], [`DeadCodeEliminationPass`], and
/// [`ValueRangePropagationPass`].
const DEFAULT_MAX_ITERATIONS: usize = 10;

/// Pass that simplifies algebraic identities.
///
/// Rewrites operations like `x xor x → 0`, `x + 0 → x`, `x * 1 → x`,
/// `ceq(x, x) → 1`, etc. The actual transformation is delegated to
/// [`crate::passes::algebraic::run`].
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] — only replaces instructions,
/// never changes the CFG.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlgebraicSimplificationPass;

impl AlgebraicSimplificationPass {
    /// Creates a new algebraic simplification pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T, H> SsaPass<T, H> for AlgebraicSimplificationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "algebraic-simplification"
    }

    fn description(&self) -> &'static str {
        "Simplify algebraic identities (x xor x = 0, x + 0 = x, etc.)"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::algebraic::run(ssa, method, host.events()))
    }
}

/// Pass that eliminates trampoline blocks and coalesces single-edge pairs.
///
/// Delegates to [`crate::passes::blockmerge::run`]. The iteration cap
/// controls how many fixpoint iterations the inner loop runs.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] — can change block predecessors,
/// successors, and remove blocks.
#[derive(Debug, Clone, Copy)]
pub struct BlockMergingPass {
    /// Maximum number of fixpoint iterations for the inner merge loop.
    pub max_iterations: usize,
}

impl Default for BlockMergingPass {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

impl BlockMergingPass {
    /// Creates a new block-merging pass with the given iteration cap.
    #[must_use]
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

impl<T, H> SsaPass<T, H> for BlockMergingPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "block-merging"
    }

    fn description(&self) -> &'static str {
        "Eliminate trampoline (single-jump) blocks"
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::blockmerge::run(
            ssa,
            method,
            host.events(),
            self.max_iterations,
        ))
    }
}

/// Pass that simplifies conditional branches and control flow.
///
/// Delegates to [`crate::passes::controlflow::run`]. Performs jump
/// threading through trampolines, branch-to-same-target simplification,
/// and dead tail removal.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] — can change branch targets and
/// remove instructions.
#[derive(Debug, Clone, Copy)]
pub struct ControlFlowSimplificationPass {
    /// Cap on iterations of the inner CFG-simplify loop.
    pub max_iterations: usize,
}

impl Default for ControlFlowSimplificationPass {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

impl ControlFlowSimplificationPass {
    /// Creates a new control-flow simplification pass with the given
    /// iteration cap.
    #[must_use]
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

impl<T, H> SsaPass<T, H> for ControlFlowSimplificationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "control-flow-simplification"
    }

    fn description(&self) -> &'static str {
        "Simplify branches, eliminate unreachable code, fold constant predicates"
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::controlflow::run(
            ssa,
            method,
            host.events(),
            self.max_iterations,
        ))
    }
}

/// Pass that eliminates redundant copy operations and trivial phi nodes.
///
/// Delegates to [`crate::passes::copying::run`]. Hosts that need a
/// post-step (e.g. CIL local-type propagation) should implement their
/// own pass that calls [`crate::passes::copying::run_with_hook`] directly.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] — replaces instructions,
/// never changes the CFG.
#[derive(Debug, Clone, Copy)]
pub struct CopyPropagationPass {
    /// Cap on iterations of the inner copy-prop loop.
    pub max_iterations: usize,
}

impl Default for CopyPropagationPass {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

impl CopyPropagationPass {
    /// Creates a new copy-propagation pass with the given iteration cap.
    #[must_use]
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

impl<T, H> SsaPass<T, H> for CopyPropagationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "copy-propagation"
    }

    fn description(&self) -> &'static str {
        "Eliminate redundant copy operations and trivial phi nodes"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::copying::run(
            ssa,
            method,
            host.events(),
            self.max_iterations,
        ))
    }
}

/// Pass that removes unreachable blocks, unused definitions, and dead phi
/// nodes within a single method.
///
/// Delegates to [`crate::passes::deadcode::run`]. Complements
/// [`DeadMethodEliminationPass`] which handles interprocedural DCE.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] — removes instructions and
/// phis but does not add new blocks.
#[derive(Debug, Clone, Copy)]
pub struct DeadCodeEliminationPass {
    /// Cap on iterations of the inner DCE loop.
    pub max_iterations: usize,
}

impl Default for DeadCodeEliminationPass {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

impl DeadCodeEliminationPass {
    /// Creates a new dead-code-elimination pass with the given iteration
    /// cap.
    #[must_use]
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

impl<T, H> SsaPass<T, H> for DeadCodeEliminationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "dead-code-elimination"
    }

    fn description(&self) -> &'static str {
        "Remove unreachable blocks, unused definitions, and op-less instructions"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::deadcode::run(
            ssa,
            method,
            host.events(),
            self.max_iterations,
        ))
    }
}

/// Global interprocedural dead-method elimination pass.
///
/// Delegates to [`crate::passes::deadcode::run_global`]. Operates over
/// the whole program via [`World<T>`](crate::world::World), not per-method.
///
/// This pass is declared as `is_global()` and never invoked per-method.
/// It marks all methods not transitively reachable from entry points as
/// dead using [`World::mark_dead`](crate::world::World::mark_dead).
#[derive(Debug, Default, Clone, Copy)]
pub struct DeadMethodEliminationPass;

impl<T, H> SsaPass<T, H> for DeadMethodEliminationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "dead-method-elimination"
    }

    fn description(&self) -> &'static str {
        "Mark methods that have no live callers and aren't entry points"
    }

    fn is_global(&self) -> bool {
        true
    }

    fn run_on_method(
        &self,
        _ssa: &mut SsaFunction<T>,
        _method: &T::MethodRef,
        _host: &H,
    ) -> Result<bool> {
        // Global pass — never invoked per-method.
        Ok(false)
    }

    fn run_global(&self, host: &H) -> Result<bool> {
        Ok(passes::deadcode::run_global::<T, _, _>(host, host.events()))
    }
}

/// Pass that eliminates redundant computations via hash-consed value
/// numbering.
///
/// Delegates to [`crate::passes::gvn::run`]. Detects when the same
/// expression is computed multiple times and replaces later uses with
/// the earlier result.
///
/// # Modification scope
///
/// [`ModificationScope::UsesOnly`] — only replaces uses of variables,
/// never modifies the CFG or adds instructions.
#[derive(Debug, Default, Clone, Copy)]
pub struct GlobalValueNumberingPass;

impl GlobalValueNumberingPass {
    /// Creates a new global value numbering pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T, H> SsaPass<T, H> for GlobalValueNumberingPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "global-value-numbering"
    }

    fn description(&self) -> &'static str {
        "Eliminate redundant computations via value numbering"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::UsesOnly
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::gvn::run(ssa, method, host.events()))
    }
}

/// Pass that hoists loop-invariant computations out of loop bodies.
///
/// Delegates to [`crate::passes::licm::run`]. Processes loops
/// innermost-first, conservatively guarding against hoisting that would
/// corrupt phi nodes or create unstable trampoline blocks.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] — adds instructions to the
/// preheader and may modify phi predecessor references.
#[derive(Debug, Default, Clone, Copy)]
pub struct LicmPass;

impl LicmPass {
    /// Creates a new LICM pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T, H> SsaPass<T, H> for LicmPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "licm"
    }

    fn description(&self) -> &'static str {
        "Hoist loop-invariant computations out of loops"
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::licm::run(ssa, method, host.events()))
    }
}

/// Pass that ensures every loop has a single preheader and a single latch.
///
/// Delegates to [`crate::passes::loopcanon::run`]. Creates new preheader
/// and/or unified latch blocks when loops have multiple non-loop
/// predecessors or multiple back-edges.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] — can add new blocks and
/// rewrite phi operands.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoopCanonicalizationPass;

impl LoopCanonicalizationPass {
    /// Creates a new loop-canonicalization pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T, H> SsaPass<T, H> for LoopCanonicalizationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "loop-canonicalization"
    }

    fn description(&self) -> &'static str {
        "Ensure each loop has a single preheader and latch"
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::loopcanon::run(ssa, method, host.events()))
    }
}

/// Re-export the predicates pass struct (defined in
/// [`crate::passes::predicates`]) and add an [`SsaPass`] impl. The
/// struct is `OpaquePredicatePass<T>`; analyssa's blanket below makes it
/// fit any host that implements [`SsaPassHost<T>`].
pub use crate::passes::predicates::OpaquePredicatePass;

impl<T, H> SsaPass<T, H> for OpaquePredicatePass<T>
where
    T: Target + Send + Sync,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "opaque-predicate"
    }

    fn description(&self) -> &'static str {
        "Remove always-true/false conditions, simplify trivial comparisons"
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::predicates::run(
            ssa,
            method,
            host.events(),
            host.ptr_size(),
        ))
    }
}

/// Pass that propagates value-range information across the SSA graph.
///
/// Delegates to [`crate::passes::ranges::run`]. Uses sparse conditional
/// range propagation to simplify branches and replace comparisons whose
/// result is provable from operand ranges.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] — replaces instructions,
/// never changes the CFG.
#[derive(Debug, Clone, Copy)]
pub struct ValueRangePropagationPass {
    /// Cap on iterations of the inner range-prop loop.
    pub max_iterations: usize,
}

impl Default for ValueRangePropagationPass {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

impl ValueRangePropagationPass {
    /// Creates a new value-range propagation pass with the given
    /// iteration cap.
    #[must_use]
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

impl<T, H> SsaPass<T, H> for ValueRangePropagationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "value-range-propagation"
    }

    fn description(&self) -> &'static str {
        "Propagate value-range information across the SSA graph"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::ranges::run(
            ssa,
            method,
            host.events(),
            self.max_iterations,
        ))
    }
}

/// Pass that reorders nested associative operations to enable constant
/// folding.
///
/// Delegates to [`crate::passes::reassociate::run`]. Rewrites
/// `(x op c1) op c2` into `x op (c1 combined c2)` so the constants
/// collapse. Uses [`SsaPassHost::ptr_size`] for correct wraparound on
/// native-size integers.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] — replaces instructions,
/// never changes the CFG.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReassociationPass;

impl ReassociationPass {
    /// Creates a new reassociation pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T, H> SsaPass<T, H> for ReassociationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "reassociation"
    }

    fn description(&self) -> &'static str {
        "Reorder associative operations to expose simplification opportunities"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::reassociate::run(
            ssa,
            method,
            host.events(),
            host.ptr_size(),
        ))
    }
}

/// Pass that replaces expensive operations with cheaper equivalents.
///
/// Delegates to [`crate::passes::strength::run`]. Transforms:
/// - `x * 2^n` → `x << n`
/// - `x / 2^n` (unsigned) → `x >> n`
/// - `x % 2^n` (unsigned) → `x & (2^n - 1)`
///
/// Signed variants are only applied when `is_non_negative` is provable.
/// The blanket impl uses a conservative `false` predicate; hosts with
/// richer range analysis should implement their own pass with a better
/// predicate.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] — replaces instructions,
/// never changes the CFG.
#[derive(Debug, Default, Clone, Copy)]
pub struct StrengthReductionPass;

impl<T, H> SsaPass<T, H> for StrengthReductionPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "strength-reduction"
    }

    fn description(&self) -> &'static str {
        "Replace expensive operations with cheaper equivalents"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        let is_non_negative = |_: SsaVarId| false;
        Ok(passes::strength::run(
            ssa,
            method,
            host.events(),
            &is_non_negative,
        ))
    }
}

/// Pass that threads jumps through predictable branch conditions.
///
/// Delegates to [`crate::passes::threading::run`]. Evaluates branch
/// conditions from the incoming path and redirects predecessors directly
/// to the proven target. Uses [`SsaPassHost::ptr_size`] for the
/// evaluator.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] — changes terminator operations
/// and predecessor relationships.
#[derive(Debug, Default, Clone, Copy)]
pub struct JumpThreadingPass;

impl JumpThreadingPass {
    /// Creates a new jump-threading pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T, H> SsaPass<T, H> for JumpThreadingPass
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    fn name(&self) -> &'static str {
        "jump-threading"
    }

    fn description(&self) -> &'static str {
        "Thread jumps through empty or predictable blocks"
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        Ok(passes::threading::run(
            ssa,
            method,
            host.events(),
            host.ptr_size(),
        ))
    }
}
