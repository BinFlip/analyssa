//! [`SsaPass`] trait and supporting types ([`ModificationScope`],
//! [`SsaPassHost`]) for declaring and implementing SSA transformation
//! passes.
//!
//! A pass is a target-agnostic transformation that runs on a single
//! method's SSA (or globally across all methods) and reports whether
//! it made changes. The scheduler in [`super::scheduler`] orchestrates
//! execution: layer assignment from capability dependencies, per-layer
//! fixpoint, parallel per-method dispatch via rayon, and
//! modification-scope-driven repair after each pass.

use crate::{
    error::Result,
    events::EventLog,
    host::{DirtySet, SsaStore},
    ir::function::SsaFunction,
    pointer::PointerSize,
    target::Target,
    world::World,
};

/// Combined host surface that the scheduler exposes to passes.
///
/// Scheduler hosts implement [`World<T>`], [`SsaStore<T>`], and
/// [`DirtySet<T>`] separately, plus this trait to surface the event
/// sink and pointer size. The scheduler hands passes a
/// `&dyn SsaPassHost<T>` so they can read the call graph, look up peer
/// methods' SSA, mark methods dirty, and record events without knowing
/// the concrete host type.
pub trait SsaPassHost<T: Target>: World<T> + SsaStore<T> + DirtySet<T> + Sync {
    /// Returns the event sink for transformation events recorded by passes.
    fn events(&self) -> &EventLog<T>;

    /// Returns the pointer width of the target's runtime.
    ///
    /// Used by passes that need to know the host's address size (e.g.,
    /// predicate evaluation, reassociation, jump threading). Hosts that
    /// do not care can accept the default of [`PointerSize::Bit64`].
    fn ptr_size(&self) -> PointerSize {
        PointerSize::Bit64
    }
}

/// Describes the extent of modifications a pass makes to the SSA function.
///
/// The scheduler uses this to select the minimum repair necessary after a
/// pass runs, avoiding expensive full SSA reconstruction when it is not
/// needed. Passes should declare the **tightest** scope that covers all
/// their modifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModificationScope {
    /// The pass only replaces uses of variables with other existing
    /// variables.
    ///
    /// SSA invariants are preserved automatically — no repair is needed
    /// after the pass. Example: GVN forwarding redundant uses to earlier
    /// defs.
    UsesOnly,

    /// The pass replaces or removes instructions but does not change
    /// the CFG.
    ///
    /// After this scope, the scheduler runs a lightweight repair to strip
    /// `Nop`s, recompute variable metadata, and eliminate trivial phis.
    /// No dominator or dominance-frontier recomputation is needed.
    /// Examples: constant propagation, copy propagation, dead-code
    /// elimination, algebraic simplification, strength reduction.
    InstructionsOnly,

    /// The pass may add or remove blocks, change predecessors or
    /// successors, or otherwise modify the control-flow graph.
    ///
    /// After this scope, the scheduler runs full SSA reconstruction
    /// (recompute dominators, place phis, rename variables). Examples:
    /// control flow unflattening, jump threading, block merging, loop
    /// canonicalization, inlining.
    CfgModifying,
}

/// An SSA transformation pass.
///
/// Generic over both the target `T` and the host adapter `H`. The host
/// type must implement [`SsaPassHost<T>`] and is fixed by the
/// [`PassScheduler`](crate::scheduling::PassScheduler) instance — all
/// passes registered with that scheduler see the same host type.
///
/// # Concurrency
///
/// Passes must be `Send + Sync` so the scheduler can run them in
/// parallel across methods via rayon. Mutation of pass state happens
/// only in [`initialize`](Self::initialize) /
/// [`finalize`](Self::finalize) (single-threaded boundary calls);
/// per-method work uses `&self` and must rely on interior mutability
/// for any cross-method state.
///
/// # Implementing a pass
///
/// Most analyssa-side passes have a pure-function body in `crate::passes`
/// and a one-line trait impl in the scheduling sub-module:
///
/// ```ignore
/// impl<T: Target, H: SsaPassHost<T>> SsaPass<T, H> for MyPass {
///     fn name(&self) -> &'static str { "my-pass" }
///     fn run_on_method(&self, ssa: &mut SsaFunction<T>, method: &T::MethodRef, host: &H) -> Result<bool> {
///         Ok(passes::my_pass::run(ssa, method, host.events()))
///     }
/// }
/// ```
///
/// Hosts that ship target-specific passes (e.g., CIL inlining) write
/// impls bounded on a host extension trait, giving the impl access to
/// host-specific methods while still being storable in a
/// `Box<dyn SsaPass<CilTarget, ConcreteCilHost>>`.
pub trait SsaPass<T: Target, H: SsaPassHost<T>>: Send + Sync {
    /// Returns a unique short name for logging and debugging.
    fn name(&self) -> &'static str;

    /// Returns a human-readable description of the pass.
    fn description(&self) -> &'static str {
        "No description available"
    }

    /// Determines whether this pass should run on a specific method.
    ///
    /// Called before [`run_on_method`](Self::run_on_method). Override to
    /// skip methods that do not need this pass (e.g., already processed,
    /// too small to be interesting). Default implementation returns `true`.
    fn should_run(&self, _method: &T::MethodRef, _host: &H) -> bool {
        true
    }

    /// Run the pass on a single method's SSA.
    ///
    /// Events are recorded directly to `host.events()`.
    ///
    /// # Arguments
    ///
    /// * `ssa` — The SSA function to transform in place.
    /// * `method` — Opaque reference identifying the method being processed.
    /// * `host` — The host adapter for reading the call graph, looking up
    ///   peer SSA, marking methods dirty, and recording events.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if any changes were made, `Ok(false)` if no changes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the pass fails to process the method (e.g.,
    /// malformed SSA, unexpected opcode).
    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool>;

    /// Run the pass on the entire program (interprocedural passes).
    ///
    /// Override for passes that need to see all methods simultaneously,
    /// such as dead-method elimination or whole-program constant
    /// propagation.
    ///
    /// # Arguments
    ///
    /// * `host` — The host adapter providing program-wide state.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if any changes were made, `Ok(false)` if no changes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the pass fails.
    fn run_global(&self, _host: &H) -> Result<bool> {
        Ok(false)
    }

    /// Returns whether this pass operates globally (across all methods).
    ///
    /// When `true`, the scheduler invokes [`run_global`](Self::run_global)
    /// instead of iterating per-method with
    /// [`run_on_method`](Self::run_on_method).
    fn is_global(&self) -> bool {
        false
    }

    /// Called once before the pass runs in a phase.
    ///
    /// Use this to initialize pass-specific state or caches. The default
    /// implementation is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    fn initialize(&mut self, _host: &H) -> Result<()> {
        Ok(())
    }

    /// Called once after the pass completes in a phase.
    ///
    /// Use this to clean up pass-specific state. The default
    /// implementation is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if finalization fails.
    fn finalize(&mut self, _host: &H) -> Result<()> {
        Ok(())
    }

    /// Declares the extent of modifications this pass makes.
    ///
    /// Default is [`ModificationScope::CfgModifying`] (conservative).
    /// Override for passes that do not modify the CFG so the scheduler
    /// can apply a lighter-weight repair.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Returns whether the pass performs its own SSA boundary repair.
    ///
    /// Passes that wrap their mutation body in
    /// [`SsaFunction::edit`](crate::ir::function::SsaFunction::edit) should
    /// return `true` so the scheduler does not run a second repair step on
    /// the same method. The pass must still report an accurate
    /// [`modification_scope`](Self::modification_scope) for ordering,
    /// documentation, and hosts that inspect pass metadata.
    fn repairs_ssa(&self) -> bool {
        false
    }

    /// Returns the capabilities this pass provides after execution.
    ///
    /// The scheduler ensures that consumers of a capability are placed in
    /// later layers than providers.
    fn provides(&self) -> &[T::Capability] {
        &[]
    }

    /// Returns the capabilities this pass requires before it can run.
    ///
    /// Unsatisfied requirements (no provider registered) fall back to
    /// phase-based ordering rather than blocking the pass.
    fn requires(&self) -> &[T::Capability] {
        &[]
    }

    /// Returns whether this pass reads other methods' SSA during
    /// [`run_on_method`](Self::run_on_method).
    ///
    /// During parallel execution, the scheduler must keep each method's
    /// SSA visible in the store so other threads can read it. When this
    /// returns `true`, the scheduler clones the SSA before processing
    /// (instead of removing it) so concurrent visibility is preserved.
    /// Passes that only modify their own method should keep the default
    /// `false` to avoid the clone overhead.
    fn reads_peer_ssa(&self) -> bool {
        false
    }

    /// Returns whether this pass requires a full scan of all methods
    /// every iteration.
    ///
    /// If `true`, the scheduler dispatches to every method with SSA,
    /// regardless of dirty-tracking state. If `false` (default), only
    /// dirty methods are processed.
    fn requires_full_scan(&self) -> bool {
        false
    }

    /// Returns the fallback execution layer for this pass.
    ///
    /// Used when capability dependencies do not constrain the pass's
    /// position. Hosts assign meaningful numbers (e.g., CIL uses
    /// Structure=0, Value=1, Simplify=2, Inline=3 by convention).
    ///
    /// Normalize-style passes (DCE, GVN, constant propagation) typically
    /// use layer 0 and are registered via the scheduler's `add_normalize`
    /// API, which takes them out of the layered execution and runs them
    /// between every layer's fixpoint iterations.
    fn fallback_layer(&self) -> usize {
        0
    }
}
