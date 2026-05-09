//! `World<T>` — a minimal interprocedural view of the program-under-analysis.
//!
//! Hosts implement `World<T>` to give analyssa-side global passes a uniform
//! handle on the set of methods that exist, who calls whom, and which methods
//! have already been pruned.
//!
//! # Design
//!
//! The trait is intentionally tiny — only what `DeadMethodEliminationPass`
//! actually needs. Methods that surface a list of `T::MethodRef` return
//! `Vec<T::MethodRef>` for now (rather than an iterator), because:
//!
//! - analyssa wants `T::MethodRef` to stay `Clone + Eq + Hash`, not require
//!   `Copy`. Returning `&[T::MethodRef]` works for static call graphs but not
//!   for hosts that compute SSA-derived call graphs on demand.
//! - `Vec<T::MethodRef>` lets the host build the slice from any backing
//!   storage (DashMap-based ones, computed-on-demand graphs) without
//!   committing to a borrow lifetime.
//! - The cost is a small allocation per call. Global passes are O(methods)
//!   anyway, so this is dominated by their own work.
//!
//! `mark_dead` takes `&self` so hosts can use interior-mutable dead-method
//! sets; this mirrors the shape that lets `EventLog::record(&self)` work for
//! parallel passes.

use crate::target::Target;

/// Minimal interprocedural view used by analyssa global passes.
pub trait World<T: Target> {
    /// All methods present in the program-under-analysis.
    fn all_methods(&self) -> Vec<T::MethodRef>;

    /// Methods that are externally reachable (program entry points,
    /// exported APIs, etc.). The set of definitely-live roots that the DCE
    /// reachability walk seeds from.
    fn entry_points(&self) -> Vec<T::MethodRef>;

    /// Methods directly called by `method`. Hosts are free to combine
    /// SSA-derived and static-call-graph information here; the result is
    /// the union of both as far as `World<T>` is concerned.
    fn callees(&self, method: &T::MethodRef) -> Vec<T::MethodRef>;

    /// `true` if `method` has already been marked dead by a prior pass.
    fn is_dead(&self, method: &T::MethodRef) -> bool;

    /// Mark `method` as dead. Implementations are expected to be
    /// interior-mutable (`&self`, not `&mut self`) so global passes can
    /// share the world by reference.
    fn mark_dead(&self, method: &T::MethodRef);

    /// Methods in reverse topological order of the call graph (callees
    /// before callers). Used by the analyssa pass scheduler to process
    /// callees before callers so interprocedural results propagate
    /// upward in the same iteration.
    ///
    /// Default implementation falls back to [`all_methods`](Self::all_methods)
    /// (no ordering guarantee). Hosts with a real call graph should
    /// override to return a topological ordering; recursion-induced
    /// cycles can be broken arbitrarily.
    fn methods_reverse_topological(&self) -> Vec<T::MethodRef> {
        self.all_methods()
    }
}
