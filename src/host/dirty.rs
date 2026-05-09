//! [`DirtySet<T>`] — host-supplied per-method scheduling state for fixpoint
//! iteration tracking.
//!
//! Two concepts live here:
//!
//! - **Dirty**: methods that need re-processing in the next fixpoint
//!   iteration. Passes mark methods dirty when they discover new work
//!   (e.g., inlining marks the caller dirty after splicing in a callee;
//!   devirtualization marks both sides dirty after retargeting a call).
//! - **Processed**: methods that any pass has modified at least once
//!   during the pipeline run. Hosts use this to drive downstream work
//!   (e.g., CIL codegen regenerates only processed methods).
//!
//! # Concurrency
//!
//! All methods take `&self` so passes can update state in parallel
//! without acquiring `&mut`. Implementations are typically `DashSet`-backed.
//!
//! # Bypassing dirty-tracking
//!
//! Passes that declare [`SsaPass::requires_full_scan`](crate::scheduling::SsaPass::requires_full_scan)
//! bypass dirty tracking — the scheduler runs them on every method every
//! iteration regardless of dirty state.

use crate::target::Target;

/// Per-method scheduling state that the pass scheduler reads and writes.
///
/// Tracks two mutually exclusive sets:
/// - **Dirty set**: methods needing re-processing on the next fixpoint
///   iteration.
/// - **Processed set**: methods that have been modified at least once
///   during the current pipeline run.
///
/// Implementations are interior-mutable (typically `DashSet`-backed) so
/// passes can update state in parallel.
pub trait DirtySet<T: Target> {
    /// Add `method` to the dirty set.
    ///
    /// Called by passes that discover a method's SSA is stale. Idempotent:
    /// marking an already-dirty method has no effect.
    fn mark_dirty(&self, method: &T::MethodRef);

    /// Returns `true` if `method` is currently in the dirty set.
    fn is_dirty(&self, method: &T::MethodRef) -> bool;

    /// Returns a snapshot of all currently-dirty methods.
    ///
    /// Used by the scheduler to determine which methods to process in an
    /// iteration.
    fn dirty_snapshot(&self) -> Vec<T::MethodRef>;

    /// Remove `method` from the dirty set.
    ///
    /// Called by the scheduler after a pass has processed `method` or
    /// when the method is no longer dirty.
    fn clear_dirty_for(&self, method: &T::MethodRef);

    /// Record that `method` has been modified by at least one pass.
    ///
    /// Idempotent: repeated calls for the same method have no effect.
    /// Hosts use the processed set to drive downstream work (e.g.,
    /// regenerating code only for modified methods).
    fn mark_processed(&self, method: &T::MethodRef);

    /// Returns `true` if `method` has been modified by any pass during
    /// the current pipeline run.
    fn is_processed(&self, method: &T::MethodRef) -> bool;
}
