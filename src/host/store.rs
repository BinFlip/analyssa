//! [`SsaStore<T>`] — host-supplied storage for per-method SSA functions,
//! used by the pass scheduler to hand SSA to passes and reclaim it.

use crate::{ir::function::SsaFunction, target::Target};

/// Per-method SSA storage that the pass scheduler reads and writes.
///
/// Implementations are interior-mutable (typically `DashMap`-backed) so
/// the scheduler can remove a method's SSA, hand it to a pass for
/// mutation, and reinsert it without acquiring `&mut` on the host.
///
/// # Concurrency
///
/// All methods take `&self` to allow parallel pass execution. Implementations
/// are responsible for thread-safety. The scheduler uses
/// [`take_ssa`](Self::take_ssa) + [`insert_ssa`](Self::insert_ssa)
/// (rather than holding a borrow across the pass call) so peer-reading
/// passes (those that look up other methods' SSA via
/// [`clone_ssa`](Self::clone_ssa)) do not deadlock.
///
/// # Dyn compatibility
///
/// Methods are kept non-generic so the trait can be used as
/// `&dyn SsaStore<T>` and as part of the
/// [`SsaPassHost<T>`](crate::scheduling::SsaPassHost) trait object
/// exposed to passes.
pub trait SsaStore<T: Target> {
    /// Returns `true` if the store has SSA for `method`.
    fn contains(&self, method: &T::MethodRef) -> bool;

    /// Remove and return the SSA for `method`, leaving the slot empty.
    ///
    /// The scheduler uses this to hand exclusive ownership of the SSA
    /// to a pass for mutation. Returns `None` if no SSA exists for
    /// `method`.
    fn take_ssa(&self, method: &T::MethodRef) -> Option<SsaFunction<T>>;

    /// Insert (or replace) the SSA for `method`.
    ///
    /// Called by the scheduler after a pass finishes mutating a method's
    /// SSA. If a previous SSA existed (it was taken but not reinserted),
    /// this replaces it.
    fn insert_ssa(&self, method: T::MethodRef, ssa: SsaFunction<T>);

    /// Returns a clone of the SSA for `method`, if present.
    ///
    /// Used by passes that read peer methods' SSA (e.g., inlining looking
    /// up the callee's body) without taking it out of the store, so other
    /// parallel passes can still access the method.
    fn clone_ssa(&self, method: &T::MethodRef) -> Option<SsaFunction<T>>;

    /// Returns a snapshot of all methods currently in the store.
    ///
    /// Used by the scheduler to seed the dirty set and to determine which
    /// methods need processing.
    fn iter_methods(&self) -> Vec<T::MethodRef>;
}
