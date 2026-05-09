//! Host adapter traits for the pass scheduler's storage and scheduling
//! primitives.
//!
//! Where [`World<T>`](crate::world::World) provides the minimal
//! "what methods exist and who calls whom" surface used by global passes,
//! the traits in this module add the storage and scheduling primitives
//! the pass scheduler needs:
//!
//! - [`SsaStore<T>`] — interior-mutable, per-method SSA storage. The
//!   scheduler removes a method's SSA via [`take_ssa`](SsaStore::take_ssa),
//!   hands it to a pass for mutation, and reinserts it via
//!   [`insert_ssa`](SsaStore::insert_ssa) without needing `&mut` on the host.
//! - [`DirtySet<T>`] — tracks which methods need re-processing in the
//!   next fixpoint iteration. Passes mark methods dirty as they discover
//!   new work (e.g. inlining flags the caller as needing re-analysis).
//!
//! # Concurrency
//!
//! Both traits use `&self` with interior mutability (typically
//! `DashMap`/`DashSet`-backed), following the
//! [`World::mark_dead`](crate::world::World::mark_dead) precedent —
//! global passes can run in parallel without acquiring `&mut` on the host.
//!
//! # Usage
//!
//! Hosts that do not run the analyssa scheduler do not need to implement
//! these traits. Hosts that do typically implement all three
//! (`World<T>`, `SsaStore<T>`, `DirtySet<T>`) on a single context type.

mod dirty;
mod store;

pub use dirty::DirtySet;
pub use store::SsaStore;
