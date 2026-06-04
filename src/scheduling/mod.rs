//! Pass-pipeline scheduling engine — orchestrates [`SsaPass`] execution
//! across methods with capability-based layering, fixpoint iteration, and
//! parallel dispatch.
//!
//! # Architecture
//!
//! The scheduler organizes passes into execution layers computed from their
//! declared [capabilities](DeobfuscationCapability). Passes that provide a
//! capability are placed before passes that require it, ensuring producers
//! run before consumers.
//!
//! # Usage
//!
//! Hosts implement [`SsaPassHost<T>`] on their context type (which bundles
//! [`World<T>`](crate::world::World), [`SsaStore<T>`](crate::host::SsaStore),
//! and [`DirtySet<T>`](crate::host::DirtySet)), register passes with a
//! [`PassScheduler`], and call [`PassScheduler::run_pipeline`].
//!
//! # Features
//!
//! - **Capability-based ordering**: passes declare `provides`/`requires`;
//!   the scheduler topologically sorts them into layers.
//! - **Fixpoint iteration**: each layer runs to convergence with normalize
//!   passes (DCE, GVN) interleaved between iterations.
//! - **Parallel dispatch**: per-method pass execution via rayon.
//! - **Modification-scope-driven repair**: after each pass, the scheduler
//!   applies the minimum SSA repair needed (uses-only, instructions-only,
//!   or full rebuild) based on the pass's declared
//!   [`ModificationScope`].
//! - **Dirty tracking**: only methods that may have changed are re-processed
//!   on subsequent iterations, unless a pass declares
//!   [`requires_full_scan`](SsaPass::requires_full_scan).

mod capability;
mod pass;
mod scheduler;

pub use capability::DeobfuscationCapability;
pub use pass::{ModificationScope, SsaPass, SsaPassHost};
pub use scheduler::{PassScheduler, PipelineConfig};
