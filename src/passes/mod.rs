//! Built-in SSA optimization passes for the analyssa deobfuscation framework.
//!
//! Each pass exposes a free function, typically named `run`, that takes
//! `&mut SsaFunction<T>` plus an event sink and mutates the IR in place.
//! Hosts can call those functions directly or use the pass wrappers re-exported
//! by this module with [`crate::scheduling::PassScheduler`].
//!
//! # Pass Categories
//!
//! - **Normalization**: [`deadcode`], [`copying`], [`gvn`], [`blockmerge`] —
//!   clean up the IR after structural transformations. Run between every
//!   pipeline layer's fixpoint iterations.
//! - **Structural**: [`controlflow`], [`threading`], [`loopcanon`], [`blockmerge`] —
//!   simplify and canonicalize the control-flow graph.
//! - **Value-level**: [`algebraic`], [`strength`], [`reassociate`], [`ranges`],
//!   [`predicates`], [`licm`] — replace expensive or redundant computations
//!   with cheaper or constant equivalents.
//!
//! # Scheduling
//!
//! The `scheduling` sub-module wraps each pass body in an
//! [`SsaPass`](crate::scheduling::SsaPass) trait impl so the
//! [`PassScheduler`](crate::scheduling::PassScheduler) can orchestrate them
//! with capability-based ordering, fixpoint iteration, and parallel dispatch.

pub mod algebraic;
pub mod blockmerge;
pub mod controlflow;
pub mod copying;
pub mod deadcode;
pub mod gvn;
pub mod licm;
pub mod loopcanon;
pub mod predicates;
pub mod ranges;
pub mod reassociate;
mod scheduling;
pub mod strength;
pub mod threading;
pub mod utils;

pub use predicates::PredicateResult;
pub use scheduling::{
    AlgebraicSimplificationPass, BlockMergingPass, ControlFlowSimplificationPass,
    CopyPropagationPass, DeadCodeEliminationPass, DeadMethodEliminationPass,
    GlobalValueNumberingPass, JumpThreadingPass, LicmPass, LoopCanonicalizationPass,
    OpaquePredicatePass, ReassociationPass, StrengthReductionPass, ValueRangePropagationPass,
};
