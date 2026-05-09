//! Target-agnostic SSA IR, analyses, and optimization pipeline.
//!
//! `analyssa` provides a reusable SSA (Static Single Assignment) intermediate representation
//! designed for lifters, deobfuscators, and binary-analysis tooling that need
//! target-specific metadata outside the IR. Hosts implement the [`Target`] trait plus
//! optional scheduler adapter traits to run analyssa analyses and transformation passes
//! over their own method storage.
//!
//! # Architecture
//!
//! The crate is organized into several layers:
//!
//! **Core IR** (`ir`): The SSA representation itself, including blocks
//! ([`crate::ir::SsaBlock`]), instructions ([`crate::ir::SsaInstruction`]), operations ([`crate::ir::SsaOp`]),
//! phi nodes ([`crate::ir::PhiNode`]), constant values ([`crate::ir::ConstValue`]), variables
//! ([`crate::ir::SsaVariable`]), exception handlers ([`crate::ir::SsaExceptionHandler`]), and the
//! function container ([`crate::ir::SsaFunction`]).
//!
//! **Graph Infrastructure** (`graph`): A generic directed graph with node,
//! edge, and indexed-graph abstractions, plus algorithms for dominators,
//! strongly-connected components, cycle detection, topological sort, and
//! traversal (BFS/DFS).
//!
//! **Analyses** (`analysis`): A suite of SSA analyses including control-flow
//! graph computation, constant propagation, def-use chains, liveness analysis,
//! memory SSA, pattern matching, phi placement, range analysis, symbolic
//! execution via an abstract evaluator, a dataflow-analysis framework (lattice,
//! solver, SCCP, reaching definitions, dataflow liveness), and a verifier.
//!
//! **Transformation Passes** (`passes`): Optimization and deobfuscation passes
//! including algebraic simplification, block merging, control-flow
//! restructuring, copy propagation, dead code elimination, global value
//! numbering (GVN), loop-invariant code motion (LICM), loop canonicalization,
//! predicate simplification, range propagation, reassociation, strength
//! reduction, and jump threading.
//!
//! **Pass Scheduling** (`scheduling`): A framework for declaring pass
//! capabilities, dependencies, and ordering via [`SsaPass`], [`SsaPassHost`],
//! and [`PassScheduler`], augmented by [`DeobfuscationCapability`] and
//! [`ModificationScope`] for dependency-aware scheduling.
//!
//! **Host Abstractions**: The [`Target`] trait defines the contract between
//! analyssa and the host instruction set. The [`World`] trait provides an
//! interprocedural view of the program under analysis. [`SsaStore`] and
//! [`DirtySet`] provide storage and dirty-tracking.
//!
//! **Utilities**: [`BitSet`] for efficient bit-vector operations in dataflow
//! analyses, [`PointerSize`] for target pointer width, and event logging
//! ([`Event`], [`EventLog`], [`EventKind`], [`DerivedStats`]) for observing
//! pass transformations.
//!
//! # Design Principles
//!
//! - **Target agnosticism**: All IR types are generic over `<T: Target>`;
//!   host-specific metadata is hidden behind associated types.
//! - **Explicit data flow**: Every SSA instruction has explicit operands
//!   (uses) and results (defs), enabling straightforward analysis.
//! - **Thread safety**: Analyses and passes use `&self` with interior
//!   mutability where needed; event logging is lock-free via `boxcar::Vec`.
//! - **Composability**: Passes declare their capabilities and dependencies
//!   through the scheduling framework, supporting automatic ordering.
//!
//! # Usage
//!
//! 1. Implement [`Target`] for your instruction set
//! 2. Implement [`SsaStore`] to manage method storage
//! 3. Optionally implement [`World`] for interprocedural passes
//! 4. Run individual analyses and passes, or use [`PassScheduler`] for
//!    automatic pipeline execution

#![deny(missing_docs)]
#![deny(rustdoc::bare_urls)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::redundant_explicit_links)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing
    )
)]

pub mod analysis;
pub mod bitset;
mod error;
pub mod events;
pub mod graph;
pub mod host;
pub mod ir;
pub mod passes;
mod pointer;
pub mod scheduling;
pub mod target;
pub mod testing;
pub mod world;

pub use bitset::BitSet;
pub use error::{Error, GraphError, Result};
pub use events::{
    DerivedStats, Event, EventBuilder, EventKind, EventListener, EventLog, NullListener,
};
pub use host::{DirtySet, SsaStore};
pub use pointer::PointerSize;
pub use scheduling::{
    DeobfuscationCapability, ModificationScope, PassScheduler, SsaPass, SsaPassHost,
};
pub use target::{Endianness, Target};
pub use testing::{MockTarget, MockType};
pub use world::World;
