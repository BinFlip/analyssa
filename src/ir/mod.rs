//! Target-agnostic SSA (Static Single Assignment) intermediate representation core.
//!
//! This module defines the generic SSA IR types used by RASSA. The IR is parameterized
//! over a [`crate::Target`] trait, allowing different host instruction sets (CIL, etc.)
//! to reuse the same SSA infrastructure. Hosts extend these types through their own
//! `Target` implementation and host-specific extension traits or inherent impls.
//!
//! # Architecture
//!
//! The SSA IR is organized into several sub-modules, each handling a specific aspect:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`block`] | Basic blocks containing phi nodes and instructions |
//! | [`instruction`] | SSA instruction wrapper with def/use metadata |
//! | [`ops`] | Decomposed SSA operations (`SsaOp` enum with all op variants) |
//! | [`phi`] | Phi nodes for merging values at control flow joins |
//! | [`value`] | Constant values, abstract value lattice, and CSE tracking |
//! | [`variable`] | Variable IDs, origins, def/use sites, and per-function allocators |
//! | [`exception`] | Exception handler preservation through SSA transformations |
//! | [`function`] | Complete SSA function representation with rebuild/repair/canonicalize |
//!
//! # SSA Construction Pipeline
//!
//! 1. **CIL to SSA**: The `SsaConverter` simulates the stack, creates explicit variables,
//!    places phi nodes at dominance frontiers, and renames to achieve single-assignment.
//! 2. **Optimization**: Analysis passes operate on the SSA form (constant propagation,
//!    copy propagation, DCE, GVN, etc.).
//! 3. **Repair/Rebuild**: After passes, `repair_ssa()` handles instruction-only changes;
//!    `rebuild_ssa()` performs full SSA reconstruction after CFG modifications.
//! 4. **Canonicalize**: Final cleanup (strip nops, remove empty blocks, compact indices)
//!    before code generation.
//!
//! # Key Design Decisions
//!
//! - **Target genericity**: All types are parameterized on `T: Target`, allowing
//!   host-specific metadata (types, methods, fields) without losing SSA structure.
//! - **Dense variable IDs**: `SsaVarId` indices are dense per-function (0, 1, 2, ...)
//!   enabling O(1) variable lookup via direct vector indexing.
//! - **Explicit def/use**: Every SSA operation tracks its destination and operands,
//!   enabling direct def-use chain construction without scanning.
//! - **Thread safety**: All IR types implement `Send` and `Sync`.

pub mod block;
pub mod exception;
pub mod function;
pub mod instruction;
pub mod ops;
pub mod phi;
pub mod value;
pub mod variable;

pub use block::{ReplaceResult, SsaBlock};
pub use exception::{native_is_filter_handler, NativeExceptionKind, SsaExceptionHandler};
pub use function::{
    CheckedReplaceResult, FunctionKind, MethodPurity, ReplacementSkipReason, ReturnInfo,
    SkippedReplacement, SsaBlockBuilder, SsaDefSpec, SsaEditOptions, SsaEditReport, SsaEditScope,
    SsaEditor, SsaFunction, SsaFunctionBuilder, SsaRollbackPolicy, TrivialPhiOptions,
};
pub use instruction::SsaInstruction;
pub use ops::{
    BinaryOpInfo, BinaryOpKind, CmpKind, FlagCondition, FlagsMask, SsaOp, UnaryOpInfo, UnaryOpKind,
};
pub use phi::{PhiNode, PhiOperand};
pub use value::{AbstractValue, ComputedOp, ComputedValue, ConstValue};
pub use variable::{DefSite, FunctionVarAllocator, SsaVarId, SsaVariable, UseSite, VariableOrigin};
