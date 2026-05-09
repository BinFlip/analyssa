//! Symbolic expression support for SSA analysis.
//!
//! This module provides symbolic expression construction and evaluation helpers
//! for SSA analysis. Expressions map directly to SSA operations and can be
//! translated to host-side solvers (e.g., Z3) for constraint solving.
//!
//! # Architecture
//!
//! The module uses [`SymbolicExpr`] as an intermediate representation between
//! SSA operations and host-side solvers:
//!
//! ```text
//! SSA Operations -> SymbolicEvaluator -> SymbolicExpr tree -> Host Solver -> Solutions
//! ```
//!
//! Rassa intentionally carries no solver dependency (Z3, etc.). Host crates
//! wire their own solver against `SymbolicExpr`/`SymbolicOp`.
//!
//! # Module Structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`ops`] | Operation types (`SymbolicOp`: add, xor, comparison, etc.) |
//! | [`expr`] | Expression tree representation (`SymbolicExpr`) with simplification |
//! | [`evaluator`] | Builds expressions from SSA operations (`SymbolicEvaluator`) |
//!
//! # Use Cases
//!
//! ## Control Flow Unflattening
//!
//! The primary use case is inverting state encodings used by control flow
//! flattening obfuscators like ConfuserEx. These obfuscators use a pattern:
//!
//! ```text
//! switch_idx = f(state)  // e.g., (state XOR C1) % N
//! switch (switch_idx) {
//!     case 0: ...; state = g0(state); break;
//!     case 1: ...; state = g1(state); break;
//!     ...
//! }
//! ```
//!
//! Given `f(state)` as a symbolic expression and a target case index, a host
//! solver solves `f(state) == target_case` for all states reaching that case.
//!
//! ## Expression Simplification
//!
//! The [`SymbolicExpr::simplify`] method performs constant folding and applies
//! algebraic identities:
//!
//! - Constant folding: `5 + 3` → `8`
//! - Identity removal: `x + 0` → `x`, `x * 1` → `x`
//! - Zero absorption: `x * 0` → `0`
//! - Self-cancellation: `x ^ x = 0`, `x - x = 0`, `x | x = x`
//! - Double negation: `--x` → `x`, `~~x` → `x`
//! - XOR constant cancellation: `(x ^ c) ^ c` → `x`
//! - Sign/ones operations: `x & -1` → `x`, `x | -1` → `-1`, `x ^ -1` → `~x`
//!
//! ## Constraint Satisfaction
//!
//! Arbitrary constraints involving 32-bit bitvector arithmetic:
//! - Arithmetic: add, sub, mul, div, rem
//! - Bitwise: and, or, xor, not, shl, shr
//! - Comparisons: eq, ne, lt, gt, le, ge (signed and unsigned)
//!
//! # Example
//!
//! ```rust,ignore
//! use analyssa::analysis::symbolic::{SymbolicExpr, SymbolicOp};
//!
//! // Build expression: (state XOR 0x12345678) % 13
//! let state = SymbolicExpr::named("state");
//! let xored = SymbolicExpr::binary(SymbolicOp::Xor, state, SymbolicExpr::constant(0x12345678));
//! let result = SymbolicExpr::binary(SymbolicOp::RemU, xored, SymbolicExpr::constant(13));
//!
//! // Hand `result` to a host-side solver to find states that produce case index 5.
//! ```
//!
//! # Performance Considerations
//!
//! Keep host-side solver instances reusable for repeated queries. `analyssa`
//! intentionally carries no solver dependency.

pub mod evaluator;
pub mod expr;
pub mod ops;

// Z3-based solvers live in host crates behind their own feature flags because
// analyssa intentionally carries no Z3 dependency.
// Hosts wire their own solver against `SymbolicExpr`/`SymbolicOp`.

pub use evaluator::SymbolicEvaluator;
pub use expr::SymbolicExpr;
pub use ops::SymbolicOp;
