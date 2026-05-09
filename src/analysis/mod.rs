//! SSA analyses: target-agnostic dataflow, liveness, type/value tracking,
//! constant evaluation, memory model, and symbolic execution.
//!
//! # Architecture
//!
//! The analysis module is organized into several layers:
//!
//! ## Core Analyses
//!
//! - **`cfg`**: Lightweight control flow graph view built from SSA function terminators.
//!   Bridges the gap between passes (which receive `SsaFunction`) and dataflow analyses
//!   (which require a CFG). Constructed in O(E) time.
//! - **`defuse`**: Def-Use index providing O(1) lookups of definition sites, use sites,
//!   and per-location variable queries. Built in O(n) time where n is instruction count.
//! - **`liveness`**: Backward dataflow to compute live-in blocks for each variable group.
//!   Used to prune phi placement (dead-on-arrival phi avoidance).
//! - **`constraints`**: Constraint types derived from branch conditions for path-aware
//!   SSA evaluation. Supports equality, inequality, signed/unsigned ordering constraints.
//! - **`range`**: Interval-based value range analysis with a lattice structure supporting
//!   constant, bounded, half-open, and union ranges for opaque predicate detection and
//!   bounds check elimination.
//!
//! ## Evaluation
//!
//! - **`evaluator`**: Hybrid concrete/symbolic SSA interpreter that computes values
//!   for arithmetic and logical operations given known inputs. Supports path-aware
//!   phi evaluation, constraint tracking, and fixed-point loop iteration.
//! - **`consts`**: Constant folding engine with caching and cycle detection. Used by
//!   multiple passes (unflattening, decryption, SCCP). Depth-limited recursive evaluation.
//! - **`resolver`**: Three-tier constant resolver composing ConstEvaluator, PhiAnalyzer,
//!   and optionally SsaEvaluator for demand-driven constant resolution.
//!
//! ## Symbolic Execution
//!
//! - **`symbolic/expr`**: Symbolic expression tree (`SymbolicExpr`) representing SSA
//!   operations as trees with constants, variables, and operation nodes.
//! - **`symbolic/ops`**: Operation types for symbolic expressions (arithmetic, bitwise,
//!   comparison with signed/unsigned variants).
//! - **`symbolic/evaluator`**: Builds symbolic expression trees from SSA operations for
//!   host-side constraint solving.
//!
//! ## SSA Structure Analysis
//!
//! - **`loops`**: Natural loop detection using dominance-based back edge detection.
//!   Computes preheaders, latches, exit edges, loop type classification, nesting
//!   relationships, and induction variable detection.
//! - **`loop_analyzer`**: Convenience wrapper around `detect_loops` providing
//!   SSA-specific loop analysis interface.
//! - **`phis`**: Phi node analysis utilities (trivial phi detection, uniform constant
//!   detection) and pruned phi placement at iterated dominance frontiers.
//! - **`algebraic`**: Algebraic identity simplification (XOR self-cancellation,
//!   identity/absorbing element detection).
//! - **`patterns`**: Obfuscation pattern detection (control flow flattening dispatchers,
//!   opaque predicates, source block identification).
//! - **`taint`**: Generic forward/backward taint propagation with configurable PHI
//!   handling modes. Used for CFF state tracking and cleanup neutralization.
//!
//! ## Memory Analysis
//!
//! - **`memory`**: Memory SSA (MSSA) for tracking versioned memory locations. Supports
//!   static fields, instance fields, array elements, and indirect accesses through
//!   a hierarchical alias analysis.
//!
//! ## Dataflow Framework
//!
//! - **`dataflow/framework`**: Generic dataflow analysis traits (`DataFlowAnalysis`,
//!   `DataFlowCfg`) and direction abstraction.
//! - **`dataflow/lattice`**: Lattice traits (MeetSemiLattice, JoinSemiLattice, Lattice)
//!   with BitSet implementations for may/must analysis.
//! - **`dataflow/solver`**: Worklist-based iterative fixpoint solver using reverse
//!   postorder traversal. Converges in O(n*h) on reducible CFGs.
//! - **`dataflow/liveness`**: Backward live variable analysis computing USE/DEF sets.
//! - **`dataflow/reaching`**: Forward reaching definitions analysis (simplified for SSA).
//! - **`dataflow/sccp`**: Sparse Conditional Constant Propagation combining sparse
//!   def-use analysis with branch condition pruning. Uses edge-based phi evaluation
//!   per Wegman & Zadeck 1991.
//!
//! ## Verification
//!
//! - **`verifier`**: SSA invariant verifier at three levels (Quick/Standard/Full)
//!   checking single-definition, def-use chains, phi operand coverage, dominance,
//!   and structural integrity.

pub mod algebraic;
pub mod cfg;
pub mod constraints;
pub mod consts;
pub mod dataflow;
pub mod defuse;
pub mod evaluator;
pub mod liveness;
pub mod loop_analyzer;
pub mod loops;
pub mod memory;
pub mod patterns;
pub mod phis;
pub mod range;
pub mod resolver;
pub mod symbolic;
pub mod taint;
pub mod verifier;

pub use algebraic::{simplify_op, SimplifyResult};
pub use cfg::SsaCfg;
pub use consts::{evaluate_const_op, ConstEvaluator};
pub use defuse::{DefUseIndex, Location};
pub use evaluator::{ControlFlow, SsaEvaluator};
pub use loop_analyzer::{LoopAnalyzer, SsaLoopAnalysis};
pub use loops::{detect_loops, InductionVar, LoopForest, LoopInfo, LoopType};
pub use patterns::PatternDetector;
pub use phis::{place_pruned_phis, PhiAnalyzer};
pub use range::ValueRange;
pub use resolver::ValueResolver;
pub use symbolic::{SymbolicEvaluator, SymbolicExpr, SymbolicOp};
pub use verifier::{SsaVerifier, VerifierError, VerifyLevel};
