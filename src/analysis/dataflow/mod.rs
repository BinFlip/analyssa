//! Data flow analysis framework for SSA form.
//!
//! Provides a generic framework for classical data flow analyses:
//!
//! - **Traits**: `DataFlowAnalysis`, `DataFlowCfg`, `MeetSemiLattice`
//! - **Solver**: Worklist-based iterative fixpoint computation via `DataFlowSolver`
//! - **Analyses**: Liveness (backward), reaching definitions (forward), SCCP
//!
//! # Architecture
//!
//! ```text
//! DataFlowCfg (CFG abstraction)     MeetSemiLattice (abstract domain)
//!         |                                   |
//!         v                                   v
//! DataFlowAnalysis<T> (transfer function + boundary)
//!         |
//!         v
//! DataFlowSolver (worklist fixpoint iteration)
//!         |
//!         v
//! AnalysisResults (per-block in/out states)
//! ```
//!
//! Hosts implement [`DataFlowCfg`] for their CFG type. Rassa supplies the
//! implementation for [`SsaCfg`](crate::analysis::cfg::SsaCfg).
//!
//! # Convergence
//!
//! The solver uses reverse postorder (forward) or postorder (backward) for
//! initial worklist ordering and propagates changes to successors/predecessors
//! when a block's state changes. On reducible CFGs, convergence typically
//! occurs in O(n) iterations where n is the loop nesting depth.

pub mod framework;
pub mod lattice;
pub mod liveness;
pub mod reaching;
pub mod sccp;
pub mod solver;

pub use framework::{AnalysisResults, DataFlowAnalysis, DataFlowCfg, Direction};
pub use liveness::{LiveVariables, LivenessResult};
pub use reaching::ReachingDefinitions;
pub use sccp::{ConstantPropagation, ScalarValue, SccpResult};
pub use solver::DataFlowSolver;
