//! Data flow analysis framework — direction, analysis trait, results.
//!
//! Defines the core abstractions for data flow analysis:
//!
//! - [`DataFlowCfg`]: Trait for control flow graphs usable with the solver.
//!   Abstracts over CFG implementations so the solver can run on SSA-level
//!   [`SsaCfg`] graphs and host-provided CFG types.
//! - [`Direction`]: Forward (entry-to-exit) or backward (exit-to-entry).
//!   Determines how information propagates and which combining operation is used.
//! - [`DataFlowAnalysis`]: Trait for a specific analysis. Implementations supply
//!   the transfer function, boundary conditions, and initial values.
//! - [`AnalysisResults`]: Per-block input and output lattice values.
//!
//! Hosts plug in their own CFG via [`DataFlowCfg`]; analyssa supplies the SSA
//! CFG (`SsaCfg`) implementation via the blanket impl at the bottom of this file.

use std::fmt::Debug;

use crate::{
    analysis::{cfg::SsaCfg, dataflow::lattice::MeetSemiLattice},
    graph::{NodeId, Predecessors, RootedGraph, Successors},
    ir::{block::SsaBlock, function::SsaFunction},
    target::Target,
};

/// Trait for control flow graphs usable with the dataflow solver.
///
/// Abstracts over CFG implementations so the solver can run on SSA-level
/// [`SsaCfg`] graphs and host-provided CFG types.
pub trait DataFlowCfg: Predecessors + Successors {
    /// Returns the entry node of the CFG.
    fn entry(&self) -> NodeId;

    /// Returns the exit nodes of the CFG.
    fn exits(&self) -> Vec<NodeId>;

    /// Returns nodes in postorder (for backward analysis).
    fn postorder(&self) -> Vec<NodeId>;

    /// Returns nodes in reverse postorder (for forward analysis).
    fn reverse_postorder(&self) -> Vec<NodeId>;
}

/// Direction of data flow analysis.
///
/// The direction determines how information propagates through the CFG
/// and which operation (meet or join) is used at control flow merge
/// points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Information flows forward, from entry to exit.
    ///
    /// At join points (blocks with multiple predecessors), values from
    /// all predecessors are combined using the meet operation. Examples:
    /// reaching definitions, available expressions, constant propagation.
    Forward,

    /// Information flows backward, from exit to entry.
    ///
    /// At split points (blocks with multiple successors), values from
    /// all successors are combined. Examples: live variables, very busy
    /// expressions.
    Backward,
}

/// A data flow analysis that can be run on SSA form.
///
/// Implementations supply the transfer function and boundary conditions;
/// the solver handles iteration to a fixpoint. Generic over `T: Target` so
/// that boundary/initial/transfer can read host-specific information from
/// `SsaFunction<T>` (e.g. argument count, types) without committing the
/// framework to one host.
pub trait DataFlowAnalysis<T: Target> {
    /// The lattice type for this analysis.
    type Lattice: MeetSemiLattice;

    /// The direction of this analysis.
    const DIRECTION: Direction;

    /// Returns the initial value at the boundary of the function.
    ///
    /// For forward analyses this is the value at function entry; for
    /// backward analyses the value at function exit(s).
    fn boundary(&self, ssa: &SsaFunction<T>) -> Self::Lattice;

    /// Returns the initial value for interior blocks (typically the lattice's top).
    fn initial(&self, ssa: &SsaFunction<T>) -> Self::Lattice;

    /// Computes the transfer function for a basic block. Given the input
    /// state to a block, returns the output state after flowing through the
    /// block.
    fn transfer(
        &self,
        block_id: usize,
        block: &SsaBlock<T>,
        input: &Self::Lattice,
        ssa: &SsaFunction<T>,
    ) -> Self::Lattice;

    /// Called when analysis is complete. Default: no post-processing.
    fn finalize(
        &mut self,
        _in_states: &[Self::Lattice],
        _out_states: &[Self::Lattice],
        _ssa: &SsaFunction<T>,
    ) {
    }
}

/// Results of a data flow analysis: per-block input and output lattice
/// values.
#[derive(Debug, Clone)]
pub struct AnalysisResults<L> {
    /// Input state for each block (before transfer function).
    pub in_states: Vec<L>,
    /// Output state for each block (after transfer function).
    pub out_states: Vec<L>,
}

impl<L: Clone> AnalysisResults<L> {
    /// Creates new analysis results with the given states.
    #[must_use]
    pub fn new(in_states: Vec<L>, out_states: Vec<L>) -> Self {
        Self {
            in_states,
            out_states,
        }
    }

    /// Returns the input state for a block, or `None` if out of bounds.
    #[must_use]
    pub fn in_state(&self, block: usize) -> Option<&L> {
        self.in_states.get(block)
    }

    /// Returns the output state for a block, or `None` if out of bounds.
    #[must_use]
    pub fn out_state(&self, block: usize) -> Option<&L> {
        self.out_states.get(block)
    }

    /// Returns the number of blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.in_states.len()
    }
}

// `DataFlowCfg` implementation for analyssa's SSA CFG.
impl<T: Target> DataFlowCfg for SsaCfg<'_, T> {
    fn entry(&self) -> NodeId {
        RootedGraph::entry(self)
    }

    fn exits(&self) -> Vec<NodeId> {
        self.exits()
    }

    fn postorder(&self) -> Vec<NodeId> {
        self.postorder()
    }

    fn reverse_postorder(&self) -> Vec<NodeId> {
        self.reverse_postorder()
    }
}
