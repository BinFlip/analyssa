//! Worklist-based data flow solver.
//!
//! This module provides the iterative solver that computes fixpoints for
//! data flow analyses. It uses a worklist algorithm with traversal order
//! optimized for the analysis direction.
//!
//! # Algorithm
//!
//! The solver iterates until a fixpoint is reached:
//!
//! **Initialization**:
//! 1. All blocks start with the analysis's `initial()` value
//! 2. Set `boundary()` value at entry (forward) or exits (backward)
//! 3. Add all blocks to the worklist in traversal order (reverse postorder
//!    for forward, postorder for backward)
//!
//! **Iteration**:
//! 1. Pop a block from the worklist (deduplicated via `in_worklist` flag)
//! 2. Compute the input/output by meeting lattice values from all
//!    predecessors (forward) or successors (backward)
//! 3. Preserve boundary values for entry/exit blocks
//! 4. Apply the transfer function to compute the output/input
//! 5. If the result changed, add affected adjacent blocks
//!    (successors for forward, predecessors for backward) to the worklist
//!
//! **Finalization**:
//! 1. Call the analysis's `finalize()` hook for any post-processing
//! 2. Return `AnalysisResults` with per-block in/out states
//!
//! # Complexity
//!
//! On reducible CFGs, converges in O(d) iterations where d is the loop
//! nesting depth (typically small). Total work: O(n * d * h) where n is
//! the block count and h is the lattice height (maximum chain length from
//! top to bottom). Each iteration processes each block at most once via
//! the worklist deduplication flag.

use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::{
    analysis::dataflow::{
        framework::{AnalysisResults, DataFlowAnalysis, DataFlowCfg, Direction},
        lattice::MeetSemiLattice,
    },
    graph::NodeId,
    ir::function::SsaFunction,
    target::Target,
};

/// Worklist-based data flow solver.
///
/// This solver computes fixpoints for data flow analyses using an iterative
/// worklist algorithm. It supports both forward and backward analyses.
///
/// # Usage
///
/// ```rust,ignore
/// use analyssa::analysis::dataflow::{DataFlowSolver, ReachingDefinitions};
///
/// let analysis = ReachingDefinitions::new(&ssa);
/// let mut solver = DataFlowSolver::new(analysis);
/// let results = solver.solve(&ssa, &graph);
///
/// // Access results
/// let in_state = results.in_state(block_id);
/// ```
pub struct DataFlowSolver<T: Target, A: DataFlowAnalysis<T>> {
    /// The analysis being solved.
    analysis: A,
    /// Input state for each block.
    in_states: Vec<A::Lattice>,
    /// Output state for each block.
    out_states: Vec<A::Lattice>,
    /// Worklist of blocks to process.
    worklist: VecDeque<usize>,
    /// Whether each block is currently in the worklist (for deduplication).
    in_worklist: Vec<bool>,
    /// Number of iterations performed.
    iterations: usize,
    _phantom: PhantomData<T>,
}

impl<T: Target, A: DataFlowAnalysis<T>> DataFlowSolver<T, A> {
    /// Creates a new solver for the given analysis.
    #[must_use]
    pub fn new(analysis: A) -> Self {
        Self {
            analysis,
            in_states: Vec::new(),
            out_states: Vec::new(),
            worklist: VecDeque::new(),
            in_worklist: Vec::new(),
            iterations: 0,
            _phantom: PhantomData,
        }
    }

    /// Solves the data flow analysis to a fixpoint.
    ///
    /// Returns the analysis results containing input and output states
    /// for each basic block.
    pub fn solve<C: DataFlowCfg>(
        mut self,
        ssa: &SsaFunction<T>,
        cfg: &C,
    ) -> AnalysisResults<A::Lattice>
    where
        A::Lattice: Clone,
    {
        let num_blocks = ssa.block_count();
        if num_blocks == 0 {
            return AnalysisResults::new(Vec::new(), Vec::new());
        }

        // Initialize states
        self.initialize(ssa, cfg);

        // Main iteration loop
        self.iterate(ssa, cfg);

        // Finalize
        self.analysis
            .finalize(&self.in_states, &self.out_states, ssa);

        AnalysisResults::new(self.in_states, self.out_states)
    }

    /// Returns the number of iterations performed.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Initializes the solver state.
    fn initialize<C: DataFlowCfg>(&mut self, ssa: &SsaFunction<T>, cfg: &C)
    where
        A::Lattice: Clone,
    {
        let num_blocks = ssa.block_count();
        let initial = self.analysis.initial(ssa);
        let boundary = self.analysis.boundary(ssa);

        // Initialize all blocks with the initial value
        self.in_states = vec![initial.clone(); num_blocks];
        self.out_states = vec![initial; num_blocks];
        self.in_worklist = vec![false; num_blocks];

        // Set boundary conditions based on direction
        match A::DIRECTION {
            Direction::Forward => {
                // Entry block gets boundary value
                let entry = cfg.entry().index();
                if let Some(slot) = self.in_states.get_mut(entry) {
                    *slot = boundary;
                }
            }
            Direction::Backward => {
                // Exit blocks get boundary value
                for exit in cfg.exits() {
                    let idx = exit.index();
                    if let Some(slot) = self.out_states.get_mut(idx) {
                        *slot = boundary.clone();
                    }
                }
            }
        }

        // Add all blocks to worklist in appropriate order
        let order = match A::DIRECTION {
            Direction::Forward => cfg.reverse_postorder(),
            Direction::Backward => cfg.postorder(),
        };

        for node in order {
            let idx = node.index();
            if let Some(slot) = self.in_worklist.get_mut(idx) {
                self.worklist.push_back(idx);
                *slot = true;
            }
        }
    }

    /// Main iteration loop.
    fn iterate<C: DataFlowCfg>(&mut self, ssa: &SsaFunction<T>, cfg: &C)
    where
        A::Lattice: Clone,
    {
        while let Some(block_idx) = self.worklist.pop_front() {
            if let Some(slot) = self.in_worklist.get_mut(block_idx) {
                *slot = false;
            }
            self.iterations = self.iterations.saturating_add(1);

            let changed = match A::DIRECTION {
                Direction::Forward => self.process_forward(block_idx, ssa, cfg),
                Direction::Backward => self.process_backward(block_idx, ssa, cfg),
            };

            if changed {
                // Add affected blocks to worklist
                self.add_affected_to_worklist(block_idx, cfg);
            }
        }
    }

    /// Processes a block in forward direction.
    ///
    /// Returns `true` if the output state changed.
    fn process_forward<C: DataFlowCfg>(
        &mut self,
        block_idx: usize,
        ssa: &SsaFunction<T>,
        cfg: &C,
    ) -> bool
    where
        A::Lattice: Clone,
    {
        // Compute input by meeting all predecessor outputs
        let node = NodeId::new(block_idx);
        let Some(current_in) = self.in_states.get(block_idx).cloned() else {
            return false;
        };
        let mut input = if cfg.predecessors(node).next().is_none() {
            // Entry block or unreachable - keep current in_state
            current_in.clone()
        } else {
            // Meet all predecessor outputs
            let mut result: Option<A::Lattice> = None;
            for pred in cfg.predecessors(node) {
                let Some(pred_out) = self.out_states.get(pred.index()) else {
                    continue;
                };
                result = Some(match result {
                    None => pred_out.clone(),
                    Some(acc) => acc.meet(pred_out),
                });
            }
            result.unwrap_or_else(|| current_in.clone())
        };

        // Special case: entry block keeps its boundary value
        if node == cfg.entry() {
            input = current_in.clone();
        }

        if let Some(slot) = self.in_states.get_mut(block_idx) {
            *slot = input.clone();
        }

        // Apply transfer function
        let Some(block) = ssa.block(block_idx) else {
            return false;
        };
        let output = self.analysis.transfer(block_idx, block, &input, ssa);

        // Check if output changed
        let Some(out_slot) = self.out_states.get_mut(block_idx) else {
            return false;
        };
        let changed = output != *out_slot;
        *out_slot = output;

        changed
    }

    /// Processes a block in backward direction.
    ///
    /// Returns `true` if the input state changed.
    fn process_backward<C: DataFlowCfg>(
        &mut self,
        block_idx: usize,
        ssa: &SsaFunction<T>,
        cfg: &C,
    ) -> bool
    where
        A::Lattice: Clone,
    {
        // Compute output by meeting all successor inputs
        let node = NodeId::new(block_idx);
        let Some(current_out) = self.out_states.get(block_idx).cloned() else {
            return false;
        };
        let mut output = if cfg.successors(node).next().is_none() {
            // Exit block or dead end - keep current out_state
            current_out.clone()
        } else {
            // Meet all successor inputs
            let mut result: Option<A::Lattice> = None;
            for succ in cfg.successors(node) {
                let Some(succ_in) = self.in_states.get(succ.index()) else {
                    continue;
                };
                result = Some(match result {
                    None => succ_in.clone(),
                    Some(acc) => acc.meet(succ_in),
                });
            }
            result.unwrap_or_else(|| current_out.clone())
        };

        // Special case: exit blocks keep their boundary value
        if cfg.exits().contains(&node) {
            output = current_out.clone();
        }

        if let Some(slot) = self.out_states.get_mut(block_idx) {
            *slot = output.clone();
        }

        // Apply transfer function (backward: input = transfer(output))
        let Some(block) = ssa.block(block_idx) else {
            return false;
        };
        let input = self.analysis.transfer(block_idx, block, &output, ssa);

        // Check if input changed
        let Some(in_slot) = self.in_states.get_mut(block_idx) else {
            return false;
        };
        let changed = input != *in_slot;
        *in_slot = input;

        changed
    }

    /// Adds affected blocks to the worklist after a change.
    fn add_affected_to_worklist<C: DataFlowCfg>(&mut self, block_idx: usize, cfg: &C) {
        let node = NodeId::new(block_idx);

        let enqueue = |idx: usize, list: &mut Vec<bool>, work: &mut VecDeque<usize>| {
            if let Some(slot) = list.get_mut(idx) {
                if !*slot {
                    work.push_back(idx);
                    *slot = true;
                }
            }
        };

        match A::DIRECTION {
            Direction::Forward => {
                // Forward: successors are affected
                for succ in cfg.successors(node) {
                    enqueue(succ.index(), &mut self.in_worklist, &mut self.worklist);
                }
            }
            Direction::Backward => {
                // Backward: predecessors are affected
                for pred in cfg.predecessors(node) {
                    enqueue(pred.index(), &mut self.in_worklist, &mut self.worklist);
                }
            }
        }
    }
}
