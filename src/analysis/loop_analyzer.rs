//! Loop analyzer for computing comprehensive loop information from SSA.
//!
//! This module provides the [`LoopAnalyzer`] which computes full
//! [`crate::analysis::loops::LoopInfo`] structures from an SSA function,
//! including preheaders, latches, exits, and loop type classification.
//!
//! # Architecture
//!
//! `LoopAnalyzer` is a thin convenience wrapper around the generic `detect_loops`
//! function from the `loops` module:
//!
//! 1. Constructs an `SsaCfg` from the SSA function
//! 2. Computes dominators using `algorithms::compute_dominators`
//! 3. Delegates to `detect_loops` for full loop analysis
//!
//! The separation between `LoopAnalyzer` and `detect_loops` allows the generic
//! loop detection to be used with non-SSA graph types (e.g., CIL CFGs, x86 CFGs)
//! while `LoopAnalyzer` provides a convenient SSA-specific interface.
//!
//! # Complexity
//!
//! Analysis: O(B^2) where B is the number of blocks (dominator computation
//! dominates the runtime). Loop detection is O(E * L) where E is edges and
//! L is the number of loops found.

use crate::{
    analysis::{
        cfg::SsaCfg,
        loops::{detect_loops, LoopForest},
    },
    graph::{algorithms, RootedGraph},
    ir::function::SsaFunction,
    target::Target,
};

/// Analyzes loops in an SSA function.
///
/// The analyzer computes:
/// - Natural loops using dominance-based back edge detection
/// - Preheader identification for each loop
/// - Latch (back edge source) identification
/// - Exit edge detection
/// - Loop type classification
/// - Loop nesting relationships
///
/// This is a thin wrapper around the generic `detect_loops` function,
/// providing a convenient SSA-specific interface.
pub struct LoopAnalyzer<'a, T: Target> {
    /// The SSA-based control flow graph used for dominance computation and loop detection.
    cfg: SsaCfg<'a, T>,
}

impl<'a, T: Target> LoopAnalyzer<'a, T> {
    /// Creates a new loop analyzer for the given SSA function.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>) -> Self {
        let cfg = SsaCfg::from_ssa(ssa);
        Self { cfg }
    }

    /// Analyzes all loops and returns a [`LoopForest`].
    ///
    /// Uses the shared `detect_loops` function which implements dominance-based
    /// back edge detection and computes preheaders, exits, loop types, and nesting.
    #[must_use]
    pub fn analyze(&self) -> LoopForest {
        let dominators = algorithms::compute_dominators(&self.cfg, self.cfg.entry());
        detect_loops(&self.cfg, &dominators)
    }
}

/// Extension trait for SSA functions to easily access loop analysis.
pub trait SsaLoopAnalysis {
    /// Analyzes loops in this function.
    fn analyze_loops(&self) -> LoopForest;
}

impl<T: Target> SsaLoopAnalysis for SsaFunction<T> {
    fn analyze_loops(&self) -> LoopForest {
        LoopAnalyzer::new(self).analyze()
    }
}
