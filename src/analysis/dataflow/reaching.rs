//! Reaching definitions analysis using the dataflow framework.
//!
//! Reaching definitions computes, for each program point, which variable
//! definitions may reach that point. A definition of variable V reaches a
//! point P if there exists a path from the definition to P that does not
//! contain another definition of V.
//!
//! # SSA Form
//!
//! In SSA form, each variable is defined exactly once, so reaching definitions
//! is simplified: a definition reaches a use iff there's a path from the
//! definition to the use. This is always true in well-formed SSA.
//!
//! This analysis is still useful for:
//! - Validating SSA construction
//! - Computing def-use chains from the CFG perspective
//! - Detecting dead definitions
//!
//! # Algorithm
//!
//! Forward data flow analysis:
//!
//! | Set | Definition |
//! |-----|------------|
//! | `GEN[B]` | Definitions created in B (phi nodes + instruction defs) |
//! | `KILL[B]` | Definitions killed in B (in SSA: none) |
//! | `IN[B]` | `∪ { OUT(p) for p in pred(B) }` |
//! | `OUT[B]` | `GEN[B] ∪ (IN[B] \\ KILL[B])` → simplifies to `GEN[B] ∪ IN[B]` |
//!
//! ## Lattice
//!
//! The `ReachingDefsResult` lattice uses `MeetSemiLattice` with:
//! - **Meet = union**: A definition reaches if it reaches from ANY predecessor
//!   (may analysis — standard for reaching definitions)
//! - **Boundary**: At function entry, initial definitions of arguments and
//!   version-0 locals are present
//! - **Initial**: Interior blocks start with empty sets

use crate::{
    analysis::dataflow::{
        framework::{DataFlowAnalysis, Direction},
        lattice::MeetSemiLattice,
    },
    bitset::BitSet,
    ir::{block::SsaBlock, function::SsaFunction, variable::SsaVarId},
    target::Target,
};

/// Reaching definitions analysis.
///
/// Computes which variable definitions may reach each block.
///
/// # Example
///
/// ```rust
/// use analyssa::{
///     analysis::{
///         dataflow::{DataFlowSolver, ReachingDefinitions},
///         SsaCfg,
///     },
///     ir::SsaVarId,
///     testing,
/// };
///
/// // Diamond CFG: block 0 branches to blocks 1 and 2, which both jump to 3.
/// let ssa = testing::diamond_phi_fixture();
/// let graph = SsaCfg::from_ssa(&ssa);
///
/// let analysis = ReachingDefinitions::new(&ssa);
/// let solver = DataFlowSolver::new(analysis);
/// let results = solver.solve(&ssa, &graph);
///
/// // Both arms of the diamond reach the merge block.
/// let condition = SsaVarId::from_index(0);
/// let left = SsaVarId::from_index(1);
/// let right = SsaVarId::from_index(2);
/// let reaching: Vec<_> = results.in_state(3).unwrap().definitions().collect();
/// assert!(reaching.contains(&left));
/// assert!(reaching.contains(&right));
///
/// // A definition does not reach the block that creates it, but does reach
/// // its successors: the condition is defined in block 0, so it reaches
/// // block 1 but is absent from block 0's entry state.
/// let entry: Vec<_> = results.in_state(0).unwrap().definitions().collect();
/// assert!(!entry.contains(&condition));
/// assert!(results.in_state(1).unwrap().definitions().any(|v| v == condition));
/// ```
pub struct ReachingDefinitions {
    /// Number of variables in the function.
    num_vars: usize,
    /// GEN sets for each block (definitions created in the block).
    gen_sets: Vec<BitSet>,
}

impl ReachingDefinitions {
    /// Creates a new reaching definitions analysis for the given SSA function.
    #[must_use]
    pub fn new<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let num_vars = ssa.variable_count();
        let num_blocks = ssa.block_count();

        // Compute GEN sets
        let mut gen_sets = Vec::with_capacity(num_blocks);

        for block in ssa.blocks() {
            let mut gen = BitSet::new(num_vars);

            // Phi nodes define variables
            for phi in block.phi_nodes() {
                if let Some(idx) = ssa.var_index(phi.result()) {
                    gen.insert(idx);
                }
            }

            // Instructions may define variables
            for instr in block.instructions() {
                for def in instr.defs() {
                    if let Some(idx) = ssa.var_index(def) {
                        gen.insert(idx);
                    }
                }
            }

            gen_sets.push(gen);
        }

        Self { num_vars, gen_sets }
    }

    /// Returns the number of variables being tracked.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_vars
    }
}

impl<T: Target> DataFlowAnalysis<T> for ReachingDefinitions {
    type Lattice = ReachingDefsResult;
    const DIRECTION: Direction = Direction::Forward;

    fn boundary(&self, ssa: &SsaFunction<T>) -> Self::Lattice {
        // At function entry, the initial definitions of arguments and locals reach
        let mut defs = BitSet::new(self.num_vars);

        // Arguments and locals have initial definitions (version 0)
        for (idx, var) in ssa.variables().iter().enumerate() {
            if var.version() == 0 && (var.origin().is_argument() || var.origin().is_local()) {
                defs.insert(idx);
            }
        }

        ReachingDefsResult { defs }
    }

    fn initial(&self, _ssa: &SsaFunction<T>) -> Self::Lattice {
        // Initially, no definitions reach interior blocks
        ReachingDefsResult {
            defs: BitSet::new(self.num_vars),
        }
    }

    fn transfer(
        &self,
        block_id: usize,
        _block: &SsaBlock<T>,
        input: &Self::Lattice,
        _ssa: &SsaFunction<T>,
    ) -> Self::Lattice {
        // OUT = GEN ∪ IN (no KILL in SSA since each variable is defined once)
        let mut result = input.defs.clone();
        if let Some(gen) = self.gen_sets.get(block_id) {
            result.union_with(gen);
        }
        ReachingDefsResult { defs: result }
    }
}

/// Result of reaching definitions analysis for a single program point.
#[derive(Debug, Clone, PartialEq)]
pub struct ReachingDefsResult {
    /// Bit vector of reaching definitions (indexed by `SsaVarId`).
    defs: BitSet,
}

impl ReachingDefsResult {
    /// Creates a new empty result.
    #[must_use]
    pub fn new(num_vars: usize) -> Self {
        Self {
            defs: BitSet::new(num_vars),
        }
    }

    /// Returns `true` if the given variable's definition reaches this point.
    #[must_use]
    pub fn reaches(&self, var: SsaVarId) -> bool {
        let idx = var.index();
        idx < self.defs.len() && self.defs.contains(idx)
    }

    /// Returns an iterator over all reaching definitions.
    pub fn definitions(&self) -> impl Iterator<Item = SsaVarId> + '_ {
        self.defs.iter().map(SsaVarId::from_index)
    }

    /// Returns the number of reaching definitions.
    #[must_use]
    pub fn count(&self) -> usize {
        self.defs.count()
    }

    /// Returns `true` if no definitions reach this point.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    /// Adds a definition to the reaching set.
    pub fn add(&mut self, var: SsaVarId) {
        let idx = var.index();
        if idx < self.defs.len() {
            self.defs.insert(idx);
        }
    }

    /// Removes a definition from the reaching set.
    pub fn remove(&mut self, var: SsaVarId) {
        let idx = var.index();
        if idx < self.defs.len() {
            self.defs.remove(idx);
        }
    }
}

impl MeetSemiLattice for ReachingDefsResult {
    /// Meet is union (may analysis: a definition reaches if it reaches from ANY predecessor).
    fn meet(&self, other: &Self) -> Self {
        let mut result = self.defs.clone();
        result.union_with(&other.defs);
        Self { defs: result }
    }

    fn is_bottom(&self) -> bool {
        // Bottom is when all definitions reach (full set).
        self.defs.is_full()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reaching_defs_result() {
        let mut result = ReachingDefsResult::new(10);
        assert!(result.is_empty());

        result.add(SsaVarId::from_index(0));
        result.add(SsaVarId::from_index(5));

        assert!(!result.is_empty());
        assert_eq!(result.count(), 2);
        assert!(result.reaches(SsaVarId::from_index(0)));
        assert!(result.reaches(SsaVarId::from_index(5)));
        assert!(!result.reaches(SsaVarId::from_index(1)));

        result.remove(SsaVarId::from_index(0));
        assert!(!result.reaches(SsaVarId::from_index(0)));
        assert_eq!(result.count(), 1);
    }

    #[test]
    fn test_reaching_defs_meet() {
        let mut a = ReachingDefsResult::new(10);
        let mut b = ReachingDefsResult::new(10);

        a.add(SsaVarId::from_index(0));
        a.add(SsaVarId::from_index(1));
        b.add(SsaVarId::from_index(1));
        b.add(SsaVarId::from_index(2));

        let result = a.meet(&b);
        assert!(result.reaches(SsaVarId::from_index(0)));
        assert!(result.reaches(SsaVarId::from_index(1)));
        assert!(result.reaches(SsaVarId::from_index(2)));
        assert_eq!(result.count(), 3);
    }

    #[test]
    fn test_reaching_defs_iterator() {
        let mut result = ReachingDefsResult::new(100);
        result.add(SsaVarId::from_index(5));
        result.add(SsaVarId::from_index(42));
        result.add(SsaVarId::from_index(99));

        let defs: Vec<_> = result.definitions().collect();
        assert_eq!(defs.len(), 3);
        assert!(defs.contains(&SsaVarId::from_index(5)));
        assert!(defs.contains(&SsaVarId::from_index(42)));
        assert!(defs.contains(&SsaVarId::from_index(99)));
    }
}
