//! Lightweight SSA repair for passes that do NOT modify the CFG structure.
//!
//! After instruction-only passes (constant propagation, copy propagation, DCE,
//! algebraic simplification), the SSA form needs minor cleanup but NOT full
//! reconstruction. The CFG topology, dominator tree, and phi placement are all
//! still valid — only instruction-level artifacts need attention.
//!
//! [`repair_ssa`](super::SsaFunction::repair_ssa) is the fast-path alternative
//! to [`rebuild_ssa`](super::SsaFunction::rebuild_ssa) for passes classified as
//! `ModificationScope::InstructionsOnly` or `UsesOnly`.
//!
//! # What it does
//!
//! 1. **Nop stripping** — Removes `Nop` instructions, reindexes variable `DefSite`s
//! 2. **Trivial phi elimination** — Removes phis where all operands resolve to one value
//! 3. **Dead phi elimination** — Removes phis whose result has no consumers
//! 4. **Variable compaction** — Removes orphaned variables and reindexes IDs
//!
//! # What it does NOT do (saving significant overhead)
//!
//! - Recompute dominators or dominance frontiers
//! - Recompute liveness
//! - Re-place phi nodes
//! - Full variable renaming
//! - Orphan origin assignment
//!
//! # When to use vs. `rebuild_ssa`
//!
//! | Pass type | Use |
//! |-----------|-----|
//! | Instruction-only (replace opcodes, substitute uses) | `repair_ssa` |
//! | CFG-modifying (add/remove blocks, change branches) | `rebuild_ssa` |

use crate::{
    ir::function::{SsaFunction, TrivialPhiOptions},
    target::Target,
};

impl<T: Target> SsaFunction<T> {
    /// Lightweight SSA repair for passes that don't modify CFG structure.
    ///
    /// This is the fast path alternative to [`rebuild_ssa`](Self::rebuild_ssa)
    /// for passes classified as `InstructionsOnly` or `UsesOnly`. It assumes
    /// the CFG topology is unchanged and only cleans up instruction-level
    /// artifacts.
    ///
    /// # What this does
    ///
    /// 1. Strips Nop instructions and reindexes variable DefSites
    /// 2. Eliminates trivial phi nodes (all operands resolve to one value)
    /// 3. Eliminates dead phi nodes (result never used)
    /// 4. Compacts orphaned variables and reindexes IDs
    ///
    /// # When to use
    ///
    /// Use this instead of `rebuild_ssa` when the pass only:
    /// - Replaces instruction opcodes/operands
    /// - Converts instructions to Nops (for DCE)
    /// - Substitutes variable uses (copy propagation, GVN)
    ///
    /// Do NOT use this if the pass:
    /// - Adds, removes, or reorders blocks
    /// - Changes branch targets (changes predecessor lists)
    /// - Converts branches to jumps (changes CFG edges)
    pub fn repair_ssa(&mut self) {
        if self.blocks.is_empty() {
            return;
        }

        self.strip_nops();
        self.eliminate_trivial_phis(&TrivialPhiOptions { reachable: None });
        self.eliminate_dead_phis();
        self.compact_variables();
        self.reindex_variables();
    }
}
