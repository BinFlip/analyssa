//! Built-in SSA optimization passes for the analyssa deobfuscation framework.
//!
//! Each pass exposes a free function, typically named `run`, that takes
//! `&mut SsaFunction<T>` plus an event sink and mutates the IR in place.
//! Hosts can call those functions directly or use the pass wrappers re-exported
//! by this module with [`crate::scheduling::PassScheduler`].
//!
//! # Pass Categories
//!
//! - **Normalization**: [`deadcode`], [`copying`], [`gvn`], [`blockmerge`] —
//!   clean up the IR after structural transformations. Run between every
//!   pipeline layer's fixpoint iterations.
//! - **Structural**: [`controlflow`], [`threading`], [`loopcanon`], [`blockmerge`] —
//!   simplify and canonicalize the control-flow graph.
//! - **Value-level**: [`algebraic`], [`strength`], [`reassociate`], [`ranges`],
//!   [`predicates`], [`licm`] — replace expensive or redundant computations
//!   with cheaper or constant equivalents.
//!
//! # Scheduling
//!
//! The `scheduling` sub-module wraps each pass body in an
//! [`SsaPass`](crate::scheduling::SsaPass) trait impl so the
//! [`PassScheduler`](crate::scheduling::PassScheduler) can orchestrate them
//! with capability-based ordering, fixpoint iteration, and parallel dispatch.

pub mod algebraic;
pub mod blockmerge;
pub mod controlflow;
pub mod copying;
pub mod deadcode;
pub mod gvn;
pub mod licm;
pub mod loopcanon;
pub mod predicates;
pub mod ranges;
pub mod reassociate;
mod scheduling;
pub mod strength;
pub mod threading;
pub mod utils;

pub use predicates::PredicateResult;
pub use scheduling::{
    AlgebraicSimplificationPass, BlockMergingPass, ControlFlowSimplificationPass,
    CopyPropagationPass, DeadCodeEliminationPass, DeadMethodEliminationPass,
    GlobalValueNumberingPass, JumpThreadingPass, LicmPass, LoopCanonicalizationPass,
    OpaquePredicatePass, ReassociationPass, StrengthReductionPass, ValueRangePropagationPass,
};

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    #[test]
    fn pass_implementations_do_not_bypass_editor_api() {
        let pass_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/passes");
        let forbidden = [
            "ssa.replace_uses_checked(",
            "ssa.propagate_copies(",
            "ssa.nop_copy_defining(",
            "ssa.replace_instruction_op(",
            "ssa.remove_instruction(",
            "ssa.repair_ssa(",
            "ssa.rebuild_ssa(",
            "ssa.strip_nops(",
            "ssa.compact_variables(",
            "ssa.block_mut(",
            "ssa.blocks_mut(",
            "ssa.variables_mut(",
        ];
        let mut violations = Vec::new();
        for entry in fs::read_dir(&pass_dir).expect("read pass source directory") {
            let entry = entry.expect("read pass source entry");
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            let source = fs::read_to_string(&path).expect("read pass source file");
            let production = source.split("#[cfg(test)]").next().unwrap_or(&source);
            for (line_idx, line) in production.lines().enumerate() {
                for pattern in forbidden {
                    if line.contains(pattern) {
                        violations.push(format!(
                            "{}:{} uses {pattern}",
                            path.display(),
                            line_idx + 1
                        ));
                    }
                }
            }
        }

        assert!(
            violations.is_empty(),
            "SSA passes must mutate through SsaFunction::edit/SsaEditor only:\n{}",
            violations.join("\n")
        );
    }
}
