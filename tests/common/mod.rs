//! Shared integration-test helpers for SSA pass tests.

pub use analyssa::testing::{
    assert_mock_has_op as assert_has_op, assert_mock_no_op as assert_no_op,
    assert_mock_valid_full as assert_valid_full, assert_mock_valid_quick as assert_valid_quick,
    assert_mock_valid_standard as assert_valid_standard, mock_op_at as op_at,
    mock_terminator_at as terminator_at, run_mock_cleanup_boundary as run_cleanup_boundary,
    run_mock_malformed_cleanup_boundary as run_malformed_cleanup_boundary,
    run_mock_normalization_boundary as run_normalization_boundary,
    run_mock_pass_boundary as run_pass_boundary,
    run_mock_pass_repaired_boundary as run_pass_repaired_boundary,
};

use analyssa::{
    events::{EventKind, EventLog},
    testing::MockTarget,
};

/// Runs a pass expected to change the function and validates both boundaries.
pub fn assert_pass_changes<F>(ssa: &mut analyssa::ir::SsaFunction<MockTarget>, label: &str, run: F)
where
    F: FnOnce(&mut analyssa::ir::SsaFunction<MockTarget>) -> bool,
{
    assert!(run_pass_boundary(ssa, label, run), "{label} should change");
}

/// Runs a pass expected to leave the function unchanged and validates both boundaries.
pub fn assert_pass_noop<F>(ssa: &mut analyssa::ir::SsaFunction<MockTarget>, label: &str, run: F)
where
    F: FnOnce(&mut analyssa::ir::SsaFunction<MockTarget>) -> bool,
{
    assert!(
        !run_pass_boundary(ssa, label, run),
        "{label} should not change"
    );
}

/// Asserts that the log contains an event kind.
pub fn assert_event(log: &EventLog<MockTarget>, kind: EventKind) {
    assert!(log.has(kind), "expected event kind {kind:?}");
}

#[cfg(test)]
mod tests {
    use analyssa::{
        events::{EventKind, EventLog},
        ir::ops::SsaOp,
        testing::{const_i32_return, MockTarget},
    };

    use super::{
        assert_event, assert_has_op, assert_no_op, assert_pass_changes, assert_pass_noop,
        assert_valid_full, assert_valid_quick, assert_valid_standard, op_at, run_cleanup_boundary,
        run_malformed_cleanup_boundary, run_normalization_boundary, run_pass_repaired_boundary,
        terminator_at,
    };

    #[test]
    fn shared_pass_harness_helpers_cover_common_assertions() {
        let mut ssa = const_i32_return(1);
        assert_valid_standard(&ssa, "standard smoke");
        assert_valid_full(&ssa, "full smoke");
        assert_valid_quick(&ssa, "quick smoke");
        assert!(matches!(op_at(&ssa, 0, 0), SsaOp::Const { .. }));
        assert!(matches!(terminator_at(&ssa, 0), SsaOp::Return { .. }));
        assert_has_op(&ssa, "const op", |op| matches!(op, SsaOp::Const { .. }));
        assert_no_op(&ssa, "native opaque op", |op| {
            matches!(op, SsaOp::NativeOpaque(_))
        });

        assert_pass_noop(&mut ssa, "no-op closure", |_| false);
        assert_pass_changes(&mut ssa, "synthetic recompute change", |ssa| {
            ssa.recompute_uses();
            true
        });
        assert!(!run_pass_repaired_boundary(
            &mut ssa,
            "repaired boundary no-op",
            |_| false
        ));
        assert!(!run_normalization_boundary(
            &mut ssa,
            "normalization boundary no-op",
            |_| false
        ));
        assert!(!run_cleanup_boundary(
            &mut ssa,
            "cleanup boundary no-op",
            |_| false
        ));
        assert!(!run_malformed_cleanup_boundary(
            &mut ssa,
            "malformed cleanup boundary no-op",
            |_| false
        ));

        let log = EventLog::<MockTarget>::new();
        log.record(EventKind::ConstantFolded)
            .at(0u32, 0usize)
            .message("smoke");
        assert_event(&log, EventKind::ConstantFolded);
    }
}
