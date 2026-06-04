//! Pass scheduler configuration tests.

mod common;

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Mutex,
};

use analyssa::{
    events::EventLog,
    host::{DirtySet, SsaStore},
    ir::{function::SsaFunction, ops::SsaOp, variable::SsaVarId},
    passes::{
        AlgebraicSimplificationPass, BlockMergingPass, ControlFlowSimplificationPass,
        CopyPropagationPass, DeadCodeEliminationPass, DeadMethodEliminationPass,
        GlobalValueNumberingPass, JumpThreadingPass, LicmPass, LoopCanonicalizationPass,
        OpaquePredicatePass, ReassociationPass, StrengthReductionPass, ValueRangePropagationPass,
    },
    scheduling::{ModificationScope, PassScheduler, SsaPass, SsaPassHost},
    testing::{self, MockTarget},
    world::World,
    Error, PipelineConfig, Result,
};

use common::assert_valid_full;

fn pass_name(pass: &dyn SsaPass<MockTarget, MockHost>) -> &'static str {
    pass.name()
}

fn pass_scope(pass: &dyn SsaPass<MockTarget, MockHost>) -> ModificationScope {
    pass.modification_scope()
}

fn pass_repairs_ssa(pass: &dyn SsaPass<MockTarget, MockHost>) -> bool {
    pass.repairs_ssa()
}

fn pass_is_global(pass: &dyn SsaPass<MockTarget, MockHost>) -> bool {
    pass.is_global()
}

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| std::process::abort())
}

fn result_or_abort<T>(result: Result<T>) -> T {
    result.unwrap_or_else(|_| std::process::abort())
}

fn err_or_abort<T>(result: Result<T>) -> Error {
    match result {
        Ok(_) => std::process::abort(),
        Err(error) => error,
    }
}

#[derive(Default)]
struct MockHost {
    ssa: Mutex<BTreeMap<u32, SsaFunction<MockTarget>>>,
    dirty: Mutex<BTreeSet<u32>>,
    processed: Mutex<BTreeSet<u32>>,
    events: EventLog<MockTarget>,
}

impl World<MockTarget> for MockHost {
    fn all_methods(&self) -> Vec<u32> {
        self.iter_methods()
    }

    fn entry_points(&self) -> Vec<u32> {
        self.iter_methods()
    }

    fn callees(&self, _method: &u32) -> Vec<u32> {
        Vec::new()
    }

    fn is_dead(&self, _method: &u32) -> bool {
        false
    }

    fn mark_dead(&self, _method: &u32) {}
}

impl SsaStore<MockTarget> for MockHost {
    fn contains(&self, method: &u32) -> bool {
        self.ssa
            .lock()
            .unwrap_or_else(|_| std::process::abort())
            .contains_key(method)
    }

    fn take_ssa(&self, method: &u32) -> Option<SsaFunction<MockTarget>> {
        self.ssa
            .lock()
            .unwrap_or_else(|_| std::process::abort())
            .remove(method)
    }

    fn insert_ssa(&self, method: u32, ssa: SsaFunction<MockTarget>) {
        self.ssa
            .lock()
            .unwrap_or_else(|_| std::process::abort())
            .insert(method, ssa);
    }

    fn clone_ssa(&self, method: &u32) -> Option<SsaFunction<MockTarget>> {
        self.ssa
            .lock()
            .unwrap_or_else(|_| std::process::abort())
            .get(method)
            .cloned()
    }

    fn iter_methods(&self) -> Vec<u32> {
        self.ssa
            .lock()
            .unwrap_or_else(|_| std::process::abort())
            .keys()
            .copied()
            .collect()
    }
}

impl DirtySet<MockTarget> for MockHost {
    fn mark_dirty(&self, method: &u32) {
        self.dirty
            .lock()
            .unwrap_or_else(|_| std::process::abort())
            .insert(*method);
    }

    fn is_dirty(&self, method: &u32) -> bool {
        self.dirty
            .lock()
            .unwrap_or_else(|_| std::process::abort())
            .contains(method)
    }

    fn dirty_snapshot(&self) -> Vec<u32> {
        self.dirty
            .lock()
            .unwrap_or_else(|_| std::process::abort())
            .iter()
            .copied()
            .collect()
    }

    fn clear_dirty_for(&self, method: &u32) {
        self.dirty
            .lock()
            .unwrap_or_else(|_| std::process::abort())
            .remove(method);
    }

    fn mark_processed(&self, method: &u32) {
        self.processed
            .lock()
            .unwrap_or_else(|_| std::process::abort())
            .insert(*method);
    }

    fn is_processed(&self, method: &u32) -> bool {
        self.processed
            .lock()
            .unwrap_or_else(|_| std::process::abort())
            .contains(method)
    }
}

impl SsaPassHost<MockTarget> for MockHost {
    fn events(&self) -> &EventLog<MockTarget> {
        &self.events
    }
}

struct BreakingPass;

impl SsaPass<MockTarget, MockHost> for BreakingPass {
    fn name(&self) -> &'static str {
        "breaking"
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        ssa.replace_instruction_op(
            0,
            1,
            SsaOp::Return {
                value: Some(SsaVarId::from_index(99)),
            },
        );
        Ok(true)
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }
}

struct FailingPass;

impl SsaPass<MockTarget, MockHost> for FailingPass {
    fn name(&self) -> &'static str {
        "failing"
    }

    fn run_on_method(
        &self,
        _ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        Err(Error::new("pass failed"))
    }
}

#[test]
fn default_pipeline_config_registers_all_builtin_passes() {
    let config = PipelineConfig::default();
    let scheduler = PassScheduler::<MockTarget, MockHost>::new(config);

    assert_eq!(scheduler.normalize_count(), 3);
    assert_eq!(scheduler.pass_count(), 11);

    let without_global = PipelineConfig {
        include_dead_method_elimination: false,
        ..PipelineConfig::default()
    };
    let scheduler = PassScheduler::<MockTarget, MockHost>::new(without_global);

    assert_eq!(scheduler.normalize_count(), 3);
    assert_eq!(scheduler.pass_count(), 10);
}

#[test]
fn verify_hard_reports_invalid_pass_output_and_rolls_back() {
    let host = MockHost::default();
    let method = 1u32;
    let original = testing::const_i32_return(7);
    host.insert_ssa(method, original.clone());

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        verify_hard: true,
        max_iterations: 1,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::new(config);
    scheduler.add_at_layer(Box::new(BreakingPass), 0);

    let error = err_or_abort(scheduler.run_pipeline(&host));

    assert!(error.to_string().contains("breaking"));
    assert!(error.to_string().contains("invalid SSA"));
    let stored = some_or_abort(host.clone_ssa(&method));
    assert_eq!(format!("{stored}"), format!("{original}"));
}

#[test]
fn scheduler_propagates_pass_errors() {
    let host = MockHost::default();
    let method = 1u32;
    host.insert_ssa(method, testing::const_i32_return(7));

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        max_iterations: 1,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::new(config);
    scheduler.add_at_layer(Box::new(FailingPass), 0);

    let error = err_or_abort(scheduler.run_pipeline(&host));

    assert!(error.to_string().contains("failing"));
    assert!(error.to_string().contains("pass failed"));
}

#[test]
fn default_scheduler_verify_hard_handles_mixed_builder_fixtures() {
    let host = MockHost::default();
    host.insert_ssa(1, testing::scalar_rewrite_fixture());
    host.insert_ssa(2, testing::diamond_phi_fixture());
    host.insert_ssa(3, testing::memory_effect_fixture());
    host.insert_ssa(4, testing::native_effect_fixture());
    host.insert_ssa(5, testing::vector_simd_fixture());

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        verify_hard: true,
        max_iterations: 2,
        max_phase_iterations: 3,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::new(config);

    let changes = result_or_abort(scheduler.run_pipeline(&host));
    assert!(changes > 0, "mixed fixtures should expose scheduler work");

    for method in host.iter_methods() {
        let ssa = some_or_abort(host.clone_ssa(&method));
        assert_valid_full(&ssa, &format!("method {method} after scheduler"));
    }
}

#[test]
fn built_in_pass_wrappers_report_expected_metadata() {
    let instructions_only = ModificationScope::InstructionsOnly;
    let uses_only = ModificationScope::UsesOnly;
    let cfg = ModificationScope::CfgModifying;

    let algebraic = AlgebraicSimplificationPass::new();
    assert_eq!(pass_name(&algebraic), "algebraic-simplification");
    assert_eq!(pass_scope(&algebraic), instructions_only);
    assert!(pass_repairs_ssa(&algebraic));
    assert!(!pass_is_global(&algebraic));

    let blockmerge = BlockMergingPass::new(3);
    assert_eq!(pass_name(&blockmerge), "block-merging");
    assert_eq!(pass_scope(&blockmerge), cfg);
    assert!(pass_repairs_ssa(&blockmerge));
    assert_eq!(blockmerge.max_iterations, 3);

    let controlflow = ControlFlowSimplificationPass::new(4);
    assert_eq!(pass_name(&controlflow), "control-flow-simplification");
    assert_eq!(pass_scope(&controlflow), cfg);
    assert!(pass_repairs_ssa(&controlflow));
    assert_eq!(controlflow.max_iterations, 4);

    let copying = CopyPropagationPass::new(5);
    assert_eq!(pass_name(&copying), "copy-propagation");
    assert_eq!(pass_scope(&copying), instructions_only);
    assert!(pass_repairs_ssa(&copying));
    assert_eq!(copying.max_iterations, 5);

    let dce = DeadCodeEliminationPass::new(6);
    assert_eq!(pass_name(&dce), "dead-code-elimination");
    assert_eq!(pass_scope(&dce), instructions_only);
    assert_eq!(dce.max_iterations, 6);

    let global_dce = DeadMethodEliminationPass;
    assert_eq!(pass_name(&global_dce), "dead-method-elimination");
    assert!(pass_is_global(&global_dce));

    let gvn = GlobalValueNumberingPass::new();
    assert_eq!(pass_name(&gvn), "global-value-numbering");
    assert_eq!(pass_scope(&gvn), uses_only);

    let licm = LicmPass::new();
    assert_eq!(pass_name(&licm), "licm");
    assert_eq!(pass_scope(&licm), cfg);
    assert!(pass_repairs_ssa(&licm));

    let loopcanon = LoopCanonicalizationPass::new();
    assert_eq!(pass_name(&loopcanon), "loop-canonicalization");
    assert_eq!(pass_scope(&loopcanon), cfg);
    assert!(pass_repairs_ssa(&loopcanon));

    let predicates = OpaquePredicatePass::<MockTarget>::new();
    assert_eq!(pass_name(&predicates), "opaque-predicate");
    assert_eq!(pass_scope(&predicates), cfg);
    assert!(pass_repairs_ssa(&predicates));

    let ranges = ValueRangePropagationPass::new(7);
    assert_eq!(pass_name(&ranges), "value-range-propagation");
    assert_eq!(pass_scope(&ranges), cfg);
    assert!(pass_repairs_ssa(&ranges));
    assert_eq!(ranges.max_iterations, 7);

    let reassociation = ReassociationPass::new();
    assert_eq!(pass_name(&reassociation), "reassociation");
    assert_eq!(pass_scope(&reassociation), instructions_only);
    assert!(pass_repairs_ssa(&reassociation));

    let strength = StrengthReductionPass;
    assert_eq!(pass_name(&strength), "strength-reduction");
    assert_eq!(pass_scope(&strength), instructions_only);
    assert!(pass_repairs_ssa(&strength));

    let threading = JumpThreadingPass::new();
    assert_eq!(pass_name(&threading), "jump-threading");
    assert_eq!(pass_scope(&threading), cfg);
    assert!(pass_repairs_ssa(&threading));
}
