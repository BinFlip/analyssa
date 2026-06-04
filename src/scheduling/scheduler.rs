//! [`PassScheduler`] — the core engine that orchestrates [`SsaPass`]
//! execution across methods.
//!
//! The scheduler organizes registered passes into execution layers
//! computed from capability dependencies, runs each layer to fixpoint
//! with normalize passes interleaved, dispatches per-method work in
//! parallel via rayon, and applies modification-scope-driven SSA repair
//! after each pass.
//!
//! # Layer computation
//!
//! Passes that do not declare capabilities fall back to their
//! [`SsaPass::fallback_layer`]. Passes with declared capabilities are
//! topologically sorted: if pass B requires a capability that pass A
//! provides, B is placed in a strictly later layer than A.
//!
//! # Normalization
//!
//! Normalize passes (DCE, GVN, copy propagation, etc.) are separate from
//! the layered passes. They run between every layer's fixpoint iterations,
//! cleaning up after each round of structural changes to expose new
//! optimization opportunities.
//!
//! # Dirty tracking
//!
//! The scheduler seeds the dirty set with all methods at pipeline start.
//! After each fixpoint iteration, the dirty set is updated: methods
//! modified during the iteration remain dirty; unmodified methods are
//! cleared. Passes that declare `requires_full_scan()` bypass dirty
//! tracking entirely.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex,
    },
};

use dashmap::DashSet;
use log::debug;
use rayon::prelude::*;

use crate::{
    analysis::verifier::{SsaVerifier, VerifyLevel},
    error::{Error, Result},
    events::EventKind,
    graph::IndexedGraph,
    ir::function::SsaFunction,
    passes::{
        AlgebraicSimplificationPass, BlockMergingPass, ControlFlowSimplificationPass,
        CopyPropagationPass, DeadCodeEliminationPass, DeadMethodEliminationPass,
        GlobalValueNumberingPass, JumpThreadingPass, LicmPass, LoopCanonicalizationPass,
        OpaquePredicatePass, ReassociationPass, StrengthReductionPass, ValueRangePropagationPass,
    },
    scheduling::pass::{ModificationScope, SsaPass, SsaPassHost},
    target::Target,
};

/// A registered pass paired with its assigned fallback layer number.
type LayeredPass<T, H> = (Box<dyn SsaPass<T, H>>, usize);

/// Built-in pass-pipeline configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineConfig {
    /// Maximum iterations for the whole scheduler pipeline.
    pub max_iterations: usize,
    /// Number of stable outer iterations required before stopping.
    pub stable_iterations: usize,
    /// Maximum fixpoint iterations for one scheduler layer.
    pub max_phase_iterations: usize,
    /// Maximum fixpoint iterations for block merging.
    pub block_merge_iterations: usize,
    /// Maximum fixpoint iterations for control-flow simplification.
    pub control_flow_iterations: usize,
    /// Maximum fixpoint iterations for copy propagation.
    pub copy_iterations: usize,
    /// Maximum fixpoint iterations for dead-code elimination.
    pub dead_code_iterations: usize,
    /// Maximum fixpoint iterations for value-range propagation.
    pub range_iterations: usize,
    /// Include interprocedural dead-method elimination in scheduler presets.
    pub include_dead_method_elimination: bool,
    /// Verify each method after every pass and fail fast on invalid SSA.
    pub verify_hard: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            stable_iterations: 1,
            max_phase_iterations: 10,
            block_merge_iterations: 10,
            control_flow_iterations: 10,
            copy_iterations: 10,
            dead_code_iterations: 50,
            range_iterations: 20,
            include_dead_method_elimination: true,
            verify_hard: false,
        }
    }
}

/// Orchestrates SSA pass execution using capability-based scheduling.
///
/// Passes are organized into execution layers computed from their declared
/// capabilities. Each layer runs all its passes to fixpoint with
/// normalization between iterations. The entire pipeline repeats until
/// global fixpoint or `max_iterations`.
///
/// # Type parameters
///
/// * `T` — The host's target type.
/// * `H` — The host adapter (must implement [`SsaPassHost<T>`]).
///   Typically a single host type implements [`crate::world::World`],
///   [`crate::host::SsaStore`], [`crate::host::DirtySet`], and
///   [`SsaPassHost<T>`] together.
pub struct PassScheduler<T, H>
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    /// Maximum iterations for the entire pipeline before stopping.
    max_iterations: usize,
    /// Stop early if no changes for this many consecutive iterations.
    stable_iterations: usize,
    /// Maximum fixpoint iterations for a single layer before moving on.
    max_phase_iterations: usize,
    /// Verify each method after every pass and return an error on invalid SSA.
    verify_hard: bool,
    /// All non-normalize passes paired with their fallback layer number.
    passes: Vec<LayeredPass<T, H>>,
    /// Normalization passes. Run between every layer's fixpoint iterations.
    normalize: Vec<Box<dyn SsaPass<T, H>>>,
    /// Phantom marker; the host type appears only in trait bounds at the
    /// `run_pipeline` entry point, but we record it here to keep the
    /// scheduler instance keyed to a specific host.
    _host: std::marker::PhantomData<fn(&H)>,
}

impl<T, H> PassScheduler<T, H>
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    /// Creates a scheduler that registers no built-in passes.
    ///
    /// Only the iteration limits and `verify_hard` flag from `config` are
    /// applied; the pass-tuning fields (e.g. `copy_iterations`,
    /// `include_dead_method_elimination`) are ignored because no built-in
    /// passes are added. Hosts that organize their own pipeline — adding
    /// every pass explicitly via [`Self::add`], [`Self::add_at_layer`], or
    /// [`Self::add_normalize`] — should start here instead of [`Self::new`]
    /// to avoid running the default pipeline alongside their own passes.
    #[must_use]
    pub fn empty(config: PipelineConfig) -> Self {
        Self {
            max_iterations: config.max_iterations,
            stable_iterations: config.stable_iterations,
            max_phase_iterations: config.max_phase_iterations,
            verify_hard: config.verify_hard,
            passes: Vec::new(),
            normalize: Vec::new(),
            _host: std::marker::PhantomData,
        }
    }

    /// Creates a scheduler from a pipeline configuration.
    ///
    /// The default configuration registers all built-in scheduler-backed
    /// passes in a deterministic order. Value-cleanup passes are registered
    /// as normalize passes so they run between layered fixpoint iterations.
    /// Structural and value/enhancement passes are registered at their
    /// conventional fallback layers. Interprocedural dead-method elimination
    /// is included by default and can be disabled with
    /// [`PipelineConfig::include_dead_method_elimination`].
    ///
    /// Hosts still provide storage and program context through
    /// [`SsaPassHost`], [`crate::host::SsaStore`], [`crate::host::DirtySet`],
    /// and [`crate::world::World`]; they do not need to hand-roll the pass
    /// order. Hosts that supply their own complete pipeline should use
    /// [`Self::empty`] instead.
    #[must_use]
    pub fn new(config: PipelineConfig) -> Self
    where
        T: Send + Sync,
    {
        let mut scheduler = Self::empty(config);

        scheduler.add_normalize(Box::new(GlobalValueNumberingPass::new()));
        scheduler.add_normalize(Box::new(CopyPropagationPass::new(config.copy_iterations)));
        scheduler.add_normalize(Box::new(DeadCodeEliminationPass::new(
            config.dead_code_iterations,
        )));

        if config.include_dead_method_elimination {
            scheduler.add(Box::new(DeadMethodEliminationPass));
        }

        scheduler.add(Box::new(ControlFlowSimplificationPass::new(
            config.control_flow_iterations,
        )));
        scheduler.add(Box::new(JumpThreadingPass::new()));
        scheduler.add(Box::new(LoopCanonicalizationPass::new()));
        scheduler.add(Box::new(BlockMergingPass::new(
            config.block_merge_iterations,
        )));
        scheduler.add(Box::new(AlgebraicSimplificationPass::new()));
        scheduler.add(Box::new(ReassociationPass::new()));
        scheduler.add(Box::new(StrengthReductionPass));
        scheduler.add(Box::new(ValueRangePropagationPass::new(
            config.range_iterations,
        )));
        scheduler.add(Box::new(OpaquePredicatePass::<T>::new()));
        scheduler.add(Box::new(LicmPass::new()));

        scheduler
    }

    /// Returns the number of non-normalize (layered) passes registered.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Returns the number of normalization passes registered.
    #[must_use]
    pub fn normalize_count(&self) -> usize {
        self.normalize.len()
    }

    /// Register a layered pass using its own `fallback_layer()` return
    /// value.
    pub fn add(&mut self, pass: Box<dyn SsaPass<T, H>>) {
        let layer = pass.fallback_layer();
        self.passes.push((pass, layer));
    }

    /// Register a layered pass at an explicit fallback layer, overriding
    /// the pass's own `fallback_layer()`.
    ///
    /// Useful for hosts that organize passes via a phase enum (e.g.,
    /// CIL: Structure=0, Value=1, Simplify=2, Inline=3) rather than
    /// per-pass-type defaults.
    pub fn add_at_layer(&mut self, pass: Box<dyn SsaPass<T, H>>, layer: usize) {
        self.passes.push((pass, layer));
    }

    /// Register a normalization pass.
    ///
    /// Normalize passes run between every layer's fixpoint iterations
    /// and are excluded from the capability dependency graph.
    pub fn add_normalize(&mut self, pass: Box<dyn SsaPass<T, H>>) {
        self.normalize.push(pass);
    }

    /// Compute execution layer assignments from capability dependencies.
    ///
    /// Uses Bellman-Ford-style relaxation: each pass starts at its
    /// fallback layer; if pass A provides a capability that pass B
    /// requires, B is pushed to at least `layer(A) + 1`.
    ///
    /// # Returns
    ///
    /// A `Vec<usize>` where element `i` is the computed layer number
    /// for `self.passes[i]`.
    ///
    /// # Errors
    ///
    /// Returns an error if a cycle is detected in the capability
    /// dependency graph.
    fn compute_layer_assignment(&self) -> Result<Vec<usize>> {
        let n = self.passes.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // Build a map from capability to provider indices.
        let mut providers: HashMap<T::Capability, Vec<usize>> = HashMap::new();
        for (i, (pass, _)) in self.passes.iter().enumerate() {
            for cap in pass.provides() {
                providers.entry(*cap).or_default().push(i);
            }
        }

        // Build a dependency graph: edge provider -> dependent.
        let mut graph: IndexedGraph<usize, ()> = IndexedGraph::with_capacity(n, n);
        for i in 0..n {
            graph.add_node(i);
        }

        let mut deps: Vec<Vec<usize>> = vec![vec![]; n];
        for (i, (pass, _)) in self.passes.iter().enumerate() {
            for cap in pass.requires() {
                if let Some(provider_indices) = providers.get(cap) {
                    for &j in provider_indices {
                        if j != i {
                            if let Some(slot) = deps.get_mut(i) {
                                slot.push(j);
                            }
                            let _ = graph.add_edge(j, i, ());
                        }
                    }
                }
            }
        }

        // Check for cycles via topological sort.
        if graph.topological_sort().is_none() {
            if let Some(cycle) = graph.find_any_cycle() {
                let names: Vec<&str> = cycle
                    .iter()
                    .filter_map(|&i| self.passes.get(i).map(|p| p.0.name()))
                    .collect();
                return Err(Error::new(format!(
                    "Cycle detected in pass capability dependencies: {}",
                    names.join(" → ")
                )));
            }
            return Err(Error::new("Cycle detected in pass capability dependencies"));
        }

        // Bellman-Ford relaxation: push layers forward to satisfy deps.
        let mut layer: Vec<usize> = self.passes.iter().map(|(_, fallback)| *fallback).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..n {
                // `deps` is read-only here and `layer` is a separate buffer, so
                // no clone is needed to satisfy the borrow checker.
                let Some(dep_list) = deps.get(i) else {
                    continue;
                };
                for &dep in dep_list {
                    let layer_i = layer.get(i).copied().unwrap_or(0);
                    let layer_dep = layer.get(dep).copied().unwrap_or(0);
                    if layer_i <= layer_dep {
                        if let Some(slot) = layer.get_mut(i) {
                            *slot = layer_dep.saturating_add(1);
                        }
                        changed = true;
                    }
                }
            }
        }

        // Log layer assignments if any were moved from fallback.
        if !deps.iter().all(Vec::is_empty) {
            let max_layer = layer.iter().copied().max().unwrap_or(0);
            debug!(
                "Capability scheduling: {} passes across {} layers",
                n,
                max_layer.saturating_add(1)
            );
            for (i, (pass, fallback)) in self.passes.iter().enumerate() {
                let layer_i = layer.get(i).copied().unwrap_or(*fallback);
                if layer_i != *fallback {
                    debug!(
                        "  pass '{}': layer {} (moved from fallback {})",
                        pass.name(),
                        layer_i,
                        fallback
                    );
                }
            }
        }

        Ok(layer)
    }

    /// Run the complete pipeline across all layers.
    ///
    /// Seeds the dirty set with all methods, then iterates layers until
    /// convergence or `max_iterations`.
    ///
    /// # Arguments
    ///
    /// * `host` — The host adapter providing method storage, dirty
    ///   tracking, and event recording.
    ///
    /// # Returns
    ///
    /// The number of outer iterations completed.
    ///
    /// # Errors
    ///
    /// Returns an error if a cycle is detected in the capability
    /// dependency graph or if any pass fails.
    pub fn run_pipeline(&mut self, host: &H) -> Result<usize> {
        // Seed the dirty set with all methods that have SSA. On the
        // first iteration every method qualifies. Hosts that pre-seed
        // their own dirty set (typically by marking newly-built methods
        // as dirty when SSA is constructed) are unaffected — `mark_dirty`
        // is idempotent.
        for method in host.iter_methods() {
            host.mark_dirty(&method);
        }

        let layer_assignment = self.compute_layer_assignment()?;

        let num_layers = layer_assignment
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m.saturating_add(1));
        let mut layer_indices: Vec<Vec<usize>> = vec![vec![]; num_layers];
        for (i, &layer) in layer_assignment.iter().enumerate() {
            if let Some(slot) = layer_indices.get_mut(layer) {
                slot.push(i);
            }
        }
        layer_indices.retain(|layer| !layer.is_empty());

        let mut stable_count: usize = 0;
        let mut iterations: usize = 0;
        let max_phase = self.max_phase_iterations;
        let max_iterations = self.max_iterations;
        let stable_iterations = self.stable_iterations;
        let verify_hard = self.verify_hard;

        for iteration in 0..max_iterations {
            iterations = iteration.saturating_add(1);
            debug!("Pipeline iteration {}/{}", iterations, max_iterations);

            let iteration_modified: DashSet<T::MethodRef> = DashSet::new();
            let mut iteration_changed = false;

            for layer in &layer_indices {
                if Self::layer_to_fixpoint(
                    host,
                    &mut self.passes,
                    layer,
                    &mut self.normalize,
                    max_phase,
                    &iteration_modified,
                    verify_hard,
                )? {
                    iteration_changed = true;
                }
            }

            // Ensure normalize runs at least once on iteration 0 even if no
            // layer pass made changes.
            if iteration == 0 && !iteration_changed && !self.normalize.is_empty() {
                iteration_changed = Self::normalize_to_fixpoint(
                    host,
                    &mut self.normalize,
                    max_phase,
                    &iteration_modified,
                    verify_hard,
                )?;
            }

            // Update dirty/stable tracking at iteration boundary.
            if iteration_changed {
                let dirty = host.dirty_snapshot();
                for m in dirty {
                    if !iteration_modified.contains(&m) {
                        host.clear_dirty_for(&m);
                    }
                }
                for entry in iteration_modified.iter() {
                    host.mark_dirty(&entry);
                }
            } else {
                let dirty = host.dirty_snapshot();
                for m in dirty {
                    host.clear_dirty_for(&m);
                }
            }

            if iteration_changed {
                stable_count = 0;
            } else {
                stable_count = stable_count.saturating_add(1);
                if stable_count >= stable_iterations {
                    debug!("Pipeline stable after {} iterations", iterations);
                    break;
                }
            }
        }

        Ok(iterations)
    }

    /// Run normalize passes to fixpoint.
    fn normalize_to_fixpoint(
        host: &H,
        passes: &mut [Box<dyn SsaPass<T, H>>],
        max_phase_iterations: usize,
        iteration_modified: &DashSet<T::MethodRef>,
        verify_hard: bool,
    ) -> Result<bool> {
        let mut any_changed = false;
        for _ in 0..max_phase_iterations {
            let changed = Self::run_passes_once(host, passes, iteration_modified, verify_hard)?;
            if !changed {
                break;
            }
            any_changed = true;
        }
        Ok(any_changed)
    }

    /// Run all passes in a single layer to fixpoint, running normalize
    /// passes between iterations.
    fn layer_to_fixpoint(
        host: &H,
        all_passes: &mut [LayeredPass<T, H>],
        layer_indices: &[usize],
        normalize_passes: &mut [Box<dyn SsaPass<T, H>>],
        max_phase_iterations: usize,
        iteration_modified: &DashSet<T::MethodRef>,
        verify_hard: bool,
    ) -> Result<bool> {
        if layer_indices.is_empty() {
            return Ok(false);
        }

        let mut phase_changed = false;

        for _ in 0..max_phase_iterations {
            let pass_changed = Self::run_layer_passes_once(
                host,
                all_passes,
                layer_indices,
                iteration_modified,
                verify_hard,
            )?;

            if !pass_changed {
                if phase_changed && !normalize_passes.is_empty() {
                    Self::normalize_to_fixpoint(
                        host,
                        normalize_passes,
                        max_phase_iterations,
                        iteration_modified,
                        verify_hard,
                    )?;
                }
                break;
            }

            phase_changed = true;

            if !normalize_passes.is_empty() {
                Self::normalize_to_fixpoint(
                    host,
                    normalize_passes,
                    max_phase_iterations,
                    iteration_modified,
                    verify_hard,
                )?;
            }
        }

        Ok(phase_changed)
    }

    /// Run all normalize passes once across all eligible methods.
    fn run_passes_once(
        host: &H,
        passes: &mut [Box<dyn SsaPass<T, H>>],
        iteration_modified: &DashSet<T::MethodRef>,
        verify_hard: bool,
    ) -> Result<bool> {
        for pass in passes.iter_mut() {
            pass.initialize(host)?;
        }

        let all_methods = Self::method_order(host, false);
        let dirty_methods = Self::method_order(host, true);
        let any_changed = AtomicBool::new(false);

        for pass in passes.iter() {
            if pass.is_global() && pass.run_global(host)? {
                any_changed.store(true, Ordering::Relaxed);
            }
        }

        for pass in passes.iter() {
            if pass.is_global() {
                continue;
            }
            let methods = if pass.requires_full_scan() {
                &all_methods
            } else {
                &dirty_methods
            };
            Self::run_single_pass(
                pass.as_ref(),
                host,
                methods,
                &any_changed,
                iteration_modified,
                verify_hard,
            )?;
        }

        for pass in passes.iter_mut() {
            pass.finalize(host)?;
        }

        Ok(any_changed.load(Ordering::Relaxed))
    }

    /// Run all passes in a specific layer once across eligible methods.
    fn run_layer_passes_once(
        host: &H,
        all_passes: &mut [LayeredPass<T, H>],
        indices: &[usize],
        iteration_modified: &DashSet<T::MethodRef>,
        verify_hard: bool,
    ) -> Result<bool> {
        for &idx in indices {
            let pass_entry = all_passes
                .get_mut(idx)
                .ok_or_else(|| Error::new(format!("scheduler: pass index {idx} out of bounds")))?;
            pass_entry.0.initialize(host)?;
        }

        let all_methods = Self::method_order(host, false);
        let dirty_methods = Self::method_order(host, true);
        let any_changed = AtomicBool::new(false);

        for &idx in indices {
            let pass_entry = all_passes
                .get(idx)
                .ok_or_else(|| Error::new(format!("scheduler: pass index {idx} out of bounds")))?;
            let pass = &pass_entry.0;
            if pass.is_global() && pass.run_global(host)? {
                any_changed.store(true, Ordering::Relaxed);
            }
        }

        for &idx in indices {
            let pass_entry = all_passes
                .get(idx)
                .ok_or_else(|| Error::new(format!("scheduler: pass index {idx} out of bounds")))?;
            let pass = &pass_entry.0;
            if pass.is_global() {
                continue;
            }
            let methods = if pass.requires_full_scan() {
                &all_methods
            } else {
                &dirty_methods
            };
            Self::run_single_pass(
                pass.as_ref(),
                host,
                methods,
                &any_changed,
                iteration_modified,
                verify_hard,
            )?;
        }

        for &idx in indices {
            let pass_entry = all_passes
                .get_mut(idx)
                .ok_or_else(|| Error::new(format!("scheduler: pass index {idx} out of bounds")))?;
            pass_entry.0.finalize(host)?;
        }

        Ok(any_changed.load(Ordering::Relaxed))
    }

    /// Determine method processing order.
    ///
    /// When `dirty_only` is true, returns only methods in the host's
    /// dirty set; otherwise returns all methods with SSA. In both cases,
    /// ordering follows reverse topological order if the host supplies
    /// one, or falls back to `host.iter_methods()`.
    fn method_order(host: &H, dirty_only: bool) -> Vec<T::MethodRef> {
        let topo = host.methods_reverse_topological();
        let order: Vec<_> = if topo.is_empty() {
            host.iter_methods()
        } else {
            topo
        };

        // Build the dirty set as a HashSet so membership is O(1); a `Vec`
        // made the per-method filter below O(methods * dirty).
        let dirty_set: Option<HashSet<T::MethodRef>> = if dirty_only {
            Some(host.dirty_snapshot().into_iter().collect())
        } else {
            None
        };

        order
            .into_iter()
            .filter(|m| host.contains(m))
            .filter(|m| dirty_set.as_ref().is_none_or(|d| d.contains(m)))
            .collect()
    }

    /// Execute a single pass across all eligible methods in parallel.
    ///
    /// For each method: takes (or clones) the SSA from the store, runs
    /// the pass, applies SSA repair based on the pass's modification
    /// scope, reinserts the SSA, and records modification.
    fn run_single_pass(
        pass: &dyn SsaPass<T, H>,
        host: &H,
        methods: &[T::MethodRef],
        any_changed: &AtomicBool,
        iteration_modified: &DashSet<T::MethodRef>,
        verify_hard: bool,
    ) -> Result<()> {
        let event_snapshot = host.events().len();
        let pass_change_count = AtomicUsize::new(0);
        let errors = Mutex::new(Vec::<String>::new());

        // Passes that read other methods' SSA need peer SSAs to remain
        // visible during parallel execution. Clone the SSA before
        // processing so the original stays readable.
        let clone_for_visibility = pass.reads_peer_ssa();

        methods.par_iter().for_each(|method| {
            if !pass.should_run(method, host) {
                return;
            }

            let mut ssa: SsaFunction<T> = if clone_for_visibility {
                let Some(cloned) = host.clone_ssa(method) else {
                    return;
                };
                cloned
            } else {
                let Some(ssa) = host.take_ssa(method) else {
                    return;
                };
                ssa
            };

            let original = if verify_hard { Some(ssa.clone()) } else { None };
            let result = pass.run_on_method(&mut ssa, method, host);
            let changed = match result {
                Ok(changed) => changed,
                Err(error) => {
                    if let Some(snapshot) = original {
                        ssa = snapshot;
                    }
                    host.insert_ssa(method.clone(), ssa);
                    if let Ok(mut guard) = errors.lock() {
                        guard.push(format!(
                            "pass '{}' failed for method {:?}: {}",
                            pass.name(),
                            method,
                            error
                        ));
                    }
                    return;
                }
            };

            if changed && !pass.repairs_ssa() {
                match pass.modification_scope() {
                    ModificationScope::UsesOnly | ModificationScope::InstructionsOnly => {
                        ssa.repair_ssa();
                    }
                    ModificationScope::CfgModifying => {
                        if let Err(e) = ssa.rebuild_ssa() {
                            if let Some(snapshot) = original {
                                ssa = snapshot;
                            }
                            host.insert_ssa(method.clone(), ssa);
                            if let Ok(mut guard) = errors.lock() {
                                guard.push(format!(
                                    "pass '{}' failed SSA rebuild for method {:?}: {}",
                                    pass.name(),
                                    method,
                                    e
                                ));
                            }
                            return;
                        }
                    }
                }
            }

            if verify_hard {
                let verifier_errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
                if !verifier_errors.is_empty() {
                    if let Some(snapshot) = original {
                        ssa = snapshot;
                    }
                    host.insert_ssa(method.clone(), ssa);
                    let details = verifier_errors
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("; ");
                    if let Ok(mut guard) = errors.lock() {
                        guard.push(format!(
                            "pass '{}' produced invalid SSA for method {:?}: {}",
                            pass.name(),
                            method,
                            details
                        ));
                    }
                    return;
                }
            }

            host.insert_ssa(method.clone(), ssa);

            if changed {
                any_changed.store(true, Ordering::Relaxed);
                pass_change_count.fetch_add(1, Ordering::Relaxed);
                host.mark_processed(method);
                iteration_modified.insert(method.clone());
            }
        });

        let failures = errors
            .into_inner()
            .map_err(|_| Error::new("scheduler: failed to read pass errors"))?;
        if let Some(first) = failures.first() {
            return Err(Error::new(first.clone()));
        }

        if log::log_enabled!(log::Level::Debug) {
            let count = pass_change_count.load(Ordering::Relaxed);
            if count > 0 {
                let event_delta = host.events().count_by_kind_since(event_snapshot);
                if event_delta.is_empty() {
                    debug!("  pass '{}' changed {} methods", pass.name(), count);
                } else {
                    let summary = format_event_delta(&event_delta);
                    if summary.is_empty() {
                        debug!("  pass '{}' changed {} methods", pass.name(), count);
                    } else {
                        debug!(
                            "  pass '{}' changed {} methods ({})",
                            pass.name(),
                            count,
                            summary
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

/// Format an event-kind delta map into a compact summary string.
///
/// Example: "93 strings decrypted, 115 constants folded".
fn format_event_delta(delta: &HashMap<EventKind, usize>) -> String {
    let mut parts: Vec<String> = delta
        .iter()
        .filter(|(kind, _)| kind.is_transformation())
        .map(|(kind, count)| format!("{} {}", count, kind.description()))
        .collect();
    parts.sort();
    parts.join(", ")
}
