//! Complete SSA function representation: blocks, variables, and maintenance operations.
//!
//! [`SsaFunction`] is the top-level container for a method's SSA form. It holds all
//! basic blocks, SSA variables, signature metadata (argument/local counts), exception
//! handlers, and the rename group system used by SSA construction and rebuild.
//!
//! # Sub-modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `canonical` | Final cleanup: strip nops, remove empty blocks, compact indices |
//! | `duplication` | Block cloning with fresh variable allocation |
//! | `queries` | Read-only analysis: return info, purity, variable tracing |
//! | `rebuild` | Full SSA reconstruction (Cytron et al. algorithm) after CFG changes |
//! | `repair` | Lightweight SSA repair for instruction-only passes |
//! | `transforms` | Mutation operations: replace uses, eliminate phis, compact, fold |
//!
//! # Structure
//!
//! ```text
//! SsaFunction<T>
//! ├── blocks: Vec<SsaBlock<T>>           // Basic blocks indexed by ID
//! ├── variables: Vec<SsaVariable<T>>      // All SSA variables (dense indexing)
//! ├── var_allocator: FunctionVarAllocator  // Dense ID allocation
//! ├── origin_versions: BTreeMap            // Origin → version list mappings
//! ├── origin_types: BTreeMap              // Origin → canonical type
//! ├── num_args / num_locals              // Method signature info
//! ├── exception_handlers: Vec<..>        // Preserved exception handler metadata
//! └── rename_groups: Vec<u32>            // Per-variable rename group IDs
//! ```
//!
//! # Construction
//!
//! Built by the `SsaConverter` (in the host crate) which:
//! 1. Simulates the CIL evaluation stack to create explicit SSA variables
//! 2. Places phi nodes at dominance frontiers (Cytron et al.)
//! 3. Renames variables to achieve single-assignment form
//! 4. Records exception handler block ranges
//!
//! # Maintenance Operations
//!
//! | Operation | Scope | Cost |
//! |-----------|-------|------|
//! | `repair_ssa()` | Instruction-only passes | Lightweight |
//! | `rebuild_ssa()` | CFG-modifying passes | Full reconstruction |
//! | `canonicalize()` | Pre-codegen cleanup | Moderate |
//!
//! # Thread Safety
//!
//! `SsaFunction` is `Send` and `Sync` once constructed.

mod canonical;
mod duplication;
mod kind;
mod queries;
mod rebuild;
mod repair;
mod transforms;

pub use kind::FunctionKind;
pub use queries::{MethodPurity, ReturnInfo};
pub use transforms::TrivialPhiOptions;

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use crate::{
    analysis::verifier::{SsaVerifier, VerifyLevel},
    ir::{
        block::SsaBlock,
        exception::SsaExceptionHandler,
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        variable::{DefSite, FunctionVarAllocator, SsaVarId, SsaVariable, VariableOrigin},
    },
    target::Target,
};

/// A complete method in SSA (Static Single Assignment) form.
///
/// The top-level container holding all SSA state for a single method:
/// basic blocks with phi nodes and instructions, all SSA variables with
/// full metadata (origins, types, def/use sites), and method signature info.
///
/// # Examples
///
/// ```rust
/// use analyssa::{ir::{function::SsaFunction, block::SsaBlock, variable::SsaVarId}, MockTarget};
///
/// let mut func: SsaFunction<MockTarget> = SsaFunction::new(2, 1);
/// func.add_block(SsaBlock::new(0));
/// func.add_block(SsaBlock::new(1));
/// func.add_block(SsaBlock::new(2));
///
/// for var in func.variables() {
///     println!("Variable: {}", var);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SsaFunction<T: Target> {
    /// Basic blocks, indexed by block ID (0..block_count()-1).
    /// Maintained as a dense vector; blocks can be removed by `canonicalize`.
    blocks: Vec<SsaBlock<T>>,

    /// All SSA variables, densely indexed by `SsaVarId` (0..len()).
    ///
    /// Invariant: `variables[i].id().index() == i` for all entries.
    /// Maintained by `create_variable()` (dense allocation) and
    /// `compact_variables()` / `reindex_variables()` (post-removal repair).
    variables: Vec<SsaVariable<T>>,

    /// Per-function allocator producing dense sequential variable IDs.
    var_allocator: FunctionVarAllocator,

    /// Maps each [`VariableOrigin`] to its SSA variable IDs, ordered by version.
    ///
    /// Enables O(1) lookup of all versions of a given origin (e.g., all
    /// SSA versions of `Local(3)`). Populated by `create_variable()` and
    /// rebuilt by `rebuild_origin_versions()` after compaction.
    origin_versions: BTreeMap<VariableOrigin, Vec<SsaVarId>>,

    /// Maps each variable origin to its canonical type.
    ///
    /// Populated during SSA construction from method signatures and type
    /// inference. Used by [`create_variable_for_origin()`](Self::create_variable_for_origin)
    /// so new versions inherit the correct type. First registration wins.
    origin_types: BTreeMap<VariableOrigin, T::Type>,

    /// Number of method arguments (including `this` for instance methods).
    num_args: usize,

    /// Number of local variables declared in the current SSA state.
    /// May be larger than `original_num_locals` if passes create temporaries.
    num_locals: usize,

    /// Number of local variables in the original method signature.
    /// Used as a floor for `shrink_num_locals()` since these locals
    /// have default-initialization semantics.
    original_num_locals: usize,

    /// Variables that control input-dependent control flow.
    ///
    /// Switches using these variables should not be simplified to constant
    /// jumps even if the value appears constant on some paths (e.g., real
    /// dispatcher variables that depend on runtime input). Set via
    /// [`mark_preserved_dispatch_var`](Self::mark_preserved_dispatch_var).
    preserved_dispatch_vars: BTreeSet<SsaVarId>,

    /// Original local variable types from the method signature.
    ///
    /// Preserved during SSA construction so code generation can emit correct
    /// type information in the output assembly. Set via
    /// [`set_original_local_types`](Self::set_original_local_types).
    original_local_types: Option<Vec<T::LocalSignature>>,

    /// Exception handlers preserved from the original method body.
    ///
    /// Each handler stores both original IL offsets and SSA block index
    /// ranges. Remapped during canonicalization; used during code generation
    /// to emit correct exception handler metadata.
    exception_handlers: Vec<SsaExceptionHandler<T>>,

    /// Per-variable rename group IDs, indexed by `SsaVarId::index()`.
    ///
    /// During SSA construction and rebuild, variables sharing the same rename
    /// group share a version stack for phi placement and renaming. See the
    /// [module docs](self) for group assignment rules.
    ///
    /// Group assignment:
    /// - `Argument(i)` → group `i`
    /// - `Local(i)` → group `num_args + i`
    /// - Stack temp at depth D → group `num_args + num_locals + D`
    /// - Orphan/pass-created → auto-incrementing from max group + 1
    /// - `u32::MAX` means no group assigned
    rename_groups: Vec<u32>,

    /// Function kind: normal, interrupt handler, etc.
    ///
    /// Defaults to [`FunctionKind::Normal`]. Set during SSA construction by
    /// frontends that need to mark functions as interrupt service routines.
    kind: FunctionKind,
}

impl<T: Target> SsaFunction<T> {
    /// Creates a new empty SSA function.
    ///
    /// # Arguments
    ///
    /// * `num_args` - Number of method arguments (including `this` for instance methods)
    /// * `num_locals` - Number of local variables declared in the method
    ///
    /// # Returns
    ///
    /// A new empty [`SsaFunction`] with no blocks or variables.
    #[must_use]
    pub fn new(num_args: usize, num_locals: usize) -> Self {
        Self {
            blocks: Vec::new(),
            variables: Vec::new(),
            var_allocator: FunctionVarAllocator::new(),
            origin_versions: BTreeMap::new(),
            origin_types: BTreeMap::new(),
            num_args,
            num_locals,
            original_num_locals: num_locals,
            preserved_dispatch_vars: BTreeSet::new(),
            original_local_types: None,
            exception_handlers: Vec::new(),
            rename_groups: Vec::new(),
            kind: FunctionKind::Normal,
        }
    }

    /// Creates a new SSA function with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `num_args` - Number of method arguments
    /// * `num_locals` - Number of local variables
    /// * `block_capacity` - Expected number of blocks
    /// * `var_capacity` - Expected number of SSA variables
    ///
    /// # Returns
    ///
    /// A new empty [`SsaFunction`] with pre-allocated storage.
    #[must_use]
    pub fn with_capacity(
        num_args: usize,
        num_locals: usize,
        block_capacity: usize,
        var_capacity: usize,
    ) -> Self {
        Self {
            blocks: Vec::with_capacity(block_capacity),
            variables: Vec::with_capacity(var_capacity),
            var_allocator: FunctionVarAllocator::new(),
            origin_versions: BTreeMap::new(),
            origin_types: BTreeMap::new(),
            num_args,
            num_locals,
            original_num_locals: num_locals,
            preserved_dispatch_vars: BTreeSet::new(),
            original_local_types: None,
            exception_handlers: Vec::new(),
            rename_groups: Vec::with_capacity(var_capacity),
            kind: FunctionKind::Normal,
        }
    }

    /// Returns the SSA blocks.
    ///
    /// # Returns
    ///
    /// A slice of all [`SsaBlock<T>`]s in this function.
    #[must_use]
    pub fn blocks(&self) -> &[SsaBlock<T>] {
        &self.blocks
    }

    /// Returns an iterator over blocks with their indices.
    ///
    /// This is a convenience method that pairs each block with its index,
    /// avoiding the common `for block_idx in 0..ssa.block_count()` pattern.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use analyssa::{MockTarget, ir::{SsaBlock, SsaFunction}};
    /// let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    /// ssa.add_block(SsaBlock::new(0));
    ///
    /// for (block_idx, block) in ssa.iter_blocks() {
    ///     println!("Block {}: {} instructions", block_idx, block.instruction_count());
    /// }
    /// ```
    pub fn iter_blocks(&self) -> impl Iterator<Item = (usize, &SsaBlock<T>)> {
        self.blocks.iter().enumerate()
    }

    /// Returns an iterator over all instructions with their block and instruction indices.
    ///
    /// This flattens the nested block/instruction structure into a single iterator,
    /// which is useful for passes that need to scan all instructions.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use analyssa::{MockTarget, ir::{ConstValue, SsaBlock, SsaFunction, SsaInstruction, SsaOp, SsaVarId}};
    /// let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    /// let mut block = SsaBlock::new(0);
    /// block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
    ///     dest: SsaVarId::from_index(0),
    ///     value: ConstValue::I32(1),
    /// }));
    /// ssa.add_block(block);
    ///
    /// for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
    ///     let op = instr.op();
    ///     assert_eq!((block_idx, instr_idx), (0, 0));
    ///     assert!(matches!(op, SsaOp::Const { .. }));
    /// }
    /// ```
    pub fn iter_instructions(&self) -> impl Iterator<Item = (usize, usize, &SsaInstruction<T>)> {
        self.blocks
            .iter()
            .enumerate()
            .flat_map(|(block_idx, block)| {
                block
                    .instructions()
                    .iter()
                    .enumerate()
                    .map(move |(instr_idx, instr)| (block_idx, instr_idx, instr))
            })
    }

    /// Returns a mutable iterator over all instructions with their block and instruction indices.
    ///
    /// This is the mutable counterpart to [`iter_instructions`], allowing passes to
    /// modify instructions while iterating. Note that structural changes (adding/removing
    /// instructions) require collecting the modifications and applying them separately.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use analyssa::{MockTarget, ir::{ConstValue, SsaBlock, SsaFunction, SsaInstruction, SsaOp, SsaVarId}};
    /// let old_var = SsaVarId::from_index(0);
    /// let new_var = SsaVarId::from_index(1);
    /// let dest = SsaVarId::from_index(2);
    /// let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    /// let mut block = SsaBlock::new(0);
    /// block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
    ///     dest: new_var,
    ///     value: ConstValue::I32(1),
    /// }));
    /// block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy { dest, src: old_var }));
    /// ssa.add_block(block);
    ///
    /// for (block_idx, instr_idx, instr) in ssa.iter_instructions_mut() {
    ///     let _ = (block_idx, instr_idx);
    ///     instr.op_mut().replace_uses(old_var, new_var);
    /// }
    /// ```
    ///
    /// # Note
    ///
    /// For passes that need to add or remove instructions, use [`blocks_mut`] to access
    /// the blocks directly, as the iterator cannot handle structural modifications.
    ///
    /// [`iter_instructions`]: Self::iter_instructions
    /// [`blocks_mut`]: Self::blocks_mut
    pub fn iter_instructions_mut(
        &mut self,
    ) -> impl Iterator<Item = (usize, usize, &mut SsaInstruction<T>)> {
        self.blocks
            .iter_mut()
            .enumerate()
            .flat_map(|(block_idx, block)| {
                block
                    .instructions_mut()
                    .iter_mut()
                    .enumerate()
                    .map(move |(instr_idx, instr)| (block_idx, instr_idx, instr))
            })
    }

    /// Returns an iterator over all phi nodes with their block and phi indices.
    ///
    /// This flattens the nested block/phi structure into a single iterator,
    /// which is useful for passes that need to analyze all phi nodes.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use analyssa::{MockTarget, ir::{PhiNode, SsaBlock, SsaFunction, SsaVarId, VariableOrigin}};
    /// let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    /// let mut block = SsaBlock::new(0);
    /// block.add_phi(PhiNode::new(SsaVarId::from_index(0), VariableOrigin::Local(0)));
    /// ssa.add_block(block);
    ///
    /// for (block_idx, phi_idx, phi) in ssa.iter_phis() {
    ///     println!("Phi {} in block {} defines {}", phi_idx, block_idx, phi.result());
    /// }
    /// ```
    pub fn iter_phis(&self) -> impl Iterator<Item = (usize, usize, &PhiNode)> {
        self.blocks
            .iter()
            .enumerate()
            .flat_map(|(block_idx, block)| {
                block
                    .phi_nodes()
                    .iter()
                    .enumerate()
                    .map(move |(phi_idx, phi)| (block_idx, phi_idx, phi))
            })
    }

    /// Returns a mutable reference to the blocks.
    ///
    /// # Returns
    ///
    /// A mutable reference to the vector of [`SsaBlock<T>`]s.
    pub fn blocks_mut(&mut self) -> &mut Vec<SsaBlock<T>> {
        &mut self.blocks
    }

    /// Returns the SSA variables.
    ///
    /// # Returns
    ///
    /// A slice of all [`SsaVariable<T>`]s in this function.
    #[must_use]
    pub fn variables(&self) -> &[SsaVariable<T>] {
        &self.variables
    }

    /// Returns a mutable reference to the variables.
    ///
    /// # Returns
    ///
    /// A mutable reference to the vector of [`SsaVariable<T>`]s.
    pub fn variables_mut(&mut self) -> &mut Vec<SsaVariable<T>> {
        &mut self.variables
    }

    /// Returns the number of method arguments.
    ///
    /// # Returns
    ///
    /// The count of method arguments, including `this` for instance methods.
    #[must_use]
    pub const fn num_args(&self) -> usize {
        self.num_args
    }

    /// Returns the number of local variables.
    ///
    /// # Returns
    ///
    /// The count of local variables declared in the method.
    #[must_use]
    pub const fn num_locals(&self) -> usize {
        self.num_locals
    }

    /// Returns the number of locals from the original method signature.
    ///
    /// With the group-based rename system, this is always equal to `num_locals`
    /// since stack temporaries use `Phi` origin instead of inflated local indices.
    #[must_use]
    pub const fn original_num_locals(&self) -> usize {
        self.original_num_locals
    }

    /// Sets the total number of local variables.
    pub fn set_num_locals(&mut self, num_locals: usize, original_num_locals: usize) {
        self.num_locals = num_locals;
        self.original_num_locals = original_num_locals;
    }

    /// Returns the number of blocks.
    ///
    /// # Returns
    ///
    /// The count of basic blocks in this function.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the number of variables.
    ///
    /// # Returns
    ///
    /// The count of SSA variables in this function.
    #[must_use]
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Returns the minimum BitSet capacity needed to index all variable IDs
    /// that appear in this function (in the variables vec, block instructions,
    /// and phi nodes).
    ///
    /// This handles cases where variable IDs don't match their position in
    /// the variables vector (e.g., in test code using `SsaVarId::from_index`
    /// without registering via `create_variable`).
    #[must_use]
    pub fn var_id_capacity(&self) -> usize {
        let from_vars = self
            .variables
            .iter()
            .map(|v| v.id().index().saturating_add(1))
            .max()
            .unwrap_or(0);
        let from_blocks = self
            .blocks
            .iter()
            .flat_map(|b| {
                let phi_ids = b.phi_nodes().iter().flat_map(|p| {
                    std::iter::once(p.result().index())
                        .chain(p.operands().iter().map(|op| op.value().index()))
                });
                let instr_ids = b.instructions().iter().flat_map(|i| {
                    i.op()
                        .dest()
                        .into_iter()
                        .chain(i.op().uses())
                        .map(|v| v.index())
                });
                phi_ids.chain(instr_ids)
            })
            .max()
            .map_or(0, |m| m.saturating_add(1));
        from_vars.max(from_blocks).max(self.variables.len())
    }

    /// Returns all variable IDs for a given origin, ordered by creation.
    ///
    /// This is O(1) via the version registry. For example,
    /// `versions_of(VariableOrigin::Local(3))` returns all SSA versions
    /// of local variable 3.
    #[must_use]
    pub fn versions_of(&self, origin: VariableOrigin) -> &[SsaVarId] {
        self.origin_versions
            .get(&origin)
            .map_or(&[], |v| v.as_slice())
    }

    /// Returns the most recently created variable ID for a given origin.
    #[must_use]
    pub fn latest_version(&self, origin: VariableOrigin) -> Option<SsaVarId> {
        self.origin_versions
            .get(&origin)
            .and_then(|v| v.last().copied())
    }

    /// Gets the local index for a variable ID.
    ///
    /// With dense IDs, this is always O(1) — the index equals `id.index()`.
    ///
    /// # Arguments
    ///
    /// * `id` - The variable ID to look up
    ///
    /// # Returns
    ///
    /// The local index (0-based), or `None` if the variable is not in this function.
    #[must_use]
    pub fn var_index(&self, id: SsaVarId) -> Option<usize> {
        let idx = id.index();
        if idx < self.variables.len() {
            Some(idx)
        } else {
            None
        }
    }

    /// Returns `true` if this function has no blocks.
    ///
    /// # Returns
    ///
    /// `true` if the function contains no blocks, `false` otherwise.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Gets a block by index.
    ///
    /// # Arguments
    ///
    /// * `index` - The block index to retrieve
    ///
    /// # Returns
    ///
    /// A reference to the block, or `None` if the index is out of bounds.
    #[must_use]
    pub fn block(&self, index: usize) -> Option<&SsaBlock<T>> {
        self.blocks.get(index)
    }

    /// Gets a mutable block by index.
    ///
    /// # Arguments
    ///
    /// * `index` - The block index to retrieve
    ///
    /// # Returns
    ///
    /// A mutable reference to the block, or `None` if the index is out of bounds.
    pub fn block_mut(&mut self, index: usize) -> Option<&mut SsaBlock<T>> {
        self.blocks.get_mut(index)
    }

    /// Gets a variable by ID. O(1) via dense indexing.
    ///
    /// # Arguments
    ///
    /// * `id` - The variable ID to look up
    ///
    /// # Returns
    ///
    /// A reference to the variable, or `None` if the ID is out of bounds.
    #[must_use]
    pub fn variable(&self, id: SsaVarId) -> Option<&SsaVariable<T>> {
        self.variables.get(id.index())
    }

    /// Gets a mutable variable by ID. O(1) via dense indexing.
    ///
    /// # Arguments
    ///
    /// * `id` - The variable ID to look up
    ///
    /// # Returns
    ///
    /// A mutable reference to the variable, or `None` if the ID is out of bounds.
    pub fn variable_mut(&mut self, id: SsaVarId) -> Option<&mut SsaVariable<T>> {
        self.variables.get_mut(id.index())
    }

    /// Adds a block to this function.
    ///
    /// # Arguments
    ///
    /// * `block` - The block to add
    pub fn add_block(&mut self, block: SsaBlock<T>) {
        self.blocks.push(block);
    }

    /// Creates a new variable with a dense ID allocated by this function.
    ///
    /// This is the **only** way to create variables. The ID is guaranteed to be
    /// dense (equal to the variable's index in the variables Vec), enabling
    /// O(1) lookup via direct indexing.
    ///
    /// If `var_type` is not `Unknown`, it is automatically registered in the
    /// origin type registry for future lookups.
    pub fn create_variable(
        &mut self,
        origin: VariableOrigin,
        version: u32,
        def_site: DefSite,
        var_type: T::Type,
    ) -> SsaVarId {
        let id = self.var_allocator.alloc();
        let var = SsaVariable::new(id, origin, version, def_site, var_type.clone());
        debug_assert_eq!(id.index(), self.variables.len());
        self.origin_versions.entry(origin).or_default().push(id);
        // Register origin type if known (first concrete type wins)
        if !T::is_unknown(&var_type) && !self.origin_types.contains_key(&origin) {
            self.origin_types.insert(origin, var_type);
        }
        self.variables.push(var);
        // Extend rename_groups to keep it in sync (default u32::MAX = no group)
        if self.rename_groups.len() <= id.index() {
            self.rename_groups
                .resize(id.index().saturating_add(1), u32::MAX);
        }
        id
    }

    /// Creates a new variable, inferring its type from the origin type registry.
    ///
    /// This is a convenience method for creating new versions of variables
    /// whose origin type was previously registered. If no type is registered
    /// for the origin, the variable gets `SsaType::Unknown`.
    pub fn create_variable_for_origin(
        &mut self,
        origin: VariableOrigin,
        version: u32,
        def_site: DefSite,
    ) -> SsaVarId {
        let var_type = self.origin_type(origin);
        self.create_variable(origin, version, def_site, var_type)
    }

    /// Registers the canonical type for a variable origin.
    ///
    /// Only registers if the type is not the host's "unknown" type. If a type
    /// is already registered for this origin, it is not overwritten (first wins).
    pub fn register_origin_type(&mut self, origin: VariableOrigin, var_type: T::Type) {
        if !T::is_unknown(&var_type) && !self.origin_types.contains_key(&origin) {
            self.origin_types.insert(origin, var_type);
        }
    }

    /// Returns the registered type for a variable origin, or the host's
    /// "unknown" type.
    #[must_use]
    pub fn origin_type(&self, origin: VariableOrigin) -> T::Type {
        self.origin_types
            .get(&origin)
            .cloned()
            .unwrap_or_else(T::unknown_type)
    }

    /// Returns the origin type registry.
    #[must_use]
    pub fn origin_types(&self) -> &BTreeMap<VariableOrigin, T::Type> {
        &self.origin_types
    }

    /// Rebuilds the origin_versions registry from the current variables list.
    ///
    /// Called after operations that modify the variables list (compact, reindex).
    fn rebuild_origin_versions(&mut self) {
        self.origin_versions.clear();
        for var in &self.variables {
            self.origin_versions
                .entry(var.origin())
                .or_default()
                .push(var.id());
        }
    }

    /// Reassigns dense variable IDs after variable removal.
    ///
    /// This must be called after removing variables from `self.variables` to restore
    /// the dense indexing invariant (`variables[i].id().index() == i`).
    ///
    /// Returns a mapping from old IDs to new IDs for updating references.
    fn reassign_dense_ids(&mut self) -> BTreeMap<SsaVarId, SsaVarId> {
        let mut remap = BTreeMap::new();
        let old_groups = std::mem::take(&mut self.rename_groups);
        self.var_allocator = FunctionVarAllocator::starting_from(self.variables.len());
        let mut new_groups = vec![u32::MAX; self.variables.len()];
        for (index, var) in self.variables.iter_mut().enumerate() {
            let old_id = var.id();
            let new_id = SsaVarId::from_index(index);
            // Carry over the rename group from the old position
            if let Some(&old_group) = old_groups.get(old_id.index()) {
                if let Some(slot) = new_groups.get_mut(index) {
                    *slot = old_group;
                }
            }
            if old_id != new_id {
                remap.insert(old_id, new_id);
                var.set_id(new_id);
            }
        }
        self.rename_groups = new_groups;
        remap
    }

    /// Remaps all variable ID references in blocks (instructions, phi nodes, terminators)
    /// using the given old-to-new ID mapping.
    fn remap_var_ids_in_blocks(&mut self, remap: &BTreeMap<SsaVarId, SsaVarId>) {
        if remap.is_empty() {
            return;
        }
        let lookup = |id: SsaVarId| -> Option<SsaVarId> { remap.get(&id).copied() };
        let resolve = |id: SsaVarId| -> SsaVarId { remap.get(&id).copied().unwrap_or(id) };

        for block in &mut self.blocks {
            // Remap phi nodes
            for phi in block.phi_nodes_mut() {
                let old_result = phi.result();
                phi.set_result(resolve(old_result));
                for operand in phi.operands_mut() {
                    let old_value = operand.value();
                    *operand = PhiOperand::new(resolve(old_value), operand.predecessor());
                }
            }
            // Remap instructions using existing remap_variables
            for instr in block.instructions_mut() {
                let new_op = instr.op().remap_variables(lookup);
                instr.set_op(new_op);
            }
        }
        // Remap preserved_dispatch_vars
        let remapped_dispatch: BTreeSet<SsaVarId> = self
            .preserved_dispatch_vars
            .iter()
            .map(|id| resolve(*id))
            .collect();
        self.preserved_dispatch_vars = remapped_dispatch;
    }

    /// Marks a variable as a preserved dispatch variable.
    ///
    /// Preserved dispatch variables control input-dependent control flow
    /// (e.g., switches that depend on runtime input rather than constants).
    /// Optimization passes should not simplify switches using these variables
    /// even if the value appears constant on some paths.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to mark as preserved.
    pub fn mark_preserved_dispatch_var(&mut self, var: SsaVarId) {
        self.preserved_dispatch_vars.insert(var);
    }

    /// Checks if a variable is a preserved dispatch variable.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to check.
    ///
    /// # Returns
    ///
    /// `true` if this variable controls input-dependent control flow.
    #[must_use]
    pub fn is_preserved_dispatch_var(&self, var: SsaVarId) -> bool {
        self.preserved_dispatch_vars.contains(&var)
    }

    /// Checks if any preserved dispatch variables are set.
    ///
    /// # Returns
    ///
    /// `true` if there are any preserved dispatch variables.
    #[must_use]
    pub fn has_preserved_dispatch_vars(&self) -> bool {
        !self.preserved_dispatch_vars.is_empty()
    }

    /// Sets the original local variable types from the method signature.
    ///
    /// These types are preserved so they can be used during code generation
    /// to maintain correct type information in the output assembly.
    ///
    /// # Arguments
    ///
    /// * `types` - The original local variable types from the method signature.
    pub fn set_original_local_types(&mut self, types: Vec<T::LocalSignature>) {
        self.original_local_types = Some(types);
    }

    /// Returns the original local variable types if set.
    ///
    /// # Returns
    ///
    /// The original local types, or `None` if not set.
    #[must_use]
    pub fn original_local_types(&self) -> Option<&[T::LocalSignature]> {
        self.original_local_types.as_deref()
    }

    /// Sets the exception handlers for this function.
    ///
    /// These are preserved from the original method body and will be
    /// remapped during code generation based on the new instruction layout.
    ///
    /// # Arguments
    ///
    /// * `handlers` - The exception handlers from the original method body.
    pub fn set_exception_handlers(&mut self, handlers: Vec<SsaExceptionHandler<T>>) {
        self.exception_handlers = handlers;
    }

    /// Returns the exception handlers for this function.
    ///
    /// # Returns
    ///
    /// A slice of exception handlers, or an empty slice if none are set.
    #[must_use]
    pub fn exception_handlers(&self) -> &[SsaExceptionHandler<T>] {
        &self.exception_handlers
    }

    /// Returns whether this function has any exception handlers.
    ///
    /// # Returns
    ///
    /// `true` if the function has at least one exception handler.
    #[must_use]
    pub fn has_exception_handlers(&self) -> bool {
        !self.exception_handlers.is_empty()
    }

    /// Returns `true` if any block in this function contains an
    /// [`InterruptReturn`](SsaOp::InterruptReturn) op.
    ///
    /// This is useful for validation: a function using `InterruptReturn`
    /// should typically be marked as [`FunctionKind::InterruptHandler`].
    ///
    /// # Returns
    ///
    /// `true` if at least one `InterruptReturn` op exists in the function.
    #[must_use]
    pub fn has_interrupt_return(&self) -> bool {
        self.blocks.iter().any(|block| {
            block
                .instructions()
                .iter()
                .any(|instr| matches!(instr.op(), SsaOp::InterruptReturn))
        })
    }

    /// Returns the function kind.
    ///
    /// # Returns
    ///
    /// The [`FunctionKind`] of this function (defaults to [`FunctionKind::Normal`]).
    #[must_use]
    pub fn kind(&self) -> FunctionKind {
        self.kind
    }

    /// Sets the function kind.
    ///
    /// # Arguments
    ///
    /// * `kind` - The [`FunctionKind`] to assign (e.g., [`FunctionKind::InterruptHandler`]).
    pub fn set_kind(&mut self, kind: FunctionKind) {
        self.kind = kind;
    }

    /// Returns the rename group for a variable.
    ///
    /// Returns `u32::MAX` if no group has been assigned (the variable was
    /// created without a rename group, e.g. by a compiler pass).
    #[must_use]
    pub fn rename_group(&self, var_id: SsaVarId) -> u32 {
        self.rename_groups
            .get(var_id.index())
            .copied()
            .unwrap_or(u32::MAX)
    }

    /// Sets the rename group for a variable.
    ///
    /// Extends the `rename_groups` vector with `u32::MAX` if needed.
    pub fn set_rename_group(&mut self, var_id: SsaVarId, group: u32) {
        let idx = var_id.index();
        if idx >= self.rename_groups.len() {
            self.rename_groups.resize(idx.saturating_add(1), u32::MAX);
        }
        if let Some(slot) = self.rename_groups.get_mut(idx) {
            *slot = group;
        }
    }

    /// Sorts instructions in all blocks in topological order.
    ///
    /// This ensures that within each block, if instruction A uses a value defined
    /// by instruction B, then B appears before A.
    ///
    /// This is called automatically by [`rebuild_ssa`](Self::rebuild_ssa) but can
    /// also be called manually after passes that may have disrupted instruction order.
    ///
    /// # Returns
    ///
    /// `true` if all blocks were successfully sorted, `false` if any block has
    /// cyclic dependencies (which indicates invalid SSA).
    pub fn sort_all_blocks_topologically(&mut self) -> bool {
        let mut all_sorted = true;
        for block in &mut self.blocks {
            if !block.sort_instructions_topologically() {
                all_sorted = false;
            }
        }
        all_sorted
    }

    /// Validates that no meaningfully-used variable has `SsaType::Unknown`.
    ///
    /// This ensures that all variables whose values are actually consumed have a
    /// concrete type. Variables are considered NOT meaningfully used if:
    /// - They have no uses at all (dead variables, stripped by DCE)
    /// - Their only uses are in `Pop` instructions (value is discarded)
    /// - Their only uses are as phi operands where the phi result is also unused
    ///
    /// # Errors
    ///
    /// Returns `Err` with a description listing the first Unknown-typed
    /// variable that has meaningful uses.
    pub fn validate_types(&self) -> Result<(), String> {
        for var in &self.variables {
            if !T::is_unknown(var.var_type()) || var.uses().is_empty() {
                continue;
            }

            // Check if all uses are in Pop instructions (value is discarded)
            let has_meaningful_use = var.uses().iter().any(|use_site| {
                if use_site.is_phi_operand {
                    // Phi operand — only meaningful if the phi result has a known type.
                    // If the phi result is also Unknown, this is just Unknown feeding
                    // Unknown (e.g., uninitialized locals in a loop), not a real error.
                    if let Some(block) = self.block(use_site.block) {
                        if let Some(phi) = block.phi(use_site.instruction) {
                            if let Some(result_var) = self.variable(phi.result()) {
                                return !T::is_unknown(result_var.var_type());
                            }
                        }
                    }
                    return false;
                }
                if let Some(block) = self.block(use_site.block) {
                    if let Some(instr) = block.instruction(use_site.instruction) {
                        return !matches!(instr.op(), SsaOp::Pop { .. });
                    }
                }
                true // Conservative: assume meaningful if we can't check
            });

            if has_meaningful_use {
                // Collect details about the meaningful uses for debugging
                let use_details: Vec<String> = var
                    .uses()
                    .iter()
                    .map(|use_site| {
                        if use_site.is_phi_operand {
                            return format!("phi in block {}", use_site.block);
                        }
                        if let Some(block) = self.block(use_site.block) {
                            if let Some(instr) = block.instruction(use_site.instruction) {
                                return format!(
                                    "block {} instr {}: {:?}",
                                    use_site.block,
                                    use_site.instruction,
                                    instr.op()
                                );
                            }
                        }
                        format!(
                            "block {} instr {}: <unknown>",
                            use_site.block, use_site.instruction
                        )
                    })
                    .collect();
                return Err(format!(
                    "Variable {} (origin={:?}) has Unknown type but is used ({} uses): [{}]",
                    var.id(),
                    var.origin(),
                    var.uses().len(),
                    use_details.join(", ")
                ));
            }
        }
        Ok(())
    }
}

impl<T: Target> SsaFunction<T> {
    /// Rebuilds SSA form after CFG modifications (e.g., control flow unflattening).
    ///
    /// This method performs a complete SSA reconstruction using the standard
    /// Cytron et al. algorithm implemented by the internal SSA rebuilder.
    ///
    /// This is necessary because after passes like control flow unflattening,
    /// the CFG structure changes significantly and PHI nodes may reference
    /// variables from removed blocks or have incorrect operands.
    pub fn rebuild_ssa(&mut self) -> crate::Result<()> {
        if self.blocks.is_empty() {
            return Ok(());
        }
        rebuild::SsaRebuilder::new(self).rebuild()
    }

    /// Validates that the SSA function is well-formed.
    ///
    /// This checks several SSA invariants:
    ///
    /// 1. **No cyclic dependencies within a block** - Operations must have a valid
    ///    topological order. If operation A uses the result of operation B, then B
    ///    must come before A in the instruction list.
    ///
    /// 2. **Single definition** - Each variable should be defined at most once
    ///    (the defining property of SSA form).
    ///
    /// 3. **Phi nodes at block start** - Phi nodes should only appear at the
    ///    beginning of blocks, not mixed with regular instructions.
    ///
    /// # Errors
    ///
    /// Returns `Err` with a description of the problem if any SSA invariant is violated,
    /// such as cyclic dependencies, duplicate definitions, or misplaced terminators.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ssa = build_ssa_from_method(&method)?;
    /// ssa.validate()?; // Returns error if SSA is malformed
    ///
    /// // After running a pass
    /// some_pass.run(&mut ssa);
    /// ssa.validate()?; // Check the pass didn't break SSA invariants
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        let errors = SsaVerifier::new(self).verify(VerifyLevel::Standard);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; "))
        }
    }

    /// Checks if the SSA function is valid without returning detailed errors.
    ///
    /// This is a convenience method that returns `true` if [`validate`](Self::validate)
    /// would return `Ok(())`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
}

impl<T: Target> fmt::Display for SsaFunction<T>
where
    SsaBlock<T>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "SSA Function ({}, {} args, {} locals):",
            self.kind, self.num_args, self.num_locals
        )?;
        writeln!(f, "  Variables: {}", self.variables.len())?;
        writeln!(f, "  Blocks: {}", self.blocks.len())?;
        writeln!(f)?;

        for block in &self.blocks {
            write!(f, "{block}")?;
        }

        Ok(())
    }
}
