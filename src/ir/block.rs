//! SSA basic blocks containing phi nodes and instructions.
//!
//! An SSA block is the SSA-form representation of a CFG basic block. It contains:
//!
//! - **Phi nodes**: At the block entry, merging values from predecessors
//! - **Instructions**: SSA-form instructions with explicit def/use
//!
//! # Block Structure
//!
//! ```text
//! Block B:
//!   // Phi nodes (executed "simultaneously" at block entry)
//!   v3 = phi(v1 from B0, v2 from B1)
//!   v6 = phi(v4 from B0, v5 from B1)
//!
//!   // Instructions (executed sequentially)
//!   v7 = add v3, v6
//!   v8 = mul v7, v3
//!   br B2
//! ```
//!
//! # Semantics
//!
//! Phi nodes are evaluated at block entry before any instructions execute.
//! Conceptually, all phi nodes in a block read their operands simultaneously,
//! then all write their results simultaneously. This avoids ordering issues
//! when one phi's result is used as another phi's operand.
//!
//! # Thread Safety
//!
//! All types in this module are `Send` and `Sync`.

use std::{
    collections::{HashMap, VecDeque},
    fmt,
};

use crate::{
    ir::{
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        variable::SsaVarId,
    },
    target::Target,
    BitSet,
};

/// Result of a variable replacement operation.
///
/// When `replace_uses` encounters an instruction whose destination equals
/// `new_var`, it skips that instruction to avoid creating self-referential
/// operations (e.g., `v5 = add(v5, v3)`). This struct reports both the
/// successful replacements and the skipped ones, allowing callers to make
/// informed decisions without post-hoc scanning.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReplaceResult {
    /// Number of uses successfully replaced.
    pub replaced: usize,
    /// Number of uses skipped due to the self-referential guard
    /// (instruction's dest == new_var, replacement would create self-reference).
    pub skipped: usize,
}

impl ReplaceResult {
    /// Returns true if all uses were replaced (nothing was skipped).
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.skipped == 0
    }
}

impl std::ops::Add for ReplaceResult {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            replaced: self.replaced.saturating_add(rhs.replaced),
            skipped: self.skipped.saturating_add(rhs.skipped),
        }
    }
}

/// An SSA basic block with phi nodes and instructions.
///
/// This represents a basic block in SSA form. It maintains a parallel structure
/// to the CFG blocks but with explicit variable information.
///
/// # Examples
///
/// ```rust
/// use analyssa::{MockTarget, ir::{PhiNode, SsaBlock, SsaInstruction, SsaOp, SsaVarId, VariableOrigin}};
///
/// let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
///
/// // Add a phi node
/// let v1 = SsaVarId::from_index(0);
/// let v2 = SsaVarId::from_index(1);
/// let result = SsaVarId::from_index(2);
/// let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
/// phi.set_operand(0, v1);
/// phi.set_operand(1, v2);
/// block.add_phi(phi);
///
/// // Add instructions
/// block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(result) }));
/// ```
#[derive(Debug, Clone)]
pub struct SsaBlock<T: Target> {
    /// Block index (matches CFG block index).
    id: usize,

    /// Phi nodes at block entry.
    ///
    /// These are evaluated "simultaneously" before any instructions.
    phi_nodes: Vec<PhiNode>,

    /// SSA instructions in execution order.
    instructions: Vec<SsaInstruction<T>>,
}

impl<T: Target> SsaBlock<T> {
    /// Creates a new empty SSA block.
    ///
    /// # Arguments
    ///
    /// * `id` - The block index (should match the corresponding CFG block)
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self {
            id,
            phi_nodes: Vec::new(),
            instructions: Vec::new(),
        }
    }

    /// Creates a new SSA block with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `id` - The block index
    /// * `phi_capacity` - Expected number of phi nodes
    /// * `instr_capacity` - Expected number of instructions
    #[must_use]
    pub fn with_capacity(id: usize, phi_capacity: usize, instr_capacity: usize) -> Self {
        Self {
            id,
            phi_nodes: Vec::with_capacity(phi_capacity),
            instructions: Vec::with_capacity(instr_capacity),
        }
    }

    /// Returns the block index.
    #[must_use]
    pub const fn id(&self) -> usize {
        self.id
    }

    /// Sets the block index.
    ///
    /// This is used during canonicalization when blocks are renumbered
    /// after empty blocks are removed.
    pub fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    /// Returns the phi nodes in this block.
    #[must_use]
    pub fn phi_nodes(&self) -> &[PhiNode] {
        &self.phi_nodes
    }

    /// Returns a mutable reference to the phi nodes.
    pub fn phi_nodes_mut(&mut self) -> &mut Vec<PhiNode> {
        &mut self.phi_nodes
    }

    /// Returns the instructions in this block.
    #[must_use]
    pub fn instructions(&self) -> &[SsaInstruction<T>] {
        &self.instructions
    }

    /// Returns a mutable reference to the instructions.
    pub fn instructions_mut(&mut self) -> &mut Vec<SsaInstruction<T>> {
        &mut self.instructions
    }

    /// Returns the number of phi nodes.
    #[must_use]
    pub fn phi_count(&self) -> usize {
        self.phi_nodes.len()
    }

    /// Returns the number of instructions.
    #[must_use]
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Returns `true` if this block has no phi nodes.
    #[must_use]
    pub fn has_no_phis(&self) -> bool {
        self.phi_nodes.is_empty()
    }

    /// Returns `true` if this block has no instructions.
    #[must_use]
    pub fn has_no_instructions(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Returns `true` if this block is completely empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.phi_nodes.is_empty() && self.instructions.is_empty()
    }

    /// Clears all phi nodes and instructions from this block.
    ///
    /// After calling this method, `is_empty()` will return `true`.
    /// The block ID is preserved.
    pub fn clear(&mut self) {
        self.phi_nodes.clear();
        self.instructions.clear();
    }

    /// Adds a phi node to this block.
    pub fn add_phi(&mut self, phi: PhiNode) {
        self.phi_nodes.push(phi);
    }

    /// Adds an instruction to this block.
    pub fn add_instruction(&mut self, instr: SsaInstruction<T>) {
        self.instructions.push(instr);
    }

    /// Gets a phi node by index.
    #[must_use]
    pub fn phi(&self, index: usize) -> Option<&PhiNode> {
        self.phi_nodes.get(index)
    }

    /// Gets a mutable phi node by index.
    pub fn phi_mut(&mut self, index: usize) -> Option<&mut PhiNode> {
        self.phi_nodes.get_mut(index)
    }

    /// Gets an instruction by index.
    #[must_use]
    pub fn instruction(&self, index: usize) -> Option<&SsaInstruction<T>> {
        self.instructions.get(index)
    }

    /// Gets a mutable instruction by index.
    pub fn instruction_mut(&mut self, index: usize) -> Option<&mut SsaInstruction<T>> {
        self.instructions.get_mut(index)
    }

    /// Gets the terminator instruction (last instruction in the block).
    ///
    /// In well-formed SSA, the last instruction should be a control flow
    /// instruction (Jump, Branch, Switch, Return, etc.).
    #[must_use]
    pub fn terminator(&self) -> Option<&SsaInstruction<T>> {
        self.instructions.last()
    }

    /// Gets the terminator operation if the block has a terminator instruction.
    ///
    /// This is a convenience method combining `terminator()` and `op()`.
    #[must_use]
    pub fn terminator_op(&self) -> Option<&SsaOp<T>> {
        self.instructions.last().map(SsaInstruction::op)
    }

    /// Returns the successor block indices for this block.
    ///
    /// The successors are determined by the terminator instruction:
    /// - Jump/Leave: single target
    /// - Branch/BranchCmp: true and false targets
    /// - Switch: all case targets plus default
    /// - Return/Throw/etc: no successors
    #[must_use]
    pub fn successors(&self) -> Vec<usize> {
        self.terminator_op()
            .map_or_else(Vec::new, super::SsaOp::successors)
    }

    /// Calls `f` for every successor block index of this block, without
    /// allocating. Allocation-free equivalent of [`successors`](Self::successors).
    pub fn for_each_successor<F>(&self, f: F)
    where
        F: FnMut(usize),
    {
        if let Some(op) = self.terminator_op() {
            op.for_each_successor(f);
        }
    }

    /// Redirects control flow targets from `old_target` to `new_target`.
    ///
    /// This modifies the block's terminator instruction in-place, redirecting any
    /// occurrences of `old_target` to `new_target`. Works with all control flow
    /// instructions: `Jump`, `Leave`, `Branch`, `BranchCmp`, and `Switch`.
    ///
    /// # Arguments
    ///
    /// * `old_target` - The block index to redirect from
    /// * `new_target` - The block index to redirect to
    ///
    /// # Returns
    ///
    /// `true` if any target was changed, `false` otherwise.
    pub fn redirect_target(&mut self, old_target: usize, new_target: usize) -> bool {
        if let Some(terminator) = self.instructions.last_mut() {
            return terminator.op_mut().redirect_target(old_target, new_target);
        }
        false
    }

    /// Sets all control flow targets to a single destination.
    ///
    /// This forces the block to unconditionally transfer control to `target`,
    /// regardless of any branch conditions. For branches, both targets are set
    /// to the same value. For other terminators (like `Return` or `Throw`),
    /// the terminator is replaced with an unconditional `Jump`.
    ///
    /// If the block has no terminator, a `Jump` instruction is added.
    ///
    /// # Arguments
    ///
    /// * `target` - The block index to jump to
    pub fn set_target(&mut self, target: usize) {
        if let Some(terminator) = self.instructions.last_mut() {
            match terminator.op_mut() {
                SsaOp::Jump { target: t } | SsaOp::Leave { target: t } => {
                    *t = target;
                }
                SsaOp::Branch {
                    true_target,
                    false_target,
                    ..
                }
                | SsaOp::BranchCmp {
                    true_target,
                    false_target,
                    ..
                } => {
                    *true_target = target;
                    *false_target = target;
                }
                SsaOp::Switch {
                    targets, default, ..
                } => {
                    *default = target;
                    for t in targets.iter_mut() {
                        *t = target;
                    }
                }
                _ => {
                    // Other terminators (Return, Throw, etc.) - replace with Jump
                    *terminator = SsaInstruction::synthetic(SsaOp::Jump { target });
                }
            }
        } else {
            // No terminator - add a Jump
            self.instructions
                .push(SsaInstruction::synthetic(SsaOp::Jump { target }));
        }
    }

    /// Replaces all instruction uses of `old_var` with `new_var` (skips phi operands).
    ///
    /// This is the safe default for optimization passes. It avoids creating
    /// cross-origin phi operand references that can break `rebuild_ssa`'s
    /// assumption that each variable flows to at most one phi origin.
    ///
    /// Instructions whose destination equals `new_var` are skipped to prevent
    /// self-referential instructions (e.g., `v5 = add(v5, v3)`).
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable ID whose uses should be replaced
    /// * `new_var` - The replacement variable ID
    ///
    /// # Returns
    ///
    /// A [`ReplaceResult`] reporting both successful replacements and skips
    /// due to the self-referential guard. Use `result.is_complete()` to check
    /// if all uses were replaced.
    ///
    /// # Related
    ///
    /// - [`replace_uses_including_phis`](Self::replace_uses_including_phis) for
    ///   internal operations that need phi operand replacement
    pub fn replace_uses(&mut self, old_var: SsaVarId, new_var: SsaVarId) -> ReplaceResult {
        let mut replaced: usize = 0;
        let mut skipped: usize = 0;

        for instr in &mut self.instructions {
            let op = instr.op_mut();
            // Skip if this would create a self-referential instruction
            if op.defs().any(|dest| dest == new_var) {
                if op.uses_var(old_var) {
                    skipped = skipped.saturating_add(1);
                }
                continue;
            }
            replaced = replaced.saturating_add(op.replace_uses(old_var, new_var));
        }

        ReplaceResult { replaced, skipped }
    }

    /// Replaces all uses of `old_var` with `new_var`, including in PHI operands.
    ///
    /// Unlike [`replace_uses`](Self::replace_uses), this method also replaces uses
    /// in PHI node operands. This is necessary for internal SSA operations that
    /// eliminate PHI nodes and need to forward their values through other PHIs.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable ID to find and replace.
    /// * `new_var` - The variable ID to use as the replacement.
    ///
    /// # Returns
    ///
    /// The number of uses that were replaced (in both instructions and PHI operands).
    ///
    /// # Safety
    ///
    /// This method is `pub(crate)` because it can create cross-origin PHI operand
    /// references if misused. The issue: `rebuild_ssa` uses a `phi_operand_origins`
    /// map that can only store ONE origin per variable. If a variable becomes a PHI
    /// operand for PHIs with different origins (e.g., Local(0) and Local(1)), only
    /// one origin is stored, causing incorrect def site classification and broken
    /// PHI placement.
    ///
    /// # When to Use
    ///
    /// Only use this method for:
    /// - **Trivial PHI elimination**: When removing a PHI like `v10 = phi(v5, v5)`,
    ///   we need to replace uses of `v10` with `v5` everywhere, including in other
    ///   PHI operands.
    /// - **Copy propagation within PHIs**: When a copy's destination is a PHI result
    ///   and we're eliminating that PHI.
    ///
    /// For optimization passes (copy propagation, GVN, etc.), use [`Self::replace_uses`]
    /// instead, which safely skips PHI operands.
    pub fn replace_uses_including_phis(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
    ) -> ReplaceResult {
        let mut result = self.replace_uses(old_var, new_var);

        // Replace in phi node operands
        for phi in &mut self.phi_nodes {
            for operand in phi.operands_mut() {
                if operand.value() == old_var {
                    *operand = PhiOperand::new(new_var, operand.predecessor());
                    result.replaced = result.replaced.saturating_add(1);
                }
            }
        }

        result
    }

    /// Finds a phi node that defines the given variable.
    #[must_use]
    pub fn find_phi_defining(&self, var: SsaVarId) -> Option<&PhiNode> {
        self.phi_nodes.iter().find(|phi| phi.result() == var)
    }

    /// Checks if this block is a trampoline block.
    ///
    /// A trampoline block is one that:
    /// - Has no phi nodes (doesn't merge values from multiple predecessors)
    /// - Contains only a single unconditional control transfer (`Jump` or `Leave`)
    ///
    /// Trampoline blocks can be bypassed by redirecting predecessors directly
    /// to their targets.
    ///
    /// # Returns
    ///
    /// `Some(target)` if this block is a trampoline to `target`, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{SsaBlock, SsaInstruction, SsaOp}};
    ///
    /// let mut block = SsaBlock::<MockTarget>::new(0);
    /// block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
    /// assert_eq!(block.is_trampoline(), Some(1));
    /// ```
    #[must_use]
    pub fn is_trampoline(&self) -> Option<usize> {
        // Blocks with phi nodes cannot be trampolines (they merge values)
        if !self.phi_nodes.is_empty() {
            return None;
        }

        // Must have exactly one operation
        if self.instructions.len() != 1 {
            return None;
        }

        // That operation must be an unconditional control transfer
        match self.instructions.first()?.op() {
            SsaOp::Jump { target } | SsaOp::Leave { target } => Some(*target),
            _ => None,
        }
    }

    /// Returns all variables defined in this block.
    ///
    /// This includes phi node results and instruction defs.
    pub fn defined_variables(&self) -> impl Iterator<Item = SsaVarId> + '_ {
        let phi_defs = self.phi_nodes.iter().map(PhiNode::result);
        let instr_defs = self.instructions.iter().filter_map(SsaInstruction::def);
        phi_defs.chain(instr_defs)
    }

    /// Returns all variables used in this block.
    ///
    /// This includes phi operands and instruction uses.
    pub fn used_variables(&self) -> impl Iterator<Item = SsaVarId> + '_ {
        let phi_uses = self.phi_nodes.iter().flat_map(PhiNode::used_variables);
        let instr_uses = self.instructions.iter().flat_map(SsaInstruction::uses);
        phi_uses.chain(instr_uses)
    }

    /// Sorts instructions within this block in topological order based on data dependencies.
    ///
    /// After sorting, if instruction A uses a value defined by instruction B (within this block),
    /// then B will appear before A in the instruction list.
    ///
    /// # Algorithm
    ///
    /// Uses Kahn's algorithm for topological sorting:
    /// 1. Build a dependency graph: instruction -> instructions it depends on
    /// 2. Start with instructions that have no dependencies within the block
    /// 3. Process in order, adding instructions whose dependencies are satisfied
    ///
    /// # Stability
    ///
    /// For instructions with no ordering constraints between them, the original
    /// relative order is preserved where possible.
    ///
    /// # Returns
    ///
    /// `true` if sorting succeeded, `false` if there was a cyclic dependency
    /// (which indicates invalid SSA). When a cycle is detected, the block is
    /// left unchanged.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{ConstValue, SsaBlock, SsaInstruction, SsaOp, SsaVarId}};
    ///
    /// let v0 = SsaVarId::from_index(0);
    /// let v1 = SsaVarId::from_index(1);
    /// let mut block = SsaBlock::<MockTarget>::new(0);
    /// block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy { dest: v1, src: v0 }));
    /// block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
    ///     dest: v0,
    ///     value: ConstValue::I32(1),
    /// }));
    ///
    /// assert!(block.sort_instructions_topologically());
    /// assert!(matches!(block.instruction(0).unwrap().op(), SsaOp::Const { .. }));
    /// ```
    pub fn sort_instructions_topologically(&mut self) -> bool {
        if self.instructions.len() <= 1 {
            return true;
        }

        // IMPORTANT: Terminators must always be at the end of the block.
        // Extract terminator instructions first, sort non-terminators, then append terminators.
        // This prevents the sorting algorithm from moving terminators to the middle.
        let mut terminators: Vec<(usize, SsaInstruction<T>)> = Vec::new();
        let mut non_terminator_indices: Vec<usize> = Vec::new();

        for (idx, instr) in self.instructions.iter().enumerate() {
            if instr.is_terminator() {
                terminators.push((idx, instr.clone()));
            } else {
                non_terminator_indices.push(idx);
            }
        }

        // If all instructions are terminators or there's nothing to sort, we're done
        if non_terminator_indices.is_empty() {
            return true;
        }

        // Build map of var_id -> instruction index that defines it (within this block)
        // Only for non-terminator instructions
        let mut def_index: HashMap<SsaVarId, usize> = HashMap::new();
        for &idx in &non_terminator_indices {
            let Some(instr) = self.instructions.get(idx) else {
                continue;
            };
            for dest in instr.defs() {
                def_index.insert(dest, idx);
            }
        }

        // Also include phi node definitions as "available from the start"
        // Find max variable index for BitSet sizing
        let max_phi_var = self
            .phi_nodes
            .iter()
            .map(|phi| phi.result().index())
            .max()
            .map_or(0, |m| m.saturating_add(1));
        let mut phi_defs = BitSet::new(max_phi_var);
        for phi in &self.phi_nodes {
            phi_defs.insert(phi.result().index());
        }

        // Build dependency graph for non-terminator instructions only
        // Map from original index to position in non_terminator_indices
        let idx_to_pos: HashMap<usize, usize> = non_terminator_indices
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();

        let n = non_terminator_indices.len();
        let mut deps: Vec<BitSet> = (0..n).map(|_| BitSet::new(n)).collect();
        let mut rdeps: Vec<BitSet> = (0..n).map(|_| BitSet::new(n)).collect();

        // Track the previous side-effecting instruction position to preserve ordering.
        // Side-effecting operations (Call, CallVirt, Stfld, etc.) must execute in their
        // original order to preserve program semantics (I/O ordering, memory effects).
        let mut prev_side_effect_pos: Option<usize> = None;

        for (pos, &idx) in non_terminator_indices.iter().enumerate() {
            let Some(instr) = self.instructions.get(idx) else {
                continue;
            };

            // Add data dependencies (def-use chains)
            instr.for_each_use(|used| {
                // Skip if defined by phi (always available)
                if used.index() < phi_defs.len() && phi_defs.contains(used.index()) {
                    return;
                }
                // Skip if not defined in this block
                if let Some(&dep_idx) = def_index.get(&used) {
                    if dep_idx != idx {
                        if let Some(&dep_pos) = idx_to_pos.get(&dep_idx) {
                            // instruction at pos depends on instruction at dep_pos
                            if let Some(d) = deps.get_mut(pos) {
                                d.insert(dep_pos);
                            }
                            if let Some(r) = rdeps.get_mut(dep_pos) {
                                r.insert(pos);
                            }
                        }
                    }
                }
            });

            // Add ordering dependency for side-effecting operations.
            // Each side-effecting instruction depends on the previous one to preserve
            // the original execution order of operations like Console.WriteLine calls.
            if !instr.op().is_pure() {
                if let Some(prev_pos) = prev_side_effect_pos {
                    // This side-effecting instruction depends on the previous one
                    if let Some(d) = deps.get_mut(pos) {
                        d.insert(prev_pos);
                    }
                    if let Some(r) = rdeps.get_mut(prev_pos) {
                        r.insert(pos);
                    }
                }
                prev_side_effect_pos = Some(pos);
            }
        }

        // Kahn's algorithm: process instructions with no unsatisfied dependencies
        let mut in_degree: Vec<usize> = deps.iter().map(BitSet::count).collect();
        let mut ready: VecDeque<usize> = VecDeque::new();

        // Find instructions with no dependencies (in_degree == 0)
        // Process in original order for stability
        for (pos, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                ready.push_back(pos);
            }
        }

        let mut sorted_positions: Vec<usize> = Vec::with_capacity(n);
        while let Some(pos) = ready.pop_front() {
            sorted_positions.push(pos);

            // Reduce in_degree for dependents
            let Some(rd) = rdeps.get(pos) else {
                continue;
            };
            for dep_pos in rd.iter() {
                if let Some(slot) = in_degree.get_mut(dep_pos) {
                    *slot = slot.saturating_sub(1);
                    if *slot == 0 {
                        ready.push_back(dep_pos);
                    }
                }
            }
        }

        // Check for cycles
        if sorted_positions.len() != n {
            // Cycle detected - this shouldn't happen in valid SSA
            // Leave the block unchanged and return false
            return false;
        }

        // Reorder instructions: non-terminators in sorted order, then terminators at end
        let mut temp: Vec<Option<SsaInstruction<T>>> =
            self.instructions.drain(..).map(Some).collect();

        // First add non-terminator instructions in sorted order
        for pos in sorted_positions {
            let Some(&original_idx) = non_terminator_indices.get(pos) else {
                continue;
            };
            if let Some(instr) = temp.get_mut(original_idx).and_then(Option::take) {
                self.instructions.push(instr);
            }
        }

        // Then add terminators at the end (in their original relative order)
        // Sort terminators by their original index to preserve order
        terminators.sort_by_key(|(idx, _)| *idx);
        for (_, instr) in terminators {
            self.instructions.push(instr);
        }

        true
    }
}

impl<T: Target> fmt::Display for SsaBlock<T>
where
    SsaInstruction<T>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "B{}:", self.id)?;

        for phi in &self.phi_nodes {
            writeln!(f, "  {phi}")?;
        }

        for instr in &self.instructions {
            writeln!(f, "  {instr}")?;
        }

        Ok(())
    }
}
