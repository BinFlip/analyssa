//! SSA instruction wrapper with explicit def/use metadata.
//!
//! Unlike stack-based CIL where operands are implicit on the evaluation stack,
//! SSA instructions have explicit operands (uses) and results (defs). Each
//! [`SsaInstruction`] wraps a decomposed [`SsaOp`] along with the original
//! host instruction and optional resolved result type.
//!
//! # Dual Representation
//!
//! Each SSA instruction stores two representations:
//!
//! 1. **Original instruction** (`T::OriginalInstruction`): The raw host instruction
//!    (e.g., a CIL opcode with its operands). Retained for debugging, source mapping,
//!    and code generation to produce faithful output.
//!
//! 2. **Decomposed SSA operation** ([`SsaOp`]): The clean `result = op(operands)` form.
//!    This is the primary representation for analysis and optimization passes.
//!
//! This dual representation enables:
//! - Direct def-use chain construction from `SsaOp::defs()` / `SsaOp::uses()`
//! - Dead code elimination (variable with no uses)
//! - Pattern matching on decomposed operations for optimization
//! - Faithful code generation from the original instruction where needed
//!
//! # Result Type Resolution
//!
//! The `result_type` field captures the precise type available during initial SSA
//! construction (when full assembly metadata is available). This type survives
//! through deobfuscation transforms and is used by rebuild and codegen to recover
//! types that cannot be inferred structurally from the op alone (e.g., call return
//! types, field types, argument/local types).

use std::fmt;

use crate::{
    ir::{ops::SsaOp, variable::SsaVarId},
    target::Target,
};

/// An SSA instruction wrapping a decomposed operation and original host instruction.
///
/// Combines the decomposed [`SsaOp`] (the canonical representation for analysis)
/// with the original host instruction (for debugging and faithful code generation)
/// and an optional resolved result type.
///
/// # Examples
///
/// ```rust
/// use analyssa::{ir::{SsaInstruction, SsaOp, SsaVarId}, MockTarget};
///
/// let left = SsaVarId::from_index(0);
/// let right = SsaVarId::from_index(1);
/// let result = SsaVarId::from_index(2);
/// let instr: SsaInstruction<MockTarget> = SsaInstruction::new(
///     <MockTarget as analyssa::Target>::synthetic_instruction(),
///     SsaOp::Add { dest: result, left, right, flags: None },
/// );
/// ```
#[derive(Debug, Clone)]
pub struct SsaInstruction<T: Target> {
    /// The original host instruction retained for debugging, source mapping,
    /// and code generation. Hosts can store a decoded instruction, source
    /// location, RVA, or `()` for synthetic instructions.
    original: T::OriginalInstruction,

    /// The decomposed SSA operation in `result = op(operands)` form.
    ///
    /// This is the authoritative representation used by all analysis and
    /// optimization passes. All data dependencies are explicit in the
    /// operation's `defs()` and `uses()`.
    op: SsaOp<T>,

    /// Optional resolved result type captured during SSA construction.
    ///
    /// Set when the SSA converter has access to full assembly metadata
    /// (TypeContext). Survives deobfuscation transforms. Used by rebuild
    /// and codegen to recover types that cannot be inferred structurally
    /// from the SSA op alone: call return types, field types, argument
    /// and local types, etc.
    result_type: Option<T::Type>,
}

impl<T: Target> SsaInstruction<T> {
    /// Creates a new SSA instruction with a decomposed operation.
    ///
    /// # Arguments
    ///
    /// * `original` - The original instruction
    /// * `op` - The decomposed SSA operation
    #[must_use]
    pub fn new(original: T::OriginalInstruction, op: SsaOp<T>) -> Self {
        Self {
            original,
            op,
            result_type: None,
        }
    }

    /// Creates an SSA instruction with only a decomposed operation (no original
    /// instruction breadcrumb).
    ///
    /// This is useful for synthetic instructions like phi nodes that don't
    /// correspond to any source-level instruction. The placeholder original
    /// is supplied by `Target::synthetic_instruction()`.
    #[must_use]
    pub fn synthetic(op: SsaOp<T>) -> Self {
        Self {
            original: T::synthetic_instruction(),
            op,
            result_type: None,
        }
    }

    /// Returns a reference to the original instruction.
    #[must_use]
    pub const fn original(&self) -> &T::OriginalInstruction {
        &self.original
    }

    /// Returns the decomposed SSA operation.
    #[must_use]
    pub const fn op(&self) -> &SsaOp<T> {
        &self.op
    }

    /// Returns a mutable reference to the decomposed SSA operation.
    pub fn op_mut(&mut self) -> &mut SsaOp<T> {
        &mut self.op
    }

    /// Sets the decomposed SSA operation.
    ///
    /// Clears `result_type` because the new op may have a different result type.
    /// Callers that know the type should call `set_result_type()` afterwards.
    pub fn set_op(&mut self, op: SsaOp<T>) {
        self.op = op;
        self.result_type = None;
    }

    /// Returns the resolved result type, if set during SSA construction.
    #[must_use]
    pub fn result_type(&self) -> Option<&T::Type> {
        self.result_type.as_ref()
    }

    /// Sets the resolved result type.
    pub fn set_result_type(&mut self, ty: Option<T::Type>) {
        self.result_type = ty;
    }

    /// Builder pattern: sets the result type and returns self.
    #[must_use]
    pub fn with_result_type(mut self, ty: T::Type) -> Self {
        self.result_type = Some(ty);
        self
    }

    /// Returns `true` if this instruction is a terminator.
    ///
    /// Terminators are instructions that end a basic block (jumps, branches, returns, throws).
    #[must_use]
    pub fn is_terminator(&self) -> bool {
        self.op.is_terminator()
    }

    /// Returns `true` if this instruction may throw an exception.
    #[must_use]
    pub fn may_throw(&self) -> bool {
        self.op.may_throw()
    }

    /// Returns `true` if this instruction is pure (has no side effects).
    ///
    /// Pure instructions can be eliminated if their result is unused.
    #[must_use]
    pub fn is_pure(&self) -> bool {
        self.op.is_pure()
    }

    /// Returns the SSA variables used (read) by this instruction.
    #[must_use]
    pub fn uses(&self) -> Vec<SsaVarId> {
        self.op.uses()
    }

    /// Calls `f` for every SSA variable used by this instruction.
    pub fn for_each_use<F>(&self, f: F)
    where
        F: FnMut(SsaVarId),
    {
        self.op.for_each_use(f);
    }

    /// Returns `true` if this instruction uses (reads) the given variable.
    ///
    /// Allocation-free; prefer over `self.uses().contains(&var)` in hot paths.
    #[must_use]
    pub fn uses_var(&self, var: SsaVarId) -> bool {
        self.op.uses_var(var)
    }

    /// Returns the SSA variable defined by this instruction, if any.
    #[must_use]
    pub fn def(&self) -> Option<SsaVarId> {
        self.op.dest()
    }

    /// Returns all SSA variables defined by this instruction.
    pub fn defs(&self) -> impl Iterator<Item = SsaVarId> + '_ {
        self.op.defs()
    }

    /// Returns `true` if this instruction defines a value.
    #[must_use]
    pub fn has_def(&self) -> bool {
        self.op.defs().next().is_some()
    }

    /// Returns `true` if this instruction has no uses.
    #[must_use]
    pub fn has_no_uses(&self) -> bool {
        let mut has_use = false;
        self.op.for_each_use(|_| has_use = true);
        !has_use
    }

    /// Returns all SSA variables referenced by this instruction.
    ///
    /// This includes both uses and the def (if present).
    #[must_use]
    pub fn all_variables(&self) -> Vec<SsaVarId> {
        let mut vars = Vec::new();
        self.op.for_each_use(|var| vars.push(var));
        vars.extend(self.op.defs());
        vars
    }

    /// Calls `f` for every SSA variable referenced (used or defined) by this
    /// instruction, without allocating.
    pub fn for_each_variable<F>(&self, mut f: F)
    where
        F: FnMut(SsaVarId),
    {
        self.op.for_each_use(&mut f);
        for def in self.op.defs() {
            f(def);
        }
    }
}

// Original-instruction accessors. Implementation routes through
// `Target::instruction_mnemonic` / `instruction_rva`, so this works for any
// host that retains a meaningful `OriginalInstruction`.
impl<T: Target> SsaInstruction<T> {
    /// Returns the instruction's mnemonic via `Target::instruction_mnemonic`.
    #[must_use]
    pub fn mnemonic(&self) -> &'static str {
        T::instruction_mnemonic(&self.original)
    }

    /// Returns the instruction's RVA via `Target::instruction_rva`.
    #[must_use]
    pub fn rva(&self) -> u64 {
        T::instruction_rva(&self.original)
    }
}

impl<T: Target> fmt::Display for SsaInstruction<T>
where
    SsaOp<T>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        ir::{ops::SsaOp, value::ConstValue},
        testing::{MockTarget, MockType},
    };

    fn var(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }

    #[test]
    fn instruction_accessors_reflect_operation_shape() {
        let instr = SsaInstruction::<MockTarget>::synthetic(SsaOp::Add {
            dest: var(2),
            left: var(0),
            right: var(1),
            flags: None,
        });

        assert_eq!(instr.original(), &());
        assert!(matches!(instr.op(), SsaOp::Add { .. }));
        assert!(instr.is_pure());
        assert!(!instr.is_terminator());
        assert!(!instr.may_throw());
        assert_eq!(instr.uses(), vec![var(0), var(1)]);
        assert_eq!(instr.def(), Some(var(2)));
        assert!(instr.has_def());
        assert!(!instr.has_no_uses());
        assert_eq!(instr.all_variables(), vec![var(0), var(1), var(2)]);
        assert_eq!(instr.mnemonic(), "<mock>");
        assert_eq!(instr.rva(), 0);
    }

    #[test]
    fn mutating_op_clears_result_type_until_reset() {
        let mut instr = SsaInstruction::<MockTarget>::synthetic(SsaOp::Const {
            dest: var(0),
            value: ConstValue::I32(1),
        })
        .with_result_type(MockType::I32);

        assert_eq!(instr.result_type(), Some(&MockType::I32));

        if let SsaOp::Const { value, .. } = instr.op_mut() {
            *value = ConstValue::I32(2);
        }
        assert_eq!(instr.result_type(), Some(&MockType::I32));

        instr.set_op(SsaOp::Return {
            value: Some(var(0)),
        });
        assert_eq!(instr.result_type(), None);
        assert!(instr.is_terminator());
        assert_eq!(instr.def(), None);
        assert!(!instr.has_def());
        assert_eq!(instr.uses(), vec![var(0)]);

        instr.set_result_type(Some(MockType::Unknown));
        assert_eq!(instr.result_type(), Some(&MockType::Unknown));
        instr.set_result_type(None);
        assert_eq!(instr.result_type(), None);
    }

    #[test]
    fn no_operand_instructions_report_no_uses() {
        let instr = SsaInstruction::<MockTarget>::new((), SsaOp::Return { value: None });

        assert!(instr.has_no_uses());
        assert_eq!(instr.all_variables(), Vec::<SsaVarId>::new());
    }
}
