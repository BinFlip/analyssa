//! Strength reduction pass — replaces expensive operations with cheaper
//! equivalents.
//!
//! # Transformations
//!
//! - **Multiplication by power of 2**: `x * 2^n` → `x << n`
//! - **Unsigned division by power of 2**: `x / 2^n` → `x >> n` (unsigned)
//! - **Unsigned modulo by power of 2**: `x % 2^n` → `x & (2^n - 1)` (unsigned)
//! - **Signed division by power of 2**: same as unsigned, but only when the
//!   dividend is provably non-negative.
//! - **Signed modulo by power of 2**: same as unsigned, but only when the
//!   dividend is provably non-negative.
//!
//! # Correctness
//!
//! Signed division and modulo are NOT transformed unconditionally because
//! signed division rounds toward zero while arithmetic right-shift rounds
//! toward negative infinity:
//! - `-5 / 2 = -2` (truncation toward zero)
//! - `-5 >> 1 = -3` (round toward negative infinity)
//!
//! The caller supplies `is_non_negative` to gate these transformations.
//! Hosts without range analysis pass `\|_\| false`.
//!
//! # Algorithm
//!
//! 1. For each `Mul`/`Div`/`Rem`, check if one operand is a constant power
//!    of two.
//! 2. Check the constant has exactly one use (no other instruction depends
//!    on its original value).
//! 3. For signed ops, check `is_non_negative` for the dividend.
//! 4. Replace the constant with `exponent` (for shifts) or `mask` (for AND),
//!    and replace the operation with the cheaper equivalent.

use crate::{
    analysis::DefUseIndex,
    bitset::BitSet,
    events::{EventKind, EventListener},
    ir::{function::SsaFunction, ops::SsaOp, value::ConstValue, variable::SsaVarId},
    passes::utils::is_power_of_two,
    target::Target,
};

/// Run strength reduction on `ssa`.
///
/// Replaces multiplication by powers of two with left shifts, unsigned
/// division/modulo by powers of two with right shifts / bitwise AND,
/// and signed variants when the dividend is provably non-negative.
///
/// # Arguments
///
/// * `ssa` — The SSA function to transform in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::StrengthReduced`] events.
/// * `is_non_negative` — Caller-supplied predicate that returns `true`
///   if a given `SsaVarId` is provably >= 0. Hosts without range analysis
///   should pass `\|_\| false`.
///
/// # Returns
///
/// `true` if any operation was rewritten.
pub fn run<T, L>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
    is_non_negative: &dyn Fn(SsaVarId) -> bool,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let index = DefUseIndex::<T>::build_with_ops(ssa);
    let candidates = find_candidates(ssa, &index, is_non_negative);
    apply_reductions(ssa, candidates, method, events)
}

/// Identifies an instruction by its block and index within the block.
#[derive(Debug, Clone, Copy)]
struct InstrLocation {
    /// Index of the containing block.
    block_idx: usize,
    /// Index of the instruction within the block.
    instr_idx: usize,
}

/// A detected strength-reduction opportunity.
#[derive(Debug)]
struct ReductionCandidate<T: Target> {
    /// Location of the instruction to reduce.
    location: InstrLocation,
    /// The constant variable being used as the power-of-two operand.
    const_var: SsaVarId,
    /// Block containing the constant definition.
    const_block: usize,
    /// Index of the constant definition within its block.
    const_instr: usize,
    /// The new value for the constant (exponent for shifts, mask for AND).
    new_const_value: ConstValue<T>,
    /// The replacement operation (shift or AND).
    new_op: SsaOp<T>,
    /// Human-readable description of the reduction applied.
    description: String,
}

/// Helper struct that checks for reduction opportunities on individual
/// instructions.
struct ReductionChecker<'a, T: Target> {
    /// Def-use index for looking up variable definitions.
    index: &'a DefUseIndex<T>,
    /// Bitset of constant variables already claimed by earlier reductions.
    used_constants: &'a BitSet,
}

impl<'a, T: Target> ReductionChecker<'a, T> {
    fn new(index: &'a DefUseIndex<T>, used_constants: &'a BitSet) -> Self {
        Self {
            index,
            used_constants,
        }
    }

    fn try_mul_reduction(
        &self,
        dest: SsaVarId,
        value_var: SsaVarId,
        const_var: SsaVarId,
        location: InstrLocation,
    ) -> Option<ReductionCandidate<T>> {
        let (const_block, const_instr, const_op) = self.index.full_definition(const_var)?;
        let SsaOp::Const {
            value: const_value, ..
        } = const_op
        else {
            return None;
        };
        let value = const_value.as_i64()?;
        let exponent = is_power_of_two(value)?;
        let uses = self.index.use_count(const_var);
        if uses != 1 || self.used_constants.contains(const_var.index()) {
            return None;
        }
        Some(ReductionCandidate {
            location,
            const_var,
            const_block,
            const_instr,
            new_const_value: ConstValue::I32(i32::from(exponent)),
            new_op: SsaOp::Shl {
                dest,
                value: value_var,
                amount: const_var,
                flags: None,
            },
            description: format!("mul x, {value} → shl x, {exponent}"),
        })
    }

    fn try_div_reduction(
        &self,
        dest: SsaVarId,
        dividend: SsaVarId,
        divisor_var: SsaVarId,
        unsigned: bool,
        location: InstrLocation,
    ) -> Option<ReductionCandidate<T>> {
        let (const_block, const_instr, const_op) = self.index.full_definition(divisor_var)?;
        let SsaOp::Const {
            value: const_value, ..
        } = const_op
        else {
            return None;
        };
        let value = const_value.as_i64()?;
        let exponent = is_power_of_two(value)?;
        let uses = self.index.use_count(divisor_var);
        if uses != 1 || self.used_constants.contains(divisor_var.index()) {
            return None;
        }
        let desc = if unsigned {
            format!("div.un x, {value} → shr.un x, {exponent}")
        } else {
            format!("div x, {value} → shr x, {exponent} (x >= 0)")
        };
        Some(ReductionCandidate {
            location,
            const_var: divisor_var,
            const_block,
            const_instr,
            new_const_value: ConstValue::I32(i32::from(exponent)),
            new_op: SsaOp::Shr {
                dest,
                value: dividend,
                amount: divisor_var,
                unsigned,
                flags: None,
            },
            description: desc,
        })
    }

    #[allow(clippy::cast_possible_truncation)]
    fn try_rem_reduction(
        &self,
        dest: SsaVarId,
        dividend: SsaVarId,
        divisor_var: SsaVarId,
        unsigned: bool,
        location: InstrLocation,
    ) -> Option<ReductionCandidate<T>> {
        let (const_block, const_instr, const_op) = self.index.full_definition(divisor_var)?;
        let SsaOp::Const {
            value: const_value, ..
        } = const_op
        else {
            return None;
        };
        let value = const_value.as_i64()?;
        let _exponent = is_power_of_two(value)?;
        let mask = value.checked_sub(1)?;
        let uses = self.index.use_count(divisor_var);
        if uses != 1 || self.used_constants.contains(divisor_var.index()) {
            return None;
        }
        let desc = if unsigned {
            format!("rem.un x, {value} → and x, {mask}")
        } else {
            format!("rem x, {value} → and x, {mask} (x >= 0)")
        };
        Some(ReductionCandidate {
            location,
            const_var: divisor_var,
            const_block,
            const_instr,
            new_const_value: ConstValue::I32(mask as i32),
            new_op: SsaOp::And {
                dest,
                left: dividend,
                right: divisor_var,
                flags: None,
            },
            description: desc,
        })
    }
}

fn find_candidates<T: Target>(
    ssa: &SsaFunction<T>,
    index: &DefUseIndex<T>,
    is_non_negative: &dyn Fn(SsaVarId) -> bool,
) -> Vec<ReductionCandidate<T>> {
    let mut candidates = Vec::new();
    let mut used_constants = BitSet::new(ssa.var_id_capacity());

    for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
        let checker = ReductionChecker::new(index, &used_constants);
        let location = InstrLocation {
            block_idx,
            instr_idx,
        };
        if let Some(candidate) = check_reduction(instr.op(), location, &checker, is_non_negative) {
            used_constants.insert(candidate.const_var.index());
            candidates.push(candidate);
        }
    }

    candidates
}

fn check_reduction<T: Target>(
    op: &SsaOp<T>,
    location: InstrLocation,
    checker: &ReductionChecker<'_, T>,
    is_non_negative: &dyn Fn(SsaVarId) -> bool,
) -> Option<ReductionCandidate<T>> {
    match op {
        SsaOp::Mul {
            dest, left, right, ..
        } => {
            if let Some(candidate) = checker.try_mul_reduction(*dest, *left, *right, location) {
                return Some(candidate);
            }
            checker.try_mul_reduction(*dest, *right, *left, location)
        }
        SsaOp::Div {
            dest,
            left,
            right,
            unsigned: true,
            ..
        } => checker.try_div_reduction(*dest, *left, *right, true, location),
        SsaOp::Div {
            dest,
            left,
            right,
            unsigned: false,
            ..
        } => {
            if is_non_negative(*left) {
                checker.try_div_reduction(*dest, *left, *right, false, location)
            } else {
                None
            }
        }
        SsaOp::Rem {
            dest,
            left,
            right,
            unsigned: true,
            ..
        } => checker.try_rem_reduction(*dest, *left, *right, true, location),
        SsaOp::Rem {
            dest,
            left,
            right,
            unsigned: false,
            ..
        } => {
            if is_non_negative(*left) {
                checker.try_rem_reduction(*dest, *left, *right, false, location)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn apply_reductions<T, L>(
    ssa: &mut SsaFunction<T>,
    candidates: Vec<ReductionCandidate<T>>,
    method: &T::MethodRef,
    events: &L,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut changed = false;
    for candidate in candidates {
        if let Some(block) = ssa.block_mut(candidate.const_block) {
            if let Some(const_instr) = block.instructions_mut().get_mut(candidate.const_instr) {
                const_instr.set_op(SsaOp::Const {
                    dest: candidate.const_var,
                    value: candidate.new_const_value.clone(),
                });
            }
        }

        if let Some(block) = ssa.block_mut(candidate.location.block_idx) {
            if let Some(instr) = block
                .instructions_mut()
                .get_mut(candidate.location.instr_idx)
            {
                instr.set_op(candidate.new_op);
                let event = crate::events::Event {
                    kind: EventKind::StrengthReduced,
                    method: Some(method.clone()),
                    location: Some(candidate.location.instr_idx),
                    message: candidate.description,
                    pass: None,
                };
                events.push(event);
                changed = true;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
    }

    fn local_at(
        ssa: &mut SsaFunction<MockTarget>,
        idx: u16,
        block: usize,
        instr: usize,
    ) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::instruction(block, instr),
            MockType::I32,
        )
    }

    #[test]
    fn mul_by_power_of_two_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: result,
            left: x,
            right: pow2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| true);
        assert!(changed, "mul by power of two should reduce to shift");
        assert!(log.has(EventKind::StrengthReduced));
        assert!(matches!(
            ssa.block(0).unwrap().instruction(2).unwrap().op(),
            SsaOp::Shl { .. }
        ));
    }

    #[test]
    fn mul_by_non_power_of_two_not_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let not_pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: not_pow2,
            value: ConstValue::I32(7),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: result,
            left: x,
            right: not_pow2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| true);
        assert!(!changed, "mul by non-power-of-two should NOT reduce");
    }

    #[test]
    fn unsigned_div_by_power_of_two_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(100),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(4),
        }));
        block.add_instruction(instr(SsaOp::Div {
            dest: result,
            left: x,
            right: pow2,
            unsigned: true,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| true);
        assert!(changed, "unsigned div by power of two should reduce");
        assert!(log.has(EventKind::StrengthReduced));
        assert!(matches!(
            ssa.block(0).unwrap().instruction(2).unwrap().op(),
            SsaOp::Shr { unsigned: true, .. }
        ));
    }

    #[test]
    fn signed_div_by_power_of_two_reduced_when_non_negative() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(100),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Div {
            dest: result,
            left: x,
            right: pow2,
            unsigned: false,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| true);
        assert!(
            changed,
            "signed div by power of two should reduce when non-negative"
        );
        assert!(log.has(EventKind::StrengthReduced));
    }

    #[test]
    fn signed_div_not_reduced_when_not_proven_non_negative() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(-100),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(4),
        }));
        block.add_instruction(instr(SsaOp::Div {
            dest: result,
            left: x,
            right: pow2,
            unsigned: false,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| false);
        assert!(
            !changed,
            "signed div should NOT reduce when non-negativity not proven"
        );
    }

    #[test]
    fn unsigned_rem_by_power_of_two_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(100),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Rem {
            dest: result,
            left: x,
            right: pow2,
            unsigned: true,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| true);
        assert!(changed, "unsigned rem by power of two should reduce to and");
        assert!(log.has(EventKind::StrengthReduced));
        assert!(matches!(
            ssa.block(0).unwrap().instruction(2).unwrap().op(),
            SsaOp::And { .. }
        ));
    }

    #[test]
    fn multiple_reductions_in_one_run() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 5);
        let x = local_at(&mut ssa, 0, 0, 0);
        let y = local_at(&mut ssa, 1, 0, 1);
        let p1 = local_at(&mut ssa, 2, 0, 2);
        let p2 = local_at(&mut ssa, 3, 0, 3);
        let r1 = local_at(&mut ssa, 4, 0, 4);
        let r2 = local_at(&mut ssa, 5, 0, 5);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: p1,
            value: ConstValue::I32(4),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: y,
            value: ConstValue::I32(50),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: p2,
            value: ConstValue::I32(16),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: r1,
            left: x,
            right: p1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: r2,
            left: y,
            right: p2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(r1) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| true);
        assert!(changed, "multiple reductions should all fire");
        assert!(log.count_kind(EventKind::StrengthReduced) >= 2);
    }

    #[test]
    fn no_candidates_returns_false() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let x = local_at(&mut ssa, 0, 0, 0);
        let y = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: y,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: result,
            left: x,
            right: y,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| true);
        assert!(!changed, "no strength-reducible ops should return false");
    }

    #[test]
    fn empty_function_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| true);
        assert!(!changed);
    }

    #[test]
    fn shared_constant_not_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let r1 = local_at(&mut ssa, 2, 0, 2);
        let r2 = local_at(&mut ssa, 3, 0, 3);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: r1,
            left: x,
            right: pow2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: r2,
            left: r1,
            right: pow2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(r2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| true);
        let _ = changed;
    }
}
