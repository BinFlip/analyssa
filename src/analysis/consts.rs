//! Constant evaluation for SSA operations.
//!
//! This module provides unified constant folding capabilities for SSA analysis.
//! The [`ConstEvaluator`] can be used by multiple passes (unflattening, decryption,
//! SCCP, etc.) to evaluate SSA operations to constant values.
//!
//! # Algorithm
//!
//! Evaluation is demand-driven and depth-limited:
//!
//! 1. **Cache check**: Results are memoized per variable for O(1) repeated queries.
//! 2. **Cycle detection**: A `BitSet` tracks variables currently being evaluated.
//!    If a variable is encountered while already on the recursion stack, evaluation
//!    returns `None` (avoiding infinite recursion through phi cycles).
//! 3. **Depth limiting**: A configurable maximum recursion depth (default 20) prevents
//!    stack overflow on long dependency chains.
//! 4. **Operation dispatch**: `evaluate_const_op` handles all pure arithmetic, bitwise,
//!    comparison, overflow-checked, and conversion operations by resolving operands
//!    through a caller-provided closure.
//!
//! ## Operations Handled
//!
//! - Arithmetic: Add, Sub, Mul, Div, Rem, Neg
//! - Bitwise: And, Or, Xor, Not, Shl, Shr
//! - Comparisons: Ceq, Clt, Cgt (signed + unsigned)
//! - Overflow-checked: AddOvf, SubOvf, MulOvf
//! - Type conversions: Conv (checked and unchecked)
//! - Copy propagation: trace-through to source
//! - Constants: direct return of Const value
//!
//! ## Operations NOT Handled
//!
//! - Calls, loads, stores, and other side-effecting operations always return `None`.
//! - `Copy` is handled by the evaluator wrapper but not by the shared `evaluate_const_op`
//!   function (callers must handle it first).
//!
//! # Complexity
//!
//! O(d) per variable where d is the depth of the definition chain (bounded by
//! `max_depth`). Caching ensures each variable is evaluated at most once.
//!
//! # Example
//!
//! ```rust
//! use analyssa::{analysis::ConstEvaluator, ir::{ConstValue, SsaVarId}, testing, PointerSize};
//!
//! let ssa = testing::const_i32_return(42);
//! let mut evaluator = ConstEvaluator::new(&ssa, PointerSize::Bit64);
//! let state_var = SsaVarId::from_index(0);
//!
//! // Inject known values from external analysis
//! evaluator.set_known(state_var, ConstValue::I32(42));
//!
//! // Evaluate a variable
//! if let Some(value) = evaluator.evaluate_var(state_var) {
//!     println!("Variable evaluates to: {:?}", value);
//! }
//!
//! // Get all computed constants
//! let constants = evaluator.into_results();
//! ```

use std::collections::HashMap;

use crate::{
    ir::{function::SsaFunction, ops::SsaOp, value::ConstValue, variable::SsaVarId},
    target::Target,
    BitSet, PointerSize,
};

/// Evaluates SSA operations to constant values.
///
/// This provides a unified implementation of constant folding that can be
/// used by multiple passes (unflattening, decryption, SCCP, etc.).
///
/// # Features
///
/// - Caches results for efficiency
/// - Detects cycles to prevent infinite recursion
/// - Supports injecting known values from external analysis
/// - Configurable depth limit
pub struct ConstEvaluator<'a, T: Target> {
    /// Reference to the SSA function being analyzed.
    ssa: &'a SsaFunction<T>,

    /// Cache of evaluated constants.
    /// `Some(value)` means the variable evaluates to that constant.
    /// `None` means the variable was evaluated but is not constant.
    cache: HashMap<SsaVarId, Option<ConstValue<T>>>,

    /// Variables currently being evaluated (for cycle detection).
    visiting: BitSet,

    /// Maximum recursion depth.
    max_depth: usize,

    /// Target pointer size for native int/uint masking.
    pointer_size: PointerSize,
}

impl<'a, T: Target> ConstEvaluator<'a, T> {
    /// Default maximum recursion depth.
    const DEFAULT_MAX_DEPTH: usize = 20;

    /// Creates a new evaluator with default settings.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to evaluate.
    /// * `ptr_size` - Target pointer size for native int/uint masking.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>, ptr_size: PointerSize) -> Self {
        Self::with_max_depth(ssa, Self::DEFAULT_MAX_DEPTH, ptr_size)
    }

    /// Creates an evaluator with a custom depth limit.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to evaluate.
    /// * `max_depth` - Maximum recursion depth for evaluation.
    /// * `ptr_size` - Target pointer size for native int/uint masking.
    #[must_use]
    pub fn with_max_depth(
        ssa: &'a SsaFunction<T>,
        max_depth: usize,
        ptr_size: PointerSize,
    ) -> Self {
        Self {
            ssa,
            cache: HashMap::new(),
            visiting: BitSet::new(ssa.variable_count().max(1)),
            max_depth,
            pointer_size: ptr_size,
        }
    }

    /// Injects a known value from external analysis.
    ///
    /// This allows passes to provide values discovered through other means
    /// (e.g., from `ctx.known_values` in decryption). Injected values take
    /// precedence over computed values.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to set.
    /// * `value` - The known constant value.
    pub fn set_known(&mut self, var: SsaVarId, value: ConstValue<T>) {
        self.cache.insert(var, Some(value));
    }

    /// Evaluates a variable to a constant if possible.
    ///
    /// Results are cached, so repeated calls with the same variable are O(1).
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to evaluate.
    ///
    /// # Returns
    ///
    /// The constant value if the variable can be evaluated, `None` otherwise.
    pub fn evaluate_var(&mut self, var: SsaVarId) -> Option<ConstValue<T>> {
        self.evaluate_var_depth(var, 0)
    }

    /// Internal evaluation with depth tracking.
    fn evaluate_var_depth(&mut self, var: SsaVarId, depth: usize) -> Option<ConstValue<T>> {
        // Check depth limit
        if depth > self.max_depth {
            return None;
        }

        // Check cache first
        if let Some(cached) = self.cache.get(&var) {
            return cached.clone();
        }

        // Cycle detection
        if var.index() < self.visiting.len() && self.visiting.contains(var.index()) {
            return None;
        }

        // Mark as visiting
        if var.index() < self.visiting.len() {
            self.visiting.insert(var.index());
        }

        // Get definition and evaluate
        let result = self
            .ssa
            .get_definition(var)
            .and_then(|op| self.evaluate_op_depth(op, depth));

        // Remove from visiting set
        if var.index() < self.visiting.len() {
            self.visiting.remove(var.index());
        }

        // Cache the result
        self.cache.insert(var, result.clone());

        result
    }

    /// Evaluates an SSA operation to a constant if possible.
    ///
    /// # Arguments
    ///
    /// * `op` - The SSA operation to evaluate.
    ///
    /// # Returns
    ///
    /// The constant value if the operation can be evaluated, `None` otherwise.
    pub fn evaluate_op(&mut self, op: &SsaOp<T>) -> Option<ConstValue<T>> {
        self.evaluate_op_depth(op, 0)
    }

    /// Internal operation evaluation with depth tracking.
    fn evaluate_op_depth(&mut self, op: &SsaOp<T>, depth: usize) -> Option<ConstValue<T>> {
        // Check depth limit
        if depth > self.max_depth {
            return None;
        }

        // Copy needs recursive evaluation that the shared helper cannot provide,
        // because it resolves a variable rather than performing arithmetic.
        if let SsaOp::Copy { src, .. } = op {
            return self.evaluate_var_depth(*src, depth.saturating_add(1));
        }

        let ptr_size = self.pointer_size;
        evaluate_const_op(
            op,
            |var| self.evaluate_var_depth(var, depth.saturating_add(1)),
            ptr_size,
        )
    }

    /// Returns all computed constants.
    ///
    /// This consumes the evaluator and returns a map of all variables
    /// that were successfully evaluated to constants.
    #[must_use]
    pub fn into_results(self) -> HashMap<SsaVarId, ConstValue<T>> {
        self.cache
            .into_iter()
            .filter_map(|(var, opt)| opt.map(|val| (var, val)))
            .collect()
    }

    /// Returns a reference to the SSA function being evaluated.
    #[must_use]
    pub fn ssa(&self) -> &SsaFunction<T> {
        self.ssa
    }

    /// Clears the evaluation cache.
    ///
    /// This is useful if the SSA function has been modified and
    /// cached results are no longer valid.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Evaluates an SSA operation to a constant value using the provided operand resolver.
///
/// This is the shared arithmetic dispatch for constant evaluation. It handles all
/// pure arithmetic, bitwise, comparison, overflow-checked, and conversion operations.
/// Callers provide a `get_const` closure that resolves an [`SsaVarId`] to its constant
/// value (if known).
///
/// # Operations not handled
///
/// - `Copy` — requires variable-level resolution (trace-through), not arithmetic.
///   Callers should handle `Copy` before calling this function.
/// - Calls, loads, stores, and other side-effecting operations — always returns `None`.
///
/// # Arguments
///
/// * `op` - The SSA operation to evaluate.
/// * `get_const` - Closure that resolves a variable to its constant value.
/// * `ptr_size` - Target pointer size for native int/uint masking.
///
/// # Returns
///
/// The constant result if all operands resolve and the operation succeeds, `None` otherwise.
pub fn evaluate_const_op<T: Target>(
    op: &SsaOp<T>,
    mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
    ptr_size: PointerSize,
) -> Option<ConstValue<T>> {
    match op {
        SsaOp::Const { value, .. } => Some(value.clone()),

        // Binary arithmetic
        SsaOp::Add { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.add(&r, ptr_size)
        }
        SsaOp::Sub { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.sub(&r, ptr_size)
        }
        SsaOp::Mul { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.mul(&r, ptr_size)
        }
        SsaOp::Div { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.div(&r, ptr_size)
        }
        SsaOp::Rem { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.rem(&r, ptr_size)
        }

        // Bitwise
        SsaOp::Xor { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.bitwise_xor(&r, ptr_size)
        }
        SsaOp::And { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.bitwise_and(&r, ptr_size)
        }
        SsaOp::Or { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.bitwise_or(&r, ptr_size)
        }

        // Shifts
        SsaOp::Shl { value, amount, .. } => {
            let v = get_const(*value)?;
            let a = get_const(*amount)?;
            v.shl(&a, ptr_size)
        }
        SsaOp::Shr {
            value,
            amount,
            unsigned,
            ..
        } => {
            let v = get_const(*value)?;
            let a = get_const(*amount)?;
            v.shr(&a, *unsigned, ptr_size)
        }

        // Unary
        SsaOp::Neg { operand, .. } => {
            let v = get_const(*operand)?;
            v.negate(ptr_size)
        }
        SsaOp::Not { operand, .. } => {
            let v = get_const(*operand)?;
            v.bitwise_not(ptr_size)
        }

        // Comparisons
        SsaOp::Ceq { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.ceq(&r)
        }
        SsaOp::Clt {
            left,
            right,
            unsigned,
            ..
        } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            if *unsigned {
                l.clt_un(&r)
            } else {
                l.clt(&r)
            }
        }
        SsaOp::Cgt {
            left,
            right,
            unsigned,
            ..
        } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            if *unsigned {
                l.cgt_un(&r)
            } else {
                l.cgt(&r)
            }
        }

        // Overflow-checked arithmetic
        SsaOp::AddOvf {
            left,
            right,
            unsigned,
            ..
        } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.add_checked(&r, *unsigned, ptr_size)
        }
        SsaOp::SubOvf {
            left,
            right,
            unsigned,
            ..
        } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.sub_checked(&r, *unsigned, ptr_size)
        }
        SsaOp::MulOvf {
            left,
            right,
            unsigned,
            ..
        } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            l.mul_checked(&r, *unsigned, ptr_size)
        }

        // Type conversions. Integer→integer and float→integer carry both the
        // signedness and the (CIL) overflow-check; integer→float carries only
        // signedness; the pointer and float-width conversions carry neither.
        SsaOp::IntConv {
            operand,
            target,
            overflow_check,
            unsigned,
            ..
        }
        | SsaOp::FloatToInt {
            operand,
            target,
            overflow_check,
            unsigned,
            ..
        } => {
            let v = get_const(*operand)?;
            let ptr_bytes = ptr_size.bytes() as u32;
            if *overflow_check {
                v.convert_to_checked(target, *unsigned, ptr_bytes)
            } else {
                v.convert_to(target, *unsigned, ptr_bytes)
            }
        }
        SsaOp::IntToFloat {
            operand,
            target,
            unsigned,
            ..
        } => {
            let v = get_const(*operand)?;
            v.convert_to(target, *unsigned, ptr_size.bytes() as u32)
        }
        // Pointers are unsigned: a pointer-width value must zero-extend, never
        // sign-extend. Passing `false` here would widen a 32-bit `0x8000_0000`
        // to `0xFFFF_FFFF_8000_0000` on a 64-bit target.
        SsaOp::IntToPtr {
            operand, target, ..
        }
        | SsaOp::PtrToInt {
            operand, target, ..
        } => {
            let v = get_const(*operand)?;
            v.convert_to(target, true, ptr_size.bytes() as u32)
        }

        // Float-to-float width change only; source signedness is meaningless.
        SsaOp::FloatConv {
            operand, target, ..
        } => {
            let v = get_const(*operand)?;
            v.convert_to(target, false, ptr_size.bytes() as u32)
        }

        // Rotates
        SsaOp::Rol { value, amount, .. } => {
            let v = get_const(*value)?;
            let a = get_const(*amount)?;
            let shift = a.as_i32()? as u32;
            match v {
                ConstValue::I8(v) => Some(ConstValue::I8(v.rotate_left(shift))),
                ConstValue::I16(v) => Some(ConstValue::I16(v.rotate_left(shift))),
                ConstValue::I32(v) => Some(ConstValue::I32(v.rotate_left(shift))),
                ConstValue::I64(v) => Some(ConstValue::I64(v.rotate_left(shift))),
                ConstValue::U8(v) => Some(ConstValue::U8(v.rotate_left(shift))),
                ConstValue::U16(v) => Some(ConstValue::U16(v.rotate_left(shift))),
                ConstValue::U32(v) => Some(ConstValue::U32(v.rotate_left(shift))),
                ConstValue::U64(v) => Some(ConstValue::U64(v.rotate_left(shift))),
                ConstValue::NativeInt(v) => Some(ConstValue::NativeInt(v.rotate_left(shift))),
                ConstValue::NativeUInt(v) => Some(ConstValue::NativeUInt(v.rotate_left(shift))),
                _ => None,
            }
        }
        SsaOp::Ror { value, amount, .. } => {
            let v = get_const(*value)?;
            let a = get_const(*amount)?;
            let shift = a.as_i32()? as u32;
            match v {
                ConstValue::I8(v) => Some(ConstValue::I8(v.rotate_right(shift))),
                ConstValue::I16(v) => Some(ConstValue::I16(v.rotate_right(shift))),
                ConstValue::I32(v) => Some(ConstValue::I32(v.rotate_right(shift))),
                ConstValue::I64(v) => Some(ConstValue::I64(v.rotate_right(shift))),
                ConstValue::U8(v) => Some(ConstValue::U8(v.rotate_right(shift))),
                ConstValue::U16(v) => Some(ConstValue::U16(v.rotate_right(shift))),
                ConstValue::U32(v) => Some(ConstValue::U32(v.rotate_right(shift))),
                ConstValue::U64(v) => Some(ConstValue::U64(v.rotate_right(shift))),
                ConstValue::NativeInt(v) => Some(ConstValue::NativeInt(v.rotate_right(shift))),
                ConstValue::NativeUInt(v) => Some(ConstValue::NativeUInt(v.rotate_right(shift))),
                _ => None,
            }
        }
        SsaOp::Rcl { value, amount, .. } => {
            let v = get_const(*value)?;
            let a = get_const(*amount)?;
            let shift = a.as_i32()? as u32;
            match v {
                ConstValue::I8(v) => Some(ConstValue::I8(v.rotate_left(shift))),
                ConstValue::I16(v) => Some(ConstValue::I16(v.rotate_left(shift))),
                ConstValue::I32(v) => Some(ConstValue::I32(v.rotate_left(shift))),
                ConstValue::I64(v) => Some(ConstValue::I64(v.rotate_left(shift))),
                ConstValue::U8(v) => Some(ConstValue::U8(v.rotate_left(shift))),
                ConstValue::U16(v) => Some(ConstValue::U16(v.rotate_left(shift))),
                ConstValue::U32(v) => Some(ConstValue::U32(v.rotate_left(shift))),
                ConstValue::U64(v) => Some(ConstValue::U64(v.rotate_left(shift))),
                ConstValue::NativeInt(v) => Some(ConstValue::NativeInt(v.rotate_left(shift))),
                ConstValue::NativeUInt(v) => Some(ConstValue::NativeUInt(v.rotate_left(shift))),
                _ => None,
            }
        }
        SsaOp::Rcr { value, amount, .. } => {
            let v = get_const(*value)?;
            let a = get_const(*amount)?;
            let shift = a.as_i32()? as u32;
            match v {
                ConstValue::I8(v) => Some(ConstValue::I8(v.rotate_right(shift))),
                ConstValue::I16(v) => Some(ConstValue::I16(v.rotate_right(shift))),
                ConstValue::I32(v) => Some(ConstValue::I32(v.rotate_right(shift))),
                ConstValue::I64(v) => Some(ConstValue::I64(v.rotate_right(shift))),
                ConstValue::U8(v) => Some(ConstValue::U8(v.rotate_right(shift))),
                ConstValue::U16(v) => Some(ConstValue::U16(v.rotate_right(shift))),
                ConstValue::U32(v) => Some(ConstValue::U32(v.rotate_right(shift))),
                ConstValue::U64(v) => Some(ConstValue::U64(v.rotate_right(shift))),
                ConstValue::NativeInt(v) => Some(ConstValue::NativeInt(v.rotate_right(shift))),
                ConstValue::NativeUInt(v) => Some(ConstValue::NativeUInt(v.rotate_right(shift))),
                _ => None,
            }
        }

        // Byte/bit manipulation
        SsaOp::BSwap { src, .. } => {
            let v = get_const(*src)?;
            match v {
                ConstValue::I16(v) => Some(ConstValue::I16(v.swap_bytes())),
                ConstValue::U16(v) => Some(ConstValue::U16(v.swap_bytes())),
                ConstValue::I32(v) => Some(ConstValue::I32(v.swap_bytes())),
                ConstValue::U32(v) => Some(ConstValue::U32(v.swap_bytes())),
                ConstValue::I64(v) => Some(ConstValue::I64(v.swap_bytes())),
                ConstValue::U64(v) => Some(ConstValue::U64(v.swap_bytes())),
                _ => None,
            }
        }
        SsaOp::BRev { src, .. } => {
            let v = get_const(*src)?;
            match v {
                ConstValue::I8(v) => Some(ConstValue::I8(v.reverse_bits())),
                ConstValue::U8(v) => Some(ConstValue::U8(v.reverse_bits())),
                ConstValue::I16(v) => Some(ConstValue::I16(v.reverse_bits())),
                ConstValue::U16(v) => Some(ConstValue::U16(v.reverse_bits())),
                ConstValue::I32(v) => Some(ConstValue::I32(v.reverse_bits())),
                ConstValue::U32(v) => Some(ConstValue::U32(v.reverse_bits())),
                ConstValue::I64(v) => Some(ConstValue::I64(v.reverse_bits())),
                ConstValue::U64(v) => Some(ConstValue::U64(v.reverse_bits())),
                _ => None,
            }
        }

        // Bit scan
        SsaOp::BitScanForward { src, .. } => {
            let v = get_const(*src)?;
            let bits = match v {
                ConstValue::I8(v) => v.trailing_zeros(),
                ConstValue::U8(v) => v.trailing_zeros(),
                ConstValue::I16(v) => v.trailing_zeros(),
                ConstValue::U16(v) => v.trailing_zeros(),
                ConstValue::I32(v) => v.trailing_zeros(),
                ConstValue::U32(v) => v.trailing_zeros(),
                ConstValue::I64(v) => v.trailing_zeros(),
                ConstValue::U64(v) => v.trailing_zeros(),
                _ => return None,
            };
            Some(ConstValue::I32(bits as i32))
        }
        SsaOp::BitScanReverse { src, .. } => {
            let v = get_const(*src)?;
            let bits = match v {
                ConstValue::I8(v) => 7u32.checked_sub(v.leading_zeros())?,
                ConstValue::U8(v) => 7u32.checked_sub(v.leading_zeros())?,
                ConstValue::I16(v) => 15u32.checked_sub(v.leading_zeros())?,
                ConstValue::U16(v) => 15u32.checked_sub(v.leading_zeros())?,
                ConstValue::I32(v) => 31u32.checked_sub(v.leading_zeros())?,
                ConstValue::U32(v) => 31u32.checked_sub(v.leading_zeros())?,
                ConstValue::I64(v) => 63u32.checked_sub(v.leading_zeros())?,
                ConstValue::U64(v) => 63u32.checked_sub(v.leading_zeros())?,
                _ => return None,
            };
            Some(ConstValue::U32(bits))
        }

        // Population count and parity
        SsaOp::Popcount { src, .. } => {
            let v = get_const(*src)?;
            let count = match v {
                ConstValue::I8(v) => v.count_ones(),
                ConstValue::U8(v) => v.count_ones(),
                ConstValue::I16(v) => v.count_ones(),
                ConstValue::U16(v) => v.count_ones(),
                ConstValue::I32(v) => v.count_ones(),
                ConstValue::U32(v) => v.count_ones(),
                ConstValue::I64(v) => v.count_ones(),
                ConstValue::U64(v) => v.count_ones(),
                _ => return None,
            };
            Some(ConstValue::I32(count as i32))
        }
        SsaOp::Parity { src, .. } => {
            let v = get_const(*src)?;
            let parity = match v {
                ConstValue::I8(v) => v.count_ones() % 2,
                ConstValue::U8(v) => v.count_ones() % 2,
                ConstValue::I16(v) => v.count_ones() % 2,
                ConstValue::U16(v) => v.count_ones() % 2,
                ConstValue::I32(v) => v.count_ones() % 2,
                ConstValue::U32(v) => v.count_ones() % 2,
                ConstValue::I64(v) => v.count_ones() % 2,
                ConstValue::U64(v) => v.count_ones() % 2,
                _ => return None,
            };
            Some(ConstValue::I32(parity as i32))
        }

        // Conditional select
        SsaOp::Select {
            condition,
            true_val,
            false_val,
            ..
        } => {
            let cond = get_const(*condition)?;
            match cond.as_i64() {
                Some(0) => get_const(*false_val),
                Some(_) => get_const(*true_val),
                None => None,
            }
        }

        // All other operations cannot be evaluated to constants
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        ir::{block::SsaBlock, instruction::SsaInstruction, variable::DefSite, VariableOrigin},
        testing,
        testing::{MockTarget, MockType},
    };

    type Cv = ConstValue<MockTarget>;

    fn var(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }

    fn resolve(id: SsaVarId) -> Option<Cv> {
        match id.index() {
            0 => Some(Cv::I32(10)),
            1 => Some(Cv::I32(3)),
            2 => Some(Cv::I32(-1)),
            3 => Some(Cv::I32(i32::MAX)),
            _ => None,
        }
    }

    #[test]
    fn const_evaluator_resolves_consts_and_injected_values() {
        let ssa = testing::const_i32_return(42);
        let mut evaluator = ConstEvaluator::new(&ssa, PointerSize::Bit64);

        assert_eq!(evaluator.ssa().block_count(), 1);
        assert_eq!(evaluator.evaluate_var(var(0)), Some(Cv::I32(42)));

        evaluator.set_known(var(0), Cv::I32(7));
        assert_eq!(evaluator.evaluate_var(var(0)), Some(Cv::I32(7)));

        evaluator.clear_cache();
        assert_eq!(evaluator.evaluate_var(var(0)), Some(Cv::I32(42)));
        assert_eq!(evaluator.into_results().get(&var(0)), Some(&Cv::I32(42)));
    }

    #[test]
    fn const_evaluator_traces_copy_chains_and_honors_depth_limit() {
        let mut ssa = crate::ir::SsaFunction::<MockTarget>::new(0, 0);
        for i in 0..3 {
            ssa.create_variable(
                VariableOrigin::Local(i),
                0,
                DefSite::instruction(0, usize::from(i)),
                MockType::I32,
            );
        }
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: var(0),
            value: Cv::I32(5),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
            dest: var(1),
            src: var(0),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
            dest: var(2),
            src: var(1),
        }));
        ssa.add_block(block);

        let mut evaluator = ConstEvaluator::new(&ssa, PointerSize::Bit64);
        assert_eq!(evaluator.evaluate_var(var(2)), Some(Cv::I32(5)));

        let mut shallow = ConstEvaluator::with_max_depth(&ssa, 0, PointerSize::Bit64);
        assert_eq!(shallow.evaluate_var(var(2)), None);
    }

    #[test]
    fn evaluate_const_op_folds_arithmetic_bitwise_and_shifts() {
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Add {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(13))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Sub {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(7))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Mul {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(30))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Div {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    unsigned: false,
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(3))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Rem {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    unsigned: false,
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(1))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::And {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(2))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Or {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(11))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Xor {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(9))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Shl {
                    dest: var(9),
                    value: var(1),
                    amount: var(1),
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(24))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Shr {
                    dest: var(9),
                    value: var(2),
                    amount: var(1),
                    unsigned: true,
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(536_870_911))
        );
    }

    #[test]
    fn evaluate_const_op_folds_unary_comparison_and_checked_arithmetic() {
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Neg {
                    dest: var(9),
                    operand: var(1),
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(-3))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Not {
                    dest: var(9),
                    operand: var(1),
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(!3))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Ceq {
                    dest: var(9),
                    left: var(0),
                    right: var(1)
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::False)
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Clt {
                    dest: var(9),
                    left: var(1),
                    right: var(0),
                    unsigned: false
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::True)
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Cgt {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    unsigned: true
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::True)
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::AddOvf {
                    dest: var(9),
                    left: var(3),
                    right: var(1),
                    unsigned: false,
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            None
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::SubOvf {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    unsigned: false,
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(7))
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::MulOvf {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    unsigned: false,
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            Some(Cv::I32(30))
        );
    }

    #[test]
    fn evaluate_const_op_returns_none_for_unknown_inputs_and_unsupported_ops() {
        assert_eq!(
            evaluate_const_op(
                &SsaOp::Add {
                    dest: var(9),
                    left: var(0),
                    right: var(99),
                    flags: None,
                },
                resolve,
                PointerSize::Bit64
            ),
            None
        );
        assert_eq!(
            evaluate_const_op(
                &SsaOp::IntConv {
                    dest: var(9),
                    operand: var(0),
                    target: MockType::I64,
                    overflow_check: false,
                    unsigned: false
                },
                resolve,
                PointerSize::Bit64
            ),
            None
        );
        assert_eq!(
            evaluate_const_op::<MockTarget>(
                &SsaOp::Return {
                    value: Some(var(0))
                },
                resolve,
                PointerSize::Bit64
            ),
            None
        );
    }
}
