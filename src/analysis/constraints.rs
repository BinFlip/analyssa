//! Constraint types for SSA path analysis.
//!
//! This module provides constraint types derived from branch conditions during
//! path-aware SSA evaluation. When traversing a specific CFG path, branch
//! conditions impose constraints on variable values that can be used to detect
//! dead code, prune infeasible paths, and improve constant propagation.
//!
//! # Constraint Types
//!
//! - [`Constraint`]: A constraint on a single value (e.g., `x == 5`, `x > 10`)
//! - [`PathConstraint`]: Associates a constraint with a specific SSA variable
//!
//! # Constraint Derivation
//!
//! Constraints are derived from comparison operations at branch points:
//!
//! | Branch Condition | True Path Constraint | False Path Constraint |
//! |------------------|---------------------|-----------------------|
//! | `ceq(x, c)`      | `x == c`            | `x != c`              |
//! | `cgt(x, c)`      | `x > c`             | `x <= c`              |
//! | `clt(x, c)`      | `x < c`             | `x >= c`              |
//!
//! # Conflict Detection
//!
//! Constraints support `conflicts_with()` to detect when two constraints
//! cannot both be true (e.g., `x == 5` conflicts with `x > 10`). This enables
//! detection of infeasible paths for opaque predicate identification.
//!
//! # Usage with SsaEvaluator
//!
//! The [`SsaEvaluator`](super::SsaEvaluator) tracks path constraints during evaluation.
//! When taking a branch, the evaluator records constraints that must hold on that path.
//!
//! For example, after taking the true branch of `if (x == 5)`:
//! - We know `x == 5` on this path
//! - This is recorded as `PathConstraint { variable: x, constraint: Constraint::Equal(ConstValue::I32(5)) }`
//!
//! # Use Cases
//!
//! - Constraint solving with Z3
//! - Value range analysis
//! - Dead code detection
//! - Path-sensitive constant propagation

use crate::{
    ir::{value::ConstValue, variable::SsaVarId},
    target::Target,
    PointerSize,
};

/// A constraint on a variable's value derived from branch conditions.
///
/// When following a specific branch path, we can derive facts about variable values.
/// For example, after taking the true branch of `if (x == 5)`, we know `x == 5`.
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint<T: Target> {
    /// Variable equals a concrete value: `x == value`
    Equal(ConstValue<T>),
    /// Variable does not equal a concrete value: `x != value`
    NotEqual(ConstValue<T>),
    /// Variable is greater than a value (signed): `x > value`
    GreaterThan(ConstValue<T>),
    /// Variable is less than a value (signed): `x < value`
    LessThan(ConstValue<T>),
    /// Variable is greater than or equal (signed): `x >= value`
    GreaterOrEqual(ConstValue<T>),
    /// Variable is less than or equal (signed): `x <= value`
    LessOrEqual(ConstValue<T>),
    /// Variable is greater than (unsigned): `(uint)x > value`
    GreaterThanUnsigned(ConstValue<T>),
    /// Variable is less than (unsigned): `(uint)x < value`
    LessThanUnsigned(ConstValue<T>),
}

impl<T: Target> Constraint<T> {
    /// Checks if a concrete value satisfies this constraint.
    ///
    /// Uses typed comparison methods from `ConstValue`.
    #[must_use]
    pub fn is_satisfied_by(&self, value: &ConstValue<T>) -> bool {
        match self {
            Self::Equal(v) => value.ceq(v).is_some_and(|r| !r.is_zero()),
            Self::NotEqual(v) => value.ceq(v).is_some_and(|r| r.is_zero()),
            Self::GreaterThan(v) => value.cgt(v).is_some_and(|r| !r.is_zero()),
            Self::LessThan(v) => value.clt(v).is_some_and(|r| !r.is_zero()),
            Self::GreaterOrEqual(v) => value.clt(v).is_some_and(|r| r.is_zero()),
            Self::LessOrEqual(v) => value.cgt(v).is_some_and(|r| r.is_zero()),
            Self::GreaterThanUnsigned(v) => value.cgt_un(v).is_some_and(|r| !r.is_zero()),
            Self::LessThanUnsigned(v) => value.clt_un(v).is_some_and(|r| !r.is_zero()),
        }
    }

    /// Returns the concrete value if this is an equality constraint.
    #[must_use]
    pub fn as_equal(&self) -> Option<&ConstValue<T>> {
        match self {
            Self::Equal(v) => Some(v),
            _ => None,
        }
    }

    /// Checks if this constraint conflicts with another (both can't be true).
    ///
    /// # Arguments
    ///
    /// * `other` - The other constraint to check against.
    /// * `ptr_size` - Target pointer size for native int/uint masking.
    #[must_use]
    pub fn conflicts_with(&self, other: &Constraint<T>, ptr_size: PointerSize) -> bool {
        match (self, other) {
            // x == a conflicts with x == b (if a != b)
            (Self::Equal(a), Self::Equal(b)) => a != b,
            // x == a conflicts with x != a
            (Self::Equal(a), Self::NotEqual(b)) | (Self::NotEqual(b), Self::Equal(a)) => a == b,
            // x == a conflicts with x > b when a <= b (i.e., a is not greater than b)
            (Self::Equal(a), Self::GreaterThan(b)) | (Self::GreaterThan(b), Self::Equal(a)) => {
                // a <= b means a is not strictly greater than b
                a.cgt(b).is_none_or(|r| r.is_zero())
            }
            // x == a conflicts with x < b when a >= b (i.e., a is not less than b)
            (Self::Equal(a), Self::LessThan(b)) | (Self::LessThan(b), Self::Equal(a)) => {
                // a >= b means a is not strictly less than b
                a.clt(b).is_none_or(|r| r.is_zero())
            }
            // x > a conflicts with x < b if ranges don't overlap
            (Self::GreaterThan(a), Self::LessThan(b))
            | (Self::LessThan(b), Self::GreaterThan(a)) => {
                // x > a AND x < b requires b > a + 1 (there must be room for at least one integer)
                // Conflicts when b <= a + 1
                let one: ConstValue<T> = ConstValue::I32(1);
                a.add(&one, ptr_size)
                    .and_then(|a_plus_1| b.cgt(&a_plus_1))
                    .is_none_or(|r| r.is_zero())
            }
            _ => false,
        }
    }
}

/// A constraint on a path derived from branch conditions.
///
/// Associates a [`Constraint`] with a specific SSA variable. These constraints
/// are accumulated during path evaluation and can be used for constraint
/// solving with Z3.
#[derive(Debug, Clone, PartialEq)]
pub struct PathConstraint<T: Target> {
    /// The variable this constraint applies to.
    pub variable: SsaVarId,
    /// The constraint on the variable's value.
    pub constraint: Constraint<T>,
}

impl<T: Target> PathConstraint<T> {
    /// Creates a new equality constraint.
    #[must_use]
    pub fn equal(variable: SsaVarId, value: ConstValue<T>) -> Self {
        Self {
            variable,
            constraint: Constraint::Equal(value),
        }
    }

    /// Creates a new inequality constraint.
    #[must_use]
    pub fn not_equal(variable: SsaVarId, value: ConstValue<T>) -> Self {
        Self {
            variable,
            constraint: Constraint::NotEqual(value),
        }
    }

    /// Creates a new less-than constraint.
    #[must_use]
    pub fn less_than(variable: SsaVarId, value: ConstValue<T>) -> Self {
        Self {
            variable,
            constraint: Constraint::LessThan(value),
        }
    }

    /// Creates a new greater-than constraint.
    #[must_use]
    pub fn greater_than(variable: SsaVarId, value: ConstValue<T>) -> Self {
        Self {
            variable,
            constraint: Constraint::GreaterThan(value),
        }
    }

    /// Checks if a concrete value satisfies this constraint.
    #[must_use]
    pub fn is_satisfied_by(&self, value: &ConstValue<T>) -> bool {
        self.constraint.is_satisfied_by(value)
    }
}
