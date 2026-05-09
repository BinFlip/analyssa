//! Decomposed SSA operations — the core opcode representation.
//!
//! This module defines [`SsaOp`], the decomposed operation that converts CIL
//! instructions into clean `result = op(operands)` form. Along with helper
//! types for classifying and extracting operation semantics.
//!
//! # Design Goals
//!
//! - **Single assignment**: Each operation produces at most one result (except void calls)
//! - **Explicit operands**: All data dependencies are explicit SSA variables — no implicit stack
//! - **Pattern matching**: Enum variants enable easy destructuring for analysis passes
//! - **Uniform access**: `dest()`, `uses()`, `replace_uses()`, `as_binary_op()`, etc.
//!
//! # Operation Categories
//!
//! | Category | Variants |
//! |----------|----------|
//! | Constants | `Const` |
//! | Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Neg`, and their `Ovf` variants |
//! | Bitwise | `And`, `Or`, `Xor`, `Not`, `Shl`, `Shr` |
//! | Comparison | `Ceq`, `Clt`, `Cgt`, `BranchCmp` (combined compare-and-branch) |
//! | Conversion | `Conv` (with overflow checking and signedness) |
//! | Control flow | `Jump`, `Branch`, `Switch`, `Return`, `Leave`, `Throw`, `Rethrow` |
//! | Memory | Field load/store, element load/store, indirect load/store |
//! | Objects | `NewObj`, `NewArr`, `CastClass`, `IsInst`, `Box`, `Unbox` |
//! | Calls | `Call`, `CallVirt`, `CallIndirect` |
//! | Prefixes | `Constrained`, `Volatile`, `Unaligned`, `TailPrefix`, `Readonly` |
//! | Synthetic | `Phi`, `Copy`, `Pop`, `Nop` |
//!
//! # Field Naming Conventions
//!
//! Consistent across all variants:
//! - `dest`: Destination SSA variable for the operation result
//! - `left`, `right`: Binary operands (left / right hand side)
//! - `operand`: Unary operand
//! - `object`: Object instance for field/method operations
//! - `array`, `index`: Array and index for element operations
//! - `addr`: Address for indirect memory operations
//! - `target`, `true_target`, `false_target`: Branch target block indices
//! - `unsigned`: Whether the operation treats values as unsigned
//! - `overflow_check`: Whether the operation checks for overflow

#![allow(missing_docs)]

use std::fmt;

use crate::{
    ir::{value::ConstValue, variable::SsaVarId},
    target::Target,
};

/// Comparison kind for `BranchCmp` operations.
///
/// Represents the comparison operator used in combined compare-and-branch
/// operations like `blt`, `beq`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpKind {
    /// Equal: `left == right`
    Eq,
    /// Not equal: `left != right`
    Ne,
    /// Less than: `left < right`
    Lt,
    /// Less than or equal: `left <= right`
    Le,
    /// Greater than: `left > right`
    Gt,
    /// Greater than or equal: `left >= right`
    Ge,
}

impl fmt::Display for CmpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eq => write!(f, "=="),
            Self::Ne => write!(f, "!="),
            Self::Lt => write!(f, "<"),
            Self::Le => write!(f, "<="),
            Self::Gt => write!(f, ">"),
            Self::Ge => write!(f, ">="),
        }
    }
}

/// Memory fence / barrier kind for atomic ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FenceKind {
    /// Full memory barrier
    Full,
    /// Acquire barrier
    Acquire,
    /// Release barrier
    Release,
    /// Acquire+Release barrier
    AcqRel,
    /// Sequentially consistent barrier
    SeqCst,
}

impl fmt::Display for FenceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Acquire => write!(f, "acquire"),
            Self::Release => write!(f, "release"),
            Self::AcqRel => write!(f, "acqrel"),
            Self::SeqCst => write!(f, "seqcst"),
        }
    }
}

/// Atomic read-modify-write operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicRmwOp {
    /// Exchange
    Xchg,
    /// Add
    Add,
    /// Sub
    Sub,
    /// And
    And,
    /// Or
    Or,
    /// Xor
    Xor,
    /// Min (signed)
    Min,
    /// Max (signed)
    Max,
}

impl fmt::Display for AtomicRmwOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Xchg => write!(f, "xchg"),
            Self::Add => write!(f, "add"),
            Self::Sub => write!(f, "sub"),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::Xor => write!(f, "xor"),
            Self::Min => write!(f, "min"),
            Self::Max => write!(f, "max"),
        }
    }
}

/// Bitmask for selecting flag bits from a flags-defining operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlagsMask(u16);

impl FlagsMask {
    pub const CARRY: Self = Self(1 << 0);
    pub const PARITY: Self = Self(1 << 1);
    pub const ADJUST: Self = Self(1 << 2);
    pub const ZERO: Self = Self(1 << 3);
    pub const SIGN: Self = Self(1 << 4);
    pub const OVERFLOW: Self = Self(1 << 5);

    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }
    pub const fn bits(self) -> u16 {
        self.0
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for FlagsMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        if self.0 & Self::CARRY.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "CF")?;
            first = false;
        }
        if self.0 & Self::PARITY.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "PF")?;
            first = false;
        }
        if self.0 & Self::ADJUST.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "AF")?;
            first = false;
        }
        if self.0 & Self::ZERO.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "ZF")?;
            first = false;
        }
        if self.0 & Self::SIGN.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "SF")?;
            first = false;
        }
        if self.0 & Self::OVERFLOW.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "OF")?;
            first = false;
        }
        if first {
            write!(f, "none")?;
        }
        Ok(())
    }
}

/// Condition code for flag-based branch operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlagCondition {
    Carry,
    NotCarry,
    Zero,
    NotZero,
    Overflow,
    NotOverflow,
    Negative,
    Positive,
    ParityEven,
    ParityOdd,
}

impl fmt::Display for FlagCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Carry => write!(f, "carry"),
            Self::NotCarry => write!(f, "not_carry"),
            Self::Zero => write!(f, "zero"),
            Self::NotZero => write!(f, "not_zero"),
            Self::Overflow => write!(f, "overflow"),
            Self::NotOverflow => write!(f, "not_overflow"),
            Self::Negative => write!(f, "negative"),
            Self::Positive => write!(f, "positive"),
            Self::ParityEven => write!(f, "parity_even"),
            Self::ParityOdd => write!(f, "parity_odd"),
        }
    }
}

/// Kind of binary operation for extracted binary op info.
///
/// This enum categorizes all binary operations in `SsaOp` for uniform
/// handling in optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOpKind {
    /// Addition: `left + right`
    Add,
    /// Addition with overflow check
    AddOvf,
    /// Subtraction: `left - right`
    Sub,
    /// Subtraction with overflow check
    SubOvf,
    /// Multiplication: `left * right`
    Mul,
    /// Multiplication with overflow check
    MulOvf,
    /// Division: `left / right`
    Div,
    /// Remainder: `left % right`
    Rem,
    /// Bitwise AND: `left & right`
    And,
    /// Bitwise OR: `left | right`
    Or,
    /// Bitwise XOR: `left ^ right`
    Xor,
    /// Shift left: `value << amount`
    Shl,
    /// Shift right: `value >> amount`
    Shr,
    /// Compare equal: `left == right`
    Ceq,
    /// Compare less than: `left < right`
    Clt,
    /// Compare greater than: `left > right`
    Cgt,
    /// Rotate left
    Rol,
    /// Rotate right
    Ror,
    /// Rotate through carry left
    Rcl,
    Rcr,
}

impl fmt::Display for BinaryOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "add"),
            Self::AddOvf => write!(f, "add.ovf"),
            Self::Sub => write!(f, "sub"),
            Self::SubOvf => write!(f, "sub.ovf"),
            Self::Mul => write!(f, "mul"),
            Self::MulOvf => write!(f, "mul.ovf"),
            Self::Div => write!(f, "div"),
            Self::Rem => write!(f, "rem"),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::Xor => write!(f, "xor"),
            Self::Shl => write!(f, "shl"),
            Self::Shr => write!(f, "shr"),
            Self::Ceq => write!(f, "ceq"),
            Self::Clt => write!(f, "clt"),
            Self::Cgt => write!(f, "cgt"),
            Self::Rol => write!(f, "rol"),
            Self::Ror => write!(f, "ror"),
            Self::Rcl => write!(f, "rcl"),
            Self::Rcr => write!(f, "rcr"),
        }
    }
}

impl BinaryOpKind {
    /// Returns `true` if this operation is commutative (`a op b == b op a`).
    ///
    /// Commutative operations can have their operands swapped without changing
    /// the result. This is useful for normalization in optimizations like GVN.
    ///
    /// # Commutative Operations
    ///
    /// - Arithmetic: `Add`, `AddOvf`, `Mul`, `MulOvf`
    /// - Bitwise: `And`, `Or`, `Xor`
    /// - Comparison: `Ceq` (equality is symmetric)
    #[must_use]
    pub const fn is_commutative(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::AddOvf
                | Self::Mul
                | Self::MulOvf
                | Self::And
                | Self::Or
                | Self::Xor
                | Self::Ceq
        )
    }

    /// Returns `true` if this is a comparison operation.
    ///
    /// Comparison operations produce a boolean result (0 or 1) based on
    /// comparing two operands.
    #[must_use]
    pub const fn is_comparison(self) -> bool {
        matches!(self, Self::Ceq | Self::Clt | Self::Cgt)
    }

    /// Returns the operation with swapped operand semantics, if applicable.
    ///
    /// For comparison operations:
    /// - `Clt` (less than) becomes `Cgt` (greater than) when operands swap
    /// - `Cgt` (greater than) becomes `Clt` (less than) when operands swap
    /// - `Ceq` (equal) stays the same (symmetric)
    ///
    /// For non-comparison operations, returns `self` unchanged.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::ir::BinaryOpKind;
    ///
    /// // a < b is equivalent to b > a
    /// assert_eq!(BinaryOpKind::Clt.swapped(), BinaryOpKind::Cgt);
    /// ```
    #[must_use]
    pub const fn swapped(self) -> Self {
        match self {
            Self::Clt => Self::Cgt,
            Self::Cgt => Self::Clt,
            other => other,
        }
    }

    /// Returns `true` if signedness affects the operation's semantics.
    ///
    /// Operations where the `unsigned` flag changes behavior:
    /// - `Div`, `Rem`: Signed vs unsigned division/remainder
    /// - `Shr`: Arithmetic (signed) vs logical (unsigned) shift
    /// - `Clt`, `Cgt`: Signed vs unsigned comparison
    ///
    /// For other operations, the unsigned flag has no effect.
    #[must_use]
    pub const fn is_signedness_sensitive(self) -> bool {
        matches!(
            self,
            Self::Div | Self::Rem | Self::Shr | Self::Clt | Self::Cgt
        )
    }
}

/// Information about a binary operation extracted from an `SsaOp`.
///
/// This provides a uniform view of binary operations for optimization passes,
/// allowing them to handle all binary ops generically without matching on
/// each variant individually.
///
/// # Example
///
/// ```rust
/// use analyssa::{MockTarget, ir::{SsaOp, SsaVarId}};
///
/// let op = SsaOp::<MockTarget>::Add {
///     dest: SsaVarId::from_index(2),
///     left: SsaVarId::from_index(0),
///     right: SsaVarId::from_index(1),
///     flags: None,
/// };
/// if let Some(info) = op.as_binary_op() {
///     // Handle all binary ops uniformly
///     println!("{} = {} {} {}", info.dest, info.left, info.kind, info.right);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryOpInfo {
    /// The kind of binary operation.
    pub kind: BinaryOpKind,
    /// Destination variable for the result.
    pub dest: SsaVarId,
    /// Left operand.
    pub left: SsaVarId,
    /// Right operand.
    pub right: SsaVarId,
    /// Whether the operation treats operands as unsigned.
    pub unsigned: bool,
    /// Optional flags variable defined by this operation.
    pub flags: Option<SsaVarId>,
}

impl BinaryOpInfo {
    /// Returns a normalized version of this operation for value numbering.
    ///
    /// For commutative operations, this ensures operands are in a canonical
    /// order (smaller variable index first). For non-commutative comparisons
    /// like `Clt` and `Cgt`, swapping operands also swaps the operation kind.
    ///
    /// This is useful for Global Value Numbering (GVN) where `a + b` and `b + a`
    /// should hash to the same value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::ir::{BinaryOpInfo, BinaryOpKind, SsaVarId};
    ///
    /// let v2 = SsaVarId::from_index(2);
    /// let v5 = SsaVarId::from_index(5);
    /// let info = BinaryOpInfo {
    ///     kind: BinaryOpKind::Add,
    ///     dest: SsaVarId::from_index(9),
    ///     left: v5,
    ///     right: v2,
    ///     unsigned: false,
    ///     flags: None,
    /// };
    /// let normalized = info.normalized();
    /// assert_eq!(normalized.left, v2);
    /// assert_eq!(normalized.right, v5);
    /// ```
    #[must_use]
    pub fn normalized(self) -> Self {
        // Only normalize if right operand should come first
        if self.right.index() < self.left.index() {
            if self.kind.is_commutative() {
                // Commutative: just swap operands
                Self {
                    left: self.right,
                    right: self.left,
                    ..self
                }
            } else if self.kind.is_comparison() {
                // Non-commutative comparison: swap operands AND operation
                Self {
                    kind: self.kind.swapped(),
                    left: self.right,
                    right: self.left,
                    ..self
                }
            } else {
                // Non-commutative, non-comparison: don't normalize
                self
            }
        } else {
            self
        }
    }

    /// Returns a tuple suitable for use as a hash key in value numbering.
    ///
    /// The tuple includes all semantically relevant fields:
    /// - Operation kind
    /// - Unsigned flag (only if the operation is signedness-sensitive)
    /// - Left and right operands
    ///
    /// For operations where signedness doesn't matter, the unsigned field
    /// is normalized to `false` to ensure consistent hashing.
    #[must_use]
    pub fn value_key(self) -> (BinaryOpKind, bool, SsaVarId, SsaVarId) {
        let unsigned = if self.kind.is_signedness_sensitive() {
            self.unsigned
        } else {
            false // Normalize for consistent hashing
        };
        (self.kind, unsigned, self.left, self.right)
    }
}

/// Kind of unary operation for extracted unary op info.
///
/// This enum categorizes all unary operations in `SsaOp` for uniform
/// handling in optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOpKind {
    /// Negation: `-operand`
    Neg,
    /// Bitwise NOT: `~operand`
    Not,
    /// Check finite
    Ckfinite,
    /// Byte swap (endian conversion)
    BSwap,
    /// Bit reverse
    BRev,
    /// Bit scan forward (find first set bit, LSB-based)
    BitScanForward,
    /// Bit scan reverse (find first set bit, MSB-based)
    BitScanReverse,
    /// Population count
    Popcount,
    /// Parity (1 if odd number of set bits)
    Parity,
}

impl fmt::Display for UnaryOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Neg => write!(f, "neg"),
            Self::Not => write!(f, "not"),
            Self::Ckfinite => write!(f, "ckfinite"),
            Self::BSwap => write!(f, "bswap"),
            Self::BRev => write!(f, "brev"),
            Self::BitScanForward => write!(f, "bsf"),
            Self::BitScanReverse => write!(f, "bsr"),
            Self::Popcount => write!(f, "popcnt"),
            Self::Parity => write!(f, "parity"),
        }
    }
}

/// Information about a unary operation extracted from an `SsaOp`.
///
/// This provides a uniform view of unary operations for optimization passes,
/// allowing them to handle all unary ops generically without matching on
/// each variant individually.
///
/// # Example
///
/// ```ignore
/// if let Some(info) = op.as_unary_op() {
///     // Handle all unary ops uniformly
///     println!("{} = {} {}", info.dest, info.kind, info.operand);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnaryOpInfo {
    /// The kind of unary operation.
    pub kind: UnaryOpKind,
    /// Destination variable for the result.
    pub dest: SsaVarId,
    /// The operand.
    pub operand: SsaVarId,
}

/// A decomposed SSA operation.
///
/// Each variant represents a single operation with explicit inputs and outputs.
/// This enables clean pattern matching for optimization and analysis passes.
///
/// # Conventions
///
/// - For operations that produce a result, the first `SsaVarId` is the destination
/// - Operands follow in the order they appear on the CIL stack (first pushed = first operand)
/// - Optional results use `Option<SsaVarId>` (e.g., calls that may not return a value)
#[derive(Debug, Clone, PartialEq)]
pub enum SsaOp<T: Target> {
    /// Load a constant value.
    ///
    /// `dest = const value`
    Const {
        dest: SsaVarId,
        value: ConstValue<T>,
    },

    /// Addition: `dest = left + right`
    Add {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        flags: Option<SsaVarId>,
    },

    /// Addition with overflow check: `dest = left + right` (throws on overflow)
    AddOvf {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
        flags: Option<SsaVarId>,
    },

    /// Subtraction: `dest = left - right`
    Sub {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        flags: Option<SsaVarId>,
    },

    /// Subtraction with overflow check: `dest = left - right`
    SubOvf {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
        flags: Option<SsaVarId>,
    },

    /// Multiplication: `dest = left * right`
    Mul {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        flags: Option<SsaVarId>,
    },

    /// Multiplication with overflow check: `dest = left * right`
    MulOvf {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
        flags: Option<SsaVarId>,
    },

    /// Division: `dest = left / right`
    Div {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
        flags: Option<SsaVarId>,
    },

    /// Remainder: `dest = left % right`
    Rem {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
        flags: Option<SsaVarId>,
    },

    /// Negation: `dest = -operand`
    Neg {
        dest: SsaVarId,
        operand: SsaVarId,
        flags: Option<SsaVarId>,
    },

    /// Bitwise AND: `dest = left & right`
    And {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        flags: Option<SsaVarId>,
    },

    /// Bitwise OR: `dest = left | right`
    Or {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        flags: Option<SsaVarId>,
    },

    /// Bitwise XOR: `dest = left ^ right`
    Xor {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        flags: Option<SsaVarId>,
    },

    /// Bitwise NOT: `dest = ~operand`
    Not {
        dest: SsaVarId,
        operand: SsaVarId,
        flags: Option<SsaVarId>,
    },

    /// Shift left: `dest = value << amount`
    Shl {
        dest: SsaVarId,
        value: SsaVarId,
        amount: SsaVarId,
        flags: Option<SsaVarId>,
    },

    /// Shift right: `dest = value >> amount`
    Shr {
        dest: SsaVarId,
        value: SsaVarId,
        amount: SsaVarId,
        unsigned: bool,
        flags: Option<SsaVarId>,
    },

    /// Rotate left: `dest = value <<< amount`
    Rol {
        dest: SsaVarId,
        value: SsaVarId,
        amount: SsaVarId,
    },

    /// Rotate right: `dest = value >>> amount`
    Ror {
        dest: SsaVarId,
        value: SsaVarId,
        amount: SsaVarId,
    },

    /// Rotate through carry left
    Rcl {
        dest: SsaVarId,
        value: SsaVarId,
        amount: SsaVarId,
    },

    /// Rotate through carry right
    Rcr {
        dest: SsaVarId,
        value: SsaVarId,
        amount: SsaVarId,
    },

    /// Byte swap (endian conversion): `dest = bswap(src)`
    BSwap { dest: SsaVarId, src: SsaVarId },

    /// Bit reverse: `dest = brev(src)`
    BRev { dest: SsaVarId, src: SsaVarId },

    /// Bit scan forward (find first set bit, LSB-based): `dest = bsf(src)`
    BitScanForward { dest: SsaVarId, src: SsaVarId },

    /// Bit scan reverse (find first set bit, MSB-based): `dest = bsr(src)`
    BitScanReverse { dest: SsaVarId, src: SsaVarId },

    /// Population count: `dest = popcnt(src)`
    Popcount { dest: SsaVarId, src: SsaVarId },

    /// Parity: `dest = parity(src)` — 1 if odd number of set bits, 0 if even
    Parity { dest: SsaVarId, src: SsaVarId },

    /// Compare equal: `dest = (left == right) ? 1 : 0`
    Ceq {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
    },

    /// Compare less than: `dest = (left < right) ? 1 : 0`
    Clt {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    },

    /// Compare greater than: `dest = (left > right) ? 1 : 0`
    Cgt {
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    },

    /// Type conversion: `dest = (target_type)operand`
    Conv {
        dest: SsaVarId,
        operand: SsaVarId,
        target: T::Type,
        overflow_check: bool,
        unsigned: bool,
    },

    /// Conditional select: `dest = condition ? true_val : false_val`
    Select {
        dest: SsaVarId,
        condition: SsaVarId,
        true_val: SsaVarId,
        false_val: SsaVarId,
    },

    /// Read condition code flags from a flags variable.
    ReadFlags {
        dest: SsaVarId,
        flags: SsaVarId,
        mask: FlagsMask,
    },

    /// Unconditional jump to a block.
    Jump { target: usize },

    /// Conditional branch: if condition is true, go to true_target, else false_target.
    Branch {
        condition: SsaVarId,
        true_target: usize,
        false_target: usize,
    },

    /// Compare and branch: if (left cmp right) goto true_target else false_target.
    ///
    /// This represents CIL comparison branch instructions like `beq`, `blt`, `bgt`, etc.
    /// These are combined compare-and-branch operations that don't produce an intermediate
    /// comparison result.
    BranchCmp {
        left: SsaVarId,
        right: SsaVarId,
        cmp: CmpKind,
        unsigned: bool,
        true_target: usize,
        false_target: usize,
    },

    /// Branch based on condition code flags.
    BranchFlags {
        flags: SsaVarId,
        condition: FlagCondition,
        true_target: usize,
        false_target: usize,
    },

    /// Switch statement: jump to `targets[value]` or default if out of range.
    Switch {
        value: SsaVarId,
        targets: Vec<usize>,
        default: usize,
    },

    /// Return from method with optional value.
    Return { value: Option<SsaVarId> },

    /// Load instance field: `dest = object.field`
    LoadField {
        dest: SsaVarId,
        object: SsaVarId,
        field: T::FieldRef,
    },

    /// Store instance field: `object.field = value`
    StoreField {
        object: SsaVarId,
        field: T::FieldRef,
        value: SsaVarId,
    },

    /// Load static field: `dest = ClassName.field`
    LoadStaticField { dest: SsaVarId, field: T::FieldRef },

    /// Store static field: `ClassName.field = value`
    StoreStaticField { field: T::FieldRef, value: SsaVarId },

    /// Load field address: `dest = &object.field`
    LoadFieldAddr {
        dest: SsaVarId,
        object: SsaVarId,
        field: T::FieldRef,
    },

    /// Load static field address: `dest = &ClassName.field`
    LoadStaticFieldAddr { dest: SsaVarId, field: T::FieldRef },

    /// Load array element: `dest = array[index]`
    LoadElement {
        dest: SsaVarId,
        array: SsaVarId,
        index: SsaVarId,
        elem_type: T::Type,
    },

    /// Store array element: `array[index] = value`
    StoreElement {
        array: SsaVarId,
        index: SsaVarId,
        value: SsaVarId,
        elem_type: T::Type,
    },

    /// Load array element address: `dest = &array[index]`
    LoadElementAddr {
        dest: SsaVarId,
        array: SsaVarId,
        index: SsaVarId,
        elem_type: T::TypeRef,
    },

    /// Get array length: `dest = array.Length`
    ArrayLength { dest: SsaVarId, array: SsaVarId },

    /// Load through pointer: `dest = *ptr`
    LoadIndirect {
        dest: SsaVarId,
        addr: SsaVarId,
        value_type: T::Type,
    },

    /// Store through pointer: `*ptr = value`
    StoreIndirect {
        addr: SsaVarId,
        value: SsaVarId,
        value_type: T::Type,
    },

    /// Create new object: `dest = new Type(args...)`
    NewObj {
        dest: SsaVarId,
        ctor: T::MethodRef,
        args: Vec<SsaVarId>,
    },

    /// Create new array: `dest = new Type[length]`
    NewArr {
        dest: SsaVarId,
        elem_type: T::TypeRef,
        length: SsaVarId,
    },

    /// Cast object to type (throws if invalid): `dest = (Type)obj`
    CastClass {
        dest: SsaVarId,
        object: SsaVarId,
        target_type: T::TypeRef,
    },

    /// Type check (returns null if invalid): `dest = obj as Type`
    IsInst {
        dest: SsaVarId,
        object: SsaVarId,
        target_type: T::TypeRef,
    },

    /// Box value type: `dest = (object)value`
    Box {
        dest: SsaVarId,
        value: SsaVarId,
        value_type: T::TypeRef,
    },

    /// Unbox to pointer: `dest = &((ValueType)obj)`
    Unbox {
        dest: SsaVarId,
        object: SsaVarId,
        value_type: T::TypeRef,
    },

    /// Unbox and copy: `dest = (ValueType)obj`
    UnboxAny {
        dest: SsaVarId,
        object: SsaVarId,
        value_type: T::TypeRef,
    },

    /// Get size of value type: `dest = sizeof(Type)`
    SizeOf {
        dest: SsaVarId,
        value_type: T::TypeRef,
    },

    /// Load runtime type token: `dest = typeof(Type).TypeHandle`
    LoadToken { dest: SsaVarId, token: T::TypeRef },

    /// Direct method call: `dest = method(args...)`
    Call {
        dest: Option<SsaVarId>,
        method: T::MethodRef,
        args: Vec<SsaVarId>,
    },

    /// Virtual method call: `dest = obj.method(args...)`
    CallVirt {
        dest: Option<SsaVarId>,
        method: T::MethodRef,
        args: Vec<SsaVarId>,
    },

    /// Indirect call through function pointer: `dest = fptr(args...)`
    CallIndirect {
        dest: Option<SsaVarId>,
        fptr: SsaVarId,
        signature: T::SigRef,
        args: Vec<SsaVarId>,
    },

    /// Load function pointer: `dest = &method`
    LoadFunctionPtr {
        dest: SsaVarId,
        method: T::MethodRef,
    },

    /// Load virtual function pointer: `dest = &obj.method`
    LoadVirtFunctionPtr {
        dest: SsaVarId,
        object: SsaVarId,
        method: T::MethodRef,
    },

    /// Load argument value: `dest = argN`
    LoadArg { dest: SsaVarId, arg_index: u16 },

    /// Load local value: `dest = localN`
    LoadLocal { dest: SsaVarId, local_index: u16 },

    /// Load argument address: `dest = &argN`
    LoadArgAddr { dest: SsaVarId, arg_index: u16 },

    /// Load local address: `dest = &localN`
    LoadLocalAddr { dest: SsaVarId, local_index: u16 },

    /// Copy value (from dup): `dest = src`
    Copy { dest: SsaVarId, src: SsaVarId },

    /// Pop value from stack (value is discarded, but we track the use)
    Pop { value: SsaVarId },

    /// Throw exception: `throw obj`
    Throw { exception: SsaVarId },

    /// Rethrow current exception (in catch handler)
    Rethrow,

    /// End finally block
    EndFinally,

    /// End filter block with result
    EndFilter { result: SsaVarId },

    /// Return from interrupt / exception handler
    InterruptReturn,

    /// Unreachable terminator.
    Unreachable,

    /// Leave protected region
    Leave { target: usize },

    /// Initialize block of memory to zero
    InitBlk {
        dest_addr: SsaVarId,
        value: SsaVarId,
        size: SsaVarId,
    },

    /// Copy block of memory
    CopyBlk {
        dest_addr: SsaVarId,
        src_addr: SsaVarId,
        size: SsaVarId,
    },

    /// Memory fence / barrier
    Fence { kind: FenceKind },

    /// Compare-and-swap: `old = *addr; if old == expected { *addr = desired; } return old`
    CmpXchg {
        dest: SsaVarId,
        addr: SsaVarId,
        expected: SsaVarId,
        desired: SsaVarId,
    },

    /// Atomic read-modify-write: `old = *addr; *addr = op(old, value); return old`
    AtomicRmw {
        dest: SsaVarId,
        addr: SsaVarId,
        value: SsaVarId,
        op: AtomicRmwOp,
    },

    /// Initialize object (for value types): `*dest = default(T)`
    InitObj {
        dest_addr: SsaVarId,
        value_type: T::TypeRef,
    },

    /// Copy object (for value types): `*dest = *src`
    CopyObj {
        dest_addr: SsaVarId,
        src_addr: SsaVarId,
        value_type: T::TypeRef,
    },

    /// Load object (value type copy): `dest = *src`
    LoadObj {
        dest: SsaVarId,
        src_addr: SsaVarId,
        value_type: T::TypeRef,
    },

    /// Store object (value type copy): `*dest = value`
    StoreObj {
        dest_addr: SsaVarId,
        value: SsaVarId,
        value_type: T::TypeRef,
    },

    /// No operation (for nop instructions)
    Nop,

    /// Breakpoint trap
    Break,

    /// Check for finite floating point: throws if not finite
    Ckfinite { dest: SsaVarId, operand: SsaVarId },

    /// Localloc: allocate stack space
    LocalAlloc { dest: SsaVarId, size: SsaVarId },

    /// Constrained virtual call prefix (affects next callvirt)
    Constrained { constraint_type: T::TypeRef },

    /// Volatile prefix (next memory access must not be reordered/cached)
    Volatile,

    /// Unaligned prefix (next memory access may be unaligned)
    Unaligned { alignment: u8 },

    /// Tail call prefix (next call is a tail call)
    TailPrefix,

    /// Readonly prefix (next ldelema returns a controlled-mutability managed pointer)
    Readonly,

    /// Phi node: merges values from different predecessors.
    ///
    /// This is placed at the beginning of blocks with multiple predecessors.
    Phi {
        dest: SsaVarId,
        operands: Vec<(usize, SsaVarId)>,
    },
}

impl<T: Target> SsaOp<T> {
    /// Returns the destination variable if this operation produces one.
    #[must_use]
    pub fn dest(&self) -> Option<SsaVarId> {
        match self {
            Self::Const { dest, .. }
            | Self::Add { dest, .. }
            | Self::AddOvf { dest, .. }
            | Self::Sub { dest, .. }
            | Self::SubOvf { dest, .. }
            | Self::Mul { dest, .. }
            | Self::MulOvf { dest, .. }
            | Self::Div { dest, .. }
            | Self::Rem { dest, .. }
            | Self::Neg { dest, .. }
            | Self::And { dest, .. }
            | Self::Or { dest, .. }
            | Self::Xor { dest, .. }
            | Self::Not { dest, .. }
            | Self::Shl { dest, .. }
            | Self::Shr { dest, .. }
            | Self::Ceq { dest, .. }
            | Self::Clt { dest, .. }
            | Self::Cgt { dest, .. }
            | Self::Conv { dest, .. }
            | Self::LoadField { dest, .. }
            | Self::LoadStaticField { dest, .. }
            | Self::LoadFieldAddr { dest, .. }
            | Self::LoadStaticFieldAddr { dest, .. }
            | Self::LoadElement { dest, .. }
            | Self::LoadElementAddr { dest, .. }
            | Self::ArrayLength { dest, .. }
            | Self::LoadIndirect { dest, .. }
            | Self::NewObj { dest, .. }
            | Self::NewArr { dest, .. }
            | Self::CastClass { dest, .. }
            | Self::IsInst { dest, .. }
            | Self::Box { dest, .. }
            | Self::Unbox { dest, .. }
            | Self::UnboxAny { dest, .. }
            | Self::SizeOf { dest, .. }
            | Self::LoadToken { dest, .. }
            | Self::LoadFunctionPtr { dest, .. }
            | Self::LoadVirtFunctionPtr { dest, .. }
            | Self::LoadArg { dest, .. }
            | Self::LoadLocal { dest, .. }
            | Self::LoadArgAddr { dest, .. }
            | Self::LoadLocalAddr { dest, .. }
            | Self::Copy { dest, .. }
            | Self::Ckfinite { dest, .. }
            | Self::LocalAlloc { dest, .. }
            | Self::LoadObj { dest, .. }
            | Self::Phi { dest, .. }
            | Self::Rol { dest, .. }
            | Self::Ror { dest, .. }
            | Self::Rcl { dest, .. }
            | Self::Rcr { dest, .. }
            | Self::BSwap { dest, .. }
            | Self::BRev { dest, .. }
            | Self::BitScanForward { dest, .. }
            | Self::BitScanReverse { dest, .. }
            | Self::Popcount { dest, .. }
            | Self::Parity { dest, .. }
            | Self::Select { dest, .. }
            | Self::CmpXchg { dest, .. }
            | Self::AtomicRmw { dest, .. }
            | Self::ReadFlags { dest, .. } => Some(*dest),

            Self::Call { dest, .. }
            | Self::CallVirt { dest, .. }
            | Self::CallIndirect { dest, .. } => *dest,

            // Operations that don't produce a result
            Self::StoreField { .. }
            | Self::StoreStaticField { .. }
            | Self::StoreElement { .. }
            | Self::StoreIndirect { .. }
            | Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::Switch { .. }
            | Self::Return { .. }
            | Self::Pop { .. }
            | Self::Throw { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::EndFilter { .. }
            | Self::Leave { .. }
            | Self::InitBlk { .. }
            | Self::CopyBlk { .. }
            | Self::InitObj { .. }
            | Self::CopyObj { .. }
            | Self::StoreObj { .. }
            | Self::Nop
            | Self::Break
            | Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly
            | Self::Fence { .. }
            | Self::InterruptReturn
            | Self::BranchFlags { .. }
            | Self::Unreachable => None,
        }
    }

    /// Sets the destination variable for operations that produce a result.
    ///
    /// This is used during SSA renaming to update the dest after assigning
    /// new SSA variable IDs. Returns `true` if the dest was updated.
    ///
    /// # Arguments
    ///
    /// * `new_dest` - The new destination variable ID
    pub fn set_dest(&mut self, new_dest: SsaVarId) -> bool {
        match self {
            Self::Const { dest, .. }
            | Self::Add { dest, .. }
            | Self::AddOvf { dest, .. }
            | Self::Sub { dest, .. }
            | Self::SubOvf { dest, .. }
            | Self::Mul { dest, .. }
            | Self::MulOvf { dest, .. }
            | Self::Div { dest, .. }
            | Self::Rem { dest, .. }
            | Self::Neg { dest, .. }
            | Self::And { dest, .. }
            | Self::Or { dest, .. }
            | Self::Xor { dest, .. }
            | Self::Not { dest, .. }
            | Self::Shl { dest, .. }
            | Self::Shr { dest, .. }
            | Self::Ceq { dest, .. }
            | Self::Clt { dest, .. }
            | Self::Cgt { dest, .. }
            | Self::Conv { dest, .. }
            | Self::LoadField { dest, .. }
            | Self::LoadStaticField { dest, .. }
            | Self::LoadFieldAddr { dest, .. }
            | Self::LoadStaticFieldAddr { dest, .. }
            | Self::LoadElement { dest, .. }
            | Self::LoadElementAddr { dest, .. }
            | Self::ArrayLength { dest, .. }
            | Self::LoadIndirect { dest, .. }
            | Self::NewObj { dest, .. }
            | Self::NewArr { dest, .. }
            | Self::CastClass { dest, .. }
            | Self::IsInst { dest, .. }
            | Self::Box { dest, .. }
            | Self::Unbox { dest, .. }
            | Self::UnboxAny { dest, .. }
            | Self::SizeOf { dest, .. }
            | Self::LoadToken { dest, .. }
            | Self::LoadFunctionPtr { dest, .. }
            | Self::LoadVirtFunctionPtr { dest, .. }
            | Self::LoadArg { dest, .. }
            | Self::LoadLocal { dest, .. }
            | Self::LoadArgAddr { dest, .. }
            | Self::LoadLocalAddr { dest, .. }
            | Self::Copy { dest, .. }
            | Self::Ckfinite { dest, .. }
            | Self::LocalAlloc { dest, .. }
            | Self::LoadObj { dest, .. }
            | Self::Phi { dest, .. }
            | Self::Rol { dest, .. }
            | Self::Ror { dest, .. }
            | Self::Rcl { dest, .. }
            | Self::Rcr { dest, .. }
            | Self::BSwap { dest, .. }
            | Self::BRev { dest, .. }
            | Self::BitScanForward { dest, .. }
            | Self::BitScanReverse { dest, .. }
            | Self::Popcount { dest, .. }
            | Self::Parity { dest, .. }
            | Self::Select { dest, .. }
            | Self::CmpXchg { dest, .. }
            | Self::AtomicRmw { dest, .. }
            | Self::ReadFlags { dest, .. } => {
                *dest = new_dest;
                true
            }

            Self::Call { dest, .. }
            | Self::CallVirt { dest, .. }
            | Self::CallIndirect { dest, .. } => {
                *dest = Some(new_dest);
                true
            }

            // Operations that don't produce a result - cannot set dest
            Self::StoreField { .. }
            | Self::StoreStaticField { .. }
            | Self::StoreElement { .. }
            | Self::StoreIndirect { .. }
            | Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::Switch { .. }
            | Self::Return { .. }
            | Self::Pop { .. }
            | Self::Throw { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::EndFilter { .. }
            | Self::Leave { .. }
            | Self::InitBlk { .. }
            | Self::CopyBlk { .. }
            | Self::InitObj { .. }
            | Self::CopyObj { .. }
            | Self::StoreObj { .. }
            | Self::Nop
            | Self::Break
            | Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly
            | Self::Fence { .. }
            | Self::InterruptReturn
            | Self::BranchFlags { .. }
            | Self::Unreachable => false,
        }
    }

    /// Returns the flags destination if this operation defines flags.
    pub fn flags_dest(&self) -> Option<SsaVarId> {
        match self {
            Self::Add { flags, .. }
            | Self::AddOvf { flags, .. }
            | Self::Sub { flags, .. }
            | Self::SubOvf { flags, .. }
            | Self::Mul { flags, .. }
            | Self::MulOvf { flags, .. }
            | Self::Div { flags, .. }
            | Self::Rem { flags, .. }
            | Self::Neg { flags, .. }
            | Self::And { flags, .. }
            | Self::Or { flags, .. }
            | Self::Xor { flags, .. }
            | Self::Not { flags, .. }
            | Self::Shl { flags, .. }
            | Self::Shr { flags, .. } => *flags,
            _ => None,
        }
    }

    /// Returns all variables used by this operation.
    #[must_use]
    #[allow(clippy::match_same_arms)] // Kept separate for clarity by operation category
    pub fn uses(&self) -> Vec<SsaVarId> {
        match self {
            Self::Const { .. } => vec![],

            Self::Add { left, right, .. }
            | Self::AddOvf { left, right, .. }
            | Self::Sub { left, right, .. }
            | Self::SubOvf { left, right, .. }
            | Self::Mul { left, right, .. }
            | Self::MulOvf { left, right, .. }
            | Self::Div { left, right, .. }
            | Self::Rem { left, right, .. }
            | Self::And { left, right, .. }
            | Self::Or { left, right, .. }
            | Self::Xor { left, right, .. }
            | Self::Ceq { left, right, .. }
            | Self::Clt { left, right, .. }
            | Self::Cgt { left, right, .. } => vec![*left, *right],

            Self::Shl { value, amount, .. }
            | Self::Shr { value, amount, .. }
            | Self::Rol { value, amount, .. }
            | Self::Ror { value, amount, .. }
            | Self::Rcl { value, amount, .. }
            | Self::Rcr { value, amount, .. } => {
                vec![*value, *amount]
            }

            Self::Neg { operand, .. }
            | Self::Not { operand, .. }
            | Self::Conv { operand, .. }
            | Self::Ckfinite { operand, .. }
            | Self::BSwap { src: operand, .. }
            | Self::BRev { src: operand, .. }
            | Self::BitScanForward { src: operand, .. }
            | Self::BitScanReverse { src: operand, .. }
            | Self::Popcount { src: operand, .. }
            | Self::Parity { src: operand, .. } => vec![*operand],

            Self::Branch { condition, .. } => vec![*condition],
            Self::BranchCmp { left, right, .. } => vec![*left, *right],
            Self::BranchFlags { flags, .. } => vec![*flags],
            Self::ReadFlags { flags, .. } => vec![*flags],
            Self::Select {
                condition,
                true_val,
                false_val,
                ..
            } => vec![*condition, *true_val, *false_val],
            Self::Switch { value, .. } => vec![*value],
            Self::Return { value } => value.iter().copied().collect(),

            Self::LoadField { object, .. } => vec![*object],
            Self::StoreField { object, value, .. } => vec![*object, *value],
            Self::LoadStaticField { .. } => vec![],
            Self::StoreStaticField { value, .. } => vec![*value],
            Self::LoadFieldAddr { object, .. } => vec![*object],
            Self::LoadStaticFieldAddr { .. } => vec![],

            Self::LoadElement { array, index, .. } | Self::LoadElementAddr { array, index, .. } => {
                vec![*array, *index]
            }
            Self::StoreElement {
                array,
                index,
                value,
                ..
            } => vec![*array, *index, *value],
            Self::ArrayLength { array, .. } => vec![*array],

            Self::LoadIndirect { addr, .. } => vec![*addr],
            Self::StoreIndirect { addr, value, .. } => vec![*addr, *value],
            Self::CmpXchg {
                addr,
                expected,
                desired,
                ..
            } => vec![*addr, *expected, *desired],
            Self::AtomicRmw { addr, value, .. } => vec![*addr, *value],

            Self::NewObj { args, .. } => args.clone(),
            Self::NewArr { length, .. } => vec![*length],
            Self::CastClass { object, .. }
            | Self::IsInst { object, .. }
            | Self::Unbox { object, .. }
            | Self::UnboxAny { object, .. } => vec![*object],
            Self::Box { value, .. } => vec![*value],
            Self::SizeOf { .. } | Self::LoadToken { .. } => vec![],

            Self::Call { args, .. } | Self::CallVirt { args, .. } => args.clone(),
            Self::CallIndirect { fptr, args, .. } => {
                let mut uses = vec![*fptr];
                uses.extend(args);
                uses
            }

            Self::LoadFunctionPtr { .. } => vec![],
            Self::LoadVirtFunctionPtr { object, .. } => vec![*object],

            Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. } => vec![],

            Self::Copy { src, .. } => vec![*src],
            Self::Pop { value } => vec![*value],

            Self::Throw { exception } => vec![*exception],
            Self::EndFilter { result } => vec![*result],

            Self::InitBlk {
                dest_addr,
                value,
                size,
            }
            | Self::CopyBlk {
                dest_addr,
                src_addr: value,
                size,
            } => vec![*dest_addr, *value, *size],

            Self::InitObj { dest_addr, .. } => vec![*dest_addr],
            Self::CopyObj {
                dest_addr,
                src_addr,
                ..
            } => vec![*dest_addr, *src_addr],
            Self::LoadObj { src_addr, .. } => vec![*src_addr],
            Self::StoreObj {
                dest_addr, value, ..
            } => vec![*dest_addr, *value],

            Self::LocalAlloc { size, .. } => vec![*size],

            Self::Phi { operands, .. } => operands.iter().map(|(_, v)| *v).collect(),

            Self::Jump { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::Leave { .. }
            | Self::Nop
            | Self::Break
            | Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly
            | Self::Fence { .. }
            | Self::InterruptReturn
            | Self::Unreachable => vec![],
        }
    }

    /// Returns `true` if this operation is a terminator (ends a basic block).
    #[must_use]
    pub const fn is_terminator(&self) -> bool {
        matches!(
            self,
            Self::Jump { .. }
                | Self::Branch { .. }
                | Self::BranchCmp { .. }
                | Self::BranchFlags { .. }
                | Self::Switch { .. }
                | Self::Return { .. }
                | Self::Throw { .. }
                | Self::Rethrow
                | Self::Leave { .. }
                | Self::EndFinally
                | Self::EndFilter { .. }
                | Self::InterruptReturn
                | Self::Unreachable
        )
    }

    /// Returns `true` if this operation may throw an exception.
    #[must_use]
    pub const fn may_throw(&self) -> bool {
        matches!(
            self,
            Self::Div { .. }
                | Self::Rem { .. }
                | Self::AddOvf { .. }
                | Self::SubOvf { .. }
                | Self::MulOvf { .. }
                | Self::Conv {
                    overflow_check: true,
                    ..
                }
                | Self::LoadField { .. }
                | Self::StoreField { .. }
                | Self::LoadElement { .. }
                | Self::StoreElement { .. }
                | Self::LoadElementAddr { .. }
                | Self::LoadIndirect { .. }
                | Self::StoreIndirect { .. }
                | Self::NewObj { .. }
                | Self::NewArr { .. }
                | Self::CastClass { .. }
                | Self::Unbox { .. }
                | Self::UnboxAny { .. }
                | Self::Call { .. }
                | Self::CallVirt { .. }
                | Self::CallIndirect { .. }
                | Self::Throw { .. }
                | Self::Ckfinite { .. }
                | Self::CmpXchg { .. }
                | Self::AtomicRmw { .. }
        )
    }

    /// Returns `true` if this operation is pure (has no side effects).
    ///
    /// Pure operations can be eliminated if their result is unused.
    #[must_use]
    pub const fn is_pure(&self) -> bool {
        matches!(
            self,
            Self::Const { .. }
                | Self::Add { .. }
                | Self::Sub { .. }
                | Self::Mul { .. }
                | Self::Neg { .. }
                | Self::And { .. }
                | Self::Or { .. }
                | Self::Xor { .. }
                | Self::Not { .. }
                | Self::Shl { .. }
                | Self::Shr { .. }
                | Self::Rol { .. }
                | Self::Ror { .. }
                | Self::Rcl { .. }
                | Self::Rcr { .. }
                | Self::BSwap { .. }
                | Self::BRev { .. }
                | Self::BitScanForward { .. }
                | Self::BitScanReverse { .. }
                | Self::Popcount { .. }
                | Self::Parity { .. }
                | Self::Select { .. }
                | Self::Ceq { .. }
                | Self::Clt { .. }
                | Self::Cgt { .. }
                | Self::Conv {
                    overflow_check: false,
                    ..
                }
                | Self::Copy { .. }
                | Self::SizeOf { .. }
                | Self::LoadToken { .. }
                | Self::LoadArg { .. }
                | Self::LoadLocal { .. }
                | Self::LoadArgAddr { .. }
                | Self::LoadLocalAddr { .. }
                | Self::Phi { .. }
                | Self::Nop
                | Self::Pop { .. }
                | Self::ReadFlags { .. }
        )
    }

    /// Replaces all uses of `old_var` with `new_var` in this operation.
    ///
    /// This is used for copy propagation and other variable substitution transformations.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable to replace.
    /// * `new_var` - The variable to use instead.
    ///
    /// # Returns
    ///
    /// The number of replacements made.
    pub fn replace_uses(&mut self, old_var: SsaVarId, new_var: SsaVarId) -> usize {
        let mut count: usize = 0;

        // Helper closure to replace a variable
        let mut replace = |var: &mut SsaVarId| {
            if *var == old_var {
                *var = new_var;
                count = count.saturating_add(1);
            }
        };

        match self {
            // Binary arithmetic and comparison branches
            Self::Add { left, right, .. }
            | Self::AddOvf { left, right, .. }
            | Self::Sub { left, right, .. }
            | Self::SubOvf { left, right, .. }
            | Self::Mul { left, right, .. }
            | Self::MulOvf { left, right, .. }
            | Self::Div { left, right, .. }
            | Self::Rem { left, right, .. }
            | Self::And { left, right, .. }
            | Self::Or { left, right, .. }
            | Self::Xor { left, right, .. }
            | Self::Ceq { left, right, .. }
            | Self::Clt { left, right, .. }
            | Self::Cgt { left, right, .. }
            | Self::BranchCmp { left, right, .. } => {
                replace(left);
                replace(right);
            }

            // Unary operations and conversion
            Self::Neg { operand, .. }
            | Self::Not { operand, .. }
            | Self::Ckfinite { operand, .. }
            | Self::Conv { operand, .. }
            | Self::BSwap { src: operand, .. }
            | Self::BRev { src: operand, .. }
            | Self::BitScanForward { src: operand, .. }
            | Self::BitScanReverse { src: operand, .. }
            | Self::Popcount { src: operand, .. }
            | Self::Parity { src: operand, .. } => {
                replace(operand);
            }

            // Shift and rotate operations
            Self::Shl { value, amount, .. }
            | Self::Shr { value, amount, .. }
            | Self::Rol { value, amount, .. }
            | Self::Ror { value, amount, .. }
            | Self::Rcl { value, amount, .. }
            | Self::Rcr { value, amount, .. } => {
                replace(value);
                replace(amount);
            }

            // Copy operation
            Self::Copy { src, .. } => {
                replace(src);
            }

            // Control flow
            Self::Branch { condition, .. } => {
                replace(condition);
            }
            Self::BranchFlags { flags, .. } => {
                replace(flags);
            }
            Self::ReadFlags { flags, .. } => {
                replace(flags);
            }
            Self::Select {
                condition,
                true_val,
                false_val,
                ..
            } => {
                replace(condition);
                replace(true_val);
                replace(false_val);
            }
            Self::Switch { value, .. }
            | Self::StoreStaticField { value, .. }
            | Self::Pop { value } => {
                replace(value);
            }
            Self::Return { value: Some(v) } => {
                replace(v);
            }

            // Object/field operations
            Self::LoadField { object, .. }
            | Self::LoadFieldAddr { object, .. }
            | Self::CastClass { object, .. }
            | Self::IsInst { object, .. }
            | Self::Box { value: object, .. }
            | Self::Unbox { object, .. }
            | Self::UnboxAny { object, .. }
            | Self::LoadVirtFunctionPtr { object, .. } => {
                replace(object);
            }
            Self::StoreField { object, value, .. } => {
                replace(object);
                replace(value);
            }

            // Array operations
            Self::LoadElement { array, index, .. } | Self::LoadElementAddr { array, index, .. } => {
                replace(array);
                replace(index);
            }
            Self::StoreElement {
                array,
                index,
                value,
                ..
            } => {
                replace(array);
                replace(index);
                replace(value);
            }
            Self::NewArr { length, .. } => {
                replace(length);
            }
            Self::ArrayLength { array, .. } => {
                replace(array);
            }

            // Indirect load/store
            Self::LoadIndirect { addr, .. } => {
                replace(addr);
            }
            Self::StoreIndirect { addr, value, .. } => {
                replace(addr);
                replace(value);
            }

            // Atomic operations
            Self::CmpXchg {
                addr,
                expected,
                desired,
                ..
            } => {
                replace(addr);
                replace(expected);
                replace(desired);
            }
            Self::AtomicRmw { addr, value, .. } => {
                replace(addr);
                replace(value);
            }

            // Calls
            Self::Call { args, .. } | Self::CallVirt { args, .. } | Self::NewObj { args, .. } => {
                for arg in args {
                    replace(arg);
                }
            }
            Self::CallIndirect { fptr, args, .. } => {
                replace(fptr);
                for arg in args {
                    replace(arg);
                }
            }

            // Other
            Self::Throw { exception } => {
                replace(exception);
            }
            Self::EndFilter { result } => {
                replace(result);
            }
            Self::Phi { operands, .. } => {
                for (_, operand) in operands {
                    replace(operand);
                }
            }
            Self::StoreObj {
                dest_addr, value, ..
            } => {
                replace(dest_addr);
                replace(value);
            }
            Self::LoadObj { src_addr, .. } => {
                replace(src_addr);
            }
            Self::LocalAlloc { size, .. } => {
                replace(size);
            }
            Self::InitObj { dest_addr, .. } => {
                replace(dest_addr);
            }
            Self::CopyObj {
                dest_addr,
                src_addr,
                ..
            } => {
                replace(dest_addr);
                replace(src_addr);
            }
            Self::CopyBlk {
                dest_addr,
                src_addr,
                size,
            } => {
                replace(dest_addr);
                replace(src_addr);
                replace(size);
            }
            Self::InitBlk {
                dest_addr,
                value,
                size,
            } => {
                replace(dest_addr);
                replace(value);
                replace(size);
            }

            // Operations without variable uses
            Self::Const { .. }
            | Self::LoadStaticField { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::Jump { .. }
            | Self::Return { value: None }
            | Self::Rethrow
            | Self::EndFinally
            | Self::Leave { .. }
            | Self::SizeOf { .. }
            | Self::LoadToken { .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::Nop
            | Self::Break
            | Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly
            | Self::Fence { .. }
            | Self::InterruptReturn
            | Self::Unreachable => {}
        }

        count
    }

    /// Remaps branch target block indices using the provided mapping function.
    ///
    /// This is used to translate RVA-based targets (from CIL instructions) to
    /// sequential block indices (used by the SSA representation).
    ///
    /// # Arguments
    ///
    /// * `remap` - A function that maps old block indices to new block indices.
    ///   Returns `None` if the target should remain unchanged.
    pub fn remap_branch_targets<F>(&mut self, remap: F)
    where
        F: Fn(usize) -> Option<usize>,
    {
        match self {
            Self::Jump { target } | Self::Leave { target } => {
                if let Some(new_target) = remap(*target) {
                    *target = new_target;
                }
            }
            Self::Branch {
                true_target,
                false_target,
                ..
            }
            | Self::BranchCmp {
                true_target,
                false_target,
                ..
            }
            | Self::BranchFlags {
                true_target,
                false_target,
                ..
            } => {
                if let Some(new_target) = remap(*true_target) {
                    *true_target = new_target;
                }
                if let Some(new_target) = remap(*false_target) {
                    *false_target = new_target;
                }
            }
            Self::Switch {
                targets, default, ..
            } => {
                for target in targets.iter_mut() {
                    if let Some(new_target) = remap(*target) {
                        *target = new_target;
                    }
                }
                if let Some(new_target) = remap(*default) {
                    *default = new_target;
                }
            }
            // All other operations don't have branch targets
            _ => {}
        }
    }

    /// Returns the successor block indices for this operation.
    ///
    /// For control flow operations (terminators), this returns the indices of
    /// all possible successor blocks:
    /// - `Jump` and `Leave`: single target block
    /// - `Branch`: true and false target blocks
    /// - `Switch`: all case targets plus the default target
    ///
    /// For non-terminator operations, returns an empty vector.
    ///
    /// # Returns
    ///
    /// A vector of successor block indices. Empty for non-branching operations.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{SsaOp, SsaVarId}};
    ///
    /// let var = SsaVarId::from_index(0);
    /// let op = SsaOp::<MockTarget>::Branch {
    ///     condition: var,
    ///     true_target: 1,
    ///     false_target: 2,
    /// };
    /// assert_eq!(op.successors(), vec![1, 2]);
    /// ```
    #[must_use]
    pub fn successors(&self) -> Vec<usize> {
        match self {
            Self::Jump { target } | Self::Leave { target } => vec![*target],
            Self::Branch {
                true_target,
                false_target,
                ..
            }
            | Self::BranchCmp {
                true_target,
                false_target,
                ..
            }
            | Self::BranchFlags {
                true_target,
                false_target,
                ..
            } => vec![*true_target, *false_target],
            Self::Switch {
                targets, default, ..
            } => {
                let mut succs = targets.clone();
                succs.push(*default);
                succs
            }
            // Return, Throw, Rethrow, EndFinally, EndFilter have no successors
            _ => vec![],
        }
    }

    /// Redirects control flow targets from `old_target` to `new_target`.
    ///
    /// This method modifies branch/jump targets in-place. It handles all control
    /// flow operations: `Jump`, `Leave`, `Branch`, `BranchCmp`, and `Switch`.
    ///
    /// # Arguments
    ///
    /// * `old_target` - The block index to redirect from
    /// * `new_target` - The block index to redirect to
    ///
    /// # Returns
    ///
    /// `true` if any target was changed, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{SsaOp, SsaVarId}};
    ///
    /// let var = SsaVarId::from_index(0);
    /// let mut op = SsaOp::<MockTarget>::Branch {
    ///     condition: var,
    ///     true_target: 2,
    ///     false_target: 3,
    /// };
    /// // Redirect all jumps to block 2 to instead go to block 5
    /// if op.redirect_target(2, 5) {
    ///     println!("Target redirected");
    /// }
    /// assert_eq!(op.successors(), vec![5, 3]);
    /// ```
    pub fn redirect_target(&mut self, old_target: usize, new_target: usize) -> bool {
        if old_target == new_target {
            return false;
        }

        match self {
            Self::Jump { target } | Self::Leave { target } if *target == old_target => {
                *target = new_target;
                true
            }
            Self::Branch {
                true_target,
                false_target,
                ..
            }
            | Self::BranchCmp {
                true_target,
                false_target,
                ..
            }
            | Self::BranchFlags {
                true_target,
                false_target,
                ..
            } => {
                let mut changed = false;
                if *true_target == old_target {
                    *true_target = new_target;
                    changed = true;
                }
                if *false_target == old_target {
                    *false_target = new_target;
                    changed = true;
                }
                changed
            }
            Self::Switch {
                targets, default, ..
            } => {
                let mut changed = false;
                if *default == old_target {
                    *default = new_target;
                    changed = true;
                }
                for target in targets.iter_mut() {
                    if *target == old_target {
                        *target = new_target;
                        changed = true;
                    }
                }
                changed
            }
            _ => false,
        }
    }

    /// Creates a clone of this operation with all variable IDs remapped.
    ///
    /// This is used for block duplication where all variable references
    /// (both destinations and uses) need to be updated to fresh IDs.
    ///
    /// # Arguments
    ///
    /// * `remap` - A function that maps old variable IDs to new ones.
    ///   If the function returns `None`, the original ID is kept.
    ///
    /// # Returns
    ///
    /// A new `SsaOp` with all variable IDs remapped.
    #[must_use]
    pub fn remap_variables<F>(&self, remap: F) -> Self
    where
        F: Fn(SsaVarId) -> Option<SsaVarId>,
    {
        // Helper to remap a single variable
        let r = |var: SsaVarId| remap(var).unwrap_or(var);

        match self.clone() {
            Self::Const { dest, value } => Self::Const {
                dest: r(dest),
                value,
            },

            Self::Add {
                dest,
                left,
                right,
                flags,
            } => Self::Add {
                dest: r(dest),
                left: r(left),
                right: r(right),
                flags: flags.map(r),
            },
            Self::AddOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Self::AddOvf {
                dest: r(dest),
                left: r(left),
                right: r(right),
                unsigned,
                flags: flags.map(r),
            },
            Self::Sub {
                dest,
                left,
                right,
                flags,
            } => Self::Sub {
                dest: r(dest),
                left: r(left),
                right: r(right),
                flags: flags.map(r),
            },
            Self::SubOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Self::SubOvf {
                dest: r(dest),
                left: r(left),
                right: r(right),
                unsigned,
                flags: flags.map(r),
            },
            Self::Mul {
                dest,
                left,
                right,
                flags,
            } => Self::Mul {
                dest: r(dest),
                left: r(left),
                right: r(right),
                flags: flags.map(r),
            },
            Self::MulOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Self::MulOvf {
                dest: r(dest),
                left: r(left),
                right: r(right),
                unsigned,
                flags: flags.map(r),
            },
            Self::Div {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Self::Div {
                dest: r(dest),
                left: r(left),
                right: r(right),
                unsigned,
                flags: flags.map(r),
            },
            Self::Rem {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Self::Rem {
                dest: r(dest),
                left: r(left),
                right: r(right),
                unsigned,
                flags: flags.map(r),
            },

            Self::Neg {
                dest,
                operand,
                flags,
            } => Self::Neg {
                dest: r(dest),
                operand: r(operand),
                flags: flags.map(r),
            },
            Self::And {
                dest,
                left,
                right,
                flags,
            } => Self::And {
                dest: r(dest),
                left: r(left),
                right: r(right),
                flags: flags.map(r),
            },
            Self::Or {
                dest,
                left,
                right,
                flags,
            } => Self::Or {
                dest: r(dest),
                left: r(left),
                right: r(right),
                flags: flags.map(r),
            },
            Self::Xor {
                dest,
                left,
                right,
                flags,
            } => Self::Xor {
                dest: r(dest),
                left: r(left),
                right: r(right),
                flags: flags.map(r),
            },
            Self::Not {
                dest,
                operand,
                flags,
            } => Self::Not {
                dest: r(dest),
                operand: r(operand),
                flags: flags.map(r),
            },

            Self::Shl {
                dest,
                value,
                amount,
                flags,
            } => Self::Shl {
                dest: r(dest),
                value: r(value),
                amount: r(amount),
                flags: flags.map(r),
            },
            Self::Shr {
                dest,
                value,
                amount,
                unsigned,
                flags,
            } => Self::Shr {
                dest: r(dest),
                value: r(value),
                amount: r(amount),
                unsigned,
                flags: flags.map(r),
            },

            Self::Rol {
                dest,
                value,
                amount,
            } => Self::Rol {
                dest: r(dest),
                value: r(value),
                amount: r(amount),
            },
            Self::Ror {
                dest,
                value,
                amount,
            } => Self::Ror {
                dest: r(dest),
                value: r(value),
                amount: r(amount),
            },
            Self::Rcl {
                dest,
                value,
                amount,
            } => Self::Rcl {
                dest: r(dest),
                value: r(value),
                amount: r(amount),
            },
            Self::Rcr {
                dest,
                value,
                amount,
            } => Self::Rcr {
                dest: r(dest),
                value: r(value),
                amount: r(amount),
            },
            Self::BSwap { dest, src } => Self::BSwap {
                dest: r(dest),
                src: r(src),
            },
            Self::BRev { dest, src } => Self::BRev {
                dest: r(dest),
                src: r(src),
            },
            Self::BitScanForward { dest, src } => Self::BitScanForward {
                dest: r(dest),
                src: r(src),
            },
            Self::BitScanReverse { dest, src } => Self::BitScanReverse {
                dest: r(dest),
                src: r(src),
            },
            Self::Popcount { dest, src } => Self::Popcount {
                dest: r(dest),
                src: r(src),
            },
            Self::Parity { dest, src } => Self::Parity {
                dest: r(dest),
                src: r(src),
            },

            Self::Ceq { dest, left, right } => Self::Ceq {
                dest: r(dest),
                left: r(left),
                right: r(right),
            },
            Self::Clt {
                dest,
                left,
                right,
                unsigned,
            } => Self::Clt {
                dest: r(dest),
                left: r(left),
                right: r(right),
                unsigned,
            },
            Self::Cgt {
                dest,
                left,
                right,
                unsigned,
            } => Self::Cgt {
                dest: r(dest),
                left: r(left),
                right: r(right),
                unsigned,
            },

            Self::Conv {
                dest,
                operand,
                target,
                overflow_check,
                unsigned,
            } => Self::Conv {
                dest: r(dest),
                operand: r(operand),
                target,
                overflow_check,
                unsigned,
            },
            Self::Ckfinite { dest, operand } => Self::Ckfinite {
                dest: r(dest),
                operand: r(operand),
            },
            Self::Select {
                dest,
                condition,
                true_val,
                false_val,
            } => Self::Select {
                dest: r(dest),
                condition: r(condition),
                true_val: r(true_val),
                false_val: r(false_val),
            },

            // Control flow - no dests, may have uses
            Self::Jump { target } => Self::Jump { target },
            Self::Branch {
                condition,
                true_target,
                false_target,
            } => Self::Branch {
                condition: r(condition),
                true_target,
                false_target,
            },
            Self::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                true_target,
                false_target,
            } => Self::BranchCmp {
                left: r(left),
                right: r(right),
                cmp,
                unsigned,
                true_target,
                false_target,
            },
            Self::BranchFlags {
                flags,
                condition,
                true_target,
                false_target,
            } => Self::BranchFlags {
                flags: r(flags),
                condition,
                true_target,
                false_target,
            },
            Self::ReadFlags { dest, flags, mask } => Self::ReadFlags {
                dest: r(dest),
                flags: r(flags),
                mask,
            },
            Self::Unreachable => Self::Unreachable,
            Self::Switch {
                value,
                targets,
                default,
            } => Self::Switch {
                value: r(value),
                targets,
                default,
            },
            Self::Return { value } => Self::Return {
                value: value.map(&r),
            },
            Self::Leave { target } => Self::Leave { target },

            // Field operations
            Self::LoadField {
                dest,
                object,
                field,
            } => Self::LoadField {
                dest: r(dest),
                object: r(object),
                field,
            },
            Self::StoreField {
                object,
                field,
                value,
            } => Self::StoreField {
                object: r(object),
                field,
                value: r(value),
            },
            Self::LoadStaticField { dest, field } => Self::LoadStaticField {
                dest: r(dest),
                field,
            },
            Self::StoreStaticField { field, value } => Self::StoreStaticField {
                field,
                value: r(value),
            },
            Self::LoadFieldAddr {
                dest,
                object,
                field,
            } => Self::LoadFieldAddr {
                dest: r(dest),
                object: r(object),
                field,
            },
            Self::LoadStaticFieldAddr { dest, field } => Self::LoadStaticFieldAddr {
                dest: r(dest),
                field,
            },

            // Array operations
            Self::LoadElement {
                dest,
                array,
                index,
                elem_type,
            } => Self::LoadElement {
                dest: r(dest),
                array: r(array),
                index: r(index),
                elem_type,
            },
            Self::StoreElement {
                array,
                index,
                value,
                elem_type,
            } => Self::StoreElement {
                array: r(array),
                index: r(index),
                value: r(value),
                elem_type,
            },
            Self::LoadElementAddr {
                dest,
                array,
                index,
                elem_type,
            } => Self::LoadElementAddr {
                dest: r(dest),
                array: r(array),
                index: r(index),
                elem_type,
            },
            Self::ArrayLength { dest, array } => Self::ArrayLength {
                dest: r(dest),
                array: r(array),
            },

            // Indirect operations
            Self::LoadIndirect {
                dest,
                addr,
                value_type,
            } => Self::LoadIndirect {
                dest: r(dest),
                addr: r(addr),
                value_type,
            },
            Self::StoreIndirect {
                addr,
                value,
                value_type,
            } => Self::StoreIndirect {
                addr: r(addr),
                value: r(value),
                value_type,
            },

            // Object operations
            Self::NewObj { dest, ctor, args } => Self::NewObj {
                dest: r(dest),
                ctor,
                args: args.into_iter().map(&r).collect(),
            },
            Self::NewArr {
                dest,
                elem_type,
                length,
            } => Self::NewArr {
                dest: r(dest),
                elem_type,
                length: r(length),
            },
            Self::CastClass {
                dest,
                object,
                target_type,
            } => Self::CastClass {
                dest: r(dest),
                object: r(object),
                target_type,
            },
            Self::IsInst {
                dest,
                object,
                target_type,
            } => Self::IsInst {
                dest: r(dest),
                object: r(object),
                target_type,
            },
            Self::Box {
                dest,
                value,
                value_type,
            } => Self::Box {
                dest: r(dest),
                value: r(value),
                value_type,
            },
            Self::Unbox {
                dest,
                object,
                value_type,
            } => Self::Unbox {
                dest: r(dest),
                object: r(object),
                value_type,
            },
            Self::UnboxAny {
                dest,
                object,
                value_type,
            } => Self::UnboxAny {
                dest: r(dest),
                object: r(object),
                value_type,
            },
            Self::SizeOf { dest, value_type } => Self::SizeOf {
                dest: r(dest),
                value_type,
            },
            Self::LoadToken { dest, token } => Self::LoadToken {
                dest: r(dest),
                token,
            },

            // Call operations
            Self::Call { dest, method, args } => Self::Call {
                dest: dest.map(&r),
                method,
                args: args.into_iter().map(&r).collect(),
            },
            Self::CallVirt { dest, method, args } => Self::CallVirt {
                dest: dest.map(&r),
                method,
                args: args.into_iter().map(&r).collect(),
            },
            Self::CallIndirect {
                dest,
                fptr,
                signature,
                args,
            } => Self::CallIndirect {
                dest: dest.map(&r),
                fptr: r(fptr),
                signature,
                args: args.into_iter().map(&r).collect(),
            },

            // Function pointer operations
            Self::LoadFunctionPtr { dest, method } => Self::LoadFunctionPtr {
                dest: r(dest),
                method,
            },
            Self::LoadVirtFunctionPtr {
                dest,
                object,
                method,
            } => Self::LoadVirtFunctionPtr {
                dest: r(dest),
                object: r(object),
                method,
            },

            // Value and address loading
            Self::LoadArg { dest, arg_index } => Self::LoadArg {
                dest: r(dest),
                arg_index,
            },
            Self::LoadLocal { dest, local_index } => Self::LoadLocal {
                dest: r(dest),
                local_index,
            },
            Self::LoadArgAddr { dest, arg_index } => Self::LoadArgAddr {
                dest: r(dest),
                arg_index,
            },
            Self::LoadLocalAddr { dest, local_index } => Self::LoadLocalAddr {
                dest: r(dest),
                local_index,
            },

            // Misc operations
            Self::Copy { dest, src } => Self::Copy {
                dest: r(dest),
                src: r(src),
            },
            Self::Pop { value } => Self::Pop { value: r(value) },
            Self::Throw { exception } => Self::Throw {
                exception: r(exception),
            },
            Self::Rethrow => Self::Rethrow,
            Self::EndFilter { result } => Self::EndFilter { result: r(result) },
            Self::EndFinally => Self::EndFinally,
            Self::InterruptReturn => Self::InterruptReturn,
            Self::Nop => Self::Nop,
            Self::Break => Self::Break,

            // Memory block operations
            Self::LocalAlloc { dest, size } => Self::LocalAlloc {
                dest: r(dest),
                size: r(size),
            },
            Self::InitObj {
                dest_addr,
                value_type,
            } => Self::InitObj {
                dest_addr: r(dest_addr),
                value_type,
            },
            Self::LoadObj {
                dest,
                src_addr,
                value_type,
            } => Self::LoadObj {
                dest: r(dest),
                src_addr: r(src_addr),
                value_type,
            },
            Self::StoreObj {
                dest_addr,
                value,
                value_type,
            } => Self::StoreObj {
                dest_addr: r(dest_addr),
                value: r(value),
                value_type,
            },
            Self::CopyObj {
                dest_addr,
                src_addr,
                value_type,
            } => Self::CopyObj {
                dest_addr: r(dest_addr),
                src_addr: r(src_addr),
                value_type,
            },
            Self::CopyBlk {
                dest_addr,
                src_addr,
                size,
            } => Self::CopyBlk {
                dest_addr: r(dest_addr),
                src_addr: r(src_addr),
                size: r(size),
            },
            Self::InitBlk {
                dest_addr,
                value,
                size,
            } => Self::InitBlk {
                dest_addr: r(dest_addr),
                value: r(value),
                size: r(size),
            },

            // Atomic operations
            Self::Fence { kind } => Self::Fence { kind },
            Self::CmpXchg {
                dest,
                addr,
                expected,
                desired,
            } => Self::CmpXchg {
                dest: r(dest),
                addr: r(addr),
                expected: r(expected),
                desired: r(desired),
            },
            Self::AtomicRmw {
                dest,
                addr,
                value,
                op,
            } => Self::AtomicRmw {
                dest: r(dest),
                addr: r(addr),
                value: r(value),
                op,
            },

            // Phi operations
            Self::Phi { dest, operands } => Self::Phi {
                dest: r(dest),
                operands: operands.into_iter().map(|(p, v)| (p, r(v))).collect(),
            },

            Self::Constrained { constraint_type } => Self::Constrained { constraint_type },
            Self::Volatile => Self::Volatile,
            Self::Unaligned { alignment } => Self::Unaligned { alignment },
            Self::TailPrefix => Self::TailPrefix,
            Self::Readonly => Self::Readonly,
        }
    }

    /// Extracts binary operation information if this is a binary operation.
    ///
    /// This method provides a uniform view of all binary operations (arithmetic,
    /// bitwise, comparison, shifts) for optimization passes that need to handle
    /// them generically.
    ///
    /// # Returns
    ///
    /// - `Some(BinaryOpInfo)` if this is a binary operation
    /// - `None` for all other operations
    ///
    /// # Supported Operations
    ///
    /// - Arithmetic: `Add`, `AddOvf`, `Sub`, `SubOvf`, `Mul`, `MulOvf`, `Div`, `Rem`
    /// - Bitwise: `And`, `Or`, `Xor`
    /// - Shifts: `Shl`, `Shr`
    /// - Comparisons: `Ceq`, `Clt`, `Cgt`
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{BinaryOpKind, SsaOp, SsaVarId}};
    ///
    /// let op = SsaOp::<MockTarget>::Add {
    ///     dest: SsaVarId::from_index(2),
    ///     left: SsaVarId::from_index(0),
    ///     right: SsaVarId::from_index(1),
    ///     flags: None,
    /// };
    /// match op.as_binary_op() {
    ///     Some(info) if info.kind == BinaryOpKind::Add => {
    ///         // Handle addition
    ///     }
    ///     Some(info) => {
    ///         // Handle other binary ops
    ///     }
    ///     None => {
    ///         // Not a binary operation
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn as_binary_op(&self) -> Option<BinaryOpInfo> {
        match *self {
            Self::Add {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Add,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::AddOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::AddOvf,
                dest,
                left,
                right,
                unsigned,
                flags,
            }),
            Self::Sub {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Sub,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::SubOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::SubOvf,
                dest,
                left,
                right,
                unsigned,
                flags,
            }),
            Self::Mul {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Mul,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::MulOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::MulOvf,
                dest,
                left,
                right,
                unsigned,
                flags,
            }),
            Self::Div {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Div,
                dest,
                left,
                right,
                unsigned,
                flags,
            }),
            Self::Rem {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Rem,
                dest,
                left,
                right,
                unsigned,
                flags,
            }),
            Self::And {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::And,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::Or {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Or,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::Xor {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Xor,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::Shl {
                dest,
                value,
                amount,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Shl,
                dest,
                left: value,
                right: amount,
                unsigned: false,
                flags,
            }),
            Self::Shr {
                dest,
                value,
                amount,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Shr,
                dest,
                left: value,
                right: amount,
                unsigned,
                flags,
            }),
            Self::Ceq { dest, left, right } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Ceq,
                dest,
                left,
                right,
                unsigned: false,
                flags: None,
            }),
            Self::Clt {
                dest,
                left,
                right,
                unsigned,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Clt,
                dest,
                left,
                right,
                unsigned,
                flags: None,
            }),
            Self::Cgt {
                dest,
                left,
                right,
                unsigned,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Cgt,
                dest,
                left,
                right,
                unsigned,
                flags: None,
            }),
            Self::Rol {
                dest,
                value,
                amount,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Rol,
                dest,
                left: value,
                right: amount,
                unsigned: false,
                flags: None,
            }),
            Self::Ror {
                dest,
                value,
                amount,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Ror,
                dest,
                left: value,
                right: amount,
                unsigned: false,
                flags: None,
            }),
            Self::Rcl {
                dest,
                value,
                amount,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Rcl,
                dest,
                left: value,
                right: amount,
                unsigned: false,
                flags: None,
            }),
            Self::Rcr {
                dest,
                value,
                amount,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Rcr,
                dest,
                left: value,
                right: amount,
                unsigned: false,
                flags: None,
            }),
            _ => None,
        }
    }

    /// Extracts unary operation information if this is a unary operation.
    ///
    /// This method provides a uniform view of all unary operations for
    /// optimization passes that need to handle them generically.
    ///
    /// # Returns
    ///
    /// - `Some(UnaryOpInfo)` if this is a unary operation
    /// - `None` for all other operations
    ///
    /// # Supported Operations
    ///
    /// - `Neg`: Negation
    /// - `Not`: Bitwise NOT
    /// - `Ckfinite`: Check finite
    ///
    /// # Note
    ///
    /// `Conv` is not included because it requires additional type information
    /// that doesn't fit the simple unary pattern.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{SsaOp, SsaVarId}};
    ///
    /// let op = SsaOp::<MockTarget>::Neg {
    ///     dest: SsaVarId::from_index(1),
    ///     operand: SsaVarId::from_index(0),
    ///     flags: None,
    /// };
    /// if let Some(info) = op.as_unary_op() {
    ///     println!("Unary {} on {}", info.kind, info.operand);
    /// }
    /// ```
    #[must_use]
    pub fn as_unary_op(&self) -> Option<UnaryOpInfo> {
        match *self {
            Self::Neg { dest, operand, .. } => Some(UnaryOpInfo {
                kind: UnaryOpKind::Neg,
                dest,
                operand,
            }),
            Self::Not { dest, operand, .. } => Some(UnaryOpInfo {
                kind: UnaryOpKind::Not,
                dest,
                operand,
            }),
            Self::Ckfinite { dest, operand } => Some(UnaryOpInfo {
                kind: UnaryOpKind::Ckfinite,
                dest,
                operand,
            }),
            Self::BSwap { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::BSwap,
                dest,
                operand: src,
            }),
            Self::BRev { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::BRev,
                dest,
                operand: src,
            }),
            Self::BitScanForward { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::BitScanForward,
                dest,
                operand: src,
            }),
            Self::BitScanReverse { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::BitScanReverse,
                dest,
                operand: src,
            }),
            Self::Popcount { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::Popcount,
                dest,
                operand: src,
            }),
            Self::Parity { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::Parity,
                dest,
                operand: src,
            }),
            _ => None,
        }
    }

    /// Returns the stack effect (pops, pushes) for this SSA operation.
    ///
    /// This represents the net effect on the evaluation stack when the operation
    /// is executed, assuming operands have already been loaded. The effect is:
    /// - pops: number of values consumed from the stack
    /// - pushes: number of values produced to the stack
    ///
    /// Note: This tracks the operation's own effect, not the loading of operands
    /// (which is tracked separately during codegen).
    #[must_use]
    pub fn stack_effect(&self) -> (u32, u32) {
        match self {
            // Binary arithmetic, comparisons, and array access - pop 2, push 1
            Self::Add { .. }
            | Self::Sub { .. }
            | Self::Mul { .. }
            | Self::Div { .. }
            | Self::Rem { .. }
            | Self::AddOvf { .. }
            | Self::SubOvf { .. }
            | Self::MulOvf { .. }
            | Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Shl { .. }
            | Self::Shr { .. }
            | Self::Rol { .. }
            | Self::Ror { .. }
            | Self::Rcl { .. }
            | Self::Rcr { .. }
            | Self::Ceq { .. }
            | Self::Clt { .. }
            | Self::Cgt { .. }
            | Self::LoadElement { .. }
            | Self::LoadElementAddr { .. } => (2, 1),

            // Pop 3, push 1 (Select, CmpXchg)
            Self::Select { .. } | Self::CmpXchg { .. } => (3, 1),

            // AtomicRmw is pop 2, push 1
            Self::AtomicRmw { .. } => (2, 1),

            // Control flow
            Self::Return { value } => {
                if value.is_some() {
                    (1, 0) // pop return value
                } else {
                    (0, 0) // void return
                }
            }
            // No stack effect (0, 0)
            Self::Jump { .. }
            | Self::Rethrow
            | Self::Leave { .. }
            | Self::EndFinally
            | Self::Copy { .. }
            | Self::Nop
            | Self::Break
            | Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly
            | Self::Phi { .. }
            | Self::Fence { .. }
            | Self::InterruptReturn
            | Self::Unreachable => (0, 0),

            // Pop 1, push 0 (1, 0)
            Self::Branch { .. }
            | Self::Switch { .. }
            | Self::Throw { .. }
            | Self::EndFilter { .. }
            | Self::Pop { .. }
            | Self::StoreStaticField { .. }
            | Self::InitObj { .. }
            | Self::BranchFlags { .. } => (1, 0),

            // Pop 2, push 0 (2, 0)
            Self::BranchCmp { .. }
            | Self::StoreField { .. }
            | Self::StoreIndirect { .. }
            | Self::StoreObj { .. }
            | Self::CopyObj { .. } => (2, 0),

            // Pop 3, push 0 (3, 0)
            Self::StoreElement { .. } | Self::InitBlk { .. } | Self::CopyBlk { .. } => (3, 0),

            // Pop 0, push 1 (0, 1)
            Self::LoadStaticField { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::SizeOf { .. }
            | Self::LoadToken { .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::Const { .. } => (0, 1),

            // Pop 1, push 1 (1, 1)
            Self::Neg { .. }
            | Self::Not { .. }
            | Self::Conv { .. }
            | Self::Ckfinite { .. }
            | Self::BSwap { .. }
            | Self::BRev { .. }
            | Self::BitScanForward { .. }
            | Self::BitScanReverse { .. }
            | Self::Popcount { .. }
            | Self::Parity { .. }
            | Self::LoadField { .. }
            | Self::LoadFieldAddr { .. }
            | Self::ArrayLength { .. }
            | Self::NewArr { .. }
            | Self::LoadIndirect { .. }
            | Self::LoadObj { .. }
            | Self::Box { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. }
            | Self::CastClass { .. }
            | Self::IsInst { .. }
            | Self::LoadVirtFunctionPtr { .. }
            | Self::LocalAlloc { .. }
            | Self::ReadFlags { .. } => (1, 1),

            // Call operations - stack effect depends on args and return type
            Self::Call { dest, args, .. } | Self::CallVirt { dest, args, .. } => {
                // args.len() will never exceed u32 for CIL methods
                #[allow(clippy::cast_possible_truncation)]
                let pops = args.len() as u32;
                let pushes = u32::from(dest.is_some());
                (pops, pushes)
            }
            Self::CallIndirect { dest, args, .. } => {
                // Indirect call pops args + function pointer
                // args.len() will never exceed u32 for CIL methods
                #[allow(clippy::cast_possible_truncation)]
                let pops = (args.len() as u32).saturating_add(1);
                let pushes = u32::from(dest.is_some());
                (pops, pushes)
            }
            Self::NewObj { args, .. } => {
                // newobj pops constructor args, always pushes new instance
                // args.len() will never exceed u32 for CIL methods
                #[allow(clippy::cast_possible_truncation)]
                let pops = args.len() as u32;
                (pops, 1)
            }
        }
    }
}

impl<T: Target> SsaOp<T> {
    /// Tries to infer the result type of this SSA operation.
    ///
    /// Dispatches per opcode group to small `Target` queries; each host
    /// supplies its own type lattice via `Target::result_type_for_const`,
    /// `arithmetic_result_type`, etc. Returns `None` for ops the host can't
    /// answer or for context-dependent ops resolved later from
    /// `SsaInstruction::result_type()`.
    #[must_use]
    pub fn infer_result_type(&self) -> Option<T::Type> {
        match self {
            Self::Const { value, .. } => T::result_type_for_const(value),
            // Type conversions carry their target.
            Self::Conv { target, .. } => Some(target.clone()),
            // Comparisons produce bool (or whatever the host defines).
            Self::Ceq { .. } | Self::Clt { .. } | Self::Cgt { .. } => T::comparison_result_type(),
            // Arithmetic/bitwise ops + SizeOf — per-host arithmetic type.
            Self::Add { .. }
            | Self::Sub { .. }
            | Self::Mul { .. }
            | Self::Div { .. }
            | Self::Rem { .. }
            | Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Shl { .. }
            | Self::Shr { .. }
            | Self::Rol { .. }
            | Self::Ror { .. }
            | Self::Rcl { .. }
            | Self::Rcr { .. }
            | Self::BSwap { .. }
            | Self::BRev { .. }
            | Self::Neg { .. }
            | Self::Not { .. }
            | Self::AddOvf { .. }
            | Self::SubOvf { .. }
            | Self::MulOvf { .. }
            | Self::SizeOf { .. } => T::arithmetic_result_type(),
            Self::UnboxAny { value_type, .. } | Self::LoadObj { value_type, .. } => {
                T::value_type_from_ref(value_type)
            }
            // Context-dependent ops — resolved from SsaInstruction::result_type().
            Self::LoadField { .. }
            | Self::LoadStaticField { .. }
            | Self::Call { dest: Some(_), .. }
            | Self::CallVirt { dest: Some(_), .. }
            | Self::CallIndirect { dest: Some(_), .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. } => None,
            Self::Box { .. }
            | Self::NewObj { .. }
            | Self::NewArr { .. }
            | Self::CastClass { .. }
            | Self::IsInst { .. } => T::object_result_type(),
            Self::ArrayLength { .. }
            | Self::LocalAlloc { .. }
            | Self::BitScanForward { .. }
            | Self::BitScanReverse { .. }
            | Self::Popcount { .. } => T::native_int_result_type(),
            Self::Ckfinite { .. } => T::ckfinite_result_type(),
            Self::Parity { .. } | Self::ReadFlags { .. } => T::comparison_result_type(),
            Self::LoadFunctionPtr { .. } | Self::LoadVirtFunctionPtr { .. } => {
                T::function_ptr_result_type()
            }
            Self::LoadElement { elem_type, .. } => Some(elem_type.clone()),
            Self::LoadIndirect { value_type, .. } => Some(value_type.clone()),
            // LoadToken: assembly-free inference is `None`; the variable's
            // declared type set during SSA construction provides the real one.
            Self::LoadToken { .. } => None,
            Self::Unbox { value_type, .. } => T::byref_value_type_from_ref(value_type),
            Self::LoadElementAddr { elem_type, .. } => T::byref_class_type_from_ref(elem_type),
            Self::LoadFieldAddr { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. } => None,
            _ => None,
        }
    }
}

impl<T: Target> fmt::Display for SsaOp<T>
where
    T::TypeRef: fmt::Display,
    T::MethodRef: fmt::Display,
    T::FieldRef: fmt::Display,
    T::SigRef: fmt::Display,
    T::Type: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const { dest, value } => write!(f, "{dest} = {value}"),
            Self::Add {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = add {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = add {left}, {right}")
                }
            }
            Self::AddOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = add.ovf{suffix} {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = add.ovf{suffix} {left}, {right}")
                }
            }
            Self::Sub {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = sub {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = sub {left}, {right}")
                }
            }
            Self::SubOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = sub.ovf{suffix} {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = sub.ovf{suffix} {left}, {right}")
                }
            }
            Self::Mul {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = mul {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = mul {left}, {right}")
                }
            }
            Self::MulOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = mul.ovf{suffix} {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = mul.ovf{suffix} {left}, {right}")
                }
            }
            Self::Div {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = div{suffix} {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = div{suffix} {left}, {right}")
                }
            }
            Self::Rem {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = rem{suffix} {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = rem{suffix} {left}, {right}")
                }
            }
            Self::Neg {
                dest,
                operand,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = neg {operand} flags={flags}")
                } else {
                    write!(f, "{dest} = neg {operand}")
                }
            }
            Self::And {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = and {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = and {left}, {right}")
                }
            }
            Self::Or {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = or {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = or {left}, {right}")
                }
            }
            Self::Xor {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = xor {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = xor {left}, {right}")
                }
            }
            Self::Not {
                dest,
                operand,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = not {operand} flags={flags}")
                } else {
                    write!(f, "{dest} = not {operand}")
                }
            }
            Self::Shl {
                dest,
                value,
                amount,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = shl {value}, {amount} flags={flags}")
                } else {
                    write!(f, "{dest} = shl {value}, {amount}")
                }
            }
            Self::Shr {
                dest,
                value,
                amount,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = shr{suffix} {value}, {amount} flags={flags}")
                } else {
                    write!(f, "{dest} = shr{suffix} {value}, {amount}")
                }
            }
            Self::Rol {
                dest,
                value,
                amount,
            } => write!(f, "{dest} = rol {value}, {amount}"),
            Self::Ror {
                dest,
                value,
                amount,
            } => write!(f, "{dest} = ror {value}, {amount}"),
            Self::Rcl {
                dest,
                value,
                amount,
            } => write!(f, "{dest} = rcl {value}, {amount}"),
            Self::Rcr {
                dest,
                value,
                amount,
            } => write!(f, "{dest} = rcr {value}, {amount}"),
            Self::BSwap { dest, src } => write!(f, "{dest} = bswap {src}"),
            Self::BRev { dest, src } => write!(f, "{dest} = brev {src}"),
            Self::BitScanForward { dest, src } => write!(f, "{dest} = bsf {src}"),
            Self::BitScanReverse { dest, src } => write!(f, "{dest} = bsr {src}"),
            Self::Popcount { dest, src } => write!(f, "{dest} = popcnt {src}"),
            Self::Parity { dest, src } => write!(f, "{dest} = parity {src}"),
            Self::Ceq { dest, left, right } => write!(f, "{dest} = ceq {left}, {right}"),
            Self::Clt {
                dest,
                left,
                right,
                unsigned,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(f, "{dest} = clt{suffix} {left}, {right}")
            }
            Self::Cgt {
                dest,
                left,
                right,
                unsigned,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(f, "{dest} = cgt{suffix} {left}, {right}")
            }
            Self::Conv {
                dest,
                operand,
                target,
                ..
            } => write!(f, "{dest} = conv.{target} {operand}"),
            Self::ReadFlags { dest, flags, mask } => {
                write!(f, "{dest} = readflags {flags}, {mask}")
            }
            Self::Select {
                dest,
                condition,
                true_val,
                false_val,
            } => write!(f, "{dest} = select {condition}, {true_val}, {false_val}"),
            Self::Jump { target } => write!(f, "jump B{target}"),
            Self::Branch {
                condition,
                true_target,
                false_target,
            } => write!(f, "branch {condition}, B{true_target}, B{false_target}"),
            Self::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                true_target,
                false_target,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(
                    f,
                    "branchcmp{suffix} {left} {cmp} {right}, B{true_target}, B{false_target}"
                )
            }
            Self::BranchFlags {
                flags: _,
                condition,
                true_target,
                false_target,
            } => {
                write!(f, "branchflags {condition} B{true_target}, B{false_target}")
            }
            Self::Switch {
                value,
                targets,
                default,
            } => {
                write!(f, "switch {value}, [")?;
                for (i, t) in targets.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "B{t}")?;
                }
                write!(f, "], B{default}")
            }
            Self::Return { value: Some(v) } => write!(f, "ret {v}"),
            Self::Return { value: None } => write!(f, "ret"),
            Self::LoadField {
                dest,
                object,
                field,
            } => {
                write!(f, "{dest} = ldfld {field}, {object}")
            }
            Self::StoreField {
                object,
                field,
                value,
            } => write!(f, "stfld {field}, {object}, {value}"),
            Self::LoadStaticField { dest, field } => write!(f, "{dest} = ldsfld {field}"),
            Self::StoreStaticField { field, value } => write!(f, "stsfld {field}, {value}"),
            Self::LoadFieldAddr {
                dest,
                object,
                field,
            } => {
                write!(f, "{dest} = ldflda {field}, {object}")
            }
            Self::LoadStaticFieldAddr { dest, field } => write!(f, "{dest} = ldsflda {field}"),
            Self::LoadElement {
                dest,
                array,
                index,
                elem_type,
            } => write!(f, "{dest} = ldelem.{elem_type} {array}[{index}]"),
            Self::StoreElement {
                array,
                index,
                value,
                elem_type,
            } => write!(f, "stelem.{elem_type} {array}[{index}], {value}"),
            Self::LoadElementAddr {
                dest, array, index, ..
            } => write!(f, "{dest} = ldelema {array}[{index}]"),
            Self::ArrayLength { dest, array } => write!(f, "{dest} = ldlen {array}"),
            Self::LoadIndirect {
                dest,
                addr,
                value_type,
            } => write!(f, "{dest} = ldind.{value_type} {addr}"),
            Self::StoreIndirect {
                addr,
                value,
                value_type,
            } => write!(f, "stind.{value_type} {addr}, {value}"),
            Self::NewObj { dest, ctor, args } => {
                write!(f, "{dest} = newobj {ctor}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Self::NewArr {
                dest,
                elem_type,
                length,
            } => write!(f, "{dest} = newarr {elem_type}[{length}]"),
            Self::CastClass {
                dest,
                object,
                target_type,
            } => write!(f, "{dest} = castclass {target_type}, {object}"),
            Self::IsInst {
                dest,
                object,
                target_type,
            } => write!(f, "{dest} = isinst {target_type}, {object}"),
            Self::Box {
                dest,
                value,
                value_type,
            } => write!(f, "{dest} = box {value_type}, {value}"),
            Self::Unbox {
                dest,
                object,
                value_type,
            } => write!(f, "{dest} = unbox {value_type}, {object}"),
            Self::UnboxAny {
                dest,
                object,
                value_type,
            } => write!(f, "{dest} = unbox.any {value_type}, {object}"),
            Self::SizeOf { dest, value_type } => write!(f, "{dest} = sizeof {value_type}"),
            Self::LoadToken { dest, token } => write!(f, "{dest} = ldtoken {token}"),
            Self::Call { dest, method, args } => {
                if let Some(d) = dest {
                    write!(f, "{d} = ")?;
                }
                write!(f, "call {method}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Self::CallVirt { dest, method, args } => {
                if let Some(d) = dest {
                    write!(f, "{d} = ")?;
                }
                write!(f, "callvirt {method}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Self::CallIndirect {
                dest, fptr, args, ..
            } => {
                if let Some(d) = dest {
                    write!(f, "{d} = ")?;
                }
                write!(f, "calli {fptr}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Self::LoadFunctionPtr { dest, method } => write!(f, "{dest} = ldftn {method}"),
            Self::LoadVirtFunctionPtr {
                dest,
                object,
                method,
            } => write!(f, "{dest} = ldvirtftn {method}, {object}"),
            Self::LoadArg { dest, arg_index } => write!(f, "{dest} = ldarg {arg_index}"),
            Self::LoadLocal { dest, local_index } => write!(f, "{dest} = ldloc {local_index}"),
            Self::LoadArgAddr { dest, arg_index } => write!(f, "{dest} = ldarga {arg_index}"),
            Self::LoadLocalAddr { dest, local_index } => {
                write!(f, "{dest} = ldloca {local_index}")
            }
            Self::Copy { dest, src } => write!(f, "{dest} = {src}"),
            Self::Pop { value } => write!(f, "pop {value}"),
            Self::Throw { exception } => write!(f, "throw {exception}"),
            Self::Rethrow => write!(f, "rethrow"),
            Self::EndFinally => write!(f, "endfinally"),
            Self::InterruptReturn => write!(f, "iret"),
            Self::Unreachable => write!(f, "unreachable"),
            Self::EndFilter { result } => write!(f, "endfilter {result}"),
            Self::Leave { target } => write!(f, "leave B{target}"),
            Self::InitBlk {
                dest_addr,
                value,
                size,
            } => write!(f, "initblk {dest_addr}, {value}, {size}"),
            Self::CopyBlk {
                dest_addr,
                src_addr,
                size,
            } => write!(f, "cpblk {dest_addr}, {src_addr}, {size}"),
            Self::Fence { kind } => write!(f, "fence {kind}"),
            Self::CmpXchg {
                dest,
                addr,
                expected,
                desired,
            } => write!(f, "{dest} = cmpxchg {addr}, {expected}, {desired}"),
            Self::AtomicRmw {
                dest,
                addr,
                value,
                op,
            } => write!(f, "{dest} = atomicrmw.{op} {addr}, {value}"),
            Self::InitObj {
                dest_addr,
                value_type,
            } => write!(f, "initobj {value_type}, {dest_addr}"),
            Self::CopyObj {
                dest_addr,
                src_addr,
                value_type,
            } => write!(f, "cpobj {value_type}, {dest_addr}, {src_addr}"),
            Self::LoadObj {
                dest,
                src_addr,
                value_type,
            } => write!(f, "{dest} = ldobj {value_type}, {src_addr}"),
            Self::StoreObj {
                dest_addr,
                value,
                value_type,
            } => write!(f, "stobj {value_type}, {dest_addr}, {value}"),
            Self::LocalAlloc { dest, size } => write!(f, "{dest} = localloc {size}"),
            Self::Constrained { constraint_type } => {
                write!(f, "constrained. {constraint_type}")
            }
            Self::Volatile => write!(f, "volatile."),
            Self::Unaligned { alignment } => write!(f, "unaligned. {alignment}"),
            Self::TailPrefix => write!(f, "tail."),
            Self::Readonly => write!(f, "readonly."),
            Self::Ckfinite { dest, operand } => write!(f, "{dest} = ckfinite {operand}"),
            Self::Nop => write!(f, "nop"),
            Self::Break => write!(f, "break"),
            Self::Phi { dest, operands } => {
                write!(f, "{dest} = phi(")?;
                for (i, (block, var)) in operands.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "B{block}: {var}")?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        ir::{value::ConstValue, variable::SsaVarId},
        testing::MockTarget,
    };

    /// Pure ops are hoistable; impure ones (calls) are not.
    #[test]
    fn is_pure_classifies_calls_and_arith() {
        let add: SsaOp<MockTarget> = SsaOp::Add {
            dest: SsaVarId::from_index(0),
            left: SsaVarId::from_index(1),
            right: SsaVarId::from_index(2),
            flags: None,
        };
        assert!(add.is_pure());

        let const_op: SsaOp<MockTarget> = SsaOp::Const {
            dest: SsaVarId::from_index(3),
            value: ConstValue::I32(42),
        };
        assert!(const_op.is_pure());

        let call: SsaOp<MockTarget> = SsaOp::Call {
            dest: Some(SsaVarId::from_index(4)),
            method: 0xAB,
            args: vec![],
        };
        assert!(!call.is_pure());
    }

    /// `Add` reports both operands; `Const` reports none.
    #[test]
    fn uses_lists_operands() {
        let v1 = SsaVarId::from_index(0);
        let v2 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);

        let op: SsaOp<MockTarget> = SsaOp::Add {
            dest,
            left: v1,
            right: v2,
            flags: None,
        };
        let uses = op.uses();
        assert_eq!(uses.len(), 2);
        assert!(uses.contains(&v1));
        assert!(uses.contains(&v2));

        let const_op: SsaOp<MockTarget> = SsaOp::Const {
            dest,
            value: ConstValue::I32(42),
        };
        assert!(const_op.uses().is_empty());
    }

    /// New rotate ops report dest and correct uses.
    #[test]
    fn rotate_ops_dest_and_uses() {
        let d = SsaVarId::from_index(0);
        let v = SsaVarId::from_index(1);
        let a = SsaVarId::from_index(2);

        let rol: SsaOp<MockTarget> = SsaOp::Rol {
            dest: d,
            value: v,
            amount: a,
        };
        assert_eq!(rol.dest(), Some(d));
        let uses = rol.uses();
        assert_eq!(uses.len(), 2);
        assert!(uses.contains(&v));
        assert!(uses.contains(&a));

        let ror: SsaOp<MockTarget> = SsaOp::Ror {
            dest: d,
            value: v,
            amount: a,
        };
        assert_eq!(ror.dest(), Some(d));
        assert!(ror.uses().contains(&v));
    }

    /// New bit-manip unary ops report dest and correct uses.
    #[test]
    fn bit_manip_ops_dest_and_uses() {
        let d = SsaVarId::from_index(0);
        let s = SsaVarId::from_index(1);

        let bswap: SsaOp<MockTarget> = SsaOp::BSwap { dest: d, src: s };
        assert_eq!(bswap.dest(), Some(d));
        assert_eq!(bswap.uses(), vec![s]);

        let brev: SsaOp<MockTarget> = SsaOp::BRev { dest: d, src: s };
        assert_eq!(brev.dest(), Some(d));
        assert_eq!(brev.uses(), vec![s]);

        let bsf: SsaOp<MockTarget> = SsaOp::BitScanForward { dest: d, src: s };
        assert_eq!(bsf.dest(), Some(d));
        assert_eq!(bsf.uses(), vec![s]);

        let bsr: SsaOp<MockTarget> = SsaOp::BitScanReverse { dest: d, src: s };
        assert_eq!(bsr.dest(), Some(d));
        assert_eq!(bsr.uses(), vec![s]);

        let popcnt: SsaOp<MockTarget> = SsaOp::Popcount { dest: d, src: s };
        assert_eq!(popcnt.dest(), Some(d));
        assert_eq!(popcnt.uses(), vec![s]);

        let parity: SsaOp<MockTarget> = SsaOp::Parity { dest: d, src: s };
        assert_eq!(parity.dest(), Some(d));
        assert_eq!(parity.uses(), vec![s]);
    }

    /// Select reports dest and three uses.
    #[test]
    fn select_dest_and_uses() {
        let d = SsaVarId::from_index(0);
        let c = SsaVarId::from_index(1);
        let t = SsaVarId::from_index(2);
        let f = SsaVarId::from_index(3);

        let op: SsaOp<MockTarget> = SsaOp::Select {
            dest: d,
            condition: c,
            true_val: t,
            false_val: f,
        };
        assert_eq!(op.dest(), Some(d));
        assert_eq!(op.uses().len(), 3);
        assert!(op.uses().contains(&c));
        assert!(op.uses().contains(&t));
        assert!(op.uses().contains(&f));
    }

    /// Atomic ops report dest and correct uses.
    #[test]
    fn atomic_ops_dest_and_uses() {
        let d = SsaVarId::from_index(0);
        let a = SsaVarId::from_index(1);
        let e = SsaVarId::from_index(2);
        let v = SsaVarId::from_index(3);

        let op: SsaOp<MockTarget> = SsaOp::CmpXchg {
            dest: d,
            addr: a,
            expected: e,
            desired: v,
        };
        assert_eq!(op.dest(), Some(d));
        assert_eq!(op.uses().len(), 3);

        let op2: SsaOp<MockTarget> = SsaOp::AtomicRmw {
            dest: d,
            addr: a,
            value: v,
            op: AtomicRmwOp::Xchg,
        };
        assert_eq!(op2.dest(), Some(d));
        assert_eq!(op2.uses().len(), 2);
    }

    /// Fence and InterruptReturn have no dest and no uses.
    #[test]
    fn fence_and_iret_no_dest_no_uses() {
        let fence: SsaOp<MockTarget> = SsaOp::Fence {
            kind: FenceKind::Full,
        };
        assert_eq!(fence.dest(), None);
        assert!(fence.uses().is_empty());

        let iret: SsaOp<MockTarget> = SsaOp::InterruptReturn;
        assert_eq!(iret.dest(), None);
        assert!(iret.uses().is_empty());
    }

    /// New pure ops are classified as pure.
    #[test]
    fn new_pure_ops_classification() {
        let d = SsaVarId::from_index(0);
        let v = SsaVarId::from_index(1);
        let a = SsaVarId::from_index(2);

        assert!(SsaOp::<MockTarget>::Rol {
            dest: d,
            value: v,
            amount: a
        }
        .is_pure());
        assert!(SsaOp::<MockTarget>::Ror {
            dest: d,
            value: v,
            amount: a
        }
        .is_pure());
        assert!(SsaOp::<MockTarget>::Rcl {
            dest: d,
            value: v,
            amount: a
        }
        .is_pure());
        assert!(SsaOp::<MockTarget>::Rcr {
            dest: d,
            value: v,
            amount: a
        }
        .is_pure());
        assert!(SsaOp::<MockTarget>::BSwap { dest: d, src: v }.is_pure());
        assert!(SsaOp::<MockTarget>::BRev { dest: d, src: v }.is_pure());
        assert!(SsaOp::<MockTarget>::BitScanForward { dest: d, src: v }.is_pure());
        assert!(SsaOp::<MockTarget>::BitScanReverse { dest: d, src: v }.is_pure());
        assert!(SsaOp::<MockTarget>::Popcount { dest: d, src: v }.is_pure());
        assert!(SsaOp::<MockTarget>::Parity { dest: d, src: v }.is_pure());
        assert!(SsaOp::<MockTarget>::Select {
            dest: d,
            condition: v,
            true_val: a,
            false_val: d,
        }
        .is_pure());
    }

    /// Side-effecting new ops are not pure.
    #[test]
    fn new_impure_ops_classification() {
        assert!(!SsaOp::<MockTarget>::Fence {
            kind: FenceKind::Full,
        }
        .is_pure());
        assert!(!SsaOp::<MockTarget>::InterruptReturn.is_pure());
        assert!(!SsaOp::<MockTarget>::CmpXchg {
            dest: SsaVarId::from_index(0),
            addr: SsaVarId::from_index(1),
            expected: SsaVarId::from_index(2),
            desired: SsaVarId::from_index(3),
        }
        .is_pure());
        assert!(!SsaOp::<MockTarget>::AtomicRmw {
            dest: SsaVarId::from_index(0),
            addr: SsaVarId::from_index(1),
            value: SsaVarId::from_index(2),
            op: AtomicRmwOp::Add,
        }
        .is_pure());
    }

    /// InterruptReturn is a terminator.
    #[test]
    fn interrupt_return_is_terminator() {
        assert!(SsaOp::<MockTarget>::InterruptReturn.is_terminator());
    }

    /// CmpXchg and AtomicRmw may throw.
    #[test]
    fn atomic_ops_may_throw() {
        assert!(SsaOp::<MockTarget>::CmpXchg {
            dest: SsaVarId::from_index(0),
            addr: SsaVarId::from_index(1),
            expected: SsaVarId::from_index(2),
            desired: SsaVarId::from_index(3),
        }
        .may_throw());
        assert!(SsaOp::<MockTarget>::AtomicRmw {
            dest: SsaVarId::from_index(0),
            addr: SsaVarId::from_index(1),
            value: SsaVarId::from_index(2),
            op: AtomicRmwOp::Xchg,
        }
        .may_throw());
    }

    /// New pure ops do not throw.
    #[test]
    fn new_pure_ops_no_throw() {
        let d = SsaVarId::from_index(0);
        let v = SsaVarId::from_index(1);
        assert!(!SsaOp::<MockTarget>::Rol {
            dest: d,
            value: v,
            amount: v
        }
        .may_throw());
        assert!(!SsaOp::<MockTarget>::BSwap { dest: d, src: v }.may_throw());
        assert!(!SsaOp::<MockTarget>::Select {
            dest: d,
            condition: v,
            true_val: v,
            false_val: v,
        }
        .may_throw());
    }

    /// as_binary_op returns info for rotate ops.
    #[test]
    fn as_binary_op_rotations() {
        let d = SsaVarId::from_index(0);
        let v = SsaVarId::from_index(1);
        let a = SsaVarId::from_index(2);

        let rol = SsaOp::<MockTarget>::Rol {
            dest: d,
            value: v,
            amount: a,
        };
        let info = rol.as_binary_op().unwrap();
        assert_eq!(info.kind, BinaryOpKind::Rol);
        assert_eq!(info.dest, d);
        assert_eq!(info.left, v);
        assert_eq!(info.right, a);

        let ror = SsaOp::<MockTarget>::Ror {
            dest: d,
            value: v,
            amount: a,
        };
        let info = ror.as_binary_op().unwrap();
        assert_eq!(info.kind, BinaryOpKind::Ror);
    }

    /// as_unary_op returns info for bit-manip ops.
    #[test]
    fn as_unary_op_bit_manip() {
        let d = SsaVarId::from_index(0);
        let s = SsaVarId::from_index(1);

        let bswap = SsaOp::<MockTarget>::BSwap { dest: d, src: s };
        let info = bswap.as_unary_op().unwrap();
        assert_eq!(info.kind, UnaryOpKind::BSwap);
        assert_eq!(info.dest, d);
        assert_eq!(info.operand, s);

        let brev = SsaOp::<MockTarget>::BRev { dest: d, src: s };
        assert_eq!(brev.as_unary_op().unwrap().kind, UnaryOpKind::BRev);

        let bsf = SsaOp::<MockTarget>::BitScanForward { dest: d, src: s };
        assert_eq!(bsf.as_unary_op().unwrap().kind, UnaryOpKind::BitScanForward);

        let bsr = SsaOp::<MockTarget>::BitScanReverse { dest: d, src: s };
        assert_eq!(bsr.as_unary_op().unwrap().kind, UnaryOpKind::BitScanReverse);

        let popcnt = SsaOp::<MockTarget>::Popcount { dest: d, src: s };
        assert_eq!(popcnt.as_unary_op().unwrap().kind, UnaryOpKind::Popcount);

        let parity = SsaOp::<MockTarget>::Parity { dest: d, src: s };
        assert_eq!(parity.as_unary_op().unwrap().kind, UnaryOpKind::Parity);
    }

    /// stack_effect for new ops.
    #[test]
    fn stack_effect_new_ops() {
        let d = SsaVarId::from_index(0);
        let v = SsaVarId::from_index(1);
        let a = SsaVarId::from_index(2);

        // Rotates: pop 2, push 1
        assert_eq!(
            SsaOp::<MockTarget>::Rol {
                dest: d,
                value: v,
                amount: a
            }
            .stack_effect(),
            (2, 1)
        );
        assert_eq!(
            SsaOp::<MockTarget>::Ror {
                dest: d,
                value: v,
                amount: a
            }
            .stack_effect(),
            (2, 1)
        );

        // Bit-manip: pop 1, push 1
        assert_eq!(
            SsaOp::<MockTarget>::BSwap { dest: d, src: v }.stack_effect(),
            (1, 1)
        );
        assert_eq!(
            SsaOp::<MockTarget>::Popcount { dest: d, src: v }.stack_effect(),
            (1, 1)
        );

        // Select: pop 3, push 1
        assert_eq!(
            SsaOp::<MockTarget>::Select {
                dest: d,
                condition: v,
                true_val: a,
                false_val: d,
            }
            .stack_effect(),
            (3, 1)
        );

        // CmpXchg: pop 3, push 1
        assert_eq!(
            SsaOp::<MockTarget>::CmpXchg {
                dest: d,
                addr: v,
                expected: a,
                desired: d,
            }
            .stack_effect(),
            (3, 1)
        );

        // AtomicRmw: pop 2, push 1
        assert_eq!(
            SsaOp::<MockTarget>::AtomicRmw {
                dest: d,
                addr: v,
                value: a,
                op: AtomicRmwOp::Add,
            }
            .stack_effect(),
            (2, 1)
        );

        // Fence: pop 0, push 0
        assert_eq!(
            SsaOp::<MockTarget>::Fence {
                kind: FenceKind::SeqCst
            }
            .stack_effect(),
            (0, 0)
        );

        // InterruptReturn: pop 0, push 0
        assert_eq!(SsaOp::<MockTarget>::InterruptReturn.stack_effect(), (0, 0));
    }

    /// replace_uses works for new ops.
    #[test]
    fn replace_uses_new_ops() {
        let d = SsaVarId::from_index(0);
        let old = SsaVarId::from_index(1);
        let new = SsaVarId::from_index(99);
        let other = SsaVarId::from_index(2);

        // Test rotation
        let mut op: SsaOp<MockTarget> = SsaOp::Rol {
            dest: d,
            value: old,
            amount: other,
        };
        assert_eq!(op.replace_uses(old, new), 1);
        assert_eq!(op.uses(), vec![new, other]);

        // Test bit-manip
        let mut op2: SsaOp<MockTarget> = SsaOp::BSwap { dest: d, src: old };
        assert_eq!(op2.replace_uses(old, new), 1);
        assert_eq!(op2.uses(), vec![new]);

        // Test Select
        let mut op3: SsaOp<MockTarget> = SsaOp::Select {
            dest: d,
            condition: old,
            true_val: other,
            false_val: old,
        };
        assert_eq!(op3.replace_uses(old, new), 2);
        assert_eq!(op3.uses(), vec![new, other, new]);

        // Test CmpXchg
        let mut op4: SsaOp<MockTarget> = SsaOp::CmpXchg {
            dest: d,
            addr: old,
            expected: other,
            desired: old,
        };
        assert_eq!(op4.replace_uses(old, new), 2);

        // Test AtomicRmw
        let mut op5: SsaOp<MockTarget> = SsaOp::AtomicRmw {
            dest: d,
            addr: old,
            value: other,
            op: AtomicRmwOp::Xor,
        };
        assert_eq!(op5.replace_uses(old, new), 1);
    }

    /// set_dest works for new dest-bearing ops.
    #[test]
    fn set_dest_new_ops() {
        let d = SsaVarId::from_index(0);
        let new_d = SsaVarId::from_index(99);
        let v = SsaVarId::from_index(1);
        let a = SsaVarId::from_index(2);

        let mut rol: SsaOp<MockTarget> = SsaOp::Rol {
            dest: d,
            value: v,
            amount: a,
        };
        assert!(rol.set_dest(new_d));
        assert_eq!(rol.dest(), Some(new_d));

        let mut bswap: SsaOp<MockTarget> = SsaOp::BSwap { dest: d, src: v };
        assert!(bswap.set_dest(new_d));
        assert_eq!(bswap.dest(), Some(new_d));

        let mut select: SsaOp<MockTarget> = SsaOp::Select {
            dest: d,
            condition: v,
            true_val: a,
            false_val: d,
        };
        assert!(select.set_dest(new_d));
        assert_eq!(select.dest(), Some(new_d));
    }

    /// set_dest returns false for ops without dest.
    #[test]
    fn set_dest_fails_for_no_dest_ops() {
        assert!(!SsaOp::<MockTarget>::Fence {
            kind: FenceKind::Full
        }
        .set_dest(SsaVarId::from_index(0)));
        assert!(!SsaOp::<MockTarget>::InterruptReturn.set_dest(SsaVarId::from_index(0)));
    }

    /// remap_variables works for new ops.
    #[test]
    fn remap_variables_new_ops() {
        use std::collections::HashMap;
        let d0 = SsaVarId::from_index(0);
        let d99 = SsaVarId::from_index(99);
        let v1 = SsaVarId::from_index(1);
        let v55 = SsaVarId::from_index(55);
        let a2 = SsaVarId::from_index(2);

        let mut map = HashMap::new();
        map.insert(d0, d99);
        map.insert(v1, v55);
        let remap = |v: SsaVarId| map.get(&v).copied();

        let rol = SsaOp::<MockTarget>::Rol {
            dest: d0,
            value: v1,
            amount: a2,
        };
        let remapped = rol.remap_variables(remap);
        assert_eq!(remapped.dest(), Some(d99));
        assert!(remapped.uses().contains(&v55));
        assert!(remapped.uses().contains(&a2));
    }

    /// FenceKind display.
    #[test]
    fn fence_kind_display() {
        assert_eq!(format!("{}", FenceKind::Full), "full");
        assert_eq!(format!("{}", FenceKind::Acquire), "acquire");
        assert_eq!(format!("{}", FenceKind::Release), "release");
        assert_eq!(format!("{}", FenceKind::AcqRel), "acqrel");
        assert_eq!(format!("{}", FenceKind::SeqCst), "seqcst");
    }

    /// AtomicRmwOp display.
    #[test]
    fn atomic_rmw_op_display() {
        assert_eq!(format!("{}", AtomicRmwOp::Xchg), "xchg");
        assert_eq!(format!("{}", AtomicRmwOp::Add), "add");
        assert_eq!(format!("{}", AtomicRmwOp::Sub), "sub");
        assert_eq!(format!("{}", AtomicRmwOp::And), "and");
        assert_eq!(format!("{}", AtomicRmwOp::Or), "or");
        assert_eq!(format!("{}", AtomicRmwOp::Xor), "xor");
        assert_eq!(format!("{}", AtomicRmwOp::Min), "min");
        assert_eq!(format!("{}", AtomicRmwOp::Max), "max");
    }

    // -----------------------------------------------------------------------
    // FlagsMask tests
    // -----------------------------------------------------------------------

    #[test]
    fn flags_mask_constants() {
        assert_ne!(FlagsMask::CARRY, FlagsMask::ZERO);
        assert_ne!(FlagsMask::CARRY, FlagsMask::OVERFLOW);
        assert_eq!(FlagsMask::CARRY.bits(), 1 << 0);
        assert_eq!(FlagsMask::ZERO.bits(), 1 << 3);
        assert_eq!(FlagsMask::OVERFLOW.bits(), 1 << 5);
        assert!(FlagsMask::from_bits(0).is_empty());
        assert!(!FlagsMask::CARRY.is_empty());
    }

    #[test]
    fn flags_mask_display() {
        assert_eq!(format!("{}", FlagsMask::CARRY), "CF");
        assert_eq!(
            format!(
                "{}",
                FlagsMask::from_bits(FlagsMask::CARRY.bits() | FlagsMask::ZERO.bits())
            ),
            "CF,ZF"
        );
        assert_eq!(format!("{}", FlagsMask::from_bits(0)), "none");
    }

    // -----------------------------------------------------------------------
    // FlagCondition tests
    // -----------------------------------------------------------------------

    #[test]
    fn flag_condition_display() {
        assert_eq!(format!("{}", FlagCondition::Carry), "carry");
        assert_eq!(format!("{}", FlagCondition::NotCarry), "not_carry");
        assert_eq!(format!("{}", FlagCondition::Zero), "zero");
        assert_eq!(format!("{}", FlagCondition::NotZero), "not_zero");
        assert_eq!(format!("{}", FlagCondition::Overflow), "overflow");
        assert_eq!(format!("{}", FlagCondition::NotOverflow), "not_overflow");
        assert_eq!(format!("{}", FlagCondition::Negative), "negative");
        assert_eq!(format!("{}", FlagCondition::Positive), "positive");
        assert_eq!(format!("{}", FlagCondition::ParityEven), "parity_even");
        assert_eq!(format!("{}", FlagCondition::ParityOdd), "parity_odd");
    }

    #[test]
    fn flag_condition_variants_are_distinct() {
        assert_ne!(FlagCondition::Carry, FlagCondition::Zero);
        assert_ne!(FlagCondition::Overflow, FlagCondition::NotOverflow);
        assert_ne!(FlagCondition::Negative, FlagCondition::Positive);
        assert_ne!(FlagCondition::ParityEven, FlagCondition::ParityOdd);
    }

    // -----------------------------------------------------------------------
    // flags_dest tests
    // -----------------------------------------------------------------------

    #[test]
    fn flags_dest_returns_flags_on_flag_setting_ops() {
        let d = SsaVarId::from_index(0);
        let v = SsaVarId::from_index(1);
        let flags_var = SsaVarId::from_index(99);

        let add: SsaOp<MockTarget> = SsaOp::Add {
            dest: d,
            left: v,
            right: v,
            flags: Some(flags_var),
        };
        assert_eq!(add.flags_dest(), Some(flags_var));

        let sub: SsaOp<MockTarget> = SsaOp::Sub {
            dest: d,
            left: v,
            right: v,
            flags: Some(flags_var),
        };
        assert_eq!(sub.flags_dest(), Some(flags_var));

        let _and: SsaOp<MockTarget> = SsaOp::And {
            dest: d,
            left: v,
            right: v,
            flags: Some(flags_var),
        };
    }

    #[test]
    fn flags_dest_is_none_when_no_flags_set() {
        let d = SsaVarId::from_index(0);
        let v = SsaVarId::from_index(1);

        let add: SsaOp<MockTarget> = SsaOp::Add {
            dest: d,
            left: v,
            right: v,
            flags: None,
        };
        assert_eq!(add.flags_dest(), None);

        let mul: SsaOp<MockTarget> = SsaOp::Mul {
            dest: d,
            left: v,
            right: v,
            flags: None,
        };
        assert_eq!(mul.flags_dest(), None);
    }

    #[test]
    fn flags_dest_is_none_for_non_flag_ops() {
        let d = SsaVarId::from_index(0);
        let v = SsaVarId::from_index(1);

        let select: SsaOp<MockTarget> = SsaOp::Select {
            dest: d,
            condition: v,
            true_val: v,
            false_val: v,
        };
        assert_eq!(select.flags_dest(), None);

        let call: SsaOp<MockTarget> = SsaOp::Call {
            dest: Some(d),
            method: 0,
            args: vec![],
        };
        assert_eq!(call.flags_dest(), None);
    }

    // -----------------------------------------------------------------------
    // ReadFlags tests
    // -----------------------------------------------------------------------

    #[test]
    fn read_flags_dest_and_uses() {
        let d = SsaVarId::from_index(0);
        let flags_var = SsaVarId::from_index(1);

        let op: SsaOp<MockTarget> = SsaOp::ReadFlags {
            dest: d,
            flags: flags_var,
            mask: FlagsMask::ZERO,
        };
        assert_eq!(op.dest(), Some(d));
        assert_eq!(op.uses(), vec![flags_var]);
        assert!(op.is_pure());
        assert!(!op.is_terminator());
        assert!(!op.may_throw());
    }

    #[test]
    fn read_flags_stack_effect() {
        let d = SsaVarId::from_index(0);
        let f = SsaVarId::from_index(1);

        let op: SsaOp<MockTarget> = SsaOp::ReadFlags {
            dest: d,
            flags: f,
            mask: FlagsMask::CARRY,
        };
        assert_eq!(op.stack_effect(), (1, 1));
    }

    #[test]
    fn read_flags_replace_uses() {
        let d = SsaVarId::from_index(0);
        let old = SsaVarId::from_index(1);
        let new = SsaVarId::from_index(99);

        let mut op: SsaOp<MockTarget> = SsaOp::ReadFlags {
            dest: d,
            flags: old,
            mask: FlagsMask::SIGN,
        };
        assert_eq!(op.replace_uses(old, new), 1);
        assert_eq!(op.uses(), vec![new]);
    }

    #[test]
    fn read_flags_remap_variables() {
        let d0 = SsaVarId::from_index(0);
        let d99 = SsaVarId::from_index(99);
        let f1 = SsaVarId::from_index(1);
        let f55 = SsaVarId::from_index(55);

        let op: SsaOp<MockTarget> = SsaOp::ReadFlags {
            dest: d0,
            flags: f1,
            mask: FlagsMask::OVERFLOW,
        };
        let remapped = op.remap_variables(|v| {
            if v == d0 {
                Some(d99)
            } else if v == f1 {
                Some(f55)
            } else {
                None
            }
        });
        assert_eq!(remapped.dest(), Some(d99));
        assert_eq!(remapped.uses(), vec![f55]);
    }

    #[test]
    fn read_flags_display() {
        let d = SsaVarId::from_index(0);
        let f = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> = SsaOp::ReadFlags {
            dest: d,
            flags: f,
            mask: FlagsMask::ZERO,
        };
        assert_eq!(format!("{op}"), "v0 = readflags v1, ZF");
    }

    // -----------------------------------------------------------------------
    // BranchFlags tests
    // -----------------------------------------------------------------------

    #[test]
    fn branch_flags_is_terminator_with_successors() {
        let f = SsaVarId::from_index(0);

        let op: SsaOp<MockTarget> = SsaOp::BranchFlags {
            flags: f,
            condition: FlagCondition::Zero,
            true_target: 1,
            false_target: 2,
        };
        assert!(op.is_terminator());
        assert!(!op.is_pure());
        assert!(!op.may_throw());
        assert_eq!(op.dest(), None);
        assert_eq!(op.uses(), vec![f]);
        assert_eq!(op.successors(), vec![1, 2]);
    }

    #[test]
    fn branch_flags_stack_effect() {
        let f = SsaVarId::from_index(0);
        let op: SsaOp<MockTarget> = SsaOp::BranchFlags {
            flags: f,
            condition: FlagCondition::Carry,
            true_target: 1,
            false_target: 2,
        };
        assert_eq!(op.stack_effect(), (1, 0));
    }

    #[test]
    fn branch_flags_redirect_target() {
        let f = SsaVarId::from_index(0);
        let mut op: SsaOp<MockTarget> = SsaOp::BranchFlags {
            flags: f,
            condition: FlagCondition::NotZero,
            true_target: 3,
            false_target: 5,
        };
        assert!(op.redirect_target(3, 7));
        assert_eq!(op.successors(), vec![7, 5]);
        assert!(!op.redirect_target(99, 42)); // no-op
    }

    #[test]
    fn branch_flags_remap_targets() {
        let f = SsaVarId::from_index(0);
        let mut op: SsaOp<MockTarget> = SsaOp::BranchFlags {
            flags: f,
            condition: FlagCondition::Overflow,
            true_target: 2,
            false_target: 4,
        };
        op.remap_branch_targets(|t| {
            if t == 2 {
                Some(10)
            } else if t == 4 {
                Some(20)
            } else {
                None
            }
        });
        assert_eq!(op.successors(), vec![10, 20]);
    }

    #[test]
    fn branch_flags_replace_uses() {
        let old = SsaVarId::from_index(1);
        let new = SsaVarId::from_index(99);

        let mut op: SsaOp<MockTarget> = SsaOp::BranchFlags {
            flags: old,
            condition: FlagCondition::Positive,
            true_target: 1,
            false_target: 2,
        };
        assert_eq!(op.replace_uses(old, new), 1);
        assert_eq!(op.uses(), vec![new]);
    }

    #[test]
    fn branch_flags_display() {
        let f = SsaVarId::from_index(0);
        let op: SsaOp<MockTarget> = SsaOp::BranchFlags {
            flags: f,
            condition: FlagCondition::Carry,
            true_target: 3,
            false_target: 7,
        };
        assert_eq!(format!("{op}"), "branchflags carry B3, B7");
    }

    // -----------------------------------------------------------------------
    // Unreachable tests
    // -----------------------------------------------------------------------

    #[test]
    fn unreachable_is_terminator_with_no_successors() {
        assert!(SsaOp::<MockTarget>::Unreachable.is_terminator());
        assert!(!SsaOp::<MockTarget>::Unreachable.is_pure());
        assert!(!SsaOp::<MockTarget>::Unreachable.may_throw());
        assert_eq!(SsaOp::<MockTarget>::Unreachable.dest(), None);
        assert!(SsaOp::<MockTarget>::Unreachable.uses().is_empty());
        assert!(SsaOp::<MockTarget>::Unreachable.successors().is_empty());
    }

    #[test]
    fn unreachable_stack_effect() {
        assert_eq!(SsaOp::<MockTarget>::Unreachable.stack_effect(), (0, 0));
    }

    #[test]
    fn unreachable_display() {
        assert_eq!(
            format!("{}", SsaOp::<MockTarget>::Unreachable),
            "unreachable"
        );
    }

    #[test]
    fn unreachable_no_variable_remap() {
        let op = SsaOp::<MockTarget>::Unreachable;
        let remapped = op.remap_variables(|_| unreachable!());
        assert_eq!(remapped, SsaOp::Unreachable);
    }

    // -----------------------------------------------------------------------
    // Flag-setting op with flags integrated tests
    // -----------------------------------------------------------------------

    #[test]
    fn flag_setting_op_has_two_defs() {
        let d = SsaVarId::from_index(0);
        let f = SsaVarId::from_index(99);
        let v = SsaVarId::from_index(1);

        let op: SsaOp<MockTarget> = SsaOp::Add {
            dest: d,
            left: v,
            right: v,
            flags: Some(f),
        };
        assert_eq!(op.dest(), Some(d));
        assert_eq!(op.flags_dest(), Some(f));
        assert!(op.is_pure());
    }

    #[test]
    fn flag_setting_op_remap_remaps_flags() {
        let d0 = SsaVarId::from_index(0);
        let d99 = SsaVarId::from_index(99);
        let f1 = SsaVarId::from_index(1);
        let f55 = SsaVarId::from_index(55);

        let op: SsaOp<MockTarget> = SsaOp::Add {
            dest: d0,
            left: d0,
            right: d0,
            flags: Some(f1),
        };
        let remapped = op.remap_variables(|v| {
            if v == d0 {
                Some(d99)
            } else if v == f1 {
                Some(f55)
            } else {
                None
            }
        });
        assert_eq!(remapped.dest(), Some(d99));
        assert_eq!(remapped.flags_dest(), Some(f55));
    }

    #[test]
    fn flag_setting_op_display_shows_flags() {
        let d = SsaVarId::from_index(0);
        let f = SsaVarId::from_index(99);
        let v = SsaVarId::from_index(1);

        let with_flags: SsaOp<MockTarget> = SsaOp::Add {
            dest: d,
            left: v,
            right: v,
            flags: Some(f),
        };
        assert_eq!(format!("{with_flags}"), "v0 = add v1, v1 flags=v99");

        let without_flags: SsaOp<MockTarget> = SsaOp::Add {
            dest: d,
            left: v,
            right: v,
            flags: None,
        };
        assert_eq!(format!("{without_flags}"), "v0 = add v1, v1");
    }
}
