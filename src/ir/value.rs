//! Value types: constants, abstract value lattice, and CSE tracking.
//!
//! This module provides the value representation system for SSA variables, enabling
//! constant propagation, abstract interpretation, and common subexpression elimination.
//!
//! # Key Types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`ConstValue`] | Concrete compile-time constants (integers, floats, strings, null, etc.) |
//! | [`AbstractValue`] | Abstract lattice for dataflow analysis (constant, range, non-null, top, bottom) |
//! | [`ComputedValue`] | Represents a pure computation result for CSE |
//! | [`ComputedOp`] | Operation codes tracked for CSE (arithmetic, bitwise, comparison, conversion) |
//!
//! # Abstract Value Lattice
//!
//! The `AbstractValue` type forms a lattice for monotone dataflow analysis:
//!
//! ```text
//!               Top
//!              / | \
//!         Const Range NonNull
//!              \ | /
//!              Bottom
//! ```
//!
//! - **Top**: No information yet (initial state, join identity)
//! - **Constant**: Known compile-time constant value
//! - **Range**: Value in a bounded inclusive range `[min, max]`
//! - **NonNull**: Known non-null reference
//! - **SameAs**: Equal to another SSA variable (for copy propagation)
//! - **Computed**: Result of a pure computation (for CSE)
//! - **Bottom**: Conflicting information (meet identity, unreachable)
//!
//! The lattice operations are:
//! - **Meet** (greatest lower bound): Used at control flow joins to compute
//!   what is definitely true on all incoming paths
//! - **Join** (least upper bound): Used to find what is possibly true on
//!   any incoming path

use std::fmt;

use crate::{ir::variable::SsaVarId, target::Endianness, target::Target, PointerSize};

/// Compile-time constant values tracked through SSA analysis.
///
/// Represents all constant forms that can appear in the IR: numeric literals,
/// string references, null, booleans, and runtime handles. Supports constant
/// folding for arithmetic, bitwise, comparison, and conversion operations.
///
/// Generic over `T: Target` so metadata handles (type refs, method refs, field refs)
/// carry the host's reference types rather than generic placeholders.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue<T: Target> {
    /// Signed 8-bit integer constant (CIL: `int8`).
    I8(i8),

    /// Signed 16-bit integer constant (CIL: `int16`).
    I16(i16),

    /// Signed 32-bit integer constant (CIL: `int32`).
    I32(i32),

    /// Signed 64-bit integer constant (CIL: `int64`).
    I64(i64),

    /// Unsigned 8-bit integer constant (CIL: `uint8`).
    U8(u8),

    /// Unsigned 16-bit integer constant (CIL: `uint16`).
    U16(u16),

    /// Unsigned 32-bit integer constant (CIL: `uint32`).
    U32(u32),

    /// Unsigned 64-bit integer constant (CIL: `uint64`).
    U64(u64),

    /// Native-size signed integer constant.
    /// Width depends on the target pointer size (32-bit or 64-bit).
    /// Automatically masked to pointer width by `mask_native()`.
    NativeInt(i64),

    /// Native-size unsigned integer constant.
    /// Width depends on the target pointer size.
    NativeUInt(u64),

    /// 32-bit IEEE 754 floating point constant.
    F32(f32),

    /// 64-bit IEEE 754 floating point constant.
    F64(f64),

    /// String constant referenced by its metadata token (index into #US heap).
    /// The actual string content is resolved during code generation.
    String(u32),

    /// Decrypted string with inline content (not just a heap index).
    /// Used by deobfuscation passes that decrypt strings at analysis time.
    /// Contains the actual string bytes for codegen to emit.
    DecryptedString(String),

    /// Null reference constant.
    Null,

    /// Boolean true constant.
    True,

    /// Boolean false constant.
    False,

    /// Runtime type handle (result of `typeof()` or `ldtoken`).
    Type(T::TypeRef),

    /// Runtime method handle (result of `ldtoken` for a method).
    MethodHandle(T::MethodRef),

    /// Runtime field handle (result of `ldtoken` for a field).
    FieldHandle(T::FieldRef),

    /// Decrypted array data from deobfuscation.
    ///
    /// Stores the raw bytes and element type of an array that was decrypted
    /// at analysis time. Code generation emits `newarr` + element stores
    /// to reconstruct the array in the output.
    DecryptedArray {
        /// Raw bytes of the array data in little-endian element layout.
        data: Vec<u8>,

        /// Reference to the element type in the host's metadata
        /// (e.g., `TypeRef`/`TypeDef` token for CIL).
        element_type_ref: T::TypeRef,

        /// Size of each array element in bytes (1 for byte, 4 for int32, etc.).
        element_size: usize,
    },
}

impl<T: Target> ConstValue<T> {
    /// Returns `true` if this is the null constant.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Returns `true` if this is a boolean constant.
    #[must_use]
    pub const fn is_bool(&self) -> bool {
        matches!(self, Self::True | Self::False)
    }

    /// Returns `true` if this is an integer constant (signed or unsigned).
    #[must_use]
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::I8(_)
                | Self::I16(_)
                | Self::I32(_)
                | Self::I64(_)
                | Self::U8(_)
                | Self::U16(_)
                | Self::U32(_)
                | Self::U64(_)
                | Self::NativeInt(_)
                | Self::NativeUInt(_)
        )
    }

    /// Returns `true` if this is a signed integer constant.
    #[must_use]
    pub const fn is_signed(&self) -> bool {
        matches!(
            self,
            Self::I8(_) | Self::I16(_) | Self::I32(_) | Self::I64(_) | Self::NativeInt(_)
        )
    }

    /// Returns `true` if this is an unsigned integer constant.
    #[must_use]
    pub const fn is_unsigned(&self) -> bool {
        matches!(
            self,
            Self::U8(_) | Self::U16(_) | Self::U32(_) | Self::U64(_) | Self::NativeUInt(_)
        )
    }

    /// Returns `true` if this is a floating-point constant.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(self, Self::F32(_) | Self::F64(_))
    }

    /// Returns `true` if this value is a string (`String` or `DecryptedString`).
    #[must_use]
    pub const fn is_string_like(&self) -> bool {
        matches!(self, Self::String(_) | Self::DecryptedString(_))
    }

    /// Returns the constant as an i32 if applicable.
    #[must_use]
    pub const fn as_i32(&self) -> Option<i32> {
        match self {
            Self::I8(v) => Some(*v as i32),
            Self::I16(v) => Some(*v as i32),
            Self::I32(v) => Some(*v),
            Self::U8(v) => Some(*v as i32),
            Self::U16(v) => Some(*v as i32),
            Self::True => Some(1),
            Self::False => Some(0),
            _ => None,
        }
    }

    /// Returns the constant as an i64 if applicable.
    #[must_use]
    #[allow(clippy::match_same_arms)] // NativeInt is semantically different from I64
    pub const fn as_i64(&self) -> Option<i64> {
        match self {
            Self::I8(v) => Some(*v as i64),
            Self::I16(v) => Some(*v as i64),
            Self::I32(v) => Some(*v as i64),
            Self::I64(v) => Some(*v),
            Self::U8(v) => Some(*v as i64),
            Self::U16(v) => Some(*v as i64),
            Self::U32(v) => Some(*v as i64),
            Self::NativeInt(v) => Some(*v),
            Self::True => Some(1),
            Self::False => Some(0),
            _ => None,
        }
    }

    /// Returns the constant as a u64 if applicable (for unsigned operations).
    #[must_use]
    #[allow(clippy::cast_sign_loss)] // Guarded by >= 0 checks
    #[allow(clippy::match_same_arms)] // NativeUInt is semantically different from U64
    pub const fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            Self::NativeUInt(v) => Some(*v),
            Self::I8(v) if *v >= 0 => Some(*v as u64),
            Self::I16(v) if *v >= 0 => Some(*v as u64),
            Self::I32(v) if *v >= 0 => Some(*v as u64),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            Self::True => Some(1),
            Self::False => Some(0),
            _ => None,
        }
    }

    /// Returns the constant as a u32 if applicable (for unsigned operations).
    #[must_use]
    #[allow(clippy::cast_sign_loss)] // Guarded by >= 0 checks
    pub const fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U8(v) => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            Self::U32(v) => Some(*v),
            Self::I8(v) if *v >= 0 => Some(*v as u32),
            Self::I16(v) if *v >= 0 => Some(*v as u32),
            Self::I32(v) if *v >= 0 => Some(*v as u32),
            Self::True => Some(1),
            Self::False => Some(0),
            _ => None,
        }
    }

    /// Returns the constant as an f32 if it's stored as F32.
    #[must_use]
    pub const fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the constant as an f64 if it's stored as F64.
    #[must_use]
    pub const fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the constant as a bool if applicable.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::False
            | Self::Null
            | Self::I8(0)
            | Self::I16(0)
            | Self::I32(0)
            | Self::I64(0)
            | Self::U8(0)
            | Self::U16(0)
            | Self::U32(0)
            | Self::U64(0) => Some(false),
            Self::True
            | Self::I8(_)
            | Self::I16(_)
            | Self::I32(_)
            | Self::I64(_)
            | Self::U8(_)
            | Self::U16(_)
            | Self::U32(_)
            | Self::U64(_) => Some(true),
            _ => None,
        }
    }

    /// Creates a boolean constant from a bool value.
    #[must_use]
    pub const fn from_bool(value: bool) -> Self {
        if value {
            Self::True
        } else {
            Self::False
        }
    }

    /// Returns the string content if this is a `DecryptedString`.
    ///
    /// `String` variants are heap references and do not carry inline content.
    #[must_use]
    pub fn as_decrypted_string(&self) -> Option<&str> {
        match self {
            Self::DecryptedString(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns `true` if this constant represents zero.
    ///
    /// This includes all numeric zero values and `False`.
    /// Useful for opaque predicate detection where `x ^ x`, `x - x`, `x * 0`, etc. produce zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        matches!(
            self,
            Self::I8(0)
                | Self::I16(0)
                | Self::I32(0)
                | Self::I64(0)
                | Self::U8(0)
                | Self::U16(0)
                | Self::U32(0)
                | Self::U64(0)
                | Self::NativeInt(0)
                | Self::NativeUInt(0)
                | Self::False
        )
    }

    /// Returns `true` if this constant represents one.
    ///
    /// This includes all numeric one values and `True`.
    /// Useful for identity operations and opaque predicate detection.
    #[must_use]
    pub const fn is_one(&self) -> bool {
        matches!(
            self,
            Self::I8(1)
                | Self::I16(1)
                | Self::I32(1)
                | Self::I64(1)
                | Self::U8(1)
                | Self::U16(1)
                | Self::U32(1)
                | Self::U64(1)
                | Self::NativeInt(1)
                | Self::NativeUInt(1)
                | Self::True
        )
    }

    /// Returns `true` if this constant represents negative one (-1).
    ///
    /// This is useful for detecting `x | -1 = -1` patterns in opaque predicates.
    #[must_use]
    pub const fn is_minus_one(&self) -> bool {
        matches!(
            self,
            Self::I8(-1) | Self::I16(-1) | Self::I32(-1) | Self::I64(-1) | Self::NativeInt(-1)
        )
    }

    /// Returns `true` if this constant has all bits set (e.g., -1 for signed, MAX for unsigned).
    ///
    /// This is useful for detecting `x & -1 = x` and `x | -1 = -1` patterns.
    #[must_use]
    pub const fn is_all_ones(&self) -> bool {
        matches!(
            self,
            Self::I8(-1)
                | Self::I16(-1)
                | Self::I32(-1)
                | Self::I64(-1)
                | Self::NativeInt(-1)
                | Self::U8(u8::MAX)
                | Self::U16(u16::MAX)
                | Self::U32(u32::MAX)
                | Self::U64(u64::MAX)
                | Self::NativeUInt(u64::MAX)
        )
    }

    /// Returns a zero constant of the same type as this constant.
    ///
    /// Useful for algebraic simplifications like `x * 0 = 0` where the result
    /// should preserve the type of the operands.
    #[must_use]
    pub const fn zero_of_same_type(&self) -> Self {
        match self {
            Self::I8(_) => Self::I8(0),
            Self::I16(_) => Self::I16(0),
            Self::I64(_) => Self::I64(0),
            Self::U8(_) => Self::U8(0),
            Self::U16(_) => Self::U16(0),
            Self::U32(_) => Self::U32(0),
            Self::U64(_) => Self::U64(0),
            Self::NativeInt(_) => Self::NativeInt(0),
            Self::NativeUInt(_) => Self::NativeUInt(0),
            Self::F32(_) => Self::F32(0.0),
            Self::F64(_) => Self::F64(0.0),
            // For non-numeric types (including I32), default to i32
            _ => Self::I32(0),
        }
    }

    /// Attempts to negate this constant.
    #[must_use]
    pub fn negate(&self, ptr_size: PointerSize) -> Option<Self> {
        match self {
            Self::I8(v) => Some(Self::I8(v.wrapping_neg())),
            Self::I16(v) => Some(Self::I16(v.wrapping_neg())),
            Self::I32(v) => Some(Self::I32(v.wrapping_neg())),
            Self::I64(v) => Some(Self::I64(v.wrapping_neg())),
            Self::NativeInt(v) => Some(Self::NativeInt(v.wrapping_neg())),
            Self::F32(v) => Some(Self::F32(-v)),
            Self::F64(v) => Some(Self::F64(-v)),
            // Unsigned negation wraps
            Self::U8(v) => Some(Self::U8(v.wrapping_neg())),
            Self::U16(v) => Some(Self::U16(v.wrapping_neg())),
            Self::U32(v) => Some(Self::U32(v.wrapping_neg())),
            Self::U64(v) => Some(Self::U64(v.wrapping_neg())),
            Self::NativeUInt(v) => Some(Self::NativeUInt(v.wrapping_neg())),
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to perform bitwise NOT on this constant.
    #[must_use]
    pub fn bitwise_not(&self, ptr_size: PointerSize) -> Option<Self> {
        match self {
            Self::I8(v) => Some(Self::I8(!v)),
            Self::I16(v) => Some(Self::I16(!v)),
            Self::I32(v) => Some(Self::I32(!v)),
            Self::I64(v) => Some(Self::I64(!v)),
            Self::U8(v) => Some(Self::U8(!v)),
            Self::U16(v) => Some(Self::U16(!v)),
            Self::U32(v) => Some(Self::U32(!v)),
            Self::U64(v) => Some(Self::U64(!v)),
            Self::NativeInt(v) => Some(Self::NativeInt(!v)),
            Self::NativeUInt(v) => Some(Self::NativeUInt(!v)),
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to perform bitwise AND on two constants.
    #[must_use]
    pub fn bitwise_and(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a & b)),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a & b)),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a & b)),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a & b)),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a & b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a & b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a & b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a & b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a & b)),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::NativeUInt(a & b)),
            // Cross-type: promote to i64 for mixed signed operations
            (Self::I32(a), Self::I64(b)) | (Self::I64(b), Self::I32(a)) => {
                Some(Self::I64(i64::from(*a) & b))
            }
            (Self::U32(a), Self::U64(b)) | (Self::U64(b), Self::U32(a)) => {
                Some(Self::U64(u64::from(*a) & b))
            }
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to perform bitwise OR on two constants.
    #[must_use]
    pub fn bitwise_or(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a | b)),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a | b)),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a | b)),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a | b)),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a | b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a | b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a | b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a | b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a | b)),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::NativeUInt(a | b)),
            (Self::I32(a), Self::I64(b)) | (Self::I64(b), Self::I32(a)) => {
                Some(Self::I64(i64::from(*a) | b))
            }
            (Self::U32(a), Self::U64(b)) | (Self::U64(b), Self::U32(a)) => {
                Some(Self::U64(u64::from(*a) | b))
            }
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to perform bitwise XOR on two constants.
    #[must_use]
    pub fn bitwise_xor(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a ^ b)),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a ^ b)),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a ^ b)),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a ^ b)),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a ^ b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a ^ b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a ^ b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a ^ b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a ^ b)),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::NativeUInt(a ^ b)),
            (Self::I32(a), Self::I64(b)) | (Self::I64(b), Self::I32(a)) => {
                Some(Self::I64(i64::from(*a) ^ b))
            }
            (Self::U32(a), Self::U64(b)) | (Self::U64(b), Self::U32(a)) => {
                Some(Self::U64(u64::from(*a) ^ b))
            }
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to shift left.
    #[must_use]
    #[allow(clippy::cast_sign_loss)] // Shift amounts are non-negative by convention
    pub fn shl(&self, amount: &Self, ptr_size: PointerSize) -> Option<Self> {
        let shift = amount.as_i32()? as u32;
        match self {
            Self::I8(v) => Some(Self::I8(v.wrapping_shl(shift))),
            Self::I16(v) => Some(Self::I16(v.wrapping_shl(shift))),
            Self::I32(v) => Some(Self::I32(v.wrapping_shl(shift))),
            Self::I64(v) => Some(Self::I64(v.wrapping_shl(shift))),
            Self::U8(v) => Some(Self::U8(v.wrapping_shl(shift))),
            Self::U16(v) => Some(Self::U16(v.wrapping_shl(shift))),
            Self::U32(v) => Some(Self::U32(v.wrapping_shl(shift))),
            Self::U64(v) => Some(Self::U64(v.wrapping_shl(shift))),
            Self::NativeInt(v) => Some(Self::NativeInt(v.wrapping_shl(shift))),
            Self::NativeUInt(v) => Some(Self::NativeUInt(v.wrapping_shl(shift))),
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to shift right (arithmetic for signed, logical for unsigned).
    #[must_use]
    #[allow(clippy::cast_sign_loss)] // Shift amounts and unsigned shifts use intentional casts
    #[allow(clippy::cast_possible_wrap)] // Wrapping is expected for logical shift operations
    pub fn shr(&self, amount: &Self, unsigned: bool, ptr_size: PointerSize) -> Option<Self> {
        let shift = amount.as_i32()? as u32;
        match self {
            Self::I8(v) => {
                if unsigned {
                    Some(Self::I8((*v as u8).wrapping_shr(shift) as i8))
                } else {
                    Some(Self::I8(v.wrapping_shr(shift)))
                }
            }
            Self::I16(v) => {
                if unsigned {
                    Some(Self::I16((*v as u16).wrapping_shr(shift) as i16))
                } else {
                    Some(Self::I16(v.wrapping_shr(shift)))
                }
            }
            Self::I32(v) => {
                if unsigned {
                    Some(Self::I32((*v as u32).wrapping_shr(shift) as i32))
                } else {
                    Some(Self::I32(v.wrapping_shr(shift)))
                }
            }
            Self::I64(v) => {
                if unsigned {
                    Some(Self::I64((*v as u64).wrapping_shr(shift) as i64))
                } else {
                    Some(Self::I64(v.wrapping_shr(shift)))
                }
            }
            Self::U8(v) => Some(Self::U8(v.wrapping_shr(shift))),
            Self::U16(v) => Some(Self::U16(v.wrapping_shr(shift))),
            Self::U32(v) => Some(Self::U32(v.wrapping_shr(shift))),
            Self::U64(v) => Some(Self::U64(v.wrapping_shr(shift))),
            Self::NativeInt(v) => {
                if unsigned {
                    Some(Self::NativeInt((*v as u64).wrapping_shr(shift) as i64))
                } else {
                    Some(Self::NativeInt(v.wrapping_shr(shift)))
                }
            }
            Self::NativeUInt(v) => Some(Self::NativeUInt(v.wrapping_shr(shift))),
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to add two constants.
    #[must_use]
    pub fn add(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a.wrapping_add(*b))),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a.wrapping_add(*b))),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a.wrapping_add(*b))),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a.wrapping_add(*b))),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a.wrapping_add(*b))),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a.wrapping_add(*b))),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a.wrapping_add(*b))),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a.wrapping_add(*b))),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a.wrapping_add(*b))),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => {
                Some(Self::NativeUInt(a.wrapping_add(*b)))
            }
            (Self::F32(a), Self::F32(b)) => Some(Self::F32(a + b)),
            (Self::F64(a), Self::F64(b)) => Some(Self::F64(a + b)),
            // Cross-type promotions
            (Self::I32(a), Self::I64(b)) | (Self::I64(b), Self::I32(a)) => {
                Some(Self::I64(i64::from(*a).wrapping_add(*b)))
            }
            (Self::U32(a), Self::U64(b)) | (Self::U64(b), Self::U32(a)) => {
                Some(Self::U64(u64::from(*a).wrapping_add(*b)))
            }
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to subtract two constants.
    #[must_use]
    pub fn sub(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a.wrapping_sub(*b))),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a.wrapping_sub(*b))),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a.wrapping_sub(*b))),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a.wrapping_sub(*b))),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a.wrapping_sub(*b))),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a.wrapping_sub(*b))),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a.wrapping_sub(*b))),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a.wrapping_sub(*b))),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a.wrapping_sub(*b))),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => {
                Some(Self::NativeUInt(a.wrapping_sub(*b)))
            }
            (Self::F32(a), Self::F32(b)) => Some(Self::F32(a - b)),
            (Self::F64(a), Self::F64(b)) => Some(Self::F64(a - b)),
            (Self::I32(a), Self::I64(b)) => Some(Self::I64(i64::from(*a).wrapping_sub(*b))),
            (Self::I64(a), Self::I32(b)) => Some(Self::I64(a.wrapping_sub(i64::from(*b)))),
            (Self::U32(a), Self::U64(b)) => Some(Self::U64(u64::from(*a).wrapping_sub(*b))),
            (Self::U64(a), Self::U32(b)) => Some(Self::U64(a.wrapping_sub(u64::from(*b)))),
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to multiply two constants.
    #[must_use]
    pub fn mul(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a.wrapping_mul(*b))),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a.wrapping_mul(*b))),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a.wrapping_mul(*b))),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a.wrapping_mul(*b))),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a.wrapping_mul(*b))),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a.wrapping_mul(*b))),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a.wrapping_mul(*b))),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a.wrapping_mul(*b))),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a.wrapping_mul(*b))),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => {
                Some(Self::NativeUInt(a.wrapping_mul(*b)))
            }
            (Self::F32(a), Self::F32(b)) => Some(Self::F32(a * b)),
            (Self::F64(a), Self::F64(b)) => Some(Self::F64(a * b)),
            (Self::I32(a), Self::I64(b)) | (Self::I64(b), Self::I32(a)) => {
                Some(Self::I64(i64::from(*a).wrapping_mul(*b)))
            }
            (Self::U32(a), Self::U64(b)) | (Self::U64(b), Self::U32(a)) => {
                Some(Self::U64(u64::from(*a).wrapping_mul(*b)))
            }
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to add two constants with overflow checking.
    ///
    /// Returns `None` if the addition would overflow.
    /// When `unsigned` is true, operands are treated as unsigned for overflow detection.
    #[must_use]
    pub fn add_checked(&self, other: &Self, unsigned: bool, ptr_size: PointerSize) -> Option<Self> {
        if unsigned {
            // Unsigned overflow check
            match (self, other) {
                (Self::I32(a), Self::I32(b)) => (*a)
                    .cast_unsigned()
                    .checked_add((*b).cast_unsigned())
                    .map(|r| Self::I32(r.cast_signed())),
                (Self::I64(a), Self::I64(b)) => (*a)
                    .cast_unsigned()
                    .checked_add((*b).cast_unsigned())
                    .map(|r| Self::I64(r.cast_signed())),
                (Self::U8(a), Self::U8(b)) => a.checked_add(*b).map(Self::U8),
                (Self::U16(a), Self::U16(b)) => a.checked_add(*b).map(Self::U16),
                (Self::U32(a), Self::U32(b)) => a.checked_add(*b).map(Self::U32),
                (Self::U64(a), Self::U64(b)) => a.checked_add(*b).map(Self::U64),
                (Self::NativeUInt(a), Self::NativeUInt(b)) => {
                    a.checked_add(*b).map(Self::NativeUInt)
                }
                _ => None,
            }
        } else {
            // Signed overflow check
            match (self, other) {
                (Self::I8(a), Self::I8(b)) => a.checked_add(*b).map(Self::I8),
                (Self::I16(a), Self::I16(b)) => a.checked_add(*b).map(Self::I16),
                (Self::I32(a), Self::I32(b)) => a.checked_add(*b).map(Self::I32),
                (Self::I64(a), Self::I64(b)) => a.checked_add(*b).map(Self::I64),
                (Self::NativeInt(a), Self::NativeInt(b)) => a.checked_add(*b).map(Self::NativeInt),
                _ => None,
            }
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to subtract two constants with overflow checking.
    ///
    /// Returns `None` if the subtraction would overflow.
    /// When `unsigned` is true, operands are treated as unsigned for overflow detection.
    #[must_use]
    pub fn sub_checked(&self, other: &Self, unsigned: bool, ptr_size: PointerSize) -> Option<Self> {
        if unsigned {
            // Unsigned overflow check
            match (self, other) {
                (Self::I32(a), Self::I32(b)) => (*a)
                    .cast_unsigned()
                    .checked_sub((*b).cast_unsigned())
                    .map(|r| Self::I32(r.cast_signed())),
                (Self::I64(a), Self::I64(b)) => (*a)
                    .cast_unsigned()
                    .checked_sub((*b).cast_unsigned())
                    .map(|r| Self::I64(r.cast_signed())),
                (Self::U8(a), Self::U8(b)) => a.checked_sub(*b).map(Self::U8),
                (Self::U16(a), Self::U16(b)) => a.checked_sub(*b).map(Self::U16),
                (Self::U32(a), Self::U32(b)) => a.checked_sub(*b).map(Self::U32),
                (Self::U64(a), Self::U64(b)) => a.checked_sub(*b).map(Self::U64),
                (Self::NativeUInt(a), Self::NativeUInt(b)) => {
                    a.checked_sub(*b).map(Self::NativeUInt)
                }
                _ => None,
            }
        } else {
            // Signed overflow check
            match (self, other) {
                (Self::I8(a), Self::I8(b)) => a.checked_sub(*b).map(Self::I8),
                (Self::I16(a), Self::I16(b)) => a.checked_sub(*b).map(Self::I16),
                (Self::I32(a), Self::I32(b)) => a.checked_sub(*b).map(Self::I32),
                (Self::I64(a), Self::I64(b)) => a.checked_sub(*b).map(Self::I64),
                (Self::NativeInt(a), Self::NativeInt(b)) => a.checked_sub(*b).map(Self::NativeInt),
                _ => None,
            }
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to multiply two constants with overflow checking.
    ///
    /// Returns `None` if the multiplication would overflow.
    /// When `unsigned` is true, operands are treated as unsigned for overflow detection.
    #[must_use]
    pub fn mul_checked(&self, other: &Self, unsigned: bool, ptr_size: PointerSize) -> Option<Self> {
        if unsigned {
            // Unsigned overflow check
            match (self, other) {
                (Self::I32(a), Self::I32(b)) => (*a)
                    .cast_unsigned()
                    .checked_mul((*b).cast_unsigned())
                    .map(|r| Self::I32(r.cast_signed())),
                (Self::I64(a), Self::I64(b)) => (*a)
                    .cast_unsigned()
                    .checked_mul((*b).cast_unsigned())
                    .map(|r| Self::I64(r.cast_signed())),
                (Self::U8(a), Self::U8(b)) => a.checked_mul(*b).map(Self::U8),
                (Self::U16(a), Self::U16(b)) => a.checked_mul(*b).map(Self::U16),
                (Self::U32(a), Self::U32(b)) => a.checked_mul(*b).map(Self::U32),
                (Self::U64(a), Self::U64(b)) => a.checked_mul(*b).map(Self::U64),
                (Self::NativeUInt(a), Self::NativeUInt(b)) => {
                    a.checked_mul(*b).map(Self::NativeUInt)
                }
                _ => None,
            }
        } else {
            // Signed overflow check
            match (self, other) {
                (Self::I8(a), Self::I8(b)) => a.checked_mul(*b).map(Self::I8),
                (Self::I16(a), Self::I16(b)) => a.checked_mul(*b).map(Self::I16),
                (Self::I32(a), Self::I32(b)) => a.checked_mul(*b).map(Self::I32),
                (Self::I64(a), Self::I64(b)) => a.checked_mul(*b).map(Self::I64),
                (Self::NativeInt(a), Self::NativeInt(b)) => a.checked_mul(*b).map(Self::NativeInt),
                _ => None,
            }
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to divide two constants. Uses `checked_div`/`checked_rem` so
    /// MIN/-1 overflows fold to `None` rather than wrapping silently.
    #[must_use]
    pub fn div(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => a.checked_div(*b).map(Self::I8),
            (Self::I16(a), Self::I16(b)) => a.checked_div(*b).map(Self::I16),
            (Self::I32(a), Self::I32(b)) => a.checked_div(*b).map(Self::I32),
            (Self::I64(a), Self::I64(b)) => a.checked_div(*b).map(Self::I64),
            (Self::U8(a), Self::U8(b)) => a.checked_div(*b).map(Self::U8),
            (Self::U16(a), Self::U16(b)) => a.checked_div(*b).map(Self::U16),
            (Self::U32(a), Self::U32(b)) => a.checked_div(*b).map(Self::U32),
            (Self::U64(a), Self::U64(b)) => a.checked_div(*b).map(Self::U64),
            (Self::NativeInt(a), Self::NativeInt(b)) => a.checked_div(*b).map(Self::NativeInt),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => a.checked_div(*b).map(Self::NativeUInt),
            // Float div by zero is inf — IEEE 754 has no panic, no overflow.
            (Self::F32(a), Self::F32(b)) => Some(Self::F32(a / b)),
            (Self::F64(a), Self::F64(b)) => Some(Self::F64(a / b)),
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to compute remainder (modulo) of two constants. Uses
    /// `checked_rem` so MIN%-1 overflows fold to `None`.
    #[must_use]
    pub fn rem(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => a.checked_rem(*b).map(Self::I8),
            (Self::I16(a), Self::I16(b)) => a.checked_rem(*b).map(Self::I16),
            (Self::I32(a), Self::I32(b)) => a.checked_rem(*b).map(Self::I32),
            (Self::I64(a), Self::I64(b)) => a.checked_rem(*b).map(Self::I64),
            (Self::U8(a), Self::U8(b)) => a.checked_rem(*b).map(Self::U8),
            (Self::U16(a), Self::U16(b)) => a.checked_rem(*b).map(Self::U16),
            (Self::U32(a), Self::U32(b)) => a.checked_rem(*b).map(Self::U32),
            (Self::U64(a), Self::U64(b)) => a.checked_rem(*b).map(Self::U64),
            (Self::NativeInt(a), Self::NativeInt(b)) => a.checked_rem(*b).map(Self::NativeInt),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => a.checked_rem(*b).map(Self::NativeUInt),
            (Self::F32(a), Self::F32(b)) => Some(Self::F32(a % b)),
            (Self::F64(a), Self::F64(b)) => Some(Self::F64(a % b)),
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to compare two constants for equality.
    #[must_use]
    #[allow(clippy::float_cmp)] // Exact comparison is correct for constant propagation
    #[allow(clippy::match_same_arms)] // NativeInt/NativeUInt are semantically different
    pub fn ceq(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::from_bool(a == b)),
            (Self::I16(a), Self::I16(b)) => Some(Self::from_bool(a == b)),
            (Self::I32(a), Self::I32(b)) => Some(Self::from_bool(a == b)),
            (Self::I64(a), Self::I64(b)) => Some(Self::from_bool(a == b)),
            (Self::U8(a), Self::U8(b)) => Some(Self::from_bool(a == b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::from_bool(a == b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::from_bool(a == b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::from_bool(a == b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::from_bool(a == b)),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::from_bool(a == b)),
            (Self::F32(a), Self::F32(b)) => Some(Self::from_bool(a == b)),
            (Self::F64(a), Self::F64(b)) => Some(Self::from_bool(a == b)),
            (Self::Null, Self::Null) | (Self::True, Self::True) | (Self::False, Self::False) => {
                Some(Self::True)
            }
            (Self::True, Self::False) | (Self::False, Self::True) => Some(Self::False),
            // Cross-type comparisons with promotion
            (Self::I32(a), Self::I64(b)) | (Self::I64(b), Self::I32(a)) => {
                Some(Self::from_bool(i64::from(*a) == *b))
            }
            (Self::U32(a), Self::U64(b)) | (Self::U64(b), Self::U32(a)) => {
                Some(Self::from_bool(u64::from(*a) == *b))
            }
            _ => None,
        }
    }

    /// Attempts to compare two constants for less-than (signed).
    #[must_use]
    #[allow(clippy::match_same_arms)] // NativeInt/NativeUInt are semantically different
    pub fn clt(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::from_bool(a < b)),
            (Self::I16(a), Self::I16(b)) => Some(Self::from_bool(a < b)),
            (Self::I32(a), Self::I32(b)) => Some(Self::from_bool(a < b)),
            (Self::I64(a), Self::I64(b)) => Some(Self::from_bool(a < b)),
            (Self::U8(a), Self::U8(b)) => Some(Self::from_bool(a < b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::from_bool(a < b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::from_bool(a < b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::from_bool(a < b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::from_bool(a < b)),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::from_bool(a < b)),
            (Self::F32(a), Self::F32(b)) => Some(Self::from_bool(a < b)),
            (Self::F64(a), Self::F64(b)) => Some(Self::from_bool(a < b)),
            (Self::I32(a), Self::I64(b)) => Some(Self::from_bool(i64::from(*a) < *b)),
            (Self::I64(a), Self::I32(b)) => Some(Self::from_bool(*a < i64::from(*b))),
            (Self::U32(a), Self::U64(b)) => Some(Self::from_bool(u64::from(*a) < *b)),
            (Self::U64(a), Self::U32(b)) => Some(Self::from_bool(*a < u64::from(*b))),
            _ => None,
        }
    }

    /// Attempts to compare two constants for less-than (unsigned).
    #[must_use]
    #[allow(clippy::cast_sign_loss)] // Unsigned comparison requires interpreting bits as unsigned
    #[allow(clippy::match_same_arms)] // NativeInt/NativeUInt are semantically different
    pub fn clt_un(&self, other: &Self) -> Option<Self> {
        // Treat values as unsigned for comparison
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::from_bool((*a as u8) < (*b as u8))),
            (Self::I16(a), Self::I16(b)) => Some(Self::from_bool((*a as u16) < (*b as u16))),
            (Self::I32(a), Self::I32(b)) => Some(Self::from_bool((*a as u32) < (*b as u32))),
            (Self::I64(a), Self::I64(b)) => Some(Self::from_bool((*a as u64) < (*b as u64))),
            (Self::U8(a), Self::U8(b)) => Some(Self::from_bool(a < b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::from_bool(a < b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::from_bool(a < b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::from_bool(a < b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => {
                Some(Self::from_bool((*a as u64) < (*b as u64)))
            }
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::from_bool(a < b)),
            // For floats, clt.un checks for unordered (NaN) or less than
            (Self::F32(a), Self::F32(b)) => {
                Some(Self::from_bool(a.is_nan() || b.is_nan() || a < b))
            }
            (Self::F64(a), Self::F64(b)) => {
                Some(Self::from_bool(a.is_nan() || b.is_nan() || a < b))
            }
            _ => None,
        }
    }

    /// Attempts to compare two constants for greater-than (signed).
    #[must_use]
    #[allow(clippy::match_same_arms)] // NativeInt/NativeUInt are semantically different
    pub fn cgt(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::from_bool(a > b)),
            (Self::I16(a), Self::I16(b)) => Some(Self::from_bool(a > b)),
            (Self::I32(a), Self::I32(b)) => Some(Self::from_bool(a > b)),
            (Self::I64(a), Self::I64(b)) => Some(Self::from_bool(a > b)),
            (Self::U8(a), Self::U8(b)) => Some(Self::from_bool(a > b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::from_bool(a > b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::from_bool(a > b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::from_bool(a > b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::from_bool(a > b)),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::from_bool(a > b)),
            (Self::F32(a), Self::F32(b)) => Some(Self::from_bool(a > b)),
            (Self::F64(a), Self::F64(b)) => Some(Self::from_bool(a > b)),
            (Self::I32(a), Self::I64(b)) => Some(Self::from_bool(i64::from(*a) > *b)),
            (Self::I64(a), Self::I32(b)) => Some(Self::from_bool(*a > i64::from(*b))),
            (Self::U32(a), Self::U64(b)) => Some(Self::from_bool(u64::from(*a) > *b)),
            (Self::U64(a), Self::U32(b)) => Some(Self::from_bool(*a > u64::from(*b))),
            _ => None,
        }
    }

    /// Attempts to compare two constants for greater-than (unsigned).
    #[must_use]
    #[allow(clippy::cast_sign_loss)] // Unsigned comparison requires interpreting bits as unsigned
    #[allow(clippy::match_same_arms)] // NativeInt/NativeUInt are semantically different
    pub fn cgt_un(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::from_bool((*a as u8) > (*b as u8))),
            (Self::I16(a), Self::I16(b)) => Some(Self::from_bool((*a as u16) > (*b as u16))),
            (Self::I32(a), Self::I32(b)) => Some(Self::from_bool((*a as u32) > (*b as u32))),
            (Self::I64(a), Self::I64(b)) => Some(Self::from_bool((*a as u64) > (*b as u64))),
            (Self::U8(a), Self::U8(b)) => Some(Self::from_bool(a > b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::from_bool(a > b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::from_bool(a > b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::from_bool(a > b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => {
                Some(Self::from_bool((*a as u64) > (*b as u64)))
            }
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::from_bool(a > b)),
            // For floats, cgt.un checks for unordered (NaN) or greater than
            (Self::F32(a), Self::F32(b)) => {
                Some(Self::from_bool(a.is_nan() || b.is_nan() || a > b))
            }
            (Self::F64(a), Self::F64(b)) => {
                Some(Self::from_bool(a.is_nan() || b.is_nan() || a > b))
            }
            _ => None,
        }
    }

    /// Masks a `ConstValue` to the target pointer width.
    ///
    /// For `NativeInt`, sign-extends from 32-bit on `Bit32`.
    /// For `NativeUInt`, zero-extends from 32-bit on `Bit32`.
    /// All other variants are returned unchanged.
    #[must_use]
    pub fn mask_native(self, ptr_size: PointerSize) -> Self {
        match self {
            Self::NativeInt(v) => Self::NativeInt(ptr_size.mask_signed(v)),
            Self::NativeUInt(v) => Self::NativeUInt(ptr_size.mask_unsigned(v)),
            other => other,
        }
    }

    /// Serializes an integer constant to bytes in the given endianness.
    ///
    /// Returns `None` for non-integer variants (floats, strings, handles,
    /// etc.).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::{ir::value::ConstValue, target::Endianness, MockTarget};
    ///
    /// let val = ConstValue::<MockTarget>::U32(0x01020304);
    /// let le = val.to_bytes(Endianness::Little).unwrap();
    /// let be = val.to_bytes(Endianness::Big).unwrap();
    /// assert_eq!(le, vec![0x04, 0x03, 0x02, 0x01]);
    /// assert_eq!(be, vec![0x01, 0x02, 0x03, 0x04]);
    /// ```
    #[must_use]
    pub fn to_bytes(&self, endianness: Endianness) -> Option<Vec<u8>> {
        match *self {
            Self::I8(v) => Some(vec![v as u8]),
            Self::U8(v) => Some(vec![v]),
            Self::I16(v) => Some(endianness.bytes_of_u16(v as u16).to_vec()),
            Self::U16(v) => Some(endianness.bytes_of_u16(v).to_vec()),
            Self::I32(v) => Some(endianness.bytes_of_u32(v as u32).to_vec()),
            Self::U32(v) => Some(endianness.bytes_of_u32(v).to_vec()),
            Self::I64(v) => Some(endianness.bytes_of_u64(v as u64).to_vec()),
            Self::U64(v) => Some(endianness.bytes_of_u64(v).to_vec()),
            Self::NativeInt(v) => Some(endianness.bytes_of_u64(v as u64).to_vec()),
            Self::NativeUInt(v) => Some(endianness.bytes_of_u64(v).to_vec()),
            _ => None,
        }
    }

    /// Deserializes a byte slice into an unsigned integer constant.
    ///
    /// The byte slice is interpreted as an unsigned integer in the given
    /// endianness. The variant is chosen based on the slice length:
    ///
    /// | Length | Variant |
    /// |--------|---------|
    /// | 1      | `U8`    |
    /// | 2      | `U16`   |
    /// | 4      | `U32`   |
    /// | 8      | `U64`   |
    ///
    /// Returns `None` for other slice lengths.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::{ir::value::ConstValue, target::Endianness, MockTarget};
    ///
    /// let be_bytes = vec![0x01, 0x02, 0x03, 0x04];
    /// let val = ConstValue::<MockTarget>::from_bytes(&be_bytes, Endianness::Big);
    /// assert_eq!(val, Some(ConstValue::U32(0x01020304)));
    /// ```
    #[must_use]
    pub fn from_bytes(bytes: &[u8], endianness: Endianness) -> Option<Self> {
        match bytes.len() {
            1 => Some(Self::U8(bytes.first().copied()?)),
            2 => Some(Self::U16(endianness.read_u16(bytes))),
            4 => Some(Self::U32(endianness.read_u32(bytes))),
            8 => Some(Self::U64(endianness.read_u64(bytes))),
            _ => None,
        }
    }
}

impl<T: Target> ConstValue<T> {
    /// Converts this constant to a different type. Forwards to
    /// `Target::convert_const`; non-CIL hosts that don't override the hook
    /// always observe `None`.
    #[must_use]
    pub fn convert_to(
        &self,
        target: &T::Type,
        unsigned_source: bool,
        ptr_bytes: u32,
    ) -> Option<Self> {
        T::convert_const(self, target, unsigned_source, ptr_bytes)
    }

    /// Converts this constant with overflow checking. Forwards to
    /// `Target::convert_const_checked`.
    #[must_use]
    pub fn convert_to_checked(
        &self,
        target: &T::Type,
        unsigned_source: bool,
        ptr_bytes: u32,
    ) -> Option<Self> {
        T::convert_const_checked(self, target, unsigned_source, ptr_bytes)
    }
}

impl<T: Target> fmt::Display for ConstValue<T>
where
    T::TypeRef: fmt::Display,
    T::MethodRef: fmt::Display,
    T::FieldRef: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I8(v) => write!(f, "{v}i8"),
            Self::I16(v) => write!(f, "{v}i16"),
            Self::I32(v) => write!(f, "{v}"),
            Self::I64(v) => write!(f, "{v}L"),
            Self::U8(v) => write!(f, "{v}u8"),
            Self::U16(v) => write!(f, "{v}u16"),
            Self::U32(v) => write!(f, "{v}u"),
            Self::U64(v) => write!(f, "{v}UL"),
            Self::NativeInt(v) => write!(f, "{v}n"),
            Self::NativeUInt(v) => write!(f, "{v}un"),
            Self::F32(v) => write!(f, "{v}f"),
            Self::F64(v) => write!(f, "{v}"),
            Self::String(idx) => write!(f, "str@{idx}"),
            Self::DecryptedString(s) => write!(f, "\"{}\"", s.escape_default()),
            Self::DecryptedArray {
                data,
                element_type_ref,
                element_size,
            } => {
                write!(
                    f,
                    "array[{}x{}]<{}>",
                    data.len()
                        .checked_div(*element_size.max(&1))
                        .unwrap_or(data.len()),
                    element_size,
                    element_type_ref
                )
            }
            Self::Null => write!(f, "null"),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
            Self::Type(t) => write!(f, "typeof({t})"),
            Self::MethodHandle(m) => write!(f, "methodof({m})"),
            Self::FieldHandle(fl) => write!(f, "fieldof({fl})"),
        }
    }
}

/// Abstract value for dataflow analysis.
///
/// This represents the abstract state of an SSA variable during analysis.
/// It forms a lattice where values can be refined as more information is gathered.
///
/// Generic over the host `Target` because the `Constant` variant carries a
/// `ConstValue<T>`.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum AbstractValue<T: Target> {
    /// No information yet (top of lattice).
    ///
    /// This is the initial state before any analysis.
    #[default]
    Top,

    /// Known constant value.
    Constant(ConstValue<T>),

    /// Known to be non-null (for reference types).
    NonNull,

    /// Value in a bounded range [min, max].
    Range {
        /// Minimum value (inclusive).
        min: i64,
        /// Maximum value (inclusive).
        max: i64,
    },

    /// Same value as another SSA variable.
    ///
    /// Used for copy propagation.
    SameAs(SsaVarId),

    /// Result of a specific computation (for CSE).
    Computed(ComputedValue),

    /// Multiple possible values (bottom of lattice for constants).
    ///
    /// This means the value cannot be determined at compile time.
    Bottom,
}

impl<T: Target> AbstractValue<T> {
    /// Returns `true` if this is the top element (no information).
    #[must_use]
    pub const fn is_top(&self) -> bool {
        matches!(self, Self::Top)
    }

    /// Returns `true` if this is the bottom element (conflicting info).
    #[must_use]
    pub const fn is_bottom(&self) -> bool {
        matches!(self, Self::Bottom)
    }

    /// Returns `true` if this is a known constant.
    #[must_use]
    pub const fn is_constant(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    /// Returns the constant value if this is a constant.
    #[must_use]
    pub const fn as_constant(&self) -> Option<&ConstValue<T>> {
        match self {
            Self::Constant(c) => Some(c),
            _ => None,
        }
    }

    /// Returns `true` if this value is known to be non-null.
    #[must_use]
    pub const fn is_non_null(&self) -> bool {
        matches!(self, Self::NonNull | Self::Constant(_))
    }

    /// Meet operation for the lattice (used at control flow joins).
    ///
    /// Returns the greatest lower bound of `self` and `other`.
    #[must_use]
    #[allow(clippy::match_same_arms)] // Arms kept separate for lattice documentation clarity
    pub fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            // Top meets anything yields the other
            (Self::Top, x) | (x, Self::Top) => x.clone(),

            // Bottom meets anything yields Bottom
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,

            // Same constants stay constant
            (Self::Constant(a), Self::Constant(b)) if a == b => Self::Constant(a.clone()),

            // Different constants become Bottom
            (Self::Constant(_), Self::Constant(_)) => Self::Bottom,

            // NonNull meets NonNull stays NonNull
            (Self::NonNull, Self::NonNull) => Self::NonNull,

            // NonNull meets Constant stays Constant (constants are non-null if not null)
            (Self::NonNull, Self::Constant(c)) | (Self::Constant(c), Self::NonNull) => {
                if c.is_null() {
                    Self::Bottom // null is not non-null
                } else {
                    Self::Constant(c.clone())
                }
            }

            // Ranges can be merged
            (
                Self::Range {
                    min: a_min,
                    max: a_max,
                },
                Self::Range {
                    min: b_min,
                    max: b_max,
                },
            ) => {
                let new_min = (*a_min).min(*b_min);
                let new_max = (*a_max).max(*b_max);
                Self::Range {
                    min: new_min,
                    max: new_max,
                }
            }

            // SameAs values must match
            (Self::SameAs(a), Self::SameAs(b)) if a == b => Self::SameAs(*a),

            // Computed values must match exactly
            (Self::Computed(a), Self::Computed(b)) if a == b => Self::Computed(a.clone()),

            // Otherwise, Bottom
            _ => Self::Bottom,
        }
    }

    /// Join operation for the lattice.
    ///
    /// Returns the least upper bound of `self` and `other`.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        match (self, other) {
            // Bottom joins anything yields the other
            (Self::Bottom, x) | (x, Self::Bottom) => x.clone(),

            // Top joins anything yields Top
            (Self::Top, _) | (_, Self::Top) => Self::Top,

            // Same values stay the same
            (a, b) if a == b => a.clone(),

            // Otherwise, Top
            _ => Self::Top,
        }
    }
}

impl<T: Target> fmt::Display for AbstractValue<T>
where
    T::TypeRef: fmt::Display,
    T::MethodRef: fmt::Display,
    T::FieldRef: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Top => write!(f, "⊤"),
            Self::Constant(c) => write!(f, "{c}"),
            Self::NonNull => write!(f, "!null"),
            Self::Range { min, max } => write!(f, "[{min}..{max}]"),
            Self::SameAs(v) => write!(f, "={v}"),
            Self::Computed(c) => write!(f, "{c}"),
            Self::Bottom => write!(f, "⊥"),
        }
    }
}

/// Computed value for common subexpression elimination (CSE).
///
/// This represents the result of a computation, enabling recognition
/// of equivalent expressions that can be eliminated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputedValue {
    /// The operation that produced this value.
    pub op: ComputedOp,
    /// The operands to the operation.
    pub operands: Vec<SsaVarId>,
}

impl ComputedValue {
    /// Creates a new computed value.
    #[must_use]
    pub fn new(op: ComputedOp, operands: Vec<SsaVarId>) -> Self {
        Self { op, operands }
    }

    /// Creates a unary computed value.
    #[must_use]
    pub fn unary(op: ComputedOp, operand: SsaVarId) -> Self {
        Self {
            op,
            operands: vec![operand],
        }
    }

    /// Creates a binary computed value.
    #[must_use]
    pub fn binary(op: ComputedOp, left: SsaVarId, right: SsaVarId) -> Self {
        Self {
            op,
            operands: vec![left, right],
        }
    }

    /// Normalizes commutative operations for better CSE.
    ///
    /// For commutative ops like add/mul, orders operands consistently
    /// so that `a + b` and `b + a` have the same computed value.
    #[must_use]
    pub fn normalized(self) -> Self {
        if self.op.is_commutative() && self.operands.len() == 2 {
            let mut ops = self.operands;
            if let (Some(a), Some(b)) = (ops.first(), ops.get(1)) {
                if a.index() > b.index() {
                    ops.swap(0, 1);
                }
            }
            Self {
                op: self.op,
                operands: ops,
            }
        } else {
            self
        }
    }
}

impl fmt::Display for ComputedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.op)?;
        for (i, op) in self.operands.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{op}")?;
        }
        write!(f, ")")
    }
}

/// Operations that can be tracked for CSE.
///
/// These represent the pure operations whose results can be reused
/// when the same operation is performed with the same operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputedOp {
    // Arithmetic
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Remainder (modulo)
    Rem,
    /// Negation
    Neg,

    // Bitwise
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
    /// Bitwise NOT
    Not,
    /// Shift left
    Shl,
    /// Shift right
    Shr,

    // Comparison
    /// Compare equal
    Ceq,
    /// Compare not equal
    Cne,
    /// Compare less than
    Clt,
    /// Compare greater than
    Cgt,
    /// Compare less than or equal
    Cle,
    /// Compare greater than or equal
    Cge,

    // Conversion
    /// Convert to int8
    ConvI1,
    /// Convert to int16
    ConvI2,
    /// Convert to int32
    ConvI4,
    /// Convert to int64
    ConvI8,
    /// Convert to uint8
    ConvU1,
    /// Convert to uint16
    ConvU2,
    /// Convert to uint32
    ConvU4,
    /// Convert to uint64
    ConvU8,
    /// Convert to float32
    ConvR4,
    /// Convert to float64
    ConvR8,
}

impl ComputedOp {
    /// Returns `true` if this operation is commutative.
    #[must_use]
    pub const fn is_commutative(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::And | Self::Or | Self::Xor | Self::Ceq | Self::Cne
        )
    }

    /// Returns `true` if this is a comparison operation.
    #[must_use]
    pub const fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Ceq | Self::Cne | Self::Clt | Self::Cgt | Self::Cle | Self::Cge
        )
    }
}

impl fmt::Display for ComputedOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Rem => "rem",
            Self::Neg => "neg",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Not => "not",
            Self::Shl => "shl",
            Self::Shr => "shr",
            Self::Ceq => "ceq",
            Self::Cne => "cne",
            Self::Clt => "clt",
            Self::Cgt => "cgt",
            Self::Cle => "cle",
            Self::Cge => "cge",
            Self::ConvI1 => "conv.i1",
            Self::ConvI2 => "conv.i2",
            Self::ConvI4 => "conv.i4",
            Self::ConvI8 => "conv.i8",
            Self::ConvU1 => "conv.u1",
            Self::ConvU2 => "conv.u2",
            Self::ConvU4 => "conv.u4",
            Self::ConvU8 => "conv.u8",
            Self::ConvR4 => "conv.r4",
            Self::ConvR8 => "conv.r8",
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::{MockTarget, MockType};

    type Cv = ConstValue<MockTarget>;
    type Av = AbstractValue<MockTarget>;

    #[test]
    fn const_value_classifies_and_extracts_scalars() {
        assert!(Cv::Null.is_null());
        assert!(Cv::True.is_bool());
        assert!(Cv::I32(-7).is_integer());
        assert!(Cv::I64(-7).is_signed());
        assert!(Cv::U64(7).is_unsigned());
        assert!(Cv::F64(1.5).is_float());
        assert!(Cv::String(3).is_string_like());
        assert!(Cv::DecryptedString("secret".to_string()).is_string_like());

        assert_eq!(Cv::I16(-2).as_i32(), Some(-2));
        assert_eq!(Cv::U32(9).as_i64(), Some(9));
        assert_eq!(Cv::I32(-1).as_u64(), None);
        assert_eq!(Cv::I32(7).as_u32(), Some(7));
        assert_eq!(Cv::F32(1.25).as_f32(), Some(1.25));
        assert_eq!(Cv::F64(2.5).as_f64(), Some(2.5));
        assert_eq!(Cv::False.as_bool(), Some(false));
        assert_eq!(Cv::I32(5).as_bool(), Some(true));
        assert_eq!(
            Cv::DecryptedString("abc".to_string()).as_decrypted_string(),
            Some("abc")
        );
        assert_eq!(Cv::String(1).as_decrypted_string(), None);
    }

    #[test]
    fn const_value_identity_helpers_preserve_numeric_intent() {
        assert!(Cv::I32(0).is_zero());
        assert!(Cv::NativeUInt(0).is_zero());
        assert!(Cv::True.is_one());
        assert!(Cv::I64(-1).is_minus_one());
        assert!(Cv::U32(u32::MAX).is_all_ones());

        assert_eq!(Cv::U16(99).zero_of_same_type(), Cv::U16(0));
        assert_eq!(Cv::F64(99.0).zero_of_same_type(), Cv::F64(0.0));
        assert_eq!(Cv::I32(99).zero_of_same_type(), Cv::I32(0));
        assert_eq!(Cv::from_bool(true), Cv::True);
        assert_eq!(Cv::from_bool(false), Cv::False);
    }

    #[test]
    fn arithmetic_and_bitwise_operations_fold_constants() {
        let ptr = PointerSize::Bit64;

        assert_eq!(Cv::I32(7).add(&Cv::I32(5), ptr), Some(Cv::I32(12)));
        assert_eq!(Cv::I32(7).sub(&Cv::I32(5), ptr), Some(Cv::I32(2)));
        assert_eq!(Cv::I32(7).mul(&Cv::I32(5), ptr), Some(Cv::I32(35)));
        assert_eq!(Cv::I32(7).div(&Cv::I32(0), ptr), None);
        assert_eq!(Cv::I32(7).div(&Cv::I32(2), ptr), Some(Cv::I32(3)));
        assert_eq!(Cv::I32(7).rem(&Cv::I32(2), ptr), Some(Cv::I32(1)));
        assert_eq!(
            Cv::I32(0b1010).bitwise_and(&Cv::I32(0b1100), ptr),
            Some(Cv::I32(0b1000))
        );
        assert_eq!(
            Cv::I32(0b1010).bitwise_or(&Cv::I32(0b0101), ptr),
            Some(Cv::I32(0b1111))
        );
        assert_eq!(
            Cv::I32(0b1010).bitwise_xor(&Cv::I32(0b0011), ptr),
            Some(Cv::I32(0b1001))
        );
        assert_eq!(Cv::I8(1).shl(&Cv::I32(3), ptr), Some(Cv::I8(8)));
        assert_eq!(Cv::I8(-8).shr(&Cv::I32(1), false, ptr), Some(Cv::I8(-4)));
        assert_eq!(Cv::I8(-8).shr(&Cv::I32(1), true, ptr), Some(Cv::I8(124)));
        assert_eq!(Cv::I32(1).add(&Cv::F32(1.0), ptr), None);
    }

    #[test]
    fn checked_arithmetic_reports_overflow() {
        let ptr = PointerSize::Bit64;

        assert_eq!(
            Cv::I8(120).add_checked(&Cv::I8(7), false, ptr),
            Some(Cv::I8(127))
        );
        assert_eq!(Cv::I8(120).add_checked(&Cv::I8(8), false, ptr), None);
        assert_eq!(
            Cv::U8(10).sub_checked(&Cv::U8(7), true, ptr),
            Some(Cv::U8(3))
        );
        assert_eq!(Cv::U8(7).sub_checked(&Cv::U8(10), true, ptr), None);
        assert_eq!(
            Cv::I16(12).mul_checked(&Cv::I16(11), false, ptr),
            Some(Cv::I16(132))
        );
        assert_eq!(Cv::I16(i16::MAX).mul_checked(&Cv::I16(2), false, ptr), None);
    }

    #[test]
    fn native_values_are_masked_to_pointer_width() {
        assert_eq!(
            Cv::NativeUInt(0x1_0000_0000).mask_native(PointerSize::Bit32),
            Cv::NativeUInt(0)
        );
        assert_eq!(
            Cv::NativeInt(0x0000_0000_8000_0000).mask_native(PointerSize::Bit32),
            Cv::NativeInt(i64::from(i32::MIN))
        );
        assert_eq!(
            Cv::NativeUInt(0x1_0000_0000).add(&Cv::NativeUInt(1), PointerSize::Bit32),
            Some(Cv::NativeUInt(1))
        );
    }

    #[test]
    fn comparisons_handle_signed_unsigned_and_nan_cases() {
        assert_eq!(Cv::I32(4).ceq(&Cv::I64(4)), Some(Cv::True));
        assert_eq!(Cv::U32(4).clt(&Cv::U64(5)), Some(Cv::True));
        assert_eq!(Cv::I8(-1).clt_un(&Cv::I8(1)), Some(Cv::False));
        assert_eq!(Cv::I8(-1).cgt_un(&Cv::I8(1)), Some(Cv::True));
        assert_eq!(Cv::F32(f32::NAN).clt_un(&Cv::F32(1.0)), Some(Cv::True));
        assert_eq!(Cv::F64(f64::NAN).cgt_un(&Cv::F64(1.0)), Some(Cv::True));
        assert_eq!(Cv::Null.ceq(&Cv::Null), Some(Cv::True));
        assert_eq!(Cv::True.ceq(&Cv::False), Some(Cv::False));
    }

    #[test]
    fn mock_target_does_not_support_const_conversions() {
        assert_eq!(Cv::I32(1).convert_to(&MockType::I64, false, 8), None);
        assert_eq!(
            Cv::I32(1).convert_to_checked(&MockType::I64, false, 8),
            None
        );
    }

    #[test]
    fn display_formats_constants_and_computed_values() {
        assert_eq!(Cv::I8(-1).to_string(), "-1i8");
        assert_eq!(Cv::U64(10).to_string(), "10UL");
        assert_eq!(
            Cv::DecryptedString("a\nb".to_string()).to_string(),
            "\"a\\nb\""
        );
        assert_eq!(
            Cv::DecryptedArray {
                data: vec![1, 0, 2, 0],
                element_type_ref: 7,
                element_size: 2
            }
            .to_string(),
            "array[2x2]<7>"
        );
        assert_eq!(Cv::Type(9).to_string(), "typeof(9)");
        assert_eq!(Cv::MethodHandle(10).to_string(), "methodof(10)");
        assert_eq!(Cv::FieldHandle(11).to_string(), "fieldof(11)");

        let computed = ComputedValue::binary(
            ComputedOp::Add,
            SsaVarId::from_index(2),
            SsaVarId::from_index(1),
        )
        .normalized();
        assert_eq!(
            computed.operands,
            vec![SsaVarId::from_index(1), SsaVarId::from_index(2)]
        );
        assert_eq!(computed.to_string(), "add(v1, v2)");
        assert_eq!(
            ComputedValue::unary(ComputedOp::Neg, SsaVarId::from_index(0)).to_string(),
            "neg(v0)"
        );
        assert!(ComputedOp::Ceq.is_commutative());
        assert!(!ComputedOp::Sub.is_commutative());
        assert!(ComputedOp::Cgt.is_comparison());
        assert!(!ComputedOp::Add.is_comparison());
        assert_eq!(ComputedOp::ConvI4.to_string(), "conv.i4");
    }

    #[test]
    fn abstract_value_lattice_meet_and_join_are_predictable() {
        let c1 = Av::Constant(Cv::I32(1));
        let c2 = Av::Constant(Cv::I32(2));

        assert!(Av::Top.is_top());
        assert!(Av::Bottom.is_bottom());
        assert!(c1.is_constant());
        assert_eq!(c1.as_constant(), Some(&Cv::I32(1)));
        assert!(Av::NonNull.is_non_null());
        assert!(!Av::Constant(Cv::Null).meet(&Av::NonNull).is_non_null());

        assert_eq!(Av::Top.meet(&c1), c1);
        assert_eq!(Av::Constant(Cv::I32(1)).meet(&c2), Av::Bottom);
        assert_eq!(
            Av::Range { min: 1, max: 3 }.meet(&Av::Range { min: -2, max: 2 }),
            Av::Range { min: -2, max: 3 }
        );
        assert_eq!(
            Av::SameAs(SsaVarId::from_index(1)).meet(&Av::SameAs(SsaVarId::from_index(1))),
            Av::SameAs(SsaVarId::from_index(1))
        );
        assert_eq!(Av::Bottom.join(&Av::NonNull), Av::NonNull);
        assert_eq!(Av::Top.join(&Av::NonNull), Av::Top);
        assert_eq!(Av::NonNull.join(&Av::NonNull), Av::NonNull);
        assert_eq!(Av::NonNull.join(&Av::Constant(Cv::I32(1))), Av::Top);
        assert_eq!(Av::Range { min: 1, max: 2 }.to_string(), "[1..2]");
    }

    // -----------------------------------------------------------------------
    // ConstValue::to_bytes — all integer widths, both endianness values
    // -----------------------------------------------------------------------

    #[test]
    fn to_bytes_i8_is_one_byte_regardless_of_endianness() {
        let val = Cv::I8(-1);
        for endianness in [Endianness::Little, Endianness::Big] {
            assert_eq!(val.to_bytes(endianness), Some(vec![0xFF]));
        }
    }

    #[test]
    fn to_bytes_u8_is_one_byte_regardless_of_endianness() {
        let val = Cv::U8(0xAB);
        for endianness in [Endianness::Little, Endianness::Big] {
            assert_eq!(val.to_bytes(endianness), Some(vec![0xAB]));
        }
    }

    #[test]
    fn to_bytes_i16_depends_on_endianness() {
        let val = Cv::I16(0x0102);
        assert_eq!(val.to_bytes(Endianness::Little), Some(vec![0x02, 0x01]),);
        assert_eq!(val.to_bytes(Endianness::Big), Some(vec![0x01, 0x02]),);
    }

    #[test]
    fn to_bytes_u32_depends_on_endianness() {
        let val = Cv::U32(0x01020304);
        assert_eq!(
            val.to_bytes(Endianness::Little),
            Some(vec![0x04, 0x03, 0x02, 0x01]),
        );
        assert_eq!(
            val.to_bytes(Endianness::Big),
            Some(vec![0x01, 0x02, 0x03, 0x04]),
        );
    }

    #[test]
    fn to_bytes_i64_depends_on_endianness() {
        let val = Cv::I64(0x0102030405060708);
        assert_eq!(
            val.to_bytes(Endianness::Little),
            Some(vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]),
        );
        assert_eq!(
            val.to_bytes(Endianness::Big),
            Some(vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]),
        );
    }

    #[test]
    fn to_bytes_native_uint_depends_on_endianness() {
        let val = Cv::NativeUInt(0x0102030405060708);
        assert_eq!(
            val.to_bytes(Endianness::Little),
            Some(vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]),
        );
        assert_eq!(
            val.to_bytes(Endianness::Big),
            Some(vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]),
        );
    }

    #[test]
    fn to_bytes_native_int_depends_on_endianness() {
        let val = Cv::NativeInt(0x0102030405060708);
        assert_eq!(
            val.to_bytes(Endianness::Little),
            Some(vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]),
        );
        assert_eq!(
            val.to_bytes(Endianness::Big),
            Some(vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]),
        );
    }

    #[test]
    fn to_bytes_returns_none_for_non_integer_variants() {
        assert_eq!(Cv::F32(1.0).to_bytes(Endianness::Little), None);
        assert_eq!(Cv::F64(1.0).to_bytes(Endianness::Little), None);
        assert_eq!(Cv::Null.to_bytes(Endianness::Little), None);
        assert_eq!(Cv::True.to_bytes(Endianness::Little), None);
        assert_eq!(Cv::False.to_bytes(Endianness::Little), None);
        assert_eq!(
            Cv::DecryptedString("hello".into()).to_bytes(Endianness::Little),
            None,
        );
    }

    // -----------------------------------------------------------------------
    // ConstValue::from_bytes — all supported byte lengths, both endiannesses
    // -----------------------------------------------------------------------

    #[test]
    fn from_bytes_1_byte_produces_u8() {
        assert_eq!(
            Cv::from_bytes(&[0xAB], Endianness::Little),
            Some(Cv::U8(0xAB)),
        );
        assert_eq!(Cv::from_bytes(&[0xAB], Endianness::Big), Some(Cv::U8(0xAB)),);
    }

    #[test]
    fn from_bytes_2_bytes_produces_u16() {
        let le = Cv::from_bytes(&[0x02, 0x01], Endianness::Little);
        assert_eq!(le, Some(Cv::U16(0x0102)));
        let be = Cv::from_bytes(&[0x01, 0x02], Endianness::Big);
        assert_eq!(be, Some(Cv::U16(0x0102)));
    }

    #[test]
    fn from_bytes_4_bytes_produces_u32() {
        let le = Cv::from_bytes(&[0x04, 0x03, 0x02, 0x01], Endianness::Little);
        assert_eq!(le, Some(Cv::U32(0x01020304)));
        let be = Cv::from_bytes(&[0x01, 0x02, 0x03, 0x04], Endianness::Big);
        assert_eq!(be, Some(Cv::U32(0x01020304)));
    }

    #[test]
    fn from_bytes_8_bytes_produces_u64() {
        let bytes_le = [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01];
        let bytes_be = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(
            Cv::from_bytes(&bytes_le, Endianness::Little),
            Some(Cv::U64(0x0102030405060708)),
        );
        assert_eq!(
            Cv::from_bytes(&bytes_be, Endianness::Big),
            Some(Cv::U64(0x0102030405060708)),
        );
    }

    #[test]
    fn from_bytes_unsupported_length_returns_none() {
        assert_eq!(Cv::from_bytes(&[], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0, 0, 0], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0; 5], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0; 6], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0; 7], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0; 9], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0; 16], Endianness::Little), None);
    }

    #[test]
    fn from_bytes_round_trips_with_to_bytes() {
        let test_values: Vec<(Cv, Cv)> = vec![
            (Cv::U8(0xAB), Cv::U8(0xAB)),
            (Cv::U16(0xABCD), Cv::U16(0xABCD)),
            (Cv::U32(0xDEAD_BEEF), Cv::U32(0xDEAD_BEEF)),
            (
                Cv::U64(0xDEAD_BEEF_CAFE_BABE),
                Cv::U64(0xDEAD_BEEF_CAFE_BABE),
            ),
            (Cv::I8(-1), Cv::U8(0xFF)),
            (Cv::I16(-128), Cv::U16(0xFF80)),
            (Cv::I32(i32::MIN), Cv::U32(i32::MIN as u32)),
            (Cv::I64(i64::MIN), Cv::U64(i64::MIN as u64)),
            // NativeInt/NativeUInt → U64 since from_bytes produces unsigned
            (Cv::NativeInt(42), Cv::U64(42)),
            (Cv::NativeUInt(100), Cv::U64(100)),
        ];

        for endianness in [Endianness::Little, Endianness::Big] {
            for (original, expected) in &test_values {
                let bytes = original.to_bytes(endianness).unwrap();
                let restored = Cv::from_bytes(&bytes, endianness).unwrap();
                assert_eq!(
                    restored, *expected,
                    "round-trip failed for {original:?} with {endianness:?}",
                );
            }
        }
    }
}
