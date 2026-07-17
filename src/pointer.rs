//! Target pointer width abstraction.
//!
//! The [`PointerSize`] enum captures the pointer widths the SSA IR core
//! supports: 8, 16, 32, 64, and 128 bits. It is used by:
//!
//! - **Arithmetic methods** on [`ConstValue`](crate::ir::value::ConstValue) to
//!   mask and sign/zero-extend values to the target's native integer width.
//! - **Host implementations** to communicate pointer size to generic passes
//!   that need to reason about pointer-sized operations.
//!
//! # Algorithmic Details
//!
//! When performing constant folding on `NativeInt`/`NativeUInt` values, the
//! result must be masked to the target pointer width. For 32-bit targets,
//! `mask_signed` truncates to `i32` and sign-extends back to `i64`, while
//! `mask_unsigned` truncates to `u32` and zero-extends back to `u64`. For
//! 64-bit targets, both are no-ops. 128-bit masking is handled by the
//! dedicated [`mask_signed_128`](PointerSize::mask_signed_128) /
//! [`mask_unsigned_128`](PointerSize::mask_unsigned_128) methods which
//! operate on `i128`/`u128`.

/// Target pointer width for 8, 16, 32, 64, and 128-bit architectures.
///
/// Hosts pick the variant that matches their target. This is used throughout
/// the constant evaluation and code generation pipeline to ensure pointer-sized
/// operations produce correctly-sized results.
///
/// # Examples
///
/// ```rust
/// use analyssa::PointerSize;
///
/// let ptr = PointerSize::Bit64;
/// assert_eq!(ptr.bytes(), 8);
/// assert_eq!(ptr.bits(), 64);
///
/// // 32-bit masking truncates to i32 then sign-extends
/// assert_eq!(PointerSize::Bit32.mask_signed(0xFFFFFFFF), -1_i64);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PointerSize {
    /// 8-bit target architecture with 1-byte pointers.
    Bit8,
    /// 16-bit target architecture with 2-byte pointers (x86 real-mode, embedded).
    Bit16,
    /// 32-bit target architecture with 4-byte pointers.
    Bit32,
    /// 64-bit target architecture with 8-byte pointers.
    Bit64,
    /// 128-bit target architecture with 16-byte pointers (experimental RV128, CHERI).
    Bit128,
}

impl PointerSize {
    /// Creates a [`PointerSize`] from a "is 64-bit" boolean.
    ///
    /// Convenience constructor for hosts that communicate bitness as a boolean
    /// flag rather than carrying the enum directly.
    ///
    /// # Arguments
    ///
    /// * `is_64bit` - `true` for 64-bit, `false` for 32-bit.
    ///
    /// # Returns
    ///
    /// [`Bit64`](PointerSize::Bit64) if `is_64bit`, [`Bit32`](PointerSize::Bit32) otherwise.
    #[must_use]
    pub fn from_is_64bit(is_64bit: bool) -> Self {
        if is_64bit {
            Self::Bit64
        } else {
            Self::Bit32
        }
    }

    /// Returns the pointer size in bytes.
    ///
    /// # Returns
    ///
    /// `1` for 8-bit, `2` for 16-bit, `4` for 32-bit, `8` for 64-bit,
    /// `16` for 128-bit targets.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::PointerSize;
    /// assert_eq!(PointerSize::Bit64.bytes(), 8);
    /// ```
    #[must_use]
    pub fn bytes(self) -> usize {
        match self {
            Self::Bit8 => 1,
            Self::Bit16 => 2,
            Self::Bit32 => 4,
            Self::Bit64 => 8,
            Self::Bit128 => 16,
        }
    }

    /// Returns the pointer size in bits.
    ///
    /// # Returns
    ///
    /// `8` for 8-bit, `16` for 16-bit, `32` for 32-bit, `64` for 64-bit,
    /// `128` for 128-bit targets.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::PointerSize;
    /// assert_eq!(PointerSize::Bit32.bits(), 32);
    /// ```
    #[must_use]
    pub fn bits(self) -> u32 {
        match self {
            Self::Bit8 => 8,
            Self::Bit16 => 16,
            Self::Bit32 => 32,
            Self::Bit64 => 64,
            Self::Bit128 => 128,
        }
    }

    /// Truncates a signed `i64` value to the target width then sign-extends it
    /// back to `i64`.
    ///
    /// For 8/16/32-bit targets: truncates to the native width then sign-extends
    /// back to `i64`. For 64-bit targets: returns the value unchanged. For
    /// 128-bit targets: returns the value unchanged (`i64` cannot hold a full
    /// 128-bit value; use [`mask_signed_128`](PointerSize::mask_signed_128)).
    ///
    /// # Arguments
    ///
    /// * `value` - The signed value to mask.
    ///
    /// # Returns
    ///
    /// The value masked to the target pointer width and sign-extended.
    #[must_use]
    pub fn mask_signed(self, value: i64) -> i64 {
        match self {
            Self::Bit8 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as i8;
                i64::from(truncated)
            }
            Self::Bit16 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as i16;
                i64::from(truncated)
            }
            Self::Bit32 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as i32;
                i64::from(truncated)
            }
            Self::Bit64 => value,
            Self::Bit128 => value,
        }
    }

    /// Truncates an unsigned `u64` value to the target width then zero-extends
    /// it back to `u64`.
    ///
    /// For 8/16/32-bit targets: truncates to the native width then zero-extends
    /// back to `u64`. For 64-bit targets: returns the value unchanged. For
    /// 128-bit targets: returns the value unchanged (`u64` cannot hold a full
    /// 128-bit value; use [`mask_unsigned_128`](PointerSize::mask_unsigned_128)).
    ///
    /// # Arguments
    ///
    /// * `value` - The unsigned value to mask.
    ///
    /// # Returns
    ///
    /// The value masked to the target pointer width and zero-extended.
    #[must_use]
    pub fn mask_unsigned(self, value: u64) -> u64 {
        match self {
            Self::Bit8 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as u8;
                u64::from(truncated)
            }
            Self::Bit16 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as u16;
                u64::from(truncated)
            }
            Self::Bit32 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as u32;
                u64::from(truncated)
            }
            Self::Bit64 => value,
            Self::Bit128 => value,
        }
    }

    /// Truncates a signed `i128` value to the target width then sign-extends
    /// it back to `i128`.
    ///
    /// Supports all pointer widths including [`Bit128`](PointerSize::Bit128)
    /// where `i128` is the native carrier.
    ///
    /// # Arguments
    ///
    /// * `value` - The signed value to mask.
    ///
    /// # Returns
    ///
    /// The value masked to the target pointer width and sign-extended.
    #[must_use]
    pub fn mask_signed_128(self, value: i128) -> i128 {
        match self {
            Self::Bit8 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as i8;
                i128::from(truncated)
            }
            Self::Bit16 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as i16;
                i128::from(truncated)
            }
            Self::Bit32 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as i32;
                i128::from(truncated)
            }
            Self::Bit64 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as i64;
                i128::from(truncated)
            }
            Self::Bit128 => value,
        }
    }

    /// Truncates an unsigned `u128` value to the target width then zero-extends
    /// it back to `u128`.
    ///
    /// Supports all pointer widths including [`Bit128`](PointerSize::Bit128)
    /// where `u128` is the native carrier.
    ///
    /// # Arguments
    ///
    /// * `value` - The unsigned value to mask.
    ///
    /// # Returns
    ///
    /// The value masked to the target pointer width and zero-extended.
    #[must_use]
    pub fn mask_unsigned_128(self, value: u128) -> u128 {
        match self {
            Self::Bit8 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as u8;
                u128::from(truncated)
            }
            Self::Bit16 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as u16;
                u128::from(truncated)
            }
            Self::Bit32 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as u32;
                u128::from(truncated)
            }
            Self::Bit64 => {
                #[allow(clippy::cast_possible_truncation)]
                let truncated = value as u64;
                u128::from(truncated)
            }
            Self::Bit128 => value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_is_64bit_maps_bitness_to_pointer_size() {
        assert_eq!(PointerSize::from_is_64bit(false), PointerSize::Bit32);
        assert_eq!(PointerSize::from_is_64bit(true), PointerSize::Bit64);
    }

    #[test]
    fn bytes_and_bits_match_arch_width() {
        assert_eq!(PointerSize::Bit8.bytes(), 1);
        assert_eq!(PointerSize::Bit8.bits(), 8);
        assert_eq!(PointerSize::Bit16.bytes(), 2);
        assert_eq!(PointerSize::Bit16.bits(), 16);
        assert_eq!(PointerSize::Bit32.bytes(), 4);
        assert_eq!(PointerSize::Bit32.bits(), 32);
        assert_eq!(PointerSize::Bit64.bytes(), 8);
        assert_eq!(PointerSize::Bit64.bits(), 64);
        assert_eq!(PointerSize::Bit128.bytes(), 16);
        assert_eq!(PointerSize::Bit128.bits(), 128);
    }

    #[test]
    fn signed_mask_truncates_and_sign_extends_values() {
        assert_eq!(PointerSize::Bit8.mask_signed(0xFF), -1_i64);
        assert_eq!(PointerSize::Bit8.mask_signed(0x7F), 127_i64);

        assert_eq!(PointerSize::Bit16.mask_signed(0xFFFF), -1_i64);
        assert_eq!(PointerSize::Bit16.mask_signed(0x7FFF), 32767_i64);

        assert_eq!(
            PointerSize::Bit32.mask_signed(0x0000_0000_8000_0000),
            i64::from(i32::MIN)
        );
        assert_eq!(
            PointerSize::Bit32.mask_signed(0x0000_0000_7fff_ffff),
            i64::from(i32::MAX)
        );
        assert_eq!(PointerSize::Bit64.mask_signed(i64::MIN), i64::MIN);
        assert_eq!(PointerSize::Bit128.mask_signed(-42), -42);
    }

    #[test]
    fn unsigned_mask_truncates_values() {
        assert_eq!(PointerSize::Bit8.mask_unsigned(0x1FF), 0xFF_u64);
        assert_eq!(PointerSize::Bit16.mask_unsigned(0x1FFFF), 0xFFFF_u64);
        assert_eq!(
            PointerSize::Bit32.mask_unsigned(0x1_ffff_ffff),
            u64::from(u32::MAX)
        );
        assert_eq!(PointerSize::Bit32.mask_unsigned(0x1_0000_0000), 0);
        assert_eq!(PointerSize::Bit64.mask_unsigned(u64::MAX), u64::MAX);
        assert_eq!(PointerSize::Bit128.mask_unsigned(42), 42);
    }

    #[test]
    fn signed_mask_128_truncates_and_sign_extends_values() {
        assert_eq!(PointerSize::Bit8.mask_signed_128(0xFF), -1_i128);
        assert_eq!(PointerSize::Bit16.mask_signed_128(0xFFFF), -1_i128);
        assert_eq!(PointerSize::Bit32.mask_signed_128(0xFFFF_FFFF), -1_i128);
        assert_eq!(
            PointerSize::Bit64.mask_signed_128(0xFFFF_FFFF_FFFF_FFFF),
            -1_i128
        );
        assert_eq!(PointerSize::Bit128.mask_signed_128(i128::MAX), i128::MAX);
    }

    #[test]
    fn unsigned_mask_128_truncates_values() {
        assert_eq!(PointerSize::Bit8.mask_unsigned_128(0x1FF), 0xFF_u128);
        assert_eq!(PointerSize::Bit16.mask_unsigned_128(0x1FFFF), 0xFFFF_u128);
        assert_eq!(
            PointerSize::Bit32.mask_unsigned_128(0x1_FFFF_FFFF),
            u32::MAX as u128
        );
        assert_eq!(
            PointerSize::Bit64.mask_unsigned_128(u128::from(u64::MAX) + 1),
            0
        );
        assert_eq!(PointerSize::Bit128.mask_unsigned_128(u128::MAX), u128::MAX);
    }
}
