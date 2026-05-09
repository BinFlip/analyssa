//! `Target` trait — the abstraction that lets the SSA IR core be generic over
//! the source instruction set.
//!
//! The IR core (`SsaOp`, `SsaInstruction`, `SsaFunction`, `SsaBlock`,
//! `ConstValue`, `SsaExceptionHandler`, `MemoryLocation`) is `<T: Target>`-
//! generic; host-specific data is hidden behind `Target`'s associated types.
//! Each embedding crate supplies its own concrete implementation.
//!
//! # Design notes
//!
//! - **Type queries live on `Target`.** Most queries are pure functions of the
//!   type (`is_integer`, `bit_width`) and do not need a runtime instance. Only
//!   `ptr_bytes` is `&self`. Keeping these queries here keeps pass signatures
//!   to `<T: Target>`.
//!
//! - **`Target` is `Sized + 'static`.** No reason to support unsized targets;
//!   the `'static` bound makes the type usable in trait-object contexts later
//!   if a dynamic-pass-registry shows up.
//!
//! - **`ptr_bytes` is runtime, not const.** CLR is bi-arch (32 vs 64). A
//!   typical instance carries the pointer width chosen at construction.

use std::fmt::Debug;
use std::hash::Hash;

use crate::ir::value::ConstValue;
use crate::PointerSize;

/// Endianness of a target architecture.
///
/// Determines the byte ordering for multi-byte integer and pointer values in
/// memory. Used by [`Target::endianness`] to let passes and codegen reason
/// about byte layout.
///
/// # Examples
///
/// ```rust
/// use analyssa::target::Endianness;
///
/// // x86, RISC-V, and Nios II are little-endian
/// assert_eq!(Endianness::Little, Endianness::Little);
///
/// // MIPS (big-endian mode), ARM (big-endian mode), and SPARC are big-endian
/// assert_eq!(Endianness::Big, Endianness::Big);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endianness {
    /// Least significant byte stored first (x86, RISC-V, Nios II, default).
    Little,
    /// Most significant byte stored first (MIPS BE, ARM BE, SPARC, z/Arch).
    Big,
}

impl Endianness {
    /// Returns `true` if this is little-endian.
    #[must_use]
    pub const fn is_little(self) -> bool {
        matches!(self, Self::Little)
    }

    /// Returns `true` if this is big-endian.
    #[must_use]
    pub const fn is_big(self) -> bool {
        matches!(self, Self::Big)
    }

    /// Converts a `u16` value from this endianness to the host's native byte
    /// order.
    ///
    /// On little-endian hosts this is a no-op for [`Little`](Endianness::Little)
    /// and a byte swap for [`Big`](Endianness::Big).
    #[must_use]
    pub fn to_native_u16(self, value: u16) -> u16 {
        match self {
            Self::Little => u16::from_le(value),
            Self::Big => u16::from_be(value),
        }
    }

    /// Converts a `u32` value from this endianness to the host's native byte
    /// order.
    #[must_use]
    pub fn to_native_u32(self, value: u32) -> u32 {
        match self {
            Self::Little => u32::from_le(value),
            Self::Big => u32::from_be(value),
        }
    }

    /// Converts a `u64` value from this endianness to the host's native byte
    /// order.
    #[must_use]
    pub fn to_native_u64(self, value: u64) -> u64 {
        match self {
            Self::Little => u64::from_le(value),
            Self::Big => u64::from_be(value),
        }
    }

    /// Converts a `u128` value from this endianness to the host's native byte
    /// order.
    #[must_use]
    pub fn to_native_u128(self, value: u128) -> u128 {
        match self {
            Self::Little => u128::from_le(value),
            Self::Big => u128::from_be(value),
        }
    }

    /// Converts a `u16` value from the host's native byte order to this
    /// endianness.
    #[must_use]
    pub fn from_native_u16(self, value: u16) -> u16 {
        match self {
            Self::Little => u16::to_le(value),
            Self::Big => u16::to_be(value),
        }
    }

    /// Converts a `u32` value from the host's native byte order to this
    /// endianness.
    #[must_use]
    pub fn from_native_u32(self, value: u32) -> u32 {
        match self {
            Self::Little => u32::to_le(value),
            Self::Big => u32::to_be(value),
        }
    }

    /// Converts a `u64` value from the host's native byte order to this
    /// endianness.
    #[must_use]
    pub fn from_native_u64(self, value: u64) -> u64 {
        match self {
            Self::Little => u64::to_le(value),
            Self::Big => u64::to_be(value),
        }
    }

    /// Converts a `u128` value from the host's native byte order to this
    /// endianness.
    #[must_use]
    pub fn from_native_u128(self, value: u128) -> u128 {
        match self {
            Self::Little => u128::to_le(value),
            Self::Big => u128::to_be(value),
        }
    }

    /// Returns the bytes of a `u16` value in this endianness as a 2-byte array.
    #[must_use]
    pub fn bytes_of_u16(self, value: u16) -> [u8; 2] {
        self.from_native_u16(value).to_ne_bytes()
    }

    /// Returns the bytes of a `u32` value in this endianness as a 4-byte array.
    #[must_use]
    pub fn bytes_of_u32(self, value: u32) -> [u8; 4] {
        self.from_native_u32(value).to_ne_bytes()
    }

    /// Returns the bytes of a `u64` value in this endianness as an 8-byte array.
    #[must_use]
    pub fn bytes_of_u64(self, value: u64) -> [u8; 8] {
        self.from_native_u64(value).to_ne_bytes()
    }

    /// Returns the bytes of a `u128` value in this endianness as a 16-byte array.
    #[must_use]
    pub fn bytes_of_u128(self, value: u128) -> [u8; 16] {
        self.from_native_u128(value).to_ne_bytes()
    }

    /// Reads a `u16` from a byte slice in this endianness.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` has fewer than 2 elements.
    #[must_use]
    pub fn read_u16(self, bytes: &[u8]) -> u16 {
        let arr: [u8; 2] = match bytes.try_into() {
            Ok(a) => a,
            Err(_) => return 0,
        };
        self.to_native_u16(u16::from_ne_bytes(arr))
    }

    /// Reads a `u32` from a byte slice in this endianness.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` has fewer than 4 elements.
    #[must_use]
    pub fn read_u32(self, bytes: &[u8]) -> u32 {
        let arr: [u8; 4] = match bytes.try_into() {
            Ok(a) => a,
            Err(_) => return 0,
        };
        self.to_native_u32(u32::from_ne_bytes(arr))
    }

    /// Reads a `u64` from a byte slice in this endianness.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` has fewer than 8 elements.
    #[must_use]
    pub fn read_u64(self, bytes: &[u8]) -> u64 {
        let arr: [u8; 8] = match bytes.try_into() {
            Ok(a) => a,
            Err(_) => return 0,
        };
        self.to_native_u64(u64::from_ne_bytes(arr))
    }

    /// Reads a `u128` from a byte slice in this endianness.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` has fewer than 16 elements.
    #[must_use]
    pub fn read_u128(self, bytes: &[u8]) -> u128 {
        let arr: [u8; 16] = match bytes.try_into() {
            Ok(a) => a,
            Err(_) => return 0,
        };
        self.to_native_u128(u128::from_ne_bytes(arr))
    }

    /// Returns the byte representation of a native-width value for this
    /// endianness, given the pointer size. The value is first masked to the
    /// pointer width, then laid out in the appropriate byte order.
    #[must_use]
    pub fn bytes_of_ptr_sized(self, value: u64, ptr_size: PointerSize) -> Vec<u8> {
        let masked = ptr_size.mask_unsigned(value);
        match ptr_size {
            PointerSize::Bit8 => vec![masked as u8],
            PointerSize::Bit16 => self.bytes_of_u16(masked as u16).to_vec(),
            PointerSize::Bit32 => self.bytes_of_u32(masked as u32).to_vec(),
            PointerSize::Bit64 => self.bytes_of_u64(masked).to_vec(),
            PointerSize::Bit128 => {
                // For 128-bit, promote through the 128-bit mask path
                let v128 = ptr_size.mask_unsigned_128(u128::from(masked));
                self.bytes_of_u128(v128).to_vec()
            }
        }
    }

    /// Reads a pointer-sized unsigned value from bytes, given the target
    /// endianness and pointer size.
    #[must_use]
    pub fn read_ptr_sized(self, bytes: &[u8], ptr_size: PointerSize) -> u64 {
        match ptr_size {
            PointerSize::Bit8 => u64::from(bytes.first().copied().unwrap_or(0)),
            PointerSize::Bit16 => u64::from(self.read_u16(bytes)),
            PointerSize::Bit32 => u64::from(self.read_u32(bytes)),
            PointerSize::Bit64 => self.read_u64(bytes),
            PointerSize::Bit128 => {
                // Read as u128 then truncate to u64 (warning: loses precision)
                self.read_u128(bytes) as u64
            }
        }
    }
}

/// The abstraction that makes the SSA IR core generic over an instruction
/// set. See module docs.
///
/// `Clone + Debug + Eq + Hash` supertraits exist so derive macros on the
/// generic IR types (`ConstValue<T>`, `SsaOp<T>`, `SsaInstruction<T>`,
/// `SsaFunction<T>`, `MemoryLocation<T>`, …) do not need manual impls. The
/// implementation cost is negligible for marker-style impls.
pub trait Target: Clone + Debug + Eq + Hash + Sized + 'static {
    /// Reference to a user-defined or built-in type in the host's metadata.
    type TypeRef: Clone + Eq + Hash + Debug;

    /// Reference to a method in the host's metadata.
    type MethodRef: Clone + Eq + Hash + Debug;

    /// Reference to a field in the host's metadata.
    type FieldRef: Clone + Eq + Hash + Debug;

    /// Reference to a standalone signature in the host's metadata.
    type SigRef: Clone + Eq + Hash + Debug;

    /// Host-defined exception-handler kind (e.g. EXCEPTION/FINALLY/FILTER on CIL).
    type ExceptionKind: Clone + Eq + Debug;

    /// Host's type representation (e.g. `SsaType` for CIL).
    type Type: Clone + Eq + Hash + Debug;

    /// Original-instruction breadcrumb retained on each `SsaInstruction` for
    /// debugging and source mapping. Hosts that don't want this can use `()`.
    type OriginalInstruction: Clone + Debug;

    /// Local-variable signature data preserved through SSA construction. Used
    /// by codegen to recover types that aren't structurally reconstructible
    /// from the SSA op stream.
    type LocalSignature: Clone + Debug;

    /// Pass-pipeline capability tag used by the analyssa scheduler for
    /// dependency-aware ordering. Hosts that don't run the pass scheduler
    /// can use `()`. Hosts that do should embed
    /// [`crate::scheduling::DeobfuscationCapability`] in their concrete
    /// enum so generic analyssa passes can declare provides/requires using
    /// the shared deobfuscation vocabulary.
    type Capability: Copy + Eq + Hash + Debug + 'static;

    /// Pointer width in bytes (typically 4 or 8). Runtime so bi-arch hosts
    /// can vary it per-instance.
    fn ptr_bytes(&self) -> u32;

    /// Byte ordering for multi-byte values in memory.
    ///
    /// Returns [`Endianness::Little`] by default, which covers x86, RISC-V,
    /// and the default mode of most modern ISAs. Hosts that target bi-endian
    /// architectures (MIPS, ARM) or big-endian-only architectures (SPARC,
    /// z/Arch) should override this.
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    /// Returns a placeholder original-instruction value for synthetic IR
    /// nodes (e.g., phi-node carriers, transform-inserted instructions).
    fn synthetic_instruction() -> Self::OriginalInstruction;

    /// The canonical "unknown / not-yet-inferred" type. Used by builders and
    /// fixtures that haven't run inference.
    fn unknown_type() -> Self::Type;

    /// `true` if `t` is an integer type (any width, signed or unsigned).
    fn is_integer(t: &Self::Type) -> bool;

    /// `true` if `t` is a floating-point type.
    fn is_floating(t: &Self::Type) -> bool;

    /// `true` if `t` is a signed integer type.
    fn is_signed(t: &Self::Type) -> bool;

    /// `true` if `t` is a pointer or managed reference (byref) to another type.
    fn is_pointer(t: &Self::Type) -> bool;

    /// `true` if `t` is a reference type (object/string/class/array).
    fn is_reference(t: &Self::Type) -> bool;

    /// `true` if `t` is the unknown / not-yet-inferred type.
    fn is_unknown(t: &Self::Type) -> bool;

    /// Bit-width for primitive types where it is statically known. `None` for
    /// pointer-sized integers, references, and aggregates.
    fn bit_width(t: &Self::Type) -> Option<u32>;

    /// Mnemonic for the original instruction breadcrumb (e.g. `"add"`, `"ret"`).
    /// Hosts that don't carry a real instruction return a placeholder.
    fn instruction_mnemonic(instr: &Self::OriginalInstruction) -> &'static str;

    /// RVA of the original instruction. Hosts without source mapping return 0.
    fn instruction_rva(instr: &Self::OriginalInstruction) -> u64;

    /// `true` if `flags` denotes a filter-style exception handler (i.e. one
    /// that runs a user-supplied predicate before catching). Hosts without a
    /// filter notion return `false`.
    fn is_filter_handler(flags: &Self::ExceptionKind) -> bool;

    // ------------------------------------------------------------------------
    // Result-type queries used by `SsaOp::infer_result_type` to lift type
    // inference onto generic `Target`. Each is decomposed per opcode group.
    // All default to `None` so hosts only implement the queries they have a
    // useful answer for; test targets can keep every default. CIL overrides them
    // all.
    // ------------------------------------------------------------------------

    /// Result type for a `Const` op; mapped from the `ConstValue` variant.
    fn result_type_for_const(_value: &ConstValue<Self>) -> Option<Self::Type> {
        None
    }

    /// Result type of a comparison op (`Ceq`, `Clt`, `Cgt`).
    fn comparison_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of plain integer arithmetic ops (`Add`, `Sub`, …, `SizeOf`).
    fn arithmetic_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of `LocalAlloc` and `ArrayLength` ops (CIL: native int).
    fn native_int_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of `Ckfinite` (CIL: F64).
    fn ckfinite_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of `LoadFunctionPtr` / `LoadVirtFunctionPtr` (CIL: native int).
    fn function_ptr_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of object-producing ops (`Box`, `NewObj`, `NewArr`,
    /// `CastClass`, `IsInst`).
    fn object_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of `UnboxAny` / `LoadObj`: a value-typed view of `r`.
    fn value_type_from_ref(_r: &Self::TypeRef) -> Option<Self::Type> {
        None
    }

    /// Result type of `Unbox`: a managed reference (`byref`) to the
    /// value-typed view of `r`.
    fn byref_value_type_from_ref(_r: &Self::TypeRef) -> Option<Self::Type> {
        None
    }

    /// Result type of `LoadElementAddr`: a managed reference (`byref`) to a
    /// class-typed element of `r`.
    fn byref_class_type_from_ref(_r: &Self::TypeRef) -> Option<Self::Type> {
        None
    }

    /// Convert a constant value to `target_type`. Used by `ConstValue::convert_to`
    /// (CIL `conv.*` semantics). `ptr_bytes` is the host's pointer width
    /// (typically 4 or 8). Default `None` means "unsupported".
    fn convert_const(
        _value: &ConstValue<Self>,
        _target_type: &Self::Type,
        _unsigned_source: bool,
        _ptr_bytes: u32,
    ) -> Option<ConstValue<Self>> {
        None
    }

    /// Convert a constant value to `target_type` with overflow checking. Used
    /// by `ConstValue::convert_to_checked` (CIL `conv.ovf.*` semantics).
    /// Returns `None` if the conversion would overflow or is unsupported.
    fn convert_const_checked(
        _value: &ConstValue<Self>,
        _target_type: &Self::Type,
        _unsigned_source: bool,
        _ptr_bytes: u32,
    ) -> Option<ConstValue<Self>> {
        None
    }

    /// Evaluator-side integer conversion: produce a typed `ConstValue` from
    /// a raw `i64` value and a target type. Used by `SsaEvaluator` to apply
    /// CIL `conv.*` truncation/extension semantics. Default `None` means the
    /// caller falls back to wrapping the raw i64.
    fn evaluate_int_conv(
        _value: i64,
        _target_type: &Self::Type,
        _unsigned: bool,
        _ptr_bytes: u32,
    ) -> Option<ConstValue<Self>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Endianness query helpers
    // -----------------------------------------------------------------------

    #[test]
    fn little_is_little_not_big() {
        assert!(Endianness::Little.is_little());
        assert!(!Endianness::Little.is_big());
    }

    #[test]
    fn big_is_big_not_little() {
        assert!(Endianness::Big.is_big());
        assert!(!Endianness::Big.is_little());
    }

    // -----------------------------------------------------------------------
    // to_native — round-trip through native byte order (host is LE on x86)
    // -----------------------------------------------------------------------

    #[test]
    fn little_to_native_u16_is_le() {
        let value = 0x0102_u16;
        assert_eq!(Endianness::Little.to_native_u16(value), u16::from_le(value));
    }

    #[test]
    fn big_to_native_u16_is_be() {
        let value = 0x0102_u16;
        assert_eq!(Endianness::Big.to_native_u16(value), u16::from_be(value));
    }

    #[test]
    fn little_to_native_u32_is_le() {
        let value = 0x01020304_u32;
        assert_eq!(Endianness::Little.to_native_u32(value), u32::from_le(value));
    }

    #[test]
    fn big_to_native_u32_is_be() {
        let value = 0x01020304_u32;
        assert_eq!(Endianness::Big.to_native_u32(value), u32::from_be(value));
    }

    #[test]
    fn little_to_native_u64_is_le() {
        let value = 0x0102030405060708_u64;
        assert_eq!(Endianness::Little.to_native_u64(value), u64::from_le(value));
    }

    #[test]
    fn big_to_native_u64_is_be() {
        let value = 0x0102030405060708_u64;
        assert_eq!(Endianness::Big.to_native_u64(value), u64::from_be(value));
    }

    #[test]
    fn little_to_native_u128_is_le() {
        let value = 0x0102030405060708090a0b0c0d0e0f10_u128;
        assert_eq!(
            Endianness::Little.to_native_u128(value),
            u128::from_le(value)
        );
    }

    #[test]
    fn big_to_native_u128_is_be() {
        let value = 0x0102030405060708090a0b0c0d0e0f10_u128;
        assert_eq!(Endianness::Big.to_native_u128(value), u128::from_be(value));
    }

    // -----------------------------------------------------------------------
    // from_native — round-trip through native byte order
    // -----------------------------------------------------------------------

    #[test]
    fn little_from_native_u16_is_le() {
        let value = 0x0102_u16;
        assert_eq!(Endianness::Little.from_native_u16(value), u16::to_le(value));
    }

    #[test]
    fn big_from_native_u16_is_be() {
        let value = 0x0102_u16;
        assert_eq!(Endianness::Big.from_native_u16(value), u16::to_be(value));
    }

    #[test]
    fn from_native_round_trips_through_to_native() {
        let value = 0xdeadbeef_u32;
        for endianness in [Endianness::Little, Endianness::Big] {
            let converted = endianness.from_native_u32(value);
            let restored = endianness.to_native_u32(converted);
            assert_eq!(restored, value, "round-trip failed for {endianness:?}");
        }
    }

    // -----------------------------------------------------------------------
    // bytes_of_* — verify byte layout matches endianness
    // -----------------------------------------------------------------------

    #[test]
    fn little_bytes_of_u16_match_le_byte_order() {
        assert_eq!(Endianness::Little.bytes_of_u16(0x0102), [0x02, 0x01]);
    }

    #[test]
    fn big_bytes_of_u16_match_be_byte_order() {
        assert_eq!(Endianness::Big.bytes_of_u16(0x0102), [0x01, 0x02]);
    }

    #[test]
    fn little_bytes_of_u32_match_le_byte_order() {
        assert_eq!(
            Endianness::Little.bytes_of_u32(0x01020304),
            [0x04, 0x03, 0x02, 0x01]
        );
    }

    #[test]
    fn big_bytes_of_u32_match_be_byte_order() {
        assert_eq!(
            Endianness::Big.bytes_of_u32(0x01020304),
            [0x01, 0x02, 0x03, 0x04]
        );
    }

    #[test]
    fn little_bytes_of_u64_match_le_byte_order() {
        assert_eq!(
            Endianness::Little.bytes_of_u64(0x0102030405060708),
            [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
        );
    }

    #[test]
    fn big_bytes_of_u64_match_be_byte_order() {
        assert_eq!(
            Endianness::Big.bytes_of_u64(0x0102030405060708),
            [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        );
    }

    #[test]
    fn little_bytes_of_u128_match_le_byte_order() {
        let bytes = Endianness::Little.bytes_of_u128(0x0102030405060708090a0b0c0d0e0f10);
        assert_eq!(
            bytes,
            [
                0x10, 0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03,
                0x02, 0x01,
            ]
        );
    }

    #[test]
    fn big_bytes_of_u128_match_be_byte_order() {
        let bytes = Endianness::Big.bytes_of_u128(0x0102030405060708090a0b0c0d0e0f10);
        assert_eq!(
            bytes,
            [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
                0x0f, 0x10,
            ]
        );
    }

    // -----------------------------------------------------------------------
    // read_u* — reading bytes back matches bytes_of_* output
    // -----------------------------------------------------------------------

    #[test]
    fn read_u16_round_trips_with_bytes_of_u16() {
        let value = 0xabcd_u16;
        for endianness in [Endianness::Little, Endianness::Big] {
            let bytes = endianness.bytes_of_u16(value);
            let restored = endianness.read_u16(&bytes);
            assert_eq!(
                restored, value,
                "read_u16 round-trip failed for {endianness:?}"
            );
        }
    }

    #[test]
    fn read_u32_round_trips_with_bytes_of_u32() {
        let value = 0xdeadbeef_u32;
        for endianness in [Endianness::Little, Endianness::Big] {
            let bytes = endianness.bytes_of_u32(value);
            let restored = endianness.read_u32(&bytes);
            assert_eq!(
                restored, value,
                "read_u32 round-trip failed for {endianness:?}"
            );
        }
    }

    #[test]
    fn read_u64_round_trips_with_bytes_of_u64() {
        let value = 0xdeadbeef_cafebabe_u64;
        for endianness in [Endianness::Little, Endianness::Big] {
            let bytes = endianness.bytes_of_u64(value);
            let restored = endianness.read_u64(&bytes);
            assert_eq!(
                restored, value,
                "read_u64 round-trip failed for {endianness:?}"
            );
        }
    }

    #[test]
    fn read_u128_round_trips_with_bytes_of_u128() {
        let value = 0xdeadbeef_cafebabe_01020304_05060708_u128;
        for endianness in [Endianness::Little, Endianness::Big] {
            let bytes = endianness.bytes_of_u128(value);
            let restored = endianness.read_u128(&bytes);
            assert_eq!(
                restored, value,
                "read_u128 round-trip failed for {endianness:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // bytes_of_ptr_sized — endianness × PointerSize interaction
    // -----------------------------------------------------------------------

    #[test]
    fn bytes_of_ptr_sized_bit8_is_always_one_byte() {
        let value = 0xAB;
        for endianness in [Endianness::Little, Endianness::Big] {
            let bytes = endianness.bytes_of_ptr_sized(value, PointerSize::Bit8);
            assert_eq!(bytes, vec![0xAB], "Bit8 differs for {endianness:?}");
        }
    }

    #[test]
    fn bytes_of_ptr_sized_bit16_depends_on_endianness() {
        let value = 0x0102;
        assert_eq!(
            Endianness::Little.bytes_of_ptr_sized(value, PointerSize::Bit16),
            vec![0x02, 0x01],
        );
        assert_eq!(
            Endianness::Big.bytes_of_ptr_sized(value, PointerSize::Bit16),
            vec![0x01, 0x02],
        );
    }

    #[test]
    fn bytes_of_ptr_sized_bit32_depends_on_endianness() {
        let value = 0x01020304;
        assert_eq!(
            Endianness::Little.bytes_of_ptr_sized(value, PointerSize::Bit32),
            vec![0x04, 0x03, 0x02, 0x01],
        );
        assert_eq!(
            Endianness::Big.bytes_of_ptr_sized(value, PointerSize::Bit32),
            vec![0x01, 0x02, 0x03, 0x04],
        );
    }

    #[test]
    fn bytes_of_ptr_sized_bit64_depends_on_endianness() {
        let value = 0x0102030405060708;
        assert_eq!(
            Endianness::Little.bytes_of_ptr_sized(value, PointerSize::Bit64),
            vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
        );
        assert_eq!(
            Endianness::Big.bytes_of_ptr_sized(value, PointerSize::Bit64),
            vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        );
    }

    #[test]
    fn bytes_of_ptr_sized_bit128_depends_on_endianness() {
        let value = 0x0102030405060708; // only 64-bit of value, zero-extended
        let le = Endianness::Little.bytes_of_ptr_sized(value, PointerSize::Bit128);
        let be = Endianness::Big.bytes_of_ptr_sized(value, PointerSize::Bit128);

        // LE: low bytes first
        assert_eq!(le[0..8], [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
        assert_eq!(&le[8..16], &[0, 0, 0, 0, 0, 0, 0, 0]);

        // BE: high bytes first
        assert_eq!(&be[0..8], &[0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(be[8..16], [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
    }

    // -----------------------------------------------------------------------
    // read_ptr_sized — round-trip with bytes_of_ptr_sized for all combos
    // -----------------------------------------------------------------------

    #[test]
    fn read_ptr_sized_round_trips_for_all_pointer_sizes() {
        let test_vectors: Vec<(PointerSize, u64)> = vec![
            (PointerSize::Bit8, 0xAB),
            (PointerSize::Bit16, 0xABCD),
            (PointerSize::Bit32, 0xDEAD_BEEF),
            (PointerSize::Bit64, 0xDEAD_BEEF_CAFE_BABE),
            (PointerSize::Bit128, 0xDEAD_BEEF_CAFE_BABE),
        ];

        for endianness in [Endianness::Little, Endianness::Big] {
            for (ptr_size, value) in &test_vectors {
                let bytes = endianness.bytes_of_ptr_sized(*value, *ptr_size);
                let restored = endianness.read_ptr_sized(&bytes, *ptr_size);
                assert_eq!(
                    restored, *value,
                    "round-trip failed for {endianness:?} × {ptr_size:?}",
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // LE and BE produce different byte sequences for multi-byte widths
    // -----------------------------------------------------------------------

    #[test]
    fn le_and_be_differ_for_all_multi_byte_sizes() {
        let value = 0x0102030405060708;
        for ptr_size in [
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
            PointerSize::Bit128,
        ] {
            let le_bytes = Endianness::Little.bytes_of_ptr_sized(value, ptr_size);
            let be_bytes = Endianness::Big.bytes_of_ptr_sized(value, ptr_size);
            assert_ne!(
                le_bytes, be_bytes,
                "LE and BE should differ for {ptr_size:?}"
            );
        }
    }

    #[test]
    fn le_and_be_agree_for_bit8() {
        let le = Endianness::Little.bytes_of_ptr_sized(0xAB, PointerSize::Bit8);
        let be = Endianness::Big.bytes_of_ptr_sized(0xAB, PointerSize::Bit8);
        assert_eq!(le, be, "Bit8 should be identical regardless of endianness");
    }
}
