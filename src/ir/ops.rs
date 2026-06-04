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

use std::fmt;

use crate::{
    ir::{value::ConstValue, variable::SsaVarId},
    target::{Target, VectorShuffleMask},
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

impl FenceKind {
    /// Returns the closest atomic ordering represented by this fence.
    #[must_use]
    pub const fn ordering(self) -> AtomicOrdering {
        match self {
            Self::Full | Self::SeqCst => AtomicOrdering::SeqCst,
            Self::Acquire => AtomicOrdering::Acquire,
            Self::Release => AtomicOrdering::Release,
            Self::AcqRel => AtomicOrdering::AcqRel,
        }
    }
}

/// Memory ordering constraint for native atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicOrdering {
    /// No cross-thread ordering beyond atomicity.
    Relaxed,
    /// Acquire ordering for operations that read memory.
    Acquire,
    /// Release ordering for operations that write memory.
    Release,
    /// Acquire and release ordering.
    AcqRel,
    /// Sequentially consistent ordering.
    SeqCst,
}

impl fmt::Display for AtomicOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relaxed => write!(f, "relaxed"),
            Self::Acquire => write!(f, "acquire"),
            Self::Release => write!(f, "release"),
            Self::AcqRel => write!(f, "acqrel"),
            Self::SeqCst => write!(f, "seqcst"),
        }
    }
}

/// Access width for native atomic memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicAccessWidth {
    /// 8-bit atomic access.
    Bits8,
    /// 16-bit atomic access.
    Bits16,
    /// 32-bit atomic access.
    Bits32,
    /// 64-bit atomic access.
    Bits64,
    /// 128-bit atomic access.
    Bits128,
    /// Target pointer-sized atomic access.
    Pointer,
}

impl AtomicAccessWidth {
    /// Returns the concrete bit width when it is target-independent.
    #[must_use]
    pub const fn bits(self) -> Option<u32> {
        match self {
            Self::Bits8 => Some(8),
            Self::Bits16 => Some(16),
            Self::Bits32 => Some(32),
            Self::Bits64 => Some(64),
            Self::Bits128 => Some(128),
            Self::Pointer => None,
        }
    }
}

impl fmt::Display for AtomicAccessWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bits8 => write!(f, "i8"),
            Self::Bits16 => write!(f, "i16"),
            Self::Bits32 => write!(f, "i32"),
            Self::Bits64 => write!(f, "i64"),
            Self::Bits128 => write!(f, "i128"),
            Self::Pointer => write!(f, "ptr"),
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

/// High-level operation family used by verifiers, lifters, and pass scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SsaOpClass {
    /// No-operation or synthetic placeholder.
    Synthetic,
    /// Scalar constant, arithmetic, bitwise, comparison, conversion, or select operation.
    Scalar,
    /// Boolean operation over scalar truth values.
    Boolean,
    /// Condition-code flag producer or consumer.
    Flags,
    /// Vector or SIMD operation.
    Vector,
    /// Ordinary memory load, store, allocation, or block operation.
    Memory,
    /// Atomic memory operation.
    Atomic,
    /// Call or function-pointer operation.
    Call,
    /// Control-flow terminator or branch operation.
    Control,
    /// Native opaque operation.
    NativeOpaque,
    /// Implicit-width native arithmetic operation.
    WideArithmetic,
    /// Metadata prefix or target constraint.
    Prefix,
}

/// Stable operation family for similarity and feature extraction.
///
/// These classes are intentionally target-generic and less granular than
/// individual opcodes. They provide a stable vocabulary for MinHash,
/// tracelet, type-flow, memory-shape, and side-effect feature extraction
/// without requiring host crates to match every [`SsaOp`] variant directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SsaSimilarityClass {
    /// Synthetic operation that primarily exists to maintain SSA form.
    Synthetic,
    /// Literal or target metadata constant.
    Constant,
    /// Scalar arithmetic operation.
    Arithmetic,
    /// Scalar bitwise or bit-manipulation operation.
    Bitwise,
    /// Scalar shift or rotate operation.
    ShiftRotate,
    /// Scalar comparison operation.
    Compare,
    /// Boolean operation over truth values.
    Boolean,
    /// Conditional value selection.
    Select,
    /// Type, representation, or checked floating-point conversion.
    Conversion,
    /// Argument, local, or function metadata access.
    TypeFlow,
    /// Memory read or address-producing operation.
    MemoryRead,
    /// Memory write operation.
    MemoryWrite,
    /// Memory read-write or bulk-memory operation.
    MemoryReadWrite,
    /// Allocation operation.
    Allocation,
    /// Atomic memory operation.
    Atomic,
    /// Memory ordering barrier.
    Fence,
    /// Call or function-pointer operation.
    Call,
    /// Control-flow terminator or branch operation.
    Control,
    /// Vector or SIMD operation.
    Vector,
    /// Condition-code flag producer or consumer.
    Flags,
    /// Implicit-width native arithmetic operation.
    WideArithmetic,
    /// Native opaque operation.
    NativeOpaque,
    /// Metadata prefix or target constraint.
    Prefix,
}

/// Canonical target-generic feature token for an SSA operation.
///
/// The token avoids host-specific metadata and variable IDs. It captures the
/// opcode family, side-effect class, arity, and definition count in a stable
/// shape suitable for deterministic similarity features.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SsaFeatureToken {
    /// Stable opcode name.
    pub opcode: &'static str,
    /// Coarse operation class.
    pub op_class: SsaOpClass,
    /// Similarity-oriented operation class.
    pub similarity_class: SsaSimilarityClass,
    /// Memory and side-effect class.
    pub effect_kind: SsaEffectKind,
    /// Number of SSA definitions produced by the operation.
    pub def_count: usize,
    /// Number of SSA variables used by the operation.
    pub use_count: usize,
    /// Whether the operation can trap or throw.
    pub may_throw: bool,
}

impl fmt::Display for SsaFeatureToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "op={};class={:?};sim={:?};effect={:?};defs={};uses={};throw={}",
            self.opcode,
            self.op_class,
            self.similarity_class,
            self.effect_kind,
            self.def_count,
            self.use_count,
            self.may_throw
        )
    }
}

/// Coarse memory/effect class for an SSA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SsaEffectKind {
    /// No side effects and no memory dependency.
    Pure,
    /// Reads memory without writing it.
    Read,
    /// Writes memory without requiring the previous value.
    Write,
    /// Reads and writes memory.
    ReadWrite,
    /// Acts as a memory ordering barrier.
    Fence,
    /// Performs an atomic memory operation.
    Atomic,
    /// Calls unknown host code.
    Call,
    /// Has target-specific effects not otherwise modeled.
    Opaque,
}

/// Abstract memory location class used by effect summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryEffectLocation {
    /// No memory location is involved.
    None,
    /// Precise location is unknown or target-specific.
    Unknown,
    /// Stack memory.
    Stack,
    /// Managed or native heap memory.
    Heap,
    /// Global or static storage.
    Global,
    /// Code memory.
    Code,
    /// Memory-mapped or port-backed I/O.
    Io,
}

/// Detailed memory access semantics for an SSA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryAccessSemantics {
    /// No memory access semantics apply.
    None,
    /// Ordinary memory access.
    Normal,
    /// Volatile memory access.
    Volatile,
    /// Atomic memory access.
    Atomic,
    /// Memory fence or barrier.
    Fence,
    /// Target-specific access whose semantics are opaque.
    Opaque,
}

/// Trap or fault class associated with an SSA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrapClass {
    /// Operation cannot trap.
    None,
    /// Operation may fault for an unknown or target-specific reason.
    Unknown,
    /// Operation may fault on invalid memory access.
    MemoryFault,
    /// Operation may fault on null reference or pointer access.
    NullAccess,
    /// Operation may fault on array bounds checks.
    Bounds,
    /// Operation may fault on integer division by zero.
    DivideByZero,
    /// Operation may fault on arithmetic overflow.
    Overflow,
    /// Operation may fault on invalid type conversion or cast.
    InvalidCast,
    /// Operation transfers a language-level exception.
    UserThrow,
    /// Operation may fault as an illegal or privileged instruction.
    IllegalInstruction,
}

/// Control-flow constraint imposed by an SSA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlEffect {
    /// Does not constrain control flow.
    None,
    /// Terminates the current basic block.
    Terminator,
    /// Calls target code and may transfer control externally.
    Call,
    /// Returns from the current function or handler.
    Return,
    /// Throws or resumes exception propagation.
    Throw,
    /// Target-specific control-flow constraint.
    Opaque,
}

/// Summary of an SSA operation's effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SsaEffects {
    /// Coarse effect class.
    pub kind: SsaEffectKind,
    /// Whether the operation may throw or trap.
    pub may_throw: bool,
    /// Abstract memory location class touched by the operation.
    pub memory: MemoryEffectLocation,
    /// Detailed memory access semantics.
    pub memory_semantics: MemoryAccessSemantics,
    /// Whether the memory access is volatile.
    pub volatile: bool,
    /// Atomic ordering when the operation has atomic or fence semantics.
    pub ordering: Option<AtomicOrdering>,
    /// Trap or fault class when known.
    pub trap: TrapClass,
    /// Control-flow constraint imposed by the operation.
    pub control: ControlEffect,
}

impl SsaEffects {
    /// Returns a pure, non-throwing effect summary.
    #[must_use]
    pub const fn pure() -> Self {
        Self {
            kind: SsaEffectKind::Pure,
            may_throw: false,
            memory: MemoryEffectLocation::None,
            memory_semantics: MemoryAccessSemantics::None,
            volatile: false,
            ordering: None,
            trap: TrapClass::None,
            control: ControlEffect::None,
        }
    }

    /// Creates an effect summary from an effect class and trap flag.
    #[must_use]
    pub const fn new(kind: SsaEffectKind, may_throw: bool) -> Self {
        let memory_semantics = match kind {
            SsaEffectKind::Pure => MemoryAccessSemantics::None,
            SsaEffectKind::Fence => MemoryAccessSemantics::Fence,
            SsaEffectKind::Atomic => MemoryAccessSemantics::Atomic,
            SsaEffectKind::Opaque | SsaEffectKind::Call => MemoryAccessSemantics::Opaque,
            SsaEffectKind::Read | SsaEffectKind::Write | SsaEffectKind::ReadWrite => {
                MemoryAccessSemantics::Normal
            }
        };
        let memory = match kind {
            SsaEffectKind::Pure | SsaEffectKind::Fence => MemoryEffectLocation::None,
            SsaEffectKind::Read
            | SsaEffectKind::Write
            | SsaEffectKind::ReadWrite
            | SsaEffectKind::Atomic
            | SsaEffectKind::Call
            | SsaEffectKind::Opaque => MemoryEffectLocation::Unknown,
        };
        let trap = if may_throw {
            TrapClass::Unknown
        } else {
            TrapClass::None
        };
        let control = match kind {
            SsaEffectKind::Call => ControlEffect::Call,
            SsaEffectKind::Opaque => ControlEffect::Opaque,
            SsaEffectKind::Pure
            | SsaEffectKind::Read
            | SsaEffectKind::Write
            | SsaEffectKind::ReadWrite
            | SsaEffectKind::Fence
            | SsaEffectKind::Atomic => ControlEffect::None,
        };
        Self {
            kind,
            may_throw,
            memory,
            memory_semantics,
            volatile: false,
            ordering: None,
            trap,
            control,
        }
    }

    /// Returns this summary with a refined memory location class.
    #[must_use]
    pub const fn with_memory(mut self, memory: MemoryEffectLocation) -> Self {
        self.memory = memory;
        self
    }

    /// Returns this summary with volatile memory semantics.
    #[must_use]
    pub const fn volatile(mut self) -> Self {
        self.volatile = true;
        self
    }

    /// Returns this summary with atomic memory semantics and ordering.
    #[must_use]
    pub const fn atomic_ordering(mut self, ordering: AtomicOrdering) -> Self {
        self.memory_semantics = MemoryAccessSemantics::Atomic;
        self.ordering = Some(ordering);
        self
    }

    /// Returns this summary with fence semantics and ordering.
    #[must_use]
    pub const fn fence_ordering(mut self, ordering: AtomicOrdering) -> Self {
        self.memory_semantics = MemoryAccessSemantics::Fence;
        self.ordering = Some(ordering);
        self
    }

    /// Returns this summary with a known trap class.
    #[must_use]
    pub const fn with_trap(mut self, trap: TrapClass) -> Self {
        self.trap = trap;
        self.may_throw = !matches!(trap, TrapClass::None);
        self
    }

    /// Returns this summary with a known control-flow constraint.
    #[must_use]
    pub const fn with_control(mut self, control: ControlEffect) -> Self {
        self.control = control;
        if matches!(self.kind, SsaEffectKind::Opaque)
            && !matches!(control, ControlEffect::None | ControlEffect::Opaque)
        {
            self.memory = MemoryEffectLocation::None;
            self.memory_semantics = MemoryAccessSemantics::None;
        }
        self
    }

    /// Returns `true` if the operation has no side effects and cannot trap.
    #[must_use]
    pub const fn is_pure(self) -> bool {
        matches!(self.kind, SsaEffectKind::Pure) && !self.may_throw
    }

    /// Returns `true` if the operation may read memory.
    #[must_use]
    pub const fn reads_memory(self) -> bool {
        matches!(
            self.kind,
            SsaEffectKind::Read
                | SsaEffectKind::ReadWrite
                | SsaEffectKind::Atomic
                | SsaEffectKind::Call
                | SsaEffectKind::Opaque
        )
    }

    /// Returns `true` if the operation may write memory.
    #[must_use]
    pub const fn writes_memory(self) -> bool {
        matches!(
            self.kind,
            SsaEffectKind::Write
                | SsaEffectKind::ReadWrite
                | SsaEffectKind::Atomic
                | SsaEffectKind::Call
                | SsaEffectKind::Opaque
        )
    }

    /// Returns `true` if the operation can be removed when all definitions are unused.
    #[must_use]
    pub const fn removable_when_unused(self) -> bool {
        self.is_pure()
    }
}

/// Original native instruction metadata retained for opaque operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NativeInstructionMetadata {
    /// Architecture or backend family that produced the instruction.
    pub architecture: Option<String>,
    /// Original instruction address when available.
    pub address: Option<u64>,
    /// Original encoded bytes when available.
    pub raw_bytes: Vec<u8>,
}

impl NativeInstructionMetadata {
    /// Creates native instruction metadata.
    #[must_use]
    pub fn new(architecture: Option<String>, address: Option<u64>, raw_bytes: Vec<u8>) -> Self {
        Self {
            architecture,
            address,
            raw_bytes,
        }
    }
}

/// Boxed payload for [`SsaOp::NativeOpaque`].
///
/// `NativeOpaque` carries an entire native instruction's worth of state
/// (mnemonic, original encoding metadata, explicit inputs/outputs, clobbers,
/// and an effect summary). Inlining that into the enum would make *every*
/// [`SsaOp`] as large as this rare variant, so the payload is held behind a
/// `Box` and the common operations stay compact.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeOpaqueData {
    /// Human-readable instruction mnemonic or description.
    pub mnemonic: String,
    /// Original native instruction metadata when known.
    pub metadata: Option<NativeInstructionMetadata>,
    /// Explicit SSA outputs defined by the instruction.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the instruction.
    pub inputs: Vec<SsaVarId>,
    /// Abstract target state clobbered by the instruction.
    pub clobbers: Vec<NativeClobber>,
    /// Conservative effect summary for optimization barriers.
    pub effects: SsaEffects,
}

/// Target register or subregister identity used by native machine-state effects.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NativeRegister {
    /// Architecture or target family that owns the register.
    pub architecture: String,
    /// Register bank or class, such as `gpr`, `xmm`, `zmm`, `p`, or `csr`.
    pub bank: String,
    /// Canonical full-register name used for alias comparisons.
    pub base: String,
    /// Specific architectural spelling, such as `al`, `eax`, `rax`, or `x0`.
    pub name: String,
    /// Bit offset within the canonical full register.
    pub bit_offset: u32,
    /// Bit width of this register view.
    pub bit_width: u32,
}

impl NativeRegister {
    /// Creates a native register descriptor.
    ///
    /// Returns `None` when any identity field is empty or `bit_width` is zero.
    #[must_use]
    pub fn new(
        architecture: impl Into<String>,
        bank: impl Into<String>,
        base: impl Into<String>,
        name: impl Into<String>,
        bit_offset: u32,
        bit_width: u32,
    ) -> Option<Self> {
        let architecture = architecture.into();
        let bank = bank.into();
        let base = base.into();
        let name = name.into();
        if architecture.is_empty() || bank.is_empty() || base.is_empty() || name.is_empty() {
            return None;
        }
        if bit_width == 0 {
            return None;
        }
        Some(Self {
            architecture,
            bank,
            base,
            name,
            bit_offset,
            bit_width,
        })
    }

    /// Returns `true` when this register view overlaps `other`.
    #[must_use]
    pub fn aliases(&self, other: &Self) -> bool {
        if self.architecture != other.architecture
            || self.bank != other.bank
            || self.base != other.base
        {
            return false;
        }
        let self_end = self.bit_offset.saturating_add(self.bit_width);
        let other_end = other.bit_offset.saturating_add(other.bit_width);
        self.bit_offset < other_end && other.bit_offset < self_end
    }

    /// Returns `true` when this register descriptor has valid identity and width fields.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.architecture.is_empty()
            && !self.bank.is_empty()
            && !self.base.is_empty()
            && !self.name.is_empty()
            && self.bit_width != 0
    }
}

/// Abstract native machine-state location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NativeStateLocation {
    /// Concrete architectural register or subregister.
    Register(NativeRegister),
    /// Register class whose concrete member is unknown or intentionally grouped.
    RegisterClass(String),
    /// Target flag register or named flag set.
    Flags(String),
    /// Architectural stack pointer state.
    StackPointer,
    /// Architectural program counter or instruction pointer state.
    ProgramCounter,
    /// Runtime vector-length configuration, such as AArch64 SVE `VL` or RISC-V `vl`.
    VectorLength,
    /// Runtime vector type/configuration state, such as RISC-V `vtype`.
    VectorConfig,
    /// Predicate or mask architectural state not represented as an SSA value.
    PredicateState(String),
    /// Control or status register.
    ControlRegister(String),
    /// Abstract memory location or memory class.
    Memory(String),
    /// Target-specific state not otherwise categorized.
    Other(String),
}

/// Access mode for a native machine-state location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeStateAccessKind {
    /// Reads the prior state value.
    Read,
    /// Writes the state without reading the prior value.
    Write,
    /// Reads and writes the state.
    ReadWrite,
    /// Clobbers the state with an unknown value.
    Clobber,
}

impl NativeStateAccessKind {
    /// Returns `true` when the access reads prior state.
    #[must_use]
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    /// Returns `true` when the access writes or clobbers state.
    #[must_use]
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite | Self::Clobber)
    }
}

/// Explicit native machine-state access for opaque and native operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NativeStateAccess {
    /// Machine-state location being accessed.
    pub location: NativeStateLocation,
    /// Access mode for the location.
    pub kind: NativeStateAccessKind,
    /// Optional access width when the state has a meaningful bit width.
    pub width_bits: Option<u32>,
    /// Whether the access is implicit in the native instruction encoding.
    pub implicit: bool,
}

impl NativeStateAccess {
    /// Creates a native machine-state access descriptor.
    ///
    /// Returns `None` when `width_bits` is present but zero.
    #[must_use]
    pub fn new(
        location: NativeStateLocation,
        kind: NativeStateAccessKind,
        width_bits: Option<u32>,
        implicit: bool,
    ) -> Option<Self> {
        if let Some(0) = width_bits {
            return None;
        }
        Some(Self {
            location,
            kind,
            width_bits,
            implicit,
        })
    }

    /// Creates an implicit read of a machine-state location.
    #[must_use]
    pub fn implicit_read(location: NativeStateLocation, width_bits: Option<u32>) -> Option<Self> {
        Self::new(location, NativeStateAccessKind::Read, width_bits, true)
    }

    /// Creates an implicit write of a machine-state location.
    #[must_use]
    pub fn implicit_write(location: NativeStateLocation, width_bits: Option<u32>) -> Option<Self> {
        Self::new(location, NativeStateAccessKind::Write, width_bits, true)
    }

    /// Creates an implicit read-write access to a machine-state location.
    #[must_use]
    pub fn implicit_read_write(
        location: NativeStateLocation,
        width_bits: Option<u32>,
    ) -> Option<Self> {
        Self::new(location, NativeStateAccessKind::ReadWrite, width_bits, true)
    }

    /// Returns `true` when this access reads prior machine state.
    #[must_use]
    pub const fn reads(&self) -> bool {
        self.kind.reads()
    }

    /// Returns `true` when this access writes or clobbers machine state.
    #[must_use]
    pub const fn writes(&self) -> bool {
        self.kind.writes()
    }

    /// Returns `true` when this machine-state access is structurally valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if let Some(0) = self.width_bits {
            return false;
        }
        match &self.location {
            NativeStateLocation::Register(register) => register.is_valid(),
            NativeStateLocation::RegisterClass(name)
            | NativeStateLocation::Flags(name)
            | NativeStateLocation::PredicateState(name)
            | NativeStateLocation::ControlRegister(name)
            | NativeStateLocation::Memory(name)
            | NativeStateLocation::Other(name) => !name.is_empty(),
            NativeStateLocation::StackPointer
            | NativeStateLocation::ProgramCounter
            | NativeStateLocation::VectorLength
            | NativeStateLocation::VectorConfig => true,
        }
    }
}

/// Abstract location clobbered by an opaque native operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NativeClobber {
    /// Structured machine-state access.
    MachineState(NativeStateAccess),
    /// Concrete target register or subregister.
    Register(NativeRegister),
    /// A target register or register alias class.
    RegisterClass(String),
    /// A target flag set such as x86 `eflags`.
    Flags(String),
    /// An abstract memory location or memory class.
    Memory(String),
    /// Target-specific state not represented by data or memory operands.
    Other(String),
}

impl NativeClobber {
    /// Returns `true` when this clobber touches register state.
    #[must_use]
    pub fn touches_registers(&self) -> bool {
        match self {
            Self::MachineState(access) => matches!(
                access.location,
                NativeStateLocation::Register(_) | NativeStateLocation::RegisterClass(_)
            ),
            Self::Register(_) | Self::RegisterClass(_) => true,
            Self::Flags(_) | Self::Memory(_) | Self::Other(_) => false,
        }
    }

    /// Returns `true` when this clobber touches flags or condition-code state.
    #[must_use]
    pub fn touches_flags(&self) -> bool {
        match self {
            Self::MachineState(access) => matches!(access.location, NativeStateLocation::Flags(_)),
            Self::Flags(_) => true,
            Self::Register(_) | Self::RegisterClass(_) | Self::Memory(_) | Self::Other(_) => false,
        }
    }

    /// Returns `true` when this clobber touches memory state.
    #[must_use]
    pub fn touches_memory(&self) -> bool {
        match self {
            Self::MachineState(access) => matches!(access.location, NativeStateLocation::Memory(_)),
            Self::Memory(_) => true,
            Self::Register(_) | Self::RegisterClass(_) | Self::Flags(_) | Self::Other(_) => false,
        }
    }
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
    /// Carry flag bit.
    pub const CARRY: Self = Self(1 << 0);
    /// Parity flag bit.
    pub const PARITY: Self = Self(1 << 1);
    /// Auxiliary carry / adjust flag bit.
    pub const ADJUST: Self = Self(1 << 2);
    /// Zero flag bit.
    pub const ZERO: Self = Self(1 << 3);
    /// Sign flag bit.
    pub const SIGN: Self = Self(1 << 4);
    /// Overflow flag bit.
    pub const OVERFLOW: Self = Self(1 << 5);

    /// Creates a flag mask from raw bits.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }
    /// Returns the raw flag bits.
    pub const fn bits(self) -> u16 {
        self.0
    }
    /// Returns `true` when the mask selects no flags.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns `true` when all bits in `other` are selected by this mask.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the union of two flag masks.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the x86/x64 arithmetic status flags.
    pub const fn x86_status() -> Self {
        Self(
            Self::CARRY.0
                | Self::PARITY.0
                | Self::ADJUST.0
                | Self::ZERO.0
                | Self::SIGN.0
                | Self::OVERFLOW.0,
        )
    }

    /// Returns the mask bit for a target-independent flag bit.
    pub const fn from_flag_bit(bit: NativeFlagBit) -> Self {
        match bit {
            NativeFlagBit::Carry => Self::CARRY,
            NativeFlagBit::Parity => Self::PARITY,
            NativeFlagBit::Adjust => Self::ADJUST,
            NativeFlagBit::Zero => Self::ZERO,
            NativeFlagBit::Sign => Self::SIGN,
            NativeFlagBit::Overflow => Self::OVERFLOW,
        }
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

/// Target-independent status flag bit used by native flag semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeFlagBit {
    /// Carry or borrow flag, such as x86 `CF` or AArch64 `C`.
    Carry,
    /// Parity flag, such as x86 `PF`.
    Parity,
    /// Auxiliary carry or adjust flag, such as x86 `AF`.
    Adjust,
    /// Zero flag, such as x86 `ZF` or AArch64 `Z`.
    Zero,
    /// Sign or negative flag, such as x86 `SF` or AArch64 `N`.
    Sign,
    /// Signed overflow flag, such as x86 `OF` or AArch64 `V`.
    Overflow,
}

/// Describes how an instruction writes one native status flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlagWriteState {
    /// The flag receives a defined value from the instruction semantics.
    Defined,
    /// The flag receives an architecturally undefined value.
    Undefined,
    /// The flag keeps its prior value.
    Preserved,
    /// The flag is architecturally cleared to zero.
    Cleared,
}

/// One native flag write performed by an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlagWrite {
    /// Flag bit affected by the instruction.
    pub bit: NativeFlagBit,
    /// Write behavior for the flag bit.
    pub state: FlagWriteState,
}

impl FlagWrite {
    /// Creates a native flag write descriptor.
    #[must_use]
    pub const fn new(bit: NativeFlagBit, state: FlagWriteState) -> Self {
        Self { bit, state }
    }

    /// Creates a descriptor for a defined flag write.
    #[must_use]
    pub const fn defined(bit: NativeFlagBit) -> Self {
        Self::new(bit, FlagWriteState::Defined)
    }

    /// Creates a descriptor for an undefined flag write.
    #[must_use]
    pub const fn undefined(bit: NativeFlagBit) -> Self {
        Self::new(bit, FlagWriteState::Undefined)
    }

    /// Creates a descriptor for a preserved flag.
    #[must_use]
    pub const fn preserved(bit: NativeFlagBit) -> Self {
        Self::new(bit, FlagWriteState::Preserved)
    }

    /// Creates a descriptor for a cleared flag.
    #[must_use]
    pub const fn cleared(bit: NativeFlagBit) -> Self {
        Self::new(bit, FlagWriteState::Cleared)
    }
}

const X86_STATUS_DEFINED: &[FlagWrite] = &[
    FlagWrite::defined(NativeFlagBit::Carry),
    FlagWrite::defined(NativeFlagBit::Parity),
    FlagWrite::defined(NativeFlagBit::Adjust),
    FlagWrite::defined(NativeFlagBit::Zero),
    FlagWrite::defined(NativeFlagBit::Sign),
    FlagWrite::defined(NativeFlagBit::Overflow),
];

const X86_LOGICAL_WRITES: &[FlagWrite] = &[
    FlagWrite::cleared(NativeFlagBit::Carry),
    FlagWrite::defined(NativeFlagBit::Parity),
    FlagWrite::undefined(NativeFlagBit::Adjust),
    FlagWrite::defined(NativeFlagBit::Zero),
    FlagWrite::defined(NativeFlagBit::Sign),
    FlagWrite::cleared(NativeFlagBit::Overflow),
];

const X86_MUL_WRITES: &[FlagWrite] = &[
    FlagWrite::defined(NativeFlagBit::Carry),
    FlagWrite::undefined(NativeFlagBit::Parity),
    FlagWrite::undefined(NativeFlagBit::Adjust),
    FlagWrite::undefined(NativeFlagBit::Zero),
    FlagWrite::undefined(NativeFlagBit::Sign),
    FlagWrite::defined(NativeFlagBit::Overflow),
];

const X86_ROTATE_WRITES: &[FlagWrite] = &[
    FlagWrite::defined(NativeFlagBit::Carry),
    FlagWrite::preserved(NativeFlagBit::Parity),
    FlagWrite::preserved(NativeFlagBit::Adjust),
    FlagWrite::preserved(NativeFlagBit::Zero),
    FlagWrite::preserved(NativeFlagBit::Sign),
    FlagWrite::defined(NativeFlagBit::Overflow),
];

const AARCH64_NZCV_DEFINED: &[FlagWrite] = &[
    FlagWrite::defined(NativeFlagBit::Sign),
    FlagWrite::defined(NativeFlagBit::Zero),
    FlagWrite::defined(NativeFlagBit::Carry),
    FlagWrite::defined(NativeFlagBit::Overflow),
];

const AARCH64_LOGICAL_WRITES: &[FlagWrite] = &[
    FlagWrite::defined(NativeFlagBit::Sign),
    FlagWrite::defined(NativeFlagBit::Zero),
    FlagWrite::cleared(NativeFlagBit::Carry),
    FlagWrite::cleared(NativeFlagBit::Overflow),
];

/// Canonical native flag producer semantics for common instruction families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlagProducerSemantics {
    /// x86/x64 add, adc, sub, sbb, cmp, inc, and dec style arithmetic flags.
    X86Arithmetic,
    /// x86/x64 and, or, xor, and test style logical flags.
    X86Logical,
    /// x86/x64 mul and imul style implicit-width multiply flags.
    X86Multiply,
    /// x86/x64 shift flags.
    X86Shift,
    /// x86/x64 rotate flags.
    X86Rotate,
    /// AArch64 adds, subs, cmp, and cmn style `NZCV` arithmetic flags.
    AArch64Arithmetic,
    /// AArch64 logical instructions that update `NZCV`, such as `ANDS`.
    AArch64Logical,
}

impl FlagProducerSemantics {
    /// Returns the flag writes performed by this native flag producer.
    #[must_use]
    pub const fn writes(self) -> &'static [FlagWrite] {
        match self {
            Self::X86Arithmetic | Self::X86Shift => X86_STATUS_DEFINED,
            Self::X86Logical => X86_LOGICAL_WRITES,
            Self::X86Multiply => X86_MUL_WRITES,
            Self::X86Rotate => X86_ROTATE_WRITES,
            Self::AArch64Arithmetic => AARCH64_NZCV_DEFINED,
            Self::AArch64Logical => AARCH64_LOGICAL_WRITES,
        }
    }

    /// Returns the set of flags whose value is defined after this producer.
    #[must_use]
    pub fn defined_mask(self) -> FlagsMask {
        let mut mask = FlagsMask::from_bits(0);
        for write in self.writes() {
            if matches!(
                write.state,
                FlagWriteState::Defined | FlagWriteState::Cleared
            ) {
                mask = mask.union(FlagsMask::from_flag_bit(write.bit));
            }
        }
        mask
    }
}

/// Condition code for flag-based branch operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlagCondition {
    /// Tests whether carry is set.
    Carry,
    /// Tests whether carry is clear.
    NotCarry,
    /// Tests whether zero is set.
    Zero,
    /// Tests whether zero is clear.
    NotZero,
    /// Tests whether overflow is set.
    Overflow,
    /// Tests whether overflow is clear.
    NotOverflow,
    /// Tests whether sign is set.
    Negative,
    /// Tests whether sign is clear.
    Positive,
    /// Tests whether parity is even.
    ParityEven,
    /// Tests whether parity is odd.
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

impl FlagCondition {
    /// Returns the status flags required to evaluate this condition.
    #[must_use]
    pub const fn required_flags(self) -> FlagsMask {
        match self {
            Self::Carry | Self::NotCarry => FlagsMask::CARRY,
            Self::Zero | Self::NotZero => FlagsMask::ZERO,
            Self::Overflow | Self::NotOverflow => FlagsMask::OVERFLOW,
            Self::Negative | Self::Positive => FlagsMask::SIGN,
            Self::ParityEven | Self::ParityOdd => FlagsMask::PARITY,
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
    /// Rotate through carry right.
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

/// Iterator over variables defined by an SSA operation.
pub struct SsaDefs<'a> {
    primary: Option<SsaVarId>,
    secondary: Option<SsaVarId>,
    extra: Option<std::slice::Iter<'a, SsaVarId>>,
}

impl<'a> SsaDefs<'a> {
    /// Creates a definition iterator from optional primary, optional secondary,
    /// and any extra definitions.
    #[must_use]
    pub fn new(
        primary: Option<SsaVarId>,
        secondary: Option<SsaVarId>,
        extra: Option<&'a [SsaVarId]>,
    ) -> Self {
        Self {
            primary,
            secondary,
            extra: extra.map(<[SsaVarId]>::iter),
        }
    }
}

impl Iterator for SsaDefs<'_> {
    type Item = SsaVarId;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(primary) = self.primary.take() {
            return Some(primary);
        }
        if let Some(secondary) = self.secondary.take() {
            return Some(secondary);
        }
        self.extra.as_mut()?.next().copied()
    }
}

/// Vector unary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorUnaryKind {
    /// Lane-wise integer or floating negation.
    Neg,
    /// Lane-wise bitwise not.
    Not,
    /// Lane-wise population count.
    Popcount,
    /// Lane-wise absolute value.
    Abs,
    /// Horizontal lane sum.
    HorizontalAdd,
}

/// Vector binary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorBinaryKind {
    /// Lane-wise addition.
    Add,
    /// Lane-wise subtraction.
    Sub,
    /// Lane-wise multiplication.
    Mul,
    /// Lane-wise division.
    Div,
    /// Lane-wise minimum.
    Min,
    /// Lane-wise maximum.
    Max,
    /// Lane-wise bitwise and.
    And,
    /// Lane-wise bitwise or.
    Or,
    /// Lane-wise bitwise xor.
    Xor,
    /// Lane-wise shift left.
    Shl,
    /// Lane-wise shift right.
    Shr,
}

/// Vector ternary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorTernaryKind {
    /// Lane-wise fused multiply-add.
    Fma,
    /// Lane-wise select controlled by a vector mask.
    Select,
}

/// Vector comparison operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorCompareKind {
    /// Lane-wise equality comparison.
    Eq,
    /// Lane-wise inequality comparison.
    Ne,
    /// Lane-wise less-than comparison.
    Lt,
    /// Lane-wise less-than-or-equal comparison.
    Le,
    /// Lane-wise greater-than comparison.
    Gt,
    /// Lane-wise greater-than-or-equal comparison.
    Ge,
}

/// Vector cast operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorCastKind {
    /// Converts lanes with signed integer interpretation.
    Signed,
    /// Converts lanes with unsigned integer interpretation.
    Unsigned,
    /// Converts lanes with floating-point interpretation.
    Float,
}

/// Vector mask unary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorMaskUnaryKind {
    /// Lane-wise predicate negation.
    Not,
}

/// Vector mask binary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorMaskBinaryKind {
    /// Lane-wise predicate conjunction.
    And,
    /// Lane-wise predicate disjunction.
    Or,
    /// Lane-wise predicate exclusive-or.
    Xor,
    /// Lane-wise `left & !right`.
    AndNot,
}

/// Scalar summary operation over vector lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorReduceKind {
    /// Adds all lanes.
    Add,
    /// Multiplies all lanes.
    Mul,
    /// Bitwise-ands all lanes.
    And,
    /// Bitwise-ors all lanes.
    Or,
    /// Bitwise-xors all lanes.
    Xor,
    /// Computes the minimum lane.
    Min,
    /// Computes the maximum lane.
    Max,
    /// Tests whether any predicate lane is true.
    Any,
    /// Tests whether all predicate lanes are true.
    All,
}

/// Operation used when extracting a scalar bitmask from a vector or mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorBitmaskKind {
    /// Extracts each lane's most significant bit, as in x86 `pmovmskb`.
    LaneMostSignificantBits,
    /// Extracts predicate lane bits from a vector mask.
    PredicateBits,
}

/// Controls predicated vector memory behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorMaskMode {
    /// Inactive lanes preserve a passthrough vector value.
    Merge,
    /// Inactive lanes are zeroed or ignored.
    Zero,
}

/// Direction of an AVX-512-style vector lane packing operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorPackKind {
    /// Packs active source lanes contiguously into the low destination lanes.
    Compress,
    /// Expands contiguous low source lanes into active destination lanes.
    Expand,
}

/// Fault behavior for vector loads that may complete partially.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorFaultMode {
    /// Ordinary load where any lane fault traps the instruction.
    Normal,
    /// First-fault load where lanes after the first fault are suppressed.
    FirstFault,
    /// Fault-only-first load where only the first active lane can fault.
    FaultOnlyFirst,
}

/// Memory layout for segmented vector memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorSegmentLayout {
    /// Interleaved structure layout, such as RVV `vlseg` or AArch64 `ld2`.
    Interleaved,
    /// Consecutive structure layout, with each segment stored contiguously.
    Consecutive,
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
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Constant value assigned to the destination.
        value: ConstValue<T>,
    },

    /// Addition: `dest = left + right`
    Add {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Addition with overflow check: `dest = left + right` (throws on overflow)
    AddOvf {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Subtraction: `dest = left - right`
    Sub {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Subtraction with overflow check: `dest = left - right`
    SubOvf {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Multiplication: `dest = left * right`
    Mul {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Multiplication with overflow check: `dest = left * right`
    MulOvf {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Native wide multiply producing low and high halves.
    WideMul {
        /// Low-half output or dividend variable.
        low: SsaVarId,
        /// High-half output or dividend variable.
        high: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
    },

    /// Division: `dest = left / right`
    Div {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Remainder: `dest = left % right`
    Rem {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Floating-point comparison producing a native flags value.
    ///
    /// Models ordered/unordered target flag semantics such as AArch64 `FCMP`
    /// producing `NZCV`. The `signaling` form records exception-sensitive
    /// comparisons such as AArch64 `FCMPE`.
    FloatCompareFlags {
        /// Native flags output variable.
        flags: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether signaling NaNs may raise a floating-point exception.
        signaling: bool,
    },

    /// Native wide divide consuming high:low dividend halves.
    WideDiv {
        /// Quotient output variable.
        quotient: SsaVarId,
        /// Remainder output variable.
        remainder: SsaVarId,
        /// High-half output or dividend variable.
        high: SsaVarId,
        /// Low-half output or dividend variable.
        low: SsaVarId,
        /// Divisor operand variable.
        divisor: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
    },

    /// Negation: `dest = -operand`
    Neg {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Bitwise AND: `dest = left & right`
    And {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Bitwise OR: `dest = left | right`
    Or {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Bitwise XOR: `dest = left ^ right`
    Xor {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Bitwise NOT: `dest = ~operand`
    Not {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Shift left: `dest = value << amount`
    Shl {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Shift right: `dest = value >> amount`
    Shr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Rotate left: `dest = value <<< amount`
    Rol {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
    },

    /// Rotate right: `dest = value >>> amount`
    Ror {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
    },

    /// Rotate through carry left
    Rcl {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
    },

    /// Rotate through carry right
    Rcr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
    },

    /// Byte swap (endian conversion): `dest = bswap(src)`
    BSwap {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Bit reverse: `dest = brev(src)`
    BRev {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Bit scan forward (find first set bit, LSB-based): `dest = bsf(src)`
    BitScanForward {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Bit scan reverse (find first set bit, MSB-based): `dest = bsr(src)`
    BitScanReverse {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Population count: `dest = popcnt(src)`
    Popcount {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Parity: `dest = parity(src)` — 1 if odd number of set bits, 0 if even
    Parity {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Compare equal: `dest = (left == right) ? 1 : 0`
    Ceq {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
    },

    /// Compare less than: `dest = (left < right) ? 1 : 0`
    Clt {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
    },

    /// Compare greater than: `dest = (left > right) ? 1 : 0`
    Cgt {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
    },

    /// Boolean conjunction: `dest = left && right`.
    BoolAnd {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
    },

    /// Boolean disjunction: `dest = left || right`.
    BoolOr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
    },

    /// Boolean exclusive-or: `dest = left != right`.
    BoolXor {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
    },

    /// Boolean negation: `dest = !value`.
    BoolNot {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
    },

    /// Type conversion: `dest = (target_type)operand`
    Conv {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target type metadata.
        target: T::Type,
        /// Whether the conversion checks overflow.
        overflow_check: bool,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
    },

    /// Representation-preserving scalar bitcast: `dest = bitcast(target)operand`.
    Bitcast {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target type metadata.
        target: T::Type,
    },

    /// Conditional select: `dest = condition ? true_val : false_val`
    Select {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Condition operand variable.
        condition: SsaVarId,
        /// Value selected when the condition is true.
        true_val: SsaVarId,
        /// Value selected when the condition is false.
        false_val: SsaVarId,
    },

    /// Read condition code flags from a flags variable.
    ReadFlags {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Flags operand variable.
        flags: SsaVarId,
        /// Flag mask to read.
        mask: FlagsMask,
    },

    /// Lane-wise unary vector operation.
    VectorUnary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Vector operation kind.
        kind: VectorUnaryKind,
    },

    /// Lane-wise binary vector operation.
    VectorBinary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Vector operation kind.
        kind: VectorBinaryKind,
    },

    /// Lane-wise ternary vector operation.
    VectorTernary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// First vector operand variable.
        first: SsaVarId,
        /// Second vector operand variable.
        second: SsaVarId,
        /// Third vector operand variable.
        third: SsaVarId,
        /// Vector operation kind.
        kind: VectorTernaryKind,
    },

    /// Predicated lane-wise unary vector operation.
    VectorPredicatedUnary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector operation kind.
        kind: VectorUnaryKind,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Predicated lane-wise binary vector operation.
    VectorPredicatedBinary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector operation kind.
        kind: VectorBinaryKind,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Predicated lane-wise ternary vector operation.
    VectorPredicatedTernary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// First vector operand variable.
        first: SsaVarId,
        /// Second vector operand variable.
        second: SsaVarId,
        /// Third vector operand variable.
        third: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector operation kind.
        kind: VectorTernaryKind,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Lane-wise vector comparison producing a vector mask.
    VectorCompare {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Vector comparison kind.
        kind: VectorCompareKind,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
    },

    /// Vector load from memory.
    VectorLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Vector store to memory.
    VectorStore {
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Predicated vector load from memory.
    VectorMaskedLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Predicated vector store to memory.
    VectorMaskedStore {
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Loads one scalar value and broadcasts it to all vector lanes.
    VectorBroadcastLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Gathers vector lanes from memory using vector indices.
    VectorGather {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Base address operand variable.
        base: SsaVarId,
        /// Vector index operand variable.
        indices: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Vector load with first-fault or fault-only-first behavior.
    VectorFaultingLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Optional fault/status output variable.
        fault: Option<SsaVarId>,
        /// Address operand variable.
        addr: SsaVarId,
        /// Optional mask operand variable.
        mask: Option<SsaVarId>,
        /// Optional passthrough vector for inactive or fault-suppressed lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Faulting load behavior.
        fault_mode: VectorFaultMode,
        /// Inactive lane behavior.
        mask_mode: VectorMaskMode,
    },

    /// Loads multiple vector segments from memory.
    VectorSegmentLoad {
        /// Destination vector variables, one per segment.
        dests: Vec<SsaVarId>,
        /// Base address operand variable.
        base: SsaVarId,
        /// Optional mask operand variable.
        mask: Option<SsaVarId>,
        /// Vector type metadata for each segment.
        vector_type: T::Type,
        /// Number of segments loaded.
        segments: u32,
        /// Segment memory layout.
        layout: VectorSegmentLayout,
    },

    /// Scatters vector lanes to memory using vector indices.
    VectorScatter {
        /// Base address operand variable.
        base: SsaVarId,
        /// Vector index operand variable.
        indices: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Stores multiple vector segments to memory.
    VectorSegmentStore {
        /// Base address operand variable.
        base: SsaVarId,
        /// Vector values to store, one per segment.
        values: Vec<SsaVarId>,
        /// Optional mask operand variable.
        mask: Option<SsaVarId>,
        /// Vector type metadata for each segment.
        vector_type: T::Type,
        /// Number of segments stored.
        segments: u32,
        /// Segment memory layout.
        layout: VectorSegmentLayout,
    },

    /// Extracts one scalar lane from a vector.
    VectorExtract {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Vector operand variable.
        vector: SsaVarId,
        /// Zero-based lane index.
        lane: u32,
    },

    /// Inserts one scalar lane into a vector.
    VectorInsert {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Vector operand variable.
        vector: SsaVarId,
        /// Zero-based lane index.
        lane: u32,
        /// Value operand variable.
        value: SsaVarId,
    },

    /// Builds a vector by splatting one scalar value to every lane.
    VectorSplat {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Shuffles lanes from one or two vector inputs.
    VectorShuffle {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Optional second shuffle input vector.
        right: Option<SsaVarId>,
        /// Vector shuffle lane selector.
        mask: VectorShuffleMask,
    },

    /// Converts vector lane values to another vector type.
    VectorCast {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Target vector type metadata.
        target_type: T::Type,
        /// Vector cast kind.
        kind: VectorCastKind,
    },

    /// Reinterprets vector bits as another vector type of the same total width.
    VectorReinterpret {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Target vector type metadata.
        target_type: T::Type,
    },

    /// Packs or expands vector lanes under a predicate mask.
    VectorPack {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive destination lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Lane element width in bits.
        element_bits: u32,
        /// Packing direction.
        kind: VectorPackKind,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Loads compact vector lanes and expands them under a predicate mask.
    VectorPackLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Base address operand variable.
        addr: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive destination lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Lane element width in bits.
        element_bits: u32,
        /// Packing direction. Must be [`VectorPackKind::Expand`].
        kind: VectorPackKind,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Compresses active vector lanes and stores them contiguously to memory.
    VectorPackStore {
        /// Base address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Lane element width in bits.
        element_bits: u32,
        /// Packing direction. Must be [`VectorPackKind::Compress`].
        kind: VectorPackKind,
    },

    /// Clears vector upper lanes according to target vector aliasing rules.
    VectorZeroUpper {
        /// `true` to clear all vector state; `false` to clear upper lanes.
        all: bool,
    },

    /// Applies a unary operation to vector mask lanes.
    VectorMaskUnary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Vector mask operation kind.
        kind: VectorMaskUnaryKind,
    },

    /// Applies a binary operation to vector mask lanes.
    VectorMaskBinary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left mask operand variable.
        left: SsaVarId,
        /// Right mask operand variable.
        right: SsaVarId,
        /// Vector mask operation kind.
        kind: VectorMaskBinaryKind,
    },

    /// Reduces vector lanes to one scalar result.
    VectorReduce {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Vector or mask operand variable.
        value: SsaVarId,
        /// Reduction operation kind.
        kind: VectorReduceKind,
    },

    /// Extracts lane predicate bits into a scalar integer mask.
    VectorBitmask {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Vector or mask operand variable.
        value: SsaVarId,
        /// Bitmask extraction kind.
        kind: VectorBitmaskKind,
    },

    /// Unconditional jump to a block.
    Jump {
        /// Target block index.
        target: usize,
    },

    /// Conditional branch: if condition is true, go to true_target, else false_target.
    Branch {
        /// Condition operand variable.
        condition: SsaVarId,
        /// Target block for the true branch.
        true_target: usize,
        /// Target block for the false branch.
        false_target: usize,
    },

    /// Compare and branch: if (left cmp right) goto true_target else false_target.
    ///
    /// This represents CIL comparison branch instructions like `beq`, `blt`, `bgt`, etc.
    /// These are combined compare-and-branch operations that don't produce an intermediate
    /// comparison result.
    BranchCmp {
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Comparison predicate.
        cmp: CmpKind,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Target block for the true branch.
        true_target: usize,
        /// Target block for the false branch.
        false_target: usize,
    },

    /// Branch based on condition code flags.
    BranchFlags {
        /// Flags operand variable.
        flags: SsaVarId,
        /// Flag condition predicate.
        condition: FlagCondition,
        /// Target block for the true branch.
        true_target: usize,
        /// Target block for the false branch.
        false_target: usize,
    },

    /// Switch statement: jump to `targets[value]` or default if out of range.
    Switch {
        /// Value operand variable.
        value: SsaVarId,
        /// Switch target block indices.
        targets: Vec<usize>,
        /// Default switch target block.
        default: usize,
    },

    /// Indirect branch through a computed target expression.
    IndirectBranch {
        /// SSA value containing the computed target address.
        target: SsaVarId,
        /// Statically recovered successor block indices, if known.
        resolved_targets: Vec<usize>,
    },

    /// Return from method with optional value.
    Return {
        /// Optional return value variable.
        value: Option<SsaVarId>,
    },

    /// Load instance field: `dest = object.field`
    LoadField {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
    },

    /// Store instance field: `object.field = value`
    StoreField {
        /// Object operand variable.
        object: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
        /// Value operand variable.
        value: SsaVarId,
    },

    /// Load static field: `dest = ClassName.field`
    LoadStaticField {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
    },

    /// Store static field: `ClassName.field = value`
    StoreStaticField {
        /// Field reference metadata.
        field: T::FieldRef,
        /// Value operand variable.
        value: SsaVarId,
    },

    /// Load field address: `dest = &object.field`
    LoadFieldAddr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
    },

    /// Load static field address: `dest = &ClassName.field`
    LoadStaticFieldAddr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
    },

    /// Load array element: `dest = array[index]`
    LoadElement {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Array operand variable.
        array: SsaVarId,
        /// Index operand variable.
        index: SsaVarId,
        /// Element type metadata.
        elem_type: T::Type,
    },

    /// Store array element: `array[index] = value`
    StoreElement {
        /// Array operand variable.
        array: SsaVarId,
        /// Index operand variable.
        index: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Element type metadata.
        elem_type: T::Type,
    },

    /// Load array element address: `dest = &array[index]`
    LoadElementAddr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Array operand variable.
        array: SsaVarId,
        /// Index operand variable.
        index: SsaVarId,
        /// Element type metadata.
        elem_type: T::TypeRef,
    },

    /// Get array length: `dest = array.Length`
    ArrayLength {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Array operand variable.
        array: SsaVarId,
    },

    /// Load through pointer: `dest = *ptr`
    LoadIndirect {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Value type metadata.
        value_type: T::Type,
    },

    /// Store through pointer: `*ptr = value`
    StoreIndirect {
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Value type metadata.
        value_type: T::Type,
    },

    /// Create new object: `dest = new Type(args...)`
    NewObj {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Constructor method reference.
        ctor: T::MethodRef,
        /// Call argument variables.
        args: Vec<SsaVarId>,
    },

    /// Create new array: `dest = new Type[length]`
    NewArr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Element type metadata.
        elem_type: T::TypeRef,
        /// Array length operand variable.
        length: SsaVarId,
    },

    /// Cast object to type (throws if invalid): `dest = (Type)obj`
    CastClass {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Target type metadata.
        target_type: T::TypeRef,
    },

    /// Type check (returns null if invalid): `dest = obj as Type`
    IsInst {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Target type metadata.
        target_type: T::TypeRef,
    },

    /// Box value type: `dest = (object)value`
    Box {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Unbox to pointer: `dest = &((ValueType)obj)`
    Unbox {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Unbox and copy: `dest = (ValueType)obj`
    UnboxAny {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Get size of value type: `dest = sizeof(Type)`
    SizeOf {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Load runtime type token: `dest = typeof(Type).TypeHandle`
    LoadToken {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Runtime token metadata.
        token: T::TypeRef,
    },

    /// Direct method call: `dest = method(args...)`
    Call {
        /// Optional destination SSA variable.
        dest: Option<SsaVarId>,
        /// Method reference metadata.
        method: T::MethodRef,
        /// Call argument variables.
        args: Vec<SsaVarId>,
    },

    /// Virtual method call: `dest = obj.method(args...)`
    CallVirt {
        /// Optional destination SSA variable.
        dest: Option<SsaVarId>,
        /// Method reference metadata.
        method: T::MethodRef,
        /// Call argument variables.
        args: Vec<SsaVarId>,
    },

    /// Indirect call through function pointer: `dest = fptr(args...)`
    CallIndirect {
        /// Optional destination SSA variable.
        dest: Option<SsaVarId>,
        /// Function pointer operand variable.
        fptr: SsaVarId,
        /// Indirect call signature metadata.
        signature: T::SigRef,
        /// Call argument variables.
        args: Vec<SsaVarId>,
    },

    /// Load function pointer: `dest = &method`
    LoadFunctionPtr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Method reference metadata.
        method: T::MethodRef,
    },

    /// Load virtual function pointer: `dest = &obj.method`
    LoadVirtFunctionPtr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Method reference metadata.
        method: T::MethodRef,
    },

    /// Load argument value: `dest = argN`
    LoadArg {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Argument index.
        arg_index: u16,
    },

    /// Load local value: `dest = localN`
    LoadLocal {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Local variable index.
        local_index: u16,
    },

    /// Load argument address: `dest = &argN`
    LoadArgAddr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Argument index.
        arg_index: u16,
    },

    /// Load local address: `dest = &localN`
    LoadLocalAddr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Local variable index.
        local_index: u16,
    },

    /// Copy value (from dup): `dest = src`
    Copy {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Pop value from stack (value is discarded, but we track the use)
    Pop {
        /// Value operand variable.
        value: SsaVarId,
    },

    /// Throw exception: `throw obj`
    Throw {
        /// Exception object variable.
        exception: SsaVarId,
    },

    /// Rethrow current exception (in catch handler)
    Rethrow,

    /// End finally block
    EndFinally,

    /// End filter block with result
    EndFilter {
        /// Filter result variable.
        result: SsaVarId,
    },

    /// Return from interrupt / exception handler
    InterruptReturn,

    /// Unreachable terminator.
    Unreachable,

    /// Leave protected region
    Leave {
        /// Target block index.
        target: usize,
    },

    /// Initialize block of memory to zero
    InitBlk {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Size operand variable.
        size: SsaVarId,
    },

    /// Copy block of memory
    CopyBlk {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Source address operand variable.
        src_addr: SsaVarId,
        /// Size operand variable.
        size: SsaVarId,
    },

    /// Memory fence / barrier
    Fence {
        /// Fence ordering kind.
        kind: FenceKind,
    },

    /// Target-specific native instruction with explicit operands, outputs, and effects.
    /// Opaque native instruction with explicit inputs/outputs, clobbers, and a
    /// conservative effect summary. The payload is boxed (see
    /// [`NativeOpaqueData`]) to keep `SsaOp` small.
    NativeOpaque(Box<NativeOpaqueData>),

    /// Compare-and-swap: `old = *addr; if old == expected { *addr = desired; } return old`
    CmpXchg {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Expected compare-exchange value.
        expected: SsaVarId,
        /// Desired compare-exchange value.
        desired: SsaVarId,
    },

    /// Atomic read-modify-write: `old = *addr; *addr = op(old, value); return old`
    AtomicRmw {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Atomic read-modify-write operation.
        op: AtomicRmwOp,
    },

    /// Native atomic load with explicit ordering, width, and volatility.
    AtomicLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Loaded value type.
        value_type: T::Type,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native atomic store with explicit ordering, width, and volatility.
    AtomicStore {
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Stored value type.
        value_type: T::Type,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native store-conditional with explicit status output and ordering.
    AtomicStoreConditional {
        /// Store status output variable.
        status: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Stored value type.
        value_type: T::Type,
        /// Ordering used when the conditional store succeeds.
        success_ordering: AtomicOrdering,
        /// Ordering used when the conditional store fails.
        failure_ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native atomic pair load with explicit ordering, width, and volatility.
    AtomicPairLoad {
        /// First destination SSA variable.
        first: SsaVarId,
        /// Second destination SSA variable.
        second: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// First loaded value type.
        first_type: T::Type,
        /// Second loaded value type.
        second_type: T::Type,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Total atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native pair store-conditional with one shared status output.
    AtomicPairStoreConditional {
        /// Store status output variable.
        status: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// First value operand variable.
        first_value: SsaVarId,
        /// Second value operand variable.
        second_value: SsaVarId,
        /// First stored value type.
        first_type: T::Type,
        /// Second stored value type.
        second_type: T::Type,
        /// Ordering used when the conditional store succeeds.
        success_ordering: AtomicOrdering,
        /// Ordering used when the conditional store fails.
        failure_ordering: AtomicOrdering,
        /// Total atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native atomic exchange with explicit ordering, width, and volatility.
    AtomicExchange {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native lock-prefixed read-modify-write operation.
    AtomicLockRmw {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Atomic read-modify-write operation.
        op: AtomicRmwOp,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native atomic compare-exchange with optional success-status output.
    AtomicCmpXchg {
        /// Old memory value output variable.
        old: SsaVarId,
        /// Optional compare-exchange success output variable.
        success: Option<SsaVarId>,
        /// Address operand variable.
        addr: SsaVarId,
        /// Expected compare-exchange value.
        expected: SsaVarId,
        /// Desired compare-exchange value.
        desired: SsaVarId,
        /// Ordering used when compare-exchange succeeds.
        success_ordering: AtomicOrdering,
        /// Ordering used when compare-exchange fails.
        failure_ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether compare-exchange may fail spuriously.
        weak: bool,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native atomic pair compare-exchange.
    AtomicPairCmpXchg {
        /// First old memory value output variable.
        old_first: SsaVarId,
        /// Second old memory value output variable.
        old_second: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// First expected compare-exchange value.
        expected_first: SsaVarId,
        /// Second expected compare-exchange value.
        expected_second: SsaVarId,
        /// First desired compare-exchange value.
        desired_first: SsaVarId,
        /// Second desired compare-exchange value.
        desired_second: SsaVarId,
        /// Ordering used when compare-exchange succeeds.
        success_ordering: AtomicOrdering,
        /// Ordering used when compare-exchange fails.
        failure_ordering: AtomicOrdering,
        /// Total atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether compare-exchange may fail spuriously.
        weak: bool,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Initialize object (for value types): `*dest = default(T)`
    InitObj {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Copy object (for value types): `*dest = *src`
    CopyObj {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Source address operand variable.
        src_addr: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Load object (value type copy): `dest = *src`
    LoadObj {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source address operand variable.
        src_addr: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Store object (value type copy): `*dest = value`
    StoreObj {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// No operation (for nop instructions)
    Nop,

    /// Breakpoint trap
    Break,

    /// Check for finite floating point: throws if not finite
    Ckfinite {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
    },

    /// Localloc: allocate stack space
    LocalAlloc {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Size operand variable.
        size: SsaVarId,
    },

    /// Constrained virtual call prefix (affects next callvirt)
    Constrained {
        /// Constrained call type metadata.
        constraint_type: T::TypeRef,
    },

    /// Volatile prefix (next memory access must not be reordered/cached)
    Volatile,

    /// Unaligned prefix (next memory access may be unaligned)
    Unaligned {
        /// Required alignment in bytes.
        alignment: u8,
    },

    /// Tail call prefix (next call is a tail call)
    TailPrefix,

    /// Readonly prefix (next ldelema returns a controlled-mutability managed pointer)
    Readonly,

    /// Phi node: merges values from different predecessors.
    ///
    /// This is placed at the beginning of blocks with multiple predecessors.
    Phi {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Incoming phi operands by predecessor block.
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
            | Self::WideMul { low: dest, .. }
            | Self::Div { dest, .. }
            | Self::Rem { dest, .. }
            | Self::WideDiv { quotient: dest, .. }
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
            | Self::BoolAnd { dest, .. }
            | Self::BoolOr { dest, .. }
            | Self::BoolXor { dest, .. }
            | Self::BoolNot { dest, .. }
            | Self::Conv { dest, .. }
            | Self::Bitcast { dest, .. }
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
            | Self::AtomicLoad { dest, .. }
            | Self::AtomicExchange { dest, .. }
            | Self::AtomicLockRmw { dest, .. }
            | Self::AtomicStoreConditional { status: dest, .. }
            | Self::AtomicPairLoad { first: dest, .. }
            | Self::AtomicPairStoreConditional { status: dest, .. }
            | Self::AtomicPairCmpXchg {
                old_first: dest, ..
            }
            | Self::ReadFlags { dest, .. }
            | Self::VectorUnary { dest, .. }
            | Self::VectorBinary { dest, .. }
            | Self::VectorTernary { dest, .. }
            | Self::VectorPredicatedUnary { dest, .. }
            | Self::VectorPredicatedBinary { dest, .. }
            | Self::VectorPredicatedTernary { dest, .. }
            | Self::VectorCompare { dest, .. }
            | Self::VectorLoad { dest, .. }
            | Self::VectorMaskedLoad { dest, .. }
            | Self::VectorBroadcastLoad { dest, .. }
            | Self::VectorGather { dest, .. }
            | Self::VectorFaultingLoad { dest, .. }
            | Self::VectorExtract { dest, .. }
            | Self::VectorInsert { dest, .. }
            | Self::VectorSplat { dest, .. }
            | Self::VectorShuffle { dest, .. }
            | Self::VectorCast { dest, .. }
            | Self::VectorReinterpret { dest, .. }
            | Self::VectorPack { dest, .. }
            | Self::VectorPackLoad { dest, .. }
            | Self::VectorMaskUnary { dest, .. }
            | Self::VectorMaskBinary { dest, .. }
            | Self::VectorReduce { dest, .. }
            | Self::VectorBitmask { dest, .. } => Some(*dest),

            Self::Call { dest, .. }
            | Self::CallVirt { dest, .. }
            | Self::CallIndirect { dest, .. } => *dest,

            Self::NativeOpaque(data) => data.outputs.first().copied(),
            Self::VectorSegmentLoad { dests: outputs, .. } => outputs.first().copied(),
            Self::AtomicCmpXchg { old, .. } => Some(*old),

            // Operations that don't produce a result
            Self::StoreField { .. }
            | Self::StoreStaticField { .. }
            | Self::StoreElement { .. }
            | Self::StoreIndirect { .. }
            | Self::AtomicStore { .. }
            | Self::FloatCompareFlags { .. }
            | Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::IndirectBranch { .. }
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
            | Self::VectorStore { .. }
            | Self::VectorMaskedStore { .. }
            | Self::VectorScatter { .. }
            | Self::VectorSegmentStore { .. }
            | Self::VectorPackStore { .. }
            | Self::VectorZeroUpper { .. }
            | Self::Unreachable => None,
        }
    }

    /// Returns all variables defined by this operation.
    #[must_use]
    pub fn defs(&self) -> SsaDefs<'_> {
        match self {
            Self::NativeOpaque(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSegmentLoad { dests, .. } => SsaDefs::new(None, None, Some(dests)),
            Self::VectorFaultingLoad { dest, fault, .. } => SsaDefs::new(Some(*dest), *fault, None),
            Self::AtomicCmpXchg { old, success, .. } => SsaDefs::new(Some(*old), *success, None),
            Self::AtomicPairLoad { first, second, .. } => {
                SsaDefs::new(Some(*first), Some(*second), None)
            }
            Self::AtomicPairCmpXchg {
                old_first,
                old_second,
                ..
            } => SsaDefs::new(Some(*old_first), Some(*old_second), None),
            Self::WideMul { low, high, .. } => SsaDefs::new(Some(*low), Some(*high), None),
            Self::WideDiv {
                quotient,
                remainder,
                ..
            } => SsaDefs::new(Some(*quotient), Some(*remainder), None),
            _ => SsaDefs::new(self.dest(), self.flags_dest(), None),
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
            | Self::WideMul { low: dest, .. }
            | Self::Div { dest, .. }
            | Self::Rem { dest, .. }
            | Self::WideDiv { quotient: dest, .. }
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
            | Self::BoolAnd { dest, .. }
            | Self::BoolOr { dest, .. }
            | Self::BoolXor { dest, .. }
            | Self::BoolNot { dest, .. }
            | Self::Conv { dest, .. }
            | Self::Bitcast { dest, .. }
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
            | Self::AtomicLoad { dest, .. }
            | Self::AtomicExchange { dest, .. }
            | Self::AtomicLockRmw { dest, .. }
            | Self::AtomicStoreConditional { status: dest, .. }
            | Self::AtomicPairLoad { first: dest, .. }
            | Self::AtomicPairStoreConditional { status: dest, .. }
            | Self::AtomicPairCmpXchg {
                old_first: dest, ..
            }
            | Self::ReadFlags { dest, .. }
            | Self::VectorUnary { dest, .. }
            | Self::VectorBinary { dest, .. }
            | Self::VectorTernary { dest, .. }
            | Self::VectorPredicatedUnary { dest, .. }
            | Self::VectorPredicatedBinary { dest, .. }
            | Self::VectorPredicatedTernary { dest, .. }
            | Self::VectorCompare { dest, .. }
            | Self::VectorLoad { dest, .. }
            | Self::VectorMaskedLoad { dest, .. }
            | Self::VectorBroadcastLoad { dest, .. }
            | Self::VectorGather { dest, .. }
            | Self::VectorFaultingLoad { dest, .. }
            | Self::VectorExtract { dest, .. }
            | Self::VectorInsert { dest, .. }
            | Self::VectorSplat { dest, .. }
            | Self::VectorShuffle { dest, .. }
            | Self::VectorCast { dest, .. }
            | Self::VectorReinterpret { dest, .. }
            | Self::VectorPack { dest, .. }
            | Self::VectorPackLoad { dest, .. }
            | Self::VectorMaskUnary { dest, .. }
            | Self::VectorMaskBinary { dest, .. }
            | Self::VectorReduce { dest, .. }
            | Self::VectorBitmask { dest, .. } => {
                *dest = new_dest;
                true
            }

            Self::NativeOpaque(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSegmentLoad { dests: outputs, .. } => {
                if let Some(first) = outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::AtomicCmpXchg { old, .. } => {
                *old = new_dest;
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
            | Self::AtomicStore { .. }
            | Self::FloatCompareFlags { .. }
            | Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::IndirectBranch { .. }
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
            | Self::VectorStore { .. }
            | Self::VectorMaskedStore { .. }
            | Self::VectorScatter { .. }
            | Self::VectorSegmentStore { .. }
            | Self::VectorPackStore { .. }
            | Self::VectorZeroUpper { .. }
            | Self::Unreachable => false,
        }
    }

    /// Replaces a definition variable without touching operand uses.
    ///
    /// Returns `true` when a primary destination or secondary flags output was
    /// changed. This is used by SSA renaming, where definitions and uses have
    /// different scoping rules.
    pub fn replace_def(&mut self, old_var: SsaVarId, new_var: SsaVarId) -> bool {
        let mut changed = false;

        if self.dest() == Some(old_var) {
            changed |= self.set_dest(new_var);
        }

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
            | Self::Shr { flags, .. }
                if *flags == Some(old_var) =>
            {
                *flags = Some(new_var);
                changed = true;
            }
            Self::FloatCompareFlags { flags, .. } if *flags == old_var => {
                *flags = new_var;
                changed = true;
            }
            _ => {}
        }

        if let Self::AtomicCmpXchg { success, .. } = self {
            if *success == Some(old_var) {
                *success = Some(new_var);
                changed = true;
            }
        }

        match self {
            Self::AtomicPairLoad { second, .. }
            | Self::AtomicPairCmpXchg {
                old_second: second, ..
            } if *second == old_var => {
                *second = new_var;
                changed = true;
            }
            Self::WideMul { high, .. } if *high == old_var => {
                *high = new_var;
                changed = true;
            }
            Self::WideDiv { remainder, .. } if *remainder == old_var => {
                *remainder = new_var;
                changed = true;
            }
            _ => {}
        }

        if let Self::NativeOpaque(data) = self {
            for output in &mut data.outputs {
                if *output == old_var {
                    *output = new_var;
                    changed = true;
                }
            }
        }

        if let Self::VectorSegmentLoad { dests, .. } = self {
            for dest in dests {
                if *dest == old_var {
                    *dest = new_var;
                    changed = true;
                }
            }
        }

        if let Self::VectorFaultingLoad { fault, .. } = self {
            if *fault == Some(old_var) {
                *fault = Some(new_var);
                changed = true;
            }
        }

        changed
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
            Self::FloatCompareFlags { flags, .. } => Some(*flags),
            _ => None,
        }
    }

    /// Calls `f` for every variable used by this operation.
    #[allow(clippy::match_same_arms)] // Kept separate for clarity by operation category
    pub fn for_each_use<F>(&self, mut f: F)
    where
        F: FnMut(SsaVarId),
    {
        match self {
            Self::Const { .. } => {}

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
            | Self::BoolAnd { left, right, .. }
            | Self::BoolOr { left, right, .. }
            | Self::BoolXor { left, right, .. }
            | Self::WideMul { left, right, .. }
            | Self::BranchCmp { left, right, .. }
            | Self::FloatCompareFlags { left, right, .. }
            | Self::VectorBinary { left, right, .. }
            | Self::VectorCompare { left, right, .. }
            | Self::VectorMaskBinary { left, right, .. } => {
                f(*left);
                f(*right);
            }

            Self::WideDiv {
                high, low, divisor, ..
            } => {
                f(*high);
                f(*low);
                f(*divisor);
            }

            Self::BoolNot { value, .. }
            | Self::Branch {
                condition: value, ..
            }
            | Self::Switch { value, .. }
            | Self::LoadField { object: value, .. }
            | Self::StoreStaticField { value, .. }
            | Self::LoadFieldAddr { object: value, .. }
            | Self::ArrayLength { array: value, .. }
            | Self::LoadIndirect { addr: value, .. }
            | Self::AtomicLoad { addr: value, .. }
            | Self::AtomicPairLoad { addr: value, .. }
            | Self::VectorUnary { value, .. }
            | Self::VectorSplat { value, .. }
            | Self::VectorCast { value, .. }
            | Self::VectorReinterpret { value, .. }
            | Self::VectorLoad { addr: value, .. }
            | Self::VectorBroadcastLoad { addr: value, .. }
            | Self::VectorExtract { vector: value, .. }
            | Self::VectorMaskUnary { mask: value, .. }
            | Self::VectorReduce { value, .. }
            | Self::VectorBitmask { value, .. }
            | Self::NewArr { length: value, .. }
            | Self::Box { value, .. }
            | Self::LoadVirtFunctionPtr { object: value, .. }
            | Self::Copy { src: value, .. }
            | Self::Pop { value }
            | Self::Throw { exception: value }
            | Self::EndFilter { result: value }
            | Self::InitObj {
                dest_addr: value, ..
            }
            | Self::LoadObj {
                src_addr: value, ..
            }
            | Self::LocalAlloc { size: value, .. }
            | Self::IndirectBranch { target: value, .. }
            | Self::BranchFlags { flags: value, .. }
            | Self::ReadFlags { flags: value, .. }
            | Self::Neg { operand: value, .. }
            | Self::Not { operand: value, .. }
            | Self::Conv { operand: value, .. }
            | Self::Bitcast { operand: value, .. }
            | Self::Ckfinite { operand: value, .. }
            | Self::BSwap { src: value, .. }
            | Self::BRev { src: value, .. }
            | Self::BitScanForward { src: value, .. }
            | Self::BitScanReverse { src: value, .. }
            | Self::Popcount { src: value, .. }
            | Self::Parity { src: value, .. }
            | Self::CastClass { object: value, .. }
            | Self::IsInst { object: value, .. }
            | Self::Unbox { object: value, .. }
            | Self::UnboxAny { object: value, .. } => f(*value),

            Self::Shl { value, amount, .. }
            | Self::Shr { value, amount, .. }
            | Self::Rol { value, amount, .. }
            | Self::Ror { value, amount, .. }
            | Self::Rcl { value, amount, .. }
            | Self::Rcr { value, amount, .. }
            | Self::StoreField {
                object: value,
                value: amount,
                ..
            }
            | Self::LoadElement {
                array: value,
                index: amount,
                ..
            }
            | Self::LoadElementAddr {
                array: value,
                index: amount,
                ..
            }
            | Self::StoreIndirect {
                addr: value,
                value: amount,
                ..
            }
            | Self::AtomicRmw {
                addr: value,
                value: amount,
                ..
            }
            | Self::AtomicStore {
                addr: value,
                value: amount,
                ..
            }
            | Self::AtomicStoreConditional {
                addr: value,
                value: amount,
                ..
            }
            | Self::AtomicExchange {
                addr: value,
                value: amount,
                ..
            }
            | Self::AtomicLockRmw {
                addr: value,
                value: amount,
                ..
            }
            | Self::VectorStore {
                addr: value,
                value: amount,
                ..
            }
            | Self::VectorInsert {
                vector: value,
                value: amount,
                ..
            }
            | Self::CopyObj {
                dest_addr: value,
                src_addr: amount,
                ..
            }
            | Self::StoreObj {
                dest_addr: value,
                value: amount,
                ..
            } => {
                f(*value);
                f(*amount);
            }

            Self::Select {
                condition,
                true_val,
                false_val,
                ..
            }
            | Self::StoreElement {
                array: condition,
                index: true_val,
                value: false_val,
                ..
            }
            | Self::CmpXchg {
                addr: condition,
                expected: true_val,
                desired: false_val,
                ..
            }
            | Self::AtomicCmpXchg {
                addr: condition,
                expected: true_val,
                desired: false_val,
                ..
            }
            | Self::AtomicPairStoreConditional {
                addr: condition,
                first_value: true_val,
                second_value: false_val,
                ..
            }
            | Self::VectorTernary {
                first: condition,
                second: true_val,
                third: false_val,
                ..
            }
            | Self::VectorMaskedStore {
                addr: condition,
                value: true_val,
                mask: false_val,
                ..
            }
            | Self::VectorScatter {
                base: condition,
                indices: true_val,
                value: false_val,
                mask: _,
                ..
            } => {
                f(*condition);
                f(*true_val);
                f(*false_val);
                if let Self::VectorScatter { mask, .. } = self {
                    f(*mask);
                }
            }

            Self::AtomicPairCmpXchg {
                addr,
                expected_first,
                expected_second,
                desired_first,
                desired_second,
                ..
            } => {
                f(*addr);
                f(*expected_first);
                f(*expected_second);
                f(*desired_first);
                f(*desired_second);
            }

            Self::VectorPredicatedUnary {
                value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorMaskedLoad {
                addr: value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorPack {
                value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorPackLoad {
                addr: value,
                mask,
                passthrough,
                ..
            } => {
                f(*value);
                f(*mask);
                if let Some(passthrough) = passthrough {
                    f(*passthrough);
                }
            }

            Self::VectorPredicatedBinary {
                left,
                right,
                mask,
                passthrough,
                ..
            }
            | Self::VectorGather {
                base: left,
                indices: right,
                mask,
                passthrough,
                ..
            } => {
                f(*left);
                f(*right);
                f(*mask);
                if let Some(passthrough) = passthrough {
                    f(*passthrough);
                }
            }

            Self::VectorPredicatedTernary {
                first,
                second,
                third,
                mask,
                passthrough,
                ..
            } => {
                f(*first);
                f(*second);
                f(*third);
                f(*mask);
                if let Some(passthrough) = passthrough {
                    f(*passthrough);
                }
            }

            Self::VectorFaultingLoad {
                addr,
                mask,
                passthrough,
                ..
            } => {
                f(*addr);
                if let Some(mask) = mask {
                    f(*mask);
                }
                if let Some(passthrough) = passthrough {
                    f(*passthrough);
                }
            }

            Self::VectorSegmentLoad { base, mask, .. } => {
                f(*base);
                if let Some(mask) = mask {
                    f(*mask);
                }
            }

            Self::VectorSegmentStore {
                base, values, mask, ..
            } => {
                f(*base);
                for value in values {
                    f(*value);
                }
                if let Some(mask) = mask {
                    f(*mask);
                }
            }

            Self::VectorShuffle { left, right, .. } => {
                f(*left);
                if let Some(right) = right {
                    f(*right);
                }
            }

            Self::VectorPackStore {
                addr, value, mask, ..
            } => {
                f(*addr);
                f(*value);
                f(*mask);
            }

            Self::NativeOpaque(data) => {
                for input in &data.inputs {
                    f(*input);
                }
            }

            Self::NewObj { args: inputs, .. }
            | Self::Call { args: inputs, .. }
            | Self::CallVirt { args: inputs, .. } => {
                for input in inputs {
                    f(*input);
                }
            }

            Self::Return { value } => {
                if let Some(value) = value {
                    f(*value);
                }
            }

            Self::CallIndirect { fptr, args, .. } => {
                f(*fptr);
                for arg in args {
                    f(*arg);
                }
            }

            Self::InitBlk {
                dest_addr,
                value,
                size,
            }
            | Self::CopyBlk {
                dest_addr,
                src_addr: value,
                size,
            } => {
                f(*dest_addr);
                f(*value);
                f(*size);
            }

            Self::Phi { operands, .. } => {
                for (_, value) in operands {
                    f(*value);
                }
            }

            Self::LoadStaticField { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::SizeOf { .. }
            | Self::LoadToken { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::Jump { .. }
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
            | Self::VectorZeroUpper { .. }
            | Self::Unreachable => {}
        }
    }

    /// Returns all variables used by this operation.
    #[must_use]
    pub fn uses(&self) -> Vec<SsaVarId> {
        let mut uses = Vec::new();
        self.for_each_use(|var| uses.push(var));
        uses
    }

    /// Returns the number of variables used by this operation.
    #[must_use]
    pub fn use_count(&self) -> usize {
        let mut count = 0usize;
        self.for_each_use(|_| count = count.saturating_add(1));
        count
    }

    /// Returns `true` if this operation uses (reads) the given variable.
    ///
    /// This is an allocation-free membership test that short-circuits on the
    /// first match, preferable to `self.uses().contains(&var)` in hot paths.
    #[must_use]
    pub fn uses_var(&self, var: SsaVarId) -> bool {
        let mut found = false;
        self.for_each_use(|used| {
            if used == var {
                found = true;
            }
        });
        found
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
                | Self::IndirectBranch { .. }
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
                | Self::FloatCompareFlags {
                    signaling: true,
                    ..
                }
                | Self::WideDiv { .. }
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
                | Self::AtomicLoad { .. }
                | Self::AtomicStore { .. }
                | Self::AtomicPairLoad { .. }
                | Self::AtomicPairStoreConditional { .. }
                | Self::AtomicExchange { .. }
                | Self::AtomicLockRmw { .. }
                | Self::AtomicStoreConditional { .. }
                | Self::AtomicCmpXchg { .. }
                | Self::AtomicPairCmpXchg { .. }
                | Self::VectorLoad { .. }
                | Self::VectorMaskedLoad { .. }
                | Self::VectorBroadcastLoad { .. }
                | Self::VectorGather { .. }
                | Self::VectorFaultingLoad { .. }
                | Self::VectorSegmentLoad { .. }
                | Self::VectorPackLoad { .. }
                | Self::VectorStore { .. }
                | Self::VectorMaskedStore { .. }
                | Self::VectorScatter { .. }
                | Self::VectorSegmentStore { .. }
                | Self::VectorPackStore { .. }
        ) || matches!(self, Self::NativeOpaque(data) if data.effects.may_throw)
    }

    /// Returns the memory and trapping effects of this operation.
    #[must_use]
    pub const fn effects(&self) -> SsaEffects {
        match self {
            Self::LoadField { .. }
            | Self::LoadStaticField { .. }
            | Self::LoadElement { .. }
            | Self::LoadIndirect { .. }
            | Self::LoadObj { .. }
            | Self::VectorLoad { .. }
            | Self::VectorMaskedLoad { .. }
            | Self::VectorBroadcastLoad { .. }
            | Self::VectorGather { .. }
            | Self::VectorFaultingLoad { .. }
            | Self::VectorSegmentLoad { .. }
            | Self::VectorPackLoad { .. } => SsaEffects::new(SsaEffectKind::Read, self.may_throw())
                .with_trap(TrapClass::MemoryFault),

            Self::StoreField { .. }
            | Self::StoreStaticField { .. }
            | Self::StoreElement { .. }
            | Self::StoreIndirect { .. }
            | Self::StoreObj { .. }
            | Self::InitObj { .. }
            | Self::VectorStore { .. }
            | Self::VectorMaskedStore { .. }
            | Self::VectorScatter { .. }
            | Self::VectorSegmentStore { .. }
            | Self::VectorPackStore { .. } => {
                SsaEffects::new(SsaEffectKind::Write, self.may_throw())
                    .with_trap(TrapClass::MemoryFault)
            }

            Self::CopyBlk { .. } | Self::InitBlk { .. } | Self::CopyObj { .. } => {
                SsaEffects::new(SsaEffectKind::ReadWrite, self.may_throw())
                    .with_trap(TrapClass::MemoryFault)
            }

            Self::CmpXchg { .. } | Self::AtomicRmw { .. } => {
                SsaEffects::new(SsaEffectKind::Atomic, self.may_throw())
                    .atomic_ordering(AtomicOrdering::SeqCst)
                    .with_trap(TrapClass::MemoryFault)
            }

            Self::AtomicLoad {
                ordering, volatile, ..
            }
            | Self::AtomicStore {
                ordering, volatile, ..
            }
            | Self::AtomicPairLoad {
                ordering, volatile, ..
            }
            | Self::AtomicExchange {
                ordering, volatile, ..
            }
            | Self::AtomicLockRmw {
                ordering, volatile, ..
            } => {
                let effects = SsaEffects::new(SsaEffectKind::Atomic, self.may_throw())
                    .atomic_ordering(*ordering)
                    .with_trap(TrapClass::MemoryFault);
                if *volatile {
                    effects.volatile()
                } else {
                    effects
                }
            }

            Self::AtomicCmpXchg {
                success_ordering,
                volatile,
                ..
            }
            | Self::AtomicStoreConditional {
                success_ordering,
                volatile,
                ..
            }
            | Self::AtomicPairStoreConditional {
                success_ordering,
                volatile,
                ..
            }
            | Self::AtomicPairCmpXchg {
                success_ordering,
                volatile,
                ..
            } => {
                let effects = SsaEffects::new(SsaEffectKind::Atomic, self.may_throw())
                    .atomic_ordering(*success_ordering)
                    .with_trap(TrapClass::MemoryFault);
                if *volatile {
                    effects.volatile()
                } else {
                    effects
                }
            }

            Self::Fence { kind } => {
                SsaEffects::new(SsaEffectKind::Fence, false).fence_ordering(kind.ordering())
            }
            Self::NativeOpaque(data) => data.effects,

            Self::FloatCompareFlags { signaling, .. } => {
                if *signaling {
                    SsaEffects::new(SsaEffectKind::Pure, true).with_trap(TrapClass::Unknown)
                } else {
                    SsaEffects::new(SsaEffectKind::Pure, false)
                }
            }

            Self::Call { .. } | Self::CallVirt { .. } | Self::CallIndirect { .. } => {
                SsaEffects::new(SsaEffectKind::Call, true).with_control(ControlEffect::Call)
            }

            Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::BranchFlags { .. }
            | Self::IndirectBranch { .. }
            | Self::Switch { .. } => SsaEffects::new(SsaEffectKind::Opaque, false)
                .with_control(ControlEffect::Terminator),

            Self::Return { .. } | Self::Leave { .. } => {
                SsaEffects::new(SsaEffectKind::Opaque, false).with_control(ControlEffect::Return)
            }

            Self::NewObj { .. }
            | Self::NewArr { .. }
            | Self::CastClass { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. }
            | Self::Box { .. }
            | Self::LocalAlloc { .. } => SsaEffects::new(SsaEffectKind::Opaque, self.may_throw()),

            Self::Throw { .. } | Self::Rethrow => SsaEffects::new(SsaEffectKind::Opaque, true)
                .with_trap(TrapClass::UserThrow)
                .with_control(ControlEffect::Throw),

            Self::EndFinally
            | Self::EndFilter { .. }
            | Self::InterruptReturn
            | Self::Unreachable => SsaEffects::new(SsaEffectKind::Opaque, self.may_throw())
                .with_control(ControlEffect::Terminator),

            Self::Break => SsaEffects::new(SsaEffectKind::Opaque, self.may_throw())
                .with_trap(TrapClass::IllegalInstruction),

            Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly => SsaEffects::new(SsaEffectKind::Opaque, self.may_throw()),

            Self::VectorZeroUpper { .. } => SsaEffects::new(SsaEffectKind::Opaque, false),

            _ => SsaEffects::new(SsaEffectKind::Pure, self.may_throw()),
        }
    }

    /// Returns the high-level operation family for this opcode.
    #[must_use]
    pub const fn class(&self) -> SsaOpClass {
        match self {
            Self::Nop | Self::Phi { .. } | Self::Copy { .. } | Self::Pop { .. } => {
                SsaOpClass::Synthetic
            }

            Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::BoolNot { .. } => SsaOpClass::Boolean,

            Self::ReadFlags { .. } | Self::BranchFlags { .. } => SsaOpClass::Flags,

            Self::VectorUnary { .. }
            | Self::VectorBinary { .. }
            | Self::VectorTernary { .. }
            | Self::VectorPredicatedUnary { .. }
            | Self::VectorPredicatedBinary { .. }
            | Self::VectorPredicatedTernary { .. }
            | Self::VectorCompare { .. }
            | Self::VectorLoad { .. }
            | Self::VectorStore { .. }
            | Self::VectorMaskedLoad { .. }
            | Self::VectorMaskedStore { .. }
            | Self::VectorBroadcastLoad { .. }
            | Self::VectorGather { .. }
            | Self::VectorFaultingLoad { .. }
            | Self::VectorSegmentLoad { .. }
            | Self::VectorScatter { .. }
            | Self::VectorSegmentStore { .. }
            | Self::VectorPackLoad { .. }
            | Self::VectorPackStore { .. }
            | Self::VectorExtract { .. }
            | Self::VectorInsert { .. }
            | Self::VectorShuffle { .. }
            | Self::VectorSplat { .. }
            | Self::VectorCast { .. }
            | Self::VectorReinterpret { .. }
            | Self::VectorPack { .. }
            | Self::VectorZeroUpper { .. }
            | Self::VectorMaskUnary { .. }
            | Self::VectorMaskBinary { .. }
            | Self::VectorReduce { .. }
            | Self::VectorBitmask { .. } => SsaOpClass::Vector,

            Self::LoadField { .. }
            | Self::StoreField { .. }
            | Self::LoadStaticField { .. }
            | Self::StoreStaticField { .. }
            | Self::LoadFieldAddr { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::LoadElement { .. }
            | Self::StoreElement { .. }
            | Self::LoadElementAddr { .. }
            | Self::ArrayLength { .. }
            | Self::LoadIndirect { .. }
            | Self::StoreIndirect { .. }
            | Self::NewObj { .. }
            | Self::NewArr { .. }
            | Self::Box { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. }
            | Self::LocalAlloc { .. }
            | Self::InitBlk { .. }
            | Self::CopyBlk { .. }
            | Self::InitObj { .. }
            | Self::CopyObj { .. }
            | Self::LoadObj { .. }
            | Self::StoreObj { .. }
            | Self::Fence { .. } => SsaOpClass::Memory,

            Self::CmpXchg { .. }
            | Self::AtomicRmw { .. }
            | Self::AtomicLoad { .. }
            | Self::AtomicStore { .. }
            | Self::AtomicPairLoad { .. }
            | Self::AtomicPairStoreConditional { .. }
            | Self::AtomicExchange { .. }
            | Self::AtomicLockRmw { .. }
            | Self::AtomicStoreConditional { .. }
            | Self::AtomicCmpXchg { .. }
            | Self::AtomicPairCmpXchg { .. } => SsaOpClass::Atomic,

            Self::Call { .. }
            | Self::CallVirt { .. }
            | Self::CallIndirect { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::LoadVirtFunctionPtr { .. } => SsaOpClass::Call,

            Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::IndirectBranch { .. }
            | Self::Switch { .. }
            | Self::Return { .. }
            | Self::Throw { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::EndFilter { .. }
            | Self::InterruptReturn
            | Self::Unreachable
            | Self::Leave { .. }
            | Self::Break => SsaOpClass::Control,

            Self::NativeOpaque(_) => SsaOpClass::NativeOpaque,

            Self::WideMul { .. } | Self::WideDiv { .. } => SsaOpClass::WideArithmetic,

            Self::FloatCompareFlags { .. } => SsaOpClass::Scalar,

            Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly => SsaOpClass::Prefix,

            Self::Add { flags: Some(_), .. }
            | Self::AddOvf { flags: Some(_), .. }
            | Self::Sub { flags: Some(_), .. }
            | Self::SubOvf { flags: Some(_), .. }
            | Self::Mul { flags: Some(_), .. }
            | Self::MulOvf { flags: Some(_), .. }
            | Self::Div { flags: Some(_), .. }
            | Self::Rem { flags: Some(_), .. }
            | Self::Neg { flags: Some(_), .. }
            | Self::And { flags: Some(_), .. }
            | Self::Or { flags: Some(_), .. }
            | Self::Xor { flags: Some(_), .. }
            | Self::Not { flags: Some(_), .. }
            | Self::Shl { flags: Some(_), .. }
            | Self::Shr { flags: Some(_), .. } => SsaOpClass::Flags,

            Self::Const { .. }
            | Self::Add { .. }
            | Self::AddOvf { .. }
            | Self::Sub { .. }
            | Self::SubOvf { .. }
            | Self::Mul { .. }
            | Self::MulOvf { .. }
            | Self::Div { .. }
            | Self::Rem { .. }
            | Self::Neg { .. }
            | Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Not { .. }
            | Self::Shl { .. }
            | Self::Shr { .. }
            | Self::Ceq { .. }
            | Self::Clt { .. }
            | Self::Cgt { .. }
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
            | Self::Ckfinite { .. }
            | Self::Conv { .. }
            | Self::Bitcast { .. }
            | Self::Select { .. }
            | Self::CastClass { .. }
            | Self::IsInst { .. }
            | Self::SizeOf { .. }
            | Self::LoadToken { .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. } => SsaOpClass::Scalar,
        }
    }

    /// Returns a stable similarity-oriented operation family.
    ///
    /// Unlike [`Self::class`], this groups scalar operations by semantic role
    /// and separates memory reads, writes, read-write operations, calls,
    /// atomics, fences, native opaque side effects, and vector instructions.
    /// Host crates can use this for deterministic feature extraction without
    /// relying on formatted opcode text.
    #[must_use]
    pub const fn similarity_class(&self) -> SsaSimilarityClass {
        match self {
            Self::Nop | Self::Phi { .. } | Self::Copy { .. } | Self::Pop { .. } => {
                SsaSimilarityClass::Synthetic
            }

            Self::Const { .. } | Self::SizeOf { .. } | Self::LoadToken { .. } => {
                SsaSimilarityClass::Constant
            }

            Self::Add { flags: Some(_), .. }
            | Self::AddOvf { flags: Some(_), .. }
            | Self::Sub { flags: Some(_), .. }
            | Self::SubOvf { flags: Some(_), .. }
            | Self::Mul { flags: Some(_), .. }
            | Self::MulOvf { flags: Some(_), .. }
            | Self::Div { flags: Some(_), .. }
            | Self::Rem { flags: Some(_), .. }
            | Self::Neg { flags: Some(_), .. }
            | Self::And { flags: Some(_), .. }
            | Self::Or { flags: Some(_), .. }
            | Self::Xor { flags: Some(_), .. }
            | Self::Not { flags: Some(_), .. }
            | Self::Shl { flags: Some(_), .. }
            | Self::Shr { flags: Some(_), .. }
            | Self::ReadFlags { .. }
            | Self::BranchFlags { .. } => SsaSimilarityClass::Flags,

            Self::Add { .. }
            | Self::AddOvf { .. }
            | Self::Sub { .. }
            | Self::SubOvf { .. }
            | Self::Mul { .. }
            | Self::MulOvf { .. }
            | Self::Div { .. }
            | Self::Rem { .. }
            | Self::FloatCompareFlags { .. }
            | Self::Neg { .. } => SsaSimilarityClass::Arithmetic,

            Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Not { .. }
            | Self::BSwap { .. }
            | Self::BRev { .. }
            | Self::BitScanForward { .. }
            | Self::BitScanReverse { .. }
            | Self::Popcount { .. }
            | Self::Parity { .. } => SsaSimilarityClass::Bitwise,

            Self::Shl { .. }
            | Self::Shr { .. }
            | Self::Rol { .. }
            | Self::Ror { .. }
            | Self::Rcl { .. }
            | Self::Rcr { .. } => SsaSimilarityClass::ShiftRotate,

            Self::Ceq { .. } | Self::Clt { .. } | Self::Cgt { .. } => SsaSimilarityClass::Compare,

            Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::BoolNot { .. } => SsaSimilarityClass::Boolean,

            Self::Select { .. } => SsaSimilarityClass::Select,

            Self::Conv { .. }
            | Self::Bitcast { .. }
            | Self::Ckfinite { .. }
            | Self::CastClass { .. }
            | Self::IsInst { .. }
            | Self::Box { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. } => SsaSimilarityClass::Conversion,

            Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::LoadVirtFunctionPtr { .. } => SsaSimilarityClass::TypeFlow,

            Self::LoadField { .. }
            | Self::LoadStaticField { .. }
            | Self::LoadFieldAddr { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::LoadElement { .. }
            | Self::LoadElementAddr { .. }
            | Self::ArrayLength { .. }
            | Self::LoadIndirect { .. }
            | Self::LoadObj { .. }
            | Self::VectorLoad { .. }
            | Self::VectorMaskedLoad { .. }
            | Self::VectorBroadcastLoad { .. }
            | Self::VectorGather { .. }
            | Self::VectorFaultingLoad { .. }
            | Self::VectorSegmentLoad { .. }
            | Self::VectorPackLoad { .. } => SsaSimilarityClass::MemoryRead,

            Self::StoreField { .. }
            | Self::StoreStaticField { .. }
            | Self::StoreElement { .. }
            | Self::StoreIndirect { .. }
            | Self::StoreObj { .. }
            | Self::InitObj { .. }
            | Self::VectorStore { .. }
            | Self::VectorMaskedStore { .. }
            | Self::VectorScatter { .. }
            | Self::VectorSegmentStore { .. }
            | Self::VectorPackStore { .. } => SsaSimilarityClass::MemoryWrite,

            Self::CopyBlk { .. } | Self::InitBlk { .. } | Self::CopyObj { .. } => {
                SsaSimilarityClass::MemoryReadWrite
            }

            Self::NewObj { .. } | Self::NewArr { .. } | Self::LocalAlloc { .. } => {
                SsaSimilarityClass::Allocation
            }

            Self::CmpXchg { .. }
            | Self::AtomicRmw { .. }
            | Self::AtomicLoad { .. }
            | Self::AtomicStore { .. }
            | Self::AtomicPairLoad { .. }
            | Self::AtomicPairStoreConditional { .. }
            | Self::AtomicExchange { .. }
            | Self::AtomicLockRmw { .. }
            | Self::AtomicStoreConditional { .. }
            | Self::AtomicCmpXchg { .. }
            | Self::AtomicPairCmpXchg { .. } => SsaSimilarityClass::Atomic,

            Self::Fence { .. } => SsaSimilarityClass::Fence,

            Self::Call { .. } | Self::CallVirt { .. } | Self::CallIndirect { .. } => {
                SsaSimilarityClass::Call
            }

            Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::IndirectBranch { .. }
            | Self::Switch { .. }
            | Self::Return { .. }
            | Self::Throw { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::EndFilter { .. }
            | Self::InterruptReturn
            | Self::Unreachable
            | Self::Leave { .. }
            | Self::Break => SsaSimilarityClass::Control,

            Self::VectorUnary { .. }
            | Self::VectorBinary { .. }
            | Self::VectorTernary { .. }
            | Self::VectorPredicatedUnary { .. }
            | Self::VectorPredicatedBinary { .. }
            | Self::VectorPredicatedTernary { .. }
            | Self::VectorCompare { .. }
            | Self::VectorExtract { .. }
            | Self::VectorInsert { .. }
            | Self::VectorShuffle { .. }
            | Self::VectorSplat { .. }
            | Self::VectorCast { .. }
            | Self::VectorReinterpret { .. }
            | Self::VectorPack { .. }
            | Self::VectorZeroUpper { .. }
            | Self::VectorMaskUnary { .. }
            | Self::VectorMaskBinary { .. }
            | Self::VectorReduce { .. }
            | Self::VectorBitmask { .. } => SsaSimilarityClass::Vector,

            Self::WideMul { .. } | Self::WideDiv { .. } => SsaSimilarityClass::WideArithmetic,

            Self::NativeOpaque(_) => SsaSimilarityClass::NativeOpaque,

            Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly => SsaSimilarityClass::Prefix,
        }
    }

    /// Returns a stable target-generic opcode name.
    ///
    /// The returned name is intentionally independent of operand IDs, target
    /// metadata, and formatting details. Use [`Self::feature_token`] when a
    /// richer canonical feature is needed.
    #[must_use]
    pub const fn opcode_name(&self) -> &'static str {
        match self {
            Self::Const { .. } => "const",
            Self::Add { .. } => "add",
            Self::AddOvf { .. } => "add.ovf",
            Self::Sub { .. } => "sub",
            Self::SubOvf { .. } => "sub.ovf",
            Self::Mul { .. } => "mul",
            Self::MulOvf { .. } => "mul.ovf",
            Self::WideMul { .. } => "wide.mul",
            Self::Div { .. } => "div",
            Self::Rem { .. } => "rem",
            Self::WideDiv { .. } => "wide.div",
            Self::Neg { .. } => "neg",
            Self::And { .. } => "and",
            Self::Or { .. } => "or",
            Self::Xor { .. } => "xor",
            Self::Not { .. } => "not",
            Self::Shl { .. } => "shl",
            Self::Shr { .. } => "shr",
            Self::Rol { .. } => "rol",
            Self::Ror { .. } => "ror",
            Self::Rcl { .. } => "rcl",
            Self::Rcr { .. } => "rcr",
            Self::BSwap { .. } => "bswap",
            Self::BRev { .. } => "brev",
            Self::BitScanForward { .. } => "bsf",
            Self::BitScanReverse { .. } => "bsr",
            Self::Popcount { .. } => "popcount",
            Self::Parity { .. } => "parity",
            Self::Ceq { .. } => "ceq",
            Self::Clt { .. } => "clt",
            Self::Cgt { .. } => "cgt",
            Self::BoolAnd { .. } => "bool.and",
            Self::BoolOr { .. } => "bool.or",
            Self::BoolXor { .. } => "bool.xor",
            Self::BoolNot { .. } => "bool.not",
            Self::Conv { .. } => "conv",
            Self::Bitcast { .. } => "bitcast",
            Self::Select { .. } => "select",
            Self::ReadFlags { .. } => "readflags",
            Self::VectorUnary { .. } => "vector.unary",
            Self::VectorBinary { .. } => "vector.binary",
            Self::VectorTernary { .. } => "vector.ternary",
            Self::VectorPredicatedUnary { .. } => "vector.predicated.unary",
            Self::VectorPredicatedBinary { .. } => "vector.predicated.binary",
            Self::VectorPredicatedTernary { .. } => "vector.predicated.ternary",
            Self::VectorCompare { .. } => "vector.compare",
            Self::VectorLoad { .. } => "vector.load",
            Self::VectorStore { .. } => "vector.store",
            Self::VectorMaskedLoad { .. } => "vector.masked.load",
            Self::VectorMaskedStore { .. } => "vector.masked.store",
            Self::VectorBroadcastLoad { .. } => "vector.broadcast.load",
            Self::VectorGather { .. } => "vector.gather",
            Self::VectorFaultingLoad { .. } => "vector.faulting.load",
            Self::VectorSegmentLoad { .. } => "vector.segment.load",
            Self::VectorScatter { .. } => "vector.scatter",
            Self::VectorSegmentStore { .. } => "vector.segment.store",
            Self::VectorPackLoad { .. } => "vector.pack.load",
            Self::VectorPackStore { .. } => "vector.pack.store",
            Self::VectorExtract { .. } => "vector.extract",
            Self::VectorInsert { .. } => "vector.insert",
            Self::VectorSplat { .. } => "vector.splat",
            Self::VectorShuffle { .. } => "vector.shuffle",
            Self::VectorCast { .. } => "vector.cast",
            Self::VectorReinterpret { .. } => "vector.reinterpret",
            Self::VectorPack { .. } => "vector.pack",
            Self::VectorZeroUpper { .. } => "vector.zero.upper",
            Self::VectorMaskUnary { .. } => "vector.mask.unary",
            Self::VectorMaskBinary { .. } => "vector.mask.binary",
            Self::VectorReduce { .. } => "vector.reduce",
            Self::VectorBitmask { .. } => "vector.bitmask",
            Self::Jump { .. } => "jump",
            Self::Branch { .. } => "branch",
            Self::BranchCmp { .. } => "branch.cmp",
            Self::BranchFlags { .. } => "branch.flags",
            Self::IndirectBranch { .. } => "branch.indirect",
            Self::Switch { .. } => "switch",
            Self::Return { .. } => "return",
            Self::LoadField { .. } => "load.field",
            Self::StoreField { .. } => "store.field",
            Self::LoadStaticField { .. } => "load.static.field",
            Self::StoreStaticField { .. } => "store.static.field",
            Self::LoadFieldAddr { .. } => "load.field.addr",
            Self::LoadStaticFieldAddr { .. } => "load.static.field.addr",
            Self::LoadElement { .. } => "load.element",
            Self::StoreElement { .. } => "store.element",
            Self::LoadElementAddr { .. } => "load.element.addr",
            Self::ArrayLength { .. } => "array.length",
            Self::LoadIndirect { .. } => "load.indirect",
            Self::StoreIndirect { .. } => "store.indirect",
            Self::NewObj { .. } => "new.obj",
            Self::NewArr { .. } => "new.arr",
            Self::CastClass { .. } => "cast.class",
            Self::IsInst { .. } => "is.inst",
            Self::Box { .. } => "box",
            Self::Unbox { .. } => "unbox",
            Self::UnboxAny { .. } => "unbox.any",
            Self::SizeOf { .. } => "sizeof",
            Self::LoadToken { .. } => "load.token",
            Self::Call { .. } => "call",
            Self::CallVirt { .. } => "call.virt",
            Self::CallIndirect { .. } => "call.indirect",
            Self::LoadFunctionPtr { .. } => "load.function.ptr",
            Self::LoadVirtFunctionPtr { .. } => "load.virt.function.ptr",
            Self::LoadArg { .. } => "load.arg",
            Self::LoadLocal { .. } => "load.local",
            Self::LoadArgAddr { .. } => "load.arg.addr",
            Self::LoadLocalAddr { .. } => "load.local.addr",
            Self::Copy { .. } => "copy",
            Self::Pop { .. } => "pop",
            Self::Throw { .. } => "throw",
            Self::Rethrow => "rethrow",
            Self::EndFinally => "end.finally",
            Self::EndFilter { .. } => "end.filter",
            Self::InterruptReturn => "interrupt.return",
            Self::Unreachable => "unreachable",
            Self::Leave { .. } => "leave",
            Self::InitBlk { .. } => "init.blk",
            Self::CopyBlk { .. } => "copy.blk",
            Self::Fence { .. } => "fence",
            Self::FloatCompareFlags { .. } => "float.compare.flags",
            Self::NativeOpaque(_) => "native.opaque",
            Self::CmpXchg { .. } => "cmpxchg",
            Self::AtomicRmw { .. } => "atomic.rmw",
            Self::AtomicLoad { .. } => "atomic.load",
            Self::AtomicStore { .. } => "atomic.store",
            Self::AtomicPairLoad { .. } => "atomic.pair.load",
            Self::AtomicPairStoreConditional { .. } => "atomic.pair.store.conditional",
            Self::AtomicExchange { .. } => "atomic.exchange",
            Self::AtomicLockRmw { .. } => "atomic.lock.rmw",
            Self::AtomicStoreConditional { .. } => "atomic.store.conditional",
            Self::AtomicCmpXchg { .. } => "atomic.cmpxchg",
            Self::AtomicPairCmpXchg { .. } => "atomic.pair.cmpxchg",
            Self::InitObj { .. } => "init.obj",
            Self::CopyObj { .. } => "copy.obj",
            Self::LoadObj { .. } => "load.obj",
            Self::StoreObj { .. } => "store.obj",
            Self::Nop => "nop",
            Self::Break => "break",
            Self::Ckfinite { .. } => "ckfinite",
            Self::LocalAlloc { .. } => "local.alloc",
            Self::Constrained { .. } => "constrained",
            Self::Volatile => "volatile",
            Self::Unaligned { .. } => "unaligned",
            Self::TailPrefix => "tail",
            Self::Readonly => "readonly",
            Self::Phi { .. } => "phi",
        }
    }

    /// Returns a canonical target-generic feature token for this operation.
    ///
    /// The token excludes variable identities and target metadata while
    /// retaining the stable opcode name, operation family, effect kind, arity,
    /// definition count, and trap behavior.
    #[must_use]
    pub fn feature_token(&self) -> SsaFeatureToken {
        SsaFeatureToken {
            opcode: self.opcode_name(),
            op_class: self.class(),
            similarity_class: self.similarity_class(),
            effect_kind: self.effects().kind,
            def_count: self.defs().count(),
            use_count: self.use_count(),
            may_throw: self.may_throw(),
        }
    }

    /// Returns `true` if this operation is pure (has no side effects).
    ///
    /// Pure operations can be eliminated if their result is unused.
    #[must_use]
    pub const fn is_pure(&self) -> bool {
        self.effects().is_pure()
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
            | Self::BoolAnd { left, right, .. }
            | Self::BoolOr { left, right, .. }
            | Self::BoolXor { left, right, .. }
            | Self::WideMul { left, right, .. }
            | Self::BranchCmp { left, right, .. } => {
                replace(left);
                replace(right);
            }
            Self::WideDiv {
                high, low, divisor, ..
            } => {
                replace(high);
                replace(low);
                replace(divisor);
            }
            Self::BoolNot { value, .. } => {
                replace(value);
            }

            // Unary operations and conversion
            Self::Neg { operand, .. }
            | Self::Not { operand, .. }
            | Self::Ckfinite { operand, .. }
            | Self::Conv { operand, .. }
            | Self::Bitcast { operand, .. }
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
            | Self::IndirectBranch { target: value, .. }
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
            Self::AtomicLoad { addr, .. } => {
                replace(addr);
            }
            Self::AtomicStore { addr, value, .. }
            | Self::AtomicStoreConditional { addr, value, .. } => {
                replace(addr);
                replace(value);
            }
            Self::AtomicPairLoad { addr, .. } => {
                replace(addr);
            }
            Self::AtomicPairStoreConditional {
                addr,
                first_value,
                second_value,
                ..
            } => {
                replace(addr);
                replace(first_value);
                replace(second_value);
            }
            Self::AtomicExchange { addr, value, .. } | Self::AtomicLockRmw { addr, value, .. } => {
                replace(addr);
                replace(value);
            }
            Self::AtomicCmpXchg {
                addr,
                expected,
                desired,
                ..
            } => {
                replace(addr);
                replace(expected);
                replace(desired);
            }
            Self::AtomicPairCmpXchg {
                addr,
                expected_first,
                expected_second,
                desired_first,
                desired_second,
                ..
            } => {
                replace(addr);
                replace(expected_first);
                replace(expected_second);
                replace(desired_first);
                replace(desired_second);
            }
            Self::VectorUnary { value, .. }
            | Self::VectorSplat { value, .. }
            | Self::VectorCast { value, .. }
            | Self::VectorReinterpret { value, .. } => {
                replace(value);
            }
            Self::VectorBinary { left, right, .. } | Self::VectorCompare { left, right, .. } => {
                replace(left);
                replace(right);
            }
            Self::VectorTernary {
                first,
                second,
                third,
                ..
            } => {
                replace(first);
                replace(second);
                replace(third);
            }
            Self::VectorPredicatedUnary {
                value,
                mask,
                passthrough,
                ..
            } => {
                replace(value);
                replace(mask);
                if let Some(passthrough) = passthrough {
                    replace(passthrough);
                }
            }
            Self::VectorPredicatedBinary {
                left,
                right,
                mask,
                passthrough,
                ..
            } => {
                replace(left);
                replace(right);
                replace(mask);
                if let Some(passthrough) = passthrough {
                    replace(passthrough);
                }
            }
            Self::VectorPredicatedTernary {
                first,
                second,
                third,
                mask,
                passthrough,
                ..
            } => {
                replace(first);
                replace(second);
                replace(third);
                replace(mask);
                if let Some(passthrough) = passthrough {
                    replace(passthrough);
                }
            }
            Self::VectorLoad { addr, .. } => {
                replace(addr);
            }
            Self::VectorStore { addr, value, .. } => {
                replace(addr);
                replace(value);
            }
            Self::VectorMaskedLoad {
                addr,
                mask,
                passthrough,
                ..
            } => {
                replace(addr);
                replace(mask);
                if let Some(passthrough) = passthrough {
                    replace(passthrough);
                }
            }
            Self::VectorMaskedStore {
                addr, value, mask, ..
            } => {
                replace(addr);
                replace(value);
                replace(mask);
            }
            Self::VectorBroadcastLoad { addr, .. } => {
                replace(addr);
            }
            Self::VectorGather {
                base,
                indices,
                mask,
                passthrough,
                ..
            } => {
                replace(base);
                replace(indices);
                replace(mask);
                if let Some(passthrough) = passthrough {
                    replace(passthrough);
                }
            }
            Self::VectorFaultingLoad {
                addr,
                mask,
                passthrough,
                ..
            } => {
                replace(addr);
                if let Some(mask) = mask {
                    replace(mask);
                }
                if let Some(passthrough) = passthrough {
                    replace(passthrough);
                }
            }
            Self::VectorSegmentLoad { base, mask, .. } => {
                replace(base);
                if let Some(mask) = mask {
                    replace(mask);
                }
            }
            Self::VectorScatter {
                base,
                indices,
                value,
                mask,
                ..
            } => {
                replace(base);
                replace(indices);
                replace(value);
                replace(mask);
            }
            Self::VectorSegmentStore {
                base, values, mask, ..
            } => {
                replace(base);
                for value in values {
                    replace(value);
                }
                if let Some(mask) = mask {
                    replace(mask);
                }
            }
            Self::VectorExtract { vector, .. } => {
                replace(vector);
            }
            Self::VectorInsert { vector, value, .. } => {
                replace(vector);
                replace(value);
            }
            Self::VectorShuffle { left, right, .. } => {
                replace(left);
                if let Some(right) = right {
                    replace(right);
                }
            }
            Self::VectorMaskUnary { mask, .. } => {
                replace(mask);
            }
            Self::VectorMaskBinary { left, right, .. } => {
                replace(left);
                replace(right);
            }
            Self::VectorPack {
                value,
                mask,
                passthrough,
                ..
            } => {
                replace(value);
                replace(mask);
                if let Some(passthrough) = passthrough {
                    replace(passthrough);
                }
            }
            Self::VectorPackLoad {
                addr,
                mask,
                passthrough,
                ..
            } => {
                replace(addr);
                replace(mask);
                if let Some(passthrough) = passthrough {
                    replace(passthrough);
                }
            }
            Self::VectorPackStore {
                addr, value, mask, ..
            } => {
                replace(addr);
                replace(value);
                replace(mask);
            }
            Self::VectorReduce { value, .. } | Self::VectorBitmask { value, .. } => {
                replace(value);
            }
            Self::FloatCompareFlags { left, right, .. } => {
                replace(left);
                replace(right);
            }
            Self::NativeOpaque(data) => {
                for input in &mut data.inputs {
                    replace(input);
                }
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
            | Self::VectorZeroUpper { .. }
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
            Self::IndirectBranch {
                resolved_targets, ..
            } => {
                for target in resolved_targets.iter_mut() {
                    if let Some(new_target) = remap(*target) {
                        *target = new_target;
                    }
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
            Self::IndirectBranch {
                resolved_targets, ..
            } => resolved_targets.clone(),
            // Return, Throw, Rethrow, EndFinally, EndFilter have no successors
            _ => vec![],
        }
    }

    /// Calls `f` for every successor block index of this operation.
    ///
    /// Allocation-free equivalent of iterating [`SsaOp::successors`]; preferred
    /// in CFG-construction and traversal hot paths.
    pub fn for_each_successor<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        match self {
            Self::Jump { target } | Self::Leave { target } => f(*target),
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
                f(*true_target);
                f(*false_target);
            }
            Self::Switch {
                targets, default, ..
            } => {
                for target in targets {
                    f(*target);
                }
                f(*default);
            }
            Self::IndirectBranch {
                resolved_targets, ..
            } => {
                for target in resolved_targets {
                    f(*target);
                }
            }
            // Return, Throw, Rethrow, EndFinally, EndFilter have no successors
            _ => {}
        }
    }

    /// Returns `true` if `block` is a successor of this operation.
    ///
    /// Allocation-free and short-circuiting; preferred over
    /// `self.successors().contains(&block)` in hot paths.
    #[must_use]
    pub fn has_successor(&self, block: usize) -> bool {
        match self {
            Self::Jump { target } | Self::Leave { target } => *target == block,
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
            } => *true_target == block || *false_target == block,
            Self::Switch {
                targets, default, ..
            } => *default == block || targets.contains(&block),
            Self::IndirectBranch {
                resolved_targets, ..
            } => resolved_targets.contains(&block),
            _ => false,
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
            Self::IndirectBranch {
                resolved_targets, ..
            } => {
                let mut changed = false;
                for target in resolved_targets.iter_mut() {
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
            Self::WideMul {
                low,
                high,
                left,
                right,
                unsigned,
            } => Self::WideMul {
                low: r(low),
                high: r(high),
                left: r(left),
                right: r(right),
                unsigned,
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
            Self::FloatCompareFlags {
                flags,
                left,
                right,
                signaling,
            } => Self::FloatCompareFlags {
                flags: r(flags),
                left: r(left),
                right: r(right),
                signaling,
            },
            Self::WideDiv {
                quotient,
                remainder,
                high,
                low,
                divisor,
                unsigned,
            } => Self::WideDiv {
                quotient: r(quotient),
                remainder: r(remainder),
                high: r(high),
                low: r(low),
                divisor: r(divisor),
                unsigned,
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
            Self::BoolAnd { dest, left, right } => Self::BoolAnd {
                dest: r(dest),
                left: r(left),
                right: r(right),
            },
            Self::BoolOr { dest, left, right } => Self::BoolOr {
                dest: r(dest),
                left: r(left),
                right: r(right),
            },
            Self::BoolXor { dest, left, right } => Self::BoolXor {
                dest: r(dest),
                left: r(left),
                right: r(right),
            },
            Self::BoolNot { dest, value } => Self::BoolNot {
                dest: r(dest),
                value: r(value),
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
            Self::Bitcast {
                dest,
                operand,
                target,
            } => Self::Bitcast {
                dest: r(dest),
                operand: r(operand),
                target,
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
            Self::IndirectBranch {
                target,
                resolved_targets,
            } => Self::IndirectBranch {
                target: r(target),
                resolved_targets,
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
            Self::AtomicLoad {
                dest,
                addr,
                value_type,
                ordering,
                width,
                volatile,
            } => Self::AtomicLoad {
                dest: r(dest),
                addr: r(addr),
                value_type,
                ordering,
                width,
                volatile,
            },
            Self::AtomicStore {
                addr,
                value,
                value_type,
                ordering,
                width,
                volatile,
            } => Self::AtomicStore {
                addr: r(addr),
                value: r(value),
                value_type,
                ordering,
                width,
                volatile,
            },
            Self::AtomicStoreConditional {
                status,
                addr,
                value,
                value_type,
                success_ordering,
                failure_ordering,
                width,
                volatile,
            } => Self::AtomicStoreConditional {
                status: r(status),
                addr: r(addr),
                value: r(value),
                value_type,
                success_ordering,
                failure_ordering,
                width,
                volatile,
            },
            Self::AtomicPairLoad {
                first,
                second,
                addr,
                first_type,
                second_type,
                ordering,
                width,
                volatile,
            } => Self::AtomicPairLoad {
                first: r(first),
                second: r(second),
                addr: r(addr),
                first_type,
                second_type,
                ordering,
                width,
                volatile,
            },
            Self::AtomicPairStoreConditional {
                status,
                addr,
                first_value,
                second_value,
                first_type,
                second_type,
                success_ordering,
                failure_ordering,
                width,
                volatile,
            } => Self::AtomicPairStoreConditional {
                status: r(status),
                addr: r(addr),
                first_value: r(first_value),
                second_value: r(second_value),
                first_type,
                second_type,
                success_ordering,
                failure_ordering,
                width,
                volatile,
            },
            Self::AtomicExchange {
                dest,
                addr,
                value,
                ordering,
                width,
                volatile,
            } => Self::AtomicExchange {
                dest: r(dest),
                addr: r(addr),
                value: r(value),
                ordering,
                width,
                volatile,
            },
            Self::AtomicLockRmw {
                dest,
                addr,
                value,
                op,
                ordering,
                width,
                volatile,
            } => Self::AtomicLockRmw {
                dest: r(dest),
                addr: r(addr),
                value: r(value),
                op,
                ordering,
                width,
                volatile,
            },
            Self::AtomicCmpXchg {
                old,
                success,
                addr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
                width,
                weak,
                volatile,
            } => Self::AtomicCmpXchg {
                old: r(old),
                success: success.map(r),
                addr: r(addr),
                expected: r(expected),
                desired: r(desired),
                success_ordering,
                failure_ordering,
                width,
                weak,
                volatile,
            },
            Self::AtomicPairCmpXchg {
                old_first,
                old_second,
                addr,
                expected_first,
                expected_second,
                desired_first,
                desired_second,
                success_ordering,
                failure_ordering,
                width,
                weak,
                volatile,
            } => Self::AtomicPairCmpXchg {
                old_first: r(old_first),
                old_second: r(old_second),
                addr: r(addr),
                expected_first: r(expected_first),
                expected_second: r(expected_second),
                desired_first: r(desired_first),
                desired_second: r(desired_second),
                success_ordering,
                failure_ordering,
                width,
                weak,
                volatile,
            },
            Self::VectorUnary { dest, value, kind } => Self::VectorUnary {
                dest: r(dest),
                value: r(value),
                kind,
            },
            Self::VectorBinary {
                dest,
                left,
                right,
                kind,
            } => Self::VectorBinary {
                dest: r(dest),
                left: r(left),
                right: r(right),
                kind,
            },
            Self::VectorTernary {
                dest,
                first,
                second,
                third,
                kind,
            } => Self::VectorTernary {
                dest: r(dest),
                first: r(first),
                second: r(second),
                third: r(third),
                kind,
            },
            Self::VectorPredicatedUnary {
                dest,
                value,
                mask,
                passthrough,
                kind,
                mode,
            } => Self::VectorPredicatedUnary {
                dest: r(dest),
                value: r(value),
                mask: r(mask),
                passthrough: passthrough.map(&r),
                kind,
                mode,
            },
            Self::VectorPredicatedBinary {
                dest,
                left,
                right,
                mask,
                passthrough,
                kind,
                mode,
            } => Self::VectorPredicatedBinary {
                dest: r(dest),
                left: r(left),
                right: r(right),
                mask: r(mask),
                passthrough: passthrough.map(&r),
                kind,
                mode,
            },
            Self::VectorPredicatedTernary {
                dest,
                first,
                second,
                third,
                mask,
                passthrough,
                kind,
                mode,
            } => Self::VectorPredicatedTernary {
                dest: r(dest),
                first: r(first),
                second: r(second),
                third: r(third),
                mask: r(mask),
                passthrough: passthrough.map(&r),
                kind,
                mode,
            },
            Self::VectorCompare {
                dest,
                left,
                right,
                kind,
                unsigned,
            } => Self::VectorCompare {
                dest: r(dest),
                left: r(left),
                right: r(right),
                kind,
                unsigned,
            },
            Self::VectorLoad {
                dest,
                addr,
                vector_type,
            } => Self::VectorLoad {
                dest: r(dest),
                addr: r(addr),
                vector_type,
            },
            Self::VectorStore {
                addr,
                value,
                vector_type,
            } => Self::VectorStore {
                addr: r(addr),
                value: r(value),
                vector_type,
            },
            Self::VectorMaskedLoad {
                dest,
                addr,
                mask,
                passthrough,
                vector_type,
                mode,
            } => Self::VectorMaskedLoad {
                dest: r(dest),
                addr: r(addr),
                mask: r(mask),
                passthrough: passthrough.map(&r),
                vector_type,
                mode,
            },
            Self::VectorMaskedStore {
                addr,
                value,
                mask,
                vector_type,
            } => Self::VectorMaskedStore {
                addr: r(addr),
                value: r(value),
                mask: r(mask),
                vector_type,
            },
            Self::VectorBroadcastLoad {
                dest,
                addr,
                vector_type,
            } => Self::VectorBroadcastLoad {
                dest: r(dest),
                addr: r(addr),
                vector_type,
            },
            Self::VectorGather {
                dest,
                base,
                indices,
                mask,
                passthrough,
                vector_type,
                mode,
            } => Self::VectorGather {
                dest: r(dest),
                base: r(base),
                indices: r(indices),
                mask: r(mask),
                passthrough: passthrough.map(&r),
                vector_type,
                mode,
            },
            Self::VectorFaultingLoad {
                dest,
                fault,
                addr,
                mask,
                passthrough,
                vector_type,
                fault_mode,
                mask_mode,
            } => Self::VectorFaultingLoad {
                dest: r(dest),
                fault: fault.map(&r),
                addr: r(addr),
                mask: mask.map(&r),
                passthrough: passthrough.map(&r),
                vector_type,
                fault_mode,
                mask_mode,
            },
            Self::VectorSegmentLoad {
                dests,
                base,
                mask,
                vector_type,
                segments,
                layout,
            } => Self::VectorSegmentLoad {
                dests: dests.into_iter().map(&r).collect(),
                base: r(base),
                mask: mask.map(&r),
                vector_type,
                segments,
                layout,
            },
            Self::VectorScatter {
                base,
                indices,
                value,
                mask,
                vector_type,
            } => Self::VectorScatter {
                base: r(base),
                indices: r(indices),
                value: r(value),
                mask: r(mask),
                vector_type,
            },
            Self::VectorSegmentStore {
                base,
                values,
                mask,
                vector_type,
                segments,
                layout,
            } => Self::VectorSegmentStore {
                base: r(base),
                values: values.into_iter().map(&r).collect(),
                mask: mask.map(&r),
                vector_type,
                segments,
                layout,
            },
            Self::VectorExtract { dest, vector, lane } => Self::VectorExtract {
                dest: r(dest),
                vector: r(vector),
                lane,
            },
            Self::VectorInsert {
                dest,
                vector,
                lane,
                value,
            } => Self::VectorInsert {
                dest: r(dest),
                vector: r(vector),
                lane,
                value: r(value),
            },
            Self::VectorSplat {
                dest,
                value,
                vector_type,
            } => Self::VectorSplat {
                dest: r(dest),
                value: r(value),
                vector_type,
            },
            Self::VectorShuffle {
                dest,
                left,
                right,
                mask,
            } => Self::VectorShuffle {
                dest: r(dest),
                left: r(left),
                right: right.map(&r),
                mask,
            },
            Self::VectorCast {
                dest,
                value,
                target_type,
                kind,
            } => Self::VectorCast {
                dest: r(dest),
                value: r(value),
                target_type,
                kind,
            },
            Self::VectorReinterpret {
                dest,
                value,
                target_type,
            } => Self::VectorReinterpret {
                dest: r(dest),
                value: r(value),
                target_type,
            },
            Self::VectorPack {
                dest,
                value,
                mask,
                passthrough,
                vector_type,
                element_bits,
                kind,
                mode,
            } => Self::VectorPack {
                dest: r(dest),
                value: r(value),
                mask: r(mask),
                passthrough: passthrough.map(&r),
                vector_type,
                element_bits,
                kind,
                mode,
            },
            Self::VectorPackLoad {
                dest,
                addr,
                mask,
                passthrough,
                vector_type,
                element_bits,
                kind,
                mode,
            } => Self::VectorPackLoad {
                dest: r(dest),
                addr: r(addr),
                mask: r(mask),
                passthrough: passthrough.map(&r),
                vector_type,
                element_bits,
                kind,
                mode,
            },
            Self::VectorPackStore {
                addr,
                value,
                mask,
                vector_type,
                element_bits,
                kind,
            } => Self::VectorPackStore {
                addr: r(addr),
                value: r(value),
                mask: r(mask),
                vector_type,
                element_bits,
                kind,
            },
            Self::VectorZeroUpper { all } => Self::VectorZeroUpper { all },
            Self::VectorMaskUnary { dest, mask, kind } => Self::VectorMaskUnary {
                dest: r(dest),
                mask: r(mask),
                kind,
            },
            Self::VectorMaskBinary {
                dest,
                left,
                right,
                kind,
            } => Self::VectorMaskBinary {
                dest: r(dest),
                left: r(left),
                right: r(right),
                kind,
            },
            Self::VectorReduce { dest, value, kind } => Self::VectorReduce {
                dest: r(dest),
                value: r(value),
                kind,
            },
            Self::VectorBitmask { dest, value, kind } => Self::VectorBitmask {
                dest: r(dest),
                value: r(value),
                kind,
            },
            Self::NativeOpaque(data) => {
                let NativeOpaqueData {
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                    clobbers,
                    effects,
                } = *data;
                Self::NativeOpaque(Box::new(NativeOpaqueData {
                    mnemonic,
                    metadata,
                    outputs: outputs.into_iter().map(&r).collect(),
                    inputs: inputs.into_iter().map(&r).collect(),
                    clobbers,
                    effects,
                }))
            }

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
            | Self::FloatCompareFlags { .. }
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
            | Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::LoadElement { .. }
            | Self::LoadElementAddr { .. }
            | Self::VectorBinary { .. }
            | Self::VectorCompare { .. }
            | Self::VectorMaskBinary { .. }
            | Self::VectorPredicatedUnary {
                passthrough: None, ..
            }
            | Self::VectorPack {
                passthrough: None, ..
            }
            | Self::VectorPackLoad {
                passthrough: None, ..
            } => (2, 1),

            // Pop 3, push 1 (Select, CmpXchg)
            Self::Select { .. }
            | Self::CmpXchg { .. }
            | Self::VectorTernary { .. }
            | Self::VectorPredicatedUnary {
                passthrough: Some(_),
                ..
            }
            | Self::VectorPredicatedBinary {
                passthrough: None, ..
            }
            | Self::VectorPack {
                passthrough: Some(_),
                ..
            }
            | Self::VectorPackLoad {
                passthrough: Some(_),
                ..
            } => (3, 1),

            Self::VectorPredicatedBinary {
                passthrough: Some(_),
                ..
            }
            | Self::VectorPredicatedTernary {
                passthrough: None, ..
            } => (4, 1),

            Self::VectorPredicatedTernary {
                passthrough: Some(_),
                ..
            } => (5, 1),

            Self::AtomicCmpXchg { success, .. } => {
                (3, 1_u32.saturating_add(u32::from(success.is_some())))
            }
            Self::AtomicPairCmpXchg { .. } => (5, 2),
            Self::WideMul { .. } => (2, 2),
            Self::WideDiv { .. } => (3, 2),

            // AtomicRmw is pop 2, push 1
            Self::AtomicRmw { .. }
            | Self::AtomicExchange { .. }
            | Self::AtomicLockRmw { .. }
            | Self::AtomicStoreConditional { .. } => (2, 1),
            Self::AtomicPairStoreConditional { .. } => (3, 1),

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
            | Self::VectorZeroUpper { .. }
            | Self::Unreachable => (0, 0),

            // Pop 1, push 0 (1, 0)
            Self::Branch { .. }
            | Self::IndirectBranch { .. }
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
            | Self::AtomicStore { .. }
            | Self::StoreObj { .. }
            | Self::CopyObj { .. }
            | Self::VectorStore { .. } => (2, 0),

            Self::VectorMaskedStore { .. } | Self::VectorPackStore { .. } => (3, 0),
            Self::VectorScatter { .. } => (4, 0),
            Self::VectorSegmentStore { values, mask, .. } => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = values.len() as u32;
                (
                    pops.saturating_add(1_u32.saturating_add(u32::from(mask.is_some()))),
                    0,
                )
            }

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
            | Self::Bitcast { .. }
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
            | Self::AtomicLoad { .. }
            | Self::LoadObj { .. }
            | Self::Box { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. }
            | Self::CastClass { .. }
            | Self::IsInst { .. }
            | Self::LoadVirtFunctionPtr { .. }
            | Self::LocalAlloc { .. }
            | Self::ReadFlags { .. }
            | Self::BoolNot { .. }
            | Self::VectorUnary { .. }
            | Self::VectorLoad { .. }
            | Self::VectorBroadcastLoad { .. }
            | Self::VectorExtract { .. }
            | Self::VectorSplat { .. }
            | Self::VectorShuffle { right: None, .. }
            | Self::VectorCast { .. }
            | Self::VectorReinterpret { .. }
            | Self::VectorMaskUnary { .. }
            | Self::VectorReduce { .. }
            | Self::VectorBitmask { .. } => (1, 1),

            Self::AtomicPairLoad { .. } => (1, 2),

            Self::VectorInsert { .. } | Self::VectorShuffle { right: Some(_), .. } => (2, 1),
            Self::VectorMaskedLoad {
                passthrough: None, ..
            } => (2, 1),
            Self::VectorMaskedLoad {
                passthrough: Some(_),
                ..
            } => (3, 1),
            Self::VectorGather {
                passthrough: None, ..
            } => (3, 1),
            Self::VectorGather {
                passthrough: Some(_),
                ..
            } => (4, 1),
            Self::VectorFaultingLoad {
                mask, passthrough, ..
            } => (
                1_u32
                    .saturating_add(u32::from(mask.is_some()))
                    .saturating_add(u32::from(passthrough.is_some())),
                1,
            ),
            Self::VectorSegmentLoad { dests, mask, .. } => {
                #[allow(clippy::cast_possible_truncation)]
                let pushes = dests.len() as u32;
                (1_u32.saturating_add(u32::from(mask.is_some())), pushes)
            }

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
            Self::NativeOpaque(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
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
            Self::Conv { target, .. } | Self::Bitcast { target, .. } => Some(target.clone()),
            // Comparisons produce bool (or whatever the host defines).
            Self::Ceq { .. }
            | Self::Clt { .. }
            | Self::Cgt { .. }
            | Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::BoolNot { .. } => T::comparison_result_type(),
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
            Self::VectorLoad { vector_type, .. }
            | Self::VectorMaskedLoad { vector_type, .. }
            | Self::VectorBroadcastLoad { vector_type, .. }
            | Self::VectorGather { vector_type, .. }
            | Self::VectorFaultingLoad { vector_type, .. }
            | Self::VectorSegmentLoad { vector_type, .. }
            | Self::VectorSplat { vector_type, .. }
            | Self::VectorPack { vector_type, .. }
            | Self::VectorPackLoad { vector_type, .. }
            | Self::VectorCast {
                target_type: vector_type,
                ..
            }
            | Self::VectorReinterpret {
                target_type: vector_type,
                ..
            } => Some(vector_type.clone()),
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
            Self::WideMul {
                low,
                high,
                left,
                right,
                unsigned,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(f, "{low}, {high} = widemul{suffix} {left}, {right}")
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
            Self::FloatCompareFlags {
                flags,
                left,
                right,
                signaling,
            } => {
                let suffix = if *signaling { ".signaling" } else { "" };
                write!(f, "{flags} = fcmp.flags{suffix} {left}, {right}")
            }
            Self::WideDiv {
                quotient,
                remainder,
                high,
                low,
                divisor,
                unsigned,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(
                    f,
                    "{quotient}, {remainder} = widediv{suffix} {high}:{low}, {divisor}"
                )
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
            Self::BoolAnd { dest, left, right } => {
                write!(f, "{dest} = bool.and {left}, {right}")
            }
            Self::BoolOr { dest, left, right } => {
                write!(f, "{dest} = bool.or {left}, {right}")
            }
            Self::BoolXor { dest, left, right } => {
                write!(f, "{dest} = bool.xor {left}, {right}")
            }
            Self::BoolNot { dest, value } => write!(f, "{dest} = bool.not {value}"),
            Self::Conv {
                dest,
                operand,
                target,
                ..
            } => write!(f, "{dest} = conv.{target} {operand}"),
            Self::Bitcast {
                dest,
                operand,
                target,
            } => write!(f, "{dest} = bitcast.{target} {operand}"),
            Self::ReadFlags { dest, flags, mask } => {
                write!(f, "{dest} = readflags {flags}, {mask}")
            }
            Self::VectorUnary { dest, value, kind } => {
                write!(f, "{dest} = vunary.{kind:?} {value}")
            }
            Self::VectorBinary {
                dest,
                left,
                right,
                kind,
            } => write!(f, "{dest} = vbinary.{kind:?} {left}, {right}"),
            Self::VectorTernary {
                dest,
                first,
                second,
                third,
                kind,
            } => write!(f, "{dest} = vternary.{kind:?} {first}, {second}, {third}"),
            Self::VectorPredicatedUnary {
                dest,
                value,
                mask,
                passthrough,
                kind,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vunary.pred.{kind:?}.{mode:?} {value}, {mask}, {passthrough}"
                    )
                } else {
                    write!(f, "{dest} = vunary.pred.{kind:?}.{mode:?} {value}, {mask}")
                }
            }
            Self::VectorPredicatedBinary {
                dest,
                left,
                right,
                mask,
                passthrough,
                kind,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vbinary.pred.{kind:?}.{mode:?} {left}, {right}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vbinary.pred.{kind:?}.{mode:?} {left}, {right}, {mask}"
                    )
                }
            }
            Self::VectorPredicatedTernary {
                dest,
                first,
                second,
                third,
                mask,
                passthrough,
                kind,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vternary.pred.{kind:?}.{mode:?} {first}, {second}, {third}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vternary.pred.{kind:?}.{mode:?} {first}, {second}, {third}, {mask}"
                    )
                }
            }
            Self::VectorCompare {
                dest,
                left,
                right,
                kind,
                unsigned,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(f, "{dest} = vcmp.{kind:?}{suffix} {left}, {right}")
            }
            Self::VectorLoad {
                dest,
                addr,
                vector_type,
            } => write!(f, "{dest} = vload.{vector_type} {addr}"),
            Self::VectorStore {
                addr,
                value,
                vector_type,
            } => write!(f, "vstore.{vector_type} {addr}, {value}"),
            Self::VectorMaskedLoad {
                dest,
                addr,
                mask,
                passthrough,
                vector_type,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vload.masked.{mode:?}.{vector_type} {addr}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vload.masked.{mode:?}.{vector_type} {addr}, {mask}"
                    )
                }
            }
            Self::VectorMaskedStore {
                addr,
                value,
                mask,
                vector_type,
            } => write!(f, "vstore.masked.{vector_type} {addr}, {value}, {mask}"),
            Self::VectorBroadcastLoad {
                dest,
                addr,
                vector_type,
            } => write!(f, "{dest} = vbroadcast.load.{vector_type} {addr}"),
            Self::VectorGather {
                dest,
                base,
                indices,
                mask,
                passthrough,
                vector_type,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vgather.{mode:?}.{vector_type} {base}, {indices}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vgather.{mode:?}.{vector_type} {base}, {indices}, {mask}"
                    )
                }
            }
            Self::VectorFaultingLoad {
                dest,
                fault,
                addr,
                mask,
                passthrough,
                vector_type,
                fault_mode,
                mask_mode,
            } => match (fault, mask, passthrough) {
                (Some(fault), Some(mask), Some(passthrough)) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {mask}, {passthrough}"
                ),
                (Some(fault), Some(mask), None) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {mask}"
                ),
                (Some(fault), None, Some(passthrough)) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {passthrough}"
                ),
                (Some(fault), None, None) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}"
                ),
                (None, Some(mask), Some(passthrough)) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {mask}, {passthrough}"
                ),
                (None, Some(mask), None) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {mask}"
                ),
                (None, None, Some(passthrough)) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {passthrough}"
                ),
                (None, None, None) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}"
                ),
            },
            Self::VectorSegmentLoad {
                dests,
                base,
                mask,
                vector_type,
                segments,
                layout,
            } => {
                if let Some(mask) = mask {
                    write!(
                        f,
                        "{dests:?} = vload.segment.{layout:?}.{segments}.{vector_type} {base}, {mask}"
                    )
                } else {
                    write!(
                        f,
                        "{dests:?} = vload.segment.{layout:?}.{segments}.{vector_type} {base}"
                    )
                }
            }
            Self::VectorScatter {
                base,
                indices,
                value,
                mask,
                vector_type,
            } => write!(
                f,
                "vscatter.{vector_type} {base}, {indices}, {value}, {mask}"
            ),
            Self::VectorSegmentStore {
                base,
                values,
                mask,
                vector_type,
                segments,
                layout,
            } => {
                if let Some(mask) = mask {
                    write!(
                        f,
                        "vstore.segment.{layout:?}.{segments}.{vector_type} {base}, {values:?}, {mask}"
                    )
                } else {
                    write!(
                        f,
                        "vstore.segment.{layout:?}.{segments}.{vector_type} {base}, {values:?}"
                    )
                }
            }
            Self::VectorExtract { dest, vector, lane } => {
                write!(f, "{dest} = vextract {vector}, {lane}")
            }
            Self::VectorInsert {
                dest,
                vector,
                lane,
                value,
            } => write!(f, "{dest} = vinsert {vector}, {lane}, {value}"),
            Self::VectorSplat {
                dest,
                value,
                vector_type,
            } => write!(f, "{dest} = vsplat.{vector_type} {value}"),
            Self::VectorShuffle {
                dest, left, right, ..
            } => {
                if let Some(right) = right {
                    write!(f, "{dest} = vshuffle {left}, {right}")
                } else {
                    write!(f, "{dest} = vshuffle {left}")
                }
            }
            Self::VectorCast {
                dest,
                value,
                target_type,
                kind,
            } => write!(f, "{dest} = vcast.{kind:?}.{target_type} {value}"),
            Self::VectorReinterpret {
                dest,
                value,
                target_type,
            } => write!(f, "{dest} = vreinterpret.{target_type} {value}"),
            Self::VectorPack {
                dest,
                value,
                mask,
                passthrough,
                vector_type,
                element_bits,
                kind,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vpack.{kind:?}.{mode:?}.e{element_bits}.{vector_type} {value}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vpack.{kind:?}.{mode:?}.e{element_bits}.{vector_type} {value}, {mask}"
                    )
                }
            }
            Self::VectorPackLoad {
                dest,
                addr,
                mask,
                passthrough,
                vector_type,
                element_bits,
                kind,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vpack.load.{kind:?}.{mode:?}.e{element_bits}.{vector_type} [{addr}], {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vpack.load.{kind:?}.{mode:?}.e{element_bits}.{vector_type} [{addr}], {mask}"
                    )
                }
            }
            Self::VectorPackStore {
                addr,
                value,
                mask,
                vector_type,
                element_bits,
                kind,
            } => write!(
                f,
                "vpack.store.{kind:?}.e{element_bits}.{vector_type} [{addr}], {value}, {mask}"
            ),
            Self::VectorZeroUpper { all } => {
                let suffix = if *all { "all" } else { "upper" };
                write!(f, "vzero.{suffix}")
            }
            Self::VectorMaskUnary { dest, mask, kind } => {
                write!(f, "{dest} = vmask.unary.{kind:?} {mask}")
            }
            Self::VectorMaskBinary {
                dest,
                left,
                right,
                kind,
            } => write!(f, "{dest} = vmask.binary.{kind:?} {left}, {right}"),
            Self::VectorReduce { dest, value, kind } => {
                write!(f, "{dest} = vreduce.{kind:?} {value}")
            }
            Self::VectorBitmask { dest, value, kind } => {
                write!(f, "{dest} = vbitmask.{kind:?} {value}")
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
            Self::IndirectBranch {
                target,
                resolved_targets,
            } => {
                write!(f, "branch.indirect {target}")?;
                if !resolved_targets.is_empty() {
                    write!(f, " [")?;
                    for (i, t) in resolved_targets.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "B{t}")?;
                    }
                    write!(f, "]")?;
                }
                Ok(())
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
            Self::NativeOpaque(data) => {
                let NativeOpaqueData {
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                    clobbers,
                    effects,
                } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "native.opaque {mnemonic}")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                if let Some(metadata) = metadata {
                    if let Some(architecture) = &metadata.architecture {
                        write!(f, " arch={architecture}")?;
                    }
                    if let Some(address) = metadata.address {
                        write!(f, " addr=0x{address:x}")?;
                    }
                    if !metadata.raw_bytes.is_empty() {
                        write!(f, " bytes={}", metadata.raw_bytes.len())?;
                    }
                }
                if !clobbers.is_empty() {
                    write!(f, " clobbers={}", clobbers.len())?;
                }
                write!(f, " effects={:?}", effects.kind)
            }
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
            Self::AtomicLoad {
                dest,
                addr,
                value_type,
                ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{dest} = atomicload{volatile}.{ordering}.{width} {value_type}, {addr}"
                )
            }
            Self::AtomicStore {
                addr,
                value,
                value_type,
                ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "atomicstore{volatile}.{ordering}.{width} {value_type}, {addr}, {value}"
                )
            }
            Self::AtomicStoreConditional {
                status,
                addr,
                value,
                value_type,
                success_ordering,
                failure_ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{status} = atomicstore.conditional{volatile}.{success_ordering}/{failure_ordering}.{width} {value_type}, {addr}, {value}"
                )
            }
            Self::AtomicPairLoad {
                first,
                second,
                addr,
                first_type,
                second_type,
                ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{first}, {second} = atomicload.pair{volatile}.{ordering}.{width} {first_type}/{second_type}, {addr}"
                )
            }
            Self::AtomicPairStoreConditional {
                status,
                addr,
                first_value,
                second_value,
                first_type,
                second_type,
                success_ordering,
                failure_ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{status} = atomicstore.conditional.pair{volatile}.{success_ordering}/{failure_ordering}.{width} {first_type}/{second_type}, {addr}, {first_value}, {second_value}"
                )
            }
            Self::AtomicExchange {
                dest,
                addr,
                value,
                ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{dest} = atomicxchg{volatile}.{ordering}.{width} {addr}, {value}"
                )
            }
            Self::AtomicLockRmw {
                dest,
                addr,
                value,
                op,
                ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{dest} = lock.atomicrmw{volatile}.{op}.{ordering}.{width} {addr}, {value}"
                )
            }
            Self::AtomicCmpXchg {
                old,
                success,
                addr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
                width,
                weak,
                volatile,
            } => {
                write!(f, "{old}")?;
                if let Some(success) = success {
                    write!(f, ", {success}")?;
                }
                let weak = if *weak { ".weak" } else { "" };
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    " = cmpxchg{weak}{volatile}.{success_ordering}/{failure_ordering}.{width} {addr}, {expected}, {desired}"
                )
            }
            Self::AtomicPairCmpXchg {
                old_first,
                old_second,
                addr,
                expected_first,
                expected_second,
                desired_first,
                desired_second,
                success_ordering,
                failure_ordering,
                width,
                weak,
                volatile,
            } => {
                let weak = if *weak { ".weak" } else { "" };
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{old_first}, {old_second} = cmpxchg.pair{weak}{volatile}.{success_ordering}/{failure_ordering}.{width} {addr}, {expected_first}, {expected_second}, {desired_first}, {desired_second}"
                )
            }
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
    use std::collections::HashMap;

    use super::*;
    use crate::{
        ir::{value::ConstValue, variable::SsaVarId},
        testing::{MockTarget, MockType},
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

    #[test]
    fn native_atomic_ops_report_defs_uses_and_effects() {
        let old = SsaVarId::from_index(0);
        let success = SsaVarId::from_index(1);
        let addr = SsaVarId::from_index(2);
        let expected = SsaVarId::from_index(3);
        let desired = SsaVarId::from_index(4);

        let cmpxchg: SsaOp<MockTarget> = SsaOp::AtomicCmpXchg {
            old,
            success: Some(success),
            addr,
            expected,
            desired,
            success_ordering: AtomicOrdering::SeqCst,
            failure_ordering: AtomicOrdering::Acquire,
            width: AtomicAccessWidth::Bits32,
            weak: false,
            volatile: false,
        };

        assert_eq!(cmpxchg.dest(), Some(old));
        assert_eq!(cmpxchg.defs().collect::<Vec<_>>(), vec![old, success]);
        assert_eq!(cmpxchg.uses(), vec![addr, expected, desired]);
        assert_eq!(cmpxchg.stack_effect(), (3, 2));
        let cmpxchg_effects = cmpxchg.effects();
        assert_eq!(cmpxchg_effects.kind, SsaEffectKind::Atomic);
        assert_eq!(
            cmpxchg_effects.memory_semantics,
            MemoryAccessSemantics::Atomic
        );
        assert_eq!(cmpxchg_effects.ordering, Some(AtomicOrdering::SeqCst));
        assert_eq!(cmpxchg_effects.trap, TrapClass::MemoryFault);
        assert!(!cmpxchg.is_pure());

        let xchg: SsaOp<MockTarget> = SsaOp::AtomicExchange {
            dest: old,
            addr,
            value: desired,
            ordering: AtomicOrdering::AcqRel,
            width: AtomicAccessWidth::Bits32,
            volatile: true,
        };
        assert_eq!(xchg.defs().collect::<Vec<_>>(), vec![old]);
        assert_eq!(xchg.uses(), vec![addr, desired]);
        assert_eq!(xchg.stack_effect(), (2, 1));
        assert_eq!(
            xchg.effects().memory_semantics,
            MemoryAccessSemantics::Atomic
        );
        assert!(xchg.effects().volatile);
        assert_eq!(
            format!("{xchg}"),
            "v0 = atomicxchg.volatile.acqrel.i32 v2, v4"
        );
    }

    #[test]
    fn boolean_ops_are_pure_and_remappable() {
        let dest = SsaVarId::from_index(0);
        let left = SsaVarId::from_index(1);
        let right = SsaVarId::from_index(2);
        let replacement = SsaVarId::from_index(9);
        let op: SsaOp<MockTarget> = SsaOp::BoolAnd { dest, left, right };

        assert_eq!(op.dest(), Some(dest));
        assert_eq!(op.uses(), vec![left, right]);
        assert_eq!(op.stack_effect(), (2, 1));
        assert!(op.is_pure());
        assert_eq!(format!("{op}"), "v0 = bool.and v1, v2");

        let remapped = op.remap_variables(|var| (var == right).then_some(replacement));
        assert_eq!(remapped.uses(), vec![left, replacement]);

        let not: SsaOp<MockTarget> = SsaOp::BoolNot { dest, value: left };
        assert_eq!(not.uses(), vec![left]);
        assert_eq!(not.stack_effect(), (1, 1));
        assert_eq!(format!("{not}"), "v0 = bool.not v1");
    }

    #[test]
    fn wide_arithmetic_ops_report_secondary_defs() {
        let low = SsaVarId::from_index(0);
        let high = SsaVarId::from_index(1);
        let left = SsaVarId::from_index(2);
        let right = SsaVarId::from_index(3);
        let mul: SsaOp<MockTarget> = SsaOp::WideMul {
            low,
            high,
            left,
            right,
            unsigned: true,
        };

        assert_eq!(mul.dest(), Some(low));
        assert_eq!(mul.defs().collect::<Vec<_>>(), vec![low, high]);
        assert_eq!(mul.uses(), vec![left, right]);
        assert_eq!(mul.stack_effect(), (2, 2));
        assert_eq!(format!("{mul}"), "v0, v1 = widemul.un v2, v3");

        let quotient = SsaVarId::from_index(4);
        let remainder = SsaVarId::from_index(5);
        let divisor = SsaVarId::from_index(6);
        let div: SsaOp<MockTarget> = SsaOp::WideDiv {
            quotient,
            remainder,
            high,
            low,
            divisor,
            unsigned: false,
        };

        assert_eq!(div.defs().collect::<Vec<_>>(), vec![quotient, remainder]);
        assert_eq!(div.uses(), vec![high, low, divisor]);
        assert_eq!(div.stack_effect(), (3, 2));
        assert!(div.may_throw());
        assert_eq!(format!("{div}"), "v4, v5 = widediv v1:v0, v6");
    }

    #[test]
    fn expanded_vector_ops_report_uses_effects_and_stack_shape() {
        let dest = SsaVarId::from_index(0);
        let addr = SsaVarId::from_index(1);
        let mask = SsaVarId::from_index(2);
        let passthrough = SsaVarId::from_index(3);
        let indices = SsaVarId::from_index(4);

        let masked_load: SsaOp<MockTarget> = SsaOp::VectorMaskedLoad {
            dest,
            addr,
            mask,
            passthrough: Some(passthrough),
            vector_type: MockType::V4I32,
            mode: VectorMaskMode::Merge,
        };
        assert_eq!(masked_load.uses(), vec![addr, mask, passthrough]);
        assert_eq!(masked_load.stack_effect(), (3, 1));
        assert_eq!(masked_load.effects().kind, SsaEffectKind::Read);

        let scatter: SsaOp<MockTarget> = SsaOp::VectorScatter {
            base: addr,
            indices,
            value: dest,
            mask,
            vector_type: MockType::V4I32,
        };
        assert_eq!(scatter.uses(), vec![addr, indices, dest, mask]);
        assert_eq!(scatter.stack_effect(), (4, 0));
        assert_eq!(scatter.effects().kind, SsaEffectKind::Write);

        let fault = SsaVarId::from_index(5);
        let faulting_load: SsaOp<MockTarget> = SsaOp::VectorFaultingLoad {
            dest,
            fault: Some(fault),
            addr,
            mask: Some(mask),
            passthrough: Some(passthrough),
            vector_type: MockType::V4I32,
            fault_mode: VectorFaultMode::FaultOnlyFirst,
            mask_mode: VectorMaskMode::Merge,
        };
        assert_eq!(faulting_load.defs().collect::<Vec<_>>(), vec![dest, fault]);
        assert_eq!(faulting_load.uses(), vec![addr, mask, passthrough]);
        assert_eq!(faulting_load.stack_effect(), (3, 1));
        assert_eq!(faulting_load.effects().kind, SsaEffectKind::Read);

        let second_dest = SsaVarId::from_index(6);
        let segment_load: SsaOp<MockTarget> = SsaOp::VectorSegmentLoad {
            dests: vec![dest, second_dest],
            base: addr,
            mask: Some(mask),
            vector_type: MockType::V4I32,
            segments: 2,
            layout: VectorSegmentLayout::Interleaved,
        };
        assert_eq!(segment_load.dest(), Some(dest));
        assert_eq!(
            segment_load.defs().collect::<Vec<_>>(),
            vec![dest, second_dest]
        );
        assert_eq!(segment_load.uses(), vec![addr, mask]);
        assert_eq!(segment_load.stack_effect(), (2, 2));
        assert_eq!(segment_load.effects().kind, SsaEffectKind::Read);

        let segment_store: SsaOp<MockTarget> = SsaOp::VectorSegmentStore {
            base: addr,
            values: vec![dest, second_dest],
            mask: Some(mask),
            vector_type: MockType::V4I32,
            segments: 2,
            layout: VectorSegmentLayout::Interleaved,
        };
        assert_eq!(segment_store.dest(), None);
        assert_eq!(segment_store.uses(), vec![addr, dest, second_dest, mask]);
        assert_eq!(segment_store.stack_effect(), (4, 0));
        assert_eq!(segment_store.effects().kind, SsaEffectKind::Write);

        let bitmask: SsaOp<MockTarget> = SsaOp::VectorBitmask {
            dest,
            value: passthrough,
            kind: VectorBitmaskKind::LaneMostSignificantBits,
        };
        assert_eq!(bitmask.uses(), vec![passthrough]);
        assert_eq!(bitmask.stack_effect(), (1, 1));
        assert!(bitmask.is_pure());
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
        assert!(FlagsMask::x86_status().contains(FlagsMask::ADJUST));
        assert!(FlagsMask::x86_status().contains(FlagsMask::OVERFLOW));
        assert_eq!(
            FlagsMask::from_flag_bit(NativeFlagBit::Carry),
            FlagsMask::CARRY
        );
        assert_eq!(FlagsMask::CARRY.union(FlagsMask::ZERO).bits(), 0b1001);
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

    #[test]
    fn flag_condition_required_flags() {
        assert_eq!(FlagCondition::Carry.required_flags(), FlagsMask::CARRY);
        assert_eq!(FlagCondition::NotZero.required_flags(), FlagsMask::ZERO);
        assert_eq!(
            FlagCondition::NotOverflow.required_flags(),
            FlagsMask::OVERFLOW
        );
        assert_eq!(FlagCondition::Positive.required_flags(), FlagsMask::SIGN);
        assert_eq!(FlagCondition::ParityOdd.required_flags(), FlagsMask::PARITY);
    }

    #[test]
    fn flag_producer_semantics_classify_defined_and_undefined_flags() {
        assert!(FlagProducerSemantics::X86Arithmetic
            .defined_mask()
            .contains(FlagsMask::x86_status()));
        assert!(FlagProducerSemantics::X86Logical
            .defined_mask()
            .contains(FlagsMask::CARRY.union(FlagsMask::OVERFLOW)));
        assert!(!FlagProducerSemantics::X86Multiply
            .defined_mask()
            .contains(FlagsMask::ZERO));
        assert!(FlagProducerSemantics::AArch64Arithmetic
            .defined_mask()
            .contains(FlagsMask::SIGN.union(FlagsMask::ZERO)));

        let logical_writes = FlagProducerSemantics::X86Logical.writes();
        assert!(logical_writes.contains(&FlagWrite::undefined(NativeFlagBit::Adjust)));
        assert!(logical_writes.contains(&FlagWrite::cleared(NativeFlagBit::Carry)));
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

    #[test]
    fn effect_summaries_classify_pure_memory_atomic_and_call_ops() {
        let d = SsaVarId::from_index(0);
        let a = SsaVarId::from_index(1);
        let b = SsaVarId::from_index(2);

        assert!(SsaOp::<MockTarget>::Add {
            dest: d,
            left: a,
            right: b,
            flags: None,
        }
        .effects()
        .is_pure());

        let load = SsaOp::<MockTarget>::LoadIndirect {
            dest: d,
            addr: a,
            value_type: MockType::I32,
        }
        .effects();
        assert_eq!(load.kind, SsaEffectKind::Read);
        assert_eq!(load.trap, TrapClass::MemoryFault);
        assert!(load.reads_memory());
        assert!(!load.writes_memory());

        let store = SsaOp::<MockTarget>::StoreIndirect {
            addr: a,
            value: b,
            value_type: MockType::I32,
        }
        .effects();
        assert_eq!(store.kind, SsaEffectKind::Write);
        assert!(store.writes_memory());

        let atomic = SsaOp::<MockTarget>::AtomicRmw {
            dest: d,
            addr: a,
            value: b,
            op: AtomicRmwOp::Add,
        }
        .effects();
        assert_eq!(atomic.kind, SsaEffectKind::Atomic);
        assert_eq!(atomic.memory_semantics, MemoryAccessSemantics::Atomic);
        assert_eq!(atomic.ordering, Some(AtomicOrdering::SeqCst));
        assert!(atomic.reads_memory());
        assert!(atomic.writes_memory());
        assert!(!atomic.removable_when_unused());

        let branch = SsaOp::<MockTarget>::Branch {
            condition: a,
            true_target: 1,
            false_target: 2,
        }
        .effects();
        assert_eq!(branch.control, ControlEffect::Terminator);

        let fence = SsaOp::<MockTarget>::Fence {
            kind: FenceKind::Acquire,
        }
        .effects();
        assert_eq!(fence.memory_semantics, MemoryAccessSemantics::Fence);
        assert_eq!(fence.ordering, Some(AtomicOrdering::Acquire));

        assert_eq!(
            SsaOp::<MockTarget>::Call {
                dest: Some(d),
                method: 1,
                args: vec![a],
            }
            .effects()
            .kind,
            SsaEffectKind::Call
        );
    }

    #[test]
    fn op_class_groups_native_scalar_vector_memory_and_control_ops() {
        let d = SsaVarId::from_index(0);
        let a = SsaVarId::from_index(1);
        let b = SsaVarId::from_index(2);

        assert_eq!(
            SsaOp::<MockTarget>::Add {
                dest: d,
                left: a,
                right: b,
                flags: None,
            }
            .class(),
            SsaOpClass::Scalar
        );
        assert_eq!(
            SsaOp::<MockTarget>::Add {
                dest: d,
                left: a,
                right: b,
                flags: Some(SsaVarId::from_index(3)),
            }
            .class(),
            SsaOpClass::Flags
        );
        assert_eq!(
            SsaOp::<MockTarget>::VectorBinary {
                dest: d,
                left: a,
                right: b,
                kind: VectorBinaryKind::Add,
            }
            .class(),
            SsaOpClass::Vector
        );
        assert_eq!(
            SsaOp::<MockTarget>::AtomicRmw {
                dest: d,
                addr: a,
                value: b,
                op: AtomicRmwOp::Add,
            }
            .class(),
            SsaOpClass::Atomic
        );
        assert_eq!(
            SsaOp::<MockTarget>::Branch {
                condition: a,
                true_target: 1,
                false_target: 2,
            }
            .class(),
            SsaOpClass::Control
        );
        assert_eq!(
            SsaOp::<MockTarget>::NativeOpaque(Box::new(NativeOpaqueData {
                mnemonic: "ud2".to_string(),
                metadata: None,
                outputs: Vec::new(),
                inputs: Vec::new(),
                clobbers: Vec::new(),
                effects: SsaEffects::new(SsaEffectKind::Opaque, true),
            }))
            .class(),
            SsaOpClass::NativeOpaque
        );
    }

    #[test]
    fn similarity_class_groups_feature_extraction_families() {
        let d = SsaVarId::from_index(0);
        let a = SsaVarId::from_index(1);
        let b = SsaVarId::from_index(2);

        assert_eq!(
            SsaOp::<MockTarget>::Const {
                dest: d,
                value: ConstValue::I32(1),
            }
            .similarity_class(),
            SsaSimilarityClass::Constant
        );
        assert_eq!(
            SsaOp::<MockTarget>::Add {
                dest: d,
                left: a,
                right: b,
                flags: None,
            }
            .similarity_class(),
            SsaSimilarityClass::Arithmetic
        );
        assert_eq!(
            SsaOp::<MockTarget>::Add {
                dest: d,
                left: a,
                right: b,
                flags: Some(SsaVarId::from_index(3)),
            }
            .similarity_class(),
            SsaSimilarityClass::Flags
        );
        assert_eq!(
            SsaOp::<MockTarget>::Xor {
                dest: d,
                left: a,
                right: b,
                flags: None,
            }
            .similarity_class(),
            SsaSimilarityClass::Bitwise
        );
        assert_eq!(
            SsaOp::<MockTarget>::Ceq {
                dest: d,
                left: a,
                right: b,
            }
            .similarity_class(),
            SsaSimilarityClass::Compare
        );
        assert_eq!(
            SsaOp::<MockTarget>::VectorFaultingLoad {
                dest: d,
                fault: None,
                addr: a,
                mask: None,
                passthrough: None,
                vector_type: MockType::V4I32,
                fault_mode: VectorFaultMode::Normal,
                mask_mode: VectorMaskMode::Zero,
            }
            .similarity_class(),
            SsaSimilarityClass::MemoryRead
        );
        assert_eq!(
            SsaOp::<MockTarget>::VectorBinary {
                dest: d,
                left: a,
                right: b,
                kind: VectorBinaryKind::Add,
            }
            .similarity_class(),
            SsaSimilarityClass::Vector
        );
        assert_eq!(
            SsaOp::<MockTarget>::AtomicExchange {
                dest: d,
                addr: a,
                value: b,
                ordering: AtomicOrdering::SeqCst,
                width: AtomicAccessWidth::Bits32,
                volatile: false,
            }
            .similarity_class(),
            SsaSimilarityClass::Atomic
        );
        assert_eq!(
            SsaOp::<MockTarget>::Fence {
                kind: FenceKind::SeqCst,
            }
            .similarity_class(),
            SsaSimilarityClass::Fence
        );
        assert_eq!(
            SsaOp::<MockTarget>::NativeOpaque(Box::new(NativeOpaqueData {
                mnemonic: "ud2".to_string(),
                metadata: None,
                outputs: Vec::new(),
                inputs: Vec::new(),
                clobbers: Vec::new(),
                effects: SsaEffects::new(SsaEffectKind::Opaque, true),
            }))
            .similarity_class(),
            SsaSimilarityClass::NativeOpaque
        );
    }

    #[test]
    fn feature_token_serializes_stable_target_generic_shape() {
        let d = SsaVarId::from_index(0);
        let a = SsaVarId::from_index(1);
        let b = SsaVarId::from_index(2);

        let op: SsaOp<MockTarget> = SsaOp::AtomicCmpXchg {
            old: d,
            success: Some(SsaVarId::from_index(3)),
            addr: a,
            expected: b,
            desired: d,
            success_ordering: AtomicOrdering::SeqCst,
            failure_ordering: AtomicOrdering::Acquire,
            width: AtomicAccessWidth::Bits32,
            weak: false,
            volatile: true,
        };
        let token = op.feature_token();

        assert_eq!(token.opcode, "atomic.cmpxchg");
        assert_eq!(token.op_class, SsaOpClass::Atomic);
        assert_eq!(token.similarity_class, SsaSimilarityClass::Atomic);
        assert_eq!(token.effect_kind, SsaEffectKind::Atomic);
        assert_eq!(token.def_count, 2);
        assert_eq!(token.use_count, 3);
        assert!(token.may_throw);
        assert_eq!(
            token.to_string(),
            "op=atomic.cmpxchg;class=Atomic;sim=Atomic;effect=Atomic;defs=2;uses=3;throw=true"
        );
    }

    #[test]
    fn native_register_aliases_track_subregister_overlap() {
        let rax = NativeRegister::new("x86_64", "gpr", "rax", "rax", 0, 64).unwrap();
        let eax = NativeRegister::new("x86_64", "gpr", "rax", "eax", 0, 32).unwrap();
        let ah = NativeRegister::new("x86_64", "gpr", "rax", "ah", 8, 8).unwrap();
        let rbx = NativeRegister::new("x86_64", "gpr", "rbx", "rbx", 0, 64).unwrap();
        let q0 = NativeRegister::new("aarch64", "simd", "v0", "q0", 0, 128).unwrap();

        assert!(rax.aliases(&eax));
        assert!(eax.aliases(&ah));
        assert!(!rax.aliases(&rbx));
        assert!(!rax.aliases(&q0));
        assert!(NativeRegister::new("x86_64", "gpr", "rax", "al", 0, 0).is_none());
    }

    #[test]
    fn native_state_accesses_classify_implicit_machine_state() {
        let rflags = NativeStateAccess::implicit_read_write(
            NativeStateLocation::Flags("rflags".to_string()),
            Some(64),
        )
        .unwrap();
        assert!(rflags.reads());
        assert!(rflags.writes());
        assert!(rflags.implicit);

        let vl = NativeStateAccess::implicit_read(NativeStateLocation::VectorLength, None).unwrap();
        assert!(vl.reads());
        assert!(!vl.writes());
        assert!(
            NativeStateAccess::implicit_write(NativeStateLocation::StackPointer, Some(0)).is_none()
        );
    }

    #[test]
    fn native_clobbers_expose_structured_machine_state_categories() {
        let rax = NativeRegister::new("x86_64", "gpr", "rax", "rax", 0, 64).unwrap();
        let reg = NativeClobber::MachineState(
            NativeStateAccess::implicit_read_write(NativeStateLocation::Register(rax), Some(64))
                .unwrap(),
        );
        let flags = NativeClobber::Flags("eflags".to_string());
        let memory = NativeClobber::MachineState(
            NativeStateAccess::implicit_write(NativeStateLocation::Memory("io".to_string()), None)
                .unwrap(),
        );

        assert!(reg.touches_registers());
        assert!(!reg.touches_memory());
        assert!(flags.touches_flags());
        assert!(memory.touches_memory());
    }

    #[test]
    fn native_opaque_tracks_outputs_inputs_and_effects() {
        let out0 = SsaVarId::from_index(0);
        let out1 = SsaVarId::from_index(1);
        let in0 = SsaVarId::from_index(2);
        let in1 = SsaVarId::from_index(3);
        let op: SsaOp<MockTarget> = SsaOp::NativeOpaque(Box::new(NativeOpaqueData {
            mnemonic: "mulx".to_string(),
            metadata: Some(NativeInstructionMetadata::new(
                Some("x86_64".to_string()),
                Some(0x1000),
                vec![0xc4, 0xe2, 0xfb, 0xf6],
            )),
            outputs: vec![out0, out1],
            inputs: vec![in0, in1],
            clobbers: vec![NativeClobber::Flags("eflags".to_string())],
            effects: SsaEffects::new(SsaEffectKind::ReadWrite, true),
        }));

        assert_eq!(op.dest(), Some(out0));
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![out0, out1]);
        assert_eq!(op.uses(), vec![in0, in1]);
        assert_eq!(op.stack_effect(), (2, 2));
        assert_eq!(
            op.effects(),
            SsaEffects::new(SsaEffectKind::ReadWrite, true)
        );
        assert!(op.may_throw());
        assert!(!op.is_pure());
    }

    #[test]
    fn native_opaque_rewrites_defs_and_uses_separately() {
        let out0 = SsaVarId::from_index(0);
        let out1 = SsaVarId::from_index(1);
        let new_out = SsaVarId::from_index(9);
        let input = SsaVarId::from_index(2);
        let new_input = SsaVarId::from_index(10);
        let mut op: SsaOp<MockTarget> = SsaOp::NativeOpaque(Box::new(NativeOpaqueData {
            mnemonic: "opaque".to_string(),
            metadata: None,
            outputs: vec![out0, out1],
            inputs: vec![input],
            clobbers: Vec::new(),
            effects: SsaEffects::pure(),
        }));

        assert!(op.replace_def(out1, new_out));
        assert_eq!(op.replace_uses(input, new_input), 1);
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![out0, new_out]);
        assert_eq!(op.uses(), vec![new_input]);

        let remapped = op.remap_variables(|var| {
            if var == out0 {
                Some(SsaVarId::from_index(20))
            } else if var == new_input {
                Some(SsaVarId::from_index(30))
            } else {
                None
            }
        });
        assert_eq!(
            remapped.defs().collect::<Vec<_>>(),
            vec![SsaVarId::from_index(20), new_out]
        );
        assert_eq!(remapped.uses(), vec![SsaVarId::from_index(30)]);
    }
}

#[cfg(test)]
mod size_guards {
    //! Regression guards on the in-memory size of the core IR value types.
    //!
    //! These bounds were established after boxing the rare fat `SsaOp` /
    //! `ConstValue` variants (`NativeOpaque`, wide atomics, decrypted
    //! string/array/vector payloads) and giving `SsaVarId` a niche so
    //! `Option<SsaVarId>` is 4 bytes. Every `SsaBlock` stores a
    //! `Vec<SsaInstruction>`, so a regrowth here multiplies across an entire
    //! function. If adding a variant trips one of these, prefer boxing the new
    //! variant's payload over relaxing the bound.
    use super::*;
    use crate::{
        ir::{instruction::SsaInstruction, value::ConstValue},
        testing::MockTarget,
    };

    #[test]
    fn core_ir_types_stay_compact() {
        assert!(
            std::mem::size_of::<Option<SsaVarId>>() <= 4,
            "Option<SsaVarId> grew to {} bytes; SsaVarId lost its niche",
            std::mem::size_of::<Option<SsaVarId>>()
        );
        assert!(
            std::mem::size_of::<ConstValue<MockTarget>>() <= 24,
            "ConstValue grew to {} bytes; box the new heap-bearing arm",
            std::mem::size_of::<ConstValue<MockTarget>>()
        );
        assert!(
            std::mem::size_of::<SsaOp<MockTarget>>() <= 40,
            "SsaOp grew to {} bytes; box the new fat variant's payload",
            std::mem::size_of::<SsaOp<MockTarget>>()
        );
        assert!(
            std::mem::size_of::<SsaInstruction<MockTarget>>() <= 48,
            "SsaInstruction grew to {} bytes",
            std::mem::size_of::<SsaInstruction<MockTarget>>()
        );
    }
}
