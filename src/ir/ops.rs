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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Operand signedness interpretation carried by an operation's payload.
///
/// Returned by [`SsaOp::arith_signedness`] for operations whose semantics
/// depend on whether operands are treated as signed or unsigned (division,
/// remainder, arithmetic-vs-logical shift, ordered comparison, conversion).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Signedness {
    /// Operands are interpreted as signed values.
    Signed,
    /// Operands are interpreted as unsigned values.
    Unsigned,
}

impl Signedness {
    /// Converts the payload `unsigned` flag carried by [`SsaOp`] variants.
    #[must_use]
    pub const fn from_unsigned(unsigned: bool) -> Self {
        if unsigned {
            Self::Unsigned
        } else {
            Self::Signed
        }
    }

    /// Returns `true` when operands are interpreted as unsigned values.
    #[must_use]
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Self::Unsigned)
    }
}

impl fmt::Display for Signedness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Signed => write!(f, "signed"),
            Self::Unsigned => write!(f, "unsigned"),
        }
    }
}

/// Direct single-address memory access extracted from an operation's payload.
///
/// Returned by [`SsaOp::memory_effect`] for operations that read or write
/// memory through exactly one address variable (indirect loads/stores, atomic
/// accesses, vector loads/stores, block/object initialization). Operations
/// with two address operands (`CopyBlk`, `CopyObj`) or structured addressing
/// payloads (field, element, gather/scatter, segment) are not representable
/// here and return `None`; hosts read those payloads directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryEffect<'a, T: Target> {
    /// SSA variable holding the accessed address.
    pub addr: SsaVarId,
    /// Whether the operation reads memory at `addr`.
    pub reads: bool,
    /// Whether the operation writes memory at `addr`.
    pub writes: bool,
    /// Accessed value type when the payload carries one.
    pub value_type: Option<&'a T::Type>,
}

/// Memory fence / barrier kind for atomic ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Bit-clear: `*p &= !value` (AArch64 `ldclr`/`stclr`).
    AndNot,
    /// Min (unsigned) (AArch64 `ldumin`/`stumin`).
    MinU,
    /// Max (unsigned) (AArch64 `ldumax`/`stumax`).
    MaxU,
}

/// Role of an SSA variable operand within an operation.
///
/// Emitted by [`SsaOp::visit_operands`] / [`SsaOp::visit_operands_mut`] for
/// every variable an operation touches: definitions first (the primary
/// destination, then secondary and flag outputs), then uses in payload order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OperandRole {
    /// Value defined by the operation. The first `Def` visited is the
    /// primary destination.
    Def,
    /// Condition-flags bundle defined alongside the primary result.
    FlagsDef,
    /// Value read by the operation.
    Use,
}

/// High-level operation family used by verifiers, lifters, and pass scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Recognized native intrinsic with a structured identity.
    NativeIntrinsic,
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Identity of a recognized native intrinsic — a native operation with no
/// primitive closed form but faithful, fully-modellable dataflow (its inputs,
/// outputs and machine-state effects are known even though the computation
/// itself is hardware-defined).
///
/// Unlike [`SsaOp::NativeOpaque`] (the genuine "unmodelled" escape hatch), a
/// [`SsaOp::NativeIntrinsic`] carries a stable identity so passes, similarity
/// and rendering can reason about *which* native op it is. New native ops are
/// added as a single [`NativeIntrinsicId`] arm rather than a new [`SsaOp`]
/// variant; the [`NativeIntrinsicData::mnemonic`] field carries the raw mnemonic for ops not
/// yet promoted to a dedicated id.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NativeIntrinsicId {
    /// x86 `cpuid` — processor identification (reads EAX/ECX, writes EAX/EBX/ECX/EDX).
    Cpuid,
    /// x86 `rdtsc` — read time-stamp counter into EDX:EAX.
    Rdtsc,
    /// x86 `rdtscp` — `rdtsc` plus TSC_AUX into ECX.
    Rdtscp,
    /// x86 `rdmsr` — read model-specific register.
    Rdmsr,
    /// x86 `wrmsr` — write model-specific register.
    Wrmsr,
    /// x86 `rdpmc` — read performance-monitoring counter.
    Rdpmc,
    /// x86 `xgetbv` — read extended control register.
    Xgetbv,
    /// x86 `xsetbv` — write extended control register.
    Xsetbv,
    /// System-call entry (`syscall`/`sysenter`/`svc`/`ecall`).
    SystemCall,
    /// System-call return (`sysret`/`sysexit`).
    SystemReturn,
    /// BMI2 `pdep` — parallel bit deposit under a mask.
    BitDeposit,
    /// BMI2 `pext` — parallel bit extract under a mask.
    BitExtract,
    /// Hardware CRC32 accumulation (`crc32`).
    Crc32,
    /// Hardware random number (`rdrand`).
    RandomNumber,
    /// Hardware random seed (`rdseed`).
    RandomSeed,
    /// AArch64 pointer authentication, carrying which PAC sub-operation ran
    /// (`pac*` sign, `aut*` authenticate, `xpac*` strip, `pacga` generic MAC).
    PointerAuth(PacKind),
    /// Virtualization / hypervisor op (x86 `vmcall`/`vmlaunch`/`vmread`/`vmwrite`,
    /// ARM `hvc`).
    Hypervisor,
    /// Privileged machine-state op with no value result (`hlt`/`cli`/`sti`,
    /// `lgdt`/`lidt`/`swapgs`/`invlpg`/`wbinvd`, ARM `cps`).
    Privileged,
    /// Control / status / system-register access (`mrs`/`msr`, RISC-V `csrr*`,
    /// MIPS `mfc0`/`rdhwr`).
    ControlRegister,
}

/// PAC sub-operation carried by [`NativeIntrinsicId::PointerAuth`].
///
/// ARMv8.3 pointer authentication signs a pointer with a cryptographic MAC in
/// its unused high bits, authenticates (validates) it before use, strips the
/// MAC back to the raw pointer, or computes a generic MAC into a register.
/// Distinguishing these is required to faithfully reconstruct the native
/// instruction role; without it every PAC op collapses to "sign".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PacKind {
    /// Sign a pointer, inserting the authentication code (`pacia`/`pacib`/
    /// `pacda`/`pacdb`/`paciasp`/`pacibsp`/`paciaz`/…).
    Sign,
    /// Authenticate (validate) a signed pointer, faulting/poisoning on tamper
    /// (`autia`/`autib`/`autda`/`autdb`/`autiasp`/`autibsp`/…).
    Authenticate,
    /// Strip the authentication code, yielding the raw pointer (`xpaci`/
    /// `xpacd`/`xpaclri`).
    Strip,
    /// Compute a generic authentication code into a register (`pacga`).
    GenericMac,
}

impl fmt::Display for PacKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Sign => "sign",
            Self::Authenticate => "auth",
            Self::Strip => "strip",
            Self::GenericMac => "genmac",
        };
        f.write_str(name)
    }
}

/// Boxed payload shared by the kind-tagged vector/native compute ops whose shape
/// is exactly a structured `kind` plus explicit SSA out/in lists. Each `SsaOp`
/// variant keeps its own identity (opcode name, effects, display) through the
/// `K` it instantiates; this struct only unifies the otherwise-duplicated layout
/// that previously lived in a dozen byte-identical `*Data` structs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KindedVecData<K> {
    /// Structured identity of the operation.
    pub kind: K,
    /// Explicit SSA outputs defined by the operation.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the operation.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload shared by vector ops parameterized by a single 8-bit immediate
/// (truth table / block-offset / category selector) plus SSA out/in lists.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VecImm8Data {
    /// The 8-bit immediate selecting the operation's mode.
    pub imm8: u8,
    /// Explicit SSA outputs defined by the operation.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the operation.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload shared by the kind-tagged native ops that also carry a native
/// mnemonic, optional source metadata, and architectural clobbers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NativeKindedData<K> {
    /// Structured identity of the operation.
    pub kind: K,
    /// Human-readable native mnemonic, for display / provenance only.
    pub mnemonic: String,
    /// Original native instruction metadata when known.
    pub metadata: Option<NativeInstructionMetadata>,
    /// Explicit SSA outputs defined by the operation.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the operation.
    pub inputs: Vec<SsaVarId>,
    /// Architectural state the operation clobbers.
    pub clobbers: Vec<NativeClobber>,
}

/// Hardware floating-point transcendental / residue function
/// ([`SsaOp::FpTranscendental`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum TranscendentalKind {
    /// Sine (`fsin`).
    Sin,
    /// Cosine (`fcos`).
    Cos,
    /// Sine and cosine together (`fsincos` — two results).
    SinCos,
    /// Tangent, pushing the `1.0` constant (`fptan` — two results).
    Tan,
    /// Arctangent of `arg1/arg0` (`fpatan` — two operands).
    Atan,
    /// `2^x - 1` (`f2xm1`).
    Exp2m1,
    /// `arg1 * log2(arg0)` (`fyl2x` — two operands).
    Ylog2,
    /// `arg1 * log2(arg0 + 1)` (`fyl2xp1` — two operands).
    Ylog2p1,
    /// Partial remainder (`fprem` — two operands).
    Rem,
    /// IEEE partial remainder (`fprem1` — two operands).
    Rem1,
    /// Scale by power of two (`fscale` — two operands).
    Scale,
    /// Extract exponent and mantissa (`fxtract` — two results).
    Extract,
}

/// Floating-point unit control / state operation ([`SsaOp::FpuControl`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum FpuControlKind {
    /// Load the FPU control word (`fldcw`).
    LoadControlWord,
    /// Store the FPU control word (`fnstcw`/`fstcw`).
    StoreControlWord,
    /// Store the FPU status word (`fnstsw`/`fstsw`).
    StoreStatusWord,
    /// Load the FPU environment (`fldenv`).
    LoadEnvironment,
    /// Store the FPU environment (`fnstenv`/`fstenv`).
    StoreEnvironment,
    /// Save full FPU state (`fnsave`/`fsave`).
    Save,
    /// Restore full FPU state (`frstor`).
    Restore,
    /// Save extended (SSE+FPU) state (`fxsave`).
    SaveExtended,
    /// Restore extended (SSE+FPU) state (`fxrstor`).
    RestoreExtended,
    /// Clear floating-point exceptions (`fnclex`/`fclex`).
    ClearExceptions,
    /// Decrement the FPU stack-top pointer (`fdecstp`).
    DecrementStackTop,
    /// Increment the FPU stack-top pointer (`fincstp`).
    IncrementStackTop,
    /// Mark an FPU register free (`ffree`).
    FreeRegister,
    /// FPU no-op (`fnop`).
    NoOp,
    /// Wait for pending FPU exceptions (`wait`/`fwait`).
    Wait,
    /// Initialize / reset the FPU unit (`fninit`/`finit`) — clears the control
    /// word, status word, tag word and resets the stack top.
    Initialize,
    /// Empty the MMX technology state (`emms`/`femms`) — marks every x87 tag
    /// word entry empty, transitioning the register file out of MMX mode.
    EmptyMmxState,
}

/// Boxed payload for [`SsaOp::NativeIntrinsic`].
///
/// Mirrors [`NativeOpaqueData`] (explicit inputs/outputs, clobbers, effect
/// summary) but adds a structured [`NativeIntrinsicId`]. Boxed for the same
/// size reason as [`NativeOpaqueData`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NativeIntrinsicData {
    /// Structured identity of the native operation.
    pub id: NativeIntrinsicId,
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

/// Namespace of a system / control / status register accessed by a
/// [`SystemOpKind::ReadSysReg`] / [`SystemOpKind::WriteSysReg`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SysRegNamespace {
    /// x86 model-specific register (`rdmsr`/`wrmsr`).
    X86Msr,
    /// x86 extended control register (`xgetbv`/`xsetbv`).
    X86Xcr,
    /// x86 control register (`mov cr`).
    X86ControlReg,
    /// x86 debug register (`mov dr`).
    X86DebugReg,
    /// AArch64 system register (`mrs`/`msr`).
    Arm64System,
    /// RISC-V control/status register (`csrr*`/`csrw*`).
    RiscvCsr,
    /// MIPS coprocessor-0 register (`mfc0`/`mtc0`).
    MipsCop0,
}

/// Structured identity of a typed native **system / privileged** operation
/// ([`SsaOp::SystemOp`]) — the first-class replacement for the system,
/// control-register, syscall, trap and cache/TLB cases formerly carried by
/// `NativeIntrinsic` / `NativeOpaque`. The kind drives a precise effect summary
/// and a distinct similarity class; there is no catch-all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SystemOpKind {
    /// Processor identification (`cpuid`).
    CpuId,
    /// Time-stamp / cycle counter read (`rdtsc`/`rdtscp`/`rdcycle`); `aux` is
    /// true when an auxiliary value (e.g. TSC_AUX) is also produced (`rdtscp`).
    Timestamp {
        /// True when an auxiliary value (e.g. `TSC_AUX` for `rdtscp`) is also produced.
        aux: bool,
    },
    /// Read a system/control/model-specific register into a GPR.
    ReadSysReg {
        /// Register file the operand names (MSR / CR / DR / AArch64 sysreg / RISC-V CSR).
        namespace: SysRegNamespace,
    },
    /// Write a GPR into a system/control/model-specific register.
    WriteSysReg {
        /// Register file the operand names (MSR / CR / DR / AArch64 sysreg / RISC-V CSR).
        namespace: SysRegNamespace,
    },
    /// Read a performance-monitoring counter (`rdpmc`).
    ReadPerfCounter,
    /// System-call entry (`syscall`/`sysenter`/`svc`/`ecall`).
    SystemCall,
    /// System-call / fast-return (`sysret`/`sysexit`).
    SystemReturn,
    /// Software trap / interrupt (`int N`/`int3`/`brk`/`bkpt`); `vector` carries
    /// the trap number when statically known.
    Trap {
        /// Trap / interrupt vector when statically known (e.g. `int 0x80`), else `None`.
        vector: Option<u8>,
    },
    /// Interrupt / exception return (`iret`/`iretq`/`eret`).
    InterruptReturn,
    /// Cache maintenance (`invd`/`wbinvd`/`clflush`/`dc`/`ic`).
    CacheMaintenance,
    /// TLB maintenance (`invlpg`/`tlbi`).
    TlbMaintenance,
    /// Privileged barrier / serialization not expressible as a plain `Fence`.
    Barrier,
    /// Privileged machine-state op with no value result (`hlt`/`cli`/`sti`,
    /// `lgdt`/`lidt`/`swapgs`, ARM `cps`).
    Privileged,
    /// Virtualization / hypervisor call (`vmcall`/`vmlaunch`, ARM `hvc`).
    Hypervisor,
    /// On-chip hardware acceleration engine operating on memory buffers (VIA
    /// PadLock `xstore`/`xcrypt*`/`xsha*`): a crypto / hash / random-number
    /// engine driven by implicit pointer/count registers, writing its output
    /// buffer to memory.
    HardwareEngine,
    /// Hardware transactional-memory control (AArch64 TME `tstart`/`tcommit`/
    /// `tcancel`/`ttest`, x86 TSX `xbegin`/`xend`/`xabort`/`xtest`), named by
    /// its [`SystemTransactionKind`].
    Transaction(SystemTransactionKind),
}

/// The hardware transactional-memory operation carried by
/// [`SystemOpKind::Transaction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SystemTransactionKind {
    /// Begin a transaction (`tstart`/`xbegin`).
    Start,
    /// Commit the current transaction (`tcommit`/`xend`).
    Commit,
    /// Abort the current transaction (`tcancel`/`xabort`).
    Cancel,
    /// Test whether executing transactionally (`ttest`/`xtest`).
    Test,
}

impl SystemOpKind {
    /// Precise effect summary for this system op — derived from the kind, never
    /// an opaque echo. Reads of machine state report `Read`; writes (incl. cache
    /// / TLB / privileged state mutation) report `Write`; control-transferring
    /// ops (syscalls, traps, hypervisor calls, interrupt returns) report `Call`
    /// with the matching control effect.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::CpuId
            | Self::Timestamp { .. }
            | Self::ReadSysReg { .. }
            | Self::ReadPerfCounter => SsaEffects::new(SsaEffectKind::Read, false),
            Self::WriteSysReg { .. }
            | Self::Privileged
            | Self::CacheMaintenance
            | Self::TlbMaintenance
            | Self::HardwareEngine
            | Self::Transaction(_) => SsaEffects::new(SsaEffectKind::Write, false),
            // `dsb`/`dmb`/`isb`/`mfence` order memory: they must classify as a
            // fence so Memory SSA emits a `MemoryOp::Barrier` and the verifier's
            // fence invariant applies, exactly like `SsaOp::Fence`. Classifying
            // them as `Write` is conservatively safe for movement but models an
            // ordering construct as a clobber. The kind does not distinguish
            // `dmb` from `isb`, so assume the strongest ordering.
            Self::Barrier => {
                SsaEffects::new(SsaEffectKind::Fence, false).fence_ordering(AtomicOrdering::SeqCst)
            }
            Self::SystemCall | Self::SystemReturn | Self::Hypervisor | Self::Trap { .. } => {
                SsaEffects::new(SsaEffectKind::Call, true).with_control(ControlEffect::Call)
            }
            // `iret`/`eret` transfer control externally to the interrupted
            // context — a `Call`-class control transfer, exactly like
            // `SystemReturn` (`sysret`/`sysexit`), NOT a `Return`. Front-ends
            // classify it as a non-terminating typed system op (its family is
            // not a block terminator), so the block structurally continues past
            // it; declaring `ControlEffect::Return` here would make a
            // non-terminator op claim a block-ending effect and fail the
            // `check_native_effects` verifier invariant.
            Self::InterruptReturn => {
                SsaEffects::new(SsaEffectKind::Call, false).with_control(ControlEffect::Call)
            }
        }
    }

    /// Stable display / fingerprint key for this system op (used by
    /// [`SsaOp::opcode_name`]).
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::CpuId => "system.cpuid",
            Self::Timestamp { .. } => "system.timestamp",
            Self::ReadSysReg { .. } => "system.sysreg.read",
            Self::WriteSysReg { .. } => "system.sysreg.write",
            Self::ReadPerfCounter => "system.perfcounter",
            Self::SystemCall => "system.syscall",
            Self::SystemReturn => "system.sysreturn",
            Self::Trap { .. } => "system.trap",
            Self::InterruptReturn => "system.iret",
            Self::CacheMaintenance => "system.cache",
            Self::TlbMaintenance => "system.tlb",
            Self::Barrier => "system.barrier",
            Self::Privileged => "system.privileged",
            Self::Hypervisor => "system.hypervisor",
            Self::HardwareEngine => "system.hwengine",
            Self::Transaction(SystemTransactionKind::Start) => "system.txn.start",
            Self::Transaction(SystemTransactionKind::Commit) => "system.txn.commit",
            Self::Transaction(SystemTransactionKind::Cancel) => "system.txn.cancel",
            Self::Transaction(SystemTransactionKind::Test) => "system.txn.test",
        }
    }
}

/// Structured identity of a typed native **compute** operation
/// ([`SsaOp::ComputeOp`]) — the first-class replacement for the hardware
/// compute intrinsics (`pdep`/`pext`, `crc32`, `rdrand`/`rdseed`, pointer
/// authentication) formerly carried opaquely by `NativeIntrinsic`. The kind
/// drives a precise effect summary and a distinct similarity class; there is no
/// catch-all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ComputeKind {
    /// BMI2 `pdep` — parallel bit deposit of the low bits of the source into the
    /// positions set in the mask. Pure and deterministic.
    BitDeposit,
    /// BMI2 `pext` — parallel bit extract of the masked bits of the source into
    /// the low bits of the result. Pure and deterministic.
    BitExtract,
    /// Hardware CRC32 accumulation (`crc32`) — pure checksum step.
    Checksum,
    /// Hardware random number / seed (`rdrand`/`rdseed`). Nondeterministic: it
    /// reads a hardware entropy source, so it is never pure / foldable;
    /// `from_entropy` distinguishes `rdseed` (true) from `rdrand` (false).
    Random {
        /// `true` for `rdseed` (conditioned entropy), `false` for `rdrand`.
        from_entropy: bool,
    },
    /// AArch64 pointer authentication carrying the PAC sub-operation
    /// (`pac*`/`aut*`/`xpac*`/`pacga`). Pure and deterministic given the key.
    PointerAuth(PacKind),
    /// MIPS DSP-ASE accumulator operation — extract / shift / shift-load of the
    /// 64-bit DSP accumulator pair, and the modular-subtract address step
    /// (`extr*`/`extp*`/`shilo*`/`mthlip`/`modsub`). The specific operation
    /// (including the rounding / saturating extract variants) is preserved in
    /// [`NativeKindedData::mnemonic`]; all are pure given their register inputs.
    MipsDspAccumulate,
}

impl ComputeKind {
    /// Precise effect summary for this compute op — derived from the kind, never
    /// an opaque echo. Bit-permute / checksum / pointer-auth are pure;
    /// random-source reads a nondeterministic hardware entropy source (`Read`)
    /// so it is never folded or eliminated.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::BitDeposit
            | Self::BitExtract
            | Self::Checksum
            | Self::PointerAuth(_)
            | Self::MipsDspAccumulate => SsaEffects::new(SsaEffectKind::Pure, false),
            Self::Random { .. } => SsaEffects::new(SsaEffectKind::Read, false),
        }
    }

    /// Stable display / fingerprint key for this compute op (used by
    /// [`SsaOp::opcode_name`]).
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::BitDeposit => "compute.pdep",
            Self::BitExtract => "compute.pext",
            Self::Checksum => "compute.crc32",
            Self::Random {
                from_entropy: false,
            } => "compute.rdrand",
            Self::Random { from_entropy: true } => "compute.rdseed",
            Self::PointerAuth(_) => "compute.pac",
            Self::MipsDspAccumulate => "compute.mips_dsp_acc",
        }
    }
}

/// Structured identity of a legacy x86 **binary-coded-decimal adjust**
/// operation ([`SsaOp::BcdAdjust`]) — the first-class, named model for the
/// `daa`/`das`/`aaa`/`aas`/`aam`/`aad` instructions. Each variant names the
/// exact hardware operation (the LLVM-intrinsic model): the lifter wires the
/// accumulator and flags through as typed SSA values rather than decomposing
/// the flag-dependent correction into branches, and never carries it opaquely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum BcdAdjustKind {
    /// Decimal adjust `AL` after addition (`daa`).
    DecimalAddAdjust,
    /// Decimal adjust `AL` after subtraction (`das`).
    DecimalSubAdjust,
    /// ASCII adjust `AL` after addition (`aaa`).
    AsciiAddAdjust,
    /// ASCII adjust `AL` after subtraction (`aas`).
    AsciiSubAdjust,
    /// ASCII adjust `AX` after multiply (`aam`); the radix rides
    /// [`BcdAdjustData::base`] (10 unless an explicit `imm8` is given).
    AsciiMulAdjust,
    /// ASCII adjust `AX` before division (`aad`); the radix rides
    /// [`BcdAdjustData::base`].
    AsciiDivAdjust,
}

impl BcdAdjustKind {
    /// Effect summary — every BCD adjust is a pure function of the accumulator
    /// and the incoming arithmetic flags (no memory, no trap, no nondeterminism).
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        SsaEffects::new(SsaEffectKind::Pure, false)
    }

    /// Stable display / fingerprint key for this BCD adjust.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::DecimalAddAdjust => "bcd.daa",
            Self::DecimalSubAdjust => "bcd.das",
            Self::AsciiAddAdjust => "bcd.aaa",
            Self::AsciiSubAdjust => "bcd.aas",
            Self::AsciiMulAdjust => "bcd.aam",
            Self::AsciiDivAdjust => "bcd.aad",
        }
    }
}

/// Boxed payload for [`SsaOp::BcdAdjust`]. Mirrors the typed-compute operand
/// shape — explicit SSA inputs/outputs (the accumulator and the flag values),
/// clobbered architectural state, optional source provenance — with the effect
/// summary and similarity class derived from the [`BcdAdjustKind`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BcdAdjustData {
    /// Structured identity of the operation.
    pub kind: BcdAdjustKind,
    /// Radix for the ASCII multiply/divide adjusts (`aam`/`aad`); 10 (or unused)
    /// for the decimal/ASCII add/subtract adjusts.
    pub base: u8,
    /// Human-readable native mnemonic, for display / provenance only.
    pub mnemonic: String,
    /// Original native instruction metadata when known.
    pub metadata: Option<NativeInstructionMetadata>,
    /// Explicit SSA outputs defined by the operation (adjusted accumulator,
    /// result flags).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the operation (source accumulator, and the
    /// incoming flags for the add/subtract adjusts).
    pub inputs: Vec<SsaVarId>,
    /// Architectural state the operation clobbers.
    pub clobbers: Vec<NativeClobber>,
}

/// Structured identity of a hardware **vector cryptographic** operation
/// ([`SsaOp::VectorCrypto`]) — the first-class, named replacement for the
/// AES / SHA / SM3 / SM4 / GF(2^8) / carry-less-multiply round and message
/// primitives. Each variant names the exact hardware operation (the LLVM-
/// intrinsic model): the lifter never decomposes a round into primitive bit
/// ops nor carries it as an opaque blob. Every variant is a pure function of
/// its vector inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum VectorCryptoKind {
    /// AES encryption round (`aesenc`).
    AesEncrypt,
    /// AES last encryption round (`aesenclast`).
    AesEncryptLast,
    /// AES decryption round (`aesdec`).
    AesDecrypt,
    /// AES last decryption round (`aesdeclast`).
    AesDecryptLast,
    /// AES inverse mix-columns (`aesimc`).
    AesInvMixColumns,
    /// AES round-key generation assist (`aeskeygenassist`).
    AesKeygenAssist,
    /// AES single encryption round without mix-columns: AddRoundKey then
    /// SubBytes and ShiftRows (AArch64 `aese`).
    AesEncryptRound,
    /// AES single decryption round without inverse mix-columns: AddRoundKey
    /// then inverse SubBytes and ShiftRows (AArch64 `aesd`).
    AesDecryptRound,
    /// AES forward mix-columns (AArch64 `aesmc`).
    AesMixColumns,
    /// SHA-1 four rounds (`sha1rnds4`).
    Sha1Rounds4,
    /// SHA-1 next `E` value (`sha1nexte`).
    Sha1NextE,
    /// SHA-1 message schedule step 1 (`sha1msg1`).
    Sha1Msg1,
    /// SHA-1 message schedule step 2 (`sha1msg2`).
    Sha1Msg2,
    /// SHA-256 two rounds (`sha256rnds2`).
    Sha256Rounds2,
    /// SHA-256 message schedule step 1 (`sha256msg1`).
    Sha256Msg1,
    /// SHA-256 message schedule step 2 (`sha256msg2`).
    Sha256Msg2,
    /// SHA-512 two rounds (`vsha512rnds2`).
    Sha512Rounds2,
    /// SHA-512 message schedule step 1 (`vsha512msg1`).
    Sha512Msg1,
    /// SHA-512 message schedule step 2 (`vsha512msg2`).
    Sha512Msg2,
    /// SM3 message schedule step 1 (`vsm3msg1`).
    Sm3Msg1,
    /// SM3 message schedule step 2 (`vsm3msg2`).
    Sm3Msg2,
    /// SM3 two rounds (`vsm3rnds2`).
    Sm3Rounds2,
    /// SM4 key expansion (`vsm4key4`).
    Sm4Key,
    /// SM4 four rounds (`vsm4rnds4`).
    Sm4Rounds,
    /// GF(2^8) affine transform (`gf2p8affineqb`).
    Gf2p8Affine,
    /// GF(2^8) inverse affine transform (`gf2p8affineinvqb`).
    Gf2p8AffineInv,
    /// GF(2^8) multiply (`gf2p8mulb`).
    Gf2p8Mul,
    /// Carry-less (polynomial) multiply (`pclmulqdq`).
    CarrylessMul,
    /// AES key-locker encryption using a wrapped key handle
    /// (`aesenc128kl`/`aesenc256kl` and their wide `aesencwide*kl` peers).
    AesEncryptKeyLocker,
    /// AES key-locker decryption using a wrapped key handle
    /// (`aesdec128kl`/`aesdec256kl` and their wide `aesdecwide*kl` peers).
    AesDecryptKeyLocker,
    /// SHA-1 hash update using the choose function (AArch64 `sha1c`).
    Sha1HashChoose,
    /// SHA-1 hash update using the majority function (AArch64 `sha1m`).
    Sha1HashMajority,
    /// SHA-1 hash update using the parity function (AArch64 `sha1p`).
    Sha1HashParity,
    /// SHA-1 fixed rotate of the working variable (AArch64 `sha1h`).
    Sha1FixedRotate,
    /// SHA-1 message schedule update step 0 (AArch64 `sha1su0`).
    Sha1ScheduleUpdate0,
    /// SHA-1 message schedule update step 1 (AArch64 `sha1su1`).
    Sha1ScheduleUpdate1,
    /// SHA-256 hash update part 1 (AArch64 `sha256h`).
    Sha256Hash,
    /// SHA-256 hash update part 2 (AArch64 `sha256h2`).
    Sha256Hash2,
    /// SHA-256 message schedule update step 0 (AArch64 `sha256su0`).
    Sha256ScheduleUpdate0,
    /// SHA-256 message schedule update step 1 (AArch64 `sha256su1`).
    Sha256ScheduleUpdate1,
    /// SM3 message schedule part 1 (AArch64 `sm3partw1`).
    Sm3PartW1,
    /// SM3 message schedule part 2 (AArch64 `sm3partw2`).
    Sm3PartW2,
    /// SM3 hash update step 1 helper (AArch64 `sm3ss1`).
    Sm3SS1,
    /// SM3 hash update T1 variant A (AArch64 `sm3tt1a`).
    Sm3TT1A,
    /// SM3 hash update T1 variant B (AArch64 `sm3tt1b`).
    Sm3TT1B,
    /// SM3 hash update T2 variant A (AArch64 `sm3tt2a`).
    Sm3TT2A,
    /// SM3 hash update T2 variant B (AArch64 `sm3tt2b`).
    Sm3TT2B,
    /// Polynomial (carry-less) multiply long over GF(2) (AArch64 `pmull`).
    PolynomialMultiply,
}

impl VectorCryptoKind {
    /// Effect summary — every vector-crypto primitive is a pure function of its
    /// inputs (no memory, no trap, no nondeterminism).
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        SsaEffects::new(SsaEffectKind::Pure, false)
    }

    /// Stable display / fingerprint key.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::AesEncrypt => "crypto.aesenc",
            Self::AesEncryptLast => "crypto.aesenclast",
            Self::AesDecrypt => "crypto.aesdec",
            Self::AesDecryptLast => "crypto.aesdeclast",
            Self::AesInvMixColumns => "crypto.aesimc",
            Self::AesKeygenAssist => "crypto.aeskeygenassist",
            Self::AesEncryptRound => "crypto.aese",
            Self::AesDecryptRound => "crypto.aesd",
            Self::AesMixColumns => "crypto.aesmc",
            Self::Sha1Rounds4 => "crypto.sha1rnds4",
            Self::Sha1NextE => "crypto.sha1nexte",
            Self::Sha1Msg1 => "crypto.sha1msg1",
            Self::Sha1Msg2 => "crypto.sha1msg2",
            Self::Sha256Rounds2 => "crypto.sha256rnds2",
            Self::Sha256Msg1 => "crypto.sha256msg1",
            Self::Sha256Msg2 => "crypto.sha256msg2",
            Self::Sha512Rounds2 => "crypto.sha512rnds2",
            Self::Sha512Msg1 => "crypto.sha512msg1",
            Self::Sha512Msg2 => "crypto.sha512msg2",
            Self::Sm3Msg1 => "crypto.sm3msg1",
            Self::Sm3Msg2 => "crypto.sm3msg2",
            Self::Sm3Rounds2 => "crypto.sm3rnds2",
            Self::Sm4Key => "crypto.sm4key",
            Self::Sm4Rounds => "crypto.sm4rnds",
            Self::Gf2p8Affine => "crypto.gf2p8affine",
            Self::Gf2p8AffineInv => "crypto.gf2p8affineinv",
            Self::Gf2p8Mul => "crypto.gf2p8mul",
            Self::CarrylessMul => "crypto.pclmulqdq",
            Self::AesEncryptKeyLocker => "crypto.aesenckl",
            Self::AesDecryptKeyLocker => "crypto.aesdeckl",
            Self::Sha1HashChoose => "crypto.sha1c",
            Self::Sha1HashMajority => "crypto.sha1m",
            Self::Sha1HashParity => "crypto.sha1p",
            Self::Sha1FixedRotate => "crypto.sha1h",
            Self::Sha1ScheduleUpdate0 => "crypto.sha1su0",
            Self::Sha1ScheduleUpdate1 => "crypto.sha1su1",
            Self::Sha256Hash => "crypto.sha256h",
            Self::Sha256Hash2 => "crypto.sha256h2",
            Self::Sha256ScheduleUpdate0 => "crypto.sha256su0",
            Self::Sha256ScheduleUpdate1 => "crypto.sha256su1",
            Self::Sm3PartW1 => "crypto.sm3partw1",
            Self::Sm3PartW2 => "crypto.sm3partw2",
            Self::Sm3SS1 => "crypto.sm3ss1",
            Self::Sm3TT1A => "crypto.sm3tt1a",
            Self::Sm3TT1B => "crypto.sm3tt1b",
            Self::Sm3TT2A => "crypto.sm3tt2a",
            Self::Sm3TT2B => "crypto.sm3tt2b",
            Self::PolynomialMultiply => "crypto.pmull",
        }
    }
}

/// The element interpretation of an AMX tile dot-product ([`TileOpKind`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TileDotKind {
    /// Signed × signed byte dot product (`tdpbssd`).
    Int8SignedSigned,
    /// Signed × unsigned byte dot product (`tdpbsud`).
    Int8SignedUnsigned,
    /// Unsigned × signed byte dot product (`tdpbusd`).
    Int8UnsignedSigned,
    /// Unsigned × unsigned byte dot product (`tdpbuud`).
    Int8UnsignedUnsigned,
    /// BF16 dot product accumulating into f32 (`tdpbf16ps`).
    Bf16,
    /// FP16 dot product accumulating into f32 (`tdpfp16ps`).
    Fp16,
    /// Complex FP16 matrix multiply, real part, accumulating into f32
    /// (`tcmmrlfp16ps`).
    ComplexFp16Real,
    /// Complex FP16 matrix multiply, imaginary part, accumulating into f32
    /// (`tcmmimfp16ps`).
    ComplexFp16Imaginary,
}

/// Structured identity of an AMX (Advanced Matrix Extensions) **tile**
/// operation ([`SsaOp::TileOp`]). Tiles are 2-D matrix registers, distinct
/// from 1-D SIMD vectors; the kind names the exact tile operation and drives a
/// precise effect summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TileOpKind {
    /// Tile matrix multiply-accumulate (`tdp*`); the [`TileDotKind`] names the
    /// element interpretation. Pure function of the source tiles.
    DotProduct(TileDotKind),
    /// Load a tile from memory (`tileloadd`/`tileloaddt1`).
    Load,
    /// Store a tile to memory (`tilestored`).
    Store,
    /// Zero a tile register (`tilezero`). Pure.
    Zero,
    /// Load the tile configuration from memory (`ldtilecfg`).
    LoadConfig,
    /// Store the tile configuration to memory (`sttilecfg`).
    StoreConfig,
    /// Release all tile state (`tilerelease`).
    Release,
}

impl TileOpKind {
    /// Precise effect summary: dot-product / zero are pure; loads read memory,
    /// stores write memory; `tilerelease` mutates architectural tile state.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::DotProduct(_) | Self::Zero => SsaEffects::new(SsaEffectKind::Pure, false),
            Self::Load | Self::LoadConfig => SsaEffects::new(SsaEffectKind::Read, false),
            Self::Store | Self::StoreConfig | Self::Release => {
                SsaEffects::new(SsaEffectKind::Write, false)
            }
        }
    }

    /// Stable display / fingerprint key.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::DotProduct(TileDotKind::Int8SignedSigned) => "tile.dp.bssd",
            Self::DotProduct(TileDotKind::Int8SignedUnsigned) => "tile.dp.bsud",
            Self::DotProduct(TileDotKind::Int8UnsignedSigned) => "tile.dp.busd",
            Self::DotProduct(TileDotKind::Int8UnsignedUnsigned) => "tile.dp.buud",
            Self::DotProduct(TileDotKind::Bf16) => "tile.dp.bf16ps",
            Self::DotProduct(TileDotKind::Fp16) => "tile.dp.fp16ps",
            Self::DotProduct(TileDotKind::ComplexFp16Real) => "tile.cmm.rlfp16ps",
            Self::DotProduct(TileDotKind::ComplexFp16Imaginary) => "tile.cmm.imfp16ps",
            Self::Load => "tile.load",
            Self::Store => "tile.store",
            Self::Zero => "tile.zero",
            Self::LoadConfig => "tile.ldcfg",
            Self::StoreConfig => "tile.stcfg",
            Self::Release => "tile.release",
        }
    }
}

/// Boxed payload for [`SsaOp::VectorPermute`] — a lane permute whose selector
/// is a **runtime index vector** (unlike [`SsaOp::VectorShuffle`], whose mask is
/// static). The `inputs` are the data source vector(s) followed by the index
/// vector (`[src, index]` for single-source `vpermd`/`vpshufb`, `[src1, src2,
/// index]` for the two-source `vpermt2*`/`vpermi2*`). Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorPermuteData {
    /// Explicit SSA outputs defined by the operation (the permuted vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the data source vector(s) then the index vector.
    pub inputs: Vec<SsaVarId>,
}

/// The specific fused multiply-then-horizontal-add operation named by a
/// [`SsaOp::VectorMultiplyAdd`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum VectorMaddKind {
    /// `pmaddwd`: signed 16×16→32 products, adjacent pairs summed.
    MultiplyAddS16,
    /// `pmaddubsw`: unsigned×signed 8×8→16 products, adjacent pairs summed with
    /// signed saturation.
    MultiplyAddU8S8Sat,
    /// `vpdpbusd`: unsigned×signed byte products, groups of four summed into a
    /// 32-bit accumulator (the destination).
    DotProductU8S8,
    /// `vpdpbusds`: as [`Self::DotProductU8S8`] with signed-saturating
    /// accumulation.
    DotProductU8S8Sat,
    /// `vpdpwssd`: signed word products, pairs summed into a 32-bit accumulator.
    DotProductS16,
    /// `vpdpwssds`: as [`Self::DotProductS16`] with signed-saturating
    /// accumulation.
    DotProductS16Sat,
    /// `vpmadd52luq`: low 52 bits of unsigned 52×52 products added to a 64-bit
    /// accumulator.
    MultiplyAdd52Lo,
    /// `vpmadd52huq`: high 52 bits of unsigned 52×52 products added to a 64-bit
    /// accumulator.
    MultiplyAdd52Hi,
    /// `vdpbf16ps`: bf16 pairs multiplied and summed into a 32-bit float
    /// accumulator.
    DotProductBf16,
    /// AMD XOP `vpmacs*`: per-lane signed multiply added to a third
    /// accumulator source.
    MultiplyAccumulate,
    /// AMD XOP `vpmacss*`: as [`Self::MultiplyAccumulate`] with signed-saturating
    /// accumulation.
    MultiplyAccumulateSat,
    /// AMD XOP `vpmadcswd`: signed word products summed in adjacent pairs and
    /// added to a third accumulator source.
    MultiplyAccumulatePairs,
    /// AMD XOP `vpmadcsswd`: as [`Self::MultiplyAccumulatePairs`] with
    /// signed-saturating accumulation.
    MultiplyAccumulatePairsSat,
    /// `vpdpbssd[s]`: signed×signed byte dot-product into a 32-bit accumulator.
    DotProductS8S8,
    /// `vpdpbssds`: saturating signed×signed byte dot-product.
    DotProductS8S8Sat,
    /// `vpdpbsud[s]`: signed×unsigned byte dot-product.
    DotProductS8U8,
    /// `vpdpbsuds`: saturating signed×unsigned byte dot-product.
    DotProductS8U8Sat,
    /// `vpdpbuud[s]`: unsigned×unsigned byte dot-product.
    DotProductU8U8,
    /// `vpdpbuuds`: saturating unsigned×unsigned byte dot-product.
    DotProductU8U8Sat,
    /// `vpdpwsud[s]`: signed×unsigned word dot-product.
    DotProductS16U16,
    /// `vpdpwsuds`: saturating signed×unsigned word dot-product.
    DotProductS16U16Sat,
    /// `vpdpwusd[s]`: unsigned×signed word dot-product.
    DotProductU16S16,
    /// `vpdpwusds`: saturating unsigned×signed word dot-product.
    DotProductU16S16Sat,
    /// `vpdpwuud[s]`: unsigned×unsigned word dot-product.
    DotProductU16U16,
    /// `vpdpwuuds`: saturating unsigned×unsigned word dot-product.
    DotProductU16U16Sat,
}

/// The specific complex-number floating-point multiply named by a
/// [`SsaOp::VectorComplexMul`]. Each lane pair holds a `(real, imag)` complex
/// value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum ComplexMulKind {
    /// `vfmulcph`/`vfmulcsh`: complex multiply `a * b`.
    Multiply,
    /// `vfcmulcph`/`vfcmulcsh`: complex multiply by the conjugate `a * conj(b)`.
    ConjugateMultiply,
    /// `vfmaddcph`/`vfmaddcsh`: complex multiply-accumulate `acc + a * b`.
    MultiplyAdd,
    /// `vfcmaddcph`/`vfcmaddcsh`: conjugate multiply-accumulate
    /// `acc + a * conj(b)`.
    ConjugateMultiplyAdd,
}

impl ComplexMulKind {
    /// Returns the stable textual identity used in similarity / display.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Multiply => "vector.cmul",
            Self::ConjugateMultiply => "vector.cmul.conj",
            Self::MultiplyAdd => "vector.cmadd",
            Self::ConjugateMultiplyAdd => "vector.cmadd.conj",
        }
    }
}

/// Boxed payload for [`SsaOp::VectorDotProduct`] — a masked floating-point dot
/// product (x86 SSE4.1 `dpps`/`dppd` and VEX `vdpps`/`vdppd`). The 8-bit
/// immediate selects which source lanes participate in the sum (high nibble)
/// and which destination lanes receive the broadcast result (low nibble);
/// `element_bits` is the lane width (32 for `*ps`, 64 for `*pd`). The `inputs`
/// are the two source vectors `[a, b]` (the destination doubles as `a`). Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorDotProductData {
    /// The 8-bit lane-participation / result-broadcast control immediate.
    pub imm8: u8,
    /// Lane width in bits (32 for `*ps`, 64 for `*pd`).
    pub element_bits: u16,
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the two source vectors `[a, b]`.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorIntDotProduct`] — an integer grouped
/// dot-product-accumulate (ARM/AArch64 `sdot`/`udot`/`usdot`): each output lane
/// sums the products of a contiguous group of `dest_bits / source_bits`
/// source-element pairs (widened from `source_bits`), added into the destination
/// lane. `signed_a`/`signed_b` give the per-operand sign-extension (they differ
/// only for the mixed `usdot`). The `inputs` are `[acc, a, b]` (the destination
/// accumulator and the two source vectors). Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorIntDotProductData {
    /// `true` when the first source's elements are sign-extended.
    pub signed_a: bool,
    /// `true` when the second source's elements are sign-extended.
    pub signed_b: bool,
    /// Source element width in bits (the group size is `dest_bits / source_bits`).
    pub source_bits: u16,
    /// Destination (accumulator) lane width in bits.
    pub dest_bits: u16,
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: `[acc, a, b]`.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorShuffleBits`] — an AVX-512 VBMI2
/// `vpshufbitqmb`: for each byte it gathers the source bit at the position
/// named by the corresponding control byte into a mask. The `outputs` are the
/// result mask; the `inputs` are the source and the control vectors. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorShuffleBitsData {
    /// Explicit SSA outputs: the result mask.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source and the control vectors.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorIntersect`] — an AVX-512 VP2INTERSECT
/// (`vp2intersectd`/`vp2intersectq`). It computes a pair of masks: the first
/// marks which elements of the first source also appear in the second, the
/// second marks the converse. The `outputs` are the two mask registers; the
/// `inputs` are the two source vectors. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorIntersectData {
    /// Explicit SSA outputs: the two result masks `[m1, m2]`.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the two source vectors `[a, b]`.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorBitfield`] — an SSE4a bit-field extract /
/// insert over the low 64 bits of a vector register (`extrq`/`insertq`).
/// `insert` selects insert (`insertq`) vs extract (`extrq`); `index` and
/// `length` are the bit position and width for the immediate-controlled forms
/// (both zero for the register-controlled forms, whose control vector is an
/// additional input). Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorBitfieldData {
    /// `true` for `insertq` (insert), `false` for `extrq` (extract).
    pub insert: bool,
    /// Bit position of the field for the immediate-controlled forms.
    pub index: u8,
    /// Bit width of the field for the immediate-controlled forms.
    pub length: u8,
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the destination vector, the source / control
    /// vector, and (for the register-controlled forms) the control vector.
    pub inputs: Vec<SsaVarId>,
}

/// Per-byte condition selecting which lanes a [`SsaOp::VectorConditionalMove`]
/// updates (Cyrix EMMI `pmv*` family). The exact test reference is part of the
/// underdocumented EMMI semantics; the variant names the documented predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ByteMoveCondition {
    /// Move where the tested byte is zero (`pmvzb`).
    Zero,
    /// Move where the tested byte is non-zero (`pmvnzb`).
    NonZero,
    /// Move where the tested byte is negative (`pmvlzb`).
    Negative,
    /// Move where the tested byte is non-negative (`pmvgezb`).
    NonNegative,
}

impl ByteMoveCondition {
    /// Stable display / fingerprint key.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Zero => "z",
            Self::NonZero => "nz",
            Self::Negative => "lz",
            Self::NonNegative => "gez",
        }
    }
}

/// Boxed payload for [`SsaOp::VectorConditionalMove`] — a Cyrix EMMI per-byte
/// conditional move (`pmvzb`/`pmvnzb`/`pmvlzb`/`pmvgezb`): selected bytes of the
/// source replace the destination under the [`ByteMoveCondition`]. The `inputs`
/// are the destination and source vectors. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorConditionalMoveData {
    /// The per-byte move condition.
    pub condition: ByteMoveCondition,
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the destination and source vectors.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorStringCompare`] — an SSE4.2 packed string
/// comparison (`pcmpestri`/`pcmpestrm`/`pcmpistri`/`pcmpistrm` and their VEX
/// peers). The 8-bit immediate selects the data format, aggregation function
/// and polarity; `explicit_length` distinguishes the explicit-length `e*`
/// forms (which consume the two index registers) from the implicit
/// null-terminated `i*` forms; `result_index` distinguishes the index-result
/// `*stri` forms from the mask-result `*strm` forms. The `inputs` are the two
/// string vectors (plus the two length registers for the explicit forms); the
/// `outputs` are the result register and the comparison flags. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorStringCompareData {
    /// The 8-bit format / aggregation / polarity control immediate.
    pub imm8: u8,
    /// `true` for the explicit-length `pcmpestr*` forms (consume two index
    /// registers); `false` for the implicit null-terminated `pcmpistr*` forms.
    pub explicit_length: bool,
    /// `true` for the index-result `*stri` forms; `false` for the mask-result
    /// `*strm` forms.
    pub result_index: bool,
    /// Explicit SSA outputs: the result register and the comparison flags.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the two string vectors (plus the two length
    /// registers for the explicit-length forms).
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorHorizontalMinPos`] — the SSE4.1
/// `phminposuw` horizontal minimum: it finds the smallest unsigned 16-bit lane
/// of the single source, placing that minimum in the low word of the result
/// and its source index in the next field (remaining lanes zeroed). The single
/// input is the source vector. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorHorizontalMinPosData {
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source vector.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorHorizontalReduce`] — a grouped, widening
/// horizontal add / subtract (AMD XOP `vphadd*`/`vphsub*`): each output lane is
/// the sum (or difference) of a contiguous group of `dest_bits / source_bits`
/// input lanes, widened from `source_bits` to `dest_bits`. The single input is
/// the source vector. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorHorizontalReduceData {
    /// `true` for horizontal subtract (`vphsub*`); `false` for add.
    pub subtract: bool,
    /// `true` when the source lanes are zero-extended; `false` sign-extended.
    pub unsigned: bool,
    /// Source lane width in bits.
    pub source_bits: u16,
    /// Destination (widened) lane width in bits.
    pub dest_bits: u16,
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source vector.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorPackNarrow`] — a two-source saturating
/// narrowing pack (`packsswb`/`packssdw`/`packuswb`/`packusdw`). Each source's
/// lanes are narrowed to half width with signed or unsigned saturation, the
/// first source filling the low half of each 128-bit granule and the second the
/// high half. The `inputs` are `[src1, src2]`. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorPackNarrowData {
    /// `true` for unsigned saturation (`packuswb`/`packusdw`); `false` for
    /// signed (`packsswb`/`packssdw`).
    pub unsigned: bool,
    /// Explicit SSA outputs defined by the operation (the packed vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the two source vectors.
    pub inputs: Vec<SsaVarId>,
}

/// The SVE data-movement operation for [`SsaOp::VectorSvePermute`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SvePermuteKind {
    /// Generate a monotonically increasing index vector (`index`).
    Index,
    /// Insert a scalar at the bottom, shifting the vector up (`insr`).
    InsertShift,
    /// Pack the active elements into the low lanes (`compact`).
    Compact,
    /// Splice two vectors at the last active predicate element (`splice`).
    Splice,
    /// Extract the last active element to a scalar (`lasta`).
    ExtractLastActive,
    /// Extract the element before the last active to a scalar (`lastb`).
    ExtractLastBefore,
    /// Copy the last active element across the destination lanes (`clasta`).
    CopyLastActive,
    /// Copy the element before the last active across the lanes (`clastb`).
    CopyLastBefore,
}

/// The floating-point helper operation for [`SsaOp::VectorFpHelper`] (SVE
/// transcendental-acceleration primitives).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FpHelperKind {
    /// Reciprocal exponent (`frecpx`).
    ReciprocalExponent,
    /// Extract the biased exponent (`flogb`).
    ExtractExponent,
    /// Exponential acceleration table lookup (`fexpa`).
    ExpAccelerate,
    /// Trigonometric multiply-add (`ftmad`).
    TrigMulAdd,
    /// Trigonometric select-multiply (`ftsmul`).
    TrigSelectMul,
    /// Trigonometric select (`ftssel`).
    TrigSelect,
}

/// The predicate-generating operation for [`SsaOp::VectorPredicateGen`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PredicateGenKind {
    /// All-true predicate, optionally restricted by a count pattern (`ptrue`).
    True,
    /// All-false predicate (`pfalse`).
    False,
    /// Advance to the next active element (`pnext`).
    Next,
    /// Select the first active element (`pfirst`).
    First,
    /// Read the first-fault register into a predicate (`rdffr`).
    ReadFfr,
    /// Unpack and widen the high half of a predicate (`punpkhi`).
    UnpackHi,
    /// Unpack and widen the low half of a predicate (`punpklo`).
    UnpackLo,
    /// Predicate-wise select between two source predicates (`psel`).
    Select,
    /// Read-after-write address-hazard predicate (`whilerw`).
    HazardRw,
    /// Write-after-read address-hazard predicate (`whilewr`).
    HazardWr,
}

/// Boxed payload for [`SsaOp::VectorSmeOuterProduct`] — an SME outer-product
/// accumulate into a ZA tile (AArch64 `smopa`/`umopa`/`fmopa`/`bfmopa`/`usmopa`/
/// `sumopa` and the subtracting `*mops`). The ZA tile is the accumulator
/// (`inputs[0]`), followed by the governing predicates and the two source
/// vectors; the result is the updated tile. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorSmeOuterProductData {
    /// `true` for the subtracting `*mops` forms.
    pub subtract: bool,
    /// `true` when the first source is signed.
    pub signed_a: bool,
    /// `true` when the second source is signed.
    pub signed_b: bool,
    /// `true` for the floating-point forms (`fmopa`/`bfmopa`).
    pub float: bool,
    /// Explicit SSA outputs (the updated ZA tile).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the ZA accumulator, governing predicates, and sources.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorMatrixMulAcc`] — an integer/floating matrix
/// multiply-accumulate over vector registers (AArch64 `smmla`/`ummla`/`usmmla`/
/// `fmmla`/`bfmmla`): two source matrices are multiplied and accumulated into the
/// destination. The `inputs` are `[acc, a, b]`. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorMatrixMulAccData {
    /// `true` when the first source matrix is signed.
    pub signed_a: bool,
    /// `true` when the second source matrix is signed.
    pub signed_b: bool,
    /// `true` for the floating-point forms (`fmmla`/`bfmmla`).
    pub float: bool,
    /// Explicit SSA outputs (the accumulated result).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the accumulator and the two source matrices.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorReverseChunks`] — reverses the order of
/// `chunk_bits`-sized chunks within each vector element (AArch64 SVE
/// `revb`/`revh`/`revw`/`revd`: byte/halfword/word/doubleword order reversal). The
/// element width comes from the operand register shape. The `inputs` are `[src]`.
/// Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorReverseChunksData {
    /// Size in bits of each reversed chunk (8/16/32/64).
    pub chunk_bits: u16,
    /// Explicit SSA outputs (the reversed vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source vector.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorCountAdjust`] — adjusts a register by an
/// implicit count of vector elements (`VL / element_bits`) or active predicate
/// lanes, optionally saturating (AArch64 SVE `sqincb`/`uqincd`/`sqincp`/`incp`/
/// `decp`/…). The count is symbolic (the vector length is runtime-defined); the op
/// names the adjustment so no concrete materialization is needed. `inputs[0]` is
/// the value being adjusted, optionally followed by the governing predicate. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorCountAdjustData {
    /// `true` to subtract the count (`*dec*`), `false` to add (`*inc*`).
    pub decrement: bool,
    /// `true` for the saturating `sq*`/`uq*` forms.
    pub saturate: bool,
    /// `true` for signed saturation (`sq*`), `false` for unsigned (`uq*`).
    pub signed: bool,
    /// `true` when the count is the active-predicate population (`incp`/`sqincp`/…),
    /// `false` when it is the element count `VL / element_bits`.
    pub by_predicate: bool,
    /// Element width selecting the count granularity (`8`/`16`/`32`/`64`).
    pub element_bits: u16,
    /// Explicit SSA outputs (the adjusted value).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the adjusted value, then any governing predicate.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorExtendInLane`] — sign- or zero-extends the
/// low `source_bits` of each `element_bits`-wide lane to the full lane width
/// (AArch64 SVE predicated `sxtb`/`uxtb`/`sxth`/`uxth`/`sxtw`/`uxtw`, and the
/// `sxtw`/`uxtw` index extension performed by SVE `adr`). `inputs[0]` is the
/// source vector, optionally followed by a governing predicate. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorExtendInLaneData {
    /// `true` to sign-extend (`sxt*`), `false` to zero-extend (`uxt*`).
    pub signed: bool,
    /// Width in bits of the low field that is extended (8/16/32).
    pub source_bits: u16,
    /// Full lane width in bits (16/32/64).
    pub element_bits: u16,
    /// Explicit SSA outputs (the extended vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source vector, then any governing predicate.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorElementCount`] — computes the element count
/// `VL / element_bits` of the current vector length, scaled by `multiplier`, into
/// a scalar register (AArch64 SVE `cntb`/`cnth`/`cntw`/`cntd`). The count is
/// symbolic (the vector length is runtime-defined); the op names the granularity
/// so no concrete materialization is needed. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorElementCountData {
    /// Element width selecting the count granularity (`8`/`16`/`32`/`64`).
    pub element_bits: u16,
    /// Constant multiplier from the `mul #imm` form (`1` when absent).
    pub multiplier: u32,
    /// Explicit SSA outputs (the scalar count).
    pub outputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorSveAddressGen`] — SVE vector address
/// generation (`adr`): each destination lane is `base + (extend(index) << shift)`.
/// `inputs[0]` is the base vector, `inputs[1]` the index vector. The optional
/// 32→64 index extension and the left-shift amount are named on the op so the
/// per-lane address computation stays lossless. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorSveAddressGenData {
    /// `Some(true)` for the `sxtw` (signed) 32→64 index extension, `Some(false)`
    /// for `uxtw` (unsigned), `None` for the plain `lsl` form (no extension).
    pub signed_extend: Option<bool>,
    /// Left-shift amount applied to the index lanes (`0..=3`).
    pub shift: u8,
    /// Explicit SSA outputs (the address vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the base vector, then the index vector.
    pub inputs: Vec<SsaVarId>,
}

/// The SVE2/NEON compute operation named by [`VectorSveComputeData`]. One
/// variant per hardware operation keeps the lift lossless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SveComputeKind {
    /// `adclb`/`adclt`: add with carry into the bottom/top long lanes.
    AddCarryBottom,
    /// Add with carry into the top long lanes.
    AddCarryTop,
    /// `sbclb`/`sbclt`: subtract with carry into the bottom/top long lanes.
    SubCarryBottom,
    /// Subtract with carry into the top long lanes.
    SubCarryTop,
    /// `bdep`: gather selected bits (PDEP-like).
    BitDeposit,
    /// `bext`: scatter selected bits (PEXT-like).
    BitExtract,
    /// `bgrp`: group bits by a mask.
    BitGroup,
    /// `histcnt`: per-element histogram count.
    Histogram,
    /// `histseg`: segmented histogram match count.
    HistogramSegment,
    /// `match`: per-element membership predicate.
    MatchElements,
    /// `nmatch`: per-element non-membership predicate.
    NoMatchElements,
    /// `sclamp`: clamp each lane to a signed `[min,max]` range.
    ClampSigned,
    /// `uclamp`: clamp each lane to an unsigned `[min,max]` range.
    ClampUnsigned,
    /// `sdivr`: reversed signed divide (`b / a`).
    DivideReversedSigned,
    /// `udivr`: reversed unsigned divide (`b / a`).
    DivideReversedUnsigned,
    /// `eorbt`: interleaving exclusive-or, bottom from top.
    InterleaveXorBottomTop,
    /// `eortb`: interleaving exclusive-or, top from bottom.
    InterleaveXorTopBottom,
    /// `sqabs`: saturating absolute value.
    SaturatingAbs,
    /// `sqneg`: saturating negate.
    SaturatingNeg,
    /// `cnot`: logical NOT per lane (`x == 0 ? 1 : 0`).
    LogicalNot,
    /// `cdot`: complex integer dot product with rotation.
    ComplexDotProduct,
    /// `sqrdcmlah`: saturating rounding doubling complex multiply-accumulate.
    ComplexMulAddRounding,
}

/// Boxed payload for [`SsaOp::VectorSveCompute`] — an SVE2/NEON compute op named
/// precisely by its [`SveComputeKind`]. `rotation` carries the complex-op rotation
/// in degrees (0/90/180/270; 0 otherwise); `element_bits` is the lane width. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorSveComputeData {
    /// The exact hardware operation.
    pub op: SveComputeKind,
    /// Lane width in bits (8/16/32/64).
    pub element_bits: u16,
    /// Complex-op rotation in degrees (0/90/180/270); 0 when not applicable.
    pub rotation: u16,
    /// Explicit SSA outputs.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs.
    pub inputs: Vec<SsaVarId>,
}

/// The predicate/FFR operation named by [`VectorPredicateOpData`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PredicateOpKind {
    /// `cntp`: count active predicate elements into a scalar.
    CountActive,
    /// `ptest`: set condition flags from a predicate.
    Test,
    /// `setffr`: initialise the first-fault register to all-true.
    SetFirstFault,
    /// `wrffr`: write a predicate into the first-fault register.
    WriteFirstFault,
}

/// Boxed payload for [`SsaOp::VectorPredicateOp`] — an SVE predicate/first-fault
/// register operation named by its [`PredicateOpKind`]. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorPredicateOpData {
    /// The exact predicate/FFR operation.
    pub op: PredicateOpKind,
    /// Governing element width in bits (8/16/32/64).
    pub element_bits: u16,
    /// Explicit SSA outputs (scalar count / flags / FFR; may be empty).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs (source predicate(s)).
    pub inputs: Vec<SsaVarId>,
}

/// The SME tile operation named by [`VectorSmeMiscData`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SmeMiscKind {
    /// `addha`: horizontal add of vector elements into a ZA tile slice.
    AddHorizontal,
    /// `addva`: vertical add of vector elements into a ZA tile slice.
    AddVertical,
    /// `zero`: zero a list of ZA tiles.
    ZeroTiles,
}

/// Boxed payload for [`SsaOp::VectorSmeMisc`] — an SME ZA-tile accumulate/zero
/// operation named by its [`SmeMiscKind`]. Pure (models ZA as explicit operands).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorSmeMiscData {
    /// The exact SME operation.
    pub op: SmeMiscKind,
    /// Element width in bits (8/16/32/64).
    pub element_bits: u16,
    /// Explicit SSA outputs (the updated ZA tile).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs (governing predicates and source vectors).
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorStructLoadReplicate`] — AArch64 `ld2r`/`ld3r`/
/// `ld4r`: load a single 2/3/4-element structure from memory and replicate each
/// element across all lanes of its destination register.
///
/// Reads memory through its address base and may fault, so it classifies as
/// [`SsaEffectKind::Read`] with [`TrapClass::MemoryFault`] — never pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorStructLoadReplicateData {
    /// Number of structure elements / destination registers (2, 3, or 4).
    pub count: u8,
    /// Element width in bits (8/16/32/64).
    pub element_bits: u16,
    /// Explicit SSA outputs (the replicated destination registers).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs (the address base register).
    pub inputs: Vec<SsaVarId>,
}

/// The condition-flag (PSTATE) manipulation named by [`KindedVecData`]. Mirrors
/// the cleaned-IR `FlagAdjustKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum FlagAdjustKind {
    /// `cfinv` — invert the carry flag.
    InvertCarry,
    /// `rmif` — rotate a register, mask, and insert into NZCV.
    RotateMaskInsert,
    /// `setf8` — set N and Z from the low 8 bits of a register.
    SetNzFrom8,
    /// `setf16` — set N and Z from the low 16 bits of a register.
    SetNzFrom16,
    /// `axflag` — convert PSTATE flags to the alternative (FP) format.
    ConvertToFpFlags,
    /// `xaflag` — convert alternative (FP) flags back to PSTATE format.
    ConvertFromFpFlags,
}

/// Boxed payload for [`SsaOp::VectorComplexAdd`] — a complex-number add with a
/// 90° or 270° rotation of one operand's imaginary lane (AArch64 `fcadd` and the
/// integer `cadd`/`sqcadd`). The `inputs` are the two complex source vectors.
/// Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorComplexAddData {
    /// `true` for a 270° rotation, `false` for 90°.
    pub rotate_270: bool,
    /// `true` for the saturating integer form (`sqcadd`).
    pub saturate: bool,
    /// Explicit SSA outputs (the complex result).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the two complex source vectors.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorPredicateBreak`] — generates a predicate from
/// a source predicate by breaking the active region at the first true element
/// (AArch64 SVE `brka`/`brkb`/`brkn`/`brkpa`/`brkpb` and their flag-setting
/// `*s` peers). `after` keeps the element that triggered the break (`brka*`);
/// `pair` selects the two-source `brkp*` forms; `propagate` selects `brkn`.
/// The `inputs` are the source predicate(s). Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorPredicateBreakData {
    /// `true` to break *after* the first true element (`brka`/`brkpa`).
    pub after: bool,
    /// `true` for the two-source paired forms (`brkpa`/`brkpb`).
    pub pair: bool,
    /// `true` for the propagate form (`brkn`).
    pub propagate: bool,
    /// Explicit SSA outputs (the generated predicate).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source predicate(s).
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorPredicateWhile`] — generates a predicate by
/// comparing an incrementing index from a scalar base against a scalar limit
/// (AArch64 SVE `whilelt`/`whilele`/`whilege`/`whilegt` signed and `whilelo`/
/// `whilels`/`whilehs`/`whilehi` unsigned). The `inputs` are `[first, last]`. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorPredicateWhileData {
    /// The comparison relating each index to the limit.
    pub kind: VectorCompareKind,
    /// `true` for the unsigned forms (`whilelo`/`whilels`/`whilehs`/`whilehi`).
    pub unsigned: bool,
    /// Explicit SSA outputs (the generated predicate).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the scalar base and limit.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorNarrowSaturate`] — a single-source saturating
/// narrowing of a vector (AArch64 `sqxtn`/`uqxtn`/`sqxtun` and the shifting
/// `sqshrn`/`uqshrn`/`sqrshrn`/`sqshrun`/`sqrshrun`/… families). Each source lane
/// is optionally right-shifted by `shift` (rounded when `rounding`), then clamped
/// to the narrow destination range — signed when `unsigned_dst` is false, to the
/// unsigned range when true — interpreting the source as signed when `signed_src`.
/// The `inputs` are `[src]`. Pure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorNarrowSaturateData {
    /// `true` when the source lanes are interpreted as signed.
    pub signed_src: bool,
    /// `true` to saturate to the unsigned narrow range (`sqxtun`/`sqshrun`/…).
    pub unsigned_dst: bool,
    /// `true` for the rounding shift forms (`sqrshrn`/`uqrshrn`/`sqrshrun`/…).
    pub rounding: bool,
    /// Right-shift amount applied before narrowing (`0` for the extract forms
    /// `sqxtn`/`uqxtn`/`sqxtun`).
    pub shift: u8,
    /// Explicit SSA outputs defined by the operation (the narrowed vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the single source vector.
    pub inputs: Vec<SsaVarId>,
}

impl VectorMaddKind {
    /// Returns the stable textual identity used in similarity / display.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::MultiplyAddS16 => "vector.madd.s16",
            Self::MultiplyAddU8S8Sat => "vector.madd.u8s8.sat",
            Self::DotProductU8S8 => "vector.dp.u8s8",
            Self::DotProductU8S8Sat => "vector.dp.u8s8.sat",
            Self::DotProductS16 => "vector.dp.s16",
            Self::DotProductS16Sat => "vector.dp.s16.sat",
            Self::MultiplyAdd52Lo => "vector.madd52.lo",
            Self::MultiplyAdd52Hi => "vector.madd52.hi",
            Self::DotProductBf16 => "vector.dp.bf16",
            Self::MultiplyAccumulate => "vector.mac",
            Self::MultiplyAccumulateSat => "vector.mac.sat",
            Self::MultiplyAccumulatePairs => "vector.macd",
            Self::MultiplyAccumulatePairsSat => "vector.macd.sat",
            Self::DotProductS8S8 => "vector.dp.s8s8",
            Self::DotProductS8S8Sat => "vector.dp.s8s8.sat",
            Self::DotProductS8U8 => "vector.dp.s8u8",
            Self::DotProductS8U8Sat => "vector.dp.s8u8.sat",
            Self::DotProductU8U8 => "vector.dp.u8u8",
            Self::DotProductU8U8Sat => "vector.dp.u8u8.sat",
            Self::DotProductS16U16 => "vector.dp.s16u16",
            Self::DotProductS16U16Sat => "vector.dp.s16u16.sat",
            Self::DotProductU16S16 => "vector.dp.u16s16",
            Self::DotProductU16S16Sat => "vector.dp.u16s16.sat",
            Self::DotProductU16U16 => "vector.dp.u16u16",
            Self::DotProductU16U16Sat => "vector.dp.u16u16.sat",
        }
    }
}

/// Structured identity of a typed native **block-string** operation
/// ([`SsaOp::BlockString`]) — the first-class replacement for the
/// `rep`-prefixed compare / scan / load string streams formerly carried
/// opaquely by `NativeOpaque`. (`rep movs`/`rep stos` already lower to the
/// structured `CopyBlk`/`InitBlk` ops and are not represented here.) The kind
/// drives a precise effect summary and a distinct similarity class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlockStringKind {
    /// `rep`/`repe`/`repne cmps*` — streamed element compare (reads two
    /// buffers, sets flags). Reads and writes machine state (counter/pointers).
    Compare,
    /// `rep`/`repe`/`repne scas*` — streamed scan of one buffer against the
    /// accumulator (reads a buffer, sets flags).
    Scan,
    /// `rep lods*` — streamed load of successive elements into the accumulator
    /// (reads a buffer, no flag effect).
    Load,
}

impl BlockStringKind {
    /// Precise effect summary — `Read` for the load stream (`lods`), `ReadWrite`
    /// for the compare / scan streams (they advance counter/pointers and set
    /// flags). Never opaque.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::Load => SsaEffects::new(SsaEffectKind::Read, false),
            Self::Compare | Self::Scan => SsaEffects::new(SsaEffectKind::ReadWrite, false),
        }
    }

    /// Stable display / fingerprint key for this block-string op (used by
    /// [`SsaOp::opcode_name`]).
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Compare => "blockstring.cmps",
            Self::Scan => "blockstring.scas",
            Self::Load => "blockstring.lods",
        }
    }
}

/// Repeat-prefix variant carried by a [`BlockStringOpData`] — preserves the
/// exact `rep` / `repe` / `repne` semantics (the loop-termination condition) so
/// the host can faithfully reconstruct the native mnemonic without re-decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlockStringPrefix {
    /// `rep` — repeat `rcx` times unconditionally (used by `lods`).
    Repeat,
    /// `repe` / `repz` — repeat while equal (ZF=1) and `rcx != 0`.
    RepeatEqual,
    /// `repne` / `repnz` — repeat while not equal (ZF=0) and `rcx != 0`.
    RepeatNotEqual,
}

/// Boxed payload for [`SsaOp::BlockString`]. Mirrors the native-op operand
/// shape — explicit SSA inputs/outputs (advanced counter/pointers, loaded
/// accumulator), clobbered architectural state (memory, flags), and optional
/// source provenance — but the effect summary and similarity class derive from
/// the [`BlockStringKind`], not an echoed opaque blob. Carries the repeat
/// prefix and element width so the native mnemonic round-trips losslessly.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlockStringOpData {
    /// Structured identity of the operation.
    pub kind: BlockStringKind,
    /// Repeat-prefix variant (`rep` / `repe` / `repne`).
    pub prefix: BlockStringPrefix,
    /// Element width streamed by the operation, in bits (8/16/32/64).
    pub element_bits: u16,
    /// Human-readable native mnemonic, for display / provenance only.
    pub mnemonic: String,
    /// Original native instruction metadata when known.
    pub metadata: Option<NativeInstructionMetadata>,
    /// Explicit SSA outputs defined by the operation (advanced counter /
    /// pointers / accumulator).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the operation (buffer addresses, count).
    pub inputs: Vec<SsaVarId>,
    /// Architectural state the operation clobbers (block memory, flags).
    pub clobbers: Vec<NativeClobber>,
}

/// Boxed payload for [`SsaOp::WideCompareExchange`] — the double-width
/// compare-and-swap (`cmpxchg8b` / `cmpxchg16b`) that cannot be expressed as a
/// single-width [`SsaOp::CmpXchg`] (which is fixed-width and pointer-typed).
/// The first-class typed replacement for the wide-CAS case of `NativeOpaque`:
/// it carries explicit `EDX:EAX`-vs-memory expected / `ECX:EBX` desired inputs
/// and `EDX:EAX` readback outputs, with a precise sequentially-consistent
/// atomic effect (never opaque).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WideCmpXchgData {
    /// `true` for the 128-bit `cmpxchg16b`; `false` for the 64-bit `cmpxchg8b`.
    pub wide: bool,
    /// Human-readable native mnemonic, for display / provenance only.
    pub mnemonic: String,
    /// Original native instruction metadata when known.
    pub metadata: Option<NativeInstructionMetadata>,
    /// Explicit SSA outputs (the `EDX:EAX` / `RDX:RAX` readback halves).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs (memory address, expected low/high, desired low/high).
    pub inputs: Vec<SsaVarId>,
    /// Architectural state the operation clobbers (ZF / flags).
    pub clobbers: Vec<NativeClobber>,
}

/// Target register or subregister identity used by native machine-state effects.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
            Self::AndNot => write!(f, "andnot"),
            Self::MinU => write!(f, "minu"),
            Self::MaxU => write!(f, "maxu"),
        }
    }
}

/// Bitmask for selecting flag bits from a flags-defining operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
/// ```rust
/// use analyssa::{ir::{SsaOp, SsaVarId, UnaryOpKind}, MockTarget};
///
/// let op = SsaOp::<MockTarget>::Neg {
///     dest: SsaVarId::from_index(1),
///     operand: SsaVarId::from_index(0),
///     flags: None,
/// };
///
/// // Handle all unary ops uniformly, without matching each variant.
/// let info = op.as_unary_op().expect("Neg is a unary op");
/// assert_eq!(info.kind, UnaryOpKind::Neg);
/// assert_eq!(info.dest, SsaVarId::from_index(1));
/// assert_eq!(info.operand, SsaVarId::from_index(0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Lane element class of a vector operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorElementKind {
    /// Integer lanes.
    Integer,
    /// Floating-point lanes.
    Float,
    /// Element class not recovered by the decoder.
    #[default]
    Unknown,
}

/// Lane element descriptor of a vector operation.
///
/// Carried on the SIMD ops so a host projecting the IR back out (e.g. the Visus
/// cleaned-IR lower leg) can render `paddd` vs `paddb` vs `addps` distinctly. The
/// descriptor travels with the op through normalization, so it is not lost the
/// way a side-table keyed by value id would be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorElement {
    /// Lane element class (integer / floating-point / unknown).
    pub kind: VectorElementKind,
    /// Width of one lane in bits (`0` when not recovered).
    pub bits: u32,
    /// `true` when the op operates on a single (scalar) lane rather than a
    /// packed vector — e.g. `addss`/`addsd` vs `addps`/`addpd`.
    pub scalar: bool,
}

/// Vector unary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Lane-wise floating-point square root.
    Sqrt,
    /// Lane-wise floating-point round to integral value (result stays
    /// floating-point): x87 `frndint`, ARM `vrint*`, AArch64 `frint*`.
    Round,
    /// Lane-wise floating-point reciprocal approximation (`1/x`): x86
    /// `vrcp14*`/`vrcp28*`/`vrcpps`, AArch64 `frecpe`.
    Reciprocal,
    /// Lane-wise floating-point reciprocal-square-root approximation
    /// (`1/sqrt(x)`): x86 `vrsqrt14*`/`vrsqrt28*`/`vrsqrtps`, AArch64 `frsqrte`.
    ReciprocalSqrt,
    /// Lane-wise floating-point exponent extraction (`vgetexp*`): returns the
    /// unbiased exponent as a floating-point value.
    GetExponent,
    /// Lane-wise floating-point mantissa extraction (`vgetmant*`): the
    /// immediate normalization/sign control rides as a parameter.
    GetMantissa,
    /// Lane-wise floating-point round-to-scale (`vrndscale*`): rounds to a
    /// number of fraction bits selected by the control immediate.
    RoundScale,
    /// Lane-wise floating-point reduction (`vreduce*`): `x - round(x)` scaled by
    /// the control immediate.
    Reduce,
    /// Lane-wise identity (pass-through). Used as the operation of a predicated
    /// merge / blend (`vpblendm*`/`vblendm*`), where each active lane copies the
    /// source and inactive lanes take the passthrough.
    Identity,
    /// Lane-wise count of leading zero bits (`vplzcntd`/`vplzcntq`).
    LeadingZeros,
    /// Lane-wise floating-point fractional part `x - trunc(x)` (AMD XOP
    /// `vfrczps`/`vfrczpd`/`vfrczss`/`vfrczsd`).
    Fraction,
    /// Lane-wise conflict detection (`vpconflictd`/`vpconflictq`): each lane
    /// receives a bitmask of the preceding lanes equal to it.
    Conflict,
    /// Lane-wise base-2 exponential `2^x` (`vexp2*`).
    Exp2,
    /// Lane-wise base-2 logarithm `log2(x)` (`vlog2*`).
    Log2,
    /// Lane-wise count of leading sign bits (`vcls`).
    LeadingSignBits,
    /// Lane-wise count of leading one bits (MSA `nloc`).
    LeadingOnes,
    /// Round toward zero to an integral value (truncate; VB6 `Fix`).
    Truncate,
    /// Round down to an integral value (floor, toward −∞; VB6 `Int`).
    Floor,
    /// Sign of the value (`-1` / `0` / `+1`; VB6 `Sgn`).
    Sign,
    /// Lane-wise bit reversal within each element (AArch64 NEON/SVE `rbit`).
    BitReverse,
}

/// Vector binary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Lane-wise bitwise and-not (`a & ~b`).
    AndNot,
    /// Lane-wise multiply keeping the high half of the product.
    MulHigh,
    /// Lane-wise signed multiply keeping the rounded high 16 bits of the
    /// scaled product (`pmulhrsw`: `((a*b >> 14) + 1) >> 1`).
    MulHighRound,
    /// Lane-wise rounding average `(a + b + 1) >> 1`.
    Avg,
    /// Lane-wise absolute difference `|a - b|`.
    AbsDiff,
    /// Lane-wise saturating addition (clamps on overflow).
    SatAdd,
    /// Lane-wise saturating subtraction (clamps on underflow).
    SatSub,
    /// Lane-wise floating-point scale: `a * 2^floor(b)` (x86 `vscalef*`).
    Scale,
    /// Lane-wise floating-point range restriction selecting a min/max-class
    /// result under an immediate control (x86 `vrange*`).
    Range,
    /// Lane-wise bitwise rotate left (x86 `vprol*`, XOP `vprot*`).
    Rol,
    /// Lane-wise bitwise rotate right (x86 `vpror*`).
    Ror,
    /// Lane-wise variable logical shift by a signed count, the sign selecting
    /// direction (AMD XOP `vpshl*`: positive shifts left, negative right).
    VariableShiftLogical,
    /// Lane-wise variable arithmetic shift by a signed count (AMD XOP `vpsha*`).
    VariableShiftArithmetic,
    /// Lane-wise apply-sign: negate / zero / keep the first operand according to
    /// the sign of the second (x86 `psign*`).
    ApplySign,
    /// 3DNow! reciprocal Newton-Raphson refinement step 1 (`pfrcpit1`).
    ReciprocalIter1,
    /// 3DNow! reciprocal Newton-Raphson refinement step 2 (`pfrcpit2`).
    ReciprocalIter2,
    /// 3DNow! reciprocal-square-root Newton-Raphson refinement step 1
    /// (`pfrsqit1`).
    ReciprocalSqrtIter1,
    /// Cyrix EMMI maximum by magnitude (`pmagw`): per lane, keep the operand
    /// with the greater absolute value.
    MaxMagnitude,
    /// Lane-wise negated multiply: `-(a * b)` (AArch64 scalar-FP `fnmul`).
    MulNegate,
    /// Bitwise OR with complemented second operand: `a | ~b` (NEON `vorn`).
    OrNot,
    /// Lane-wise add of absolute values: `|a| + |b|` (MSA `add_a`).
    AddAbs,
    /// Lane-wise integer remainder: `a % b` (MSA `mod_s`/`mod_u`). Sign-agnostic
    /// like [`Self::Div`]; the per-lane signedness comes from the element type.
    Rem,
    /// Lane-wise bitwise NOR: `~(a | b)` (MSA `nor.v`).
    Nor,
}

/// Vector ternary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorTernaryKind {
    /// Lane-wise fused multiply-add.
    Fma,
    /// Lane-wise select controlled by a vector mask.
    Select,
    /// Lane-wise floating-point fix-up under an immediate selector table (x86
    /// `vfixupimm*`): patches special inputs (NaN/inf/zero/…) of the first
    /// operand using the classification of the second and a control immediate.
    FixupImm,
    /// Lane-wise funnel shift left: concatenate the first two operands and shift
    /// left by the third (count), taking the high half (x86 `vpshldv*`).
    FunnelLeft,
    /// Lane-wise funnel shift right: concatenate the first two operands and
    /// shift right by the third (count), taking the low half (x86 `vpshrdv*`).
    FunnelRight,
}

/// Vector comparison operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Lane-wise unordered test (either operand is NaN) — x86 `cmpps` predicate
    /// `UNORD`.
    Unordered,
    /// Lane-wise ordered test (neither operand is NaN) — x86 `cmpps` predicate
    /// `ORD`.
    Ordered,
    /// Lane-wise not-less-than (true when `!(a < b)`, including unordered) —
    /// x86 `cmpps` predicate `NLT`.
    NotLt,
    /// Lane-wise not-less-or-equal (true when `!(a <= b)`, including unordered)
    /// — x86 `cmpps` predicate `NLE`.
    NotLe,
    /// Lane-wise not-greater-or-equal (true when `!(a >= b)`, including
    /// unordered) — x86 `cmpps` predicate `NGE`.
    NotGe,
    /// Lane-wise not-greater-than (true when `!(a > b)`, including unordered) —
    /// x86 `cmpps` predicate `NGT`.
    NotGt,
    /// Lane-wise constant-true predicate — x86 `cmpps` `TRUE`.
    AlwaysTrue,
    /// Lane-wise constant-false predicate — x86 `cmpps` `FALSE`.
    AlwaysFalse,
}

/// Vector cast operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorMaskUnaryKind {
    /// Lane-wise predicate negation.
    Not,
}

/// Vector mask binary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorBitmaskKind {
    /// Extracts each lane's most significant bit, as in x86 `pmovmskb`.
    LaneMostSignificantBits,
    /// Extracts predicate lane bits from a vector mask.
    PredicateBits,
}

/// Controls predicated vector memory behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorMaskMode {
    /// Inactive lanes preserve a passthrough vector value.
    Merge,
    /// Inactive lanes are zeroed or ignored.
    Zero,
}

/// Direction of an AVX-512-style vector lane packing operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorPackKind {
    /// Packs active source lanes contiguously into the low destination lanes.
    Compress,
    /// Expands contiguous low source lanes into active destination lanes.
    Expand,
}

/// Fault behavior for vector loads that may complete partially.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T::Type: serde::Serialize, T::TypeRef: serde::Serialize, \
                     T::MethodRef: serde::Serialize, T::FieldRef: serde::Serialize, \
                     T::SigRef: serde::Serialize",
        deserialize = "T::Type: serde::Deserialize<'de>, T::TypeRef: serde::Deserialize<'de>, \
                       T::MethodRef: serde::Deserialize<'de>, T::FieldRef: serde::Deserialize<'de>, \
                       T::SigRef: serde::Deserialize<'de>"
    ))
)]
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

    /// Integer→integer conversion: `dest = (target_int)operand`.
    ///
    /// Covers widening (`movzx`/`movsx`, CIL `conv.u*`/`conv.i*` from a narrower
    /// source), narrowing (sub-register extraction, `conv.*` from a wider
    /// source), and equal-width reinterpretation — the physical widen/narrow is a
    /// consequence of the source and `target` widths, not a distinct operation,
    /// so it is derived on demand rather than committed at lift time (the lifter
    /// does not always know the source width). `unsigned` selects zero- vs
    /// sign-extension semantics; `overflow_check` is CIL `conv.ovf.*` (may throw).
    IntConv {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target integer type metadata.
        target: T::Type,
        /// Whether the conversion checks overflow (CIL `conv.ovf.*`).
        overflow_check: bool,
        /// Whether the source is interpreted unsigned (zero- vs sign-extension).
        unsigned: bool,
    },

    /// Integer→pointer conversion: `dest = (target_ptr)operand`. The value crosses
    /// from the integer domain into the pointer domain (an address computation
    /// result, or a `conv` to a pointer-typed slot).
    IntToPtr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target pointer type metadata.
        target: T::Type,
    },

    /// Pointer→integer conversion: `dest = (target_int)operand`. The value crosses
    /// from the pointer domain into the integer domain.
    PtrToInt {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target integer type metadata.
        target: T::Type,
    },

    /// Integer→floating-point conversion: `dest = (target_float)operand`
    /// (`sitofp`/`uitofp`, x87 integer load, CIL `conv.r*`). `unsigned` selects
    /// the source interpretation.
    IntToFloat {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target floating-point type metadata.
        target: T::Type,
        /// Whether the integer source is interpreted unsigned.
        unsigned: bool,
    },

    /// Floating-point→integer conversion: `dest = (target_int)operand`
    /// (`fptosi`/`fptoui`, x87 store-as-integer, CIL `conv.ovf.*` from a float).
    /// `unsigned` selects the destination interpretation; `overflow_check` is the
    /// CIL checked form (may throw).
    FloatToInt {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target integer type metadata.
        target: T::Type,
        /// Whether the conversion checks overflow (CIL `conv.ovf.*`).
        overflow_check: bool,
        /// Whether the integer destination is interpreted unsigned.
        unsigned: bool,
    },

    /// Floating-point→floating-point width change: `dest = (target_float)operand`
    /// (`fpext`/`fptrunc`, e.g. `cvtss2sd`).
    FloatConv {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target floating-point type metadata.
        target: T::Type,
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

    /// Scaled pointer address computation: `dest = base + index*stride + offset`.
    ///
    /// Models a native memory-addressing form (`[base + index*scale + disp]`) as
    /// one structural op instead of a shredded `Shl`/`Mul`/`Add` chain, so a
    /// field access reads as a field access and array indexing as indexing.
    /// `index` is absent for a plain `base + offset`; `stride` is the index
    /// scale in bytes and `offset` the signed byte displacement. `result_type`
    /// carries the pointer type the address evaluates to.
    PtrAdd {
        /// Destination SSA variable (a pointer).
        dest: SsaVarId,
        /// Base address operand.
        base: SsaVarId,
        /// Scaled index operand, when the address adds one.
        index: Option<SsaVarId>,
        /// Scale applied to `index`, in bytes.
        stride: u64,
        /// Constant displacement from the base, in bytes.
        offset: i64,
        /// Pointer type metadata the address evaluates to.
        result_type: T::Type,
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
        /// Lane element descriptor (class / width / scalar).
        element: VectorElement,
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
        /// Lane element descriptor (class / width / scalar).
        element: VectorElement,
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

    /// Recognized native intrinsic with explicit inputs/outputs, clobbers, and a
    /// conservative effect summary, plus a structured [`NativeIntrinsicId`]. Used
    /// for hardware ops with no primitive closed form but faithful dataflow
    /// (`cpuid`/`rdtsc`/`pdep`/`pext`/`crc32`/`rdrand`/PAC/…). The payload is
    /// boxed (see [`NativeIntrinsicData`]) to keep `SsaOp` small.
    NativeIntrinsic(Box<NativeIntrinsicData>),

    /// Typed native **system / privileged** operation (`cpuid`, time-stamp reads,
    /// system/control-register access, syscalls, traps, cache/TLB maintenance,
    /// privileged state ops). The first-class typed replacement for the system
    /// cases of `NativeIntrinsic` / `NativeOpaque`: a structured
    /// [`SystemOpKind`] identity drives a precise effect summary and a distinct
    /// similarity class. The payload is boxed (see [`NativeKindedData`]).
    SystemOp(Box<NativeKindedData<SystemOpKind>>),

    /// Typed native **compute** intrinsic (`pdep`/`pext`, `crc32`,
    /// `rdrand`/`rdseed`, pointer authentication). The first-class typed
    /// replacement for the compute cases of `NativeIntrinsic`: a structured
    /// [`ComputeKind`] identity drives a precise effect summary (pure, except
    /// nondeterministic random sources) and a distinct similarity class. The
    /// payload is boxed (see [`NativeKindedData`]).
    ComputeOp(Box<NativeKindedData<ComputeKind>>),

    /// Typed legacy x86 **binary-coded-decimal adjust**
    /// (`daa`/`das`/`aaa`/`aas`/`aam`/`aad`). The [`BcdAdjustKind`] identity
    /// drives a pure effect summary and an arithmetic similarity class; the
    /// accumulator and flags flow through as typed SSA values. The payload is
    /// boxed (see [`BcdAdjustData`]).
    BcdAdjust(Box<BcdAdjustData>),

    /// Typed hardware **vector cryptographic** operation (AES / SHA / SM3 /
    /// SM4 / GF(2^8) / carry-less multiply). The first-class, named replacement
    /// for what would otherwise be an opaque crypto blob: a [`VectorCryptoKind`]
    /// identity drives a precise (pure) effect summary and the `Vector`
    /// similarity class. The payload is boxed (see [`KindedVecData`]).
    VectorCrypto(Box<KindedVecData<VectorCryptoKind>>),

    /// Typed AMX **tile** operation (matrix multiply-accumulate, tile load /
    /// store / zero / release, configuration load / store). A [`TileOpKind`]
    /// identity drives a precise effect summary (pure / read / write). The
    /// payload is boxed (see [`KindedVecData`]).
    TileOp(Box<KindedVecData<TileOpKind>>),

    /// Typed vector **lane permute by a runtime index vector** (x86
    /// `vpermd`/`vpermps`/`vpermb`/`vpshufb`/`vpermt2*`/`vpermi2*` and the
    /// variable `vpermilps`/`vpermilpd`). Distinct from [`Self::VectorShuffle`]
    /// (static mask). Pure. The payload is boxed (see [`VectorPermuteData`]).
    VectorPermute(Box<VectorPermuteData>),

    /// Typed vector **fused multiply-then-horizontal-add** (x86 `pmaddwd`/
    /// `pmaddubsw`/`vpdpbusd[s]`/`vpdpwssd[s]`/`vpmadd52luq`/`vpmadd52huq`).
    /// Multiplies two source vectors and horizontally sums adjacent products,
    /// optionally accumulating into a running destination. Pure. The payload is
    /// boxed (see [`KindedVecData`]).
    VectorMultiplyAdd(Box<KindedVecData<VectorMaddKind>>),

    /// Typed vector **two-source saturating narrowing pack** (x86 `packsswb`/
    /// `packssdw`/`packuswb`/`packusdw`). Pure. The payload is boxed (see
    /// [`VectorPackNarrowData`]).
    VectorPackNarrow(Box<VectorPackNarrowData>),
    /// Single-source saturating narrowing (see [`VectorNarrowSaturateData`]).
    VectorNarrowSaturate(Box<VectorNarrowSaturateData>),
    /// Predicate-generation from scalar loop bounds (see [`VectorPredicateWhileData`]).
    VectorPredicateWhile(Box<VectorPredicateWhileData>),
    /// Predicate-break generation (see [`VectorPredicateBreakData`]).
    VectorPredicateBreak(Box<VectorPredicateBreakData>),
    /// Complex-number add with rotation (see [`VectorComplexAddData`]).
    VectorComplexAdd(Box<VectorComplexAddData>),
    /// Adjust a register by an implicit element/predicate count (see [`VectorCountAdjustData`]).
    VectorCountAdjust(Box<VectorCountAdjustData>),
    /// Sign/zero-extend the low field of each lane in place (see [`VectorExtendInLaneData`]).
    VectorExtendInLane(Box<VectorExtendInLaneData>),
    /// Read the symbolic element count `VL / element_bits` into a scalar (see
    /// [`VectorElementCountData`]).
    VectorElementCount(Box<VectorElementCountData>),
    /// SVE vector address generation `base + (extend(index) << shift)` (see
    /// [`VectorSveAddressGenData`]).
    VectorSveAddressGen(Box<VectorSveAddressGenData>),
    /// Manipulate the condition flags directly (see [`KindedVecData`]).
    FlagAdjust(Box<KindedVecData<FlagAdjustKind>>),
    /// Load-and-replicate a 2/3/4-element structure (see [`VectorStructLoadReplicateData`]).
    VectorStructLoadReplicate(Box<VectorStructLoadReplicateData>),
    /// SME ZA-tile accumulate/zero (see [`VectorSmeMiscData`]).
    VectorSmeMisc(Box<VectorSmeMiscData>),
    /// SVE predicate/first-fault-register operation (see [`VectorPredicateOpData`]).
    VectorPredicateOp(Box<VectorPredicateOpData>),
    /// SVE2/NEON compute op named by its kind (see [`VectorSveComputeData`]).
    VectorSveCompute(Box<VectorSveComputeData>),
    /// Reverse chunks within each element (see [`VectorReverseChunksData`]).
    VectorReverseChunks(Box<VectorReverseChunksData>),
    /// Matrix multiply-accumulate over vectors (see [`VectorMatrixMulAccData`]).
    VectorMatrixMulAcc(Box<VectorMatrixMulAccData>),
    /// SME outer-product accumulate into a ZA tile (see [`VectorSmeOuterProductData`]).
    VectorSmeOuterProduct(Box<VectorSmeOuterProductData>),
    /// Predicate generation (const/iterate/ffr/unpack/select; see [`KindedVecData`]).
    VectorPredicateGen(Box<KindedVecData<PredicateGenKind>>),
    /// SVE floating-point transcendental helper (see [`KindedVecData`]).
    VectorFpHelper(Box<KindedVecData<FpHelperKind>>),
    /// SVE data-movement permute/extract (see [`KindedVecData`]).
    VectorSvePermute(Box<KindedVecData<SvePermuteKind>>),

    /// Typed vector **arbitrary three-input bitwise logic** (x86 `vpternlogd`/
    /// `vpternlogq`), selected by an 8-bit truth table. Pure. The payload is
    /// boxed (see [`VecImm8Data`]).
    VectorTernaryLogic(Box<VecImm8Data>),

    /// Typed vector **floating-point dot product** (x86 SSE4.1 `dpps`/`dppd`,
    /// VEX `vdpps`/`vdppd`), selected by an 8-bit lane-participation / result-
    /// broadcast immediate. Pure. The payload is boxed (see
    /// [`VectorDotProductData`]).
    VectorDotProduct(Box<VectorDotProductData>),

    /// Typed vector **multi-block sum of absolute differences** (x86 SSE4.1
    /// `mpsadbw`, VEX `vmpsadbw`, AVX-512 `vdbpsadbw`), selected by an 8-bit
    /// block-offset immediate. Pure. The payload is boxed (see
    /// [`VecImm8Data`]).
    VectorMultiSad(Box<VecImm8Data>),

    /// Typed vector **integer dot-product-accumulate** (ARM/AArch64 `sdot`/
    /// `udot`/`usdot`): each lane accumulates the sum of a group of widened
    /// integer element products. Pure. The payload is boxed (see
    /// [`VectorIntDotProductData`]).
    VectorIntDotProduct(Box<VectorIntDotProductData>),

    /// Typed vector **packed string comparison** (SSE4.2 `pcmpestri`/
    /// `pcmpestrm`/`pcmpistri`/`pcmpistrm` and VEX peers), selected by an 8-bit
    /// format / aggregation / polarity immediate. Pure. The payload is boxed
    /// (see [`VectorStringCompareData`]).
    VectorStringCompare(Box<VectorStringCompareData>),

    /// Typed vector **bit-field extract / insert** over the low 64 bits of a
    /// vector register (SSE4a `extrq`/`insertq`). Pure. The payload is boxed
    /// (see [`VectorBitfieldData`]).
    VectorBitfield(Box<VectorBitfieldData>),

    /// Typed vector **element-intersection to a mask pair** (AVX-512
    /// `vp2intersectd`/`vp2intersectq`). Pure. The payload is boxed (see
    /// [`VectorIntersectData`]).
    VectorIntersect(Box<VectorIntersectData>),

    /// Typed vector **bit-shuffle to a mask** (AVX-512 `vpshufbitqmb`). Pure.
    /// The payload is boxed (see [`VectorShuffleBitsData`]).
    VectorShuffleBits(Box<VectorShuffleBitsData>),

    /// Typed vector **per-byte conditional move** (Cyrix EMMI `pmvzb`/`pmvnzb`/
    /// `pmvlzb`/`pmvgezb`). Pure. The payload is boxed (see
    /// [`VectorConditionalMoveData`]).
    VectorConditionalMove(Box<VectorConditionalMoveData>),

    /// Typed vector **horizontal minimum with position** (SSE4.1 `phminposuw`):
    /// the smallest unsigned 16-bit lane and its source index. Pure. The
    /// payload is boxed (see [`VectorHorizontalMinPosData`]).
    VectorHorizontalMinPos(Box<VectorHorizontalMinPosData>),

    /// Typed vector **complex-number floating-point multiply** (x86 `vfmulcph`/
    /// `vfcmulcph`/`vfmaddcph`/`vfcmaddcph` and `sh` peers). Pure. The payload
    /// is boxed (see [`KindedVecData`]).
    VectorComplexMul(Box<KindedVecData<ComplexMulKind>>),

    /// Typed vector **floating-point lane classification to a mask** (x86
    /// `vfpclass*`), selected by an 8-bit category immediate. Pure. The payload
    /// is boxed (see [`VecImm8Data`]).
    VectorClassify(Box<VecImm8Data>),

    /// Typed vector **grouped widening horizontal add / subtract** (AMD XOP
    /// `vphadd*`/`vphsub*`). Pure. The payload is boxed (see
    /// [`VectorHorizontalReduceData`]).
    VectorHorizontalReduce(Box<VectorHorizontalReduceData>),

    /// Typed native **block-string** operation (`rep`/`repe`/`repne`
    /// `cmps`/`scas`/`lods`). The first-class typed replacement for the
    /// rep-string-stream case of `NativeOpaque`: a structured
    /// [`BlockStringKind`] identity drives a precise effect summary (memory
    /// read / read-write) and a distinct similarity class. The payload is boxed
    /// (see [`BlockStringOpData`]). (`rep movs`/`rep stos` use `CopyBlk`/
    /// `InitBlk`, not this op.)
    BlockString(Box<BlockStringOpData>),

    /// Typed native **wide compare-and-swap** (`cmpxchg8b` / `cmpxchg16b`). The
    /// first-class typed replacement for the wide-CAS case of `NativeOpaque`: a
    /// sequentially-consistent atomic op with explicit register-pair inputs /
    /// outputs (see [`WideCmpXchgData`]). Single-width CAS uses [`Self::CmpXchg`].
    WideCompareExchange(Box<WideCmpXchgData>),

    /// Typed native **flags computation** — defines an architectural-flags
    /// value (`EFLAGS` / NZCV) as a pure function of its inputs, for operations
    /// whose precise per-flag semantics the lifter does not decompose (`bsf`/
    /// `bsr`/`popcnt`/`bt` zero-flag side effects). The first-class typed
    /// replacement for the flags-only case of `NativeOpaque`: it is pure and
    /// deterministic, so optimization can still reason about and eliminate it.
    ComputeFlags {
        /// The defined architectural-flags value.
        dest: SsaVarId,
        /// The input values the flags are computed from.
        inputs: Vec<SsaVarId>,
    },

    /// Typed native **call-clobber** marker — defines fresh, undefined values
    /// for the caller-saved registers a preceding [`Self::Call`] clobbers (the
    /// `Call` op carries a single `dest`, so the remaining clobbered registers
    /// need an owning def for the verifier). The first-class typed replacement
    /// for the call-clobber case of `NativeOpaque`: pure (the call already
    /// happened) and freely eliminable when its outputs are unread.
    CallClobber {
        /// The caller-saved register values invalidated by the call.
        outputs: Vec<SsaVarId>,
    },

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

    /// Classify a floating-point value, producing an integer mask describing the
    /// operand's IEEE-754 category (RISC-V `fclass`, MIPS `class.fmt`). The exact
    /// bit layout is the native instruction's; the result is a plain integer.
    FpClassify {
        /// Destination SSA variable (integer classification mask).
        dest: SsaVarId,
        /// Floating-point operand variable.
        operand: SsaVarId,
    },

    /// Hardware floating-point transcendental / residue op with no primitive
    /// closed form (x87 `fsin`/`fcos`/`fpatan`/`f2xm1`/`fyl2x`/`fprem`/…). The
    /// payload ([`KindedVecData`]) is boxed to keep `SsaOp` compact.
    FpTranscendental(Box<KindedVecData<TranscendentalKind>>),

    /// Floating-point unit control / state op (x87 `fldcw`/`fnstcw`/`fnstsw`/
    /// `fldenv`/`fnsave`/`frstor`/`fnclex`/`fdecstp`/`ffree`/`fxsave`/…). The
    /// payload ([`KindedVecData`]) is boxed to keep `SsaOp` compact.
    FpuControl(Box<KindedVecData<FpuControlKind>>),

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
        let mut dest = None;
        self.visit_operands(|role, var| {
            if dest.is_none() && matches!(role, OperandRole::Def) {
                dest = Some(var);
            }
        });
        dest
    }

    /// Returns all variables defined by this operation.
    #[must_use]
    pub fn defs(&self) -> SsaDefs<'_> {
        match self {
            Self::NativeOpaque(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::NativeIntrinsic(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::SystemOp(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::ComputeOp(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::BcdAdjust(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorCrypto(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::TileOp(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPermute(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorMultiplyAdd(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPackNarrow(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorNarrowSaturate(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPredicateWhile(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPredicateBreak(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorComplexAdd(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorCountAdjust(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorExtendInLane(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorElementCount(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSveAddressGen(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::FlagAdjust(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorStructLoadReplicate(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSmeMisc(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPredicateOp(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSveCompute(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorReverseChunks(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorMatrixMulAcc(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSmeOuterProduct(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPredicateGen(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorFpHelper(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSvePermute(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorTernaryLogic(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorDotProduct(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorMultiSad(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorIntDotProduct(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorStringCompare(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorBitfield(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorIntersect(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorShuffleBits(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorConditionalMove(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorHorizontalMinPos(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorComplexMul(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorClassify(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorHorizontalReduce(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::BlockString(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::WideCompareExchange(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::CallClobber { outputs } => SsaDefs::new(None, None, Some(outputs)),
            Self::ComputeFlags { dest, .. } => SsaDefs::new(Some(*dest), None, None),
            Self::FpTranscendental(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::FpuControl(data) => SsaDefs::new(None, None, Some(&data.outputs)),
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
            | Self::IntConv { dest, .. }
            | Self::IntToPtr { dest, .. }
            | Self::PtrToInt { dest, .. }
            | Self::IntToFloat { dest, .. }
            | Self::FloatToInt { dest, .. }
            | Self::FloatConv { dest, .. }
            | Self::Bitcast { dest, .. }
            | Self::LoadField { dest, .. }
            | Self::LoadStaticField { dest, .. }
            | Self::LoadFieldAddr { dest, .. }
            | Self::LoadStaticFieldAddr { dest, .. }
            | Self::LoadElement { dest, .. }
            | Self::LoadElementAddr { dest, .. }
            | Self::PtrAdd { dest, .. }
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
            | Self::FpClassify { dest, .. }
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
            | Self::ComputeFlags { dest, .. }
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
            Self::CallClobber { outputs } => {
                if let Some(first) = outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::NativeIntrinsic(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::SystemOp(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::ComputeOp(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::BcdAdjust(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorCrypto(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::TileOp(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPermute(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorMultiplyAdd(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPackNarrow(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorNarrowSaturate(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPredicateWhile(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPredicateBreak(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorComplexAdd(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorCountAdjust(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorExtendInLane(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorElementCount(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSveAddressGen(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::FlagAdjust(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorStructLoadReplicate(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSmeMisc(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPredicateOp(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSveCompute(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorReverseChunks(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorMatrixMulAcc(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSmeOuterProduct(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPredicateGen(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorFpHelper(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSvePermute(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorTernaryLogic(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorDotProduct(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorMultiSad(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorIntDotProduct(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorStringCompare(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorBitfield(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorIntersect(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorShuffleBits(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorConditionalMove(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorHorizontalMinPos(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorComplexMul(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorClassify(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorHorizontalReduce(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::BlockString(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::WideCompareExchange(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::FpTranscendental(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::FpuControl(data) => {
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
    /// Replaces every definition equal to `old_var` — the primary
    /// destination, secondary outputs (high halves, status/fault outputs,
    /// native output lists), and flag outputs. Returns `true` when at least
    /// one definition was changed. This is used by SSA renaming, where
    /// definitions and uses have different scoping rules.
    pub fn replace_def(&mut self, old_var: SsaVarId, new_var: SsaVarId) -> bool {
        let mut changed = false;
        self.visit_operands_mut(|role, var| {
            if matches!(role, OperandRole::Def | OperandRole::FlagsDef) && *var == old_var {
                *var = new_var;
                changed = true;
            }
        });
        changed
    }

    /// Returns the flags destination if this operation defines flags.
    pub fn flags_dest(&self) -> Option<SsaVarId> {
        let mut flags = None;
        self.visit_operands(|role, var| {
            if flags.is_none() && matches!(role, OperandRole::FlagsDef) {
                flags = Some(var);
            }
        });
        flags
    }

    /// Visits every SSA variable operand together with its [`OperandRole`].
    ///
    /// Operands are visited in payload order: definitions first (the primary
    /// destination, then secondary and flag outputs), then uses in payload
    /// order. This is the single source of truth for operand traversal —
    /// [`Self::dest`], [`Self::flags_dest`], [`Self::for_each_use`],
    /// [`Self::replace_uses`], and [`Self::replace_def`] are expressed over
    /// it.
    #[allow(clippy::match_same_arms)] // Kept separate for clarity by operation category
    pub fn visit_operands<F>(&self, mut f: F)
    where
        F: FnMut(OperandRole, SsaVarId),
    {
        match self {
            Self::PtrAdd {
                dest, base, index, ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *base);
                if let Some(index) = index {
                    f(OperandRole::Use, *index);
                }
            }
            Self::Const { dest, .. }
            | Self::LoadStaticField { dest, .. }
            | Self::LoadStaticFieldAddr { dest, .. }
            | Self::SizeOf { dest, .. }
            | Self::LoadToken { dest, .. }
            | Self::LoadFunctionPtr { dest, .. }
            | Self::LoadArg { dest, .. }
            | Self::LoadLocal { dest, .. }
            | Self::LoadArgAddr { dest, .. }
            | Self::LoadLocalAddr { dest, .. } => f(OperandRole::Def, *dest),

            Self::ComputeFlags { dest, inputs } => {
                f(OperandRole::Def, *dest);
                for input in inputs {
                    f(OperandRole::Use, *input);
                }
            }

            Self::CallClobber { outputs } => {
                for output in outputs {
                    f(OperandRole::Def, *output);
                }
            }

            Self::Add {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::AddOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Sub {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::SubOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Mul {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::MulOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Div {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Rem {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::And {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Or {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Xor {
                dest,
                left,
                right,
                flags,
                ..
            } => {
                f(OperandRole::Def, *dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, *flags_v);
                }
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
            }

            Self::Neg {
                dest,
                operand,
                flags,
                ..
            }
            | Self::Not {
                dest,
                operand,
                flags,
                ..
            } => {
                f(OperandRole::Def, *dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, *flags_v);
                }
                f(OperandRole::Use, *operand);
            }

            Self::Shl {
                dest,
                value,
                amount,
                flags,
                ..
            }
            | Self::Shr {
                dest,
                value,
                amount,
                flags,
                ..
            } => {
                f(OperandRole::Def, *dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, *flags_v);
                }
                f(OperandRole::Use, *value);
                f(OperandRole::Use, *amount);
            }

            Self::Rol {
                dest,
                value,
                amount,
                ..
            }
            | Self::Ror {
                dest,
                value,
                amount,
                ..
            }
            | Self::Rcl {
                dest,
                value,
                amount,
                ..
            }
            | Self::Rcr {
                dest,
                value,
                amount,
                ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *value);
                f(OperandRole::Use, *amount);
            }

            Self::WideMul {
                low,
                high,
                left,
                right,
                ..
            } => {
                f(OperandRole::Def, *low);
                f(OperandRole::Def, *high);
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
            }

            Self::WideDiv {
                quotient,
                remainder,
                high,
                low,
                divisor,
                ..
            } => {
                f(OperandRole::Def, *quotient);
                f(OperandRole::Def, *remainder);
                f(OperandRole::Use, *high);
                f(OperandRole::Use, *low);
                f(OperandRole::Use, *divisor);
            }

            Self::Ceq {
                dest, left, right, ..
            }
            | Self::Clt {
                dest, left, right, ..
            }
            | Self::Cgt {
                dest, left, right, ..
            }
            | Self::BoolAnd {
                dest, left, right, ..
            }
            | Self::BoolOr {
                dest, left, right, ..
            }
            | Self::BoolXor {
                dest, left, right, ..
            }
            | Self::VectorBinary {
                dest, left, right, ..
            }
            | Self::VectorCompare {
                dest, left, right, ..
            }
            | Self::VectorMaskBinary {
                dest, left, right, ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
            }

            Self::FloatCompareFlags {
                flags, left, right, ..
            } => {
                f(OperandRole::FlagsDef, *flags);
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
            }

            Self::BranchCmp { left, right, .. } => {
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
            }

            Self::BoolNot { dest, value, .. }
            | Self::Copy {
                dest, src: value, ..
            }
            | Self::IntConv {
                dest,
                operand: value,
                ..
            }
            | Self::IntToPtr {
                dest,
                operand: value,
                ..
            }
            | Self::PtrToInt {
                dest,
                operand: value,
                ..
            }
            | Self::IntToFloat {
                dest,
                operand: value,
                ..
            }
            | Self::FloatToInt {
                dest,
                operand: value,
                ..
            }
            | Self::FloatConv {
                dest,
                operand: value,
                ..
            }
            | Self::Bitcast {
                dest,
                operand: value,
                ..
            }
            | Self::Ckfinite {
                dest,
                operand: value,
                ..
            }
            | Self::FpClassify {
                dest,
                operand: value,
                ..
            }
            | Self::BSwap {
                dest, src: value, ..
            }
            | Self::BRev {
                dest, src: value, ..
            }
            | Self::BitScanForward {
                dest, src: value, ..
            }
            | Self::BitScanReverse {
                dest, src: value, ..
            }
            | Self::Popcount {
                dest, src: value, ..
            }
            | Self::Parity {
                dest, src: value, ..
            }
            | Self::LoadField {
                dest,
                object: value,
                ..
            }
            | Self::LoadFieldAddr {
                dest,
                object: value,
                ..
            }
            | Self::ArrayLength {
                dest, array: value, ..
            }
            | Self::LoadIndirect {
                dest, addr: value, ..
            }
            | Self::AtomicLoad {
                dest, addr: value, ..
            }
            | Self::VectorUnary { dest, value, .. }
            | Self::VectorSplat { dest, value, .. }
            | Self::VectorCast { dest, value, .. }
            | Self::VectorReinterpret { dest, value, .. }
            | Self::VectorLoad {
                dest, addr: value, ..
            }
            | Self::VectorBroadcastLoad {
                dest, addr: value, ..
            }
            | Self::VectorExtract {
                dest,
                vector: value,
                ..
            }
            | Self::VectorMaskUnary {
                dest, mask: value, ..
            }
            | Self::VectorReduce { dest, value, .. }
            | Self::VectorBitmask { dest, value, .. }
            | Self::NewArr {
                dest,
                length: value,
                ..
            }
            | Self::Box { dest, value, .. }
            | Self::LoadVirtFunctionPtr {
                dest,
                object: value,
                ..
            }
            | Self::LocalAlloc {
                dest, size: value, ..
            }
            | Self::CastClass {
                dest,
                object: value,
                ..
            }
            | Self::IsInst {
                dest,
                object: value,
                ..
            }
            | Self::Unbox {
                dest,
                object: value,
                ..
            }
            | Self::UnboxAny {
                dest,
                object: value,
                ..
            }
            | Self::LoadObj {
                dest,
                src_addr: value,
                ..
            }
            | Self::ReadFlags {
                dest, flags: value, ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *value);
            }

            Self::Branch {
                condition: value, ..
            }
            | Self::Switch { value, .. }
            | Self::StoreStaticField { value, .. }
            | Self::Pop { value }
            | Self::Throw { exception: value }
            | Self::EndFilter { result: value }
            | Self::InitObj {
                dest_addr: value, ..
            }
            | Self::IndirectBranch { target: value, .. }
            | Self::BranchFlags { flags: value, .. } => f(OperandRole::Use, *value),

            Self::LoadElement {
                dest,
                array: a,
                index: b,
                ..
            }
            | Self::LoadElementAddr {
                dest,
                array: a,
                index: b,
                ..
            }
            | Self::AtomicRmw {
                dest,
                addr: a,
                value: b,
                ..
            }
            | Self::AtomicExchange {
                dest,
                addr: a,
                value: b,
                ..
            }
            | Self::AtomicLockRmw {
                dest,
                addr: a,
                value: b,
                ..
            }
            | Self::AtomicStoreConditional {
                status: dest,
                addr: a,
                value: b,
                ..
            }
            | Self::VectorInsert {
                dest,
                vector: a,
                value: b,
                ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *a);
                f(OperandRole::Use, *b);
            }

            Self::StoreField {
                object: a,
                value: b,
                ..
            }
            | Self::StoreIndirect {
                addr: a, value: b, ..
            }
            | Self::AtomicStore {
                addr: a, value: b, ..
            }
            | Self::VectorStore {
                addr: a, value: b, ..
            }
            | Self::CopyObj {
                dest_addr: a,
                src_addr: b,
                ..
            }
            | Self::StoreObj {
                dest_addr: a,
                value: b,
                ..
            } => {
                f(OperandRole::Use, *a);
                f(OperandRole::Use, *b);
            }

            Self::Select {
                dest,
                condition: a,
                true_val: b,
                false_val: c,
                ..
            }
            | Self::CmpXchg {
                dest,
                addr: a,
                expected: b,
                desired: c,
                ..
            }
            | Self::VectorTernary {
                dest,
                first: a,
                second: b,
                third: c,
                ..
            }
            | Self::AtomicPairStoreConditional {
                status: dest,
                addr: a,
                first_value: b,
                second_value: c,
                ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *a);
                f(OperandRole::Use, *b);
                f(OperandRole::Use, *c);
            }

            Self::StoreElement {
                array: a,
                index: b,
                value: c,
                ..
            }
            | Self::VectorMaskedStore {
                addr: a,
                value: b,
                mask: c,
                ..
            }
            | Self::VectorPackStore {
                addr: a,
                value: b,
                mask: c,
                ..
            }
            | Self::InitBlk {
                dest_addr: a,
                value: b,
                size: c,
                ..
            }
            | Self::CopyBlk {
                dest_addr: a,
                src_addr: b,
                size: c,
                ..
            } => {
                f(OperandRole::Use, *a);
                f(OperandRole::Use, *b);
                f(OperandRole::Use, *c);
            }

            Self::VectorScatter {
                base: a,
                indices: b,
                value: c,
                mask: d,
                ..
            } => {
                f(OperandRole::Use, *a);
                f(OperandRole::Use, *b);
                f(OperandRole::Use, *c);
                f(OperandRole::Use, *d);
            }

            Self::AtomicCmpXchg {
                old,
                success,
                addr,
                expected,
                desired,
                ..
            } => {
                f(OperandRole::Def, *old);
                if let Some(success_v) = success {
                    f(OperandRole::Def, *success_v);
                }
                f(OperandRole::Use, *addr);
                f(OperandRole::Use, *expected);
                f(OperandRole::Use, *desired);
            }

            Self::AtomicPairLoad {
                first,
                second,
                addr,
                ..
            } => {
                f(OperandRole::Def, *first);
                f(OperandRole::Def, *second);
                f(OperandRole::Use, *addr);
            }

            Self::AtomicPairCmpXchg {
                old_first,
                old_second,
                addr,
                expected_first,
                expected_second,
                desired_first,
                desired_second,
                ..
            } => {
                f(OperandRole::Def, *old_first);
                f(OperandRole::Def, *old_second);
                f(OperandRole::Use, *addr);
                f(OperandRole::Use, *expected_first);
                f(OperandRole::Use, *expected_second);
                f(OperandRole::Use, *desired_first);
                f(OperandRole::Use, *desired_second);
            }

            Self::VectorPredicatedUnary {
                dest,
                value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorMaskedLoad {
                dest,
                addr: value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorPack {
                dest,
                value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorPackLoad {
                dest,
                addr: value,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *value);
                f(OperandRole::Use, *mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v);
                }
            }

            Self::VectorPredicatedBinary {
                dest,
                left,
                right,
                mask,
                passthrough,
                ..
            }
            | Self::VectorGather {
                dest,
                base: left,
                indices: right,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
                f(OperandRole::Use, *mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v);
                }
            }

            Self::VectorPredicatedTernary {
                dest,
                first,
                second,
                third,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *first);
                f(OperandRole::Use, *second);
                f(OperandRole::Use, *third);
                f(OperandRole::Use, *mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v);
                }
            }

            Self::VectorFaultingLoad {
                dest,
                fault,
                addr,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, *dest);
                if let Some(fault_v) = fault {
                    f(OperandRole::Def, *fault_v);
                }
                f(OperandRole::Use, *addr);
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, *mask_v);
                }
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v);
                }
            }

            Self::VectorSegmentLoad {
                dests, base, mask, ..
            } => {
                for item in dests {
                    f(OperandRole::Def, *item);
                }
                f(OperandRole::Use, *base);
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, *mask_v);
                }
            }

            Self::VectorSegmentStore {
                base, values, mask, ..
            } => {
                f(OperandRole::Use, *base);
                for item in values {
                    f(OperandRole::Use, *item);
                }
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, *mask_v);
                }
            }

            Self::VectorShuffle {
                dest, left, right, ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *left);
                if let Some(right_v) = right {
                    f(OperandRole::Use, *right_v);
                }
            }

            Self::NativeOpaque(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }

            Self::NativeIntrinsic(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::SystemOp(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::ComputeOp(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::BcdAdjust(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorCrypto(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::TileOp(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPermute(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorMultiplyAdd(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPackNarrow(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorNarrowSaturate(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPredicateWhile(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPredicateBreak(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorComplexAdd(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorCountAdjust(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorExtendInLane(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorElementCount(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
            }
            Self::VectorSveAddressGen(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::FlagAdjust(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorStructLoadReplicate(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorSmeMisc(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPredicateOp(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorSveCompute(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorReverseChunks(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorMatrixMulAcc(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorSmeOuterProduct(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPredicateGen(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorFpHelper(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorSvePermute(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorTernaryLogic(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorDotProduct(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorMultiSad(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorIntDotProduct(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorStringCompare(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorBitfield(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorIntersect(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorShuffleBits(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorConditionalMove(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorHorizontalMinPos(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorComplexMul(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorClassify(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorHorizontalReduce(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::BlockString(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::WideCompareExchange(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }

            Self::FpTranscendental(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }

            Self::FpuControl(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }

            Self::NewObj { dest, args, .. } => {
                f(OperandRole::Def, *dest);
                for item in args {
                    f(OperandRole::Use, *item);
                }
            }

            Self::Call { dest, args, .. } | Self::CallVirt { dest, args, .. } => {
                if let Some(dest_v) = dest {
                    f(OperandRole::Def, *dest_v);
                }
                for item in args {
                    f(OperandRole::Use, *item);
                }
            }

            Self::CallIndirect {
                dest, fptr, args, ..
            } => {
                if let Some(dest_v) = dest {
                    f(OperandRole::Def, *dest_v);
                }
                f(OperandRole::Use, *fptr);
                for item in args {
                    f(OperandRole::Use, *item);
                }
            }

            Self::Return { value } => {
                if let Some(value_v) = value {
                    f(OperandRole::Use, *value_v);
                }
            }

            Self::Phi { dest, operands } => {
                f(OperandRole::Def, *dest);
                for (_, item) in operands {
                    f(OperandRole::Use, *item);
                }
            }

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
            | Self::VectorZeroUpper { .. }
            | Self::Unreachable => {}
        }
    }

    /// Visits every SSA variable operand mutably together with its [`OperandRole`].
    ///
    /// The mutable counterpart of [`Self::visit_operands`], visiting the same
    /// operands in the same order. Substitution passes ([`Self::replace_uses`],
    /// [`Self::replace_def`]) rewrite variables in place through it.
    #[allow(clippy::match_same_arms)] // Kept separate for clarity by operation category
    pub fn visit_operands_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(OperandRole, &mut SsaVarId),
    {
        match self {
            Self::PtrAdd {
                dest, base, index, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, base);
                if let Some(index) = index {
                    f(OperandRole::Use, index);
                }
            }
            Self::Const { dest, .. }
            | Self::LoadStaticField { dest, .. }
            | Self::LoadStaticFieldAddr { dest, .. }
            | Self::SizeOf { dest, .. }
            | Self::LoadToken { dest, .. }
            | Self::LoadFunctionPtr { dest, .. }
            | Self::LoadArg { dest, .. }
            | Self::LoadLocal { dest, .. }
            | Self::LoadArgAddr { dest, .. }
            | Self::LoadLocalAddr { dest, .. } => f(OperandRole::Def, dest),

            Self::ComputeFlags { dest, inputs } => {
                f(OperandRole::Def, dest);
                for input in inputs {
                    f(OperandRole::Use, input);
                }
            }

            Self::CallClobber { outputs } => {
                for output in outputs {
                    f(OperandRole::Def, output);
                }
            }

            Self::Add {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::AddOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Sub {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::SubOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Mul {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::MulOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Div {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Rem {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::And {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Or {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Xor {
                dest,
                left,
                right,
                flags,
                ..
            } => {
                f(OperandRole::Def, dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, flags_v);
                }
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
            }

            Self::Neg {
                dest,
                operand,
                flags,
                ..
            }
            | Self::Not {
                dest,
                operand,
                flags,
                ..
            } => {
                f(OperandRole::Def, dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, flags_v);
                }
                f(OperandRole::Use, operand);
            }

            Self::Shl {
                dest,
                value,
                amount,
                flags,
                ..
            }
            | Self::Shr {
                dest,
                value,
                amount,
                flags,
                ..
            } => {
                f(OperandRole::Def, dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, flags_v);
                }
                f(OperandRole::Use, value);
                f(OperandRole::Use, amount);
            }

            Self::Rol {
                dest,
                value,
                amount,
                ..
            }
            | Self::Ror {
                dest,
                value,
                amount,
                ..
            }
            | Self::Rcl {
                dest,
                value,
                amount,
                ..
            }
            | Self::Rcr {
                dest,
                value,
                amount,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, value);
                f(OperandRole::Use, amount);
            }

            Self::WideMul {
                low,
                high,
                left,
                right,
                ..
            } => {
                f(OperandRole::Def, low);
                f(OperandRole::Def, high);
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
            }

            Self::WideDiv {
                quotient,
                remainder,
                high,
                low,
                divisor,
                ..
            } => {
                f(OperandRole::Def, quotient);
                f(OperandRole::Def, remainder);
                f(OperandRole::Use, high);
                f(OperandRole::Use, low);
                f(OperandRole::Use, divisor);
            }

            Self::Ceq {
                dest, left, right, ..
            }
            | Self::Clt {
                dest, left, right, ..
            }
            | Self::Cgt {
                dest, left, right, ..
            }
            | Self::BoolAnd {
                dest, left, right, ..
            }
            | Self::BoolOr {
                dest, left, right, ..
            }
            | Self::BoolXor {
                dest, left, right, ..
            }
            | Self::VectorBinary {
                dest, left, right, ..
            }
            | Self::VectorCompare {
                dest, left, right, ..
            }
            | Self::VectorMaskBinary {
                dest, left, right, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
            }

            Self::FloatCompareFlags {
                flags, left, right, ..
            } => {
                f(OperandRole::FlagsDef, flags);
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
            }

            Self::BranchCmp { left, right, .. } => {
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
            }

            Self::BoolNot { dest, value, .. }
            | Self::Copy {
                dest, src: value, ..
            }
            | Self::IntConv {
                dest,
                operand: value,
                ..
            }
            | Self::IntToPtr {
                dest,
                operand: value,
                ..
            }
            | Self::PtrToInt {
                dest,
                operand: value,
                ..
            }
            | Self::IntToFloat {
                dest,
                operand: value,
                ..
            }
            | Self::FloatToInt {
                dest,
                operand: value,
                ..
            }
            | Self::FloatConv {
                dest,
                operand: value,
                ..
            }
            | Self::Bitcast {
                dest,
                operand: value,
                ..
            }
            | Self::Ckfinite {
                dest,
                operand: value,
                ..
            }
            | Self::FpClassify {
                dest,
                operand: value,
                ..
            }
            | Self::BSwap {
                dest, src: value, ..
            }
            | Self::BRev {
                dest, src: value, ..
            }
            | Self::BitScanForward {
                dest, src: value, ..
            }
            | Self::BitScanReverse {
                dest, src: value, ..
            }
            | Self::Popcount {
                dest, src: value, ..
            }
            | Self::Parity {
                dest, src: value, ..
            }
            | Self::LoadField {
                dest,
                object: value,
                ..
            }
            | Self::LoadFieldAddr {
                dest,
                object: value,
                ..
            }
            | Self::ArrayLength {
                dest, array: value, ..
            }
            | Self::LoadIndirect {
                dest, addr: value, ..
            }
            | Self::AtomicLoad {
                dest, addr: value, ..
            }
            | Self::VectorUnary { dest, value, .. }
            | Self::VectorSplat { dest, value, .. }
            | Self::VectorCast { dest, value, .. }
            | Self::VectorReinterpret { dest, value, .. }
            | Self::VectorLoad {
                dest, addr: value, ..
            }
            | Self::VectorBroadcastLoad {
                dest, addr: value, ..
            }
            | Self::VectorExtract {
                dest,
                vector: value,
                ..
            }
            | Self::VectorMaskUnary {
                dest, mask: value, ..
            }
            | Self::VectorReduce { dest, value, .. }
            | Self::VectorBitmask { dest, value, .. }
            | Self::NewArr {
                dest,
                length: value,
                ..
            }
            | Self::Box { dest, value, .. }
            | Self::LoadVirtFunctionPtr {
                dest,
                object: value,
                ..
            }
            | Self::LocalAlloc {
                dest, size: value, ..
            }
            | Self::CastClass {
                dest,
                object: value,
                ..
            }
            | Self::IsInst {
                dest,
                object: value,
                ..
            }
            | Self::Unbox {
                dest,
                object: value,
                ..
            }
            | Self::UnboxAny {
                dest,
                object: value,
                ..
            }
            | Self::LoadObj {
                dest,
                src_addr: value,
                ..
            }
            | Self::ReadFlags {
                dest, flags: value, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, value);
            }

            Self::Branch {
                condition: value, ..
            }
            | Self::Switch { value, .. }
            | Self::StoreStaticField { value, .. }
            | Self::Pop { value }
            | Self::Throw { exception: value }
            | Self::EndFilter { result: value }
            | Self::InitObj {
                dest_addr: value, ..
            }
            | Self::IndirectBranch { target: value, .. }
            | Self::BranchFlags { flags: value, .. } => f(OperandRole::Use, value),

            Self::LoadElement {
                dest,
                array: a,
                index: b,
                ..
            }
            | Self::LoadElementAddr {
                dest,
                array: a,
                index: b,
                ..
            }
            | Self::AtomicRmw {
                dest,
                addr: a,
                value: b,
                ..
            }
            | Self::AtomicExchange {
                dest,
                addr: a,
                value: b,
                ..
            }
            | Self::AtomicLockRmw {
                dest,
                addr: a,
                value: b,
                ..
            }
            | Self::AtomicStoreConditional {
                status: dest,
                addr: a,
                value: b,
                ..
            }
            | Self::VectorInsert {
                dest,
                vector: a,
                value: b,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
            }

            Self::StoreField {
                object: a,
                value: b,
                ..
            }
            | Self::StoreIndirect {
                addr: a, value: b, ..
            }
            | Self::AtomicStore {
                addr: a, value: b, ..
            }
            | Self::VectorStore {
                addr: a, value: b, ..
            }
            | Self::CopyObj {
                dest_addr: a,
                src_addr: b,
                ..
            }
            | Self::StoreObj {
                dest_addr: a,
                value: b,
                ..
            } => {
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
            }

            Self::Select {
                dest,
                condition: a,
                true_val: b,
                false_val: c,
                ..
            }
            | Self::CmpXchg {
                dest,
                addr: a,
                expected: b,
                desired: c,
                ..
            }
            | Self::VectorTernary {
                dest,
                first: a,
                second: b,
                third: c,
                ..
            }
            | Self::AtomicPairStoreConditional {
                status: dest,
                addr: a,
                first_value: b,
                second_value: c,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
                f(OperandRole::Use, c);
            }

            Self::StoreElement {
                array: a,
                index: b,
                value: c,
                ..
            }
            | Self::VectorMaskedStore {
                addr: a,
                value: b,
                mask: c,
                ..
            }
            | Self::VectorPackStore {
                addr: a,
                value: b,
                mask: c,
                ..
            }
            | Self::InitBlk {
                dest_addr: a,
                value: b,
                size: c,
                ..
            }
            | Self::CopyBlk {
                dest_addr: a,
                src_addr: b,
                size: c,
                ..
            } => {
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
                f(OperandRole::Use, c);
            }

            Self::VectorScatter {
                base: a,
                indices: b,
                value: c,
                mask: d,
                ..
            } => {
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
                f(OperandRole::Use, c);
                f(OperandRole::Use, d);
            }

            Self::AtomicCmpXchg {
                old,
                success,
                addr,
                expected,
                desired,
                ..
            } => {
                f(OperandRole::Def, old);
                if let Some(success_v) = success {
                    f(OperandRole::Def, success_v);
                }
                f(OperandRole::Use, addr);
                f(OperandRole::Use, expected);
                f(OperandRole::Use, desired);
            }

            Self::AtomicPairLoad {
                first,
                second,
                addr,
                ..
            } => {
                f(OperandRole::Def, first);
                f(OperandRole::Def, second);
                f(OperandRole::Use, addr);
            }

            Self::AtomicPairCmpXchg {
                old_first,
                old_second,
                addr,
                expected_first,
                expected_second,
                desired_first,
                desired_second,
                ..
            } => {
                f(OperandRole::Def, old_first);
                f(OperandRole::Def, old_second);
                f(OperandRole::Use, addr);
                f(OperandRole::Use, expected_first);
                f(OperandRole::Use, expected_second);
                f(OperandRole::Use, desired_first);
                f(OperandRole::Use, desired_second);
            }

            Self::VectorPredicatedUnary {
                dest,
                value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorMaskedLoad {
                dest,
                addr: value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorPack {
                dest,
                value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorPackLoad {
                dest,
                addr: value,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, value);
                f(OperandRole::Use, mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, passthrough_v);
                }
            }

            Self::VectorPredicatedBinary {
                dest,
                left,
                right,
                mask,
                passthrough,
                ..
            }
            | Self::VectorGather {
                dest,
                base: left,
                indices: right,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
                f(OperandRole::Use, mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, passthrough_v);
                }
            }

            Self::VectorPredicatedTernary {
                dest,
                first,
                second,
                third,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, first);
                f(OperandRole::Use, second);
                f(OperandRole::Use, third);
                f(OperandRole::Use, mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, passthrough_v);
                }
            }

            Self::VectorFaultingLoad {
                dest,
                fault,
                addr,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, dest);
                if let Some(fault_v) = fault {
                    f(OperandRole::Def, fault_v);
                }
                f(OperandRole::Use, addr);
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, mask_v);
                }
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, passthrough_v);
                }
            }

            Self::VectorSegmentLoad {
                dests, base, mask, ..
            } => {
                for item in dests {
                    f(OperandRole::Def, item);
                }
                f(OperandRole::Use, base);
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, mask_v);
                }
            }

            Self::VectorSegmentStore {
                base, values, mask, ..
            } => {
                f(OperandRole::Use, base);
                for item in values {
                    f(OperandRole::Use, item);
                }
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, mask_v);
                }
            }

            Self::VectorShuffle {
                dest, left, right, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, left);
                if let Some(right_v) = right {
                    f(OperandRole::Use, right_v);
                }
            }

            Self::NativeOpaque(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }

            Self::NativeIntrinsic(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::SystemOp(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::ComputeOp(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::BcdAdjust(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorCrypto(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::TileOp(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPermute(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorMultiplyAdd(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPackNarrow(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorNarrowSaturate(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPredicateWhile(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPredicateBreak(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorComplexAdd(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorCountAdjust(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorExtendInLane(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorElementCount(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
            }
            Self::VectorSveAddressGen(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::FlagAdjust(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorStructLoadReplicate(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorSmeMisc(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPredicateOp(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorSveCompute(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorReverseChunks(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorMatrixMulAcc(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorSmeOuterProduct(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPredicateGen(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorFpHelper(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorSvePermute(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorTernaryLogic(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorDotProduct(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorMultiSad(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorIntDotProduct(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorStringCompare(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorBitfield(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorIntersect(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorShuffleBits(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorConditionalMove(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorHorizontalMinPos(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorComplexMul(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorClassify(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorHorizontalReduce(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::BlockString(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::WideCompareExchange(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }

            Self::FpTranscendental(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }

            Self::FpuControl(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }

            Self::NewObj { dest, args, .. } => {
                f(OperandRole::Def, dest);
                for item in args {
                    f(OperandRole::Use, item);
                }
            }

            Self::Call { dest, args, .. } | Self::CallVirt { dest, args, .. } => {
                if let Some(dest_v) = dest {
                    f(OperandRole::Def, dest_v);
                }
                for item in args {
                    f(OperandRole::Use, item);
                }
            }

            Self::CallIndirect {
                dest, fptr, args, ..
            } => {
                if let Some(dest_v) = dest {
                    f(OperandRole::Def, dest_v);
                }
                f(OperandRole::Use, fptr);
                for item in args {
                    f(OperandRole::Use, item);
                }
            }

            Self::Return { value } => {
                if let Some(value_v) = value {
                    f(OperandRole::Use, value_v);
                }
            }

            Self::Phi { dest, operands } => {
                f(OperandRole::Def, dest);
                for (_, item) in operands {
                    f(OperandRole::Use, item);
                }
            }

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
            | Self::VectorZeroUpper { .. }
            | Self::Unreachable => {}
        }
    }

    /// Calls `f` for every variable used by this operation.
    pub fn for_each_use<F>(&self, mut f: F)
    where
        F: FnMut(SsaVarId),
    {
        self.visit_operands(|role, var| {
            if matches!(role, OperandRole::Use) {
                f(var);
            }
        });
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
                | Self::IntConv {
                    overflow_check: true,
                    ..
                }
                | Self::FloatToInt {
                    overflow_check: true,
                    ..
                }
                | Self::LoadField { .. }
                | Self::StoreField { .. }
                | Self::LoadStaticField { .. }
                | Self::StoreStaticField { .. }
                | Self::LoadElement { .. }
                | Self::StoreElement { .. }
                | Self::LoadElementAddr { .. }
                | Self::LoadIndirect { .. }
                | Self::StoreIndirect { .. }
                | Self::LoadObj { .. }
                | Self::StoreObj { .. }
                | Self::InitObj { .. }
                | Self::CopyObj { .. }
                | Self::InitBlk { .. }
                | Self::CopyBlk { .. }
                | Self::NewObj { .. }
                | Self::NewArr { .. }
                | Self::CastClass { .. }
                | Self::Unbox { .. }
                | Self::UnboxAny { .. }
                | Self::Call { .. }
                | Self::CallVirt { .. }
                | Self::CallIndirect { .. }
                | Self::Throw { .. }
                | Self::Rethrow
                | Self::Break
                | Self::Ckfinite { .. }
                | Self::CmpXchg { .. }
                | Self::WideCompareExchange { .. }
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
                | Self::VectorStructLoadReplicate(_)
                | Self::VectorStore { .. }
                | Self::VectorMaskedStore { .. }
                | Self::VectorScatter { .. }
                | Self::VectorSegmentStore { .. }
                | Self::VectorPackStore { .. }
        ) || matches!(self, Self::NativeOpaque(data) if data.effects.may_throw)
            || matches!(self, Self::NativeIntrinsic(data) if data.effects.may_throw)
            || matches!(self, Self::SystemOp(data) if data.kind.effects().may_throw)
            || matches!(self, Self::ComputeOp(data) if data.kind.effects().may_throw)
            || matches!(self, Self::BcdAdjust(data) if data.kind.effects().may_throw)
            || matches!(self, Self::VectorCrypto(data) if data.kind.effects().may_throw)
            || matches!(self, Self::TileOp(data) if data.kind.effects().may_throw)
            || matches!(self, Self::BlockString(data) if data.kind.effects().may_throw)
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
            | Self::VectorPackLoad { .. }
            | Self::VectorStructLoadReplicate(_) => {
                SsaEffects::new(SsaEffectKind::Read, self.may_throw())
                    .with_trap(TrapClass::MemoryFault)
            }

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
            Self::NativeIntrinsic(data) => data.effects,
            Self::SystemOp(data) => data.kind.effects(),
            Self::ComputeOp(data) => data.kind.effects(),
            Self::BcdAdjust(data) => data.kind.effects(),
            Self::VectorCrypto(data) => data.kind.effects(),
            Self::TileOp(data) => data.kind.effects(),
            // Pure value-producing native/vector compute: no memory,
            // no traps, no control effects — results depend only on operands.
            Self::VectorPermute(_)
            | Self::VectorMultiplyAdd(_)
            | Self::VectorPackNarrow(_)
            | Self::VectorNarrowSaturate(_)
            | Self::VectorPredicateWhile(_)
            | Self::VectorPredicateBreak(_)
            | Self::VectorComplexAdd(_)
            | Self::VectorCountAdjust(_)
            | Self::VectorExtendInLane(_)
            | Self::VectorElementCount(_)
            | Self::VectorSveAddressGen(_)
            | Self::FlagAdjust(_)
            | Self::VectorSmeMisc(_)
            | Self::VectorSveCompute(_)
            | Self::VectorReverseChunks(_)
            | Self::VectorMatrixMulAcc(_)
            | Self::VectorSmeOuterProduct(_)
            | Self::VectorPredicateGen(_)
            | Self::VectorFpHelper(_)
            | Self::VectorSvePermute(_)
            | Self::VectorTernaryLogic(_)
            | Self::VectorDotProduct(_)
            | Self::VectorMultiSad(_)
            | Self::VectorIntDotProduct(_)
            | Self::VectorStringCompare(_)
            | Self::VectorBitfield(_)
            | Self::VectorIntersect(_)
            | Self::VectorShuffleBits(_)
            | Self::VectorConditionalMove(_)
            | Self::VectorHorizontalMinPos(_)
            | Self::VectorComplexMul(_)
            | Self::VectorClassify(_)
            | Self::VectorHorizontalReduce(_) => SsaEffects::new(SsaEffectKind::Pure, false),
            // `setffr`/`wrffr` write the SVE first-fault register, which is not
            // modeled as an SSA operand and whose `outputs` may be empty. Pure +
            // zero defs means DCE deletes them outright, silently dropping the
            // FFR initialization a following first-faulting load depends on.
            Self::VectorPredicateOp(data) => match data.op {
                PredicateOpKind::SetFirstFault | PredicateOpKind::WriteFirstFault => {
                    SsaEffects::new(SsaEffectKind::Opaque, false)
                }
                _ => SsaEffects::new(SsaEffectKind::Pure, false),
            },
            Self::BlockString(data) => data.kind.effects(),
            Self::WideCompareExchange { .. } => {
                SsaEffects::new(SsaEffectKind::Atomic, self.may_throw())
                    .atomic_ordering(AtomicOrdering::SeqCst)
                    .with_trap(TrapClass::MemoryFault)
            }

            // Transcendentals compute a value (pure, modulo FP exception flags);
            // FPU control ops mutate FPU control/status/tag state — a barrier.
            Self::FpTranscendental { .. } => SsaEffects::new(SsaEffectKind::Pure, false),
            Self::FpuControl { .. } => SsaEffects::new(SsaEffectKind::Opaque, false),

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

            // Pure value-producing ops: scalar/vector arithmetic, bitwise,
            // comparison, boolean, bit-manipulation, conversions, address
            // computation, and SSA bookkeeping. The `may_throw` bit is threaded
            // from `may_throw()` so the trapping members of this group
            // (`Div`/`Rem`/`WideDiv`, the overflow-checked arithmetic,
            // `Ckfinite`, checked `Conv`, `LoadElementAddr`) report a trap while
            // the rest stay non-throwing. `Rol`/`Ror` belong here; the
            // carry-coupled `Rcl`/`Rcr` do NOT (see the opaque arm below).
            Self::Add { .. }
            | Self::Sub { .. }
            | Self::Mul { .. }
            | Self::WideMul { .. }
            | Self::Div { .. }
            | Self::Rem { .. }
            | Self::WideDiv { .. }
            | Self::AddOvf { .. }
            | Self::SubOvf { .. }
            | Self::MulOvf { .. }
            | Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Neg { .. }
            | Self::Not { .. }
            | Self::Shl { .. }
            | Self::Shr { .. }
            | Self::Rol { .. }
            | Self::Ror { .. }
            | Self::Ceq { .. }
            | Self::Cgt { .. }
            | Self::Clt { .. }
            | Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::BoolNot { .. }
            | Self::BSwap { .. }
            | Self::BRev { .. }
            | Self::BitScanForward { .. }
            | Self::BitScanReverse { .. }
            | Self::Popcount { .. }
            | Self::Parity { .. }
            | Self::Bitcast { .. }
            | Self::IntConv { .. }
            | Self::IntToPtr { .. }
            | Self::PtrToInt { .. }
            | Self::IntToFloat { .. }
            | Self::FloatToInt { .. }
            | Self::FloatConv { .. }
            | Self::Ckfinite { .. }
            | Self::FpClassify { .. }
            | Self::Const { .. }
            | Self::Copy { .. }
            | Self::Select { .. }
            | Self::SizeOf { .. }
            | Self::ReadFlags { .. }
            | Self::ComputeFlags { .. }
            | Self::CallClobber { .. }
            | Self::Phi { .. }
            | Self::Nop
            | Self::Pop { .. }
            | Self::LoadArg { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::LoadFieldAddr { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::LoadElementAddr { .. }
            | Self::PtrAdd { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::LoadToken { .. }
            | Self::VectorUnary { .. }
            | Self::VectorBinary { .. }
            | Self::VectorTernary { .. }
            | Self::VectorPredicatedUnary { .. }
            | Self::VectorPredicatedBinary { .. }
            | Self::VectorPredicatedTernary { .. }
            | Self::VectorCompare { .. }
            | Self::VectorCast { .. }
            | Self::VectorReinterpret { .. }
            | Self::VectorExtract { .. }
            | Self::VectorInsert { .. }
            | Self::VectorSplat { .. }
            | Self::VectorShuffle { .. }
            | Self::VectorPack { .. }
            | Self::VectorReduce { .. }
            | Self::VectorBitmask { .. }
            | Self::VectorMaskUnary { .. }
            | Self::VectorMaskBinary { .. } => {
                SsaEffects::new(SsaEffectKind::Pure, self.may_throw())
            }

            // Memory-reading value ops: read architectural/object state that is
            // not modelled as an SSA operand, so they are NOT pure (must not be
            // hoisted or value-numbered as pure). Throw bit threaded from
            // `may_throw()` for the reference-faulting members.
            Self::LoadLocal { .. }
            | Self::IsInst { .. }
            | Self::ArrayLength { .. }
            | Self::LoadVirtFunctionPtr { .. } => {
                SsaEffects::new(SsaEffectKind::Read, self.may_throw())
            }

            // Carry-coupled rotates read and write the carry flag, which is not
            // an explicit SSA operand, so they have a hidden input/output and
            // must never be value-numbered, reordered, or eliminated as pure.
            Self::Rcl { .. } | Self::Rcr { .. } => {
                SsaEffects::new(SsaEffectKind::Opaque, self.may_throw())
            } // NOTE: this match is intentionally exhaustive — there is NO `_`
              // catch-all. A newly added `SsaOp` variant must be classified here
              // explicitly or the crate will not compile, which is what keeps a
              // forgotten op from being silently treated as a pure value.
        }
    }

    /// Returns the high-level operation family for this opcode.
    #[must_use]
    pub const fn class(&self) -> SsaOpClass {
        match self {
            Self::Nop
            | Self::Phi { .. }
            | Self::Copy { .. }
            | Self::Pop { .. }
            | Self::CallClobber { .. } => SsaOpClass::Synthetic,

            Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::BoolNot { .. } => SsaOpClass::Boolean,

            Self::FlagAdjust(_) => SsaOpClass::Flags,
            Self::ReadFlags { .. } | Self::BranchFlags { .. } | Self::ComputeFlags { .. } => {
                SsaOpClass::Flags
            }

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
            | Self::PtrAdd { .. }
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
            Self::NativeIntrinsic(_) => SsaOpClass::NativeIntrinsic,
            Self::SystemOp(_) => SsaOpClass::Call,
            Self::ComputeOp(_) => SsaOpClass::Scalar,
            Self::BcdAdjust(_) => SsaOpClass::Scalar,
            Self::VectorCrypto(_) => SsaOpClass::Vector,
            Self::TileOp(_) => SsaOpClass::Vector,
            Self::VectorPermute(_) => SsaOpClass::Vector,
            Self::VectorMultiplyAdd(_) => SsaOpClass::Vector,
            Self::VectorPackNarrow(_) => SsaOpClass::Vector,
            Self::VectorNarrowSaturate(_) => SsaOpClass::Vector,
            Self::VectorPredicateWhile(_) => SsaOpClass::Vector,
            Self::VectorPredicateBreak(_) => SsaOpClass::Vector,
            Self::VectorComplexAdd(_) => SsaOpClass::Vector,
            Self::VectorCountAdjust(_) => SsaOpClass::Vector,
            Self::VectorExtendInLane(_) => SsaOpClass::Vector,
            Self::VectorElementCount(_) => SsaOpClass::Vector,
            Self::VectorSveAddressGen(_) => SsaOpClass::Vector,
            Self::VectorStructLoadReplicate(_) => SsaOpClass::Vector,
            Self::VectorSmeMisc(_) => SsaOpClass::Vector,
            Self::VectorPredicateOp(_) => SsaOpClass::Vector,
            Self::VectorSveCompute(_) => SsaOpClass::Vector,
            Self::VectorReverseChunks(_) => SsaOpClass::Vector,
            Self::VectorMatrixMulAcc(_) => SsaOpClass::Vector,
            Self::VectorSmeOuterProduct(_) => SsaOpClass::Vector,
            Self::VectorPredicateGen(_) => SsaOpClass::Vector,
            Self::VectorFpHelper(_) => SsaOpClass::Vector,
            Self::VectorSvePermute(_) => SsaOpClass::Vector,
            Self::VectorTernaryLogic(_) => SsaOpClass::Vector,
            Self::VectorDotProduct(_) => SsaOpClass::Vector,
            Self::VectorMultiSad(_) => SsaOpClass::Vector,
            Self::VectorIntDotProduct(_) => SsaOpClass::Vector,
            Self::VectorStringCompare(_) => SsaOpClass::Vector,
            Self::VectorBitfield(_) => SsaOpClass::Vector,
            Self::VectorIntersect(_) => SsaOpClass::Vector,
            Self::VectorShuffleBits(_) => SsaOpClass::Vector,
            Self::VectorConditionalMove(_) => SsaOpClass::Vector,
            Self::VectorHorizontalMinPos(_) => SsaOpClass::Vector,
            Self::VectorComplexMul(_) => SsaOpClass::Vector,
            Self::VectorClassify(_) => SsaOpClass::Vector,
            Self::VectorHorizontalReduce(_) => SsaOpClass::Vector,
            Self::BlockString(_) => SsaOpClass::Memory,
            Self::WideCompareExchange { .. } => SsaOpClass::Atomic,
            Self::FpTranscendental { .. } | Self::FpuControl { .. } => SsaOpClass::NativeOpaque,

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
            | Self::FpClassify { .. }
            | Self::IntConv { .. }
            | Self::IntToPtr { .. }
            | Self::PtrToInt { .. }
            | Self::IntToFloat { .. }
            | Self::FloatToInt { .. }
            | Self::FloatConv { .. }
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

    /// Returns the operand signedness carried by this operation's payload.
    ///
    /// `Some` only for operations whose payload carries an `unsigned` flag
    /// (overflow-checked arithmetic, wide multiply/divide, division and
    /// remainder, shift-right, ordered comparison, conversion,
    /// compare-and-branch, lane-wise vector comparison); `None` for
    /// sign-agnostic operations.
    #[must_use]
    pub const fn arith_signedness(&self) -> Option<Signedness> {
        match self {
            Self::AddOvf { unsigned, .. }
            | Self::SubOvf { unsigned, .. }
            | Self::MulOvf { unsigned, .. }
            | Self::WideMul { unsigned, .. }
            | Self::Div { unsigned, .. }
            | Self::WideDiv { unsigned, .. }
            | Self::Rem { unsigned, .. }
            | Self::Shr { unsigned, .. }
            | Self::Clt { unsigned, .. }
            | Self::Cgt { unsigned, .. }
            | Self::IntConv { unsigned, .. }
            | Self::IntToFloat { unsigned, .. }
            | Self::FloatToInt { unsigned, .. }
            | Self::BranchCmp { unsigned, .. }
            | Self::VectorCompare { unsigned, .. } => Some(Signedness::from_unsigned(*unsigned)),
            _ => None,
        }
    }

    /// Returns the comparison relation carried by this operation's payload.
    ///
    /// Covers scalar compare-and-set (`Ceq`/`Clt`/`Cgt`), compare-and-branch
    /// (`BranchCmp`), and lane-wise vector comparison (`VectorCompare`).
    /// Signedness is reported separately by [`Self::arith_signedness`].
    /// `None` for non-comparison operations and for `FloatCompareFlags`,
    /// which produces a flag bundle rather than a single relation.
    #[must_use]
    pub const fn compare_kind(&self) -> Option<CmpKind> {
        match self {
            Self::Ceq { .. } => Some(CmpKind::Eq),
            Self::Clt { .. } => Some(CmpKind::Lt),
            Self::Cgt { .. } => Some(CmpKind::Gt),
            Self::BranchCmp { cmp, .. } => Some(*cmp),
            Self::VectorCompare { kind, .. } => match kind {
                VectorCompareKind::Eq => Some(CmpKind::Eq),
                VectorCompareKind::Ne => Some(CmpKind::Ne),
                VectorCompareKind::Lt => Some(CmpKind::Lt),
                VectorCompareKind::Le => Some(CmpKind::Le),
                VectorCompareKind::Gt => Some(CmpKind::Gt),
                VectorCompareKind::Ge => Some(CmpKind::Ge),
                // Unordered / ordered / not-lt / not-le / constant predicates
                // have no clean scalar [`CmpKind`] equivalent.
                VectorCompareKind::Unordered
                | VectorCompareKind::Ordered
                | VectorCompareKind::NotLt
                | VectorCompareKind::NotLe
                | VectorCompareKind::NotGe
                | VectorCompareKind::NotGt
                | VectorCompareKind::AlwaysTrue
                | VectorCompareKind::AlwaysFalse => None,
            },
            _ => None,
        }
    }

    /// Returns the direct single-address memory access for this operation.
    ///
    /// `Some` for operations that read or write memory through exactly one
    /// address variable: indirect loads/stores, atomic accesses, vector
    /// loads/stores, and single-destination block/object initialization. The
    /// access direction is semantic (a store-conditional writes; an exchange
    /// or read-modify-write both reads and writes). `None` for operations
    /// with two address operands (`CopyBlk`, `CopyObj`) and for structured
    /// addressing payloads (field, element, gather/scatter, segment), which
    /// hosts read directly from the payload.
    #[must_use]
    pub const fn memory_effect(&self) -> Option<MemoryEffect<'_, T>> {
        match self {
            Self::LoadIndirect {
                addr, value_type, ..
            }
            | Self::AtomicLoad {
                addr, value_type, ..
            } => Some(MemoryEffect {
                addr: *addr,
                reads: true,
                writes: false,
                value_type: Some(value_type),
            }),
            Self::StoreIndirect {
                addr, value_type, ..
            }
            | Self::AtomicStore {
                addr, value_type, ..
            }
            | Self::AtomicStoreConditional {
                addr, value_type, ..
            } => Some(MemoryEffect {
                addr: *addr,
                reads: false,
                writes: true,
                value_type: Some(value_type),
            }),
            Self::CmpXchg { addr, .. }
            | Self::AtomicRmw { addr, .. }
            | Self::AtomicExchange { addr, .. }
            | Self::AtomicLockRmw { addr, .. }
            | Self::AtomicCmpXchg { addr, .. }
            | Self::AtomicPairCmpXchg { addr, .. } => Some(MemoryEffect {
                addr: *addr,
                reads: true,
                writes: true,
                value_type: None,
            }),
            Self::AtomicPairLoad { addr, .. }
            | Self::VectorLoad { addr, .. }
            | Self::VectorMaskedLoad { addr, .. }
            | Self::VectorBroadcastLoad { addr, .. }
            | Self::VectorFaultingLoad { addr, .. }
            | Self::VectorPackLoad { addr, .. } => Some(MemoryEffect {
                addr: *addr,
                reads: true,
                writes: false,
                value_type: None,
            }),
            Self::AtomicPairStoreConditional { addr, .. }
            | Self::VectorStore { addr, .. }
            | Self::VectorMaskedStore { addr, .. }
            | Self::VectorPackStore { addr, .. } => Some(MemoryEffect {
                addr: *addr,
                reads: false,
                writes: true,
                value_type: None,
            }),
            Self::InitBlk { dest_addr, .. }
            | Self::InitObj { dest_addr, .. }
            | Self::StoreObj { dest_addr, .. } => Some(MemoryEffect {
                addr: *dest_addr,
                reads: false,
                writes: true,
                value_type: None,
            }),
            Self::LoadObj { src_addr, .. } => Some(MemoryEffect {
                addr: *src_addr,
                reads: true,
                writes: false,
                value_type: None,
            }),
            _ => None,
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
            Self::Nop
            | Self::Phi { .. }
            | Self::Copy { .. }
            | Self::Pop { .. }
            | Self::CallClobber { .. } => SsaSimilarityClass::Synthetic,

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
            | Self::BranchFlags { .. }
            | Self::ComputeFlags { .. }
            | Self::FlagAdjust(_) => SsaSimilarityClass::Flags,

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

            Self::IntConv { .. }
            | Self::IntToPtr { .. }
            | Self::PtrToInt { .. }
            | Self::IntToFloat { .. }
            | Self::FloatToInt { .. }
            | Self::FloatConv { .. }
            | Self::Bitcast { .. }
            | Self::Ckfinite { .. }
            | Self::FpClassify { .. }
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
            | Self::PtrAdd { .. }
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
            | Self::VectorCrypto(_)
            | Self::TileOp(_)
            | Self::VectorPermute(_)
            | Self::VectorMultiplyAdd(_)
            | Self::VectorPackNarrow(_)
            | Self::VectorNarrowSaturate(_)
            | Self::VectorPredicateWhile(_)
            | Self::VectorPredicateBreak(_)
            | Self::VectorComplexAdd(_)
            | Self::VectorCountAdjust(_)
            | Self::VectorExtendInLane(_)
            | Self::VectorElementCount(_)
            | Self::VectorSveAddressGen(_)
            | Self::VectorStructLoadReplicate(_)
            | Self::VectorSmeMisc(_)
            | Self::VectorPredicateOp(_)
            | Self::VectorSveCompute(_)
            | Self::VectorReverseChunks(_)
            | Self::VectorMatrixMulAcc(_)
            | Self::VectorSmeOuterProduct(_)
            | Self::VectorPredicateGen(_)
            | Self::VectorFpHelper(_)
            | Self::VectorSvePermute(_)
            | Self::VectorTernaryLogic(_)
            | Self::VectorDotProduct(_)
            | Self::VectorMultiSad(_)
            | Self::VectorIntDotProduct(_)
            | Self::VectorStringCompare(_)
            | Self::VectorBitfield(_)
            | Self::VectorIntersect(_)
            | Self::VectorShuffleBits(_)
            | Self::VectorConditionalMove(_)
            | Self::VectorHorizontalMinPos(_)
            | Self::VectorComplexMul(_)
            | Self::VectorClassify(_)
            | Self::VectorHorizontalReduce(_)
            | Self::VectorBitmask { .. } => SsaSimilarityClass::Vector,

            Self::WideMul { .. } | Self::WideDiv { .. } => SsaSimilarityClass::WideArithmetic,

            Self::NativeOpaque(_) | Self::NativeIntrinsic(_) => SsaSimilarityClass::NativeOpaque,
            Self::SystemOp(_) => SsaSimilarityClass::Call,
            Self::ComputeOp(data) => match data.kind {
                ComputeKind::BitDeposit | ComputeKind::BitExtract | ComputeKind::PointerAuth(_) => {
                    SsaSimilarityClass::Bitwise
                }
                ComputeKind::Checksum | ComputeKind::MipsDspAccumulate => {
                    SsaSimilarityClass::Arithmetic
                }
                ComputeKind::Random { .. } => SsaSimilarityClass::Call,
            },
            Self::BcdAdjust(_) => SsaSimilarityClass::Arithmetic,
            Self::BlockString(data) => match data.kind {
                BlockStringKind::Load => SsaSimilarityClass::MemoryRead,
                BlockStringKind::Compare | BlockStringKind::Scan => {
                    SsaSimilarityClass::MemoryReadWrite
                }
            },
            Self::WideCompareExchange { .. } => SsaSimilarityClass::Atomic,
            Self::FpTranscendental { .. } | Self::FpuControl { .. } => {
                SsaSimilarityClass::NativeOpaque
            }

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
            Self::IntConv { .. } => "conv",
            Self::IntToPtr { .. } => "inttoptr",
            Self::PtrToInt { .. } => "ptrtoint",
            Self::IntToFloat { .. } => "inttofloat",
            Self::FloatToInt { .. } => "floattoint",
            Self::FloatConv { .. } => "fconv",
            Self::Bitcast { .. } => "bitcast",
            Self::Select { .. } => "select",
            Self::ReadFlags { .. } => "readflags",
            Self::ComputeFlags { .. } => "flags.compute",
            Self::CallClobber { .. } => "call.clobber",
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
            Self::PtrAdd { .. } => "ptradd",
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
            Self::NativeIntrinsic(_) => "native.intrinsic",
            Self::SystemOp(data) => data.kind.kind_str(),
            Self::ComputeOp(data) => data.kind.kind_str(),
            Self::BcdAdjust(data) => data.kind.kind_str(),
            Self::VectorCrypto(data) => data.kind.kind_str(),
            Self::TileOp(data) => data.kind.kind_str(),
            Self::VectorPermute(_) => "vector.permute",
            Self::VectorMultiplyAdd(data) => data.kind.kind_str(),
            Self::VectorPackNarrow(data) => {
                if data.unsigned {
                    "vector.pack.narrow.u"
                } else {
                    "vector.pack.narrow.s"
                }
            }
            Self::VectorNarrowSaturate(data) => {
                if data.unsigned_dst {
                    "vector.narrow.saturate.u"
                } else {
                    "vector.narrow.saturate.s"
                }
            }
            Self::VectorPredicateWhile(_) => "vector.while",
            Self::VectorPredicateBreak(_) => "vector.break",
            Self::VectorComplexAdd(_) => "vector.cadd",
            Self::VectorCountAdjust(_) => "vector.countadj",
            Self::VectorExtendInLane(_) => "vector.xtl",
            Self::VectorElementCount(_) => "vector.cnt",
            Self::VectorSveAddressGen(_) => "vector.adr",
            Self::FlagAdjust(_) => "flags.adjust",
            Self::VectorStructLoadReplicate(_) => "vector.ldNr",
            Self::VectorSmeMisc(_) => "vector.sme.misc",
            Self::VectorPredicateOp(_) => "vector.predop",
            Self::VectorSveCompute(_) => "vector.sve.compute",
            Self::VectorReverseChunks(_) => "vector.revchunks",
            Self::VectorMatrixMulAcc(_) => "vector.mmla",
            Self::VectorSmeOuterProduct(_) => "vector.sme.mopa",
            Self::VectorPredicateGen(_) => "vector.pgen",
            Self::VectorFpHelper(_) => "vector.fphelper",
            Self::VectorSvePermute(_) => "vector.sve.perm",
            Self::VectorTernaryLogic(_) => "vector.ternlog",
            Self::VectorDotProduct(_) => "vector.dotproduct",
            Self::VectorMultiSad(_) => "vector.mpsadbw",
            Self::VectorIntDotProduct(_) => "vector.intdot",
            Self::VectorStringCompare(_) => "vector.pcmpstr",
            Self::VectorBitfield(_) => "vector.bitfield",
            Self::VectorIntersect(_) => "vector.p2intersect",
            Self::VectorShuffleBits(_) => "vector.shufbitqmb",
            Self::VectorConditionalMove(_) => "vector.condmove",
            Self::VectorHorizontalMinPos(_) => "vector.phminposuw",
            Self::VectorComplexMul(data) => data.kind.kind_str(),
            Self::VectorClassify(_) => "vector.fpclass",
            Self::VectorHorizontalReduce(_) => "vector.hreduce",
            Self::BlockString(data) => data.kind.kind_str(),
            Self::WideCompareExchange(data) => {
                if data.wide {
                    "atomic.cmpxchg16b"
                } else {
                    "atomic.cmpxchg8b"
                }
            }
            Self::FpTranscendental { .. } => "fp.transcendental",
            Self::FpuControl { .. } => "fpu.control",
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
            Self::FpClassify { .. } => "fpclassify",
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
    /// The number of uses that were replaced.
    pub fn replace_uses(&mut self, old_var: SsaVarId, new_var: SsaVarId) -> usize {
        let mut count: usize = 0;
        self.visit_operands_mut(|role, var| {
            if matches!(role, OperandRole::Use) && *var == old_var {
                *var = new_var;
                count = count.saturating_add(1);
            }
        });
        count
    }

    /// Rewrites every `Use` operand via `lookup`, in a single pass.
    ///
    /// The batch counterpart to [`Self::replace_uses`]: a substitution with many
    /// `(old → new)` pairs visits each operand **once** and consults `lookup`,
    /// instead of calling `replace_uses` per pair (which re-walks every operand
    /// for every pair — `O(operands × pairs)`). `lookup` returns the replacement
    /// for a used variable, or `None` to leave it unchanged. Returns the number
    /// of operands rewritten.
    pub fn replace_uses_with<F>(&mut self, mut lookup: F) -> usize
    where
        F: FnMut(SsaVarId) -> Option<SsaVarId>,
    {
        let mut count: usize = 0;
        self.visit_operands_mut(|role, var| {
            if matches!(role, OperandRole::Use) {
                if let Some(new_var) = lookup(*var) {
                    *var = new_var;
                    count = count.saturating_add(1);
                }
            }
        });
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
        // Operand remapping is delegated to the single exhaustive operand
        // visitor so this method can never silently drop a variable when a
        // new op variant is added. `visit_operands_mut` walks every
        // Def/FlagsDef/Use; block targets are remapped separately by
        // `remap_branch_targets`, so they are intentionally left untouched.
        let mut out = self.clone();
        out.visit_operands_mut(|_role, var| {
            if let Some(new) = remap(*var) {
                *var = new;
            }
        });
        out
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
            // Address computation pops the base and its optional index, pushes
            // the address. (A native-only op; never on the CIL eval stack.)
            Self::PtrAdd { index, .. } => (1u32.saturating_add(u32::from(index.is_some())), 1),
            // Flags computed from N inputs - pop N, push 1.
            Self::ComputeFlags { inputs, .. } => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = inputs.len() as u32;
                (pops, 1)
            }
            // Call-clobber defines N caller-saved registers - pop 0, push N.
            Self::CallClobber { outputs } => {
                #[allow(clippy::cast_possible_truncation)]
                let pushes = outputs.len() as u32;
                (0, pushes)
            }
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
            | Self::IntConv { .. }
            | Self::IntToPtr { .. }
            | Self::PtrToInt { .. }
            | Self::IntToFloat { .. }
            | Self::FloatToInt { .. }
            | Self::FloatConv { .. }
            | Self::Bitcast { .. }
            | Self::Ckfinite { .. }
            | Self::FpClassify { .. }
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
            Self::NativeIntrinsic(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::SystemOp(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::ComputeOp(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::BcdAdjust(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorCrypto(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::TileOp(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorPermute(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorMultiplyAdd(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorPackNarrow(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorNarrowSaturate(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorPredicateWhile(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorPredicateBreak(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorComplexAdd(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorCountAdjust(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorExtendInLane(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorElementCount(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (0, pushes)
            }
            Self::VectorSveAddressGen(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::FlagAdjust(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorStructLoadReplicate(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorSmeMisc(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorPredicateOp(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorSveCompute(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorReverseChunks(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorMatrixMulAcc(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorSmeOuterProduct(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorPredicateGen(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorFpHelper(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorSvePermute(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorTernaryLogic(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorDotProduct(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorMultiSad(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorIntDotProduct(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorStringCompare(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorBitfield(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorIntersect(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorShuffleBits(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorConditionalMove(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorHorizontalMinPos(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorComplexMul(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorClassify(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::VectorHorizontalReduce(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::BlockString(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::WideCompareExchange(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::FpTranscendental(data) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = data.inputs.len() as u32;
                #[allow(clippy::cast_possible_truncation)]
                let pushes = data.outputs.len() as u32;
                (pops, pushes)
            }
            Self::FpuControl(data) => {
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
            Self::IntConv { target, .. }
            | Self::IntToPtr { target, .. }
            | Self::PtrToInt { target, .. }
            | Self::IntToFloat { target, .. }
            | Self::FloatToInt { target, .. }
            | Self::FloatConv { target, .. }
            | Self::Bitcast { target, .. } => Some(target.clone()),
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
            Self::FpClassify { .. } => T::native_int_result_type(),
            Self::Parity { .. } | Self::ReadFlags { .. } | Self::ComputeFlags { .. } => {
                T::comparison_result_type()
            }
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
            Self::PtrAdd { result_type, .. } => Some(result_type.clone()),
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
            Self::IntConv {
                dest,
                operand,
                target,
                ..
            } => write!(f, "{dest} = conv.{target} {operand}"),
            Self::IntToPtr {
                dest,
                operand,
                target,
            } => write!(f, "{dest} = inttoptr.{target} {operand}"),
            Self::PtrToInt {
                dest,
                operand,
                target,
            } => write!(f, "{dest} = ptrtoint.{target} {operand}"),
            Self::IntToFloat {
                dest,
                operand,
                target,
                ..
            } => write!(f, "{dest} = inttofloat.{target} {operand}"),
            Self::FloatToInt {
                dest,
                operand,
                target,
                ..
            } => write!(f, "{dest} = floattoint.{target} {operand}"),
            Self::FloatConv {
                dest,
                operand,
                target,
            } => write!(f, "{dest} = fconv.{target} {operand}"),
            Self::Bitcast {
                dest,
                operand,
                target,
            } => write!(f, "{dest} = bitcast.{target} {operand}"),
            Self::ReadFlags { dest, flags, mask } => {
                write!(f, "{dest} = readflags {flags}, {mask}")
            }
            Self::ComputeFlags { dest, inputs } => {
                write!(f, "{dest} = flags.compute")?;
                for (i, input) in inputs.iter().enumerate() {
                    write!(f, "{} {input}", if i == 0 { "" } else { "," })?;
                }
                Ok(())
            }
            Self::CallClobber { outputs } => {
                for (i, output) in outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{output}")?;
                }
                write!(f, " = call.clobber")
            }
            Self::VectorUnary {
                dest, value, kind, ..
            } => {
                write!(f, "{dest} = vunary.{kind:?} {value}")
            }
            Self::VectorBinary {
                dest,
                left,
                right,
                kind,
                ..
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
            Self::PtrAdd {
                dest,
                base,
                index,
                stride,
                offset,
                ..
            } => {
                write!(f, "{dest} = ptradd {base}")?;
                if let Some(index) = index {
                    write!(f, " + {index}*{stride}")?;
                }
                if *offset != 0 {
                    write!(f, " + {offset}")?;
                }
                Ok(())
            }
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
            Self::NativeIntrinsic(data) => {
                let NativeIntrinsicData {
                    id,
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
                write!(f, "native.intrinsic.{id:?} {mnemonic}")?;
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
            Self::SystemOp(data) => {
                let NativeKindedData {
                    kind,
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                    clobbers,
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
                write!(f, "{} {mnemonic}", kind.kind_str())?;
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
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::ComputeOp(data) => {
                let NativeKindedData {
                    kind,
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                    clobbers,
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
                write!(f, "{} {mnemonic}", kind.kind_str())?;
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
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::BcdAdjust(data) => {
                let BcdAdjustData {
                    kind,
                    base,
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                    clobbers,
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
                write!(f, "{} {mnemonic}", kind.kind_str())?;
                if matches!(
                    kind,
                    BcdAdjustKind::AsciiMulAdjust | BcdAdjustKind::AsciiDivAdjust
                ) {
                    write!(f, " base={base}")?;
                }
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
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::VectorCrypto(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
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
                write!(f, "{}", kind.kind_str())?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::TileOp(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
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
                write!(f, "{}", kind.kind_str())?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::VectorPermute(data) => {
                let VectorPermuteData { outputs, inputs } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.permute")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorMultiplyAdd(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
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
                write!(f, "{}", kind.kind_str())?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorSvePermute(data) => {
                let KindedVecData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.sve.perm")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorFpHelper(data) => {
                let KindedVecData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.fphelper")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorPredicateGen(data) => {
                let KindedVecData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.pgen")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorSmeOuterProduct(data) => {
                let VectorSmeOuterProductData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.sme.mopa")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorMatrixMulAcc(data) => {
                let VectorMatrixMulAccData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.mmla")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorReverseChunks(data) => {
                let VectorReverseChunksData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.revchunks")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorCountAdjust(data) => {
                let VectorCountAdjustData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.countadj")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorExtendInLane(data) => {
                let VectorExtendInLaneData {
                    signed,
                    source_bits,
                    element_bits,
                    outputs,
                    inputs,
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
                let kind = if *signed { "sxt" } else { "uxt" };
                write!(f, "vector.{kind} i{source_bits}->i{element_bits}")?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                Ok(())
            }
            Self::VectorElementCount(data) => {
                let VectorElementCountData {
                    element_bits,
                    multiplier,
                    outputs,
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
                write!(f, "vector.cnt e{element_bits} x{multiplier}")?;
                Ok(())
            }
            Self::VectorSveAddressGen(data) => {
                let VectorSveAddressGenData {
                    signed_extend,
                    shift,
                    outputs,
                    inputs,
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
                let ext = match signed_extend {
                    Some(true) => "sxtw",
                    Some(false) => "uxtw",
                    None => "lsl",
                };
                write!(f, "vector.adr {ext} #{shift}")?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                Ok(())
            }
            Self::FlagAdjust(data) => {
                let KindedVecData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "flags.adjust")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorStructLoadReplicate(data) => {
                let VectorStructLoadReplicateData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.ldNr")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorSmeMisc(data) => {
                let VectorSmeMiscData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.sme.misc")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorPredicateOp(data) => {
                let VectorPredicateOpData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.predop")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorSveCompute(data) => {
                let VectorSveComputeData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.sve.compute")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorComplexAdd(data) => {
                let VectorComplexAddData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.cadd")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorPredicateBreak(data) => {
                let VectorPredicateBreakData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.break")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorPredicateWhile(data) => {
                let VectorPredicateWhileData { outputs, inputs, .. } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.while")?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorNarrowSaturate(data) => {
                let VectorNarrowSaturateData {
                    unsigned_dst,
                    outputs,
                    inputs,
                    ..
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
                write!(
                    f,
                    "{}",
                    if *unsigned_dst {
                        "vector.narrow.saturate.u"
                    } else {
                        "vector.narrow.saturate.s"
                    }
                )?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorPackNarrow(data) => {
                let VectorPackNarrowData {
                    unsigned,
                    outputs,
                    inputs,
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
                write!(
                    f,
                    "{}",
                    if *unsigned {
                        "vector.pack.narrow.u"
                    } else {
                        "vector.pack.narrow.s"
                    }
                )?;
                if !inputs.is_empty() {
                    write!(f, " ")?;
                    for (i, input) in inputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{input}")?;
                    }
                }
                Ok(())
            }
            Self::VectorTernaryLogic(data) => {
                let VecImm8Data {
                    imm8,
                    outputs,
                    inputs,
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
                write!(f, "vector.ternlog")?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                write!(f, " imm={imm8:#04x}")
            }
            Self::VectorDotProduct(data) => {
                let VectorDotProductData {
                    imm8,
                    element_bits,
                    outputs,
                    inputs,
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
                write!(f, "vector.dotproduct.{element_bits}")?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                write!(f, " imm={imm8:#04x}")
            }
            Self::VectorMultiSad(data) => {
                let VecImm8Data {
                    imm8,
                    outputs,
                    inputs,
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
                write!(f, "vector.mpsadbw")?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                write!(f, " imm={imm8:#04x}")
            }
            Self::VectorIntDotProduct(data) => {
                let VectorIntDotProductData {
                    signed_a,
                    signed_b,
                    source_bits,
                    dest_bits,
                    outputs,
                    inputs,
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
                write!(f, "vector.intdot")?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                write!(
                    f,
                    " s_a={signed_a} s_b={signed_b} src={source_bits} dst={dest_bits}"
                )
            }
            Self::VectorStringCompare(data) => {
                let VectorStringCompareData {
                    imm8,
                    explicit_length,
                    result_index,
                    outputs,
                    inputs,
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
                write!(
                    f,
                    "vector.pcmp{}str{}",
                    if *explicit_length { "e" } else { "i" },
                    if *result_index { "i" } else { "m" }
                )?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                write!(f, " imm={imm8:#04x}")
            }
            Self::VectorBitfield(data) => {
                let VectorBitfieldData {
                    insert,
                    index,
                    length,
                    outputs,
                    inputs,
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
                write!(
                    f,
                    "vector.{}",
                    if *insert { "insertq" } else { "extrq" }
                )?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                write!(f, " index={index} length={length}")
            }
            Self::VectorIntersect(data) => {
                let VectorIntersectData { outputs, inputs } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.p2intersect")?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                Ok(())
            }
            Self::VectorShuffleBits(data) => {
                let VectorShuffleBitsData { outputs, inputs } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.shufbitqmb")?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                Ok(())
            }
            Self::VectorConditionalMove(data) => {
                let VectorConditionalMoveData {
                    condition,
                    outputs,
                    inputs,
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
                write!(f, "vector.condmove.{}", condition.kind_str())?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                Ok(())
            }
            Self::VectorHorizontalMinPos(data) => {
                let VectorHorizontalMinPosData { outputs, inputs } = data.as_ref();
                if !outputs.is_empty() {
                    for (i, output) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{output}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "vector.phminposuw")?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                Ok(())
            }
            Self::VectorComplexMul(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
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
                write!(f, "{}", kind.kind_str())?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                Ok(())
            }
            Self::VectorClassify(data) => {
                let VecImm8Data {
                    imm8,
                    outputs,
                    inputs,
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
                write!(f, "vector.fpclass")?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                write!(f, " imm={imm8:#04x}")
            }
            Self::VectorHorizontalReduce(data) => {
                let VectorHorizontalReduceData {
                    subtract,
                    source_bits,
                    dest_bits,
                    outputs,
                    inputs,
                    ..
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
                write!(
                    f,
                    "vector.hreduce.{} {}->{}",
                    if *subtract { "sub" } else { "add" },
                    source_bits,
                    dest_bits
                )?;
                for input in inputs {
                    write!(f, " {input}")?;
                }
                Ok(())
            }
            Self::BlockString(data) => {
                let BlockStringOpData {
                    kind,
                    prefix: _,
                    element_bits: _,
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                    clobbers,
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
                write!(f, "{} {mnemonic}", kind.kind_str())?;
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
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::WideCompareExchange(data) => {
                let WideCmpXchgData {
                    wide,
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                    clobbers,
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
                write!(
                    f,
                    "{} {mnemonic}",
                    if *wide {
                        "atomic.cmpxchg16b"
                    } else {
                        "atomic.cmpxchg8b"
                    }
                )?;
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
                write!(f, " effects=Atomic")
            }
            Self::FpTranscendental(data) => {
                for (i, dest) in data.outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{dest}")?;
                }
                if !data.outputs.is_empty() {
                    write!(f, " = ")?;
                }
                write!(f, "fp.transcendental.{:?}", data.kind)?;
                for arg in &data.inputs {
                    write!(f, " {arg}")?;
                }
                Ok(())
            }
            Self::FpuControl(data) => {
                for (i, dest) in data.outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{dest}")?;
                }
                if !data.outputs.is_empty() {
                    write!(f, " = ")?;
                }
                write!(f, "fpu.control.{:?}", data.kind)?;
                for arg in &data.inputs {
                    write!(f, " {arg}")?;
                }
                Ok(())
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
            Self::FpClassify { dest, operand } => write!(f, "{dest} = fpclassify {operand}"),
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
        // `Rcl`/`Rcr` rotate *through carry*: the carry flag is an implicit
        // input and output with no SSA operand, so they are NOT pure and must
        // never be value-numbered or eliminated as if they were.
        assert!(!SsaOp::<MockTarget>::Rcl {
            dest: d,
            value: v,
            amount: a
        }
        .is_pure());
        assert!(!SsaOp::<MockTarget>::Rcr {
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

        // AArch64 `ld2r`/`ld3r`/`ld4r` reads memory through its address base, so
        // it must classify exactly like its sibling vector loads. Treating it as
        // pure would let GVN CSE it across an intervening store and let DCE
        // delete it outright.
        let struct_load_replicate = SsaOp::<MockTarget>::VectorStructLoadReplicate(Box::new(
            VectorStructLoadReplicateData {
                count: 2,
                element_bits: 32,
                outputs: vec![d],
                inputs: vec![a],
            },
        ))
        .effects();
        assert_eq!(struct_load_replicate.kind, SsaEffectKind::Read);
        assert_eq!(struct_load_replicate.trap, TrapClass::MemoryFault);
        assert!(struct_load_replicate.reads_memory());
        assert!(!struct_load_replicate.writes_memory());
        assert!(
            !struct_load_replicate.is_pure(),
            "ld2r/ld3r/ld4r is a memory load and must never be pure"
        );
        assert!(
            !struct_load_replicate.removable_when_unused(),
            "a faulting load must not be removable just because its dests are unused"
        );

        // `setffr`/`wrffr` write the FFR, which is not an SSA operand. With no
        // outputs, a pure classification lets DCE delete them.
        let setffr = SsaOp::<MockTarget>::VectorPredicateOp(Box::new(VectorPredicateOpData {
            op: PredicateOpKind::SetFirstFault,
            element_bits: 32,
            outputs: vec![],
            inputs: vec![],
        }))
        .effects();
        assert!(
            !setffr.removable_when_unused(),
            "setffr writes the first-fault register and must not be DCE'd"
        );

        // A predicate op that only computes a value stays pure.
        let count_active =
            SsaOp::<MockTarget>::VectorPredicateOp(Box::new(VectorPredicateOpData {
                op: PredicateOpKind::CountActive,
                element_bits: 32,
                outputs: vec![d],
                inputs: vec![a],
            }))
            .effects();
        assert!(count_active.is_pure(), "cntp computes a value and is pure");

        // `dsb`/`dmb`/`isb` order memory: they are fences, not plain writes.
        let barrier = SsaOp::<MockTarget>::SystemOp(Box::new(NativeKindedData {
            kind: SystemOpKind::Barrier,
            mnemonic: "dmb".into(),
            metadata: None,
            clobbers: vec![],
            outputs: vec![],
            inputs: vec![],
        }))
        .effects();
        assert_eq!(
            barrier.kind,
            SsaEffectKind::Fence,
            "a memory barrier must classify as a fence, not a write"
        );
        assert_eq!(barrier.ordering, Some(AtomicOrdering::SeqCst));

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

    /// A [`SystemOpKind`] is always carried by [`SsaOp::SystemOp`], which is
    /// **not** a structural terminator ([`SsaOp::is_terminator`] excludes it).
    /// The `check_native_effects` verifier invariant rejects any op whose
    /// `effects().control` is block-ending (`Terminator`/`Return`/`Throw`)
    /// unless it is a terminator — so **no** system-op kind may declare a
    /// block-ending control effect. Regression guard for `iret`/`eret`
    /// (`SystemOpKind::InterruptReturn`), which formerly declared
    /// `ControlEffect::Return` and made every mid-block `iret` fail lift SSA
    /// verification.
    #[test]
    fn system_op_kinds_never_declare_a_block_ending_control_effect() {
        let all = [
            SystemOpKind::CpuId,
            SystemOpKind::Timestamp { aux: false },
            SystemOpKind::Timestamp { aux: true },
            SystemOpKind::ReadSysReg {
                namespace: SysRegNamespace::X86Msr,
            },
            SystemOpKind::WriteSysReg {
                namespace: SysRegNamespace::Arm64System,
            },
            SystemOpKind::ReadPerfCounter,
            SystemOpKind::SystemCall,
            SystemOpKind::SystemReturn,
            SystemOpKind::Trap { vector: None },
            SystemOpKind::Trap { vector: Some(0x80) },
            SystemOpKind::InterruptReturn,
            SystemOpKind::CacheMaintenance,
            SystemOpKind::TlbMaintenance,
            SystemOpKind::Barrier,
            SystemOpKind::Privileged,
            SystemOpKind::Hypervisor,
            SystemOpKind::HardwareEngine,
            SystemOpKind::Transaction(SystemTransactionKind::Start),
            SystemOpKind::Transaction(SystemTransactionKind::Commit),
            SystemOpKind::Transaction(SystemTransactionKind::Cancel),
            SystemOpKind::Transaction(SystemTransactionKind::Test),
        ];
        for kind in all {
            let control = kind.effects().control;
            assert!(
                !matches!(
                    control,
                    ControlEffect::Terminator | ControlEffect::Return | ControlEffect::Throw
                ),
                "SystemOpKind {kind:?} declares block-ending control {control:?} but \
                 SsaOp::SystemOp is not a terminator",
            );
        }
        // `iret`/`eret` specifically transfers control externally like `sysret`.
        assert_eq!(
            SystemOpKind::InterruptReturn.effects().control,
            ControlEffect::Call,
        );
        assert!(!SsaOp::<MockTarget>::SystemOp(Box::new(NativeKindedData {
            kind: SystemOpKind::InterruptReturn,
            mnemonic: String::from("iret"),
            metadata: None,
            outputs: Vec::new(),
            inputs: Vec::new(),
            clobbers: Vec::new(),
        }))
        .is_terminator());
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
                element: VectorElement::default(),
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
        assert_eq!(
            SsaOp::<MockTarget>::NativeIntrinsic(Box::new(NativeIntrinsicData {
                id: NativeIntrinsicId::Rdtsc,
                mnemonic: "rdtsc".to_string(),
                metadata: None,
                outputs: vec![d],
                inputs: Vec::new(),
                clobbers: Vec::new(),
                effects: SsaEffects::new(SsaEffectKind::Opaque, false),
            }))
            .class(),
            SsaOpClass::NativeIntrinsic
        );
    }

    /// Builds a battery of representative ops covering every visitor arm shape.
    fn visitor_battery() -> Vec<SsaOp<MockTarget>> {
        let v: Vec<SsaVarId> = (0..8).map(SsaVarId::from_index).collect();
        vec![
            SsaOp::Const {
                dest: v[0],
                value: ConstValue::I32(7),
            },
            SsaOp::Add {
                dest: v[0],
                left: v[1],
                right: v[2],
                flags: Some(v[3]),
            },
            SsaOp::Neg {
                dest: v[0],
                operand: v[1],
                flags: None,
            },
            SsaOp::Shr {
                dest: v[0],
                value: v[1],
                amount: v[2],
                unsigned: true,
                flags: Some(v[3]),
            },
            SsaOp::WideMul {
                low: v[0],
                high: v[1],
                left: v[2],
                right: v[3],
                unsigned: false,
            },
            SsaOp::WideDiv {
                quotient: v[0],
                remainder: v[1],
                high: v[2],
                low: v[3],
                divisor: v[4],
                unsigned: false,
            },
            SsaOp::FloatCompareFlags {
                flags: v[0],
                left: v[1],
                right: v[2],
                signaling: false,
            },
            SsaOp::Select {
                dest: v[0],
                condition: v[1],
                true_val: v[2],
                false_val: v[3],
            },
            SsaOp::StoreIndirect {
                addr: v[0],
                value: v[1],
                value_type: MockType::I32,
            },
            SsaOp::AtomicCmpXchg {
                old: v[0],
                success: Some(v[1]),
                addr: v[2],
                expected: v[3],
                desired: v[4],
                success_ordering: AtomicOrdering::SeqCst,
                failure_ordering: AtomicOrdering::Relaxed,
                width: AtomicAccessWidth::Bits64,
                weak: false,
                volatile: false,
            },
            SsaOp::AtomicPairLoad {
                first: v[0],
                second: v[1],
                addr: v[2],
                first_type: MockType::I64,
                second_type: MockType::I64,
                ordering: AtomicOrdering::Acquire,
                width: AtomicAccessWidth::Bits128,
                volatile: false,
            },
            SsaOp::Call {
                dest: Some(v[0]),
                method: 3,
                args: vec![v[1], v[2], v[3]],
            },
            SsaOp::CallIndirect {
                dest: None,
                fptr: v[0],
                signature: 0,
                args: vec![v[1]],
            },
            SsaOp::Return { value: Some(v[0]) },
            SsaOp::Return { value: None },
            SsaOp::Phi {
                dest: v[0],
                operands: vec![(0, v[1]), (1, v[2])],
            },
            SsaOp::NativeOpaque(Box::new(NativeOpaqueData {
                mnemonic: "ud2".to_string(),
                metadata: None,
                outputs: vec![v[0], v[1]],
                inputs: vec![v[2], v[3]],
                clobbers: Vec::new(),
                effects: SsaEffects::new(SsaEffectKind::Opaque, true),
            })),
            SsaOp::NativeIntrinsic(Box::new(NativeIntrinsicData {
                id: NativeIntrinsicId::Cpuid,
                mnemonic: "cpuid".to_string(),
                metadata: None,
                outputs: vec![v[0], v[1]],
                inputs: vec![v[2]],
                clobbers: Vec::new(),
                effects: SsaEffects::new(SsaEffectKind::Opaque, false),
            })),
            SsaOp::BcdAdjust(Box::new(BcdAdjustData {
                kind: BcdAdjustKind::AsciiMulAdjust,
                base: 10,
                mnemonic: "aam".to_string(),
                metadata: None,
                outputs: vec![v[0], v[1]],
                inputs: vec![v[2]],
                clobbers: Vec::new(),
            })),
            SsaOp::VectorDotProduct(Box::new(VectorDotProductData {
                imm8: 0xff,
                element_bits: 32,
                outputs: vec![v[0]],
                inputs: vec![v[1], v[2]],
            })),
            SsaOp::VectorMultiSad(Box::new(VecImm8Data {
                imm8: 0x05,
                outputs: vec![v[0]],
                inputs: vec![v[1], v[2]],
            })),
            SsaOp::VectorStringCompare(Box::new(VectorStringCompareData {
                imm8: 0x0c,
                explicit_length: true,
                result_index: true,
                outputs: vec![v[0], v[1]],
                inputs: vec![v[2], v[3]],
            })),
            SsaOp::VectorHorizontalMinPos(Box::new(VectorHorizontalMinPosData {
                outputs: vec![v[0]],
                inputs: vec![v[1]],
            })),
            SsaOp::VectorConditionalMove(Box::new(VectorConditionalMoveData {
                condition: ByteMoveCondition::Negative,
                outputs: vec![v[0]],
                inputs: vec![v[1], v[2]],
            })),
            SsaOp::VectorIntersect(Box::new(VectorIntersectData {
                outputs: vec![v[0], v[1]],
                inputs: vec![v[2], v[3]],
            })),
            SsaOp::VectorShuffleBits(Box::new(VectorShuffleBitsData {
                outputs: vec![v[0]],
                inputs: vec![v[1], v[2]],
            })),
            SsaOp::VectorBitfield(Box::new(VectorBitfieldData {
                insert: false,
                index: 4,
                length: 8,
                outputs: vec![v[0]],
                inputs: vec![v[1]],
            })),
            SsaOp::Jump { target: 4 },
            SsaOp::Nop,
        ]
    }

    /// A typed `VectorBitfield` reports its output as a def / inputs as uses, is
    /// pure, and survives a variable remap with its fields intact.
    #[test]
    fn vector_bitfield_defs_uses_effects_and_remap() {
        let a = SsaVarId::from_index(0);
        let b = SsaVarId::from_index(1);
        let c = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::VectorBitfield(Box::new(VectorBitfieldData {
            insert: true,
            index: 16,
            length: 8,
            outputs: vec![a],
            inputs: vec![b, c],
        }));
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
        assert_eq!(op.uses(), vec![b, c]);
        assert_eq!(op.effects().kind, SsaEffectKind::Pure);
        let remapped = op.remap_variables(|x| Some(SsaVarId::from_index(x.index() + 10)));
        let SsaOp::VectorBitfield(data) = &remapped else {
            unreachable!("remap must preserve the VectorBitfield variant")
        };
        assert!(data.insert);
        assert_eq!(data.index, 16);
        assert_eq!(data.length, 8);
        assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
        assert_eq!(
            data.inputs,
            vec![SsaVarId::from_index(11), SsaVarId::from_index(12)]
        );
    }

    /// A typed `VectorHorizontalMinPos` reports its output as a def / input as a
    /// use, is pure, and survives a variable remap.
    #[test]
    fn vector_horizontal_minpos_defs_uses_effects_and_remap() {
        let a = SsaVarId::from_index(0);
        let b = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> =
            SsaOp::VectorHorizontalMinPos(Box::new(VectorHorizontalMinPosData {
                outputs: vec![a],
                inputs: vec![b],
            }));
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
        assert_eq!(op.uses(), vec![b]);
        assert_eq!(op.effects().kind, SsaEffectKind::Pure);
        let remapped = op.remap_variables(|x| Some(SsaVarId::from_index(x.index() + 10)));
        let SsaOp::VectorHorizontalMinPos(data) = &remapped else {
            unreachable!("remap must preserve the VectorHorizontalMinPos variant")
        };
        assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
        assert_eq!(data.inputs, vec![SsaVarId::from_index(11)]);
    }

    /// A typed `VectorStringCompare` reports outputs as defs / inputs as uses,
    /// is pure, and survives a variable remap with its imm8 and flags intact.
    #[test]
    fn vector_string_compare_defs_uses_effects_and_remap() {
        let v: Vec<SsaVarId> = (0..4).map(SsaVarId::from_index).collect();
        let op: SsaOp<MockTarget> = SsaOp::VectorStringCompare(Box::new(VectorStringCompareData {
            imm8: 0x0c,
            explicit_length: false,
            result_index: false,
            outputs: vec![v[0], v[1]],
            inputs: vec![v[2], v[3]],
        }));
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![v[0], v[1]]);
        assert_eq!(op.uses(), vec![v[2], v[3]]);
        assert_eq!(op.effects().kind, SsaEffectKind::Pure);
        let remapped = op.remap_variables(|x| Some(SsaVarId::from_index(x.index() + 10)));
        let SsaOp::VectorStringCompare(data) = &remapped else {
            unreachable!("remap must preserve the VectorStringCompare variant")
        };
        assert_eq!(data.imm8, 0x0c);
        assert!(!data.explicit_length);
        assert!(!data.result_index);
        assert_eq!(
            data.outputs,
            vec![SsaVarId::from_index(10), SsaVarId::from_index(11)]
        );
        assert_eq!(
            data.inputs,
            vec![SsaVarId::from_index(12), SsaVarId::from_index(13)]
        );
    }

    /// A typed `VectorMultiSad` reports its output as a def / inputs as uses, is
    /// pure, and survives a variable remap with its imm8 intact.
    #[test]
    fn vector_multi_sad_defs_uses_effects_and_remap() {
        let a = SsaVarId::from_index(0);
        let b = SsaVarId::from_index(1);
        let c = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::VectorMultiSad(Box::new(VecImm8Data {
            imm8: 0x05,
            outputs: vec![a],
            inputs: vec![b, c],
        }));
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
        assert_eq!(op.uses(), vec![b, c]);
        assert_eq!(op.effects().kind, SsaEffectKind::Pure);
        let remapped = op.remap_variables(|v| match v {
            v if v == a => Some(SsaVarId::from_index(10)),
            v if v == b => Some(SsaVarId::from_index(11)),
            v if v == c => Some(SsaVarId::from_index(12)),
            _ => None,
        });
        let SsaOp::VectorMultiSad(data) = &remapped else {
            unreachable!("remap must preserve the VectorMultiSad variant")
        };
        assert_eq!(data.imm8, 0x05);
        assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
        assert_eq!(
            data.inputs,
            vec![SsaVarId::from_index(11), SsaVarId::from_index(12)]
        );
    }

    /// A typed `VectorDotProduct` reports its output as a def / inputs as uses,
    /// is pure, and survives a variable remap with its imm8 and width intact.
    #[test]
    fn vector_dot_product_defs_uses_effects_and_remap() {
        let a = SsaVarId::from_index(0);
        let b = SsaVarId::from_index(1);
        let c = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::VectorDotProduct(Box::new(VectorDotProductData {
            imm8: 0x31,
            element_bits: 64,
            outputs: vec![a],
            inputs: vec![b, c],
        }));
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
        assert_eq!(op.uses(), vec![b, c]);
        assert_eq!(op.effects().kind, SsaEffectKind::Pure);
        let remapped = op.remap_variables(|v| match v {
            v if v == a => Some(SsaVarId::from_index(10)),
            v if v == b => Some(SsaVarId::from_index(11)),
            v if v == c => Some(SsaVarId::from_index(12)),
            _ => None,
        });
        let SsaOp::VectorDotProduct(data) = &remapped else {
            unreachable!("remap must preserve the VectorDotProduct variant")
        };
        assert_eq!(data.imm8, 0x31);
        assert_eq!(data.element_bits, 64);
        assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
        assert_eq!(
            data.inputs,
            vec![SsaVarId::from_index(11), SsaVarId::from_index(12)]
        );
    }

    /// A typed `BcdAdjust` reports outputs as defs / inputs as uses, is pure,
    /// and survives a variable remap with its kind and radix intact.
    #[test]
    fn bcd_adjust_defs_uses_effects_and_remap() {
        let a = SsaVarId::from_index(0);
        let b = SsaVarId::from_index(1);
        let c = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::BcdAdjust(Box::new(BcdAdjustData {
            kind: BcdAdjustKind::AsciiDivAdjust,
            base: 16,
            mnemonic: "aad".to_string(),
            metadata: None,
            outputs: vec![a, b],
            inputs: vec![c],
            clobbers: Vec::new(),
        }));
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![a, b]);
        assert_eq!(op.uses(), vec![c]);
        assert_eq!(op.effects().kind, SsaEffectKind::Pure);
        assert!(!op.effects().may_throw);
        let remapped = op.remap_variables(|v| match v {
            v if v == a => Some(SsaVarId::from_index(10)),
            v if v == b => Some(SsaVarId::from_index(11)),
            v if v == c => Some(SsaVarId::from_index(12)),
            _ => None,
        });
        let SsaOp::BcdAdjust(data) = &remapped else {
            unreachable!("remap must preserve the BcdAdjust variant")
        };
        assert_eq!(data.kind, BcdAdjustKind::AsciiDivAdjust);
        assert_eq!(data.base, 16);
        assert_eq!(
            data.outputs,
            vec![SsaVarId::from_index(10), SsaVarId::from_index(11)]
        );
        assert_eq!(data.inputs, vec![SsaVarId::from_index(12)]);
    }

    /// Compile-time completeness guard. This exhaustive match (deliberately no
    /// `_` arm) names every `SsaOp` variant, so adding a new variant breaks the
    /// build here until it is consciously handled. That is the signal to (1)
    /// classify it in the exhaustive `effects()` match, (2) cover its operands
    /// in `visit_operands` / `visit_operands_mut`, and (3) add a sample to
    /// `visitor_battery` so the invariant tests exercise it. Never executed — it
    /// exists purely for the exhaustiveness check.
    #[allow(dead_code)]
    fn op_variant_exhaustiveness_sentinel(op: SsaOp<MockTarget>) {
        match op {
            SsaOp::Const { .. } => {}
            SsaOp::Add { .. } => {}
            SsaOp::AddOvf { .. } => {}
            SsaOp::Sub { .. } => {}
            SsaOp::SubOvf { .. } => {}
            SsaOp::Mul { .. } => {}
            SsaOp::MulOvf { .. } => {}
            SsaOp::WideMul { .. } => {}
            SsaOp::Div { .. } => {}
            SsaOp::Rem { .. } => {}
            SsaOp::FloatCompareFlags { .. } => {}
            SsaOp::WideDiv { .. } => {}
            SsaOp::Neg { .. } => {}
            SsaOp::And { .. } => {}
            SsaOp::Or { .. } => {}
            SsaOp::Xor { .. } => {}
            SsaOp::Not { .. } => {}
            SsaOp::Shl { .. } => {}
            SsaOp::Shr { .. } => {}
            SsaOp::Rol { .. } => {}
            SsaOp::Ror { .. } => {}
            SsaOp::Rcl { .. } => {}
            SsaOp::Rcr { .. } => {}
            SsaOp::BSwap { .. } => {}
            SsaOp::BRev { .. } => {}
            SsaOp::BitScanForward { .. } => {}
            SsaOp::BitScanReverse { .. } => {}
            SsaOp::Popcount { .. } => {}
            SsaOp::Parity { .. } => {}
            SsaOp::Ceq { .. } => {}
            SsaOp::Clt { .. } => {}
            SsaOp::Cgt { .. } => {}
            SsaOp::BoolAnd { .. } => {}
            SsaOp::BoolOr { .. } => {}
            SsaOp::BoolXor { .. } => {}
            SsaOp::BoolNot { .. } => {}
            SsaOp::IntConv { .. } => {}
            SsaOp::IntToPtr { .. } => {}
            SsaOp::PtrToInt { .. } => {}
            SsaOp::IntToFloat { .. } => {}
            SsaOp::FloatToInt { .. } => {}
            SsaOp::FloatConv { .. } => {}
            SsaOp::Bitcast { .. } => {}
            SsaOp::Select { .. } => {}
            SsaOp::ReadFlags { .. } => {}
            SsaOp::VectorUnary { .. } => {}
            SsaOp::VectorBinary { .. } => {}
            SsaOp::VectorTernary { .. } => {}
            SsaOp::VectorPredicatedUnary { .. } => {}
            SsaOp::VectorPredicatedBinary { .. } => {}
            SsaOp::VectorPredicatedTernary { .. } => {}
            SsaOp::VectorCompare { .. } => {}
            SsaOp::VectorLoad { .. } => {}
            SsaOp::VectorStore { .. } => {}
            SsaOp::VectorMaskedLoad { .. } => {}
            SsaOp::VectorMaskedStore { .. } => {}
            SsaOp::VectorBroadcastLoad { .. } => {}
            SsaOp::VectorGather { .. } => {}
            SsaOp::VectorFaultingLoad { .. } => {}
            SsaOp::VectorSegmentLoad { .. } => {}
            SsaOp::VectorScatter { .. } => {}
            SsaOp::VectorSegmentStore { .. } => {}
            SsaOp::VectorExtract { .. } => {}
            SsaOp::VectorInsert { .. } => {}
            SsaOp::VectorSplat { .. } => {}
            SsaOp::VectorShuffle { .. } => {}
            SsaOp::VectorCast { .. } => {}
            SsaOp::VectorReinterpret { .. } => {}
            SsaOp::VectorPack { .. } => {}
            SsaOp::VectorPackLoad { .. } => {}
            SsaOp::VectorPackStore { .. } => {}
            SsaOp::VectorZeroUpper { .. } => {}
            SsaOp::VectorMaskUnary { .. } => {}
            SsaOp::VectorMaskBinary { .. } => {}
            SsaOp::VectorReduce { .. } => {}
            SsaOp::VectorBitmask { .. } => {}
            SsaOp::Jump { .. } => {}
            SsaOp::Branch { .. } => {}
            SsaOp::BranchCmp { .. } => {}
            SsaOp::BranchFlags { .. } => {}
            SsaOp::Switch { .. } => {}
            SsaOp::IndirectBranch { .. } => {}
            SsaOp::Return { .. } => {}
            SsaOp::LoadField { .. } => {}
            SsaOp::StoreField { .. } => {}
            SsaOp::LoadStaticField { .. } => {}
            SsaOp::StoreStaticField { .. } => {}
            SsaOp::LoadFieldAddr { .. } => {}
            SsaOp::LoadStaticFieldAddr { .. } => {}
            SsaOp::LoadElement { .. } => {}
            SsaOp::StoreElement { .. } => {}
            SsaOp::LoadElementAddr { .. } => {}
            SsaOp::PtrAdd { .. } => {}
            SsaOp::ArrayLength { .. } => {}
            SsaOp::LoadIndirect { .. } => {}
            SsaOp::StoreIndirect { .. } => {}
            SsaOp::NewObj { .. } => {}
            SsaOp::NewArr { .. } => {}
            SsaOp::CastClass { .. } => {}
            SsaOp::IsInst { .. } => {}
            SsaOp::Box { .. } => {}
            SsaOp::Unbox { .. } => {}
            SsaOp::UnboxAny { .. } => {}
            SsaOp::SizeOf { .. } => {}
            SsaOp::LoadToken { .. } => {}
            SsaOp::Call { .. } => {}
            SsaOp::CallVirt { .. } => {}
            SsaOp::CallIndirect { .. } => {}
            SsaOp::LoadFunctionPtr { .. } => {}
            SsaOp::LoadVirtFunctionPtr { .. } => {}
            SsaOp::LoadArg { .. } => {}
            SsaOp::LoadLocal { .. } => {}
            SsaOp::LoadArgAddr { .. } => {}
            SsaOp::LoadLocalAddr { .. } => {}
            SsaOp::Copy { .. } => {}
            SsaOp::Pop { .. } => {}
            SsaOp::Throw { .. } => {}
            SsaOp::Rethrow => {}
            SsaOp::EndFinally => {}
            SsaOp::EndFilter { .. } => {}
            SsaOp::InterruptReturn => {}
            SsaOp::Unreachable => {}
            SsaOp::Leave { .. } => {}
            SsaOp::InitBlk { .. } => {}
            SsaOp::CopyBlk { .. } => {}
            SsaOp::Fence { .. } => {}
            SsaOp::NativeOpaque(_) => {}
            SsaOp::NativeIntrinsic(_) => {}
            SsaOp::SystemOp(_) => {}
            SsaOp::ComputeOp(_) => {}
            SsaOp::BcdAdjust(_) => {}
            SsaOp::VectorCrypto(_) => {}
            SsaOp::TileOp(_) => {}
            SsaOp::VectorPermute(_) => {}
            SsaOp::VectorMultiplyAdd(_) => {}
            SsaOp::VectorPackNarrow(_) => {}
            SsaOp::VectorNarrowSaturate(_) => {}
            SsaOp::VectorPredicateWhile(_) => {}
            SsaOp::VectorPredicateBreak(_) => {}
            SsaOp::VectorComplexAdd(_) => {}
            SsaOp::VectorCountAdjust(_) => {}
            SsaOp::VectorExtendInLane(_) => {}
            SsaOp::VectorElementCount(_) => {}
            SsaOp::VectorSveAddressGen(_) => {}
            SsaOp::FlagAdjust(_) => {}
            SsaOp::VectorStructLoadReplicate(_) => {}
            SsaOp::VectorSmeMisc(_) => {}
            SsaOp::VectorPredicateOp(_) => {}
            SsaOp::VectorSveCompute(_) => {}
            SsaOp::VectorReverseChunks(_) => {}
            SsaOp::VectorMatrixMulAcc(_) => {}
            SsaOp::VectorSmeOuterProduct(_) => {}
            SsaOp::VectorPredicateGen(_) => {}
            SsaOp::VectorFpHelper(_) => {}
            SsaOp::VectorSvePermute(_) => {}
            SsaOp::VectorTernaryLogic(_) => {}
            SsaOp::VectorDotProduct(_) => {}
            SsaOp::VectorMultiSad(_) => {}
            SsaOp::VectorIntDotProduct(_) => {}
            SsaOp::VectorStringCompare(_) => {}
            SsaOp::VectorBitfield(_) => {}
            SsaOp::VectorIntersect(_) => {}
            SsaOp::VectorShuffleBits(_) => {}
            SsaOp::VectorConditionalMove(_) => {}
            SsaOp::VectorHorizontalMinPos(_) => {}
            SsaOp::VectorComplexMul(_) => {}
            SsaOp::VectorClassify(_) => {}
            SsaOp::VectorHorizontalReduce(_) => {}
            SsaOp::BlockString(_) => {}
            SsaOp::WideCompareExchange(_) => {}
            SsaOp::ComputeFlags { .. } => {}
            SsaOp::CallClobber { .. } => {}
            SsaOp::CmpXchg { .. } => {}
            SsaOp::AtomicRmw { .. } => {}
            SsaOp::AtomicLoad { .. } => {}
            SsaOp::AtomicStore { .. } => {}
            SsaOp::AtomicStoreConditional { .. } => {}
            SsaOp::AtomicPairLoad { .. } => {}
            SsaOp::AtomicPairStoreConditional { .. } => {}
            SsaOp::AtomicExchange { .. } => {}
            SsaOp::AtomicLockRmw { .. } => {}
            SsaOp::AtomicCmpXchg { .. } => {}
            SsaOp::AtomicPairCmpXchg { .. } => {}
            SsaOp::InitObj { .. } => {}
            SsaOp::CopyObj { .. } => {}
            SsaOp::LoadObj { .. } => {}
            SsaOp::StoreObj { .. } => {}
            SsaOp::Nop => {}
            SsaOp::Break => {}
            SsaOp::Ckfinite { .. } => {}
            SsaOp::FpClassify { .. } => {}
            SsaOp::FpTranscendental(_) => {}
            SsaOp::FpuControl(_) => {}
            SsaOp::LocalAlloc { .. } => {}
            SsaOp::Constrained { .. } => {}
            SsaOp::Volatile => {}
            SsaOp::Unaligned { .. } => {}
            SsaOp::TailPrefix => {}
            SsaOp::Readonly => {}
            SsaOp::Phi { .. } => {}
        }
    }

    /// Constructs one sample of every `SsaOp` variant (all 200). Field values
    /// are placeholders chosen only to be type-valid; the tests assert
    /// structural invariants, not semantics. Kept complete by
    /// `op_variant_exhaustiveness_sentinel`: adding a variant breaks the build
    /// there until a sample is added here too.
    fn all_sample_ops() -> Vec<SsaOp<MockTarget>> {
        let sv = SsaVarId::from_index(1);
        vec![
            SsaOp::Const {
                dest: sv,
                value: ConstValue::I32(0),
            },
            SsaOp::Add {
                dest: sv,
                left: sv,
                right: sv,
                flags: Some(sv),
            },
            SsaOp::AddOvf {
                dest: sv,
                left: sv,
                right: sv,
                unsigned: false,
                flags: Some(sv),
            },
            SsaOp::Sub {
                dest: sv,
                left: sv,
                right: sv,
                flags: Some(sv),
            },
            SsaOp::SubOvf {
                dest: sv,
                left: sv,
                right: sv,
                unsigned: false,
                flags: Some(sv),
            },
            SsaOp::Mul {
                dest: sv,
                left: sv,
                right: sv,
                flags: Some(sv),
            },
            SsaOp::MulOvf {
                dest: sv,
                left: sv,
                right: sv,
                unsigned: false,
                flags: Some(sv),
            },
            SsaOp::WideMul {
                low: sv,
                high: sv,
                left: sv,
                right: sv,
                unsigned: false,
            },
            SsaOp::Div {
                dest: sv,
                left: sv,
                right: sv,
                unsigned: false,
                flags: Some(sv),
            },
            SsaOp::Rem {
                dest: sv,
                left: sv,
                right: sv,
                unsigned: false,
                flags: Some(sv),
            },
            SsaOp::FloatCompareFlags {
                flags: sv,
                left: sv,
                right: sv,
                signaling: false,
            },
            SsaOp::WideDiv {
                quotient: sv,
                remainder: sv,
                high: sv,
                low: sv,
                divisor: sv,
                unsigned: false,
            },
            SsaOp::Neg {
                dest: sv,
                operand: sv,
                flags: Some(sv),
            },
            SsaOp::And {
                dest: sv,
                left: sv,
                right: sv,
                flags: Some(sv),
            },
            SsaOp::Or {
                dest: sv,
                left: sv,
                right: sv,
                flags: Some(sv),
            },
            SsaOp::Xor {
                dest: sv,
                left: sv,
                right: sv,
                flags: Some(sv),
            },
            SsaOp::Not {
                dest: sv,
                operand: sv,
                flags: Some(sv),
            },
            SsaOp::Shl {
                dest: sv,
                value: sv,
                amount: sv,
                flags: Some(sv),
            },
            SsaOp::Shr {
                dest: sv,
                value: sv,
                amount: sv,
                unsigned: false,
                flags: Some(sv),
            },
            SsaOp::Rol {
                dest: sv,
                value: sv,
                amount: sv,
            },
            SsaOp::Ror {
                dest: sv,
                value: sv,
                amount: sv,
            },
            SsaOp::Rcl {
                dest: sv,
                value: sv,
                amount: sv,
            },
            SsaOp::Rcr {
                dest: sv,
                value: sv,
                amount: sv,
            },
            SsaOp::BSwap { dest: sv, src: sv },
            SsaOp::BRev { dest: sv, src: sv },
            SsaOp::BitScanForward { dest: sv, src: sv },
            SsaOp::BitScanReverse { dest: sv, src: sv },
            SsaOp::Popcount { dest: sv, src: sv },
            SsaOp::Parity { dest: sv, src: sv },
            SsaOp::Ceq {
                dest: sv,
                left: sv,
                right: sv,
            },
            SsaOp::Clt {
                dest: sv,
                left: sv,
                right: sv,
                unsigned: false,
            },
            SsaOp::Cgt {
                dest: sv,
                left: sv,
                right: sv,
                unsigned: false,
            },
            SsaOp::BoolAnd {
                dest: sv,
                left: sv,
                right: sv,
            },
            SsaOp::BoolOr {
                dest: sv,
                left: sv,
                right: sv,
            },
            SsaOp::BoolXor {
                dest: sv,
                left: sv,
                right: sv,
            },
            SsaOp::BoolNot {
                dest: sv,
                value: sv,
            },
            SsaOp::IntConv {
                dest: sv,
                operand: sv,
                target: MockType::I32,
                overflow_check: false,
                unsigned: false,
            },
            SsaOp::IntToPtr {
                dest: sv,
                operand: sv,
                target: MockType::I32,
            },
            SsaOp::PtrToInt {
                dest: sv,
                operand: sv,
                target: MockType::I32,
            },
            SsaOp::IntToFloat {
                dest: sv,
                operand: sv,
                target: MockType::I32,
                unsigned: false,
            },
            SsaOp::FloatToInt {
                dest: sv,
                operand: sv,
                target: MockType::I32,
                overflow_check: false,
                unsigned: false,
            },
            SsaOp::FloatConv {
                dest: sv,
                operand: sv,
                target: MockType::I32,
            },
            SsaOp::Bitcast {
                dest: sv,
                operand: sv,
                target: MockType::I32,
            },
            SsaOp::Select {
                dest: sv,
                condition: sv,
                true_val: sv,
                false_val: sv,
            },
            SsaOp::ReadFlags {
                dest: sv,
                flags: sv,
                mask: FlagsMask::from_bits(0),
            },
            SsaOp::VectorUnary {
                dest: sv,
                value: sv,
                kind: VectorUnaryKind::Neg,
                element: VectorElement {
                    kind: VectorElementKind::Integer,
                    bits: 32,
                    scalar: false,
                },
            },
            SsaOp::VectorBinary {
                dest: sv,
                left: sv,
                right: sv,
                kind: VectorBinaryKind::Add,
                element: VectorElement {
                    kind: VectorElementKind::Integer,
                    bits: 32,
                    scalar: false,
                },
            },
            SsaOp::VectorTernary {
                dest: sv,
                first: sv,
                second: sv,
                third: sv,
                kind: VectorTernaryKind::Fma,
            },
            SsaOp::VectorPredicatedUnary {
                dest: sv,
                value: sv,
                mask: sv,
                passthrough: Some(sv),
                kind: VectorUnaryKind::Neg,
                mode: VectorMaskMode::Merge,
            },
            SsaOp::VectorPredicatedBinary {
                dest: sv,
                left: sv,
                right: sv,
                mask: sv,
                passthrough: Some(sv),
                kind: VectorBinaryKind::Add,
                mode: VectorMaskMode::Merge,
            },
            SsaOp::VectorPredicatedTernary {
                dest: sv,
                first: sv,
                second: sv,
                third: sv,
                mask: sv,
                passthrough: Some(sv),
                kind: VectorTernaryKind::Fma,
                mode: VectorMaskMode::Merge,
            },
            SsaOp::VectorCompare {
                dest: sv,
                left: sv,
                right: sv,
                kind: VectorCompareKind::Eq,
                unsigned: false,
            },
            SsaOp::VectorLoad {
                dest: sv,
                addr: sv,
                vector_type: MockType::I32,
            },
            SsaOp::VectorStore {
                addr: sv,
                value: sv,
                vector_type: MockType::I32,
            },
            SsaOp::VectorMaskedLoad {
                dest: sv,
                addr: sv,
                mask: sv,
                passthrough: Some(sv),
                vector_type: MockType::I32,
                mode: VectorMaskMode::Merge,
            },
            SsaOp::VectorMaskedStore {
                addr: sv,
                value: sv,
                mask: sv,
                vector_type: MockType::I32,
            },
            SsaOp::VectorBroadcastLoad {
                dest: sv,
                addr: sv,
                vector_type: MockType::I32,
            },
            SsaOp::VectorGather {
                dest: sv,
                base: sv,
                indices: sv,
                mask: sv,
                passthrough: Some(sv),
                vector_type: MockType::I32,
                mode: VectorMaskMode::Merge,
            },
            SsaOp::VectorFaultingLoad {
                dest: sv,
                fault: Some(sv),
                addr: sv,
                mask: Some(sv),
                passthrough: Some(sv),
                vector_type: MockType::I32,
                fault_mode: VectorFaultMode::Normal,
                mask_mode: VectorMaskMode::Merge,
            },
            SsaOp::VectorSegmentLoad {
                dests: vec![sv],
                base: sv,
                mask: Some(sv),
                vector_type: MockType::I32,
                segments: 0,
                layout: VectorSegmentLayout::Interleaved,
            },
            SsaOp::VectorScatter {
                base: sv,
                indices: sv,
                value: sv,
                mask: sv,
                vector_type: MockType::I32,
            },
            SsaOp::VectorSegmentStore {
                base: sv,
                values: vec![sv],
                mask: Some(sv),
                vector_type: MockType::I32,
                segments: 0,
                layout: VectorSegmentLayout::Interleaved,
            },
            SsaOp::VectorExtract {
                dest: sv,
                vector: sv,
                lane: 0,
            },
            SsaOp::VectorInsert {
                dest: sv,
                vector: sv,
                lane: 0,
                value: sv,
            },
            SsaOp::VectorSplat {
                dest: sv,
                value: sv,
                vector_type: MockType::I32,
            },
            SsaOp::VectorShuffle {
                dest: sv,
                left: sv,
                right: Some(sv),
                mask: VectorShuffleMask::new(vec![crate::target::VectorShuffleLane::Zero]),
            },
            SsaOp::VectorCast {
                dest: sv,
                value: sv,
                target_type: MockType::I32,
                kind: VectorCastKind::Signed,
            },
            SsaOp::VectorReinterpret {
                dest: sv,
                value: sv,
                target_type: MockType::I32,
            },
            SsaOp::VectorPack {
                dest: sv,
                value: sv,
                mask: sv,
                passthrough: Some(sv),
                vector_type: MockType::I32,
                element_bits: 0,
                kind: VectorPackKind::Compress,
                mode: VectorMaskMode::Merge,
            },
            SsaOp::VectorPackLoad {
                dest: sv,
                addr: sv,
                mask: sv,
                passthrough: Some(sv),
                vector_type: MockType::I32,
                element_bits: 0,
                kind: VectorPackKind::Compress,
                mode: VectorMaskMode::Merge,
            },
            SsaOp::VectorPackStore {
                addr: sv,
                value: sv,
                mask: sv,
                vector_type: MockType::I32,
                element_bits: 0,
                kind: VectorPackKind::Compress,
            },
            SsaOp::VectorZeroUpper { all: false },
            SsaOp::VectorMaskUnary {
                dest: sv,
                mask: sv,
                kind: VectorMaskUnaryKind::Not,
            },
            SsaOp::VectorMaskBinary {
                dest: sv,
                left: sv,
                right: sv,
                kind: VectorMaskBinaryKind::And,
            },
            SsaOp::VectorReduce {
                dest: sv,
                value: sv,
                kind: VectorReduceKind::Add,
            },
            SsaOp::VectorBitmask {
                dest: sv,
                value: sv,
                kind: VectorBitmaskKind::LaneMostSignificantBits,
            },
            SsaOp::Jump { target: 0 },
            SsaOp::Branch {
                condition: sv,
                true_target: 0,
                false_target: 0,
            },
            SsaOp::BranchCmp {
                left: sv,
                right: sv,
                cmp: CmpKind::Eq,
                unsigned: false,
                true_target: 0,
                false_target: 0,
            },
            SsaOp::BranchFlags {
                flags: sv,
                condition: FlagCondition::Carry,
                true_target: 0,
                false_target: 0,
            },
            SsaOp::Switch {
                value: sv,
                targets: vec![0usize],
                default: 0,
            },
            SsaOp::IndirectBranch {
                target: sv,
                resolved_targets: vec![0usize],
            },
            SsaOp::Return { value: Some(sv) },
            SsaOp::LoadField {
                dest: sv,
                object: sv,
                field: 0u32,
            },
            SsaOp::StoreField {
                object: sv,
                field: 0u32,
                value: sv,
            },
            SsaOp::LoadStaticField {
                dest: sv,
                field: 0u32,
            },
            SsaOp::StoreStaticField {
                field: 0u32,
                value: sv,
            },
            SsaOp::LoadFieldAddr {
                dest: sv,
                object: sv,
                field: 0u32,
            },
            SsaOp::LoadStaticFieldAddr {
                dest: sv,
                field: 0u32,
            },
            SsaOp::LoadElement {
                dest: sv,
                array: sv,
                index: sv,
                elem_type: MockType::I32,
            },
            SsaOp::StoreElement {
                array: sv,
                index: sv,
                value: sv,
                elem_type: MockType::I32,
            },
            SsaOp::LoadElementAddr {
                dest: sv,
                array: sv,
                index: sv,
                elem_type: 0u32,
            },
            SsaOp::PtrAdd {
                dest: sv,
                base: sv,
                index: Some(sv),
                stride: 4,
                offset: 8,
                result_type: MockType::I64,
            },
            SsaOp::ArrayLength {
                dest: sv,
                array: sv,
            },
            SsaOp::LoadIndirect {
                dest: sv,
                addr: sv,
                value_type: MockType::I32,
            },
            SsaOp::StoreIndirect {
                addr: sv,
                value: sv,
                value_type: MockType::I32,
            },
            SsaOp::NewObj {
                dest: sv,
                ctor: 0u32,
                args: vec![sv],
            },
            SsaOp::NewArr {
                dest: sv,
                elem_type: 0u32,
                length: sv,
            },
            SsaOp::CastClass {
                dest: sv,
                object: sv,
                target_type: 0u32,
            },
            SsaOp::IsInst {
                dest: sv,
                object: sv,
                target_type: 0u32,
            },
            SsaOp::Box {
                dest: sv,
                value: sv,
                value_type: 0u32,
            },
            SsaOp::Unbox {
                dest: sv,
                object: sv,
                value_type: 0u32,
            },
            SsaOp::UnboxAny {
                dest: sv,
                object: sv,
                value_type: 0u32,
            },
            SsaOp::SizeOf {
                dest: sv,
                value_type: 0u32,
            },
            SsaOp::LoadToken {
                dest: sv,
                token: 0u32,
            },
            SsaOp::Call {
                dest: Some(sv),
                method: 0u32,
                args: vec![sv],
            },
            SsaOp::CallVirt {
                dest: Some(sv),
                method: 0u32,
                args: vec![sv],
            },
            SsaOp::CallIndirect {
                dest: Some(sv),
                fptr: sv,
                signature: 0u32,
                args: vec![sv],
            },
            SsaOp::LoadFunctionPtr {
                dest: sv,
                method: 0u32,
            },
            SsaOp::LoadVirtFunctionPtr {
                dest: sv,
                object: sv,
                method: 0u32,
            },
            SsaOp::LoadArg {
                dest: sv,
                arg_index: 0,
            },
            SsaOp::LoadLocal {
                dest: sv,
                local_index: 0,
            },
            SsaOp::LoadArgAddr {
                dest: sv,
                arg_index: 0,
            },
            SsaOp::LoadLocalAddr {
                dest: sv,
                local_index: 0,
            },
            SsaOp::Copy { dest: sv, src: sv },
            SsaOp::Pop { value: sv },
            SsaOp::Throw { exception: sv },
            SsaOp::Rethrow,
            SsaOp::EndFinally,
            SsaOp::EndFilter { result: sv },
            SsaOp::InterruptReturn,
            SsaOp::Unreachable,
            SsaOp::Leave { target: 0 },
            SsaOp::InitBlk {
                dest_addr: sv,
                value: sv,
                size: sv,
            },
            SsaOp::CopyBlk {
                dest_addr: sv,
                src_addr: sv,
                size: sv,
            },
            SsaOp::Fence {
                kind: FenceKind::Full,
            },
            SsaOp::NativeOpaque(Box::new(NativeOpaqueData {
                mnemonic: String::new(),
                metadata: None,
                outputs: vec![sv],
                inputs: vec![sv],
                clobbers: Vec::new(),
                effects: SsaEffects::new(SsaEffectKind::Opaque, false),
            })),
            SsaOp::NativeIntrinsic(Box::new(NativeIntrinsicData {
                id: NativeIntrinsicId::Cpuid,
                mnemonic: String::new(),
                metadata: None,
                outputs: vec![sv],
                inputs: vec![sv],
                clobbers: Vec::new(),
                effects: SsaEffects::new(SsaEffectKind::Opaque, false),
            })),
            SsaOp::SystemOp(Box::new(NativeKindedData {
                kind: SystemOpKind::CpuId,
                mnemonic: String::new(),
                metadata: None,
                outputs: vec![sv],
                inputs: vec![sv],
                clobbers: Vec::new(),
            })),
            SsaOp::ComputeOp(Box::new(NativeKindedData {
                kind: ComputeKind::BitDeposit,
                mnemonic: String::new(),
                metadata: None,
                outputs: vec![sv],
                inputs: vec![sv],
                clobbers: Vec::new(),
            })),
            SsaOp::BcdAdjust(Box::new(BcdAdjustData {
                kind: BcdAdjustKind::DecimalAddAdjust,
                base: 0,
                mnemonic: String::new(),
                metadata: None,
                outputs: vec![sv],
                inputs: vec![sv],
                clobbers: Vec::new(),
            })),
            SsaOp::VectorCrypto(Box::new(KindedVecData {
                kind: VectorCryptoKind::AesEncrypt,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::TileOp(Box::new(KindedVecData {
                kind: TileOpKind::Zero,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorPermute(Box::new(VectorPermuteData {
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorMultiplyAdd(Box::new(KindedVecData {
                kind: VectorMaddKind::MultiplyAddS16,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorPackNarrow(Box::new(VectorPackNarrowData {
                unsigned: false,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorNarrowSaturate(Box::new(VectorNarrowSaturateData {
                signed_src: false,
                unsigned_dst: false,
                rounding: false,
                shift: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorPredicateWhile(Box::new(VectorPredicateWhileData {
                kind: VectorCompareKind::Eq,
                unsigned: false,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorPredicateBreak(Box::new(VectorPredicateBreakData {
                after: false,
                pair: false,
                propagate: false,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorComplexAdd(Box::new(VectorComplexAddData {
                rotate_270: false,
                saturate: false,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorCountAdjust(Box::new(VectorCountAdjustData {
                decrement: false,
                saturate: false,
                signed: false,
                by_predicate: false,
                element_bits: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorExtendInLane(Box::new(VectorExtendInLaneData {
                signed: false,
                source_bits: 0,
                element_bits: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorElementCount(Box::new(VectorElementCountData {
                element_bits: 0,
                multiplier: 0,
                outputs: vec![sv],
            })),
            SsaOp::VectorSveAddressGen(Box::new(VectorSveAddressGenData {
                signed_extend: Some(false),
                shift: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::FlagAdjust(Box::new(KindedVecData {
                kind: FlagAdjustKind::InvertCarry,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorStructLoadReplicate(Box::new(VectorStructLoadReplicateData {
                count: 0,
                element_bits: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorSmeMisc(Box::new(VectorSmeMiscData {
                op: SmeMiscKind::AddHorizontal,
                element_bits: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorPredicateOp(Box::new(VectorPredicateOpData {
                op: PredicateOpKind::CountActive,
                element_bits: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorSveCompute(Box::new(VectorSveComputeData {
                op: SveComputeKind::AddCarryBottom,
                element_bits: 0,
                rotation: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorReverseChunks(Box::new(VectorReverseChunksData {
                chunk_bits: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorMatrixMulAcc(Box::new(VectorMatrixMulAccData {
                signed_a: false,
                signed_b: false,
                float: false,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorSmeOuterProduct(Box::new(VectorSmeOuterProductData {
                subtract: false,
                signed_a: false,
                signed_b: false,
                float: false,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorPredicateGen(Box::new(KindedVecData {
                kind: PredicateGenKind::True,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorFpHelper(Box::new(KindedVecData {
                kind: FpHelperKind::ReciprocalExponent,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorSvePermute(Box::new(KindedVecData {
                kind: SvePermuteKind::Index,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorTernaryLogic(Box::new(VecImm8Data {
                imm8: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorDotProduct(Box::new(VectorDotProductData {
                imm8: 0,
                element_bits: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorMultiSad(Box::new(VecImm8Data {
                imm8: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorIntDotProduct(Box::new(VectorIntDotProductData {
                signed_a: false,
                signed_b: false,
                source_bits: 0,
                dest_bits: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorStringCompare(Box::new(VectorStringCompareData {
                imm8: 0,
                explicit_length: false,
                result_index: false,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorBitfield(Box::new(VectorBitfieldData {
                insert: false,
                index: 0,
                length: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorIntersect(Box::new(VectorIntersectData {
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorShuffleBits(Box::new(VectorShuffleBitsData {
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorConditionalMove(Box::new(VectorConditionalMoveData {
                condition: ByteMoveCondition::Zero,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorHorizontalMinPos(Box::new(VectorHorizontalMinPosData {
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorComplexMul(Box::new(KindedVecData {
                kind: ComplexMulKind::Multiply,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorClassify(Box::new(VecImm8Data {
                imm8: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::VectorHorizontalReduce(Box::new(VectorHorizontalReduceData {
                subtract: false,
                unsigned: false,
                source_bits: 0,
                dest_bits: 0,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::BlockString(Box::new(BlockStringOpData {
                kind: BlockStringKind::Compare,
                prefix: BlockStringPrefix::Repeat,
                element_bits: 0,
                mnemonic: String::new(),
                metadata: None,
                outputs: vec![sv],
                inputs: vec![sv],
                clobbers: Vec::new(),
            })),
            SsaOp::WideCompareExchange(Box::new(WideCmpXchgData {
                wide: false,
                mnemonic: String::new(),
                metadata: None,
                outputs: vec![sv],
                inputs: vec![sv],
                clobbers: Vec::new(),
            })),
            SsaOp::ComputeFlags {
                dest: sv,
                inputs: vec![sv],
            },
            SsaOp::CallClobber { outputs: vec![sv] },
            SsaOp::CmpXchg {
                dest: sv,
                addr: sv,
                expected: sv,
                desired: sv,
            },
            SsaOp::AtomicRmw {
                dest: sv,
                addr: sv,
                value: sv,
                op: AtomicRmwOp::Xchg,
            },
            SsaOp::AtomicLoad {
                dest: sv,
                addr: sv,
                value_type: MockType::I32,
                ordering: AtomicOrdering::Relaxed,
                width: AtomicAccessWidth::Bits8,
                volatile: false,
            },
            SsaOp::AtomicStore {
                addr: sv,
                value: sv,
                value_type: MockType::I32,
                ordering: AtomicOrdering::Relaxed,
                width: AtomicAccessWidth::Bits8,
                volatile: false,
            },
            SsaOp::AtomicStoreConditional {
                status: sv,
                addr: sv,
                value: sv,
                value_type: MockType::I32,
                success_ordering: AtomicOrdering::Relaxed,
                failure_ordering: AtomicOrdering::Relaxed,
                width: AtomicAccessWidth::Bits8,
                volatile: false,
            },
            SsaOp::AtomicPairLoad {
                first: sv,
                second: sv,
                addr: sv,
                first_type: MockType::I32,
                second_type: MockType::I32,
                ordering: AtomicOrdering::Relaxed,
                width: AtomicAccessWidth::Bits8,
                volatile: false,
            },
            SsaOp::AtomicPairStoreConditional {
                status: sv,
                addr: sv,
                first_value: sv,
                second_value: sv,
                first_type: MockType::I32,
                second_type: MockType::I32,
                success_ordering: AtomicOrdering::Relaxed,
                failure_ordering: AtomicOrdering::Relaxed,
                width: AtomicAccessWidth::Bits8,
                volatile: false,
            },
            SsaOp::AtomicExchange {
                dest: sv,
                addr: sv,
                value: sv,
                ordering: AtomicOrdering::Relaxed,
                width: AtomicAccessWidth::Bits8,
                volatile: false,
            },
            SsaOp::AtomicLockRmw {
                dest: sv,
                addr: sv,
                value: sv,
                op: AtomicRmwOp::Xchg,
                ordering: AtomicOrdering::Relaxed,
                width: AtomicAccessWidth::Bits8,
                volatile: false,
            },
            SsaOp::AtomicCmpXchg {
                old: sv,
                success: Some(sv),
                addr: sv,
                expected: sv,
                desired: sv,
                success_ordering: AtomicOrdering::Relaxed,
                failure_ordering: AtomicOrdering::Relaxed,
                width: AtomicAccessWidth::Bits8,
                weak: false,
                volatile: false,
            },
            SsaOp::AtomicPairCmpXchg {
                old_first: sv,
                old_second: sv,
                addr: sv,
                expected_first: sv,
                expected_second: sv,
                desired_first: sv,
                desired_second: sv,
                success_ordering: AtomicOrdering::Relaxed,
                failure_ordering: AtomicOrdering::Relaxed,
                width: AtomicAccessWidth::Bits8,
                weak: false,
                volatile: false,
            },
            SsaOp::InitObj {
                dest_addr: sv,
                value_type: 0u32,
            },
            SsaOp::CopyObj {
                dest_addr: sv,
                src_addr: sv,
                value_type: 0u32,
            },
            SsaOp::LoadObj {
                dest: sv,
                src_addr: sv,
                value_type: 0u32,
            },
            SsaOp::StoreObj {
                dest_addr: sv,
                value: sv,
                value_type: 0u32,
            },
            SsaOp::Nop,
            SsaOp::Break,
            SsaOp::Ckfinite {
                dest: sv,
                operand: sv,
            },
            SsaOp::FpClassify {
                dest: sv,
                operand: sv,
            },
            SsaOp::FpTranscendental(Box::new(KindedVecData {
                kind: TranscendentalKind::Sin,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::FpuControl(Box::new(KindedVecData {
                kind: FpuControlKind::LoadControlWord,
                outputs: vec![sv],
                inputs: vec![sv],
            })),
            SsaOp::LocalAlloc { dest: sv, size: sv },
            SsaOp::Constrained {
                constraint_type: 0u32,
            },
            SsaOp::Volatile,
            SsaOp::Unaligned { alignment: 0 },
            SsaOp::TailPrefix,
            SsaOp::Readonly,
            SsaOp::Phi {
                dest: sv,
                operands: vec![(0usize, sv)],
            },
        ]
    }

    /// Across ALL 200 variants: the operand visitor agrees with `defs()` /
    /// `uses()`, the primary `dest()` is the first `Def`-role operand, and
    /// `opcode_name()` is unique per variant. This is the runtime half of the
    /// completeness guarantee (the sentinel is the compile-time half).
    #[test]
    fn all_variants_visitor_defs_uses_and_opcode_are_consistent() {
        let ops = all_sample_ops();
        assert_eq!(
            ops.len(),
            200,
            "every SsaOp variant must have exactly one sample"
        );
        let mut names = std::collections::HashSet::new();
        for op in &ops {
            let mut visitor_defs = Vec::new();
            let mut visitor_uses = Vec::new();
            let mut first_def = None;
            op.visit_operands(|role, var| match role {
                OperandRole::Def => {
                    if first_def.is_none() {
                        first_def = Some(var);
                    }
                    visitor_defs.push(var);
                }
                OperandRole::FlagsDef => visitor_defs.push(var),
                OperandRole::Use => visitor_uses.push(var),
            });
            assert_eq!(
                visitor_defs,
                op.defs().collect::<Vec<_>>(),
                "defs mismatch for {op}"
            );
            assert_eq!(visitor_uses, op.uses(), "uses mismatch for {op}");
            assert_eq!(op.dest(), first_def, "dest is not the first def for {op}");
            assert!(
                names.insert(op.opcode_name()),
                "opcode_name {:?} is not unique",
                op.opcode_name()
            );
        }
        assert_eq!(names.len(), 200, "opcode_name must be unique per variant");
    }

    /// Cross-checks the classification methods against each other for every op
    /// in the battery: `is_pure()` must mean exactly "pure kind and cannot
    /// throw", and `effects().may_throw` must agree with the `may_throw()`
    /// predicate. These invariants are what the generic passes (DCE, GVN, LICM)
    /// rely on, so a mis-specified op is caught here. (`Rcl`/`Rcr` regressions,
    /// for instance, surface as a purity mismatch.)
    #[test]
    fn classification_methods_are_self_consistent() {
        for op in all_sample_ops() {
            let eff = op.effects();
            assert_eq!(
                op.is_pure(),
                eff.kind == SsaEffectKind::Pure && !eff.may_throw,
                "is_pure() disagrees with effects() for {op}"
            );
            assert_eq!(
                eff.may_throw,
                op.may_throw(),
                "effects().may_throw disagrees with may_throw() for {op}"
            );
        }
    }

    /// The visitor must agree with `defs()` (definition set and order) and
    /// `uses()` (use set and order) for every representative op shape.
    #[test]
    fn visit_operands_agrees_with_defs_and_uses() {
        for op in visitor_battery() {
            let mut visited_defs = Vec::new();
            let mut visited_uses = Vec::new();
            op.visit_operands(|role, var| match role {
                OperandRole::Def | OperandRole::FlagsDef => visited_defs.push(var),
                OperandRole::Use => visited_uses.push(var),
            });

            let defs: Vec<SsaVarId> = op.defs().collect();
            assert_eq!(visited_defs, defs, "defs mismatch for {op}");
            assert_eq!(visited_uses, op.uses(), "uses mismatch for {op}");
            if matches!(op, SsaOp::FloatCompareFlags { .. }) {
                // Defines only a flags bundle; there is no primary destination.
                assert_eq!(op.dest(), None);
            } else {
                assert_eq!(op.dest(), defs.first().copied(), "dest mismatch for {op}");
            }
        }
    }

    /// `replace_def` rewrites secondary outputs uniformly, including the
    /// native-intrinsic output list (previously only `NativeOpaque` was
    /// covered).
    #[test]
    fn replace_def_covers_native_intrinsic_outputs() {
        let old = SsaVarId::from_index(1);
        let new = SsaVarId::from_index(9);
        let mut op: SsaOp<MockTarget> = SsaOp::NativeIntrinsic(Box::new(NativeIntrinsicData {
            id: NativeIntrinsicId::Rdtsc,
            mnemonic: "rdtsc".to_string(),
            metadata: None,
            outputs: vec![SsaVarId::from_index(0), old],
            inputs: vec![old],
            clobbers: Vec::new(),
            effects: SsaEffects::new(SsaEffectKind::Opaque, false),
        }));
        assert!(op.replace_def(old, new));
        let SsaOp::NativeIntrinsic(data) = &op else {
            unreachable!()
        };
        assert_eq!(data.outputs, vec![SsaVarId::from_index(0), new]);
        assert_eq!(data.inputs, vec![old], "uses must stay untouched");
    }

    /// A typed `SystemOp` reports its outputs as defs and inputs as uses, its
    /// effects derive from the kind (never opaque), and `remap_variables`
    /// rewrites both lists while leaving the kind/metadata intact.
    #[test]
    fn system_op_defs_uses_effects_and_remap() {
        let a = SsaVarId::from_index(1);
        let b = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::SystemOp(Box::new(NativeKindedData {
            kind: SystemOpKind::ReadSysReg {
                namespace: SysRegNamespace::X86Msr,
            },
            mnemonic: "rdmsr".to_string(),
            metadata: None,
            outputs: vec![a],
            inputs: vec![b],
            clobbers: Vec::new(),
        }));

        // defs == outputs, uses == inputs.
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
        assert!(op.uses().contains(&b), "input must appear in uses");

        // Effects are precise (Read for a sysreg read), never Opaque.
        assert_eq!(op.effects().kind, SsaEffectKind::Read);

        // remap rewrites both inputs and outputs.
        let remapped = op.remap_variables(|v| match v {
            v if v == a => Some(SsaVarId::from_index(10)),
            v if v == b => Some(SsaVarId::from_index(20)),
            _ => None,
        });
        let SsaOp::SystemOp(data) = &remapped else {
            unreachable!("remap must preserve the SystemOp variant")
        };
        assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
        assert_eq!(data.inputs, vec![SsaVarId::from_index(20)]);
        assert_eq!(
            data.kind,
            SystemOpKind::ReadSysReg {
                namespace: SysRegNamespace::X86Msr
            },
            "kind must survive remap"
        );
    }

    /// A typed `ComputeOp` reports outputs as defs / inputs as uses, derives
    /// precise effects from the kind (pure for bit-permute, `Read` for the
    /// nondeterministic random source), and survives `remap_variables`.
    #[test]
    fn compute_op_defs_uses_effects_and_remap() {
        let a = SsaVarId::from_index(1);
        let b = SsaVarId::from_index(2);
        let pdep: SsaOp<MockTarget> = SsaOp::ComputeOp(Box::new(NativeKindedData {
            kind: ComputeKind::BitDeposit,
            mnemonic: "pdep".to_string(),
            metadata: None,
            outputs: vec![a],
            inputs: vec![b],
            clobbers: Vec::new(),
        }));
        assert_eq!(pdep.defs().collect::<Vec<_>>(), vec![a]);
        assert!(pdep.uses().contains(&b));
        // pdep is pure; rdrand reads a nondeterministic entropy source.
        assert_eq!(pdep.effects().kind, SsaEffectKind::Pure);

        let rdrand: SsaOp<MockTarget> = SsaOp::ComputeOp(Box::new(NativeKindedData {
            kind: ComputeKind::Random {
                from_entropy: false,
            },
            mnemonic: "rdrand".to_string(),
            metadata: None,
            outputs: vec![a],
            inputs: Vec::new(),
            clobbers: Vec::new(),
        }));
        assert_eq!(rdrand.effects().kind, SsaEffectKind::Read);

        let remapped = pdep.remap_variables(|v| match v {
            v if v == a => Some(SsaVarId::from_index(10)),
            v if v == b => Some(SsaVarId::from_index(20)),
            _ => None,
        });
        let SsaOp::ComputeOp(data) = &remapped else {
            unreachable!("remap must preserve the ComputeOp variant")
        };
        assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
        assert_eq!(data.inputs, vec![SsaVarId::from_index(20)]);
        assert_eq!(data.kind, ComputeKind::BitDeposit);
    }

    /// A typed `BlockString` reports outputs as defs / inputs as uses, derives
    /// precise effects from the kind (`ReadWrite` for compare, `Read` for load),
    /// preserves prefix/element_bits, and survives `remap_variables`.
    #[test]
    fn block_string_defs_uses_effects_and_remap() {
        let a = SsaVarId::from_index(1);
        let b = SsaVarId::from_index(2);
        let cmps: SsaOp<MockTarget> = SsaOp::BlockString(Box::new(BlockStringOpData {
            kind: BlockStringKind::Compare,
            prefix: BlockStringPrefix::RepeatEqual,
            element_bits: 8,
            mnemonic: "repe cmps".to_string(),
            metadata: None,
            outputs: vec![a],
            inputs: vec![b],
            clobbers: Vec::new(),
        }));
        assert_eq!(cmps.defs().collect::<Vec<_>>(), vec![a]);
        assert!(cmps.uses().contains(&b));
        assert_eq!(cmps.effects().kind, SsaEffectKind::ReadWrite);

        let lods: SsaOp<MockTarget> = SsaOp::BlockString(Box::new(BlockStringOpData {
            kind: BlockStringKind::Load,
            prefix: BlockStringPrefix::Repeat,
            element_bits: 32,
            mnemonic: "rep lods".to_string(),
            metadata: None,
            outputs: vec![a],
            inputs: Vec::new(),
            clobbers: Vec::new(),
        }));
        assert_eq!(lods.effects().kind, SsaEffectKind::Read);

        let remapped = cmps.remap_variables(|v| match v {
            v if v == a => Some(SsaVarId::from_index(10)),
            v if v == b => Some(SsaVarId::from_index(20)),
            _ => None,
        });
        let SsaOp::BlockString(data) = &remapped else {
            unreachable!("remap must preserve the BlockString variant")
        };
        assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
        assert_eq!(data.inputs, vec![SsaVarId::from_index(20)]);
        assert_eq!(data.prefix, BlockStringPrefix::RepeatEqual);
        assert_eq!(data.element_bits, 8);
    }

    /// A typed `WideCompareExchange` reports outputs as defs / inputs as uses,
    /// has a sequentially-consistent atomic effect (never opaque), and survives
    /// `remap_variables` with its `wide` flag intact.
    #[test]
    fn wide_compare_exchange_defs_uses_effects_and_remap() {
        let a = SsaVarId::from_index(1);
        let b = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::WideCompareExchange(Box::new(WideCmpXchgData {
            wide: true,
            mnemonic: "cmpxchg16b".to_string(),
            metadata: None,
            outputs: vec![a],
            inputs: vec![b],
            clobbers: Vec::new(),
        }));
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
        assert!(op.uses().contains(&b));
        assert_eq!(op.effects().kind, SsaEffectKind::Atomic);
        assert_eq!(op.effects().ordering, Some(AtomicOrdering::SeqCst));

        let remapped = op.remap_variables(|v| match v {
            v if v == a => Some(SsaVarId::from_index(10)),
            v if v == b => Some(SsaVarId::from_index(20)),
            _ => None,
        });
        let SsaOp::WideCompareExchange(data) = &remapped else {
            unreachable!("remap must preserve the WideCompareExchange variant")
        };
        assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
        assert_eq!(data.inputs, vec![SsaVarId::from_index(20)]);
        assert!(data.wide, "wide flag must survive remap");
    }

    /// A typed `ComputeFlags` defines its flags value, uses its inputs, is pure
    /// (so optimization can eliminate it), and survives `remap_variables`.
    #[test]
    fn compute_flags_defs_uses_effects_and_remap() {
        let dest = SsaVarId::from_index(1);
        let a = SsaVarId::from_index(2);
        let b = SsaVarId::from_index(3);
        let op: SsaOp<MockTarget> = SsaOp::ComputeFlags {
            dest,
            inputs: vec![a, b],
        };
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![dest]);
        assert!(op.uses().contains(&a) && op.uses().contains(&b));
        assert_eq!(op.effects().kind, SsaEffectKind::Pure);

        let remapped = op.remap_variables(|v| match v {
            v if v == dest => Some(SsaVarId::from_index(11)),
            v if v == a => Some(SsaVarId::from_index(12)),
            v if v == b => Some(SsaVarId::from_index(13)),
            _ => None,
        });
        let SsaOp::ComputeFlags { dest, inputs } = &remapped else {
            unreachable!("remap must preserve the ComputeFlags variant")
        };
        assert_eq!(*dest, SsaVarId::from_index(11));
        assert_eq!(
            inputs,
            &vec![SsaVarId::from_index(12), SsaVarId::from_index(13)]
        );
    }

    /// A typed `CallClobber` defines all its outputs, has no uses, is pure (so
    /// dead clobbers are eliminable), and survives `remap_variables`.
    #[test]
    fn call_clobber_defs_uses_effects_and_remap() {
        let a = SsaVarId::from_index(1);
        let b = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::CallClobber {
            outputs: vec![a, b],
        };
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![a, b]);
        assert!(op.uses().is_empty());
        assert_eq!(op.effects().kind, SsaEffectKind::Pure);

        let remapped = op.remap_variables(|v| match v {
            v if v == a => Some(SsaVarId::from_index(11)),
            v if v == b => Some(SsaVarId::from_index(12)),
            _ => None,
        });
        let SsaOp::CallClobber { outputs } = &remapped else {
            unreachable!("remap must preserve the CallClobber variant")
        };
        assert_eq!(
            outputs,
            &vec![SsaVarId::from_index(11), SsaVarId::from_index(12)]
        );
    }

    #[test]
    fn payload_accessors_report_signedness_compare_and_memory() {
        let d = SsaVarId::from_index(0);
        let a = SsaVarId::from_index(1);
        let b = SsaVarId::from_index(2);

        let udiv: SsaOp<MockTarget> = SsaOp::Div {
            dest: d,
            left: a,
            right: b,
            unsigned: true,
            flags: None,
        };
        assert_eq!(udiv.arith_signedness(), Some(Signedness::Unsigned));
        assert_eq!(udiv.compare_kind(), None);
        assert!(udiv.memory_effect().is_none());

        let clt: SsaOp<MockTarget> = SsaOp::Clt {
            dest: d,
            left: a,
            right: b,
            unsigned: false,
        };
        assert_eq!(clt.arith_signedness(), Some(Signedness::Signed));
        assert_eq!(clt.compare_kind(), Some(CmpKind::Lt));

        let branch_cmp: SsaOp<MockTarget> = SsaOp::BranchCmp {
            left: a,
            right: b,
            cmp: CmpKind::Ge,
            unsigned: true,
            true_target: 1,
            false_target: 2,
        };
        assert_eq!(branch_cmp.compare_kind(), Some(CmpKind::Ge));
        assert_eq!(branch_cmp.arith_signedness(), Some(Signedness::Unsigned));

        let add: SsaOp<MockTarget> = SsaOp::Add {
            dest: d,
            left: a,
            right: b,
            flags: None,
        };
        assert_eq!(add.arith_signedness(), None);
        assert_eq!(add.compare_kind(), None);
        assert!(add.memory_effect().is_none());

        let load: SsaOp<MockTarget> = SsaOp::LoadIndirect {
            dest: d,
            addr: a,
            value_type: MockType::I32,
        };
        let effect = load.memory_effect().expect("load has a memory effect");
        assert_eq!(effect.addr, a);
        assert!(effect.reads);
        assert!(!effect.writes);
        assert_eq!(effect.value_type, Some(&MockType::I32));

        let rmw: SsaOp<MockTarget> = SsaOp::AtomicRmw {
            dest: d,
            addr: a,
            value: b,
            op: AtomicRmwOp::Add,
        };
        let effect = rmw.memory_effect().expect("rmw has a memory effect");
        assert!(effect.reads);
        assert!(effect.writes);
        assert_eq!(effect.value_type, None);
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
                element: VectorElement::default(),
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
