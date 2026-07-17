//! Test fixtures for target-agnostic examples and downstream host tests.
//!
//! These helpers intentionally use [`crate::MockTarget`] so examples and tests
//! can exercise analyssa APIs without depending on a concrete instruction-set host.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use crate::{
    analysis::{SsaVerifier, VerifyLevel},
    error::Result,
    ir::{
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder, VectorFaultingLoadSpec},
        ops::{
            AtomicAccessWidth, AtomicOrdering, FenceKind, NativeClobber, SsaEffectKind, SsaEffects,
            SsaOp, VectorBinaryKind, VectorCompareKind, VectorElement, VectorElementKind,
            VectorFaultMode, VectorMaskMode, VectorSegmentLayout,
        },
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::{
        ScalableVectorMaskShape, ScalableVectorShape, Target, VectorLaneKind,
        VectorLengthMultiplier, VectorMaskPolicy, VectorMaskShape, VectorShape, VectorTailPolicy,
    },
    world::World,
};

fn fixture_result<T>(result: Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            log::error!("mock fixture construction failed: {error}");
            std::process::abort();
        }
    }
}

fn fixture_option<T>(value: Option<T>, message: &str) -> T {
    match value {
        Some(value) => value,
        None => {
            log::error!("{message}");
            std::process::abort();
        }
    }
}

/// A minimal [`Target`] impl for IR-core unit tests, doctests, and downstream
/// integration tests. Has no dependency on any host metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MockTarget;

/// Tiny stand-in type used to verify the IR core can carry an opaque
/// `T::Type` without depending on a host's type system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MockType {
    /// Unknown or intentionally unspecified type.
    Unknown,
    /// 32-bit integer test type.
    I32,
    /// 64-bit integer test type.
    I64,
    /// 64-bit floating-point test type.
    F64,
    /// Managed-reference test type.
    Ref,
    /// Native-pointer test type.
    Ptr,
    /// 128-bit vector of four 32-bit integer lanes.
    V4I32,
    /// 128-bit vector of two 64-bit floating-point lanes.
    V2F64,
    /// Four-lane vector mask represented as one bit per lane.
    Mask4,
    /// Runtime-scalable vector of at least four 32-bit integer lanes.
    NxV4I32,
    /// Runtime-scalable predicate mask of at least four lanes.
    NxMask4,
}

impl fmt::Display for MockType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => write!(f, "unknown"),
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::F64 => write!(f, "f64"),
            Self::Ref => write!(f, "ref"),
            Self::Ptr => write!(f, "ptr"),
            Self::V4I32 => write!(f, "v4i32"),
            Self::V2F64 => write!(f, "v2f64"),
            Self::Mask4 => write!(f, "mask4"),
            Self::NxV4I32 => write!(f, "nxv4i32"),
            Self::NxMask4 => write!(f, "nxmask4"),
        }
    }
}

impl Target for MockTarget {
    type TypeRef = u32;
    type MethodRef = u32;
    type FieldRef = u32;
    type SigRef = u32;
    type ExceptionKind = u32;
    type Type = MockType;
    type OriginalInstruction = ();
    type LocalSignature = ();
    type Capability = ();

    fn ptr_bytes(&self) -> u32 {
        8
    }

    fn synthetic_instruction() -> Self::OriginalInstruction {}

    fn unknown_type() -> Self::Type {
        MockType::Unknown
    }

    fn is_integer(t: &Self::Type) -> bool {
        matches!(t, MockType::I32 | MockType::I64)
    }

    fn is_floating(t: &Self::Type) -> bool {
        matches!(t, MockType::F64)
    }

    fn is_signed(t: &Self::Type) -> bool {
        matches!(t, MockType::I32 | MockType::I64 | MockType::F64)
    }

    fn is_pointer(t: &Self::Type) -> bool {
        matches!(t, MockType::Ptr)
    }

    fn is_reference(t: &Self::Type) -> bool {
        matches!(t, MockType::Ref)
    }

    fn is_unknown(t: &Self::Type) -> bool {
        matches!(t, MockType::Unknown)
    }

    fn bit_width(t: &Self::Type) -> Option<u32> {
        match t {
            MockType::I32 => Some(32),
            MockType::I64 | MockType::F64 => Some(64),
            MockType::V4I32 | MockType::V2F64 => Some(128),
            MockType::Mask4 | MockType::NxMask4 => Some(4),
            _ => None,
        }
    }

    fn vector_shape(t: &Self::Type) -> Option<VectorShape> {
        match t {
            MockType::V4I32 => VectorShape::new(4, VectorLaneKind::Integer, 32, 128),
            MockType::V2F64 => VectorShape::new(2, VectorLaneKind::Float, 64, 128),
            _ => None,
        }
    }

    fn scalable_vector_shape(t: &Self::Type) -> Option<ScalableVectorShape> {
        match t {
            MockType::NxV4I32 => ScalableVectorShape::new(
                4,
                VectorLaneKind::Integer,
                32,
                VectorLengthMultiplier::one(),
                VectorTailPolicy::Agnostic,
                VectorMaskPolicy::Agnostic,
            ),
            _ => None,
        }
    }

    fn vector_type(shape: VectorShape) -> Option<Self::Type> {
        match shape {
            VectorShape {
                lane_count: 4,
                lane_kind: VectorLaneKind::Integer,
                lane_bits: 32,
                total_bits: 128,
            } => Some(MockType::V4I32),
            VectorShape {
                lane_count: 2,
                lane_kind: VectorLaneKind::Float,
                lane_bits: 64,
                total_bits: 128,
            } => Some(MockType::V2F64),
            _ => None,
        }
    }

    fn scalable_vector_type(shape: ScalableVectorShape) -> Option<Self::Type> {
        match shape {
            ScalableVectorShape {
                min_lane_count: 4,
                lane_kind: VectorLaneKind::Integer,
                lane_bits: 32,
                length_multiplier,
                tail_policy: VectorTailPolicy::Agnostic,
                mask_policy: VectorMaskPolicy::Agnostic,
            } if length_multiplier == VectorLengthMultiplier::one() => Some(MockType::NxV4I32),
            _ => None,
        }
    }

    fn vector_lane_type(shape: VectorShape) -> Option<Self::Type> {
        match shape.lane_kind {
            VectorLaneKind::Integer if shape.lane_bits == 32 => Some(MockType::I32),
            VectorLaneKind::Float if shape.lane_bits == 64 => Some(MockType::F64),
            _ => None,
        }
    }

    fn scalable_vector_lane_type(shape: ScalableVectorShape) -> Option<Self::Type> {
        match shape.lane_kind {
            VectorLaneKind::Integer if shape.lane_bits == 32 => Some(MockType::I32),
            VectorLaneKind::Float if shape.lane_bits == 64 => Some(MockType::F64),
            _ => None,
        }
    }

    fn vector_mask_shape(t: &Self::Type) -> Option<VectorMaskShape> {
        match t {
            MockType::Mask4 => VectorMaskShape::new(4, 1),
            _ => None,
        }
    }

    fn scalable_vector_mask_shape(t: &Self::Type) -> Option<ScalableVectorMaskShape> {
        match t {
            MockType::NxMask4 => ScalableVectorMaskShape::new(4, 1, VectorLengthMultiplier::one()),
            _ => None,
        }
    }

    fn vector_mask_type(shape: VectorMaskShape) -> Option<Self::Type> {
        if shape.lane_count == 4 && shape.lane_bits == 1 {
            Some(MockType::Mask4)
        } else {
            None
        }
    }

    fn scalable_vector_mask_type(shape: ScalableVectorMaskShape) -> Option<Self::Type> {
        if shape.min_lane_count == 4
            && shape.lane_bits == 1
            && shape.length_multiplier == VectorLengthMultiplier::one()
        {
            Some(MockType::NxMask4)
        } else {
            None
        }
    }

    fn instruction_mnemonic(_instr: &Self::OriginalInstruction) -> &'static str {
        "<mock>"
    }

    fn instruction_rva(_instr: &Self::OriginalInstruction) -> u64 {
        0
    }

    fn is_filter_handler(_flags: &Self::ExceptionKind) -> bool {
        false
    }
}

/// Minimal [`World`] implementation backed by deterministic collections.
///
/// This is useful for testing interprocedural passes that only need method
/// reachability and dead-method marking.
pub struct MockWorld {
    all: Vec<u32>,
    entries: Vec<u32>,
    callees: BTreeMap<u32, Vec<u32>>,
    dead: RefCell<BTreeSet<u32>>,
}

impl MockWorld {
    /// Creates a mock world from method IDs, entry points, and call edges.
    #[must_use]
    pub fn new(
        all: impl IntoIterator<Item = u32>,
        entries: impl IntoIterator<Item = u32>,
        edges: impl IntoIterator<Item = (u32, u32)>,
    ) -> Self {
        let mut callees: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (from, to) in edges {
            callees.entry(from).or_default().push(to);
        }
        Self {
            all: all.into_iter().collect(),
            entries: entries.into_iter().collect(),
            callees,
            dead: RefCell::new(BTreeSet::new()),
        }
    }

    /// Returns the methods marked dead so far.
    #[must_use]
    pub fn dead_set(&self) -> BTreeSet<u32> {
        self.dead.borrow().clone()
    }
}

impl World<MockTarget> for MockWorld {
    fn all_methods(&self) -> Vec<u32> {
        self.all.clone()
    }

    fn entry_points(&self) -> Vec<u32> {
        self.entries.clone()
    }

    fn callees(&self, method: &u32) -> Vec<u32> {
        self.callees.get(method).cloned().unwrap_or_default()
    }

    fn is_dead(&self, method: &u32) -> bool {
        self.dead.borrow().contains(method)
    }

    fn mark_dead(&self, method: &u32) {
        self.dead.borrow_mut().insert(*method);
    }
}

/// Returns a mock `i32` temporary definition specification.
#[must_use]
pub const fn mock_i32() -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(MockType::I32)
}

/// Returns a mock `i64` temporary definition specification.
#[must_use]
pub const fn mock_i64() -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(MockType::I64)
}

/// Returns a mock native pointer temporary definition specification.
#[must_use]
pub const fn mock_ptr() -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(MockType::Ptr)
}

/// Returns a mock managed reference temporary definition specification.
#[must_use]
pub const fn mock_ref() -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(MockType::Ref)
}

/// Returns a mock `v4i32` vector temporary definition specification.
#[must_use]
pub const fn mock_v4i32() -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(MockType::V4I32)
}

/// Returns a mock four-lane vector mask temporary definition specification.
#[must_use]
pub const fn mock_mask4() -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(MockType::Mask4)
}

/// Creates a one-block `MockTarget` SSA function that returns an `i32` constant.
///
#[must_use]
pub fn const_i32_return(value: i32) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    fixture_result(builder.in_block(0, |block| {
        let dest = block.const_i32(SsaDefSpec::local(0, MockType::I32), value)?;
        block.ret(Some(dest))?;
        Ok(())
    }));
    fixture_result(builder.finish())
}

/// Creates a scalar rewrite fixture for optimization pass tests.
///
/// The fixture contains constants, algebraic identities, duplicate expressions,
/// reassociation opportunities, strength-reduction opportunities, copy chains,
/// a dead definition, and a constant branch.
#[must_use]
pub fn scalar_rewrite_fixture() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 12);
    builder.ensure_block(2);
    let copied = fixture_result(builder.in_block(0, |block| {
        let x = block.const_i32(mock_i32(), 7)?;
        let zero = block.const_i32(mock_i32(), 0)?;
        let scale = block.const_i32(mock_i32(), 8)?;
        let c2 = block.const_i32(mock_i32(), 2)?;
        let sum = block.add(mock_i32(), x, zero)?;
        let duplicate_sum = block.add(mock_i32(), zero, x)?;
        let reassoc_inner = block.add(mock_i32(), duplicate_sum, c2)?;
        let reassoc_outer = block.add(mock_i32(), reassoc_inner, scale)?;
        let shifted = block.mul(mock_i32(), reassoc_outer, scale)?;
        let copied = block.copy(mock_i32(), shifted)?;
        let _unused = block.const_i32(mock_i32(), 99)?;
        let branch_const = block.const_bool(mock_i32(), true)?;
        block.branch(branch_const, 1, 2)?;
        let _ = sum;
        Ok(copied)
    }));
    fixture_result(builder.in_block(1, |block| {
        block.ret(Some(copied))?;
        Ok(())
    }));
    fixture_result(builder.in_block(2, |block| {
        let value = block.const_i32(mock_i32(), 0)?;
        block.ret(Some(value))?;
        Ok(())
    }));
    fixture_result(builder.finish())
}

/// Creates a diamond CFG fixture with a merge phi.
///
#[must_use]
pub fn diamond_phi_fixture() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 4);
    builder.ensure_block(3);
    fixture_result(builder.in_block(0, |block| {
        let condition = block.const_bool(mock_i32(), true)?;
        block.branch(condition, 1, 2)?;
        Ok(())
    }));
    let left = fixture_result(builder.in_block(1, |block| {
        let value = block.const_i32(mock_i32(), 10)?;
        block.jump(3)?;
        Ok(value)
    }));
    let right = fixture_result(builder.in_block(2, |block| {
        let value = block.const_i32(mock_i32(), 20)?;
        block.jump(3)?;
        Ok(value)
    }));
    fixture_result(builder.in_block(3, |block| {
        let merged = block.phi(SsaDefSpec::local(0, MockType::I32), [(1, left), (2, right)])?;
        block.ret(Some(merged))?;
        Ok(())
    }));
    fixture_result(builder.finish())
}

/// Creates a loop fixture with a deferred backedge phi operand.
///
#[must_use]
pub fn loop_counter_fixture() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 2);
    builder.ensure_block(2);
    let zero = fixture_result(builder.in_block(0, |block| {
        let zero = block.const_i32(mock_i32(), 0)?;
        block.jump(1)?;
        Ok(zero)
    }));
    let (counter, next) = fixture_result(builder.in_block(1, |block| {
        let counter = block.empty_phi(SsaDefSpec::local(0, MockType::I32))?;
        let one = block.const_i32(mock_i32(), 1)?;
        let next = block.add(mock_i32(), counter, one)?;
        let limit = block.const_i32(mock_i32(), 4)?;
        let keep_going = block.clt(mock_i32(), next, limit, false)?;
        block.branch(keep_going, 1, 2)?;
        Ok((counter, next))
    }));
    fixture_result(builder.add_phi_operand(1, counter, 0, zero));
    fixture_result(builder.add_phi_operand(1, counter, 1, next));
    fixture_result(builder.in_block(2, |block| {
        block.ret(Some(counter))?;
        Ok(())
    }));
    fixture_result(builder.finish())
}

/// Creates a memory fixture with field, indirect, atomic, and fence effects.
#[must_use]
pub fn memory_effect_fixture() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 4);
    fixture_result(builder.in_block(0, |block| {
        let object = block.const_value(mock_ref(), ConstValue::Null)?;
        let addr = block.const_value(mock_ptr(), ConstValue::NativeUInt(0x1000))?;
        let value = block.const_i32(mock_i32(), 100)?;
        block.store_field(object, 1, value)?;
        let loaded = block.load_field(mock_i32(), object, 1)?;
        block.store_indirect(addr, loaded, MockType::I32)?;
        let old = block.atomic_exchange(
            mock_i32(),
            addr,
            value,
            AtomicOrdering::AcqRel,
            AtomicAccessWidth::Bits32,
            true,
        )?;
        block.fence(FenceKind::Acquire)?;
        block.ret(Some(old))?;
        Ok(())
    }));
    fixture_result(builder.finish())
}

/// Creates a native-effect fixture with flags, opaque outputs, clobbers, and wide arithmetic.
#[must_use]
pub fn native_effect_fixture() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 4);
    builder.ensure_block(1);
    fixture_result(builder.in_block(0, |block| {
        let left = block.const_i32(mock_i32(), 7)?;
        let right = block.const_i32(mock_i32(), 5)?;
        let (sum, flags) = block.add_with_flags(mock_i32(), mock_i32(), left, right)?;
        let zero = block.read_flags(mock_i32(), flags, crate::ir::FlagsMask::ZERO)?;
        let (_low, _high) = block.wide_mul(mock_i32(), mock_i32(), sum, right, false)?;
        let _opaque = block.native_opaque(
            &[mock_i32(), mock_i32()],
            "fixture.opaque",
            None,
            vec![zero],
            vec![NativeClobber::Flags("mock-flags".to_string())],
            SsaEffects::new(SsaEffectKind::Opaque, false),
        )?;
        block.branch_flags(flags, crate::ir::FlagCondition::Zero, 1, 1)?;
        Ok(())
    }));
    fixture_result(builder.in_block(1, |block| {
        block.ret(None)?;
        Ok(())
    }));
    fixture_result(builder.finish())
}

/// Creates a SIMD fixture covering vector arithmetic, masks, faulting loads, and segments.
#[must_use]
pub fn vector_simd_fixture() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 4);
    fixture_result(builder.in_block(0, |block| {
        let addr = block.const_value(mock_ptr(), ConstValue::NativeUInt(0x3000))?;
        let scalar = block.const_i32(mock_i32(), 9)?;
        let vector = block.vector_splat(mock_v4i32(), scalar, MockType::V4I32)?;
        let loaded = block.vector_load(mock_v4i32(), addr, MockType::V4I32)?;
        let added = block.vector_binary(
            mock_v4i32(),
            vector,
            loaded,
            VectorBinaryKind::Add,
            VectorElement {
                kind: VectorElementKind::Integer,
                bits: 32,
                scalar: false,
            },
        )?;
        let mask =
            block.vector_compare(mock_mask4(), added, loaded, VectorCompareKind::Eq, false)?;
        let (_faulting, fault) = block.vector_faulting_load(VectorFaultingLoadSpec {
            dest_def: mock_v4i32(),
            fault_def: Some(mock_mask4()),
            addr,
            mask: Some(mask),
            passthrough: Some(added),
            vector_type: MockType::V4I32,
            fault_mode: VectorFaultMode::FirstFault,
            mask_mode: VectorMaskMode::Merge,
        })?;
        let segments = block.vector_segment_load(
            &[mock_v4i32(), mock_v4i32()],
            addr,
            Some(mask),
            MockType::V4I32,
            2,
            VectorSegmentLayout::Interleaved,
        )?;
        block.vector_segment_store(
            addr,
            segments,
            Some(mask),
            MockType::V4I32,
            2,
            VectorSegmentLayout::Interleaved,
        )?;
        block.ret(fault)?;
        Ok(())
    }));
    fixture_result(builder.finish())
}

/// Verifies mock SSA at [`VerifyLevel::Full`] and panics with labeled errors.
pub fn assert_mock_valid_full(ssa: &SsaFunction<MockTarget>, label: &str) {
    let errors = SsaVerifier::new(ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{label} verifier errors: {errors:?}");
}

/// Verifies mock SSA at [`VerifyLevel::Standard`] and panics with labeled errors.
pub fn assert_mock_valid_standard(ssa: &SsaFunction<MockTarget>, label: &str) {
    let errors = SsaVerifier::new(ssa).verify(VerifyLevel::Standard);
    assert!(errors.is_empty(), "{label} verifier errors: {errors:?}");
}

/// Verifies mock SSA at [`VerifyLevel::Quick`] and panics with labeled errors.
pub fn assert_mock_valid_quick(ssa: &SsaFunction<MockTarget>, label: &str) {
    let errors = SsaVerifier::new(ssa).verify(VerifyLevel::Quick);
    assert!(errors.is_empty(), "{label} verifier errors: {errors:?}");
}

/// Runs a mock pass, validates both Full verifier boundaries, and returns whether it changed SSA.
pub fn run_mock_pass_boundary<F>(ssa: &mut SsaFunction<MockTarget>, label: &str, run: F) -> bool
where
    F: FnOnce(&mut SsaFunction<MockTarget>) -> bool,
{
    assert_mock_valid_full(ssa, &format!("{label} before"));
    let changed = run(ssa);
    ssa.recompute_uses();
    assert_mock_valid_full(ssa, &format!("{label} after"));
    changed
}

/// Runs a mock cleanup pass, validates Quick input and Full output.
pub fn run_mock_cleanup_boundary<F>(ssa: &mut SsaFunction<MockTarget>, label: &str, run: F) -> bool
where
    F: FnOnce(&mut SsaFunction<MockTarget>) -> bool,
{
    assert_mock_valid_quick(ssa, &format!("{label} before"));
    let changed = run(ssa);
    ssa.recompute_uses();
    assert_mock_valid_full(ssa, &format!("{label} after"));
    changed
}

/// Runs a mock cleanup pass for deliberately malformed input and validates Full output.
pub fn run_mock_malformed_cleanup_boundary<F>(
    ssa: &mut SsaFunction<MockTarget>,
    label: &str,
    run: F,
) -> bool
where
    F: FnOnce(&mut SsaFunction<MockTarget>) -> bool,
{
    let changed = run(ssa);
    ssa.recompute_uses();
    assert_mock_valid_full(ssa, &format!("{label} after"));
    changed
}

/// Runs a mock normalization pass, validates Standard input and Full output.
pub fn run_mock_normalization_boundary<F>(
    ssa: &mut SsaFunction<MockTarget>,
    label: &str,
    run: F,
) -> bool
where
    F: FnOnce(&mut SsaFunction<MockTarget>) -> bool,
{
    assert_mock_valid_standard(ssa, &format!("{label} before"));
    let changed = run(ssa);
    ssa.recompute_uses();
    assert_mock_valid_full(ssa, &format!("{label} after"));
    changed
}

/// Runs a mock pass, applies lightweight repair, validates, and returns whether it changed SSA.
pub fn run_mock_pass_repaired_boundary<F>(
    ssa: &mut SsaFunction<MockTarget>,
    label: &str,
    run: F,
) -> bool
where
    F: FnOnce(&mut SsaFunction<MockTarget>) -> bool,
{
    assert_mock_valid_full(ssa, &format!("{label} before"));
    let changed = run(ssa);
    if changed {
        ssa.repair_ssa();
    } else {
        ssa.recompute_uses();
    }
    assert_mock_valid_full(ssa, &format!("{label} after repair"));
    changed
}

/// Returns the number of mock SSA instructions matching `predicate`.
pub fn mock_op_count<F>(ssa: &SsaFunction<MockTarget>, mut predicate: F) -> usize
where
    F: FnMut(&SsaOp<MockTarget>) -> bool,
{
    ssa.iter_instructions()
        .filter(|(_, _, instr)| predicate(instr.op()))
        .count()
}

/// Asserts that at least one mock SSA instruction matches `predicate`.
pub fn assert_mock_has_op<F>(ssa: &SsaFunction<MockTarget>, label: &str, predicate: F)
where
    F: FnMut(&SsaOp<MockTarget>) -> bool,
{
    assert!(
        mock_op_count(ssa, predicate) > 0,
        "{label} should be present"
    );
}

/// Asserts that no mock SSA instruction matches `predicate`.
pub fn assert_mock_no_op<F>(ssa: &SsaFunction<MockTarget>, label: &str, predicate: F)
where
    F: FnMut(&SsaOp<MockTarget>) -> bool,
{
    assert_eq!(mock_op_count(ssa, predicate), 0, "{label} should be absent");
}

/// Returns a mock SSA instruction operation by block and instruction index.
#[must_use]
pub fn mock_op_at(ssa: &SsaFunction<MockTarget>, block: usize, instr: usize) -> &SsaOp<MockTarget> {
    fixture_option(
        ssa.block(block)
            .and_then(|ssa_block| ssa_block.instruction(instr))
            .map(|instruction| instruction.op()),
        "mock instruction should exist",
    )
}

/// Returns a mock SSA block terminator operation by block index.
#[must_use]
pub fn mock_terminator_at(ssa: &SsaFunction<MockTarget>, block: usize) -> &SsaOp<MockTarget> {
    fixture_option(
        ssa.block(block)
            .and_then(|ssa_block| ssa_block.terminator_op()),
        "mock terminator should exist",
    )
}

/// Appends a synthetic `i32` local definition to a mock SSA function.
pub fn create_i32_local(
    ssa: &mut SsaFunction<MockTarget>,
    block: usize,
    instruction: usize,
) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(instruction as u16),
        0,
        DefSite::instruction(block, instruction),
        MockType::I32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::ConstValue, Endianness};

    #[test]
    fn mock_target_does_not_pull_in_host_metadata() {
        assert_eq!(MockTarget.ptr_bytes(), 8);
        assert_eq!(MockTarget.endianness(), Endianness::Little);
        assert!(MockTarget::is_integer(&MockType::I32));
        assert!(!MockTarget::is_integer(&MockType::Ref));
        assert!(MockTarget::is_floating(&MockType::F64));
        assert!(MockTarget::is_signed(&MockType::I64));
        assert!(MockTarget::is_pointer(&MockType::Ptr));
        assert!(MockTarget::is_reference(&MockType::Ref));
        assert!(MockTarget::is_unknown(&MockType::Unknown));
        assert_eq!(MockTarget::bit_width(&MockType::I64), Some(64));
        assert_eq!(MockTarget::unknown_type(), MockType::Unknown);
        assert_eq!(MockTarget::synthetic_instruction(), ());
        assert_eq!(MockTarget::instruction_mnemonic(&()), "<mock>");
        assert_eq!(MockTarget::instruction_rva(&()), 0);
        assert!(!MockTarget::is_filter_handler(&0));
    }

    #[test]
    fn default_type_and_conversion_hooks_are_unsupported_for_mock_target() {
        let value = ConstValue::<MockTarget>::I32(1);

        assert_eq!(MockTarget::result_type_for_const(&value), None);
        assert_eq!(MockTarget::comparison_result_type(), None);
        assert_eq!(MockTarget::arithmetic_result_type(), None);
        assert_eq!(MockTarget::native_int_result_type(), None);
        assert_eq!(MockTarget::ckfinite_result_type(), None);
        assert_eq!(MockTarget::function_ptr_result_type(), None);
        assert_eq!(MockTarget::object_result_type(), None);
        assert_eq!(MockTarget::value_type_from_ref(&1), None);
        assert_eq!(MockTarget::byref_value_type_from_ref(&1), None);
        assert_eq!(MockTarget::byref_class_type_from_ref(&1), None);
        assert_eq!(
            MockTarget::convert_const(&value, &MockType::I64, false, MockTarget.ptr_bytes()),
            None
        );
        assert_eq!(
            MockTarget::convert_const_checked(
                &value,
                &MockType::I64,
                false,
                MockTarget.ptr_bytes()
            ),
            None
        );
        assert_eq!(
            MockTarget::evaluate_int_conv(1, &MockType::I64, false, MockTarget.ptr_bytes()),
            None
        );
    }

    #[test]
    fn mock_world_basic_queries() {
        let world = MockWorld::new([1u32, 2, 3, 4], [1u32], [(1u32, 2), (2, 3)]);

        assert_eq!(world.all_methods(), vec![1, 2, 3, 4]);
        assert_eq!(world.entry_points(), vec![1]);
        assert_eq!(world.callees(&1), vec![2]);
        assert_eq!(world.callees(&2), vec![3]);
        assert_eq!(world.methods_reverse_topological(), vec![1, 2, 3, 4]);
        assert!(world.callees(&4).is_empty());
        assert!(!world.is_dead(&4));

        world.mark_dead(&4);
        assert!(world.is_dead(&4));
        assert_eq!(world.dead_set(), [4u32].into_iter().collect());
    }

    #[test]
    fn const_i32_return_builds_valid_fixture() {
        let ssa = const_i32_return(42);

        assert_eq!(ssa.variables().len(), 1);
        assert_eq!(
            ssa.block(0).map(|block| block.instructions().len()),
            Some(2)
        );
    }
}
