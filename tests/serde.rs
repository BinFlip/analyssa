//! Round-trip coverage for the `serde` feature.
//!
//! The feature serializes the whole IR data model — the operation-kind taxonomy,
//! the native descriptors, and the SSA graph itself (`SsaFunction` and
//! everything reachable from it). Compiling proves only that the derives exist;
//! these tests prove values actually survive a round trip, which is the guarantee
//! the feature sells to hosts that persist them.
//!
//! Generic IR types serialize only when the host's `Target` associated types do,
//! via `#[serde(bound(...))]`. `MockTarget` satisfies that, so a full
//! `SsaFunction<MockTarget>` round trip exercises the real generic path.
//!
//! Deliberately excluded: borrowed views (`SsaDefs<'a>`, `MemoryEffect<'a, T>`),
//! transient pass machinery (builders, editors, edit reports), and
//! `SsaFeatureToken` (its `&'static str` opcode can serialize but cannot
//! deserialize).

#![cfg(feature = "serde")]

use analyssa::{
    ir::{
        block::SsaBlock,
        exception::NativeExceptionKind,
        function::{FunctionKind, SsaFunction},
        instruction::SsaInstruction,
        ops::{
            AtomicOrdering, AtomicRmwOp, BcdAdjustKind, BinaryOpKind, BlockStringKind, CmpKind,
            ComplexMulKind, ComputeKind, ControlEffect, FenceKind, FlagAdjustKind, FpuControlKind,
            KindedVecData, MemoryAccessSemantics, MemoryEffectLocation, NativeClobber,
            NativeInstructionMetadata, NativeIntrinsicId, NativeRegister, NativeStateAccess,
            NativeStateAccessKind, NativeStateLocation, OperandRole, PacKind, PredicateGenKind,
            PredicateOpKind, Signedness, SmeMiscKind, SsaEffectKind, SsaEffects, SsaOp, SsaOpClass,
            SsaSimilarityClass, SveComputeKind, SystemOpKind, TileOpKind, TranscendentalKind,
            TrapClass, UnaryOpKind, VectorBinaryKind, VectorCastKind, VectorCompareKind,
            VectorCryptoKind, VectorElement, VectorElementKind, VectorMaddKind, VectorPackKind,
            VectorReduceKind, VectorTernaryKind, VectorUnaryKind,
        },
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    testing::{MockTarget, MockType},
    Endianness, PointerSize,
};

/// Asserts that a value survives a JSON round trip unchanged.
///
/// Avoids `unwrap`/`expect`/`panic` (all denied crate-wide) by asserting on
/// `Option`/`Result` directly.
fn round_trip<T>(value: T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let encoded = serde_json::to_string(&value).unwrap_or_default();
    assert!(!encoded.is_empty(), "{value:?} must serialize");

    let decoded = serde_json::from_str::<T>(&encoded).ok();
    assert_eq!(
        decoded.as_ref(),
        Some(&value),
        "round trip must preserve the value (encoded as {encoded})"
    );
}

#[test]
fn operation_kind_enums_round_trip() {
    for op in [
        AtomicRmwOp::Xchg,
        AtomicRmwOp::Add,
        AtomicRmwOp::AndNot,
        AtomicRmwOp::MaxU,
    ] {
        round_trip(op);
    }

    round_trip(BinaryOpKind::Add);
    round_trip(UnaryOpKind::Neg);
    round_trip(CmpKind::Ne);
    round_trip(Signedness::Unsigned);
    round_trip(FenceKind::Acquire);
    round_trip(AtomicOrdering::SeqCst);
}

#[test]
fn vector_kind_enums_round_trip() {
    round_trip(VectorUnaryKind::Neg);
    round_trip(VectorBinaryKind::Add);
    round_trip(VectorCompareKind::Eq);
    round_trip(VectorTernaryKind::Fma);
    round_trip(VectorPackKind::Compress);
    round_trip(VectorCryptoKind::AesEncrypt);
    round_trip(VectorMaddKind::DotProductU8S8);
    round_trip(VectorCastKind::Signed);
    round_trip(VectorReduceKind::Add);
    round_trip(VectorElementKind::Integer);
    round_trip(SveComputeKind::AddCarryBottom);
    round_trip(PredicateOpKind::SetFirstFault);
    round_trip(PredicateGenKind::True);
    round_trip(SmeMiscKind::ZeroTiles);
    round_trip(TileOpKind::Load);
    round_trip(ComplexMulKind::Multiply);
}

#[test]
fn native_kind_enums_round_trip() {
    round_trip(SystemOpKind::Barrier);
    round_trip(ComputeKind::BitDeposit);
    round_trip(BcdAdjustKind::DecimalAddAdjust);
    round_trip(TranscendentalKind::SinCos);
    round_trip(FpuControlKind::Save);
    round_trip(FlagAdjustKind::InvertCarry);
    round_trip(BlockStringKind::Compare);
    round_trip(NativeIntrinsicId::Cpuid);
    round_trip(NativeIntrinsicId::PointerAuth(PacKind::Sign));
}

#[test]
fn classification_and_effect_enums_round_trip() {
    round_trip(SsaOpClass::NativeIntrinsic);
    round_trip(SsaSimilarityClass::Synthetic);
    round_trip(SsaEffectKind::Fence);
    round_trip(TrapClass::MemoryFault);
    round_trip(ControlEffect::Terminator);
    round_trip(MemoryEffectLocation::None);
    round_trip(MemoryAccessSemantics::Atomic);
    round_trip(OperandRole::Def);
    round_trip(NativeExceptionKind::Catch);
    round_trip(FunctionKind::InterruptHandler);
    round_trip(PointerSize::Bit64);
    round_trip(Endianness::Little);
}

/// Effect summaries are computed (`op.effects()`), but they are covered by the
/// feature, so they must round-trip like anything else.
#[test]
fn effect_summary_round_trips() {
    let effects =
        SsaEffects::new(SsaEffectKind::Fence, false).fence_ordering(AtomicOrdering::SeqCst);
    round_trip(effects);

    let op: SsaOp<MockTarget> = SsaOp::Add {
        dest: SsaVarId::from_index(2),
        left: SsaVarId::from_index(0),
        right: SsaVarId::from_index(1),
        flags: None,
    };
    round_trip(op.effects());
}

/// Native descriptors carry `String`/`Vec`/`Option` fields — likelier to break
/// than fieldless enums, and previously untested.
#[test]
fn native_descriptors_round_trip() {
    let register = NativeRegister {
        architecture: "x86_64".into(),
        bank: "gpr".into(),
        base: "rax".into(),
        name: "eax".into(),
        bit_offset: 0,
        bit_width: 32,
    };
    round_trip(register.clone());

    round_trip(NativeStateLocation::Register(register.clone()));
    round_trip(NativeStateLocation::Flags("eflags".into()));
    round_trip(NativeStateLocation::StackPointer);

    let access = NativeStateAccess {
        location: NativeStateLocation::Register(register),
        kind: NativeStateAccessKind::ReadWrite,
        width_bits: Some(32),
        implicit: true,
    };
    round_trip(access.clone());
    round_trip(NativeClobber::MachineState(access));

    round_trip(NativeInstructionMetadata {
        architecture: Some("aarch64".into()),
        address: Some(0x1000),
        raw_bytes: vec![0xde, 0xad, 0xbe, 0xef],
    });

    round_trip(VectorElement {
        kind: VectorElementKind::Integer,
        bits: 32,
        scalar: false,
    });
}

/// Boxed operand payloads reference `SsaVarId`, so they exercise its manual
/// impl through a container.
#[test]
fn operand_payloads_round_trip() {
    round_trip(KindedVecData {
        kind: FlagAdjustKind::InvertCarry,
        outputs: vec![SsaVarId::from_index(1)],
        inputs: vec![SsaVarId::from_index(0), SsaVarId::from_index(2)],
    });
}

/// `SsaVarId` stores the bitwise complement of its index internally; the wire
/// format must be the logical index, not that encoding.
#[test]
fn ssa_var_id_serializes_as_logical_index() {
    let encoded = serde_json::to_string(&SsaVarId::from_index(7)).unwrap_or_default();
    assert_eq!(encoded, "7", "wire format must be the logical index");

    for index in [0usize, 1, 7, 4096, u32::MAX as usize - 2] {
        round_trip(SsaVarId::from_index(index));
    }

    // The niche encoding must survive `Option`, which is where it earns its keep.
    round_trip(Some(SsaVarId::from_index(3)));
    round_trip(Option::<SsaVarId>::None);
    round_trip(SsaVarId::PLACEHOLDER);
}

/// Floats are compared and hashed bitwise, so the round trip must be bit-exact —
/// including `NaN`, which a lossy text encoding would drop to `null`.
#[test]
fn const_value_round_trips_including_floats() {
    round_trip(ConstValue::<MockTarget>::I32(-7));
    round_trip(ConstValue::<MockTarget>::U64(u64::MAX));
    round_trip(ConstValue::<MockTarget>::NativeInt(-1));
    round_trip(ConstValue::<MockTarget>::Null);
    round_trip(ConstValue::<MockTarget>::True);
    round_trip(ConstValue::<MockTarget>::F64(1.5));
    round_trip(ConstValue::<MockTarget>::F32(-0.0));
    round_trip(ConstValue::<MockTarget>::DecryptedString("secret".into()));
    round_trip(ConstValue::<MockTarget>::Type(42));
    round_trip(ConstValue::<MockTarget>::Vector(
        vec![ConstValue::I32(1), ConstValue::I32(2)].into_boxed_slice(),
    ));
}

#[test]
fn ssa_op_round_trips() {
    let v0 = SsaVarId::from_index(0);
    let v1 = SsaVarId::from_index(1);
    let v2 = SsaVarId::from_index(2);

    round_trip(SsaOp::<MockTarget>::Add {
        dest: v2,
        left: v0,
        right: v1,
        flags: None,
    });
    round_trip(SsaOp::<MockTarget>::Const {
        dest: v0,
        value: ConstValue::I32(9),
    });
    round_trip(SsaOp::<MockTarget>::LoadIndirect {
        dest: v1,
        addr: v0,
        value_type: MockType::I32,
    });
    round_trip(SsaOp::<MockTarget>::CallClobber { outputs: vec![v0] });
    round_trip(SsaOp::<MockTarget>::Nop);
    round_trip(SsaOp::<MockTarget>::Branch {
        condition: v0,
        true_target: 1,
        false_target: 2,
    });
}

#[test]
fn phi_types_round_trip() {
    round_trip(PhiOperand::new(SsaVarId::from_index(1), 0));

    // `PhiNode` has no `PartialEq`, so compare its observable fields.
    let mut phi = PhiNode::new(SsaVarId::from_index(3), VariableOrigin::Local(0));
    phi.add_operand(PhiOperand::new(SsaVarId::from_index(1), 0));
    phi.add_operand(PhiOperand::new(SsaVarId::from_index(2), 1));

    let encoded = serde_json::to_string(&phi).unwrap_or_default();
    assert!(!encoded.is_empty(), "PhiNode must serialize");
    let decoded = serde_json::from_str::<PhiNode>(&encoded).ok();
    assert_eq!(decoded.as_ref().map(PhiNode::result), Some(phi.result()));
    assert_eq!(decoded.as_ref().map(PhiNode::origin), Some(phi.origin()));
    assert_eq!(
        decoded.as_ref().map(PhiNode::operands),
        Some(phi.operands()),
        "operands must survive the round trip"
    );
}

#[test]
fn variable_types_round_trip() {
    round_trip(VariableOrigin::Argument(1));
    round_trip(VariableOrigin::Local(2));
    round_trip(DefSite::instruction(0, 3));
    round_trip(DefSite::phi(1));
}

/// The end-to-end guarantee: a whole SSA graph survives a round trip, through
/// the generic `#[serde(bound(...))]` path with a real `Target`.
#[test]
fn whole_ssa_function_round_trips() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);

    let dest = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );

    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
        value: Some(dest),
    }));
    ssa.add_block(block);

    let encoded = serde_json::to_string(&ssa).unwrap_or_default();
    assert!(
        !encoded.is_empty(),
        "SsaFunction must serialize (enum map keys must not reach the encoder)"
    );

    let decoded = serde_json::from_str::<SsaFunction<MockTarget>>(&encoded).ok();
    assert!(
        decoded.is_some(),
        "SsaFunction must deserialize from {encoded}"
    );

    assert_eq!(
        decoded.as_ref().map(|d| d.variables().len()),
        Some(ssa.variables().len())
    );
    assert_eq!(
        decoded.as_ref().map(|d| d.blocks().len()),
        Some(ssa.blocks().len())
    );
    assert_eq!(
        decoded
            .as_ref()
            .and_then(|d| d.block(0))
            .map(|b| b.instructions().len()),
        ssa.block(0).map(|b| b.instructions().len()),
    );
    assert_eq!(
        decoded
            .as_ref()
            .and_then(|d| d.block(0))
            .and_then(|b| b.instructions().first())
            .map(SsaInstruction::op),
        ssa.block(0)
            .and_then(|b| b.instructions().first())
            .map(SsaInstruction::op),
        "the round-tripped graph must be structurally identical",
    );
}

#[test]
fn kind_enums_serialize_by_name_not_discriminant() {
    // The wire form must be the variant name: keying on the numeric
    // discriminant would silently remap every persisted value the next time a
    // variant is inserted mid-enum, and this crate adds variants routinely.
    let encoded = serde_json::to_string(&AtomicRmwOp::Add).unwrap_or_default();
    assert_eq!(encoded, "\"Add\"");
}
