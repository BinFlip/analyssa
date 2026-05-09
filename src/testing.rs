//! Test fixtures for target-agnostic examples and downstream host tests.
//!
//! These helpers intentionally use [`crate::MockTarget`] so examples and tests
//! can exercise analyssa APIs without depending on a concrete instruction-set host.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::{
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Target,
    world::World,
};

/// A minimal [`Target`] impl for IR-core unit tests, doctests, and downstream
/// integration tests. Has no dependency on any host metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MockTarget;

/// Tiny stand-in type used to verify the IR core can carry an opaque
/// `T::Type` without depending on a host's type system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
            _ => None,
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

/// Creates a one-block `MockTarget` SSA function that returns an `i32` constant.
#[must_use]
pub fn const_i32_return(value: i32) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 0);
    let dest = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );

    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest,
        value: ConstValue::I32(value),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
        value: Some(dest),
    }));
    ssa.add_block(block);
    ssa
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
    use crate::Endianness;

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
