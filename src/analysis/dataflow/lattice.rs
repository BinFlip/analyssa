//! Lattice traits for data flow analysis.
//!
//! A lattice is a mathematical structure defining how abstract values combine
//! at control flow join/split points. This module provides the fundamental
//! traits that analysis domains must implement, along with lattice instances
//! for common types.
//!
//! # Lattice Theory
//!
//! For data flow analysis, lattices model the abstract values of program state:
//!
//! - **Partial Order**: Elements can be compared (≤)
//! - **Meet (∧)**: Greatest lower bound — combines information from multiple
//!   paths (intersection for must-analysis, union for may-analysis)
//! - **Join (∨)**: Least upper bound — merges information conservatively
//! - **Top (⊤)**: Greatest element ("no information" / "unknown")
//! - **Bottom (⊥)**: Least element ("conflict" / "all information")
//!
//! # Trait Hierarchy
//!
//! ```text
//! MeetSemiLattice (meet + is_bottom)
//!     |
//! JoinSemiLattice (join + is_top)
//!     |
//! Lattice (top + bottom + both operations)
//! ```
//!
//! # Forward vs Backward Analysis
//!
//! - **Forward analyses** (reaching definitions, constant propagation):
//!   Use `MeetSemiLattice::meet()` at join points (multiple predecessors)
//! - **Backward analyses** (liveness): Use `JoinSemiLattice::join()` at split
//!   points (multiple successors)
//!
//! The solver automatically selects the appropriate operation based on the
//! `Direction` enum. For `BitSet`-based analyses:
//! - Meet = union (for may-analysis like reaching definitions)
//! - Join = intersection (for must-analysis)
//!
//! # Requirements
//!
//! Meet operation must satisfy:
//! - Idempotent: `x.meet(x) = x`
//! - Commutative: `x.meet(y) = y.meet(x)`
//! - Associative: `x.meet(y.meet(z)) = (x.meet(y)).meet(z)`

use std::fmt::Debug;

use crate::bitset::BitSet;

/// A meet semi-lattice with a meet (greatest lower bound) operation.
///
/// The meet operation combines information from multiple control flow paths.
/// It must satisfy:
///
/// - **Idempotent**: `x.meet(x) = x`
/// - **Commutative**: `x.meet(y) = y.meet(x)`
/// - **Associative**: `x.meet(y.meet(z)) = (x.meet(y)).meet(z)`
///
/// # Examples
///
/// ```rust,ignore
/// use analyssa::analysis::dataflow::MeetSemiLattice;
///
/// impl MeetSemiLattice for ConstantLattice {
///     fn meet(&self, other: &Self) -> Self {
///         match (self, other) {
///             (Self::Top, x) | (x, Self::Top) => x.clone(),
///             (Self::Const(a), Self::Const(b)) if a == b => Self::Const(*a),
///             _ => Self::Bottom,
///         }
///     }
/// }
/// ```
pub trait MeetSemiLattice: Clone + Debug + PartialEq {
    /// Computes the meet (greatest lower bound) of two lattice elements.
    ///
    /// The meet represents combining information from two paths that merge.
    #[must_use]
    fn meet(&self, other: &Self) -> Self;

    /// Returns `true` if this is the bottom element.
    ///
    /// The bottom element represents "all information" or "conflict".
    /// Once bottom is reached, further meets cannot change the value.
    fn is_bottom(&self) -> bool;
}

/// A join semi-lattice with a join (least upper bound) operation.
///
/// The join operation combines information when paths split (for backward analysis)
/// or when we want to widen the approximation.
///
/// It must satisfy:
///
/// - **Idempotent**: `x.join(x) = x`
/// - **Commutative**: `x.join(y) = y.join(x)`
/// - **Associative**: `x.join(y.join(z)) = (x.join(y)).join(z)`
pub trait JoinSemiLattice: Clone + Debug + PartialEq {
    /// Computes the join (least upper bound) of two lattice elements.
    ///
    /// The join represents the least specific value that covers both inputs.
    #[must_use]
    fn join(&self, other: &Self) -> Self;

    /// Returns `true` if this is the top element.
    ///
    /// The top element represents "no information" or "unknown".
    /// It is the identity for meet: `x.meet(top) = x`.
    fn is_top(&self) -> bool;
}

/// A complete lattice with both meet and join operations.
///
/// Most data flow analyses operate over complete lattices, which have
/// both a greatest and least element, plus meet and join operations.
///
/// # Required Properties
///
/// - All properties of `MeetSemiLattice` and `JoinSemiLattice`
/// - **Absorption**: `x.meet(x.join(y)) = x` and `x.join(x.meet(y)) = x`
pub trait Lattice: MeetSemiLattice + JoinSemiLattice {
    /// Returns the top (⊤) element of the lattice.
    ///
    /// Top represents "no information" and is the identity for meet.
    fn top() -> Self;

    /// Returns the bottom (⊥) element of the lattice.
    ///
    /// Bottom represents "all information" or "conflict".
    fn bottom() -> Self;
}

// Lattice trait implementations for BitSet (defined in crate::utils::bitset)

impl MeetSemiLattice for BitSet {
    /// Meet is union for reaching definitions (may analysis).
    fn meet(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.union_with(other);
        result
    }

    fn is_bottom(&self) -> bool {
        // For may analysis, bottom is full set
        self.count() == self.len()
    }
}

impl JoinSemiLattice for BitSet {
    /// Join is intersection for reaching definitions.
    fn join(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.intersect_with(other);
        result
    }

    fn is_top(&self) -> bool {
        self.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset_meet_union() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(3);

        let mut b = BitSet::new(10);
        b.insert(2);
        b.insert(3);

        let result = a.meet(&b);

        // Meet is union: {1, 3} ∪ {2, 3} = {1, 2, 3}
        assert!(result.contains(1));
        assert!(result.contains(2));
        assert!(result.contains(3));
        assert!(!result.contains(0));
        assert!(!result.contains(4));
    }

    #[test]
    fn test_bitset_meet_idempotent() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(5);

        let result = a.meet(&a);

        // Idempotent: x.meet(x) = x
        assert_eq!(a, result);
    }

    #[test]
    fn test_bitset_meet_commutative() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(3);

        let mut b = BitSet::new(10);
        b.insert(2);
        b.insert(4);

        // Commutative: x.meet(y) = y.meet(x)
        assert_eq!(a.meet(&b), b.meet(&a));
    }

    #[test]
    fn test_bitset_meet_associative() {
        let mut a = BitSet::new(10);
        a.insert(1);

        let mut b = BitSet::new(10);
        b.insert(2);

        let mut c = BitSet::new(10);
        c.insert(3);

        // Associative: x.meet(y.meet(z)) = (x.meet(y)).meet(z)
        let left = a.meet(&b.meet(&c));
        let right = a.meet(&b).meet(&c);
        assert_eq!(left, right);
    }

    #[test]
    fn test_bitset_join_intersection() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(2);
        a.insert(3);

        let mut b = BitSet::new(10);
        b.insert(2);
        b.insert(3);
        b.insert(4);

        let result = a.join(&b);

        // Join is intersection: {1, 2, 3} ∩ {2, 3, 4} = {2, 3}
        assert!(!result.contains(1));
        assert!(result.contains(2));
        assert!(result.contains(3));
        assert!(!result.contains(4));
    }

    #[test]
    fn test_bitset_join_idempotent() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(5);

        let result = a.join(&a);

        // Idempotent: x.join(x) = x
        assert_eq!(a, result);
    }

    #[test]
    fn test_bitset_join_commutative() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(3);

        let mut b = BitSet::new(10);
        b.insert(2);
        b.insert(3);

        // Commutative: x.join(y) = y.join(x)
        assert_eq!(a.join(&b), b.join(&a));
    }

    #[test]
    fn test_bitset_is_top_empty() {
        let empty = BitSet::new(10);
        assert!(empty.is_top());

        let mut non_empty = BitSet::new(10);
        non_empty.insert(0);
        assert!(!non_empty.is_top());
    }

    #[test]
    fn test_bitset_is_bottom_full() {
        let full = BitSet::full(10);
        assert!(full.is_bottom());

        let mut partial = BitSet::new(10);
        partial.insert(0);
        assert!(!partial.is_bottom());
    }

    #[test]
    fn test_bitset_meet_with_empty() {
        let empty = BitSet::new(10);

        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(2);

        // Meet with empty (top) should give the other set
        let result = a.meet(&empty);
        assert!(result.contains(1));
        assert!(result.contains(2));
        assert_eq!(result.count(), 2);
    }

    #[test]
    fn test_bitset_join_with_empty() {
        let empty = BitSet::new(10);

        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(2);

        // Join with empty (top) should give empty
        let result = a.join(&empty);
        assert!(result.is_empty());
    }
}
