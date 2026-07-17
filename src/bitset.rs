//! A bit vector for efficient set operations.
//!
//! This module provides a compact bit set implementation optimized for
//! set operations commonly used in data flow analysis and other algorithms
//! that track sets of entities identified by small integers.
//!
//! # Features
//!
//! - Efficient storage: 64 elements per word
//! - Set operations: union, intersection, difference
//! - Iteration over set elements
//! - Clone-on-write friendly design
//!
//! # Example
//!
//! ```rust
//! use analyssa::BitSet;
//!
//! let mut set = BitSet::new(100);
//! set.insert(0);
//! set.insert(50);
//! set.insert(99);
//!
//! assert!(set.contains(50));
//! assert_eq!(set.count(), 3);
//!
//! for idx in set.iter() {
//!     println!("Set contains: {}", idx);
//! }
//! ```

/// A bit vector for efficient set operations.
///
/// This is commonly used for analyses that track sets of definitions,
/// variables, or other entities identified by small integers.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct BitSet {
    /// The bits, stored as a vector of words.
    words: Vec<u64>,
    /// The number of bits in the set.
    len: usize,
}

impl BitSet {
    /// Creates a new empty bit set with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let num_words = capacity.div_ceil(64);
        Self {
            words: vec![0; num_words],
            len: capacity,
        }
    }

    /// Creates a new bit set with all bits set.
    #[must_use]
    pub fn full(capacity: usize) -> Self {
        let num_words = capacity.div_ceil(64);
        let mut words = vec![u64::MAX; num_words];

        // Clear the excess bits in the last word
        if !capacity.is_multiple_of(64) {
            if let Some(last) = words.last_mut() {
                *last = (1u64 << (capacity % 64)).saturating_sub(1);
            }
        }

        Self {
            words,
            len: capacity,
        }
    }

    /// Returns the capacity of this bit set.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the bit set has no bits set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Sets the bit at the given index.
    ///
    /// Returns `true` if the bit was newly set (was previously unset).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    pub fn insert(&mut self, index: usize) -> bool {
        assert!(index < self.len, "index out of bounds");
        let word = index / 64;
        let bit = index % 64;
        let mask = 1u64 << bit;
        let Some(slot) = self.words.get_mut(word) else {
            return false;
        };
        let was_set = *slot & mask != 0;
        *slot |= mask;
        !was_set
    }

    /// Clears the bit at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    pub fn remove(&mut self, index: usize) {
        assert!(index < self.len, "index out of bounds");
        let word = index / 64;
        let bit = index % 64;
        if let Some(slot) = self.words.get_mut(word) {
            *slot &= !(1u64 << bit);
        }
    }

    /// Returns `true` if the bit at the given index is set.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        assert!(index < self.len, "index out of bounds");
        let word = index / 64;
        let bit = index % 64;
        self.words
            .get(word)
            .is_some_and(|w| (w & (1u64 << bit)) != 0)
    }

    /// Returns `true` if the bit at `index` is set, treating an out-of-range
    /// index as unset instead of panicking.
    ///
    /// Use this where the index comes from data that may legitimately name a
    /// position outside the set — e.g. a CFG terminator referencing a block that
    /// was never recovered, where "not in the set" is the correct answer rather
    /// than a programming error.
    #[must_use]
    pub fn contains_checked(&self, index: usize) -> bool {
        index < self.len && self.contains(index)
    }

    /// Sets the bit at `index`, ignoring an out-of-range index instead of
    /// panicking. Returns `true` if the bit was newly set.
    ///
    /// The bounds-tolerant counterpart to [`Self::insert`]; see
    /// [`Self::contains_checked`] for when that is the right behavior.
    pub fn insert_checked(&mut self, index: usize) -> bool {
        index < self.len && self.insert(index)
    }

    /// Returns the number of bits set.
    #[must_use]
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Returns `true` if every bit in `0..len()` is set.
    ///
    /// Short-circuits on the first non-full word instead of popcounting the
    /// whole set, which is cheaper than comparing [`count`](Self::count) to
    /// [`len`](Self::len) for the common "is this a full set" lattice check.
    #[must_use]
    pub fn is_full(&self) -> bool {
        let full_words = self.len / 64;
        let rem = self.len % 64;
        for i in 0..full_words {
            match self.words.get(i) {
                Some(&w) if w == u64::MAX => {}
                _ => return false,
            }
        }
        if rem != 0 {
            // Low `rem` bits set, computed without subtraction.
            let mask = !(u64::MAX << rem);
            match self.words.get(full_words) {
                Some(&w) if (w & mask) == mask => {}
                _ => return false,
            }
        }
        true
    }

    /// Clears all bits.
    pub fn clear(&mut self) {
        for word in &mut self.words {
            *word = 0;
        }
    }

    /// Sets all bits.
    pub fn fill(&mut self) {
        for word in &mut self.words {
            *word = u64::MAX;
        }
        // Clear excess bits in last word
        if !self.len.is_multiple_of(64) {
            if let Some(last) = self.words.last_mut() {
                *last = (1u64 << (self.len % 64)).saturating_sub(1);
            }
        }
    }

    /// Computes the union with another bit set (in place).
    ///
    /// Returns `true` if `self` changed.
    pub fn union_with(&mut self, other: &Self) -> bool {
        assert_eq!(self.len, other.len, "bit sets must have same length");
        let mut changed = false;
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            let old = *a;
            *a |= *b;
            changed |= old != *a;
        }
        changed
    }

    /// Computes the intersection with another bit set (in place).
    ///
    /// Returns `true` if `self` changed.
    pub fn intersect_with(&mut self, other: &Self) -> bool {
        assert_eq!(self.len, other.len, "bit sets must have same length");
        let mut changed = false;
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            let old = *a;
            *a &= *b;
            changed |= old != *a;
        }
        changed
    }

    /// Computes the difference with another bit set (in place).
    ///
    /// Removes all bits that are set in `other` from `self`.
    /// Returns `true` if `self` changed.
    pub fn difference_with(&mut self, other: &Self) -> bool {
        assert_eq!(self.len, other.len, "bit sets must have same length");
        let mut changed = false;
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            let old = *a;
            *a &= !*b;
            changed |= old != *a;
        }
        changed
    }

    /// Returns an iterator over the indices of set bits.
    pub fn iter(&self) -> BitSetIter<'_> {
        BitSetIter {
            set: self,
            word_idx: 0,
            bit_idx: 0,
        }
    }
}

impl std::fmt::Debug for BitSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        let mut first = true;
        for i in self.iter() {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{i}")?;
            first = false;
        }
        write!(f, "}}")
    }
}

/// Iterator over the set bits in a `BitSet`.
pub struct BitSetIter<'a> {
    set: &'a BitSet,
    word_idx: usize,
    bit_idx: usize,
}

impl Iterator for BitSetIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.word_idx < self.set.words.len() {
            let word = *self.set.words.get(self.word_idx)?;
            // Mask off the bits we have already consumed in this word, then
            // jump straight to the next set bit. This makes iteration cost
            // O(set bits + words) instead of O(capacity): all-zero words and
            // long runs of zero bits are skipped in a single step.
            let masked = if self.bit_idx >= 64 {
                0
            } else {
                word & (u64::MAX << self.bit_idx)
            };
            if masked == 0 {
                self.word_idx = self.word_idx.saturating_add(1);
                self.bit_idx = 0;
                continue;
            }
            let bit = masked.trailing_zeros() as usize;
            let idx = self
                .word_idx
                .checked_mul(64)
                .and_then(|v| v.checked_add(bit))?;
            if idx >= self.set.len {
                return None;
            }
            self.bit_idx = bit.saturating_add(1);
            return Some(idx);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset_basic() {
        let mut bs = BitSet::new(100);
        assert!(bs.is_empty());
        assert_eq!(bs.count(), 0);

        bs.insert(0);
        bs.insert(50);
        bs.insert(99);

        assert!(!bs.is_empty());
        assert_eq!(bs.count(), 3);
        assert!(bs.contains(0));
        assert!(bs.contains(50));
        assert!(bs.contains(99));
        assert!(!bs.contains(1));
    }

    /// The bounds-tolerant accessors report/ignore out-of-range indices instead
    /// of panicking, and otherwise behave exactly like their strict versions.
    #[test]
    fn checked_accessors_tolerate_out_of_range_indices() {
        let mut bs = BitSet::new(64);

        // In range: identical to `insert`/`contains`.
        assert!(bs.insert_checked(0), "newly set bit reports true");
        assert!(!bs.insert_checked(0), "already-set bit reports false");
        assert!(bs.contains_checked(0));
        assert!(!bs.contains_checked(1));

        // Out of range: unset rather than a panic, and inserting is a no-op.
        assert!(!bs.contains_checked(64));
        assert!(!bs.contains_checked(usize::MAX));
        assert!(!bs.insert_checked(64), "out-of-range insert sets nothing");
        assert_eq!(bs.count(), 1, "out-of-range insert must not grow the set");
    }

    #[test]
    fn test_bitset_remove() {
        let mut bs = BitSet::new(100);
        bs.insert(42);
        assert!(bs.contains(42));

        bs.remove(42);
        assert!(!bs.contains(42));
    }

    #[test]
    fn test_bitset_full() {
        let bs = BitSet::full(100);
        assert_eq!(bs.count(), 100);
        for i in 0..100 {
            assert!(bs.contains(i), "bit {i} should be set");
        }
    }

    #[test]
    fn test_bitset_is_full() {
        // Empty capacity is vacuously full.
        assert!(BitSet::new(0).is_full());
        // Exercise both word-aligned and partial-word capacities.
        for cap in [1usize, 63, 64, 65, 100, 128, 129] {
            let full = BitSet::full(cap);
            assert!(full.is_full(), "full({cap}) should be full");
            assert_eq!(full.is_full(), full.count() == full.len());

            let mut almost = BitSet::full(cap);
            almost.remove(cap.saturating_sub(1));
            assert!(!almost.is_full(), "full({cap}) minus a bit is not full");

            assert!(!BitSet::new(cap).is_full() || cap == 0);
        }
    }

    #[test]
    fn test_bitset_union() {
        let mut a = BitSet::new(100);
        let mut b = BitSet::new(100);

        a.insert(0);
        a.insert(1);
        b.insert(1);
        b.insert(2);

        let changed = a.union_with(&b);
        assert!(changed);
        assert!(a.contains(0));
        assert!(a.contains(1));
        assert!(a.contains(2));
        assert_eq!(a.count(), 3);
    }

    #[test]
    fn test_bitset_intersect() {
        let mut a = BitSet::new(100);
        let mut b = BitSet::new(100);

        a.insert(0);
        a.insert(1);
        a.insert(2);
        b.insert(1);
        b.insert(2);
        b.insert(3);

        let changed = a.intersect_with(&b);
        assert!(changed);
        assert!(!a.contains(0));
        assert!(a.contains(1));
        assert!(a.contains(2));
        assert!(!a.contains(3));
        assert_eq!(a.count(), 2);
    }

    #[test]
    fn test_bitset_difference() {
        let mut a = BitSet::new(100);
        let mut b = BitSet::new(100);

        a.insert(0);
        a.insert(1);
        a.insert(2);
        b.insert(1);

        let changed = a.difference_with(&b);
        assert!(changed);
        assert!(a.contains(0));
        assert!(!a.contains(1));
        assert!(a.contains(2));
        assert_eq!(a.count(), 2);
    }

    #[test]
    fn test_bitset_iter() {
        let mut bs = BitSet::new(100);
        bs.insert(5);
        bs.insert(42);
        bs.insert(99);

        let bits: Vec<_> = bs.iter().collect();
        assert_eq!(bits, vec![5, 42, 99]);
    }

    #[test]
    fn test_bitset_clear_fill() {
        let mut bs = BitSet::new(100);
        bs.insert(50);
        assert_eq!(bs.count(), 1);

        bs.clear();
        assert!(bs.is_empty());

        bs.fill();
        assert_eq!(bs.count(), 100);
    }
}
