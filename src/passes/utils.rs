//! Shared utility functions used across SSA transformation passes.
//!
//! Provides:
//!
//! - [`is_power_of_two`] — detects whether an integer is a positive power
//!   of two (used by strength reduction).
//! - [`resolve_chain`] — follows a chain of transitive mappings to find
//!   the ultimate target (used by trampoline resolution, copy propagation,
//!   and control flow simplification).

use std::collections::{BTreeMap, BTreeSet};

/// Test whether an integer is a positive power of two.
///
/// Returns `Some(exponent)` where `exponent` is `log2(value)` if `value`
/// is `2^n` for some `0 ≤ n ≤ 63`. Returns `None` for non-positive or
/// non-power-of-two values.
///
/// Used by strength reduction (`x * 2^n → x << n`, `x / 2^n → x >> n`,
/// `x % 2^n → x & (2^n - 1)`).
///
/// # Arguments
///
/// * `value` — The integer to test. Must be positive and a power of two.
///
/// # Returns
///
/// `Some(exponent)` where `2^exponent == value`, or `None` if `value` is
/// not a positive power of two.
#[must_use]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_truncation)]
pub fn is_power_of_two(value: i64) -> Option<u8> {
    if value <= 0 {
        return None;
    }
    let value = value as u64;
    if value.is_power_of_two() {
        Some(value.trailing_zeros() as u8)
    } else {
        None
    }
}

/// Follow a chain of transitive mappings to find the ultimate target.
///
/// Given a map `{key → value}` where values may themselves be keys,
/// follows the chain until reaching a value that is not a key. Handles
/// cycles by stopping when a previously visited key is encountered.
///
/// Used by:
/// - Trampoline resolution: `block → block → block` becomes `block → ultimate`.
/// - Copy propagation: `var → var → var` becomes `var → ultimate`.
/// - Control flow simplification: resolving branch targets through chains.
///
/// # Arguments
///
/// * `map` — The transitive mapping (e.g., `{1: 2, 2: 3, 3: 4}`).
/// * `start` — The starting key.
///
/// # Returns
///
/// The ultimate target after following the chain. If `start` is not a key
/// in `map`, returns `start` unchanged.
#[must_use]
pub fn resolve_chain<K>(map: &BTreeMap<K, K>, start: K) -> K
where
    K: Copy + Ord,
{
    let mut current = start;
    let mut visited = BTreeSet::new();

    while let Some(&next) = map.get(&current) {
        if !visited.insert(current) {
            break;
        }
        current = next;
    }

    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_mappings() {
        let mut map = BTreeMap::new();
        map.insert(1, 2);
        map.insert(2, 3);
        map.insert(3, 4);

        assert_eq!(resolve_chain(&map, 1), 4);
        assert_eq!(resolve_chain(&map, 2), 4);
        assert_eq!(resolve_chain(&map, 3), 4);
        assert_eq!(resolve_chain(&map, 4), 4);
    }

    #[test]
    fn handles_cycles() {
        let mut map = BTreeMap::new();
        map.insert(1, 2);
        map.insert(2, 1);

        let result = resolve_chain(&map, 1);
        assert!(result == 1 || result == 2);
    }

    #[test]
    fn single_step() {
        let mut map = BTreeMap::new();
        map.insert(5, 10);

        assert_eq!(resolve_chain(&map, 5), 10);
    }

    #[test]
    fn empty_map() {
        let map: BTreeMap<usize, usize> = BTreeMap::new();
        assert_eq!(resolve_chain(&map, 42), 42);
    }

    #[test]
    fn self_loop() {
        let mut map = BTreeMap::new();
        map.insert(1, 1);

        assert_eq!(resolve_chain(&map, 1), 1);
    }
}
