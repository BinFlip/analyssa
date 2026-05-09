//! Exception handler preservation through SSA transformations.
//!
//! This module provides [`SsaExceptionHandler`] which preserves exception handler metadata
//! from the original method body through SSA construction, optimization, and code generation.
//! Without this preservation, exception handler regions would become invalid when SSA
//! transformations change instruction sizes or reorder blocks.
//!
//! # Preservation Pipeline
//!
//! 1. **Original offsets**: The raw IL byte offsets for try/handler regions are captured
//!    during initial SSA construction and stored verbatim in each `SsaExceptionHandler`.
//! 2. **Block mapping**: During SSA construction, the offset-based regions are translated
//!    to block-index-based regions (`try_start_block`, `handler_start_block`, etc.).
//! 3. **Block remapping**: During canonicalization (`remap_block_indices`), block indices
//!    are updated to reflect block removal and renumbering.
//! 4. **Offset regeneration**: During code generation, block offsets are used to compute
//!    new IL offsets for the output method body.
//!
//! # Edge Cases
//!
//! - **Empty block removal**: When a block at an exclusive end boundary is removed,
//!   `remap_block_indices` finds the next surviving block to preserve boundary semantics.
//! - **Multiple handlers**: Each handler is preserved independently; there is no limit
//!   on the number of handlers per method.
//! - **Filter handlers**: The `class_token_or_filter` field serves double duty as either
//!   a caught exception type token or the offset of a filter expression, depending on flags.

use crate::target::Target;

/// Exception handler information preserved in SSA form.
///
/// Stores both the original IL byte offsets and the SSA block index mapping for
/// a single exception handler clause (try/catch/finally/fault/filter). The original
/// offsets are preserved verbatim from the method body; block indices are set during
/// SSA construction and remapped during canonicalization and code generation.
///
/// # Fields
///
/// | Field | Purpose |
/// |-------|---------|
/// | `flags` | Host-defined exception handler kind (EXCEPTION, FILTER, FINALLY, FAULT) |
/// | `try_offset`/`try_length` | Original IL byte range of the protected try block |
/// | `handler_offset`/`handler_length` | Original IL byte range of the handler code |
/// | `class_token_or_filter` | Caught exception type token or filter offset |
/// | `*_block` fields | SSA block indices (set during SSA construction) |
///
/// Generic over the host `Target` so `flags` carries a host-defined exception-kind type.
#[derive(Debug, Clone)]
pub struct SsaExceptionHandler<T: Target> {
    /// Host-defined flags identifying the handler kind:
    /// EXCEPTION, FILTER, FINALLY, or FAULT. For CIL targets, this is
    /// `ExceptionHandlerFlags` from the original method metadata.
    pub flags: T::ExceptionKind,

    /// Original IL byte offset of the protected try region start.
    pub try_offset: u32,

    /// Length of the protected try region in IL bytes.
    pub try_length: u32,

    /// Original IL byte offset of the handler code start.
    pub handler_offset: u32,

    /// Length of the handler code in IL bytes.
    pub handler_length: u32,

    /// Dual-purpose field: for EXCEPTION handlers this is the metadata token
    /// of the caught exception type; for FILTER handlers this is the IL offset
    /// of the filter expression.
    pub class_token_or_filter: u32,

    /// SSA block index where the try region starts.
    /// Set during SSA construction from offset-to-block mapping.
    pub try_start_block: Option<usize>,

    /// SSA block index where the try region ends (exclusive).
    /// Set during SSA construction. If the boundary block is removed during
    /// canonicalization, [`remap_block_indices`](Self::remap_block_indices)
    /// advances to the next surviving block.
    pub try_end_block: Option<usize>,

    /// SSA block index where the handler code starts.
    /// Set during SSA construction from the handler offset.
    pub handler_start_block: Option<usize>,

    /// SSA block index where the handler code ends (exclusive).
    /// Set during SSA construction. Same exclusive-boundary behavior as
    /// [`try_end_block`](Self::try_end_block).
    pub handler_end_block: Option<usize>,

    /// SSA block index where the filter expression starts.
    /// Only meaningful for FILTER-type handlers. Set during SSA construction.
    pub filter_start_block: Option<usize>,
}

/// A concrete [`ExceptionKind`](crate::target::Target::ExceptionKind) for native
/// ISA targets (x86 SEH, DWARF EH on Linux/ARM/RISC-V, etc.).
///
/// This enum provides a ready-to-use handler-kind type for frontends that target
/// native instruction sets rather than CIL. Use it as the associated type:
///
/// ```ignore
/// impl Target for MyX86Target {
///     type ExceptionKind = NativeExceptionKind;
///     // ...
/// }
/// ```
///
/// # Variant Mapping
///
/// | Variant | SEH (x86 Windows) | DWARF (Linux) | CIL equivalent |
/// |---------|-------------------|----------------|----------------|
/// | `Catch` | `__except` block | `DW_EH_encoding` landing pad | `EXCEPTION` |
/// | `Filter` | `__except(filter)` expression | — | `FILTER` |
/// | `Finally` | `__finally` block | — | `FINALLY` |
/// | `Fault` | Vectored exception handler | — | `FAULT` |
///
/// # Helper
///
/// Use [`native_is_filter_handler`] as the `Target::is_filter_handler` implementation
/// to correctly identify `Filter` variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeExceptionKind {
    /// Catch handler — runs when an exception of a matching type occurs.
    /// Maps to SEH `__except`, DWARF landing pads, and CIL `EXCEPTION`.
    Catch,

    /// Filter handler — runs a user-supplied predicate to decide whether
    /// to catch the exception. Maps to SEH `__except(filter)` and CIL `FILTER`.
    Filter,

    /// Finally handler — always executes when the try block exits (normally
    /// or via exception). Maps to SEH `__finally` and CIL `FINALLY`.
    Finally,

    /// Fault handler — like finally but only executes on exception (not
    /// normal exit). Maps to CIL `FAULT` and Vectored Exception Handlers.
    Fault,
}

/// Returns `true` if `flags` denotes a [`Filter`](NativeExceptionKind::Filter) handler.
///
/// Use this as the `Target::is_filter_handler` implementation when using
/// [`NativeExceptionKind`] as your `ExceptionKind`:
///
/// ```ignore
/// impl Target for MyTarget {
///     type ExceptionKind = NativeExceptionKind;
///
///     fn is_filter_handler(flags: &Self::ExceptionKind) -> bool {
///         native_is_filter_handler(flags)
///     }
/// }
/// ```
#[must_use]
pub fn native_is_filter_handler(flags: &NativeExceptionKind) -> bool {
    matches!(flags, NativeExceptionKind::Filter)
}

/// Finds the next surviving block index at or after `start` in the remap table.
///
/// Used for exclusive end-block boundaries (`try_end_block`, `handler_end_block`).
/// When an end-boundary block is removed during canonicalization, we need to find the
/// next block that survived to preserve the boundary semantics.
fn find_next_surviving(block_remap: &[Option<usize>], start: usize) -> Option<usize> {
    block_remap.get(start..)?.iter().find_map(|entry| *entry)
}

impl<T: Target> SsaExceptionHandler<T> {
    /// Returns the filter offset for FILTER handlers.
    ///
    /// Routes through `Target::is_filter_handler` so non-CIL hosts that don't
    /// distinguish filter handlers always observe `None` here.
    #[must_use]
    pub fn filter_offset(&self) -> Option<u32> {
        if T::is_filter_handler(&self.flags) {
            Some(self.class_token_or_filter)
        } else {
            None
        }
    }

    /// Checks if block indices have been set for offset remapping.
    #[must_use]
    pub fn has_block_mapping(&self) -> bool {
        self.try_start_block.is_some() && self.handler_start_block.is_some()
    }

    /// Remaps all block index fields using the provided canonicalization remapping.
    ///
    /// Called during [`crate::ir::function::SsaFunction::canonicalize`] to update exception handler block
    /// references after empty blocks are removed and remaining blocks are renumbered.
    ///
    /// # Start blocks vs. End blocks
    ///
    /// - **Start blocks** (`try_start_block`, `handler_start_block`, `filter_start_block`):
    ///   These map exactly through `block_remap`. If the referenced block was removed,
    ///   the field becomes `None`.
    ///
    /// - **End blocks** (`try_end_block`, `handler_end_block`): These are exclusive
    ///   boundaries. If the boundary block was removed, we advance to the next
    ///   surviving block via `find_next_surviving` (private helper) to preserve
    ///   the boundary semantics.
    ///
    /// # Arguments
    ///
    /// * `block_remap` - A slice indexed by old block ID, where each entry is:
    ///   - `Some(new_id)` if the block was kept and renumbered to `new_id`
    ///   - `None` if the block was removed
    ///
    /// # Example
    ///
    /// ```text
    /// Before canonicalization: blocks [0, 1, 2, 3, 4]  (block 1 removed)
    /// After canonicalization:  blocks [0, 2, 3, 4] → renumbered [0, 1, 2, 3]
    /// block_remap = [Some(0), None, Some(1), Some(2), Some(3)]
    /// ```
    pub fn remap_block_indices(&mut self, block_remap: &[Option<usize>]) {
        // Start blocks: must map exactly (protected, so should always survive)
        self.try_start_block = self
            .try_start_block
            .and_then(|idx| block_remap.get(idx).copied().flatten());

        // End blocks (exclusive boundaries): if removed, use next surviving block
        self.try_end_block = self.try_end_block.and_then(|idx| {
            block_remap
                .get(idx)
                .copied()
                .flatten()
                .or_else(|| find_next_surviving(block_remap, idx))
        });

        self.handler_start_block = self
            .handler_start_block
            .and_then(|idx| block_remap.get(idx).copied().flatten());

        // End block (exclusive boundary): if removed, use next surviving block
        self.handler_end_block = self.handler_end_block.and_then(|idx| {
            block_remap
                .get(idx)
                .copied()
                .flatten()
                .or_else(|| find_next_surviving(block_remap, idx))
        });

        self.filter_start_block = self
            .filter_start_block
            .and_then(|idx| block_remap.get(idx).copied().flatten());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::MockTarget;

    fn handler(flags: u32) -> SsaExceptionHandler<MockTarget> {
        SsaExceptionHandler {
            flags,
            try_offset: 10,
            try_length: 20,
            handler_offset: 30,
            handler_length: 40,
            class_token_or_filter: 50,
            try_start_block: Some(0),
            try_end_block: Some(2),
            handler_start_block: Some(3),
            handler_end_block: Some(5),
            filter_start_block: Some(4),
        }
    }

    #[test]
    fn mock_target_never_reports_filter_offsets() {
        assert_eq!(handler(0).filter_offset(), None);
    }

    #[test]
    fn block_mapping_requires_try_and_handler_starts() {
        let mut mapped = handler(0);
        assert!(mapped.has_block_mapping());

        mapped.handler_start_block = None;
        assert!(!mapped.has_block_mapping());
    }

    #[test]
    fn remap_updates_surviving_blocks_and_exclusive_end_boundaries() {
        let mut mapped = handler(0);

        mapped.remap_block_indices(&[Some(0), None, None, Some(1), Some(2), Some(3)]);

        assert_eq!(mapped.try_start_block, Some(0));
        assert_eq!(mapped.try_end_block, Some(1));
        assert_eq!(mapped.handler_start_block, Some(1));
        assert_eq!(mapped.handler_end_block, Some(3));
        assert_eq!(mapped.filter_start_block, Some(2));
    }

    #[test]
    fn remap_clears_removed_starts_and_out_of_bounds_indices() {
        let mut mapped = handler(0);

        mapped.remap_block_indices(&[None, Some(0), Some(1)]);

        assert_eq!(mapped.try_start_block, None);
        assert_eq!(mapped.try_end_block, Some(1));
        assert_eq!(mapped.handler_start_block, None);
        assert_eq!(mapped.handler_end_block, None);
        assert_eq!(mapped.filter_start_block, None);
    }

    // -----------------------------------------------------------------------
    // NativeExceptionKind tests
    // -----------------------------------------------------------------------

    #[test]
    fn native_exception_kind_variants_are_distinct() {
        assert_ne!(NativeExceptionKind::Catch, NativeExceptionKind::Filter);
        assert_ne!(NativeExceptionKind::Catch, NativeExceptionKind::Finally);
        assert_ne!(NativeExceptionKind::Catch, NativeExceptionKind::Fault);
        assert_ne!(NativeExceptionKind::Filter, NativeExceptionKind::Finally);
        assert_ne!(NativeExceptionKind::Filter, NativeExceptionKind::Fault);
        assert_ne!(NativeExceptionKind::Finally, NativeExceptionKind::Fault);
    }

    #[test]
    fn native_filter_handler_is_correctly_identified() {
        assert!(!native_is_filter_handler(&NativeExceptionKind::Catch));
        assert!(native_is_filter_handler(&NativeExceptionKind::Filter));
        assert!(!native_is_filter_handler(&NativeExceptionKind::Finally));
        assert!(!native_is_filter_handler(&NativeExceptionKind::Fault));
    }

    #[test]
    fn native_exception_kind_is_clone_and_eq() {
        let a = NativeExceptionKind::Catch;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn native_filter_handler_identified_by_helper() {
        assert!(!native_is_filter_handler(&NativeExceptionKind::Catch));
        assert!(native_is_filter_handler(&NativeExceptionKind::Filter));
        assert!(!native_is_filter_handler(&NativeExceptionKind::Finally));
        assert!(!native_is_filter_handler(&NativeExceptionKind::Fault));
    }
}
