//! Function kind classification for the SSA IR.
//!
//! Distinguishes normal functions from interrupt service routines (ISRs) and
//! other special function types. The kind is stored on [`SsaFunction`](super::SsaFunction)
//! and drives validation, code generation, and pass behavior.
//!
//! # Interrupt Handlers
//!
//! Functions marked as [`InterruptHandler`](FunctionKind::InterruptHandler) use
//! [`SsaOp::InterruptReturn`](crate::ir::ops::SsaOp::InterruptReturn) instead of
//! [`SsaOp::Return`](crate::ir::ops::SsaOp::Return) to return from the handler.
//! The frontend is responsible for:
//!
//! - Setting the function kind during SSA construction
//! - Emitting `InterruptReturn` as the terminator of the exit block
//! - Preserving any interrupt-frame state (saved registers, error code, etc.)
//!   as explicit SSA variables
//!
//! # Examples
//!
//! ```rust
//! use analyssa::ir::function::{FunctionKind, SsaFunction};
//! use analyssa::MockTarget;
//!
//! let mut func = SsaFunction::<MockTarget>::new(0, 0);
//! assert_eq!(func.kind(), FunctionKind::Normal);
//!
//! func.set_kind(FunctionKind::InterruptHandler);
//! assert_eq!(func.kind(), FunctionKind::InterruptHandler);
//! ```

use std::fmt;

/// Classification of a function's execution context.
///
/// Defaults to [`Normal`](FunctionKind::Normal) for all functions constructed
/// via [`SsaFunction::new`](super::SsaFunction::new) or
/// [`SsaFunction::with_capacity`](super::SsaFunction::with_capacity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionKind {
    /// A regular function with standard call/return semantics.
    ///
    /// Terminated by [`SsaOp::Return`](crate::ir::ops::SsaOp::Return). May
    /// contain exception handlers and `Throw`/`Rethrow` operations.
    Normal,

    /// An interrupt service routine (ISR) or exception handler entry.
    ///
    /// Terminated by [`SsaOp::InterruptReturn`](crate::ir::ops::SsaOp::InterruptReturn)
    /// which restores the interrupted context rather than returning to a caller
    /// in the ordinary sense. Frontends targeting ARM, MIPS, RISC-V, or x86
    /// IDT entries should use this kind.
    ///
    /// # Validation notes
    ///
    /// - An `InterruptHandler` function should not contain `Return` ops
    ///   (all paths must end with `InterruptReturn`)
    /// - ISRs may have restricted register usage and no exception handlers
    ///   (depends on the target ISA)
    InterruptHandler,
}

impl FunctionKind {
    /// Returns `true` if this is an interrupt service routine.
    #[must_use]
    pub const fn is_interrupt_handler(self) -> bool {
        matches!(self, Self::InterruptHandler)
    }

    /// Returns `true` if this is a normal function.
    #[must_use]
    pub const fn is_normal(self) -> bool {
        matches!(self, Self::Normal)
    }
}

impl fmt::Display for FunctionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::InterruptHandler => write!(f, "interrupt_handler"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_kind_is_normal() {
        assert!(FunctionKind::Normal.is_normal());
        assert!(!FunctionKind::Normal.is_interrupt_handler());
    }

    #[test]
    fn interrupt_handler_classification() {
        assert!(FunctionKind::InterruptHandler.is_interrupt_handler());
        assert!(!FunctionKind::InterruptHandler.is_normal());
    }

    #[test]
    fn display_normal() {
        assert_eq!(FunctionKind::Normal.to_string(), "normal");
    }

    #[test]
    fn display_interrupt_handler() {
        assert_eq!(
            FunctionKind::InterruptHandler.to_string(),
            "interrupt_handler"
        );
    }

    #[test]
    fn kind_is_copy_and_eq() {
        assert_eq!(FunctionKind::Normal, FunctionKind::Normal);
        assert_eq!(
            FunctionKind::InterruptHandler,
            FunctionKind::InterruptHandler
        );
        assert_ne!(FunctionKind::Normal, FunctionKind::InterruptHandler);
    }
}
