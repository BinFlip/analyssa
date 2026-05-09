//! Error and result types used throughout the analyssa codebase.
//!
//! This module defines the primary error type [`Error`], which wraps a
//! free-form message string. All analyssa APIs that can fail return
//! [`Result<T>`] (aliased from `std::result::Result<T, Error>`), keeping
//! the error surface minimal so hosts can wrap it in their own error enum
//! via `From<Error>`.
//!
//! # Errors
//!
//! Functions and methods in analyssa return [`Result`] when they can encounter
//! graph inconsistencies, SSA validation failures, or other unexpected
//! conditions during construction, rebuilding, or transformation passes.
//!
//! # Aliases
//!
//! [`GraphError`] is a type alias for [`Error`] used specifically in graph
//! algorithm contexts, providing semantic clarity without introducing a
//! separate type.

use thiserror::Error;

/// Primary error type for analyssa operations.
///
/// Wraps a free-form message string describing the failure. This single
/// error type is used across all analyssa subsystems: graph algorithms, SSA
/// construction and rebuild, validation, and transformation passes.
///
/// Hosts that need to distinguish failure kinds can wrap this in their own
/// error enum via a `From<Error>` impl.
///
/// # Examples
///
/// ```rust
/// use analyssa::Error;
///
/// let err = Error::new("bad graph edge");
/// assert_eq!(err.to_string(), "bad graph edge");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct Error(pub String);

impl Error {
    /// Constructs a new error with the given message.
    ///
    /// # Arguments
    ///
    /// * `msg` - Anything that implements `Into<String>` (typically a `&str`
    ///   or `String`) describing the failure.
    ///
    /// # Returns
    ///
    /// A new `Error` instance wrapping the message.
    pub fn new<S: Into<String>>(msg: S) -> Self {
        Self(msg.into())
    }
}

/// Graph-operation error alias.
///
/// Type alias for [`struct@Error`] used in graph algorithm contexts (dominator
/// computation, cycle detection, topological sort, etc.) to distinguish
/// graph-specific failures from other error types while sharing the same
/// underlying representation.
pub type GraphError = Error;

/// Convenience alias for results from analyssa APIs.
///
/// Shorthand for `std::result::Result<T, Error>`. This is the standard
/// return type for fallible analyssa operations.
///
/// # Type Parameters
///
/// * `T` - The success type
/// * `E` - The error type (defaults to [`struct@Error`])
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_preserves_message_for_display_and_debug() {
        let err = Error::new("bad graph edge");

        assert_eq!(err, Error("bad graph edge".to_string()));
        assert_eq!(err.to_string(), "bad graph edge");
        assert!(format!("{err:?}").contains("bad graph edge"));
    }

    #[test]
    fn aliases_have_expected_shapes() {
        let graph_err: GraphError = Error::new("missing node");
        let result: Result<()> = Err(graph_err.clone());

        assert_eq!(result, Err(graph_err));
    }
}
