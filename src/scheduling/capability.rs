//! Shared deobfuscation capability vocabulary used for pass ordering.
//!
//! Hosts that run the pass scheduler embed [`DeobfuscationCapability`] in
//! their concrete `T::Capability` enum. Generic analyssa passes declare
//! [`provides`](crate::scheduling::SsaPass::provides) and
//! [`requires`](crate::scheduling::SsaPass::requires) using these variants;
//! the host's `From<DeobfuscationCapability>` impl bridges into its own
//! capability enum so target-specific tags can sit alongside the shared
//! vocabulary.
//!
//! These milestones describe the *outcome* of a pass, not its mechanism —
//! "static fields have been resolved to concrete constants" applies just
//! as well to a CIL field-decryption pass as to a VB6 GoSub-table resolver.

/// Capability milestone produced or consumed by a deobfuscation pass.
///
/// The scheduler uses these to compute pass-execution order: if pass A
/// provides a capability and pass B requires it, B is scheduled in a
/// later layer than A.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeobfuscationCapability {
    /// Static field values have been resolved to concrete constants.
    ///
    /// Provided by passes that analyze static initializers or field
    /// decryptors. Required by passes that need constant values for
    /// further simplification.
    ResolvedStaticFields,
    /// Encrypted strings have been decrypted to their plaintext values.
    ///
    /// Provided by string decryption passes. Required by passes that
    /// analyze string content (e.g., proxy call resolution).
    DecryptedStrings,
    /// Control flow flattening has been reversed to normal structured
    /// control flow.
    ///
    /// Provided by CFG unflattening passes. Required by passes that
    /// assume natural loop structure.
    RestoredControlFlow,
    /// Opaque predicates have been simplified or removed.
    ///
    /// Provided by [`OpaquePredicatePass`](crate::passes::OpaquePredicatePass).
    /// Required by passes that need unobscured branch targets.
    SimplifiedPredicates,
    /// Proxy or virtual calls have been devirtualized to direct calls.
    ///
    /// Provided by devirtualization passes. Required by passes that
    /// need accurate call graphs (e.g., inlining, dead method detection).
    DevirtualizedCalls,
    /// Small or pure methods have been inlined at their call sites.
    ///
    /// Provided by inlining passes. Required by passes that benefit
    /// from a flattened call graph.
    InlinedMethods,
}
