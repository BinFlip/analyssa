//! Event logging for the SSA optimization pipeline.
//!
//! Passes record events through an [`EventListener<T>`]; hosts that don't care
//! pass [`NullListener`] to opt out without changing the call site. Hosts that
//! do care use [`EventLog<T>`], a thread-safe append-only collection that
//! itself impls `EventListener<T>`.
//!
//! # Architecture
//!
//! - [`EventKind`] — non-generic enum of event categories.
//! - [`Event<T>`] — a recorded event, parameterized over the host's
//!   `T::MethodRef` so each event can name the method it occurred in.
//! - [`EventListener<T>`] — sink trait; `push` accepts a fully-formed event.
//!   The default `record` method returns an [`EventBuilder`] for the fluent
//!   `events.record(kind).at(...).message(...)` API.
//! - [`NullListener`] — discards every event. Useful when running passes
//!   without observation (unit tests, CI, callers that don't need an event
//!   trace).
//! - [`EventLog<T>`] — concrete listener storing events for later inspection,
//!   summary, and filtering.
//!
//! # Example (analyssa-side, MockTarget)
//!
//! ```rust
//! use analyssa::{events::{EventKind, EventLog, EventListener}, MockTarget};
//!
//! let log: EventLog<MockTarget> = EventLog::new();
//! let method: u32 = 0x06000001;
//!
//! log.record(EventKind::ConstantFolded)
//!     .at(method, 0x42)
//!     .message("42 + 0 = 42");
//!
//! assert!(log.has(EventKind::ConstantFolded));
//! ```

use std::{
    collections::{HashMap, HashSet},
    fmt,
    time::Duration,
};

use crate::target::Target;

/// Categories of events that can be logged. Target-agnostic — labels are
/// purely descriptive and carry no host types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// A string was decrypted and inlined.
    StringDecrypted,
    /// A constant value was decrypted via emulation of a decryptor method.
    ConstantDecrypted,
    /// A constant value was folded/propagated.
    ConstantFolded,
    /// A conditional branch was simplified to unconditional.
    BranchSimplified,
    /// An instruction was removed.
    InstructionRemoved,
    /// A basic block was removed.
    BlockRemoved,
    /// A method call was inlined.
    MethodInlined,
    /// A phi node was simplified.
    PhiSimplified,
    /// An unknown value was resolved to a constant.
    ValueResolved,
    /// A method was marked as dead (unreachable).
    MethodMarkedDead,
    /// Control flow was restructured (e.g., unflattening).
    ControlFlowRestructured,
    /// An opaque predicate was identified and removed.
    OpaquePredicateRemoved,
    /// A copy operation was propagated away.
    CopyPropagated,
    /// An array was decrypted.
    ArrayDecrypted,
    /// An expensive operation was strength-reduced.
    StrengthReduced,
    /// Orphaned variables were removed from the variable table.
    VariablesCompacted,
    /// An encrypted method body was decrypted (anti-tamper).
    MethodBodyDecrypted,
    /// An encrypted resource was decrypted and re-injected (e.g. .NET Reactor
    /// Stage 7 resource encryption).
    ResourceDecrypted,
    /// Anti-tamper protection was removed.
    AntiTamperRemoved,
    /// An obfuscation artifact was removed (method, type, metadata).
    ArtifactRemoved,
    /// Code regeneration completed.
    CodeRegenerated,
}

impl EventKind {
    /// Returns a human-readable description of this event kind.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::StringDecrypted => "string decrypted",
            Self::ConstantDecrypted => "constant decrypted",
            Self::ConstantFolded => "constant folded",
            Self::BranchSimplified => "branch simplified",
            Self::InstructionRemoved => "instruction removed",
            Self::BlockRemoved => "block removed",
            Self::MethodInlined => "method inlined",
            Self::PhiSimplified => "phi simplified",
            Self::ValueResolved => "value resolved",
            Self::MethodMarkedDead => "method marked dead",
            Self::ControlFlowRestructured => "control flow restructured",
            Self::OpaquePredicateRemoved => "opaque predicate removed",
            Self::CopyPropagated => "copy propagated",
            Self::ArrayDecrypted => "array decrypted",
            Self::StrengthReduced => "strength reduced",
            Self::VariablesCompacted => "variables compacted",
            Self::MethodBodyDecrypted => "method body decrypted",
            Self::ResourceDecrypted => "resource decrypted",
            Self::AntiTamperRemoved => "anti-tamper removed",
            Self::ArtifactRemoved => "artifact removed",
            Self::CodeRegenerated => "code regenerated",
        }
    }

    /// Returns true if this event represents a code transformation.
    #[must_use]
    pub fn is_transformation(&self) -> bool {
        !matches!(self, Self::CodeRegenerated)
    }
}

impl fmt::Display for EventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.description())
    }
}

/// A single logged event.
#[derive(Debug, Clone)]
pub struct Event<T: Target> {
    /// The type of event.
    pub kind: EventKind,
    /// The method where the event occurred (if applicable).
    pub method: Option<T::MethodRef>,
    /// Location within the method (offset or block ID).
    pub location: Option<usize>,
    /// Human-readable description.
    pub message: String,
    /// Associated pass name (if from a pass).
    pub pass: Option<String>,
}

impl<T: Target> fmt::Display for Event<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.kind, self.message)
    }
}

/// Sink for events recorded by passes. The `push` method accepts a
/// fully-formed event; the default `record` method opens a fluent
/// [`EventBuilder`] that calls `push` on drop.
///
/// `Self: Sized` is required for `record` so trait objects (`&dyn
/// EventListener<T>`) can still receive events via `push` — they just can't
/// open a builder. Concrete listeners use the builder.
pub trait EventListener<T: Target> {
    /// Append an event to this listener.
    fn push(&self, event: Event<T>);

    /// Returns `true` if this listener actually records events.
    ///
    /// Defaults to `true`. Sinks that discard everything (e.g.
    /// [`NullListener`]) override this to `false` so the [`EventBuilder`] can
    /// skip constructing the event and its default message entirely — avoiding
    /// per-event allocation on the hot logging path when no one is observing.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Open a fluent builder for an event of `kind`. The event is appended
    /// when the builder is dropped, mirroring the legacy `EventLog::record`
    /// API.
    fn record(&self, kind: EventKind) -> EventBuilder<'_, T, Self>
    where
        Self: Sized,
    {
        EventBuilder::new(self, kind)
    }
}

/// No-op listener: every event is silently discarded.
///
/// Used when callers want to run a pass without observation — e.g. unit
/// tests that only check the resulting SSA, or production hosts that don't
/// surface event traces.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullListener;

impl<T: Target> EventListener<T> for NullListener {
    fn push(&self, _event: Event<T>) {}

    fn is_enabled(&self) -> bool {
        false
    }
}

/// Builder for creating events with a fluent API. Created via
/// [`EventListener::record`]. The event is automatically appended to the
/// owning listener when the builder is dropped.
pub struct EventBuilder<'a, T: Target, L: EventListener<T> + ?Sized> {
    listener: &'a L,
    kind: EventKind,
    method: Option<T::MethodRef>,
    location: Option<usize>,
    message: Option<String>,
    pass: Option<String>,
}

impl<'a, T: Target, L: EventListener<T> + ?Sized> EventBuilder<'a, T, L> {
    fn new(listener: &'a L, kind: EventKind) -> Self {
        Self {
            listener,
            kind,
            method: None,
            location: None,
            message: None,
            pass: None,
        }
    }

    /// Sets the method and location where the event occurred. Accepts
    /// anything convertible into `T::MethodRef` so hosts whose metadata uses
    /// a richer wrapper type can pass the underlying ID directly.
    pub fn at(mut self, method: impl Into<T::MethodRef>, location: usize) -> Self {
        self.method = Some(method.into());
        self.location = Some(location);
        self
    }

    /// Sets only the method (for method-level events without specific location).
    pub fn method(mut self, method: impl Into<T::MethodRef>) -> Self {
        self.method = Some(method.into());
        self
    }

    /// Sets the location (for when method is already set or not applicable).
    pub fn location(mut self, location: usize) -> Self {
        self.location = Some(location);
        self
    }

    /// Sets a custom message describing the event.
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Associates this event with a specific pass.
    pub fn pass(mut self, pass_name: impl Into<String>) -> Self {
        self.pass = Some(pass_name.into());
        self
    }
}

impl<T: Target, L: EventListener<T> + ?Sized> Drop for EventBuilder<'_, T, L> {
    fn drop(&mut self) {
        // Skip building the event (and its default-message allocation) entirely
        // when the sink discards everything.
        if !self.listener.is_enabled() {
            return;
        }

        let message = self
            .message
            .take()
            .unwrap_or_else(|| self.kind.description().to_string());

        let event = Event {
            kind: self.kind,
            method: self.method.take(),
            location: self.location.take(),
            message,
            pass: self.pass.take(),
        };

        self.listener.push(event);
    }
}

/// Concrete event sink storing events for later inspection, summary, and
/// filtering. Thread-safe append: events can be recorded concurrently from
/// multiple threads using shared references (`&self`) thanks to
/// [`boxcar::Vec`].
pub struct EventLog<T: Target> {
    events: boxcar::Vec<Event<T>>,
}

impl<T: Target> Default for EventLog<T> {
    fn default() -> Self {
        Self {
            events: boxcar::Vec::new(),
        }
    }
}

impl<T: Target> fmt::Debug for EventLog<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventLog")
            .field("len", &self.len())
            .finish()
    }
}

impl<T: Target> Clone for EventLog<T> {
    fn clone(&self) -> Self {
        let new_log = Self::new();
        for (_, event) in &self.events {
            new_log.events.push(event.clone());
        }
        new_log
    }
}

impl<T: Target> EventListener<T> for EventLog<T> {
    fn push(&self, event: Event<T>) {
        self.events.push(event);
    }
}

impl<T: Target> EventLog<T> {
    /// Creates an empty event log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a fluent builder for an event of `kind`. Mirrors
    /// [`EventListener::record`] as an inherent method so callers don't have
    /// to import the trait. The event is appended when the builder is
    /// dropped.
    pub fn record(&self, kind: EventKind) -> EventBuilder<'_, T, Self> {
        EventBuilder::new(self, kind)
    }

    /// Returns true if no events have been logged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.count() == 0
    }

    /// Returns the total number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.count()
    }

    /// Merges another event log into this one.
    pub fn merge(&self, other: &EventLog<T>) {
        for (_, event) in &other.events {
            self.events.push(event.clone());
        }
    }

    /// Returns true if any event of the given kind exists.
    #[must_use]
    pub fn has(&self, kind: EventKind) -> bool {
        self.events.iter().any(|(_, e)| e.kind == kind)
    }

    /// Returns true if any of the given event kinds exist.
    #[must_use]
    pub fn has_any(&self, kinds: &[EventKind]) -> bool {
        self.events.iter().any(|(_, e)| kinds.contains(&e.kind))
    }

    /// Counts events of the given kind.
    #[must_use]
    pub fn count_kind(&self, kind: EventKind) -> usize {
        self.events.iter().filter(|(_, e)| e.kind == kind).count()
    }

    /// Returns an iterator over all events.
    pub fn iter(&self) -> impl Iterator<Item = &Event<T>> {
        self.events.iter().map(|(_, e)| e)
    }

    /// Returns an iterator over events of a specific kind.
    pub fn filter_kind(&self, kind: EventKind) -> impl Iterator<Item = &Event<T>> + '_ {
        self.events
            .iter()
            .filter_map(move |(_, e)| if e.kind == kind { Some(e) } else { None })
    }

    /// Takes ownership of the events by cloning into a new EventLog.
    ///
    /// This is useful when the context is being consumed and you need to
    /// extract the events. Since `boxcar::Vec` is append-only and doesn't
    /// support draining, this creates a clone.
    #[must_use]
    pub fn take(&self) -> EventLog<T> {
        self.clone()
    }

    /// Consumes the log and returns its events, moving them out instead of
    /// cloning.
    ///
    /// Prefer this over [`take`](Self::take) when the log is owned and no
    /// longer needed: `take` must deep-clone every event (including its
    /// `String` message) because `boxcar::Vec` is append-only, whereas this
    /// drains the backing store by value.
    #[must_use]
    pub fn into_events(self) -> Vec<Event<T>> {
        self.events.into_iter().collect()
    }

    /// Returns an iterator over events for a specific method.
    pub fn filter_method<'a>(
        &'a self,
        method: &'a T::MethodRef,
    ) -> impl Iterator<Item = &'a Event<T>> + 'a {
        self.events.iter().filter_map(move |(_, e)| {
            if e.method.as_ref() == Some(method) {
                Some(e)
            } else {
                None
            }
        })
    }

    /// Returns an iterator over transformation events only.
    pub fn transformations(&self) -> impl Iterator<Item = &Event<T>> + '_ {
        self.events.iter().filter_map(|(_, e)| {
            if e.kind.is_transformation() {
                Some(e)
            } else {
                None
            }
        })
    }

    /// Counts events grouped by kind.
    #[must_use]
    pub fn count_by_kind(&self) -> HashMap<EventKind, usize> {
        let mut counts: HashMap<EventKind, usize> = HashMap::new();
        for (_, event) in &self.events {
            let entry = counts.entry(event.kind).or_insert(0);
            *entry = entry.saturating_add(1);
        }
        counts
    }

    /// Counts events grouped by kind, starting from the given offset.
    ///
    /// Used by the scheduler to compute per-pass event deltas without
    /// iterating the entire log.
    #[must_use]
    pub fn count_by_kind_since(&self, offset: usize) -> HashMap<EventKind, usize> {
        let mut counts: HashMap<EventKind, usize> = HashMap::new();
        for (idx, event) in &self.events {
            if idx >= offset {
                let entry = counts.entry(event.kind).or_insert(0);
                *entry = entry.saturating_add(1);
            }
        }
        counts
    }

    /// Returns the number of transformation events.
    #[must_use]
    pub fn transformation_count(&self) -> usize {
        self.events
            .iter()
            .filter(|(_, e)| e.kind.is_transformation())
            .count()
    }

    /// Returns the number of unique methods with events.
    #[must_use]
    pub fn methods_affected(&self) -> usize {
        self.events
            .iter()
            .filter_map(|(_, e)| e.method.as_ref())
            .collect::<HashSet<_>>()
            .len()
    }

    /// Generates a human-readable summary of all events.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_empty() {
            return "no events".to_string();
        }

        let counts = self.count_by_kind();

        let mut parts: Vec<String> = counts
            .iter()
            .filter(|(k, _)| k.is_transformation())
            .map(|(kind, count)| format!("{} {}", count, kind.description()))
            .collect();

        if parts.is_empty() {
            return format!("{} events", self.len());
        }

        parts.sort();
        parts.join(", ")
    }
}

/// Iterator wrapper for [`EventLog`] that yields `&Event<T>`.
pub struct EventLogIter<'a, T: Target> {
    inner: boxcar::Iter<'a, Event<T>>,
}

impl<'a, T: Target> Iterator for EventLogIter<'a, T> {
    type Item = &'a Event<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, e)| e)
    }
}

impl<'a, T: Target> IntoIterator for &'a EventLog<T> {
    type Item = &'a Event<T>;
    type IntoIter = EventLogIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        EventLogIter {
            inner: self.events.iter(),
        }
    }
}

impl<T: Target> Extend<Event<T>> for EventLog<T> {
    fn extend<I: IntoIterator<Item = Event<T>>>(&mut self, iter: I) {
        for event in iter {
            self.events.push(event);
        }
    }
}

impl<T: Target> FromIterator<Event<T>> for EventLog<T> {
    fn from_iter<I: IntoIterator<Item = Event<T>>>(iter: I) -> Self {
        let log = Self::new();
        for event in iter {
            log.events.push(event);
        }
        log
    }
}

/// Statistics derived from an [`EventLog`]. Counts are by [`EventKind`] and
/// independent of `T`, so this struct is not generic.
#[derive(Debug, Clone, Default)]
pub struct DerivedStats {
    /// Number of methods that had any transformations.
    pub methods_transformed: usize,
    /// Number of strings decrypted.
    pub strings_decrypted: usize,
    /// Number of arrays decrypted.
    pub arrays_decrypted: usize,
    /// Number of constants folded.
    pub constants_folded: usize,
    /// Number of constants decrypted.
    pub constants_decrypted: usize,
    /// Number of instructions removed.
    pub instructions_removed: usize,
    /// Number of blocks removed.
    pub blocks_removed: usize,
    /// Number of branches simplified.
    pub branches_simplified: usize,
    /// Number of opaque predicates removed.
    pub opaque_predicates_removed: usize,
    /// Number of methods inlined.
    pub methods_inlined: usize,
    /// Number of methods marked dead.
    pub methods_marked_dead: usize,
    /// Number of methods with code regenerated.
    pub methods_regenerated: usize,
    /// Number of artifacts removed (methods, types, metadata).
    pub artifacts_removed: usize,
    /// Number of pass iterations.
    pub iterations: usize,
    /// Processing time.
    pub total_time: Duration,
}

impl DerivedStats {
    /// Computes statistics from an event log.
    #[must_use]
    pub fn from_log<T: Target>(log: &EventLog<T>) -> Self {
        let counts = log.count_by_kind();
        let get = |kind: EventKind| counts.get(&kind).copied().unwrap_or(0);

        Self {
            methods_transformed: log.methods_affected(),
            strings_decrypted: get(EventKind::StringDecrypted),
            arrays_decrypted: get(EventKind::ArrayDecrypted),
            constants_folded: get(EventKind::ConstantFolded),
            constants_decrypted: get(EventKind::ConstantDecrypted),
            instructions_removed: get(EventKind::InstructionRemoved),
            blocks_removed: get(EventKind::BlockRemoved),
            branches_simplified: get(EventKind::BranchSimplified),
            opaque_predicates_removed: get(EventKind::OpaquePredicateRemoved),
            methods_inlined: get(EventKind::MethodInlined),
            methods_marked_dead: get(EventKind::MethodMarkedDead),
            methods_regenerated: get(EventKind::CodeRegenerated),
            artifacts_removed: get(EventKind::ArtifactRemoved),
            iterations: 0,
            total_time: Duration::ZERO,
        }
    }

    /// Sets the total processing time.
    #[must_use]
    pub fn with_time(mut self, time: Duration) -> Self {
        self.total_time = time;
        self
    }

    /// Sets the number of iterations.
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Generates a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if self.methods_transformed > 0 {
            parts.push(format!("{} methods", self.methods_transformed));
        }

        if self.strings_decrypted > 0 {
            parts.push(format!("{} strings decrypted", self.strings_decrypted));
        }
        if self.arrays_decrypted > 0 {
            parts.push(format!("{} arrays decrypted", self.arrays_decrypted));
        }
        if self.constants_decrypted > 0 {
            parts.push(format!("{} constants decrypted", self.constants_decrypted));
        }

        if self.constants_folded > 0 {
            parts.push(format!("{} constants folded", self.constants_folded));
        }
        if self.instructions_removed > 0 {
            parts.push(format!(
                "{} instructions removed",
                self.instructions_removed
            ));
        }
        if self.blocks_removed > 0 {
            parts.push(format!("{} blocks removed", self.blocks_removed));
        }
        if self.branches_simplified > 0 {
            parts.push(format!("{} branches simplified", self.branches_simplified));
        }
        if self.methods_inlined > 0 {
            parts.push(format!("{} inlined", self.methods_inlined));
        }
        if self.opaque_predicates_removed > 0 {
            parts.push(format!(
                "{} opaque predicates",
                self.opaque_predicates_removed
            ));
        }

        if self.methods_marked_dead > 0 {
            parts.push(format!("{} dead methods", self.methods_marked_dead));
        }
        if self.methods_regenerated > 0 {
            parts.push(format!("{} regenerated", self.methods_regenerated));
        }
        if self.artifacts_removed > 0 {
            parts.push(format!("{} artifacts removed", self.artifacts_removed));
        }

        let stats = if parts.is_empty() {
            "no transformations".to_string()
        } else {
            parts.join(", ")
        };

        if self.total_time.as_millis() > 0 {
            format!(
                "{} in {:?} ({} iterations)",
                stats, self.total_time, self.iterations
            )
        } else {
            stats
        }
    }
}

impl fmt::Display for DerivedStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

/// Truncates a string for display, adding ellipsis if needed.
#[must_use]
pub fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let end = max_len.saturating_sub(3);
        let split = s
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= end)
            .last()
            .unwrap_or(0);
        format!("{}...", &s[..split])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{sync::Arc, thread};

    use crate::testing::MockTarget;

    type Method = <MockTarget as Target>::MethodRef;

    fn method(id: u32) -> Method {
        id
    }

    #[test]
    fn empty_log() {
        let log: EventLog<MockTarget> = EventLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert!(!log.has(EventKind::StringDecrypted));
    }

    #[test]
    fn record_event() {
        let log: EventLog<MockTarget> = EventLog::new();
        let m = method(0x06000001);

        log.record(EventKind::StringDecrypted)
            .at(m, 0x10)
            .message("decrypted: \"hello\"");

        assert!(!log.is_empty());
        assert_eq!(log.len(), 1);
        assert!(log.has(EventKind::StringDecrypted));

        let event = log.iter().next().unwrap();
        assert_eq!(event.method, Some(m));
        assert_eq!(event.location, Some(0x10));
        assert_eq!(event.message, "decrypted: \"hello\"");
    }

    #[test]
    fn null_listener_discards() {
        let listener = NullListener;
        let m = method(0x06000001);

        EventListener::<MockTarget>::record(&listener, EventKind::StringDecrypted)
            .at(m, 0x10)
            .message("dropped on the floor");

        // No public observation surface — the test is that nothing panics
        // and the listener compiles.
    }

    #[test]
    fn multiple_events() {
        let log: EventLog<MockTarget> = EventLog::new();
        let m = method(0x06000001);

        log.record(EventKind::StringDecrypted)
            .at(m, 0x10)
            .message("first");
        log.record(EventKind::ConstantFolded)
            .at(m, 0x20)
            .message("second");

        assert_eq!(log.len(), 2);
        assert!(log.has(EventKind::StringDecrypted));
        assert!(log.has(EventKind::ConstantFolded));
        assert!(!log.has(EventKind::BlockRemoved));
    }

    #[test]
    fn has_any() {
        let log: EventLog<MockTarget> = EventLog::new();
        log.record(EventKind::StringDecrypted)
            .at(method(0x06000001), 0x10);

        assert!(log.has_any(&[EventKind::StringDecrypted, EventKind::ArrayDecrypted]));
        assert!(!log.has_any(&[EventKind::BlockRemoved, EventKind::MethodInlined]));
    }

    #[test]
    fn merge() {
        let log1: EventLog<MockTarget> = EventLog::new();
        let log2: EventLog<MockTarget> = EventLog::new();
        let m = method(0x06000001);

        log1.record(EventKind::StringDecrypted).at(m, 0x10);
        log2.record(EventKind::ConstantFolded).at(m, 0x20);

        log1.merge(&log2);

        assert_eq!(log1.len(), 2);
        assert!(log1.has(EventKind::StringDecrypted));
        assert!(log1.has(EventKind::ConstantFolded));
    }

    #[test]
    fn summary() {
        let log: EventLog<MockTarget> = EventLog::new();
        let m = method(0x06000001);

        log.record(EventKind::StringDecrypted).at(m, 0x10);
        log.record(EventKind::StringDecrypted).at(m, 0x20);
        log.record(EventKind::ConstantFolded).at(m, 0x30);

        let summary = log.summary();
        assert!(summary.contains("2 string decrypted"));
        assert!(summary.contains("1 constant folded"));
    }

    #[test]
    fn count_by_kind() {
        let log: EventLog<MockTarget> = EventLog::new();
        let m = method(0x06000001);

        log.record(EventKind::StringDecrypted).at(m, 0x10);
        log.record(EventKind::StringDecrypted).at(m, 0x20);
        log.record(EventKind::ConstantFolded).at(m, 0x30);

        let counts = log.count_by_kind();
        assert_eq!(counts.get(&EventKind::StringDecrypted), Some(&2));
        assert_eq!(counts.get(&EventKind::ConstantFolded), Some(&1));
        assert_eq!(counts.get(&EventKind::BlockRemoved), None);
    }

    #[test]
    fn count_by_kind_since() {
        let log: EventLog<MockTarget> = EventLog::new();
        let m = method(0x06000001);

        log.record(EventKind::StringDecrypted).at(m, 0x10);
        log.record(EventKind::StringDecrypted).at(m, 0x20);

        let offset = log.len();

        log.record(EventKind::ConstantFolded).at(m, 0x30);
        log.record(EventKind::ConstantFolded).at(m, 0x40);
        log.record(EventKind::StringDecrypted).at(m, 0x50);

        let counts = log.count_by_kind_since(offset);
        assert_eq!(counts.get(&EventKind::ConstantFolded), Some(&2));
        assert_eq!(counts.get(&EventKind::StringDecrypted), Some(&1));
        assert_eq!(counts.get(&EventKind::BlockRemoved), None);

        let all = log.count_by_kind_since(0);
        assert_eq!(all.get(&EventKind::StringDecrypted), Some(&3));
        assert_eq!(all.get(&EventKind::ConstantFolded), Some(&2));
    }

    #[test]
    fn derived_stats() {
        let log: EventLog<MockTarget> = EventLog::new();
        let m1 = method(0x06000001);
        let m2 = method(0x06000002);

        log.record(EventKind::StringDecrypted).at(m1, 0x10);
        log.record(EventKind::StringDecrypted).at(m2, 0x20);
        log.record(EventKind::ConstantFolded).at(m1, 0x30);

        let stats = DerivedStats::from_log(&log);
        assert_eq!(stats.methods_transformed, 2);
        assert_eq!(stats.strings_decrypted, 2);
        assert_eq!(stats.constants_folded, 1);
    }

    #[test]
    fn filter_methods() {
        let log: EventLog<MockTarget> = EventLog::new();
        let m1 = method(0x06000001);
        let m2 = method(0x06000002);

        log.record(EventKind::StringDecrypted).at(m1, 0x10);
        log.record(EventKind::ConstantFolded).at(m2, 0x20);
        log.record(EventKind::BlockRemoved).at(m1, 0x30);

        let m1_events: Vec<_> = log.filter_method(&m1).collect();
        assert_eq!(m1_events.len(), 2);
    }

    #[test]
    fn transformations_filter() {
        let log: EventLog<MockTarget> = EventLog::new();
        let m = method(0x06000001);

        log.record(EventKind::StringDecrypted).at(m, 0x10);
        log.record(EventKind::BlockRemoved).at(m, 0x20);

        let transformations: Vec<_> = log.transformations().collect();
        assert_eq!(transformations.len(), 2);
    }

    #[test]
    fn event_with_pass() {
        let log: EventLog<MockTarget> = EventLog::new();
        let m = method(0x06000001);

        log.record(EventKind::ConstantFolded)
            .at(m, 0x10)
            .pass("ConstantFolding")
            .message("42 + 0 → 42");

        let event = log.iter().next().unwrap();
        assert_eq!(event.pass.as_deref(), Some("ConstantFolding"));
    }

    #[test]
    fn default_message() {
        let log: EventLog<MockTarget> = EventLog::new();
        let m = method(0x06000001);

        log.record(EventKind::StringDecrypted).at(m, 0x10);

        let event = log.iter().next().unwrap();
        assert_eq!(event.message, "string decrypted");
    }

    #[test]
    fn thread_safe_append() {
        let log: Arc<EventLog<MockTarget>> = Arc::new(EventLog::new());
        let mut handles = vec![];

        for i in 0..4u32 {
            let log_clone = Arc::clone(&log);
            handles.push(thread::spawn(move || {
                for j in 0..100u32 {
                    let m = method(
                        0x06000000u32
                            .saturating_add(i.saturating_mul(100))
                            .saturating_add(j),
                    );
                    log_clone
                        .record(EventKind::StringDecrypted)
                        .at(m, j as usize)
                        .message(format!("thread {} event {}", i, j));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(log.len(), 400);
    }

    #[test]
    fn into_events_moves_without_cloning() {
        let log: EventLog<MockTarget> = EventLog::new();
        log.record(EventKind::StringDecrypted)
            .at(method(0x0600_0001), 0)
            .message("a");
        log.record(EventKind::ConstantFolded)
            .at(method(0x0600_0002), 1)
            .message("b");
        let events = log.into_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, EventKind::StringDecrypted);
        assert_eq!(events[1].kind, EventKind::ConstantFolded);
    }

    #[test]
    fn null_listener_is_disabled() {
        let null = NullListener;
        assert!(!EventListener::<MockTarget>::is_enabled(&null));
        // Recording through a disabled listener must not panic and yields nothing.
        EventListener::<MockTarget>::record(&null, EventKind::StringDecrypted)
            .at(method(0x0600_0003), 0)
            .message("ignored");
    }

    #[test]
    fn truncate_string_short() {
        assert_eq!(truncate_string("hi", 10), "hi");
    }

    #[test]
    fn truncate_string_long() {
        let result = truncate_string("hello world", 8);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 8);
    }
}
