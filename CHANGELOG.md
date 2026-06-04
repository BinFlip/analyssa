# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-03

Major release: a target-agnostic **native SSA substrate** for modern lifters,
new construction/editing infrastructure, and a broad performance/memory pass.
A few IR enum shapes changed (see **Changed**).

### Added

- Native SSA substrate: target-independent pointer sizes & endianness;
  first-class SIMD/vector operations with target-independent lane/vector-type
  semantics; native atomics (exchange, lock-RMW, compare-exchange); native
  opaque operations with machine-state clobbers; multi-output SSA definitions;
  boolean ops and native condition helpers; native flag semantics; and implicit
  wide (low/high, quotient/remainder) arithmetic.
- Memory effect summaries, exception/interrupt support, and native operation
  classes plus target-generic feature tokens.
- Fluent SSA builder (`ir::function::builder`) and a checked, verifier-preserving
  SSA editor (`ir::function::editor`); all built-in passes migrated onto the
  checked mutation APIs.
- Recommended normalization-pipeline API, pass bisection/debug hooks, and
  structured verifier diagnostics.
- Expanded `Target` trait (vector descriptors, pointer sizes, endianness),
  `MockTarget`, and a much larger test suite (builder, scheduling, verifier,
  pipeline, and canonicalization coverage).
- Allocation-free helpers: `SsaOp::{uses_var, for_each_successor, has_successor}`,
  `SsaInstruction::{uses_var, for_each_variable}`, `SsaBlock::for_each_successor`,
  `SymbolicExpr::for_each_variable`, `SsaFunction::compute_predecessors`,
  `BitSet::is_full`, `EventLog::into_events`, `EventListener::is_enabled`.

### Changed (breaking)

- `SsaOp::NativeOpaque` is now a tuple variant wrapping a boxed payload
  (`NativeOpaque(Box<NativeOpaqueData>)`) instead of an inline struct variant.
- `ConstValue` heap-bearing arms are now boxed: `Vector(Box<[ConstValue<T>]>)`,
  `DecryptedString(Box<str>)`, `DecryptedArray(Box<DecryptedArrayData<T>>)`.
- `SsaOp` and `ConstValue` gained many new variants for the native substrate;
  exhaustive downstream matches must handle them.

### Performance & memory

- Shrunk core IR types ~3–4×: `SsaOp` 168→40 B, `SsaInstruction` 176→48 B,
  `Option<SsaVarId>` 16→4 B (niche-encoded `SsaVarId`), `ConstValue` 40→24 B,
  `PhiOperand` 16→8 B; guarded by a `size_of` regression test.
- Removed hot-loop allocations (`uses()`/`successors()` purge, word-skipping
  `BitSet` iterator, DFS/postorder scratch reuse, cached solver exit set).
- `DefUseIndex` uses dense `Vec`-indexed storage instead of `BTreeMap`.
- De-quadratized DCE, GVN removal, jump-threading safety check, LICM invariant
  detection, predicate branch evaluation, trivial-phi predecessor build, and
  block-merge redirection; dataflow solver uses an RPO-priority worklist.
- `DirectedGraph` stores neighbor ids inline, `IndexedGraph` dedups edges in
  O(1), and cycle detection is iterative (stack-safe on deep graphs).
- Scheduler dirty-set membership is O(1); logging allocates nothing under
  `NullListener`.

## [0.1.0] - 2026-05-09

Initial standalone release.

### Added

- Target-agnostic SSA IR for blocks, instructions, phi nodes, variables, values,
  exception handlers, and functions.
- SSA analyses for CFGs, constants, liveness, memory, phi placement, symbolic
  expressions, dataflow, def-use, loop structure, and verification.
- Optimization and deobfuscation passes including algebraic simplification,
  block merging, control-flow cleanup, copy propagation, dead-code elimination,
  GVN, LICM, loop canonicalization, reassociation, scheduling, strength
  reduction, and jump threading.
- Generic graph and bitset utilities used by the IR and analyses.
- Host adapter traits and a pass scheduler for integrating analyssa into target
  lifters without tying the crate to one instruction set or metadata model.
