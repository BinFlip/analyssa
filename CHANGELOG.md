# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-16

Native operation coverage for modern lifters — x87/FPU, SVE/SME, AMX tiles,
vector crypto, system/compute intrinsics, PAC — plus a domain-typed conversion
family replacing the single `Conv` op. Several effect-classification and
value-numbering correctness fixes ship alongside; see **Fixed**.

### Changed (breaking)

- **`SsaOp::Conv` is removed**, replaced by a domain-typed conversion family so
  each conversion carries only the fields that are meaningful for it:

  | Old | New | Notes |
  |-----|-----|-------|
  | `Conv` (int → int) | `IntConv` | Identical field set — mechanical rename. |
  | `Conv` (int → ptr) | `IntToPtr` | No `overflow_check`/`unsigned`. |
  | `Conv` (ptr → int) | `PtrToInt` | No `overflow_check`/`unsigned`. |
  | `Conv` (int → float) | `IntToFloat` | Keeps `overflow_check`/`unsigned`. |
  | `Conv` (float → int) | `FloatToInt` | Keeps `overflow_check`/`unsigned`. |
  | `Conv` (float → float) | `FloatConv` | Width change only; drops both flags. |

  `PtrAdd` is a new address-computation op, not a `Conv` replacement.
- **`SsaOp` gained 55 variants** (146 → 200), and `VectorUnaryKind` (+20),
  `VectorBinaryKind` (+23), `VectorCompareKind` (+8), `VectorTernaryKind` (+3),
  `AtomicRmwOp` (+3), and `SsaOpClass` (+`NativeIntrinsic`) also grew. No enum in
  this crate is `#[non_exhaustive]`, so downstream exhaustive matches must handle
  the new variants. This is deliberate: a lifter that silently ignores an
  unhandled op is a bug, so op additions are intended to break the build.
- **`ConstValue` equality is now structural, and the type is `Eq + Hash`.**
  Float arms compare and hash **bitwise**, which makes `Eq` reflexive and lets
  constants be hash-map keys. Two visible consequences: `F64(NAN) == F64(NAN)` is
  now `true`, and `F64(0.0) == F64(-0.0)` is now `false` (they are
  distinguishable at runtime via `1.0 / x`). IEEE-754 *semantic* comparison is
  unchanged and still lives in `ceq` and the `c*_un` family.
- **`SsaOp` and its operand payloads are now `Eq + Hash`.** No `Target` change is
  required — the trait already demanded `Eq + Hash` on its associated types.

### Fixed

- **`VectorStructLoadReplicate` (AArch64 `ld2r`/`ld3r`/`ld4r`) was classified
  `Pure`** despite reading memory through its address base. GVN could CSE two of
  them across an intervening store, and DCE could delete them outright. It now
  classifies as `Read` + `TrapClass::MemoryFault` and reports `may_throw()`, like
  every other vector load.
- **GVN could merge distinct `ComputeFlags` computations.** `ComputeFlags` models
  the flags of `bsf`/`bsr`/`popcnt`/`bt` alike but carries no opcode
  discriminator, so its result is not a function of its SSA operands; two
  different native flag computations over the same operands produced one key and
  the wrong flags value. It is no longer value-numbered.
- **GVN could merge `CallClobber` markers**, aliasing the fresh undefined values
  of two different calls' caller-saved registers (all of its operands are
  definitions, which the key normalizes to a sentinel). It is no longer
  value-numbered. `Phi` is likewise excluded — it is block-relative, and the key
  does not encode the defining block.
- **`SsaEditor::nop_instruction` left the removed value's `result_type` stamped
  on the `Nop`.** `set_op_preserving_type` now clears the type when the new op
  has no destination.
- **`SystemOpKind::Barrier` (`dsb`/`dmb`/`isb`/`mfence`) was classified `Write`.**
  It is an ordering construct, not a clobber: it now classifies as `Fence`, so
  Memory SSA emits a `MemoryOp::Barrier` and the verifier's fence invariant
  applies.
- **`setffr`/`wrffr` were deletable by DCE.** They write the SVE first-fault
  register, which is not an SSA operand, and their `outputs` may be empty — pure
  plus zero definitions meant DCE removed them, dropping the FFR initialization a
  following first-faulting load depends on. They now report `Opaque` effects.
- **Pointer conversions sign-extended when folded.** `IntToPtr`/`PtrToInt`
  hardcoded a signed source, so a 32-bit `0x8000_0000` would widen to
  `0xFFFF_FFFF_8000_0000` on a 64-bit target. Pointers are unsigned and now
  zero-extend. (Latent: no in-tree `Target` implements `convert_const`.)

- **LICM did not converge on functions with deep loop-invariant dependency
  chains.** The hoist phase moved only one dependency "wave" per invocation —
  leaving dependent invariants for a later call — so deeply-chained invariants in
  large loops needed O(chain-depth) expensive invocations and routinely exhausted
  the driving fixpoint's iteration cap with work still pending. It now hoists a
  loop's entire invariant chain in one pass, inserting in topological order so a
  hoisted definition always precedes its hoisted uses.
- **Loop canonicalization oscillated against CFG simplification.** `controlflow`
  (jump threading) and `blockmerge` (trampoline elimination) now preserve
  canonical loop preheaders and unified latches instead of removing them as empty
  forwarding blocks — which `loopcanon` immediately re-inserted, so the
  normalization fixpoint never settled. Loop-simplify form is now stable (the
  same trade-off LLVM's `simplifycfg` makes).

### Performance

- **GVN's generic value key is now structured rather than a formatted string.**
  It stores the operand-normalized `SsaOp` and probes via derived `Eq`/`Hash`,
  removing a per-candidate deep clone, a `Debug` render, and a `String`
  allocation, and turning every map probe from a string compare into a
  structural one.
- LICM `can_hoist` is O(1) per candidate instead of O(loop). The
  "result feeds a phi on a loop back-edge" test is precomputed once per loop as a
  single backward taint propagation seeded from the back-edge phi operands,
  replacing a fresh forward def-use traversal per candidate
  (O(candidates × loop) → O(loop)).
- LICM invariant detection and hoist-availability no longer scan every variable
  in the function once per loop; the loop-body-defined variable set is built once
  per loop in O(loop-body) (O(loops × variables) → O(loops × loop-body)).
- Loop canonicalization re-analyzes the loop forest once per pass instead of once
  per transformation: each pass now canonicalizes every loop the forest reports
  (still one transformation per loop per pass, preserving phi-management
  simplicity), turning O(transformations × loop-analysis) into
  O(passes × loop-analysis).
- GVN builds the CFG and dominator tree once per run rather than rebuilding it
  (and rescanning every block) for each redundant value pair.

Together these make the normalization pipeline converge in a couple of iterations
on control-flow-flattened inputs (many small nested loops with long invariant
chains), where it previously ran to the iteration cap without converging and
spent super-linear time per call. Output is unchanged.

### Added

- Native intrinsic modeling: `NativeIntrinsic`, `SystemOp`, `ComputeOp`,
  `VectorCrypto`, `TileOp`, `BcdAdjust`, x87/FPU (`FpTranscendental`,
  `FpuControl`), and pointer authentication (`PacKind`).
- SVE/SME and AMX operations: predicate/first-fault ops, SVE compute, SME
  outer-product and ZA-tile ops, matrix multiply-accumulate, tile operations.
- **`serde` feature** (off by default): serializes the IR data model — the
  operation-kind taxonomy, the native descriptors, and the SSA graph itself
  (`SsaFunction` and everything reachable from it: blocks, instructions, ops,
  phis, variables, constants, exception handlers).

  Generic IR types carry `#[serde(bound(...))]`, so `SsaFunction<T>` is
  serializable exactly when the host's `Target` associated types are. Hosts that
  don't serialize the IR are unaffected: the impl does not apply to them, and no
  serde bound is forced onto `Target`.

  Two encoding decisions are part of the wire contract:
  - `SsaVarId` serializes as its **logical index**, not its internal
    complement encoding (an `Option`-niche layout optimization that must not leak
    into a persisted format).
  - Maps keyed by `VariableOrigin` encode as `(key, value)` sequences. A derived
    map would compile but fail at runtime with "key must be a string" in every
    format that requires string keys, JSON included.

  Not covered: borrowed views (`SsaDefs<'a>`, `MemoryEffect<'a, T>`), transient
  pass machinery (builders, editors, edit reports), and `SsaFeatureToken` — its
  `&'static str` opcode can serialize but cannot deserialize, so a derive there
  would be a one-way trap.
- `num_enum::{IntoPrimitive, TryFromPrimitive}` on seven kind enums. **Note:**
  this pins those enums' numeric discriminants as public API — inserting a
  variant mid-enum silently remaps any value a host persisted by discriminant.
  The `serde` derives encode by variant *name* and are not affected.
- New public helpers: `SsaVarId::as_u32`, `SsaInstruction::set_op_preserving_type`,
  `SsaOp::{visit_operands, visit_operands_mut, replace_uses_with,
  arith_signedness, compare_kind, memory_effect}`, and a `fp_classify` builder.
- `ConstValue` scalar folds recurse lane-wise into `Vector` operands (previously
  returned `None`).
- `SsaEditor::replace_uses_checked_with` — replace instruction uses against a
  caller-supplied dominator tree, letting a pass that rewrites many variables
  (e.g. GVN) build the tree once and reuse it across the whole batch instead of
  per replacement.
- `BitSet::{contains_checked, insert_checked}` — bounds-tolerant accessors that
  treat an out-of-range index as unset rather than panicking, for callers whose
  index comes from data that may legitimately name a position outside the set
  (e.g. a terminator referencing a block that was never recovered).
- The lattice traits are re-exported from `analysis::dataflow`
  (`MeetSemiLattice`, `JoinSemiLattice`, `Lattice`), matching every other trait
  the module documents. They were previously reachable only through
  `analysis::dataflow::lattice::`, so the path the docs told you to use did not
  compile.

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
