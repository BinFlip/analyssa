# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
