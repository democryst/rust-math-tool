# Implementation Plan: Phase 2 - Differentiation Engine

This phase implements Automatic Differentiation (AD) to support gradient calculations for AI models.

## 🤖 Workflow Alignment (CLAUDE.md)

1.  **Zero Assumption Policy**: I will implement forward-mode AD using Dual Numbers. For reverse-mode AD, I will implement a simplified computational graph capable of handling basic scalar operations first.
2.  **Evidence-Based Completion**: Gradients will be verified against numerical finite differences using `proptest`.
3.  **Reversibility**: The AD engine architecture is R1 (Costly to Reverse) as it dictates how neural network layers will be built.

## Proposed Changes

### [NEW] src/core/autodiff.rs
- `Dual` struct for forward-mode AD.
- `Node` and `Graph` structs for reverse-mode AD.

### [MODIFY] src/lib.rs
- Export the `autodiff` module.

## Verification Plan

### Automated Tests
- `cargo test`: Unit tests for Dual number arithmetic.
- Property-based tests: Verify that $f'(x)$ calculated via AD matches $(f(x+h) - f(x))/h$ within a small epsilon.

### Manual Verification
- Review terminal output for gradient verification evidence.
