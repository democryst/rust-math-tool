# Refactor Plan: Autodiff Engine Optimization

This plan refactors the `autodiff` engine to improve stability, performance, and developer experience.

## 🤖 Workflow Alignment (CLAUDE.md)

1.  **Zero Assumption Policy**: I am assuming the graph is a Directed Acyclic Graph (DAG).
2.  **Mandatory Appliance Grounding**: This plan **MUST** be sent to the local appliance (`http://127.0.0.1:6789/query`) for verification before execution.
3.  **Constructive Dissent**: Replacing recursive propagation with topological sort is **R2 (Easily Reversed)** but significantly improves stability (prevents stack overflow).
4.  **Evidence-Based Completion**: Verified by passing all existing integration tests.

## Proposed Changes

### [MODIFY] src/core/autodiff.rs
- Implement topological sort for non-recursive backpropagation.
- Implement operator overloading (`Add`, `Sub`, `Mul`) for `Rc<Node>`.
- Refactor `Op` logic to be more modular.

## Verification Plan

### Automated Tests
- Run `cargo test` and verify `tests/autodiff_tests.rs` passes.
- Run `examples/xor_training.rs` to verify training still works with the new API.
