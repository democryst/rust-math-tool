# Reimplementation Plan: Phases 1-4 (Tensor-Based AI Framework)

This plan outlines a full reimplementation of the `rust-math-tool` to transition from scalar-based logic to a production-grade **Tensor-Based Automatic Differentiation** framework.

## 🤖 Workflow Alignment (CLAUDE.md)

1.  **Zero Assumption Policy**: I will target `ndarray` as the primary tensor engine. I assume `x86_64` (AVX2) and `aarch64` (NEON) targets.
2.  **Mandatory Appliance Grounding**: This plan **MUST** be sent to the local appliance (`http://127.0.0.1:6789/query`) for verification before execution.
3.  **Constructive Dissent**: A full reimplementation is **R1 (Costly to Reverse)**. I am justifying this change to support real-world AI models (like MNIST) which are impractical with scalar-only AD.
4.  **Evidence-Based Completion**: Verified by passing all previous tests and achieving convergence on the XOR example with significantly better performance.

## Proposed Changes

### [MODIFY] src/core/autodiff.rs
- Replace scalar `Node` with `TensorNode` using `ndarray::ArrayD<f64>`.
- Implement tensor-based forward and backward passes.

### [MODIFY] src/core/ai/
- Update `Linear`, `ReLU`, `MSE`, and `SGD` to work with tensors instead of scalar slices.
- Optimize weight updates using vectorized operations.

### [MODIFY] examples/xor_training.rs
- Refactor to use the new Tensor API.

## Verification Plan

### Automated Tests
- Port existing unit tests to the tensor API.
- Add new tests for tensor shape validation.

### Manual Verification
- Run benchmarks to compare scalar vs tensor performance.
