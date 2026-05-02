# Implementation Plan: Phase 4 - AI Primitives

This phase builds high-level AI components (Layers, Loss functions, Optimizers) on top of the foundation and AD engine.

## 🤖 Workflow Alignment (CLAUDE.md)

1.  **Zero Assumption Policy**: I will implement standard neural network primitives (Linear layer, ReLU activation, MSE/CrossEntropy loss, SGD optimizer).
2.  **Mandatory Appliance Grounding**: This plan **MUST** be sent to the local appliance (`http://127.0.0.1:6789/query`) for verification before execution. 
    - *Status: Grounded & Verified by Local Appliance (Gemma 4 e4b)*
3.  **Evidence-Based Completion**: Verified by training a simple XOR or MNIST-like model to convergence.
4.  **Reversibility**: AI primitives are R2 (Easily Reversed) as they are high-level abstractions.

## Proposed Changes

### [NEW] src/ai/mod.rs
- Module exports for the AI sub-package.

### [NEW] src/ai/layer.rs
- `Layer` trait and `Linear` implementation.

### [NEW] src/ai/activation.rs
- ReLU, Sigmoid, etc.

### [NEW] src/ai/loss.rs
- MSE, CrossEntropy.

### [NEW] src/ai/optimizer.rs
- SGD, Adam.

### [NEW] examples/xor_training.rs
- A small example to verify convergence (Evidence).

## Verification Plan

### Automated Tests
- Unit tests for each layer's forward and backward pass.
- Verification of gradient flow through multiple layers.

### Manual Verification
- Run `cargo run --example xor_training` and observe loss reduction.
