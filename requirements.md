# Software Requirements Specification (SRS): Mathematical Framework for AI in Rust (Extended)

## 1. Project Overview
This document outlines the technical requirements for a high-performance mathematical toolkit in Rust. It focuses on providing "Zero Assumption" performance, type-safe tensor operations, and a hybrid differentiation engine for AI and physics-based simulations.

## 2. Expanded Functional Requirements

### 2.1 Linear Algebra & Complex Systems
- **Core Tensors**: N-dimensional array support with efficient slicing, broadcasting, and lazy evaluation.
- **Decompositions**: LU, QR, SVD, and Cholesky (with support for sparse matrices).
- **Complex Analysis**: Support for complex-valued tensors (z=a+bi) and Wirtinger Calculus for non-holomorphic activation functions.
- **Hardware Acceleration**: Native bindings for BLAS/LAPACK and wgpu for cross-platform GPU acceleration.

### 2.2 Dual Differentiation Engine
- **Forward-Mode (Dual Numbers)**:
  - Implementation of the Dual Space a+bϵ where ϵ²=0.
  - Designed for high-precision Jacobians in low-parameter models.
- **Reverse-Mode (Tape-Based)**:
  - Wengert Tape implementation for building and traversing computational graphs.
  - Support for "checkpointing" to trade computation for memory in deep graphs.
- **Hybrid Switching**: Logic to automatically select differentiation mode based on the input/output ratio of the target function.

### 2.3 Information Geometry & Signal Analysis
- **Manifold Learning**: Support for Fisher Information Matrices (FIM) and Riemannian metrics for natural gradient descent.
- **Signal Processing**: High-performance Fast Fourier Transforms (FFT) and Wavelet transforms via SIMD-accelerated kernels.
- **Distance Metrics**: KL-Divergence, Wasserstein Distance (Optimal Transport), and Cosine Similarity.

### 2.4 Numerical Optimization
- **First-Order**: SGD, Adam, AdamW, and RMSprop with decoupled weight decay.
- **Second-Order**: L-BFGS and Conjugate Gradient methods for high-precision convergence.
- **Constraint Solvers**: Linear and Quadratic Programming (LP/QP) via interior-point methods.

## 3. Non-Functional Requirements

### 3.1 Performance & Safety
- **SIMD First**: Mandatory auto-vectorization for x86_64 (AVX-512) and aarch64 (NEON).
- **Memory Integrity**: No unsafe code allowed in higher-level abstractions; `unsafe` is strictly quarantined to hardware-level intrinsics.
- **Numerical Stability**: Implementation of Kahan Summation and Stochastic Rounding to mitigate floating-point drift.

### 3.2 Developer Experience
- **Compile-Time Verification**: Use of const generics to validate matrix dimensions, preventing runtime "Shape Mismatch" errors.
- **Error Handling**: All mathematical singularities (division by zero, non-invertible matrices) must return a `MathError: Result` type.

## 4. Technical Stack
| Component | Tool/Crate | Purpose |
| :--- | :--- | :--- |
| Linear Algebra | `ndarray` / `nalgebra` | Core matrix and vector logic. |
| Deep Learning | `burn` / `dfdx` | Computational graphs and autodiff. |
| Complex Math | `num-complex` | Complex number types. |
| FFT/Signals | `rustfft` | Signal processing kernels. |
| Optimization | `argmin` | Iterative solver framework. |
| GPU/SIMD | `wgpu` / `std::simd` | Hardware acceleration. |

## 5. Implementation Roadmap (Hardening Lifecycle)
- **Phase 1: Foundation (R2)**: Establish `ndarray` environment and complex-number support.
- **Phase 2: Hybrid Autodiff (R1)**: Implement Dual Numbers (Forward) and Tape-based (Reverse) engines.
  - **Gate**: Cross-verify gradients against finite difference methods (10⁻⁷ tolerance).
- **Phase 3: Manifolds & Signals (R1)**: Integrate FFT, KL-Divergence, and FIM calculations.
- **Phase 4: Optimization & Constraints (R2)**: Build Adam/L-BFGS optimizers and LP/QP solvers.
- **Phase 5: Integrity Gate (R0)**: Formal audit of unsafe blocks and final production hardening.

## 6. Verification & Integrity
- **Property-Based Testing**: Use `proptest` to verify mathematical identities (e.g., A⋅A⁻¹=I).
- **Integrity Gate**: Mathematical claims must be verified against logical constraints before committing to the main branch.
- **Fuzzing**: Continuous input fuzzing to detect edge-case overflows or NaN generation.

---
> "The Rust logic gate is the law; the LLM is merely the worker." - CLAUDE.md
