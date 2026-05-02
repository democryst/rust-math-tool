# Implementation Plan: Phase 3 - Performance Hardening

This phase focuses on optimizing the mathematical kernels using hardware acceleration (BLAS/LAPACK) and SIMD.

## 🤖 Workflow Alignment (CLAUDE.md)

1.  **Zero Assumption Policy**: I will target `x86_64` (AVX2) and `aarch64` (NEON). I will enable the `blas` feature in `ndarray` and integrate `netlib` or `openblas`.
2.  **Evidence-Based Completion**: Completion will be verified using the `criterion` benchmarking suite, comparing our kernels against baseline `ndarray` performance without BLAS.
3.  **Reversibility**: Integration of BLAS is R1 as it introduces external library dependencies (libblas/liblapack).

## Proposed Changes

### [MODIFY] Cargo.toml
- Enable `ndarray/blas` feature.
- Add `blas-src` or `openblas-src` for static linking.
- Add `criterion` to `[dev-dependencies]`.

### [NEW] benches/math_bench.rs
- Benchmarks for vector addition and matrix multiplication.

### [MODIFY] src/core/matrix.rs
- Ensure matrix multiplication is routed through BLAS where appropriate.

## Verification Plan

### Automated Benchmarks
- `cargo bench`: Measure throughput and latency of core kernels.
- Verify that BLAS-accelerated multiplication outperforms the pure Rust implementation for large matrices.

### Manual Verification
- Review benchmark reports to ensure < 10% overhead relative to standard BLAS implementations.
