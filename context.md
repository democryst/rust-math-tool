# Project Context — rust-math-tool

**Last Updated**: 2026-05-02  
**Branch**: `main` (commit: `ddcf2f2`)  
**Toolchain**: Rust stable + Rust nightly (for fuzzing)

---

## 🎯 Project Goal

A production-grade, `Zero Assumption` mathematical toolkit for AI in Rust. Built around a hybrid automatic differentiation engine, type-safe tensor operations, and a programmable Integrity Gate runtime auditor.

---

## 📦 Dependencies

```toml
ndarray = { version = "0.17.2", features = ["blas"] }
openblas-src = { version = "0.10.15", features = ["static"] }
proptest = "1.11.0"
num-complex = "0.4.6"
rustfft = "6.2.0"

[dev-dependencies]
criterion = "0.8.2"
```

---

## 🏗️ Completed Implementation Roadmap

### Phase 1 — Complex Number Foundation ✅
- **`src/core/complex.rs`**: `ComplexTensor` wrapping `Rc<ComplexNode>` with `ArrayD<Complex64>`.
- Operator overloading: `Add`, `Sub`, `Mul` for owned and reference `ComplexTensor`.
- Backpropagation stub for `ComplexOp::{Add, Sub, Mul, MatMul}`.
- `ComplexTensor::verify()` routes through `IntegrityGate`.

### Phase 2 — Hybrid Autodiff ✅
- **`src/core/dual.rs`**: `Dual { real: f64, eps: f64 }` struct with full `ε²=0` arithmetic.
- `DualTensor`: `ArrayD<Dual>` with `set_gradient_seed()` for Forward-Mode.
- **`src/core/hybrid.rs`**: `HybridEngine::optimize_and_run()` — routes to Forward-Mode if `input_dim < output_dim`, else Reverse-Mode.
- **Verification Gate**: `test_verification_gate_hybrid_autodiff` cross-checks Forward vs Reverse vs Finite Difference at `10⁻⁷` tolerance.

### Phase 3 — Manifolds & Signals ✅
- **`src/core/signal.rs`**: `fft()` and `ifft()` via `rustfft` with correct 1/N normalization.
- **`src/core/metrics.rs`**: `kl_divergence`, `cosine_similarity`, `wasserstein_distance_1d`.
- **`src/core/geometry.rs`**: `empirical_fisher_information_matrix(S)` = `(1/N) * S^T * S`.
- **Bug Fixed**: `Matrix::add()` was missing dimension parity check — caught by `cargo-fuzz` and patched.

### Phase 4 — Optimization & Constraints ✅
- **`src/core/ai/optimizer.rs`**: Extended with:
  - `Adam` (bias-corrected adaptive moments, β1=0.9, β2=0.999)
  - `AdamW` (Adam + decoupled weight decay applied before update)
  - `RMSprop` (running mean-square normalization, α=0.99)
  - `LBFGS` (two-loop recursion with m-history (s,y) curvature pairs + H₀ scaling)
- **`src/core/solver.rs`**:
  - `solve_lp()` — log-barrier interior-point with decaying `μ` weight
  - `solve_qp()` — same method + quadratic Hessian `Qx` term
  - Key design: `μ *= 0.995` each iteration so barrier fades to reveal true optimum.

### Phase 5 — Integrity Gate ✅ (Pre-existing)
- **`src/core/integrity.rs`**: `IntegrityGate::verify_claim()` — runtime NaN/Inf audit.
- **`fuzz/fuzz_targets/fuzz_target_1.rs`**: `cargo-fuzz` harness for Tensor/Matrix stability.
- Fuzzer successfully caught `Matrix::add` dimension bug on first run.

---

## 🗂️ Full Module Map

```
src/
├── lib.rs                  # Public API, //! crate-level rustdoc
└── core/
    ├── autodiff.rs         # Tensor (Rc<Node>), non-recursive topo sort, backward()
    ├── complex.rs          # ComplexTensor (C-valued ops + backprop stub)
    ├── dual.rs             # Dual, DualTensor (Forward-Mode AD)
    ├── hybrid.rs           # HybridEngine (mode auto-selector)
    ├── vector.rs           # Vector, MathError { DimensionMismatch, Singularity }
    ├── matrix.rs           # Matrix, multiply, add, transpose
    ├── signal.rs           # fft, ifft (rustfft)
    ├── metrics.rs          # kl_divergence, cosine_similarity, wasserstein_distance_1d
    ├── geometry.rs         # empirical_fisher_information_matrix
    ├── solver.rs           # solve_lp, solve_qp (log-barrier, decaying μ)
    ├── integrity.rs        # IntegrityGate::verify_claim
    └── ai/
        ├── mod.rs          # Re-exports
        ├── layer.rs        # Linear (forward pass)
        ├── activation.rs   # ReLU
        ├── loss.rs         # MSE
        └── optimizer.rs    # SGD, Adam, AdamW, RMSprop, LBFGS
```

---

## 🧪 Test Suite (18 tests total, 100% pass)

### `tests/autodiff_tests.rs` (6 tests)
- `test_autodiff_basic` — basic reverse-mode graph
- `test_dual_number_arithmetic` — Dual arithmetic rules
- `test_verification_gate_hybrid_autodiff` — tri-method gradient cross-check
- `test_adam_converges_on_quadratic` — Adam on `f(x)=x²` from x=5 → ~0
- `test_solve_lp_simple` — LP: maximize x1+x2 s.t. x1+x2 ≤ 1
- `test_solve_qp_simple` — QP: minimize ‖x‖² s.t. x1+x2 ≥ 1

### `tests/math_tests.rs` (12 tests)
- `test_vector_addition`, `test_matrix_multiplication`
- `test_vector_addition_is_commutative` (proptest)
- `test_matrix_addition_is_commutative` (proptest)
- `test_dimension_mismatch_errors`
- `test_complex_tensor_addition`, `test_complex_tensor_multiplication`
- `test_complex_addition_is_commutative` (proptest)
- `test_fft_ifft_inverse` — IFFT(FFT(x)) == x at 1e-10
- `test_kl_divergence` — D_KL(P||P) = 0
- `test_cosine_similarity` — orthogonal=0, parallel=1
- `test_empirical_fisher_information` — diagonal validation

---

## 🛠️ Developer Workflow

```bash
make test        # cargo test
make bench       # cargo bench (criterion)
make fuzz        # cargo +nightly fuzz run fuzz_target_1 -- -max_total_time=10
make fuzz-long   # 60s fuzz session
make doc         # cargo doc --no-deps --open
make example     # cargo run --example xor_training --release
make gate        # check + test + fuzz (full CI pipeline)
make check       # clippy
```

---

## ⚠️ Known Warnings (non-blocking)

These are cosmetic warnings — all tests and builds pass:

| File | Warning |
|---|---|
| `src/core/integrity.rs` | `unused import: std::process::Command` |
| `src/core/dual.rs` | `unused import: IxDyn` |
| `src/core/hybrid.rs` | `non_camel_case_types` on `F_rev`, `F_fwd` type params |
| `src/core/integrity.rs` | `unused variable: context` |

Fix with: `cargo fix --lib -p rust-math-tool` (safe, auto-applies all)

---

## 🏛️ Architecture Decisions

| Decision | Rationale |
|---|---|
| `Tensor = Rc<Node>` newtype | Orphan rule workaround for `Add`/`Sub`/`Mul` traits |
| Non-recursive topo sort | Prevents stack overflow in deep DAGs |
| `ComplexTensor` separate from `Tensor` | Avoids breaking existing `f64` API with generics |
| Empirical FIM via `S^T S` | Exact analytical FIM is `O(P²)` — empirical proxy is standard in practice |
| LP/QP decaying μ barrier | Fixed μ biases solution to interior; μ decay reveals true optimum |
| Proptest for commutative proofs | Generates thousands of random inputs to catch edge cases |

---

## 📜 Governance (CLAUDE.md)

1. **Zero Assumption Policy**: Never hallucinate infrastructure.
2. **Anti-Automation Rule**: AI must not call the Gemma appliance programmatically.
3. **Evidence-Based Completion**: Task is not done until terminal output confirms it.
4. **Constructive Dissent**: Surface Blast Radius, Blind Spots, Reversibility before major changes.
5. **Scope Drift Detection**: Flag if task shifts from primary objective.
6. **Reversibility**: R0=irreversible (ask), R1=costly (justify), R2=easy (just do it).
