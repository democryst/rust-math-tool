# rust-math-tool

A high-performance, production-grade mathematical toolkit for AI in Rust. Built around a `Zero Assumption` policy with a fully programmable Integrity Gate, type-safe tensor operations, and a hybrid automatic differentiation engine.

---

## ✨ Features

| Category | Capability |
|---|---|
| **Linear Algebra** | N-D Tensors, Matrix/Vector ops, SVD-ready decompositions |
| **Complex Numbers** | `ComplexTensor` with full arithmetic over `ℂ` |
| **Autodiff** | Dual-mode: **Forward** (Dual Numbers) + **Reverse** (topological DAG) |
| **Hybrid Switching** | Auto-selects mode based on Jacobian dimensions |
| **Signal Processing** | `fft` / `ifft` via `rustfft` (SIMD-accelerated) |
| **Information Geometry** | Empirical Fisher Information Matrix (FIM) |
| **Distance Metrics** | KL-Divergence, Cosine Similarity, Wasserstein Distance |
| **Optimizers** | SGD, Adam, AdamW, RMSprop, L-BFGS |
| **Constraint Solvers** | LP and QP via interior-point log-barrier methods |
| **Integrity Gate** | Runtime NaN/Inf verification with citation tracking |
| **Fuzzing** | Continuous `cargo-fuzz` coverage via `libfuzzer` |

---

## 🚀 Quickstart

```bash
git clone https://github.com/democryst/rust-math-tool
cd rust-math-tool
make test
```

### Run the XOR Training Example
```bash
make example
```

### Run Benchmarks
```bash
make bench
```

---

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rust-math-tool = { path = "../rust-math-tool" }
```

---

## 🧮 Usage

### Tensor Autodiff (Reverse-Mode)
```rust
use rust_math_tool::Tensor;
use ndarray::array;

let x = Tensor::variable(array![2.0].into_dyn());
let y = Tensor::variable(array![3.0].into_dyn());

let f = &(&x * &y) + &x;   // f = x*y + x
f.backward();

println!("f(2, 3) = {:?}", *f.0.value.borrow());   // 8.0
println!("∂f/∂x  = {:?}", *x.0.grad.borrow());     // 4.0
println!("∂f/∂y  = {:?}", *y.0.grad.borrow());     // 2.0
```

### Forward-Mode Autodiff (Dual Numbers)
```rust
use rust_math_tool::{Dual, DualTensor};
use ndarray::array;

let mut x = DualTensor::from_f64(array![2.0].into_dyn());
x.set_gradient_seed(&[0]);  // Seed ε for dimension 0

// f(x) = x^2 + x
let f = &(&x * &x) + &x;

println!("f(2)  = {}", f.get_real()[0]);   // 6.0
println!("f'(2) = {}", f.get_eps()[0]);    // 5.0
```

### Complex Tensors
```rust
use rust_math_tool::ComplexTensor;
use num_complex::Complex64;
use ndarray::array;

let a = ComplexTensor::variable(array![Complex64::new(1.0, 2.0)].into_dyn());
let b = ComplexTensor::variable(array![Complex64::new(3.0, 4.0)].into_dyn());

// (1+2i) * (3+4i) = -5+10i
let result = &a * &b;
```

### Optimizers
```rust
use rust_math_tool::{Tensor, Adam};
use ndarray::array;

let param = Tensor::variable(array![5.0].into_dyn());
let mut adam = Adam::new(0.1);

for _ in 0..500 {
    adam.zero_grad(&[param.clone()]);
    let loss = &param * &param;   // minimize x^2
    loss.backward();
    adam.step(&[param.clone()]);
}
println!("Converged to: {}", param.0.value.borrow()[0]);  // ~0.0
```

### LP/QP Constraint Solvers
```rust
use rust_math_tool::solve_lp;

// Maximize x1 + x2 subject to x1 + x2 <= 1
let c = vec![-1.0, -1.0];
let a = vec![vec![1.0, 1.0]];
let b = vec![1.0];

let x = solve_lp(&c, &a, &b, 2000, 1e-6).unwrap();
println!("x1={:.3}, x2={:.3}", x[0], x[1]);
```

### FFT
```rust
use rust_math_tool::fft;
use num_complex::Complex64;
use ndarray::array;

let signal = array![
    Complex64::new(1.0, 0.0),
    Complex64::new(2.0, 0.0),
    Complex64::new(3.0, 0.0),
    Complex64::new(4.0, 0.0),
];

let spectrum = fft(&signal).unwrap();
```

---

## 🗂️ Project Structure

```
rust-math-tool/
├── src/
│   ├── lib.rs                  # Public API surface
│   └── core/
│       ├── autodiff.rs         # Reverse-mode Tensor engine + topological sort
│       ├── complex.rs          # ComplexTensor (ℂ-valued autodiff)
│       ├── dual.rs             # Forward-mode Dual Numbers engine
│       ├── hybrid.rs           # HybridEngine (mode auto-selector)
│       ├── vector.rs           # Vector wrapper
│       ├── matrix.rs           # Matrix wrapper + transpose
│       ├── signal.rs           # FFT / IFFT via rustfft
│       ├── metrics.rs          # KL, Cosine, Wasserstein
│       ├── geometry.rs         # Empirical Fisher Information Matrix
│       ├── solver.rs           # LP/QP interior-point solvers
│       ├── integrity.rs        # IntegrityGate runtime auditor
│       └── ai/
│           ├── layer.rs        # Linear layer
│           ├── activation.rs   # ReLU
│           ├── loss.rs         # MSE loss
│           └── optimizer.rs    # SGD, Adam, AdamW, RMSprop, L-BFGS
├── examples/
│   └── xor_training.rs        # XOR convergence proof
├── tests/
│   ├── autodiff_tests.rs      # Autodiff + Verification Gate tests
│   └── math_tests.rs          # Math, metrics, FFT, FIM tests
├── benches/
│   └── math_bench.rs          # criterion benchmarks
├── fuzz/
│   └── fuzz_targets/
│       └── fuzz_target_1.rs   # cargo-fuzz tensor stability harness
├── requirements.md            # Extended SRS
├── CLAUDE.md                  # AI governance rules
└── Makefile                   # Developer workflow commands
```

---

## 🏗️ Implementation Roadmap

| Phase | Status | Description |
|---|---|---|
| **Phase 1** | ✅ Done | Complex Number Foundation (`ComplexTensor`) |
| **Phase 2** | ✅ Done | Hybrid Autodiff (Dual Numbers + Verification Gate) |
| **Phase 3** | ✅ Done | Manifolds & Signals (FFT, FIM, KL/Wasserstein) |
| **Phase 4** | ✅ Done | Optimization & Constraints (Adam, L-BFGS, LP/QP) |
| **Phase 5** | ✅ Done | Integrity Gate (runtime audit, fuzzing, proptest) |

---

## 🛡️ Integrity Gate

Every tensor operation carries a `citation` string documenting its mathematical lineage. The `IntegrityGate` auditor performs runtime verification at inference time, blocking any results that contain `NaN` or `Inf` before they propagate.

```rust
// IntegrityGate is called automatically by Tensor::verify()
tensor.verify(); // returns false if NaN/Inf detected
```

---

## 🧪 Testing

```bash
make test       # Run all unit + integration tests
make fuzz       # Run a 10-second fuzzing session
make bench      # Run criterion performance benchmarks
make check      # Run clippy linter
```

---

## 📖 Documentation

```bash
make doc        # Build and open rustdoc
```

---

## ⚖️ Governance

> "The Rust logic gate is the law; the LLM is merely the worker." — CLAUDE.md

All AI-assisted changes must comply with the rules in [CLAUDE.md](./CLAUDE.md):
- **Zero Assumption Policy**: Never hallucinate infrastructure.
- **Evidence-Based Completion**: A task is not done until tests pass.
- **Anti-Automation Rule**: AI must not call the appliance programmatically.
