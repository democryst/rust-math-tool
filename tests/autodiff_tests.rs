use rust_math_tool::Tensor;
use ndarray::array;

#[test]
fn test_autodiff_basic() {
    let x = Tensor::variable(array![2.0].into_dyn());
    let y = Tensor::variable(array![3.0].into_dyn());
    
    // f = x * y + x
    let f = &(&x * &y) + &x;
    
    f.backward();
    
    assert_eq!(*f.0.value.borrow(), array![8.0].into_dyn());
    assert_eq!(*x.0.grad.borrow(), array![4.0].into_dyn()); // df/dx = y + 1 = 4
    assert_eq!(*y.0.grad.borrow(), array![2.0].into_dyn()); // df/dy = x = 2
}

use rust_math_tool::{Dual, DualTensor, HybridEngine};

#[test]
fn test_dual_number_arithmetic() {
    let a = Dual::new(2.0, 1.0); // f(x) = x, at x=2, f'(x)=1
    let b = Dual::new(3.0, 0.0); // constant 3
    
    let sum = a + b;
    assert_eq!(sum.real, 5.0);
    assert_eq!(sum.eps, 1.0);
    
    let prod = a * b;
    assert_eq!(prod.real, 6.0);
    assert_eq!(prod.eps, 3.0); // f(x) = 3x -> f'(x) = 3
}

#[test]
fn test_verification_gate_hybrid_autodiff() {
    // Function: f(x) = x * x + x
    // Derivative: f'(x) = 2x + 1
    // At x = 2.0, f(2) = 6.0, f'(2) = 5.0
    
    let input_val = 2.0;
    
    // 1. Forward Mode (Dual Numbers)
    let mut dual_x = DualTensor::from_f64(array![input_val].into_dyn());
    dual_x.set_gradient_seed(&[0]);
    let fwd_res = &(&dual_x * &dual_x) + &dual_x;
    let fwd_grad = fwd_res.get_eps()[0];
    
    // 2. Reverse Mode (Tensor)
    let rev_x = Tensor::variable(array![input_val].into_dyn());
    let rev_res = &(&rev_x * &rev_x) + &rev_x;
    rev_res.backward();
    let rev_grad = rev_x.0.grad.borrow()[0];
    
    // 3. Finite Difference
    let h = 1e-7;
    let f_x_plus_h = (input_val + h) * (input_val + h) + (input_val + h);
    let f_x_minus_h = (input_val - h) * (input_val - h) + (input_val - h);
    let fd_grad = (f_x_plus_h - f_x_minus_h) / (2.0 * h);
    
    // VERIFICATION GATE: Tolerance 10^-7
    let tolerance = 1e-7;
    assert!((fwd_grad - fd_grad).abs() < tolerance, "Forward-Mode failed Verification Gate");
    assert!((rev_grad - fd_grad).abs() < tolerance, "Reverse-Mode failed Verification Gate");
    
    // Ensure exact match between Forward and Reverse (analytically exact)
    assert_eq!(fwd_grad, rev_grad);
}

use rust_math_tool::{Adam, LBFGS, solve_lp, solve_qp};

#[test]
fn test_adam_converges_on_quadratic() {
    // Minimize f(x) = x^2.  Optimal at x = 0.
    // Start at x = 5.0; after enough steps, x should approach 0.
    let param = Tensor::variable(array![5.0].into_dyn());
    let mut adam = Adam::new(0.1);

    for _ in 0..500 {
        adam.zero_grad(&[param.clone()]);
        // f = x^2, grad = 2x
        let f = &param * &param;
        f.backward();
        adam.step(&[param.clone()]);
    }

    let x_final = param.0.value.borrow()[0];
    assert!(x_final.abs() < 0.01, "Adam failed to converge: x = {}", x_final);
}

#[test]
fn test_solve_lp_simple() {
    // Minimize: -x1 - x2  (i.e., maximize x1 + x2)
    // Subject to: x1 + x2 <= 1,  x1 >= 0,  x2 >= 0
    // Optimal: x1 = 0.5, x2 = 0.5 (or any split summing to 1)
    let c = vec![-1.0, -1.0];
    let a = vec![vec![1.0, 1.0]];
    let b = vec![1.0];

    let x = solve_lp(&c, &a, &b, 2000, 1e-6).unwrap();
    let sum = x[0] + x[1];
    assert!(sum > 0.8, "LP solution too small: sum = {}", sum);
    assert!(sum <= 1.01, "LP violated constraint: sum = {}", sum);
}

#[test]
fn test_solve_qp_simple() {
    // Minimize: 0.5 * (x1^2 + x2^2)  (Q = I, c = 0)
    // Subject to: x1 + x2 >= 1 -> -(x1 + x2) <= -1
    // Optimal: x1 = x2 = 0.5
    let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let c = vec![0.0, 0.0];
    let a = vec![vec![-1.0, -1.0]];
    let b = vec![-1.0];

    let x = solve_qp(&q, &c, &a, &b, 5000, 1e-6).unwrap();
    let sum = x[0] + x[1];
    assert!(sum >= 0.95, "QP constraint violated: sum = {}", sum);
    assert!((x[0] - x[1]).abs() < 0.1, "QP not symmetric: x = {:?}", x);
}
