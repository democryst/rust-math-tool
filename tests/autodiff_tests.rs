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
