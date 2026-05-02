use rust_math_tool::{Vector, Matrix};
use ndarray::{Array1, Array2};

#[test]
fn test_vector_addition() {
    let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
    let result = v1.add(&v2).unwrap();
    assert_eq!(*result.tensor.0.value.borrow(), Array1::from_vec(vec![5.0, 7.0, 9.0]).into_dyn());
}

#[test]
fn test_matrix_multiplication() {
    let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let result = m1.multiply(&m2).unwrap();
    
    let expected = Array2::from_shape_vec((2, 2), vec![19.0, 22.0, 43.0, 50.0]).unwrap().into_dyn();
    assert_eq!(*result.tensor.0.value.borrow(), expected);
}

use proptest::prelude::*;

proptest! {
    #[test]
    fn test_vector_addition_is_commutative(a in any::<Vec<f64>>(), b in any::<Vec<f64>>()) {
        let len = a.len().min(b.len());
        if len == 0 { return Ok(()); }
        let a_trunc = a[..len].to_vec();
        let b_trunc = b[..len].to_vec();
        
        let v1 = Vector::new(a_trunc.clone());
        let v2 = Vector::new(b_trunc.clone());
        
        let res1 = v1.add(&v2).unwrap();
        
        let v3 = Vector::new(b_trunc);
        let v4 = Vector::new(a_trunc);
        let res2 = v3.add(&v4).unwrap();
        
        assert_eq!(*res1.tensor.0.value.borrow(), *res2.tensor.0.value.borrow());
    }

    #[test]
    fn test_matrix_addition_is_commutative(a in any::<Vec<f64>>(), b in any::<Vec<f64>>()) {
        let len = a.len().min(b.len());
        if len == 0 { return Ok(()); }
        let rows = 2;
        let cols = len / 2;
        if cols == 0 { return Ok(()); }
        let data_len = rows * cols;
        
        let a_data = a[..data_len].to_vec();
        let b_data = b[..data_len].to_vec();
        
        let m1 = Matrix::new(rows, cols, a_data.clone()).unwrap();
        let m2 = Matrix::new(rows, cols, b_data.clone()).unwrap();
        let res1 = m1.add(&m2).unwrap();
        
        let m3 = Matrix::new(rows, cols, b_data).unwrap();
        let m4 = Matrix::new(rows, cols, a_data).unwrap();
        let res2 = m3.add(&m4).unwrap();
        
        assert_eq!(*res1.tensor.0.value.borrow(), *res2.tensor.0.value.borrow());
    }
}

#[test]
fn test_dimension_mismatch_errors() {
    let v1 = Vector::new(vec![1.0, 2.0]);
    let v2 = Vector::new(vec![1.0, 2.0, 3.0]);
    assert!(v1.add(&v2).is_err());
    assert!(v1.sub(&v2).is_err());

    let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m2 = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
    assert!(m1.multiply(&m2).is_err());
}

use rust_math_tool::ComplexTensor;
use num_complex::Complex64;
use ndarray::array;

#[test]
fn test_complex_tensor_addition() {
    let c1 = ComplexTensor::variable(array![Complex64::new(1.0, 2.0)].into_dyn());
    let c2 = ComplexTensor::variable(array![Complex64::new(3.0, 4.0)].into_dyn());
    
    let result = &c1 + &c2;
    assert_eq!(*result.0.value.borrow(), array![Complex64::new(4.0, 6.0)].into_dyn());
}

#[test]
fn test_complex_tensor_multiplication() {
    // (1 + 2i) * (3 + 4i) = 3 + 4i + 6i - 8 = -5 + 10i
    let c1 = ComplexTensor::variable(array![Complex64::new(1.0, 2.0)].into_dyn());
    let c2 = ComplexTensor::variable(array![Complex64::new(3.0, 4.0)].into_dyn());
    
    let result = &c1 * &c2;
    assert_eq!(*result.0.value.borrow(), array![Complex64::new(-5.0, 10.0)].into_dyn());
}

proptest! {
    #[test]
    fn test_complex_addition_is_commutative(a_re in any::<f64>(), a_im in any::<f64>(), b_re in any::<f64>(), b_im in any::<f64>()) {
        let c1 = ComplexTensor::variable(array![Complex64::new(a_re, a_im)].into_dyn());
        let c2 = ComplexTensor::variable(array![Complex64::new(b_re, b_im)].into_dyn());
        
        let res1 = &c1 + &c2;
        let res2 = &c2 + &c1;
        
        assert_eq!(*res1.0.value.borrow(), *res2.0.value.borrow());
    }
}

use rust_math_tool::{fft, ifft, kl_divergence, cosine_similarity, wasserstein_distance_1d, empirical_fisher_information_matrix};

#[test]
fn test_fft_ifft_inverse() {
    let input = ndarray::array![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    
    let f = fft(&input).unwrap();
    let inv = ifft(&f).unwrap();
    
    // Check if IFFT(FFT(x)) == x
    for (a, b) in input.iter().zip(inv.iter()) {
        assert!((a.re - b.re).abs() < 1e-10);
        assert!((a.im - b.im).abs() < 1e-10);
    }
}

#[test]
fn test_kl_divergence() {
    let p = ndarray::array![0.3, 0.7];
    let q = ndarray::array![0.3, 0.7];
    let kl = kl_divergence(&p, &q).unwrap();
    assert!((kl - 0.0).abs() < 1e-10); // D_KL(P||P) = 0
}

#[test]
fn test_cosine_similarity() {
    let a = ndarray::array![1.0, 0.0];
    let b = ndarray::array![0.0, 1.0];
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!((sim - 0.0).abs() < 1e-10); // Orthogonal vectors = 0
    
    let c = ndarray::array![1.0, 0.0];
    let sim2 = cosine_similarity(&a, &c).unwrap();
    assert!((sim2 - 1.0).abs() < 1e-10); // Parallel vectors = 1
}

#[test]
fn test_empirical_fisher_information() {
    // 2 samples, 3 parameters
    // Gradients for sample 1: [1.0, 0.0, -1.0]
    // Gradients for sample 2: [0.0, 1.0, 2.0]
    let scores = Matrix::new(2, 3, vec![
        1.0, 0.0, -1.0,
        0.0, 1.0, 2.0
    ]).unwrap();
    
    let fim = empirical_fisher_information_matrix(&scores).unwrap();
    
    // FIM should be 3x3 matrix
    let (r, c) = fim.shape();
    assert_eq!(r, 3);
    assert_eq!(c, 3);
    
    // Check diagonal elements (1/N * sum(g_i^2))
    // param 0: (1^2 + 0^2)/2 = 0.5
    // param 1: (0^2 + 1^2)/2 = 0.5
    // param 2: ((-1)^2 + 2^2)/2 = 2.5
    let data = fim.tensor.0.value.borrow().clone().into_raw_vec_and_offset().0;
    assert!((data[0] - 0.5).abs() < 1e-10);
    assert!((data[4] - 0.5).abs() < 1e-10);
    assert!((data[8] - 2.5).abs() < 1e-10);
}
