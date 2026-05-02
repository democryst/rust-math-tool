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
