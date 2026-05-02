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
