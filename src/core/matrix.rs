use crate::core::autodiff::{Node, matmul, add, sub};
use std::rc::Rc;
use ndarray::Array2;
use crate::core::vector::{MathError, Result};

/// Matrix operations wrapper around Rc<Node>
pub struct Matrix {
    pub node: Rc<Node>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        let array = Array2::from_shape_vec((rows, cols), data)
            .map_err(|_| MathError::DimensionMismatch { expected: rows * cols, found: 0 })? // Simplified error
            .into_dyn();
        
        Ok(Self {
            node: Node::variable(array),
        } )
    }

    pub fn shape(&self) -> (usize, usize) {
        let shape = self.node.value.borrow().shape().to_vec();
        (shape[0], shape[1])
    }

    pub fn multiply(&self, other: &Matrix) -> Result<Matrix> {
        let (_, c1) = self.shape();
        let (r2, _) = other.shape();
        
        if c1 != r2 {
            return Err(MathError::DimensionMismatch {
                expected: c1,
                found: r2,
            });
        }
        
        Ok(Matrix {
            node: matmul(&self.node, &other.node),
        })
    }

    pub fn add(&self, other: &Matrix) -> Result<Matrix> {
        Ok(Matrix {
            node: add(&self.node, &other.node),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication() {
        let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = m1.multiply(&m2).unwrap();
        
        let expected = Array2::from_shape_vec((2, 2), vec![19.0, 22.0, 43.0, 50.0]).unwrap().into_dyn();
        assert_eq!(*result.node.value.borrow(), expected);
    }
}
