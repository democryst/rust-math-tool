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

