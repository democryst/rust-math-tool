use crate::core::autodiff::{Tensor, add, sub};
use ndarray::Array1;

#[derive(Debug, PartialEq)]
pub enum MathError {
    DimensionMismatch { expected: usize, found: usize },
}

pub type Result<T> = std::result::Result<T, MathError>;

/// Vector operations wrapper around Tensor
pub struct Vector {
    pub tensor: Tensor,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        let array = Array1::from_vec(data).into_dyn();
        Self {
            tensor: Tensor::variable(array),
        }
    }

    pub fn dim(&self) -> usize {
        self.tensor.0.value.borrow().len()
    }

    pub fn add(&self, other: &Vector) -> Result<Vector> {
        if self.dim() != other.dim() {
            return Err(MathError::DimensionMismatch {
                expected: self.dim(),
                found: other.dim(),
            });
        }
        Ok(Vector {
            tensor: add(&self.tensor, &other.tensor),
        })
    }

    pub fn sub(&self, other: &Vector) -> Result<Vector> {
        if self.dim() != other.dim() {
            return Err(MathError::DimensionMismatch {
                expected: self.dim(),
                found: other.dim(),
            });
        }
        Ok(Vector {
            tensor: sub(&self.tensor, &other.tensor),
        })
    }

    pub fn backward(&self) {
        self.tensor.backward();
    }
}
