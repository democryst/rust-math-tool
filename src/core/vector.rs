use crate::core::autodiff::{Node, add, sub};
use std::rc::Rc;
use ndarray::Array1;

#[derive(Debug, PartialEq)]
pub enum MathError {
    DimensionMismatch { expected: usize, found: usize },
}

pub type Result<T> = std::result::Result<T, MathError>;

/// Vector operations wrapper around Rc<Node>
pub struct Vector {
    pub node: Rc<Node>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        let array = Array1::from_vec(data).into_dyn();
        Self {
            node: Node::variable(array),
        }
    }

    pub fn dim(&self) -> usize {
        self.node.value.borrow().len()
    }

    pub fn add(&self, other: &Vector) -> Result<Vector> {
        if self.dim() != other.dim() {
            return Err(MathError::DimensionMismatch {
                expected: self.dim(),
                found: other.dim(),
            });
        }
        Ok(Vector {
            node: add(&self.node, &other.node),
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
            node: sub(&self.node, &other.node),
        })
    }

    pub fn backward(&self) {
        self.node.backward();
    }
}

