extern crate openblas_src;

pub mod core {
    pub mod vector;
    pub mod matrix;
    pub mod autodiff;
    pub mod ai;
}

pub use crate::core::vector::{Vector, MathError};
pub use crate::core::matrix::Matrix;
pub use crate::core::autodiff::{Tensor, add, sub, mul, matmul, relu, square, mean};
pub use crate::core::ai::{Layer, Linear, ReLU, MSE, SGD};
