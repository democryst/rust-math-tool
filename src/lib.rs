extern crate openblas_src;

pub mod core {
    pub mod vector;
    pub mod matrix;
    pub mod autodiff;
    pub mod ai;
    pub mod integrity;
    pub mod complex;
}

pub use crate::core::vector::{Vector, MathError};
pub use crate::core::matrix::Matrix;
pub use crate::core::autodiff::{Tensor, add, sub, mul, matmul, relu, square, mean};
pub use crate::core::complex::{ComplexTensor, ComplexOp, add_complex, sub_complex, mul_complex, matmul_complex};
pub use crate::core::ai::{Layer, Linear, ReLU, MSE, SGD};
