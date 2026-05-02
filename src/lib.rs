extern crate openblas_src;

pub mod core {
    pub mod vector;
    pub mod matrix;
    pub mod autodiff;
}

pub use crate::core::vector::{Vector, MathError};
pub use crate::core::matrix::Matrix;
pub use crate::core::autodiff::Dual;
