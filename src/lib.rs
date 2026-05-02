pub mod core {
    pub mod vector;
    pub mod matrix;
}

pub use crate::core::vector::{Vector, MathError};
pub use crate::core::matrix::Matrix;
