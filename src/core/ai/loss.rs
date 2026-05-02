use crate::core::autodiff::{Tensor, sub, square, mean};

pub struct MSE;

impl MSE {
    pub fn loss(predicted: &Tensor, target: &Tensor) -> Tensor {
        let diff = sub(predicted, target);
        let squared = square(&diff);
        mean(&squared)
    }
}
