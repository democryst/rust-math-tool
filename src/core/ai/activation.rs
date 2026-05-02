use crate::core::autodiff::{Tensor, relu};
use super::layer::Layer;

pub struct ReLU;

impl Layer for ReLU {
    fn forward(&self, input: Tensor) -> Tensor {
        relu(&input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}
