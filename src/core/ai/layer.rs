use crate::core::autodiff::{Tensor, matmul, add};

pub trait Layer {
    fn forward(&self, input: Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
}

pub struct Linear {
    pub weights: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let w_val = ndarray::Array2::from_elem((in_features, out_features), 0.1).into_dyn();
        let b_val = ndarray::Array2::from_elem((1, out_features), 0.0).into_dyn();
        
        Self {
            weights: Tensor::variable(w_val),
            bias: Tensor::variable(b_val),
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: Tensor) -> Tensor {
        let mm = matmul(&input, &self.weights);
        add(&mm, &self.bias)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}
