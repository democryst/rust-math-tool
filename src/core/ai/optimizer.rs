use crate::core::autodiff::Tensor;

pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    pub fn step(&self, parameters: &[Tensor]) {
        for param in parameters {
            let mut val = param.0.value.borrow_mut();
            let grad = param.0.grad.borrow();
            *val -= &(&*grad * self.learning_rate);
        }
    }

    pub fn zero_grad(&self, parameters: &[Tensor]) {
        for param in parameters {
            let shape = param.0.grad.borrow().shape().to_vec();
            *param.0.grad.borrow_mut() = ndarray::ArrayD::zeros(ndarray::IxDyn(&shape));
        }
    }
}
