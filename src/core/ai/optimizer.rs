use crate::core::autodiff::Node;
use std::rc::Rc;

pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    pub fn step(&self, parameters: &[Rc<Node>]) {
        for param in parameters {
            let mut val = param.value.borrow_mut();
            let grad = param.grad.borrow();
            *val -= &(&*grad * self.learning_rate);
        }
    }

    pub fn zero_grad(&self, parameters: &[Rc<Node>]) {
        for param in parameters {
            let shape = param.grad.borrow().shape().to_vec();
            *param.grad.borrow_mut() = ndarray::ArrayD::zeros(ndarray::IxDyn(&shape));
        }
    }
}
