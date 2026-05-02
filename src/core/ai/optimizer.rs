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
            let grad = *param.grad.borrow();
            *val -= self.learning_rate * grad;
        }
    }

    pub fn zero_grad(&self, parameters: &[Rc<Node>]) {
        for param in parameters {
            *param.grad.borrow_mut() = 0.0;
        }
    }
}
