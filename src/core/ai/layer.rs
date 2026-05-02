use crate::core::autodiff::{Node, matmul, add};
use std::rc::Rc;
use ndarray::ArrayD;

pub trait Layer {
    fn forward(&self, input: Rc<Node>) -> Rc<Node>;
    fn parameters(&self) -> Vec<Rc<Node>>;
}

pub struct Linear {
    pub weights: Rc<Node>,
    pub bias: Rc<Node>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Xavier/Glorot initialization would be better, but using 0.1 for consistency
        let w_val = ndarray::Array2::from_elem((in_features, out_features), 0.1).into_dyn();
        let b_val = ndarray::Array2::from_elem((1, out_features), 0.0).into_dyn();
        
        Self {
            weights: Node::variable(w_val),
            bias: Node::variable(b_val),
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: Rc<Node>) -> Rc<Node> {
        let mm = matmul(&input, &self.weights);
        add(&mm, &self.bias)
    }

    fn parameters(&self) -> Vec<Rc<Node>> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}
