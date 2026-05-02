use crate::core::autodiff::{Node, add, mul};
use std::rc::Rc;

pub trait Layer {
    fn forward(&self, input: &[Rc<Node>]) -> Vec<Rc<Node>>;
    fn parameters(&self) -> Vec<Rc<Node>>;
}

pub struct Linear {
    pub weights: Vec<Vec<Rc<Node>>>, // [out_features][in_features]
    pub bias: Vec<Rc<Node>>,         // [out_features]
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut weights = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            let mut row = Vec::with_capacity(in_features);
            for _ in 0..in_features {
                // Initialize with small random-ish values or 0.1 for now
                row.push(Node::variable(0.1));
            }
            weights.push(row);
        }
        
        let mut bias = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            bias.push(Node::variable(0.0));
        }
        
        Self { weights, bias }
    }
}

impl Layer for Linear {
    fn forward(&self, input: &[Rc<Node>]) -> Vec<Rc<Node>> {
        let mut output = Vec::with_capacity(self.weights.len());
        
        for (i, weight_row) in self.weights.iter().enumerate() {
            let mut sum = Node::constant(0.0);
            for (j, w) in weight_row.iter().enumerate() {
                let product = mul(w, &input[j]);
                sum = add(&sum, &product);
            }
            sum = add(&sum, &self.bias[i]);
            output.push(sum);
        }
        
        output
    }

    fn parameters(&self) -> Vec<Rc<Node>> {
        let mut params = Vec::new();
        for row in &self.weights {
            for w in row {
                params.push(w.clone());
            }
        }
        for b in &self.bias {
            params.push(b.clone());
        }
        params
    }
}
