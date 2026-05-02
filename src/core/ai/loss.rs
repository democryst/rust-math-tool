use crate::core::autodiff::{Node, sub, mul, add};
use std::rc::Rc;

pub struct MSE;

impl MSE {
    pub fn loss(predicted: &[Rc<Node>], target: &[Rc<Node>]) -> Rc<Node> {
        let mut total_loss = Node::constant(0.0);
        let n = predicted.len() as f64;
        
        for i in 0..predicted.len() {
            let diff = sub(&predicted[i], &target[i]);
            let squared_diff = mul(&diff, &diff);
            total_loss = add(&total_loss, &squared_diff);
        }
        
        mul(&total_loss, &Node::constant(1.0 / n))
    }
}
