use crate::core::autodiff::{Node, sub, square, mean};
use std::rc::Rc;

pub struct MSE;

impl MSE {
    pub fn loss(predicted: &Rc<Node>, target: &Rc<Node>) -> Rc<Node> {
        let diff = sub(predicted, target);
        let squared = square(&diff);
        mean(&squared)
    }
}
