use crate::core::autodiff::{Node, relu};
use std::rc::Rc;
use super::layer::Layer;

pub struct ReLU;

impl Layer for ReLU {
    fn forward(&self, input: Rc<Node>) -> Rc<Node> {
        relu(&input)
    }

    fn parameters(&self) -> Vec<Rc<Node>> {
        Vec::new()
    }
}
