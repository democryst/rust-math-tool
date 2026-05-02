use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;
use std::ops::{Add, Sub, Mul};

#[derive(Clone, Debug)]
pub enum Op {
    Constant,
    Add(Rc<Node>, Rc<Node>),
    Sub(Rc<Node>, Rc<Node>),
    Mul(Rc<Node>, Rc<Node>), // Element-wise or MatMul? Let's implement MatMul separately
    MatMul(Rc<Node>, Rc<Node>),
    ReLU(Rc<Node>),
    Square(Rc<Node>),
    Mean(Rc<Node>),
}

#[derive(Debug)]
pub struct Node {
    pub value: RefCell<ArrayD<f64>>,
    pub grad: RefCell<ArrayD<f64>>,
    pub op: Op,
}

impl Node {
    pub fn new(value: ArrayD<f64>, op: Op) -> Rc<Self> {
        let shape = value.shape().to_vec();
        Rc::new(Self {
            value: RefCell::new(value),
            grad: RefCell::new(ArrayD::zeros(IxDyn(&shape))),
            op,
        })
    }

    pub fn variable(value: ArrayD<f64>) -> Rc<Self> {
        Self::new(value, Op::Constant)
    }

    pub fn constant(value: ArrayD<f64>) -> Rc<Self> {
        Self::new(value, Op::Constant)
    }

    pub fn backward(self: &Rc<Self>) {
        let shape = self.grad.borrow().shape().to_vec();
        *self.grad.borrow_mut() = ArrayD::ones(IxDyn(&shape));
        self.propagate();
    }

    fn propagate(self: &Rc<Self>) {
        let g = self.grad.borrow().clone();
        match &self.op {
            Op::Add(a, b) => {
                let mut ag = a.grad.borrow_mut();
                
                if ag.shape() != g.shape() {
                    let summed = g.sum_axis(ndarray::Axis(0));
                    let reshaped = summed.into_shape_with_order(ag.shape()).unwrap();
                    *ag += &reshaped.into_dyn();
                } else {
                    *ag += &g;
                }
                drop(ag);

                if !Rc::ptr_eq(a, b) {
                    let mut bg = b.grad.borrow_mut();
                    if bg.shape() != g.shape() {
                        let summed = g.sum_axis(ndarray::Axis(0));
                        let reshaped = summed.into_shape_with_order(bg.shape()).unwrap();
                        *bg += &reshaped.into_dyn();
                    } else {
                        *bg += &g;
                    }
                    drop(bg);
                } else {
                    // If a == b, gradient is doubled
                    let mut ag = a.grad.borrow_mut();
                    if ag.shape() != g.shape() {
                        let summed = g.sum_axis(ndarray::Axis(0));
                        let reshaped = summed.into_shape_with_order(ag.shape()).unwrap();
                        *ag += &reshaped.into_dyn();
                    } else {
                        *ag += &g;
                    }
                }
                
                a.propagate();
                if !Rc::ptr_eq(a, b) {
                    b.propagate();
                }
            }
            Op::Sub(a, b) => {
                *a.grad.borrow_mut() += &g;
                if !Rc::ptr_eq(a, b) {
                    *b.grad.borrow_mut() -= &g;
                } else {
                    // x - x = 0, grad is 0, so no update needed if pointers match
                }
                a.propagate();
                if !Rc::ptr_eq(a, b) {
                    b.propagate();
                }
            }
            Op::Mul(a, b) => {
                let (a_grad_update, b_grad_update) = {
                    let a_val = a.value.borrow();
                    let b_val = b.value.borrow();
                    (&g * &*b_val, &g * &*a_val)
                };
                *a.grad.borrow_mut() += &a_grad_update;
                if !Rc::ptr_eq(a, b) {
                    *b.grad.borrow_mut() += &b_grad_update;
                } else {
                    *a.grad.borrow_mut() += &b_grad_update;
                }
                a.propagate();
                if !Rc::ptr_eq(a, b) {
                    b.propagate();
                }
            }
            Op::MatMul(a, b) => {
                let (a_grad_update, b_grad_update) = {
                    let a_val = a.value.borrow();
                    let b_val = b.value.borrow();
                    let a2 = a_val.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                    let b2 = b_val.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                    let g2 = g.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                    
                    (g2.dot(&b2.t()).into_dyn(), a2.t().dot(&g2).into_dyn())
                };
                
                *a.grad.borrow_mut() += &a_grad_update;
                *b.grad.borrow_mut() += &b_grad_update;
                
                a.propagate();
                if !Rc::ptr_eq(a, b) {
                    b.propagate();
                }
            }
            Op::ReLU(a) => {
                {
                    let a_val = a.value.borrow();
                    let mut a_grad = a.grad.borrow_mut();
                    for (ag, (av, gv)) in a_grad.iter_mut().zip(a_val.iter().zip(g.iter())) {
                        if *av > 0.0 {
                            *ag += gv;
                        }
                    }
                }
                a.propagate();
            }
            Op::Square(a) => {
                {
                    let a_val = a.value.borrow();
                    *a.grad.borrow_mut() += &(&g * &*a_val * 2.0);
                }
                a.propagate();
            }
            Op::Mean(a) => {
                {
                    let n = a.value.borrow().len() as f64;
                    let g_val = *g.iter().next().unwrap();
                    let mut a_grad = a.grad.borrow_mut();
                    a_grad.mapv_inplace(|v| v + g_val / n);
                }
                a.propagate();
            }
            _ => {}
        }
    }
}

pub fn matmul(a: &Rc<Node>, b: &Rc<Node>) -> Rc<Node> {
    let a_val = a.value.borrow().clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let b_val = b.value.borrow().clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    Node::new(a_val.dot(&b_val).into_dyn(), Op::MatMul(a.clone(), b.clone()))
}

pub fn add(a: &Rc<Node>, b: &Rc<Node>) -> Rc<Node> {
    Node::new(&*a.value.borrow() + &*b.value.borrow(), Op::Add(a.clone(), b.clone()))
}

pub fn sub(a: &Rc<Node>, b: &Rc<Node>) -> Rc<Node> {
    Node::new(&*a.value.borrow() - &*b.value.borrow(), Op::Sub(a.clone(), b.clone()))
}

pub fn mul(a: &Rc<Node>, b: &Rc<Node>) -> Rc<Node> {
    Node::new(&*a.value.borrow() * &*b.value.borrow(), Op::Mul(a.clone(), b.clone()))
}

pub fn relu(a: &Rc<Node>) -> Rc<Node> {
    let val = a.value.borrow().mapv(|v| if v > 0.0 { v } else { 0.0 });
    Node::new(val, Op::ReLU(a.clone()))
}

pub fn square(a: &Rc<Node>) -> Rc<Node> {
    let val = a.value.borrow().mapv(|v| v * v);
    Node::new(val, Op::Square(a.clone()))
}

pub fn mean(a: &Rc<Node>) -> Rc<Node> {
    let val = a.value.borrow().mean().unwrap_or(0.0);
    Node::new(ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), val), Op::Mean(a.clone()))
}
