use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;
use std::ops::{Add, Sub, Mul};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub enum Op {
    Constant,
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
    MatMul(Tensor, Tensor),
    ReLU(Tensor),
    Square(Tensor),
    Mean(Tensor),
}

#[derive(Debug)]
pub struct Node {
    pub value: RefCell<ArrayD<f64>>,
    pub grad: RefCell<ArrayD<f64>>,
    pub op: Op,
    pub citation: String,
}

#[derive(Clone, Debug)]
pub struct Tensor(pub Rc<Node>);

impl Tensor {
    pub fn new(value: ArrayD<f64>, op: Op) -> Self {
        let shape = value.shape().to_vec();
        let citation = format!("{:?} operation on shape {:?}", op, shape);
        Self(Rc::new(Node {
            value: RefCell::new(value),
            grad: RefCell::new(ArrayD::zeros(IxDyn(&shape))),
            op,
            citation,
        }))
    }

    pub fn variable(value: ArrayD<f64>) -> Self {
        let shape = value.shape().to_vec();
        Self(Rc::new(Node {
            value: RefCell::new(value),
            grad: RefCell::new(ArrayD::zeros(IxDyn(&shape))),
            op: Op::Constant,
            citation: format!("Variable initialized with shape {:?}", shape),
        }))
    }

    pub fn constant(value: ArrayD<f64>) -> Self {
        let shape = value.shape().to_vec();
        Self(Rc::new(Node {
            value: RefCell::new(value),
            grad: RefCell::new(ArrayD::zeros(IxDyn(&shape))),
            op: Op::Constant,
            citation: format!("Constant initialized with shape {:?}", shape),
        }))
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        self.build_topo(&mut visited, &mut topo);

        let shape = self.0.grad.borrow().shape().to_vec();
        *self.0.grad.borrow_mut() = ArrayD::ones(IxDyn(&shape));

        for node in topo.iter().rev() {
            node.propagate_step();
        }
    }

    pub fn verify(&self) -> bool {
        use crate::core::integrity::IntegrityGate;
        let val = self.0.value.borrow();
        let claim = format!("Tensor result {:?} is correct", *val);
        IntegrityGate::verify_claim(&claim, &self.0.citation)
    }

    fn build_topo(&self, visited: &mut HashSet<*const Node>, topo: &mut Vec<Tensor>) {
        let ptr = Rc::as_ptr(&self.0);
        if !visited.contains(&ptr) {
            visited.insert(ptr);
            match &self.0.op {
                Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::MatMul(a, b) => {
                    a.build_topo(visited, topo);
                    b.build_topo(visited, topo);
                }
                Op::ReLU(a) | Op::Square(a) | Op::Mean(a) => {
                    a.build_topo(visited, topo);
                }
                Op::Constant => {}
            }
            topo.push(self.clone());
        }
    }

    fn propagate_step(&self) {
        let g = self.0.grad.borrow().clone();
        match &self.0.op {
            Op::Add(a, b) => {
                let mut ag = a.0.grad.borrow_mut();
                if ag.shape() != g.shape() {
                    let summed = g.sum_axis(ndarray::Axis(0));
                    let reshaped = summed.into_shape_with_order(ag.shape()).expect("Gradient shape mismatch during Add propagation");
                    *ag += &reshaped.into_dyn();
                } else {
                    *ag += &g;
                }
                drop(ag);

                if !Rc::ptr_eq(&a.0, &b.0) {
                    let mut bg = b.0.grad.borrow_mut();
                    if bg.shape() != g.shape() {
                        let summed = g.sum_axis(ndarray::Axis(0));
                        let reshaped = summed.into_shape_with_order(bg.shape()).expect("Gradient shape mismatch during broadcast Add propagation");
                        *bg += &reshaped.into_dyn();
                    } else {
                        *bg += &g;
                    }
                } else {
                    let mut ag = a.0.grad.borrow_mut();
                    if ag.shape() != g.shape() {
                        let summed = g.sum_axis(ndarray::Axis(0));
                        let reshaped = summed.into_shape_with_order(ag.shape()).expect("Gradient shape mismatch during identity Add propagation");
                        *ag += &reshaped.into_dyn();
                    } else {
                        *ag += &g;
                    }
                }
            }
            Op::Sub(a, b) => {
                *a.0.grad.borrow_mut() += &g;
                if !Rc::ptr_eq(&a.0, &b.0) {
                    *b.0.grad.borrow_mut() -= &g;
                }
            }
            Op::Mul(a, b) => {
                let (a_update, b_update) = {
                    let av = a.0.value.borrow();
                    let bv = b.0.value.borrow();
                    (&g * &*bv, &g * &*av)
                };
                *a.0.grad.borrow_mut() += &a_update;
                if !Rc::ptr_eq(&a.0, &b.0) {
                    *b.0.grad.borrow_mut() += &b_update;
                } else {
                    *a.0.grad.borrow_mut() += &b_update;
                }
            }
            Op::MatMul(a, b) => {
                let (a_update, b_update) = {
                    let av = a.0.value.borrow();
                    let bv = b.0.value.borrow();
                    let a2 = av.clone().into_dimensionality::<ndarray::Ix2>().expect("MatMul requires 2D input A");
                    let b2 = bv.clone().into_dimensionality::<ndarray::Ix2>().expect("MatMul requires 2D input B");
                    let g2 = g.clone().into_dimensionality::<ndarray::Ix2>().expect("MatMul requires 2D gradient");
                    (g2.dot(&b2.t()).into_dyn(), a2.t().dot(&g2).into_dyn())
                };
                *a.0.grad.borrow_mut() += &a_update;
                *b.0.grad.borrow_mut() += &b_update;
            }
            Op::ReLU(a) => {
                let av = a.0.value.borrow();
                let mut ag = a.0.grad.borrow_mut();
                for (agg, (avv, gv)) in ag.iter_mut().zip(av.iter().zip(g.iter())) {
                    if *avv > 0.0 {
                        *agg += gv;
                    }
                }
            }
            Op::Square(a) => {
                let av = a.0.value.borrow();
                *a.0.grad.borrow_mut() += &(&g * &*av * 2.0);
            }
            Op::Mean(a) => {
                let n = a.0.value.borrow().len() as f64;
                let g_val = *g.iter().next().expect("Mean gradient cannot be empty");
                let mut ag = a.0.grad.borrow_mut();
                ag.mapv_inplace(|v| v + g_val / n);
            }
            Op::Constant => {}
        }
    }
}

// Operator Overloading for Tensor
impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Tensor {
        add(&self, &rhs)
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        add(self, rhs)
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Tensor {
        sub(&self, &rhs)
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        sub(self, rhs)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Tensor {
        mul(&self, &rhs)
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        mul(self, rhs)
    }
}

// Helper functions
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_val = a.0.value.borrow().clone().into_dimensionality::<ndarray::Ix2>().expect("matmul: input A must be 2D");
    let b_val = b.0.value.borrow().clone().into_dimensionality::<ndarray::Ix2>().expect("matmul: input B must be 2D");
    Tensor::new(a_val.dot(&b_val).into_dyn(), Op::MatMul(a.clone(), b.clone()))
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    Tensor::new(&*a.0.value.borrow() + &*b.0.value.borrow(), Op::Add(a.clone(), b.clone()))
}

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    Tensor::new(&*a.0.value.borrow() - &*b.0.value.borrow(), Op::Sub(a.clone(), b.clone()))
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    Tensor::new(&*a.0.value.borrow() * &*b.0.value.borrow(), Op::Mul(a.clone(), b.clone()))
}

pub fn relu(a: &Tensor) -> Tensor {
    let val = a.0.value.borrow().mapv(|v| if v > 0.0 { v } else { 0.0 });
    Tensor::new(val, Op::ReLU(a.clone()))
}

pub fn square(a: &Tensor) -> Tensor {
    let val = a.0.value.borrow().mapv(|v| v * v);
    Tensor::new(val, Op::Square(a.clone()))
}

pub fn mean(a: &Tensor) -> Tensor {
    let val = a.0.value.borrow().mean().unwrap_or(0.0);
    Tensor::new(ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), val), Op::Mean(a.clone()))
}
