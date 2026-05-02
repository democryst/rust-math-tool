use ndarray::{ArrayD, IxDyn};
use num_complex::Complex64;
use std::cell::RefCell;
use std::rc::Rc;
use std::ops::{Add, Sub, Mul};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub enum ComplexOp {
    Constant,
    Add(ComplexTensor, ComplexTensor),
    Sub(ComplexTensor, ComplexTensor),
    Mul(ComplexTensor, ComplexTensor),
    MatMul(ComplexTensor, ComplexTensor),
}

#[derive(Debug)]
pub struct ComplexNode {
    pub value: RefCell<ArrayD<Complex64>>,
    pub grad: RefCell<ArrayD<Complex64>>,
    pub op: ComplexOp,
    pub citation: String,
}

#[derive(Clone, Debug)]
pub struct ComplexTensor(pub Rc<ComplexNode>);

impl ComplexTensor {
    pub fn new(value: ArrayD<Complex64>, op: ComplexOp) -> Self {
        let shape = value.shape().to_vec();
        let citation = format!("{:?} operation on shape {:?}", op, shape);
        Self(Rc::new(ComplexNode {
            value: RefCell::new(value),
            grad: RefCell::new(ArrayD::zeros(IxDyn(&shape))),
            op,
            citation,
        }))
    }

    pub fn variable(value: ArrayD<Complex64>) -> Self {
        let shape = value.shape().to_vec();
        Self(Rc::new(ComplexNode {
            value: RefCell::new(value),
            grad: RefCell::new(ArrayD::zeros(IxDyn(&shape))),
            op: ComplexOp::Constant,
            citation: format!("Complex Variable initialized with shape {:?}", shape),
        }))
    }

    pub fn constant(value: ArrayD<Complex64>) -> Self {
        let shape = value.shape().to_vec();
        Self(Rc::new(ComplexNode {
            value: RefCell::new(value),
            grad: RefCell::new(ArrayD::zeros(IxDyn(&shape))),
            op: ComplexOp::Constant,
            citation: format!("Complex Constant initialized with shape {:?}", shape),
        }))
    }

    pub fn verify(&self) -> bool {
        use crate::core::integrity::IntegrityGate;
        let val = self.0.value.borrow();
        let claim = format!("ComplexTensor result {:?} is correct", *val);
        IntegrityGate::verify_claim(&claim, &self.0.citation)
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        self.build_topo(&mut visited, &mut topo);

        let shape = self.0.grad.borrow().shape().to_vec();
        *self.0.grad.borrow_mut() = ArrayD::from_elem(IxDyn(&shape), Complex64::new(1.0, 0.0));

        for node in topo.iter().rev() {
            node.propagate_step();
        }
    }

    fn build_topo(&self, visited: &mut HashSet<*const ComplexNode>, topo: &mut Vec<ComplexTensor>) {
        let ptr = Rc::as_ptr(&self.0);
        if !visited.contains(&ptr) {
            visited.insert(ptr);
            match &self.0.op {
                ComplexOp::Add(a, b) | ComplexOp::Sub(a, b) | ComplexOp::Mul(a, b) | ComplexOp::MatMul(a, b) => {
                    a.build_topo(visited, topo);
                    b.build_topo(visited, topo);
                }
                ComplexOp::Constant => {}
            }
            topo.push(self.clone());
        }
    }

    fn propagate_step(&self) {
        let g = self.0.grad.borrow().clone();
        match &self.0.op {
            ComplexOp::Add(a, b) => {
                let mut ag = a.0.grad.borrow_mut();
                if ag.shape() != g.shape() {
                    let summed = g.sum_axis(ndarray::Axis(0));
                    let reshaped = summed.into_shape_with_order(ag.shape()).expect("Gradient shape mismatch during Complex Add propagation");
                    *ag += &reshaped.into_dyn();
                } else {
                    *ag += &g;
                }
                drop(ag);

                if !Rc::ptr_eq(&a.0, &b.0) {
                    let mut bg = b.0.grad.borrow_mut();
                    if bg.shape() != g.shape() {
                        let summed = g.sum_axis(ndarray::Axis(0));
                        let reshaped = summed.into_shape_with_order(bg.shape()).expect("Gradient shape mismatch during broadcast Complex Add propagation");
                        *bg += &reshaped.into_dyn();
                    } else {
                        *bg += &g;
                    }
                } else {
                    let mut ag = a.0.grad.borrow_mut();
                    if ag.shape() != g.shape() {
                        let summed = g.sum_axis(ndarray::Axis(0));
                        let reshaped = summed.into_shape_with_order(ag.shape()).expect("Gradient shape mismatch during identity Complex Add propagation");
                        *ag += &reshaped.into_dyn();
                    } else {
                        *ag += &g;
                    }
                }
            }
            ComplexOp::Sub(a, b) => {
                *a.0.grad.borrow_mut() += &g;
                if !Rc::ptr_eq(&a.0, &b.0) {
                    *b.0.grad.borrow_mut() -= &g;
                }
            }
            ComplexOp::Mul(a, b) => {
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
            ComplexOp::MatMul(a, b) => {
                let (a_update, b_update) = {
                    let av = a.0.value.borrow();
                    let bv = b.0.value.borrow();
                    let a2 = av.clone().into_dimensionality::<ndarray::Ix2>().expect("Complex MatMul requires 2D input A");
                    let b2 = bv.clone().into_dimensionality::<ndarray::Ix2>().expect("Complex MatMul requires 2D input B");
                    let g2 = g.clone().into_dimensionality::<ndarray::Ix2>().expect("Complex MatMul requires 2D gradient");
                    // Note: Wirtinger calculus requires taking the conjugate transpose for complex gradients
                    // We'll use simple dot product for now, which may need adjustment in Phase 2
                    (g2.dot(&b2.t()).into_dyn(), a2.t().dot(&g2).into_dyn())
                };
                *a.0.grad.borrow_mut() += &a_update;
                *b.0.grad.borrow_mut() += &b_update;
            }
            ComplexOp::Constant => {}
        }
    }
}

pub fn add_complex(a: &ComplexTensor, b: &ComplexTensor) -> ComplexTensor {
    ComplexTensor::new(&*a.0.value.borrow() + &*b.0.value.borrow(), ComplexOp::Add(a.clone(), b.clone()))
}

pub fn sub_complex(a: &ComplexTensor, b: &ComplexTensor) -> ComplexTensor {
    ComplexTensor::new(&*a.0.value.borrow() - &*b.0.value.borrow(), ComplexOp::Sub(a.clone(), b.clone()))
}

pub fn mul_complex(a: &ComplexTensor, b: &ComplexTensor) -> ComplexTensor {
    ComplexTensor::new(&*a.0.value.borrow() * &*b.0.value.borrow(), ComplexOp::Mul(a.clone(), b.clone()))
}

pub fn matmul_complex(a: &ComplexTensor, b: &ComplexTensor) -> ComplexTensor {
    let a_val = a.0.value.borrow().clone().into_dimensionality::<ndarray::Ix2>().expect("matmul: input A must be 2D");
    let b_val = b.0.value.borrow().clone().into_dimensionality::<ndarray::Ix2>().expect("matmul: input B must be 2D");
    ComplexTensor::new(a_val.dot(&b_val).into_dyn(), ComplexOp::MatMul(a.clone(), b.clone()))
}

impl Add for &ComplexTensor {
    type Output = ComplexTensor;
    fn add(self, rhs: Self) -> ComplexTensor {
        add_complex(self, rhs)
    }
}

impl Sub for &ComplexTensor {
    type Output = ComplexTensor;
    fn sub(self, rhs: Self) -> ComplexTensor {
        sub_complex(self, rhs)
    }
}

impl Mul for &ComplexTensor {
    type Output = ComplexTensor;
    fn mul(self, rhs: Self) -> ComplexTensor {
        mul_complex(self, rhs)
    }
}

impl Add for ComplexTensor {
    type Output = ComplexTensor;
    fn add(self, rhs: Self) -> ComplexTensor {
        add_complex(&self, &rhs)
    }
}

impl Sub for ComplexTensor {
    type Output = ComplexTensor;
    fn sub(self, rhs: Self) -> ComplexTensor {
        sub_complex(&self, &rhs)
    }
}

impl Mul for ComplexTensor {
    type Output = ComplexTensor;
    fn mul(self, rhs: Self) -> ComplexTensor {
        mul_complex(&self, &rhs)
    }
}
