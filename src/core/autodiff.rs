use std::ops::{Add, Sub, Mul, Div};

/// Dual Number for Forward-Mode Automatic Differentiation
/// Represents a value and its derivative: v + v'ε
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    pub value: f64,
    pub grad: f64,
}

impl Dual {
    pub fn new(value: f64, grad: f64) -> Self {
        Self { value, grad }
    }

    pub fn constant(value: f64) -> Self {
        Self { value, grad: 0.0 }
    }

    pub fn variable(value: f64) -> Self {
        Self { value, grad: 1.0 }
    }

    pub fn powi(&self, n: i32) -> Self {
        Self {
            value: self.value.powi(n),
            grad: self.grad * (n as f64) * self.value.powi(n - 1),
        }
    }
}

impl Add for Dual {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            grad: self.grad + other.grad,
        }
    }
}

impl Sub for Dual {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            grad: self.grad - other.grad,
        }
    }
}

impl Mul for Dual {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            grad: self.grad * other.value + self.value * other.grad,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let denom = other.value * other.value;
        Self {
            value: self.value / other.value,
            grad: (self.grad * other.value - self.value * other.grad) / denom,
        }
    }
}

/// Simplified Reverse-Mode AD for scalars
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Op {
    Constant(f64),
    Add(Rc<Node>, Rc<Node>),
    Sub(Rc<Node>, Rc<Node>),
    Mul(Rc<Node>, Rc<Node>),
    ReLU(Rc<Node>),
}

#[derive(Debug)]
pub struct Node {
    pub value: RefCell<f64>,
    pub grad: RefCell<f64>,
    pub op: Op,
}

impl Node {
    pub fn constant(v: f64) -> Rc<Self> {
        Rc::new(Self {
            value: RefCell::new(v),
            grad: RefCell::new(0.0),
            op: Op::Constant(v),
        })
    }

    pub fn variable(v: f64) -> Rc<Self> {
        Rc::new(Self {
            value: RefCell::new(v),
            grad: RefCell::new(0.0),
            op: Op::Constant(v),
        })
    }

    pub fn backward(self: &Rc<Self>) {
        *self.grad.borrow_mut() = 1.0;
        self.propagate();
    }

    fn propagate(self: &Rc<Self>) {
        let g = *self.grad.borrow();
        match &self.op {
            Op::Add(a, b) => {
                *a.grad.borrow_mut() += g;
                *b.grad.borrow_mut() += g;
                a.propagate();
                b.propagate();
            }
            Op::Sub(a, b) => {
                *a.grad.borrow_mut() += g;
                *b.grad.borrow_mut() -= g;
                a.propagate();
                b.propagate();
            }
            Op::Mul(a, b) => {
                *a.grad.borrow_mut() += g * *b.value.borrow();
                *b.grad.borrow_mut() += g * *a.value.borrow();
                a.propagate();
                b.propagate();
            }
            Op::ReLU(a) => {
                if *a.value.borrow() > 0.0 {
                    *a.grad.borrow_mut() += g;
                }
                a.propagate();
            }
            Op::Constant(_) => {}
        }
    }
}

pub fn add(a: &Rc<Node>, b: &Rc<Node>) -> Rc<Node> {
    Rc::new(Node {
        value: RefCell::new(*a.value.borrow() + *b.value.borrow()),
        grad: RefCell::new(0.0),
        op: Op::Add(a.clone(), b.clone()),
    })
}

pub fn sub(a: &Rc<Node>, b: &Rc<Node>) -> Rc<Node> {
    Rc::new(Node {
        value: RefCell::new(*a.value.borrow() - *b.value.borrow()),
        grad: RefCell::new(0.0),
        op: Op::Sub(a.clone(), b.clone()),
    })
}

pub fn mul(a: &Rc<Node>, b: &Rc<Node>) -> Rc<Node> {
    Rc::new(Node {
        value: RefCell::new(*a.value.borrow() * *b.value.borrow()),
        grad: RefCell::new(0.0),
        op: Op::Mul(a.clone(), b.clone()),
    })
}

pub fn relu(a: &Rc<Node>) -> Rc<Node> {
    let v = *a.value.borrow();
    Rc::new(Node {
        value: RefCell::new(if v > 0.0 { v } else { 0.0 }),
        grad: RefCell::new(0.0),
        op: Op::ReLU(a.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_dual_basic_ops() {
        // f(x) = x^2 + 3x + 5
        // f'(x) = 2x + 3
        // x = 2 => f(2) = 4 + 6 + 5 = 15, f'(2) = 4 + 3 = 7
        let x = Dual::variable(2.0);
        let result = (x * x) + (Dual::constant(3.0) * x) + Dual::constant(5.0);
        
        assert_eq!(result.value, 15.0);
        assert_eq!(result.grad, 7.0);
    }

    proptest! {
        #[test]
        fn test_dual_vs_finite_difference(val in -100.0..100.0) {
            let x = Dual::variable(val);
            // f(x) = x^3 - 2x
            // f'(x) = 3x^2 - 2
            let f = |x: Dual| (x * x * x) - (Dual::constant(2.0) * x);
            let result = f(x);
            
            let h = 1e-6;
            let f_val = |x: f64| (x * x * x) - (2.0 * x);
            let numerical_grad = (f_val(val + h) - f_val(val)) / h;
            
            let diff = (result.grad - numerical_grad).abs();
            assert!(diff < 1e-3, "AD grad {} vs Numerical grad {}", result.grad, numerical_grad);
        }
    }

    #[test]
    fn test_reverse_mode_basic() {
        // f(x, y) = x * y + x
        // df/dx = y + 1
        // df/dy = x
        // x=2, y=3 => f=9, df/dx=4, df/dy=2
        let x = Node::variable(2.0);
        let y = Node::variable(3.0);
        let xy = mul(&x, &y);
        let f = add(&xy, &x);
        
        f.backward();
        
        assert_eq!(*f.value.borrow(), 8.0);
        assert_eq!(*x.grad.borrow(), 4.0);
        assert_eq!(*y.grad.borrow(), 2.0);
    }
}
