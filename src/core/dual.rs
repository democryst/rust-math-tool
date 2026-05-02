use ndarray::{ArrayD, IxDyn};
use std::ops::{Add, Sub, Mul, Div};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    pub real: f64,
    pub eps: f64,
}

impl Dual {
    pub fn new(real: f64, eps: f64) -> Self {
        Self { real, eps }
    }

    pub fn constant(real: f64) -> Self {
        Self { real, eps: 0.0 }
    }
}

// (a + bϵ) + (c + dϵ) = (a + c) + (b + d)ϵ
impl Add for Dual {
    type Output = Dual;
    fn add(self, other: Self) -> Dual {
        Dual::new(self.real + other.real, self.eps + other.eps)
    }
}

// (a + bϵ) - (c + dϵ) = (a - c) + (b - d)ϵ
impl Sub for Dual {
    type Output = Dual;
    fn sub(self, other: Self) -> Dual {
        Dual::new(self.real - other.real, self.eps - other.eps)
    }
}

// (a + bϵ) * (c + dϵ) = ac + (ad + bc)ϵ
impl Mul for Dual {
    type Output = Dual;
    fn mul(self, other: Self) -> Dual {
        Dual::new(
            self.real * other.real,
            self.real * other.eps + self.eps * other.real,
        )
    }
}

// (a + bϵ) / (c + dϵ) = (a/c) + ((bc - ad)/c^2)ϵ
impl Div for Dual {
    type Output = Dual;
    fn div(self, other: Self) -> Dual {
        Dual::new(
            self.real / other.real,
            (self.eps * other.real - self.real * other.eps) / (other.real * other.real),
        )
    }
}

#[derive(Clone, Debug)]
pub struct DualTensor {
    pub value: ArrayD<Dual>,
}

impl DualTensor {
    pub fn new(value: ArrayD<Dual>) -> Self {
        Self { value }
    }

    pub fn from_f64(value: ArrayD<f64>) -> Self {
        let dual_array = value.mapv(|x| Dual::constant(x));
        Self { value: dual_array }
    }

    pub fn set_gradient_seed(&mut self, index: &[usize]) {
        if let Some(elem) = self.value.get_mut(index) {
            elem.eps = 1.0;
        }
    }

    pub fn get_real(&self) -> ArrayD<f64> {
        self.value.mapv(|x| x.real)
    }

    pub fn get_eps(&self) -> ArrayD<f64> {
        self.value.mapv(|x| x.eps)
    }
}

// Simple implementations for DualTensor math
impl Add for &DualTensor {
    type Output = DualTensor;
    fn add(self, rhs: Self) -> DualTensor {
        let mut result = ndarray::ArrayD::from_elem(self.value.raw_dim(), Dual::constant(0.0));
        ndarray::azip!((res in &mut result, a in &self.value, b in &rhs.value) *res = *a + *b);
        DualTensor::new(result)
    }
}

impl Sub for &DualTensor {
    type Output = DualTensor;
    fn sub(self, rhs: Self) -> DualTensor {
        let mut result = ndarray::ArrayD::from_elem(self.value.raw_dim(), Dual::constant(0.0));
        ndarray::azip!((res in &mut result, a in &self.value, b in &rhs.value) *res = *a - *b);
        DualTensor::new(result)
    }
}

impl Mul for &DualTensor {
    type Output = DualTensor;
    fn mul(self, rhs: Self) -> DualTensor {
        let mut result = ndarray::ArrayD::from_elem(self.value.raw_dim(), Dual::constant(0.0));
        ndarray::azip!((res in &mut result, a in &self.value, b in &rhs.value) *res = *a * *b);
        DualTensor::new(result)
    }
}
