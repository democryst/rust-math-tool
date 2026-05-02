use crate::core::autodiff::Tensor;
use ndarray::{ArrayD, IxDyn};

// ─── SGD ─────────────────────────────────────────────────────────────────────

pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    pub fn step(&self, parameters: &[Tensor]) {
        for param in parameters {
            let mut val = param.0.value.borrow_mut();
            let grad = param.0.grad.borrow();
            *val -= &(&*grad * self.learning_rate);
        }
    }

    pub fn zero_grad(&self, parameters: &[Tensor]) {
        for param in parameters {
            let shape = param.0.grad.borrow().shape().to_vec();
            *param.0.grad.borrow_mut() = ArrayD::zeros(IxDyn(&shape));
        }
    }
}

// ─── Adam ─────────────────────────────────────────────────────────────────────
// Implements: Adam (Adaptive Moment Estimation)
// Paper: Kingma & Ba (2015) - https://arxiv.org/abs/1412.6980
// Update rule:
//   m_t = β1 * m_{t-1} + (1 - β1) * g_t        (1st moment / mean)
//   v_t = β2 * v_{t-1} + (1 - β2) * g_t^2      (2nd moment / variance)
//   m̂_t = m_t / (1 - β1^t)                     (bias-corrected)
//   v̂_t = v_t / (1 - β2^t)                     (bias-corrected)
//   θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)

pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub t: usize,
    pub m: Vec<ArrayD<f64>>,
    pub v: Vec<ArrayD<f64>>,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, m: vec![], v: vec![] }
    }

    pub fn with_betas(lr: f64, beta1: f64, beta2: f64, eps: f64) -> Self {
        Self { lr, beta1, beta2, eps, t: 0, m: vec![], v: vec![] }
    }

    pub fn step(&mut self, parameters: &[Tensor]) {
        self.t += 1;

        // Lazily initialize moment buffers on first step
        if self.m.is_empty() {
            for p in parameters {
                let shape = p.0.value.borrow().shape().to_vec();
                self.m.push(ArrayD::zeros(IxDyn(&shape)));
                self.v.push(ArrayD::zeros(IxDyn(&shape)));
            }
        }

        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, param) in parameters.iter().enumerate() {
            let grad = param.0.grad.borrow().clone();

            // Update first moment
            self.m[i] = &self.m[i] * self.beta1 + &(&grad * (1.0 - self.beta1));
            // Update second moment
            self.v[i] = &self.v[i] * self.beta2 + &(&grad * &grad * (1.0 - self.beta2));

            // Bias-corrected moments
            let m_hat = &self.m[i] / bc1;
            let v_hat = &self.v[i] / bc2;

            // Parameter update
            let update = &m_hat / &(v_hat.mapv(f64::sqrt) + self.eps);
            *param.0.value.borrow_mut() -= &(&update * self.lr);
        }
    }

    pub fn zero_grad(&self, parameters: &[Tensor]) {
        for param in parameters {
            let shape = param.0.grad.borrow().shape().to_vec();
            *param.0.grad.borrow_mut() = ArrayD::zeros(IxDyn(&shape));
        }
    }
}

// ─── AdamW ────────────────────────────────────────────────────────────────────
// Implements: AdamW (Adam with Decoupled Weight Decay)
// Paper: Loshchilov & Hutter (2019) - https://arxiv.org/abs/1711.05101
// Adds: θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})

pub struct AdamW {
    pub adam: Adam,
    pub weight_decay: f64,
}

impl AdamW {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self { adam: Adam::new(lr), weight_decay }
    }

    pub fn step(&mut self, parameters: &[Tensor]) {
        // Apply decoupled weight decay directly to parameters BEFORE the Adam update
        for param in parameters {
            let wd = self.weight_decay;
            let lr = self.adam.lr;
            let mut val = param.0.value.borrow_mut();
            *val *= 1.0 - lr * wd;
        }
        self.adam.step(parameters);
    }

    pub fn zero_grad(&self, parameters: &[Tensor]) {
        self.adam.zero_grad(parameters);
    }
}

// ─── RMSprop ──────────────────────────────────────────────────────────────────
// Implements: RMSprop (Root Mean Square Propagation)
// Proposed by: Hinton (2012)
// Update rule:
//   v_t = α * v_{t-1} + (1 - α) * g_t^2
//   θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)

pub struct RMSprop {
    pub lr: f64,
    pub alpha: f64,
    pub eps: f64,
    pub v: Vec<ArrayD<f64>>,
}

impl RMSprop {
    pub fn new(lr: f64) -> Self {
        Self { lr, alpha: 0.99, eps: 1e-8, v: vec![] }
    }

    pub fn step(&mut self, parameters: &[Tensor]) {
        if self.v.is_empty() {
            for p in parameters {
                let shape = p.0.value.borrow().shape().to_vec();
                self.v.push(ArrayD::zeros(IxDyn(&shape)));
            }
        }

        for (i, param) in parameters.iter().enumerate() {
            let grad = param.0.grad.borrow().clone();

            self.v[i] = &self.v[i] * self.alpha + &(&grad * &grad * (1.0 - self.alpha));

            let update = &grad / &(self.v[i].mapv(f64::sqrt) + self.eps);
            *param.0.value.borrow_mut() -= &(&update * self.lr);
        }
    }

    pub fn zero_grad(&self, parameters: &[Tensor]) {
        for param in parameters {
            let shape = param.0.grad.borrow().shape().to_vec();
            *param.0.grad.borrow_mut() = ArrayD::zeros(IxDyn(&shape));
        }
    }
}

// ─── L-BFGS ───────────────────────────────────────────────────────────────────
// Implements: Limited-memory BFGS (L-BFGS)
// Paper: Liu & Nocedal (1989) - https://link.springer.com/article/10.1007/BF01589116
// L-BFGS approximates the inverse Hessian using m most-recent (s, y) pairs.
// Update rule (two-loop recursion):
//   q = ∇f
//   for each (s_i, y_i) in reverse:
//     α_i = ρ_i * s_i^T * q; q = q - α_i * y_i
//   r = H_0 * q  (initial Hessian approximation, typically I)
//   for each (s_i, y_i) in forward:
//     β_i = ρ_i * y_i^T * r; r = r + s_i * (α_i - β_i)
//   θ = θ - r

pub struct LBFGS {
    pub lr: f64,
    pub m: usize,                   // Memory size (history)
    s_history: Vec<ArrayD<f64>>,    // s_k = θ_{k+1} - θ_k
    y_history: Vec<ArrayD<f64>>,    // y_k = ∇f_{k+1} - ∇f_k
    prev_params: Vec<ArrayD<f64>>,
    prev_grads: Vec<ArrayD<f64>>,
}

impl LBFGS {
    pub fn new(lr: f64, m: usize) -> Self {
        Self { lr, m, s_history: vec![], y_history: vec![], prev_params: vec![], prev_grads: vec![] }
    }

    pub fn step(&mut self, parameters: &[Tensor]) {
        let grads: Vec<ArrayD<f64>> = parameters.iter().map(|p| p.0.grad.borrow().clone()).collect();
        let params: Vec<ArrayD<f64>> = parameters.iter().map(|p| p.0.value.borrow().clone()).collect();

        if !self.prev_params.is_empty() {
            // Flatten all parameter/gradient deltas into single vectors
            let s: ArrayD<f64> = self.flatten_diff(&params, &self.prev_params.clone());
            let y: ArrayD<f64> = self.flatten_diff(&grads, &self.prev_grads.clone());

            // Only store if curvature condition (y^T s > 0) holds
            let ys = y.iter().zip(s.iter()).map(|(a, b)| a * b).sum::<f64>();
            if ys > 1e-10 {
                self.s_history.push(s);
                self.y_history.push(y);
                if self.s_history.len() > self.m {
                    self.s_history.remove(0);
                    self.y_history.remove(0);
                }
            }
        }

        // Compute L-BFGS direction via two-loop recursion on flattened grad
        let flat_grad = self.flatten(&grads);
        let direction = self.two_loop_recursion(flat_grad);

        // Apply update to each parameter tensor
        let mut offset = 0;
        for param in parameters.iter() {
            let shape = param.0.value.borrow().shape().to_vec();
            let len: usize = shape.iter().product();
            let d_slice = ArrayD::from_shape_vec(
                IxDyn(&shape),
                direction.iter().skip(offset).take(len).cloned().collect(),
            ).expect("L-BFGS: direction slice shape mismatch");
            *param.0.value.borrow_mut() -= &(&d_slice * self.lr);
            offset += len;
        }

        self.prev_params = params;
        self.prev_grads = grads;
    }

    fn flatten(&self, arrays: &[ArrayD<f64>]) -> ArrayD<f64> {
        let flat: Vec<f64> = arrays.iter().flat_map(|a| a.iter().cloned()).collect();
        ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).expect("L-BFGS flatten failed")
    }

    fn flatten_diff(&self, a: &[ArrayD<f64>], b: &[ArrayD<f64>]) -> ArrayD<f64> {
        let flat: Vec<f64> = a.iter().zip(b.iter())
            .flat_map(|(ai, bi)| ai.iter().zip(bi.iter()).map(|(x, y)| x - y))
            .collect();
        ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).expect("L-BFGS flatten_diff failed")
    }

    fn two_loop_recursion(&self, grad: ArrayD<f64>) -> ArrayD<f64> {
        let k = self.s_history.len();
        if k == 0 {
            return grad;
        }

        let mut q = grad.clone();
        let mut alphas = vec![0.0f64; k];

        // Backward pass
        for i in (0..k).rev() {
            let rho = 1.0 / self.y_history[i].iter().zip(self.s_history[i].iter()).map(|(y, s)| y * s).sum::<f64>();
            let alpha = rho * self.s_history[i].iter().zip(q.iter()).map(|(s, qi)| s * qi).sum::<f64>();
            alphas[i] = alpha;
            q = &q - &(&self.y_history[i] * alpha);
        }

        // Initial Hessian approximation H_0 = (s_k^T y_k / y_k^T y_k) * I
        let last = k - 1;
        let sy = self.s_history[last].iter().zip(self.y_history[last].iter()).map(|(s, y)| s * y).sum::<f64>();
        let yy = self.y_history[last].iter().map(|y| y * y).sum::<f64>();
        let h0_scale = if yy > 1e-12 { sy / yy } else { 1.0 };
        let mut r = &q * h0_scale;

        // Forward pass
        for i in 0..k {
            let rho = 1.0 / self.y_history[i].iter().zip(self.s_history[i].iter()).map(|(y, s)| y * s).sum::<f64>();
            let beta = rho * self.y_history[i].iter().zip(r.iter()).map(|(y, ri)| y * ri).sum::<f64>();
            r = &r + &(&self.s_history[i] * (alphas[i] - beta));
        }

        r
    }

    pub fn zero_grad(&self, parameters: &[Tensor]) {
        for param in parameters {
            let shape = param.0.grad.borrow().shape().to_vec();
            *param.0.grad.borrow_mut() = ArrayD::zeros(IxDyn(&shape));
        }
    }
}
