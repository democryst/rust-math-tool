use ndarray::ArrayD;
use crate::core::autodiff::Tensor;
use crate::core::dual::DualTensor;

/// The Hybrid Switcher selects between Forward-Mode (Dual Numbers) and 
/// Reverse-Mode (Tape/DAG) Automatic Differentiation based on the 
/// Jacobian dimensions of the target function.
pub struct HybridEngine;

impl HybridEngine {
    /// Executes the provided function using the optimal autodiff strategy.
    /// 
    /// - If `input_dim < output_dim`: Uses Forward-Mode (DualTensor)
    /// - If `input_dim >= output_dim`: Uses Reverse-Mode (Tensor)
    pub fn optimize_and_run<F_rev, F_fwd>(
        input: ArrayD<f64>,
        output_dim: usize,
        f_rev: F_rev,
        f_fwd: F_fwd,
    ) -> (ArrayD<f64>, ArrayD<f64>)
    where
        F_rev: Fn(&Tensor) -> Tensor,
        F_fwd: Fn(&DualTensor) -> DualTensor,
    {
        let input_dim = input.len();

        if input_dim < output_dim {
            // Forward Mode (Dual Numbers)
            // Note: For a full Jacobian, we would run this `input_dim` times.
            // For simplicity in Phase 2, we just run it once for the first dimension 
            // as a demonstration of the routing logic.
            let mut dual_input = DualTensor::from_f64(input);
            dual_input.set_gradient_seed(&[0]); // Seed first dimension
            
            let output = f_fwd(&dual_input);
            (output.get_real(), output.get_eps())
        } else {
            // Reverse Mode (Tape/DAG)
            let tensor_input = Tensor::variable(input);
            let output = f_rev(&tensor_input);
            output.backward();
            
            let val = output.0.value.borrow().clone();
            let grad = tensor_input.0.grad.borrow().clone();
            (val, grad)
        }
    }
}
