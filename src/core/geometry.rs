use crate::core::matrix::Matrix;
use crate::core::vector::MathError;

/// Computes the Empirical Fisher Information Matrix (FIM).
/// 
/// The exact Fisher Information Matrix is the expected value of the outer product of the score function.
/// In practice (such as for Natural Gradient Descent), this is approximated using empirical samples.
/// 
/// `score_vectors` is an N x P matrix, where N is the number of samples and P is the number of parameters.
/// Each row represents the gradient of the log-likelihood for a single sample: ∇_θ log p(x_i | θ).
/// 
/// Returns a P x P matrix representing the empirical FIM.
pub fn empirical_fisher_information_matrix(score_vectors: &Matrix) -> Result<Matrix, MathError> {
    let (num_samples, num_params) = score_vectors.shape();
    
    if num_samples == 0 || num_params == 0 {
        return Err(MathError::DimensionMismatch { expected: 1, found: 0 });
    }
    
    // FIM = (1 / N) * \sum_{i=1}^{N} [ g_i * g_i^T ]
    // This is mathematically equivalent to (1 / N) * (S^T * S)
    // where S is the N x P score matrix.
    
    let score_t = score_vectors.transpose()?;
    let sum_outer_products = score_t.multiply(score_vectors)?;
    
    // Scale by 1 / N
    let scale_factor = 1.0 / (num_samples as f64);
    let mut scaled_data = sum_outer_products.tensor.0.value.borrow().clone().into_raw_vec_and_offset().0;
    for val in scaled_data.iter_mut() {
        *val *= scale_factor;
    }
    
    Matrix::new(num_params, num_params, scaled_data)
}
