use ndarray::Array1;
use crate::core::vector::MathError;

/// Computes the Kullback-Leibler Divergence: D_KL(P || Q) = sum(P * log(P / Q))
/// Requires P and Q to be valid probability distributions (sum to 1, values > 0).
pub fn kl_divergence(p: &Array1<f64>, q: &Array1<f64>) -> Result<f64, MathError> {
    if p.len() != q.len() {
        return Err(MathError::DimensionMismatch { expected: p.len(), found: q.len() });
    }
    
    let mut div = 0.0;
    for (p_i, q_i) in p.iter().zip(q.iter()) {
        if *p_i < 0.0 || *q_i <= 0.0 {
            // In a strict statistical sense, P must be >= 0 and Q > 0.
            return Err(MathError::Singularity("KL Divergence requires strictly positive probabilities".to_string()));
        }
        if *p_i > 0.0 {
            div += *p_i * (*p_i / *q_i).ln();
        }
    }
    
    Ok(div)
}

/// Computes the Cosine Similarity: (A dot B) / (||A|| * ||B||)
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64, MathError> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch { expected: a.len(), found: b.len() });
    }
    
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(MathError::Singularity("Cannot compute cosine similarity of a zero vector".to_string()));
    }
    
    Ok(dot_product / (norm_a * norm_b))
}

/// Computes the 1D Wasserstein Distance (Earth Mover's Distance) for discrete distributions.
/// Assuming p and q are probability mass functions defined on the same evenly spaced 1D grid.
pub fn wasserstein_distance_1d(p: &Array1<f64>, q: &Array1<f64>) -> Result<f64, MathError> {
    if p.len() != q.len() {
        return Err(MathError::DimensionMismatch { expected: p.len(), found: q.len() });
    }
    
    // Compute Cumulative Distribution Functions (CDF)
    let mut cdf_p = 0.0;
    let mut cdf_q = 0.0;
    let mut distance = 0.0;
    
    for (p_i, q_i) in p.iter().zip(q.iter()) {
        cdf_p += p_i;
        cdf_q += q_i;
        distance += (cdf_p - cdf_q).abs();
    }
    
    Ok(distance)
}
