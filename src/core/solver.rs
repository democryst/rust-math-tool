use crate::core::vector::MathError;

/// Solves a Linear Program (LP) in standard form:
///   minimize    c^T x
///   subject to  Ax <= b,  x >= 0
///
/// Uses an interior-point log-barrier method with a decaying barrier weight μ.
/// μ starts at 1.0 and is multiplied by `mu_decay` each outer iteration,
/// so the barrier term fades and the solution converges to the true LP optimum.
pub fn solve_lp(
    c: &[f64],
    a: &[Vec<f64>],
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, MathError> {
    let n = c.len();
    let m = a.len();

    if b.len() != m {
        return Err(MathError::DimensionMismatch { expected: m, found: b.len() });
    }
    for row in a {
        if row.len() != n {
            return Err(MathError::DimensionMismatch { expected: n, found: row.len() });
        }
    }

    let mut x = vec![0.1f64; n];
    let lr = 0.005;
    let mu_decay = 0.995;
    let mut mu = 1.0f64; // barrier weight

    for _ in 0..max_iter {
        let mut grad = c.to_vec();

        // Log-barrier for Ax <= b with decaying weight mu
        for i in 0..m {
            let ax_i: f64 = a[i].iter().zip(x.iter()).map(|(a_ij, x_j)| a_ij * x_j).sum();
            let slack = b[i] - ax_i;
            if slack <= 1e-12 {
                // Nudge x back inside feasibility
                for j in 0..n {
                    x[j] -= 0.01 * a[i][j].abs().max(0.01);
                    x[j] = x[j].max(1e-9);
                }
                continue;
            }
            for j in 0..n {
                grad[j] += mu * a[i][j] / slack;
            }
        }

        // Log-barrier for x >= 0
        for j in 0..n {
            x[j] = x[j].max(1e-9);
            grad[j] -= mu / x[j];
        }

        let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < tol {
            break;
        }

        for j in 0..n {
            x[j] -= lr * grad[j];
            x[j] = x[j].max(1e-9);
        }

        mu *= mu_decay;
    }

    Ok(x)
}

/// Solves a Quadratic Program (QP) in standard form:
///   minimize    (1/2) x^T Q x + c^T x
///   subject to  Ax <= b,  x >= 0
///
/// Uses an interior-point log-barrier method with a decaying barrier weight μ.
pub fn solve_qp(
    q: &[Vec<f64>],
    c: &[f64],
    a: &[Vec<f64>],
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, MathError> {
    let n = c.len();
    let m = a.len();

    if q.len() != n {
        return Err(MathError::DimensionMismatch { expected: n, found: q.len() });
    }
    if b.len() != m {
        return Err(MathError::DimensionMismatch { expected: m, found: b.len() });
    }

    let mut x = vec![0.1f64; n];
    let lr = 0.002;
    let mu_decay = 0.995;
    let mut mu = 1.0f64;

    for _ in 0..max_iter {
        // Gradient of (1/2) x^T Q x + c^T x
        let mut grad: Vec<f64> = c.to_vec();
        for i in 0..n {
            let qx_i: f64 = q[i].iter().zip(x.iter()).map(|(q_ij, x_j)| q_ij * x_j).sum();
            grad[i] += qx_i;
        }

        // Log-barrier for Ax <= b
        for i in 0..m {
            let ax_i: f64 = a[i].iter().zip(x.iter()).map(|(a_ij, x_j)| a_ij * x_j).sum();
            let slack = b[i] - ax_i;
            if slack <= 1e-12 {
                // Nudge back into feasibility
                for j in 0..n {
                    x[j] -= 0.01 * a[i][j].abs().max(0.01);
                    x[j] = x[j].max(1e-9);
                }
                continue;
            }
            for j in 0..n {
                grad[j] += mu * a[i][j] / slack;
            }
        }

        // Log-barrier for x >= 0
        for j in 0..n {
            x[j] = x[j].max(1e-9);
            grad[j] -= mu / x[j];
        }

        let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < tol {
            break;
        }

        for j in 0..n {
            x[j] -= lr * grad[j];
            x[j] = x[j].max(1e-9);
        }

        mu *= mu_decay;
    }

    Ok(x)
}
