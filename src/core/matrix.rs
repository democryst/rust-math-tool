use ndarray::Array2;
use super::vector::MathError;

pub type Result<T> = std::result::Result<T, MathError>;

/// Basic Matrix operations wrapper around ndarray::Array2
pub struct Matrix {
    data: Array2<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(MathError::DimensionMismatch {
                expected: rows * cols,
                found: data.len(),
            });
        }
        
        let arr = Array2::from_shape_vec((rows, cols), data)
            .map_err(|_| MathError::DimensionMismatch { 
                expected: rows * cols, 
                found: 0 // Simplification for shape error
            })?;
            
        Ok(Self { data: arr })
    }

    pub fn shape(&self) -> (usize, usize) {
        let shape = self.data.shape();
        (shape[0], shape[1])
    }

    pub fn multiply(&self, other: &Matrix) -> Result<Matrix> {
        let (_r1, c1) = self.shape();
        let (r2, _c2) = other.shape();
        
        if c1 != r2 {
            return Err(MathError::DimensionMismatch {
                expected: c1,
                found: r2,
            });
        }
        
        Ok(Matrix {
            data: self.data.dot(&other.data),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication() {
        let m1 = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let m2 = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let result = m1.multiply(&m2).unwrap();
        
        assert_eq!(result.shape(), (2, 2));
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [7+18+33, 8+20+36] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [28+45+66, 32+50+72] = [139, 154]
        assert_eq!(result.data[[0, 0]], 58.0);
        assert_eq!(result.data[[1, 1]], 154.0);
    }
}
