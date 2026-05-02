use ndarray::Array1;

#[derive(Debug, PartialEq)]
pub enum MathError {
    DimensionMismatch { expected: usize, found: usize },
}

pub type Result<T> = std::result::Result<T, MathError>;

/// Basic Vector operations wrapper around ndarray::Array1
pub struct Vector {
    data: Array1<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data: Array1::from_vec(data),
        }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }

    pub fn add(&self, other: &Vector) -> Result<Vector> {
        if self.dim() != other.dim() {
            return Err(MathError::DimensionMismatch {
                expected: self.dim(),
                found: other.dim(),
            });
        }
        Ok(Vector {
            data: &self.data + &other.data,
        })
    }

    pub fn sub(&self, other: &Vector) -> Result<Vector> {
        if self.dim() != other.dim() {
            return Err(MathError::DimensionMismatch {
                expected: self.dim(),
                found: other.dim(),
            });
        }
        Ok(Vector {
            data: &self.data - &other.data,
        })
    }

    pub fn scale(&self, scalar: f64) -> Vector {
        Vector {
            data: &self.data * scalar,
        }
    }
    
    pub fn dot(&self, other: &Vector) -> Result<f64> {
        if self.dim() != other.dim() {
            return Err(MathError::DimensionMismatch {
                expected: self.dim(),
                found: other.dim(),
            });
        }
        Ok(self.data.dot(&other.data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_addition() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let result = v1.add(&v2).unwrap();
        assert_eq!(result.data, Array1::from_vec(vec![5.0, 7.0, 9.0]));
    }

    #[test]
    fn test_vector_dimension_mismatch() {
        let v1 = Vector::new(vec![1.0, 2.0]);
        let v2 = Vector::new(vec![1.0, 2.0, 3.0]);
        let result = v1.add(&v2);
        assert!(matches!(result, Err(MathError::DimensionMismatch { .. })));
    }

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn test_addition_is_commutative(a in any::<Vec<f64>>(), b in any::<Vec<f64>>()) {
            // Only test equal lengths
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];
            
            let v1 = Vector::new(a.to_vec());
            let v2 = Vector::new(b.to_vec());
            
            let res1 = v1.add(&v2).unwrap();
            let res2 = v2.add(&v1).unwrap();
            
            assert_eq!(res1.data, res2.data);
        }
    }
}
