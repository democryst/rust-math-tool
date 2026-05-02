#![no_main]

use libfuzzer_sys::fuzz_target;
use rust_math_tool::{Vector, Matrix};

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }
    
    // Split data into two halves
    let mid = data.len() / 2;
    let (left, right) = data.split_at(mid);
    
    // Convert u8 bytes to f64 by simple casting (or just mapping directly to prevent NaN complexity)
    let left_f64: Vec<f64> = left.iter().map(|&x| (x as f64) / 255.0).collect();
    let right_f64: Vec<f64> = right.iter().map(|&x| (x as f64) / 255.0).collect();

    // 1. Vector Operations
    let v1 = Vector::new(left_f64.clone());
    let v2 = Vector::new(right_f64.clone());
    let _ = v1.add(&v2);
    let _ = v1.sub(&v2);

    // 2. Matrix Operations
    let rows = 2;
    let cols_left = left_f64.len() / rows;
    let cols_right = right_f64.len() / rows;
    
    if cols_left > 0 && cols_right > 0 {
        if let Ok(m1) = Matrix::new(rows, cols_left, left_f64[..rows * cols_left].to_vec()) {
            if let Ok(m2) = Matrix::new(rows, cols_right, right_f64[..rows * cols_right].to_vec()) {
                let _ = m1.add(&m2);
                let _ = m1.multiply(&m2);
            }
        }
    }
});
