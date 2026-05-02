use ndarray::Array1;
use num_complex::Complex64;
use rustfft::FftPlanner;
use crate::core::vector::MathError;

/// Performs a Fast Fourier Transform (FFT) on a 1D complex array.
pub fn fft(input: &Array1<Complex64>) -> Result<Array1<Complex64>, MathError> {
    let mut planner = FftPlanner::new();
    let fft_plan = planner.plan_fft_forward(input.len());

    let mut buffer = input.to_vec();
    fft_plan.process(&mut buffer);

    Ok(Array1::from_vec(buffer))
}

/// Performs an Inverse Fast Fourier Transform (IFFT) on a 1D complex array.
pub fn ifft(input: &Array1<Complex64>) -> Result<Array1<Complex64>, MathError> {
    let mut planner = FftPlanner::new();
    let ifft_plan = planner.plan_fft_inverse(input.len());

    let mut buffer = input.to_vec();
    ifft_plan.process(&mut buffer);

    // Normalize by N since rustfft does not normalize inverse transforms automatically
    let n = input.len() as f64;
    for val in buffer.iter_mut() {
        *val = *val / n;
    }

    Ok(Array1::from_vec(buffer))
}
