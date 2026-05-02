use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_math_tool::{Vector, Matrix};

fn bench_vector_addition(c: &mut Criterion) {
    let size = 1_000_000;
    let v1 = Vector::new(vec![1.0; size]);
    let v2 = Vector::new(vec![2.0; size]);
    
    c.bench_function("vector_addition_1M", |b| {
        b.iter(|| v1.add(black_box(&v2)).unwrap())
    });
}

fn bench_matrix_multiplication(c: &mut Criterion) {
    let size = 500;
    let m1 = Matrix::new(size, size, vec![1.0; size * size]).unwrap();
    let m2 = Matrix::new(size, size, vec![2.0; size * size]).unwrap();
    
    c.bench_function("matrix_multiplication_500x500", |b| {
        b.iter(|| m1.multiply(black_box(&m2)).unwrap())
    });
}

criterion_group!(benches, bench_vector_addition, bench_matrix_multiplication);
criterion_main!(benches);
