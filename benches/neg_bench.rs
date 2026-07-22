use arrow_array::{Float64Array, Int32Array};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    {
        let data = Int32Array::from((0..10_000_000).map(|_| rng.i32(..)).collect_vec());

        let our_res = arrow_compile_compute::arith::neg_wrapping(&data).unwrap();
        let arr_res = arrow_arith::numeric::neg_wrapping(&data).unwrap();
        assert_eq!(
            arrow_ord::cmp::eq(&our_res, &arr_res).unwrap().true_count(),
            our_res.len()
        );

        c.bench_function("neg i32/llvm", |b| {
            b.iter(|| arrow_compile_compute::arith::neg_wrapping(black_box(&data)).unwrap())
        });

        c.bench_function("neg i32/arrow", |b| {
            b.iter(|| arrow_arith::numeric::neg_wrapping(black_box(&data)).unwrap())
        });
    }

    {
        let data = Float64Array::from((0..10_000_000).map(|_| rng.f64()).collect_vec());

        let our_res = arrow_compile_compute::arith::neg_wrapping(&data).unwrap();
        let arr_res = arrow_arith::numeric::neg_wrapping(&data).unwrap();
        assert_eq!(
            arrow_ord::cmp::eq(&our_res, &arr_res).unwrap().true_count(),
            our_res.len()
        );

        c.bench_function("neg f64/llvm", |b| {
            b.iter(|| arrow_compile_compute::arith::neg_wrapping(black_box(&data)).unwrap())
        });

        c.bench_function("neg f64/arrow", |b| {
            b.iter(|| arrow_arith::numeric::neg_wrapping(black_box(&data)).unwrap())
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
