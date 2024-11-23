use arrow_array::{Int32Array, Int64Array};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    let data1 = (0..100_000).map(|_| rng.i32(..)).collect_vec();
    let data1 = Int32Array::from(data1);
    let data2 = (0..100_000).map(|_| rng.i64(..)).collect_vec();
    let data2 = Int64Array::from(data2);

    let _r = arrow_compile_compute::cmp::lt(&data1, &data2).unwrap();
    c.bench_function("parallel/serial", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                black_box(arrow_compile_compute::cmp::lt(&data1, &data2).unwrap());
            }
        });
    });

    c.bench_function("parallel/rayon", |b| {
        b.iter(|| {
            (0..1000).into_par_iter().for_each(|_| {
                black_box(arrow_compile_compute::cmp::lt(&data1, &data2).unwrap());
            });
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
