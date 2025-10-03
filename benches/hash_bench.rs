use arrow_array::{Int32Array, Int64Array};
use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    {
        let data = (0..100_000_000).map(|_| rng.i32(..)).collect_vec();
        let data = Int32Array::from(data);

        c.bench_function("murmur i32", |b| {
            b.iter(|| arrow_compile_compute::compute::hash(&data).unwrap())
        });

        c.bench_function("unchained i32", |b| {
            b.iter(|| arrow_compile_compute::compute::hash_unchained(&data).unwrap())
        });
    }

    {
        let data = (0..100_000_000).map(|_| rng.i64(..)).collect_vec();
        let data = Int64Array::from(data);

        c.bench_function("murmur i64", |b| {
            b.iter(|| arrow_compile_compute::compute::hash(&data).unwrap())
        });

        c.bench_function("unchained i64", |b| {
            b.iter(|| arrow_compile_compute::compute::hash_unchained(&data).unwrap())
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
