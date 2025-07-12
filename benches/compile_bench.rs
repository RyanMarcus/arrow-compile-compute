use arrow_array::Int32Array;
use arrow_compile_compute::{cmp::ComparisonKernel, Kernel, Predicate};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    let data1 = (0..1).map(|_| rng.i32(..)).collect_vec();
    let data1 = Int32Array::from(data1);
    let data2 = (0..1).map(|_| rng.i32(..)).collect_vec();
    let data2 = Int32Array::from(data2);

    c.bench_function("compile", |b| {
        b.iter(|| black_box(ComparisonKernel::compile(&(&data1, &data2), Predicate::Lt).unwrap()));
    });

    c.bench_function("overhead/arrow", |b| {
        b.iter(|| black_box(arrow_ord::cmp::lt(&data1, &data2)));
    });

    let k = ComparisonKernel::compile(&(&data1, &data2), Predicate::Lt).unwrap();
    c.bench_function("overhead/kernel llvm", |b| {
        b.iter(|| black_box(k.call((&data1, &data2))));
    });

    arrow_compile_compute::cmp::lt(&data1, &data2).unwrap();
    c.bench_function("overhead/ht llvm", |b| {
        b.iter(|| black_box(arrow_compile_compute::cmp::lt(&data1, &data2).unwrap()));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
