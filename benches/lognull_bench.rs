use std::sync::Arc;

use arrow_array::{types::Int32Type, Array, DictionaryArray, Int32Array, RunArray};
use arrow_compile_compute::logical_nulls;
use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    {
        let values = Int32Array::from(
            (0..1000)
                .map(|_| rng.i32(0..6))
                .map(|v| if v == 5 { None } else { Some(v) })
                .collect_vec(),
        );
        let mut run_ends = Vec::new();
        let mut last_end = 0;
        for _ in 0..1000 {
            let end = last_end + rng.i32(1..10);
            run_ends.push(Some(end));
            last_end = end;
        }
        let data = RunArray::try_new(&Int32Array::from(run_ends), &values).unwrap();

        let keys = Int32Array::from((0..100_000_000).map(|_| rng.i32(0..100)).collect_vec());
        let values = Int32Array::from(
            (0..100)
                .map(|x| {
                    if x % 2 == 0 {
                        None
                    } else {
                        Some(rng.i32(0..100))
                    }
                })
                .collect_vec(),
        );

        let dict = DictionaryArray::<Int32Type>::new(keys, Arc::new(values));

        c.bench_function("dict/arrow", |b| b.iter(|| dict.logical_nulls()));
        c.bench_function("ree/arrow", |b| b.iter(|| data.logical_nulls()));

        c.bench_function("dict/llvm", |b| b.iter(|| logical_nulls(&dict).unwrap()));
        c.bench_function("ree/llvm", |b| b.iter(|| logical_nulls(&data).unwrap()));
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
