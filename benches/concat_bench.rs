use std::sync::Arc;

use arrow_array::{cast::AsArray, types::Int32Type, Array, ArrayRef, Int32Array};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::DataType;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    let mut to_concat = Vec::new();
    for _ in 0..10 {
        let vals = (0..1_000_000).map(|_| rng.i32(..)).collect_vec();
        let vals = Int32Array::from(vals);
        to_concat.push(vals);
    }
    let to_concat_refs = to_concat.iter().map(|x| x as &dyn Array).collect_vec();
    let arrow_res = arrow_select::concat::concat(&to_concat_refs).unwrap();
    let our_res = arrow_compile_compute::select::concat(&to_concat_refs).unwrap();
    assert_eq!(
        arrow_res.as_primitive::<Int32Type>(),
        our_res.as_primitive::<Int32Type>()
    );

    c.bench_function("concat i32/arrow", |b| {
        b.iter(|| black_box(arrow_select::concat::concat(&to_concat_refs).unwrap()));
    });
    c.bench_function("concat i32/llvm", |b| {
        b.iter(|| black_box(arrow_compile_compute::select::concat(&to_concat_refs).unwrap()));
    });

    let mut to_concat = Vec::new();
    for idx in 0..10 {
        if idx % 2 == 0 {
            let vals = (0..1_000_000).map(|_| rng.i32(..)).collect_vec();
            let vals = Int32Array::from(vals);
            to_concat.push(Arc::new(vals) as ArrayRef);
        } else {
            let vals = (0..1_000_000).map(|_| rng.i32(0..20)).collect_vec();
            let vals = Int32Array::from(vals);
            let vals = arrow_cast::cast(
                &vals,
                &dictionary_data_type(DataType::Int8, DataType::Int32),
            )
            .unwrap();
            to_concat.push(Arc::new(vals) as ArrayRef);
        }
    }
    let to_concat_refs = to_concat.iter().map(|x| x as &dyn Array).collect_vec();

    c.bench_function("concat cast i32/arrow", |b| {
        b.iter(|| {
            let arrs = to_concat
                .iter()
                .map(|x| arrow_cast::cast(x, &DataType::Int32).unwrap())
                .collect_vec();
            let refs = arrs.iter().map(|x| x as &dyn Array).collect_vec();
            black_box(arrow_select::concat::concat(&refs).unwrap());
        });
    });
    c.bench_function("concat cast i32/llvm", |b| {
        b.iter(|| black_box(arrow_compile_compute::select::concat(&to_concat_refs).unwrap()));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
