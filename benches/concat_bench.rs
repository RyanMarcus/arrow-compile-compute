use std::sync::Arc;

use arrow_array::{
    cast::AsArray, types::Int32Type, Array, ArrayRef, BooleanArray, FixedSizeListArray, Int32Array,
};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::{DataType, Field};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;

fn fixed_size_bool_array(values: Vec<bool>, list_size: i32) -> FixedSizeListArray {
    FixedSizeListArray::try_new(
        Arc::new(Field::new_list_field(DataType::Boolean, false)),
        list_size,
        Arc::new(BooleanArray::from(values)),
        None,
    )
    .unwrap()
}

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

    let list_size = 8;
    let rows_per_array = 1_000_000;
    let mut to_concat = Vec::new();
    for _ in 0..2 {
        let vals = (0..rows_per_array * list_size)
            .map(|_| rng.bool())
            .collect_vec();
        to_concat.push(fixed_size_bool_array(vals, list_size as i32));
    }
    let to_concat_refs = to_concat.iter().map(|x| x as &dyn Array).collect_vec();
    let arrow_res = arrow_select::concat::concat(&to_concat_refs).unwrap();
    let our_res = arrow_compile_compute::select::concat(&to_concat_refs).unwrap();
    assert_eq!(
        arrow_res.as_fixed_size_list().values().as_boolean(),
        our_res.as_fixed_size_list().values().as_boolean()
    );

    c.bench_function("concat fixed bool[8]/arrow", |b| {
        b.iter(|| black_box(arrow_select::concat::concat(&to_concat_refs).unwrap()));
    });
    c.bench_function("concat fixed bool[8]/llvm", |b| {
        b.iter(|| black_box(arrow_compile_compute::select::concat(&to_concat_refs).unwrap()));
    });

    let bools = (0..to_concat
        .iter()
        .map(|arr| arr.values().len())
        .sum::<usize>())
        .map(|idx| idx % 3 == 0)
        .collect_vec();
    c.bench_function("boolean array from Vec<bool>", |b| {
        b.iter_batched(
            || black_box(bools.clone()),
            |bools| black_box(BooleanArray::from(bools)),
            BatchSize::LargeInput,
        );
    });

    let bytes = bools
        .iter()
        .map(|value| if *value { 1_u8 } else { 0_u8 })
        .collect_vec();
    c.bench_function("fixed bool writer byte materialize", |b| {
        b.iter(|| {
            let bools = black_box(&bytes)
                .iter()
                .map(|value| *value != 0)
                .collect_vec();
            black_box(BooleanArray::from(bools));
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
