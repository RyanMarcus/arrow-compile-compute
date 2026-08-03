use std::sync::Arc;

use arrow_array::{
    cast::AsArray, types::Int32Type, types::Int8Type, DictionaryArray, Float64Array, Int32Array,
    Int8Array,
};
use arrow_schema::DataType;
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

        c.bench_function("arith::neg_wrapping(array(i32)) 10m rows/llvm warm", |b| {
            b.iter(|| arrow_compile_compute::arith::neg_wrapping(black_box(&data)).unwrap())
        });

        c.bench_function("arith::neg_wrapping(array(i32)) 10m rows/arrow", |b| {
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

        c.bench_function("arith::neg_wrapping(array(f64)) 10m rows/llvm warm", |b| {
            b.iter(|| arrow_compile_compute::arith::neg_wrapping(black_box(&data)).unwrap())
        });

        c.bench_function("arith::neg_wrapping(array(f64)) 10m rows/arrow", |b| {
            b.iter(|| arrow_arith::numeric::neg_wrapping(black_box(&data)).unwrap())
        });
    }

    // Dictionary layout: the JIT negates through the encoding; stock arrow has
    // no dictionary negation, so it must decode (cast) first.
    {
        let keys = Int8Array::from((0..10_000_000).map(|_| rng.i8(0..100)).collect_vec());
        let values = Int32Array::from((0..100).map(|_| rng.i32(..)).collect_vec());
        let data = DictionaryArray::<Int8Type>::new(keys, Arc::new(values));

        let ours = arrow_compile_compute::arith::neg_wrapping(&data).unwrap();
        let ours = arrow_cast::cast(&ours, &DataType::Int32).unwrap();
        let decoded = arrow_cast::cast(&data, &DataType::Int32).unwrap();
        let theirs = arrow_arith::numeric::neg_wrapping(&decoded).unwrap();
        assert_eq!(
            ours.as_primitive::<Int32Type>(),
            theirs.as_primitive::<Int32Type>()
        );

        c.bench_function(
            "arith::neg_wrapping(dictionary(i8, i32)) 10m rows/llvm warm",
            |b| b.iter(|| arrow_compile_compute::arith::neg_wrapping(black_box(&data)).unwrap()),
        );

        c.bench_function("arith::neg_wrapping(dictionary(i8, i32)) 10m rows/arrow", |b| {
            b.iter(|| {
                let decoded = arrow_cast::cast(black_box(&data), &DataType::Int32).unwrap();
                arrow_arith::numeric::neg_wrapping(&decoded).unwrap()
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
