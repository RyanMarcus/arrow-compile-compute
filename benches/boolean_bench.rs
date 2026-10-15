//! Boolean kernels, always through the public API.
//!
//! Coverage:
//!   * `boolean::not` — JIT `NotKernel`, a single `bit_not` in the DSL loop.
//!     Benchmarked on plain, dictionary, and run-end encoded boolean input:
//!     `normalized_base_type` sees through both encodings, so the JIT negates
//!     without materializing while stock arrow must decode first.
//!   * `boolean::is_null` / `boolean::is_not_null` — validity-only ops.
//!     `is_not_null` is a buffer clone (no JIT); `is_null` is that plus a
//!     `not` kernel call. Run on a nullable primitive array, since on a
//!     non-null input both sides collapse to allocating a constant bitmap.
//!
//! Baselines are the stock `arrow_arith::boolean` kernels. Those take a
//! `&BooleanArray`, so the encoded-layout baselines decode inside the timed
//! closure — that is what an arrow-rs user would have to write.

use std::sync::Arc;

use arrow_array::{
    types::{Int32Type, Int8Type},
    BooleanArray, DictionaryArray, Int32Array, Int8Array, RunArray,
};
use arrow_schema::DataType;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

const N: usize = 10_000_000;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    // ---- plain boolean, no nulls -------------------------------------------
    {
        let data = BooleanArray::from((0..N).map(|_| rng.bool()).collect_vec());

        let ours = arrow_compile_compute::boolean::not(&data).unwrap();
        let theirs = arrow_arith::boolean::not(&data).unwrap();
        assert_eq!(ours, theirs);

        c.bench_function("boolean::not(array(bool)) 10m rows/llvm warm", |b| {
            b.iter(|| arrow_compile_compute::boolean::not(black_box(&data)).unwrap())
        });
        c.bench_function("boolean::not(array(bool)) 10m rows/arrow", |b| {
            b.iter(|| arrow_arith::boolean::not(black_box(&data)).unwrap())
        });
    }

    // ---- plain boolean, 10% null: exercises the validity copy in `call` -----
    {
        let data = BooleanArray::from(
            (0..N)
                .map(|_| (rng.u32(0..100) >= 10).then(|| rng.bool()))
                .collect_vec(),
        );

        let ours = arrow_compile_compute::boolean::not(&data).unwrap();
        let theirs = arrow_arith::boolean::not(&data).unwrap();
        assert_eq!(ours, theirs);

        c.bench_function(
            "boolean::not(array(bool)) 10m rows, 10% nulls/llvm warm",
            |b| b.iter(|| arrow_compile_compute::boolean::not(black_box(&data)).unwrap()),
        );
        c.bench_function("boolean::not(array(bool)) 10m rows, 10% nulls/arrow", |b| {
            b.iter(|| arrow_arith::boolean::not(black_box(&data)).unwrap())
        });
    }

    // ---- dictionary-encoded boolean ----------------------------------------
    // Stock arrow has no dictionary `not`, so the baseline decodes first.
    {
        let keys = Int8Array::from((0..N).map(|_| rng.i8(0..2)).collect_vec());
        let values = BooleanArray::from(vec![false, true]);
        let data = DictionaryArray::<Int8Type>::new(keys, Arc::new(values));

        let ours = arrow_compile_compute::boolean::not(&data).unwrap();
        let decoded = arrow_cast::cast(&data, &DataType::Boolean).unwrap();
        let theirs = arrow_arith::boolean::not(decoded.as_any().downcast_ref().unwrap()).unwrap();
        assert_eq!(ours, theirs);

        c.bench_function(
            "boolean::not(dictionary(i8, bool)) 10m rows/llvm warm",
            |b| b.iter(|| arrow_compile_compute::boolean::not(black_box(&data)).unwrap()),
        );
        c.bench_function("boolean::not(dictionary(i8, bool)) 10m rows/arrow", |b| {
            b.iter(|| {
                let decoded = arrow_cast::cast(black_box(&data), &DataType::Boolean).unwrap();
                arrow_arith::boolean::not(decoded.as_any().downcast_ref().unwrap()).unwrap()
            })
        });
    }

    // ---- run-end encoded boolean -------------------------------------------
    // Runs of 1..40 with the value flipping between runs, so no two adjacent
    // runs are equal (a well-formed REE array).
    {
        let mut run_ends = Vec::new();
        let mut run_values = Vec::new();
        let mut value = true;
        let mut length = 0usize;
        while length < N {
            length = (length + rng.usize(1..40)).min(N);
            run_ends.push(length as i32);
            run_values.push(value);
            value = !value;
        }
        let data = RunArray::<Int32Type>::try_new(
            &Int32Array::from(run_ends),
            &BooleanArray::from(run_values),
        )
        .unwrap();

        let ours = arrow_compile_compute::boolean::not(&data).unwrap();
        let decoded = arrow_cast::cast(&data, &DataType::Boolean).unwrap();
        let theirs = arrow_arith::boolean::not(decoded.as_any().downcast_ref().unwrap()).unwrap();
        assert_eq!(ours, theirs);

        c.bench_function(
            "boolean::not(run_end_encoded(i32, bool)) 10m rows/llvm warm",
            |b| b.iter(|| arrow_compile_compute::boolean::not(black_box(&data)).unwrap()),
        );
        c.bench_function(
            "boolean::not(run_end_encoded(i32, bool)) 10m rows/arrow",
            |b| {
                b.iter(|| {
                    let decoded = arrow_cast::cast(black_box(&data), &DataType::Boolean).unwrap();
                    arrow_arith::boolean::not(decoded.as_any().downcast_ref().unwrap()).unwrap()
                })
            },
        );
    }

    // ---- validity-only kernels ---------------------------------------------
    {
        let data = Int32Array::from(
            (0..N)
                .map(|_| (rng.u32(0..100) >= 10).then(|| rng.i32(..)))
                .collect_vec(),
        );

        let ours = arrow_compile_compute::boolean::is_null(&data).unwrap();
        let theirs = arrow_arith::boolean::is_null(&data).unwrap();
        assert_eq!(ours, theirs);

        c.bench_function(
            "boolean::is_null(array(i32)) 10m rows, 10% nulls/llvm warm",
            |b| b.iter(|| arrow_compile_compute::boolean::is_null(black_box(&data)).unwrap()),
        );
        c.bench_function(
            "boolean::is_null(array(i32)) 10m rows, 10% nulls/arrow",
            |b| b.iter(|| arrow_arith::boolean::is_null(black_box(&data)).unwrap()),
        );

        let ours = arrow_compile_compute::boolean::is_not_null(&data).unwrap();
        let theirs = arrow_arith::boolean::is_not_null(&data).unwrap();
        assert_eq!(ours, theirs);

        c.bench_function(
            "boolean::is_not_null(array(i32)) 10m rows, 10% nulls/llvm warm",
            |b| b.iter(|| arrow_compile_compute::boolean::is_not_null(black_box(&data)).unwrap()),
        );
        c.bench_function(
            "boolean::is_not_null(array(i32)) 10m rows, 10% nulls/arrow",
            |b| b.iter(|| arrow_arith::boolean::is_not_null(black_box(&data)).unwrap()),
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
