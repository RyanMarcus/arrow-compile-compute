use arrow_array::{cast::AsArray, types::Int32Type, Int32Array};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

// arrow-rs `aggregate::{sum, product, min, max}` are ungrouped reductions (whole
// array -> scalar). Their ungrouped counterparts here live in reduction.rs, exposed
// as `compute::{sum, product, min, max}`. Those are the valid oracle matches; the
// `aggregate2` GROUP BY kernels are a different operation (arrow has no group-by)
// and are deliberately not benchmarked here.
//
// Integer inputs are used throughout: integer add/mul are associative mod 2^N, so
// the crate's reduction and arrow's fold produce identical wrapped values, letting
// the correctness asserts be exact. `min`/`max` also compare exactly (ordering).
pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    // sum (i32). compute::sum keeps the input type and wraps, matching
    // arrow_arith::aggregate::sum.
    {
        let data = Int32Array::from((0..10_000_000).map(|_| rng.i32(..)).collect_vec());

        let our = arrow_compile_compute::compute::sum(&data).unwrap();
        let our = our.as_primitive::<Int32Type>().value(0);
        let arr = arrow_arith::aggregate::sum(&data).unwrap();
        assert_eq!(our, arr);

        c.bench_function("compute::sum(array(i32)) 10m rows/llvm warm", |b| {
            b.iter(|| black_box(arrow_compile_compute::compute::sum(black_box(&data)).unwrap()))
        });
        c.bench_function("compute::sum(array(i32)) 10m rows/arrow", |b| {
            b.iter(|| black_box(arrow_arith::aggregate::sum(black_box(&data))))
        });
    }

    // product (i32). +/-1 values so the product never overflows and matches
    // exactly regardless of overflow policy; per-element multiply cost is the same.
    {
        let data = Int32Array::from(
            (0..10_000_000)
                .map(|_| if rng.bool() { 1i32 } else { -1 })
                .collect_vec(),
        );

        let our = arrow_compile_compute::compute::product(&data).unwrap();
        let our = our.as_primitive::<Int32Type>().value(0);
        let arr = arrow_arith::aggregate::product(&data).unwrap();
        assert_eq!(our, arr);

        c.bench_function("compute::product(array(i32)) 10m rows/llvm warm", |b| {
            b.iter(|| black_box(arrow_compile_compute::compute::product(black_box(&data)).unwrap()))
        });
        c.bench_function("compute::product(array(i32)) 10m rows/arrow", |b| {
            b.iter(|| black_box(arrow_arith::aggregate::product(black_box(&data))))
        });
    }

    // min / max (i32).
    {
        let data = Int32Array::from((0..10_000_000).map(|_| rng.i32(..)).collect_vec());

        let our_min = arrow_compile_compute::compute::min(&data).unwrap();
        let our_min = our_min.as_primitive::<Int32Type>().value(0);
        assert_eq!(our_min, arrow_arith::aggregate::min(&data).unwrap());

        let our_max = arrow_compile_compute::compute::max(&data).unwrap();
        let our_max = our_max.as_primitive::<Int32Type>().value(0);
        assert_eq!(our_max, arrow_arith::aggregate::max(&data).unwrap());

        c.bench_function("compute::min(array(i32)) 10m rows/llvm warm", |b| {
            b.iter(|| black_box(arrow_compile_compute::compute::min(black_box(&data)).unwrap()))
        });
        c.bench_function("compute::min(array(i32)) 10m rows/arrow", |b| {
            b.iter(|| black_box(arrow_arith::aggregate::min(black_box(&data))))
        });

        c.bench_function("compute::max(array(i32)) 10m rows/llvm warm", |b| {
            b.iter(|| black_box(arrow_compile_compute::compute::max(black_box(&data)).unwrap()))
        });
        c.bench_function("compute::max(array(i32)) 10m rows/arrow", |b| {
            b.iter(|| black_box(arrow_arith::aggregate::max(black_box(&data))))
        });
    }

    // sum over a dictionary layout: the JIT folds through the encoding; stock
    // arrow has no dictionary sum, so it must decode (cast) first.
    {
        use arrow_array::{types::Int8Type, DictionaryArray, Int8Array};
        use arrow_schema::DataType;
        use std::sync::Arc;

        let keys = Int8Array::from((0..10_000_000).map(|_| rng.i8(0..100)).collect_vec());
        let values = Int32Array::from((0..100).map(|_| rng.i32(-1000..1000)).collect_vec());
        let data = DictionaryArray::<Int8Type>::new(keys, Arc::new(values));

        let ours = arrow_compile_compute::compute::sum(&data).unwrap();
        let ours = ours.as_primitive::<Int32Type>().value(0);
        let decoded = arrow_cast::cast(&data, &DataType::Int32).unwrap();
        let theirs = arrow_arith::aggregate::sum(decoded.as_primitive::<Int32Type>()).unwrap();
        assert_eq!(ours, theirs);

        c.bench_function("compute::sum(dictionary(i8, i32)) 10m rows/llvm warm", |b| {
            b.iter(|| black_box(arrow_compile_compute::compute::sum(black_box(&data)).unwrap()))
        });
        c.bench_function("compute::sum(dictionary(i8, i32)) 10m rows/arrow", |b| {
            b.iter(|| {
                let decoded = arrow_cast::cast(black_box(&data), &DataType::Int32).unwrap();
                black_box(arrow_arith::aggregate::sum(decoded.as_primitive::<Int32Type>()))
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
