//! Input-combination matrix, with the JIT cost split into compile vs direct call.
//!
//! For every (operator, input layout) we emit three Criterion benchmarks:
//!   * `<name>/llvm compile` — `Kernel::compile`, the one-time LLVM IR-gen + JIT
//!     cost (paid once per kernel shape, then cached forever).
//!   * `<name>/llvm direct`  — `Kernel::call` on an already-compiled kernel: the
//!     warm, steady-state cost.
//!   * `<name>/arrow`        — the equivalent stock arrow-rs kernel.
//!
//! Benchmark names use this crate's public function names plus argument types, e.g.
//! `cmp::lt(dictionary(i32, i32), scalar(i32))`.
//!
//! For dictionary inputs the JIT reads the encoding directly; the arrow path must
//! first decode (`arrow_cast::cast` to a primitive), which is the honest arrow way
//! to apply these kernels to an encoded column.

use std::sync::Arc;

use arrow_array::{
    cast::AsArray, types::Int32Type, Array, Datum, DictionaryArray, Int32Array, StringArray,
};
use arrow_compile_compute::{
    arith::{DSLUnaryOp, UnaryOpKernel},
    cmp::ComparisonKernel,
    compute::{ReductionKernel, ReductionKernelType},
    Kernel, Predicate,
};
use arrow_schema::DataType;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

const N: i32 = 1_000_000;

/// Assert two numeric results are logically equal after normalizing to Int32
/// (handles a dictionary result vs a primitive result).
fn assert_i32_eq(ours: &dyn Array, theirs: &dyn Array) {
    let a = arrow_cast::cast(ours, &DataType::Int32).unwrap();
    let b = arrow_cast::cast(theirs, &DataType::Int32).unwrap();
    assert_eq!(a.as_primitive::<Int32Type>(), b.as_primitive::<Int32Type>());
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    let a = Int32Array::from((0..N).map(|_| rng.i32(..)).collect_vec());
    let b = Int32Array::from((0..N).map(|_| rng.i32(..)).collect_vec());
    let scalar = Int32Array::new_scalar(0);

    let keys = Int32Array::from((0..N).map(|_| rng.i32(0..128)).collect_vec());
    let dict_vals = Int32Array::from((0..128).map(|_| rng.i32(..)).collect_vec());
    let dict = DictionaryArray::<Int32Type>::new(keys, Arc::new(dict_vals));

    let sa = StringArray::from((0..N).map(|_| format!("{:08x}", rng.u32(..))).collect_vec());
    let sb = StringArray::from((0..N).map(|_| format!("{:08x}", rng.u32(..))).collect_vec());

    // ---- cmp::lt -----------------------------------------------------------
    {
        let (da, db): (&dyn Datum, &dyn Datum) = (&a, &b);
        assert_eq!(
            arrow_compile_compute::cmp::lt(da, db).unwrap(),
            arrow_ord::cmp::lt(da, db).unwrap()
        );
        c.bench_function(
            "cmp::lt(array(i32), array(i32)) 1m rows/llvm compile",
            |be| {
                be.iter(|| black_box(ComparisonKernel::compile(&(da, db), Predicate::Lt).unwrap()))
            },
        );
        let k = ComparisonKernel::compile(&(da, db), Predicate::Lt).unwrap();
        c.bench_function(
            "cmp::lt(array(i32), array(i32)) 1m rows/llvm direct",
            |be| be.iter(|| black_box(k.call((da, db)).unwrap())),
        );
        c.bench_function("cmp::lt(array(i32), array(i32)) 1m rows/arrow", |be| {
            be.iter(|| black_box(arrow_ord::cmp::lt(da, db).unwrap()))
        });
    }
    {
        let (da, ds): (&dyn Datum, &dyn Datum) = (&a, &scalar);
        assert_eq!(
            arrow_compile_compute::cmp::lt(da, ds).unwrap(),
            arrow_ord::cmp::lt(da, ds).unwrap()
        );
        c.bench_function(
            "cmp::lt(array(i32), scalar(i32)) 1m rows/llvm compile",
            |be| {
                be.iter(|| black_box(ComparisonKernel::compile(&(da, ds), Predicate::Lt).unwrap()))
            },
        );
        let k = ComparisonKernel::compile(&(da, ds), Predicate::Lt).unwrap();
        c.bench_function(
            "cmp::lt(array(i32), scalar(i32)) 1m rows/llvm direct",
            |be| be.iter(|| black_box(k.call((da, ds)).unwrap())),
        );
        c.bench_function("cmp::lt(array(i32), scalar(i32)) 1m rows/arrow", |be| {
            be.iter(|| black_box(arrow_ord::cmp::lt(da, ds).unwrap()))
        });
    }
    {
        let (dd, ds): (&dyn Datum, &dyn Datum) = (&dict, &scalar);
        assert_eq!(
            arrow_compile_compute::cmp::lt(dd, ds).unwrap(),
            arrow_ord::cmp::lt(dd, ds).unwrap()
        );
        c.bench_function(
            "cmp::lt(dictionary(i32, i32), scalar(i32)) 1m rows/llvm compile",
            |be| {
                be.iter(|| black_box(ComparisonKernel::compile(&(dd, ds), Predicate::Lt).unwrap()))
            },
        );
        let k = ComparisonKernel::compile(&(dd, ds), Predicate::Lt).unwrap();
        c.bench_function(
            "cmp::lt(dictionary(i32, i32), scalar(i32)) 1m rows/llvm direct",
            |be| be.iter(|| black_box(k.call((dd, ds)).unwrap())),
        );
        c.bench_function(
            "cmp::lt(dictionary(i32, i32), scalar(i32)) 1m rows/arrow",
            |be| be.iter(|| black_box(arrow_ord::cmp::lt(dd, ds).unwrap())),
        );
    }
    {
        let (dsa, dsb): (&dyn Datum, &dyn Datum) = (&sa, &sb);
        assert_eq!(
            arrow_compile_compute::cmp::lt(dsa, dsb).unwrap(),
            arrow_ord::cmp::lt(dsa, dsb).unwrap()
        );
        c.bench_function(
            "cmp::lt(array(utf8), array(utf8)) 1m rows/llvm compile",
            |be| {
                be.iter(|| {
                    black_box(ComparisonKernel::compile(&(dsa, dsb), Predicate::Lt).unwrap())
                })
            },
        );
        let k = ComparisonKernel::compile(&(dsa, dsb), Predicate::Lt).unwrap();
        c.bench_function(
            "cmp::lt(array(utf8), array(utf8)) 1m rows/llvm direct",
            |be| be.iter(|| black_box(k.call((dsa, dsb)).unwrap())),
        );
        c.bench_function("cmp::lt(array(utf8), array(utf8)) 1m rows/arrow", |be| {
            be.iter(|| black_box(arrow_ord::cmp::lt(dsa, dsb).unwrap()))
        });
    }

    // ---- numeric::neg_wrapping ---------------------------------------------
    {
        let d: &dyn Datum = &a;
        assert_i32_eq(
            &arrow_compile_compute::arith::neg_wrapping(d).unwrap(),
            &arrow_arith::numeric::neg_wrapping(&a).unwrap(),
        );
        c.bench_function(
            "arith::neg_wrapping(array(i32)) 1m rows/llvm compile",
            |be| be.iter(|| black_box(UnaryOpKernel::compile(&d, DSLUnaryOp::Neg).unwrap())),
        );
        let k = UnaryOpKernel::compile(&d, DSLUnaryOp::Neg).unwrap();
        c.bench_function(
            "arith::neg_wrapping(array(i32)) 1m rows/llvm direct",
            |be| be.iter(|| black_box(k.call(d).unwrap())),
        );
        c.bench_function("arith::neg_wrapping(array(i32)) 1m rows/arrow", |be| {
            be.iter(|| black_box(arrow_arith::numeric::neg_wrapping(&a).unwrap()))
        });
    }
    {
        // JIT negates the dictionary directly; arrow must decode (cast) first.
        let d: &dyn Datum = &dict;
        let decoded = arrow_cast::cast(&dict, &DataType::Int32).unwrap();
        assert_i32_eq(
            &arrow_compile_compute::arith::neg_wrapping(d).unwrap(),
            &arrow_arith::numeric::neg_wrapping(&decoded).unwrap(),
        );
        c.bench_function(
            "arith::neg_wrapping(dictionary(i32, i32)) 1m rows/llvm compile",
            |be| be.iter(|| black_box(UnaryOpKernel::compile(&d, DSLUnaryOp::Neg).unwrap())),
        );
        let k = UnaryOpKernel::compile(&d, DSLUnaryOp::Neg).unwrap();
        c.bench_function(
            "arith::neg_wrapping(dictionary(i32, i32)) 1m rows/llvm direct",
            |be| be.iter(|| black_box(k.call(d).unwrap())),
        );
        c.bench_function(
            "arith::neg_wrapping(dictionary(i32, i32)) 1m rows/arrow",
            |be| {
                be.iter(|| {
                    let decoded = arrow_cast::cast(&dict, &DataType::Int32).unwrap();
                    black_box(arrow_arith::numeric::neg_wrapping(&decoded).unwrap())
                })
            },
        );
    }

    // ---- aggregate::sum ----------------------------------------------------
    {
        let d: &dyn Datum = &a;
        assert_i32_eq(
            &arrow_compile_compute::compute::sum(d).unwrap(),
            &Int32Array::from(vec![arrow_arith::aggregate::sum(&a).unwrap()]),
        );
        c.bench_function("compute::sum(array(i32)) 1m rows/llvm compile", |be| {
            be.iter(|| black_box(ReductionKernel::compile(&d, ReductionKernelType::Sum).unwrap()))
        });
        let k = ReductionKernel::compile(&d, ReductionKernelType::Sum).unwrap();
        c.bench_function("compute::sum(array(i32)) 1m rows/llvm direct", |be| {
            be.iter(|| black_box(k.call(d).unwrap()))
        });
        c.bench_function("compute::sum(array(i32)) 1m rows/arrow", |be| {
            be.iter(|| black_box(arrow_arith::aggregate::sum(&a).unwrap()))
        });
    }
    {
        // JIT sums the dictionary directly; arrow must decode (cast) first.
        let d: &dyn Datum = &dict;
        let decoded = arrow_cast::cast(&dict, &DataType::Int32).unwrap();
        let decoded_i32 = decoded.as_primitive::<Int32Type>();
        assert_i32_eq(
            &arrow_compile_compute::compute::sum(d).unwrap(),
            &Int32Array::from(vec![arrow_arith::aggregate::sum(decoded_i32).unwrap()]),
        );
        c.bench_function(
            "compute::sum(dictionary(i32, i32)) 1m rows/llvm compile",
            |be| {
                be.iter(|| {
                    black_box(ReductionKernel::compile(&d, ReductionKernelType::Sum).unwrap())
                })
            },
        );
        let k = ReductionKernel::compile(&d, ReductionKernelType::Sum).unwrap();
        c.bench_function(
            "compute::sum(dictionary(i32, i32)) 1m rows/llvm direct",
            |be| be.iter(|| black_box(k.call(d).unwrap())),
        );
        c.bench_function("compute::sum(dictionary(i32, i32)) 1m rows/arrow", |be| {
            be.iter(|| {
                let decoded = arrow_cast::cast(&dict, &DataType::Int32).unwrap();
                let decoded = decoded.as_primitive::<Int32Type>();
                black_box(arrow_arith::aggregate::sum(decoded).unwrap())
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
