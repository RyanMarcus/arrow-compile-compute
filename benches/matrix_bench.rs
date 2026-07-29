//! Comparison kernels across encoding layouts, always through the public API.
//!
//! Dimensions (from the benchmarking plan):
//!   * types: i32 and i64 (floats intentionally omitted for comparisons)
//!   * layouts: plain array, dictionary(i8 keys), run-end encoded, and
//!     run-end encoded of dictionary — all four built from the SAME
//!     run-structured logical values, so rows are directly comparable
//!   * an lt/eq/gt trio on the plain i32 layout, kept as evidence that the
//!     three predicates cost the same (they lower to one compare instruction)
//!   * the known worst case: cmp::eq(dictionary, run_end_encoded)
//!
//! Every LLVM measurement is a warm public-API call (`cmp::lt` etc.): the
//! kernel is compiled and cached by the correctness assert before timing, so
//! the timed loop pays cache lookup + dispatch + execution, never compilation.
//!
//! Arrow baselines pass encoded arrays directly when stock arrow-rs supports
//! them (dictionary) and decode first when it does not (run-end encoded),
//! which is the honest way an arrow-rs user would apply these kernels.

use std::sync::Arc;

use arrow_array::{
    types::{Int32Type, Int8Type},
    Array, ArrayRef, Datum, DictionaryArray, Int32Array, Int64Array, Int8Array, RunArray,
};
use arrow_schema::DataType;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

const N: usize = 1_000_000;
const CARDINALITY: i32 = 100;

/// Run-structured logical values: runs of 1..40 equal values drawn from a
/// small domain, so dictionary and run-end encodings are both meaningful.
fn generate_logical_values(rng: &mut fastrand::Rng) -> Vec<i32> {
    let mut values = Vec::with_capacity(N + 40);
    while values.len() < N {
        let value = rng.i32(0..CARDINALITY);
        for _ in 0..rng.usize(1..40) {
            values.push(value);
        }
    }
    values.truncate(N);
    values
}

/// Compress consecutive equal values into (run_ends, run_values).
fn to_runs(values: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut run_ends = Vec::new();
    let mut run_values = Vec::new();
    for (index, &value) in values.iter().enumerate() {
        if run_values.last() == Some(&value) {
            *run_ends.last_mut().unwrap() = (index + 1) as i32;
        } else {
            run_values.push(value);
            run_ends.push((index + 1) as i32);
        }
    }
    (run_ends, run_values)
}

fn assert_matches_arrow(ours: &arrow_array::BooleanArray, theirs: &arrow_array::BooleanArray) {
    assert_eq!(ours.len(), theirs.len());
    assert_eq!(ours.true_count(), theirs.true_count());
}

/// Bench one (layout, scalar) comparison: warm public-API lt vs the arrow
/// baseline closure.
fn bench_lt_layout(
    c: &mut Criterion,
    name: &str,
    encoded: &dyn Datum,
    scalar: &dyn Datum,
    arrow_baseline: impl Fn() -> arrow_array::BooleanArray,
) {
    let ours = arrow_compile_compute::cmp::lt(encoded, scalar).unwrap();
    assert_matches_arrow(&ours, &arrow_baseline());

    c.bench_function(&format!("{name}/llvm warm"), |b| {
        b.iter(|| black_box(arrow_compile_compute::cmp::lt(encoded, scalar).unwrap()))
    });
    c.bench_function(&format!("{name}/arrow"), |b| {
        b.iter(|| black_box(arrow_baseline()))
    });
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    let logical = generate_logical_values(&mut rng);
    let (run_ends, run_values) = to_runs(&logical);

    // ---- i32 layouts, all encoding the same logical values ------------------
    let plain_i32 = Int32Array::from(logical.clone());
    let dict_i32 = DictionaryArray::<Int8Type>::new(
        Int8Array::from(logical.iter().map(|&v| v as i8).collect_vec()),
        Arc::new(Int32Array::from((0..CARDINALITY).collect_vec())),
    );
    let ree_i32 = RunArray::<Int32Type>::try_new(
        &Int32Array::from(run_ends.clone()),
        &Int32Array::from(run_values.clone()),
    )
    .unwrap();
    let ree_dict_i32 = {
        let dict_values: ArrayRef = Arc::new(DictionaryArray::<Int8Type>::new(
            Int8Array::from(run_values.iter().map(|&v| v as i8).collect_vec()),
            Arc::new(Int32Array::from((0..CARDINALITY).collect_vec())),
        ));
        RunArray::<Int32Type>::try_new(&Int32Array::from(run_ends.clone()), &dict_values).unwrap()
    };
    let scalar_i32 = Int32Array::new_scalar(CARDINALITY / 2);

    // ---- i64 layouts of the same logical values ------------------------------
    let plain_i64 = Int64Array::from(logical.iter().map(|&v| v as i64).collect_vec());
    let dict_i64 = DictionaryArray::<Int8Type>::new(
        Int8Array::from(logical.iter().map(|&v| v as i8).collect_vec()),
        Arc::new(Int64Array::from((0..CARDINALITY as i64).collect_vec())),
    );
    let ree_i64 = RunArray::<Int32Type>::try_new(
        &Int32Array::from(run_ends.clone()),
        &Int64Array::from(run_values.iter().map(|&v| v as i64).collect_vec()),
    )
    .unwrap();
    let ree_dict_i64 = {
        let dict_values: ArrayRef = Arc::new(DictionaryArray::<Int8Type>::new(
            Int8Array::from(run_values.iter().map(|&v| v as i8).collect_vec()),
            Arc::new(Int64Array::from((0..CARDINALITY as i64).collect_vec())),
        ));
        RunArray::<Int32Type>::try_new(&Int32Array::from(run_ends.clone()), &dict_values).unwrap()
    };
    let scalar_i64 = Int64Array::new_scalar((CARDINALITY / 2) as i64);

    // ---- lt/eq/gt trio on plain i32: the three predicates should cost the
    // same cycle count; keeping all three documents that claim.
    {
        let (d, s): (&dyn Datum, &dyn Datum) = (&plain_i32, &scalar_i32);
        for (op_name, ours, arrow_op) in [
            (
                "lt",
                arrow_compile_compute::cmp::lt as fn(&dyn Datum, &dyn Datum) -> _,
                arrow_ord::cmp::lt as fn(&dyn Datum, &dyn Datum) -> _,
            ),
            ("eq", arrow_compile_compute::cmp::eq, arrow_ord::cmp::eq),
            ("gt", arrow_compile_compute::cmp::gt, arrow_ord::cmp::gt),
        ] {
            assert_matches_arrow(&ours(d, s).unwrap(), &arrow_op(d, s).unwrap());
            let name = format!("cmp::{op_name}(array(i32), scalar(i32)) 1m rows");
            c.bench_function(&format!("{name}/llvm warm"), |b| {
                b.iter(|| black_box(ours(d, s).unwrap()))
            });
            c.bench_function(&format!("{name}/arrow"), |b| {
                b.iter(|| black_box(arrow_op(d, s).unwrap()))
            });
        }
    }

    // ---- lt across encodings, i32 -------------------------------------------
    // Dictionary: stock arrow compares dictionaries directly.
    bench_lt_layout(
        c,
        "cmp::lt(dictionary(i8, i32), scalar(i32)) 1m rows",
        &dict_i32,
        &scalar_i32,
        || arrow_ord::cmp::lt(&dict_i32, &scalar_i32).unwrap(),
    );
    // Run-end: stock arrow has no run-end comparison; it must decode first.
    bench_lt_layout(
        c,
        "cmp::lt(run_end_encoded(i32, i32), scalar(i32)) 1m rows",
        &ree_i32,
        &scalar_i32,
        || {
            let decoded = arrow_cast::cast(&ree_i32, &DataType::Int32).unwrap();
            arrow_ord::cmp::lt(&decoded, &scalar_i32).unwrap()
        },
    );
    bench_lt_layout(
        c,
        "cmp::lt(run_end_encoded(i32, dictionary(i8, i32)), scalar(i32)) 1m rows",
        &ree_dict_i32,
        &scalar_i32,
        || {
            let decoded = arrow_cast::cast(&ree_dict_i32, &DataType::Int32).unwrap();
            arrow_ord::cmp::lt(&decoded, &scalar_i32).unwrap()
        },
    );

    // ---- lt across encodings, i64 -------------------------------------------
    bench_lt_layout(
        c,
        "cmp::lt(array(i64), scalar(i64)) 1m rows",
        &plain_i64,
        &scalar_i64,
        || arrow_ord::cmp::lt(&plain_i64, &scalar_i64).unwrap(),
    );
    bench_lt_layout(
        c,
        "cmp::lt(dictionary(i8, i64), scalar(i64)) 1m rows",
        &dict_i64,
        &scalar_i64,
        || arrow_ord::cmp::lt(&dict_i64, &scalar_i64).unwrap(),
    );
    bench_lt_layout(
        c,
        "cmp::lt(run_end_encoded(i32, i64), scalar(i64)) 1m rows",
        &ree_i64,
        &scalar_i64,
        || {
            let decoded = arrow_cast::cast(&ree_i64, &DataType::Int64).unwrap();
            arrow_ord::cmp::lt(&decoded, &scalar_i64).unwrap()
        },
    );
    bench_lt_layout(
        c,
        "cmp::lt(run_end_encoded(i32, dictionary(i8, i64)), scalar(i64)) 1m rows",
        &ree_dict_i64,
        &scalar_i64,
        || {
            let decoded = arrow_cast::cast(&ree_dict_i64, &DataType::Int64).unwrap();
            arrow_ord::cmp::lt(&decoded, &scalar_i64).unwrap()
        },
    );

    // ---- known worst case: mixed encodings, array vs array ------------------
    {
        let (dd, dr): (&dyn Datum, &dyn Datum) = (&dict_i32, &ree_i32);
        let arrow_baseline = || {
            let dict_decoded = arrow_cast::cast(&dict_i32, &DataType::Int32).unwrap();
            let ree_decoded = arrow_cast::cast(&ree_i32, &DataType::Int32).unwrap();
            arrow_ord::cmp::eq(&dict_decoded, &ree_decoded).unwrap()
        };
        let ours = arrow_compile_compute::cmp::eq(dd, dr).unwrap();
        assert_matches_arrow(&ours, &arrow_baseline());

        let name = "cmp::eq(dictionary(i8, i32), run_end_encoded(i32, i32)) 1m rows";
        c.bench_function(&format!("{name}/llvm warm"), |b| {
            b.iter(|| black_box(arrow_compile_compute::cmp::eq(dd, dr).unwrap()))
        });
        c.bench_function(&format!("{name}/arrow"), |b| {
            b.iter(|| black_box(arrow_baseline()))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
