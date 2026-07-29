use std::{collections::HashSet, sync::Arc};

use arrow_array::{Array, Int32Array, Int64Array, Int8Array, UInt64Array};
use arrow_compile_compute::SortOptions;
use arrow_ord::sort::SortColumn;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut vals = HashSet::new();
    let llvm_options = SortOptions::default();
    let arrow_options = arrow_schema::SortOptions {
        descending: llvm_options.descending,
        nulls_first: llvm_options.nulls_first,
    };

    while vals.len() < 1_000_000 {
        vals.insert(rng.u64(..));
    }
    let mut vals = vals.into_iter().collect_vec();
    rng.shuffle(&mut vals);

    let data = UInt64Array::from(vals.clone());
    let arrow_perm = arrow_ord::sort::sort_to_indices(&data, Some(arrow_options), None).unwrap();
    let our_perm = arrow_compile_compute::sort::sort_to_indices(&data, llvm_options).unwrap();
    assert_eq!(arrow_perm, our_perm);

    c.bench_function("sort::sort_to_indices(array(u64))/arrow", |b| {
        b.iter(|| {
            black_box(arrow_ord::sort::sort_to_indices(&data, Some(arrow_options), None).unwrap())
        });
    });
    c.bench_function("sort::sort_to_indices(array(u64))/llvm warm", |b| {
        b.iter(|| {
            black_box(arrow_compile_compute::sort::sort_to_indices(&data, llvm_options).unwrap())
        });
    });

    let data = UInt64Array::from(
        vals.iter()
            .map(|x| if rng.bool() { Some(*x) } else { None })
            .collect_vec(),
    );
    let arrow_perm = arrow_ord::sort::sort_to_indices(&data, Some(arrow_options), None).unwrap();
    let our_perm = arrow_compile_compute::sort::sort_to_indices(&data, llvm_options).unwrap();
    let arrow_sorted = arrow_select::take::take(&data, &arrow_perm, None).unwrap();
    let our_sorted = arrow_select::take::take(&data, &our_perm, None).unwrap();
    assert_eq!(arrow_sorted.as_ref(), our_sorted.as_ref());

    c.bench_function("sort::sort_to_indices(array(nullable u64))/arrow", |b| {
        b.iter(|| {
            black_box(arrow_ord::sort::sort_to_indices(&data, Some(arrow_options), None).unwrap())
        });
    });
    c.bench_function(
        "sort::sort_to_indices(array(nullable u64))/llvm warm",
        |b| {
            b.iter(|| {
                black_box(
                    arrow_compile_compute::sort::sort_to_indices(&data, llvm_options).unwrap(),
                )
            });
        },
    );

    let c1 = (0..1_000_000).map(|_| rng.i8(0..4)).collect_vec();
    let c2 = (0..1_000_000).map(|_| rng.i32(0..8)).collect_vec();
    let c3 = (0..1_000_000)
        .map(|_| if rng.bool() { Some(rng.i64(..)) } else { None })
        .collect_vec();
    let c1: Arc<dyn Array> = Arc::new(Int8Array::from(c1));
    let c2: Arc<dyn Array> = Arc::new(Int32Array::from(c2));
    let c3: Arc<dyn Array> = Arc::new(Int64Array::from(c3));

    let arrow_perm = arrow_ord::sort::lexsort_to_indices(
        &[
            SortColumn {
                values: c1.clone(),
                options: Some(arrow_options),
            },
            SortColumn {
                values: c2.clone(),
                options: Some(arrow_options),
            },
            SortColumn {
                values: c3.clone(),
                options: Some(arrow_options),
            },
        ],
        None,
    )
    .unwrap();
    let our_perm =
        arrow_compile_compute::sort::multicol_sort_to_indices(&[&c1, &c2, &c3], &[llvm_options; 3])
            .unwrap();
    for column in [&c1, &c2, &c3] {
        let arrow_sorted = arrow_select::take::take(column.as_ref(), &arrow_perm, None).unwrap();
        let our_sorted = arrow_select::take::take(column.as_ref(), &our_perm, None).unwrap();
        assert_eq!(arrow_sorted.as_ref(), our_sorted.as_ref());
    }

    c.bench_function(
        "sort::multicol_sort_to_indices(array(i8), array(i32), array(nullable i64))/arrow",
        |b| {
            b.iter(|| {
                arrow_ord::sort::lexsort_to_indices(
                    &[
                        SortColumn {
                            values: c1.clone(),
                            options: Some(arrow_options),
                        },
                        SortColumn {
                            values: c2.clone(),
                            options: Some(arrow_options),
                        },
                        SortColumn {
                            values: c3.clone(),
                            options: Some(arrow_options),
                        },
                    ],
                    None,
                )
                .unwrap()
            })
        },
    );
    c.bench_function(
        "sort::multicol_sort_to_indices(array(i8), array(i32), array(nullable i64))/llvm warm",
        |b| {
            b.iter(|| {
                arrow_compile_compute::sort::multicol_sort_to_indices(
                    &[&c1, &c2, &c3],
                    &[llvm_options; 3],
                )
                .unwrap()
            })
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
