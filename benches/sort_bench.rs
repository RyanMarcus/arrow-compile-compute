use std::{collections::HashSet, sync::Arc};

use arrow_array::{Array, Int32Array, Int64Array, Int8Array, UInt64Array};
use arrow_compile_compute::SortOptions;
use arrow_ord::sort::SortColumn;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut vals = HashSet::new();

    while vals.len() < 1_000_000 {
        vals.insert(rng.u64(..));
    }
    let mut vals = vals.into_iter().collect_vec();
    rng.shuffle(&mut vals);

    let data = UInt64Array::from(vals.clone());
    let arrow_perm = arrow_ord::sort::sort_to_indices(&data, None, None).unwrap();
    let our_perm =
        arrow_compile_compute::sort::sort_to_indices(&data, SortOptions::default()).unwrap();
    assert_eq!(arrow_perm, our_perm);

    c.bench_function("sort i32/arrow", |b| {
        b.iter(|| black_box(arrow_ord::sort::sort_to_indices(&data, None, None).unwrap()));
    });
    c.bench_function("sort i32/llvm", |b| {
        b.iter(|| {
            black_box(
                arrow_compile_compute::sort::sort_to_indices(&data, SortOptions::default())
                    .unwrap(),
            )
        });
    });

    let data = UInt64Array::from(
        vals.iter()
            .map(|x| if rng.bool() { Some(*x) } else { None })
            .collect_vec(),
    );
    c.bench_function("sort nullable i32/arrow", |b| {
        b.iter(|| black_box(arrow_ord::sort::sort_to_indices(&data, None, None).unwrap()));
    });
    c.bench_function("sort nullable i32/llvm", |b| {
        b.iter(|| {
            black_box(
                arrow_compile_compute::sort::sort_to_indices(&data, SortOptions::default())
                    .unwrap(),
            )
        });
    });

    let c1 = (0..1_000_000).map(|_| rng.i8(0..4)).collect_vec();
    let c2 = (0..1_000_000).map(|_| rng.i32(0..8)).collect_vec();
    let c3 = (0..1_000_000)
        .map(|_| if rng.bool() { Some(rng.i64(..)) } else { None })
        .collect_vec();
    let c1: Arc<dyn Array> = Arc::new(Int8Array::from(c1));
    let c2: Arc<dyn Array> = Arc::new(Int32Array::from(c2));
    let c3: Arc<dyn Array> = Arc::new(Int64Array::from(c3));

    c.bench_function("sort multicol/arrow", |b| {
        b.iter(|| {
            arrow_ord::sort::lexsort_to_indices(
                &[
                    SortColumn {
                        values: c1.clone(),
                        options: None,
                    },
                    SortColumn {
                        values: c2.clone(),
                        options: None,
                    },
                    SortColumn {
                        values: c3.clone(),
                        options: None,
                    },
                ],
                None,
            )
            .unwrap()
        })
    });
    c.bench_function("sort multicol/llvm", |b| {
        b.iter(|| {
            arrow_compile_compute::sort::multicol_sort_to_indices(
                &[&c1, &c2, &c3],
                &[
                    SortOptions::default(),
                    SortOptions::default(),
                    SortOptions::default(),
                ],
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
