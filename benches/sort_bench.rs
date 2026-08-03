use std::{collections::HashSet, sync::Arc};

use arrow_array::{Array, Int32Array, Int64Array, Int8Array, UInt64Array};
use arrow_compile_compute::SortOptions;
use arrow_ord::sort::SortColumn;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

fn bench_multicol_case(c: &mut Criterion, name: &str, columns: &[Arc<dyn Array>]) {
    let llvm_options = SortOptions::default();
    let arrow_options = arrow_schema::SortOptions {
        descending: llvm_options.descending,
        nulls_first: llvm_options.nulls_first,
    };
    let arrow_columns = columns
        .iter()
        .map(|values| SortColumn {
            values: values.clone(),
            options: Some(arrow_options),
        })
        .collect_vec();
    let llvm_columns = columns.iter().map(|column| column.as_ref()).collect_vec();
    let options = vec![llvm_options; columns.len()];

    // Tie groups may permute differently between implementations, so compare
    // the sorted values per column rather than the index permutations.
    let arrow_perm = arrow_ord::sort::lexsort_to_indices(&arrow_columns, None).unwrap();
    let our_perm =
        arrow_compile_compute::sort::multicol_sort_to_indices(&llvm_columns, &options).unwrap();
    for column in columns {
        let arrow_sorted = arrow_select::take::take(column.as_ref(), &arrow_perm, None).unwrap();
        let our_sorted = arrow_select::take::take(column.as_ref(), &our_perm, None).unwrap();
        assert_eq!(arrow_sorted.as_ref(), our_sorted.as_ref());
    }

    c.bench_function(&format!("{name}/arrow"), |b| {
        b.iter(|| arrow_ord::sort::lexsort_to_indices(&arrow_columns, None).unwrap())
    });
    c.bench_function(&format!("{name}/llvm warm"), |b| {
        b.iter(|| {
            arrow_compile_compute::sort::multicol_sort_to_indices(&llvm_columns, &options).unwrap()
        })
    });
}

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

    c.bench_function("sort::sort_to_indices(array(u64)) 1m rows/arrow", |b| {
        b.iter(|| {
            black_box(arrow_ord::sort::sort_to_indices(&data, Some(arrow_options), None).unwrap())
        });
    });
    c.bench_function("sort::sort_to_indices(array(u64)) 1m rows/llvm warm", |b| {
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

    c.bench_function(
        "sort::sort_to_indices(array(nullable u64)) 1m rows/arrow",
        |b| {
            b.iter(|| {
                black_box(
                    arrow_ord::sort::sort_to_indices(&data, Some(arrow_options), None).unwrap(),
                )
            });
        },
    );
    c.bench_function(
        "sort::sort_to_indices(array(nullable u64)) 1m rows/llvm warm",
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

    bench_multicol_case(
        c,
        "sort::multicol_sort_to_indices(array(i8), array(i32), array(nullable i64)) 1m rows",
        &[c1.clone(), c2.clone(), c3],
    );

    let high_cardinality_i32: Arc<dyn Array> = Arc::new(Int32Array::from(
        vals.iter().map(|value| *value as i32).collect_vec(),
    ));
    bench_multicol_case(
        c,
        "sort::multicol_sort_to_indices(array(i8), array(i32)) 2 word key 1m rows",
        &[c1, high_cardinality_i32],
    );

    let four_word_columns = (0..3)
        .map(|column| {
            Arc::new(UInt64Array::from(
                vals.iter()
                    .map(|value| value.rotate_left(column * 17))
                    .collect_vec(),
            )) as Arc<dyn Array>
        })
        .collect_vec();
    bench_multicol_case(
        c,
        "sort::multicol_sort_to_indices(array(u64) x3) 4 word key 1m rows",
        &four_word_columns,
    );

    let eight_word_columns = (0..7)
        .map(|column| {
            Arc::new(UInt64Array::from(
                vals.iter()
                    .map(|value| value.rotate_left(column * 9))
                    .collect_vec(),
            )) as Arc<dyn Array>
        })
        .collect_vec();
    bench_multicol_case(
        c,
        "sort::multicol_sort_to_indices(array(u64) x7) 8 word key 1m rows",
        &eight_word_columns,
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
