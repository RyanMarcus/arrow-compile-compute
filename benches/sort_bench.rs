use std::{collections::HashSet, sync::Arc};

use arrow_array::{Array, Int32Array, Int64Array, Int8Array, UInt64Array};
use arrow_compile_compute::SortOptions;
use arrow_ord::sort::SortColumn;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

fn bench_multicol_case(c: &mut Criterion, name: &str, columns: &[Arc<dyn Array>]) {
    let arrow_columns = columns
        .iter()
        .map(|values| SortColumn {
            values: values.clone(),
            options: None,
        })
        .collect_vec();
    let llvm_columns = columns.iter().map(|column| column.as_ref()).collect_vec();
    let options = vec![SortOptions::default(); columns.len()];
    let arrow_name = format!("{name}/arrow");
    let llvm_name = format!("{name}/llvm");

    c.bench_function(&arrow_name, |b| {
        b.iter(|| arrow_ord::sort::lexsort_to_indices(&arrow_columns, None).unwrap())
    });
    c.bench_function(&llvm_name, |b| {
        b.iter(|| {
            arrow_compile_compute::sort::multicol_sort_to_indices(&llvm_columns, &options).unwrap()
        })
    });
}

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

    bench_multicol_case(c, "sort multicol", &[c1.clone(), c2.clone(), c3]);
    let high_cardinality_i32: Arc<dyn Array> = Arc::new(Int32Array::from(
        vals.iter().map(|value| *value as i32).collect_vec(),
    ));
    bench_multicol_case(c, "sort multicol 2 words", &[c1, high_cardinality_i32]);

    let four_word_columns = (0..3)
        .map(|column| {
            Arc::new(UInt64Array::from(
                vals.iter()
                    .map(|value| value.rotate_left(column * 17))
                    .collect_vec(),
            )) as Arc<dyn Array>
        })
        .collect_vec();
    bench_multicol_case(c, "sort multicol 4 words", &four_word_columns);

    let eight_word_columns = (0..7)
        .map(|column| {
            Arc::new(UInt64Array::from(
                vals.iter()
                    .map(|value| value.rotate_left(column * 9))
                    .collect_vec(),
            )) as Arc<dyn Array>
        })
        .collect_vec();
    bench_multicol_case(c, "sort multicol 8 words", &eight_word_columns);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
