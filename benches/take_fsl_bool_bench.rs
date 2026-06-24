use std::sync::Arc;

use arrow_array::{BooleanArray, FixedSizeListArray, UInt64Array};
use arrow_compile_compute::select;
use arrow_schema::{DataType, Field};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

const LIST_SIZE: i32 = 1024;
const ROWS: usize = 4096;
const TAKES: usize = 4096;

fn make_data() -> FixedSizeListArray {
    let mut rng = fastrand::Rng::with_seed(42);
    let values = (0..ROWS * LIST_SIZE as usize)
        .map(|_| rng.bool())
        .collect_vec();

    FixedSizeListArray::try_new(
        Arc::new(Field::new_list_field(DataType::Boolean, false)),
        LIST_SIZE,
        Arc::new(BooleanArray::from(values)),
        None,
    )
    .unwrap()
}

fn make_indices() -> UInt64Array {
    let mut rng = fastrand::Rng::with_seed(4242);
    UInt64Array::from((0..TAKES).map(|_| rng.u64(0..ROWS as u64)).collect_vec())
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let data = make_data();
    let idxes = make_indices();

    let arrow_res = arrow_select::take::take(&data, &idxes, None).unwrap();
    let our_res = select::take(&data, &idxes).unwrap();
    assert_eq!(our_res.as_ref(), arrow_res.as_ref());

    let mut group = c.benchmark_group("take fixed bool[1024]");
    group.sample_size(10);

    group.bench_function("execute/llvm", |b| {
        b.iter(|| select::take(black_box(&data), black_box(&idxes)).unwrap())
    });

    group.bench_function("execute/arrow", |b| {
        b.iter(|| arrow_select::take::take(black_box(&data), black_box(&idxes), None).unwrap())
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
