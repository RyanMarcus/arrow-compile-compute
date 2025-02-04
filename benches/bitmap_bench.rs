use arrow_array::{cast::AsArray, types::UInt64Type, BooleanArray, UInt64Array};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;

fn bitmap_to_vec(bm: &BooleanArray) -> UInt64Array {
    UInt64Array::from(
        bm.values()
            .set_indices()
            .map(|idx| idx as u64)
            .collect_vec(),
    )
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    {
        let data = (0..10_000_000).map(|_| rng.bool()).collect_vec();
        let data = BooleanArray::from(data);

        let arr_res = bitmap_to_vec(&data);
        let our_res = arrow_compile_compute::cast::cast(&data, &DataType::UInt64)
            .unwrap()
            .as_primitive::<UInt64Type>()
            .clone();
        assert_eq!(our_res, arr_res);

        c.bench_function("bitmap to vec/llvm", |b| {
            b.iter(|| arrow_compile_compute::cast::cast(&data, &DataType::UInt64).unwrap());
        });

        c.bench_function("bitmap to vec/arrow", |b| {
            b.iter(|| bitmap_to_vec(&data));
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
