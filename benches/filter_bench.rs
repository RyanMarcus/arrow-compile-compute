use arrow_array::{
    cast::AsArray, types::Int64Type, Array, BooleanArray, Int32Array, Int64Array, Int8Array,
    RunArray,
};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;

fn generate_random_ree_array(num_run_ends: usize) -> RunArray<Int64Type> {
    let mut rng = fastrand::Rng::with_seed(42 + num_run_ends as u64);
    let ree_array_run_ends = (0..num_run_ends)
        .map(|_| rng.i64(1..40))
        .scan(0, |acc, x| {
            *acc = *acc + x;
            Some(*acc)
        })
        .collect_vec();
    let ree_array_values = (0..num_run_ends).map(|_| rng.i64(-5..5)).collect_vec();
    RunArray::try_new(
        &Int64Array::from(ree_array_run_ends),
        &Int64Array::from(ree_array_values),
    )
    .unwrap()
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    {
        let data = (0..10_000_000).map(|_| rng.i32(..)).collect_vec();
        let data = Int32Array::from(data);
        let bools: BooleanArray = (0..10_000_000).map(|_| Some(rng.bool())).collect();

        let arr_res = arrow_select::filter::filter(&data, &bools).unwrap();
        let our_res = arrow_compile_compute::compute::filter(&data, &bools).unwrap();
        assert_eq!(our_res.len(), arr_res.len());
        assert_eq!(
            arrow_ord::cmp::eq(&our_res, &arr_res).unwrap().true_count(),
            our_res.len()
        );

        c.bench_function("filter prim i32/llvm", |b| {
            b.iter(|| arrow_compile_compute::compute::filter(&data, &bools).unwrap())
        });

        c.bench_function("filter prim i32/arrow", |b| {
            b.iter(|| arrow_select::filter::filter(&data, &bools).unwrap())
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
