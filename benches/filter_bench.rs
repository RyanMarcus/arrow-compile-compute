use arrow_array::{
    cast::AsArray,
    types::{Int32Type, Int64Type},
    Array, BooleanArray, Int32Array, Int64Array, RunArray,
};
use arrow_schema::DataType;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    {
        let data = (0..10_000_000).map(|_| rng.i32(..)).collect_vec();
        let mask = (0..10_000_000).map(|_| rng.bool()).collect_vec();
        let data = Int32Array::from(data);
        let mask = BooleanArray::from(mask);

        let arr_res = arrow_select::filter::filter(&data, &mask).unwrap();
        let our_res = arrow_compile_compute::select::filter(&data, &mask).unwrap();
        assert_eq!(our_res.len(), arr_res.len());
        assert_eq!(
            arrow_ord::cmp::eq(&our_res, &arr_res).unwrap().true_count(),
            our_res.len()
        );

        c.bench_function("select::filter(array(i32), array(bool))/llvm warm", |b| {
            b.iter(|| arrow_compile_compute::select::filter(&data, &mask).unwrap())
        });

        c.bench_function("select::filter(array(i32), array(bool))/arrow", |b| {
            b.iter(|| arrow_select::filter::filter(&data, &mask).unwrap())
        });
    }

    {
        let data = Int32Array::from((0..1_000_000).map(|_| rng.i32(..)).collect_vec());
        let rees = Int64Array::from(
            (0..1_000_000)
                .map(|_| rng.i64(1..40))
                .scan(0_i64, |state, x| {
                    *state += x;
                    Some(*state)
                })
                .collect_vec(),
        );
        let data = RunArray::<Int64Type>::try_new(&rees, &data).unwrap();
        let mask = (0..data.len()).map(|_| rng.bool()).collect_vec();
        let mask = BooleanArray::from(mask);

        let arr_res = arrow_select::filter::filter(&data, &mask).unwrap();
        let arr_res = arrow_cast::cast(&arr_res, &DataType::Int32).unwrap();
        let our_res = arrow_compile_compute::select::filter(&data, &mask).unwrap();
        assert_eq!(
            our_res.as_primitive::<Int32Type>(),
            arr_res.as_primitive::<Int32Type>()
        );

        c.bench_function(
            "select::filter(run_end_encoded(i64, i32), array(bool))/llvm warm",
            |b| b.iter(|| arrow_compile_compute::select::filter(&data, &mask).unwrap()),
        );

        c.bench_function(
            "select::filter(run_end_encoded(i64, i32), array(bool))/arrow",
            |b| {
                b.iter(|| {
                    let filtered = arrow_select::filter::filter(&data, &mask).unwrap();
                    black_box(arrow_cast::cast(&filtered, &DataType::Int32).unwrap())
                })
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
