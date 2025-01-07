use arrow_array::{
    cast::AsArray,
    types::{Int32Type, Int64Type},
    Int32Array, Int64Array,
};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    {
        let data = (0..10_000_000).map(|_| rng.i32(..)).collect_vec();
        let data = Int32Array::from(data);

        let arr_res = arrow_arith::aggregate::min(&data).unwrap();
        let our_res = arrow_compile_compute::compute::min(&data)
            .unwrap()
            .unwrap()
            .as_primitive::<Int32Type>()
            .clone()
            .value(0);
        assert_eq!(our_res, arr_res);

        c.bench_function("min i32/llvm", |b| {
            b.iter(|| arrow_compile_compute::compute::min(&data).unwrap().unwrap());
        });

        c.bench_function("min i32/arrow", |b| {
            b.iter(|| arrow_arith::aggregate::min(&data).unwrap());
        });
    }

    {
        let vec_data = (0..10_000_000).map(|_| rng.i64(-50..50)).collect_vec();
        let prim_data = Int64Array::from(vec_data);
        let data = arrow_cast::cast::cast(
            &prim_data,
            &dictionary_data_type(DataType::Int8, DataType::Int64),
        )
        .unwrap();
        let our_res = arrow_compile_compute::compute::min(&data)
            .unwrap()
            .unwrap()
            .as_primitive::<Int64Type>()
            .clone()
            .value(0);
        let arr_res = arrow_arith::aggregate::min(&prim_data).unwrap();
        assert_eq!(our_res, arr_res);

        c.bench_function("min dict i64/llvm", |b| {
            b.iter(|| arrow_compile_compute::compute::min(&data).unwrap().unwrap())
        });

        c.bench_function("min dict i64/arrow", |b| {
            b.iter(|| {
                arrow_arith::aggregate::min(
                    arrow_cast::cast::cast(&data, &DataType::Int64)
                        .unwrap()
                        .as_primitive::<Int64Type>(),
                )
                .unwrap()
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
