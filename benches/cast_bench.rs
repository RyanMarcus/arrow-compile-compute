use arrow_array::{
    cast::AsArray, types::Int64Type, Array, Int32Array, Int64Array, Int8Array, RunArray,
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

        let arr_res = arrow_cast::cast::cast(&data, &DataType::Int64).unwrap();
        let our_res = arrow_compile_compute::cast::cast(&data, &DataType::Int64).unwrap();
        assert_eq!(our_res.len(), arr_res.len());
        assert_eq!(
            arrow_ord::cmp::eq(&our_res, &arr_res).unwrap().true_count(),
            our_res.len()
        );

        c.bench_function("convert prim i32 to i64/llvm", |b| {
            b.iter(|| arrow_compile_compute::cast::cast(&data, &DataType::Int64).unwrap())
        });

        c.bench_function("convert prim i32 to i64/arrow", |b| {
            b.iter(|| arrow_cast::cast::cast(&data, &DataType::Int64).unwrap())
        });
    }

    {
        let data = (0..10_000_000).map(|_| rng.i8(..)).collect_vec();
        let data = Int8Array::from(data);

        let arr_res = arrow_cast::cast::cast(&data, &DataType::Int16).unwrap();
        let our_res = arrow_compile_compute::cast::cast(&data, &DataType::Int16).unwrap();
        assert_eq!(our_res.len(), arr_res.len());
        assert_eq!(
            arrow_ord::cmp::eq(&our_res, &arr_res).unwrap().true_count(),
            our_res.len()
        );

        c.bench_function("convert prim i8 to i16/llvm", |b| {
            b.iter(|| arrow_compile_compute::cast::cast(&data, &DataType::Int16).unwrap())
        });

        c.bench_function("convert prim i8 to i16/arrow", |b| {
            b.iter(|| arrow_cast::cast::cast(&data, &DataType::Int16).unwrap())
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
        let our_prim = arrow_compile_compute::cast::cast(&data, &DataType::Int64).unwrap();
        assert_eq!(our_prim.len(), prim_data.len());
        assert_eq!(
            arrow_ord::cmp::eq(&our_prim, &prim_data)
                .unwrap()
                .true_count(),
            our_prim.len()
        );

        c.bench_function("convert dict i64 to i64/llvm", |b| {
            b.iter(|| arrow_compile_compute::cast::cast(&data, &DataType::Int64).unwrap())
        });

        c.bench_function("convert dict i64 to i64/arrow", |b| {
            b.iter(|| arrow_cast::cast::cast(&data, &DataType::Int64).unwrap())
        });
    }

    {
        let data = generate_random_ree_array(100_000);
        let prim_data = Int64Array::from_iter(data.downcast::<Int64Array>().unwrap());
        let our_prim = arrow_compile_compute::cast::cast(&data, &DataType::Int64)
            .unwrap()
            .as_primitive()
            .clone();
        assert_eq!(prim_data, our_prim);

        c.bench_function("convert ree i64 to i64/llvm", |b| {
            b.iter(|| arrow_compile_compute::cast::cast(&data, &DataType::Int64).unwrap())
        });

        c.bench_function("convert ree i64 to i64/arrow", |b| {
            b.iter(|| Int64Array::from_iter(data.downcast::<Int64Array>().unwrap()))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
