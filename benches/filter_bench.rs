use arrow_array::{cast::AsArray, types::Int32Type, Array, BooleanArray, Int32Array, StringArray};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;

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

    {
        let data = (0..10_000_000).map(|_| rng.i32(-5..5)).collect_vec();
        let data = Int32Array::from(data);
        let data = arrow_cast::cast(
            &data,
            &dictionary_data_type(DataType::Int8, DataType::Int32),
        )
        .unwrap();
        let bools: BooleanArray = (0..10_000_000).map(|_| Some(rng.bool())).collect();

        let arr_res = arrow_select::filter::filter(&data, &bools).unwrap();
        let arr_res = arrow_cast::cast(&arr_res, &DataType::Int32).unwrap();

        let our_res = arrow_compile_compute::compute::filter(&data, &bools).unwrap();
        let our_res = arrow_cast::cast(&our_res, &DataType::Int32).unwrap();
        assert_eq!(arr_res.len(), bools.true_count());
        assert_eq!(our_res.len(), bools.true_count());

        assert_eq!(
            arrow_ord::cmp::eq(&our_res, &arr_res).unwrap().true_count(),
            our_res.len()
        );

        c.bench_function("filtercast dict i32/llvm", |b| {
            b.iter(|| arrow_compile_compute::compute::filter(&data, &bools).unwrap())
        });

        c.bench_function("filter dict i32/arrow", |b| {
            b.iter(|| arrow_select::filter::filter(&data, &bools).unwrap())
        });

        c.bench_function("filtercast dict i32/arrow", |b| {
            b.iter(|| {
                arrow_cast::cast(
                    &arrow_select::filter::filter(&data, &bools).unwrap(),
                    &DataType::Int32,
                )
                .unwrap()
            })
        });
    }

    {
        let data = (0..10_000_000)
            .map(|_| {
                let size = rng.usize(0..32);
                String::from_utf8((0..size).map(|_| rng.u8(32..126)).collect_vec()).unwrap()
            })
            .collect_vec();
        let data = StringArray::from(data);
        let bools: BooleanArray = (0..data.len()).map(|_| Some(rng.bool())).collect();

        let arr_res = arrow_select::filter::filter(&data, &bools)
            .unwrap()
            .as_string::<i32>()
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect_vec();
        let our_res = arrow_compile_compute::compute::filter(&data, &bools)
            .unwrap()
            .as_string_view()
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect_vec();
        assert_eq!(our_res, arr_res);

        c.bench_function("filter str/llvm", |b| {
            b.iter(|| arrow_compile_compute::compute::filter(&data, &bools).unwrap())
        });

        c.bench_function("filter str/arrow", |b| {
            b.iter(|| arrow_select::filter::filter(&data, &bools).unwrap())
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
