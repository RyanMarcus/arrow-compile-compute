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
        let indexes = (0..100_000)
            .map(|_| Some(rng.i32(0..data.len() as i32)))
            .sorted()
            .collect_vec();
        let indexes = Int32Array::from(indexes);

        let arr_res = arrow_select::take::take(&data, &indexes, None)
            .unwrap()
            .as_primitive::<Int32Type>()
            .clone();
        let our_res = arrow_compile_compute::compute::take(&data, &indexes)
            .unwrap()
            .as_primitive::<Int32Type>()
            .clone();
        assert_eq!(arr_res, our_res);

        c.bench_function("take i32/llvm", |b| {
            b.iter(|| arrow_compile_compute::compute::take(&data, &indexes).unwrap())
        });

        c.bench_function("take i32/arrow", |b| {
            b.iter(|| arrow_select::take::take(&data, &indexes, None).unwrap())
        });
    }

    {
        let data = (0..1_000_000)
            .map(|_| {
                (0..rng.usize(0..1024))
                    .map(|_| rng.lowercase())
                    .collect::<String>()
            })
            .collect_vec();
        let data = StringArray::from(data);
        let indexes = (0..1_000)
            .map(|_| Some(rng.i32(0..data.len() as i32)))
            .sorted()
            .collect_vec();
        let indexes = Int32Array::from(indexes);

        let arr_res = arrow_select::take::take(&data, &indexes, None)
            .unwrap()
            .as_string::<i32>()
            .clone();
        let arr_res = arr_res.iter().map(|x| x.unwrap()).collect_vec();

        let our_res = arrow_compile_compute::compute::take(&data, &indexes)
            .unwrap()
            .as_string_view()
            .clone();
        let our_res = our_res.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(arr_res, our_res);

        c.bench_function("take str/llvm", |b| {
            b.iter(|| arrow_compile_compute::compute::take(&data, &indexes).unwrap())
        });

        c.bench_function("take str/arrow", |b| {
            b.iter(|| arrow_select::take::take(&data, &indexes, None).unwrap())
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
