use arrow_array::{types::Int64Type, Float32Array, Int32Array, Int64Array, RunArray, StringArray};
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
        let data1 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..1000)).collect_vec());
        let data2 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..1000)).collect_vec());

        let arrow_answer = arrow_ord::cmp::eq(&data1, &data2).unwrap();
        let llvm_answer = arrow_compile_compute::cmp::eq(&data1, &data2).unwrap();
        assert_eq!(arrow_answer, llvm_answer);

        c.bench_function("cmpi32/execute arrow", |b| {
            b.iter(|| arrow_ord::cmp::eq(&data1, &data2).unwrap())
        });
        c.bench_function("cmpi32/execute llvm", |b| {
            b.iter(|| arrow_compile_compute::cmp::eq(&data1, &data2).unwrap());
        });
    }

    {
        let data1 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..1000)).collect_vec());
        let data2 = Int64Array::from((0..10_000_000).map(|_| rng.i64(0..1000)).collect_vec());

        let llvm_answer_kernel = arrow_compile_compute::cmp::eq(&data1, &data2).unwrap();
        let arrow_answer =
            arrow_ord::cmp::eq(&arrow_cast::cast(&data1, &DataType::Int64).unwrap(), &data2)
                .unwrap();
        assert_eq!(arrow_answer, llvm_answer_kernel);
        c.bench_function("cast_eq/execute llvm", |b| {
            b.iter(|| arrow_compile_compute::cmp::eq(&data1, &data2).unwrap())
        });

        c.bench_function("cast_eq/execute arrow", |b| {
            b.iter(|| {
                arrow_ord::cmp::eq(&arrow_cast::cast(&data1, &DataType::Int64).unwrap(), &data2)
                    .unwrap()
            })
        });
    }

    {
        let data1 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..100)).collect_vec());
        let data2 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..100)).collect_vec());
        let dict_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32));

        let data1 = arrow_cast::cast(&data1, &dict_type).unwrap();
        let data2 = arrow_cast::cast(&data2, &dict_type).unwrap();

        let llvm_answer = arrow_compile_compute::cmp::eq(&data1, &data2).unwrap();
        let arrow_answer = arrow_ord::cmp::eq(&data1, &data2).unwrap();
        assert_eq!(llvm_answer, arrow_answer);

        c.bench_function("dict_i32/execute arrow", |b| {
            b.iter(|| arrow_ord::cmp::eq(&data1, &data2).unwrap())
        });
        c.bench_function("dict_i32/execute llvm", |b| {
            b.iter(|| arrow_ord::cmp::eq(&data1, &data2).unwrap());
        });
    }

    {
        let data1 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..100)).collect_vec());
        let data2 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..100)).collect_vec());
        let dict_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32));

        let data1 = arrow_cast::cast(&data1, &dict_type).unwrap();

        let llvm_answer = arrow_compile_compute::cmp::eq(&data1, &data2).unwrap();
        let arrow_answer = arrow_ord::cmp::eq(&data1, &data2).unwrap();
        assert_eq!(llvm_answer, arrow_answer);

        c.bench_function("dict_prim_i32/execute arrow", |b| {
            b.iter(|| arrow_ord::cmp::eq(&data1, &data2).unwrap())
        });
        c.bench_function("dict_prim_i32/execute llvm", |b| {
            b.iter(|| arrow_compile_compute::cmp::eq(&data1, &data2).unwrap());
        });
    }

    {
        let data1 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..1000)).collect_vec());
        let data2 = Int32Array::new_scalar(50);

        let arrow_answer = arrow_ord::cmp::gt_eq(&data1, &data2).unwrap();
        let llvm_answer = arrow_compile_compute::cmp::gt_eq(&data1, &data2).unwrap();
        assert_eq!(arrow_answer, llvm_answer);

        c.bench_function("i32scalar/execute arrow", |b| {
            b.iter(|| arrow_ord::cmp::gt_eq(&data1, &data2).unwrap())
        });

        c.bench_function("i32scalar/execute llvm", |b| {
            b.iter(|| arrow_compile_compute::cmp::gt_eq(&data1, &data2).unwrap())
        });
    }

    {
        let data1 = Float32Array::from((0..10_000_000).map(|_| rng.f32()).collect_vec());
        let data2 = Float32Array::from((0..10_000_000).map(|_| rng.f32()).collect_vec());

        let arrow_answer = arrow_ord::cmp::lt(&data1, &data2).unwrap();
        let llvm_answer = arrow_compile_compute::cmp::lt(&data1, &data2).unwrap();
        assert_eq!(arrow_answer, llvm_answer);

        c.bench_function("f32_lt/execute arrow", |b| {
            b.iter(|| arrow_ord::cmp::lt(&data1, &data2).unwrap())
        });
        c.bench_function("f32_lt/execute llvm", |b| {
            b.iter(|| arrow_compile_compute::cmp::lt(&data1, &data2).unwrap())
        });
    }

    {
        let random_strings: Vec<String> = (0..1_000_000)
            .map(|_| {
                let len = rng.usize(5..15);
                (0..len).map(|_| (rng.u8(97..123)) as char).collect()
            })
            .collect();
        let scalar = StringArray::new_scalar(random_strings[500_000].clone());
        let data = StringArray::from(random_strings);

        let arrow_answer = arrow_ord::cmp::lt(&data, &scalar).unwrap();
        let llvm_answer = arrow_compile_compute::cmp::lt(&data, &scalar).unwrap();
        assert_eq!(arrow_answer, llvm_answer);

        c.bench_function("strlt/execute arrow", |b| {
            b.iter(|| arrow_ord::cmp::lt(&data, &scalar).unwrap())
        });
        c.bench_function("strlt/execute llvm", |b| {
            b.iter(|| arrow_compile_compute::cmp::lt(&data, &scalar).unwrap())
        });
    }

    {
        let arr = generate_random_ree_array(100_000);
        let sca = Int64Array::new_scalar(0);

        c.bench_function("ree_eq/execute llvm", |b| {
            b.iter(|| arrow_compile_compute::cmp::eq(&arr, &sca).unwrap())
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
