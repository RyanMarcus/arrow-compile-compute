use arrow_array::{types::Int64Type, Array, Datum, Float32Array, Int32Array, Int64Array, RunArray};
use arrow_compile_compute::{CodeGen, Predicate};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use inkwell::context::Context;
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

    c.bench_function("compile", |b| {
        b.iter(|| {
            let ctx = Context::create();
            let cg = CodeGen::new(&ctx);
            cg.primitive_primitive_cmp(
                &arrow_schema::DataType::Int32,
                false,
                &arrow_schema::DataType::Int32,
                false,
                Predicate::Eq,
            )
            .unwrap();
        })
    });

    {
        let data1 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..1000)).collect_vec());
        let data2 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..1000)).collect_vec());
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &arrow_schema::DataType::Int32,
                false,
                &arrow_schema::DataType::Int32,
                false,
                Predicate::Eq,
            )
            .unwrap();

        let arrow_answer = arrow_ord::cmp::eq(&data1, &data2).unwrap();
        let llvm_answer = f.call(&data1, &data2).unwrap();
        assert_eq!(arrow_answer, llvm_answer);

        c.bench_function("i32/execute arrow", |b| {
            b.iter(|| arrow_ord::cmp::eq(&data1, &data2).unwrap())
        });
        c.bench_function("i32/execute llvm", |b| {
            b.iter(|| f.call(&data1, &data2));
        });
    }

    {
        let data1 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..1000)).collect_vec());
        let data2 = Int64Array::from((0..10_000_000).map(|_| rng.i64(0..1000)).collect_vec());
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &arrow_schema::DataType::Int32,
                false,
                &arrow_schema::DataType::Int64,
                false,
                Predicate::Eq,
            )
            .unwrap();
        let arrow_answer =
            arrow_ord::cmp::eq(&arrow_cast::cast(&data1, &DataType::Int64).unwrap(), &data2)
                .unwrap();
        let llvm_answer = f.call(&data1, &data2).unwrap();
        assert_eq!(arrow_answer, llvm_answer);

        c.bench_function("cast_eq/execute arrow", |b| {
            b.iter(|| {
                arrow_ord::cmp::eq(&arrow_cast::cast(&data1, &DataType::Int64).unwrap(), &data2)
                    .unwrap()
            })
        });
        c.bench_function("cast_eq/execute llvm", |b| {
            b.iter(|| f.call(&data1, &data2));
        });
    }

    {
        let data1 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..100)).collect_vec());
        let data2 = Int64Array::from((0..10_000_000).map(|_| rng.i64(0..100)).collect_vec());
        let dict_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int64));

        let data1 = arrow_cast::cast(&data1, &dict_type).unwrap();
        let data2 = arrow_cast::cast(&data2, &dict_type).unwrap();

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(&dict_type, false, &dict_type, false, Predicate::Eq)
            .unwrap();
        let arrow_answer = arrow_ord::cmp::eq(&data1, &data2).unwrap();
        let llvm_answer = f.call(&data1, &data2).unwrap();
        assert_eq!(arrow_answer, llvm_answer);

        c.bench_function("dict_i64/execute arrow", |b| {
            b.iter(|| {
                arrow_ord::cmp::eq(&arrow_cast::cast(&data1, &DataType::Int64).unwrap(), &data2)
                    .unwrap()
            })
        });
        c.bench_function("dict_i64/execute llvm", |b| {
            b.iter(|| f.call(&data1, &data2));
        });
    }

    {
        let ree_data = generate_random_ree_array(100_000);
        let pri_data = Int64Array::from((0..ree_data.len()).map(|_| rng.i64(-5..5)).collect_vec());

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &pri_data.data_type(),
                false,
                &ree_data.data_type(),
                false,
                Predicate::Eq,
            )
            .unwrap();

        let ree_as_prim = Int64Array::from_iter(ree_data.downcast::<Int64Array>().unwrap());
        let arrow_answer = arrow_ord::cmp::eq(&ree_as_prim, &pri_data).unwrap();
        let llvm_answer = f.call(&pri_data, &ree_data).unwrap();
        assert_eq!(arrow_answer, llvm_answer);

        c.bench_function("ree_i64/execute llvm", |b| {
            b.iter(|| f.call(&pri_data, &ree_data));
        });
    }

    {
        let data1 = Int32Array::from((0..10_000_000).map(|_| rng.i32(0..1000)).collect_vec());
        let data2 = Int32Array::new_scalar(50);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &arrow_schema::DataType::Int32,
                false,
                &arrow_schema::DataType::Int32,
                true,
                Predicate::Eq,
            )
            .unwrap();

        let arrow_answer = arrow_ord::cmp::eq(&data1, &data2).unwrap();
        let llvm_answer = f.call(&data1, data2.get().0).unwrap();
        assert_eq!(arrow_answer, llvm_answer);

        c.bench_function("i32scalar/execute arrow", |b| {
            b.iter(|| arrow_ord::cmp::eq(&data1, &data2).unwrap())
        });
        c.bench_function("i32scalar/execute llvm", |b| {
            b.iter(|| f.call(&data1, data2.get().0));
        });
    }

    {
        let data1 = Float32Array::from((0..10_000_000).map(|_| rng.f32()).collect_vec());
        let data2 = Float32Array::from((0..10_000_000).map(|_| rng.f32()).collect_vec());
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &arrow_schema::DataType::Float32,
                false,
                &arrow_schema::DataType::Float32,
                false,
                Predicate::Lt,
            )
            .unwrap();

        let arrow_answer = arrow_ord::cmp::lt(&data1, &data2).unwrap();
        let llvm_answer = f.call(&data1, &data2).unwrap();
        assert_eq!(arrow_answer, llvm_answer);

        c.bench_function("f32/execute arrow", |b| {
            b.iter(|| arrow_ord::cmp::eq(&data1, &data2).unwrap())
        });
        c.bench_function("f32/execute llvm", |b| {
            b.iter(|| f.call(&data1, &data2));
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
