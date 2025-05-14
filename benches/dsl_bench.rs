use arrow_array::{cast::AsArray, types::Int32Type, ArrayRef, Int32Array};
use arrow_compile_compute::dsl::{DSLKernel, KernelOutputType};
use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    {
        // max(data1[idxes], data2[idxes])
        let data1 = (0..1_000_000).map(|_| rng.i32(..)).collect_vec();
        let data2 = (0..1_000_000).map(|_| rng.i32(..)).collect_vec();
        let idxes = (0..1_000_000).map(|_| rng.i32(0..1_000_000)).collect_vec();
        let data1 = Int32Array::from(data1);
        let data2 = Int32Array::from(data2);
        let idxes = Int32Array::from(idxes);

        fn arrow_compute(data1: &Int32Array, data2: &Int32Array, idxes: &Int32Array) -> ArrayRef {
            let i1 = arrow_select::take::take(data1, idxes, None).unwrap();
            let i2 = arrow_select::take::take(data2, idxes, None).unwrap();
            let cmp = arrow_ord::cmp::gt(&i1, &i2).unwrap();
            let idxes = cmp
                .values()
                .iter()
                .enumerate()
                .map(|(idx, b)| if b { (0, idx) } else { (1, idx) })
                .collect_vec();
            arrow_select::interleave::interleave(&[&i1, &i2], &idxes).unwrap()
        }
        let arrow_res = arrow_compute(&data1, &data2, &idxes);

        let k = DSLKernel::compile(&[&data1, &data2, &idxes], |ctx| {
            let dat1 = ctx.get_input(0)?;
            let dat2 = ctx.get_input(1)?;
            let idxs = ctx.get_input(2)?;

            idxs.into_iter()
                .map(|i| vec![dat1.at(&i[0]), dat2.at(&i[0])])
                .map(|i| vec![i[0].gt(&i[1]).select(&i[0], &i[1])])
                .collect(KernelOutputType::Array)
        })
        .unwrap();
        let our_res = k.call(&[&data1, &data2, &idxes]).unwrap();

        assert_eq!(
            arrow_res.as_primitive::<Int32Type>(),
            our_res.as_primitive::<Int32Type>()
        );

        c.bench_function("dsl idx max/llvm", |b| {
            b.iter(|| k.call(&[&data1, &data2, &idxes]).unwrap())
        });

        c.bench_function("dsl idx max/arrow", |b| {
            b.iter(|| arrow_compute(&data1, &data2, &idxes))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
