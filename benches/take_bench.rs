use arrow_array::{Array, Int32Array, UInt64Array};
use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    {
        let data = (0..10_000_000).map(|_| rng.i32(..)).collect_vec();
        let idxes = (0..1_000_000)
            .map(|_| rng.u64(0..data.len() as u64))
            .collect_vec();
        let data = Int32Array::from(data);
        let idxes = UInt64Array::from(idxes);

        let arr_res = arrow_select::take::take(&data, &idxes, None).unwrap();
        let our_res = arrow_compile_compute::select::take(&data, &idxes).unwrap();
        assert_eq!(our_res.len(), arr_res.len());
        assert_eq!(
            arrow_ord::cmp::eq(&our_res, &arr_res).unwrap().true_count(),
            our_res.len()
        );

        c.bench_function("take i32/llvm", |b| {
            b.iter(|| arrow_compile_compute::select::take(&data, &idxes).unwrap())
        });

        c.bench_function("take i32/arrow", |b| {
            b.iter(|| arrow_select::take::take(&data, &idxes, None).unwrap())
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
