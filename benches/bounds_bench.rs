use arrow_array::{Int32Array, Scalar};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);
    let values = (0..1_000_000)
        .map(|_| rng.i32(-10_000..10_000))
        .collect::<Vec<_>>();

    let values_array = Int32Array::from(values);
    let lb_scalar = Scalar::new(Int32Array::from(vec![-50_000]));
    let ub_scalar = Scalar::new(Int32Array::from(vec![50_000]));

    let llvm_scalar =
        arrow_compile_compute::cmp::bounds(&values_array, &lb_scalar, &ub_scalar).unwrap();
    let arrow_scalar = {
        let gte = arrow_ord::cmp::gt_eq(&values_array, &lb_scalar).unwrap();
        let lt = arrow_ord::cmp::lt(&values_array, &ub_scalar).unwrap();
        let combined = arrow_arith::boolean::and(&gte, &lt).unwrap();
        combined.true_count() == combined.len()
    };
    assert_eq!(llvm_scalar, arrow_scalar);

    c.bench_function("bounds scalar llvm/i32", |b| {
        b.iter(|| {
            black_box(
                arrow_compile_compute::cmp::bounds(
                    black_box(&values_array),
                    black_box(&lb_scalar),
                    black_box(&ub_scalar),
                )
                .unwrap(),
            )
        })
    });

    c.bench_function("bounds scalar arrow/i32", |b| {
        b.iter(|| {
            let gte =
                arrow_ord::cmp::gt_eq(black_box(&values_array), black_box(&lb_scalar)).unwrap();
            let lt = arrow_ord::cmp::lt(black_box(&values_array), black_box(&ub_scalar)).unwrap();
            let combined = arrow_arith::boolean::and(&gte, &lt).unwrap();
            black_box(combined.true_count() == combined.len())
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
