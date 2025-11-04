use arrow_array::{
    builder::{FixedSizeListBuilder, Float32Builder},
    cast::AsArray,
    types::Float32Type,
    Datum, Scalar,
};
use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    {
        let mut vecs = FixedSizeListBuilder::new(Float32Builder::new(), 768);
        for _v_count in 0..16384 {
            for _d in 0..768 {
                vecs.values().append_value(2.0 * rng.f32() - 1.0);
            }
            vecs.append(true);
        }
        let vecs = vecs.finish();

        let mut q = FixedSizeListBuilder::new(Float32Builder::new(), 768);
        for _d in 0..768 {
            q.values().append_value(2.0 * rng.f32() - 1.0);
        }
        q.append(true);
        let q = Scalar::new(q.finish());

        c.bench_function("dot/llvm", |b| {
            b.iter(|| arrow_compile_compute::vec::dot(&q, &vecs).unwrap())
        });

        c.bench_function("dot/arrow", |b| {
            b.iter(|| {
                let q = q.get().0;
                let q = q.as_fixed_size_list().value(0);
                let q_v = q.as_primitive::<Float32Type>();
                let mut f32b = Float32Builder::new();
                for el in vecs.iter() {
                    match el {
                        Some(v) => {
                            let res: f32 = v
                                .as_primitive::<Float32Type>()
                                .values()
                                .iter()
                                .zip(q_v.values().iter())
                                .map(|(a, b)| a * b)
                                .sum();
                            f32b.append_value(res);
                        }
                        None => f32b.append_null(),
                    }
                }
                f32b.finish()
            })
        });

        c.bench_function("norm/llvm", |b| {
            b.iter(|| arrow_compile_compute::vec::norm(&vecs).unwrap())
        });

        c.bench_function("norm/arrow", |b| {
            b.iter(|| {
                let mut b = FixedSizeListBuilder::new(Float32Builder::new(), vecs.value_length());
                for el in vecs.iter() {
                    match el {
                        Some(v) => {
                            let v = v.as_primitive::<Float32Type>();
                            let norm = v.values().iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                            v.values()
                                .iter()
                                .map(|x| x / norm)
                                .for_each(|x| b.values().append_value(x));

                            b.append(true);
                        }
                        None => b.append(false),
                    }
                }
                b.finish()
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
