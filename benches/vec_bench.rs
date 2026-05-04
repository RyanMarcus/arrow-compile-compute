use arrow_array::{
    builder::{FixedSizeListBuilder, Float32Builder},
    cast::AsArray,
    types::Float32Type,
    Datum, Float32Array, Scalar,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

const NN_VALUES: usize = 256;
const NN_DIMS: usize = 8;

fn make_vec_array(values: &[[f32; NN_DIMS]]) -> arrow_array::FixedSizeListArray {
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), NN_DIMS as i32);
    for row in values {
        builder.values().append_slice(row);
        builder.append(true);
    }
    builder.finish()
}

fn make_query(query: [f32; NN_DIMS]) -> Scalar<arrow_array::FixedSizeListArray> {
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), NN_DIMS as i32);
    builder.values().append_slice(&query);
    builder.append(true);
    Scalar::new(builder.finish())
}

fn squared_norms(values: &[[f32; NN_DIMS]]) -> Float32Array {
    Float32Array::from(
        values
            .iter()
            .map(|row| row.iter().map(|v| v * v).sum::<f32>())
            .collect::<Vec<_>>(),
    )
}

fn pack_dim_major(values: &[[f32; NN_DIMS]]) -> Vec<f32> {
    let mut packed = vec![0.0; NN_VALUES * NN_DIMS];
    for dim in 0..NN_DIMS {
        for row in 0..NN_VALUES {
            packed[dim * NN_VALUES + row] = values[row][dim];
        }
    }
    packed
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn nearest_neighbor_avx512_f32x8_256(
    query: &[f32; NN_DIMS],
    packed_values: &[f32],
    value_squared_norms: &[f32],
) -> usize {
    use std::arch::x86_64::{
        _mm512_add_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_storeu_ps,
        _mm512_sub_ps,
    };

    let mut best_idx = 0usize;
    let mut best_dist = f32::MAX;
    let mut dist_buf = [0.0f32; 16];

    for block in 0..(NN_VALUES / 16) {
        let mut dot = _mm512_set1_ps(0.0);
        for dim in 0..NN_DIMS {
            let values = _mm512_loadu_ps(packed_values.as_ptr().add(dim * NN_VALUES + block * 16));
            dot = _mm512_add_ps(dot, _mm512_mul_ps(_mm512_set1_ps(query[dim]), values));
        }

        let norms = _mm512_loadu_ps(value_squared_norms.as_ptr().add(block * 16));
        let dists = _mm512_sub_ps(norms, _mm512_add_ps(dot, dot));
        _mm512_storeu_ps(dist_buf.as_mut_ptr(), dists);

        for (lane, dist) in dist_buf.iter().copied().enumerate() {
            if dist < best_dist {
                best_dist = dist;
                best_idx = block * 16 + lane;
            }
        }
    }

    best_idx
}

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

    {
        let mut values = [[0.0; NN_DIMS]; NN_VALUES];
        for value in values.iter_mut() {
            for dim in value.iter_mut() {
                *dim = 2.0 * rng.f32() - 1.0;
            }
        }
        let mut query = [0.0; NN_DIMS];
        for dim in query.iter_mut() {
            *dim = 2.0 * rng.f32() - 1.0;
        }

        let values_arr = make_vec_array(&values);
        let query_arr = make_query(query);
        let value_squared_norms = squared_norms(&values);
        let packed_values = pack_dim_major(&values);
        let expected = arrow_compile_compute::vec::nearest_neighbor(
            &query_arr,
            &values_arr,
            &value_squared_norms,
        )
        .unwrap()
        .unwrap() as usize;

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            let avx512_expected = unsafe {
                nearest_neighbor_avx512_f32x8_256(
                    &query,
                    &packed_values,
                    value_squared_norms.values(),
                )
            };
            assert_eq!(expected, avx512_expected);
        }

        c.bench_function("nearest_neighbor 256x8/llvm-dsl", |b| {
            b.iter(|| {
                black_box(
                    arrow_compile_compute::vec::nearest_neighbor(
                        black_box(&query_arr),
                        black_box(&values_arr),
                        black_box(&value_squared_norms),
                    )
                    .unwrap(),
                )
            })
        });

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            c.bench_function("nearest_neighbor 256x8/manual-avx512", |b| {
                b.iter(|| unsafe {
                    black_box(nearest_neighbor_avx512_f32x8_256(
                        black_box(&query),
                        black_box(&packed_values),
                        black_box(value_squared_norms.values()),
                    ))
                })
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
