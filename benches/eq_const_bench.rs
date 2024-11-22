use std::arch::x86_64::{
    __m256d, __m256i, _mm256_cmpeq_epi64, _mm256_load_si256, _mm256_movemask_pd, _mm256_set1_epi64x,
};

use arrow_array::Int64Array;
use arrow_compile_compute::compile_eq_const;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

fn eq_const_native(arr: &[i64], c: i64, out: &mut [u32]) {
    let mut buf: u32 = 0;
    let mut buf_idx: u32 = 0;
    let mut out_idx: usize = 0;

    unsafe {
        let target = _mm256_set1_epi64x(c);

        for chunk in arr.chunks_exact(4) {
            let chunk = _mm256_load_si256(chunk.as_ptr() as *const __m256i);
            let cmp = _mm256_cmpeq_epi64(chunk, target);
            let mask = _mm256_movemask_pd(std::mem::transmute::<__m256i, __m256d>(cmp)) as u32;

            buf |= mask << buf_idx;
            buf_idx += 4;
            if buf_idx >= 32 {
                *out.get_unchecked_mut(out_idx) = buf;
                out_idx += 1;
                buf_idx = 0;
                buf = 0;
            }
        }
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("compile", |b| b.iter(|| compile_eq_const()));

    let mut rng = fastrand::Rng::with_seed(42);
    let data = Int64Array::from((0..10_000_000).map(|_| rng.i64(0..1000)).collect_vec());
    let val = Int64Array::new_scalar(5);
    let f = compile_eq_const();

    c.bench_function("execute compiled", |b| b.iter(|| f.execute(&data, &val)));
    c.bench_function("execute arrow", |b| {
        b.iter(|| arrow_ord::cmp::eq(&data, &val))
    });

    let mut out = vec![0_u32; data.len().div_ceil(32)];
    c.bench_function("execute native", |b| {
        b.iter(|| eq_const_native(data.values(), 5, &mut out))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
