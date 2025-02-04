use arrow_array::StringArray;
use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    let random_strings = (0..1_000_000)
        .map(|_| {
            String::from_utf8(
                (0..rng.usize(4..1024))
                    .map(|_| rng.alphanumeric() as u8)
                    .collect_vec(),
            )
            .unwrap()
        })
        .collect_vec();

    let bytes = StringArray::from_iter_values(random_strings.iter());

    let prefix = StringArray::new_scalar(&"abcd");

    c.bench_function("starts_with/custom", |b| {
        b.iter(|| starts_with(&bytes, "abcd".as_bytes()))
    });

    c.bench_function("starts_with/arrow", |b| {
        b.iter(|| arrow_string::like::starts_with(&bytes, &prefix).unwrap())
    });
}

fn starts_with(bytes: &StringArray, prefix: &[u8]) -> Vec<u8> {
    let prefix = u32::from_le_bytes(prefix.try_into().unwrap());
    let mut to_return = Vec::with_capacity(bytes.offsets().len() - 1);
    for chunk in bytes.offsets().chunks_exact(8) {
        let mut mask = 0;
        for i in 0..8 {
            unsafe {
                let data = bytes
                    .value_data()
                    .get_unchecked(chunk[i] as usize..chunk[i] as usize + 4);
                let data = std::ptr::read_unaligned(data.as_ptr() as *const u32);
                if data == prefix {
                    mask |= 1 << i;
                }
            }
        }
        to_return.push(mask);
    }

    to_return
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
