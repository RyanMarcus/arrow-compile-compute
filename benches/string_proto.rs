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

    let custom = starts_with(&bytes, "abcd".as_bytes());
    let arrow = arrow_string::like::starts_with(&bytes, &prefix).unwrap();
    for (idx, expected) in arrow.iter().enumerate() {
        let actual = custom[idx / 8] & (1 << (idx % 8)) != 0;
        assert_eq!(Some(actual), expected);
    }
    let llvm = arrow_compile_compute::cmp::starts_with(&bytes, &prefix).unwrap();
    assert_eq!(arrow, llvm);

    c.bench_function(
        "cmp::starts_with(array(utf8), scalar(utf8)) 1m rows/llvm warm",
        |b| b.iter(|| arrow_compile_compute::cmp::starts_with(&bytes, &prefix).unwrap()),
    );

    c.bench_function(
        "cmp::starts_with(array(utf8), scalar(utf8)) 1m rows/arrow",
        |b| b.iter(|| arrow_string::like::starts_with(&bytes, &prefix).unwrap()),
    );

    // Hand-written first-4-bytes prototype: a lower bound, not a comparable
    // implementation (packed bitmask output, fixed 4-byte prefix only).
    c.bench_function(
        "starts_with first-4-bytes prototype (array(utf8)) 1m rows/custom",
        |b| b.iter(|| starts_with(&bytes, "abcd".as_bytes())),
    );
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
