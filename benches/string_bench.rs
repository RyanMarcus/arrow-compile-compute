//! String predicate benchmarks, always through the public API.
//!
//! Coverage and how each op executes in this crate:
//!   * `cmp::starts_with` / `cmp::ends_with` — JIT `StringStartEndKernel`;
//!     per-row cost is length-independent (prefix/suffix bytes only), so these
//!     run on the long-string dataset (1m rows, 4-1024 bytes each).
//!   * `cmp::contains` — hybrid: JIT'd byte iterator + `memchr` substring
//!     search in a Rust closure. Scan-bound, so it uses the short-string
//!     dataset (1m rows, 8-32 bytes) to keep per-iteration times reasonable.
//!   * `cmp::like` — hybrid: the pattern is parsed into one of several
//!     specialized Rust closures (exact / prefix / suffix / infix / general
//!     multi-wildcard). One benchmark per pattern shape, since each shape
//!     takes a different code path in BOTH engines.
//!
//! Baselines are the stock `arrow_string::like` kernels. The hand-written
//! first-4-bytes prototype is kept as an unpaired lower bound for the
//! starts_with case.

use arrow_array::StringArray;
use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(42);

    // ---- long strings: prefix/suffix predicates (length-independent) --------
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

    {
        let suffix = StringArray::new_scalar(&"wxyz");
        let llvm = arrow_compile_compute::cmp::ends_with(&bytes, &suffix).unwrap();
        let arrow = arrow_string::like::ends_with(&bytes, &suffix).unwrap();
        assert_eq!(arrow, llvm);

        c.bench_function(
            "cmp::ends_with(array(utf8), scalar(utf8)) 1m rows/llvm warm",
            |b| b.iter(|| arrow_compile_compute::cmp::ends_with(&bytes, &suffix).unwrap()),
        );
        c.bench_function(
            "cmp::ends_with(array(utf8), scalar(utf8)) 1m rows/arrow",
            |b| b.iter(|| arrow_string::like::ends_with(&bytes, &suffix).unwrap()),
        );
    }

    // ---- short strings: scan-bound predicates (contains, like) --------------
    let short_strings = (0..1_000_000)
        .map(|_| {
            String::from_utf8(
                (0..rng.usize(8..32))
                    .map(|_| rng.lowercase() as u8)
                    .collect_vec(),
            )
            .unwrap()
        })
        .collect_vec();
    let short = StringArray::from_iter_values(short_strings.iter());

    {
        let needle_scalar = StringArray::new_scalar(&"abc");
        let llvm = arrow_compile_compute::cmp::contains(&short, b"abc").unwrap();
        let arrow = arrow_string::like::contains(&short, &needle_scalar).unwrap();
        assert_eq!(arrow, llvm);

        c.bench_function(
            "cmp::contains(array(utf8), scalar(utf8)) 1m rows/llvm warm",
            |b| b.iter(|| arrow_compile_compute::cmp::contains(&short, b"abc").unwrap()),
        );
        c.bench_function(
            "cmp::contains(array(utf8), scalar(utf8)) 1m rows/arrow",
            |b| b.iter(|| arrow_string::like::contains(&short, &needle_scalar).unwrap()),
        );
    }

    // One benchmark per LIKE pattern shape: each hits a different specialized
    // path in this crate's pattern compiler and in arrow's LIKE engine.
    for (shape, pattern) in [
        ("'abc%'", "abc%"),
        ("'%xyz'", "%xyz"),
        ("'%abc%'", "%abc%"),
        ("'%abc%xyz'", "%abc%xyz"),
        ("'%abc%xyz%'", "%abc%xyz%"),
    ] {
        let pattern_scalar = StringArray::new_scalar(pattern);
        let llvm = arrow_compile_compute::cmp::like(&short, pattern.as_bytes(), None).unwrap();
        let arrow = arrow_string::like::like(&short, &pattern_scalar).unwrap();
        assert_eq!(arrow, llvm, "pattern {pattern}");

        let name = format!("cmp::like(array(utf8), {shape}) 1m rows");
        c.bench_function(&format!("{name}/llvm warm"), |b| {
            b.iter(|| arrow_compile_compute::cmp::like(&short, pattern.as_bytes(), None).unwrap())
        });
        c.bench_function(&format!("{name}/arrow"), |b| {
            b.iter(|| arrow_string::like::like(&short, &pattern_scalar).unwrap())
        });
    }
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
