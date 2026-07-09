use arrow_array::{
    Float32Array, Float64Array, Int32Array, Int64Array, UInt32Array, UInt64Array,
};
use proptest::collection::vec;
use proptest::option;
use proptest::prelude::any;
use proptest::proptest;
use proptest::strategy::Strategy;

// Lengths span the SIMD body and the scalar tail loop.
fn i32_vec() -> impl Strategy<Value = Vec<i32>> {
    (1usize..1000).prop_flat_map(|len| vec(any::<i32>(), len))
}

fn i64_vec() -> impl Strategy<Value = Vec<i64>> {
    (1usize..1000).prop_flat_map(|len| vec(any::<i64>(), len))
}

fn u32_vec() -> impl Strategy<Value = Vec<u32>> {
    (1usize..1000).prop_flat_map(|len| vec(any::<u32>(), len))
}

fn u64_vec() -> impl Strategy<Value = Vec<u64>> {
    (1usize..1000).prop_flat_map(|len| vec(any::<u64>(), len))
}

fn opt_i32_vec() -> impl Strategy<Value = Vec<Option<i32>>> {
    (1usize..1000).prop_flat_map(|len| vec(option::of(any::<i32>()), len))
}

// Bounded, finite floats: avoids NaN/inf so element-wise equality is well defined.
fn f32_vec() -> impl Strategy<Value = Vec<f32>> {
    (1usize..1000).prop_flat_map(|len| vec(-1e18f32..1e18f32, len))
}

fn f64_vec() -> impl Strategy<Value = Vec<f64>> {
    (1usize..1000).prop_flat_map(|len| vec(-1e300f64..1e300f64, len))
}

proptest! {
    #[test]
    fn test_neg_i32_matches_arrow(arr in i32_vec()) {
        let arr = Int32Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_i64_matches_arrow(arr in i64_vec()) {
        let arr = Int64Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    // Unsigned wrapping negation: arrow's `neg_wrapping` supports unsigned
    // integers (returning the same unsigned type), so this diffs cleanly.
    // Only the *checked* `neg` rejects unsigned.
    #[test]
    fn test_neg_u32_matches_arrow(arr in u32_vec()) {
        let arr = UInt32Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_u64_matches_arrow(arr in u64_vec()) {
        let arr = UInt64Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_f32_matches_arrow(arr in f32_vec()) {
        let arr = Float32Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_f64_matches_arrow(arr in f64_vec()) {
        let arr = Float64Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_i32_nulls_match_arrow(arr in opt_i32_vec()) {
        // Nulls are propagated outside the compiled loop; check the validity
        // bitmap survives negation identically to arrow.
        let arr = Int32Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_i32_involution(arr in i32_vec()) {
        // -(-x) == x for all i32 (wrapping negation is its own inverse,
        // including i32::MIN).
        let original = Int32Array::from(arr);
        let once = arrow_compile_compute::arith::neg_wrapping(&original).unwrap();
        let twice = arrow_compile_compute::arith::neg_wrapping(&once).unwrap();
        assert_eq!(twice.as_ref(), &original);
    }
}
