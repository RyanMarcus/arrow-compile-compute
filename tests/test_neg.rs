use arrow_array::{Float32Array, Float64Array, Int32Array, Int64Array, UInt32Array, UInt64Array};
use proptest::proptest;

proptest! {
    #[test]
    fn test_neg_i32_matches_arrow(arr: Vec<i32>) {
        let arr = Int32Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_i64_matches_arrow(arr: Vec<i64>) {
        let arr = Int64Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    // Unsigned wrapping negation: arrow's `neg_wrapping` supports unsigned
    // integers (returning the same unsigned type), so this diffs cleanly.
    // Only the *checked* `neg` rejects unsigned.
    #[test]
    fn test_neg_u32_matches_arrow(arr: Vec<u32>) {
        let arr = UInt32Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_u64_matches_arrow(arr: Vec<u64>) {
        let arr = UInt64Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    // Full float domain: NaN, +/-inf, +/-0.0 are all generated here.
    #[test]
    fn test_neg_f32_matches_arrow(arr: Vec<f32>) {
        let arr = Float32Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_f64_matches_arrow(arr: Vec<f64>) {
        let arr = Float64Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_f64_nulls_match_arrow(arr: Vec<Option<f64>>) {
        let arr = Float64Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_i32_nulls_match_arrow(arr: Vec<Option<i32>>) {
        // Nulls are propagated outside the compiled loop; check the validity
        // bitmap survives negation identically to arrow.
        let arr = Int32Array::from(arr);
        let our_res = arrow_compile_compute::arith::neg_wrapping(&arr).unwrap();
        let arrow_res = arrow_arith::numeric::neg_wrapping(&arr).unwrap();
        assert_eq!(&our_res, &arrow_res);
    }

    #[test]
    fn test_neg_i32_involution(arr: Vec<i32>) {
        // -(-x) == x for all i32 (wrapping negation is its own inverse,
        // including i32::MIN).
        let original = Int32Array::from(arr);
        let once = arrow_compile_compute::arith::neg_wrapping(&original).unwrap();
        let twice = arrow_compile_compute::arith::neg_wrapping(&once).unwrap();
        assert_eq!(twice.as_ref(), &original);
    }
}
