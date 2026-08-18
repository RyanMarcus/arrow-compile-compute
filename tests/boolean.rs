use arrow_array::{BooleanArray, Datum, Int32Array, Scalar};
use proptest::proptest;

proptest! {
    #[test]
    fn test_boolean_not_matches_arrow(arr: Vec<Option<bool>>) {
        let arr = BooleanArray::from(arr);
        let datum = &arr as &dyn Datum;
        let our_res = arrow_compile_compute::boolean::not(datum).unwrap();
        let arrow_res = arrow_arith::boolean::not(&arr).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_boolean_is_null_matches_arrow(arr: Vec<Option<i32>>) {
        let arr = Int32Array::from(arr);
        let datum = &arr as &dyn Datum;
        let our_res = arrow_compile_compute::boolean::is_null(datum).unwrap();
        let arrow_res = arrow_arith::boolean::is_null(&arr).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_boolean_is_not_null_matches_arrow(arr: Vec<Option<i32>>) {
        let arr = Int32Array::from(arr);
        let datum = &arr as &dyn Datum;
        let our_res = arrow_compile_compute::boolean::is_not_null(datum).unwrap();
        let arrow_res = arrow_arith::boolean::is_not_null(&arr).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_boolean_scalar_kernels(value: Option<bool>) {
        let arr = BooleanArray::from(vec![value]);
        let scalar = Scalar::new(arr.clone());

        let our_not = arrow_compile_compute::boolean::not(&scalar).unwrap();
        let arrow_not = arrow_arith::boolean::not(&arr).unwrap();
        assert_eq!(our_not, arrow_not);

        let our_is_null = arrow_compile_compute::boolean::is_null(&scalar).unwrap();
        let arrow_is_null = arrow_arith::boolean::is_null(&arr).unwrap();
        assert_eq!(our_is_null, arrow_is_null);

        let our_is_not_null = arrow_compile_compute::boolean::is_not_null(&scalar).unwrap();
        let arrow_is_not_null = arrow_arith::boolean::is_not_null(&arr).unwrap();
        assert_eq!(our_is_not_null, arrow_is_not_null);
    }
}
