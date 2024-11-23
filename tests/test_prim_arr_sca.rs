use arrow_array::{Float32Array, Float64Array, Int32Array, Int64Array};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::DataType;
use proptest::proptest;

proptest! {
    #[test]
    fn test_i32_arr_sca_eq(arr: Vec<i32>, sca: i32) {
        let arr1 = Int32Array::from(arr);
        let arr2 = Int32Array::new_scalar(sca);

        let our_res = arrow_compile_compute::cmp::eq(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::eq(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_i64_arr_sca_eq(arr: Vec<i64>, sca: i64) {
        let arr1 = Int64Array::from(arr);
        let arr2 = Int64Array::new_scalar(sca);

        let our_res = arrow_compile_compute::cmp::eq(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::eq(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_f64_arr_sca_lt(arr: Vec<f64>, sca: f64) {
        let arr1 = Float64Array::from(arr);
        let arr2 = Float64Array::new_scalar(sca);

        let our_res = arrow_compile_compute::cmp::eq(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::eq(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_f32_arr_sca_lt(arr: Vec<f32>, sca: f32) {
        let arr1 = Float32Array::from(arr);
        let arr2 = Float32Array::new_scalar(sca);

        let our_res = arrow_compile_compute::cmp::eq(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::eq(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_i64_dict_sca_eq(arr: Vec<i64>, sca: i64) {
        let arr1 = Int64Array::from(arr);
        let arr1 = arrow_cast::cast(&arr1, &dictionary_data_type(DataType::Int64, DataType::Int64)).unwrap();
        let arr2 = Int64Array::new_scalar(sca);

        let our_res = arrow_compile_compute::cmp::eq(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::eq(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }
}
