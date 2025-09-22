use arrow_array::{
    Float32Array, Float64Array, Int32Array, Int64Array, StringArray, StringViewArray, UInt32Array,
    UInt64Array,
};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::DataType;
use itertools::Itertools;
use proptest::proptest;

proptest! {
    #[test]
    fn test_iter_i32(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());
        let v = arrow_compile_compute::iter::iter_nonnull_i64(&arr1).unwrap().collect_vec();
        assert_eq!(v, arr.iter().map(|&x| x as i64).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_i32_nulls(arr: Vec<Option<i32>>) {
        let arr1 = Int32Array::from(arr.clone());
        let v = arrow_compile_compute::iter::iter_nonnull_i64(&arr1).unwrap().collect_vec();
        assert_eq!(v, arr.iter().filter_map(|&x| x.map(|x| x as i64)).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_i32_nulls_option(arr: Vec<Option<i32>>) {
        let arr1 = Int32Array::from(arr.clone());
        let v = arrow_compile_compute::iter::iter_i64(&arr1).unwrap().collect_vec();
        assert_eq!(v, arr.iter().map(|x| x.map(|x| x as i64)).collect_vec())
    }

    #[test]
    fn test_iter_i64(arr: Vec<i64>) {
        let arr1 = Int64Array::from(arr.clone());
        let v = arrow_compile_compute::iter::iter_nonnull_i64(&arr1).unwrap().collect_vec();
        assert_eq!(v, arr.iter().map(|&x| x as i64).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_u32(arr: Vec<u32>) {
        let arr1 = UInt32Array::from(arr.clone());
        let v = arrow_compile_compute::iter::iter_nonnull_u64(&arr1).unwrap().collect_vec();
        assert_eq!(v, arr.iter().map(|&x| x as u64).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_u64(arr: Vec<u64>) {
        let arr1 = UInt64Array::from(arr.clone());
        let v = arrow_compile_compute::iter::iter_nonnull_u64(&arr1).unwrap().collect_vec();
        assert_eq!(v, arr.iter().map(|&x| x as u64).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_f32(arr: Vec<f32>) {
        let arr1 = Float32Array::from(arr.clone());
        let v = arrow_compile_compute::iter::iter_nonnull_f64(&arr1).unwrap().collect_vec();
        assert_eq!(v, arr.iter().map(|&x| x as f64).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_f64(arr: Vec<f64>) {
        let arr1 = Float64Array::from(arr.clone());
        let v = arrow_compile_compute::iter::iter_nonnull_f64(&arr1).unwrap().collect_vec();
        assert_eq!(v, arr.iter().map(|&x| x as f64).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_str(arr: Vec<String>) {
        let arr1 = StringArray::from(arr.clone());
        let v = arrow_compile_compute::iter::iter_nonnull_bytes(&arr1).unwrap()
            .map(|x| String::from_utf8(x.to_vec()).unwrap()).collect_vec();
        assert_eq!(v, arr);
    }

    #[test]
    fn test_iter_str_view(arr: Vec<String>) {
        let arr1 = StringViewArray::from(arr.clone());
        let v = arrow_compile_compute::iter::iter_nonnull_bytes(&arr1).unwrap()
            .map(|x| String::from_utf8(x.to_vec()).unwrap()).collect_vec();
        assert_eq!(v, arr);
    }

    #[test]
    fn test_iter_str_view_nulls(arr: Vec<Option<String>>) {
        let arr1 = StringViewArray::from(arr.clone());
        let v = arrow_compile_compute::iter::iter_nonnull_bytes(&arr1).unwrap()
            .map(|x| String::from_utf8(x.to_vec()).unwrap()).collect_vec();
        assert_eq!(v, arr.into_iter().filter_map(|x| x).collect_vec());
    }

    #[test]
    fn test_iter_i32_dict(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());
        let dt = dictionary_data_type(DataType::Int64, DataType::Int32);
        let arr1_dict = arrow_cast::cast(&arr1, &dt).unwrap();
        let v = arrow_compile_compute::iter::iter_nonnull_i64(&arr1_dict).unwrap().collect_vec();
        assert_eq!(v, arr.iter().map(|&x| x as i64).collect::<Vec<_>>());
    }

}
