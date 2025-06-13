use arrow_array::{cast::AsArray, types::Int64Type, Int32Array, StringArray};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::DataType;
use proptest::proptest;

proptest! {
    #[test]
    fn test_prim_i32_cast_prim_i64(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());

        let our_res = arrow_compile_compute::cast::cast(&arr1, &DataType::Int64).unwrap();
        let arrow_res = arrow_cast::cast(&arr1, &DataType::Int64).unwrap();
        assert_eq!(our_res.len(), arr.len());

        assert_eq!(arrow_ord::cmp::eq(&our_res, &arrow_res).unwrap().true_count(), our_res.len())
    }

    #[test]
    fn test_dict_i32_cast_prim_i64(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());
        let dt = dictionary_data_type(DataType::Int64, DataType::Int32);
        let arr1_dict = arrow_cast::cast(&arr1, &dt).unwrap();

        let our_res: Int32Array = arrow_compile_compute::cast::cast(&arr1_dict, &DataType::Int32).unwrap().as_primitive().clone();
        assert_eq!(our_res.len(), arr.len());
        assert_eq!(our_res, arr1);
    }

    #[test]
    fn test_prim_i32_cast_dict(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());
        let dt = dictionary_data_type(DataType::Int64, DataType::Int32);

        let arr_res = arrow_cast::cast(&arr1, &dt).unwrap();
        let arr_res = arr_res.as_dictionary::<Int64Type>();

        let our_res = arrow_compile_compute::cast::cast(&arr1, &dt).unwrap();
        let our_res = our_res.as_dictionary::<Int64Type>();
        assert_eq!(our_res.len(), arr_res.len());
    }

    #[test]
    fn test_str_cast_dict(arr: Vec<String>) {
        let arr1 = StringArray::from(arr.clone());
        let dt = dictionary_data_type(DataType::Int64, DataType::Utf8);

        let arr_res = arrow_cast::cast(&arr1, &dt).unwrap();
        let arr_res = arr_res.as_dictionary::<Int64Type>();

        let our_res = arrow_compile_compute::cast::cast(&arr1, &dt).unwrap();
        let our_res = our_res.as_dictionary::<Int64Type>();
        assert_eq!(our_res.len(), arr_res.len());
    }
}
