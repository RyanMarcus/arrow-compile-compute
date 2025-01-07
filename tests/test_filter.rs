use arrow_array::{cast::AsArray, BooleanArray, Int32Array};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::DataType;
use proptest::proptest;

proptest! {
    #[test]
    fn test_filter_i32(arr: Vec<(i32, bool)>) {
        let arr1 = Int32Array::from_iter(arr.iter().map(|(i, _b)| Some(*i)));
        let filter = BooleanArray::from_iter(arr.iter().map(|(_i, b)| Some(*b)));

        let arrow_res: Int32Array = arrow_select::filter::filter(&arr1, &filter).unwrap().as_primitive().clone();
        let our_res: Int32Array = arrow_compile_compute::compute::filter(&arr1, &filter).unwrap().as_primitive().clone();

        assert_eq!(arrow_res, our_res);
    }

    #[test]
    fn test_filter_dict_i32(arr: Vec<(i8, bool)>) {
        let arr1 = Int32Array::from_iter(arr.iter().map(|(i, _b)| Some(*i as i32)));
        let arr1 = arrow_cast::cast(&arr1, &dictionary_data_type(DataType::Int16, DataType::Int32)).unwrap();

        let filter = BooleanArray::from_iter(arr.iter().map(|(_i, b)| Some(*b)));

        let arrow_res = arrow_select::filter::filter(&arr1, &filter).unwrap();
        let arrow_res: Int32Array = arrow_cast::cast(&arrow_res, &DataType::Int32).unwrap().as_primitive().clone();

        let our_res: Int32Array = arrow_compile_compute::compute::filter(&arr1, &filter).unwrap().as_primitive().clone();
        assert_eq!(arrow_res, our_res);
    }
}
