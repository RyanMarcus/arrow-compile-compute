use arrow_array::{cast::AsArray, BooleanArray, Int32Array};
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


}
