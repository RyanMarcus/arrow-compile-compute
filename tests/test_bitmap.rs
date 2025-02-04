use arrow_array::{cast::AsArray, BooleanArray, Int32Array, UInt64Array};
use arrow_schema::DataType;
use proptest::proptest;

proptest! {
    #[test]
    fn test_bitmap_to_u64(arr: Vec<bool>) {
        let ba = BooleanArray::from(arr.clone());

        let our_res = arrow_compile_compute::cast::cast(&ba, &DataType::UInt64).unwrap().as_primitive().clone();

        let manual = arr.iter().enumerate().filter(|(_, &x)| x).map(|(i, _)| i as u64).collect::<Vec<u64>>();
        let manual = UInt64Array::from(manual);

        assert_eq!(our_res, manual);
    }
}
