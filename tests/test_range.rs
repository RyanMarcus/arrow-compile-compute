use arrow_array::{Int32Array, Int64Array, UInt64Array};
use arrow_schema::DataType;
use proptest::proptest;

use arrow_compile_compute::dictionary_data_type;

proptest! {
    #[test]
    fn test_range_i64(values: Vec<i64>) {
        let array = Int64Array::from(values.clone());
        let result = arrow_compile_compute::compute::range(&array);

        if values.is_empty() {
            assert!(result.is_err(), "expected an error for empty input");
        } else {
            let (range, min) = result.unwrap();
            let expected_min = *values.iter().min().unwrap() as i128;
            let expected_max = *values.iter().max().unwrap() as i128;
            let expected_range = (expected_max - expected_min) as u128;

            assert_eq!(min, expected_min);
            assert_eq!(range, expected_range);
        }
    }

    #[test]
    fn test_range_u64(values: Vec<u64>) {
        let array = UInt64Array::from(values.clone());
        let result = arrow_compile_compute::compute::range(&array);

        if values.is_empty() {
            assert!(result.is_err(), "expected an error for empty input");
        } else {
            let (range, min) = result.unwrap();
            let expected_min = *values.iter().min().unwrap() as i128;
            let expected_max = *values.iter().max().unwrap() as u128;
            let expected_range = expected_max - (expected_min as u128);

            assert_eq!(min, expected_min);
            assert_eq!(range, expected_range);
        }
    }

    #[test]
    fn test_range_dictionary(values: Vec<i32>) {
        let primitive = Int32Array::from(values.clone());
        let dt = dictionary_data_type(DataType::Int32, DataType::Int32);
        let dictionary = arrow_cast::cast(&primitive, &dt).unwrap();
        let result = arrow_compile_compute::compute::range(dictionary.as_ref());

        if values.is_empty() {
            assert!(result.is_err(), "expected an error for empty input");
        } else {
            let (range, min) = result.unwrap();
            let expected_min = *values.iter().min().unwrap() as i128;
            let expected_max = *values.iter().max().unwrap() as i128;
            let expected_range = (expected_max - expected_min) as u128;

            assert_eq!(min, expected_min);
            assert_eq!(range, expected_range);
        }
    }
}
