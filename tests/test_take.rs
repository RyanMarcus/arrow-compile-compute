use arrow_array::{
    cast::AsArray, types::Int32Type, Array, Int32Array, Int64Array, StringArray, UInt32Array,
    UInt64Array,
};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::DataType;
use proptest::proptest;

proptest! {
    #[test]
    fn test_take_i32_with_u32(arr: Vec<(i32, bool)>) {
        let data = Int32Array::from(arr.iter().map(|(value, _)| *value).collect::<Vec<i32>>());
        let indexes = UInt32Array::from(arr.iter().enumerate()
            .filter_map(|(i, (_, valid))| if *valid { Some(i as u32) } else { None })
            .collect::<Vec<u32>>());
        let results = Int32Array::from(arr.iter().filter_map(|(value, valid)| if *valid { Some(*value) } else { None })
            .collect::<Vec<i32>>());

        let expected = Int32Array::from(results);
        let actual: Int32Array = arrow_compile_compute::select::take(&data, &indexes).unwrap().as_primitive().clone();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_take_i32_with_u64(arr: Vec<(i32, bool)>) {
        let data = Int32Array::from(arr.iter().map(|(value, _)| *value).collect::<Vec<i32>>());
        let indexes = UInt64Array::from(arr.iter().enumerate()
            .filter_map(|(i, (_, valid))| if *valid { Some(i as u64) } else { None })
            .collect::<Vec<u64>>());
        let results = Int32Array::from(arr.iter().filter_map(|(value, valid)| if *valid { Some(*value) } else { None })
            .collect::<Vec<i32>>());

        let expected = Int32Array::from(results);
        let actual: Int32Array = arrow_compile_compute::select::take(&data, &indexes).unwrap().as_primitive().clone();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_take_i32_with_i32(arr: Vec<(i32, bool)>) {
        let data = Int32Array::from(arr.iter().map(|(value, _)| *value).collect::<Vec<i32>>());
        let indexes = Int32Array::from(arr.iter().enumerate()
            .filter_map(|(i, (_, valid))| if *valid { Some(i as i32) } else { None })
            .collect::<Vec<i32>>());
        let results = Int32Array::from(arr.iter().filter_map(|(value, valid)| if *valid { Some(*value) } else { None })
            .collect::<Vec<i32>>());

        let expected = Int32Array::from(results);
        let actual: Int32Array = arrow_compile_compute::select::take(&data, &indexes).unwrap().as_primitive().clone();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_take_i32_with_i64(arr: Vec<(i32, bool)>) {
        let data = Int32Array::from(arr.iter().map(|(value, _)| *value).collect::<Vec<i32>>());
        let indexes = Int64Array::from(arr.iter().enumerate()
            .filter_map(|(i, (_, valid))| if *valid { Some(i as i64) } else { None })
            .collect::<Vec<i64>>());
        let results = Int32Array::from(arr.iter().filter_map(|(value, valid)| if *valid { Some(*value) } else { None })
            .collect::<Vec<i32>>());

        let expected = Int32Array::from(results);
        let actual: Int32Array = arrow_compile_compute::select::take(&data, &indexes).unwrap().as_primitive().clone();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_take_str_with_i32(arr: Vec<(String, bool)>) {
        let data = StringArray::from(arr.iter().map(|(value, _)| value.clone()).collect::<Vec<String>>());
        let indexes = Int32Array::from(arr.iter().enumerate()
            .filter_map(|(i, (_, valid))| if *valid { Some(i as i32) } else { None })
            .collect::<Vec<i32>>());
        let results = arr.iter().filter_map(|(value, valid)| if *valid { Some(value.clone()) } else { None })
            .collect::<Vec<String>>();

        let expected = StringArray::from(results);
        let actual = arrow_compile_compute::select::take(&data, &indexes).unwrap();
        if expected.is_empty() {
            assert!(actual.is_empty());
        } else {
            let actual = actual.as_string::<i32>().clone();

            assert_eq!(expected.len(), actual.len());
            for (expected, actual) in expected.iter().zip(actual.iter()) {
                assert_eq!(expected, actual);
            }
        }
    }

    #[test]
    fn test_take_str_dict_with_i32(arr: Vec<(String, bool)>) {
        let data = StringArray::from(arr.iter().map(|(value, _)| value.clone()).collect::<Vec<String>>());
        let indexes = Int32Array::from(arr.iter().enumerate()
            .filter_map(|(i, (_, valid))| if *valid { Some(i as i32) } else { None })
            .collect::<Vec<i32>>());
        let results = arr.iter().filter_map(|(value, valid)| if *valid { Some(value.clone()) } else { None })
            .collect::<Vec<String>>();

        let dict = arrow_cast::cast::cast(&data, &dictionary_data_type(DataType::Int32, DataType::Utf8)).unwrap();

        let expected = StringArray::from(results);
        let actual = arrow_compile_compute::select::take(&dict, &indexes).unwrap();
        if expected.is_empty() {
            assert!(actual.is_empty());
        } else {
            let actual = actual.as_string::<i32>().clone();

            assert_eq!(expected.len(), actual.len());
            for (expected, actual) in expected.iter().zip(actual.iter()) {
                assert_eq!(expected, actual);
            }
        }
    }


}

#[test]
fn test_take_i32_large_vectorized() {
    let data = Int32Array::from((0..256).map(|v| v as i32).collect::<Vec<_>>());
    let idxes = UInt32Array::from((0..128u32).rev().collect::<Vec<_>>());

    let res = arrow_compile_compute::select::take(&data, &idxes).unwrap();

    let values = res.as_primitive::<Int32Type>().values().to_vec();
    let expected = (0..128).rev().map(|v| v as i32).collect::<Vec<_>>();
    assert_eq!(values, expected);
}
