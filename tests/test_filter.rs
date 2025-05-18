use arrow_array::{cast::AsArray, types::Int32Type, BooleanArray, Int32Array, StringArray};
use arrow_compile_compute::{dictionary_data_type, select};
use arrow_schema::DataType;
use proptest::proptest;

proptest! {
    #[test]
    fn test_filter_i32(input: Vec<(i32, bool)>) {
        let mut data = Vec::new();
        let mut mask = Vec::new();
        let mut result = Vec::new();
        for (i, f) in input.into_iter() {
            data.push(i);
            mask.push(f);
            if f {
                result.push(i);
            }
        }

        let data = Int32Array::from(data);
        let mask = BooleanArray::from(mask);
        let result = Int32Array::from(result);

        let our_res = select::filter(&data, &mask).unwrap();
        let our_res = our_res.as_primitive::<Int32Type>();

        assert_eq!(our_res, &result);
    }

    #[test]
    fn test_filter_str(input: Vec<(String, bool)>) {
        let mut data = Vec::new();
        let mut mask = Vec::new();
        let mut result = Vec::new();
        for (i, f) in input.into_iter() {
            data.push(i.clone());
            mask.push(f);
            if f {
                result.push(i);
            }
        }

        let data = StringArray::from(data);
        let mask = BooleanArray::from(mask);
        let result = StringArray::from(result);

        let our_res = select::filter(&data, &mask).unwrap();
        let our_res = our_res.as_string::<i32>();

        assert_eq!(our_res, &result);
    }

    #[test]
    fn test_filter_str_dict(input: Vec<(String, bool)>) {
        let mut data = Vec::new();
        let mut mask = Vec::new();
        let mut result = Vec::new();
        for (i, f) in input.into_iter() {
            data.push(i.clone());
            mask.push(f);
            if f {
                result.push(i);
            }
        }

        let data = StringArray::from(data);
        let data = arrow_cast::cast::cast(&data, &dictionary_data_type(DataType::Int32, DataType::Utf8)).unwrap();
        let mask = BooleanArray::from(mask);
        let result = StringArray::from(result);

        let our_res = select::filter(&data, &mask).unwrap();
        let our_res = our_res.as_string::<i32>();

        assert_eq!(our_res, &result);
    }
}
