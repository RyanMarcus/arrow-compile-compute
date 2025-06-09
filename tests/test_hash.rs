use arrow_array::{Array, Int32Array, Int64Array, StringArray};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::DataType;
use proptest::proptest;

proptest! {
    #[test]
    fn test_hash_i32(input: Vec<i32>) {
        let input = Int32Array::from(input);
        let hashed = arrow_compile_compute::compute::hash(&input).unwrap();
        assert_eq!(hashed.len(), input.len());
    }

    #[test]
    fn test_hash_i64(input: Vec<i64>) {
        let input = Int64Array::from(input);
        let hashed = arrow_compile_compute::compute::hash(&input).unwrap();
        assert_eq!(hashed.len(), input.len());
    }

    #[test]
    fn test_hash_i32_dict(input: Vec<i32>) {
        let input = Int32Array::from(input);
        let input = arrow_compile_compute::cast::cast(
            &input,
            &dictionary_data_type(DataType::Int64, DataType::Int32)
        ).unwrap();
        let hashed = arrow_compile_compute::compute::hash(&input).unwrap();
        assert_eq!(hashed.len(), input.len());
    }

    #[test]
    fn test_hash_str(input: Vec<String>) {
        let input = StringArray::from(input);
        let hashed = arrow_compile_compute::compute::hash(&input).unwrap();
        assert_eq!(hashed.len(), input.len());
    }

    #[test]
    fn test_hash_str_dict(input: Vec<String>) {
        let input = StringArray::from(input);
        let input = arrow_cast::cast::cast(
            &input,
            &dictionary_data_type(DataType::Int64, DataType::Utf8)
        ).unwrap();
        let hashed = arrow_compile_compute::compute::hash(&input).unwrap();
        assert_eq!(hashed.len(), input.len());
    }
}
