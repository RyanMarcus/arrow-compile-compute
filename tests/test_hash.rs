use arrow_array::{Int16Array, Int32Array, Int64Array, Int8Array, UInt32Array};
use proptest::proptest;

proptest! {
    #[test]
    fn test_hash_i8(arr: Vec<i8>) {
        let arr1 = Int8Array::from(arr.clone());
        let result = arrow_compile_compute::compute::hash(&arr1).unwrap();
        assert_eq!(arr1.len(), result.len());
    }

    #[test]
    fn test_hash_i16(arr: Vec<i16>) {
        let arr1 = Int16Array::from(arr.clone());
        let result = arrow_compile_compute::compute::hash(&arr1).unwrap();
        assert_eq!(arr1.len(), result.len());
    }

    #[test]
    fn test_hash_i32(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());
        let result = arrow_compile_compute::compute::hash(&arr1).unwrap();
        assert_eq!(arr1.len(), result.len());
    }

    #[test]
    fn test_hash_u32(arr: Vec<u32>) {
        let arr1 = UInt32Array::from(arr.clone());
        let result = arrow_compile_compute::compute::hash(&arr1).unwrap();
        assert_eq!(arr1.len(), result.len());
    }

    #[test]
    fn test_hash_i64(arr: Vec<i64>) {
        let arr1 = Int64Array::from(arr.clone());
        let result = arrow_compile_compute::compute::hash(&arr1).unwrap();
        assert_eq!(arr1.len(), result.len());
    }
}
