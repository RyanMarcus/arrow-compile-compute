use arrow_array::{
    cast::AsArray,
    types::{Float32Type, Int32Type, UInt32Type},
    Float32Array, Int32Array, StringArray, UInt32Array,
};
use proptest::proptest;

proptest! {
    #[test]
    fn test_prim_i32_min(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());

        let our_res = arrow_compile_compute::compute::min(&arr1).unwrap().map(|arr| arr.as_primitive::<Int32Type>().value(0));
        let arrow_res = arrow_arith::aggregate::min(&arr1);
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_prim_u32_min(arr: Vec<u32>) {
        let arr1 = UInt32Array::from(arr.clone());

        let our_res = arrow_compile_compute::compute::min(&arr1).unwrap().map(|arr| arr.as_primitive::<UInt32Type>().value(0));
        let arrow_res = arrow_arith::aggregate::min(&arr1);
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_prim_f32_min(arr: Vec<f32>) {
        let arr1 = Float32Array::from(arr.clone());

        let our_res = arrow_compile_compute::compute::min(&arr1).unwrap().map(|arr| arr.as_primitive::<Float32Type>().value(0));
        let arrow_res = arrow_arith::aggregate::min(&arr1);
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_prim_i32_max(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());

        let our_res = arrow_compile_compute::compute::max(&arr1).unwrap().map(|arr| arr.as_primitive::<Int32Type>().value(0));
        let arrow_res = arrow_arith::aggregate::max(&arr1);
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_prim_u32_max(arr: Vec<u32>) {
        let arr1 = UInt32Array::from(arr.clone());

        let our_res = arrow_compile_compute::compute::max(&arr1).unwrap().map(|arr| arr.as_primitive::<UInt32Type>().value(0));
        let arrow_res = arrow_arith::aggregate::max(&arr1);
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_prim_f32_max(arr: Vec<f32>) {
        let arr1 = Float32Array::from(arr.clone());

        let our_res = arrow_compile_compute::compute::max(&arr1).unwrap().map(|arr| arr.as_primitive::<Float32Type>().value(0));
        let arrow_res = arrow_arith::aggregate::max(&arr1);
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_string_max(arr: Vec<String>) {
        let arr1 = StringArray::from(arr.clone());

        let true_max = arr.iter().max().cloned();
        let our_res = arrow_compile_compute::compute::max(&arr1).unwrap().map(|arr| {
            let arr = arr.as_string::<i32>();
            arr.value(0).to_string()
        });
        assert_eq!(true_max, our_res,
            "expected {:?} ({:?}), got {:?} ({:?})",
            true_max, true_max.as_ref().map(|s| s.bytes()),
            our_res, our_res.as_ref().map(|s| s.bytes())
        );
    }
}
