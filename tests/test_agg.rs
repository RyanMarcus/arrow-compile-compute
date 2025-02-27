use arrow_array::{
    cast::AsArray,
    types::{Float32Type, Int32Type, UInt32Type},
    Float32Array, Int32Array, StringArray, UInt32Array,
};
use arrow_buffer::NullBuffer;
use itertools::Itertools;
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
    fn test_prim_i32_min_nullable(arr: Vec<(bool, i32)>) {
        let mask = arr.iter().map(|(b, _)| *b).collect_vec();
        let data = arr.iter().map(|(_, i)| *i).collect_vec();
        let arr1 = Int32Array::from(data);
        let arr1 =
            Int32Array::try_new(arr1.values().clone(), Some(NullBuffer::from(mask))).unwrap();

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

    #[test]
    fn test_string_min(arr: Vec<String>) {
        let arr1 = StringArray::from(arr.clone());

        let true_min = arr.iter().min().cloned();
        let our_res = arrow_compile_compute::compute::min(&arr1).unwrap().map(|arr| {
            let arr = arr.as_string::<i32>();
            arr.value(0).to_string()
        });
        assert_eq!(true_min, our_res,
            "expected {:?} ({:?}), got {:?} ({:?})",
            true_min, true_min.as_ref().map(|s| s.bytes()),
            our_res, our_res.as_ref().map(|s| s.bytes())
        );
    }

    #[test]
    fn test_string_min_nullable(arr: Vec<(bool, String)>) {
        let mask = arr.iter().map(|(b, _)| *b).collect_vec();
        let data = arr.iter().map(|(_, i)| i.clone()).collect_vec();
        let arr1 = StringArray::from(data);
        let arr1 = StringArray::try_new(arr1.offsets().clone(), arr1.values().clone(), Some(NullBuffer::from(mask))).unwrap();
        let min = arr.iter().filter_map(|(a, b)| a.then(|| b)).min().cloned();


        let our_res = arrow_compile_compute::compute::min(&arr1).unwrap().map(|arr| arr.as_string::<i32>().value(0).to_string());
        assert_eq!(our_res, min);
    }
}
