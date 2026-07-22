use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{Int32Type, Int64Type},
    Array, Int32Array, LargeListArray, ListArray, PrimitiveArray, RunArray, StringArray,
    StringViewArray,
};
use arrow_compile_compute::{dictionary_data_type, run_end_data_type};
use arrow_schema::{DataType, Field};
use itertools::Itertools;
use proptest::proptest;

proptest! {
    #[test]
    fn test_prim_i32_cast_prim_i64(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());

        let our_res = arrow_compile_compute::cast::cast(&arr1, &DataType::Int64).unwrap();
        let arrow_res = arrow_cast::cast(&arr1, &DataType::Int64).unwrap();
        assert_eq!(our_res.len(), arr.len());

        assert_eq!(arrow_ord::cmp::eq(&our_res, &arrow_res).unwrap().true_count(), our_res.len())
    }

    #[test]
    fn test_prim_i32_nullable_cast_prim_i64(arr: Vec<Option<i32>>) {
        let arr1 = Int32Array::from(arr.clone());

        let our_res = arrow_compile_compute::cast::cast(&arr1, &DataType::Int64).unwrap();
        let arrow_res = arrow_cast::cast(&arr1, &DataType::Int64).unwrap();
        assert_eq!(our_res.len(), arrow_res.len());
    }

    #[test]
    fn test_list_cast_matches_arrow_or_preserves_shape(
        rows: Vec<Option<Vec<Option<i32>>>>,
        large_list: bool,
        target_leaf_index in 0usize..4,
    ) {
        let input: Arc<dyn Array> = if large_list {
            Arc::new(LargeListArray::from_iter_primitive::<Int32Type, _, _>(
                rows.clone(),
            ))
        } else {
            Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
                rows.clone(),
            ))
        };
        let target_leaf = match target_leaf_index {
            0 => DataType::Int32,
            1 => DataType::Int64,
            2 => DataType::Float32,
            _ => DataType::Float64,
        };
        let target_field = Arc::new(Field::new_list_field(target_leaf.clone(), true));
        let target = if large_list {
            DataType::LargeList(target_field)
        } else {
            DataType::List(target_field)
        };

        let our_res = arrow_compile_compute::cast::cast(input.as_ref(), &target).unwrap();
        if let Ok(arrow_res) = arrow_cast::cast(input.as_ref(), &target) {
            assert_eq!(&our_res, &arrow_res);
        } else {
            assert_eq!(our_res.data_type(), &target);
            assert_eq!(our_res.len(), input.len());
            if large_list {
                let input = input.as_list::<i64>();
                let output = our_res.as_list::<i64>();
                assert_eq!(output.values().data_type(), &target_leaf);
                for row in 0..input.len() {
                    assert_eq!(output.is_null(row), input.is_null(row));
                    assert_eq!(output.value_length(row), input.value_length(row));
                }
            } else {
                let input = input.as_list::<i32>();
                let output = our_res.as_list::<i32>();
                assert_eq!(output.values().data_type(), &target_leaf);
                for row in 0..input.len() {
                    assert_eq!(output.is_null(row), input.is_null(row));
                    assert_eq!(output.value_length(row), input.value_length(row));
                }
            }
        }
    }

    #[test]
    fn test_dict_i32_cast_prim_i64(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());
        let dt = dictionary_data_type(DataType::Int64, DataType::Int32);
        let arr1_dict = arrow_cast::cast(&arr1, &dt).unwrap();

        let our_res: Int32Array = arrow_compile_compute::cast::cast(&arr1_dict, &DataType::Int32).unwrap().as_primitive().clone();
        assert_eq!(our_res.len(), arr.len());
        assert_eq!(our_res, arr1);
    }

    #[test]
    fn test_prim_i32_cast_dict(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());

        let dt = dictionary_data_type(DataType::Int64, DataType::Int32);
        let our_res = arrow_compile_compute::cast::cast(&arr1, &dt).unwrap();
        let our_res = our_res.as_dictionary::<Int64Type>().downcast_dict::<PrimitiveArray<Int32Type>>().unwrap();
        let our_res = our_res.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(our_res, arr);
    }

    #[test]
    fn test_prim_i32_cast_ree(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());
        let dt = run_end_data_type(&DataType::Int64, &DataType::Int32);

        let our_res = arrow_compile_compute::cast::cast(&arr1, &dt).unwrap();
        let our_res = our_res.as_any().downcast_ref::<RunArray<Int64Type>>().unwrap();
        let our_res = our_res.downcast::<PrimitiveArray<Int32Type>>().unwrap();
        let our_res = our_res.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(our_res, arr);
    }

    #[test]
    fn test_str_cast_dict(arr: Vec<String>) {
        let arr1 = StringArray::from(arr.clone());
        let dt = dictionary_data_type(DataType::Int64, DataType::Utf8);

        let our_res = arrow_compile_compute::cast::cast(&arr1, &dt).unwrap();
        let our_res = our_res.as_dictionary::<Int64Type>();
        let our_res = our_res.downcast_dict::<StringArray>().unwrap();
        let our_res = our_res.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(arr, our_res);
    }

    #[test]
    fn test_str_view_cast_dict(arr: Vec<String>) {
        let arr1 = StringViewArray::from(arr.clone());
        let dt = dictionary_data_type(DataType::Int64, DataType::Utf8);

        let our_res = arrow_compile_compute::cast::cast(&arr1, &dt).unwrap();
        let our_res = our_res.as_dictionary::<Int64Type>();
        let our_res = our_res.downcast_dict::<StringArray>().unwrap();
        let our_res = our_res.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(arr, our_res);
    }

    #[test]
    fn test_str_cast_view(arr: Vec<String>) {
        let arr1 = StringArray::from(arr.clone());

        let as_view = arrow_compile_compute::cast::cast(&arr1, &DataType::Utf8View).unwrap();
        let as_view = as_view.as_string_view();

        assert_eq!(as_view.len(), arr.len());
        let view_res = as_view.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(arr, view_res);
    }
}
