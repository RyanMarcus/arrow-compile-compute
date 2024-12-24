use arrow_array::{Int32Array, UInt32Array};
use arrow_compile_compute::dictionary_data_type;
use arrow_schema::DataType;
use proptest::prelude::any;
use proptest::{prop_assume, proptest};

use proptest::collection::vec;
use proptest::strategy::Strategy;

fn vecs_of_equal_length() -> impl Strategy<Value = (Vec<i32>, Vec<i32>)> {
    (1usize..1000).prop_flat_map(|len| (vec(any::<i32>(), len), vec(any::<i32>(), len)))
}

fn u32_vecs_of_equal_length() -> impl Strategy<Value = (Vec<u32>, Vec<u32>)> {
    (1usize..1000).prop_flat_map(|len| {
        (
            vec(any::<u32>(), len.clone()),
            vec(any::<u32>(), len.clone()),
        )
    })
}

proptest! {
    #[test]
    fn test_prim_prim_i32_eq_self(arr: Vec<i32>) {
        let arr1 = Int32Array::from(arr.clone());
        let arr2 = Int32Array::from(arr.clone());

        let our_res = arrow_compile_compute::cmp::eq(&arr1, &arr2).unwrap();
        assert_eq!(our_res.len(), arr.len());
        assert_eq!(our_res.true_count(), arr.len());
    }

    #[test]
    fn test_prim_prim_i32_eq(arrs in vecs_of_equal_length()) {
        let (arr1, arr2) = arrs;
        prop_assume!(arr1.len() == arr2.len());
        let arr1 = Int32Array::from(arr1);
        let arr2 = Int32Array::from(arr2);

        let our_res = arrow_compile_compute::cmp::eq(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::eq(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_prim_prim_i32_neq(arrs in vecs_of_equal_length()) {
        let (arr1, arr2) = arrs;
        prop_assume!(arr1.len() == arr2.len());
        let arr1 = Int32Array::from(arr1);
        let arr2 = Int32Array::from(arr2);

        let our_res = arrow_compile_compute::cmp::neq(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::neq(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_prim_prim_i32_lt(arrs in vecs_of_equal_length()) {
        let (arr1, arr2) = arrs;
        prop_assume!(arr1.len() == arr2.len());
        let arr1 = Int32Array::from(arr1);
        let arr2 = Int32Array::from(arr2);

        let our_res = arrow_compile_compute::cmp::lt(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::lt(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_prim_prim_u32_lt(arrs in u32_vecs_of_equal_length()) {
        let (arr1, arr2) = arrs;
        prop_assume!(arr1.len() == arr2.len());
        let arr1 = UInt32Array::from(arr1);
        let arr2 = UInt32Array::from(arr2);

        let our_res = arrow_compile_compute::cmp::lt(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::lt(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_prim_prim_i32_lt_eq(arrs in vecs_of_equal_length()) {
        let (arr1, arr2) = arrs;
        prop_assume!(arr1.len() == arr2.len());
        let arr1 = Int32Array::from(arr1);
        let arr2 = Int32Array::from(arr2);

        let our_res = arrow_compile_compute::cmp::lt_eq(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::lt_eq(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_prim_prim_i32_gt(arrs in vecs_of_equal_length()) {
        let (arr1, arr2) = arrs;
        prop_assume!(arr1.len() == arr2.len());
        let arr1 = Int32Array::from(arr1);
        let arr2 = Int32Array::from(arr2);

        let our_res = arrow_compile_compute::cmp::gt(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::gt(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_prim_prim_i32_gt_eq(arrs in vecs_of_equal_length()) {
        let (arr1, arr2) = arrs;
        prop_assume!(arr1.len() == arr2.len());
        let arr1 = Int32Array::from(arr1);
        let arr2 = Int32Array::from(arr2);

        let our_res = arrow_compile_compute::cmp::gt_eq(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::gt_eq(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }

    #[test]
    fn test_dict_dict_i32_eq(arrs in vecs_of_equal_length()) {
        let (arr1, arr2) = arrs;
        prop_assume!(arr1.len() == arr2.len());
        let dt = dictionary_data_type(DataType::Int32, DataType::Int32);
        let arr1 = arrow_cast::cast(&Int32Array::from(arr1), &dt).unwrap();
        let arr2 = arrow_cast::cast(&Int32Array::from(arr2), &dt).unwrap();

        let our_res = arrow_compile_compute::cmp::eq(&arr1, &arr2).unwrap();
        let arrow_res = arrow_ord::cmp::eq(&arr1, &arr2).unwrap();
        assert_eq!(our_res, arrow_res);
    }
}
