use std::collections::HashSet;

use arrow_array::{Array, Float16Array, Float32Array, Float64Array, Int32Array, StringArray};
use arrow_compile_compute::SortOptions;
use itertools::Itertools;
use proptest::{prelude::any, proptest};

proptest! {
    #[test]
    fn test_sort_i32_nonull(data: HashSet<i32>) {
        let mut rng = fastrand::Rng::with_seed(42);
        let mut data = data.into_iter().collect_vec();
        rng.shuffle(&mut data);

        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| data[*i as usize]);

        let res = arrow_compile_compute::sort::sort_to_indices(
            &Int32Array::from(data),
            SortOptions::default())
        .unwrap();

        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, p);
    }

    #[test]
    fn test_sort_i32_stable_nonull(mut data: Vec<i32>) {
        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| data[*i as usize]);

        let res = arrow_compile_compute::sort::sort_to_indices(
            &Int32Array::from(data),
            SortOptions::default())
        .unwrap();

        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, p);
    }

    #[test]
    fn test_sort_str(data: Vec<String>) {
        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| &data[*i as usize]);


        let res = arrow_compile_compute::sort::sort_to_indices(
            &StringArray::from(data),
            SortOptions::default())
        .unwrap();

        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, p);
    }


    #[test]
    fn test_sort_f16(data: Vec<f32>) {
        let data = data.into_iter().map(|x| half::f16::from_f32(x)).collect_vec();
        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| float_ord::FloatOrd(data[*i as usize].to_f32()));

        let res = arrow_compile_compute::sort::sort_to_indices(
            &Float16Array::from(data.clone()),
            SortOptions::default())
        .unwrap();
        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(res, p);
    }


    #[test]
    fn test_sort_f32(data: Vec<f32>) {
        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| float_ord::FloatOrd(data[*i as usize]));

        let res = arrow_compile_compute::sort::sort_to_indices(
            &Float32Array::from(data.clone()),
            SortOptions::default())
        .unwrap();
        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(res, p);
    }

    #[test]
    fn test_sort_f64(data: Vec<f64>) {
        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| float_ord::FloatOrd(data[*i as usize]));

        let res = arrow_compile_compute::sort::sort_to_indices(
            &Float64Array::from(data.clone()),
            SortOptions::default())
        .unwrap();
        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(res, p);
    }

    #[test]
    fn test_lower_bound_proptest(
        mut sorted in proptest::collection::vec(any::<i32>(), 0..40),
        keys in proptest::collection::vec(any::<i32>(), 0..40),
    ) {
        sorted.sort();
        let expected: Vec<u32> = keys
            .iter()
            .map(|key| sorted.partition_point(|v| *v < *key) as u32)
            .collect();

        let sorted_array = Int32Array::from(sorted.clone());
        let key_array = Int32Array::from(keys.clone());
        let key_refs: [&dyn Array; 1] = [&key_array];
        let sorted_refs: [&dyn Array; 1] = [&sorted_array];
        let options = [SortOptions::default()];

        let result = arrow_compile_compute::sort::lower_bound(&key_refs, &sorted_refs, &options)
            .unwrap();
        let indices = result.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(indices, expected);
    }
}

#[test]
fn test_lower_bound_single_column() {
    let sorted = Int32Array::from(vec![1, 3, 5, 5, 7]);
    let keys = Int32Array::from(vec![0, 2, 5, 6, 8]);

    let key_refs: [&dyn Array; 1] = [&keys];
    let sorted_refs: [&dyn Array; 1] = [&sorted];
    let options = [SortOptions::default()];

    let res = arrow_compile_compute::sort::lower_bound(&key_refs, &sorted_refs, &options).unwrap();
    let indices = res.into_iter().map(|x| x.unwrap()).collect_vec();
    assert_eq!(indices, vec![0, 1, 2, 4, 5]);
}

#[test]
fn test_lower_bound_multi_column() {
    let sorted_a = Int32Array::from(vec![1, 1, 2, 2, 3]);
    let sorted_b = Int32Array::from(vec![10, 20, 5, 15, 0]);
    let key_a = Int32Array::from(vec![0, 2, 2, 4]);
    let key_b = Int32Array::from(vec![0, 5, 12, 0]);

    let key_refs: [&dyn Array; 2] = [&key_a, &key_b];
    let sorted_refs: [&dyn Array; 2] = [&sorted_a, &sorted_b];
    let options = [SortOptions::default(), SortOptions::default()];

    let res = arrow_compile_compute::sort::lower_bound(&key_refs, &sorted_refs, &options).unwrap();
    let indices = res.into_iter().map(|x| x.unwrap()).collect_vec();
    assert_eq!(indices, vec![0, 2, 3, 5]);
}

#[test]
fn test_lower_bound_with_nulls() {
    let sorted = Int32Array::from(vec![Some(1), Some(3), None]);
    let keys = Int32Array::from(vec![Some(0), Some(3), None]);

    let key_refs: [&dyn Array; 1] = [&keys];
    let sorted_refs: [&dyn Array; 1] = [&sorted];
    let options = [SortOptions::default()];

    let res = arrow_compile_compute::sort::lower_bound(&key_refs, &sorted_refs, &options).unwrap();
    let indices = res.into_iter().map(|x| x.unwrap()).collect_vec();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn test_lower_bound_descending() {
    let sorted_values = vec![9, 7, 7, 5, 3];
    let key_values = vec![10, 7, 6, 0];

    let sorted = Int32Array::from(sorted_values.clone());
    let keys = Int32Array::from(key_values.clone());
    let key_refs: [&dyn Array; 1] = [&keys];
    let sorted_refs: [&dyn Array; 1] = [&sorted];
    let options = [SortOptions {
        descending: true,
        nulls_first: false,
    }];

    let res = arrow_compile_compute::sort::lower_bound(&key_refs, &sorted_refs, &options).unwrap();
    let indices = res.into_iter().map(|x| x.unwrap()).collect_vec();

    let expected: Vec<u32> = key_values
        .iter()
        .map(|key| sorted_values.iter().position(|v| *v <= *key).unwrap_or(sorted_values.len()) as u32)
        .collect();

    assert_eq!(indices, expected);
}

#[test]
fn test_lower_bound_nulls_first() {
    let sorted = Int32Array::from(vec![None, Some(1), Some(3)]);
    let keys = Int32Array::from(vec![Some(0), None, Some(2)]);

    let key_refs: [&dyn Array; 1] = [&keys];
    let sorted_refs: [&dyn Array; 1] = [&sorted];
    let options = [SortOptions {
        descending: false,
        nulls_first: true,
    }];

    let res = arrow_compile_compute::sort::lower_bound(&key_refs, &sorted_refs, &options).unwrap();
    let indices = res.into_iter().map(|x| x.unwrap()).collect_vec();

    assert_eq!(indices, vec![1, 0, 2]);
}
