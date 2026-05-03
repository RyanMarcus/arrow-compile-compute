use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{Int16Type, Int32Type},
    DictionaryArray, Int16Array, Int32Array, RunArray, Scalar,
};
use arrow_compile_compute::compute;
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_compute_min_i32(values in prop::collection::vec(any::<i32>(), 1..256)) {
        let expected = values.iter().copied().min().unwrap();
        let arr = Int32Array::from(values);

        let result = compute::min(&arr).unwrap();
        let result = result.as_primitive::<Int32Type>();
        prop_assert_eq!(result.len(), 1);
        prop_assert_eq!(result.value(0), expected);
    }

    #[test]
    fn test_compute_max_i32(values in prop::collection::vec(any::<i32>(), 1..256)) {
        let expected = values.iter().copied().max().unwrap();
        let arr = Int32Array::from(values);

        let result = compute::max(&arr).unwrap();
        let result = result.as_primitive::<Int32Type>();
        prop_assert_eq!(result.len(), 1);
        prop_assert_eq!(result.value(0), expected);
    }

    #[test]
    fn test_compute_argmin_i32(values in prop::collection::vec(any::<i32>(), 1..256)) {
        let min_value = values.iter().copied().min().unwrap();
        let expected = values.iter().position(|value| *value == min_value).unwrap() as u64;
        let arr = Int32Array::from(values);

        prop_assert_eq!(compute::argmin(&arr).unwrap(), Some(expected));
    }

    #[test]
    fn test_compute_argmax_i32(values in prop::collection::vec(any::<i32>(), 1..256)) {
        let max_value = values.iter().copied().max().unwrap();
        let expected = values.iter().position(|value| *value == max_value).unwrap() as u64;
        let arr = Int32Array::from(values);

        prop_assert_eq!(compute::argmax(&arr).unwrap(), Some(expected));
    }

    #[test]
    fn test_compute_min_nullable_i32(values in prop::collection::vec(prop::option::of(any::<i32>()), 0..256)) {
        let expected = values.iter().copied().flatten().min();
        let arr = Int32Array::from(values);

        let result = compute::min(&arr).unwrap();
        let result = result.as_primitive::<Int32Type>();
        prop_assert_eq!(result.len(), 1);
        prop_assert_eq!(result.iter().next().unwrap(), expected);
    }

    #[test]
    fn test_compute_max_nullable_i32(values in prop::collection::vec(prop::option::of(any::<i32>()), 0..256)) {
        let expected = values.iter().copied().flatten().max();
        let arr = Int32Array::from(values);

        let result = compute::max(&arr).unwrap();
        let result = result.as_primitive::<Int32Type>();
        prop_assert_eq!(result.len(), 1);
        prop_assert_eq!(result.iter().next().unwrap(), expected);
    }

    #[test]
    fn test_compute_argmin_nullable_i32(values in prop::collection::vec(prop::option::of(any::<i32>()), 0..256)) {
        let min_value = values.iter().copied().flatten().min();
        let expected = min_value.and_then(|min_value| {
            values.iter().position(|value| *value == Some(min_value)).map(|idx| idx as u64)
        });
        let arr = Int32Array::from(values);

        prop_assert_eq!(compute::argmin(&arr).unwrap(), expected);
    }

    #[test]
    fn test_compute_argmax_nullable_i32(values in prop::collection::vec(prop::option::of(any::<i32>()), 0..256)) {
        let max_value = values.iter().copied().flatten().max();
        let expected = max_value.and_then(|max_value| {
            values.iter().position(|value| *value == Some(max_value)).map(|idx| idx as u64)
        });
        let arr = Int32Array::from(values);

        prop_assert_eq!(compute::argmax(&arr).unwrap(), expected);
    }
}

#[test]
fn test_compute_min_dictionary_logical_nulls() {
    let keys = Int32Array::from(vec![0, 1, 2, 1, 0]);
    let values = Int32Array::from(vec![Some(10), None, Some(3)]);
    let dict = DictionaryArray::<Int32Type>::new(keys, Arc::new(values));

    let result = compute::min(&dict).unwrap();
    let result = result.as_primitive::<Int32Type>();

    assert_eq!(result.iter().collect::<Vec<_>>(), vec![Some(3)]);
}

#[test]
fn test_compute_argmax_run_end_logical_nulls() {
    let run_ends = Int16Array::from(vec![2i16, 4, 5]);
    let values = Int32Array::from(vec![Some(1), None, Some(9)]);
    let ree = RunArray::<Int16Type>::try_new(&run_ends, &values).unwrap();

    assert_eq!(compute::argmax(&ree).unwrap(), Some(4));
}

#[test]
fn test_compute_reductions_accept_scalar() {
    let scalar = Int32Array::new_scalar(7);

    let min = compute::min(&scalar).unwrap();
    assert_eq!(
        min.as_primitive::<Int32Type>().iter().collect::<Vec<_>>(),
        vec![Some(7)]
    );

    let max = compute::max(&scalar).unwrap();
    assert_eq!(
        max.as_primitive::<Int32Type>().iter().collect::<Vec<_>>(),
        vec![Some(7)]
    );

    assert_eq!(compute::argmin(&scalar).unwrap(), Some(0));
    assert_eq!(compute::argmax(&scalar).unwrap(), Some(0));
}

#[test]
fn test_compute_reductions_accept_null_scalar() {
    let scalar = Scalar::new(Int32Array::from(vec![None::<i32>]));

    let min = compute::min(&scalar).unwrap();
    assert_eq!(
        min.as_primitive::<Int32Type>().iter().collect::<Vec<_>>(),
        vec![None]
    );

    let max = compute::max(&scalar).unwrap();
    assert_eq!(
        max.as_primitive::<Int32Type>().iter().collect::<Vec<_>>(),
        vec![None]
    );

    assert_eq!(compute::argmin(&scalar).unwrap(), None);
    assert_eq!(compute::argmax(&scalar).unwrap(), None);
}
