use std::sync::Arc;

use arrow_array::{
    cast::AsArray, Array, BooleanArray, FixedSizeListArray, Int32Array, Scalar, StringArray,
};
use arrow_schema::{DataType, Field};
use itertools::Itertools;

fn bool_lists(rows: &[[bool; 3]]) -> FixedSizeListArray {
    let values = BooleanArray::from(
        rows.iter()
            .flat_map(|row| row.iter().copied())
            .collect_vec(),
    );
    FixedSizeListArray::try_new(
        Arc::new(Field::new_list_field(DataType::Boolean, false)),
        3,
        Arc::new(values),
        None,
    )
    .unwrap()
}

fn string_lists(rows: &[&[&str; 2]]) -> FixedSizeListArray {
    let values = StringArray::from(
        rows.iter()
            .flat_map(|row| row.iter().map(|value| value.to_string()))
            .collect_vec(),
    );
    FixedSizeListArray::try_new(
        Arc::new(Field::new_list_field(DataType::Utf8, false)),
        2,
        Arc::new(values),
        None,
    )
    .unwrap()
}

fn collect_bool_rows(arr: &FixedSizeListArray) -> Vec<Vec<bool>> {
    (0..arr.len())
        .map(|idx| {
            let row = arr.value(idx);
            row.as_boolean().iter().map(|v| v.unwrap()).collect_vec()
        })
        .collect_vec()
}

fn collect_string_rows(arr: &FixedSizeListArray) -> Vec<Vec<String>> {
    (0..arr.len())
        .map(|idx| {
            let row = arr.value(idx);
            row.as_string::<i32>()
                .iter()
                .map(|v| v.unwrap().to_string())
                .collect_vec()
        })
        .collect_vec()
}

#[test]
fn filter_fixed_size_list_boolean() {
    let data = bool_lists(&[
        [true, false, true],
        [false, false, true],
        [true, true, false],
    ]);
    let mask = BooleanArray::from(vec![true, false, true]);

    let res = arrow_compile_compute::select::filter(&data, &mask).unwrap();
    let res = res.as_fixed_size_list();

    assert_eq!(
        collect_bool_rows(res),
        vec![vec![true, false, true], vec![true, true, false]]
    );
}

#[test]
fn take_fixed_size_list_utf8() {
    let data = string_lists(&[&["a", "b"], &["hello", "world"], &["x", "y"]]);
    let idx = Int32Array::from(vec![2, 0, 2]);

    let res = arrow_compile_compute::select::take(&data, &idx).unwrap();
    let res = res.as_fixed_size_list();

    assert_eq!(
        collect_string_rows(res),
        vec![
            vec!["x".to_string(), "y".to_string()],
            vec!["a".to_string(), "b".to_string()],
            vec!["x".to_string(), "y".to_string()],
        ]
    );
}

#[test]
fn concat_fixed_size_list_boolean_and_utf8() {
    let b1 = bool_lists(&[[true, false, true]]);
    let b2 = bool_lists(&[[false, true, false], [true, true, true]]);
    let res =
        arrow_compile_compute::select::concat(&[&b1 as &dyn Array, &b2 as &dyn Array]).unwrap();
    assert_eq!(
        collect_bool_rows(res.as_fixed_size_list()),
        vec![
            vec![true, false, true],
            vec![false, true, false],
            vec![true, true, true],
        ]
    );

    let s1 = string_lists(&[&["a", "b"]]);
    let s2 = string_lists(&[&["c", "d"], &["e", "f"]]);
    let res =
        arrow_compile_compute::select::concat(&[&s1 as &dyn Array, &s2 as &dyn Array]).unwrap();
    assert_eq!(
        collect_string_rows(res.as_fixed_size_list()),
        vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string(), "d".to_string()],
            vec!["e".to_string(), "f".to_string()],
        ]
    );
}

#[test]
fn compare_fixed_size_list_utf8_lexicographic() {
    let lhs = string_lists(&[&["a", "b"], &["a", "c"], &["b", "a"]]);
    let rhs = string_lists(&[&["a", "b"], &["b", "a"], &["a", "z"]]);

    let eq = arrow_compile_compute::cmp::eq(&lhs, &rhs).unwrap();
    let lt = arrow_compile_compute::cmp::lt(&lhs, &rhs).unwrap();

    assert_eq!(
        eq.iter().map(|v| v.unwrap()).collect_vec(),
        vec![true, false, false]
    );
    assert_eq!(
        lt.iter().map(|v| v.unwrap()).collect_vec(),
        vec![false, true, false]
    );
}

#[test]
fn compare_scalar_fixed_size_list_boolean_and_utf8() {
    let bool_data = bool_lists(&[[true, false, true], [false, false, true]]);
    let bool_scalar = Scalar::new(bool_lists(&[[true, false, true]]));
    let eq = arrow_compile_compute::cmp::eq(&bool_data, &bool_scalar).unwrap();
    assert_eq!(
        eq.iter().map(|v| v.unwrap()).collect_vec(),
        vec![true, false]
    );

    let string_data = string_lists(&[&["a", "b"], &["x", "y"]]);
    let string_scalar = Scalar::new(string_lists(&[&["a", "b"]]));
    let eq = arrow_compile_compute::cmp::eq(&string_data, &string_scalar).unwrap();
    assert_eq!(
        eq.iter().map(|v| v.unwrap()).collect_vec(),
        vec![true, false]
    );
}
