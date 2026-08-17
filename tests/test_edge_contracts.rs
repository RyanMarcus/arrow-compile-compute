//! Edge-of-contract tests: sliced (offset) arrays through every fast path,
//! and invalid inputs that must produce errors rather than panics.

use arrow_array::{
    cast::AsArray, types::Int32Type, Array, BooleanArray, Int32Array, RunArray, StringArray,
    UInt64Array,
};
use arrow_compile_compute::{cmp, select, SortOptions};

fn sliced_strings() -> StringArray {
    let data: Vec<String> = (0..1000)
        .map(|i| format!("row{}abc{}", i, "x".repeat(i % 7)))
        .collect();
    let arr = StringArray::from(data.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    arr.slice(13, 700)
}

#[test]
fn sliced_string_like_matches_arrow() {
    let arr = sliced_strings();
    let ours = cmp::like(&arr, b"%abc%", None).unwrap();
    let arrow = arrow_string::like::like(&arr, &StringArray::new_scalar("%abc%")).unwrap();
    assert_eq!(ours, arrow);

    let ours = cmp::like(&arr, b"row1%", None).unwrap();
    let arrow = arrow_string::like::like(&arr, &StringArray::new_scalar("row1%")).unwrap();
    assert_eq!(ours, arrow);
}

#[test]
fn sliced_string_contains_matches_arrow() {
    let arr = sliced_strings();
    // row-dependent needle: matches differ across the slice boundary, so a
    // fast path ignoring the offset cannot pass by accident
    let ours = cmp::contains(&arr, b"row2").unwrap();
    let arrow =
        arrow_string::like::contains(&arr, &StringArray::new_scalar("row2")).unwrap();
    assert_eq!(ours, arrow);
    assert!(ours.true_count() > 0 && ours.true_count() < ours.len());

    let ours = cmp::like(&arr, b"%row2%", None).unwrap();
    let arrow = arrow_string::like::like(&arr, &StringArray::new_scalar("%row2%")).unwrap();
    assert_eq!(ours, arrow);
}

#[test]
fn sliced_bool_take_matches_arrow() {
    let bools = BooleanArray::from((0..500).map(|i| i % 3 == 0).collect::<Vec<_>>());
    let bools = bools.slice(7, 400);
    let idx = UInt64Array::from((0..399u64).rev().collect::<Vec<_>>());

    let ours = select::take(&bools, &idx).unwrap();
    let arrow = arrow_select::take::take(&bools, &idx, None).unwrap();
    assert_eq!(ours.as_boolean(), arrow.as_boolean());
}

#[test]
fn sliced_ree_logical_nulls_matches_arrow() {
    let values = Int32Array::from(
        (0..100)
            .map(|i| if i % 4 == 0 { None } else { Some(i) })
            .collect::<Vec<_>>(),
    );
    let ends = Int32Array::from((1..=100).map(|i| i * 3).collect::<Vec<_>>());
    let ree = RunArray::<Int32Type>::try_new(&ends, &values).unwrap();
    let ree = ree.slice(17, 200);

    let ours = arrow_compile_compute::logical_nulls(&ree).unwrap();
    let arrow = ree.logical_nulls();
    assert_eq!(
        ours.as_ref().map(|n| n.inner().clone()),
        arrow.as_ref().map(|n| n.inner().clone())
    );
}

#[test]
fn sliced_ree_filter_matches_arrow() {
    let values = Int32Array::from((0..100).collect::<Vec<_>>());
    let ends = Int32Array::from((1..=100).map(|i| i * 5).collect::<Vec<_>>());
    let ree = RunArray::<Int32Type>::try_new(&ends, &values).unwrap();
    let ree = ree.slice(3, 450);
    let mask = BooleanArray::from((0..450).map(|i| i % 2 == 0).collect::<Vec<_>>());

    let ours = select::filter(&ree, &mask).unwrap();
    let arrow = arrow_select::filter::filter(&ree, &mask).unwrap();
    let arrow = arrow_cast::cast(&arrow, &arrow_schema::DataType::Int32).unwrap();
    assert_eq!(ours.as_primitive::<Int32Type>(), arrow.as_primitive::<Int32Type>());
}

#[test]
fn sliced_nullable_sort_matches_arrow() {
    let data = Int32Array::from(
        (0..100_000)
            .map(|i| if i % 7 == 0 { None } else { Some((i * 31) % 1000) })
            .collect::<Vec<_>>(),
    );
    let data = data.slice(11, 99_000);

    // note: our SortOptions::default() is nulls-last; arrow's default is
    // nulls-first, so pass arrow explicit options that match ours
    let ours = arrow_compile_compute::sort::sort_to_indices(&data, SortOptions::default()).unwrap();
    let arrow = arrow_ord::sort::sort_to_indices(
        &data,
        Some(arrow_schema::SortOptions {
            descending: false,
            nulls_first: false,
        }),
        None,
    )
    .unwrap();
    // arrow's sort is not deterministic on ties, so compare the sorted values
    let ours_taken = arrow_select::take::take(&data, &ours, None).unwrap();
    let arrow_taken = arrow_select::take::take(&data, &arrow, None).unwrap();
    assert_eq!(
        ours_taken.as_primitive::<Int32Type>(),
        arrow_taken.as_primitive::<Int32Type>()
    );
}

#[test]
fn interleave_invalid_indices_do_not_panic() {
    let a = Int32Array::from(vec![1, 2, 3]);
    let b = Int32Array::from(vec![4, 5, 6]);
    // both invalid shapes must return Err — a panic fails the test harness
    assert!(
        select::interleave(&[&a, &b], &[(0, 0), (5, 0)]).is_err(),
        "interleave with bad array index returned Ok"
    );
    assert!(
        select::interleave(&[&a, &b], &[(0, 0), (1, 99)]).is_err(),
        "interleave with bad element index returned Ok"
    );
}

#[test]
fn nullable_filter_mask_matches_arrow() {
    use arrow_buffer::{BooleanBuffer, NullBuffer};
    let data = Int32Array::from(vec![10, 20, 30, 40, 50, 60]);
    // value bits all true, but slots 1 and 4 are null — arrow excludes them
    let values = BooleanBuffer::from(vec![true, true, true, true, true, true]);
    let validity = NullBuffer::from(vec![true, false, true, true, false, true]);
    let mask = BooleanArray::new(values, Some(validity));

    let arrow = arrow_select::filter::filter(&data, &mask).unwrap();
    let ours = select::filter(&data, &mask).unwrap();
    assert_eq!(
        ours.as_primitive::<Int32Type>(),
        arrow.as_primitive::<Int32Type>()
    );
}
