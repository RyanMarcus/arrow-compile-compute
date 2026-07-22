use arrow_array::{
    builder::{Int32Builder, LargeListBuilder, ListBuilder, StringBuilder},
    cast::AsArray,
    types::{Int16Type, Int32Type},
    BooleanArray, Int16Array, Int32Array, LargeListArray, ListArray, RunArray, UInt32Array,
};
use proptest::prelude::*;

type NullableI32Rows = Vec<Option<Vec<Option<i32>>>>;
type NullableStringRows = Vec<Option<Vec<Option<String>>>>;
type NestedI32Rows = Vec<Option<Vec<Option<Vec<Option<i32>>>>>>;

fn nullable_i32_rows_strategy() -> impl Strategy<Value = NullableI32Rows> {
    prop::collection::vec(
        prop::option::of(prop::collection::vec(prop::option::of(any::<i32>()), 0..10)),
        0..24,
    )
}

fn nullable_string_rows_strategy() -> impl Strategy<Value = NullableStringRows> {
    prop::collection::vec(
        prop::option::of(prop::collection::vec(
            prop::option::of("[a-zA-Z0-9]{0,12}"),
            0..10,
        )),
        0..24,
    )
}

fn nested_i32_rows_strategy() -> impl Strategy<Value = NestedI32Rows> {
    prop::collection::vec(
        prop::option::of(prop::collection::vec(
            prop::option::of(prop::collection::vec(prop::option::of(any::<i32>()), 0..8)),
            0..8,
        )),
        0..16,
    )
}

fn build_i32_lists(rows: &NullableI32Rows) -> ListArray {
    let mut builder = ListBuilder::new(Int32Builder::new());
    for row in rows {
        match row {
            Some(values) => {
                for value in values {
                    builder.values().append_option(*value);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    builder.finish()
}

fn build_large_string_lists(rows: &NullableStringRows) -> LargeListArray {
    let mut builder = LargeListBuilder::new(StringBuilder::new());
    for row in rows {
        match row {
            Some(values) => {
                for value in values {
                    builder.values().append_option(value.as_deref());
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    builder.finish()
}

fn build_nested_i32_lists(rows: &NestedI32Rows) -> ListArray {
    let mut builder = ListBuilder::new(ListBuilder::new(Int32Builder::new()));
    for row in rows {
        match row {
            Some(inner_rows) => {
                for inner_row in inner_rows {
                    match inner_row {
                        Some(values) => {
                            for value in values {
                                builder.values().values().append_option(*value);
                            }
                            builder.values().append(true);
                        }
                        None => builder.values().append(false),
                    }
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    builder.finish()
}

#[test]
fn filter_ree_list_preserves_child_nulls() {
    let rows = vec![Some(vec![Some(1), None]), Some(vec![Some(3), Some(4)])];
    let values = build_i32_lists(&rows);
    let run_ends = Int16Array::from(vec![2, 4]);
    let data = RunArray::<Int16Type>::try_new(&run_ends, &values).unwrap();
    let mask = BooleanArray::from(vec![true, false, true, true]);

    let actual = arrow_compile_compute::select::filter(&data, &mask).unwrap();
    let expected = build_i32_lists(&vec![rows[0].clone(), rows[1].clone(), rows[1].clone()]);

    assert_eq!(actual.as_list::<i32>(), &expected);
}

#[test]
fn take_nested_ree_list_preserves_child_nulls() {
    let rows = vec![Some(vec![Some(1), None])];
    let lists = build_i32_lists(&rows);
    let inner_run_ends = Int16Array::from(vec![2]);
    let inner = RunArray::<Int16Type>::try_new(&inner_run_ends, &lists).unwrap();
    let outer_run_ends = Int32Array::from(vec![1, 4]);
    let data = RunArray::<Int32Type>::try_new(&outer_run_ends, &inner).unwrap();
    let indices = UInt32Array::from(vec![3, 0, 2]);

    let actual = arrow_compile_compute::select::take(&data, &indices).unwrap();
    let expected = build_i32_lists(&vec![rows[0].clone(), rows[0].clone(), rows[0].clone()]);

    assert_eq!(actual.as_list::<i32>(), &expected);
}

proptest! {
    #[test]
    fn filter_list_i32_matches_arrow(
        rows in nullable_i32_rows_strategy(),
        raw_mask in prop::collection::vec(any::<bool>(), 0..24),
    ) {
        let data = build_i32_lists(&rows);
        let mask = BooleanArray::from(
            (0..rows.len())
                .map(|index| raw_mask.get(index).copied().unwrap_or(false))
                .collect::<Vec<_>>(),
        );

        let actual = arrow_compile_compute::select::filter(&data, &mask).unwrap();
        let expected = arrow_select::filter::filter(&data, &mask).unwrap();

        prop_assert_eq!(actual.data_type(), expected.data_type());
        prop_assert_eq!(actual.as_list::<i32>(), expected.as_list::<i32>());
    }

    #[test]
    fn take_list_i32_matches_arrow(
        rows in nullable_i32_rows_strategy(),
        raw_indices in prop::collection::vec(any::<u32>(), 0..40),
    ) {
        let data = build_i32_lists(&rows);
        let indices = if rows.is_empty() {
            UInt32Array::from(Vec::<u32>::new())
        } else {
            UInt32Array::from(
                raw_indices
                    .into_iter()
                    .map(|index| index % rows.len() as u32)
                    .collect::<Vec<_>>(),
            )
        };

        let actual = arrow_compile_compute::select::take(&data, &indices).unwrap();
        let expected = arrow_select::take::take(&data, &indices, None).unwrap();

        prop_assert_eq!(actual.data_type(), expected.data_type());
        prop_assert_eq!(actual.as_list::<i32>(), expected.as_list::<i32>());
    }

    #[test]
    fn filter_large_list_string_matches_arrow(
        rows in nullable_string_rows_strategy(),
        raw_mask in prop::collection::vec(any::<bool>(), 0..24),
    ) {
        let data = build_large_string_lists(&rows);
        let mask = BooleanArray::from(
            (0..rows.len())
                .map(|index| raw_mask.get(index).copied().unwrap_or(false))
                .collect::<Vec<_>>(),
        );

        let actual = arrow_compile_compute::select::filter(&data, &mask).unwrap();
        let expected = arrow_select::filter::filter(&data, &mask).unwrap();

        prop_assert_eq!(actual.data_type(), expected.data_type());
        prop_assert_eq!(actual.as_list::<i64>(), expected.as_list::<i64>());
    }

    #[test]
    fn take_large_list_string_matches_arrow(
        rows in nullable_string_rows_strategy(),
        raw_indices in prop::collection::vec(any::<u32>(), 0..40),
    ) {
        let data = build_large_string_lists(&rows);
        let indices = if rows.is_empty() {
            UInt32Array::from(Vec::<u32>::new())
        } else {
            UInt32Array::from(
                raw_indices
                    .into_iter()
                    .map(|index| index % rows.len() as u32)
                    .collect::<Vec<_>>(),
            )
        };

        let actual = arrow_compile_compute::select::take(&data, &indices).unwrap();
        let expected = arrow_select::take::take(&data, &indices, None).unwrap();

        prop_assert_eq!(actual.data_type(), expected.data_type());
        prop_assert_eq!(actual.as_list::<i64>(), expected.as_list::<i64>());
    }

    #[test]
    fn filter_nested_list_matches_arrow(
        rows in nested_i32_rows_strategy(),
        raw_mask in prop::collection::vec(any::<bool>(), 0..16),
    ) {
        let data = build_nested_i32_lists(&rows);
        let mask = BooleanArray::from(
            (0..rows.len())
                .map(|index| raw_mask.get(index).copied().unwrap_or(false))
                .collect::<Vec<_>>(),
        );

        let actual = arrow_compile_compute::select::filter(&data, &mask).unwrap();
        let expected = arrow_select::filter::filter(&data, &mask).unwrap();

        prop_assert_eq!(actual.data_type(), expected.data_type());
        prop_assert_eq!(actual.as_list::<i32>(), expected.as_list::<i32>());
    }

    #[test]
    fn take_nested_list_matches_arrow(
        rows in nested_i32_rows_strategy(),
        raw_indices in prop::collection::vec(any::<u32>(), 0..32),
    ) {
        let data = build_nested_i32_lists(&rows);
        let indices = if rows.is_empty() {
            UInt32Array::from(Vec::<u32>::new())
        } else {
            UInt32Array::from(
                raw_indices
                    .into_iter()
                    .map(|index| index % rows.len() as u32)
                    .collect::<Vec<_>>(),
            )
        };

        let actual = arrow_compile_compute::select::take(&data, &indices).unwrap();
        let expected = arrow_select::take::take(&data, &indices, None).unwrap();

        prop_assert_eq!(actual.data_type(), expected.data_type());
        prop_assert_eq!(actual.as_list::<i32>(), expected.as_list::<i32>());
    }
}
