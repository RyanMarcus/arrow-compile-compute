use arrow_array::{
    builder::{Int32Builder, ListBuilder},
    Array,
};
use arrow_compile_compute::list;
use proptest::prelude::*;

fn build_list_array(rows: &[Option<Vec<i32>>]) -> arrow_array::ListArray {
    let mut builder = ListBuilder::new(Int32Builder::new());
    for row in rows {
        match row {
            Some(values) => {
                builder.values().append_slice(values);
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    builder.finish()
}

proptest! {
    #[test]
    fn list_len_matches_row_lengths(rows in prop::collection::vec(
        prop::option::of(prop::collection::vec(any::<i32>(), 0..16)),
        0..128
    )) {
        let arr = build_list_array(&rows);
        let result = list::len(&arr).unwrap();
        let actual = result.iter().collect::<Vec<_>>();
        let expected = rows
            .iter()
            .map(|row| row.as_ref().map(|values| values.len() as u64))
            .collect::<Vec<_>>();

        prop_assert_eq!(actual, expected);
        prop_assert_eq!(result.len(), arr.len());
    }
}
