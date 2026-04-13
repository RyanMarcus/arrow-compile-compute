use arrow_array::{builder::BinaryBuilder, Array, BinaryArray, Int32Array};
use arrow_compile_compute::{sort, SortOptions};
use itertools::Itertools;
use proptest::collection::vec;
use proptest::option;
use proptest::prelude::*;

const MAX_LEN: usize = 40;
const MAX_BINARY_LEN: usize = 8;

fn sort_options_strategy() -> impl Strategy<Value = SortOptions> {
    (any::<bool>(), any::<bool>()).prop_map(|(descending, nulls_first)| SortOptions {
        descending,
        nulls_first,
    })
}

fn binary_bytes_strategy() -> impl Strategy<Value = Vec<u8>> {
    vec(
        prop_oneof![Just(0u8), Just(255u8), any::<u8>()],
        0..=MAX_BINARY_LEN,
    )
}

fn row_data_strategy() -> impl Strategy<Value = (Vec<Option<i32>>, Vec<Option<Vec<u8>>>)> {
    (0..=MAX_LEN).prop_flat_map(|len| {
        (
            vec(option::of(any::<i32>()), len),
            vec(option::of(binary_bytes_strategy()), len),
        )
    })
}

fn binary_array_from_rows(rows: &[Option<Vec<u8>>]) -> BinaryArray {
    let mut builder = BinaryBuilder::new();
    for row in rows {
        match row {
            Some(bytes) => builder.append_value(bytes),
            None => builder.append_null(),
        }
    }
    builder.finish()
}

fn lexicographic_sort_indices(arr: &BinaryArray) -> Vec<u32> {
    let mut indices = (0..arr.len() as u32).collect_vec();
    indices.sort_by(|lhs, rhs| {
        arr.value(*lhs as usize)
            .cmp(arr.value(*rhs as usize))
            .then_with(|| lhs.cmp(rhs))
    });
    indices
}

proptest! {
    #[test]
    fn test_sort_norm_matches_multicol_sort_proptest(
        (ints, bytes) in row_data_strategy(),
        int_options in sort_options_strategy(),
        byte_options in sort_options_strategy(),
    ) {
        let ints_arr = Int32Array::from(ints);
        let bytes_arr = binary_array_from_rows(&bytes);

        let datum_inputs = [
            (&ints_arr as &dyn arrow_array::Datum, int_options),
            (&bytes_arr as &dyn arrow_array::Datum, byte_options),
        ];
        let array_inputs: [&dyn Array; 2] = [&ints_arr, &bytes_arr];
        let options = [int_options, byte_options];

        let normalized = sort::normalize_columns(&datum_inputs).unwrap();
        let expected = sort::multicol_sort_to_indices(&array_inputs, &options)
            .unwrap()
            .values()
            .iter()
            .copied()
            .collect_vec();

        prop_assert_eq!(lexicographic_sort_indices(&normalized), expected);
    }
}
