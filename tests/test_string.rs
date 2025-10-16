use arrow_array::{BooleanArray, StringArray};
use proptest::collection::vec;
use proptest::option;
use proptest::prelude::*;

// Limit vector sizes to keep proptests fast while covering edge cases.
const MAX_LEN: usize = 32;
const MAX_PATTERN_LEN: usize = 16;

fn optional_string_vec_strategy() -> impl Strategy<Value = Vec<Option<String>>> {
    vec(option::of(any::<String>()), 0..=MAX_LEN)
}

fn byte_pattern_strategy() -> impl Strategy<Value = Vec<u8>> {
    vec(any::<u8>(), 0..=MAX_PATTERN_LEN)
}

proptest! {
    #[test]
    fn proptest_starts_with_scalar(haystack in optional_string_vec_strategy(), needle in any::<String>()) {
        let haystack_arr = StringArray::from(haystack.clone());
        let needle_scalar = StringArray::new_scalar(needle.as_str());

        let result = arrow_compile_compute::cmp::starts_with(&haystack_arr, &needle_scalar).unwrap();

        let expected_bools = haystack
            .iter()
            .map(|maybe_value| maybe_value.as_ref().map(|value| value.starts_with(needle.as_str())).unwrap_or(false))
            .collect::<Vec<bool>>();
        let expected = BooleanArray::from(expected_bools);

        assert_eq!(result, expected);
    }

    #[test]
    fn proptest_ends_with_scalar(haystack in optional_string_vec_strategy(), needle in any::<String>()) {
        let haystack_arr = StringArray::from(haystack.clone());
        let needle_scalar = StringArray::new_scalar(needle.as_str());

        let result = arrow_compile_compute::cmp::ends_with(&haystack_arr, &needle_scalar).unwrap();

        let expected_bools = haystack
            .iter()
            .map(|maybe_value| maybe_value.as_ref().map(|value| value.ends_with(needle.as_str())).unwrap_or(false))
            .collect::<Vec<bool>>();
        let expected = BooleanArray::from(expected_bools);

        assert_eq!(result, expected);
    }


    #[test]
    fn proptest_contains_bytes(haystack in optional_string_vec_strategy(), needle in byte_pattern_strategy()) {
        let haystack_arr = StringArray::from(haystack.clone());
        let result = arrow_compile_compute::cmp::contains(&haystack_arr, &needle).unwrap();

        let expected_bools = haystack
            .iter()
            .map(|maybe_value| {
                maybe_value
                    .as_ref()
                    .map(|value| {
                        if needle.is_empty() {
                            true
                        } else {
                            let haystack_bytes = value.as_bytes();
                            haystack_bytes.windows(needle.len()).any(|window| window == needle.as_slice())
                        }
                    })
                    .unwrap_or(false)
            })
            .collect::<Vec<bool>>();
        let expected = BooleanArray::from(expected_bools);

        assert_eq!(result, expected);
    }
}
