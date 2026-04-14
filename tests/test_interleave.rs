use arrow_array::{cast::AsArray, Array, Int32Array};
use itertools::Itertools;
use proptest::proptest;

proptest! {
    #[test]
    fn test_interleave_nullable_i32(
        values in proptest::collection::vec(
            proptest::collection::vec(proptest::option::of(proptest::prelude::any::<i32>()), 0..10),
            1..5
        ),
        raw_indices in proptest::collection::vec((0usize..20, 0usize..20), 0..40)
    ) {
        let arrays = values
            .iter()
            .map(|value| Int32Array::from(value.clone()))
            .collect_vec();
        let refs = arrays.iter().map(|array| array as &dyn Array).collect_vec();

        let indices = raw_indices
            .into_iter()
            .filter_map(|(array_idx, element_idx)| {
                let array_idx = array_idx % refs.len();
                let len = refs[array_idx].len();
                if len == 0 {
                    None
                } else {
                    Some((array_idx, element_idx % len))
                }
            })
            .collect_vec();

        let expected = indices
            .iter()
            .map(|(array_idx, element_idx)| values[*array_idx][*element_idx])
            .collect_vec();

        let actual = arrow_compile_compute::select::interleave(&refs, &indices).unwrap();
        let actual = actual
            .as_primitive::<arrow_array::types::Int32Type>()
            .iter()
            .collect_vec();

        assert_eq!(actual, expected);
    }
}
