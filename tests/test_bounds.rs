use arrow_array::{Int32Array, Scalar};
use proptest::{prelude::any, proptest};

proptest! {
    #[test]
    fn test_bounds_scalar(
        values in proptest::collection::vec(any::<i32>(), 0..64),
        lb in any::<i32>(),
        ub in any::<i32>(),
    ) {
        let expected = values.iter().all(|value| lb <= *value && *value < ub);

        let values_array = Int32Array::from(values);
        let lb_scalar = Scalar::new(Int32Array::from(vec![lb]));
        let ub_scalar = Scalar::new(Int32Array::from(vec![ub]));

        let result = arrow_compile_compute::cmp::bounds(&values_array, &lb_scalar, &ub_scalar)
            .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bounds_array(
        rows in proptest::collection::vec((any::<i32>(), any::<i32>(), any::<i32>()), 0..64),
    ) {
        let expected = rows
            .iter()
            .all(|(value, lb, ub)| *lb <= *value && *value < *ub);

        let values_array = Int32Array::from(rows.iter().map(|(value, _, _)| *value).collect::<Vec<_>>());
        let lb_array = Int32Array::from(rows.iter().map(|(_, lb, _)| *lb).collect::<Vec<_>>());
        let ub_array = Int32Array::from(rows.iter().map(|(_, _, ub)| *ub).collect::<Vec<_>>());

        let result = arrow_compile_compute::cmp::bounds(&values_array, &lb_array, &ub_array)
            .unwrap();

        assert_eq!(result, expected);
    }
}
