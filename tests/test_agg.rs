use std::collections::HashMap;

use arrow_array::{
    cast::AsArray,
    types::{Int32Type, Int64Type},
    Array, Int32Array,
};
use arrow_compile_compute::aggregate;
use proptest::proptest;

proptest! {
    #[test]
    fn test_ungrouped_sum(arr: Vec<i32>) {
        let sum = arr.iter().copied().map(|x| x as i64).sum::<i64>();
        let arr = Int32Array::from(arr);

        let mut agg = aggregate::sum(arr.data_type(), None).unwrap();
        agg.ingest_ungrouped(&arr);
        let res = agg.finish();
        let res = res.as_primitive::<Int64Type>();
        if arr.is_empty() {
            assert_eq!(res.len(), 0);
        } else {
            assert_eq!(res.len(), 1);
            assert_eq!(res.value(0), sum);
        }
    }

    #[test]
    fn test_ungrouped_min(arr: Vec<i32>) {
        let min = arr.iter().copied().min();
        let arr = Int32Array::from(arr);

        let mut agg = aggregate::min(arr.data_type(), None).unwrap();
        agg.ingest_ungrouped(&arr);
        let res = agg.finish();
        let res = res.as_primitive::<Int32Type>();

        match min {
            Some(v) => {
                assert_eq!(res.len(), 1, "expected len {}, got {}", 1, res.len());
                assert_eq!(res.value(0), v);
            }
            _ => {
                assert_eq!(res.len(), 0);
            }
        };

    }

    #[test]
    fn test_grouped_count(arr: Vec<i32>) {
        let mut ticket_ht = HashMap::new();
        let mut tickets = Vec::new();
        for el in arr.iter() {
            let curr_len = ticket_ht.len();
            tickets.push(*ticket_ht.entry(*el).or_insert(curr_len) as u64);
        }
        let arr = Int32Array::from(arr);

        let mut agg = aggregate::count(None).unwrap();
        agg.ingest_grouped(&tickets, &arr);
        let res = agg.finish();
        assert_eq!(res.len(), ticket_ht.len(),
            "there were {} unique values, but {} outputs", ticket_ht.len(), res.len());

        let agg = aggregate::count(None).unwrap();
        agg.ingest_grouped_atomic(&tickets, &arr).unwrap();
        let res = agg.finish();
        assert_eq!(res.len(), ticket_ht.len());
    }
}
