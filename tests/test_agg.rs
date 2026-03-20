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

    #[test]
    fn test_grouped_most_recent_merge(left: Vec<(i8, i32)>, right: Vec<(i8, i32)>) {
        let mut ticket_ht = HashMap::new();
        let mut expected = Vec::new();
        let mut left_expected = Vec::new();
        let mut right_expected = Vec::new();
        let mut left_tickets = Vec::new();
        let mut right_tickets = Vec::new();

        for (key, value) in left.iter().copied() {
            let curr_len = ticket_ht.len();
            let ticket = *ticket_ht.entry(key).or_insert(curr_len);
            if ticket == expected.len() {
                expected.push(value);
                left_expected.push(Some(value));
                right_expected.push(None);
            } else {
                expected[ticket] = value;
                left_expected[ticket] = Some(value);
            }
            left_tickets.push(ticket as u64);
        }

        for (key, value) in right.iter().copied() {
            let curr_len = ticket_ht.len();
            let ticket = *ticket_ht.entry(key).or_insert(curr_len);
            if ticket == expected.len() {
                expected.push(value);
                left_expected.push(None);
                right_expected.push(Some(value));
            } else {
                expected[ticket] = value;
                right_expected[ticket] = Some(value);
            }
            right_tickets.push(ticket as u64);
        }

        let left_arr = Int32Array::from(left.iter().map(|(_, value)| *value).collect::<Vec<_>>());
        let right_arr = Int32Array::from(right.iter().map(|(_, value)| *value).collect::<Vec<_>>());

        let mut sequential = aggregate::most_recent(left_arr.data_type(), None).unwrap();
        sequential.ingest_grouped(&left_tickets, &left_arr);
        sequential.ingest_grouped(&right_tickets, &right_arr);
        let sequential = sequential.finish();
        let sequential = sequential.as_primitive::<Int32Type>();
        assert_eq!(sequential.len(), expected.len());
        assert_eq!(sequential.values(), expected.as_slice());

        let mut left_agg = aggregate::most_recent(left_arr.data_type(), None).unwrap();
        left_agg.ingest_grouped(&left_tickets, &left_arr);

        let mut right_agg = aggregate::most_recent(right_arr.data_type(), None).unwrap();
        right_agg.ingest_grouped(&right_tickets, &right_arr);

        let merged = left_agg.merge(right_agg).finish();
        let merged = merged.as_primitive::<Int32Type>();
        assert_eq!(merged.len(), expected.len());
        for (idx, value) in merged.values().iter().copied().enumerate() {
            match (left_expected[idx], right_expected[idx]) {
                (Some(left), Some(right)) => {
                    assert!(
                        value == left || value == right,
                        "group {idx} should be either {left} or {right}, got {value}",
                    );
                }
                (Some(left), None) => assert_eq!(value, left, "group {idx}"),
                (None, Some(right)) => assert_eq!(value, right, "group {idx}"),
                (None, None) => panic!("group {idx} missing from both inputs"),
            }
        }
    }
}
