use std::collections::HashMap;

use arrow_array::{
    cast::AsArray,
    types::{Int32Type, Int64Type},
    Array, Int32Array, UInt64Array,
};
use arrow_compile_compute::aggregate;
use arrow_compile_compute::aggregate::Aggregator;
use proptest::proptest;

#[test]
fn test_grouped_sum_merge() {
    let mut left = aggregate::sum(&arrow_schema::DataType::Int32).unwrap();
    left.ingest(
        &[&Int32Array::from(vec![10, 20, 30])],
        &UInt64Array::from(vec![0, 1, 0]),
    )
    .unwrap();

    let mut right = aggregate::sum(&arrow_schema::DataType::Int32).unwrap();
    right
        .ingest(
            &[&Int32Array::from(vec![5, 7, 11])],
            &UInt64Array::from(vec![1, 2, 2]),
        )
        .unwrap();

    left.merge(*right).unwrap();
    let result = left.finish().unwrap();
    assert_eq!(result.as_primitive::<Int64Type>().values(), &[40, 25, 18]);
}

proptest! {
    #[test]
    fn test_grouped_sum(arr: Vec<Option<i32>>) {
        let mut ticket_ht = HashMap::new();
        let mut tickets = Vec::with_capacity(arr.len());
        for value in &arr {
            let group = value.unwrap_or_default() % 7;
            let next_ticket = ticket_ht.len();
            tickets.push(*ticket_ht.entry(group).or_insert(next_ticket) as u64);
        }

        let mut expected = vec![0_i64; ticket_ht.len()];
        for (value, &ticket) in arr.iter().zip(&tickets) {
            if let Some(value) = value {
                expected[ticket as usize] += i64::from(*value);
            }
        }

        let data = Int32Array::from(arr);
        let mut agg = aggregate::sum(data.data_type()).unwrap();
        agg.ingest(&[&data], &UInt64Array::from(tickets)).unwrap();
        let result = agg.finish().unwrap();
        let result = result.as_primitive::<Int64Type>();

        assert_eq!(result.values(), expected.as_slice());
    }

    #[test]
    fn test_ungrouped_min(arr: Vec<i32>) {
        let min = arr.iter().copied().min();
        let arr = Int32Array::from(arr);

        let mut agg = aggregate::min(arr.data_type()).unwrap();
        if !arr.is_empty() {
            agg.ensure_capacity(1);
        }
        agg.ingest_ungrouped(&[&arr]).unwrap();
        let res = agg.finish().unwrap();
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
    fn test_ungrouped_min_nulls(arr: Vec<Option<i32>>) {
        // nulls must be skipped, matching arrow's min
        let arr = Int32Array::from(arr);
        let expected = arrow_arith::aggregate::min(&arr);

        let mut agg = aggregate::min(arr.data_type()).unwrap();
        if !arr.is_empty() {
            agg.ensure_capacity(1);
        }
        agg.ingest_ungrouped(&[&arr]).unwrap();
        let res = agg.finish().unwrap();
        let res = res.as_primitive::<Int32Type>();

        match expected {
            Some(v) => {
                assert_eq!(res.len(), 1);
                assert_eq!(res.value(0), v);
            }
            None => {
                // Empty or all-null: ours returns 0-length only for the empty case.
                if arr.is_empty() {
                    assert_eq!(res.len(), 0);
                }
            }
        }
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

        let mut agg = aggregate::count().unwrap();
        agg.ensure_capacity(ticket_ht.len());
        agg.ingest(&[&arr], &UInt64Array::from(tickets)).unwrap();
        let res = agg.finish().unwrap();
        assert_eq!(res.len(), ticket_ht.len(),
            "there were {} unique values, but {} outputs", ticket_ht.len(), res.len());
    }
}
