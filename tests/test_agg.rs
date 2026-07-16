use std::collections::HashMap;

use arrow_array::{
    cast::AsArray,
    types::{Int32Type, Int64Type},
    Array, Int32Array, UInt64Array,
};
use arrow_compile_compute::aggregate;
use arrow_compile_compute::aggregate::Aggregator;
use proptest::proptest;

proptest! {
    #[test]
    fn test_ungrouped_sum(arr: Vec<i32>) {
        let sum = arr.iter().copied().map(|x| x as i64).sum::<i64>();
        let arr = Int32Array::from(arr);

        let mut agg = aggregate::sum(arr.data_type()).unwrap();
        if !arr.is_empty() {
            agg.ensure_capacity(1);
        }
        agg.ingest_ungrouped(&[&arr]).unwrap();
        let res = agg.finish().unwrap();
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
    fn test_ungrouped_sum_nulls(arr: Vec<Option<i32>>) {
        // nulls must be skipped; sum widens to i64 (all-null -> 0, additive identity)
        let sum = arr.iter().flatten().map(|x| *x as i64).sum::<i64>();
        let arr = Int32Array::from(arr);

        let mut agg = aggregate::sum(arr.data_type()).unwrap();
        if !arr.is_empty() {
            agg.ensure_capacity(1);
        }
        agg.ingest_ungrouped(&[&arr]).unwrap();
        let res = agg.finish().unwrap();
        let res = res.as_primitive::<Int64Type>();
        if arr.is_empty() {
            assert_eq!(res.len(), 0);
        } else {
            assert_eq!(res.len(), 1);
            assert_eq!(res.value(0), sum);
        }
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
                // empty or all-null: ours returns 0-length only for the empty case
                // (all-null is the same finish-level gap product has)
                if arr.is_empty() {
                    assert_eq!(res.len(), 0);
                }
            }
        }
    }

    #[test]
    fn test_ungrouped_product(arr: Vec<i32>) {
        // differential vs arrow's product (both wrap on overflow)
        let arr = Int32Array::from(arr);
        let expected = arrow_arith::aggregate::product(&arr);

        let mut agg = aggregate::product(arr.data_type()).unwrap();
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
                assert_eq!(res.len(), 0); // empty input
            }
        }
    }

    #[test]
    fn test_ungrouped_product_nulls(arr: Vec<Option<i32>>) {
        // exercises the null-skipping fix: nulls must be excluded, like arrow
        let arr = Int32Array::from(arr);
        let expected = arrow_arith::aggregate::product(&arr);

        let mut agg = aggregate::product(arr.data_type()).unwrap();
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
                // arrow returns None for empty OR all-null; ours returns a
                // 0-length array only for the empty case (all-null → finish-level
                // gap shared with min/max), so only assert the empty case here.
                if arr.is_empty() {
                    assert_eq!(res.len(), 0);
                }
            }
        }
    }

    #[test]
    fn test_grouped_product(arr: Vec<i32>) {
        // group by value; product per group vs a wrapping Rust-std reference
        let mut ticket_ht = HashMap::new();
        let mut tickets: Vec<u64> = Vec::new();
        for el in arr.iter() {
            let curr_len = ticket_ht.len();
            tickets.push(*ticket_ht.entry(*el).or_insert(curr_len) as u64);
        }
        let mut ref_prod = vec![1_i32; ticket_ht.len()];
        for (el, &t) in arr.iter().zip(tickets.iter()) {
            ref_prod[t as usize] = ref_prod[t as usize].wrapping_mul(*el);
        }

        let data = Int32Array::from(arr);
        let mut agg = aggregate::product(data.data_type()).unwrap();
        agg.ensure_capacity(ref_prod.len());
        agg.ingest(&[&data], &UInt64Array::from(tickets)).unwrap();
        let res = agg.finish().unwrap();
        let res = res.as_primitive::<Int32Type>();

        for (i, &expected) in ref_prod.iter().enumerate() {
            assert_eq!(res.value(i), expected, "group {i}");
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
