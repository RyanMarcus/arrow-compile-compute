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

        let mut agg = aggregate::sum(arr.data_type()).unwrap();
        agg.ingest_ungrouped(&arr);
        let res = agg.finish();
        let res = res.as_primitive::<Int64Type>();
        assert_eq!(res.len(), 1);
        assert_eq!(res.value(0), sum);
    }

    #[test]
    fn test_ungrouped_min(arr: Vec<i32>) {
        let min = arr.iter().copied().min().unwrap_or(0);
        let arr = Int32Array::from(arr);

        let mut agg = aggregate::min(arr.data_type()).unwrap();
        agg.ingest_ungrouped(&arr);
        let res = agg.finish();
        let res = res.as_primitive::<Int32Type>();
        assert_eq!(res.len(), 1);
        assert_eq!(res.value(0), min);
    }
}
