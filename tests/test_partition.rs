use arrow_array::Int32Array;
use itertools::Itertools;
use proptest::{prelude::any, proptest};

proptest! {
    #[test]
    fn test_partition_i32(pairs in proptest::collection::vec((any::<i32>(), 0..10), 0..60)) {
        let vals = Int32Array::from(pairs.iter().map(|(x, _)| *x).collect_vec());
        let idxs = Int32Array::from(pairs.iter().map(|(_, y)| *y).collect_vec());

        let res = arrow_compile_compute::select::partition(&vals, &idxs, Some(10)).unwrap();
        assert_eq!(res.len(), 10);
        assert_eq!(res.iter().map(|x| x.len()).sum::<usize>(), pairs.len());
    }
}
