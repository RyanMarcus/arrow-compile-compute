use arrow_array::{cast::AsArray, types::Int32Type, Int32Array, StringArray};
use itertools::Itertools;
use proptest::{prelude::any, proptest};

proptest! {
    #[test]
    fn test_partition_i32(pairs in proptest::collection::vec((any::<i32>(), 0..10), 0..60)) {
        let vals = Int32Array::from(pairs.iter().map(|(x, _)| *x).collect_vec());
        let idxs = Int32Array::from(pairs.iter().map(|(_, y)| *y).collect_vec());

        let res = arrow_compile_compute::select::partition(&vals, &idxs).unwrap();
        let expected_nparts = pairs
            .iter()
            .map(|(_, idx)| *idx as usize)
            .max()
            .map(|idx| idx + 1)
            .unwrap_or(0);

        assert_eq!(res.len(), expected_nparts);
        assert_eq!(res.iter().map(|x| x.len()).sum::<usize>(), pairs.len());

        let expected = (0..expected_nparts)
            .map(|part_idx| {
                pairs
                    .iter()
                    .filter(|(_, idx)| *idx as usize == part_idx)
                    .map(|(value, _)| *value)
                    .collect_vec()
            })
            .collect_vec();
        let actual = res
            .iter()
            .map(|part| part.as_primitive::<Int32Type>().values().to_vec())
            .collect_vec();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_partition_str(pairs in proptest::collection::vec((any::<String>(), 0..10), 0..60)) {
        let vals = StringArray::from(pairs.iter().map(|(x, _)| x.clone()).collect_vec());
        let idxs = Int32Array::from(pairs.iter().map(|(_, y)| *y).collect_vec());

        let res = arrow_compile_compute::select::partition(&vals, &idxs).unwrap();
        let expected_nparts = pairs
            .iter()
            .map(|(_, idx)| *idx as usize)
            .max()
            .map(|idx| idx + 1)
            .unwrap_or(0);

        assert_eq!(res.len(), expected_nparts);
        assert_eq!(res.iter().map(|x| x.len()).sum::<usize>(), pairs.len());

        let expected = (0..expected_nparts)
            .map(|part_idx| {
                pairs
                    .iter()
                    .filter(|(_, idx)| *idx as usize == part_idx)
                    .map(|(value, _)| value.clone())
                    .collect_vec()
            })
            .collect_vec();
        let actual = res
            .iter()
            .map(|part| {
                part.as_string::<i32>()
                    .iter()
                    .map(|value| value.unwrap().to_owned())
                    .collect_vec()
            })
            .collect_vec();

        assert_eq!(actual, expected);
    }
}
