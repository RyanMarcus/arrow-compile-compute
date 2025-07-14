use std::collections::HashSet;

use arrow_array::{Int32Array, StringArray};
use arrow_compile_compute::SortOptions;
use itertools::Itertools;
use proptest::proptest;

proptest! {
    #[test]
    fn test_sort_i32_nonull(data: HashSet<i32>) {
        let mut rng = fastrand::Rng::with_seed(42);
        let mut data = data.into_iter().collect_vec();
        rng.shuffle(&mut data);

        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| data[*i as usize]);

        let res = arrow_compile_compute::cmp::sort_to_indices(
            &Int32Array::from(data),
            SortOptions::default())
        .unwrap();

        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, p);
    }

    #[test]
    fn test_sort_i32_stable_nonull(mut data: Vec<i32>) {
        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| data[*i as usize]);

        let res = arrow_compile_compute::cmp::sort_to_indices(
            &Int32Array::from(data),
            SortOptions::default())
        .unwrap();

        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, p);
    }

    #[test]
    fn test_sort_str(data: Vec<String>) {
        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| &data[*i as usize]);


        let res = arrow_compile_compute::cmp::sort_to_indices(
            &StringArray::from(data),
            SortOptions::default())
        .unwrap();

        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, p);
    }
}
