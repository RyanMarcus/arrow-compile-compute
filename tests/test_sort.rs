use std::collections::HashSet;

use arrow_array::{
    Array, Datum, Float16Array, Float32Array, Float64Array, Int32Array, Scalar, StringArray,
};
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

        let res = arrow_compile_compute::sort::sort_to_indices(
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

        let res = arrow_compile_compute::sort::sort_to_indices(
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


        let res = arrow_compile_compute::sort::sort_to_indices(
            &StringArray::from(data),
            SortOptions::default())
        .unwrap();

        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, p);
    }


    #[test]
    fn test_sort_f16(data: Vec<f32>) {
        let data = data.into_iter().map(|x| half::f16::from_f32(x)).collect_vec();
        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| float_ord::FloatOrd(data[*i as usize].to_f32()));

        let res = arrow_compile_compute::sort::sort_to_indices(
            &Float16Array::from(data.clone()),
            SortOptions::default())
        .unwrap();
        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(res, p);
    }


    #[test]
    fn test_sort_f32(data: Vec<f32>) {
        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| float_ord::FloatOrd(data[*i as usize]));

        let res = arrow_compile_compute::sort::sort_to_indices(
            &Float32Array::from(data.clone()),
            SortOptions::default())
        .unwrap();
        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(res, p);
    }

    #[test]
    fn test_sort_f64(data: Vec<f64>) {
        let mut p = (0..data.len() as u32).collect_vec();
        p.sort_by_key(|i| float_ord::FloatOrd(data[*i as usize]));

        let res = arrow_compile_compute::sort::sort_to_indices(
            &Float64Array::from(data.clone()),
            SortOptions::default())
        .unwrap();
        let res = res.into_iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(res, p);
    }
}

#[test]
fn test_lower_bound_interface() {
    let array = Int32Array::from(vec![1, 3, 5, 7, 9]);
    let arrays: Vec<&dyn Array> = vec![&array];
    let key = Scalar::new(Int32Array::from(vec![4]));
    let options = vec![SortOptions::default()];

    let idx = arrow_compile_compute::sort::lower_bound(
        arrays.as_slice(),
        &[&key as &dyn Datum],
        &options,
    )
    .unwrap();

    assert_eq!(idx, 2);
}
