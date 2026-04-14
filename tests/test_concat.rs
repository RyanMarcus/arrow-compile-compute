use arrow_array::{cast::AsArray, types::Int32Type, Array, Int32Array, StringArray};
use itertools::Itertools;
use proptest::proptest;

proptest! {
    #[test]
    fn test_concat_i32(data_pre: Vec<i32>, mut data: Vec<Vec<i32>>) {
        data.push(data_pre);
        let combined = data.iter().flat_map(|v| v.iter().copied()).collect_vec();
        let arrays = data.into_iter().map(|v| Int32Array::from(v)).collect_vec();
        let refs = arrays.iter().map(|x| x as &dyn Array).collect_vec();

        let result = arrow_compile_compute::select::concat(&refs).unwrap();
        let result = result.as_primitive::<Int32Type>();
        let result = result.iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(result, combined);
    }

    #[test]
    fn test_concat_nullable_i32(data_pre: Vec<Option<i32>>, mut data: Vec<Vec<Option<i32>>>) {
        data.push(data_pre);
        let combined = data.iter().flat_map(|v| v.iter().copied()).collect_vec();
        let arrays = data.into_iter().map(|v| Int32Array::from(v)).collect_vec();
        let refs = arrays.iter().map(|x| x as &dyn Array).collect_vec();

        let result = arrow_compile_compute::select::concat(&refs).unwrap();
        let result = result.as_primitive::<Int32Type>();
        let result = result.iter().map(|x| x.clone()).collect_vec();

        assert_eq!(result, combined);
    }

    #[test]
    fn test_concat_nullable_str(data1: Vec<Option<String>>, data2: Vec<Option<String>>) {
        let data = vec![data1, data2];
        let combined = data.iter().flat_map(|v| v.iter().cloned()).collect_vec();
        let arrays = data.into_iter().map(|v| StringArray::from(v)).collect_vec();
        let refs = arrays.iter().map(|x| x as &dyn Array).collect_vec();

        let result = arrow_compile_compute::select::concat(&refs).unwrap();
        let result = result.as_string_view();
        let result = result.iter().map(|x| x.map(|s| s.to_string())).collect_vec();

        assert_eq!(result, combined);
    }
}

#[test]
fn test_concat_bools() {
    use arrow_array::{Array, BooleanArray};

    let d1 = BooleanArray::from(vec![true, false, true]);
    let d2 = BooleanArray::from(vec![false, true]);

    let res =
        arrow_compile_compute::select::concat(&[&d1 as &dyn Array, &d2 as &dyn Array]).unwrap();
    let res = res.as_any().downcast_ref::<BooleanArray>().unwrap();

    let vals = (0..res.len()).map(|i| res.value(i)).collect_vec();
    assert_eq!(vals, vec![true, false, true, false, true]);
}

#[test]
fn test_concat_i32_long_vectorized() {
    let d1 = Int32Array::from((0..96).collect_vec());
    let d2 = Int32Array::from((96..181).collect_vec());

    let res =
        arrow_compile_compute::select::concat(&[&d1 as &dyn Array, &d2 as &dyn Array]).unwrap();
    let res = res.as_primitive::<Int32Type>();
    let vals = res.iter().map(|x| x.unwrap()).collect_vec();

    assert_eq!(vals, (0..181).collect_vec());
}
