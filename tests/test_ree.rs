use arrow_array::{Array, Int32Array, RunArray};
use arrow_schema::DataType;
use proptest::prelude::any;
use proptest::proptest;

use proptest::collection::vec;
use proptest::strategy::Strategy;

fn run_end_encoding_arrays() -> impl Strategy<Value = (Vec<i32>, Vec<i32>)> {
    (1usize..1000).prop_flat_map(|len| {
        let run_lenghts = vec(0_usize..100_usize, len);
        let run_ends = run_lenghts.prop_map(|v| {
            v.iter()
                .map(|x| x + 1)
                .scan(0, |acc, x| {
                    *acc += x;
                    Some(*acc as i32)
                })
                .collect::<Vec<i32>>()
        });
        let values = vec(any::<i32>(), len);

        (run_ends, values)
    })
}

proptest! {
    #[test]
    fn test_ree_to_prim((re, vals) in run_end_encoding_arrays()) {
        let re = Int32Array::from(re);
        let vals = Int32Array::from(vals);
        let run_array = RunArray::try_new(&re, &vals).unwrap();

        let arrow_casted = Int32Array::from_iter(run_array.downcast::<Int32Array>().unwrap());
        let our_casted = arrow_compile_compute::cast::cast(&run_array, &DataType::Int32).unwrap();

        assert_eq!(&arrow_casted as &dyn Array, &our_casted);
    }
}
