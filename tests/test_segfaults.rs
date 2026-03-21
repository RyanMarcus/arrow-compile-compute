use arrow_array::{BooleanArray, Int32Array};
use arrow_compile_compute::{select, ArrowKernelError};

#[test]
fn filter_rejects_mismatched_lengths_before_oob_access() {
    let data = Int32Array::from(vec![123]);
    let filter = BooleanArray::from(vec![false; 1024]);

    let err = select::filter(&data, &filter).unwrap_err();
    assert!(matches!(err, ArrowKernelError::SizeMismatch));
}

#[test]
fn filter_rejects_out_of_range_true_bits_before_jit_loads() {
    let data = Int32Array::from(vec![123]);
    let mut mask = vec![false; 1024];
    mask[1023] = true;
    let filter = BooleanArray::from(mask);

    let err = select::filter(&data, &filter).unwrap_err();
    assert!(matches!(err, ArrowKernelError::SizeMismatch));
}
