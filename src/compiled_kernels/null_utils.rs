use arrow_array::{make_array, ArrayRef, Datum};
use arrow_buffer::NullBuffer;

use crate::{logical_nulls, ArrowKernelError};

pub fn replace_nulls(arr: ArrayRef, nulls: Option<NullBuffer>) -> ArrayRef {
    let data = arr.to_data().into_builder().nulls(nulls);
    make_array(unsafe { data.build_unchecked() })
}

/// Given a list of arrays, intersect their null buffers and set `res`'s null
/// buffer to the result, returning a new array. Useful for kernel
/// implementations that combine multiple sources together and want to preserve
/// an "any input is null -> output is null" semantic.
///
/// ```
/// use std::sync::Arc;
///
/// use arrow_array::{cast::AsArray, types::Int32Type, Int32Array, Scalar};
/// use arrow_compile_compute::intersect_and_copy_nulls;
///
/// let mask = Int32Array::from(vec![Some(1), None, Some(3)]);
/// let null_scalar = Scalar::new(Int32Array::from(vec![None]));
/// let values = Arc::new(Int32Array::from(vec![10, 20, 30]));
///
/// let result = intersect_and_copy_nulls(&[&mask, &null_scalar], values).unwrap();
/// let result = result.as_primitive::<Int32Type>();
///
/// assert_eq!(result.iter().collect::<Vec<_>>(), vec![None, None, None]);
/// ```
pub fn intersect_and_copy_nulls(
    arrs: &[&dyn Datum],
    res: ArrayRef,
) -> Result<ArrayRef, ArrowKernelError> {
    let mut final_nb: Option<NullBuffer> = None;

    for arr in arrs {
        let (arr, is_scalar) = arr.get();
        if let Some(nb) = logical_nulls(arr)? {
            if is_scalar && nb.is_null(0) {
                final_nb = final_nb
                    .map(|final_nb| NullBuffer::new_null(final_nb.len()))
                    .or_else(|| Some(NullBuffer::new_null(res.len())));
            } else if !is_scalar {
                final_nb = final_nb
                    .map(|final_nb| NullBuffer::new(final_nb.inner() & nb.inner()))
                    .or_else(|| Some(nb.clone()))
            }
        }
    }

    Ok(replace_nulls(res, final_nb))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray,
        types::{Int16Type, Int32Type},
        DictionaryArray, Int16Array, Int32Array, RunArray, Scalar,
    };
    use arrow_buffer::NullBuffer;
    use itertools::Itertools;

    use crate::compiled_kernels::null_utils::replace_nulls;

    use super::intersect_and_copy_nulls;

    #[test]
    fn test_intersect_and_copy_nulls_arrays() {
        let arr1 = Int32Array::from(vec![Some(1), None, Some(3), Some(4)]);
        let arr2 = Int32Array::from(vec![Some(10), Some(20), None, Some(40)]);
        let res = Arc::new(Int32Array::from(vec![100, 200, 300, 400]));

        let result = intersect_and_copy_nulls(&[&arr1, &arr2], res).unwrap();
        let result = result.as_primitive::<Int32Type>();

        assert_eq!(
            result.iter().collect::<Vec<_>>(),
            vec![Some(100), None, None, Some(400)]
        );
    }

    #[test]
    fn test_intersect_and_copy_nulls_with_null_scalar() {
        let arr = Int32Array::from(vec![Some(1), None, Some(3)]);
        let scalar = Scalar::new(Int32Array::from(vec![None]));
        let res = Arc::new(Int32Array::from(vec![7, 8, 9]));

        let result = intersect_and_copy_nulls(&[&arr, &scalar], res).unwrap();
        let result = result.as_primitive::<Int32Type>();

        assert_eq!(result.iter().collect::<Vec<_>>(), vec![None, None, None]);
    }

    #[test]
    fn test_intersect_and_copy_nulls_scalar_scalar() {
        let lhs = Scalar::new(Int32Array::from(vec![None]));
        let rhs = Int32Array::new_scalar(5);
        let res = Arc::new(Int32Array::from(vec![42]));

        let result = intersect_and_copy_nulls(&[&lhs, &rhs], res).unwrap();
        let result = result.as_primitive::<Int32Type>();

        assert_eq!(result.iter().collect::<Vec<_>>(), vec![None]);
    }

    #[test]
    fn test_intersect_and_copy_nulls_dict_input_to_primitive_output_uses_logical_nulls() {
        let dict = DictionaryArray::<Int32Type>::new(
            Int32Array::from(vec![0, 1, 2, 2, 1, 0]),
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30)])),
        );
        let res = Arc::new(Int32Array::from(vec![100, 200, 300, 400, 500, 600]));

        let result = intersect_and_copy_nulls(&[&dict], res).unwrap();
        let result = result.as_primitive::<Int32Type>();

        assert_eq!(
            result.iter().collect::<Vec<_>>(),
            vec![Some(100), None, Some(300), Some(400), None, Some(600)]
        );
    }

    #[test]
    fn test_intersect_and_copy_nulls_ree_input_to_primitive_output_uses_logical_nulls() {
        let run_ends = Int16Array::from(vec![2i16, 5, 6, 9]);
        let values = Int32Array::from(vec![Some(10), None, Some(30), None]);
        let ree = RunArray::<Int16Type>::try_new(&run_ends, &values).unwrap();
        let res = Arc::new(Int32Array::from(vec![100, 101, 102, 103, 104, 105, 106, 107, 108]));

        let result = intersect_and_copy_nulls(&[&ree], res).unwrap();
        let result = result.as_primitive::<Int32Type>();

        assert_eq!(
            result.iter().collect::<Vec<_>>(),
            vec![
                Some(100),
                Some(101),
                None,
                None,
                None,
                Some(105),
                None,
                None,
                None,
            ]
        );
    }

    #[test]
    fn test_replace_nulls() {
        let arr = Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4]));
        let with_nulls = replace_nulls(
            arr,
            Some(NullBuffer::from(vec![true, false, true, true, true])),
        );
        let with_nulls = with_nulls.as_primitive::<Int32Type>();
        assert_eq!(
            with_nulls.iter().collect_vec(),
            vec![Some(0), None, Some(2), Some(3), Some(4)]
        );
    }

    #[test]
    fn test_replace_nulls_dict() {
        let arr = Arc::new(DictionaryArray::<Int32Type>::new(
            Int32Array::from(vec![0, 0, 1, 1, 2, 2]),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ));
        let with_nulls = replace_nulls(
            arr,
            Some(NullBuffer::from(vec![true, false, true, true, true, false])),
        );
        let with_nulls = with_nulls.as_dictionary::<Int32Type>();
        let with_nulls = with_nulls.downcast_dict::<Int32Array>().unwrap();
        assert_eq!(
            with_nulls.into_iter().collect_vec(),
            vec![Some(10), None, Some(20), Some(20), Some(30), None]
        );

        let arr = Arc::new(DictionaryArray::<Int32Type>::new(
            Int32Array::from(vec![0, 0, 1, 1, 2, 2]),
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30)])),
        ));
        let with_nulls = replace_nulls(
            arr,
            Some(NullBuffer::from(vec![true, false, true, true, true, false])),
        );
        let with_nulls = with_nulls.as_dictionary::<Int32Type>();
        let with_nulls = with_nulls.downcast_dict::<Int32Array>().unwrap();
        assert_eq!(
            with_nulls.into_iter().collect_vec(),
            vec![Some(10), None, None, None, Some(30), None]
        );
    }
}
