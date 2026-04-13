use std::cmp::Ordering;

use arrow_array::{
    cast::AsArray,
    types::{
        Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
        UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, BinaryArray, Datum, UInt32Array,
};
use itertools::Itertools;

use crate::{
    compiled_kernels::{sort_norm::normalize_columns, ArrowKernelError},
    PrimitiveType,
};

pub fn sort_multi_col(data: &[(&dyn Datum, SortOptions)]) -> Result<UInt32Array, ArrowKernelError> {
    if data.is_empty() {
        return Err(ArrowKernelError::ArgumentMismatch(
            "sort requires at least one column".to_string(),
        ));
    }

    let normalized = normalize_columns(data)?;
    Ok(sort_indices_from_normalized(&normalized))
}

pub fn sort_col(data: &dyn Datum, opts: SortOptions) -> Result<UInt32Array, ArrowKernelError> {
    let (arr, is_scalar) = data.get();
    if is_scalar {
        return Ok(UInt32Array::from(vec![0]));
    }

    let pt = PrimitiveType::for_arrow_type(arr.data_type());
    let arr = crate::arrow_interface::cast::cast(arr, &pt.as_arrow_type())?;

    match pt {
        PrimitiveType::I8 => sort_primitive::<Int8Type>(&arr, &opts),
        PrimitiveType::I16 => sort_primitive::<Int16Type>(&arr, &opts),
        PrimitiveType::I32 => sort_primitive::<Int32Type>(&arr, &opts),
        PrimitiveType::I64 => sort_primitive::<Int64Type>(&arr, &opts),
        PrimitiveType::U8 => sort_primitive::<UInt8Type>(&arr, &opts),
        PrimitiveType::U16 => sort_primitive::<UInt16Type>(&arr, &opts),
        PrimitiveType::U32 => sort_primitive::<UInt32Type>(&arr, &opts),
        PrimitiveType::U64 => sort_primitive::<UInt64Type>(&arr, &opts),
        PrimitiveType::F16 => sort_primitive::<Float16Type>(&arr, &opts),
        PrimitiveType::F32 => sort_primitive::<Float32Type>(&arr, &opts),
        PrimitiveType::F64 => sort_primitive::<Float64Type>(&arr, &opts),
        PrimitiveType::P64x2 => {
            let normed = normalize_columns(&[(&arr.as_ref(), opts.clone())])?;
            Ok(sort_indices_from_normalized(&normed))
        }
        PrimitiveType::List(_, _) => todo!(),
    }
}

pub fn top_k(
    data: &[(&dyn Datum, SortOptions)],
    k: usize,
) -> Result<UInt32Array, ArrowKernelError> {
    if data.is_empty() {
        return Err(ArrowKernelError::ArgumentMismatch(
            "top_k requires at least one column".to_string(),
        ));
    }

    let normed = normalize_columns(data)?;
    let mut indices = (0..normed.len() as u32).collect_vec();
    if k == 0 || indices.is_empty() {
        return Ok(UInt32Array::from(Vec::<u32>::new()));
    }

    let k = k.min(indices.len());
    if k < indices.len() {
        indices.select_nth_unstable_by(k, |a, b| cmp_normalized_indices(&normed, *a, *b));
        indices.truncate(k);
    }
    indices.sort_by(|a, b| cmp_normalized_indices(&normed, *a, *b));

    Ok(UInt32Array::from(indices))
}

pub fn lower_bound(
    keys: (&dyn Datum, SortOptions),
    sorted: &dyn Array,
) -> Result<UInt32Array, ArrowKernelError> {
    let (normalized_keys, normalized_sorted) = normalize_lower_bound_inputs(keys, sorted)?;
    let bounds = (0..normalized_keys.len())
        .map(|idx| {
            // safety: idx is within bounds by construction
            let key = unsafe { normalized_keys.value_unchecked(idx) };
            lower_bound_row(&normalized_sorted, key)
        })
        .collect_vec();
    Ok(UInt32Array::from(bounds))
}

fn sort_indices_from_normalized(keys: &BinaryArray) -> UInt32Array {
    let mut indices = (0..keys.len() as u32).collect_vec();
    indices.sort_by(|lhs, rhs| cmp_normalized_indices(keys, *lhs, *rhs));
    UInt32Array::from(indices)
}

/// Single column fast-path sort
fn sort_primitive<T: ArrowPrimitiveType>(
    data: &dyn Datum,
    opts: &SortOptions,
) -> Result<UInt32Array, ArrowKernelError> {
    let (values, is_scalar) = data.get();
    if is_scalar {
        return Ok(UInt32Array::from(vec![0]));
    }

    let values = values.as_primitive::<T>();
    let mut indices = (0..values.len() as u32).collect_vec();

    if let Some(nulls) = values.nulls() {
        indices.sort_by(|a, b| {
            let a = *a as usize;
            let b = *b as usize;
            match (nulls.is_valid(a), nulls.is_valid(b)) {
                (true, true) => unsafe {
                    // safety: by construction, all values in `indices` are in bounds
                    let cmp =
                        T::Native::compare(values.value_unchecked(a), values.value_unchecked(b));
                    if opts.descending {
                        cmp.reverse()
                    } else {
                        cmp
                    }
                },
                (true, false) => {
                    if opts.nulls_first {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                }
                (false, true) => {
                    if opts.nulls_first {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                }
                (false, false) => Ordering::Equal,
            }
        });
    } else {
        indices.sort_by(|a, b| {
            let a = *a as usize;
            let b = *b as usize;
            let cmp = unsafe {
                // safety: by construction, all values in `indices` are in bounds
                T::Native::compare(values.value_unchecked(a), values.value_unchecked(b))
            };
            if opts.descending {
                cmp.reverse()
            } else {
                cmp
            }
        });
    }

    Ok(UInt32Array::from(indices))
}

fn cmp_normalized_indices(keys: &BinaryArray, lhs: u32, rhs: u32) -> Ordering {
    keys.value(lhs as usize)
        .cmp(keys.value(rhs as usize))
        .then_with(|| lhs.cmp(&rhs))
}

fn lower_bound_row(sorted: &BinaryArray, key: &[u8]) -> u32 {
    let mut left = 0usize;
    let mut right = sorted.len();
    while left < right {
        let mid = left + (right - left) / 2;
        // safety: mid is always in [left, right), and right <= sorted.len()
        let mid_value = unsafe { sorted.value_unchecked(mid) };
        if mid_value < key {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left as u32
}

fn normalize_lower_bound_inputs(
    keys: (&dyn Datum, SortOptions),
    sorted: &dyn Array,
) -> Result<(BinaryArray, BinaryArray), ArrowKernelError> {
    let (key, opts) = keys;
    let key_arr = key.get().0;
    let key_pt = PrimitiveType::for_arrow_type(key_arr.data_type());
    let sorted_pt = PrimitiveType::for_arrow_type(sorted.data_type());
    let dom_pt = PrimitiveType::dominant(key_pt, sorted_pt).ok_or_else(|| {
        ArrowKernelError::UnsupportedArguments(format!(
            "could not compare values of {:?} and {:?}",
            key_arr.data_type(),
            sorted.data_type()
        ))
    })?;

    let (cast_key, cast_sorted): (ArrayRef, ArrayRef) = match dom_pt {
        PrimitiveType::P64x2 => (
            crate::arrow_interface::cast::cast(key_arr, key_arr.data_type())?,
            crate::arrow_interface::cast::cast(sorted, sorted.data_type())?,
        ),
        PrimitiveType::List(_, _) => {
            return Err(ArrowKernelError::UnsupportedArguments(
                "lower_bound does not yet support list columns".to_string(),
            ));
        }
        _ => {
            let arrow_type = dom_pt.as_arrow_type();
            (
                crate::arrow_interface::cast::cast(key_arr, &arrow_type)?,
                crate::arrow_interface::cast::cast(sorted, &arrow_type)?,
            )
        }
    };

    let key_inputs = [(&cast_key as &dyn Datum, opts)];
    let sorted_inputs = [(&cast_sorted as &dyn Datum, opts)];

    let normalized_keys = normalize_columns(&key_inputs)?;
    let normalized_sorted = normalize_columns(&sorted_inputs)?;
    Ok((normalized_keys, normalized_sorted))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy, Default)]
pub struct SortOptions {
    pub descending: bool,
    pub nulls_first: bool,
}

impl SortOptions {
    pub fn reverse(self) -> Self {
        Self {
            descending: !self.descending,
            nulls_first: !self.nulls_first,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{sort, SortOptions};
    use arrow_array::{
        cast::AsArray, types::Int32Type, Array, ArrayRef, Datum, Float32Array, Int32Array,
        Int64Array, StringArray, UInt32Array,
    };
    use itertools::Itertools;
    use std::sync::Arc;

    #[test]
    fn test_sort_i32_nonnull_fwd() {
        let data = vec![-10, 2, 7, 100, -274, 18, -22, 10000, -193, 18, 22, 12000];
        let mut perm = (0..data.len() as u32).collect_vec();
        perm.sort_by_key(|x| data[*x as usize]);
        let arr = Int32Array::from(data);

        let our_res = sort::sort_to_indices(&arr, SortOptions::default()).unwrap();
        assert_eq!(our_res, UInt32Array::from(perm));
    }

    #[test]
    fn test_sort_i32_dups() {
        let data = (0..1000).map(|_| 1).collect_vec();
        let data = Arc::new(Int32Array::from(data)) as ArrayRef;

        sort::sort_to_indices(data.as_ref(), SortOptions::default()).unwrap();
    }

    #[test]
    fn test_sort_i32_null() {
        let mut rng = fastrand::Rng::with_seed(42);
        let data = (0..32)
            .map(|_| {
                if rng.bool() {
                    Some(rng.i32(-1000..1000))
                } else {
                    None
                }
            })
            .collect_vec();

        let arr = Arc::new(Int32Array::from(data.clone())) as ArrayRef;

        let our_res = sort::sort_to_indices(arr.as_ref(), SortOptions::default())
            .unwrap()
            .into_iter()
            .map(|x| x.unwrap())
            .collect_vec();
        let num_nulls = arr.null_count();
        for i in 0..num_nulls {
            assert!(
                arr.is_null(our_res[our_res.len() - i - 1] as usize),
                "index {} (null index {}) was not null",
                our_res[our_res.len() - i - 1],
                i
            );
        }

        let sorted = data.iter().filter_map(|x| *x).sorted().collect_vec();
        let perm_values = our_res
            .iter()
            .filter(|idx| !arr.is_null(**idx as usize))
            .map(|idx| arr.as_primitive::<Int32Type>().value(*idx as usize))
            .collect_vec();
        assert_eq!(perm_values, sorted);

        let our_res = sort::sort_to_indices(arr.as_ref(), SortOptions::default().reverse())
            .unwrap()
            .into_iter()
            .map(|x| x.unwrap())
            .collect_vec();

        for i in 0..num_nulls {
            assert!(
                arr.is_null(our_res[i] as usize),
                "index {} (null index {}) was not null",
                our_res[our_res.len() - i - 1],
                i
            );
        }
    }

    #[test]
    fn test_sort_i32_nulls_first() {
        let data = vec![Some(1), None, Some(3), None, Some(2)];
        let arr = Arc::new(Int32Array::from(data)) as ArrayRef;

        let perm = sort::sort_to_indices(
            arr.as_ref(),
            SortOptions {
                descending: false,
                nulls_first: true,
            },
        )
        .unwrap()
        .into_iter()
        .map(|x| x.unwrap())
        .collect_vec();

        assert_eq!(perm, vec![1, 3, 0, 4, 2]);
    }

    #[test]
    fn test_sort_i32_nonnull_rev() {
        let data = vec![-10, 2, 7, 100, -274, 18, -22, 10000, -193, 18, 22, 12000];
        let mut perm = (0..data.len() as u32).collect_vec();
        perm.sort_by_key(|x| -data[*x as usize]);
        let data = Arc::new(Int32Array::from(data)) as ArrayRef;

        let our_res =
            sort::sort_to_indices(data.as_ref(), SortOptions::default().reverse()).unwrap();
        assert_eq!(our_res, UInt32Array::from(perm));
    }

    #[test]
    fn test_sort_multiarray_nonull() {
        let mut rng = fastrand::Rng::with_seed(42);
        let mut values = (0..1000).map(|_| rng.i32(..)).unique().collect_vec();

        rng.shuffle(&mut values);
        let arr1_data = values.clone();
        let arr1 = Int32Array::from(arr1_data.clone());

        rng.shuffle(&mut values);
        let arr2_data = values.clone();
        let arr2 = Int64Array::from(arr2_data.iter().copied().map(i64::from).collect_vec());

        let mut perm = (0..values.len() as u32).collect_vec();
        perm.sort_by_key(|x| (arr1_data[*x as usize], arr2_data[*x as usize]));

        let input = [&arr1 as &dyn Array, &arr2 as &dyn Array];
        let our_res = sort::multicol_sort_to_indices(&input, &[SortOptions::default(); 2]).unwrap();
        assert_eq!(our_res, UInt32Array::from(perm));
    }

    #[test]
    fn test_sort_string() {
        let data = StringArray::from(vec!["hello", "this", "is", "a", "test"]);
        let res = sort::sort_to_indices(&data, SortOptions::default()).unwrap();
        let res = res.iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(res, vec![3, 0, 2, 4, 1]);
    }

    #[test]
    fn test_sort_f32() {
        let data = Float32Array::from(vec![32.0, 16.0, f32::NAN, f32::INFINITY]);
        let res = sort::sort_to_indices(&data, SortOptions::default()).unwrap();
        let res = res.iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(res, vec![1, 0, 3, 2]);
    }

    #[test]
    fn test_topk_i32_nonnull_fwd() {
        let data = vec![-10, 2, 7, 100, -274, 18, -22, 10000, -193, 18, 22, 12000];
        let arr = Int32Array::from(data);
        let input = [&arr as &dyn Array];

        let our_res = sort::topk_indices(&input, &[SortOptions::default()], 3).unwrap();
        let our_res = our_res.values().to_vec();
        assert_eq!(our_res, vec![4, 8, 6]);
    }
}
