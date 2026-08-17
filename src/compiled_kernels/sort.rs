use std::cmp::Ordering;

use arrow_array::{
    cast::AsArray,
    types::{
        Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
        UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrayRef, ArrowPrimitiveType, BinaryArray, Datum, UInt32Array,
};
use itertools::Itertools;

use crate::{
    compiled_kernels::{
        dsl2::DSLBuffer,
        sort_norm::{normalize_columns, normalize_fixed_width_columns},
        ArrowKernelError,
    },
    PrimitiveType,
};

pub fn sort_multi_col(data: &[(&dyn Datum, SortOptions)]) -> Result<UInt32Array, ArrowKernelError> {
    if data.is_empty() {
        return Err(ArrowKernelError::ArgumentMismatch(
            "sort requires at least one column".to_string(),
        ));
    }

    // try the fast path
    if let Some((keys, words)) = normalize_fixed_width_columns(data)? {
        return match words {
            2 => sort_fixed_width_records::<2>(keys),
            3 => sort_fixed_width_records::<3>(keys),
            4 => sort_fixed_width_records::<4>(keys),
            5 => sort_fixed_width_records::<5>(keys),
            6 => sort_fixed_width_records::<6>(keys),
            7 => sort_fixed_width_records::<7>(keys),
            8 => sort_fixed_width_records::<8>(keys),
            _ => Err(ArrowKernelError::InternalError(format!(
                "unsupported fixed sort key width {words}"
            ))),
        };
    }

    // fall back to the slow path
    let normalized = normalize_columns(data)?;
    Ok(sort_indices_from_normalized(&normalized))
}

fn sort_fixed_width_records<const N: usize>(
    mut keys: DSLBuffer,
) -> Result<UInt32Array, ArrowKernelError>
where
    [u64; N]: bytemuck::Pod,
{
    let records =
        bytemuck::try_cast_slice_mut::<u8, [u64; N]>(keys.buf.as_slice_mut()).map_err(|error| {
            ArrowKernelError::InternalError(format!(
                "could not view fixed sort keys as {N}-word records: {error}"
            ))
        })?;

    records.sort_unstable();
    Ok(UInt32Array::from(
        records
            .iter()
            .map(|record| record[N - 1] as u32)
            .collect_vec(),
    ))
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
            let normed = normalize_columns(&[(&arr.as_ref(), opts)])?;
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

/// Order-preserving packed sort keys: the value's bits are transformed so
/// plain unsigned order matches `ArrowNativeTypeOp::compare` (IEEE total
/// order for floats — the same transform `total_cmp` uses), then packed with
/// the row index in the low 32 bits. The packed keys sort with no comparator
/// at all, and ties resolve on the ascending index, so the output stays
/// deterministic. Descending order inverts the key bits before packing,
/// which reverses the value order while leaving the tie-break ascending.
trait PackedSortKey {
    type Packed: Ord + Copy;
    /// Number of digit passes an LSD radix sort needs to cover the value
    /// bits (the index bits below them start in order and stay in order
    /// because the radix passes are stable).
    const KEY_PASSES: usize;
    fn pack(self, index: u32, descending: bool) -> Self::Packed;
    fn unpack_index(packed: Self::Packed) -> u32;
    fn digit(packed: Self::Packed, pass: usize) -> usize;
}

macro_rules! packed_sort_key {
    ($t:ty, $u:ty, $packed:ty, |$v:ident| $key:expr) => {
        impl PackedSortKey for $t {
            type Packed = $packed;
            const KEY_PASSES: usize = (<$u>::BITS as usize).div_ceil(RADIX_DIGIT_BITS);
            #[inline]
            fn pack(self, index: u32, descending: bool) -> $packed {
                let $v = self;
                let key: $u = $key;
                let key = if descending { !key } else { key };
                ((key as $packed) << 32) | index as $packed
            }
            #[inline]
            fn unpack_index(packed: $packed) -> u32 {
                packed as u32
            }
            #[inline]
            fn digit(packed: $packed, pass: usize) -> usize {
                (packed >> (32 + RADIX_DIGIT_BITS * pass)) as usize & (RADIX_BINS - 1)
            }
        }
    };
}

/// Above this length an LSD radix sort over the packed keys beats the
/// comparison sort.
const RADIX_SORT_THRESHOLD: usize = 50_000;

/// 11-bit digits: 6 passes cover a 64-bit key (vs 8 with bytes), trading a
/// 16 KB histogram for 25% less scatter traffic.
const RADIX_DIGIT_BITS: usize = 11;
const RADIX_BINS: usize = 1 << RADIX_DIGIT_BITS;

/// Stable LSD radix sort over the value digits of packed sort keys. The index
/// bits below the value are already in ascending order in the input, and
/// stable passes keep equal-valued keys in that order, so the result is
/// identical to fully sorting the packed keys.
fn radix_sort_packed<K: PackedSortKey>(keys: &mut Vec<K::Packed>) {
    let n = keys.len();
    let mut src = std::mem::take(keys);
    let mut dst: Vec<K::Packed> = Vec::with_capacity(n);
    // SAFETY: every pass writes all n slots before anything reads them
    #[allow(clippy::uninit_vec)]
    unsafe {
        dst.set_len(n)
    };

    for pass in 0..K::KEY_PASSES {
        let mut counts = [0usize; RADIX_BINS];
        for key in src.iter() {
            counts[K::digit(*key, pass)] += 1;
        }
        if counts.iter().any(|count| *count == n) {
            continue;
        }
        let mut cursors = [0usize; RADIX_BINS];
        let mut sum = 0;
        for digit in 0..RADIX_BINS {
            cursors[digit] = sum;
            sum += counts[digit];
        }
        for key in src.iter() {
            let digit = K::digit(*key, pass);
            dst[cursors[digit]] = *key;
            cursors[digit] += 1;
        }
        std::mem::swap(&mut src, &mut dst);
    }

    *keys = src;
}

packed_sort_key!(u8, u8, u64, |v| v);
packed_sort_key!(u16, u16, u64, |v| v);
packed_sort_key!(u32, u32, u64, |v| v);
packed_sort_key!(u64, u64, u128, |v| v);
packed_sort_key!(i8, u8, u64, |v| (v as u8) ^ 0x80);
packed_sort_key!(i16, u16, u64, |v| (v as u16) ^ 0x8000);
packed_sort_key!(i32, u32, u64, |v| (v as u32) ^ 0x8000_0000);
packed_sort_key!(i64, u64, u128, |v| (v as u64) ^ (1 << 63));
packed_sort_key!(half::f16, u16, u64, |v| {
    let bits = v.to_bits();
    if bits & 0x8000 != 0 {
        !bits
    } else {
        bits | 0x8000
    }
});
packed_sort_key!(f32, u32, u64, |v| {
    let bits = v.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
});
packed_sort_key!(f64, u64, u128, |v| {
    let bits = v.to_bits();
    if bits >> 63 != 0 {
        !bits
    } else {
        bits | (1 << 63)
    }
});

/// Single column fast-path sort
fn sort_primitive<T: ArrowPrimitiveType>(
    data: &dyn Datum,
    opts: &SortOptions,
) -> Result<UInt32Array, ArrowKernelError>
where
    T::Native: PackedSortKey,
{
    let (values, is_scalar) = data.get();
    if is_scalar {
        return Ok(UInt32Array::from(vec![0]));
    }

    let values = values.as_primitive::<T>();
    if values.len() > u32::MAX as usize {
        // output indices (and the packed sort keys) hold row numbers in 32 bits
        return Err(ArrowKernelError::UnsupportedArguments(
            "sort_to_indices supports at most u32::MAX rows".to_string(),
        ));
    }
    let mut valids = Vec::with_capacity(values.len());
    let mut nulls = Vec::new();

    if let Some(validity) = values.nulls() {
        // one word-wise pass over the validity buffer builds both lists:
        // set bits pack valids, cleared bits push nulls
        nulls.reserve(validity.null_count());
        let chunks = validity.inner().bit_chunks();
        let mut base = 0usize;
        for chunk in chunks.iter().chain(std::iter::once(chunks.remainder_bits())) {
            let bits_in_chunk = 64.min(values.len() - base);
            if bits_in_chunk == 0 {
                break;
            }
            let mut set = chunk;
            while set != 0 {
                let index = base + set.trailing_zeros() as usize;
                // safety: set bits come from the validity bitmap, so the
                // index is in bounds and marked valid
                let value = unsafe { values.value_unchecked(index) };
                valids.push(value.pack(index as u32, opts.descending));
                set &= set - 1;
            }
            let mut clear = !chunk & (u64::MAX >> (64 - bits_in_chunk as u32));
            while clear != 0 {
                nulls.push((base + clear.trailing_zeros() as usize) as u32);
                clear &= clear - 1;
            }
            base += 64;
            if base >= values.len() {
                break;
            }
        }
    } else {
        for (index, value) in values.values().iter().enumerate() {
            valids.push(value.pack(index as u32, opts.descending));
        }
    }

    if valids.len() >= RADIX_SORT_THRESHOLD {
        radix_sort_packed::<T::Native>(&mut valids);
    } else {
        valids.sort_unstable();
    }

    let mut indices = Vec::with_capacity(values.len());
    if opts.nulls_first {
        indices.extend(nulls);
        indices.extend(
            valids
                .into_iter()
                .map(<T::Native as PackedSortKey>::unpack_index),
        );
    } else {
        indices.extend(
            valids
                .into_iter()
                .map(<T::Native as PackedSortKey>::unpack_index),
        );
        indices.extend(nulls);
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
    use crate::{sort, ArrowKernelError, SortOptions};
    use arrow_array::{
        cast::AsArray, types::Int32Type, Array, ArrayRef, Float16Array, Float32Array, Float64Array,
        Int16Array, Int32Array, Int64Array, Int8Array, StringArray, UInt16Array, UInt32Array,
        UInt64Array, UInt8Array,
    };
    use arrow_ord::sort::SortColumn;
    use arrow_schema::SortOptions as ArrowSortOptions;
    use half::f16;
    use itertools::Itertools;
    use std::sync::Arc;

    fn assert_multicol_sort_matches_arrow(columns: &[ArrayRef], options: &[SortOptions]) {
        let arrow_columns = columns
            .iter()
            .zip(options)
            .map(|(values, options)| SortColumn {
                values: values.clone(),
                options: Some(ArrowSortOptions {
                    descending: options.descending,
                    nulls_first: options.nulls_first,
                }),
            })
            .collect_vec();
        let expected = arrow_ord::sort::lexsort_to_indices(&arrow_columns, None).unwrap();
        let inputs = columns.iter().map(|column| column.as_ref()).collect_vec();
        let actual = sort::multicol_sort_to_indices(&inputs, options).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_radix_matches_comparison_sort() {
        use super::{radix_sort_packed, PackedSortKey};
        let mut rng = fastrand::Rng::with_seed(7);
        for descending in [false, true] {
            // heavy ties exercise stability; full-range values exercise digits
            for values in [
                (0..100_000).map(|_| rng.u64(0..50)).collect_vec(),
                (0..100_000).map(|_| rng.u64(..)).collect_vec(),
            ] {
                let packed = values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v.pack(i as u32, descending))
                    .collect_vec();
                let mut expected = packed.clone();
                expected.sort_unstable();
                let mut actual = packed;
                radix_sort_packed::<u64>(&mut actual);
                assert_eq!(actual, expected);
            }
        }
    }

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
    fn test_sort_fixed_width_three_columns() {
        let first = Int8Array::from(vec![i8::MIN, -1, -1, -1, -1, -1, i8::MAX]);
        let second = Int32Array::from(vec![0, i32::MIN, -1, -1, -1, -1, i32::MAX]);
        let third = Int64Array::from(vec![
            Some(0),
            Some(i64::MAX),
            None,
            Some(i64::MIN),
            Some(-1),
            Some(-1),
            Some(0),
        ]);
        let input = [
            &first as &dyn Array,
            &second as &dyn Array,
            &third as &dyn Array,
        ];

        let result = sort::multicol_sort_to_indices(&input, &[SortOptions::default(); 3]).unwrap();

        assert_eq!(result, UInt32Array::from(vec![0, 1, 3, 4, 5, 2, 6]));
    }

    #[test]
    fn test_sort_fixed_width_all_numeric_types_and_options() {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int8Array::from(vec![Some(0), None, Some(-1), Some(1)])),
            Arc::new(UInt8Array::from(vec![Some(2), Some(1), None, Some(0)])),
            Arc::new(Int16Array::from(vec![Some(-2), Some(2), Some(0), None])),
            Arc::new(UInt16Array::from(vec![None, Some(3), Some(1), Some(2)])),
            Arc::new(Int32Array::from(vec![
                Some(i32::MIN),
                Some(0),
                None,
                Some(i32::MAX),
            ])),
            Arc::new(UInt32Array::from(vec![
                Some(u32::MAX),
                None,
                Some(0),
                Some(1),
            ])),
            Arc::new(Int64Array::from(vec![
                Some(i64::MAX),
                Some(-1),
                Some(i64::MIN),
                None,
            ])),
            Arc::new(UInt64Array::from(vec![
                None,
                Some(0),
                Some(u64::MAX),
                Some(1),
            ])),
            Arc::new(Float16Array::from(vec![
                Some(f16::NEG_ZERO),
                Some(f16::INFINITY),
                None,
                Some(f16::NAN),
            ])),
            Arc::new(Float32Array::from(vec![
                Some(f32::NEG_INFINITY),
                None,
                Some(-0.0),
                Some(f32::from_bits(0x7fc0_0001)),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(f64::from_bits(0xfff8_0000_0000_0001)),
                Some(0.0),
                Some(f64::INFINITY),
                None,
            ])),
        ];
        let options = (0..columns.len())
            .map(|index| SortOptions {
                descending: index % 2 == 0,
                nulls_first: index % 3 == 0,
            })
            .collect_vec();

        assert_multicol_sort_matches_arrow(&columns, &options);
    }

    #[test]
    fn test_sort_fixed_width_float_total_order_and_ties() {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int8Array::from(vec![0; 10])),
            Arc::new(Float64Array::from(vec![
                Some(f64::from_bits(0xfff8_0000_0000_0002)),
                Some(f64::NEG_INFINITY),
                Some(-0.0),
                Some(0.0),
                Some(f64::INFINITY),
                Some(f64::from_bits(0x7ff8_0000_0000_0001)),
                Some(f64::from_bits(0x7ff8_0000_0000_0002)),
                None,
                Some(1.0),
                Some(1.0),
            ])),
        ];

        for options in [
            SortOptions::default(),
            SortOptions {
                descending: true,
                nulls_first: false,
            },
            SortOptions {
                descending: false,
                nulls_first: true,
            },
            SortOptions {
                descending: true,
                nulls_first: true,
            },
        ] {
            assert_multicol_sort_matches_arrow(&columns, &[SortOptions::default(), options]);
        }
    }

    #[test]
    fn test_sort_fixed_width_sliced_arrays() {
        let first = Int32Array::from(vec![Some(99), None, Some(-1), Some(0), Some(-1), Some(99)]);
        let second = UInt64Array::from(vec![100, 4, 3, 2, 1, 0]);
        let columns = vec![
            Arc::new(first.slice(1, 4)) as ArrayRef,
            Arc::new(second.slice(1, 4)) as ArrayRef,
        ];
        let options = [
            SortOptions {
                descending: false,
                nulls_first: true,
            },
            SortOptions {
                descending: true,
                nulls_first: false,
            },
        ];

        assert_multicol_sort_matches_arrow(&columns, &options);
    }

    #[test]
    fn test_sort_wide_numeric_and_mixed_string_fallbacks() {
        let wide = (0..8)
            .map(|column| {
                Arc::new(UInt64Array::from(vec![column, 10 - column, column, 5])) as ArrayRef
            })
            .collect_vec();
        assert_multicol_sort_matches_arrow(&wide, &[SortOptions::default(); 8]);

        let mixed: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![1, 1, 0, 1])),
            Arc::new(StringArray::from(vec!["b", "a", "z", "a"])),
        ];
        assert_multicol_sort_matches_arrow(&mixed, &[SortOptions::default(); 2]);
    }

    #[test]
    fn test_sort_fixed_width_empty_and_mismatched_lengths() {
        let empty_first = Int32Array::from(Vec::<i32>::new());
        let empty_second = UInt64Array::from(Vec::<u64>::new());
        let result = sort::multicol_sort_to_indices(
            &[&empty_first as &dyn Array, &empty_second as &dyn Array],
            &[SortOptions::default(); 2],
        )
        .unwrap();
        assert!(result.is_empty());

        let short = Int32Array::from(vec![1]);
        let long = Int32Array::from(vec![1, 2]);
        let error = sort::multicol_sort_to_indices(
            &[&short as &dyn Array, &long as &dyn Array],
            &[SortOptions::default(); 2],
        )
        .unwrap_err();
        assert!(matches!(error, ArrowKernelError::SizeMismatch));
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
