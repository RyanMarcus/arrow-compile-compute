/// Array-array and array-scalar comparison functions.
///
/// These functions can compare any array types (e.g., compare a dictionary
/// array and a run-end encoded array). Comparisons between arrays of different
/// base types (e.g., `Int32` and `Int64`) will perform the comparison after
/// casting to a dominant type.
///
/// Just like [the default `arrow`
/// kernels](https://docs.rs/arrow/latest/arrow/compute/kernels/cmp/index.html),
/// these functions use `Datum` to represent scalar or array types, and IEEE 754
/// floating point order.
///
/// ```
/// use arrow_array::{Int32Array, Int8Array, DictionaryArray};
/// use arrow_compile_compute::cmp;
/// use std::sync::Arc;
///
/// let primitive = Int32Array::from(vec![1, 2, 3, 2]);
/// let keys = Int8Array::from(vec![0, 1, 2, 1]);
/// let values = Int32Array::from(vec![1, 2, 3]);
/// let dict_array = DictionaryArray::new(keys, Arc::new(values));
///
/// // element-wise equal comparison
/// let result = cmp::eq(&primitive, &dict_array).unwrap();
/// assert_eq!(result.value(0), true);  // 1 == 1
/// assert_eq!(result.value(1), true);  // 2 == 2
/// assert_eq!(result.value(2), true);  // 3 == 3
/// assert_eq!(result.value(3), true);  // 2 == 2
/// ```
pub mod cmp {
    use std::sync::LazyLock;

    use arrow_array::BooleanArray;
    use arrow_array::Datum;

    use crate::new_kernels::ComparisonKernel;
    use crate::new_kernels::KernelCache;
    use crate::ArrowKernelError;
    use crate::Predicate;

    static CMP_PROGRAM_CACHE: LazyLock<KernelCache<ComparisonKernel>> =
        LazyLock::new(KernelCache::new);

    /// Compute a bitvector for `lhs < rhs`
    pub fn lt(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Lt)
    }

    /// Compute a bitvector for `lhs <= rhs`
    pub fn lt_eq(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Lte)
    }

    /// Compute a bitvector for `lhs > rhs`
    pub fn gt(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Gt)
    }

    /// Compute a bitvector for `lhs >= rhs`
    pub fn gt_eq(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Gte)
    }

    /// Compute a bitvector for `lhs == rhs`
    pub fn eq(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Eq)
    }

    /// Compute a bitvector for `lhs != rhs`
    pub fn neq(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Ne)
    }
}

/// Covert arrays between different data types and layouts
pub mod cast {
    use std::sync::LazyLock;

    use arrow_array::Array;
    use arrow_array::ArrayRef;
    use arrow_schema::DataType;

    use crate::new_kernels::CastToDictKernel;
    use crate::new_kernels::CastToFlatKernel;
    use crate::new_kernels::KernelCache;
    use crate::ArrowKernelError;

    static CAST_PROGRAM_CACHE: LazyLock<KernelCache<CastToFlatKernel>> =
        LazyLock::new(KernelCache::new);
    static CAST_TO_DICT_PROGRAM_CACHE: LazyLock<KernelCache<CastToDictKernel>> =
        LazyLock::new(KernelCache::new);

    /// Try to convert `lhs` into an array of type `to_type`.
    ///
    /// For example, to cast an integer array to a dictionary array:
    ///
    /// ```
    /// use arrow_array::{Int32Array, DictionaryArray};
    /// use arrow_schema::DataType;
    /// use arrow_compile_compute::cast::cast;
    ///
    /// let arr = Int32Array::from(vec![1, 2, 1]);
    /// let dict_array = cast(&arr, &DataType::Dictionary(
    ///     Box::new(DataType::Int8),
    ///     Box::new(DataType::Int32),
    /// )).unwrap();
    /// ```
    pub fn cast(lhs: &dyn Array, to_type: &DataType) -> Result<ArrayRef, ArrowKernelError> {
        match to_type {
            DataType::Dictionary(..) => CAST_TO_DICT_PROGRAM_CACHE.get(lhs, to_type.clone()),
            _ => CAST_PROGRAM_CACHE.get(lhs, to_type.clone()),
        }
    }
}

/// Run closures over Arrow arrays.
///
/// Iterating over an Arrow [`arrow_array::Array`] requires handling each
/// possible data type and data layout (e.g., a dictionary encoded array of
/// 32-bit signed integers). These functions abstract over encoding and
/// width-extend types.
///
/// **In general, you should prefer using a compute kernel instead of these
/// functions**, such as [`cmp::lt`]. The functions here have
/// function call overhead, while the other kernels do not.
///
/// Printing out all even numbers from an array:
/// ```
/// use arrow_array::Int32Array;
/// use arrow_compile_compute::apply::apply_i64;
///
/// let arr = Int32Array::from(vec![1, 2, 3, 4, 5]);
/// apply_i64(&arr, |i| if i % 2 == 0 {
///     println!("{}", i);
/// });
/// ```
pub mod apply {

    use std::sync::LazyLock;

    use arrow_array::Array;

    use crate::{
        new_kernels::{FloatFuncCache, IntFuncCache, StrFuncCache, UIntFuncCache},
        ArrowKernelError,
    };

    static FLOAT_FUNC_CACHE: LazyLock<FloatFuncCache> = LazyLock::new(FloatFuncCache::default);
    static INT_FUNC_CACHE: LazyLock<IntFuncCache> = LazyLock::new(IntFuncCache::default);
    static UINT_FUNC_CACHE: LazyLock<UIntFuncCache> = LazyLock::new(UIntFuncCache::default);
    static STRING_FUNC_CACHE: LazyLock<StrFuncCache> = LazyLock::new(StrFuncCache::default);

    /// Iterate over data casted to `f64`.
    ///
    /// Works over any data type that can be casted to an `f64` (e.g., a
    /// dictionary-encoded array of `i32`s.)
    ///
    /// ```
    /// use arrow_array::Int32Array;
    /// use arrow_compile_compute::apply::apply_f64;
    ///
    /// let data = Int32Array::from(vec![1, 2, 3]);
    /// let mut res = 0.0;
    /// apply_f64(&data, |i| res += i / 2.0);
    /// assert_eq!(res, 3.0);
    /// ```
    pub fn apply_f64<F: FnMut(f64)>(data: &dyn Array, func: F) -> Result<(), ArrowKernelError> {
        FLOAT_FUNC_CACHE.call(data, func)
    }

    /// Iterate over data casted to `i64`.
    ///
    /// Works over any data type that can be casted to an `i64` (e.g., a
    /// dictionary-encoded array of `i32`s.)
    ///
    /// ```
    /// use arrow_array::Int32Array;
    /// use arrow_compile_compute::apply::apply_i64;
    ///
    /// let data = Int32Array::from(vec![1, 2, 3]);
    /// let mut res = 0;
    /// apply_i64(&data, |i| res += i);
    /// assert_eq!(res, 6);
    /// ```
    pub fn apply_i64<F: FnMut(i64)>(data: &dyn Array, func: F) -> Result<(), ArrowKernelError> {
        INT_FUNC_CACHE.call(data, func)
    }

    /// Iterate over data casted to `u64`.
    ///
    /// Works over any data type that can be casted to an `u64` (e.g., a
    /// dictionary-encoded array of `u32`s.)
    ///
    /// ```
    /// use arrow_array::UInt32Array;
    /// use arrow_compile_compute::apply::apply_u64;
    ///
    /// let data = UInt32Array::from(vec![1, 2, 3]);
    /// let mut res = 0;
    /// apply_u64(&data, |i| res += i);
    /// assert_eq!(res, 6);
    /// ```
    pub fn apply_u64<F: FnMut(u64)>(data: &dyn Array, func: F) -> Result<(), ArrowKernelError> {
        UINT_FUNC_CACHE.call(data, func)
    }

    /// Iterate over data casted to a byte slice.
    ///
    /// Works over any data type that can be casted to a byte slice (e.g., a
    /// dictionary-encoded array of strings).
    ///
    /// ```
    /// use arrow_array::StringArray;
    /// use arrow_compile_compute::apply::apply_str;
    ///
    /// let data = StringArray::from(vec!["hello ", "world"]);
    /// let mut res = Vec::new();
    /// apply_str(&data, |i| res.extend_from_slice(i));
    /// assert_eq!(std::str::from_utf8(&res).unwrap(), "hello world");
    /// ```
    pub fn apply_str<F: FnMut(&[u8])>(data: &dyn Array, func: F) -> Result<(), ArrowKernelError> {
        STRING_FUNC_CACHE.call(data, func)
    }
}

pub mod select {
    use std::sync::LazyLock;

    use arrow_array::{Array, ArrayRef, BooleanArray};

    use crate::{
        new_kernels::{FilterKernel, KernelCache, TakeKernel},
        ArrowKernelError,
    };

    static TAKE_PROGRAM_CACHE: LazyLock<KernelCache<TakeKernel>> = LazyLock::new(KernelCache::new);
    static FILTER_PROGRAM_CACHE: LazyLock<KernelCache<FilterKernel>> =
        LazyLock::new(KernelCache::new);

    /// Extract the elements in `data` at the indices specified in `idxes`.
    ///
    /// This function computes `data[idxes]`. No bounds checking is performed.
    ///
    /// ```
    /// use arrow_array::{StringArray, Int32Array, Array};
    /// use arrow_array::cast::AsArray;
    /// use arrow_compile_compute::select;
    ///
    /// let data = StringArray::from(vec!["this", "is", "a", "test"]);
    /// let idxs = Int32Array::from(vec![2, 3]);
    /// let res = select::take(&data, &idxs).unwrap();
    /// let res = res.as_string::<i32>();
    ///
    /// assert_eq!(res.value(0), "a");
    /// assert_eq!(res.value(1), "test");
    /// assert_eq!(res.len(), 2);
    /// ```
    pub fn take(data: &dyn Array, idxes: &dyn Array) -> Result<ArrayRef, ArrowKernelError> {
        TAKE_PROGRAM_CACHE.get((data, idxes), ())
    }

    /// Extracts the elements corresponding with the true elements of `filter`.
    ///
    /// This function computes `data[filter]`. Panics if `data` and `filter` do
    /// not have the same length.
    ///
    /// ```
    /// use arrow_array::{StringArray, BooleanArray, Int32Array, Array};
    /// use arrow_array::cast::AsArray;
    /// use arrow_compile_compute::select;
    ///
    /// let data = StringArray::from(vec!["this", "is", "a", "test"]);
    /// let filter = BooleanArray::from(vec![false, true, true, false]);
    /// let res = select::filter(&data, &filter).unwrap();
    /// let res = res.as_string::<i32>();
    ///
    /// assert_eq!(res.value(0), "is");
    /// assert_eq!(res.value(1), "a");
    /// assert_eq!(res.len(), 2);
    /// ```
    pub fn filter(data: &dyn Array, filter: &BooleanArray) -> Result<ArrayRef, ArrowKernelError> {
        FILTER_PROGRAM_CACHE.get((data, filter), ())
    }
}
