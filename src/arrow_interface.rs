pub mod cmp {
    use std::sync::LazyLock;

    use arrow_array::BooleanArray;
    use arrow_array::Datum;

    use crate::new_kernels::KernelCache;
    use crate::ArrowKernelError;
    use crate::ComparisonKernel;
    use crate::Predicate;

    static CMP_PROGRAM_CACHE: LazyLock<KernelCache<ComparisonKernel>> =
        LazyLock::new(KernelCache::new);

    pub fn lt(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Lt)
    }

    pub fn lt_eq(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Lte)
    }

    pub fn gt(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Gt)
    }

    pub fn gt_eq(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Gte)
    }

    pub fn eq(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Eq)
    }

    pub fn neq(lhs: &dyn Datum, rhs: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        CMP_PROGRAM_CACHE.get((lhs, rhs), Predicate::Ne)
    }
}

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

    pub fn cast(lhs: &dyn Array, to_type: &DataType) -> Result<ArrayRef, ArrowKernelError> {
        match to_type {
            DataType::Dictionary(..) => CAST_TO_DICT_PROGRAM_CACHE.get(lhs, to_type.clone()),
            _ => CAST_PROGRAM_CACHE.get(lhs, to_type.clone()),
        }
    }
}

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

    /// Iterate over data casted to `f64`. Works over any data type that can be
    /// casted to an `f64` (e.g., a dictionary-encoded array of `i32`s.)
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

    /// Iterate over data casted to `i64`. Works over any data type that can be
    /// casted to an `i64` (e.g., a dictionary-encoded array of `i32`s.)
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

    /// Iterate over data casted to `u64`. Works over any data type that can be
    /// casted to an `u64` (e.g., a dictionary-encoded array of `u32`s.)
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

    /// Iterate over data casted to a byte slice. Works over any data type that
    /// can be casted to a byte slice (e.g., a dictionary-encoded array of
    /// strings).
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

    use arrow_array::{Array, ArrayRef};

    use crate::{
        new_kernels::{KernelCache, TakeKernel},
        ArrowKernelError,
    };

    static TAKE_PROGRAM_CACHE: LazyLock<KernelCache<TakeKernel>> = LazyLock::new(KernelCache::new);

    pub fn take(data: &dyn Array, idxes: &dyn Array) -> Result<ArrayRef, ArrowKernelError> {
        TAKE_PROGRAM_CACHE.get((data, idxes), ())
    }
}
