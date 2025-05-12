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

    pub fn apply_f64<F: FnMut(f64)>(data: &dyn Array, func: F) -> Result<(), ArrowKernelError> {
        FLOAT_FUNC_CACHE.call(data, func)
    }
    pub fn apply_i64<F: FnMut(i64)>(data: &dyn Array, func: F) -> Result<(), ArrowKernelError> {
        INT_FUNC_CACHE.call(data, func)
    }
    pub fn apply_u64<F: FnMut(u64)>(data: &dyn Array, func: F) -> Result<(), ArrowKernelError> {
        UINT_FUNC_CACHE.call(data, func)
    }
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
