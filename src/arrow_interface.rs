use std::{
    collections::HashMap,
    sync::{LazyLock, RwLock},
};

use crate::{
    aggregate::Aggregation, CompiledAggFunc, CompiledBinaryFunc, CompiledConvertFunc,
    CompiledFilterFunc, Predicate, PrimitiveType,
};

use arrow_array::array::{
    Int16Array, Int32Array, Int64Array, Int8Array, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow_array::types::{
    Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use arrow_array::{cast::AsArray, Array, ArrayRef, BooleanArray, Datum};
use arrow_schema::{ArrowError, DataType};
use inkwell::context::Context;
use ouroboros::self_referencing;

use crate::CodeGen;

static GLOBAL_PROGRAM_CACHE: LazyLock<ProgramCache> = LazyLock::new(ProgramCache::new);

macro_rules! convert_int {
    ($ty:ty, $arr_type:ty,$arr:expr) => {
        match $arr.data_type() {
            DataType::Int8 => <$ty>::try_from($arr.as_primitive::<Int8Type>().value(0))
                .ok()
                .map(|i| Arc::new(<$arr_type>::from(vec![i])) as ArrayRef),
            DataType::Int16 => <$ty>::try_from($arr.as_primitive::<Int16Type>().value(0))
                .ok()
                .map(|i| Arc::new(<$arr_type>::from(vec![i])) as ArrayRef),
            DataType::Int32 => <$ty>::try_from($arr.as_primitive::<Int32Type>().value(0))
                .ok()
                .map(|i| Arc::new(<$arr_type>::from(vec![i])) as ArrayRef),
            DataType::Int64 => <$ty>::try_from($arr.as_primitive::<Int64Type>().value(0))
                .ok()
                .map(|i| Arc::new(<$arr_type>::from(vec![i])) as ArrayRef),
            DataType::UInt8 => <$ty>::try_from($arr.as_primitive::<UInt8Type>().value(0))
                .ok()
                .map(|i| Arc::new(<$arr_type>::from(vec![i])) as ArrayRef),
            DataType::UInt16 => <$ty>::try_from($arr.as_primitive::<UInt16Type>().value(0))
                .ok()
                .map(|i| Arc::new(<$arr_type>::from(vec![i])) as ArrayRef),
            DataType::UInt32 => <$ty>::try_from($arr.as_primitive::<UInt32Type>().value(0))
                .ok()
                .map(|i| Arc::new(<$arr_type>::from(vec![i])) as ArrayRef),
            DataType::UInt64 => <$ty>::try_from($arr.as_primitive::<UInt64Type>().value(0))
                .ok()
                .map(|i| Arc::new(<$arr_type>::from(vec![i])) as ArrayRef),
            _ => None,
        }
    };
}

#[self_referencing]
pub struct SelfContainedBinaryFunc {
    ctx: Context,

    #[borrows(ctx)]
    #[not_covariant]
    cf: CompiledBinaryFunc<'this>,
}
unsafe impl Send for SelfContainedBinaryFunc {}
unsafe impl Sync for SelfContainedBinaryFunc {}

impl SelfContainedBinaryFunc {
    pub fn build(
        dt1: &DataType,
        lhs_scalar: bool,
        dt2: &DataType,
        rhs_scalar: bool,
        pred: Predicate,
    ) -> Result<SelfContainedBinaryFunc, ArrowError> {
        let ctx = Context::create();

        SelfContainedBinaryFuncTryBuilder {
            ctx,
            cf_builder: |ctx| {
                let cg = CodeGen::new(ctx);
                cg.primitive_primitive_cmp(dt1, lhs_scalar, dt2, rhs_scalar, pred)
            },
        }
        .try_build()
    }

    pub fn call(&self, arr1: &dyn Datum, arr2: &dyn Datum) -> Result<BooleanArray, ArrowError> {
        self.with_cf(|cf| cf.call(arr1.get().0, arr2.get().0))
    }
}

#[self_referencing]
pub struct SelfContainedConvertFunc {
    ctx: Context,

    #[borrows(ctx)]
    #[not_covariant]
    cf: CompiledConvertFunc<'this>,
}
unsafe impl Send for SelfContainedConvertFunc {}
unsafe impl Sync for SelfContainedConvertFunc {}

#[self_referencing]
pub struct SelfContainedFilterFunc {
    ctx: Context,

    #[borrows(ctx)]
    #[not_covariant]
    cf: CompiledFilterFunc<'this>,
}
unsafe impl Send for SelfContainedFilterFunc {}
unsafe impl Sync for SelfContainedFilterFunc {}

#[self_referencing]
pub struct SelfContainedAggFunc {
    ctx: Context,

    #[borrows(ctx)]
    #[not_covariant]
    cf: CompiledAggFunc<'this>,
}
unsafe impl Send for SelfContainedAggFunc {}
unsafe impl Sync for SelfContainedAggFunc {}

fn build_cast(src: &DataType, tar: &DataType) -> Result<SelfContainedConvertFunc, ArrowError> {
    let ctx = Context::create();
    if matches!(src, DataType::Boolean) && matches!(tar, DataType::UInt64) {
        return Ok(SelfContainedConvertFuncTryBuilder {
            ctx,
            cf_builder: |ctx| {
                let cg = CodeGen::new(ctx);
                cg.compile_bitmap_to_vec()
            },
        }
        .try_build()?);
    }

    SelfContainedConvertFuncTryBuilder {
        ctx,
        cf_builder: |ctx| {
            let cg = CodeGen::new(ctx);
            cg.cast_to_primitive(src, tar)
        },
    }
    .try_build()
}

fn build_hash(src: &DataType) -> Result<SelfContainedConvertFunc, ArrowError> {
    let ctx = Context::create();
    SelfContainedConvertFuncTryBuilder {
        ctx,
        cf_builder: |ctx| {
            let cg = CodeGen::new(ctx);
            cg.compile_murmur2(src)
        },
    }
    .try_build()
}

fn build_filter(src: &DataType) -> Result<SelfContainedFilterFunc, ArrowError> {
    let ctx = Context::create();
    SelfContainedFilterFuncTryBuilder {
        ctx,
        cf_builder: |ctx| {
            let cg = CodeGen::new(ctx);
            cg.compile_filter(src)
        },
    }
    .try_build()
}

fn build_agg(
    src: &DataType,
    nullable: bool,
    agg: Aggregation,
) -> Result<SelfContainedAggFunc, ArrowError> {
    let ctx = Context::create();
    SelfContainedAggFuncTryBuilder {
        ctx,
        cf_builder: |ctx| {
            let cg = CodeGen::new(ctx);
            match src {
                DataType::Utf8 => cg.string_minmax(PrimitiveType::I32, agg, nullable),
                DataType::LargeUtf8 => cg.string_minmax(PrimitiveType::I64, agg, nullable),
                _ => {
                    if nullable {
                        cg.compile_ungrouped_agg_with_nulls(src, agg)
                    } else {
                        cg.compile_ungrouped_aggregation(src, agg)
                    }
                }
            }
        },
    }
    .try_build()
}

type CmpFuncSig = (DataType, bool, DataType, bool, Predicate);
type CovFuncSig = (DataType, DataType);
type AggFuncSig = (DataType, bool, Aggregation);
struct ProgramCache {
    cmp_cache: RwLock<HashMap<CmpFuncSig, SelfContainedBinaryFunc>>,
    cov_cache: RwLock<HashMap<CovFuncSig, SelfContainedConvertFunc>>,
    hash_cache: RwLock<HashMap<DataType, SelfContainedConvertFunc>>,
    flt_cache: RwLock<HashMap<DataType, SelfContainedFilterFunc>>,
    agg_cache: RwLock<HashMap<AggFuncSig, SelfContainedAggFunc>>,
}

impl ProgramCache {
    fn new() -> Self {
        ProgramCache {
            cmp_cache: RwLock::default(),
            cov_cache: RwLock::default(),
            hash_cache: RwLock::default(),
            flt_cache: RwLock::default(),
            agg_cache: RwLock::default(),
        }
    }

    fn cmp(
        &self,
        arr1: &dyn Datum,
        arr2: &dyn Datum,
        pred: Predicate,
    ) -> Result<BooleanArray, ArrowError> {
        let (d1, d2) = normalize_order(arr1, arr2);
        let (arr1, is_scalar1) = d1.get();
        let (arr2, is_scalar2) = d2.get();

        let new_arr2 = if arr1.data_type().is_primitive()
            && arr2.data_type().is_primitive()
            && arr1.data_type() != arr2.data_type()
            && is_scalar2
        {
            // see if we can convert arr2 (scalar) to arr1's type
            match arr1.data_type() {
                DataType::Int8 => convert_int!(i8, Int8Array, arr2),
                DataType::Int16 => convert_int!(i16, Int16Array, arr2),
                DataType::Int32 => convert_int!(i32, Int32Array, arr2),
                DataType::Int64 => convert_int!(i64, Int64Array, arr2),
                DataType::UInt8 => convert_int!(u8, UInt8Array, arr2),
                DataType::UInt16 => convert_int!(u16, UInt16Array, arr2),
                DataType::UInt32 => convert_int!(u32, UInt32Array, arr2),
                DataType::UInt64 => convert_int!(u64, UInt64Array, arr2),
                //DataType::Float16 => convert_float!(f16, Float16Array, arr2),
                //DataType::Float32 => convert_float!(f32, Float32Array, arr2),
                //DataType::Float64 => convert_float!(f64, Float64Array, arr2),
                _ => None,
            }
        } else {
            None
        };

        let arr2 = new_arr2.as_deref().unwrap_or(arr2);

        let sig = (
            arr1.data_type().clone(),
            is_scalar1,
            arr2.data_type().clone(),
            is_scalar2,
            pred,
        );

        {
            let lcache = self.cmp_cache.read().unwrap();
            if let Some(f) = lcache.get(&sig) {
                return f.with_cf(|cf| cf.call(arr1, arr2));
            }
        }

        // small race here: it is possible that multiple different threads will
        // see there is no function, then compile it, then store it. This is
        // fine a price to pay for not holding the lock for all function
        // compilation.
        let new_f = SelfContainedBinaryFunc::build(
            arr1.data_type(),
            is_scalar1,
            arr2.data_type(),
            is_scalar2,
            pred,
        )?;
        let result = new_f.with_cf(|cf| cf.call(arr1, arr2));
        self.cmp_cache.write().unwrap().insert(sig, new_f);
        result
    }

    fn cast(&self, arr1: &dyn Array, target_dt: &DataType) -> Result<ArrayRef, ArrowError> {
        let sig = (arr1.data_type().clone(), target_dt.clone());

        {
            let lcache = self.cov_cache.read().unwrap();
            if let Some(f) = lcache.get(&sig) {
                return f.with_cf(|cf| cf.call(arr1));
            }
        }

        // small race here: it is possible that multiple different threads will
        // see there is no function, then compile it, then store it. This is
        // fine a price to pay for not holding the lock for all function
        // compilation.
        let new_f = build_cast(arr1.data_type(), target_dt)?;
        let result = new_f.with_cf(|cf| cf.call(arr1));
        self.cov_cache.write().unwrap().insert(sig, new_f);
        result
    }

    fn hash(&self, arr1: &dyn Array) -> Result<ArrayRef, ArrowError> {
        let sig = arr1.data_type().clone();

        {
            let lcache = self.hash_cache.read().unwrap();
            if let Some(f) = lcache.get(&sig) {
                return f.with_cf(|cf| cf.call(arr1));
            }
        }

        // small race here: it is possible that multiple different threads will
        // see there is no function, then compile it, then store it. This is
        // fine a price to pay for not holding the lock for all function
        // compilation.
        let new_f = build_hash(arr1.data_type())?;
        let result = new_f.with_cf(|cf| cf.call(arr1));
        self.hash_cache.write().unwrap().insert(sig, new_f);
        result
    }

    fn filter(&self, arr1: &dyn Array, bool: &BooleanArray) -> Result<ArrayRef, ArrowError> {
        let sig = arr1.data_type().clone();

        {
            let lcache = self.flt_cache.read().unwrap();
            if let Some(f) = lcache.get(&sig) {
                return f.with_cf(|cf| cf.call(arr1, bool));
            }
        }

        // small race here: it is possible that multiple different threads will
        // see there is no function, then compile it, then store it. This is
        // fine a price to pay for not holding the lock for all function
        // compilation.
        let new_f = build_filter(arr1.data_type())?;
        let result = new_f.with_cf(|cf| cf.call(arr1, bool));
        self.flt_cache.write().unwrap().insert(sig, new_f);
        result
    }

    fn agg(
        &self,
        arr1: &dyn Array,
        agg: Aggregation,
    ) -> Result<Option<Box<dyn Array>>, ArrowError> {
        let nullable = arr1
            .nulls()
            .map(|nulls| nulls.null_count() > 0)
            .unwrap_or(false);
        let sig = (arr1.data_type().clone(), nullable, agg);

        {
            let lcache = self.agg_cache.read().unwrap();
            if let Some(f) = lcache.get(&sig) {
                return f.with_cf(|cf| cf.call(arr1));
            }
        }

        // small race here: it is possible that multiple different threads will
        // see there is no function, then compile it, then store it. This is
        // fine a price to pay for not holding the lock for all function
        // compilation.
        let new_f = build_agg(arr1.data_type(), nullable, agg)?;
        let result = new_f.with_cf(|cf| cf.call(arr1));
        self.agg_cache.write().unwrap().insert(sig, new_f);
        result
    }
}
use std::sync::Arc;

/// If either of dt1 or dt2 is a primitive, put it first. If one is scalar and a
/// different type, check the range of the scalar and possibly convert it.
fn normalize_order<'a>(d1: &'a dyn Datum, d2: &'a dyn Datum) -> (&'a dyn Datum, &'a dyn Datum) {
    let (arr1, scalar1) = d1.get();
    let (arr2, scalar2) = d2.get();

    let (d1, d2) = if scalar1 { (d2, d1) } else { (d1, d2) };

    if !arr1.data_type().is_primitive() && (!scalar2 && arr2.data_type().is_primitive()) {
        return (d2, d1);
    }

    (d1, d2)
}

/// Compare arrays to other arrays and scalars, similar to
/// [`arrow_ord::cmp`](https://docs.rs/arrow-ord/53.3.0/arrow_ord/cmp/index.html)
///
/// ```rust
/// use arrow_array::Int32Array;
/// let arr1 = Int32Array::from(vec![1, 2, 3, 4, 5]);
/// let arr2 = Int32Array::from(vec![0, 2, 3, 4, 0]);
///
/// let result_from_arrow = arrow_ord::cmp::eq(&arr1, &arr2).unwrap();
/// let result_from_this_crate = arrow_compile_compute::cmp::eq(&arr1, &arr2).unwrap();
/// assert_eq!(result_from_arrow, result_from_this_crate);
/// ```
///
/// In addition to comparing arrays to scalars, this crate can compare arrays of
/// different datatypes, as long as they can be losslessly casted between each
/// other. For example, we can compare an Int32 array to an Int64 array:
///
/// ```rust
/// use arrow_array::{Int32Array, Int64Array};
/// let arr1 = Int32Array::from(vec![1, 2, 3, 4, 5]);
/// let arr2 = Int64Array::from(vec![0, 2, 3, 4, 0]);
///
/// // arrow_ord will throw an error
/// assert!(arrow_ord::cmp::eq(&arr1, &arr2).is_err());
///
/// // this crate will compile code to do the cast
/// let _result_from_this_crate = arrow_compile_compute::cmp::eq(&arr1, &arr2).unwrap();
/// ```
///
/// Like `arrow-rs`, floating point types are compared with `total_ord`.
///
/// Note that the algorithm for comparing run-end encoded (REE) arrays to
/// constants is suboptimal. Ideally, we would only compare the values of the
/// REE array to constants, and then return a REE boolean array. But the
/// `arrow-rs` API returns a `BooleanArray` for comparison functions. So we use
/// the naive, suboptimal algorithm that produces a materialized `BooleanArray`
/// to match the API.
///
pub mod cmp {
    use arrow_array::{BooleanArray, Datum};
    use arrow_schema::ArrowError;

    use crate::Predicate;

    use super::GLOBAL_PROGRAM_CACHE;

    /// Compare two arrays for equality.
    pub fn eq(arr1: &dyn Datum, arr2: &dyn Datum) -> Result<BooleanArray, ArrowError> {
        GLOBAL_PROGRAM_CACHE.cmp(arr1, arr2, Predicate::Eq)
    }

    /// Compare two arrays for inequality (not equal)
    pub fn neq(arr1: &dyn Datum, arr2: &dyn Datum) -> Result<BooleanArray, ArrowError> {
        GLOBAL_PROGRAM_CACHE.cmp(arr1, arr2, Predicate::Ne)
    }

    /// Compute where `arr1` < `arr2`
    pub fn lt(arr1: &dyn Datum, arr2: &dyn Datum) -> Result<BooleanArray, ArrowError> {
        GLOBAL_PROGRAM_CACHE.cmp(arr1, arr2, Predicate::Lt)
    }

    /// Compute where `arr1` <= `arr2`
    pub fn lt_eq(arr1: &dyn Datum, arr2: &dyn Datum) -> Result<BooleanArray, ArrowError> {
        GLOBAL_PROGRAM_CACHE.cmp(arr1, arr2, Predicate::Lte)
    }

    /// Compute where `arr1` > `arr2`
    pub fn gt(arr1: &dyn Datum, arr2: &dyn Datum) -> Result<BooleanArray, ArrowError> {
        GLOBAL_PROGRAM_CACHE.cmp(arr1, arr2, Predicate::Gt)
    }

    /// Compute where `arr1` >= `arr2`
    pub fn gt_eq(arr1: &dyn Datum, arr2: &dyn Datum) -> Result<BooleanArray, ArrowError> {
        GLOBAL_PROGRAM_CACHE.cmp(arr1, arr2, Predicate::Gte)
    }
}

pub mod cast {
    use arrow_array::{Array, ArrayRef};
    use arrow_schema::{ArrowError, DataType};

    use super::GLOBAL_PROGRAM_CACHE;

    /// Allows casting from various types to primitive types. For example, you
    /// can cast from an run-end encoded array to a primitive array:
    /// ```
    /// use arrow_array::{Int64Array, Int32Array, PrimitiveArray, RunArray};
    /// let res = Int32Array::from(vec![5, 6, 10, 11, 12]);
    /// let val = Int64Array::from(vec![1, 2, 3, 4, 5]);
    /// let ree_arr = RunArray::try_new(&PrimitiveArray::from(res), &PrimitiveArray::from(val)).unwrap();
    /// let ree_arr = ree_arr.downcast::<Int64Array>().unwrap();
    ///
    /// let result = Int64Array::from_iter(ree_arr).values().to_vec();
    /// let expected = vec![1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 5];
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// You can also cast a boolean array to a `UInt64Array`, resulting in the
    /// indexes of the "on" bits.
    pub fn cast(arr1: &dyn Array, target: &DataType) -> Result<ArrayRef, ArrowError> {
        GLOBAL_PROGRAM_CACHE.cast(arr1, target)
    }
}

pub mod compute {

    use arrow_array::{cast::AsArray, Array, ArrayRef, BooleanArray, UInt64Array};
    use arrow_schema::ArrowError;

    use crate::aggregate::Aggregation;

    use super::GLOBAL_PROGRAM_CACHE;

    pub fn hash(arr1: &dyn Array) -> Result<UInt64Array, ArrowError> {
        GLOBAL_PROGRAM_CACHE
            .hash(arr1)
            .map(|arr| arr.as_primitive().clone())
    }

    pub fn filter(arr1: &dyn Array, filter: &BooleanArray) -> Result<ArrayRef, ArrowError> {
        GLOBAL_PROGRAM_CACHE.filter(arr1, filter)
    }

    pub fn min(arr1: &dyn Array) -> Result<Option<Box<dyn Array>>, ArrowError> {
        GLOBAL_PROGRAM_CACHE.agg(arr1, Aggregation::Min)
    }

    pub fn max(arr1: &dyn Array) -> Result<Option<Box<dyn Array>>, ArrowError> {
        GLOBAL_PROGRAM_CACHE.agg(arr1, Aggregation::Max)
    }

    pub fn sum(arr1: &dyn Array) -> Result<Option<Box<dyn Array>>, ArrowError> {
        GLOBAL_PROGRAM_CACHE.agg(arr1, Aggregation::Sum)
    }
}
