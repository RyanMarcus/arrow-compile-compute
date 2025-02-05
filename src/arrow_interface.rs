use std::{
    collections::HashMap,
    sync::{LazyLock, RwLock},
};

use crate::{
    aggregate::Aggregation, CompiledAggFunc, CompiledBinaryFunc, CompiledConvertFunc,
    CompiledFilterFunc, Predicate, PrimitiveType,
};
use arrow_array::{Array, ArrayRef, BooleanArray, Datum};
use arrow_schema::{ArrowError, DataType};
use inkwell::context::Context;
use ouroboros::self_referencing;

use crate::CodeGen;

static GLOBAL_PROGRAM_CACHE: LazyLock<ProgramCache> = LazyLock::new(ProgramCache::new);

#[self_referencing]
pub struct SelfContainedBinaryFunc {
    ctx: Context,

    #[borrows(ctx)]
    #[not_covariant]
    cf: CompiledBinaryFunc<'this>,
}
unsafe impl Send for SelfContainedBinaryFunc {}
unsafe impl Sync for SelfContainedBinaryFunc {}

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

fn build_prim_prim_cmp(
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
                DataType::Utf8 => {
                    if nullable {
                        return Err(ArrowError::ComputeError(
                            "nullable string aggs not yet supported".into(),
                        ));
                    } else {
                        cg.string_minmax(PrimitiveType::I32, agg)
                    }
                }
                DataType::LargeUtf8 => {
                    if nullable {
                        return Err(ArrowError::ComputeError(
                            "nullable string aggs not yet supported".into(),
                        ));
                    } else {
                        cg.string_minmax(PrimitiveType::I64, agg)
                    }
                }
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
        let (arr1, arr2) = normalize_order(arr1, arr2);
        let (d1, is_scalar1) = arr1.get();
        let (d2, is_scalar2) = arr2.get();

        let sig = (
            d1.data_type().clone(),
            is_scalar1,
            d2.data_type().clone(),
            is_scalar2,
            pred,
        );

        {
            let lcache = self.cmp_cache.read().unwrap();
            if let Some(f) = lcache.get(&sig) {
                return f.with_cf(|cf| cf.call(d1, d2));
            }
        }

        // small race here: it is possible that multiple different threads will
        // see there is no function, then compile it, then store it. This is
        // fine a price to pay for not holding the lock for all function
        // compilation.
        let new_f =
            build_prim_prim_cmp(d1.data_type(), is_scalar1, d2.data_type(), is_scalar2, pred)?;
        let result = new_f.with_cf(|cf| cf.call(d1, d2));
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

/// if either of dt1 or dt2 is a primitive, put it first
fn normalize_order<'a>(d1: &'a dyn Datum, d2: &'a dyn Datum) -> (&'a dyn Datum, &'a dyn Datum) {
    let (arr1, scalar1) = d1.get();
    let (arr2, scalar2) = d2.get();

    if scalar1 {
        return (d2, d1);
    }

    if scalar2 {
        return (d1, d2);
    }

    if arr1.data_type().is_primitive() {
        return (d1, d2);
    }

    if arr2.data_type().is_primitive() {
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
