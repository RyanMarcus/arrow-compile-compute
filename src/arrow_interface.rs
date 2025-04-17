use std::{
    collections::HashMap,
    sync::{LazyLock, RwLock},
};

use crate::{
    aggregate::Aggregation, CompiledAggFunc, CompiledBinaryFunc, CompiledConvertFunc,
    CompiledFilterFunc, CompiledTakeFunc, Predicate,
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
pub struct SelfContainedTakeFunc {
    ctx: Context,

    #[borrows(ctx)]
    #[not_covariant]
    cf: CompiledTakeFunc<'this>,
}
unsafe impl Send for SelfContainedTakeFunc {}
unsafe impl Sync for SelfContainedTakeFunc {}

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
        return SelfContainedConvertFuncTryBuilder {
            ctx,
            cf_builder: |ctx| {
                let cg = CodeGen::new(ctx);
                cg.compile_bitmap_to_vec()
            },
        }
        .try_build();
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
            match src {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                    cg.compile_filter_random_access(src)
                }
                _ => cg.compile_filter_block(src),
            }
        },
    }
    .try_build()
}

fn build_take(data: &DataType, idxes: &DataType) -> Result<SelfContainedTakeFunc, ArrowError> {
    let ctx = Context::create();
    SelfContainedTakeFuncTryBuilder {
        ctx,
        cf_builder: |ctx| {
            let cg = CodeGen::new(ctx);
            cg.compile_take(idxes, data)
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
                DataType::Utf8 => cg.string_minmax(src, agg, nullable),
                DataType::LargeUtf8 => cg.string_minmax(src, agg, nullable),
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

type CovFuncSig = (DataType, DataType);
type TakeFuncSig = (DataType, DataType);
type AggFuncSig = (DataType, bool, Aggregation);
struct ProgramCache {
    cov_cache: RwLock<HashMap<CovFuncSig, SelfContainedConvertFunc>>,
    hash_cache: RwLock<HashMap<DataType, SelfContainedConvertFunc>>,
    flt_cache: RwLock<HashMap<DataType, SelfContainedFilterFunc>>,
    tak_cache: RwLock<HashMap<TakeFuncSig, SelfContainedTakeFunc>>,
    agg_cache: RwLock<HashMap<AggFuncSig, SelfContainedAggFunc>>,
}

impl ProgramCache {
    fn new() -> Self {
        ProgramCache {
            cov_cache: RwLock::default(),
            hash_cache: RwLock::default(),
            flt_cache: RwLock::default(),
            tak_cache: RwLock::default(),
            agg_cache: RwLock::default(),
        }
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

    fn take(&self, data: &dyn Array, indexes: &dyn Array) -> Result<ArrayRef, ArrowError> {
        let sig = (data.data_type().clone(), indexes.data_type().clone());

        {
            let lcache = self.tak_cache.read().unwrap();
            if let Some(f) = lcache.get(&sig) {
                return f.with_cf(|cf| cf.call(data, indexes));
            }
        }

        // small race here: it is possible that multiple different threads will
        // see there is no function, then compile it, then store it. This is
        // fine a price to pay for not holding the lock for all function
        // compilation.
        let new_f = build_take(data.data_type(), indexes.data_type())?;
        let result = new_f.with_cf(|cf| cf.call(data, indexes));
        self.tak_cache.write().unwrap().insert(sig, new_f);
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

    pub fn take(data: &dyn Array, indexes: &dyn Array) -> Result<ArrayRef, ArrowError> {
        GLOBAL_PROGRAM_CACHE.take(data, indexes)
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
