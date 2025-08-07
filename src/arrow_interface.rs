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

    use arrow_array::Array;
    use arrow_array::BooleanArray;
    use arrow_array::Datum;
    use arrow_array::UInt32Array;

    pub use crate::compiled_kernels::ComparisonKernel;
    use crate::compiled_kernels::KernelCache;
    use crate::compiled_kernels::SortKernel;
    use crate::compiled_kernels::SortOptions;
    use crate::ArrowKernelError;
    use crate::Predicate;

    static CMP_PROGRAM_CACHE: LazyLock<KernelCache<ComparisonKernel>> =
        LazyLock::new(KernelCache::new);

    static SORT_PROGRAM_CACHE: LazyLock<KernelCache<SortKernel>> = LazyLock::new(KernelCache::new);

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

    /// Returns an array of indices that would sort the input array. Combine
    /// this kernel with `take` to physically sort the array.
    pub fn sort_to_indices(
        arr: &dyn Array,
        options: SortOptions,
    ) -> Result<UInt32Array, ArrowKernelError> {
        SORT_PROGRAM_CACHE.get(vec![arr], vec![options])
    }

    /// Returns an array of indices that would sort the input arrays. Combine
    /// this kernel with `take` to physically sort the array.
    pub fn multicol_sort_to_indices(
        arr: &[&dyn Array],
        options: &[SortOptions],
    ) -> Result<UInt32Array, ArrowKernelError> {
        SORT_PROGRAM_CACHE.get(arr.to_vec(), options.to_vec())
    }
}

/// Covert arrays between different data types and layouts
pub mod cast {
    use std::sync::LazyLock;

    use arrow_array::Array;
    use arrow_array::ArrayRef;
    use arrow_schema::DataType;

    use crate::compiled_kernels::CastKernel;
    use crate::compiled_kernels::KernelCache;
    use crate::ArrowKernelError;

    static CAST_PROGRAM_CACHE: LazyLock<KernelCache<CastKernel>> = LazyLock::new(KernelCache::new);

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
        CAST_PROGRAM_CACHE.get(lhs, to_type.clone())
    }
}

pub mod iter {
    use std::sync::{Arc, LazyLock};

    use arrow_array::Array;

    use crate::{
        compiled_kernels::{ArrowIter, IterFuncHolder, KernelCache},
        ArrowKernelError, PrimitiveType,
    };

    static ITER_FUNC_CACHE: LazyLock<KernelCache<Arc<IterFuncHolder>>> =
        LazyLock::new(KernelCache::new);

    /// Iterates over an array, converting the array values to `i64`.
    ///
    /// # Example
    /// ```
    /// use arrow_array::Int32Array;
    /// use arrow_compile_compute::iter::iter_i64;
    ///
    /// let arr = Int32Array::from(vec![1, 2, 3]);
    /// let iter = iter_i64(&arr).unwrap();
    /// assert_eq!(iter.collect::<Vec<_>>(), vec![1, 2, 3]);
    /// ```
    pub fn iter_i64(array: &dyn Array) -> Result<impl Iterator<Item = i64>, ArrowKernelError> {
        let ifh = ITER_FUNC_CACHE.get(array, PrimitiveType::I64)?;
        let i = ArrowIter::<i64>::new(array, ifh)?;
        Ok(i)
    }

    /// Iterates over an array, converting the array values to `u64`.
    ///
    /// # Example
    /// ```
    /// use arrow_array::Int32Array;
    /// use arrow_compile_compute::iter::iter_u64;
    ///
    /// let arr = Int32Array::from(vec![1, 2, 3]);
    /// let iter = iter_u64(&arr).unwrap();
    /// assert_eq!(iter.collect::<Vec<_>>(), vec![1, 2, 3]);
    /// ```
    pub fn iter_u64(array: &dyn Array) -> Result<impl Iterator<Item = u64>, ArrowKernelError> {
        let ifh = ITER_FUNC_CACHE.get(array, PrimitiveType::U64)?;
        let i = ArrowIter::<u64>::new(array, ifh)?;
        Ok(i)
    }

    /// Iterates over an array, converting the array values to `f64`.
    ///
    /// # Example
    /// ```
    /// use arrow_array::Int32Array;
    /// use arrow_compile_compute::iter::iter_f64;
    ///
    /// let arr = Int32Array::from(vec![1, 2, 3]);
    /// let iter = iter_f64(&arr).unwrap();
    /// assert_eq!(iter.collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn iter_f64(array: &dyn Array) -> Result<impl Iterator<Item = f64>, ArrowKernelError> {
        let ifh = ITER_FUNC_CACHE.get(array, PrimitiveType::F64)?;
        let i = ArrowIter::<f64>::new(array, ifh)?;
        Ok(i)
    }

    pub fn iter_bytes(array: &dyn Array) -> Result<impl Iterator<Item = &[u8]>, ArrowKernelError> {
        let ifh = ITER_FUNC_CACHE.get(array, PrimitiveType::P64x2)?;
        let i = ArrowIter::<&[u8]>::new(array, ifh)?;
        Ok(i)
    }
}

/// Selection kernels, like `filter` and `take`.
pub mod select {
    use std::sync::LazyLock;

    use arrow_array::{make_array, Array, ArrayRef, BooleanArray};

    use crate::{
        compiled_kernels::{ConcatKernel, FilterKernel, KernelCache, TakeKernel},
        ArrowKernelError,
    };

    static TAKE_PROGRAM_CACHE: LazyLock<KernelCache<TakeKernel>> = LazyLock::new(KernelCache::new);
    static FILTER_PROGRAM_CACHE: LazyLock<KernelCache<FilterKernel>> =
        LazyLock::new(KernelCache::new);
    static CONCAT_PROGRAM_CACHE: LazyLock<KernelCache<ConcatKernel>> =
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

    /// Concatenates multiple arrays into a single array.
    ///
    /// The logical type of each array must be the same, but the physical type
    /// may differ.
    ///
    /// ```
    /// use arrow_array::{Int32Array, Array};
    /// use arrow_compile_compute::select::concat;
    ///
    /// let data1 = Int32Array::from(vec![1, 2, 3, 4]);
    /// let data2 = Int32Array::from(vec![5, 6, 7, 8]);
    /// let res = concat(&[&data1, &data2]).unwrap();
    ///
    /// assert_eq!(res.len(), 8);
    /// ```
    pub fn concat(data: &[&dyn Array]) -> Result<ArrayRef, ArrowKernelError> {
        if data.len() == 1 {
            return Ok(make_array(data[0].to_data()));
        }
        CONCAT_PROGRAM_CACHE.get(data, ())
    }
}

/// Computations (like hashing) over Arrow arrays.
pub mod compute {
    use std::sync::LazyLock;

    use arrow_array::{Array, Datum, UInt64Array};
    use cardinality_estimator::CardinalityEstimator;

    use crate::{
        compiled_kernels::{HashKernel, KernelCache},
        ArrowKernelError,
    };

    static HASH_PROGRAM_CACHE: LazyLock<KernelCache<HashKernel>> = LazyLock::new(KernelCache::new);

    /// Compute a 64-bit hash for each element in `data`.
    ///
    /// This function computes a hash of the input, returning a `UInt64Array`
    /// where each value is the hash of the corresponding element in the input.
    ///
    /// # Example
    ///
    /// ```
    /// use arrow_array::{Int32Array, Datum, UInt64Array};
    /// use arrow_compile_compute::compute::hash;
    ///
    /// let arr = Int32Array::from(vec![10, 20, 10]);
    /// let hashes: UInt64Array = hash(&arr).unwrap();
    /// assert_eq!(hashes.len(), arr.len());
    ///
    /// // The same input produces the same hash
    /// assert_eq!(hashes.value(0), hashes.value(2));
    /// ```
    pub fn hash(data: &dyn Datum) -> Result<UInt64Array, ArrowKernelError> {
        HASH_PROGRAM_CACHE.get(data, ())
    }

    /// Compute an approximation of the maximum run length inside of an array.
    /// Uses a constant amount of time.
    ///
    /// # Example
    ///
    /// ```
    /// use arrow_array::{Int32Array, Datum, UInt64Array};
    /// use arrow_compile_compute::compute::approx_max_run_length;
    /// let arr = Int32Array::from(vec![0, 10, 10, 10, 20, 30, 50]);
    /// let max_run = approx_max_run_length(&arr).unwrap();
    /// assert_eq!(max_run, 3);
    /// ```
    pub fn approx_max_run_length(data: &dyn Array) -> Result<u64, ArrowKernelError> {
        if data.len() < 2 {
            return Ok(1);
        }

        let mut without_last = data.slice(0, data.len() - 1);
        let mut without_first = data.slice(1, data.len() - 1);

        if without_last.len() > 8192 {
            without_last = without_last.slice(0, 8192);
            without_first = without_first.slice(0, 8192);
        }

        let bitmap = crate::cmp::eq(&without_last, &without_first)?;
        let mut cur_run = 0;
        let mut max_run = 0;
        for el in bitmap.iter() {
            if let Some(true) = el {
                cur_run += 1;
                max_run = max_run.max(cur_run);
            } else {
                cur_run = 0;
            }
        }

        Ok(max_run + 1)
    }

    /// Compute an approximation of the percentage of values that are distinct.
    /// Runs in constant time.
    ///
    /// # Example
    ///
    /// ```
    /// use arrow_array::{Int32Array, Datum, UInt64Array};
    /// use arrow_compile_compute::compute::approx_perc_distinct;
    /// let arr = Int32Array::from(vec![10, 20, 20, 20, 30, 10, 10, 1, 2, 3]);
    /// let pdist = approx_perc_distinct(&arr).unwrap();
    /// assert!(pdist > 0.5 && pdist < 0.7);
    /// ```
    pub fn approx_perc_distinct(data: &dyn Array) -> Result<f32, ArrowKernelError> {
        if data.len() <= 1 {
            return Ok(1.0);
        }

        let data = if data.len() > 8192 {
            data.slice(0, 8192)
        } else {
            data.slice(0, data.len())
        };

        let hashed = hash(&data)?;
        let mut ce = CardinalityEstimator::<u64>::new();
        hashed.iter().flatten().for_each(|x| ce.insert_hash(x));
        Ok(ce.estimate() as f32 / data.len() as f32)
    }
}

/// Grouped aggregation kernels, following a ["ticketing" approach](https://arxiv.org/abs/2505.04153).
///
/// Aggregators can `ingest` data, `merge` with other aggregators (of the same
/// type), and `finalize` to produce the final aggregation results.
///
/// # Example
///
/// ```rust
/// # use arrow_array::{Int32Array, Int64Array, cast::AsArray, types::Int64Type};
/// # use arrow_schema::DataType;
/// # use arrow_compile_compute::aggregate::SumAggregator;
/// # use itertools::Itertools;
///
/// // thread 1:
/// let mut agg1 = SumAggregator::new(&[&DataType::Int32]);
/// agg1.ingest_grouped(
///     &[0, 1, 0, 1, 0, 1],
///     &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
/// );
/// // more calls to `ingest` here...
///
/// // thread 2:
/// let mut agg2 = SumAggregator::new(&[&DataType::Int32]);
/// agg2.ingest_grouped(
///     &[0, 1, 0, 1, 0, 1],
///     &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
/// );
/// // more calls to `ingest` here...
///
/// // join thread 1 and thread 2, merge results and get answer
/// let agg = agg1.merge(agg2);
/// let res = agg.finish();
/// let res = res
///     .as_primitive::<Int64Type>()
///     .values()
///     .iter()
///     .copied()
///     .collect_vec();
/// assert_eq!(res, vec![18, 24]);
/// ```
///
///
pub mod aggregate {
    use arrow_schema::DataType;

    pub use crate::compiled_kernels::{
        CountAggregator, MaxAggregator, MinAggregator, SumAggregator,
    };
    use crate::ArrowKernelError;

    /// Creates a new sum aggregator. Final results are 64-bit versions of their
    /// inputs (e.g., `f32` is summed to `f64`).
    pub fn sum(ty: &DataType) -> Result<SumAggregator, ArrowKernelError> {
        Ok(SumAggregator::new(&[ty]))
    }

    /// Creates a new min aggregator. Final results will match the input type.
    pub fn min(ty: &DataType) -> Result<MinAggregator, ArrowKernelError> {
        Ok(MinAggregator::new(&[ty]))
    }

    /// Creates a new max aggregator. Final results will match the input type.
    pub fn max(ty: &DataType) -> Result<MaxAggregator, ArrowKernelError> {
        Ok(MaxAggregator::new(&[ty]))
    }

    /// Creates a new count aggregator. Final results will be `u64`.
    pub fn count() -> Result<CountAggregator, ArrowKernelError> {
        Ok(CountAggregator::new(&[]))
    }
}

pub mod arith {
    use std::sync::LazyLock;

    use arrow_array::{ArrayRef, Datum};

    use crate::{
        compiled_kernels::{BinOp, BinOpKernel, KernelCache},
        ArrowKernelError,
    };

    static BINOP_PROGRAM_CACHE: LazyLock<KernelCache<BinOpKernel>> =
        LazyLock::new(KernelCache::new);

    pub fn add(left: &dyn Datum, right: &dyn Datum) -> Result<ArrayRef, ArrowKernelError> {
        BINOP_PROGRAM_CACHE.get((left, right), BinOp::Add)
    }

    pub fn sub_wrapping(left: &dyn Datum, right: &dyn Datum) -> Result<ArrayRef, ArrowKernelError> {
        BINOP_PROGRAM_CACHE.get((left, right), BinOp::Sub)
    }

    pub fn mul_wrapping(left: &dyn Datum, right: &dyn Datum) -> Result<ArrayRef, ArrowKernelError> {
        BINOP_PROGRAM_CACHE.get((left, right), BinOp::Mul)
    }

    pub fn div(left: &dyn Datum, right: &dyn Datum) -> Result<ArrayRef, ArrowKernelError> {
        BINOP_PROGRAM_CACHE.get((left, right), BinOp::Div)
    }

    pub fn rem(left: &dyn Datum, right: &dyn Datum) -> Result<ArrayRef, ArrowKernelError> {
        BINOP_PROGRAM_CACHE.get((left, right), BinOp::Rem)
    }
}
