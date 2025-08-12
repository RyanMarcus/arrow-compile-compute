use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{LazyLock, RwLock},
};

use arrow_array::Array;
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    module::Module,
    values::{BasicValueEnum, FunctionValue, IntValue, PointerValue},
    AddressSpace, OptimizationLevel,
};
use itertools::Itertools;
use ouroboros::self_referencing;

use crate::{
    compiled_iter::{datum_to_iter, generate_next},
    compiled_kernels::{aggregate::minmax::MinMaxAgg, link_req_helpers, optimize_module},
    declare_blocks, increment_pointer, PrimitiveType,
};

mod count;
mod minmax;
mod sum;

pub use count::CountAgg;
pub use sum::SumAgg;

#[derive(Default)]
pub struct StringSaver {
    data: Vec<Box<[u8]>>,
}

impl StringSaver {
    pub fn insert(&mut self, data: &[u8]) -> u128 {
        self.data.push(Box::from(data));
        let last = self.data.last().unwrap();
        let start = last.as_ptr();
        let end = start.wrapping_add(last.len());
        ((start as u64) as u128) | (((end as u64) as u128) << 64)
    }

    pub fn finalize(self) -> Vec<Box<[u8]>> {
        self.data
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum AggType {
    Count,
    Sum,
    Min,
    Max,
}

pub trait AggAlloc {
    fn get_mut_ptr(&mut self) -> *mut c_void;
    fn ensure_capacity(&mut self, capacity: usize);
    fn current_capacity(&self) -> usize;
}

pub trait Aggregation {
    type Allocation: AggAlloc;
    type Output;

    fn new(pts: &[PrimitiveType]) -> Self;
    fn allocate(&self, num_tickets: usize) -> Self::Allocation;
    fn ptype(&self) -> PrimitiveType;
    fn merge_allocs(&self, alloc1: Self::Allocation, alloc2: Self::Allocation) -> Self::Allocation;
    fn agg_type() -> AggType;

    /// Returns the LLVM function that performs *ungrouped* aggregation. The
    /// function should take:
    ///
    /// * A pointer to the aggregation state.
    /// * A pointer to the input iterator.
    fn llvm_ungrouped_agg_func<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        next_func: FunctionValue<'a>,
    ) -> FunctionValue<'a> {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();
        let fn_type = ctx
            .void_type()
            .fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
        let func = llvm_mod.add_function("agg", fn_type, None);
        let b = ctx.create_builder();
        let agg_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let iter_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
        declare_blocks!(ctx, func, entry, loop_cond, loop_body, exit);

        b.position_at_end(entry);
        let buf_ptr = b
            .build_alloca(self.ptype().llvm_type(ctx), "buf_ptr")
            .unwrap();
        b.build_unconditional_branch(loop_cond).unwrap();

        b.position_at_end(loop_cond);
        let had_next = b
            .build_call(next_func, &[iter_ptr.into(), buf_ptr.into()], "next")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        b.build_conditional_branch(had_next, loop_body, exit)
            .unwrap();

        b.position_at_end(loop_body);
        let value = b
            .build_load(self.ptype().llvm_type(ctx), buf_ptr, "value")
            .unwrap();
        self.llvm_agg_one(ctx, llvm_mod, &b, agg_ptr, i64_type.const_zero(), value);
        b.build_unconditional_branch(loop_cond).unwrap();

        b.position_at_end(exit);
        b.build_return(None).unwrap();
        func
    }

    /// Returns the LLVM function that performs the aggregation. The function
    /// should take:
    ///
    /// * A pointer to the aggregation state.
    /// * A pointer to the ticket array.
    /// * A pointer to the input iterator.
    fn llvm_agg_func<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        next_func: FunctionValue<'a>,
    ) -> FunctionValue<'a> {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();
        let fn_type = ctx
            .void_type()
            .fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
        let func = llvm_mod.add_function("agg", fn_type, None);
        let b = ctx.create_builder();
        let agg_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let ticket_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
        let iter_ptr = func.get_nth_param(2).unwrap().into_pointer_value();
        declare_blocks!(ctx, func, entry, loop_cond, loop_body, exit);

        b.position_at_end(entry);
        let buf_ptr = b
            .build_alloca(self.ptype().llvm_type(ctx), "buf_ptr")
            .unwrap();
        let ticket_ptr_ptr = b.build_alloca(ptr_type, "ticket_ptr_ptr").unwrap();
        b.build_store(ticket_ptr_ptr, ticket_ptr).unwrap();
        b.build_unconditional_branch(loop_cond).unwrap();

        b.position_at_end(loop_cond);
        let had_next = b
            .build_call(next_func, &[iter_ptr.into(), buf_ptr.into()], "next")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        b.build_conditional_branch(had_next, loop_body, exit)
            .unwrap();

        b.position_at_end(loop_body);
        let ticket_ptr = b
            .build_load(ptr_type, ticket_ptr_ptr, "ticket_ptr")
            .unwrap()
            .into_pointer_value();
        let ticket = b
            .build_load(i64_type, ticket_ptr, "ticket")
            .unwrap()
            .into_int_value();
        let value = b
            .build_load(self.ptype().llvm_type(ctx), buf_ptr, "value")
            .unwrap();
        self.llvm_agg_one(ctx, llvm_mod, &b, agg_ptr, ticket, value);
        let new_ticket_ptr = increment_pointer!(ctx, b, ticket_ptr, 8);
        b.build_store(ticket_ptr_ptr, new_ticket_ptr).unwrap();
        b.build_unconditional_branch(loop_cond).unwrap();

        b.position_at_end(exit);
        b.build_return(None).unwrap();
        func
    }

    fn llvm_agg_one<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        b: &Builder<'a>,
        alloc_ptr: PointerValue<'a>,
        ticket: IntValue<'a>,
        value: BasicValueEnum<'a>,
    );
    fn finalize(&self, alloc: Self::Allocation) -> Self::Output;
}

impl<T: Copy + Default> AggAlloc for Vec<T> {
    fn get_mut_ptr(&mut self) -> *mut c_void {
        self.as_mut_ptr() as *mut c_void
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.resize_with(capacity, Default::default);
    }

    fn current_capacity(&self) -> usize {
        self.len()
    }
}

#[self_referencing]
struct GroupedAggFunc {
    ctx: Context,

    #[borrows(ctx)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void)>,
}
unsafe impl Send for GroupedAggFunc {}
unsafe impl Sync for GroupedAggFunc {}

fn compile_grouped_agg_func<A: Aggregation>(agg: &A, inp: &dyn Array) -> GroupedAggFunc {
    GroupedAggFuncBuilder {
        ctx: Context::create(),
        func_builder: |ctx| {
            let llvm_mod = ctx.create_module("grouped_agg_func");
            let ih = datum_to_iter(&inp).unwrap();
            let next_func =
                generate_next(ctx, &llvm_mod, "agg_next", inp.data_type(), &ih).unwrap();
            let func = agg.llvm_agg_func(ctx, &llvm_mod, next_func);

            llvm_mod.verify().unwrap();
            optimize_module(&llvm_mod).unwrap();
            let ee = llvm_mod
                .create_jit_execution_engine(OptimizationLevel::Aggressive)
                .unwrap();
            link_req_helpers(&llvm_mod, &ee).unwrap();

            let agg_func = unsafe {
                ee.get_function::<unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void)>(
                    func.get_name().to_str().unwrap(),
                )
                .unwrap()
            };

            agg_func
        },
    }
    .build()
}

#[self_referencing]
struct UngroupedAggFunc {
    ctx: Context,

    #[borrows(ctx)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut c_void)>,
}
unsafe impl Send for UngroupedAggFunc {}
unsafe impl Sync for UngroupedAggFunc {}

fn compile_ungrouped_agg_func<A: Aggregation>(agg: &A, inp: &dyn Array) -> UngroupedAggFunc {
    UngroupedAggFuncBuilder {
        ctx: Context::create(),
        func_builder: |ctx| {
            let llvm_mod = ctx.create_module("ungrouped_agg_func");
            let ih = datum_to_iter(&inp).unwrap();
            let next_func =
                generate_next(ctx, &llvm_mod, "agg_next", inp.data_type(), &ih).unwrap();
            let func = agg.llvm_ungrouped_agg_func(ctx, &llvm_mod, next_func);

            llvm_mod.verify().unwrap();
            optimize_module(&llvm_mod).unwrap();
            let ee = llvm_mod
                .create_jit_execution_engine(OptimizationLevel::Aggressive)
                .unwrap();
            link_req_helpers(&llvm_mod, &ee).unwrap();

            let agg_func = unsafe {
                ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void)>(
                    func.get_name().to_str().unwrap(),
                )
                .unwrap()
            };

            agg_func
        },
    }
    .build()
}

static GROUPED_AGG_CACHE: LazyLock<RwLock<HashMap<(DataType, AggType), GroupedAggFunc>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

static UNGROUPED_AGG_CACHE: LazyLock<RwLock<HashMap<(DataType, AggType), UngroupedAggFunc>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

pub struct Aggregator<A: Aggregation> {
    agg: A,
    alloc: A::Allocation,
}

impl<A: Aggregation> Aggregator<A> {
    pub fn new(dts: &[&DataType]) -> Self {
        let pts = dts
            .iter()
            .map(|dt| PrimitiveType::for_arrow_type(dt))
            .collect_vec();
        let agg = A::new(&pts);
        let alloc = agg.allocate(1);
        Self { agg, alloc }
    }

    /// Ingests new *ungrouped* data into the aggregator. This is equivalent to
    /// calling `ingest_grouped` with 0 as every group ID, but much faster.
    pub fn ingest_ungrouped(&mut self, data: &dyn Array) {
        assert!(!data.is_nullable(), "can only aggregate non-nullable data");
        if data.is_empty() {
            return;
        }
        self.alloc.ensure_capacity(1);

        let cache_key = (data.data_type().clone(), A::agg_type());
        let mut ih = datum_to_iter(&data).unwrap();
        {
            let hm = UNGROUPED_AGG_CACHE.read().unwrap();
            if let Some(func) = hm.get(&cache_key) {
                unsafe {
                    func.borrow_func()
                        .call(self.alloc.get_mut_ptr(), ih.get_mut_ptr());
                }
                return;
            }
        }

        // same logic as KernelCache
        let func = compile_ungrouped_agg_func(&self.agg, data);
        unsafe {
            func.borrow_func()
                .call(self.alloc.get_mut_ptr(), ih.get_mut_ptr());
        }

        let mut hm = UNGROUPED_AGG_CACHE.write().unwrap();
        hm.entry(cache_key).or_insert(func);
    }

    /// Ingests new data into the aggregator. The `tickets` slice should be the
    /// group IDs of the corresponding elements in `data`.
    pub fn ingest_grouped(&mut self, tickets: &[u64], data: &dyn Array) {
        assert!(!data.is_nullable(), "can only aggregate non-nullable data");
        assert_eq!(
            tickets.len(),
            data.len(),
            "tickets and data must have the same length"
        );
        if tickets.is_empty() {
            return;
        }
        let max_ticket = tickets.iter().copied().max().unwrap_or(0) + 1;
        self.alloc.ensure_capacity(max_ticket as usize);

        let mut ih = datum_to_iter(&data).unwrap();

        let cache_key = (data.data_type().clone(), A::agg_type());
        {
            let hm = GROUPED_AGG_CACHE.read().unwrap();
            if let Some(func) = hm.get(&cache_key) {
                unsafe {
                    func.borrow_func().call(
                        self.alloc.get_mut_ptr(),
                        tickets.as_ptr() as *const c_void,
                        ih.get_mut_ptr(),
                    );
                }
                return;
            }
        }

        // same logic as KernelCache
        let func = compile_grouped_agg_func(&self.agg, data);
        unsafe {
            func.borrow_func().call(
                self.alloc.get_mut_ptr(),
                tickets.as_ptr() as *const c_void,
                ih.get_mut_ptr(),
            );
        }

        let mut hm = GROUPED_AGG_CACHE.write().unwrap();
        hm.entry(cache_key).or_insert(func);
    }

    pub fn merge(self, other: Self) -> Self {
        if self.alloc.current_capacity() < other.alloc.current_capacity() {
            return other.merge(self);
        }
        let alloc = self.agg.merge_allocs(self.alloc, other.alloc);
        Self {
            agg: self.agg,
            alloc,
        }
    }

    pub fn finish(self) -> A::Output {
        self.agg.finalize(self.alloc)
    }
}

pub type CountAggregator = Aggregator<CountAgg>;
pub type SumAggregator = Aggregator<SumAgg>;
pub type MinAggregator = Aggregator<MinMaxAgg<true>>;
pub type MaxAggregator = Aggregator<MinMaxAgg<false>>;

#[cfg(test)]
mod tests {
    use arrow_array::{
        cast::AsArray,
        types::{Int32Type, Int64Type},
        Int32Array, StringArray,
    };
    use arrow_schema::DataType;
    use itertools::Itertools;

    use crate::compiled_kernels::{CountAggregator, MaxAggregator, MinAggregator, SumAggregator};

    fn assert_send<T: Send>() {}

    #[test]
    fn aggs_are_send() {
        assert_send::<CountAggregator>();
        assert_send::<SumAggregator>();
        assert_send::<MinAggregator>();
        assert_send::<MaxAggregator>();
    }

    #[test]
    fn test_count_aggregator() {
        let mut agg = CountAggregator::new(&[]);
        agg.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        );
        agg.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        );
        let res = agg.finish().values().iter().copied().collect_vec();
        assert_eq!(res, vec![6, 6]);
    }

    #[test]
    fn test_count_merge_aggregator() {
        let mut agg1 = CountAggregator::new(&[]);
        agg1.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        );
        let mut agg2 = CountAggregator::new(&[]);
        agg2.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        );
        let agg = agg1.merge(agg2);
        let res = agg.finish().values().iter().copied().collect_vec();
        assert_eq!(res, vec![6, 6]);
    }

    #[test]
    fn test_sum_aggregator() {
        let mut agg = SumAggregator::new(&[&DataType::Int32]);
        agg.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        );
        agg.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        );
        let res = agg.finish();
        let res = res
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        assert_eq!(res, vec![18, 24]);
    }

    #[test]
    fn test_sum_merge_aggregator() {
        let mut agg1 = SumAggregator::new(&[&DataType::Int32]);
        agg1.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        );

        let mut agg2 = SumAggregator::new(&[&DataType::Int32]);
        agg2.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        );
        let agg = agg1.merge(agg2);
        let res = agg.finish();
        let res = res
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        assert_eq!(res, vec![18, 24]);
    }

    #[test]
    fn test_min_aggregator() {
        let mut agg = MinAggregator::new(&[&DataType::Int32]);
        agg.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, -2, 3, 4, 5, 6]),
        );
        agg.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![0, 2, 3000, 4, 50, 60]),
        );
        let res = agg.finish();
        let res = res
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        assert_eq!(res, vec![0, -2]);
    }

    #[test]
    fn test_min_merge_aggregator() {
        let mut agg1 = MinAggregator::new(&[&DataType::Int32]);
        agg1.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, -2, 3, 4, 5, 6]),
        );
        let mut agg2 = MinAggregator::new(&[&DataType::Int32]);
        agg2.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![0, 2, 3000, 4, 50, 60]),
        );
        let agg = agg1.merge(agg2);
        let res = agg.finish();
        let res = res
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        assert_eq!(res, vec![0, -2]);
    }

    #[test]
    fn test_max_aggregator() {
        let mut agg = MaxAggregator::new(&[&DataType::Int32]);
        agg.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, -2, 3, 4, 5, 6]),
        );
        agg.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![0, 2, 3000, 4, 50, 60]),
        );
        let res = agg.finish();
        let res = res
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        assert_eq!(res, vec![3000, 60]);
    }

    #[test]
    fn test_max_merge_aggregator() {
        let mut agg1 = MaxAggregator::new(&[&DataType::Int32]);
        agg1.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, -2, 3, 4, 5, 6]),
        );
        let mut agg2 = MaxAggregator::new(&[&DataType::Int32]);
        agg2.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![0, 2, 3000, 4, 50, 60]),
        );
        let agg = agg1.merge(agg2);
        let res = agg.finish();
        let res = res
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        assert_eq!(res, vec![3000, 60]);
    }

    #[test]
    fn test_min_str_aggregator() {
        let mut agg = MinAggregator::new(&[&DataType::Utf8]);
        agg.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &StringArray::from(vec!["apple", "banana", "cherry", "date", "elder", "fig"]),
        );
        agg.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &StringArray::from(vec!["zeta", "gamma", "luma", "puma", "alpha", "mango"]),
        );
        let res = agg.finish();
        let res = res
            .as_binary_view()
            .iter()
            .map(|x| std::str::from_utf8(x.unwrap()).unwrap())
            .collect_vec();
        assert_eq!(res, vec!["alpha", "banana"]);
    }

    #[test]
    fn test_min_str_merge_aggregator() {
        let mut agg1 = MinAggregator::new(&[&DataType::Utf8]);
        agg1.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &StringArray::from(vec!["apple", "banana", "cherry", "date", "elder", "fig"]),
        );
        let mut agg2 = MinAggregator::new(&[&DataType::Utf8]);
        agg2.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &StringArray::from(vec!["zeta", "gamma", "luma", "puma", "alpha", "mango"]),
        );
        let res = agg1.merge(agg2).finish();
        let res = res
            .as_binary_view()
            .iter()
            .map(|x| std::str::from_utf8(x.unwrap()).unwrap())
            .collect_vec();
        assert_eq!(res, vec!["alpha", "banana"]);
    }

    #[test]
    fn test_ungrouped_sum() {
        let mut agg = SumAggregator::new(&[&DataType::Int32]);
        agg.ingest_ungrouped(&Int32Array::from(vec![1, 2, 3, 4, 5, 6]));
        agg.ingest_ungrouped(&Int32Array::from(vec![1, 2, 3, 4, 5, 6]));
        let res = agg.finish();
        let res = res
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        assert_eq!(res, vec![2 * (6 + 5 + 4 + 3 + 2 + 1)]);
    }
}
