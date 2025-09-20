use std::{
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
    compiled_kernels::{
        aggregate::minmax::MinMaxAgg, link_req_helpers, optimize_module, KernelCache,
    },
    declare_blocks, increment_pointer, ArrowKernelError, Kernel, PrimitiveType,
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
    fn get_ptr(&self) -> *const c_void;
    fn get_mut_ptr(&mut self) -> *mut c_void {
        self.get_ptr() as *mut c_void
    }

    /// Ensure that the allocator is ready to get a ticket up to `capacity`.
    /// Only call this function when you know that such a ticket will be ingested.
    fn ensure_capacity(&mut self, capacity: usize);

    /// The current maximum ticket value that can be aggregated
    fn current_capacity(&self) -> usize;

    /// Preallocate -- but do not create slots for -- an expected number of
    /// unique values.
    fn preallocate_capacity(&mut self, expected_unique: usize);
}

impl<T: Copy + Default> AggAlloc for Vec<T> {
    fn get_ptr(&self) -> *const c_void {
        self.as_ptr() as *const c_void
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.resize_with(capacity, Default::default);
    }

    fn preallocate_capacity(&mut self, expected_unique: usize) {
        self.reserve(expected_unique);
    }

    fn current_capacity(&self) -> usize {
        self.len()
    }
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
        atomic: bool,
    ) -> Option<FunctionValue<'a>> {
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

        if atomic {
            if !self.llvm_agg_one_atomic(ctx, llvm_mod, &b, agg_ptr, ticket, value) {
                return None;
            }
        } else {
            self.llvm_agg_one(ctx, llvm_mod, &b, agg_ptr, ticket, value);
        }
        let new_ticket_ptr = increment_pointer!(ctx, b, ticket_ptr, 8);
        b.build_store(ticket_ptr_ptr, new_ticket_ptr).unwrap();
        b.build_unconditional_branch(loop_cond).unwrap();

        b.position_at_end(exit);
        b.build_return(None).unwrap();
        Some(func)
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

    /// Returns true if supported, false otherwise.
    fn llvm_agg_one_atomic<'a>(
        &self,
        _ctx: &'a Context,
        _llvm_mod: &Module<'a>,
        _b: &Builder<'a>,
        _alloc_ptr: PointerValue<'a>,
        _ticket: IntValue<'a>,
        _value: BasicValueEnum<'a>,
    ) -> bool {
        false
    }

    fn finalize(&self, alloc: Self::Allocation) -> Self::Output;
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

impl Kernel for GroupedAggFunc {
    type Key = (Vec<DataType>, AggType, bool);

    type Input<'a> = (&'a [&'a dyn Array], &'a [u64], *mut c_void);

    type Params = (AggType, bool);

    type Output = ();

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, super::ArrowKernelError> {
        let (data, tickets, alloc_ptr) = inp;
        let mut ih = datum_to_iter(&data[0]).unwrap();

        unsafe {
            self.borrow_func().call(
                alloc_ptr,
                tickets.as_ptr() as *const c_void,
                ih.get_mut_ptr(),
            )
        };

        Ok(())
    }

    fn compile(
        inp: &Self::Input<'_>,
        params: Self::Params,
    ) -> Result<Self, super::ArrowKernelError> {
        let (data, _tickets, _alloc_ptr) = inp;
        let (agg_type, atomic) = params;
        match agg_type {
            AggType::Count => compile_grouped_agg_func::<CountAgg>(data[0], atomic),
            AggType::Sum => compile_grouped_agg_func::<SumAgg>(data[0], atomic),
            AggType::Min => compile_grouped_agg_func::<MinMaxAgg<true>>(data[0], atomic),
            AggType::Max => compile_grouped_agg_func::<MinMaxAgg<false>>(data[0], atomic),
        }
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        let dts = i.0.iter().map(|arr| arr.data_type().clone()).collect_vec();
        Ok((dts, p.0, p.1))
    }
}

fn compile_grouped_agg_func<A: Aggregation>(
    inp: &dyn Array,
    atomic: bool,
) -> Result<GroupedAggFunc, ArrowKernelError> {
    let dts = vec![PrimitiveType::for_arrow_type(inp.data_type())];
    let agg = A::new(&dts);
    GroupedAggFuncTryBuilder {
        ctx: Context::create(),
        func_builder: |ctx| {
            let llvm_mod = ctx.create_module("grouped_agg_func");
            let ih = datum_to_iter(&inp).unwrap();
            let next_func =
                generate_next(ctx, &llvm_mod, "agg_next", inp.data_type(), &ih).unwrap();
            let func = agg
                .llvm_agg_func(ctx, &llvm_mod, next_func, atomic)
                .ok_or(ArrowKernelError::AtomicAggNotSupported)?;

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

            Ok(agg_func)
        },
    }
    .try_build()
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

impl Kernel for UngroupedAggFunc {
    type Key = (Vec<DataType>, AggType);

    type Input<'a> = (&'a [&'a dyn Array], *mut c_void);

    type Params = AggType;

    type Output = ();

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, super::ArrowKernelError> {
        let (data, alloc_ptr) = inp;
        let mut ih = datum_to_iter(&data[0]).unwrap();

        unsafe { self.borrow_func().call(alloc_ptr, ih.get_mut_ptr()) };

        Ok(())
    }

    fn compile(
        inp: &Self::Input<'_>,
        params: Self::Params,
    ) -> Result<Self, super::ArrowKernelError> {
        let (data, _alloc_ptr) = inp;
        Ok(match params {
            AggType::Count => compile_ungrouped_agg_func::<CountAgg>(data[0]),
            AggType::Sum => compile_ungrouped_agg_func::<SumAgg>(data[0]),
            AggType::Min => compile_ungrouped_agg_func::<MinMaxAgg<true>>(data[0]),
            AggType::Max => compile_ungrouped_agg_func::<MinMaxAgg<false>>(data[0]),
        })
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        let dts = i.0.iter().map(|arr| arr.data_type().clone()).collect_vec();
        Ok((dts, *p))
    }
}

fn compile_ungrouped_agg_func<A: Aggregation>(inp: &dyn Array) -> UngroupedAggFunc {
    let dts = vec![PrimitiveType::for_arrow_type(inp.data_type())];
    let agg = A::new(&dts);
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

static GROUPED_AGG_CACHE: LazyLock<KernelCache<GroupedAggFunc>> = LazyLock::new(KernelCache::new);

static UNGROUPED_AGG_CACHE: LazyLock<KernelCache<UngroupedAggFunc>> =
    LazyLock::new(KernelCache::new);

pub struct Aggregator<A: Aggregation> {
    agg: A,
    alloc: RwLock<A::Allocation>,
}

impl<A: Aggregation> Aggregator<A> {
    pub fn new(dts: &[&DataType], expected_unique: usize) -> Self {
        let pts = dts
            .iter()
            .map(|dt| PrimitiveType::for_arrow_type(dt))
            .collect_vec();
        let agg = A::new(&pts);
        let mut alloc = agg.allocate(0);
        alloc.preallocate_capacity(expected_unique);
        Self {
            agg,
            alloc: RwLock::new(alloc),
        }
    }

    /// Ingests new *ungrouped* data into the aggregator. This is equivalent to
    /// calling `ingest_grouped` with 0 as every group ID, but much faster.
    pub fn ingest_ungrouped(&mut self, data: &dyn Array) {
        assert!(!data.is_nullable(), "can only aggregate non-nullable data");
        if data.is_empty() {
            return;
        }
        if self.alloc.get_mut().unwrap().current_capacity() < 1 {
            self.alloc.get_mut().unwrap().ensure_capacity(1);
        }
        UNGROUPED_AGG_CACHE
            .get(
                (&[data], self.alloc.get_mut().unwrap().get_mut_ptr()),
                A::agg_type(),
            )
            .unwrap();
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
        let max_ticket = tickets.iter().copied().max().unwrap_or(0) as usize + 1;
        self.alloc.get_mut().unwrap().ensure_capacity(max_ticket);
        GROUPED_AGG_CACHE
            .get(
                (
                    &[data],
                    tickets,
                    self.alloc.get_mut().unwrap().get_mut_ptr(),
                ),
                (A::agg_type(), false),
            )
            .unwrap();
    }

    /// Ingests new data into the aggregator atomically, if possible. The
    /// `tickets` slice should be the group IDs of the corresponding elements in
    /// `data`.
    ///
    /// Note that the return value may be `Ok` for an empty array even if that
    /// data type cannot be aggregated atomically.
    pub fn ingest_grouped_atomic(
        &self,
        tickets: &[u64],
        data: &dyn Array,
    ) -> Result<(), ArrowKernelError> {
        assert!(!data.is_nullable(), "can only aggregate non-nullable data");
        assert_eq!(
            tickets.len(),
            data.len(),
            "tickets and data must have the same length"
        );
        if tickets.is_empty() {
            return Ok(());
        }
        let max_ticket = tickets.iter().copied().max().unwrap_or(0) as usize + 1;
        let alloc = self.alloc.read().unwrap();
        let alloc = if alloc.current_capacity() <= max_ticket {
            std::mem::drop(alloc);
            let mut alloc = self.alloc.write().unwrap();
            alloc.ensure_capacity(max_ticket);
            std::mem::drop(alloc);
            self.alloc.read().unwrap()
        } else {
            alloc
        };

        GROUPED_AGG_CACHE.get(
            (&[data], tickets, alloc.get_ptr() as *mut c_void),
            (A::agg_type(), true),
        )
    }

    pub fn merge(mut self, mut other: Self) -> Self {
        if self.alloc.get_mut().unwrap().current_capacity()
            < other.alloc.get_mut().unwrap().current_capacity()
        {
            return other.merge(self);
        }
        let alloc = self.agg.merge_allocs(
            self.alloc.into_inner().unwrap(),
            other.alloc.into_inner().unwrap(),
        );
        Self {
            agg: self.agg,
            alloc: RwLock::new(alloc),
        }
    }

    pub fn finish(self) -> A::Output {
        self.agg.finalize(self.alloc.into_inner().unwrap())
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
        let mut agg = CountAggregator::new(&[], 1024);
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
    fn test_count_empty() {
        let agg = CountAggregator::new(&[], 1024);
        let res = agg.finish().values().iter().copied().collect_vec();
        assert_eq!(res, vec![]);
    }

    #[test]
    fn test_count_aggregator_atomic() {
        let agg = CountAggregator::new(&[], 1024);
        agg.ingest_grouped_atomic(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        )
        .unwrap();
        agg.ingest_grouped_atomic(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        )
        .unwrap();
        let res = agg.finish().values().iter().copied().collect_vec();
        assert_eq!(res, vec![6, 6]);
    }

    #[test]
    fn test_count_merge_aggregator() {
        let mut agg1 = CountAggregator::new(&[], 1024);
        agg1.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        );
        let mut agg2 = CountAggregator::new(&[], 1024);
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
        let mut agg = SumAggregator::new(&[&DataType::Int32], 1024);
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
    fn test_sum_aggregator_atomic() {
        let agg = SumAggregator::new(&[&DataType::Int32], 1024);
        agg.ingest_grouped_atomic(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        )
        .unwrap();
        agg.ingest_grouped_atomic(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        )
        .unwrap();
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
        let mut agg1 = SumAggregator::new(&[&DataType::Int32], 1024);
        agg1.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        );

        let mut agg2 = SumAggregator::new(&[&DataType::Int32], 1024);
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
        let mut agg = MinAggregator::new(&[&DataType::Int32], 1024);
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
    fn test_min_aggregator_atomic() {
        let agg = MinAggregator::new(&[&DataType::Int32], 1024);
        agg.ingest_grouped_atomic(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, -2, 3, 4, 5, 6]),
        )
        .unwrap();
        agg.ingest_grouped_atomic(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![0, 2, 3000, 4, 50, 60]),
        )
        .unwrap();
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
        let mut agg1 = MinAggregator::new(&[&DataType::Int32], 1024);
        agg1.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, -2, 3, 4, 5, 6]),
        );
        let mut agg2 = MinAggregator::new(&[&DataType::Int32], 1024);
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
        let mut agg = MaxAggregator::new(&[&DataType::Int32], 1024);
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
    fn test_max_aggregator_atomic() {
        let agg = MaxAggregator::new(&[&DataType::Int32], 1024);
        agg.ingest_grouped_atomic(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, -2, 3, 4, 5, 6]),
        )
        .unwrap();
        agg.ingest_grouped_atomic(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![0, 2, 3000, 4, 50, 60]),
        )
        .unwrap();
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
        let mut agg1 = MaxAggregator::new(&[&DataType::Int32], 1024);
        agg1.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &Int32Array::from(vec![1, -2, 3, 4, 5, 6]),
        );
        let mut agg2 = MaxAggregator::new(&[&DataType::Int32], 1024);
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
        let mut agg = MinAggregator::new(&[&DataType::Utf8], 1024);
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
        let mut agg1 = MinAggregator::new(&[&DataType::Utf8], 1024);
        agg1.ingest_grouped(
            &[0, 1, 0, 1, 0, 1],
            &StringArray::from(vec!["apple", "banana", "cherry", "date", "elder", "fig"]),
        );
        let mut agg2 = MinAggregator::new(&[&DataType::Utf8], 1024);
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
        let mut agg = SumAggregator::new(&[&DataType::Int32], 1024);
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
