use arrow_array::UInt64Array;
use inkwell::{context::Context, module::Module};

use crate::{
    compiled_kernels::aggregate::{AggType, Aggregation},
    increment_pointer, PrimitiveType,
};

#[derive(Default)]
pub struct CountAgg {}

impl Aggregation for CountAgg {
    type Allocation = Vec<u64>;

    type Output = UInt64Array;

    fn new(_pts: &[PrimitiveType]) -> Self {
        CountAgg {}
    }

    fn allocate(&self, num_tickets: usize) -> Self::Allocation {
        vec![0; num_tickets]
    }

    fn ptype(&self) -> PrimitiveType {
        PrimitiveType::max_width_type()
    }

    fn agg_type() -> AggType {
        AggType::Count
    }

    fn merge_allocs(
        &self,
        mut alloc1: Self::Allocation,
        alloc2: Self::Allocation,
    ) -> Self::Allocation {
        alloc1.iter_mut().zip(alloc2).for_each(|(a1, a2)| *a1 += a2);
        alloc1
    }

    fn finalize(&self, alloc: Self::Allocation) -> Self::Output {
        UInt64Array::from(alloc)
    }

    fn llvm_agg_one<'a>(
        &self,
        ctx: &'a Context,
        _llvm_mod: &Module<'a>,
        b: &inkwell::builder::Builder<'a>,
        alloc_ptr: inkwell::values::PointerValue<'a>,
        ticket: inkwell::values::IntValue<'a>,
        _value: inkwell::values::BasicValueEnum<'a>,
    ) {
        let i64_type = ctx.i64_type();
        let one = ctx.i64_type().const_int(1, false);
        let agg_ptr = increment_pointer!(ctx, b, alloc_ptr, 8, ticket);

        let curr_count = b
            .build_load(i64_type, agg_ptr, "curr_count")
            .unwrap()
            .into_int_value();
        let new_count = b.build_int_add(curr_count, one, "new_count").unwrap();
        b.build_store(agg_ptr, new_count).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::Int32Array;
    use arrow_schema::DataType;
    use inkwell::{context::Context, OptimizationLevel};
    use itertools::Itertools;

    use crate::{
        compiled_iter::{datum_to_iter, generate_next},
        compiled_kernels::{
            aggregate::{count::CountAgg, AggAlloc, Aggregation},
            link_req_helpers,
        },
    };

    #[test]
    fn test_count_agg() {
        let tickets: Vec<u64> = vec![0, 0, 0, 1, 1, 0, 2, 3, 0, 1];
        let data = Int32Array::from((0..tickets.len() as i32).collect_vec());

        let agg = CountAgg::new(&[]);
        let mut alloc = agg.allocate(4);

        let ctx = Context::create();
        let llvm_mod = ctx.create_module("count_agg");
        let mut ih = datum_to_iter(&data).unwrap();
        let next_func = generate_next(&ctx, &llvm_mod, "next", &DataType::Int32, &ih).unwrap();
        let func = agg.llvm_agg_func(&ctx, &llvm_mod, next_func);

        llvm_mod.verify().unwrap();
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&llvm_mod, &ee).unwrap();

        let agg_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void)>(
                func.get_name().to_str().unwrap(),
            )
            .unwrap()
        };

        unsafe {
            agg_func.call(
                alloc.get_mut_ptr(),
                tickets.as_ptr() as *const c_void,
                ih.get_mut_ptr(),
            );
        }

        let res = agg.finalize(alloc);
        let res = res.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, vec![5, 3, 1, 1])
    }
}
