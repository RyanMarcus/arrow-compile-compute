use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array, UInt64Array};
use inkwell::{
    context::Context, module::Module, types::BasicType, values::BasicValue, AtomicOrdering,
    AtomicRMWBinOp,
};

use crate::{compiled_kernels::aggregate::Aggregation, increment_pointer, PrimitiveType};

pub struct SumAgg {
    pt: PrimitiveType,
}

impl Aggregation for SumAgg {
    type Allocation = Vec<u64>;

    type Output = ArrayRef;

    fn new(pts: &[PrimitiveType]) -> Self {
        assert_eq!(pts.len(), 1, "sum aggregation requires exactly one input");
        let pt = pts[0];
        assert!(pt != PrimitiveType::P64x2, "cannot sum strings");
        Self { pt }
    }

    fn allocate(&self, num_tickets: usize) -> Self::Allocation {
        vec![0; num_tickets]
    }

    fn ptype(&self) -> PrimitiveType {
        self.pt
    }

    fn agg_type() -> super::AggType {
        super::AggType::Sum
    }

    fn merge_allocs(
        &self,
        mut alloc1: Self::Allocation,
        alloc2: Self::Allocation,
    ) -> Self::Allocation {
        match self.pt {
            PrimitiveType::I8 | PrimitiveType::I16 | PrimitiveType::I32 | PrimitiveType::I64 => {
                let mut alloc1: Vec<i64> = bytemuck::cast_vec(alloc1);
                let alloc2: Vec<i64> = bytemuck::cast_vec(alloc2);
                alloc1.iter_mut().zip(alloc2).for_each(|(a, b)| *a += b);
                bytemuck::cast_vec(alloc1)
            }
            PrimitiveType::U8 | PrimitiveType::U16 | PrimitiveType::U32 | PrimitiveType::U64 => {
                alloc1.iter_mut().zip(alloc2).for_each(|(a, b)| *a += b);
                alloc1
            }
            PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                let mut alloc1: Vec<f64> = bytemuck::cast_vec(alloc1);
                let alloc2: Vec<f64> = bytemuck::cast_vec(alloc2);
                alloc1.iter_mut().zip(alloc2).for_each(|(a, b)| *a += b);
                bytemuck::cast_vec(alloc1)
            }
            PrimitiveType::P64x2 | PrimitiveType::List(_, _) => unreachable!(),
        }
    }

    fn finalize(&self, alloc: Self::Allocation) -> Self::Output {
        match self.pt {
            PrimitiveType::I8 | PrimitiveType::I16 | PrimitiveType::I32 | PrimitiveType::I64 => {
                let v: Vec<i64> = bytemuck::cast_vec(alloc);
                Arc::new(Int64Array::from(v))
            }
            PrimitiveType::U8 | PrimitiveType::U16 | PrimitiveType::U32 | PrimitiveType::U64 => {
                Arc::new(UInt64Array::from(alloc))
            }
            PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                let v: Vec<f64> = bytemuck::cast_vec(alloc);
                Arc::new(Float64Array::from(v))
            }
            PrimitiveType::P64x2 | PrimitiveType::List(_, _) => unreachable!(),
        }
    }

    fn llvm_agg_one<'a>(
        &self,
        ctx: &'a Context,
        _llvm_mod: &Module<'a>,
        b: &inkwell::builder::Builder<'a>,
        alloc_ptr: inkwell::values::PointerValue<'a>,
        ticket: inkwell::values::IntValue<'a>,
        value: inkwell::values::BasicValueEnum<'a>,
    ) {
        let sum_type = if self.pt.is_int() {
            ctx.i64_type().as_basic_type_enum()
        } else {
            ctx.f64_type().as_basic_type_enum()
        };
        let agg_ptr = increment_pointer!(ctx, b, alloc_ptr, 8, ticket);
        let value = if self.pt.is_int() {
            if self.pt.is_signed() {
                b.build_int_s_extend_or_bit_cast(value.into_int_value(), ctx.i64_type(), "value")
                    .unwrap()
                    .as_basic_value_enum()
            } else {
                b.build_int_z_extend_or_bit_cast(value.into_int_value(), ctx.i64_type(), "value")
                    .unwrap()
                    .as_basic_value_enum()
            }
        } else {
            b.build_float_ext(value.into_float_value(), ctx.f64_type(), "value")
                .unwrap()
                .as_basic_value_enum()
        };

        let curr_sum = b.build_load(sum_type, agg_ptr, "curr_sum").unwrap();
        let new_sum = if self.pt.is_int() {
            b.build_int_add(curr_sum.into_int_value(), value.into_int_value(), "new_sum")
                .unwrap()
                .as_basic_value_enum()
        } else {
            b.build_float_add(
                curr_sum.into_float_value(),
                value.into_float_value(),
                "new_sum",
            )
            .unwrap()
            .as_basic_value_enum()
        };
        b.build_store(agg_ptr, new_sum).unwrap();
    }

    fn llvm_agg_one_atomic<'a>(
        &self,
        ctx: &'a Context,
        _llvm_mod: &Module<'a>,
        b: &inkwell::builder::Builder<'a>,
        alloc_ptr: inkwell::values::PointerValue<'a>,
        ticket: inkwell::values::IntValue<'a>,
        value: inkwell::values::BasicValueEnum<'a>,
    ) -> bool {
        if !self.pt.is_int() {
            // TODO incl floats once inkwell adds support
            return false;
        }

        let agg_ptr = increment_pointer!(ctx, b, alloc_ptr, 8, ticket);
        let value = if self.pt.is_signed() {
            b.build_int_s_extend_or_bit_cast(value.into_int_value(), ctx.i64_type(), "value")
                .unwrap()
        } else {
            b.build_int_z_extend_or_bit_cast(value.into_int_value(), ctx.i64_type(), "value")
                .unwrap()
        };
        b.build_atomicrmw(
            AtomicRMWBinOp::Add,
            agg_ptr,
            value,
            AtomicOrdering::Monotonic,
        )
        .unwrap();

        true
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int64Type, Int32Array};
    use arrow_schema::DataType;
    use inkwell::{context::Context, OptimizationLevel};
    use itertools::Itertools;

    use crate::{
        compiled_iter::{datum_to_iter, generate_next},
        compiled_kernels::{
            aggregate::{sum::SumAgg, AggAlloc, Aggregation},
            link_req_helpers,
        },
        PrimitiveType,
    };

    #[test]
    fn test_sum_agg() {
        let tickets: Vec<u64> = vec![0, 1, 2, 3, 0, 0, 1];
        let data = Int32Array::from(vec![5, 6, 7, 8, 1, 1, 2]);

        let agg = SumAgg::new(&[PrimitiveType::I32]);
        let mut alloc = agg.allocate(4);

        let ctx = Context::create();
        let llvm_mod = ctx.create_module("sum_agg");
        let mut ih = datum_to_iter(&data).unwrap();
        let next_func = generate_next(&ctx, &llvm_mod, "next", &DataType::Int32, &ih).unwrap();
        let func = agg
            .llvm_agg_func(&ctx, &llvm_mod, next_func, false)
            .unwrap();

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
        let res = res.as_primitive::<Int64Type>();
        let res = res.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, vec![7, 8, 7, 8])
    }
}
