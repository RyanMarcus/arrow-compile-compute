use std::ffi::c_void;

use arrow_array::Array;
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    types::IntType,
    values::{IntValue, PointerValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};
use ouroboros::self_referencing;

use crate::{
    compiled_iter::{datum_to_iter, generate_next},
    compiled_kernels::{link_req_helpers, optimize_module},
    declare_blocks, set_noalias_params, ArrowKernelError, Kernel, PrimitiveType,
};

#[self_referencing]
pub struct RangeKernel {
    dt: DataType,
    context: Context,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut i128, *mut i128)>,
}
unsafe impl Send for RangeKernel {}
unsafe impl Sync for RangeKernel {}

impl RangeKernel {
    fn ensure_supported(dt: &DataType) -> Result<PrimitiveType, ArrowKernelError> {
        match dt {
            DataType::Int8 => Ok(PrimitiveType::I8),
            DataType::Int16 => Ok(PrimitiveType::I16),
            DataType::Int32 => Ok(PrimitiveType::I32),
            DataType::Int64 => Ok(PrimitiveType::I64),
            DataType::UInt8 => Ok(PrimitiveType::U8),
            DataType::UInt16 => Ok(PrimitiveType::U16),
            DataType::UInt32 => Ok(PrimitiveType::U32),
            DataType::UInt64 => Ok(PrimitiveType::U64),
            DataType::Dictionary(_, value_type) => Self::ensure_supported(value_type),
            DataType::RunEndEncoded(_, values) => Self::ensure_supported(values.data_type()),
            _ => Err(ArrowKernelError::UnsupportedArguments(format!(
                "range kernel only supports integer values (found {})",
                dt
            ))),
        }
    }

    fn llvm_update_min_max<'ctx>(
        builder: &Builder<'ctx>,
        value: IntValue<'ctx>,
        min_ptr: PointerValue<'ctx>,
        max_ptr: PointerValue<'ctx>,
        int_type: IntType<'ctx>,
        signed: bool,
    ) {
        let curr_min = builder
            .build_load(int_type, min_ptr, "curr_min")
            .unwrap()
            .into_int_value();
        let curr_max = builder
            .build_load(int_type, max_ptr, "curr_max")
            .unwrap()
            .into_int_value();

        let min_pred = if signed {
            IntPredicate::SLT
        } else {
            IntPredicate::ULT
        };
        let max_pred = if signed {
            IntPredicate::SGT
        } else {
            IntPredicate::UGT
        };

        let is_new_min = builder
            .build_int_compare(min_pred, value, curr_min, "is_new_min")
            .unwrap();
        let new_min = builder
            .build_select(is_new_min, value, curr_min, "next_min")
            .unwrap()
            .into_int_value();
        builder.build_store(min_ptr, new_min).unwrap();

        let is_new_max = builder
            .build_int_compare(max_pred, value, curr_max, "is_new_max")
            .unwrap();
        let new_max = builder
            .build_select(is_new_max, value, curr_max, "next_max")
            .unwrap()
            .into_int_value();
        builder.build_store(max_ptr, new_max).unwrap();
    }
}

impl Kernel for RangeKernel {
    type Key = DataType;

    type Input<'a> = &'a dyn Array;

    type Params = ();

    type Output = (u128, i128);

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        if inp.data_type() != self.borrow_dt() {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "range kernel expected {} but got {}",
                self.borrow_dt(),
                inp.data_type()
            )));
        }

        if inp.len() == 0 {
            return Err(ArrowKernelError::ArgumentMismatch(
                "range kernel requires a non-empty array".to_string(),
            ));
        }

        if inp.is_nullable() {
            return Err(ArrowKernelError::ArgumentMismatch(
                "range kernel requires a non-nullable array".to_string(),
            ));
        }

        let mut iter = datum_to_iter(&inp)?;
        let mut min = 0_i128;
        let mut max = 0_i128;

        unsafe {
            self.borrow_func()
                .call(iter.get_mut_ptr(), &mut min, &mut max);
        }

        assert!(max >= min, "computed max is smaller than min");

        let diff = max - min;
        Ok((diff as u128, min))
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let array = *inp;
        let dt = array.data_type();
        let pt = Self::ensure_supported(dt)?;
        RangeKernelTryBuilder {
            dt: dt.clone(),
            context: Context::create(),
            func_builder: |ctx| {
                let llvm_mod = ctx.create_module("range_kernel");

                let iter = datum_to_iter(&array)?;
                let next_func =
                    generate_next(ctx, &llvm_mod, "range_iter", dt, &iter).ok_or_else(|| {
                        ArrowKernelError::UnsupportedArguments(format!(
                            "range kernel could not generate iterator for {}",
                            dt
                        ))
                    })?;

                let ptr_type = ctx.ptr_type(AddressSpace::default());
                let func_ty = ctx
                    .void_type()
                    .fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
                let func = llvm_mod.add_function("range", func_ty, None);
                set_noalias_params(&func);

                let builder = ctx.create_builder();
                declare_blocks!(ctx, func, entry, loop_cond, loop_body, exit);

                let element_type = pt.llvm_type(ctx).into_int_type();
                let signed = pt.is_signed();

                builder.position_at_end(entry);
                let iter_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
                let out_min_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
                let out_max_ptr = func.get_nth_param(2).unwrap().into_pointer_value();
                let buf_ptr = builder.build_alloca(element_type, "buf").unwrap();
                let min_ptr = builder.build_alloca(element_type, "min_local").unwrap();
                let max_ptr = builder.build_alloca(element_type, "max_local").unwrap();

                builder
                    .build_call(next_func, &[iter_ptr.into(), buf_ptr.into()], "get_first")
                    .unwrap();

                let first_value = builder
                    .build_load(element_type, buf_ptr, "first_value")
                    .unwrap()
                    .into_int_value();
                builder.build_store(min_ptr, first_value).unwrap();
                builder.build_store(max_ptr, first_value).unwrap();
                builder.build_unconditional_branch(loop_cond).unwrap();

                builder.position_at_end(loop_cond);
                let has_next = builder
                    .build_call(next_func, &[iter_ptr.into(), buf_ptr.into()], "has_next")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                builder
                    .build_conditional_branch(has_next, loop_body, exit)
                    .unwrap();

                builder.position_at_end(loop_body);
                let current_value = builder
                    .build_load(element_type, buf_ptr, "current_value")
                    .unwrap()
                    .into_int_value();
                RangeKernel::llvm_update_min_max(
                    &builder,
                    current_value,
                    min_ptr,
                    max_ptr,
                    element_type,
                    signed,
                );
                builder.build_unconditional_branch(loop_cond).unwrap();

                builder.position_at_end(exit);
                let current_min = builder
                    .build_load(element_type, min_ptr, "min")
                    .unwrap()
                    .into_int_value();
                let current_max = builder
                    .build_load(element_type, max_ptr, "max")
                    .unwrap()
                    .into_int_value();
                let i128_type = ctx.i128_type();
                let min_ext = if signed {
                    builder
                        .build_int_s_extend(current_min, i128_type, "min_ext")
                        .unwrap()
                } else {
                    builder
                        .build_int_z_extend(current_min, i128_type, "min_ext")
                        .unwrap()
                };
                let max_ext = if signed {
                    builder
                        .build_int_s_extend(current_max, i128_type, "max_ext")
                        .unwrap()
                } else {
                    builder
                        .build_int_z_extend(current_max, i128_type, "max_ext")
                        .unwrap()
                };
                builder.build_store(out_min_ptr, min_ext).unwrap();
                builder.build_store(out_max_ptr, max_ext).unwrap();
                builder.build_return(None).unwrap();

                llvm_mod.verify().unwrap();
                optimize_module(&llvm_mod)?;
                let ee = llvm_mod
                    .create_jit_execution_engine(OptimizationLevel::Aggressive)
                    .unwrap();
                link_req_helpers(&llvm_mod, &ee).unwrap();

                let jit_fn = unsafe {
                    ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i128, *mut i128)>(
                        func.get_name().to_str().unwrap(),
                    )
                    .unwrap()
                };

                Ok(jit_fn)
            },
        }
        .try_build()
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok(i.data_type().clone())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int32Array, UInt64Array};

    use super::RangeKernel;
    use crate::Kernel;

    #[test]
    fn test_signed_range() {
        let data = Int32Array::from(vec![1, -10, 5, 100]);
        let kernel = RangeKernel::compile(&(&data as &dyn arrow_array::Array), ()).unwrap();
        let (range, min) = kernel.call(&data).unwrap();
        assert_eq!(range, 110);
        assert_eq!(min, -10);
    }

    #[test]
    fn test_unsigned_range() {
        let data = UInt64Array::from(vec![5_u64, 10, 255, 64]);
        let kernel = RangeKernel::compile(&(&data as &dyn arrow_array::Array), ()).unwrap();
        let (range, min) = kernel.call(&data).unwrap();
        assert_eq!(range, 250);
        assert_eq!(min, 5);
    }
}
