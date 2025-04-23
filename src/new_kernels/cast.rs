use std::ffi::c_void;

use arrow_array::{make_array, Array, ArrayRef};
use arrow_buffer::Buffer;
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use inkwell::{context::Context, execution_engine::JitFunction, AddressSpace, OptimizationLevel};
use ouroboros::self_referencing;

use crate::{
    declare_blocks, empty_array_for, increment_pointer,
    new_iter::{datum_to_iter, generate_next, generate_next_block, IteratorHolder},
    new_kernels::{gen_convert_numeric_vec, optimize_module},
    PrimitiveType,
};

use super::{ArrowKernelError, Kernel};

#[self_referencing]
pub struct CastKernel {
    context: Context,
    lhs_data_type: DataType,
    tar_data_type: DataType,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut c_void)>,
}

unsafe impl Sync for CastKernel {}
unsafe impl Send for CastKernel {}

impl Kernel for CastKernel {
    type Key = (DataType, DataType);

    type Input<'a> = &'a dyn Array;

    type Params = DataType;

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        if inp.data_type() != self.borrow_lhs_data_type() {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "kernel expected {:?}, got {:?}",
                self.borrow_lhs_data_type(),
                inp.data_type()
            )));
        }

        if inp.len() == 0 {
            return Ok(empty_array_for(self.borrow_tar_data_type()));
        }

        let mut inp_iter = datum_to_iter(&inp)?;
        if self.borrow_tar_data_type().is_primitive() {
            let out_prim = PrimitiveType::for_arrow_type(self.borrow_tar_data_type());
            let out_size = inp.len() * out_prim.width();
            let mut buf = vec![0u8; out_size];
            let arr_data = unsafe {
                self.borrow_func()
                    .call(inp_iter.get_mut_ptr(), buf.as_mut_ptr() as *mut c_void);

                ArrayDataBuilder::new(self.borrow_tar_data_type().clone())
                    .nulls(inp.nulls().cloned())
                    .buffers(vec![Buffer::from(buf)])
                    .len(inp.len())
                    .build_unchecked()
            };

            return Ok(make_array(arr_data));
        }

        todo!()
    }

    fn compile(inp: Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let in_type = inp.data_type();
        let out_type = &params;
        assert_ne!(
            in_type, out_type,
            "cannot compile kernel for identical types {:?}",
            in_type
        );

        let in_iter = datum_to_iter(&inp)?;

        let ctx = Context::create();
        if out_type.is_primitive() {
            CastKernelTryBuilder {
                context: ctx,
                lhs_data_type: inp.data_type().clone(),
                tar_data_type: out_type.clone(),
                func_builder: |ctx| generate_block_cast_to_flat(ctx, inp, &in_iter, out_type),
            }
            .try_build()
        } else {
            todo!()
        }
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((i.data_type().clone(), p.clone()))
    }
}

pub fn generate_block_cast_to_flat<'a>(
    ctx: &'a Context,
    lhs: &dyn Array,
    lhs_iter: &IteratorHolder,
    to: &DataType,
) -> Result<JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void)>, ArrowKernelError> {
    let lhs_prim = PrimitiveType::for_arrow_type(lhs.data_type());
    let lhs_type = lhs_prim.llvm_type(ctx);
    let to_prim = PrimitiveType::for_arrow_type(to);
    let ptr_type = ctx.ptr_type(AddressSpace::default());

    let build = ctx.create_builder();
    let module = ctx.create_module("cmp_kernel");

    // assume a 512-bit vector width (will still work for smaller vector widths)
    let vec_type = lhs_prim
        .llvm_vec_type(ctx, 32)
        .ok_or_else(|| ArrowKernelError::NonVectorizableType(lhs.data_type().clone()))?;
    let next_block = generate_next_block::<32>(ctx, &module, "cast", lhs.data_type(), lhs_iter)
        .ok_or_else(|| ArrowKernelError::NonVectorizableType(lhs.data_type().clone()))?;

    let next = generate_next(ctx, &module, "cast", lhs.data_type(), lhs_iter).unwrap();

    let fn_type = ctx
        .void_type()
        .fn_type(&[ptr_type.into(), ptr_type.into()], false);
    let function = module.add_function("cast", fn_type, None);
    let lhs_iter_ptr = function.get_nth_param(0).unwrap().into_pointer_value();

    declare_blocks!(ctx, function, entry, block_cond, block_body, tail_cond, tail_body, exit);
    build.position_at_end(entry);
    let out_ptr_ptr = build.build_alloca(ptr_type, "out_ptr").unwrap();
    build
        .build_store(
            out_ptr_ptr,
            function.get_nth_param(1).unwrap().into_pointer_value(),
        )
        .unwrap();
    let vbuf = build.build_alloca(vec_type, "vbuf").unwrap();
    let buf = build.build_alloca(lhs_type, "buf").unwrap();
    build.build_unconditional_branch(block_cond).unwrap();

    build.position_at_end(block_cond);
    let had_next = build
        .build_call(next_block, &[lhs_iter_ptr.into(), vbuf.into()], "get_next")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_conditional_branch(had_next, block_body, tail_cond)
        .unwrap();

    build.position_at_end(block_body);
    let data = build
        .build_load(vec_type, vbuf, "data")
        .unwrap()
        .into_vector_value();
    let converted = gen_convert_numeric_vec(ctx, &build, data, lhs_prim, to_prim);
    let out_ptr = build
        .build_load(ptr_type, out_ptr_ptr, "out_ptr")
        .unwrap()
        .into_pointer_value();
    build.build_store(out_ptr, converted).unwrap();
    let new_out_ptr = increment_pointer!(
        ctx,
        build,
        out_ptr,
        to_prim.width() * vec_type.get_size() as usize
    );
    build.build_store(out_ptr_ptr, new_out_ptr).unwrap();
    build.build_unconditional_branch(block_cond).unwrap();

    build.position_at_end(tail_cond);
    let had_next = build
        .build_call(next, &[lhs_iter_ptr.into(), buf.into()], "get_next")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_conditional_branch(had_next, tail_body, exit)
        .unwrap();

    build.position_at_end(tail_body);
    let data = build.build_load(lhs_type, buf, "data").unwrap();
    let data = build
        .build_bit_cast(
            data,
            lhs_prim.llvm_vec_type(ctx, 1).unwrap(),
            "singleton_vec",
        )
        .unwrap()
        .into_vector_value();

    let converted = gen_convert_numeric_vec(ctx, &build, data, lhs_prim, to_prim);
    let out_ptr = build
        .build_load(ptr_type, out_ptr_ptr, "out_ptr")
        .unwrap()
        .into_pointer_value();
    build.build_store(out_ptr, converted).unwrap();
    let new_out_ptr = increment_pointer!(ctx, build, out_ptr, to_prim.width());
    build.build_store(out_ptr_ptr, new_out_ptr).unwrap();
    build.build_unconditional_branch(tail_cond).unwrap();

    build.position_at_end(exit);
    build.build_return(None).unwrap();

    module.verify().unwrap();
    optimize_module(&module)?;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    Ok(unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void)>(
            function.get_name().to_str().unwrap(),
        )
        .unwrap()
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, Int64Array, UInt8Array};
    use arrow_schema::DataType;
    use itertools::Itertools;

    use crate::{new_kernels::cast::CastKernel, Kernel};

    #[test]
    fn test_i32_to_i64() {
        let data = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let expected: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
        let k = CastKernel::compile(&data, DataType::Int64).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i32_to_i64_block() {
        let data = Int32Array::from((0..200).collect_vec());
        let expected: ArrayRef = Arc::new(Int64Array::from((0..200).collect_vec()));
        let k = CastKernel::compile(&data, DataType::Int64).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i64_to_u8_block() {
        let data = Int64Array::from((0..200).collect_vec());
        let expected: ArrayRef = Arc::new(UInt8Array::from((0..200).collect_vec()));
        let k = CastKernel::compile(&data, DataType::UInt8).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }
}
