use arrow_array::{BooleanArray, Datum};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType;
use inkwell::execution_engine::JitFunction;
use inkwell::{context::Context, AddressSpace};
use inkwell::{IntPredicate, OptimizationLevel};
use ouroboros::self_referencing;
use std::ffi::c_void;

use crate::new_iter::{array_to_iter, generate_next, generate_next_block, IteratorHolder};
use crate::{declare_blocks, increment_pointer, PrimitiveType};

use super::ArrowKernelError;

#[self_referencing]
pub struct ComparisonKernel {
    context: Context,
    lhs_data_type: DataType,
    rhs_data_type: DataType,
    lhs_scalar: bool,
    rhs_scalar: bool,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u64)>,
}

impl ComparisonKernel {
    pub fn compile(
        lhs: &DataType,
        lhs_iter: &IteratorHolder,
        lhs_scalar: bool,
        rhs: &DataType,
        rhs_iter: &IteratorHolder,
        rhs_scalar: bool,
    ) -> ComparisonKernel {
        let ctx = Context::create();
        ComparisonKernelBuilder {
            context: ctx,
            lhs_data_type: lhs.clone(),
            rhs_data_type: rhs.clone(),
            lhs_scalar,
            rhs_scalar,
            func_builder: |ctx| {
                generate_llvm_cmp_kernel(ctx, lhs, lhs_iter, lhs_scalar, rhs, rhs_iter, rhs_scalar)
            },
        }
        .build()
    }

    pub fn call(&self, a: &dyn Datum, b: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        let (a, a_scalar) = a.get();
        let (b, b_scalar) = b.get();

        if (!a_scalar && !b_scalar) && (a.len() != b.len()) {
            return Err(ArrowKernelError::SizeMismatch);
        }

        if !self.with_lhs_data_type(|lhs_dt| a.data_type() == lhs_dt) {
            return Err(ArrowKernelError::ArgumentMismatch);
        }

        if !self.with_lhs_data_type(|rhs_dt| b.data_type() == rhs_dt) {
            return Err(ArrowKernelError::ArgumentMismatch);
        }

        let mut a_iter = array_to_iter(a);
        let mut b_iter = array_to_iter(b);
        let mut out = vec![0_u64; a.len().div_ceil(64)];

        self.with_func(|func| unsafe {
            func.call(a_iter.get_mut_ptr(), b_iter.get_mut_ptr(), out.as_mut_ptr())
        });

        let buf = BooleanBuffer::new(out.into(), 0, a.len());
        Ok(BooleanArray::new(
            buf,
            NullBuffer::union(a.nulls(), b.nulls()),
        ))
    }
}

fn generate_llvm_cmp_kernel<'a>(
    ctx: &'a Context,
    lhs: &DataType,
    lhs_iter: &IteratorHolder,
    lhs_scalar: bool,
    rhs: &DataType,
    rhs_iter: &IteratorHolder,
    rhs_scalar: bool,
) -> JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u64)> {
    let module = ctx.create_module("cmp_kernel");
    let build = ctx.create_builder();
    let i64_type = ctx.i64_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let void_type = ctx.void_type();
    let lhs_prim = PrimitiveType::for_arrow_type(lhs);
    let rhs_prim = PrimitiveType::for_arrow_type(rhs);
    let lhs_vec = lhs_prim.llvm_vec_type(&ctx, 64).unwrap();
    let rhs_vec = rhs_prim.llvm_vec_type(&ctx, 64).unwrap();
    let lhs_llvm = lhs_prim.llvm_type(&ctx);
    let rhs_llvm = rhs_prim.llvm_type(&ctx);

    let lhs_next_block =
        generate_next_block::<64>(&ctx, &module, "next_lhs_block", lhs, lhs_iter).unwrap();
    let rhs_next_block =
        generate_next_block::<64>(&ctx, &module, "next_rhs_block", rhs, rhs_iter).unwrap();
    let lhs_next = generate_next(&ctx, &module, "next_lhs", lhs, lhs_iter).unwrap();
    let rhs_next = generate_next(&ctx, &module, "next_rhs", rhs, rhs_iter).unwrap();

    let fn_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
    let cmp = module.add_function("cmp", fn_type, None);
    let lhs_ptr = cmp.get_nth_param(0).unwrap();
    let rhs_ptr = cmp.get_nth_param(1).unwrap();
    let out_ptr = cmp.get_nth_param(2).unwrap();

    declare_blocks!(ctx, cmp, entry, block_cond, block_body, tail_cond, tail_body, exit);

    build.position_at_end(entry);
    let out_ptr_ptr = build.build_alloca(ptr_type, "out_ptr_ptr").unwrap();
    build.build_store(out_ptr_ptr, out_ptr).unwrap();

    let lhs_vec_buf = build.build_alloca(lhs_vec, "lhs_vec_buf").unwrap();
    let rhs_vec_buf = build.build_alloca(rhs_vec, "rhs_vec_buf").unwrap();
    let lhs_single_buf = build.build_alloca(lhs_llvm, "lhs_single_buf").unwrap();
    let rhs_single_buf = build.build_alloca(rhs_llvm, "rhs_single_buf").unwrap();
    let tail_buf_ptr = build.build_alloca(i64_type, "tail_buf_ptr").unwrap();
    build
        .build_store(tail_buf_ptr, i64_type.const_zero())
        .unwrap();
    let tail_buf_idx_ptr = build.build_alloca(i64_type, "tail_buf_idx").unwrap();
    build
        .build_store(tail_buf_idx_ptr, i64_type.const_zero())
        .unwrap();
    build.build_unconditional_branch(block_cond).unwrap();

    build.position_at_end(block_cond);
    let had_lhs = build
        .build_call(
            lhs_next_block,
            &[lhs_ptr.into(), lhs_vec_buf.into()],
            "lhs_next",
        )
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_call(
            rhs_next_block,
            &[rhs_ptr.into(), rhs_vec_buf.into()],
            "rhs_next",
        )
        .unwrap();
    build
        .build_conditional_branch(had_lhs, block_body, tail_cond)
        .unwrap();

    build.position_at_end(block_body);
    let lvec = build
        .build_load(lhs_vec, lhs_vec_buf, "lvec")
        .unwrap()
        .into_vector_value();
    let rvec = build
        .build_load(rhs_vec, rhs_vec_buf, "rvec")
        .unwrap()
        .into_vector_value();
    let res = build
        .build_int_compare(IntPredicate::ULT, lvec, rvec, "block_cmp_result")
        .unwrap();
    let res = build.build_bit_cast(res, i64_type, "res_u64").unwrap();
    let out_ptr = build
        .build_load(ptr_type, out_ptr_ptr, "out_ptr")
        .unwrap()
        .into_pointer_value();
    build.build_store(out_ptr, res).unwrap();
    let next_out_ptr = increment_pointer!(ctx, build, out_ptr, 8);
    build.build_store(out_ptr_ptr, next_out_ptr).unwrap();
    build.build_unconditional_branch(block_cond).unwrap();

    build.position_at_end(tail_cond);
    let had_lhs = build
        .build_call(
            lhs_next,
            &[lhs_ptr.into(), lhs_single_buf.into()],
            "lhs_next",
        )
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_call(
            rhs_next,
            &[rhs_ptr.into(), rhs_single_buf.into()],
            "rhs_next",
        )
        .unwrap();
    build
        .build_conditional_branch(had_lhs, tail_body, exit)
        .unwrap();

    build.position_at_end(tail_body);
    let lv = build
        .build_load(lhs_llvm, lhs_single_buf, "lv")
        .unwrap()
        .into_int_value();
    let rv = build
        .build_load(rhs_llvm, rhs_single_buf, "rv")
        .unwrap()
        .into_int_value();
    let res = build
        .build_int_compare(IntPredicate::ULT, lv, rv, "cmp_single")
        .unwrap();
    let res = build
        .build_int_z_extend_or_bit_cast(res, i64_type, "casted_cmp")
        .unwrap();

    let tail_buf_idx = build
        .build_load(i64_type, tail_buf_idx_ptr, "tail_buf_idx")
        .unwrap()
        .into_int_value();
    let res = build
        .build_left_shift(res, tail_buf_idx, "shifted")
        .unwrap();
    let tail_buf = build
        .build_load(i64_type, tail_buf_ptr, "tail_buf")
        .unwrap()
        .into_int_value();
    let new_tail_buf = build.build_or(res, tail_buf, "new_tail_buf").unwrap();
    build.build_store(tail_buf_ptr, new_tail_buf).unwrap();
    let new_tail_buf_idx = build
        .build_int_add(
            tail_buf_idx,
            i64_type.const_int(1, false),
            "new_tail_buf_idx",
        )
        .unwrap();
    build
        .build_store(tail_buf_idx_ptr, new_tail_buf_idx)
        .unwrap();
    let out_ptr = build
        .build_load(ptr_type, out_ptr_ptr, "out_ptr")
        .unwrap()
        .into_pointer_value();

    build.build_store(out_ptr, new_tail_buf).unwrap();
    build.build_unconditional_branch(tail_cond).unwrap();

    build.position_at_end(exit);
    build.build_return(None).unwrap();

    module.verify().unwrap();
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u64)>(
            cmp.get_name().to_str().unwrap(),
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, BooleanArray, UInt32Array};
    use itertools::Itertools;

    use crate::{new_iter::array_to_iter, new_kernels::cmp::ComparisonKernel};

    #[test]
    fn test_num_num_cmp() {
        let a = UInt32Array::from(vec![1, 2, 3]);
        let b = UInt32Array::from(vec![11, 0, 13]);
        let a_iter = array_to_iter(&a);
        let b_iter = array_to_iter(&b);
        let k =
            ComparisonKernel::compile(a.data_type(), &a_iter, false, b.data_type(), &b_iter, false);
        let r = k.call(&a, &b).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true, false, true]))
    }

    #[test]
    fn test_num_num_block_cmp() {
        let a = UInt32Array::from((0..100).collect_vec());
        let b = UInt32Array::from((1..101).collect_vec());
        let a_iter = array_to_iter(&a);
        let b_iter = array_to_iter(&b);
        let k =
            ComparisonKernel::compile(a.data_type(), &a_iter, false, b.data_type(), &b_iter, false);
        let r = k.call(&a, &b).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true; 100]))
    }
}
