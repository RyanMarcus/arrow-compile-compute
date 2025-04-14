use arrow_array::{BooleanArray, Datum};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType;
use inkwell::execution_engine::JitFunction;
use inkwell::module::{Linkage, Module};
use inkwell::values::FunctionValue;
use inkwell::OptimizationLevel;
use inkwell::{context::Context, AddressSpace};
use ouroboros::self_referencing;
use std::ffi::c_void;

use crate::new_iter::{datum_to_iter, generate_next, generate_next_block, IteratorHolder};
use crate::{declare_blocks, increment_pointer, ComparisonType, Predicate, PrimitiveType};

use super::ArrowKernelError;

#[self_referencing]
pub struct ComparisonKernel {
    context: Context,
    lhs_data_type: DataType,
    rhs_data_type: DataType,
    rhs_scalar: bool,
    pred: Predicate,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u64)>,
}

impl ComparisonKernel {
    pub fn compile(
        lhs: &dyn Datum,
        rhs: &dyn Datum,
        pred: Predicate,
    ) -> Result<ComparisonKernel, ArrowKernelError> {
        let (lhs_arr, lhs_scalar) = lhs.get();
        let (rhs_arr, rhs_scalar) = rhs.get();

        if lhs_scalar && rhs_scalar {
            return Err(ArrowKernelError::UnsupportedArguments(
                "scalar-scalar comparison".to_string(),
            ));
        }

        if lhs_scalar && !rhs_scalar {
            return ComparisonKernel::compile(rhs, lhs, pred.flip());
        }

        let lhs_iter = datum_to_iter(lhs)?;
        let rhs_iter = datum_to_iter(rhs)?;
        let ctx = Context::create();
        Ok(ComparisonKernelBuilder {
            context: ctx,
            lhs_data_type: lhs_arr.data_type().clone(),
            rhs_data_type: rhs_arr.data_type().clone(),
            rhs_scalar,
            pred,
            func_builder: |ctx| generate_llvm_cmp_kernel(ctx, lhs, &lhs_iter, rhs, &rhs_iter, pred),
        }
        .build())
    }

    pub fn call(&self, a: &dyn Datum, b: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        let (a_arr, a_scalar) = a.get();
        let (b_arr, b_scalar) = b.get();
        assert!(
            !a_scalar,
            "should swap scalar to 2nd argument (this is a bug)"
        );

        if !b_scalar && (a_arr.len() != b_arr.len()) {
            return Err(ArrowKernelError::SizeMismatch);
        }

        if !self.with_rhs_scalar(|rhs_scalar| b_scalar == *rhs_scalar) {
            return Err(ArrowKernelError::ArgumentMismatch);
        }

        if !self.with_lhs_data_type(|lhs_dt| a_arr.data_type() == lhs_dt) {
            return Err(ArrowKernelError::ArgumentMismatch);
        }

        if !self.with_lhs_data_type(|rhs_dt| b_arr.data_type() == rhs_dt) {
            return Err(ArrowKernelError::ArgumentMismatch);
        }

        let mut a_iter = datum_to_iter(a)?;
        let mut b_iter = datum_to_iter(b)?;
        a_iter.assert_non_null();
        b_iter.assert_non_null();
        let mut out = vec![0_u64; a_arr.len().div_ceil(64)];

        self.with_func(|func| unsafe {
            func.call(a_iter.get_mut_ptr(), b_iter.get_mut_ptr(), out.as_mut_ptr())
        });

        let buf = BooleanBuffer::new(out.into(), 0, a_arr.len());
        Ok(BooleanArray::new(
            buf,
            NullBuffer::union(a_arr.nulls(), b_arr.nulls()),
        ))
    }
}

fn generate_llvm_cmp_kernel<'a>(
    ctx: &'a Context,
    lhs: &dyn Datum,
    lhs_iter: &IteratorHolder,
    rhs: &dyn Datum,
    rhs_iter: &IteratorHolder,
    pred: Predicate,
) -> JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u64)> {
    let (lhs_arr, _lhs_scalar) = lhs.get();
    let (rhs_arr, _rhs_scalar) = rhs.get();
    let lhs_dt = lhs_arr.data_type();
    let rhs_dt = rhs_arr.data_type();

    let module = ctx.create_module("cmp_kernel");
    let build = ctx.create_builder();
    let i64_type = ctx.i64_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let void_type = ctx.void_type();
    let lhs_prim = PrimitiveType::for_arrow_type(lhs_dt);
    let rhs_prim = PrimitiveType::for_arrow_type(rhs_dt);
    let lhs_vec = lhs_prim.llvm_vec_type(&ctx, 64).unwrap();
    let rhs_vec = rhs_prim.llvm_vec_type(&ctx, 64).unwrap();
    let lhs_llvm = lhs_prim.llvm_type(&ctx);
    let rhs_llvm = rhs_prim.llvm_type(&ctx);
    let dom_prim_type = PrimitiveType::dominant(lhs_prim, rhs_prim).unwrap();

    let lhs_next_block =
        generate_next_block::<64>(&ctx, &module, "next_lhs_block", lhs_dt, lhs_iter).unwrap();
    let rhs_next_block =
        generate_next_block::<64>(&ctx, &module, "next_rhs_block", rhs_dt, rhs_iter).unwrap();
    let lhs_next = generate_next(&ctx, &module, "next_lhs", lhs_dt, lhs_iter).unwrap();
    let rhs_next = generate_next(&ctx, &module, "next_rhs", rhs_dt, rhs_iter).unwrap();

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

    let res = match dom_prim_type.comparison_type() {
        ComparisonType::Int { signed } => build
            .build_int_compare(pred.as_int_pred(signed), lvec, rvec, "block_cmp_result")
            .unwrap(),
        ComparisonType::Float => {
            let convert = add_float_vec_to_int_vec(ctx, &module, 64, dom_prim_type);
            let lhs = build
                .build_call(convert, &[lvec.into()], "lhs_converted")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_vector_value();
            let rhs = build
                .build_call(convert, &[rvec.into()], "rhs_converted")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_vector_value();
            build
                .build_int_compare(pred.as_int_pred(true), lhs, rhs, "block_cmp_result")
                .unwrap()
        }
        ComparisonType::String => unreachable!("no block comparison for strings"),
    };

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
    let lv = build.build_load(lhs_llvm, lhs_single_buf, "lv").unwrap();
    let rv = build.build_load(rhs_llvm, rhs_single_buf, "rv").unwrap();

    let res = match dom_prim_type.comparison_type() {
        ComparisonType::Int { signed } => build
            .build_int_compare(
                pred.as_int_pred(signed),
                lv.into_int_value(),
                rv.into_int_value(),
                "cmp_single",
            )
            .unwrap(),
        ComparisonType::Float => {
            let convert = add_float_to_int(ctx, &module, dom_prim_type);
            let lhs = build
                .build_call(convert, &[lv.into()], "lhs_converted")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left();
            let rhs = build
                .build_call(convert, &[rv.into()], "rhs_converted")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left();
            build
                .build_int_compare(
                    pred.as_int_pred(true),
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                    "cmp_single",
                )
                .unwrap()
        }
        ComparisonType::String => todo!(),
    };

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

fn add_float_vec_to_int_vec<'a>(
    ctx: &'a Context,
    module: &Module<'a>,
    v_size: u32,
    ptype: PrimitiveType,
) -> FunctionValue<'a> {
    // apply the following algorithm to get a total float ordering
    // left ^= (((left >> 63) as u64) >> 1) as i64;
    // right ^= (((right >> 63) as u64) >> 1) as i64;
    // left.cmp(right)

    let inp_type = ptype.llvm_vec_type(ctx, v_size).unwrap();
    let int_type = PrimitiveType::int_with_width(ptype.width())
        .llvm_type(ctx)
        .into_int_type();
    let ret_type = int_type.vec_type(v_size);

    let func_type = ret_type.fn_type(&[inp_type.into()], false);
    let func = module.add_function("fvec_to_int", func_type, Some(Linkage::Private));
    let fvec = func.get_nth_param(0).unwrap().into_vector_value();

    declare_blocks!(ctx, func, entry);
    let builder = ctx.create_builder();
    builder.position_at_end(entry);

    let vminus1 = builder
        .build_insert_element(
            int_type.vec_type(v_size).const_zero(),
            int_type.const_int(ptype.width() as u64 * 8 - 1, false),
            int_type.const_zero(),
            "vminus1",
        )
        .unwrap();
    let vminus1 = builder
        .build_shuffle_vector(
            vminus1,
            int_type.vec_type(v_size).get_undef(),
            int_type.vec_type(v_size).const_zero(),
            "vminus1_bcast",
        )
        .unwrap();
    let v1 = builder
        .build_insert_element(
            int_type.vec_type(v_size).const_zero(),
            int_type.const_int(1, false),
            int_type.const_zero(),
            "v1",
        )
        .unwrap();
    let v1 = builder
        .build_shuffle_vector(
            v1,
            int_type.vec_type(v_size).get_undef(),
            int_type.vec_type(v_size).const_zero(),
            "v1_bcast",
        )
        .unwrap();

    let cleft = builder
        .build_bit_cast(fvec, int_type.vec_type(v_size), "cleft")
        .unwrap()
        .into_vector_value();
    let left = builder
        .build_right_shift(cleft, vminus1, true, "sleft")
        .unwrap();
    let left = builder.build_right_shift(left, v1, false, "sleft").unwrap();
    let res = builder.build_xor(cleft, left, "left").unwrap();
    builder.build_return(Some(&res)).unwrap();

    return func;
}

fn add_float_to_int<'a>(
    ctx: &'a Context,
    module: &Module<'a>,
    ptype: PrimitiveType,
) -> FunctionValue<'a> {
    // apply the following algorithm to get a total float ordering
    // left ^= (((left >> 63) as u64) >> 1) as i64;
    // right ^= (((right >> 63) as u64) >> 1) as i64;
    // left.cmp(right)

    let inp_type = ptype.llvm_type(ctx);
    let int_type = PrimitiveType::int_with_width(ptype.width())
        .llvm_type(ctx)
        .into_int_type();

    let func_type = int_type.fn_type(&[inp_type.into()], false);
    let func = module.add_function("vec_to_int", func_type, Some(Linkage::Private));
    let float = func.get_nth_param(0).unwrap().into_float_value();

    declare_blocks!(ctx, func, entry);
    let builder = ctx.create_builder();
    builder.position_at_end(entry);

    let vminus1 = int_type.const_int(ptype.width() as u64 * 8 - 1, false);
    let v1 = int_type.const_int(1, false);

    let cleft = builder
        .build_bit_cast(float, int_type, "cleft")
        .unwrap()
        .into_int_value();
    let left = builder
        .build_right_shift(cleft, vminus1, true, "sleft")
        .unwrap();
    let left = builder.build_right_shift(left, v1, false, "sleft").unwrap();
    let res = builder.build_xor(cleft, left, "left").unwrap();
    builder.build_return(Some(&res)).unwrap();

    return func;
}

#[cfg(test)]
mod tests {
    use arrow_array::{BooleanArray, Float32Array, Scalar, UInt32Array};
    use itertools::Itertools;

    use crate::{new_kernels::cmp::ComparisonKernel, Predicate};

    #[test]
    fn test_num_num_cmp() {
        let a = UInt32Array::from(vec![1, 2, 3]);
        let b = UInt32Array::from(vec![11, 0, 13]);
        let k = ComparisonKernel::compile(&a, &b, Predicate::Lt).unwrap();
        let r = k.call(&a, &b).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true, false, true]))
    }

    #[test]
    fn test_num_num_block_cmp() {
        let a = UInt32Array::from((0..100).collect_vec());
        let b = UInt32Array::from((1..101).collect_vec());
        let k = ComparisonKernel::compile(&a, &b, Predicate::Lt).unwrap();
        let r = k.call(&a, &b).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true; 100]))
    }

    #[test]
    fn test_num_num_block_scalar_cmp() {
        let a = UInt32Array::from((0..100).collect_vec());
        let b = Scalar::new(UInt32Array::from(vec![5]));
        let k = ComparisonKernel::compile(&a, &b, Predicate::Lt).unwrap();
        let r = k.call(&a, &b).unwrap();

        let expected_result = (0..100).map(|i| i < 5).collect::<Vec<bool>>();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_num_num_float_cmp() {
        let a = Float32Array::from((0..100).map(|i| i as f32).collect_vec());
        let b = Float32Array::from((1..101).map(|i| i as f32).collect_vec());
        let k = ComparisonKernel::compile(&a, &b, Predicate::Lt).unwrap();
        let r = k.call(&a, &b).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true; 100]))
    }

    #[test]
    fn test_num_num_float_scalar_cmp() {
        let a = Float32Array::from((0..100).map(|i| i as f32).collect_vec());
        let b = Scalar::new(Float32Array::from(vec![5.0]));
        let k = ComparisonKernel::compile(&a, &b, Predicate::Lt).unwrap();
        let r = k.call(&a, &b).unwrap();

        let expected_result = (0..100).map(|i| i < 5).collect::<Vec<bool>>();
        assert_eq!(r, BooleanArray::from(expected_result));
    }
}
