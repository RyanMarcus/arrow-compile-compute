use arrow_array::{BooleanArray, Datum};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use inkwell::execution_engine::JitFunction;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::{Linkage, Module};

use inkwell::values::{BasicValue, FunctionValue};
use inkwell::{context::Context, AddressSpace};
use inkwell::{IntPredicate, OptimizationLevel};
use ouroboros::self_referencing;
use std::ffi::c_void;

use crate::new_iter::{datum_to_iter, generate_next, generate_next_block, IteratorHolder};
use crate::new_kernels::writers::{ArrayWriter, BooleanWriter, WriterAllocation};
use crate::new_kernels::{gen_convert_numeric_vec, optimize_module};
use crate::{
    declare_blocks, increment_pointer, pointer_diff, ComparisonType, Predicate, PrimitiveType,
};

use super::{ArrowKernelError, Kernel};

#[self_referencing]
pub struct ComparisonKernel {
    context: Context,
    lhs_data_type: DataType,
    rhs_data_type: DataType,
    rhs_scalar: bool,
    pred: Predicate,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void)>,
}

unsafe impl Sync for ComparisonKernel {}
unsafe impl Send for ComparisonKernel {}

impl Kernel for ComparisonKernel {
    type Key = (DataType, DataType, bool, Predicate);
    type Input<'a> = (&'a dyn Datum, &'a dyn Datum);
    type Params = Predicate;
    type Output = BooleanArray;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (a, b) = inp;
        let (a_arr, a_scalar) = a.get();
        let (b_arr, b_scalar) = b.get();
        assert!(
            !a_scalar,
            "should swap scalar to 2nd argument (this is a bug)"
        );

        if !b_scalar && (a_arr.len() != b_arr.len()) {
            return Err(ArrowKernelError::SizeMismatch);
        }

        if a_arr.is_empty() {
            return Ok(BooleanArray::new_null(0));
        }

        if !self.with_rhs_scalar(|rhs_scalar| b_scalar == *rhs_scalar) {
            return Err(ArrowKernelError::ArgumentMismatch(
                "Expected RHS to be scalar".to_string(),
            ));
        }

        if !self.with_lhs_data_type(|lhs_dt| a_arr.data_type() == lhs_dt) {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "LHS did not match, expected {:?}, found {:?}",
                self.borrow_lhs_data_type(),
                a_arr.data_type()
            )));
        }

        if !self.with_rhs_data_type(|rhs_dt| b_arr.data_type() == rhs_dt) {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "RHS did not match, expected {:?}, found {:?}",
                self.borrow_rhs_data_type(),
                b_arr.data_type()
            )));
        }

        let mut a_iter = datum_to_iter(a)?;
        let mut b_iter = datum_to_iter(b)?;
        let mut alloc = BooleanWriter::allocate(a_arr.len(), PrimitiveType::U8);

        self.with_func(|func| unsafe {
            func.call(a_iter.get_mut_ptr(), b_iter.get_mut_ptr(), alloc.get_ptr())
        });

        Ok(alloc.to_array(a_arr.len(), NullBuffer::union(a_arr.nulls(), b_arr.nulls())))
    }

    fn compile(inp: &Self::Input<'_>, pred: Predicate) -> Result<Self, ArrowKernelError> {
        let (lhs, rhs) = *inp;
        let (lhs_arr, lhs_scalar) = lhs.get();
        let (rhs_arr, rhs_scalar) = rhs.get();

        if lhs_scalar && rhs_scalar {
            return Err(ArrowKernelError::UnsupportedArguments(
                "scalar-scalar comparison".to_string(),
            ));
        }

        if lhs_scalar && !rhs_scalar {
            return ComparisonKernel::compile(&(rhs, lhs), pred.flip());
        }

        let lhs_iter = datum_to_iter(lhs)?;
        let rhs_iter = datum_to_iter(rhs)?;
        let ctx = Context::create();
        ComparisonKernelTryBuilder {
            context: ctx,
            lhs_data_type: lhs_arr.data_type().clone(),
            rhs_data_type: rhs_arr.data_type().clone(),
            rhs_scalar,
            pred,
            func_builder: |ctx| match generate_block_llvm_cmp_kernel(
                ctx, lhs, &lhs_iter, rhs, &rhs_iter, pred,
            ) {
                Ok(k) => Ok(k),
                Err(ArrowKernelError::NonVectorizableType(_)) => {
                    generate_llvm_cmp_kernel(ctx, lhs, &lhs_iter, rhs, &rhs_iter, pred)
                }
                Err(e) => Err(e),
            },
        }
        .try_build()
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Predicate,
    ) -> Result<Self::Key, ArrowKernelError> {
        let (lhs, rhs) = *i;
        if lhs.get().1 {
            return Self::get_key_for_input(&(rhs, lhs), &p.flip());
        }

        if lhs.get().1 {
            return Err(ArrowKernelError::UnsupportedArguments(
                "scalar-scalar comparison".to_string(),
            ));
        }
        Ok((
            lhs.get().0.data_type().clone(),
            rhs.get().0.data_type().clone(),
            rhs.get().1,
            *p,
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
) -> Result<
    JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void)>,
    ArrowKernelError,
> {
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

    if lhs_prim != rhs_prim {
        return Err(ArrowKernelError::ArgumentMismatch(format!(
            "nonblock cmp cannot do type conversion, saw {:?} and {:?}",
            lhs_dt, rhs_dt
        )));
    }

    let lhs_llvm = lhs_prim.llvm_type(ctx);
    let rhs_llvm = rhs_prim.llvm_type(ctx);

    let lhs_next = generate_next(ctx, &module, "next_lhs", lhs_dt, lhs_iter).unwrap();
    let rhs_next = generate_next(ctx, &module, "next_rhs", rhs_dt, rhs_iter).unwrap();

    let fn_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
    let cmp = module.add_function("cmp", fn_type, None);
    let lhs_ptr = cmp.get_nth_param(0).unwrap();
    let rhs_ptr = cmp.get_nth_param(1).unwrap();
    let out_ptr = cmp.get_nth_param(2).unwrap().into_pointer_value();

    declare_blocks!(ctx, cmp, entry, block_cond, block_body, exit);

    build.position_at_end(entry);
    let lhs_buf = build.build_alloca(lhs_llvm, "lhs_single_buf").unwrap();
    let rhs_buf = build.build_alloca(rhs_llvm, "rhs_single_buf").unwrap();
    let bool_writer = BooleanWriter::llvm_init(ctx, &module, &build, PrimitiveType::U8, out_ptr);
    build.build_unconditional_branch(block_cond).unwrap();

    build.position_at_end(block_cond);
    let had_lhs = build
        .build_call(lhs_next, &[lhs_ptr.into(), lhs_buf.into()], "lhs_next")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_call(rhs_next, &[rhs_ptr.into(), rhs_buf.into()], "rhs_next")
        .unwrap();
    build
        .build_conditional_branch(had_lhs, block_body, exit)
        .unwrap();

    build.position_at_end(block_body);
    let lv = build.build_load(lhs_llvm, lhs_buf, "lv").unwrap();
    let rv = build.build_load(rhs_llvm, rhs_buf, "rv").unwrap();

    let res = match lhs_prim.comparison_type() {
        ComparisonType::Int { .. } => unreachable!("ints should use block compare"),
        ComparisonType::Float => unreachable!("floats should use block compare"),
        ComparisonType::String => {
            let memcmp = add_memcmp(ctx, &module);
            let res = build
                .build_call(memcmp, &[lv.into(), rv.into()], "res")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            build
                .build_int_compare(
                    pred.as_int_pred(true),
                    res,
                    i64_type.const_zero(),
                    "cmp_res",
                )
                .unwrap()
        }
    };
    bool_writer.llvm_ingest(ctx, &build, res.as_basic_value_enum());
    build.build_unconditional_branch(block_cond).unwrap();

    build.position_at_end(exit);
    bool_writer.llvm_flush(ctx, &build);
    build.build_return(None).unwrap();

    module.verify().unwrap();
    optimize_module(&module)?;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    Ok(unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void)>(
            cmp.get_name().to_str().unwrap(),
        )
        .unwrap()
    })
}

fn generate_block_llvm_cmp_kernel<'a>(
    ctx: &'a Context,
    lhs: &dyn Datum,
    lhs_iter: &IteratorHolder,
    rhs: &dyn Datum,
    rhs_iter: &IteratorHolder,
    pred: Predicate,
) -> Result<
    JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void)>,
    ArrowKernelError,
> {
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
    let lhs_vec = lhs_prim
        .llvm_vec_type(ctx, 64)
        .ok_or_else(|| ArrowKernelError::NonVectorizableType(lhs_dt.clone()))?;
    let rhs_vec = rhs_prim
        .llvm_vec_type(ctx, 64)
        .ok_or_else(|| ArrowKernelError::NonVectorizableType(rhs_dt.clone()))?;
    let lhs_llvm = lhs_prim.llvm_type(ctx);
    let rhs_llvm = rhs_prim.llvm_type(ctx);
    let dom_prim_type = PrimitiveType::dominant(lhs_prim, rhs_prim).unwrap();
    let dom_llvm = dom_prim_type.llvm_type(ctx);

    let lhs_next_block =
        generate_next_block::<64>(ctx, &module, "next_lhs_block", lhs_dt, lhs_iter).unwrap();
    let rhs_next_block =
        generate_next_block::<64>(ctx, &module, "next_rhs_block", rhs_dt, rhs_iter).unwrap();
    let lhs_next = generate_next(ctx, &module, "next_lhs", lhs_dt, lhs_iter).unwrap();
    let rhs_next = generate_next(ctx, &module, "next_rhs", rhs_dt, rhs_iter).unwrap();

    let fn_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
    let cmp = module.add_function("cmp", fn_type, None);
    let lhs_ptr = cmp.get_nth_param(0).unwrap();
    let rhs_ptr = cmp.get_nth_param(1).unwrap();
    let alloc_ptr = cmp.get_nth_param(2).unwrap().into_pointer_value();

    declare_blocks!(ctx, cmp, entry, block_cond, block_body, tail_cond, tail_body, exit);

    build.position_at_end(entry);

    let writer = BooleanWriter::llvm_init(ctx, &module, &build, PrimitiveType::U8, alloc_ptr);
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

    let lvec = gen_convert_numeric_vec(ctx, &build, lvec, lhs_prim, dom_prim_type);
    let rvec = gen_convert_numeric_vec(ctx, &build, rvec, rhs_prim, dom_prim_type);

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
    writer.llvm_ingest_64_bools(ctx, &build, res);
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

    let lv_v = build
        .build_bit_cast(lv, lhs_prim.llvm_vec_type(ctx, 1).unwrap(), "lv_v")
        .unwrap()
        .into_vector_value();
    let rv_v = build
        .build_bit_cast(rv, rhs_prim.llvm_vec_type(ctx, 1).unwrap(), "rv_v")
        .unwrap()
        .into_vector_value();
    let lv_v = gen_convert_numeric_vec(ctx, &build, lv_v, lhs_prim, dom_prim_type);
    let rv_v = gen_convert_numeric_vec(ctx, &build, rv_v, rhs_prim, dom_prim_type);
    let lv = build.build_bit_cast(lv_v, dom_llvm, "lv").unwrap();
    let rv = build.build_bit_cast(rv_v, dom_llvm, "rv").unwrap();

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

    writer.llvm_ingest(ctx, &build, res.into());
    build.build_unconditional_branch(tail_cond).unwrap();

    build.position_at_end(exit);
    writer.llvm_flush(ctx, &build);
    build.build_return(None).unwrap();

    module.verify().unwrap();
    optimize_module(&module)?;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    Ok(unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void)>(
            cmp.get_name().to_str().unwrap(),
        )
        .unwrap()
    })
}

pub fn add_float_vec_to_int_vec<'a>(
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

    func
}

pub fn add_float_to_int<'a>(
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
    let func = module.add_function("float_to_int", func_type, Some(Linkage::Private));
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

    func
}

/// Adds or returns the `memcmp` function to the module.
///
/// `memcmp` takes two `PrimitiveType::P64x2` values.
pub fn add_memcmp<'a>(ctx: &'a Context, module: &Module<'a>) -> FunctionValue<'a> {
    if let Some(func) = module.get_function("memcmp") {
        return func;
    }

    let i64_type = ctx.i64_type();
    let i8_type = ctx.i8_type();
    let str_type = PrimitiveType::P64x2.llvm_type(ctx);

    let fn_type = i64_type.fn_type(
        &[
            str_type.into(), // first string
            str_type.into(), // second string
        ],
        false,
    );

    let umin = Intrinsic::find("llvm.umin").unwrap();
    let umin_f = umin.get_declaration(module, &[i64_type.into()]).unwrap();

    let builder = ctx.create_builder();
    let function = module.add_function("memcmp", fn_type, Some(Linkage::Private));

    declare_blocks!(
        ctx,
        function,
        entry,
        for_cond,
        for_body,
        early_return,
        no_diff
    );

    builder.position_at_end(entry);
    let ptr1 = function.get_nth_param(0).unwrap().into_struct_value();
    let ptr2 = function.get_nth_param(1).unwrap().into_struct_value();

    let start_ptr1 = builder
        .build_extract_value(ptr1, 0, "start_ptr1")
        .unwrap()
        .into_pointer_value();
    let end_ptr1 = builder
        .build_extract_value(ptr1, 1, "end_ptr1")
        .unwrap()
        .into_pointer_value();
    let start_ptr2 = builder
        .build_extract_value(ptr2, 0, "start_ptr2")
        .unwrap()
        .into_pointer_value();
    let end_ptr2 = builder
        .build_extract_value(ptr2, 1, "end_ptr2")
        .unwrap()
        .into_pointer_value();

    let len1 = pointer_diff!(ctx, builder, start_ptr1, end_ptr1);
    let len2 = pointer_diff!(ctx, builder, start_ptr2, end_ptr2);

    let len = builder
        .build_call(umin_f, &[len1.into(), len2.into()], "len")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();

    let index_ptr = builder.build_alloca(i64_type, "index_ptr").unwrap();
    builder
        .build_store(index_ptr, i64_type.const_zero())
        .unwrap();

    builder.build_unconditional_branch(for_cond).unwrap();

    builder.position_at_end(for_cond);
    let index = builder
        .build_load(i64_type, index_ptr, "index")
        .unwrap()
        .into_int_value();
    let cmp = builder
        .build_int_compare(IntPredicate::ULT, index, len, "loop_cmp")
        .unwrap();
    builder
        .build_conditional_branch(cmp, for_body, no_diff)
        .unwrap();

    builder.position_at_end(for_body);
    let index = builder
        .build_load(i64_type, index_ptr, "index")
        .unwrap()
        .into_int_value();
    let elem1_ptr = increment_pointer!(ctx, builder, start_ptr1, 1, index);
    let elem2_ptr = increment_pointer!(ctx, builder, start_ptr2, 1, index);
    let elem1 = builder
        .build_load(i8_type, elem1_ptr, "elem1")
        .unwrap()
        .into_int_value();
    let elem2 = builder
        .build_load(i8_type, elem2_ptr, "elem2")
        .unwrap()
        .into_int_value();

    let elem1 = builder
        .build_int_z_extend_or_bit_cast(elem1, i64_type, "z_elem1")
        .unwrap();
    let elem2 = builder
        .build_int_z_extend_or_bit_cast(elem2, i64_type, "z_elem2")
        .unwrap();

    let diff = builder.build_int_sub(elem1, elem2, "sub").unwrap();

    let value_cmp = builder
        .build_int_compare(IntPredicate::EQ, diff, i64_type.const_zero(), "value_cmp")
        .unwrap();

    let inc_index = builder
        .build_int_add(index, i64_type.const_int(1, false), "inc_index")
        .unwrap();
    builder.build_store(index_ptr, inc_index).unwrap();

    builder
        .build_conditional_branch(value_cmp, for_cond, early_return)
        .unwrap();

    builder.position_at_end(early_return);
    builder.build_return(Some(&diff)).unwrap();

    builder.position_at_end(no_diff);
    let eq_len = builder
        .build_int_compare(IntPredicate::EQ, len1, len2, "is_eq")
        .unwrap();
    let arr1_longer = builder
        .build_int_compare(IntPredicate::UGT, len1, len2, "arr1_longer")
        .unwrap();

    let res = builder
        .build_select(
            eq_len,
            i64_type.const_zero(),
            builder
                .build_select(
                    arr1_longer,
                    i64_type.const_int(1, true),
                    i64_type.const_int(-1_i64 as u64, false),
                    "diff",
                )
                .unwrap()
                .into_int_value(),
            "same",
        )
        .unwrap()
        .into_int_value();
    builder.build_return(Some(&res)).unwrap();

    function
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use arrow_array::{
        BooleanArray, Float32Array, Int32Array, Int64Array, Scalar, StringArray, StringViewArray,
        UInt32Array,
    };
    use arrow_schema::DataType;
    use itertools::Itertools;

    use crate::{
        dictionary_data_type,
        new_kernels::{cmp::ComparisonKernel, Kernel},
        Predicate,
    };

    #[test]
    fn test_num_num_cmp() {
        let a = UInt32Array::from(vec![1, 2, 3]);
        let b = UInt32Array::from(vec![11, 0, 13]);
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        let r = k.call((&a, &b)).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true, false, true]))
    }

    #[test]
    fn test_num_num_block_cmp() {
        let a = UInt32Array::from((0..100).collect_vec());
        let b = UInt32Array::from((1..101).collect_vec());
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        let r = k.call((&a, &b)).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true; 100]))
    }

    #[test]
    fn test_num_num_block_scalar_cmp() {
        let a = UInt32Array::from((0..100).collect_vec());
        let b = Scalar::new(UInt32Array::from(vec![5]));
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..100).map(|i| i < 5).collect::<Vec<bool>>();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_num_num_float_cmp() {
        let a = Float32Array::from((0..100).map(|i| i as f32).collect_vec());
        let b = Float32Array::from((1..101).map(|i| i as f32).collect_vec());
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        let r = k.call((&a, &b)).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true; 100]))
    }

    #[test]
    fn test_num_num_float_scalar_cmp() {
        let a = Float32Array::from((0..100).map(|i| i as f32).collect_vec());
        let b = Scalar::new(Float32Array::from(vec![5.0]));
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..100).map(|i| i < 5).collect::<Vec<bool>>();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_num_num_float_scalar_cmp_convert() {
        let a = Float32Array::from(vec![-93294.49]);
        let b = Float32Array::new_scalar(-205150180000.0);
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let res = (-93294.49_f32).total_cmp(&-205150180000.0) == Ordering::Less;
        assert_eq!(r, BooleanArray::from(vec![res]));
    }

    #[test]
    fn test_int32_int64_cmp() {
        let a = Int32Array::from(vec![1, 2, 3, 4]);
        let b = Int64Array::from(vec![4, 3, 2, 1]);
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = vec![true, true, false, false];
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_string_string_cmp() {
        let a = StringArray::from(vec!["apple", "banana", "cherry"]);
        let b = StringArray::from(vec!["banana", "cherry", "apple"]);
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = vec![true, true, false];
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_string_scalar_cmp() {
        let values = (0..100).map(|i| format!("value{}", i)).collect_vec();
        let a = StringArray::from(values);
        let b = Scalar::new(StringArray::from(vec!["value50"]));
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Eq).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..100).map(|i| i == 50).collect_vec();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_string_view_scalar_cmp() {
        let values = (0..100).map(|i| format!("value{}", i)).collect_vec();
        let a = StringViewArray::from(values);
        let b = Scalar::new(StringArray::from(vec!["value50"]));
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Eq).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..100).map(|i| i == 50).collect_vec();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_string_64_scalar_cmp() {
        let values = (0..64).map(|i| format!("value{}", i)).collect_vec();
        let a = StringArray::from(values);
        let b = Scalar::new(StringArray::from(vec!["value50"]));
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Eq).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..64).map(|i| i == 50).collect_vec();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_dict_string_scalar_cmp() {
        let values = (0..100).map(|i| format!("value{}", i % 4)).collect_vec();
        let a = StringArray::from(values);
        let a =
            arrow_cast::cast(&a, &dictionary_data_type(DataType::Int8, DataType::Utf8)).unwrap();

        let b = Scalar::new(StringArray::from(vec!["value2"]));

        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Eq).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..100).map(|i| i % 4 == 2).collect_vec();
        assert_eq!(r, BooleanArray::from(expected_result));
    }
}
