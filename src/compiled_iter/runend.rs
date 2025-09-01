use std::ffi::c_void;

use arrow_array::{types::RunEndIndexType, Array, ArrowPrimitiveType, PrimitiveArray, RunArray};
use arrow_buffer::ArrowNativeType;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    values::{BasicValue, FunctionValue, IntValue, PointerValue},
    AddressSpace, IntPredicate,
};
use repr_offset::ReprOffset;

use crate::{declare_blocks, increment_pointer, PrimitiveType};

use super::{array_to_iter, primitive::PrimitiveIterator, IteratorHolder};

/// An iterator for run-end encoded data. Contains pointers to the *iterators* for
/// the underlying run ends and values.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct RunEndIterator {
    /// the run ends, which must be i16, i32, or i64
    run_ends: *const c_void,

    /// the value iterator, not the raw value array
    val_iter: *const c_void,

    /// the position within the run_ends iterator (not the logical position)
    pos: u64,

    /// the total number of run ends (and values)
    len: u64,

    /// logical position (output index)
    logical_pos: u64,

    /// logical length (total number produced)
    logical_len: u64,

    /// the remaining number of values in the current run
    remaining: u64,
}

impl RunEndIterator {
    pub fn llvm_re_iter_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let ptr_ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_RUN_ENDS);
        let ptr = builder
            .build_load(ctx.ptr_type(AddressSpace::default()), ptr_ptr, "run_ends")
            .unwrap()
            .into_pointer_value();
        ptr.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        ptr
    }

    pub fn llvm_val_iter_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let ptr_ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_VAL_ITER);
        let ptr = builder
            .build_load(ctx.ptr_type(AddressSpace::default()), ptr_ptr, "val_iter")
            .unwrap()
            .into_pointer_value();
        ptr.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        ptr
    }

    pub fn llvm_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_POS);
        builder
            .build_load(ctx.i64_type(), ptr, "pos")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_logical_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_LOGICAL_POS);
        builder
            .build_load(ctx.i64_type(), ptr, "log_pos")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_len<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_LEN);
        let len = builder
            .build_load(ctx.i64_type(), ptr, "len")
            .unwrap()
            .into_int_value();
        len.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        len
    }

    pub fn llvm_logical_len<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_LOGICAL_LEN);
        let len = builder
            .build_load(ctx.i64_type(), ptr, "log_len")
            .unwrap()
            .into_int_value();
        len.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        len
    }

    pub fn llvm_remaining<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_REMAINING);
        builder
            .build_load(ctx.i64_type(), ptr, "remaining")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_inc_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_POS);
        let curr_pos = builder
            .build_load(ctx.i64_type(), ptr, "curr_pos")
            .unwrap()
            .into_int_value();
        let new_pos = builder.build_int_add(curr_pos, amt, "new_pos").unwrap();
        builder.build_store(ptr, new_pos).unwrap();
    }

    pub fn llvm_inc_logical_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_LOGICAL_POS);
        let curr_pos = builder
            .build_load(ctx.i64_type(), ptr, "curr_log_pos")
            .unwrap()
            .into_int_value();
        let new_pos = builder.build_int_add(curr_pos, amt, "new_log_pos").unwrap();
        builder.build_store(ptr, new_pos).unwrap();
    }

    pub fn llvm_set_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
        new_pos: IntValue<'a>,
    ) {
        let ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_POS);
        builder.build_store(ptr, new_pos).unwrap();
    }

    pub fn llvm_dec_remaining<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let remaining_ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_REMAINING);
        let remaining_val = builder
            .build_load(ctx.i64_type(), remaining_ptr, "remaining")
            .unwrap()
            .into_int_value();
        let new_remaining = builder
            .build_int_sub(remaining_val, amt, "new_remaining")
            .unwrap();
        builder.build_store(remaining_ptr, new_remaining).unwrap();
    }

    pub fn llvm_set_remaining<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let remaining_ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_REMAINING);
        builder.build_store(remaining_ptr, amt).unwrap();
    }
}

impl<R: RunEndIndexType + ArrowPrimitiveType> From<&RunArray<R>> for IteratorHolder {
    fn from(arr: &RunArray<R>) -> Self {
        let re = arr.run_ends().inner().clone(); // note: .inner() removes slicing offset
        let re: PrimitiveArray<R> = PrimitiveArray::new(re, None);
        let run_ends = Box::new(array_to_iter(&re));
        let values = Box::new(array_to_iter(arr.values()));

        let first_idx = re
            .values()
            .partition_point(|x| x.as_usize() <= arr.offset());

        let prev_end = if first_idx == 0 {
            0
        } else {
            re.value(first_idx - 1).as_usize()
        };
        let first_partition_size = if re.is_empty() {
            0
        } else {
            re.value(first_idx).as_usize() - prev_end
        };
        let first_remaining = first_partition_size - (arr.offset() - prev_end);

        let iter = RunEndIterator {
            run_ends: run_ends.get_ptr(),
            val_iter: values.get_ptr(),
            pos: first_idx as u64,
            len: re.len() as u64,
            logical_pos: 0,
            logical_len: arr.len() as u64,
            remaining: first_remaining as u64,
        };

        IteratorHolder::RunEnd {
            arr: Box::new(iter),
            run_ends,
            values,
        }
    }
}

/// Adds a function that uses binary search to find the position of an index in
/// a run array
pub fn add_bsearch<'a>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    re_iter: &PrimitiveIterator,
    ty: PrimitiveType,
) -> FunctionValue<'a> {
    let fname = format!("bsearch_w{}", ty.width());
    if let Some(func) = llvm_mod.get_function(&fname) {
        return func;
    }

    let i64_t = ctx.i64_type();
    let ptr_t = ctx.ptr_type(AddressSpace::default());
    let run_t = ty.llvm_type(ctx).into_int_type();

    let func_t = i64_t.fn_type(&[ptr_t.into(), run_t.into()], false);
    let func = llvm_mod.add_function(&fname, func_t, None); // Some(Linkage::Private)

    declare_blocks!(ctx, func, entry, loop_cond, loop_body, exit);
    let b = ctx.create_builder();
    b.position_at_end(entry);
    let iter_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let target_val = func.get_nth_param(1).unwrap().into_int_value();
    let lo_ptr_ptr = b.build_alloca(ptr_t, "low_ptr_ptr").unwrap();
    let hi_ptr_ptr = b.build_alloca(ptr_t, "hi_ptr_ptr").unwrap();

    b.build_store(lo_ptr_ptr, re_iter.llvm_data(ctx, &b, iter_ptr))
        .unwrap();
    b.build_store(
        hi_ptr_ptr,
        increment_pointer!(
            ctx,
            b,
            re_iter.llvm_data(ctx, &b, iter_ptr),
            ty.width(),
            re_iter.llvm_len(ctx, &b, iter_ptr)
        ),
    )
    .unwrap();
    b.build_unconditional_branch(loop_cond).unwrap();

    b.position_at_end(loop_cond);
    let lo_ptr = b
        .build_load(ptr_t, lo_ptr_ptr, "lo_ptr")
        .unwrap()
        .into_pointer_value();
    let hi_ptr = b
        .build_load(ptr_t, hi_ptr_ptr, "hi_ptr")
        .unwrap()
        .into_pointer_value();
    let cond = b
        .build_int_compare(IntPredicate::ULT, lo_ptr, hi_ptr, "cond")
        .unwrap();
    b.build_conditional_branch(cond, loop_body, exit).unwrap();

    b.position_at_end(loop_body);
    let lo_ptr = b
        .build_load(ptr_t, lo_ptr_ptr, "lo_ptr")
        .unwrap()
        .into_pointer_value();
    let hi_ptr = b
        .build_load(ptr_t, hi_ptr_ptr, "hi_ptr")
        .unwrap()
        .into_pointer_value();
    let diff = b.build_ptr_diff(run_t, hi_ptr, lo_ptr, "diff").unwrap();
    let dist_to_mid = b
        .build_int_signed_div(diff, i64_t.const_int(2, false), "dist_to_mid")
        .unwrap();
    let mid_ptr = increment_pointer!(ctx, b, lo_ptr, ty.width(), dist_to_mid);
    let mid_val = b
        .build_load(run_t, mid_ptr, "mid_val")
        .unwrap()
        .into_int_value();

    let gt_tar = b
        .build_int_compare(IntPredicate::SGT, mid_val, target_val, "gt_tar")
        .unwrap();
    let new_lo_ptr = b
        .build_select(
            gt_tar,
            lo_ptr,
            increment_pointer!(ctx, &b, mid_ptr, ty.width()),
            "new_lo_ptr",
        )
        .unwrap();
    let new_hi_ptr = b
        .build_select(gt_tar, mid_ptr, hi_ptr, "new_hi_ptr")
        .unwrap();
    b.build_store(lo_ptr_ptr, new_lo_ptr).unwrap();
    b.build_store(hi_ptr_ptr, new_hi_ptr).unwrap();
    b.build_unconditional_branch(loop_cond).unwrap();

    b.position_at_end(exit);
    let hi_ptr = b
        .build_load(ptr_t, hi_ptr_ptr, "lo_ptr")
        .unwrap()
        .into_pointer_value();
    let idx = b
        .build_ptr_diff(run_t, hi_ptr, re_iter.llvm_data(ctx, &b, iter_ptr), "idx")
        .unwrap();
    let val = b.build_load(run_t, hi_ptr, "val").unwrap().into_int_value();
    let eq_val = b
        .build_int_compare(IntPredicate::EQ, val, target_val, "eq_target")
        .unwrap();
    let idx = b
        .build_select(
            eq_val,
            b.build_int_add(idx, i64_t.const_int(1, true), "inc_idx")
                .unwrap(),
            idx,
            "idx",
        )
        .unwrap();
    b.build_return(Some(&idx)).unwrap();

    func
}

#[cfg(test)]
mod tests {
    use std::{ffi::c_void, sync::Arc};

    use arrow_array::{
        cast::AsArray, types::Int32Type, Array, Int32Array, Int32RunArray, RunArray,
    };
    use arrow_data::ArrayDataBuilder;
    use arrow_schema::{DataType, Field};
    use inkwell::{context::Context, OptimizationLevel};

    use crate::{
        compiled_iter::{
            array_to_iter, datum_to_iter, generate_next, generate_next_block,
            generate_random_access, IteratorHolder,
        },
        PrimitiveType,
    };

    use super::add_bsearch;

    #[test]
    fn test_ree_iter_block_noslice() {
        let data = Int32Array::from(vec![1, 2, 3, 4]);
        let ends = Int32Array::from(vec![4, 8, 17, 18]);
        let ree = RunArray::try_new(&ends, &data).unwrap();

        let mut iter = datum_to_iter(&ree).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func =
            generate_next_block::<8>(&ctx, &module, "iter_block_next", ree.data_type(), &iter)
                .unwrap();
        let fname = func.get_name().to_str().unwrap();

        let next_func = generate_next(&ctx, &module, "iter_next", ree.data_type(), &iter).unwrap();
        let next_fname = next_func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_block = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let next = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(next_fname)
                .unwrap()
        };

        let mut buf = [0_i32; 8];
        let mut sbuf: i32 = 0;
        unsafe {
            assert!(next_block.call(iter.get_mut_ptr(), buf.as_mut_ptr()));
            assert_eq!(buf, [1, 1, 1, 1, 2, 2, 2, 2]);
            assert!(next_block.call(iter.get_mut_ptr(), buf.as_mut_ptr()));
            assert_eq!(buf, [3, 3, 3, 3, 3, 3, 3, 3]);
            assert!(!next_block.call(iter.get_mut_ptr(), buf.as_mut_ptr()));

            assert!(next.call(iter.get_mut_ptr(), &mut sbuf));
            assert_eq!(sbuf, 3);
            assert!(next.call(iter.get_mut_ptr(), &mut sbuf));
            assert_eq!(sbuf, 4);
            assert!(!next.call(iter.get_mut_ptr(), &mut sbuf));
        };
    }

    #[test]
    fn test_ree_iter_block_slice() {
        let data = Int32Array::from(vec![1, 2, 3, 4]);
        let ends = Int32Array::from(vec![4, 8, 17, 18]);
        let ree_full = RunArray::try_new(&ends, &data).unwrap();
        let ree = ree_full.slice(4, 10);

        let mut iter = datum_to_iter(&ree).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func =
            generate_next_block::<8>(&ctx, &module, "iter_block_next", ree.data_type(), &iter)
                .unwrap();
        let fname = func.get_name().to_str().unwrap();

        let next_func = generate_next(&ctx, &module, "iter_next", ree.data_type(), &iter).unwrap();
        let next_fname = next_func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_block = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let next = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(next_fname)
                .unwrap()
        };

        let mut buf = [0_i32; 8];
        let mut sbuf: i32 = 0;
        unsafe {
            assert!(next_block.call(iter.get_mut_ptr(), buf.as_mut_ptr()));
            assert_eq!(buf, [2, 2, 2, 2, 3, 3, 3, 3]);
            assert!(
                !next_block.call(iter.get_mut_ptr(), buf.as_mut_ptr()),
                "expected false, but got slice with {:?}",
                buf
            );

            assert_eq!(next.call(iter.get_mut_ptr(), &mut sbuf), true);
            assert_eq!(sbuf, 3);
            assert_eq!(next.call(iter.get_mut_ptr(), &mut sbuf), true);
            assert_eq!(sbuf, 3);
            assert_eq!(next.call(iter.get_mut_ptr(), &mut sbuf), false);
        };
    }

    #[test]
    fn test_re_iter() {
        let values = Int32Array::from(vec![10, 20]);
        let res = Int32Array::from(vec![2, 4]);
        let ree = Int32RunArray::try_new(&res, &values).unwrap();
        let mut iter = datum_to_iter(&ree).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_runend");

        let func = generate_next(&ctx, &module, "runend", ree.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut res = 0;
        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 10);

        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 10);

        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 20);

        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 20);

        assert!(!unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
    }

    #[test]
    fn test_re_iter_slice() {
        let values = Int32Array::from(vec![10, 20, 30]);
        let res = Int32Array::from(vec![2, 4, 6]);
        let ree_full = Int32RunArray::try_new(&res, &values).unwrap();
        let ree = ree_full.slice(2, 4);
        let mut iter = datum_to_iter(&ree).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_runend");

        let func = generate_next(&ctx, &module, "runend", ree.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut res = 0;
        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 20);

        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 20);

        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 30);

        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 30);

        assert!(!unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
    }

    #[test]
    fn test_re_zero_runs_iter() {
        let ree_array_type = DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int32, false)),
            Arc::new(Field::new("values", DataType::Int32, true)),
        );
        let values = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let res = Int32Array::from(vec![0, 1, 1, 4, 4]);
        let builder = ArrayDataBuilder::new(ree_array_type)
            .len(4)
            .add_child_data(res.to_data())
            .add_child_data(values.to_data());
        let array_data = unsafe { builder.build_unchecked() };

        let ree = Int32RunArray::from(array_data);
        let mut iter = datum_to_iter(&ree).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_runend");

        let func = generate_next(&ctx, &module, "runend", ree.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut res = 0;
        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 20);

        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 40);

        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 40);

        assert!(unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
        assert_eq!(res, 40);

        assert!(!unsafe { next_func.call(iter.get_mut_ptr(), &mut res as *mut i32) });
    }

    #[test]
    fn test_re_random_access() {
        let values = Int32Array::from(vec![10, 20, 30, 40]);
        let res = Int32Array::from(vec![2, 4, 5, 10]);
        let ree = Int32RunArray::try_new(&res, &values).unwrap();
        let mut iter = datum_to_iter(&ree).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_runend");

        let func = generate_random_access(&ctx, &module, "access", ree.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let access_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i32>(fname)
                .unwrap()
        };

        for idx in 0..ree.len() {
            let p_idx = ree.get_physical_index(idx);
            let val = ree.values().as_primitive::<Int32Type>().value(p_idx);
            assert_eq!(
                val,
                unsafe { access_func.call(iter.get_mut_ptr(), idx as u64) },
                "mismatch at idx {} (p_idx = {})",
                idx,
                p_idx
            );
        }
    }

    #[test]
    fn test_bsearch_i32() {
        let runs = Int32Array::from(vec![1, 10, 20, 23, 24, 24, 30]);
        let ih = array_to_iter(&runs);
        if let IteratorHolder::Primitive(iter) = &ih {
            let ctx = Context::create();
            let module = ctx.create_module("test_runend");
            let func = add_bsearch(&ctx, &module, &iter, PrimitiveType::I32);
            let fname = func.get_name().to_str().unwrap();

            module.verify().unwrap();
            let ee = module
                .create_jit_execution_engine(OptimizationLevel::None)
                .unwrap();
            let bsearch = unsafe {
                ee.get_function::<unsafe extern "C" fn(*const c_void, i32) -> i64>(fname)
                    .unwrap()
            };

            unsafe {
                assert_eq!(bsearch.call(ih.get_ptr(), 0), 0);
                assert_eq!(bsearch.call(ih.get_ptr(), 1), 1);
                assert_eq!(bsearch.call(ih.get_ptr(), 5), 1);
                assert_eq!(bsearch.call(ih.get_ptr(), 10), 2);
                assert_eq!(bsearch.call(ih.get_ptr(), 11), 2);
                assert_eq!(bsearch.call(ih.get_ptr(), 20), 3);
                assert_eq!(bsearch.call(ih.get_ptr(), 24), 6);
                assert_eq!(bsearch.call(ih.get_ptr(), 25), 6);
                assert_eq!(bsearch.call(ih.get_ptr(), 29), 6);
            }
        } else {
            panic!("non-primitive iterator for primitive array");
        }
    }
}
