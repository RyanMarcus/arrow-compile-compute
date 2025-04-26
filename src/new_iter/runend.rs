use std::ffi::c_void;

use arrow_array::{
    types::RunEndIndexType, ArrowNativeTypeOp, ArrowPrimitiveType, PrimitiveArray, RunArray,
};
use arrow_buffer::ArrowNativeType;
use inkwell::{
    builder::Builder,
    context::Context,
    values::{IntValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::increment_pointer;

use super::{array_to_iter, IteratorHolder};

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
        builder
            .build_load(ctx.ptr_type(AddressSpace::default()), ptr_ptr, "run_ends")
            .unwrap()
            .into_pointer_value()
    }

    pub fn llvm_val_iter_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let ptr_ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_VAL_ITER);
        builder
            .build_load(ctx.ptr_type(AddressSpace::default()), ptr_ptr, "val_iter")
            .unwrap()
            .into_pointer_value()
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

    pub fn llvm_len<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let ptr = increment_pointer!(ctx, builder, ptr, RunEndIterator::OFFSET_LEN);
        builder
            .build_load(ctx.i64_type(), ptr, "len")
            .unwrap()
            .into_int_value()
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
        let re = arr.run_ends().inner().clone();
        let re: PrimitiveArray<R> = PrimitiveArray::new(re, None);
        let run_ends = Box::new(array_to_iter(&re));
        let values = Box::new(array_to_iter(arr.values()));

        let first_pos = re
            .values()
            .iter()
            .enumerate()
            .find(|(_idx, val)| !val.is_zero())
            .map(|(idx, _val)| idx)
            .unwrap_or(0);

        let first_remaining = if first_pos < re.len() {
            re.value(first_pos).as_usize()
        } else {
            0
        };

        let iter = RunEndIterator {
            run_ends: run_ends.get_ptr(),
            val_iter: values.get_ptr(),
            pos: first_pos as u64,
            len: re.len() as u64,
            remaining: first_remaining as u64,
        };
        IteratorHolder::RunEnd {
            arr: Box::new(iter),
            run_ends,
            values,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{ffi::c_void, sync::Arc};

    use arrow_array::{Array, Int32Array, Int32RunArray, RunArray};
    use arrow_data::ArrayDataBuilder;
    use arrow_schema::{DataType, Field};
    use inkwell::{context::Context, OptimizationLevel};

    use crate::new_iter::{datum_to_iter, generate_next, generate_next_block};

    #[test]
    fn test_ree_iter_block() {
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
            assert_eq!(next_block.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [1, 1, 1, 1, 2, 2, 2, 2]);
            assert_eq!(next_block.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [3, 3, 3, 3, 3, 3, 3, 3]);
            assert_eq!(next_block.call(iter.get_mut_ptr(), buf.as_mut_ptr()), false);

            assert_eq!(next.call(iter.get_mut_ptr(), &mut sbuf), true);
            assert_eq!(sbuf, 3);
            assert_eq!(next.call(iter.get_mut_ptr(), &mut sbuf), true);
            assert_eq!(sbuf, 4);
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
}
