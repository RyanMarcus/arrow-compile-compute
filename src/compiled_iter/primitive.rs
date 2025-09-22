use std::{ffi::c_void, sync::Arc};

use crate::increment_pointer;
use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray};
use inkwell::{
    builder::Builder,
    context::Context,
    values::{BasicValue, IntValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

/// An iterator for primitive (densely packed) data.
///
/// * `data` is a pointer to the densely packed data buffer
///
/// * `pos` is the current position in the iterator, all reads are relative to
///   this position
///
/// * `len` is the length of the data from the start of the `data` pointer
///   (i.e., not accounting for `pos`)
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct PrimitiveIterator {
    data: *const c_void,
    pos: u64,
    len: u64,
    array_ref: Arc<dyn Array>,
}

impl<K: ArrowPrimitiveType> From<&PrimitiveArray<K>> for Box<PrimitiveIterator> {
    fn from(value: &PrimitiveArray<K>) -> Self {
        Box::new(PrimitiveIterator {
            data: value.values().as_ptr() as *const c_void,
            pos: value.offset() as u64, // always zero
            len: (value.len() + value.offset()) as u64,
            array_ref: Arc::new(value.clone()),
        })
    }
}

impl PrimitiveIterator {
    pub fn llvm_len<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let len_ptr = increment_pointer!(ctx, builder, ptr, PrimitiveIterator::OFFSET_LEN);
        let len = builder
            .build_load(ctx.i64_type(), len_ptr, "len")
            .unwrap()
            .into_int_value();
        len.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        len
    }

    pub fn llvm_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let offset_ptr = increment_pointer!(ctx, builder, ptr, PrimitiveIterator::OFFSET_POS);
        builder
            .build_load(ctx.i64_type(), offset_ptr, "pos")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_data<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let data_ptr_ptr = increment_pointer!(ctx, builder, ptr, PrimitiveIterator::OFFSET_DATA);
        let ptr = builder
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                data_ptr_ptr,
                "data_ptr",
            )
            .unwrap()
            .into_pointer_value();
        ptr.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        ptr
    }

    pub fn llvm_increment_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let pos_ptr = increment_pointer!(ctx, builder, ptr, PrimitiveIterator::OFFSET_POS);
        let pos = builder
            .build_load(ctx.i64_type(), pos_ptr, "pos")
            .unwrap()
            .into_int_value();
        let new_pos = builder.build_int_add(pos, amt, "new_pos").unwrap();
        builder.build_store(pos_ptr, new_pos).unwrap();
    }

    pub fn localize_struct<'a>(
        &self,
        ctx: &'a Context,
        b: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let stype = ctx.struct_type(
            &[
                ctx.ptr_type(AddressSpace::default()).into(),
                ctx.i64_type().into(),
                ctx.i64_type().into(),
            ],
            false,
        );
        let new_ptr = b.build_alloca(stype, "local_struct").unwrap();
        b.build_store(new_ptr, self.llvm_data(ctx, b, ptr)).unwrap();
        b.build_store(
            increment_pointer!(ctx, b, new_ptr, 8),
            self.llvm_pos(ctx, b, ptr),
        )
        .unwrap();
        b.build_store(
            increment_pointer!(ctx, b, new_ptr, 16),
            self.llvm_len(ctx, b, ptr),
        )
        .unwrap();
        new_ptr
    }
}

#[cfg(test)]
mod test {
    use std::ffi::c_void;

    use arrow_array::{Array, Int32Array};
    use inkwell::{context::Context, OptimizationLevel};
    use itertools::Itertools;

    use crate::compiled_iter::{
        array_to_iter, generate_next, generate_next_block, generate_random_access,
    };

    #[test]
    fn test_primitive_iter_block() {
        let data = Int32Array::from((0..16).collect_vec());
        let mut iter = array_to_iter(&data);

        let iter_ptr = iter.get_mut_ptr();
        unsafe {
            let pos: u64 = (iter_ptr.add(8) as *mut u64).read();
            let len: u64 = (iter_ptr.add(16) as *mut u64).read();

            assert_eq!(pos, 0);
            assert_eq!(len, 16);
        }

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func =
            generate_next_block::<8>(&ctx, &module, "iter_prim_test", data.data_type(), &iter)
                .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut buf = [0_i32; 8];
        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [0, 1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [8, 9, 10, 11, 12, 13, 14, 15]);
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), false);
        };
    }

    #[test]
    fn test_primitive_iter_block_slice() {
        let data_full = Int32Array::from((0..24).collect_vec());
        let data = data_full.slice(4, 16);
        let mut iter = array_to_iter(&data);

        let iter_ptr = iter.get_mut_ptr();
        unsafe {
            let pos: u64 = (iter_ptr.add(8) as *mut u64).read();
            let len: u64 = (iter_ptr.add(16) as *mut u64).read();

            assert_eq!(pos, 0);
            assert_eq!(len, 16);
        }

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func =
            generate_next_block::<8>(&ctx, &module, "iter_prim_test", data.data_type(), &iter)
                .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut buf = [0_i32; 8];
        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [4, 5, 6, 7, 8, 9, 10, 11]);
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [12, 13, 14, 15, 16, 17, 18, 19]);
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), false);
        };
    }

    #[test]
    fn test_primitive_iter_nonblock() {
        let data = Int32Array::from((0..5).collect_vec());
        let mut iter = array_to_iter(&data);

        let iter_ptr = iter.get_mut_ptr();
        unsafe {
            let pos: u64 = (iter_ptr.add(8) as *mut u64).read();
            let len: u64 = (iter_ptr.add(16) as *mut u64).read();

            assert_eq!(pos, 0);
            assert_eq!(len, 5);
        }

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_next(&ctx, &module, "iter_prim_test", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut buf: i32 = 0;
        unsafe {
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 0);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 1);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 2);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 3);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 4);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                false
            );
        };
    }

    #[test]
    fn test_primitive_iter_nonblock_slice() {
        let data_full = Int32Array::from((0..5).collect_vec());
        let data = data_full.slice(2, 3);
        let mut iter = array_to_iter(&data);

        let iter_ptr = iter.get_mut_ptr();
        unsafe {
            let pos: u64 = (iter_ptr.add(8) as *mut u64).read();
            let len: u64 = (iter_ptr.add(16) as *mut u64).read();

            assert_eq!(pos, 0);
            assert_eq!(len, 3);
        }

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_next(&ctx, &module, "iter_prim_test", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut buf: i32 = 0;
        unsafe {
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 2);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 3);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 4);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                false
            );
        };
    }

    #[test]
    fn test_primitive_random_access() {
        let data = Int32Array::from((0..5).collect_vec());
        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_random_access(&ctx, &module, "iter_prim_test", data.data_type(), &iter)
            .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i32>(fname)
                .unwrap()
        };

        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), 0), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 1), 1);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 2), 2);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 3), 3);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 4), 4);
        };
    }

    #[test]
    fn test_primitive_random_access_slice() {
        let data_full = Int32Array::from((0..5).collect_vec());
        let data = data_full.slice(2, 3);
        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_random_access(&ctx, &module, "iter_prim_test", data.data_type(), &iter)
            .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i32>(fname)
                .unwrap()
        };

        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), 0), 2);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 1), 3);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 2), 4);
        };
    }
}
