use arrow_array::{Array, BooleanArray};
use inkwell::{
    builder::Builder,
    context::Context,
    values::{IntValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::{increment_pointer, mark_load_invariant};

/// An iterator for bitmap data. Contains a pointer to the bitmap buffer and the
/// data buffer, along with a `pos` and `len` just like primitive iterators.
/// Note that each element pointed to by `data` contains 8 items/bits.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct BitmapIterator {
    data: *const u8,
    slice_offset: u64,
    pos: u64,
    len: u64,
    pub(super) array_ref: BooleanArray,
}

impl From<&BooleanArray> for Box<BitmapIterator> {
    fn from(value: &BooleanArray) -> Self {
        Box::new(BitmapIterator {
            data: value.values().values().as_ptr(),
            slice_offset: value.offset() as u64,
            pos: 0,
            len: value.len() as u64,
            array_ref: value.clone(),
        })
    }
}

impl BitmapIterator {
    pub fn llvm_get_data_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let data_ptr_ptr = increment_pointer!(ctx, build, ptr, BitmapIterator::OFFSET_DATA);
        let data_ptr = build
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                data_ptr_ptr,
                "data_ptr",
            )
            .unwrap();
        mark_load_invariant!(ctx, data_ptr);
        data_ptr.into_pointer_value()
    }

    pub fn llvm_slice_offset<'a>(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let slice_offset_ptr =
            increment_pointer!(ctx, build, ptr, BitmapIterator::OFFSET_SLICE_OFFSET);
        let val = build
            .build_load(ctx.i64_type(), slice_offset_ptr, "slice_offset")
            .unwrap()
            .into_int_value();
        mark_load_invariant!(ctx, val);
        val
    }

    pub fn llvm_pos<'a>(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let pos_ptr = increment_pointer!(ctx, build, ptr, BitmapIterator::OFFSET_POS);
        build
            .build_load(ctx.i64_type(), pos_ptr, "pos")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_len<'a>(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let len_ptr = increment_pointer!(ctx, build, ptr, BitmapIterator::OFFSET_LEN);
        let len = build.build_load(ctx.i64_type(), len_ptr, "len").unwrap();
        mark_load_invariant!(ctx, len);
        len.into_int_value()
    }

    pub fn llvm_increment_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &Builder<'a>,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let pos_ptr = increment_pointer!(ctx, builder, ptr, BitmapIterator::OFFSET_POS);
        let pos = builder
            .build_load(ctx.i64_type(), pos_ptr, "pos")
            .unwrap()
            .into_int_value();
        let new_pos = builder.build_int_add(pos, amt, "new_pos").unwrap();
        builder.build_store(pos_ptr, new_pos).unwrap();
    }

    pub fn llvm_reset<'a>(&self, ctx: &'a Context, builder: &Builder<'a>, ptr: PointerValue<'a>) {
        let pos_ptr = increment_pointer!(ctx, builder, ptr, BitmapIterator::OFFSET_POS);
        builder
            .build_store(pos_ptr, ctx.i64_type().const_zero())
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use inkwell::{context::Context, OptimizationLevel};

    use crate::compiled_iter::array_to_iter;

    #[test]
    fn test_bitmap_iter() {
        use arrow_array::BooleanArray;
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false, true, true, false, true, false,
        ]);

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_bitmap_iter");
        let func = iter.generate_next(&ctx, &module);
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut bool) -> bool>(fname)
                .unwrap()
        };

        let mut buf = false;
        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), false);
        }
    }

    #[test]
    fn test_bitmap_iter_slice() {
        use arrow_array::BooleanArray;
        let full_data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false, true, true, false, true, false,
        ]);

        let data = full_data.slice(2, 8);

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_bitmap_iter");
        let func = iter.generate_next(&ctx, &module);
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut bool) -> bool>(fname)
                .unwrap()
        };

        let mut buf = false;
        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(!buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert!(buf);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), false);
        }
    }

    #[test]
    fn test_bitmap_random_access() {
        use arrow_array::BooleanArray;
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false, true, true, false, true, false,
        ]);

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_bitmap_iter");
        let func = iter.generate_random_access(&ctx, &module).unwrap();
        let fname = func.get_name().to_str().unwrap();
        assert_eq!(
            func.get_type().get_return_type().unwrap(),
            ctx.bool_type().into()
        );

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> bool>(fname)
                .unwrap()
        };

        unsafe {
            assert!(next_func.call(iter.get_mut_ptr(), 0));
            assert!(next_func.call(iter.get_mut_ptr(), 1));
            assert!(!next_func.call(iter.get_mut_ptr(), 2));
            assert!(next_func.call(iter.get_mut_ptr(), 3));
            assert!(!next_func.call(iter.get_mut_ptr(), 4));
            assert!(!next_func.call(iter.get_mut_ptr(), 5));
            assert!(!next_func.call(iter.get_mut_ptr(), 6));
            assert!(!next_func.call(iter.get_mut_ptr(), 7));
            assert!(next_func.call(iter.get_mut_ptr(), 8));
            assert!(next_func.call(iter.get_mut_ptr(), 9));
            assert!(!next_func.call(iter.get_mut_ptr(), 10));
            assert!(next_func.call(iter.get_mut_ptr(), 11));
            assert!(!next_func.call(iter.get_mut_ptr(), 12));
            assert!(next_func.call(iter.get_mut_ptr(), 3));
            assert!(!next_func.call(iter.get_mut_ptr(), 4));
            assert!(next_func.call(iter.get_mut_ptr(), 8));
            assert!(next_func.call(iter.get_mut_ptr(), 9));
            assert!(!next_func.call(iter.get_mut_ptr(), 10));
        };
    }

    #[test]
    fn test_bitmap_random_access_slice() {
        use arrow_array::BooleanArray;
        let full_data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false, true, true, false, true, false,
        ]);
        let data = full_data.slice(2, 8);

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_bitmap_iter");
        let func = iter.generate_random_access(&ctx, &module).unwrap();
        let fname = func.get_name().to_str().unwrap();
        assert_eq!(
            func.get_type().get_return_type().unwrap(),
            ctx.bool_type().into()
        );

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> bool>(fname)
                .unwrap()
        };

        unsafe {
            assert!(next_func.call(iter.get_mut_ptr(), 1));
            assert!(!next_func.call(iter.get_mut_ptr(), 2));
            assert!(!next_func.call(iter.get_mut_ptr(), 3));
            assert!(!next_func.call(iter.get_mut_ptr(), 4));
            assert!(!next_func.call(iter.get_mut_ptr(), 5));
            assert!(next_func.call(iter.get_mut_ptr(), 6));
            assert!(next_func.call(iter.get_mut_ptr(), 7));
            assert!(!next_func.call(iter.get_mut_ptr(), 3));
            assert!(!next_func.call(iter.get_mut_ptr(), 4));
            assert!(next_func.call(iter.get_mut_ptr(), 7));
        };
    }
}
