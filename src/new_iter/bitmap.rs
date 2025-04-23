use arrow_array::{Array, BooleanArray};
use inkwell::{
    builder::Builder,
    context::Context,
    values::{IntValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::increment_pointer;

/// An iterator for bitmap data. Contains a pointer to the bitmap buffer and the
/// data buffer, along with a `pos` and `len` just like primitive iterators.
/// Note that each element pointed to by `data` contains 8 items/bits.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct BitmapIterator {
    data: *const u8,
    pos: u64,
    len: u64,
}

impl BitmapIterator {
    pub fn llvm_get_data_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let data_ptr_ptr = increment_pointer!(ctx, build, ptr, BitmapIterator::OFFSET_DATA);
        build
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                data_ptr_ptr,
                "data_ptr",
            )
            .unwrap()
            .into_pointer_value()
    }

    pub fn llvm_pos<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
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
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let len_ptr = increment_pointer!(ctx, build, ptr, BitmapIterator::OFFSET_LEN);
        build
            .build_load(ctx.i64_type(), len_ptr, "len")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_increment_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
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
}

impl From<&BooleanArray> for Box<BitmapIterator> {
    fn from(value: &BooleanArray) -> Self {
        Box::new(BitmapIterator {
            data: value.values().values().as_ptr(),
            pos: value.offset() as u64,
            len: (value.len() + value.offset()) as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::Array;
    use inkwell::{context::Context, OptimizationLevel};

    use crate::new_iter::{array_to_iter, generate_next, generate_random_access};

    #[test]
    fn test_bitmap_iter() {
        use arrow_array::BooleanArray;
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false, true, true, false, true, false,
        ]);

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_bitmap_iter");
        let func = generate_next(&ctx, &module, "bitmap_iter", data.data_type(), &iter).unwrap();
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
            assert_eq!(buf, 1);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 1);
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
            assert_eq!(buf, 0);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 0);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 0);
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
            assert_eq!(buf, 1);
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
            assert_eq!(buf, 0);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                false
            );
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
        let func =
            generate_random_access(&ctx, &module, "bitmap_iter", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i8>(fname)
                .unwrap()
        };

        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), 0), 1);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 1), 1);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 2), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 3), 1);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 4), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 5), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 6), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 7), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 8), 1);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 9), 1);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 10), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 11), 1);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 12), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 3), 1);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 4), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 8), 1);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 9), 1);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 10), 0);
        };
    }
}
