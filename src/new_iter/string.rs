use arrow_array::{Array, GenericStringArray, StringArray};
use inkwell::{
    builder::Builder,
    context::Context,
    values::{BasicValue, IntValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::increment_pointer;

/// An iterator for string data. Contains a pointer to the offset buffer and the
/// data buffer, along with a `pos` and `len` just like primitive iterators.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct StringIterator {
    offsets: *const i32,
    data: *const u8,
    pos: u64,
    len: u64,
}

impl From<&StringArray> for Box<StringIterator> {
    fn from(value: &StringArray) -> Self {
        Box::new(StringIterator {
            offsets: value.offsets().as_ptr(),
            data: value.values().as_ptr(),
            pos: value.offset() as u64,
            len: (value.len() + value.offset()) as u64,
        })
    }
}

impl StringIterator {
    pub fn llvm_get_offset_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let offset_ptr_ptr = increment_pointer!(ctx, build, ptr, StringIterator::OFFSET_OFFSETS);
        let ptr = build
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                offset_ptr_ptr,
                "offset_ptr",
            )
            .unwrap()
            .into_pointer_value();
        ptr.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        ptr
    }

    pub fn llvm_get_data_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let data_ptr_ptr = increment_pointer!(ctx, build, ptr, StringIterator::OFFSET_DATA);
        let ptr = build
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

    pub fn llvm_pos<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let pos_ptr = increment_pointer!(ctx, build, ptr, StringIterator::OFFSET_POS);
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
        let len_ptr = increment_pointer!(ctx, build, ptr, StringIterator::OFFSET_LEN);
        let len = build
            .build_load(ctx.i64_type(), len_ptr, "len")
            .unwrap()
            .into_int_value();
        len.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        len
    }

    pub fn llvm_increment_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let pos_ptr = increment_pointer!(ctx, builder, ptr, StringIterator::OFFSET_POS);
        let pos = builder
            .build_load(ctx.i64_type(), pos_ptr, "pos")
            .unwrap()
            .into_int_value();
        let new_pos = builder.build_int_add(pos, amt, "new_pos").unwrap();
        builder.build_store(pos_ptr, new_pos).unwrap();
    }
}

/// Same as `StringIterator`, but with 64 bit offsets.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct LargeStringIterator {
    offsets: *const i64,
    data: *const u8,
    pos: u64,
    len: u64,
}

impl LargeStringIterator {
    pub fn llvm_get_offset_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let offset_ptr_ptr =
            increment_pointer!(ctx, build, ptr, LargeStringIterator::OFFSET_OFFSETS);
        build
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                offset_ptr_ptr,
                "offset_ptr",
            )
            .unwrap()
            .into_pointer_value()
    }

    pub fn llvm_get_data_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let data_ptr_ptr = increment_pointer!(ctx, build, ptr, LargeStringIterator::OFFSET_DATA);
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
        let pos_ptr = increment_pointer!(ctx, build, ptr, LargeStringIterator::OFFSET_POS);
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
        let len_ptr = increment_pointer!(ctx, build, ptr, LargeStringIterator::OFFSET_LEN);
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
        let pos_ptr = increment_pointer!(ctx, builder, ptr, LargeStringIterator::OFFSET_POS);
        let pos = builder
            .build_load(ctx.i64_type(), pos_ptr, "pos")
            .unwrap()
            .into_int_value();
        let new_pos = builder.build_int_add(pos, amt, "new_pos").unwrap();
        builder.build_store(pos_ptr, new_pos).unwrap();
    }
}

impl From<&GenericStringArray<i64>> for Box<LargeStringIterator> {
    fn from(value: &GenericStringArray<i64>) -> Self {
        Box::new(LargeStringIterator {
            offsets: value.offsets().as_ptr(),
            data: value.values().as_ptr(),
            pos: value.offset() as u64,
            len: (value.len() + value.offset()) as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{Datum, LargeStringArray, StringArray};
    use inkwell::{context::Context, OptimizationLevel};

    use crate::{
        new_iter::{datum_to_iter, generate_next, generate_random_access},
        pointers_to_str,
    };

    #[test]
    fn test_string_scalar() {
        let s = StringArray::new_scalar("hello");
        let mut iter = datum_to_iter(&s).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_scalar_prim");
        let func_next = generate_next(&ctx, &module, "next", s.get().0.data_type(), &iter).unwrap();
        let fname_next = func_next.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u128) -> bool>(fname_next)
                .unwrap()
        };

        unsafe {
            let mut b: u128 = 0;
            assert!(next_func.call(iter.get_mut_ptr(), &mut b));
            let b = b.to_le_bytes();
            let ptr1 = u64::from_le_bytes(b[0..8].try_into().unwrap());
            let ptr2 = u64::from_le_bytes(b[8..16].try_into().unwrap());
            let len = (ptr2 - ptr1) as usize;
            assert_eq!(len, 5);

            let slice = std::slice::from_raw_parts(ptr1 as *const u8, len);
            let string = std::str::from_utf8(slice).unwrap();
            assert_eq!(string, "hello");
        }
    }

    #[test]
    fn test_large_string_scalar() {
        let s = LargeStringArray::new_scalar("hello");
        let mut iter = datum_to_iter(&s).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_scalar_prim");
        let func_next = generate_next(&ctx, &module, "next", s.get().0.data_type(), &iter).unwrap();
        let fname_next = func_next.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u128) -> bool>(fname_next)
                .unwrap()
        };

        unsafe {
            let mut b: u128 = 0;
            assert!(next_func.call(iter.get_mut_ptr(), &mut b));
            let b = b.to_le_bytes();
            let ptr1 = u64::from_le_bytes(b[0..8].try_into().unwrap());
            let ptr2 = u64::from_le_bytes(b[8..16].try_into().unwrap());
            let len = (ptr2 - ptr1) as usize;
            assert_eq!(len, 5);

            let slice = std::slice::from_raw_parts(ptr1 as *const u8, len);
            let string = std::str::from_utf8(slice).unwrap();
            assert_eq!(string, "hello");
        }
    }

    #[test]
    fn test_string_random_access() {
        let data = StringArray::from(vec!["this", "is", "a", "test"]);
        let mut iter = datum_to_iter(&data).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_string_random_access");
        let func_access =
            generate_random_access(&ctx, &module, "access", data.get().0.data_type(), &iter)
                .unwrap();
        let fname = func_access.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> u128>(fname)
                .unwrap()
        };

        unsafe {
            let b = func.call(iter.get_mut_ptr(), 0);
            assert_eq!(pointers_to_str(b), "this");

            let b = func.call(iter.get_mut_ptr(), 2);
            assert_eq!(pointers_to_str(b), "a");
        }
    }

    #[test]
    fn test_string_next() {
        let data = StringArray::from(vec!["this", "is", "a", "test"]);
        let mut iter = datum_to_iter(&data).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_string_next");
        let func_access =
            generate_next(&ctx, &module, "access", data.get().0.data_type(), &iter).unwrap();
        let fname = func_access.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u128) -> bool>(fname)
                .unwrap()
        };

        unsafe {
            let mut b: u128 = 0;
            assert!(func.call(iter.get_mut_ptr(), &mut b));
            assert_eq!(pointers_to_str(b), "this");

            assert!(func.call(iter.get_mut_ptr(), &mut b));
            assert_eq!(pointers_to_str(b), "is");

            assert!(func.call(iter.get_mut_ptr(), &mut b));
            assert_eq!(pointers_to_str(b), "a");

            assert!(func.call(iter.get_mut_ptr(), &mut b));
            assert_eq!(pointers_to_str(b), "test");

            assert!(!func.call(iter.get_mut_ptr(), &mut b));
        }
    }

    #[test]
    fn test_large_string_random_access() {
        let data = LargeStringArray::from(vec!["this", "is", "a", "test"]);
        let mut iter = datum_to_iter(&data).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_string_random_access");
        let func_access =
            generate_random_access(&ctx, &module, "access", data.get().0.data_type(), &iter)
                .unwrap();
        let fname = func_access.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> u128>(fname)
                .unwrap()
        };

        unsafe {
            let b = func.call(iter.get_mut_ptr(), 0);
            assert_eq!(pointers_to_str(b), "this");

            let b = func.call(iter.get_mut_ptr(), 2);
            assert_eq!(pointers_to_str(b), "a");
        }
    }

    #[test]
    fn test_large_string_next() {
        let data = LargeStringArray::from(vec!["this", "is", "a", "test"]);
        let mut iter = datum_to_iter(&data).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_string_next");
        let func_access =
            generate_next(&ctx, &module, "access", data.get().0.data_type(), &iter).unwrap();
        let fname = func_access.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u128) -> bool>(fname)
                .unwrap()
        };

        unsafe {
            let mut b: u128 = 0;
            assert!(func.call(iter.get_mut_ptr(), &mut b));
            assert_eq!(pointers_to_str(b), "this");

            assert!(func.call(iter.get_mut_ptr(), &mut b));
            assert_eq!(pointers_to_str(b), "is");

            assert!(func.call(iter.get_mut_ptr(), &mut b));
            assert_eq!(pointers_to_str(b), "a");

            assert!(func.call(iter.get_mut_ptr(), &mut b));
            assert_eq!(pointers_to_str(b), "test");

            assert!(!func.call(iter.get_mut_ptr(), &mut b));
        }
    }
}
