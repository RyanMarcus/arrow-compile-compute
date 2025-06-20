use arrow_array::{Array, BooleanArray};
use inkwell::{
    builder::Builder,
    context::Context,
    values::{IntValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::increment_pointer;

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct SetBitIterator {
    data: *const u8,
    index: u64,
    end_index: u64,
    byte: u8,
    end_byte: u8,
}

impl From<&BooleanArray> for Box<SetBitIterator> {
    fn from(value: &BooleanArray) -> Self {
        let all_byte = value.values().values()[value.offset() / 8];
        let index_into_byte = value.offset() % 8;
        // We might have stuff in the byte that shouldn't be iterated over
        let mask = !((1u8 << index_into_byte) - 1);
        let byte = all_byte & mask;

        // Note that we are indexing by byte (with each byte containing
        // eight values)
        let start_index = ((value.offset() as u64) / 8) + 1;
        let end_index = 1 + ((value.len() + (value.offset() % 8) - 1) / 8) as u64;

        let all_end_byte = value.values().values()[end_index as usize - 1];
        let index_into_end_byte = (value.len() + value.offset()) % 8;
        let end_byte = if index_into_end_byte == 0 {
            all_end_byte
        } else {
            let end_mask = (!0u8) >> (8 - index_into_end_byte);
            all_end_byte & end_mask
        };

        println!("Working");
        println!("Input. Offset: {}, Len: {}, All Byte: {:08b}", value.offset(), value.len(), all_byte);
        println!("start_index: {}, end_index: {}, byte: {:08b}", start_index, end_index, byte);
        println!("all_end_byte: {:08b}, end_byte: {:08b}", all_end_byte, end_byte);

        Box::new(SetBitIterator {
            data: value.values().values().as_ptr(),
            index: start_index,
            end_index,
            byte,
            end_byte,
        })
    }
}

impl SetBitIterator {
    pub fn llvm_get_data_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let data_ptr_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_DATA);

        build
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                data_ptr_ptr,
                "data_ptr",
            )
            .unwrap()
            .into_pointer_value()
    }

    pub fn llvm_get_index<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let index_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_INDEX);
        build
            .build_load(ctx.i64_type(), index_ptr, "index")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_get_end_index<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let end_index_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_END_INDEX);
        build
            .build_load(ctx.i64_type(), end_index_ptr, "end_index")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_get_byte<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let byte_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_BYTE);
        build
            .build_load(ctx.i8_type(), byte_ptr, "byte")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_get_end_byte<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let end_byte_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_END_BYTE);
        build
            .build_load(ctx.i8_type(), end_byte_ptr, "end_byte")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_clear_trailing_bit<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) {
        let byte_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_BYTE);
        let byte = build
            .build_load(ctx.i8_type(), byte_ptr, "load_byte")
            .unwrap()
            .into_int_value();
        // Use no unsigned wrap since we know we'll never call this function when byte == 0.
        let byte_minus_one = build
            .build_int_nuw_sub(byte, ctx.i8_type().const_int(1, false), "byte_minus_one")
            .unwrap();
        let byte_and = build.build_and(byte, byte_minus_one, "byte_and").unwrap();
        build.build_store(byte_ptr, byte_and).unwrap();
    }

    pub fn llvm_load_byte_at_index<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) {
        let data_ptr = self.llvm_get_data_ptr(ctx, build, ptr);
        let curr_index = self.llvm_get_index(ctx, build, ptr);
        let byte_in_data_ptr =
            unsafe { build.build_gep(ctx.i8_type(), data_ptr, &[curr_index], "byte_in_data_ptr") }
                .unwrap();
        let byte_in_data = build
            .build_load(ctx.i8_type(), byte_in_data_ptr, "load_byte_in_data")
            .unwrap();
        let byte_pointer = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_BYTE);
        build.build_store(byte_pointer, byte_in_data).unwrap();
    }

    pub fn llvm_load_end_byte<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) {
        let byte_pointer = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_BYTE);
        let end_byte = self.llvm_get_end_byte(ctx, build, ptr);
        build.build_store(byte_pointer, end_byte).unwrap();
    }

    pub fn llvm_increment_index<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let curr_pos_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_INDEX);
        let curr_pos = build
            .build_load(ctx.i64_type(), curr_pos_ptr, "curr_pos")
            .unwrap()
            .into_int_value();
        let new_pos = build.build_int_add(curr_pos, amt, "new_pos").unwrap();
        build.build_store(curr_pos_ptr, new_pos).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{Array, BooleanArray};
    use inkwell::{context::Context, OptimizationLevel};

    use crate::new_iter::{array_to_setbit_iter, generate_next};

    #[test]
    fn test_setbit_slice() {
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false, true, true,
        ]);
        let data2 = data.slice(0, 8);
        
        array_to_setbit_iter(&data);
        let mut iter = array_to_setbit_iter(&data2);

        let ctx = Context::create();
        let module = ctx.create_module("setbit_test");
        let func = generate_next(&ctx, &module, "setbit_iter", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();
        
        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>(fname)
                .unwrap()
        };

        let mut buf: u64 = 0;
        unsafe {
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 0);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 1);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 3);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
        }
    }

    #[test]
    fn test_setbit_slice2() {
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false, true, true,
        ]);
        let data2 = data.slice(1, 9);
        
        array_to_setbit_iter(&data);
        let mut iter = array_to_setbit_iter(&data2);

        let ctx = Context::create();
        let module = ctx.create_module("setbit_test");
        let func = generate_next(&ctx, &module, "setbit_iter", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();
        
        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>(fname)
                .unwrap()
        };

        let mut buf: u64 = 0;
        unsafe {
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 1);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 3);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 8);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 9);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
        }
    }

    #[test]
    fn test_setbit_slice3() {
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false, true, true,
        ]);
        let data2 = data.slice(1, 8);
        
        array_to_setbit_iter(&data);
        let mut iter = array_to_setbit_iter(&data2);

        let ctx = Context::create();
        let module = ctx.create_module("setbit_test");
        let func = generate_next(&ctx, &module, "setbit_iter", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();
        
        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>(fname)
                .unwrap()
        };

        let mut buf: u64 = 0;
        unsafe {
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 1);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 3);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 8);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
        }
    }

    #[test]
    fn test_setbit_slice4() {
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false,
            true, true, false, true, false, false, false, false,
            true, true, false, true, false, false, false, false,
        ]);
        let data2 = data.slice(7, 10);
        
        array_to_setbit_iter(&data);
        let mut iter = array_to_setbit_iter(&data2);

        let ctx = Context::create();
        let module = ctx.create_module("setbit_test");
        let func = generate_next(&ctx, &module, "setbit_iter", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();
        
        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>(fname)
                .unwrap()
        };

        let mut buf: u64 = 0;
        unsafe {
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 8);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 9);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 11);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 16);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
        }
    }


    #[test]
    fn test_setbit_iter() {
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false, true, true,
        ]);

        let mut iter = array_to_setbit_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("setbit_test");
        let func = generate_next(&ctx, &module, "setbit_iter", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();
        
        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>(fname)
                .unwrap()
        };

        let mut buf: u64 = 0;
        unsafe {
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 0);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 1);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 3);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 8);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 9);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
        }
    }

    #[test]
    fn test_setbit_iter2() {
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false,
        ]);

        let mut iter = array_to_setbit_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("setbit_test");
        let func = generate_next(&ctx, &module, "setbit_iter", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();
        
        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>(fname)
                .unwrap()
        };

        let mut buf: u64 = 0;
        unsafe {
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 0);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 1);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 3);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                false
            );
        }
    }
}
