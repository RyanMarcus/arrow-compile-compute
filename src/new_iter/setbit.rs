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
    long: u64,
    last_long: u64,
}

fn get_long_offset_len(bit_offset: usize, bit_len: usize) -> (usize, usize) {
    let start_index = bit_offset / 64;
    let long_len = 1 + ((bit_len + (bit_offset % 64) - 1) / 64);
    (start_index, long_len)
}

fn get_byte_offset_len(bit_offset: usize, bit_len: usize) -> (usize, usize) {
    let start_index = bit_offset / 8;
    let byte_len = 1 + ((bit_len + (bit_offset % 8) - 1) / 8);
    (start_index, byte_len)
}

fn get_first_last_bytes(data: &[u8], bit_offset: usize, bit_len: usize) -> (u8, u8) {
    let (byte_offset, byte_len) = get_byte_offset_len(bit_offset, bit_len);

    // Get the first byte
    // We need to remove least significant bits because of slicing
    let all_first_byte = data[byte_offset];
    let index_into_first_byte = bit_offset % 8;
    let mask = !((1u8 << index_into_first_byte) - 1);
    let mut first_byte = all_first_byte & mask;

    // Get the last byte
    // We need to remove significant bits because of slicing
    let all_last_byte = data[byte_offset + byte_len - 1];
    let index_into_end_byte = (bit_len + bit_offset) % 8;
    let mut last_byte = if index_into_end_byte == 0 {
        all_last_byte
    } else {
        let end_mask = (!0u8) >> (8 - index_into_end_byte);
        all_last_byte & end_mask
    };

    // Special case where the first byte is the same as the last byte
    if byte_len == 1 {
        let one_byte = first_byte & last_byte;
        first_byte = one_byte;
        last_byte = one_byte;
    }

    (first_byte, last_byte)
}

fn get_first_last_long(data: &[u8], long_len: usize, byte_offset: usize, byte_len: usize, first_byte: u8, last_byte: u8) -> (u64, u64) {
    let last_byte_offset = byte_offset + byte_len - 1;
    
    // Put the first byte into the first long
    let mut first_long: [u8; 8]= [0; 8];
    let position_in_first_long = byte_offset % 8;
    first_long[position_in_first_long] = first_byte;
    
    // Put the remaining bytes in the first long
    let next_8_multiple = byte_offset + (8 - (byte_offset % 8));
    for i in (byte_offset + 1)..next_8_multiple.min(byte_offset + byte_len) {
        first_long[i % 8] = data[i];
    }

    // Put the last byte into the last long
    let mut last_long: [u8; 8]= [0; 8];
    let position_in_last_long = last_byte_offset % 8;
    last_long[position_in_last_long] = last_byte;

    // Put the remaining bytes into the last long
    let last_8_multiple = last_byte_offset - (last_byte_offset % 8);
    for i in last_8_multiple.max(byte_offset)..last_byte_offset {
        last_long[i % 8] = data[i];
    }

    // Special case where the first long and last long are the same
    if long_len == 1 {
        first_long[position_in_last_long] = last_byte;
        last_long[position_in_first_long] = first_byte;
    }

    println!("{:?}", first_long);

    let first_long = u64::from_le_bytes(first_long);
    let last_long = u64::from_le_bytes(last_long);

    (first_long, last_long)
}

impl From<&BooleanArray> for Box<SetBitIterator> {
    fn from(value: &BooleanArray) -> Self {
        // Note that an array like [true, true, false, false], [true, false, true, false]
        // is packed as 0011, 0101.

        // Generate the first and last bytes
        let (first_byte, last_byte) = get_first_last_bytes(value.values().values(), value.offset(), value.len());
        
        // Generate the first and last longs
        let (byte_offset, byte_len) = get_byte_offset_len(value.offset(), value.len());
        let (mut long_offset, long_len) = get_long_offset_len(value.offset(), value.len());
        let (first_long, last_long) = get_first_last_long(value.values().values(), long_len, byte_offset, byte_len, first_byte, last_byte);

        long_offset += 1;

        Box::new(SetBitIterator {
            data: value.values().values().as_ptr(),
            index: long_offset as u64,
            end_index: long_len as u64,
            long: first_long,
            last_long
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

    pub fn llvm_get_long<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let byte_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_LONG);
        build
            .build_load(ctx.i64_type(), byte_ptr, "long")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_get_last_long<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let end_byte_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_LAST_LONG);
        build
            .build_load(ctx.i64_type(), end_byte_ptr, "last_long")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_clear_trailing_bit<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) {
        let long_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_LONG);
        let long = build
            .build_load(ctx.i64_type(), long_ptr, "load_long")
            .unwrap()
            .into_int_value();
        // Use no unsigned wrap since we know we'll never call this function when byte == 0.
        let long_minus_one = build
            .build_int_nuw_sub(long, ctx.i64_type().const_int(1, false), "long_minus_one")
            .unwrap();
        let long_and = build.build_and(long, long_minus_one, "long_and").unwrap();
        build.build_store(long_ptr, long_and).unwrap();
    }

    pub fn llvm_load_long_at_index<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) {
        let data_ptr = self.llvm_get_data_ptr(ctx, build, ptr);
        let curr_index = self.llvm_get_index(ctx, build, ptr);
        let byte_curr_index = build
            .build_int_mul(curr_index, ctx.i64_type().const_int(4, false), "byte_curr_index")
            .unwrap();
        let byte_in_data_ptr =
            unsafe { build.build_gep(ctx.i8_type(), data_ptr, &[byte_curr_index], "byte_in_data_ptr") }
                .unwrap();
        let long_in_data_ptr = build
            .build_bit_cast(byte_in_data_ptr, ctx.ptr_type(AddressSpace::default()), "byte_to_long_ptr")
            .unwrap()
            .into_pointer_value();
        let long_in_data = build
            .build_load(ctx.i64_type(), long_in_data_ptr, "load_long_in_data")
            .unwrap();
        let long_pointer = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_LONG);
        build.build_store(long_pointer, long_in_data).unwrap();
    }

    pub fn llvm_load_last_long<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) {
        let long_pointer = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_LONG);
        let last_long = self.llvm_get_last_long(ctx, build, ptr);
        build.build_store(long_pointer, last_long).unwrap();
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

    use crate::new_iter::{array_to_setbit_iter, generate_next, setbit::{get_byte_offset_len, get_first_last_bytes, get_first_last_long, get_long_offset_len}};

    #[test]
    fn test_setbit_get_long_offset_len() {
        let (start, len) = get_long_offset_len(0, 64);
        assert_eq!(start, 0);
        assert_eq!(len, 1);

        let (start, len) = get_long_offset_len(1, 64);
        assert_eq!(start, 0);
        assert_eq!(len, 2);

        let (start, len) = get_long_offset_len(63, 66);
        assert_eq!(start, 0);
        assert_eq!(len, 3);

        let (start, len) = get_long_offset_len(0, 10);
        assert_eq!(start, 0);
        assert_eq!(len, 1);

        let (start, len) = get_long_offset_len(64, 64);
        assert_eq!(start, 1);
        assert_eq!(len, 1);

        let (start, len) = get_long_offset_len(64, 65);
        assert_eq!(start, 1);
        assert_eq!(len, 2);
    }

    #[test]
    fn test_setbit_get_byte_offset_len() {
        let (start, len) = get_byte_offset_len(0, 64);
        assert_eq!(start, 0);
        assert_eq!(len, 8);

        let (start, len) = get_byte_offset_len(1, 64);
        assert_eq!(start, 0);
        assert_eq!(len, 9);

        let (start, len) = get_byte_offset_len(63, 66);
        assert_eq!(start, 7);
        assert_eq!(len, 10);

        let (start, len) = get_byte_offset_len(0, 10);
        assert_eq!(start, 0);
        assert_eq!(len, 2);

        let (start, len) = get_byte_offset_len(64, 64);
        assert_eq!(start, 8);
        assert_eq!(len, 8);

        let (start, len) = get_byte_offset_len(64, 65);
        assert_eq!(start, 8);
        assert_eq!(len, 9);
    }

    #[test]
    fn test_setbit_get_first_last_bytes() {
        let data: Vec<u8> = vec![1, 2, 3];
        let (first, last) = get_first_last_bytes(&data, 0, 24);
        assert_eq!(first, 1);
        assert_eq!(last, 3);

        let data: Vec<u8> = vec![1, 2];
        let (first, last) = get_first_last_bytes(&data, 0, 16);
        assert_eq!(first, 1);
        assert_eq!(last, 2);

        let data: Vec<u8> = vec![1];
        let (first, last) = get_first_last_bytes(&data, 0, 8);
        assert_eq!(first, 1);
        assert_eq!(last, 1);

        let data: Vec<u8> = vec![0b0000_0001, 0b0000_0010, 0b0000_0011];
        let (first, last) = get_first_last_bytes(&data, 4, 16);
        assert_eq!(first, 0);
        assert_eq!(last, 0b0000_0011);

        let data: Vec<u8> = vec![0b0000_0001, 0b0000_0010, 0b0000_0011];
        let (first, last) = get_first_last_bytes(&data, 4, 13);
        assert_eq!(first, 0);
        assert_eq!(last, 0b0000_0001);

        let data: Vec<u8> = vec![0b0000_0001, 0b0000_0010, 0b0000_0011];
        let (first, last) = get_first_last_bytes(&data, 4, 12);
        assert_eq!(first, 0);
        assert_eq!(last, 0b0000_0010);

        let data: Vec<u8> = vec![0b0000_0001, 0b0000_0010, 0b0000_0011];
        let (first, last) = get_first_last_bytes(&data, 0, 4);
        assert_eq!(first, 0b0000_0001);
        assert_eq!(last, 0b0000_0001);
    
        let data: Vec<u8> = vec![0b0000_0001, 0b0000_0010, 0b0000_0011];
        let (first, last) = get_first_last_bytes(&data, 1, 4);
        assert_eq!(first, 0b0000_0000);
        assert_eq!(last, 0b0000_0000);

        let data: Vec<u8> = vec![0b0000_0001, 0b1111_1111, 0b0000_0011];
        let (first, last) = get_first_last_bytes(&data, 1, 9);
        assert_eq!(first, 0b0000_0000);
        assert_eq!(last, 0b0000_0011);

        let data: Vec<u8> = vec![0b0000_0001, 0b1111_1111, 0b0000_1111];
        let (first, last) = get_first_last_bytes(&data, 1, 17);
        assert_eq!(first, 0b0000_0000);
        assert_eq!(last, 0b0000_0011);

        let data: Vec<u8> = vec![0b1101_0101];
        let (first, last) = get_first_last_bytes(&data, 4, 2);
        assert_eq!(first, 0b0001_0000);
        assert_eq!(last, 0b0001_0000);

        let data: Vec<u8> = vec![0b1101_0101, 0b1111_1111];
        let (first, last) = get_first_last_bytes(&data, 6, 4);
        assert_eq!(first, 0b1100_0000);
        assert_eq!(last, 0b0000_0011);
    }

    #[test]
    fn test_setbit_get_first_last_long() {
        let data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04];
        let (_, long_len) = get_long_offset_len(0, 64);
        let (first, last) = get_first_last_long(&data, long_len, 0, 4, 0xA0, 0xB0);
        assert_eq!(first, 0xB00302A0);
        assert_eq!(last, 0xB00302A0);

        let data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let (_, long_len) = get_long_offset_len(0, 64);
        let (first, last) = get_first_last_long(&data, long_len, 0, 8, 0xA0, 0xB0);
        assert_eq!(first, 0xB0070605040302A0);
        assert_eq!(last, 0xB0070605040302A0);

        let data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10];
        let (_, long_len) = get_long_offset_len(0, 128);
        let (first, last) = get_first_last_long(&data, long_len, 0, 16, 0xA0, 0xB0);
        assert_eq!(first, 0x08070605040302A0);
        assert_eq!(last, 0xB00F0E0D0C0B0A09);

        let data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10];
        let (_, long_len) = get_long_offset_len(32, 64);
        let (first, last) = get_first_last_long(&data, long_len, 4, 8, 0xA0, 0xB0);
        assert_eq!(first, 0x080706A000000000);
        assert_eq!(last, 0x00000000B00B0A09);

        let data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10];
        let (_, long_len) = get_long_offset_len(32, 32);
        let (first, last) = get_first_last_long(&data, long_len, 4, 4, 0xA0, 0xB0);
        assert_eq!(first, 0xB00706A000000000);
        assert_eq!(last, 0xB00706A000000000);

        let data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10];
        let (_, long_len) = get_long_offset_len(32, 48);
        let (first, last) = get_first_last_long(&data, long_len, 4, 6, 0xA0, 0xB0);
        assert_eq!(first, 0x080706A000000000);
        assert_eq!(last, 0x000000000000B009);

        let data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
                                 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18];
        let (_, long_len) = get_long_offset_len(32, 128);
        let (first, last) = get_first_last_long(&data, long_len, 4, 16, 0xA0, 0xB0);
        assert_eq!(first, 0x080706A000000000);
        assert_eq!(last, 0x00000000B0131211);

        let data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
                                 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18];
        let (_, long_len) = get_long_offset_len(32, 8);
        let (first, last) = get_first_last_long(&data, long_len, 4, 1, 0xA0, 0xA0);
        assert_eq!(first, 0x000000A000000000);
        assert_eq!(last, 0x000000A000000000);
    }

    #[test]
    fn test_setbit_slice() {
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false, true, true,
        ]);
        let data2 = data.slice(0, 8);
        
        array_to_setbit_iter(&data);
        let mut iter = array_to_setbit_iter(&data2);
        println!("{:?}", iter);

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
    fn test_setbit_slice5() {
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, true, true, true,
        ]);
        let data2 = data.slice(1, 4);
        
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

    #[test]
    fn test_setbit_iter3() {
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false,
            false, false, false, false, false, false, false, true,
            false, false, false, false, false, false, false, true,
            false, false, false, false, false, false, false, true,
            false, false, false, false, false, false, false, true,
            false, false, false, false, false, false, false, true,
            false, false, false, false, false, false, false, true,
            false, false, false, false, false, false, false, true,
            false, false, false, false, false, false, false, true,
            false, false, false, false, false, false, false, true,
            false, false, false, false, false, false, false, true,
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
            assert_eq!(buf, 15);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 23);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 31);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 39);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 47);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 55);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 63);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 71);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 79);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut u64),
                true
            );
            assert_eq!(buf, 87);


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
