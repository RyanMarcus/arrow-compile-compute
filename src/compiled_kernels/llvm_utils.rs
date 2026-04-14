use std::ffi::c_void;

use inkwell::{
    context::Context,
    module::{Linkage, Module},
    types::BasicType,
    values::FunctionValue,
    AddressSpace,
};

use crate::{compiled_writers::ViewBufferWriter, PrimitiveType};

#[no_mangle]
pub extern "C" fn str_writer_append_bytes(ptr: *const u8, len: u64, vec: *mut c_void) {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    let vec = unsafe { &mut *(vec as *mut Vec<u8>) };
    vec.extend_from_slice(bytes);
}

#[no_mangle]
pub extern "C" fn str_view_writer_append_bytes(
    str_ptr: *const u8,
    len: u64,
    view_ptr: *mut u128,
    data: *mut c_void,
) {
    let bytes = unsafe { std::slice::from_raw_parts(str_ptr, len as usize) };
    let data = unsafe { &mut *(data as *mut ViewBufferWriter) };
    let view = data.write(bytes);

    unsafe {
        *view_ptr = view as u128;
    }
}

#[derive(Default)]
pub struct StringSaver {
    data: Vec<Box<[u8]>>,
}

impl StringSaver {
    pub fn insert(&mut self, data: &[u8]) -> u128 {
        self.data.push(Box::from(data));
        let last = self.data.last().unwrap();
        let start = last.as_ptr();
        let end = start.wrapping_add(last.len());
        ((start as u64) as u128) | (((end as u64) as u128) << 64)
    }

    pub fn finalize(self) -> Vec<Box<[u8]>> {
        self.data
    }
}

pub fn llvm_add_save_to_string_saver<'ctx, 'a>(
    ctx: &'ctx Context,
    module: &'a Module<'ctx>,
) -> FunctionValue<'a> {
    module
        .get_function("save_to_string_saver")
        .unwrap_or_else(|| {
            module.add_function(
                "save_to_string_saver",
                ctx.i128_type().fn_type(
                    &[
                        ctx.ptr_type(AddressSpace::default()).into(),
                        ctx.i64_type().into(),
                        ctx.ptr_type(AddressSpace::default()).into(),
                        ctx.ptr_type(AddressSpace::default()).into(),
                    ],
                    false,
                ),
                None,
            )
        })
}

#[no_mangle]
pub extern "C" fn save_to_string_saver(
    str_ptr: *const u8,
    len: u64,
    saver: *mut c_void,
    out_ptr: *mut u128,
) {
    let bytes = unsafe { std::slice::from_raw_parts(str_ptr, len as usize) };
    let data = unsafe { &mut *(saver as *mut StringSaver) };
    let result = data.insert(bytes);

    unsafe {
        *out_ptr = result as u128;
    }
}

pub fn llvm_add_save_ptrs_string_saver<'ctx, 'a>(
    ctx: &'ctx Context,
    module: &'a Module<'ctx>,
) -> FunctionValue<'a> {
    module
        .get_function("save_ptrs_to_string_saver")
        .unwrap_or_else(|| {
            module.add_function(
                "save_ptrs_to_string_saver",
                PrimitiveType::P64x2.llvm_type(ctx).fn_type(
                    &[
                        ctx.ptr_type(AddressSpace::default()).into(),
                        ctx.ptr_type(AddressSpace::default()).into(),
                        ctx.ptr_type(AddressSpace::default()).into(),
                    ],
                    false,
                ),
                Some(Linkage::External),
            )
        })
}

#[no_mangle]
pub extern "C" fn save_ptrs_to_string_saver(
    str_ptr1: *const u8,
    str_ptr2: *const u8,
    saver: *mut c_void,
) -> u128 {
    let bytes = unsafe {
        let len = str_ptr2.offset_from_unsigned(str_ptr1);
        std::slice::from_raw_parts(str_ptr1, len)
    };
    let data = unsafe { &mut *(saver as *mut StringSaver) };
    data.insert(bytes)
}
