use std::ffi::c_void;

use crate::{compiled_kernels::aggregate::StringSaver, compiled_writers::ViewBufferWriter};

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
