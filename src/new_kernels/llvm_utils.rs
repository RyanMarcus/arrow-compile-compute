use std::ffi::c_void;

use inkwell::{context::Context, module::Module, values::FunctionValue, AddressSpace};

#[no_mangle]
pub extern "C" fn str_writer_append_bytes(ptr: *const u8, len: u64, vec: *mut c_void) {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    let vec = unsafe { &mut *(vec as *mut Vec<u8>) };
    vec.extend_from_slice(bytes);
}

#[no_mangle]
pub extern "C" fn debug_i64(i: i64) {
    println!("From generated code: {}", i);
}

#[allow(dead_code)]
pub fn get_or_add_debug_i64<'a>(ctx: &'a Context, llvm_mod: &Module<'a>) -> FunctionValue<'a> {
    llvm_mod.get_function("debug_i64").unwrap_or_else(|| {
        llvm_mod.add_function(
            "debug_i64",
            ctx.void_type().fn_type(&[ctx.i64_type().into()], false),
            None,
        )
    })
}

#[no_mangle]
pub extern "C" fn debug_ptr(ptr: *const c_void) {
    println!("From generated code: {:?}", ptr);
}

#[allow(dead_code)]
pub fn get_or_add_debug_ptr<'a>(ctx: &'a Context, llvm_mod: &Module<'a>) -> FunctionValue<'a> {
    llvm_mod.get_function("debug_ptr").unwrap_or_else(|| {
        llvm_mod.add_function(
            "debug_ptr",
            ctx.void_type()
                .fn_type(&[ctx.ptr_type(AddressSpace::default()).into()], false),
            None,
        )
    })
}
