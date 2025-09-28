use std::ffi::c_void;

use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    values::{FunctionValue, IntValue},
    AddressSpace,
};

#[no_mangle]
pub extern "C" fn debug_i64(i: i64) {
    println!("From generated code: {}", i);
}

#[allow(dead_code)]
pub fn llvm_debug_i64<'a>(ctx: &'a Context, llvm_mod: &Module<'a>, b: &Builder, val: IntValue) {
    let func = get_or_add_debug_i64(ctx, llvm_mod);
    b.build_call(func, &[val.into()], "debug_i64").unwrap();
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
