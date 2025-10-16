use std::ffi::c_void;

use inkwell::{
    context::Context,
    module::{Linkage, Module},
    values::FunctionValue,
    IntPredicate,
};

use crate::{
    compiled_kernels::{aggregate::StringSaver, cmp::add_memcmp, writers::ViewBufferWriter},
    declare_blocks, increment_pointer, pointer_diff, PrimitiveType,
};

/// Generates a string `starts_with` function. The first argument is the
/// haystack, the second is the needle.
pub fn add_str_startswith<'a>(ctx: &'a Context, llvm_mod: &Module<'a>) -> FunctionValue<'a> {
    if let Some(func) = llvm_mod.get_function("str_startswith") {
        return func;
    }

    let memcmp = add_memcmp(ctx, llvm_mod);
    let bool_type = ctx.bool_type();
    let str_type = PrimitiveType::P64x2.llvm_type(&ctx).into_struct_type();

    let func = llvm_mod.add_function(
        "str_startswith",
        bool_type.fn_type(&[str_type.into(), str_type.into()], false),
        Some(Linkage::Private),
    );

    declare_blocks!(ctx, func, entry, too_short, check);
    let b = ctx.create_builder();

    let haystack = func.get_nth_param(0).unwrap().into_struct_value();
    let needle = func.get_nth_param(1).unwrap().into_struct_value();

    b.position_at_end(entry);
    let haystack_start = b
        .build_extract_value(haystack, 0, "haystack_start")
        .unwrap()
        .into_pointer_value();
    let haystack_end = b
        .build_extract_value(haystack, 1, "haystack_end")
        .unwrap()
        .into_pointer_value();
    let haystack_len = pointer_diff!(ctx, b, haystack_start, haystack_end);

    let needle_start = b
        .build_extract_value(needle, 0, "needle_start")
        .unwrap()
        .into_pointer_value();
    let needle_end = b
        .build_extract_value(needle, 1, "needle_end")
        .unwrap()
        .into_pointer_value();
    let needle_len = pointer_diff!(ctx, b, needle_start, needle_end);

    let can_fit = b
        .build_int_compare(IntPredicate::SLE, needle_len, haystack_len, "can_fit")
        .unwrap();
    b.build_conditional_branch(can_fit, check, too_short)
        .unwrap();

    b.position_at_end(too_short);
    b.build_return(Some(&bool_type.const_zero())).unwrap();

    b.position_at_end(check);
    let haystack_end = increment_pointer!(ctx, b, haystack_start, 1, needle_len);
    let trunc_haystack = str_type.const_zero();
    let trunc_haystack = b
        .build_insert_value(trunc_haystack, haystack_start, 0, "start")
        .unwrap();
    let trunc_haystack = b
        .build_insert_value(trunc_haystack, haystack_end, 1, "end")
        .unwrap()
        .into_struct_value();
    let cmp = b
        .build_call(
            memcmp,
            &[trunc_haystack.into(), needle.into()],
            "memcmp_result",
        )
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    let res = b
        .build_int_compare(
            IntPredicate::EQ,
            cmp,
            ctx.i64_type().const_zero(),
            "could_match",
        )
        .unwrap();
    b.build_return(Some(&res)).unwrap();

    func
}

/// Generates a string `ends_with` function. The first argument is the
/// haystack, the second is the needle.
pub fn add_str_endswith<'a>(ctx: &'a Context, llvm_mod: &Module<'a>) -> FunctionValue<'a> {
    if let Some(func) = llvm_mod.get_function("str_endswith") {
        return func;
    }

    let memcmp = add_memcmp(ctx, llvm_mod);
    let bool_type = ctx.bool_type();
    let str_type = PrimitiveType::P64x2.llvm_type(&ctx).into_struct_type();

    let func = llvm_mod.add_function(
        "str_endswith",
        bool_type.fn_type(&[str_type.into(), str_type.into()], false),
        Some(Linkage::Private),
    );

    declare_blocks!(ctx, func, entry, too_short, check);
    let b = ctx.create_builder();

    let haystack = func.get_nth_param(0).unwrap().into_struct_value();
    let needle = func.get_nth_param(1).unwrap().into_struct_value();

    b.position_at_end(entry);
    let haystack_start = b
        .build_extract_value(haystack, 0, "haystack_start")
        .unwrap()
        .into_pointer_value();
    let haystack_end = b
        .build_extract_value(haystack, 1, "haystack_end")
        .unwrap()
        .into_pointer_value();
    let haystack_len = pointer_diff!(ctx, b, haystack_start, haystack_end);

    let needle_start = b
        .build_extract_value(needle, 0, "needle_start")
        .unwrap()
        .into_pointer_value();
    let needle_end = b
        .build_extract_value(needle, 1, "needle_end")
        .unwrap()
        .into_pointer_value();
    let needle_len = pointer_diff!(ctx, b, needle_start, needle_end);

    let can_fit = b
        .build_int_compare(IntPredicate::SLE, needle_len, haystack_len, "can_fit")
        .unwrap();
    b.build_conditional_branch(can_fit, check, too_short)
        .unwrap();

    b.position_at_end(too_short);
    b.build_return(Some(&bool_type.const_zero())).unwrap();

    b.position_at_end(check);
    let neg_needle_len = b.build_int_neg(needle_len, "neg_needle_len").unwrap();
    let suffix_start = increment_pointer!(ctx, b, haystack_end, 1, neg_needle_len);
    let trunc_haystack = str_type.const_zero();
    let trunc_haystack = b
        .build_insert_value(trunc_haystack, suffix_start, 0, "start")
        .unwrap();
    let trunc_haystack = b
        .build_insert_value(trunc_haystack, haystack_end, 1, "end")
        .unwrap()
        .into_struct_value();
    let cmp = b
        .build_call(
            memcmp,
            &[trunc_haystack.into(), needle.into()],
            "memcmp_result",
        )
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    let res = b
        .build_int_compare(
            IntPredicate::EQ,
            cmp,
            ctx.i64_type().const_zero(),
            "could_match",
        )
        .unwrap();
    b.build_return(Some(&res)).unwrap();

    func
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::{context::Context, AddressSpace, OptimizationLevel};

    #[test]
    fn str_startswith_returns_expected_results() {
        let ctx = Context::create();
        let module = ctx.create_module("str_startswith_test");
        let str_fn = add_str_startswith(&ctx, &module);

        let bool_type = ctx.bool_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();

        let wrapper = module.add_function(
            "startswith_wrapper",
            bool_type.fn_type(
                &[
                    ptr_type.into(),
                    i64_type.into(),
                    ptr_type.into(),
                    i64_type.into(),
                ],
                false,
            ),
            None,
        );

        let entry = ctx.append_basic_block(wrapper, "entry");
        let builder = ctx.create_builder();
        builder.position_at_end(entry);

        let hay_ptr = wrapper.get_nth_param(0).unwrap().into_pointer_value();
        let hay_len = wrapper.get_nth_param(1).unwrap().into_int_value();
        let needle_ptr = wrapper.get_nth_param(2).unwrap().into_pointer_value();
        let needle_len = wrapper.get_nth_param(3).unwrap().into_int_value();

        let hay_end = increment_pointer!(ctx, builder, hay_ptr, 1, hay_len);
        let needle_end = increment_pointer!(ctx, builder, needle_ptr, 1, needle_len);

        let str_type = PrimitiveType::P64x2.llvm_type(&ctx).into_struct_type();

        let hay_struct = builder
            .build_insert_value(str_type.const_zero(), hay_ptr, 0, "hay_start")
            .unwrap();
        let hay_struct = builder
            .build_insert_value(hay_struct, hay_end, 1, "hay_end")
            .unwrap()
            .into_struct_value();

        let needle_struct = builder
            .build_insert_value(str_type.const_zero(), needle_ptr, 0, "needle_start")
            .unwrap();
        let needle_struct = builder
            .build_insert_value(needle_struct, needle_end, 1, "needle_end")
            .unwrap()
            .into_struct_value();

        let call = builder
            .build_call(
                str_fn,
                &[hay_struct.into(), needle_struct.into()],
                "strncmp",
            )
            .unwrap();
        let res = call.try_as_basic_value().unwrap_left().into_int_value();
        builder.build_return(Some(&res)).unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let wrapper_fn = unsafe {
            ee.get_function::<unsafe extern "C" fn(*const u8, i64, *const u8, i64) -> bool>(
                "startswith_wrapper",
            )
            .unwrap()
        };

        unsafe {
            let hay = b"hello world" as &[u8];
            let needle = b"hello" as &[u8];
            assert!(wrapper_fn.call(
                hay.as_ptr(),
                hay.len() as i64,
                needle.as_ptr(),
                needle.len() as i64,
            ));

            let exact = b"hello" as &[u8];
            assert!(wrapper_fn.call(
                exact.as_ptr(),
                exact.len() as i64,
                exact.as_ptr(),
                exact.len() as i64,
            ));

            let short_hay = b"hi" as &[u8];
            assert!(!wrapper_fn.call(
                short_hay.as_ptr(),
                short_hay.len() as i64,
                needle.as_ptr(),
                needle.len() as i64,
            ));

            let mismatched = b"hello" as &[u8];
            let mismatch = b"hella" as &[u8];
            assert!(!wrapper_fn.call(
                mismatched.as_ptr(),
                mismatched.len() as i64,
                mismatch.as_ptr(),
                mismatch.len() as i64,
            ));

            let any = b"rust" as &[u8];
            let empty = b"" as &[u8];
            assert!(wrapper_fn.call(
                any.as_ptr(),
                any.len() as i64,
                empty.as_ptr(),
                empty.len() as i64,
            ));
        }
    }

    #[test]
    fn str_endswith_returns_expected_results() {
        let ctx = Context::create();
        let module = ctx.create_module("str_endswith_test");
        let str_fn = add_str_endswith(&ctx, &module);

        let bool_type = ctx.bool_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();

        let wrapper = module.add_function(
            "endswith_wrapper",
            bool_type.fn_type(
                &[
                    ptr_type.into(),
                    i64_type.into(),
                    ptr_type.into(),
                    i64_type.into(),
                ],
                false,
            ),
            None,
        );

        let entry = ctx.append_basic_block(wrapper, "entry");
        let builder = ctx.create_builder();
        builder.position_at_end(entry);

        let hay_ptr = wrapper.get_nth_param(0).unwrap().into_pointer_value();
        let hay_len = wrapper.get_nth_param(1).unwrap().into_int_value();
        let needle_ptr = wrapper.get_nth_param(2).unwrap().into_pointer_value();
        let needle_len = wrapper.get_nth_param(3).unwrap().into_int_value();

        let hay_end = increment_pointer!(ctx, builder, hay_ptr, 1, hay_len);
        let needle_end = increment_pointer!(ctx, builder, needle_ptr, 1, needle_len);

        let str_type = PrimitiveType::P64x2.llvm_type(&ctx).into_struct_type();

        let hay_struct = builder
            .build_insert_value(str_type.const_zero(), hay_ptr, 0, "hay_start")
            .unwrap();
        let hay_struct = builder
            .build_insert_value(hay_struct, hay_end, 1, "hay_end")
            .unwrap()
            .into_struct_value();

        let needle_struct = builder
            .build_insert_value(str_type.const_zero(), needle_ptr, 0, "needle_start")
            .unwrap();
        let needle_struct = builder
            .build_insert_value(needle_struct, needle_end, 1, "needle_end")
            .unwrap()
            .into_struct_value();

        let call = builder
            .build_call(
                str_fn,
                &[hay_struct.into(), needle_struct.into()],
                "strncmp",
            )
            .unwrap();
        let res = call.try_as_basic_value().unwrap_left().into_int_value();
        builder.build_return(Some(&res)).unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let wrapper_fn = unsafe {
            ee.get_function::<unsafe extern "C" fn(*const u8, i64, *const u8, i64) -> bool>(
                "endswith_wrapper",
            )
            .unwrap()
        };

        unsafe {
            let hay = b"hello world" as &[u8];
            let suffix = b"world" as &[u8];
            assert!(wrapper_fn.call(
                hay.as_ptr(),
                hay.len() as i64,
                suffix.as_ptr(),
                suffix.len() as i64,
            ));

            let exact = b"hello" as &[u8];
            assert!(wrapper_fn.call(
                exact.as_ptr(),
                exact.len() as i64,
                exact.as_ptr(),
                exact.len() as i64,
            ));

            let mismatch = b"hella" as &[u8];
            assert!(!wrapper_fn.call(
                exact.as_ptr(),
                exact.len() as i64,
                mismatch.as_ptr(),
                mismatch.len() as i64,
            ));

            let short_hay = b"hi" as &[u8];
            assert!(!wrapper_fn.call(
                short_hay.as_ptr(),
                short_hay.len() as i64,
                exact.as_ptr(),
                exact.len() as i64,
            ));

            let any = b"rust" as &[u8];
            let empty = b"" as &[u8];
            assert!(wrapper_fn.call(
                any.as_ptr(),
                any.len() as i64,
                empty.as_ptr(),
                empty.len() as i64,
            ));

            let not_suffix = b"hello" as &[u8];
            assert!(!wrapper_fn.call(
                hay.as_ptr(),
                hay.len() as i64,
                not_suffix.as_ptr(),
                not_suffix.len() as i64,
            ));
        }
    }
}
