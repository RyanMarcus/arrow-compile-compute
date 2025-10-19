use inkwell::{
    context::Context,
    intrinsics::Intrinsic,
    module::{Linkage, Module},
    types::StructType,
    values::FunctionValue,
    IntPredicate,
};

use crate::{
    compiled_kernels::cmp::add_memcmp, declare_blocks, increment_pointer, pointer_diff,
    PrimitiveType,
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

fn finder_struct_type(ctx: &Context) -> StructType {
    let finder_type = ctx.struct_type(&[ctx.i64_type().into(), ctx.i64_type().into()], false);
    finder_type
}

pub fn add_create_finder<'a>(ctx: &'a Context, llvm_mod: &Module<'a>) -> FunctionValue<'a> {
    if let Some(func) = llvm_mod.get_function("create_str_finder") {
        return func;
    }

    let str_type = PrimitiveType::P64x2.llvm_type(&ctx).into_struct_type();
    let finder_type = finder_struct_type(ctx);
    let i64_type = ctx.i64_type();
    let i64_one = i64_type.const_int(1, false);

    let rotate_left = Intrinsic::find("llvm.fshl")
        .unwrap()
        .get_declaration(llvm_mod, &[i64_type.into()])
        .unwrap();

    let func = llvm_mod.add_function(
        "create_str_finder",
        finder_type.fn_type(&[str_type.into()], false),
        Some(Linkage::Private),
    );

    declare_blocks!(ctx, func, entry, loop_cond, loop_body, exit);
    let b = ctx.create_builder();
    b.position_at_end(entry);
    let needle_hash_ptr = b.build_alloca(i64_type, "needle_hash_ptr").unwrap();
    b.build_store(needle_hash_ptr, i64_type.const_zero())
        .unwrap();

    let str = func.get_nth_param(0).unwrap().into_struct_value();
    let str_start_ptr = b
        .build_extract_value(str, 0, "str_start")
        .unwrap()
        .into_pointer_value();
    let str_end_ptr = b
        .build_extract_value(str, 1, "str_end")
        .unwrap()
        .into_pointer_value();
    let strlen = pointer_diff!(ctx, b, str_start_ptr, str_end_ptr);
    b.build_unconditional_branch(loop_cond).unwrap();

    b.position_at_end(loop_cond);
    let i = b.build_phi(i64_type, "loop_idx").unwrap();
    i.add_incoming(&[(&i64_type.const_zero(), entry)]);
    let cmp = b
        .build_int_compare(
            IntPredicate::ULT,
            i.as_basic_value().into_int_value(),
            strlen,
            "cmp",
        )
        .unwrap();
    b.build_conditional_branch(cmp, loop_body, exit).unwrap();

    b.position_at_end(loop_body);
    let curr_idx = i.as_basic_value().into_int_value();
    let curr_hash = b
        .build_load(i64_type, needle_hash_ptr, "curr_hash")
        .unwrap()
        .into_int_value();
    let curr_char = b
        .build_load(
            ctx.i8_type(),
            increment_pointer!(ctx, b, str_start_ptr, 1, curr_idx),
            "curr_char",
        )
        .unwrap()
        .into_int_value();
    let curr_char = b
        .build_int_z_extend(curr_char, i64_type, "zext_char")
        .unwrap();

    let next_hash = b
        .build_int_add(
            b.build_left_shift(curr_hash, i64_one, "times_2").unwrap(),
            curr_char,
            "next_hash",
        )
        .unwrap();
    b.build_store(needle_hash_ptr, next_hash).unwrap();

    let i_next = b
        .build_int_add(i.as_basic_value().into_int_value(), i64_one, "i_next")
        .unwrap();
    i.add_incoming(&[(&i_next, loop_body)]);
    b.build_unconditional_branch(loop_cond).unwrap();

    b.position_at_end(exit);
    let final_hash = b
        .build_load(i64_type, needle_hash_ptr, "final_hash")
        .unwrap();
    let final_mod = b
        .build_call(
            rotate_left,
            &[
                i64_one.into(),
                i64_one.into(),
                b.build_int_sub(strlen, i64_one, "m1").unwrap().into(),
            ],
            "coef",
        )
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();

    let to_return = finder_type.const_zero();
    let to_return = b
        .build_insert_value(to_return, final_hash, 0, "with_hash")
        .unwrap();
    let to_return = b
        .build_insert_value(to_return, final_mod, 1, "with_mod")
        .unwrap();

    b.build_return(Some(&to_return)).unwrap();
    func
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

    #[test]
    fn create_finder_returns_expected_hash_and_modulo() {
        let ctx = Context::create();
        let module = ctx.create_module("create_finder_test");
        let finder_fn = add_create_finder(&ctx, &module);

        let void_type = ctx.void_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();

        let wrapper = module.add_function(
            "create_finder_wrapper",
            void_type.fn_type(
                &[
                    ptr_type.into(),
                    i64_type.into(),
                    ptr_type.into(),
                    ptr_type.into(),
                ],
                false,
            ),
            None,
        );

        let entry = ctx.append_basic_block(wrapper, "entry");
        let builder = ctx.create_builder();
        builder.position_at_end(entry);

        let str_ptr = wrapper.get_nth_param(0).unwrap().into_pointer_value();
        let str_len = wrapper.get_nth_param(1).unwrap().into_int_value();
        let hash_out_ptr = wrapper.get_nth_param(2).unwrap().into_pointer_value();
        let modulo_out_ptr = wrapper.get_nth_param(3).unwrap().into_pointer_value();

        let str_end_ptr = increment_pointer!(ctx, builder, str_ptr, 1, str_len);
        let str_type = PrimitiveType::P64x2.llvm_type(&ctx).into_struct_type();
        let str_struct = builder
            .build_insert_value(str_type.const_zero(), str_ptr, 0, "needle_start")
            .unwrap();
        let str_struct = builder
            .build_insert_value(str_struct, str_end_ptr, 1, "needle_end")
            .unwrap()
            .into_struct_value();

        let finder = builder
            .build_call(finder_fn, &[str_struct.into()], "finder")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_struct_value();

        let hash_value = builder
            .build_extract_value(finder, 0, "hash")
            .unwrap()
            .into_int_value();
        let modulo_value = builder
            .build_extract_value(finder, 1, "modulo")
            .unwrap()
            .into_int_value();
        builder.build_store(hash_out_ptr, hash_value).unwrap();
        builder.build_store(modulo_out_ptr, modulo_value).unwrap();
        builder.build_return(None).unwrap();

        module.print_to_stderr();
        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let wrapper_fn = unsafe {
            ee.get_function::<unsafe extern "C" fn(*const u8, i64, *mut i64, *mut i64)>(
                "create_finder_wrapper",
            )
            .unwrap()
        };

        unsafe {
            let needle = b"rust" as &[u8];
            let mut hash_out = 0i64;
            let mut modulo_out = 0i64;
            wrapper_fn.call(
                needle.as_ptr(),
                needle.len() as i64,
                &mut hash_out,
                &mut modulo_out,
            );

            let expected_hash = needle.iter().enumerate().fold(0i64, |acc, (idx, byte)| {
                acc + (*byte as i64) * 1i64.rotate_left(idx as u32)
            });
            let expected_modulo = 1i64.rotate_left(needle.len() as u32 - 1);

            assert_eq!(hash_out, expected_hash);
            assert_eq!(modulo_out, expected_modulo);

            let empty = b"" as &[u8];
            let mut empty_hash = -1i64;
            let mut empty_mod = -1i64;
            wrapper_fn.call(
                empty.as_ptr(),
                empty.len() as i64,
                &mut empty_hash,
                &mut empty_mod,
            );

            assert_eq!(empty_hash, 0);
            assert_eq!(empty_mod, 1);
        }
    }
}
