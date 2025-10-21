use inkwell::{
    context::Context,
    module::{Linkage, Module},
    types::StructType,
    values::FunctionValue,
    AddressSpace, IntPredicate,
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
        .unwrap()
        .into_int_value();

    let len_zero_block = ctx.append_basic_block(func, "len_zero");
    let len_nonzero_block = ctx.append_basic_block(func, "len_nonzero");
    let len_lt64_block = ctx.append_basic_block(func, "len_lt64");
    let len_ge64_block = ctx.append_basic_block(func, "len_ge64");
    let mod_merge_block = ctx.append_basic_block(func, "mod_merge");

    let len_is_zero = b
        .build_int_compare(
            IntPredicate::EQ,
            strlen,
            i64_type.const_zero(),
            "len_is_zero",
        )
        .unwrap();
    b.build_conditional_branch(len_is_zero, len_zero_block, len_nonzero_block)
        .unwrap();

    b.position_at_end(len_zero_block);
    b.build_unconditional_branch(mod_merge_block).unwrap();

    b.position_at_end(len_nonzero_block);
    let len_lt_64 = b
        .build_int_compare(
            IntPredicate::ULT,
            strlen,
            i64_type.const_int(64, false),
            "len_lt_64",
        )
        .unwrap();
    b.build_conditional_branch(len_lt_64, len_lt64_block, len_ge64_block)
        .unwrap();

    b.position_at_end(len_lt64_block);
    let shift_amt = b.build_int_sub(strlen, i64_one, "shift_amt").unwrap();
    let mod_value = b.build_left_shift(i64_one, shift_amt, "mod_value").unwrap();
    b.build_unconditional_branch(mod_merge_block).unwrap();

    b.position_at_end(len_ge64_block);
    b.build_unconditional_branch(mod_merge_block).unwrap();

    b.position_at_end(mod_merge_block);
    let final_mod = b.build_phi(i64_type, "final_mod").unwrap();
    final_mod.add_incoming(&[(&i64_type.const_zero(), len_zero_block)]);
    final_mod.add_incoming(&[(&mod_value, len_lt64_block)]);
    final_mod.add_incoming(&[(&i64_type.const_zero(), len_ge64_block)]);

    let final_mod = final_mod.as_basic_value().into_int_value();

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

pub fn add_str_contains<'a>(ctx: &'a Context, llvm_mod: &Module<'a>) -> FunctionValue<'a> {
    if let Some(func) = llvm_mod.get_function("str_contains") {
        return func;
    }

    let init_func = add_create_finder(ctx, llvm_mod);
    let memcmp = add_memcmp(ctx, llvm_mod);

    let i64_type = ctx.i64_type();
    let i64_one = i64_type.const_int(1, false);
    let not_found = i64_type.const_int((-1i64) as u64, true);
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let str_type = PrimitiveType::P64x2.llvm_type(ctx).into_struct_type();
    let func = llvm_mod.add_function(
        "str_contains",
        i64_type.fn_type(&[str_type.into(), str_type.into()], false),
        Some(Linkage::Private),
    );

    declare_blocks!(
        ctx,
        func,
        entry,
        initial_read,
        loop_cond,
        loop_body,
        hash_match,
        no_match,
        advance,
        exit
    );
    let b = ctx.create_builder();
    b.position_at_end(entry);
    let needle = func.get_nth_param(0).unwrap().into_struct_value();
    let haystack = func.get_nth_param(1).unwrap().into_struct_value();

    let init_data = b
        .build_call(init_func, &[needle.into()], "init")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_struct_value();
    let needle_hash = b
        .build_extract_value(init_data, 0, "needle_hash")
        .unwrap()
        .into_int_value();
    let modulo = b
        .build_extract_value(init_data, 1, "modulo")
        .unwrap()
        .into_int_value();

    let str_start = b
        .build_extract_value(haystack, 0, "str_start")
        .unwrap()
        .into_pointer_value();
    let str_end = b
        .build_extract_value(haystack, 1, "str_end")
        .unwrap()
        .into_pointer_value();
    let strlen = pointer_diff!(ctx, b, str_start, str_end);
    let needle_start = b
        .build_extract_value(needle, 0, "needle_start")
        .unwrap()
        .into_pointer_value();
    let needle_end = b
        .build_extract_value(needle, 1, "needle_end")
        .unwrap()
        .into_pointer_value();
    let needle_len = pointer_diff!(ctx, b, needle_start, needle_end);
    let curr_hash_ptr = b.build_alloca(i64_type, "curr_hash_ptr").unwrap();
    let needle_fits = b
        .build_int_compare(IntPredicate::UGE, strlen, needle_len, "needle_fits")
        .unwrap();
    b.build_conditional_branch(needle_fits, initial_read, exit)
        .unwrap();

    b.position_at_end(initial_read);
    // compute the hash for the first `needle_len` chars of the haystack
    let init_end_ptr = increment_pointer!(ctx, b, str_start, 1, needle_len);
    let prefix = b
        .build_insert_value(str_type.const_zero(), str_start, 0, "with_start")
        .unwrap();
    let prefix = b
        .build_insert_value(prefix, init_end_ptr, 1, "with_end")
        .unwrap()
        .into_struct_value();
    let init_hash = b
        .build_call(init_func, &[prefix.into()], "haystack_prefix")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_struct_value();
    let init_hash = b
        .build_extract_value(init_hash, 0, "prefix_hash")
        .unwrap()
        .into_int_value();
    b.build_store(curr_hash_ptr, init_hash).unwrap();
    b.build_unconditional_branch(loop_cond).unwrap();

    b.position_at_end(loop_cond);
    let curr_ptr = b.build_phi(ptr_type, "curr_ptr").unwrap();
    curr_ptr.add_incoming(&[(&str_start, initial_read)]);
    let remaining_len = pointer_diff!(
        ctx,
        b,
        curr_ptr.as_basic_value().into_pointer_value(),
        str_end
    );
    let can_search = b
        .build_int_compare(IntPredicate::UGE, remaining_len, needle_len, "can_search")
        .unwrap();
    b.build_conditional_branch(can_search, loop_body, exit)
        .unwrap();

    b.position_at_end(loop_body);
    let check_at = curr_ptr.as_basic_value().into_pointer_value();
    let curr_hash = b
        .build_load(i64_type, curr_hash_ptr, "curr_hash")
        .unwrap()
        .into_int_value();
    let hash_matches = b
        .build_int_compare(IntPredicate::EQ, curr_hash, needle_hash, "hash_matches")
        .unwrap();
    b.build_conditional_branch(hash_matches, hash_match, no_match)
        .unwrap();

    b.position_at_end(hash_match);
    let potential_match = b
        .build_insert_value(str_type.const_zero(), check_at, 0, "with_start")
        .unwrap();
    let potential_match = b
        .build_insert_value(
            potential_match,
            increment_pointer!(ctx, b, check_at, 1, needle_len),
            1,
            "with_end",
        )
        .unwrap()
        .into_struct_value();
    let memcmp_result = b
        .build_call(
            memcmp,
            &[potential_match.into(), needle.into()],
            "memcmp_result",
        )
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    let bytes_equal = b
        .build_int_compare(
            IntPredicate::EQ,
            memcmp_result,
            i64_type.const_zero(),
            "memcmp_result_eq",
        )
        .unwrap();
    let match_index = pointer_diff!(ctx, b, str_start, check_at);
    b.build_conditional_branch(bytes_equal, exit, no_match)
        .unwrap();

    b.position_at_end(no_match);
    let next_ptr = increment_pointer!(ctx, b, check_at, 1);
    let remaining_after = pointer_diff!(ctx, b, next_ptr, str_end);
    let has_next = b
        .build_int_compare(IntPredicate::UGE, remaining_after, needle_len, "has_next")
        .unwrap();
    b.build_conditional_branch(has_next, advance, exit).unwrap();

    b.position_at_end(advance);
    let curr_hash = b
        .build_load(i64_type, curr_hash_ptr, "curr_hash_next")
        .unwrap()
        .into_int_value();
    let first_char = b
        .build_load(ctx.i8_type(), check_at, "first_char")
        .unwrap()
        .into_int_value();
    let first_char = b
        .build_int_z_extend(first_char, i64_type, "first_char_i64")
        .unwrap();
    let remove = b.build_int_mul(first_char, modulo, "remove").unwrap();
    let after_drop = b.build_int_sub(curr_hash, remove, "after_drop").unwrap();
    let shifted = b.build_left_shift(after_drop, i64_one, "shifted").unwrap();
    let next_char_ptr = increment_pointer!(ctx, b, check_at, 1, needle_len);
    let next_char = b
        .build_load(ctx.i8_type(), next_char_ptr, "next_char")
        .unwrap()
        .into_int_value();
    let next_char = b
        .build_int_z_extend(next_char, i64_type, "next_char_i64")
        .unwrap();
    let next_hash = b.build_int_add(shifted, next_char, "next_hash").unwrap();
    b.build_store(curr_hash_ptr, next_hash).unwrap();
    curr_ptr.add_incoming(&[(&next_ptr, advance)]);
    b.build_unconditional_branch(loop_cond).unwrap();

    b.position_at_end(exit);
    let result = b.build_phi(i64_type, "result").unwrap();
    result.add_incoming(&[(&not_found, entry)]);
    result.add_incoming(&[(&not_found, loop_cond)]);
    result.add_incoming(&[(&match_index, hash_match)]);
    result.add_incoming(&[(&not_found, no_match)]);
    b.build_return(Some(&result.as_basic_value().into_int_value()))
        .unwrap();

    func
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::{
        context::Context, module::Module, values::FunctionValue, AddressSpace, OptimizationLevel,
    };

    fn build_contains_wrapper<'ctx>(
        ctx: &'ctx Context,
        module: &Module<'ctx>,
        str_fn: FunctionValue<'ctx>,
        name: &str,
    ) -> FunctionValue<'ctx> {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();

        let wrapper = module.add_function(
            name,
            i64_type.fn_type(
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

        let needle_ptr = wrapper.get_nth_param(0).unwrap().into_pointer_value();
        let needle_len = wrapper.get_nth_param(1).unwrap().into_int_value();
        let hay_ptr = wrapper.get_nth_param(2).unwrap().into_pointer_value();
        let hay_len = wrapper.get_nth_param(3).unwrap().into_int_value();

        let needle_end = increment_pointer!(ctx, builder, needle_ptr, 1, needle_len);
        let hay_end = increment_pointer!(ctx, builder, hay_ptr, 1, hay_len);

        let str_type = PrimitiveType::P64x2.llvm_type(&ctx).into_struct_type();

        let needle_struct = builder
            .build_insert_value(str_type.const_zero(), needle_ptr, 0, "needle_start")
            .unwrap();
        let needle_struct = builder
            .build_insert_value(needle_struct, needle_end, 1, "needle_end")
            .unwrap()
            .into_struct_value();

        let hay_struct = builder
            .build_insert_value(str_type.const_zero(), hay_ptr, 0, "hay_start")
            .unwrap();
        let hay_struct = builder
            .build_insert_value(hay_struct, hay_end, 1, "hay_end")
            .unwrap()
            .into_struct_value();

        let call = builder
            .build_call(
                str_fn,
                &[needle_struct.into(), hay_struct.into()],
                "contains",
            )
            .unwrap();
        let res = call.try_as_basic_value().unwrap_left().into_int_value();
        builder.build_return(Some(&res)).unwrap();

        wrapper
    }

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

            let expected_hash = needle
                .iter()
                .fold(0i64, |acc, byte| (acc << 1).wrapping_add(*byte as i64));
            let shift = ((needle.len() as u64).wrapping_sub(1)) & 63;
            let expected_modulo = if needle.is_empty() {
                0
            } else if needle.len() < 64 {
                1i64 << ((needle.len() - 1) as u32)
            } else {
                0
            };

            assert_eq!(hash_out, expected_hash);
            assert_eq!(modulo_out, expected_modulo);

            let mut long_needle = vec![b'a'; 80];
            long_needle[0] = b'z';
            let mut long_hash_out = 0i64;
            let mut long_modulo_out = 0i64;
            wrapper_fn.call(
                long_needle.as_ptr(),
                long_needle.len() as i64,
                &mut long_hash_out,
                &mut long_modulo_out,
            );

            let expected_long_hash = long_needle.iter().fold(0i64, |acc, byte| {
                acc.wrapping_shl(1).wrapping_add(*byte as i64)
            });

            assert_eq!(long_modulo_out, 0);
            assert_eq!(long_hash_out, expected_long_hash);
        }
    }

    #[test]
    fn str_contains_returns_expected_positions() {
        let ctx = Context::create();
        let module = ctx.create_module("str_contains_test");
        let str_fn = add_str_contains(&ctx, &module);

        build_contains_wrapper(&ctx, &module, str_fn, "contains_wrapper");

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let wrapper_fn = unsafe {
            ee.get_function::<unsafe extern "C" fn(*const u8, i64, *const u8, i64) -> i64>(
                "contains_wrapper",
            )
            .unwrap()
        };

        unsafe {
            let hay = b"hello world" as &[u8];

            let needle = b"hello" as &[u8];
            assert_eq!(
                wrapper_fn.call(
                    needle.as_ptr(),
                    needle.len() as i64,
                    hay.as_ptr(),
                    hay.len() as i64,
                ),
                0,
            );

            let needle = b"world" as &[u8];
            assert_eq!(
                wrapper_fn.call(
                    needle.as_ptr(),
                    needle.len() as i64,
                    hay.as_ptr(),
                    hay.len() as i64,
                ),
                6,
            );

            let needle = b"ld" as &[u8];
            assert_eq!(
                wrapper_fn.call(
                    needle.as_ptr(),
                    needle.len() as i64,
                    hay.as_ptr(),
                    hay.len() as i64,
                ),
                9,
            );

            let missing = b"nope" as &[u8];
            assert_eq!(
                wrapper_fn.call(
                    missing.as_ptr(),
                    missing.len() as i64,
                    hay.as_ptr(),
                    hay.len() as i64,
                ),
                -1,
            );

            let empty = b"" as &[u8];
            assert_eq!(
                wrapper_fn.call(
                    empty.as_ptr(),
                    empty.len() as i64,
                    hay.as_ptr(),
                    hay.len() as i64,
                ),
                0,
            );

            let short_hay = b"hi" as &[u8];
            let long_needle = b"longer" as &[u8];
            assert_eq!(
                wrapper_fn.call(
                    long_needle.as_ptr(),
                    long_needle.len() as i64,
                    short_hay.as_ptr(),
                    short_hay.len() as i64,
                ),
                -1,
            );

            let hay = b"bananas" as &[u8];
            let needle = b"ana" as &[u8];
            assert_eq!(
                wrapper_fn.call(
                    needle.as_ptr(),
                    needle.len() as i64,
                    hay.as_ptr(),
                    hay.len() as i64,
                ),
                1,
            );
        }
    }

    #[test]
    fn str_contains_handles_long_patterns() {
        let ctx = Context::create();
        let module = ctx.create_module("str_contains_long_test");
        let str_fn = add_str_contains(&ctx, &module);

        build_contains_wrapper(&ctx, &module, str_fn, "contains_wrapper");

        module.print_to_stderr();
        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let wrapper_fn = unsafe {
            ee.get_function::<unsafe extern "C" fn(*const u8, i64, *const u8, i64) -> i64>(
                "contains_wrapper",
            )
            .unwrap()
        };

        unsafe {
            let prefix = b"prefix-without-pattern-";
            let mut hay = prefix.to_vec();
            let repeating: Vec<u8> = (0..140).map(|i| b'a' + (i % 26) as u8).collect();
            hay.extend_from_slice(&repeating);

            let needle_offset = 45usize;
            let needle_len = 70usize;
            let needle = repeating[needle_offset..needle_offset + needle_len].to_vec();
            let expected = hay
                .windows(needle.len())
                .position(|window| window == needle)
                .unwrap() as i64;

            assert_eq!(
                wrapper_fn.call(
                    needle.as_ptr(),
                    needle.len() as i64,
                    hay.as_ptr(),
                    hay.len() as i64,
                ),
                expected,
            );

            let missing: Vec<u8> = (0..needle_len).map(|i| b'z' - ((i as u8) % 26)).collect();
            assert_eq!(
                wrapper_fn.call(
                    missing.as_ptr(),
                    missing.len() as i64,
                    hay.as_ptr(),
                    hay.len() as i64,
                ),
                -1,
            );
        }
    }
}
