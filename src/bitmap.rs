use arrow_schema::{ArrowError, DataType};
use inkwell::{
    builder::Builder,
    intrinsics::Intrinsic,
    module::Linkage,
    types::StructType,
    values::{FunctionValue, IntValue, PointerValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};

use crate::{
    debug_2x_u64, debug_flag1, debug_flag2, debug_u64b, declare_blocks, CodeGen,
    CompiledConvertFunc,
};

impl<'ctx> CodeGen<'ctx> {
    pub(crate) fn div_ceil<'a>(
        &'a self,
        builder: &'a Builder,
        v: IntValue<'a>,
        i: u64,
    ) -> IntValue<'a> {
        // divide by i, rounding up
        let denominator = self.context.i64_type().const_int(i, false);
        let div = builder
            .build_int_unsigned_div(v, denominator, "div")
            .unwrap();
        let remainder = builder
            .build_int_unsigned_rem(v, denominator, "remainder")
            .unwrap();
        let has_remainder = builder
            .build_int_compare(
                IntPredicate::NE,
                remainder,
                self.context.i64_type().const_zero(),
                "has_remainder",
            )
            .unwrap();
        builder
            .build_select(
                has_remainder,
                builder
                    .build_int_add(div, self.context.i64_type().const_int(1, false), "one")
                    .unwrap(),
                div,
                "div_ceil",
            )
            .unwrap()
            .into_int_value()
    }

    pub(crate) fn struct_for_bitmap_iter(&self) -> StructType {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i64_type = self.context.i64_type();
        self.context.struct_type(
            &[
                ptr_type.into(), // data
                i64_type.into(), // current 64-bit index
                i64_type.into(), // current chunk
                i64_type.into(), // number of chunks (length)
            ],
            false,
        )
    }

    pub(crate) fn initialize_iter_bitmap<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        arr: PointerValue,
        len: IntValue,
    ) -> PointerValue<'a>
    where
        'ctx: 'a,
    {
        let i64_type = self.context.i64_type();
        let iter_type = self.struct_for_bitmap_iter();
        let ptr = builder.build_alloca(iter_type, "bitmap_iter_ptr").unwrap();

        let arr_ptr = builder
            .build_struct_gep(iter_type, ptr, 0, "arr_ptr_ptr")
            .unwrap();
        builder.build_store(arr_ptr, arr).unwrap();

        let idx_ptr = builder
            .build_struct_gep(iter_type, ptr, 1, "idx_ptr")
            .unwrap();
        builder.build_store(idx_ptr, i64_type.const_zero()).unwrap();

        let chunk_ptr = builder
            .build_struct_gep(iter_type, ptr, 2, "chunk_ptr")
            .unwrap();
        builder
            .build_store(chunk_ptr, i64_type.const_zero())
            .unwrap();

        let len_ptr = builder
            .build_struct_gep(iter_type, ptr, 3, "len_ptr")
            .unwrap();
        builder.build_store(len_ptr, len).unwrap();

        ptr
    }

    pub(crate) fn has_next_iter_bitmap<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        iter: PointerValue<'a>,
    ) -> IntValue<'a>
    where
        'ctx: 'a,
    {
        let i64_type = self.context.i64_type();
        let iter_type = self.struct_for_bitmap_iter();

        let idx_ptr = builder
            .build_struct_gep(iter_type, iter, 1, "idx_ptr")
            .unwrap();
        let idx = builder
            .build_load(i64_type, idx_ptr, "idx")
            .unwrap()
            .into_int_value();

        let chunk_ptr = builder
            .build_struct_gep(iter_type, iter, 2, "chunk_ptr")
            .unwrap();
        let chunk = builder
            .build_load(i64_type, chunk_ptr, "chunk")
            .unwrap()
            .into_int_value();

        let len_ptr = builder
            .build_struct_gep(iter_type, iter, 3, "len_ptr")
            .unwrap();
        let len_bits = builder
            .build_load(i64_type, len_ptr, "len")
            .unwrap()
            .into_int_value();
        let len_bytes = self.div_ceil(builder, len_bits, 8);

        let before_end = builder
            .build_int_compare(IntPredicate::ULT, idx, len_bytes, "at_end")
            .unwrap();
        let chunk_nonempty = builder
            .build_int_compare(IntPredicate::NE, chunk, i64_type.const_zero(), "empty")
            .unwrap();

        builder
            .build_or(before_end, chunk_nonempty, "has_next")
            .unwrap()
    }

    pub(crate) fn gen_iter_bitmap(&self, label: &str) -> FunctionValue {
        let i64_type = self.context.i64_type();
        let i1_type = self.context.bool_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let iter_type = self.struct_for_bitmap_iter();

        let count_trailing_zeros = Intrinsic::find("llvm.cttz").unwrap();
        let count_trailing_zeros_f = count_trailing_zeros
            .get_declaration(&self.module, &[i64_type.into()])
            .unwrap();

        let memcpy = Intrinsic::find("llvm.memcpy").unwrap();
        let memcpy_f = memcpy
            .get_declaration(
                &self.module,
                &[ptr_type.into(), ptr_type.into(), i64_type.into()],
            )
            .unwrap();

        let fn_type = i64_type.fn_type(
            &[
                ptr_type.into(), // iter struct
            ],
            false,
        );
        let function = self.module.add_function(
            &format!("{}_bitmap_iter", label),
            fn_type,
            Some(Linkage::Private),
        );

        let builder = self.context.create_builder();
        declare_blocks!(
            self.context,
            function,
            entry,          // start
            check_curr,     // check if there are values in the current chunk
            use_curr_chunk, // return a value from the current chunk
            fetch_next,     // load another chunk, either a full chunk or a partial chunk
            fetch_full,     // fetch a full chunk
            fetch_partial   // fetch a partial (tail) chunk
        );

        builder.position_at_end(entry);
        let iter_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let data_ptr = builder
            .build_load(
                ptr_type,
                builder
                    .build_struct_gep(iter_type, iter_ptr, 0, "data_ptr_ptr")
                    .unwrap(),
                "data_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let curr_idx_ptr = builder
            .build_struct_gep(iter_type, iter_ptr, 1, "curr_idx_ptr")
            .unwrap();
        let curr_chunk_ptr = builder
            .build_struct_gep(iter_type, iter_ptr, 2, "chunk_ptr")
            .unwrap();
        let len = builder
            .build_load(
                i64_type,
                builder
                    .build_struct_gep(iter_type, iter_ptr, 3, "len_ptr")
                    .unwrap(),
                "len",
            )
            .unwrap()
            .into_int_value();
        builder.build_unconditional_branch(check_curr).unwrap();

        builder.position_at_end(check_curr);
        let curr_chunk = builder
            .build_load(i64_type, curr_chunk_ptr, "chunk")
            .unwrap()
            .into_int_value();
        let have_more = builder
            .build_int_compare(
                IntPredicate::NE,
                curr_chunk,
                i64_type.const_zero(),
                "have_more",
            )
            .unwrap();
        builder
            .build_conditional_branch(have_more, use_curr_chunk, fetch_next)
            .unwrap();

        builder.position_at_end(fetch_next);
        // check to see if there are at least 8 bytes left
        let curr_idx = builder
            .build_load(i64_type, curr_idx_ptr, "curr_idx")
            .unwrap()
            .into_int_value();
        let curr_bit_idx = builder
            .build_int_mul(curr_idx, i64_type.const_int(8, false), "curr_bit_idx")
            .unwrap();
        let remaining_bits = builder
            .build_int_sub(len, curr_bit_idx, "remaining")
            .unwrap();

        let full_chunk_left = builder
            .build_int_compare(
                IntPredicate::UGE,
                remaining_bits,
                i64_type.const_int(64, false),
                "has_full",
            )
            .unwrap();
        let next_chunk_ptr = self.increment_pointer(&builder, data_ptr, 1, curr_idx);
        builder
            .build_conditional_branch(full_chunk_left, fetch_full, fetch_partial)
            .unwrap();

        builder.position_at_end(fetch_full);
        let next_chunk = builder
            .build_load(i64_type, next_chunk_ptr, "next_chunk")
            .unwrap();
        builder.build_store(curr_chunk_ptr, next_chunk).unwrap();
        let next_idx = builder
            .build_int_add(curr_idx, i64_type.const_int(8, false), "new_idx")
            .unwrap();
        builder.build_store(curr_idx_ptr, next_idx).unwrap();
        builder.build_unconditional_branch(check_curr).unwrap();

        builder.position_at_end(fetch_partial);
        // divide by 8, rounding up
        let eight = i64_type.const_int(8, false);
        let div = builder
            .build_int_unsigned_div(remaining_bits, eight, "div")
            .unwrap();
        let remainder = builder
            .build_int_unsigned_rem(remaining_bits, eight, "remainder")
            .unwrap();
        let has_remainder = builder
            .build_int_compare(
                IntPredicate::NE,
                remainder,
                i64_type.const_zero(),
                "has_remainder",
            )
            .unwrap();
        let remaining_bytes = builder
            .build_select(
                has_remainder,
                builder
                    .build_int_add(div, i64_type.const_int(1, false), "one")
                    .unwrap(),
                div,
                "remaining_bytes",
            )
            .unwrap()
            .into_int_value();

        builder
            .build_call(
                memcpy_f,
                &[
                    curr_chunk_ptr.into(),
                    next_chunk_ptr.into(),
                    remaining_bytes.into(),
                    i1_type.const_zero().into(),
                ],
                "memcpy",
            )
            .unwrap();
        // we assume that the curr_idx is 8 past the current chunk
        let next_idx = builder
            .build_int_add(curr_idx, i64_type.const_int(8, false), "new_idx")
            .unwrap();
        builder.build_store(curr_idx_ptr, next_idx).unwrap();
        builder.build_unconditional_branch(check_curr).unwrap();

        builder.position_at_end(use_curr_chunk);
        let tmp = builder
            .build_and(
                curr_chunk,
                builder.build_int_neg(curr_chunk, "neg").unwrap(),
                "tmp",
            )
            .unwrap();

        let trailing_zeros = builder
            .build_call(
                count_trailing_zeros_f,
                &[curr_chunk.into(), i1_type.const_zero().into()],
                "tz",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        let curr_chunk_idx = builder
            .build_int_sub(
                builder
                    .build_load(i64_type, curr_idx_ptr, "curr_idx")
                    .unwrap()
                    .into_int_value(),
                i64_type.const_int(8, false),
                "sub8",
            )
            .unwrap();

        let res = builder
            .build_int_add(
                builder
                    .build_int_mul(curr_chunk_idx, i64_type.const_int(8, false), "base")
                    .unwrap(),
                trailing_zeros,
                "res",
            )
            .unwrap();
        let new_chunk = builder.build_xor(curr_chunk, tmp, "new_chunk").unwrap();
        builder.build_store(curr_chunk_ptr, new_chunk).unwrap();
        builder.build_return(Some(&res)).unwrap();
        function
    }

    pub fn compile_bitmap_to_vec(self) -> Result<CompiledConvertFunc<'ctx>, ArrowError> {
        let builder = self.context.create_builder();
        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());

        let fn_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into(), ptr_type.into()], false);
        let function = self.module.add_function("bitmap_to_vec", fn_type, None);

        let next = self.gen_iter_bitmap("bitmap_iter");

        declare_blocks!(self.context, function, entry, loop_cond, loop_body, exit);

        builder.position_at_end(entry);
        let data_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let len = function.get_nth_param(1).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
        let iter = self.initialize_iter_bitmap(&builder, data_ptr, len);
        let curr_out_idx_ptr = builder.build_alloca(i64_type, "curr_out_idx_ptr").unwrap();
        builder
            .build_store(curr_out_idx_ptr, i64_type.const_zero())
            .unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(loop_cond);
        let has_next = self.has_next_iter_bitmap(&builder, iter);
        builder
            .build_conditional_branch(has_next, loop_body, exit)
            .unwrap();

        builder.position_at_end(loop_body);
        let val_to_add = builder
            .build_call(next, &[iter.into()], "val_to_add")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        let curr_out_idx = builder
            .build_load(i64_type, curr_out_idx_ptr, "curr_out_idx")
            .unwrap()
            .into_int_value();

        let curr_out_ptr = self.increment_pointer(&builder, out_ptr, 8, curr_out_idx);
        builder.build_store(curr_out_ptr, val_to_add).unwrap();
        let next_out_idx = builder
            .build_int_add(curr_out_idx, i64_type.const_int(1, false), "next_out_idx")
            .unwrap();
        builder.build_store(curr_out_idx_ptr, next_out_idx).unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(exit);
        let curr_out_idx = builder
            .build_load(i64_type, curr_out_idx_ptr, "curr_out_idx")
            .unwrap()
            .into_int_value();
        builder.build_return(Some(&curr_out_idx)).unwrap();

        self.module.verify().unwrap();
        //self.optimize()?;
        self.module.print_to_stderr();
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        ee.add_global_mapping(&self.debug_2x_u64, debug_2x_u64 as usize);
        ee.add_global_mapping(&self.debug_u64b, debug_u64b as usize);
        ee.add_global_mapping(&self.debug_flag1, debug_flag1 as usize);
        ee.add_global_mapping(&self.debug_flag2, debug_flag2 as usize);

        Ok(CompiledConvertFunc {
            _cg: self,
            src_dt: DataType::Boolean,
            tar_dt: DataType::UInt64,
            f: unsafe { ee.get_function("bitmap_to_vec").unwrap() },
        })
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{cast::AsArray, BooleanArray, UInt64Array};
    use inkwell::context::Context;
    use itertools::Itertools;

    use crate::CodeGen;

    #[test]
    fn test_bitmap_to_vec() {
        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);

        let mut rng = fastrand::Rng::with_seed(42);
        let values: Vec<bool> = (0..1000).map(|_| rng.bool()).collect_vec();
        let array = BooleanArray::from(values.clone());

        let cf = codegen.compile_bitmap_to_vec().unwrap();
        let result: UInt64Array = cf.call(&array).unwrap().as_primitive().clone();

        // Verify the results
        let true_indexes = values
            .iter()
            .enumerate()
            .filter_map(|(idx, val)| val.then(|| idx as u64))
            .collect_vec();
        let our_indexes = result.values().to_vec();
        assert_eq!(true_indexes, our_indexes);
    }
}
