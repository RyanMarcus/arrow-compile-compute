use arrow_schema::{ArrowError, DataType};
use inkwell::{
    builder::Builder,
    intrinsics::Intrinsic,
    module::Linkage,
    types::StructType,
    values::{BasicValue, FunctionValue, IntValue, PointerValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};

use crate::{aggregate::Aggregation, declare_blocks, CodeGen, CompiledAggFunc, PrimitiveType};

impl<'ctx> CodeGen<'ctx> {
    fn add_memcmp(&self) -> FunctionValue {
        let i64_type = self.context.i64_type();
        let i8_type = self.context.i8_type();
        let str_type = self.string_return_type();

        let fn_type = i64_type.fn_type(
            &[
                str_type.into(), // first string
                str_type.into(), // second string
            ],
            false,
        );

        let umin = Intrinsic::find("llvm.umin").unwrap();
        let umin_f = umin
            .get_declaration(&self.module, &[i64_type.into()])
            .unwrap();

        let builder = self.context.create_builder();
        let function = self.module.add_function("memcmp", fn_type, None);

        declare_blocks!(
            self.context,
            function,
            entry,
            for_cond,
            for_body,
            early_return,
            no_diff
        );

        builder.position_at_end(entry);
        let ptr1 = function.get_nth_param(0).unwrap().into_struct_value();
        let ptr2 = function.get_nth_param(1).unwrap().into_struct_value();

        let start_ptr1 = builder
            .build_extract_value(ptr1, 0, "start_ptr1")
            .unwrap()
            .into_pointer_value();
        let end_ptr1 = builder
            .build_extract_value(ptr1, 1, "end_ptr1")
            .unwrap()
            .into_pointer_value();
        let start_ptr2 = builder
            .build_extract_value(ptr2, 0, "start_ptr2")
            .unwrap()
            .into_pointer_value();
        let end_ptr2 = builder
            .build_extract_value(ptr2, 1, "end_ptr2")
            .unwrap()
            .into_pointer_value();

        let len1 = self.pointer_diff(&builder, start_ptr1, end_ptr1);
        let len2 = self.pointer_diff(&builder, start_ptr2, end_ptr2);

        let len = builder
            .build_call(umin_f, &[len1.into(), len2.into()], "len")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();

        let index_ptr = builder.build_alloca(i64_type, "index_ptr").unwrap();
        builder
            .build_store(index_ptr, i64_type.const_zero())
            .unwrap();

        builder.build_unconditional_branch(for_cond).unwrap();

        builder.position_at_end(for_cond);
        let index = builder
            .build_load(i64_type, index_ptr, "index")
            .unwrap()
            .as_basic_value_enum()
            .into_int_value();
        let cmp = builder
            .build_int_compare(IntPredicate::ULT, index, len, "loop_cmp")
            .unwrap();
        builder
            .build_conditional_branch(cmp, for_body, no_diff)
            .unwrap();

        builder.position_at_end(for_body);
        let index = builder
            .build_load(i64_type, index_ptr, "index")
            .unwrap()
            .into_int_value();
        let elem1_ptr = self.increment_pointer(&builder, start_ptr1, 1, index);
        let elem2_ptr = self.increment_pointer(&builder, start_ptr2, 1, index);
        let elem1 = builder
            .build_load(i8_type, elem1_ptr, "elem1")
            .unwrap()
            .into_int_value();
        let elem2 = builder
            .build_load(i8_type, elem2_ptr, "elem2")
            .unwrap()
            .into_int_value();

        let elem1 = builder
            .build_int_z_extend_or_bit_cast(elem1, i64_type, "z_elem1")
            .unwrap();
        let elem2 = builder
            .build_int_z_extend_or_bit_cast(elem2, i64_type, "z_elem2")
            .unwrap();

        let diff = builder.build_int_sub(elem1, elem2, "sub").unwrap();

        let value_cmp = builder
            .build_int_compare(IntPredicate::EQ, diff, i64_type.const_zero(), "value_cmp")
            .unwrap();

        let inc_index = builder
            .build_int_add(index, i64_type.const_int(1, false), "inc_index")
            .unwrap();
        builder.build_store(index_ptr, inc_index).unwrap();

        builder
            .build_conditional_branch(value_cmp, for_cond, early_return)
            .unwrap();

        builder.position_at_end(early_return);
        builder.build_return(Some(&diff)).unwrap();

        builder.position_at_end(no_diff);
        let eq_len = builder
            .build_int_compare(IntPredicate::EQ, len1, len2, "is_eq")
            .unwrap();
        let arr1_longer = builder
            .build_int_compare(IntPredicate::UGT, len1, len2, "arr1_longer")
            .unwrap();

        let res = builder
            .build_select(
                eq_len,
                i64_type.const_zero(),
                builder
                    .build_select(
                        arr1_longer,
                        i64_type.const_int(1, true),
                        i64_type.const_int(-1_i64 as u64, false),
                        "diff",
                    )
                    .unwrap()
                    .into_int_value(),
                "same",
            )
            .unwrap()
            .into_int_value();
        builder.build_return(Some(&res)).unwrap();

        function
    }

    /// Adds a function to the current context that takes in a start and end
    /// string pointer, along with a base buffer pointer, and returns a view.
    pub(crate) fn add_ptrx2_to_view(&self) -> FunctionValue {
        let ctx = self.context;
        let i1_type = ctx.bool_type();
        let i32_type = ctx.i32_type();
        let i64_type = ctx.i64_type();
        let i128_type = ctx.i128_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let str_type = self.string_return_type();

        let memcpy = Intrinsic::find("llvm.memcpy").unwrap();
        let memcpy_f = memcpy
            .get_declaration(
                &self.module,
                &[ptr_type.into(), ptr_type.into(), i64_type.into()],
            )
            .unwrap();

        let func_type = i128_type.fn_type(&[str_type.into(), ptr_type.into()], false);
        let func = self.module.add_function("ptrx2_to_view", func_type, None);

        let ptrs = func.get_nth_param(0).unwrap().into_struct_value();
        let base_ptr = func.get_nth_param(1).unwrap().into_pointer_value();

        declare_blocks!(self.context, func, entry, fits, no_fit, exit);
        let builder = ctx.create_builder();
        builder.position_at_end(entry);
        let ptr1 = builder
            .build_extract_value(ptrs, 0, "ptr1")
            .unwrap()
            .into_pointer_value();
        let ptr2 = builder
            .build_extract_value(ptrs, 1, "ptr2")
            .unwrap()
            .into_pointer_value();
        let to_return_ptr = builder.build_alloca(i128_type, "to_return_ptr").unwrap();
        let len_u64 = self.pointer_diff(&builder, ptr1, ptr2);
        let len = builder
            .build_int_truncate(len_u64, i32_type, "len_u32")
            .unwrap();
        let len_128 = builder
            .build_int_z_extend(len, i128_type, "len_128")
            .unwrap();
        builder.build_store(to_return_ptr, len_128).unwrap();
        let is_short = builder
            .build_int_compare(IntPredicate::ULE, len, i32_type.const_int(12, false), "cmp")
            .unwrap();
        builder
            .build_conditional_branch(is_short, fits, no_fit)
            .unwrap();

        builder.position_at_end(fits);
        builder
            .build_call(
                memcpy_f,
                &[
                    self.increment_pointer(
                        &builder,
                        to_return_ptr,
                        4,
                        i64_type.const_int(1, false),
                    )
                    .into(),
                    ptr1.into(),
                    len_u64.into(),
                    i1_type.const_zero().into(),
                ],
                "memcpy",
            )
            .unwrap();
        builder.build_unconditional_branch(exit).unwrap();

        builder.position_at_end(no_fit);
        let prefix = builder.build_load(i32_type, ptr1, "prefix").unwrap();
        builder
            .build_store(
                self.increment_pointer(&builder, to_return_ptr, 4, i64_type.const_int(1, false)),
                prefix,
            )
            .unwrap();
        let offset = self.pointer_diff(&builder, base_ptr, ptr1);
        let offset = builder
            .build_int_truncate(offset, i32_type, "offset")
            .unwrap();
        builder
            .build_store(
                self.increment_pointer(&builder, to_return_ptr, 4, i64_type.const_int(3, false)),
                offset,
            )
            .unwrap();
        builder.build_unconditional_branch(exit).unwrap();

        builder.position_at_end(exit);
        let result = builder
            .build_load(i128_type, to_return_ptr, "result")
            .unwrap();
        builder.build_return(Some(&result)).unwrap();

        func
    }

    fn string_return_type(&self) -> StructType<'ctx> {
        (PrimitiveType::P64x2)
            .llvm_type(self.context)
            .into_struct_type()
    }

    pub(crate) fn struct_for_iter_string_primitive(&self) -> StructType {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i64_type = self.context.i64_type();
        self.context.struct_type(
            &[
                ptr_type.into(), // ptr to offsets
                ptr_type.into(), // ptr to data
                i64_type.into(), // idx
                i64_type.into(), // len
            ],
            false,
        )
    }

    pub(crate) fn get_string_base_data_ptr<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        iter: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let struct_type = self.struct_for_iter_string_primitive();
        let data_ptr_ptr = builder
            .build_struct_gep(struct_type, iter, 1, "data_ptr_ptr")
            .unwrap();
        builder
            .build_load(
                self.context.ptr_type(AddressSpace::default()),
                data_ptr_ptr,
                "data_ptr",
            )
            .unwrap()
            .into_pointer_value()
    }

    pub(crate) fn initialize_iter_string_primitive<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        arr_ptr: PointerValue,
        len: IntValue,
    ) -> PointerValue<'a>
    where
        'ctx: 'a,
    {
        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let iter_type = self.struct_for_iter_string_primitive();
        let ptr = builder.build_alloca(iter_type, "prim_iter_ptr").unwrap();

        let offset_ptr = builder
            .build_load(ptr_type, arr_ptr, "offset_ptr")
            .unwrap()
            .into_pointer_value();
        let data_ptr = builder
            .build_load(
                ptr_type,
                self.increment_pointer(builder, arr_ptr, 8, i64_type.const_int(1, false)),
                "data_ptr",
            )
            .unwrap()
            .into_pointer_value();

        let off_ptr_ptr = builder
            .build_struct_gep(iter_type, ptr, 0, "off_ptr_ptr")
            .unwrap();
        builder.build_store(off_ptr_ptr, offset_ptr).unwrap();

        let dat_ptr_ptr = builder
            .build_struct_gep(iter_type, ptr, 1, "dat_ptr_ptr")
            .unwrap();
        builder.build_store(dat_ptr_ptr, data_ptr).unwrap();

        let idx_ptr = builder
            .build_struct_gep(iter_type, ptr, 2, "idx_ptr")
            .unwrap();
        builder.build_store(idx_ptr, i64_type.const_zero()).unwrap();

        let len_ptr = builder
            .build_struct_gep(iter_type, ptr, 3, "len_ptr")
            .unwrap();
        builder.build_store(len_ptr, len).unwrap();

        ptr
    }

    pub(crate) fn gen_iter_string_primitive(
        &self,
        label: &str,
        prim_width_type: PrimitiveType,
    ) -> FunctionValue {
        assert!(
            matches!(prim_width_type, PrimitiveType::I32)
                || matches!(prim_width_type, PrimitiveType::I64),
            "Only I32 and I64 widths are supported for string iterators"
        );

        let access =
            self.generate_string_random_access(&format!("{}_getter", label), prim_width_type);
        let builder = self.context.create_builder();

        let i64_type = self.context.i64_type();
        let i1_type = self.context.bool_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let iter_type = self.struct_for_iter_string_primitive();

        let fn_type = i1_type.fn_type(
            &[
                ptr_type.into(), // iter struct
                ptr_type.into(), // out
            ],
            false,
        );
        let function = self.module.add_function(
            &format!("{}_primitive_iter_string", label),
            fn_type,
            Some(Linkage::Private),
        );

        declare_blocks!(self.context, function, entry, load, exit);
        builder.position_at_end(entry);
        let iter_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let out_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let idx_ptr = builder
            .build_struct_gep(iter_type, iter_ptr, 2, "idx_ptr")
            .unwrap();
        let len_ptr = builder
            .build_struct_gep(iter_type, iter_ptr, 3, "len_ptr")
            .unwrap();
        let idx = builder
            .build_load(i64_type, idx_ptr, "idx")
            .unwrap()
            .into_int_value();
        let len = builder
            .build_load(i64_type, len_ptr, "len")
            .unwrap()
            .into_int_value();

        let cmp = builder
            .build_int_compare(IntPredicate::SGE, idx, len, "cmp")
            .unwrap();
        builder.build_conditional_branch(cmp, exit, load).unwrap();

        builder.position_at_end(load);
        let val = builder
            .build_call(access, &[iter_ptr.into(), idx.into()], "val")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_struct_value();
        builder.build_store(out_ptr, val).unwrap();

        let inc_index = builder
            .build_int_add(idx, i64_type.const_int(1, false), "inc_index")
            .unwrap();
        builder.build_store(idx_ptr, inc_index).unwrap();
        builder
            .build_return(Some(&i1_type.const_all_ones()))
            .unwrap();

        builder.position_at_end(exit);
        builder.build_return(Some(&i1_type.const_zero())).unwrap();

        function
    }

    pub(crate) fn generate_string_random_access(
        &self,
        label: &str,
        prim_width_type: PrimitiveType,
    ) -> FunctionValue {
        assert!(
            matches!(prim_width_type, PrimitiveType::I32)
                || matches!(prim_width_type, PrimitiveType::I64),
            "Only I32 and I64 widths are supported for string iterators (not {:?})",
            prim_width_type
        );

        let builder = self.context.create_builder();

        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let wtype = prim_width_type.llvm_type(self.context);
        let iter_type = self.struct_for_iter_primitive();
        let ret_type = self.string_return_type();

        let fn_type = ret_type.fn_type(
            &[
                ptr_type.into(), // iter struct
                i64_type.into(), // index
            ],
            false,
        );
        let function = self.module.add_function(
            &format!("{}_primitive_access_string", label),
            fn_type,
            Some(Linkage::Private),
        );

        let entry = self.context.append_basic_block(function, "entry");
        builder.position_at_end(entry);
        let iter_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let off_ptr_ptr = builder
            .build_struct_gep(iter_type, iter_ptr, 0, "arr_ptr_ptr")
            .unwrap();
        let off_ptr = builder
            .build_load(ptr_type, off_ptr_ptr, "arr_ptr")
            .unwrap()
            .into_pointer_value();
        let data_ptr_ptr = builder
            .build_struct_gep(iter_type, iter_ptr, 1, "data_ptr_ptr")
            .unwrap();
        let data_ptr = builder
            .build_load(ptr_type, data_ptr_ptr, "data_ptr")
            .unwrap()
            .into_pointer_value();
        let idx = function.get_nth_param(1).unwrap().into_int_value();

        let off1 = builder
            .build_load(
                wtype,
                self.increment_pointer(&builder, off_ptr, prim_width_type.width(), idx),
                "off1",
            )
            .unwrap()
            .into_int_value();

        let off2 = builder
            .build_load(
                wtype,
                self.increment_pointer(
                    &builder,
                    off_ptr,
                    prim_width_type.width(),
                    builder
                        .build_int_add(idx, i64_type.const_int(1, false), "inc_idx")
                        .unwrap(),
                ),
                "off2",
            )
            .unwrap()
            .into_int_value();

        let (off1, off2) = if matches!(prim_width_type, PrimitiveType::I32) {
            (
                builder
                    .build_int_z_extend(off1, i64_type, "off1_ext")
                    .unwrap(),
                builder
                    .build_int_z_extend(off2, i64_type, "off2_ext")
                    .unwrap(),
            )
        } else {
            (off1, off2)
        };

        let ptr1 = self.increment_pointer(&builder, data_ptr, 1, off1);
        let ptr2 = self.increment_pointer(&builder, data_ptr, 1, off2);

        let to_return = ret_type.const_zero();
        let to_return = builder
            .build_insert_value(to_return, ptr1, 0, "to_return")
            .unwrap();
        let to_return = builder
            .build_insert_value(to_return, ptr2, 1, "to_return")
            .unwrap();

        builder.build_return(Some(&to_return)).unwrap();

        function
    }

    pub(crate) fn string_minmax(
        self,
        dt: &DataType,
        agg: Aggregation,
        nullable: bool,
    ) -> Result<CompiledAggFunc<'ctx>, ArrowError> {
        let is_min = match agg {
            Aggregation::Min => true,
            Aggregation::Max => false,
            Aggregation::Sum => {
                return Err(ArrowError::ComputeError(
                    "cannot compute sum of strings".to_string(),
                ))
            }
        };

        let builder = self.context.create_builder();
        let i64_type = self.context.i64_type();
        let i1_type = self.context.bool_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let str_type = self.string_return_type();
        // Create a function called `next` that handles both the nullable and
        // non-nullable case. Each call to next will produce a new string value.
        // Note that the branch between nullable and non-nullable does not
        // appear in the compiled code.
        let next = {
            let next_f = self.module.add_function(
                "get_next",
                i1_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false),
                Some(Linkage::Private),
            );
            declare_blocks!(self.context, next_f, entry);
            let str_iter_ptr = next_f.get_nth_param(0).unwrap().into_pointer_value();
            let bit_iter_ptr = next_f.get_nth_param(1).unwrap().into_pointer_value();
            let out_ptr = next_f.get_nth_param(2).unwrap().into_pointer_value();

            builder.position_at_end(entry);
            if nullable {
                let string_getter = self.gen_random_access_for("string_getter", dt);
                declare_blocks!(self.context, next_f, load_str, exit);
                let next_bit = self.gen_iter_bitmap("null_map");

                let idx_ptr = builder.build_alloca(i64_type, "idx_ptr").unwrap();
                let had_next = builder
                    .build_call(next_bit, &[bit_iter_ptr.into(), idx_ptr.into()], "had_next")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                builder
                    .build_conditional_branch(had_next, load_str, exit)
                    .unwrap();

                builder.position_at_end(load_str);
                let idx = builder.build_load(i64_type, idx_ptr, "idx").unwrap();
                let res = builder
                    .build_call(string_getter, &[str_iter_ptr.into(), idx.into()], "string")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left();
                builder.build_store(out_ptr, res).unwrap();
                builder
                    .build_return(Some(&i1_type.const_all_ones()))
                    .unwrap();

                builder.position_at_end(exit);
                builder.build_return(Some(&i1_type.const_zero())).unwrap();
            } else {
                let next_str = self.gen_single_iter_for("next_string", dt);
                let res = builder
                    .build_call(next_str, &[str_iter_ptr.into(), out_ptr.into()], "string")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left();
                builder.build_return(Some(&res)).unwrap();
            }

            next_f
        };

        let memcmp = self.add_memcmp();

        let fn_type = i1_type.fn_type(
            &[
                ptr_type.into(), // data
                ptr_type.into(), // null map
                i64_type.into(), // len
                ptr_type.into(), // output
            ],
            false,
        );
        let function = self.module.add_function("agg_str", fn_type, None);

        declare_blocks!(
            self.context,
            function,
            entry,     // start
            loop_cond, // see if we have more
            loop_body, // compare to current best
            end        // return
        );

        builder.position_at_end(entry);

        let null_map_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let len = function.get_nth_param(2).unwrap().into_int_value();
        let curr_best_ptr = function.get_nth_param(3).unwrap().into_pointer_value();

        let iter = self.initialize_iter_string_primitive(
            &builder,
            function.get_nth_param(0).unwrap().into_pointer_value(),
            len,
        );
        let nulls = self.initialize_iter_bitmap(&builder, null_map_ptr, len);

        let candidate_ptr = builder.build_alloca(str_type, "candidate_ptr").unwrap();
        let had_any = builder
            .build_call(
                next,
                &[iter.into(), nulls.into(), curr_best_ptr.into()],
                "curr_best",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        builder
            .build_conditional_branch(had_any, loop_cond, end)
            .unwrap();

        builder.position_at_end(loop_cond);
        let had_next = builder
            .build_call(
                next,
                &[iter.into(), nulls.into(), candidate_ptr.into()],
                "candidate",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        builder
            .build_conditional_branch(had_next, loop_body, end)
            .unwrap();

        builder.position_at_end(loop_body);
        let next = builder
            .build_load(str_type, candidate_ptr, "next")
            .unwrap()
            .into_struct_value();
        let curr_best = builder
            .build_load(str_type, curr_best_ptr, "curr_best")
            .unwrap()
            .into_struct_value();
        let cmp = builder
            .build_call(memcmp, &[curr_best.into(), next.into()], "cmp")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();

        let cmp = builder
            .build_int_compare(
                if is_min {
                    IntPredicate::SLT
                } else {
                    IntPredicate::SGT
                },
                cmp,
                i64_type.const_zero(),
                "cmp",
            )
            .unwrap();
        let new_val = builder
            .build_select(cmp, curr_best, next, "new_val")
            .unwrap();
        builder.build_store(curr_best_ptr, new_val).unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(end);
        let wrote_result = builder.build_phi(i1_type, "wrote_result").unwrap();
        wrote_result.add_incoming(&[
            (&i1_type.const_all_ones(), loop_cond),
            (&i1_type.const_zero(), entry),
        ]);

        builder
            .build_return(Some(&wrote_result.as_basic_value().into_int_value()))
            .unwrap();

        self.module.verify().unwrap();
        self.module.print_to_stderr();
        //        self.optimize()?;
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        Ok(CompiledAggFunc {
            _cg: self,
            nullable,
            src_dt: dt.clone(),
            f: unsafe { ee.get_function("agg_str").unwrap() },
        })
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, StringArray};
    use arrow_schema::DataType;
    use inkwell::context::Context;

    use crate::{aggregate::Aggregation, CodeGen};

    #[test]
    fn test_string_min() {
        let data = StringArray::from(vec!["this", "is", "a", "test"]);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .string_minmax(&DataType::Utf8, Aggregation::Min, false)
            .unwrap();

        let expected = StringArray::from(vec!["a"]);
        let result = f.call(&data).unwrap().unwrap();
        assert_eq!(&result, &(Box::new(expected) as Box<dyn Array>));
    }

    #[test]
    fn test_string_max() {
        let data = StringArray::from(vec!["this", "is", "a", "test"]);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .string_minmax(&DataType::Utf8, Aggregation::Max, false)
            .unwrap();

        let expected = StringArray::from(vec!["this"]);
        let result = f.call(&data).unwrap().unwrap();
        assert_eq!(&result, &(Box::new(expected) as Box<dyn Array>));
    }

    #[test]
    fn test_string_len_max() {
        let data = StringArray::from(vec!["®", " "]);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .string_minmax(&DataType::Utf8, Aggregation::Max, false)
            .unwrap();

        assert!("®" > " ");
        let expected = StringArray::from(vec!["®"]);
        let result = f.call(&data).unwrap().unwrap();
        assert_eq!(&result, &(Box::new(expected) as Box<dyn Array>));
    }

    #[test]
    fn test_string_min_eqlen() {
        let data = StringArray::from(vec!["thisa", "this", "thisa", "thisaz"]);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .string_minmax(&DataType::Utf8, Aggregation::Min, false)
            .unwrap();

        let expected = StringArray::from(vec!["this"]);
        let result = f.call(&data).unwrap().unwrap();
        assert_eq!(&result, &(Box::new(expected) as Box<dyn Array>));
    }

    #[test]
    fn test_string_max_eqlen() {
        let data = StringArray::from(vec!["thisa", "thiszz", "thisa", "thisaz"]);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .string_minmax(&DataType::Utf8, Aggregation::Max, false)
            .unwrap();

        let expected = StringArray::from(vec!["thiszz"]);
        let result = f.call(&data).unwrap().unwrap();
        assert_eq!(&result, &(Box::new(expected) as Box<dyn Array>));
    }

    #[test]
    fn test_string_max_null() {
        let data = StringArray::from(vec![Some("this"), None, Some("a"), Some("test")]);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .string_minmax(&DataType::Utf8, Aggregation::Max, true)
            .unwrap();

        let expected = StringArray::from(vec!["this"]);
        let result = f.call(&data).unwrap().unwrap();
        assert_eq!(&result, &(Box::new(expected) as Box<dyn Array>));
    }
}
