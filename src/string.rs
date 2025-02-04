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

    fn string_return_type(&self) -> StructType<'ctx> {
        self.context.struct_type(
            &[
                self.context.ptr_type(AddressSpace::default()).into(),
                self.context.ptr_type(AddressSpace::default()).into(),
            ],
            false,
        )
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

    pub(crate) fn initialize_iter_string_primitive<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        offset_ptr: PointerValue,
        data_ptr: PointerValue,
        len: IntValue,
    ) -> PointerValue<'a>
    where
        'ctx: 'a,
    {
        let i64_type = self.context.i64_type();
        let iter_type = self.struct_for_iter_string_primitive();
        let ptr = builder.build_alloca(iter_type, "prim_iter_ptr").unwrap();

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

    pub(crate) fn has_next_iter_string_primitive<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        iter: PointerValue<'a>,
    ) -> IntValue<'a>
    where
        'ctx: 'a,
    {
        let i64_type = self.context.i64_type();
        let iter_type = self.struct_for_iter_string_primitive();

        let idx_ptr = builder
            .build_struct_gep(iter_type, iter, 2, "idx_ptr")
            .unwrap();
        let idx = builder
            .build_load(i64_type, idx_ptr, "idx")
            .unwrap()
            .into_int_value();

        let len_ptr = builder
            .build_struct_gep(iter_type, iter, 3, "len_ptr")
            .unwrap();
        let len = builder
            .build_load(i64_type, len_ptr, "len")
            .unwrap()
            .into_int_value();

        builder
            .build_int_compare(IntPredicate::ULT, idx, len, "has_next")
            .unwrap()
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

        let builder = self.context.create_builder();

        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let wtype = prim_width_type.llvm_type(self.context);
        let iter_type = self.struct_for_iter_primitive();
        let ret_type = self.string_return_type();

        let fn_type = ret_type.fn_type(
            &[
                ptr_type.into(), // iter struct
            ],
            false,
        );
        let function = self.module.add_function(
            &format!("{}_primitive_iter_string", label),
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
        let idx_ptr = builder
            .build_struct_gep(iter_type, iter_ptr, 2, "idx_ptr")
            .unwrap();
        let idx = builder
            .build_load(i64_type, idx_ptr, "idx")
            .unwrap()
            .into_int_value();

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

        let inc_index = builder
            .build_int_add(idx, i64_type.const_int(1, false), "inc_index")
            .unwrap();
        builder.build_store(idx_ptr, inc_index).unwrap();

        builder.build_return(Some(&to_return)).unwrap();

        function
    }

    pub(crate) fn string_minmax(
        self,
        offset_type: PrimitiveType,
        agg: Aggregation,
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
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let str_type = self.string_return_type();

        let next = self.gen_iter_string_primitive("agg_iter", offset_type);
        let memcmp = self.add_memcmp();

        let fn_type = self.context.void_type().fn_type(
            &[
                ptr_type.into(), // data
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
        let offset_ptr = builder
            .build_load(
                ptr_type,
                function.get_nth_param(0).unwrap().into_pointer_value(),
                "offset_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let data_ptr = builder
            .build_load(
                ptr_type,
                self.increment_pointer(
                    &builder,
                    function.get_nth_param(0).unwrap().into_pointer_value(),
                    8,
                    i64_type.const_int(1, false),
                ),
                "data_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let len = function.get_nth_param(1).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(2).unwrap().into_pointer_value();

        let iter = self.initialize_iter_string_primitive(&builder, offset_ptr, data_ptr, len);
        let curr_best_ptr = builder.build_alloca(str_type, "curr_best_ptr").unwrap();
        let first = builder
            .build_call(next, &[iter.into()], "curr_best")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_struct_value();
        builder.build_store(curr_best_ptr, first).unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(loop_cond);
        let has_next = self.has_next_iter_string_primitive(&builder, iter);
        builder
            .build_conditional_branch(has_next, loop_body, end)
            .unwrap();

        builder.position_at_end(loop_body);
        let next = builder
            .build_call(next, &[iter.into()], "next")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
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
        let curr_best = builder
            .build_load(str_type, curr_best_ptr, "curr_best")
            .unwrap()
            .into_struct_value();
        builder.build_store(out_ptr, curr_best).unwrap();
        builder.build_return(None).unwrap();

        self.module.verify().unwrap();
        //self.module.print_to_stderr();
        self.optimize()?;
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        Ok(CompiledAggFunc {
            _cg: self,
            src_dt: match offset_type {
                PrimitiveType::I32 => DataType::Utf8,
                PrimitiveType::I64 => DataType::LargeUtf8,
                _ => unreachable!(),
            },
            f: unsafe { ee.get_function("agg_str").unwrap() },
        })
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, StringArray};
    use inkwell::context::Context;

    use crate::{aggregate::Aggregation, CodeGen};

    #[test]
    fn test_string_min() {
        let data = StringArray::from(vec!["this", "is", "a", "test"]);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .string_minmax(crate::PrimitiveType::I32, Aggregation::Min)
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
            .string_minmax(crate::PrimitiveType::I32, Aggregation::Max)
            .unwrap();

        let expected = StringArray::from(vec!["this"]);
        let result = f.call(&data).unwrap().unwrap();
        assert_eq!(&result, &(Box::new(expected) as Box<dyn Array>));
    }

    #[test]
    fn test_string_len_max() {
        let data = StringArray::from(vec!["®", " "]);

        println!("{:?}", data.offsets());

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .string_minmax(crate::PrimitiveType::I32, Aggregation::Max)
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
            .string_minmax(crate::PrimitiveType::I32, Aggregation::Min)
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
            .string_minmax(crate::PrimitiveType::I32, Aggregation::Max)
            .unwrap();

        let expected = StringArray::from(vec!["thiszz"]);
        let result = f.call(&data).unwrap().unwrap();
        assert_eq!(&result, &(Box::new(expected) as Box<dyn Array>));
    }
}
