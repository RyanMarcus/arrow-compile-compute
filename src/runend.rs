use inkwell::{
    builder::Builder,
    intrinsics::Intrinsic,
    module::Linkage,
    types::StructType,
    values::{FunctionValue, IntValue, PointerValue},
    AddressSpace, IntPredicate,
};

use crate::{CodeGen, PrimitiveType};

impl<'ctx> CodeGen<'ctx> {
    pub(crate) fn struct_for_iter_re(&self) -> StructType {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i64_type = self.context.i64_type();
        self.context.struct_type(
            &[
                ptr_type.into(), // run ends
                ptr_type.into(), // values
                i64_type.into(), // curr pos
                i64_type.into(), // remaining in run
                i64_type.into(), // remaining overall
            ],
            false,
        )
    }

    pub(crate) fn initialize_iter_re<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        ptr: PointerValue<'a>,
        logical_len: IntValue,
    ) -> PointerValue<'a>
    where
        'ctx: 'a,
    {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i64_type = self.context.i64_type();
        let iter_type = self.struct_for_iter_re();

        // ptr points to a rust struct with two pointers, the run end array and
        // the value array
        let run_end_ptr = builder
            .build_load(ptr_type, ptr, "run_end_ptr")
            .unwrap()
            .into_pointer_value();
        let values_ptr = builder
            .build_load(
                ptr_type,
                self.increment_pointer(builder, ptr, 8, i64_type.const_int(1, false)),
                "run_end_ptr",
            )
            .unwrap();
        let _phys_len = builder
            .build_load(
                i64_type,
                self.increment_pointer(builder, ptr, 8, i64_type.const_int(2, false)),
                "phys_len",
            )
            .unwrap();
        let start_at = builder
            .build_load(
                ptr_type,
                self.increment_pointer(builder, ptr, 8, i64_type.const_int(3, false)),
                "offset",
            )
            .unwrap();
        let remaining = builder
            .build_load(
                ptr_type,
                self.increment_pointer(builder, ptr, 8, i64_type.const_int(4, false)),
                "offset",
            )
            .unwrap();

        let to_return = builder.build_alloca(iter_type, "iter_ptr").unwrap();
        let struct_re_ptr_ptr = builder
            .build_struct_gep(iter_type, to_return, 0, "re_ptr_ptr")
            .unwrap();
        builder.build_store(struct_re_ptr_ptr, run_end_ptr).unwrap();

        let struct_val_ptr_ptr = builder
            .build_struct_gep(iter_type, to_return, 1, "val_ptr_ptr")
            .unwrap();
        builder.build_store(struct_val_ptr_ptr, values_ptr).unwrap();

        let curr_pos_ptr = builder
            .build_struct_gep(iter_type, to_return, 2, "curr_pos_ptr")
            .unwrap();
        builder.build_store(curr_pos_ptr, start_at).unwrap();

        let remaining_ptr = builder
            .build_struct_gep(iter_type, to_return, 3, "remaining_ptr")
            .unwrap();
        builder.build_store(remaining_ptr, remaining).unwrap();

        let len_ptr = builder
            .build_struct_gep(iter_type, to_return, 4, "len_ptr")
            .unwrap();
        builder.build_store(len_ptr, logical_len).unwrap();

        to_return
    }

    pub(crate) fn has_next_iter_re<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        iter: PointerValue<'a>,
    ) -> IntValue<'a>
    where
        'ctx: 'a,
    {
        let i64_type = self.context.i64_type();
        let iter_type = self.struct_for_iter_re();
        let overall_remaining_ptr = builder
            .build_struct_gep(iter_type, iter, 4, "len_ptr")
            .unwrap();
        let overall_remaining = builder
            .build_load(i64_type, overall_remaining_ptr, "overall_remaining")
            .unwrap()
            .into_int_value();

        builder
            .build_int_compare(
                IntPredicate::UGT,
                overall_remaining,
                i64_type.const_zero(),
                "is_overall_remaining_positive",
            )
            .unwrap()
    }

    pub(crate) fn gen_re_primitive(
        &self,
        label: &str,
        re_prim_type: PrimitiveType,
        value_prim_type: PrimitiveType,
    ) -> FunctionValue {
        let builder = self.context.create_builder();
        let bool_type = self.context.bool_type();
        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let run_end_type = re_prim_type.llvm_type(self.context);
        let chunk_type = value_prim_type.llvm_vec_type(self.context, 64).unwrap();
        let iter_type = self.struct_for_iter_re();

        let umin = Intrinsic::find("llvm.umin").unwrap();
        let umin_f = umin
            .get_declaration(&self.module, &[i64_type.into()])
            .unwrap();

        let fn_type = chunk_type.fn_type(
            &[
                ptr_type.into(), // iter struct
            ],
            false,
        );
        let function = self.module.add_function(
            &format!("{}_re_iter_chunk", label),
            fn_type,
            Some(Linkage::Private),
        );

        let entry = self.context.append_basic_block(function, "entry");
        let load_prev_run_end = self
            .context
            .append_basic_block(function, "load_prev_run_end");
        let check_has_another_run = self.context.append_basic_block(function, "check_next_run");
        let check_chunk_full = self
            .context
            .append_basic_block(function, "check_chunk_full");
        let check_current_run = self
            .context
            .append_basic_block(function, "check_current_run");
        let load_new_run = self.context.append_basic_block(function, "load_new_run");
        let fill_block = self.context.append_basic_block(function, "fill_block");
        let exit = self.context.append_basic_block(function, "exit");
        let ptr = function.get_nth_param(0).unwrap().into_pointer_value();

        //
        // Entry block: initialize variables, enter loop
        //
        builder.position_at_end(entry);
        let run_ends_ptr = builder
            .build_load(
                ptr_type,
                builder
                    .build_struct_gep(iter_type, ptr, 0, "keys_ptr_ptr")
                    .unwrap(),
                "keys_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let vals_ptr = builder
            .build_load(
                ptr_type,
                builder
                    .build_struct_gep(iter_type, ptr, 1, "vals_ptr_ptr")
                    .unwrap(),
                "vals_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let curr_pos_ptr = builder
            .build_struct_gep(iter_type, ptr, 2, "curr_pos_ptr")
            .unwrap();
        let remaining_ptr = builder
            .build_struct_gep(iter_type, ptr, 3, "remaining_ptr")
            .unwrap();
        let overall_remaining_ptr = builder
            .build_struct_gep(iter_type, ptr, 4, "len_ptr")
            .unwrap();

        let to_return = builder.build_alloca(chunk_type, "chunk").unwrap();
        builder
            .build_store(to_return, chunk_type.const_zero())
            .unwrap();
        let curr_block_idx_ptr = builder
            .build_alloca(i64_type, "curr_block_idx_ptr")
            .unwrap();
        builder
            .build_store(curr_block_idx_ptr, i64_type.const_zero())
            .unwrap();

        let prev_run_end_ptr = builder
            .build_alloca(run_end_type, "prev_run_end_ptr")
            .unwrap();
        builder
            .build_store(prev_run_end_ptr, run_end_type.const_zero())
            .unwrap();

        let curr_pos = builder
            .build_load(i64_type, curr_pos_ptr, "curr_pos")
            .unwrap()
            .into_int_value();
        let cmp = builder
            .build_int_compare(IntPredicate::EQ, curr_pos, i64_type.const_zero(), "cmp")
            .unwrap();
        builder
            .build_conditional_branch(cmp, check_chunk_full, load_prev_run_end)
            .unwrap();

        //
        // Load prev run end: subtract one from curr_pos and load the run end
        // from that position. Then, continue.
        //
        builder.position_at_end(load_prev_run_end);
        let idx = builder
            .build_int_sub(curr_pos, i64_type.const_int(1, false), "idx")
            .unwrap();
        let prev_run_end = builder
            .build_load(
                run_end_type,
                self.increment_pointer(&builder, run_ends_ptr, re_prim_type.width(), idx),
                "prev_run_end",
            )
            .unwrap()
            .into_int_value();
        builder.build_store(prev_run_end_ptr, prev_run_end).unwrap();
        builder
            .build_unconditional_branch(check_chunk_full)
            .unwrap();

        //
        // Check chunk full: see if the current chunk is full, if so, return. If
        // not, check to see if the current run is exhausted.
        //
        builder.position_at_end(check_chunk_full);
        let curr_block_idx = builder
            .build_load(i64_type, curr_block_idx_ptr, "curr_block_idx")
            .unwrap()
            .into_int_value();
        let curr_block_full = builder
            .build_int_compare(
                IntPredicate::UGE,
                curr_block_idx,
                i64_type.const_int(64, false),
                "is_full",
            )
            .unwrap();
        builder
            .build_conditional_branch(curr_block_full, exit, check_current_run)
            .unwrap();

        //
        // Check current run: see if the current run has any values remaining.
        // If so, jump to fill block. If not, load a new run.
        //
        builder.position_at_end(check_current_run);
        let remaining = builder
            .build_load(i64_type, remaining_ptr, "remaining")
            .unwrap()
            .into_int_value();
        let is_run_complete = builder
            .build_int_compare(
                IntPredicate::EQ,
                remaining,
                i64_type.const_zero(),
                "is_run_complete",
            )
            .unwrap();
        builder
            .build_conditional_branch(is_run_complete, check_has_another_run, fill_block)
            .unwrap();

        //
        // Check next run: see if there is another run to load, if not, exit.
        //
        builder.position_at_end(check_has_another_run);
        let overall_remaining = builder
            .build_load(i64_type, overall_remaining_ptr, "overall_reamining")
            .unwrap()
            .into_int_value();
        let cmp = builder
            .build_int_compare(
                IntPredicate::EQ,
                overall_remaining,
                i64_type.const_zero(),
                "cmp",
            )
            .unwrap();
        builder
            .build_conditional_branch(cmp, exit, load_new_run)
            .unwrap();

        //
        // Load new run: increment our curr position, load a new run, and jump
        // to check current run (in case the run has length zero)
        //
        builder.position_at_end(load_new_run);
        let curr_pos = builder
            .build_load(i64_type, curr_pos_ptr, "curr_pos")
            .unwrap()
            .into_int_value();

        let new_run_end = builder
            .build_load(
                run_end_type,
                self.increment_pointer(&builder, run_ends_ptr, re_prim_type.width(), curr_pos),
                "new_remaining",
            )
            .unwrap()
            .into_int_value();
        let prev_run_end = builder
            .build_load(run_end_type, prev_run_end_ptr, "prev_run_end")
            .unwrap()
            .into_int_value();

        let new_remaining = builder
            .build_int_sub(new_run_end, prev_run_end, "new_remaining")
            .unwrap();
        let casted_new_remaining = builder
            .build_int_z_extend(new_remaining, i64_type, "zext")
            .unwrap();

        builder
            .build_store(remaining_ptr, casted_new_remaining)
            .unwrap();
        builder.build_store(prev_run_end_ptr, new_run_end).unwrap();

        let new_pos = builder
            .build_int_add(curr_pos, i64_type.const_int(1, false), "new_pos")
            .unwrap();
        builder.build_store(curr_pos_ptr, new_pos).unwrap();
        builder
            .build_unconditional_branch(check_current_run)
            .unwrap();

        //
        // Fill block: put up to 64 values into the current block. Jump to check
        // chunk full afterwards.
        //
        builder.position_at_end(fill_block);
        let remaining_in_run = builder
            .build_load(i64_type, remaining_ptr, "remaining")
            .unwrap()
            .into_int_value();
        let curr_block_idx = builder
            .build_load(i64_type, curr_block_idx_ptr, "curr_block_idx")
            .unwrap()
            .into_int_value();
        let remaining_empty_slots = builder
            .build_int_sub(
                i64_type.const_int(64, false),
                curr_block_idx,
                "remaining_in_chunk",
            )
            .unwrap();

        let to_fill = builder
            .build_call(
                umin_f,
                &[remaining_in_run.into(), remaining_empty_slots.into()],
                "to_fill",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();

        let curr_pos = builder
            .build_load(i64_type, curr_pos_ptr, "curr_pos")
            .unwrap()
            .into_int_value();
        // we subtract one here since we increment in the "load new run" block
        let val_idx = builder
            .build_int_sub(curr_pos, i64_type.const_int(1, false), "val_idx")
            .unwrap();

        let curr_val = builder
            .build_load(
                value_prim_type.llvm_type(self.context),
                self.increment_pointer(&builder, vals_ptr, value_prim_type.width(), val_idx),
                "value",
            )
            .unwrap()
            .into_int_value();

        // broadcast curr_val into a vector
        let value_v = builder
            .build_insert_element(
                chunk_type.const_zero(),
                curr_val,
                i64_type.const_zero(),
                "value_v",
            )
            .unwrap();
        let value_v = builder
            .build_shuffle_vector(
                value_v,
                chunk_type.get_undef(),
                chunk_type.const_zero(),
                "value_v",
            )
            .unwrap();

        // build a mask that 1 in the slots we want to insert value into
        // ex: suppose block size = 8, to_fill = 2, curr_pos = 3
        // desired mask: 00011000
        // formula: ((1 << to_fill) - 1) << curr_pos
        //
        let mask = builder
            .build_left_shift(
                builder
                    .build_int_sub(
                        builder
                            .build_left_shift(i64_type.const_int(1, false), to_fill, "mask")
                            .unwrap(),
                        i64_type.const_int(1, false),
                        "mask",
                    )
                    .unwrap(),
                curr_block_idx,
                "mask",
            )
            .unwrap();
        let max_width_cond = builder
            .build_int_compare(
                IntPredicate::EQ,
                to_fill,
                i64_type.const_int(64, false),
                "is_full",
            )
            .unwrap();
        let mask = builder
            .build_select(max_width_cond, i64_type.const_all_ones(), mask, "mask")
            .unwrap()
            .into_int_value();

        let mask = builder
            .build_bit_cast(mask, bool_type.vec_type(64), "mask_v")
            .unwrap()
            .into_vector_value();

        let curr_chunk = builder
            .build_load(chunk_type, to_return, "curr")
            .unwrap()
            .into_vector_value();
        let new_chunk = builder
            .build_select(mask, value_v, curr_chunk, "new_curr")
            .unwrap();
        builder.build_store(to_return, new_chunk).unwrap();

        let new_block_idx = builder
            .build_int_add(curr_block_idx, to_fill, "new_block_idx")
            .unwrap();
        builder
            .build_store(curr_block_idx_ptr, new_block_idx)
            .unwrap();

        let new_remaining = builder
            .build_int_sub(remaining_in_run, to_fill, "new_remaining")
            .unwrap();
        builder.build_store(remaining_ptr, new_remaining).unwrap();

        let remaining_overall = builder
            .build_load(i64_type, overall_remaining_ptr, "remaining_overall")
            .unwrap()
            .into_int_value();
        let filled = builder
            .build_call(
                umin_f,
                &[remaining_overall.into(), to_fill.into()],
                "filled",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        let new_remaining_overall = builder
            .build_int_sub(remaining_overall, filled, "new_remaining_overall")
            .unwrap();
        builder
            .build_store(overall_remaining_ptr, new_remaining_overall)
            .unwrap();

        builder
            .build_unconditional_branch(check_chunk_full)
            .unwrap();

        //
        // Exit: return the chunk
        //
        builder.position_at_end(exit);
        let chunk = builder
            .build_load(chunk_type, to_return, "chunk")
            .unwrap()
            .into_vector_value();
        builder.build_return(Some(&chunk)).unwrap();
        function
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray, types::Int64Type, Array, Int32Array, Int64Array, PrimitiveArray, RunArray,
    };
    use arrow_ord::cmp;
    use arrow_schema::{DataType, Field};
    use inkwell::context::Context;
    use itertools::Itertools;

    use crate::{run_end_data_type, test_utils::generate_random_ree_array, CodeGen, Predicate};

    #[test]
    fn test_llvm_ree_i64() {
        let mut rng = fastrand::Rng::with_seed(42);
        let f1 = Arc::new(Field::new("run_ends", DataType::Int64, false));
        let f2 = Arc::new(Field::new("values", DataType::Int64, true));
        let dt = DataType::RunEndEncoded(f1, f2);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(&dt, false, &DataType::Int64, false, Predicate::Eq)
            .unwrap();

        for num_run_ends in [0, 5, 100, 101, 128, 129, 10_000] {
            let ree_array = generate_random_ree_array(num_run_ends);
            let other = Int64Array::from(
                (0..ree_array.len() as usize)
                    .map(|_| rng.i64(-5..5))
                    .collect_vec(),
            );

            let res = f.call(&ree_array, &other).unwrap();
            let as_prim = Int64Array::from_iter(ree_array.downcast::<Int64Array>().unwrap());
            let arrow_res = cmp::eq(&as_prim, &other).unwrap();
            assert_eq!(
                res,
                arrow_res,
                "incorrect result on array of size {} (num run ends = {})",
                ree_array.len(),
                num_run_ends
            );
        }
    }

    #[test]
    fn test_simple_ree() {
        let res = Int32Array::from(vec![5, 6, 10, 11, 12]);
        let val = Int64Array::from(vec![1, 2, 3, 4, 5]);

        let ree_arr =
            RunArray::try_new(&PrimitiveArray::from(res), &PrimitiveArray::from(val)).unwrap();
        let ree_arr = ree_arr.downcast::<Int64Array>().unwrap();
        let as_prim = Int64Array::from_iter(ree_arr);

        let other = Int64Array::from(vec![1, 1, 0, 1, 1, 2, 3, 0, 3, 3, 4, 5]);
        let mask = cmp::eq(&other, &as_prim).unwrap();

        let f1 = Arc::new(Field::new("run_ends", DataType::Int32, false));
        let f2 = Arc::new(Field::new("values", DataType::Int64, true));
        let dt = DataType::RunEndEncoded(f1, f2);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(&dt, false, &DataType::Int64, false, Predicate::Eq)
            .unwrap();
        let got_mask = f.call(&ree_arr, &other).unwrap();
        assert_eq!(mask, got_mask);
    }

    #[test]
    fn test_2block_ree() {
        let res = Int32Array::from(vec![5, 50, 100]);
        let val = Int64Array::from(vec![1, 2, 3]);

        let ree_arr =
            RunArray::try_new(&PrimitiveArray::from(res), &PrimitiveArray::from(val)).unwrap();
        let ree_arr = ree_arr.downcast::<Int64Array>().unwrap();

        let as_prim = Int64Array::from_iter(ree_arr);
        let mut rng = fastrand::Rng::with_seed(42);
        let other = Int64Array::from((0..100).map(|_| rng.i64(1..3)).collect_vec());
        let mask = cmp::eq(&other, &as_prim).unwrap();

        let f1 = Arc::new(Field::new("run_ends", DataType::Int32, false));
        let f2 = Arc::new(Field::new("values", DataType::Int64, true));
        let dt = DataType::RunEndEncoded(f1, f2);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(&DataType::Int64, false, &dt, false, Predicate::Eq)
            .unwrap();
        let got_mask = f.call(&other, &ree_arr).unwrap();
        assert_eq!(mask, got_mask);
    }

    #[test]
    fn test_llvm_ree_dict_i64() {
        let mut rng = fastrand::Rng::with_seed(42);
        let f1 = Arc::new(Field::new("run_ends", DataType::Int64, false));
        let f2 = Arc::new(Field::new("values", DataType::Int64, true));
        let dt = DataType::RunEndEncoded(f1, f2);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let dict_type = DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Int64));
        let f = cg
            .primitive_primitive_cmp(&dict_type, false, &dt, false, Predicate::Eq)
            .unwrap();

        for num_run_ends in [0, 5, 100, 101, 128, 129, 10_000] {
            let ree_array = generate_random_ree_array(num_run_ends);
            let other = Int64Array::from(
                (0..ree_array.len() as usize)
                    .map(|_| rng.i64(-5..5))
                    .collect_vec(),
            );
            let other = arrow_cast::cast(&other, &dict_type).unwrap();

            let as_prim = Int64Array::from_iter(ree_array.downcast::<Int64Array>().unwrap());

            let res = f.call(&other, &ree_array).unwrap();
            let arrow_res = cmp::eq(&other, &as_prim).unwrap();
            assert_eq!(
                res,
                arrow_res,
                "incorrect result on array of size {} (num run ends = {})",
                ree_array.len(),
                num_run_ends
            );
        }
    }

    #[test]
    fn test_ree_to_ree() {
        let f1 = Arc::new(Field::new("run_ends", DataType::Int64, false));
        let f2 = Arc::new(Field::new("values", DataType::Int64, true));
        let dt = DataType::RunEndEncoded(f1, f2);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(&dt, false, &dt, false, Predicate::Eq)
            .unwrap();

        let ree1 = generate_random_ree_array(100);
        let ree2 = generate_random_ree_array(100);

        let result = f.call(&ree1, &ree2).unwrap();
        assert_eq!(result.true_count(), result.len());
    }

    #[test]
    fn test_manual_cast_ree_to_primitive() {
        let res = Int32Array::from(vec![5, 6, 10, 11, 12]);
        let val = Int64Array::from(vec![1, 2, 3, 4, 5]);

        let ree_arr =
            RunArray::try_new(&PrimitiveArray::from(res), &PrimitiveArray::from(val)).unwrap();
        let ree_arr = ree_arr.downcast::<Int64Array>().unwrap();

        let result = Int64Array::from_iter(ree_arr).values().to_vec();

        let expected = vec![1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 5];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cast_ree_i32_to_prim_i32() {
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);

        let cf = cg
            .cast_to_primitive(
                &run_end_data_type(&DataType::Int64, &DataType::Int64),
                &DataType::Int64,
            )
            .unwrap();

        for num_run_ends in [0, 5, 100, 101, 128, 129, 10_000] {
            let ree_array = generate_random_ree_array(num_run_ends);
            let typed = ree_array.downcast::<Int64Array>().unwrap();
            let result = Int64Array::from_iter(typed);

            let our_result = cf
                .call(&ree_array)
                .unwrap()
                .as_primitive::<Int64Type>()
                .clone();
            assert_eq!(result, our_result);
        }
    }

    #[test]
    fn test_ree_sliced_i32() {
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let run_ends = Int32Array::from(vec![10, 20, 30]);
        let values = Int32Array::from(vec![1, 2, 3]);
        let ree_array = RunArray::try_new(&run_ends, &values).unwrap();

        let f = cg
            .primitive_primitive_cmp(
                &ree_array.data_type(),
                false,
                &DataType::Int32,
                true,
                Predicate::Eq,
            )
            .unwrap();

        let sliced = ree_array.slice(15, 10);
        let res = f.call(&sliced, &Int32Array::from(vec![0])).unwrap();

        let typed = sliced.downcast::<Int32Array>().unwrap();
        let arr_array = Int32Array::from_iter(typed);
        let arrow_res = cmp::eq(&arr_array, &Int32Array::new_scalar(0)).unwrap();
        assert_eq!(res, arrow_res);
    }
}
