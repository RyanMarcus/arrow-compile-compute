use arrow_schema::{ArrowError, DataType};
use enum_as_inner::EnumAsInner;
use inkwell::{intrinsics::Intrinsic, AddressSpace, IntPredicate, OptimizationLevel};

use crate::{declare_blocks, CodeGen, CompiledAggFunc, PrimitiveType};

#[derive(Clone, Copy, PartialEq, Eq, Hash, EnumAsInner)]
pub enum Aggregation {
    Min,
    Max,
    Sum,
}

impl Aggregation {
    fn llvm_func_suffix(&self, ptype: PrimitiveType) -> String {
        let prefix = if ptype.is_int() {
            if ptype.is_signed() {
                "s"
            } else {
                "u"
            }
        } else {
            "f"
        };

        match self {
            Aggregation::Min => format!("{}min", prefix),
            Aggregation::Max => format!("{}max", prefix),
            Aggregation::Sum => "add".into(),
        }
    }
}

impl<'ctx> CodeGen<'ctx> {
    pub fn compile_ungrouped_aggregation(
        self,
        dt: &DataType,
        agg: Aggregation,
    ) -> Result<CompiledAggFunc<'ctx>, ArrowError> {
        let prim = PrimitiveType::for_arrow_type(dt);
        let prim_llvm_type = prim.llvm_type(&self.context);
        let pair_type = prim.llvm_vec_type(&self.context, 2);
        let prim_block_type = prim.llvm_vec_type(&self.context, 64);

        let builder = self.context.create_builder();
        let i64_type = self.context.i64_type();
        let i1_type = self.context.bool_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());

        let next = self.gen_iter_for("iter", dt);

        let agg_intrinsic = Intrinsic::find(&format!(
            "llvm.vector.reduce.{}",
            agg.llvm_func_suffix(prim)
        ))
        .expect(&format!(
            "unable to find intrinsic for suffix {}",
            agg.llvm_func_suffix(prim)
        ));
        let agg_f = agg_intrinsic
            .get_declaration(&self.module, &[prim_block_type.into()])
            .unwrap();
        let agg_f_pair = agg_intrinsic
            .get_declaration(&self.module, &[pair_type.into()])
            .unwrap();

        let fn_type = i1_type.fn_type(
            &[
                ptr_type.into(), // arr1
                ptr_type.into(), // null pointer (for null bitmap)
                i64_type.into(), // len
                ptr_type.into(), // out
            ],
            false,
        );
        let function = self.module.add_function("agg", fn_type, None);

        let arr1_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let len = function.get_nth_param(2).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(3).unwrap().into_pointer_value();

        let entry = self.context.append_basic_block(function, "entry");
        let loop_cond = self.context.append_basic_block(function, "loop_cond");
        let loop_body = self.context.append_basic_block(function, "loop_body");
        let tail_cond = self.context.append_basic_block(function, "tail_cond");
        let init_tail = self.context.append_basic_block(function, "init_tail");
        let tail_loop_cond = self.context.append_basic_block(function, "tail_loop_cond");
        let tail_loop_body = self.context.append_basic_block(function, "tail_loop_body");
        let end = self.context.append_basic_block(function, "end");

        builder.position_at_end(entry);
        let iter_ptr = self.initialize_iter(&builder, arr1_ptr, len, dt);
        let remaining_ptr = builder.build_alloca(i64_type, "remaining_ptr").unwrap();
        builder.build_store(remaining_ptr, len).unwrap();

        let accum_ptr = builder.build_alloca(prim_llvm_type, "accum").unwrap();
        // store a neutral value into accum
        let to_store = match agg {
            Aggregation::Min => prim.max_value(&self.context),
            Aggregation::Max => prim.min_value(&self.context),
            Aggregation::Sum => prim.zero(&self.context),
        };
        builder.build_store(accum_ptr, to_store).unwrap();

        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(loop_cond);
        let remaining = builder
            .build_load(i64_type, remaining_ptr, "remaining")
            .unwrap()
            .into_int_value();
        let over_64 = builder
            .build_int_compare(
                IntPredicate::UGE,
                remaining,
                i64_type.const_int(64, false),
                "over_64",
            )
            .unwrap();
        builder
            .build_conditional_branch(over_64, loop_body, tail_cond)
            .unwrap();

        builder.position_at_end(loop_body);
        let block = builder
            .build_call(next, &[iter_ptr.into()], "init_block")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_vector_value();

        let block_agg = builder
            .build_call(agg_f, &[block.into()], "init_agg")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left();
        let curr_agg = builder
            .build_load(prim_llvm_type, accum_ptr, "curr_agg")
            .unwrap();

        // Compare block_agg with curr_agg, store the result back into accum_ptr.
        // To do this, we'll create a vector of size 2 and then aggregate the
        // vector.
        let with_first = builder
            .build_insert_element(
                pair_type.const_zero(),
                block_agg,
                i64_type.const_int(0, false),
                "with_first",
            )
            .unwrap();
        let pair = builder
            .build_insert_element(with_first, curr_agg, i64_type.const_int(1, false), "pair")
            .unwrap();
        let new_agg = builder
            .build_call(agg_f_pair, &[pair.into()], "new_agg")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left();
        builder.build_store(accum_ptr, new_agg).unwrap();

        let remaining = builder
            .build_load(i64_type, remaining_ptr, "remaining")
            .unwrap()
            .into_int_value();
        let new_remaining = builder
            .build_int_sub(remaining, i64_type.const_int(64, false), "new_remaining")
            .unwrap();
        builder.build_store(remaining_ptr, new_remaining).unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(tail_cond);
        let remaining = builder
            .build_load(i64_type, remaining_ptr, "remaining")
            .unwrap()
            .into_int_value();
        let gt0 = builder
            .build_int_compare(IntPredicate::UGT, remaining, i64_type.const_zero(), "gt0")
            .unwrap();
        builder
            .build_conditional_branch(gt0, init_tail, end)
            .unwrap();

        builder.position_at_end(init_tail);
        let tail_block = builder
            .build_call(next, &[iter_ptr.into()], "init_block")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_vector_value();
        let tail_idx_ptr = builder.build_alloca(i64_type, "tail_ptr_idx").unwrap();
        builder
            .build_store(tail_idx_ptr, i64_type.const_zero())
            .unwrap();
        builder.build_unconditional_branch(tail_loop_body).unwrap();

        builder.position_at_end(tail_loop_cond);
        let remaining = builder
            .build_load(i64_type, remaining_ptr, "remaining")
            .unwrap()
            .into_int_value();
        let tail_idx = builder
            .build_load(i64_type, tail_idx_ptr, "tail_idx")
            .unwrap()
            .into_int_value();
        let have_more = builder
            .build_int_compare(IntPredicate::ULT, tail_idx, remaining, "have_more")
            .unwrap();
        builder
            .build_conditional_branch(have_more, tail_loop_body, end)
            .unwrap();

        builder.position_at_end(tail_loop_body);
        let tail_idx = builder
            .build_load(i64_type, tail_idx_ptr, "tail_idx")
            .unwrap()
            .into_int_value();
        let element = builder
            .build_extract_element(tail_block, tail_idx, "element")
            .unwrap();
        let curr_agg = builder
            .build_load(prim_llvm_type, accum_ptr, "curr_agg")
            .unwrap();
        let with_first = builder
            .build_insert_element(
                pair_type.const_zero(),
                element,
                i64_type.const_int(0, false),
                "with_first",
            )
            .unwrap();
        let pair = builder
            .build_insert_element(with_first, curr_agg, i64_type.const_int(1, false), "pair")
            .unwrap();
        let new_agg = builder
            .build_call(agg_f_pair, &[pair.into()], "new_agg")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left();
        builder.build_store(accum_ptr, new_agg).unwrap();

        let new_tail_idx = builder
            .build_int_add(tail_idx, i64_type.const_int(1, false), "new_tail_idx")
            .unwrap();
        builder.build_store(tail_idx_ptr, new_tail_idx).unwrap();

        builder.build_unconditional_branch(tail_loop_cond).unwrap();

        builder.position_at_end(end);
        let curr_agg = builder
            .build_load(prim_llvm_type, accum_ptr, "curr_agg")
            .unwrap();
        builder.build_store(out_ptr, curr_agg).unwrap();
        builder
            .build_return(Some(&i1_type.const_all_ones()))
            .unwrap();

        self.module.verify().unwrap();
        self.optimize()?;
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        Ok(CompiledAggFunc {
            _cg: self,
            nullable: false,
            src_dt: dt.clone(),
            f: unsafe { ee.get_function("agg").unwrap() },
        })
    }

    pub fn compile_ungrouped_agg_with_nulls(
        self,
        dt: &DataType,
        agg: Aggregation,
    ) -> Result<CompiledAggFunc<'ctx>, ArrowError> {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i64_type = self.context.i64_type();
        let i1_type = self.context.bool_type();
        let prim = PrimitiveType::for_arrow_type(dt);
        let dtype = prim.llvm_type(self.context);

        let next_bit = self.gen_iter_bitmap("agg");
        let access = self.gen_random_access_for("agg", dt);

        let agg_intrinsic = Intrinsic::find(&format!("llvm.{}", agg.llvm_func_suffix(prim)))
            .expect(&format!(
                "unable to find intrinsic for suffix {}",
                agg.llvm_func_suffix(prim)
            ));
        let agg_f = agg_intrinsic
            .get_declaration(&self.module, &[dtype.into()])
            .unwrap();

        let fn_type = i1_type.fn_type(
            &[
                ptr_type.into(), // arr1
                ptr_type.into(), // null map
                i64_type.into(), // len
                ptr_type.into(), // out
            ],
            false,
        );
        let function = self.module.add_function("agg", fn_type, None);

        let arr1_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let nulls_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let len = function.get_nth_param(2).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(3).unwrap().into_pointer_value();

        let builder = self.context.create_builder();
        declare_blocks!(self.context, function, entry, loop_cond, loop_body, exit);

        builder.position_at_end(entry);
        let null_iter = self.initialize_iter_bitmap(&builder, nulls_ptr, len);
        let data_iter = self.initialize_iter(&builder, arr1_ptr, len, dt);
        let idx_out_ptr = builder.build_alloca(i64_type, "idx_out_ptr").unwrap();

        // store the first value
        let had_first = builder
            .build_call(
                next_bit,
                &[null_iter.into(), idx_out_ptr.into()],
                "had_first",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        let first_idx = builder
            .build_load(i64_type, idx_out_ptr, "first_idx")
            .unwrap();
        let val = builder
            .build_call(access, &[data_iter.into(), first_idx.into()], "value")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left();
        builder.build_store(out_ptr, val).unwrap();

        builder
            .build_conditional_branch(had_first, loop_cond, exit)
            .unwrap();

        builder.position_at_end(loop_cond);
        let had_next = builder
            .build_call(
                next_bit,
                &[null_iter.into(), idx_out_ptr.into()],
                "next_bit",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        builder
            .build_conditional_branch(had_next, loop_body, exit)
            .unwrap();

        builder.position_at_end(loop_body);
        let idx = builder
            .build_load(i64_type, idx_out_ptr, "idx")
            .unwrap()
            .into_int_value();
        let val = builder
            .build_call(access, &[data_iter.into(), idx.into()], "value")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left();
        let curr_val = builder.build_load(dtype, out_ptr, "curr_val").unwrap();
        let new_val = builder
            .build_call(agg_f, &[val.into(), curr_val.into()], "new_val")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left();
        builder.build_store(out_ptr, new_val).unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(exit);
        let had_any = builder.build_phi(i1_type, "had_any").unwrap();
        had_any.add_incoming(&[
            (&i1_type.const_zero(), entry), // if we are coming from entry, no values
            (&i1_type.const_all_ones(), loop_cond), // otherwise, we did have values
        ]);
        let had_any = had_any.as_basic_value().into_int_value();
        builder.build_return(Some(&had_any)).unwrap();

        self.module.verify().unwrap();
        self.optimize()?;
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        Ok(CompiledAggFunc {
            _cg: self,
            nullable: true,
            src_dt: dt.clone(),
            f: unsafe { ee.get_function("agg").unwrap() },
        })
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{cast::AsArray, types::Int32Type, Int32Array, UInt32Array};
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use inkwell::context::Context;
    use itertools::Itertools;

    use crate::CodeGen;

    use super::Aggregation;

    const SIZES_TO_TRY: &[usize] = &[5, 50, 64, 100, 128, 200, 2048, 2049, 14415];

    #[test]
    fn test_i32_min_agg_nulls() {
        let data = vec![10, 20, 5, 1, -20, 30, 0];
        let mask = vec![true, true, true, true, false, true, false];

        let arr = Int32Array::from(data);
        let arr_with_nulls =
            Int32Array::try_new(arr.values().clone(), Some(NullBuffer::from(mask))).unwrap();
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .compile_ungrouped_agg_with_nulls(&DataType::Int32, Aggregation::Min)
            .unwrap();
        let r: Int32Array = f
            .call(&arr_with_nulls)
            .unwrap()
            .unwrap()
            .as_primitive()
            .clone();
        assert_eq!(r.len(), 1);
        assert_eq!(r.value(0), 1);
    }

    #[test]
    fn test_i32_min_agg() {
        let mut rng = fastrand::Rng::with_seed(42);
        for &sz in SIZES_TO_TRY {
            let data = (0..sz).map(|_| rng.i32(..)).collect_vec();
            let data = Int32Array::from(data);
            let arrow_min = arrow_arith::aggregate::min(&data).unwrap();

            let ctx = Context::create();
            let cg = CodeGen::new(&ctx);
            let f = cg
                .compile_ungrouped_aggregation(&DataType::Int32, Aggregation::Min)
                .unwrap();

            let p_arr: Int32Array = f.call(&data).unwrap().unwrap().as_primitive().clone();
            assert_eq!(p_arr.len(), 1);
            assert_eq!(p_arr.value(0), arrow_min);
        }
    }

    #[test]
    fn test_i32_max_agg() {
        let mut rng = fastrand::Rng::with_seed(42);
        for &sz in SIZES_TO_TRY {
            let data = (0..sz).map(|_| rng.i32(..)).collect_vec();
            let data = Int32Array::from(data);
            let arrow_min = arrow_arith::aggregate::max(&data).unwrap();

            let ctx = Context::create();
            let cg = CodeGen::new(&ctx);
            let f = cg
                .compile_ungrouped_aggregation(&DataType::Int32, Aggregation::Max)
                .unwrap();

            let p_arr: Int32Array = f.call(&data).unwrap().unwrap().as_primitive().clone();
            assert_eq!(p_arr.len(), 1);
            assert_eq!(p_arr.value(0), arrow_min);
        }
    }

    #[test]
    fn test_u32_min_agg() {
        let mut rng = fastrand::Rng::with_seed(42);
        for &sz in SIZES_TO_TRY {
            let data = (0..sz).map(|_| rng.u32(..)).collect_vec();
            let data = UInt32Array::from(data);
            let arrow_min = arrow_arith::aggregate::min(&data).unwrap();

            let ctx = Context::create();
            let cg = CodeGen::new(&ctx);
            let f = cg
                .compile_ungrouped_aggregation(&DataType::UInt32, Aggregation::Min)
                .unwrap();

            let p_arr: UInt32Array = f.call(&data).unwrap().unwrap().as_primitive().clone();
            assert_eq!(p_arr.len(), 1);
            assert_eq!(p_arr.value(0), arrow_min);
        }
    }

    #[test]
    fn test_empty_min_agg() {
        let data = Int32Array::from(Vec::<i32>::new());

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .compile_ungrouped_aggregation(&DataType::Int32, Aggregation::Min)
            .unwrap();

        let p_arr = f.call(&data).unwrap();
        assert!(p_arr.is_none());
    }

    #[test]
    fn test_single_min_agg() {
        let data = Int32Array::from(vec![1]);

        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .compile_ungrouped_aggregation(&DataType::Int32, Aggregation::Min)
            .unwrap();

        let p_arr = f
            .call(&data)
            .unwrap()
            .unwrap()
            .as_primitive::<Int32Type>()
            .clone();
        assert_eq!(p_arr.value(0), 1);
    }
}
