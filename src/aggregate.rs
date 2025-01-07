use arrow_schema::{ArrowError, DataType};
use inkwell::{intrinsics::Intrinsic, AddressSpace, IntPredicate, OptimizationLevel};

use crate::{CodeGen, CompiledAggFunc, PrimitiveType};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
            Aggregation::Sum => format!("{}add", prefix),
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
        let ptr_type = self.context.ptr_type(AddressSpace::default());

        let next = self.gen_iter_for("iter", dt);

        let agg_intrinsic = Intrinsic::find(&format!(
            "llvm.vector.reduce.{}",
            agg.llvm_func_suffix(prim)
        ))
        .unwrap();
        let agg_f = agg_intrinsic
            .get_declaration(&self.module, &[prim_block_type.into()])
            .unwrap();
        let agg_f_pair = agg_intrinsic
            .get_declaration(&self.module, &[pair_type.into()])
            .unwrap();

        let fn_type = self.context.void_type().fn_type(
            &[
                ptr_type.into(), // arr1
                i64_type.into(), // len
                ptr_type.into(), // out
            ],
            false,
        );
        let function = self.module.add_function("agg", fn_type, None);

        let arr1_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let len = function.get_nth_param(1).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(2).unwrap().into_pointer_value();

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
        builder.build_return(None).unwrap();

        self.optimize()?;
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        Ok(CompiledAggFunc {
            _cg: self,
            src_dt: dt.clone(),
            f: unsafe { ee.get_function("agg").ok().unwrap() },
        })
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{cast::AsArray, types::Int32Type, Int32Array, UInt32Array};
    use arrow_schema::DataType;
    use inkwell::context::Context;
    use itertools::Itertools;

    use crate::CodeGen;

    use super::Aggregation;

    const SIZES_TO_TRY: &[usize] = &[5, 50, 64, 100, 128, 200, 2048, 2049, 14415];

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
            println!("{:?}", data);
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
