use arrow_schema::{ArrowError, DataType};
use inkwell::{intrinsics::Intrinsic, AddressSpace, IntPredicate, OptimizationLevel};

use crate::{
    declare_blocks, CodeGen, CompiledConvertFunc, CompiledFilterFunc, CompiledTakeFunc,
    PrimitiveType,
};

const MURMUR_C1: u64 = 0xff51afd7ed558ccd;
const MURMUR_C2: u64 = 0xc4ceb9fe1a85ec53;
const UPPER_MASK: u64 = 0x00000000FFFFFFFF;

impl<'a> CodeGen<'a> {
    pub fn compile_murmur2(self, dt: &DataType) -> Result<CompiledConvertFunc<'a>, ArrowError> {
        let builder = self.context.create_builder();

        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let prim_type = PrimitiveType::for_arrow_type(dt);

        let fn_type = self.context.i64_type().fn_type(
            &[
                ptr_type.into(), // src
                i64_type.into(), // len
                ptr_type.into(), // tar
            ],
            false,
        );
        let function = self.module.add_function("hash", fn_type, None);

        let arr1_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let len = function.get_nth_param(1).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(2).unwrap().into_pointer_value();

        let next = self
            .gen_block_iter_for("source", dt)
            .expect("hash assumes a block iterator");
        let entry = self.context.append_basic_block(function, "entry");
        let loop_cond = self.context.append_basic_block(function, "loop_cond");
        let loop_body = self.context.append_basic_block(function, "loop_body");
        let end = self.context.append_basic_block(function, "end");

        builder.position_at_end(entry);
        let out_idx_ptr = builder.build_alloca(i64_type, "out_idx_ptr").unwrap();
        builder
            .build_store(out_idx_ptr, i64_type.const_zero())
            .unwrap();
        let iter_ptr = self.initialize_iter(&builder, arr1_ptr, len, dt);
        builder.build_unconditional_branch(loop_body).unwrap();

        builder.position_at_end(loop_body);
        let block = builder
            .build_call(next, &[iter_ptr.into()], "block")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_vector_value();

        let v33 = builder
            .build_insert_element(
                i64_type.vec_type(64).const_zero(),
                i64_type.const_int(33, false),
                i64_type.const_zero(),
                "v1",
            )
            .unwrap();
        let v33 = builder
            .build_shuffle_vector(
                v33,
                i64_type.vec_type(64).get_undef(),
                i64_type.vec_type(64).const_zero(),
                "v1_bcast",
            )
            .unwrap();
        let v_upper = builder
            .build_insert_element(
                i64_type.vec_type(64).const_zero(),
                i64_type.const_int(UPPER_MASK, false),
                i64_type.const_zero(),
                "v1",
            )
            .unwrap();
        let v_upper = builder
            .build_shuffle_vector(
                v_upper,
                i64_type.vec_type(64).get_undef(),
                i64_type.vec_type(64).const_zero(),
                "v1_bcast",
            )
            .unwrap();
        let vm_c1 = builder
            .build_insert_element(
                i64_type.vec_type(64).const_zero(),
                i64_type.const_int(MURMUR_C1, false),
                i64_type.const_zero(),
                "v1",
            )
            .unwrap();
        let vm_c1 = builder
            .build_shuffle_vector(
                vm_c1,
                i64_type.vec_type(64).get_undef(),
                i64_type.vec_type(64).const_zero(),
                "v1_bcast",
            )
            .unwrap();
        let vm_c2 = builder
            .build_insert_element(
                i64_type.vec_type(64).const_zero(),
                i64_type.const_int(MURMUR_C2, false),
                i64_type.const_zero(),
                "v1",
            )
            .unwrap();
        let vm_c2 = builder
            .build_shuffle_vector(
                vm_c2,
                i64_type.vec_type(64).get_undef(),
                i64_type.vec_type(64).const_zero(),
                "v1_bcast",
            )
            .unwrap();

        let block = self.gen_convert_vec(&builder, block, prim_type, PrimitiveType::U64);
        // v = v ^ (v >> 33);
        let block = builder
            .build_xor(
                block,
                builder.build_right_shift(block, v33, false, "shr").unwrap(),
                "xor",
            )
            .unwrap();

        // v = (v & UPPER_MASK).wrapping_mul(MURMUR_C1 & UPPER_MASK);
        let block = builder
            .build_int_nuw_mul(
                builder.build_and(block, v_upper, "and_outer").unwrap(),
                builder.build_and(vm_c1, v_upper, "and_inner").unwrap(),
                "mul",
            )
            .unwrap();

        // v = v ^ (v >> 33);
        let block = builder
            .build_xor(
                block,
                builder.build_right_shift(block, v33, false, "shr").unwrap(),
                "xor",
            )
            .unwrap();

        // v = (v & UPPER_MASK).wrapping_mul(MURMUR_C2 & UPPER_MASK);
        let block = builder
            .build_int_nuw_mul(
                builder.build_and(block, v_upper, "and_outer").unwrap(),
                builder.build_and(vm_c2, v_upper, "and_inner").unwrap(),
                "mul",
            )
            .unwrap();

        // v = v ^ (v >> 33);
        let block = builder
            .build_xor(
                block,
                builder.build_right_shift(block, v33, false, "shr").unwrap(),
                "xor",
            )
            .unwrap();

        let out_idx = builder
            .build_load(i64_type, out_idx_ptr, "out_idx")
            .unwrap()
            .into_int_value();
        let out_pos =
            self.increment_pointer(&builder, out_ptr, PrimitiveType::U64.width(), out_idx);
        builder.build_store(out_pos, block).unwrap();

        let inc_out_pos = builder
            .build_int_add(out_idx, i64_type.const_int(64, false), "inc_out_pos")
            .unwrap();
        builder.build_store(out_idx_ptr, inc_out_pos).unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(loop_cond);
        let has_next = self.has_next_iter(&builder, iter_ptr, dt);
        builder
            .build_conditional_branch(has_next, loop_body, end)
            .unwrap();

        builder.position_at_end(end);
        builder.build_return(Some(&len)).unwrap();

        self.optimize()?;
        self.module
            .verify()
            .map_err(|e| ArrowError::ComputeError(format!("Error compiling kernel: {}", e)))?;
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        Ok(CompiledConvertFunc {
            _cg: self,
            src_dt: dt.clone(),
            tar_dt: DataType::UInt64,
            f: unsafe { ee.get_function("hash").ok().unwrap() },
        })
    }

    pub fn compile_filter_random_access(
        self,
        dt: &DataType,
    ) -> Result<CompiledFilterFunc<'a>, ArrowError> {
        let builder = self.context.create_builder();

        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let prim_type = PrimitiveType::for_arrow_type(dt);

        let convert = if matches!(dt, DataType::Utf8 | DataType::LargeUtf8) {
            Some(self.add_ptrx2_to_view())
        } else {
            None
        };

        let fn_type = i64_type.fn_type(
            &[
                ptr_type.into(), // data (source)
                ptr_type.into(), // bools
                i64_type.into(), // len
                ptr_type.into(), // tar
            ],
            false,
        );
        let function = self.module.add_function("filter", fn_type, None);

        let data_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let filter_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let len = function.get_nth_param(2).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(3).unwrap().into_pointer_value();

        let access_data = self.gen_random_access_for("data", dt);
        let next_idx = self.gen_iter_bitmap("idx");

        declare_blocks!(self.context, function, entry, loop_cond, loop_body, exit);

        builder.position_at_end(entry);
        let data_iter = self.initialize_iter(&builder, data_ptr, len, dt);
        let filter_iter = self.initialize_iter(&builder, filter_ptr, len, &DataType::Boolean);
        let filter_idx_ptr = builder.build_alloca(i64_type, "idx_ptr").unwrap();
        let out_idx_ptr = builder.build_alloca(i64_type, "out_idx_ptr").unwrap();
        builder
            .build_store(out_idx_ptr, i64_type.const_zero())
            .unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(loop_cond);
        let had_next = builder
            .build_call(
                next_idx,
                &[filter_iter.into(), filter_idx_ptr.into()],
                "next_idx",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        builder
            .build_conditional_branch(had_next, loop_body, exit)
            .unwrap();

        builder.position_at_end(loop_body);
        let filter_idx = builder.build_load(i64_type, filter_idx_ptr, "idx").unwrap();
        let mut itm = builder
            .build_call(access_data, &[data_iter.into(), filter_idx.into()], "itm")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left();
        if let Some(convert) = convert {
            // transform the pointer pair into a view
            let base_ptr = self.get_string_base_data_ptr(&builder, data_iter);
            itm = builder
                .build_call(convert, &[itm.into(), base_ptr.into()], "view")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left();
        }

        let out_idx = builder
            .build_load(i64_type, out_idx_ptr, "out_idx")
            .unwrap()
            .into_int_value();
        let inc_out_ptr = self.increment_pointer(&builder, out_ptr, prim_type.width(), out_idx);
        builder.build_store(inc_out_ptr, itm).unwrap();
        let inc_out_idx = builder
            .build_int_add(out_idx, i64_type.const_int(1, false), "inc_out_idx")
            .unwrap();
        builder.build_store(out_idx_ptr, inc_out_idx).unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(exit);
        let out_idx = builder
            .build_load(i64_type, out_idx_ptr, "out_idx")
            .unwrap()
            .into_int_value();
        builder.build_return(Some(&out_idx)).unwrap();

        self.optimize()?;
        self.module
            .verify()
            .map_err(|e| ArrowError::ComputeError(format!("Error compiling kernel: {}", e)))?;
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        Ok(CompiledFilterFunc {
            _cg: self,
            src_dt: dt.clone(),
            f: unsafe { ee.get_function("filter").ok().unwrap() },
        })
    }

    /// Compiles a function to materialize filtered arrays. The compiled
    /// function takes an array of a particular type and boolean array, and
    /// returns a new primitive array containing the values that are true in the
    /// boolean array.
    pub fn compile_filter_block(self, dt: &DataType) -> Result<CompiledFilterFunc<'a>, ArrowError> {
        let builder = self.context.create_builder();

        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let prim_type = PrimitiveType::for_arrow_type(dt);
        let prim_block_type = prim_type.llvm_vec_type(self.context, 64);
        let bool_block_type = self.context.bool_type().vec_type(64);

        let cstore = Intrinsic::find("llvm.masked.compressstore").unwrap();
        let cstore_f = cstore
            .get_declaration(&self.module, &[prim_block_type.into()])
            .unwrap();

        let popcount = Intrinsic::find("llvm.ctpop").unwrap();
        let popcount_f = popcount
            .get_declaration(&self.module, &[i64_type.into()])
            .unwrap();

        let fn_type = i64_type.fn_type(
            &[
                ptr_type.into(), // src
                ptr_type.into(), // bool array
                i64_type.into(), // len
                ptr_type.into(), // tar
            ],
            false,
        );
        let function = self.module.add_function("filter", fn_type, None);

        let arr1_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let filter_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let len = function.get_nth_param(2).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(3).unwrap().into_pointer_value();

        let next = self
            .gen_block_iter_for("source", dt)
            .expect("filter assumes a block iterator");
        // we treat the boolean array as a u8 array, requesting a block size of
        // 8, so we get 64 bits per block
        let bm_next = self.gen_iter_primitive("bitmap", PrimitiveType::U8, 8);

        let entry = self.context.append_basic_block(function, "entry");
        let loop_cond = self.context.append_basic_block(function, "loop_cond");
        let loop_body = self.context.append_basic_block(function, "loop_body");
        let end = self.context.append_basic_block(function, "end");

        builder.position_at_end(entry);
        let out_idx_ptr = builder.build_alloca(i64_type, "out_idx_ptr").unwrap();
        builder
            .build_store(out_idx_ptr, i64_type.const_zero())
            .unwrap();
        let iter_ptr = self.initialize_iter(&builder, arr1_ptr, len, dt);
        let bm_len = builder
            .build_int_unsigned_div(len, i64_type.const_int(8, false), "div8")
            .unwrap();
        let rem = builder
            .build_int_unsigned_rem(len, i64_type.const_int(8, false), "rem8")
            .unwrap();
        let bm_len = builder
            .build_select(
                builder
                    .build_int_compare(
                        inkwell::IntPredicate::EQ,
                        rem,
                        i64_type.const_zero(),
                        "is_rem",
                    )
                    .unwrap(),
                bm_len,
                builder
                    .build_int_add(bm_len, i64_type.const_int(1, false), "p1")
                    .unwrap(),
                "bm_len",
            )
            .unwrap()
            .into_int_value();

        let bm_iter_ptr = self.initialize_iter(&builder, filter_ptr, bm_len, &DataType::UInt8);
        builder.build_unconditional_branch(loop_body).unwrap();

        builder.position_at_end(loop_body);
        let block = builder
            .build_call(next, &[iter_ptr.into()], "block")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_vector_value();
        let u8_block = builder
            .build_call(bm_next, &[bm_iter_ptr.into()], "bm_block")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_vector_value();
        let bool_block = builder
            .build_bit_cast(u8_block, bool_block_type, "bool_block")
            .unwrap();

        let out_idx = builder
            .build_load(i64_type, out_idx_ptr, "out_idx")
            .unwrap()
            .into_int_value();
        let out_pos = self.increment_pointer(&builder, out_ptr, prim_type.width(), out_idx);
        builder
            .build_call(
                cstore_f,
                &[block.into(), out_pos.into(), bool_block.into()],
                "comp_store",
            )
            .unwrap();

        let num_els = builder
            .build_call(
                popcount_f,
                &[builder
                    .build_bit_cast(bool_block, i64_type, "as_i64")
                    .unwrap()
                    .into()],
                "num_els",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        let inc_out_pos = builder
            .build_int_add(out_idx, num_els, "inc_out_pos")
            .unwrap();
        builder.build_store(out_idx_ptr, inc_out_pos).unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(loop_cond);
        let has_next = self.has_next_iter(&builder, iter_ptr, dt);
        builder
            .build_conditional_branch(has_next, loop_body, end)
            .unwrap();

        builder.position_at_end(end);
        let out_idx = builder
            .build_load(i64_type, out_idx_ptr, "out_idx")
            .unwrap()
            .into_int_value();
        builder.build_return(Some(&out_idx)).unwrap();

        self.optimize()?;
        self.module
            .verify()
            .map_err(|e| ArrowError::ComputeError(format!("Error compiling kernel: {}", e)))?;
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        Ok(CompiledFilterFunc {
            _cg: self,
            src_dt: dt.clone(),
            f: unsafe { ee.get_function("filter").ok().unwrap() },
        })
    }

    /// Compiles a function to take elements from an array based on indices.
    pub fn compile_take(
        self,
        idx_dt: &DataType,
        data_dt: &DataType,
    ) -> Result<CompiledTakeFunc<'a>, ArrowError> {
        let builder = self.context.create_builder();

        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let prim_type = PrimitiveType::for_arrow_type(data_dt);

        let fn_type = self.context.void_type().fn_type(
            &[
                ptr_type.into(), // src
                ptr_type.into(), // idx array
                i64_type.into(), // len of idx array
                ptr_type.into(), // output
            ],
            false,
        );
        let function = self.module.add_function("take", fn_type, None);

        let data_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let take_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let len = function.get_nth_param(2).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(3).unwrap().into_pointer_value();

        let access_idx = self.gen_random_access_for("get_idx", idx_dt);
        let access_data = self.gen_random_access_for("get_dat", data_dt);

        let convert = if matches!(data_dt, DataType::Utf8 | DataType::LargeUtf8) {
            Some(self.add_ptrx2_to_view())
        } else {
            None
        };

        declare_blocks!(self.context, function, entry, loop_cond, loop_body, exit);

        builder.position_at_end(entry);
        let curr_idx_ptr = builder.build_alloca(i64_type, "curr_idx_ptr").unwrap();
        builder
            .build_store(curr_idx_ptr, i64_type.const_zero())
            .unwrap();
        let idx_iter = self.initialize_iter(&builder, take_ptr, len, idx_dt);
        let data_iter = self.initialize_iter(&builder, data_ptr, len, data_dt);
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(loop_cond);
        let curr_idx = builder
            .build_load(i64_type, curr_idx_ptr, "curr_idx")
            .unwrap()
            .into_int_value();
        let cond = builder
            .build_int_compare(IntPredicate::ULT, curr_idx, len, "cond")
            .unwrap();
        builder
            .build_conditional_branch(cond, loop_body, exit)
            .unwrap();

        builder.position_at_end(loop_body);
        let curr_idx = builder
            .build_load(i64_type, curr_idx_ptr, "curr_idx")
            .unwrap()
            .into_int_value();

        let data_idx = builder
            .build_call(access_idx, &[idx_iter.into(), curr_idx.into()], "data_idx")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();

        let data_idx = builder
            .build_int_z_extend_or_bit_cast(data_idx, i64_type, "zext")
            .unwrap();

        let mut next_data = builder
            .build_call(
                access_data,
                &[data_iter.into(), data_idx.into()],
                "next_data",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left();

        if let Some(convert) = convert {
            // transform the pointer pair into a view
            let base_ptr = self.get_string_base_data_ptr(&builder, data_iter);
            next_data = builder
                .build_call(convert, &[next_data.into(), base_ptr.into()], "view")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left();
        }

        let out_loc = self.increment_pointer(&builder, out_ptr, prim_type.width(), curr_idx);
        builder.build_store(out_loc, next_data).unwrap();
        let inc_idx = builder
            .build_int_add(curr_idx, i64_type.const_int(1, false), "inc_idx")
            .unwrap();
        builder.build_store(curr_idx_ptr, inc_idx).unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(exit);
        builder.build_return(None).unwrap();

        self.optimize()?;
        self.module
            .verify()
            .map_err(|e| ArrowError::ComputeError(format!("Error compiling kernel: {}", e)))?;
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        Ok(CompiledTakeFunc {
            _cg: self,
            data_dt: data_dt.clone(),
            take_dt: idx_dt.clone(),
            f: unsafe { ee.get_function("take").ok().unwrap() },
        })
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        cast::AsArray,
        types::{Int32Type, Int64Type, UInt64Type},
        Array, BooleanArray, Int32Array, Int64Array, StringArray,
    };
    use inkwell::context::Context;
    use itertools::Itertools;

    use crate::{test_utils::generate_random_ree_array, CodeGen};

    #[test]
    fn test_hash_i32() {
        let data = vec![0, 0, 1, 2, 3];
        let data = Int32Array::from(data);

        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);
        let compiled_func = codegen
            .compile_murmur2(data.data_type())
            .expect("Failed to compile murmur2 function");
        let result = compiled_func.call(&data).unwrap();
        let result = result.as_primitive::<UInt64Type>().values().to_vec();
        assert_eq!(result.len(), data.len());
        assert_eq!(result[0], result[1]);
        assert_ne!(result[0], result[2]);
        assert_ne!(result[0], result[3]);
    }

    #[test]
    fn test_hash_many_i32() {
        let mut rng = fastrand::Rng::with_seed(42);
        let data = (0..10_000).map(|_| rng.i32(..)).collect_vec();
        let data = Int32Array::from(data);

        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);
        let compiled_func = codegen
            .compile_murmur2(data.data_type())
            .expect("Failed to compile murmur2 function");
        let result = compiled_func.call(&data).unwrap();
        let result = result.as_primitive::<UInt64Type>().values().to_vec();
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_filter_i32_block() {
        let mut rng = fastrand::Rng::with_seed(42);
        let data = (0..10_000).map(|_| rng.i32(..)).collect_vec();
        let data = Int32Array::from(data);
        let mask: BooleanArray = (0..10_000).map(|_| Some(rng.bool())).collect();

        let arr_filtered = arrow_select::filter::filter(&data, &mask)
            .unwrap()
            .as_primitive::<Int32Type>()
            .clone();

        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);
        let compiled_func = codegen
            .compile_filter_block(data.data_type())
            .expect("Failed to compile filter function");
        let result = compiled_func
            .call(&data, &mask)
            .unwrap()
            .as_primitive::<Int32Type>()
            .clone();

        assert_eq!(arr_filtered, result);
    }

    #[test]
    fn test_filter_i32_random_access() {
        let mut rng = fastrand::Rng::with_seed(42);
        let data = (0..10_000).map(|_| rng.i32(..)).collect_vec();
        let data = Int32Array::from(data);
        let mask: BooleanArray = (0..10_000).map(|_| Some(rng.bool())).collect();

        let arr_filtered = arrow_select::filter::filter(&data, &mask)
            .unwrap()
            .as_primitive::<Int32Type>()
            .clone();

        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);
        let compiled_func = codegen
            .compile_filter_random_access(data.data_type())
            .expect("Failed to compile filter function");
        let result = compiled_func
            .call(&data, &mask)
            .unwrap()
            .as_primitive::<Int32Type>()
            .clone();

        assert_eq!(arr_filtered, result);
    }

    #[test]
    fn test_filter_i64_ree() {
        let data = generate_random_ree_array(50);
        let prim_data = Int64Array::from_iter(data.downcast::<Int64Array>().unwrap());

        let mut rng = fastrand::Rng::with_seed(42);
        let mask: BooleanArray = (0..prim_data.len()).map(|_| Some(rng.bool())).collect();

        let arr_filtered = arrow_select::filter::filter(&prim_data, &mask)
            .unwrap()
            .as_primitive::<Int64Type>()
            .clone();

        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);
        let compiled_func = codegen
            .compile_filter_block(data.data_type())
            .expect("Failed to compile filter function");
        let result = compiled_func
            .call(&data, &mask)
            .unwrap()
            .as_primitive::<Int64Type>()
            .clone();

        assert_eq!(arr_filtered, result);
    }

    #[test]
    fn test_take_i32() {
        let data = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let indices = Int32Array::from(vec![0, 2, 4]);

        let arr_taken = arrow_select::take::take(&data, &indices, None)
            .unwrap()
            .as_primitive::<Int32Type>()
            .clone();

        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);
        let compiled_func = codegen
            .compile_take(indices.data_type(), data.data_type())
            .expect("Failed to compile take function");
        let result = compiled_func
            .call(&data, &indices)
            .unwrap()
            .as_primitive::<Int32Type>()
            .clone();

        assert_eq!(arr_taken, result);
    }

    #[test]
    fn test_filter_str() {
        let data = StringArray::from(vec!["hello", "world", "rust"]);
        let indices = BooleanArray::from(vec![true, false, true]);

        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);
        let compiled_func = codegen
            .compile_filter_random_access(data.data_type())
            .unwrap();
        let result = compiled_func.call(&data, &indices).unwrap();
        let result = result.as_string_view().iter().collect_vec();
        assert_eq!(result, vec![Some("hello"), Some("rust")]);
    }

    #[test]
    fn test_take_str() {
        let data = StringArray::from(vec!["hello", "world", "rust"]);
        let indices = Int32Array::from(vec![0, 2]);

        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);
        let compiled_func = codegen
            .compile_take(indices.data_type(), data.data_type())
            .expect("Failed to compile take function");
        let result = compiled_func.call(&data, &indices).unwrap();
        let result = result.as_string_view().iter().collect_vec();
        assert_eq!(result, vec![Some("hello"), Some("rust")]);
    }

    #[test]
    fn test_take_long_str() {
        let data = StringArray::from(vec![
            "hello",
            "world",
            "rust",
            "this is more than twelve characters",
        ]);
        let indices = Int32Array::from(vec![0, 2, 3]);

        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);
        let compiled_func = codegen
            .compile_take(indices.data_type(), data.data_type())
            .expect("Failed to compile take function");
        let result = compiled_func.call(&data, &indices).unwrap();
        let result = result.as_string_view().iter().collect_vec();
        assert_eq!(
            result,
            vec![
                Some("hello"),
                Some("rust"),
                Some("this is more than twelve characters")
            ]
        );
    }
}
