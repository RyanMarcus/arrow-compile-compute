use inkwell::{
    builder::Builder,
    module::Linkage,
    types::StructType,
    values::{FunctionValue, IntValue, PointerValue},
    AddressSpace, IntPredicate,
};

use crate::{CodeGen, PrimitiveType};

impl<'ctx> CodeGen<'ctx> {
    pub(crate) fn struct_for_iter_primitive(&self) -> StructType {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i64_type = self.context.i64_type();
        self.context
            .struct_type(&[ptr_type.into(), i64_type.into(), i64_type.into()], false)
    }

    pub(crate) fn initialize_iter_primitive<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        arr: PointerValue,
        len: IntValue,
    ) -> PointerValue<'a>
    where
        'ctx: 'a,
    {
        let i64_type = self.context.i64_type();
        let iter_type = self.struct_for_iter_primitive();
        let ptr = builder.build_alloca(iter_type, "prim_iter_ptr").unwrap();

        let arr_ptr = builder
            .build_struct_gep(iter_type, ptr, 0, "arr_ptr_ptr")
            .unwrap();
        builder.build_store(arr_ptr, arr).unwrap();

        let idx_ptr = builder
            .build_struct_gep(iter_type, ptr, 1, "idx_ptr")
            .unwrap();
        builder.build_store(idx_ptr, i64_type.const_zero()).unwrap();

        let len_ptr = builder
            .build_struct_gep(iter_type, ptr, 2, "len_ptr")
            .unwrap();
        builder.build_store(len_ptr, len).unwrap();

        ptr
    }

    pub(crate) fn has_next_iter_primitive<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        iter: PointerValue<'a>,
    ) -> IntValue<'a>
    where
        'ctx: 'a,
    {
        let i64_type = self.context.i64_type();
        let iter_type = self.struct_for_iter_primitive();

        let idx_ptr = builder
            .build_struct_gep(iter_type, iter, 1, "idx_ptr")
            .unwrap();
        let idx = builder
            .build_load(i64_type, idx_ptr, "idx")
            .unwrap()
            .into_int_value();

        let len_ptr = builder
            .build_struct_gep(iter_type, iter, 2, "len_ptr")
            .unwrap();
        let len = builder
            .build_load(i64_type, len_ptr, "len")
            .unwrap()
            .into_int_value();

        builder
            .build_int_compare(IntPredicate::ULT, idx, len, "has_next")
            .unwrap()
    }

    pub(crate) fn gen_iter_primitive(
        &self,
        label: &str,
        prim_type: PrimitiveType,
        block_size: usize,
    ) -> FunctionValue {
        let builder = self.context.create_builder();

        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let dtype = prim_type.llvm_type(self.context);
        let chunk_type = prim_type.llvm_vec_type(self.context, block_size as u32);
        let iter_type = self.struct_for_iter_primitive();

        let fn_type = chunk_type.fn_type(
            &[
                ptr_type.into(), // iter struct
            ],
            false,
        );
        let function = self.module.add_function(
            &format!("{}_primitive_iter_chunk", label),
            fn_type,
            Some(Linkage::Private),
        );

        let entry = self.context.append_basic_block(function, "entry");
        let full_chunk = self.context.append_basic_block(function, "full_chunk");
        let partial_chunk = self.context.append_basic_block(function, "partial_chunk");
        let partial_loop_cond = self
            .context
            .append_basic_block(function, "partial_loop_cond");
        let partial_loop_body = self
            .context
            .append_basic_block(function, "partial_loop_body");
        let partial_loop_end = self
            .context
            .append_basic_block(function, "partial_loop_end");

        let ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        builder.position_at_end(entry);
        let arr_ptr_ptr = builder
            .build_struct_gep(iter_type, ptr, 0, "arr_ptr_ptr")
            .unwrap();
        let arr_ptr = builder
            .build_load(ptr_type, arr_ptr_ptr, "arr_ptr")
            .unwrap()
            .into_pointer_value();
        let idx_ptr = builder
            .build_struct_gep(iter_type, ptr, 1, "idx_ptr")
            .unwrap();
        let idx = builder
            .build_load(i64_type, idx_ptr, "idx")
            .unwrap()
            .into_int_value();
        let len_ptr = builder.build_struct_gep(iter_type, ptr, 2, "len").unwrap();
        let len = builder
            .build_load(i64_type, len_ptr, "len")
            .unwrap()
            .into_int_value();

        let chunk_ptr = self.increment_pointer(&builder, arr_ptr, prim_type.width(), idx);
        let remaining = builder.build_int_sub(len, idx, "remaining").unwrap();
        let have_full_left = builder
            .build_int_compare(
                IntPredicate::UGE,
                remaining,
                i64_type.const_int(block_size as u64, false),
                "cmp",
            )
            .unwrap();
        builder
            .build_conditional_branch(have_full_left, full_chunk, partial_chunk)
            .unwrap();

        builder.position_at_end(full_chunk);
        let chunk = builder.build_load(chunk_type, chunk_ptr, "chunk").unwrap();
        let inc_idx = builder
            .build_int_add(idx, i64_type.const_int(block_size as u64, false), "inc_idx")
            .unwrap();
        builder.build_store(idx_ptr, inc_idx).unwrap();
        builder.build_return(Some(&chunk)).unwrap();

        builder.position_at_end(partial_chunk);
        let partial_chunk_ptr = builder.build_alloca(chunk_type, "partial_ptr").unwrap();
        builder
            .build_store(partial_chunk_ptr, chunk_type.const_zero())
            .unwrap();
        let partial_idx_ptr = builder.build_alloca(i64_type, "p_idx").unwrap();
        builder
            .build_store(partial_idx_ptr, i64_type.const_zero())
            .unwrap();
        builder
            .build_unconditional_branch(partial_loop_cond)
            .unwrap();

        builder.position_at_end(partial_loop_cond);
        let idx = builder
            .build_load(i64_type, idx_ptr, "idx")
            .unwrap()
            .into_int_value();
        let cmp = builder
            .build_int_compare(IntPredicate::ULT, idx, len, "cmp")
            .unwrap();
        builder
            .build_conditional_branch(cmp, partial_loop_body, partial_loop_end)
            .unwrap();

        builder.position_at_end(partial_loop_body);
        let partial_idx = builder
            .build_load(i64_type, partial_idx_ptr, "partial_idx")
            .unwrap()
            .into_int_value();
        let idx = builder
            .build_load(i64_type, idx_ptr, "idx")
            .unwrap()
            .into_int_value();
        let ptr = self.increment_pointer(&builder, arr_ptr, prim_type.width(), idx);
        let val = builder.build_load(dtype, ptr, "val").unwrap();
        let partial_chunk = builder
            .build_load(chunk_type, partial_chunk_ptr, "partial_chunk")
            .unwrap()
            .into_vector_value();
        let partial_chunk = builder
            .build_insert_element(partial_chunk, val, partial_idx, "partial_chunk")
            .unwrap();
        builder
            .build_store(partial_chunk_ptr, partial_chunk)
            .unwrap();
        builder
            .build_store(
                partial_idx_ptr,
                builder
                    .build_int_add(partial_idx, i64_type.const_int(1, false), "new_partial_idx")
                    .unwrap(),
            )
            .unwrap();
        builder
            .build_store(
                idx_ptr,
                builder
                    .build_int_add(idx, i64_type.const_int(1, false), "new_idx")
                    .unwrap(),
            )
            .unwrap();
        builder
            .build_unconditional_branch(partial_loop_cond)
            .unwrap();

        builder.position_at_end(partial_loop_end);
        let partial_chunk = builder
            .build_load(chunk_type, partial_chunk_ptr, "partial_chunk")
            .unwrap()
            .into_vector_value();
        builder.build_return(Some(&partial_chunk)).unwrap();

        function
    }
}

#[cfg(test)]
mod tests {

    use arrow_array::{
        cast::AsArray, Float16Array, Float32Array, Float64Array, Int16Array, Int32Array,
        Int64Array, Int8Array, UInt16Array, UInt32Array, UInt8Array,
    };
    use arrow_ord::cmp;
    use arrow_schema::DataType;
    use inkwell::context::Context;
    use itertools::Itertools;

    use crate::Predicate;

    use super::*;

    const SIZES_TO_TRY: &[usize] = &[0, 50, 64, 100, 128, 200, 2048, 2049, 14415];

    #[test]
    fn test_llvm_i32_i64() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &DataType::Int32,
                false,
                &DataType::Int64,
                false,
                Predicate::Lt,
            )
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Int32Array::from((0..arr_len).map(|_| rng.i32(-10..10)).collect_vec());
            let arr2 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());

            let res = f.call(&arr1, &arr2).unwrap();
            let arrow_res =
                cmp::lt(&arrow_cast::cast(&arr1, &DataType::Int64).unwrap(), &arr2).unwrap();
            assert_eq!(
                res, arrow_res,
                "incorrect result on array of size {}",
                arr_len
            );
        }
    }

    #[test]
    fn test_llvm_i64_lt() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &DataType::Int64,
                false,
                &DataType::Int64,
                false,
                Predicate::Lt,
            )
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());
            let arr2 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());

            let res = f.call(&arr1, &arr2).unwrap();
            let arrow_res = cmp::lt(&arr1, &arr2).unwrap();
            assert_eq!(
                res, arrow_res,
                "incorrect result on array of size {}",
                arr_len
            );
        }
    }

    #[test]
    fn test_llvm_f64_lt() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &DataType::Float64,
                false,
                &DataType::Float64,
                false,
                Predicate::Lt,
            )
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Float64Array::from((0..arr_len).map(|_| rng.f64()).collect_vec());
            let arr2 = Float64Array::from((0..arr_len).map(|_| rng.f64()).collect_vec());

            let res = f.call(&arr1, &arr2).unwrap();
            let arrow_res = cmp::lt(&arr1, &arr2).unwrap();
            assert_eq!(
                res, arrow_res,
                "incorrect result on array of size {}",
                arr_len
            );
        }
    }

    #[test]
    fn test_llvm_f16_f32_lt() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &DataType::Float16,
                false,
                &DataType::Float32,
                false,
                Predicate::Lt,
            )
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Float16Array::from(
                (0..arr_len)
                    .map(|_| half::f16::from_f32(rng.f32()))
                    .collect_vec(),
            );
            let arr2 = Float32Array::from((0..arr_len).map(|_| rng.f32()).collect_vec());

            let res = f.call(&arr1, &arr2).unwrap();

            let converted = arrow_cast::cast(&arr1, &DataType::Float32).unwrap();
            let arrow_res = cmp::lt(&converted, &arr2).unwrap();
            assert_eq!(
                res, arrow_res,
                "incorrect result on array of size {}",
                arr_len
            );
        }
    }

    #[test]
    fn test_llvm_f16_lt() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &DataType::Float16,
                false,
                &DataType::Float16,
                false,
                Predicate::Lt,
            )
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Float16Array::from(
                (0..arr_len)
                    .map(|_| half::f16::from_f32(rng.f32()))
                    .collect_vec(),
            );
            let arr2 = Float16Array::from(
                (0..arr_len)
                    .map(|_| half::f16::from_f32(rng.f32()))
                    .collect_vec(),
            );

            let res = f.call(&arr1, &arr2).unwrap();
            let arrow_res = cmp::lt(&arr1, &arr2).unwrap();
            assert_eq!(
                res, arrow_res,
                "incorrect result on array of size {}",
                arr_len
            );
        }
    }

    #[test]
    fn test_llvm_i64() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &DataType::Int64,
                false,
                &DataType::Int64,
                false,
                Predicate::Eq,
            )
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());
            let arr2 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());

            let res = f.call(&arr1, &arr2).unwrap();
            let arrow_res = cmp::eq(&arr1, &arr2).unwrap();
            assert_eq!(
                res, arrow_res,
                "incorrect result on array of size {}",
                arr_len
            );
        }
    }

    #[test]
    fn test_llvm_i32() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &DataType::Int32,
                false,
                &DataType::Int32,
                false,
                Predicate::Eq,
            )
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Int32Array::from((0..arr_len).map(|_| rng.i32(-10..10)).collect_vec());
            let arr2 = Int32Array::from((0..arr_len).map(|_| rng.i32(-10..10)).collect_vec());

            let res = f.call(&arr1, &arr2).unwrap();
            let arrow_res = cmp::eq(&arr1, &arr2).unwrap();
            assert_eq!(
                res, arrow_res,
                "incorrect result on array of size {}",
                arr_len
            );
        }
    }

    #[test]
    fn test_llvm_i16() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &DataType::Int16,
                false,
                &DataType::Int16,
                false,
                Predicate::Eq,
            )
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Int16Array::from((0..arr_len).map(|_| rng.i16(-10..10)).collect_vec());
            let arr2 = Int16Array::from((0..arr_len).map(|_| rng.i16(-10..10)).collect_vec());

            let res = f.call(&arr1, &arr2).unwrap();
            let arrow_res = cmp::eq(&arr1, &arr2).unwrap();
            assert_eq!(
                res, arrow_res,
                "incorrect result on array of size {}",
                arr_len
            );
        }
    }

    #[test]
    fn test_llvm_i8() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &DataType::Int8,
                false,
                &DataType::Int8,
                false,
                Predicate::Eq,
            )
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Int8Array::from((0..arr_len).map(|_| rng.i8(-10..10)).collect_vec());
            let arr2 = Int8Array::from((0..arr_len).map(|_| rng.i8(-10..10)).collect_vec());

            let res = f.call(&arr1, &arr2).unwrap();
            let arrow_res = cmp::eq(&arr1, &arr2).unwrap();
            assert_eq!(
                res, arrow_res,
                "incorrect result on array of size {}",
                arr_len
            );
        }
    }

    #[test]
    fn test_prim_sliced_i32() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let f = cg
            .primitive_primitive_cmp(
                &DataType::Int32,
                false,
                &DataType::Int32,
                true,
                Predicate::Eq,
            )
            .unwrap();

        let arr1 = Int32Array::from((0..1000).map(|_| rng.i32(-10..10)).collect_vec());
        let sliced = arr1.slice(10, 50);
        let res = f.call(&sliced, &Int32Array::from(vec![0])).unwrap();
        let arrow_res = cmp::eq(&sliced, &Int32Array::new_scalar(0)).unwrap();
        assert_eq!(res, arrow_res);
    }

    #[test]
    fn test_cast_prim_i32_to_prim_i64() {
        // Create a context
        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);

        // Generate some test data
        let mut rng = fastrand::Rng::with_seed(42);
        let int32_values: Vec<i32> = (0..1000).map(|_| rng.i32(..)).collect_vec();
        let int32_array = Int32Array::from(int32_values.clone());

        let src_dt = DataType::Int32;
        let tar_dt = DataType::Int64;

        let cf = codegen.cast_to_primitive(&src_dt, &tar_dt).unwrap();
        let result: Int64Array = cf.call(&int32_array).unwrap().as_primitive().clone();

        // Verify the results
        for i in 0..int32_values.len() {
            assert_eq!(result.value(i), int32_values[i] as i64);
        }
        assert_eq!(result.len(), int32_array.len());
    }

    #[test]
    fn test_cast_prim_i16_to_prim_i64() {
        // Create a context
        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);

        // Generate some test data
        let mut rng = fastrand::Rng::with_seed(42);
        let int16_values: Vec<i16> = (0..1000).map(|_| rng.i16(..)).collect_vec();
        let int16_array = Int16Array::from(int16_values.clone());

        let src_dt = DataType::Int16;
        let tar_dt = DataType::Int64;

        let cf = codegen.cast_to_primitive(&src_dt, &tar_dt).unwrap();
        let result: Int64Array = cf.call(&int16_array).unwrap().as_primitive().clone();

        // Verify the results
        for i in 0..int16_values.len() {
            assert_eq!(result.value(i), int16_values[i] as i64);
        }
        assert_eq!(result.len(), int16_array.len());
    }

    #[test]
    fn test_cast_prim_i16_to_prim_i8() {
        // Create a context
        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);

        // Generate some test data
        let mut rng = fastrand::Rng::with_seed(42);
        let int16_values: Vec<i16> = (0..1000).map(|_| rng.i16(-100..100)).collect_vec();
        let int16_array = Int16Array::from(int16_values.clone());

        let src_dt = DataType::Int16;
        let tar_dt = DataType::Int8;

        let cf = codegen.cast_to_primitive(&src_dt, &tar_dt).unwrap();
        let result: Int8Array = cf.call(&int16_array).unwrap().as_primitive().clone();

        // Verify the results
        for i in 0..int16_values.len() {
            assert_eq!(result.value(i), int16_values[i] as i8);
        }
        assert_eq!(result.len(), int16_array.len());
    }

    #[test]
    fn test_cast_prim_u16_to_prim_u8() {
        // Create a context
        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);

        // Generate some test data
        let mut rng = fastrand::Rng::with_seed(42);
        let int16_values: Vec<u16> = (0..1000).map(|_| rng.u16(0..256)).collect_vec();
        let int16_array = UInt16Array::from(int16_values.clone());

        let src_dt = DataType::UInt16;
        let tar_dt = DataType::UInt8;

        let cf = codegen.cast_to_primitive(&src_dt, &tar_dt).unwrap();
        let result: UInt8Array = cf.call(&int16_array).unwrap().as_primitive().clone();

        // Verify the results
        for i in 0..int16_values.len() {
            assert_eq!(result.value(i), int16_values[i] as u8);
        }
        assert_eq!(result.len(), int16_array.len());
    }

    #[test]
    fn test_cast_prim_u16_to_prim_u32() {
        // Create a context
        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);

        // Generate some test data
        let mut rng = fastrand::Rng::with_seed(42);
        let int16_values: Vec<u16> = (0..1000).map(|_| rng.u16(..)).collect_vec();
        let int16_array = UInt16Array::from(int16_values.clone());

        let src_dt = DataType::UInt16;
        let tar_dt = DataType::UInt32;

        let cf = codegen.cast_to_primitive(&src_dt, &tar_dt).unwrap();
        let result: UInt32Array = cf.call(&int16_array).unwrap().as_primitive().clone();

        // Verify the results
        for i in 0..int16_values.len() {
            assert_eq!(result.value(i), int16_values[i] as u32);
        }
        assert_eq!(result.len(), int16_array.len());
    }
}
