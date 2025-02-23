use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    intrinsics::Intrinsic,
    module::Linkage,
    types::{BasicType, StructType},
    values::{BasicValue, FunctionValue, IntValue, PointerValue},
    AddressSpace,
};

use crate::{declare_blocks, CodeGen, PrimitiveType};

impl<'ctx> CodeGen<'ctx> {
    pub(crate) fn struct_for_iter_dict(&self) -> StructType {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        self.context
            .struct_type(&[ptr_type.into(), ptr_type.into()], false)
    }

    pub(crate) fn initialize_iter_dict(
        &self,
        builder: &Builder<'ctx>,
        ptr: PointerValue,
        ktype: &DataType,
        vtype: &DataType,
        len: IntValue,
    ) -> PointerValue {
        // `ptr` points to two other pointers, the first for the keys and the
        // second for the values
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i64_type = self.context.i64_type();
        let arr_keys = builder
            .build_load(ptr_type, ptr, "key_ptr")
            .unwrap()
            .into_pointer_value();
        let inc_ptr = self.increment_pointer(builder, ptr, 8, i64_type.const_int(1, false));
        let arr_vals = builder
            .build_load(ptr_type, inc_ptr, "val_ptr")
            .unwrap()
            .into_pointer_value();

        let iter_type = self.struct_for_iter_dict();
        let ptr = builder.build_alloca(iter_type, "dict_iter_ptr").unwrap();
        let key_iter = self.initialize_iter(builder, arr_keys, len, ktype);
        let key_ptr = builder
            .build_struct_gep(iter_type, ptr, 0, "dict_key_ptr")
            .unwrap();
        builder.build_store(key_ptr, key_iter).unwrap();

        assert!(
            vtype.is_primitive(),
            "current iteration code assumes values are primitive"
        );
        let val_iter = self.initialize_iter(builder, arr_vals, len, vtype);
        let val_ptr = builder
            .build_struct_gep(iter_type, ptr, 1, "dict_val_ptr")
            .unwrap();
        builder.build_store(val_ptr, val_iter).unwrap();

        ptr
    }

    pub(crate) fn has_next_iter_dict<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        iter: PointerValue<'a>,
    ) -> IntValue<'a>
    where
        'ctx: 'a,
    {
        let iter_type = self.struct_for_iter_dict();
        let ptr_type = self.context.ptr_type(AddressSpace::default());

        let prim_ptr_ptr = builder
            .build_struct_gep(iter_type, iter, 0, "prim_ptr")
            .unwrap();
        let prim_ptr = builder
            .build_load(ptr_type, prim_ptr_ptr, "prim_ptr")
            .unwrap()
            .into_pointer_value();

        self.has_next_iter_primitive(builder, prim_ptr)
    }

    pub(crate) fn gen_dict_primitive(
        &self,
        label: &str,
        key_prim_type: PrimitiveType,
        value_prim_type: PrimitiveType,
    ) -> FunctionValue {
        let prim_f = self.gen_iter_primitive(&format!("{}_sub", label), key_prim_type, 64);
        let builder = self.context.create_builder();

        let bool_type = self.context.bool_type();
        let i64_type = self.context.i64_type();
        let i32_type = self.context.i32_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let ptr_int_chunk = i64_type.vec_type(64);
        let chunk_type = value_prim_type.llvm_vec_type(self.context, 64);
        let iter_type = self.struct_for_iter_primitive();

        let fn_type = chunk_type.fn_type(
            &[
                ptr_type.into(), // iter struct
            ],
            false,
        );
        let function = self.module.add_function(
            &format!("{}_dict_iter_chunk", label),
            fn_type,
            Some(Linkage::Private),
        );

        let entry = self.context.append_basic_block(function, "entry");
        let ptr = function.get_nth_param(0).unwrap().into_pointer_value();

        builder.position_at_end(entry);
        let values_iter_ptr_ptr = builder
            .build_struct_gep(iter_type, ptr, 1, "dict_prim_ptr")
            .unwrap();
        let values_iter_ptr = builder
            .build_load(ptr_type, values_iter_ptr_ptr, "values_ptr")
            .unwrap()
            .into_pointer_value();
        let values_ptr = builder
            .build_load(ptr_type, values_iter_ptr, "values_ptr")
            .unwrap()
            .into_pointer_value();
        let values_ptr_int = builder
            .build_ptr_to_int(values_ptr, i64_type, "values_ptr_int")
            .unwrap();
        let values_ptr_chunk = builder
            .build_insert_element(
                ptr_int_chunk.const_zero(),
                values_ptr_int,
                i64_type.const_zero(),
                "ptr_chunk",
            )
            .unwrap();
        let values_ptr_chunk = builder
            .build_shuffle_vector(
                values_ptr_chunk,
                ptr_int_chunk.get_undef(),
                ptr_int_chunk.const_zero(),
                "splatted",
            )
            .unwrap();

        let widths = builder
            .build_insert_element(
                i64_type.vec_type(64).const_zero(),
                i64_type.const_int(value_prim_type.width() as u64, false),
                i64_type.const_zero(),
                "widths",
            )
            .unwrap();
        let widths = builder
            .build_shuffle_vector(
                widths,
                i64_type.vec_type(64).get_undef(),
                i64_type.vec_type(64).const_zero(),
                "widths_splatted",
            )
            .unwrap();

        let prim_ptr_ptr = builder
            .build_struct_gep(iter_type, ptr, 0, "dict_prim_ptr")
            .unwrap();
        let prim_ptr = builder
            .build_load(ptr_type, prim_ptr_ptr, "prim_ptr")
            .unwrap();
        let key_chunk = builder
            .build_call(prim_f, &[prim_ptr.into()], "key_chunk")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_vector_value();
        let key_chunk =
            self.gen_convert_vec(&builder, key_chunk, key_prim_type, PrimitiveType::I64);

        let key_offsets = builder.build_int_mul(key_chunk, widths, "offsets").unwrap();
        let value_ptrs = builder
            .build_int_add(values_ptr_chunk, key_offsets, "value_ptrs")
            .unwrap();
        let value_ptrs = builder
            .build_int_to_ptr(value_ptrs, ptr_type.vec_type(64), "value_ptrs")
            .unwrap();

        let gather = Intrinsic::find("llvm.masked.gather").unwrap();
        let gather_f = gather
            .get_declaration(
                &self.module,
                &[chunk_type.into(), ptr_type.vec_type(64).into()],
            )
            .unwrap();

        let mask_int = i64_type.const_all_ones();
        let mask_vec = builder
            .build_bit_cast(mask_int, bool_type.vec_type(64), "mask")
            .unwrap()
            .into_vector_value();
        let passthru = chunk_type.const_zero();
        let gathered_values = builder
            .build_call(
                gather_f,
                &[
                    value_ptrs.into(),
                    i32_type.const_zero().into(),
                    mask_vec.into(),
                    passthru.into(),
                ],
                "gathered_values",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_vector_value();

        builder.build_return(Some(&gathered_values)).unwrap();
        function
    }

    pub(crate) fn gen_random_access_dict(
        &self,
        label: &str,
        ktype: &DataType,
        vtype: &DataType,
    ) -> FunctionValue {
        let key_getter = self.gen_random_access_for(&format!("{}_get_keys", label), ktype);
        let val_getter = self.gen_random_access_for(&format!("{}_get_values", label), vtype);

        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let iter_type = self.struct_for_iter_dict();
        let prim_type = PrimitiveType::for_arrow_type(vtype)
            .llvm_type(self.context)
            .as_basic_type_enum();

        let fn_type = prim_type.fn_type(
            &[
                ptr_type.into(), // iter struct
                i64_type.into(), // index
            ],
            false,
        );
        let builder = self.context.create_builder();

        let function = self.module.add_function(
            &format!("{}_dict_random_access", label),
            fn_type,
            Some(Linkage::Private),
        );

        declare_blocks!(self.context, function, entry);

        builder.position_at_end(entry);
        let iter_ptr = function
            .get_nth_param(0)
            .unwrap()
            .as_basic_value_enum()
            .into_pointer_value();
        let idx = function
            .get_nth_param(1)
            .unwrap()
            .as_basic_value_enum()
            .into_int_value();

        let key_iter_ptr_ptr = builder
            .build_struct_gep(iter_type, iter_ptr, 0, "key_iter_ptr")
            .unwrap();
        let value_iter_ptr_ptr = builder
            .build_struct_gep(iter_type, iter_ptr, 1, "value_iter_ptr")
            .unwrap();
        let key_iter_ptr = builder
            .build_load(ptr_type, key_iter_ptr_ptr, "key_iter_ptr")
            .unwrap();
        let value_iter_ptr = builder
            .build_load(ptr_type, value_iter_ptr_ptr, "value_iter_ptr")
            .unwrap();

        let key = builder
            .build_call(key_getter, &[key_iter_ptr.into(), idx.into()], "key")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        let key = builder
            .build_int_z_extend_or_bit_cast(key, i64_type, "key")
            .unwrap();
        let val = builder
            .build_call(val_getter, &[value_iter_ptr.into(), key.into()], "val")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();

        builder.build_return(Some(&val)).unwrap();
        function
    }
}

#[cfg(test)]
mod tests {

    use arrow_array::{cast::AsArray, BooleanArray, Int32Array, Int64Array};
    use arrow_ord::cmp;
    use arrow_schema::DataType;
    use inkwell::context::Context;
    use itertools::Itertools;

    use crate::{dictionary_data_type, CodeGen, Predicate};

    const SIZES_TO_TRY: &[usize] = &[0, 50, 64, 100, 128, 200, 2048, 2049];

    #[test]
    fn test_dict_prim_lt() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let dict_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int64));
        let f = cg
            .primitive_primitive_cmp(&dict_type, false, &DataType::Int64, false, Predicate::Lt)
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());
            let arr2 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());

            let arr1 = arrow_cast::cast(&arr1, &dict_type).unwrap();

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
    fn test_dict_dict_lt() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let dict_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int64));
        let f = cg
            .primitive_primitive_cmp(&dict_type, false, &dict_type, false, Predicate::Eq)
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());
            let arr2 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());

            let arr1 = arrow_cast::cast(&arr1, &dict_type).unwrap();
            let arr2 = arrow_cast::cast(&arr2, &dict_type).unwrap();

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
    fn test_dict8_dict16_lt() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let dict_type1 = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int64));
        let dict_type2 = DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Int64));
        let f = cg
            .primitive_primitive_cmp(&dict_type1, false, &dict_type2, false, Predicate::Eq)
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());
            let arr2 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());

            let arr1 = arrow_cast::cast(&arr1, &dict_type1).unwrap();
            let arr2 = arrow_cast::cast(&arr2, &dict_type2).unwrap();

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
    fn test_dict_dict_cast() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let dict_type1 = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int64));
        let dict_type2 = DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Int32));
        let f = cg
            .primitive_primitive_cmp(&dict_type1, false, &dict_type2, false, Predicate::Eq)
            .unwrap();

        for &arr_len in SIZES_TO_TRY {
            let arr1 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());
            let arr2 = Int64Array::from((0..arr_len).map(|_| rng.i64(-10..10)).collect_vec());

            let arr1 = arrow_cast::cast(&arr1, &dict_type1).unwrap();
            let arr2 = arrow_cast::cast(&arr2, &dict_type2).unwrap();

            let res = f.call(&arr1, &arr2).unwrap();
            let t_arr2 = arrow_cast::cast(&arr2, &dict_type1).unwrap();
            let arrow_res = cmp::eq(&arr1, &t_arr2).unwrap();
            assert_eq!(
                res, arrow_res,
                "incorrect result on array of size {}",
                arr_len
            );
        }
    }

    #[test]
    fn test_cast_dict_i32_to_prim_i32() {
        // Create a context
        let ctx = Context::create();
        let codegen = CodeGen::new(&ctx);

        // Generate some test data
        let mut rng = fastrand::Rng::with_seed(42);
        let int32_values: Vec<i32> = (0..1000).map(|_| rng.i8(0..10) as i32 + 100).collect_vec();
        let int32_array = Int32Array::from(int32_values.clone());

        let src_dt = dictionary_data_type(DataType::Int8, DataType::Int32);
        let tar_dt = DataType::Int32;

        let int32_dict_array = arrow_cast::cast(&int32_array, &src_dt).unwrap();

        let cf = codegen.cast_to_primitive(&src_dt, &tar_dt).unwrap();
        let result: Int32Array = cf.call(&int32_dict_array).unwrap().as_primitive().clone();

        // Verify the results
        for i in 0..int32_values.len() {
            assert_eq!(result.value(i), int32_values[i]);
        }
        assert_eq!(result.len(), int32_array.len());
    }

    #[test]
    fn test_dict_sliced_i32() {
        let mut rng = fastrand::Rng::with_seed(42);
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let dict_dt = dictionary_data_type(DataType::Int8, DataType::Int32);
        let f = cg
            .primitive_primitive_cmp(&dict_dt, false, &DataType::Int32, true, Predicate::Eq)
            .unwrap();

        let arr1 = Int32Array::from((0..1000).map(|_| rng.i32(-5..5)).collect_vec());
        let arr1 = arrow_cast::cast(&arr1, &dict_dt).unwrap();

        let sliced = arr1.slice(10, 50);
        let res = f.call(&sliced, &Int32Array::from(vec![0])).unwrap();
        let arrow_res = cmp::eq(&sliced, &Int32Array::new_scalar(0)).unwrap();
        assert_eq!(res, arrow_res);
    }

    #[test]
    fn test_dict_filter_simple_block() {
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let dict_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32));

        let f = cg.compile_filter_block(&dict_type).unwrap();

        let data = Int32Array::from(vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 10]);
        let data = arrow_cast::cast(&data, &dict_type).unwrap();

        let ba = BooleanArray::from(vec![
            true, false, false, true, false, true, false, false, false, true,
        ]);

        let true_result = Int32Array::from(vec![1, 2, 2, 10]);
        let our_filtered: Int32Array = f.call(&data, &ba).unwrap().as_primitive().clone();

        assert_eq!(our_filtered.len(), 4);
        assert_eq!(true_result, our_filtered);
    }

    #[test]
    fn test_dict_filter_simple_random_access() {
        let ctx = Context::create();
        let cg = CodeGen::new(&ctx);
        let dict_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32));

        let f = cg.compile_filter_random_access(&dict_type).unwrap();

        let data = Int32Array::from(vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 10]);
        let data = arrow_cast::cast(&data, &dict_type).unwrap();

        let ba = BooleanArray::from(vec![
            true, false, false, true, false, true, false, false, false, true,
        ]);

        let true_result = Int32Array::from(vec![1, 2, 2, 10]);
        let our_filtered: Int32Array = f.call(&data, &ba).unwrap().as_primitive().clone();

        assert_eq!(our_filtered.len(), 4);
        assert_eq!(true_result, our_filtered);
    }
}
