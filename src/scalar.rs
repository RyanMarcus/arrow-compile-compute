use inkwell::{
    builder::Builder,
    module::Linkage,
    types::StructType,
    values::{FunctionValue, IntValue, PointerValue},
    AddressSpace, IntPredicate,
};

use crate::{CodeGen, PrimitiveType};

impl<'ctx> CodeGen<'ctx> {
    pub(crate) fn struct_for_iter_scalar(&self) -> StructType {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i64_type = self.context.i64_type();
        self.context.struct_type(
            &[
                ptr_type.into(), // value (singleton)
                i64_type.into(), // produced so far
                i64_type.into(), // total to produce
            ],
            false,
        )
    }

    pub(crate) fn initialize_iter_scalar<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        arr: PointerValue,
        len: IntValue,
    ) -> PointerValue<'a>
    where
        'ctx: 'a,
    {
        let i64_type = self.context.i64_type();
        let iter_type = self.struct_for_iter_scalar();
        let ptr = builder.build_alloca(iter_type, "scalar_iter_ptr").unwrap();

        let val_ptr = builder
            .build_struct_gep(iter_type, ptr, 0, "val_ptr_ptr")
            .unwrap();
        builder.build_store(val_ptr, arr).unwrap();

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

    pub(crate) fn has_next_iter_scalar<'a>(
        &'a self,
        builder: &'a Builder<'a>,
        iter: PointerValue<'a>,
    ) -> IntValue<'a>
    where
        'ctx: 'a,
    {
        let i64_type = self.context.i64_type();
        let iter_type = self.struct_for_iter_scalar();

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

    pub(crate) fn gen_iter_scalar(&self, label: &str, prim_type: PrimitiveType) -> FunctionValue {
        let builder = self.context.create_builder();

        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let dtype = prim_type.llvm_type(self.context);
        let chunk_type = prim_type.llvm_vec_type(self.context, 64).unwrap();
        let iter_type = self.struct_for_iter_scalar();

        let fn_type = chunk_type.fn_type(
            &[
                ptr_type.into(), // iter struct
            ],
            false,
        );
        let function = self.module.add_function(
            &format!("{}_scalar_iter_chunk", label),
            fn_type,
            Some(Linkage::Private),
        );

        let entry = self.context.append_basic_block(function, "entry");

        let ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        builder.position_at_end(entry);
        let val_ptr_ptr = builder
            .build_struct_gep(iter_type, ptr, 0, "val_ptr_ptr")
            .unwrap();
        let val_ptr = builder
            .build_load(ptr_type, val_ptr_ptr, "val_ptr")
            .unwrap()
            .into_pointer_value();

        let value = builder.build_load(dtype, val_ptr, "value").unwrap();

        // broadcast value to all lanes
        let value_v = builder
            .build_insert_element(
                chunk_type.const_zero(),
                value,
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

        let idx_ptr = builder
            .build_struct_gep(iter_type, ptr, 1, "idx_ptr")
            .unwrap();
        let idx = builder
            .build_load(i64_type, idx_ptr, "idx")
            .unwrap()
            .into_int_value();
        let new_idx = builder
            .build_int_add(idx, i64_type.const_int(64, false), "new_idx")
            .unwrap();
        builder.build_store(idx_ptr, new_idx).unwrap();

        builder.build_return(Some(&value_v)).unwrap();
        function
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Datum, Int32Array};
    use arrow_schema::DataType;
    use inkwell::context::Context;

    use crate::{CodeGen, Predicate};

    #[test]
    fn test_prim_scalar_eq() {
        let arr = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let sca = Int32Array::new_scalar(2);

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
        let mask = f.call(&arr, sca.get().0).unwrap();
        let arrow_result = arrow_ord::cmp::eq(&arr, &sca).unwrap();
        assert_eq!(arrow_result, mask);
    }
}
