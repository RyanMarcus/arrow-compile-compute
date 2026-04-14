use std::{ffi::c_void, sync::Arc};

use arrow_array::{
    cast::AsArray,
    types::{
        ArrowDictionaryKeyType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
        UInt64Type, UInt8Type,
    },
    ArrayRef, DictionaryArray,
};
use inkwell::{
    module::Linkage,
    values::{FunctionValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use super::{
    ArrayOutput, DictionaryKeyType, LeafWriter, LeafWriterAllocation, PrimitiveArrayWriter,
    WriterAllocation, WriterSpec,
};
use crate::{
    compiled_kernels::ht::{generate_hash_func, generate_lookup_or_insert, TicketTable},
    declare_blocks, increment_pointer, PrimitiveType,
};

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct DictAllocation {
    tt: TicketTable,
    keys_ptr: *mut c_void,
    values_ptr: *mut c_void,
    keys: Box<ArrayOutput>,
    values: Box<WriterAllocation>,
    key_type: DictionaryKeyType,
}

impl DictAllocation {
    pub fn allocate(
        expected_count: usize,
        key_type: DictionaryKeyType,
        value_spec: &WriterSpec,
    ) -> Self {
        let mut keys = Box::new(PrimitiveArrayWriter::allocate(
            expected_count,
            key_type.primitive_type(),
        ));
        let mut values = Box::new(value_spec.allocate(expected_count));
        Self {
            tt: TicketTable::new(
                expected_count * 2,
                value_spec.storage_type().as_arrow_type(),
                key_type.data_type(),
            ),
            keys_ptr: keys.get_ptr(),
            values_ptr: values.get_ptr(),
            keys,
            values,
            key_type,
        }
    }

    pub fn get_ptr(&mut self) -> *mut c_void {
        self as *mut Self as *mut c_void
    }

    pub fn reserve_for_additional(&mut self, count: usize) {
        self.keys.reserve_for_additional(count);
        self.values.reserve_for_additional(count);
        self.keys_ptr = self.keys.get_ptr();
        self.values_ptr = self.values.get_ptr();
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn into_array_ref_with_len(
        self,
        len: usize,
        nulls: Option<arrow_buffer::NullBuffer>,
    ) -> ArrayRef {
        match self.key_type {
            DictionaryKeyType::Int8 => Arc::new(self.into_typed_array::<Int8Type>(len, nulls)),
            DictionaryKeyType::Int16 => Arc::new(self.into_typed_array::<Int16Type>(len, nulls)),
            DictionaryKeyType::Int32 => Arc::new(self.into_typed_array::<Int32Type>(len, nulls)),
            DictionaryKeyType::Int64 => Arc::new(self.into_typed_array::<Int64Type>(len, nulls)),
            DictionaryKeyType::UInt8 => Arc::new(self.into_typed_array::<UInt8Type>(len, nulls)),
            DictionaryKeyType::UInt16 => Arc::new(self.into_typed_array::<UInt16Type>(len, nulls)),
            DictionaryKeyType::UInt32 => Arc::new(self.into_typed_array::<UInt32Type>(len, nulls)),
            DictionaryKeyType::UInt64 => Arc::new(self.into_typed_array::<UInt64Type>(len, nulls)),
        }
    }

    pub fn into_array_ref(self, nulls: Option<arrow_buffer::NullBuffer>) -> ArrayRef {
        let len = self.len();
        self.into_array_ref_with_len(len, nulls)
    }

    fn into_typed_array<K: ArrowDictionaryKeyType>(
        self,
        len: usize,
        nulls: Option<arrow_buffer::NullBuffer>,
    ) -> DictionaryArray<K> {
        let keys = (*self.keys).to_array(len, nulls);
        let keys = keys.as_primitive::<K>().clone();
        let values = self.values.into_array_ref_with_len(self.tt.len(), None);
        unsafe { DictionaryArray::<K>::new_unchecked(keys, values) }
    }
}

pub struct DictWriter<'a> {
    ht_ptr: PointerValue<'a>,
    ingest_func: FunctionValue<'a>,
}

impl<'a> DictWriter<'a> {
    pub fn llvm_init(
        ctx: &'a inkwell::context::Context,
        llvm_mod: &inkwell::module::Module<'a>,
        build: &inkwell::builder::Builder<'a>,
        key_type: DictionaryKeyType,
        value_spec: &WriterSpec,
        alloc_ptr: inkwell::values::PointerValue<'a>,
    ) -> Self {
        match key_type {
            DictionaryKeyType::Int8 => {
                Self::llvm_init_typed::<Int8Type>(ctx, llvm_mod, build, value_spec, alloc_ptr)
            }
            DictionaryKeyType::Int16 => {
                Self::llvm_init_typed::<Int16Type>(ctx, llvm_mod, build, value_spec, alloc_ptr)
            }
            DictionaryKeyType::Int32 => {
                Self::llvm_init_typed::<Int32Type>(ctx, llvm_mod, build, value_spec, alloc_ptr)
            }
            DictionaryKeyType::Int64 => {
                Self::llvm_init_typed::<Int64Type>(ctx, llvm_mod, build, value_spec, alloc_ptr)
            }
            DictionaryKeyType::UInt8 => {
                Self::llvm_init_typed::<UInt8Type>(ctx, llvm_mod, build, value_spec, alloc_ptr)
            }
            DictionaryKeyType::UInt16 => {
                Self::llvm_init_typed::<UInt16Type>(ctx, llvm_mod, build, value_spec, alloc_ptr)
            }
            DictionaryKeyType::UInt32 => {
                Self::llvm_init_typed::<UInt32Type>(ctx, llvm_mod, build, value_spec, alloc_ptr)
            }
            DictionaryKeyType::UInt64 => {
                Self::llvm_init_typed::<UInt64Type>(ctx, llvm_mod, build, value_spec, alloc_ptr)
            }
        }
    }

    pub fn llvm_ingest(
        &self,
        _ctx: &'a inkwell::context::Context,
        build: &inkwell::builder::Builder<'a>,
        val: inkwell::values::BasicValueEnum<'a>,
    ) {
        build
            .build_call(
                self.ingest_func,
                &[self.ht_ptr.into(), val.into()],
                "dict_ingest",
            )
            .unwrap();
    }

    pub fn llvm_flush(
        &self,
        _ctx: &'a inkwell::context::Context,
        _build: &inkwell::builder::Builder<'a>,
    ) {
        // no-op for dictionary writers
    }

    fn llvm_init_typed<K: ArrowDictionaryKeyType>(
        ctx: &'a inkwell::context::Context,
        llvm_mod: &inkwell::module::Module<'a>,
        build: &inkwell::builder::Builder<'a>,
        value_spec: &WriterSpec,
        alloc_ptr: inkwell::values::PointerValue<'a>,
    ) -> Self {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let value_type = value_spec.storage_type();
        let key_writer = PrimitiveArrayWriter::llvm_init(
            ctx,
            llvm_mod,
            build,
            PrimitiveType::for_arrow_type(&K::DATA_TYPE),
            build
                .build_load(
                    ptr_type,
                    increment_pointer!(ctx, build, alloc_ptr, DictAllocation::OFFSET_KEYS_PTR),
                    "keys_ptr",
                )
                .unwrap()
                .into_pointer_value(),
        );
        let value_writer_ptr = build
            .build_load(
                ptr_type,
                increment_pointer!(ctx, build, alloc_ptr, DictAllocation::OFFSET_VALUES_PTR),
                "values_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let value_writer = value_spec.llvm_init(ctx, llvm_mod, build, value_writer_ptr);

        let dummy_ht = TicketTable::new(0, value_type.as_arrow_type(), K::DATA_TYPE);
        let hash_func = generate_hash_func(ctx, llvm_mod, value_type);
        let ht_lookup = generate_lookup_or_insert(ctx, llvm_mod, &dummy_ht);
        let ht_ptr = increment_pointer!(ctx, build, alloc_ptr, DictAllocation::OFFSET_TT);
        let i8_type = ctx.i8_type();

        let ingest_func = {
            let b = ctx.create_builder();
            let func_type = ctx
                .bool_type()
                .fn_type(&[ptr_type.into(), value_type.llvm_type(ctx).into()], false);
            let func = llvm_mod.add_function(
                &format!("dict_writer_ingest_{}_{}", K::DATA_TYPE, value_type),
                func_type,
                Some(Linkage::Private),
            );

            let ht_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
            let value = func.get_nth_param(1).unwrap();

            declare_blocks!(ctx, func, entry, is_new, not_new, table_full);
            b.position_at_end(entry);
            let hash = b
                .build_call(hash_func, &[value.into()], "hash")
                .unwrap()
                .try_as_basic_value()
                .unwrap_basic()
                .into_int_value();
            let is_new_ptr = b.build_alloca(ctx.i8_type(), "is_new_ptr").unwrap();
            let ticket_val = b
                .build_call(
                    ht_lookup,
                    &[ht_ptr.into(), value.into(), hash.into(), is_new_ptr.into()],
                    "ht_lookup",
                )
                .unwrap()
                .try_as_basic_value()
                .unwrap_basic()
                .into_int_value();
            let status = b
                .build_load(i8_type, is_new_ptr, "status")
                .unwrap()
                .into_int_value();
            b.build_switch(
                status,
                table_full,
                &[
                    (i8_type.const_int(0, false), not_new),
                    (i8_type.const_int(1, false), is_new),
                    (i8_type.const_int(2, false), table_full),
                ],
            )
            .unwrap();

            b.position_at_end(not_new);
            key_writer.llvm_ingest(ctx, &b, ticket_val.into());
            b.build_return(Some(&ctx.bool_type().const_int(1, false)))
                .unwrap();

            b.position_at_end(is_new);
            key_writer.llvm_ingest(ctx, &b, ticket_val.into());
            value_writer.llvm_ingest(ctx, &b, value);
            b.build_return(Some(&ctx.bool_type().const_int(1, false)))
                .unwrap();

            b.position_at_end(table_full);
            b.build_return(Some(&ctx.bool_type().const_int(0, false)))
                .unwrap();

            func
        };

        Self {
            ht_ptr,
            ingest_func,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int8Type, Int32Array};
    use inkwell::{context::Context, AddressSpace, OptimizationLevel};
    use itertools::Itertools;

    use crate::{
        compiled_kernels::link_req_helpers,
        compiled_writers::{DictionaryKeyType, WriterSpec},
        declare_blocks, PrimitiveType,
    };

    #[test]
    fn test_dict_writer() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("test_primitive_array_writer");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let spec = WriterSpec::Dictionary(
            DictionaryKeyType::Int8,
            Box::new(WriterSpec::Primitive(PrimitiveType::I32)),
        );

        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );

        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);
        let dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = spec.llvm_init(&ctx, &llvm_mod, &build, dest);

        let mut expected = Vec::new();
        for _ in 0..10 {
            for i in 1000..1010 {
                writer.llvm_ingest(&ctx, &build, ctx.i32_type().const_int(i, true).into());
                expected.push(i as i32);
            }
        }

        writer.llvm_flush(&ctx, &build);
        build.build_return(None).unwrap();

        llvm_mod.verify().unwrap();
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&llvm_mod, &ee).unwrap();

        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut data = spec.allocate(100);
        unsafe {
            f.call(data.get_ptr());
        }
        data.reserve_for_additional(100);
        unsafe {
            f.call(data.get_ptr());
        }
        expected.extend(expected.clone());
        let data = data.into_array_ref_with_len(200, None);
        let data = data
            .as_dictionary::<Int8Type>()
            .downcast_dict::<Int32Array>()
            .unwrap();
        let data: Vec<i32> = data.into_iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(data, expected);
    }

    #[test]
    fn test_dict_writer_to_array_ref_uses_logical_len() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("test_dict_writer_to_array_ref");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let spec = WriterSpec::Dictionary(
            DictionaryKeyType::Int8,
            Box::new(WriterSpec::Primitive(PrimitiveType::I32)),
        );

        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );

        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);
        let dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = spec.llvm_init(&ctx, &llvm_mod, &build, dest);

        let mut expected = Vec::new();
        for _ in 0..2 {
            for i in 1000..1003 {
                writer.llvm_ingest(&ctx, &build, ctx.i32_type().const_int(i, true).into());
                expected.push(i as i32);
            }
        }

        writer.llvm_flush(&ctx, &build);
        build.build_return(None).unwrap();

        llvm_mod.verify().unwrap();
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&llvm_mod, &ee).unwrap();

        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut data = spec.allocate(expected.len());
        unsafe {
            f.call(data.get_ptr());
        }

        let data = data.into_array_ref(None);
        let data = data
            .as_dictionary::<Int8Type>()
            .downcast_dict::<Int32Array>()
            .unwrap();
        let data: Vec<i32> = data.into_iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(data, expected);
    }
}
