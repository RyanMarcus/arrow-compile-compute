use std::{
    ffi::c_void,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use arrow_array::{
    cast::AsArray,
    types::{
        ArrowDictionaryKeyType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
        UInt64Type, UInt8Type,
    },
    ArrayRef, DictionaryArray,
};
use inkwell::{values::PointerValue, AddressSpace};
use repr_offset::ReprOffset;

use crate::{
    compiled_kernels::ht::{
        generate_hash_func_named, generate_lookup_or_insert_named, TicketTable,
    },
    compiled_writers::{
        AnyRuntime, AnyWriter, AnyWriterEmitter, DictionaryKeyType, PrimitiveWriter, Writer,
        WriterCodegen, WriterEmitter, WriterRuntime,
    },
    declare_blocks, increment_pointer, ArrowKernelError, PrimitiveType,
};

pub struct DictionaryWriter {
    /// for naming hash functions to avoid collisions with multiple dict writers in the same kernel
    helper_id: usize,
    key_type: DictionaryKeyType,
    value_type: PrimitiveType,
    keys: PrimitiveWriter,
    values: Box<AnyWriter>,
}

impl DictionaryWriter {
    pub fn compile(
        key_type: DictionaryKeyType,
        value_type: PrimitiveType,
        values: AnyWriter,
    ) -> Result<Self, ArrowKernelError> {
        if matches!(value_type, PrimitiveType::List(_, _)) {
            return Err(ArrowKernelError::InternalError(
                "dictionary writer does not support list values".into(),
            ));
        }
        Ok(Self {
            helper_id: DICTIONARY_HELPER_ID.fetch_add(1, Ordering::Relaxed),
            key_type,
            value_type,
            keys: PrimitiveWriter::compile(key_type.primitive_type())?,
            values: Box::new(values),
        })
    }

    fn get_child_ptr<'ctx>(
        codegen: WriterCodegen<'ctx, '_>,
        runtime_ptr: PointerValue<'ctx>,
        offset: usize,
        name: &str,
    ) -> PointerValue<'ctx> {
        codegen
            .builder
            .build_load(
                codegen.ctx.ptr_type(AddressSpace::default()),
                increment_pointer!(codegen.ctx, codegen.builder, runtime_ptr, offset),
                name,
            )
            .unwrap()
            .into_pointer_value()
    }
}

impl Writer for DictionaryWriter {
    fn allocate(&self, size: usize) -> AnyRuntime {
        let mut runtime = DictionaryWriterRuntime {
            table: TicketTable::new(
                size.saturating_mul(2).max(1),
                self.value_type.as_arrow_type(),
                self.key_type.data_type(),
            ),
            key_runtime_ptr: std::ptr::null_mut(),
            value_runtime_ptr: std::ptr::null_mut(),
            num_values: 0,
            keys: Box::new(self.keys.allocate(size)),
            values: Box::new(self.values.allocate(size)),
            key_type: self.key_type,
        };
        runtime.key_runtime_ptr = runtime.keys.as_ptr();
        runtime.value_runtime_ptr = runtime.values.as_ptr();
        runtime.into()
    }

    fn llvm_init<'ctx, 'borrow>(
        &self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
    ) {
        self.keys.llvm_init(
            codegen,
            Self::get_child_ptr(
                codegen,
                runtime_ptr,
                DictionaryWriterRuntime::OFFSET_KEY_RUNTIME_PTR,
                "dictionary_key_runtime",
            ),
        );
        self.values.llvm_init(
            codegen,
            Self::get_child_ptr(
                codegen,
                runtime_ptr,
                DictionaryWriterRuntime::OFFSET_VALUE_RUNTIME_PTR,
                "dictionary_value_runtime",
            ),
        );

        generate_hash_func_named(
            codegen.ctx,
            codegen.module,
            self.value_type,
            &format!("dictionary_writer_hash_{}", self.helper_id),
        );
        let table = TicketTable::new(
            0,
            self.value_type.as_arrow_type(),
            self.key_type.data_type(),
        );
        generate_lookup_or_insert_named(
            codegen.ctx,
            codegen.module,
            &table,
            &format!("dictionary_writer_lookup_{}", self.helper_id),
        );
    }

    fn llvm_write<'ctx, 'borrow, F>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        f: F,
    ) -> Result<(), ArrowKernelError>
    where
        F: Fn(&mut AnyWriterEmitter<'ctx, 'borrow>) -> Result<(), ArrowKernelError>,
    {
        let mut emitter = DictionaryWriterEmitter {
            codegen,
            runtime_ptr,
            key_runtime_ptr: Self::get_child_ptr(
                codegen,
                runtime_ptr,
                DictionaryWriterRuntime::OFFSET_KEY_RUNTIME_PTR,
                "dictionary_key_runtime",
            ),
            value_runtime_ptr: Self::get_child_ptr(
                codegen,
                runtime_ptr,
                DictionaryWriterRuntime::OFFSET_VALUE_RUNTIME_PTR,
                "dictionary_value_runtime",
            ),
            keys: &self.keys,
            values: self.values.as_ref(),
            hash_func: codegen
                .module
                .get_function(&format!("dictionary_writer_hash_{}", self.helper_id))
                .unwrap(),
            lookup_func: codegen
                .module
                .get_function(&format!("dictionary_writer_lookup_{}", self.helper_id))
                .unwrap(),
            used: false,
        }
        .into();
        f(&mut emitter)
    }
}

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct DictionaryWriterRuntime {
    table: TicketTable,
    key_runtime_ptr: *mut c_void,
    value_runtime_ptr: *mut c_void,
    /// Includes values appended after the ticket table becomes full.
    num_values: u64,
    keys: Box<AnyRuntime>,
    values: Box<AnyRuntime>,
    key_type: DictionaryKeyType,
}

impl DictionaryWriterRuntime {
    fn into_typed_array<K: ArrowDictionaryKeyType>(
        self,
        len: usize,
    ) -> Result<ArrayRef, ArrowKernelError> {
        let keys = self.keys.to_array(len)?;
        let keys = keys.as_primitive::<K>().clone();
        let values = self.values.to_array(self.num_values as usize)?;
        let array = DictionaryArray::<K>::try_new(keys, values).map_err(|error| {
            ArrowKernelError::InternalError(format!(
                "dictionary writer produced an invalid array: {error}"
            ))
        })?;
        Ok(Arc::new(array))
    }
}

impl WriterRuntime for DictionaryWriterRuntime {
    fn as_ptr(&mut self) -> *mut c_void {
        (self as *mut Self).cast()
    }

    fn reserve_for_additional(&mut self, count: usize) -> Result<(), ArrowKernelError> {
        self.keys.reserve_for_additional(count)?;
        self.values.reserve_for_additional(count)?;
        self.key_runtime_ptr = self.keys.as_ptr();
        self.value_runtime_ptr = self.values.as_ptr();
        Ok(())
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn to_array(self, len: usize) -> Result<ArrayRef, ArrowKernelError> {
        match self.key_type {
            DictionaryKeyType::Int8 => self.into_typed_array::<Int8Type>(len),
            DictionaryKeyType::Int16 => self.into_typed_array::<Int16Type>(len),
            DictionaryKeyType::Int32 => self.into_typed_array::<Int32Type>(len),
            DictionaryKeyType::Int64 => self.into_typed_array::<Int64Type>(len),
            DictionaryKeyType::UInt8 => self.into_typed_array::<UInt8Type>(len),
            DictionaryKeyType::UInt16 => self.into_typed_array::<UInt16Type>(len),
            DictionaryKeyType::UInt32 => self.into_typed_array::<UInt32Type>(len),
            DictionaryKeyType::UInt64 => self.into_typed_array::<UInt64Type>(len),
        }
    }
}

pub struct DictionaryWriterEmitter<'ctx, 'borrow> {
    codegen: WriterCodegen<'ctx, 'borrow>,
    runtime_ptr: PointerValue<'ctx>,
    key_runtime_ptr: PointerValue<'ctx>,
    value_runtime_ptr: PointerValue<'ctx>,
    keys: &'borrow PrimitiveWriter,
    values: &'borrow AnyWriter,
    hash_func: inkwell::values::FunctionValue<'ctx>,
    lookup_func: inkwell::values::FunctionValue<'ctx>,
    used: bool,
}

static DICTIONARY_HELPER_ID: AtomicUsize = AtomicUsize::new(0);

impl<'ctx, 'borrow> WriterEmitter<'ctx, 'borrow> for DictionaryWriterEmitter<'ctx, 'borrow> {
    fn emit(
        &mut self,
        value: inkwell::values::BasicValueEnum<'ctx>,
    ) -> Result<(), ArrowKernelError> {
        if self.used {
            return Err(ArrowKernelError::InternalError(
                "emit called on non-empty dictionary emitter".into(),
            ));
        }
        self.used = true;

        let hash = self
            .codegen
            .builder
            .build_call(self.hash_func, &[value.into()], "dictionary_hash")
            .unwrap()
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();
        let status_ptr = self
            .codegen
            .builder
            .build_alloca(self.codegen.ctx.i8_type(), "dictionary_status")
            .unwrap();
        let table_ptr = increment_pointer!(
            self.codegen.ctx,
            self.codegen.builder,
            self.runtime_ptr,
            DictionaryWriterRuntime::OFFSET_TABLE
        );
        let ticket = self
            .codegen
            .builder
            .build_call(
                self.lookup_func,
                &[
                    table_ptr.into(),
                    value.into(),
                    hash.into(),
                    status_ptr.into(),
                ],
                "dictionary_ticket",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_basic();
        let status = self
            .codegen
            .builder
            .build_load(self.codegen.ctx.i8_type(), status_ptr, "dictionary_status")
            .unwrap()
            .into_int_value();
        let num_values_ptr = increment_pointer!(
            self.codegen.ctx,
            self.codegen.builder,
            self.runtime_ptr,
            DictionaryWriterRuntime::OFFSET_NUM_VALUES
        );

        let func = self
            .codegen
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();
        declare_blocks!(
            self.codegen.ctx,
            func,
            dictionary_existing,
            dictionary_new,
            dictionary_full,
            dictionary_exit
        );
        self.codegen
            .builder
            .build_switch(
                status,
                dictionary_full,
                &[
                    (self.codegen.ctx.i8_type().const_zero(), dictionary_existing),
                    (
                        self.codegen.ctx.i8_type().const_int(1, false),
                        dictionary_new,
                    ),
                ],
            )
            .unwrap();

        self.codegen.builder.position_at_end(dictionary_existing);
        self.keys
            .llvm_write(self.codegen, self.key_runtime_ptr, |e| e.emit(ticket))?;
        self.codegen
            .builder
            .build_unconditional_branch(dictionary_exit)
            .unwrap();

        self.codegen.builder.position_at_end(dictionary_new);
        self.keys
            .llvm_write(self.codegen, self.key_runtime_ptr, |e| e.emit(ticket))?;
        self.values
            .llvm_write(self.codegen, self.value_runtime_ptr, |e| e.emit(value))?;
        let num_values = self
            .codegen
            .builder
            .build_load(
                self.codegen.ctx.i64_type(),
                num_values_ptr,
                "dictionary_num_values",
            )
            .unwrap()
            .into_int_value();
        let next_num_values = self
            .codegen
            .builder
            .build_int_add(
                num_values,
                self.codegen.ctx.i64_type().const_int(1, false),
                "dictionary_next_num_values",
            )
            .unwrap();
        self.codegen
            .builder
            .build_store(num_values_ptr, next_num_values)
            .unwrap();
        self.codegen
            .builder
            .build_unconditional_branch(dictionary_exit)
            .unwrap();

        self.codegen.builder.position_at_end(dictionary_full);
        let num_values = self
            .codegen
            .builder
            .build_load(
                self.codegen.ctx.i64_type(),
                num_values_ptr,
                "dictionary_num_values",
            )
            .unwrap()
            .into_int_value();
        let fallback_ticket = self
            .codegen
            .builder
            .build_int_truncate_or_bit_cast(
                num_values,
                self.keys
                    .primitive_type()
                    .llvm_type(self.codegen.ctx)
                    .into_int_type(),
                "dictionary_fallback_ticket",
            )
            .unwrap();
        self.values
            .llvm_write(self.codegen, self.value_runtime_ptr, |e| e.emit(value))?;
        self.keys
            .llvm_write(self.codegen, self.key_runtime_ptr, |e| {
                e.emit(fallback_ticket.into())
            })?;
        let next_num_values = self
            .codegen
            .builder
            .build_int_add(
                num_values,
                self.codegen.ctx.i64_type().const_int(1, false),
                "dictionary_next_num_values",
            )
            .unwrap();
        self.codegen
            .builder
            .build_store(num_values_ptr, next_num_values)
            .unwrap();
        self.codegen
            .builder
            .build_unconditional_branch(dictionary_exit)
            .unwrap();

        self.codegen.builder.position_at_end(dictionary_exit);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int8Type, BinaryArray};
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};

    use crate::{
        compiled_kernels::link_req_helpers,
        compiled_writers::{
            DictionaryKeyType, Writer, WriterCodegen, WriterEmitter, WriterRuntime, WriterSpec,
        },
        declare_blocks, PrimitiveType,
    };

    #[test]
    fn dictionary_writer_composes_and_falls_back_when_table_is_full() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers_dictionary_writer");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let append_func = llvm_mod.add_function(
            "append",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );
        declare_blocks!(ctx, append_func, entry);
        build.position_at_end(entry);

        let writer = WriterSpec::Dictionary(DictionaryKeyType::Int8, Box::new(WriterSpec::String))
            .compile()
            .unwrap();
        let runtime_ptr = append_func.get_nth_param(0).unwrap().into_pointer_value();
        let codegen = WriterCodegen {
            ctx: &ctx,
            module: &llvm_mod,
            builder: &build,
        };
        let string_type = PrimitiveType::P64x2.llvm_type(&ctx).into_struct_type();
        let input = ["alpha", "beta", "alpha", "gamma", "beta"];

        writer.llvm_init(codegen, runtime_ptr);
        for value in input {
            let start = value.as_ptr();
            let end = start.wrapping_add(value.len());
            let value = string_type.const_named_struct(&[
                ctx.i64_type()
                    .const_int(start as usize as u64, false)
                    .const_to_pointer(ptr_type)
                    .into(),
                ctx.i64_type()
                    .const_int(end as usize as u64, false)
                    .const_to_pointer(ptr_type)
                    .into(),
            ]);
            writer
                .llvm_write(codegen, runtime_ptr, |emitter| {
                    emitter.emit(value.as_basic_value_enum())
                })
                .unwrap();
        }
        build.build_return(None).unwrap();

        llvm_mod.verify().unwrap();

        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&llvm_mod, &ee).unwrap();
        let append = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(
                append_func.get_name().to_str().unwrap(),
            )
            .unwrap()
        };
        // Allocating one row creates a two-slot hash table. Reserving grows the
        // child writers but intentionally leaves the table at two slots, so
        // the third distinct value exercises the full-table fallback.
        let mut runtime = writer.allocate(1);
        runtime.reserve_for_additional(input.len()).unwrap();
        unsafe {
            append.call(runtime.as_ptr());
        }
        runtime.reserve_for_additional(input.len()).unwrap();
        unsafe {
            append.call(runtime.as_ptr());
        }

        let array = runtime.to_array_ref().unwrap();
        let dictionary = array.as_dictionary::<Int8Type>();
        assert_eq!(dictionary.keys().values(), &[0, 1, 0, 2, 1, 0, 1, 0, 3, 1]);
        let values = dictionary
            .values()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let values: Vec<&str> = values
            .iter()
            .map(|value| std::str::from_utf8(value.unwrap()).unwrap())
            .collect();
        assert_eq!(values, ["alpha", "beta", "gamma", "gamma"]);
    }
}
