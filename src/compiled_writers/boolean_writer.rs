use std::{ffi::c_void, sync::Arc};

use arrow_array::{ArrayRef, BooleanArray};
use arrow_buffer::{BooleanBuffer, Buffer};
use inkwell::{
    values::{BasicValueEnum, PointerValue, VectorValue},
    AddressSpace, IntPredicate,
};
use repr_offset::ReprOffset;

use crate::{
    compiled_writers::{
        AnyRuntime, AnyWriterEmitter, Writer, WriterCodegen, WriterEmitter, WriterRuntime,
    },
    declare_blocks, increment_pointer, ArrowKernelError,
};

pub struct BooleanWriter;

impl BooleanWriter {
    pub fn compile() -> Self {
        Self
    }
}

impl Writer for BooleanWriter {
    fn allocate(&self, size: usize) -> AnyRuntime {
        let mut alloc = vec![0; size.div_ceil(8)];
        let alloc_ptr = alloc.as_mut_ptr();
        BooleanWriterRuntime {
            alloc_ptr,
            alloc,
            num_written: 0,
            buf: 0,
            buf_idx: 0,
        }
        .into()
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
        let mut emitter = BooleanWriterEmitter {
            codegen,
            runtime_ptr,
            used: false,
        }
        .into();
        f(&mut emitter)
    }

    fn llvm_write_multiple<'ctx, 'borrow>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        values: VectorValue<'ctx>,
    ) -> Result<(), ArrowKernelError> {
        let ctx = codegen.ctx;
        let builder = codegen.builder;

        if values.get_type().get_size() != 64 {
            for idx in 0..values.get_type().get_size() {
                let value = builder
                    .build_extract_element(
                        values,
                        ctx.i64_type().const_int(idx as u64, false),
                        "boolean_writer_multiple_value",
                    )
                    .unwrap();
                self.llvm_write(codegen, runtime_ptr, |emitter| emitter.emit(value))?;
            }
            return Ok(());
        }

        let i8_type = ctx.i8_type();
        let i64_type = ctx.i64_type();
        let i128_type = ctx.i128_type();
        let packed = builder
            .build_bit_cast(values, i64_type, "packed_boolean_values")
            .unwrap()
            .into_int_value();
        let alloc_ptr_ptr = increment_pointer!(
            ctx,
            builder,
            runtime_ptr,
            BooleanWriterRuntime::OFFSET_ALLOC_PTR
        );
        let buf_ptr =
            increment_pointer!(ctx, builder, runtime_ptr, BooleanWriterRuntime::OFFSET_BUF);
        let buf_idx_ptr = increment_pointer!(
            ctx,
            builder,
            runtime_ptr,
            BooleanWriterRuntime::OFFSET_BUF_IDX
        );
        let num_written_ptr = increment_pointer!(
            ctx,
            builder,
            runtime_ptr,
            BooleanWriterRuntime::OFFSET_NUM_WRITTEN
        );

        let alloc_ptr = builder
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                alloc_ptr_ptr,
                "boolean_alloc_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let buf_idx = builder
            .build_load(i8_type, buf_idx_ptr, "boolean_buf_idx")
            .unwrap()
            .into_int_value();

        let func = builder.get_insert_block().unwrap().get_parent().unwrap();
        declare_blocks!(
            ctx,
            func,
            boolean_block_aligned,
            boolean_block_unaligned,
            boolean_block_exit
        );
        let is_aligned = builder
            .build_int_compare(
                IntPredicate::EQ,
                buf_idx,
                i8_type.const_zero(),
                "boolean_block_is_aligned",
            )
            .unwrap();
        builder
            .build_conditional_branch(is_aligned, boolean_block_aligned, boolean_block_unaligned)
            .unwrap();

        builder.position_at_end(boolean_block_aligned);
        builder.build_store(alloc_ptr, packed).unwrap();
        builder
            .build_unconditional_branch(boolean_block_exit)
            .unwrap();

        builder.position_at_end(boolean_block_unaligned);
        let buf = builder
            .build_load(i8_type, buf_ptr, "boolean_buf")
            .unwrap()
            .into_int_value();
        let buf = builder
            .build_int_z_extend(buf, i128_type, "boolean_buf_i128")
            .unwrap();
        let packed = builder
            .build_int_z_extend(packed, i128_type, "packed_booleans_i128")
            .unwrap();
        let shift = builder
            .build_int_z_extend(buf_idx, i128_type, "boolean_buf_idx_i128")
            .unwrap();
        let packed = builder
            .build_left_shift(packed, shift, "shifted_packed_booleans")
            .unwrap();
        let combined = builder
            .build_or(buf, packed, "combined_packed_booleans")
            .unwrap();
        let completed = builder
            .build_int_truncate(combined, i64_type, "completed_boolean_bytes")
            .unwrap();
        builder.build_store(alloc_ptr, completed).unwrap();
        let remaining = builder
            .build_right_shift(
                combined,
                i128_type.const_int(64, false),
                false,
                "remaining_boolean_bits",
            )
            .unwrap();
        let remaining = builder
            .build_int_truncate(remaining, i8_type, "remaining_boolean_byte")
            .unwrap();
        builder.build_store(buf_ptr, remaining).unwrap();
        builder
            .build_unconditional_branch(boolean_block_exit)
            .unwrap();

        builder.position_at_end(boolean_block_exit);
        builder
            .build_store(
                alloc_ptr_ptr,
                increment_pointer!(ctx, builder, alloc_ptr, 8),
            )
            .unwrap();
        let num_written = builder
            .build_load(i64_type, num_written_ptr, "boolean_num_written")
            .unwrap()
            .into_int_value();
        let num_written = builder
            .build_int_add(
                num_written,
                i64_type.const_int(64, false),
                "boolean_new_num_written",
            )
            .unwrap();
        builder.build_store(num_written_ptr, num_written).unwrap();

        Ok(())
    }

    fn llvm_write_block<'ctx, 'borrow>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        values: VectorValue<'ctx>,
        logical_len: u32,
    ) -> Result<(), ArrowKernelError> {
        let packed_type = codegen.ctx.custom_width_int_type(logical_len);
        if values.get_type().get_size() != logical_len {
            return Err(ArrowKernelError::InternalError(format!(
                "boolean block has {} lanes for {logical_len} values",
                values.get_type().get_size()
            )));
        }
        let packed = codegen
            .builder
            .build_bit_cast(values, packed_type, "packed_boolean_block")
            .unwrap()
            .into_int_value();
        if !logical_len.is_multiple_of(8) {
            for idx in 0..logical_len {
                let shifted = codegen
                    .builder
                    .build_right_shift(
                        packed,
                        packed_type.const_int(idx as u64, false),
                        false,
                        "boolean_block_shifted",
                    )
                    .unwrap();
                let value = codegen
                    .builder
                    .build_int_truncate(shifted, codegen.ctx.bool_type(), "boolean_block_value")
                    .unwrap()
                    .into();
                self.llvm_write(codegen, runtime_ptr, |emitter| emitter.emit(value))?;
            }
            return Ok(());
        }

        let i8_type = codegen.ctx.i8_type();
        let i64_type = codegen.ctx.i64_type();
        let buf_idx_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            BooleanWriterRuntime::OFFSET_BUF_IDX
        );
        let buf_idx = codegen
            .builder
            .build_load(i8_type, buf_idx_ptr, "boolean_block_buf_idx")
            .unwrap()
            .into_int_value();
        let func = codegen
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();
        declare_blocks!(
            codegen.ctx,
            func,
            boolean_block_direct,
            boolean_block_unaligned,
            boolean_block_done
        );
        let aligned = codegen
            .builder
            .build_int_compare(
                IntPredicate::EQ,
                buf_idx,
                i8_type.const_zero(),
                "boolean_block_aligned",
            )
            .unwrap();
        codegen
            .builder
            .build_conditional_branch(aligned, boolean_block_direct, boolean_block_unaligned)
            .unwrap();

        codegen.builder.position_at_end(boolean_block_direct);
        let alloc_ptr_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            BooleanWriterRuntime::OFFSET_ALLOC_PTR
        );
        let alloc_ptr = codegen
            .builder
            .build_load(
                codegen.ctx.ptr_type(AddressSpace::default()),
                alloc_ptr_ptr,
                "boolean_block_alloc_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let store = codegen.builder.build_store(alloc_ptr, packed).unwrap();
        store.set_alignment(1).unwrap();
        codegen
            .builder
            .build_store(
                alloc_ptr_ptr,
                increment_pointer!(
                    codegen.ctx,
                    codegen.builder,
                    alloc_ptr,
                    logical_len as usize / 8
                ),
            )
            .unwrap();
        let num_written_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            BooleanWriterRuntime::OFFSET_NUM_WRITTEN
        );
        let num_written = codegen
            .builder
            .build_load(i64_type, num_written_ptr, "boolean_block_num_written")
            .unwrap()
            .into_int_value();
        let new_num_written = codegen
            .builder
            .build_int_add(
                num_written,
                i64_type.const_int(logical_len as u64, false),
                "boolean_block_new_num_written",
            )
            .unwrap();
        codegen
            .builder
            .build_store(num_written_ptr, new_num_written)
            .unwrap();
        codegen
            .builder
            .build_unconditional_branch(boolean_block_done)
            .unwrap();

        codegen.builder.position_at_end(boolean_block_unaligned);
        let wide_type = codegen.ctx.custom_width_int_type(logical_len + 8);
        let buf_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            BooleanWriterRuntime::OFFSET_BUF
        );
        let buf = codegen
            .builder
            .build_load(i8_type, buf_ptr, "boolean_block_buf")
            .unwrap()
            .into_int_value();
        let buf = codegen
            .builder
            .build_int_z_extend(buf, wide_type, "boolean_block_buf_wide")
            .unwrap();
        let packed = codegen
            .builder
            .build_int_z_extend(packed, wide_type, "boolean_block_values_wide")
            .unwrap();
        let shift = codegen
            .builder
            .build_int_z_extend(buf_idx, wide_type, "boolean_block_shift")
            .unwrap();
        let packed = codegen
            .builder
            .build_left_shift(packed, shift, "boolean_block_values_shifted")
            .unwrap();
        let combined = codegen
            .builder
            .build_or(buf, packed, "boolean_block_combined")
            .unwrap();
        let completed = codegen
            .builder
            .build_int_truncate(combined, packed_type, "boolean_block_completed")
            .unwrap();
        let alloc_ptr_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            BooleanWriterRuntime::OFFSET_ALLOC_PTR
        );
        let alloc_ptr = codegen
            .builder
            .build_load(
                codegen.ctx.ptr_type(AddressSpace::default()),
                alloc_ptr_ptr,
                "boolean_block_alloc_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let store = codegen.builder.build_store(alloc_ptr, completed).unwrap();
        store.set_alignment(1).unwrap();
        codegen
            .builder
            .build_store(
                alloc_ptr_ptr,
                increment_pointer!(
                    codegen.ctx,
                    codegen.builder,
                    alloc_ptr,
                    logical_len as usize / 8
                ),
            )
            .unwrap();
        let remaining = codegen
            .builder
            .build_right_shift(
                combined,
                wide_type.const_int(logical_len as u64, false),
                false,
                "boolean_block_remaining",
            )
            .unwrap();
        let remaining = codegen
            .builder
            .build_int_truncate(remaining, i8_type, "boolean_block_remaining_byte")
            .unwrap();
        codegen.builder.build_store(buf_ptr, remaining).unwrap();
        let num_written_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            BooleanWriterRuntime::OFFSET_NUM_WRITTEN
        );
        let num_written = codegen
            .builder
            .build_load(i64_type, num_written_ptr, "boolean_block_num_written")
            .unwrap()
            .into_int_value();
        let new_num_written = codegen
            .builder
            .build_int_add(
                num_written,
                i64_type.const_int(logical_len as u64, false),
                "boolean_block_new_num_written",
            )
            .unwrap();
        codegen
            .builder
            .build_store(num_written_ptr, new_num_written)
            .unwrap();
        codegen
            .builder
            .build_unconditional_branch(boolean_block_done)
            .unwrap();
        codegen.builder.position_at_end(boolean_block_done);
        Ok(())
    }
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct BooleanWriterRuntime {
    alloc_ptr: *mut u8,
    alloc: Vec<u8>,
    num_written: u64,
    buf: u8,
    buf_idx: u8,
}

impl WriterRuntime for BooleanWriterRuntime {
    fn as_ptr(&mut self) -> *mut c_void {
        (self as *mut Self).cast()
    }

    fn reserve_for_additional(&mut self, count: usize) -> Result<(), ArrowKernelError> {
        unsafe {
            let base = self.alloc.as_mut_ptr();
            let bytes_written = self.alloc_ptr.offset_from_unsigned(base);
            let bytes_to_preserve = bytes_written + usize::from(self.buf_idx > 0);
            self.alloc.set_len(bytes_to_preserve);
            self.alloc.resize(
                bytes_written + (usize::from(self.buf_idx) + count).div_ceil(8),
                0,
            );
            self.alloc_ptr = self.alloc.as_mut_ptr().add(bytes_written);
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.num_written as usize
    }

    fn to_array(mut self, len: usize) -> Result<ArrayRef, ArrowKernelError> {
        if self.buf_idx > 0 {
            unsafe {
                let byte_idx = self.alloc_ptr.offset_from_unsigned(self.alloc.as_ptr());
                self.alloc[byte_idx] = self.buf;
            }
        }
        let values = BooleanBuffer::new(Buffer::from(self.alloc), 0, len);
        Ok(Arc::new(BooleanArray::new(values, None)))
    }
}

pub struct BooleanWriterEmitter<'ctx, 'borrow> {
    codegen: WriterCodegen<'ctx, 'borrow>,
    runtime_ptr: PointerValue<'ctx>,
    used: bool,
}

impl<'ctx, 'borrow> WriterEmitter<'ctx, 'borrow> for BooleanWriterEmitter<'ctx, 'borrow> {
    fn emit(&mut self, val: BasicValueEnum<'ctx>) -> Result<(), ArrowKernelError> {
        if self.used {
            return Err(ArrowKernelError::InternalError(
                "emit called on non-empty boolean emitter".into(),
            ));
        }
        self.used = true;

        let val = val.into_int_value();
        if val.get_type().get_bit_width() != 1 {
            return Err(ArrowKernelError::InternalError(
                "boolean writer expected an i1 value".into(),
            ));
        }

        let i8_type = self.codegen.ctx.i8_type();
        let i64_type = self.codegen.ctx.i64_type();
        let buf_ptr = increment_pointer!(
            self.codegen.ctx,
            self.codegen.builder,
            self.runtime_ptr,
            BooleanWriterRuntime::OFFSET_BUF
        );
        let buf_idx_ptr = increment_pointer!(
            self.codegen.ctx,
            self.codegen.builder,
            self.runtime_ptr,
            BooleanWriterRuntime::OFFSET_BUF_IDX
        );
        let alloc_ptr_ptr = increment_pointer!(
            self.codegen.ctx,
            self.codegen.builder,
            self.runtime_ptr,
            BooleanWriterRuntime::OFFSET_ALLOC_PTR
        );
        let num_written_ptr = increment_pointer!(
            self.codegen.ctx,
            self.codegen.builder,
            self.runtime_ptr,
            BooleanWriterRuntime::OFFSET_NUM_WRITTEN
        );

        let buf_idx = self
            .codegen
            .builder
            .build_load(i8_type, buf_idx_ptr, "boolean_buf_idx")
            .unwrap()
            .into_int_value();
        let buf = self
            .codegen
            .builder
            .build_load(i8_type, buf_ptr, "boolean_buf")
            .unwrap()
            .into_int_value();
        let val = self
            .codegen
            .builder
            .build_int_z_extend(val, i8_type, "boolean_value_i8")
            .unwrap();
        let shifted = self
            .codegen
            .builder
            .build_left_shift(val, buf_idx, "boolean_shifted_value")
            .unwrap();
        let new_buf = self
            .codegen
            .builder
            .build_or(buf, shifted, "boolean_new_buf")
            .unwrap();
        self.codegen.builder.build_store(buf_ptr, new_buf).unwrap();

        let new_buf_idx = self
            .codegen
            .builder
            .build_int_add(buf_idx, i8_type.const_int(1, false), "boolean_new_buf_idx")
            .unwrap();
        self.codegen
            .builder
            .build_store(buf_idx_ptr, new_buf_idx)
            .unwrap();

        let num_written = self
            .codegen
            .builder
            .build_load(i64_type, num_written_ptr, "boolean_num_written")
            .unwrap()
            .into_int_value();
        let new_num_written = self
            .codegen
            .builder
            .build_int_add(
                num_written,
                i64_type.const_int(1, false),
                "boolean_new_num_written",
            )
            .unwrap();
        self.codegen
            .builder
            .build_store(num_written_ptr, new_num_written)
            .unwrap();

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
            boolean_byte_full,
            boolean_write_exit
        );
        let byte_full = self
            .codegen
            .builder
            .build_int_compare(
                IntPredicate::EQ,
                new_buf_idx,
                i8_type.const_int(8, false),
                "boolean_byte_is_full",
            )
            .unwrap();
        self.codegen
            .builder
            .build_conditional_branch(byte_full, boolean_byte_full, boolean_write_exit)
            .unwrap();

        self.codegen.builder.position_at_end(boolean_byte_full);
        let alloc_ptr = self
            .codegen
            .builder
            .build_load(
                self.codegen.ctx.ptr_type(AddressSpace::default()),
                alloc_ptr_ptr,
                "boolean_alloc_ptr",
            )
            .unwrap()
            .into_pointer_value();
        self.codegen
            .builder
            .build_store(alloc_ptr, new_buf)
            .unwrap();
        let next_alloc_ptr =
            increment_pointer!(self.codegen.ctx, self.codegen.builder, alloc_ptr, 1);
        self.codegen
            .builder
            .build_store(alloc_ptr_ptr, next_alloc_ptr)
            .unwrap();
        self.codegen
            .builder
            .build_store(buf_ptr, i8_type.const_zero())
            .unwrap();
        self.codegen
            .builder
            .build_store(buf_idx_ptr, i8_type.const_zero())
            .unwrap();
        self.codegen
            .builder
            .build_unconditional_branch(boolean_write_exit)
            .unwrap();

        self.codegen.builder.position_at_end(boolean_write_exit);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, BooleanArray};
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};

    use super::BooleanWriter;
    use crate::{
        compiled_writers::{
            AnyWriter, Writer, WriterCodegen, WriterEmitter, WriterRuntime, WriterSpec,
        },
        declare_blocks,
    };

    #[test]
    fn boolean_writer_spec_compiles_boolean_writer() {
        let spec = WriterSpec::for_data_type(&arrow_schema::DataType::Boolean);
        assert!(matches!(spec.compile().unwrap(), AnyWriter::Boolean(_)));
    }

    fn compile_write_function<'ctx>(
        ctx: &'ctx Context,
        values: &[bool],
    ) -> (
        inkwell::module::Module<'ctx>,
        inkwell::values::FunctionValue<'ctx>,
        BooleanWriter,
    ) {
        let llvm_mod = ctx.create_module("compiled_writers_boolean_writer");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );

        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);
        let dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = BooleanWriter::compile();
        let codegen = WriterCodegen {
            ctx,
            module: &llvm_mod,
            builder: &build,
        };

        writer.llvm_init(codegen, dest);
        for value in values {
            writer
                .llvm_write(codegen, dest, |emitter| {
                    emitter.emit(
                        ctx.bool_type()
                            .const_int(u64::from(*value), false)
                            .as_basic_value_enum(),
                    )
                })
                .unwrap();
        }
        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();

        (llvm_mod, func, writer)
    }

    #[test]
    fn boolean_writer_jit_packs_scalar_values() {
        let values = [
            true, false, true, true, false, false, true, false, true, true, false,
        ];
        let ctx = Context::create();
        let (llvm_mod, func, writer) = compile_write_function(&ctx, &values);
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut runtime = writer.allocate(values.len());
        unsafe {
            f.call(runtime.as_ptr());
        }

        assert_eq!(runtime.len(), values.len());
        let array = runtime.to_array_ref().unwrap();
        assert_eq!(array.as_boolean(), &BooleanArray::from(values.to_vec()));
    }

    #[test]
    fn boolean_writer_jit_packs_vector_values() {
        let block = (0..64).map(|i| i % 3 == 0).collect::<Vec<_>>();
        let scalars = [true, false, true];
        let expected = block
            .iter()
            .copied()
            .chain(scalars)
            .chain(block.iter().copied())
            .collect::<Vec<_>>();
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers_boolean_writer_vector");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );

        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);
        let dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = BooleanWriter::compile();
        let codegen = WriterCodegen {
            ctx: &ctx,
            module: &llvm_mod,
            builder: &build,
        };
        let values = block
            .iter()
            .map(|value| ctx.bool_type().const_int(u64::from(*value), false))
            .collect::<Vec<_>>();
        let values = inkwell::types::VectorType::const_vector(&values);
        writer
            .llvm_write_block(codegen, dest, values, block.len() as u32)
            .unwrap();
        for value in scalars {
            writer
                .llvm_write(codegen, dest, |emitter| {
                    emitter.emit(
                        ctx.bool_type()
                            .const_int(u64::from(value), false)
                            .as_basic_value_enum(),
                    )
                })
                .unwrap();
        }
        writer
            .llvm_write_block(codegen, dest, values, block.len() as u32)
            .unwrap();
        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();

        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };
        let mut runtime = writer.allocate(expected.len());
        unsafe {
            f.call(runtime.as_ptr());
        }

        runtime.reserve_for_additional(expected.len()).unwrap();
        unsafe {
            f.call(runtime.as_ptr());
        }

        let expected = expected
            .iter()
            .copied()
            .chain(expected.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(runtime.len(), expected.len());
        let array = runtime.to_array_ref().unwrap();
        assert_eq!(array.as_boolean(), &BooleanArray::from(expected));
    }

    #[test]
    fn boolean_writer_reserves_and_appends_after_partial_byte() {
        let values = [true, false, true, true, false];
        let ctx = Context::create();
        let (llvm_mod, func, writer) = compile_write_function(&ctx, &values);
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut runtime = writer.allocate(values.len());
        unsafe {
            f.call(runtime.as_ptr());
        }
        runtime.reserve_for_additional(values.len()).unwrap();
        unsafe {
            f.call(runtime.as_ptr());
        }

        let expected = values.into_iter().chain(values).collect::<Vec<_>>();
        assert_eq!(runtime.len(), expected.len());
        let array = runtime.to_array_ref().unwrap();
        assert_eq!(array.as_boolean(), &BooleanArray::from(expected));
    }
}
