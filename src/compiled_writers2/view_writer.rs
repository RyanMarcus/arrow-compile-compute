use std::{ffi::c_void, sync::Arc};

use arrow_array::{types::BinaryViewType, ArrayRef, GenericByteViewArray};
use inkwell::{
    intrinsics::Intrinsic,
    module::Linkage,
    values::{BasicValueEnum, PointerValue},
    AddressSpace, IntPredicate,
};
use repr_offset::ReprOffset;

use crate::{
    compiled_writers::ViewBufferWriter,
    compiled_writers2::{
        AnyRuntime, AnyWriterEmitter, Writer, WriterCodegen, WriterEmitter, WriterRuntime,
    },
    declare_blocks, increment_pointer, pointer_diff, ArrowKernelError, PrimitiveType,
};

/// Writer for binary and UTF-8 view arrays.
pub struct StringViewWriter;

impl StringViewWriter {
    pub fn compile() -> Self {
        Self
    }
}

impl Writer for StringViewWriter {
    fn allocate(&self, size: usize) -> AnyRuntime {
        let mut views = vec![0_u128; size];
        let views_ptr = views.as_mut_ptr();
        StringViewWriterRuntime {
            views_ptr,
            views,
            data: ViewBufferWriter::new(),
            num_written: 0,
        }
        .into()
    }

    fn llvm_init<'ctx, 'borrow>(
        &self,
        _codegen: WriterCodegen<'ctx, 'borrow>,
        _runtime_ptr: PointerValue<'ctx>,
    ) {
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
        let mut emitter = AnyWriterEmitter::StringView(StringViewWriterEmitter {
            codegen,
            runtime_ptr,
            used: false,
        });
        f(&mut emitter)?;
        let AnyWriterEmitter::StringView(emitter) = emitter else {
            unreachable!()
        };
        if !emitter.used {
            return Err(ArrowKernelError::InternalError(
                "emit not called on string view emitter".into(),
            ));
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct StringViewWriterRuntime {
    views_ptr: *mut u128,
    views: Vec<u128>,
    data: ViewBufferWriter,
    num_written: u64,
}

impl WriterRuntime for StringViewWriterRuntime {
    fn as_ptr(&mut self) -> *mut c_void {
        (self as *mut Self).cast()
    }

    fn reserve_for_additional(&mut self, count: usize) -> Result<(), ArrowKernelError> {
        unsafe {
            let base = self.views.as_mut_ptr();
            let views_written = self.views_ptr.offset_from_unsigned(base);
            self.views.set_len(views_written);
            self.views.resize(views_written + count, 0);
            self.views_ptr = self.views.as_mut_ptr().add(views_written);
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.num_written as usize
    }

    fn to_array(mut self, len: usize) -> Result<ArrayRef, ArrowKernelError> {
        self.views.truncate(len);
        let array = GenericByteViewArray::<BinaryViewType>::try_new(
            self.views.into(),
            self.data.into_buffers(),
            None,
        )
        .map_err(|error| {
            ArrowKernelError::InternalError(format!(
                "string view writer produced an invalid array: {error}"
            ))
        })?;
        Ok(Arc::new(array))
    }
}

pub struct StringViewWriterEmitter<'ctx, 'borrow> {
    codegen: WriterCodegen<'ctx, 'borrow>,
    runtime_ptr: PointerValue<'ctx>,
    used: bool,
}

impl<'ctx, 'borrow> WriterEmitter<'ctx, 'borrow> for StringViewWriterEmitter<'ctx, 'borrow> {
    fn emit(&mut self, val: BasicValueEnum<'ctx>) -> Result<(), ArrowKernelError> {
        if self.used {
            return Err(ArrowKernelError::InternalError(
                "emit called on non-empty string view emitter".into(),
            ));
        }
        self.used = true;

        let expected_type = PrimitiveType::P64x2.llvm_type(self.codegen.ctx);
        if val.get_type() != expected_type {
            return Err(ArrowKernelError::InternalError(format!(
                "string view writer expected LLVM type {expected_type}, got {}",
                val.get_type()
            )));
        }

        let codegen = self.codegen;
        let ptr_type = codegen.ctx.ptr_type(AddressSpace::default());
        let i64_type = codegen.ctx.i64_type();
        let i128_type = codegen.ctx.i128_type();
        let val = val.into_struct_value();
        let start = codegen
            .builder
            .build_extract_value(val, 0, "view_start")
            .unwrap()
            .into_pointer_value();
        let end = codegen
            .builder
            .build_extract_value(val, 1, "view_end")
            .unwrap()
            .into_pointer_value();
        let len = pointer_diff!(codegen.ctx, codegen.builder, start, end);

        let views_ptr_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            self.runtime_ptr,
            StringViewWriterRuntime::OFFSET_VIEWS_PTR
        );
        let views_ptr = codegen
            .builder
            .build_load(ptr_type, views_ptr_ptr, "string_view_output")
            .unwrap()
            .into_pointer_value();
        let tmp_view_ptr = codegen.builder.build_alloca(i128_type, "tmp_view").unwrap();
        let view_with_len = codegen
            .builder
            .build_int_z_extend(len, i128_type, "view_with_len")
            .unwrap();
        codegen
            .builder
            .build_store(tmp_view_ptr, view_with_len)
            .unwrap();

        let func = codegen
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();
        declare_blocks!(codegen.ctx, func, short_view, long_view, view_write_exit);
        let is_short = codegen
            .builder
            .build_int_compare(
                IntPredicate::ULE,
                len,
                i64_type.const_int(12, false),
                "is_short_view",
            )
            .unwrap();
        codegen
            .builder
            .build_conditional_branch(is_short, short_view, long_view)
            .unwrap();

        codegen.builder.position_at_end(short_view);
        let memcpy = Intrinsic::find("llvm.memcpy").unwrap();
        let memcpy = memcpy
            .get_declaration(
                codegen.module,
                &[ptr_type.into(), ptr_type.into(), i64_type.into()],
            )
            .unwrap();
        codegen
            .builder
            .build_call(
                memcpy,
                &[
                    increment_pointer!(codegen.ctx, codegen.builder, tmp_view_ptr, 4).into(),
                    start.into(),
                    len.into(),
                    codegen.ctx.bool_type().const_zero().into(),
                ],
                "copy_inline_view",
            )
            .unwrap();
        let inline_view = codegen
            .builder
            .build_load(i128_type, tmp_view_ptr, "inline_view")
            .unwrap();
        codegen.builder.build_store(views_ptr, inline_view).unwrap();
        codegen
            .builder
            .build_unconditional_branch(view_write_exit)
            .unwrap();

        codegen.builder.position_at_end(long_view);
        let append = codegen
            .module
            .get_function("str_view_writer_append_bytes")
            .unwrap_or_else(|| {
                codegen.module.add_function(
                    "str_view_writer_append_bytes",
                    codegen.ctx.void_type().fn_type(
                        &[
                            ptr_type.into(),
                            i64_type.into(),
                            ptr_type.into(),
                            ptr_type.into(),
                        ],
                        false,
                    ),
                    Some(Linkage::External),
                )
            });
        let data_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            self.runtime_ptr,
            StringViewWriterRuntime::OFFSET_DATA
        );
        codegen
            .builder
            .build_call(
                append,
                &[start.into(), len.into(), views_ptr.into(), data_ptr.into()],
                "append_long_view",
            )
            .unwrap();
        codegen
            .builder
            .build_unconditional_branch(view_write_exit)
            .unwrap();

        codegen.builder.position_at_end(view_write_exit);
        let next_views_ptr = increment_pointer!(codegen.ctx, codegen.builder, views_ptr, 16);
        codegen
            .builder
            .build_store(views_ptr_ptr, next_views_ptr)
            .unwrap();

        let num_written_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            self.runtime_ptr,
            StringViewWriterRuntime::OFFSET_NUM_WRITTEN
        );
        let num_written = codegen
            .builder
            .build_load(i64_type, num_written_ptr, "string_views_written")
            .unwrap()
            .into_int_value();
        let new_num_written = codegen
            .builder
            .build_int_add(
                num_written,
                i64_type.const_int(1, false),
                "new_string_views_written",
            )
            .unwrap();
        codegen
            .builder
            .build_store(num_written_ptr, new_num_written)
            .unwrap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::BinaryViewArray;
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};

    use crate::{
        compiled_writers2::{
            AnyWriter, Writer, WriterCodegen, WriterEmitter, WriterPlan, WriterRuntime,
        },
        declare_blocks, PrimitiveType,
    };

    #[test]
    fn string_view_writer_plan_compiles_view_writer() {
        for data_type in [
            arrow_schema::DataType::Utf8View,
            arrow_schema::DataType::BinaryView,
        ] {
            let plan = WriterPlan::for_data_type(&data_type).unwrap();
            assert!(matches!(
                plan.compile().unwrap(),
                AnyWriter::StringView(_)
            ));
        }
    }

    #[test]
    fn string_view_writer_jit_writes_inline_and_buffered_values_and_reserves() {
        let values: [&[u8]; 7] = [
            b"",
            b"short",
            b"123456789012",
            b"1234567890123",
            b"a much longer string view value",
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255],
            &[255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ];
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers2_string_view_writer");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );
        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);

        let runtime_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = WriterPlan::StringView.compile().unwrap();
        let codegen = WriterCodegen {
            ctx: &ctx,
            module: &llvm_mod,
            builder: &build,
        };
        let string_type = PrimitiveType::P64x2.llvm_type(&ctx).into_struct_type();
        writer.llvm_init(codegen, runtime_ptr);
        for value in values {
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
        writer.llvm_flush(codegen, runtime_ptr);
        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();

        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        crate::compiled_kernels::link_req_helpers(&llvm_mod, &ee).unwrap();
        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut runtime = writer.allocate(values.len());
        unsafe {
            f.call(runtime.as_ptr());
        }
        assert_eq!(runtime.len(), values.len());

        runtime.reserve_for_additional(values.len()).unwrap();
        unsafe {
            f.call(runtime.as_ptr());
        }
        assert_eq!(runtime.len(), values.len() * 2);

        let array = runtime.to_array_ref().unwrap();
        let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
        let expected: Vec<&[u8]> = values.into_iter().chain(values).collect();
        let actual: Vec<&[u8]> = array.iter().map(Option::unwrap).collect();
        assert_eq!(actual, expected);
    }
}
