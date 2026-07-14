use std::{ffi::c_void, sync::Arc};

use arrow_array::{
    cast::AsArray,
    types::{Int32Type, Int64Type},
    ListArray,
};
use arrow_data::ArrayDataBuilder;
use arrow_schema::{DataType, Field};
use inkwell::{
    values::{BasicValue, BasicValueEnum, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::{
    compiled_writers2::{
        AnyRuntime, AnyWriter, AnyWriterEmitter, PrimitiveWriter, Writer, WriterCodegen,
        WriterEmitter, WriterRuntime,
    },
    increment_pointer, ArrowKernelError, PrimitiveType,
};

pub struct ListWriter {
    offsets: PrimitiveWriter,
    inner: Box<AnyWriter>,
}

impl ListWriter {
    fn get_offset_ptr<'a>(
        &self,
        codegen: WriterCodegen<'a, '_>,
        runtime_ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let offset_ptr_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            ListWriterRuntime::OFFSET_OFFSET_RUNTIME_PTR
        );
        let offset_ptr = codegen
            .builder
            .build_load(
                codegen.ctx.ptr_type(AddressSpace::default()),
                offset_ptr_ptr,
                "offset_ptr",
            )
            .unwrap()
            .into_pointer_value();
        offset_ptr
    }

    fn get_value_ptr<'a>(
        &self,
        codegen: WriterCodegen<'a, '_>,
        runtime_ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let value_ptr_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            ListWriterRuntime::OFFSET_VALUE_RUNTIME_PTR
        );
        let value_ptr = codegen
            .builder
            .build_load(
                codegen.ctx.ptr_type(AddressSpace::default()),
                value_ptr_ptr,
                "value_ptr",
            )
            .unwrap()
            .into_pointer_value();
        value_ptr
    }
}

impl Writer for ListWriter {
    fn allocate(&self, size: usize) -> AnyRuntime {
        let mut runtime = ListWriterRuntime {
            offset_runtime_ptr: std::ptr::null_mut(),
            value_runtime_ptr: std::ptr::null_mut(),
            curr_count: 0,
            offset_runtime: Box::new(self.offsets.allocate(size)),
            value_runtime: Box::new(self.inner.allocate(size)),
        };
        runtime.offset_runtime_ptr = runtime.offset_runtime.as_ptr();
        runtime.value_runtime_ptr = runtime.value_runtime.as_ptr();
        runtime.into()
    }

    fn llvm_init<'ctx, 'borrow>(
        &self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
    ) {
        self.offsets
            .llvm_init(codegen, self.get_offset_ptr(codegen, runtime_ptr));
        self.inner
            .llvm_init(codegen, self.get_value_ptr(codegen, runtime_ptr));
    }

    fn llvm_write<'ctx, 'borrow, F>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        f: F,
    ) -> Result<(), crate::ArrowKernelError>
    where
        F: Fn(&mut AnyWriterEmitter<'ctx, 'borrow>) -> Result<(), crate::ArrowKernelError>,
    {
        let curr_count_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            ListWriterRuntime::OFFSET_CURR_COUNT
        );
        let curr_count = codegen
            .builder
            .build_load(codegen.ctx.i64_type(), curr_count_ptr, "curr_count")
            .unwrap();

        let mut emitter = ListWriterEmitter {
            codegen,
            run_counter_ptr: curr_count_ptr,
            value_ptr: self.get_value_ptr(codegen, runtime_ptr),
            value_writer: self.inner.as_ref(),
        }
        .into();

        f(&mut emitter)?;

        self.offsets
            .llvm_write(codegen, self.get_offset_ptr(codegen, runtime_ptr), |e| {
                e.emit(curr_count)
            })?;

        Ok(())
    }

    fn llvm_flush<'ctx, 'borrow>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
    ) {
        let curr_count_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            ListWriterRuntime::OFFSET_CURR_COUNT
        );
        let curr_count = codegen
            .builder
            .build_load(codegen.ctx.i64_type(), curr_count_ptr, "final_count")
            .unwrap()
            .into_int_value();
        self.offsets
            .llvm_write(codegen, self.get_offset_ptr(codegen, runtime_ptr), |e| {
                e.emit(match self.offsets.primitive_type() {
                    PrimitiveType::I32 => codegen
                        .builder
                        .build_int_truncate(curr_count, codegen.ctx.i32_type(), "final_count_i32")
                        .unwrap()
                        .as_basic_value_enum(),
                    PrimitiveType::I64 => curr_count.as_basic_value_enum(),
                    _ => unreachable!("invalid list offset type"),
                })
            })
            .unwrap();

        self.offsets
            .llvm_flush(codegen, self.get_offset_ptr(codegen, runtime_ptr));
        self.inner
            .llvm_flush(codegen, self.get_value_ptr(codegen, runtime_ptr));
    }
}

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct ListWriterRuntime {
    offset_runtime_ptr: *mut c_void,
    value_runtime_ptr: *mut c_void,
    curr_count: u64,
    offset_runtime: Box<AnyRuntime>,
    value_runtime: Box<AnyRuntime>,
}

impl WriterRuntime for ListWriterRuntime {
    fn as_ptr(&mut self) -> *mut std::ffi::c_void {
        self as *mut Self as *mut std::ffi::c_void
    }

    fn reserve_for_additional(&mut self, count: usize) -> Result<(), ArrowKernelError> {
        self.offset_runtime.reserve_for_additional(count)?;
        self.value_runtime.reserve_for_additional(count)?;
        self.offset_runtime_ptr = self.offset_runtime.as_ptr();
        self.value_runtime_ptr = self.value_runtime.as_ptr();
        Ok(())
    }

    fn len(&self) -> usize {
        self.offset_runtime.len() - 1
    }

    fn to_array(self, len: usize) -> Result<arrow_array::ArrayRef, crate::ArrowKernelError> {
        let offsets = self.offset_runtime.to_array(len + 1)?;

        let value_len = match offsets.data_type() {
            DataType::Int32 => offsets.as_primitive::<Int32Type>().value(len) as usize,
            DataType::Int64 => offsets.as_primitive::<Int64Type>().value(len) as usize,
            dt => {
                return Err(ArrowKernelError::InternalError(format!(
                    "list offsets must be Int32 or Int64, got {dt}"
                )))
            }
        };

        let values = self.value_runtime.to_array(value_len)?;
        let field = Field::new("item", values.data_type().clone(), false);
        let data_type = DataType::List(Arc::new(field));
        let offsets = offsets.to_data();
        let data = unsafe {
            ArrayDataBuilder::new(data_type)
                .len(len)
                .add_buffer(offsets.buffers()[0].clone())
                .add_child_data(values.to_data())
                .build_unchecked()
        };

        Ok(Arc::new(ListArray::from(data)))
    }
}

pub struct ListWriterEmitter<'ctx, 'borrow> {
    codegen: WriterCodegen<'ctx, 'borrow>,
    run_counter_ptr: PointerValue<'ctx>,
    value_ptr: PointerValue<'ctx>,
    value_writer: &'borrow AnyWriter,
}
impl<'ctx, 'borrow> WriterEmitter<'ctx, 'borrow> for ListWriterEmitter<'ctx, 'borrow> {
    fn emit(&mut self, val: BasicValueEnum<'ctx>) -> Result<(), crate::ArrowKernelError> {
        let i64_type = self.codegen.ctx.i64_type();
        self.value_writer
            .llvm_write(self.codegen, self.value_ptr, |e| e.emit(val))?;

        let curr_run_count = self
            .codegen
            .builder
            .build_load(i64_type, self.run_counter_ptr, "curr_run_count")
            .unwrap()
            .into_int_value();
        let new_run_count = self
            .codegen
            .builder
            .build_int_add(
                curr_run_count,
                i64_type.const_int(1, false),
                "new_run_count",
            )
            .unwrap();
        self.codegen
            .builder
            .build_store(self.run_counter_ptr, new_run_count)
            .unwrap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int32Type};
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};

    use super::ListWriter;
    use crate::{
        compiled_writers2::{
            AnyWriter, PrimitiveWriter, Writer, WriterCodegen, WriterEmitter, WriterRuntime,
        },
        declare_blocks, PrimitiveType,
    };

    #[test]
    fn list_writer_runtime_to_array_appends_final_offset() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers2_list_writer");
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
        let writer = ListWriter {
            offsets: PrimitiveWriter::compile(PrimitiveType::I32).unwrap(),
            inner: Box::new(AnyWriter::Primitive(
                PrimitiveWriter::compile(PrimitiveType::I32).unwrap(),
            )),
        };
        let codegen = WriterCodegen {
            ctx: &ctx,
            module: &llvm_mod,
            builder: &build,
        };

        writer.llvm_init(codegen, dest);
        writer
            .llvm_write(codegen, dest, |emitter| {
                emitter.emit(ctx.i32_type().const_int(10, true).as_basic_value_enum())?;
                emitter.emit(ctx.i32_type().const_int(11, true).as_basic_value_enum())
            })
            .unwrap();
        writer.llvm_write(codegen, dest, |_emitter| Ok(())).unwrap();
        writer
            .llvm_write(codegen, dest, |emitter| {
                emitter.emit(ctx.i32_type().const_int(12, true).as_basic_value_enum())?;
                emitter.emit(ctx.i32_type().const_int(13, true).as_basic_value_enum())?;
                emitter.emit(ctx.i32_type().const_int(14, true).as_basic_value_enum())
            })
            .unwrap();
        writer.llvm_flush(codegen, dest);

        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();

        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut runtime = writer.allocate(5);
        unsafe {
            f.call(runtime.as_ptr());
        }

        let array = runtime.to_array(3).unwrap();
        let list = array.as_list::<i32>();
        assert_eq!(list.offsets().as_ref(), &[0, 2, 2, 5]);
        assert_eq!(
            list.values().as_primitive::<Int32Type>().values(),
            &[10, 11, 12, 13, 14]
        );
    }

    #[test]
    fn list_writer_writes_nested_list() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers2_nested_list_writer");
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
        let writer = ListWriter {
            offsets: PrimitiveWriter::compile(PrimitiveType::I32).unwrap(),
            inner: Box::new(AnyWriter::List(ListWriter {
                offsets: PrimitiveWriter::compile(PrimitiveType::I32).unwrap(),
                inner: Box::new(AnyWriter::Primitive(
                    PrimitiveWriter::compile(PrimitiveType::I32).unwrap(),
                )),
            })),
        };
        let codegen = WriterCodegen {
            ctx: &ctx,
            module: &llvm_mod,
            builder: &build,
        };

        writer.llvm_init(codegen, dest);
        writer
            .llvm_write(codegen, dest, |emitter| {
                emitter.emit(ctx.i32_type().const_int(10, true).as_basic_value_enum())?;
                emitter.emit(ctx.i32_type().const_int(11, true).as_basic_value_enum())
            })
            .unwrap();
        writer.llvm_write(codegen, dest, |_emitter| Ok(())).unwrap();
        writer
            .llvm_write(codegen, dest, |emitter| {
                emitter.emit(ctx.i32_type().const_int(12, true).as_basic_value_enum())?;
                emitter.emit(ctx.i32_type().const_int(13, true).as_basic_value_enum())?;
                emitter.emit(ctx.i32_type().const_int(14, true).as_basic_value_enum())
            })
            .unwrap();
        writer.llvm_flush(codegen, dest);

        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();

        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut runtime = writer.allocate(5);
        unsafe {
            f.call(runtime.as_ptr());
        }

        let array = runtime.to_array(3).unwrap();
        let outer = array.as_list::<i32>();
        assert_eq!(outer.offsets().as_ref(), &[0, 2, 2, 5]);

        let inner = outer.values().as_list::<i32>();
        assert_eq!(inner.offsets().as_ref(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(
            inner.values().as_primitive::<Int32Type>().values(),
            &[10, 11, 12, 13, 14]
        );
    }
}
