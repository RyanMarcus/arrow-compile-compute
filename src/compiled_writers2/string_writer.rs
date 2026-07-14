use std::{ffi::c_void, sync::Arc};

use arrow_array::{make_array, ArrayRef};
use arrow_buffer::Buffer;
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    values::{BasicValue, BasicValueEnum, FunctionValue, PointerValue},
};
use repr_offset::ReprOffset;

use crate::{
    compiled_kernels::llvm_add_str_writer_append_bytes,
    compiled_writers2::{
        AnyRuntime, AnyWriterEmitter, PrimitiveWriter, PrimitiveWriterRuntime, Writer,
        WriterEmitter, WriterRuntime,
    },
    increment_pointer, pointer_diff, ArrowKernelError, PrimitiveType,
};

pub struct StringWriter {
    offset_writer: PrimitiveWriter,
}

impl StringWriter {
    pub fn compile(offset_type: PrimitiveType) -> Result<Self, ArrowKernelError> {
        if !matches!(offset_type, PrimitiveType::I32 | PrimitiveType::I64) {
            return Err(ArrowKernelError::InternalError(format!(
                "string offset type must be I32 or I64, got {offset_type}"
            )));
        }

        Ok(Self {
            offset_writer: PrimitiveWriter::compile(offset_type)?,
        })
    }
}

impl Writer for StringWriter {
    fn allocate(&self, size: usize) -> AnyRuntime {
        StringWriterRuntime {
            offsets: Box::new(self.offset_writer.allocate(size + 1)),
            bytes: Vec::with_capacity(4096),
            curr_offset: 0,
        }
        .into()
    }

    fn llvm_init<'a>(&self, ctx: &'a Context, b: &Builder<'a>, runtime_ptr: PointerValue<'a>) {
        self.offset_writer.llvm_init(
            ctx,
            b,
            increment_pointer!(ctx, b, runtime_ptr, StringWriterRuntime::OFFSET_OFFSETS),
        );
    }

    fn llvm_write<'ctx, 'borrow, F>(
        &'borrow self,
        ctx: &'ctx Context,
        module: &'borrow Module<'ctx>,
        build: &'borrow Builder<'ctx>,
        runtime_ptr: PointerValue<'ctx>,
        f: F,
    ) -> Result<(), ArrowKernelError>
    where
        F: Fn(&mut AnyWriterEmitter<'ctx, 'borrow>) -> Result<(), ArrowKernelError>,
    {
        let extend = llvm_add_str_writer_append_bytes(ctx, module);

        let mut emitter = AnyWriterEmitter::StringWriterEmitter(StringWriterEmitter {
            ctx,
            builder: build,
            module,
            offset_writer: &self.offset_writer,
            offset_writer_ptr: increment_pointer!(
                ctx,
                build,
                runtime_ptr,
                StringWriterRuntime::OFFSET_OFFSETS
            ),
            offset_counter_ptr: increment_pointer!(
                ctx,
                build,
                runtime_ptr,
                StringWriterRuntime::OFFSET_CURR_OFFSET
            ),
            vec_ptr: increment_pointer!(ctx, build, runtime_ptr, StringWriterRuntime::OFFSET_BYTES),
            extend,
        });

        f(&mut emitter)
    }

    fn llvm_flush<'ctx, 'borrow>(
        &'borrow self,
        ctx: &'ctx Context,
        module: &'borrow Module<'ctx>,
        build: &'borrow Builder<'ctx>,
        runtime_ptr: PointerValue<'ctx>,
    ) {
        let curr_offset_ptr = increment_pointer!(
            ctx,
            build,
            runtime_ptr,
            StringWriterRuntime::OFFSET_CURR_OFFSET
        );
        let curr_offset = build
            .build_load(ctx.i64_type(), curr_offset_ptr, "final_string_offset")
            .unwrap()
            .into_int_value();
        let offset_writer_ptr =
            increment_pointer!(ctx, build, runtime_ptr, StringWriterRuntime::OFFSET_OFFSETS);

        self.offset_writer
            .llvm_write(ctx, module, build, offset_writer_ptr, |e| {
                let offset = match self.offset_writer.primitive_type() {
                    PrimitiveType::I32 => build
                        .build_int_truncate(curr_offset, ctx.i32_type(), "final_string_offset_i32")
                        .unwrap()
                        .as_basic_value_enum(),
                    PrimitiveType::I64 => curr_offset.as_basic_value_enum(),
                    pt => {
                        return Err(ArrowKernelError::InternalError(format!(
                            "unsupported string offset type {pt}"
                        )));
                    }
                };
                e.emit(offset)
            })
            .unwrap();
        self.offset_writer
            .llvm_flush(ctx, module, build, offset_writer_ptr);
    }
}

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct StringWriterRuntime {
    offsets: Box<AnyRuntime>,
    bytes: Vec<u8>,
    curr_offset: u64,
}

impl WriterRuntime for StringWriterRuntime {
    fn as_ptr(&mut self) -> *mut c_void {
        self as *mut StringWriterRuntime as *mut c_void
    }

    fn reserve_for_additional(&mut self, count: usize) -> Result<(), ArrowKernelError> {
        self.offsets.reserve_for_additional(count)
    }

    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn to_array(self, len: usize) -> Result<ArrayRef, ArrowKernelError> {
        let offsets = self.offsets.to_array(len + 1)?;
        let data = Buffer::from(self.bytes);

        let dt = match offsets.data_type() {
            DataType::Int32 => DataType::Binary,
            DataType::Int64 => DataType::LargeBinary,
            _ => unreachable!("invalid string offset type"),
        };

        let offsets = offsets.to_data();
        let array = unsafe {
            make_array(
                ArrayDataBuilder::new(dt)
                    .len(len)
                    .add_buffer(offsets.buffers()[0].clone())
                    .add_buffer(data)
                    .build_unchecked(),
            )
        };
        Ok(Arc::new(array))
    }
}

pub struct StringWriterEmitter<'ctx, 'borrow> {
    ctx: &'ctx Context,
    module: &'borrow Module<'ctx>,
    builder: &'borrow Builder<'ctx>,

    offset_writer: &'borrow PrimitiveWriter,
    offset_writer_ptr: PointerValue<'ctx>,

    offset_counter_ptr: PointerValue<'ctx>,
    vec_ptr: PointerValue<'ctx>,
    extend: FunctionValue<'ctx>,
}
impl<'ctx, 'borrow> WriterEmitter<'ctx, 'borrow> for StringWriterEmitter<'ctx, 'borrow> {
    fn emit(&mut self, val: BasicValueEnum<'ctx>) -> Result<(), ArrowKernelError> {
        let val = val.into_struct_value();
        let ptr1 = self
            .builder
            .build_extract_value(val, 0, "ptr1")
            .unwrap()
            .into_pointer_value();
        let ptr2 = self
            .builder
            .build_extract_value(val, 1, "ptr2")
            .unwrap()
            .into_pointer_value();
        let len = pointer_diff!(self.ctx, self.builder, ptr1, ptr2);
        self.builder
            .build_call(
                self.extend,
                &[ptr1.into(), len.into(), self.vec_ptr.into()],
                "extend",
            )
            .unwrap();

        let curr_offset = self
            .builder
            .build_load(self.ctx.i64_type(), self.offset_counter_ptr, "curr_offset")
            .unwrap()
            .into_int_value();

        self.offset_writer.llvm_write(
            self.ctx,
            self.module,
            self.builder,
            self.offset_writer_ptr,
            |e| {
                let offset = match self.offset_writer.primitive_type() {
                    PrimitiveType::I32 => self
                        .builder
                        .build_int_truncate(curr_offset, self.ctx.i32_type(), "string_offset_i32")
                        .unwrap()
                        .as_basic_value_enum(),
                    PrimitiveType::I64 => curr_offset.as_basic_value_enum(),
                    _ => unreachable!("invalid string offset type"),
                };
                e.emit(offset)
            },
        )?;

        let new_offset = self
            .builder
            .build_int_add(curr_offset, len, "new_offset")
            .unwrap();
        self.builder
            .build_store(self.offset_counter_ptr, new_offset)
            .unwrap();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::BinaryArray;
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};

    use super::StringWriter;
    use crate::{
        compiled_writers2::{Writer, WriterEmitter, WriterRuntime},
        declare_blocks, PrimitiveType,
    };

    #[test]
    fn string_writer_jit_writes_utf8_values() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers2_string_writer");
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
        let writer = StringWriter::compile(PrimitiveType::I32).unwrap();
        let string_type = PrimitiveType::P64x2.llvm_type(&ctx).into_struct_type();
        let strings = ["alpha", "", "beta", "gamma"];

        writer.llvm_init(&ctx, &build, dest);
        for s in strings {
            let start = s.as_ptr();
            let end = start.wrapping_add(s.len());
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
                .llvm_write(&ctx, &llvm_mod, &build, dest, |emitter| {
                    emitter.emit(value.as_basic_value_enum())
                })
                .unwrap();
        }
        writer.llvm_flush(&ctx, &llvm_mod, &build, dest);
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

        let mut runtime = writer.allocate(strings.len());
        unsafe {
            f.call(runtime.as_ptr());
        }

        let array = runtime.to_array(strings.len()).unwrap();
        let array = array.as_any().downcast_ref::<BinaryArray>().unwrap();
        let actual: Vec<String> = array
            .iter()
            .map(|s| String::from_utf8(s.unwrap().to_vec()).unwrap())
            .collect();
        assert_eq!(actual, strings);
        assert_eq!(array.value_offsets(), &[0, 5, 5, 9, 14]);
    }
}
