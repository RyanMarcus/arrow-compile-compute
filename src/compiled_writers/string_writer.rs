use std::{ffi::c_void, sync::Arc};

use arrow_array::{make_array, ArrayRef};
use arrow_buffer::Buffer;
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use inkwell::{
    context::Context,
    module::{Linkage, Module},
    values::{BasicValue, BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace, IntPredicate,
};
use repr_offset::ReprOffset;

use crate::{
    compiled_writers::{
        AnyRuntime, AnyWriterEmitter, PrimitiveWriter, Writer, WriterCodegen, WriterEmitter,
        WriterRuntime,
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

    fn get_offset_ptr<'a>(
        &self,
        codegen: WriterCodegen<'a, '_>,
        runtime_ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let offset_ptr_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            StringWriterRuntime::OFFSET_OFFSET_RUNTIME_PTR
        );
        codegen
            .builder
            .build_load(
                codegen.ctx.ptr_type(inkwell::AddressSpace::default()),
                offset_ptr_ptr,
                "offset_runtime_ptr",
            )
            .unwrap()
            .into_pointer_value()
    }
}

impl Writer for StringWriter {
    fn allocate(&self, size: usize) -> AnyRuntime {
        let mut runtime = StringWriterRuntime {
            offset_runtime_ptr: std::ptr::null_mut(),
            offsets: Box::new(self.offset_writer.allocate(size + 1)),
            bytes: Vec::with_capacity(4096),
            bytes_ptr: std::ptr::null_mut(),
            bytes_len: 0,
            bytes_cap: 0,
            curr_offset: 0,
        };
        runtime.offset_runtime_ptr = runtime.offsets.as_ptr();
        runtime.bytes_ptr = runtime.bytes.as_mut_ptr();
        runtime.bytes_cap = runtime.bytes.capacity();
        runtime.into()
    }

    fn llvm_init<'ctx, 'borrow>(
        &self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
    ) {
        self.offset_writer
            .llvm_init(codegen, self.get_offset_ptr(codegen, runtime_ptr));
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
        let reserve = llvm_add_str_writer_reserve(codegen.ctx, codegen.module);

        let mut emitter = AnyWriterEmitter::String(StringWriterEmitter {
            codegen,
            offset_writer: &self.offset_writer,
            offset_writer_ptr: self.get_offset_ptr(codegen, runtime_ptr),
            offset_counter_ptr: increment_pointer!(
                codegen.ctx,
                codegen.builder,
                runtime_ptr,
                StringWriterRuntime::OFFSET_CURR_OFFSET
            ),
            runtime_ptr,
            reserve,
        });

        f(&mut emitter)
    }
}

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct StringWriterRuntime {
    offset_runtime_ptr: *mut c_void,
    offsets: Box<AnyRuntime>,
    bytes: Vec<u8>,
    // JIT-visible mirror of `bytes`: the kernel appends through these fields
    // with an inline capacity check and memcpy, calling back into Rust only
    // to grow. `bytes.len()` is stale until the mirror is synced back.
    bytes_ptr: *mut u8,
    bytes_len: usize,
    bytes_cap: usize,
    curr_offset: u64,
}

/// Cold path for the JIT's inline string append: syncs the mirrored length
/// back into the Vec, grows it, and refreshes the mirrored pointer/capacity.
#[no_mangle]
pub extern "C" fn str_writer_reserve(runtime: *mut c_void, needed: u64) {
    let rt = unsafe { &mut *(runtime as *mut StringWriterRuntime) };
    unsafe { rt.bytes.set_len(rt.bytes_len) };
    rt.bytes.reserve(needed as usize - rt.bytes.len());
    rt.bytes_ptr = rt.bytes.as_mut_ptr();
    rt.bytes_cap = rt.bytes.capacity();
}

fn llvm_add_str_writer_reserve<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
) -> FunctionValue<'ctx> {
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    module
        .get_function("str_writer_reserve")
        .unwrap_or_else(|| {
            module.add_function(
                "str_writer_reserve",
                ctx.void_type()
                    .fn_type(&[ptr_type.into(), ctx.i64_type().into()], false),
                Some(Linkage::External),
            )
        })
}

impl WriterRuntime for StringWriterRuntime {
    fn as_ptr(&mut self) -> *mut c_void {
        self as *mut StringWriterRuntime as *mut c_void
    }

    fn reserve_for_additional(&mut self, count: usize) -> Result<(), ArrowKernelError> {
        self.offsets.reserve_for_additional(count + 1)?;
        self.offset_runtime_ptr = self.offsets.as_ptr();
        Ok(())
    }

    fn len(&self) -> usize {
        self.offsets.len()
    }

    fn to_array(mut self, len: usize) -> Result<ArrayRef, ArrowKernelError> {
        let written = self.offsets.len();
        if len > written {
            return Err(ArrowKernelError::InternalError(format!(
                "string writer contains {written} values but {len} were requested"
            )));
        }
        if len == written {
            self.offsets.append_integer(self.curr_offset)?;
        }
        let offsets = self.offsets.to_array(len + 1)?;
        // safety: the JIT initialized bytes up to the mirrored length
        unsafe { self.bytes.set_len(self.bytes_len) };
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
    codegen: WriterCodegen<'ctx, 'borrow>,

    offset_writer: &'borrow PrimitiveWriter,
    offset_writer_ptr: PointerValue<'ctx>,

    offset_counter_ptr: PointerValue<'ctx>,
    runtime_ptr: PointerValue<'ctx>,
    reserve: FunctionValue<'ctx>,
}
impl<'ctx, 'borrow> WriterEmitter<'ctx, 'borrow> for StringWriterEmitter<'ctx, 'borrow> {
    fn emit(&mut self, val: BasicValueEnum<'ctx>) -> Result<(), ArrowKernelError> {
        let codegen = self.codegen;
        let ctx = codegen.ctx;
        let b = codegen.builder;
        let val = val.into_struct_value();
        let ptr1 = b
            .build_extract_value(val, 0, "ptr1")
            .unwrap()
            .into_pointer_value();
        let ptr2 = b
            .build_extract_value(val, 1, "ptr2")
            .unwrap()
            .into_pointer_value();
        let len = pointer_diff!(ctx, b, ptr1, ptr2);

        // inline append: capacity check + memcpy, with a cold call to grow.
        // The old path called into Rust for every string just to run
        // extend_from_slice.
        let i64_type = ctx.i64_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let len_ptr = increment_pointer!(
            ctx,
            b,
            self.runtime_ptr,
            StringWriterRuntime::OFFSET_BYTES_LEN
        );
        let cap_ptr = increment_pointer!(
            ctx,
            b,
            self.runtime_ptr,
            StringWriterRuntime::OFFSET_BYTES_CAP
        );
        let bytes_len = b
            .build_load(i64_type, len_ptr, "bytes_len")
            .unwrap()
            .into_int_value();
        let bytes_cap = b
            .build_load(i64_type, cap_ptr, "bytes_cap")
            .unwrap()
            .into_int_value();
        let needed = b.build_int_add(bytes_len, len, "bytes_needed").unwrap();
        let fits = b
            .build_int_compare(IntPredicate::ULE, needed, bytes_cap, "bytes_fit")
            .unwrap();

        let func = b.get_insert_block().unwrap().get_parent().unwrap();
        let grow_block = ctx.append_basic_block(func, "str_grow");
        let append_block = ctx.append_basic_block(func, "str_append");
        b.build_conditional_branch(fits, append_block, grow_block)
            .unwrap();

        b.position_at_end(grow_block);
        b.build_call(
            self.reserve,
            &[self.runtime_ptr.into(), needed.into()],
            "reserve",
        )
        .unwrap();
        b.build_unconditional_branch(append_block).unwrap();

        b.position_at_end(append_block);
        let bytes_ptr = b
            .build_load(
                ptr_type,
                increment_pointer!(
                    ctx,
                    b,
                    self.runtime_ptr,
                    StringWriterRuntime::OFFSET_BYTES_PTR
                ),
                "bytes_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let dst = increment_pointer!(ctx, b, bytes_ptr, 1, bytes_len);
        b.build_memcpy(dst, 1, ptr1, 1, len).unwrap();
        b.build_store(len_ptr, needed).unwrap();

        let curr_offset = self
            .codegen
            .builder
            .build_load(
                codegen.ctx.i64_type(),
                self.offset_counter_ptr,
                "curr_offset",
            )
            .unwrap()
            .into_int_value();

        self.offset_writer
            .llvm_write(codegen, self.offset_writer_ptr, |e| {
                let offset = match self.offset_writer.primitive_type() {
                    PrimitiveType::I32 => codegen
                        .builder
                        .build_int_truncate(
                            curr_offset,
                            codegen.ctx.i32_type(),
                            "string_offset_i32",
                        )
                        .unwrap()
                        .as_basic_value_enum(),
                    PrimitiveType::I64 => curr_offset.as_basic_value_enum(),
                    _ => unreachable!("invalid string offset type"),
                };
                e.emit(offset)
            })?;

        let new_offset = codegen
            .builder
            .build_int_add(curr_offset, len, "new_offset")
            .unwrap();
        codegen
            .builder
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
        compiled_writers::{Writer, WriterCodegen, WriterEmitter, WriterRuntime},
        declare_blocks, PrimitiveType,
    };

    #[test]
    fn string_writer_jit_writes_utf8_values() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers_string_writer");
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
        let codegen = WriterCodegen {
            ctx: &ctx,
            module: &llvm_mod,
            builder: &build,
        };

        writer.llvm_init(codegen, dest);
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
                .llvm_write(codegen, dest, |emitter| {
                    emitter.emit(value.as_basic_value_enum())
                })
                .unwrap();
        }
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
        runtime.reserve_for_additional(strings.len()).unwrap();
        unsafe {
            f.call(runtime.as_ptr());
        }

        let array = runtime.to_array(strings.len() * 2).unwrap();
        let array = array.as_any().downcast_ref::<BinaryArray>().unwrap();
        let actual: Vec<String> = array
            .iter()
            .map(|s| String::from_utf8(s.unwrap().to_vec()).unwrap())
            .collect();
        assert_eq!(
            actual,
            strings
                .into_iter()
                .chain(strings)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        );
        assert_eq!(array.value_offsets(), &[0, 5, 5, 9, 14, 19, 19, 23, 28]);
    }
}
