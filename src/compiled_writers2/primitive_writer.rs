use arrow_array::{make_array, ArrayRef};
use arrow_buffer::Buffer;
use arrow_data::ArrayDataBuilder;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    values::{BasicValue, BasicValueEnum, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::{
    compiled_writers2::{AnyRuntime, AnyWriterEmitter, Writer, WriterEmitter, WriterRuntime},
    increment_pointer, ArrowKernelError, PrimitiveType,
};

pub struct PrimitiveWriter {
    pt: PrimitiveType,
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct PrimitiveWriterRuntime {
    alloc: Vec<u128>,
    alloc_ptr: *mut u8,
    max_len: usize,
    pt: PrimitiveType,
}

impl WriterRuntime for PrimitiveWriterRuntime {
    fn as_ptr(&mut self) -> *mut std::ffi::c_void {
        (self as *mut Self).cast()
    }

    fn reserve_for_additional(&mut self, count: usize) -> Result<(), ArrowKernelError> {
        unsafe {
            let base = self.alloc.as_mut_ptr() as *mut u8;
            let bytes_written = self.alloc_ptr.offset_from_unsigned(base);
            let items_to_preserve = bytes_written.div_ceil(16);
            self.alloc.set_len(items_to_preserve);
            self.alloc.resize(
                items_to_preserve + (count * self.pt.width()).div_ceil(16),
                0,
            );
            let new_base = self.alloc.as_mut_ptr() as *mut u8;
            self.alloc_ptr = new_base.byte_add(bytes_written);
            self.max_len = self.len() + count;
        }
        Ok(())
    }

    fn len(&self) -> usize {
        let base = self.alloc.as_ptr() as *const u8;
        let bytes_written = unsafe { self.alloc_ptr.offset_from_unsigned(base) };
        assert_eq!(bytes_written % self.pt.width(), 0);
        bytes_written / self.pt.width()
    }

    fn to_array(self, len: usize) -> Result<ArrayRef, ArrowKernelError> {
        let mut buf = Buffer::from(self.alloc);
        let sliced = buf.slice_with_length(0, len * self.pt.width());
        if buf.len() > len * self.pt.width() * 2 {
            // over 2x over allocated, trim to exact size
            let vec = sliced.to_vec();
            buf = Buffer::from(vec);
        } else {
            buf = sliced;
        }

        let ad = unsafe {
            ArrayDataBuilder::new(self.pt.as_arrow_type())
                .add_buffer(buf)
                .len(len)
                .build_unchecked()
        };
        Ok(make_array(ad))
    }
}

impl PrimitiveWriter {
    pub fn primitive_type(&self) -> PrimitiveType {
        self.pt
    }

    pub fn compile(pt: PrimitiveType) -> Result<PrimitiveWriter, ArrowKernelError> {
        if matches!(pt, PrimitiveType::List(_, _) | PrimitiveType::P64x2) {
            return Err(ArrowKernelError::InternalError(
                "unsupported primitive type in writer".into(),
            ));
        }

        Ok(PrimitiveWriter { pt })
    }
}

impl Writer for PrimitiveWriter {
    fn allocate(&self, size: usize) -> AnyRuntime {
        let mut pwr = PrimitiveWriterRuntime {
            alloc: Vec::new(),
            alloc_ptr: std::ptr::null_mut(),
            max_len: size,
            pt: self.pt,
        };

        pwr.alloc.resize((self.pt.width() * size).div_ceil(16), 0);
        pwr.alloc_ptr = pwr.alloc.as_mut_ptr() as *mut u8;
        pwr.into()
    }

    fn llvm_init<'a>(
        &self,
        _ctx: &'a Context,
        _build: &Builder<'a>,
        _runtime_ptr: PointerValue<'a>,
    ) {
    }

    fn llvm_write<'ctx, 'borrow, F>(
        &'borrow self,
        ctx: &'ctx Context,
        _module: &'borrow Module<'ctx>,
        b: &'borrow Builder<'ctx>,
        runtime_ptr: PointerValue<'ctx>,
        f: F,
    ) -> Result<(), ArrowKernelError>
    where
        F: Fn(&mut AnyWriterEmitter<'ctx, 'borrow>) -> Result<(), ArrowKernelError>,
    {
        let mut emitter = PrimitiveWriterEmitter {
            ctx,
            b,
            runtime_ptr,
            pt: self.pt,
            used: false,
        }
        .into();
        f(&mut emitter)
    }
}

pub struct PrimitiveWriterEmitter<'ctx, 'borrow> {
    ctx: &'ctx Context,
    b: &'borrow Builder<'ctx>,
    runtime_ptr: PointerValue<'ctx>,
    pt: PrimitiveType,
    used: bool,
}

impl<'ctx, 'borrow> WriterEmitter<'ctx, 'borrow> for PrimitiveWriterEmitter<'ctx, 'borrow> {
    fn emit(&mut self, val: BasicValueEnum<'ctx>) -> Result<(), ArrowKernelError> {
        if self.used {
            return Err(ArrowKernelError::InternalError(
                "emit called on non-empty primitive emitter".into(),
            ));
        }
        self.used = true;

        let ptr_type = self.ctx.ptr_type(AddressSpace::default());
        let curr_alloc_ptr_ptr = increment_pointer!(
            self.ctx,
            self.b,
            self.runtime_ptr,
            PrimitiveWriterRuntime::OFFSET_ALLOC_PTR
        );
        let curr_alloc_ptr = self
            .b
            .build_load(ptr_type, curr_alloc_ptr_ptr, "curr_alloc_ptr")
            .unwrap()
            .as_basic_value_enum()
            .into_pointer_value();
        self.b.build_store(curr_alloc_ptr, val).unwrap();

        let new_alloc_ptr = increment_pointer!(self.ctx, self.b, curr_alloc_ptr, self.pt.width());
        self.b
            .build_store(curr_alloc_ptr_ptr, new_alloc_ptr)
            .unwrap();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int32Type};
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};

    use super::PrimitiveWriter;
    use crate::{
        compiled_writers2::{Writer, WriterEmitter, WriterRuntime},
        declare_blocks, PrimitiveType,
    };

    #[test]
    fn primitive_writer_jit_writes_values_to_array() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers2_primitive_writer");
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
        let writer = PrimitiveWriter::compile(PrimitiveType::I32).unwrap();

        writer.llvm_init(&ctx, &build, dest);
        for value in [17_i32, -4, 99] {
            writer
                .llvm_write(&ctx, &llvm_mod, &build, dest, |emitter| {
                    emitter.emit(
                        ctx.i32_type()
                            .const_int(value as u64, true)
                            .as_basic_value_enum(),
                    )
                })
                .unwrap();
        }
        writer.llvm_flush(&ctx, &llvm_mod, &build, dest);

        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();

        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut runtime = writer.allocate(3);
        unsafe {
            f.call(runtime.as_ptr());
        }

        let array = runtime.to_array(3).unwrap();
        let values = array.as_primitive::<Int32Type>().values();
        assert_eq!(values, &[17, -4, 99]);
    }
}
