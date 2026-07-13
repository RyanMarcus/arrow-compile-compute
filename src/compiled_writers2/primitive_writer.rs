use arrow_array::{make_array, ArrayRef};
use arrow_buffer::Buffer;
use arrow_data::ArrayDataBuilder;
use inkwell::{
    builder::Builder,
    context::Context,
    values::{BasicValue, BasicValueEnum, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::{
    compiled_writers2::{
        AnyRuntime, AnyWriterEmitter, Writer as WriterTrait, WriterEmitter, WriterRuntime,
    },
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
        let buf = Buffer::from(self.alloc);
        let buf = buf.slice_with_length(0, len * self.pt.width());
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

    pub fn allocate(&self, size: usize) -> PrimitiveWriterRuntime {
        let mut pwr = PrimitiveWriterRuntime {
            alloc: Vec::new(),
            alloc_ptr: std::ptr::null_mut(),
            max_len: size,
            pt: self.pt,
        };

        pwr.alloc.resize((self.pt.width() * size).div_ceil(16), 0);
        pwr.alloc_ptr = pwr.alloc.as_mut_ptr() as *mut u8;
        pwr
    }

    pub fn llvm_init(&self, _ctx: &Context, _b: &Builder, _ptr: PointerValue) {}
    pub fn llvm_write<
        'a,
        F: Fn(&mut PrimitiveWriterEmitter<'a>) -> Result<(), ArrowKernelError>,
    >(
        &self,
        ctx: &'a Context,
        b: &Builder<'a>,
        ptr: PointerValue,
        f: F,
    ) -> Result<(), ArrowKernelError> {
        let mut emitter = PrimitiveWriterEmitter::default();
        f(&mut emitter)?;

        if let Some(val) = emitter.val {
            let ptr_type = ctx.ptr_type(AddressSpace::default());
            let curr_alloc_ptr_ptr =
                increment_pointer!(ctx, b, ptr, PrimitiveWriterRuntime::OFFSET_ALLOC_PTR);
            let curr_alloc_ptr = b
                .build_load(ptr_type, curr_alloc_ptr_ptr, "curr_alloc_ptr")
                .unwrap()
                .as_basic_value_enum()
                .into_pointer_value();
            b.build_store(curr_alloc_ptr, val).unwrap();

            let new_alloc_ptr = increment_pointer!(ctx, b, curr_alloc_ptr, self.pt.width());
            b.build_store(curr_alloc_ptr_ptr, new_alloc_ptr).unwrap();
        }

        Ok(())
    }
    pub fn llvm_flush(&self, _ctx: &Context, _b: &Builder, _ptr: PointerValue) {}
}

impl WriterTrait for PrimitiveWriter {
    fn allocate(&self, size: usize) -> AnyRuntime {
        AnyRuntime::PrimitiveWriterRuntime(PrimitiveWriter::allocate(self, size))
    }

    fn llvm_init<'a>(&self, ctx: &'a Context, build: &Builder<'a>, runtime_ptr: PointerValue<'a>) {
        PrimitiveWriter::llvm_init(self, ctx, build, runtime_ptr);
    }

    fn llvm_write<'ctx, 'borrow, F>(
        &'borrow self,
        ctx: &'ctx Context,
        build: &'borrow Builder<'ctx>,
        runtime_ptr: PointerValue<'ctx>,
        f: F,
    ) -> Result<(), ArrowKernelError>
    where
        F: Fn(&mut AnyWriterEmitter<'ctx, 'borrow>) -> Result<(), ArrowKernelError>,
    {
        let mut emitter =
            AnyWriterEmitter::PrimitiveWriterEmitter(PrimitiveWriterEmitter::default());
        f(&mut emitter)?;

        let AnyWriterEmitter::PrimitiveWriterEmitter(emitter) = emitter else {
            return Err(ArrowKernelError::InternalError(
                "primitive writer received non-primitive emitter".into(),
            ));
        };

        PrimitiveWriter::llvm_write(self, ctx, build, runtime_ptr, |inner| {
            if let Some(val) = emitter.val {
                inner.emit(val)?;
            }
            Ok(())
        })
    }

    fn llvm_flush<'a>(&self, ctx: &'a Context, build: &Builder<'a>, runtime_ptr: PointerValue<'a>) {
        PrimitiveWriter::llvm_flush(self, ctx, build, runtime_ptr);
    }
}

#[derive(Default)]
pub struct PrimitiveWriterEmitter<'a> {
    val: Option<BasicValueEnum<'a>>,
}

impl<'ctx, 'borrow> WriterEmitter<'ctx, 'borrow> for PrimitiveWriterEmitter<'ctx> {
    fn emit(&mut self, val: BasicValueEnum<'ctx>) -> Result<(), ArrowKernelError> {
        PrimitiveWriterEmitter::emit(self, val)
    }
}

impl<'a> PrimitiveWriterEmitter<'a> {
    pub fn emit(&mut self, val: BasicValueEnum<'a>) -> Result<(), ArrowKernelError> {
        if self.val.is_some() {
            return Err(ArrowKernelError::InternalError(
                "emit called on non-empty primitive emitter".into(),
            ));
        }
        self.val = Some(val);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int32Type};
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};

    use super::PrimitiveWriter;
    use crate::{compiled_writers2::WriterRuntime, declare_blocks, PrimitiveType};

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
                .llvm_write(&ctx, &build, dest, |emitter| {
                    emitter.emit(
                        ctx.i32_type()
                            .const_int(value as u64, true)
                            .as_basic_value_enum(),
                    )
                })
                .unwrap();
        }
        writer.llvm_flush(&ctx, &build, dest);

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
