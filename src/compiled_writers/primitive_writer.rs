use arrow_array::{make_array, ArrayRef};
use arrow_buffer::Buffer;
use arrow_data::ArrayDataBuilder;
use inkwell::{
    intrinsics::Intrinsic,
    values::{BasicValue, BasicValueEnum, PointerValue, VectorValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::{
    compiled_writers::{
        AnyRuntime, AnyWriterEmitter, Writer, WriterCodegen, WriterEmitter, WriterRuntime,
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
        let mut buf = Buffer::from(self.alloc);
        let sliced = buf.slice_with_length(0, len * self.pt.width());
        if len > 0 && buf.len() > len * self.pt.width() * 2 {
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

impl PrimitiveWriterRuntime {
    /// Appends terminal integer metadata while finalizing a composed array.
    ///
    /// This is not a general runtime ingestion path. It exists for metadata
    /// such as the final string/list offset or run end and should normally be
    /// called at most once per materialized array.
    pub(super) fn append_integer(&mut self, value: u64) -> Result<(), ArrowKernelError> {
        if self.len() == self.max_len {
            self.reserve_for_additional(1)?;
        }

        unsafe {
            match self.pt {
                PrimitiveType::I16 => self.alloc_ptr.cast::<i16>().write_unaligned(value as i16),
                PrimitiveType::I32 => self.alloc_ptr.cast::<i32>().write_unaligned(value as i32),
                PrimitiveType::I64 => self.alloc_ptr.cast::<i64>().write_unaligned(value as i64),
                pt => {
                    return Err(ArrowKernelError::InternalError(format!(
                        "cannot append integer metadata to primitive writer {pt}"
                    )));
                }
            }
            self.alloc_ptr = self.alloc_ptr.byte_add(self.pt.width());
        }
        Ok(())
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
        let units = (self.pt.width() * size).div_ceil(16);
        let mut alloc = Vec::with_capacity(units);
        // SAFETY: this storage is only ever exposed through to_array, which
        // slices to the prefix the kernel wrote via alloc_ptr. Zero-filling
        // here would memset (and page in) the whole output buffer just for
        // the kernel to overwrite it.
        #[allow(clippy::uninit_vec)]
        unsafe {
            alloc.set_len(units)
        };

        let mut pwr = PrimitiveWriterRuntime {
            alloc,
            alloc_ptr: std::ptr::null_mut(),
            max_len: size,
            pt: self.pt,
        };

        pwr.alloc_ptr = pwr.alloc.as_mut_ptr() as *mut u8;
        pwr.into()
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
        let mut emitter = PrimitiveWriterEmitter {
            codegen,
            runtime_ptr,
            pt: self.pt,
            used: false,
        }
        .into();
        f(&mut emitter)
    }

    fn write_head_offset(&self) -> Option<usize> {
        Some(PrimitiveWriterRuntime::OFFSET_ALLOC_PTR)
    }

    fn llvm_write_block<'ctx, 'borrow>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        values: VectorValue<'ctx>,
        logical_len: u32,
        head_slot: Option<PointerValue<'ctx>>,
    ) -> Result<(), ArrowKernelError> {
        if values.get_type().get_size() != logical_len {
            return Err(ArrowKernelError::InternalError(format!(
                "primitive block has {} lanes for {logical_len} values",
                values.get_type().get_size()
            )));
        }
        let curr_alloc_ptr_ptr = head_slot.unwrap_or_else(|| {
            increment_pointer!(
                codegen.ctx,
                codegen.builder,
                runtime_ptr,
                PrimitiveWriterRuntime::OFFSET_ALLOC_PTR
            )
        });
        let curr_alloc_ptr = codegen
            .builder
            .build_load(
                codegen.ctx.ptr_type(AddressSpace::default()),
                curr_alloc_ptr_ptr,
                "primitive_block_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let store = codegen.builder.build_store(curr_alloc_ptr, values).unwrap();
        store.set_alignment(1).unwrap();
        let new_alloc_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            curr_alloc_ptr,
            self.pt.width() * logical_len as usize
        );
        codegen
            .builder
            .build_store(curr_alloc_ptr_ptr, new_alloc_ptr)
            .unwrap();
        Ok(())
    }

    fn supports_masked_block(&self) -> bool {
        true
    }

    fn llvm_write_block_masked<'ctx, 'borrow>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        values: VectorValue<'ctx>,
        mask: VectorValue<'ctx>,
        head_slot: Option<PointerValue<'ctx>>,
    ) -> Result<(), ArrowKernelError> {
        let ctx = codegen.ctx;
        let b = codegen.builder;
        let curr_alloc_ptr_ptr = head_slot.unwrap_or_else(|| {
            increment_pointer!(ctx, b, runtime_ptr, PrimitiveWriterRuntime::OFFSET_ALLOC_PTR)
        });
        let curr_alloc_ptr = b
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                curr_alloc_ptr_ptr,
                "primitive_masked_ptr",
            )
            .unwrap()
            .into_pointer_value();

        let compress = Intrinsic::find("llvm.masked.compressstore").unwrap();
        let compress = compress
            .get_declaration(codegen.module, &[values.get_type().into()])
            .ok_or_else(|| {
                ArrowKernelError::InternalError("no compressstore for vector type".into())
            })?;
        b.build_call(
            compress,
            &[values.into(), curr_alloc_ptr.into(), mask.into()],
            "compressstore",
        )
        .unwrap();

        // advance the write head by the number of selected lanes
        let lanes = mask.get_type().get_size();
        let mask_int_ty = ctx.custom_width_int_type(lanes);
        let mask_int = b
            .build_bit_cast(mask, mask_int_ty, "mask_int")
            .unwrap()
            .into_int_value();
        let ctpop = Intrinsic::find("llvm.ctpop").unwrap();
        let ctpop = ctpop
            .get_declaration(codegen.module, &[mask_int_ty.into()])
            .unwrap();
        let count = b
            .build_call(ctpop, &[mask_int.into()], "mask_count")
            .unwrap()
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();
        let count = b
            .build_int_z_extend_or_bit_cast(count, ctx.i64_type(), "mask_count64")
            .unwrap();
        let new_alloc_ptr = increment_pointer!(ctx, b, curr_alloc_ptr, self.pt.width(), count);
        b.build_store(curr_alloc_ptr_ptr, new_alloc_ptr).unwrap();
        Ok(())
    }
}

pub struct PrimitiveWriterEmitter<'ctx, 'borrow> {
    codegen: WriterCodegen<'ctx, 'borrow>,
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

        let ptr_type = self.codegen.ctx.ptr_type(AddressSpace::default());
        let curr_alloc_ptr_ptr = increment_pointer!(
            self.codegen.ctx,
            self.codegen.builder,
            self.runtime_ptr,
            PrimitiveWriterRuntime::OFFSET_ALLOC_PTR
        );
        let curr_alloc_ptr = self
            .codegen
            .builder
            .build_load(ptr_type, curr_alloc_ptr_ptr, "curr_alloc_ptr")
            .unwrap()
            .as_basic_value_enum()
            .into_pointer_value();
        self.codegen
            .builder
            .build_store(curr_alloc_ptr, val)
            .unwrap();

        let new_alloc_ptr = increment_pointer!(
            self.codegen.ctx,
            self.codegen.builder,
            curr_alloc_ptr,
            self.pt.width()
        );
        self.codegen
            .builder
            .build_store(curr_alloc_ptr_ptr, new_alloc_ptr)
            .unwrap();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int32Type};
    use inkwell::{
        context::Context, types::VectorType, values::BasicValue, AddressSpace, OptimizationLevel,
    };

    use super::PrimitiveWriter;
    use crate::{
        compiled_writers::{Writer, WriterCodegen, WriterEmitter, WriterRuntime},
        declare_blocks, PrimitiveType,
    };

    #[test]
    fn primitive_writer_materializes_empty_array_with_valid_alignment() {
        let writer = PrimitiveWriter::compile(PrimitiveType::I32).unwrap();
        let runtime = writer.allocate(1);
        let array = runtime.to_array(0).unwrap();
        assert_eq!(array.len(), 0);
        assert_eq!(array.data_type(), &arrow_schema::DataType::Int32);
    }

    #[test]
    fn primitive_writer_jit_writes_values_to_array() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers_primitive_writer");
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
        let codegen = WriterCodegen {
            ctx: &ctx,
            module: &llvm_mod,
            builder: &build,
        };

        writer.llvm_init(codegen, dest);
        for value in [17_i32, -4, 99] {
            writer
                .llvm_write(codegen, dest, |emitter| {
                    emitter.emit(
                        ctx.i32_type()
                            .const_int(value as u64, true)
                            .as_basic_value_enum(),
                    )
                })
                .unwrap();
        }
        let values = [8_i32, 9, 10].map(|value| ctx.i32_type().const_int(value as u64, true));
        let values = VectorType::const_vector(&values);
        writer.llvm_write_block(codegen, dest, values, 3, None).unwrap();
        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();

        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut runtime = writer.allocate(6);
        unsafe {
            f.call(runtime.as_ptr());
        }

        let array = runtime.to_array(6).unwrap();
        let values = array.as_primitive::<Int32Type>().values();
        assert_eq!(values, &[17, -4, 99, 8, 9, 10]);
    }
}
