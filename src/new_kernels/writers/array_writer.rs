use std::ffi::c_void;

use crate::{declare_blocks, increment_pointer, PrimitiveType};
use arrow_array::{make_array, ArrayRef, ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_data::ArrayDataBuilder;
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    values::{BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use super::{ArrayWriter, WriterAllocation};

/// Writer for primitive arrays (ints and floats).
pub struct PrimitiveArrayWriter<'a> {
    ingest_func: FunctionValue<'a>,
    global_alloc_ptr_ptr: PointerValue<'a>,
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct ArrayOutput {
    out_ptr: *mut c_void,
    out: Vec<u128>,
    pt: PrimitiveType,
}

impl WriterAllocation for ArrayOutput {
    type Output = ArrayRef;

    fn get_ptr(&mut self) -> *mut c_void {
        self.out_ptr
    }
    fn to_array(self, len: usize, nulls: Option<NullBuffer>) -> Self::Output {
        let buf = Buffer::from(self.out);
        let buf = buf.slice_with_length(0, len * self.pt.width());
        let ad = unsafe {
            ArrayDataBuilder::new(self.pt.as_arrow_type())
                .add_buffer(buf)
                .nulls(nulls)
                .len(len)
                .build_unchecked()
        };
        make_array(ad)
    }
}

impl ArrayOutput {
    pub fn into_primitive_array<T: ArrowPrimitiveType>(
        self,
        len: usize,
        nulls: Option<NullBuffer>,
    ) -> PrimitiveArray<T> {
        let buf = Buffer::from(self.out);
        let buf = buf.slice_with_length(0, len * self.pt.width());
        let buf = buf.slice_with_length(0, len * self.pt.width());
        let ad = unsafe {
            ArrayDataBuilder::new(T::DATA_TYPE)
                .add_buffer(buf)
                .nulls(nulls)
                .len(len)
                .build_unchecked()
        };
        PrimitiveArray::<T>::from(ad)
    }
}

impl<'a> ArrayWriter<'a> for PrimitiveArrayWriter<'a> {
    type Allocation = ArrayOutput;
    fn allocate(expected_count: usize, ty: PrimitiveType) -> Self::Allocation {
        let mut data = vec![0_u128; (ty.width() * expected_count).div_ceil(16)];
        assert!(data.capacity() > 0 || expected_count == 0);
        let data_ptr = data.as_mut_ptr() as *mut c_void;
        ArrayOutput {
            out_ptr: data_ptr,
            out: data,
            pt: ty,
        }
    }

    fn llvm_init(
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        ty: PrimitiveType,
        alloc_ptr: PointerValue<'a>,
    ) -> Self {
        let ptr_type = ctx.ptr_type(AddressSpace::default());

        let global_alloc_ptr = llvm_mod.add_global(ptr_type, None, "ARRAY_WRITER_ALLOC_PTR");
        global_alloc_ptr.set_initializer(&ptr_type.const_null());
        global_alloc_ptr.set_linkage(Linkage::Private);
        let global_alloc_ptr_ptr = global_alloc_ptr.as_pointer_value();

        build.build_store(global_alloc_ptr_ptr, alloc_ptr).unwrap();

        let width = ty.width();
        let func_name = format!("ingest_prim_{}b", width);

        // Create or retrieve the ingest function:
        let ingest_func = llvm_mod.get_function(&func_name).unwrap_or_else(|| {
            let b2 = ctx.create_builder();
            let fn_type = ctx
                .void_type()
                .fn_type(&[ptr_type.into(), ty.llvm_type(ctx).into()], false);
            let func = llvm_mod.add_function(&func_name, fn_type, Some(Linkage::Private));
            declare_blocks!(ctx, func, entry);
            b2.position_at_end(entry);
            let out_ptr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
            let val = func.get_nth_param(1).unwrap();

            let curr_ptr = b2
                .build_load(ptr_type, out_ptr_ptr, "curr_ptr")
                .unwrap()
                .into_pointer_value();

            b2.build_store(curr_ptr, val).unwrap();
            let new_ptr = increment_pointer!(ctx, b2, curr_ptr, width);
            b2.build_store(out_ptr_ptr, new_ptr).unwrap();
            b2.build_return(None).unwrap();
            func
        });

        PrimitiveArrayWriter {
            ingest_func,
            global_alloc_ptr_ptr,
        }
    }

    fn llvm_ingest(&self, _ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>) {
        build
            .build_call(
                self.ingest_func,
                &[self.global_alloc_ptr_ptr.into(), val.into()],
                "ingest",
            )
            .unwrap();
    }

    fn llvm_flush(&self, _ctx: &'a Context, _build: &Builder<'a>) {
        // No-op for primitive arrays
    }
}

#[cfg(test)]
mod tests {
    use super::PrimitiveArrayWriter;
    use crate::{
        declare_blocks,
        new_kernels::writers::{ArrayWriter, WriterAllocation},
        PrimitiveType,
    };
    use arrow_array::{cast::AsArray, types::Int32Type};
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};
    use std::ffi::c_void;

    #[test]
    fn test_primitive_array_writer() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("test_primitive_array_writer");
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
        let writer =
            PrimitiveArrayWriter::llvm_init(&ctx, &llvm_mod, &build, PrimitiveType::I32, dest);

        for i in 0..10 {
            writer.llvm_ingest(
                &ctx,
                &build,
                ctx.i32_type().const_int(i, true).as_basic_value_enum(),
            );
        }

        writer.llvm_flush(&ctx, &build);
        build.build_return(None).unwrap();

        llvm_mod.verify().unwrap();
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut data = PrimitiveArrayWriter::allocate(10, PrimitiveType::I32);
        unsafe {
            f.call(data.get_ptr());
        }
        let data = data.to_array(10, None);
        let data = data.as_primitive::<Int32Type>().values();

        assert_eq!(data, &(0..10).collect::<Vec<_>>());
    }
}
