use std::{ffi::c_void, sync::Arc};

use crate::{declare_blocks, declare_global_pointer, increment_pointer, PrimitiveType};
use arrow_array::{make_array, ArrayRef, FixedSizeListArray};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::Field;
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    values::{BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use super::{ArrayWriter, WriterAllocation};

/// Writer for fixed-size lists of primitives
pub struct FixedSizeListWriter<'a> {
    ingest_func: FunctionValue<'a>,
    pt: PrimitiveType,
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct FixedSizeListWriterAlloc {
    out_ptr: *mut c_void,
    out: Vec<u128>,
    element_pt: PrimitiveType,
    list_size: usize,
}

impl WriterAllocation for FixedSizeListWriterAlloc {
    type Output = FixedSizeListArray;

    fn get_ptr(&mut self) -> *mut c_void {
        self.out_ptr
    }

    fn to_array(self, len: usize, nulls: Option<NullBuffer>) -> Self::Output {
        let buf = Buffer::from(self.out);
        let buf = buf.slice_with_length(0, len * self.element_pt.width() * self.list_size);
        let ad = unsafe {
            ArrayDataBuilder::new(self.element_pt.as_arrow_type())
                .add_buffer(buf)
                .len(len * self.list_size)
                .build_unchecked()
        };
        let flat_data = make_array(ad);

        FixedSizeListArray::new(
            Arc::new(Field::new_list_field(
                self.element_pt.as_arrow_type(),
                false,
            )),
            self.list_size as i32,
            Arc::new(flat_data),
            nulls,
        )
    }

    fn to_array_ref(self, len: usize, nulls: Option<arrow_buffer::NullBuffer>) -> ArrayRef {
        Arc::new(self.to_array(len, nulls))
    }

    fn add_last_written_offset(&mut self, offset: usize) {
        self.out_ptr = self
            .out_ptr
            .wrapping_add(self.element_pt.width() * self.list_size * offset);
    }
}

impl<'a> ArrayWriter<'a> for FixedSizeListWriter<'a> {
    type Allocation = FixedSizeListWriterAlloc;
    fn allocate(expected_count: usize, ty: PrimitiveType) -> Self::Allocation {
        match ty {
            PrimitiveType::List(el_type, list_size) => {
                let mut data = vec![0_u128; (ty.width() * expected_count).div_ceil(16)];
                assert!(data.capacity() > 0 || expected_count == 0);
                let data_ptr = data.as_mut_ptr() as *mut c_void;
                FixedSizeListWriterAlloc {
                    out_ptr: data_ptr,
                    out: data,
                    list_size,
                    element_pt: el_type.into(),
                }
            }
            _ => panic!("Unsupported type {} for FixedSizeListWriter", ty),
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

        let global_alloc_ptr_ptr =
            declare_global_pointer!(llvm_mod, FSL_WRITER_ALLOC_PTR).as_pointer_value();

        build.build_store(global_alloc_ptr_ptr, alloc_ptr).unwrap();

        let width = ty.width();
        let func_name = format!("ingest_fsl_{}b", width);

        let ingest_func = {
            let b2 = ctx.create_builder();
            let fn_type = ctx.void_type().fn_type(&[ty.llvm_type(ctx).into()], false);
            let func = llvm_mod.add_function(&func_name, fn_type, Some(Linkage::Private));
            declare_blocks!(ctx, func, entry);
            b2.position_at_end(entry);
            let val = func.get_nth_param(0).unwrap();

            let curr_ptr = b2
                .build_load(ptr_type, global_alloc_ptr_ptr, "curr_ptr")
                .unwrap()
                .into_pointer_value();

            b2.build_store(curr_ptr, val).unwrap();
            let new_ptr = increment_pointer!(ctx, b2, curr_ptr, width);
            b2.build_store(global_alloc_ptr_ptr, new_ptr).unwrap();
            b2.build_return(None).unwrap();
            func
        };

        FixedSizeListWriter {
            ingest_func,
            pt: ty,
        }
    }

    fn llvm_ingest_type(&self, ctx: &'a Context) -> inkwell::types::BasicTypeEnum<'a> {
        self.pt.llvm_type(ctx)
    }

    fn llvm_ingest(&self, _ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>) {
        build
            .build_call(self.ingest_func, &[val.into()], "ingest")
            .unwrap();
    }

    fn llvm_flush(&self, _ctx: &'a Context, _build: &Builder<'a>) {
        // no-op
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int32Type};
    use inkwell::{context::Context, AddressSpace, OptimizationLevel};

    use crate::{
        compiled_writers::{
            fixed_size_list_writer::FixedSizeListWriter, ArrayWriter, WriterAllocation,
        },
        declare_blocks, ListItemType, PrimitiveType,
    };

    #[test]
    fn test_fsl_writer_i32() {
        let ctx = Context::create();

        let i32_type = ctx.i32_type();

        let llvm_mod = ctx.create_module("test_fsl_array_writer");
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
        let writer = FixedSizeListWriter::llvm_init(
            &ctx,
            &llvm_mod,
            &build,
            PrimitiveType::List(ListItemType::I32, 4),
            dest,
        );

        for i in 0..10 {
            let to_write = i32_type.vec_type(4).const_zero();
            let to_write = build
                .build_insert_element(
                    to_write,
                    i32_type.const_int(i, true),
                    i32_type.const_int(0, false),
                    "insert0",
                )
                .unwrap();
            let to_write = build
                .build_insert_element(
                    to_write,
                    i32_type.const_int(i + 1, true),
                    i32_type.const_int(1, false),
                    "insert1",
                )
                .unwrap();
            let to_write = build
                .build_insert_element(
                    to_write,
                    i32_type.const_int(i + 2, true),
                    i32_type.const_int(2, false),
                    "insert2",
                )
                .unwrap();
            let to_write = build
                .build_insert_element(
                    to_write,
                    i32_type.const_int(i + 3, true),
                    i32_type.const_int(3, false),
                    "insert3",
                )
                .unwrap();
            writer.llvm_ingest(&ctx, &build, to_write.into());
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

        let mut data = FixedSizeListWriter::allocate(10, PrimitiveType::List(ListItemType::I32, 4));
        unsafe {
            f.call(data.get_ptr());
        }
        let data = data.to_array(10, None);

        for i in 0..10 {
            let el = data.value(i);
            let i = i as i32;
            assert_eq!(
                el.as_primitive::<Int32Type>().values(),
                &[i, i + 1, i + 2, i + 3]
            );
        }
    }
}
