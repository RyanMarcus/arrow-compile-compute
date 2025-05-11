use crate::writers::ArrayWriter as ArrayWriterTrait;
use crate::{declare_blocks, increment_pointer, pointer_diff, PrimitiveType};
use arrow_array::types::GenericStringType;
use arrow_array::{GenericByteArray, OffsetSizeTrait, StringArray};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use inkwell::intrinsics::Intrinsic;
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    values::{BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace,
};

/// Writer for primitive arrays (ints and floats).
pub struct StringArrayWriter<'a> {
    ingest_func: FunctionValue<'a>,
    offsets_out_ptr_ptr: PointerValue<'a>,
    base_data_ptr: PointerValue<'a>,
    data_out_ptr_ptr: PointerValue<'a>,
}

impl<'a> StringArrayWriter<'a> {
    /// Allocate a new string array writer.
    /// `element_type` is the LLVM type of the offset (either i32 or i64)
    pub fn allocate_array_writer(
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        offsets: PointerValue<'a>,
        data: PointerValue<'a>,
        element_type: PrimitiveType,
    ) -> StringArrayWriter<'a> {
        let ptr_type = ctx.ptr_type(AddressSpace::default());

        // write the initial zero -- each call to ingest will just write the end point
        build
            .build_store(offsets, element_type.llvm_type(ctx).const_zero())
            .unwrap();
        let next_offset = increment_pointer!(ctx, build, offsets, element_type.width());

        let offsets_out_ptr_ptr = build.build_alloca(ptr_type, "offsets_out_ptr_ptr").unwrap();
        build.build_store(offsets_out_ptr_ptr, next_offset).unwrap();

        let data_out_ptr_ptr = build.build_alloca(ptr_type, "data_out_ptr_ptr").unwrap();
        build.build_store(data_out_ptr_ptr, data).unwrap();

        let width = element_type.width();
        let func_name = format!("ingest_str_{}b", width);

        let memcpy = Intrinsic::find("llvm.memcpy").unwrap();
        let memcpy_f = memcpy
            .get_declaration(
                llvm_mod,
                &[ptr_type.into(), ptr_type.into(), ctx.i64_type().into()],
            )
            .unwrap();

        // Create or retrieve the ingest function:
        let ingest_func = llvm_mod.get_function(&func_name).unwrap_or_else(|| {
            let b2 = ctx.create_builder();
            let fn_type = ctx.void_type().fn_type(
                &[
                    ptr_type.into(),
                    ptr_type.into(),
                    ptr_type.into(),
                    PrimitiveType::P64x2.llvm_type(ctx).into(),
                ],
                false,
            );
            let func = llvm_mod.add_function(&func_name, fn_type, Some(Linkage::Private));
            declare_blocks!(ctx, func, entry);
            b2.position_at_end(entry);
            let offsets_out_ptr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
            let data_out_ptr_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
            let base_data_ptr = func.get_nth_param(2).unwrap().into_pointer_value();
            let val = func.get_nth_param(3).unwrap().into_struct_value();

            let ptr1 = b2
                .build_extract_value(val, 0, "ptr1")
                .unwrap()
                .into_pointer_value();
            let ptr2 = b2
                .build_extract_value(val, 1, "ptr2")
                .unwrap()
                .into_pointer_value();

            let data_out_ptr = b2
                .build_load(ptr_type, data_out_ptr_ptr, "data_out_ptr")
                .unwrap()
                .into_pointer_value();
            let offset_out_ptr = b2
                .build_load(ptr_type, offsets_out_ptr_ptr, "offset_out_ptr")
                .unwrap()
                .into_pointer_value();
            let len = pointer_diff!(ctx, b2, ptr1, ptr2);
            let prev_offset = pointer_diff!(ctx, b2, base_data_ptr, data_out_ptr);
            let curr_offset = b2.build_int_add(prev_offset, len, "curr_offset").unwrap();
            let curr_offset = b2
                .build_int_truncate_or_bit_cast(
                    curr_offset,
                    element_type.llvm_type(ctx).into_int_type(),
                    "curr_offset",
                )
                .unwrap();
            b2.build_store(offset_out_ptr, curr_offset).unwrap();
            b2.build_call(
                memcpy_f,
                &[
                    data_out_ptr.into(),
                    ptr1.into(),
                    len.into(),
                    ctx.bool_type().const_zero().into(),
                ],
                "copy_str",
            )
            .unwrap();

            // update both pointers
            let new_offset_ptr = increment_pointer!(ctx, b2, offset_out_ptr, element_type.width());
            let new_data_ptr = increment_pointer!(ctx, b2, data_out_ptr, 1, len);
            b2.build_store(offsets_out_ptr_ptr, new_offset_ptr).unwrap();
            b2.build_store(data_out_ptr_ptr, new_data_ptr).unwrap();

            b2.build_return(None).unwrap();
            func
        });

        StringArrayWriter {
            ingest_func,
            offsets_out_ptr_ptr,
            base_data_ptr: data,
            data_out_ptr_ptr,
        }
    }

    pub unsafe fn array_from_buffers<T: OffsetSizeTrait>(
        mut offsets: Vec<T>,
        mut data: Vec<u8>,
    ) -> GenericByteArray<GenericStringType<T>> {
        let last_offset = offsets.last().map(|o| o.as_usize()).unwrap_or(0);
        data.truncate(last_offset);
        offsets.shrink_to_fit();
        data.shrink_to_fit();

        let offsets = ScalarBuffer::from(offsets);
        let data = Buffer::from(data);

        let offsets = OffsetBuffer::new_unchecked(offsets);
        GenericByteArray::new(offsets, data, None)
    }
}

impl<'a> ArrayWriterTrait<'a> for StringArrayWriter<'a> {
    fn ingest(&self, _ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>) {
        build
            .build_call(
                self.ingest_func,
                &[
                    self.offsets_out_ptr_ptr.into(),
                    self.data_out_ptr_ptr.into(),
                    self.base_data_ptr.into(),
                    val.into(),
                ],
                "ingest",
            )
            .unwrap();
    }

    fn flush(&self, _ctx: &'a Context, _build: &Builder<'a>) {
        // No-op for string arrays
    }
}

#[cfg(test)]
mod tests {
    use super::StringArrayWriter;
    use crate::{declare_blocks, writers::ArrayWriter, PrimitiveType};
    use inkwell::{context::Context, AddressSpace, OptimizationLevel};
    use itertools::Itertools;
    use std::ffi::c_void;

    #[test]
    fn test_string_array_writer() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("test_string_array_writer");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();

        let func = llvm_mod.add_function(
            "test",
            ctx.void_type()
                .fn_type(&[ptr_type.into(), ptr_type.into()], false),
            None,
        );

        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);
        let offsets_out = func.get_nth_param(0).unwrap().into_pointer_value();
        let data_out = func.get_nth_param(1).unwrap().into_pointer_value();
        let writer = StringArrayWriter::allocate_array_writer(
            &ctx,
            &llvm_mod,
            &build,
            offsets_out,
            data_out,
            PrimitiveType::I32,
        );

        let strs = ["this", "is", "a", "test!"];

        for s in strs {
            let ptr1 = s.as_ptr();
            let ptr2 = ptr1.wrapping_add(s.len());

            let px2 = PrimitiveType::P64x2
                .llvm_type(&ctx)
                .into_struct_type()
                .const_named_struct(&[
                    i64_type.const_int(ptr1 as usize as u64, false).into(),
                    i64_type.const_int(ptr2 as usize as u64, false).into(),
                ]);
            writer.ingest(&ctx, &build, px2.into());
        }

        writer.flush(&ctx, &build);
        build.build_return(None).unwrap();

        llvm_mod.verify().unwrap();
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void)>(
                func.get_name().to_str().unwrap(),
            )
            .unwrap()
        };

        let mut offsets = vec![0_i32; 4 + 1];
        let mut data = vec![0_u8; 100];
        unsafe {
            f.call(
                offsets.as_mut_ptr() as *mut c_void,
                data.as_mut_ptr() as *mut c_void,
            );
        }

        let arr = unsafe { StringArrayWriter::array_from_buffers(offsets, data) };
        let arr = arr.iter().map(|s| s.unwrap()).collect_vec();
        assert_eq!(arr, strs);
    }
}
