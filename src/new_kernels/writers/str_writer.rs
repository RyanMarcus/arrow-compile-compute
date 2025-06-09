use std::ffi::c_void;
use std::marker::PhantomData;

use crate::{declare_blocks, increment_pointer, pointer_diff, PrimitiveType};
use arrow_array::{GenericBinaryArray, GenericByteArray, OffsetSizeTrait};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    values::{BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use super::{ArrayWriter, WriterAllocation};

/// Writer for string arrays (utf8 or bytes)
pub struct StringArrayWriter<'a, T: OffsetSizeTrait> {
    ingest_func: FunctionValue<'a>,
    offsets_out_ptr_ptr: PointerValue<'a>,
    data_out_vec_ptr: PointerValue<'a>,
    last_offset_ptr: PointerValue<'a>,
    _pd: PhantomData<T>,
}

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct StringAllocation<T: OffsetSizeTrait> {
    offsets_ptr: *mut c_void,
    offsets: Vec<T>,
    data: Vec<u8>,
    pt: PrimitiveType,
}

impl<T: OffsetSizeTrait> WriterAllocation for StringAllocation<T> {
    type Output = GenericBinaryArray<T>;

    fn get_ptr(&mut self) -> *mut c_void {
        self as *mut Self as *mut c_void
    }

    fn to_array(mut self, len: usize, nulls: Option<arrow_buffer::NullBuffer>) -> Self::Output {
        self.offsets.truncate(len + 1);
        let last_offset = self.offsets.last().map(|o| o.as_usize()).unwrap_or(0);
        self.data.truncate(last_offset);

        self.offsets.shrink_to_fit();
        self.data.shrink_to_fit();

        let offsets = ScalarBuffer::from(self.offsets);
        let data = Buffer::from(self.data);

        let offsets = unsafe { OffsetBuffer::new_unchecked(offsets) };
        GenericByteArray::new(offsets, data, nulls)
    }
}

impl<'a, T: OffsetSizeTrait> ArrayWriter<'a> for StringArrayWriter<'a, T> {
    type Allocation = StringAllocation<T>;

    fn allocate(expected_count: usize, ty: PrimitiveType) -> Self::Allocation {
        let mut offsets = vec![T::zero(); (expected_count + 1) * ty.width()];
        let data = Vec::with_capacity(expected_count);
        let offsets_ptr = offsets.as_mut_ptr() as *mut c_void;
        StringAllocation {
            offsets_ptr,
            offsets,
            data,
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
        assert_eq!(ty, PrimitiveType::P64x2, "string writer type must be P64x2");
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();
        let offset_element_type = if T::IS_LARGE {
            ctx.i64_type()
        } else {
            ctx.i32_type()
        };

        let offset_ptr_ptr = increment_pointer!(
            ctx,
            build,
            alloc_ptr,
            StringAllocation::<T>::OFFSET_OFFSETS_PTR
        );
        let offset_ptr = build
            .build_load(ptr_type, offset_ptr_ptr, "offset_base")
            .unwrap()
            .into_pointer_value();

        // write the initial zero -- each call to ingest will just write the end point
        build
            .build_store(offset_ptr, offset_element_type.const_zero())
            .unwrap();
        let next_offset = increment_pointer!(ctx, build, offset_ptr, T::get_byte_width());
        build.build_store(offset_ptr_ptr, next_offset).unwrap();

        let last_offset_ptr = build.build_alloca(ptr_type, "last_offset_ptr").unwrap();
        build
            .build_store(last_offset_ptr, i64_type.const_zero())
            .unwrap();

        let extend_f = llvm_mod
            .get_function("str_writer_append_bytes")
            .unwrap_or_else(|| {
                llvm_mod.add_function(
                    "str_writer_append_bytes",
                    ctx.void_type()
                        .fn_type(&[ptr_type.into(), i64_type.into(), ptr_type.into()], false),
                    Some(Linkage::External),
                )
            });

        // Create or retrieve the ingest function:
        let func_name = format!("ingest_str_{}b", T::get_byte_width());
        let ingest_func = llvm_mod.get_function(&func_name).unwrap_or_else(|| {
            let b2 = ctx.create_builder();
            let fn_type = ctx.void_type().fn_type(
                &[
                    ptr_type.into(), // offsets
                    ptr_type.into(), // vec ptr
                    ptr_type.into(), // last offset ptr
                    PrimitiveType::P64x2.llvm_type(ctx).into(),
                ],
                false,
            );
            let func = llvm_mod.add_function(&func_name, fn_type, Some(Linkage::Private));
            declare_blocks!(ctx, func, entry);
            b2.position_at_end(entry);
            let offsets_out_ptr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
            let data_out_vec_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
            let last_offset_ptr = func.get_nth_param(2).unwrap().into_pointer_value();
            let val = func.get_nth_param(3).unwrap().into_struct_value();

            let ptr1 = b2
                .build_extract_value(val, 0, "ptr1")
                .unwrap()
                .into_pointer_value();
            let ptr2 = b2
                .build_extract_value(val, 1, "ptr2")
                .unwrap()
                .into_pointer_value();

            let offset_out_ptr = b2
                .build_load(ptr_type, offsets_out_ptr_ptr, "offset_out_ptr")
                .unwrap()
                .into_pointer_value();
            let len = pointer_diff!(ctx, b2, ptr1, ptr2);
            let prev_offset = b2
                .build_load(i64_type, last_offset_ptr, "last_offset")
                .unwrap()
                .into_int_value();
            let curr_offset = b2.build_int_add(prev_offset, len, "curr_offset").unwrap();
            let curr_offset = b2
                .build_int_truncate_or_bit_cast(curr_offset, offset_element_type, "curr_offset")
                .unwrap();
            b2.build_store(offset_out_ptr, curr_offset).unwrap();
            b2.build_call(
                extend_f,
                &[ptr1.into(), len.into(), data_out_vec_ptr.into()],
                "extend",
            )
            .unwrap();

            // update pointers
            let new_offset_ptr = increment_pointer!(ctx, b2, offset_out_ptr, T::get_byte_width());
            b2.build_store(offsets_out_ptr_ptr, new_offset_ptr).unwrap();
            b2.build_store(last_offset_ptr, curr_offset).unwrap();

            b2.build_return(None).unwrap();
            func
        });

        StringArrayWriter {
            ingest_func,
            offsets_out_ptr_ptr: offset_ptr_ptr,
            data_out_vec_ptr: increment_pointer!(
                ctx,
                build,
                alloc_ptr,
                StringAllocation::<T>::OFFSET_DATA
            ),
            last_offset_ptr,
            _pd: PhantomData,
        }
    }

    fn llvm_ingest(&self, _ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>) {
        build
            .build_call(
                self.ingest_func,
                &[
                    self.offsets_out_ptr_ptr.into(),
                    self.data_out_vec_ptr.into(),
                    self.last_offset_ptr.into(),
                    val.into(),
                ],
                "ingest",
            )
            .unwrap();
    }

    fn llvm_flush(&self, _ctx: &'a Context, _build: &Builder<'a>) {
        // No-op for string arrays
    }
}

#[cfg(test)]
mod tests {
    use super::StringArrayWriter;
    use crate::{
        declare_blocks,
        new_kernels::{
            link_req_helpers,
            writers::{ArrayWriter, WriterAllocation},
        },
        PrimitiveType,
    };
    use arrow_array::StringArray;
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
        let alloc_ptr = func.get_nth_param(0).unwrap().into_pointer_value();

        let writer = StringArrayWriter::<i32>::llvm_init(
            &ctx,
            &llvm_mod,
            &build,
            PrimitiveType::P64x2,
            alloc_ptr,
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
            writer.llvm_ingest(&ctx, &build, px2.into());
        }

        writer.llvm_flush(&ctx, &build);
        build.build_return(None).unwrap();

        llvm_mod.verify().unwrap();
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&llvm_mod, &ee).unwrap();

        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut alloc = StringArrayWriter::allocate(100, PrimitiveType::I32);

        unsafe {
            f.call(alloc.get_ptr());
        }

        let arr = alloc.to_array(4, None);
        let arr = StringArray::from(arr);
        let arr = arr.iter().map(|s| s.unwrap()).collect_vec();
        assert_eq!(arr, strs);
    }
}
