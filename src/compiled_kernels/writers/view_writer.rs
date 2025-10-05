use std::{ffi::c_void, sync::Arc};

use arrow_array::{types::BinaryViewType, ArrayRef, GenericByteViewArray};
use arrow_buffer::{Buffer, NullBuffer};
use inkwell::{
    context::Context, intrinsics::Intrinsic, module::Linkage, values::FunctionValue, AddressSpace,
    IntPredicate,
};
use repr_offset::ReprOffset;

use crate::{
    compiled_kernels::writers::{ArrayWriter, WriterAllocation},
    declare_blocks, declare_global_pointer, increment_pointer, pointer_diff, PrimitiveType,
};

pub struct ViewBufferWriter {
    buffers: Vec<Vec<u8>>,
}

impl ViewBufferWriter {
    pub fn new() -> Self {
        ViewBufferWriter {
            buffers: vec![Vec::with_capacity(4096)],
        }
    }

    pub fn write(&mut self, data: &[u8]) -> u128 {
        debug_assert!(data.len() > 12);
        let last_buf = self.buffers.last_mut().unwrap();
        let offset = if last_buf.capacity() >= data.len() {
            let offset = last_buf.len() as i32;
            last_buf.extend_from_slice(data);
            offset
        } else {
            let next_size =
                usize::max(1024 * 1024 * 64, 2 * (last_buf.len() + last_buf.capacity()));
            let next_size = usize::max(next_size, data.len()); // for > 64MB strings
            let mut next_buf = Vec::with_capacity(next_size);
            next_buf.extend_from_slice(data);
            self.buffers.push(next_buf);
            0
        };

        let buffer_index = self.buffers.len() as i32 - 1;
        let len = data.len() as i32;
        let prefix = i32::from_le_bytes(data[0..4].try_into().unwrap());

        // casting to a `u32` first is required so that the sign bit doesn't get
        // extended
        (len as u128)
            | ((prefix as u32) as u128) << 32
            | ((buffer_index as u32) as u128) << 64
            | ((offset as u32) as u128) << 96
    }

    pub fn into_buffers(self) -> Vec<Buffer> {
        self.buffers.into_iter().map(Buffer::from).collect()
    }
}

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct StringViewAllocation {
    views_ptr: *mut u128,
    views: Vec<u128>,
    data: ViewBufferWriter,
}

impl WriterAllocation for StringViewAllocation {
    type Output = GenericByteViewArray<BinaryViewType>;

    fn get_ptr(&mut self) -> *mut c_void {
        self as *mut Self as *mut c_void
    }

    fn to_array(mut self, len: usize, nulls: Option<NullBuffer>) -> Self::Output {
        self.views.truncate(len);
        let buffers = self.data.into_buffers();
        #[cfg(test)]
        {
            Self::Output::new(self.views.clone().into(), buffers.clone(), nulls.clone());
        }
        unsafe { GenericByteViewArray::new_unchecked(self.views.into(), buffers, nulls) }
    }

    fn to_array_ref(self, len: usize, nulls: Option<arrow_buffer::NullBuffer>) -> ArrayRef {
        Arc::new(self.to_array(len, nulls))
    }

    fn add_last_written_offset(&mut self, _offset: usize) {
        // do nothing -- pointed updated by LLVM gen'd code
    }
}

/// Writer for view arrays (utf8 or bytes)
pub struct StringViewWriter<'a> {
    ingest_func: FunctionValue<'a>,
}

impl<'a> ArrayWriter<'a> for StringViewWriter<'a> {
    type Allocation = StringViewAllocation;

    fn allocate(expected_count: usize, ty: crate::PrimitiveType) -> Self::Allocation {
        assert_eq!(
            ty,
            crate::PrimitiveType::P64x2,
            "string view must have double pointer type"
        );

        let mut views = vec![0_u128; expected_count];
        let data = ViewBufferWriter::new();
        StringViewAllocation {
            views_ptr: views.as_mut_ptr(),
            views,
            data,
        }
    }

    fn llvm_init(
        ctx: &'a inkwell::context::Context,
        llvm_mod: &inkwell::module::Module<'a>,
        build: &inkwell::builder::Builder<'a>,
        ty: crate::PrimitiveType,
        alloc_ptr: inkwell::values::PointerValue<'a>,
    ) -> Self {
        assert_eq!(
            ty,
            crate::PrimitiveType::P64x2,
            "string view must have double pointer type"
        );
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i128_type = ctx.i128_type();
        let i64_type = ctx.i64_type();

        let alloc_global = declare_global_pointer!(llvm_mod, STRING_VIEW_ALLOC_PTR_PTR);
        build
            .build_store(alloc_global.as_pointer_value(), alloc_ptr)
            .unwrap();

        let extend_f = llvm_mod
            .get_function("str_view_writer_append_bytes")
            .unwrap_or_else(|| {
                llvm_mod.add_function(
                    "str_view_writer_append_bytes",
                    ctx.i128_type().fn_type(
                        &[
                            ptr_type.into(),
                            i64_type.into(),
                            ptr_type.into(),
                            ptr_type.into(),
                        ],
                        false,
                    ),
                    Some(Linkage::External),
                )
            });

        let memcpy = Intrinsic::find("llvm.memcpy").unwrap();
        let memcpy_f = memcpy
            .get_declaration(
                llvm_mod,
                &[ptr_type.into(), ptr_type.into(), i64_type.into()],
            )
            .unwrap();

        // Create or retrieve the ingest function:
        let ingest_func_name = "ingest_str_view";
        let ingest_func = {
            let b2 = ctx.create_builder();
            let fn_type = ctx
                .void_type()
                .fn_type(&[PrimitiveType::P64x2.llvm_type(ctx).into()], false);

            let func = llvm_mod.add_function(ingest_func_name, fn_type, Some(Linkage::Private));
            declare_blocks!(ctx, func, entry, short_string, long_string, exit);
            b2.position_at_end(entry);
            let val = func.get_nth_param(0).unwrap().into_struct_value();
            let alloc_ptr = b2
                .build_load(ptr_type, alloc_global.as_pointer_value(), "alloc_ptr")
                .unwrap()
                .into_pointer_value();
            let out_ptr_ptr =
                increment_pointer!(ctx, b2, alloc_ptr, StringViewAllocation::OFFSET_VIEWS_PTR);
            let out_ptr = b2
                .build_load(ptr_type, out_ptr_ptr, "view_ptr")
                .unwrap()
                .into_pointer_value();
            let ptr1 = b2
                .build_extract_value(val, 0, "ptr1")
                .unwrap()
                .into_pointer_value();
            let ptr2 = b2
                .build_extract_value(val, 1, "ptr2")
                .unwrap()
                .into_pointer_value();
            let len = pointer_diff!(ctx, b2, ptr1, ptr2);
            let cmp = b2
                .build_int_compare(IntPredicate::SLE, len, i64_type.const_int(12, true), "cmp")
                .unwrap();
            let tmp_view_ptr = b2.build_alloca(i128_type, "tmp_view_ptr").unwrap();
            let view_with_len = b2
                .build_int_z_extend(len, i128_type, "view_with_len")
                .unwrap();
            b2.build_store(tmp_view_ptr, view_with_len).unwrap();
            b2.build_conditional_branch(cmp, short_string, long_string)
                .unwrap();

            b2.position_at_end(short_string);
            // pack the entire string into a view
            b2.build_call(
                memcpy_f,
                &[
                    increment_pointer!(ctx, b2, tmp_view_ptr, 4).into(),
                    ptr1.into(),
                    len.into(),
                    ctx.bool_type().const_zero().into(),
                ],
                "memcpy",
            )
            .unwrap();
            let tmp_view = b2.build_load(i128_type, tmp_view_ptr, "tmp_view").unwrap();
            b2.build_store(out_ptr, tmp_view).unwrap();
            b2.build_unconditional_branch(exit).unwrap();

            b2.position_at_end(long_string);
            // pack the prefix into a view, add rest to buffers
            let writer_ptr =
                increment_pointer!(ctx, b2, alloc_ptr, StringViewAllocation::OFFSET_DATA);
            b2.build_call(
                extend_f,
                &[
                    ptr1.into(),       // str ptr
                    len.into(),        // str len
                    out_ptr.into(),    // out ptr
                    writer_ptr.into(), // data ptr
                ],
                "extend",
            )
            .unwrap();
            b2.build_unconditional_branch(exit).unwrap();

            b2.position_at_end(exit);
            let next_out_ptr = increment_pointer!(ctx, b2, out_ptr, 16);
            b2.build_store(out_ptr_ptr, next_out_ptr).unwrap();
            b2.build_return(None).unwrap();

            func
        };

        StringViewWriter { ingest_func }
    }

    fn llvm_ingest_type(&self, ctx: &'a Context) -> inkwell::types::BasicTypeEnum<'a> {
        PrimitiveType::P64x2.llvm_type(ctx)
    }

    fn llvm_ingest(
        &self,
        _ctx: &'a inkwell::context::Context,
        build: &inkwell::builder::Builder<'a>,
        val: inkwell::values::BasicValueEnum<'a>,
    ) {
        build
            .build_call(self.ingest_func, &[val.into()], "ingest_view")
            .unwrap();
    }

    fn llvm_flush(
        &self,
        _ctx: &'a inkwell::context::Context,
        _build: &inkwell::builder::Builder<'a>,
    ) {
        // no op
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use inkwell::{context::Context, AddressSpace, OptimizationLevel};
    use itertools::Itertools;

    use crate::{
        compiled_kernels::{
            link_req_helpers,
            writers::{
                view_writer::{StringViewWriter, ViewBufferWriter},
                ArrayWriter, WriterAllocation,
            },
        },
        declare_blocks, PrimitiveType,
    };

    #[test]
    fn test_view_buffer_writer() {
        let mut writer = ViewBufferWriter::new();
        writer.write(b"Hello, this is a test!");
        writer.write(b"Hello world, this is another test");
        let buffers = writer.into_buffers();
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].len(), 55);
    }

    #[test]
    fn test_view_buffer_writer_large_strs() {
        let mut writer = ViewBufferWriter::new();
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let s = (0..8 * 1024).map(|_| rng.u8(..)).collect::<Vec<_>>();
            assert!(writer.write(&s) > 0);
        }
        let buffers = writer.into_buffers();
        assert!(buffers.len() > 1);
    }

    #[test]
    fn test_view_writer() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("test_string_view_writer");
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

        let writer =
            StringViewWriter::llvm_init(&ctx, &llvm_mod, &build, PrimitiveType::P64x2, alloc_ptr);

        let strs = [
            "this",
            "is",
            "a",
            "test!",
            "with at least one very long string that must be buffered",
            "...",
            "and another very long string that must be buffered",
        ];

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

        let mut alloc = StringViewWriter::allocate(100, PrimitiveType::P64x2);

        unsafe {
            f.call(alloc.get_ptr());
        }

        let arr = alloc.to_array(strs.len(), None);
        let arr = arr
            .iter()
            .map(|s| std::str::from_utf8(s.unwrap()).unwrap())
            .collect_vec();
        assert_eq!(arr, strs);
    }
}
