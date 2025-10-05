use std::ffi::c_void;
use std::sync::Arc;

use super::{ArrayWriter, WriterAllocation};
use crate::{declare_blocks, increment_pointer, PrimitiveType};
use arrow_array::{ArrayRef, BooleanArray};
use arrow_buffer::{BooleanBuffer, Buffer};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::types::BasicTypeEnum;
use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};
use inkwell::AddressSpace;
use inkwell::IntPredicate;
use repr_offset::ReprOffset;

pub struct BooleanWriter<'a> {
    ingest_func: FunctionValue<'a>,
    ingest_u64_func: FunctionValue<'a>,
    flush_func: FunctionValue<'a>,
    buf_ptr: PointerValue<'a>,
    buf_idx_ptr: PointerValue<'a>,
    out_ptr_ptr: PointerValue<'a>,
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct BooleanAllocation {
    data_ptr: *mut c_void,
    data: Vec<u8>,
}

impl WriterAllocation for BooleanAllocation {
    type Output = BooleanArray;

    fn get_ptr(&mut self) -> *mut c_void {
        self as *mut Self as *mut c_void
    }

    fn to_array(self, len: usize, nulls: Option<arrow_buffer::NullBuffer>) -> Self::Output {
        let buf = Buffer::from(self.data);
        let bb = BooleanBuffer::new(buf, 0, len);
        BooleanArray::new(bb, nulls)
    }

    fn to_array_ref(self, len: usize, nulls: Option<arrow_buffer::NullBuffer>) -> ArrayRef {
        Arc::new(self.to_array(len, nulls))
    }

    fn add_last_written_offset(&mut self, _offset: usize) {
        unimplemented!("cannot add to offset of boolean writer")
    }
}

impl<'a> ArrayWriter<'a> for BooleanWriter<'a> {
    type Allocation = BooleanAllocation;

    fn allocate(expected_count: usize, ty: PrimitiveType) -> Self::Allocation {
        assert_eq!(ty, PrimitiveType::U8);
        let mut data = vec![0u8; expected_count.div_ceil(8)];
        BooleanAllocation {
            data_ptr: data.as_mut_ptr() as *mut c_void,
            data,
        }
    }

    fn llvm_init(
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        ty: PrimitiveType,
        alloc_ptr: PointerValue<'a>,
    ) -> Self {
        assert_eq!(ty, PrimitiveType::U8);
        let ptr_type = ctx.ptr_type(AddressSpace::default());

        let buf_ptr = build.build_alloca(ctx.i8_type(), "bool_buf_ptr").unwrap();
        build
            .build_store(buf_ptr, ctx.i8_type().const_zero())
            .unwrap();
        let buf_idx_ptr = build
            .build_alloca(ctx.i8_type(), "bool_buf_idx_ptr")
            .unwrap();
        build
            .build_store(buf_idx_ptr, ctx.i8_type().const_zero())
            .unwrap();
        let out_ptr_ptr = build
            .build_alloca(ctx.ptr_type(AddressSpace::default()), "bool_out_ptr_ptr")
            .unwrap();
        let base_out_ptr = build
            .build_load(
                ptr_type,
                increment_pointer!(ctx, build, alloc_ptr, BooleanAllocation::OFFSET_DATA_PTR),
                "base_out_ptr",
            )
            .unwrap()
            .into_pointer_value();
        build.build_store(out_ptr_ptr, base_out_ptr).unwrap();

        let ingest_func = llvm_mod.get_function("ingest_boolean").unwrap_or_else(|| {
            let build = ctx.create_builder();
            let bool_type = ctx.bool_type();
            let i8_type = ctx.i8_type();
            let func = llvm_mod.add_function(
                "ingest_boolean",
                ctx.void_type().fn_type(
                    &[
                        ptr_type.into(),
                        ptr_type.into(),
                        ptr_type.into(),
                        bool_type.into(),
                    ],
                    false,
                ),
                Some(Linkage::Private),
            );

            declare_blocks!(ctx, func, entry, flush_buff, exit);
            build.position_at_end(entry);
            let buf_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
            let buf_idx_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
            let out_ptr_ptr = func.get_nth_param(2).unwrap().into_pointer_value();
            let val = func.get_nth_param(3).unwrap().into_int_value();

            let curr_buf_idx = build
                .build_load(i8_type, buf_idx_ptr, "curr_buf_idx")
                .unwrap()
                .into_int_value();
            let curr_buf = build
                .build_load(i8_type, buf_ptr, "curr_buf")
                .unwrap()
                .into_int_value();
            let val = build
                .build_int_z_extend(val, ctx.i8_type(), "zext")
                .unwrap();
            let val = build
                .build_left_shift(val, curr_buf_idx, "shifted")
                .unwrap();
            let new_buf = build.build_or(curr_buf, val, "new_buf").unwrap();
            build.build_store(buf_ptr, new_buf).unwrap();

            let new_buf_idx = build
                .build_int_add(
                    curr_buf_idx,
                    ctx.i8_type().const_int(1, false),
                    "new_buf_idx",
                )
                .unwrap();
            let need_flush = build
                .build_int_compare(
                    IntPredicate::ULT,
                    new_buf_idx,
                    i8_type.const_int(8, false),
                    "need_flush",
                )
                .unwrap();
            build
                .build_conditional_branch(need_flush, exit, flush_buff)
                .unwrap();

            build.position_at_end(flush_buff);
            let curr_out_ptr = build
                .build_load(ptr_type, out_ptr_ptr, "curr_out_ptr")
                .unwrap()
                .into_pointer_value();
            build.build_store(curr_out_ptr, new_buf).unwrap();
            build.build_store(buf_ptr, i8_type.const_zero()).unwrap();
            build
                .build_store(buf_idx_ptr, i8_type.const_zero())
                .unwrap();
            build
                .build_store(out_ptr_ptr, increment_pointer!(ctx, build, curr_out_ptr, 1))
                .unwrap();
            build.build_return(None).unwrap();

            build.position_at_end(exit);
            build.build_store(buf_idx_ptr, new_buf_idx).unwrap();
            build.build_return(None).unwrap();

            func
        });

        let ingest_u64_func = llvm_mod
            .get_function("ingest_64_booleans")
            .unwrap_or_else(|| {
                let build = ctx.create_builder();
                let i64_type = ctx.i64_type();
                let func = llvm_mod.add_function(
                    "ingest_64_booleans",
                    ctx.void_type()
                        .fn_type(&[ptr_type.into(), i64_type.into()], false),
                    Some(Linkage::Private),
                );

                declare_blocks!(ctx, func, entry);
                build.position_at_end(entry);
                let out_ptr_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
                let val = func.get_nth_param(1).unwrap().into_int_value();

                let curr_out_ptr = build
                    .build_load(ptr_type, out_ptr_ptr, "curr_out_ptr")
                    .unwrap()
                    .into_pointer_value();
                build.build_store(curr_out_ptr, val).unwrap();
                build
                    .build_store(out_ptr_ptr, increment_pointer!(ctx, build, curr_out_ptr, 8))
                    .unwrap();
                build.build_return(None).unwrap();
                func
            });

        let flush_func = llvm_mod.get_function("flush_boolean").unwrap_or_else(|| {
            let build = ctx.create_builder();
            let ptr_type = ctx.ptr_type(AddressSpace::default());
            let i8_type = ctx.i8_type();
            let func = llvm_mod.add_function(
                "flush_boolean",
                ctx.void_type()
                    .fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false),
                Some(Linkage::Private),
            );

            declare_blocks!(ctx, func, entry, flush_buff, exit);
            build.position_at_end(entry);
            let buf_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
            let buf_idx_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
            let out_ptr_ptr = func.get_nth_param(2).unwrap().into_pointer_value();

            let buf_idx = build
                .build_load(i8_type, buf_idx_ptr, "buf_idx")
                .unwrap()
                .into_int_value();
            let have_data = build
                .build_int_compare(
                    IntPredicate::UGT,
                    buf_idx,
                    i8_type.const_zero(),
                    "have_data",
                )
                .unwrap();
            build
                .build_conditional_branch(have_data, flush_buff, exit)
                .unwrap();

            build.position_at_end(flush_buff);
            let curr_out_ptr = build
                .build_load(ptr_type, out_ptr_ptr, "curr_out_ptr")
                .unwrap()
                .into_pointer_value();
            let curr_buf = build
                .build_load(i8_type, buf_ptr, "buf")
                .unwrap()
                .into_int_value();
            build.build_store(curr_out_ptr, curr_buf).unwrap();
            build.build_unconditional_branch(exit).unwrap();

            build.position_at_end(exit);
            build.build_return(None).unwrap();

            func
        });

        BooleanWriter {
            buf_ptr,
            buf_idx_ptr,
            out_ptr_ptr,
            ingest_func,
            ingest_u64_func,
            flush_func,
        }
    }

    fn llvm_ingest_type(&self, ctx: &'a Context) -> inkwell::types::BasicTypeEnum<'a> {
        BasicTypeEnum::IntType(ctx.bool_type())
    }

    fn llvm_ingest(&self, _ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>) {
        let val = val.into_int_value();
        build
            .build_call(
                self.ingest_func,
                &[
                    self.buf_ptr.into(),
                    self.buf_idx_ptr.into(),
                    self.out_ptr_ptr.into(),
                    val.into(),
                ],
                "ingest",
            )
            .unwrap();
    }

    fn llvm_flush(&self, _ctx: &'a Context, build: &Builder<'a>) {
        build
            .build_call(
                self.flush_func,
                &[
                    self.buf_ptr.into(),
                    self.buf_idx_ptr.into(),
                    self.out_ptr_ptr.into(),
                ],
                "flush",
            )
            .unwrap();
    }
}

impl<'a> BooleanWriter<'a> {
    /// Ingests 64 booleans at once from an LLVM i64. This bypasses the writer's
    /// buffer, and writes directly to the output array. Thus, you should not
    /// interleave calls to this function with calls to `llvm_ingest`. Instead,
    /// call this function exclusively for the "body" of the array, then call
    /// `llvm_ingest` for the "tail" of the array.
    pub fn llvm_ingest_64_bools(
        &self,
        _ctx: &'a Context,
        build: &Builder<'a>,
        val: BasicValueEnum<'a>,
    ) {
        build
            .build_call(
                self.ingest_u64_func,
                &[self.out_ptr_ptr.into(), val.into()],
                "ingest_u64",
            )
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::BooleanArray;
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};
    use itertools::Itertools;

    use crate::{
        compiled_kernels::writers::{ArrayWriter, WriterAllocation},
        declare_blocks, PrimitiveType,
    };

    use super::BooleanWriter;

    #[test]
    fn test_bool_writer() {
        let ctx = Context::create();
        let bool_type = ctx.bool_type();
        let llvm_mod = ctx.create_module("test_bool_writer");
        let build = ctx.create_builder();

        let func = llvm_mod.add_function(
            "test",
            ctx.void_type()
                .fn_type(&[ctx.ptr_type(AddressSpace::default()).into()], false),
            None,
        );

        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);
        let dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = BooleanWriter::llvm_init(&ctx, &llvm_mod, &build, PrimitiveType::U8, dest);

        for _ in 0..10 {
            writer.llvm_ingest(
                &ctx,
                &build,
                bool_type.const_all_ones().as_basic_value_enum(),
            );
        }

        for _ in 0..5 {
            writer.llvm_ingest(&ctx, &build, bool_type.const_zero().as_basic_value_enum());
        }

        for _ in 0..10 {
            writer.llvm_ingest(
                &ctx,
                &build,
                bool_type.const_all_ones().as_basic_value_enum(),
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

        let mut data = BooleanWriter::allocate(25, PrimitiveType::U8);

        unsafe {
            f.call(data.get_ptr() as *mut c_void);
        }
        let res = data.to_array(25, None);
        let expected = BooleanArray::from(
            [true]
                .repeat(10)
                .iter()
                .chain([false].repeat(5).iter())
                .chain([true].repeat(10).iter())
                .copied()
                .collect_vec(),
        );
        assert_eq!(res, expected);
    }

    #[test]
    fn test_bool_writer_alternating() {
        let ctx = Context::create();
        let bool_type = ctx.bool_type();
        let llvm_mod = ctx.create_module("test_bool_writer");
        let build = ctx.create_builder();

        let func = llvm_mod.add_function(
            "test",
            ctx.void_type()
                .fn_type(&[ctx.ptr_type(AddressSpace::default()).into()], false),
            None,
        );

        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);
        let dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = BooleanWriter::llvm_init(&ctx, &llvm_mod, &build, PrimitiveType::U8, dest);

        for i in 0..1000 {
            writer.llvm_ingest(
                &ctx,
                &build,
                if i % 2 == 0 {
                    bool_type.const_all_ones().as_basic_value_enum()
                } else {
                    bool_type.const_zero().as_basic_value_enum()
                },
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

        let mut data = BooleanWriter::allocate(1000, PrimitiveType::U8);

        unsafe {
            f.call(data.get_ptr() as *mut c_void);
        }
        let res = data.to_array(1000, None);
        let expected = BooleanArray::from((0..1000).map(|i| i % 2 == 0).collect_vec());
        assert_eq!(res, expected);
    }
}
