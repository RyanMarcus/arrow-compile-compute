use super::ArrayWriter;
use crate::{declare_blocks, increment_pointer};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};
use inkwell::AddressSpace;
use inkwell::IntPredicate;

pub struct BooleanWriter<'a> {
    ingest_func: FunctionValue<'a>,
    flush_func: FunctionValue<'a>,
    buf_ptr: PointerValue<'a>,
    buf_idx_ptr: PointerValue<'a>,
    out_ptr_ptr: PointerValue<'a>,
}

impl<'a> BooleanWriter<'a> {
    pub fn allocate_boolean_writer(
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        dest: PointerValue<'a>,
    ) -> BooleanWriter<'a> {
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
        build.build_store(out_ptr_ptr, dest).unwrap();

        let ingest_func = llvm_mod.get_function("ingest_boolean").unwrap_or_else(|| {
            let build = ctx.create_builder();
            let ptr_type = ctx.ptr_type(AddressSpace::default());
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
            flush_func,
        }
    }
}
impl<'a> ArrayWriter<'a> for BooleanWriter<'a> {
    fn ingest(&self, _ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>) {
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

    fn flush(&self, _ctx: &'a Context, build: &Builder<'a>) {
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

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::BooleanArray;
    use arrow_buffer::BooleanBuffer;
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};
    use itertools::Itertools;

    use crate::{declare_blocks, writers::ArrayWriter};

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
        let writer = BooleanWriter::allocate_boolean_writer(&ctx, &llvm_mod, &build, dest);

        for _ in 0..10 {
            writer.ingest(
                &ctx,
                &build,
                bool_type.const_all_ones().as_basic_value_enum(),
            );
        }

        for _ in 0..5 {
            writer.ingest(&ctx, &build, bool_type.const_zero().as_basic_value_enum());
        }

        for _ in 0..10 {
            writer.ingest(
                &ctx,
                &build,
                bool_type.const_all_ones().as_basic_value_enum(),
            );
        }

        writer.flush(&ctx, &build);
        build.build_return(None).unwrap();

        llvm_mod.verify().unwrap();
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };

        let mut data = vec![0_u8; 25_usize.div_ceil(8)];
        unsafe {
            f.call(data.as_mut_ptr() as *mut c_void);
        }
        let res = BooleanArray::new(BooleanBuffer::new(data.into(), 0, 25), None);
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
}
