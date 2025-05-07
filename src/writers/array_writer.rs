use crate::writers::ArrayWriter as ArrayWriterTrait;
use crate::{declare_blocks, increment_pointer, PrimitiveType};
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    values::{BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace,
};

/// Writer for primitive arrays (ints and floats).
pub struct PrimitiveArrayWriter<'a> {
    ingest_func: FunctionValue<'a>,
    out_ptr_ptr: PointerValue<'a>,
}

impl<'a> PrimitiveArrayWriter<'a> {
    /// Allocate a new primitive array writer.
    /// `element_type` is the LLVM type of each element (e.g., i32, f64).
    pub fn allocate_array_writer(
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        dest: PointerValue<'a>,
        element_type: PrimitiveType,
    ) -> PrimitiveArrayWriter<'a> {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let out_ptr_ptr = build.build_alloca(ptr_type, "out_ptr_ptr").unwrap();
        build.build_store(out_ptr_ptr, dest).unwrap();

        let width = element_type.width();
        let func_name = format!("ingest_prim_{}b", width);

        // Create or retrieve the ingest function:
        let ingest_func = llvm_mod.get_function(&func_name).unwrap_or_else(|| {
            let b2 = ctx.create_builder();
            let fn_type = ctx.void_type().fn_type(
                &[ptr_type.into(), element_type.llvm_type(ctx).into()],
                false,
            );
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
            out_ptr_ptr,
        }
    }
}

impl<'a> ArrayWriterTrait<'a> for PrimitiveArrayWriter<'a> {
    fn ingest(&self, _ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>) {
        build
            .build_call(
                self.ingest_func,
                &[self.out_ptr_ptr.into(), val.into()],
                "ingest",
            )
            .unwrap();
    }

    fn flush(&self, _ctx: &'a Context, _build: &Builder<'a>) {
        // No-op for primitive arrays
    }
}

#[cfg(test)]
mod tests {
    use super::PrimitiveArrayWriter;
    use crate::{declare_blocks, writers::ArrayWriter, PrimitiveType};
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
        let writer = PrimitiveArrayWriter::allocate_array_writer(
            &ctx,
            &llvm_mod,
            &build,
            dest,
            PrimitiveType::I32,
        );

        for i in 0..10 {
            writer.ingest(
                &ctx,
                &build,
                ctx.i32_type().const_int(i, true).as_basic_value_enum(),
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

        let mut data = vec![0_i32; 10];
        unsafe {
            f.call(data.as_mut_ptr() as *mut c_void);
        }

        assert_eq!(data, (0..10).collect::<Vec<_>>());
    }
}
