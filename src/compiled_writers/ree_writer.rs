use std::{ffi::c_void, sync::Arc};

use arrow_array::{
    cast::AsArray,
    types::{Int16Type, Int32Type, Int64Type, RunEndIndexType},
    Array, ArrayRef, RunArray,
};
use arrow_buffer::NullBuffer;
use arrow_data::ArrayDataBuilder;
use arrow_schema::{DataType, Field};
use inkwell::{
    module::Linkage,
    values::{FunctionValue, PointerValue},
    AddressSpace, IntPredicate,
};
use repr_offset::ReprOffset;

use super::{
    ArrayOutput, LeafWriter, LeafWriterAllocation, PrimitiveArrayWriter, RunEndType,
    WriterAllocation, WriterSpec,
};
use crate::{
    compiled_kernels::cmp::add_memcmp, declare_blocks, declare_global_pointer, increment_pointer,
    ComparisonType, PrimitiveType,
};

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct REEAllocation {
    res_ptr: *mut c_void,
    values_ptr: *mut c_void,
    last_val: Box<[u8]>,
    num_unique: u64,
    curr_run_end: u64,
    res: Box<ArrayOutput>,
    values: Box<WriterAllocation>,
    run_end_type: RunEndType,
}

impl REEAllocation {
    pub fn allocate(
        expected_count: usize,
        run_end_type: RunEndType,
        value_spec: &WriterSpec,
    ) -> Self {
        let value_type = value_spec.storage_type();
        let mut run_ends = Box::new(PrimitiveArrayWriter::allocate(
            expected_count,
            run_end_type.primitive_type(),
        ));
        let mut values = Box::new(value_spec.allocate(expected_count));
        let last_val = vec![0; value_type.width()].into_boxed_slice();
        Self {
            res_ptr: run_ends.get_ptr(),
            values_ptr: values.get_ptr(),
            last_val,
            num_unique: 0,
            curr_run_end: 0,
            res: run_ends,
            values,
            run_end_type,
        }
    }

    pub fn get_ptr(&mut self) -> *mut c_void {
        self as *mut Self as *mut c_void
    }

    pub fn reserve_for_additional(&mut self, count: usize) {
        if self.curr_run_end > 0 && self.num_unique > 0 {
            self.res.rewind_one();
            self.values.rewind_one();
        }
        self.res.reserve_for_additional(count);
        self.values.reserve_for_additional(count);
        self.res_ptr = self.res.get_ptr();
        self.values_ptr = self.values.get_ptr();
    }

    pub fn len(&self) -> usize {
        usize::try_from(self.curr_run_end).unwrap()
    }

    pub fn into_array_ref_with_len(self, len: usize, nulls: Option<NullBuffer>) -> ArrayRef {
        match self.run_end_type {
            RunEndType::Int16 => Arc::new(self.into_typed_array::<Int16Type>(len, nulls)),
            RunEndType::Int32 => Arc::new(self.into_typed_array::<Int32Type>(len, nulls)),
            RunEndType::Int64 => Arc::new(self.into_typed_array::<Int64Type>(len, nulls)),
        }
    }

    pub fn into_array_ref(self, nulls: Option<NullBuffer>) -> ArrayRef {
        let len = self.len();
        self.into_array_ref_with_len(len, nulls)
    }

    fn into_typed_array<K: RunEndIndexType>(
        self,
        len: usize,
        nulls: Option<NullBuffer>,
    ) -> RunArray<K> {
        let run_ends = (*self.res).to_array(self.num_unique as usize, None);
        let run_ends = run_ends.as_primitive::<K>();
        let values = self
            .values
            .into_array_ref_with_len(self.num_unique as usize, nulls);

        let ree_array_type = DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", K::DATA_TYPE, false)),
            Arc::new(Field::new("values", values.data_type().clone(), true)),
        );

        let logical_len = RunArray::logical_len(run_ends);
        assert_eq!(logical_len, len);
        let builder = ArrayDataBuilder::new(ree_array_type)
            .len(len)
            .add_child_data(run_ends.to_data())
            .add_child_data(values.to_data());
        let array_data = unsafe { builder.build_unchecked() };
        array_data.into()
    }
}

pub struct REEWriter<'a> {
    ingest_func: FunctionValue<'a>,
    flush_func: FunctionValue<'a>,
}

impl<'a> REEWriter<'a> {
    pub fn llvm_init(
        ctx: &'a inkwell::context::Context,
        llvm_mod: &inkwell::module::Module<'a>,
        build: &inkwell::builder::Builder<'a>,
        run_end_type: RunEndType,
        value_spec: &WriterSpec,
        alloc_ptr: PointerValue<'a>,
    ) -> Self {
        match run_end_type {
            RunEndType::Int16 => {
                Self::llvm_init_typed::<Int16Type>(ctx, llvm_mod, build, value_spec, alloc_ptr)
            }
            RunEndType::Int32 => {
                Self::llvm_init_typed::<Int32Type>(ctx, llvm_mod, build, value_spec, alloc_ptr)
            }
            RunEndType::Int64 => {
                Self::llvm_init_typed::<Int64Type>(ctx, llvm_mod, build, value_spec, alloc_ptr)
            }
        }
    }

    pub fn llvm_ingest(
        &self,
        _ctx: &'a inkwell::context::Context,
        build: &inkwell::builder::Builder<'a>,
        val: inkwell::values::BasicValueEnum<'a>,
    ) {
        build
            .build_call(self.ingest_func, &[val.into()], "ingest_ree")
            .unwrap();
    }

    pub fn llvm_flush(
        &self,
        _ctx: &'a inkwell::context::Context,
        build: &inkwell::builder::Builder<'a>,
    ) {
        build.build_call(self.flush_func, &[], "flush").unwrap();
    }

    fn llvm_init_typed<K: RunEndIndexType>(
        ctx: &'a inkwell::context::Context,
        llvm_mod: &inkwell::module::Module<'a>,
        build: &inkwell::builder::Builder<'a>,
        value_spec: &WriterSpec,
        alloc_ptr: PointerValue<'a>,
    ) -> Self {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let value_type = value_spec.storage_type();
        let alloc_ptr_ptr = declare_global_pointer!(llvm_mod, REE_ALLOC_PTR);
        build
            .build_store(alloc_ptr_ptr.as_pointer_value(), alloc_ptr)
            .unwrap();

        let run_end_ptr = build
            .build_load(
                ptr_type,
                increment_pointer!(ctx, build, alloc_ptr, REEAllocation::OFFSET_RES_PTR),
                "re_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let run_end_writer = PrimitiveArrayWriter::llvm_init(
            ctx,
            llvm_mod,
            build,
            PrimitiveType::for_arrow_type(&K::DATA_TYPE),
            run_end_ptr,
        );

        let value_ptr = build
            .build_load(
                ptr_type,
                increment_pointer!(ctx, build, alloc_ptr, REEAllocation::OFFSET_VALUES_PTR),
                "val_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let value_writer = value_spec.llvm_init(ctx, llvm_mod, build, value_ptr);

        let ingest_func = {
            let i64_type = ctx.i64_type();
            let llvm_ty = value_type.llvm_type(ctx);
            let func_type = ctx.void_type().fn_type(&[llvm_ty.into()], false);
            let func = llvm_mod.add_function(
                &format!("ingest_ree_{}_{}", K::DATA_TYPE, value_type),
                func_type,
                Some(Linkage::Private),
            );
            let val_to_insert = func.get_nth_param(0).unwrap();
            declare_blocks!(
                ctx,
                func,
                entry,
                has_value,
                matches_prev,
                insert_first,
                insert_next
            );
            let b2 = ctx.create_builder();
            b2.position_at_end(entry);
            let alloc_ptr = b2
                .build_load(ptr_type, alloc_ptr_ptr.as_pointer_value(), "alloc_ptr")
                .unwrap()
                .into_pointer_value();
            let curr_run_end_ptr =
                increment_pointer!(ctx, b2, alloc_ptr, REEAllocation::OFFSET_CURR_RUN_END);
            let curr_run_end = b2
                .build_load(i64_type, curr_run_end_ptr, "curr_run_end")
                .unwrap()
                .into_int_value();
            let new_run_end = b2
                .build_int_add(curr_run_end, i64_type.const_int(1, true), "new_run_end")
                .unwrap();
            b2.build_store(curr_run_end_ptr, new_run_end).unwrap();
            let have_prev = b2
                .build_int_compare(
                    IntPredicate::NE,
                    curr_run_end,
                    i64_type.const_zero(),
                    "has_value",
                )
                .unwrap();
            let last_val_ptr =
                increment_pointer!(ctx, b2, alloc_ptr, REEAllocation::OFFSET_LAST_VAL);
            let last_val_ptr = b2
                .build_load(ptr_type, last_val_ptr, "last_val_ptr")
                .unwrap()
                .into_pointer_value();
            b2.build_conditional_branch(have_prev, has_value, insert_first)
                .unwrap();

            b2.position_at_end(has_value);
            let last_val = b2.build_load(llvm_ty, last_val_ptr, "last_val").unwrap();
            let matches = match value_type.comparison_type() {
                ComparisonType::Int { .. } | ComparisonType::Float => {
                    let last_val_int = b2
                        .build_bit_cast(
                            last_val,
                            PrimitiveType::int_with_width(value_type.width()).llvm_type(ctx),
                            "last_val_int",
                        )
                        .unwrap()
                        .into_int_value();
                    let new_val_int = b2
                        .build_bit_cast(
                            val_to_insert,
                            PrimitiveType::int_with_width(value_type.width()).llvm_type(ctx),
                            "new_val_int",
                        )
                        .unwrap()
                        .into_int_value();
                    b2.build_int_compare(IntPredicate::EQ, last_val_int, new_val_int, "matches")
                        .unwrap()
                }
                ComparisonType::String => {
                    let memcmp = add_memcmp(ctx, llvm_mod);
                    let cmp = b2
                        .build_call(
                            memcmp,
                            &[last_val.into(), val_to_insert.into()],
                            "memcmp_res",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                        .into_int_value();
                    b2.build_int_compare(IntPredicate::EQ, cmp, i64_type.const_zero(), "matches")
                        .unwrap()
                }
                ComparisonType::List(_) => todo!("implement list ree writer"),
            };
            b2.build_conditional_branch(matches, matches_prev, insert_next)
                .unwrap();

            b2.position_at_end(matches_prev);
            b2.build_return(None).unwrap();

            b2.position_at_end(insert_next);
            let llvm_run_end_type = PrimitiveType::for_arrow_type(&K::DATA_TYPE)
                .llvm_type(ctx)
                .into_int_type();
            let converted = b2
                .build_int_truncate_or_bit_cast(curr_run_end, llvm_run_end_type, "casted_run_end")
                .unwrap();
            run_end_writer.llvm_ingest(ctx, &b2, converted.into());
            value_writer.llvm_ingest(ctx, &b2, last_val);
            b2.build_store(last_val_ptr, val_to_insert).unwrap();
            let num_unique_ptr =
                increment_pointer!(ctx, b2, alloc_ptr, REEAllocation::OFFSET_NUM_UNIQUE);
            let curr_unique = b2
                .build_load(i64_type, num_unique_ptr, "curr_unique")
                .unwrap()
                .into_int_value();
            let new_unique = b2
                .build_int_add(curr_unique, i64_type.const_int(1, false), "new_unique")
                .unwrap();
            b2.build_store(num_unique_ptr, new_unique).unwrap();
            b2.build_return(None).unwrap();

            b2.position_at_end(insert_first);
            b2.build_store(last_val_ptr, val_to_insert).unwrap();
            let num_unique_ptr =
                increment_pointer!(ctx, b2, alloc_ptr, REEAllocation::OFFSET_NUM_UNIQUE);
            let curr_unique = b2
                .build_load(i64_type, num_unique_ptr, "curr_unique")
                .unwrap()
                .into_int_value();
            let new_unique = b2
                .build_int_add(curr_unique, i64_type.const_int(1, false), "new_unique")
                .unwrap();
            b2.build_store(num_unique_ptr, new_unique).unwrap();
            b2.build_return(None).unwrap();
            func
        };

        let flush_func = {
            let i64_type = ctx.i64_type();
            let llvm_ty = value_type.llvm_type(ctx);
            let func_type = ctx.void_type().fn_type(&[], false);
            let func = llvm_mod.add_function(
                &format!("flush_ree_{}_{}", K::DATA_TYPE, value_type),
                func_type,
                Some(Linkage::Private),
            );
            declare_blocks!(ctx, func, entry, has_value, no_value);
            let b2 = ctx.create_builder();
            b2.position_at_end(entry);
            let alloc_ptr = b2
                .build_load(ptr_type, alloc_ptr_ptr.as_pointer_value(), "alloc_ptr")
                .unwrap()
                .into_pointer_value();
            let curr_run_end_ptr =
                increment_pointer!(ctx, b2, alloc_ptr, REEAllocation::OFFSET_CURR_RUN_END);
            let curr_run_end = b2
                .build_load(i64_type, curr_run_end_ptr, "curr_run_end")
                .unwrap()
                .into_int_value();
            let is_empty = b2
                .build_int_compare(
                    IntPredicate::EQ,
                    curr_run_end,
                    i64_type.const_zero(),
                    "no_value",
                )
                .unwrap();
            b2.build_conditional_branch(is_empty, no_value, has_value)
                .unwrap();

            b2.position_at_end(has_value);
            let last_val_ptr =
                increment_pointer!(ctx, b2, alloc_ptr, REEAllocation::OFFSET_LAST_VAL);
            let last_val_ptr = b2
                .build_load(ptr_type, last_val_ptr, "last_val_ptr")
                .unwrap()
                .into_pointer_value();
            let llvm_run_end_type = PrimitiveType::for_arrow_type(&K::DATA_TYPE)
                .llvm_type(ctx)
                .into_int_type();
            let converted_run_end = b2
                .build_int_truncate_or_bit_cast(curr_run_end, llvm_run_end_type, "casted_run_end")
                .unwrap();
            let last_val = b2.build_load(llvm_ty, last_val_ptr, "last_val").unwrap();
            run_end_writer.llvm_ingest(ctx, &b2, converted_run_end.into());
            value_writer.llvm_ingest(ctx, &b2, last_val);
            b2.build_unconditional_branch(no_value).unwrap();

            b2.position_at_end(no_value);
            run_end_writer.llvm_flush(ctx, &b2);
            value_writer.llvm_flush(ctx, &b2);
            b2.build_return(None).unwrap();
            func
        };

        Self {
            ingest_func,
            flush_func,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{types::Int32Type, BinaryArray, Int32Array, RunArray};
    use inkwell::{context::Context, AddressSpace, OptimizationLevel};
    use itertools::Itertools;

    use crate::{
        compiled_kernels::link_req_helpers,
        compiled_writers::{RunEndType, WriterSpec},
        declare_blocks, PrimitiveType,
    };

    #[test]
    fn test_ree_writer_int_full() {
        let data: Vec<i32> = vec![1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 50];
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("test_primitive_array_writer");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let spec = WriterSpec::RunEndEncoded(
            RunEndType::Int32,
            Box::new(WriterSpec::Primitive(PrimitiveType::I32)),
        );

        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );

        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);
        let dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = spec.llvm_init(&ctx, &llvm_mod, &build, dest);

        for el in &data {
            writer.llvm_ingest(
                &ctx,
                &build,
                ctx.i32_type().const_int(*el as u64, true).into(),
            );
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

        let mut alloc = spec.allocate(100);
        unsafe {
            f.call(alloc.get_ptr());
        }
        alloc.reserve_for_additional(data.len());
        unsafe {
            f.call(alloc.get_ptr());
        }
        let ree = alloc.into_array_ref_with_len(2 * data.len(), None);
        let ree = ree.as_any().downcast_ref::<RunArray<Int32Type>>().unwrap();
        let ree = ree.downcast::<Int32Array>().unwrap();
        let ree_vec = ree.into_iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(
            ree_vec,
            data.iter().chain(data.iter()).copied().collect_vec()
        );
    }

    #[test]
    fn test_ree_writer_int_partial() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("test_primitive_array_writer");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let spec = WriterSpec::RunEndEncoded(
            RunEndType::Int64,
            Box::new(WriterSpec::Primitive(PrimitiveType::I32)),
        );

        let func = llvm_mod.add_function(
            "test",
            ctx.void_type()
                .fn_type(&[ptr_type.into(), ctx.i32_type().into()], false),
            None,
        );

        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);
        let dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let val = func.get_nth_param(1).unwrap().into_int_value();
        let writer = spec.llvm_init(&ctx, &llvm_mod, &build, dest);

        writer.llvm_ingest(&ctx, &build, val.into());
        build.build_return(None).unwrap();

        llvm_mod.verify().unwrap();
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&llvm_mod, &ee).unwrap();

        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, i32)>(
                func.get_name().to_str().unwrap(),
            )
            .unwrap()
        };

        let mut alloc = spec.allocate(100);

        unsafe {
            f.call(alloc.get_ptr(), 1);
        }
        match &alloc {
            crate::compiled_writers::WriterAllocation::RunEndEncoded(alloc) => {
                assert_eq!(alloc.curr_run_end, 1);
                assert_eq!(
                    i32::from_le_bytes(alloc.last_val[0..4].try_into().unwrap()),
                    1
                );
                assert_eq!(alloc.num_unique, 1);
            }
            _ => unreachable!(),
        }

        unsafe {
            f.call(alloc.get_ptr(), 1);
        }
        match &alloc {
            crate::compiled_writers::WriterAllocation::RunEndEncoded(alloc) => {
                assert_eq!(alloc.curr_run_end, 2);
                assert_eq!(
                    i32::from_le_bytes(alloc.last_val[0..4].try_into().unwrap()),
                    1
                );
                assert_eq!(alloc.num_unique, 1);
            }
            _ => unreachable!(),
        }

        unsafe {
            f.call(alloc.get_ptr(), 1);
        }
        match &alloc {
            crate::compiled_writers::WriterAllocation::RunEndEncoded(alloc) => {
                assert_eq!(alloc.curr_run_end, 3);
                assert_eq!(
                    i32::from_le_bytes(alloc.last_val[0..4].try_into().unwrap()),
                    1
                );
                assert_eq!(alloc.num_unique, 1);
            }
            _ => unreachable!(),
        }

        unsafe {
            f.call(alloc.get_ptr(), 2);
        }
        match &alloc {
            crate::compiled_writers::WriterAllocation::RunEndEncoded(alloc) => {
                assert_eq!(alloc.curr_run_end, 4);
                assert_eq!(
                    i32::from_le_bytes(alloc.last_val[0..4].try_into().unwrap()),
                    2
                );
                assert_eq!(alloc.num_unique, 2);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_ree_writer_str_full() {
        let data: Vec<&str> = vec!["hello", "this", "this", "is", "is", "a test"];
        let ctx = Context::create();
        let i64_type = ctx.i64_type();
        let llvm_mod = ctx.create_module("test_array_writer");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let spec = WriterSpec::RunEndEncoded(RunEndType::Int32, Box::new(WriterSpec::String));

        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );

        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);
        let dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = spec.llvm_init(&ctx, &llvm_mod, &build, dest);

        for el in &data {
            let start = el.as_ptr();
            let end = start.wrapping_add(el.len());
            let px2 = PrimitiveType::P64x2
                .llvm_type(&ctx)
                .into_struct_type()
                .const_named_struct(&[
                    i64_type.const_int(start as usize as u64, false).into(),
                    i64_type.const_int(end as usize as u64, false).into(),
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

        let mut alloc = spec.allocate(100);

        unsafe {
            f.call(alloc.get_ptr());
        }
        alloc.reserve_for_additional(data.len());
        unsafe {
            f.call(alloc.get_ptr());
        }
        let ree = alloc.into_array_ref_with_len(2 * data.len(), None);
        let ree = ree.as_any().downcast_ref::<RunArray<Int32Type>>().unwrap();
        let ree = ree.downcast::<BinaryArray>().unwrap();
        let ree_vec = ree
            .into_iter()
            .map(|x| std::str::from_utf8(x.unwrap()).unwrap())
            .collect_vec();
        assert_eq!(
            ree_vec,
            data.iter().chain(data.iter()).copied().collect_vec()
        );
    }

    #[test]
    fn test_ree_writer_to_array_ref_uses_logical_len() {
        let data: Vec<i32> = vec![1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 50];
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("test_ree_writer_to_array_ref");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let spec = WriterSpec::RunEndEncoded(
            RunEndType::Int32,
            Box::new(WriterSpec::Primitive(PrimitiveType::I32)),
        );

        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );

        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);
        let dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = spec.llvm_init(&ctx, &llvm_mod, &build, dest);

        for el in &data {
            writer.llvm_ingest(
                &ctx,
                &build,
                ctx.i32_type().const_int(*el as u64, true).into(),
            );
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

        let mut alloc = spec.allocate(data.len());

        unsafe {
            f.call(alloc.get_ptr());
        }

        let ree = alloc.into_array_ref(None);
        let ree = ree.as_any().downcast_ref::<RunArray<Int32Type>>().unwrap();
        let ree = ree.downcast::<Int32Array>().unwrap();
        let ree_vec = ree.into_iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(ree_vec, data);
    }
}
