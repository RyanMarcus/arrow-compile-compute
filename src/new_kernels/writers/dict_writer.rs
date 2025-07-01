use std::{ffi::c_void, marker::PhantomData};

use arrow_array::{
    cast::AsArray, make_array, types::ArrowDictionaryKeyType, Array, DictionaryArray,
};

use inkwell::{
    module::Linkage,
    values::{FunctionValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::{
    declare_blocks, increment_pointer,
    new_kernels::{
        ht::{generate_hash_func, generate_lookup_or_insert, TicketTable},
        writers::{array_writer::ArrayOutput, ArrayWriter, PrimitiveArrayWriter, WriterAllocation},
    },
    PrimitiveType,
};

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct DictAllocation<'a, K: ArrowDictionaryKeyType, VW: ArrayWriter<'a>> {
    tt: TicketTable,
    keys_ptr: *mut c_void,
    values_ptr: *mut c_void,
    keys: Box<ArrayOutput>,
    values: Box<VW::Allocation>,

    #[roff(offset = "OFFSET_MARKER")]
    _marker: PhantomData<K>,
}

impl<'a, K: ArrowDictionaryKeyType, VW: ArrayWriter<'a>> WriterAllocation
    for DictAllocation<'a, K, VW>
{
    type Output = DictionaryArray<K>;

    fn get_ptr(&mut self) -> *mut c_void {
        self as *mut Self as *mut c_void
    }

    fn to_array(self, len: usize, nulls: Option<arrow_buffer::NullBuffer>) -> Self::Output {
        let keys = self.keys.to_array(len, nulls);
        let keys = keys.as_primitive::<K>().clone();
        let values = make_array(self.values.to_array(self.tt.len(), None).to_data());
        unsafe { DictionaryArray::<K>::new_unchecked(keys, values) }
    }
}

pub struct DictWriter<'a, K: ArrowDictionaryKeyType, VW: ArrayWriter<'a>> {
    ht_ptr: PointerValue<'a>,
    ingest_func: FunctionValue<'a>,
    _phantom1: std::marker::PhantomData<K>,
    _phantom2: std::marker::PhantomData<VW>,
}

impl<'a, K: ArrowDictionaryKeyType, VW: ArrayWriter<'a>> ArrayWriter<'a> for DictWriter<'a, K, VW> {
    type Allocation = DictAllocation<'a, K, VW>;

    fn allocate(expected_count: usize, ty: PrimitiveType) -> Self::Allocation {
        let tt = TicketTable::new(expected_count * 2, ty.as_arrow_type(), K::DATA_TYPE);
        let mut kw = Box::new(PrimitiveArrayWriter::allocate(
            expected_count,
            PrimitiveType::for_arrow_type(&K::DATA_TYPE),
        ));
        let mut vw = Box::new(VW::allocate(expected_count, ty));
        DictAllocation {
            tt,
            keys_ptr: kw.get_ptr(),
            values_ptr: vw.get_ptr(),
            keys: kw,
            values: vw,
            _marker: PhantomData,
        }
    }

    fn llvm_init(
        ctx: &'a inkwell::context::Context,
        llvm_mod: &inkwell::module::Module<'a>,
        build: &inkwell::builder::Builder<'a>,
        ty: PrimitiveType,
        alloc_ptr: inkwell::values::PointerValue<'a>,
    ) -> Self {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let kw = PrimitiveArrayWriter::llvm_init(
            ctx,
            llvm_mod,
            build,
            PrimitiveType::for_arrow_type(&K::DATA_TYPE),
            build
                .build_load(
                    ptr_type,
                    increment_pointer!(
                        ctx,
                        build,
                        alloc_ptr,
                        DictAllocation::<K, VW>::OFFSET_KEYS_PTR
                    ),
                    "keys_ptr",
                )
                .unwrap()
                .into_pointer_value(),
        );
        let vw_ptr = build
            .build_load(
                ptr_type,
                increment_pointer!(
                    ctx,
                    build,
                    alloc_ptr,
                    DictAllocation::<K, VW>::OFFSET_VALUES_PTR
                ),
                "values_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let vw = VW::llvm_init(ctx, llvm_mod, build, ty, vw_ptr);

        let i8_type = ctx.i8_type();

        let dummy_ht = TicketTable::new(0, ty.as_arrow_type(), K::DATA_TYPE);
        let hash_func = generate_hash_func(ctx, llvm_mod, ty);
        let ht_lookup = generate_lookup_or_insert(ctx, llvm_mod, &dummy_ht);

        let ht_ptr = increment_pointer!(ctx, build, alloc_ptr, DictAllocation::<K, VW>::OFFSET_TT);

        let ingest_func = {
            let b = ctx.create_builder();
            let func_type = ctx.bool_type().fn_type(
                &[
                    ptr_type.into(),          // pointer to HT
                    ty.llvm_type(ctx).into(), // value to ingest
                ],
                false,
            );
            let func =
                llvm_mod.add_function("dict_writer_ingest", func_type, Some(Linkage::Private));

            let ht_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
            let value = func.get_nth_param(1).unwrap();

            declare_blocks!(ctx, func, entry, is_new, not_new, table_full);
            b.position_at_end(entry);
            let hash = b
                .build_call(hash_func, &[value.into()], "hash")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            let is_new_ptr = b.build_alloca(ctx.i8_type(), "is_new_ptr").unwrap();
            let ticket_val = b
                .build_call(
                    ht_lookup,
                    &[ht_ptr.into(), value.into(), hash.into(), is_new_ptr.into()],
                    "ht_lookup",
                )
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            let status = b
                .build_load(i8_type, is_new_ptr, "status")
                .unwrap()
                .into_int_value();
            b.build_switch(
                status,
                table_full,
                &[
                    (i8_type.const_int(0, false), not_new),
                    (i8_type.const_int(1, false), is_new),
                    (i8_type.const_int(2, false), table_full),
                ],
            )
            .unwrap();

            b.position_at_end(not_new);
            kw.llvm_ingest(ctx, &b, ticket_val.into());
            b.build_return(Some(&ctx.bool_type().const_int(1, false)))
                .unwrap();

            b.position_at_end(is_new);
            kw.llvm_ingest(ctx, &b, ticket_val.into());
            vw.llvm_ingest(ctx, &b, value);
            b.build_return(Some(&ctx.bool_type().const_int(1, false)))
                .unwrap();

            b.position_at_end(table_full);
            b.build_return(Some(&ctx.bool_type().const_int(0, false)))
                .unwrap();

            func
        };

        DictWriter {
            ht_ptr,
            ingest_func,
            _phantom1: std::marker::PhantomData,
            _phantom2: std::marker::PhantomData,
        }
    }

    fn llvm_ingest(
        &self,
        _ctx: &'a inkwell::context::Context,
        build: &inkwell::builder::Builder<'a>,
        val: inkwell::values::BasicValueEnum<'a>,
    ) {
        build
            .build_call(
                self.ingest_func,
                &[self.ht_ptr.into(), val.into()],
                "dict_ingest",
            )
            .unwrap();
    }

    fn llvm_flush(
        &self,
        _ctx: &'a inkwell::context::Context,
        _build: &inkwell::builder::Builder<'a>,
    ) {
        // no-op for dictionary writer
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{types::Int8Type, Int32Array};
    use inkwell::{context::Context, AddressSpace, OptimizationLevel};
    use itertools::Itertools;

    use crate::{
        declare_blocks,
        new_kernels::{
            link_req_helpers,
            writers::{
                dict_writer::DictWriter, ArrayWriter, PrimitiveArrayWriter, WriterAllocation,
            },
        },
        PrimitiveType,
    };

    #[test]
    fn test_dict_writer() {
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
        let writer = DictWriter::<Int8Type, PrimitiveArrayWriter>::llvm_init(
            &ctx,
            &llvm_mod,
            &build,
            PrimitiveType::I32,
            dest,
        );

        let mut expected = Vec::new();
        for _ in 0..10 {
            for i in 1000..1010 {
                writer.llvm_ingest(&ctx, &build, ctx.i32_type().const_int(i, true).into());
                expected.push(i as i32);
            }
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

        let mut data =
            DictWriter::<Int8Type, PrimitiveArrayWriter>::allocate(100, PrimitiveType::I32);
        unsafe {
            f.call(data.get_ptr());
        }
        let data = data.to_array(100, None);
        let data = data.downcast_dict::<Int32Array>().unwrap();
        let data: Vec<i32> = data.into_iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(data, expected);
    }
}
