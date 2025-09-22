use std::sync::Arc;

use arrow_array::{types::ByteViewType, Array, GenericByteViewArray};
use arrow_buffer::Buffer;
use inkwell::{
    builder::Builder,
    context::Context,
    values::{BasicValue, IntValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::increment_pointer;

/// An iterator for view data.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct ViewIterator {
    /// Pointer to the view data. Each entry is either a length and offset, or a full bytestring.
    views: *const u128,

    /// Pointer to an array of pointers to underlying data buffers (indexed by views)
    buffers_arr: *const *const u8,

    /// Held to ensure `buffers_arr` is not dangling
    data_buffers_ptr: Vec<*const u8>,

    /// Held to ensure all entries in `data_buffers_ptr` are valid
    data_buffers: Vec<Buffer>,

    pos: u64,
    len: u64,

    array_ref: Arc<dyn Array>,
}

impl<T: ByteViewType> From<&GenericByteViewArray<T>> for Box<ViewIterator> {
    fn from(value: &GenericByteViewArray<T>) -> Self {
        let data_buffers: Vec<Buffer> = value.data_buffers().to_vec();
        assert!(
            data_buffers.len() < i32::MAX as usize,
            "more than i32::MAX data buffers"
        );
        let data_buffers_ptr: Vec<*const u8> = data_buffers.iter().map(|db| db.as_ptr()).collect();
        Box::new(ViewIterator {
            views: value.views().as_ptr(),
            buffers_arr: data_buffers_ptr.as_ptr(),
            data_buffers_ptr,
            data_buffers,
            pos: 0,
            len: value.len() as u64,
            array_ref: Arc::new(value.clone()),
        })
    }
}

impl ViewIterator {
    pub fn llvm_pos<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let pos_ptr = increment_pointer!(ctx, build, ptr, ViewIterator::OFFSET_POS);
        build
            .build_load(ctx.i64_type(), pos_ptr, "pos")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_len<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let len_ptr = increment_pointer!(ctx, build, ptr, ViewIterator::OFFSET_LEN);
        let len = build
            .build_load(ctx.i64_type(), len_ptr, "len")
            .unwrap()
            .into_int_value();
        len.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        len
    }

    pub fn llvm_view_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let view_ptr_ptr = increment_pointer!(ctx, build, ptr, ViewIterator::OFFSET_VIEWS);
        let view_ptr = build
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                view_ptr_ptr,
                "view_ptr",
            )
            .unwrap()
            .into_pointer_value();
        view_ptr
            .as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        view_ptr
    }

    pub fn llvm_increment_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let pos_ptr = increment_pointer!(ctx, builder, ptr, ViewIterator::OFFSET_POS);
        let pos = builder
            .build_load(ctx.i64_type(), pos_ptr, "pos")
            .unwrap()
            .into_int_value();
        let new_pos = builder.build_int_add(pos, amt, "new_pos").unwrap();
        builder.build_store(pos_ptr, new_pos).unwrap();
    }

    pub fn llvm_buffer_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
        buffer_idx: IntValue<'a>,
    ) -> PointerValue<'a> {
        let buffer_arr_ptr_ptr =
            increment_pointer!(ctx, build, ptr, ViewIterator::OFFSET_BUFFERS_ARR);
        let buffer_arr_ptr = build
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                buffer_arr_ptr_ptr,
                "view_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let buffer_idx = build
            .build_int_s_extend_or_bit_cast(buffer_idx, ctx.i64_type(), "ext_buffer_idx")
            .unwrap();
        let buffer_ptr = increment_pointer!(ctx, build, buffer_arr_ptr, 8, buffer_idx);
        build
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                buffer_ptr,
                "buffer_base",
            )
            .unwrap()
            .into_pointer_value()
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{builder::GenericByteViewBuilder, Array, StringViewArray};
    use inkwell::{context::Context, OptimizationLevel};
    use itertools::Itertools;

    use crate::{
        compiled_iter::{datum_to_iter, generate_random_access},
        pointers_to_str,
    };

    #[test]
    fn test_view_random_access() {
        let strs = vec![
            "this",
            "is",
            "a test",
            "with one string longer than 12 chars",
            "end",
        ];
        let view = StringViewArray::from(strs.clone());
        let mut iter = datum_to_iter(&view).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_string_view_random_access");
        let func_access =
            generate_random_access(&ctx, &module, "access", view.data_type(), &iter).unwrap();
        let fname = func_access.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> u128>(fname)
                .unwrap()
        };

        for (idx, str) in strs.into_iter().enumerate() {
            unsafe {
                let b = func.call(iter.get_mut_ptr(), idx as u64);
                assert_eq!(pointers_to_str(b), str);
            }
        }
    }

    #[test]
    fn test_view_random_access_multiblock() {
        let mut rng = fastrand::Rng::with_seed(42);
        let mut builder = GenericByteViewBuilder::new().with_fixed_block_size(1024);
        let mut gold = Vec::new();

        for _ in 0..1000 {
            let str = (0..32).map(|_| rng.u8(65..=90)).collect_vec();
            let str = String::from_utf8(str).unwrap();
            gold.push(str.clone());
            builder.append_value(str);
        }
        let strs = builder.finish();
        assert!(strs.data_buffers().len() > 8);

        let view = StringViewArray::from(strs.clone());
        let mut iter = datum_to_iter(&view).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_string_view_random_access");
        let func_access =
            generate_random_access(&ctx, &module, "access", view.data_type(), &iter).unwrap();
        let fname = func_access.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> u128>(fname)
                .unwrap()
        };

        for (idx, str) in gold.into_iter().enumerate() {
            unsafe {
                let b = func.call(iter.get_mut_ptr(), idx as u64);
                assert_eq!(pointers_to_str(b), str);
            }
        }
    }
}
