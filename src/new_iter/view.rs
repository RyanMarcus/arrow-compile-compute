use std::ptr;

use arrow_array::{types::ByteViewType, Array, GenericByteViewArray};
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
    views: *const u128,
    buffers_arr: *const *const u8,
    data_buffers: Vec<*const u8>,
    pos: u64,
    len: u64,
}

impl<T: ByteViewType> From<GenericByteViewArray<T>> for ViewIterator {
    fn from(value: GenericByteViewArray<T>) -> Self {
        let data_buffers: Vec<*const u8> =
            value.data_buffers().iter().map(|db| db.as_ptr()).collect();
        ViewIterator {
            views: value.views().as_ptr(),
            buffers_arr: data_buffers.as_ptr(),
            data_buffers,
            pos: 0,
            len: value.len() as u64,
        }
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
