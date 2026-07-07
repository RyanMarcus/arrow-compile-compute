use std::{ffi::c_void, sync::Arc};

use arrow_array::{Array, GenericListArray};
use inkwell::{
    builder::Builder,
    context::Context,
    values::{IntValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::{increment_pointer, mark_load_invariant};

use super::{array_to_iter, IteratorHolder};

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct ListIterator {
    offsets: *const c_void,
    child_iter: *const c_void,
    initial_pos: u64,
    pos: u64,
    len: u64,
    offset_width: usize,
    child: Box<IteratorHolder>,
    array_ref: Arc<dyn Array>,
}

impl From<&GenericListArray<i32>> for Box<ListIterator> {
    fn from(arr: &GenericListArray<i32>) -> Self {
        let child = Box::new(array_to_iter(arr.values().as_ref()));
        Box::new(ListIterator {
            offsets: arr.offsets().as_ptr() as *const c_void,
            child_iter: child.get_ptr(),
            initial_pos: arr.offset() as u64,
            pos: arr.offset() as u64,
            len: (arr.offset() + arr.len()) as u64,
            offset_width: 4,
            child,
            array_ref: Arc::new(arr.clone()),
        })
    }
}

impl From<&GenericListArray<i64>> for Box<ListIterator> {
    fn from(arr: &GenericListArray<i64>) -> Self {
        let child = Box::new(array_to_iter(arr.values().as_ref()));
        Box::new(ListIterator {
            offsets: arr.offsets().as_ptr() as *const c_void,
            child_iter: child.get_ptr(),
            initial_pos: arr.offset() as u64,
            pos: arr.offset() as u64,
            len: (arr.offset() + arr.len()) as u64,
            offset_width: 8,
            child,
            array_ref: Arc::new(arr.clone()),
        })
    }
}

impl ListIterator {
    pub fn child(&self) -> &IteratorHolder {
        &self.child
    }

    pub fn offset_width(&self) -> usize {
        self.offset_width
    }

    pub fn llvm_offsets<'a>(
        &self,
        ctx: &'a Context,
        builder: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let ptr_ptr = increment_pointer!(ctx, builder, ptr, ListIterator::OFFSET_OFFSETS);
        let ptr = builder
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                ptr_ptr,
                "list_offsets",
            )
            .unwrap()
            .into_pointer_value();
        mark_load_invariant!(ctx, ptr);
        ptr
    }

    pub fn llvm_child_iter<'a>(
        &self,
        ctx: &'a Context,
        builder: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let ptr_ptr = increment_pointer!(ctx, builder, ptr, ListIterator::OFFSET_CHILD_ITER);
        let ptr = builder
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                ptr_ptr,
                "list_child_iter",
            )
            .unwrap()
            .into_pointer_value();
        mark_load_invariant!(ctx, ptr);
        ptr
    }

    pub fn llvm_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let ptr = increment_pointer!(ctx, builder, ptr, ListIterator::OFFSET_POS);
        builder
            .build_load(ctx.i64_type(), ptr, "list_pos")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_len<'a>(
        &self,
        ctx: &'a Context,
        builder: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let ptr = increment_pointer!(ctx, builder, ptr, ListIterator::OFFSET_LEN);
        let len = builder
            .build_load(ctx.i64_type(), ptr, "list_len")
            .unwrap()
            .into_int_value();
        mark_load_invariant!(ctx, len);
        len
    }

    pub fn llvm_increment_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &Builder<'a>,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let pos_ptr = increment_pointer!(ctx, builder, ptr, ListIterator::OFFSET_POS);
        let pos = builder
            .build_load(ctx.i64_type(), pos_ptr, "list_pos")
            .unwrap()
            .into_int_value();
        let new_pos = builder.build_int_add(pos, amt, "list_new_pos").unwrap();
        builder.build_store(pos_ptr, new_pos).unwrap();
    }

    pub fn llvm_reset<'a>(&self, ctx: &'a Context, builder: &Builder<'a>, ptr: PointerValue<'a>) {
        let pos_ptr = increment_pointer!(ctx, builder, ptr, ListIterator::OFFSET_POS);
        let initial_pos_ptr =
            increment_pointer!(ctx, builder, ptr, ListIterator::OFFSET_INITIAL_POS);
        let initial_pos = builder
            .build_load(ctx.i64_type(), initial_pos_ptr, "list_initial_pos")
            .unwrap()
            .into_int_value();
        mark_load_invariant!(ctx, initial_pos);
        builder.build_store(pos_ptr, initial_pos).unwrap();
    }

    /// Gets the list offset at `idx` as a 64-bit signed integer.
    pub fn llvm_offset_at<'a>(
        &self,
        ctx: &'a Context,
        builder: &Builder<'a>,
        offsets: PointerValue<'a>,
        idx: IntValue<'a>,
    ) -> IntValue<'a> {
        match self.offset_width {
            4 => {
                let raw = builder
                    .build_load(
                        ctx.i32_type(),
                        increment_pointer!(ctx, builder, offsets, 4, idx),
                        "list_offset_i32",
                    )
                    .unwrap()
                    .into_int_value();
                builder
                    .build_int_s_extend(raw, ctx.i64_type(), "list_offset_i64")
                    .unwrap()
            }
            8 => builder
                .build_load(
                    ctx.i64_type(),
                    increment_pointer!(ctx, builder, offsets, 8, idx),
                    "list_offset_i64",
                )
                .unwrap()
                .into_int_value(),
            _ => unreachable!("invalid list offset width"),
        }
    }
}
