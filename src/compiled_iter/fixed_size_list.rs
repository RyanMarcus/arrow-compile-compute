use std::{ffi::c_void, sync::Arc};

use crate::{increment_pointer, mark_load_invariant, ListItemType, PrimitiveType};
use arrow_array::{Array, FixedSizeListArray};
use inkwell::{
    builder::Builder,
    context::Context,
    values::{BasicValue, IntValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use super::{array_to_iter, IteratorHolder};

/// An iterator for fixed-size list data. Only lists that contain primitive
/// values and strings are supported.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct FixedSizeListIterator {
    child_iter: *const c_void,
    initial_pos: u64,
    pos: u64,
    len: u64,
    list_size: usize,
    list_ptype: ListItemType,
    child: Box<IteratorHolder>,
    pub(super) array_ref: Arc<dyn Array>,
}

impl From<&FixedSizeListArray> for Box<FixedSizeListIterator> {
    fn from(arr: &FixedSizeListArray) -> Self {
        let list_ptype = ListItemType::for_arrow_type(&arr.value_type());
        let child = Box::new(array_to_iter(arr.values().as_ref()));
        let child_iter = child.get_ptr();

        Box::new(FixedSizeListIterator {
            child_iter,
            initial_pos: arr.offset() as u64,
            pos: arr.offset() as u64,
            len: (arr.offset() + arr.len()) as u64,
            list_size: arr.value_length() as usize,
            list_ptype,
            child,
            array_ref: Arc::new(arr.clone()),
        })
    }
}

impl FixedSizeListIterator {
    pub fn ptype(&self) -> PrimitiveType {
        PrimitiveType::List(self.list_ptype, self.list_size)
    }

    pub fn child(&self) -> &IteratorHolder {
        &self.child
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::{ffi::c_void, sync::Arc};

    use arrow_array::{FixedSizeListArray, Int32Array};
    use arrow_data::ArrayData;
    use arrow_schema::{DataType, Field};
    use inkwell::{context::Context, AddressSpace, OptimizationLevel};

    use crate::{compiled_iter::array_to_iter, PrimitiveType};

    fn make_list_array() -> FixedSizeListArray {
        let values = Int32Array::from((0..12).collect::<Vec<_>>());
        let field = Arc::new(Field::new("item", DataType::Int32, false));
        let array_data = ArrayData::builder(DataType::FixedSizeList(field, 4))
            .len(3)
            .add_child_data(values.into_data())
            .build()
            .unwrap();
        FixedSizeListArray::from(array_data)
    }

    #[test]
    fn test_fixed_size_list_random_access() {
        let data = make_list_array();
        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_fixed_size_list_random_access");
        let access_fn = iter.generate_random_access(&ctx, &module).unwrap();

        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i32_type = ctx.i32_type();
        let i64_type = ctx.i64_type();
        let wrap_type = ctx
            .void_type()
            .fn_type(&[ptr_type.into(), i64_type.into(), ptr_type.into()], false);
        let wrap_fn = module.add_function("iter_fixed_list_access_wrapper", wrap_type, None);
        let entry = ctx.append_basic_block(wrap_fn, "entry");
        let builder = ctx.create_builder();
        builder.position_at_end(entry);

        let iter_ptr = wrap_fn.get_nth_param(0).unwrap().into_pointer_value();
        let idx = wrap_fn.get_nth_param(1).unwrap().into_int_value();
        let out_ptr = wrap_fn.get_nth_param(2).unwrap().into_pointer_value();

        let vec_value = builder
            .build_call(access_fn, &[iter_ptr.into(), idx.into()], "vec")
            .unwrap()
            .try_as_basic_value()
            .unwrap_basic()
            .into_vector_value();
        let vec_type = vec_value.get_type();
        let lanes = vec_type.get_size();
        let idx_type = ctx.i32_type();
        for lane in 0..lanes {
            let idx_i32 = idx_type.const_int(lane as u64, false);
            let idx_i64 = i64_type.const_int(lane as u64, false);
            let lane_val_name = format!("lane_{lane}");
            let lane_val = builder
                .build_extract_element(vec_value, idx_i32, lane_val_name.as_str())
                .unwrap()
                .into_int_value();
            let lane_ptr_name = format!("lane_ptr_{lane}");
            let lane_ptr = unsafe {
                builder
                    .build_in_bounds_gep(i32_type, out_ptr, &[idx_i64], lane_ptr_name.as_str())
                    .unwrap()
            };
            builder.build_store(lane_ptr, lane_val).unwrap();
        }
        builder.build_return(None).unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let access = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64, *mut i32)>(
                "iter_fixed_list_access_wrapper",
            )
            .unwrap()
        };

        let mut out = [0_i32; 4];
        unsafe {
            access.call(iter.get_mut_ptr(), 0, out.as_mut_ptr());
            assert_eq!(out, [0, 1, 2, 3]);

            access.call(iter.get_mut_ptr(), 1, out.as_mut_ptr());
            assert_eq!(out, [4, 5, 6, 7]);

            access.call(iter.get_mut_ptr(), 2, out.as_mut_ptr());
            assert_eq!(out, [8, 9, 10, 11]);
        }
    }

    #[test]
    fn test_fixed_size_list_next() {
        let data = make_list_array();
        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_fixed_size_list_next");
        let next_fn = iter.generate_next(&ctx, &module);

        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let bool_type = ctx.bool_type();
        let i32_type = ctx.i32_type();
        let i64_type = ctx.i64_type();
        let (lane_prim, lane_count) = match PrimitiveType::for_arrow_type(data.data_type()) {
            PrimitiveType::List(item, len) => (PrimitiveType::from(item), len as u32),
            _ => unreachable!("expected fixed size list"),
        };
        let vec_type = lane_prim.llvm_vec_type(&ctx, lane_count).unwrap();

        let wrap_type = bool_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let wrap_fn = module.add_function("iter_fixed_list_next_wrapper", wrap_type, None);
        let entry = ctx.append_basic_block(wrap_fn, "entry");
        let builder = ctx.create_builder();
        builder.position_at_end(entry);

        let iter_ptr = wrap_fn.get_nth_param(0).unwrap().into_pointer_value();
        let out_ptr = wrap_fn.get_nth_param(1).unwrap().into_pointer_value();

        let tmp = builder.build_alloca(vec_type, "tmp").unwrap();
        let result = builder
            .build_call(next_fn, &[iter_ptr.into(), tmp.into()], "call")
            .unwrap()
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();

        let store_block = ctx.append_basic_block(wrap_fn, "store");
        let end_block = ctx.append_basic_block(wrap_fn, "end");
        builder
            .build_conditional_branch(result, store_block, end_block)
            .unwrap();

        builder.position_at_end(store_block);
        let vec_value = builder
            .build_load(vec_type, tmp, "vec")
            .unwrap()
            .into_vector_value();
        let lanes = vec_type.get_size();
        let idx_type = ctx.i32_type();
        for lane in 0..lanes {
            let idx_i32 = idx_type.const_int(lane as u64, false);
            let idx_i64 = i64_type.const_int(lane as u64, false);
            let lane_val_name = format!("lane_{lane}");
            let lane_val = builder
                .build_extract_element(vec_value, idx_i32, lane_val_name.as_str())
                .unwrap()
                .into_int_value();
            let lane_ptr_name = format!("lane_ptr_{lane}");
            let lane_ptr = unsafe {
                builder
                    .build_in_bounds_gep(i32_type, out_ptr, &[idx_i64], lane_ptr_name.as_str())
                    .unwrap()
            };
            builder.build_store(lane_ptr, lane_val).unwrap();
        }
        builder.build_unconditional_branch(end_block).unwrap();

        builder.position_at_end(end_block);
        builder.build_return(Some(&result)).unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(
                "iter_fixed_list_next_wrapper",
            )
            .unwrap()
        };

        let mut out = [0_i32; 4];
        unsafe {
            assert!(next.call(iter.get_mut_ptr(), out.as_mut_ptr()));
            assert_eq!(out, [0, 1, 2, 3]);
            out.fill(0);

            assert!(next.call(iter.get_mut_ptr(), out.as_mut_ptr()));
            assert_eq!(out, [4, 5, 6, 7]);
            out.fill(0);

            assert!(next.call(iter.get_mut_ptr(), out.as_mut_ptr()));
            assert_eq!(out, [8, 9, 10, 11]);
            out.fill(0);

            assert!(!next.call(iter.get_mut_ptr(), out.as_mut_ptr()));
        }
    }
}

impl FixedSizeListIterator {
    pub fn llvm_len<'a>(
        &self,
        ctx: &'a Context,
        builder: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let len_ptr = increment_pointer!(ctx, builder, ptr, FixedSizeListIterator::OFFSET_LEN);
        let len = builder
            .build_load(ctx.i64_type(), len_ptr, "len")
            .unwrap()
            .into_int_value();
        len.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        len
    }

    pub fn llvm_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let offset_ptr = increment_pointer!(ctx, builder, ptr, FixedSizeListIterator::OFFSET_POS);
        builder
            .build_load(ctx.i64_type(), offset_ptr, "pos")
            .unwrap()
            .into_int_value()
    }

    pub fn llvm_child_iter<'a>(
        &self,
        ctx: &'a Context,
        builder: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let data_ptr_ptr =
            increment_pointer!(ctx, builder, ptr, FixedSizeListIterator::OFFSET_CHILD_ITER);
        let ptr = builder
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                data_ptr_ptr,
                "data_ptr",
            )
            .unwrap()
            .into_pointer_value();
        mark_load_invariant!(ctx, ptr);
        ptr
    }

    pub fn llvm_increment_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &Builder<'a>,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let pos_ptr = increment_pointer!(ctx, builder, ptr, FixedSizeListIterator::OFFSET_POS);
        let pos = builder
            .build_load(ctx.i64_type(), pos_ptr, "pos")
            .unwrap()
            .into_int_value();
        let new_pos = builder.build_int_add(pos, amt, "new_pos").unwrap();
        builder.build_store(pos_ptr, new_pos).unwrap();
    }

    pub fn llvm_reset<'a>(&self, ctx: &'a Context, builder: &Builder<'a>, ptr: PointerValue<'a>) {
        let pos_ptr = increment_pointer!(ctx, builder, ptr, FixedSizeListIterator::OFFSET_POS);
        let initial_pos_ptr =
            increment_pointer!(ctx, builder, ptr, FixedSizeListIterator::OFFSET_INITIAL_POS);
        let initial_pos = builder
            .build_load(ctx.i64_type(), initial_pos_ptr, "initial_pos")
            .unwrap()
            .into_int_value();
        mark_load_invariant!(ctx, initial_pos);
        builder.build_store(pos_ptr, initial_pos).unwrap();
    }

    pub fn localize_struct<'a>(
        &self,
        _ctx: &'a Context,
        _b: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        ptr
    }
}
