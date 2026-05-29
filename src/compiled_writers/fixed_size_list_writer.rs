use std::{ffi::c_void, sync::Arc};

use crate::{
    declare_blocks, declare_global_pointer, increment_pointer, ListItemType, PrimitiveType,
};
use arrow_array::{builder::BinaryBuilder, make_array, ArrayRef, BooleanArray, FixedSizeListArray};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::Field;
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    values::{BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use super::{LeafWriter, LeafWriterAllocation};

/// Writer for fixed-size lists of primitives
pub struct FixedSizeListWriter<'a> {
    ingest_func: FunctionValue<'a>,
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct FixedSizeListWriterAlloc {
    out_ptr: *mut c_void,
    out: Vec<u128>,
    list_item: ListItemType,
    element_pt: PrimitiveType,
    list_size: usize,
}

impl LeafWriterAllocation for FixedSizeListWriterAlloc {
    type Output = FixedSizeListArray;

    fn get_ptr(&mut self) -> *mut c_void {
        &mut self.out_ptr as *mut *mut c_void as *mut c_void
    }

    fn reserve_for_additional(&mut self, count: usize) {
        unsafe {
            let bytes_written = self
                .out_ptr
                .offset_from_unsigned(self.out.as_ptr() as *mut c_void);
            let items_to_preserve = bytes_written.div_ceil(16);
            self.out.set_len(items_to_preserve);
            self.out.resize(
                items_to_preserve + (count * self.element_pt.width() * self.list_size).div_ceil(16),
                0,
            );
            let new_base = self.out.as_mut_ptr() as *mut c_void;
            self.out_ptr = new_base.byte_add(bytes_written);
        }
    }

    fn rewind_one(&mut self) {
        unsafe {
            let width = self.element_pt.width() * self.list_size;
            let base = self.out.as_mut_ptr() as *mut c_void;
            let bytes_written = self.out_ptr.offset_from_unsigned(base);
            self.out_ptr = base.byte_add(bytes_written - width);
        }
    }

    fn to_array(self, len: usize, nulls: Option<NullBuffer>) -> Self::Output {
        let list_size = self.list_size;
        let element_pt = self.element_pt;
        let list_item = self.list_item;
        let out = self.out;
        let values_len = len * list_size;
        let flat_data = match element_pt {
            PrimitiveType::U8 if matches!(list_item, ListItemType::Boolean) => {
                let bytes =
                    unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, values_len) };
                let bools = bytes.iter().map(|v| *v != 0).collect::<Vec<_>>();
                Arc::new(BooleanArray::from(bools)) as ArrayRef
            }
            PrimitiveType::P64x2 => {
                let ptrs =
                    unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u128, values_len) };
                let mut b = BinaryBuilder::new();
                for &v in ptrs {
                    let start_ptr = (v as u64) as *const u8;
                    let end_ptr = ((v >> 64) as u64) as *const u8;
                    assert!(end_ptr >= start_ptr);
                    let value = unsafe {
                        let len = end_ptr.offset_from_unsigned(start_ptr);
                        std::slice::from_raw_parts(start_ptr, len)
                    };
                    b.append_value(value);
                }
                Arc::new(b.finish()) as ArrayRef
            }
            _ => {
                let buf = Buffer::from(out);
                let buf = buf.slice_with_length(0, len * element_pt.width() * list_size);
                let ad = unsafe {
                    ArrayDataBuilder::new(element_pt.as_arrow_type())
                        .add_buffer(buf)
                        .len(values_len)
                        .build_unchecked()
                };
                make_array(ad)
            }
        };

        FixedSizeListArray::new(
            Arc::new(Field::new_list_field(flat_data.data_type().clone(), false)),
            list_size as i32,
            flat_data,
            nulls,
        )
    }

    fn to_array_ref(self, nulls: Option<arrow_buffer::NullBuffer>) -> ArrayRef {
        let len = self.len();
        Arc::new(self.to_array(len, nulls))
    }

    fn len(&self) -> usize {
        let offset = unsafe { self.out_ptr.byte_offset_from(self.out.as_ptr()) };
        let offset = usize::try_from(offset).unwrap();
        let width = self.element_pt.width() * self.list_size;
        assert_eq!(offset % width, 0);
        offset / width
    }
}

impl<'a> LeafWriter<'a> for FixedSizeListWriter<'a> {
    type Allocation = FixedSizeListWriterAlloc;
    fn allocate(expected_count: usize, ty: PrimitiveType) -> Self::Allocation {
        match ty {
            PrimitiveType::List(el_type, list_size) => {
                let mut data = vec![0_u128; (ty.width() * expected_count).div_ceil(16)];
                assert!(data.capacity() > 0 || expected_count == 0);
                let data_ptr = data.as_mut_ptr() as *mut c_void;
                FixedSizeListWriterAlloc {
                    out_ptr: data_ptr,
                    out: data,
                    list_size,
                    list_item: el_type,
                    element_pt: el_type.physical_primitive_type(),
                }
            }
            _ => panic!("Unsupported type {} for FixedSizeListWriter", ty),
        }
    }

    fn llvm_init(
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        ty: PrimitiveType,
        alloc_ptr: PointerValue<'a>,
    ) -> Self {
        let ptr_type = ctx.ptr_type(AddressSpace::default());

        let global_alloc_ptr_ptr =
            declare_global_pointer!(llvm_mod, FSL_WRITER_ALLOC_PTR).as_pointer_value();

        build.build_store(global_alloc_ptr_ptr, alloc_ptr).unwrap();

        let width = ty.width();
        let func_name = format!("ingest_fsl_{ty}")
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>();

        let ingest_func = {
            let b2 = ctx.create_builder();
            let fn_type = ctx.void_type().fn_type(&[ty.llvm_type(ctx).into()], false);
            let func = llvm_mod.add_function(&func_name, fn_type, Some(Linkage::Private));
            declare_blocks!(ctx, func, entry);
            b2.position_at_end(entry);
            let val = func.get_nth_param(0).unwrap();

            let alloc_ptr = b2
                .build_load(ptr_type, global_alloc_ptr_ptr, "alloc_ptr")
                .unwrap()
                .into_pointer_value();
            let curr_ptr = b2
                .build_load(ptr_type, alloc_ptr, "curr_ptr")
                .unwrap()
                .into_pointer_value();

            if let PrimitiveType::List(ListItemType::Boolean, list_size) = ty {
                let i8_type = ctx.i8_type();
                let val = val.into_array_value();
                for idx in 0..list_size {
                    let bool_val = b2
                        .build_extract_value(val, idx as u32, "bool_lane")
                        .unwrap()
                        .into_int_value();
                    let byte_val = b2
                        .build_int_z_extend(bool_val, i8_type, "bool_byte")
                        .unwrap();
                    let lane_ptr = increment_pointer!(ctx, b2, curr_ptr, idx);
                    b2.build_store(lane_ptr, byte_val).unwrap();
                }
            } else {
                b2.build_store(curr_ptr, val).unwrap();
            }
            let new_ptr = increment_pointer!(ctx, b2, curr_ptr, width);
            b2.build_store(alloc_ptr, new_ptr).unwrap();
            b2.build_return(None).unwrap();
            func
        };

        FixedSizeListWriter { ingest_func }
    }

    fn llvm_ingest(&self, _ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>) {
        build
            .build_call(self.ingest_func, &[val.into()], "ingest")
            .unwrap();
    }

    fn llvm_flush(&self, _ctx: &'a Context, _build: &Builder<'a>) {
        // no-op
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int32Type};
    use inkwell::{context::Context, AddressSpace, OptimizationLevel};

    use crate::{
        compiled_writers::{
            fixed_size_list_writer::FixedSizeListWriter, LeafWriter, LeafWriterAllocation,
        },
        declare_blocks, ListItemType, PrimitiveType,
    };

    #[test]
    fn test_fsl_writer_i32() {
        let ctx = Context::create();

        let i32_type = ctx.i32_type();

        let llvm_mod = ctx.create_module("test_fsl_array_writer");
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
        let writer = FixedSizeListWriter::llvm_init(
            &ctx,
            &llvm_mod,
            &build,
            PrimitiveType::List(ListItemType::I32, 4),
            dest,
        );

        for i in 0..10 {
            let to_write = i32_type.vec_type(4).const_zero();
            let to_write = build
                .build_insert_element(
                    to_write,
                    i32_type.const_int(i, true),
                    i32_type.const_int(0, false),
                    "insert0",
                )
                .unwrap();
            let to_write = build
                .build_insert_element(
                    to_write,
                    i32_type.const_int(i + 1, true),
                    i32_type.const_int(1, false),
                    "insert1",
                )
                .unwrap();
            let to_write = build
                .build_insert_element(
                    to_write,
                    i32_type.const_int(i + 2, true),
                    i32_type.const_int(2, false),
                    "insert2",
                )
                .unwrap();
            let to_write = build
                .build_insert_element(
                    to_write,
                    i32_type.const_int(i + 3, true),
                    i32_type.const_int(3, false),
                    "insert3",
                )
                .unwrap();
            writer.llvm_ingest(&ctx, &build, to_write.into());
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

        let mut data = FixedSizeListWriter::allocate(10, PrimitiveType::List(ListItemType::I32, 4));
        unsafe {
            f.call(data.get_ptr());
        }
        data.reserve_for_additional(10);
        unsafe {
            f.call(data.get_ptr());
        }
        let data = data.to_array(20, None);

        for i in 0..20 {
            let el = data.value(i);
            let i = (i % 10) as i32;
            assert_eq!(
                el.as_primitive::<Int32Type>().values(),
                &[i, i + 1, i + 2, i + 3]
            );
        }
    }
}
