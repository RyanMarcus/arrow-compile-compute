use std::{ffi::c_void, sync::Arc};

use crate::{declare_blocks, increment_pointer, ListItemType, PrimitiveType};
use arrow_array::{builder::BinaryBuilder, make_array, ArrayRef, FixedSizeListArray};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::Field;
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    values::{BasicValue, BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace,
};

use super::{BooleanAllocation, BooleanWriter, LeafWriter, LeafWriterAllocation};

/// Writer for fixed-size lists of primitives
pub enum FixedSizeListWriter<'a> {
    Raw {
        ingest_func: FunctionValue<'a>,
    },
    Boolean {
        writer: BooleanWriter<'a>,
        list_size: usize,
    },
}

#[derive(Debug)]
enum FixedSizeListWriterAllocStorage {
    Raw {
        out_ptr: *mut c_void,
        out: Vec<u128>,
        element_pt: PrimitiveType,
    },
    Boolean(BooleanAllocation),
}

#[derive(Debug)]
pub struct FixedSizeListWriterAlloc {
    storage: FixedSizeListWriterAllocStorage,
    list_size: usize,
}

impl LeafWriterAllocation for FixedSizeListWriterAlloc {
    type Output = FixedSizeListArray;

    fn get_ptr(&mut self) -> *mut c_void {
        match &mut self.storage {
            FixedSizeListWriterAllocStorage::Raw { out_ptr, .. } => {
                out_ptr as *mut *mut c_void as *mut c_void
            }
            FixedSizeListWriterAllocStorage::Boolean(alloc) => alloc.get_ptr(),
        }
    }

    fn reserve_for_additional(&mut self, count: usize) {
        match &mut self.storage {
            FixedSizeListWriterAllocStorage::Raw {
                out_ptr,
                out,
                element_pt,
            } => unsafe {
                let bytes_written = out_ptr.offset_from_unsigned(out.as_ptr() as *mut c_void);
                let items_to_preserve = bytes_written.div_ceil(16);
                out.set_len(items_to_preserve);
                out.resize(
                    items_to_preserve + (count * element_pt.width() * self.list_size).div_ceil(16),
                    0,
                );
                let new_base = out.as_mut_ptr() as *mut c_void;
                *out_ptr = new_base.byte_add(bytes_written);
            },
            FixedSizeListWriterAllocStorage::Boolean(alloc) => {
                alloc.reserve_for_additional(count * self.list_size)
            }
        }
    }

    fn rewind_one(&mut self) {
        match &mut self.storage {
            FixedSizeListWriterAllocStorage::Raw {
                out_ptr,
                out,
                element_pt,
            } => unsafe {
                let width = element_pt.width() * self.list_size;
                let base = out.as_mut_ptr() as *mut c_void;
                let bytes_written = out_ptr.offset_from_unsigned(base);
                *out_ptr = base.byte_add(bytes_written - width);
            },
            FixedSizeListWriterAllocStorage::Boolean(alloc) => {
                for _ in 0..self.list_size {
                    alloc.rewind_one();
                }
            }
        }
    }

    fn to_array(self, len: usize, nulls: Option<NullBuffer>) -> Self::Output {
        let list_size = self.list_size;
        let values_len = len * list_size;
        let flat_data = match self.storage {
            FixedSizeListWriterAllocStorage::Boolean(alloc) => {
                Arc::new(alloc.to_array(values_len, None)) as ArrayRef
            }
            FixedSizeListWriterAllocStorage::Raw {
                out, element_pt, ..
            } => match element_pt {
                PrimitiveType::P64x2 => {
                    let ptrs = unsafe {
                        std::slice::from_raw_parts(out.as_ptr() as *const u128, values_len)
                    };
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
            },
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
        match &self.storage {
            FixedSizeListWriterAllocStorage::Raw {
                out_ptr,
                out,
                element_pt,
            } => {
                let offset = unsafe { out_ptr.byte_offset_from(out.as_ptr()) };
                let offset = usize::try_from(offset).unwrap();
                let width = element_pt.width() * self.list_size;
                assert_eq!(offset % width, 0);
                offset / width
            }
            FixedSizeListWriterAllocStorage::Boolean(alloc) => {
                assert_eq!(alloc.len() % self.list_size, 0);
                alloc.len() / self.list_size
            }
        }
    }
}

impl<'a> LeafWriter<'a> for FixedSizeListWriter<'a> {
    type Allocation = FixedSizeListWriterAlloc;
    fn allocate(expected_count: usize, ty: PrimitiveType) -> Self::Allocation {
        match ty {
            PrimitiveType::List(el_type, list_size) => {
                let storage = if matches!(el_type, ListItemType::Boolean) {
                    FixedSizeListWriterAllocStorage::Boolean(BooleanWriter::allocate(
                        expected_count * list_size,
                        PrimitiveType::U8,
                    ))
                } else {
                    let mut data = vec![0_u128; (ty.width() * expected_count).div_ceil(16)];
                    assert!(data.capacity() > 0 || expected_count == 0);
                    let data_ptr = data.as_mut_ptr() as *mut c_void;
                    FixedSizeListWriterAllocStorage::Raw {
                        out_ptr: data_ptr,
                        out: data,
                        element_pt: el_type.physical_primitive_type(),
                    }
                };
                FixedSizeListWriterAlloc { storage, list_size }
            }
            _ => panic!("Unsupported type {} for FixedSizeListWriter", ty),
        }
    }

    fn llvm_init(
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        ty: PrimitiveType,
        _alloc_ptr: PointerValue<'a>,
    ) -> Self {
        if let PrimitiveType::List(ListItemType::Boolean, list_size) = ty {
            return FixedSizeListWriter::Boolean {
                writer: BooleanWriter::llvm_init(
                    ctx,
                    llvm_mod,
                    build,
                    PrimitiveType::U8,
                    _alloc_ptr,
                ),
                list_size,
            };
        }

        let ptr_type = ctx.ptr_type(AddressSpace::default());

        let width = ty.width();
        let func_name = format!("ingest_fsl_{ty}")
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>();

        let ingest_func = {
            let b2 = ctx.create_builder();
            let fn_type = ctx
                .void_type()
                .fn_type(&[ptr_type.into(), ty.llvm_type(ctx).into()], false);
            let func = llvm_mod.add_function(&func_name, fn_type, Some(Linkage::Private));
            declare_blocks!(ctx, func, entry);
            b2.position_at_end(entry);
            let alloc_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
            let val = func.get_nth_param(1).unwrap();
            let curr_ptr = b2
                .build_load(ptr_type, alloc_ptr, "curr_ptr")
                .unwrap()
                .into_pointer_value();

            b2.build_store(curr_ptr, val).unwrap();
            let new_ptr = increment_pointer!(ctx, b2, curr_ptr, width);
            b2.build_store(alloc_ptr, new_ptr).unwrap();
            b2.build_return(None).unwrap();
            func
        };

        FixedSizeListWriter::Raw { ingest_func }
    }
}

impl<'a> FixedSizeListWriter<'a> {
    pub(super) fn emit_ingest(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        alloc_ptr: PointerValue<'a>,
        val: BasicValueEnum<'a>,
    ) {
        match self {
            FixedSizeListWriter::Raw { ingest_func, .. } => {
                build
                    .build_call(*ingest_func, &[alloc_ptr.into(), val.into()], "ingest")
                    .unwrap();
            }
            FixedSizeListWriter::Boolean { writer, list_size } => {
                let val = val.into_array_value();
                let full_chunks = list_size / 64;
                for chunk_idx in 0..full_chunks {
                    let chunk = build
                        .build_extract_value(val, chunk_idx as u32, "bool_chunk")
                        .unwrap();
                    writer.emit_ingest_64_bools(ctx, build, alloc_ptr, chunk);
                }

                let tail = list_size % 64;
                if tail > 0 {
                    let chunk = build
                        .build_extract_value(val, full_chunks as u32, "bool_tail_chunk")
                        .unwrap()
                        .into_int_value();
                    for idx in 0..tail {
                        let shifted = build
                            .build_right_shift(
                                chunk,
                                ctx.i64_type().const_int(idx as u64, false),
                                false,
                                "bool_tail_shifted",
                            )
                            .unwrap();
                        let bool_val = build
                            .build_int_truncate(shifted, ctx.bool_type(), "bool_tail_bit")
                            .unwrap()
                            .as_basic_value_enum();
                        writer.emit_ingest(ctx, build, alloc_ptr, bool_val);
                    }
                }
            }
        }
    }

    pub(super) fn emit_flush(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        alloc_ptr: PointerValue<'a>,
    ) {
        match self {
            FixedSizeListWriter::Raw { .. } => {}
            FixedSizeListWriter::Boolean { writer, .. } => writer.emit_flush(ctx, build, alloc_ptr),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int32Type};
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};

    use crate::{
        compiled_writers::{
            fixed_size_list_writer::FixedSizeListWriter, LeafWriter, LeafWriterAllocation, Writer,
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
        let writer = Writer::FixedSizeList(FixedSizeListWriter::llvm_init(
            &ctx,
            &llvm_mod,
            &build,
            PrimitiveType::List(ListItemType::I32, 4),
            dest,
        ))
        .bind(dest);

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

    #[test]
    fn test_fsl_writer_boolean_uses_packed_values() {
        let ctx = Context::create();

        let list_type = PrimitiveType::List(ListItemType::Boolean, 3)
            .llvm_type(&ctx)
            .into_array_type();

        let llvm_mod = ctx.create_module("test_fsl_bool_writer");
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
        let writer = Writer::FixedSizeList(FixedSizeListWriter::llvm_init(
            &ctx,
            &llvm_mod,
            &build,
            PrimitiveType::List(ListItemType::Boolean, 3),
            dest,
        ))
        .bind(dest);

        let rows = [
            [true, false, true],
            [false, false, true],
            [true, true, false],
            [false, true, false],
            [true, false, false],
        ];

        for row in rows {
            let mut to_write = list_type.const_zero();
            let packed = row
                .into_iter()
                .enumerate()
                .fold(0_u64, |acc, (idx, value)| acc | ((value as u64) << idx));
            to_write = build
                .build_insert_value(
                    to_write,
                    ctx.i64_type().const_int(packed, false),
                    0,
                    "insert_bool_chunk",
                )
                .unwrap()
                .into_array_value();
            writer.llvm_ingest(&ctx, &build, to_write.as_basic_value_enum());
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

        let mut data =
            FixedSizeListWriter::allocate(5, PrimitiveType::List(ListItemType::Boolean, 3));
        unsafe {
            f.call(data.get_ptr());
        }
        data.reserve_for_additional(5);
        unsafe {
            f.call(data.get_ptr());
        }
        let data = data.to_array(10, None);

        assert_eq!(data.values().as_boolean().values().len(), 30);
        for i in 0..10 {
            let el = data.value(i);
            let bools = el.as_boolean();
            let expected = rows[i % rows.len()];
            assert_eq!(
                bools.iter().map(|v| v.unwrap()).collect::<Vec<_>>(),
                expected
            );
        }
    }
}
