use std::{ffi::c_void, sync::Arc};

use arrow_array::{
    builder::BinaryBuilder, ArrayRef, FixedSizeListArray, Float16Array, Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, Int8Array, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow_buffer::MutableBuffer;
use arrow_schema::Field;
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    types::BasicType,
    values::{FunctionValue, IntValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::{increment_pointer, mark_load_invariant, ListItemType, PrimitiveType};

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct DSLBuffer {
    pub ty: PrimitiveType,
    pub len: u64,
    pub buf: MutableBuffer,
    pub ptr: *mut c_void,
}

impl DSLBuffer {
    pub fn new(ty: PrimitiveType, len: usize) -> Self {
        let mut buf = MutableBuffer::new(ty.width() * len);
        buf.resize(ty.width() * len, 0);
        let ptr = buf.as_mut_ptr() as *mut c_void;
        Self {
            ty,
            len: len as u64,
            buf,
            ptr,
        }
    }

    pub fn empty_like(other: &Self) -> Self {
        Self::new(other.ty, 0)
    }

    pub fn ensure_capacity(&mut self, capacity: usize) {
        if self.len as usize >= capacity {
            return;
        }
        let capacity_bytes = capacity * self.ty.width();

        self.buf.resize(capacity_bytes, 0);
        self.len = capacity as u64;
        self.ptr = self.buf.as_mut_ptr() as *mut c_void;
    }

    pub fn into_array(mut self) -> ArrayRef {
        self.buf.truncate(self.ty.width() * self.len as usize);
        match self.ty {
            PrimitiveType::I8 => Arc::new(Int8Array::new(self.buf.into(), None)),
            PrimitiveType::I16 => Arc::new(Int16Array::new(self.buf.into(), None)),
            PrimitiveType::I32 => Arc::new(Int32Array::new(self.buf.into(), None)),
            PrimitiveType::I64 => Arc::new(Int64Array::new(self.buf.into(), None)),
            PrimitiveType::U8 => Arc::new(UInt8Array::new(self.buf.into(), None)),
            PrimitiveType::U16 => Arc::new(UInt16Array::new(self.buf.into(), None)),
            PrimitiveType::U32 => Arc::new(UInt32Array::new(self.buf.into(), None)),
            PrimitiveType::U64 => Arc::new(UInt64Array::new(self.buf.into(), None)),
            PrimitiveType::F16 => Arc::new(Float16Array::new(self.buf.into(), None)),
            PrimitiveType::F32 => Arc::new(Float32Array::new(self.buf.into(), None)),
            PrimitiveType::F64 => Arc::new(Float64Array::new(self.buf.into(), None)),
            PrimitiveType::P64x2 => {
                let mut b = BinaryBuilder::new();
                let slice = bytemuck::cast_slice::<u8, u128>(self.buf.as_slice());

                for &v in slice {
                    let start_ptr = (v as u64) as *const u8;
                    let end_ptr = ((v >> 64) as u64) as *const u8;
                    assert!(end_ptr >= start_ptr);

                    let value = unsafe {
                        let len = end_ptr.offset_from_unsigned(start_ptr) as usize;
                        std::slice::from_raw_parts(start_ptr, len)
                    };
                    b.append_value(value);
                }

                Arc::new(b.finish())
            }
            PrimitiveType::List(ty, size) => {
                let values: ArrayRef = match ty {
                    ListItemType::I8 => Arc::new(Int8Array::new(self.buf.into(), None)),
                    ListItemType::I16 => Arc::new(Int16Array::new(self.buf.into(), None)),
                    ListItemType::I32 => Arc::new(Int32Array::new(self.buf.into(), None)),
                    ListItemType::I64 => Arc::new(Int64Array::new(self.buf.into(), None)),
                    ListItemType::U8 => Arc::new(UInt8Array::new(self.buf.into(), None)),
                    ListItemType::U16 => Arc::new(UInt16Array::new(self.buf.into(), None)),
                    ListItemType::U32 => Arc::new(UInt32Array::new(self.buf.into(), None)),
                    ListItemType::U64 => Arc::new(UInt64Array::new(self.buf.into(), None)),
                    ListItemType::F16 => Arc::new(Float16Array::new(self.buf.into(), None)),
                    ListItemType::F32 => Arc::new(Float32Array::new(self.buf.into(), None)),
                    ListItemType::F64 => Arc::new(Float64Array::new(self.buf.into(), None)),
                    ListItemType::P64x2 => todo!(),
                };
                Arc::new(FixedSizeListArray::new(
                    Field::new_list_field(PrimitiveType::from(ty).as_arrow_type(), false).into(),
                    size as i32,
                    values,
                    None,
                ))
            }
        }
    }

    pub fn as_ptr(&mut self) -> *mut c_void {
        self as *mut Self as *mut c_void
    }

    pub fn buffer_len<'a>(ctx: &'a Context, b: &'a Builder, ptr: PointerValue<'a>) -> IntValue<'a> {
        let len = b
            .build_load(
                ctx.i64_type(),
                increment_pointer!(ctx, b, ptr, DSLBuffer::OFFSET_LEN),
                "len",
            )
            .unwrap()
            .into_int_value();
        mark_load_invariant!(ctx, len);

        len
    }

    pub fn generate_buffer_accessor<'a>(
        ctx: &'a Context,
        module: &Module<'a>,
        pt: PrimitiveType,
    ) -> FunctionValue<'a> {
        let func_name = format!("buffer_accessor_{}", pt);
        if let Some(func) = module.get_function(&func_name) {
            return func;
        }

        let ret_ty = pt.llvm_type(ctx);
        let ptr_ty = ctx.ptr_type(AddressSpace::default());
        let i64_ty = ctx.i64_type();
        let func_ty = ret_ty.fn_type(&[ptr_ty.into(), i64_ty.into()], false);

        let func = module.add_function(&func_name, func_ty, Some(Linkage::Private));
        let ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let idx = func.get_nth_param(1).unwrap().into_int_value();

        let entry = ctx.append_basic_block(func, "entry");
        let b = ctx.create_builder();
        b.position_at_end(entry);
        let buf_ptr = b
            .build_load(
                ptr_ty,
                increment_pointer!(ctx, b, ptr, DSLBuffer::OFFSET_PTR),
                "buf_ptr",
            )
            .unwrap()
            .into_pointer_value();
        mark_load_invariant!(ctx, buf_ptr);

        let ptr_to_elem = increment_pointer!(ctx, b, buf_ptr, pt.width(), idx);
        let el = b.build_load(ret_ty, ptr_to_elem, "load_el").unwrap();
        b.build_return(Some(&el)).unwrap();

        func
    }

    pub fn generate_buffer_writer<'a>(
        ctx: &'a Context,
        module: &Module<'a>,
        pt: PrimitiveType,
    ) -> FunctionValue<'a> {
        let func_name = format!("buffer_writer_{}", pt);
        if let Some(func) = module.get_function(&func_name) {
            return func;
        }

        let write_ty = pt.llvm_type(ctx);
        let ptr_ty = ctx.ptr_type(AddressSpace::default());
        let i64_ty = ctx.i64_type();
        let func_ty = ctx
            .void_type()
            .fn_type(&[ptr_ty.into(), i64_ty.into(), write_ty.into()], false);

        let func = module.add_function(&func_name, func_ty, Some(Linkage::Private));
        let ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let idx = func.get_nth_param(1).unwrap().into_int_value();
        let value = func.get_nth_param(2).unwrap();

        let entry = ctx.append_basic_block(func, "entry");
        let b = ctx.create_builder();

        b.position_at_end(entry);
        let buf_ptr = b
            .build_load(
                ptr_ty,
                increment_pointer!(ctx, b, ptr, DSLBuffer::OFFSET_PTR),
                "buf_ptr",
            )
            .unwrap()
            .into_pointer_value();
        mark_load_invariant!(ctx, buf_ptr);

        let ptr_to_elem = increment_pointer!(ctx, b, buf_ptr, pt.width(), idx);
        b.build_store(ptr_to_elem, value).unwrap();
        b.build_return(None).unwrap();

        func
    }
}
