use std::sync::Arc;

use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, BooleanArray, PrimitiveArray};
use arrow_buffer::ToByteSlice;
use arrow_schema::DataType;
use half::f16;
use inkwell::{
    builder::Builder,
    context::Context,
    values::{BasicValue, BasicValueEnum, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::{increment_pointer, mark_load_invariant, ListItemType, PrimitiveType};

use super::IteratorHolder;

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct ScalarPrimitiveIterator {
    /// the scalar value, packed to the right of a u64
    val: u64,

    /// the width, in bytes, of the scalar
    pub width: u8,

    pub ptype: PrimitiveType,
}

impl ScalarPrimitiveIterator {
    pub fn new(val: u64, width: u8, ptype: PrimitiveType) -> Self {
        assert!(
            !matches!(ptype, PrimitiveType::P64x2 | PrimitiveType::List(_, _)),
            "invalid primitive scalar type"
        );
        Self { val, width, ptype }
    }

    pub fn llvm_val_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        increment_pointer!(ctx, builder, ptr, ScalarPrimitiveIterator::OFFSET_VAL)
    }
}

impl From<i8> for IteratorHolder {
    fn from(val: i8) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val as u8 as u64,
            1,
            PrimitiveType::I8,
        )))
    }
}

impl From<i16> for IteratorHolder {
    fn from(val: i16) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val as u16 as u64,
            2,
            PrimitiveType::I16,
        )))
    }
}

impl From<i32> for IteratorHolder {
    fn from(val: i32) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val as u32 as u64,
            4,
            PrimitiveType::I32,
        )))
    }
}

impl From<i64> for IteratorHolder {
    fn from(val: i64) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val as u64,
            8,
            PrimitiveType::I64,
        )))
    }
}

impl From<u8> for IteratorHolder {
    fn from(val: u8) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val as u64,
            1,
            PrimitiveType::U8,
        )))
    }
}

impl From<u16> for IteratorHolder {
    fn from(val: u16) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val as u64,
            2,
            PrimitiveType::U16,
        )))
    }
}

impl From<u32> for IteratorHolder {
    fn from(val: u32) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val as u64,
            4,
            PrimitiveType::U32,
        )))
    }
}

impl From<u64> for IteratorHolder {
    fn from(val: u64) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val,
            8,
            PrimitiveType::U64,
        )))
    }
}

impl From<f16> for IteratorHolder {
    fn from(val: f16) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            (val.to_f64()).to_bits(),
            2,
            PrimitiveType::F16,
        )))
    }
}

impl From<f32> for IteratorHolder {
    fn from(val: f32) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val.to_bits() as u64,
            4,
            PrimitiveType::F32,
        )))
    }
}

impl From<f64> for IteratorHolder {
    fn from(val: f64) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val.to_bits(),
            8,
            PrimitiveType::F64,
        )))
    }
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct ScalarBooleanIterator {
    val: u8,
}

impl ScalarBooleanIterator {
    pub fn new(val: bool) -> Self {
        Self { val: val as u8 }
    }

    pub fn llvm_val_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        increment_pointer!(ctx, builder, ptr, ScalarPrimitiveIterator::OFFSET_VAL)
    }
}

impl From<bool> for IteratorHolder {
    fn from(val: bool) -> Self {
        IteratorHolder::ScalarBoolean(Box::new(ScalarBooleanIterator::new(val)))
    }
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct ScalarStringIterator {
    ptr1: *const u8,
    ptr2: *const u8,
    val: Box<str>,
    pub(super) data_type: DataType,
}

impl ScalarStringIterator {
    pub(super) fn new(val: Box<str>, data_type: DataType) -> Box<Self> {
        let p1 = val.as_ptr();
        let p2 = p1.wrapping_add(val.len());
        Box::new(Self {
            ptr1: p1,
            ptr2: p2,
            val,
            data_type,
        })
    }

    pub fn llvm_val_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> (PointerValue<'a>, PointerValue<'a>) {
        let ptr1 = increment_pointer!(ctx, builder, ptr, ScalarStringIterator::OFFSET_PTR1);
        let ptr2 = increment_pointer!(ctx, builder, ptr, ScalarStringIterator::OFFSET_PTR2);
        (ptr1, ptr2)
    }
}

impl From<Box<str>> for IteratorHolder {
    fn from(val: Box<str>) -> Self {
        IteratorHolder::ScalarString(ScalarStringIterator::new(val, DataType::Utf8))
    }
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct ScalarBinaryIterator {
    ptr1: *const u8,
    ptr2: *const u8,
    val: Box<[u8]>,
    pub(super) data_type: DataType,
}

impl ScalarBinaryIterator {
    pub(super) fn new(val: Box<[u8]>, data_type: DataType) -> Box<Self> {
        let p1 = val.as_ptr();
        let p2 = p1.wrapping_add(val.len());
        Box::new(Self {
            ptr1: p1,
            ptr2: p2,
            val,
            data_type,
        })
    }

    pub fn llvm_val_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> (PointerValue<'a>, PointerValue<'a>) {
        let ptr1 = increment_pointer!(ctx, builder, ptr, ScalarBinaryIterator::OFFSET_PTR1);
        let ptr2 = increment_pointer!(ctx, builder, ptr, ScalarBinaryIterator::OFFSET_PTR2);
        (ptr1, ptr2)
    }
}

impl From<Box<[u8]>> for IteratorHolder {
    fn from(val: Box<[u8]>) -> Self {
        IteratorHolder::ScalarBinary(ScalarBinaryIterator::new(val, DataType::Binary))
    }
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct ScalarVectorIterator {
    ptype: ListItemType,
    l: usize,
    val: Box<[u8]>,
    #[allow(dead_code)]
    owner: Option<ArrayRef>,
    pub(super) data_type: DataType,
}

impl ScalarVectorIterator {
    pub fn from_primitive<K: ArrowPrimitiveType>(
        arr: &PrimitiveArray<K>,
        data_type: DataType,
    ) -> Box<Self> {
        let ptype = PrimitiveType::for_arrow_type(arr.data_type())
            .try_into()
            .unwrap();
        let l = arr.len();
        let val = arr
            .values()
            .inner()
            .to_byte_slice()
            .to_vec()
            .into_boxed_slice();
        Box::new(ScalarVectorIterator {
            ptype,
            l,
            val,
            owner: None,
            data_type,
        })
    }

    pub fn from_boolean(arr: &BooleanArray, data_type: DataType) -> Box<Self> {
        let l = arr.len();
        let val = arr.values().sliced().as_slice().to_vec().into_boxed_slice();
        Box::new(ScalarVectorIterator {
            ptype: ListItemType::Boolean,
            l,
            val,
            owner: Some(Arc::new(arr.clone())),
            data_type,
        })
    }

    pub fn from_pointer_pairs(
        ptype: ListItemType,
        ptrs: Vec<u128>,
        owner: ArrayRef,
        data_type: DataType,
    ) -> Box<Self> {
        let l = ptrs.len();
        let val = bytemuck::cast_slice::<u128, u8>(&ptrs)
            .to_vec()
            .into_boxed_slice();
        Box::new(ScalarVectorIterator {
            ptype,
            l,
            val,
            owner: Some(owner),
            data_type,
        })
    }

    pub fn llvm_val<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> BasicValueEnum<'a> {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let val_ptr_ptr = increment_pointer!(ctx, builder, ptr, ScalarBinaryIterator::OFFSET_VAL);
        let val_ptr = builder
            .build_load(ptr_type, val_ptr_ptr, "val_ptr")
            .unwrap()
            .into_pointer_value();
        mark_load_invariant!(ctx, val_ptr);

        match self.ptype {
            ListItemType::Boolean => {
                let mut out = self.ptype().llvm_type(ctx).into_vector_type().const_zero();
                for idx in 0..self.l {
                    let byte_ptr = increment_pointer!(ctx, builder, val_ptr, idx / 8);
                    let byte = builder
                        .build_load(ctx.i8_type(), byte_ptr, "bool_byte")
                        .unwrap()
                        .into_int_value();
                    let shifted = builder
                        .build_right_shift(
                            byte,
                            ctx.i8_type().const_int((idx % 8) as u64, false),
                            false,
                            "bool_shifted",
                        )
                        .unwrap();
                    let value = builder
                        .build_int_truncate(shifted, ctx.bool_type(), "bool_value")
                        .unwrap();
                    out = builder
                        .build_insert_element(
                            out,
                            value,
                            ctx.i64_type().const_int(idx as u64, false),
                            "bool_insert",
                        )
                        .unwrap();
                }
                out.as_basic_value_enum()
            }
            ListItemType::P64x2 => {
                let mut out = self.ptype().llvm_type(ctx).into_array_type().const_zero();
                let lane_ty = PrimitiveType::P64x2.llvm_type(ctx);
                for idx in 0..self.l {
                    let lane_ptr = increment_pointer!(ctx, builder, val_ptr, 16 * idx);
                    let lane = builder.build_load(lane_ty, lane_ptr, "str_lane").unwrap();
                    out = builder
                        .build_insert_value(out, lane, idx as u32, "str_insert")
                        .unwrap()
                        .into_array_value();
                }
                out.as_basic_value_enum()
            }
            _ => {
                let vec_type = PrimitiveType::from(self.ptype)
                    .llvm_vec_type(ctx, self.l as u32)
                    .unwrap();
                let val = builder.build_load(vec_type, val_ptr, "val").unwrap();
                val.as_instruction_value()
                    .unwrap()
                    .set_alignment(1)
                    .unwrap();
                mark_load_invariant!(ctx, val);
                val.as_basic_value_enum()
            }
        }
    }

    pub fn ptype(&self) -> PrimitiveType {
        PrimitiveType::List(self.ptype, self.l)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{BooleanArray, Float32Array, Int32Array, LargeStringArray, Scalar};
    use arrow_schema::DataType;
    use inkwell::{context::Context, OptimizationLevel};

    use crate::compiled_iter::{datum_to_iter, generate_next, generate_next_block, IteratorHolder};

    #[test]
    fn data_type_preserves_scalar_string_offset_width() {
        let scalar = Scalar::new(LargeStringArray::from(vec!["hello"]));
        let iter = datum_to_iter(&scalar).unwrap();

        assert_eq!(iter.data_type(), DataType::LargeUtf8);
    }

    #[test]
    fn scalar_boolean_data_type_is_boolean() {
        assert_eq!(IteratorHolder::from(true).data_type(), DataType::Boolean);
    }

    #[test]
    fn scalar_primitive_data_type_comes_from_primitive_type() {
        assert_eq!(IteratorHolder::from(42_i32).data_type(), DataType::Int32);
        assert_eq!(
            IteratorHolder::from(42.0_f64).data_type(),
            DataType::Float64
        );
    }

    #[test]
    fn test_scalar_int() {
        let s = Int32Array::from(vec![42]);
        let s = Scalar::new(s);
        let mut iter = datum_to_iter(&s).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_scalar_prim");
        let func_block =
            generate_next_block::<4>(&ctx, &module, "iter_prim_test", &DataType::Int32, &iter)
                .unwrap();
        let func_next =
            generate_next(&ctx, &module, "iter_prim_test", &DataType::Int32, &iter).unwrap();

        let fname_block = func_block.get_name().to_str().unwrap();
        let fname_next = func_next.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname_block)
                .unwrap()
        };
        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname_next)
                .unwrap()
        };

        let mut buf = [0_i32; 4];
        unsafe {
            assert_eq!(
                next_block_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()),
                true
            );
            assert_eq!(buf, [42, 42, 42, 42]);
            assert_eq!(
                next_block_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()),
                true
            );
            assert_eq!(buf, [42, 42, 42, 42]);
        };

        let mut buf = [0_i32; 1];
        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [42]);
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [42]);
        };
    }

    #[test]
    fn test_scalar_float() {
        let f = 42.31894;
        let s = Float32Array::from(vec![f]);
        let s = Scalar::new(s);
        let mut iter = datum_to_iter(&s).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_scalar_prim");
        let func_block = generate_next_block::<4>(
            &ctx,
            &module,
            "iter_block_prim_test",
            &DataType::Float32,
            &iter,
        )
        .unwrap();
        let func_next =
            generate_next(&ctx, &module, "iter_prim_test", &DataType::Float32, &iter).unwrap();

        let fname_block = func_block.get_name().to_str().unwrap();
        let fname_next = func_next.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut f32) -> bool>(fname_block)
                .unwrap()
        };
        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut f32) -> bool>(fname_next)
                .unwrap()
        };

        let mut buf: f32 = 0.0;
        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert_eq!(buf, f);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert_eq!(buf, f);
        };

        #[repr(align(16))]
        struct Buf([f32; 4]);
        let mut buf = Buf([0_f32; 4]);
        unsafe {
            assert_eq!(
                next_block_func.call(iter.get_mut_ptr(), buf.0.as_mut_ptr()),
                true
            );
            assert_eq!(buf.0, [f, f, f, f]);
            assert_eq!(
                next_block_func.call(iter.get_mut_ptr(), buf.0.as_mut_ptr()),
                true
            );
            assert_eq!(buf.0, [f, f, f, f]);
        };
    }

    #[test]
    fn test_scalar_bool() {
        let s = BooleanArray::new_scalar(true);
        let mut iter = datum_to_iter(&s).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_scalar_bool");
        let func_next =
            generate_next(&ctx, &module, "iter_prim_test", &DataType::Boolean, &iter).unwrap();

        let fname_next = func_next.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u8) -> bool>(fname_next)
                .unwrap()
        };

        let mut buf = [0_u8; 1];
        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [1]);
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [1]);
        };
    }
}
