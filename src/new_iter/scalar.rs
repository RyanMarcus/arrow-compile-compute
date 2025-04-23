#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct ScalarPrimitiveIterator {
    /// the scalar value, packed to the right of a u64
    val: u64,

    /// the width, in bytes, of the scalar
    pub width: u8,
}

impl ScalarPrimitiveIterator {
    pub fn new(val: u64, width: u8) -> Self {
        Self { val, width }
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
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val as u8 as u64, 1)))
    }
}

impl From<i16> for IteratorHolder {
    fn from(val: i16) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val as u16 as u64,
            2,
        )))
    }
}

impl From<i32> for IteratorHolder {
    fn from(val: i32) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val as u32 as u64,
            4,
        )))
    }
}

impl From<i64> for IteratorHolder {
    fn from(val: i64) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val as u64, 8)))
    }
}

impl From<u8> for IteratorHolder {
    fn from(val: u8) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val as u64, 1)))
    }
}

impl From<u16> for IteratorHolder {
    fn from(val: u16) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val as u64, 2)))
    }
}

impl From<u32> for IteratorHolder {
    fn from(val: u32) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val as u64, 4)))
    }
}

impl From<u64> for IteratorHolder {
    fn from(val: u64) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val, 8)))
    }
}

use half::f16;
use inkwell::{builder::Builder, context::Context, values::PointerValue};
use repr_offset::ReprOffset;

use crate::increment_pointer;

use super::IteratorHolder;
impl From<f16> for IteratorHolder {
    fn from(val: f16) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            (val.to_f64()).to_bits(),
            2,
        )))
    }
}

impl From<f32> for IteratorHolder {
    fn from(val: f32) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val.to_bits() as u64,
            4,
        )))
    }
}

impl From<f64> for IteratorHolder {
    fn from(val: f64) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val.to_bits(), 8)))
    }
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct ScalarStringIterator {
    ptr1: *const u8,
    ptr2: *const u8,
    val: Box<str>,
}

impl ScalarStringIterator {
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
        let p1 = val.as_ptr();
        let p2 = p1.wrapping_add(val.len());
        IteratorHolder::ScalarString(Box::new(ScalarStringIterator {
            ptr1: p1,
            ptr2: p2,
            val,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{Float32Array, Int32Array, Scalar};
    use arrow_schema::DataType;
    use inkwell::{context::Context, OptimizationLevel};

    use crate::new_iter::{datum_to_iter, generate_next, generate_next_block};

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
}
