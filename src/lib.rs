//! Compiled compute functions for the [arrow-rs](https://github.com/apache/arrow-rs).
//!
//! There are two interfaces for using the JIT-compiled compute functions.
//!
//! * [`cmp`](cmp/index.html), a simple, `arrow`-like, entrypoint: which provides
//!   functions like `cmp::eq`. This is probably what you want.
//!
//! * [`CodeGen`](struct.CodeGen.html), which allows for more control over the
//!   underlying LLVM IR and context.
//!
//! This crate supports several operations that the vectorized/interpreted
//! `arrow-rs` packages do not support. See the [package
//! readme](https://github.com/RyanMarcus/arrow-compile-compute) for more
//! information.

use std::{ffi::c_void, sync::Arc};

use arrow_array::{
    cast::AsArray,
    types::{
        Date32Type, Date64Type, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type,
        Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrayRef, BooleanArray, PrimitiveArray, RunArray,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType, Field};
use half::f16;
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    module::Module,
    passes::PassBuilderOptions,
    targets::{CodeModel, RelocMode, Target, TargetMachine},
    types::{BasicTypeEnum, VectorType},
    values::{BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue, VectorValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};

mod aggregate;
mod arrow_interface;
mod compute_funcs;
mod dict;
mod primitive;
mod runend;
mod scalar;

pub use arrow_interface::cast;
pub use arrow_interface::cmp;
pub use arrow_interface::compute;

/// Utility function to create the appropiate `DataType` for a dictionary array
pub fn dictionary_data_type(key_type: DataType, val_type: DataType) -> DataType {
    DataType::Dictionary(Box::new(key_type.clone()), Box::new(val_type.clone()))
}

/// Utility function to create the appropiate `DataType` for a run-end encoded
/// array
pub fn run_end_data_type(run_type: &DataType, val_type: &DataType) -> DataType {
    let f1 = Arc::new(Field::new("run_ends", run_type.clone(), false));
    let f2 = Arc::new(Field::new("values", val_type.clone(), true));
    DataType::RunEndEncoded(f1, f2)
}

fn buffer_to_primitive(b: Buffer, nulls: Option<NullBuffer>, tar_dt: &DataType) -> Box<dyn Array> {
    let len = b.len() / PrimitiveType::for_arrow_type(tar_dt).width();
    match tar_dt {
        DataType::Int8 => Box::new(PrimitiveArray::<Int8Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::Int16 => Box::new(PrimitiveArray::<Int16Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::Int32 => Box::new(PrimitiveArray::<Int32Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::Int64 => Box::new(PrimitiveArray::<Int64Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::UInt8 => Box::new(PrimitiveArray::<UInt8Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::UInt16 => Box::new(PrimitiveArray::<UInt16Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::UInt32 => Box::new(PrimitiveArray::<UInt32Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::UInt64 => Box::new(PrimitiveArray::<UInt64Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::Float16 => Box::new(PrimitiveArray::<Float16Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::Float32 => Box::new(PrimitiveArray::<Float32Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::Float64 => Box::new(PrimitiveArray::<Float64Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::Date32 => Box::new(PrimitiveArray::<Date32Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        DataType::Date64 => Box::new(PrimitiveArray::<Date64Type>::new(
            ScalarBuffer::new(b, 0, len),
            nulls,
        )),
        _ => unreachable!("cannot cast buffer to {}", tar_dt),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimitiveType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
}

impl PrimitiveType {
    fn width(&self) -> usize {
        match self {
            PrimitiveType::I8 | PrimitiveType::U8 => 1,
            PrimitiveType::I16 | PrimitiveType::U16 => 2,
            PrimitiveType::I32 | PrimitiveType::U32 => 4,
            PrimitiveType::I64 | PrimitiveType::U64 => 8,
            PrimitiveType::F16 => 2,
            PrimitiveType::F32 => 4,
            PrimitiveType::F64 => 8,
        }
    }

    fn llvm_type<'a>(&self, ctx: &'a Context) -> BasicTypeEnum<'a> {
        match self {
            PrimitiveType::I8 | PrimitiveType::U8 => ctx.i8_type().into(),
            PrimitiveType::I16 | PrimitiveType::U16 => ctx.i16_type().into(),
            PrimitiveType::I32 | PrimitiveType::U32 => ctx.i32_type().into(),
            PrimitiveType::I64 | PrimitiveType::U64 => ctx.i64_type().into(),
            PrimitiveType::F16 => ctx.f16_type().into(),
            PrimitiveType::F32 => ctx.f32_type().into(),
            PrimitiveType::F64 => ctx.f64_type().into(),
        }
    }

    fn llvm_vec_type<'a>(&self, ctx: &'a Context, size: u32) -> VectorType<'a> {
        match self {
            PrimitiveType::I8 | PrimitiveType::U8 => ctx.i8_type().vec_type(size),
            PrimitiveType::I16 | PrimitiveType::U16 => ctx.i16_type().vec_type(size),
            PrimitiveType::I32 | PrimitiveType::U32 => ctx.i32_type().vec_type(size),
            PrimitiveType::I64 | PrimitiveType::U64 => ctx.i64_type().vec_type(size),
            PrimitiveType::F16 => ctx.f16_type().vec_type(size),
            PrimitiveType::F32 => ctx.f32_type().vec_type(size),
            PrimitiveType::F64 => ctx.f64_type().vec_type(size),
        }
    }

    fn for_arrow_type(dt: &DataType) -> Self {
        match dt {
            DataType::Int8 => PrimitiveType::I8,
            DataType::Int16 => PrimitiveType::I16,
            DataType::Int32 => PrimitiveType::I32,
            DataType::Int64 => PrimitiveType::I64,
            DataType::UInt8 => PrimitiveType::U8,
            DataType::UInt16 => PrimitiveType::U16,
            DataType::UInt32 => PrimitiveType::U32,
            DataType::UInt64 => PrimitiveType::U64,
            DataType::Float16 => PrimitiveType::F16,
            DataType::Float32 => PrimitiveType::F32,
            DataType::Float64 => PrimitiveType::F64,
            DataType::Dictionary(_k, v) => PrimitiveType::for_arrow_type(v),
            DataType::RunEndEncoded(_k, v) => PrimitiveType::for_arrow_type(v.data_type()),
            _ => todo!(),
        }
    }

    fn as_arrow_type(&self) -> DataType {
        match self {
            PrimitiveType::I8 => DataType::Int8,
            PrimitiveType::I16 => DataType::Int16,
            PrimitiveType::I32 => DataType::Int32,
            PrimitiveType::I64 => DataType::Int64,
            PrimitiveType::U8 => DataType::UInt8,
            PrimitiveType::U16 => DataType::UInt16,
            PrimitiveType::U32 => DataType::UInt32,
            PrimitiveType::U64 => DataType::UInt64,
            PrimitiveType::F16 => DataType::Float16,
            PrimitiveType::F32 => DataType::Float32,
            PrimitiveType::F64 => DataType::Float64,
        }
    }

    /// Returns the primitive int type with the given width in bytes.
    ///
    /// For example, calling with `width = 8` will give `I64`.
    fn int_with_width(width: usize) -> PrimitiveType {
        match width {
            8 => PrimitiveType::I64,
            4 => PrimitiveType::I32,
            2 => PrimitiveType::I16,
            1 => PrimitiveType::I8,
            _ => unreachable!("width must be 8, 4, 2, or 1"),
        }
    }

    fn is_signed(&self) -> bool {
        match self {
            PrimitiveType::I8 | PrimitiveType::I16 | PrimitiveType::I32 | PrimitiveType::I64 => {
                true
            }
            PrimitiveType::U8 | PrimitiveType::U16 | PrimitiveType::U32 | PrimitiveType::U64 => {
                false
            }
            PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => true,
        }
    }

    fn is_int(&self) -> bool {
        match self {
            PrimitiveType::I8
            | PrimitiveType::I16
            | PrimitiveType::I32
            | PrimitiveType::I64
            | PrimitiveType::U8
            | PrimitiveType::U16
            | PrimitiveType::U32
            | PrimitiveType::U64 => true,
            PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => false,
        }
    }

    /// Returns the "best" common value to cast both types to in order to
    /// perform a comparison
    fn dominant(lhs_prim: PrimitiveType, rhs_prim: PrimitiveType) -> PrimitiveType {
        if lhs_prim.is_signed() != rhs_prim.is_signed() {
            PrimitiveType::I64
        } else if lhs_prim.width() >= rhs_prim.width() {
            lhs_prim
        } else {
            rhs_prim
        }
    }

    fn zero<'a>(&self, ctx: &'a Context) -> BasicValueEnum<'a> {
        self.llvm_type(ctx).const_zero()
    }

    fn max_value<'a>(&self, ctx: &'a Context) -> BasicValueEnum<'a> {
        match self {
            PrimitiveType::I8 => self
                .llvm_type(&ctx)
                .into_int_type()
                .const_int(i8::MAX as u64, true)
                .as_basic_value_enum(),
            PrimitiveType::I16 => self
                .llvm_type(&ctx)
                .into_int_type()
                .const_int(i16::MAX as u64, true)
                .as_basic_value_enum(),
            PrimitiveType::I32 => self
                .llvm_type(&ctx)
                .into_int_type()
                .const_int(i32::MAX as u64, true)
                .as_basic_value_enum(),
            PrimitiveType::I64 => self
                .llvm_type(&ctx)
                .into_int_type()
                .const_int(i8::MAX as u64, true)
                .as_basic_value_enum(),
            PrimitiveType::U8 | PrimitiveType::U16 | PrimitiveType::U32 | PrimitiveType::U64 => {
                self.llvm_type(&ctx)
                    .into_int_type()
                    .const_all_ones()
                    .as_basic_value_enum()
            }
            PrimitiveType::F16 => self
                .llvm_type(&ctx)
                .into_float_type()
                .const_float(f16::INFINITY.to_f64())
                .as_basic_value_enum(),
            PrimitiveType::F32 => self
                .llvm_type(&ctx)
                .into_float_type()
                .const_float(f32::INFINITY as f64)
                .as_basic_value_enum(),
            PrimitiveType::F64 => self
                .llvm_type(&ctx)
                .into_float_type()
                .const_float(f64::INFINITY)
                .as_basic_value_enum(),
        }
    }

    fn min_value<'a>(&self, ctx: &'a Context) -> BasicValueEnum<'a> {
        match self {
            PrimitiveType::I8 => self
                .llvm_type(&ctx)
                .into_int_type()
                .const_int(i8::MIN as u64, true)
                .as_basic_value_enum(),
            PrimitiveType::I16 => self
                .llvm_type(&ctx)
                .into_int_type()
                .const_int(i16::MIN as u64, true)
                .as_basic_value_enum(),
            PrimitiveType::I32 => self
                .llvm_type(&ctx)
                .into_int_type()
                .const_int(i32::MIN as u64, true)
                .as_basic_value_enum(),
            PrimitiveType::I64 => self
                .llvm_type(&ctx)
                .into_int_type()
                .const_int(i64::MIN as u64, true)
                .as_basic_value_enum(),
            PrimitiveType::U8 | PrimitiveType::U16 | PrimitiveType::U32 | PrimitiveType::U64 => {
                self.llvm_type(&ctx)
                    .into_int_type()
                    .const_zero()
                    .as_basic_value_enum()
            }
            PrimitiveType::F16 => self
                .llvm_type(&ctx)
                .into_float_type()
                .const_float(f16::NEG_INFINITY.to_f64())
                .as_basic_value_enum(),
            PrimitiveType::F32 => self
                .llvm_type(&ctx)
                .into_float_type()
                .const_float(f32::NEG_INFINITY as f64)
                .as_basic_value_enum(),
            PrimitiveType::F64 => self
                .llvm_type(&ctx)
                .into_float_type()
                .const_float(f64::NEG_INFINITY)
                .as_basic_value_enum(),
        }
    }
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
/// Represents different logical predicates that can be used for function
/// compilation. Signedness is automatically determined.
pub enum Predicate {
    Eq,
    Ne,
    Lt,
    Lte,
    Gt,
    Gte,
}

impl Predicate {
    fn as_int_pred(&self, signed: bool) -> IntPredicate {
        match (self, signed) {
            (Predicate::Eq, _) => IntPredicate::EQ,
            (Predicate::Ne, _) => IntPredicate::NE,
            (Predicate::Lt, true) => IntPredicate::SLT,
            (Predicate::Lt, false) => IntPredicate::ULT,
            (Predicate::Lte, true) => IntPredicate::SLE,
            (Predicate::Lte, false) => IntPredicate::ULE,
            (Predicate::Gt, true) => IntPredicate::SGT,
            (Predicate::Gt, false) => IntPredicate::UGT,
            (Predicate::Gte, true) => IntPredicate::SGE,
            (Predicate::Gte, false) => IntPredicate::UGE,
        }
    }
}

#[repr(C)]
struct PtrHolder {
    arr1: *const c_void,
    arr2: *const c_void,
    phys_len: u64,
    start_at: u64,
    remaining: u64,
}

/// Turns the given array into a pointer and some optional data. The pointer is
/// valid for as long as the optional data and the original data are around.
fn arr_to_ptr(arr: &dyn Array) -> (Option<Box<PtrHolder>>, *const c_void) {
    match arr.data_type() {
        DataType::Boolean => (
            None,
            arr.as_boolean().values().values().as_ptr() as *const c_void,
        ),
        DataType::Int8 => (
            None,
            arr.as_primitive::<Int8Type>().values().as_ptr() as *const c_void,
        ),
        DataType::Int16 => (
            None,
            arr.as_primitive::<Int16Type>().values().as_ptr() as *const c_void,
        ),
        DataType::Int32 => (
            None,
            arr.as_primitive::<Int32Type>().values().as_ptr() as *const c_void,
        ),
        DataType::Int64 => (
            None,
            arr.as_primitive::<Int64Type>().values().as_ptr() as *const c_void,
        ),
        DataType::UInt8 => (
            None,
            arr.as_primitive::<UInt8Type>().values().as_ptr() as *const c_void,
        ),
        DataType::UInt16 => (
            None,
            arr.as_primitive::<UInt16Type>().values().as_ptr() as *const c_void,
        ),
        DataType::UInt32 => (
            None,
            arr.as_primitive::<UInt32Type>().values().as_ptr() as *const c_void,
        ),
        DataType::UInt64 => (
            None,
            arr.as_primitive::<UInt64Type>().values().as_ptr() as *const c_void,
        ),
        DataType::Float16 => (
            None,
            arr.as_primitive::<Float16Type>().values().as_ptr() as *const c_void,
        ),
        DataType::Float32 => (
            None,
            arr.as_primitive::<Float32Type>().values().as_ptr() as *const c_void,
        ),
        DataType::Float64 => (
            None,
            arr.as_primitive::<Float64Type>().values().as_ptr() as *const c_void,
        ),
        DataType::Dictionary(_, _) => {
            let arr = arr.as_any_dictionary();
            let (key_data, key_ptr) = arr_to_ptr(arr.keys());
            let (val_data, val_ptr) = arr_to_ptr(arr.values());

            // TODO support nested dictionary arrays
            assert!(key_data.is_none() && val_data.is_none());

            let holder = Box::new(PtrHolder {
                arr1: key_ptr,
                arr2: val_ptr,
                phys_len: arr.len() as u64,
                start_at: 0,
                remaining: 0,
            });

            let ptr = &*holder as *const PtrHolder;
            let ptr = ptr as *const c_void;
            (Some(holder), ptr)
        }
        DataType::RunEndEncoded(run_type, _va_type) => {
            let run_type = run_type.data_type();
            let (start_at, remaining) = match run_type {
                DataType::Int16 => {
                    let ra = arr.as_any().downcast_ref::<RunArray<Int16Type>>().unwrap();
                    let start_at = ra.get_start_physical_index();
                    let starting_run_end = ra.run_ends().inner().get(start_at).unwrap();
                    let remaining = *starting_run_end as usize - arr.offset();
                    (start_at, remaining)
                }
                DataType::Int32 => {
                    let ra = arr.as_any().downcast_ref::<RunArray<Int32Type>>().unwrap();
                    let start_at = ra.get_start_physical_index();
                    let starting_run_end = ra.run_ends().inner().get(start_at).unwrap();
                    let remaining = *starting_run_end as usize - arr.offset();
                    (start_at, remaining)
                }
                DataType::Int64 => {
                    let ra = arr.as_any().downcast_ref::<RunArray<Int64Type>>().unwrap();
                    let start_at = ra.get_start_physical_index();
                    let starting_run_end = ra.run_ends().inner().get(start_at).unwrap();
                    let remaining = *starting_run_end as usize - arr.offset();
                    (start_at, remaining)
                }
                _ => unreachable!("invalid run end type (i16, i32, i64 are supported)"),
            };

            let arr_data = arr.to_data();
            let children = arr_data.child_data();
            let re_ptr = children[0].buffer::<u8>(0).as_ptr() as *const c_void;
            let va_ptr = children[1].buffer::<u8>(0).as_ptr() as *const c_void;
            assert!(!re_ptr.is_null(), "run end pointer was null");
            assert!(!va_ptr.is_null(), "value pointer was null");
            assert_eq!(children[0].len(), children[1].len());

            let holder = Box::new(PtrHolder {
                arr1: re_ptr,
                arr2: va_ptr,
                phys_len: children[0].len() as u64,
                start_at: start_at as u64 + 1,
                remaining: remaining as u64,
            });
            let ptr = (&*holder as *const PtrHolder) as *const c_void;
            (Some(holder), ptr)
        }
        _ => todo!(),
    }
}

#[no_mangle]
pub(crate) extern "C" fn printd(label: i64, x: i32) {
    println!("msg {}: {}", label, x);
}

#[no_mangle]
pub(crate) extern "C" fn printd64(label: i64, x: i64) {
    println!("msg {}: {}", label, x);
}

#[no_mangle]
pub(crate) extern "C" fn printv64(label: i64, x: *const i64) {
    let arr: &[i64; 64] = unsafe { &*(x as *const [i64; 64]) };
    println!("msg {}: {:?}", label, arr);
}

#[used]
static EXTERNAL_FN: [extern "C" fn(i64, i32); 1] = [printd];
#[used]
static EXTERNAL_FN64: [extern "C" fn(i64, i64); 1] = [printd64];
#[used]
static EXTERNAL_FNV64: [extern "C" fn(i64, *const i64); 1] = [printv64];

/// A compiled function that owns its own LLVM context. Maps two arrays to an
/// array of booleans.
pub struct CompiledBinaryFunc<'ctx> {
    _cg: CodeGen<'ctx>,
    lhs_dt: DataType,
    lhs_scalar: bool,
    rhs_dt: DataType,
    rhs_scalar: bool,
    f: JitFunction<'ctx, unsafe extern "C" fn(*const c_void, *const c_void, u64, *mut u64)>,
}

impl CompiledBinaryFunc<'_> {
    /// Verify that `arr1` and `arr2` match the types this function was compiled
    /// for, then execute the function and return the result.
    pub fn call(&self, arr1: &dyn Array, arr2: &dyn Array) -> Result<BooleanArray, ArrowError> {
        if arr1.data_type() != &self.lhs_dt {
            return Err(ArrowError::ComputeError(format!(
                "arg 1 had wrong type (expected {:?}, found {:?})",
                self.lhs_dt,
                arr1.data_type()
            )));
        }
        if self.lhs_scalar && arr1.len() != 1 {
            return Err(ArrowError::ComputeError(format!(
                "arg 1 was suppoesd to be scalar, but had length {} (should be 1)",
                arr1.len()
            )));
        }

        if arr2.data_type() != &self.rhs_dt {
            return Err(ArrowError::ComputeError(format!(
                "arg 2 had wrong type (expected {:?}, found {:?})",
                self.rhs_dt,
                arr2.data_type()
            )));
        }
        if self.rhs_scalar && arr2.len() != 1 {
            return Err(ArrowError::ComputeError(format!(
                "arg 2 was suppoesd to be scalar, but had length {} (should be 1)",
                arr2.len()
            )));
        }

        if !self.lhs_scalar && !self.rhs_scalar && arr1.len() != arr2.len() {
            return Err(ArrowError::ComputeError(format!(
                "arrays did not have same length ({} and {})",
                arr1.len(),
                arr2.len()
            )));
        }

        // handle length 0 arrays, since our kernels assume len > 0 (this is
        // mostly so we can do one iteration of the loop prior to checking the
        // loop condition)
        if arr1.is_empty() {
            return Ok(BooleanArray::new_null(0));
        }

        let (data1, ptr1) = arr_to_ptr(arr1);
        let (data2, ptr2) = arr_to_ptr(arr2);

        let mut buf = vec![0_u64; arr1.len().div_ceil(64)];
        unsafe {
            self.f.call(ptr1, ptr2, arr1.len() as u64, buf.as_mut_ptr());
        }

        std::mem::drop(data1);
        std::mem::drop(data2);

        let buf = Buffer::from_vec(buf);
        let bb = BooleanBuffer::new(buf, 0, arr1.len());
        let nulls = NullBuffer::union(arr1.nulls(), arr2.nulls());
        Ok(BooleanArray::new(bb, nulls))
    }
}

/// A compiled function that owns its own LLVM context. Maps an array to another
/// array.
pub struct CompiledConvertFunc<'ctx> {
    _cg: CodeGen<'ctx>,
    src_dt: DataType,
    tar_dt: DataType,
    f: JitFunction<'ctx, unsafe extern "C" fn(*const c_void, u64, *mut c_void)>,
}

impl CompiledConvertFunc<'_> {
    /// Verify that `arr1` and `arr2` match the types this function was compiled
    /// for, then execute the function and return the result.
    pub fn call(&self, arr1: &dyn Array) -> Result<ArrayRef, ArrowError> {
        if arr1.data_type() != &self.src_dt {
            return Err(ArrowError::ComputeError(format!(
                "arg 1 had wrong type (expected {:?}, found {:?})",
                self.src_dt,
                arr1.data_type()
            )));
        }

        // handle length 0 arrays, since our kernels assume len > 0 (this is
        // mostly so we can do one iteration of the loop prior to checking the
        // loop condition)
        if arr1.is_empty() {
            let buf = Buffer::from_vec(Vec::<u128>::with_capacity(0));
            return Ok(buffer_to_primitive(buf, arr1.nulls().cloned(), &self.tar_dt).into());
        }

        let (data1, ptr1) = arr_to_ptr(arr1);

        let mut buf = vec![
            0_u8;
            arr1.len().next_multiple_of(64)
                * PrimitiveType::for_arrow_type(&self.tar_dt).width()
        ];
        unsafe {
            self.f
                .call(ptr1, arr1.len() as u64, buf.as_mut_ptr() as *mut c_void);
        }

        std::mem::drop(data1);
        let buf = Buffer::from_vec(buf);
        let unsliced = buffer_to_primitive(buf, arr1.nulls().cloned(), &self.tar_dt);
        let sliced = unsliced.slice(0, arr1.len());
        Ok(sliced)
    }
}

/// A compiled function that owns its own LLVM context. Maps an array and a
/// boolean array to another array.
pub struct CompiledFilterFunc<'ctx> {
    _cg: CodeGen<'ctx>,
    src_dt: DataType,
    f: JitFunction<'ctx, unsafe extern "C" fn(*const c_void, *const c_void, u64, *mut c_void)>,
}

impl CompiledFilterFunc<'_> {
    /// Verify that `arr1` matchess the types this function was compiled for,
    /// then execute the function and return the result.
    pub fn call(&self, arr1: &dyn Array, ba: &BooleanArray) -> Result<ArrayRef, ArrowError> {
        if arr1.data_type() != &self.src_dt {
            return Err(ArrowError::ComputeError(format!(
                "arg 1 had wrong type (expected {:?}, found {:?})",
                self.src_dt,
                arr1.data_type()
            )));
        }

        let prim_type = PrimitiveType::for_arrow_type(&self.src_dt);

        // handle length 0 arrays, since our kernels assume len > 0 (this is
        // mostly so we can do one iteration of the loop prior to checking the
        // loop condition)
        let true_count = ba.true_count();
        if arr1.is_empty() || true_count == 0 {
            let buf = Buffer::from_vec(Vec::<u128>::with_capacity(0));
            return Ok(
                buffer_to_primitive(buf, arr1.nulls().cloned(), &prim_type.as_arrow_type()).into(),
            );
        }

        let (data, ptr) = arr_to_ptr(arr1);
        let (bdata, bptr) = arr_to_ptr(ba);

        let mut buf = vec![0_u8; true_count * prim_type.width()];

        unsafe {
            self.f.call(
                ptr,
                bptr,
                arr1.len() as u64,
                buf.as_mut_ptr() as *mut c_void,
            );
        }

        std::mem::drop(data);
        std::mem::drop(bdata);
        let buf = Buffer::from_vec(buf);

        let unsliced = buffer_to_primitive(buf, arr1.nulls().cloned(), &prim_type.as_arrow_type());

        let sliced = unsliced.slice(0, ba.true_count());
        Ok(sliced)
    }
}

/// A compiled function that owns its own LLVM context. Maps an array to a
/// single value.
pub struct CompiledAggFunc<'ctx> {
    _cg: CodeGen<'ctx>,
    src_dt: DataType,
    f: JitFunction<'ctx, unsafe extern "C" fn(*const c_void, u64, *mut c_void)>,
}

impl CompiledAggFunc<'_> {
    /// Verify that `arr1` matchess the types this function was compiled for,
    /// then execute the function and return the result.
    pub fn call(&self, arr1: &dyn Array) -> Result<Option<Box<dyn Array>>, ArrowError> {
        if arr1.data_type() != &self.src_dt {
            return Err(ArrowError::ComputeError(format!(
                "arg 1 had wrong type (expected {:?}, found {:?})",
                self.src_dt,
                arr1.data_type()
            )));
        }

        let prim_type = PrimitiveType::for_arrow_type(&self.src_dt);

        // handle length 0 arrays
        if arr1.is_empty() {
            return Ok(None);
        }

        let (data, ptr) = arr_to_ptr(arr1);

        let mut buf = vec![0_u8; prim_type.width()];

        unsafe {
            self.f
                .call(ptr, arr1.len() as u64, buf.as_mut_ptr() as *mut c_void);
        }

        std::mem::drop(data);

        let buf = Buffer::from_vec(buf);
        let res = buffer_to_primitive(buf, None, &prim_type.as_arrow_type());

        Ok(Some(res.into()))
    }
}

/// Code generation routines. Used to generate `CompiledFunc`s.
///
/// The `cmp` interface automatically caches compiled functions for reuse, but
/// this interface does not (i.e., each time you use a function here, the
/// underlying kernel will be recompiled).
pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    /// Create a new codegen object within an LLVM context.
    /// ```rust
    /// use inkwell::context::Context;
    /// use arrow_compile_compute::CodeGen;
    /// let ctx = Context::create();
    /// let cg = CodeGen::new(&ctx);
    /// ```
    pub fn new(ctx: &Context) -> CodeGen {
        let module = ctx.create_module("jit");
        CodeGen {
            context: ctx,
            module,
        }
    }

    fn optimize(&self) -> Result<(), ArrowError> {
        Target::initialize_native(&inkwell::targets::InitializationConfig::default()).unwrap();
        let triple = TargetMachine::get_default_triple();
        let cpu = TargetMachine::get_host_cpu_name().to_string();
        let features = TargetMachine::get_host_cpu_features().to_string();
        let target = Target::from_triple(&triple).unwrap();
        let machine = target
            .create_target_machine(
                &triple,
                &cpu,
                &features,
                OptimizationLevel::Aggressive,
                RelocMode::Default,
                CodeModel::Default,
            )
            .unwrap();

        self.module
            .run_passes("default<O3>", &machine, PassBuilderOptions::create())
            .map_err(|e| ArrowError::ComputeError(format!("Error optimizing kernel: {}", e)))?;
        Ok(())
    }

    fn increment_pointer<'a>(
        &'a self,
        builder: &'a Builder,
        p: PointerValue<'a>,
        w: usize,
        inc: IntValue<'a>,
    ) -> PointerValue<'a>
    where
        'ctx: 'a,
    {
        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let as_int = builder.build_ptr_to_int(p, i64_type, "as_int").unwrap();
        let incr = builder
            .build_int_mul(inc, i64_type.const_int(w as u64, false), "incr")
            .unwrap();
        let incr_int = builder.build_int_add(as_int, incr, "incr_ptr").unwrap();
        let ptr = builder.build_int_to_ptr(incr_int, ptr_type, "ptr").unwrap();
        ptr
    }

    fn gen_convert_vec<'a>(
        &'a self,
        builder: &'a Builder,
        v: VectorValue<'a>,
        src: PrimitiveType,
        dst: PrimitiveType,
    ) -> VectorValue<'a>
    where
        'ctx: 'a,
    {
        if src == dst {
            return v;
        }

        let dst_llvm = dst.llvm_type(self.context);

        match (src.is_int(), dst.is_int()) {
            // int to int
            (true, true) => {
                let dst_vec = dst_llvm.into_int_type().vec_type(64);
                if src.width() > dst.width() {
                    builder.build_int_truncate(v, dst_vec, "trunc").unwrap()
                } else if src.is_signed() {
                    builder.build_int_s_extend(v, dst_vec, "sext").unwrap()
                } else {
                    builder.build_int_z_extend(v, dst_vec, "sext").unwrap()
                }
            }
            // int to float
            (true, false) => {
                let dst_vec = dst_llvm.into_float_type().vec_type(64);
                if src.is_signed() {
                    builder
                        .build_signed_int_to_float(v, dst_vec, "sitf")
                        .unwrap()
                } else {
                    builder
                        .build_signed_int_to_float(v, dst_vec, "uitf")
                        .unwrap()
                }
            }
            // float to int
            (false, true) => {
                let dst_vec = dst_llvm.into_int_type().vec_type(64);
                if dst.is_signed() {
                    builder
                        .build_float_to_signed_int(v, dst_vec, "ftsi")
                        .unwrap()
                } else {
                    builder
                        .build_float_to_unsigned_int(v, dst_vec, "ftui")
                        .unwrap()
                }
            }
            // float to float
            (false, false) => {
                let dst_vec = dst_llvm.into_float_type().vec_type(64);
                if src.width() > dst.width() {
                    builder.build_float_trunc(v, dst_vec, "ftrun").unwrap()
                } else {
                    builder.build_float_ext(v, dst_vec, "fext").unwrap()
                }
            }
        }
    }

    fn gen_iter_for(&self, label: &str, dt: &DataType) -> FunctionValue {
        match dt {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64 => {
                self.gen_iter_primitive(label, PrimitiveType::for_arrow_type(dt), 64)
            }
            DataType::Dictionary(k_dt, v_dt) => self.gen_dict_primitive(
                label,
                PrimitiveType::for_arrow_type(k_dt),
                PrimitiveType::for_arrow_type(v_dt),
            ),
            DataType::RunEndEncoded(re_dt, v_dt) => self.gen_re_primitive(
                label,
                PrimitiveType::for_arrow_type(re_dt.data_type()),
                PrimitiveType::for_arrow_type(v_dt.data_type()),
            ),
            _ => todo!(),
        }
    }

    fn initialize_iter<'a>(
        &'a self,
        builder: &'a Builder<'ctx>,
        ptr: PointerValue<'a>,
        len: IntValue<'a>,
        dt: &DataType,
    ) -> PointerValue<'a> {
        match dt {
            DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64 => self.initialize_iter_primitive(builder, ptr, len),
            DataType::Dictionary(_k_dt, _v_dt) => self.initialize_iter_dict(builder, ptr, len),
            DataType::RunEndEncoded(_re_dt, _v_dt) => self.initialize_iter_re(builder, ptr, len),
            _ => todo!(),
        }
    }

    fn has_next_iter<'a>(
        &'a self,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
        dt: &DataType,
    ) -> IntValue<'a> {
        match dt {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64 => self.has_next_iter_primitive(builder, ptr),
            DataType::Dictionary(_, _) => self.has_next_iter_dict(builder, ptr),
            DataType::RunEndEncoded(_, _) => self.has_next_iter_re(builder, ptr),
            _ => todo!(),
        }
    }

    fn convert_float_for_total_cmp<'a>(
        &'a self,
        builder: &'a Builder,
        fvec: VectorValue<'a>,
        ptype: PrimitiveType,
    ) -> VectorValue<'a> {
        // apply the following algorithm to get a total float ordering
        // left ^= (((left >> 63) as u64) >> 1) as i64;
        // right ^= (((right >> 63) as u64) >> 1) as i64;
        // left.cmp(right)

        let int_type = PrimitiveType::int_with_width(ptype.width())
            .llvm_type(self.context)
            .into_int_type();

        let vminus1 = builder
            .build_insert_element(
                int_type.vec_type(64).const_zero(),
                int_type.const_int(ptype.width() as u64 * 8 - 1, false),
                int_type.const_zero(),
                "vminus1",
            )
            .unwrap();
        let vminus1 = builder
            .build_shuffle_vector(
                vminus1,
                int_type.vec_type(64).get_undef(),
                int_type.vec_type(64).const_zero(),
                "vminus1_bcast",
            )
            .unwrap();
        let v1 = builder
            .build_insert_element(
                int_type.vec_type(64).const_zero(),
                int_type.const_int(1, false),
                int_type.const_zero(),
                "v1",
            )
            .unwrap();
        let v1 = builder
            .build_shuffle_vector(
                v1,
                int_type.vec_type(64).get_undef(),
                int_type.vec_type(64).const_zero(),
                "v1_bcast",
            )
            .unwrap();

        let cleft = builder
            .build_bit_cast(fvec, int_type.vec_type(64), "cleft")
            .unwrap()
            .into_vector_value();
        let left = builder
            .build_right_shift(cleft, vminus1, false, "sleft")
            .unwrap();
        let left = builder.build_right_shift(left, v1, true, "sleft").unwrap();
        builder.build_xor(cleft, left, "left").unwrap()
    }

    /// Generate a `CompiledFunc` for the given datatypes and predicate.
    pub fn primitive_primitive_cmp(
        self,
        lhs_dt: &DataType,
        lhs_scalar: bool,
        rhs_dt: &DataType,
        rhs_scalar: bool,
        p: Predicate,
    ) -> Result<CompiledBinaryFunc<'ctx>, ArrowError> {
        let lhs_prim = PrimitiveType::for_arrow_type(lhs_dt);
        let rhs_prim = PrimitiveType::for_arrow_type(rhs_dt);
        let com_prim = PrimitiveType::dominant(lhs_prim, rhs_prim);
        let builder = self.context.create_builder();

        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());

        let lhs_iter_next = if !lhs_scalar {
            self.gen_iter_for("left", lhs_dt)
        } else {
            self.gen_iter_scalar("left", PrimitiveType::for_arrow_type(lhs_dt))
        };
        let rhs_iter_next = if !rhs_scalar {
            self.gen_iter_for("right", rhs_dt)
        } else {
            self.gen_iter_scalar("right", PrimitiveType::for_arrow_type(rhs_dt))
        };

        let fn_type = self.context.void_type().fn_type(
            &[
                ptr_type.into(), // arr1
                ptr_type.into(), // arr2
                i64_type.into(), // len
                ptr_type.into(), // out
            ],
            false,
        );
        let function = self.module.add_function("eq_arr", fn_type, None);

        let arr1_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let arr2_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let len = function.get_nth_param(2).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(3).unwrap().into_pointer_value();

        let entry = self.context.append_basic_block(function, "entry");
        let loop_cond = self.context.append_basic_block(function, "loop_cond");
        let loop_body = self.context.append_basic_block(function, "loop_body");
        let end = self.context.append_basic_block(function, "end");

        builder.position_at_end(entry);
        let lhs_iter_ptr = if !lhs_scalar {
            self.initialize_iter(&builder, arr1_ptr, len, lhs_dt)
        } else {
            self.initialize_iter_scalar(&builder, arr1_ptr, len)
        };
        let rhs_iter_ptr = if !rhs_scalar {
            self.initialize_iter(&builder, arr2_ptr, len, rhs_dt)
        } else {
            self.initialize_iter_scalar(&builder, arr2_ptr, len)
        };

        let out_idx_ptr = builder.build_alloca(ptr_type, "out_idx_ptr").unwrap();
        builder
            .build_store(out_idx_ptr, i64_type.const_zero())
            .unwrap();
        builder.build_unconditional_branch(loop_body).unwrap();

        builder.position_at_end(loop_body);
        let lhs_chunk = builder
            .build_call(lhs_iter_next, &[lhs_iter_ptr.into()], "lhs")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_vector_value();
        let lhs_chunk = self.gen_convert_vec(&builder, lhs_chunk, lhs_prim, com_prim);
        let rhs_chunk = builder
            .build_call(rhs_iter_next, &[rhs_iter_ptr.into()], "rhs")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_vector_value();
        let rhs_chunk = self.gen_convert_vec(&builder, rhs_chunk, rhs_prim, com_prim);
        let mask = if com_prim.is_int() {
            builder.build_int_compare(
                p.as_int_pred(com_prim.is_signed()),
                lhs_chunk,
                rhs_chunk,
                "mask",
            )
        } else {
            let left = self.convert_float_for_total_cmp(&builder, lhs_chunk, com_prim);
            let right = self.convert_float_for_total_cmp(&builder, rhs_chunk, com_prim);
            builder.build_int_compare(p.as_int_pred(true), left, right, "mask")
        }
        .unwrap();

        let mask = builder
            .build_bit_cast(mask, i64_type, "mask_casted")
            .unwrap();
        let out_idx = builder
            .build_load(i64_type, out_idx_ptr, "out_idx")
            .unwrap()
            .into_int_value();
        let this_out_ptr = self.increment_pointer(&builder, out_ptr, 8, out_idx);
        builder.build_store(this_out_ptr, mask).unwrap();
        let next_out_idx = builder
            .build_int_add(out_idx, i64_type.const_int(1, false), "next_out_idx")
            .unwrap();
        builder.build_store(out_idx_ptr, next_out_idx).unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(loop_cond);
        let has_next = if !lhs_scalar {
            self.has_next_iter(&builder, lhs_iter_ptr, lhs_dt)
        } else {
            self.has_next_iter_scalar(&builder, lhs_iter_ptr)
        };

        builder
            .build_conditional_branch(has_next, loop_body, end)
            .unwrap();

        builder.position_at_end(end);
        builder.build_return(None).unwrap();

        self.optimize()?;
        self.module
            .verify()
            .map_err(|e| ArrowError::ComputeError(format!("Error compiling kernel: {}", e)))?;
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        Ok(CompiledBinaryFunc {
            _cg: self,
            lhs_dt: lhs_dt.clone(),
            lhs_scalar,
            rhs_dt: rhs_dt.clone(),
            rhs_scalar,
            f: unsafe { ee.get_function("eq_arr").ok().unwrap() },
        })
    }

    pub fn cast_to_primitive(
        self,
        src_dt: &DataType,
        tar_dt: &DataType,
    ) -> Result<CompiledConvertFunc<'ctx>, ArrowError> {
        let builder = self.context.create_builder();

        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());

        let fn_type = self.context.void_type().fn_type(
            &[
                ptr_type.into(), // src
                i64_type.into(), // len
                ptr_type.into(), // tar
            ],
            false,
        );
        let function = self.module.add_function("cast_to_prim", fn_type, None);

        let arr1_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let len = function.get_nth_param(1).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(2).unwrap().into_pointer_value();

        let next = self.gen_iter_for("source", src_dt);
        let entry = self.context.append_basic_block(function, "entry");
        let loop_cond = self.context.append_basic_block(function, "loop_cond");
        let loop_body = self.context.append_basic_block(function, "loop_body");
        let end = self.context.append_basic_block(function, "end");

        let src_prim_type = PrimitiveType::for_arrow_type(src_dt);
        let tar_prim_type = PrimitiveType::for_arrow_type(tar_dt);

        builder.position_at_end(entry);
        let out_idx_ptr = builder.build_alloca(i64_type, "out_idx_ptr").unwrap();
        builder
            .build_store(out_idx_ptr, i64_type.const_zero())
            .unwrap();
        let iter_ptr = self.initialize_iter(&builder, arr1_ptr, len, src_dt);
        builder.build_unconditional_branch(loop_body).unwrap();

        builder.position_at_end(loop_body);
        let block = builder
            .build_call(next, &[iter_ptr.into()], "block")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_vector_value();
        let block = self.gen_convert_vec(&builder, block, src_prim_type, tar_prim_type);
        let out_idx = builder
            .build_load(i64_type, out_idx_ptr, "out_idx")
            .unwrap()
            .into_int_value();
        let out_pos = self.increment_pointer(&builder, out_ptr, tar_prim_type.width(), out_idx);
        builder.build_store(out_pos, block).unwrap();

        let inc_out_pos = builder
            .build_int_add(out_idx, i64_type.const_int(64, false), "inc_out_pos")
            .unwrap();
        builder.build_store(out_idx_ptr, inc_out_pos).unwrap();
        builder.build_unconditional_branch(loop_cond).unwrap();

        builder.position_at_end(loop_cond);
        let out_pos = builder
            .build_load(i64_type, out_idx_ptr, "out_pos")
            .unwrap()
            .into_int_value();
        let cmp = builder
            .build_int_compare(IntPredicate::SGE, out_pos, len, "cmp")
            .unwrap();
        builder
            .build_conditional_branch(cmp, end, loop_body)
            .unwrap();

        builder.position_at_end(end);
        builder.build_return(None).unwrap();

        //self.optimize()?;
        self.module
            .verify()
            .map_err(|e| ArrowError::ComputeError(format!("Error compiling kernel: {}", e)))?;
        let ee = self
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        Ok(CompiledConvertFunc {
            _cg: self,
            src_dt: src_dt.clone(),
            tar_dt: tar_dt.clone(),
            f: unsafe { ee.get_function("cast_to_prim").ok().unwrap() },
        })
    }
}

#[cfg(test)]
pub mod test_utils {
    use arrow_array::{types::Int64Type, Int64Array, RunArray};
    use itertools::Itertools;

    pub fn generate_random_ree_array(num_run_ends: usize) -> RunArray<Int64Type> {
        let mut rng = fastrand::Rng::with_seed(42 + num_run_ends as u64);
        let ree_array_run_ends = (0..num_run_ends)
            .map(|_| rng.i64(1..40))
            .scan(0, |acc, x| {
                *acc = *acc + x;
                Some(*acc)
            })
            .collect_vec();
        let ree_array_values = (0..num_run_ends).map(|_| rng.i64(-5..5)).collect_vec();
        RunArray::try_new(
            &Int64Array::from(ree_array_run_ends),
            &Int64Array::from(ree_array_values),
        )
        .unwrap()
    }
}
