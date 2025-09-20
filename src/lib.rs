//! Compiled compute functions for the [arrow-rs](https://github.com/apache/arrow-rs).

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Date64Array,
    FixedSizeBinaryArray, Float16Array, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, LargeBinaryArray, LargeStringArray, NullArray, StringArray,
    StringViewArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field};
use inkwell::{
    attributes::{Attribute, AttributeLoc},
    context::Context,
    types::{BasicTypeEnum, VectorType},
    values::FunctionValue,
    AddressSpace, IntPredicate,
};

mod arrow_interface;
mod compiled_iter;
mod compiled_kernels;

pub use arrow_interface::aggregate;
pub use arrow_interface::arith;
pub use arrow_interface::cast;
pub use arrow_interface::cmp;
pub use arrow_interface::compute;
pub use arrow_interface::iter;
pub use arrow_interface::select;

pub use compiled_kernels::ArrowKernelError;
pub use compiled_kernels::Kernel;
pub use compiled_kernels::SortOptions;

macro_rules! mark_load_invariant {
    ($ctx:expr, $instruction:expr) => {{
        use inkwell::values::BasicValue;
        $instruction
            .as_instruction_value()
            .unwrap()
            .set_metadata($ctx.metadata_node(&[]), $ctx.get_kind_id("invariant.load"))
            .unwrap();
    }};
}
pub(crate) use mark_load_invariant;

macro_rules! declare_global_pointer {
    ($module:expr, $label:ident) => {{
        let ptr_type = $module.get_context().ptr_type(AddressSpace::default());
        let global_var = $module.add_global(ptr_type, None, stringify!($label));
        global_var.set_initializer(&ptr_type.const_null());
        global_var.set_linkage(inkwell::module::Linkage::Private);
        global_var
    }};
}
pub(crate) use declare_global_pointer;

/// Declare a set of basic blocks at once
macro_rules! declare_blocks {
    ($ctx:expr, $func:expr, $name:ident) => {
        let $name = $ctx.append_basic_block($func, stringify!($name));
    };
    ($ctx:expr, $func:expr, $name:ident, $($more:ident),+) => {
        let $name = $ctx.append_basic_block($func, stringify!($name));
        declare_blocks!($ctx, $func, $($more),+);
    };
}
pub(crate) use declare_blocks;

macro_rules! pointer_diff {
    ($ctx: expr, $builder: expr, $ptr1: expr, $ptr2: expr) => {{
        let as_int1 = $builder
            .build_ptr_to_int($ptr1, $ctx.i64_type(), "as_int1")
            .unwrap();
        let as_int2 = $builder
            .build_ptr_to_int($ptr2, $ctx.i64_type(), "as_int2")
            .unwrap();
        $builder.build_int_sub(as_int2, as_int1, "diff").unwrap()
    }};
}
pub(crate) use pointer_diff;

/// Increments a pointer by a fixed number of bytes or with a stride
macro_rules! increment_pointer {
    ($ctx: expr, $builder: expr, $ptr: expr, $offset: expr) => {
        unsafe {
            $builder
                .build_gep(
                    $ctx.i8_type(),
                    $ptr,
                    &[$ctx.i64_type().const_int($offset as u64, false)],
                    "inc_ptr",
                )
                .unwrap()
        }
    };
    ($ctx: expr, $builder: expr, $ptr: expr, $stride: expr, $offset: expr) => {
        unsafe {
            let i64_type = $ctx.i64_type();
            let tmp = $builder
                .build_int_mul(
                    i64_type.const_int($stride as u64, false),
                    $offset,
                    "strided",
                )
                .unwrap();
            $builder
                .build_gep($ctx.i8_type(), $ptr, &[tmp], "inc_ptr")
                .unwrap()
        }
    };
}
pub(crate) use increment_pointer;

/// Mark each parameter of the given function as noalias for LLVM IR
/// optimizations.
fn set_noalias_params(func: &FunctionValue) {
    let context = func.get_type().get_context();
    let noalias_kind_id = Attribute::get_named_enum_kind_id("noalias");
    for i in 0..func.count_params() {
        let attr = context.create_enum_attribute(noalias_kind_id, 0);
        func.add_attribute(AttributeLoc::Param(i), attr);
    }
}

#[cfg(test)]
unsafe fn pointers_to_str(ptrs: u128) -> String {
    let b = ptrs.to_le_bytes();
    let ptr1 = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let ptr2 = u64::from_le_bytes(b[8..16].try_into().unwrap());
    let len = (ptr2 - ptr1) as usize;

    unsafe {
        let slice = std::slice::from_raw_parts(ptr1 as *const u8, len);
        let string = std::str::from_utf8(slice).unwrap();
        string.to_string()
    }
}

/// Utility function to create the appropriate `DataType` for a dictionary array
pub fn dictionary_data_type(key_type: DataType, val_type: DataType) -> DataType {
    DataType::Dictionary(Box::new(key_type), Box::new(val_type))
}

/// Utility function to create the appropriate `DataType` for a run-end encoded
/// array
pub fn run_end_data_type(run_type: &DataType, val_type: &DataType) -> DataType {
    let f1 = Arc::new(Field::new("run_ends", run_type.clone(), false));
    let f2 = Arc::new(Field::new("values", val_type.clone(), true));
    DataType::RunEndEncoded(f1, f2)
}

pub fn empty_array_for(dt: &DataType) -> ArrayRef {
    match dt {
        DataType::Null => Arc::new(NullArray::new(0)),
        DataType::Boolean => Arc::new(BooleanArray::new_null(0)),
        DataType::Int8 => Arc::new(Int8Array::new_null(0)),
        DataType::Int16 => Arc::new(Int16Array::new_null(0)),
        DataType::Int32 => Arc::new(Int32Array::new_null(0)),
        DataType::Int64 => Arc::new(Int64Array::new_null(0)),
        DataType::UInt8 => Arc::new(UInt8Array::new_null(0)),
        DataType::UInt16 => Arc::new(UInt16Array::new_null(0)),
        DataType::UInt32 => Arc::new(UInt32Array::new_null(0)),
        DataType::UInt64 => Arc::new(UInt64Array::new_null(0)),
        DataType::Float16 => Arc::new(Float16Array::new_null(0)),
        DataType::Float32 => Arc::new(Float32Array::new_null(0)),
        DataType::Float64 => Arc::new(Float64Array::new_null(0)),
        DataType::Timestamp(_time_unit, _) => todo!(),
        DataType::Date32 => Arc::new(Date32Array::new_null(0)),
        DataType::Date64 => Arc::new(Date64Array::new_null(0)),
        DataType::Time32(_time_unit) => todo!(),
        DataType::Time64(_time_unit) => todo!(),
        DataType::Duration(_time_unit) => todo!(),
        DataType::Interval(_interval_unit) => todo!(),
        DataType::Binary => Arc::new(BinaryArray::new_null(0)),
        DataType::FixedSizeBinary(s) => Arc::new(FixedSizeBinaryArray::new_null(*s, 0)),
        DataType::LargeBinary => Arc::new(LargeBinaryArray::new_null(0)),
        DataType::BinaryView => Arc::new(BinaryViewArray::new_null(0)),
        DataType::Utf8 => Arc::new(StringArray::new_null(0)),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::new_null(0)),
        DataType::Utf8View => Arc::new(StringViewArray::new_null(0)),
        DataType::List(_field) => todo!(),
        DataType::ListView(_field) => todo!(),
        DataType::FixedSizeList(_field, _) => todo!(),
        DataType::LargeList(_field) => todo!(),
        DataType::LargeListView(_field) => todo!(),
        DataType::Struct(_fields) => todo!(),
        DataType::Union(_union_fields, _union_mode) => todo!(),
        DataType::Dictionary(_key, _val) => todo!(),
        DataType::Decimal128(_, _) => todo!(),
        DataType::Decimal256(_, _) => todo!(),
        DataType::Map(_field, _) => todo!(),
        DataType::RunEndEncoded(_field, _field1) => todo!(),
    }
}

pub fn logical_arrow_type(dt: &DataType) -> DataType {
    match dt {
        DataType::Dictionary(_kt, vt) => logical_arrow_type(vt),
        DataType::RunEndEncoded(_re_type, vt) => logical_arrow_type(vt.data_type()),
        _ => dt.clone(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    P64x2,
    F16,
    F32,
    F64,
}

#[derive(Copy, Clone, Debug)]
enum ComparisonType {
    Int { signed: bool },
    Float,
    String,
}

impl std::fmt::Display for PrimitiveType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrimitiveType::I8 => "I8".fmt(f),
            PrimitiveType::I16 => "I16".fmt(f),
            PrimitiveType::I32 => "I32".fmt(f),
            PrimitiveType::I64 => "I64".fmt(f),
            PrimitiveType::U8 => "U8".fmt(f),
            PrimitiveType::U16 => "U16".fmt(f),
            PrimitiveType::U32 => "U32".fmt(f),
            PrimitiveType::U64 => "U64".fmt(f),
            PrimitiveType::P64x2 => "P64x2".fmt(f),
            PrimitiveType::F16 => "F16".fmt(f),
            PrimitiveType::F32 => "F32".fmt(f),
            PrimitiveType::F64 => "F64".fmt(f),
        }
    }
}

impl PrimitiveType {
    const fn max_width() -> usize {
        16
    }

    const fn max_width_type() -> Self {
        PrimitiveType::P64x2
    }

    const fn width(&self) -> usize {
        // if any of these widths are updated, make sure to check `max_width` as
        // well!
        match self {
            PrimitiveType::I8 | PrimitiveType::U8 => 1,
            PrimitiveType::I16 | PrimitiveType::U16 => 2,
            PrimitiveType::I32 | PrimitiveType::U32 => 4,
            PrimitiveType::I64 | PrimitiveType::U64 => 8,
            PrimitiveType::P64x2 => 16,
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
            PrimitiveType::P64x2 => ctx
                .struct_type(
                    &[
                        ctx.ptr_type(AddressSpace::default()).into(),
                        ctx.ptr_type(AddressSpace::default()).into(),
                    ],
                    false,
                )
                .into(),
            PrimitiveType::F16 => ctx.f16_type().into(),
            PrimitiveType::F32 => ctx.f32_type().into(),
            PrimitiveType::F64 => ctx.f64_type().into(),
        }
    }

    fn llvm_vec_type<'a>(&self, ctx: &'a Context, size: u32) -> Option<VectorType<'a>> {
        match self {
            PrimitiveType::I8 | PrimitiveType::U8 => Some(ctx.i8_type().vec_type(size)),
            PrimitiveType::I16 | PrimitiveType::U16 => Some(ctx.i16_type().vec_type(size)),
            PrimitiveType::I32 | PrimitiveType::U32 => Some(ctx.i32_type().vec_type(size)),
            PrimitiveType::I64 | PrimitiveType::U64 => Some(ctx.i64_type().vec_type(size)),
            PrimitiveType::P64x2 => None,
            PrimitiveType::F16 => Some(ctx.f16_type().vec_type(size)),
            PrimitiveType::F32 => Some(ctx.f32_type().vec_type(size)),
            PrimitiveType::F64 => Some(ctx.f64_type().vec_type(size)),
        }
    }

    fn for_arrow_type(dt: &DataType) -> Self {
        match dt {
            DataType::Boolean => PrimitiveType::U8,
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
            DataType::Utf8 => PrimitiveType::P64x2, // string view
            DataType::LargeUtf8 => PrimitiveType::P64x2, // string view
            DataType::Utf8View => PrimitiveType::P64x2, // string view
            _ => todo!("no prim type for {:?}", dt),
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
            PrimitiveType::P64x2 => DataType::Utf8,
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
            16 => PrimitiveType::P64x2,
            8 => PrimitiveType::I64,
            4 => PrimitiveType::I32,
            2 => PrimitiveType::I16,
            1 => PrimitiveType::I8,
            _ => unreachable!("width must be 16, 8, 4, 2, or 1"),
        }
    }

    fn is_signed(&self) -> bool {
        match self {
            PrimitiveType::I8 | PrimitiveType::I16 | PrimitiveType::I32 | PrimitiveType::I64 => {
                true
            }
            PrimitiveType::U8
            | PrimitiveType::U16
            | PrimitiveType::U32
            | PrimitiveType::U64
            | PrimitiveType::P64x2 => false,
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
            PrimitiveType::P64x2 | PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                false
            }
        }
    }

    /// Returns the "best" common value to cast both types to in order to
    /// perform a comparison. Returns `None` if there is no compatible type.
    fn dominant(lhs_prim: PrimitiveType, rhs_prim: PrimitiveType) -> Option<PrimitiveType> {
        if matches!(lhs_prim, PrimitiveType::P64x2) && matches!(rhs_prim, PrimitiveType::P64x2) {
            Some(PrimitiveType::P64x2)
        } else if matches!(lhs_prim, PrimitiveType::P64x2)
            || matches!(rhs_prim, PrimitiveType::P64x2)
        {
            None
        } else if lhs_prim.is_signed() != rhs_prim.is_signed() {
            Some(PrimitiveType::I64)
        } else if lhs_prim.width() >= rhs_prim.width() {
            Some(lhs_prim)
        } else {
            Some(rhs_prim)
        }
    }

    fn comparison_type(&self) -> ComparisonType {
        match self {
            PrimitiveType::I8 | PrimitiveType::I16 | PrimitiveType::I32 | PrimitiveType::I64 => {
                ComparisonType::Int { signed: true }
            }
            PrimitiveType::U8 | PrimitiveType::U16 | PrimitiveType::U32 | PrimitiveType::U64 => {
                ComparisonType::Int { signed: false }
            }
            PrimitiveType::P64x2 => ComparisonType::String,
            PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => ComparisonType::Float,
        }
    }
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
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

    /// Returns the variant of this predicate that produces the same answer if
    /// the operands are flipped. For example:
    ///
    /// `(A == B) == (B == A)`
    /// `(A > B) == (B < A)`
    fn flip(&self) -> Predicate {
        match self {
            Predicate::Eq => Predicate::Eq,
            Predicate::Ne => Predicate::Ne,
            Predicate::Lt => Predicate::Gt,
            Predicate::Lte => Predicate::Gte,
            Predicate::Gt => Predicate::Lt,
            Predicate::Gte => Predicate::Lte,
        }
    }
}
