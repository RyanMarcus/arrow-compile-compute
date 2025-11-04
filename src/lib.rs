//! Compiled compute functions for the [arrow-rs](https://github.com/apache/arrow-rs).

use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    make_array,
    types::{Int16Type, Int32Type, Int64Type, RunEndIndexType},
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Date64Array, Datum,
    FixedSizeBinaryArray, Float16Array, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, LargeBinaryArray, LargeStringArray, NullArray, PrimitiveArray,
    StringArray, StringViewArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_buffer::NullBuffer;
use arrow_data::ArrayDataBuilder;
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
pub(crate) mod compiled_writers;
mod llvm_debug;

pub use arrow_interface::aggregate;
pub use arrow_interface::arith;
pub use arrow_interface::cast;
pub use arrow_interface::cmp;
pub use arrow_interface::compute;
pub use arrow_interface::iter;
pub use arrow_interface::select;
pub use arrow_interface::sort;
pub use arrow_interface::vec;

pub use compiled_kernels::compile_string_like;
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
        global_var.set_thread_local(true);
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
pub enum PrimitiveSuperType {
    Int,
    UInt,
    Float,
    String,
    List(ListItemType, usize),
}

impl PrimitiveSuperType {
    pub fn list_type_into_inner(&self) -> PrimitiveSuperType {
        match self {
            PrimitiveSuperType::List(t, _) => PrimitiveSuperType::from(PrimitiveType::from(*t)),
            _ => *self,
        }
    }
}

impl From<PrimitiveType> for PrimitiveSuperType {
    fn from(value: PrimitiveType) -> Self {
        match value {
            PrimitiveType::I8 | PrimitiveType::I16 | PrimitiveType::I32 | PrimitiveType::I64 => {
                PrimitiveSuperType::Int
            }
            PrimitiveType::U8 | PrimitiveType::U16 | PrimitiveType::U32 | PrimitiveType::U64 => {
                PrimitiveSuperType::UInt
            }
            PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                PrimitiveSuperType::Float
            }
            PrimitiveType::P64x2 => PrimitiveSuperType::String,
            PrimitiveType::List(t, s) => PrimitiveSuperType::List(t, s),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ListItemType {
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
    P64x2,
}

impl TryFrom<PrimitiveType> for ListItemType {
    type Error = ArrowKernelError;

    fn try_from(value: PrimitiveType) -> Result<Self, Self::Error> {
        Ok(match value {
            PrimitiveType::I8 => ListItemType::I8,
            PrimitiveType::I16 => ListItemType::I16,
            PrimitiveType::I32 => ListItemType::I32,
            PrimitiveType::I64 => ListItemType::I64,
            PrimitiveType::U8 => ListItemType::U8,
            PrimitiveType::U16 => ListItemType::U16,
            PrimitiveType::U32 => ListItemType::U32,
            PrimitiveType::U64 => ListItemType::U64,
            PrimitiveType::F16 => ListItemType::F16,
            PrimitiveType::F32 => ListItemType::F32,
            PrimitiveType::F64 => ListItemType::F64,
            PrimitiveType::P64x2 => ListItemType::P64x2,
            PrimitiveType::List(_, _) => {
                return Err(ArrowKernelError::UnsupportedArguments(format!(
                    "nested lists not supported"
                )))
            }
        })
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
    F16,
    F32,
    F64,
    P64x2,
    List(ListItemType, usize),
}

#[derive(Copy, Clone, Debug)]
enum ComparisonType {
    Int { signed: bool },
    Float,
    String,
    List,
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
            PrimitiveType::List(item_type, size) => {
                write!(f, "List({:?}x{})", item_type, size)
            }
        }
    }
}

impl PrimitiveType {
    fn width(&self) -> usize {
        match self {
            PrimitiveType::I8 | PrimitiveType::U8 => 1,
            PrimitiveType::I16 | PrimitiveType::U16 => 2,
            PrimitiveType::I32 | PrimitiveType::U32 => 4,
            PrimitiveType::I64 | PrimitiveType::U64 => 8,
            PrimitiveType::P64x2 => 16,
            PrimitiveType::F16 => 2,
            PrimitiveType::F32 => 4,
            PrimitiveType::F64 => 8,
            PrimitiveType::List(t, s) => PrimitiveType::from(*t).width() * s,
        }
    }

    fn llvm_type<'a>(&self, ctx: &'a Context) -> BasicTypeEnum<'a> {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        match self {
            PrimitiveType::I8 | PrimitiveType::U8 => ctx.i8_type().into(),
            PrimitiveType::I16 | PrimitiveType::U16 => ctx.i16_type().into(),
            PrimitiveType::I32 | PrimitiveType::U32 => ctx.i32_type().into(),
            PrimitiveType::I64 | PrimitiveType::U64 => ctx.i64_type().into(),
            PrimitiveType::P64x2 => ctx
                .struct_type(&[ptr_type.into(), ptr_type.into()], false)
                .into(),
            PrimitiveType::F16 => ctx.f16_type().into(),
            PrimitiveType::F32 => ctx.f32_type().into(),
            PrimitiveType::F64 => ctx.f64_type().into(),
            PrimitiveType::List(t, s) => PrimitiveType::from(*t)
                .llvm_vec_type(ctx, *s as u32)
                .unwrap()
                .into(),
        }
    }

    fn llvm_vec_type<'a>(&self, ctx: &'a Context, size: u32) -> Option<VectorType<'a>> {
        match self {
            PrimitiveType::I8 | PrimitiveType::U8 => Some(ctx.i8_type().vec_type(size)),
            PrimitiveType::I16 | PrimitiveType::U16 => Some(ctx.i16_type().vec_type(size)),
            PrimitiveType::I32 | PrimitiveType::U32 => Some(ctx.i32_type().vec_type(size)),
            PrimitiveType::I64 | PrimitiveType::U64 => Some(ctx.i64_type().vec_type(size)),
            PrimitiveType::F16 => Some(ctx.f16_type().vec_type(size)),
            PrimitiveType::F32 => Some(ctx.f32_type().vec_type(size)),
            PrimitiveType::F64 => Some(ctx.f64_type().vec_type(size)),
            PrimitiveType::P64x2 => None,
            PrimitiveType::List(_, _) => None,
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
            DataType::Utf8 => PrimitiveType::P64x2, // string
            DataType::Binary => PrimitiveType::P64x2, // binary
            DataType::LargeUtf8 => PrimitiveType::P64x2, // string view
            DataType::Utf8View => PrimitiveType::P64x2, // string view
            DataType::FixedSizeList(f, l) => PrimitiveType::List(
                PrimitiveType::for_arrow_type(f.data_type())
                    .try_into()
                    .unwrap(),
                *l as usize,
            ),
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
            PrimitiveType::List(t, s) => DataType::FixedSizeList(
                Arc::new(Field::new_list_field(
                    PrimitiveType::from(*t).as_arrow_type(),
                    false,
                )),
                *s as i32,
            ),
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
            PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => true,
            _ => false,
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
            _ => false,
        }
    }

    /// Returns the "best" common value to cast both types to in order to
    /// perform a comparison. Returns `None` if there is no compatible type.
    fn dominant(lhs_prim: PrimitiveType, rhs_prim: PrimitiveType) -> Option<PrimitiveType> {
        if let PrimitiveType::List(lt, ls) = lhs_prim {
            if let PrimitiveType::List(rt, rs) = rhs_prim {
                if lt == rt && ls == rs {
                    return Some(PrimitiveType::List(lt, ls));
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }

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
            PrimitiveType::List(_, _) => ComparisonType::List,
            PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => ComparisonType::Float,
        }
    }

    /// Returns either self or the inner type of a list
    pub fn list_type_into_inner(&self) -> PrimitiveType {
        match self {
            PrimitiveType::List(t, _) => PrimitiveType::from(*t),
            _ => *self,
        }
    }
}

impl From<ListItemType> for PrimitiveType {
    fn from(value: ListItemType) -> Self {
        match value {
            ListItemType::I8 => PrimitiveType::I8,
            ListItemType::I16 => PrimitiveType::I16,
            ListItemType::I32 => PrimitiveType::I32,
            ListItemType::I64 => PrimitiveType::I64,
            ListItemType::U8 => PrimitiveType::U8,
            ListItemType::U16 => PrimitiveType::U16,
            ListItemType::U32 => PrimitiveType::U32,
            ListItemType::U64 => PrimitiveType::U64,
            ListItemType::F16 => PrimitiveType::F16,
            ListItemType::F32 => PrimitiveType::F32,
            ListItemType::F64 => PrimitiveType::F64,
            ListItemType::P64x2 => PrimitiveType::P64x2,
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

struct ArrayDatum(ArrayRef);
impl Datum for ArrayDatum {
    fn get(&self) -> (&dyn Array, bool) {
        (&self.0, false)
    }
}

fn make_ree_unchecked(res: &dyn Array, data: &dyn Array, len: usize) -> ArrayRef {
    let ree_array_type = DataType::RunEndEncoded(
        Arc::new(Field::new(
            "run_ends",
            res.data_type().clone(),
            res.is_nullable(),
        )),
        Arc::new(Field::new(
            "values",
            data.data_type().clone(),
            data.is_nullable(),
        )),
    );
    let builder = ArrayDataBuilder::new(ree_array_type)
        .len(len)
        .add_child_data(res.to_data())
        .add_child_data(data.to_data());

    let array_data = unsafe { builder.build_unchecked() };
    make_array(array_data)
}

fn ree_logical_nulls<K: RunEndIndexType>(
    arr: &dyn Array,
) -> Result<Option<NullBuffer>, ArrowKernelError> {
    let arr = arr.as_run::<K>();
    let res = PrimitiveArray::<K>::new(arr.run_ends().inner().clone(), None);
    if let Some(re_nulls) = arr.values().logical_nulls() {
        let re_nulls = BooleanArray::new(re_nulls.into_inner(), None);
        let re_nulls = make_ree_unchecked(&res, &re_nulls, arr.len());
        let re_nulls = crate::arrow_interface::cast::cast(&re_nulls, &DataType::Boolean)?;
        let re_nulls = re_nulls.as_boolean().clone().into_parts().0;
        let re_nulls = NullBuffer::new(re_nulls);
        Ok(Some(re_nulls))
    } else {
        Ok(None)
    }
}

pub fn logical_nulls(arr: &dyn Array) -> Result<Option<NullBuffer>, ArrowKernelError> {
    match arr.data_type() {
        DataType::Dictionary(_k_dt, _v_dt) => {
            let arr = arr.as_any_dictionary();

            match (arr.keys().is_nullable(), arr.values().is_nullable()) {
                (false, false) => Ok(None),
                (false, true) => {
                    let v_nulls = arr.values().logical_nulls().unwrap();
                    let ba = BooleanArray::new(v_nulls.inner().clone(), None);
                    let nulls = crate::arrow_interface::select::take(&ba, arr.keys())?;
                    let nulls = nulls.as_boolean().clone();
                    let nb = NullBuffer::new(nulls.into_parts().0);
                    Ok(Some(nb))
                }
                (true, false) => Ok(arr.keys().logical_nulls()),
                (true, true) => Ok(arr.logical_nulls()),
            }
        }
        DataType::RunEndEncoded(f_re, _f_val) => match f_re.data_type() {
            DataType::Int16 => ree_logical_nulls::<Int16Type>(arr),
            DataType::Int32 => ree_logical_nulls::<Int32Type>(arr),
            DataType::Int64 => ree_logical_nulls::<Int64Type>(arr),
            _ => unreachable!("invalide run end type"),
        },
        _ => Ok(arr.logical_nulls()),
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use arrow_array::{
        types::Int8Type, Array, DictionaryArray, Int16Array, Int32Array, Int8Array, RunArray,
        StringArray,
    };
    use arrow_buffer::NullBuffer;
    use arrow_data::ArrayDataBuilder;
    use arrow_schema::{DataType, Field};
    use std::sync::Arc;

    fn assert_nullbuffers_equal(a: Option<NullBuffer>, b: Option<NullBuffer>) {
        match (a, b) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                let ab = BooleanArray::new(a.inner().clone(), None);
                let bb = BooleanArray::new(b.inner().clone(), None);
                assert_eq!(ab, bb);
            }
            (a, b) => panic!(
                "Null buffer mismatch: left={:?}, right={:?}",
                a.is_some(),
                b.is_some()
            ),
        }
    }

    #[test]
    fn logical_nulls_dictionary_matches_arrow() {
        // a dictionary with null values but no null keys
        let values = StringArray::from(vec![Some("hello"), None, Some("world")]);
        let keys = Int8Array::from(vec![0, 1, 2, 2, 1, 0]);
        let dict = DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap();

        let arr = &dict as &dyn Array;
        let expected = arr.logical_nulls();
        let actual = super::logical_nulls(arr).unwrap();
        assert_nullbuffers_equal(actual, expected);

        // a dictionary with null values and null keys
        let values = StringArray::from(vec![Some("hello"), None, Some("world")]);
        let keys = Int8Array::from(vec![Some(0), Some(1), Some(2), None, None, Some(0)]);
        let dict = DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap();

        let arr = &dict as &dyn Array;
        let expected = arr.logical_nulls();
        let actual = super::logical_nulls(arr).unwrap();
        assert_nullbuffers_equal(actual, expected);
    }

    #[test]
    fn logical_nulls_run_end_matches_arrow() {
        // Build a RunEndEncoded<Int16, Int32> array with runs:
        // run_ends: [2, 5, 6, 9] => length 9 with runs [0..2], [2..5], [5..6], [6..9]
        // values:   [10, null, 30, null] => validities per run: [T, F, T, F]
        // So overall nulls for positions:
        // 0..1 => valid, 2..4 => null, 5 => valid, 6..8 => null
        let run_ends = Int16Array::from(vec![2i16, 5, 6, 9]);
        let values = Int32Array::from(vec![Some(10), None, Some(30), None]);

        let ree_array_type = DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int16, false)),
            Arc::new(Field::new("values", DataType::Int32, true)),
        );

        let data = unsafe {
            ArrayDataBuilder::new(ree_array_type)
                .len(9)
                .add_child_data(run_ends.to_data())
                .add_child_data(values.to_data())
                .build_unchecked()
        };

        let run_arr: RunArray<Int16Type> = RunArray::from(data);

        let arr = &run_arr as &dyn Array;
        let expected = arr.logical_nulls();
        let actual = super::logical_nulls(arr).unwrap();

        assert_nullbuffers_equal(actual, expected);
    }
}
