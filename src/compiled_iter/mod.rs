mod bitmap;
mod dictionary;
mod fixed_size_list;
mod list;
mod primitive;
mod runend;
mod scalar;
mod setbit;
mod string;
mod view;

use std::ffi::c_void;

use arrow_array::{
    cast::AsArray,
    types::{
        BinaryViewType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
        Int8Type, StringViewType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, BooleanArray, Datum, GenericStringArray, Int16RunArray, Int32RunArray, Int64RunArray,
};

use arrow_schema::DataType;
use bitmap::BitmapIterator;
use dictionary::DictionaryIterator;
use inkwell::{
    builder::Builder,
    context::Context,
    intrinsics::Intrinsic,
    module::{Linkage, Module},
    types::BasicType,
    values::{BasicValue, FunctionValue, IntValue, PointerValue},
    AddressSpace, IntPredicate,
};
use primitive::PrimitiveIterator;
use runend::{add_bsearch, RunEndIterator};
use scalar::{ScalarPrimitiveIterator, ScalarStringIterator};
use setbit::SetBitIterator;
use string::{LargeStringIterator, StringIterator};

use crate::{
    compiled_iter::{
        fixed_size_list::FixedSizeListIterator,
        list::ListIterator,
        scalar::{ScalarBinaryIterator, ScalarBooleanIterator, ScalarVectorIterator},
        view::ViewIterator,
    },
    declare_blocks, increment_pointer, mark_load_invariant, set_noalias_params, ArrowKernelError,
    ListItemType, PrimitiveType,
};

fn build_fixed_size_list_value<'a>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    build: &Builder<'a>,
    dt: &DataType,
    iter: &FixedSizeListIterator,
    iter_ptr: PointerValue<'a>,
    row_idx: IntValue<'a>,
) -> Option<inkwell::values::BasicValueEnum<'a>> {
    let DataType::FixedSizeList(_, list_size) = dt else {
        unreachable!("fixed-size-list iterator with non-list data type {dt:?}");
    };

    let child_access = iter.child().generate_random_access(ctx, llvm_mod)?;
    let child_iter_ptr = iter.llvm_child_iter(ctx, build, iter_ptr);
    let i64_type = ctx.i64_type();
    let list_size_i64 = i64_type.const_int(*list_size as u64, false);
    let base_idx = build
        .build_int_mul(row_idx, list_size_i64, "fsl_child_base")
        .unwrap();
    let ptype = PrimitiveType::for_arrow_type(dt);
    let llvm_type = ptype.llvm_type(ctx);

    match ptype {
        PrimitiveType::List(ListItemType::Boolean, size) => {
            let mut out = llvm_type.into_vector_type().const_zero();
            for lane in 0..size as u64 {
                let child_idx = build
                    .build_int_add(
                        base_idx,
                        i64_type.const_int(lane, false),
                        "fsl_bool_child_idx",
                    )
                    .unwrap();
                let value = build
                    .build_call(
                        child_access,
                        &[child_iter_ptr.into(), child_idx.into()],
                        "fsl_bool_value",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_int_value();
                let value = build
                    .build_int_truncate(value, ctx.bool_type(), "fsl_bool_value_i1")
                    .unwrap();
                out = build
                    .build_insert_element(
                        out,
                        value,
                        i64_type.const_int(lane, false),
                        "fsl_bool_insert",
                    )
                    .unwrap();
            }
            Some(out.as_basic_value_enum())
        }
        PrimitiveType::List(ListItemType::P64x2, _) => {
            let mut out = llvm_type.into_array_type().const_zero();
            for lane in 0..*list_size as u64 {
                let child_idx = build
                    .build_int_add(base_idx, i64_type.const_int(lane, false), "fsl_child_idx")
                    .unwrap();
                let value = build
                    .build_call(
                        child_access,
                        &[child_iter_ptr.into(), child_idx.into()],
                        "fsl_child_value",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic();
                out = build
                    .build_insert_value(out, value, lane as u32, "fsl_insert")
                    .unwrap()
                    .into_array_value();
            }
            Some(out.as_basic_value_enum())
        }
        PrimitiveType::List(item, _) => {
            let inner = PrimitiveType::from(item);
            let mut out = inner.llvm_vec_type(ctx, *list_size as u32)?.const_zero();
            for lane in 0..*list_size as u64 {
                let child_idx = build
                    .build_int_add(base_idx, i64_type.const_int(lane, false), "fsl_child_idx")
                    .unwrap();
                let value = build
                    .build_call(
                        child_access,
                        &[child_iter_ptr.into(), child_idx.into()],
                        "fsl_child_value",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic();
                out = build
                    .build_insert_element(out, value, i64_type.const_int(lane, false), "fsl_insert")
                    .unwrap();
            }
            Some(out.as_basic_value_enum())
        }
        _ => unreachable!("fixed-size-list data type did not map to list primitive type"),
    }
}

fn list_value_llvm_type<'a>(ctx: &'a Context) -> inkwell::types::StructType<'a> {
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    ctx.struct_type(
        &[
            ptr_type.into(),
            ctx.i64_type().into(),
            ctx.i64_type().into(),
        ],
        false,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IteratorValueType {
    Boolean,
    Primitive(PrimitiveType),
    VariableList,
}

impl IteratorValueType {
    fn llvm_type<'a>(self, ctx: &'a Context) -> inkwell::types::BasicTypeEnum<'a> {
        match self {
            IteratorValueType::Boolean => ctx.i8_type().into(),
            IteratorValueType::Primitive(ptype) => ptype.llvm_type(ctx),
            IteratorValueType::VariableList => list_value_llvm_type(ctx).into(),
        }
    }

    fn vectorizable_primitive(self) -> Option<PrimitiveType> {
        match self {
            IteratorValueType::Boolean => None,
            IteratorValueType::Primitive(ptype)
                if !matches!(ptype, PrimitiveType::P64x2 | PrimitiveType::List(_, _)) =>
            {
                Some(ptype)
            }
            IteratorValueType::Primitive(_) | IteratorValueType::VariableList => None,
        }
    }

    fn block_llvm_type<'a>(
        self,
        ctx: &'a Context,
        rows: u32,
    ) -> Option<inkwell::types::BasicTypeEnum<'a>> {
        match self {
            IteratorValueType::Boolean => Some(ctx.bool_type().vec_type(rows).into()),
            IteratorValueType::Primitive(PrimitiveType::List(item, size)) => {
                let lanes = rows.checked_mul(size as u32)?;
                match item {
                    ListItemType::Boolean => Some(ctx.bool_type().vec_type(lanes).into()),
                    ListItemType::P64x2 => None,
                    _ => PrimitiveType::from(item)
                        .llvm_vec_type(ctx, lanes)
                        .map(Into::into),
                }
            }
            IteratorValueType::Primitive(ptype) => ptype.llvm_vec_type(ctx, rows).map(Into::into),
            IteratorValueType::VariableList => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IteratorCodegenInfo {
    value_type: IteratorValueType,
    random_access: bool,
    next_block: bool,
}

fn build_variable_list_value<'a>(
    ctx: &'a Context,
    build: &Builder<'a>,
    iter: &ListIterator,
    iter_ptr: PointerValue<'a>,
    row_idx: IntValue<'a>,
) -> inkwell::values::BasicValueEnum<'a> {
    let i64_type = ctx.i64_type();
    let offsets = iter.llvm_offsets(ctx, build, iter_ptr);
    let start = iter.llvm_offset_at(ctx, build, offsets, row_idx);
    let end = iter.llvm_offset_at(
        ctx,
        build,
        offsets,
        build
            .build_int_add(row_idx, i64_type.const_int(1, false), "list_next_row")
            .unwrap(),
    );
    let child_iter = iter.llvm_child_iter(ctx, build, iter_ptr);
    let row_type = list_value_llvm_type(ctx);
    let row = row_type.const_zero();
    let row = build
        .build_insert_value(row, child_iter, 0, "list_row_child")
        .unwrap();
    let row = build
        .build_insert_value(row, start, 1, "list_row_start")
        .unwrap();
    build
        .build_insert_value(row, end, 2, "list_row_end")
        .unwrap()
        .as_basic_value_enum()
}

pub fn array_to_setbit_iter(arr: &BooleanArray) -> Result<IteratorHolder, ArrowKernelError> {
    Ok(IteratorHolder::SetBit(Box::new(SetBitIterator::from(arr))))
}

/// Convert an Arrow Array into an IteratorHolder, which contains the C-style
/// iterator along with all the other C-style iterator needed. For example, a
/// dictionary array will get a C-style iterator and then two sub-iterators for
/// the keys and values.
fn array_to_iter(arr: &dyn Array) -> IteratorHolder {
    match arr.data_type() {
        arrow_schema::DataType::Null => todo!(),
        arrow_schema::DataType::Boolean => IteratorHolder::Bitmap(arr.as_boolean().into()),
        arrow_schema::DataType::Int8 => {
            IteratorHolder::Primitive(arr.as_primitive::<Int8Type>().into())
        }
        arrow_schema::DataType::Int16 => {
            IteratorHolder::Primitive(arr.as_primitive::<Int16Type>().into())
        }
        arrow_schema::DataType::Int32 => {
            IteratorHolder::Primitive(arr.as_primitive::<Int32Type>().into())
        }
        arrow_schema::DataType::Int64 => {
            IteratorHolder::Primitive(arr.as_primitive::<Int64Type>().into())
        }
        arrow_schema::DataType::UInt8 => {
            IteratorHolder::Primitive(arr.as_primitive::<UInt8Type>().into())
        }
        arrow_schema::DataType::UInt16 => {
            IteratorHolder::Primitive(arr.as_primitive::<UInt16Type>().into())
        }
        arrow_schema::DataType::UInt32 => {
            IteratorHolder::Primitive(arr.as_primitive::<UInt32Type>().into())
        }
        arrow_schema::DataType::UInt64 => {
            IteratorHolder::Primitive(arr.as_primitive::<UInt64Type>().into())
        }
        arrow_schema::DataType::Float16 => {
            IteratorHolder::Primitive(arr.as_primitive::<Float16Type>().into())
        }
        arrow_schema::DataType::Float32 => {
            IteratorHolder::Primitive(arr.as_primitive::<Float32Type>().into())
        }
        arrow_schema::DataType::Float64 => {
            IteratorHolder::Primitive(arr.as_primitive::<Float64Type>().into())
        }
        arrow_schema::DataType::Timestamp(_time_unit, _) => {
            IteratorHolder::Primitive(arr.as_primitive::<Int64Type>().into())
        }
        arrow_schema::DataType::Date32 => todo!(),
        arrow_schema::DataType::Date64 => todo!(),
        arrow_schema::DataType::Time32(_time_unit) => todo!(),
        arrow_schema::DataType::Time64(_time_unit) => todo!(),
        arrow_schema::DataType::Duration(_time_unit) => todo!(),
        arrow_schema::DataType::Interval(_interval_unit) => todo!(),
        arrow_schema::DataType::Binary => IteratorHolder::String(arr.as_binary().into()),
        arrow_schema::DataType::FixedSizeBinary(_) => todo!(),
        arrow_schema::DataType::LargeBinary => {
            IteratorHolder::LargeString(arr.as_binary::<i64>().into())
        }
        arrow_schema::DataType::BinaryView => {
            IteratorHolder::View(arr.as_byte_view::<BinaryViewType>().into())
        }

        arrow_schema::DataType::Utf8 => IteratorHolder::String(arr.as_string().into()),
        arrow_schema::DataType::LargeUtf8 => IteratorHolder::LargeString(
            arr.as_any()
                .downcast_ref::<GenericStringArray<i64>>()
                .unwrap()
                .into(),
        ),
        arrow_schema::DataType::Utf8View => {
            IteratorHolder::View(arr.as_byte_view::<StringViewType>().into())
        }
        arrow_schema::DataType::List(_field) => IteratorHolder::List(arr.as_list::<i32>().into()),
        arrow_schema::DataType::ListView(_field) => todo!(),
        arrow_schema::DataType::FixedSizeList(_field, _) => {
            IteratorHolder::FixedSizeList(arr.as_fixed_size_list().into())
        }
        arrow_schema::DataType::LargeList(_field) => {
            IteratorHolder::List(arr.as_list::<i64>().into())
        }
        arrow_schema::DataType::LargeListView(_field) => todo!(),
        arrow_schema::DataType::Struct(_fields) => todo!(),
        arrow_schema::DataType::Union(_union_fields, _union_mode) => todo!(),
        arrow_schema::DataType::Dictionary(key_type, _value_type) => match key_type.as_ref() {
            arrow_schema::DataType::Int8 => arr.as_dictionary::<Int8Type>().into(),
            arrow_schema::DataType::Int16 => arr.as_dictionary::<Int16Type>().into(),
            arrow_schema::DataType::Int32 => arr.as_dictionary::<Int32Type>().into(),
            arrow_schema::DataType::Int64 => arr.as_dictionary::<Int64Type>().into(),
            arrow_schema::DataType::UInt8 => arr.as_dictionary::<UInt8Type>().into(),
            arrow_schema::DataType::UInt16 => arr.as_dictionary::<UInt16Type>().into(),
            arrow_schema::DataType::UInt32 => arr.as_dictionary::<UInt32Type>().into(),
            arrow_schema::DataType::UInt64 => arr.as_dictionary::<UInt64Type>().into(),
            _ => unreachable!(),
        },
        arrow_schema::DataType::Decimal32(_, _) => todo!(),
        arrow_schema::DataType::Decimal64(_, _) => todo!(),
        arrow_schema::DataType::Decimal128(_, _) => todo!(),
        arrow_schema::DataType::Decimal256(_, _) => todo!(),
        arrow_schema::DataType::Map(_field, _) => todo!(),
        arrow_schema::DataType::RunEndEncoded(re, _values) => match re.data_type() {
            DataType::Int16 => arr.as_any().downcast_ref::<Int16RunArray>().unwrap().into(),
            DataType::Int32 => arr.as_any().downcast_ref::<Int32RunArray>().unwrap().into(),
            DataType::Int64 => arr.as_any().downcast_ref::<Int64RunArray>().unwrap().into(),
            _ => unreachable!("invalid run end type"),
        },
    }
}

pub fn datum_to_iter(val: &dyn Datum) -> Result<IteratorHolder, ArrowKernelError> {
    let (d, is_scalar) = val.get();
    if !is_scalar {
        Ok(array_to_iter(d))
    } else {
        match d.data_type() {
            DataType::Boolean => Ok(d.as_boolean().value(0).into()),
            DataType::Int8 => Ok(d.as_primitive::<Int8Type>().value(0).into()),
            DataType::Int16 => Ok(d.as_primitive::<Int16Type>().value(0).into()),
            DataType::Int32 => Ok(d.as_primitive::<Int32Type>().value(0).into()),
            DataType::Int64 => Ok(d.as_primitive::<Int64Type>().value(0).into()),
            DataType::UInt8 => Ok(d.as_primitive::<UInt8Type>().value(0).into()),
            DataType::UInt16 => Ok(d.as_primitive::<UInt16Type>().value(0).into()),
            DataType::UInt32 => Ok(d.as_primitive::<UInt32Type>().value(0).into()),
            DataType::UInt64 => Ok(d.as_primitive::<UInt64Type>().value(0).into()),
            DataType::Float16 => Ok(d.as_primitive::<Float16Type>().value(0).into()),
            DataType::Float32 => Ok(d.as_primitive::<Float32Type>().value(0).into()),
            DataType::Float64 => Ok(d.as_primitive::<Float64Type>().value(0).into()),
            DataType::Timestamp(_time_unit, _) => todo!(),
            DataType::Date32 => todo!(),
            DataType::Date64 => todo!(),
            DataType::Time32(_time_unit) => todo!(),
            DataType::Time64(__time_unit) => todo!(),
            DataType::Duration(_time_unit) => todo!(),
            DataType::Interval(_interval_unit) => todo!(),
            DataType::Binary => Ok(IteratorHolder::ScalarBinary(ScalarBinaryIterator::new(
                d.as_binary::<i32>().value(0).to_owned().into_boxed_slice(),
                DataType::Binary,
            ))),
            DataType::FixedSizeBinary(_) => todo!(),
            DataType::LargeBinary => Ok(IteratorHolder::ScalarBinary(ScalarBinaryIterator::new(
                d.as_binary::<i64>().value(0).to_owned().into_boxed_slice(),
                DataType::LargeBinary,
            ))),
            DataType::BinaryView => todo!(),
            DataType::Utf8 => Ok(IteratorHolder::ScalarString(ScalarStringIterator::new(
                d.as_string::<i32>().value(0).to_owned().into_boxed_str(),
                DataType::Utf8,
            ))),
            DataType::LargeUtf8 => Ok(IteratorHolder::ScalarString(ScalarStringIterator::new(
                d.as_string::<i64>().value(0).to_owned().into_boxed_str(),
                DataType::LargeUtf8,
            ))),
            DataType::Utf8View => Ok(IteratorHolder::ScalarString(ScalarStringIterator::new(
                d.as_string_view().value(0).to_owned().into_boxed_str(),
                DataType::Utf8View,
            ))),
            DataType::List(_field) => todo!(),
            DataType::ListView(_field) => todo!(),
            DataType::FixedSizeList(_, _) => {
                let value = d.as_fixed_size_list().value(0);
                match value.data_type() {
                    DataType::Int8 => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_primitive(
                            value.as_primitive::<Int8Type>(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::Int16 => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_primitive(
                            value.as_primitive::<Int16Type>(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::Int32 => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_primitive(
                            value.as_primitive::<Int32Type>(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::Int64 => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_primitive(
                            value.as_primitive::<Int64Type>(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::UInt8 => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_primitive(
                            value.as_primitive::<UInt8Type>(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::UInt16 => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_primitive(
                            value.as_primitive::<UInt16Type>(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::UInt32 => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_primitive(
                            value.as_primitive::<UInt32Type>(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::UInt64 => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_primitive(
                            value.as_primitive::<UInt64Type>(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::Float16 => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_primitive(
                            value.as_primitive::<Float16Type>(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::Float32 => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_primitive(
                            value.as_primitive::<Float32Type>(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::Float64 => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_primitive(
                            value.as_primitive::<Float64Type>(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::Boolean => Ok(IteratorHolder::ScalarVec(
                        ScalarVectorIterator::from_boolean(
                            value.as_boolean(),
                            d.data_type().clone(),
                        ),
                    )),
                    DataType::Utf8 => {
                        let arr = value.as_string::<i32>();
                        let ptrs = (0..arr.len())
                            .map(|idx| {
                                let value = arr.value(idx);
                                let start = value.as_ptr() as u128;
                                let end = value.as_ptr().wrapping_add(value.len()) as u128;
                                start | (end << 64)
                            })
                            .collect::<Vec<_>>();
                        Ok(IteratorHolder::ScalarVec(
                            ScalarVectorIterator::from_pointer_pairs(
                                ListItemType::P64x2,
                                ptrs,
                                value.clone(),
                                d.data_type().clone(),
                            ),
                        ))
                    }
                    DataType::LargeUtf8 => {
                        let arr = value.as_string::<i64>();
                        let ptrs = (0..arr.len())
                            .map(|idx| {
                                let value = arr.value(idx);
                                let start = value.as_ptr() as u128;
                                let end = value.as_ptr().wrapping_add(value.len()) as u128;
                                start | (end << 64)
                            })
                            .collect::<Vec<_>>();
                        Ok(IteratorHolder::ScalarVec(
                            ScalarVectorIterator::from_pointer_pairs(
                                ListItemType::P64x2,
                                ptrs,
                                value.clone(),
                                d.data_type().clone(),
                            ),
                        ))
                    }
                    DataType::Utf8View => {
                        let arr = value.as_string_view();
                        let ptrs = (0..arr.len())
                            .map(|idx| {
                                let value = arr.value(idx);
                                let start = value.as_ptr() as u128;
                                let end = value.as_ptr().wrapping_add(value.len()) as u128;
                                start | (end << 64)
                            })
                            .collect::<Vec<_>>();
                        Ok(IteratorHolder::ScalarVec(
                            ScalarVectorIterator::from_pointer_pairs(
                                ListItemType::P64x2,
                                ptrs,
                                value.clone(),
                                d.data_type().clone(),
                            ),
                        ))
                    }
                    _ => todo!(),
                }
            }
            DataType::LargeList(_field) => todo!(),
            DataType::LargeListView(_field) => todo!(),
            DataType::Struct(_fields) => todo!(),
            DataType::Union(_union_fields, _union_mode) => todo!(),
            DataType::Decimal128(_, _) => todo!(),
            DataType::Decimal256(_, _) => todo!(),
            DataType::Map(_field, _) => todo!(),
            _ => Err(ArrowKernelError::UnsupportedScalar(d.data_type().clone())),
        }
    }
}

/// A holder for a C-style iterator. Created by `array_to_iter`, and used by
/// compiled LLVM functions.
#[derive(Debug)]
pub enum IteratorHolder {
    Primitive(Box<PrimitiveIterator>),
    String(Box<StringIterator>),
    LargeString(Box<LargeStringIterator>),
    View(Box<ViewIterator>),
    Bitmap(Box<BitmapIterator>),
    SetBit(Box<SetBitIterator>),
    Dictionary {
        arr: Box<DictionaryIterator>,
        keys: Box<IteratorHolder>,
        values: Box<IteratorHolder>,
    },
    RunEnd {
        arr: Box<RunEndIterator>,
        run_ends: Box<IteratorHolder>,
        values: Box<IteratorHolder>,
    },
    FixedSizeList(Box<FixedSizeListIterator>),
    List(Box<ListIterator>),
    ScalarPrimitive(Box<ScalarPrimitiveIterator>),
    ScalarBoolean(Box<ScalarBooleanIterator>),
    ScalarString(Box<ScalarStringIterator>),
    ScalarBinary(Box<ScalarBinaryIterator>),
    ScalarVec(Box<ScalarVectorIterator>),
}

impl IteratorHolder {
    /// Returns the Arrow data type represented by this iterator.
    pub fn data_type(&self) -> DataType {
        match self {
            IteratorHolder::Primitive(iter) => iter.array_ref.data_type().clone(),
            IteratorHolder::String(iter) => iter.array_ref.data_type().clone(),
            IteratorHolder::LargeString(iter) => iter.array_ref.data_type().clone(),
            IteratorHolder::View(iter) => iter.array_ref.data_type().clone(),
            IteratorHolder::Bitmap(_) => DataType::Boolean,
            IteratorHolder::SetBit(_) => DataType::Boolean,
            IteratorHolder::Dictionary { arr, .. } => arr.array_ref.data_type().clone(),
            IteratorHolder::RunEnd { arr, .. } => arr.array_ref.data_type().clone(),
            IteratorHolder::FixedSizeList(iter) => iter.array_ref.data_type().clone(),
            IteratorHolder::List(iter) => iter.array_ref.data_type().clone(),
            IteratorHolder::ScalarPrimitive(iter) => iter.ptype.as_arrow_type(),
            IteratorHolder::ScalarBoolean(_) => DataType::Boolean,
            IteratorHolder::ScalarString(iter) => iter.data_type.clone(),
            IteratorHolder::ScalarBinary(iter) => iter.data_type.clone(),
            IteratorHolder::ScalarVec(iter) => iter.data_type.clone(),
        }
    }

    /// Gets a mutable pointer to the inner iterator, suitable for passing to
    /// compiled LLVM functions.
    pub fn get_mut_ptr(&mut self) -> *mut c_void {
        match self {
            IteratorHolder::Primitive(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::String(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::LargeString(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::View(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::Bitmap(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::SetBit(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::Dictionary { arr: iter, .. } => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::RunEnd { arr: iter, .. } => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::FixedSizeList(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::List(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::ScalarPrimitive(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::ScalarBoolean(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::ScalarString(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::ScalarBinary(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::ScalarVec(iter) => &mut **iter as *mut _ as *mut c_void,
        }
    }

    pub fn as_primitive(&self) -> &PrimitiveIterator {
        match self {
            IteratorHolder::Primitive(iter) => iter,
            _ => panic!("as_primitive called on non-primitive iterator holder"),
        }
    }

    /// Gets a const pointer to the inner iterator, suitable for passing to
    /// compiled LLVM functions.
    pub fn get_ptr(&self) -> *const c_void {
        match self {
            IteratorHolder::Primitive(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::String(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::LargeString(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::View(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::Bitmap(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::SetBit(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::Dictionary { arr: iter, .. } => &**iter as *const _ as *const c_void,
            IteratorHolder::RunEnd { arr: iter, .. } => &**iter as *const _ as *const c_void,
            IteratorHolder::FixedSizeList(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::List(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::ScalarPrimitive(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::ScalarBoolean(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::ScalarString(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::ScalarBinary(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::ScalarVec(iter) => &**iter as *const _ as *const c_void,
        }
    }

    /// Potentially copy the iterator into the current stack frame, allowing
    /// better LLVM optimizations.
    pub fn localize_struct<'a>(
        &self,
        ctx: &'a Context,
        b: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        match self {
            IteratorHolder::Primitive(i) => i.localize_struct(ctx, b, ptr),
            IteratorHolder::FixedSizeList(i) => i.localize_struct(ctx, b, ptr),
            _ => ptr,
        }
    }

    fn codegen_info(&self) -> IteratorCodegenInfo {
        match self {
            IteratorHolder::Primitive(_) => {
                let value_type =
                    IteratorValueType::Primitive(PrimitiveType::for_arrow_type(&self.data_type()));
                IteratorCodegenInfo {
                    value_type,
                    random_access: true,
                    next_block: value_type.vectorizable_primitive().is_some(),
                }
            }
            IteratorHolder::String(_)
            | IteratorHolder::LargeString(_)
            | IteratorHolder::View(_) => IteratorCodegenInfo {
                value_type: IteratorValueType::Primitive(PrimitiveType::P64x2),
                random_access: true,
                next_block: false,
            },
            IteratorHolder::Bitmap(_) => IteratorCodegenInfo {
                value_type: IteratorValueType::Boolean,
                random_access: true,
                next_block: true,
            },
            IteratorHolder::SetBit(_) => IteratorCodegenInfo {
                value_type: IteratorValueType::Primitive(PrimitiveType::U64),
                random_access: false,
                next_block: false,
            },
            IteratorHolder::Dictionary { keys, values, .. } => {
                let keys = keys.codegen_info();
                let values = values.codegen_info();
                IteratorCodegenInfo {
                    value_type: values.value_type,
                    random_access: keys.random_access && values.random_access,
                    next_block: keys.next_block
                        && values.random_access
                        && values.value_type.vectorizable_primitive().is_some(),
                }
            }
            IteratorHolder::RunEnd {
                run_ends, values, ..
            } => {
                let run_ends = run_ends.codegen_info();
                let values = values.codegen_info();
                IteratorCodegenInfo {
                    value_type: values.value_type,
                    random_access: run_ends.random_access && values.random_access,
                    next_block: run_ends.random_access
                        && values.random_access
                        && values.value_type.vectorizable_primitive().is_some(),
                }
            }
            IteratorHolder::FixedSizeList(iter) => IteratorCodegenInfo {
                value_type: IteratorValueType::Primitive(iter.ptype()),
                random_access: iter.child().codegen_info().random_access,
                next_block: iter.child().codegen_info().random_access
                    && !matches!(iter.ptype(), PrimitiveType::List(ListItemType::P64x2, _)),
            },
            IteratorHolder::List(_) => IteratorCodegenInfo {
                value_type: IteratorValueType::VariableList,
                random_access: true,
                next_block: false,
            },
            IteratorHolder::ScalarPrimitive(iter) => {
                let value_type = IteratorValueType::Primitive(iter.ptype);
                IteratorCodegenInfo {
                    value_type,
                    random_access: false,
                    next_block: value_type.vectorizable_primitive().is_some(),
                }
            }
            IteratorHolder::ScalarBoolean(_) => IteratorCodegenInfo {
                value_type: IteratorValueType::Primitive(PrimitiveType::U8),
                random_access: false,
                next_block: false,
            },
            IteratorHolder::ScalarString(_) | IteratorHolder::ScalarBinary(_) => {
                IteratorCodegenInfo {
                    value_type: IteratorValueType::Primitive(PrimitiveType::P64x2),
                    random_access: false,
                    next_block: false,
                }
            }
            IteratorHolder::ScalarVec(iter) => IteratorCodegenInfo {
                value_type: IteratorValueType::Primitive(iter.ptype()),
                random_access: true,
                next_block: false,
            },
        }
    }

    fn primitive_codegen_name(ptype: PrimitiveType) -> String {
        match ptype {
            PrimitiveType::List(item_type, size) => {
                let item_type = match item_type {
                    ListItemType::Boolean => "boolean",
                    ListItemType::I8 => "i8",
                    ListItemType::I16 => "i16",
                    ListItemType::I32 => "i32",
                    ListItemType::I64 => "i64",
                    ListItemType::U8 => "u8",
                    ListItemType::U16 => "u16",
                    ListItemType::U32 => "u32",
                    ListItemType::U64 => "u64",
                    ListItemType::F16 => "f16",
                    ListItemType::F32 => "f32",
                    ListItemType::F64 => "f64",
                    ListItemType::P64x2 => "p64x2",
                };
                format!("list_{item_type}_{size}")
            }
            _ => ptype.to_string().to_ascii_lowercase(),
        }
    }

    fn codegen_shape_name(&self) -> String {
        match self {
            IteratorHolder::Primitive(_) => format!(
                "primitive_{}",
                Self::primitive_codegen_name(PrimitiveType::for_arrow_type(&self.data_type()))
            ),
            IteratorHolder::String(_) => "string32".to_owned(),
            IteratorHolder::LargeString(_) => "string64".to_owned(),
            IteratorHolder::View(_) => "view".to_owned(),
            IteratorHolder::Bitmap(_) => "bitmap".to_owned(),
            IteratorHolder::SetBit(_) => "set_bit".to_owned(),
            IteratorHolder::Dictionary { keys, values, .. } => {
                let keys = keys.codegen_shape_name();
                let values = values.codegen_shape_name();
                format!(
                    "dictionary_k{}_{}_v{}_{}",
                    keys.len(),
                    keys,
                    values.len(),
                    values
                )
            }
            IteratorHolder::RunEnd {
                run_ends, values, ..
            } => {
                let run_ends = run_ends.codegen_shape_name();
                let values = values.codegen_shape_name();
                format!(
                    "run_end_r{}_{}_v{}_{}",
                    run_ends.len(),
                    run_ends,
                    values.len(),
                    values
                )
            }
            IteratorHolder::FixedSizeList(iter) => {
                let child = iter.child().codegen_shape_name();
                format!(
                    "fixed_size_{}_c{}_{}",
                    Self::primitive_codegen_name(iter.ptype()),
                    child.len(),
                    child
                )
            }
            IteratorHolder::List(iter) => {
                let child = iter.child().codegen_shape_name();
                format!("list_w{}_c{}_{}", iter.offset_width(), child.len(), child)
            }
            IteratorHolder::ScalarPrimitive(iter) => format!(
                "scalar_primitive_{}",
                Self::primitive_codegen_name(iter.ptype)
            ),
            IteratorHolder::ScalarBoolean(_) => "scalar_boolean".to_owned(),
            IteratorHolder::ScalarString(_) => "scalar_string".to_owned(),
            IteratorHolder::ScalarBinary(_) => "scalar_binary".to_owned(),
            IteratorHolder::ScalarVec(iter) => format!(
                "scalar_vector_{}",
                Self::primitive_codegen_name(iter.ptype())
            ),
        }
    }

    fn codegen_label(&self) -> String {
        format!("iterator_{}", self.codegen_shape_name())
    }

    /// Adds or reuses the reset function for this iterator's code-generation shape.
    ///
    /// The generated function's signature is:
    ///
    /// `fn reset(iter: ptr)`
    pub fn generate_reset<'a>(&self, ctx: &'a Context, llvm_mod: &Module<'a>) -> FunctionValue<'a> {
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let name = format!("{}_reset", self.codegen_label());

        if let Some(existing) = llvm_mod.get_function(&name) {
            assert_eq!(
                existing.get_type(),
                ctx.void_type().fn_type(&[ptr_type.into()], false)
            );
            return existing;
        }

        let fn_type = ctx.void_type().fn_type(&[ptr_type.into()], false);
        let reset = llvm_mod.add_function(
            &name,
            fn_type,
            Some(
                #[cfg(test)]
                Linkage::External,
                #[cfg(not(test))]
                Linkage::Private,
            ),
        );
        set_noalias_params(&reset);
        let iter_ptr = reset.get_nth_param(0).unwrap().into_pointer_value();

        match self {
            IteratorHolder::Primitive(iter) => {
                declare_blocks!(ctx, reset, entry);
                build.position_at_end(entry);
                iter.llvm_reset(ctx, &build, iter_ptr);
                build.build_return(None).unwrap();
                reset
            }
            IteratorHolder::String(iter) => {
                declare_blocks!(ctx, reset, entry);
                build.position_at_end(entry);
                iter.llvm_reset(ctx, &build, iter_ptr);
                build.build_return(None).unwrap();
                reset
            }
            IteratorHolder::LargeString(iter) => {
                declare_blocks!(ctx, reset, entry);
                build.position_at_end(entry);
                iter.llvm_reset(ctx, &build, iter_ptr);
                build.build_return(None).unwrap();
                reset
            }
            IteratorHolder::View(iter) => {
                declare_blocks!(ctx, reset, entry);
                build.position_at_end(entry);
                iter.llvm_reset(ctx, &build, iter_ptr);
                build.build_return(None).unwrap();
                reset
            }
            IteratorHolder::Bitmap(iter) => {
                declare_blocks!(ctx, reset, entry);
                build.position_at_end(entry);
                iter.llvm_reset(ctx, &build, iter_ptr);
                build.build_return(None).unwrap();
                reset
            }
            IteratorHolder::SetBit(iter) => {
                declare_blocks!(ctx, reset, entry);
                build.position_at_end(entry);
                iter.llvm_reset(ctx, &build, iter_ptr);
                build.build_return(None).unwrap();
                reset
            }
            IteratorHolder::Dictionary { arr, keys, values } => {
                let key_reset = keys.generate_reset(ctx, llvm_mod);
                let value_reset = values.generate_reset(ctx, llvm_mod);

                declare_blocks!(ctx, reset, entry);
                build.position_at_end(entry);
                let key_ptr = arr.llvm_key_iter_ptr(ctx, &build, iter_ptr);
                let val_ptr = arr.llvm_val_iter_ptr(ctx, &build, iter_ptr);
                build
                    .build_call(key_reset, &[key_ptr.into()], "reset_key_iter")
                    .unwrap();
                build
                    .build_call(value_reset, &[val_ptr.into()], "reset_value_iter")
                    .unwrap();
                build.build_return(None).unwrap();
                reset
            }
            IteratorHolder::RunEnd {
                arr,
                run_ends,
                values,
            } => {
                let re_reset = run_ends.generate_reset(ctx, llvm_mod);
                let value_reset = values.generate_reset(ctx, llvm_mod);

                declare_blocks!(ctx, reset, entry);
                build.position_at_end(entry);
                let re_ptr = arr.llvm_re_iter_ptr(ctx, &build, iter_ptr);
                let val_ptr = arr.llvm_val_iter_ptr(ctx, &build, iter_ptr);
                build
                    .build_call(re_reset, &[re_ptr.into()], "reset_run_end_iter")
                    .unwrap();
                build
                    .build_call(value_reset, &[val_ptr.into()], "reset_value_iter")
                    .unwrap();
                arr.llvm_reset(ctx, &build, iter_ptr);
                build.build_return(None).unwrap();
                reset
            }
            IteratorHolder::FixedSizeList(iter) => {
                declare_blocks!(ctx, reset, entry);
                build.position_at_end(entry);
                iter.llvm_reset(ctx, &build, iter_ptr);
                build.build_return(None).unwrap();
                reset
            }
            IteratorHolder::List(iter) => {
                let child_reset = iter.child().generate_reset(ctx, llvm_mod);

                declare_blocks!(ctx, reset, entry);
                build.position_at_end(entry);
                let child_ptr = iter.llvm_child_iter(ctx, &build, iter_ptr);
                build
                    .build_call(child_reset, &[child_ptr.into()], "reset_list_child_iter")
                    .unwrap();
                iter.llvm_reset(ctx, &build, iter_ptr);
                build.build_return(None).unwrap();
                reset
            }
            IteratorHolder::ScalarPrimitive(_)
            | IteratorHolder::ScalarBoolean(_)
            | IteratorHolder::ScalarString(_)
            | IteratorHolder::ScalarBinary(_)
            | IteratorHolder::ScalarVec(_) => {
                declare_blocks!(ctx, reset, entry);
                build.position_at_end(entry);
                build.build_return(None).unwrap();
                reset
            }
        }
    }

    pub fn generate_random_access_block<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        rows: u32,
    ) -> Option<FunctionValue<'a>> {
        let vec_type = self.codegen_info().value_type.block_llvm_type(ctx, rows)?;
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();
        let fn_type = ctx
            .void_type()
            .fn_type(&[ptr_type.into(), i64_type.into(), ptr_type.into()], false);
        let name = format!("{}_access_block_{}", self.codegen_label(), rows);
        if let Some(existing) = llvm_mod.get_function(&name) {
            return Some(existing);
        }

        let access = llvm_mod.add_function(&name, fn_type, Some(Linkage::Private));
        let iter_ptr = access.get_nth_param(0).unwrap().into_pointer_value();
        let idx = access.get_nth_param(1).unwrap().into_int_value();
        let out_ptr = access.get_nth_param(2).unwrap().into_pointer_value();
        let build = ctx.create_builder();

        match self {
            IteratorHolder::Primitive(iter) => {
                declare_blocks!(ctx, access, entry);
                build.position_at_end(entry);
                let ptype = PrimitiveType::for_arrow_type(&self.data_type());
                let data_ptr = iter.llvm_data(ctx, &build, iter_ptr);
                let data_ptr = increment_pointer!(ctx, build, data_ptr, ptype.width(), idx);
                let value = build.build_load(vec_type, data_ptr, "block").unwrap();
                value
                    .as_instruction_value()
                    .unwrap()
                    .set_alignment(1)
                    .unwrap();
                build.build_store(out_ptr, value).unwrap();
                build.build_return(None).unwrap();
            }
            IteratorHolder::Bitmap(iter) => {
                let bit_count = rows;
                let aligned_bytes = bit_count.div_ceil(8);
                let aligned_width = aligned_bytes * 8;
                let aligned_type = ctx.custom_width_int_type(aligned_width);
                let packed_type = ctx.custom_width_int_type(bit_count);
                let unaligned_type = ctx.custom_width_int_type(aligned_width + 8);
                declare_blocks!(ctx, access, entry, aligned, unaligned, merge);

                build.position_at_end(entry);
                let data = iter.llvm_get_data_ptr(ctx, &build, iter_ptr);
                let bit_idx = build
                    .build_int_add(
                        iter.llvm_slice_offset(ctx, &build, iter_ptr),
                        idx,
                        "bit_idx",
                    )
                    .unwrap();
                let byte_idx = build
                    .build_right_shift(bit_idx, i64_type.const_int(3, false), false, "byte_idx")
                    .unwrap();
                let shift = build
                    .build_and(bit_idx, i64_type.const_int(7, false), "bit_shift")
                    .unwrap();
                let src = increment_pointer!(ctx, build, data, 1, byte_idx);
                let is_aligned = build
                    .build_int_compare(IntPredicate::EQ, shift, i64_type.const_zero(), "aligned")
                    .unwrap();
                build
                    .build_conditional_branch(is_aligned, aligned, unaligned)
                    .unwrap();

                build.position_at_end(aligned);
                let aligned_value = build
                    .build_load(aligned_type, src, "aligned_bits")
                    .unwrap()
                    .into_int_value();
                aligned_value
                    .as_instruction_value()
                    .unwrap()
                    .set_alignment(1)
                    .unwrap();
                let aligned_value = if aligned_width == bit_count {
                    aligned_value
                } else {
                    build
                        .build_int_truncate(aligned_value, packed_type, "aligned_packed")
                        .unwrap()
                };
                build.build_unconditional_branch(merge).unwrap();
                let aligned_end = build.get_insert_block().unwrap();

                build.position_at_end(unaligned);
                let unaligned_value = build
                    .build_load(unaligned_type, src, "unaligned_bits")
                    .unwrap()
                    .into_int_value();
                unaligned_value
                    .as_instruction_value()
                    .unwrap()
                    .set_alignment(1)
                    .unwrap();
                let wide_shift = build
                    .build_int_cast(shift, unaligned_type, "wide_shift")
                    .unwrap();
                let unaligned_value = build
                    .build_right_shift(unaligned_value, wide_shift, false, "shifted_bits")
                    .unwrap();
                let unaligned_value = build
                    .build_int_truncate(unaligned_value, packed_type, "unaligned_packed")
                    .unwrap();
                build.build_unconditional_branch(merge).unwrap();
                let unaligned_end = build.get_insert_block().unwrap();

                build.position_at_end(merge);
                let phi = build.build_phi(packed_type, "packed_bits").unwrap();
                phi.add_incoming(&[
                    (&aligned_value, aligned_end),
                    (&unaligned_value, unaligned_end),
                ]);
                let values = build
                    .build_bit_cast(phi.as_basic_value(), vec_type, "boolean_block")
                    .unwrap();
                build.build_store(out_ptr, values).unwrap();
                build.build_return(None).unwrap();
            }
            IteratorHolder::FixedSizeList(iter) => {
                let PrimitiveType::List(_, list_size) = iter.ptype() else {
                    unreachable!()
                };
                let child_rows = rows.checked_mul(list_size as u32)?;
                let child_access = iter
                    .child()
                    .generate_random_access_block(ctx, llvm_mod, child_rows)?;
                declare_blocks!(ctx, access, entry);
                build.position_at_end(entry);
                let child_idx = build
                    .build_int_mul(
                        idx,
                        i64_type.const_int(list_size as u64, false),
                        "fsl_block_child_idx",
                    )
                    .unwrap();
                let child_ptr = iter.llvm_child_iter(ctx, &build, iter_ptr);
                build
                    .build_call(
                        child_access,
                        &[child_ptr.into(), child_idx.into(), out_ptr.into()],
                        "fsl_child_block",
                    )
                    .unwrap();
                build.build_return(None).unwrap();
            }
            _ => return None,
        }
        Some(access)
    }

    pub fn generate_gather_block<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        rows: u32,
    ) -> Option<FunctionValue<'a>> {
        let output_type = self.codegen_info().value_type.block_llvm_type(ctx, rows)?;
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();
        let index_type = i64_type.vec_type(rows);
        let fn_type = ctx.void_type().fn_type(
            &[ptr_type.into(), index_type.into(), ptr_type.into()],
            false,
        );
        let name = format!("{}_gather_block_{}", self.codegen_label(), rows);
        if let Some(existing) = llvm_mod.get_function(&name) {
            return Some(existing);
        }
        let gather = llvm_mod.add_function(&name, fn_type, Some(Linkage::Private));
        let iter_ptr = gather.get_nth_param(0).unwrap().into_pointer_value();
        let indices = gather.get_nth_param(1).unwrap().into_vector_value();
        let out_ptr = gather.get_nth_param(2).unwrap().into_pointer_value();
        let build = ctx.create_builder();
        declare_blocks!(ctx, gather, entry);
        build.position_at_end(entry);

        match self {
            IteratorHolder::FixedSizeList(iter) => {
                let PrimitiveType::List(item, list_size) = iter.ptype() else {
                    unreachable!()
                };
                let row_access = self.generate_random_access_block(ctx, llvm_mod, 1)?;
                let row_type = self.codegen_info().value_type.block_llvm_type(ctx, 1)?;
                let row_buf = build.build_alloca(row_type, "gather_row").unwrap();
                match item {
                    ListItemType::Boolean => {
                        let output_vec = output_type.into_vector_type();
                        let row_vec = row_type.into_vector_type();
                        let output_int = ctx.custom_width_int_type(output_vec.get_size());
                        let row_int = ctx.custom_width_int_type(row_vec.get_size());
                        let mut output = output_int.const_zero();
                        for row in 0..rows {
                            let idx = build
                                .build_extract_element(
                                    indices,
                                    i64_type.const_int(row as u64, false),
                                    "gather_idx",
                                )
                                .unwrap();
                            build
                                .build_call(
                                    row_access,
                                    &[iter_ptr.into(), idx.into(), row_buf.into()],
                                    "gather_fsl_row",
                                )
                                .unwrap();
                            let value = build
                                .build_load(row_vec, row_buf, "gather_boolean_row")
                                .unwrap();
                            let value = build
                                .build_bit_cast(value, row_int, "gather_boolean_row_packed")
                                .unwrap()
                                .into_int_value();
                            let value = build
                                .build_int_z_extend(value, output_int, "gather_boolean_row_wide")
                                .unwrap();
                            let value = build
                                .build_left_shift(
                                    value,
                                    output_int.const_int((row as usize * list_size) as u64, false),
                                    "gather_boolean_row_shifted",
                                )
                                .unwrap();
                            output = build
                                .build_or(output, value, "gather_boolean_rows")
                                .unwrap();
                        }
                        let output = build
                            .build_bit_cast(output, output_vec, "gather_boolean_block")
                            .unwrap();
                        build.build_store(out_ptr, output).unwrap();
                    }
                    ListItemType::P64x2 => return None,
                    _ => {
                        let output_vec = output_type.into_vector_type();
                        let row_vec = row_type.into_vector_type();
                        let mut output = output_vec.const_zero();
                        for row in 0..rows {
                            let idx = build
                                .build_extract_element(
                                    indices,
                                    i64_type.const_int(row as u64, false),
                                    "gather_idx",
                                )
                                .unwrap();
                            build
                                .build_call(
                                    row_access,
                                    &[iter_ptr.into(), idx.into(), row_buf.into()],
                                    "gather_fsl_row",
                                )
                                .unwrap();
                            let value = build
                                .build_load(row_vec, row_buf, "gather_numeric_row")
                                .unwrap()
                                .into_vector_value();
                            for child in 0..list_size as u32 {
                                let lane = build
                                    .build_extract_element(
                                        value,
                                        i64_type.const_int(child as u64, false),
                                        "gather_child",
                                    )
                                    .unwrap();
                                output = build
                                    .build_insert_element(
                                        output,
                                        lane,
                                        i64_type.const_int(
                                            (row as usize * list_size + child as usize) as u64,
                                            false,
                                        ),
                                        "gather_insert_child",
                                    )
                                    .unwrap();
                            }
                        }
                        build.build_store(out_ptr, output).unwrap();
                    }
                }
            }
            IteratorHolder::Bitmap(_) => {
                let scalar_access = self.generate_random_access(ctx, llvm_mod)?;
                let output_vec = output_type.into_vector_type();
                let output_int = ctx.custom_width_int_type(rows);
                let mut output = output_int.const_zero();
                for row in 0..rows {
                    let idx = build
                        .build_extract_element(
                            indices,
                            i64_type.const_int(row as u64, false),
                            "gather_idx",
                        )
                        .unwrap();
                    let value = build
                        .build_call(
                            scalar_access,
                            &[iter_ptr.into(), idx.into()],
                            "gather_boolean",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                        .into_int_value();
                    let value = build
                        .build_int_truncate(value, ctx.bool_type(), "gather_boolean_i1")
                        .unwrap();
                    let value = build
                        .build_int_z_extend(value, output_int, "gather_boolean_wide")
                        .unwrap();
                    let value = build
                        .build_left_shift(
                            value,
                            output_int.const_int(row as u64, false),
                            "gather_boolean_shifted",
                        )
                        .unwrap();
                    output = build.build_or(output, value, "gather_booleans").unwrap();
                }
                let output = build
                    .build_bit_cast(output, output_vec, "gather_boolean_block")
                    .unwrap();
                build.build_store(out_ptr, output).unwrap();
            }
            _ => {
                let scalar_access = self.generate_random_access(ctx, llvm_mod)?;
                let output_vec = output_type.into_vector_type();
                let mut output = output_vec.const_zero();
                for row in 0..rows {
                    let idx = build
                        .build_extract_element(
                            indices,
                            i64_type.const_int(row as u64, false),
                            "gather_idx",
                        )
                        .unwrap();
                    let value = build
                        .build_call(
                            scalar_access,
                            &[iter_ptr.into(), idx.into()],
                            "gather_value",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic();
                    output = build
                        .build_insert_element(
                            output,
                            value,
                            i64_type.const_int(row as u64, false),
                            "gather_insert",
                        )
                        .unwrap();
                }
                build.build_store(out_ptr, output).unwrap();
            }
        }
        build.build_return(None).unwrap();
        Some(gather)
    }

    /// Adds or reuses the block-next function for this iterator's code-generation shape.
    ///
    /// The generated function fetches `rows` elements and advances the iterator.
    /// Its signature is:
    ///
    /// `fn next_block(iter: ptr, out: ptr_to_vec_of_size_n) -> bool`
    pub fn generate_next_block<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        n: u32,
    ) -> Option<FunctionValue<'a>> {
        let build = ctx.create_builder();
        let info = self.codegen_info();
        if !info.next_block {
            return None;
        }
        let dt = self.data_type();
        let vec_type = info.value_type.block_llvm_type(ctx, n)?;
        let bool_type = ctx.bool_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();
        let llvm_n = i64_type.const_int(n as u64, false);

        let fn_type = bool_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let name = format!("{}_next_block_{}", self.codegen_label(), n);
        if let Some(existing) = llvm_mod.get_function(&name) {
            assert_eq!(existing.get_type(), fn_type);
            return Some(existing);
        }
        let next = llvm_mod.add_function(
            &name,
            fn_type,
            Some(
                #[cfg(test)]
                Linkage::External,
                #[cfg(not(test))]
                Linkage::Private,
            ),
        );
        let iter_ptr = next.get_nth_param(0).unwrap().into_pointer_value();
        let out_ptr = next.get_nth_param(1).unwrap().into_pointer_value();

        let res = match self {
            IteratorHolder::Primitive(primitive_iter) => {
                let ptype = PrimitiveType::for_arrow_type(&dt);
                declare_blocks!(ctx, next, entry, none_left, get_next);

                build.position_at_end(entry);
                let curr_pos = primitive_iter.llvm_pos(ctx, &build, iter_ptr);
                let curr_len = primitive_iter.llvm_len(ctx, &build, iter_ptr);

                let remaining = build
                    .build_int_sub(curr_len, curr_pos, "remaining")
                    .unwrap();
                let have_enough = build
                    .build_int_compare(IntPredicate::UGE, remaining, llvm_n, "have_enough")
                    .unwrap();
                build
                    .build_conditional_branch(have_enough, get_next, none_left)
                    .unwrap();

                build.position_at_end(none_left);
                build
                    .build_return(Some(&bool_type.const_int(0, false)))
                    .unwrap();

                build.position_at_end(get_next);
                // there are at least n elements left, we can load them and increment
                let data_ptr = primitive_iter.llvm_data(ctx, &build, iter_ptr);
                let data_ptr = increment_pointer!(ctx, build, data_ptr, ptype.width(), curr_pos);
                let vec = build.build_load(vec_type, data_ptr, "vec").unwrap();
                vec.as_instruction_value()
                    .unwrap()
                    .set_alignment(1)
                    .unwrap();

                build.build_store(out_ptr, vec).unwrap();
                primitive_iter.llvm_increment_pos(ctx, &build, iter_ptr, llvm_n);
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                next
            }
            IteratorHolder::Bitmap(iter) => {
                let access = self.generate_random_access_block(ctx, llvm_mod, n)?;
                declare_blocks!(ctx, next, entry, none_left, get_next);
                build.position_at_end(entry);
                let curr_pos = iter.llvm_pos(ctx, &build, iter_ptr);
                let remaining = build
                    .build_int_sub(iter.llvm_len(ctx, &build, iter_ptr), curr_pos, "remaining")
                    .unwrap();
                let have_enough = build
                    .build_int_compare(IntPredicate::UGE, remaining, llvm_n, "have_enough")
                    .unwrap();
                build
                    .build_conditional_branch(have_enough, get_next, none_left)
                    .unwrap();
                build.position_at_end(none_left);
                build.build_return(Some(&bool_type.const_zero())).unwrap();
                build.position_at_end(get_next);
                build
                    .build_call(
                        access,
                        &[iter_ptr.into(), curr_pos.into(), out_ptr.into()],
                        "bitmap_block",
                    )
                    .unwrap();
                iter.llvm_increment_pos(ctx, &build, iter_ptr, llvm_n);
                build
                    .build_return(Some(&bool_type.const_all_ones()))
                    .unwrap();
                next
            }
            IteratorHolder::FixedSizeList(iter) => {
                let access = self.generate_random_access_block(ctx, llvm_mod, n)?;
                declare_blocks!(ctx, next, entry, none_left, get_next);
                build.position_at_end(entry);
                let curr_pos = iter.llvm_pos(ctx, &build, iter_ptr);
                let remaining = build
                    .build_int_sub(iter.llvm_len(ctx, &build, iter_ptr), curr_pos, "remaining")
                    .unwrap();
                let have_enough = build
                    .build_int_compare(IntPredicate::UGE, remaining, llvm_n, "have_enough")
                    .unwrap();
                build
                    .build_conditional_branch(have_enough, get_next, none_left)
                    .unwrap();
                build.position_at_end(none_left);
                build.build_return(Some(&bool_type.const_zero())).unwrap();
                build.position_at_end(get_next);
                build
                    .build_call(
                        access,
                        &[iter_ptr.into(), curr_pos.into(), out_ptr.into()],
                        "fixed_size_list_block",
                    )
                    .unwrap();
                iter.llvm_increment_pos(ctx, &build, iter_ptr, llvm_n);
                build
                    .build_return(Some(&bool_type.const_all_ones()))
                    .unwrap();
                next
            }
            IteratorHolder::Dictionary { arr, keys, values } => match &dt {
                DataType::Dictionary(k_dt, _) => {
                    let vec_type = vec_type.into_vector_type();
                    let key_block_next = keys
                        .generate_next_block(ctx, llvm_mod, n)
                        .expect("dictionary block capability requires block-readable keys");

                    let value_access = values
                        .generate_random_access(ctx, llvm_mod)
                        .expect("dictionary block capability requires random-access values");

                    declare_blocks!(ctx, next, entry, none_left, get_next);

                    build.position_at_end(entry);
                    let key_prim_type = PrimitiveType::for_arrow_type(k_dt);
                    let key_iter = arr.llvm_key_iter_ptr(ctx, &build, iter_ptr);
                    let key_vec_type = key_prim_type.llvm_vec_type(ctx, n).unwrap();
                    let key_buf = build.build_alloca(key_vec_type, "key_block").unwrap();
                    let key_block_result = build
                        .build_call(
                            key_block_next,
                            &[key_iter.into(), key_buf.into()],
                            "key_block_result",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                        .into_int_value();
                    build
                        .build_conditional_branch(key_block_result, get_next, none_left)
                        .unwrap();

                    build.position_at_end(none_left);
                    build
                        .build_return(Some(&bool_type.const_int(0, false)))
                        .unwrap();

                    build.position_at_end(get_next);
                    let key_vec = build
                        .build_load(key_vec_type, key_buf, "key_vec")
                        .unwrap()
                        .into_vector_value();
                    let key_vec = build
                        .build_int_cast(key_vec, i64_type.vec_type(n), "key_vec_cast")
                        .unwrap();
                    let mut out_vec = vec_type.const_zero();
                    for idx in 0..n as u64 {
                        let key = build
                            .build_extract_element(key_vec, i64_type.const_int(idx, false), "key")
                            .unwrap()
                            .into_int_value();
                        let val_iter = arr.llvm_val_iter_ptr(ctx, &build, iter_ptr);
                        let value = build
                            .build_call(value_access, &[val_iter.into(), key.into()], "value")
                            .unwrap()
                            .try_as_basic_value()
                            .unwrap_basic();
                        out_vec = build
                            .build_insert_element(
                                out_vec,
                                value,
                                i64_type.const_int(idx, false),
                                &format!("insert{}", idx),
                            )
                            .unwrap();
                    }
                    build.build_store(out_ptr, out_vec).unwrap();
                    build
                        .build_return(Some(&bool_type.const_int(1, false)))
                        .unwrap();

                    next
                }
                _ => unreachable!("dict iterator but not dict data type ({:?})", dt),
            },
            IteratorHolder::RunEnd {
                arr,
                run_ends,
                values,
            } => match &dt {
                DataType::RunEndEncoded(_, _) => {
                    let vec_type = vec_type.into_vector_type();
                    let access_ends = run_ends
                        .generate_random_access(ctx, llvm_mod)
                        .expect("run-end block capability requires random-access run ends");
                    let access_values = values
                        .generate_random_access(ctx, llvm_mod)
                        .expect("run-end block capability requires random-access values");

                    let umin = Intrinsic::find("llvm.umin").unwrap();
                    let umin_f = umin.get_declaration(llvm_mod, &[i64_type.into()]).unwrap();
                    declare_blocks!(
                        ctx,
                        next,
                        entry,
                        check_for_next,
                        fetch_next,
                        not_enough,
                        loop_cond,
                        check_vec_full,
                        fill_vec,
                        exit
                    );

                    build.position_at_end(entry);
                    let orig_pos = arr.llvm_pos(ctx, &build, iter_ptr);
                    let orig_rem = arr.llvm_remaining(ctx, &build, iter_ptr);
                    let ends_iter = arr.llvm_re_iter_ptr(ctx, &build, iter_ptr);
                    let vals_iter = arr.llvm_val_iter_ptr(ctx, &build, iter_ptr);
                    let vbuf = build.build_alloca(vec_type, "vbuf").unwrap();
                    build.build_store(vbuf, vec_type.const_zero()).unwrap();
                    let buf_idx_ptr = build.build_alloca(i64_type, "buf_idx").unwrap();
                    build
                        .build_store(buf_idx_ptr, i64_type.const_zero())
                        .unwrap();

                    let log_pos = arr.llvm_logical_pos(ctx, &build, iter_ptr);
                    let log_len = arr.llvm_logical_len(ctx, &build, iter_ptr);
                    let log_rem = build.build_int_sub(log_len, log_pos, "log_rem").unwrap();
                    let log_have_more = build
                        .build_int_compare(IntPredicate::UGE, log_rem, llvm_n, "have_log_next")
                        .unwrap();
                    build
                        .build_conditional_branch(log_have_more, loop_cond, not_enough)
                        .unwrap();

                    build.position_at_end(check_vec_full);
                    let buf_idx = build
                        .build_load(i64_type, buf_idx_ptr, "buf_idx")
                        .unwrap()
                        .into_int_value();
                    let res = build
                        .build_int_compare(
                            IntPredicate::ULT,
                            buf_idx,
                            i64_type.const_int(n as u64, false),
                            "vec_full",
                        )
                        .unwrap();
                    build
                        .build_conditional_branch(res, loop_cond, exit)
                        .unwrap();

                    build.position_at_end(loop_cond);
                    let remaining = arr.llvm_remaining(ctx, &build, iter_ptr);
                    let res = build
                        .build_int_compare(
                            IntPredicate::UGT,
                            remaining,
                            i64_type.const_zero(),
                            "have_remaining",
                        )
                        .unwrap();
                    build
                        .build_conditional_branch(res, fill_vec, check_for_next)
                        .unwrap();

                    build.position_at_end(check_for_next);
                    arr.llvm_inc_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                    let pos = arr.llvm_pos(ctx, &build, iter_ptr);
                    let len = arr.llvm_len(ctx, &build, iter_ptr);
                    let phs_have_more = build
                        .build_int_compare(IntPredicate::ULT, pos, len, "have_phs_next")
                        .unwrap();
                    build
                        .build_conditional_branch(phs_have_more, fetch_next, not_enough)
                        .unwrap();

                    build.position_at_end(not_enough);
                    arr.llvm_set_remaining(ctx, &build, iter_ptr, orig_rem);
                    arr.llvm_set_pos(ctx, &build, iter_ptr, orig_pos);
                    build.build_return(Some(&bool_type.const_zero())).unwrap();

                    build.position_at_end(fetch_next);
                    let my_end = build
                        .build_call(access_ends, &[ends_iter.into(), pos.into()], "my_end")
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                        .into_int_value();

                    let pr_end = build
                        .build_call(
                            access_ends,
                            &[
                                ends_iter.into(),
                                build
                                    .build_int_sub(pos, i64_type.const_int(1, false), "minus1")
                                    .unwrap()
                                    .into(),
                            ],
                            "pr_end",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                        .into_int_value();
                    let remaining = build.build_int_sub(my_end, pr_end, "remaining").unwrap();
                    let remaining = build
                        .build_int_cast(remaining, i64_type, "remaining_cast")
                        .unwrap();
                    arr.llvm_set_remaining(ctx, &build, iter_ptr, remaining);
                    build.build_unconditional_branch(loop_cond).unwrap();

                    build.position_at_end(fill_vec);
                    let pos = arr.llvm_pos(ctx, &build, iter_ptr);
                    let curr_val = build
                        .build_call(access_values, &[vals_iter.into(), pos.into()], "val")
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic();
                    let curr_val = build
                        .build_insert_element(
                            vec_type.const_zero(),
                            curr_val,
                            i64_type.const_zero(),
                            "curr_val_vec",
                        )
                        .unwrap();
                    let curr_val = build
                        .build_shuffle_vector(
                            curr_val,
                            vec_type.get_undef(),
                            vec_type.const_zero(),
                            "broadcasted",
                        )
                        .unwrap();

                    let remaining_values = arr.llvm_remaining(ctx, &build, iter_ptr);
                    let buf_idx = build
                        .build_load(i64_type, buf_idx_ptr, "buf_idx")
                        .unwrap()
                        .into_int_value();
                    let remaining_slots = build
                        .build_int_sub(
                            i64_type.const_int(n as u64, false),
                            buf_idx,
                            "remaining_slots",
                        )
                        .unwrap();
                    let to_fill = build
                        .build_call(
                            umin_f,
                            &[remaining_slots.into(), remaining_values.into()],
                            "to_fill",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                        .into_int_value();

                    // build a mask that 1 in the slots we want to insert value into
                    // ex: suppose block size = 8, to_fill = 2, curr_pos = 3
                    // desired mask: 00011000
                    // formula: ((1 << to_fill) - 1) << curr_pos
                    //
                    let mask = build
                        .build_left_shift(
                            build
                                .build_int_sub(
                                    build
                                        .build_left_shift(
                                            i64_type.const_int(1, false),
                                            to_fill,
                                            "mask",
                                        )
                                        .unwrap(),
                                    i64_type.const_int(1, false),
                                    "mask",
                                )
                                .unwrap(),
                            buf_idx,
                            "mask",
                        )
                        .unwrap();
                    let max_width_cond = build
                        .build_int_compare(
                            IntPredicate::EQ,
                            to_fill,
                            i64_type.const_int(n as u64, false),
                            "is_full",
                        )
                        .unwrap();
                    let mask = build
                        .build_select(max_width_cond, i64_type.const_all_ones(), mask, "mask")
                        .unwrap()
                        .into_int_value();

                    let mask = match n {
                        8 => build
                            .build_int_truncate(mask, ctx.i8_type(), "mask")
                            .unwrap(),
                        16 => build
                            .build_int_truncate(mask, ctx.i16_type(), "mask")
                            .unwrap(),
                        32 => build
                            .build_int_truncate(mask, ctx.i32_type(), "mask")
                            .unwrap(),
                        64 => mask,
                        _ => return None,
                    };

                    let mask = build
                        .build_bit_cast(mask, bool_type.vec_type(n), "mask_v")
                        .unwrap()
                        .into_vector_value();

                    let curr_buf = build
                        .build_load(vec_type, vbuf, "curr_buf")
                        .unwrap()
                        .into_vector_value();

                    let new_buf = build
                        .build_select(mask, curr_val, curr_buf, "new_buf")
                        .unwrap();

                    build.build_store(vbuf, new_buf).unwrap();
                    arr.llvm_dec_remaining(ctx, &build, iter_ptr, to_fill);
                    let new_buf_ptr = build
                        .build_int_add(buf_idx, to_fill, "new_buf_ptr")
                        .unwrap();
                    build.build_store(buf_idx_ptr, new_buf_ptr).unwrap();
                    build.build_unconditional_branch(check_vec_full).unwrap();

                    build.position_at_end(exit);
                    let result = build.build_load(vec_type, vbuf, "result").unwrap();
                    arr.llvm_inc_logical_pos(
                        ctx,
                        &build,
                        iter_ptr,
                        i64_type.const_int(n as u64, false),
                    );
                    build.build_store(out_ptr, result).unwrap();
                    build
                        .build_return(Some(&bool_type.const_all_ones()))
                        .unwrap();

                    next
                }
                _ => unreachable!("run-end iterator but not run-end data type ({:?})", dt),
            },
            IteratorHolder::ScalarPrimitive(s) => {
                let vec_type = vec_type.into_vector_type();
                let ptype = s.ptype;
                let llvm_type = ptype.llvm_type(ctx);
                assert_eq!(ptype.width(), s.width as usize);
                let get_next_single = self.generate_next(ctx, llvm_mod);
                declare_blocks!(ctx, next, entry);
                build.position_at_end(entry);
                let val_buf = build.build_alloca(llvm_type, "val_buf").unwrap();
                build
                    .build_call(get_next_single, &[iter_ptr.into(), val_buf.into()], "get")
                    .unwrap();
                let constant = build.build_load(llvm_type, val_buf, "constant").unwrap();
                constant
                    .as_instruction_value()
                    .unwrap()
                    .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                    .unwrap();
                let v = build
                    .build_insert_element(
                        vec_type.const_zero(),
                        constant,
                        i64_type.const_zero(),
                        "v",
                    )
                    .unwrap();
                let v = build
                    .build_shuffle_vector(
                        v,
                        vec_type.const_zero(),
                        vec_type.const_zero(),
                        "splatted",
                    )
                    .unwrap();
                build.build_store(out_ptr, v).unwrap();
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();
                next
            }
            IteratorHolder::String(_)
            | IteratorHolder::LargeString(_)
            | IteratorHolder::View(_)
            | IteratorHolder::SetBit(_)
            | IteratorHolder::List(_)
            | IteratorHolder::ScalarBoolean(_)
            | IteratorHolder::ScalarString(_)
            | IteratorHolder::ScalarBinary(_)
            | IteratorHolder::ScalarVec(_) => {
                unreachable!("next-block capability disagrees with iterator variant")
            }
        };

        Some(res)
    }

    /// Adds or reuses the scalar-next function for this iterator's code-generation shape.
    ///
    /// The generated function fetches the next element and advances the iterator.
    /// Its signature is:
    ///
    /// `fn next(iter: ptr, out: ptr_to_el) -> bool`
    pub fn generate_next<'a>(&self, ctx: &'a Context, llvm_mod: &Module<'a>) -> FunctionValue<'a> {
        let build = ctx.create_builder();
        let dt = self.data_type();
        let llvm_type = self.codegen_info().value_type.llvm_type(ctx);
        let bool_type = ctx.bool_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();

        let fn_type = bool_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let name = format!("{}_next", self.codegen_label());
        if let Some(existing) = llvm_mod.get_function(&name) {
            assert_eq!(existing.get_type(), fn_type);
            return existing;
        }
        let next = llvm_mod.add_function(
            &name,
            fn_type,
            Some(
                #[cfg(test)]
                Linkage::External,
                #[cfg(not(test))]
                Linkage::Private,
            ),
        );
        let iter_ptr = next.get_nth_param(0).unwrap().into_pointer_value();
        let out_ptr = next.get_nth_param(1).unwrap().into_pointer_value();
        set_noalias_params(&next);
        match self {
            IteratorHolder::Primitive(primitive_iter) => {
                let ptype = PrimitiveType::for_arrow_type(&dt);
                declare_blocks!(ctx, next, entry, none_left, get_next);

                build.position_at_end(entry);
                let curr_pos = primitive_iter.llvm_pos(ctx, &build, iter_ptr);
                let curr_len = primitive_iter.llvm_len(ctx, &build, iter_ptr);
                let have_more = build
                    .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                    .unwrap();
                build
                    .build_conditional_branch(have_more, get_next, none_left)
                    .unwrap();

                build.position_at_end(none_left);
                build
                    .build_return(Some(&bool_type.const_int(0, false)))
                    .unwrap();

                build.position_at_end(get_next);
                // there are at least n elements left, we can load them and increment
                let data_ptr = primitive_iter.llvm_data(ctx, &build, iter_ptr);
                let data_ptr = increment_pointer!(ctx, build, data_ptr, ptype.width(), curr_pos);
                let out = build.build_load(llvm_type, data_ptr, "elem").unwrap();
                build.build_store(out_ptr, out).unwrap();
                primitive_iter.llvm_increment_pos(
                    ctx,
                    &build,
                    iter_ptr,
                    i64_type.const_int(1, false),
                );
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                next
            }
            IteratorHolder::String(iter) => {
                let access = self.generate_random_access(ctx, llvm_mod).unwrap();
                declare_blocks!(ctx, next, entry, none_left, get_next);

                build.position_at_end(entry);
                let curr_pos = iter.llvm_pos(ctx, &build, iter_ptr);
                let curr_len = iter.llvm_len(ctx, &build, iter_ptr);
                let have_more = build
                    .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                    .unwrap();
                build
                    .build_conditional_branch(have_more, get_next, none_left)
                    .unwrap();

                build.position_at_end(none_left);
                build
                    .build_return(Some(&bool_type.const_int(0, false)))
                    .unwrap();

                build.position_at_end(get_next);
                // there are at least n elements left, we can load them and increment
                let result = build
                    .build_call(access, &[iter_ptr.into(), curr_pos.into()], "access_result")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic();

                build.build_store(out_ptr, result).unwrap();
                iter.llvm_increment_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                next
            }
            IteratorHolder::LargeString(iter) => {
                let access = self.generate_random_access(ctx, llvm_mod).unwrap();
                declare_blocks!(ctx, next, entry, none_left, get_next);

                build.position_at_end(entry);
                let curr_pos = iter.llvm_pos(ctx, &build, iter_ptr);
                let curr_len = iter.llvm_len(ctx, &build, iter_ptr);
                let have_more = build
                    .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                    .unwrap();
                build
                    .build_conditional_branch(have_more, get_next, none_left)
                    .unwrap();

                build.position_at_end(none_left);
                build
                    .build_return(Some(&bool_type.const_int(0, false)))
                    .unwrap();

                build.position_at_end(get_next);
                // there are at least n elements left, we can load them and increment
                let result = build
                    .build_call(access, &[iter_ptr.into(), curr_pos.into()], "access_result")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic();

                build.build_store(out_ptr, result).unwrap();
                iter.llvm_increment_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                next
            }
            IteratorHolder::View(iter) => {
                let access = self.generate_random_access(ctx, llvm_mod).unwrap();
                declare_blocks!(ctx, next, entry, none_left, get_next);

                build.position_at_end(entry);
                let curr_pos = iter.llvm_pos(ctx, &build, iter_ptr);
                let curr_len = iter.llvm_len(ctx, &build, iter_ptr);
                let have_more = build
                    .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                    .unwrap();
                build
                    .build_conditional_branch(have_more, get_next, none_left)
                    .unwrap();

                build.position_at_end(none_left);
                build
                    .build_return(Some(&bool_type.const_int(0, false)))
                    .unwrap();

                build.position_at_end(get_next);
                // there are at least n elements left, we can load them and increment
                let result = build
                    .build_call(access, &[iter_ptr.into(), curr_pos.into()], "access_result")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic();

                build.build_store(out_ptr, result).unwrap();
                iter.llvm_increment_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                next
            }
            IteratorHolder::Bitmap(bitmap_iterator) => {
                let access = self.generate_random_access(ctx, llvm_mod).unwrap();
                declare_blocks!(ctx, next, entry, none_left, get_next);

                build.position_at_end(entry);
                let curr_pos = bitmap_iterator.llvm_pos(ctx, &build, iter_ptr);
                let curr_len = bitmap_iterator.llvm_len(ctx, &build, iter_ptr);
                let have_more = build
                    .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                    .unwrap();
                build
                    .build_conditional_branch(have_more, get_next, none_left)
                    .unwrap();

                build.position_at_end(none_left);
                build
                    .build_return(Some(&bool_type.const_int(0, false)))
                    .unwrap();

                build.position_at_end(get_next);
                let result = build
                    .build_call(access, &[iter_ptr.into(), curr_pos.into()], "access_result")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic();
                build.build_store(out_ptr, result).unwrap();
                bitmap_iterator.llvm_increment_pos(
                    ctx,
                    &build,
                    iter_ptr,
                    i64_type.const_int(1, false),
                );
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                next
            }
            IteratorHolder::SetBit(it) => {
                declare_blocks!(
                    ctx,
                    next,
                    entry,
                    head_cond,
                    head_body,
                    main_cond,
                    main_body,
                    fetch_next_segment,
                    use_curr_segment,
                    increment_and_return_from_segment,
                    return_from_segment,
                    tail_cond,
                    tail_body,
                    exit
                );

                let cttz_id =
                    Intrinsic::find("llvm.cttz").expect("llvm.cttz not in Intrinsic list");
                let cttz_i64 = cttz_id
                    .get_declaration(llvm_mod, &[ctx.i64_type().into()])
                    .expect("Couldn't declare llvm.cttz.i64");

                build.position_at_end(entry);
                build.build_unconditional_branch(head_cond).unwrap();

                build.position_at_end(head_cond);
                let (header_ptr, header_pos, header_len) =
                    it.llvm_header_info(ctx, &build, iter_ptr);
                let have_header = build
                    .build_int_compare(IntPredicate::ULT, header_pos, header_len, "have_header")
                    .unwrap();
                build
                    .build_conditional_branch(have_header, head_body, main_cond)
                    .unwrap();

                build.position_at_end(head_body);
                let res = build
                    .build_load(
                        i64_type,
                        increment_pointer!(ctx, build, header_ptr, 8, header_pos),
                        "head_val",
                    )
                    .unwrap()
                    .into_int_value();
                it.llvm_inc_header_pos(ctx, &build, iter_ptr);
                build.build_store(out_ptr, res).unwrap();

                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                // check if the current segment is zero or not
                build.position_at_end(main_cond);
                let curr_segment = it.llvm_get_current_u64(ctx, &build, iter_ptr);
                let is_zero = build
                    .build_int_compare(
                        IntPredicate::EQ,
                        curr_segment,
                        i64_type.const_zero(),
                        "is_curr_zero",
                    )
                    .unwrap();
                build
                    .build_conditional_branch(is_zero, main_body, use_curr_segment)
                    .unwrap();

                // check if we still have segments left to read
                build.position_at_end(main_body);
                let curr_segment_idx = it.llvm_get_curr_segment_pos(ctx, &build, iter_ptr);
                let segment_len = it.llvm_get_num_segments(ctx, &build, iter_ptr);
                let cmp = build
                    .build_int_compare(IntPredicate::ULT, curr_segment_idx, segment_len, "cmp")
                    .unwrap();
                build
                    .build_conditional_branch(cmp, fetch_next_segment, tail_cond)
                    .unwrap();

                build.position_at_end(fetch_next_segment);
                let new_segment = it.llvm_get_segment(ctx, &build, curr_segment_idx, iter_ptr);
                it.llvm_set_current_u64(ctx, &build, new_segment, iter_ptr);
                let rel_segment_idx = build
                    .build_int_sub(
                        curr_segment_idx,
                        it.llvm_segment_start(ctx, &build, iter_ptr),
                        "rel_segment_idx",
                    )
                    .unwrap();
                let segment_bit_offset = build
                    .build_int_mul(
                        rel_segment_idx,
                        i64_type.const_int(64, false),
                        "segment_bit_offset",
                    )
                    .unwrap();
                let segment_base = build
                    .build_int_add(
                        it.llvm_head_len(ctx, &build, iter_ptr),
                        segment_bit_offset,
                        "segment_base",
                    )
                    .unwrap();
                it.llvm_set_current_bit_idx(ctx, &build, segment_base, iter_ptr);
                it.llvm_inc_curr_segment(ctx, &build, iter_ptr);
                build.build_unconditional_branch(main_cond).unwrap();

                build.position_at_end(use_curr_segment);
                let num_trailing = build
                    .build_call(
                        cttz_i64,
                        &[curr_segment.into(), bool_type.const_all_ones().into()],
                        "num_trailing",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_int_value();
                it.llvm_clear_last(ctx, &build, iter_ptr);
                let res =
                    it.llvm_add_and_get_current_bit_idx(ctx, &build, iter_ptr, num_trailing, false);
                build.build_store(out_ptr, res).unwrap();
                let is_now_zero = build
                    .build_int_compare(
                        IntPredicate::EQ,
                        it.llvm_get_current_u64(ctx, &build, iter_ptr),
                        i64_type.const_zero(),
                        "is_now_zero",
                    )
                    .unwrap();
                build
                    .build_conditional_branch(
                        is_now_zero,
                        increment_and_return_from_segment,
                        return_from_segment,
                    )
                    .unwrap();

                build.position_at_end(increment_and_return_from_segment);
                it.llvm_add_and_get_current_bit_idx(
                    ctx,
                    &build,
                    iter_ptr,
                    i64_type.const_int(64, false),
                    true,
                );
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                build.position_at_end(return_from_segment);
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                build.position_at_end(tail_cond);
                let rel_segments_done = build
                    .build_int_sub(
                        it.llvm_get_curr_segment_pos(ctx, &build, iter_ptr),
                        it.llvm_segment_start(ctx, &build, iter_ptr),
                        "segments_done",
                    )
                    .unwrap();
                let tail_bit_offset = build
                    .build_int_mul(
                        rel_segments_done,
                        i64_type.const_int(64, false),
                        "tail_bit_offset",
                    )
                    .unwrap();
                let tail_base = build
                    .build_int_add(
                        it.llvm_head_len(ctx, &build, iter_ptr),
                        tail_bit_offset,
                        "tail_base",
                    )
                    .unwrap();
                it.llvm_set_current_bit_idx(ctx, &build, tail_base, iter_ptr);
                let (tail_ptr, tail_pos, tail_len) = it.llvm_tail_info(ctx, &build, iter_ptr);
                let have_tail = build
                    .build_int_compare(IntPredicate::ULT, tail_pos, tail_len, "have_tail")
                    .unwrap();
                build
                    .build_conditional_branch(have_tail, tail_body, exit)
                    .unwrap();

                build.position_at_end(tail_body);
                let res = build
                    .build_load(
                        i64_type,
                        increment_pointer!(ctx, build, tail_ptr, 8, tail_pos),
                        "tail_val",
                    )
                    .unwrap()
                    .into_int_value();
                it.llvm_inc_tail_pos(ctx, &build, iter_ptr);
                let res = it.llvm_add_and_get_current_bit_idx(ctx, &build, iter_ptr, res, false);
                build.build_store(out_ptr, res).unwrap();
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                build.position_at_end(exit);
                build
                    .build_return(Some(&bool_type.const_int(0, false)))
                    .unwrap();

                next
            }
            IteratorHolder::Dictionary { arr, keys, values } => match &dt {
                DataType::Dictionary(k_dt, _) => {
                    let key_next = keys.generate_next(ctx, llvm_mod);
                    let values_access = values
                        .generate_random_access(ctx, llvm_mod)
                        .expect("dictionary iteration requires random-access values");
                    declare_blocks!(ctx, next, entry, none_left, fetch);

                    build.position_at_end(entry);
                    let key_type = PrimitiveType::for_arrow_type(k_dt).llvm_type(ctx);
                    let key_iter = arr.llvm_key_iter_ptr(ctx, &build, iter_ptr);
                    let val_iter = arr.llvm_val_iter_ptr(ctx, &build, iter_ptr);

                    let key_buf = build.build_alloca(key_type, "key_buf").unwrap();
                    let had_next = build
                        .build_call(key_next, &[key_iter.into(), key_buf.into()], "next_key")
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                        .into_int_value();
                    build
                        .build_conditional_branch(had_next, fetch, none_left)
                        .unwrap();

                    build.position_at_end(fetch);
                    let next_key = build
                        .build_int_cast(
                            build
                                .build_load(key_type, key_buf, "next_key")
                                .unwrap()
                                .into_int_value(),
                            i64_type,
                            "casted_key",
                        )
                        .unwrap();
                    let value = build
                        .build_call(values_access, &[val_iter.into(), next_key.into()], "value")
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic();
                    build.build_store(out_ptr, value).unwrap();
                    build
                        .build_return(Some(&bool_type.const_int(1, false)))
                        .unwrap();

                    build.position_at_end(none_left);
                    build
                        .build_return(Some(&bool_type.const_int(0, false)))
                        .unwrap();
                    next
                }
                _ => unreachable!("dict iterator but not dict data type ({:?})", dt),
            },
            IteratorHolder::RunEnd {
                arr,
                run_ends,
                values,
            } => match &dt {
                DataType::RunEndEncoded(_, _) => {
                    let re_access = run_ends
                        .generate_random_access(ctx, llvm_mod)
                        .expect("run-end iteration requires random-access run ends");
                    let val_access = values
                        .generate_random_access(ctx, llvm_mod)
                        .expect("run-end iteration requires random-access values");

                    declare_blocks!(
                        ctx,
                        next,
                        entry,
                        check_remaining,
                        has_remaining,
                        none_remaining,
                        load_next_run,
                        exhausted
                    );

                    build.position_at_end(entry);
                    let val_iter_ptr = arr.llvm_val_iter_ptr(ctx, &build, iter_ptr);
                    let re_iter_ptr = arr.llvm_re_iter_ptr(ctx, &build, iter_ptr);
                    let log_pos = arr.llvm_logical_pos(ctx, &build, iter_ptr);
                    let log_len = arr.llvm_logical_len(ctx, &build, iter_ptr);
                    let log_have_more = build
                        .build_int_compare(IntPredicate::ULT, log_pos, log_len, "have_log_next")
                        .unwrap();
                    build
                        .build_conditional_branch(log_have_more, check_remaining, exhausted)
                        .unwrap();

                    build.position_at_end(check_remaining);
                    let remaining = arr.llvm_remaining(ctx, &build, iter_ptr);
                    let res = build
                        .build_int_compare(
                            IntPredicate::UGT,
                            remaining,
                            i64_type.const_zero(),
                            "has_remaining",
                        )
                        .unwrap();
                    build
                        .build_conditional_branch(res, has_remaining, none_remaining)
                        .unwrap();

                    build.position_at_end(has_remaining);
                    // there are values left in the current run
                    let curr_pos = arr.llvm_pos(ctx, &build, iter_ptr);
                    let val = build
                        .build_call(val_access, &[val_iter_ptr.into(), curr_pos.into()], "value")
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic();
                    build.build_store(out_ptr, val).unwrap();
                    arr.llvm_dec_remaining(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                    arr.llvm_inc_logical_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                    build
                        .build_return(Some(&bool_type.const_all_ones()))
                        .unwrap();

                    build.position_at_end(none_remaining);
                    // there are no values left in the current run -- either load a
                    // new run, or return false
                    arr.llvm_inc_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                    let curr_pos = arr.llvm_pos(ctx, &build, iter_ptr);
                    let re_len = arr.llvm_len(ctx, &build, iter_ptr);
                    let have_another_run = build
                        .build_int_compare(IntPredicate::ULT, curr_pos, re_len, "another_run")
                        .unwrap();
                    build
                        .build_conditional_branch(have_another_run, load_next_run, exhausted)
                        .unwrap();

                    build.position_at_end(load_next_run);
                    let curr_pos = arr.llvm_pos(ctx, &build, iter_ptr);
                    let my_end = build
                        .build_call(re_access, &[re_iter_ptr.into(), curr_pos.into()], "my_end")
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                        .into_int_value();
                    let prev_end = build
                        .build_call(
                            re_access,
                            &[
                                re_iter_ptr.into(),
                                build
                                    .build_int_sub(
                                        curr_pos,
                                        i64_type.const_int(1, false),
                                        "prev_pos",
                                    )
                                    .unwrap()
                                    .into(),
                            ],
                            "prev_end",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                        .into_int_value();
                    let new_remaining = build
                        .build_int_sub(my_end, prev_end, "new_remaining")
                        .unwrap();
                    let new_remaining_cast = build
                        .build_int_cast(new_remaining, i64_type, "new_remaining_cast")
                        .unwrap();
                    arr.llvm_set_remaining(ctx, &build, iter_ptr, new_remaining_cast);
                    build.build_unconditional_branch(check_remaining).unwrap();

                    build.position_at_end(exhausted);
                    build.build_return(Some(&bool_type.const_zero())).unwrap();

                    next
                }
                _ => unreachable!("run-end iterator but not run-end data type ({:?})", dt),
            },
            IteratorHolder::FixedSizeList(iter) => {
                declare_blocks!(ctx, next, entry, none_left, get_next);

                build.position_at_end(entry);
                let curr_pos = iter.llvm_pos(ctx, &build, iter_ptr);
                let curr_len = iter.llvm_len(ctx, &build, iter_ptr);
                let have_more = build
                    .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                    .unwrap();
                build
                    .build_conditional_branch(have_more, get_next, none_left)
                    .unwrap();

                build.position_at_end(none_left);
                build
                    .build_return(Some(&bool_type.const_int(0, false)))
                    .unwrap();

                build.position_at_end(get_next);
                let out = build_fixed_size_list_value(
                    ctx, llvm_mod, &build, &dt, iter, iter_ptr, curr_pos,
                )
                .expect("fixed-size-list iteration requires a random-access child");
                build.build_store(out_ptr, out).unwrap();
                iter.llvm_increment_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                next
            }
            IteratorHolder::List(iter) => {
                declare_blocks!(ctx, next, entry, none_left, get_next);

                build.position_at_end(entry);
                let curr_pos = iter.llvm_pos(ctx, &build, iter_ptr);
                let curr_len = iter.llvm_len(ctx, &build, iter_ptr);
                let have_more = build
                    .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                    .unwrap();
                build
                    .build_conditional_branch(have_more, get_next, none_left)
                    .unwrap();

                build.position_at_end(none_left);
                build
                    .build_return(Some(&bool_type.const_int(0, false)))
                    .unwrap();

                build.position_at_end(get_next);
                let out = build_variable_list_value(ctx, &build, iter, iter_ptr, curr_pos);
                build.build_store(out_ptr, out).unwrap();
                iter.llvm_increment_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                next
            }
            IteratorHolder::ScalarPrimitive(s) => {
                let ptype = PrimitiveType::for_arrow_type(&dt);
                assert_eq!(ptype.width(), s.width as usize);
                declare_blocks!(ctx, next, entry);
                build.position_at_end(entry);
                let ptr = s.llvm_val_ptr(ctx, &build, iter_ptr);
                let constant = match &dt {
                    DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64 => {
                        let data = build
                            .build_load(i64_type, ptr, "const_u64")
                            .unwrap()
                            .into_int_value();
                        data.as_instruction_value()
                            .unwrap()
                            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                            .unwrap();
                        build
                            .build_int_cast(data, llvm_type.into_int_type(), "casted")
                            .unwrap()
                            .as_basic_value_enum()
                    }
                    DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                        let data = build
                            .build_load(i64_type, ptr, "const_u64")
                            .unwrap()
                            .into_int_value();
                        data.as_instruction_value()
                            .unwrap()
                            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                            .unwrap();
                        let data = build
                            .build_int_cast(
                                data,
                                PrimitiveType::int_with_width(s.width as usize)
                                    .llvm_type(ctx)
                                    .into_int_type(),
                                "const_int",
                            )
                            .unwrap();
                        build
                            .build_bit_cast(data, llvm_type, "const_float")
                            .unwrap()
                    }
                    _ => unreachable!(),
                };
                build.build_store(out_ptr, constant).unwrap();
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();
                next
            }
            IteratorHolder::ScalarBoolean(s) => {
                declare_blocks!(ctx, next, entry);
                build.position_at_end(entry);
                let ptr = s.llvm_val_ptr(ctx, &build, iter_ptr);
                let constant = build
                    .build_load(ctx.i8_type(), ptr, "const_bool_u8")
                    .unwrap()
                    .into_int_value();
                mark_load_invariant!(ctx, constant);
                let constant = build
                    .build_int_truncate(constant, ctx.bool_type(), "trunc_const")
                    .unwrap();
                build.build_store(out_ptr, constant).unwrap();
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();
                next
            }
            IteratorHolder::ScalarString(s) => {
                let ptr_type = ctx.ptr_type(AddressSpace::default());
                let ret_type = PrimitiveType::P64x2.llvm_type(ctx).into_struct_type();
                declare_blocks!(ctx, next, entry);
                build.position_at_end(entry);
                let (ptr1, ptr2) = s.llvm_val_ptr(ctx, &build, iter_ptr);
                let ptr1 = build
                    .build_load(ptr_type, ptr1, "ptr1")
                    .unwrap()
                    .into_pointer_value();
                ptr1.as_instruction_value()
                    .unwrap()
                    .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                    .unwrap();
                let ptr2 = build
                    .build_load(ptr_type, ptr2, "ptr2")
                    .unwrap()
                    .into_pointer_value();
                ptr2.as_instruction_value()
                    .unwrap()
                    .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                    .unwrap();

                let to_return = ret_type.const_zero();
                let to_return = build
                    .build_insert_value(to_return, ptr1, 0, "to_return")
                    .unwrap();
                let to_return = build
                    .build_insert_value(to_return, ptr2, 1, "to_return")
                    .unwrap();
                build.build_store(out_ptr, to_return).unwrap();
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                next
            }
            IteratorHolder::ScalarBinary(s) => {
                let ptr_type = ctx.ptr_type(AddressSpace::default());
                let ret_type = PrimitiveType::P64x2.llvm_type(ctx).into_struct_type();
                declare_blocks!(ctx, next, entry);
                build.position_at_end(entry);
                let (ptr1, ptr2) = s.llvm_val_ptr(ctx, &build, iter_ptr);
                let ptr1 = build
                    .build_load(ptr_type, ptr1, "ptr1")
                    .unwrap()
                    .into_pointer_value();
                ptr1.as_instruction_value()
                    .unwrap()
                    .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                    .unwrap();
                let ptr2 = build
                    .build_load(ptr_type, ptr2, "ptr2")
                    .unwrap()
                    .into_pointer_value();
                ptr2.as_instruction_value()
                    .unwrap()
                    .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                    .unwrap();

                let to_return = ret_type.const_zero();
                let to_return = build
                    .build_insert_value(to_return, ptr1, 0, "to_return")
                    .unwrap();
                let to_return = build
                    .build_insert_value(to_return, ptr2, 1, "to_return")
                    .unwrap();
                build.build_store(out_ptr, to_return).unwrap();
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                next
            }
            IteratorHolder::ScalarVec(iter) => {
                declare_blocks!(ctx, next, entry);
                build.position_at_end(entry);
                let val = iter.llvm_val(ctx, &build, iter_ptr);
                build.build_store(out_ptr, val).unwrap();
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                next
            }
        }
    }

    /// Adds or reuses the random-access function for this iterator's code-generation shape.
    ///
    /// The generated function indexes from the base data without advancing the
    /// iterator. Its signature is:
    ///
    /// `fn access(iter: ptr, el: u64) -> T`
    pub fn generate_random_access<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
    ) -> Option<FunctionValue<'a>> {
        let info = self.codegen_info();
        if !info.random_access {
            return None;
        }

        let build = ctx.create_builder();
        let dt = self.data_type();
        let llvm_type = info.value_type.llvm_type(ctx);
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();

        let fn_type = llvm_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let name = format!("{}_access", self.codegen_label());
        if let Some(existing) = llvm_mod.get_function(&name) {
            assert_eq!(existing.get_type(), fn_type);
            return Some(existing);
        }
        let access_f = llvm_mod.add_function(
            &name,
            fn_type,
            Some(
                #[cfg(test)]
                Linkage::External,
                #[cfg(not(test))]
                Linkage::Private,
            ),
        );
        let iter_ptr = access_f.get_nth_param(0).unwrap().into_pointer_value();
        let idx = access_f.get_nth_param(1).unwrap().into_int_value();

        match self {
            IteratorHolder::Primitive(primitive_iter) => {
                let ptype = PrimitiveType::for_arrow_type(&dt);
                declare_blocks!(ctx, access_f, entry);

                build.position_at_end(entry);
                let data_ptr = primitive_iter.llvm_data(ctx, &build, iter_ptr);
                let data_ptr = increment_pointer!(ctx, build, data_ptr, ptype.width(), idx);
                let out = build.build_load(llvm_type, data_ptr, "elem").unwrap();
                build.build_return(Some(&out)).unwrap();

                Some(access_f)
            }
            IteratorHolder::FixedSizeList(iter) => {
                declare_blocks!(ctx, access_f, entry);

                build.position_at_end(entry);
                let out =
                    build_fixed_size_list_value(ctx, llvm_mod, &build, &dt, iter, iter_ptr, idx)
                        .expect("fixed-size-list access requires a random-access child");
                build.build_return(Some(&out)).unwrap();

                Some(access_f)
            }
            IteratorHolder::List(iter) => {
                declare_blocks!(ctx, access_f, entry);

                build.position_at_end(entry);
                let out = build_variable_list_value(ctx, &build, iter, iter_ptr, idx);
                build.build_return(Some(&out)).unwrap();

                Some(access_f)
            }
            IteratorHolder::String(ih) => {
                let i32_type = ctx.i32_type();
                let ret_type = PrimitiveType::P64x2.llvm_type(ctx).into_struct_type();

                declare_blocks!(ctx, access_f, entry);

                build.position_at_end(entry);
                let offsets = ih.llvm_get_offset_ptr(ctx, &build, iter_ptr);
                let offset1 = build
                    .build_load(
                        i32_type,
                        increment_pointer!(ctx, build, offsets, 4, idx),
                        "offset1",
                    )
                    .unwrap()
                    .into_int_value();
                let offset2 = build
                    .build_load(
                        i32_type,
                        increment_pointer!(
                            ctx,
                            build,
                            offsets,
                            4,
                            build
                                .build_int_add(idx, i64_type.const_int(1, false), "inc")
                                .unwrap()
                        ),
                        "offset2",
                    )
                    .unwrap()
                    .into_int_value();

                let offset1 = build
                    .build_int_z_extend(offset1, i64_type, "offset1")
                    .unwrap();
                let offset2 = build
                    .build_int_z_extend(offset2, i64_type, "offset1")
                    .unwrap();

                let data = ih.llvm_get_data_ptr(ctx, &build, iter_ptr);
                let ptr1 = increment_pointer!(ctx, build, data, 1, offset1);
                let ptr2 = increment_pointer!(ctx, build, data, 1, offset2);
                let to_return = ret_type.const_zero();
                let to_return = build
                    .build_insert_value(to_return, ptr1, 0, "to_return")
                    .unwrap();
                let to_return = build
                    .build_insert_value(to_return, ptr2, 1, "to_return")
                    .unwrap();
                build.build_return(Some(&to_return)).unwrap();
                Some(access_f)
            }
            IteratorHolder::LargeString(ih) => {
                let ret_type = PrimitiveType::P64x2.llvm_type(ctx).into_struct_type();

                declare_blocks!(ctx, access_f, entry);

                build.position_at_end(entry);
                let offsets = ih.llvm_get_offset_ptr(ctx, &build, iter_ptr);
                let offset1 = build
                    .build_load(
                        i64_type,
                        increment_pointer!(ctx, build, offsets, 8, idx),
                        "offset1",
                    )
                    .unwrap()
                    .into_int_value();
                let offset2 = build
                    .build_load(
                        i64_type,
                        increment_pointer!(
                            ctx,
                            build,
                            offsets,
                            8,
                            build
                                .build_int_add(idx, i64_type.const_int(1, false), "inc")
                                .unwrap()
                        ),
                        "offset2",
                    )
                    .unwrap()
                    .into_int_value();

                let data = ih.llvm_get_data_ptr(ctx, &build, iter_ptr);
                let ptr1 = increment_pointer!(ctx, build, data, 1, offset1);
                let ptr2 = increment_pointer!(ctx, build, data, 1, offset2);
                let to_return = ret_type.const_zero();
                let to_return = build
                    .build_insert_value(to_return, ptr1, 0, "to_return")
                    .unwrap();
                let to_return = build
                    .build_insert_value(to_return, ptr2, 1, "to_return")
                    .unwrap();
                build.build_return(Some(&to_return)).unwrap();
                Some(access_f)
            }
            IteratorHolder::View(iter) => {
                let i128_type = ctx.i128_type();
                let i32_type = ctx.i32_type();
                let ret_type = PrimitiveType::P64x2.llvm_type(ctx).into_struct_type();

                declare_blocks!(ctx, access_f, entry, short_str, long_str);

                build.position_at_end(entry);
                let view_ptr = iter.llvm_view_ptr(ctx, &build, iter_ptr);
                let our_view_ptr = increment_pointer!(ctx, build, view_ptr, 16, idx);
                let our_view = build
                    .build_load(i128_type, our_view_ptr, "our_view")
                    .unwrap()
                    .into_int_value();
                let as_vec = build
                    .build_bit_cast(our_view, i32_type.vec_type(4), "as_vec")
                    .unwrap()
                    .into_vector_value();
                let str_len = build
                    .build_extract_element(as_vec, i64_type.const_zero(), "str_len")
                    .unwrap()
                    .into_int_value();
                let str_len = build
                    .build_int_s_extend(str_len, i64_type, "ext_len")
                    .unwrap();
                let cmp = build
                    .build_int_compare(
                        IntPredicate::SLE,
                        str_len,
                        i64_type.const_int(12, true),
                        "cmp",
                    )
                    .unwrap();
                build
                    .build_conditional_branch(cmp, short_str, long_str)
                    .unwrap();

                build.position_at_end(short_str);
                let ptr1 = increment_pointer!(ctx, build, our_view_ptr, 4);
                let ptr2 = increment_pointer!(ctx, build, ptr1, 1, str_len);
                let to_return = ret_type.const_zero();
                let to_return = build
                    .build_insert_value(to_return, ptr1, 0, "to_return")
                    .unwrap();
                let to_return = build
                    .build_insert_value(to_return, ptr2, 1, "to_return")
                    .unwrap();
                build.build_return(Some(&to_return)).unwrap();

                build.position_at_end(long_str);
                let buf_idx = build
                    .build_extract_element(as_vec, i64_type.const_int(2, false), "buf_idx")
                    .unwrap()
                    .into_int_value();
                let buf_base = iter.llvm_buffer_ptr(ctx, &build, iter_ptr, buf_idx);
                let offset = build
                    .build_extract_element(as_vec, i64_type.const_int(3, false), "offset")
                    .unwrap()
                    .into_int_value();
                let offset = build
                    .build_int_s_extend_or_bit_cast(offset, i64_type, "offset_ext")
                    .unwrap();
                let ptr1 = increment_pointer!(ctx, build, buf_base, 1, offset);
                let ptr2 = increment_pointer!(ctx, build, ptr1, 1, str_len);
                let to_return = ret_type.const_zero();
                let to_return = build
                    .build_insert_value(to_return, ptr1, 0, "to_return")
                    .unwrap();
                let to_return = build
                    .build_insert_value(to_return, ptr2, 1, "to_return")
                    .unwrap();
                build.build_return(Some(&to_return)).unwrap();

                Some(access_f)
            }
            IteratorHolder::Bitmap(bitmap_iterator) => {
                let ptype = PrimitiveType::for_arrow_type(&dt);
                declare_blocks!(ctx, access_f, entry);

                build.position_at_end(entry);
                let data_ptr = bitmap_iterator.llvm_get_data_ptr(ctx, &build, iter_ptr);
                let slice_offset = bitmap_iterator.llvm_slice_offset(ctx, &build, iter_ptr);
                let bit_index = build
                    .build_int_add(slice_offset, idx, "slice_offset_plus_index")
                    .unwrap();
                let byte_index = build
                    .build_right_shift(bit_index, i64_type.const_int(3, false), false, "byte_index")
                    .unwrap();
                let bit_in_byte_i64 = build
                    .build_and(bit_index, i64_type.const_int(7, false), "bit_in_byte_i64")
                    .unwrap();
                let bit_in_byte_i8 = build
                    .build_int_truncate(bit_in_byte_i64, ctx.i8_type(), "bit_in_byte_i8")
                    .unwrap();
                let data_byte_ptr =
                    increment_pointer!(ctx, build, data_ptr, ptype.width(), byte_index);
                let data_byte_i8 = build
                    .build_load(ctx.i8_type(), data_byte_ptr, "get_data_byte_i8")
                    .unwrap()
                    .into_int_value();
                let data_byte_shifted_i8 = build
                    .build_right_shift(data_byte_i8, bit_in_byte_i8, false, "data_byte_shifted_i8")
                    .unwrap();
                let data_bit_i8 = build
                    .build_and(
                        data_byte_shifted_i8,
                        ctx.i8_type().const_int(1, false),
                        "data_bit_i8",
                    )
                    .unwrap();
                build.build_return(Some(&data_bit_i8)).unwrap();

                Some(access_f)
            }
            IteratorHolder::Dictionary { arr, keys, values } => match &dt {
                DataType::Dictionary(_, _) => {
                    let keys_access = keys
                        .generate_random_access(ctx, llvm_mod)
                        .expect("dictionary access requires random-access keys");
                    let values_access = values
                        .generate_random_access(ctx, llvm_mod)
                        .expect("dictionary access requires random-access values");

                    declare_blocks!(ctx, access_f, entry);

                    build.position_at_end(entry);
                    let key = build
                        .build_call(
                            keys_access,
                            &[
                                arr.llvm_key_iter_ptr(ctx, &build, iter_ptr).into(),
                                idx.into(),
                            ],
                            "key",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                        .into_int_value();
                    let key_conv = build
                        .build_int_cast(key, ctx.i64_type(), "key_conv")
                        .unwrap();

                    let value = build
                        .build_call(
                            values_access,
                            &[
                                arr.llvm_val_iter_ptr(ctx, &build, iter_ptr).into(),
                                key_conv.into(),
                            ],
                            "value",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic();
                    build.build_return(Some(&value)).unwrap();

                    Some(access_f)
                }
                _ => unreachable!("dictionary iterator but non-iterator data type ({:?})", dt),
            },
            IteratorHolder::RunEnd {
                arr,
                run_ends,
                values,
            } => match &dt {
                DataType::RunEndEncoded(r_dt, _) => {
                    let runs_prim_type = PrimitiveType::for_arrow_type(r_dt.data_type());
                    let runs_t = runs_prim_type.llvm_type(ctx).into_int_type();
                    let bsearch =
                        add_bsearch(ctx, llvm_mod, run_ends.as_primitive(), runs_prim_type);
                    let value_access = values
                        .generate_random_access(ctx, llvm_mod)
                        .expect("run-end access requires random-access values");

                    declare_blocks!(ctx, access_f, entry);
                    build.position_at_end(entry);
                    let idx = build
                        .build_int_truncate_or_bit_cast(idx, runs_t, "casted_idx")
                        .unwrap();

                    let v_idx = build
                        .build_call(
                            bsearch,
                            &[
                                arr.llvm_re_iter_ptr(ctx, &build, iter_ptr).into(),
                                idx.into(),
                            ],
                            "v_idx",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                        .into_int_value();
                    let v = build
                        .build_call(
                            value_access,
                            &[
                                arr.llvm_val_iter_ptr(ctx, &build, iter_ptr).into(),
                                v_idx.into(),
                            ],
                            "v",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic();
                    build.build_return(Some(&v)).unwrap();
                    Some(access_f)
                }
                _ => unreachable!("run-end iterator but non-iterator data type ({:?})", dt),
            },
            IteratorHolder::ScalarVec(iter) => {
                let val = iter.llvm_val(ctx, &build, iter_ptr);
                build.build_return(Some(&val)).unwrap();
                Some(access_f)
            }
            IteratorHolder::SetBit(_)
            | IteratorHolder::ScalarPrimitive(_)
            | IteratorHolder::ScalarBoolean(_)
            | IteratorHolder::ScalarString(_)
            | IteratorHolder::ScalarBinary(_) => {
                unreachable!("random-access capability disagrees with iterator variant")
            }
        }
    }
}

/// Generates code to fetch the length of an iterator. Useful for calling
/// `@llvm.assume` when two iterators have equal length.
pub fn get_iterator_length<'a>(
    ctx: &'a Context,
    builder: &Builder<'a>,
    ih: &IteratorHolder,
    iter_ptr: PointerValue<'a>,
) -> Option<IntValue<'a>> {
    match ih {
        IteratorHolder::Primitive(iter) => Some(iter.llvm_len(ctx, builder, iter_ptr)),
        IteratorHolder::String(iter) => Some(iter.llvm_len(ctx, builder, iter_ptr)),
        IteratorHolder::LargeString(iter) => Some(iter.llvm_len(ctx, builder, iter_ptr)),
        IteratorHolder::View(iter) => Some(iter.llvm_len(ctx, builder, iter_ptr)),
        IteratorHolder::Bitmap(iter) => Some(iter.llvm_len(ctx, builder, iter_ptr)),
        IteratorHolder::SetBit(..) => None,
        IteratorHolder::Dictionary { arr, keys, .. } => {
            let keys_ptr = arr.llvm_key_iter_ptr(ctx, builder, iter_ptr);
            get_iterator_length(ctx, builder, keys, keys_ptr)
        }
        IteratorHolder::RunEnd { .. } => None,
        IteratorHolder::FixedSizeList(iter) => Some(iter.llvm_len(ctx, builder, iter_ptr)),
        IteratorHolder::List(iter) => Some(iter.llvm_len(ctx, builder, iter_ptr)),
        IteratorHolder::ScalarPrimitive(..) => None,
        IteratorHolder::ScalarBoolean(..) => None,
        IteratorHolder::ScalarString(..) => None,
        IteratorHolder::ScalarBinary(..) => None,
        IteratorHolder::ScalarVec(..) => None,
    }
}

#[cfg(test)]
mod codegen_cache_tests {
    use std::sync::Arc;

    use arrow_array::{
        types::{Int16Type, Int32Type, Int8Type},
        BooleanArray, DictionaryArray, Int16Array, Int32Array, Int8Array, LargeListArray,
        ListArray, RunArray,
    };
    use inkwell::context::Context;

    use super::{array_to_iter, array_to_setbit_iter};

    #[test]
    fn reuses_generated_functions_for_the_same_iterator_shape() {
        let first = Int32Array::from(vec![1, 2, 3]);
        let second = Int32Array::from(vec![4, 5, 6]);
        let first = array_to_iter(&first);
        let second = array_to_iter(&second);
        let ctx = Context::create();
        let module = ctx.create_module("iterator_codegen_cache");

        let first_access = first.generate_random_access(&ctx, &module).unwrap();
        let first_next = first.generate_next(&ctx, &module);
        let first_next_block = first.generate_next_block(&ctx, &module, 8).unwrap();
        let first_reset = first.generate_reset(&ctx, &module);
        let function_count = module.get_functions().count();

        let second_access = second.generate_random_access(&ctx, &module).unwrap();
        let second_next = second.generate_next(&ctx, &module);
        let second_next_block = second.generate_next_block(&ctx, &module, 8).unwrap();
        let second_reset = second.generate_reset(&ctx, &module);

        assert_eq!(first_access, second_access);
        assert_eq!(first_next, second_next);
        assert_eq!(first_next_block, second_next_block);
        assert_eq!(first_reset, second_reset);
        assert_eq!(module.get_functions().count(), function_count);
        module.verify().unwrap();
    }

    #[test]
    fn gives_different_iterator_shapes_different_structural_names() {
        let primitive = Int32Array::from(vec![1, 2, 3]);
        let bitmap = BooleanArray::from(vec![true, false, true]);
        let primitive = array_to_iter(&primitive);
        let bitmap = array_to_iter(&bitmap);
        let ctx = Context::create();
        let module = ctx.create_module("iterator_codegen_names");

        let primitive_access = primitive.generate_random_access(&ctx, &module).unwrap();
        let bitmap_access = bitmap.generate_random_access(&ctx, &module).unwrap();
        let primitive_name = primitive_access.get_name().to_str().unwrap();
        let bitmap_name = bitmap_access.get_name().to_str().unwrap();

        assert_eq!(primitive_name, "iterator_primitive_i32_access");
        assert_eq!(bitmap_name, "iterator_bitmap_access");
        assert_ne!(primitive_access, bitmap_access);
        module.verify().unwrap();
    }

    #[test]
    fn gives_nested_iterator_shapes_structural_names() {
        let rows = [Some(vec![Some(1), Some(2)]), Some(vec![Some(3), Some(4)])];
        let list = ListArray::from_iter_primitive::<Int32Type, _, _>(rows.clone());
        let large_list = LargeListArray::from_iter_primitive::<Int32Type, _, _>(rows);
        let dictionary =
            DictionaryArray::<Int8Type>::new(Int8Array::from(vec![0, 1]), Arc::new(list));
        let run_ends = Int16Array::from(vec![2, 4]);
        let run_end = RunArray::<Int16Type>::try_new(&run_ends, dictionary.values()).unwrap();

        let list = array_to_iter(dictionary.values().as_ref());
        let large_list = array_to_iter(&large_list);
        let dictionary = array_to_iter(&dictionary);
        let run_end = array_to_iter(&run_end);
        let ctx = Context::create();
        let module = ctx.create_module("nested_iterator_codegen_names");

        let list_access = list.generate_random_access(&ctx, &module).unwrap();
        let large_list_access = large_list.generate_random_access(&ctx, &module).unwrap();
        let dictionary_access = dictionary.generate_random_access(&ctx, &module).unwrap();
        let run_end_access = run_end.generate_random_access(&ctx, &module).unwrap();

        assert!(list_access
            .get_name()
            .to_str()
            .unwrap()
            .starts_with("iterator_list_w4_"));
        assert!(large_list_access
            .get_name()
            .to_str()
            .unwrap()
            .starts_with("iterator_list_w8_"));
        assert!(dictionary_access
            .get_name()
            .to_str()
            .unwrap()
            .starts_with("iterator_dictionary_"));
        assert!(run_end_access
            .get_name()
            .to_str()
            .unwrap()
            .starts_with("iterator_run_end_"));
        assert!(dictionary_access
            .get_type()
            .get_return_type()
            .unwrap()
            .is_struct_type());
        assert!(run_end_access
            .get_type()
            .get_return_type()
            .unwrap()
            .is_struct_type());
        dictionary.generate_next(&ctx, &module);
        run_end.generate_next(&ctx, &module);
        module.verify().unwrap();
    }

    #[test]
    fn separates_block_functions_by_width() {
        let values = Int32Array::from(vec![1, 2, 3]);
        let iter = array_to_iter(&values);
        let ctx = Context::create();
        let module = ctx.create_module("iterator_block_widths");

        let block_8 = iter.generate_next_block(&ctx, &module, 8).unwrap();
        let block_64 = iter.generate_next_block(&ctx, &module, 64).unwrap();

        assert_eq!(
            block_8.get_name().to_str().unwrap(),
            "iterator_primitive_i32_next_block_8"
        );
        assert_eq!(
            block_64.get_name().to_str().unwrap(),
            "iterator_primitive_i32_next_block_64"
        );
        assert_ne!(block_8, block_64);
        module.verify().unwrap();
    }

    #[test]
    fn unsupported_generators_do_not_add_llvm_functions() {
        let list = ListArray::from_iter_primitive::<Int32Type, _, _>([Some(vec![Some(1)])]);
        let run_ends = Int16Array::from(vec![1]);
        let run_end = RunArray::<Int16Type>::try_new(&run_ends, &list).unwrap();
        let set_bits = BooleanArray::from(vec![true, false, true]);
        let list = array_to_iter(&list);
        let run_end = array_to_iter(&run_end);
        let set_bits = array_to_setbit_iter(&set_bits).unwrap();
        let ctx = Context::create();
        let module = ctx.create_module("unsupported_iterator_codegen");

        assert!(list.generate_next_block(&ctx, &module, 8).is_none());
        assert!(run_end.generate_next_block(&ctx, &module, 8).is_none());
        assert!(set_bits.generate_random_access(&ctx, &module).is_none());
        assert_eq!(module.get_functions().count(), 0);
        module.verify().unwrap();
    }
}
