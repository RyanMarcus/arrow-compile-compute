use std::fmt::Debug;

use arrow_array::{cast::AsArray, BooleanArray, Datum};
use arrow_schema::DataType;

use super::{errors::DSLError, expressions::KernelExpression};
use crate::PrimitiveType;

#[derive(Clone)]
pub enum KernelInput<'a> {
    Datum(usize, &'a dyn Datum),
    SetBits(usize, &'a BooleanArray),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum KernelInputType {
    Standard,
    SetBit,
}

impl Debug for KernelInput<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelInput::Datum(index, datum) => {
                write!(f, "Datum({}, {:?})", index, datum.get().0.data_type())
            }
            KernelInput::SetBits(index, set_bits) => {
                write!(f, "SetBits({}, {:?})", index, set_bits)
            }
        }
    }
}

impl<'a> KernelInput<'a> {
    pub fn as_expr(&self) -> KernelExpression<'a> {
        KernelExpression::Item(self.clone())
    }

    pub fn data_type(&self) -> DataType {
        match self {
            KernelInput::Datum(_, datum) => datum.get().0.data_type().clone(),
            KernelInput::SetBits(..) => DataType::UInt64,
        }
    }

    pub fn into_set_bits(self) -> Result<KernelInput<'a>, DSLError> {
        match self {
            KernelInput::Datum(idx, datum) => {
                if datum.get().1 {
                    Err(DSLError::BooleanExpected(
                        "cannot convert scalar to set bit iterator".to_string(),
                    ))
                } else if let Some(bool) = datum.get().0.as_boolean_opt() {
                    Ok(KernelInput::SetBits(idx, bool))
                } else {
                    Err(DSLError::BooleanExpected(format!(
                        "cannot convert {} to set bit iterator",
                        datum.get().0.data_type()
                    )))
                }
            }
            KernelInput::SetBits(..) => Ok(self),
        }
    }

    pub fn index(&self) -> usize {
        match self {
            KernelInput::Datum(index, _) => *index,
            KernelInput::SetBits(index, _) => *index,
        }
    }

    pub(super) fn input_type(&self) -> KernelInputType {
        match self {
            KernelInput::Datum(_, _) => KernelInputType::Standard,
            KernelInput::SetBits(_, _) => KernelInputType::SetBit,
        }
    }

    pub fn at(&self, idx: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::At {
            iter: Box::new(self.clone()),
            idx: Box::new(idx.clone()),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DictKeyType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RunEndType {
    Int16,
    Int32,
    Int64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum KernelOutputType {
    Array,
    String,
    View,
    Boolean,
    FixedSizeList(usize),
    Dictionary(DictKeyType),
    RunEnd(RunEndType),
}

impl KernelOutputType {
    pub fn for_data_type(dt: &DataType) -> Result<KernelOutputType, DSLError> {
        match dt {
            DataType::Boolean => Ok(KernelOutputType::Boolean),
            DataType::Null
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
            | DataType::Float64
            | DataType::Timestamp(..)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(..)
            | DataType::Time64(..)
            | DataType::Duration(..)
            | DataType::Interval(..) => Ok(KernelOutputType::Array),
            DataType::Binary | DataType::LargeBinary | DataType::Utf8 | DataType::LargeUtf8 => {
                Ok(KernelOutputType::String)
            }
            DataType::BinaryView | DataType::Utf8View => Ok(KernelOutputType::View),
            DataType::FixedSizeBinary(_) => todo!(),
            DataType::List(..) => todo!(),
            DataType::ListView(..) => todo!(),
            DataType::FixedSizeList(_f, l) => Ok(KernelOutputType::FixedSizeList(*l as usize)),
            DataType::LargeList(..) => todo!(),
            DataType::LargeListView(..) => todo!(),
            DataType::Struct(..) => todo!(),
            DataType::Union(..) => todo!(),
            DataType::Dictionary(k, _v) => match k.as_ref() {
                DataType::Int8 => Ok(KernelOutputType::Dictionary(DictKeyType::Int8)),
                DataType::Int16 => Ok(KernelOutputType::Dictionary(DictKeyType::Int16)),
                DataType::Int32 => Ok(KernelOutputType::Dictionary(DictKeyType::Int32)),
                DataType::Int64 => Ok(KernelOutputType::Dictionary(DictKeyType::Int64)),
                DataType::UInt8 => Ok(KernelOutputType::Dictionary(DictKeyType::UInt8)),
                DataType::UInt16 => Ok(KernelOutputType::Dictionary(DictKeyType::UInt16)),
                DataType::UInt32 => Ok(KernelOutputType::Dictionary(DictKeyType::UInt32)),
                DataType::UInt64 => Ok(KernelOutputType::Dictionary(DictKeyType::UInt64)),
                _ => Err(DSLError::TypeMismatch(format!(
                    "dictionary key type must be an int, but has type {}",
                    dt
                ))),
            },
            DataType::Decimal128(_, _) => todo!(),
            DataType::Decimal256(_, _) => todo!(),
            DataType::Map(..) => todo!(),
            DataType::RunEndEncoded(k, _v) => match k.data_type() {
                DataType::Int16 => Ok(KernelOutputType::RunEnd(RunEndType::Int16)),
                DataType::Int32 => Ok(KernelOutputType::RunEnd(RunEndType::Int32)),
                DataType::Int64 => Ok(KernelOutputType::RunEnd(RunEndType::Int64)),
                _ => Err(DSLError::TypeMismatch(format!(
                    "run end type must i16, i32, or i64, but has type {}",
                    dt
                ))),
            },
        }
    }

    pub fn can_collect(&self, dt: &DataType) -> Result<(), DSLError> {
        match self {
            KernelOutputType::Array => match PrimitiveType::for_arrow_type(dt) {
                PrimitiveType::P64x2 | PrimitiveType::List(_, _) => {
                    return Err(DSLError::TypeMismatch(format!(
                        "array output type can only collect primitives, but expression has type {}",
                        PrimitiveType::for_arrow_type(dt)
                    )))
                }
                _ => {}
            },
            KernelOutputType::String => {
                if PrimitiveType::for_arrow_type(dt) != PrimitiveType::P64x2 {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot collect type {} into string",
                        dt
                    )));
                }
            }
            KernelOutputType::Boolean => {
                if dt != &DataType::Boolean && dt != &DataType::UInt8 {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot collect type {} into boolean",
                        dt
                    )));
                }
            }
            KernelOutputType::View => {
                if PrimitiveType::for_arrow_type(dt) != PrimitiveType::P64x2 {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot collect type {} into view",
                        dt
                    )));
                }
            }
            KernelOutputType::Dictionary(_) => {}
            KernelOutputType::RunEnd(_) => {}
            KernelOutputType::FixedSizeList(l) => {
                return match PrimitiveType::for_arrow_type(dt) {
                    PrimitiveType::List(_, s) if s == *l => Ok(()),
                    _ => Err(DSLError::TypeMismatch(format!(
                        "cannot collect type {} into fixed size list<{}>",
                        dt, l
                    ))),
                }
            }
        }

        Ok(())
    }
}

pub fn base_type(dt: &DataType) -> DataType {
    match dt {
        DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View => DataType::Utf8,
        DataType::Dictionary(_k, v) => *v.clone(),
        DataType::RunEndEncoded(_re, v) => v.data_type().clone(),
        _ => dt.clone(),
    }
}
