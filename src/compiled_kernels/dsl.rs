use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    ffi::c_void,
    fmt::Debug,
    sync::Arc,
};

use arrow_array::{
    cast::AsArray,
    types::{
        ArrowDictionaryKeyType, Int16Type, Int32Type, Int64Type, Int8Type, RunEndIndexType,
        UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrayRef, BooleanArray, Datum, DictionaryArray, RunArray, StringArray,
};
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    module::{Linkage, Module},
    types::{BasicType, BasicTypeEnum},
    values::{BasicValue, BasicValueEnum, FunctionValue, PointerValue, VectorValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};
use itertools::Itertools;
use ouroboros::self_referencing;
use repr_offset::ReprOffset;
use thiserror::Error;

use crate::{
    compiled_iter::{
        array_to_setbit_iter, datum_to_iter, generate_next, generate_next_block,
        generate_random_access, IteratorHolder,
    },
    compiled_kernels::{
        cmp::{add_float_vec_to_int_vec, add_memcmp},
        gen_convert_numeric_vec, link_req_helpers, optimize_module,
        writers::{
            ArrayWriter, BooleanWriter, DictWriter, PrimitiveArrayWriter, REEWriter,
            StringViewWriter, WriterAllocation,
        },
    },
    declare_blocks, increment_pointer, set_noalias_params, ComparisonType, Predicate,
    PrimitiveType,
};

use super::{writers::StringArrayWriter, ArrowKernelError};

const VEC_SIZE: u32 = 32;

#[derive(Debug, Error)]
pub enum DSLError {
    #[error("Invalid input index: {0}")]
    InvalidInputIndex(usize),
    #[error("Invalid kernel output length: {0}")]
    InvalidKernelOutputLength(usize),
    #[error("Type mismatch: {0}")]
    TypeMismatch(String),
    #[error("Boolean expected: {0}")]
    BooleanExpected(String),
    #[error("Unused input: {0}")]
    UnusedInput(String),
    #[error("Unsupported dictionary value type: {0}")]
    UnsupportedDictionaryValueType(DataType),
    #[error("No iteration")]
    NoIteration,
    #[error("Not vectorizable: {0}")]
    NotVectorizable(&'static str),
}

#[derive(Clone)]
pub enum KernelInput<'a> {
    Datum(usize, &'a dyn Datum),
    SetBits(usize, &'a BooleanArray),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum KernelInputType {
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
    fn as_expr(&self) -> KernelExpression<'a> {
        KernelExpression::Item(self.clone())
    }

    fn data_type(&self) -> DataType {
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

    fn index(&self) -> usize {
        match self {
            KernelInput::Datum(index, _) => *index,
            KernelInput::SetBits(index, _) => *index,
        }
    }

    fn input_type(&self) -> KernelInputType {
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
            DataType::FixedSizeList(..) => todo!(),
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
    /// Determines if this output type can collect the passed data type. For
    /// example, the `Boolean` output type can only collect expressions that
    /// result in booleans.
    fn can_collect(&self, dt: &DataType) -> Result<(), DSLError> {
        match self {
            KernelOutputType::Array => {
                if PrimitiveType::for_arrow_type(dt) == PrimitiveType::P64x2 {
                    return Err(DSLError::TypeMismatch(format!(
                        "array output type can only collect primitives, but expression has type {}",
                        dt
                    )));
                }
            }
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
        };

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

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum KernelExpression<'a> {
    Item(KernelInput<'a>),
    IntConst(u64, bool),
    Truncate(Box<KernelExpression<'a>>, usize),
    Cmp(
        Predicate,
        Box<KernelExpression<'a>>,
        Box<KernelExpression<'a>>,
    ),
    And(Box<KernelExpression<'a>>, Box<KernelExpression<'a>>),
    Or(Box<KernelExpression<'a>>, Box<KernelExpression<'a>>),
    Not(Box<KernelExpression<'a>>),

    Select {
        cond: Box<KernelExpression<'a>>,
        v1: Box<KernelExpression<'a>>,
        v2: Box<KernelExpression<'a>>,
    },
    At {
        iter: Box<KernelInput<'a>>,
        idx: Box<KernelExpression<'a>>,
    },
    Convert(Box<KernelExpression<'a>>, PrimitiveType),
    Add(Box<KernelExpression<'a>>, Box<KernelExpression<'a>>),
    Sub(Box<KernelExpression<'a>>, Box<KernelExpression<'a>>),
    Mul(Box<KernelExpression<'a>>, Box<KernelExpression<'a>>),
    Div(Box<KernelExpression<'a>>, Box<KernelExpression<'a>>),
    Rem(Box<KernelExpression<'a>>, Box<KernelExpression<'a>>),
}

impl From<u64> for KernelExpression<'_> {
    fn from(value: u64) -> Self {
        KernelExpression::IntConst(value, false)
    }
}

impl From<i64> for KernelExpression<'_> {
    fn from(value: i64) -> Self {
        KernelExpression::IntConst(u64::from_le_bytes(value.to_le_bytes()), true)
    }
}

impl From<i32> for KernelExpression<'_> {
    fn from(value: i32) -> Self {
        (value as i64).into()
    }
}

#[allow(dead_code)]
impl<'a> KernelExpression<'a> {
    pub fn cmp(&self, pred: Predicate, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::Cmp(pred, Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn gt(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        self.cmp(Predicate::Gt, other)
    }
    pub fn eq(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        self.cmp(Predicate::Eq, other)
    }
    pub fn lt(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        self.cmp(Predicate::Lt, other)
    }
    pub fn or(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::Or(Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn and(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::And(Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn not(&self) -> KernelExpression<'a> {
        KernelExpression::Not(Box::new(self.clone()))
    }
    pub fn truncate(&self, size: usize) -> KernelExpression<'a> {
        KernelExpression::Truncate(Box::new(self.clone()), size)
    }
    pub fn select(
        &self,
        a: &KernelExpression<'a>,
        b: &KernelExpression<'a>,
    ) -> KernelExpression<'a> {
        KernelExpression::Select {
            cond: Box::new(self.clone()),
            v1: Box::new(a.clone()),
            v2: Box::new(b.clone()),
        }
    }
    pub fn convert(&self, pt: PrimitiveType) -> KernelExpression<'a> {
        KernelExpression::Convert(Box::new(self.clone()), pt)
    }
    pub fn add(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::Add(Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn sub(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::Sub(Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn div(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::Div(Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn rem(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::Rem(Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn mul(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::Mul(Box::new(self.clone()), Box::new(other.clone()))
    }

    fn descend<F: FnMut(&Self)>(&self, f: &mut F) {
        match self {
            KernelExpression::Item(..) => f(self),
            KernelExpression::Truncate(e, _) => {
                f(self);
                e.descend(f);
            }
            KernelExpression::Cmp(.., lhs, rhs) => {
                f(self);
                lhs.descend(f);
                rhs.descend(f);
            }
            KernelExpression::And(lhs, rhs) => {
                f(self);
                lhs.descend(f);
                rhs.descend(f);
            }
            KernelExpression::Or(lhs, rhs) => {
                f(self);
                lhs.descend(f);
                rhs.descend(f);
            }
            KernelExpression::Not(c) => {
                f(self);
                c.descend(f);
            }
            KernelExpression::Select { cond, v1, v2 } => {
                f(self);
                cond.descend(f);
                v1.descend(f);
                v2.descend(f);
            }
            KernelExpression::At { idx, .. } => {
                f(self);
                idx.descend(f);
            }
            KernelExpression::Convert(expr, ..) => {
                f(self);
                expr.descend(f);
            }
            KernelExpression::IntConst(..) => f(self),
            KernelExpression::Add(lhs, rhs)
            | KernelExpression::Sub(lhs, rhs)
            | KernelExpression::Mul(lhs, rhs)
            | KernelExpression::Div(lhs, rhs)
            | KernelExpression::Rem(lhs, rhs) => {
                f(self);
                lhs.descend(f);
                rhs.descend(f);
            }
        }
    }

    fn iterated_indexes(&self) -> Vec<(KernelInputType, usize)> {
        let mut h = BTreeSet::new();
        self.descend(&mut |e| {
            if let KernelExpression::Item(kernel_input) = e {
                h.insert((kernel_input.input_type(), kernel_input.index()));
            }
        });
        h.into_iter().collect()
    }

    fn accessed_indexes(&self) -> Vec<usize> {
        let mut h = BTreeSet::new();
        self.descend(&mut |e| {
            if let KernelExpression::At { iter, .. } = e {
                h.insert(iter.index());
            }
        });
        h.into_iter().collect()
    }

    fn get_type(&self) -> DataType {
        match self {
            KernelExpression::Item(kernel_input) => base_type(&kernel_input.data_type()),
            KernelExpression::Truncate(..) => DataType::Binary,
            KernelExpression::Select { v1, .. } => v1.get_type(),
            KernelExpression::Cmp(..)
            | KernelExpression::And(..)
            | KernelExpression::Or(..)
            | KernelExpression::Not(..) => DataType::Boolean,
            KernelExpression::IntConst(..) => DataType::UInt64,
            KernelExpression::At { iter, .. } => base_type(&iter.data_type()),
            KernelExpression::Convert(_expr, pt) => pt.as_arrow_type(),
            KernelExpression::Add(lhs, _rhs)
            | KernelExpression::Sub(lhs, _rhs)
            | KernelExpression::Mul(lhs, _rhs)
            | KernelExpression::Div(lhs, _rhs)
            | KernelExpression::Rem(lhs, _rhs) => lhs.get_type(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_block<'b>(
        &self,
        ctx: &'b Context,
        build: &Builder<'b>,
        vec_bufs: &HashMap<usize, PointerValue<'b>>,
        iter_llvm_types: &HashMap<usize, BasicTypeEnum<'b>>,
    ) -> Result<VectorValue<'b>, DSLError> {
        match self {
            KernelExpression::Item(kernel_input) => {
                let buf = vec_bufs[&kernel_input.index()];
                let llvm_type = iter_llvm_types[&kernel_input.index()];
                Ok(build
                    .build_load(llvm_type, buf, "load")
                    .unwrap()
                    .into_vector_value())
            }
            KernelExpression::IntConst(v, s) => {
                let i64_type = ctx.i64_type();
                let vec_type = i64_type.vec_type(VEC_SIZE);
                let v = i64_type.const_int(*v, *s);
                let v = build
                    .build_insert_element(vec_type.const_zero(), v, i64_type.const_zero(), "v")
                    .unwrap();
                let v = build
                    .build_shuffle_vector(
                        v,
                        vec_type.const_zero(),
                        vec_type.const_zero(),
                        "splatted",
                    )
                    .unwrap();
                Ok(v)
            }
            KernelExpression::Truncate(_kernel_expression, _) => todo!(),
            KernelExpression::And(lhs, rhs) => {
                let lhs = lhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                let rhs = rhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                Ok(build.build_and(lhs, rhs, "and").unwrap())
            }
            KernelExpression::Or(lhs, rhs) => {
                let lhs = lhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                let rhs = rhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                Ok(build.build_or(lhs, rhs, "or").unwrap())
            }
            KernelExpression::Not(c) => {
                let c = c.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                Ok(build.build_not(c, "not").unwrap())
            }
            KernelExpression::Select { cond, v1, v2 } => {
                let cond = cond.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                let v1 = v1.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                let v2 = v2.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                Ok(build
                    .build_select(cond, v1, v2, "select")
                    .unwrap()
                    .into_vector_value())
            }
            KernelExpression::Convert(c, tar_pt) => {
                let in_ty = PrimitiveType::for_arrow_type(&c.get_type());
                let c = c.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                Ok(gen_convert_numeric_vec(ctx, build, c, in_ty, *tar_pt))
            }
            KernelExpression::Add(lhs, rhs) => {
                let pt = PrimitiveType::for_arrow_type(&lhs.get_type());
                if pt != PrimitiveType::for_arrow_type(&rhs.get_type()) {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot add values of different types {} and {}",
                        pt,
                        PrimitiveType::for_arrow_type(&rhs.get_type())
                    )));
                }

                let lhs = lhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                let rhs = rhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                match pt {
                    PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64
                    | PrimitiveType::U8
                    | PrimitiveType::U16
                    | PrimitiveType::U32
                    | PrimitiveType::U64 => Ok(build.build_int_add(lhs, rhs, "add").unwrap()),
                    PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                        Ok(build.build_float_add(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveType::P64x2 => Err(DSLError::TypeMismatch(
                        "cannot add string types".to_string(),
                    )),
                }
            }
            KernelExpression::Sub(lhs, rhs) => {
                let pt = PrimitiveType::for_arrow_type(&lhs.get_type());
                if pt != PrimitiveType::for_arrow_type(&rhs.get_type()) {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot subtract values of different types {} and {}",
                        pt,
                        PrimitiveType::for_arrow_type(&rhs.get_type())
                    )));
                }

                let lhs = lhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                let rhs = rhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                match pt {
                    PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64
                    | PrimitiveType::U8
                    | PrimitiveType::U16
                    | PrimitiveType::U32
                    | PrimitiveType::U64 => Ok(build.build_int_sub(lhs, rhs, "add").unwrap()),
                    PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                        Ok(build.build_float_sub(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveType::P64x2 => Err(DSLError::TypeMismatch(
                        "cannot subtract string types".to_string(),
                    )),
                }
            }
            KernelExpression::Mul(lhs, rhs) => {
                let pt = PrimitiveType::for_arrow_type(&lhs.get_type());
                if pt != PrimitiveType::for_arrow_type(&rhs.get_type()) {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot multiply values of different types {} and {}",
                        pt,
                        PrimitiveType::for_arrow_type(&rhs.get_type())
                    )));
                }

                let lhs = lhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                let rhs = rhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                match pt {
                    PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64
                    | PrimitiveType::U8
                    | PrimitiveType::U16
                    | PrimitiveType::U32
                    | PrimitiveType::U64 => Ok(build.build_int_mul(lhs, rhs, "mul").unwrap()),
                    PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                        Ok(build.build_float_mul(lhs, rhs, "mul").unwrap())
                    }
                    PrimitiveType::P64x2 => Err(DSLError::TypeMismatch(
                        "cannot multiply string types".to_string(),
                    )),
                }
            }
            KernelExpression::Div(lhs, rhs) => {
                let pt = PrimitiveType::for_arrow_type(&lhs.get_type());
                if pt != PrimitiveType::for_arrow_type(&rhs.get_type()) {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot divide values of different types {} and {}",
                        pt,
                        PrimitiveType::for_arrow_type(&rhs.get_type())
                    )));
                }

                let lhs = lhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                let rhs = rhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                match pt {
                    PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64 => {
                        Ok(build.build_int_signed_div(lhs, rhs, "div").unwrap())
                    }
                    PrimitiveType::U8
                    | PrimitiveType::U16
                    | PrimitiveType::U32
                    | PrimitiveType::U64 => {
                        Ok(build.build_int_unsigned_div(lhs, rhs, "div").unwrap())
                    }
                    PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                        Ok(build.build_float_div(lhs, rhs, "div").unwrap())
                    }
                    PrimitiveType::P64x2 => Err(DSLError::TypeMismatch(
                        "cannot divide string types".to_string(),
                    )),
                }
            }
            KernelExpression::Rem(lhs, rhs) => {
                let pt = PrimitiveType::for_arrow_type(&lhs.get_type());
                if pt != PrimitiveType::for_arrow_type(&rhs.get_type()) {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot rem values of different types {} and {}",
                        pt,
                        PrimitiveType::for_arrow_type(&rhs.get_type())
                    )));
                }

                let lhs = lhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                let rhs = rhs.compile_block(ctx, build, vec_bufs, iter_llvm_types)?;
                match pt {
                    PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64 => {
                        Ok(build.build_int_signed_rem(lhs, rhs, "rem").unwrap())
                    }
                    PrimitiveType::U8
                    | PrimitiveType::U16
                    | PrimitiveType::U32
                    | PrimitiveType::U64 => {
                        Ok(build.build_int_unsigned_rem(lhs, rhs, "rem").unwrap())
                    }
                    PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                        Ok(build.build_float_rem(lhs, rhs, "rem").unwrap())
                    }
                    PrimitiveType::P64x2 => Err(DSLError::TypeMismatch(
                        "cannot rem string types".to_string(),
                    )),
                }
            }
            KernelExpression::Cmp(..) => Err(DSLError::NotVectorizable("cmp operator")),
            KernelExpression::At { .. } => Err(DSLError::NotVectorizable("at operator")),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compile<'b>(
        &self,
        ctx: &'b Context,
        llvm_mod: &Module<'b>,
        build: &Builder<'b>,
        bufs: &HashMap<usize, PointerValue<'b>>,
        accessors: &HashMap<usize, FunctionValue<'b>>,
        iter_ptrs: &[PointerValue<'b>],
        iter_llvm_types: &HashMap<usize, BasicTypeEnum<'b>>,
    ) -> Result<BasicValueEnum<'b>, DSLError> {
        match self {
            KernelExpression::Item(kernel_input) => {
                let buf = bufs[&kernel_input.index()];
                let llvm_type = iter_llvm_types[&kernel_input.index()];
                Ok(build.build_load(llvm_type, buf, "load").unwrap())
            }
            KernelExpression::Truncate(_kernel_expression, _) => todo!(),
            KernelExpression::Cmp(predicate, lhs, rhs) => {
                let lhs_v = lhs.compile(
                    ctx,
                    llvm_mod,
                    build,
                    bufs,
                    accessors,
                    iter_ptrs,
                    iter_llvm_types,
                )?;
                let rhs_v = rhs.compile(
                    ctx,
                    llvm_mod,
                    build,
                    bufs,
                    accessors,
                    iter_ptrs,
                    iter_llvm_types,
                )?;

                let lhs_ptype = PrimitiveType::for_arrow_type(&lhs.get_type());
                let rhs_ptype = PrimitiveType::for_arrow_type(&rhs.get_type());

                let res = match (lhs_ptype.comparison_type(), rhs_ptype.comparison_type()) {
                    (ComparisonType::String, ComparisonType::String) => {
                        let memcmp = add_memcmp(ctx, llvm_mod);
                        let res = build
                            .build_call(memcmp, &[lhs_v.into(), rhs_v.into()], "memcmp_res")
                            .unwrap()
                            .try_as_basic_value()
                            .unwrap_left()
                            .into_int_value();
                        build
                            .build_int_compare(
                                predicate.as_int_pred(true),
                                res,
                                ctx.i64_type().const_zero(),
                                "cmp_res",
                            )
                            .unwrap()
                            .as_basic_value_enum()
                    }
                    (ComparisonType::String, _) | (_, ComparisonType::String) => {
                        return Err(DSLError::TypeMismatch(format!(
                            "cannot compare {} and {}",
                            lhs.get_type(),
                            rhs.get_type()
                        )));
                    }
                    _ => {
                        let dom_type = PrimitiveType::dominant(lhs_ptype, rhs_ptype).unwrap();
                        let lhs_v = build
                            .build_bit_cast(
                                lhs_v,
                                lhs_ptype.llvm_vec_type(ctx, 1).unwrap(),
                                "single_vec_lhs",
                            )
                            .unwrap()
                            .into_vector_value();
                        let rhs_v = build
                            .build_bit_cast(
                                rhs_v,
                                rhs_ptype.llvm_vec_type(ctx, 1).unwrap(),
                                "single_vec_lhs",
                            )
                            .unwrap()
                            .into_vector_value();
                        let clhs = gen_convert_numeric_vec(ctx, build, lhs_v, lhs_ptype, dom_type);
                        let crhs = gen_convert_numeric_vec(ctx, build, rhs_v, rhs_ptype, dom_type);

                        let vec_res = match dom_type.comparison_type() {
                            ComparisonType::Int { signed } => build
                                .build_int_compare(predicate.as_int_pred(signed), clhs, crhs, "cmp")
                                .unwrap(),
                            ComparisonType::Float => {
                                let convert = add_float_vec_to_int_vec(
                                    ctx,
                                    llvm_mod,
                                    lhs_v.get_type().get_size(),
                                    dom_type,
                                );
                                let lhs = build
                                    .build_call(convert, &[clhs.into()], "lhs_converted")
                                    .unwrap()
                                    .try_as_basic_value()
                                    .unwrap_left()
                                    .into_vector_value();
                                let rhs = build
                                    .build_call(convert, &[crhs.into()], "rhs_converted")
                                    .unwrap()
                                    .try_as_basic_value()
                                    .unwrap_left()
                                    .into_vector_value();
                                build
                                    .build_int_compare(predicate.as_int_pred(true), lhs, rhs, "cmp")
                                    .unwrap()
                            }
                            ComparisonType::String => unreachable!(),
                        };

                        build
                            .build_bit_cast(vec_res, ctx.bool_type(), "sing_vec_to_bool")
                            .unwrap()
                    }
                };

                Ok(res.as_basic_value_enum())
            }
            KernelExpression::And(lhs, rhs) => {
                let lhs_v = lhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )?
                    .into_int_value();
                let rhs_v = rhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )?
                    .into_int_value();
                Ok(build.build_and(lhs_v, rhs_v, "and").unwrap().into())
            }
            KernelExpression::Or(lhs, rhs) => {
                let lhs_v = lhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )?
                    .into_int_value();
                let rhs_v = rhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )?
                    .into_int_value();
                Ok(build.build_or(lhs_v, rhs_v, "or").unwrap().into())
            }
            KernelExpression::Not(e) => {
                let e = e
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )?
                    .into_int_value();
                Ok(build.build_not(e, "not").unwrap().into())
            }
            KernelExpression::Select { cond, v1, v2 } => {
                if cond.get_type() != DataType::Boolean {
                    return Err(DSLError::BooleanExpected(format!(
                        "first parameter to select should be a boolean, found {:?}",
                        cond.get_type()
                    )));
                }
                let cond_v = cond
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )?
                    .into_int_value();
                let a_v = v1.compile(
                    ctx,
                    llvm_mod,
                    build,
                    bufs,
                    accessors,
                    iter_ptrs,
                    iter_llvm_types,
                )?;
                let b_v = v2.compile(
                    ctx,
                    llvm_mod,
                    build,
                    bufs,
                    accessors,
                    iter_ptrs,
                    iter_llvm_types,
                )?;

                if a_v.get_type() != b_v.get_type() {
                    return Err(DSLError::TypeMismatch(format!(
                        "select operands must have the same type (saw {} and {})",
                        v1.get_type(),
                        v2.get_type()
                    )));
                }

                Ok(build.build_select(cond_v, a_v, b_v, "selection").unwrap())
            }
            KernelExpression::At { iter, idx } => {
                let acessor = accessors[&iter.index()];
                let iter_ptr = iter_ptrs[iter.index()];
                let idx_type = idx.get_type();
                if !idx_type.is_integer() {
                    return Err(DSLError::TypeMismatch(format!(
                        "at parameter must be integer, got {}",
                        idx.get_type()
                    )));
                }
                let idx = idx
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )?
                    .into_int_value();
                let idx = build
                    .build_int_z_extend_or_bit_cast(idx, ctx.i64_type(), "zext")
                    .unwrap();
                Ok(build
                    .build_call(acessor, &[iter_ptr.into(), idx.into()], "at")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left())
            }
            KernelExpression::Convert(expr, pt) => {
                let to_convert = expr.compile(
                    ctx,
                    llvm_mod,
                    build,
                    bufs,
                    accessors,
                    iter_ptrs,
                    iter_llvm_types,
                )?;
                let src_pt = PrimitiveType::for_arrow_type(&expr.get_type());
                let dst_pt = *pt;

                if src_pt == dst_pt {
                    Ok(to_convert)
                } else {
                    let singleton_vec = build
                        .build_bit_cast(
                            to_convert,
                            src_pt.llvm_vec_type(ctx, 1).unwrap(),
                            "singleton_vec",
                        )
                        .unwrap()
                        .into_vector_value();

                    let res = gen_convert_numeric_vec(ctx, build, singleton_vec, src_pt, dst_pt);
                    Ok(build
                        .build_bit_cast(res, dst_pt.llvm_type(ctx), "converted_val")
                        .unwrap())
                }
            }
            KernelExpression::IntConst(v, s) => Ok(ctx.i64_type().const_int(*v, *s).into()),
            KernelExpression::Add(lhs, rhs) => {
                let lhs_pt = PrimitiveType::for_arrow_type(&lhs.get_type());
                let rhs_pt = PrimitiveType::for_arrow_type(&rhs.get_type());

                if lhs_pt != rhs_pt {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot add different types: {} and {}",
                        lhs_pt, rhs_pt
                    )));
                }

                let lhs = lhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )
                    .unwrap();
                let rhs = rhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )
                    .unwrap();

                match lhs_pt {
                    PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64
                    | PrimitiveType::U8
                    | PrimitiveType::U16
                    | PrimitiveType::U32
                    | PrimitiveType::U64 => Ok(build
                        .build_int_add(lhs.into_int_value(), rhs.into_int_value(), "add_int")
                        .unwrap()
                        .into()),
                    PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => Ok(build
                        .build_float_add(
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "add_float",
                        )
                        .unwrap()
                        .into()),
                    PrimitiveType::P64x2 => Err(DSLError::TypeMismatch(
                        "cannot add string types".to_string(),
                    )),
                }
            }
            KernelExpression::Div(lhs, rhs) => {
                let lhs_pt = PrimitiveType::for_arrow_type(&lhs.get_type());
                let rhs_pt = PrimitiveType::for_arrow_type(&rhs.get_type());

                if lhs_pt != rhs_pt {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot divide different types: {} and {}",
                        lhs_pt, rhs_pt
                    )));
                }

                let lhs = lhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )
                    .unwrap();
                let rhs = rhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )
                    .unwrap();

                match lhs_pt {
                    PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64 => Ok(build
                        .build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "div_int")
                        .unwrap()
                        .into()),
                    PrimitiveType::U8
                    | PrimitiveType::U16
                    | PrimitiveType::U32
                    | PrimitiveType::U64 => Ok(build
                        .build_int_unsigned_div(
                            lhs.into_int_value(),
                            rhs.into_int_value(),
                            "div_int",
                        )
                        .unwrap()
                        .into()),
                    PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => Ok(build
                        .build_float_div(
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "div_float",
                        )
                        .unwrap()
                        .into()),
                    PrimitiveType::P64x2 => Err(DSLError::TypeMismatch(
                        "cannot divide string types".to_string(),
                    )),
                }
            }
            KernelExpression::Mul(lhs, rhs) => {
                let lhs_pt = PrimitiveType::for_arrow_type(&lhs.get_type());
                let rhs_pt = PrimitiveType::for_arrow_type(&rhs.get_type());

                if lhs_pt != rhs_pt {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot multiply different types: {} and {}",
                        lhs_pt, rhs_pt
                    )));
                }

                let lhs = lhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )
                    .unwrap();
                let rhs = rhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )
                    .unwrap();

                match lhs_pt {
                    PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64
                    | PrimitiveType::U8
                    | PrimitiveType::U16
                    | PrimitiveType::U32
                    | PrimitiveType::U64 => Ok(build
                        .build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "mul_int")
                        .unwrap()
                        .into()),
                    PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => Ok(build
                        .build_float_mul(
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "mul_float",
                        )
                        .unwrap()
                        .into()),
                    PrimitiveType::P64x2 => Err(DSLError::TypeMismatch(
                        "cannot multiply string types".to_string(),
                    )),
                }
            }
            KernelExpression::Rem(lhs, rhs) => {
                let lhs_pt = PrimitiveType::for_arrow_type(&lhs.get_type());
                let rhs_pt = PrimitiveType::for_arrow_type(&rhs.get_type());

                if lhs_pt != rhs_pt {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot rem different types: {} and {}",
                        lhs_pt, rhs_pt
                    )));
                }

                let lhs = lhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )
                    .unwrap();
                let rhs = rhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )
                    .unwrap();

                match lhs_pt {
                    PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64 => Ok(build
                        .build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "rem_int")
                        .unwrap()
                        .into()),
                    PrimitiveType::U8
                    | PrimitiveType::U16
                    | PrimitiveType::U32
                    | PrimitiveType::U64 => Ok(build
                        .build_int_unsigned_rem(
                            lhs.into_int_value(),
                            rhs.into_int_value(),
                            "rem_int",
                        )
                        .unwrap()
                        .into()),
                    PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => Ok(build
                        .build_float_rem(
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "rem_float",
                        )
                        .unwrap()
                        .into()),
                    PrimitiveType::P64x2 => Err(DSLError::TypeMismatch(
                        "cannot compute remainder of string types".to_string(),
                    )),
                }
            }
            KernelExpression::Sub(lhs, rhs) => {
                let lhs_pt = PrimitiveType::for_arrow_type(&lhs.get_type());
                let rhs_pt = PrimitiveType::for_arrow_type(&rhs.get_type());

                if lhs_pt != rhs_pt {
                    return Err(DSLError::TypeMismatch(format!(
                        "cannot sub different types: {} and {}",
                        lhs_pt, rhs_pt
                    )));
                }

                let lhs = lhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )
                    .unwrap();
                let rhs = rhs
                    .compile(
                        ctx,
                        llvm_mod,
                        build,
                        bufs,
                        accessors,
                        iter_ptrs,
                        iter_llvm_types,
                    )
                    .unwrap();

                match lhs_pt {
                    PrimitiveType::I8
                    | PrimitiveType::I16
                    | PrimitiveType::I32
                    | PrimitiveType::I64
                    | PrimitiveType::U8
                    | PrimitiveType::U16
                    | PrimitiveType::U32
                    | PrimitiveType::U64 => Ok(build
                        .build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "sub_int")
                        .unwrap()
                        .into()),
                    PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => Ok(build
                        .build_float_sub(
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "sub_float",
                        )
                        .unwrap()
                        .into()),
                    PrimitiveType::P64x2 => Err(DSLError::TypeMismatch(
                        "cannot subtract string types".to_string(),
                    )),
                }
            }
        }
    }
}

pub struct KernelContext<'a> {
    inputs: Vec<KernelInput<'a>>,
}

impl<'a> KernelContext<'a> {
    pub fn get_input(&self, idx: usize) -> Result<KernelInput<'a>, DSLError> {
        self.inputs
            .get(idx)
            .cloned()
            .ok_or(DSLError::InvalidInputIndex(idx))
    }

    pub fn iter_over(&self, inputs: Vec<KernelInput<'a>>) -> BaseKernelProgram<'a> {
        BaseKernelProgram { inputs }
    }
}

pub struct BaseKernelProgram<'a> {
    inputs: Vec<KernelInput<'a>>,
}

#[allow(dead_code)]
impl<'a> BaseKernelProgram<'a> {
    pub fn map<F: Fn(&[KernelExpression<'a>]) -> Vec<KernelExpression<'a>>>(
        self,
        f: F,
    ) -> MappedKernelProgram<'a> {
        let exprs = self.inputs.iter().map(|inp| inp.as_expr()).collect_vec();
        MappedKernelProgram {
            inputs: self.inputs,
            expr: f(&exprs),
        }
    }

    pub fn filter<F: Fn(&[KernelExpression<'a>]) -> KernelExpression<'a>>(
        self,
        f: F,
    ) -> FilteredKernelProgram<'a> {
        let exprs = self.inputs.iter().map(|inp| inp.as_expr()).collect_vec();
        FilteredKernelProgram {
            inputs: self.inputs,
            cond: f(&exprs),
        }
    }

    pub fn collect(self, strategy: KernelOutputType) -> Result<SealedKernelProgram<'a>, DSLError> {
        SealedKernelProgram::try_new(self.inputs, None, None, strategy)
    }
}

pub struct FilteredKernelProgram<'a> {
    inputs: Vec<KernelInput<'a>>,
    cond: KernelExpression<'a>,
}

#[allow(dead_code)]
impl<'a> FilteredKernelProgram<'a> {
    pub fn map<F: Fn(&[KernelExpression<'a>]) -> Vec<KernelExpression<'a>>>(
        self,
        f: F,
    ) -> FilterMappedKernelProgram<'a> {
        let exprs = self.inputs.iter().map(|inp| inp.as_expr()).collect_vec();
        FilterMappedKernelProgram {
            inputs: self.inputs,
            cond: self.cond,
            expr: f(&exprs),
        }
    }

    pub fn collect(self, strategy: KernelOutputType) -> Result<SealedKernelProgram<'a>, DSLError> {
        SealedKernelProgram::try_new(self.inputs, Some(self.cond), None, strategy)
    }
}

pub struct MappedKernelProgram<'a> {
    inputs: Vec<KernelInput<'a>>,
    expr: Vec<KernelExpression<'a>>,
}

#[allow(dead_code)]
impl<'a> MappedKernelProgram<'a> {
    pub fn map<F: Fn(&[KernelExpression<'a>]) -> Vec<KernelExpression<'a>>>(
        self,
        f: F,
    ) -> MappedKernelProgram<'a> {
        MappedKernelProgram {
            inputs: self.inputs,
            expr: f(&self.expr),
        }
    }

    pub fn collect(self, strategy: KernelOutputType) -> Result<SealedKernelProgram<'a>, DSLError> {
        SealedKernelProgram::try_new(self.inputs, None, Some(self.expr), strategy)
    }
}

pub struct FilterMappedKernelProgram<'a> {
    inputs: Vec<KernelInput<'a>>,
    cond: KernelExpression<'a>,
    expr: Vec<KernelExpression<'a>>,
}

#[allow(dead_code)]
impl<'a> FilterMappedKernelProgram<'a> {
    pub fn map<F: Fn(&[KernelExpression<'a>]) -> Vec<KernelExpression<'a>>>(
        self,
        f: F,
    ) -> FilterMappedKernelProgram<'a> {
        FilterMappedKernelProgram {
            inputs: self.inputs,
            cond: self.cond,
            expr: f(&self.expr),
        }
    }

    pub fn collect(self, strategy: KernelOutputType) -> Result<SealedKernelProgram<'a>, DSLError> {
        SealedKernelProgram::try_new(self.inputs, Some(self.cond), Some(self.expr), strategy)
    }
}

pub struct SealedKernelProgram<'a> {
    _inputs: Vec<KernelInput<'a>>,
    cond: Option<KernelExpression<'a>>,
    expr: KernelExpression<'a>,
    strategy: KernelOutputType,
    out_type: DataType,
}

impl<'a> SealedKernelProgram<'a> {
    pub fn try_new(
        inputs: Vec<KernelInput<'a>>,
        cond: Option<KernelExpression<'a>>,
        expr: Option<Vec<KernelExpression<'a>>>,
        strategy: KernelOutputType,
    ) -> Result<Self, DSLError> {
        let expr = if let Some(mut expr) = expr {
            if expr.len() != 1 {
                return Err(DSLError::InvalidKernelOutputLength(expr.len()));
            }
            expr.pop().unwrap()
        } else {
            if inputs.len() != 1 {
                return Err(DSLError::InvalidKernelOutputLength(inputs.len()));
            }
            inputs[0].as_expr()
        };

        if let Some(cond) = &cond {
            if cond.get_type() != DataType::Boolean {
                return Err(DSLError::BooleanExpected(format!(
                    "filter should have a boolean type, found {}",
                    cond.get_type()
                )));
            }
        }

        let out_type = expr.get_type();
        let res = Self {
            _inputs: inputs,
            cond,
            expr,
            strategy,
            out_type,
        };
        res.strategy().can_collect(res.out_type())?;

        Ok(res)
    }

    pub fn out_type(&self) -> &DataType {
        &self.out_type
    }

    pub fn strategy(&self) -> KernelOutputType {
        self.strategy
    }

    fn iterated_indexes(&self) -> Vec<(KernelInputType, usize)> {
        let mut set = BTreeSet::new();
        if let Some(cond) = &self.cond {
            set.extend(cond.iterated_indexes());
        }
        set.extend(self.expr.iterated_indexes());
        set.into_iter().collect()
    }

    pub fn accessed_indexes(&self) -> Vec<usize> {
        let mut set = BTreeSet::new();
        if let Some(cond) = &self.cond {
            set.extend(cond.accessed_indexes());
        }
        set.extend(self.expr.accessed_indexes());

        set.into_iter().collect()
    }

    pub fn expr(&self) -> &KernelExpression<'a> {
        &self.expr
    }

    pub fn filter(&self) -> Option<&KernelExpression<'a>> {
        self.cond.as_ref()
    }
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct KernelParameters {
    base_ptr: *const c_void,
    holder: Vec<*mut c_void>,
}

impl KernelParameters {
    pub fn new(holder: Vec<*mut c_void>) -> Self {
        let base_ptr = holder.as_ptr() as *const c_void;
        Self { holder, base_ptr }
    }

    pub fn get_mut_ptr(&mut self) -> *mut c_void {
        assert_eq!(self.base_ptr, self.holder.as_ptr() as *const c_void);
        self.base_ptr as *mut c_void
    }

    pub fn llvm_get<'a>(
        ctx: &'a Context,
        build: &Builder<'a>,
        base_ptr: PointerValue<'a>,
        idx: usize,
    ) -> PointerValue<'a> {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let res = build
            .build_load(
                ptr_type,
                increment_pointer!(ctx, build, base_ptr, 8 * idx),
                "inc_ptr",
            )
            .unwrap()
            .into_pointer_value();
        res.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        res
    }
}

#[self_referencing]
pub struct DSLKernel {
    context: Context,
    input_types: Vec<DataType>,
    is_scalar: Vec<bool>,
    output_strategy: KernelOutputType,
    out_type: DataType,

    #[borrows(context)]
    #[covariant]
    func: (
        Vec<KernelInputType>,
        JitFunction<'this, unsafe extern "C" fn(*mut c_void) -> u64>,
    ),
}

impl DSLKernel {
    pub fn compile<F: Fn(KernelContext) -> Result<SealedKernelProgram, DSLError>>(
        inputs: &[&dyn Datum],
        f: F,
    ) -> Result<DSLKernel, DSLError> {
        let ctx = Context::create();
        let input_types = inputs
            .iter()
            .map(|x| x.get().0.data_type().clone())
            .collect_vec();
        let is_scalar = inputs.iter().map(|x| x.get().1).collect_vec();

        let context = KernelContext {
            inputs: inputs
                .iter()
                .enumerate()
                .map(|(idx, itm)| KernelInput::Datum(idx, *itm))
                .collect(),
        };
        let program = f(context)?;
        let out_type = program.out_type().clone();
        let output_strategy = program.strategy();

        DSLKernelTryBuilder {
            context: ctx,
            input_types,
            is_scalar,
            out_type,
            output_strategy,
            func_builder: |ctx| build_kernel(ctx, inputs, program),
        }
        .try_build()
    }

    pub fn call(&self, inputs: &[&dyn Datum]) -> Result<ArrayRef, ArrowKernelError> {
        if inputs.len() != self.borrow_input_types().len() {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "expected {} arguments, got {}",
                self.borrow_input_types().len(),
                inputs.len()
            )));
        }
        for (idx, ((input, expected_type), expected_scalar)) in inputs
            .iter()
            .zip(self.borrow_input_types().iter())
            .zip(self.borrow_is_scalar().iter())
            .enumerate()
        {
            if input.get().0.data_type() != expected_type {
                return Err(ArrowKernelError::ArgumentMismatch(format!(
                    "for argument {}, expected type: {}, got: {}",
                    idx,
                    expected_type,
                    input.get().0.data_type()
                )));
            }
            if input.get().1 != *expected_scalar {
                return Err(ArrowKernelError::ArgumentMismatch(format!(
                    "for argument {}, expected scalar: {}, got: {}",
                    idx,
                    expected_scalar,
                    input.get().1
                )));
            }
        }

        let max_len = inputs
            .iter()
            .map(|&input| input.get().0.len())
            .max()
            .unwrap_or(0);

        let mut ihs: Vec<IteratorHolder> = inputs
            .iter()
            .zip(self.borrow_func().0.iter())
            .map(|(&input, ty)| match ty {
                KernelInputType::Standard => datum_to_iter(input),
                KernelInputType::SetBit => array_to_setbit_iter(input.get().0.as_boolean()),
            })
            .try_collect()?;
        let mut ptrs = Vec::new();
        ptrs.extend(ihs.iter_mut().map(|ih| ih.get_mut_ptr()));
        let p_out_type = PrimitiveType::for_arrow_type(self.borrow_out_type());

        match self.borrow_output_strategy() {
            KernelOutputType::Array => {
                let mut alloc = PrimitiveArrayWriter::allocate(max_len, p_out_type);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) } as usize;
                Ok(alloc.to_array(num_results, None))
            }
            KernelOutputType::String => {
                let mut alloc = StringArrayWriter::<i32>::allocate(max_len, p_out_type);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) } as usize;
                let arr = alloc.to_array(num_results, None);

                let data = unsafe {
                    arr.into_data()
                        .into_builder()
                        .data_type(DataType::Utf8)
                        .build_unchecked()
                };

                Ok(Arc::new(StringArray::from(data)))
            }
            KernelOutputType::Boolean => {
                let mut alloc = BooleanWriter::allocate(max_len, p_out_type);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                Ok(Arc::new(alloc.to_array(num_results as usize, None)))
            }
            KernelOutputType::View => {
                let mut alloc = StringViewWriter::allocate(max_len, p_out_type);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                Ok(Arc::new(alloc.to_array(num_results as usize, None)))
            }
            KernelOutputType::Dictionary(key) => Ok(match key {
                DictKeyType::Int8 => {
                    Arc::new(self.exec_to_dict::<Int8Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::Int16 => {
                    Arc::new(self.exec_to_dict::<Int16Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::Int32 => {
                    Arc::new(self.exec_to_dict::<Int32Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::Int64 => {
                    Arc::new(self.exec_to_dict::<Int64Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::UInt8 => {
                    Arc::new(self.exec_to_dict::<UInt8Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::UInt16 => {
                    Arc::new(self.exec_to_dict::<UInt16Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::UInt32 => {
                    Arc::new(self.exec_to_dict::<UInt32Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::UInt64 => {
                    Arc::new(self.exec_to_dict::<UInt64Type>(ptrs, max_len, p_out_type))
                }
            }),
            KernelOutputType::RunEnd(re_type) => Ok(match re_type {
                RunEndType::Int16 => {
                    Arc::new(self.exec_to_ree::<Int16Type>(ptrs, max_len, p_out_type))
                }
                RunEndType::Int32 => {
                    Arc::new(self.exec_to_ree::<Int32Type>(ptrs, max_len, p_out_type))
                }
                RunEndType::Int64 => {
                    Arc::new(self.exec_to_ree::<Int64Type>(ptrs, max_len, p_out_type))
                }
            }),
        }
    }

    fn exec_to_dict<K: ArrowDictionaryKeyType>(
        &self,
        mut ptrs: Vec<*mut c_void>,
        max_len: usize,
        pt: PrimitiveType,
    ) -> DictionaryArray<K> {
        match pt {
            PrimitiveType::P64x2 => {
                let mut alloc = DictWriter::<K, StringArrayWriter<i32>>::allocate(max_len, pt);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                alloc.to_array(num_results as usize, None)
            }
            _ => {
                let mut alloc = DictWriter::<K, PrimitiveArrayWriter>::allocate(max_len, pt);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                alloc.to_array(num_results as usize, None)
            }
        }
    }

    fn exec_to_ree<K: RunEndIndexType>(
        &self,
        mut ptrs: Vec<*mut c_void>,
        max_len: usize,
        pt: PrimitiveType,
    ) -> RunArray<K> {
        match pt {
            PrimitiveType::P64x2 => {
                let mut alloc = REEWriter::<K, StringArrayWriter<i32>>::allocate(max_len, pt);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                alloc.to_array(num_results as usize, None)
            }
            _ => {
                let mut alloc = REEWriter::<K, PrimitiveArrayWriter>::allocate(max_len, pt);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                alloc.to_array(num_results as usize, None)
            }
        }
    }
}

fn build_kernel<'a>(
    ctx: &'a Context,
    inputs: &[&dyn Datum],
    program: SealedKernelProgram<'_>,
) -> Result<
    (
        Vec<KernelInputType>,
        JitFunction<'a, unsafe extern "C" fn(*mut c_void) -> u64>,
    ),
    DSLError,
> {
    match program.strategy() {
        KernelOutputType::Array => {
            build_kernel_with_writer::<PrimitiveArrayWriter>(ctx, inputs, program)
        }
        KernelOutputType::String => {
            build_kernel_with_writer::<StringArrayWriter<i32>>(ctx, inputs, program)
        }
        KernelOutputType::Boolean => {
            build_kernel_with_writer::<BooleanWriter>(ctx, inputs, program)
        }
        KernelOutputType::View => {
            build_kernel_with_writer::<StringViewWriter>(ctx, inputs, program)
        }
        KernelOutputType::Dictionary(key) => match key {
            DictKeyType::Int8 => build_dict_kernel::<Int8Type>(ctx, inputs, program),
            DictKeyType::Int16 => build_dict_kernel::<Int16Type>(ctx, inputs, program),
            DictKeyType::Int32 => build_dict_kernel::<Int32Type>(ctx, inputs, program),
            DictKeyType::Int64 => build_dict_kernel::<Int64Type>(ctx, inputs, program),
            DictKeyType::UInt8 => build_dict_kernel::<UInt8Type>(ctx, inputs, program),
            DictKeyType::UInt16 => build_dict_kernel::<UInt16Type>(ctx, inputs, program),
            DictKeyType::UInt32 => build_dict_kernel::<UInt32Type>(ctx, inputs, program),
            DictKeyType::UInt64 => build_dict_kernel::<UInt64Type>(ctx, inputs, program),
        },
        KernelOutputType::RunEnd(key) => match key {
            RunEndType::Int16 => build_ree_kernel::<Int16Type>(ctx, inputs, program),
            RunEndType::Int32 => build_ree_kernel::<Int32Type>(ctx, inputs, program),
            RunEndType::Int64 => build_ree_kernel::<Int64Type>(ctx, inputs, program),
        },
    }
}

fn build_dict_kernel<'a, T: ArrowDictionaryKeyType>(
    ctx: &'a Context,
    inputs: &[&dyn Datum],
    program: SealedKernelProgram<'_>,
) -> Result<
    (
        Vec<KernelInputType>,
        JitFunction<'a, unsafe extern "C" fn(*mut c_void) -> u64>,
    ),
    DSLError,
> {
    if program.out_type().is_primitive() {
        build_kernel_with_writer::<DictWriter<T, PrimitiveArrayWriter>>(ctx, inputs, program)
    } else {
        match program.out_type() {
            DataType::Binary | DataType::Utf8 => build_kernel_with_writer::<
                DictWriter<T, StringArrayWriter<i32>>,
            >(ctx, inputs, program),
            DataType::LargeBinary | DataType::LargeUtf8 => build_kernel_with_writer::<
                DictWriter<T, StringArrayWriter<i64>>,
            >(ctx, inputs, program),
            _ => Err(DSLError::UnsupportedDictionaryValueType(
                program.out_type().clone(),
            )),
        }
    }
}

fn build_ree_kernel<'a, T: RunEndIndexType>(
    ctx: &'a Context,
    inputs: &[&dyn Datum],
    program: SealedKernelProgram<'_>,
) -> Result<
    (
        Vec<KernelInputType>,
        JitFunction<'a, unsafe extern "C" fn(*mut c_void) -> u64>,
    ),
    DSLError,
> {
    if program.out_type().is_primitive() {
        build_kernel_with_writer::<REEWriter<T, PrimitiveArrayWriter>>(ctx, inputs, program)
    } else {
        match program.out_type() {
            DataType::Binary | DataType::Utf8 => build_kernel_with_writer::<
                REEWriter<T, StringArrayWriter<i32>>,
            >(ctx, inputs, program),
            DataType::LargeBinary | DataType::LargeUtf8 => build_kernel_with_writer::<
                REEWriter<T, StringArrayWriter<i64>>,
            >(ctx, inputs, program),
            _ => Err(DSLError::UnsupportedDictionaryValueType(
                program.out_type().clone(),
            )),
        }
    }
}

fn build_kernel_with_writer<'a, W: ArrayWriter<'a>>(
    ctx: &'a Context,
    inputs: &[&dyn Datum],
    program: SealedKernelProgram<'_>,
) -> Result<
    (
        Vec<KernelInputType>,
        JitFunction<'a, unsafe extern "C" fn(*mut c_void) -> u64>,
    ),
    DSLError,
> {
    let llvm_mod = ctx.create_module("kernel");
    let builder = ctx.create_builder();
    let i64_type = ctx.i64_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());

    let num_total_inputs = inputs.len() + 1;
    let func_type = i64_type.fn_type(&[ptr_type.into()], false);
    let func_outer = llvm_mod.add_function("kernel", func_type, None);
    let func_inner = llvm_mod.add_function(
        "inner_kernel",
        i64_type.fn_type(
            &(0..num_total_inputs).map(|_| ptr_type.into()).collect_vec(),
            false,
        ),
        Some(Linkage::Private),
    );
    set_noalias_params(&func_inner);

    //
    // Outer function
    //
    declare_blocks!(ctx, func_outer, entry);
    builder.position_at_end(entry);
    let param_ptr = func_outer.get_nth_param(0).unwrap().into_pointer_value();
    let ptr_params = (0..num_total_inputs)
        .map(|idx| KernelParameters::llvm_get(ctx, &builder, param_ptr, idx).into())
        .collect_vec();
    let num_produced = builder
        .build_call(func_inner, &ptr_params, "num_produced")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left();
    builder.build_return(Some(&num_produced)).unwrap();

    //
    // Inner function
    //
    let out_type = program.out_type();
    let out_prim_type = PrimitiveType::for_arrow_type(out_type);

    let indexes_to_iter = program.iterated_indexes();
    if indexes_to_iter.is_empty() {
        return Err(DSLError::NoIteration);
    }

    let ihs: Vec<IteratorHolder> = {
        let idx_to_type: HashMap<usize, KernelInputType> = indexes_to_iter
            .iter()
            .map(|(ty, idx)| (*idx, *ty))
            .collect();

        inputs
            .iter()
            .enumerate()
            .map(|(idx, inp)| {
                if let Some(ty) = idx_to_type.get(&idx) {
                    match ty {
                        KernelInputType::Standard => datum_to_iter(*inp),
                        KernelInputType::SetBit => array_to_setbit_iter(inp.get().0.as_boolean()),
                    }
                } else {
                    datum_to_iter(*inp)
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    };

    let all_inputs_scalar = indexes_to_iter
        .iter()
        .map(|(_ty, idx)| idx)
        .all(|idx| inputs[*idx].get().1);

    let mut iter_llvm_types = HashMap::new();
    for (ty, idx) in indexes_to_iter.iter() {
        let ptype = match ty {
            KernelInputType::Standard => {
                PrimitiveType::for_arrow_type(inputs[*idx].get().0.data_type())
            }
            KernelInputType::SetBit => PrimitiveType::U64,
        };
        iter_llvm_types.insert(*idx, ptype.llvm_type(ctx));
    }
    let next_funcs: HashMap<usize, FunctionValue> = indexes_to_iter
        .iter()
        .map(|(_ty, idx)| {
            (
                *idx,
                generate_next(
                    ctx,
                    &llvm_mod,
                    &format!("next{}", idx),
                    inputs[*idx].get().0.data_type(),
                    &ihs[*idx],
                )
                .unwrap(),
            )
        })
        .collect();

    let next_block_funcs: HashMap<usize, Option<FunctionValue>> = indexes_to_iter
        .iter()
        .map(|(_ty, idx)| {
            (
                *idx,
                generate_next_block::<64>(
                    ctx,
                    &llvm_mod,
                    &format!("next_block{}", idx),
                    inputs[*idx].get().0.data_type(),
                    &ihs[*idx],
                ),
            )
        })
        .collect();

    let get_funcs: HashMap<usize, FunctionValue> = program
        .accessed_indexes()
        .iter()
        .map(|idx| {
            (
                *idx,
                generate_random_access(
                    ctx,
                    &llvm_mod,
                    &format!("get{}", idx),
                    inputs[*idx].get().0.data_type(),
                    &ihs[*idx],
                )
                .unwrap(),
            )
        })
        .collect();

    declare_blocks!(
        ctx,
        func_inner,
        entry,
        loop_cond,
        filter_check,
        loop_body,
        exit
    );
    builder.position_at_end(entry);
    let mut iter_ptrs = Vec::new();
    for i in 0..inputs.len() {
        let ptr = func_inner
            .get_nth_param(i as u32)
            .unwrap()
            .into_pointer_value();
        iter_ptrs.push(ihs[i].localize_struct(ctx, &builder, ptr));
    }

    let out_ptr = func_inner
        .get_nth_param(inputs.len() as u32)
        .unwrap()
        .into_pointer_value();
    let writer = W::llvm_init(ctx, &llvm_mod, &builder, out_prim_type, out_ptr);

    let bufs: HashMap<usize, PointerValue> = indexes_to_iter
        .iter()
        .map(|(_ty, idx)| {
            (
                *idx,
                builder
                    .build_alloca(iter_llvm_types[idx], &format!("buf{}", idx))
                    .unwrap(),
            )
        })
        .collect();

    let produced_ptr = builder.build_alloca(i64_type, "out_count").unwrap();
    builder
        .build_store(produced_ptr, i64_type.const_zero())
        .unwrap();

    // possibly add a block-iteration fast path TODO
    let mut uses_blocks = false;
    if next_block_funcs.values().all(|f| f.is_some())
        && program.filter().is_none()
        && !all_inputs_scalar
    {
        // potentially use block iteration, allocate buffers
        let mut vec_bufs = HashMap::new();
        let mut vec_types = HashMap::new();
        for (ty, idx) in indexes_to_iter.iter() {
            let ptype = match ty {
                KernelInputType::Standard => {
                    PrimitiveType::for_arrow_type(inputs[*idx].get().0.data_type())
                }
                KernelInputType::SetBit => PrimitiveType::U64,
            };
            let vec_buf = ptype
                .llvm_vec_type(ctx, 64)
                .map(|t| builder.build_alloca(t, &format!("vbuf{}", t)).unwrap());
            vec_bufs.insert(idx, vec_buf);
            vec_types.insert(idx, ptype.llvm_vec_type(ctx, 64));
        }

        if vec_bufs.values().all(|x| x.is_some()) {
            let vec_bufs = vec_bufs
                .into_iter()
                .map(|(k, v)| (*k, v.unwrap()))
                .collect();
            let vec_types = vec_types
                .into_iter()
                .map(|(k, v)| (*k, v.unwrap().as_basic_type_enum()))
                .collect();

            // all our inputs support block iteration, see if our program does
            declare_blocks!(ctx, func_inner, block_loop_cond, block_loop_body);
            builder.position_at_end(block_loop_body);
            let res = program
                .expr()
                .compile_block(ctx, &builder, &vec_bufs, &vec_types);
            match res {
                Ok(mut v) => {
                    // send `v` to the writer, loop back
                    if writer.llvm_ingest_type(ctx) == ctx.bool_type().as_basic_type_enum() {
                        v = builder
                            .build_int_compare(
                                IntPredicate::NE,
                                v,
                                v.get_type().const_zero(),
                                "to_bool",
                            )
                            .unwrap();
                    }

                    writer.llvm_ingest_block(ctx, &builder, v);
                    let curr_produced = builder
                        .build_load(i64_type, produced_ptr, "curr_produced")
                        .unwrap()
                        .into_int_value();
                    let new_produced = builder
                        .build_int_add(
                            curr_produced,
                            i64_type.const_int(v.get_type().get_size() as u64, false),
                            "new_produced",
                        )
                        .unwrap();
                    builder.build_store(produced_ptr, new_produced).unwrap();
                    builder.build_unconditional_branch(block_loop_cond).unwrap();

                    builder.position_at_end(block_loop_cond);
                    let mut had_nexts = Vec::new();
                    // call next on each of the iterators that we care about
                    for (_ty, param_idx) in indexes_to_iter.iter() {
                        let buf = vec_bufs[param_idx];
                        let next_func = next_block_funcs[param_idx].unwrap();
                        let iter_ptr = iter_ptrs[*param_idx];
                        had_nexts.push(
                            builder
                                .build_call(
                                    next_func,
                                    &[iter_ptr.into(), buf.into()],
                                    &format!("next{}", param_idx),
                                )
                                .unwrap()
                                .try_as_basic_value()
                                .unwrap_left()
                                .into_int_value(),
                        );
                    }

                    // AND-together all has-nexts
                    let mut accum = had_nexts.pop().unwrap();
                    for val in had_nexts {
                        accum = builder.build_and(accum, val, "accum").unwrap();
                    }
                    builder
                        .build_conditional_branch(accum, block_loop_body, loop_cond)
                        .unwrap();

                    builder.position_at_end(entry);
                    builder.build_unconditional_branch(block_loop_cond).unwrap();

                    uses_blocks = true;
                }
                Err(e) => {
                    println!("Unable to compile blocked version of kernel: {}", e);
                    for buf in vec_bufs.into_values() {
                        buf.as_instruction().unwrap().remove_from_basic_block();
                    }
                    block_loop_cond.remove_from_function().unwrap();
                    block_loop_body.remove_from_function().unwrap();
                }
            }
        }
    }

    if !uses_blocks {
        builder.position_at_end(entry);
        builder.build_unconditional_branch(loop_cond).unwrap();
    }

    builder.position_at_end(loop_cond);
    let mut had_nexts = Vec::new();
    // call next on each of the iterators that we care about (note the
    // difference between the index of the iterator we care about and the index
    // of the parameter value)
    for (_ty, param_idx) in indexes_to_iter.iter() {
        let buf = bufs[param_idx];
        let next_func = next_funcs[param_idx];
        let iter_ptr = iter_ptrs[*param_idx];
        had_nexts.push(
            builder
                .build_call(
                    next_func,
                    &[iter_ptr.into(), buf.into()],
                    &format!("next{}", param_idx),
                )
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value(),
        );
    }

    // AND-together all has-nexts
    let mut accum = had_nexts.pop().unwrap();
    for val in had_nexts {
        accum = builder.build_and(accum, val, "accum").unwrap();
    }
    builder
        .build_conditional_branch(accum, filter_check, exit)
        .unwrap();

    builder.position_at_end(filter_check);
    match program.filter() {
        Some(cond) => {
            let result = cond.compile(
                ctx,
                &llvm_mod,
                &builder,
                &bufs,
                &get_funcs,
                &iter_ptrs,
                &iter_llvm_types,
            )?;
            builder
                .build_conditional_branch(result.into_int_value(), loop_body, loop_cond)
                .unwrap();
        }
        None => {
            builder.build_unconditional_branch(loop_body).unwrap();
        }
    }

    builder.position_at_end(loop_body);
    let mut result = program.expr().compile(
        ctx,
        &llvm_mod,
        &builder,
        &bufs,
        &get_funcs,
        &iter_ptrs,
        &iter_llvm_types,
    )?;

    if writer.llvm_ingest_type(ctx) == ctx.bool_type().as_basic_type_enum() {
        result = builder
            .build_int_compare(
                IntPredicate::NE,
                result.into_int_value(),
                result.get_type().const_zero().into_int_value(),
                "not_zero",
            )
            .unwrap()
            .as_basic_value_enum();
    }

    writer.llvm_ingest(ctx, &builder, result);
    let curr_produced = builder
        .build_load(i64_type, produced_ptr, "curr_produced")
        .unwrap()
        .into_int_value();
    let new_produced = builder
        .build_int_add(curr_produced, i64_type.const_int(1, false), "new_produced")
        .unwrap();
    builder.build_store(produced_ptr, new_produced).unwrap();

    // if all of our inputs are scalar, then the next function will return true
    // forever -- we need to jump to the exit after the first iteration.
    if all_inputs_scalar {
        builder.build_unconditional_branch(exit).unwrap();
    } else {
        builder.build_unconditional_branch(loop_cond).unwrap();
    }

    builder.position_at_end(exit);
    writer.llvm_flush(ctx, &builder);
    let produced = builder
        .build_load(i64_type, produced_ptr, "produced")
        .unwrap()
        .into_int_value();
    builder.build_return(Some(&produced)).unwrap();

    llvm_mod.verify().unwrap();
    optimize_module(&llvm_mod).unwrap();
    let ee = llvm_mod
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();
    link_req_helpers(&llvm_mod, &ee).unwrap();

    // build an access map for the caller -- for accessed indexes, assume
    // standard, but override that if we use the set bit iterator in the
    // expression
    let mut access_map = BTreeMap::new();
    for idx in program.accessed_indexes() {
        access_map.insert(idx, KernelInputType::Standard);
    }
    for (ty, idx) in indexes_to_iter {
        access_map.insert(idx, ty);
    }

    if access_map.len() != inputs.len() {
        return Err(DSLError::UnusedInput(format!(
            "{} inputs were used, but {} inputs were provided",
            access_map.len(),
            inputs.len()
        )));
    }
    let access_map = access_map.into_values().collect_vec();

    Ok((access_map, unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void) -> u64>(
            func_outer.get_name().to_str().unwrap(),
        )
        .unwrap()
    }))
}

#[cfg(test)]
mod test {

    use arrow_array::{
        cast::AsArray,
        types::{BinaryViewType, Int32Type, Int64Type, UInt64Type},
        BooleanArray, Float32Array, Int32Array, StringArray,
    };
    use arrow_schema::DataType;
    use itertools::Itertools;

    use crate::{compiled_kernels::dsl::DSLKernel, dictionary_data_type, PrimitiveType};

    use super::KernelOutputType;

    #[test]
    fn test_dsl_int_between() {
        let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let sca1 = Int32Array::new_scalar(1);
        let sca2 = Int32Array::new_scalar(6);

        let k = DSLKernel::compile(&[&data, &sca1, &sca2], |ctx| {
            let data = ctx.get_input(0)?;
            let scalar1 = ctx.get_input(1)?;
            let scalar2 = ctx.get_input(2)?;

            ctx.iter_over(vec![data, scalar1, scalar2])
                .map(|i| vec![i[0].gt(&i[1]), i[0].lt(&i[2])])
                .map(|i| vec![i[0].and(&i[1])])
                .collect(KernelOutputType::Boolean)
        })
        .unwrap();

        let res = k.call(&[&data, &sca1, &sca2]);
        let res = res.unwrap();

        assert_eq!(
            res.as_boolean(),
            &BooleanArray::from(vec![false, true, true, true, true, false])
        );
    }

    #[test]
    fn test_dsl_str_between() {
        let data = StringArray::from(vec!["a", "b", "c", "d", "e", "f"]);
        let sca1 = StringArray::new_scalar("a");
        let sca2 = StringArray::new_scalar("f");

        let k = DSLKernel::compile(&[&data, &sca1, &sca2], |ctx| {
            let data = ctx.get_input(0)?;
            let scalar1 = ctx.get_input(1)?;
            let scalar2 = ctx.get_input(2)?;

            ctx.iter_over(vec![data, scalar1, scalar2])
                .map(|i| vec![i[0].gt(&i[1]), i[0].lt(&i[2])])
                .map(|i| vec![i[0].and(&i[1])])
                .collect(KernelOutputType::Boolean)
        })
        .unwrap();

        let res = k.call(&[&data, &sca1, &sca2]);
        let res = res.unwrap();

        assert_eq!(
            res.as_boolean(),
            &BooleanArray::from(vec![false, true, true, true, true, false])
        );
    }

    #[test]
    fn test_dsl_int_max() {
        let data1 = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let data2 = Int32Array::from(vec![-1, 20, -3, 40, -5, 1]);

        let k = DSLKernel::compile(&[&data1, &data2], |ctx| {
            let lhs = ctx.get_input(0)?;
            let rhs = ctx.get_input(1)?;

            ctx.iter_over(vec![lhs, rhs])
                .map(|i| vec![i[0].gt(&i[1]).select(&i[0], &i[1])])
                .collect(KernelOutputType::Array)
        })
        .unwrap();

        let res = k.call(&[&data1, &data2]).unwrap();

        assert_eq!(
            res.as_primitive::<Int32Type>(),
            &Int32Array::from(vec![1, 20, 3, 40, 5, 6])
        );
    }

    #[test]
    fn test_dsl_at_max() {
        // compute max(data3[data1], data3[data2])
        let data1 = Int32Array::from(vec![0, 1, 2, 3, 4, 5]);
        let data2 = Int32Array::from(vec![2, 1, 0, 5, 4, 3]);
        let data3 = Int32Array::from(vec![0, 10, 20, 30, 40, 50]);

        let k = DSLKernel::compile(&[&data1, &data2, &data3], |ctx| {
            let lhs = ctx.get_input(0)?;
            let rhs = ctx.get_input(1)?;
            let dat = ctx.get_input(2)?;

            ctx.iter_over(vec![lhs, rhs])
                .map(|i| vec![dat.at(&i[0]), dat.at(&i[1])])
                .map(|i| vec![i[0].gt(&i[1]).select(&i[0], &i[1])])
                .collect(KernelOutputType::Array)
        })
        .unwrap();

        let res = k.call(&[&data1, &data2, &data3]).unwrap();

        assert_eq!(
            res.as_primitive::<Int32Type>(),
            &Int32Array::from(vec![20, 10, 20, 50, 40, 50])
        );
    }

    #[test]
    fn test_dsl_string_flatten() {
        let odata = vec!["this", "this", "is", "a", "a", "test"];
        let data = StringArray::from(odata.clone());
        let data = arrow_cast::cast(
            &data,
            &dictionary_data_type(DataType::UInt8, DataType::Utf8),
        )
        .unwrap();

        let k = DSLKernel::compile(&[&data], |ctx| {
            let inp = ctx.get_input(0)?;
            ctx.iter_over(vec![inp]).collect(KernelOutputType::String)
        })
        .unwrap();

        let res = k.call(&[&data]).unwrap();
        let res = res
            .as_string::<i32>()
            .iter()
            .map(|x| x.unwrap())
            .collect_vec();
        assert_eq!(res, odata);
    }

    #[test]
    fn test_kernel_set_bit_iter() {
        let data = BooleanArray::from(vec![true, true, false, true, false, true]);
        let k = DSLKernel::compile(&[&data], |ctx| {
            let inp = ctx.get_input(0)?.into_set_bits()?;
            ctx.iter_over(vec![inp]).collect(KernelOutputType::Array)
        })
        .unwrap();
        let res = k.call(&[&data]).unwrap();
        let res = res
            .as_primitive::<UInt64Type>()
            .iter()
            .map(|x| x.unwrap())
            .collect_vec();
        assert_eq!(res, vec![0, 1, 3, 5]);
    }

    #[test]
    fn test_kernel_set_bit_idx() {
        let filter = BooleanArray::from(vec![true, true, false, true, false, true]);
        let data = Int32Array::from(vec![10, 20, 30, 40, 50, 60]);
        let k = DSLKernel::compile(&[&filter, &data], |ctx| {
            let filter = ctx.get_input(0)?.into_set_bits()?;
            let data = ctx.get_input(1)?;
            ctx.iter_over(vec![filter])
                .map(|i| vec![data.at(&i[0])])
                .collect(KernelOutputType::Array)
        })
        .unwrap();
        let res = k.call(&[&filter, &data]).unwrap();
        let res = res
            .as_primitive::<Int32Type>()
            .iter()
            .map(|x| x.unwrap())
            .collect_vec();
        assert_eq!(res, vec![10, 20, 40, 60]);
    }

    #[test]
    fn test_kernel_filter_lt() {
        let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let scalar1 = Int32Array::new_scalar(5);
        let k = DSLKernel::compile(&[&data, &scalar1], |ctx| {
            ctx.iter_over(vec![ctx.get_input(0)?, ctx.get_input(1)?])
                .filter(|v| v[0].lt(&v[1]))
                .map(|v| vec![v[0].clone()])
                .collect(KernelOutputType::Array)
        })
        .unwrap();
        let res = k.call(&[&data, &scalar1]).unwrap();
        let res = res.as_primitive::<Int32Type>();
        assert_eq!(res.values(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_kernel_convert_i32_to_i64() {
        let data = Int32Array::from(vec![1, 2, 3, -4, 5, 6]);
        let k = DSLKernel::compile(&[&data], |ctx| {
            ctx.iter_over(vec![ctx.get_input(0)?])
                .map(|v| vec![v[0].clone().convert(PrimitiveType::I64)])
                .collect(KernelOutputType::Array)
        })
        .unwrap();
        let res = k.call(&[&data]).unwrap();
        let res = res.as_primitive::<Int64Type>();
        assert_eq!(res.values(), &[1, 2, 3, -4, 5, 6]);
    }

    #[test]
    fn test_kernel_convert_f32_to_i64() {
        let data = Float32Array::from(vec![1.2, 2.1, 3.4, -4.6, 5.7, 6.0]);
        let k = DSLKernel::compile(&[&data], |ctx| {
            ctx.iter_over(vec![ctx.get_input(0)?])
                .map(|v| vec![v[0].clone().convert(PrimitiveType::I64)])
                .collect(KernelOutputType::Array)
        })
        .unwrap();
        let res = k.call(&[&data]).unwrap();
        let res = res.as_primitive::<Int64Type>();
        assert_eq!(res.values(), &[1, 2, 3, -4, 5, 6]);
    }

    #[test]
    fn test_kernel_convert_str_to_view() {
        let strs = vec!["this", "is", "a test with at least one long string"];
        let data = StringArray::from(strs.clone());
        let k = DSLKernel::compile(&[&data], |ctx| {
            ctx.iter_over(vec![ctx.get_input(0)?])
                .collect(KernelOutputType::View)
        })
        .unwrap();
        let res = k.call(&[&data]).unwrap();
        let res = res.as_byte_view::<BinaryViewType>();
        let res = res
            .iter()
            .map(|b| std::str::from_utf8(b.unwrap()).unwrap())
            .collect_vec();
        assert_eq!(res, strs);
    }

    #[test]
    fn test_kernel_convert_str_to_view_single() {
        let strs = vec!["𑚀aAￒￚA"];
        let data = StringArray::from(strs.clone());
        let k = DSLKernel::compile(&[&data], |ctx| {
            ctx.iter_over(vec![ctx.get_input(0)?])
                .collect(KernelOutputType::View)
        })
        .unwrap();
        let res = k.call(&[&data]).unwrap();
        let res = res.as_byte_view::<BinaryViewType>();
        let res = res
            .iter()
            .map(|b| std::str::from_utf8(b.unwrap()).unwrap())
            .collect_vec();
        assert_eq!(res, strs);
    }
}
