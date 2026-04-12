use std::collections::HashSet;
use std::fmt::Debug;
#[cfg(test)]
use std::ops::{BitAnd, BitOr, BitXor};
use std::sync::Arc;
use std::{ffi::c_void, marker::PhantomData};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
#[cfg(test)]
use arrow_array::{ArrowPrimitiveType, PrimitiveArray};

use arrow_array::ArrayRef;
use arrow_array::{Datum, UInt32Array, UInt64Array};
use arrow_schema::DataType;
use inkwell::context::Context;
use inkwell::types::{BasicType, BasicTypeEnum};
use inkwell::{FloatPredicate, IntPredicate};
use itertools::Itertools;
use strum_macros::EnumIter;

use crate::compiled_iter::IteratorHolder;
use crate::compiled_kernels::dsl::base_type;
use crate::{ArrowKernelError, Predicate, PrimitiveType};

mod buffer;
mod compiler;
mod resolver;
mod runtime;
mod two_d;
mod writers;

pub use self::writers::OutputSpec;
pub use buffer::DSLBuffer;
pub use compiler::compile;
pub use runtime::RunnableDSLFunction;
pub use writers::{OutputSlot, WriterSpec};

pub struct DSLContext {
    symbol_counter: usize,
}

impl DSLContext {
    pub fn new() -> Self {
        Self { symbol_counter: 0 }
    }

    pub fn next_symbol(&mut self) -> usize {
        let sym = self.symbol_counter;
        self.symbol_counter += 1;
        sym
    }
}

#[derive(Debug)]
pub enum DSLArgumentType {
    Datum { dt: DataType, is_scalar: bool },
    SetBits,
    TwoDArray(Vec<DataType>),
    Buffer(PrimitiveType),
}

impl DSLArgumentType {
    /// Validates that a given argument matches this type. Called once per kernel invocation.
    pub fn matches<'a>(&self, arg: &DSLArgument<'a>) -> Result<(), String> {
        match (self, arg) {
            (DSLArgumentType::Datum { dt, is_scalar }, DSLArgument::Datum(datum)) => {
                let (arg_arr, arg_is_scalar) = datum.get();
                if arg_is_scalar != *is_scalar {
                    return Err(format!(
                        "scalar / non-scalar mismatch, expected {}, got {}",
                        is_scalar, arg_is_scalar
                    ));
                }
                if arg_arr.data_type() != dt {
                    return Err(format!(
                        "type mismatch, expected {}, got {}",
                        dt,
                        arg_arr.data_type()
                    ));
                }
                Ok(())
            }
            (DSLArgumentType::TwoDArray(dts), DSLArgument::TwoDArray(datums)) => {
                if dts.len() != datums.len() {
                    return Err(format!(
                        "dimension mismatch, expected {} dimensions, got {}",
                        dts.len(),
                        datums.len()
                    ));
                }
                for (dt, datum) in dts.iter().zip(datums.iter()) {
                    let (arg_arr, _) = datum.get();
                    if arg_arr.data_type() != dt {
                        return Err(format!(
                            "type mismatch, expected {}, got {}",
                            dt,
                            arg_arr.data_type()
                        ));
                    }
                }
                Ok(())
            }
            (DSLArgumentType::Buffer(pt1), DSLArgument::Buffer { primitive_type, .. }) => {
                if pt1 != primitive_type {
                    return Err(format!(
                        "buffer type mismatch, expected {}, got {}",
                        pt1, primitive_type
                    ));
                }
                Ok(())
            }
            (DSLArgumentType::SetBits, DSLArgument::Datum(d)) => {
                let (arr, is_scalar) = d.get();
                if is_scalar {
                    return Err("set bit iterator cannot be built over scalar".to_string());
                }

                arr.as_boolean_opt()
                    .ok_or_else(|| {
                        format!(
                            "set bit iterator expected boolean array, got {}",
                            arr.data_type()
                        )
                    })
                    .map(|_| ())
            }
            _ => Err(format!("argument type mismatch, expected {:?}", self)),
        }
    }

    pub fn is_set_bit(&self) -> bool {
        matches!(self, DSLArgumentType::SetBits)
    }
}

pub enum DSLArgument<'a> {
    Datum(&'a dyn Datum),
    TwoDArray(Vec<&'a dyn Datum>),
    Buffer {
        ptr: *mut c_void,
        primitive_type: PrimitiveType,
        _borrow: PhantomData<&'a mut buffer::DSLBuffer>,
    },
}

impl<'a> DSLArgument<'a> {
    pub fn datum<T>(value: &'a T) -> Self
    where
        T: Datum,
    {
        DSLArgument::Datum(value)
    }

    pub fn two_d(values: impl IntoIterator<Item = &'a dyn Datum>) -> Self {
        DSLArgument::TwoDArray(values.into_iter().collect())
    }

    pub fn buffer(value: &'a mut buffer::DSLBuffer) -> Self {
        DSLArgument::Buffer {
            ptr: value.as_ptr(),
            primitive_type: value.ty,
            _borrow: PhantomData,
        }
    }

    pub fn get_type(&self, dsl_ty: &DSLType) -> DSLArgumentType {
        match self {
            DSLArgument::Datum(datum) => match dsl_ty {
                DSLType::ConstScalar(..) | DSLType::Scalar(..) | DSLType::Array(..) => {
                    let (arr, is_scalar) = datum.get();
                    let dt = arr.data_type().clone();
                    DSLArgumentType::Datum { dt, is_scalar }
                }
                DSLType::SetBits(_) => {
                    let (arr, is_scalar) = datum.get();
                    assert!(
                        arr.as_boolean_opt().is_some(),
                        "non-boolean set bit iterator"
                    );
                    assert!(!is_scalar, "scalar set bit iterator");
                    DSLArgumentType::SetBits
                }
                _ => unreachable!("invalid DSL type for arg type"),
            },
            DSLArgument::TwoDArray(datums) => DSLArgumentType::TwoDArray(
                datums
                    .iter()
                    .map(|x| x.get().0.data_type().clone())
                    .collect_vec(),
            ),
            DSLArgument::Buffer { primitive_type, .. } => DSLArgumentType::Buffer(*primitive_type),
        }
    }

    pub fn as_datum(&self) -> Option<&dyn Datum> {
        match self {
            DSLArgument::Datum(d) => Some(*d),
            DSLArgument::TwoDArray(_) => None,
            DSLArgument::Buffer { .. } => todo!(),
        }
    }

    pub fn as_two_d(&self) -> Option<&[&dyn Datum]> {
        match self {
            DSLArgument::TwoDArray(a) => Some(a),
            _ => None,
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            DSLArgument::Datum(d) => d.get().0.data_type().clone(),
            DSLArgument::TwoDArray(a) => a[0].get().0.data_type().clone(),
            DSLArgument::Buffer { primitive_type, .. } => primitive_type.as_arrow_type(),
        }
    }

    /// Check if ths argument can take on a value of the given type.
    pub fn is_compatible_with(&self, ty: &DSLType) -> bool {
        match (self, ty) {
            (DSLArgument::Datum(d), DSLType::SetBits(_)) => {
                d.get().0.data_type() == &DataType::Boolean
            }
            (DSLArgument::Datum(..), DSLType::Boolean) => true,
            (DSLArgument::Datum(..), DSLType::Primitive(..)) => true,
            (DSLArgument::Datum(..), DSLType::ConstScalar(..)) => true,
            (DSLArgument::Datum(..), DSLType::Scalar(..)) => true,
            (DSLArgument::Datum(..), DSLType::Array(..)) => true,
            (DSLArgument::TwoDArray(..), DSLType::TwoDArray(..)) => true,
            (DSLArgument::Buffer { .. }, DSLType::Buffer(..)) => true,
            _ => false,
        }
    }
}

macro_rules! dsl_args {
    () => {
        Vec::new()
    };
    ([$($value:expr),* $(,)?]) => {
        vec![$crate::compiled_kernels::dsl2::DSLArgument::two_d([
            $(&$value as &dyn arrow_array::Datum),*
        ])]
    };
    ([$($value:expr),* $(,)?] $(, $($rest:tt)*)?) => {{
        let mut args = vec![
            $crate::compiled_kernels::dsl2::DSLArgument::two_d([
                $(&$value as &dyn arrow_array::Datum),*
            ])
        ];
        $(args.extend(dsl_args![$($rest)*]);)?
        args
    }};
    (&mut $buffer:expr) => {
        vec![$crate::compiled_kernels::dsl2::DSLArgument::buffer(&mut $buffer)]
    };
    (&mut $buffer:expr $(, $($rest:tt)*)?) => {{
        let mut args = vec![$crate::compiled_kernels::dsl2::DSLArgument::buffer(&mut $buffer)];
        $(args.extend(dsl_args![$($rest)*]);)?
        args
    }};
    ($value:expr) => {
        vec![$crate::compiled_kernels::dsl2::DSLArgument::datum(&$value)]
    };
    ($value:expr $(, $($rest:tt)*)?) => {{
        let mut args = vec![$crate::compiled_kernels::dsl2::DSLArgument::datum(&$value)];
        $(args.extend(dsl_args![$($rest)*]);)?
        args
    }};
}
pub(crate) use dsl_args;

impl Debug for DSLArgument<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Datum(arg0) => f
                .debug_tuple("Datum")
                .field(arg0.get().0.data_type())
                .finish(),
            Self::TwoDArray(_) => f.debug_tuple("TwoDArray").finish(),
            Self::Buffer {
                ptr,
                primitive_type,
                ..
            } => f
                .debug_struct("Buffer")
                .field("ptr", ptr)
                .field("primitive_type", primitive_type)
                .finish(),
        }
    }
}

#[derive(Clone)]
pub enum DSLType {
    Boolean,
    Primitive(PrimitiveType),
    ConstScalar(Arc<dyn Datum>),
    Scalar(Box<DSLType>),
    Array(Box<DSLType>, String),
    SetBits(String),
    TwoDArray(Box<DSLType>),
    Buffer(PrimitiveType, String),
}

impl PartialEq for DSLType {
    fn eq(&self, other: &DSLType) -> bool {
        match (self, other) {
            (DSLType::Boolean, DSLType::Boolean) => true,
            (DSLType::Primitive(a), DSLType::Primitive(b)) => a == b,
            (DSLType::ConstScalar(a), DSLType::ConstScalar(b)) => {
                a.get().0.data_type() == b.get().0.data_type()
            }
            (DSLType::Scalar(a), DSLType::Scalar(b)) => a == b,
            (DSLType::Array(a, a_len), DSLType::Array(b, b_len)) => a == b && a_len == b_len,
            (DSLType::SetBits(a_len), DSLType::SetBits(b_len)) => a_len == b_len,
            (DSLType::TwoDArray(a), DSLType::TwoDArray(b)) => a == b,
            (DSLType::Buffer(a, a_len), DSLType::Buffer(b, b_len)) => a == b && a_len == b_len,
            _ => false,
        }
    }
}

impl Debug for DSLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DSLType::Boolean => write!(f, "Boolean"),
            DSLType::Primitive(ty) => f.debug_tuple("Primitive").field(ty).finish(),
            DSLType::SetBits(len_expr) => f.debug_tuple("SetBits").field(len_expr).finish(),
            DSLType::ConstScalar(x) => write!(f, "ConstScalar({:?})", x.get().0),
            DSLType::Scalar(ty) => f.debug_tuple("Scalar").field(ty).finish(),
            DSLType::Array(ty, len_expr) => {
                f.debug_tuple("Array").field(ty).field(len_expr).finish()
            }
            DSLType::TwoDArray(ty) => f.debug_tuple("TwoDArray").field(ty).finish(),
            DSLType::Buffer(ty, len) => f.debug_tuple("Buffer").field(ty).field(len).finish(),
        }
    }
}

impl DSLType {
    pub fn iter_type(&self) -> Option<DSLType> {
        match self {
            DSLType::Array(ty, ..) => Some(*ty.clone()),
            DSLType::SetBits(..) => Some(DSLType::Primitive(PrimitiveType::U64)),
            DSLType::TwoDArray(ty, ..) => Some(*ty.clone()),
            DSLType::Buffer(ty, ..) => Some(DSLType::Primitive(*ty)),
            DSLType::Scalar(ty) => Some(*ty.clone()),
            DSLType::ConstScalar(_) => Some(self.clone()),
            _ => None,
        }
    }

    /// The DSLType of the given [`IteratorHolder`]. Note the returned DSLType
    /// will always have an empty size tag.
    pub fn of_iterator_holder(ih: &IteratorHolder) -> Self {
        match ih {
            IteratorHolder::Primitive(pi) => {
                Self::array_of(PrimitiveType::for_arrow_type(pi.data_type()), "")
            }
            IteratorHolder::String(..)
            | IteratorHolder::LargeString(..)
            | IteratorHolder::View(..) => Self::array_of(PrimitiveType::P64x2, ""),
            IteratorHolder::Bitmap(..) => DSLType::Array(Box::new(DSLType::Boolean), String::new()),
            IteratorHolder::SetBit(..) => DSLType::SetBits(String::new()),
            IteratorHolder::Dictionary { values, .. } => Self::of_iterator_holder(values),
            IteratorHolder::RunEnd { values, .. } => Self::of_iterator_holder(values),
            IteratorHolder::FixedSizeList(it) => Self::array_of(it.ptype(), ""),
            IteratorHolder::ScalarPrimitive(it) => DSLType::scalar_of(it.ptype),
            IteratorHolder::ScalarString(..) => DSLType::scalar_of(PrimitiveType::P64x2),
            IteratorHolder::ScalarBinary(..) => DSLType::scalar_of(PrimitiveType::P64x2),
            IteratorHolder::ScalarVec(it) => DSLType::scalar_of(it.ptype()),
        }
    }

    pub fn is_signed(&self) -> Option<bool> {
        match self {
            DSLType::Boolean => None,
            DSLType::Primitive(pt) => Some(pt.is_signed()),
            DSLType::SetBits(..) => Some(false),
            DSLType::ConstScalar(v) => {
                Some(PrimitiveType::for_arrow_type(v.get().0.data_type()).is_signed())
            }
            DSLType::Scalar(t) => t.is_signed(),
            DSLType::Array(t, ..) => t.is_signed(),
            DSLType::TwoDArray(t, ..) => t.is_signed(),
            DSLType::Buffer(t, ..) => Some(t.is_signed()),
        }
    }

    /// True if the resulting type, when iterated, never ends (scalars)
    pub fn is_infinite(&self) -> bool {
        match self {
            DSLType::ConstScalar(..) | DSLType::Scalar(..) => true,
            _ => false,
        }
    }

    pub fn is_buffer(&self) -> bool {
        matches!(self, DSLType::Buffer(..))
    }

    pub fn llvm_type<'a>(&self, ctx: &'a Context) -> Option<BasicTypeEnum<'a>> {
        match self {
            DSLType::Boolean => Some(ctx.bool_type().as_basic_type_enum()),
            DSLType::Primitive(pt) => Some(pt.llvm_type(ctx)),
            DSLType::SetBits(..) => Some(ctx.i64_type().as_basic_type_enum()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<(&DSLType, &str)> {
        match self {
            DSLType::Array(t, name) => Some((t, name)),
            _ => None,
        }
    }

    pub fn as_primitive(&self) -> Option<PrimitiveType> {
        match self {
            DSLType::Primitive(pt) => Some(*pt),
            _ => None,
        }
    }

    pub fn size_tag(&self) -> Option<&str> {
        match self {
            DSLType::Array(_, name) | DSLType::SetBits(name) | DSLType::Buffer(_, name) => {
                Some(name)
            }
            _ => None,
        }
    }

    pub fn two_d_array_of(pt: PrimitiveType) -> Self {
        DSLType::TwoDArray(Box::new(DSLType::array_of(pt, "")))
    }

    pub fn array_of<S: Into<String>>(pt: PrimitiveType, len_expr: S) -> Self {
        DSLType::Array(Box::new(DSLType::Primitive(pt)), len_expr.into())
    }

    pub fn array_like<S: Into<String>>(arr: &dyn Datum, len_expr: S) -> Self {
        let (arr, is_scalar) = arr.get();
        match base_type(arr.data_type()) {
            DataType::Null => todo!(),
            DataType::Boolean => {
                if is_scalar {
                    DSLType::Scalar(Box::new(DSLType::Boolean))
                } else {
                    DSLType::Array(Box::new(DSLType::Boolean), len_expr.into())
                }
            }
            _ => {
                let pt = PrimitiveType::for_arrow_type(arr.data_type());
                if is_scalar {
                    Self::scalar_of(pt)
                } else {
                    DSLType::Array(Box::new(DSLType::Primitive(pt)), len_expr.into())
                }
            }
        }
    }

    pub fn set_bits<S: Into<String>>(len_expr: S) -> Self {
        DSLType::SetBits(len_expr.into())
    }

    pub fn scalar_of(pt: PrimitiveType) -> Self {
        DSLType::Scalar(Box::new(DSLType::Primitive(pt)))
    }

    pub fn buffer_of<S: Into<String>>(pt: PrimitiveType, len_expr: S) -> Self {
        DSLType::Buffer(pt, len_expr.into())
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, EnumIter)]
pub enum DSLArithBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[cfg(test)]
impl DSLArithBinOp {
    pub fn arrow_compute(&self, arr1: &dyn Datum, arr2: &dyn Datum) -> ArrayRef {
        match self {
            Self::Add => arrow_arith::numeric::add(arr1, arr2).unwrap(),
            Self::Sub => arrow_arith::numeric::sub_wrapping(arr1, arr2).unwrap(),
            Self::Mul => arrow_arith::numeric::mul_wrapping(arr1, arr2).unwrap(),
            Self::Div => arrow_arith::numeric::div(arr1, arr2).unwrap(),
            Self::Rem => arrow_arith::numeric::rem(arr1, arr2).unwrap(),
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, EnumIter)]
pub enum DSLBitwiseBinOp {
    And,
    Or,
    Xor,
}

#[cfg(test)]
impl DSLBitwiseBinOp {
    pub fn arrow_compute<T>(
        &self,
        arr1: &PrimitiveArray<T>,
        arr2: &PrimitiveArray<T>,
    ) -> PrimitiveArray<T>
    where
        T: ArrowPrimitiveType,
        T::Native:
            BitAnd<Output = T::Native> + BitOr<Output = T::Native> + BitXor<Output = T::Native>,
    {
        match self {
            Self::And => arrow_arith::bitwise::bitwise_and(arr1, arr2).unwrap(),
            Self::Or => arrow_arith::bitwise::bitwise_or(arr1, arr2).unwrap(),
            Self::Xor => arrow_arith::bitwise::bitwise_xor(arr1, arr2).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DSLComparison {
    Lt,
    Lte,
    Gt,
    Gte,
    Eq,
    Neq,
}

impl DSLComparison {
    pub fn as_str(&self) -> &'static str {
        match self {
            DSLComparison::Lt => "lt",
            DSLComparison::Lte => "lte",
            DSLComparison::Gt => "gt",
            DSLComparison::Gte => "gte",
            DSLComparison::Eq => "eq",
            DSLComparison::Neq => "neq",
        }
    }

    pub fn as_float_predicate(&self) -> FloatPredicate {
        match self {
            DSLComparison::Eq => FloatPredicate::OEQ,
            DSLComparison::Neq => FloatPredicate::ONE,
            DSLComparison::Lt => FloatPredicate::OLT,
            DSLComparison::Lte => FloatPredicate::OLE,
            DSLComparison::Gt => FloatPredicate::OGT,
            DSLComparison::Gte => FloatPredicate::OGE,
        }
    }

    pub fn as_int_predicate(&self, signed: bool) -> IntPredicate {
        match self {
            DSLComparison::Eq => IntPredicate::EQ,
            DSLComparison::Neq => IntPredicate::NE,
            DSLComparison::Lt => {
                if signed {
                    IntPredicate::SLT
                } else {
                    IntPredicate::ULT
                }
            }
            DSLComparison::Lte => {
                if signed {
                    IntPredicate::SLE
                } else {
                    IntPredicate::ULE
                }
            }
            DSLComparison::Gt => {
                if signed {
                    IntPredicate::SGT
                } else {
                    IntPredicate::UGT
                }
            }
            DSLComparison::Gte => {
                if signed {
                    IntPredicate::SGE
                } else {
                    IntPredicate::UGE
                }
            }
        }
    }
}

impl From<Predicate> for DSLComparison {
    fn from(value: Predicate) -> Self {
        match value {
            Predicate::Eq => DSLComparison::Eq,
            Predicate::Ne => DSLComparison::Neq,
            Predicate::Lt => DSLComparison::Lt,
            Predicate::Lte => DSLComparison::Lte,
            Predicate::Gt => DSLComparison::Gt,
            Predicate::Gte => DSLComparison::Gte,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DSLValue {
    name: usize,
    ty: DSLType,
}

impl DSLValue {
    pub fn new(ctx: &mut DSLContext, ty: DSLType) -> Self {
        Self {
            name: ctx.next_symbol(),
            ty,
        }
    }

    pub fn expr(&self) -> DSLExpr {
        DSLExpr::Value(self.clone())
    }

    pub fn u64(value: u64) -> Self {
        Self {
            name: 0,
            ty: DSLType::ConstScalar(Arc::new(UInt64Array::new_scalar(value))),
        }
    }

    pub fn u32(value: u32) -> Self {
        Self {
            name: 0,
            ty: DSLType::ConstScalar(Arc::new(UInt32Array::new_scalar(value))),
        }
    }
}

pub struct DSLFunction {
    name: String,
    params: Vec<DSLValue>,
    body: Vec<DSLStmt>,
    ret: Vec<OutputSpec>,
}

impl DSLFunction {
    pub fn new<T: Into<String>>(name: T) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            body: Vec::new(),
            ret: Vec::new(),
        }
    }

    pub fn add_arg(&mut self, ctx: &mut DSLContext, ty: DSLType) -> DSLValue {
        let value = DSLValue::new(ctx, ty);
        self.params.push(value.clone());
        value
    }

    pub fn add_body(&mut self, stmt: DSLStmt) {
        self.body.push(stmt);
    }

    pub fn add_ret<S: Into<String>>(&mut self, ty: WriterSpec, length_tag: S) {
        self.ret.push(OutputSpec::new(ty, length_tag));
    }

    pub fn get_param(&self, n: usize) -> Option<DSLValue> {
        self.params.get(n).cloned()
    }

    pub fn accessed_parameters(&self) -> HashSet<usize> {
        let mut params = HashSet::new();
        for stmt in self.body.iter() {
            stmt.accessed_parameters(&mut params);
        }
        params
    }

    pub fn iterated_parameters(&self) -> HashSet<usize> {
        let mut params = HashSet::new();
        for stmt in self.body.iter() {
            stmt.iterated_parameters(&mut params);
        }
        params
    }
}

pub enum DSLStmt {
    Emit {
        index: DSLExpr,
        value: DSLExpr,
    },
    Set {
        buf: DSLValue,
        index: DSLExpr,
        value: DSLExpr,
    },
    If {
        cond: DSLExpr,
        then: Vec<DSLStmt>,
    },
    ForEach(DSLForEach),
    ForRange(DSLForRange),
}

impl Into<Vec<DSLStmt>> for DSLStmt {
    fn into(self) -> Vec<DSLStmt> {
        vec![self]
    }
}

impl DSLStmt {
    pub fn for_each<
        S: Into<Vec<DSLStmt>>,
        F: FnOnce(&[DSLValue]) -> Result<S, ArrowKernelError>,
    >(
        ctx: &mut DSLContext,
        to_iter: &[DSLValue],
        f: F,
    ) -> Result<DSLStmt, ArrowKernelError> {
        let loop_vars: Vec<DSLValue> = to_iter
            .iter()
            .map(|dv| {
                dv.ty
                    .iter_type()
                    .ok_or_else(|| ArrowKernelError::NonIterableType(dv.ty.clone()))
                    .map(|dt| DSLValue::new(ctx, dt))
            })
            .try_collect()?;

        let body = f(&loop_vars)?;

        Ok(DSLStmt::ForEach(DSLForEach {
            loop_vars,
            iterators: to_iter.iter().cloned().collect_vec(),
            body: body.into(),
        }))
    }

    pub fn for_range<S: Into<Vec<DSLStmt>>, F: FnOnce(&DSLValue) -> Result<S, ArrowKernelError>>(
        ctx: &mut DSLContext,
        start: DSLExpr,
        end: DSLExpr,
        f: F,
    ) -> Result<DSLStmt, ArrowKernelError> {
        let loop_var = DSLValue::new(ctx, DSLType::Primitive(PrimitiveType::U64));
        let body = f(&loop_var)?;

        Ok(DSLStmt::ForRange(DSLForRange {
            loop_var,
            start,
            end,
            body: body.into(),
        }))
    }

    pub fn cond<S: Into<Vec<DSLStmt>>>(
        cond: DSLExpr,
        then: S,
    ) -> Result<DSLStmt, ArrowKernelError> {
        Ok(DSLStmt::If {
            cond,
            then: then.into(),
        })
    }

    pub fn emit(index: u32, value: DSLExpr) -> Result<DSLStmt, ArrowKernelError> {
        Ok(DSLStmt::Emit {
            index: DSLExpr::Value(DSLValue::u32(index)),
            value,
        })
    }

    pub fn emit_dynamic(index: DSLExpr, value: DSLExpr) -> Result<DSLStmt, ArrowKernelError> {
        Ok(DSLStmt::Emit { index, value })
    }

    pub fn set(
        buf: &DSLValue,
        index: &DSLExpr,
        value: &DSLExpr,
    ) -> Result<DSLStmt, ArrowKernelError> {
        Ok(DSLStmt::Set {
            buf: buf.clone(),
            index: index.clone(),
            value: value.clone(),
        })
    }

    pub fn accessed_parameters(&self, params: &mut HashSet<usize>) {
        match self {
            DSLStmt::If { cond, then } => {
                cond.accessed_parameters(params);
                for stmt in then.iter() {
                    stmt.accessed_parameters(params);
                }
            }
            DSLStmt::ForEach(dslfor) => {
                for stmt in dslfor.body.iter() {
                    stmt.accessed_parameters(params);
                }
            }
            DSLStmt::ForRange(dslfor) => {
                for stmt in dslfor.body.iter() {
                    stmt.accessed_parameters(params);
                }
            }
            DSLStmt::Emit { index, value } => {
                index.accessed_parameters(params);
                value.accessed_parameters(params);
            }
            DSLStmt::Set { buf, index, value } => {
                params.insert(buf.name);
                index.accessed_parameters(params);
                value.accessed_parameters(params);
            }
        }
    }

    pub fn iterated_parameters(&self, params: &mut HashSet<usize>) {
        match self {
            DSLStmt::ForEach(dslfor) => {
                params.extend(dslfor.iterators.iter().map(|x| x.name));
                for stmt in dslfor.body.iter() {
                    stmt.iterated_parameters(params);
                }
            }
            DSLStmt::If { then, .. } => {
                for stmt in then.iter() {
                    stmt.iterated_parameters(params);
                }
            }
            _ => {}
        }
    }
}

pub struct DSLForEach {
    /// variables inside the for loop
    loop_vars: Vec<DSLValue>,

    /// values to iterate over
    iterators: Vec<DSLValue>,
    body: Vec<DSLStmt>,
}

pub struct DSLForRange {
    /// variable inside the for loop
    loop_var: DSLValue,

    start: DSLExpr,
    end: DSLExpr,
    body: Vec<DSLStmt>,
}

#[derive(Clone, Debug)]
pub enum DSLExpr {
    Compare(DSLComparison, Box<DSLExpr>, Box<DSLExpr>),
    At(Box<DSLValue>, Vec<Box<DSLExpr>>),
    Value(DSLValue),
    Cast(Box<DSLExpr>, PrimitiveType),
    CastToBool(Box<DSLExpr>),
    ArithBinOp(DSLArithBinOp, Box<DSLExpr>, Box<DSLExpr>),
    BitwiseBinOp(DSLBitwiseBinOp, Box<DSLExpr>, Box<DSLExpr>),
    Len(Box<DSLValue>),
}

impl DSLExpr {
    pub fn cmp(&self, other: &DSLExpr, op: DSLComparison) -> Result<DSLExpr, ArrowKernelError> {
        Ok(DSLExpr::Compare(
            op,
            Box::new(self.clone()),
            Box::new(other.clone()),
        ))
    }

    pub fn at(&self, index: &DSLExpr) -> Result<DSLExpr, ArrowKernelError> {
        match self {
            DSLExpr::Value(v) => Ok(DSLExpr::At(
                Box::new(v.clone()),
                vec![Box::new(index.clone())],
            )),
            DSLExpr::At(base2d, outer_idx) => {
                if outer_idx.len() != 1 {
                    return Err(ArrowKernelError::InvalidAtSource(self.clone()));
                }
                let idxes = vec![outer_idx[0].clone(), Box::new(index.clone())];
                Ok(DSLExpr::At(base2d.clone(), idxes))
            }
            _ => Err(ArrowKernelError::InvalidAtSource(self.clone())),
        }
    }

    pub fn primitive_cast(&self, ty: PrimitiveType) -> Result<DSLExpr, ArrowKernelError> {
        if self.get_type() == DSLType::Primitive(ty) {
            return Ok(self.clone());
        }

        match self.get_type() {
            DSLType::ConstScalar(_) | DSLType::Boolean | DSLType::Primitive(_) => {
                Ok(DSLExpr::Cast(Box::new(self.clone()), ty))
            }
            _ => Err(ArrowKernelError::InvalidCast(
                self.get_type(),
                DSLType::Primitive(ty),
            )),
        }
    }

    pub fn cast_to_bool(&self) -> Result<DSLExpr, ArrowKernelError> {
        match self.get_type() {
            DSLType::Boolean => Ok(self.clone()),
            DSLType::Primitive(_) => Ok(DSLExpr::CastToBool(Box::new(self.clone()))),
            DSLType::ConstScalar(_) => Ok(DSLExpr::CastToBool(Box::new(self.clone()))),
            _ => Err(ArrowKernelError::DSLInvalidType(
                "cannot cast type to bool",
                self.get_type(),
            )),
        }
    }

    pub fn arith(&self, op: DSLArithBinOp, rhs: DSLExpr) -> Result<DSLExpr, ArrowKernelError> {
        if self.get_type() != rhs.get_type() {
            return Err(ArrowKernelError::DSLTypeMismatch(
                "arith bin op",
                self.get_type(),
                rhs.get_type(),
            ));
        }

        Ok(DSLExpr::ArithBinOp(
            op,
            Box::new(self.clone()),
            Box::new(rhs.clone()),
        ))
    }

    pub fn bitwise(&self, op: DSLBitwiseBinOp, rhs: DSLExpr) -> Result<DSLExpr, ArrowKernelError> {
        if self.get_type() != rhs.get_type() {
            return Err(ArrowKernelError::DSLTypeMismatch(
                "bitwise bin op",
                self.get_type(),
                rhs.get_type(),
            ));
        }

        Ok(DSLExpr::BitwiseBinOp(
            op,
            Box::new(self.clone()),
            Box::new(rhs.clone()),
        ))
    }

    pub fn len(&self) -> Result<DSLExpr, ArrowKernelError> {
        match self {
            DSLExpr::Value(v) => Ok(DSLExpr::Len(Box::new(v.clone()))),
            _ => Err(ArrowKernelError::DSLInvalidType(
                "cannot get length of this type",
                self.get_type(),
            )),
        }
    }

    pub fn get_type(&self) -> DSLType {
        match self {
            DSLExpr::Compare(..) => DSLType::Boolean,
            DSLExpr::At(val, idxes) => {
                let mut t = val.ty.clone();
                for _ in idxes {
                    t = t
                        .iter_type()
                        .expect(&format!("unable to iterate type: {:?}", t));
                }
                t
            }
            DSLExpr::Value(v) => v.ty.clone(),
            DSLExpr::BitwiseBinOp(_, v, _) | DSLExpr::ArithBinOp(_, v, _) => v.get_type().clone(),
            DSLExpr::Cast(_, pt) => DSLType::Primitive(*pt),
            DSLExpr::CastToBool(_) => DSLType::Boolean,
            DSLExpr::Len(_) => DSLType::Primitive(PrimitiveType::U64),
        }
    }

    fn accessed_parameters(&self, params: &mut HashSet<usize>) {
        match self {
            DSLExpr::Compare(_, l, r)
            | DSLExpr::ArithBinOp(_, l, r)
            | DSLExpr::BitwiseBinOp(_, l, r) => {
                l.accessed_parameters(params);
                r.accessed_parameters(params);
            }
            DSLExpr::At(arr, idx) => {
                params.insert(arr.name);
                idx.iter().for_each(|i| i.accessed_parameters(params));
            }
            DSLExpr::Value(_) | DSLExpr::Len(_) => {}
            DSLExpr::Cast(val, _) => val.accessed_parameters(params),
            DSLExpr::CastToBool(val) => val.accessed_parameters(params),
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            DSLExpr::Value(v) => match &v.ty {
                DSLType::ConstScalar(v) => v
                    .get()
                    .0
                    .as_primitive_opt::<UInt32Type>()
                    .map(|v| v.value(0)),
                _ => None,
            },
            _ => None,
        }
    }
}

impl From<usize> for DSLExpr {
    fn from(value: usize) -> Self {
        DSLExpr::Value(DSLValue::u64(value as u64))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use arrow_array::{
        cast::AsArray,
        types::{Int32Type, UInt64Type},
        ArrayRef, BooleanArray, Datum, Int32Array, StringArray, UInt32Array,
    };
    use arrow_buffer::{Buffer, MutableBuffer, ScalarBuffer};
    use arrow_schema::DataType;
    use itertools::Itertools;

    use crate::{
        compiled_kernels::{
            cast::coalesce_type,
            dsl::DSLKernel,
            dsl2::{
                buffer::DSLBuffer, compiler::compile, writers::WriterSpec, DSLArgument,
                DSLComparison, DSLContext, DSLForEach, DSLFunction, DSLStmt, DSLType, OutputSpec,
            },
        },
        PrimitiveType,
    };

    #[test]
    fn test_dsl2_lt() {
        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("lt");
        let arr = Int32Array::new_null(0);
        let scal = Int32Array::new_scalar(5);

        let arr1 = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::I32, "n"));
        let scal_arg = func.add_arg(&mut ctx, DSLType::scalar_of(PrimitiveType::I32));

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arr1, scal_arg], |loop_vars| {
                let val = &loop_vars[0];
                let sca = &loop_vars[1];
                let cmp = val.expr().cmp(&sca.expr(), DSLComparison::Lt)?;
                DSLStmt::emit(0, cmp)
            })
            .unwrap(),
        );
        func.add_ret(WriterSpec::Boolean, "n");
        assert!(func.accessed_parameters().is_empty());

        let func = compile(func, dsl_args![arr, scal]).unwrap();

        let arr1 = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let scal = Int32Array::new_scalar(3);
        let result = func.run(&dsl_args![arr1, scal]).unwrap();
        assert_eq!(result.len(), 1);
        let out = result.into_iter().next().unwrap();
        let out = out.as_boolean();
        assert_eq!(
            out.into_iter().map(|x| x.unwrap()).collect_vec(),
            vec![true, true, false, false, false]
        );

        let arr1 = Int32Array::from(vec![10, 20, 30, 40]);
        let scal = Int32Array::new_scalar(3);
        let result = func.run(&dsl_args![arr1, scal]).unwrap();
        assert_eq!(result.len(), 1);
        let out = result.into_iter().next().unwrap();
        let out = out.as_boolean();
        assert_eq!(
            out.into_iter().map(|x| x.unwrap()).collect_vec(),
            vec![false, false, false, false]
        );
    }

    #[test]
    fn test_dsl2_run_into_appends() {
        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("append");
        let arr_input = Int32Array::new_null(0);

        let arr = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::I32, "n"));
        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arr], |loop_vars| {
                DSLStmt::emit(0, loop_vars[0].expr())
            })
            .unwrap(),
        );
        func.add_ret(WriterSpec::Primitive(PrimitiveType::I32), "n");

        let func = compile(func, dsl_args![arr_input]).unwrap();
        let arr1 = Int32Array::from(vec![1, 2, 3]);
        let arr2 = Int32Array::from(vec![4, 5, 6]);
        let mut output =
            OutputSpec::new(WriterSpec::Primitive(PrimitiveType::I32), "n").allocate(6);

        func.run_into(&dsl_args![arr1], std::slice::from_mut(&mut output))
            .unwrap();
        func.run_into(&dsl_args![arr2], std::slice::from_mut(&mut output))
            .unwrap();

        let out = output.into_array_ref(None);
        let out = out.as_primitive::<Int32Type>();
        assert_eq!(
            out.into_iter().map(|x| x.unwrap()).collect_vec(),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn test_dsl2_take() {
        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("take");
        let arr_input = Int32Array::new_null(0);
        let idxs_input = UInt32Array::new_null(0);

        let arr1 = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::I32, "n"));
        let idxs = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::U32, "m"));

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[idxs], |loop_vars| {
                let idx = &loop_vars[0];
                let val = arr1
                    .expr()
                    .at(&idx.expr().primitive_cast(PrimitiveType::U64)?)?;
                DSLStmt::emit(0, val)
            })
            .unwrap(),
        );
        func.add_ret(WriterSpec::Primitive(PrimitiveType::I32), "m");

        let func = compile(func, dsl_args![arr_input, idxs_input]).unwrap();
        let arr1 = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let arr2 = UInt32Array::from(vec![0, 4, 4, 1]);
        let result = func.run(&dsl_args![arr1, arr2]).unwrap();
        assert_eq!(result.len(), 1);
        let out = result.into_iter().next().unwrap();
        let out = out.as_primitive::<Int32Type>();
        assert_eq!(
            out.into_iter().map(|x| x.unwrap()).collect_vec(),
            vec![10, 50, 50, 20]
        );

        let arr2 = UInt32Array::from(vec![1, 2, 1]);
        let result = func.run(&dsl_args![arr1, arr2]).unwrap();
        assert_eq!(result.len(), 1);
        let out = result.into_iter().next().unwrap();
        let out = out.as_primitive::<Int32Type>();
        assert_eq!(
            out.into_iter().map(|x| x.unwrap()).collect_vec(),
            vec![20, 30, 20]
        );
    }

    #[test]
    fn test_dsl2_filter() {
        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("filter");
        let arr_input = Int32Array::new_null(0);
        let bools_input = BooleanArray::new_null(0);

        let arr1 = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::I32, "n"));
        let bools = func.add_arg(&mut ctx, DSLType::set_bits("n"));

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[bools], |loop_vars| {
                let idx = &loop_vars[0];
                let val = arr1.expr().at(&idx.expr())?;
                DSLStmt::emit(0, val)
            })
            .unwrap(),
        );
        func.add_ret(WriterSpec::Primitive(PrimitiveType::I32), "<= n");

        let func = compile(func, dsl_args![arr_input, bools_input]).unwrap();

        let arr1 = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let arr2 = BooleanArray::from(vec![true, false, false, true, false]);
        let result = func.run(&dsl_args![arr1, arr2]).unwrap();
        assert_eq!(result.len(), 1);
        let out = result.into_iter().next().unwrap();
        let out = out.as_primitive::<Int32Type>();
        assert_eq!(
            out.into_iter().map(|x| x.unwrap()).collect_vec(),
            vec![10, 40]
        );
    }

    #[test]
    fn test_dsl2_partition() {
        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("partition");
        let arr_input = Int32Array::new_null(0);
        let part_idx_input = UInt32Array::new_null(0);

        let arr1 = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::I32, "n"));
        let part_idx = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::U32, "n"));

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arr1, part_idx], |loop_vars| {
                let val = &loop_vars[0];
                let idx = &loop_vars[1];
                DSLStmt::emit_dynamic(idx.expr(), val.expr())
            })
            .unwrap(),
        );
        func.add_ret(WriterSpec::Primitive(PrimitiveType::I32), "<= n");
        func.add_ret(WriterSpec::Primitive(PrimitiveType::I32), "<= n");
        func.add_ret(WriterSpec::Primitive(PrimitiveType::I32), "<= n");

        let func = compile(func, dsl_args![arr_input, part_idx_input]).unwrap();

        let arr1 = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let arr2 = UInt32Array::from(vec![0, 0, 1, 2, 1]);
        let result = func.run(&dsl_args![arr1, arr2]).unwrap();
        assert_eq!(result.len(), 3);
        let first = result[0].clone();
        let second = result[1].clone();
        let third = result[2].clone();
        let first = first.as_primitive::<Int32Type>();
        let second = second.as_primitive::<Int32Type>();
        let third = third.as_primitive::<Int32Type>();

        assert_eq!(first.iter().map(|x| x.unwrap()).collect_vec(), vec![10, 20]);
        assert_eq!(
            second.iter().map(|x| x.unwrap()).collect_vec(),
            vec![30, 50]
        );
        assert_eq!(third.iter().map(|x| x.unwrap()).collect_vec(), vec![40]);
    }

    #[test]
    fn test_dsl2_merge() {
        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("merge");
        let part0 = Int32Array::new_null(0);
        let part1 = Int32Array::new_null(0);
        let part2 = Int32Array::new_null(0);
        let part_idxes_input = UInt32Array::new_null(0);
        let elem_idxes_input = UInt32Array::new_null(0);

        let arr = func.add_arg(&mut ctx, DSLType::two_d_array_of(PrimitiveType::I32));
        let part_idxes = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::U32, "n"));
        let elem_idxes = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::U32, "n"));

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[part_idxes, elem_idxes], |loop_vars| {
                let part_idx = &loop_vars[0].expr().primitive_cast(PrimitiveType::U64)?;
                let elem_idx = &loop_vars[1].expr().primitive_cast(PrimitiveType::U64)?;
                let val = arr.expr().at(part_idx)?.at(elem_idx)?;
                DSLStmt::emit(0, val)
            })
            .unwrap(),
        );
        func.add_ret(WriterSpec::Primitive(PrimitiveType::I32), "n");

        let func = compile(
            func,
            dsl_args![[part0, part1, part2], part_idxes_input, elem_idxes_input],
        )
        .unwrap();

        let part1 = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let part2 = Int32Array::from(vec![6, 7, 8, 9, 10]);
        let part3 = Int32Array::from(vec![11, 12, 13, 14, 15]);
        let part_idx = UInt32Array::from(vec![0, 0, 0, 1, 2, 0]);
        let elem_idx = UInt32Array::from(vec![0, 1, 2, 0, 1, 2]);
        let result = func
            .run(&dsl_args![[part1, part2, part3], part_idx, elem_idx])
            .unwrap();
        assert_eq!(result.len(), 1);
        let result = result.into_iter().next().unwrap();
        let result = result.as_primitive::<Int32Type>();
        let result = result.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(result, vec![1, 2, 3, 6, 12, 3]);
    }

    #[test]
    fn test_dsl2_max_agg() {
        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("max_agg");
        let arr_input = Int32Array::new_null(0);
        let tic_input = UInt32Array::new_null(0);
        let mut buf_input = DSLBuffer::new(PrimitiveType::I32, 0);

        let arr = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::I32, "n"));
        let tic = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::U32, "n"));
        let buf = func.add_arg(&mut ctx, DSLType::buffer_of(PrimitiveType::I32, "k"));

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arr, tic], |loop_vars| {
                let new_val = loop_vars[0].expr();
                let tic_val = loop_vars[1].expr().primitive_cast(PrimitiveType::U64)?;
                let old_val = buf.expr().at(&tic_val)?;

                let cmp = new_val.cmp(&old_val, DSLComparison::Gt)?;
                DSLStmt::cond(cmp, DSLStmt::set(&buf, &tic_val, &new_val)?)
            })
            .unwrap(),
        );

        let func = compile(func, dsl_args![arr_input, tic_input, &mut buf_input]).unwrap();

        let mut buf = DSLBuffer::new(PrimitiveType::I32, 3);
        let arr = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let tic = UInt32Array::from(vec![0, 1, 0, 2, 1]);

        func.run(&dsl_args![arr, tic, &mut buf]).unwrap();

        let res = buf.into_array();
        let res = res.as_primitive::<Int32Type>().clone();
        assert_eq!(res, Int32Array::from(vec![30, 50, 40]));
    }

    #[test]
    fn test_dsl2_partial_strcmp() {
        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("partial_strcmp");
        let arr_input = StringArray::new_null(0);
        let sca_input = StringArray::new_scalar("test");
        let sel_input = BooleanArray::new_null(0);

        let arr = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::P64x2, "n"));
        let sca = func.add_arg(&mut ctx, DSLType::scalar_of(PrimitiveType::P64x2));
        let sel = func.add_arg(&mut ctx, DSLType::set_bits("m"));

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[sel, sca], |loop_vars| {
                let idx = loop_vars[0].expr();
                let sca = loop_vars[1].expr();
                let val = arr.expr().at(&idx)?;
                let cmp = val.cmp(&sca, DSLComparison::Eq)?;

                DSLStmt::cond(cmp, DSLStmt::emit(0, idx)?)
            })
            .unwrap(),
        );
        func.add_ret(WriterSpec::Primitive(PrimitiveType::U64), "<= m");

        let func = compile(func, dsl_args![arr_input, sca_input, sel_input]).unwrap();

        let arr = StringArray::from(vec!["this", "this", "a", "test"]);
        let sca = StringArray::new_scalar("this");
        let sel = BooleanArray::from(vec![true, false, true, true]);
        let result = func.run(&dsl_args![arr, sca, sel]).unwrap();
        assert_eq!(result.len(), 1);
        let result = result.into_iter().next().unwrap();
        let result = result.as_primitive::<UInt64Type>();
        let result = result.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_dsl2_concat_strs() {
        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("concat_strs");
        let arr1_input = StringArray::new_null(0);
        let arr2_input = StringArray::new_null(0);

        let arr1 = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::P64x2, "n"));
        let arr2 = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::P64x2, "m"));

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arr1], |loop_vars| {
                DSLStmt::emit(0, loop_vars[0].expr())
            })
            .unwrap(),
        );
        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arr2], |loop_vars| {
                DSLStmt::emit(0, loop_vars[0].expr())
            })
            .unwrap(),
        );
        func.add_ret(WriterSpec::String, "n + m");

        let func = compile(func, dsl_args![arr1_input, arr2_input]).unwrap();
        let arr = StringArray::from(vec!["one", "two", "three"]);
        let result = func.run(&dsl_args![arr, arr]).unwrap();
        assert_eq!(result.len(), 1);
        let result = result.into_iter().next().unwrap();
        let result = coalesce_type(result, &DataType::Utf8).unwrap();
        let result = result.as_string::<i32>();
        let result = result.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(result, vec!["one", "two", "three", "one", "two", "three"]);
    }
}
