use paste::paste;
use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use inkwell::{
    intrinsics::Intrinsic,
    llvm_sys::{
        LLVMFastMathAllowContract, LLVMFastMathAllowReassoc, LLVMFastMathAllowReciprocal,
        LLVMFastMathApproxFunc,
    },
    types::BasicTypeEnum,
    values::{BasicValue, BasicValueEnum, VectorValue},
    IntPredicate,
};

use super::{
    context::{CompilationContext, VEC_SIZE},
    errors::DSLError,
    string_funcs::{add_str_endswith, add_str_startswith},
    types::{base_type, KernelInput, KernelInputType},
};
use crate::{
    compiled_kernels::{
        cmp::{add_float_vec_to_int_vec, add_memcmp},
        gen_convert_numeric_vec,
    },
    declare_blocks, increment_pointer, pointer_diff, ComparisonType, Predicate, PrimitiveSuperType,
    PrimitiveType,
};

macro_rules! vec_or_scalar_op {
    ($builder: expr, $func_name: ident, $v1: expr, $v2: expr) => {{
        let instr = match ($v1, $v2) {
            (BasicValueEnum::IntValue(v1), BasicValueEnum::IntValue(v2)) => Ok(
                paste! { $builder. [<build_int_ $func_name>] (v1, v2, "int_op") }
                    .unwrap()
                    .as_basic_value_enum(),
            ),
            (BasicValueEnum::FloatValue(v1), BasicValueEnum::FloatValue(v2)) => Ok(
                paste! { $builder. [<build_float_ $func_name>] (v1, v2, "float_op") }
                    .unwrap()
                    .as_basic_value_enum(),
            ),
            (BasicValueEnum::VectorValue(v1), BasicValueEnum::VectorValue(v2)) => {
                match (
                    v1.get_type().get_element_type(),
                    v2.get_type().get_element_type(),
                ) {
                    (BasicTypeEnum::IntType(_), BasicTypeEnum::IntType(_)) => Ok(
                        paste! { $builder. [<build_int_ $func_name>] (v1, v2, "int_vec_op") }
                            .unwrap()
                            .as_basic_value_enum(),
                    ),
                    (BasicTypeEnum::FloatType(_), BasicTypeEnum::FloatType(_)) => Ok(
                        paste! { $builder. [<build_float_ $func_name>] (v1, v2, "float_vec_op") }
                            .unwrap()
                            .as_basic_value_enum(),
                    ),
                    _ => Err(DSLError::TypeMismatch(format!(
                        "invalid vector types for {}",
                        stringify!($func_name)
                    ))),
                }
            }
            _ => panic!("Unsupported types for addition"),
        };

        if let Ok(instr) = instr {
            instr.as_instruction_value().unwrap().set_fast_math_flags(
                LLVMFastMathAllowContract
                    | LLVMFastMathAllowReassoc
                    | LLVMFastMathAllowReciprocal
                    | LLVMFastMathApproxFunc,
            );
        }

        instr
    }};
}

macro_rules! signed_vec_or_scalar_op {
    ($builder: expr, $func_name: ident, $signed: expr, $v1: expr, $v2: expr) => {{
        let instr = match ($v1, $v2) {
            (BasicValueEnum::IntValue(v1), BasicValueEnum::IntValue(v2)) => Ok(
                if $signed {
                    paste! { $builder. [<build_int_signed_ $func_name>] (v1, v2, "int_op") }
                        .unwrap()
                        .as_basic_value_enum()
                } else {
                    paste! { $builder. [<build_int_unsigned_ $func_name>] (v1, v2, "int_op") }
                        .unwrap()
                        .as_basic_value_enum()
                }
            ),
            (BasicValueEnum::FloatValue(v1), BasicValueEnum::FloatValue(v2)) => Ok(
                paste! { $builder. [<build_float_ $func_name>] (v1, v2, "float_op") }
                    .unwrap()
                    .as_basic_value_enum()
            ),
            (BasicValueEnum::VectorValue(v1), BasicValueEnum::VectorValue(v2)) => {
                match (
                    v1.get_type().get_element_type(),
                    v2.get_type().get_element_type(),
                ) {
                    (BasicTypeEnum::IntType(_), BasicTypeEnum::IntType(_)) => Ok(
                        if $signed {
                            paste! { $builder. [<build_int_signed_ $func_name>] (v1, v2, "int_vec_op") }
                        } else {
                            paste! { $builder. [<build_int_unsigned_ $func_name>] (v1, v2, "int_vec_op") }
                        }
                        .unwrap()
                        .as_basic_value_enum()
                    ),
                    (BasicTypeEnum::FloatType(_), BasicTypeEnum::FloatType(_)) => Ok(
                        paste! { $builder. [<build_float_ $func_name>] (v1, v2, "float_vec_op") }
                            .unwrap()
                            .as_basic_value_enum()
                    ),
                    _ => Err(DSLError::TypeMismatch(format!(
                        "invalid vector types for {}",
                        stringify!($func_name)
                    ))),
                }
            }
            _ => panic!("Unsupported types for addition"),
        };

        if let Ok(instr) = instr {
            instr.as_instruction_value().unwrap().set_fast_math_flags(
                LLVMFastMathAllowContract
                    | LLVMFastMathAllowReassoc
                    | LLVMFastMathAllowReciprocal
                    | LLVMFastMathApproxFunc,
            );
        }

        instr
    }};
}

// Shared state reused while recursively lowering expressions.
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
    Powi(Box<KernelExpression<'a>>, i32),
    Sqrt(Box<KernelExpression<'a>>),
    VectorSum(Box<KernelExpression<'a>>),
    Splat(Box<KernelExpression<'a>>, usize),
    StartsWith(Box<KernelExpression<'a>>, Box<KernelExpression<'a>>),
    EndsWith(Box<KernelExpression<'a>>, Box<KernelExpression<'a>>),
    StrLen(Box<KernelExpression<'a>>),
    Substring {
        expr: Box<KernelExpression<'a>>,
        start: Box<KernelExpression<'a>>,
        len: Box<KernelExpression<'a>>,
        check: bool,
    },
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
    pub fn powi(&self, power: i32) -> KernelExpression<'a> {
        KernelExpression::Powi(Box::new(self.clone()), power)
    }
    pub fn sqrt(&self) -> KernelExpression<'a> {
        KernelExpression::Sqrt(Box::new(self.clone()))
    }

    /// Sums all the elements into a vector, mapping the vector to a scalar
    pub fn vec_sum(&self) -> KernelExpression<'a> {
        KernelExpression::VectorSum(Box::new(self.clone()))
    }

    /// Turns a scalar into a vector by repeating it. The resulting vector's size is `size`.
    pub fn splat(&self, size: usize) -> KernelExpression<'a> {
        KernelExpression::Splat(Box::new(self.clone()), size)
    }

    pub fn starts_with(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::StartsWith(Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn ends_with(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::EndsWith(Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn str_len(&self) -> KernelExpression<'a> {
        KernelExpression::StrLen(Box::new(self.clone()))
    }
    pub fn substring(
        &self,
        start: &KernelExpression<'a>,
        len: &KernelExpression<'a>,
    ) -> KernelExpression<'a> {
        KernelExpression::Substring {
            expr: Box::new(self.clone()),
            start: Box::new(start.clone()),
            len: Box::new(len.clone()),
            check: true,
        }
    }
    pub unsafe fn substring_unchecked(
        &self,
        start: &KernelExpression<'a>,
        len: &KernelExpression<'a>,
    ) -> KernelExpression<'a> {
        KernelExpression::Substring {
            expr: Box::new(self.clone()),
            start: Box::new(start.clone()),
            len: Box::new(len.clone()),
            check: false,
        }
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
            KernelExpression::Not(c) | KernelExpression::Powi(c, _) | KernelExpression::Sqrt(c) => {
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
            KernelExpression::Convert(expr, ..)
            | KernelExpression::StrLen(expr)
            | KernelExpression::VectorSum(expr)
            | KernelExpression::Splat(expr, _) => {
                f(self);
                expr.descend(f);
            }
            KernelExpression::IntConst(..) => f(self),
            KernelExpression::Add(lhs, rhs)
            | KernelExpression::Sub(lhs, rhs)
            | KernelExpression::Mul(lhs, rhs)
            | KernelExpression::Div(lhs, rhs)
            | KernelExpression::Rem(lhs, rhs)
            | KernelExpression::StartsWith(lhs, rhs)
            | KernelExpression::EndsWith(lhs, rhs) => {
                f(self);
                lhs.descend(f);
                rhs.descend(f);
            }
            KernelExpression::Substring {
                expr, start, len, ..
            } => {
                f(self);
                expr.descend(f);
                start.descend(f);
                len.descend(f);
            }
        }
    }

    pub(super) fn iterated_indexes(&self) -> Vec<(KernelInputType, usize)> {
        let mut h = BTreeSet::new();
        self.descend(&mut |e| {
            if let KernelExpression::Item(kernel_input) = e {
                h.insert((kernel_input.input_type(), kernel_input.index()));
            }
        });
        h.into_iter().collect()
    }

    pub(super) fn accessed_indexes(&self) -> Vec<usize> {
        let mut h = BTreeSet::new();
        self.descend(&mut |e| {
            if let KernelExpression::At { iter, .. } = e {
                h.insert(iter.index());
            }
        });
        h.into_iter().collect()
    }

    pub(super) fn get_type(&self) -> DataType {
        match self {
            KernelExpression::Item(kernel_input) => base_type(&kernel_input.data_type()),
            KernelExpression::Truncate(..) => DataType::Binary,
            KernelExpression::Select { v1, .. } => v1.get_type(),
            KernelExpression::Cmp(..)
            | KernelExpression::And(..)
            | KernelExpression::Or(..)
            | KernelExpression::Not(..)
            | KernelExpression::StartsWith(..)
            | KernelExpression::EndsWith(..) => DataType::Boolean,
            KernelExpression::IntConst(..) | KernelExpression::StrLen(..) => DataType::UInt64,
            KernelExpression::At { iter, .. } => base_type(&iter.data_type()),
            KernelExpression::Convert(_expr, pt) => pt.as_arrow_type(),
            KernelExpression::Add(lhs, _)
            | KernelExpression::Sub(lhs, _)
            | KernelExpression::Mul(lhs, _)
            | KernelExpression::Div(lhs, _)
            | KernelExpression::Rem(lhs, _)
            | KernelExpression::Powi(lhs, _)
            | KernelExpression::Sqrt(lhs) => lhs.get_type(),
            KernelExpression::VectorSum(expr) => PrimitiveType::for_arrow_type(&expr.get_type())
                .list_type_into_inner()
                .as_arrow_type(),
            KernelExpression::Splat(expr, s) => DataType::FixedSizeList(
                Arc::new(Field::new_list_field(expr.get_type(), false)),
                *s as i32,
            ),
            KernelExpression::Substring { expr, .. } => expr.get_type(),
        }
    }

    fn is_string(&self) -> bool {
        match self.get_type() {
            DataType::Binary
            | DataType::FixedSizeBinary(_)
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View => true,
            _ => false,
        }
    }

    pub(super) fn compile_block<'ctx, 'b>(
        &self,
        compilation: &CompilationContext<'ctx, 'b>,
    ) -> Result<VectorValue<'ctx>, DSLError> {
        let ctx = compilation.llvm_ctx;
        let build = compilation.builder;
        let vec_bufs = compilation.vec_bufs;
        let iter_llvm_types = compilation.iter_llvm_types;
        let iter_ptrs = compilation.iter_ptrs;
        let blocked_access_funcs = compilation.blocked_access_funcs;
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
            KernelExpression::And(..) | KernelExpression::Or(..) => {
                Err(DSLError::NotVectorizable("short-circuit operator"))
            }

            KernelExpression::Not(c) => {
                let c = c.compile_block(compilation)?;
                Ok(build.build_not(c, "not").unwrap())
            }
            KernelExpression::Select { cond, v1, v2 } => {
                let cond = cond.compile_block(compilation)?;
                let v1 = v1.compile_block(compilation)?;
                let v2 = v2.compile_block(compilation)?;
                Ok(build
                    .build_select(cond, v1, v2, "select")
                    .unwrap()
                    .into_vector_value())
            }
            KernelExpression::Convert(c, tar_pt) => {
                let in_ty = PrimitiveType::for_arrow_type(&c.get_type());
                let c = c.compile_block(compilation)?;
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

                let lhs = lhs.compile_block(compilation)?;
                let rhs = rhs.compile_block(compilation)?;
                match PrimitiveSuperType::from(pt) {
                    PrimitiveSuperType::Int | PrimitiveSuperType::UInt => {
                        Ok(build.build_int_add(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::Float => {
                        Ok(build.build_float_add(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::String => Err(DSLError::TypeMismatch(
                        "cannot add string types".to_string(),
                    )),
                    PrimitiveSuperType::List(_, _) => Err(DSLError::TypeMismatch(
                        "cannot block add list types".to_string(),
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

                let lhs = lhs.compile_block(compilation)?;
                let rhs = rhs.compile_block(compilation)?;
                match PrimitiveSuperType::from(pt) {
                    PrimitiveSuperType::Int | PrimitiveSuperType::UInt => {
                        Ok(build.build_int_sub(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::Float => {
                        Ok(build.build_float_sub(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::String => Err(DSLError::TypeMismatch(
                        "cannot subtract string types".to_string(),
                    )),
                    PrimitiveSuperType::List(_, _) => Err(DSLError::TypeMismatch(
                        "cannot block add list types".to_string(),
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

                let lhs = lhs.compile_block(compilation)?;
                let rhs = rhs.compile_block(compilation)?;
                match PrimitiveSuperType::from(pt) {
                    PrimitiveSuperType::Int | PrimitiveSuperType::UInt => {
                        Ok(build.build_int_mul(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::Float => {
                        Ok(build.build_float_mul(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::String => Err(DSLError::TypeMismatch(
                        "cannot multiply string types".to_string(),
                    )),
                    PrimitiveSuperType::List(_, _) => Err(DSLError::TypeMismatch(
                        "cannot block add list types".to_string(),
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

                let lhs = lhs.compile_block(compilation)?;
                let rhs = rhs.compile_block(compilation)?;
                match PrimitiveSuperType::from(pt) {
                    PrimitiveSuperType::Int => {
                        Ok(build.build_int_signed_div(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::UInt => {
                        Ok(build.build_int_unsigned_div(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::Float => {
                        Ok(build.build_float_div(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::String => Err(DSLError::TypeMismatch(
                        "cannot divide string types".to_string(),
                    )),
                    PrimitiveSuperType::List(_, _) => Err(DSLError::TypeMismatch(
                        "cannot block add list types".to_string(),
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

                let lhs = lhs.compile_block(compilation)?;
                let rhs = rhs.compile_block(compilation)?;
                match PrimitiveSuperType::from(pt) {
                    PrimitiveSuperType::Int => {
                        Ok(build.build_int_signed_rem(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::UInt => {
                        Ok(build.build_int_unsigned_rem(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::Float => {
                        Ok(build.build_float_rem(lhs, rhs, "add").unwrap())
                    }
                    PrimitiveSuperType::String => Err(DSLError::TypeMismatch(
                        "cannot rem string types".to_string(),
                    )),
                    PrimitiveSuperType::List(_, _) => Err(DSLError::TypeMismatch(
                        "cannot block add list types".to_string(),
                    )),
                }
            }
            KernelExpression::Powi(lhs, power) => {
                let f = Intrinsic::find("llvm.powi").unwrap();
                let func = match lhs.get_type() {
                    DataType::Float16 => f.get_declaration(
                        compilation.llvm_mod,
                        &[
                            ctx.f16_type().vec_type(VEC_SIZE).into(),
                            ctx.i32_type().into(),
                        ],
                    ),
                    DataType::Float32 => f.get_declaration(
                        compilation.llvm_mod,
                        &[
                            ctx.f32_type().vec_type(VEC_SIZE).into(),
                            ctx.i32_type().into(),
                        ],
                    ),
                    DataType::Float64 => f.get_declaration(
                        compilation.llvm_mod,
                        &[
                            ctx.f64_type().vec_type(VEC_SIZE).into(),
                            ctx.i32_type().into(),
                        ],
                    ),
                    _ => {
                        return Err(DSLError::TypeMismatch(
                            "cannot pow-i non-float types".to_string(),
                        ))
                    }
                }
                .unwrap();

                let inp = lhs.compile_block(compilation)?;
                let call = build
                    .build_call(
                        func,
                        &[
                            inp.into(),
                            ctx.i32_type().const_int(*power as u64, true).into(),
                        ],
                        "powi",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_vector_value();
                call.as_instruction_value().unwrap().set_fast_math_flags(
                    LLVMFastMathAllowContract
                        | LLVMFastMathAllowReassoc
                        | LLVMFastMathAllowReciprocal
                        | LLVMFastMathApproxFunc,
                );

                Ok(call)
            }
            KernelExpression::Sqrt(expr) => {
                let f = Intrinsic::find("llvm.sqrt").unwrap();
                let func = match expr.get_type() {
                    DataType::Float16 => f.get_declaration(
                        compilation.llvm_mod,
                        &[ctx.f16_type().vec_type(VEC_SIZE).into()],
                    ),
                    DataType::Float32 => f.get_declaration(
                        compilation.llvm_mod,
                        &[ctx.f32_type().vec_type(VEC_SIZE).into()],
                    ),
                    DataType::Float64 => f.get_declaration(
                        compilation.llvm_mod,
                        &[ctx.f64_type().vec_type(VEC_SIZE).into()],
                    ),
                    _ => {
                        return Err(DSLError::TypeMismatch(
                            "cannot sqrt non-float types".to_string(),
                        ))
                    }
                }
                .unwrap();

                let inp = expr.compile_block(compilation)?;
                let call = build
                    .build_call(func, &[inp.into()], "sqrt")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_vector_value();
                call.as_instruction().unwrap().set_fast_math_flags(
                    LLVMFastMathAllowContract
                        | LLVMFastMathAllowReassoc
                        | LLVMFastMathAllowReciprocal
                        | LLVMFastMathApproxFunc,
                );
                Ok(call)
            }
            KernelExpression::VectorSum(..) => {
                Err(DSLError::NotVectorizable("vector sum operator"))
            }
            KernelExpression::Splat(..) => Err(DSLError::NotVectorizable("splat operator")),
            KernelExpression::Cmp(..) => Err(DSLError::NotVectorizable("cmp operator")),
            KernelExpression::At { iter, idx } => {
                if !idx.get_type().is_integer() {
                    return Err(DSLError::TypeMismatch(format!(
                        "at parameter must be integer, got {}",
                        idx.get_type()
                    )));
                }

                let elem_pt = PrimitiveType::for_arrow_type(&iter.data_type());

                let iter_ptr = *iter_ptrs
                    .get(iter.index())
                    .ok_or(DSLError::InvalidInputIndex(iter.index()))?;

                let idx_vec = idx.compile_block(compilation)?;

                let i64_type = ctx.i64_type();
                let lane_count = idx_vec.get_type().get_size();
                if lane_count != VEC_SIZE {
                    return Err(DSLError::NotVectorizable(
                        "at operator requires 64-lane vector",
                    ));
                }
                let vec_i64_type = i64_type.vec_type(lane_count);
                let idx_vec = if idx_vec.get_type() == vec_i64_type {
                    idx_vec
                } else {
                    build
                        .build_int_cast(idx_vec, vec_i64_type, "idx_to_i64")
                        .unwrap()
                };

                let vec_type =
                    elem_pt
                        .llvm_vec_type(ctx, lane_count)
                        .ok_or(DSLError::NotVectorizable(
                            "at operator requires primitive element type",
                        ))?;

                let access_fn = blocked_access_funcs
                    .get(&(iter.index(), lane_count))
                    .ok_or(DSLError::NotVectorizable(
                        "at operator missing blocked random access",
                    ))?;

                let gathered = build
                    .build_call(
                        *access_fn,
                        &[iter_ptr.into(), idx_vec.into()],
                        "blocked_access",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_vector_value();

                if gathered.get_type() != vec_type {
                    let casted = build
                        .build_bit_cast(gathered, vec_type, "at_cast")
                        .unwrap()
                        .into_vector_value();
                    Ok(casted)
                } else {
                    Ok(gathered)
                }
            }
            KernelExpression::StartsWith(..)
            | KernelExpression::EndsWith(..)
            | KernelExpression::StrLen(..)
            | KernelExpression::Substring { .. } => {
                Err(DSLError::NotVectorizable("string operator"))
            }
        }
    }

    pub(super) fn compile<'ctx, 'b>(
        &self,
        compilation: &CompilationContext<'ctx, 'b>,
    ) -> Result<BasicValueEnum<'ctx>, DSLError> {
        let ctx = compilation.llvm_ctx;
        let llvm_mod = compilation.llvm_mod;
        let build = compilation.builder;
        let bufs = compilation.bufs;
        let accessors = compilation.accessors;
        let iter_ptrs = compilation.iter_ptrs;
        let iter_llvm_types = compilation.iter_llvm_types;
        match self {
            KernelExpression::Item(kernel_input) => {
                let buf = bufs[&kernel_input.index()];
                let llvm_type = iter_llvm_types[&kernel_input.index()];
                Ok(build.build_load(llvm_type, buf, "load").unwrap())
            }
            KernelExpression::Truncate(_kernel_expression, _) => todo!(),
            KernelExpression::Cmp(predicate, lhs, rhs) => {
                let lhs_v = lhs.compile(compilation)?;
                let rhs_v = rhs.compile(compilation)?;

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
                            ComparisonType::String | ComparisonType::List => unreachable!(),
                        };

                        build
                            .build_bit_cast(vec_res, ctx.bool_type(), "sing_vec_to_bool")
                            .unwrap()
                    }
                };

                Ok(res.as_basic_value_enum())
            }
            KernelExpression::And(lhs, rhs) => {
                let orig_block = build.get_insert_block().unwrap();
                let func = orig_block.get_parent().unwrap();
                declare_blocks!(ctx, func, rhs_and, finish_and);

                let lhs = lhs.compile(compilation)?.into_int_value();
                build
                    .build_conditional_branch(lhs, rhs_and, finish_and)
                    .unwrap();

                build.position_at_end(rhs_and);
                let rhs = rhs.compile(compilation)?.into_int_value();
                build.build_unconditional_branch(finish_and).unwrap();

                build.position_at_end(finish_and);
                let phi = build.build_phi(ctx.bool_type(), "and_result").unwrap();
                phi.add_incoming(&[(&ctx.bool_type().const_zero(), orig_block), (&rhs, rhs_and)]);
                Ok(phi.as_basic_value())
            }
            KernelExpression::Or(lhs, rhs) => {
                let orig_block = build.get_insert_block().unwrap();
                let func = orig_block.get_parent().unwrap();
                declare_blocks!(ctx, func, rhs_or, finish_or);

                let lhs = lhs.compile(compilation)?.into_int_value();
                build
                    .build_conditional_branch(lhs, finish_or, rhs_or)
                    .unwrap();

                build.position_at_end(rhs_or);
                let rhs = rhs.compile(compilation)?.into_int_value();
                build.build_unconditional_branch(finish_or).unwrap();

                build.position_at_end(finish_or);
                let phi = build.build_phi(ctx.bool_type(), "or_result").unwrap();
                phi.add_incoming(&[
                    (&ctx.bool_type().const_all_ones(), orig_block),
                    (&rhs, rhs_or),
                ]);
                Ok(phi.as_basic_value())
            }
            KernelExpression::Not(e) => {
                let e = e.compile(compilation)?.into_int_value();
                Ok(build.build_not(e, "not").unwrap().into())
            }
            KernelExpression::Select { cond, v1, v2 } => {
                if cond.get_type() != DataType::Boolean {
                    return Err(DSLError::BooleanExpected(format!(
                        "first parameter to select should be a boolean, found {:?}",
                        cond.get_type()
                    )));
                }
                let cond_v = cond.compile(compilation)?.into_int_value();
                let a_v = v1.compile(compilation)?;
                let b_v = v2.compile(compilation)?;

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
                let idx = idx.compile(compilation)?.into_int_value();
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
                let to_convert = expr.compile(compilation)?;
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

                let lhs = lhs.compile(compilation).unwrap();
                let rhs = rhs.compile(compilation).unwrap();

                vec_or_scalar_op!(build, add, lhs, rhs)
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

                let lhs = lhs.compile(compilation).unwrap();
                let rhs = rhs.compile(compilation).unwrap();

                let signed = matches!(
                    PrimitiveSuperType::from(lhs_pt).list_type_into_inner(),
                    PrimitiveSuperType::Int
                );

                signed_vec_or_scalar_op!(build, div, signed, lhs, rhs)
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

                let lhs = lhs.compile(compilation).unwrap();
                let rhs = rhs.compile(compilation).unwrap();

                vec_or_scalar_op!(build, mul, lhs, rhs)
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

                let lhs = lhs.compile(compilation).unwrap();
                let rhs = rhs.compile(compilation).unwrap();

                let signed = matches!(
                    PrimitiveSuperType::from(lhs_pt).list_type_into_inner(),
                    PrimitiveSuperType::Int
                );
                signed_vec_or_scalar_op!(build, rem, signed, lhs, rhs)
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

                let lhs = lhs.compile(compilation)?;
                let rhs = rhs.compile(compilation)?;

                vec_or_scalar_op!(build, sub, lhs, rhs)
            }
            KernelExpression::Powi(inp, power) => {
                let f = Intrinsic::find("llvm.powi").unwrap();
                let func = match inp.get_type() {
                    DataType::Float16 | DataType::Float32 | DataType::Float64 => f.get_declaration(
                        compilation.llvm_mod,
                        &[
                            PrimitiveType::for_arrow_type(&inp.get_type())
                                .llvm_type(ctx)
                                .into(),
                            ctx.i32_type().into(),
                        ],
                    ),
                    DataType::FixedSizeList(t, s) => {
                        let t = PrimitiveType::for_arrow_type(t.data_type());
                        if !matches!(
                            t,
                            PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64
                        ) {
                            return Err(DSLError::TypeMismatch(format!(
                                "cannot pow-i vector of non-float type {}",
                                t
                            )));
                        }

                        let t = t.llvm_vec_type(ctx, s as u32).unwrap();
                        f.get_declaration(compilation.llvm_mod, &[t.into(), ctx.i32_type().into()])
                    }
                    _ => {
                        return Err(DSLError::TypeMismatch(format!(
                            "cannot pow-i non-float type {}",
                            inp.get_type()
                        )))
                    }
                }
                .unwrap();

                let inp = inp.compile(compilation)?;
                let call = build
                    .build_call(
                        func,
                        &[
                            inp.into(),
                            ctx.i32_type().const_int(*power as u64, true).into(),
                        ],
                        "powi",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left();

                call.as_instruction_value().unwrap().set_fast_math_flags(
                    LLVMFastMathAllowContract
                        | LLVMFastMathAllowReassoc
                        | LLVMFastMathAllowReciprocal
                        | LLVMFastMathApproxFunc,
                );

                Ok(call)
            }
            KernelExpression::Sqrt(expr) => {
                let f = Intrinsic::find("llvm.sqrt").unwrap();
                let func = match expr.get_type() {
                    DataType::Float16 => {
                        f.get_declaration(compilation.llvm_mod, &[ctx.f16_type().into()])
                    }
                    DataType::Float32 => {
                        f.get_declaration(compilation.llvm_mod, &[ctx.f32_type().into()])
                    }
                    DataType::Float64 => {
                        f.get_declaration(compilation.llvm_mod, &[ctx.f64_type().into()])
                    }
                    _ => {
                        return Err(DSLError::TypeMismatch(
                            "cannot sqrt non-float types".to_string(),
                        ))
                    }
                }
                .unwrap();

                let inp = expr.compile(compilation)?;
                let call = build
                    .build_call(func, &[inp.into()], "sqrt")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left();
                call.as_instruction_value().unwrap().set_fast_math_flags(
                    LLVMFastMathAllowContract
                        | LLVMFastMathAllowReassoc
                        | LLVMFastMathAllowReciprocal
                        | LLVMFastMathApproxFunc,
                );
                Ok(call)
            }
            KernelExpression::VectorSum(expr) => {
                let child_type = PrimitiveType::for_arrow_type(&expr.get_type());
                if let PrimitiveType::List(t, _s) = child_type {
                    let input = expr.compile(compilation)?.into_vector_value();
                    let ptype: PrimitiveType = t.into();
                    match PrimitiveSuperType::from(ptype) {
                        PrimitiveSuperType::Int | PrimitiveSuperType::UInt => {
                            let reducer = Intrinsic::find("llvm.vector.reduce.add").unwrap();
                            let reducer = reducer
                                .get_declaration(compilation.llvm_mod, &[input.get_type().into()])
                                .unwrap();

                            Ok(build
                                .build_call(reducer, &[input.into()], "vec_sum")
                                .unwrap()
                                .try_as_basic_value()
                                .unwrap_left())
                        }
                        PrimitiveSuperType::Float => {
                            let reducer = Intrinsic::find("llvm.vector.reduce.fadd").unwrap();
                            let reducer = reducer
                                .get_declaration(compilation.llvm_mod, &[input.get_type().into()])
                                .unwrap();

                            let call = build
                                .build_call(
                                    reducer,
                                    &[ptype.llvm_type(ctx).const_zero().into(), input.into()],
                                    "vec_sum",
                                )
                                .unwrap()
                                .try_as_basic_value()
                                .unwrap_left();

                            call.as_instruction_value().unwrap().set_fast_math_flags(
                                LLVMFastMathAllowContract
                                    | LLVMFastMathAllowReassoc
                                    | LLVMFastMathAllowReciprocal
                                    | LLVMFastMathApproxFunc,
                            );

                            Ok(call)
                        }
                        PrimitiveSuperType::String => {
                            return Err(DSLError::TypeMismatch(
                                "cannot sum string vectors".to_string(),
                            ))
                        }
                        PrimitiveSuperType::List(_, _) => unreachable!(),
                    }
                } else {
                    Err(DSLError::TypeMismatch(format!(
                        "cannot sum non-list type: {}",
                        child_type
                    )))
                }
            }
            KernelExpression::Splat(expr, size) => {
                let c_type = expr.get_type();
                let p_type = PrimitiveType::for_arrow_type(&c_type)
                    .llvm_vec_type(ctx, *size as u32)
                    .ok_or_else(|| {
                        DSLError::TypeMismatch(format!(
                            "cannot turn type {} into a vector for splat",
                            c_type
                        ))
                    })?;
                let val = expr.compile(compilation)?;
                let singleton = build
                    .build_insert_element(
                        p_type.const_zero(),
                        val,
                        ctx.i32_type().const_zero(),
                        "singleton",
                    )
                    .unwrap();
                Ok(build
                    .build_shuffle_vector(
                        singleton,
                        p_type.get_poison(),
                        p_type.const_zero(),
                        "splatted",
                    )
                    .unwrap()
                    .into())
            }
            KernelExpression::StartsWith(haystack, needle) => {
                if !haystack.is_string() || !needle.is_string() {
                    Err(DSLError::TypeMismatch(
                        "StartsWith only takes string types".to_string(),
                    ))?
                }

                let haystack = haystack.compile(compilation).unwrap();
                let needle = needle.compile(compilation).unwrap();

                let func = add_str_startswith(ctx, llvm_mod);
                Ok(build
                    .build_call(func, &[haystack.into(), needle.into()], "starts_with")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left())
            }
            KernelExpression::EndsWith(haystack, needle) => {
                if !haystack.is_string() || !needle.is_string() {
                    Err(DSLError::TypeMismatch(
                        "EndsWith only takes string types".to_string(),
                    ))?
                }

                let haystack = haystack.compile(compilation).unwrap();
                let needle = needle.compile(compilation).unwrap();

                let func = add_str_endswith(ctx, llvm_mod);
                Ok(build
                    .build_call(func, &[haystack.into(), needle.into()], "ends_with")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left())
            }
            KernelExpression::StrLen(expr) => {
                if !expr.is_string() {
                    Err(DSLError::TypeMismatch(
                        "StrLen takes string type".to_string(),
                    ))?
                }
                let expr = expr.compile(compilation).unwrap().into_struct_value();
                let start_ptr = expr.get_field_at_index(0).unwrap().into_pointer_value();
                let end_ptr = expr.get_field_at_index(1).unwrap().into_pointer_value();
                Ok(pointer_diff!(ctx, build, start_ptr, end_ptr).as_basic_value_enum())
            }
            KernelExpression::Substring {
                expr,
                start,
                len,
                check,
            } => {
                if !expr.is_string() {
                    Err(DSLError::TypeMismatch(
                        "Substring's first parameter must be of string type".to_string(),
                    ))?
                }
                let expr = expr.compile(compilation).unwrap().into_struct_value();
                let start = start.compile(compilation).unwrap().into_int_value();
                let len = len.compile(compilation).unwrap().into_int_value();

                let start_ptr = expr.get_field_at_index(0).unwrap().into_pointer_value();
                let end_ptr = expr.get_field_at_index(1).unwrap().into_pointer_value();

                let mut computed_start = increment_pointer!(ctx, build, start_ptr, 1, start);
                let mut computed_end = increment_pointer!(ctx, build, computed_start, 1, len);

                if *check {
                    let orig_len = pointer_diff!(ctx, build, start_ptr, end_ptr);
                    let last_in_substr = build.build_int_add(start, len, "last_in_substr").unwrap();
                    let is_oob = build
                        .build_int_compare(IntPredicate::UGT, last_in_substr, orig_len, "is_oob")
                        .unwrap();
                    computed_start = build
                        .build_select(is_oob, end_ptr, computed_start, "computed_start")
                        .unwrap()
                        .into_pointer_value();
                    computed_end = build
                        .build_select(is_oob, end_ptr, computed_end, "computed_end")
                        .unwrap()
                        .into_pointer_value();
                }

                let str_type = PrimitiveType::P64x2.llvm_type(ctx);
                let to_return = str_type.const_zero().into_struct_value();
                let to_return = build
                    .build_insert_value(to_return, computed_start, 0, "to_return")
                    .unwrap();
                let to_return = build
                    .build_insert_value(to_return, computed_end, 1, "to_return")
                    .unwrap();

                Ok(to_return.as_basic_value_enum())
            }
        }
    }
}
