use inkwell::{
    context::Context,
    types::{BasicType, BasicTypeEnum},
    values::{BasicValue, BasicValueEnum},
};

use crate::compiled_kernels::dsl2::{compiler::DSLCompilationContext, DSLExpr, DSLType, DSLValue};

#[derive(Debug)]
pub struct DSLReduce {
    /// variables inside the reduction
    pub(crate) loop_vars: Vec<DSLValue>,

    /// values to iterate over
    pub(crate) iterators: Vec<DSLValue>,

    /// body of the reduction
    pub(crate) body: DSLExpr,

    /// the bound local holding the final reduction result
    pub(crate) result: DSLValue,

    /// type of reduction
    pub(crate) reduction_type: DSLReductionType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DSLReductionType {
    And,
    Or,
}

impl DSLReductionType {
    pub(crate) fn accum_type<'a>(&self, ctx: &'a Context) -> BasicTypeEnum<'a> {
        match self {
            DSLReductionType::And | DSLReductionType::Or => ctx.bool_type().as_basic_type_enum(),
        }
    }

    pub(crate) fn initial_value<'a>(&self, ctx: &'a Context) -> BasicValueEnum<'a> {
        match self {
            DSLReductionType::And => ctx.bool_type().const_all_ones().as_basic_value_enum(),
            DSLReductionType::Or => ctx.bool_type().const_zero().as_basic_value_enum(),
        }
    }

    pub(crate) fn update<'ctx, 'a>(
        &self,
        ctx: &mut DSLCompilationContext<'ctx, 'a>,
        accum: BasicValueEnum<'a>,
        next: BasicValueEnum<'a>,
    ) -> BasicValueEnum<'a> {
        match self {
            DSLReductionType::And => ctx
                .b
                .build_and(accum.into_int_value(), next.into_int_value(), "and")
                .unwrap()
                .as_basic_value_enum(),
            DSLReductionType::Or => ctx
                .b
                .build_or(accum.into_int_value(), next.into_int_value(), "or")
                .unwrap()
                .as_basic_value_enum(),
        }
    }
}

impl DSLReductionType {
    pub fn can_reduce_expr(&self, expr: &DSLExpr) -> bool {
        match self {
            DSLReductionType::And | DSLReductionType::Or => {
                matches!(expr.get_type(), DSLType::Boolean)
            }
        }
    }

    pub fn out_type(&self) -> DSLType {
        match self {
            DSLReductionType::And | DSLReductionType::Or => DSLType::Boolean,
        }
    }
}
