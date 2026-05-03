use inkwell::{
    context::Context,
    types::{BasicType, BasicTypeEnum},
    values::{BasicValue, BasicValueEnum, IntValue},
};

use crate::PrimitiveType;
use crate::{
    compiled_kernels::dsl2::{
        compiler::{compile_compare_values, DSLCompilationContext, KernelReturnCode},
        DSLComparison, DSLExpr, DSLType, DSLValue,
    },
    ArrowKernelError,
};

#[derive(Debug)]
pub struct DSLReduce {
    /// variables inside the reduction
    pub(crate) loop_vars: Vec<DSLValue>,

    /// values to iterate over
    pub(crate) iterators: Vec<DSLValue>,

    /// body of the reduction
    pub(crate) body: DSLExpr,

    /// whether the current row should be included in the reduction
    pub(crate) include: Option<DSLExpr>,

    /// the bound local holding the final reduction result
    pub(crate) result: DSLValue,

    /// type of reduction
    pub(crate) reduction_type: DSLReductionType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DSLReductionType {
    And,
    Or,
    Min,
    ArgMin,
    Max,
    ArgMax,
}

impl DSLReductionType {
    pub(crate) fn accum_type<'a>(
        &self,
        ctx: &'a Context,
        iter_type: &DSLType,
    ) -> BasicTypeEnum<'a> {
        match self {
            DSLReductionType::And | DSLReductionType::Or => ctx.bool_type().as_basic_type_enum(),
            DSLReductionType::Min
            | DSLReductionType::ArgMin
            | DSLReductionType::Max
            | DSLReductionType::ArgMax => ctx
                .struct_type(
                    &[
                        ctx.bool_type().into(),            // is init
                        ctx.i64_type().into(),             // curr idx
                        ctx.i64_type().into(),             // best idx
                        iter_type.llvm_type(ctx).unwrap(), // best value
                    ],
                    false,
                )
                .as_basic_type_enum(),
        }
    }

    pub(crate) fn initial_value<'a>(
        &self,
        ctx: &'a Context,
        iter_type: &DSLType,
    ) -> BasicValueEnum<'a> {
        match self {
            DSLReductionType::And => ctx.bool_type().const_all_ones().as_basic_value_enum(),
            DSLReductionType::Or => ctx.bool_type().const_zero().as_basic_value_enum(),
            DSLReductionType::Min
            | DSLReductionType::ArgMin
            | DSLReductionType::Max
            | DSLReductionType::ArgMax => self.accum_type(ctx, iter_type).const_zero(),
        }
    }

    pub(crate) fn update<'ctx, 'a>(
        &self,
        ctx: &mut DSLCompilationContext<'ctx, 'a>,
        ty: &DSLType,
        accum: BasicValueEnum<'a>,
        next: BasicValueEnum<'a>,
        include: IntValue<'a>,
    ) -> Result<BasicValueEnum<'a>, ArrowKernelError> {
        match self {
            DSLReductionType::And => {
                let updated = ctx
                    .b
                    .build_and(accum.into_int_value(), next.into_int_value(), "and")
                    .unwrap();
                Ok(ctx
                    .b
                    .build_select(include, updated, accum.into_int_value(), "and_include")
                    .unwrap()
                    .as_basic_value_enum())
            }
            DSLReductionType::Or => {
                let updated = ctx
                    .b
                    .build_or(accum.into_int_value(), next.into_int_value(), "or")
                    .unwrap();
                Ok(ctx
                    .b
                    .build_select(include, updated, accum.into_int_value(), "or_include")
                    .unwrap()
                    .as_basic_value_enum())
            }
            DSLReductionType::Min
            | DSLReductionType::ArgMin
            | DSLReductionType::Max
            | DSLReductionType::ArgMax => {
                let accum = accum.into_struct_value();
                let is_init = ctx
                    .b
                    .build_extract_value(accum, 0, "is_init")
                    .unwrap()
                    .into_int_value();
                let curr_idx = ctx
                    .b
                    .build_extract_value(accum, 1, "curr_idx")
                    .unwrap()
                    .into_int_value();
                let best_idx = ctx
                    .b
                    .build_extract_value(accum, 2, "best_idx")
                    .unwrap()
                    .into_int_value();
                let val = ctx.b.build_extract_value(accum, 3, "val").unwrap();

                let cmp_op = match self {
                    DSLReductionType::Min | DSLReductionType::ArgMin => DSLComparison::Lt,
                    DSLReductionType::Max | DSLReductionType::ArgMax => DSLComparison::Gt,
                    DSLReductionType::And | DSLReductionType::Or => unreachable!(),
                };
                let is_better =
                    compile_compare_values(ctx, cmp_op, ty, next, val)?.into_int_value();
                let is_uninit = ctx.b.build_not(is_init, "is_uninit").unwrap();
                let should_update = ctx
                    .b
                    .build_or(is_uninit, is_better, "should_update")
                    .unwrap()
                    .as_basic_value_enum()
                    .into_int_value();
                let should_update = ctx
                    .b
                    .build_and(include, should_update, "should_update_included")
                    .unwrap();
                let new_is_init = ctx.b.build_or(is_init, include, "new_is_init").unwrap();
                let new_v = ctx
                    .b
                    .build_select(should_update, next, val, "new_v")
                    .unwrap();
                let new_best_idx = ctx
                    .b
                    .build_select(should_update, curr_idx, best_idx, "new_best_idx")
                    .unwrap();
                let accum = ctx
                    .b
                    .build_insert_value(accum, new_is_init, 0, "is_init")
                    .unwrap();
                let accum = ctx
                    .b
                    .build_insert_value(
                        accum,
                        ctx.b
                            .build_int_add(
                                curr_idx,
                                ctx.ctx.i64_type().const_int(1, false),
                                "new_idx",
                            )
                            .unwrap(),
                        1,
                        "curr_idx",
                    )
                    .unwrap();
                let accum = ctx
                    .b
                    .build_insert_value(accum, new_best_idx, 2, "curr_idx")
                    .unwrap();
                let accum = ctx
                    .b
                    .build_insert_value(accum, new_v, 3, "curr_idx")
                    .unwrap();

                Ok(accum.as_basic_value_enum())
            }
        }
    }

    pub(crate) fn output_value<'ctx, 'a>(
        &self,
        ctx: &mut DSLCompilationContext<'ctx, 'a>,
        accum: BasicValueEnum<'a>,
    ) -> Result<BasicValueEnum<'a>, ArrowKernelError> {
        match self {
            DSLReductionType::And | DSLReductionType::Or => Ok(accum),
            DSLReductionType::Min
            | DSLReductionType::ArgMin
            | DSLReductionType::Max
            | DSLReductionType::ArgMax => {
                let accum = accum.into_struct_value();
                let is_init = ctx
                    .b
                    .build_extract_value(accum, 0, "reduce_is_init")
                    .unwrap()
                    .into_int_value();

                let reduce_has_value = ctx.ctx.append_basic_block(*ctx.func, "reduce_has_value");
                let reduce_empty = ctx.ctx.append_basic_block(*ctx.func, "reduce_empty");
                ctx.b
                    .build_conditional_branch(is_init, reduce_has_value, reduce_empty)
                    .unwrap();

                ctx.b.position_at_end(reduce_empty);
                ctx.b
                    .build_return(Some(
                        &ctx.ctx
                            .i64_type()
                            .const_int(KernelReturnCode::EmptyReduction.into(), false),
                    ))
                    .unwrap();

                ctx.b.position_at_end(reduce_has_value);
                let field = match self {
                    DSLReductionType::Min | DSLReductionType::Max => 3,
                    DSLReductionType::ArgMin | DSLReductionType::ArgMax => 2,
                    DSLReductionType::And | DSLReductionType::Or => unreachable!(),
                };
                Ok(ctx
                    .b
                    .build_extract_value(accum, field, "reduce_output")
                    .unwrap())
            }
        }
    }
}

impl DSLReductionType {
    pub fn can_reduce_expr(&self, expr: &DSLExpr) -> bool {
        match self {
            DSLReductionType::And | DSLReductionType::Or => {
                matches!(expr.get_type(), DSLType::Boolean)
            }
            DSLReductionType::Min
            | DSLReductionType::ArgMin
            | DSLReductionType::Max
            | DSLReductionType::ArgMax => {
                matches!(
                    expr.get_type(),
                    DSLType::Primitive(pt) if !matches!(pt, PrimitiveType::List(..))
                )
            }
        }
    }

    pub fn out_type(&self, expr: &DSLExpr) -> DSLType {
        match self {
            DSLReductionType::And | DSLReductionType::Or => DSLType::Boolean,
            DSLReductionType::Min | DSLReductionType::Max => expr.get_type(),
            DSLReductionType::ArgMin | DSLReductionType::ArgMax => {
                DSLType::Primitive(PrimitiveType::U64)
            }
        }
    }
}
