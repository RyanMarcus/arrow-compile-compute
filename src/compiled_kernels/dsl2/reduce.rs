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
    Sum,
    Product,
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
            DSLReductionType::Sum | DSLReductionType::Product => ctx
                .struct_type(
                    &[
                        ctx.bool_type().into(),            // is init (saw >= 1 value)
                        iter_type.llvm_type(ctx).unwrap(), // running accumulator
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
            // {is_init = false, accumulator = additive identity 0}
            DSLReductionType::Sum => self.accum_type(ctx, iter_type).const_zero(),
            // {is_init = false, accumulator = multiplicative identity 1}
            DSLReductionType::Product => {
                let struct_ty = self.accum_type(ctx, iter_type).into_struct_type();
                let is_init = ctx.bool_type().const_zero().as_basic_value_enum();
                let one = match iter_type {
                    DSLType::Primitive(pt) if pt.is_float() => iter_type
                        .llvm_type(ctx)
                        .unwrap()
                        .into_float_type()
                        .const_float(1.0)
                        .as_basic_value_enum(),
                    _ => iter_type
                        .llvm_type(ctx)
                        .unwrap()
                        .into_int_type()
                        .const_int(1, false)
                        .as_basic_value_enum(),
                };
                struct_ty
                    .const_named_struct(&[is_init, one])
                    .as_basic_value_enum()
            }
        }
    }

    pub(crate) fn update<'ctx, 'a>(
        &self,
        ctx: &mut DSLCompilationContext<'ctx, 'a>,
        ty: &DSLType,
        accum: BasicValueEnum<'ctx>,
        next: BasicValueEnum<'ctx>,
        include: IntValue<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, ArrowKernelError> {
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
                    DSLReductionType::And
                    | DSLReductionType::Or
                    | DSLReductionType::Sum
                    | DSLReductionType::Product => unreachable!(),
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
            DSLReductionType::Sum | DSLReductionType::Product => {
                let accum = accum.into_struct_value();
                let is_init = ctx
                    .b
                    .build_extract_value(accum, 0, "is_init")
                    .unwrap()
                    .into_int_value();
                let val = ctx.b.build_extract_value(accum, 1, "acc_val").unwrap();

                let is_float = matches!(ty, DSLType::Primitive(pt) if pt.is_float());
                let combined = match (self, is_float) {
                    (DSLReductionType::Sum, false) => ctx
                        .b
                        .build_int_add(val.into_int_value(), next.into_int_value(), "sum")
                        .unwrap()
                        .as_basic_value_enum(),
                    (DSLReductionType::Sum, true) => ctx
                        .b
                        .build_float_add(val.into_float_value(), next.into_float_value(), "sum")
                        .unwrap()
                        .as_basic_value_enum(),
                    (DSLReductionType::Product, false) => ctx
                        .b
                        .build_int_mul(val.into_int_value(), next.into_int_value(), "product")
                        .unwrap()
                        .as_basic_value_enum(),
                    (DSLReductionType::Product, true) => ctx
                        .b
                        .build_float_mul(val.into_float_value(), next.into_float_value(), "product")
                        .unwrap()
                        .as_basic_value_enum(),
                    _ => unreachable!(),
                };

                // only fold in the new value when the row is included; is_init
                // tracks whether any value was seen so an empty/all-null reduction
                // can report EmptyReduction in output_value.
                let new_val = ctx
                    .b
                    .build_select(include, combined, val, "new_val")
                    .unwrap();
                let new_is_init = ctx.b.build_or(is_init, include, "new_is_init").unwrap();
                let accum = ctx
                    .b
                    .build_insert_value(accum, new_is_init, 0, "is_init")
                    .unwrap();
                let accum = ctx
                    .b
                    .build_insert_value(accum, new_val, 1, "acc_val")
                    .unwrap();
                Ok(accum.as_basic_value_enum())
            }
        }
    }

    pub(crate) fn output_value<'ctx, 'a>(
        &self,
        ctx: &mut DSLCompilationContext<'ctx, 'a>,
        accum: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, ArrowKernelError> {
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
                    DSLReductionType::And
                    | DSLReductionType::Or
                    | DSLReductionType::Sum
                    | DSLReductionType::Product => unreachable!(),
                };
                Ok(ctx
                    .b
                    .build_extract_value(accum, field, "reduce_output")
                    .unwrap())
            }
            DSLReductionType::Sum | DSLReductionType::Product => {
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
                Ok(ctx
                    .b
                    .build_extract_value(accum, 1, "reduce_output")
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
            | DSLReductionType::ArgMax
            | DSLReductionType::Sum
            | DSLReductionType::Product => {
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
            DSLReductionType::Min
            | DSLReductionType::Max
            | DSLReductionType::Sum
            | DSLReductionType::Product => expr.get_type(),
            DSLReductionType::ArgMin | DSLReductionType::ArgMax => {
                DSLType::Primitive(PrimitiveType::U64)
            }
        }
    }
}
