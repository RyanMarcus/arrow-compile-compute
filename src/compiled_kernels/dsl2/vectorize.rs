use std::collections::HashMap;

use crate::{
    compiled_kernels::dsl2::{
        compiler::DSLCompilationContext, DSLExpr, DSLForEach, DSLStmt, DSLType, DSLValue,
    },
    PrimitiveType,
};

pub fn vectorize_for_each(ctx: &DSLCompilationContext, f: &DSLForEach) -> Option<DSLForEach> {
    // Scalar iterators are infinite. Vectorizing a loop made entirely of scalar
    // iterators would emit a full block even though the logical output length
    // is still 1, which corrupts output sizing.
    if f.iterators.iter().all(|iv| iv.ty.is_infinite()) {
        return None;
    }

    // see if every iterator has a next block function
    for iv in f.iterators.iter() {
        ctx.next_block_funcs.get(&iv.name)?;
    }

    // see if all loop vars have a block type
    let mut loop_var_types = HashMap::new();
    let mut loop_vars = Vec::new();
    for lv in f.loop_vars.iter() {
        let new_ty = lv.ty.block_type()?;
        loop_var_types.insert(lv.name, new_ty.clone());
        loop_vars.push(DSLValue {
            name: lv.name,
            ty: new_ty,
        });
    }

    // see if all statments and expressions can be vectorized
    let mut body = Vec::new();
    for stmt in f.body.iter() {
        match stmt {
            DSLStmt::Emit { index, value } => {
                // require static index
                index.as_u32()?;
                let expr = value.try_vectorize(&loop_var_types)?;
                body.push(DSLStmt::EmitBlock {
                    index: index.clone(),
                    value: expr,
                });
            }
            _ => return None,
        }
    }

    Some(DSLForEach {
        loop_vars,
        iterators: f.iterators.clone(),
        body,
    })
}

impl DSLExpr {
    fn try_vectorize(&self, lvm: &HashMap<usize, DSLType>) -> Option<DSLExpr> {
        match self {
            DSLExpr::Value(v) => match v.ty {
                DSLType::Primitive(pt) => match pt {
                    PrimitiveType::List(_, _) => Some(self.clone()),
                    _ => {
                        let ty = lvm.get(&v.name)?.clone();
                        Some(DSLExpr::Value(DSLValue { name: v.name, ty }))
                    }
                },
                DSLType::ConstScalar(..) => Some(DSLExpr::Splat(Box::new(self.clone()), 64)),
                _ => None,
            },

            DSLExpr::BitNot(v) => Some(DSLExpr::BitNot(Box::new(v.try_vectorize(lvm)?))),

            DSLExpr::BitwiseBinOp(op, lhs, rhs) => Some(DSLExpr::BitwiseBinOp(
                *op,
                Box::new(lhs.try_vectorize(lvm)?),
                Box::new(rhs.try_vectorize(lvm)?),
            )),

            DSLExpr::ArithBinOp(op, lhs, rhs) => Some(DSLExpr::ArithBinOp(
                *op,
                Box::new(lhs.try_vectorize(lvm)?),
                Box::new(rhs.try_vectorize(lvm)?),
            )),
            DSLExpr::Compare(op, lhs, rhs) => Some(DSLExpr::Compare(
                *op,
                Box::new(lhs.try_vectorize(lvm)?),
                Box::new(rhs.try_vectorize(lvm)?),
            )),
            DSLExpr::Cast(expr, pt) => {
                let vec_expr = expr.try_vectorize(lvm)?;
                Some(DSLExpr::Cast(Box::new(vec_expr), pt.as_list(64)?))
            }
            DSLExpr::BitCast(expr, pt) => {
                let vec_expr = expr.try_vectorize(lvm)?;
                Some(DSLExpr::BitCast(Box::new(vec_expr), pt.as_list(64)?))
            }
            _ => None,
        }
    }
}
