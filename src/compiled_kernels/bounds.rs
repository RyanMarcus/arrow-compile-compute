use arrow_array::{cast::AsArray, Datum};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::dsl2::{
        compile, DSLArgument, DSLBitwiseBinOp, DSLComparison, DSLContext, DSLFunction,
        DSLReductionType, DSLStmt, DSLType, RunnableDSLFunction,
    },
    compiled_writers::WriterSpec,
    ArrowKernelError, Kernel,
};

pub struct BoundsKernel {
    k: RunnableDSLFunction,
}
unsafe impl Sync for BoundsKernel {}
unsafe impl Send for BoundsKernel {}

impl Kernel for BoundsKernel {
    type Key = (DataType, bool, DataType, bool, DataType, bool);

    type Input<'a>
        = (&'a dyn Datum, &'a dyn Datum, &'a dyn Datum)
    where
        Self: 'a;

    type Params = ();

    type Output = bool;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (x, lb, ub) = inp;
        let (x_arr, _) = x.get();

        if x_arr.is_empty() {
            return Ok(true);
        }

        let res = self.k.run(&[
            DSLArgument::Datum(x),
            DSLArgument::Datum(lb),
            DSLArgument::Datum(ub),
        ])?[0]
            .clone();

        Ok(res.as_boolean().value(0))
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (x, lb, ub) = inp;
        let w = WriterSpec::Boolean;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("bounds");
        let x_arg = func.add_arg(&mut ctx, DSLType::array_like(*x, "n"));
        let lb_arg = func.add_arg(&mut ctx, DSLType::array_like(*lb, "n"));
        let ub_arg = func.add_arg(&mut ctx, DSLType::array_like(*ub, "n"));
        func.add_ret(w, "<= n");

        let reduction = DSLStmt::reduce(
            &mut ctx,
            DSLReductionType::And,
            &[x_arg, lb_arg, ub_arg],
            |loop_vars| {
                let x = &loop_vars[0];
                let lb = &loop_vars[1];
                let ub = &loop_vars[2];

                let is_gte_lb = x.expr().cmp(&lb.expr(), DSLComparison::Gte)?;
                let is_lt_ub = x.expr().cmp(&ub.expr(), DSLComparison::Lt)?;

                is_gte_lb.bitwise(DSLBitwiseBinOp::And, is_lt_ub)
            },
        )?;

        func.add_body(reduction.stmt);
        func.add_body(DSLStmt::emit(0, reduction.value.expr())?);
        let func = compile(
            func,
            [
                DSLArgument::Datum(*x),
                DSLArgument::Datum(*lb),
                DSLArgument::Datum(*ub),
            ],
        )?;

        Ok(BoundsKernel { k: func })
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        let (x, is1) = i.0.get();
        let (lb, is2) = i.1.get();
        let (ub, is3) = i.2.get();
        Ok((
            x.data_type().clone(),
            is1,
            lb.data_type().clone(),
            is2,
            ub.data_type().clone(),
            is3,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, Scalar};
    use std::sync::Arc;

    #[test]
    fn test_bounds_kernel_array_inputs() {
        let x = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let lb = Arc::new(Int32Array::from(vec![1, 1, 1]));
        let ub = Arc::new(Int32Array::from(vec![4, 4, 4]));

        let kernel = BoundsKernel::compile(
            &(
                x.as_ref() as &dyn Datum,
                lb.as_ref() as &dyn Datum,
                ub.as_ref() as &dyn Datum,
            ),
            (),
        )
        .unwrap();

        let result = kernel
            .call((
                x.as_ref() as &dyn Datum,
                lb.as_ref() as &dyn Datum,
                ub.as_ref() as &dyn Datum,
            ))
            .unwrap();

        assert!(result);
    }

    #[test]
    fn test_bounds_kernel_scalar_bounds() {
        let x = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let lb = Scalar::new(Int32Array::from(vec![1]));
        let ub = Scalar::new(Int32Array::from(vec![4]));

        let kernel = BoundsKernel::compile(
            &(
                x.as_ref() as &dyn Datum,
                &lb as &dyn Datum,
                &ub as &dyn Datum,
            ),
            (),
        )
        .unwrap();

        let result = kernel
            .call((
                x.as_ref() as &dyn Datum,
                &lb as &dyn Datum,
                &ub as &dyn Datum,
            ))
            .unwrap();

        assert!(result);
    }

    #[test]
    fn test_bounds_kernel_out_of_bounds() {
        let x = Arc::new(Int32Array::from(vec![1, 2, 4]));
        let lb = Scalar::new(Int32Array::from(vec![1]));
        let ub = Scalar::new(Int32Array::from(vec![4]));

        let kernel = BoundsKernel::compile(
            &(
                x.as_ref() as &dyn Datum,
                &lb as &dyn Datum,
                &ub as &dyn Datum,
            ),
            (),
        )
        .unwrap();

        let result = kernel
            .call((
                x.as_ref() as &dyn Datum,
                &lb as &dyn Datum,
                &ub as &dyn Datum,
            ))
            .unwrap();

        assert!(!result);
    }

    #[test]
    fn test_bounds_kernel_empty_input() {
        let x = Arc::new(Int32Array::from(Vec::<i32>::new()));
        let lb = Scalar::new(Int32Array::from(vec![1]));
        let ub = Scalar::new(Int32Array::from(vec![4]));

        let kernel = BoundsKernel::compile(
            &(
                x.as_ref() as &dyn Datum,
                &lb as &dyn Datum,
                &ub as &dyn Datum,
            ),
            (),
        )
        .unwrap();

        let result = kernel
            .call((
                x.as_ref() as &dyn Datum,
                &lb as &dyn Datum,
                &ub as &dyn Datum,
            ))
            .unwrap();

        assert!(result);
    }
}
