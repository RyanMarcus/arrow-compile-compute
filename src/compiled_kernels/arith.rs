use arrow_array::{ArrayRef, Datum};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        dsl2::{
            compile, DSLArgument, DSLArithBinOp, DSLContext, DSLFunction, DSLStmt, DSLType,
            RunnableDSLFunction, WriterSpec,
        },
        null_utils::intersect_and_copy_nulls,
    },
    ArrowKernelError, Kernel, PrimitiveType,
};

pub struct BinOpKernel(RunnableDSLFunction);
unsafe impl Sync for BinOpKernel {}
unsafe impl Send for BinOpKernel {}

impl Kernel for BinOpKernel {
    type Key = (DataType, bool, DataType, bool, DSLArithBinOp);

    type Input<'a>
        = (&'a dyn Datum, &'a dyn Datum)
    where
        Self: 'a;

    type Params = DSLArithBinOp;

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let res = self
            .0
            .run(&[DSLArgument::Datum(inp.0), DSLArgument::Datum(inp.1)])
            .map(|x| x[0].clone())?;

        intersect_and_copy_nulls(&[inp.0, inp.1], res)
    }

    fn compile(arrs: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr1, arr2) = arrs;
        let pt1 = PrimitiveType::for_arrow_type(arr1.get().0.data_type());
        let pt2 = PrimitiveType::for_arrow_type(arr2.get().0.data_type());
        let res =
            PrimitiveType::dominant(pt1, pt2).ok_or(ArrowKernelError::TypeMismatch(pt1, pt2))?;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("arith_binop");
        let arg1 = func.add_arg(&mut ctx, DSLType::array_like(*arr1, "n"));
        let arg2 = func.add_arg(&mut ctx, DSLType::array_like(*arr2, "n"));
        func.add_ret(WriterSpec::Primitive(res), "n");

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arg1, arg2], |loop_vars| {
                let v1 = loop_vars[0].expr();
                let v2 = loop_vars[1].expr();
                let el = v1.arith(params, v2)?;
                DSLStmt::emit(0, el)
            })
            .unwrap(),
        );
        let func = compile(func, [DSLArgument::Datum(*arr1), DSLArgument::Datum(*arr2)]).unwrap();
        Ok(BinOpKernel(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        let (arr1, is_scalar1) = i.0.get();
        let (arr2, is_scalar2) = i.1.get();
        Ok((
            arr1.data_type().clone(),
            is_scalar1,
            arr2.data_type().clone(),
            is_scalar2,
            *p,
        ))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        cast::AsArray,
        types::{Float32Type, Int32Type, UInt32Type},
        Float32Array, Int32Array, UInt32Array,
    };
    use strum::IntoEnumIterator;

    use crate::{
        compiled_kernels::{arith::BinOpKernel, dsl2::DSLArithBinOp},
        Kernel,
    };

    #[test]
    fn test_arith_i32() {
        let arr1 = Int32Array::from(vec![1, 0, 2, 4, -10, 20, 900000, 1000000]);
        let arr2 = Int32Array::from(vec![2, 1, 3, -5, -5, 10, 1000000, 1000000]);
        for op in DSLArithBinOp::iter() {
            let k = BinOpKernel::compile(&(&arr1, &arr2), op).unwrap();
            let res = k.call((&arr1, &arr2)).unwrap();
            let res = res.as_primitive::<Int32Type>();

            let arrow_res = op.arrow_compute(&arr1, &arr2);
            let arrow_res = arrow_res.as_primitive::<Int32Type>();
            assert_eq!(res, arrow_res, "failed for op {:?}", op);
        }
    }

    #[test]
    fn test_arith_scalar_i32() {
        let arr1 = Int32Array::from(vec![1, 0, 2, 4, -10, 20, 900000, 1000000]);
        let arr2 = Int32Array::new_scalar(5);
        for op in DSLArithBinOp::iter() {
            let k = BinOpKernel::compile(&(&arr1, &arr2), op).unwrap();
            let res = k.call((&arr1, &arr2)).unwrap();
            let res = res.as_primitive::<Int32Type>();

            let arrow_res = op.arrow_compute(&arr1, &arr2);
            let arrow_res = arrow_res.as_primitive::<Int32Type>();
            assert_eq!(res, arrow_res, "failed for op {:?}", op);
        }
    }

    #[test]
    fn test_arith_scalar_i32_nulls() {
        let arr1 = Int32Array::from(vec![Some(1), None, Some(2), Some(4)]);
        let arr2 = Int32Array::new_scalar(5);
        for op in DSLArithBinOp::iter() {
            let k = BinOpKernel::compile(&(&arr1, &arr2), op).unwrap();
            let res = k.call((&arr1, &arr2)).unwrap();
            let res = res.as_primitive::<Int32Type>();

            let arrow_res = op.arrow_compute(&arr1, &arr2);
            let arrow_res = arrow_res.as_primitive::<Int32Type>();
            assert_eq!(res, arrow_res, "failed for op {:?}", op);
        }
    }

    #[test]
    fn test_arith_f32() {
        let arr1 = Float32Array::from(vec![1.0, 0.0, 2.0, 4.0, -10.0, 20.0, 900000.0, 1000000.0]);
        let arr2 = Float32Array::from(vec![2.0, 1.0, 3.0, -5.0, -5.0, 10.0, 1000000.0, 1000000.0]);
        for op in DSLArithBinOp::iter() {
            let k = BinOpKernel::compile(&(&arr1, &arr2), op).unwrap();
            let res = k.call((&arr1, &arr2)).unwrap();
            let res = res.as_primitive::<Float32Type>();

            let arrow_res = op.arrow_compute(&arr1, &arr2);
            let arrow_res = arrow_res.as_primitive::<Float32Type>();
            assert_eq!(res, arrow_res, "failed for op {:?}", op);
        }
    }

    #[test]
    fn test_arith_scalar_f32() {
        let arr1 = Float32Array::from(vec![1.0, 0.0, 2.0, 4.0, -10.0, 20.0, 900000.0, 1000000.0]);
        let arr2 = Float32Array::new_scalar(5.0);
        for op in DSLArithBinOp::iter() {
            let k = BinOpKernel::compile(&(&arr1, &arr2), op).unwrap();
            let res = k.call((&arr1, &arr2)).unwrap();
            let res = res.as_primitive::<Float32Type>();

            let arrow_res = op.arrow_compute(&arr1, &arr2);
            let arrow_res = arrow_res.as_primitive::<Float32Type>();
            assert_eq!(res, arrow_res, "failed for op {:?}", op);
        }
    }

    #[test]
    fn test_arith_u32() {
        let arr1 = UInt32Array::from(vec![1, 0, 2, 4, 10, 20, 900000, 1000000]);
        let arr2 = UInt32Array::from(vec![2, 1, 3, 5, 5, 10, 1000000, 1000000]);
        for op in DSLArithBinOp::iter() {
            let k = BinOpKernel::compile(&(&arr1, &arr2), op).unwrap();
            let res = k.call((&arr1, &arr2)).unwrap();
            let res = res.as_primitive::<UInt32Type>();

            let arrow_res = op.arrow_compute(&arr1, &arr2);
            let arrow_res = arrow_res.as_primitive::<UInt32Type>();
            assert_eq!(res, arrow_res, "failed for op {:?}", op);
        }
    }

    #[test]
    fn test_arith_scalar_u32() {
        let arr1 = UInt32Array::from(vec![1, 0, 2, 4, 10, 20, 900000, 1000000]);
        let arr2 = UInt32Array::new_scalar(5);
        for op in DSLArithBinOp::iter() {
            let k = BinOpKernel::compile(&(&arr1, &arr2), op).unwrap();
            let res = k.call((&arr1, &arr2)).unwrap();
            let res = res.as_primitive::<UInt32Type>();

            let arrow_res = op.arrow_compute(&arr1, &arr2);
            let arrow_res = arrow_res.as_primitive::<UInt32Type>();
            assert_eq!(res, arrow_res, "failed for op {:?}", op);
        }
    }

    #[test]
    fn test_arith_scalar_scalar_u32() {
        let arr1 = UInt32Array::new_scalar(5);
        let arr2 = UInt32Array::new_scalar(10);
        for op in DSLArithBinOp::iter() {
            let k = BinOpKernel::compile(&(&arr1, &arr2), op).unwrap();
            let res = k.call((&arr1, &arr2)).unwrap();
            let res = res.as_primitive::<UInt32Type>();

            let arrow_res = op.arrow_compute(&arr1, &arr2);
            let arrow_res = arrow_res.as_primitive::<UInt32Type>();
            assert_eq!(res, arrow_res, "failed for op {:?}", op);
        }
    }
}
