use arrow_array::{ArrayRef, Datum};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        dsl2::{
            compile, DSLArgument, DSLArithBinOp, DSLContext, DSLFunction, DSLStmt, DSLType,
            DSLUnaryOp, RunnableDSLFunction, WriterSpec,
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
                let v1 = loop_vars[0].expr().primitive_cast(res)?;
                let v2 = loop_vars[1].expr().primitive_cast(res)?;
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

pub struct UnaryOpKernel(RunnableDSLFunction);
unsafe impl Sync for UnaryOpKernel {}
unsafe impl Send for UnaryOpKernel {}

impl Kernel for UnaryOpKernel {
    type Key = (DataType, bool, DSLUnaryOp);

    type Input<'a>
        = &'a dyn Datum
    where
        Self: 'a;

    type Params = DSLUnaryOp;

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let res = self
            .0
            .run(&[DSLArgument::Datum(inp)])
            .map(|x| x[0].clone())?;

        intersect_and_copy_nulls(&[inp], res)
    }

    fn compile(arr: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let pt = PrimitiveType::for_arrow_type(arr.get().0.data_type());

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("arith_unaryop");
        let arg = func.add_arg(&mut ctx, DSLType::array_like(*arr, "n"));
        func.add_ret(WriterSpec::Primitive(pt), "n");

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arg], |loop_vars| {
                let el = match params {
                    DSLUnaryOp::Neg => loop_vars[0].expr().neg()?,
                    DSLUnaryOp::Abs => loop_vars[0].expr().abs()?,
                };
                DSLStmt::emit(0, el)
            })
            .unwrap(),
        );
        let func = compile(func, [DSLArgument::Datum(*arr)]).unwrap();
        Ok(UnaryOpKernel(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        let (arr, is_scalar) = i.get();
        Ok((arr.data_type().clone(), is_scalar, *p))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        cast::AsArray,
        types::{Float32Type, Float64Type, Int32Type, UInt32Type},
        Datum, Float32Array, Float64Array, Int32Array, UInt32Array, UInt64Array,
    };
    use strum::IntoEnumIterator;

    use crate::{
        compiled_kernels::{
            arith::{BinOpKernel, UnaryOpKernel},
            dsl2::{DSLArithBinOp, DSLUnaryOp},
        },
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

    // These test `Neg` directly rather than iterating `DSLUnaryOp::iter()`:
    // not every unary op has an arrow oracle (`abs` has none, so its
    // `arrow_compute` arm is `unreachable!`), so a shared-oracle loop can't
    // cover all variants. `abs` is verified by the dedicated `test_abs_*` tests.
    #[test]
    fn test_neg_i32() {
        let arr = Int32Array::from(vec![1, 0, 2, -4, -10, 20, i32::MIN, i32::MAX]);
        let k = UnaryOpKernel::compile(&(&arr as &dyn Datum), DSLUnaryOp::Neg).unwrap();
        let res = k.call(&arr).unwrap();
        let res = res.as_primitive::<Int32Type>();

        let arrow_res = DSLUnaryOp::Neg.arrow_compute(&arr);
        let arrow_res = arrow_res.as_primitive::<Int32Type>();
        assert_eq!(res, arrow_res);
    }

    #[test]
    fn test_neg_f32() {
        let arr = Float32Array::from(vec![1.0, 0.0, -2.0, 4.0, -10.0, 20.0]);
        let k = UnaryOpKernel::compile(&(&arr as &dyn Datum), DSLUnaryOp::Neg).unwrap();
        let res = k.call(&arr).unwrap();
        let res = res.as_primitive::<Float32Type>();

        let arrow_res = DSLUnaryOp::Neg.arrow_compute(&arr);
        let arrow_res = arrow_res.as_primitive::<Float32Type>();
        assert_eq!(res, arrow_res);
    }

    #[test]
    fn test_neg_scalar_i32_nulls() {
        let arr = Int32Array::from(vec![Some(1), None, Some(-2), Some(4)]);
        let k = UnaryOpKernel::compile(&(&arr as &dyn Datum), DSLUnaryOp::Neg).unwrap();
        let res = k.call(&arr).unwrap();
        let res = res.as_primitive::<Int32Type>();

        let arrow_res = DSLUnaryOp::Neg.arrow_compute(&arr);
        let arrow_res = arrow_res.as_primitive::<Int32Type>();
        assert_eq!(res, arrow_res);
    }

    #[test]
    fn test_abs_i32() {
        // hard-coded edge inputs: MIN wraps to itself, MAX, 0, negatives
        let arr = Int32Array::from(vec![0, 1, -1, 4, -10, i32::MAX, i32::MIN, -(i32::MAX)]);
        let k = UnaryOpKernel::compile(&(&arr as &dyn Datum), DSLUnaryOp::Abs).unwrap();
        let res = k.call(&arr).unwrap();
        let res = res.as_primitive::<Int32Type>();

        let expected = Int32Array::from(
            (0..arr.len())
                .map(|i| arr.value(i).wrapping_abs())
                .collect::<Vec<_>>(),
        );
        assert_eq!(res, &expected);
    }

    #[test]
    fn test_abs_u32_identity() {
        // unsigned abs is the identity
        let arr = UInt32Array::from(vec![0, 1, 5, u32::MAX, 900000]);
        let k = UnaryOpKernel::compile(&(&arr as &dyn Datum), DSLUnaryOp::Abs).unwrap();
        let res = k.call(&arr).unwrap();
        let res = res.as_primitive::<UInt32Type>();

        let expected = UInt32Array::from(
            (0..arr.len()).map(|i| arr.value(i)).collect::<Vec<_>>(),
        );
        assert_eq!(res, &expected);
    }

    #[test]
    fn test_abs_f64() {
        // hard-coded float edge inputs: NaN, +/-inf, -0.0, 0.0, negatives
        let arr = Float64Array::from(vec![
            0.0,
            -0.0,
            1.0,
            -2.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
        ]);
        let k = UnaryOpKernel::compile(&(&arr as &dyn Datum), DSLUnaryOp::Abs).unwrap();
        let res = k.call(&arr).unwrap();
        let res = res.as_primitive::<Float64Type>();

        for i in 0..arr.len() {
            let expected = arr.value(i).abs();
            let got = res.value(i);
            if expected.is_nan() {
                assert!(got.is_nan(), "expected NaN at {i}, got {got}");
            } else {
                // compare bit patterns so -0.0 vs 0.0 is distinguished
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "mismatch at {i}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_abs_i32_nulls() {
        let arr = Int32Array::from(vec![Some(1), None, Some(-2), Some(i32::MIN)]);
        let k = UnaryOpKernel::compile(&(&arr as &dyn Datum), DSLUnaryOp::Abs).unwrap();
        let res = k.call(&arr).unwrap();
        let res = res.as_primitive::<Int32Type>();

        let expected = Int32Array::from(vec![Some(1), None, Some(2), Some(i32::MIN)]);
        assert_eq!(res, &expected);
    }

    #[test]
    fn test_abs_empty() {
        let arr = Int32Array::from(Vec::<i32>::new());
        let k = UnaryOpKernel::compile(&(&arr as &dyn Datum), DSLUnaryOp::Abs).unwrap();
        let res = k.call(&arr).unwrap();
        let res = res.as_primitive::<Int32Type>();
        assert_eq!(res.len(), 0);
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
    fn test_arith_u32_f64() {
        let arr1 = UInt32Array::from(vec![1, 0, 2]);
        let arr1_f = Float64Array::from(vec![1.0, 0.0, 2.0]);
        let arr2 = Float64Array::from(vec![2.0, 1.0, 3.0]);
        for op in DSLArithBinOp::iter() {
            let k = BinOpKernel::compile(&(&arr1, &arr2), op).unwrap();
            let res = k.call((&arr1, &arr2)).unwrap();
            let res = res.as_primitive::<Float64Type>();

            let arrow_res = op.arrow_compute(&arr1_f, &arr2);
            let arrow_res = arrow_res.as_primitive::<Float64Type>();
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
    fn test_arith_scalar_f64_u64_len32() {
        let lhs = Float64Array::new_scalar(1.0);
        let rhs = UInt64Array::from((1_u64..=32).collect::<Vec<_>>());
        let op = DSLArithBinOp::Div;
        let k = BinOpKernel::compile(&(&lhs, &rhs), op).unwrap();
        let res = k.call((&lhs, &rhs)).unwrap();
        let res = res.as_primitive::<Float64Type>();

        let rhs_f = Float64Array::from((1_u64..=32).map(|v| v as f64).collect::<Vec<_>>());
        let arrow_res = op.arrow_compute(&lhs, &rhs_f);
        let arrow_res = arrow_res.as_primitive::<Float64Type>();
        assert_eq!(res, arrow_res, "failed for op {:?}", op);
    }

    #[test]
    fn test_arith_scalar_scalar_u32() {
        let arr1 = UInt32Array::new_scalar(5);
        let arr2 = UInt32Array::new_scalar(10);
        for op in DSLArithBinOp::iter() {
            let k = BinOpKernel::compile(&(&arr1, &arr2), op).unwrap();
            assert!(
                !k.0.vectorized,
                "scalar/scalar kernels should stay scalar for op {:?}",
                op
            );
            let res = k.call((&arr1, &arr2)).unwrap();
            let res = res.as_primitive::<UInt32Type>();

            let arrow_res = op.arrow_compute(&arr1, &arr2);
            let arrow_res = arrow_res.as_primitive::<UInt32Type>();
            assert_eq!(res, arrow_res, "failed for op {:?}", op);
        }
    }
}
