use arrow_array::{ArrayRef, Datum};
use arrow_schema::DataType;
use strum_macros::EnumIter;

use crate::{
    compiled_kernels::dsl::{DSLKernel, KernelOutputType},
    ArrowKernelError, Kernel, PrimitiveType,
};

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, EnumIter)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[cfg(test)]
impl BinOp {
    pub fn arrow_compute(&self, arr1: &dyn Datum, arr2: &dyn Datum) -> ArrayRef {
        match self {
            BinOp::Add => arrow_arith::numeric::add(arr1, arr2).unwrap(),
            BinOp::Sub => arrow_arith::numeric::sub_wrapping(arr1, arr2).unwrap(),
            BinOp::Mul => arrow_arith::numeric::mul_wrapping(arr1, arr2).unwrap(),
            BinOp::Div => arrow_arith::numeric::div(arr1, arr2).unwrap(),
            BinOp::Rem => arrow_arith::numeric::rem(arr1, arr2).unwrap(),
        }
    }
}

pub struct BinOpKernel {
    k: DSLKernel,
}
unsafe impl Sync for BinOpKernel {}
unsafe impl Send for BinOpKernel {}

impl Kernel for BinOpKernel {
    type Key = (DataType, bool, DataType, bool, BinOp);

    type Input<'a>
        = (&'a dyn Datum, &'a dyn Datum)
    where
        Self: 'a;

    type Params = BinOp;

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        self.k.call(&[inp.0, inp.1])
    }

    fn compile(arrs: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr1, arr2) = arrs;
        let pt1 = PrimitiveType::for_arrow_type(arr1.get().0.data_type());
        let pt2 = PrimitiveType::for_arrow_type(arr2.get().0.data_type());
        let res =
            PrimitiveType::dominant(pt1, pt2).ok_or(ArrowKernelError::TypeMismatch(pt1, pt2))?;

        let out_type = KernelOutputType::for_data_type(&res.as_arrow_type())
            .map_err(ArrowKernelError::DSLError)?;

        Ok(BinOpKernel {
            k: DSLKernel::compile(&[*arr1, *arr2], |ctx| {
                let arr1 = ctx.get_input(0)?;
                let arr2 = ctx.get_input(1)?;
                ctx.iter_over(vec![arr1, arr2])
                    .map(|i| vec![i[0].convert(res), i[1].convert(res)])
                    .map(|i| match params {
                        BinOp::Add => vec![i[0].add(&i[1])],
                        BinOp::Sub => vec![i[0].sub(&i[1])],
                        BinOp::Mul => vec![i[0].mul(&i[1])],
                        BinOp::Div => vec![i[0].div(&i[1])],
                        BinOp::Rem => vec![i[0].rem(&i[1])],
                    })
                    .collect(out_type)
            })
            .map_err(ArrowKernelError::DSLError)?,
        })
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
        compiled_kernels::arith::{BinOp, BinOpKernel},
        Kernel,
    };

    #[test]
    fn test_arith_i32() {
        let arr1 = Int32Array::from(vec![1, 0, 2, 4, -10, 20, 900000, 1000000]);
        let arr2 = Int32Array::from(vec![2, 1, 3, -5, -5, 10, 1000000, 1000000]);
        for op in BinOp::iter() {
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
        for op in BinOp::iter() {
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
        for op in BinOp::iter() {
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
        for op in BinOp::iter() {
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
        for op in BinOp::iter() {
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
        for op in BinOp::iter() {
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
        for op in BinOp::iter() {
            let k = BinOpKernel::compile(&(&arr1, &arr2), op).unwrap();
            let res = k.call((&arr1, &arr2)).unwrap();
            let res = res.as_primitive::<UInt32Type>();

            let arrow_res = op.arrow_compute(&arr1, &arr2);
            let arrow_res = arrow_res.as_primitive::<UInt32Type>();
            assert_eq!(res, arrow_res, "failed for op {:?}", op);
        }
    }
}
