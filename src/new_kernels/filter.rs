use arrow_array::{Array, ArrayRef, BooleanArray};
use arrow_schema::DataType;

use crate::{
    dsl::{DSLKernel, KernelOutputType},
    ArrowKernelError, PrimitiveType,
};

use crate::new_kernels::Kernel;

pub struct FilterKernel(DSLKernel);
unsafe impl Sync for FilterKernel {}
unsafe impl Send for FilterKernel {}

impl Kernel for FilterKernel {
    type Key = (DataType, DataType);

    type Input<'a>
        = (&'a dyn Array, &'a BooleanArray)
    where
        Self: 'a;

    type Params = ();

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        self.0.call(&[&inp.0, &inp.1])
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, filt) = inp;

        let out_type = if PrimitiveType::for_arrow_type(arr.data_type()) == PrimitiveType::P64x2 {
            KernelOutputType::String
        } else {
            KernelOutputType::Array
        };

        if let DataType::RunEndEncoded(_, _) = inp.0.data_type() {
            Ok(FilterKernel(
                DSLKernel::compile(&[arr, filt], |ctx| {
                    let arr = ctx.get_input(0)?;
                    let filt = ctx.get_input(1)?;
                    ctx.iter_over(vec![arr, filt])
                        .filter(|i| i[1].eq(&0.into()).not())
                        .map(|i| vec![i[0].clone()])
                        .collect(out_type)
                })
                .map_err(ArrowKernelError::DSLError)?,
            ))
        } else {
            Ok(FilterKernel(
                DSLKernel::compile(&[arr, filt], |ctx| {
                    let arr = ctx.get_input(0)?;
                    let filt = ctx.get_input(1)?.into_set_bits()?;
                    ctx.iter_over(vec![filt])
                        .map(|i| vec![arr.at(&i[0])])
                        .collect(out_type)
                })
                .map_err(ArrowKernelError::DSLError)?,
            ))
        }
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((i.0.data_type().clone(), i.1.data_type().clone()))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{cast::AsArray, types::Int32Type, BooleanArray, Int32Array, RunArray};
    use arrow_schema::DataType;

    use crate::{dictionary_data_type, new_kernels::Kernel};

    use super::FilterKernel;

    #[test]
    fn test_filter_i32() {
        let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let filt = BooleanArray::from(vec![true, true, false, false, true, true]);

        let k = FilterKernel::compile(&(&data, &filt), ()).unwrap();
        let res = k.call((&data, &filt)).unwrap();

        assert_eq!(res.as_primitive::<Int32Type>().values(), &[1, 2, 5, 6]);
    }

    #[test]
    fn test_filter_dict() {
        let data = Int32Array::from(vec![1, 1, 2, 2, 2, 3]);
        let data = arrow_cast::cast(
            &data,
            &dictionary_data_type(DataType::Int8, DataType::Int32),
        )
        .unwrap();
        let filt = BooleanArray::from(vec![true, true, false, false, true, true]);

        let k = FilterKernel::compile(&(&data, &filt), ()).unwrap();
        let res = k.call((&data, &filt)).unwrap();

        assert_eq!(res.as_primitive::<Int32Type>().values(), &[1, 1, 2, 3]);
    }

    #[test]
    fn test_filter_ree() {
        let data = Int32Array::from(vec![1, 2, 3]);
        let ends = Int32Array::from(vec![2, 4, 6]);
        let data = RunArray::<Int32Type>::try_new(&ends, &data).unwrap();

        let filt = BooleanArray::from(vec![true, true, false, false, true, true]);

        let k = FilterKernel::compile(&(&data, &filt), ()).unwrap();
        let res = k.call((&data, &filt)).unwrap();

        assert_eq!(res.as_primitive::<Int32Type>().values(), &[1, 1, 3, 3]);
    }
}
