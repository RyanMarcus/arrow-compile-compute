use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;

use crate::{
    dsl::{DSLKernel, KernelOutputType},
    ArrowKernelError, PrimitiveType,
};

use super::Kernel;

pub struct TakeKernel(DSLKernel);
unsafe impl Sync for TakeKernel {}
unsafe impl Send for TakeKernel {}

impl Kernel for TakeKernel {
    type Key = (DataType, DataType);

    type Input<'a>
        = (&'a dyn Array, &'a dyn Array)
    where
        Self: 'a;

    type Params = ();

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        self.0.call(&[&inp.0, &inp.1])
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, idx) = inp;
        if !PrimitiveType::for_arrow_type(idx.data_type()).is_int() {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "indexes for take must be integer, got {}",
                idx.data_type()
            )));
        }
        let out_type = if PrimitiveType::for_arrow_type(arr.data_type()) == PrimitiveType::P64x2 {
            KernelOutputType::String
        } else {
            KernelOutputType::Array
        };

        Ok(TakeKernel(
            DSLKernel::compile(&[arr, idx], |ctx| {
                let arr = ctx.get_input(0)?;
                let idx = ctx.get_input(1)?;
                idx.into_iter()
                    .map(|i| vec![arr.at(&i[0])])
                    .collect(out_type)
            })
            .map_err(|e| ArrowKernelError::DSLError(e))?,
        ))
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
    use arrow_array::{cast::AsArray, types::Int32Type, Int32Array, StringArray, UInt8Array};
    use itertools::Itertools;

    use crate::Kernel;

    use super::TakeKernel;

    #[test]
    fn test_take_i32() {
        let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let idxes = UInt8Array::from(vec![0, 0, 3, 3, 1]);

        let k = TakeKernel::compile(&(&data, &idxes), ()).unwrap();
        let res = k.call((&data, &idxes)).unwrap();

        assert_eq!(res.as_primitive::<Int32Type>().values(), &[1, 1, 4, 4, 2]);
    }

    #[test]
    fn test_take_str_4() {
        let data = StringArray::from(vec!["this", "is", "a", "test"]);
        let idxes = UInt8Array::from(vec![0, 0, 1, 3]);
        let k = TakeKernel::compile(&(&data, &idxes), ()).unwrap();
        let res = k.call((&data, &idxes)).unwrap();
        let res = res
            .as_string::<i32>()
            .iter()
            .map(|x| x.unwrap())
            .collect_vec();
        assert_eq!(&res, &["this", "this", "is", "test"]);
    }

    #[test]
    fn test_take_str_empty() {
        let data = StringArray::from(vec![""]);
        let idxes = UInt8Array::from(vec![0]);
        let k = TakeKernel::compile(&(&data, &idxes), ()).unwrap();
        let res = k.call((&data, &idxes)).unwrap();
        let res = res
            .as_string::<i32>()
            .iter()
            .map(|x| x.unwrap())
            .collect_vec();
        assert_eq!(&res, &[""]);
    }

    #[test]
    fn test_take_str_none() {
        let data = StringArray::from(vec!["test"]);
        let idxes = UInt8Array::from(Vec::<u8>::new());
        let k = TakeKernel::compile(&(&data, &idxes), ()).unwrap();
        let res = k.call((&data, &idxes)).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn test_take_str_2nd() {
        let data = StringArray::from(vec!["", "x"]);
        let idxes = UInt8Array::from(vec![1]);
        let k = TakeKernel::compile(&(&data, &idxes), ()).unwrap();
        let res = k.call((&data, &idxes)).unwrap();
        let res = res
            .as_string::<i32>()
            .iter()
            .map(|x| x.unwrap())
            .collect_vec();
        assert_eq!(&res, &["x"]);
    }
}
