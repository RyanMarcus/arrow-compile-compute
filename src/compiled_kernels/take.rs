use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;

use crate::compiled_kernels::dsl::{DSLKernel, KernelOutputType};
use crate::{logical_nulls, ArrowKernelError, PrimitiveType};

use crate::compiled_kernels::{replace_nulls, Kernel};

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
        let (arr, idx) = inp;
        if idx.nulls().is_some() {
            return Err(ArrowKernelError::UnsupportedArguments(
                "indexes for take must not be nullable".to_string(),
            ));
        }
        let mut res = self.0.call(&[&arr, &idx])?;

        if let Some(nulls) = logical_nulls(arr)? {
            let ba = BooleanArray::new(nulls.into_inner(), None);
            let nulls = crate::arrow_interface::select::take(&ba, idx)?;
            let nulls = NullBuffer::new(nulls.as_boolean().clone().into_parts().0);
            res = replace_nulls(res, Some(nulls));
        }

        Ok(res)
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, idx) = inp;
        if !PrimitiveType::for_arrow_type(idx.data_type()).is_int() {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "indexes for take must be integer, got {}",
                idx.data_type()
            )));
        }

        // TODO more robust output type selecction
        let out_type = if PrimitiveType::for_arrow_type(arr.data_type()) == PrimitiveType::P64x2 {
            KernelOutputType::String
        } else if arr.data_type() == &DataType::Boolean {
            KernelOutputType::Boolean
        } else {
            KernelOutputType::Array
        };

        Ok(TakeKernel(
            DSLKernel::compile(&[arr, idx], |ctx| {
                let arr = ctx.get_input(0)?;
                let idx = ctx.get_input(1)?;
                ctx.iter_over(vec![idx])
                    .map(|i| vec![arr.at(&i[0])])
                    .collect(out_type)
            })
            .map_err(ArrowKernelError::DSLError)?,
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
    use arrow_array::{
        cast::AsArray,
        types::{Int32Type, Int64Type},
        BooleanArray, Int32Array, Int64Array, RunArray, StringArray, UInt16Array, UInt32Array,
        UInt8Array,
    };
    use itertools::Itertools;

    use crate::compiled_kernels::Kernel;

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
    fn test_take_i32nullable() {
        let data = Int32Array::from(vec![Some(1), None, Some(3), Some(4), Some(5), Some(6)]);
        let idxes = UInt8Array::from(vec![0, 0, 3, 3, 1]);

        let k = TakeKernel::compile(&(&data, &idxes), ()).unwrap();
        let res = k.call((&data, &idxes)).unwrap();

        assert_eq!(
            res.as_primitive::<Int32Type>().iter().collect_vec(),
            &[Some(1), Some(1), Some(4), Some(4), None]
        );
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

    #[test]
    fn test_take_ree() {
        let values = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let rees = Int64Array::from(vec![10, 20, 30, 40, 50]);
        let data = RunArray::<Int64Type>::try_new(&rees, &values).unwrap();
        let idxes = UInt16Array::from(vec![3, 7, 15, 16, 23, 32, 41]);
        let k = TakeKernel::compile(&(&data, &idxes), ()).unwrap();
        let res = k.call((&data, &idxes)).unwrap();
        let res = res.as_primitive::<Int32Type>();
        assert_eq!(res.values(), &[1, 1, 2, 2, 3, 4, 5]);
    }

    #[test]
    fn test_take_bool() {
        let data = BooleanArray::from(vec![true, false, true, false, false, false]);
        let idxes = UInt32Array::from(vec![0, 2, 5]);
        let k = TakeKernel::compile(&(&data, &idxes), ()).unwrap();
        let res = k.call((&data, &idxes)).unwrap();
        let res = res.as_boolean();
        assert_eq!(res.values().iter().collect_vec(), &[true, true, false]);
    }
}
