use arrow_array::{Array, ArrayRef, UInt64Array};
use arrow_schema::DataType;
use itertools::Itertools;

use crate::compiled_kernels::cast::coalesce_type;
use crate::compiled_kernels::dsl2::{
    self, dsl_args, DSLContext, DSLFunction, DSLStmt, DSLType, WriterSpec,
};
use crate::compiled_kernels::null_utils::{copy_selected_nulls, has_any_nulls};
use crate::{arrow_interface, iter, normalized_base_type, ArrowKernelError, PrimitiveType};

use crate::compiled_kernels::dsl2::RunnableDSLFunction;
use crate::compiled_kernels::Kernel;

pub struct TakeKernel(RunnableDSLFunction);
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

        if has_any_nulls(idx) {
            return Err(ArrowKernelError::UnsupportedArguments(
                "indexes for take must not be nullable".to_string(),
            ));
        }

        // bounds checking
        let in_bounds = arrow_interface::cmp::bounds(
            &idx,
            &UInt64Array::new_scalar(0),
            &UInt64Array::new_scalar(arr.len() as u64),
        )?;
        if !in_bounds {
            return Err(ArrowKernelError::OutOfBounds(arr.len()));
        }

        let mut res = self.0.run(&dsl_args!(arr, idx))?[0].clone();
        res = coalesce_type(res, &normalized_base_type(arr.data_type()))?;
        if has_any_nulls(arr) {
            let indices = iter::iter_nonnull_u64(idx)?
                .map(|x| x as usize)
                .collect_vec();
            res = copy_selected_nulls(arr, res, &indices)?;
        }

        Ok(res)
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, idx) = *inp;

        if !PrimitiveType::for_arrow_type(idx.data_type()).is_int() {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "indexes for take must be integer, got {}",
                idx.data_type()
            )));
        }

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("take");
        let arg_arr = func.add_arg(&mut ctx, DSLType::array_like(&arr, "n"));
        let arg_idx = func.add_arg(&mut ctx, DSLType::array_like(&idx, "m"));
        func.add_ret(WriterSpec::for_base_type_of_datum(&arr), "m");

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arg_idx], |loop_vars| {
                let idx = loop_vars[0].expr().primitive_cast(PrimitiveType::U64)?;
                DSLStmt::emit(0, arg_arr.expr().at(&idx)?)
            })
            .unwrap(),
        );

        let func = dsl2::compile(func, dsl_args![arr, idx]).unwrap();

        Ok(TakeKernel(func))
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
    use std::sync::Arc;

    use arrow_array::{
        builder::{FixedSizeListBuilder, Float32Builder, Int32Builder, ListBuilder},
        cast::AsArray,
        types::{Float32Type, Int32Type, Int64Type},
        BooleanArray, FixedSizeListArray, Int32Array, Int64Array, RunArray, StringArray,
        UInt16Array, UInt32Array, UInt8Array,
    };
    use arrow_schema::{DataType, Field};
    use itertools::Itertools;

    use crate::{compiled_kernels::Kernel, dictionary_data_type};

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
    fn test_take_recursive_ree_dict() {
        let values = Int32Array::from(vec![1, 2, 3]);
        let values = arrow_cast::cast(
            &values,
            &dictionary_data_type(DataType::Int8, DataType::Int32),
        )
        .unwrap();
        let ends = Int32Array::from(vec![2, 4, 6]);
        let data = RunArray::<Int32Type>::try_new(&ends, values.as_ref()).unwrap();
        let idxes = UInt8Array::from(vec![0, 2, 5]);

        let k = TakeKernel::compile(&(&data, &idxes), ()).unwrap();
        let res = k.call((&data, &idxes)).unwrap();

        assert_eq!(res.data_type(), &DataType::Int32);
        assert_eq!(res.as_primitive::<Int32Type>().values(), &[1, 2, 3]);
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

    #[test]
    fn test_take_fixed_size_list_bool() {
        let values = BooleanArray::from(vec![
            true, false, true, false, true, false, false, false, true,
        ]);
        let data = FixedSizeListArray::try_new(
            Arc::new(Field::new_list_field(DataType::Boolean, false)),
            3,
            Arc::new(values),
            None,
        )
        .unwrap();
        let idxes = UInt32Array::from(vec![2, 0]);

        let k = TakeKernel::compile(&(&data, &idxes), ()).unwrap();
        let res = k.call((&data, &idxes)).unwrap();
        let res = res.as_fixed_size_list();

        assert_eq!(
            res.values().as_boolean().values().iter().collect_vec(),
            &[false, false, true, true, false, true]
        );
    }

    #[test]
    fn test_take_vec() {
        let mut b = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        b.values().append_slice(&[0.0, 1.0]);
        b.append(true);
        b.values().append_slice(&[2.0, 3.0]);
        b.append(true);
        b.values().append_slice(&[4.0, 5.0]);
        b.append(true);

        let data = b.finish();
        let sel = Int32Array::from(vec![1]);
        let k = TakeKernel::compile(&(&data, &sel), ()).unwrap();
        let res = k.call((&data, &sel)).unwrap();
        let res = res.as_fixed_size_list();
        let val = res.value(0);
        let val = val.as_primitive::<Float32Type>();
        assert_eq!(val.values(), &[2.0, 3.0]);
    }

    #[test]
    fn test_take_list_repeated_nullable_rows_and_values() {
        let mut data = ListBuilder::new(Int32Builder::new());
        data.values().append_value(1);
        data.values().append_null();
        data.values().append_value(3);
        data.append(true);
        data.append(false);
        data.append(true);
        data.values().append_null();
        data.values().append_value(5);
        data.append(true);
        let data = data.finish();
        let indices = UInt32Array::from(vec![3, 0, 0, 1]);

        let kernel = TakeKernel::compile(&(&data, &indices), ()).unwrap();
        let actual = kernel.call((&data, &indices)).unwrap();
        let expected = arrow_select::take::take(&data, &indices, None).unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(actual.as_list::<i32>(), expected.as_list::<i32>());
    }

    #[test]
    fn test_take_nested_list_nullable_rows_and_values() {
        let mut data = ListBuilder::new(ListBuilder::new(Int32Builder::new()));
        data.values().values().append_value(1);
        data.values().values().append_null();
        data.values().append(true);
        data.values().append(false);
        data.append(true);
        data.append(true);
        data.values().values().append_slice(&[3, 4]);
        data.values().append(true);
        data.append(true);
        let data = data.finish();
        let indices = UInt32Array::from(vec![2, 0, 2]);

        let kernel = TakeKernel::compile(&(&data, &indices), ()).unwrap();
        let actual = kernel.call((&data, &indices)).unwrap();
        let expected = arrow_select::take::take(&data, &indices, None).unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(actual.as_list::<i32>(), expected.as_list::<i32>());
    }
}
