use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;

use crate::compiled_kernels::cast::coalesce_type;
use crate::compiled_kernels::dsl2::{
    compile, dsl_args, DSLArgument, DSLContext, DSLFunction, DSLStmt, DSLType, RunnableDSLFunction,
};
use crate::compiled_kernels::null_utils::replace_nulls;
use crate::compiled_writers::WriterSpec;
use crate::{logical_nulls, normalized_base_type, ArrowKernelError};

use crate::compiled_kernels::Kernel;

pub struct FilterKernel(RunnableDSLFunction);
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
        if inp.0.len() != inp.1.len() {
            return Err(ArrowKernelError::SizeMismatch);
        }

        let mut res = self.0.run(&dsl_args!(inp.0, inp.1))?[0].clone();
        if let Some(nulls) = logical_nulls(inp.0)? {
            let ba = BooleanArray::new(nulls.into_inner(), None);
            let filtered_nulls = crate::arrow_interface::select::filter(&ba, inp.1)?;
            let filtered_nulls = filtered_nulls.as_boolean();
            let filtered_nulls = NullBuffer::new(filtered_nulls.clone().into_parts().0);
            res = replace_nulls(res, Some(filtered_nulls));
        }

        let base_dt = normalized_base_type(inp.0.data_type());
        res = coalesce_type(res, &base_dt)?;

        Ok(res)
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, filt) = inp;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("filter");
        let arr_arg = func.add_arg(&mut ctx, DSLType::array_like(arr, "n"));

        if matches!(arr.data_type(), DataType::RunEndEncoded(_, _)) {
            let fil_arg = func.add_arg(&mut ctx, DSLType::array_like(filt, "n"));
            func.add_ret(WriterSpec::for_base_type_of_datum(arr), "<= n");

            func.add_body(
                DSLStmt::for_each(&mut ctx, &[arr_arg, fil_arg], |loop_vars| {
                    let item = loop_vars[0].expr();
                    let filt = loop_vars[1].expr();
                    DSLStmt::cond(filt, DSLStmt::emit(0, item)?)
                })
                .unwrap(),
            );
        } else {
            let fil_arg = func.add_arg(&mut ctx, DSLType::set_bits("m"));
            func.add_ret(WriterSpec::for_base_type_of_datum(arr), "m");

            func.add_body(
                DSLStmt::for_each(&mut ctx, &[fil_arg], |loop_vars| {
                    let idx = loop_vars[0].expr();
                    let item = arr_arg.expr().at(&idx)?;
                    DSLStmt::emit(0, item)
                })
                .unwrap(),
            );
        }

        let func = compile(func, [DSLArgument::Datum(arr), DSLArgument::Datum(filt)])?;
        Ok(FilterKernel(func))
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
    use arrow_array::{cast::AsArray, types::Int32Type, Array, BooleanArray, Int32Array, RunArray};
    use arrow_schema::DataType;
    use itertools::Itertools;

    use crate::{compiled_kernels::Kernel, dictionary_data_type};

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
    fn test_filter_i32nullable() {
        let data = Int32Array::from(vec![Some(1), None, Some(3), Some(4), Some(5), Some(6)]);
        let filt = BooleanArray::from(vec![true, true, false, false, true, true]);

        let k = FilterKernel::compile(&(&data, &filt), ()).unwrap();
        let res = k.call((&data, &filt)).unwrap();

        assert_eq!(
            res.as_primitive::<Int32Type>().iter().collect_vec(),
            vec![Some(1), None, Some(5), Some(6)]
        )
    }

    #[test]
    fn test_filter_i32_long() {
        let data = Int32Array::from((0..1000).collect_vec());
        let filt = BooleanArray::from((0..1000).map(|x| x < 500).collect_vec());
        let k = FilterKernel::compile(&(&data, &filt), ()).unwrap();
        let res = k.call((&data, &filt)).unwrap();
        assert_eq!(res.len(), 500);
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

    #[test]
    fn test_filter_recursive_ree_dict() {
        let values = Int32Array::from(vec![1, 2, 3]);
        let values = arrow_cast::cast(
            &values,
            &dictionary_data_type(DataType::Int8, DataType::Int32),
        )
        .unwrap();
        let ends = Int32Array::from(vec![2, 4, 6]);
        let data = RunArray::<Int32Type>::try_new(&ends, values.as_ref()).unwrap();
        let filt = BooleanArray::from(vec![true, false, true, false, true, false]);

        let k = FilterKernel::compile(&(&data, &filt), ()).unwrap();
        let res = k.call((&data, &filt)).unwrap();

        assert_eq!(res.data_type(), &DataType::Int32);
        assert_eq!(res.as_primitive::<Int32Type>().values(), &[1, 2, 3]);
    }

    #[test]
    fn stress_test_filter_i32() {
        let data = Int32Array::from(vec![10; 256]);

        std::thread::scope(|s| {
            for tid in 0..16 {
                let ptid = tid;
                let pdata = data.clone();
                s.spawn(move || {
                    let mut rng = fastrand::Rng::with_seed(ptid);
                    for _ in 0..10000 {
                        let filt = BooleanArray::from((0..256).map(|_| rng.bool()).collect_vec());
                        crate::arrow_interface::select::filter(&pdata, &filt).unwrap();
                    }
                });
            }
        });
    }

    #[test]
    fn test_filter_string() {
        let data = arrow_array::StringArray::from(vec!["a", "b", "c", "d"]);
        let filt = BooleanArray::from(vec![true, false, true, false]);

        let k = FilterKernel::compile(&(&data, &filt), ()).unwrap();
        let res = k.call((&data, &filt)).unwrap();
        let res = res.as_string::<i32>();

        assert_eq!(res.iter().collect_vec(), vec![Some("a"), Some("c")]);
    }

    #[test]
    fn test_filter_string_empty() {
        let data = arrow_array::StringArray::from(Vec::<&str>::new());
        let filt = BooleanArray::from(Vec::<bool>::new());

        let k = FilterKernel::compile(&(&data, &filt), ()).unwrap();
        let res = k.call((&data, &filt)).unwrap();
        let res = res.as_string::<i32>();

        assert_eq!(res.len(), 0);
        assert_eq!(res.iter().collect_vec(), Vec::<Option<&str>>::new());
    }
}
