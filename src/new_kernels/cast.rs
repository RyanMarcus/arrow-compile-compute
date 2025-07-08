use std::sync::Arc;

use super::{ArrowKernelError, Kernel};
use crate::{
    dsl::{DSLKernel, KernelOutputType},
    PrimitiveType,
};
use arrow_array::{
    cast::AsArray,
    make_array,
    types::{Int16Type, Int32Type, Int64Type, Int8Type},
    Array, ArrayRef, StringArray, StringViewArray,
};
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;

pub struct CastKernel {
    k: DSLKernel,
    tar: DataType,
}
unsafe impl Sync for CastKernel {}
unsafe impl Send for CastKernel {}

fn coalesce_type(res: ArrayRef, tar: &DataType) -> Result<ArrayRef, ArrowKernelError> {
    if res.data_type() == tar {
        return Ok(res);
    }

    // might need to translate binary type to string type
    match (res.data_type(), tar) {
        (DataType::Binary, DataType::Utf8) => {
            let res = res.as_binary();
            let (offsets, data, nulls) = res.clone().into_parts();
            debug_assert!(
                StringArray::try_new(offsets.clone(), data.clone(), nulls.clone()).is_ok()
            );
            let s = Arc::new(unsafe { StringArray::new_unchecked(offsets, data, nulls) });
            Ok(s)
        }
        (DataType::BinaryView, DataType::Utf8View) => {
            let res = res.as_binary_view();
            let (view, bufs, nulls) = res.clone().into_parts();
            let s = Arc::new(StringViewArray::new(view, bufs, nulls));
            Ok(s)
        }
        (DataType::Dictionary(kt, _vt), DataType::Dictionary(t_kt, t_vt)) if kt == t_kt => {
            match kt.as_ref() {
                DataType::Int8 => {
                    let res = res.as_dictionary::<Int8Type>();
                    Ok(Arc::new(res.with_values(coalesce_type(
                        res.values().clone(),
                        t_vt.as_ref(),
                    )?)))
                }
                DataType::Int16 => {
                    let res = res.as_dictionary::<Int16Type>();
                    Ok(Arc::new(res.with_values(coalesce_type(
                        res.values().clone(),
                        t_vt.as_ref(),
                    )?)))
                }
                DataType::Int32 => {
                    let res = res.as_dictionary::<Int32Type>();
                    Ok(Arc::new(res.with_values(coalesce_type(
                        res.values().clone(),
                        t_vt.as_ref(),
                    )?)))
                }
                DataType::Int64 => {
                    let res = res.as_dictionary::<Int64Type>();
                    Ok(Arc::new(res.with_values(coalesce_type(
                        res.values().clone(),
                        t_vt.as_ref(),
                    )?)))
                }
                _ => unreachable!("invalid dictionary key type {}", kt),
            }
        }
        (DataType::RunEndEncoded(re, _v), DataType::RunEndEncoded(t_re, t_v))
            if re.data_type() == t_re.data_type() =>
        {
            let arr = res.into_data();
            let res = arr.child_data()[0].clone();
            let val = make_array(arr.child_data()[1].clone());
            let val = coalesce_type(val, t_v.data_type())?;

            Ok(make_array(unsafe {
                ArrayDataBuilder::new(tar.clone())
                    .len(arr.len())
                    .add_child_data(res)
                    .add_child_data(val.into_data())
                    .build_unchecked()
            }))
        }

        _ => todo!("unable to coalesce {} into {}", res.data_type(), tar),
    }
}

impl Kernel for CastKernel {
    type Key = (DataType, DataType);

    type Input<'a>
        = &'a dyn Array
    where
        Self: 'a;

    type Params = DataType;

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let res = self.k.call(&[&inp])?;
        coalesce_type(res, &self.tar)
    }

    fn compile(arr: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let tar = params;
        let out_type = KernelOutputType::for_data_type(&tar).map_err(ArrowKernelError::DSLError)?;

        Ok(CastKernel {
            k: DSLKernel::compile(&[arr], |ctx| {
                let arr = ctx.get_input(0)?;
                ctx.iter_over(vec![arr])
                    .map(|i| vec![i[0].convert(PrimitiveType::for_arrow_type(&tar))])
                    .collect(out_type)
            })
            .map_err(ArrowKernelError::DSLError)?,
            tar,
        })
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((i.data_type().clone(), p.clone()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray,
        types::{Int32Type, Int8Type},
        Array, ArrayRef, DictionaryArray, Int32Array, Int64Array, RunArray, StringArray,
        UInt8Array,
    };
    use arrow_schema::DataType;
    use itertools::Itertools;

    use crate::{
        dictionary_data_type,
        new_kernels::{CastKernel, Kernel},
        run_end_data_type,
    };

    #[test]
    fn test_i32_to_i64() {
        let data = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let expected: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
        let k = CastKernel::compile(&(&data as &dyn Array), DataType::Int64).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i32_to_i64_block() {
        let data = Int32Array::from((0..200).collect_vec());
        let expected: ArrayRef = Arc::new(Int64Array::from((0..200).collect_vec()));
        let k = CastKernel::compile(&(&data as &dyn Array), DataType::Int64).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i64_to_u8_block() {
        let data = Int64Array::from((0..200).collect_vec());
        let expected: ArrayRef = Arc::new(UInt8Array::from((0..200).collect_vec()));
        let k = CastKernel::compile(&(&data as &dyn Array), DataType::UInt8).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i32_to_dict() {
        let data = Int32Array::from(vec![1, 1, 1, 2, 2, 300, 300, 400]);
        let k = CastKernel::compile(
            &(&data as &dyn Array),
            dictionary_data_type(DataType::Int8, DataType::Int32),
        )
        .unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(res.len(), 8);

        let res = res.as_dictionary::<Int8Type>();
        assert_eq!(
            &[1, 2, 300, 400],
            res.values().as_primitive::<Int32Type>().values()
        );
        assert_eq!(&[0, 0, 0, 1, 1, 2, 2, 3], res.keys().values());
    }

    #[test]
    fn test_dict_to_i32() {
        let keys = UInt8Array::from(vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5]);
        let values = Int32Array::from(vec![-100, -200, -300, -400, -500, -600]);
        let da = DictionaryArray::new(keys, Arc::new(values));
        let k = CastKernel::compile(&(&da as &dyn Array), DataType::Int32).unwrap();

        let res = k.call(&da).unwrap();
        assert_eq!(
            res.as_primitive::<Int32Type>().values(),
            &[-100, -100, -100, -200, -200, -200, -300, -300, -300, -400, -500, -500, -600, -600]
        );
    }

    #[test]
    fn test_str_to_dict() {
        let data = StringArray::from(vec![
            "this",
            "this",
            "is",
            "a test",
            "a test",
            "a string that is longer than 12 chars",
        ]);
        let k = CastKernel::compile(
            &(&data as &dyn Array),
            dictionary_data_type(DataType::Int8, DataType::Utf8),
        )
        .unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(res.len(), 6);

        let res = res.as_dictionary::<Int8Type>();
        let strv = res.values().as_string::<i32>();
        assert_eq!("this", strv.value(0));
        assert_eq!("is", strv.value(1));
        assert_eq!("a test", strv.value(2));
        assert_eq!(&[0, 0, 1, 2, 2, 3], res.keys().values());
    }

    #[test]
    fn test_dict_to_str_view() {
        let data = StringArray::from(vec![
            "this",
            "this",
            "is",
            "a test",
            "a string that is longer than 12 chars",
            "a test",
        ]);
        let ddata =
            arrow_cast::cast::cast(&data, &dictionary_data_type(DataType::Int8, DataType::Utf8))
                .unwrap();
        let k = CastKernel::compile(&(&ddata as &dyn Array), DataType::Utf8View).unwrap();
        let res = k.call(&ddata).unwrap();
        let res = res.as_string_view();

        assert_eq!(res.len(), data.len());
        for (ours, orig) in res.iter().zip(data.iter()) {
            assert_eq!(ours, orig);
        }
    }

    #[test]
    fn test_dict_to_str_flat() {
        let data = StringArray::from(vec![
            "this",
            "this",
            "is",
            "a test",
            "a test",
            "a string that is longer than 12 chars",
        ]);
        let ddata =
            arrow_cast::cast::cast(&data, &dictionary_data_type(DataType::Int8, DataType::Utf8))
                .unwrap();
        let k = CastKernel::compile(&(&ddata as &dyn Array), DataType::Utf8).unwrap();
        let res = k.call(&ddata).unwrap();
        let res = res.as_string::<i32>();

        assert_eq!(res.len(), data.len());
        for (ours, orig) in res.iter().zip(data.iter()) {
            assert_eq!(ours, orig);
        }
    }

    #[test]
    fn test_str_dict_to_ree() {
        let data = StringArray::from(vec![
            "this",
            "this",
            "is",
            "a test",
            "a test",
            "a string that is longer than 12 chars",
        ]);
        let ddata =
            arrow_cast::cast::cast(&data, &dictionary_data_type(DataType::Int8, DataType::Utf8))
                .unwrap();

        let k = CastKernel::compile(
            &(&ddata as &dyn Array),
            run_end_data_type(&DataType::Int32, &DataType::Utf8),
        )
        .unwrap();
        let res = k.call(&ddata).unwrap();
        let res = res.as_any().downcast_ref::<RunArray<Int32Type>>().unwrap();
        let res = res.downcast::<StringArray>().unwrap();

        assert_eq!(res.len(), data.len());
        for (ours, orig) in res.into_iter().zip(data.iter()) {
            assert_eq!(ours, orig);
        }
    }
}
