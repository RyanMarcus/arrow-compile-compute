use super::{ArrowKernelError, Kernel};
use crate::{
    dsl::{DSLKernel, KernelOutputType},
    PrimitiveType,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;

pub struct CastKernel(DSLKernel);
unsafe impl Sync for CastKernel {}
unsafe impl Send for CastKernel {}

impl Kernel for CastKernel {
    type Key = (DataType, DataType);

    type Input<'a>
        = &'a dyn Array
    where
        Self: 'a;

    type Params = DataType;

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        self.0.call(&[&inp])
    }

    fn compile(arr: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let tar = params;
        let out_type = KernelOutputType::for_data_type(&tar).map_err(ArrowKernelError::DSLError)?;

        Ok(CastKernel(
            DSLKernel::compile(&[arr], |ctx| {
                let arr = ctx.get_input(0)?;
                ctx.iter_over(vec![arr])
                    .map(|i| vec![i[0].convert(PrimitiveType::for_arrow_type(&tar))])
                    .collect(out_type)
            })
            .map_err(ArrowKernelError::DSLError)?,
        ))
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
        Array, ArrayRef, DictionaryArray, Int32Array, Int64Array, StringArray, UInt8Array,
    };
    use arrow_schema::DataType;
    use itertools::Itertools;

    use crate::{
        dictionary_data_type,
        new_kernels::{CastKernel, Kernel},
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
        let strv = res.values().as_string_view();
        assert_eq!("this", strv.value(0));
        assert_eq!("is", strv.value(1));
        assert_eq!("a test", strv.value(2));
        assert_eq!(&[0, 0, 1, 2, 2, 3], res.keys().values());
    }

    /*#[test]
    fn test_dict_to_str_view() {
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
        let k = DSLCastToFlatKernel::compile(&(&ddata as &dyn Array), DataType::Utf8View).unwrap();
        let res = k.call(&ddata).unwrap();
        let res = res.as_string_view();

        assert_eq!(res.len(), data.len());
        for (ours, orig) in res.iter().zip(data.iter()) {
            assert_eq!(ours, orig);
        }
    }*/

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
}
