use arrow_array::{cast::AsArray, Array, Datum, UInt64Array};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        dsl2::{
            compile, DSLArgument, DSLContext, DSLFunction, DSLStmt, DSLType, RunnableDSLFunction,
            WriterSpec,
        },
        null_utils::replace_nulls,
    },
    logical_nulls, normalized_base_type, ArrowKernelError, Kernel, PrimitiveType,
};

pub struct ListLenKernel(RunnableDSLFunction);
unsafe impl Sync for ListLenKernel {}
unsafe impl Send for ListLenKernel {}

impl Kernel for ListLenKernel {
    type Key = DataType;

    type Input<'a>
        = &'a dyn Array
    where
        Self: 'a;

    type Params = ();

    type Output = UInt64Array;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let datum = &inp as &dyn Datum;
        let mut res = self.0.run(&[DSLArgument::Datum(datum)])?[0].clone();
        if let Some(nulls) = logical_nulls(inp)? {
            res = replace_nulls(res, Some(nulls));
        }
        Ok(res.as_primitive::<arrow_array::types::UInt64Type>().clone())
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        if !matches!(
            normalized_base_type(inp.data_type()),
            DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
        ) {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "len requires list input, got {}",
                inp.data_type()
            )));
        }

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("list_len");
        let data = *inp;
        let datum = &data as &dyn Datum;
        let arg = func.add_arg(&mut ctx, DSLType::array_like(datum, "n"));
        func.add_ret(WriterSpec::Primitive(PrimitiveType::U64), "n");
        func.add_body(DSLStmt::for_each(&mut ctx, &[arg], |loop_vars| {
            DSLStmt::emit(0, loop_vars[0].expr().list_len()?)
        })?);

        Ok(Self(compile(func, [DSLArgument::Datum(datum)])?))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok(i.data_type().clone())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        builder::{FixedSizeListBuilder, Int32Builder, LargeListBuilder, ListBuilder},
        types::{Int16Type, Int8Type, UInt8Type},
        Array, DictionaryArray, Int16Array, Int32Array, Int8Array, UInt8Array,
    };

    use super::ListLenKernel;
    use crate::Kernel;

    fn collect_lengths(arr: &dyn Array) -> Vec<Option<u64>> {
        let kernel = ListLenKernel::compile(&arr, ()).unwrap();
        kernel.call(arr).unwrap().iter().collect()
    }

    fn list_array() -> arrow_array::ListArray {
        let mut builder = ListBuilder::new(Int32Builder::new());
        builder.values().append_slice(&[1, 2, 3]);
        builder.append(true);
        builder.append(true);
        builder.append(false);
        builder.values().append_slice(&[4]);
        builder.append(true);
        builder.finish()
    }

    fn large_list_array() -> arrow_array::LargeListArray {
        let mut builder = LargeListBuilder::new(Int32Builder::new());
        builder.values().append_slice(&[10, 11]);
        builder.append(true);
        builder.append(true);
        builder.append(false);
        builder.values().append_slice(&[12, 13, 14, 15]);
        builder.append(true);
        builder.finish()
    }

    #[test]
    fn test_len_fixed_size_list() {
        let mut builder = FixedSizeListBuilder::new(Int32Builder::new(), 3);
        builder.values().append_slice(&[1, 2, 3]);
        builder.append(true);
        builder.values().append_slice(&[4, 5, 6]);
        builder.append(false);
        builder.values().append_slice(&[7, 8, 9]);
        builder.append(true);
        let arr = builder.finish();

        assert_eq!(collect_lengths(&arr), vec![Some(3), None, Some(3)]);
    }

    #[test]
    fn test_len_list() {
        let arr = list_array();
        assert_eq!(collect_lengths(&arr), vec![Some(3), Some(0), None, Some(1)]);
    }

    #[test]
    fn test_len_large_list() {
        let arr = large_list_array();
        assert_eq!(collect_lengths(&arr), vec![Some(2), Some(0), None, Some(4)]);
    }

    #[test]
    fn test_len_sliced_list() {
        let arr = list_array();
        let sliced = arr.slice(1, 3);

        assert_eq!(collect_lengths(&sliced), vec![Some(0), None, Some(1)]);
    }

    #[test]
    fn test_len_dictionary_large_list() {
        let values = large_list_array();
        let keys = Int8Array::from(vec![Some(0), Some(1), None, Some(3), Some(2)]);
        let dict = DictionaryArray::<Int8Type>::new(keys, Arc::new(values));

        assert_eq!(
            collect_lengths(&dict),
            vec![Some(2), Some(0), None, Some(4), None]
        );
    }

    #[test]
    fn test_len_nested_dictionary_large_list() {
        let values = large_list_array();
        let inner_keys = UInt8Array::from(vec![0, 3, 1, 2]);
        let inner = DictionaryArray::<UInt8Type>::new(inner_keys, Arc::new(values));
        let outer_keys = Int16Array::from(vec![Some(0), Some(1), Some(2), None, Some(3)]);
        let outer = DictionaryArray::<Int16Type>::new(outer_keys, Arc::new(inner));

        assert_eq!(
            collect_lengths(&outer),
            vec![Some(2), Some(4), Some(0), None, None]
        );
    }

    #[test]
    fn test_len_rejects_non_list() {
        let arr = Int32Array::from(vec![1, 2, 3]);
        assert!(ListLenKernel::compile(&(&arr as &dyn Array), ()).is_err());
    }
}
