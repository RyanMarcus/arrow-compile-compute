use arrow_array::{Array, ArrayRef, Datum, UInt64Array};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        cast::coalesce_type,
        dsl2::{
            compile, DSLArgument, DSLContext, DSLFunction, DSLStmt, DSLType, RunnableDSLFunction,
        },
        null_utils::replace_nulls,
    },
    compiled_writers::WriterSpec,
    logical_arrow_type, logical_nulls, ArrowKernelError, Kernel, PrimitiveType,
};

fn validate_values(values: &[&dyn Array]) -> Result<DataType, ArrowKernelError> {
    let Some(first) = values.first() else {
        return Err(ArrowKernelError::UnsupportedArguments(
            "Cannot interleave empty array list".to_string(),
        ));
    };

    let expected = logical_arrow_type(first.data_type());
    for value in &values[1..] {
        let actual = logical_arrow_type(value.data_type());
        if actual != expected {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "all arrays for interleave must have the same logical type (expected {}, got {})",
                expected, actual
            )));
        }
    }

    Ok(expected)
}

fn interleave_nulls(
    values: &[&dyn Array],
    array_indices: &UInt64Array,
    element_indices: &UInt64Array,
) -> Result<Option<NullBuffer>, ArrowKernelError> {
    if !values.iter().any(|array| array.is_nullable()) {
        return Ok(None);
    }

    let source_nulls = values
        .iter()
        .map(|array| logical_nulls(*array))
        .collect::<Result<Vec<_>, _>>()?;

    let nulls = BooleanBuffer::collect_bool(array_indices.len(), |idx| {
        let array_idx = array_indices.value(idx) as usize;
        let element_idx = element_indices.value(idx) as usize;
        source_nulls[array_idx]
            .as_ref()
            .map(|nulls| nulls.is_valid(element_idx))
            .unwrap_or(true)
    });
    Ok(Some(NullBuffer::new(nulls)))
}

pub struct InterleaveKernel(RunnableDSLFunction);
unsafe impl Sync for InterleaveKernel {}
unsafe impl Send for InterleaveKernel {}

impl Kernel for InterleaveKernel {
    type Key = (Vec<DataType>, DataType, DataType);

    type Input<'a>
        = (&'a [&'a dyn Array], &'a UInt64Array, &'a UInt64Array)
    where
        Self: 'a;

    type Params = ();

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (values, array_indices, element_indices) = inp;
        let output_type = validate_values(values)?;

        let values_arg = DSLArgument::two_d(values.iter().map(|value| value as &dyn Datum));
        let mut result = self.0.run(&[
            values_arg,
            DSLArgument::datum(array_indices),
            DSLArgument::datum(element_indices),
        ])?[0]
            .clone();

        if let Some(nulls) = interleave_nulls(values, array_indices, element_indices)? {
            result = replace_nulls(result, Some(nulls));
        }

        coalesce_type(result, &output_type)
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (values, array_indices, element_indices) = inp;
        validate_values(values)?;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("interleave");
        let values_arg = func.add_arg(
            &mut ctx,
            DSLType::two_d_array_of(PrimitiveType::for_arrow_type(values[0].data_type())),
        );
        let array_idx_arg = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::U64, "n"));
        let element_idx_arg = func.add_arg(&mut ctx, DSLType::array_of(PrimitiveType::U64, "n"));
        func.add_ret(WriterSpec::for_base_type_of_datum(&values[0]), "n");
        func.add_body(DSLStmt::for_each(
            &mut ctx,
            &[array_idx_arg, element_idx_arg],
            |loop_vars| {
                let array_idx = loop_vars[0].expr();
                let element_idx = loop_vars[1].expr();
                let value = values_arg.expr().at(&array_idx)?.at(&element_idx)?;
                DSLStmt::emit(0, value)
            },
        )?);

        Ok(Self(compile(
            func,
            [
                DSLArgument::two_d(values.iter().map(|value| value as &dyn Datum)),
                DSLArgument::datum(array_indices),
                DSLArgument::datum(element_indices),
            ],
        )?))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((
            i.0.iter().map(|array| array.data_type().clone()).collect(),
            i.1.data_type().clone(),
            i.2.data_type().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        builder::{FixedSizeListBuilder, Float32Builder},
        cast::AsArray,
        types::{Float32Type, Int32Type},
        Int32Array, StringArray, UInt64Array,
    };
    use itertools::Itertools;

    use crate::{compiled_kernels::Kernel, select};

    use super::InterleaveKernel;

    #[test]
    fn test_interleave_i32() {
        let left = Int32Array::from(vec![1, 4, 6]);
        let right = Int32Array::from(vec![2, 3, 5]);
        let array_indices = UInt64Array::from(vec![0, 1, 1, 0, 1, 0]);
        let element_indices = UInt64Array::from(vec![0, 0, 1, 1, 2, 2]);

        let kernel =
            InterleaveKernel::compile(&(&[&left, &right], &array_indices, &element_indices), ())
                .unwrap();
        let out = kernel
            .call((&[&left, &right], &array_indices, &element_indices))
            .unwrap();

        assert_eq!(
            out.as_primitive::<Int32Type>().values(),
            &[1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn test_interleave_preserves_nulls() {
        let left = Int32Array::from(vec![Some(1), None, Some(5)]);
        let right = Int32Array::from(vec![Some(2), Some(3), None]);

        let out = select::interleave(&[&left, &right], &[(0, 0), (1, 0), (0, 1), (1, 2)]).unwrap();

        assert_eq!(
            out.as_primitive::<Int32Type>().iter().collect_vec(),
            vec![Some(1), Some(2), None, None]
        );
    }

    #[test]
    fn test_interleave_strings() {
        let left = StringArray::from(vec!["a", "d"]);
        let right = StringArray::from(vec!["b", "c"]);

        let out = select::interleave(&[&left, &right], &[(0, 0), (1, 0), (1, 1), (0, 1)]).unwrap();
        let out = out
            .as_string::<i32>()
            .iter()
            .map(|value| value.unwrap())
            .collect_vec();

        assert_eq!(out, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_interleave_fixed_size_lists() {
        let mut left = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        left.values().append_slice(&[1.0, 2.0]);
        left.append(true);
        left.values().append_slice(&[5.0, 6.0]);
        left.append(true);
        let left = left.finish();

        let mut right = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        right.values().append_slice(&[3.0, 4.0]);
        right.append(true);
        let right = right.finish();

        let out = select::interleave(&[&left, &right], &[(0, 0), (1, 0), (0, 1)]).unwrap();
        let out = out.as_fixed_size_list();

        let first = out.value(0).as_primitive::<Float32Type>().values().to_vec();
        let second = out.value(1).as_primitive::<Float32Type>().values().to_vec();
        let third = out.value(2).as_primitive::<Float32Type>().values().to_vec();

        assert_eq!(first, vec![1.0, 2.0]);
        assert_eq!(second, vec![3.0, 4.0]);
        assert_eq!(third, vec![5.0, 6.0]);
    }
}
