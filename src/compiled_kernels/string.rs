use arrow_array::{cast::AsArray, Array, BooleanArray, Datum};
use arrow_buffer::BooleanBufferBuilder;
use arrow_schema::DataType;
use memchr::memmem::Finder;

use crate::{
    compiled_kernels::dsl::{DSLKernel, KernelOutputType},
    logical_nulls, ArrowKernelError, Kernel,
};

pub fn string_contains(data: &dyn Array, pattern: &[u8]) -> Result<BooleanArray, ArrowKernelError> {
    let finder = Finder::new(pattern);

    if data.null_count() == 0 {
        let mut builder = BooleanBufferBuilder::new(data.len());
        for bytes in crate::arrow_interface::iter::iter_nonnull_bytes(data)? {
            builder.append(finder.find(bytes).is_some());
        }
        return Ok(BooleanArray::from(builder.finish()));
    }

    let mut last_position = 0;
    let mut builder = BooleanBufferBuilder::new(data.len());
    for (idx, bytes) in crate::arrow_interface::iter::iter_nonnull_bytes(data)?.indexed() {
        if idx > last_position {
            builder.append_n(idx - last_position, false);
        }
        builder.append(finder.find(bytes).is_some());
        last_position = idx + 1;
    }

    if last_position < data.len() {
        builder.append_n(data.len() - last_position, false);
    }

    Ok(BooleanArray::from(builder.finish()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StringKernelType {
    StartsWith,
    EndsWith,
}

pub struct StringStartEndKernel(DSLKernel);
unsafe impl Sync for StringStartEndKernel {}
unsafe impl Send for StringStartEndKernel {}

impl Kernel for StringStartEndKernel {
    type Key = (DataType, DataType, bool, StringKernelType);

    type Input<'a>
        = (&'a dyn Array, &'a dyn Datum)
    where
        Self: 'a;

    type Params = StringKernelType;

    type Output = BooleanArray;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (needle, is_scalar) = inp.1.get();
        if is_scalar && needle.is_null(0) {
            return Ok(BooleanArray::from(vec![false; inp.0.len()]));
        }

        let mut res = self.0.call(&[&inp.0, inp.1])?.as_boolean().clone();
        if let Some(nulls) = logical_nulls(inp.0)? {
            let b1 = nulls.inner();
            let b2 = res.values();
            res = BooleanArray::from(b1 & b2);
        }
        Ok(res)
    }

    fn compile(inp: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, needle) = inp;

        Ok(StringStartEndKernel(
            DSLKernel::compile(&[arr, *needle], |ctx| {
                let arr = ctx.get_input(0)?;
                let needle = ctx.get_input(1)?;
                ctx.iter_over(vec![arr, needle])
                    .map(|i| match params {
                        StringKernelType::StartsWith => vec![i[0].starts_with(&i[1])],
                        StringKernelType::EndsWith => vec![i[0].ends_with(&i[1])],
                    })
                    .collect(KernelOutputType::Boolean)
            })
            .map_err(ArrowKernelError::DSLError)?,
        ))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        let haystack_type = i.0.data_type().clone();
        let (needle_data, is_scalar) = i.1.get();
        Ok((
            haystack_type,
            needle_data.data_type().clone(),
            is_scalar,
            *p,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{StringKernelType, StringStartEndKernel};
    use crate::compiled_kernels::Kernel;
    use arrow_array::{BooleanArray, Scalar, StringArray, StringViewArray};

    #[test]
    fn test_string_starts_with_kernel() {
        let source = StringArray::from(vec!["foobar", "barfoo", "foobaz"]);
        let needle = Scalar::new(StringArray::from(vec!["foo"]));
        let kernel =
            StringStartEndKernel::compile(&(&source, &needle), StringKernelType::StartsWith)
                .unwrap();
        let result = kernel.call((&source, &needle)).unwrap();

        assert_eq!(result, BooleanArray::from(vec![true, false, true]));
    }

    #[test]
    fn test_string_ends_with_kernel() {
        let source = StringArray::from(vec!["foobar", "barfoo", "bazfoo"]);
        let needle = Scalar::new(StringArray::from(vec!["foo"]));

        let kernel =
            StringStartEndKernel::compile(&(&source, &needle), StringKernelType::EndsWith).unwrap();
        let result = kernel.call((&source, &needle)).unwrap();

        assert_eq!(result, BooleanArray::from(vec![false, true, true]));
    }

    #[test]
    fn test_string_starts_with_kernel_nulls() {
        let source = StringArray::from(vec![
            Some("prefix-value"),
            None,
            Some("other"),
            Some("prefix"),
        ]);
        let needle = Scalar::new(StringArray::from(vec!["prefix"]));

        let kernel =
            StringStartEndKernel::compile(&(&source, &needle), StringKernelType::StartsWith)
                .unwrap();
        let result = kernel.call((&source, &needle)).unwrap();

        assert_eq!(result, BooleanArray::from(vec![true, false, false, true]));
    }

    #[test]
    fn test_string_view_ends_with_kernel() {
        let source = StringViewArray::from(vec!["alpha", "beta-suffix", "suffix"]);
        let needle = Scalar::new(StringArray::from(vec!["suffix"]));

        let kernel =
            StringStartEndKernel::compile(&(&source, &needle), StringKernelType::EndsWith).unwrap();
        let result = kernel.call((&source, &needle)).unwrap();

        assert_eq!(result, BooleanArray::from(vec![false, true, true]));
    }
}
