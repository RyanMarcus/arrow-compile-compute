use arrow_array::{cast::AsArray, Array, BooleanArray};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        dsl2::{
            compile, dsl_args, DSLArgument, DSLContext, DSLFunction, DSLStmt, DSLType,
            RunnableDSLFunction,
        },
        null_utils::replace_nulls,
    },
    compiled_writers::WriterSpec,
    normalized_base_type, ArrowKernelError, Kernel,
};

pub struct NotKernel(RunnableDSLFunction);
unsafe impl Sync for NotKernel {}
unsafe impl Send for NotKernel {}

impl Kernel for NotKernel {
    type Key = DataType;
    type Input<'a> = &'a dyn Array;
    type Params = ();
    type Output = BooleanArray;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let res = self.0.run(&dsl_args!(inp))?[0].clone();
        let res = replace_nulls(res, inp.nulls().cloned());
        let res = res.as_boolean().clone();
        Ok(res)
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        if normalized_base_type(inp.data_type()) != DataType::Boolean {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "input must be boolean, got {}",
                inp.data_type()
            )));
        }

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("filter");
        let arr_arg = func.add_arg(&mut ctx, DSLType::array_like(inp, "n"));
        func.add_ret(WriterSpec::Boolean, "n");

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arr_arg], |loop_vars| {
                let item = loop_vars[0].expr().bit_not()?;
                DSLStmt::emit(0, item)
            })
            .unwrap(),
        );

        let func = compile(func, [DSLArgument::Datum(inp)])?;
        Ok(NotKernel(func))
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
    use super::*;
    use arrow_array::{Array, BooleanArray};

    #[test]
    fn test_not() {
        let input = BooleanArray::from(vec![Some(true), Some(false), None]);
        let kernel = NotKernel::compile(&(&input as &dyn Array), ()).unwrap();
        let output = kernel.call(&input).unwrap();

        let expected = BooleanArray::from(vec![Some(false), Some(true), None]);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_not_empty() {
        let input = BooleanArray::from(Vec::<Option<bool>>::new());
        let kernel = NotKernel::compile(&(&input as &dyn Array), ()).unwrap();
        let output = kernel.call(&input).unwrap();

        assert_eq!(output, input);
    }

    #[test]
    fn test_not_rejects_non_boolean_input() {
        let input = arrow_array::Int32Array::from(vec![1, 2, 3]);
        let result = NotKernel::compile(&(&input as &dyn Array), ());

        assert!(matches!(
            result,
            Err(ArrowKernelError::UnsupportedArguments(_))
        ));
    }
}
