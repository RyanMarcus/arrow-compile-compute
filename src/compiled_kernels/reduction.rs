use arrow_array::{new_null_array, ArrayRef, BooleanArray, Datum};
use arrow_schema::DataType;

use crate::compiled_kernels::dsl2::{
    self, DSLArgument, DSLContext, DSLFunction, DSLReductionType, DSLStmt, DSLType, WriterSpec,
};
use crate::compiled_kernels::Kernel;
use crate::{logical_nulls, normalized_base_type, ArrowKernelError, PrimitiveType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionKernelType {
    Min,
    Max,
    ArgMin,
    ArgMax,
}

impl ReductionKernelType {
    fn dsl_reduction_type(self) -> DSLReductionType {
        match self {
            Self::Min => DSLReductionType::Min,
            Self::Max => DSLReductionType::Max,
            Self::ArgMin => DSLReductionType::ArgMin,
            Self::ArgMax => DSLReductionType::ArgMax,
        }
    }

    fn is_arg(self) -> bool {
        matches!(self, Self::ArgMin | Self::ArgMax)
    }
}

pub struct ReductionKernel {
    k: dsl2::RunnableDSLFunction,
    has_nulls: bool,
    output_type: DataType,
}
unsafe impl Sync for ReductionKernel {}
unsafe impl Send for ReductionKernel {}

impl Kernel for ReductionKernel {
    type Key = (DataType, ReductionKernelType, bool, bool);

    type Input<'a>
        = &'a dyn Datum
    where
        Self: 'a;

    type Params = ReductionKernelType;

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let result = if self.has_nulls {
            let nulls = logical_nulls(inp.get().0)?.ok_or_else(|| {
                ArrowKernelError::UnsupportedArguments(
                    "nullable reduction kernel called with non-null input".to_string(),
                )
            })?;
            let validity = BooleanArray::new(nulls.into_inner(), None);
            self.k
                .run(&[DSLArgument::Datum(inp), DSLArgument::Datum(&validity)])
        } else {
            self.k.run(&[DSLArgument::Datum(inp)])
        };

        match result {
            Ok(result) => Ok(result[0].clone()),
            Err(ArrowKernelError::RuntimeEmptyReduction) => {
                Ok(new_null_array(&self.output_type, 1))
            }
            Err(err) => Err(err),
        }
    }

    fn compile(inp: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, _) = inp.get();

        let data_type = arr.data_type();
        if data_type == &DataType::Boolean {
            return Err(ArrowKernelError::UnsupportedArguments(
                "min/max reductions do not support boolean values".to_string(),
            ));
        }
        let primitive_type = PrimitiveType::for_arrow_type(data_type);
        if matches!(primitive_type, PrimitiveType::List(..)) {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "min/max reductions do not support fixed-size-list values, got {data_type}"
            )));
        }

        let has_nulls = logical_nulls(arr)?.is_some();
        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("reduce");
        let arr = func.add_arg(&mut ctx, DSLType::array_like(*inp, "n"));
        let validity = BooleanArray::from(Vec::<bool>::new());
        let reduced = if has_nulls {
            let valid = func.add_arg(&mut ctx, DSLType::array_like(&validity, "n"));
            DSLStmt::reduce_where(
                &mut ctx,
                params.dsl_reduction_type(),
                &[arr, valid],
                |loop_vars| Ok(loop_vars[0].expr()),
                |loop_vars| Ok(loop_vars[1].expr()),
            )?
        } else {
            DSLStmt::reduce(&mut ctx, params.dsl_reduction_type(), &[arr], |loop_vars| {
                Ok(loop_vars[0].expr())
            })?
        };

        func.add_body(reduced.stmt);
        func.add_body(DSLStmt::emit(0, reduced.value.expr())?);
        let output = if params.is_arg() {
            WriterSpec::Primitive(PrimitiveType::U64)
        } else {
            WriterSpec::for_base_type_of_datum(*inp)
        };
        let output_type = if params.is_arg() {
            DataType::UInt64
        } else {
            normalized_base_type(data_type)
        };
        func.add_ret(output, "<= n");

        let k = if has_nulls {
            dsl2::compile(
                func,
                [DSLArgument::Datum(*inp), DSLArgument::Datum(&validity)],
            )?
        } else {
            dsl2::compile(func, [DSLArgument::Datum(*inp)])?
        };

        Ok(Self {
            k,
            has_nulls,
            output_type,
        })
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        let (arr, is_scalar) = i.get();
        Ok((
            arr.data_type().clone(),
            *p,
            logical_nulls(arr)?.is_some(),
            is_scalar,
        ))
    }
}
