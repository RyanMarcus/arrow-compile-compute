use std::sync::LazyLock;

use arrow_array::{Array, ArrayRef, BooleanArray, Datum, UInt64Array};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        aggregate2::Aggregator,
        dsl2::{
            compile, DSLArgument, DSLBuffer, DSLContext, DSLFunction, DSLStmt, DSLType, DSLValue,
            RunnableDSLFunction,
        },
        DSLArithBinOp, KernelCache,
    },
    logical_arrow_type, logical_nulls, ArrowKernelError, Kernel, PrimitiveType,
};

fn sum_primitive_type(pt: PrimitiveType) -> Result<PrimitiveType, ArrowKernelError> {
    Ok(match pt {
        PrimitiveType::I8 | PrimitiveType::I16 | PrimitiveType::I32 | PrimitiveType::I64 => {
            PrimitiveType::I64
        }
        PrimitiveType::U8 | PrimitiveType::U16 | PrimitiveType::U32 | PrimitiveType::U64 => {
            PrimitiveType::U64
        }
        PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => PrimitiveType::F64,
        PrimitiveType::P64x2 | PrimitiveType::List(_, _) => {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "sum only supports numeric types, got {pt:?}"
            )))
        }
    })
}

pub struct SumAggKernel {
    kernel: RunnableDSLFunction,
    has_nulls: bool,
}

unsafe impl Send for SumAggKernel {}
unsafe impl Sync for SumAggKernel {}

impl Kernel for SumAggKernel {
    type Key = (DataType, bool);
    type Input<'a> = (&'a mut DSLBuffer, &'a dyn Datum, &'a UInt64Array);
    type Params = ();
    type Output = ();

    fn call(&self, input: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (buffer, data, tickets) = input;
        let validity;
        let mut args = vec![
            DSLArgument::buffer(buffer),
            DSLArgument::Datum(data),
            DSLArgument::datum(tickets),
        ];
        if self.has_nulls {
            let nulls = logical_nulls(data.get().0)?.unwrap();
            validity = BooleanArray::new(nulls.into_inner(), None);
            args.push(DSLArgument::Datum(&validity));
        }
        self.kernel.run(&args)?;
        Ok(())
    }

    fn compile(input: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (buffer, data, tickets) = input;
        let sum_type = sum_primitive_type(PrimitiveType::for_arrow_type(data.get().0.data_type()))?;
        let has_nulls = logical_nulls(data.get().0)?.is_some();

        let mut context = DSLContext::new();
        let mut function = DSLFunction::new("sum");
        let buffer_arg = function.add_arg(&mut context, DSLType::buffer_of(sum_type, "k"));
        let data_arg = function.add_arg(&mut context, DSLType::array_like(*data, "n"));
        let tickets_arg = function.add_arg(&mut context, DSLType::array_like(tickets, "n"));
        let validity = BooleanArray::from(Vec::<bool>::new());

        if has_nulls {
            let validity_arg = function.add_arg(&mut context, DSLType::array_like(&validity, "n"));
            function.add_body(
                DSLStmt::for_each(
                    &mut context,
                    &[tickets_arg, data_arg, validity_arg],
                    |loop_vars| {
                        let ticket = loop_vars[0].expr();
                        let value = loop_vars[1].expr().primitive_cast(sum_type)?;
                        let valid = loop_vars[2].expr();
                        let current = buffer_arg.expr().at(&ticket)?;
                        DSLStmt::cond(
                            valid,
                            DSLStmt::set(
                                &buffer_arg,
                                &ticket,
                                &current.arith(DSLArithBinOp::Add, value)?,
                            )?,
                        )
                    },
                )
                .unwrap(),
            );
        } else {
            function.add_body(
                DSLStmt::for_each(&mut context, &[tickets_arg, data_arg], |loop_vars| {
                    let ticket = loop_vars[0].expr();
                    let value = loop_vars[1].expr().primitive_cast(sum_type)?;
                    let current = buffer_arg.expr().at(&ticket)?;
                    DSLStmt::set(
                        &buffer_arg,
                        &ticket,
                        &current.arith(DSLArithBinOp::Add, value)?,
                    )
                })
                .unwrap(),
            );
        }

        let mut empty_buffer = DSLBuffer::empty_like(buffer);
        let mut args = vec![
            DSLArgument::buffer(&mut empty_buffer),
            DSLArgument::Datum(*data),
            DSLArgument::datum(tickets),
        ];
        if has_nulls {
            args.push(DSLArgument::Datum(&validity));
        }

        Ok(Self {
            kernel: compile(function, args)?,
            has_nulls,
        })
    }

    fn get_key_for_input(
        input: &Self::Input<'_>,
        _params: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        let data_type = input.1.get().0.data_type().clone();
        sum_primitive_type(PrimitiveType::for_arrow_type(&data_type))?;
        Ok((data_type, logical_nulls(input.1.get().0)?.is_some()))
    }
}

pub struct SumMergeKernel(RunnableDSLFunction);

unsafe impl Send for SumMergeKernel {}
unsafe impl Sync for SumMergeKernel {}

impl Kernel for SumMergeKernel {
    type Key = PrimitiveType;
    type Input<'a> = (&'a mut DSLBuffer, &'a mut DSLBuffer);
    type Params = ();
    type Output = ();

    fn call(&self, input: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        self.0
            .run(&[DSLArgument::buffer(input.0), DSLArgument::buffer(input.1)])?;
        Ok(())
    }

    fn compile(input: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (left, right) = input;
        let mut context = DSLContext::new();
        let mut function = DSLFunction::new("merge_sum");
        let left_arg = function.add_arg(&mut context, DSLType::buffer_of(left.ty, "k"));
        let right_arg = function.add_arg(&mut context, DSLType::buffer_of(right.ty, "k"));

        function.add_body(
            DSLStmt::for_range(
                &mut context,
                DSLValue::u64(0).expr(),
                left_arg.expr().len()?,
                |index| {
                    let index = index.expr();
                    let left_value = left_arg.expr().at(&index)?;
                    let right_value = right_arg.expr().at(&index)?;
                    DSLStmt::set(
                        &left_arg,
                        &index,
                        &left_value.arith(DSLArithBinOp::Add, right_value)?,
                    )
                },
            )
            .unwrap(),
        );

        Ok(Self(compile(
            function,
            [
                DSLArgument::buffer(&mut DSLBuffer::empty_like(left)),
                DSLArgument::buffer(&mut DSLBuffer::empty_like(right)),
            ],
        )?))
    }

    fn get_key_for_input(
        input: &Self::Input<'_>,
        _params: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok(input.0.ty)
    }
}

static AGG_PROGRAM_CACHE: LazyLock<KernelCache<SumAggKernel>> = LazyLock::new(KernelCache::new);
static MERGE_PROGRAM_CACHE: LazyLock<KernelCache<SumMergeKernel>> = LazyLock::new(KernelCache::new);

pub struct SumAggregator {
    buffer: DSLBuffer,
}

impl SumAggregator {
    pub fn new(input_type: PrimitiveType) -> Result<Self, ArrowKernelError> {
        Ok(Self {
            buffer: DSLBuffer::new(sum_primitive_type(input_type)?, 0),
        })
    }
}

impl Aggregator for SumAggregator {
    fn output_type(types: &[&DataType]) -> Result<DataType, ArrowKernelError> {
        if types.len() != 1 {
            return Err(ArrowKernelError::ArgumentMismatch(
                "sum takes exactly one input type".to_string(),
            ));
        }

        let pt = sum_primitive_type(PrimitiveType::for_arrow_type(&logical_arrow_type(
            &types[0],
        )))?;
        Ok(pt.as_arrow_type())
    }

    fn create(types: &[&DataType]) -> Result<Box<Self>, ArrowKernelError> {
        let output_type = Self::output_type(types)?;
        Ok(Box::new(Self {
            buffer: DSLBuffer::new(PrimitiveType::for_arrow_type(&output_type), 0),
        }))
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.buffer.ensure_capacity(capacity);
    }

    fn ingest(
        &mut self,
        data: &[&dyn Array],
        tickets: &UInt64Array,
    ) -> Result<(), ArrowKernelError> {
        if data.len() != 1 {
            return Err(ArrowKernelError::ArgumentMismatch(
                "sum ingest takes exactly one input".to_string(),
            ));
        }
        self.ensure_capacity_for_tickets(tickets);
        AGG_PROGRAM_CACHE.get((&mut self.buffer, &data[0], tickets), ())
    }

    fn merge(&mut self, mut other: Self) -> Result<(), ArrowKernelError> {
        if self.buffer.len < other.buffer.len {
            self.ensure_capacity(other.buffer.len as usize);
        } else if other.buffer.len < self.buffer.len {
            other.ensure_capacity(self.buffer.len as usize);
        }
        MERGE_PROGRAM_CACHE.get((&mut self.buffer, &mut other.buffer), ())
    }

    fn finish(self: Box<Self>) -> Result<ArrayRef, ArrowKernelError> {
        Ok(self.buffer.into_array())
    }
}
