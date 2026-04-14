use std::sync::LazyLock;

use arrow_array::{Array, ArrayRef, Datum, UInt64Array};
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
    ArrowKernelError, Kernel, PrimitiveType,
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

pub struct SumAggKernel(RunnableDSLFunction);
unsafe impl Send for SumAggKernel {}
unsafe impl Sync for SumAggKernel {}

impl Kernel for SumAggKernel {
    type Key = DataType;

    type Input<'a> = (&'a mut DSLBuffer, &'a dyn Datum, &'a UInt64Array);

    type Params = ();

    type Output = ();

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, super::ArrowKernelError> {
        let (buf, data, tickets) = inp;

        self.0.run(&[
            DSLArgument::buffer(buf),
            DSLArgument::Datum(data),
            DSLArgument::datum(tickets),
        ])?;

        Ok(())
    }

    fn compile(
        inp: &Self::Input<'_>,
        _params: Self::Params,
    ) -> Result<Self, super::ArrowKernelError> {
        let (buf, data, tickets) = inp;
        let input_pt = PrimitiveType::for_arrow_type(data.get().0.data_type());
        let sum_pt = sum_primitive_type(input_pt)?;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("sum");
        let buf_arg = func.add_arg(&mut ctx, DSLType::buffer_of(sum_pt, "k"));
        let dat_arg = func.add_arg(&mut ctx, DSLType::array_like(*data, "n"));
        let tic_arg = func.add_arg(&mut ctx, DSLType::array_like(tickets, "n"));

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[tic_arg, dat_arg], |loop_vars| {
                let ticket = loop_vars[0].expr();
                let value = loop_vars[1].expr().primitive_cast(sum_pt)?;
                let cur = buf_arg.expr().at(&ticket)?;

                DSLStmt::set(&buf_arg, &ticket, &cur.arith(DSLArithBinOp::Add, value)?)
            })
            .unwrap(),
        );

        let func = compile(
            func,
            [
                DSLArgument::buffer(&mut DSLBuffer::empty_like(buf)),
                DSLArgument::Datum(*data),
                DSLArgument::datum(tickets),
            ],
        )?;

        Ok(Self(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        let dt = i.1.get().0.data_type().clone();
        let pt = PrimitiveType::for_arrow_type(&dt);
        sum_primitive_type(pt)?;
        Ok(dt)
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

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, super::ArrowKernelError> {
        let (buf1, buf2) = inp;
        self.0
            .run(&[DSLArgument::buffer(buf1), DSLArgument::buffer(buf2)])?;
        Ok(())
    }

    fn compile(
        inp: &Self::Input<'_>,
        _params: Self::Params,
    ) -> Result<Self, super::ArrowKernelError> {
        let (buf1, buf2) = inp;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("merge_sum");
        let buf1_arg = func.add_arg(&mut ctx, DSLType::buffer_of(buf1.ty, "k"));
        let buf2_arg = func.add_arg(&mut ctx, DSLType::buffer_of(buf2.ty, "k"));

        func.add_body(
            DSLStmt::for_range(
                &mut ctx,
                DSLValue::u64(0).expr(),
                buf1_arg.expr().len()?,
                |idx| {
                    let idx = idx.expr();
                    let lhs = buf1_arg.expr().at(&idx)?;
                    let rhs = buf2_arg.expr().at(&idx)?;
                    DSLStmt::set(&buf1_arg, &idx, &lhs.arith(DSLArithBinOp::Add, rhs)?)
                },
            )
            .unwrap(),
        );

        let func = compile(
            func,
            [
                DSLArgument::buffer(&mut DSLBuffer::empty_like(buf1)),
                DSLArgument::buffer(&mut DSLBuffer::empty_like(buf2)),
            ],
        )?;

        Ok(Self(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        Ok(i.0.ty)
    }
}

static AGG_PROGRAM_CACHE: LazyLock<KernelCache<SumAggKernel>> = LazyLock::new(KernelCache::new);
static MERGE_PROGRAM_CACHE: LazyLock<KernelCache<SumMergeKernel>> = LazyLock::new(KernelCache::new);

pub struct SumAggregator {
    buf: DSLBuffer,
}

impl SumAggregator {
    pub fn new(pt: PrimitiveType) -> Result<Self, ArrowKernelError> {
        Ok(Self {
            buf: DSLBuffer::new(sum_primitive_type(pt)?, 0),
        })
    }
}

impl Aggregator for SumAggregator {
    fn create(tys: &[&DataType]) -> Result<Box<Self>, ArrowKernelError> {
        if tys.len() != 1 {
            return Err(ArrowKernelError::ArgumentMismatch(
                "sum create takes exactly one input".to_string(),
            ));
        }
        Ok(Box::new(Self::new(PrimitiveType::for_arrow_type(&tys[0]))?))
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.buf.ensure_capacity(capacity);
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
        self.debug_assert_capacity_for_tickets(tickets, self.buf.len as usize);
        AGG_PROGRAM_CACHE.get((&mut self.buf, &data[0], tickets), ())?;
        Ok(())
    }

    fn merge(&mut self, mut other: Self) -> Result<(), ArrowKernelError> {
        if self.buf.len < other.buf.len {
            self.ensure_capacity(other.buf.len as usize);
        } else if other.buf.len < self.buf.len {
            other.ensure_capacity(self.buf.len as usize);
        }

        MERGE_PROGRAM_CACHE.get((&mut self.buf, &mut other.buf), ())?;
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<ArrayRef, ArrowKernelError> {
        Ok(self.buf.into_array())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{cast::AsArray, types::Float64Type, types::Int64Type, types::UInt64Type};
    use arrow_array::{Float32Array, Int32Array, StringArray};
    use itertools::Itertools;

    use super::*;

    fn write_i64_buffer(buf: &mut DSLBuffer, values: &[i64]) {
        assert_eq!(buf.ty, PrimitiveType::I64);
        let dst = bytemuck::cast_slice_mut::<u8, i64>(buf.buf.as_slice_mut());
        dst[..values.len()].copy_from_slice(values);
    }

    fn write_f64_buffer(buf: &mut DSLBuffer, values: &[f64]) {
        assert_eq!(buf.ty, PrimitiveType::F64);
        let dst = bytemuck::cast_slice_mut::<u8, f64>(buf.buf.as_slice_mut());
        dst[..values.len()].copy_from_slice(values);
    }

    #[test]
    fn test_sum_agg_kernel_updates_signed_groups() {
        let mut buf = DSLBuffer::new(PrimitiveType::I64, 4);
        let tickets = UInt64Array::from(vec![0, 1, 0, 1, 2, 0]);
        let data1 = Int32Array::from(vec![5, 6, -1, 4, 7, 3]);
        AGG_PROGRAM_CACHE
            .get((&mut buf, &data1, &tickets), ())
            .unwrap();
        let data2 = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        AGG_PROGRAM_CACHE
            .get((&mut buf, &data2, &tickets), ())
            .unwrap();

        let result = buf.into_array();
        let result = result
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(result, vec![17, 16, 12, 0]);
    }

    #[test]
    fn test_sum_agg_kernel_updates_float_groups() {
        let mut buf = DSLBuffer::new(PrimitiveType::F64, 3);
        let tickets = UInt64Array::from(vec![0, 1, 0, 2]);
        let data = Float32Array::from(vec![1.5, 2.25, 3.5, -4.0]);

        AGG_PROGRAM_CACHE
            .get((&mut buf, &data, &tickets), ())
            .unwrap();

        let result = buf.into_array();
        let result = result
            .as_primitive::<Float64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(result, vec![5.0, 2.25, -4.0]);
    }

    #[test]
    fn test_sum_merge_kernel_merges_groups() {
        let mut lhs = DSLBuffer::new(PrimitiveType::I64, 4);
        let mut rhs = DSLBuffer::new(PrimitiveType::I64, 4);
        write_i64_buffer(&mut lhs, &[1, -2, 0, 4]);
        write_i64_buffer(&mut rhs, &[5, 10, 3, -1]);

        MERGE_PROGRAM_CACHE.get((&mut lhs, &mut rhs), ()).unwrap();

        let result = lhs.into_array();
        let result = result
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(result, vec![6, 8, 3, 3]);
    }

    #[test]
    fn test_sum_aggregator_ingest_merge_and_finish() {
        let tickets1 = UInt64Array::from(vec![0, 1, 0, 3, 3]);
        let tickets2 = UInt64Array::from(vec![1, 1, 2]);
        let tickets3 = UInt64Array::from(vec![0, 4, 4, 4]);
        let data1 = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let data2 = Int32Array::from(vec![5, 6, 7]);
        let data3 = Int32Array::from(vec![2, 4, 6, 8]);

        let mut agg1 = SumAggregator::new(PrimitiveType::I32).unwrap();
        agg1.ensure_capacity(4);
        agg1.ingest(&[&data1], &tickets1).unwrap();
        agg1.ingest(&[&data2], &tickets2).unwrap();

        let mut agg2 = SumAggregator::new(PrimitiveType::I32).unwrap();
        agg2.ensure_capacity(5);
        agg2.ingest(&[&data3], &tickets3).unwrap();

        agg1.merge(agg2).unwrap();

        let result = Box::new(agg1).finish().unwrap();
        let result = result
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(result, vec![42, 31, 7, 90, 18]);
    }

    #[test]
    fn test_sum_aggregator_finish_unsigned_widens_to_u64() {
        let tickets = UInt64Array::from(vec![0, 1, 0]);
        let data = arrow_array::UInt32Array::from(vec![1, 2, 3]);
        let mut agg = SumAggregator::new(PrimitiveType::U32).unwrap();
        agg.ensure_capacity(2);
        agg.ingest(&[&data], &tickets).unwrap();

        let result = Box::new(agg).finish().unwrap();
        let result = result
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(result, vec![4, 2]);
    }

    #[test]
    fn test_sum_kernel_rejects_non_numeric_types() {
        let mut buf = DSLBuffer::new(PrimitiveType::U64, 2);
        let tickets = UInt64Array::from(vec![0, 1]);
        let data = StringArray::from(vec!["a", "b"]);

        let err = AGG_PROGRAM_CACHE
            .get((&mut buf, &data, &tickets), ())
            .unwrap_err();
        match err {
            ArrowKernelError::UnsupportedArguments(msg) => {
                assert!(msg.contains("sum only supports numeric types"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn ensure_capacity_should_track_elements() {
        let mut agg = SumAggregator::new(PrimitiveType::F32).unwrap();
        agg.ensure_capacity(3);
        agg.ensure_capacity(5);

        assert_eq!(agg.buf.len, 5);
        assert_eq!(agg.buf.ty, PrimitiveType::F64);
    }

    #[test]
    fn merge_kernel_supports_float_buffers() {
        let mut lhs = DSLBuffer::new(PrimitiveType::F64, 2);
        let mut rhs = DSLBuffer::new(PrimitiveType::F64, 2);
        write_f64_buffer(&mut lhs, &[1.25, -2.0]);
        write_f64_buffer(&mut rhs, &[0.75, 3.0]);

        MERGE_PROGRAM_CACHE.get((&mut lhs, &mut rhs), ()).unwrap();

        let result = lhs.into_array();
        let result = result
            .as_primitive::<Float64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(result, vec![2.0, 1.0]);
    }
}
