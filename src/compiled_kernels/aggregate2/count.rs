use std::sync::LazyLock;

use arrow_array::{Array, ArrayRef, UInt64Array};
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

pub struct CountAggKernel(RunnableDSLFunction);
unsafe impl Send for CountAggKernel {}
unsafe impl Sync for CountAggKernel {}

impl Kernel for CountAggKernel {
    type Key = ();

    type Input<'a> = (&'a mut DSLBuffer, &'a UInt64Array);

    type Params = ();

    type Output = ();

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, super::ArrowKernelError> {
        let (buf, tickets) = inp;

        self.0
            .run(&[DSLArgument::buffer(buf), DSLArgument::datum(tickets)])?;

        Ok(())
    }

    fn compile(
        inp: &Self::Input<'_>,
        _params: Self::Params,
    ) -> Result<Self, super::ArrowKernelError> {
        let (buf, tickets) = inp;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("count");
        let buf_arg = func.add_arg(&mut ctx, DSLType::buffer_of(PrimitiveType::U64, "k"));
        let tic_arg = func.add_arg(&mut ctx, DSLType::array_like(tickets, "n"));

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[tic_arg], |loop_vars| {
                let ticket = loop_vars[0].expr();
                let cur = buf_arg.expr().at(&ticket)?;

                DSLStmt::set(
                    &buf_arg,
                    &ticket,
                    &cur.arith(
                        DSLArithBinOp::Add,
                        DSLValue::u64(1).expr().primitive_cast(PrimitiveType::U64)?,
                    )?,
                )
            })
            .unwrap(),
        );

        let func = compile(
            func,
            [
                DSLArgument::buffer(&mut DSLBuffer::empty_like(buf)),
                DSLArgument::datum(tickets),
            ],
        )?;

        Ok(Self(func))
    }

    fn get_key_for_input(
        _i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        Ok(())
    }
}

pub struct CountMergeKernel(RunnableDSLFunction);
unsafe impl Send for CountMergeKernel {}
unsafe impl Sync for CountMergeKernel {}

impl Kernel for CountMergeKernel {
    type Key = ();

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
        let mut func = DSLFunction::new("merge_count");
        let buf1_arg = func.add_arg(&mut ctx, DSLType::buffer_of(PrimitiveType::U64, "k"));
        let buf2_arg = func.add_arg(&mut ctx, DSLType::buffer_of(PrimitiveType::U64, "k"));

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
        _i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        Ok(())
    }
}

static AGG_PROGRAM_CACHE: LazyLock<KernelCache<CountAggKernel>> = LazyLock::new(KernelCache::new);
static MERGE_PROGRAM_CACHE: LazyLock<KernelCache<CountMergeKernel>> =
    LazyLock::new(KernelCache::new);

pub struct CountAggregator {
    buf: DSLBuffer,
}

impl Aggregator for CountAggregator {
    fn create(_tys: &[&DataType]) -> Result<Box<Self>, ArrowKernelError> {
        return Ok(Box::new(Self {
            buf: DSLBuffer::new(PrimitiveType::U64, 0),
        }));
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.buf.ensure_capacity(capacity);
    }

    fn ingest(
        &mut self,
        _data: &[&dyn Array],
        tickets: &UInt64Array,
    ) -> Result<(), ArrowKernelError> {
        self.debug_assert_capacity_for_tickets(tickets, self.buf.len as usize);
        AGG_PROGRAM_CACHE.get((&mut self.buf, tickets), ())?;
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
    use arrow_array::{cast::AsArray, Int32Array};
    use itertools::Itertools;

    use super::*;

    fn write_u64_buffer(buf: &mut DSLBuffer, values: &[u64]) {
        assert_eq!(buf.ty, PrimitiveType::U64);
        let dst = bytemuck::cast_slice_mut::<u8, u64>(buf.buf.as_slice_mut());
        dst[..values.len()].copy_from_slice(values);
    }

    #[test]
    fn test_count_agg_kernel_updates_groups() {
        let mut buf = DSLBuffer::new(PrimitiveType::U64, 4);
        let tickets = UInt64Array::from(vec![0, 1, 0, 1, 0, 1, 3]);

        AGG_PROGRAM_CACHE.get((&mut buf, &tickets), ()).unwrap();

        let result = buf.into_array();
        let result = result
            .as_primitive::<arrow_array::types::UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(result, vec![3, 3, 0, 1]);
    }

    #[test]
    fn test_count_merge_kernel_merges_groups() {
        let mut lhs = DSLBuffer::new(PrimitiveType::U64, 4);
        let mut rhs = DSLBuffer::new(PrimitiveType::U64, 4);
        write_u64_buffer(&mut lhs, &[1, 2, 0, 4]);
        write_u64_buffer(&mut rhs, &[5, 0, 3, 1]);

        MERGE_PROGRAM_CACHE.get((&mut lhs, &mut rhs), ()).unwrap();

        let result = lhs.into_array();
        let result = result
            .as_primitive::<arrow_array::types::UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(result, vec![6, 2, 3, 5]);
    }

    #[test]
    fn test_count_aggregator_ingest_merge_and_finish() {
        let tickets1 = UInt64Array::from(vec![0, 1, 0, 3, 3]);
        let tickets2 = UInt64Array::from(vec![1, 1, 2]);
        let tickets3 = UInt64Array::from(vec![0, 4, 4, 4]);
        let data = Int32Array::from(vec![10, 20, 30, 40, 50]);

        let mut agg1 = CountAggregator::create(&[]).unwrap();
        agg1.ensure_capacity(4);
        agg1.ingest(&[&data], &tickets1).unwrap();
        agg1.ingest(&[&data], &tickets2).unwrap();

        let mut agg2 = CountAggregator::create(&[]).unwrap();
        agg2.ensure_capacity(5);
        agg2.ingest(&[&data], &tickets3).unwrap();

        agg1.merge(*agg2).unwrap();

        let result = Box::new(agg1).finish().unwrap();
        let result = result
            .as_primitive::<arrow_array::types::UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(result, vec![3, 3, 1, 2, 3]);
    }

    #[test]
    fn ensure_capacity_should_track_elements() {
        let mut agg = CountAggregator::create(&[]).unwrap();
        agg.ensure_capacity(3);
        agg.ensure_capacity(5);

        assert_eq!(agg.buf.len, 5);
    }
}
