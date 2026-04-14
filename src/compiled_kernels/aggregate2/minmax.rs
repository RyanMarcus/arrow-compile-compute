use std::{ffi::c_void, ptr::null, sync::LazyLock};

use arrow_array::{Array, ArrayRef, Datum, UInt64Array};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        aggregate2::Aggregator,
        dsl2::{
            compile, DSLArgument, DSLBitwiseBinOp, DSLBuffer, DSLComparison, DSLContext,
            DSLFunction, DSLStmt, DSLType, DSLValue, RunnableDSLFunction,
        },
        llvm_utils::StringSaver,
        KernelCache,
    },
    ArrowKernelError, Kernel, PrimitiveType,
};

pub struct MinMaxAggKernel(RunnableDSLFunction);
unsafe impl Send for MinMaxAggKernel {}
unsafe impl Sync for MinMaxAggKernel {}

impl Kernel for MinMaxAggKernel {
    type Key = (DataType, bool);

    type Input<'a> = (
        &'a mut DSLBuffer,
        &'a mut DSLBuffer,
        &'a dyn Datum,
        &'a UInt64Array,
        &'a mut StringSaver,
    );

    type Params = bool;

    type Output = ();

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, super::ArrowKernelError> {
        let (used, buf, data, tickets, ss) = inp;
        self.0.run(&[
            DSLArgument::buffer(used),
            DSLArgument::buffer(buf),
            DSLArgument::Datum(data),
            DSLArgument::Datum(tickets),
            DSLArgument::string_saver(ss),
        ])?;
        Ok(())
    }

    fn compile(
        inp: &Self::Input<'_>,
        is_min: Self::Params,
    ) -> Result<Self, super::ArrowKernelError> {
        let (used, buf, data, tickets, _ss) = inp;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("minmax");
        let use_arg = func.add_arg(&mut ctx, DSLType::buffer_of(PrimitiveType::U8, "k"));
        let buf_arg = func.add_arg(
            &mut ctx,
            DSLType::buffer_of(PrimitiveType::for_arrow_type(data.get().0.data_type()), "k"),
        );
        let dat_arg = func.add_arg(&mut ctx, DSLType::array_like(*data, "n"));
        let tic_arg = func.add_arg(&mut ctx, DSLType::array_like(tickets, "n"));
        let ss_arg = func.add_arg(&mut ctx, DSLType::StringSaver);

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[tic_arg, dat_arg], |loop_vars| {
                let ticket = loop_vars[0].expr();
                let data = loop_vars[1].expr();
                let used = use_arg.expr().at(&ticket)?;
                let cur = buf_arg.expr().at(&ticket)?;

                DSLStmt::cond_else(
                    used.cast_to_bool()?,
                    DSLStmt::cond(
                        data.cmp(
                            &cur,
                            if is_min {
                                DSLComparison::Lt
                            } else {
                                DSLComparison::Gt
                            },
                        )?,
                        DSLStmt::set_with_saver(&buf_arg, &ticket, &data, &ss_arg)?,
                    )?,
                    vec![
                        DSLStmt::set_with_saver(&buf_arg, &ticket, &data, &ss_arg)?,
                        DSLStmt::set(
                            &use_arg,
                            &ticket,
                            &DSLValue::u8(1).expr().primitive_cast(PrimitiveType::U8)?,
                        )?,
                    ],
                )
            })
            .unwrap(),
        );

        let func = compile(
            func,
            [
                DSLArgument::buffer(&mut DSLBuffer::empty_like(used)),
                DSLArgument::buffer(&mut DSLBuffer::empty_like(buf)),
                DSLArgument::Datum(*data),
                DSLArgument::Datum(tickets),
                DSLArgument::StringSaver(null::<StringSaver>() as *mut c_void),
            ],
        )?;

        Ok(Self(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        Ok((i.2.get().0.data_type().clone(), *p))
    }
}

pub struct MinMaxMergeKernel(RunnableDSLFunction);
unsafe impl Send for MinMaxMergeKernel {}
unsafe impl Sync for MinMaxMergeKernel {}

impl Kernel for MinMaxMergeKernel {
    type Key = (DataType, bool);

    type Input<'a> = (
        &'a mut DSLBuffer,
        &'a mut DSLBuffer,
        &'a mut DSLBuffer,
        &'a mut DSLBuffer,
    );

    type Params = bool;

    type Output = ();

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, super::ArrowKernelError> {
        let (used1, buf1, used2, buf2) = inp;
        self.0.run(&[
            DSLArgument::buffer(used1),
            DSLArgument::buffer(buf1),
            DSLArgument::buffer(used2),
            DSLArgument::buffer(buf2),
        ])?;
        Ok(())
    }

    fn compile(
        inp: &Self::Input<'_>,
        is_min: Self::Params,
    ) -> Result<Self, super::ArrowKernelError> {
        let (used1, buf1, used2, buf2) = inp;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("merge_minmax");
        let use1_arg = func.add_arg(&mut ctx, DSLType::buffer_of(PrimitiveType::U8, "k"));
        let buf1_arg = func.add_arg(&mut ctx, DSLType::buffer_of(buf1.ty, "k"));
        let use2_arg = func.add_arg(&mut ctx, DSLType::buffer_of(PrimitiveType::U8, "k"));
        let buf2_arg = func.add_arg(&mut ctx, DSLType::buffer_of(buf2.ty, "k"));

        func.add_body(
            DSLStmt::for_range(
                &mut ctx,
                DSLValue::u64(0).expr(),
                use1_arg.expr().len()?,
                |idx| {
                    let idx = idx.expr();
                    let use1 = use1_arg.expr().at(&idx)?.cast_to_bool()?;
                    let buf1 = buf1_arg.expr().at(&idx)?;
                    let use2 = use2_arg.expr().at(&idx)?.cast_to_bool()?;
                    let buf2 = buf2_arg.expr().at(&idx)?;

                    let only2 = use2.bitwise(DSLBitwiseBinOp::And, use1.bit_not()?)?;
                    let both = use1.bitwise(DSLBitwiseBinOp::And, use2.clone())?;
                    let new_val = if is_min {
                        buf1.cmp(&buf2, DSLComparison::Lt)?
                            .select(buf1, buf2.clone())?
                    } else {
                        buf1.cmp(&buf2, DSLComparison::Gt)?
                            .select(buf1, buf2.clone())?
                    };

                    DSLStmt::cond_else(
                        only2,
                        vec![
                            DSLStmt::set(&use1_arg, &idx, &use2)?,
                            DSLStmt::set(&buf1_arg, &idx, &buf2)?,
                        ],
                        DSLStmt::cond(both, DSLStmt::set(&buf1_arg, &idx, &new_val)?)?,
                    )
                },
            )
            .unwrap(),
        );

        let func = compile(
            func,
            [
                DSLArgument::buffer(&mut DSLBuffer::empty_like(used1)),
                DSLArgument::buffer(&mut DSLBuffer::empty_like(buf1)),
                DSLArgument::buffer(&mut DSLBuffer::empty_like(used2)),
                DSLArgument::buffer(&mut DSLBuffer::empty_like(buf2)),
            ],
        )?;

        Ok(Self(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        Ok((i.1.ty.as_arrow_type(), *p))
    }
}

static AGG_PROGRAM_CACHE: LazyLock<KernelCache<MinMaxAggKernel>> = LazyLock::new(KernelCache::new);
static MERGE_PROGRAM_CACHE: LazyLock<KernelCache<MinMaxMergeKernel>> =
    LazyLock::new(KernelCache::new);

pub struct MinMaxAggregator<const IS_MIN: bool> {
    used: DSLBuffer,
    buf: DSLBuffer,
    ss: Vec<StringSaver>,
}

impl<const IS_MIN: bool> MinMaxAggregator<IS_MIN> {
    pub fn new(pt: PrimitiveType) -> Self {
        Self {
            used: DSLBuffer::new(PrimitiveType::U8, 0),
            buf: DSLBuffer::new(pt, 0),
            ss: Vec::new(),
        }
    }
}

impl<const IS_MIN: bool> Aggregator for MinMaxAggregator<IS_MIN> {
    fn create(tys: &[&DataType]) -> Result<Box<Self>, ArrowKernelError> {
        if tys.len() != 1 {
            return Err(ArrowKernelError::ArgumentMismatch(
                "minmax agg must have exactly one input type".to_string(),
            ));
        }

        Ok(Box::new(Self::new(PrimitiveType::for_arrow_type(&tys[0]))))
    }
    fn ensure_capacity(&mut self, capacity: usize) {
        self.used.ensure_capacity(capacity);
        self.buf.ensure_capacity(capacity);
    }

    fn ingest(
        &mut self,
        data: &[&dyn Array],
        tickets: &UInt64Array,
    ) -> Result<(), ArrowKernelError> {
        if data.len() != 1 {
            return Err(ArrowKernelError::ArgumentMismatch(
                "minmax agg ingest must have exactly one input array".to_string(),
            ));
        }
        self.debug_assert_capacity_for_tickets(tickets, self.buf.len as usize);
        if self.ss.is_empty() {
            self.ss.push(StringSaver::default());
        }
        AGG_PROGRAM_CACHE.get(
            (
                &mut self.used,
                &mut self.buf,
                &data[0],
                &tickets,
                self.ss.last_mut().unwrap(),
            ),
            IS_MIN,
        )?;
        Ok(())
    }

    fn merge(&mut self, mut other: Self) -> Result<(), ArrowKernelError> {
        if self.buf.len < other.buf.len {
            self.ensure_capacity(other.buf.len as usize);
        } else if other.buf.len < self.buf.len {
            other.ensure_capacity(self.buf.len as usize);
        }

        MERGE_PROGRAM_CACHE.get(
            (
                &mut self.used,
                &mut self.buf,
                &mut other.used,
                &mut other.buf,
            ),
            IS_MIN,
        )?;
        self.ss.extend(other.ss);
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<ArrayRef, ArrowKernelError> {
        Ok(self.buf.into_array())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        cast::AsArray,
        types::{Int32Type, UInt8Type},
        Int32Array, StringArray, UInt64Array,
    };
    use itertools::Itertools;

    use super::*;

    fn write_u8_buffer(buf: &mut DSLBuffer, values: &[u8]) {
        assert_eq!(buf.ty, PrimitiveType::U8);
        buf.buf.as_slice_mut()[..values.len()].copy_from_slice(values);
    }

    fn write_i32_buffer(buf: &mut DSLBuffer, values: &[i32]) {
        assert_eq!(buf.ty, PrimitiveType::I32);
        let dst = bytemuck::cast_slice_mut::<u8, i32>(buf.buf.as_slice_mut());
        dst[..values.len()].copy_from_slice(values);
    }

    #[test]
    fn test_minmax_agg_kernel_updates_numeric_groups() {
        let mut used = DSLBuffer::new(PrimitiveType::U8, 3);
        let mut buf = DSLBuffer::new(PrimitiveType::I32, 3);
        let mut ss = StringSaver::default();
        let tickets = UInt64Array::from(vec![0, 1, 0, 1, 0, 1]);
        let data1 = Int32Array::from(vec![1, -2, 3, 4, 5, 6]);
        AGG_PROGRAM_CACHE
            .get((&mut used, &mut buf, &data1, &tickets, &mut ss), true)
            .unwrap();
        let data2 = Int32Array::from(vec![0, 2, 3000, 4, 50, 60]);
        AGG_PROGRAM_CACHE
            .get((&mut used, &mut buf, &data2, &tickets, &mut ss), true)
            .unwrap();

        let used = used.into_array();
        let used = used
            .as_primitive::<UInt8Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        let buf = buf.into_array();
        let buf = buf
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(used, vec![1, 1, 0]);
        assert_eq!(buf, vec![0, -2, 0]);
    }

    #[test]
    fn test_minmax_merge_kernel_merges_numeric_groups() {
        let mut used1 = DSLBuffer::new(PrimitiveType::U8, 2);
        let mut buf1 = DSLBuffer::new(PrimitiveType::I32, 2);
        let mut used2 = DSLBuffer::new(PrimitiveType::U8, 2);
        let mut buf2 = DSLBuffer::new(PrimitiveType::I32, 2);
        write_u8_buffer(&mut used1, &[1, 1]);
        write_i32_buffer(&mut buf1, &[5, 6]);
        write_u8_buffer(&mut used2, &[1, 1]);
        write_i32_buffer(&mut buf2, &[3000, 60]);

        MERGE_PROGRAM_CACHE
            .get((&mut used1, &mut buf1, &mut used2, &mut buf2), false)
            .unwrap();

        let used1 = used1.into_array();
        let used1 = used1
            .as_primitive::<UInt8Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        let buf1 = buf1.into_array();
        let buf1 = buf1
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(used1, vec![1, 1]);
        assert_eq!(buf1, vec![3000, 60]);
    }

    #[test]
    fn test_minmax_agg_kernel_updates_string_groups() {
        let mut used = DSLBuffer::new(PrimitiveType::U8, 2);
        let mut buf = DSLBuffer::new(PrimitiveType::P64x2, 2);
        let mut ss = StringSaver::default();
        let tickets = UInt64Array::from(vec![0, 1, 0, 1, 0, 1]);
        let data1 = StringArray::from(vec!["apple", "banana", "cherry", "date", "elder", "fig"]);
        AGG_PROGRAM_CACHE
            .get((&mut used, &mut buf, &data1, &tickets, &mut ss), true)
            .unwrap();
        let data2 = StringArray::from(vec!["zeta", "gamma", "luma", "puma", "alpha", "mango"]);
        AGG_PROGRAM_CACHE
            .get((&mut used, &mut buf, &data2, &tickets, &mut ss), true)
            .unwrap();

        let used = used.into_array();
        let used = used
            .as_primitive::<UInt8Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        let buf = buf.into_array();
        let buf = buf
            .as_binary::<i32>()
            .iter()
            .map(|value| std::str::from_utf8(value.unwrap()).unwrap())
            .collect_vec();

        assert_eq!(used, vec![1, 1]);
        assert_eq!(buf, vec!["alpha", "banana"]);
    }

    #[test]
    fn test_minmax_string_winners_survive_input_drop() {
        let mut agg = MinMaxAggregator::<false>::new(PrimitiveType::P64x2);
        agg.ensure_capacity(128);

        let mut expected = Vec::with_capacity(128);
        for ticket in 0..128 {
            expected.push(format!("group-{ticket:03}-winner-{}", "a".repeat(512)));
        }

        let tickets = UInt64Array::from(
            (0..128u64)
                .flat_map(|ticket| [ticket, ticket])
                .collect_vec(),
        );

        {
            let losers = StringArray::from(
                (0..128)
                    .flat_map(|ticket| {
                        [
                            format!("group-{ticket:03}-loser-{}", "z".repeat(512)),
                            expected[ticket].clone(),
                        ]
                    })
                    .collect_vec(),
            );
            agg.ingest(&[&losers], &tickets).unwrap();
            std::mem::drop(losers);
        }

        let mut churn = Vec::with_capacity(20_000);
        for i in 0..20_000 {
            churn.push(format!("churn-{i:05}-{}", "q".repeat(512)).into_bytes());
        }

        let result = Box::new(agg).finish().unwrap();
        let result = result
            .as_binary::<i32>()
            .iter()
            .map(|value| std::str::from_utf8(value.unwrap()).unwrap().to_string())
            .collect_vec();

        assert_eq!(result, expected);
    }

    #[test]
    fn ensure_capacity_should_track_elements() {
        let mut agg = MinMaxAggregator::<true>::new(PrimitiveType::I32);
        agg.ensure_capacity(3);

        agg.ensure_capacity(4);

        assert_eq!(agg.used.len, 4);
        assert_eq!(agg.buf.len, 4);
    }

    #[test]
    fn test_minmax_string_merge_survives_rhs_drop() {
        let tickets1 = UInt64Array::from(vec![0, 2]);
        let tickets2 = UInt64Array::from(vec![1, 3]);
        let data1 = StringArray::from(vec![
            format!("lhs-{}", "b".repeat(256)),
            format!("lhs-{}", "d".repeat(256)),
        ]);
        let expected_rhs = vec![
            format!("rhs-{}", "a".repeat(256)),
            format!("rhs-{}", "c".repeat(256)),
        ];
        let data2 = StringArray::from(expected_rhs.clone());

        let mut agg1 = MinMaxAggregator::<true>::new(PrimitiveType::P64x2);
        agg1.ensure_capacity(4);
        agg1.ingest(&[&data1], &tickets1).unwrap();

        let mut agg2 = MinMaxAggregator::<true>::new(PrimitiveType::P64x2);
        agg2.ensure_capacity(4);
        agg2.ingest(&[&data2], &tickets2).unwrap();

        agg1.merge(agg2).unwrap();

        let mut churn = Vec::with_capacity(10_000);
        for i in 0..10_000 {
            churn.push(format!("churn-{i}-{}", "q".repeat(256)).into_bytes());
        }

        let result = Box::new(agg1).finish().unwrap();
        let result = result
            .as_binary::<i32>()
            .iter()
            .map(|value| value.map(|v| std::str::from_utf8(v).unwrap().to_string()))
            .collect_vec();

        assert_eq!(
            result,
            vec![
                Some(format!("lhs-{}", "b".repeat(256))),
                Some(expected_rhs[0].clone()),
                Some(format!("lhs-{}", "d".repeat(256))),
                Some(expected_rhs[1].clone()),
            ]
        );
    }
}
