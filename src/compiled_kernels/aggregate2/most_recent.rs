use std::{ffi::c_void, ptr::null, sync::LazyLock};

use arrow_array::{builder::BinaryBuilder, make_array, Array, ArrayRef, Datum};
use arrow_buffer::NullBuffer;
use arrow_data::ArrayData;
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        aggregate2::Aggregator,
        dsl2::{
            compile, DSLArgument, DSLBitwiseBinOp, DSLBuffer, DSLContext, DSLFunction, DSLStmt,
            DSLType, DSLValue, RunnableDSLFunction,
        },
        llvm_utils::StringSaver,
        KernelCache,
    },
    ArrowKernelError, Kernel, PrimitiveType,
};

pub struct MostRecentAggKernel(RunnableDSLFunction);
unsafe impl Send for MostRecentAggKernel {}
unsafe impl Sync for MostRecentAggKernel {}

impl Kernel for MostRecentAggKernel {
    type Key = arrow_schema::DataType;

    type Input<'a> = (
        &'a mut DSLBuffer,
        &'a mut DSLBuffer,
        &'a dyn Datum,
        &'a arrow_array::UInt64Array,
        &'a mut StringSaver,
    );

    type Params = ();

    type Output = ();

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, super::ArrowKernelError> {
        let (used, buf, data, tickets, ss) = inp;
        self.0.run(&[
            DSLArgument::buffer(used),
            DSLArgument::buffer(buf),
            DSLArgument::Datum(data),
            DSLArgument::datum(tickets),
            DSLArgument::string_saver(ss),
        ])?;
        Ok(())
    }

    fn compile(
        inp: &Self::Input<'_>,
        _params: Self::Params,
    ) -> Result<Self, super::ArrowKernelError> {
        let (used, buf, data, tickets, _ss) = inp;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("most_recent");
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

                Ok(vec![
                    DSLStmt::set_with_saver(&buf_arg, &ticket, &data, &ss_arg)?,
                    DSLStmt::set(
                        &use_arg,
                        &ticket,
                        &DSLValue::u8(1).expr().primitive_cast(PrimitiveType::U8)?,
                    )?,
                ])
            })
            .unwrap(),
        );

        let func = compile(
            func,
            [
                DSLArgument::buffer(&mut DSLBuffer::empty_like(used)),
                DSLArgument::buffer(&mut DSLBuffer::empty_like(buf)),
                DSLArgument::Datum(*data),
                DSLArgument::datum(tickets),
                DSLArgument::StringSaver(null::<StringSaver>() as *mut c_void),
            ],
        )?;

        Ok(Self(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        Ok(i.2.get().0.data_type().clone())
    }
}

pub struct MostRecentMergeKernel(RunnableDSLFunction);
unsafe impl Send for MostRecentMergeKernel {}
unsafe impl Sync for MostRecentMergeKernel {}

impl Kernel for MostRecentMergeKernel {
    type Key = arrow_schema::DataType;

    type Input<'a> = (
        &'a mut DSLBuffer,
        &'a mut DSLBuffer,
        &'a mut DSLBuffer,
        &'a mut DSLBuffer,
    );

    type Params = ();

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
        _params: Self::Params,
    ) -> Result<Self, super::ArrowKernelError> {
        let (used1, buf1, used2, buf2) = inp;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("merge_most_recent");
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
                    let use2 = use2_arg.expr().at(&idx)?.cast_to_bool()?;
                    let only2 = use2.bitwise(DSLBitwiseBinOp::And, use1.bit_not()?)?;
                    let buf2 = buf2_arg.expr().at(&idx)?;

                    DSLStmt::cond(
                        only2,
                        vec![
                            DSLStmt::set(&use1_arg, &idx, &use2)?,
                            DSLStmt::set(&buf1_arg, &idx, &buf2)?,
                        ],
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
        _p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        Ok(i.1.ty.as_arrow_type())
    }
}

static AGG_PROGRAM_CACHE: LazyLock<KernelCache<MostRecentAggKernel>> =
    LazyLock::new(KernelCache::new);
static MERGE_PROGRAM_CACHE: LazyLock<KernelCache<MostRecentMergeKernel>> =
    LazyLock::new(KernelCache::new);

pub struct MostRecentAggregator {
    used: DSLBuffer,
    buf: DSLBuffer,
    ss: Vec<StringSaver>,
}

impl MostRecentAggregator {
    pub fn new(pt: PrimitiveType) -> Self {
        Self {
            used: DSLBuffer::new(PrimitiveType::U8, 0),
            buf: DSLBuffer::new(pt, 0),
            ss: Vec::new(),
        }
    }
}

impl Aggregator for MostRecentAggregator {
    fn create(tys: &[&DataType]) -> Result<Box<Self>, ArrowKernelError> {
        if tys.len() != 1 {
            return Err(ArrowKernelError::ArgumentMismatch(
                "most recent agg create takes exactly one input".to_string(),
            ));
        }
        Ok(Box::new(Self::new(PrimitiveType::for_arrow_type(tys[0]))))
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.used.ensure_capacity(capacity);
        self.buf.ensure_capacity(capacity);
    }

    fn ingest(
        &mut self,
        data: &[&dyn Array],
        tickets: &arrow_array::UInt64Array,
    ) -> Result<(), ArrowKernelError> {
        if data.len() != 1 {
            return Err(ArrowKernelError::ArgumentMismatch(
                "most recent agg ingest takes exactly one input".to_string(),
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
                tickets,
                self.ss.last_mut().unwrap(),
            ),
            (),
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
            (),
        )?;
        self.ss.extend(other.ss);
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<ArrayRef, ArrowKernelError> {
        let MostRecentAggregator {
            used,
            mut buf,
            ss: _,
        } = *self;
        let len = buf.len as usize;
        let null_map = NullBuffer::from_iter(used.buf.as_slice()[..len].iter().map(|x| *x > 0));

        match buf.ty {
            PrimitiveType::P64x2 => {
                let mut builder = BinaryBuilder::new();
                let values = bytemuck::cast_slice::<u8, u128>(&buf.buf.as_slice()[..len * 16]);
                for (idx, &raw) in values.iter().enumerate() {
                    if used.buf.as_slice()[idx] == 0 {
                        builder.append_null();
                        continue;
                    }
                    let start_ptr = raw as u64 as *const u8;
                    let end_ptr = (raw >> 64) as u64 as *const u8;
                    let value = unsafe {
                        let bytes = end_ptr.offset_from_unsigned(start_ptr);
                        std::slice::from_raw_parts(start_ptr, bytes)
                    };
                    builder.append_value(value);
                }
                Ok(std::sync::Arc::new(builder.finish()))
            }
            PrimitiveType::List(item_type, _) => {
                buf.buf.truncate(buf.ty.width() * len);
                let child_type = PrimitiveType::from(item_type);
                let child_data = ArrayData::builder(child_type.as_arrow_type())
                    .len(buf.buf.len() / child_type.width())
                    .add_buffer(buf.buf.into())
                    .align_buffers(true)
                    .build()
                    .unwrap();
                let ad = ArrayData::builder(buf.ty.as_arrow_type())
                    .len(len)
                    .nulls(Some(null_map))
                    .add_child_data(child_data)
                    .build()
                    .unwrap();
                Ok(make_array(ad))
            }
            _ => {
                buf.buf.truncate(buf.ty.width() * len);
                let ad = ArrayData::builder(buf.ty.as_arrow_type())
                    .len(len)
                    .add_buffer(buf.buf.into())
                    .align_buffers(true)
                    .nulls(Some(null_map))
                    .build()
                    .unwrap();
                Ok(make_array(ad))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray,
        types::{Float32Type, Int32Type, UInt8Type},
        Array, FixedSizeListArray, Float32Array, Int32Array, StringArray,
    };
    use arrow_data::ArrayData;
    use itertools::Itertools;

    use crate::ListItemType;

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
    fn test_most_recent_agg_kernel_updates_numeric_groups() {
        let mut used = DSLBuffer::new(PrimitiveType::U8, 3);
        let mut buf = DSLBuffer::new(PrimitiveType::I32, 3);
        let mut ss = StringSaver::default();
        let tickets = arrow_array::UInt64Array::from(vec![0, 1, 0, 1, 0, 1]);
        let data = Int32Array::from(vec![5, 6, 7, 8, 1, 2]);

        AGG_PROGRAM_CACHE
            .get((&mut used, &mut buf, &data, &tickets, &mut ss), ())
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
        assert_eq!(buf, vec![1, 2, 0]);
    }

    #[test]
    fn test_most_recent_agg_kernel_updates_string_groups() {
        let mut used = DSLBuffer::new(PrimitiveType::U8, 2);
        let mut buf = DSLBuffer::new(PrimitiveType::P64x2, 2);
        let mut ss = StringSaver::default();
        let tickets = arrow_array::UInt64Array::from(vec![0, 1, 0, 1, 0, 1]);
        let data = StringArray::from(vec!["a", "b", "c", "d", "e", "f"]);

        AGG_PROGRAM_CACHE
            .get((&mut used, &mut buf, &data, &tickets, &mut ss), ())
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
        assert_eq!(buf, vec!["e", "f"]);
    }

    #[test]
    fn test_most_recent_merge_kernel_prefers_lhs_when_present() {
        let mut used1 = DSLBuffer::new(PrimitiveType::U8, 4);
        let mut buf1 = DSLBuffer::new(PrimitiveType::I32, 4);
        let mut used2 = DSLBuffer::new(PrimitiveType::U8, 4);
        let mut buf2 = DSLBuffer::new(PrimitiveType::I32, 4);

        write_u8_buffer(&mut used1, &[1, 0, 1, 0]);
        write_i32_buffer(&mut buf1, &[10, 0, 30, 0]);
        write_u8_buffer(&mut used2, &[0, 1, 1, 1]);
        write_i32_buffer(&mut buf2, &[99, 20, 300, 40]);

        MERGE_PROGRAM_CACHE
            .get((&mut used1, &mut buf1, &mut used2, &mut buf2), ())
            .unwrap();

        let used = used1.into_array();
        let used = used
            .as_primitive::<UInt8Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        let buf = buf1.into_array();
        let buf = buf
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(used, vec![1, 1, 1, 1]);
        assert_eq!(buf, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_most_recent_aggregator_string_merge_survives_rhs_drop() {
        let tickets1 = arrow_array::UInt64Array::from(vec![0, 2]);
        let tickets2 = arrow_array::UInt64Array::from(vec![1, 3]);
        let data1 = StringArray::from(vec![
            format!("lhs-{}", "a".repeat(256)),
            format!("lhs-{}", "b".repeat(256)),
        ]);
        let expected_rhs = vec![
            format!("rhs-{}", "x".repeat(256)),
            format!("rhs-{}", "y".repeat(256)),
        ];
        let data2 = StringArray::from(expected_rhs.clone());

        let mut agg1 = MostRecentAggregator::new(PrimitiveType::P64x2);
        agg1.ensure_capacity(4);
        agg1.ingest(&[&data1], &tickets1).unwrap();

        let mut agg2 = MostRecentAggregator::new(PrimitiveType::P64x2);
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
                Some(format!("lhs-{}", "a".repeat(256))),
                Some(expected_rhs[0].clone()),
                Some(format!("lhs-{}", "b".repeat(256))),
                Some(expected_rhs[1].clone()),
            ]
        );
    }

    #[test]
    fn test_most_recent_aggregator_fixed_size_list_finish_preserves_nulls() {
        let tickets = arrow_array::UInt64Array::from(vec![0, 2, 0]);
        let values = Float32Array::from(vec![1.0, 2.0, 5.0, 6.0, 7.0, 8.0]);
        let field = Arc::new(arrow_schema::Field::new(
            "item",
            arrow_schema::DataType::Float32,
            false,
        ));
        let array_data = ArrayData::builder(arrow_schema::DataType::FixedSizeList(field, 2))
            .len(3)
            .add_child_data(values.into_data())
            .build()
            .unwrap();
        let data = FixedSizeListArray::from(array_data);

        let mut agg = MostRecentAggregator::new(PrimitiveType::List(ListItemType::F32, 2));
        agg.ensure_capacity(3);
        agg.ingest(&[&data], &tickets).unwrap();

        let result = Box::new(agg).finish().unwrap();
        let result = result
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result.is_null(1), true);
        let values = result
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        assert_eq!(values, vec![7.0, 8.0, 0.0, 0.0, 5.0, 6.0]);
    }
}
