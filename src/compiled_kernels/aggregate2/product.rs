use std::sync::LazyLock;

use arrow_array::{Array, ArrayRef, BooleanArray, Datum, UInt64Array};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        aggregate2::Aggregator,
        dsl2::{
            compile, DSLArgument, DSLBitwiseBinOp, DSLBuffer, DSLContext, DSLFunction, DSLStmt,
            DSLType, DSLValue, RunnableDSLFunction,
        },
        DSLArithBinOp, KernelCache,
    },
    logical_nulls, ArrowKernelError, Kernel, PrimitiveType,
};

/// `product` accumulates in the input type (no widening) and wraps on overflow,
/// matching arrow's `product` / `product_checked`-less semantics.
fn check_numeric(pt: PrimitiveType) -> Result<(), ArrowKernelError> {
    match pt {
        PrimitiveType::P64x2 | PrimitiveType::List(_, _) => Err(
            ArrowKernelError::UnsupportedArguments(format!(
                "product only supports numeric types, got {pt:?}"
            )),
        ),
        _ => Ok(()),
    }
}

pub struct ProductAggKernel {
    k: RunnableDSLFunction,
    has_nulls: bool,
}
unsafe impl Send for ProductAggKernel {}
unsafe impl Sync for ProductAggKernel {}

impl Kernel for ProductAggKernel {
    // `has_nulls` is part of the key: nullable inputs compile a validity-aware
    // kernel, non-nullable ones the faster plain kernel.
    type Key = (DataType, bool);

    type Input<'a> = (&'a mut DSLBuffer, &'a mut DSLBuffer, &'a dyn Datum, &'a UInt64Array);

    type Params = ();

    type Output = ();

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (used, buf, data, tickets) = inp;
        let validity;
        let mut args = vec![
            DSLArgument::buffer(used),
            DSLArgument::buffer(buf),
            DSLArgument::Datum(data),
            DSLArgument::Datum(tickets),
        ];
        if self.has_nulls {
            let nulls = logical_nulls(data.get().0)?.ok_or_else(|| {
                ArrowKernelError::UnsupportedArguments(
                    "nullable product kernel called with non-null input".to_string(),
                )
            })?;
            validity = BooleanArray::new(nulls.into_inner(), None);
            args.push(DSLArgument::Datum(&validity));
        }
        self.k.run(&args)?;
        Ok(())
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (used, buf, data, tickets) = inp;
        let input_pt = PrimitiveType::for_arrow_type(data.get().0.data_type());
        check_numeric(input_pt)?;
        let has_nulls = logical_nulls(data.get().0)?.is_some();

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("product");
        let use_arg = func.add_arg(&mut ctx, DSLType::buffer_of(PrimitiveType::U8, "k"));
        let buf_arg = func.add_arg(&mut ctx, DSLType::buffer_of(input_pt, "k"));
        let dat_arg = func.add_arg(&mut ctx, DSLType::array_like(*data, "n"));
        let tic_arg = func.add_arg(&mut ctx, DSLType::array_like(tickets, "n"));

        // an empty placeholder, only used for its (boolean) type at compile time
        let validity = BooleanArray::from(Vec::<bool>::new());

        if has_nulls {
            let val_arg = func.add_arg(&mut ctx, DSLType::array_like(&validity, "n"));
            func.add_body(
                DSLStmt::for_each(&mut ctx, &[tic_arg, dat_arg, val_arg], |loop_vars| {
                    let ticket = loop_vars[0].expr();
                    let data = loop_vars[1].expr();
                    let valid = loop_vars[2].expr();
                    let used = use_arg.expr().at(&ticket)?;
                    let cur = buf_arg.expr().at(&ticket)?;

                    // only accumulate valid (non-null) slots, so nulls are skipped
                    DSLStmt::cond(
                        valid,
                        DSLStmt::cond_else(
                            used.cast_to_bool()?,
                            DSLStmt::set(
                                &buf_arg,
                                &ticket,
                                &cur.arith(DSLArithBinOp::Mul, data.clone())?,
                            )?,
                            vec![
                                DSLStmt::set(&buf_arg, &ticket, &data)?,
                                DSLStmt::set(
                                    &use_arg,
                                    &ticket,
                                    &DSLValue::u8(1).expr().primitive_cast(PrimitiveType::U8)?,
                                )?,
                            ],
                        )?,
                    )
                })
                .unwrap(),
            );
        } else {
            func.add_body(
                DSLStmt::for_each(&mut ctx, &[tic_arg, dat_arg], |loop_vars| {
                    let ticket = loop_vars[0].expr();
                    let data = loop_vars[1].expr();
                    let used = use_arg.expr().at(&ticket)?;
                    let cur = buf_arg.expr().at(&ticket)?;

                    DSLStmt::cond_else(
                        used.cast_to_bool()?,
                        // group already has a value: multiply the running product
                        DSLStmt::set(
                            &buf_arg,
                            &ticket,
                            &cur.arith(DSLArithBinOp::Mul, data.clone())?,
                        )?,
                        // first value for this group: store it and mark the group used
                        vec![
                            DSLStmt::set(&buf_arg, &ticket, &data)?,
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
        }

        let mut empty_used = DSLBuffer::empty_like(used);
        let mut empty_buf = DSLBuffer::empty_like(buf);
        let mut args = vec![
            DSLArgument::buffer(&mut empty_used),
            DSLArgument::buffer(&mut empty_buf),
            DSLArgument::Datum(*data),
            DSLArgument::Datum(tickets),
        ];
        if has_nulls {
            args.push(DSLArgument::Datum(&validity));
        }
        let func = compile(func, args)?;

        Ok(Self {
            k: func,
            has_nulls,
        })
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        let dt = i.2.get().0.data_type().clone();
        check_numeric(PrimitiveType::for_arrow_type(&dt))?;
        let has_nulls = logical_nulls(i.2.get().0)?.is_some();
        Ok((dt, has_nulls))
    }
}

pub struct ProductMergeKernel(RunnableDSLFunction);
unsafe impl Send for ProductMergeKernel {}
unsafe impl Sync for ProductMergeKernel {}

impl Kernel for ProductMergeKernel {
    type Key = PrimitiveType;

    type Input<'a> = (
        &'a mut DSLBuffer,
        &'a mut DSLBuffer,
        &'a mut DSLBuffer,
        &'a mut DSLBuffer,
    );

    type Params = ();

    type Output = ();

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (used1, buf1, used2, buf2) = inp;
        self.0.run(&[
            DSLArgument::buffer(used1),
            DSLArgument::buffer(buf1),
            DSLArgument::buffer(used2),
            DSLArgument::buffer(buf2),
        ])?;
        Ok(())
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (_used1, buf1, _used2, buf2) = inp;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("merge_product");
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
                    let product = buf1.arith(DSLArithBinOp::Mul, buf2.clone())?;

                    DSLStmt::cond_else(
                        only2,
                        vec![
                            DSLStmt::set(&use1_arg, &idx, &use2)?,
                            DSLStmt::set(&buf1_arg, &idx, &buf2)?,
                        ],
                        DSLStmt::cond(both, DSLStmt::set(&buf1_arg, &idx, &product)?)?,
                    )
                },
            )
            .unwrap(),
        );

        let func = compile(
            func,
            [
                DSLArgument::buffer(&mut DSLBuffer::empty_like(_used1)),
                DSLArgument::buffer(&mut DSLBuffer::empty_like(buf1)),
                DSLArgument::buffer(&mut DSLBuffer::empty_like(_used2)),
                DSLArgument::buffer(&mut DSLBuffer::empty_like(buf2)),
            ],
        )?;

        Ok(Self(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok(i.1.ty)
    }
}

static AGG_PROGRAM_CACHE: LazyLock<KernelCache<ProductAggKernel>> = LazyLock::new(KernelCache::new);
static MERGE_PROGRAM_CACHE: LazyLock<KernelCache<ProductMergeKernel>> =
    LazyLock::new(KernelCache::new);

pub struct ProductAggregator {
    used: DSLBuffer,
    buf: DSLBuffer,
}

impl ProductAggregator {
    pub fn new(pt: PrimitiveType) -> Result<Self, ArrowKernelError> {
        check_numeric(pt)?;
        Ok(Self {
            used: DSLBuffer::new(PrimitiveType::U8, 0),
            buf: DSLBuffer::new(pt, 0),
        })
    }
}

impl Aggregator for ProductAggregator {
    fn create(tys: &[&DataType]) -> Result<Box<Self>, ArrowKernelError> {
        if tys.len() != 1 {
            return Err(ArrowKernelError::ArgumentMismatch(
                "product create takes exactly one input".to_string(),
            ));
        }
        Ok(Box::new(Self::new(PrimitiveType::for_arrow_type(tys[0]))?))
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
                "product ingest takes exactly one input".to_string(),
            ));
        }
        self.ensure_capacity_for_tickets(tickets);
        AGG_PROGRAM_CACHE.get((&mut self.used, &mut self.buf, &data[0], tickets), ())?;
        Ok(())
    }

    fn merge(&mut self, mut other: Self) -> Result<(), ArrowKernelError> {
        if self.buf.len < other.buf.len {
            self.ensure_capacity(other.buf.len as usize);
        } else if other.buf.len < self.buf.len {
            other.ensure_capacity(self.buf.len as usize);
        }

        MERGE_PROGRAM_CACHE.get(
            (&mut self.used, &mut self.buf, &mut other.used, &mut other.buf),
            (),
        )?;
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
        types::{Int32Type, Int64Type},
        Int32Array, Int64Array, StringArray,
    };
    use itertools::Itertools;

    use super::*;

    #[test]
    fn test_product_aggregator_grouped() {
        // group 0: 2*4*1 = 8; group 1: 3*5 = 15; group 2: 7
        let tickets = UInt64Array::from(vec![0, 1, 0, 1, 2, 0]);
        let data = Int32Array::from(vec![2, 3, 4, 5, 7, 1]);

        let mut agg = ProductAggregator::new(PrimitiveType::I32).unwrap();
        agg.ensure_capacity(3);
        agg.ingest(&[&data], &tickets).unwrap();

        let result = Box::new(agg).finish().unwrap();
        let result = result
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(result, vec![8, 15, 7]);
    }

    #[test]
    fn test_product_aggregator_ingest_merge_and_finish() {
        // agg1: g0=2, g1=3, g2=5 ; agg2: g0=7, g3=11 ; merged: g0=14, g1=3, g2=5, g3=11
        let mut agg1 = ProductAggregator::new(PrimitiveType::I32).unwrap();
        agg1.ensure_capacity(3);
        agg1.ingest(
            &[&Int32Array::from(vec![2, 3, 5])],
            &UInt64Array::from(vec![0, 1, 2]),
        )
        .unwrap();

        let mut agg2 = ProductAggregator::new(PrimitiveType::I32).unwrap();
        agg2.ensure_capacity(4);
        agg2.ingest(
            &[&Int32Array::from(vec![7, 11])],
            &UInt64Array::from(vec![0, 3]),
        )
        .unwrap();

        agg1.merge(agg2).unwrap();

        let result = Box::new(agg1).finish().unwrap();
        let result = result
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();

        assert_eq!(result, vec![14, 3, 5, 11]);
    }

    #[test]
    fn test_product_ungrouped_matches_arrow() {
        // ungrouped product = product of the whole array, compared to arrow's kernel
        let data = Int64Array::from(vec![1, 2, 3, 4, 5]);

        let mut agg = ProductAggregator::new(PrimitiveType::I64).unwrap();
        agg.ingest_ungrouped(&[&data]).unwrap();
        let result = Box::new(agg).finish().unwrap();
        let result = result.as_primitive::<Int64Type>();

        assert_eq!(result.len(), 1);
        let arrow = arrow_arith::aggregate::product(&data).unwrap();
        assert_eq!(result.value(0), arrow);
        assert_eq!(result.value(0), 120);
    }

    #[test]
    fn test_product_wraps_on_overflow_like_arrow() {
        // i32 product that overflows: must wrap, matching arrow (no checked variant)
        let data = Int32Array::from(vec![100_000, 100_000]); // 1e10 > i32::MAX -> wraps
        let mut agg = ProductAggregator::new(PrimitiveType::I32).unwrap();
        agg.ingest_ungrouped(&[&data]).unwrap();
        let result = Box::new(agg).finish().unwrap();
        let result = result.as_primitive::<Int32Type>();

        let arrow = arrow_arith::aggregate::product(&data).unwrap();
        assert_eq!(result.value(0), arrow);
        assert_eq!(result.value(0), 100_000i32.wrapping_mul(100_000));
    }

    #[test]
    fn test_product_rejects_non_numeric_types() {
        let data = StringArray::from(vec!["a", "b"]);
        match ProductAggregator::create(&[data.data_type()]) {
            Err(ArrowKernelError::UnsupportedArguments(msg)) => {
                assert!(msg.contains("product only supports numeric types"));
            }
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("expected an error for non-numeric type"),
        }
    }

    #[test]
    fn test_product_nulls_match_arrow() {
        // arrow skips nulls: product([2, null, 3]) == 6
        let data = Int64Array::from(vec![Some(2), None, Some(3)]);
        let mut agg = ProductAggregator::new(PrimitiveType::I64).unwrap();
        agg.ingest_ungrouped(&[&data]).unwrap();
        let result = Box::new(agg).finish().unwrap();
        let result = result.as_primitive::<Int64Type>();

        let arrow = arrow_arith::aggregate::product(&data).unwrap();
        assert_eq!(result.value(0), arrow, "our product must skip nulls like arrow");
    }
}
