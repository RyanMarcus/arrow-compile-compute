use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{UInt64Type, UInt8Type},
    Array, ArrayRef, BooleanArray,
};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType;
use itertools::Itertools;

use crate::{
    compiled_kernels::{
        cast::coalesce_type,
        dsl2::{
            compile, DSLArgument, DSLBuffer, DSLContext, DSLFunction, DSLStmt, DSLType, DSLValue,
            RunnableDSLFunction,
        },
        null_utils::replace_nulls,
        DSLArithBinOp,
    },
    iter, logical_nulls, normalized_base_type, ArrowKernelError, Kernel, PrimitiveType,
};

pub struct PartitionKernel(RunnableDSLFunction);
unsafe impl Send for PartitionKernel {}
unsafe impl Sync for PartitionKernel {}

impl Kernel for PartitionKernel {
    type Key = (DataType, DataType);

    type Input<'a> = (&'a dyn Array, &'a dyn Array);

    type Params = ();

    type Output = Vec<ArrayRef>;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (arr, idxes) = inp;

        if arr.is_empty() {
            return Ok(Vec::new());
        }

        if idxes.is_nullable() {
            return Err(ArrowKernelError::UnsupportedArguments(
                "partition got nullable index list".to_string(),
            ));
        }

        let nparts = iter::iter_nonnull_u64(idxes)?.max().ok_or_else(|| {
            ArrowKernelError::UnsupportedArguments("partition got empty index list".to_string())
        })? as usize
            + 1;

        let mut sum_buf = DSLBuffer::new(PrimitiveType::U64, nparts + 1);
        let mut out_buf = DSLBuffer::new(PrimitiveType::for_arrow_type(arr.data_type()), arr.len());

        self.0.run(&[
            DSLArgument::datum(&arr),
            DSLArgument::datum(&idxes),
            DSLArgument::buffer(&mut sum_buf),
            DSLArgument::buffer(&mut out_buf),
        ])?;

        let mut res = out_buf.into_array();
        if arr.data_type() == &DataType::Boolean {
            // need to convert from i8 to boolean
            let ints = res.as_primitive::<UInt8Type>();
            res = Arc::new(BooleanArray::from_unary(ints, |x| x != 0));
        }

        let part_offsets = sum_buf.into_array();
        let part_offsets = part_offsets.as_primitive::<UInt64Type>().values();
        if arr.is_nullable() {
            let nulls = logical_nulls(arr)?.unwrap();
            let mut part_offsets = part_offsets.to_vec();
            part_offsets.insert(0, 0);
            let mut perm = vec![0; nulls.len()];
            for (data_idx, part_idx) in iter::iter_nonnull_u64(idxes)?.enumerate() {
                let my_pos = part_offsets[part_idx as usize];
                part_offsets[part_idx as usize] += 1;
                perm[my_pos as usize] = data_idx;
            }

            let new_nulls = NullBuffer::new(BooleanBuffer::collect_bool(nulls.len(), |idx| {
                nulls.is_valid(perm[idx])
            }));
            res = replace_nulls(res, Some(new_nulls));
        }

        res = coalesce_type(res, &normalized_base_type(arr.data_type()))?;

        let mut consumed = 0;
        let partitions = part_offsets[..part_offsets.len() - 1]
            .iter()
            .map(|offset| {
                let offset = offset - consumed;
                let part = res.slice(0, offset as usize);
                res = res.slice(offset as usize, res.len() - offset as usize);
                consumed += offset;
                part
            })
            .collect_vec();

        Ok(partitions)
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let func = partition_kernel(inp.0, inp.1)?;
        Ok(Self(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((i.0.data_type().clone(), i.1.data_type().clone()))
    }
}

pub fn partition_kernel(
    arr: &dyn Array,
    idxes: &dyn Array,
) -> Result<RunnableDSLFunction, ArrowKernelError> {
    let mut ctx = DSLContext::new();
    let mut func = DSLFunction::new("partition");
    let arr_arg = func.add_arg(&mut ctx, DSLType::array_like(&arr, "n"));
    let idxes_arg = func.add_arg(&mut ctx, DSLType::array_like(&idxes, "n"));
    let sum_buf = func.add_arg(&mut ctx, DSLType::buffer_of(PrimitiveType::U64, "k"));
    let out_buf = func.add_arg(
        &mut ctx,
        DSLType::buffer_of(PrimitiveType::for_arrow_type(arr.data_type()), "n"),
    );

    // compute histogram of partition indexes (number of entries in each partition)
    func.add_body(
        DSLStmt::for_each(&mut ctx, std::slice::from_ref(&idxes_arg), |loop_vars| {
            let idx = loop_vars[0].expr().primitive_cast(PrimitiveType::U64)?;
            let idx = idx.arith(
                DSLArithBinOp::Add,
                DSLValue::u64(1).expr().primitive_cast(PrimitiveType::U64)?,
            )?;
            let curr = sum_buf
                .expr()
                .at(&idx)?
                .primitive_cast(PrimitiveType::U64)?;
            let next = curr.arith(
                DSLArithBinOp::Add,
                DSLValue::u64(1).expr().primitive_cast(PrimitiveType::U64)?,
            )?;
            DSLStmt::set(&sum_buf, &idx, &next)
        })
        .unwrap(),
    );

    // compute cumulative sum of histogram, which become cursors
    func.add_body(
        DSLStmt::for_range(
            &mut ctx,
            DSLValue::u64(1).expr().primitive_cast(PrimitiveType::U64)?,
            sum_buf.expr().len()?,
            |i| {
                let prev = sum_buf.expr().at(&i.expr().arith(
                    DSLArithBinOp::Sub,
                    DSLValue::u64(1).expr().primitive_cast(PrimitiveType::U64)?,
                )?)?;
                let curr = sum_buf.expr().at(&i.expr())?;
                let new_val = prev.arith(DSLArithBinOp::Add, curr)?;
                DSLStmt::set(&sum_buf, &i.expr(), &new_val)
            },
        )
        .unwrap(),
    );

    // write into out buf
    func.add_body(
        DSLStmt::for_each(&mut ctx, &[arr_arg, idxes_arg], |loop_vars| {
            let val = loop_vars[0].expr();
            let part_idx = loop_vars[1].expr().primitive_cast(PrimitiveType::U64)?;
            let cursor = sum_buf.expr().at(&part_idx)?;
            let new_cursor = cursor.arith(
                DSLArithBinOp::Add,
                DSLValue::u64(1).expr().primitive_cast(PrimitiveType::U64)?,
            )?;

            Ok(vec![
                DSLStmt::set(&out_buf, &cursor, &val)?,
                DSLStmt::set(&sum_buf, &part_idx, &new_cursor)?,
            ])
        })
        .unwrap(),
    );

    let mut sum_buf = DSLBuffer::new(PrimitiveType::U64, 128);
    let mut out_buf = DSLBuffer::new(PrimitiveType::for_arrow_type(arr.data_type()), arr.len());
    let func = compile(
        func,
        [
            DSLArgument::Datum(&arr),
            DSLArgument::Datum(&idxes),
            DSLArgument::buffer(&mut sum_buf),
            DSLArgument::buffer(&mut out_buf),
        ],
    )?;

    Ok(func)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray, types::Int32Type, Array, ArrayRef, BooleanArray, Int32Array, UInt32Array,
    };
    use itertools::Itertools;

    use crate::{
        compiled_kernels::{partition::PartitionKernel, Kernel},
        ArrowKernelError,
    };

    fn run_partition(
        data: &dyn Array,
        part: &dyn Array,
    ) -> Result<Vec<ArrayRef>, ArrowKernelError> {
        let kernel = PartitionKernel::compile(&(data, part), ())?;
        kernel.call((data, part))
    }

    #[test]
    fn test_part_i32_nonulls() {
        let data: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
        let res = run_partition(data.as_ref(), part.as_ref()).unwrap();

        assert_eq!(res.len(), 2);
        assert_eq!(
            res[0].as_primitive::<Int32Type>().values(),
            &[1, 3, 5, 7, 9]
        );
        assert_eq!(
            res[1].as_primitive::<Int32Type>().values(),
            &[2, 4, 6, 8, 10]
        );
    }

    #[test]
    fn test_part_i32_nulls() {
        let data: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(1),
            Some(2),
            None,
            Some(4),
            Some(5),
            None,
            Some(7),
            Some(8),
            Some(9),
            None,
        ]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
        let res = run_partition(data.as_ref(), part.as_ref()).unwrap();

        assert_eq!(res.len(), 2);
        assert_eq!(
            res[0].as_primitive::<Int32Type>().iter().collect_vec(),
            &[Some(1), None, Some(5), Some(7), Some(9)]
        );
        assert_eq!(
            res[1].as_primitive::<Int32Type>().iter().collect_vec(),
            &[Some(2), Some(4), None, Some(8), None]
        );
    }

    #[test]
    fn test_part_bool() {
        let data: ArrayRef = Arc::new(BooleanArray::from(vec![
            true, true, true, true, true, false, false, false, false, false,
        ]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
        let res = run_partition(data.as_ref(), part.as_ref()).unwrap();

        assert_eq!(res.len(), 2);
        assert_eq!(
            res[0].as_boolean().iter().map(|x| x.unwrap()).collect_vec(),
            &[true, true, true, false, false]
        );
        assert_eq!(
            res[1].as_boolean().iter().map(|x| x.unwrap()).collect_vec(),
            &[true, true, false, false, false]
        );
    }

    #[test]
    fn test_part_strs() {
        let data: ArrayRef = Arc::new(arrow_array::StringArray::from(vec![
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        ]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
        let res = run_partition(data.as_ref(), part.as_ref()).unwrap();

        assert_eq!(res.len(), 2);
        assert_eq!(
            res[0]
                .as_string::<i32>()
                .iter()
                .map(|x| x.unwrap())
                .collect_vec(),
            &["a", "c", "e", "g", "i"]
        );
        assert_eq!(
            res[1]
                .as_string::<i32>()
                .iter()
                .map(|x| x.unwrap())
                .collect_vec(),
            &["b", "d", "f", "h", "j"]
        );
    }

    #[test]
    fn test_part_infers_partition_count_from_max_index() {
        let data: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 2, 0]));
        let res = run_partition(data.as_ref(), part.as_ref()).unwrap();

        assert_eq!(res.len(), 3);
        assert_eq!(res[0].as_primitive::<Int32Type>().values(), &[10, 30]);
        assert!(res[1].is_empty());
        assert_eq!(res[2].as_primitive::<Int32Type>().values(), &[20]);
    }
}
