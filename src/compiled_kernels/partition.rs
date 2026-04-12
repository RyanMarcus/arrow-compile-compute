use arrow_array::{Array, ArrayRef, UInt64Array};
use itertools::Itertools;

use crate::{
    compiled_kernels::{
        dsl2::{
            compile, DSLArgument, DSLBuffer, DSLContext, DSLExpr, DSLFunction, DSLStmt, DSLType,
            DSLValue,
        },
        DSLArithBinOp,
    },
    iter, ArrowKernelError, PrimitiveType,
};

pub fn partition(
    arr: &dyn Array,
    part_idxes: &dyn Array,
    nparts: Option<usize>,
) -> Result<Vec<ArrayRef>, ArrowKernelError> {
    if arr.len() != part_idxes.len() {
        return Err(ArrowKernelError::SizeMismatch);
    }

    if part_idxes.is_nullable() {
        return Err(ArrowKernelError::UnsupportedArguments(
            "partition indexes cannot be nullable".to_string(),
        ));
    };

    let nparts = match nparts {
        Some(x) => x,
        None => iter::iter_nonnull_u64(part_idxes)?
            .max()
            .map(|x| x as usize + 1)
            .unwrap_or(0),
    };

    if nparts == 0 {
        return Ok(vec![]);
    }

    let mut part_offsets = vec![0_u64; nparts + 1];
    for idx in iter::iter_nonnull_u64(part_idxes)? {
        let idx = idx as usize;
        if idx >= nparts {
            return Err(ArrowKernelError::OutOfBounds(nparts));
        }
        part_offsets[idx + 1] += 1;
    }
    for idx in 1..part_offsets.len() {
        part_offsets[idx] += part_offsets[idx - 1];
    }

    let mut indexes = vec![0; arr.len()];
    iter::iter_nonnull_u64(part_idxes)?
        .enumerate()
        .for_each(|(arr_idx, part_idx)| {
            let part_idx = part_idx as usize;
            let new_pos = part_offsets[part_idx] as usize;
            part_offsets[part_idx] += 1;
            indexes[new_pos] = arr_idx as u64;
        });
    let indexes = UInt64Array::from(indexes);

    let mut res = crate::arrow_interface::select::take(arr, &indexes)?;
    let mut consumed = 0;
    let partitions = part_offsets[..part_offsets.len() - 1]
        .into_iter()
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

pub fn partition_kernel(arr: &ArrayRef, idxes: &ArrayRef) -> Result<(), ArrowKernelError> {
    let mut ctx = DSLContext::new();
    let mut func = DSLFunction::new("partition");
    let arr_arg = func.add_arg(&mut ctx, DSLType::array_like(arr, "n"));
    let idxes_arg = func.add_arg(&mut ctx, DSLType::array_like(idxes, "n"));
    let sum_buf = func.add_arg(&mut ctx, DSLType::buffer_of(PrimitiveType::U64, "k"));
    let out_buf = func.add_arg(
        &mut ctx,
        DSLType::buffer_of(PrimitiveType::for_arrow_type(arr.data_type()), "n"),
    );

    // compute histogram of partition indexes (number of entries in each partition)
    func.add_body(
        DSLStmt::for_each(&mut ctx, &[idxes_arg.clone()], |loop_vars| {
            let idx = loop_vars[0].expr().primitive_cast(PrimitiveType::U64)?;
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
    compile(
        func,
        [
            DSLArgument::Datum(arr),
            DSLArgument::Datum(idxes),
            DSLArgument::buffer(&mut sum_buf),
            DSLArgument::buffer(&mut out_buf),
        ],
    )
    .unwrap();

    todo!()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray, types::Int32Type, ArrayRef, BooleanArray, Int32Array, UInt32Array,
    };
    use itertools::Itertools;

    use crate::{
        compiled_kernels::partition::{partition, partition_kernel},
        ArrowKernelError,
    };

    #[test]
    fn test_part_dsl() {
        let data: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
        partition_kernel(&data, &part);
    }

    #[test]
    fn test_part_i32_nonulls() {
        let data: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
        let res = partition(&data, &part, None).unwrap();

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
        let res = partition(&data, &part, None).unwrap();

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
        let res = partition(&data, &part, None).unwrap();

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
        let res = partition(&data, &part, None).unwrap();

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
        let res = partition(&data, &part, None).unwrap();

        assert_eq!(res.len(), 3);
        assert_eq!(res[0].as_primitive::<Int32Type>().values(), &[10, 30]);
        assert!(res[1].is_empty());
        assert_eq!(res[2].as_primitive::<Int32Type>().values(), &[20]);
    }

    #[test]
    fn test_part_rejects_out_of_bounds_partition_index() {
        let data: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 2, 1]));
        let err = partition(&data, &part, Some(2)).unwrap_err();

        assert!(matches!(err, ArrowKernelError::OutOfBounds(2)));
    }
}
