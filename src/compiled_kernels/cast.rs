use std::sync::Arc;

use super::{ArrowKernelError, Kernel};
use crate::{
    compiled_kernels::dsl2::{
        compile, DSLArgument, DSLContext, DSLFunction, DSLStmt, DSLType, RunnableDSLFunction,
    },
    compiled_writers::WriterSpec,
    intersect_and_copy_nulls, logical_arrow_type, PrimitiveType,
};
use arrow_array::{
    cast::AsArray,
    make_array,
    types::{Int16Type, Int32Type, Int64Type, Int8Type, UInt8Type},
    Array, ArrayRef, BooleanArray, LargeStringArray, StringArray, StringViewArray,
};
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;

pub fn coalesce_type(res: ArrayRef, tar: &DataType) -> Result<ArrayRef, ArrowKernelError> {
    if res.data_type() == tar {
        return Ok(res);
    }

    // might need to translate binary type to string type
    match (res.data_type(), tar) {
        (DataType::UInt8, DataType::Boolean) => {
            let res = res.as_primitive::<UInt8Type>();
            let res = BooleanArray::from_unary(res, |x| x > 0);
            Ok(Arc::new(res))
        }
        (DataType::Binary, DataType::Utf8) => {
            let res = res.as_binary();
            let (offsets, data, nulls) = res.clone().into_parts();
            debug_assert!(
                StringArray::try_new(offsets.clone(), data.clone(), nulls.clone()).is_ok()
            );
            let s = Arc::new(unsafe { StringArray::new_unchecked(offsets, data, nulls) });
            Ok(s)
        }
        (DataType::LargeBinary, DataType::LargeUtf8) => {
            let res = res.as_binary::<i64>();
            let (offsets, data, nulls) = res.clone().into_parts();
            debug_assert!(
                LargeStringArray::try_new(offsets.clone(), data.clone(), nulls.clone()).is_ok()
            );
            let s = Arc::new(unsafe { LargeStringArray::new_unchecked(offsets, data, nulls) });
            Ok(s)
        }
        (DataType::BinaryView, DataType::Utf8View) => {
            let res = res.as_binary_view();
            let (view, bufs, nulls) = res.clone().into_parts();
            let s = Arc::new(StringViewArray::new(view, bufs, nulls));
            Ok(s)
        }
        (DataType::Dictionary(kt, _vt), DataType::Dictionary(t_kt, t_vt)) if kt == t_kt => {
            match kt.as_ref() {
                DataType::Int8 => {
                    let res = res.as_dictionary::<Int8Type>();
                    Ok(Arc::new(res.with_values(coalesce_type(
                        res.values().clone(),
                        t_vt.as_ref(),
                    )?)))
                }
                DataType::Int16 => {
                    let res = res.as_dictionary::<Int16Type>();
                    Ok(Arc::new(res.with_values(coalesce_type(
                        res.values().clone(),
                        t_vt.as_ref(),
                    )?)))
                }
                DataType::Int32 => {
                    let res = res.as_dictionary::<Int32Type>();
                    Ok(Arc::new(res.with_values(coalesce_type(
                        res.values().clone(),
                        t_vt.as_ref(),
                    )?)))
                }
                DataType::Int64 => {
                    let res = res.as_dictionary::<Int64Type>();
                    Ok(Arc::new(res.with_values(coalesce_type(
                        res.values().clone(),
                        t_vt.as_ref(),
                    )?)))
                }
                _ => unreachable!("invalid dictionary key type {}", kt),
            }
        }
        (DataType::RunEndEncoded(re, _v), DataType::RunEndEncoded(t_re, t_v))
            if re.data_type() == t_re.data_type() =>
        {
            let arr = res.into_data();
            let res = arr.child_data()[0].clone();
            let val = make_array(arr.child_data()[1].clone());
            let val = coalesce_type(val, t_v.data_type())?;

            Ok(make_array(unsafe {
                ArrayDataBuilder::new(tar.clone())
                    .len(arr.len())
                    .add_child_data(res)
                    .add_child_data(val.into_data())
                    .build_unchecked()
            }))
        }
        (DataType::FixedSizeList(src, src_len), DataType::FixedSizeList(tar, tar_len))
            if src_len == tar_len && src.data_type() == tar.data_type() =>
        {
            Ok(res)
        }

        _ => todo!("unable to coalesce {} into {}", res.data_type(), tar),
    }
}

pub struct CastKernel {
    k: RunnableDSLFunction,
    tar: DataType,
}
unsafe impl Sync for CastKernel {}
unsafe impl Send for CastKernel {}

impl Kernel for CastKernel {
    type Key = (DataType, DataType);

    type Input<'a>
        = &'a dyn Array
    where
        Self: 'a;

    type Params = DataType;

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        if inp.data_type() == &self.tar {
            return Ok(make_array(inp.to_data()));
        }
        let res = self.k.run(&[DSLArgument::datum(&inp)])?[0].clone();
        assert_eq!(
            inp.len(),
            res.len(),
            "cast result had different lengths (expected: {} got: {})",
            inp.len(),
            res.len()
        );
        let res = intersect_and_copy_nulls(&[&inp], res)?;
        coalesce_type(res, &self.tar)
    }

    fn compile(arr: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let tar = params;

        let w = WriterSpec::for_data_type(&tar);

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("cast");
        let arg1 = func.add_arg(&mut ctx, DSLType::array_like(arr, "n"));
        func.add_ret(w, "n");
        func.add_body(DSLStmt::for_each(&mut ctx, &[arg1], |loop_vars| {
            let v = &loop_vars[0].expr();
            let v = if logical_arrow_type(&tar) == DataType::Boolean {
                v.cast_to_bool()?
            } else {
                v.primitive_cast(PrimitiveType::for_arrow_type(&tar))?
            };
            DSLStmt::emit(0, v)
        })?);
        let func = compile(func, [DSLArgument::Datum(arr)])?;

        Ok(CastKernel { k: func, tar })
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((i.data_type().clone(), p.clone()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray,
        types::{Int16Type, Int32Type, Int8Type},
        Array, ArrayRef, DictionaryArray, Int16Array, Int32Array, Int64Array, RunArray,
        StringArray, UInt8Array,
    };
    use arrow_schema::{DataType, Field};
    use itertools::Itertools;

    use crate::{
        compiled_kernels::{CastKernel, Kernel},
        dictionary_data_type, run_end_data_type,
    };

    #[test]
    fn test_i32_to_i64() {
        let data = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let expected: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
        let k = CastKernel::compile(&(&data as &dyn Array), DataType::Int64).unwrap();
        assert!(k.k.vectorized);
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i32_to_i64_block() {
        let data = Int32Array::from((0..200).collect_vec());
        let expected: ArrayRef = Arc::new(Int64Array::from((0..200).collect_vec()));
        let k = CastKernel::compile(&(&data as &dyn Array), DataType::Int64).unwrap();
        assert!(k.k.vectorized);
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i32nullable_to_i64_block() {
        let data = Int32Array::from(
            (0..50)
                .map(|x| if x % 2 == 0 { Some(x) } else { None })
                .collect_vec(),
        );
        let expected: ArrayRef = Arc::new(Int64Array::from(
            (0..50)
                .map(|x| if x % 2 == 0 { Some(x as i64) } else { None })
                .collect_vec(),
        ));
        let k = CastKernel::compile(&(&data as &dyn Array), DataType::Int64).unwrap();
        assert!(k.k.vectorized);
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i64_to_u8_block() {
        let data = Int64Array::from((0..200).collect_vec());
        let expected: ArrayRef = Arc::new(UInt8Array::from((0..200).collect_vec()));
        let k = CastKernel::compile(&(&data as &dyn Array), DataType::UInt8).unwrap();
        assert!(k.k.vectorized);
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i32_to_i32_short_circuit() {
        let data = Int32Array::from(vec![Some(1), None, Some(3), Some(4)]);
        let k = CastKernel::compile(&(&data as &dyn Array), DataType::Int32).unwrap();

        let res = k.call(&data).unwrap();
        let res = res.as_primitive::<Int32Type>();

        assert_eq!(res, &data);
    }

    #[test]
    fn test_i32_to_dict() {
        let data = Int32Array::from(vec![1, 1, 1, 2, 2, 300, 300, 400]);
        let k = CastKernel::compile(
            &(&data as &dyn Array),
            dictionary_data_type(DataType::Int8, DataType::Int32),
        )
        .unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(res.len(), 8);

        let res = res.as_dictionary::<Int8Type>();
        assert_eq!(
            &[1, 2, 300, 400],
            res.values().as_primitive::<Int32Type>().values()
        );
        assert_eq!(&[0, 0, 0, 1, 1, 2, 2, 3], res.keys().values());
    }

    #[test]
    fn test_dict_to_i32() {
        let keys = UInt8Array::from(vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5]);
        let values = Int32Array::from(vec![-100, -200, -300, -400, -500, -600]);
        let da = DictionaryArray::new(keys, Arc::new(values));
        let k = CastKernel::compile(&(&da as &dyn Array), DataType::Int32).unwrap();
        assert!(k.k.vectorized);

        let res = k.call(&da).unwrap();
        assert_eq!(
            res.as_primitive::<Int32Type>().values(),
            &[-100, -100, -100, -200, -200, -200, -300, -300, -300, -400, -500, -500, -600, -600]
        );
    }

    #[test]
    fn test_str_to_dict() {
        let data = StringArray::from(vec![
            "this",
            "this",
            "is",
            "a test",
            "a test",
            "a string that is longer than 12 chars",
        ]);
        let k = CastKernel::compile(
            &(&data as &dyn Array),
            dictionary_data_type(DataType::Int8, DataType::Utf8),
        )
        .unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(res.len(), 6);

        let res = res.as_dictionary::<Int8Type>();
        let strv = res.values().as_string::<i32>();
        assert_eq!("this", strv.value(0));
        assert_eq!("is", strv.value(1));
        assert_eq!("a test", strv.value(2));
        assert_eq!(&[0, 0, 1, 2, 2, 3], res.keys().values());
    }

    #[test]
    fn test_dict_to_str_view() {
        let data = StringArray::from(vec![
            "this",
            "this",
            "is",
            "a test",
            "a string that is longer than 12 chars",
            "a test",
        ]);
        let ddata =
            arrow_cast::cast::cast(&data, &dictionary_data_type(DataType::Int8, DataType::Utf8))
                .unwrap();
        let k = CastKernel::compile(&(&ddata as &dyn Array), DataType::Utf8View).unwrap();
        let res = k.call(&ddata).unwrap();
        let res = res.as_string_view();

        assert_eq!(res.len(), data.len());
        for (ours, orig) in res.iter().zip(data.iter()) {
            assert_eq!(ours, orig);
        }
    }

    #[test]
    fn test_dict_to_str_flat() {
        let data = StringArray::from(vec![
            "this",
            "this",
            "is",
            "a test",
            "a test",
            "a string that is longer than 12 chars",
        ]);
        let ddata =
            arrow_cast::cast::cast(&data, &dictionary_data_type(DataType::Int8, DataType::Utf8))
                .unwrap();
        let k = CastKernel::compile(&(&ddata as &dyn Array), DataType::Utf8).unwrap();
        let res = k.call(&ddata).unwrap();
        let res = res.as_string::<i32>();

        assert_eq!(res.len(), data.len());
        for (ours, orig) in res.iter().zip(data.iter()) {
            assert_eq!(ours, orig);
        }
    }

    #[test]
    fn test_str_dict_to_ree() {
        let data = StringArray::from(vec![
            "this",
            "this",
            "is",
            "a test",
            "a test",
            "a string that is longer than 12 chars",
        ]);
        let ddata =
            arrow_cast::cast::cast(&data, &dictionary_data_type(DataType::Int8, DataType::Utf8))
                .unwrap();

        let k = CastKernel::compile(
            &(&ddata as &dyn Array),
            run_end_data_type(&DataType::Int32, &DataType::Utf8),
        )
        .unwrap();
        let res = k.call(&ddata).unwrap();
        let res = res.as_any().downcast_ref::<RunArray<Int32Type>>().unwrap();
        let res = res.downcast::<StringArray>().unwrap();

        assert_eq!(res.len(), data.len());
        for (ours, orig) in res.into_iter().zip(data.iter()) {
            assert_eq!(ours, orig);
        }
    }

    #[test]
    fn test_ree_bool_to_bool() {
        use arrow_array::BooleanArray;

        let data = BooleanArray::from(vec![true, false, true]);
        let res = Int16Array::from(vec![2, 4, 800]);
        let re = RunArray::<Int16Type>::try_new(&res, &data).unwrap();

        let k = CastKernel::compile(&(&re as &dyn Array), DataType::Boolean).unwrap();
        let res = k.call(&re).unwrap();
        let res = res.as_boolean();

        let re = re.downcast::<BooleanArray>().unwrap();
        assert_eq!(res.len(), re.len());
        for (ours, orig) in res.iter().zip(re.into_iter()) {
            assert_eq!(ours, orig);
        }
    }

    #[test]
    fn test_fixed_size_list_f16_to_f32() {
        let mut builder = arrow_array::builder::FixedSizeListBuilder::new(
            arrow_array::builder::Float16Builder::new(),
            4,
        );
        for _ in 0..2 {
            for value in [1.0_f32, 2.5, 3.25, 4.0] {
                builder.values().append_value(half::f16::from_f32(value));
            }
            builder.append(true);
        }
        let data = builder.finish();

        let k = CastKernel::compile(
            &(&data as &dyn Array),
            DataType::FixedSizeList(Arc::new(Field::new_list_field(DataType::Float32, true)), 4),
        )
        .unwrap();
        let res = k.call(&data).unwrap();
        let res = res.as_fixed_size_list();
        let values = res
            .values()
            .as_primitive::<arrow_array::types::Float32Type>();

        assert_eq!(res.len(), 2);
        assert_eq!(values.values(), &[1.0, 2.5, 3.25, 4.0, 1.0, 2.5, 3.25, 4.0]);
    }

    #[test]
    fn test_utf8_to_large_utf8() {
        let data = StringArray::from(vec!["hello", "world", "large utf8"]);
        let k = CastKernel::compile(&(&data as &dyn Array), DataType::LargeUtf8).unwrap();
        let res = k.call(&data).unwrap();
        let res = res.as_string::<i64>();

        assert_eq!(res.len(), data.len());
        for (ours, orig) in res.iter().zip(data.iter()) {
            assert_eq!(ours, orig);
        }
    }
}
