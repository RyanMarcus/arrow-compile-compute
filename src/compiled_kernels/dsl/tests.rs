use std::sync::Arc;

use arrow_array::{
    builder::{FixedSizeListBuilder, Int32Builder},
    cast::AsArray,
    types::{BinaryViewType, Float32Type, Int32Type, Int64Type, UInt64Type},
    Array, BooleanArray, Float32Array, Int32Array, Scalar, StringArray,
};
use arrow_schema::DataType;
use itertools::Itertools;

use crate::{compiled_kernels::dsl::DSLKernel, dictionary_data_type, PrimitiveType};

use super::KernelOutputType;

#[test]
fn test_dsl_int_between() {
    let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
    let sca1 = Int32Array::new_scalar(1);
    let sca2 = Int32Array::new_scalar(6);

    let k = DSLKernel::compile(&[&data, &sca1, &sca2], |ctx| {
        let data = ctx.get_input(0)?;
        let scalar1 = ctx.get_input(1)?;
        let scalar2 = ctx.get_input(2)?;

        ctx.iter_over(vec![data, scalar1, scalar2])
            .map(|i| vec![i[0].gt(&i[1]), i[0].lt(&i[2])])
            .map(|i| vec![i[0].and(&i[1])])
            .collect(KernelOutputType::Boolean)
    })
    .unwrap();

    let res = k.call(&[&data, &sca1, &sca2]);
    let res = res.unwrap();

    assert_eq!(
        res.as_boolean(),
        &BooleanArray::from(vec![false, true, true, true, true, false])
    );
}

#[test]
fn test_dsl_str_between() {
    let data = StringArray::from(vec!["a", "b", "c", "d", "e", "f"]);
    let sca1 = StringArray::new_scalar("a");
    let sca2 = StringArray::new_scalar("f");

    let k = DSLKernel::compile(&[&data, &sca1, &sca2], |ctx| {
        let data = ctx.get_input(0)?;
        let scalar1 = ctx.get_input(1)?;
        let scalar2 = ctx.get_input(2)?;

        ctx.iter_over(vec![data, scalar1, scalar2])
            .map(|i| vec![i[0].gt(&i[1]), i[0].lt(&i[2])])
            .map(|i| vec![i[0].and(&i[1])])
            .collect(KernelOutputType::Boolean)
    })
    .unwrap();

    let res = k.call(&[&data, &sca1, &sca2]);
    let res = res.unwrap();

    assert_eq!(
        res.as_boolean(),
        &BooleanArray::from(vec![false, true, true, true, true, false])
    );
}

#[test]
fn test_dsl_int_max() {
    let data1 = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
    let data2 = Int32Array::from(vec![-1, 20, -3, 40, -5, 1]);

    let k = DSLKernel::compile(&[&data1, &data2], |ctx| {
        let lhs = ctx.get_input(0)?;
        let rhs = ctx.get_input(1)?;

        ctx.iter_over(vec![lhs, rhs])
            .map(|i| vec![i[0].gt(&i[1]).select(&i[0], &i[1])])
            .collect(KernelOutputType::Array)
    })
    .unwrap();

    let res = k.call(&[&data1, &data2]).unwrap();

    assert_eq!(
        res.as_primitive::<Int32Type>(),
        &Int32Array::from(vec![1, 20, 3, 40, 5, 6])
    );
}

#[test]
fn test_dsl_at_max() {
    // compute max(data3[data1], data3[data2])
    let data1 = Int32Array::from(vec![0, 1, 2, 3, 4, 5]);
    let data2 = Int32Array::from(vec![2, 1, 0, 5, 4, 3]);
    let data3 = Int32Array::from(vec![0, 10, 20, 30, 40, 50]);

    let k = DSLKernel::compile(&[&data1, &data2, &data3], |ctx| {
        let lhs = ctx.get_input(0)?;
        let rhs = ctx.get_input(1)?;
        let dat = ctx.get_input(2)?;

        ctx.iter_over(vec![lhs, rhs])
            .map(|i| vec![dat.at(&i[0]), dat.at(&i[1])])
            .map(|i| vec![i[0].gt(&i[1]).select(&i[0], &i[1])])
            .collect(KernelOutputType::Array)
    })
    .unwrap();

    let res = k.call(&[&data1, &data2, &data3]).unwrap();

    assert_eq!(
        res.as_primitive::<Int32Type>(),
        &Int32Array::from(vec![20, 10, 20, 50, 40, 50])
    );
}

#[test]
fn test_dsl_string_flatten() {
    let odata = vec!["this", "this", "is", "a", "a", "test"];
    let data = StringArray::from(odata.clone());
    let data = arrow_cast::cast(
        &data,
        &dictionary_data_type(DataType::UInt8, DataType::Utf8),
    )
    .unwrap();

    let k = DSLKernel::compile(&[&data], |ctx| {
        let inp = ctx.get_input(0)?;
        ctx.iter_over(vec![inp]).collect(KernelOutputType::String)
    })
    .unwrap();

    let res = k.call(&[&data]).unwrap();
    let res = res
        .as_string::<i32>()
        .iter()
        .map(|x| x.unwrap())
        .collect_vec();
    assert_eq!(res, odata);
}

#[test]
fn test_kernel_set_bit_iter() {
    let data = BooleanArray::from(vec![true, true, false, true, false, true]);
    let k = DSLKernel::compile(&[&data], |ctx| {
        let inp = ctx.get_input(0)?.into_set_bits()?;
        ctx.iter_over(vec![inp]).collect(KernelOutputType::Array)
    })
    .unwrap();
    let res = k.call(&[&data]).unwrap();
    let res = res
        .as_primitive::<UInt64Type>()
        .iter()
        .map(|x| x.unwrap())
        .collect_vec();
    assert_eq!(res, vec![0, 1, 3, 5]);
}

#[test]
fn test_kernel_set_bit_idx() {
    let filter = BooleanArray::from(vec![true, true, false, true, false, true]);
    let data = Int32Array::from(vec![10, 20, 30, 40, 50, 60]);
    let k = DSLKernel::compile(&[&filter, &data], |ctx| {
        let filter = ctx.get_input(0)?.into_set_bits()?;
        let data = ctx.get_input(1)?;
        ctx.iter_over(vec![filter])
            .map(|i| vec![data.at(&i[0])])
            .collect(KernelOutputType::Array)
    })
    .unwrap();
    let res = k.call(&[&filter, &data]).unwrap();
    let res = res
        .as_primitive::<Int32Type>()
        .iter()
        .map(|x| x.unwrap())
        .collect_vec();
    assert_eq!(res, vec![10, 20, 40, 60]);
}

#[test]
fn test_kernel_filter_lt() {
    let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let scalar1 = Int32Array::new_scalar(5);
    let k = DSLKernel::compile(&[&data, &scalar1], |ctx| {
        ctx.iter_over(vec![ctx.get_input(0)?, ctx.get_input(1)?])
            .filter(|v| v[0].lt(&v[1]))
            .map(|v| vec![v[0].clone()])
            .collect(KernelOutputType::Array)
    })
    .unwrap();
    let res = k.call(&[&data, &scalar1]).unwrap();
    let res = res.as_primitive::<Int32Type>();
    assert_eq!(res.values(), &[1, 2, 3, 4]);
}

#[test]
fn test_kernel_convert_i32_to_i64() {
    let data = Int32Array::from(vec![1, 2, 3, -4, 5, 6]);
    let k = DSLKernel::compile(&[&data], |ctx| {
        ctx.iter_over(vec![ctx.get_input(0)?])
            .map(|v| vec![v[0].clone().convert(PrimitiveType::I64)])
            .collect(KernelOutputType::Array)
    })
    .unwrap();
    let res = k.call(&[&data]).unwrap();
    let res = res.as_primitive::<Int64Type>();
    assert_eq!(res.values(), &[1, 2, 3, -4, 5, 6]);
}

#[test]
fn test_kernel_convert_f32_to_i64() {
    let data = Float32Array::from(vec![1.2, 2.1, 3.4, -4.6, 5.7, 6.0]);
    let k = DSLKernel::compile(&[&data], |ctx| {
        ctx.iter_over(vec![ctx.get_input(0)?])
            .map(|v| vec![v[0].clone().convert(PrimitiveType::I64)])
            .collect(KernelOutputType::Array)
    })
    .unwrap();
    let res = k.call(&[&data]).unwrap();
    let res = res.as_primitive::<Int64Type>();
    assert_eq!(res.values(), &[1, 2, 3, -4, 5, 6]);
}

#[test]
fn test_kernel_convert_str_to_view() {
    let strs = vec!["this", "is", "a test with at least one long string"];
    let data = StringArray::from(strs.clone());
    let k = DSLKernel::compile(&[&data], |ctx| {
        ctx.iter_over(vec![ctx.get_input(0)?])
            .collect(KernelOutputType::View)
    })
    .unwrap();
    let res = k.call(&[&data]).unwrap();
    let res = res.as_byte_view::<BinaryViewType>();
    let res = res
        .iter()
        .map(|b| std::str::from_utf8(b.unwrap()).unwrap())
        .collect_vec();
    assert_eq!(res, strs);
}

#[test]
fn test_kernel_convert_str_to_view_single() {
    let strs = vec!["𑚀aAￒￚA"];
    let data = StringArray::from(strs.clone());
    let k = DSLKernel::compile(&[&data], |ctx| {
        ctx.iter_over(vec![ctx.get_input(0)?])
            .collect(KernelOutputType::View)
    })
    .unwrap();
    let res = k.call(&[&data]).unwrap();
    let res = res.as_byte_view::<BinaryViewType>();
    let res = res
        .iter()
        .map(|b| std::str::from_utf8(b.unwrap()).unwrap())
        .collect_vec();
    assert_eq!(res, strs);
}

#[test]
fn test_kernel_powi() {
    let data = Float32Array::from(vec![1.0, 2.0, 3.0]);
    let k = DSLKernel::compile(&[&data], |ctx| {
        ctx.iter_over(vec![ctx.get_input(0)?])
            .map(|x| vec![x[0].powi(2)])
            .collect(KernelOutputType::Array)
    })
    .unwrap();
    let res = k.call(&[&data]).unwrap();
    let res = res.as_primitive::<Float32Type>();
    let res = res.iter().map(|x| x.unwrap()).collect_vec();
    assert_eq!(res, vec![1.0, 4.0, 9.0]);
}

#[test]
fn test_kernel_sqrt() {
    let data = Float32Array::from(vec![1.0, 4.0, 9.0]);
    let k = DSLKernel::compile(&[&data], |ctx| {
        ctx.iter_over(vec![ctx.get_input(0)?])
            .map(|x| vec![x[0].sqrt()])
            .collect(KernelOutputType::Array)
    })
    .unwrap();
    let res = k.call(&[&data]).unwrap();
    let res = res.as_primitive::<Float32Type>();
    let res = res.iter().map(|x| x.unwrap()).collect_vec();
    assert_eq!(res, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_kernel_add_vecs() {
    let ib = Int32Builder::new();
    let mut vecs = FixedSizeListBuilder::new(ib, 4);
    vecs.values().append_value(1);
    vecs.values().append_value(2);
    vecs.values().append_value(3);
    vecs.values().append_value(4);
    vecs.append(true);
    vecs.values().append_value(4);
    vecs.values().append_value(5);
    vecs.values().append_value(6);
    vecs.values().append_value(7);
    vecs.append(true);
    let vecs: Arc<dyn Array> = Arc::new(vecs.finish());

    let ib = Int32Builder::new();
    let mut to_add = FixedSizeListBuilder::new(ib, 4);
    to_add.values().append_value(1);
    to_add.values().append_value(10);
    to_add.values().append_value(100);
    to_add.values().append_value(1000);
    to_add.append(true);
    let to_add = Scalar::new(to_add.finish());

    let k = DSLKernel::compile(&[&vecs, &to_add], |ctx| {
        ctx.iter_over(vec![ctx.get_input(0)?, ctx.get_input(1)?])
            .map(|i| vec![i[0].add(&i[1])])
            .collect(KernelOutputType::FixedSizeList(4))
    })
    .unwrap();
    let res = k.call(&[&vecs, &to_add]).unwrap();
    let res = res.as_fixed_size_list();
    assert_eq!(res.len(), 2);

    assert_eq!(
        res.value(0).as_primitive::<Int32Type>().values(),
        &[2, 12, 103, 1004]
    );

    assert_eq!(
        res.value(1).as_primitive::<Int32Type>().values(),
        &[5, 15, 106, 1007]
    );
}

#[test]
fn test_kernel_dot_vecs() {
    let ib = Int32Builder::new();
    let mut vecs = FixedSizeListBuilder::new(ib, 4);
    vecs.values().append_value(1);
    vecs.values().append_value(2);
    vecs.values().append_value(3);
    vecs.values().append_value(4);
    vecs.append(true);
    vecs.values().append_value(4);
    vecs.values().append_value(5);
    vecs.values().append_value(6);
    vecs.values().append_value(7);
    vecs.append(true);
    let vecs: Arc<dyn Array> = Arc::new(vecs.finish());

    let ib = Int32Builder::new();
    let mut to_add = FixedSizeListBuilder::new(ib, 4);
    to_add.values().append_value(1);
    to_add.values().append_value(10);
    to_add.values().append_value(100);
    to_add.values().append_value(1000);
    to_add.append(true);
    let to_add = Scalar::new(to_add.finish());

    let k = DSLKernel::compile(&[&vecs, &to_add], |ctx| {
        ctx.iter_over(vec![ctx.get_input(0)?, ctx.get_input(1)?])
            .map(|i| vec![i[0].mul(&i[1])])
            .map(|i| vec![i[0].vec_sum()])
            .collect(KernelOutputType::Array)
    })
    .unwrap();
    let res = k.call(&[&vecs, &to_add]).unwrap();
    let res = res.as_primitive::<Int32Type>().values();
    assert_eq!(
        res,
        &[
            1 + 2 * 10 + 3 * 100 + 4 * 1000,
            4 + 5 * 10 + 6 * 100 + 7 * 1000
        ]
    );
}

#[test]
fn test_kernel_vec_sum() {
    let ib = Int32Builder::new();
    let mut vecs = FixedSizeListBuilder::new(ib, 4);
    vecs.values().append_value(1);
    vecs.values().append_value(2);
    vecs.values().append_value(3);
    vecs.values().append_value(4);
    vecs.append(true);
    vecs.values().append_value(4);
    vecs.values().append_value(5);
    vecs.values().append_value(6);
    vecs.values().append_value(7);
    vecs.append(true);
    let vecs: Arc<dyn Array> = Arc::new(vecs.finish());

    let k = DSLKernel::compile(&[&vecs], |ctx| {
        ctx.iter_over(vec![ctx.get_input(0)?])
            .map(|i| vec![i[0].vec_sum()])
            .collect(KernelOutputType::Array)
    })
    .unwrap();
    let res = k.call(&[&vecs]).unwrap();
    let res = res.as_primitive::<Int32Type>().values();
    assert_eq!(res, &[4 + 3 + 2 + 1, 4 + 5 + 6 + 7]);
}
