use std::{ffi::c_void, sync::Arc};

use arrow_array::{
    Array, BinaryArray, BooleanArray, Datum, FixedSizeListArray, Int32Array, Int32RunArray,
    LargeStringArray, Scalar, StringArray, StringViewArray,
};
use arrow_data::ArrayData;
use arrow_schema::{DataType, Field};
use inkwell::{
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction},
    module::Module,
    values::FunctionValue,
    AddressSpace, OptimizationLevel,
};

use crate::pointers_to_str;

use super::{
    array_to_iter, array_to_setbit_iter, datum_to_iter, generate_next, generate_reset_iterator,
    IteratorHolder,
};

unsafe fn pointers_to_bytes(ptrs: u128) -> Vec<u8> {
    let b = ptrs.to_le_bytes();
    let ptr1 = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let ptr2 = u64::from_le_bytes(b[8..16].try_into().unwrap());
    let len = (ptr2 - ptr1) as usize;
    std::slice::from_raw_parts(ptr1 as *const u8, len).to_vec()
}

fn build_engine<'a>(
    ctx: &'a Context,
    module_name: &str,
    label: &str,
    dt: &DataType,
    ih: &IteratorHolder,
) -> (ExecutionEngine<'a>, String, String) {
    let module = ctx.create_module(module_name);
    let next = generate_next(ctx, &module, label, dt, ih).unwrap();
    let reset = generate_reset_iterator(ctx, &module, label, ih).unwrap();
    module.verify().unwrap();

    let next_name = next.get_name().to_str().unwrap().to_owned();
    let reset_name = reset.get_name().to_str().unwrap().to_owned();
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    (ee, next_name, reset_name)
}

fn add_i32_vec_wrapper<'a>(
    ctx: &'a Context,
    module: &Module<'a>,
    name: &str,
    next: FunctionValue<'a>,
    lanes: u32,
) -> FunctionValue<'a> {
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let bool_type = ctx.bool_type();
    let i32_type = ctx.i32_type();
    let i64_type = ctx.i64_type();
    let vec_type = i32_type.vec_type(lanes);

    let wrap_type = bool_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
    let wrap_fn = module.add_function(name, wrap_type, None);
    let entry = ctx.append_basic_block(wrap_fn, "entry");
    let store_block = ctx.append_basic_block(wrap_fn, "store");
    let end_block = ctx.append_basic_block(wrap_fn, "end");
    let builder = ctx.create_builder();
    builder.position_at_end(entry);

    let iter_ptr = wrap_fn.get_nth_param(0).unwrap().into_pointer_value();
    let out_ptr = wrap_fn.get_nth_param(1).unwrap().into_pointer_value();
    let tmp = builder.build_alloca(vec_type, "tmp").unwrap();
    let result = builder
        .build_call(next, &[iter_ptr.into(), tmp.into()], "call_next")
        .unwrap()
        .try_as_basic_value()
        .unwrap_basic()
        .into_int_value();

    builder
        .build_conditional_branch(result, store_block, end_block)
        .unwrap();

    builder.position_at_end(store_block);
    let vec_value = builder
        .build_load(vec_type, tmp, "vec")
        .unwrap()
        .into_vector_value();
    for lane in 0..lanes {
        let lane_val = builder
            .build_extract_element(
                vec_value,
                i32_type.const_int(lane as u64, false),
                &format!("lane_{lane}"),
            )
            .unwrap()
            .into_int_value();
        let lane_ptr = unsafe {
            builder
                .build_in_bounds_gep(
                    i32_type,
                    out_ptr,
                    &[i64_type.const_int(lane as u64, false)],
                    &format!("lane_ptr_{lane}"),
                )
                .unwrap()
        };
        builder.build_store(lane_ptr, lane_val).unwrap();
    }
    builder
        .build_return(Some(&bool_type.const_all_ones()))
        .unwrap();

    builder.position_at_end(end_block);
    builder.build_return(Some(&bool_type.const_zero())).unwrap();

    wrap_fn
}

fn build_i32_vec_engine<'a>(
    ctx: &'a Context,
    module_name: &str,
    label: &str,
    dt: &DataType,
    ih: &IteratorHolder,
    lanes: u32,
) -> (ExecutionEngine<'a>, String, String) {
    let module = ctx.create_module(module_name);
    let next = generate_next(ctx, &module, label, dt, ih).unwrap();
    let reset = generate_reset_iterator(ctx, &module, label, ih).unwrap();
    let wrapper_name = format!("{label}_wrapper");
    let wrapper = add_i32_vec_wrapper(ctx, &module, &wrapper_name, next, lanes);
    module.verify().unwrap();

    let wrapper_name = wrapper.get_name().to_str().unwrap().to_owned();
    let reset_name = reset.get_name().to_str().unwrap().to_owned();
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    (ee, wrapper_name, reset_name)
}

unsafe fn collect_i32(
    next: &JitFunction<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>,
    iter: &mut IteratorHolder,
) -> Vec<i32> {
    let mut out = 0_i32;
    let mut result = Vec::new();
    while next.call(iter.get_mut_ptr(), &mut out as *mut i32) {
        result.push(out);
    }
    result
}

unsafe fn collect_u8(
    next: &JitFunction<unsafe extern "C" fn(*mut c_void, *mut u8) -> bool>,
    iter: &mut IteratorHolder,
) -> Vec<u8> {
    let mut out = 0_u8;
    let mut result = Vec::new();
    while next.call(iter.get_mut_ptr(), &mut out as *mut u8) {
        result.push(out);
    }
    result
}

unsafe fn collect_u64(
    next: &JitFunction<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>,
    iter: &mut IteratorHolder,
) -> Vec<u64> {
    let mut out = 0_u64;
    let mut result = Vec::new();
    while next.call(iter.get_mut_ptr(), &mut out as *mut u64) {
        result.push(out);
    }
    result
}

unsafe fn collect_strings(
    next: &JitFunction<unsafe extern "C" fn(*mut c_void, *mut u128) -> bool>,
    iter: &mut IteratorHolder,
) -> Vec<String> {
    let mut out = 0_u128;
    let mut result = Vec::new();
    while next.call(iter.get_mut_ptr(), &mut out as *mut u128) {
        result.push(pointers_to_str(out));
    }
    result
}

unsafe fn collect_i32x4(
    next: &JitFunction<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>,
    iter: &mut IteratorHolder,
) -> Vec<[i32; 4]> {
    let mut out = [0_i32; 4];
    let mut result = Vec::new();
    while next.call(iter.get_mut_ptr(), out.as_mut_ptr()) {
        result.push(out);
    }
    result
}

fn make_fixed_size_list_array() -> FixedSizeListArray {
    let values = Int32Array::from((0..12).collect::<Vec<_>>());
    let field = Arc::new(Field::new("item", DataType::Int32, false));
    let array_data = ArrayData::builder(DataType::FixedSizeList(field, 4))
        .len(3)
        .add_child_data(values.into_data())
        .build()
        .unwrap();
    FixedSizeListArray::from(array_data)
}

fn make_fixed_size_list_scalar() -> Scalar<FixedSizeListArray> {
    let values = Int32Array::from(vec![7, 8, 9, 10]);
    let field = Arc::new(Field::new("item", DataType::Int32, false));
    let array_data = ArrayData::builder(DataType::FixedSizeList(field, 4))
        .len(1)
        .add_child_data(values.into_data())
        .build()
        .unwrap();
    Scalar::new(FixedSizeListArray::from(array_data))
}

#[test]
fn test_reset_primitive_iterator() {
    let data = Int32Array::from(vec![10, 11, 12, 13]).slice(1, 3);
    let mut iter = array_to_iter(&data);
    let ctx = Context::create();
    let (ee, next_name, reset_name) = build_engine(
        &ctx,
        "reset_primitive",
        "reset_primitive",
        data.data_type(),
        &iter,
    );

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let first = collect_i32(&next, &mut iter);
        assert_eq!(first, vec![11, 12, 13]);
        reset.call(iter.get_mut_ptr());
        let second = collect_i32(&next, &mut iter);
        assert_eq!(second, first);
    }
}

#[test]
fn test_reset_string_iterator() {
    let data = StringArray::from(vec!["zero", "one", "two", "three"]).slice(1, 3);
    let mut iter = datum_to_iter(&data).unwrap();
    let ctx = Context::create();
    let (ee, next_name, reset_name) = build_engine(
        &ctx,
        "reset_string",
        "reset_string",
        data.get().0.data_type(),
        &iter,
    );

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u128) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let first = collect_strings(&next, &mut iter);
        assert_eq!(first, vec!["one", "two", "three"]);
        reset.call(iter.get_mut_ptr());
        let second = collect_strings(&next, &mut iter);
        assert_eq!(second, first);
    }
}

#[test]
fn test_reset_large_string_iterator() {
    let data = LargeStringArray::from(vec!["alpha", "beta", "gamma"]);
    let mut iter = datum_to_iter(&data).unwrap();
    let ctx = Context::create();
    let (ee, next_name, reset_name) = build_engine(
        &ctx,
        "reset_large_string",
        "reset_large_string",
        data.get().0.data_type(),
        &iter,
    );

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u128) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let first = collect_strings(&next, &mut iter);
        assert_eq!(first, vec!["alpha", "beta", "gamma"]);
        reset.call(iter.get_mut_ptr());
        let second = collect_strings(&next, &mut iter);
        assert_eq!(second, first);
    }
}

#[test]
fn test_reset_view_iterator() {
    let data = StringViewArray::from(vec!["this", "is", "view"]);
    let mut iter = datum_to_iter(&data).unwrap();
    let ctx = Context::create();
    let (ee, next_name, reset_name) =
        build_engine(&ctx, "reset_view", "reset_view", data.data_type(), &iter);

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u128) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let first = collect_strings(&next, &mut iter);
        assert_eq!(first, vec!["this", "is", "view"]);
        reset.call(iter.get_mut_ptr());
        let second = collect_strings(&next, &mut iter);
        assert_eq!(second, first);
    }
}

#[test]
fn test_reset_bitmap_iterator() {
    let data = BooleanArray::from(vec![true, false, true, true]).slice(1, 3);
    let mut iter = array_to_iter(&data);
    let ctx = Context::create();
    let (ee, next_name, reset_name) = build_engine(
        &ctx,
        "reset_bitmap",
        "reset_bitmap",
        data.data_type(),
        &iter,
    );

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u8) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let first = collect_u8(&next, &mut iter);
        assert_eq!(first, vec![0, 1, 1]);
        reset.call(iter.get_mut_ptr());
        let second = collect_u8(&next, &mut iter);
        assert_eq!(second, first);
    }
}

#[test]
fn test_reset_setbit_iterator() {
    let data = BooleanArray::from(vec![false, true, false, true, true, false, true]).slice(1, 5);
    let mut iter = array_to_setbit_iter(&data).unwrap();
    let ctx = Context::create();
    let (ee, next_name, reset_name) = build_engine(
        &ctx,
        "reset_setbit",
        "reset_setbit",
        data.data_type(),
        &iter,
    );

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let first = collect_u64(&next, &mut iter);
        assert_eq!(first, vec![0, 2, 3]);
        reset.call(iter.get_mut_ptr());
        let second = collect_u64(&next, &mut iter);
        assert_eq!(second, first);
    }
}

#[test]
fn test_reset_dictionary_iterator() {
    let data = Int32Array::from(vec![10, 20, 30, 40, 50]);
    let data = arrow_cast::cast(
        &data,
        &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
    )
    .unwrap()
    .slice(1, 3);
    let mut iter = array_to_iter(&data);
    let ctx = Context::create();
    let (ee, next_name, reset_name) =
        build_engine(&ctx, "reset_dict", "reset_dict", data.data_type(), &iter);

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let first = collect_i32(&next, &mut iter);
        assert_eq!(first, vec![20, 30, 40]);
        reset.call(iter.get_mut_ptr());
        let second = collect_i32(&next, &mut iter);
        assert_eq!(second, first);
    }
}

#[test]
fn test_reset_run_end_iterator() {
    let values = Int32Array::from(vec![10, 20, 30]);
    let run_ends = Int32Array::from(vec![2, 4, 6]);
    let data = Int32RunArray::try_new(&run_ends, &values)
        .unwrap()
        .slice(2, 4);
    let mut iter = datum_to_iter(&data).unwrap();
    let ctx = Context::create();
    let (ee, next_name, reset_name) = build_engine(
        &ctx,
        "reset_run_end",
        "reset_run_end",
        data.data_type(),
        &iter,
    );

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let first = collect_i32(&next, &mut iter);
        assert_eq!(first, vec![20, 20, 30, 30]);
        reset.call(iter.get_mut_ptr());
        let second = collect_i32(&next, &mut iter);
        assert_eq!(second, first);
    }
}

#[test]
fn test_reset_fixed_size_list_iterator() {
    let data = make_fixed_size_list_array();
    let mut iter = array_to_iter(&data);
    let ctx = Context::create();
    let (ee, next_name, reset_name) = build_i32_vec_engine(
        &ctx,
        "reset_fixed_size_list",
        "reset_fixed_size_list",
        data.data_type(),
        &iter,
        4,
    );

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let first = collect_i32x4(&next, &mut iter);
        assert_eq!(first, vec![[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]);
        reset.call(iter.get_mut_ptr());
        let second = collect_i32x4(&next, &mut iter);
        assert_eq!(second, first);
    }
}

#[test]
fn test_reset_scalar_primitive_iterator() {
    let data = Int32Array::new_scalar(42);
    let mut iter = datum_to_iter(&data).unwrap();
    let ctx = Context::create();
    let (ee, next_name, reset_name) = build_engine(
        &ctx,
        "reset_scalar_primitive",
        "reset_scalar_primitive",
        data.get().0.data_type(),
        &iter,
    );

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let mut out = 0_i32;
        assert!(next.call(iter.get_mut_ptr(), &mut out as *mut i32));
        assert_eq!(out, 42);
        reset.call(iter.get_mut_ptr());
        assert!(next.call(iter.get_mut_ptr(), &mut out as *mut i32));
        assert_eq!(out, 42);
    }
}

#[test]
fn test_reset_scalar_string_iterator() {
    let data = StringArray::new_scalar("hello");
    let mut iter = datum_to_iter(&data).unwrap();
    let ctx = Context::create();
    let (ee, next_name, reset_name) = build_engine(
        &ctx,
        "reset_scalar_string",
        "reset_scalar_string",
        data.get().0.data_type(),
        &iter,
    );

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u128) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let mut out = 0_u128;
        assert!(next.call(iter.get_mut_ptr(), &mut out as *mut u128));
        assert_eq!(pointers_to_str(out), "hello");
        reset.call(iter.get_mut_ptr());
        assert!(next.call(iter.get_mut_ptr(), &mut out as *mut u128));
        assert_eq!(pointers_to_str(out), "hello");
    }
}

#[test]
fn test_reset_scalar_binary_iterator() {
    let data = BinaryArray::new_scalar(b"hello");
    let mut iter = datum_to_iter(&data).unwrap();
    let ctx = Context::create();
    let (ee, next_name, reset_name) = build_engine(
        &ctx,
        "reset_scalar_binary",
        "reset_scalar_binary",
        data.get().0.data_type(),
        &iter,
    );

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u128) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let mut out = 0_u128;
        assert!(next.call(iter.get_mut_ptr(), &mut out as *mut u128));
        assert_eq!(pointers_to_bytes(out), b"hello".to_vec());
        reset.call(iter.get_mut_ptr());
        assert!(next.call(iter.get_mut_ptr(), &mut out as *mut u128));
        assert_eq!(pointers_to_bytes(out), b"hello".to_vec());
    }
}

#[test]
fn test_reset_scalar_vector_iterator() {
    let data = make_fixed_size_list_scalar();
    let mut iter = datum_to_iter(&data).unwrap();
    let ctx = Context::create();
    let (ee, next_name, reset_name) = build_i32_vec_engine(
        &ctx,
        "reset_scalar_vec",
        "reset_scalar_vec",
        data.get().0.data_type(),
        &iter,
        4,
    );

    let next = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(&next_name)
            .unwrap()
    };
    let reset = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void)>(&reset_name)
            .unwrap()
    };

    unsafe {
        let mut out = [0_i32; 4];
        assert!(next.call(iter.get_mut_ptr(), out.as_mut_ptr()));
        assert_eq!(out, [7, 8, 9, 10]);
        reset.call(iter.get_mut_ptr());
        out.fill(0);
        assert!(next.call(iter.get_mut_ptr(), out.as_mut_ptr()));
        assert_eq!(out, [7, 8, 9, 10]);
    }
}
