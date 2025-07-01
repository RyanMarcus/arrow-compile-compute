mod bitmap;
mod dictionary;
mod primitive;
mod runend;
mod scalar;
mod setbit;
mod string;
mod view;

use std::ffi::c_void;

use arrow_array::{
    cast::AsArray,
    types::{
        BinaryViewType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
        Int8Type, StringViewType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, BooleanArray, Datum, GenericStringArray, Int16RunArray, Int32RunArray, Int64RunArray,
};

use arrow_schema::DataType;
use bitmap::BitmapIterator;
use dictionary::DictionaryIterator;
use inkwell::{
    context::Context,
    intrinsics::Intrinsic,
    module::{Linkage, Module},
    types::BasicType,
    values::{BasicValue, FunctionValue},
    AddressSpace, IntPredicate,
};
use primitive::PrimitiveIterator;
use runend::{add_bsearch, RunEndIterator};
use scalar::{ScalarPrimitiveIterator, ScalarStringIterator};
use setbit::SetBitIterator;
use string::{LargeStringIterator, StringIterator};

use crate::{
    declare_blocks, increment_pointer, new_iter::view::ViewIterator, ArrowKernelError,
    PrimitiveType,
};

pub fn array_to_setbit_iter(arr: &BooleanArray) -> Result<IteratorHolder, ArrowKernelError> {
    Ok(IteratorHolder::SetBit((arr).into()))
}

/// Convert an Arrow Array into an IteratorHolder, which contains the C-style
/// iterator along with all the other C-style iterator needed. For example, a
/// dictionary array will get a C-style iterator and then two sub-iterators for
/// the keys and values.
fn array_to_iter(arr: &dyn Array) -> IteratorHolder {
    match arr.data_type() {
        arrow_schema::DataType::Null => todo!(),
        arrow_schema::DataType::Boolean => IteratorHolder::Bitmap(arr.as_boolean().into()),
        arrow_schema::DataType::Int8 => {
            IteratorHolder::Primitive(arr.as_primitive::<Int8Type>().into())
        }
        arrow_schema::DataType::Int16 => {
            IteratorHolder::Primitive(arr.as_primitive::<Int16Type>().into())
        }
        arrow_schema::DataType::Int32 => {
            IteratorHolder::Primitive(arr.as_primitive::<Int32Type>().into())
        }
        arrow_schema::DataType::Int64 => {
            IteratorHolder::Primitive(arr.as_primitive::<Int64Type>().into())
        }
        arrow_schema::DataType::UInt8 => {
            IteratorHolder::Primitive(arr.as_primitive::<UInt8Type>().into())
        }
        arrow_schema::DataType::UInt16 => {
            IteratorHolder::Primitive(arr.as_primitive::<UInt16Type>().into())
        }
        arrow_schema::DataType::UInt32 => {
            IteratorHolder::Primitive(arr.as_primitive::<UInt32Type>().into())
        }
        arrow_schema::DataType::UInt64 => {
            IteratorHolder::Primitive(arr.as_primitive::<UInt64Type>().into())
        }
        arrow_schema::DataType::Float16 => {
            IteratorHolder::Primitive(arr.as_primitive::<Float16Type>().into())
        }
        arrow_schema::DataType::Float32 => {
            IteratorHolder::Primitive(arr.as_primitive::<Float32Type>().into())
        }
        arrow_schema::DataType::Float64 => {
            IteratorHolder::Primitive(arr.as_primitive::<Float64Type>().into())
        }
        arrow_schema::DataType::Timestamp(_time_unit, _) => {
            IteratorHolder::Primitive(arr.as_primitive::<Int64Type>().into())
        }
        arrow_schema::DataType::Date32 => todo!(),
        arrow_schema::DataType::Date64 => todo!(),
        arrow_schema::DataType::Time32(_time_unit) => todo!(),
        arrow_schema::DataType::Time64(_time_unit) => todo!(),
        arrow_schema::DataType::Duration(_time_unit) => todo!(),
        arrow_schema::DataType::Interval(_interval_unit) => todo!(),
        arrow_schema::DataType::Binary => todo!(),
        arrow_schema::DataType::FixedSizeBinary(_) => todo!(),
        arrow_schema::DataType::LargeBinary => todo!(),
        arrow_schema::DataType::BinaryView => {
            IteratorHolder::View(arr.as_byte_view::<BinaryViewType>().into())
        }

        arrow_schema::DataType::Utf8 => IteratorHolder::String(arr.as_string().into()),
        arrow_schema::DataType::LargeUtf8 => IteratorHolder::LargeString(
            arr.as_any()
                .downcast_ref::<GenericStringArray<i64>>()
                .unwrap()
                .into(),
        ),
        arrow_schema::DataType::Utf8View => {
            IteratorHolder::View(arr.as_byte_view::<StringViewType>().into())
        }
        arrow_schema::DataType::List(_field) => todo!(),
        arrow_schema::DataType::ListView(_field) => todo!(),
        arrow_schema::DataType::FixedSizeList(_field, _) => todo!(),
        arrow_schema::DataType::LargeList(_field) => todo!(),
        arrow_schema::DataType::LargeListView(_field) => todo!(),
        arrow_schema::DataType::Struct(_fields) => todo!(),
        arrow_schema::DataType::Union(_union_fields, _union_mode) => todo!(),
        arrow_schema::DataType::Dictionary(key_type, _value_type) => match key_type.as_ref() {
            arrow_schema::DataType::Int8 => arr.as_dictionary::<Int8Type>().into(),
            arrow_schema::DataType::Int16 => arr.as_dictionary::<Int16Type>().into(),
            arrow_schema::DataType::Int32 => arr.as_dictionary::<Int32Type>().into(),
            arrow_schema::DataType::Int64 => arr.as_dictionary::<Int64Type>().into(),
            arrow_schema::DataType::UInt8 => arr.as_dictionary::<UInt8Type>().into(),
            arrow_schema::DataType::UInt16 => arr.as_dictionary::<UInt16Type>().into(),
            arrow_schema::DataType::UInt32 => arr.as_dictionary::<UInt32Type>().into(),
            arrow_schema::DataType::UInt64 => arr.as_dictionary::<UInt64Type>().into(),
            _ => unreachable!(),
        },
        arrow_schema::DataType::Decimal128(_, _) => todo!(),
        arrow_schema::DataType::Decimal256(_, _) => todo!(),
        arrow_schema::DataType::Map(_field, _) => todo!(),
        arrow_schema::DataType::RunEndEncoded(re, _values) => match re.data_type() {
            DataType::Int16 => arr.as_any().downcast_ref::<Int16RunArray>().unwrap().into(),
            DataType::Int32 => arr.as_any().downcast_ref::<Int32RunArray>().unwrap().into(),
            DataType::Int64 => arr.as_any().downcast_ref::<Int64RunArray>().unwrap().into(),
            _ => unreachable!("invalid run end type"),
        },
    }
}

pub fn datum_to_iter(val: &dyn Datum) -> Result<IteratorHolder, ArrowKernelError> {
    let (d, is_scalar) = val.get();
    if !is_scalar {
        Ok(array_to_iter(d))
    } else {
        match d.data_type() {
            DataType::Int8 => Ok(d.as_primitive::<Int8Type>().value(0).into()),
            DataType::Int16 => Ok(d.as_primitive::<Int16Type>().value(0).into()),
            DataType::Int32 => Ok(d.as_primitive::<Int32Type>().value(0).into()),
            DataType::Int64 => Ok(d.as_primitive::<Int64Type>().value(0).into()),
            DataType::UInt8 => Ok(d.as_primitive::<UInt8Type>().value(0).into()),
            DataType::UInt16 => Ok(d.as_primitive::<UInt16Type>().value(0).into()),
            DataType::UInt32 => Ok(d.as_primitive::<UInt32Type>().value(0).into()),
            DataType::UInt64 => Ok(d.as_primitive::<UInt64Type>().value(0).into()),
            DataType::Float16 => Ok(d.as_primitive::<Float16Type>().value(0).into()),
            DataType::Float32 => Ok(d.as_primitive::<Float32Type>().value(0).into()),
            DataType::Float64 => Ok(d.as_primitive::<Float64Type>().value(0).into()),
            DataType::Timestamp(_time_unit, _) => todo!(),
            DataType::Date32 => todo!(),
            DataType::Date64 => todo!(),
            DataType::Time32(_time_unit) => todo!(),
            DataType::Time64(__time_unit) => todo!(),
            DataType::Duration(_time_unit) => todo!(),
            DataType::Interval(_interval_unit) => todo!(),
            DataType::Binary => todo!(),
            DataType::FixedSizeBinary(_) => todo!(),
            DataType::LargeBinary => todo!(),
            DataType::BinaryView => todo!(),
            DataType::Utf8 => Ok(d
                .as_string::<i32>()
                .value(0)
                .to_owned()
                .into_boxed_str()
                .into()),
            DataType::LargeUtf8 => Ok(d
                .as_string::<i64>()
                .value(0)
                .to_owned()
                .into_boxed_str()
                .into()),
            DataType::Utf8View => todo!(),
            DataType::List(_field) => todo!(),
            DataType::ListView(_field) => todo!(),
            DataType::FixedSizeList(_field, _) => todo!(),
            DataType::LargeList(_field) => todo!(),
            DataType::LargeListView(_field) => todo!(),
            DataType::Struct(_fields) => todo!(),
            DataType::Union(_union_fields, _union_mode) => todo!(),
            DataType::Decimal128(_, _) => todo!(),
            DataType::Decimal256(_, _) => todo!(),
            DataType::Map(_field, _) => todo!(),
            _ => Err(ArrowKernelError::UnsupportedScalar(d.data_type().clone())),
        }
    }
}

/// A holder for a C-style iterator. Created by `array_to_iter`, and used by
/// compiled LLVM functions.
#[derive(Debug)]
pub enum IteratorHolder {
    Primitive(Box<PrimitiveIterator>),
    String(Box<StringIterator>),
    LargeString(Box<LargeStringIterator>),
    View(Box<ViewIterator>),
    Bitmap(Box<BitmapIterator>),
    SetBit(Box<SetBitIterator>),
    Dictionary {
        arr: Box<DictionaryIterator>,
        keys: Box<IteratorHolder>,
        values: Box<IteratorHolder>,
    },
    RunEnd {
        arr: Box<RunEndIterator>,
        run_ends: Box<IteratorHolder>,
        values: Box<IteratorHolder>,
    },
    ScalarPrimitive(Box<ScalarPrimitiveIterator>),
    ScalarString(Box<ScalarStringIterator>),
}

impl IteratorHolder {
    /// Gets a mutable pointer to the inner iterator, suitable for passing to
    /// compiled LLVM functions.
    pub fn get_mut_ptr(&mut self) -> *mut c_void {
        match self {
            IteratorHolder::Primitive(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::String(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::LargeString(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::View(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::Bitmap(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::SetBit(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::Dictionary { arr: iter, .. } => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::RunEnd { arr: iter, .. } => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::ScalarPrimitive(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::ScalarString(iter) => &mut **iter as *mut _ as *mut c_void,
        }
    }

    pub fn as_primitive(&self) -> &PrimitiveIterator {
        match self {
            IteratorHolder::Primitive(iter) => iter,
            _ => panic!("as_primitive called on non-primitive iterator holder"),
        }
    }

    /// Gets a const pointer to the inner iterator, suitable for passing to
    /// compiled LLVM functions.
    pub fn get_ptr(&self) -> *const c_void {
        match self {
            IteratorHolder::Primitive(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::String(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::LargeString(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::View(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::Bitmap(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::SetBit(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::Dictionary { arr: iter, .. } => &**iter as *const _ as *const c_void,
            IteratorHolder::RunEnd { arr: iter, .. } => &**iter as *const _ as *const c_void,
            IteratorHolder::ScalarPrimitive(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::ScalarString(iter) => &**iter as *const _ as *const c_void,
        }
    }
}

/// This adds a `next_block` function to the module for the given iterator. When
/// called, this `next_block` function will fetch `n` elements from the
/// iterator, advancing the iterator's offset. The generated function's signature is:
///
/// fn next_block(iter: ptr, out: ptr_to_vec_of_size_n) -> bool
///
pub fn generate_next_block<'a, const N: u32>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    label: &str,
    dt: &DataType,
    ih: &IteratorHolder,
) -> Option<FunctionValue<'a>> {
    let build = ctx.create_builder();
    let ptype = PrimitiveType::for_arrow_type(dt);
    let vec_type = ptype.llvm_vec_type(ctx, N)?;
    let llvm_type = ptype.llvm_type(ctx);
    let bool_type = ctx.bool_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let i64_type = ctx.i64_type();
    let llvm_n = i64_type.const_int(N as u64, false);

    let fn_type = bool_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
    let next = llvm_mod.add_function(
        &format!("{}_next_block_{}", label, N),
        fn_type,
        Some(
            #[cfg(test)]
            Linkage::External,
            #[cfg(not(test))]
            Linkage::Private,
        ),
    );
    let iter_ptr = next.get_nth_param(0).unwrap().into_pointer_value();
    let out_ptr = next.get_nth_param(1).unwrap().into_pointer_value();

    match ih {
        IteratorHolder::Primitive(primitive_iter) => {
            declare_blocks!(ctx, next, entry, none_left, get_next);

            build.position_at_end(entry);
            let curr_pos = primitive_iter.llvm_pos(ctx, &build, iter_ptr);
            let curr_len = primitive_iter.llvm_len(ctx, &build, iter_ptr);

            let remaining = build
                .build_int_sub(curr_len, curr_pos, "remaining")
                .unwrap();
            let have_enough = build
                .build_int_compare(IntPredicate::UGE, remaining, llvm_n, "have_enough")
                .unwrap();
            build
                .build_conditional_branch(have_enough, get_next, none_left)
                .unwrap();

            build.position_at_end(none_left);
            build
                .build_return(Some(&bool_type.const_int(0, false)))
                .unwrap();

            build.position_at_end(get_next);
            // there are at least n elements left, we can load them and increment
            let data_ptr = primitive_iter.llvm_data(ctx, &build, iter_ptr);
            let data_ptr = increment_pointer!(ctx, build, data_ptr, ptype.width(), curr_pos);
            let vec = build.build_load(vec_type, data_ptr, "vec").unwrap();
            build.build_store(out_ptr, vec).unwrap();
            primitive_iter.llvm_increment_pos(ctx, &build, iter_ptr, llvm_n);
            build
                .build_return(Some(&bool_type.const_int(1, false)))
                .unwrap();

            Some(next)
        }
        IteratorHolder::String(_) | IteratorHolder::LargeString(_) => None,
        IteratorHolder::View(_) => None,
        IteratorHolder::Bitmap(_bitmap_iterator) => todo!(),
        IteratorHolder::SetBit(_) => unimplemented!("No block iterator for setbit!"),
        IteratorHolder::Dictionary { arr, keys, values } => match dt {
            DataType::Dictionary(k_dt, v_dt) => {
                let key_block_next =
                    generate_next_block::<N>(ctx, llvm_mod, &format!("{}_key", label), k_dt, keys)
                        .unwrap();

                let value_access = generate_random_access(
                    ctx,
                    llvm_mod,
                    &format!("{}_value", label),
                    v_dt,
                    values,
                )
                .unwrap();

                declare_blocks!(
                    ctx,
                    next,
                    entry,
                    none_left,
                    get_next,
                    loop_cond_block,
                    loop_body_block,
                    after_loop_block
                );

                build.position_at_end(entry);
                let key_prim_type = PrimitiveType::for_arrow_type(k_dt);
                let key_iter = arr.llvm_key_iter_ptr(ctx, &build, iter_ptr);
                let key_vec_type = key_prim_type.llvm_vec_type(ctx, N).unwrap();
                let key_buf = build.build_alloca(key_vec_type, "key_block").unwrap();
                let key_block_result = build
                    .build_call(
                        key_block_next,
                        &[key_iter.into(), key_buf.into()],
                        "key_block_result",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                build
                    .build_conditional_branch(key_block_result, get_next, none_left)
                    .unwrap();

                build.position_at_end(none_left);
                build
                    .build_return(Some(&bool_type.const_int(0, false)))
                    .unwrap();

                build.position_at_end(get_next);
                let loop_counter_type = ctx.i64_type();
                let loop_counter = build
                    .build_alloca(loop_counter_type, "loop_counter")
                    .unwrap();
                build
                    .build_store(loop_counter, loop_counter_type.const_int(0, false))
                    .unwrap();
                build.build_unconditional_branch(loop_cond_block).unwrap();

                build.position_at_end(loop_cond_block);
                let loop_index = build
                    .build_load(loop_counter_type, loop_counter, "loop_index")
                    .unwrap()
                    .into_int_value();
                let loop_cond = build
                    .build_int_compare(
                        IntPredicate::ULT,
                        loop_index,
                        loop_counter_type.const_int(N as u64, false),
                        "loop_cond",
                    )
                    .unwrap();
                build
                    .build_conditional_branch(loop_cond, loop_body_block, after_loop_block)
                    .unwrap();

                build.position_at_end(loop_body_block);
                let key_vec = build
                    .build_load(key_vec_type, key_buf, "key_vec")
                    .unwrap()
                    .into_vector_value();
                let key_uncast = build
                    .build_extract_element(key_vec, loop_index, "key")
                    .unwrap()
                    .into_int_value();
                let key = build
                    .build_int_cast(key_uncast, i64_type, "key_cast")
                    .unwrap();
                let val_iter = arr.llvm_val_iter_ptr(ctx, &build, iter_ptr);
                let value = build
                    .build_call(value_access, &[val_iter.into(), key.into()], "value")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left();
                let val_prim_type = PrimitiveType::for_arrow_type(v_dt);
                let elem_ptr =
                    increment_pointer!(ctx, &build, out_ptr, val_prim_type.width(), loop_index);
                build.build_store(elem_ptr, value).unwrap();
                let new_index = build
                    .build_int_add(
                        loop_index,
                        loop_counter_type.const_int(1, false),
                        "new_index",
                    )
                    .unwrap();
                build.build_store(loop_counter, new_index).unwrap();
                build.build_unconditional_branch(loop_cond_block).unwrap();

                build.position_at_end(after_loop_block);
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                Some(next)
            }
            _ => unreachable!("dict iterator but not dict data type ({:?})", dt),
        },
        IteratorHolder::RunEnd {
            arr,
            run_ends,
            values,
        } => match dt {
            DataType::RunEndEncoded(re, v) => {
                let access_ends = generate_random_access(
                    ctx,
                    llvm_mod,
                    "ree_block_ends",
                    re.data_type(),
                    run_ends,
                )?;
                let access_values = generate_random_access(
                    ctx,
                    llvm_mod,
                    "ree_block_values",
                    v.data_type(),
                    values,
                )?;

                let umin = Intrinsic::find("llvm.umin").unwrap();
                let umin_f = umin.get_declaration(llvm_mod, &[i64_type.into()]).unwrap();
                declare_blocks!(
                    ctx,
                    next,
                    entry,
                    check_for_next,
                    fetch_next,
                    not_enough,
                    loop_cond,
                    check_vec_full,
                    fill_vec,
                    exit
                );

                build.position_at_end(entry);
                let orig_pos = arr.llvm_pos(ctx, &build, iter_ptr);
                let orig_rem = arr.llvm_remaining(ctx, &build, iter_ptr);
                let ends_iter = arr.llvm_re_iter_ptr(ctx, &build, iter_ptr);
                let vals_iter = arr.llvm_val_iter_ptr(ctx, &build, iter_ptr);
                let vbuf = build.build_alloca(vec_type, "vbuf").unwrap();
                build.build_store(vbuf, vec_type.const_zero()).unwrap();
                let buf_idx_ptr = build.build_alloca(i64_type, "buf_idx").unwrap();
                build
                    .build_store(buf_idx_ptr, i64_type.const_zero())
                    .unwrap();
                build.build_unconditional_branch(loop_cond).unwrap();

                build.position_at_end(check_vec_full);
                let buf_idx = build
                    .build_load(i64_type, buf_idx_ptr, "buf_idx")
                    .unwrap()
                    .into_int_value();
                let res = build
                    .build_int_compare(
                        IntPredicate::ULT,
                        buf_idx,
                        i64_type.const_int(N as u64, false),
                        "vec_full",
                    )
                    .unwrap();
                build
                    .build_conditional_branch(res, loop_cond, exit)
                    .unwrap();

                build.position_at_end(loop_cond);
                let remaining = arr.llvm_remaining(ctx, &build, iter_ptr);
                let res = build
                    .build_int_compare(
                        IntPredicate::UGT,
                        remaining,
                        i64_type.const_zero(),
                        "have_remaining",
                    )
                    .unwrap();
                build
                    .build_conditional_branch(res, fill_vec, check_for_next)
                    .unwrap();

                build.position_at_end(check_for_next);
                arr.llvm_inc_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                let pos = arr.llvm_pos(ctx, &build, iter_ptr);
                let len = arr.llvm_len(ctx, &build, iter_ptr);
                let res = build
                    .build_int_compare(IntPredicate::ULT, pos, len, "have_next")
                    .unwrap();
                build
                    .build_conditional_branch(res, fetch_next, not_enough)
                    .unwrap();

                build.position_at_end(not_enough);
                arr.llvm_set_remaining(ctx, &build, iter_ptr, orig_rem);
                arr.llvm_set_pos(ctx, &build, iter_ptr, orig_pos);
                build.build_return(Some(&bool_type.const_zero())).unwrap();

                build.position_at_end(fetch_next);
                let my_end = build
                    .build_call(access_ends, &[ends_iter.into(), pos.into()], "my_end")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let pr_end = build
                    .build_call(
                        access_ends,
                        &[
                            ends_iter.into(),
                            build
                                .build_int_sub(pos, i64_type.const_int(1, false), "minus1")
                                .unwrap()
                                .into(),
                        ],
                        "pr_end",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let remaining = build.build_int_sub(my_end, pr_end, "remaining").unwrap();
                arr.llvm_set_remaining(ctx, &build, iter_ptr, remaining);
                build.build_unconditional_branch(loop_cond).unwrap();

                build.position_at_end(fill_vec);
                let pos = arr.llvm_pos(ctx, &build, iter_ptr);
                let curr_val = build
                    .build_call(access_values, &[vals_iter.into(), pos.into()], "val")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left();
                let curr_val = build
                    .build_insert_element(
                        vec_type.const_zero(),
                        curr_val,
                        i64_type.const_zero(),
                        "curr_val_vec",
                    )
                    .unwrap();
                let curr_val = build
                    .build_shuffle_vector(
                        curr_val,
                        vec_type.get_undef(),
                        vec_type.const_zero(),
                        "broadcasted",
                    )
                    .unwrap();

                let remaining_values = arr.llvm_remaining(ctx, &build, iter_ptr);
                let buf_idx = build
                    .build_load(i64_type, buf_idx_ptr, "buf_idx")
                    .unwrap()
                    .into_int_value();
                let remaining_slots = build
                    .build_int_sub(
                        i64_type.const_int(N as u64, false),
                        buf_idx,
                        "remaining_slots",
                    )
                    .unwrap();
                let to_fill = build
                    .build_call(
                        umin_f,
                        &[remaining_slots.into(), remaining_values.into()],
                        "to_fill",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();

                // build a mask that 1 in the slots we want to insert value into
                // ex: suppose block size = 8, to_fill = 2, curr_pos = 3
                // desired mask: 00011000
                // formula: ((1 << to_fill) - 1) << curr_pos
                //
                let mask = build
                    .build_left_shift(
                        build
                            .build_int_sub(
                                build
                                    .build_left_shift(i64_type.const_int(1, false), to_fill, "mask")
                                    .unwrap(),
                                i64_type.const_int(1, false),
                                "mask",
                            )
                            .unwrap(),
                        buf_idx,
                        "mask",
                    )
                    .unwrap();
                let max_width_cond = build
                    .build_int_compare(
                        IntPredicate::EQ,
                        to_fill,
                        i64_type.const_int(N as u64, false),
                        "is_full",
                    )
                    .unwrap();
                let mask = build
                    .build_select(max_width_cond, i64_type.const_all_ones(), mask, "mask")
                    .unwrap()
                    .into_int_value();

                let mask = match N {
                    8 => build
                        .build_int_truncate(mask, ctx.i8_type(), "mask")
                        .unwrap(),
                    16 => build
                        .build_int_truncate(mask, ctx.i16_type(), "mask")
                        .unwrap(),
                    32 => build
                        .build_int_truncate(mask, ctx.i32_type(), "mask")
                        .unwrap(),
                    64 => mask,
                    _ => return None,
                };

                let mask = build
                    .build_bit_cast(mask, bool_type.vec_type(N), "mask_v")
                    .unwrap()
                    .into_vector_value();

                let curr_buf = build
                    .build_load(vec_type, vbuf, "curr_buf")
                    .unwrap()
                    .into_vector_value();

                let new_buf = build
                    .build_select(mask, curr_val, curr_buf, "new_buf")
                    .unwrap();

                build.build_store(vbuf, new_buf).unwrap();
                arr.llvm_dec_remaining(ctx, &build, iter_ptr, to_fill);
                let new_buf_ptr = build
                    .build_int_add(buf_idx, to_fill, "new_buf_ptr")
                    .unwrap();
                build.build_store(buf_idx_ptr, new_buf_ptr).unwrap();
                build.build_unconditional_branch(check_vec_full).unwrap();

                build.position_at_end(exit);
                let result = build.build_load(vec_type, vbuf, "result").unwrap();
                build.build_store(out_ptr, result).unwrap();
                build
                    .build_return(Some(&bool_type.const_all_ones()))
                    .unwrap();

                Some(next)
            }
            _ => unreachable!("run-end iterator but not run-end data type ({:?})", dt),
        },
        IteratorHolder::ScalarPrimitive(s) => {
            assert_eq!(ptype.width(), s.width as usize);
            let get_next_single = generate_next(
                ctx,
                llvm_mod,
                &format!("scaler_block_single_next_{}", label),
                dt,
                ih,
            )?;
            declare_blocks!(ctx, next, entry);
            build.position_at_end(entry);
            let val_buf = build.build_alloca(llvm_type, "val_buf").unwrap();
            build
                .build_call(get_next_single, &[iter_ptr.into(), val_buf.into()], "get")
                .unwrap();
            let constant = build.build_load(llvm_type, val_buf, "constant").unwrap();
            constant
                .as_instruction_value()
                .unwrap()
                .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                .unwrap();
            let v = build
                .build_insert_element(vec_type.const_zero(), constant, i64_type.const_zero(), "v")
                .unwrap();
            let v = build
                .build_shuffle_vector(v, vec_type.const_zero(), vec_type.const_zero(), "splatted")
                .unwrap();
            build.build_store(out_ptr, v).unwrap();
            build
                .build_return(Some(&bool_type.const_int(1, false)))
                .unwrap();
            Some(next)
        }
        IteratorHolder::ScalarString(_) => todo!(),
    }
}

/// This adds a `next` function to the module for the given iterator. When
/// called, this `next` function will fetch the next element from the
/// iterator, advancing the iterator's offset. The generated function's signature is:
/// fn next(iter: ptr, out: ptr_to_el) -> bool
///
pub fn generate_next<'a>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    label: &str,
    dt: &DataType,
    ih: &IteratorHolder,
) -> Option<FunctionValue<'a>> {
    let build = ctx.create_builder();
    let ptype = PrimitiveType::for_arrow_type(dt);
    let llvm_type = ptype.llvm_type(ctx);
    let bool_type = ctx.bool_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let i64_type = ctx.i64_type();

    let fn_type = bool_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
    let next = llvm_mod.add_function(
        &format!("{}_next", label),
        fn_type,
        Some(
            #[cfg(test)]
            Linkage::External,
            #[cfg(not(test))]
            Linkage::Private,
        ),
    );
    let iter_ptr = next.get_nth_param(0).unwrap().into_pointer_value();
    let out_ptr = next.get_nth_param(1).unwrap().into_pointer_value();

    match ih {
        IteratorHolder::Primitive(primitive_iter) => {
            declare_blocks!(ctx, next, entry, none_left, get_next);

            build.position_at_end(entry);
            let curr_pos = primitive_iter.llvm_pos(ctx, &build, iter_ptr);
            let curr_len = primitive_iter.llvm_len(ctx, &build, iter_ptr);
            let have_more = build
                .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                .unwrap();
            build
                .build_conditional_branch(have_more, get_next, none_left)
                .unwrap();

            build.position_at_end(none_left);
            build
                .build_return(Some(&bool_type.const_int(0, false)))
                .unwrap();

            build.position_at_end(get_next);
            // there are at least n elements left, we can load them and increment
            let data_ptr = primitive_iter.llvm_data(ctx, &build, iter_ptr);
            let data_ptr = increment_pointer!(ctx, build, data_ptr, ptype.width(), curr_pos);
            let out = build.build_load(llvm_type, data_ptr, "elem").unwrap();
            build.build_store(out_ptr, out).unwrap();
            primitive_iter.llvm_increment_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
            build
                .build_return(Some(&bool_type.const_int(1, false)))
                .unwrap();

            Some(next)
        }
        IteratorHolder::String(iter) => {
            let access = generate_random_access(ctx, llvm_mod, label, dt, ih).unwrap();
            declare_blocks!(ctx, next, entry, none_left, get_next);

            build.position_at_end(entry);
            let curr_pos = iter.llvm_pos(ctx, &build, iter_ptr);
            let curr_len = iter.llvm_len(ctx, &build, iter_ptr);
            let have_more = build
                .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                .unwrap();
            build
                .build_conditional_branch(have_more, get_next, none_left)
                .unwrap();

            build.position_at_end(none_left);
            build
                .build_return(Some(&bool_type.const_int(0, false)))
                .unwrap();

            build.position_at_end(get_next);
            // there are at least n elements left, we can load them and increment
            let result = build
                .build_call(access, &[iter_ptr.into(), curr_pos.into()], "access_result")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left();

            build.build_store(out_ptr, result).unwrap();
            iter.llvm_increment_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
            build
                .build_return(Some(&bool_type.const_int(1, false)))
                .unwrap();

            Some(next)
        }
        IteratorHolder::LargeString(iter) => {
            let access = generate_random_access(ctx, llvm_mod, label, dt, ih).unwrap();
            declare_blocks!(ctx, next, entry, none_left, get_next);

            build.position_at_end(entry);
            let curr_pos = iter.llvm_pos(ctx, &build, iter_ptr);
            let curr_len = iter.llvm_len(ctx, &build, iter_ptr);
            let have_more = build
                .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                .unwrap();
            build
                .build_conditional_branch(have_more, get_next, none_left)
                .unwrap();

            build.position_at_end(none_left);
            build
                .build_return(Some(&bool_type.const_int(0, false)))
                .unwrap();

            build.position_at_end(get_next);
            // there are at least n elements left, we can load them and increment
            let result = build
                .build_call(access, &[iter_ptr.into(), curr_pos.into()], "access_result")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left();

            build.build_store(out_ptr, result).unwrap();
            iter.llvm_increment_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
            build
                .build_return(Some(&bool_type.const_int(1, false)))
                .unwrap();

            Some(next)
        }
        IteratorHolder::View(iter) => {
            let access = generate_random_access(ctx, llvm_mod, label, dt, ih).unwrap();
            declare_blocks!(ctx, next, entry, none_left, get_next);

            build.position_at_end(entry);
            let curr_pos = iter.llvm_pos(ctx, &build, iter_ptr);
            let curr_len = iter.llvm_len(ctx, &build, iter_ptr);
            let have_more = build
                .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                .unwrap();
            build
                .build_conditional_branch(have_more, get_next, none_left)
                .unwrap();

            build.position_at_end(none_left);
            build
                .build_return(Some(&bool_type.const_int(0, false)))
                .unwrap();

            build.position_at_end(get_next);
            // there are at least n elements left, we can load them and increment
            let result = build
                .build_call(access, &[iter_ptr.into(), curr_pos.into()], "access_result")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left();

            build.build_store(out_ptr, result).unwrap();
            iter.llvm_increment_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
            build
                .build_return(Some(&bool_type.const_int(1, false)))
                .unwrap();

            Some(next)
        }
        IteratorHolder::Bitmap(bitmap_iterator) => {
            let access = generate_random_access(ctx, llvm_mod, label, dt, ih).unwrap();
            declare_blocks!(ctx, next, entry, none_left, get_next);

            build.position_at_end(entry);
            let curr_pos = bitmap_iterator.llvm_pos(ctx, &build, iter_ptr);
            let curr_len = bitmap_iterator.llvm_len(ctx, &build, iter_ptr);
            let have_more = build
                .build_int_compare(IntPredicate::ULT, curr_pos, curr_len, "have_enough")
                .unwrap();
            build
                .build_conditional_branch(have_more, get_next, none_left)
                .unwrap();

            build.position_at_end(none_left);
            build
                .build_return(Some(&bool_type.const_int(0, false)))
                .unwrap();

            build.position_at_end(get_next);
            let result = build
                .build_call(access, &[iter_ptr.into(), curr_pos.into()], "access_result")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left();
            build.build_store(out_ptr, result).unwrap();
            bitmap_iterator.llvm_increment_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
            build
                .build_return(Some(&bool_type.const_int(1, false)))
                .unwrap();

            Some(next)
        }
        IteratorHolder::SetBit(setbit_iterator) => {
            declare_blocks!(
                ctx,
                next,
                entry,
                while_loop,
                get_next_byte_loop,
                none_left,
                get_next,
                use_byte
            );

            let cttz_id = Intrinsic::find("llvm.cttz").expect("llvm.cttz not in Intrinsic list");
            cttz_id
                .get_declaration(llvm_mod, &[ctx.i8_type().into()])
                .expect("Couldn't declare llvm.cttz.i8");

            let cttz_i8 = llvm_mod
                .get_function("llvm.cttz.i8")
                .expect("llvm.cttz.i8 should have been declared in the CodeGen constructor");

            build.position_at_end(entry);
            build.build_unconditional_branch(while_loop).unwrap();

            build.position_at_end(while_loop);
            let curr_pos = setbit_iterator.llvm_get_pos(ctx, &build, iter_ptr);
            let curr_byte = setbit_iterator.llvm_get_byte(ctx, &build, iter_ptr);
            let cmp_to_zero = build
                .build_int_compare(
                    IntPredicate::EQ,
                    curr_byte,
                    ctx.i8_type().const_int(0, false),
                    "cmp_byte_to_zero",
                )
                .unwrap();
            build
                .build_conditional_branch(cmp_to_zero, get_next_byte_loop, use_byte)
                .unwrap();

            build.position_at_end(get_next_byte_loop);
            let bit_len = setbit_iterator.llvm_get_len(ctx, &build, iter_ptr);
            let bytes_len = build
                .build_int_add(bit_len, i64_type.const_int(7, false), "round_up")
                .unwrap();
            let bytes_len = build
                .build_right_shift(
                    bytes_len,
                    i64_type.const_int(3, false),
                    false,
                    "divide_by_8",
                )
                .unwrap();
            let cmp_pos_to_len = build
                .build_int_compare(IntPredicate::EQ, curr_pos, bytes_len, "cmp_pos_to_len")
                .unwrap();
            build
                .build_conditional_branch(cmp_pos_to_len, none_left, get_next)
                .unwrap();

            build.position_at_end(none_left);
            build
                .build_return(Some(&bool_type.const_int(0, false)))
                .unwrap();

            build.position_at_end(get_next);
            setbit_iterator.llvm_load_byte_at_pos(ctx, &build, iter_ptr);
            setbit_iterator.llvm_increment_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
            build.build_unconditional_branch(while_loop).unwrap();

            build.position_at_end(use_byte);

            let is_zero_undef = ctx.bool_type().const_int(0, false);
            let tz_i8 = build
                .build_call(
                    cttz_i8,
                    &[curr_byte.into(), is_zero_undef.into()],
                    "cttz_i8",
                )
                .unwrap()
                .try_as_basic_value()
                .left()
                .expect("cttz should return a value")
                .into_int_value();
            let tz_i64 = build
                .build_int_z_extend(tz_i8, i64_type, "extend_tz")
                .unwrap();
            setbit_iterator.llvm_clear_trailing_bit(ctx, &build, iter_ptr);
            let setbit_position = build
                .build_int_nuw_sub(curr_pos, i64_type.const_int(1, false), "pos_minus_one")
                .unwrap();
            let setbit_position = build
                .build_int_mul(setbit_position, i64_type.const_int(8, false), "pos_mul_8")
                .unwrap();
            let setbit_position = build
                .build_int_add(setbit_position, tz_i64, "add_tz")
                .unwrap();
            build.build_store(out_ptr, setbit_position).unwrap();
            build
                .build_return(Some(&bool_type.const_int(1, false)))
                .unwrap();

            Some(next)
        }
        IteratorHolder::Dictionary { arr, keys, values } => match dt {
            DataType::Dictionary(k_dt, v_dt) => {
                let key_next =
                    generate_next(ctx, llvm_mod, &format!("{}_key", label), k_dt, keys).unwrap();
                let values_access = generate_random_access(
                    ctx,
                    llvm_mod,
                    &format!("{}_value", label),
                    v_dt,
                    values,
                )
                .unwrap();
                declare_blocks!(ctx, next, entry, none_left, fetch);

                build.position_at_end(entry);
                let key_type = PrimitiveType::for_arrow_type(k_dt).llvm_type(ctx);
                let key_iter = arr.llvm_key_iter_ptr(ctx, &build, iter_ptr);
                let val_iter = arr.llvm_val_iter_ptr(ctx, &build, iter_ptr);

                let key_buf = build.build_alloca(key_type, "key_buf").unwrap();
                let had_next = build
                    .build_call(key_next, &[key_iter.into(), key_buf.into()], "next_key")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                build
                    .build_conditional_branch(had_next, fetch, none_left)
                    .unwrap();

                build.position_at_end(fetch);
                let next_key = build
                    .build_int_cast(
                        build
                            .build_load(key_type, key_buf, "next_key")
                            .unwrap()
                            .into_int_value(),
                        i64_type,
                        "casted_key",
                    )
                    .unwrap();
                let value = build
                    .build_call(values_access, &[val_iter.into(), next_key.into()], "value")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left();
                build.build_store(out_ptr, value).unwrap();
                build
                    .build_return(Some(&bool_type.const_int(1, false)))
                    .unwrap();

                build.position_at_end(none_left);
                build
                    .build_return(Some(&bool_type.const_int(0, false)))
                    .unwrap();
                Some(next)
            }
            _ => unreachable!("dict iterator but not dict data type ({:?})", dt),
        },
        IteratorHolder::RunEnd {
            arr,
            run_ends,
            values,
        } => match dt {
            DataType::RunEndEncoded(res, data) => {
                let re_access = generate_random_access(
                    ctx,
                    llvm_mod,
                    &format!("re_access_{}", label),
                    res.data_type(),
                    run_ends,
                )
                .unwrap();
                let val_access = generate_random_access(
                    ctx,
                    llvm_mod,
                    &format!("val_access_{}", label),
                    data.data_type(),
                    values,
                )
                .unwrap();

                declare_blocks!(
                    ctx,
                    next,
                    entry,
                    check_remaining,
                    has_remaining,
                    none_remaining,
                    load_next_run,
                    exhausted
                );

                build.position_at_end(entry);
                let val_iter_ptr = arr.llvm_val_iter_ptr(ctx, &build, iter_ptr);
                let re_iter_ptr = arr.llvm_re_iter_ptr(ctx, &build, iter_ptr);
                build.build_unconditional_branch(check_remaining).unwrap();

                build.position_at_end(check_remaining);
                let remaining = arr.llvm_remaining(ctx, &build, iter_ptr);
                let res = build
                    .build_int_compare(
                        IntPredicate::UGT,
                        remaining,
                        i64_type.const_zero(),
                        "has_remaining",
                    )
                    .unwrap();
                build
                    .build_conditional_branch(res, has_remaining, none_remaining)
                    .unwrap();

                build.position_at_end(has_remaining);
                // there are values left in the current run
                let curr_pos = arr.llvm_pos(ctx, &build, iter_ptr);
                let val = build
                    .build_call(val_access, &[val_iter_ptr.into(), curr_pos.into()], "value")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left();
                build.build_store(out_ptr, val).unwrap();
                arr.llvm_dec_remaining(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                build
                    .build_return(Some(&bool_type.const_all_ones()))
                    .unwrap();

                build.position_at_end(none_remaining);
                // there are no values left in the current run -- either load a
                // new run, or return false
                arr.llvm_inc_pos(ctx, &build, iter_ptr, i64_type.const_int(1, false));
                let curr_pos = arr.llvm_pos(ctx, &build, iter_ptr);
                let re_len = arr.llvm_len(ctx, &build, iter_ptr);
                let have_another_run = build
                    .build_int_compare(IntPredicate::ULT, curr_pos, re_len, "another_run")
                    .unwrap();
                build
                    .build_conditional_branch(have_another_run, load_next_run, exhausted)
                    .unwrap();

                build.position_at_end(load_next_run);
                let curr_pos = arr.llvm_pos(ctx, &build, iter_ptr);
                let my_end = build
                    .build_call(re_access, &[re_iter_ptr.into(), curr_pos.into()], "my_end")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let prev_end = build
                    .build_call(
                        re_access,
                        &[
                            re_iter_ptr.into(),
                            build
                                .build_int_sub(curr_pos, i64_type.const_int(1, false), "prev_pos")
                                .unwrap()
                                .into(),
                        ],
                        "prev_end",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let new_remaining = build
                    .build_int_sub(my_end, prev_end, "new_remaining")
                    .unwrap();
                arr.llvm_set_remaining(ctx, &build, iter_ptr, new_remaining);
                build.build_unconditional_branch(check_remaining).unwrap();

                build.position_at_end(exhausted);
                build.build_return(Some(&bool_type.const_zero())).unwrap();

                Some(next)
            }
            _ => unreachable!("run-end iterator but not run-end data type ({:?})", dt),
        },
        IteratorHolder::ScalarPrimitive(s) => {
            assert_eq!(ptype.width(), s.width as usize);
            declare_blocks!(ctx, next, entry);
            build.position_at_end(entry);
            let ptr = s.llvm_val_ptr(ctx, &build, iter_ptr);
            let constant = match dt {
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64 => {
                    let data = build
                        .build_load(i64_type, ptr, "const_u64")
                        .unwrap()
                        .into_int_value();
                    data.as_instruction_value()
                        .unwrap()
                        .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                        .unwrap();
                    build
                        .build_int_cast(data, llvm_type.into_int_type(), "casted")
                        .unwrap()
                        .as_basic_value_enum()
                }
                DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                    let data = build
                        .build_load(i64_type, ptr, "const_u64")
                        .unwrap()
                        .into_int_value();
                    data.as_instruction_value()
                        .unwrap()
                        .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                        .unwrap();
                    let data = build
                        .build_int_cast(
                            data,
                            PrimitiveType::int_with_width(s.width as usize)
                                .llvm_type(ctx)
                                .into_int_type(),
                            "const_int",
                        )
                        .unwrap();
                    build
                        .build_bit_cast(data, llvm_type, "const_float")
                        .unwrap()
                }
                _ => unreachable!(),
            };
            build.build_store(out_ptr, constant).unwrap();
            build
                .build_return(Some(&bool_type.const_int(1, false)))
                .unwrap();
            Some(next)
        }
        IteratorHolder::ScalarString(s) => {
            let ptr_type = ctx.ptr_type(AddressSpace::default());
            let ret_type = PrimitiveType::P64x2.llvm_type(ctx).into_struct_type();
            declare_blocks!(ctx, next, entry);
            build.position_at_end(entry);
            let (ptr1, ptr2) = s.llvm_val_ptr(ctx, &build, iter_ptr);
            let ptr1 = build
                .build_load(ptr_type, ptr1, "ptr1")
                .unwrap()
                .into_pointer_value();
            ptr1.as_instruction_value()
                .unwrap()
                .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                .unwrap();
            let ptr2 = build
                .build_load(ptr_type, ptr2, "ptr2")
                .unwrap()
                .into_pointer_value();
            ptr2.as_instruction_value()
                .unwrap()
                .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
                .unwrap();

            let to_return = ret_type.const_zero();
            let to_return = build
                .build_insert_value(to_return, ptr1, 0, "to_return")
                .unwrap();
            let to_return = build
                .build_insert_value(to_return, ptr2, 1, "to_return")
                .unwrap();
            build.build_store(out_ptr, to_return).unwrap();
            build
                .build_return(Some(&bool_type.const_int(1, false)))
                .unwrap();

            Some(next)
        }
    }
}

/// This adds an `access` function to the module for the given iterator. When
/// called, this `access` function will fetch an element from the iterator at
/// the position given by the 2nd parameter, *without* advancing the iterator's
/// offset. Indexing is done from the base data, not from the iterator's current
/// position. The generated function's signature is:
///
/// fn access(iter: ptr, el: u64) -> T
///
pub fn generate_random_access<'a>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    label: &str,
    dt: &DataType,
    ih: &IteratorHolder,
) -> Option<FunctionValue<'a>> {
    let build = ctx.create_builder();
    let ptype = PrimitiveType::for_arrow_type(dt);
    let llvm_type = ptype.llvm_type(ctx);
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let i64_type = ctx.i64_type();

    let fn_type = llvm_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
    let access_f = llvm_mod.add_function(
        &format!("{}_access", label),
        fn_type,
        Some(
            #[cfg(test)]
            Linkage::External,
            #[cfg(not(test))]
            Linkage::Private,
        ),
    );
    let iter_ptr = access_f.get_nth_param(0).unwrap().into_pointer_value();
    let idx = access_f.get_nth_param(1).unwrap().into_int_value();

    match ih {
        IteratorHolder::Primitive(primitive_iter) => {
            declare_blocks!(ctx, access_f, entry);

            build.position_at_end(entry);
            let data_ptr = primitive_iter.llvm_data(ctx, &build, iter_ptr);
            let data_ptr = increment_pointer!(ctx, build, data_ptr, ptype.width(), idx);
            let out = build.build_load(llvm_type, data_ptr, "elem").unwrap();
            build.build_return(Some(&out)).unwrap();

            Some(access_f)
        }
        IteratorHolder::String(ih) => {
            let i32_type = ctx.i32_type();
            let ret_type = PrimitiveType::P64x2.llvm_type(ctx).into_struct_type();

            declare_blocks!(ctx, access_f, entry);

            build.position_at_end(entry);
            let offsets = ih.llvm_get_offset_ptr(ctx, &build, iter_ptr);
            let offset1 = build
                .build_load(
                    i32_type,
                    increment_pointer!(ctx, build, offsets, 4, idx),
                    "offset1",
                )
                .unwrap()
                .into_int_value();
            let offset2 = build
                .build_load(
                    i32_type,
                    increment_pointer!(
                        ctx,
                        build,
                        offsets,
                        4,
                        build
                            .build_int_add(idx, i64_type.const_int(1, false), "inc")
                            .unwrap()
                    ),
                    "offset2",
                )
                .unwrap()
                .into_int_value();

            let offset1 = build
                .build_int_z_extend(offset1, i64_type, "offset1")
                .unwrap();
            let offset2 = build
                .build_int_z_extend(offset2, i64_type, "offset1")
                .unwrap();

            let data = ih.llvm_get_data_ptr(ctx, &build, iter_ptr);
            let ptr1 = increment_pointer!(ctx, build, data, 1, offset1);
            let ptr2 = increment_pointer!(ctx, build, data, 1, offset2);
            let to_return = ret_type.const_zero();
            let to_return = build
                .build_insert_value(to_return, ptr1, 0, "to_return")
                .unwrap();
            let to_return = build
                .build_insert_value(to_return, ptr2, 1, "to_return")
                .unwrap();
            build.build_return(Some(&to_return)).unwrap();
            Some(access_f)
        }
        IteratorHolder::LargeString(ih) => {
            let ret_type = PrimitiveType::P64x2.llvm_type(ctx).into_struct_type();

            declare_blocks!(ctx, access_f, entry);

            build.position_at_end(entry);
            let offsets = ih.llvm_get_offset_ptr(ctx, &build, iter_ptr);
            let offset1 = build
                .build_load(
                    i64_type,
                    increment_pointer!(ctx, build, offsets, 8, idx),
                    "offset1",
                )
                .unwrap()
                .into_int_value();
            let offset2 = build
                .build_load(
                    i64_type,
                    increment_pointer!(
                        ctx,
                        build,
                        offsets,
                        8,
                        build
                            .build_int_add(idx, i64_type.const_int(1, false), "inc")
                            .unwrap()
                    ),
                    "offset2",
                )
                .unwrap()
                .into_int_value();

            let data = ih.llvm_get_data_ptr(ctx, &build, iter_ptr);
            let ptr1 = increment_pointer!(ctx, build, data, 1, offset1);
            let ptr2 = increment_pointer!(ctx, build, data, 1, offset2);
            let to_return = ret_type.const_zero();
            let to_return = build
                .build_insert_value(to_return, ptr1, 0, "to_return")
                .unwrap();
            let to_return = build
                .build_insert_value(to_return, ptr2, 1, "to_return")
                .unwrap();
            build.build_return(Some(&to_return)).unwrap();
            Some(access_f)
        }
        IteratorHolder::View(iter) => {
            let i128_type = ctx.i128_type();
            let i32_type = ctx.i32_type();
            let ret_type = PrimitiveType::P64x2.llvm_type(ctx).into_struct_type();

            declare_blocks!(ctx, access_f, entry, short_str, long_str);

            build.position_at_end(entry);
            let view_ptr = iter.llvm_view_ptr(ctx, &build, iter_ptr);
            let our_view_ptr = increment_pointer!(ctx, build, view_ptr, 16, idx);
            let our_view = build
                .build_load(i128_type, our_view_ptr, "our_view")
                .unwrap()
                .into_int_value();
            let as_vec = build
                .build_bit_cast(our_view, i32_type.vec_type(4), "as_vec")
                .unwrap()
                .into_vector_value();
            let str_len = build
                .build_extract_element(as_vec, i64_type.const_zero(), "str_len")
                .unwrap()
                .into_int_value();
            let str_len = build
                .build_int_s_extend(str_len, i64_type, "ext_len")
                .unwrap();
            let cmp = build
                .build_int_compare(
                    IntPredicate::SLE,
                    str_len,
                    i64_type.const_int(12, true),
                    "cmp",
                )
                .unwrap();
            build
                .build_conditional_branch(cmp, short_str, long_str)
                .unwrap();

            build.position_at_end(short_str);
            let ptr1 = increment_pointer!(ctx, build, our_view_ptr, 4);
            let ptr2 = increment_pointer!(ctx, build, ptr1, 1, str_len);
            let to_return = ret_type.const_zero();
            let to_return = build
                .build_insert_value(to_return, ptr1, 0, "to_return")
                .unwrap();
            let to_return = build
                .build_insert_value(to_return, ptr2, 1, "to_return")
                .unwrap();
            build.build_return(Some(&to_return)).unwrap();

            build.position_at_end(long_str);
            let buf_idx = build
                .build_extract_element(as_vec, i64_type.const_int(2, false), "buf_idx")
                .unwrap()
                .into_int_value();
            let buf_base = iter.llvm_buffer_ptr(ctx, &build, iter_ptr, buf_idx);
            let offset = build
                .build_extract_element(as_vec, i64_type.const_int(3, false), "offset")
                .unwrap()
                .into_int_value();
            let offset = build
                .build_int_s_extend_or_bit_cast(offset, i64_type, "offset_ext")
                .unwrap();
            let ptr1 = increment_pointer!(ctx, build, buf_base, 1, offset);
            let ptr2 = increment_pointer!(ctx, build, ptr1, 1, str_len);
            let to_return = ret_type.const_zero();
            let to_return = build
                .build_insert_value(to_return, ptr1, 0, "to_return")
                .unwrap();
            let to_return = build
                .build_insert_value(to_return, ptr2, 1, "to_return")
                .unwrap();
            build.build_return(Some(&to_return)).unwrap();

            Some(access_f)
        }
        IteratorHolder::Bitmap(bitmap_iterator) => {
            declare_blocks!(ctx, access_f, entry);

            build.position_at_end(entry);
            let data_ptr = bitmap_iterator.llvm_get_data_ptr(ctx, &build, iter_ptr);
            let byte_index = build
                .build_right_shift(idx, i64_type.const_int(3, false), false, "byte_index")
                .unwrap();
            let bit_in_byte_i64 = build
                .build_and(idx, i64_type.const_int(7, false), "bit_in_byte_i64")
                .unwrap();
            let bit_in_byte_i8 = build
                .build_int_truncate(bit_in_byte_i64, ctx.i8_type(), "bit_in_byte_i8")
                .unwrap();
            let data_byte_ptr = increment_pointer!(ctx, build, data_ptr, ptype.width(), byte_index);
            let data_byte_i8 = build
                .build_load(ctx.i8_type(), data_byte_ptr, "get_data_byte_i8")
                .unwrap()
                .into_int_value();
            let data_byte_shifted_i8 = build
                .build_right_shift(data_byte_i8, bit_in_byte_i8, false, "data_byte_shifted_i8")
                .unwrap();
            let data_bit_i8 = build
                .build_and(
                    data_byte_shifted_i8,
                    ctx.i8_type().const_int(1, false),
                    "data_bit_i8",
                )
                .unwrap();
            build.build_return(Some(&data_bit_i8)).unwrap();

            Some(access_f)
        }
        IteratorHolder::SetBit(_) => None,
        IteratorHolder::Dictionary { arr, keys, values } => match dt {
            DataType::Dictionary(k_dt, v_dt) => {
                let keys_access = generate_random_access(
                    ctx,
                    llvm_mod,
                    &format!("{}_key_get", label),
                    k_dt,
                    keys,
                )?;
                let values_access = generate_random_access(
                    ctx,
                    llvm_mod,
                    &format!("{}_value_get", label),
                    v_dt,
                    values,
                )?;

                declare_blocks!(ctx, access_f, entry);

                build.position_at_end(entry);
                let key = build
                    .build_call(
                        keys_access,
                        &[
                            arr.llvm_key_iter_ptr(ctx, &build, iter_ptr).into(),
                            idx.into(),
                        ],
                        "key",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let key_conv = build
                    .build_int_cast(key, ctx.i64_type(), "key_conv")
                    .unwrap();

                let value = build
                    .build_call(
                        values_access,
                        &[
                            arr.llvm_val_iter_ptr(ctx, &build, iter_ptr).into(),
                            key_conv.into(),
                        ],
                        "value",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left();
                build.build_return(Some(&value)).unwrap();

                Some(access_f)
            }
            _ => unreachable!("dictionary iterator but non-iterator data type ({:?})", dt),
        },
        IteratorHolder::RunEnd {
            arr,
            run_ends,
            values,
        } => match dt {
            DataType::RunEndEncoded(r_dt, v_dt) => {
                let runs_prim_type = PrimitiveType::for_arrow_type(r_dt.data_type());
                let runs_t = runs_prim_type.llvm_type(ctx).into_int_type();
                let bsearch = add_bsearch(ctx, llvm_mod, run_ends.as_primitive(), runs_prim_type);
                let value_access = generate_random_access(
                    ctx,
                    llvm_mod,
                    &format!("{}_val_get", label),
                    v_dt.data_type(),
                    values,
                )?;

                declare_blocks!(ctx, access_f, entry);
                build.position_at_end(entry);
                let idx = build
                    .build_int_truncate_or_bit_cast(idx, runs_t, "casted_idx")
                    .unwrap();

                let v_idx = build
                    .build_call(
                        bsearch,
                        &[
                            arr.llvm_re_iter_ptr(ctx, &build, iter_ptr).into(),
                            idx.into(),
                        ],
                        "v_idx",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let v = build
                    .build_call(
                        value_access,
                        &[
                            arr.llvm_val_iter_ptr(ctx, &build, iter_ptr).into(),
                            v_idx.into(),
                        ],
                        "v",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left();
                build.build_return(Some(&v)).unwrap();
                Some(access_f)
            }
            _ => unreachable!("run-end iterator but non-iterator data type ({:?})", dt),
        },
        IteratorHolder::ScalarPrimitive(_s) => todo!(),
        IteratorHolder::ScalarString(_) => todo!(),
    }
}
