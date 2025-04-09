use std::ffi::c_void;

use arrow_array::{
    cast::AsArray,
    types::{
        ArrowDictionaryKeyType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type,
        Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrowPrimitiveType, BooleanArray, Datum, DictionaryArray, GenericStringArray,
    PrimitiveArray, StringArray,
};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    module::{Linkage, Module},
    types::BasicType,
    values::{FunctionValue, IntValue, PointerValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};
use ouroboros::self_referencing;
use repr_offset::ReprOffset;

use crate::{declare_blocks, PrimitiveType};

macro_rules! increment_pointer {
    ($ctx: expr, $builder: expr, $ptr: expr, $offset: expr) => {
        unsafe {
            $builder
                .build_gep(
                    $ctx.i8_type(),
                    $ptr,
                    &[$ctx.i64_type().const_int($offset as u64, false)],
                    "inc_ptr",
                )
                .unwrap()
        }
    };
    ($ctx: expr, $builder: expr, $ptr: expr, $stride: expr, $offset: expr) => {
        unsafe {
            let i64_type = $ctx.i64_type();
            let tmp = $builder
                .build_int_mul(
                    i64_type.const_int($stride as u64, false),
                    $offset,
                    "strided",
                )
                .unwrap();
            $builder
                .build_gep($ctx.i8_type(), $ptr, &[tmp], "inc_ptr")
                .unwrap()
        }
    };
}

/// Convert an Arrow Array into an IteratorHolder, which contains the C-style
/// iterator along with all the other C-style iterator needed. For example, a
/// dictionary array will get a C-style iterator and then two sub-iterators for
/// the keys and values.
pub fn array_to_iter(arr: &dyn Array) -> IteratorHolder {
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
        arrow_schema::DataType::BinaryView => todo!(),
        arrow_schema::DataType::Utf8 => IteratorHolder::String(arr.as_string().into()),
        arrow_schema::DataType::LargeUtf8 => IteratorHolder::LargeString(
            arr.as_any()
                .downcast_ref::<GenericStringArray<i64>>()
                .unwrap()
                .into(),
        ),
        arrow_schema::DataType::Utf8View => todo!(),
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
        arrow_schema::DataType::RunEndEncoded(_field, _field1) => todo!(),
    }
}

/// A holder for a C-style iterator. Created by `array_to_iter`, and used by
/// compiled LLVM functions.
#[derive(Debug)]
pub enum IteratorHolder {
    Primitive(Box<PrimitiveIterator>),
    String(Box<StringIterator>),
    LargeString(Box<LargeStringIterator>),
    Bitmap(Box<BitmapIterator>),
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
}

impl IteratorHolder {
    /// Gets a mutable pointer to the inner iterator, suitable for passing to
    /// compiled LLVM functions.
    pub fn get_mut_ptr(&mut self) -> *mut c_void {
        match self {
            IteratorHolder::Primitive(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::String(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::LargeString(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::Bitmap(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::Dictionary { arr: iter, .. } => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::RunEnd { arr: iter, .. } => &mut **iter as *mut _ as *mut c_void,
        }
    }

    /// Gets a const pointer to the inner iterator, suitable for passing to
    /// compiled LLVM functions.
    pub fn get_ptr(&self) -> *const c_void {
        match self {
            IteratorHolder::Primitive(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::String(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::LargeString(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::Bitmap(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::Dictionary { arr: iter, .. } => &**iter as *const _ as *const c_void,
            IteratorHolder::RunEnd { arr: iter, .. } => &**iter as *const _ as *const c_void,
        }
    }

    /// Sanity check to ensure all pointers in this iterator, and it's children,
    /// are non-null. Panics otherwise.
    pub fn assert_non_null(&self) {
        match self {
            IteratorHolder::Primitive(iter) => assert!(!iter.data.is_null()),
            IteratorHolder::String(iter) => {
                assert!(!iter.data.is_null());
                assert!(!iter.offsets.is_null());
            }
            IteratorHolder::LargeString(iter) => {
                assert!(!iter.data.is_null());
                assert!(!iter.offsets.is_null());
            }
            IteratorHolder::Bitmap(iter) => assert!(!iter.data.is_null()),
            IteratorHolder::Dictionary { arr, keys, values } => {
                assert!(!arr.key_iter.is_null());
                assert!(!arr.val_iter.is_null());
                keys.assert_non_null();
                values.assert_non_null();
            }
            IteratorHolder::RunEnd {
                arr,
                run_ends,
                values,
            } => {
                assert!(!arr.val_iter.is_null());
                assert!(!arr.val_iter.is_null());
                run_ends.assert_non_null();
                values.assert_non_null();
            }
        }
    }
}

/// An iterator for primitive (densely packed) data.
///
/// * `data` is a pointer to the densely packed data buffer
///
/// * `pos` is the current position in the iterator, all reads are relative to
/// this position
///
/// * `len` is the length of the data from the start of the `data` pointer
/// (i.e., not accounting for `pos`)
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct PrimitiveIterator {
    data: *const c_void,
    pos: u64,
    len: u64,
}

impl<K: ArrowPrimitiveType> From<&PrimitiveArray<K>> for Box<PrimitiveIterator> {
    fn from(value: &PrimitiveArray<K>) -> Self {
        Box::new(PrimitiveIterator {
            data: value.values().as_ptr() as *const c_void,
            pos: value.offset() as u64,
            len: (value.len() + value.offset()) as u64,
        })
    }
}

impl PrimitiveIterator {
    fn llvm_len<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let len_ptr = increment_pointer!(ctx, builder, ptr, PrimitiveIterator::OFFSET_LEN);
        builder
            .build_load(ctx.i64_type(), len_ptr, "len")
            .unwrap()
            .into_int_value()
    }

    fn llvm_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let offset_ptr = increment_pointer!(ctx, builder, ptr, PrimitiveIterator::OFFSET_POS);
        builder
            .build_load(ctx.i64_type(), offset_ptr, "pos")
            .unwrap()
            .into_int_value()
    }

    fn llvm_data<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let data_ptr_ptr = increment_pointer!(ctx, builder, ptr, PrimitiveIterator::OFFSET_DATA);
        builder
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                data_ptr_ptr,
                "data_ptr",
            )
            .unwrap()
            .into_pointer_value()
    }

    fn llvm_increment_pos<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
        amt: IntValue<'a>,
    ) {
        let pos_ptr = increment_pointer!(ctx, builder, ptr, PrimitiveIterator::OFFSET_POS);
        let pos = builder
            .build_load(ctx.i64_type(), pos_ptr, "pos")
            .unwrap()
            .into_int_value();
        let new_pos = builder.build_int_add(pos, amt, "new_pos").unwrap();
        builder.build_store(pos_ptr, new_pos).unwrap();
    }
}

/// An iterator for string data. Contains a pointer to the offset buffer and the
/// data buffer, along with a `pos` and `len` just like primitive iterators.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct StringIterator {
    offsets: *const i32,
    data: *const u8,
    pos: u64,
    len: u64,
}

impl From<&StringArray> for Box<StringIterator> {
    fn from(value: &StringArray) -> Self {
        Box::new(StringIterator {
            offsets: value.offsets().as_ptr(),
            data: value.values().as_ptr(),
            pos: value.offset() as u64,
            len: (value.len() + value.offset()) as u64,
        })
    }
}

/// Same as `StringIterator`, but with 64 bit offsets.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct LargeStringIterator {
    offsets: *const i64,
    data: *const u8,
    pos: u64,
    len: u64,
}

impl From<&GenericStringArray<i64>> for Box<LargeStringIterator> {
    fn from(value: &GenericStringArray<i64>) -> Self {
        Box::new(LargeStringIterator {
            offsets: value.offsets().as_ptr(),
            data: value.values().as_ptr(),
            pos: value.offset() as u64,
            len: (value.len() + value.offset()) as u64,
        })
    }
}

/// An iterator for bitmap data. Contains a pointer to the bitmap buffer and the
/// data buffer, along with a `pos` and `len` just like primitive iterators.
/// Note that each element pointed to by `data` contains 8 items/bits.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct BitmapIterator {
    data: *const u8,
    pos: u64,
    len: u64,
}

impl From<&BooleanArray> for Box<BitmapIterator> {
    fn from(value: &BooleanArray) -> Self {
        Box::new(BitmapIterator {
            data: value.values().values().as_ptr(),
            pos: value.offset() as u64,
            len: (value.len() + value.offset()) as u64,
        })
    }
}

/// An iterator for dictionary data. Contains pointers to the *iterators* for
/// the underlying keys and values. To access the element at position `i`, you
/// want to compute `value[key[i]]`. The `key_iter` is used for tracking current
/// position and length.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct DictionaryIterator {
    key_iter: *const c_void,
    val_iter: *const c_void,
}

impl DictionaryIterator {
    pub fn llvm_key_iter_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let ptr_ptr = increment_pointer!(ctx, builder, ptr, DictionaryIterator::OFFSET_KEY_ITER);
        builder
            .build_load(ctx.ptr_type(AddressSpace::default()), ptr_ptr, "key_iter")
            .unwrap()
            .into_pointer_value()
    }

    pub fn llvm_val_iter_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let ptr_ptr = increment_pointer!(ctx, builder, ptr, DictionaryIterator::OFFSET_VAL_ITER);
        builder
            .build_load(ctx.ptr_type(AddressSpace::default()), ptr_ptr, "val_iter")
            .unwrap()
            .into_pointer_value()
    }
}

impl<K: ArrowDictionaryKeyType> From<&DictionaryArray<K>> for IteratorHolder {
    fn from(arr: &DictionaryArray<K>) -> Self {
        let keys = Box::new(array_to_iter(arr.keys()));
        let values = Box::new(array_to_iter(arr.values()));
        let iter = DictionaryIterator {
            key_iter: keys.get_ptr(),
            val_iter: values.get_ptr(),
        };
        IteratorHolder::Dictionary {
            arr: Box::new(iter),
            keys,
            values,
        }
    }
}

/// An iterator for run-end encoded data. Contains pointers to the *iterators* for
/// the underlying run ends and values.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct RunEndIterator {
    /// the run ends, which must be i16, i32, or i64
    run_ends: *const c_void,

    /// the value iterator, not the raw value array
    val_iter: *const c_void,
    pos: u64,
    len: u64,
}

/// This adds a `next_block` function to the module for the given iterator. When
/// called, this `next_block` function will fetch `n` elements from the
/// iterator, advancing the iterator's offset. The generated function's signature is:
/// fn next_block(iter: ptr, out: ptr_to_vec_of_size_n) -> bool
///
fn generate_next_block<'a, const N: u32>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    label: &str,
    dt: &DataType,
    ih: &IteratorHolder,
) -> Option<FunctionValue<'a>> {
    let build = ctx.create_builder();
    let ptype = PrimitiveType::for_arrow_type(dt);
    let vec_type = ptype.llvm_vec_type(&ctx, N)?;
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

            return Some(next);
        }
        IteratorHolder::String(_) | IteratorHolder::LargeString(_) => return None,
        IteratorHolder::Bitmap(_bitmap_iterator) => todo!(),
        IteratorHolder::Dictionary { .. } => todo!(),
        IteratorHolder::RunEnd { .. } => todo!(),
    };
}

/// This adds a `next` function to the module for the given iterator. When
/// called, this `next` function will fetch the next element from the
/// iterator, advancing the iterator's offset. The generated function's signature is:
/// fn next(iter: ptr, out: ptr_to_el) -> bool
///
fn generate_next<'a>(
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
        &format!("{}_next_block", label),
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

            return Some(next);
        }
        IteratorHolder::String(_) | IteratorHolder::LargeString(_) => return None,
        IteratorHolder::Bitmap(_bitmap_iterator) => todo!(),
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
                return Some(next);
            }
            _ => unreachable!("dict iterator but not dict data type ({:?})", dt),
        },
        IteratorHolder::RunEnd { .. } => todo!(),
    };
}

/// This adds an `access` function to the module for the given iterator. When
/// called, this `access` function will fetch an element from the iterator at
/// the position given by the 2nd parameter, *without* advancing the iterator's
/// offset. Indexing is done from the base data, not from the iterator's current
/// position. The generated function's signature is:
///
/// fn access(iter: ptr, el: u64) -> T
///
fn generate_random_access<'a>(
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
    let next = llvm_mod.add_function(
        &format!("{}_access", label),
        fn_type,
        Some(
            #[cfg(test)]
            Linkage::External,
            #[cfg(not(test))]
            Linkage::Private,
        ),
    );
    let iter_ptr = next.get_nth_param(0).unwrap().into_pointer_value();
    let idx = next.get_nth_param(1).unwrap().into_int_value();

    match ih {
        IteratorHolder::Primitive(primitive_iter) => {
            declare_blocks!(ctx, next, entry);

            build.position_at_end(entry);
            let data_ptr = primitive_iter.llvm_data(ctx, &build, iter_ptr);
            let data_ptr = increment_pointer!(ctx, build, data_ptr, ptype.width(), idx);
            let out = build.build_load(llvm_type, data_ptr, "elem").unwrap();
            build.build_return(Some(&out)).unwrap();

            return Some(next);
        }
        IteratorHolder::String(_) | IteratorHolder::LargeString(_) => return None,
        IteratorHolder::Bitmap(_bitmap_iterator) => todo!(),
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

                declare_blocks!(ctx, next, entry);

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

                return Some(next);
            }
            _ => unreachable!("dictionary iterator but non-iterator data type ({:?})", dt),
        },
        IteratorHolder::RunEnd { .. } => todo!(),
    };
}

/// Not working yet
fn generate_llvm_cmp_kernel<'a>(
    ctx: &'a Context,
    lhs: &DataType,
    lhs_iter: &IteratorHolder,
    lhs_scalar: bool,
    rhs: &DataType,
    rhs_iter: &IteratorHolder,
    rhs_scalar: bool,
) -> JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u64)> {
    let module = ctx.create_module("cmp_kernel");
    let build = ctx.create_builder();
    let i64_type = ctx.i64_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let void_type = ctx.void_type();
    let lhs_prim = PrimitiveType::for_arrow_type(lhs);
    let rhs_prim = PrimitiveType::for_arrow_type(rhs);
    let lhs_vec = lhs_prim.llvm_vec_type(&ctx, 64).unwrap();
    let rhs_vec = rhs_prim.llvm_vec_type(&ctx, 64).unwrap();
    let lhs_llvm = lhs_prim.llvm_type(&ctx);
    let rhs_llvm = rhs_prim.llvm_type(&ctx);

    let lhs_next_block =
        generate_next_block::<64>(&ctx, &module, "next_lhs_block", lhs, lhs_iter).unwrap();
    let rhs_next_block =
        generate_next_block::<64>(&ctx, &module, "next_rhs_block", rhs, rhs_iter).unwrap();
    let lhs_next = generate_next(&ctx, &module, "next_lhs", lhs, lhs_iter).unwrap();
    let rhs_next = generate_next(&ctx, &module, "next_rhs", rhs, rhs_iter).unwrap();

    let fn_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
    let cmp = module.add_function("cmp", fn_type, None);
    let lhs_ptr = cmp.get_nth_param(0).unwrap();
    let rhs_ptr = cmp.get_nth_param(1).unwrap();
    let out_ptr = cmp.get_nth_param(2).unwrap();

    declare_blocks!(ctx, cmp, entry, block_cond, block_body, tail_cond, tail_body, exit);

    build.position_at_end(entry);
    let out_ptr_ptr = build.build_alloca(ptr_type, "out_ptr_ptr").unwrap();
    build.build_store(out_ptr_ptr, out_ptr).unwrap();

    let lhs_vec_buf = build.build_alloca(lhs_vec, "lhs_vec_buf").unwrap();
    let rhs_vec_buf = build.build_alloca(rhs_vec, "rhs_vec_buf").unwrap();
    let lhs_single_buf = build.build_alloca(lhs_llvm, "lhs_single_buf").unwrap();
    let rhs_single_buf = build.build_alloca(rhs_llvm, "rhs_single_buf").unwrap();
    let tail_buf_ptr = build.build_alloca(i64_type, "tail_buf_ptr").unwrap();
    build
        .build_store(tail_buf_ptr, i64_type.const_zero())
        .unwrap();
    let tail_buf_idx_ptr = build.build_alloca(i64_type, "tail_buf_idx").unwrap();
    build
        .build_store(tail_buf_idx_ptr, i64_type.const_zero())
        .unwrap();
    build.build_unconditional_branch(block_cond).unwrap();

    build.position_at_end(block_cond);
    let had_lhs = build
        .build_call(
            lhs_next_block,
            &[lhs_ptr.into(), lhs_vec_buf.into()],
            "lhs_next",
        )
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_call(
            rhs_next_block,
            &[rhs_ptr.into(), rhs_vec_buf.into()],
            "rhs_next",
        )
        .unwrap();
    build
        .build_conditional_branch(had_lhs, block_body, tail_cond)
        .unwrap();

    build.position_at_end(block_body);
    let lvec = build
        .build_load(lhs_vec, lhs_vec_buf, "lvec")
        .unwrap()
        .into_vector_value();
    let rvec = build
        .build_load(rhs_vec, rhs_vec_buf, "rvec")
        .unwrap()
        .into_vector_value();
    let res = build
        .build_int_compare(IntPredicate::ULT, lvec, rvec, "block_cmp_result")
        .unwrap();
    let res = build.build_bit_cast(res, i64_type, "res_u64").unwrap();
    let out_ptr = build
        .build_load(ptr_type, out_ptr_ptr, "out_ptr")
        .unwrap()
        .into_pointer_value();
    build.build_store(out_ptr, res).unwrap();
    let next_out_ptr = increment_pointer!(ctx, build, out_ptr, 8);
    build.build_store(out_ptr_ptr, next_out_ptr).unwrap();
    build.build_unconditional_branch(block_cond).unwrap();

    build.position_at_end(tail_cond);
    let had_lhs = build
        .build_call(
            lhs_next,
            &[lhs_ptr.into(), lhs_single_buf.into()],
            "lhs_next",
        )
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_call(
            rhs_next,
            &[rhs_ptr.into(), rhs_single_buf.into()],
            "rhs_next",
        )
        .unwrap();
    build
        .build_conditional_branch(had_lhs, tail_body, exit)
        .unwrap();

    build.position_at_end(tail_body);
    let lv = build
        .build_load(lhs_llvm, lhs_single_buf, "lv")
        .unwrap()
        .into_int_value();
    let rv = build
        .build_load(rhs_llvm, rhs_single_buf, "rv")
        .unwrap()
        .into_int_value();
    let res = build
        .build_int_compare(IntPredicate::ULT, lv, rv, "cmp_single")
        .unwrap();
    let res = build.build_int_cast(res, i64_type, "casted_cmp").unwrap();
    let tail_buf_idx = build
        .build_load(i64_type, tail_buf_idx_ptr, "tail_buf_idx")
        .unwrap()
        .into_int_value();
    let res = build
        .build_left_shift(res, tail_buf_idx, "shifted")
        .unwrap();
    let tail_buf = build
        .build_load(i64_type, tail_buf_ptr, "tail_buf")
        .unwrap()
        .into_int_value();
    let new_tail_buf = build.build_or(res, tail_buf, "new_tail_buf").unwrap();
    build.build_store(tail_buf_ptr, new_tail_buf).unwrap();
    let new_tail_buf_idx = build
        .build_int_add(
            tail_buf_idx,
            i64_type.const_int(1, false),
            "new_tail_buf_idx",
        )
        .unwrap();
    build
        .build_store(tail_buf_idx_ptr, new_tail_buf_idx)
        .unwrap();
    build.build_unconditional_branch(tail_cond).unwrap();

    build.position_at_end(exit);
    build.build_return(None).unwrap();

    module.verify().unwrap();
    module.print_to_stderr();
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();

    unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u64)>(
            cmp.get_name().to_str().unwrap(),
        )
        .unwrap()
    }
}

pub fn construct_cmp_kernel(
    lhs: &DataType,
    lhs_iter: &IteratorHolder,
    lhs_scalar: bool,
    rhs: &DataType,
    rhs_iter: &IteratorHolder,
    rhs_scalar: bool,
) -> ComparisonKernel {
    let ctx = Context::create();
    ComparisonKernelBuilder {
        context: ctx,
        lhs_data_type: lhs.clone(),
        rhs_data_type: rhs.clone(),
        lhs_scalar,
        rhs_scalar,
        func_builder: |ctx| {
            generate_llvm_cmp_kernel(ctx, lhs, lhs_iter, lhs_scalar, rhs, rhs_iter, rhs_scalar)
        },
    }
    .build()
}

#[derive(Debug)]
pub enum ArrowKernelError {
    SizeMismatch,
    ArgumentMismatch,
}

#[self_referencing]
pub struct ComparisonKernel {
    context: Context,
    lhs_data_type: DataType,
    rhs_data_type: DataType,
    lhs_scalar: bool,
    rhs_scalar: bool,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u64)>,
}

impl ComparisonKernel {
    pub fn call(&self, a: &dyn Datum, b: &dyn Datum) -> Result<BooleanArray, ArrowKernelError> {
        let (a, a_scalar) = a.get();
        let (b, b_scalar) = b.get();

        if (!a_scalar && !b_scalar) && (a.len() != b.len()) {
            return Err(ArrowKernelError::SizeMismatch);
        }

        let mut a_iter = array_to_iter(a);
        let mut b_iter = array_to_iter(b);
        let mut out = vec![0_u64; a.len().div_ceil(64)];

        self.with_func(|func| unsafe {
            func.call(a_iter.get_mut_ptr(), b_iter.get_mut_ptr(), out.as_mut_ptr())
        });

        let buf = BooleanBuffer::new(out.into(), 0, a.len());
        Ok(BooleanArray::new(
            buf,
            NullBuffer::union(a.nulls(), b.nulls()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::{ffi::c_void, sync::Arc};

    use arrow_array::{Array, DictionaryArray, Int32Array, Int8Array};
    use arrow_schema::DataType;
    use inkwell::{context::Context, OptimizationLevel};
    use itertools::Itertools;

    use super::{
        array_to_iter, construct_cmp_kernel, generate_next, generate_next_block,
        generate_random_access,
    };

    #[test]
    fn test_num_num_cmp() {
        let a = Int32Array::from(vec![1, 2, 3]);
        let b = Int32Array::from(vec![11, 0, 13]);
        let a_iter = array_to_iter(&a);
        let b_iter = array_to_iter(&b);
        let k = construct_cmp_kernel(a.data_type(), &a_iter, false, b.data_type(), &b_iter, false);
        let r = k.call(&a, &b).unwrap();
        panic!("{:?}", r);
    }

    #[test]
    fn test_primitive_iter_block() {
        let data = Int32Array::from((0..16).collect_vec());
        let mut iter = array_to_iter(&data);

        let iter_ptr = iter.get_mut_ptr();
        unsafe {
            let pos: u64 = (iter_ptr.add(8) as *mut u64).read();
            let len: u64 = (iter_ptr.add(16) as *mut u64).read();

            assert_eq!(pos, 0);
            assert_eq!(len, 16);
        }

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func =
            generate_next_block::<8>(&ctx, &module, "iter_prim_test", data.data_type(), &iter)
                .unwrap();
        let fname = func.get_name().to_str().unwrap();
        func.print_to_stderr();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut buf = [0_i32; 8];
        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [0, 1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [8, 9, 10, 11, 12, 13, 14, 15]);
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), false);
        };
    }

    #[test]
    fn test_primitive_iter_nonblock() {
        let data = Int32Array::from((0..5).collect_vec());
        let mut iter = array_to_iter(&data);

        let iter_ptr = iter.get_mut_ptr();
        unsafe {
            let pos: u64 = (iter_ptr.add(8) as *mut u64).read();
            let len: u64 = (iter_ptr.add(16) as *mut u64).read();

            assert_eq!(pos, 0);
            assert_eq!(len, 5);
        }

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_next(&ctx, &module, "iter_prim_test", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();
        func.print_to_stderr();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut buf: i32 = 0;
        unsafe {
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 0);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 1);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 2);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 3);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                true
            );
            assert_eq!(buf, 4);
            assert_eq!(
                next_func.call(iter.get_mut_ptr(), &mut buf as *mut i32),
                false
            );
        };
    }

    #[test]
    fn test_primitive_random_access() {
        let data = Int32Array::from((0..5).collect_vec());
        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_random_access(&ctx, &module, "iter_prim_test", data.data_type(), &iter)
            .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i32>(fname)
                .unwrap()
        };

        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), 0), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 1), 1);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 2), 2);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 3), 3);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 4), 4);
        };
    }

    #[test]
    fn test_dict_random_access() {
        let data = Int32Array::from(vec![0, 0, 10, 10, 20, 20, 30, 30, 40, 40]);
        let data = arrow_cast::cast(
            &data,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        )
        .unwrap();

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        println!("dt: {:?}", data.data_type());
        let func = generate_random_access(&ctx, &module, "iter_dict_test", data.data_type(), &iter)
            .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i32>(fname)
                .unwrap()
        };

        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), 0), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 1), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 2), 10);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 3), 10);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 4), 20);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 5), 20);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 6), 30);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 7), 30);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 8), 40);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 9), 40);
        };
    }

    #[test]
    fn test_dict_recur_random_access() {
        let keys = Int8Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let values = Int32Array::from(vec![0, 10, 20, 30]);
        let da1 = DictionaryArray::new(keys, Arc::new(values));

        let parent_keys = Int8Array::from(vec![0, 2, 4, 6]);
        let da2 = DictionaryArray::new(parent_keys, Arc::new(da1));

        let mut iter = array_to_iter(&da2);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        println!("dt: {:?}", da2.data_type());
        let func = generate_random_access(&ctx, &module, "iter_dict_test", da2.data_type(), &iter)
            .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i32>(fname)
                .unwrap()
        };

        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), 0), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 1), 10);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 2), 20);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 3), 30);
        };
    }

    #[test]
    fn test_dict_iter_nonblock() {
        let data = Int32Array::from(vec![0, 0, 10, 10, 20, 20]);
        let data = arrow_cast::cast(
            &data,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        )
        .unwrap();

        let mut iter = array_to_iter(&data);
        iter.assert_non_null();

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_next(&ctx, &module, "iter_dict_test", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        module.print_to_stderr();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool>(fname)
                .unwrap()
        };

        unsafe {
            let mut res: i32 = 0;
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 0);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 0);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 10);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 10);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 20);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 20);
            assert!(!next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
        };
    }
}
