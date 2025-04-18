use std::ffi::c_void;

use arrow_array::{
    cast::AsArray,
    types::{
        ArrowDictionaryKeyType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type,
        Int64Type, Int8Type, RunEndIndexType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrowPrimitiveType, BooleanArray, Datum, DictionaryArray, GenericStringArray,
    PrimitiveArray, RunArray, StringArray,
};
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    types::BasicType,
    values::{BasicValue, FunctionValue, IntValue, PointerValue},
    AddressSpace, IntPredicate,
};
use repr_offset::ReprOffset;

use crate::{declare_blocks, increment_pointer, ArrowKernelError, PrimitiveType};

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
    Bitmap(Box<BitmapIterator>), // all three methods, returns 0 or 1
    // TODO(paul): SetBitsOfBitmap() -- only single get next, no random access or block
    // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
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
            IteratorHolder::Bitmap(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::Dictionary { arr: iter, .. } => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::RunEnd { arr: iter, .. } => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::ScalarPrimitive(iter) => &mut **iter as *mut _ as *mut c_void,
            IteratorHolder::ScalarString(iter) => &mut **iter as *mut _ as *mut c_void,
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
            IteratorHolder::ScalarPrimitive(iter) => &**iter as *const _ as *const c_void,
            IteratorHolder::ScalarString(iter) => &**iter as *const _ as *const c_void,
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
            IteratorHolder::ScalarPrimitive(_) => {}
            IteratorHolder::ScalarString(_) => {}
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

impl<R: RunEndIndexType + ArrowPrimitiveType> From<&RunArray<R>> for IteratorHolder {
    fn from(arr: &RunArray<R>) -> Self {
        let re = arr.run_ends().inner().clone();
        let re: PrimitiveArray<R> = PrimitiveArray::new(re, None);
        let run_ends = Box::new(array_to_iter(&re));
        let values = Box::new(array_to_iter(arr.values()));
        let iter = RunEndIterator {
            run_ends: run_ends.get_ptr(),
            val_iter: values.get_ptr(),
            pos: 0,
            len: arr.len() as u64,
        };
        IteratorHolder::RunEnd {
            arr: Box::new(iter),
            run_ends,
            values,
        }
    }
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct ScalarPrimitiveIterator {
    /// the scalar value, packed to the right of a u64
    val: u64,

    /// the width, in bytes, of the scalar
    width: u8,
}

impl ScalarPrimitiveIterator {
    pub fn new(val: u64, width: u8) -> Self {
        Self { val, width }
    }

    pub fn llvm_val_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        increment_pointer!(ctx, builder, ptr, ScalarPrimitiveIterator::OFFSET_VAL)
    }
}

impl From<i8> for IteratorHolder {
    fn from(val: i8) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val as u8 as u64, 1)))
    }
}

impl From<i16> for IteratorHolder {
    fn from(val: i16) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val as u16 as u64,
            2,
        )))
    }
}

impl From<i32> for IteratorHolder {
    fn from(val: i32) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val as u32 as u64,
            4,
        )))
    }
}

impl From<i64> for IteratorHolder {
    fn from(val: i64) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val as u64, 8)))
    }
}

impl From<u8> for IteratorHolder {
    fn from(val: u8) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val as u64, 1)))
    }
}

impl From<u16> for IteratorHolder {
    fn from(val: u16) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val as u64, 2)))
    }
}

impl From<u32> for IteratorHolder {
    fn from(val: u32) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val as u64, 4)))
    }
}

impl From<u64> for IteratorHolder {
    fn from(val: u64) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val as u64, 8)))
    }
}

use half::f16;
impl From<f16> for IteratorHolder {
    fn from(val: f16) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            (val.to_f64()).to_bits(),
            2,
        )))
    }
}

impl From<f32> for IteratorHolder {
    fn from(val: f32) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(
            val.to_bits() as u64,
            4,
        )))
    }
}

impl From<f64> for IteratorHolder {
    fn from(val: f64) -> Self {
        IteratorHolder::ScalarPrimitive(Box::new(ScalarPrimitiveIterator::new(val.to_bits(), 8)))
    }
}

#[derive(Debug)]
pub struct ScalarStringIterator {
    val: Box<str>,
}

impl From<Box<str>> for IteratorHolder {
    fn from(val: Box<str>) -> Self {
        IteratorHolder::ScalarString(Box::new(ScalarStringIterator { val }))
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
    let vec_type = ptype.llvm_vec_type(&ctx, N)?;
    let llvm_type = ptype.llvm_type(&ctx);
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

                return Some(next);
            }
            _ => unreachable!("dict iterator but not dict data type ({:?})", dt),
        },
        IteratorHolder::RunEnd { .. } => todo!(),
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
            return Some(next);
        }
        IteratorHolder::ScalarString(_) => todo!(),
    };
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
            return Some(next);
        }
        IteratorHolder::ScalarString(_) => todo!(),
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
        IteratorHolder::ScalarPrimitive(_s) => todo!(),
        IteratorHolder::ScalarString(_) => todo!(),
    };
}

#[cfg(test)]
mod tests {
    use std::{ffi::c_void, sync::Arc};

    use arrow_array::{Array, DictionaryArray, Float32Array, Int32Array, Int8Array, Scalar};
    use arrow_schema::DataType;
    use inkwell::{context::Context, OptimizationLevel};
    use itertools::Itertools;

    use super::{
        array_to_iter, datum_to_iter, generate_next, generate_next_block, generate_random_access,
    };

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
    fn test_dict_iter_block1() {
        let data = Int32Array::from(vec![10, 20, 30]);
        let data = arrow_cast::cast(
            &data,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        )
        .unwrap();

        let mut iter = array_to_iter(&data);
        iter.assert_non_null();

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func =
            generate_next_block::<8>(&ctx, &module, "dict_iter_block1", data.data_type(), &iter)
                .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut out_buf = [0_i32; 8];

        let result = unsafe { next_block_func.call(iter.get_mut_ptr(), out_buf.as_mut_ptr()) };

        assert!(
            !result,
            "expected false since the dict size is less than the block size"
        );
    }

    #[test]
    fn test_dict_iter_block2() {
        let data = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let data = arrow_cast::cast(
            &data,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        )
        .unwrap();

        let mut iter = array_to_iter(&data);
        iter.assert_non_null();

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func =
            generate_next_block::<2>(&ctx, &module, "dict_iter_block1", data.data_type(), &iter)
                .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut out_buf = [0_i32; 2];

        unsafe {
            let ret1 = next_block_func.call(iter.get_mut_ptr(), out_buf.as_mut_ptr());
            assert!(ret1, "First call should return true");
            assert_eq!(out_buf, [10, 20], "First block should have [10, 20]");

            let ret2 = next_block_func.call(iter.get_mut_ptr(), out_buf.as_mut_ptr());
            assert!(ret2, "Second call should return true");
            assert_eq!(out_buf, [30, 40], "Second block should have [30, 40]");

            let ret3 = next_block_func.call(iter.get_mut_ptr(), out_buf.as_mut_ptr());
            assert!(!ret3, "Third call should return false");
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

    #[test]
    fn test_dict_recur_iter_nonblock() {
        let keys = Int8Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let values = Int32Array::from(vec![0, 10, 20, 30]);
        let da1 = DictionaryArray::new(keys, Arc::new(values));

        let parent_keys = Int8Array::from(vec![0, 2, 4, 6]);
        let da2 = DictionaryArray::new(parent_keys, Arc::new(da1));

        let mut iter = array_to_iter(&da2);
        iter.assert_non_null();

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_next(&ctx, &module, "iter_dict_test", da2.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
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
            assert_eq!(res, 10);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 20);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 30);
            assert!(!next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
        };
    }

    #[test]
    fn test_scalar_int() {
        let s = Int32Array::from(vec![42]);
        let s = Scalar::new(s);
        let mut iter = datum_to_iter(&s).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_scalar_prim");
        let func_block =
            generate_next_block::<4>(&ctx, &module, "iter_prim_test", &DataType::Int32, &iter)
                .unwrap();
        let func_next =
            generate_next(&ctx, &module, "iter_prim_test", &DataType::Int32, &iter).unwrap();

        let fname_block = func_block.get_name().to_str().unwrap();
        let fname_next = func_next.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname_block)
                .unwrap()
        };
        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname_next)
                .unwrap()
        };

        let mut buf = [0_i32; 4];
        unsafe {
            assert_eq!(
                next_block_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()),
                true
            );
            assert_eq!(buf, [42, 42, 42, 42]);
            assert_eq!(
                next_block_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()),
                true
            );
            assert_eq!(buf, [42, 42, 42, 42]);
        };

        let mut buf = [0_i32; 1];
        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [42]);
            assert_eq!(next_func.call(iter.get_mut_ptr(), buf.as_mut_ptr()), true);
            assert_eq!(buf, [42]);
        };
    }

    #[test]
    fn test_scalar_float() {
        let f = 42.31894;
        let s = Float32Array::from(vec![f]);
        let s = Scalar::new(s);
        let mut iter = datum_to_iter(&s).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("test_scalar_prim");
        let func_block = generate_next_block::<4>(
            &ctx,
            &module,
            "iter_block_prim_test",
            &DataType::Float32,
            &iter,
        )
        .unwrap();
        let func_next =
            generate_next(&ctx, &module, "iter_prim_test", &DataType::Float32, &iter).unwrap();

        let fname_block = func_block.get_name().to_str().unwrap();
        let fname_next = func_next.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut f32) -> bool>(fname_block)
                .unwrap()
        };
        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut f32) -> bool>(fname_next)
                .unwrap()
        };

        let mut buf: f32 = 0.0;
        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert_eq!(buf, f);
            assert_eq!(next_func.call(iter.get_mut_ptr(), &mut buf), true);
            assert_eq!(buf, f);
        };

        #[repr(align(16))]
        struct Buf([f32; 4]);
        let mut buf = Buf([0_f32; 4]);
        unsafe {
            assert_eq!(
                next_block_func.call(iter.get_mut_ptr(), buf.0.as_mut_ptr()),
                true
            );
            assert_eq!(buf.0, [f, f, f, f]);
            assert_eq!(
                next_block_func.call(iter.get_mut_ptr(), buf.0.as_mut_ptr()),
                true
            );
            assert_eq!(buf.0, [f, f, f, f]);
        };
    }
}
