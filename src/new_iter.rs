use std::{ffi::c_void, intrinsics::unreachable};

use arrow_array::{
    cast::AsArray,
    types::{
        Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
        UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrowPrimitiveType, BooleanArray, GenericStringArray, PrimitiveArray, StringArray,
};

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
        arrow_schema::DataType::Time32(time_unit) => todo!(),
        arrow_schema::DataType::Time64(time_unit) => todo!(),
        arrow_schema::DataType::Duration(time_unit) => todo!(),
        arrow_schema::DataType::Interval(interval_unit) => todo!(),
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
        arrow_schema::DataType::List(field) => todo!(),
        arrow_schema::DataType::ListView(field) => todo!(),
        arrow_schema::DataType::FixedSizeList(field, _) => todo!(),
        arrow_schema::DataType::LargeList(field) => todo!(),
        arrow_schema::DataType::LargeListView(field) => todo!(),
        arrow_schema::DataType::Struct(fields) => todo!(),
        arrow_schema::DataType::Union(union_fields, union_mode) => todo!(),
        arrow_schema::DataType::Dictionary(key_type, value_type) => match key_type.as_ref() {
            arrow_schema::DataType::Int8 => {
                let arr = arr.as_dictionary::<Int8Type>();
                let keys = array_to_iter(arr.keys());
                let values = array_to_iter(arr.values());
                todo!()
            }
            arrow_schema::DataType::Int16 => todo!(),
            arrow_schema::DataType::Int32 => todo!(),
            arrow_schema::DataType::Int64 => todo!(),
            arrow_schema::DataType::UInt8 => todo!(),
            arrow_schema::DataType::UInt16 => todo!(),
            arrow_schema::DataType::UInt32 => todo!(),
            arrow_schema::DataType::UInt64 => todo!(),
            _ => unreachable!(),
        },
        arrow_schema::DataType::Decimal128(_, _) => todo!(),
        arrow_schema::DataType::Decimal256(_, _) => todo!(),
        arrow_schema::DataType::Map(field, _) => todo!(),
        arrow_schema::DataType::RunEndEncoded(field, field1) => todo!(),
    }
}

/// A holder for a C-style iterator. Created by `array_to_iter`, and used by
/// compiled LLVM functions.
enum IteratorHolder {
    Primitive(Box<PrimitiveIterator>),
    String(Box<StringIterator>),
    LargeString(Box<LargeStringIterator>),
    Bitmap(Box<BitmapIterator>),
    Dictionary(Box<DictionaryIterator>, Vec<Box<IteratorHolder>>),
    RunEnd(Box<RunEndIterator>, Vec<Box<IteratorHolder>>),
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
struct PrimitiveIterator {
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

/// An iterator for string data. Contains a pointer to the offset buffer and the
/// data buffer, along with a `pos` and `len` just like primitive iterators.
#[repr(C)]
struct StringIterator {
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
struct LargeStringIterator {
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
struct BitmapIterator {
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
/// want to compute `value[key[i]]`.
#[repr(C)]
struct DictionaryIterator {
    key_iter: *const c_void,
    val_iter: *const c_void,
    pos: u64,
    len: u64,
}

/// An iterator for run-end encoded data. Contains pointers to the *iterators* for
/// the underlying run ends and values.
#[repr(C)]
struct RunEndIterator {
    /// the run ends, which must be i16, i32, or i64
    run_ends: *const c_void,

    /// the value iterator, not the raw value array
    val_iter: *const c_void,
    pos: u64,
    len: u64,
}
