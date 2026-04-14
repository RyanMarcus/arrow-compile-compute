use std::ffi::c_void;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Datum};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    types::BasicTypeEnum,
    values::{BasicValueEnum, PointerValue, VectorValue},
};

mod array_writer;
mod bool_writer;
mod dict_writer;
mod fixed_size_list_writer;
mod ree_writer;
mod str_writer;
mod view_writer;

pub use array_writer::ArrayOutput;
pub use array_writer::PrimitiveArrayWriter;
pub use bool_writer::BooleanAllocation;
pub use bool_writer::BooleanWriter;
pub use dict_writer::DictAllocation;
pub use dict_writer::DictWriter;
pub use fixed_size_list_writer::FixedSizeListWriter;
pub use fixed_size_list_writer::FixedSizeListWriterAlloc;
pub use ree_writer::REEAllocation;
pub use ree_writer::REEWriter;
pub use str_writer::StringAllocation;
pub use str_writer::StringArrayWriter;
pub use view_writer::StringViewAllocation;
pub use view_writer::StringViewWriter;
pub use view_writer::ViewBufferWriter;

use crate::{normalized_base_type, ListItemType, PrimitiveType};

pub trait LeafWriter<'a> {
    type Allocation: LeafWriterAllocation;

    fn allocate(expected_count: usize, ty: PrimitiveType) -> Self::Allocation;

    fn llvm_init(
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        ty: PrimitiveType,
        alloc_ptr: PointerValue<'a>,
    ) -> Self;

    fn llvm_ingest_type(&self, ctx: &'a Context) -> BasicTypeEnum<'a>;
    fn llvm_ingest(&self, ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>);
    fn llvm_flush(&self, ctx: &'a Context, build: &Builder<'a>);

    fn llvm_ingest_block(&self, ctx: &'a Context, build: &Builder<'a>, vals: VectorValue<'a>) {
        let i64_type = ctx.i64_type();
        for idx in 0..vals.get_type().get_size() {
            let val = build
                .build_extract_element(
                    vals,
                    i64_type.const_int(idx as u64, false),
                    &format!("val{}", idx),
                )
                .unwrap();
            self.llvm_ingest(ctx, build, val);
        }
    }
}

pub trait LeafWriterAllocation {
    type Output: Array;

    fn get_ptr(&mut self) -> *mut c_void;
    fn reserve_for_additional(&mut self, count: usize);

    fn rewind_one(&mut self) {
        panic!("rewind_one unsupported for this writer allocation");
    }

    fn len(&self) -> usize;
    fn to_array(self, len: usize, nulls: Option<NullBuffer>) -> Self::Output;
    fn to_array_ref(self, nulls: Option<NullBuffer>) -> ArrayRef;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DictionaryKeyType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl DictionaryKeyType {
    pub fn data_type(self) -> DataType {
        match self {
            Self::Int8 => DataType::Int8,
            Self::Int16 => DataType::Int16,
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
            Self::UInt8 => DataType::UInt8,
            Self::UInt16 => DataType::UInt16,
            Self::UInt32 => DataType::UInt32,
            Self::UInt64 => DataType::UInt64,
        }
    }

    pub fn primitive_type(self) -> PrimitiveType {
        PrimitiveType::for_arrow_type(&self.data_type())
    }

    pub fn for_data_type(dt: &DataType) -> Self {
        match dt {
            DataType::Int8 => Self::Int8,
            DataType::Int16 => Self::Int16,
            DataType::Int32 => Self::Int32,
            DataType::Int64 => Self::Int64,
            DataType::UInt8 => Self::UInt8,
            DataType::UInt16 => Self::UInt16,
            DataType::UInt32 => Self::UInt32,
            DataType::UInt64 => Self::UInt64,
            _ => panic!("unsupported dictionary key type {dt}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RunEndType {
    Int16,
    Int32,
    Int64,
}

impl RunEndType {
    pub fn data_type(self) -> DataType {
        match self {
            Self::Int16 => DataType::Int16,
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
        }
    }

    pub fn primitive_type(self) -> PrimitiveType {
        PrimitiveType::for_arrow_type(&self.data_type())
    }

    pub fn for_data_type(dt: &DataType) -> Self {
        match dt {
            DataType::Int16 => Self::Int16,
            DataType::Int32 => Self::Int32,
            DataType::Int64 => Self::Int64,
            _ => panic!("unsupported run-end type {dt}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum WriterSpec {
    Primitive(PrimitiveType),
    Boolean,
    String,
    LargeString,
    StringView,
    FixedSizeList(ListItemType, usize),
    Dictionary(DictionaryKeyType, Box<WriterSpec>),
    RunEndEncoded(RunEndType, Box<WriterSpec>),
}

impl WriterSpec {
    pub fn storage_type(&self) -> PrimitiveType {
        match self {
            Self::Primitive(pt) => {
                assert!(
                    !matches!(pt, PrimitiveType::P64x2 | PrimitiveType::List(_, _)),
                    "use a dedicated writer spec for strings and fixed-size lists",
                );
                *pt
            }
            Self::Boolean => PrimitiveType::U8,
            Self::String | Self::LargeString | Self::StringView => PrimitiveType::P64x2,
            Self::FixedSizeList(item, size) => PrimitiveType::List(*item, *size),
            Self::Dictionary(_, values) | Self::RunEndEncoded(_, values) => values.storage_type(),
        }
    }

    pub fn allocate(&self, expected_count: usize) -> WriterAllocation {
        match self {
            Self::Primitive(pt) => {
                WriterAllocation::Primitive(PrimitiveArrayWriter::allocate(expected_count, *pt))
            }
            Self::Boolean => WriterAllocation::Boolean(BooleanWriter::allocate(
                expected_count,
                self.storage_type(),
            )),
            Self::String => WriterAllocation::String(StringArrayWriter::allocate(
                expected_count,
                self.storage_type(),
            )),
            Self::LargeString => WriterAllocation::LargeString(StringArrayWriter::allocate(
                expected_count,
                self.storage_type(),
            )),
            Self::StringView => WriterAllocation::StringView(StringViewWriter::allocate(
                expected_count,
                self.storage_type(),
            )),
            Self::FixedSizeList(item, size) => WriterAllocation::FixedSizeList(
                FixedSizeListWriter::allocate(expected_count, PrimitiveType::List(*item, *size)),
            ),
            Self::Dictionary(key, values) => WriterAllocation::Dictionary(Box::new(
                DictAllocation::allocate(expected_count, *key, values),
            )),
            Self::RunEndEncoded(run_end, values) => WriterAllocation::RunEndEncoded(Box::new(
                REEAllocation::allocate(expected_count, *run_end, values),
            )),
        }
    }

    pub fn llvm_init<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        alloc_ptr: PointerValue<'a>,
    ) -> Writer<'a> {
        let storage_type = self.storage_type();
        match self {
            Self::Primitive(_) => Writer::Primitive(PrimitiveArrayWriter::llvm_init(
                ctx,
                llvm_mod,
                build,
                storage_type,
                alloc_ptr,
            )),
            Self::Boolean => Writer::Boolean(BooleanWriter::llvm_init(
                ctx,
                llvm_mod,
                build,
                storage_type,
                alloc_ptr,
            )),
            Self::String => Writer::String(StringArrayWriter::llvm_init(
                ctx,
                llvm_mod,
                build,
                storage_type,
                alloc_ptr,
            )),
            Self::LargeString => Writer::LargeString(StringArrayWriter::llvm_init(
                ctx,
                llvm_mod,
                build,
                storage_type,
                alloc_ptr,
            )),
            Self::StringView => Writer::StringView(StringViewWriter::llvm_init(
                ctx,
                llvm_mod,
                build,
                storage_type,
                alloc_ptr,
            )),
            Self::FixedSizeList(_, _) => Writer::FixedSizeList(FixedSizeListWriter::llvm_init(
                ctx,
                llvm_mod,
                build,
                storage_type,
                alloc_ptr,
            )),
            Self::Dictionary(key, values) => Writer::Dictionary(Box::new(DictWriter::llvm_init(
                ctx, llvm_mod, build, *key, values, alloc_ptr,
            ))),
            Self::RunEndEncoded(run_end, values) => Writer::RunEndEncoded(Box::new(
                REEWriter::llvm_init(ctx, llvm_mod, build, *run_end, values, alloc_ptr),
            )),
        }
    }

    pub fn for_base_type_of_datum(datum: &dyn Datum) -> Self {
        match normalized_base_type(datum.get().0.data_type()) {
            DataType::Boolean => Self::Boolean,
            DataType::Binary
            | DataType::FixedSizeBinary(_)
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View => Self::String,
            DataType::FixedSizeList(field, len) => Self::FixedSizeList(
                PrimitiveType::for_arrow_type(field.data_type())
                    .try_into()
                    .unwrap(),
                len as usize,
            ),
            dt => Self::Primitive(PrimitiveType::for_arrow_type(&dt)),
        }
    }

    pub fn for_data_type(dt: &DataType) -> Self {
        match dt {
            DataType::Null => todo!(),
            DataType::Boolean => Self::Boolean,
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Timestamp(_, _)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Duration(_)
            | DataType::Interval(_) => Self::Primitive(PrimitiveType::for_arrow_type(dt)),
            DataType::Binary | DataType::FixedSizeBinary(_) | DataType::Utf8 => Self::String,
            DataType::LargeBinary | DataType::LargeUtf8 => Self::LargeString,
            DataType::BinaryView | DataType::Utf8View => Self::StringView,
            DataType::List(_) => todo!(),
            DataType::ListView(_) => todo!(),
            DataType::FixedSizeList(field, len) => Self::FixedSizeList(
                PrimitiveType::for_arrow_type(field.data_type())
                    .try_into()
                    .unwrap(),
                *len as usize,
            ),
            DataType::LargeList(_) => todo!(),
            DataType::LargeListView(_) => todo!(),
            DataType::Struct(_) => todo!(),
            DataType::Union(_, _) => todo!(),
            DataType::Dictionary(key, values) => Self::Dictionary(
                DictionaryKeyType::for_data_type(key.as_ref()),
                Box::new(Self::for_data_type(values.as_ref())),
            ),
            DataType::Decimal128(_, _) => todo!(),
            DataType::Decimal256(_, _) => todo!(),
            DataType::Map(_, _) => todo!(),
            DataType::RunEndEncoded(run_end, values) => Self::RunEndEncoded(
                RunEndType::for_data_type(run_end.data_type()),
                Box::new(Self::for_data_type(values.data_type())),
            ),
        }
    }
}

pub enum WriterAllocation {
    Primitive(ArrayOutput),
    Boolean(BooleanAllocation),
    String(StringAllocation<i32>),
    LargeString(StringAllocation<i64>),
    StringView(StringViewAllocation),
    FixedSizeList(FixedSizeListWriterAlloc),
    Dictionary(Box<DictAllocation>),
    RunEndEncoded(Box<REEAllocation>),
}

impl WriterAllocation {
    pub fn get_ptr(&mut self) -> *mut c_void {
        match self {
            Self::Primitive(alloc) => alloc.get_ptr(),
            Self::Boolean(alloc) => alloc.get_ptr(),
            Self::String(alloc) => alloc.get_ptr(),
            Self::LargeString(alloc) => alloc.get_ptr(),
            Self::StringView(alloc) => alloc.get_ptr(),
            Self::FixedSizeList(alloc) => alloc.get_ptr(),
            Self::Dictionary(alloc) => alloc.get_ptr(),
            Self::RunEndEncoded(alloc) => alloc.get_ptr(),
        }
    }

    pub fn reserve_for_additional(&mut self, count: usize) {
        match self {
            Self::Primitive(alloc) => alloc.reserve_for_additional(count),
            Self::Boolean(alloc) => alloc.reserve_for_additional(count),
            Self::String(alloc) => alloc.reserve_for_additional(count),
            Self::LargeString(alloc) => alloc.reserve_for_additional(count),
            Self::StringView(alloc) => alloc.reserve_for_additional(count),
            Self::FixedSizeList(alloc) => alloc.reserve_for_additional(count),
            Self::Dictionary(alloc) => alloc.reserve_for_additional(count),
            Self::RunEndEncoded(alloc) => alloc.reserve_for_additional(count),
        }
    }

    pub fn rewind_one(&mut self) {
        match self {
            Self::Primitive(alloc) => alloc.rewind_one(),
            Self::Boolean(alloc) => alloc.rewind_one(),
            Self::String(alloc) => alloc.rewind_one(),
            Self::LargeString(alloc) => alloc.rewind_one(),
            Self::StringView(alloc) => alloc.rewind_one(),
            Self::FixedSizeList(alloc) => alloc.rewind_one(),
            Self::Dictionary(_) => {
                panic!("rewind_one unsupported for dictionary writer allocation")
            }
            Self::RunEndEncoded(_) => {
                panic!("rewind_one unsupported for run-end writer allocation")
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Primitive(alloc) => alloc.len(),
            Self::Boolean(alloc) => alloc.len(),
            Self::String(alloc) => alloc.len(),
            Self::LargeString(alloc) => alloc.len(),
            Self::StringView(alloc) => alloc.len(),
            Self::FixedSizeList(alloc) => alloc.len(),
            Self::Dictionary(alloc) => alloc.len(),
            Self::RunEndEncoded(alloc) => alloc.len(),
        }
    }

    pub fn into_array_ref_with_len(self, len: usize, nulls: Option<NullBuffer>) -> ArrayRef {
        match self {
            Self::Primitive(alloc) => alloc.to_array(len, nulls),
            Self::Boolean(alloc) => Arc::new(alloc.to_array(len, nulls)),
            Self::String(alloc) => Arc::new(alloc.to_array(len, nulls)),
            Self::LargeString(alloc) => Arc::new(alloc.to_array(len, nulls)),
            Self::StringView(alloc) => Arc::new(alloc.to_array(len, nulls)),
            Self::FixedSizeList(alloc) => Arc::new(alloc.to_array(len, nulls)),
            Self::Dictionary(alloc) => alloc.into_array_ref_with_len(len, nulls),
            Self::RunEndEncoded(alloc) => alloc.into_array_ref_with_len(len, nulls),
        }
    }

    pub fn into_array_ref(self, nulls: Option<NullBuffer>) -> ArrayRef {
        match self {
            Self::Primitive(alloc) => alloc.to_array_ref(nulls),
            Self::Boolean(alloc) => alloc.to_array_ref(nulls),
            Self::String(alloc) => alloc.to_array_ref(nulls),
            Self::LargeString(alloc) => alloc.to_array_ref(nulls),
            Self::StringView(alloc) => alloc.to_array_ref(nulls),
            Self::FixedSizeList(alloc) => alloc.to_array_ref(nulls),
            Self::Dictionary(alloc) => alloc.into_array_ref(nulls),
            Self::RunEndEncoded(alloc) => alloc.into_array_ref(nulls),
        }
    }
}

pub enum Writer<'a> {
    Primitive(PrimitiveArrayWriter<'a>),
    Boolean(BooleanWriter<'a>),
    String(StringArrayWriter<'a, i32>),
    LargeString(StringArrayWriter<'a, i64>),
    StringView(StringViewWriter<'a>),
    FixedSizeList(FixedSizeListWriter<'a>),
    Dictionary(Box<DictWriter<'a>>),
    RunEndEncoded(Box<REEWriter<'a>>),
}

impl<'a> Writer<'a> {
    pub fn llvm_ingest_type(&self, ctx: &'a Context) -> BasicTypeEnum<'a> {
        match self {
            Self::Primitive(writer) => writer.llvm_ingest_type(ctx),
            Self::Boolean(writer) => writer.llvm_ingest_type(ctx),
            Self::String(writer) => writer.llvm_ingest_type(ctx),
            Self::LargeString(writer) => writer.llvm_ingest_type(ctx),
            Self::StringView(writer) => writer.llvm_ingest_type(ctx),
            Self::FixedSizeList(writer) => writer.llvm_ingest_type(ctx),
            Self::Dictionary(writer) => writer.llvm_ingest_type(ctx),
            Self::RunEndEncoded(writer) => writer.llvm_ingest_type(ctx),
        }
    }

    pub fn llvm_ingest(&self, ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>) {
        match self {
            Self::Primitive(writer) => writer.llvm_ingest(ctx, build, val),
            Self::Boolean(writer) => writer.llvm_ingest(ctx, build, val),
            Self::String(writer) => writer.llvm_ingest(ctx, build, val),
            Self::LargeString(writer) => writer.llvm_ingest(ctx, build, val),
            Self::StringView(writer) => writer.llvm_ingest(ctx, build, val),
            Self::FixedSizeList(writer) => writer.llvm_ingest(ctx, build, val),
            Self::Dictionary(writer) => writer.llvm_ingest(ctx, build, val),
            Self::RunEndEncoded(writer) => writer.llvm_ingest(ctx, build, val),
        }
    }

    pub fn llvm_ingest_block(&self, ctx: &'a Context, build: &Builder<'a>, vals: VectorValue<'a>) {
        match self {
            Self::Primitive(writer) => writer.llvm_ingest_block(ctx, build, vals),
            Self::Boolean(writer) => writer.llvm_ingest_block(ctx, build, vals),
            Self::String(writer) => writer.llvm_ingest_block(ctx, build, vals),
            Self::LargeString(writer) => writer.llvm_ingest_block(ctx, build, vals),
            Self::StringView(writer) => writer.llvm_ingest_block(ctx, build, vals),
            Self::FixedSizeList(writer) => writer.llvm_ingest_block(ctx, build, vals),
            Self::Dictionary(writer) => {
                let i64_type = ctx.i64_type();
                for idx in 0..vals.get_type().get_size() {
                    let val = build
                        .build_extract_element(
                            vals,
                            i64_type.const_int(idx as u64, false),
                            &format!("val{}", idx),
                        )
                        .unwrap();
                    writer.llvm_ingest(ctx, build, val);
                }
            }
            Self::RunEndEncoded(writer) => {
                let i64_type = ctx.i64_type();
                for idx in 0..vals.get_type().get_size() {
                    let val = build
                        .build_extract_element(
                            vals,
                            i64_type.const_int(idx as u64, false),
                            &format!("val{}", idx),
                        )
                        .unwrap();
                    writer.llvm_ingest(ctx, build, val);
                }
            }
        }
    }

    pub fn llvm_flush(&self, ctx: &'a Context, build: &Builder<'a>) {
        match self {
            Self::Primitive(writer) => writer.llvm_flush(ctx, build),
            Self::Boolean(writer) => writer.llvm_flush(ctx, build),
            Self::String(writer) => writer.llvm_flush(ctx, build),
            Self::LargeString(writer) => writer.llvm_flush(ctx, build),
            Self::StringView(writer) => writer.llvm_flush(ctx, build),
            Self::FixedSizeList(writer) => writer.llvm_flush(ctx, build),
            Self::Dictionary(writer) => writer.llvm_flush(ctx, build),
            Self::RunEndEncoded(writer) => writer.llvm_flush(ctx, build),
        }
    }
}
