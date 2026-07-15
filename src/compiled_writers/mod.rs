mod boolean_writer;
mod dictionary_writer;
mod fixed_size_list_writer;
mod list_writer;
mod primitive_writer;
mod run_end_writer;
mod string_writer;
mod view_writer;

use std::ffi::c_void;

use arrow_array::{make_array, ArrayRef, Datum};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use enum_dispatch::enum_dispatch;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    values::{BasicValueEnum, PointerValue, VectorValue},
};

use dictionary_writer::{DictionaryWriter, DictionaryWriterEmitter, DictionaryWriterRuntime};
use fixed_size_list_writer::{
    FixedSizeListWriter, FixedSizeListWriterEmitter, FixedSizeListWriterRuntime,
};
use list_writer::{ListWriter, ListWriterEmitter, ListWriterRuntime};
use run_end_writer::{RunEndWriter, RunEndWriterEmitter, RunEndWriterRuntime};
use view_writer::{StringViewWriter, StringViewWriterEmitter, StringViewWriterRuntime};

use crate::{
    compiled_writers::string_writer::{StringWriter, StringWriterEmitter, StringWriterRuntime},
    normalized_base_type, ArrowKernelError, ListItemType, PrimitiveType,
};

pub use boolean_writer::{BooleanWriter, BooleanWriterEmitter, BooleanWriterRuntime};
pub use primitive_writer::{PrimitiveWriter, PrimitiveWriterEmitter, PrimitiveWriterRuntime};
pub use view_writer::ViewBufferWriter;

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

#[enum_dispatch]
pub trait WriterRuntime {
    fn as_ptr(&mut self) -> *mut c_void;

    /// Reserves space for future writes and refreshes any child runtime pointers.
    fn reserve_for_additional(&mut self, count: usize) -> Result<(), ArrowKernelError>;
    fn len(&self) -> usize;

    /// Finalizes the runtime state and materializes an Arrow array.
    fn to_array(self, len: usize) -> Result<ArrayRef, ArrowKernelError>;

    fn to_array_ref(self) -> Result<ArrayRef, ArrowKernelError>
    where
        Self: Sized,
    {
        let len = self.len();
        self.to_array(len)
    }
}

#[enum_dispatch]
pub trait WriterEmitter<'ctx, 'borrow> {
    fn emit(&mut self, val: BasicValueEnum<'ctx>) -> Result<(), ArrowKernelError>;
}

#[derive(Clone, Copy)]
pub struct WriterCodegen<'ctx, 'borrow> {
    pub ctx: &'ctx Context,
    pub module: &'borrow Module<'ctx>,
    pub builder: &'borrow Builder<'ctx>,
}

#[enum_dispatch]
pub trait Writer {
    /// Allocates the runtime state used by generated code.
    ///
    /// `size` will be the maximum number of items that can be written. Writing
    /// more items will cause invalid writes / segfaults / UB.
    fn allocate(&self, size: usize) -> AnyRuntime;

    /// Emits initialization code for this writer and its composed children.
    ///
    /// `runtime_ptr` must point to runtime state returned by [`Writer::allocate`].
    fn llvm_init<'ctx, 'borrow>(
        &self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
    );

    /// Emits code for one logical write operation.
    ///
    /// The callback describes the value through a writer-specific emitter.
    /// Scalar emitters generally accept one call to [`WriterEmitter::emit`],
    /// while composite emitters may accept multiple calls to describe the
    /// contents of one logical value.
    fn llvm_write<'ctx, 'borrow, F>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        f: F,
    ) -> Result<(), ArrowKernelError>
    where
        F: Fn(&mut AnyWriterEmitter<'ctx, 'borrow>) -> Result<(), ArrowKernelError>;

    /// Writes each lane of `values` as a separate logical value.
    ///
    /// Writers may override this to ingest a vector more efficiently. The
    /// default preserves the single-value emitter contract by creating a new
    /// emitter for every lane.
    fn llvm_write_multiple<'ctx, 'borrow>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        values: VectorValue<'ctx>,
    ) -> Result<(), ArrowKernelError> {
        for idx in 0..values.get_type().get_size() {
            let value = codegen
                .builder
                .build_extract_element(
                    values,
                    codegen.ctx.i64_type().const_int(idx as u64, false),
                    "writer_multiple_value",
                )
                .unwrap();
            self.llvm_write(codegen, runtime_ptr, |emitter| emitter.emit(value))?;
        }
        Ok(())
    }
}

#[enum_dispatch(Writer)]
pub enum AnyWriter {
    Boolean(BooleanWriter),
    Dictionary(DictionaryWriter),
    FixedSizeList(FixedSizeListWriter),
    Primitive(PrimitiveWriter),
    List(ListWriter),
    RunEnd(RunEndWriter),
    String(StringWriter),
    StringView(StringViewWriter),
}

#[enum_dispatch(WriterEmitter)]
pub enum AnyWriterEmitter<'ctx, 'borrow> {
    Boolean(BooleanWriterEmitter<'ctx, 'borrow>),
    Dictionary(DictionaryWriterEmitter<'ctx, 'borrow>),
    FixedSizeList(FixedSizeListWriterEmitter<'ctx, 'borrow>),
    Primitive(PrimitiveWriterEmitter<'ctx, 'borrow>),
    List(ListWriterEmitter<'ctx, 'borrow>),
    RunEnd(RunEndWriterEmitter<'ctx, 'borrow>),
    String(StringWriterEmitter<'ctx, 'borrow>),
    StringView(StringViewWriterEmitter<'ctx, 'borrow>),
}

#[enum_dispatch(WriterRuntime)]
pub enum AnyRuntime {
    Boolean(BooleanWriterRuntime),
    Dictionary(DictionaryWriterRuntime),
    FixedSizeList(FixedSizeListWriterRuntime),
    Primitive(PrimitiveWriterRuntime),
    List(ListWriterRuntime),
    RunEnd(RunEndWriterRuntime),
    String(StringWriterRuntime),
    StringView(StringViewWriterRuntime),
}

impl AnyRuntime {
    fn append_integer(&mut self, value: u64) -> Result<(), ArrowKernelError> {
        match self {
            Self::Primitive(runtime) => runtime.append_integer(value),
            _ => Err(ArrowKernelError::InternalError(
                "integer metadata must use a primitive writer runtime".into(),
            )),
        }
    }
}

pub enum WriterPlan {
    Boolean,
    Primitive(PrimitiveType),
    Dictionary(DictionaryKeyType, Box<WriterPlan>),
    RunEnd(RunEndType, Box<WriterPlan>),
    FixedSizeList(usize, Box<WriterPlan>),
    VariableSizeList(Box<WriterPlan>),
    String,
    LargeString,
    StringView,
}

impl WriterPlan {
    /// The logical type being stored. Used by dictionary and REE writers to
    /// know what kind of equality test to use.
    fn storage_type(&self) -> PrimitiveType {
        match self {
            Self::Boolean => PrimitiveType::U8,
            Self::Primitive(pt) => *pt,
            Self::Dictionary(_, values) | Self::RunEnd(_, values) => values.storage_type(),
            Self::FixedSizeList(size, values) => {
                let item_type = match values.as_ref() {
                    Self::Boolean => crate::ListItemType::Boolean,
                    _ => values
                        .storage_type()
                        .try_into()
                        .expect("fixed-size list item type must be scalar"),
                };
                PrimitiveType::List(item_type, *size)
            }
            Self::VariableSizeList(_) => {
                unreachable!("variable-size lists do not have a scalar storage type")
            }
            Self::String | Self::LargeString | Self::StringView => PrimitiveType::P64x2,
        }
    }

    pub fn for_primitive_type(pt: PrimitiveType) -> Self {
        match pt {
            PrimitiveType::I8
            | PrimitiveType::I16
            | PrimitiveType::I32
            | PrimitiveType::I64
            | PrimitiveType::U8
            | PrimitiveType::U16
            | PrimitiveType::U32
            | PrimitiveType::U64
            | PrimitiveType::F16
            | PrimitiveType::F32
            | PrimitiveType::F64 => WriterPlan::Primitive(pt),
            PrimitiveType::P64x2 => WriterPlan::String,
            PrimitiveType::List(lit, s) => WriterPlan::FixedSizeList(
                s,
                Box::new(match lit {
                    crate::ListItemType::Boolean => WriterPlan::Boolean,
                    _ => WriterPlan::for_primitive_type(PrimitiveType::from(lit)),
                }),
            ),
        }
    }

    pub fn for_data_type(dt: &DataType) -> Result<Self, ArrowKernelError> {
        match dt {
            DataType::Null => todo!(),
            DataType::Boolean => Ok(Self::Boolean),
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
            | DataType::Binary
            | DataType::Utf8 => Ok(Self::for_primitive_type(PrimitiveType::for_arrow_type(dt))),
            DataType::LargeBinary | DataType::LargeUtf8 => Ok(Self::LargeString),
            DataType::Utf8View | DataType::BinaryView => Ok(Self::StringView),
            DataType::LargeList(field) | DataType::List(field) => Ok(WriterPlan::VariableSizeList(
                Box::new(Self::for_data_type(field.data_type())?),
            )),
            DataType::FixedSizeList(field, s) => Ok(WriterPlan::FixedSizeList(
                *s as usize,
                Box::new(Self::for_data_type(field.data_type())?),
            )),
            DataType::Dictionary(k, v) => Ok(WriterPlan::Dictionary(
                DictionaryKeyType::for_data_type(k),
                Box::new(Self::for_data_type(v)?),
            )),
            DataType::RunEndEncoded(re, v) => Ok(WriterPlan::RunEnd(
                RunEndType::for_data_type(re.data_type()),
                Box::new(Self::for_data_type(v.data_type())?),
            )),
            _ => todo!(),
        }
    }

    pub fn compile(&self) -> Result<AnyWriter, ArrowKernelError> {
        match self {
            WriterPlan::Boolean => Ok(AnyWriter::Boolean(BooleanWriter::compile())),
            WriterPlan::Primitive(primitive_type) => Ok(AnyWriter::Primitive(
                PrimitiveWriter::compile(*primitive_type)?,
            )),
            WriterPlan::Dictionary(dictionary_key_type, writer_plan) => {
                Ok(AnyWriter::Dictionary(DictionaryWriter::compile(
                    *dictionary_key_type,
                    writer_plan.storage_type(),
                    writer_plan.compile()?,
                )?))
            }
            WriterPlan::RunEnd(run_end_type, writer_plan) => {
                Ok(AnyWriter::RunEnd(RunEndWriter::compile(
                    *run_end_type,
                    writer_plan.storage_type(),
                    writer_plan.compile()?,
                )?))
            }
            WriterPlan::FixedSizeList(_, writer_plan) => Ok(AnyWriter::FixedSizeList(
                FixedSizeListWriter::compile(self.storage_type(), writer_plan.compile()?)?,
            )),
            WriterPlan::VariableSizeList(_) => todo!(),
            WriterPlan::String => Ok(AnyWriter::String(StringWriter::compile(
                PrimitiveType::I32,
            )?)),
            WriterPlan::LargeString => Ok(AnyWriter::String(StringWriter::compile(
                PrimitiveType::I64,
            )?)),
            WriterPlan::StringView => Ok(AnyWriter::StringView(StringViewWriter::compile())),
        }
    }
}

/// Schema-level description of a compiled output writer.
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
            Self::Primitive(pt) => *pt,
            Self::Boolean => PrimitiveType::U8,
            Self::String | Self::LargeString | Self::StringView => PrimitiveType::P64x2,
            Self::FixedSizeList(item, size) => PrimitiveType::List(*item, *size),
            Self::Dictionary(_, values) | Self::RunEndEncoded(_, values) => values.storage_type(),
        }
    }

    fn writer_plan(&self) -> WriterPlan {
        match self {
            Self::Primitive(pt) => WriterPlan::Primitive(*pt),
            Self::Boolean => WriterPlan::Boolean,
            Self::String => WriterPlan::String,
            Self::LargeString => WriterPlan::LargeString,
            Self::StringView => WriterPlan::StringView,
            Self::FixedSizeList(item, size) => WriterPlan::FixedSizeList(
                *size,
                Box::new(match item {
                    ListItemType::Boolean => WriterPlan::Boolean,
                    ListItemType::P64x2 => WriterPlan::String,
                    _ => WriterPlan::Primitive(PrimitiveType::from(*item)),
                }),
            ),
            Self::Dictionary(key, values) => {
                WriterPlan::Dictionary(*key, Box::new(values.writer_plan()))
            }
            Self::RunEndEncoded(run_end, values) => {
                WriterPlan::RunEnd(*run_end, Box::new(values.writer_plan()))
            }
        }
    }

    pub fn compile(&self) -> Result<AnyWriter, ArrowKernelError> {
        self.writer_plan().compile()
    }

    pub fn allocate(&self, expected_count: usize) -> WriterAllocation {
        let writer = self.compile().unwrap();
        WriterAllocation {
            runtime: writer.allocate(expected_count),
        }
    }

    pub fn llvm_init<'ctx, 'borrow>(
        &self,
        ctx: &'ctx Context,
        module: &'borrow Module<'ctx>,
        builder: &'borrow Builder<'ctx>,
        runtime_ptr: PointerValue<'ctx>,
    ) -> BoundWriter<'ctx> {
        let writer = self.compile().unwrap();
        writer.llvm_init(
            WriterCodegen {
                ctx,
                module,
                builder,
            },
            runtime_ptr,
        );
        BoundWriter {
            writer,
            runtime_ptr,
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
                ListItemType::for_arrow_type(field.data_type()),
                len as usize,
            ),
            dt => Self::Primitive(PrimitiveType::for_arrow_type(&dt)),
        }
    }

    pub fn for_data_type(dt: &DataType) -> Self {
        match dt {
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
            DataType::FixedSizeList(field, len) => Self::FixedSizeList(
                ListItemType::for_arrow_type(field.data_type()),
                *len as usize,
            ),
            DataType::Dictionary(key, values) => Self::Dictionary(
                DictionaryKeyType::for_data_type(key),
                Box::new(Self::for_data_type(values)),
            ),
            DataType::RunEndEncoded(run_end, values) => Self::RunEndEncoded(
                RunEndType::for_data_type(run_end.data_type()),
                Box::new(Self::for_data_type(values.data_type())),
            ),
            _ => todo!("unsupported writer data type {dt}"),
        }
    }
}

pub struct WriterAllocation {
    runtime: AnyRuntime,
}

impl WriterAllocation {
    pub fn get_ptr(&mut self) -> *mut c_void {
        self.runtime.as_ptr()
    }

    pub fn reserve_for_additional(&mut self, count: usize) {
        self.runtime.reserve_for_additional(count).unwrap();
    }

    pub fn len(&self) -> usize {
        self.runtime.len()
    }

    pub fn into_array_ref_with_len(self, len: usize, nulls: Option<NullBuffer>) -> ArrayRef {
        let array = self.runtime.to_array(len).unwrap();
        if nulls.is_none() {
            return array;
        }
        let data = array.to_data().into_builder().nulls(nulls);
        make_array(unsafe { data.build_unchecked() })
    }

    pub fn into_array_ref(self, nulls: Option<NullBuffer>) -> ArrayRef {
        let len = self.len();
        self.into_array_ref_with_len(len, nulls)
    }
}

pub struct BoundWriter<'ctx> {
    writer: AnyWriter,
    runtime_ptr: PointerValue<'ctx>,
}

impl<'ctx> BoundWriter<'ctx> {
    pub fn llvm_ingest<'call>(
        &'call self,
        ctx: &'ctx Context,
        module: &'call Module<'ctx>,
        builder: &'call Builder<'ctx>,
        value: BasicValueEnum<'ctx>,
    ) {
        self.writer
            .llvm_write(
                WriterCodegen {
                    ctx,
                    module,
                    builder,
                },
                self.runtime_ptr,
                |emitter| emitter.emit(value),
            )
            .unwrap();
    }

    pub fn llvm_ingest_block<'call>(
        &'call self,
        ctx: &'ctx Context,
        module: &'call Module<'ctx>,
        builder: &'call Builder<'ctx>,
        values: VectorValue<'ctx>,
    ) {
        self.writer
            .llvm_write_multiple(
                WriterCodegen {
                    ctx,
                    module,
                    builder,
                },
                self.runtime_ptr,
                values,
            )
            .unwrap();
    }
}
