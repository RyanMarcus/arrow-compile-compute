mod boolean_writer;
mod dictionary_writer;
mod list_writer;
mod primitive_writer;
mod run_end_writer;
mod string_writer;

use std::ffi::c_void;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use enum_dispatch::enum_dispatch;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    values::{BasicValueEnum, PointerValue},
};

use dictionary_writer::{DictionaryWriter, DictionaryWriterEmitter, DictionaryWriterRuntime};
use list_writer::{ListWriter, ListWriterEmitter, ListWriterRuntime};
use run_end_writer::{RunEndWriter, RunEndWriterEmitter, RunEndWriterRuntime};

use crate::{
    compiled_writers::{DictionaryKeyType, RunEndType},
    compiled_writers2::string_writer::{StringWriter, StringWriterEmitter, StringWriterRuntime},
    ArrowKernelError, PrimitiveType,
};

pub use boolean_writer::{BooleanWriter, BooleanWriterEmitter, BooleanWriterRuntime};
pub use primitive_writer::{PrimitiveWriter, PrimitiveWriterEmitter, PrimitiveWriterRuntime};

#[enum_dispatch]
pub trait WriterRuntime {
    fn as_ptr(&mut self) -> *mut c_void;

    /// Reserves space for future writes and refreshes any child runtime pointers.
    ///
    /// This may only be called before [`Writer::llvm_flush`] has executed for
    /// this runtime. A flushed runtime is finalized and may only be converted
    /// into an array.
    fn reserve_for_additional(&mut self, count: usize) -> Result<(), ArrowKernelError>;
    fn len(&self) -> usize;
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
    fn allocate(&self, size: usize) -> AnyRuntime;

    fn llvm_init<'ctx, 'borrow>(
        &self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
    );

    fn llvm_write<'ctx, 'borrow, F>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        f: F,
    ) -> Result<(), ArrowKernelError>
    where
        F: Fn(&mut AnyWriterEmitter<'ctx, 'borrow>) -> Result<(), ArrowKernelError>;

    /// Finalizes this writer and all composed child writers.
    ///
    /// This must be emitted exactly once, after all writes and runtime
    /// reservations are complete. The runtime cannot be written to or resized
    /// after the generated flush code has executed.
    fn llvm_flush<'ctx, 'borrow>(
        &'borrow self,
        _codegen: WriterCodegen<'ctx, 'borrow>,
        _runtime_ptr: PointerValue<'ctx>,
    ) {
    }
}

#[enum_dispatch(Writer)]
pub enum AnyWriter {
    BooleanWriter,
    DictionaryWriter,
    PrimitiveWriter,
    ListWriter,
    RunEndWriter,
    StringWriter,
}

#[enum_dispatch(WriterEmitter)]
pub enum AnyWriterEmitter<'ctx, 'borrow> {
    BooleanWriterEmitter(BooleanWriterEmitter<'ctx, 'borrow>),
    DictionaryWriterEmitter(DictionaryWriterEmitter<'ctx, 'borrow>),
    PrimitiveWriterEmitter(PrimitiveWriterEmitter<'ctx, 'borrow>),
    ListWriterEmitter(ListWriterEmitter<'ctx, 'borrow>),
    RunEndWriterEmitter(RunEndWriterEmitter<'ctx, 'borrow>),
    StringWriterEmitter(StringWriterEmitter<'ctx, 'borrow>),
}

#[enum_dispatch(WriterRuntime)]
pub enum AnyRuntime {
    BooleanWriterRuntime,
    DictionaryWriterRuntime,
    PrimitiveWriterRuntime,
    ListWriterRuntime,
    RunEndWriterRuntime,
    StringWriterRuntime,
}

pub enum WriterPlan {
    Boolean,
    Primitive(PrimitiveType),
    Dictionary(DictionaryKeyType, Box<WriterPlan>),
    RunEnd(RunEndType, Box<WriterPlan>),
    FixedSizeList(usize, Box<WriterPlan>),
    VariableSizeList(Box<WriterPlan>),
    String,
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
            Self::FixedSizeList(size, values) => PrimitiveType::List(
                values
                    .storage_type()
                    .try_into()
                    .expect("fixed-size list item type must be scalar"),
                *size,
            ),
            Self::VariableSizeList(_) => {
                unreachable!("variable-size lists do not have a scalar storage type")
            }
            Self::String | Self::StringView => PrimitiveType::P64x2,
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
                Box::new(WriterPlan::for_primitive_type(PrimitiveType::from(lit))),
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
            | DataType::LargeBinary
            | DataType::Utf8
            | DataType::LargeUtf8 => {
                Ok(Self::for_primitive_type(PrimitiveType::for_arrow_type(dt)))
            }
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
            WriterPlan::Boolean => Ok(AnyWriter::BooleanWriter(BooleanWriter::compile())),
            WriterPlan::Primitive(primitive_type) => Ok(AnyWriter::PrimitiveWriter(
                PrimitiveWriter::compile(*primitive_type)?,
            )),
            WriterPlan::Dictionary(dictionary_key_type, writer_plan) => {
                Ok(AnyWriter::DictionaryWriter(DictionaryWriter::compile(
                    *dictionary_key_type,
                    writer_plan.storage_type(),
                    writer_plan.compile()?,
                )?))
            }
            WriterPlan::RunEnd(run_end_type, writer_plan) => {
                Ok(AnyWriter::RunEndWriter(RunEndWriter::compile(
                    *run_end_type,
                    writer_plan.storage_type(),
                    writer_plan.compile()?,
                )?))
            }
            WriterPlan::FixedSizeList(_, _) => todo!(),
            WriterPlan::VariableSizeList(_) => todo!(),
            WriterPlan::String => Ok(AnyWriter::StringWriter(StringWriter::compile(
                PrimitiveType::I32,
            )?)),
            WriterPlan::StringView => todo!(),
        }
    }
}
