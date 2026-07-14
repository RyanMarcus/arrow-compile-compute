//mod boolean_writer;
mod list_writer;
mod primitive_writer;
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

use list_writer::{ListWriter, ListWriterEmitter, ListWriterRuntime};

use crate::{
    compiled_writers::{DictionaryKeyType, RunEndType},
    compiled_writers2::string_writer::{StringWriter, StringWriterEmitter, StringWriterRuntime},
    ArrowKernelError, PrimitiveType,
};

pub use primitive_writer::{PrimitiveWriter, PrimitiveWriterEmitter, PrimitiveWriterRuntime};

#[enum_dispatch]
pub trait WriterRuntime {
    fn as_ptr(&mut self) -> *mut c_void;
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

#[enum_dispatch]
pub trait Writer {
    fn allocate(&self, size: usize) -> AnyRuntime;

    fn llvm_init<'a>(&self, ctx: &'a Context, build: &Builder<'a>, runtime_ptr: PointerValue<'a>);

    fn llvm_write<'ctx, 'borrow, F>(
        &'borrow self,
        ctx: &'ctx Context,
        module: &'borrow Module<'ctx>,
        build: &'borrow Builder<'ctx>,
        runtime_ptr: PointerValue<'ctx>,
        f: F,
    ) -> Result<(), ArrowKernelError>
    where
        F: Fn(&mut AnyWriterEmitter<'ctx, 'borrow>) -> Result<(), ArrowKernelError>;

    fn llvm_flush<'ctx, 'borrow>(
        &'borrow self,
        ctx: &'ctx Context,
        module: &'borrow Module<'ctx>,
        build: &'borrow Builder<'ctx>,
        runtime_ptr: PointerValue<'ctx>,
    ) {
    }
}

#[enum_dispatch(Writer)]
pub enum AnyWriter {
    PrimitiveWriter,
    ListWriter,
    StringWriter,
}

#[enum_dispatch(WriterEmitter)]
pub enum AnyWriterEmitter<'ctx, 'borrow> {
    PrimitiveWriterEmitter(PrimitiveWriterEmitter<'ctx, 'borrow>),
    ListWriterEmitter(ListWriterEmitter<'ctx, 'borrow>),
    StringWriterEmitter(StringWriterEmitter<'ctx, 'borrow>),
}

#[enum_dispatch(WriterRuntime)]
pub enum AnyRuntime {
    PrimitiveWriterRuntime,
    ListWriterRuntime,
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
            DataType::Boolean
            | DataType::Int8
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
            WriterPlan::Boolean => todo!(),
            WriterPlan::Primitive(primitive_type) => Ok(AnyWriter::PrimitiveWriter(
                PrimitiveWriter::compile(*primitive_type)?,
            )),
            WriterPlan::Dictionary(dictionary_key_type, writer_plan) => todo!(),
            WriterPlan::RunEnd(run_end_type, writer_plan) => todo!(),
            WriterPlan::FixedSizeList(_, writer_plan) => todo!(),
            WriterPlan::VariableSizeList(writer_plan) => todo!(),
            WriterPlan::String => Ok(AnyWriter::StringWriter(StringWriter::compile(
                PrimitiveType::I32,
            )?)),
            WriterPlan::StringView => todo!(),
        }
    }
}
