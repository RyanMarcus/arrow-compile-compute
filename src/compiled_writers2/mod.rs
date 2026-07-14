mod boolean_writer;
mod dictionary_writer;
mod fixed_size_list_writer;
mod list_writer;
mod primitive_writer;
mod run_end_writer;
mod string_writer;
mod view_writer;

use std::ffi::c_void;

use arrow_array::ArrayRef;
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
    FixedSizeListWriter,
    PrimitiveWriter,
    ListWriter,
    RunEndWriter,
    StringWriter,
    StringViewWriter,
}

#[enum_dispatch(WriterEmitter)]
pub enum AnyWriterEmitter<'ctx, 'borrow> {
    BooleanWriterEmitter(BooleanWriterEmitter<'ctx, 'borrow>),
    DictionaryWriterEmitter(DictionaryWriterEmitter<'ctx, 'borrow>),
    FixedSizeListWriterEmitter(FixedSizeListWriterEmitter<'ctx, 'borrow>),
    PrimitiveWriterEmitter(PrimitiveWriterEmitter<'ctx, 'borrow>),
    ListWriterEmitter(ListWriterEmitter<'ctx, 'borrow>),
    RunEndWriterEmitter(RunEndWriterEmitter<'ctx, 'borrow>),
    StringWriterEmitter(StringWriterEmitter<'ctx, 'borrow>),
    StringViewWriterEmitter(StringViewWriterEmitter<'ctx, 'borrow>),
}

#[enum_dispatch(WriterRuntime)]
pub enum AnyRuntime {
    BooleanWriterRuntime,
    DictionaryWriterRuntime,
    FixedSizeListWriterRuntime,
    PrimitiveWriterRuntime,
    ListWriterRuntime,
    RunEndWriterRuntime,
    StringWriterRuntime,
    StringViewWriterRuntime,
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
            WriterPlan::FixedSizeList(_, writer_plan) => Ok(AnyWriter::FixedSizeListWriter(
                FixedSizeListWriter::compile(self.storage_type(), writer_plan.compile()?)?,
            )),
            WriterPlan::VariableSizeList(_) => todo!(),
            WriterPlan::String => Ok(AnyWriter::StringWriter(StringWriter::compile(
                PrimitiveType::I32,
            )?)),
            WriterPlan::StringView => Ok(AnyWriter::StringViewWriter(StringViewWriter::compile())),
        }
    }
}
