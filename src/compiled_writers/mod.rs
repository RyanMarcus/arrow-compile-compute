//! Composable LLVM-backed writers for building Arrow arrays.
//!
//! Most callers work with [`WriterSpec`], a recursive description of an output
//! column. Create a spec directly or with [`WriterSpec::for_data_type`], call
//! [`WriterSpec::allocate`] to obtain a [`WriterAllocation`], and pass its
//! opaque pointer to the JIT function. During IR generation,
//! [`WriterSpec::llvm_init`] binds the compiled writer to that pointer and
//! returns a [`BoundWriter`], whose ingest methods emit writes. After execution,
//! consume the allocation with [`WriterAllocation::into_array_ref`] to finalize
//! any remaining metadata and produce the Arrow array.
//!
//! The lower-level pieces used to implement a writer are:
//!
//! - [`Writer`], which allocates runtime state and emits the LLVM operations for
//!   a logical write;
//! - [`WriterEmitter`], the per-write interface through which scalar or
//!   composite values are supplied;
//! - [`WriterRuntime`], which owns the Rust buffers, supports reservation, and
//!   materializes the finished array;
//! - [`AnyWriter`], [`AnyWriterEmitter`], and [`AnyRuntime`], which provide
//!   enum-based dispatch so writers can recursively compose child writers.
//!
//! Finalization happens in Rust when the runtime becomes an array; generated
//! code therefore does not need a separate flush step.

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
    compiled_iter::IteratorHolder,
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

    #[cfg(test)]
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

#[repr(u64)]
#[derive(Clone, Copy)]
pub(crate) enum WriterKind {
    Boolean,
    Dictionary,
    FixedSizeList,
    Primitive,
    List,
    RunEnd,
    String,
    StringView,
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

#[no_mangle]
pub(crate) extern "C" fn writer_reserve_for_additional(
    runtime_ptr: *mut c_void,
    count: u64,
    writer_kind: WriterKind,
) {
    let count = count as usize;
    unsafe {
        match writer_kind {
            WriterKind::Boolean => (&mut *runtime_ptr.cast::<BooleanWriterRuntime>())
                .reserve_for_additional(count)
                .unwrap(),
            WriterKind::Dictionary => (&mut *runtime_ptr.cast::<DictionaryWriterRuntime>())
                .reserve_for_additional(count)
                .unwrap(),
            WriterKind::FixedSizeList => (&mut *runtime_ptr.cast::<FixedSizeListWriterRuntime>())
                .reserve_for_additional(count)
                .unwrap(),
            WriterKind::Primitive => (&mut *runtime_ptr.cast::<PrimitiveWriterRuntime>())
                .reserve_for_additional(count)
                .unwrap(),
            WriterKind::List => (&mut *runtime_ptr.cast::<ListWriterRuntime>())
                .reserve_for_additional(count)
                .unwrap(),
            WriterKind::RunEnd => (&mut *runtime_ptr.cast::<RunEndWriterRuntime>())
                .reserve_for_additional(count)
                .unwrap(),
            WriterKind::String => (&mut *runtime_ptr.cast::<StringWriterRuntime>())
                .reserve_for_additional(count)
                .unwrap(),
            WriterKind::StringView => (&mut *runtime_ptr.cast::<StringViewWriterRuntime>())
                .reserve_for_additional(count)
                .unwrap(),
        }
    }
}

impl AnyWriter {
    fn kind(&self) -> WriterKind {
        match self {
            Self::Boolean(_) => WriterKind::Boolean,
            Self::Dictionary(_) => WriterKind::Dictionary,
            Self::FixedSizeList(_) => WriterKind::FixedSizeList,
            Self::Primitive(_) => WriterKind::Primitive,
            Self::List(_) => WriterKind::List,
            Self::RunEnd(_) => WriterKind::RunEnd,
            Self::String(_) => WriterKind::String,
            Self::StringView(_) => WriterKind::StringView,
        }
    }

    fn llvm_reserve_for_additional<'ctx, 'borrow>(
        &self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        count: inkwell::values::IntValue<'ctx>,
    ) {
        // TODO: Generate a writer-specific capacity check in LLVM and call this
        // Rust helper only on the slow path when the current allocation is full.
        let helper = codegen
            .module
            .get_function("writer_reserve_for_additional")
            .unwrap_or_else(|| {
                let ptr_type = codegen.ctx.ptr_type(inkwell::AddressSpace::default());
                codegen.module.add_function(
                    "writer_reserve_for_additional",
                    codegen.ctx.void_type().fn_type(
                        &[
                            ptr_type.into(),
                            codegen.ctx.i64_type().into(),
                            codegen.ctx.i64_type().into(),
                        ],
                        false,
                    ),
                    None,
                )
            });
        codegen
            .builder
            .build_call(
                helper,
                &[
                    runtime_ptr.into(),
                    count.into(),
                    codegen
                        .ctx
                        .i64_type()
                        .const_int(self.kind() as u64, false)
                        .into(),
                ],
                "reserve_writer",
            )
            .unwrap();
    }

    /// Writes a value using the static structure of the iterator that produced it.
    ///
    /// A variable-size list iterator produces a `{ child_iter, start, end }`
    /// descriptor. Copying that descriptor recursively accesses each child and
    /// calls this method again, until a scalar value reaches the ordinary writer.
    pub(crate) fn llvm_write_from_iterator<'ctx, 'borrow>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        source_iter: &IteratorHolder,
        mut value: BasicValueEnum<'ctx>,
    ) -> Result<(), ArrowKernelError> {
        // Dictionary and run-end accessors already return decoded logical values,
        // so continue with the holder for their value iterator.
        match source_iter {
            IteratorHolder::Dictionary { values, .. } => {
                return self.llvm_write_from_iterator(codegen, runtime_ptr, values, value);
            }
            IteratorHolder::RunEnd { values, .. } => {
                return self.llvm_write_from_iterator(codegen, runtime_ptr, values, value);
            }
            _ => {}
        }

        // A list value is a row descriptor; copy its child range into the
        // destination list, recursively handling nested lists.
        if let (AnyWriter::List(writer), IteratorHolder::List(source)) = (self, source_iter) {
            return writer.llvm_write_from_list(codegen, runtime_ptr, source.child(), value);
        }

        if source_iter.data_type() == DataType::Boolean
            && value.into_int_value().get_type().get_bit_width() != 1
        {
            value = codegen
                .builder
                .build_int_truncate(
                    value.into_int_value(),
                    codegen.ctx.bool_type(),
                    "list_child_bool",
                )
                .unwrap()
                .into();
        }

        // Non-list values are recursion leaves and can use the ordinary scalar
        // writer for the destination type.
        self.llvm_write(codegen, runtime_ptr, |emitter| emitter.emit(value))
    }
}

/// A recursive description of an output writer and its composed child writers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum WriterSpec {
    Primitive(PrimitiveType),
    Boolean,
    String,
    LargeString,
    StringView,
    List(Box<WriterSpec>),
    LargeList(Box<WriterSpec>),
    FixedSizeList(Box<WriterSpec>, usize),
    Dictionary(DictionaryKeyType, Box<WriterSpec>),
    RunEndEncoded(RunEndType, Box<WriterSpec>),
}

impl WriterSpec {
    /// Returns the value type accepted by the generated writer code.
    pub fn storage_type(&self) -> PrimitiveType {
        match self {
            Self::Primitive(pt) => *pt,
            Self::Boolean => PrimitiveType::U8,
            Self::String | Self::LargeString | Self::StringView => PrimitiveType::P64x2,
            Self::List(_) | Self::LargeList(_) => {
                unreachable!("variable-size lists do not have a scalar storage type")
            }
            Self::FixedSizeList(values, size) => {
                let item_type = match values.as_ref() {
                    Self::Boolean => ListItemType::Boolean,
                    _ => values
                        .storage_type()
                        .try_into()
                        .expect("fixed-size list item type must be scalar"),
                };
                PrimitiveType::List(item_type, *size)
            }
            Self::Dictionary(_, values) | Self::RunEndEncoded(_, values) => values.storage_type(),
        }
    }

    pub fn for_primitive_type(primitive_type: PrimitiveType) -> Self {
        match primitive_type {
            PrimitiveType::P64x2 => Self::String,
            PrimitiveType::List(item_type, size) => Self::FixedSizeList(
                Box::new(match item_type {
                    ListItemType::Boolean => Self::Boolean,
                    _ => Self::for_primitive_type(PrimitiveType::from(item_type)),
                }),
                size,
            ),
            primitive_type => Self::Primitive(primitive_type),
        }
    }

    pub fn compile(&self) -> Result<AnyWriter, ArrowKernelError> {
        match self {
            Self::Boolean => Ok(AnyWriter::Boolean(BooleanWriter::compile())),
            Self::Primitive(primitive_type) => Ok(AnyWriter::Primitive(PrimitiveWriter::compile(
                *primitive_type,
            )?)),
            Self::Dictionary(dictionary_key_type, values) => {
                Ok(AnyWriter::Dictionary(DictionaryWriter::compile(
                    *dictionary_key_type,
                    values.storage_type(),
                    values.compile()?,
                )?))
            }
            Self::RunEndEncoded(run_end_type, values) => Ok(AnyWriter::RunEnd(
                RunEndWriter::compile(*run_end_type, values.storage_type(), values.compile()?)?,
            )),
            Self::FixedSizeList(values, _) => Ok(AnyWriter::FixedSizeList(
                FixedSizeListWriter::compile(self.storage_type(), values.compile()?)?,
            )),
            Self::List(values) => Ok(AnyWriter::List(ListWriter::compile(
                PrimitiveType::I32,
                values.compile()?,
            )?)),
            Self::LargeList(values) => Ok(AnyWriter::List(ListWriter::compile(
                PrimitiveType::I64,
                values.compile()?,
            )?)),
            Self::String => Ok(AnyWriter::String(StringWriter::compile(
                PrimitiveType::I32,
            )?)),
            Self::LargeString => Ok(AnyWriter::String(StringWriter::compile(
                PrimitiveType::I64,
            )?)),
            Self::StringView => Ok(AnyWriter::StringView(StringViewWriter::compile())),
        }
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
                Box::new(Self::for_data_type(field.data_type())),
                len as usize,
            ),
            DataType::List(field) => Self::List(Box::new(Self::for_data_type(field.data_type()))),
            DataType::LargeList(field) => {
                Self::LargeList(Box::new(Self::for_data_type(field.data_type())))
            }
            dt => Self::for_primitive_type(PrimitiveType::for_arrow_type(&dt)),
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
            | DataType::Interval(_) => Self::for_primitive_type(PrimitiveType::for_arrow_type(dt)),
            DataType::Binary | DataType::FixedSizeBinary(_) | DataType::Utf8 => Self::String,
            DataType::LargeBinary | DataType::LargeUtf8 => Self::LargeString,
            DataType::BinaryView | DataType::Utf8View => Self::StringView,
            DataType::FixedSizeList(field, len) => Self::FixedSizeList(
                Box::new(Self::for_data_type(field.data_type())),
                *len as usize,
            ),
            DataType::List(field) => Self::List(Box::new(Self::for_data_type(field.data_type()))),
            DataType::LargeList(field) => {
                Self::LargeList(Box::new(Self::for_data_type(field.data_type())))
            }
            DataType::Dictionary(key, values) => Self::Dictionary(
                DictionaryKeyType::for_data_type(key),
                Box::new(Self::for_data_type(values)),
            ),
            DataType::Decimal32(_, _) => todo!(),
            DataType::Decimal64(_, _) => todo!(),
            DataType::Decimal128(_, _) => todo!(),
            DataType::Decimal256(_, _) => todo!(),
            DataType::Map(_, _) => todo!(),
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

    pub fn llvm_ingest_from_iterator<'call>(
        &'call self,
        ctx: &'ctx Context,
        module: &'call Module<'ctx>,
        builder: &'call Builder<'ctx>,
        source_iter: &IteratorHolder,
        value: BasicValueEnum<'ctx>,
    ) {
        self.writer
            .llvm_write_from_iterator(
                WriterCodegen {
                    ctx,
                    module,
                    builder,
                },
                self.runtime_ptr,
                source_iter,
                value,
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
