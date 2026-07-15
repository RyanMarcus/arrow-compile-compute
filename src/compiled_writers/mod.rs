//! LLVM-backed Arrow array writers.
//!
//! This module is the bridge between the Rust objects that own output buffers
//! and the LLVM IR that appends values into those buffers. There are four
//! related concepts:
//!
//! 1. [`WriterSpec`] describes the logical output type. It is cheap, cloneable
//!    metadata such as "primitive i32", "string", or "dictionary with i32
//!    values".
//! 2. [`WriterAllocation`] owns the Rust-side allocation for one output column.
//!    The allocation is created before the JIT function runs, passed to LLVM as
//!    an opaque pointer, and converted into an Arrow array after the JIT
//!    function returns.
//! 3. [`Writer`] owns the generated writer logic for a spec. It may contain
//!    LLVM helper functions and nested child writers, but it is deliberately not
//!    the public append API because it is not tied to a particular allocation
//!    pointer.
//! 4. [`BoundWriter`] is a writer bound to the allocation pointer that is valid
//!    in the current LLVM IR context. It is the public IR-generation API for
//!    appending values and flushing buffered state.
//!
//! The distinction between [`Writer`] and [`BoundWriter`] is important. An LLVM
//! [`PointerValue`] is an SSA value: a pointer computed in one function, helper,
//! or block is not automatically valid in another. Earlier versions of this
//! module hid the current allocation pointer in a thread-local LLVM global, but
//! that was fragile for JIT linking on macOS and also made the active allocation
//! implicit. The current design passes allocation pointers explicitly and makes
//! Rust code bind writer logic to the pointer that is valid at the place where
//! IR is being emitted.
//!
//! A typical consumer follows this lifecycle:
//!
//! ```ignore
//! use arrow_compile_compute::{compiled_writers::WriterSpec, PrimitiveType};
//!
//! let spec = WriterSpec::Primitive(PrimitiveType::I32);
//! let mut allocation = spec.allocate(expected_rows);
//!
//! // Pass this pointer to the compiled function as an opaque output pointer.
//! let output_ptr = allocation.get_ptr();
//!
//! // After the compiled function has run, consume the allocation as an Arrow
//! // array. The length normally comes from the kernel's output-length logic.
//! let array = allocation.into_array_ref_with_len(output_len, None);
//! ```
//!
//! During IR generation, the opaque output pointer is materialized as an LLVM
//! [`PointerValue`] and bound to the writer:
//!
//! ```ignore
//! let writer = spec.llvm_init(ctx, llvm_mod, build, output_alloc_ptr);
//!
//! writer.llvm_ingest(ctx, build, value);
//! writer.llvm_flush(ctx, build);
//! ```
//!
//! Composite writers use the same rule recursively. If a dictionary or
//! run-end-encoded helper derives a child allocation pointer from the parent
//! allocation, it calls [`BoundWriter::with_alloc`] and emits child writes
//! through that rebound writer:
//!
//! ```ignore
//! child_writer.with_alloc(child_alloc_ptr).llvm_ingest(ctx, build, child_value);
//! ```
//!
//! The author of IR-generation code is responsible for ensuring that a
//! [`BoundWriter`] is used only where its allocation pointer is valid LLVM IR.
//! Misuse should be caught by LLVM verification or compilation; the Rust object
//! itself does not extend the lifetime or dominance of the underlying SSA value.

use std::ffi::c_void;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Datum};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
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

/// Logical description of an output writer.
///
/// A `WriterSpec` is schema-like metadata. Use it to allocate the Rust-side
/// output storage with [`WriterSpec::allocate`] and to create LLVM writer logic
/// with [`WriterSpec::llvm_init`].
///
/// ```ignore
/// let spec = WriterSpec::for_data_type(&arrow_schema::DataType::Int32);
/// let mut allocation = spec.allocate(expected_rows);
/// ```
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

    /// Allocate Rust-side output storage for this writer spec.
    ///
    /// The returned allocation owns the buffers that the compiled function will
    /// mutate. Pass [`WriterAllocation::get_ptr`] to the JIT function, then
    /// consume the allocation with [`WriterAllocation::into_array_ref`] or
    /// [`WriterAllocation::into_array_ref_with_len`].
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

    /// Create the unbound LLVM writer logic for this spec.
    ///
    /// This is crate-private because callers should normally use
    /// [`WriterSpec::llvm_init`], which binds the writer to the allocation
    /// pointer immediately. Composite writers use this lower-level form to
    /// create child writer logic once and later bind it to child allocation
    /// pointers derived inside helper functions.
    pub(crate) fn llvm_writer<'a>(
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

    /// Create a [`BoundWriter`] for this spec and allocation pointer.
    ///
    /// The `alloc_ptr` must be valid in the LLVM function or helper where
    /// subsequent [`BoundWriter::llvm_ingest`] and [`BoundWriter::llvm_flush`]
    /// calls are emitted.
    pub fn llvm_init<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        alloc_ptr: PointerValue<'a>,
    ) -> BoundWriter<'a, 'a> {
        self.llvm_writer(ctx, llvm_mod, build, alloc_ptr)
            .bind(alloc_ptr)
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
                ListItemType::for_arrow_type(field.data_type()),
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
            DataType::Decimal32(_, _) => todo!(),
            DataType::Decimal64(_, _) => todo!(),
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

/// Rust-side storage owned by an output writer.
///
/// `WriterAllocation` values live outside the generated LLVM function. The JIT
/// receives [`WriterAllocation::get_ptr`] as an opaque pointer, writes into it,
/// and the Rust caller later consumes the allocation as an Arrow array.
///
/// ```ignore
/// let mut allocation = spec.allocate(expected_rows);
/// let raw_ptr = allocation.get_ptr();
///
/// run_compiled_kernel(raw_ptr);
///
/// let array = allocation.into_array_ref_with_len(output_len, nulls);
/// ```
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

/// Generated writer logic that is not bound to an allocation pointer.
///
/// This type owns LLVM helper functions and nested writer state, but it does
/// not expose public ingest or flush methods. Bind it to a valid LLVM allocation
/// pointer with [`Writer::bind`] before emitting writes.
///
/// Most callers should use [`WriterSpec::llvm_init`], which creates a
/// [`Writer`] and immediately returns a [`BoundWriter`].
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
    /// Attach this writer logic to the allocation pointer currently valid in
    /// the emitted LLVM IR.
    pub fn bind(self, alloc_ptr: PointerValue<'a>) -> BoundWriter<'a, 'a> {
        BoundWriter {
            writer: BoundWriterTarget::Owned(self),
            alloc_ptr,
        }
    }

    fn emit_ingest(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        alloc_ptr: PointerValue<'a>,
        val: BasicValueEnum<'a>,
    ) {
        match self {
            Self::Primitive(writer) => writer.emit_ingest(build, alloc_ptr, val),
            Self::Boolean(writer) => writer.emit_ingest(ctx, build, alloc_ptr, val),
            Self::String(writer) => writer.emit_ingest(build, alloc_ptr, val),
            Self::LargeString(writer) => writer.emit_ingest(build, alloc_ptr, val),
            Self::StringView(writer) => writer.emit_ingest(build, alloc_ptr, val),
            Self::FixedSizeList(writer) => writer.emit_ingest(ctx, build, alloc_ptr, val),
            Self::Dictionary(writer) => writer.emit_ingest(ctx, build, alloc_ptr, val),
            Self::RunEndEncoded(writer) => writer.emit_ingest(ctx, build, alloc_ptr, val),
        }
    }

    fn emit_ingest_block(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        alloc_ptr: PointerValue<'a>,
        vals: VectorValue<'a>,
    ) {
        match self {
            Self::Primitive(writer) => writer.emit_ingest_block(build, alloc_ptr, vals),
            Self::Boolean(writer) => writer.emit_ingest_block(ctx, build, alloc_ptr, vals),
            Self::String(_)
            | Self::LargeString(_)
            | Self::StringView(_)
            | Self::FixedSizeList(_) => {
                let i64_type = ctx.i64_type();
                for idx in 0..vals.get_type().get_size() {
                    let val = build
                        .build_extract_element(
                            vals,
                            i64_type.const_int(idx as u64, false),
                            &format!("val{}", idx),
                        )
                        .unwrap();
                    self.emit_ingest(ctx, build, alloc_ptr, val);
                }
            }
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
                    writer.emit_ingest(ctx, build, alloc_ptr, val);
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
                    writer.emit_ingest(ctx, build, alloc_ptr, val);
                }
            }
        }
    }

    fn emit_flush(&self, ctx: &'a Context, build: &Builder<'a>, alloc_ptr: PointerValue<'a>) {
        match self {
            Self::Primitive(writer) => writer.emit_flush(ctx, build, alloc_ptr),
            Self::Boolean(writer) => writer.emit_flush(ctx, build, alloc_ptr),
            Self::String(writer) => writer.emit_flush(build, alloc_ptr),
            Self::LargeString(writer) => writer.emit_flush(build, alloc_ptr),
            Self::StringView(writer) => writer.emit_flush(build, alloc_ptr),
            Self::FixedSizeList(writer) => writer.emit_flush(ctx, build, alloc_ptr),
            Self::Dictionary(writer) => writer.emit_flush(ctx, build, alloc_ptr),
            Self::RunEndEncoded(writer) => writer.emit_flush(ctx, build, alloc_ptr),
        }
    }
}

enum BoundWriterTarget<'a, 'w> {
    Owned(Writer<'a>),
    Borrowed(&'w Writer<'a>),
}

/// Writer logic attached to a specific LLVM allocation pointer.
///
/// `BoundWriter` is the public API for appending to an output allocation during
/// IR generation. The allocation pointer must be valid at the point where the
/// emitted call is inserted.
///
/// ```ignore
/// let writer = spec.llvm_init(ctx, llvm_mod, build, output_alloc_ptr);
/// writer.llvm_ingest(ctx, build, value);
/// writer.llvm_flush(ctx, build);
/// ```
pub struct BoundWriter<'a, 'w> {
    writer: BoundWriterTarget<'a, 'w>,
    alloc_ptr: PointerValue<'a>,
}

impl<'a, 'w> BoundWriter<'a, 'w> {
    fn writer(&self) -> &Writer<'a> {
        match &self.writer {
            BoundWriterTarget::Owned(writer) => writer,
            BoundWriterTarget::Borrowed(writer) => writer,
        }
    }

    /// Reuse this writer logic with another allocation pointer.
    ///
    /// This is primarily for composite writers. For example, a dictionary
    /// writer can derive the keys and values allocation pointers from its
    /// parent allocation and temporarily bind child writer logic to those
    /// derived pointers before emitting child appends.
    pub fn with_alloc<'b>(&'b self, alloc_ptr: PointerValue<'a>) -> BoundWriter<'a, 'b> {
        BoundWriter {
            writer: BoundWriterTarget::Borrowed(self.writer()),
            alloc_ptr,
        }
    }

    /// Emit IR that appends one value into this bound allocation.
    pub fn llvm_ingest(&self, ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>) {
        self.writer().emit_ingest(ctx, build, self.alloc_ptr, val);
    }

    /// Emit IR that appends each lane of a vector value into this bound
    /// allocation.
    pub fn llvm_ingest_block(&self, ctx: &'a Context, build: &Builder<'a>, vals: VectorValue<'a>) {
        self.writer()
            .emit_ingest_block(ctx, build, self.alloc_ptr, vals);
    }

    /// Emit IR that flushes any buffered writer state into this bound
    /// allocation.
    pub fn llvm_flush(&self, ctx: &'a Context, build: &Builder<'a>) {
        self.writer().emit_flush(ctx, build, self.alloc_ptr);
    }
}
