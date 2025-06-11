use std::{ffi::c_void, sync::Arc};

use arrow_array::Array;
use arrow_buffer::NullBuffer;
use enum_as_inner::EnumAsInner;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    values::{BasicValueEnum, PointerValue},
};

mod array_writer;
mod bool_writer;
mod dict_writer;
mod str_writer;

pub use array_writer::PrimitiveArrayWriter;
pub use bool_writer::BooleanWriter;
pub use dict_writer::DictWriter;
pub use str_writer::StringArrayWriter;

pub enum EWriterAllocation {
    Primitive(EArrayOutput),
    Boolean(EBooleanAllocation),
    String(EStringAllocation<i32>),
    LargeString(EStringAllocation<i64>),
}

impl EWriterAllocation {
    pub fn to_array(self, len: usize, nulls: Option<NullBuffer>) -> Arc<dyn Array> {
        match self {
            EWriterAllocation::Primitive(output) => output.to_array(len, nulls),
            EWriterAllocation::Boolean(output) => Arc::new(output.to_array(len, nulls)),
            EWriterAllocation::String(output) => Arc::new(output.to_array(len, nulls)),
            EWriterAllocation::LargeString(output) => Arc::new(output.to_array(len, nulls)),
        }
    }

    pub fn get_ptr(&mut self) -> *mut c_void {
        match self {
            EWriterAllocation::Primitive(output) => output.get_ptr(),
            EWriterAllocation::Boolean(output) => output.get_ptr(),
            EWriterAllocation::String(output) => output.get_ptr(),
            EWriterAllocation::LargeString(output) => output.get_ptr(),
        }
    }
}

pub enum EArrayWriter<'a> {
    Primitive(EPrimitiveArrayWriter<'a>),
    Boolean(EBooleanWriter<'a>),
    //Dictionary(EDictWriter<'a>),
    String(EStringArrayWriter<'a, i32>),
    LargeString(EStringArrayWriter<'a, i64>),
}

impl<'a> EArrayWriter<'a> {
    pub fn llvm_ingest(&self, ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>) {
        match self {
            EArrayWriter::Primitive(w) => {
                w.llvm_ingest(ctx, build, val);
            }
            EArrayWriter::Boolean(w) => {
                w.llvm_ingest(ctx, build, val);
            }
            EArrayWriter::String(w) => {
                w.llvm_ingest(ctx, build, val);
            }
            EArrayWriter::LargeString(w) => {
                w.llvm_ingest(ctx, build, val);
            }
        }
    }
    pub fn llvm_flush(&self, ctx: &'a Context, build: &Builder<'a>) {
        match self {
            EArrayWriter::Primitive(w) => {
                w.llvm_flush(ctx, build);
            }
            EArrayWriter::Boolean(w) => {
                w.llvm_flush(ctx, build);
            }
            EArrayWriter::String(w) => {
                w.llvm_flush(ctx, build);
            }
            EArrayWriter::LargeString(w) => {
                w.llvm_flush(ctx, build);
            }
        }
    }
}

use crate::{
    new_kernels::writers::{
        array_writer::{ArrayOutput, EArrayOutput, EPrimitiveArrayWriter},
        bool_writer::{BooleanAllocation, EBooleanAllocation, EBooleanWriter},
        str_writer::{EStringAllocation, EStringArrayWriter, StringAllocation},
    },
    PrimitiveType,
};
pub trait ArrayWriter<'a> {
    type Allocation: WriterAllocation;
    fn allocate(expected_count: usize, ty: PrimitiveType) -> Self::Allocation;

    fn llvm_init(
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        ty: PrimitiveType,
        alloc_ptr: PointerValue<'a>,
    ) -> Self;
    fn llvm_ingest(&self, ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>);
    fn llvm_flush(&self, ctx: &'a Context, build: &Builder<'a>);
}

pub trait WriterAllocation {
    type Output: Array;
    fn get_ptr(&mut self) -> *mut c_void;
    fn to_array(self, len: usize, nulls: Option<NullBuffer>) -> Self::Output;
}
