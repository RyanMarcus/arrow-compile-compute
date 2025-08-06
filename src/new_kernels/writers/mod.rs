use std::ffi::c_void;

use arrow_array::Array;
use arrow_buffer::NullBuffer;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    values::{BasicValueEnum, PointerValue, VectorValue},
};

mod array_writer;
mod bool_writer;
mod dict_writer;
mod ree_writer;
mod str_writer;
mod view_writer;

pub use array_writer::PrimitiveArrayWriter;
pub use bool_writer::BooleanWriter;
pub use dict_writer::DictWriter;
pub use ree_writer::REEWriter;
pub use str_writer::StringArrayWriter;
pub use view_writer::StringViewWriter;
pub(crate) use view_writer::ViewBufferWriter;

use crate::PrimitiveType;
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

pub trait WriterAllocation {
    type Output: Array;
    fn get_ptr(&mut self) -> *mut c_void;
    fn to_array(self, len: usize, nulls: Option<NullBuffer>) -> Self::Output;
}
