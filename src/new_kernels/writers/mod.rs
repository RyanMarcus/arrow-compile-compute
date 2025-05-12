use inkwell::{builder::Builder, context::Context, values::BasicValueEnum};

mod array_writer;
mod bool_writer;
mod str_writer;

pub use array_writer::PrimitiveArrayWriter;
pub use bool_writer::BooleanWriter;
pub use str_writer::StringArrayWriter;
pub trait ArrayWriter<'a> {
    fn ingest(&self, ctx: &'a Context, build: &Builder<'a>, val: BasicValueEnum<'a>);
    fn flush(&self, ctx: &'a Context, build: &Builder<'a>);
}
