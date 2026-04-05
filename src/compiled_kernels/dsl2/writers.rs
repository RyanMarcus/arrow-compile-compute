use std::ffi::c_void;

use arrow_array::{ArrayRef, Datum};
use arrow_buffer::NullBuffer;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    types::BasicTypeEnum,
    values::{BasicValueEnum, PointerValue, VectorValue},
};

use crate::{
    compiled_kernels::dsl2::DSLType,
    compiled_writers::{
        ArrayOutput, ArrayWriter, BooleanAllocation, BooleanWriter, FixedSizeListWriter,
        FixedSizeListWriterAlloc, PrimitiveArrayWriter, StringAllocation, StringArrayWriter,
        StringViewAllocation, StringViewWriter, WriterAllocation,
    },
    ListItemType, PrimitiveType,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriterSpec {
    Primitive(PrimitiveType),
    Boolean,
    String,
    LargeString,
    StringView,
    FixedSizeList(ListItemType, usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputSpec {
    spec: WriterSpec,
    length_tag: String,
}

impl OutputSpec {
    pub fn new<S: Into<String>>(spec: WriterSpec, length_tag: S) -> Self {
        Self {
            spec,
            length_tag: length_tag.into(),
        }
    }

    pub fn spec(&self) -> WriterSpec {
        self.spec
    }

    pub fn length_tag(&self) -> &str {
        &self.length_tag
    }

    pub fn allocate(&self, expected_count: usize) -> OutputSlot {
        let alloc = match self.spec {
            WriterSpec::Primitive(pt) => OutputWriterAllocation::Primitive(
                PrimitiveArrayWriter::allocate(expected_count, pt),
            ),
            WriterSpec::Boolean => OutputWriterAllocation::Boolean(BooleanWriter::allocate(
                expected_count,
                self.spec.storage_type(),
            )),
            WriterSpec::String => OutputWriterAllocation::String(
                StringArrayWriter::<i32>::allocate(expected_count, self.spec.storage_type()),
            ),
            WriterSpec::LargeString => OutputWriterAllocation::LargeString(
                StringArrayWriter::<i64>::allocate(expected_count, self.spec.storage_type()),
            ),
            WriterSpec::StringView => OutputWriterAllocation::StringView(
                StringViewWriter::allocate(expected_count, self.spec.storage_type()),
            ),
            WriterSpec::FixedSizeList(item, size) => OutputWriterAllocation::FixedSizeList(
                FixedSizeListWriter::allocate(expected_count, PrimitiveType::List(item, size)),
            ),
        };

        OutputSlot {
            spec: self.spec,
            length_tag: self.length_tag.clone(),
            alloc,
        }
    }
}

impl WriterSpec {
    pub fn storage_type(self) -> PrimitiveType {
        match self {
            Self::Primitive(pt) => {
                assert!(
                    !matches!(pt, PrimitiveType::P64x2 | PrimitiveType::List(_, _)),
                    "use a dedicated writer spec for strings and fixed-size lists",
                );
                pt
            }
            Self::Boolean => PrimitiveType::U8,
            Self::String | Self::LargeString | Self::StringView => PrimitiveType::P64x2,
            Self::FixedSizeList(item, size) => PrimitiveType::List(item, size),
        }
    }

    pub fn for_base_type_of_datum(datum: &dyn Datum) -> Self {
        let pt = PrimitiveType::for_arrow_type(datum.get().0.data_type());
        match pt {
            PrimitiveType::P64x2 => Self::String,
            PrimitiveType::List(t, s) => Self::FixedSizeList(t, s),
            _ => Self::Primitive(pt),
        }
    }
}

pub struct OutputSlot {
    spec: WriterSpec,
    length_tag: String,
    alloc: OutputWriterAllocation,
}

impl OutputSlot {
    pub fn spec(&self) -> WriterSpec {
        self.spec
    }

    pub fn length_tag(&self) -> &str {
        &self.length_tag
    }

    pub fn get_ptr(&mut self) -> *mut c_void {
        self.alloc.get_ptr()
    }

    pub fn reserve_for_additional(&mut self, count: usize) {
        self.alloc.reserve_for_additional(count);
    }

    pub fn into_array_ref(self, nulls: Option<NullBuffer>) -> ArrayRef {
        self.alloc.into_array_ref(nulls)
    }

    pub fn llvm_init<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        build: &Builder<'a>,
        alloc_ptr: PointerValue<'a>,
    ) -> OutputWriter<'a> {
        let storage_type = self.spec.storage_type();
        match self.spec {
            WriterSpec::Primitive(_) => OutputWriter::Primitive(PrimitiveArrayWriter::llvm_init(
                ctx,
                llvm_mod,
                build,
                storage_type,
                alloc_ptr,
            )),
            WriterSpec::Boolean => OutputWriter::Boolean(BooleanWriter::llvm_init(
                ctx,
                llvm_mod,
                build,
                storage_type,
                alloc_ptr,
            )),
            WriterSpec::String => OutputWriter::String(StringArrayWriter::<i32>::llvm_init(
                ctx,
                llvm_mod,
                build,
                storage_type,
                alloc_ptr,
            )),
            WriterSpec::LargeString => OutputWriter::LargeString(
                StringArrayWriter::<i64>::llvm_init(ctx, llvm_mod, build, storage_type, alloc_ptr),
            ),
            WriterSpec::StringView => OutputWriter::StringView(StringViewWriter::llvm_init(
                ctx,
                llvm_mod,
                build,
                storage_type,
                alloc_ptr,
            )),
            WriterSpec::FixedSizeList(_, _) => OutputWriter::FixedSizeList(
                FixedSizeListWriter::llvm_init(ctx, llvm_mod, build, storage_type, alloc_ptr),
            ),
        }
    }
}

pub enum OutputWriterAllocation {
    Primitive(ArrayOutput),
    Boolean(BooleanAllocation),
    String(StringAllocation<i32>),
    LargeString(StringAllocation<i64>),
    StringView(StringViewAllocation),
    FixedSizeList(FixedSizeListWriterAlloc),
}

impl OutputWriterAllocation {
    fn get_ptr(&mut self) -> *mut c_void {
        match self {
            Self::Primitive(alloc) => alloc.get_ptr(),
            Self::Boolean(alloc) => alloc.get_ptr(),
            Self::String(alloc) => alloc.get_ptr(),
            Self::LargeString(alloc) => alloc.get_ptr(),
            Self::StringView(alloc) => alloc.get_ptr(),
            Self::FixedSizeList(alloc) => alloc.get_ptr(),
        }
    }

    fn reserve_for_additional(&mut self, count: usize) {
        match self {
            Self::Primitive(alloc) => alloc.reserve_for_additional(count),
            Self::Boolean(alloc) => alloc.reserve_for_additional(count),
            Self::String(alloc) => alloc.reserve_for_additional(count),
            Self::LargeString(alloc) => alloc.reserve_for_additional(count),
            Self::StringView(alloc) => alloc.reserve_for_additional(count),
            Self::FixedSizeList(alloc) => alloc.reserve_for_additional(count),
        }
    }

    fn into_array_ref(self, nulls: Option<NullBuffer>) -> ArrayRef {
        match self {
            Self::Primitive(alloc) => alloc.to_array_ref(nulls),
            Self::Boolean(alloc) => alloc.to_array_ref(nulls),
            Self::String(alloc) => alloc.to_array_ref(nulls),
            Self::LargeString(alloc) => alloc.to_array_ref(nulls),
            Self::StringView(alloc) => alloc.to_array_ref(nulls),
            Self::FixedSizeList(alloc) => alloc.to_array_ref(nulls),
        }
    }
}

pub enum OutputWriter<'a> {
    Primitive(PrimitiveArrayWriter<'a>),
    Boolean(BooleanWriter<'a>),
    String(StringArrayWriter<'a, i32>),
    LargeString(StringArrayWriter<'a, i64>),
    StringView(StringViewWriter<'a>),
    FixedSizeList(FixedSizeListWriter<'a>),
}

impl<'a> OutputWriter<'a> {
    pub fn llvm_ingest_type(&self, ctx: &'a Context) -> BasicTypeEnum<'a> {
        match self {
            Self::Primitive(writer) => writer.llvm_ingest_type(ctx),
            Self::Boolean(writer) => writer.llvm_ingest_type(ctx),
            Self::String(writer) => writer.llvm_ingest_type(ctx),
            Self::LargeString(writer) => writer.llvm_ingest_type(ctx),
            Self::StringView(writer) => writer.llvm_ingest_type(ctx),
            Self::FixedSizeList(writer) => writer.llvm_ingest_type(ctx),
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
        }
    }

    pub fn accepted_type(&self) -> DSLType {
        match self {
            OutputWriter::Primitive(w) => DSLType::Primitive(w.primitive_type()),
            OutputWriter::Boolean(_) => DSLType::Boolean,
            OutputWriter::String(_) => DSLType::Primitive(PrimitiveType::P64x2),
            OutputWriter::LargeString(_) => DSLType::Primitive(PrimitiveType::P64x2),
            OutputWriter::StringView(_) => DSLType::Primitive(PrimitiveType::P64x2),
            OutputWriter::FixedSizeList(w) => DSLType::Primitive(w.primitive_type()),
        }
    }
}

#[cfg(test)]
mod tests {
    use inkwell::{context::Context, values::BasicValue, AddressSpace};

    use super::{OutputSpec, OutputWriter, OutputWriterAllocation, WriterSpec};
    use crate::{declare_blocks, ListItemType};

    #[test]
    fn writer_spec_allocates_matching_variant() {
        let primitive =
            OutputSpec::new(WriterSpec::Primitive(crate::PrimitiveType::I32), "rows").allocate(8);
        assert!(matches!(
            primitive.alloc,
            OutputWriterAllocation::Primitive(_)
        ));

        let boolean = OutputSpec::new(WriterSpec::Boolean, "rows").allocate(8);
        assert!(matches!(boolean.alloc, OutputWriterAllocation::Boolean(_)));

        let string = OutputSpec::new(WriterSpec::String, "rows").allocate(8);
        assert!(matches!(string.alloc, OutputWriterAllocation::String(_)));

        let list =
            OutputSpec::new(WriterSpec::FixedSizeList(ListItemType::I32, 4), "rows").allocate(8);
        assert!(matches!(
            list.alloc,
            OutputWriterAllocation::FixedSizeList(_)
        ));
    }

    #[test]
    fn output_spec_preserves_length_tag() {
        let spec = OutputSpec::new(WriterSpec::Boolean, "n");
        let slot = spec.allocate(8);

        assert_eq!(spec.spec(), WriterSpec::Boolean);
        assert_eq!(spec.length_tag(), "n");
        assert_eq!(slot.spec(), WriterSpec::Boolean);
        assert_eq!(slot.length_tag(), "n");
    }

    #[test]
    fn primitive_writer_codegen_smoke() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("dsl2_output_writer_primitive");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());

        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );
        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);

        let dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let slot =
            OutputSpec::new(WriterSpec::Primitive(crate::PrimitiveType::I32), "rows").allocate(4);
        assert_eq!(
            slot.spec(),
            WriterSpec::Primitive(crate::PrimitiveType::I32)
        );

        let writer = slot.llvm_init(&ctx, &llvm_mod, &build, dest);
        assert!(matches!(writer, OutputWriter::Primitive(_)));
        writer.llvm_ingest(
            &ctx,
            &build,
            ctx.i32_type().const_int(7, true).as_basic_value_enum(),
        );
        writer.llvm_flush(&ctx, &build);

        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();
    }

    #[test]
    fn string_and_boolean_codegen_smoke() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("dsl2_output_writer_string_bool");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());

        let func = llvm_mod.add_function(
            "test",
            ctx.void_type()
                .fn_type(&[ptr_type.into(), ptr_type.into()], false),
            None,
        );
        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);

        let string_dest = func.get_nth_param(0).unwrap().into_pointer_value();
        let bool_dest = func.get_nth_param(1).unwrap().into_pointer_value();

        let string_slot = OutputSpec::new(WriterSpec::String, "rows").allocate(4);
        let string_writer = string_slot.llvm_init(&ctx, &llvm_mod, &build, string_dest);
        assert!(matches!(string_writer, OutputWriter::String(_)));
        let string_type = crate::PrimitiveType::P64x2
            .llvm_type(&ctx)
            .into_struct_type();
        let null_ptr = ptr_type.const_null();
        let empty = string_type.const_named_struct(&[null_ptr.into(), null_ptr.into()]);
        string_writer.llvm_ingest(&ctx, &build, empty.as_basic_value_enum());
        string_writer.llvm_flush(&ctx, &build);

        let bool_slot = OutputSpec::new(WriterSpec::Boolean, "rows").allocate(4);
        let bool_writer = bool_slot.llvm_init(&ctx, &llvm_mod, &build, bool_dest);
        assert!(matches!(bool_writer, OutputWriter::Boolean(_)));
        bool_writer.llvm_ingest(
            &ctx,
            &build,
            ctx.bool_type().const_int(1, false).as_basic_value_enum(),
        );
        bool_writer.llvm_flush(&ctx, &build);

        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();
    }
}
