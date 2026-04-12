use std::ffi::c_void;

use arrow_array::ArrayRef;
use arrow_buffer::NullBuffer;
use inkwell::{builder::Builder, context::Context, module::Module, values::PointerValue};

use crate::{
    compiled_kernels::dsl2::DSLType,
    compiled_writers::{Writer, WriterAllocation},
    PrimitiveType,
};

pub use crate::compiled_writers::WriterSpec;

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

    pub fn spec(&self) -> &WriterSpec {
        &self.spec
    }

    pub fn length_tag(&self) -> &str {
        &self.length_tag
    }

    pub fn allocate(&self, expected_count: usize) -> OutputSlot {
        OutputSlot {
            spec: self.spec.clone(),
            length_tag: self.length_tag.clone(),
            alloc: self.spec.allocate(expected_count),
        }
    }
}

pub struct OutputSlot {
    spec: WriterSpec,
    length_tag: String,
    alloc: WriterAllocation,
}

impl OutputSlot {
    pub fn spec(&self) -> &WriterSpec {
        &self.spec
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
    ) -> Writer<'a> {
        self.spec.llvm_init(ctx, llvm_mod, build, alloc_ptr)
    }
}

pub fn accepted_type(spec: &WriterSpec) -> DSLType {
    match spec {
        WriterSpec::Primitive(pt) => DSLType::Primitive(*pt),
        WriterSpec::Boolean => DSLType::Boolean,
        WriterSpec::String | WriterSpec::LargeString | WriterSpec::StringView => {
            DSLType::Primitive(PrimitiveType::P64x2)
        }
        WriterSpec::FixedSizeList(item, size) => {
            DSLType::Primitive(PrimitiveType::List(*item, *size))
        }
        WriterSpec::Dictionary(_, values) | WriterSpec::RunEndEncoded(_, values) => {
            accepted_type(values)
        }
    }
}

#[cfg(test)]
mod tests {
    use inkwell::{context::Context, values::BasicValue, AddressSpace};

    use super::{accepted_type, OutputSpec, WriterSpec};
    use crate::{
        compiled_writers::{DictionaryKeyType, WriterAllocation},
        declare_blocks, ListItemType,
    };

    #[test]
    fn writer_spec_allocates_matching_variant() {
        let primitive =
            OutputSpec::new(WriterSpec::Primitive(crate::PrimitiveType::I32), "rows").allocate(8);
        assert!(matches!(primitive.alloc, WriterAllocation::Primitive(_)));

        let boolean = OutputSpec::new(WriterSpec::Boolean, "rows").allocate(8);
        assert!(matches!(boolean.alloc, WriterAllocation::Boolean(_)));

        let string = OutputSpec::new(WriterSpec::String, "rows").allocate(8);
        assert!(matches!(string.alloc, WriterAllocation::String(_)));

        let list =
            OutputSpec::new(WriterSpec::FixedSizeList(ListItemType::I32, 4), "rows").allocate(8);
        assert!(matches!(list.alloc, WriterAllocation::FixedSizeList(_)));

        let dict = OutputSpec::new(
            WriterSpec::Dictionary(
                DictionaryKeyType::Int8,
                Box::new(WriterSpec::Primitive(crate::PrimitiveType::I32)),
            ),
            "rows",
        )
        .allocate(8);
        assert!(matches!(dict.alloc, WriterAllocation::Dictionary(_)));
    }

    #[test]
    fn output_spec_preserves_length_tag() {
        let spec = OutputSpec::new(WriterSpec::Boolean, "n");
        let slot = spec.allocate(8);

        assert_eq!(spec.spec(), &WriterSpec::Boolean);
        assert_eq!(spec.length_tag(), "n");
        assert_eq!(slot.spec(), &WriterSpec::Boolean);
        assert_eq!(slot.length_tag(), "n");
    }

    #[test]
    fn accepted_type_flattens_composite_writers() {
        let dict = WriterSpec::Dictionary(
            DictionaryKeyType::Int8,
            Box::new(WriterSpec::Primitive(crate::PrimitiveType::I32)),
        );
        assert_eq!(
            accepted_type(&dict),
            crate::compiled_kernels::dsl2::DSLType::Primitive(crate::PrimitiveType::I32)
        );
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
            &WriterSpec::Primitive(crate::PrimitiveType::I32)
        );

        let writer = slot.llvm_init(&ctx, &llvm_mod, &build, dest);
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
        let string_type = crate::PrimitiveType::P64x2
            .llvm_type(&ctx)
            .into_struct_type();
        let null_ptr = ptr_type.const_null();
        let empty = string_type.const_named_struct(&[null_ptr.into(), null_ptr.into()]);
        string_writer.llvm_ingest(&ctx, &build, empty.as_basic_value_enum());
        string_writer.llvm_flush(&ctx, &build);

        let bool_slot = OutputSpec::new(WriterSpec::Boolean, "rows").allocate(4);
        let bool_writer = bool_slot.llvm_init(&ctx, &llvm_mod, &build, bool_dest);
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
