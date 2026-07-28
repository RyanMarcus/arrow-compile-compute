use std::cmp::Ordering;

use inkwell::{
    builder::Builder,
    context::Context,
    types::BasicType,
    values::{BasicValueEnum, InstructionOpcode},
};

use crate::{NumericPrimitiveType, PrimitiveType};

pub(crate) fn cast_numeric<'ctx>(
    ctx: &'ctx Context,
    builder: &Builder<'ctx>,
    value: BasicValueEnum<'ctx>,
    source_type: NumericPrimitiveType,
    target_type: NumericPrimitiveType,
) -> BasicValueEnum<'ctx> {
    let target_primitive = PrimitiveType::from(target_type);
    let target_llvm_type = if value.is_vector_value() {
        target_primitive
            .llvm_vec_type(ctx, value.get_type().into_vector_type().get_size())
            .unwrap()
            .as_basic_type_enum()
    } else {
        target_primitive.llvm_type(ctx)
    };

    let opcode = match (source_type.is_integer(), target_type.is_integer()) {
        (true, true) => match target_type.width().cmp(&source_type.width()) {
            Ordering::Less => InstructionOpcode::Trunc,
            Ordering::Equal => return value,
            Ordering::Greater if source_type.is_signed() => InstructionOpcode::SExt,
            Ordering::Greater => InstructionOpcode::ZExt,
        },
        (true, false) if source_type.is_signed() => InstructionOpcode::SIToFP,
        (true, false) => InstructionOpcode::UIToFP,
        (false, true) if target_type.is_signed() => InstructionOpcode::FPToSI,
        (false, true) => InstructionOpcode::FPToUI,
        (false, false) => match target_type.width().cmp(&source_type.width()) {
            Ordering::Less => InstructionOpcode::FPTrunc,
            Ordering::Equal => return value,
            Ordering::Greater => InstructionOpcode::FPExt,
        },
    };

    builder
        .build_cast(opcode, value, target_llvm_type, "cast")
        .unwrap()
}
