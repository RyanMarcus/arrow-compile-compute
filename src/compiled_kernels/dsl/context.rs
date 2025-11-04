use std::collections::HashMap;

use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    types::BasicTypeEnum,
    values::{FunctionValue, PointerValue},
};

pub(super) const VEC_SIZE: u32 = 64;

pub(super) struct CompilationContext<'ctx, 'b> {
    pub llvm_ctx: &'ctx Context,
    pub llvm_mod: &'b Module<'ctx>,
    pub builder: &'b Builder<'ctx>,
    pub bufs: &'b HashMap<usize, PointerValue<'ctx>>,
    pub accessors: &'b HashMap<usize, FunctionValue<'ctx>>,
    pub vec_bufs: &'b HashMap<usize, PointerValue<'ctx>>,
    pub blocked_access_funcs: &'b HashMap<(usize, u32), FunctionValue<'ctx>>,
    pub iter_ptrs: &'b [PointerValue<'ctx>],
    pub iter_llvm_types: &'b HashMap<usize, BasicTypeEnum<'ctx>>,
}
