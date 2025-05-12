mod apply;
mod cast;
mod cmp;
pub mod dsl;
mod ht;
mod llvm_utils;
mod take;
mod writers;
use std::{collections::HashMap, sync::RwLock};

pub use apply::FloatFuncCache;
pub use apply::IntFuncCache;
pub use apply::StrFuncCache;
pub use apply::UIntFuncCache;
use arrow_schema::DataType;
pub use cast::CastToDictKernel;
pub use cast::CastToFlatKernel;
pub use cmp::ComparisonKernel;
use inkwell::execution_engine::ExecutionEngine;
use llvm_utils::str_writer_append_bytes;
pub use take::TakeKernel;

use dsl::DSLError;
use inkwell::{
    builder::Builder,
    context::Context,
    intrinsics::Intrinsic,
    module::Module,
    passes::PassBuilderOptions,
    targets::{CodeModel, RelocMode, Target, TargetMachine},
    values::{FunctionValue, VectorValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};

use crate::{declare_blocks, increment_pointer, pointer_diff, PrimitiveType};

#[derive(Debug)]
pub enum ArrowKernelError {
    SizeMismatch,
    ArgumentMismatch(String),
    UnsupportedArguments(String),
    UnsupportedScalar(DataType),
    LLVMError(String),
    NonVectorizableType(DataType),
    DictionaryFullError(DataType),
    DSLError(DSLError),
}

pub trait Kernel: Sized {
    type Key: std::hash::Hash + std::cmp::Eq;
    type Input<'a>
    where
        Self: 'a;
    type Params;
    type Output;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError>;

    fn compile(inp: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError>;
    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError>;
}

pub struct KernelCache<K: Kernel> {
    map: RwLock<HashMap<K::Key, K>>,
}

impl<K: Kernel> KernelCache<K> {
    pub fn new() -> Self {
        return KernelCache {
            map: RwLock::new(HashMap::new()),
        };
    }
    pub fn get(
        &self,
        input: K::Input<'_>,
        param: K::Params,
    ) -> Result<K::Output, ArrowKernelError> {
        let key = K::get_key_for_input(&input, &param)?;

        {
            let map = self.map.read().unwrap();
            if let Some(kernel) = map.get(&key) {
                return kernel.call(input);
            }
        }

        // There is a chance that multiple threads will see that a kernel is
        // missing, compile it, and then attempt to insert it. This seems
        // preferable to having to hold a write lock during kernel compilation.
        let kernel = K::compile(&input, param)?;
        let result = kernel.call(input);

        {
            let mut map = self.map.write().unwrap();
            map.entry(key).or_insert(kernel);
        }

        result
    }
}

fn link_req_helpers(module: &Module, ee: &ExecutionEngine) -> Result<(), ArrowKernelError> {
    if let Some(func) = module.get_function("str_writer_append_bytes") {
        ee.add_global_mapping(&func, str_writer_append_bytes as usize);
    }

    Ok(())
}

fn optimize_module(module: &Module) -> Result<(), ArrowKernelError> {
    Target::initialize_native(&inkwell::targets::InitializationConfig::default()).unwrap();
    let triple = TargetMachine::get_default_triple();
    let cpu = TargetMachine::get_host_cpu_name().to_string();
    let features = TargetMachine::get_host_cpu_features().to_string();
    let target = Target::from_triple(&triple).unwrap();
    let machine = target
        .create_target_machine(
            &triple,
            &cpu,
            &features,
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::Default,
        )
        .unwrap();

    module
        .run_passes("default<O3>", &machine, PassBuilderOptions::create())
        .map_err(|e| ArrowKernelError::LLVMError(e.to_string()))
}

/// Emit code to convert a vector of numeric values to a different numeric type,
/// (e.g., from i32 to f64).
fn gen_convert_numeric_vec<'ctx>(
    ctx: &'ctx Context,
    builder: &Builder<'ctx>,
    v: VectorValue<'ctx>,
    src: PrimitiveType,
    dst: PrimitiveType,
) -> VectorValue<'ctx> {
    if src == dst {
        return v;
    }

    let dst_llvm = dst.llvm_vec_type(ctx, v.get_type().get_size()).unwrap();

    match (src.is_int(), dst.is_int()) {
        // int to int
        (true, true) => {
            if src.width() > dst.width() {
                builder.build_int_truncate(v, dst_llvm, "trunc").unwrap()
            } else if src.is_signed() {
                builder.build_int_s_extend(v, dst_llvm, "sext").unwrap()
            } else {
                builder.build_int_z_extend(v, dst_llvm, "zext").unwrap()
            }
        }
        // int to float
        (true, false) => {
            if src.is_signed() {
                builder
                    .build_signed_int_to_float(v, dst_llvm, "sitf")
                    .unwrap()
            } else {
                builder
                    .build_signed_int_to_float(v, dst_llvm, "uitf")
                    .unwrap()
            }
        }
        // float to int
        (false, true) => {
            if dst.is_signed() {
                builder
                    .build_float_to_signed_int(v, dst_llvm, "ftsi")
                    .unwrap()
            } else {
                builder
                    .build_float_to_unsigned_int(v, dst_llvm, "ftui")
                    .unwrap()
            }
        }
        // float to float
        (false, false) => {
            if src.width() > dst.width() {
                builder.build_float_trunc(v, dst_llvm, "ftrun").unwrap()
            } else {
                builder.build_float_ext(v, dst_llvm, "fext").unwrap()
            }
        }
    }
}

/// Adds a function to the current context that takes in a start and end string
/// pointer, along with a base buffer pointer, and returns a view. The buffer
/// index is always zero.
pub fn add_ptrx2_to_view<'a>(ctx: &'a Context, llvm_mod: &Module<'a>) -> FunctionValue<'a> {
    let i1_type = ctx.bool_type();
    let i32_type = ctx.i32_type();
    let i64_type = ctx.i64_type();
    let i128_type = ctx.i128_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let str_type = PrimitiveType::P64x2.llvm_type(ctx);

    let memcpy = Intrinsic::find("llvm.memcpy").unwrap();
    let memcpy_f = memcpy
        .get_declaration(
            &llvm_mod,
            &[ptr_type.into(), ptr_type.into(), i64_type.into()],
        )
        .unwrap();

    let func_type = i128_type.fn_type(&[str_type.into(), ptr_type.into()], false);
    let func = llvm_mod.add_function("ptrx2_to_view", func_type, None);

    let ptrs = func.get_nth_param(0).unwrap().into_struct_value();
    let base_ptr = func.get_nth_param(1).unwrap().into_pointer_value();

    declare_blocks!(ctx, func, entry, fits, no_fit, exit);
    let builder = ctx.create_builder();
    builder.position_at_end(entry);
    let ptr1 = builder
        .build_extract_value(ptrs, 0, "ptr1")
        .unwrap()
        .into_pointer_value();
    let ptr2 = builder
        .build_extract_value(ptrs, 1, "ptr2")
        .unwrap()
        .into_pointer_value();
    let to_return_ptr = builder.build_alloca(i128_type, "to_return_ptr").unwrap();
    let len_u64 = pointer_diff!(ctx, &builder, ptr1, ptr2);
    let len = builder
        .build_int_truncate(len_u64, i32_type, "len_u32")
        .unwrap();
    let len_128 = builder
        .build_int_z_extend(len, i128_type, "len_128")
        .unwrap();
    builder.build_store(to_return_ptr, len_128).unwrap();
    let is_short = builder
        .build_int_compare(IntPredicate::ULE, len, i32_type.const_int(12, false), "cmp")
        .unwrap();
    builder
        .build_conditional_branch(is_short, fits, no_fit)
        .unwrap();

    builder.position_at_end(fits);
    builder
        .build_call(
            memcpy_f,
            &[
                increment_pointer!(
                    ctx,
                    &builder,
                    to_return_ptr,
                    4,
                    i64_type.const_int(1, false)
                )
                .into(),
                ptr1.into(),
                len_u64.into(),
                i1_type.const_zero().into(),
            ],
            "memcpy",
        )
        .unwrap();
    builder.build_unconditional_branch(exit).unwrap();

    builder.position_at_end(no_fit);
    let prefix = builder.build_load(i32_type, ptr1, "prefix").unwrap();
    builder
        .build_store(
            increment_pointer!(
                ctx,
                &builder,
                to_return_ptr,
                4,
                i64_type.const_int(1, false)
            ),
            prefix,
        )
        .unwrap();
    let offset = pointer_diff!(ctx, &builder, base_ptr, ptr1);
    let offset = builder
        .build_int_truncate(offset, i32_type, "offset")
        .unwrap();
    builder
        .build_store(
            increment_pointer!(
                ctx,
                &builder,
                to_return_ptr,
                4,
                i64_type.const_int(3, false)
            ),
            offset,
        )
        .unwrap();
    builder.build_unconditional_branch(exit).unwrap();

    builder.position_at_end(exit);
    let result = builder
        .build_load(i128_type, to_return_ptr, "result")
        .unwrap();
    builder.build_return(Some(&result)).unwrap();

    func
}

#[cfg(test)]
mod tests {
    use arrow_array::{BooleanArray, Int32Array, Scalar};

    use crate::Predicate;

    use super::{ComparisonKernel, KernelCache};

    #[test]
    fn test_kernel_cache_cmp() {
        let cache = KernelCache::<ComparisonKernel>::new();
        let d1 = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let d2 = Scalar::new(Int32Array::from(vec![3]));

        let r = cache.get((&d1, &d2), Predicate::Lt).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true, true, false, false, false]));

        let r = cache.get((&d1, &d2), Predicate::Lt).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true, true, false, false, false]));
    }
}
