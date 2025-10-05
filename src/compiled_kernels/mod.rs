mod aggregate;
mod arith;
mod cast;
mod cmp;
mod concat;
pub mod dsl;
mod filter;
mod ht;
mod llvm_utils;
mod partition;
mod rust_iter;
mod sort;
mod take;
mod writers;
use std::sync::Arc;
use std::{collections::HashMap, sync::RwLock};

pub use aggregate::{CountAggregator, MaxAggregator, MinAggregator, SumAggregator};
pub use arith::BinOp;
pub use arith::BinOpKernel;
use arrow_array::make_array;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
pub use cast::CastKernel;
pub use cmp::ComparisonKernel;
pub use concat::concat_all;
pub use filter::FilterKernel;
pub use ht::{HashFunction, HashKernel};
use inkwell::execution_engine::ExecutionEngine;
use llvm_utils::str_writer_append_bytes;
pub use partition::PartitionKernel;
pub use rust_iter::{ArrowIter, ArrowNullableIter, IterFuncHolder};
pub use sort::{SortKernel, SortOptions};
pub use take::TakeKernel;

use dsl::DSLError;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    passes::PassBuilderOptions,
    targets::{CodeModel, RelocMode, Target, TargetMachine},
    values::VectorValue,
    OptimizationLevel,
};
use thiserror::Error;

use crate::compiled_kernels::llvm_utils::save_to_string_saver;
use crate::compiled_kernels::llvm_utils::str_view_writer_append_bytes;
use crate::llvm_debug::debug_i64;
use crate::llvm_debug::debug_ptr;
use crate::PrimitiveType;

#[derive(Debug, Error)]
pub enum ArrowKernelError {
    #[error("input sizes did not match")]
    SizeMismatch,

    #[error("argument mismatch: {0}")]
    ArgumentMismatch(String),

    #[error("unsupported argument: {0}")]
    UnsupportedArguments(String),

    #[error("unsupported scalar type: {0}")]
    UnsupportedScalar(DataType),

    #[error("underlying llvm error: {0}")]
    LLVMError(String),

    #[error("Datatype {0} cannot be vectorized")]
    NonVectorizableType(DataType),

    #[error("Dictionary of type {0} is full")]
    DictionaryFullError(DataType),

    #[error("Type mismatch: expected {0:?}, got {1:?}")]
    TypeMismatch(PrimitiveType, PrimitiveType),

    #[error("Atomic aggregation not supported")]
    AtomicAggNotSupported,

    #[error("dsl error")]
    DSLError(#[from] DSLError),
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
        KernelCache {
            map: RwLock::new(HashMap::new()),
        }
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

    if let Some(func) = module.get_function("str_view_writer_append_bytes") {
        ee.add_global_mapping(&func, str_view_writer_append_bytes as usize);
    }

    if let Some(func) = module.get_function("save_to_string_saver") {
        ee.add_global_mapping(&func, save_to_string_saver as usize);
    }

    if let Some(func) = module.get_function("debug_i64") {
        println!("linking debug_i64");
        ee.add_global_mapping(&func, debug_i64 as usize);
    }

    if let Some(func) = module.get_function("debug_ptr") {
        ee.add_global_mapping(&func, debug_ptr as usize);
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

pub fn replace_nulls(arr: Arc<dyn Array>, nulls: Option<NullBuffer>) -> ArrayRef {
    let data = arr.into_data().into_builder().nulls(nulls);
    make_array(unsafe { data.build_unchecked() })
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
