mod cmp;

use std::{collections::HashMap, sync::RwLock};

use arrow_schema::DataType;
pub use cmp::ComparisonKernel;
use inkwell::{
    module::Module,
    passes::PassBuilderOptions,
    targets::{CodeModel, RelocMode, Target, TargetMachine},
    OptimizationLevel,
};

#[derive(Debug)]
pub enum ArrowKernelError {
    SizeMismatch,
    ArgumentMismatch(String),
    UnsupportedArguments(String),
    UnsupportedScalar(DataType),
    LLVMError(String),
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

        let mut map = self.map.write().unwrap();
        if let Some(kernel) = map.get(&key) {
            return kernel.call(input);
        }

        let kernel = K::compile(&input, param)?;
        let result = kernel.call(input);
        map.insert(key, kernel);
        result
    }
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
