mod cmp;

use std::{collections::HashMap, sync::RwLock};

use arrow_schema::DataType;
pub use cmp::ComparisonKernel;

#[derive(Debug)]
pub enum ArrowKernelError {
    SizeMismatch,
    ArgumentMismatch(String),
    UnsupportedArguments(String),
    UnsupportedScalar(DataType),
}

pub trait Kernel: Sized {
    type Key: std::hash::Hash + std::cmp::Eq;
    type Input<'a>
    where
        Self: 'a;
    type Params;
    type Output;

    fn call<'a>(&self, inp: Self::Input<'a>) -> Result<Self::Output, ArrowKernelError>;

    fn compile<'a>(inp: &Self::Input<'a>, params: Self::Params) -> Result<Self, ArrowKernelError>;
    fn get_key_for_input<'a>(
        i: &Self::Input<'a>,
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
    pub fn get<'a>(
        &self,
        input: K::Input<'a>,
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
