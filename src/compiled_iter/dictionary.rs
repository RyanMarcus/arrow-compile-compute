use std::{ffi::c_void, sync::Arc};

use arrow_array::{types::ArrowDictionaryKeyType, Array, DictionaryArray};
use inkwell::{builder::Builder, context::Context, values::PointerValue, AddressSpace};
use repr_offset::ReprOffset;

use crate::increment_pointer;

use super::{array_to_iter, IteratorHolder};

/// An iterator for dictionary data. Contains pointers to the *iterators* for
/// the underlying keys and values. To access the element at position `i`, you
/// want to compute `value[key[i]]`. The `key_iter` is used for tracking current
/// position and length.
#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct DictionaryIterator {
    key_iter: *const c_void,
    val_iter: *const c_void,
    array_ref: Arc<dyn Array>,
}

impl<K: ArrowDictionaryKeyType> From<&DictionaryArray<K>> for IteratorHolder {
    fn from(arr: &DictionaryArray<K>) -> Self {
        let arr = Arc::new(arr.clone());
        let keys = Box::new(array_to_iter(arr.keys()));
        let values = Box::new(array_to_iter(arr.values()));
        let iter = DictionaryIterator {
            key_iter: keys.get_ptr(),
            val_iter: values.get_ptr(),
            array_ref: arr,
        };
        IteratorHolder::Dictionary {
            arr: Box::new(iter),
            keys,
            values,
        }
    }
}

impl DictionaryIterator {
    pub fn llvm_key_iter_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let ptr_ptr = increment_pointer!(ctx, builder, ptr, DictionaryIterator::OFFSET_KEY_ITER);
        builder
            .build_load(ctx.ptr_type(AddressSpace::default()), ptr_ptr, "key_iter")
            .unwrap()
            .into_pointer_value()
    }

    pub fn llvm_val_iter_ptr<'a>(
        &self,
        ctx: &'a Context,
        builder: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let ptr_ptr = increment_pointer!(ctx, builder, ptr, DictionaryIterator::OFFSET_VAL_ITER);
        builder
            .build_load(ctx.ptr_type(AddressSpace::default()), ptr_ptr, "val_iter")
            .unwrap()
            .into_pointer_value()
    }
}

#[cfg(test)]
mod tests {
    use std::{ffi::c_void, sync::Arc};

    use arrow_array::{Array, DictionaryArray, Int32Array, Int8Array};
    use arrow_schema::DataType;
    use inkwell::{context::Context, OptimizationLevel};

    use crate::compiled_iter::{
        array_to_iter, generate_next, generate_next_block, generate_random_access,
    };

    #[test]
    fn test_dict_iter_block1() {
        let data = Int32Array::from(vec![10, 20, 30]);
        let data = arrow_cast::cast(
            &data,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        )
        .unwrap();

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func =
            generate_next_block::<8>(&ctx, &module, "dict_iter_block1", data.data_type(), &iter)
                .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut out_buf = [0_i32; 8];

        let result = unsafe { next_block_func.call(iter.get_mut_ptr(), out_buf.as_mut_ptr()) };

        assert!(
            !result,
            "expected false since the dict size is less than the block size"
        );
    }

    #[test]
    fn test_dict_iter_block2() {
        let data = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let data = arrow_cast::cast(
            &data,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        )
        .unwrap();

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func =
            generate_next_block::<2>(&ctx, &module, "dict_iter_block1", data.data_type(), &iter)
                .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut out_buf = [0_i32; 2];

        unsafe {
            let ret1 = next_block_func.call(iter.get_mut_ptr(), out_buf.as_mut_ptr());
            assert!(ret1, "First call should return true");
            assert_eq!(out_buf, [10, 20], "First block should have [10, 20]");

            let ret2 = next_block_func.call(iter.get_mut_ptr(), out_buf.as_mut_ptr());
            assert!(ret2, "Second call should return true");
            assert_eq!(out_buf, [30, 40], "Second block should have [30, 40]");

            let ret3 = next_block_func.call(iter.get_mut_ptr(), out_buf.as_mut_ptr());
            assert!(!ret3, "Third call should return false");
        };
    }

    #[test]
    fn test_dict_iter_block2_slice() {
        let data = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let data = arrow_cast::cast(
            &data,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        )
        .unwrap();
        let data = data.slice(2, 3);

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func =
            generate_next_block::<2>(&ctx, &module, "dict_iter_block1", data.data_type(), &iter)
                .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut i32) -> bool>(fname)
                .unwrap()
        };

        let mut out_buf = [0_i32; 2];

        unsafe {
            let ret2 = next_block_func.call(iter.get_mut_ptr(), out_buf.as_mut_ptr());
            assert!(ret2, "Second call should return true");
            assert_eq!(out_buf, [30, 40], "Second block should have [30, 40]");

            let ret3 = next_block_func.call(iter.get_mut_ptr(), out_buf.as_mut_ptr());
            assert!(!ret3, "Third call should return false");
        };
    }

    #[test]
    fn test_dict_random_access() {
        let data = Int32Array::from(vec![0, 0, 10, 10, 20, 20, 30, 30, 40, 40]);
        let data = arrow_cast::cast(
            &data,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        )
        .unwrap();

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_random_access(&ctx, &module, "iter_dict_test", data.data_type(), &iter)
            .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i32>(fname)
                .unwrap()
        };

        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), 0), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 1), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 2), 10);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 3), 10);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 4), 20);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 5), 20);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 6), 30);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 7), 30);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 8), 40);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 9), 40);
        };
    }

    #[test]
    fn test_dict_random_access_slice() {
        let data = Int32Array::from(vec![0, 0, 10, 10, 20, 20, 30, 30, 40, 40]);
        let data = arrow_cast::cast(
            &data,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        )
        .unwrap();
        let data = data.slice(2, 6);

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_random_access(&ctx, &module, "iter_dict_test", data.data_type(), &iter)
            .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i32>(fname)
                .unwrap()
        };

        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), 0), 10);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 1), 10);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 2), 20);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 3), 20);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 4), 30);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 5), 30);
        };
    }

    #[test]
    fn test_dict_recur_random_access() {
        let keys = Int8Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let values = Int32Array::from(vec![0, 10, 20, 30]);
        let da1 = DictionaryArray::new(keys, Arc::new(values));

        let parent_keys = Int8Array::from(vec![0, 2, 4, 6]);
        let da2 = DictionaryArray::new(parent_keys, Arc::new(da1));

        let mut iter = array_to_iter(&da2);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_random_access(&ctx, &module, "iter_dict_test", da2.data_type(), &iter)
            .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i32>(fname)
                .unwrap()
        };

        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), 0), 0);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 1), 10);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 2), 20);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 3), 30);
        };
    }

    #[test]
    fn test_dict_recur_random_access_slice() {
        let keys = Int8Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let values = Int32Array::from(vec![0, 10, 20, 30]);
        let da1 = DictionaryArray::new(keys, Arc::new(values));

        let parent_keys = Int8Array::from(vec![0, 2, 4, 6]);
        let da2 = DictionaryArray::new(parent_keys, Arc::new(da1));
        let da2 = da2.slice(2, 2);

        let mut iter = array_to_iter(&da2);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_random_access(&ctx, &module, "iter_dict_test", da2.data_type(), &iter)
            .unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i32>(fname)
                .unwrap()
        };

        unsafe {
            assert_eq!(next_func.call(iter.get_mut_ptr(), 0), 20);
            assert_eq!(next_func.call(iter.get_mut_ptr(), 1), 30);
        };
    }

    #[test]
    fn test_dict_iter_nonblock() {
        let data = Int32Array::from(vec![0, 0, 10, 10, 20, 20]);
        let data = arrow_cast::cast(
            &data,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        )
        .unwrap();

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_next(&ctx, &module, "iter_dict_test", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool>(fname)
                .unwrap()
        };

        unsafe {
            let mut res: i32 = 0;
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 0);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 0);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 10);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 10);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 20);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 20);
            assert!(!next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
        };
    }

    #[test]
    fn test_dict_iter_nonblock_slice() {
        let data = Int32Array::from(vec![0, 0, 10, 10, 20, 20]);
        let data = arrow_cast::cast(
            &data,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        )
        .unwrap();
        let data = data.slice(2, 4);

        let mut iter = array_to_iter(&data);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_next(&ctx, &module, "iter_dict_test", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool>(fname)
                .unwrap()
        };

        unsafe {
            let mut res: i32 = 0;
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 10);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 10);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 20);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 20);
            assert!(!next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
        };
    }

    #[test]
    fn test_dict_recur_iter_nonblock() {
        let keys = Int8Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let values = Int32Array::from(vec![0, 10, 20, 30]);
        let da1 = DictionaryArray::new(keys, Arc::new(values));

        let parent_keys = Int8Array::from(vec![0, 2, 4, 6]);
        let da2 = DictionaryArray::new(parent_keys, Arc::new(da1));

        let mut iter = array_to_iter(&da2);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_next(&ctx, &module, "iter_dict_test", da2.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool>(fname)
                .unwrap()
        };

        unsafe {
            let mut res: i32 = 0;
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 0);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 10);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 20);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 30);
            assert!(!next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
        };
    }

    #[test]
    fn test_dict_recur_iter_nonblock_slice() {
        let keys = Int8Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let values = Int32Array::from(vec![0, 10, 20, 30]);
        let da1 = DictionaryArray::new(keys, Arc::new(values));

        let parent_keys = Int8Array::from(vec![0, 2, 4, 6]);
        let da2 = DictionaryArray::new(parent_keys, Arc::new(da1));
        let da2 = da2.slice(2, 2);

        let mut iter = array_to_iter(&da2);

        let ctx = Context::create();
        let module = ctx.create_module("test_iter");
        let func = generate_next(&ctx, &module, "iter_dict_test", da2.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool>(fname)
                .unwrap()
        };

        unsafe {
            let mut res: i32 = 0;
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 20);
            assert!(next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
            assert_eq!(res, 30);
            assert!(!next_func.call(iter.get_mut_ptr(), &mut res as *mut i32 as *mut c_void));
        };
    }
}
