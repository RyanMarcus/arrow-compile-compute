use std::{collections::HashMap, ffi::c_void, sync::RwLock, u64};

use arrow_array::Array;
use arrow_schema::DataType;
use inkwell::{
    context::Context, execution_engine::JitFunction, AddressSpace, IntPredicate, OptimizationLevel,
};
use ouroboros::self_referencing;

use crate::{
    declare_blocks, increment_pointer,
    new_iter::{datum_to_iter, generate_next, IteratorHolder},
    new_kernels::{gen_convert_numeric_vec, optimize_module},
    PrimitiveType,
};

use super::ArrowKernelError;

const BLOCK_SIZE: usize = 64;

pub trait ApplyType: Copy {
    fn primitive_type() -> PrimitiveType;
    unsafe fn call<F: FnMut(Self)>(f: &mut F, data: *mut c_void, len: usize) {
        let nums = unsafe { std::slice::from_raw_parts(data as *const Self, len) };
        for num in nums {
            f(*num);
        }
    }
}
impl ApplyType for i64 {
    fn primitive_type() -> PrimitiveType {
        PrimitiveType::I64
    }
}
impl ApplyType for f64 {
    fn primitive_type() -> PrimitiveType {
        PrimitiveType::F64
    }
}
impl ApplyType for u64 {
    fn primitive_type() -> PrimitiveType {
        PrimitiveType::U64
    }
}
impl ApplyType for &[u8] {
    fn primitive_type() -> PrimitiveType {
        PrimitiveType::P64x2
    }

    unsafe fn call<F: FnMut(Self)>(f: &mut F, data: *mut c_void, len: usize) {
        let nums = unsafe { std::slice::from_raw_parts(data as *const u128, len) };
        for num in nums {
            let start = (num & (u64::MAX as u128)) as u64 as *const u8;
            let end = (num >> 64) as u64 as *const u8;
            let len = end.offset_from(start);
            debug_assert!(len >= 0);
            let s = std::slice::from_raw_parts(start, len as usize);
            f(s);
        }
    }
}

extern "C" fn trampoline<T: ApplyType, F: FnMut(T)>(
    user_data: *mut c_void,
    data: *mut c_void,
    len: u64,
) {
    unsafe {
        let f = &mut (*(user_data as *mut F));
        T::call(f, data, len as usize);
    }
}

#[self_referencing]
pub struct RustFuncKernel<T> {
    context: Context,
    inp_data_type: DataType,
    pd: std::marker::PhantomData<T>,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<
        'this,
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void),
    >,
}

pub type IntFuncCache = FuncCache<i64>;
pub type UIntFuncCache = FuncCache<u64>;
pub type FloatFuncCache = FuncCache<f64>;
pub type StrFuncCache<'a> = FuncCache<&'a [u8]>;

#[derive(Default)]
pub struct FuncCache<T: ApplyType> {
    map: RwLock<HashMap<DataType, RustFuncKernel<T>>>,
}

unsafe impl<T: ApplyType> Send for FuncCache<T> {}
unsafe impl<T: ApplyType> Sync for FuncCache<T> {}

impl<T: ApplyType> FuncCache<T> {
    pub fn call<F: FnMut(T)>(&self, arr: &dyn Array, f: F) -> Result<(), ArrowKernelError> {
        let key = arr.data_type().clone();
        {
            let guard = self.map.read().unwrap();
            if let Some(k) = guard.get(&key) {
                return k.call(arr, f);
            }
        }
        // otherwise compile and insert
        let k = RustFuncKernel::compile(arr)?;
        let r = k.call(arr, f);

        {
            let mut map = self.map.write().unwrap();
            map.entry(key).or_insert(k);
        }

        r
    }
}

impl<T: ApplyType> RustFuncKernel<T> {
    fn call<F: FnMut(T)>(&self, inp: &dyn Array, f: F) -> Result<(), ArrowKernelError> {
        let mut iter = datum_to_iter(&inp)?;

        let mut f = Box::new(f);
        self.with_func(|func| unsafe {
            let ptype = PrimitiveType::for_arrow_type(inp.data_type());
            let mut buf = vec![0; ptype.width() * BLOCK_SIZE];

            func.call(
                iter.get_mut_ptr(),
                buf.as_mut_ptr() as *mut c_void,
                trampoline::<T, F> as *mut c_void,
                f.as_mut() as *mut _ as *mut c_void,
            );

            std::mem::drop(buf);
        });

        Ok(())
    }

    fn compile(inp: &dyn Array) -> Result<Self, ArrowKernelError> {
        let ih = datum_to_iter(&inp)?;
        RustFuncKernelTryBuilder {
            context: Context::create(),
            inp_data_type: inp.data_type().clone(),
            pd: std::marker::PhantomData::default(),
            func_builder: |ctx| {
                generate_call(ctx, inp.data_type(), &ih, T::primitive_type(), BLOCK_SIZE)
            },
        }
        .try_build()
    }
}

fn generate_call<'a>(
    ctx: &'a Context,
    dt: &DataType,
    ih: &IteratorHolder,
    rust_expected_type: PrimitiveType,
    chunk_size: usize,
) -> Result<
    JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void)>,
    ArrowKernelError,
> {
    let module = ctx.create_module("call");
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let i64_type = ctx.i64_type();
    let input_prim_type = PrimitiveType::for_arrow_type(dt);
    let rust_expected_llvm_type = rust_expected_type.llvm_type(ctx);
    let input_type = input_prim_type.llvm_type(ctx);
    let func = module.add_function(
        "call_rust",
        ctx.void_type().fn_type(
            &[
                ptr_type.into(),
                ptr_type.into(),
                ptr_type.into(),
                ptr_type.into(),
            ],
            false,
        ),
        None,
    );

    let rust_func_type = ctx
        .void_type()
        .fn_type(&[ptr_type.into(), ptr_type.into(), i64_type.into()], false);
    let next = generate_next(ctx, &module, "call", dt, ih).unwrap();

    let build = ctx.create_builder();
    declare_blocks!(ctx, func, entry, loop_cond, loop_body, flush_buffer, exit);

    build.position_at_end(entry);
    let iter_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let rust_buf_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
    let func_ptr = func.get_nth_param(2).unwrap().into_pointer_value();
    let ud_ptr = func.get_nth_param(3).unwrap().into_pointer_value();
    let buf = build.build_alloca(input_type, "buf").unwrap();
    let offset_ptr = build.build_alloca(i64_type, "offset_ptr").unwrap();
    build
        .build_store(offset_ptr, i64_type.const_zero())
        .unwrap();
    build.build_unconditional_branch(loop_cond).unwrap();

    build.position_at_end(loop_cond);
    let res = build
        .build_call(next, &[iter_ptr.into(), buf.into()], "had_next")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_conditional_branch(res, loop_body, exit)
        .unwrap();

    build.position_at_end(loop_body);
    let val = build.build_load(input_type, buf, "val").unwrap();

    let val = match rust_expected_type {
        PrimitiveType::P64x2 => val,
        _ => {
            let in_v_type = input_prim_type.llvm_vec_type(ctx, 1).unwrap();
            let val = build
                .build_bit_cast(val, in_v_type, "singleton_vec")
                .unwrap()
                .into_vector_value();
            let r = gen_convert_numeric_vec(ctx, &build, val, input_prim_type, rust_expected_type);
            build
                .build_bit_cast(r, rust_expected_llvm_type, "vec_to_single")
                .unwrap()
        }
    };

    let curr_offset = build
        .build_load(i64_type, offset_ptr, "curr_offset")
        .unwrap()
        .into_int_value();
    build
        .build_store(
            increment_pointer!(
                ctx,
                build,
                rust_buf_ptr,
                rust_expected_type.width(),
                curr_offset
            ),
            val,
        )
        .unwrap();
    let next_offset = build
        .build_int_add(curr_offset, i64_type.const_int(1, false), "next_offset")
        .unwrap();
    build.build_store(offset_ptr, next_offset).unwrap();

    let cmp = build
        .build_int_compare(
            IntPredicate::UGE,
            next_offset,
            i64_type.const_int(chunk_size as u64, false),
            "buf_full",
        )
        .unwrap();

    build
        .build_conditional_branch(cmp, flush_buffer, loop_cond)
        .unwrap();

    build.position_at_end(flush_buffer);
    let curr_buf_len = build
        .build_load(i64_type, offset_ptr, "curr_buf_len")
        .unwrap()
        .into_int_value();
    build
        .build_indirect_call(
            rust_func_type,
            func_ptr,
            &[ud_ptr.into(), rust_buf_ptr.into(), curr_buf_len.into()],
            "call_rust",
        )
        .unwrap();
    build
        .build_store(offset_ptr, i64_type.const_zero())
        .unwrap();
    build.build_unconditional_branch(loop_cond).unwrap();

    build.position_at_end(exit);
    let curr_buf_len = build
        .build_load(i64_type, offset_ptr, "curr_buf_len")
        .unwrap()
        .into_int_value();
    build
        .build_indirect_call(
            rust_func_type,
            func_ptr,
            &[ud_ptr.into(), rust_buf_ptr.into(), curr_buf_len.into()],
            "call_rust",
        )
        .unwrap();
    build.build_return(None).unwrap();

    module.verify().unwrap();
    optimize_module(&module)?;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    Ok(unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void)>(
            func.get_name().to_str().unwrap(),
        )
        .unwrap()
    })
}

#[cfg(test)]
mod tests {

    use arrow_array::{
        Float32Array, Float64Array, Int32Array, Int64Array, StringArray, UInt32Array, UInt64Array,
    };

    use itertools::Itertools;

    use crate::new_kernels::apply::RustFuncKernel;

    #[test]
    fn test_apply_i32() {
        let mut v = Vec::new();
        let f = |x: i64| {
            v.push(x);
        };

        let data = Int32Array::from((-2000..2000).collect_vec());
        let k = RustFuncKernel::compile(&data).unwrap();
        k.call(&data, f).unwrap();
        assert_eq!(v, (-2000..2000).collect_vec());
    }

    #[test]
    fn test_apply_i64() {
        let mut v = Vec::new();
        let f = |x: i64| {
            v.push(x);
        };

        let data = Int64Array::from((-2000..2000).collect_vec());
        let k = RustFuncKernel::compile(&data).unwrap();
        k.call(&data, f).unwrap();
        assert_eq!(v, (-2000..2000).collect_vec());
    }

    #[test]
    fn test_apply_u32() {
        let mut v = Vec::new();
        let f = |x: u64| {
            v.push(x);
        };

        let data = UInt32Array::from((0..2000).collect_vec());
        let k = RustFuncKernel::compile(&data).unwrap();
        k.call(&data, f).unwrap();
        assert_eq!(v, (0..2000).collect_vec());
    }

    #[test]
    fn test_apply_u64() {
        let mut v = Vec::new();
        let f = |x: u64| {
            v.push(x);
        };

        let data = UInt64Array::from((0..2000).collect_vec());
        let k = RustFuncKernel::compile(&data).unwrap();
        k.call(&data, f).unwrap();
        assert_eq!(v, (0..2000).collect_vec());
    }

    #[test]
    fn test_apply_f32() {
        let mut v = Vec::new();
        let f = |x: f64| {
            v.push(x);
        };

        let data = Float32Array::from((0..2000).map(|x| x as f32).collect_vec());
        let k = RustFuncKernel::compile(&data).unwrap();
        k.call(&data, f).unwrap();
        assert_eq!(v, (0..2000).map(|x| x as f64).collect_vec());
    }

    #[test]
    fn test_apply_f64() {
        let mut v = Vec::new();
        let f = |x: f64| {
            v.push(x);
        };

        let data = Float64Array::from((0..2000).map(|x| x as f64).collect_vec());
        let k = RustFuncKernel::compile(&data).unwrap();
        k.call(&data, f).unwrap();
        assert_eq!(v, (0..2000).map(|x| x as f64).collect_vec());
    }

    #[test]
    fn test_apply_str() {
        let mut v = Vec::new();
        let f = |x: &[u8]| {
            v.push(std::str::from_utf8(x).unwrap().to_string());
        };

        let data = StringArray::from((-2000..2000).map(|i| format!("{}", i)).collect_vec());
        let k = RustFuncKernel::compile(&data).unwrap();
        k.call(&data, f).unwrap();
        assert_eq!(v, (-2000..2000).map(|i| format!("{}", i)).collect_vec());
    }
}
