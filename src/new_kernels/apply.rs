use std::ffi::c_void;

use arrow_array::{Array, Datum};
use arrow_schema::DataType;
use inkwell::{
    context::Context, execution_engine::JitFunction, module::Module, AddressSpace,
    OptimizationLevel,
};
use ouroboros::self_referencing;

use crate::{
    declare_blocks,
    new_iter::{datum_to_iter, generate_next_block, IteratorHolder},
    new_kernels::{gen_convert_numeric_vec, optimize_module},
    PrimitiveType,
};

use super::ArrowKernelError;

pub trait ApplyType: Copy {
    fn primitive_type() -> PrimitiveType;
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
}

extern "C" fn trampoline<T: ApplyType, F: FnMut(T)>(user_data: *mut c_void, data: *mut c_void) {
    let nums: [T; 2] = unsafe { *(data as *const [T; 2]) };
    let f = unsafe { &mut (*(user_data as *mut F)) };
    for num in nums {
        f(num);
    }
}

#[self_referencing]
pub struct RustFuncKernel<T> {
    context: Context,
    inp_data_type: DataType,
    pd: std::marker::PhantomData<T>,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void)>,
}

impl<T: ApplyType> RustFuncKernel<T> {
    fn call<F: FnMut(T)>(&self, inp: &dyn Array, f: F) -> Result<(), ArrowKernelError> {
        let mut iter = datum_to_iter(&inp)?;

        let mut f = Box::new(f);
        self.with_func(|func| unsafe {
            func.call(
                iter.get_mut_ptr(),
                trampoline::<T, F> as *mut c_void,
                f.as_mut() as *mut _ as *mut c_void,
            )
        });

        Ok(())
    }

    fn compile(inp: &dyn Array) -> Result<Self, ArrowKernelError> {
        let ih = datum_to_iter(&inp)?;
        RustFuncKernelTryBuilder {
            context: Context::create(),
            inp_data_type: inp.data_type().clone(),
            pd: std::marker::PhantomData::default(),
            func_builder: |ctx| generate_call::<2>(ctx, inp.data_type(), &ih, T::primitive_type()),
        }
        .try_build()
    }
}

fn generate_call<'a, const N: u32>(
    ctx: &'a Context,
    dt: &DataType,
    ih: &IteratorHolder,
    rust_expected_type: PrimitiveType,
) -> Result<
    JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void)>,
    ArrowKernelError,
> {
    let module = ctx.create_module("call");
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let input_prim_type = PrimitiveType::for_arrow_type(dt);
    let input_type = input_prim_type.llvm_type(ctx);
    let input_vec_type = input_prim_type.llvm_vec_type(ctx, N).unwrap();
    let rust_vec_type = rust_expected_type.llvm_vec_type(ctx, N).unwrap();
    let func = module.add_function(
        "call_rust",
        ctx.void_type()
            .fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false),
        None,
    );

    let rust_func_type = ctx
        .void_type()
        .fn_type(&[ptr_type.into(), ptr_type.into()], false);
    let next = generate_next_block::<N>(ctx, &module, "call", dt, ih).unwrap();

    let build = ctx.create_builder();
    declare_blocks!(ctx, func, entry, loop_cond, loop_body, exit);

    build.position_at_end(entry);
    let buf = build.build_alloca(input_vec_type, "buf").unwrap();
    let rbuf = build.build_alloca(rust_vec_type, "rbuf").unwrap();
    let iter_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let func_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
    let ud_ptr = func.get_nth_param(2).unwrap().into_pointer_value();
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
    let val = build
        .build_load(input_vec_type, buf, "val")
        .unwrap()
        .into_vector_value();
    let val = gen_convert_numeric_vec(ctx, &build, val, input_prim_type, rust_expected_type);
    build.build_store(rbuf, val).unwrap();
    build
        .build_indirect_call(
            rust_func_type,
            func_ptr,
            &[ud_ptr.into(), rbuf.into()],
            "call_rust",
        )
        .unwrap();
    build.build_unconditional_branch(loop_cond).unwrap();

    build.position_at_end(exit);
    build.build_return(None).unwrap();

    module.verify().unwrap();
    optimize_module(&module)?;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    Ok(unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void)>(
            func.get_name().to_str().unwrap(),
        )
        .unwrap()
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::Int32Array;
    use arrow_schema::DataType;
    use inkwell::context::Context;

    use crate::{new_iter::datum_to_iter, new_kernels::apply::RustFuncKernel};

    #[test]
    fn test_rust_call() {
        let mut v = Vec::new();
        let f = |x: i64| {
            v.push(x);
        };

        let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let k = RustFuncKernel::compile(&data).unwrap();
        k.call(&data, f).unwrap();

        assert_eq!(v, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
