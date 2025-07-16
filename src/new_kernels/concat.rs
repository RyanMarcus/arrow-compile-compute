use std::{ffi::c_void, sync::Arc};

use arrow_array::{Array, ArrayRef};
use arrow_buffer::{BooleanBufferBuilder, NullBuffer};
use arrow_schema::DataType;
use inkwell::{context::Context, execution_engine::JitFunction, AddressSpace, OptimizationLevel};
use itertools::Itertools;
use ouroboros::self_referencing;

use crate::{
    declare_blocks,
    dsl::KernelParameters,
    new_iter::{datum_to_iter, generate_next, IteratorHolder},
    new_kernels::{
        cast::coalesce_type,
        link_req_helpers, optimize_module,
        writers::{ArrayWriter, PrimitiveArrayWriter, StringViewWriter, WriterAllocation},
    },
    ArrowKernelError, Kernel, PrimitiveType,
};

#[self_referencing]
pub struct ConcatKernel {
    context: Context,
    types: Vec<DataType>,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut c_void)>,
}

unsafe impl Sync for ConcatKernel {}
unsafe impl Send for ConcatKernel {}

impl Kernel for ConcatKernel {
    type Key = Vec<DataType>;

    type Input<'a> = &'a [&'a dyn Array];

    type Params = ();

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let total_size: usize = inp.iter().map(|x| x.len()).sum();

        let pt = PrimitiveType::for_arrow_type(inp[0].data_type());
        let mut iters: Vec<IteratorHolder> = inp
            .iter()
            .map(|x| datum_to_iter(x))
            .collect::<Result<Vec<_>, ArrowKernelError>>()?;
        let mut kp =
            KernelParameters::new(iters.iter_mut().map(|ih| ih.get_mut_ptr()).collect_vec());
        let nulls = {
            if inp.iter().any(|x| x.is_nullable()) {
                let mut bb = BooleanBufferBuilder::new(total_size);
                for x in inp.iter() {
                    if let Some(nulls) = x.logical_nulls() {
                        bb.append_buffer(nulls.inner());
                    } else {
                        bb.append_n(x.len(), true);
                    }
                }
                Some(NullBuffer::new(bb.finish()))
            } else {
                None
            }
        };

        match pt {
            PrimitiveType::P64x2 => {
                let mut alloc = StringViewWriter::allocate(total_size, pt);
                unsafe {
                    self.borrow_func().call(kp.get_mut_ptr(), alloc.get_ptr());
                }
                let view = Arc::new(alloc.to_array(total_size, nulls));
                coalesce_type(
                    view,
                    &match inp[0].data_type() {
                        DataType::Utf8 => DataType::Utf8View,
                        DataType::LargeUtf8 => DataType::Utf8View,
                        DataType::Binary => DataType::BinaryView,
                        DataType::LargeBinary => DataType::BinaryView,
                        _ => inp[0].data_type().clone(),
                    },
                )
            }
            _ => {
                let mut alloc = PrimitiveArrayWriter::allocate(total_size, pt);
                unsafe {
                    self.borrow_func().call(kp.get_mut_ptr(), alloc.get_ptr());
                }
                let view = Arc::new(alloc.to_array(total_size, nulls));
                coalesce_type(view, inp[0].data_type())
            }
        }
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        if inp.len() < 2 {
            return Err(ArrowKernelError::ArgumentMismatch(
                "concat on empty sequence of arrays".to_string(),
            ));
        }

        let types = inp.iter().map(|x| x.data_type().clone()).collect();
        ConcatKernelTryBuilder {
            context: Context::create(),
            types,
            func_builder: |ctx| match PrimitiveType::for_arrow_type(inp[0].data_type()) {
                PrimitiveType::P64x2 => build_concat::<StringViewWriter>(ctx, inp),
                _ => build_concat::<PrimitiveArrayWriter>(ctx, inp),
            },
        }
        .try_build()
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        if i.is_empty() {
            return Err(ArrowKernelError::ArgumentMismatch(
                "concat on empty sequence of arrays".to_string(),
            ));
        }
        Ok(i.iter().map(|x| x.data_type().clone()).collect())
    }
}

fn build_concat<'a, W: ArrayWriter<'a>>(
    ctx: &'a Context,
    data: &[&dyn Array],
) -> Result<JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void)>, ArrowKernelError> {
    let llvm_mod = ctx.create_module("concat");

    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let func = llvm_mod.add_function(
        "concat",
        ctx.void_type()
            .fn_type(&[ptr_type.into(), ptr_type.into()], false),
        None,
    );

    // generate all the next functions we need
    let mut next_funcs = Vec::new();
    for (idx, arr) in data.iter().enumerate() {
        next_funcs.push(
            generate_next(
                ctx,
                &llvm_mod,
                &format!("next{}", idx),
                arr.data_type(),
                &datum_to_iter(arr)?,
            )
            .unwrap(),
        );
    }

    // ensure all types are the same
    let pt = PrimitiveType::for_arrow_type(data[0].data_type());
    for arr in data.iter().skip(1) {
        let lpt = PrimitiveType::for_arrow_type(arr.data_type());
        if lpt != pt {
            return Err(ArrowKernelError::TypeMismatch(pt, lpt));
        }
    }

    let b = ctx.create_builder();
    declare_blocks!(ctx, func, entry, exit);
    let kp_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let alloc_ptr = func.get_nth_param(1).unwrap().into_pointer_value();

    b.position_at_end(entry);
    let kernel_ptrs = (0..data.len())
        .map(|idx| KernelParameters::llvm_get(ctx, &b, kp_ptr, idx))
        .collect_vec();
    let writer = W::llvm_init(ctx, &llvm_mod, &b, pt, alloc_ptr);
    let blocks = (0..data.len())
        .map(|_| {
            declare_blocks!(ctx, func, loop_cond, loop_body);
            (loop_cond, loop_body)
        })
        .collect_vec();
    let llvm_type = pt.llvm_type(ctx);
    let buf_ptr = b.build_alloca(llvm_type, "buf").unwrap();
    b.build_unconditional_branch(blocks[0].0).unwrap();

    for idx in 0..data.len() {
        let (loop_cond, loop_body) = &blocks[idx];
        let next_block = if idx == data.len() - 1 {
            exit
        } else {
            blocks[idx + 1].0
        };
        let iter_ptr = kernel_ptrs[idx];
        let next_func = next_funcs[idx];

        b.position_at_end(*loop_cond);
        let had_next = b
            .build_call(next_func, &[iter_ptr.into(), buf_ptr.into()], "next")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        b.build_conditional_branch(had_next, *loop_body, next_block)
            .unwrap();

        b.position_at_end(*loop_body);
        let buf = b.build_load(llvm_type, buf_ptr, "buf").unwrap();
        writer.llvm_ingest(ctx, &b, buf);
        b.build_unconditional_branch(*loop_cond).unwrap();
    }

    b.position_at_end(exit);
    writer.llvm_flush(ctx, &b);
    b.build_return(None).unwrap();

    llvm_mod.verify().unwrap();
    optimize_module(&llvm_mod)?;
    let ee = llvm_mod
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();
    link_req_helpers(&llvm_mod, &ee)?;

    let concat_func = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void)>("concat")
            .unwrap()
    };

    Ok(concat_func)
}

#[cfg(test)]
mod tests {
    use arrow_array::{cast::AsArray, types::Int32Type, Array, Int32Array, StringArray};
    use itertools::Itertools;

    use crate::{new_kernels::concat::ConcatKernel, Kernel};

    #[test]
    fn test_concat_i32() {
        let d1 = Int32Array::from(vec![1, 2, 3, 4]);
        let d2 = Int32Array::from(vec![5, 6, 7, 8]);

        let k =
            ConcatKernel::compile(&[&d1 as &dyn Array, &d2 as &dyn Array].as_slice(), ()).unwrap();

        let res = k.call(&[&d1 as &dyn Array, &d2 as &dyn Array]).unwrap();
        let res = res.as_primitive::<Int32Type>();
        let res = res.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_concat_strs() {
        let d1 = StringArray::from(vec!["hello", "world"]);
        let d2 = StringArray::from(vec![
            "!",
            "!",
            "this is a longer string that is more than 12 chars",
        ]);

        let k =
            ConcatKernel::compile(&[&d1 as &dyn Array, &d2 as &dyn Array].as_slice(), ()).unwrap();

        let res = k.call(&[&d1 as &dyn Array, &d2 as &dyn Array]).unwrap();
        let res = res.as_string_view();
        let res = res.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(
            res,
            &[
                "hello",
                "world",
                "!",
                "!",
                "this is a longer string that is more than 12 chars"
            ]
        );
    }
}
