use std::{ffi::c_void, sync::LazyLock};

use arrow_array::{Array, ArrayRef};
use arrow_buffer::{BooleanBufferBuilder, NullBuffer};
use arrow_schema::DataType;
use inkwell::{context::Context, execution_engine::JitFunction, AddressSpace, OptimizationLevel};
use ouroboros::self_referencing;

use crate::{
    compiled_iter::{datum_to_iter, generate_next},
    compiled_kernels::{
        cast::coalesce_type,
        link_req_helpers, optimize_module,
        writers::{ArrayWriter, PrimitiveArrayWriter, StringViewWriter, WriterAllocation},
        KernelCache,
    },
    declare_blocks, logical_arrow_type, ArrowKernelError, Kernel, PrimitiveType,
};

pub fn concat_all(data: &[&dyn Array]) -> Result<ArrayRef, ArrowKernelError> {
    match PrimitiveType::for_arrow_type(data[0].data_type()) {
        PrimitiveType::P64x2 => concat_with_writer::<StringViewWriter>(data),
        _ => concat_with_writer::<PrimitiveArrayWriter>(data),
    }
}

fn concat_with_writer<'a, W>(data: &[&dyn Array]) -> Result<ArrayRef, ArrowKernelError>
where
    W: ArrayWriter<'a>,
{
    let total_els = data.iter().map(|x| x.len()).sum::<usize>();
    let mut alloc = W::allocate(
        total_els,
        PrimitiveType::for_arrow_type(data[0].data_type()),
    );

    // create all the nulls, if needed
    let nulls = if data.iter().any(|arr| arr.is_nullable()) {
        let mut bb = BooleanBufferBuilder::new(total_els);
        for el in data.iter() {
            match el.logical_nulls() {
                Some(nb) => bb.append_buffer(nb.inner()),
                None => bb.append_n(el.len(), true),
            }
        }
        Some(NullBuffer::new(bb.finish()))
    } else {
        None
    };

    for arr in data {
        CONCAT_PROGRAM_CACHE.get((*arr, alloc.get_ptr()), ())?;
        alloc.add_last_written_offset(arr.len());
    }

    let arr = alloc.to_array_ref(total_els, nulls);
    coalesce_type(
        arr,
        &match logical_arrow_type(data[0].data_type()) {
            DataType::Utf8 => DataType::Utf8View,
            DataType::LargeUtf8 => DataType::Utf8View,
            DataType::Binary => DataType::BinaryView,
            DataType::LargeBinary => DataType::BinaryView,
            dt => dt,
        },
    )
}

static CONCAT_PROGRAM_CACHE: LazyLock<KernelCache<ConcatKernel>> = LazyLock::new(KernelCache::new);

#[self_referencing]
pub struct ConcatKernel {
    context: Context,
    dtype: DataType,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut c_void)>,
}

unsafe impl Sync for ConcatKernel {}
unsafe impl Send for ConcatKernel {}

impl Kernel for ConcatKernel {
    type Key = DataType;

    type Input<'a> = (&'a dyn Array, *mut c_void);

    type Params = ();

    type Output = ();

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (arr, ptr) = inp;
        let mut iter = datum_to_iter(&arr)?;

        unsafe {
            self.borrow_func().call(iter.get_mut_ptr(), ptr);
        }

        Ok(())
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, _ptr) = inp;
        ConcatKernelTryBuilder {
            context: Context::create(),
            dtype: arr.data_type().clone(),
            func_builder: |ctx| match PrimitiveType::for_arrow_type(arr.data_type()) {
                PrimitiveType::P64x2 => build_concat::<StringViewWriter>(ctx, *arr),
                _ => build_concat::<PrimitiveArrayWriter>(ctx, *arr),
            },
        }
        .try_build()
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok(i.0.data_type().clone())
    }
}

fn build_concat<'a, W: ArrayWriter<'a>>(
    ctx: &'a Context,
    data: &dyn Array,
) -> Result<JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void)>, ArrowKernelError> {
    let llvm_mod = ctx.create_module("concat");
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let pt = PrimitiveType::for_arrow_type(data.data_type());
    let func = llvm_mod.add_function(
        "concat",
        ctx.void_type()
            .fn_type(&[ptr_type.into(), ptr_type.into()], false),
        None,
    );

    let next = generate_next(
        ctx,
        &llvm_mod,
        "next",
        data.data_type(),
        &datum_to_iter(&data)?,
    )
    .unwrap();

    let b = ctx.create_builder();
    declare_blocks!(ctx, func, entry, loop_cond, loop_body, exit);
    let iter_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let alloc_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
    let llvm_type = pt.llvm_type(ctx);

    b.position_at_end(entry);
    let writer = W::llvm_init(ctx, &llvm_mod, &b, pt, alloc_ptr);
    let buf_ptr = b.build_alloca(llvm_type, "buf").unwrap();
    b.build_unconditional_branch(loop_cond).unwrap();

    b.position_at_end(loop_cond);
    let had_next = b
        .build_call(next, &[iter_ptr.into(), buf_ptr.into()], "next")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    b.build_conditional_branch(had_next, loop_body, exit)
        .unwrap();

    b.position_at_end(loop_body);
    let buf = b.build_load(llvm_type, buf_ptr, "buf").unwrap();
    writer.llvm_ingest(ctx, &b, buf);
    b.build_unconditional_branch(loop_cond).unwrap();

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
    use arrow_array::{Int32Array, StringArray};

    use itertools::Itertools;

    use crate::{
        compiled_kernels::{
            concat::ConcatKernel,
            writers::{ArrayWriter, PrimitiveArrayWriter, StringViewWriter, WriterAllocation},
        },
        Kernel,
    };

    #[test]
    fn test_concat_i32() {
        let d1 = Int32Array::from(vec![1, 2, 3, 4]);
        let d2 = Int32Array::from(vec![5, 6, 7, 8]);
        let mut alloc = PrimitiveArrayWriter::allocate(8, crate::PrimitiveType::I32);

        let k = ConcatKernel::compile(&(&d1, alloc.get_ptr()), ()).unwrap();

        k.call((&d1, alloc.get_ptr())).unwrap();
        alloc.add_last_written_offset(4);
        k.call((&d2, alloc.get_ptr())).unwrap();
        alloc.add_last_written_offset(4);

        let res: Int32Array = alloc.into_primitive_array(8, None);
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
        let mut alloc = StringViewWriter::allocate(5, crate::PrimitiveType::P64x2);

        let k = ConcatKernel::compile(&(&d1, alloc.get_ptr()), ()).unwrap();
        k.call((&d1, alloc.get_ptr())).unwrap();
        alloc.add_last_written_offset(2);
        k.call((&d2, alloc.get_ptr())).unwrap();
        alloc.add_last_written_offset(3);

        let res = alloc.to_array(5, None);
        let res = res
            .iter()
            .map(|x| std::str::from_utf8(x.unwrap()).unwrap())
            .collect_vec();
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
