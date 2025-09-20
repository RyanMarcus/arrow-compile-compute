use std::ffi::c_void;

use arrow_array::{Array, ArrayRef, BooleanArray};
use arrow_schema::DataType;
use inkwell::{
    context::Context, execution_engine::JitFunction, values::BasicValue, AddressSpace,
    OptimizationLevel,
};
use itertools::Itertools;
use ouroboros::self_referencing;

use crate::{
    compiled_iter::{datum_to_iter, generate_next, IteratorHolder},
    compiled_kernels::{
        cast::coalesce_type,
        dsl::{base_type, KernelParameters},
        link_req_helpers, optimize_module,
        writers::{
            ArrayWriter, BooleanWriter, PrimitiveArrayWriter, StringViewWriter, WriterAllocation,
        },
    },
    declare_blocks, set_noalias_params, ArrowKernelError, Kernel, PrimitiveType,
};

#[self_referencing]
pub struct PartitionKernel {
    context: Context,
    nparts: usize,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<
        'this,
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void),
    >,
}

unsafe impl Sync for PartitionKernel {}
unsafe impl Send for PartitionKernel {}

impl Kernel for PartitionKernel {
    type Key = (DataType, bool, usize);

    type Input<'a> = (&'a dyn Array, &'a dyn Array);

    type Params = usize;

    type Output = Vec<ArrayRef>;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (arr, part) = inp;

        if part.is_nullable() {
            return Err(ArrowKernelError::UnsupportedArguments(
                "Partition array must not be nullable".to_string(),
            ));
        }

        if part.len() != arr.len() {
            return Err(ArrowKernelError::SizeMismatch);
        }

        let arr_ih = datum_to_iter(&arr)?;
        let part_ih = datum_to_iter(&part)?;
        let null_ih = arr
            .nulls()
            .map(|b| BooleanArray::from(b.clone().into_inner()))
            .map(|b| datum_to_iter(&b))
            .transpose()?;

        // compute the size of each partition
        let mut part_sizes = vec![0_usize; *self.borrow_nparts()];
        match arr.nulls() {
            Some(nulls) => {
                crate::iter::iter_nonnull_u64(part)?
                    .zip(nulls.iter())
                    .filter(|(_x, not_null)| *not_null)
                    .map(|(x, _)| x)
                    .for_each(|x| part_sizes[x as usize] += 1);
            }
            None => {
                crate::iter::iter_nonnull_u64(part)?.for_each(|x| part_sizes[x as usize] += 1);
            }
        };

        let base = base_type(arr.data_type());
        let ptype = PrimitiveType::for_arrow_type(&base);
        let res = match &base {
            DataType::Utf8 => call_kernel_with_writer::<StringViewWriter>(
                ptype,
                arr_ih,
                part_ih,
                null_ih,
                part_sizes,
                self.borrow_func(),
            ),
            DataType::Boolean => call_kernel_with_writer::<BooleanWriter>(
                ptype,
                arr_ih,
                part_ih,
                null_ih,
                part_sizes,
                self.borrow_func(),
            ),
            t if t.is_primitive() => call_kernel_with_writer::<PrimitiveArrayWriter>(
                ptype,
                arr_ih,
                part_ih,
                null_ih,
                part_sizes,
                self.borrow_func(),
            ),
            _ => todo!("cannot partition this type"),
        };
        let base = if base == DataType::Utf8 {
            DataType::Utf8View
        } else {
            base
        };
        res?.into_iter().map(|x| coalesce_type(x, &base)).collect()
    }

    fn compile(inp: &Self::Input<'_>, nparts: Self::Params) -> Result<Self, ArrowKernelError> {
        PartitionKernelTryBuilder {
            context: Context::create(),
            nparts,
            func_builder: |ctx| {
                let base = base_type(inp.0.data_type());
                match base {
                    DataType::Utf8 => build_partition::<StringViewWriter>(
                        ctx,
                        nparts,
                        inp.0,
                        inp.0
                            .nulls()
                            .map(|b| BooleanArray::from(b.clone().into_inner()))
                            .as_ref(),
                        inp.1,
                    ),
                    DataType::Boolean => build_partition::<BooleanWriter>(
                        ctx,
                        nparts,
                        inp.0,
                        inp.0
                            .nulls()
                            .map(|b| BooleanArray::from(b.clone().into_inner()))
                            .as_ref(),
                        inp.1,
                    ),
                    t if t.is_primitive() => build_partition::<PrimitiveArrayWriter>(
                        ctx,
                        nparts,
                        inp.0,
                        inp.0
                            .nulls()
                            .map(|b| BooleanArray::from(b.clone().into_inner()))
                            .as_ref(),
                        inp.1,
                    ),
                    _ => todo!("cannot partition this type"),
                }
            },
        }
        .try_build()
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((i.0.data_type().clone(), i.0.nulls().is_some(), *p))
    }
}

fn call_kernel_with_writer<'a, W: ArrayWriter<'a>>(
    ptype: PrimitiveType,
    mut arr_ih: IteratorHolder,
    mut part_ih: IteratorHolder,
    mut null_ih: Option<IteratorHolder>,
    part_sizes: Vec<usize>,
    func: &JitFunction<
        'a,
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void),
    >,
) -> Result<Vec<ArrayRef>, ArrowKernelError> {
    let mut allocs = part_sizes
        .iter()
        .map(|size| W::allocate(*size, ptype))
        .collect_vec();
    let alloc_ptrs = allocs.iter_mut().map(|a| a.get_ptr()).collect_vec();
    let mut kp = KernelParameters::new(alloc_ptrs);
    unsafe {
        func.call(
            arr_ih.get_mut_ptr(),
            null_ih
                .as_mut()
                .map(|ih| ih.get_mut_ptr())
                .unwrap_or(std::ptr::null_mut()),
            part_ih.get_mut_ptr(),
            kp.get_mut_ptr(),
        );
    }
    Ok(allocs
        .into_iter()
        .zip(part_sizes)
        .map(|(a, size)| a.to_array_ref(size, None))
        .collect_vec())
}

fn build_partition<'a, W: ArrayWriter<'a>>(
    ctx: &'a Context,
    nparts: usize,
    arr: &dyn Array,
    nulls: Option<&BooleanArray>,
    partition_idxes: &dyn Array,
) -> Result<
    JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void)>,
    ArrowKernelError,
> {
    let llvm_mod = ctx.create_module("partition");
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let ptype = PrimitiveType::for_arrow_type(arr.data_type());
    let idx_ptype = PrimitiveType::for_arrow_type(partition_idxes.data_type());
    let idx_llvm_type = idx_ptype.llvm_type(ctx);

    let llvm_type = ptype.llvm_type(ctx);

    let fn_type = ctx.void_type().fn_type(
        &[
            ptr_type.into(), // array iterator
            ptr_type.into(), // null iterator (optional)
            ptr_type.into(), // partition iterator
            ptr_type.into(), // pointer to writer pointers
        ],
        false,
    );
    let func = llvm_mod.add_function("partition", fn_type, None);
    let b = ctx.create_builder();

    let arr_ih = datum_to_iter(&arr)?;
    let idx_ih = datum_to_iter(&partition_idxes)?;
    let null_ih = nulls.map(|nulls| datum_to_iter(&&nulls)).transpose()?;

    let arr_next = generate_next(ctx, &llvm_mod, "arr", arr.data_type(), &arr_ih).unwrap();
    let idx_next =
        generate_next(ctx, &llvm_mod, "idx", partition_idxes.data_type(), &idx_ih).unwrap();
    let null_next = null_ih
        .as_ref()
        .map(|ih| generate_next(ctx, &llvm_mod, "null", &DataType::Boolean, ih).unwrap());
    let arr_iter_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let null_iter_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
    let idx_iter_ptr = func.get_nth_param(2).unwrap().into_pointer_value();

    declare_blocks!(ctx, func, entry, loop_cond, null_check, loop_body, exit);

    b.position_at_end(entry);
    let writer_ptr_ptr = func.get_nth_param(3).unwrap().into_pointer_value();
    let writers = (0..nparts)
        .map(|writer_offset| {
            let alloc_ptr = KernelParameters::llvm_get(ctx, &b, writer_ptr_ptr, writer_offset);
            W::llvm_init(ctx, &llvm_mod, &b, ptype, alloc_ptr)
        })
        .collect_vec();
    let arr_buf_ptr = b.build_alloca(llvm_type, "arr_buf_ptr").unwrap();
    let idx_buf_ptr = b.build_alloca(idx_llvm_type, "idx_buf_ptr").unwrap();
    let null_buf_ptr = b.build_alloca(ctx.bool_type(), "null_buf_ptr").unwrap();

    let arr_iter_ptr = arr_ih.localize_struct(ctx, &b, arr_iter_ptr);
    let idx_iter_ptr = idx_ih.localize_struct(ctx, &b, idx_iter_ptr);

    b.build_unconditional_branch(loop_cond).unwrap();

    b.position_at_end(loop_cond);
    let had_next = b
        .build_call(arr_next, &[arr_iter_ptr.into(), arr_buf_ptr.into()], "next")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    b.build_call(
        idx_next,
        &[idx_iter_ptr.into(), idx_buf_ptr.into()],
        "next_idx",
    )
    .unwrap();
    b.build_conditional_branch(had_next, null_check, exit)
        .unwrap();

    b.position_at_end(null_check);
    match null_next {
        Some(null_next) => {
            b.build_call(
                null_next,
                &[null_iter_ptr.into(), null_buf_ptr.into()],
                "had_next_null",
            )
            .unwrap();
            let was_nonnull = b
                .build_load(ctx.bool_type(), null_buf_ptr, "was_nonnull")
                .unwrap()
                .into_int_value();
            b.build_conditional_branch(was_nonnull, loop_body, loop_cond)
                .unwrap();
        }
        None => {
            b.build_unconditional_branch(loop_body).unwrap();
        }
    };

    b.position_at_end(loop_body);
    let part_idx = b
        .build_load(idx_llvm_type, idx_buf_ptr, "part_idx")
        .unwrap()
        .into_int_value();
    let part_idx = b
        .build_int_z_extend_or_bit_cast(part_idx, ctx.i64_type(), "part_idx")
        .unwrap();
    let val = b.build_load(llvm_type, arr_buf_ptr, "val").unwrap();

    let switch_blocks = writers
        .iter()
        .enumerate()
        .map(|(writer_offset, writer)| {
            let block = ctx.append_basic_block(func, &format!("write_to_{}", writer_offset));
            let b = ctx.create_builder();
            b.position_at_end(block);
            if base_type(arr.data_type()) == DataType::Boolean {
                writer.llvm_ingest(
                    ctx,
                    &b,
                    b.build_int_truncate(val.into_int_value(), ctx.bool_type(), "trunc")
                        .unwrap()
                        .as_basic_value_enum(),
                );
            } else {
                writer.llvm_ingest(ctx, &b, val);
            }
            b.build_unconditional_branch(loop_cond).unwrap();
            (ctx.i64_type().const_int(writer_offset as u64, false), block)
        })
        .collect_vec();
    b.build_switch(part_idx, exit, &switch_blocks).unwrap();

    b.position_at_end(exit);
    for writer in writers {
        writer.llvm_flush(ctx, &b);
    }
    b.build_return(None).unwrap();

    set_noalias_params(&func);
    llvm_mod.verify().unwrap();
    optimize_module(&llvm_mod)?;
    let ee = llvm_mod
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();
    link_req_helpers(&llvm_mod, &ee)?;

    let partition_func = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void)>(
            "partition",
        )
        .unwrap()
    };

    Ok(partition_func)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray, types::Int32Type, ArrayRef, BooleanArray, Int32Array, UInt32Array,
    };
    use itertools::Itertools;

    use crate::{compiled_kernels::partition::PartitionKernel, Kernel};

    #[test]
    fn test_part_i32_nonulls() {
        let data: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
        let k = PartitionKernel::compile(&(&data, &part), 2).unwrap();
        let res = k.call((&data, &part)).unwrap();

        assert_eq!(res.len(), 2);
        assert_eq!(
            res[0].as_primitive::<Int32Type>().values(),
            &[1, 3, 5, 7, 9]
        );
        assert_eq!(
            res[1].as_primitive::<Int32Type>().values(),
            &[2, 4, 6, 8, 10]
        );
    }

    #[test]
    fn test_part_i32_nulls() {
        let data: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(1),
            Some(2),
            None,
            Some(4),
            Some(5),
            None,
            Some(7),
            Some(8),
            Some(9),
            None,
        ]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
        let k = PartitionKernel::compile(&(&data, &part), 2).unwrap();
        let res = k.call((&data, &part)).unwrap();

        assert_eq!(res.len(), 2);
        assert_eq!(res[0].as_primitive::<Int32Type>().values(), &[1, 5, 7, 9]);
        assert_eq!(res[1].as_primitive::<Int32Type>().values(), &[2, 4, 8]);
    }

    #[test]
    fn test_part_bool() {
        let data: ArrayRef = Arc::new(BooleanArray::from(vec![
            true, true, true, true, true, false, false, false, false, false,
        ]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
        let k = PartitionKernel::compile(&(&data, &part), 2).unwrap();
        let res = k.call((&data, &part)).unwrap();

        assert_eq!(res.len(), 2);
        assert_eq!(
            res[0].as_boolean().iter().map(|x| x.unwrap()).collect_vec(),
            &[true, true, true, false, false]
        );
        assert_eq!(
            res[1].as_boolean().iter().map(|x| x.unwrap()).collect_vec(),
            &[true, true, false, false, false]
        );
    }

    #[test]
    fn test_part_strs() {
        let data: ArrayRef = Arc::new(arrow_array::StringArray::from(vec![
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        ]));
        let part: ArrayRef = Arc::new(UInt32Array::from(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
        let k = PartitionKernel::compile(&(&data, &part), 2).unwrap();
        let res = k.call((&data, &part)).unwrap();

        assert_eq!(res.len(), 2);
        assert_eq!(
            res[0]
                .as_string_view()
                .iter()
                .map(|x| x.unwrap())
                .collect_vec(),
            &["a", "c", "e", "g", "i"]
        );
        assert_eq!(
            res[1]
                .as_string_view()
                .iter()
                .map(|x| x.unwrap())
                .collect_vec(),
            &["b", "d", "f", "h", "j"]
        );
    }
}
