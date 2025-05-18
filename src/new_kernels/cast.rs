use std::ffi::c_void;

use arrow_array::{
    cast::AsArray,
    make_array, new_empty_array,
    types::{
        Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrayRef, RunArray,
};
use arrow_buffer::Buffer;
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use inkwell::{context::Context, execution_engine::JitFunction, AddressSpace, OptimizationLevel};
use ouroboros::self_referencing;

use crate::{
    declare_blocks, empty_array_for, increment_pointer,
    new_iter::{datum_to_iter, generate_next, generate_next_block, IteratorHolder},
    new_kernels::{
        add_ptrx2_to_view, gen_convert_numeric_vec,
        ht::{generate_hash_func, generate_lookup_or_insert, TicketTable},
        optimize_module,
    },
    PrimitiveType,
};

use super::{ArrowKernelError, Kernel};

/// Iterates over an array's string/binary data buffers
fn iter_buffers<'a>(inp: &'a dyn Array) -> Box<dyn Iterator<Item = &'a Buffer> + 'a> {
    match inp.data_type() {
        DataType::Utf8 => Box::new(vec![inp.as_string::<i32>().values()].into_iter()),
        DataType::LargeUtf8 => Box::new(vec![inp.as_string::<i64>().values()].into_iter()),
        DataType::Utf8View => Box::new(inp.as_string_view().data_buffers().iter()),
        DataType::Dictionary(kt, _vt) => match **kt {
            DataType::UInt8 => iter_buffers(inp.as_dictionary::<UInt8Type>().values()),
            DataType::UInt16 => iter_buffers(inp.as_dictionary::<UInt16Type>().values()),
            DataType::UInt32 => iter_buffers(inp.as_dictionary::<UInt32Type>().values()),
            DataType::UInt64 => iter_buffers(inp.as_dictionary::<UInt64Type>().values()),
            DataType::Int8 => iter_buffers(inp.as_dictionary::<Int8Type>().values()),
            DataType::Int16 => iter_buffers(inp.as_dictionary::<Int16Type>().values()),
            DataType::Int32 => iter_buffers(inp.as_dictionary::<Int32Type>().values()),
            DataType::Int64 => iter_buffers(inp.as_dictionary::<Int64Type>().values()),
            _ => unreachable!("invalid dict key type"),
        },
        DataType::RunEndEncoded(_re_t, v_t) => match v_t.data_type() {
            DataType::Int16 => iter_buffers(
                inp.as_any()
                    .downcast_ref::<RunArray<Int16Type>>()
                    .unwrap()
                    .values(),
            ),
            DataType::Int32 => iter_buffers(
                inp.as_any()
                    .downcast_ref::<RunArray<Int32Type>>()
                    .unwrap()
                    .values(),
            ),
            DataType::Int64 => iter_buffers(
                inp.as_any()
                    .downcast_ref::<RunArray<Int64Type>>()
                    .unwrap()
                    .values(),
            ),
            _ => unreachable!("invalid dict key type"),
        },
        _ => Box::new(vec![].into_iter()),
    }
}

#[self_referencing]
pub struct CastToFlatKernel {
    context: Context,
    lhs_data_type: DataType,
    tar_data_type: DataType,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut c_void)>,
}

unsafe impl Sync for CastToFlatKernel {}
unsafe impl Send for CastToFlatKernel {}

impl Kernel for CastToFlatKernel {
    type Key = (DataType, DataType);

    type Input<'a> = &'a dyn Array;

    type Params = DataType;

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        if inp.data_type() != self.borrow_lhs_data_type() {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "kernel expected {:?}, got {:?}",
                self.borrow_lhs_data_type(),
                inp.data_type()
            )));
        }

        if inp.is_empty() {
            return Ok(empty_array_for(self.borrow_tar_data_type()));
        }

        let mut inp_iter = datum_to_iter(&inp)?;
        let tar_dt = self.borrow_tar_data_type().clone();
        if tar_dt.is_primitive() {
            let out_prim = PrimitiveType::for_arrow_type(&tar_dt);
            let out_size = inp.len() * out_prim.width();
            let mut buf = vec![0u8; out_size];
            let arr_data = unsafe {
                self.borrow_func()
                    .call(inp_iter.get_mut_ptr(), buf.as_mut_ptr() as *mut c_void);

                ArrayDataBuilder::new(tar_dt)
                    .nulls(inp.nulls().cloned())
                    .buffers(vec![Buffer::from(buf)])
                    .len(inp.len())
                    .build_unchecked()
            };

            return Ok(make_array(arr_data));
        } else if matches!(tar_dt, DataType::Utf8View | DataType::BinaryView) {
            let out_size = inp.len() * 16;
            let mut buf = vec![0u8; out_size];
            let arr_data = unsafe {
                self.borrow_func()
                    .call(inp_iter.get_mut_ptr(), buf.as_mut_ptr() as *mut c_void);

                ArrayDataBuilder::new(tar_dt)
                    .nulls(inp.nulls().cloned())
                    .buffers(vec![Buffer::from(buf)])
                    .add_buffers(iter_buffers(inp).cloned())
                    .len(inp.len())
                    .build_unchecked()
            };
            return Ok(make_array(arr_data));
        }

        todo!()
    }

    fn compile(inp: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let in_type = inp.data_type();
        let out_type = &params;
        assert_ne!(
            in_type, out_type,
            "cannot compile kernel for identical types {:?}",
            in_type
        );

        let in_iter = datum_to_iter(inp)?;

        let ctx = Context::create();
        if out_type.is_primitive() {
            CastToFlatKernelTryBuilder {
                context: ctx,
                lhs_data_type: inp.data_type().clone(),
                tar_data_type: out_type.clone(),
                func_builder: |ctx| generate_block_cast_to_flat(ctx, *inp, &in_iter, out_type),
            }
            .try_build()
        } else if matches!(out_type, DataType::Utf8View | DataType::BinaryView) {
            CastToFlatKernelTryBuilder {
                context: ctx,
                lhs_data_type: inp.data_type().clone(),
                tar_data_type: out_type.clone(),
                func_builder: |ctx| generate_cast_to_view(ctx, *inp, &in_iter),
            }
            .try_build()
        } else {
            todo!()
        }
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((i.data_type().clone(), p.clone()))
    }
}

pub fn generate_cast_to_view<'a>(
    ctx: &'a Context,
    lhs: &dyn Array,
    lhs_iter: &IteratorHolder,
) -> Result<JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void)>, ArrowKernelError> {
    let i64_type = ctx.i64_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let lhs_prim_type = PrimitiveType::for_arrow_type(lhs.data_type());
    let lhs_type = lhs_prim_type.llvm_type(ctx);
    let module = ctx.create_module("cast_kernel");

    let func = module.add_function(
        "cast_to_view",
        ctx.void_type()
            .fn_type(&[ptr_type.into(), ptr_type.into()], false),
        None,
    );

    let convert = add_ptrx2_to_view(ctx, &module);
    let next = generate_next(ctx, &module, "next", lhs.data_type(), lhs_iter).unwrap();
    let build = ctx.create_builder();

    declare_blocks!(ctx, func, entry, loop_cond, loop_body, exit);
    build.position_at_end(entry);
    let iter_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let out_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
    let out_idx_ptr = build.build_alloca(i64_type, "out_idx_ptr").unwrap();
    build
        .build_store(out_idx_ptr, i64_type.const_zero())
        .unwrap();
    let buf = build.build_alloca(lhs_type, "buf").unwrap();
    build.build_unconditional_branch(loop_cond).unwrap();

    build.position_at_end(loop_cond);
    let res = build
        .build_call(next, &[iter_ptr.into(), buf.into()], "next")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_conditional_branch(res, loop_body, exit)
        .unwrap();

    build.position_at_end(loop_body);
    let val = build
        .build_load(lhs_type, buf, "val")
        .unwrap()
        .into_struct_value();
    let base_ptr = lhs_iter
        .llvm_get_base_ptr(ctx, &build, iter_ptr)
        .ok_or_else(|| {
            ArrowKernelError::ArgumentMismatch(format!(
                "unable to get base data pointer for type {:?}",
                lhs.data_type()
            ))
        })?;
    let val = build
        .build_call(convert, &[val.into(), base_ptr.into()], "converted")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left();
    let curr_out_idx = build
        .build_load(i64_type, out_idx_ptr, "curr_out_idx")
        .unwrap()
        .into_int_value();
    let curr_out_ptr = increment_pointer!(ctx, build, out_ptr, 16, curr_out_idx);
    build.build_store(curr_out_ptr, val).unwrap();

    let next_out_idx = build
        .build_int_add(curr_out_idx, i64_type.const_int(1, false), "next_out_idx")
        .unwrap();
    build.build_store(out_idx_ptr, next_out_idx).unwrap();
    build.build_unconditional_branch(loop_cond).unwrap();

    build.position_at_end(exit);
    build.build_return(None).unwrap();

    module.verify().unwrap();
    optimize_module(&module)?;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    Ok(unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void)>(
            func.get_name().to_str().unwrap(),
        )
        .unwrap()
    })
}

pub fn generate_block_cast_to_flat<'a>(
    ctx: &'a Context,
    lhs: &dyn Array,
    lhs_iter: &IteratorHolder,
    to: &DataType,
) -> Result<JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void)>, ArrowKernelError> {
    let lhs_prim = PrimitiveType::for_arrow_type(lhs.data_type());
    let lhs_type = lhs_prim.llvm_type(ctx);
    let to_prim = PrimitiveType::for_arrow_type(to);
    let ptr_type = ctx.ptr_type(AddressSpace::default());

    let build = ctx.create_builder();
    let module = ctx.create_module("cast_kernel");

    // assume a 512-bit vector width (will still work for smaller vector widths)
    let vec_type = lhs_prim
        .llvm_vec_type(ctx, 32)
        .ok_or_else(|| ArrowKernelError::NonVectorizableType(lhs.data_type().clone()))?;
    let next_block = generate_next_block::<32>(ctx, &module, "cast", lhs.data_type(), lhs_iter)
        .ok_or_else(|| ArrowKernelError::NonVectorizableType(lhs.data_type().clone()))?;

    let next = generate_next(ctx, &module, "cast", lhs.data_type(), lhs_iter).unwrap();

    let fn_type = ctx
        .void_type()
        .fn_type(&[ptr_type.into(), ptr_type.into()], false);
    let function = module.add_function("cast", fn_type, None);
    let lhs_iter_ptr = function.get_nth_param(0).unwrap().into_pointer_value();

    declare_blocks!(ctx, function, entry, block_cond, block_body, tail_cond, tail_body, exit);
    build.position_at_end(entry);
    let out_ptr_ptr = build.build_alloca(ptr_type, "out_ptr").unwrap();
    build
        .build_store(
            out_ptr_ptr,
            function.get_nth_param(1).unwrap().into_pointer_value(),
        )
        .unwrap();
    let vbuf = build.build_alloca(vec_type, "vbuf").unwrap();
    let buf = build.build_alloca(lhs_type, "buf").unwrap();
    build.build_unconditional_branch(block_cond).unwrap();

    build.position_at_end(block_cond);
    let had_next = build
        .build_call(next_block, &[lhs_iter_ptr.into(), vbuf.into()], "get_next")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_conditional_branch(had_next, block_body, tail_cond)
        .unwrap();

    build.position_at_end(block_body);
    let data = build
        .build_load(vec_type, vbuf, "data")
        .unwrap()
        .into_vector_value();
    let converted = gen_convert_numeric_vec(ctx, &build, data, lhs_prim, to_prim);
    let out_ptr = build
        .build_load(ptr_type, out_ptr_ptr, "out_ptr")
        .unwrap()
        .into_pointer_value();
    build.build_store(out_ptr, converted).unwrap();
    let new_out_ptr = increment_pointer!(
        ctx,
        build,
        out_ptr,
        to_prim.width() * vec_type.get_size() as usize
    );
    build.build_store(out_ptr_ptr, new_out_ptr).unwrap();
    build.build_unconditional_branch(block_cond).unwrap();

    build.position_at_end(tail_cond);
    let had_next = build
        .build_call(next, &[lhs_iter_ptr.into(), buf.into()], "get_next")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_conditional_branch(had_next, tail_body, exit)
        .unwrap();

    build.position_at_end(tail_body);
    let data = build.build_load(lhs_type, buf, "data").unwrap();
    let data = build
        .build_bit_cast(
            data,
            lhs_prim.llvm_vec_type(ctx, 1).unwrap(),
            "singleton_vec",
        )
        .unwrap()
        .into_vector_value();

    let converted = gen_convert_numeric_vec(ctx, &build, data, lhs_prim, to_prim);
    let out_ptr = build
        .build_load(ptr_type, out_ptr_ptr, "out_ptr")
        .unwrap()
        .into_pointer_value();
    build.build_store(out_ptr, converted).unwrap();
    let new_out_ptr = increment_pointer!(ctx, build, out_ptr, to_prim.width());
    build.build_store(out_ptr_ptr, new_out_ptr).unwrap();
    build.build_unconditional_branch(tail_cond).unwrap();

    build.position_at_end(exit);
    build.build_return(None).unwrap();

    module.verify().unwrap();
    optimize_module(&module)?;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    Ok(unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void)>(
            function.get_name().to_str().unwrap(),
        )
        .unwrap()
    })
}

pub fn generate_cast_to_dict<'a>(
    ctx: &'a Context,
    lhs: &dyn Array,
    lhs_iter: &IteratorHolder,
    dict_key_type: &DataType,
    dict_val_type: &DataType,
) -> Result<
    JitFunction<
        'a,
        unsafe extern "C" fn(*mut TicketTable, *mut c_void, *mut c_void, *mut c_void) -> i64,
    >,
    ArrowKernelError,
> {
    let module = ctx.create_module("cmp_kernel");
    let build = ctx.create_builder();

    let next = generate_next(ctx, &module, "cast", lhs.data_type(), lhs_iter).unwrap();
    let ticket = generate_lookup_or_insert(
        ctx,
        &module,
        &TicketTable::new(0, dict_val_type.clone(), dict_key_type.clone()),
    );
    let lhs_prim_type = PrimitiveType::for_arrow_type(lhs.data_type());
    let lhs_type = lhs_prim_type.llvm_type(ctx);
    let val_prim_type = PrimitiveType::for_arrow_type(dict_val_type);
    let key_prim_type = PrimitiveType::for_arrow_type(dict_key_type);

    let hash = generate_hash_func(ctx, &module, lhs_prim_type);

    let i8_type = ctx.i8_type();
    let i64_type = ctx.i64_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let func = module.add_function(
        "cast",
        i64_type.fn_type(
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

    declare_blocks!(
        ctx,
        func,
        entry,
        loop_cond,
        loop_body,
        previously_seen_value,
        new_value,
        table_full,
        exit
    );

    build.position_at_end(entry);
    let tt_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let iter_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
    let out_keys = func.get_nth_param(2).unwrap().into_pointer_value();
    let out_vals = func.get_nth_param(3).unwrap().into_pointer_value();
    let val_idx_ptr = build.build_alloca(i64_type, "val_idx_ptr").unwrap();
    let key_idx_ptr = build.build_alloca(i64_type, "key_idx_ptr").unwrap();
    build
        .build_store(key_idx_ptr, i64_type.const_zero())
        .unwrap();
    build
        .build_store(val_idx_ptr, i64_type.const_zero())
        .unwrap();
    let buf_ptr = build.build_alloca(lhs_type, "buf_ptr").unwrap();
    let ht_res_ptr = build.build_alloca(i8_type, "ht_res_ptr").unwrap();
    build.build_unconditional_branch(loop_cond).unwrap();

    build.position_at_end(loop_cond);
    let had_next = build
        .build_call(next, &[iter_ptr.into(), buf_ptr.into()], "had_next")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    build
        .build_conditional_branch(had_next, loop_body, exit)
        .unwrap();

    build.position_at_end(loop_body);
    let next_val = build.build_load(lhs_type, buf_ptr, "next_val").unwrap();
    let hashed = build
        .build_call(hash, &[next_val.into()], "hashed")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    let ticket = build
        .build_call(
            ticket,
            &[
                tt_ptr.into(),
                next_val.into(),
                hashed.into(),
                ht_res_ptr.into(),
            ],
            "ticket",
        )
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    let status = build
        .build_load(i8_type, ht_res_ptr, "status")
        .unwrap()
        .into_int_value();
    build
        .build_switch(
            status,
            table_full,
            &[
                (i8_type.const_int(0, false), previously_seen_value),
                (i8_type.const_int(1, false), new_value),
                (i8_type.const_int(2, false), table_full),
            ],
        )
        .unwrap();

    build.position_at_end(new_value);
    let cur_val_idx = build
        .build_load(i64_type, val_idx_ptr, "cur_val_idx")
        .unwrap()
        .into_int_value();
    let next_val_ptr = increment_pointer!(ctx, build, out_vals, val_prim_type.width(), cur_val_idx);
    // TODO convert types here
    assert_eq!(lhs_prim_type, val_prim_type);
    let converted = if let IteratorHolder::String(str_iter) = lhs_iter {
        let convert = add_ptrx2_to_view(ctx, &module);
        build
            .build_call(
                convert,
                &[
                    next_val.into(),
                    str_iter.llvm_get_data_ptr(ctx, &build, iter_ptr).into(),
                ],
                "str_view",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
    } else {
        next_val
    };
    build.build_store(next_val_ptr, converted).unwrap();

    let next_val_idx = build
        .build_int_add(cur_val_idx, i64_type.const_int(1, false), "next_val_idx")
        .unwrap();
    build.build_store(val_idx_ptr, next_val_idx).unwrap();
    build
        .build_unconditional_branch(previously_seen_value)
        .unwrap();

    build.position_at_end(previously_seen_value);
    let cur_key_idx = build
        .build_load(i64_type, key_idx_ptr, "cur_key_idx")
        .unwrap()
        .into_int_value();

    let cur_key_ptr = increment_pointer!(ctx, build, out_keys, key_prim_type.width(), cur_key_idx);
    build.build_store(cur_key_ptr, ticket).unwrap();

    let next_key_idx = build
        .build_int_add(cur_key_idx, i64_type.const_int(1, false), "next_key_idx")
        .unwrap();
    build.build_store(key_idx_ptr, next_key_idx).unwrap();
    build.build_unconditional_branch(loop_cond).unwrap();

    build.position_at_end(table_full);
    build
        .build_return(Some(&i64_type.const_int((-1_i64) as u64, true)))
        .unwrap();

    build.position_at_end(exit);

    let val_idx = build
        .build_load(i64_type, val_idx_ptr, "val_idx")
        .unwrap()
        .into_int_value();
    build.build_return(Some(&val_idx)).unwrap();

    module.verify().unwrap();
    optimize_module(&module)?;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    Ok(unsafe { ee.get_function(func.get_name().to_str().unwrap()).unwrap() })
}

#[self_referencing]
pub struct CastToDictKernel {
    context: Context,
    lhs_data_type: DataType,
    key_data_type: DataType,
    val_data_type: DataType,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<
        'this,
        unsafe extern "C" fn(*mut TicketTable, *mut c_void, *mut c_void, *mut c_void) -> i64,
    >,
}

unsafe impl Sync for CastToDictKernel {}
unsafe impl Send for CastToDictKernel {}

impl Kernel for CastToDictKernel {
    type Key = (DataType, DataType);

    type Input<'a> = &'a dyn Array;

    type Params = DataType;

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let len = inp.len();

        if len == 0 {
            return Ok(new_empty_array(&DataType::Dictionary(
                Box::new(self.borrow_key_data_type().clone()),
                Box::new(self.borrow_val_data_type().clone()),
            )));
        }

        let num_values = usize::min(
            len * 2,
            match self.borrow_key_data_type() {
                DataType::Int8 => i8::MAX as usize,
                DataType::Int16 => i16::MAX as usize,
                DataType::Int32 => i32::MAX as usize,
                DataType::Int64 => i64::MAX as usize,
                _ => {
                    return Err(ArrowKernelError::UnsupportedArguments(format!(
                        "dictionary key type should be a signed int, got {}",
                        self.borrow_key_data_type()
                    )))
                }
            },
        );

        let key_width = PrimitiveType::for_arrow_type(self.borrow_key_data_type()).width();
        let val_width = PrimitiveType::for_arrow_type(self.borrow_val_data_type()).width();

        let mut k_data = vec![0_u8; len * key_width];
        let mut v_data = vec![0_u8; num_values * val_width];
        let mut tt = TicketTable::new(
            num_values,
            self.borrow_val_data_type().clone(),
            self.borrow_key_data_type().clone(),
        );
        let mut iter = datum_to_iter(&inp)?;
        let result = unsafe {
            self.borrow_func().call(
                &mut tt,
                iter.get_mut_ptr(),
                k_data.as_mut_ptr() as *mut c_void,
                v_data.as_mut_ptr() as *mut c_void,
            )
        };

        if result < 0 {
            return Err(ArrowKernelError::DictionaryFullError(
                self.borrow_key_data_type().clone(),
            ));
        }
        v_data.shrink_to_fit();
        let num_values = result as usize;
        let data = unsafe {
            let values = ArrayDataBuilder::new(
                PrimitiveType::for_arrow_type(self.borrow_val_data_type()).as_arrow_type(),
            )
            .add_buffer(Buffer::from(v_data))
            .add_buffers(iter_buffers(inp).cloned())
            .len(num_values)
            .build_unchecked();

            ArrayDataBuilder::new(DataType::Dictionary(
                Box::new(self.borrow_key_data_type().clone()),
                Box::new(self.borrow_val_data_type().clone()),
            ))
            .add_buffer(Buffer::from(k_data))
            .add_child_data(values)
            .len(len)
            .build_unchecked()
        };

        Ok(make_array(data))
    }

    fn compile(inp: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let out_type = &params;
        assert_ne!(
            inp.data_type(),
            out_type,
            "cannot compile kernel for identical types {:?}",
            out_type
        );

        if let DataType::Dictionary(k_dt, v_dt) = out_type {
            let in_iter = datum_to_iter(inp)?;
            let ctx = Context::create();
            CastToDictKernelTryBuilder {
                context: ctx,
                lhs_data_type: inp.data_type().clone(),
                key_data_type: *k_dt.clone(),
                val_data_type: *v_dt.clone(),
                func_builder: |ctx| generate_cast_to_dict(ctx, *inp, &in_iter, k_dt, v_dt),
            }
            .try_build()
        } else {
            panic!(
                "cannot compile dict kernel with non-dict target: {:?}",
                out_type
            );
        }
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((i.data_type().clone(), p.clone()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray,
        types::{Int32Type, Int8Type},
        Array, ArrayRef, DictionaryArray, Int32Array, Int64Array, StringArray, UInt8Array,
    };
    use arrow_schema::DataType;
    use itertools::Itertools;

    use crate::{
        dictionary_data_type,
        new_kernels::{
            cast::{CastToDictKernel, CastToFlatKernel},
            Kernel,
        },
    };

    #[test]
    fn test_i32_to_i64() {
        let data = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let expected: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
        let k = CastToFlatKernel::compile(&(&data as &dyn Array), DataType::Int64).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i32_to_i64_block() {
        let data = Int32Array::from((0..200).collect_vec());
        let expected: ArrayRef = Arc::new(Int64Array::from((0..200).collect_vec()));
        let k = CastToFlatKernel::compile(&(&data as &dyn Array), DataType::Int64).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i64_to_u8_block() {
        let data = Int64Array::from((0..200).collect_vec());
        let expected: ArrayRef = Arc::new(UInt8Array::from((0..200).collect_vec()));
        let k = CastToFlatKernel::compile(&(&data as &dyn Array), DataType::UInt8).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(&res, &expected);
    }

    #[test]
    fn test_i32_to_dict() {
        let data = Int32Array::from(vec![1, 1, 1, 2, 2, 300, 300, 400]);
        let k = CastToDictKernel::compile(
            &(&data as &dyn Array),
            dictionary_data_type(DataType::Int8, DataType::Int32),
        )
        .unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(res.len(), 8);

        let res = res.as_dictionary::<Int8Type>();
        assert_eq!(
            &[1, 2, 300, 400],
            res.values().as_primitive::<Int32Type>().values()
        );
        assert_eq!(&[0, 0, 0, 1, 1, 2, 2, 3], res.keys().values());
    }

    #[test]
    fn test_dict_to_i32() {
        let keys = UInt8Array::from(vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5]);
        let values = Int32Array::from(vec![-100, -200, -300, -400, -500, -600]);
        let da = DictionaryArray::new(keys, Arc::new(values));
        let k = CastToFlatKernel::compile(&(&da as &dyn Array), DataType::Int32).unwrap();

        let res = k.call(&da).unwrap();
        assert_eq!(
            res.as_primitive::<Int32Type>().values(),
            &[-100, -100, -100, -200, -200, -200, -300, -300, -300, -400, -500, -500, -600, -600]
        );
    }

    #[test]
    fn test_str_to_dict() {
        let data = StringArray::from(vec![
            "this",
            "this",
            "is",
            "a test",
            "a test",
            "a string that is longer than 12 chars",
        ]);
        let k = CastToDictKernel::compile(
            &(&data as &dyn Array),
            dictionary_data_type(DataType::Int8, DataType::Utf8),
        )
        .unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(res.len(), 6);
        let res = res.as_dictionary::<Int8Type>();
        let strv = res.values().as_string_view();
        assert_eq!("this", strv.value(0));
        assert_eq!("is", strv.value(1));
        assert_eq!("a test", strv.value(2));
        assert_eq!(&[0, 0, 1, 2, 2, 3], res.keys().values());
    }

    #[test]
    fn test_dict_to_str() {
        let data = StringArray::from(vec![
            "this",
            "this",
            "is",
            "a test",
            "a test",
            "a string that is longer than 12 chars",
        ]);
        let ddata =
            arrow_cast::cast::cast(&data, &dictionary_data_type(DataType::Int8, DataType::Utf8))
                .unwrap();
        let k = CastToFlatKernel::compile(&(&ddata as &dyn Array), DataType::Utf8View).unwrap();
        let res = k.call(&ddata).unwrap();
        let res = res.as_string_view();

        assert_eq!(res.len(), data.len());
        for (ours, orig) in res.iter().zip(data.iter()) {
            assert_eq!(ours, orig);
        }
    }
}
