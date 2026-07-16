use std::{cmp::Ordering, collections::HashMap, ffi::c_void};

use arrow_array::{
    cast::AsArray,
    types::{
        Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
        UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    BooleanArray, Datum,
};
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    intrinsics::Intrinsic,
    llvm_sys::{
        LLVMFastMathAllowContract, LLVMFastMathAllowReassoc, LLVMFastMathAllowReciprocal,
        LLVMFastMathApproxFunc,
    },
    module::{Linkage, Module},
    types::{BasicType, BasicTypeEnum},
    values::{BasicValue, BasicValueEnum, FunctionValue, IntValue, VectorValue},
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
};
use itertools::Itertools;
use ouroboros::self_referencing;

use crate::{
    compiled_iter::{array_to_setbit_iter, datum_to_iter, get_iterator_length, IteratorHolder},
    compiled_kernels::{
        cmp::{add_float_to_int, add_float_vec_to_int_vec, add_memcmp},
        dsl2::{
            add_str_endswith, add_str_startswith, buffer::DSLBuffer, runtime::RunnableDSLFunction,
            two_d, vectorize, writers::accepted_type, DSLArgument, DSLArgumentType, DSLArithBinOp,
            DSLBitwiseBinOp, DSLComparison, DSLExpr, DSLFunction, DSLStmt, DSLStringPredicate,
            DSLType, DSLValue,
        },
        link_req_helpers,
        llvm_utils::llvm_add_save_ptrs_string_saver,
        optimize_module,
    },
    compiled_writers::{BoundWriter, WriterSpec},
    increment_pointer, set_noalias_params, ArrowKernelError, ComparisonType, ListItemType,
    NumericPrimitiveType, PrimitiveType,
};

pub enum KernelReturnCode {
    Success,
    InvalidEmitIndex,
    EmptyReduction,
    Unknown(u64),
}

impl From<KernelReturnCode> for u64 {
    fn from(code: KernelReturnCode) -> Self {
        match code {
            KernelReturnCode::Success => 0,
            KernelReturnCode::InvalidEmitIndex => 1,
            KernelReturnCode::EmptyReduction => 2,
            KernelReturnCode::Unknown(code) => code,
        }
    }
}

impl From<u64> for KernelReturnCode {
    fn from(code: u64) -> Self {
        match code {
            0 => KernelReturnCode::Success,
            1 => KernelReturnCode::InvalidEmitIndex,
            2 => KernelReturnCode::EmptyReduction,
            code => KernelReturnCode::Unknown(code),
        }
    }
}

fn dsl_type_to_llvm_type<'a>(ctx: &'a Context, v: &DSLType) -> BasicTypeEnum<'a> {
    let ptr_type = ctx.ptr_type(AddressSpace::default()).as_basic_type_enum();
    match v {
        DSLType::Boolean => ctx.bool_type().as_basic_type_enum(),
        DSLType::Primitive(pt) => pt.llvm_type(ctx),
        DSLType::ConstScalar(datum) => {
            let dt = datum.get().0.data_type();
            match dt {
                DataType::Boolean => ctx.bool_type().as_basic_type_enum(),
                _ => PrimitiveType::for_arrow_type(dt).llvm_type(ctx),
            }
        }
        DSLType::VarList(_) => v.llvm_type(ctx).unwrap(),
        DSLType::Buffer(..)
        | DSLType::Scalar(..)
        | DSLType::SetBits(..)
        | DSLType::TwoDArray(..)
        | DSLType::Array(..)
        | DSLType::StringSaver => ptr_type,
    }
}

fn build_entry_alloca<'ctx>(
    ctx: &DSLCompilationContext<'ctx, '_>,
    ty: BasicTypeEnum<'ctx>,
    name: &str,
) -> inkwell::values::PointerValue<'ctx> {
    let builder = ctx.ctx.create_builder();
    let entry = ctx.func.get_first_basic_block().unwrap();
    if let Some(first_instr) = entry.get_first_instruction() {
        builder.position_before(&first_instr);
    } else {
        builder.position_at_end(entry);
    }
    builder.build_alloca(ty, name).unwrap()
}

pub struct DSLCompilationContext<'ctx, 'a> {
    pub ctx: &'ctx Context,
    pub module: &'a Module<'ctx>,
    pub func: &'a FunctionValue<'ctx>,
    pub b: &'a Builder<'ctx>,
    /// Maps DSL value IDs to their current LLVM values.
    pub st: HashMap<usize, BasicValueEnum<'ctx>>,
    /// Maps iterable argument IDs to the iterator holders that back them.
    pub iterator_holders: HashMap<usize, IteratorHolder>,
    /// Maps DSL value IDs to the array argument IDs they came from.
    pub value_sources: HashMap<usize, usize>,
    /// Maps randomly accessed value IDs to their generated accessor functions.
    pub access_funcs: HashMap<usize, FunctionValue<'ctx>>,
    /// Maps writable buffer IDs to their generated writer functions.
    pub writer_funcs: HashMap<usize, FunctionValue<'ctx>>,
    /// Maps iterable value IDs to their LLVM lengths, when known.
    pub lengths: HashMap<usize, Option<IntValue<'ctx>>>,
    /// Describes each output writer in return-value order.
    pub output_specs: Vec<WriterSpec>,
    /// Holds each initialized output writer in return-value order.
    pub outputs: Vec<BoundWriter<'ctx>>,
    /// Records whether compilation emitted a vectorized loop.
    pub did_vectorize: &'a mut bool,
}

#[self_referencing]
pub struct CompiledDSLFunction {
    pub ctx: Context,
    pub f: DSLFunction,
    pub arg_types: Vec<DSLArgumentType>,

    #[borrows(ctx, f)]
    #[covariant]
    pub compiled: JitFunction<'this, unsafe extern "C" fn(*mut c_void) -> u64>,
}

pub fn compile<'a>(
    f: DSLFunction,
    args: impl IntoIterator<Item = DSLArgument<'a>>,
) -> Result<RunnableDSLFunction, ArrowKernelError> {
    let ctx = Context::create();
    let args = args.into_iter().collect_vec();
    let arg_types = args
        .iter()
        .zip(f.params.iter().map(|p| &p.ty))
        .map(|(arg, dsl_type)| arg.get_type(dsl_type))
        .collect_vec();

    let mut did_vectorize = false;
    let func = CompiledDSLFunctionTryBuilder {
        ctx,
        f,
        arg_types,
        compiled_builder: |ctx, f| compile_inner(ctx, f, args, &mut did_vectorize),
    }
    .try_build()?;
    RunnableDSLFunction::new(func, did_vectorize)
}

pub fn compile_inner<'ctx, 'args>(
    ctx: &'ctx Context,
    f: &DSLFunction,
    args: impl IntoIterator<Item = DSLArgument<'args>>,
    did_vectorize: &mut bool,
) -> Result<JitFunction<'ctx, unsafe extern "C" fn(*mut c_void) -> u64>, ArrowKernelError> {
    let module = ctx.create_module("dsl2");
    let args: Vec<_> = args.into_iter().collect();

    // validate parameters
    assert_eq!(
        args.len(),
        f.params.len(),
        "number of arguments ({}) does not match number of parameters ({})",
        args.len(),
        f.params.len(),
    );
    for (i, (arg, param)) in args.iter().zip(f.params.iter()).enumerate() {
        if !arg.is_compatible_with(&param.ty) {
            return Err(ArrowKernelError::ArgumentTypeMismatch(
                i,
                format!("{:?}", arg),
                param.ty.clone(),
            ));
        }
    }

    let names_to_args: HashMap<usize, _> = args
        .into_iter()
        .zip(f.params.iter())
        .map(|(a, p)| (p.name, a))
        .collect();
    let names_to_params: HashMap<usize, &DSLValue> = f.params.iter().map(|p| (p.name, p)).collect();

    let func = {
        let llvm_params = f
            .params
            .iter()
            .map(|arg| dsl_type_to_llvm_type(ctx, &arg.ty))
            .chain(std::iter::repeat_n(
                ctx.ptr_type(AddressSpace::default()).into(),
                f.ret.len(),
            ))
            .map(|x| x.into())
            .collect_vec();
        module.add_function(
            &f.name,
            ctx.i64_type().fn_type(&llvm_params, false),
            Some(Linkage::Private),
        )
    };
    set_noalias_params(&func);

    let mut st = HashMap::new();
    let mut access_funcs = HashMap::new();
    let mut writer_funcs = HashMap::new();
    let mut ihs = HashMap::new();
    let mut lengths = HashMap::new();

    // load LLVM function parameters into the symbol table
    for (idx, param) in f.params.iter().enumerate() {
        st.insert(param.name, func.get_nth_param(idx as u32).unwrap());
    }

    // first, initialize all iterators, getters, and outputs
    let init_block = ctx.append_basic_block(func, "init");
    let b = ctx.create_builder();
    b.position_at_end(init_block);

    // create all iterator holders
    for param in f.params.iter() {
        let arg = &names_to_args[&param.name];
        match &param.ty {
            DSLType::Scalar(_) | DSLType::Array(_, _) => {
                let ih = datum_to_iter(
                    arg.as_datum()
                        .ok_or_else(|| ArrowKernelError::NonIterableType(param.ty.clone()))?,
                )?;
                let ptr = ih.localize_struct(ctx, &b, st[&param.name].into_pointer_value());
                st.insert(param.name, ptr.as_basic_value_enum());
                ihs.insert(param.name, ih);
            }
            DSLType::SetBits(_) => {
                if arg.data_type() != DataType::Boolean {
                    return Err(ArrowKernelError::InvalidSetBitType(arg.data_type()));
                }

                let ih = array_to_setbit_iter(&BooleanArray::new_null(0))?;
                let ptr = ih.localize_struct(ctx, &b, st[&param.name].into_pointer_value());
                st.insert(param.name, ptr.as_basic_value_enum());
                ihs.insert(param.name, ih);
            }
            _ => {}
        }
    }

    // setup accessors (random access)
    for idx in f.accessed_parameters() {
        let param = names_to_params[&idx];
        let arg = &names_to_args[&idx];

        match &param.ty {
            DSLType::Scalar(t) | DSLType::Array(t, _) => {
                let ih = &ihs[&idx];
                let detected_dtype = DSLType::of_iterator_holder(ih).iter_type().unwrap();
                if &detected_dtype != t.as_ref() {
                    return Err(ArrowKernelError::DSLTypeMismatch(
                        "accessed parameters",
                        *t.clone(),
                        detected_dtype,
                    ));
                }

                let access_func = ih.generate_random_access(ctx, &module).unwrap();
                access_funcs.insert(param.name, access_func);
            }
            DSLType::Buffer(t, _) => {
                access_funcs.insert(
                    param.name,
                    DSLBuffer::generate_buffer_accessor(ctx, &module, *t),
                );
                writer_funcs.insert(
                    param.name,
                    DSLBuffer::generate_buffer_writer(ctx, &module, *t),
                );
            }
            DSLType::TwoDArray(..) => {
                access_funcs.insert(
                    param.name,
                    two_d::generate_two_d_access(ctx, &module, arg.as_two_d().unwrap())?,
                );
            }
            _ => return Err(ArrowKernelError::NonRandomAccessType(param.ty.clone())),
        };
    }

    // setup iterators
    for idx in f.iterated_parameters() {
        let param = names_to_params[&idx];

        match &param.ty {
            DSLType::Scalar(t) | DSLType::Array(t, ..) => {
                let ih = &ihs[&idx];
                let detected_dtype = DSLType::of_iterator_holder(ih).iter_type().unwrap();
                if &detected_dtype != t.as_ref() {
                    return Err(ArrowKernelError::DSLTypeMismatch(
                        "iterated parameters",
                        *t.clone(),
                        detected_dtype,
                    ));
                }
            }
            DSLType::SetBits(_) => {}
            _ => return Err(ArrowKernelError::NonIterableType(param.ty.clone())),
        };
    }

    // compute all lengths
    for param in f.params.iter() {
        if let Some(ih) = ihs.get(&param.name) {
            let length = get_iterator_length(ctx, &b, ih, st[&param.name].into_pointer_value());
            lengths.insert(param.name, length);
        } else if param.ty.is_buffer() {
            let len = DSLBuffer::buffer_len(ctx, &b, st[&param.name].into_pointer_value());
            lengths.insert(param.name, Some(len));
        }
    }

    // figure out which iterated values have related lengths
    let mut length_groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for idx in f.iterated_parameters() {
        let param = &names_to_params[&idx];
        if let Some((_ty, n)) = param.ty.as_array() {
            length_groups.entry(n).or_default().push(param.name);
        }
    }

    for (_, group) in length_groups.into_iter() {
        if group.len() <= 1 {
            continue;
        }

        let lengths = group.iter().filter_map(|idx| lengths[idx]).collect_vec();
        if lengths.len() <= 1 {
            continue;
        }

        let assume = Intrinsic::find("llvm.assume").unwrap();
        let assume = assume.get_declaration(&module, &[]).unwrap();

        for length in lengths[1..].iter() {
            let v = b
                .build_int_compare(IntPredicate::EQ, lengths[0], *length, "assumed_length_eq")
                .unwrap();
            b.build_call(assume, &[v.into()], "assume").unwrap();
        }
    }

    // setup outputs
    let mut output_specs = Vec::new();
    let mut writers = Vec::new();
    let output_param_offset = func.count_params() as usize - f.ret.len();
    for (idx, ret) in f.ret.iter().enumerate() {
        let os = ret.allocate(0);
        output_specs.push(ret.spec().clone());
        let w = os.llvm_init(
            ctx,
            &module,
            &b,
            func.get_nth_param((output_param_offset + idx) as u32)
                .unwrap()
                .into_pointer_value(),
        );
        writers.push(w);
    }

    let value_sources = f
        .params
        .iter()
        .map(|param| (param.name, param.name))
        .collect();

    let mut dsl_ctx = DSLCompilationContext {
        ctx,
        module: &module,
        func: &func,
        b: &b,
        st,
        iterator_holders: ihs,
        value_sources,
        access_funcs,
        writer_funcs,
        output_specs,
        lengths,
        outputs: writers,
        did_vectorize,
    };

    for stmt in f.body.iter() {
        compile_stmt(&mut dsl_ctx, stmt)?;
    }

    b.build_return(Some(&ctx.i64_type().const_zero())).unwrap();

    // add the wrapper function so we can extract a function with a consistent sig
    let ptr_ty = ctx.ptr_type(AddressSpace::default());
    let wrapper_fn = module.add_function(
        "wrapped",
        ctx.i64_type().fn_type(&[ptr_ty.into()], false),
        None,
    );
    let entry = ctx.append_basic_block(wrapper_fn, "entry");
    b.position_at_end(entry);
    let base_ptr = wrapper_fn.get_nth_param(0).unwrap().into_pointer_value();
    let args = func
        .get_param_iter()
        .enumerate()
        .map(|(i, p)| {
            assert!(p.get_type().is_pointer_type());
            b.build_load(
                ptr_ty,
                increment_pointer!(ctx, b, base_ptr, 8 * i),
                &format!("load_arg_{}", i),
            )
            .unwrap()
            .into()
        })
        .collect_vec();
    let result = b
        .build_call(func, &args, "call_inner")
        .unwrap()
        .try_as_basic_value()
        .unwrap_basic();
    b.build_return(Some(&result)).unwrap();

    module.verify().unwrap();
    optimize_module(&module)?;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();
    link_req_helpers(&module, &ee).unwrap();

    let f = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void) -> u64>(
            wrapper_fn.get_name().to_str().unwrap(),
        )
        .unwrap()
    };

    Ok(f)
}

fn compile_stmt<'ctx, 'a>(
    ctx: &mut DSLCompilationContext<'ctx, 'a>,
    stmt: &DSLStmt,
) -> Result<(), ArrowKernelError> {
    match stmt {
        DSLStmt::Emit { index, value } => {
            if ctx.outputs.is_empty() {
                return Err(ArrowKernelError::EmitWithNoOutputs);
            }

            if let Some(idx) = index.as_u32() {
                if idx as usize >= ctx.outputs.len() {
                    return Err(ArrowKernelError::InvalidIndex("emit index out of range"));
                }
                let accepted = accepted_type(&ctx.output_specs[idx as usize]);
                if value.get_type() != accepted {
                    return Err(ArrowKernelError::DSLTypeMismatch(
                        "emit",
                        accepted,
                        value.get_type(),
                    ));
                }
                let compiled_value = compile_expr(ctx, value)?;
                if matches!(accepted, DSLType::VarList(_)) {
                    let source_name = match value {
                        DSLExpr::At(value, _) => ctx.value_sources[&value.name],
                        DSLExpr::Value(value) => ctx.value_sources[&value.name],
                        _ => return Err(ArrowKernelError::InvalidAtSource(value.clone())),
                    };
                    ctx.outputs[idx as usize].llvm_ingest_from_iterator(
                        ctx.ctx,
                        ctx.module,
                        ctx.b,
                        &ctx.iterator_holders[&source_name],
                        compiled_value,
                    );
                } else {
                    ctx.outputs[idx as usize].llvm_ingest(
                        ctx.ctx,
                        ctx.module,
                        ctx.b,
                        compiled_value,
                    );
                }
                return Ok(());
            } else {
                if matches!(value.get_type(), DSLType::VarList(_)) {
                    return Err(ArrowKernelError::DSLInvalidType(
                        "dynamic emit does not support variable-size lists",
                        value.get_type(),
                    ));
                }
                for spec in ctx.output_specs.iter() {
                    let accepted = accepted_type(spec);
                    if value.get_type() != accepted {
                        return Err(ArrowKernelError::DSLTypeMismatch(
                            "dynamic emit",
                            accepted,
                            value.get_type(),
                        ));
                    }
                }
            }

            let index = compile_expr(ctx, index)?;
            let value = compile_expr(ctx, value)?;

            let emit_worked = ctx.ctx.append_basic_block(*ctx.func, "emit_worked");
            let emit_failed = ctx.ctx.append_basic_block(*ctx.func, "emit_failed");
            let switch_blocks = ctx
                .outputs
                .iter()
                .enumerate()
                .map(|(idx, o)| {
                    let emit_block = ctx
                        .ctx
                        .append_basic_block(*ctx.func, &format!("emit_block_{}", idx));
                    let b = ctx.ctx.create_builder();
                    b.position_at_end(emit_block);
                    o.llvm_ingest(ctx.ctx, ctx.module, &b, value);
                    b.build_unconditional_branch(emit_worked).unwrap();

                    (ctx.ctx.i32_type().const_int(idx as u64, false), emit_block)
                })
                .collect_vec();

            ctx.b
                .build_switch(index.into_int_value(), emit_failed, &switch_blocks)
                .unwrap();
            ctx.b.position_at_end(emit_failed);
            ctx.b
                .build_return(Some(
                    &ctx.ctx
                        .i64_type()
                        .const_int(KernelReturnCode::InvalidEmitIndex.into(), false),
                ))
                .unwrap();
            ctx.b.position_at_end(emit_worked);
        }
        DSLStmt::EmitBlock { index, value } => {
            let index = index.as_u32().ok_or_else(|| {
                ArrowKernelError::DSLTypeMismatch(
                    "block emit requires const index",
                    DSLType::scalar_of(PrimitiveType::U32),
                    index.get_type(),
                )
            })?;

            let value = compile_expr(ctx, value)?;
            ctx.outputs[index as usize].llvm_ingest_block(
                ctx.ctx,
                ctx.module,
                ctx.b,
                value.into_vector_value(),
            );
        }
        DSLStmt::Set {
            buf,
            index,
            value,
            saver,
        } => {
            if !matches!(value.get_type(), DSLType::Primitive(_) | DSLType::Boolean) {
                return Err(ArrowKernelError::DSLInvalidType(
                    "values in set statements must be primitive or boolean",
                    value.get_type(),
                ));
            }

            if !buf.ty.is_buffer() {
                return Err(ArrowKernelError::DSLTypeMismatch(
                    "set must be on a buffer",
                    DSLType::Buffer(value.get_type().as_primitive().unwrap(), "".into()),
                    buf.ty.clone(),
                ));
            }
            let index = compile_expr(ctx, index)?;
            let mut value_llvm = compile_expr(ctx, value)?;
            // upcast boolean values to i8
            if value_llvm.get_type() == ctx.ctx.bool_type().as_basic_type_enum() {
                value_llvm = ctx
                    .b
                    .build_int_z_extend(value_llvm.into_int_value(), ctx.ctx.i8_type(), "ext_bool")
                    .unwrap()
                    .as_basic_value_enum();
            } else if let Some(saver) = saver.as_ref() {
                if value.get_type() == DSLType::Primitive(PrimitiveType::P64x2) {
                    let save = llvm_add_save_ptrs_string_saver(ctx.ctx, ctx.module);

                    let ptr1 = ctx
                        .b
                        .build_extract_value(value_llvm.into_struct_value(), 0, "ptr1")
                        .unwrap();
                    let ptr2 = ctx
                        .b
                        .build_extract_value(value_llvm.into_struct_value(), 1, "ptr2")
                        .unwrap();
                    value_llvm = ctx
                        .b
                        .build_call(
                            save,
                            &[ptr1.into(), ptr2.into(), ctx.st[&saver.name].into()],
                            "save",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic();
                }
            }
            let func = ctx.writer_funcs[&buf.name];

            ctx.b
                .build_call(
                    func,
                    &[ctx.st[&buf.name].into(), index.into(), value_llvm.into()],
                    "set",
                )
                .unwrap();
        }
        DSLStmt::If { cond, then, else_ } => {
            let cond = compile_expr(ctx, cond)?;

            let if_path = ctx.ctx.append_basic_block(*ctx.func, "if_path");
            let else_path = ctx.ctx.append_basic_block(*ctx.func, "else_path");
            let after_if = ctx.ctx.append_basic_block(*ctx.func, "after_if");

            ctx.b
                .build_conditional_branch(cond.into_int_value(), if_path, else_path)
                .unwrap();

            ctx.b.position_at_end(if_path);
            for stmt in then {
                compile_stmt(ctx, stmt)?;
            }
            ctx.b.build_unconditional_branch(after_if).unwrap();

            ctx.b.position_at_end(else_path);
            for stmt in else_ {
                compile_stmt(ctx, stmt)?;
            }
            ctx.b.build_unconditional_branch(after_if).unwrap();

            ctx.b.position_at_end(after_if);
        }
        DSLStmt::ForEach(floop) => {
            // attempt to vectorize
            if let Some(floop) = vectorize::vectorize_for_each(ctx, floop) {
                compile_stmt(ctx, &DSLStmt::ForEachBlock(floop))?;
                // continue below to generate the tail loop
            }

            let bufs = floop
                .loop_vars
                .iter()
                .map(|iter| iter.ty.llvm_type(ctx.ctx).unwrap())
                .enumerate()
                .map(|(i, llvm_type)| build_entry_alloca(ctx, llvm_type, &format!("iter_buf{}", i)))
                .collect_vec();

            let loop_header = ctx.ctx.append_basic_block(*ctx.func, "loop_header");
            let loop_body = ctx.ctx.append_basic_block(*ctx.func, "loop_body");
            let loop_end = ctx.ctx.append_basic_block(*ctx.func, "loop_end");

            ctx.b.build_unconditional_branch(loop_header).unwrap();
            // in the loop header, call next on all iterators and `and` together
            // the results
            ctx.b.position_at_end(loop_header);

            let mut last_res = None;
            for (iter, buf_ptr) in floop.iterators.iter().zip(bufs.iter()) {
                let next_func = ctx.iterator_holders[&iter.name].generate_next(ctx.ctx, ctx.module);
                let iter_ptr = ctx.st[&iter.name];
                let had_next = ctx
                    .b
                    .build_call(
                        next_func,
                        &[iter_ptr.into(), (*buf_ptr).into()],
                        &format!("next{}", iter.name),
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_int_value();

                last_res = last_res
                    .map(|v| ctx.b.build_and(v, had_next, "and").unwrap())
                    .or(Some(had_next))
            }
            ctx.b
                .build_conditional_branch(last_res.unwrap(), loop_body, loop_end)
                .unwrap();

            ctx.b.position_at_end(loop_body);
            for (loop_var, buf_ptr) in floop.loop_vars.iter().zip(bufs) {
                let val = ctx
                    .b
                    .build_load(
                        loop_var.ty.llvm_type(ctx.ctx).unwrap(),
                        buf_ptr,
                        &format!("load{}", loop_var.name),
                    )
                    .unwrap();

                // truncate to boolean if neeeded
                let val = match loop_var.ty {
                    DSLType::Boolean => ctx
                        .b
                        .build_int_truncate(val.into_int_value(), ctx.ctx.bool_type(), "to_bool")
                        .unwrap()
                        .as_basic_value_enum(),
                    _ => val,
                };

                ctx.st.insert(loop_var.name, val);
            }
            for (loop_var, iterator) in floop.loop_vars.iter().zip(floop.iterators.iter()) {
                let source = ctx.value_sources[&iterator.name];
                ctx.value_sources.insert(loop_var.name, source);
            }
            for stmt in floop.body.iter() {
                compile_stmt(ctx, stmt)?;
            }

            // if every loop variable is a scalar, just do one iteration
            if floop.iterators.iter().all(|it| it.ty.is_infinite()) {
                ctx.b.build_unconditional_branch(loop_end).unwrap();
            } else {
                ctx.b.build_unconditional_branch(loop_header).unwrap();
            }

            ctx.b.position_at_end(loop_end);
            floop.iterators.iter().for_each(|itr| {
                let reset = ctx.iterator_holders[&itr.name].generate_reset(ctx.ctx, ctx.module);
                ctx.b
                    .build_call(reset, &[ctx.st[&itr.name].into()], "reset")
                    .unwrap();
            });
        }
        DSLStmt::ForEachBlock(bfloop) => {
            *ctx.did_vectorize = true;

            let bufs = bfloop
                .loop_vars
                .iter()
                .map(|iter| iter.ty.llvm_type(ctx.ctx).unwrap())
                .enumerate()
                .map(|(i, llvm_type)| build_entry_alloca(ctx, llvm_type, &format!("iter_buf{}", i)))
                .collect_vec();

            let loop_header = ctx.ctx.append_basic_block(*ctx.func, "bloop_header");
            let loop_body = ctx.ctx.append_basic_block(*ctx.func, "bloop_body");
            let loop_end = ctx.ctx.append_basic_block(*ctx.func, "bloop_end");

            ctx.b.build_unconditional_branch(loop_header).unwrap();
            // in the loop header, call next on all iterators and `and` together
            // the results
            ctx.b.position_at_end(loop_header);

            let mut last_res = None;
            for (iter, buf_ptr) in bfloop.iterators.iter().zip(bufs.iter()) {
                let next_func = ctx.iterator_holders[&iter.name]
                    .generate_next_block::<64>(ctx.ctx, ctx.module)
                    .expect("vectorized loop requires block-readable iterators");
                let iter_ptr = ctx.st[&iter.name];
                let had_next = ctx
                    .b
                    .build_call(
                        next_func,
                        &[iter_ptr.into(), (*buf_ptr).into()],
                        &format!("next_block{}", iter.name),
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_int_value();

                last_res = last_res
                    .map(|v| ctx.b.build_and(v, had_next, "and").unwrap())
                    .or(Some(had_next))
            }
            ctx.b
                .build_conditional_branch(last_res.unwrap(), loop_body, loop_end)
                .unwrap();

            ctx.b.position_at_end(loop_body);
            for (loop_var, buf_ptr) in bfloop.loop_vars.iter().zip(bufs) {
                let val = ctx
                    .b
                    .build_load(
                        loop_var.ty.llvm_type(ctx.ctx).unwrap(),
                        buf_ptr,
                        &format!("load{}", loop_var.name),
                    )
                    .unwrap();

                ctx.st.insert(loop_var.name, val);
            }

            for stmt in bfloop.body.iter() {
                compile_stmt(ctx, stmt)?;
            }

            // if every loop variable is a scalar, just do one iteration
            if bfloop.iterators.iter().all(|it| it.ty.is_infinite()) {
                ctx.b.build_unconditional_branch(loop_end).unwrap();
            } else {
                ctx.b.build_unconditional_branch(loop_header).unwrap();
            }
            ctx.b.position_at_end(loop_end);
        }
        DSLStmt::Reduce(dslred) => {
            let body_type = dslred.body.get_type();
            let accum_type = dslred.reduction_type.accum_type(ctx.ctx, &body_type);

            let bufs = dslred
                .loop_vars
                .iter()
                .map(|iter| iter.ty.llvm_type(ctx.ctx).unwrap())
                .enumerate()
                .map(|(i, llvm_type)| build_entry_alloca(ctx, llvm_type, &format!("iter_buf{}", i)))
                .collect_vec();

            let loop_header = ctx.ctx.append_basic_block(*ctx.func, "reduce_loop_header");
            let loop_body = ctx.ctx.append_basic_block(*ctx.func, "reduce_loop_body");
            let loop_end = ctx.ctx.append_basic_block(*ctx.func, "reduce_loop_end");
            let accum_ptr = build_entry_alloca(ctx, accum_type, "accum");
            ctx.b
                .build_store(
                    accum_ptr,
                    dslred.reduction_type.initial_value(ctx.ctx, &body_type),
                )
                .unwrap();

            ctx.b.build_unconditional_branch(loop_header).unwrap();

            // in the loop header, call next on all iterators and `and` together
            // the results
            ctx.b.position_at_end(loop_header);

            let mut last_res = None;
            for (iter, buf_ptr) in dslred.iterators.iter().zip(bufs.iter()) {
                let next_func = ctx.iterator_holders[&iter.name].generate_next(ctx.ctx, ctx.module);
                let iter_ptr = ctx.st[&iter.name];
                let had_next = ctx
                    .b
                    .build_call(
                        next_func,
                        &[iter_ptr.into(), (*buf_ptr).into()],
                        &format!("next{}", iter.name),
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_int_value();

                last_res = last_res
                    .map(|v| ctx.b.build_and(v, had_next, "and").unwrap())
                    .or(Some(had_next))
            }
            ctx.b
                .build_conditional_branch(last_res.unwrap(), loop_body, loop_end)
                .unwrap();

            ctx.b.position_at_end(loop_body);
            for (loop_var, buf_ptr) in dslred.loop_vars.iter().zip(bufs) {
                let val = ctx
                    .b
                    .build_load(
                        loop_var.ty.llvm_type(ctx.ctx).unwrap(),
                        buf_ptr,
                        &format!("load{}", loop_var.name),
                    )
                    .unwrap();

                // truncate to boolean if neeeded
                let val = match loop_var.ty {
                    DSLType::Boolean => ctx
                        .b
                        .build_int_truncate(val.into_int_value(), ctx.ctx.bool_type(), "to_bool")
                        .unwrap()
                        .as_basic_value_enum(),
                    _ => val,
                };

                ctx.st.insert(loop_var.name, val);
            }

            let next = compile_expr(ctx, &dslred.body)?;
            let include = if let Some(include) = &dslred.include {
                compile_expr(ctx, include)?.into_int_value()
            } else {
                ctx.ctx.bool_type().const_all_ones()
            };
            let accum = ctx.b.build_load(accum_type, accum_ptr, "accum").unwrap();
            let new = dslred
                .reduction_type
                .update(ctx, &body_type, accum, next, include)?;
            ctx.b.build_store(accum_ptr, new).unwrap();

            // if every loop variable is a scalar, just do one iteration
            if dslred.iterators.iter().all(|it| it.ty.is_infinite()) {
                ctx.b.build_unconditional_branch(loop_end).unwrap();
            } else {
                ctx.b.build_unconditional_branch(loop_header).unwrap();
            }

            ctx.b.position_at_end(loop_end);
            let result = ctx
                .b
                .build_load(accum_type, accum_ptr, "reduce_result")
                .unwrap();
            let result = dslred.reduction_type.output_value(ctx, result)?;
            ctx.st.insert(dslred.result.name, result);
            dslred.iterators.iter().for_each(|itr| {
                let reset = ctx.iterator_holders[&itr.name].generate_reset(ctx.ctx, ctx.module);
                ctx.b
                    .build_call(reset, &[ctx.st[&itr.name].into()], "reset")
                    .unwrap();
            });
        }
        DSLStmt::ForRange(dslfor) => {
            let start_at = compile_expr(ctx, &dslfor.start)?;
            let end_at = compile_expr(ctx, &dslfor.end)?.into_int_value();

            let counter_ptr = build_entry_alloca(ctx, ctx.ctx.i64_type().into(), "loop_counter");
            ctx.b.build_store(counter_ptr, start_at).unwrap();

            let loop_header = ctx.ctx.append_basic_block(*ctx.func, "loop_header");
            let loop_body = ctx.ctx.append_basic_block(*ctx.func, "loop_body");
            let loop_end = ctx.ctx.append_basic_block(*ctx.func, "loop_end");

            ctx.b.build_unconditional_branch(loop_header).unwrap();
            ctx.b.position_at_end(loop_header);
            let curr_counter = ctx
                .b
                .build_load(ctx.ctx.i64_type(), counter_ptr, "curr_counter")
                .unwrap()
                .into_int_value();
            let cond = ctx
                .b
                .build_int_compare(IntPredicate::ULT, curr_counter, end_at, "loop_cond")
                .unwrap();

            ctx.b
                .build_conditional_branch(cond, loop_body, loop_end)
                .unwrap();

            ctx.b.position_at_end(loop_body);
            let curr_counter = ctx
                .b
                .build_load(ctx.ctx.i64_type(), counter_ptr, "curr_counter")
                .unwrap()
                .into_int_value();
            ctx.st
                .insert(dslfor.loop_var.name, curr_counter.as_basic_value_enum());

            for stmt in dslfor.body.iter() {
                compile_stmt(ctx, stmt)?;
            }

            let next_counter = ctx
                .b
                .build_int_add(
                    curr_counter,
                    ctx.ctx.i64_type().const_int(1, false),
                    "new_counter",
                )
                .unwrap();
            ctx.b.build_store(counter_ptr, next_counter).unwrap();
            ctx.b.build_unconditional_branch(loop_header).unwrap();

            ctx.b.position_at_end(loop_end);
        }
    }
    Ok(())
}

pub fn compile_compare_values<'ctx, 'a>(
    ctx: &DSLCompilationContext<'ctx, 'a>,
    op: DSLComparison,
    lhs_type: &DSLType,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
) -> Result<BasicValueEnum<'ctx>, ArrowKernelError> {
    match lhs_type {
        DSLType::Boolean => Ok(ctx
            .b
            .build_int_compare(
                op.as_int_predicate(false),
                lhs.into_int_value(),
                rhs.into_int_value(),
                "cmp",
            )
            .unwrap()
            .as_basic_value_enum()),
        DSLType::Primitive(pt) => match pt.comparison_type() {
            ComparisonType::Float => {
                let cvt = add_float_to_int(ctx.ctx, ctx.module, *pt);
                let lhs = ctx
                    .b
                    .build_call(cvt, &[lhs.into()], "cvt_lhs_for_total_order")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic();
                let rhs = ctx
                    .b
                    .build_call(cvt, &[rhs.into()], "cvt_rhs_for_total_order")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic();
                Ok(ctx
                    .b
                    .build_int_compare(
                        op.as_int_predicate(true),
                        lhs.into_int_value(),
                        rhs.into_int_value(),
                        "cmp",
                    )
                    .unwrap()
                    .as_basic_value_enum())
            }
            ComparisonType::Int { signed } => Ok(ctx
                .b
                .build_int_compare(
                    op.as_int_predicate(signed),
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                    "cmp",
                )
                .unwrap()
                .as_basic_value_enum()),
            ComparisonType::String => {
                let memcmp = add_memcmp(ctx.ctx, ctx.module);
                let memcmp_res = ctx
                    .b
                    .build_call(memcmp, &[lhs.into(), rhs.into()], "memcmp_res")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_int_value();

                Ok(ctx
                    .b
                    .build_int_compare(
                        op.as_int_predicate(true),
                        memcmp_res,
                        ctx.ctx.i64_type().const_zero(),
                        "cmp",
                    )
                    .unwrap()
                    .as_basic_value_enum())
            }
            ComparisonType::List(_) => compare_list_values(ctx, op, *pt, lhs, rhs),
        },
        _ => Err(ArrowKernelError::DSLInvalidType(
            "comparison requires boolean or primitive values",
            lhs_type.clone(),
        )),
    }
}

fn compare_list_values<'ctx, 'a>(
    ctx: &DSLCompilationContext<'ctx, 'a>,
    op: DSLComparison,
    pt: PrimitiveType,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
) -> Result<BasicValueEnum<'ctx>, ArrowKernelError> {
    let PrimitiveType::List(item, size) = pt else {
        unreachable!("compare_list_values called with non-list type");
    };

    if item != ListItemType::Boolean && lhs.is_vector_value() && rhs.is_vector_value() {
        return match PrimitiveType::from(item).comparison_type() {
            ComparisonType::Int { signed } => Ok(ctx
                .b
                .build_int_compare(
                    op.as_int_predicate(signed),
                    lhs.into_vector_value(),
                    rhs.into_vector_value(),
                    "cmpv",
                )
                .unwrap()
                .as_basic_value_enum()),
            ComparisonType::Float => {
                let cvt = add_float_vec_to_int_vec(
                    ctx.ctx,
                    ctx.module,
                    size as u32,
                    pt.list_type_into_inner(),
                );
                let lhs = ctx
                    .b
                    .build_call(cvt, &[lhs.into()], "cvt_lhs_for_total_order")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic();
                let rhs = ctx
                    .b
                    .build_call(cvt, &[rhs.into()], "cvt_rhs_for_total_order")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic();
                Ok(ctx
                    .b
                    .build_int_compare(
                        op.as_int_predicate(true),
                        lhs.into_vector_value(),
                        rhs.into_vector_value(),
                        "cmpfv",
                    )
                    .unwrap()
                    .as_basic_value_enum())
            }
            ComparisonType::String | ComparisonType::List(_) => {
                Err(ArrowKernelError::DSLInvalidType(
                    "comparison not implemented for this vectorized list type",
                    DSLType::Primitive(pt),
                ))
            }
        };
    }

    let bool_type = ctx.ctx.bool_type();
    let mut all_equal = bool_type.const_all_ones();
    let mut any_less = bool_type.const_zero();
    let mut any_greater = bool_type.const_zero();

    for idx in 0..size {
        let (lhs_el, rhs_el, el_ty) = extract_list_compare_elements(ctx, item, idx, lhs, rhs);
        let eq = compile_compare_values(ctx, DSLComparison::Eq, &el_ty, lhs_el, rhs_el)?
            .into_int_value();
        let lt = compile_compare_values(ctx, DSLComparison::Lt, &el_ty, lhs_el, rhs_el)?
            .into_int_value();
        let gt = compile_compare_values(ctx, DSLComparison::Gt, &el_ty, lhs_el, rhs_el)?
            .into_int_value();

        let less_at_idx = ctx.b.build_and(all_equal, lt, "list_less_at_idx").unwrap();
        let greater_at_idx = ctx
            .b
            .build_and(all_equal, gt, "list_greater_at_idx")
            .unwrap();
        any_less = ctx
            .b
            .build_or(any_less, less_at_idx, "list_any_less")
            .unwrap();
        any_greater = ctx
            .b
            .build_or(any_greater, greater_at_idx, "list_any_greater")
            .unwrap();
        all_equal = ctx.b.build_and(all_equal, eq, "list_all_equal").unwrap();
    }

    let result = match op {
        DSLComparison::Eq => all_equal,
        DSLComparison::Neq => ctx.b.build_not(all_equal, "list_neq").unwrap(),
        DSLComparison::Lt => any_less,
        DSLComparison::Gt => any_greater,
        DSLComparison::Lte => ctx.b.build_or(any_less, all_equal, "list_lte").unwrap(),
        DSLComparison::Gte => ctx.b.build_or(any_greater, all_equal, "list_gte").unwrap(),
    };

    Ok(result.as_basic_value_enum())
}

fn extract_list_compare_elements<'ctx, 'a>(
    ctx: &DSLCompilationContext<'ctx, 'a>,
    item: ListItemType,
    idx: usize,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
) -> (BasicValueEnum<'ctx>, BasicValueEnum<'ctx>, DSLType) {
    let i = ctx.ctx.i64_type().const_int(idx as u64, false);
    match item {
        ListItemType::Boolean => {
            let lhs = ctx
                .b
                .build_extract_element(lhs.into_vector_value(), i, "lhs_bool")
                .unwrap();
            let rhs = ctx
                .b
                .build_extract_element(rhs.into_vector_value(), i, "rhs_bool")
                .unwrap();
            (lhs, rhs, DSLType::Boolean)
        }
        ListItemType::P64x2 => {
            let lhs = ctx
                .b
                .build_extract_value(lhs.into_array_value(), idx as u32, "lhs_str")
                .unwrap()
                .as_basic_value_enum();
            let rhs = ctx
                .b
                .build_extract_value(rhs.into_array_value(), idx as u32, "rhs_str")
                .unwrap()
                .as_basic_value_enum();
            (lhs, rhs, DSLType::Primitive(PrimitiveType::P64x2))
        }
        _ => {
            let lhs = ctx
                .b
                .build_extract_element(lhs.into_vector_value(), i, "lhs_lane")
                .unwrap()
                .as_basic_value_enum();
            let rhs = ctx
                .b
                .build_extract_element(rhs.into_vector_value(), i, "rhs_lane")
                .unwrap()
                .as_basic_value_enum();
            (lhs, rhs, DSLType::Primitive(PrimitiveType::from(item)))
        }
    }
}

fn compile_expr<'ctx, 'a>(
    ctx: &DSLCompilationContext<'ctx, 'a>,
    expr: &DSLExpr,
) -> Result<BasicValueEnum<'ctx>, ArrowKernelError> {
    match expr {
        DSLExpr::StringPredicate(pred, lhs, rhs) => {
            let lhs = compile_expr(ctx, lhs)?;
            let rhs = compile_expr(ctx, rhs)?;
            let predicate = match pred {
                DSLStringPredicate::StartsWith => add_str_startswith(ctx.ctx, ctx.module),
                DSLStringPredicate::EndsWith => add_str_endswith(ctx.ctx, ctx.module),
            };

            Ok(ctx
                .b
                .build_call(predicate, &[lhs.into(), rhs.into()], pred.as_str())
                .unwrap()
                .try_as_basic_value()
                .unwrap_basic())
        }
        DSLExpr::Compare(op, lhs, rhs) => {
            let lhs_type = lhs.get_type();
            let rhs_type = rhs.get_type();

            if lhs_type != rhs_type {
                return Err(ArrowKernelError::DSLTypeMismatch(
                    "comparison",
                    lhs_type,
                    rhs_type,
                ));
            }

            let lhs = compile_expr(ctx, lhs)?;
            let rhs = compile_expr(ctx, rhs)?;
            compile_compare_values(ctx, *op, &lhs_type, lhs, rhs)
        }
        DSLExpr::At(v, i) => {
            let res = match &v.ty {
                DSLType::Scalar(..) | DSLType::Array(..) | DSLType::Buffer(..) => {
                    if i.len() != 1 {
                        return Err(ArrowKernelError::InvalidIndex(
                            "non 2D-array should have a single index",
                        ));
                    }
                    let i = i.first().unwrap();
                    let access_func = ctx.access_funcs[&v.name];
                    let iter_ptr = ctx.st[&v.name];
                    if i.get_type() != DSLType::Primitive(PrimitiveType::U64) {
                        return Err(ArrowKernelError::DSLTypeMismatch(
                            "indexing requires unsigned 64 bit integers",
                            DSLType::Primitive(PrimitiveType::U64),
                            i.get_type(),
                        ));
                    }
                    let idx = compile_expr(ctx, i)?;
                    ctx.b
                        .build_call(
                            access_func,
                            &[iter_ptr.into(), idx.into()],
                            "access_by_index",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                }
                DSLType::TwoDArray(..) => {
                    if i.len() != 2 {
                        return Err(ArrowKernelError::InvalidIndex("2D array needs two indices"));
                    }

                    let access_func = ctx.access_funcs[&v.name];
                    let arr_ptr = ctx.st[&v.name];
                    for idx in i {
                        if idx.get_type() != DSLType::Primitive(PrimitiveType::U64) {
                            return Err(ArrowKernelError::DSLTypeMismatch(
                                "indexing requires unsigned 64 bit integers",
                                DSLType::Primitive(PrimitiveType::U64),
                                idx.get_type(),
                            ));
                        }
                    }

                    let idx0 = compile_expr(ctx, &i[0])?;
                    let idx1 = compile_expr(ctx, &i[1])?;
                    ctx.b
                        .build_call(
                            access_func,
                            &[arr_ptr.into(), idx0.into(), idx1.into()],
                            "access_2d_by_index",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic()
                }
                _ => todo!(),
            };

            if let Some(DSLType::Boolean) = v.ty.iter_type() {
                // truncate to boolean
                let res = ctx
                    .b
                    .build_int_truncate(res.into_int_value(), ctx.ctx.bool_type(), "trunc_to_bool")
                    .unwrap();
                Ok(res.into())
            } else {
                Ok(res)
            }
        }
        DSLExpr::Value(v) => match &v.ty {
            DSLType::ConstScalar(v) => {
                let v = v.get().0;
                match v.data_type() {
                    DataType::UInt64 => Ok(ctx
                        .ctx
                        .i64_type()
                        .const_int(v.as_primitive::<UInt64Type>().value(0), false)
                        .as_basic_value_enum()),
                    DataType::UInt32 => Ok(ctx
                        .ctx
                        .i32_type()
                        .const_int(v.as_primitive::<UInt32Type>().value(0) as u64, false)
                        .as_basic_value_enum()),
                    DataType::UInt16 => Ok(ctx
                        .ctx
                        .i16_type()
                        .const_int(v.as_primitive::<UInt16Type>().value(0) as u64, false)
                        .as_basic_value_enum()),
                    DataType::UInt8 => Ok(ctx
                        .ctx
                        .i8_type()
                        .const_int(v.as_primitive::<UInt8Type>().value(0) as u64, false)
                        .as_basic_value_enum()),
                    DataType::Float32 => Ok(ctx
                        .ctx
                        .f32_type()
                        .const_float(v.as_primitive::<Float32Type>().value(0) as f64)
                        .as_basic_value_enum()),
                    _ => todo!(),
                }
            }
            _ => Ok(ctx.st[&v.name]),
        },
        DSLExpr::Cast(val, tar_pt) => {
            let orig_type = val.get_type();
            let tar_type = DSLType::Primitive(*tar_pt);
            let tar_llvm_type = tar_pt.llvm_type(ctx.ctx);
            let val = compile_expr(ctx, val)?;

            match orig_type {
                DSLType::Boolean => {
                    if tar_pt.is_int() {
                        Ok(ctx
                            .b
                            .build_int_cast(
                                val.into_int_value(),
                                tar_llvm_type.into_int_type(),
                                "cast",
                            )
                            .unwrap()
                            .as_basic_value_enum())
                    } else {
                        unimplemented!("invalid cast")
                    }
                }
                DSLType::Primitive(orig_pt) => {
                    match (
                        orig_pt.as_numeric_primitive_type(),
                        tar_pt.as_numeric_primitive_type(),
                    ) {
                        (None, None) => {}
                        (None, Some(_)) | (Some(_), None) => {
                            return Err(ArrowKernelError::DSLTypeMismatch(
                                "cannot cast numeric type to non-numeric",
                                orig_type,
                                tar_type,
                            ))
                        }
                        (Some(orig_pt), Some(tar_pt)) => {
                            return cast_numeric(ctx, val, orig_pt, tar_pt)
                        }
                    }

                    // neither type is numeric
                    match (orig_pt, tar_pt) {
                        (PrimitiveType::List(s_t, s_l), &PrimitiveType::List(t_t, t_l)) => {
                            if s_l != t_l {
                                return Err(ArrowKernelError::DSLTypeMismatch(
                                    "cannot cast between vectors of different sizes",
                                    orig_type,
                                    tar_type,
                                ));
                            }
                            if !s_t.is_numeric() || !t_t.is_numeric() {
                                return Err(ArrowKernelError::DSLTypeMismatch(
                                    "can only cast fixed-size-lists with numeric child types",
                                    orig_type,
                                    tar_type,
                                ));
                            }

                            match (
                                PrimitiveType::from(s_t).as_numeric_primitive_type(),
                                PrimitiveType::from(t_t).as_numeric_primitive_type(),
                            ) {
                                (Some(orig_pt), Some(tar_pt)) => {
                                    return Ok(cast_numeric_vec(
                                        ctx,
                                        val.into_vector_value(),
                                        orig_pt,
                                        tar_pt,
                                    )?
                                    .as_basic_value_enum())
                                }
                                _ => Err(ArrowKernelError::DSLTypeMismatch(
                                    "cannot cast between string and non-string vectors",
                                    orig_type,
                                    tar_type,
                                )),
                            }
                        }
                        _ => Err(ArrowKernelError::DSLTypeMismatch(
                            "unsupported primitive cast",
                            orig_type,
                            tar_type,
                        )),
                    }
                }
                DSLType::ConstScalar(ref s) => {
                    let res = scalar_to_llvm(ctx, s.as_ref())?;
                    if res.get_type() != tar_llvm_type {
                        return Err(ArrowKernelError::DSLTypeMismatch(
                            "const scalar type mismatch",
                            orig_type,
                            tar_type,
                        ));
                    }
                    Ok(res)
                }
                _ => unimplemented!("invalid cast"),
            }
        }
        DSLExpr::CastToBool(v) => {
            let child = compile_expr(ctx, v)?;
            match v.get_type() {
                DSLType::Boolean => Ok(child),
                DSLType::Primitive(pt) => {
                    let nt = pt.as_numeric_primitive_type().ok_or_else(|| {
                        ArrowKernelError::DSLInvalidType(
                            "cannot cast non-numeric type to bool",
                            v.get_type(),
                        )
                    })?;

                    let res = if nt.is_integer() {
                        let child = child.into_int_value();
                        ctx.b
                            .build_int_compare(
                                IntPredicate::NE,
                                child,
                                child.get_type().const_zero(),
                                "ne_zero",
                            )
                            .unwrap()
                    } else {
                        let child = child.into_float_value();
                        ctx.b
                            .build_float_compare(
                                FloatPredicate::ONE,
                                child,
                                child.get_type().const_zero(),
                                "ne_zero",
                            )
                            .unwrap()
                    };
                    Ok(res.as_basic_value_enum())
                }
                _ => Err(ArrowKernelError::DSLInvalidType(
                    "cannot cast type to bool",
                    v.get_type(),
                )),
            }
        }
        DSLExpr::BitCast(v, tar_pt) => {
            let v = compile_expr(ctx, v)?;
            Ok(ctx
                .b
                .build_bit_cast(v, tar_pt.llvm_type(ctx.ctx), "bitcast")
                .unwrap())
        }
        DSLExpr::FloatToTotalOrderSInt(v) => {
            let pt = v.get_type().as_primitive().unwrap();
            let v = compile_expr(ctx, v)?;
            let cvt_f = add_float_to_int(ctx.ctx, ctx.module, pt);
            let res = ctx
                .b
                .build_call(cvt_f, &[v.into()], "float_to_total_order_sint")
                .unwrap()
                .try_as_basic_value()
                .unwrap_basic();
            Ok(res)
        }
        DSLExpr::Sqrt(v) => {
            let value = compile_expr(ctx, v)?;
            let decl_tys = match v.get_type().as_primitive().unwrap() {
                PrimitiveType::F16 => vec![ctx.ctx.f16_type().into()],
                PrimitiveType::F32 => vec![ctx.ctx.f32_type().into()],
                PrimitiveType::F64 => vec![ctx.ctx.f64_type().into()],
                PrimitiveType::List(item, size) => {
                    let inner = PrimitiveType::from(item);
                    let vec_ty = inner.llvm_vec_type(ctx.ctx, size as u32).unwrap();
                    vec![vec_ty.into()]
                }
                other => {
                    return Err(ArrowKernelError::DSLInvalidType(
                        "sqrt requires float inputs",
                        DSLType::Primitive(other),
                    ))
                }
            };
            let sqrt = Intrinsic::find("llvm.sqrt").unwrap();
            let sqrt = sqrt.get_declaration(ctx.module, &decl_tys).unwrap();
            let result = ctx
                .b
                .build_call(sqrt, &[value.into()], "sqrt")
                .unwrap()
                .try_as_basic_value()
                .unwrap_basic();
            result.as_instruction_value().unwrap().set_fast_math_flags(
                LLVMFastMathAllowContract
                    | LLVMFastMathAllowReassoc
                    | LLVMFastMathAllowReciprocal
                    | LLVMFastMathApproxFunc,
            );
            Ok(result)
        }
        DSLExpr::VecSum(v) => {
            let primitive = v.get_type().as_primitive().unwrap();
            let PrimitiveType::List(item, _) = primitive else {
                return Err(ArrowKernelError::DSLInvalidType(
                    "vec_sum requires a fixed-size-list value",
                    v.get_type(),
                ));
            };

            let inner = PrimitiveType::from(item);
            let value = compile_expr(ctx, v)?.into_vector_value();
            match inner {
                PrimitiveType::I8
                | PrimitiveType::I16
                | PrimitiveType::I32
                | PrimitiveType::I64
                | PrimitiveType::U8
                | PrimitiveType::U16
                | PrimitiveType::U32
                | PrimitiveType::U64 => {
                    let reducer = Intrinsic::find("llvm.vector.reduce.add").unwrap();
                    let reducer = reducer
                        .get_declaration(ctx.module, &[value.get_type().into()])
                        .unwrap();
                    Ok(ctx
                        .b
                        .build_call(reducer, &[value.into()], "vec_sum")
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic())
                }
                PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                    let reducer = Intrinsic::find("llvm.vector.reduce.fadd").unwrap();
                    let reducer = reducer
                        .get_declaration(ctx.module, &[value.get_type().into()])
                        .unwrap();
                    let result = ctx
                        .b
                        .build_call(
                            reducer,
                            &[inner.llvm_type(ctx.ctx).const_zero().into(), value.into()],
                            "vec_sum",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic();
                    result.as_instruction_value().unwrap().set_fast_math_flags(
                        LLVMFastMathAllowContract
                            | LLVMFastMathAllowReassoc
                            | LLVMFastMathAllowReciprocal
                            | LLVMFastMathApproxFunc,
                    );
                    Ok(result)
                }
                _ => Err(ArrowKernelError::DSLInvalidType(
                    "vec_sum requires numeric vector inputs",
                    v.get_type(),
                )),
            }
        }
        DSLExpr::ListLen(v) => match v.get_type() {
            DSLType::VarList(_) => {
                let row = compile_expr(ctx, v)?.into_struct_value();
                let start = ctx
                    .b
                    .build_extract_value(row, 1, "list_start")
                    .unwrap()
                    .into_int_value();
                let end = ctx
                    .b
                    .build_extract_value(row, 2, "list_end")
                    .unwrap()
                    .into_int_value();
                Ok(ctx
                    .b
                    .build_int_sub(end, start, "list_len")
                    .unwrap()
                    .as_basic_value_enum())
            }
            DSLType::Primitive(PrimitiveType::List(_, size)) => Ok(ctx
                .ctx
                .i64_type()
                .const_int(size as u64, false)
                .as_basic_value_enum()),
            _ => Err(ArrowKernelError::DSLInvalidType(
                "list_len requires a list row value",
                v.get_type(),
            )),
        },
        DSLExpr::Splat(v, size) => {
            let pt = v.get_type().as_primitive().unwrap();
            let vec_ty = pt.llvm_vec_type(ctx.ctx, *size as u32).ok_or_else(|| {
                ArrowKernelError::DSLInvalidType("cannot splat this primitive type", v.get_type())
            })?;
            let value = compile_expr(ctx, v)?;
            let singleton = ctx
                .b
                .build_insert_element(
                    vec_ty.const_zero(),
                    value,
                    ctx.ctx.i32_type().const_zero(),
                    "splat_insert",
                )
                .unwrap();
            Ok(ctx
                .b
                .build_shuffle_vector(singleton, vec_ty.get_poison(), vec_ty.const_zero(), "splat")
                .unwrap()
                .as_basic_value_enum())
        }
        DSLExpr::ArithBinOp(op, lhs, rhs) => {
            let lhs_v = compile_expr(ctx, lhs)?;
            let rhs_v = compile_expr(ctx, rhs)?;

            if lhs_v.get_type() != rhs_v.get_type() {
                return Err(ArrowKernelError::DSLTypeMismatch(
                    "arith operator",
                    lhs.get_type(),
                    rhs.get_type(),
                ));
            }

            let result = match lhs.get_type().as_primitive().ok_or_else(|| {
                ArrowKernelError::DSLInvalidType(
                    "arith operators require primitive values",
                    lhs.get_type(),
                )
            })? {
                PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                    let lhs_v = lhs_v.into_float_value();
                    let rhs_v = rhs_v.into_float_value();
                    match op {
                        DSLArithBinOp::Add => ctx.b.build_float_add(lhs_v, rhs_v, "fadd").unwrap(),
                        DSLArithBinOp::Sub => ctx.b.build_float_sub(lhs_v, rhs_v, "fsub").unwrap(),
                        DSLArithBinOp::Mul => ctx.b.build_float_mul(lhs_v, rhs_v, "fmul").unwrap(),
                        DSLArithBinOp::Div => ctx.b.build_float_div(lhs_v, rhs_v, "fdiv").unwrap(),
                        DSLArithBinOp::Rem => ctx.b.build_float_rem(lhs_v, rhs_v, "frem").unwrap(),
                    }
                    .as_basic_value_enum()
                }
                PrimitiveType::I8
                | PrimitiveType::I16
                | PrimitiveType::I32
                | PrimitiveType::I64
                | PrimitiveType::U8
                | PrimitiveType::U16
                | PrimitiveType::U32
                | PrimitiveType::U64 => {
                    let lhs_v = lhs_v.into_int_value();
                    let rhs_v = rhs_v.into_int_value();
                    match op {
                        DSLArithBinOp::Add => ctx.b.build_int_add(lhs_v, rhs_v, "iadd").unwrap(),
                        DSLArithBinOp::Sub => ctx.b.build_int_sub(lhs_v, rhs_v, "isub").unwrap(),
                        DSLArithBinOp::Mul => ctx.b.build_int_mul(lhs_v, rhs_v, "imul").unwrap(),
                        DSLArithBinOp::Div => {
                            if lhs.get_type().is_signed().unwrap() {
                                ctx.b.build_int_signed_div(lhs_v, rhs_v, "idiv").unwrap()
                            } else {
                                ctx.b.build_int_unsigned_div(lhs_v, rhs_v, "udiv").unwrap()
                            }
                        }
                        DSLArithBinOp::Rem => {
                            if lhs.get_type().is_signed().unwrap() {
                                ctx.b.build_int_signed_rem(lhs_v, rhs_v, "irem").unwrap()
                            } else {
                                ctx.b.build_int_unsigned_rem(lhs_v, rhs_v, "urem").unwrap()
                            }
                        }
                    }
                    .as_basic_value_enum()
                }
                PrimitiveType::List(item, _) => {
                    let inner = PrimitiveType::from(item);
                    if inner.is_float() {
                        let lhs_v = lhs_v.into_vector_value();
                        let rhs_v = rhs_v.into_vector_value();
                        match op {
                            DSLArithBinOp::Add => {
                                ctx.b.build_float_add(lhs_v, rhs_v, "vfadd").unwrap()
                            }
                            DSLArithBinOp::Sub => {
                                ctx.b.build_float_sub(lhs_v, rhs_v, "vfsub").unwrap()
                            }
                            DSLArithBinOp::Mul => {
                                ctx.b.build_float_mul(lhs_v, rhs_v, "vfmul").unwrap()
                            }
                            DSLArithBinOp::Div => {
                                ctx.b.build_float_div(lhs_v, rhs_v, "vfdiv").unwrap()
                            }
                            DSLArithBinOp::Rem => {
                                ctx.b.build_float_rem(lhs_v, rhs_v, "vfrem").unwrap()
                            }
                        }
                        .as_basic_value_enum()
                    } else if inner.is_int() {
                        let lhs_v = lhs_v.into_vector_value();
                        let rhs_v = rhs_v.into_vector_value();
                        match op {
                            DSLArithBinOp::Add => {
                                ctx.b.build_int_add(lhs_v, rhs_v, "viadd").unwrap()
                            }
                            DSLArithBinOp::Sub => {
                                ctx.b.build_int_sub(lhs_v, rhs_v, "visub").unwrap()
                            }
                            DSLArithBinOp::Mul => {
                                ctx.b.build_int_mul(lhs_v, rhs_v, "vimul").unwrap()
                            }
                            DSLArithBinOp::Div => {
                                if inner.is_signed() {
                                    ctx.b.build_int_signed_div(lhs_v, rhs_v, "vidiv").unwrap()
                                } else {
                                    ctx.b.build_int_unsigned_div(lhs_v, rhs_v, "vudiv").unwrap()
                                }
                            }
                            DSLArithBinOp::Rem => {
                                if inner.is_signed() {
                                    ctx.b.build_int_signed_rem(lhs_v, rhs_v, "virem").unwrap()
                                } else {
                                    ctx.b.build_int_unsigned_rem(lhs_v, rhs_v, "vurem").unwrap()
                                }
                            }
                        }
                        .as_basic_value_enum()
                    } else {
                        return Err(ArrowKernelError::DSLInvalidType(
                            "arith operators require numeric vector elements",
                            lhs.get_type(),
                        ));
                    }
                }
                PrimitiveType::P64x2 => {
                    return Err(ArrowKernelError::DSLInvalidType(
                        "arith operators do not support string values",
                        lhs.get_type(),
                    ))
                }
            };

            Ok(result)
        }
        DSLExpr::BitwiseBinOp(op, lhs, rhs) => {
            let lhs_v = compile_expr(ctx, lhs)?;
            let rhs_v = compile_expr(ctx, rhs)?;

            if lhs_v.get_type() != rhs_v.get_type() {
                return Err(ArrowKernelError::DSLTypeMismatch(
                    "bitwise operator",
                    lhs.get_type(),
                    rhs.get_type(),
                ));
            }

            Ok(if lhs_v.get_type().is_int_type() {
                let lhs_v = lhs_v.into_int_value();
                let rhs_v = rhs_v.into_int_value();
                match op {
                    DSLBitwiseBinOp::And => ctx.b.build_and(lhs_v, rhs_v, "band").unwrap(),
                    DSLBitwiseBinOp::Or => ctx.b.build_or(lhs_v, rhs_v, "bor").unwrap(),
                    DSLBitwiseBinOp::Xor => ctx.b.build_xor(lhs_v, rhs_v, "bxor").unwrap(),
                }
                .as_basic_value_enum()
            } else if lhs_v.is_vector_value() {
                let lhs_v = lhs_v.into_vector_value();
                let rhs_v = rhs_v.into_vector_value();
                match op {
                    DSLBitwiseBinOp::And => ctx.b.build_and(lhs_v, rhs_v, "band").unwrap(),
                    DSLBitwiseBinOp::Or => ctx.b.build_or(lhs_v, rhs_v, "bor").unwrap(),
                    DSLBitwiseBinOp::Xor => ctx.b.build_xor(lhs_v, rhs_v, "bxor").unwrap(),
                }
                .as_basic_value_enum()
            } else {
                return Err(ArrowKernelError::DSLInvalidType(
                    "bitwise ops require int or vec",
                    lhs.get_type(),
                ));
            })
        }
        DSLExpr::Len(v) => ctx.lengths[&v.name]
            .ok_or_else(|| {
                ArrowKernelError::DSLInvalidType(
                    "length is not available for input of this type",
                    v.ty.clone(),
                )
            })
            .map(|x| x.as_basic_value_enum()),
        DSLExpr::Bswap(v) => {
            let v = compile_expr(ctx, v)?.into_int_value();

            if v.get_type().get_bit_width() == 8 {
                return Ok(v.as_basic_value_enum());
            }

            let bswap_in = Intrinsic::find("llvm.bswap").unwrap();
            let func = bswap_in
                .get_declaration(ctx.module, &[v.get_type().into()])
                .unwrap();
            let res = ctx
                .b
                .build_call(func, &[v.into()], "bswap")
                .unwrap()
                .try_as_basic_value()
                .unwrap_basic();
            Ok(res)
        }
        DSLExpr::BitNot(v) => {
            let value = compile_expr(ctx, v)?;
            match v.get_type() {
                DSLType::Boolean => Ok(ctx
                    .b
                    .build_not(value.into_int_value(), "bit_not")
                    .unwrap()
                    .as_basic_value_enum()),
                DSLType::Primitive(pt) if pt.is_int() => Ok(ctx
                    .b
                    .build_not(value.into_int_value(), "bit_not")
                    .unwrap()
                    .as_basic_value_enum()),
                DSLType::Primitive(pt) if pt.is_float() => {
                    let int_type = PrimitiveType::int_with_width(pt.width())
                        .llvm_type(ctx.ctx)
                        .into_int_type();
                    let value_bits = ctx
                        .b
                        .build_bit_cast(value, int_type, "bit_not_bits")
                        .unwrap()
                        .into_int_value();
                    let inverted_bits = ctx.b.build_not(value_bits, "bit_not").unwrap();
                    Ok(ctx
                        .b
                        .build_bit_cast(inverted_bits, pt.llvm_type(ctx.ctx), "bit_not_float")
                        .unwrap())
                }
                DSLType::Primitive(pt) if matches!(pt, PrimitiveType::List(_, _)) => {
                    let int_v_type =
                        PrimitiveType::int_with_width(pt.list_type_into_inner().width())
                            .llvm_vec_type(ctx.ctx, 64)
                            .unwrap();

                    let value_bits = ctx
                        .b
                        .build_bit_cast(value, int_v_type, "bit_not_bits")
                        .unwrap()
                        .into_vector_value();
                    let inverted_bits = ctx.b.build_not(value_bits, "bit_not").unwrap();
                    Ok(ctx
                        .b
                        .build_bit_cast(inverted_bits, pt.llvm_type(ctx.ctx), "bit_not_vec")
                        .unwrap())
                }
                _ => Err(ArrowKernelError::DSLInvalidType(
                    "invalid type for bit_not",
                    v.get_type(),
                )),
            }
        }

        DSLExpr::Neg(v) => {
            let value = compile_expr(ctx, v)?;
            match v.get_type() {
                DSLType::Primitive(pt) if pt.is_int() => Ok(ctx
                    .b
                    .build_int_neg(value.into_int_value(), "neg")
                    .unwrap()
                    .as_basic_value_enum()),
                DSLType::Primitive(pt) if pt.is_float() => Ok(ctx
                    .b
                    .build_float_neg(value.into_float_value(), "neg")
                    .unwrap()
                    .as_basic_value_enum()),
                DSLType::Primitive(PrimitiveType::List(item, _)) => {
                    if PrimitiveType::from(item).is_float() {
                        Ok(ctx
                            .b
                            .build_float_neg(value.into_vector_value(), "neg")
                            .unwrap()
                            .as_basic_value_enum())
                    } else {
                        Ok(ctx
                            .b
                            .build_int_neg(value.into_vector_value(), "neg")
                            .unwrap()
                            .as_basic_value_enum())
                    }
                }
                _ => Err(ArrowKernelError::DSLInvalidType(
                    "invalid type for neg",
                    v.get_type(),
                )),
            }
        }

        DSLExpr::Abs(v) => {
            let value = compile_expr(ctx, v)?;
            // `false` for the intrinsic's `is_int_min_poison` operand keeps the
            // wrapping semantics: abs(iN::MIN) == iN::MIN.
            let no_poison = ctx.ctx.bool_type().const_zero();
            match v.get_type() {
                // unsigned integers are already non-negative, so abs is identity
                DSLType::Primitive(pt) if pt.is_int() && !pt.is_signed() => Ok(value),
                DSLType::Primitive(pt) if pt.is_int() => {
                    let abs = Intrinsic::find("llvm.abs").unwrap();
                    let abs = abs.get_declaration(ctx.module, &[value.get_type()]).unwrap();
                    Ok(ctx
                        .b
                        .build_call(abs, &[value.into(), no_poison.into()], "abs")
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic())
                }
                DSLType::Primitive(pt) if pt.is_float() => {
                    let fabs = Intrinsic::find("llvm.fabs").unwrap();
                    let fabs = fabs.get_declaration(ctx.module, &[value.get_type()]).unwrap();
                    Ok(ctx
                        .b
                        .build_call(fabs, &[value.into()], "abs")
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_basic())
                }
                DSLType::Primitive(PrimitiveType::List(item, _)) => {
                    let inner = PrimitiveType::from(item);
                    if inner.is_float() {
                        let fabs = Intrinsic::find("llvm.fabs").unwrap();
                        let fabs = fabs.get_declaration(ctx.module, &[value.get_type()]).unwrap();
                        Ok(ctx
                            .b
                            .build_call(fabs, &[value.into()], "abs")
                            .unwrap()
                            .try_as_basic_value()
                            .unwrap_basic())
                    } else if inner.is_signed() {
                        let abs = Intrinsic::find("llvm.abs").unwrap();
                        let abs = abs.get_declaration(ctx.module, &[value.get_type()]).unwrap();
                        Ok(ctx
                            .b
                            .build_call(abs, &[value.into(), no_poison.into()], "abs")
                            .unwrap()
                            .try_as_basic_value()
                            .unwrap_basic())
                    } else {
                        Ok(value)
                    }
                }
                _ => Err(ArrowKernelError::DSLInvalidType(
                    "invalid type for abs",
                    v.get_type(),
                )),
            }
        }

        DSLExpr::Select(cond, v1, v2) => {
            let cond_v = compile_expr(ctx, cond)?.into_int_value();
            let v1_v = compile_expr(ctx, v1)?;
            let v2_v = compile_expr(ctx, v2)?;
            Ok(ctx
                .b
                .build_select(cond_v, v1_v, v2_v, "select")
                .unwrap()
                .as_basic_value_enum())
        }
    }
}

fn cast_numeric<'ctx, 'a>(
    ctx: &DSLCompilationContext<'ctx, 'a>,
    val: BasicValueEnum<'ctx>,
    orig_pt: NumericPrimitiveType,
    tar_pt: NumericPrimitiveType,
) -> Result<BasicValueEnum<'ctx>, ArrowKernelError> {
    let tar_type = PrimitiveType::from(tar_pt).llvm_type(ctx.ctx);
    Ok(match (orig_pt.is_integer(), tar_pt.is_integer()) {
        // int to int
        (true, true) => match tar_pt.width().cmp(&orig_pt.width()) {
            Ordering::Less => ctx
                .b
                .build_int_truncate(val.into_int_value(), tar_type.into_int_type(), "cast")
                .unwrap()
                .as_basic_value_enum(),
            Ordering::Equal => val,
            Ordering::Greater => if orig_pt.is_signed() {
                ctx.b
                    .build_int_s_extend(val.into_int_value(), tar_type.into_int_type(), "cast")
            } else {
                ctx.b
                    .build_int_z_extend(val.into_int_value(), tar_type.into_int_type(), "cast")
            }
            .unwrap()
            .as_basic_value_enum(),
        },
        // int to float
        (true, false) => if orig_pt.is_signed() {
            ctx.b.build_signed_int_to_float(
                val.into_int_value(),
                tar_type.into_float_type(),
                "cast",
            )
        } else {
            ctx.b.build_unsigned_int_to_float(
                val.into_int_value(),
                tar_type.into_float_type(),
                "cast",
            )
        }
        .unwrap()
        .as_basic_value_enum(),
        // float to int
        (false, true) => if orig_pt.is_signed() {
            ctx.b.build_float_to_signed_int(
                val.into_float_value(),
                tar_type.into_int_type(),
                "cast",
            )
        } else {
            ctx.b.build_float_to_unsigned_int(
                val.into_float_value(),
                tar_type.into_int_type(),
                "cast",
            )
        }
        .unwrap()
        .as_basic_value_enum(),
        // float to float
        (false, false) => ctx
            .b
            .build_float_cast(val.into_float_value(), tar_type.into_float_type(), "cast")
            .unwrap()
            .as_basic_value_enum(),
    })
}

fn cast_numeric_vec<'ctx, 'a>(
    ctx: &DSLCompilationContext<'ctx, 'a>,
    val: VectorValue<'ctx>,
    orig_pt: NumericPrimitiveType,
    tar_pt: NumericPrimitiveType,
) -> Result<VectorValue<'ctx>, ArrowKernelError> {
    let tar_type = PrimitiveType::from(tar_pt)
        .llvm_vec_type(ctx.ctx, val.get_type().get_size())
        .unwrap();
    Ok(match (orig_pt.is_integer(), tar_pt.is_integer()) {
        // int to int
        (true, true) => match tar_pt.width().cmp(&orig_pt.width()) {
            Ordering::Less => ctx.b.build_int_truncate(val, tar_type, "cast").unwrap(),
            Ordering::Equal => val,
            Ordering::Greater => if orig_pt.is_signed() {
                ctx.b.build_int_s_extend(val, tar_type, "cast")
            } else {
                ctx.b.build_int_z_extend(val, tar_type, "cast")
            }
            .unwrap(),
        },
        // int to float
        (true, false) => if orig_pt.is_signed() {
            ctx.b.build_signed_int_to_float(val, tar_type, "cast")
        } else {
            ctx.b.build_unsigned_int_to_float(val, tar_type, "cast")
        }
        .unwrap(),
        // float to int
        (false, true) => if orig_pt.is_signed() {
            ctx.b.build_float_to_signed_int(val, tar_type, "cast")
        } else {
            ctx.b.build_float_to_unsigned_int(val, tar_type, "cast")
        }
        .unwrap(),
        // float to float
        (false, false) => ctx.b.build_float_cast(val, tar_type, "cast").unwrap(),
    })
}

fn scalar_to_llvm<'ctx, 'a>(
    ctx: &DSLCompilationContext<'ctx, 'a>,
    val: &dyn Datum,
) -> Result<BasicValueEnum<'ctx>, ArrowKernelError> {
    let (arr, is_scalar) = val.get();
    assert!(is_scalar, "scalar_to_llvm called with non-scalar value");

    let res = match PrimitiveType::for_arrow_type(arr.data_type()) {
        PrimitiveType::I8 => arr.as_primitive_opt::<Int8Type>().map(|x| {
            ctx.ctx
                .i8_type()
                .const_int(x.value(0) as u64, true)
                .as_basic_value_enum()
        }),
        PrimitiveType::I16 => arr.as_primitive_opt::<Int16Type>().map(|x| {
            ctx.ctx
                .i16_type()
                .const_int(x.value(0) as u64, true)
                .as_basic_value_enum()
        }),
        PrimitiveType::I32 => arr.as_primitive_opt::<Int32Type>().map(|x| {
            ctx.ctx
                .i32_type()
                .const_int(x.value(0) as u64, true)
                .as_basic_value_enum()
        }),
        PrimitiveType::I64 => arr.as_primitive_opt::<Int64Type>().map(|x| {
            ctx.ctx
                .i64_type()
                .const_int(x.value(0) as u64, true)
                .as_basic_value_enum()
        }),
        PrimitiveType::U8 => arr.as_primitive_opt::<UInt8Type>().map(|x| {
            ctx.ctx
                .i8_type()
                .const_int(x.value(0) as u64, false)
                .as_basic_value_enum()
        }),
        PrimitiveType::U16 => arr.as_primitive_opt::<UInt16Type>().map(|x| {
            ctx.ctx
                .i16_type()
                .const_int(x.value(0) as u64, false)
                .as_basic_value_enum()
        }),
        PrimitiveType::U32 => arr.as_primitive_opt::<UInt32Type>().map(|x| {
            ctx.ctx
                .i32_type()
                .const_int(x.value(0) as u64, false)
                .as_basic_value_enum()
        }),
        PrimitiveType::U64 => arr.as_primitive_opt::<UInt64Type>().map(|x| {
            ctx.ctx
                .i64_type()
                .const_int(x.value(0), false)
                .as_basic_value_enum()
        }),
        PrimitiveType::F16 => arr.as_primitive_opt::<Float16Type>().map(|x| {
            ctx.ctx
                .f16_type()
                .const_float(x.value(0).to_f64())
                .as_basic_value_enum()
        }),
        PrimitiveType::F32 => arr.as_primitive_opt::<Float32Type>().map(|x| {
            ctx.ctx
                .f32_type()
                .const_float(x.value(0) as f64)
                .as_basic_value_enum()
        }),
        PrimitiveType::F64 => arr.as_primitive_opt::<Float64Type>().map(|x| {
            ctx.ctx
                .f32_type()
                .const_float(x.value(0))
                .as_basic_value_enum()
        }),
        PrimitiveType::P64x2 => todo!(),
        PrimitiveType::List(_, _) => todo!(),
    };

    Ok(res.expect("non-primitive const scalar in compilation"))
}
