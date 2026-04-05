use std::{cmp::Ordering, collections::HashMap, ffi::c_void};

use arrow_array::{
    cast::AsArray,
    types::{UInt32Type, UInt64Type},
    BooleanArray, Datum,
};
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    intrinsics::Intrinsic,
    module::{Linkage, Module},
    types::{BasicType, BasicTypeEnum, PointerType},
    values::{BasicValue, BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};
use itertools::Itertools;
use ouroboros::self_referencing;

use crate::{
    compiled_iter::{
        array_to_setbit_iter, datum_to_iter, generate_next, generate_random_access,
        get_iterator_length,
    },
    compiled_kernels::{
        cmp::add_memcmp,
        dsl::{DSLError, KernelParameters},
        dsl2::{
            buffer,
            runtime::RunnableDSLFunction,
            two_d,
            writers::{OutputSlot, OutputWriter},
            DSL2Error, DSLArgument, DSLArgumentType, DSLComparison, DSLExpr, DSLFunction, DSLStmt,
            DSLType, DSLValue,
        },
        link_req_helpers, optimize_module,
    },
    compiled_writers::{ArrayWriter, PrimitiveArrayWriter},
    increment_pointer, set_noalias_params, NumericPrimitiveType, PrimitiveSuperType, PrimitiveType,
};

pub enum KernelReturnCode {
    Success,
    InvalidEmitIndex,
    Unknown(u64),
}

impl From<KernelReturnCode> for u64 {
    fn from(code: KernelReturnCode) -> Self {
        match code {
            KernelReturnCode::Success => 0,
            KernelReturnCode::InvalidEmitIndex => 1,
            KernelReturnCode::Unknown(code) => code,
        }
    }
}

impl From<u64> for KernelReturnCode {
    fn from(code: u64) -> Self {
        match code {
            0 => KernelReturnCode::Success,
            1 => KernelReturnCode::InvalidEmitIndex,
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
                _ => PrimitiveType::for_arrow_type(&dt).llvm_type(ctx),
            }
        }
        DSLType::Buffer(..)
        | DSLType::Scalar(..)
        | DSLType::SetBits(..)
        | DSLType::TwoDArray(..)
        | DSLType::Array(..) => ptr_type,
    }
}

pub struct DSLCompilationContext<'ctx, 'a> {
    pub ctx: &'ctx Context,
    pub module: &'a Module<'ctx>,
    pub func: &'a FunctionValue<'ctx>,
    pub b: &'a Builder<'ctx>,
    pub st: HashMap<usize, BasicValueEnum<'a>>,
    pub access_funcs: HashMap<usize, FunctionValue<'a>>,
    pub next_funcs: HashMap<usize, FunctionValue<'a>>,
    pub writer_funcs: HashMap<usize, FunctionValue<'a>>,
    pub outputs: Vec<OutputWriter<'a>>,
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
) -> Result<RunnableDSLFunction, DSL2Error> {
    let ctx = Context::create();
    let args = args.into_iter().collect_vec();
    let arg_types = args
        .iter()
        .zip(f.params.iter().map(|p| &p.ty))
        .map(|(arg, dsl_type)| arg.get_type(dsl_type))
        .collect_vec();

    let func = CompiledDSLFunctionTryBuilder {
        ctx,
        f,
        arg_types,
        compiled_builder: |ctx, f| compile_inner(ctx, f, args),
    }
    .try_build()?;
    RunnableDSLFunction::new(func)
}

pub fn compile_inner<'ctx, 'args>(
    ctx: &'ctx Context,
    f: &DSLFunction,
    args: impl IntoIterator<Item = DSLArgument<'args>>,
) -> Result<JitFunction<'ctx, unsafe extern "C" fn(*mut c_void) -> u64>, DSL2Error> {
    let module = ctx.create_module("dsl2");
    let args: Vec<_> = args.into_iter().collect();

    // validate parameters
    assert_eq!(args.len(), f.params.len());
    for (i, (arg, param)) in args.iter().zip(f.params.iter()).enumerate() {
        if !arg.is_compatible_with(&param.ty) {
            return Err(DSL2Error::ArgumentTypeMismatch(
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
            .map(|arg| dsl_type_to_llvm_type(&ctx, &arg.ty))
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
    let mut next_funcs = HashMap::new();
    let mut writer_funcs = HashMap::new();

    // load LLVM function parameters into the symbol table
    for (idx, param) in f.params.iter().enumerate() {
        st.insert(param.name, func.get_nth_param(idx as u32).unwrap());
    }

    // first, initialize all iterators, getters, and outputs
    let init_block = ctx.append_basic_block(func, "init");
    let b = ctx.create_builder();
    b.position_at_end(init_block);

    // setup accessors (random access)
    for idx in f.accessed_parameters() {
        let param = names_to_params[&idx];
        let arg = &names_to_args[&idx];

        match &param.ty {
            DSLType::Scalar(t) | DSLType::Array(t, _) => {
                let ih = datum_to_iter(arg.as_datum().unwrap())?;
                let ptr = ih.localize_struct(&ctx, &b, st[&param.name].into_pointer_value());
                st.insert(param.name, ptr.as_basic_value_enum());

                let detected_dtype = DSLType::of_iterator_holder(&ih).iter_type().unwrap();
                if &detected_dtype != t.as_ref() {
                    return Err(DSL2Error::TypeMismatch(
                        "accessed parameters",
                        *t.clone(),
                        detected_dtype,
                    ));
                }

                let access_func = generate_random_access(
                    &ctx,
                    &module,
                    &format!("access_{}", param.name),
                    &arg.data_type(),
                    &ih,
                )
                .unwrap();
                access_funcs.insert(param.name, access_func);
            }
            DSLType::Buffer(t) => {
                access_funcs.insert(
                    param.name,
                    buffer::generate_buffer_accessor(&ctx, &module, *t),
                );
                writer_funcs.insert(
                    param.name,
                    buffer::generate_buffer_writer(&ctx, &module, *t),
                );
            }
            DSLType::TwoDArray(..) => {
                access_funcs.insert(
                    param.name,
                    two_d::generate_two_d_access(&ctx, &module, arg.as_two_d().unwrap())?,
                );
            }
            _ => return Err(DSL2Error::NonRandomAccessType(param.ty.clone())),
        };
    }

    // setup iterators
    let mut iterator_holders = HashMap::new();
    for idx in f.iterated_parameters() {
        let param = names_to_params[&idx];
        let arg = &names_to_args[&idx];

        match &param.ty {
            DSLType::Scalar(t) | DSLType::Array(t, ..) => {
                let ih = datum_to_iter(
                    arg.as_datum()
                        .ok_or_else(|| DSL2Error::NonIterableType(param.ty.clone()))?,
                )?;

                let ptr = ih.localize_struct(&ctx, &b, st[&param.name].into_pointer_value());
                st.insert(param.name, ptr.as_basic_value_enum());
                let detected_dtype = DSLType::of_iterator_holder(&ih).iter_type().unwrap();
                if &detected_dtype != t.as_ref() {
                    return Err(DSL2Error::TypeMismatch(
                        "iterated parameters",
                        *t.clone(),
                        detected_dtype,
                    ));
                }

                let next_func = generate_next(
                    &ctx,
                    &module,
                    &format!("next_{}", param.name),
                    &arg.data_type(),
                    &ih,
                )
                .unwrap();
                next_funcs.insert(param.name, next_func);
                iterator_holders.insert(param.name, ih);
            }
            DSLType::SetBits(_) => {
                let ih = array_to_setbit_iter(&BooleanArray::new_null(0))?;
                let ptr = ih.localize_struct(&ctx, &b, st[&param.name].into_pointer_value());
                st.insert(param.name, ptr.as_basic_value_enum());

                if arg.data_type() != DataType::Boolean {
                    return Err(DSL2Error::InvalidSetBitType(arg.data_type()));
                }

                let next_func = generate_next(
                    &ctx,
                    &module,
                    &format!("next_setbit_{}", param.name),
                    &arg.data_type(),
                    &ih,
                )
                .unwrap();
                next_funcs.insert(param.name, next_func);
            }
            _ => return Err(DSL2Error::NonIterableType(param.ty.clone())),
        };
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

        let lengths = group
            .iter()
            .filter_map(|idx| {
                get_iterator_length(
                    &ctx,
                    &b,
                    &iterator_holders[idx],
                    st[idx].into_pointer_value(),
                )
            })
            .collect_vec();

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
    let mut writers = Vec::new();
    let output_param_offset = func.count_params() as usize - f.ret.len();
    for (idx, ret) in f.ret.iter().enumerate() {
        let os = ret.allocate(0);
        let w = os.llvm_init(
            &ctx,
            &module,
            &b,
            func.get_nth_param((output_param_offset + idx) as u32)
                .unwrap()
                .into_pointer_value(),
        );
        writers.push(w);
    }

    let mut dsl_ctx = DSLCompilationContext {
        ctx: &ctx,
        module: &module,
        func: &func,
        b: &b,
        st,
        access_funcs,
        next_funcs,
        writer_funcs,
        outputs: writers,
    };

    for stmt in f.body.iter() {
        compile_stmt(&mut dsl_ctx, &stmt)?;
    }

    // flush all outputs
    for w in dsl_ctx.outputs.iter() {
        w.llvm_flush(ctx, &b);
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

    //module.print_to_stderr();
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
) -> Result<(), DSL2Error> {
    match stmt {
        DSLStmt::Emit { index, value } => {
            if ctx.outputs.is_empty() {
                return Err(DSL2Error::EmitWithNoOutputs);
            }

            if let Some(idx) = index.as_u32() {
                let accepted = ctx.outputs[idx as usize].accepted_type();
                if value.get_type() != accepted {
                    return Err(DSL2Error::TypeMismatch("emit", accepted, value.get_type()));
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
                    o.llvm_ingest(ctx.ctx, &b, value);
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
        DSLStmt::Set {
            buf_idx,
            index,
            value,
        } => {
            let index = compile_expr(ctx, index)?;
            let value = compile_expr(ctx, value)?;
            let func = ctx.writer_funcs[buf_idx];

            ctx.b
                .build_call(
                    func,
                    &[ctx.st[buf_idx].into(), index.into(), value.into()],
                    "set",
                )
                .unwrap();
        }
        DSLStmt::If { cond, then } => {
            let cond = compile_expr(ctx, cond)?;

            let if_path = ctx.ctx.append_basic_block(*ctx.func, "if_path");
            let after_if = ctx.ctx.append_basic_block(*ctx.func, "after_if");

            ctx.b
                .build_conditional_branch(cond.into_int_value(), if_path, after_if)
                .unwrap();
            ctx.b.position_at_end(if_path);
            for stmt in then {
                compile_stmt(ctx, stmt)?;
            }
            ctx.b.build_unconditional_branch(after_if).unwrap();

            ctx.b.position_at_end(after_if);
        }
        DSLStmt::For(floop) => {
            let loop_header = ctx.ctx.append_basic_block(*ctx.func, "loop_header");
            let loop_body = ctx.ctx.append_basic_block(*ctx.func, "loop_body");
            let loop_end = ctx.ctx.append_basic_block(*ctx.func, "loop_end");

            ctx.b.build_unconditional_branch(loop_header).unwrap();
            // in the loop header, call next on all iterators and `and` together
            // the results
            ctx.b.position_at_end(loop_header);

            let bufs = floop
                .loop_vars
                .iter()
                .map(|iter| iter.ty.llvm_type(ctx.ctx).unwrap())
                .enumerate()
                .map(|(i, llvm_type)| {
                    ctx.b
                        .build_alloca(llvm_type, &format!("iter_buf{}", i))
                        .unwrap()
                })
                .collect_vec();

            let mut last_res = None;
            for (iter, buf_ptr) in floop.iterators.iter().zip(bufs.iter()) {
                let next_func = ctx.next_funcs[&iter.name];
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
                ctx.st.insert(loop_var.name, val);
            }
            for stmt in floop.body.iter() {
                compile_stmt(ctx, &stmt)?;
            }
            ctx.b.build_unconditional_branch(loop_header).unwrap();

            ctx.b.position_at_end(loop_end);
        }
    }
    Ok(())
}

fn compile_expr<'ctx, 'a>(
    ctx: &DSLCompilationContext<'ctx, 'a>,
    expr: &DSLExpr,
) -> Result<BasicValueEnum<'a>, DSL2Error> {
    match expr {
        DSLExpr::Compare(op, lhs, rhs) => {
            let lhs_type = lhs.get_type();
            let rhs_type = rhs.get_type();

            if lhs_type != rhs_type {
                return Err(DSL2Error::TypeMismatch("comparison", lhs_type, rhs_type));
            }

            let signed = lhs_type.is_signed().unwrap();
            let lhs = compile_expr(ctx, lhs)?;
            let rhs = compile_expr(ctx, rhs)?;

            if lhs.is_float_value() {
                Ok(ctx
                    .b
                    .build_float_compare(
                        op.as_float_predicate(),
                        lhs.into_float_value(),
                        rhs.into_float_value(),
                        "cmp",
                    )
                    .unwrap()
                    .as_basic_value_enum())
            } else if lhs.is_int_value() {
                Ok(ctx
                    .b
                    .build_int_compare(
                        op.as_int_predicate(signed),
                        lhs.into_int_value(),
                        rhs.into_int_value(),
                        "cmp",
                    )
                    .unwrap()
                    .as_basic_value_enum())
            } else if lhs_type.as_primitive() == Some(PrimitiveType::P64x2) {
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
            } else {
                todo!()
            }
        }
        DSLExpr::At(v, i) => match &v.ty {
            DSLType::Scalar(..) | DSLType::Array(..) | DSLType::Buffer(..) => {
                if i.len() != 1 {
                    return Err(DSL2Error::InvalidIndex(
                        "non 2D-array should have a single index",
                    ));
                }
                let i = i.first().unwrap();
                let access_func = ctx.access_funcs[&v.name];
                let iter_ptr = ctx.st[&v.name];
                if i.get_type() != DSLType::Primitive(PrimitiveType::U64) {
                    return Err(DSL2Error::TypeMismatch(
                        "indexing requires unsigned 64 bit integers",
                        DSLType::Primitive(PrimitiveType::U64),
                        i.get_type(),
                    ));
                }
                let idx = compile_expr(ctx, i)?;
                Ok(ctx
                    .b
                    .build_call(
                        access_func,
                        &[iter_ptr.into(), idx.into()],
                        "access_by_index",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic())
            }
            DSLType::TwoDArray(..) => {
                if i.len() != 2 {
                    return Err(DSL2Error::InvalidIndex("2D array needs two indices"));
                }

                let access_func = ctx.access_funcs[&v.name];
                let arr_ptr = ctx.st[&v.name];
                for idx in i {
                    if idx.get_type() != DSLType::Primitive(PrimitiveType::U64) {
                        return Err(DSL2Error::TypeMismatch(
                            "indexing requires unsigned 64 bit integers",
                            DSLType::Primitive(PrimitiveType::U64),
                            idx.get_type(),
                        ));
                    }
                }

                let idx0 = compile_expr(ctx, &i[0])?;
                let idx1 = compile_expr(ctx, &i[1])?;
                Ok(ctx
                    .b
                    .build_call(
                        access_func,
                        &[arr_ptr.into(), idx0.into(), idx1.into()],
                        "access_2d_by_index",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic())
            }
            _ => todo!(),
        },
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
                    _ => todo!(),
                }
            }
            _ => Ok(ctx.st[&v.name]),
        },
        DSLExpr::Cast(val, tar_pt) => {
            let orig_type = val.get_type();
            let tar_type = tar_pt.llvm_type(&ctx.ctx);
            let val = compile_expr(ctx, val)?;

            match orig_type {
                DSLType::Boolean => {
                    if tar_pt.is_int() {
                        Ok(ctx
                            .b
                            .build_int_cast(val.into_int_value(), tar_type.into_int_type(), "cast")
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
                        (None, None) => todo!(),
                        (None, Some(_)) => todo!(),
                        (Some(_), None) => todo!(),
                        (Some(orig_pt), Some(tar_pt)) => {
                            return Ok(match (orig_pt.is_integer(), tar_pt.is_integer()) {
                                // int to int
                                (true, true) => match tar_pt.width().cmp(&orig_pt.width()) {
                                    Ordering::Less => ctx
                                        .b
                                        .build_int_truncate(
                                            val.into_int_value(),
                                            tar_type.into_int_type(),
                                            "cast",
                                        )
                                        .unwrap()
                                        .as_basic_value_enum(),
                                    Ordering::Equal => val,
                                    Ordering::Greater => if orig_pt.is_signed() {
                                        ctx.b.build_int_s_extend(
                                            val.into_int_value(),
                                            tar_type.into_int_type(),
                                            "cast",
                                        )
                                    } else {
                                        ctx.b.build_int_z_extend(
                                            val.into_int_value(),
                                            tar_type.into_int_type(),
                                            "cast",
                                        )
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
                                    .build_float_cast(
                                        val.into_float_value(),
                                        tar_type.into_float_type(),
                                        "cast",
                                    )
                                    .unwrap()
                                    .as_basic_value_enum(),
                            });
                        }
                    }
                }
                _ => unimplemented!("invalid cast"),
            }
        }
    }
}
