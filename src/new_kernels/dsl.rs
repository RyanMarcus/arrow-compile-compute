use std::{collections::BTreeSet, ffi::c_void, fmt::Debug, sync::Arc};

use arrow_array::{make_array, ArrayRef, BooleanArray, Datum};
use arrow_buffer::{BooleanBuffer, Buffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    module::Module,
    types::BasicTypeEnum,
    values::{BasicValue, BasicValueEnum, PointerValue},
    AddressSpace, OptimizationLevel,
};
use itertools::Itertools;
use ouroboros::self_referencing;
use repr_offset::ReprOffset;

use crate::{
    declare_blocks, increment_pointer,
    new_iter::{datum_to_iter, generate_next, IteratorHolder},
    new_kernels::{
        cmp::{add_float_vec_to_int_vec, add_memcmp},
        gen_convert_numeric_vec, optimize_module,
    },
    writers::{ArrayWriter, BooleanWriter, PrimitiveArrayWriter},
    ComparisonType, Predicate, PrimitiveType,
};

use super::ArrowKernelError;

#[derive(Debug)]
pub enum DSLError {
    InvalidInputIndex(usize),
    InvalidKernelOutputLength(usize),
    TypeMismatch(String),
    BooleanExpected(String),
}

#[derive(Clone)]
pub enum KernelInput<'a> {
    Datum(usize, &'a dyn Datum),
    SetBits(usize, &'a BooleanArray),
}

impl<'a> Debug for KernelInput<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelInput::Datum(index, datum) => {
                write!(f, "Datum({}, {:?})", index, datum.get().0.data_type())
            }
            KernelInput::SetBits(index, set_bits) => {
                write!(f, "SetBits({}, {:?})", index, set_bits)
            }
        }
    }
}

impl<'a> KernelInput<'a> {
    pub fn into_iter(self) -> KernelIterator<'a> {
        KernelIterator {
            data: vec![KernelExpression::Item(self)],
        }
    }

    fn data_type(&self) -> DataType {
        match self {
            KernelInput::Datum(_, datum) => datum.get().0.data_type().clone(),
            KernelInput::SetBits(..) => DataType::UInt64,
        }
    }

    fn index(&self) -> usize {
        match self {
            KernelInput::Datum(index, _) => *index,
            KernelInput::SetBits(index, _) => *index,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum KernelOutputType {
    Array,
    Boolean,
    Dictionary,
    RunEnd,
}

pub type KernelOutput<'a> = (KernelOutputType, KernelExpression<'a>);

pub struct KernelIterator<'a> {
    data: Vec<KernelExpression<'a>>,
}

impl<'a> KernelIterator<'a> {
    pub fn zip(mut self, other: KernelIterator<'a>) -> KernelIterator<'a> {
        self.data.extend(other.data);
        self
    }

    pub fn map<F: Fn(&[KernelExpression<'a>]) -> Vec<KernelExpression<'a>>>(
        self,
        f: F,
    ) -> KernelIterator<'a> {
        KernelIterator {
            data: f(&self.data),
        }
    }

    pub fn collect(mut self, ty: KernelOutputType) -> Result<KernelOutput<'a>, DSLError> {
        if self.data.len() != 1 {
            Err(DSLError::InvalidKernelOutputLength(self.data.len()))
        } else {
            Ok((ty, self.data.pop().unwrap()))
        }
    }
}

#[derive(Clone, Debug)]
pub enum KernelExpression<'a> {
    Item(KernelInput<'a>),
    Truncate(Box<KernelExpression<'a>>, usize),
    Cmp(
        Predicate,
        Box<KernelExpression<'a>>,
        Box<KernelExpression<'a>>,
    ),
    And(Box<KernelExpression<'a>>, Box<KernelExpression<'a>>),
    Or(Box<KernelExpression<'a>>, Box<KernelExpression<'a>>),
    Select(
        Box<KernelExpression<'a>>,
        Box<KernelExpression<'a>>,
        Box<KernelExpression<'a>>,
    ),
}

impl<'a> KernelExpression<'a> {
    pub fn cmp(&self, pred: Predicate, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::Cmp(pred, Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn gt(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        self.cmp(Predicate::Gt, other)
    }
    pub fn eq(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        self.cmp(Predicate::Eq, other)
    }
    pub fn lt(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        self.cmp(Predicate::Lt, other)
    }
    pub fn or(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::Or(Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn and(&self, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::And(Box::new(self.clone()), Box::new(other.clone()))
    }
    pub fn truncate(&self, size: usize) -> KernelExpression<'a> {
        KernelExpression::Truncate(Box::new(self.clone()), size)
    }
    pub fn select(
        &self,
        a: &KernelExpression<'a>,
        b: &KernelExpression<'a>,
    ) -> KernelExpression<'a> {
        KernelExpression::Select(
            Box::new(self.clone()),
            Box::new(a.clone()),
            Box::new(b.clone()),
        )
    }

    fn descend<F: FnMut(&Self)>(&self, f: &mut F) {
        match self {
            KernelExpression::Item(..) => f(self),
            KernelExpression::Truncate(e, _) => {
                f(self);
                e.descend(f);
            }
            KernelExpression::Cmp(.., lhs, rhs) => {
                f(self);
                lhs.descend(f);
                rhs.descend(f);
            }
            KernelExpression::And(lhs, rhs) => {
                f(self);
                lhs.descend(f);
                rhs.descend(f);
            }
            KernelExpression::Or(lhs, rhs) => {
                f(self);
                lhs.descend(f);
                rhs.descend(f);
            }
            KernelExpression::Select(a, b, c) => {
                f(self);
                a.descend(f);
                b.descend(f);
                c.descend(f);
            }
        }
    }

    fn iterated_indexes(&self) -> Vec<usize> {
        let mut h = BTreeSet::new();
        self.descend(&mut |e| match e {
            KernelExpression::Item(kernel_input) => {
                h.insert(kernel_input.index());
            }
            _ => {}
        });
        h.into_iter().collect()
    }

    fn get_type(&self) -> DataType {
        match self {
            KernelExpression::Item(kernel_input) => kernel_input.data_type(),
            KernelExpression::Truncate(..) => DataType::Binary,
            KernelExpression::Select(_cond, lhs, _rhs) => lhs.get_type(),
            KernelExpression::Cmp(..) | KernelExpression::And(..) | KernelExpression::Or(..) => {
                DataType::Boolean
            }
        }
    }

    fn compile<'b>(
        &self,
        ctx: &'b Context,
        llvm_mod: &Module<'b>,
        build: &Builder<'b>,
        bufs: &[PointerValue<'b>],
        llvm_types: &[BasicTypeEnum<'b>],
    ) -> Result<BasicValueEnum<'b>, DSLError> {
        match self {
            KernelExpression::Item(kernel_input) => {
                let buf = bufs[kernel_input.index()];
                let llvm_type = llvm_types[kernel_input.index()];
                Ok(build.build_load(llvm_type, buf, "load").unwrap())
            }
            KernelExpression::Truncate(_kernel_expression, _) => todo!(),
            KernelExpression::Cmp(predicate, lhs, rhs) => {
                let lhs_v = lhs.compile(ctx, llvm_mod, build, bufs, llvm_types)?;
                let rhs_v = rhs.compile(ctx, llvm_mod, build, bufs, llvm_types)?;

                let lhs_ptype = PrimitiveType::for_arrow_type(&lhs.get_type());
                let rhs_ptype = PrimitiveType::for_arrow_type(&rhs.get_type());

                let res = match (lhs_ptype.comparison_type(), rhs_ptype.comparison_type()) {
                    (ComparisonType::String, ComparisonType::String) => {
                        let memcmp = add_memcmp(ctx, llvm_mod);
                        let res = build
                            .build_call(memcmp, &[lhs_v.into(), rhs_v.into()], "memcmp_res")
                            .unwrap()
                            .try_as_basic_value()
                            .unwrap_left()
                            .into_int_value();
                        build
                            .build_int_compare(
                                predicate.as_int_pred(true),
                                res,
                                ctx.i64_type().const_zero(),
                                "cmp_res",
                            )
                            .unwrap()
                            .as_basic_value_enum()
                    }
                    (ComparisonType::String, _) | (_, ComparisonType::String) => {
                        return Err(DSLError::TypeMismatch(format!(
                            "cannot compare {} and {}",
                            lhs.get_type(),
                            rhs.get_type()
                        )));
                    }
                    _ => {
                        let dom_type = PrimitiveType::dominant(lhs_ptype, rhs_ptype).unwrap();
                        let lhs_v = build
                            .build_bit_cast(
                                lhs_v,
                                lhs_ptype.llvm_vec_type(ctx, 1).unwrap(),
                                "single_vec_lhs",
                            )
                            .unwrap()
                            .into_vector_value();
                        let rhs_v = build
                            .build_bit_cast(
                                rhs_v,
                                rhs_ptype.llvm_vec_type(ctx, 1).unwrap(),
                                "single_vec_lhs",
                            )
                            .unwrap()
                            .into_vector_value();
                        let clhs = gen_convert_numeric_vec(ctx, build, lhs_v, lhs_ptype, dom_type);
                        let crhs = gen_convert_numeric_vec(ctx, build, rhs_v, rhs_ptype, dom_type);

                        let vec_res = match dom_type.comparison_type() {
                            ComparisonType::Int { signed } => build
                                .build_int_compare(predicate.as_int_pred(signed), clhs, crhs, "cmp")
                                .unwrap(),
                            ComparisonType::Float => {
                                let convert = add_float_vec_to_int_vec(
                                    ctx,
                                    llvm_mod,
                                    lhs_v.get_type().get_size(),
                                    dom_type,
                                );
                                let lhs = build
                                    .build_call(convert, &[clhs.into()], "lhs_converted")
                                    .unwrap()
                                    .try_as_basic_value()
                                    .unwrap_left()
                                    .into_vector_value();
                                let rhs = build
                                    .build_call(convert, &[crhs.into()], "rhs_converted")
                                    .unwrap()
                                    .try_as_basic_value()
                                    .unwrap_left()
                                    .into_vector_value();
                                build
                                    .build_int_compare(predicate.as_int_pred(true), lhs, rhs, "cmp")
                                    .unwrap()
                            }
                            ComparisonType::String => unreachable!(),
                        };

                        build
                            .build_bit_cast(vec_res, ctx.bool_type(), "sing_vec_to_bool")
                            .unwrap()
                    }
                };

                Ok(res.as_basic_value_enum())
            }
            KernelExpression::And(lhs, rhs) => {
                let lhs_v = lhs
                    .compile(ctx, llvm_mod, build, bufs, llvm_types)?
                    .into_int_value();
                let rhs_v = rhs
                    .compile(ctx, llvm_mod, build, bufs, llvm_types)?
                    .into_int_value();
                Ok(build.build_and(lhs_v, rhs_v, "and").unwrap().into())
            }
            KernelExpression::Or(lhs, rhs) => {
                let lhs_v = lhs
                    .compile(ctx, llvm_mod, build, bufs, llvm_types)?
                    .into_int_value();
                let rhs_v = rhs
                    .compile(ctx, llvm_mod, build, bufs, llvm_types)?
                    .into_int_value();
                Ok(build.build_or(lhs_v, rhs_v, "or").unwrap().into())
            }
            KernelExpression::Select(cond, a, b) => {
                if cond.get_type() != DataType::Boolean {
                    return Err(DSLError::BooleanExpected(format!(
                        "first parameter to select should be a boolean, found {:?}",
                        cond.get_type()
                    )));
                }
                let cond_v = cond
                    .compile(ctx, llvm_mod, build, bufs, llvm_types)?
                    .into_int_value();
                let a_v = a.compile(ctx, llvm_mod, build, bufs, llvm_types)?;
                let b_v = b.compile(ctx, llvm_mod, build, bufs, llvm_types)?;

                if a_v.get_type() != b_v.get_type() {
                    return Err(DSLError::TypeMismatch(format!(
                        "select operands must have the same type (saw {} and {})",
                        a.get_type(),
                        b.get_type()
                    )));
                }

                Ok(build.build_select(cond_v, a_v, b_v, "selection").unwrap())
            }
        }
    }
}

pub struct KernelContext<'a> {
    inputs: Vec<KernelInput<'a>>,
}

impl<'a> KernelContext<'a> {
    pub fn get_input(&self, idx: usize) -> Result<KernelInput<'a>, DSLError> {
        self.inputs
            .get(idx)
            .cloned()
            .ok_or_else(|| DSLError::InvalidInputIndex(idx))
    }
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
struct KernelParameters {
    base_ptr: *const c_void,
    holder: Vec<*mut c_void>,
}

impl KernelParameters {
    pub fn new(holder: Vec<*mut c_void>) -> Self {
        let base_ptr = holder.as_ptr() as *const c_void;
        Self { holder, base_ptr }
    }

    pub fn get_mut_ptr(&mut self) -> *mut c_void {
        assert_eq!(self.base_ptr, self.holder.as_ptr() as *const c_void);
        self.base_ptr as *mut c_void
    }

    pub fn llvm_get<'a>(
        ctx: &'a Context,
        build: &Builder<'a>,
        base_ptr: PointerValue<'a>,
        idx: usize,
    ) -> PointerValue<'a> {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        build
            .build_load(
                ptr_type,
                increment_pointer!(ctx, build, base_ptr, 8 * idx),
                "inc_ptr",
            )
            .unwrap()
            .into_pointer_value()
    }
}

#[self_referencing]
pub struct DSLKernel {
    context: Context,
    input_types: Vec<DataType>,
    is_scalar: Vec<bool>,
    output_strategy: KernelOutputType,
    out_type: DataType,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void) -> u64>,
}

impl DSLKernel {
    pub fn compile<F: Fn(KernelContext) -> Result<KernelOutput, DSLError>>(
        inputs: &[&dyn Datum],
        f: F,
    ) -> Result<DSLKernel, DSLError> {
        let ctx = Context::create();
        let input_types = inputs
            .iter()
            .map(|x| x.get().0.data_type().clone())
            .collect_vec();
        let is_scalar = inputs.iter().map(|x| x.get().1).collect_vec();

        let context = KernelContext {
            inputs: inputs
                .iter()
                .enumerate()
                .map(|(idx, itm)| KernelInput::Datum(idx, *itm))
                .collect(),
        };
        let (out_strategy, expr) = f(context)?;
        let out_type = expr.get_type();

        DSLKernelTryBuilder {
            context: ctx,
            input_types,
            is_scalar,
            out_type,
            output_strategy: out_strategy,
            func_builder: |ctx| build_kernel(ctx, inputs, expr, out_strategy),
        }
        .try_build()
    }

    pub fn call(&self, inputs: &[&dyn Datum]) -> Result<ArrayRef, ArrowKernelError> {
        if inputs.len() != self.borrow_input_types().len() {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "expected {} arguments, got {}",
                self.borrow_input_types().len(),
                inputs.len()
            )));
        }
        for (idx, ((input, expected_type), expected_scalar)) in inputs
            .iter()
            .zip(self.borrow_input_types().iter())
            .zip(self.borrow_is_scalar().iter())
            .enumerate()
        {
            if input.get().0.data_type() != expected_type {
                return Err(ArrowKernelError::ArgumentMismatch(format!(
                    "for argument {}, expected type: {}, got: {}",
                    idx,
                    expected_type,
                    input.get().0.data_type()
                )));
            }
            if input.get().1 != *expected_scalar {
                return Err(ArrowKernelError::ArgumentMismatch(format!(
                    "for argument {}, expected scalar: {}, got: {}",
                    idx,
                    expected_scalar,
                    input.get().1
                )));
            }
        }

        let max_len = inputs
            .iter()
            .map(|&input| input.get().0.len())
            .max()
            .unwrap_or(0);

        let mut ihs: Vec<IteratorHolder> = inputs
            .iter()
            .map(|&input| datum_to_iter(input))
            .try_collect()?;
        let mut ptrs = Vec::new();
        ptrs.extend(ihs.iter_mut().map(|ih| ih.get_mut_ptr()));

        match self.borrow_output_strategy() {
            KernelOutputType::Array => {
                let width = PrimitiveType::for_arrow_type(self.borrow_out_type()).width();
                let mut out_buf = vec![0_u8; max_len * width];
                ptrs.push(out_buf.as_mut_ptr() as *mut c_void);
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().call(kp.get_mut_ptr()) };

                let res = unsafe {
                    make_array(
                        ArrayDataBuilder::new(self.borrow_out_type().clone())
                            .nulls(None)
                            .buffers(vec![Buffer::from(out_buf)])
                            .len(max_len)
                            .build_unchecked(),
                    )
                };
                Ok(res.slice(0, num_results as usize))
            }
            KernelOutputType::Boolean => {
                let mut out_buf = vec![0_u8; max_len.div_ceil(8)];
                ptrs.push(out_buf.as_mut_ptr() as *mut c_void);
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().call(kp.get_mut_ptr()) };

                let out_buf = BooleanBuffer::new(out_buf.into(), 0, num_results as usize);
                let res = BooleanArray::new(out_buf.into(), None);
                Ok(Arc::new(res) as ArrayRef)
            }
            KernelOutputType::Dictionary => todo!(),
            KernelOutputType::RunEnd => todo!(),
        }
    }
}

fn build_kernel<'a>(
    ctx: &'a Context,
    inputs: &[&dyn Datum],
    expr: KernelExpression,
    out_strategy: KernelOutputType,
) -> Result<JitFunction<'a, unsafe extern "C" fn(*mut c_void) -> u64>, DSLError> {
    let llvm_mod = ctx.create_module("kernel");
    let builder = ctx.create_builder();
    let i64_type = ctx.i64_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());

    let addl_parameters = match out_strategy {
        KernelOutputType::Array => 1,
        KernelOutputType::Boolean => 1,
        KernelOutputType::RunEnd => 2,
        KernelOutputType::Dictionary => 3,
    };
    let func_type = i64_type.fn_type(&[ptr_type.into()], false);
    let func = llvm_mod.add_function("kernel", func_type, None);

    let out_type = expr.get_type();
    let out_prim_type = PrimitiveType::for_arrow_type(&out_type);

    let indexes_to_iter = expr.iterated_indexes();
    let mut next_funcs = Vec::new();
    for idx in indexes_to_iter.iter() {
        let ih = datum_to_iter(inputs[*idx]).unwrap();
        next_funcs.push(
            generate_next(
                &ctx,
                &llvm_mod,
                &format!("iter{}", idx),
                inputs[*idx].get().0.data_type(),
                &ih,
            )
            .unwrap(),
        );
    }

    let mut iter_llvm_types = Vec::new();
    for idx in indexes_to_iter.iter() {
        let ptype = PrimitiveType::for_arrow_type(inputs[*idx].get().0.data_type());
        iter_llvm_types.push(ptype.llvm_type(&ctx));
    }

    declare_blocks!(ctx, func, entry, loop_cond, loop_body, exit);
    builder.position_at_end(entry);
    let param_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let iter_ptrs = (0..inputs.len())
        .map(|i| KernelParameters::llvm_get(ctx, &builder, param_ptr, i))
        .collect_vec();
    let out_ptrs = (inputs.len()..inputs.len() + addl_parameters)
        .map(|i| KernelParameters::llvm_get(ctx, &builder, param_ptr, i))
        .collect_vec();
    println!(
        "Num inputs: {} Num outputs: {}",
        inputs.len(),
        out_ptrs.len()
    );

    let writer = match out_strategy {
        KernelOutputType::Array => Box::new(PrimitiveArrayWriter::allocate_array_writer(
            ctx,
            &llvm_mod,
            &builder,
            out_ptrs[0],
            out_prim_type,
        )) as Box<dyn ArrayWriter>,
        KernelOutputType::Boolean => Box::new(BooleanWriter::allocate_boolean_writer(
            &ctx,
            &llvm_mod,
            &builder,
            out_ptrs[0],
        )) as Box<dyn ArrayWriter>,
        KernelOutputType::Dictionary => todo!(),
        KernelOutputType::RunEnd => todo!(),
    };

    let bufs = indexes_to_iter
        .iter()
        .zip(iter_llvm_types.iter())
        .map(|(idx, llvm_type)| {
            builder
                .build_alloca(*llvm_type, &format!("buf{}", idx))
                .unwrap()
        })
        .collect_vec();

    let produced_ptr = builder.build_alloca(i64_type, "out_count").unwrap();
    builder
        .build_store(produced_ptr, i64_type.const_zero())
        .unwrap();

    builder.build_unconditional_branch(loop_cond).unwrap();

    builder.position_at_end(loop_cond);
    let mut had_nexts = Vec::new();
    // call next on each of the iterators that we care about (note the
    // difference between the index of the iterator we care about and the index
    // of the parameter value)
    for (iter_idx, param_idx) in indexes_to_iter.iter().enumerate() {
        let buf = bufs[iter_idx];
        let next_func = next_funcs[iter_idx];
        let iter_ptr = iter_ptrs[*param_idx];
        had_nexts.push(
            builder
                .build_call(
                    next_func,
                    &[iter_ptr.into(), buf.into()],
                    &format!("next{}", param_idx),
                )
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value(),
        );
    }

    // AND-together all has-nexts
    let mut accum = had_nexts.pop().unwrap();
    for val in had_nexts {
        accum = builder.build_and(accum, val, "accum").unwrap();
    }
    builder
        .build_conditional_branch(accum, loop_body, exit)
        .unwrap();

    builder.position_at_end(loop_body);
    let result = expr.compile(&ctx, &llvm_mod, &builder, &bufs, &iter_llvm_types)?;
    writer.ingest(&ctx, &builder, result);
    let curr_produced = builder
        .build_load(i64_type, produced_ptr, "curr_produced")
        .unwrap()
        .into_int_value();
    let new_produced = builder
        .build_int_add(curr_produced, i64_type.const_int(1, false), "new_produced")
        .unwrap();
    builder.build_store(produced_ptr, new_produced).unwrap();
    builder.build_unconditional_branch(loop_cond).unwrap();

    builder.position_at_end(exit);
    writer.flush(&ctx, &builder);
    let produced = builder
        .build_load(i64_type, produced_ptr, "produced")
        .unwrap()
        .into_int_value();
    builder.build_return(Some(&produced)).unwrap();

    llvm_mod.print_to_stderr();
    llvm_mod.verify().unwrap();
    optimize_module(&llvm_mod).unwrap();
    let ee = llvm_mod
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    Ok(unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void) -> u64>(
            func.get_name().to_str().unwrap(),
        )
        .unwrap()
    })
}

#[cfg(test)]
mod test {

    use arrow_array::{cast::AsArray, types::Int32Type, BooleanArray, Int32Array, StringArray};

    use crate::new_kernels::dsl::DSLKernel;

    use super::KernelOutputType;

    #[test]
    fn test_dsl_int_between() {
        let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let sca1 = Int32Array::new_scalar(1);
        let sca2 = Int32Array::new_scalar(6);

        let k = DSLKernel::compile(&[&data, &sca1, &sca2], |ctx| {
            let data = ctx.get_input(0)?;
            let scalar1 = ctx.get_input(1)?;
            let scalar2 = ctx.get_input(2)?;

            data.into_iter()
                .zip(scalar1.into_iter())
                .zip(scalar2.into_iter())
                .map(|i| vec![i[0].gt(&i[1]), i[0].lt(&i[2])])
                .map(|i| vec![i[0].and(&i[1])])
                .collect(KernelOutputType::Boolean)
        })
        .unwrap();

        let res = k.call(&[&data, &sca1, &sca2]);
        let res = res.unwrap();

        assert_eq!(
            res.as_boolean(),
            &BooleanArray::from(vec![false, true, true, true, true, false])
        );
    }

    #[test]
    fn test_dsl_str_between() {
        let data = StringArray::from(vec!["a", "b", "c", "d", "e", "f"]);
        let sca1 = StringArray::new_scalar("a");
        let sca2 = StringArray::new_scalar("f");

        let k = DSLKernel::compile(&[&data, &sca1, &sca2], |ctx| {
            let data = ctx.get_input(0)?;
            let scalar1 = ctx.get_input(1)?;
            let scalar2 = ctx.get_input(2)?;

            data.into_iter()
                .zip(scalar1.into_iter())
                .zip(scalar2.into_iter())
                .map(|i| vec![i[0].gt(&i[1]), i[0].lt(&i[2])])
                .map(|i| vec![i[0].and(&i[1])])
                .collect(KernelOutputType::Boolean)
        })
        .unwrap();

        let res = k.call(&[&data, &sca1, &sca2]);
        let res = res.unwrap();

        assert_eq!(
            res.as_boolean(),
            &BooleanArray::from(vec![false, true, true, true, true, false])
        );
    }

    #[test]
    fn test_dsl_int_max() {
        let data1 = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let data2 = Int32Array::from(vec![-1, 20, -3, 40, -5, 1]);

        let k = DSLKernel::compile(&[&data1, &data2], |ctx| {
            let lhs = ctx.get_input(0)?;
            let rhs = ctx.get_input(1)?;

            lhs.into_iter()
                .zip(rhs.into_iter())
                .map(|i| vec![i[0].gt(&i[1]).select(&i[0], &i[1])])
                .collect(KernelOutputType::Array)
        })
        .unwrap();

        let res = k.call(&[&data1, &data2]).unwrap();

        assert_eq!(
            res.as_primitive::<Int32Type>(),
            &Int32Array::from(vec![1, 20, 3, 40, 5, 6])
        );
    }
}
