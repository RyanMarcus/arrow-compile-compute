use std::{
    collections::{BTreeSet, HashSet},
    fmt::Debug,
    slice::RChunks,
};

use arrow_array::{Array, BooleanArray, Datum};
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    types::VectorType,
    values::{BasicValue, BasicValueEnum, PointerValue},
    AddressSpace,
};
use itertools::Itertools;

use crate::{
    declare_blocks, increment_pointer,
    new_iter::{datum_to_iter, generate_next, generate_next_block},
    new_kernels::{
        cmp::{add_float_vec_to_int_vec, add_memcmp},
        gen_convert_numeric_vec,
    },
    ComparisonType, Predicate, PrimitiveType,
};

#[derive(Debug)]
pub enum DSLError {
    InvalidInputIndex(usize),
    InvalidKernelOutputLength(usize),
    TypeMismatch(PrimitiveType, PrimitiveType, String),
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

#[derive(Debug)]
pub enum KernelOutputType {
    Array,
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
}

impl<'a> KernelExpression<'a> {
    pub fn cmp(&self, pred: Predicate, other: &KernelExpression<'a>) -> KernelExpression<'a> {
        KernelExpression::Cmp(pred, Box::new(self.clone()), Box::new(other.clone()))
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
    pub fn truncate(&self, size: usize) -> KernelExpression<'a> {
        KernelExpression::Truncate(Box::new(self.clone()), size)
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
        llvm_types: &[VectorType<'b>],
    ) -> Result<BasicValueEnum<'b>, DSLError> {
        match self {
            KernelExpression::Item(kernel_input) => {
                let buf = bufs[kernel_input.index()];
                let llvm_type = llvm_types[kernel_input.index()];
                Ok(build.build_load(llvm_type, buf, "load").unwrap())
            }
            KernelExpression::Truncate(_kernel_expression, _) => todo!(),
            KernelExpression::Cmp(predicate, lhs, rhs) => {
                let lhs_v = lhs
                    .compile(ctx, llvm_mod, build, bufs, llvm_types)?
                    .into_vector_value();
                let rhs_v = rhs
                    .compile(ctx, llvm_mod, build, bufs, llvm_types)?
                    .into_vector_value();

                let lhs_ptype = PrimitiveType::for_arrow_type(&lhs.get_type());
                let rhs_ptype = PrimitiveType::for_arrow_type(&rhs.get_type());

                let res = match (lhs_ptype.comparison_type(), rhs_ptype.comparison_type()) {
                    (ComparisonType::String, ComparisonType::String) => {
                        let memcmp = add_memcmp(ctx, llvm_mod);
                        todo!()
                    }
                    (ComparisonType::String, _) | (_, ComparisonType::String) => {
                        return Err(DSLError::TypeMismatch(
                            lhs_ptype,
                            rhs_ptype,
                            "cannot compare these types".to_string(),
                        ));
                    }
                    _ => {
                        let dom_type = PrimitiveType::dominant(lhs_ptype, rhs_ptype).unwrap();
                        let clhs = gen_convert_numeric_vec(ctx, build, lhs_v, lhs_ptype, dom_type);
                        let crhs = gen_convert_numeric_vec(ctx, build, rhs_v, rhs_ptype, dom_type);

                        match dom_type.comparison_type() {
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
                        }
                    }
                };

                Ok(res.as_basic_value_enum())
            }
            KernelExpression::And(lhs, rhs) => {
                let lhs_v = lhs
                    .compile(ctx, llvm_mod, build, bufs, llvm_types)?
                    .into_vector_value();
                let rhs_v = rhs
                    .compile(ctx, llvm_mod, build, bufs, llvm_types)?
                    .into_vector_value();
                Ok(build.build_and(lhs_v, rhs_v, "and").unwrap().into())
            }
            KernelExpression::Or(lhs, rhs) => {
                let lhs_v = lhs
                    .compile(ctx, llvm_mod, build, bufs, llvm_types)?
                    .into_vector_value();
                let rhs_v = rhs
                    .compile(ctx, llvm_mod, build, bufs, llvm_types)?
                    .into_vector_value();
                Ok(build.build_or(lhs_v, rhs_v, "and").unwrap().into())
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

fn build_kernel<'a, F: Fn(KernelContext) -> Result<KernelOutput, DSLError>>(
    inputs: &[&'a dyn Datum],
    f: F,
) -> Result<(), DSLError> {
    let context = KernelContext {
        inputs: inputs
            .iter()
            .enumerate()
            .map(|(idx, itm)| KernelInput::Datum(idx, *itm))
            .collect(),
    };
    let (out_strategy, expr) = f(context)?;

    let ctx = Context::create();
    let llvm_mod = ctx.create_module("kernel");
    let builder = ctx.create_builder();
    let i64_type = ctx.i64_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());

    let addl_parameters = match out_strategy {
        KernelOutputType::Array => 1,
        KernelOutputType::RunEnd => 2,
        KernelOutputType::Dictionary => 3,
    };
    let func_type = i64_type.fn_type(
        &vec![ptr_type.into(); inputs.len() + addl_parameters],
        false,
    );
    let func = llvm_mod.add_function("kernel", func_type, None);

    let out_type = expr.get_type();
    let out_prim_type = PrimitiveType::for_arrow_type(&out_type);
    declare_blocks!(ctx, func, entry, loop_cond, loop_body, tail_cond, tail_body, exit);
    builder.position_at_end(entry);
    let out_idx_ptr = builder.build_alloca(i64_type, "out_idx_ptr").unwrap();
    builder
        .build_store(out_idx_ptr, i64_type.const_zero())
        .unwrap();
    let produced_ptr = builder.build_alloca(i64_type, "produced_ptr").unwrap();
    builder
        .build_store(produced_ptr, i64_type.const_zero())
        .unwrap();
    let out_ptrs = (inputs.len()..inputs.len() + addl_parameters)
        .map(|i| func.get_nth_param(i as u32).unwrap().into_pointer_value())
        .collect_vec();

    let indexes_to_iter = expr.iterated_indexes();
    let next_block_funcs = indexes_to_iter
        .iter()
        .map(|idx| {
            let ih = datum_to_iter(inputs[*idx]).unwrap();
            generate_next_block::<64>(
                &ctx,
                &llvm_mod,
                &format!("iter{}", idx),
                inputs[*idx].get().0.data_type(),
                &ih,
            )
            .unwrap()
        })
        .collect_vec();
    let next_funcs = indexes_to_iter
        .iter()
        .map(|idx| {
            let ih = datum_to_iter(inputs[*idx]).unwrap();
            generate_next(
                &ctx,
                &llvm_mod,
                &format!("iter{}", idx),
                inputs[*idx].get().0.data_type(),
                &ih,
            )
            .unwrap()
        })
        .collect_vec();
    let iter_llvm_vec_types = indexes_to_iter
        .iter()
        .map(|idx| {
            let ptype = PrimitiveType::for_arrow_type(inputs[*idx].get().0.data_type());
            ptype.llvm_vec_type(&ctx, 64).unwrap()
        })
        .collect_vec();
    let vbufs = indexes_to_iter
        .iter()
        .zip(iter_llvm_vec_types.iter())
        .map(|(idx, llvm_type)| {
            builder
                .build_alloca(*llvm_type, &format!("vbuf{}", idx))
                .unwrap()
        })
        .collect_vec();
    let iter_llvm_types = indexes_to_iter
        .iter()
        .map(|idx| {
            let ptype = PrimitiveType::for_arrow_type(inputs[*idx].get().0.data_type());
            ptype.llvm_type(&ctx)
        })
        .collect_vec();
    let bufs = indexes_to_iter
        .iter()
        .zip(iter_llvm_types.iter())
        .map(|(idx, llvm_type)| {
            builder
                .build_alloca(*llvm_type, &format!("buf{}", idx))
                .unwrap()
        })
        .collect_vec();

    let iter_ptrs = inputs
        .iter()
        .enumerate()
        .map(|(idx, _input)| func.get_nth_param(idx as u32).unwrap().into_pointer_value())
        .collect_vec();
    let out_count_ptr = builder.build_alloca(i64_type, "out_count").unwrap();
    builder
        .build_store(out_count_ptr, i64_type.const_zero())
        .unwrap();

    builder.build_unconditional_branch(loop_cond).unwrap();

    builder.position_at_end(loop_cond);
    let mut had_nexts = Vec::new();
    // call next on each of the iterators that we care about (note the
    // difference between the index of the iterator we care about and the index
    // of the parameter value)
    for (iter_idx, param_idx) in indexes_to_iter.iter().enumerate() {
        let buf = vbufs[iter_idx];
        let iter_ptr = iter_ptrs[*param_idx];
        let next_func = next_block_funcs[iter_idx];
        had_nexts.push(
            builder
                .build_call(
                    next_func,
                    &[buf.into(), iter_ptr.into()],
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
        .build_conditional_branch(accum, loop_body, tail_cond)
        .unwrap();

    builder.position_at_end(loop_body);
    let chunk = expr.compile(&ctx, &llvm_mod, &builder, &vbufs, &iter_llvm_vec_types)?;
    let out_idx = builder
        .build_load(i64_type, out_idx_ptr, "out_idx")
        .unwrap()
        .into_int_value();
    let produced = builder
        .build_load(i64_type, produced_ptr, "produced")
        .unwrap()
        .into_int_value();

    match out_strategy {
        KernelOutputType::Array => {
            let dst_ptr =
                increment_pointer!(ctx, builder, out_ptrs[0], out_prim_type.width(), out_idx);
            builder.build_store(dst_ptr, chunk).unwrap();
            let new_out_idx = builder
                .build_int_add(out_idx, i64_type.const_int(64, false), "new_out_idx")
                .unwrap();
            builder.build_store(out_idx_ptr, new_out_idx).unwrap();
            let new_produced = builder
                .build_int_add(produced, i64_type.const_int(64, false), "new_produced")
                .unwrap();
            builder.build_store(produced_ptr, new_produced).unwrap();
        }
        KernelOutputType::Dictionary => todo!(),
        KernelOutputType::RunEnd => todo!(),
    };

    builder.build_unconditional_branch(loop_cond).unwrap();

    builder.position_at_end(tail_cond);
    let mut had_nexts = Vec::new();
    // call next (single) on each of the iterators
    for (iter_idx, param_idx) in indexes_to_iter.iter().enumerate() {
        let buf = bufs[iter_idx];
        let iter_ptr = iter_ptrs[*param_idx];
        let next_func = next_funcs[iter_idx];
        had_nexts.push(
            builder
                .build_call(
                    next_func,
                    &[buf.into(), iter_ptr.into()],
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
        .build_conditional_branch(accum, tail_body, exit)
        .unwrap();

    llvm_mod.print_to_stderr();
    Ok(())
}

#[cfg(test)]
mod test {
    use arrow_array::Int32Array;

    use super::{build_kernel, KernelOutputType};

    #[test]
    fn test_dsl() {
        let data = Int32Array::from(vec![1, 2, 3]);
        let sca = Int32Array::new_scalar(2);

        let k = build_kernel(&[&data, &sca], |ctx| {
            let data = ctx.get_input(0)?;
            let scalar = ctx.get_input(1)?;

            data.into_iter()
                .zip(scalar.into_iter())
                .map(|i| vec![i[0].eq(&i[1])])
                .collect(KernelOutputType::Array)
        })
        .unwrap();

        println!("{:?}", k);
        todo!()
    }
}
