use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::c_void;
use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{
        ArrowDictionaryKeyType, Int16Type, Int32Type, Int64Type, Int8Type, RunEndIndexType,
        UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrayRef, Datum, DictionaryArray, RunArray, StringArray,
};
use arrow_schema::DataType;
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    module::Linkage,
    types::BasicType,
    values::{BasicValue, FunctionValue, PointerValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};
use itertools::Itertools;
use ouroboros::self_referencing;
use repr_offset::ReprOffset;

use super::{
    context::{CompilationContext, VEC_SIZE},
    errors::DSLError,
    expressions::KernelExpression,
    types::{DictKeyType, KernelInput, KernelInputType, KernelOutputType, RunEndType},
};
use crate::{
    compiled_iter::{
        array_to_setbit_iter, datum_to_iter, generate_blocked_random_access, generate_next,
        generate_next_block, generate_random_access, IteratorHolder,
    },
    compiled_kernels::{link_req_helpers, optimize_module, ArrowKernelError},
    compiled_writers::{
        ArrayWriter, BooleanWriter, DictWriter, FixedSizeListWriter, PrimitiveArrayWriter,
        REEWriter, StringArrayWriter, StringViewWriter, WriterAllocation,
    },
    declare_blocks, increment_pointer, set_noalias_params, PrimitiveType,
};

pub struct KernelContext<'a> {
    inputs: Vec<KernelInput<'a>>,
}

impl<'a> KernelContext<'a> {
    pub fn get_input(&self, idx: usize) -> Result<KernelInput<'a>, DSLError> {
        self.inputs
            .get(idx)
            .cloned()
            .ok_or(DSLError::InvalidInputIndex(idx))
    }

    pub fn iter_over(&self, inputs: Vec<KernelInput<'a>>) -> BaseKernelProgram<'a> {
        BaseKernelProgram { inputs }
    }
}

pub struct BaseKernelProgram<'a> {
    inputs: Vec<KernelInput<'a>>,
}

#[allow(dead_code)]
impl<'a> BaseKernelProgram<'a> {
    pub fn map<F: Fn(&[KernelExpression<'a>]) -> Vec<KernelExpression<'a>>>(
        self,
        f: F,
    ) -> MappedKernelProgram<'a> {
        let exprs = self.inputs.iter().map(|inp| inp.as_expr()).collect_vec();
        MappedKernelProgram {
            inputs: self.inputs,
            expr: f(&exprs),
        }
    }

    pub fn filter<F: Fn(&[KernelExpression<'a>]) -> KernelExpression<'a>>(
        self,
        f: F,
    ) -> FilteredKernelProgram<'a> {
        let exprs = self.inputs.iter().map(|inp| inp.as_expr()).collect_vec();
        FilteredKernelProgram {
            inputs: self.inputs,
            cond: f(&exprs),
        }
    }

    pub fn collect(self, strategy: KernelOutputType) -> Result<SealedKernelProgram<'a>, DSLError> {
        SealedKernelProgram::try_new(self.inputs, None, None, strategy)
    }
}

pub struct FilteredKernelProgram<'a> {
    inputs: Vec<KernelInput<'a>>,
    cond: KernelExpression<'a>,
}

#[allow(dead_code)]
impl<'a> FilteredKernelProgram<'a> {
    pub fn map<F: Fn(&[KernelExpression<'a>]) -> Vec<KernelExpression<'a>>>(
        self,
        f: F,
    ) -> FilterMappedKernelProgram<'a> {
        let exprs = self.inputs.iter().map(|inp| inp.as_expr()).collect_vec();
        FilterMappedKernelProgram {
            inputs: self.inputs,
            cond: self.cond,
            expr: f(&exprs),
        }
    }

    pub fn collect(self, strategy: KernelOutputType) -> Result<SealedKernelProgram<'a>, DSLError> {
        SealedKernelProgram::try_new(self.inputs, Some(self.cond), None, strategy)
    }
}

pub struct MappedKernelProgram<'a> {
    inputs: Vec<KernelInput<'a>>,
    expr: Vec<KernelExpression<'a>>,
}

#[allow(dead_code)]
impl<'a> MappedKernelProgram<'a> {
    pub fn map<F: Fn(&[KernelExpression<'a>]) -> Vec<KernelExpression<'a>>>(
        self,
        f: F,
    ) -> MappedKernelProgram<'a> {
        MappedKernelProgram {
            inputs: self.inputs,
            expr: f(&self.expr),
        }
    }

    pub fn collect(self, strategy: KernelOutputType) -> Result<SealedKernelProgram<'a>, DSLError> {
        SealedKernelProgram::try_new(self.inputs, None, Some(self.expr), strategy)
    }
}

pub struct FilterMappedKernelProgram<'a> {
    inputs: Vec<KernelInput<'a>>,
    cond: KernelExpression<'a>,
    expr: Vec<KernelExpression<'a>>,
}

#[allow(dead_code)]
impl<'a> FilterMappedKernelProgram<'a> {
    pub fn map<F: Fn(&[KernelExpression<'a>]) -> Vec<KernelExpression<'a>>>(
        self,
        f: F,
    ) -> FilterMappedKernelProgram<'a> {
        FilterMappedKernelProgram {
            inputs: self.inputs,
            cond: self.cond,
            expr: f(&self.expr),
        }
    }

    pub fn collect(self, strategy: KernelOutputType) -> Result<SealedKernelProgram<'a>, DSLError> {
        SealedKernelProgram::try_new(self.inputs, Some(self.cond), Some(self.expr), strategy)
    }
}

pub struct SealedKernelProgram<'a> {
    _inputs: Vec<KernelInput<'a>>,
    cond: Option<KernelExpression<'a>>,
    expr: KernelExpression<'a>,
    strategy: KernelOutputType,
    out_type: DataType,
}

impl<'a> SealedKernelProgram<'a> {
    pub fn try_new(
        inputs: Vec<KernelInput<'a>>,
        cond: Option<KernelExpression<'a>>,
        expr: Option<Vec<KernelExpression<'a>>>,
        strategy: KernelOutputType,
    ) -> Result<Self, DSLError> {
        let expr = if let Some(mut expr) = expr {
            if expr.len() != 1 {
                return Err(DSLError::InvalidKernelOutputLength(expr.len()));
            }
            expr.pop().unwrap()
        } else {
            if inputs.len() != 1 {
                return Err(DSLError::InvalidKernelOutputLength(inputs.len()));
            }
            inputs[0].as_expr()
        };

        if let Some(cond) = &cond {
            if cond.get_type() != DataType::Boolean {
                return Err(DSLError::BooleanExpected(format!(
                    "filter should have a boolean type, found {}",
                    cond.get_type()
                )));
            }
        }

        let out_type = expr.get_type();
        let res = Self {
            _inputs: inputs,
            cond,
            expr,
            strategy,
            out_type,
        };
        res.strategy().can_collect(res.out_type())?;

        Ok(res)
    }

    pub fn out_type(&self) -> &DataType {
        &self.out_type
    }

    pub fn strategy(&self) -> KernelOutputType {
        self.strategy
    }

    fn iterated_indexes(&self) -> Vec<(KernelInputType, usize)> {
        let mut set = BTreeSet::new();
        if let Some(cond) = &self.cond {
            set.extend(cond.iterated_indexes());
        }
        set.extend(self.expr.iterated_indexes());
        set.into_iter().collect()
    }

    pub fn accessed_indexes(&self) -> Vec<usize> {
        let mut set = BTreeSet::new();
        if let Some(cond) = &self.cond {
            set.extend(cond.accessed_indexes());
        }
        set.extend(self.expr.accessed_indexes());

        set.into_iter().collect()
    }

    pub fn expr(&self) -> &KernelExpression<'a> {
        &self.expr
    }

    pub fn filter(&self) -> Option<&KernelExpression<'a>> {
        self.cond.as_ref()
    }
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct KernelParameters {
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
        let res = build
            .build_load(
                ptr_type,
                increment_pointer!(ctx, build, base_ptr, 8 * idx),
                "inc_ptr",
            )
            .unwrap()
            .into_pointer_value();
        res.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        res
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
    func: (
        Vec<KernelInputType>,
        JitFunction<'this, unsafe extern "C" fn(*mut c_void) -> u64>,
    ),
}

impl DSLKernel {
    pub fn compile<F: Fn(KernelContext) -> Result<SealedKernelProgram, DSLError>>(
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
        let program = f(context)?;
        let out_type = program.out_type().clone();
        let output_strategy = program.strategy();

        DSLKernelTryBuilder {
            context: ctx,
            input_types,
            is_scalar,
            out_type,
            output_strategy,
            func_builder: |ctx| build_kernel(ctx, inputs, program),
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
            .zip(self.borrow_func().0.iter())
            .map(|(&input, ty)| match ty {
                KernelInputType::Standard => datum_to_iter(input),
                KernelInputType::SetBit => array_to_setbit_iter(input.get().0.as_boolean()),
            })
            .try_collect()?;
        let mut ptrs = Vec::new();
        ptrs.extend(ihs.iter_mut().map(|ih| ih.get_mut_ptr()));
        let p_out_type = PrimitiveType::for_arrow_type(self.borrow_out_type());

        match self.borrow_output_strategy() {
            KernelOutputType::Array => {
                let mut alloc = PrimitiveArrayWriter::allocate(max_len, p_out_type);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) } as usize;
                Ok(alloc.to_array(num_results, None))
            }
            KernelOutputType::String => {
                let mut alloc = StringArrayWriter::<i32>::allocate(max_len, p_out_type);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) } as usize;
                let arr = alloc.to_array(num_results, None);

                let data = unsafe {
                    arr.into_data()
                        .into_builder()
                        .data_type(DataType::Utf8)
                        .build_unchecked()
                };

                Ok(Arc::new(StringArray::from(data)))
            }
            KernelOutputType::Boolean => {
                let mut alloc = BooleanWriter::allocate(max_len, p_out_type);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                Ok(Arc::new(alloc.to_array(num_results as usize, None)))
            }
            KernelOutputType::View => {
                let mut alloc = StringViewWriter::allocate(max_len, p_out_type);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                Ok(Arc::new(alloc.to_array(num_results as usize, None)))
            }
            KernelOutputType::Dictionary(key) => Ok(match key {
                DictKeyType::Int8 => {
                    Arc::new(self.exec_to_dict::<Int8Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::Int16 => {
                    Arc::new(self.exec_to_dict::<Int16Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::Int32 => {
                    Arc::new(self.exec_to_dict::<Int32Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::Int64 => {
                    Arc::new(self.exec_to_dict::<Int64Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::UInt8 => {
                    Arc::new(self.exec_to_dict::<UInt8Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::UInt16 => {
                    Arc::new(self.exec_to_dict::<UInt16Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::UInt32 => {
                    Arc::new(self.exec_to_dict::<UInt32Type>(ptrs, max_len, p_out_type))
                }
                DictKeyType::UInt64 => {
                    Arc::new(self.exec_to_dict::<UInt64Type>(ptrs, max_len, p_out_type))
                }
            }),
            KernelOutputType::RunEnd(re_type) => Ok(match re_type {
                RunEndType::Int16 => {
                    Arc::new(self.exec_to_ree::<Int16Type>(ptrs, max_len, p_out_type))
                }
                RunEndType::Int32 => {
                    Arc::new(self.exec_to_ree::<Int32Type>(ptrs, max_len, p_out_type))
                }
                RunEndType::Int64 => {
                    Arc::new(self.exec_to_ree::<Int64Type>(ptrs, max_len, p_out_type))
                }
            }),
            KernelOutputType::FixedSizeList(_s) => {
                let mut alloc = FixedSizeListWriter::allocate(max_len, p_out_type);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) } as usize;
                Ok(alloc.to_array_ref(num_results, None))
            }
        }
    }

    fn exec_to_dict<K: ArrowDictionaryKeyType>(
        &self,
        mut ptrs: Vec<*mut c_void>,
        max_len: usize,
        pt: PrimitiveType,
    ) -> DictionaryArray<K> {
        match pt {
            PrimitiveType::P64x2 => {
                let mut alloc = DictWriter::<K, StringArrayWriter<i32>>::allocate(max_len, pt);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                alloc.to_array(num_results as usize, None)
            }
            _ => {
                let mut alloc = DictWriter::<K, PrimitiveArrayWriter>::allocate(max_len, pt);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                alloc.to_array(num_results as usize, None)
            }
        }
    }

    fn exec_to_ree<K: RunEndIndexType>(
        &self,
        mut ptrs: Vec<*mut c_void>,
        max_len: usize,
        pt: PrimitiveType,
    ) -> RunArray<K> {
        match pt {
            PrimitiveType::P64x2 => {
                let mut alloc = REEWriter::<K, StringArrayWriter<i32>>::allocate(max_len, pt);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                alloc.to_array(num_results as usize, None)
            }
            _ => {
                let mut alloc = REEWriter::<K, PrimitiveArrayWriter>::allocate(max_len, pt);
                ptrs.push(alloc.get_ptr());
                let mut kp = KernelParameters::new(ptrs);

                let num_results = unsafe { self.borrow_func().1.call(kp.get_mut_ptr()) };
                alloc.to_array(num_results as usize, None)
            }
        }
    }
}

fn build_kernel<'a>(
    ctx: &'a Context,
    inputs: &[&dyn Datum],
    program: SealedKernelProgram<'_>,
) -> Result<
    (
        Vec<KernelInputType>,
        JitFunction<'a, unsafe extern "C" fn(*mut c_void) -> u64>,
    ),
    DSLError,
> {
    match program.strategy() {
        KernelOutputType::Array => {
            build_kernel_with_writer::<PrimitiveArrayWriter>(ctx, inputs, program)
        }
        KernelOutputType::String => {
            build_kernel_with_writer::<StringArrayWriter<i32>>(ctx, inputs, program)
        }
        KernelOutputType::Boolean => {
            build_kernel_with_writer::<BooleanWriter>(ctx, inputs, program)
        }
        KernelOutputType::View => {
            build_kernel_with_writer::<StringViewWriter>(ctx, inputs, program)
        }
        KernelOutputType::FixedSizeList(_s) => {
            build_kernel_with_writer::<FixedSizeListWriter>(ctx, inputs, program)
        }
        KernelOutputType::Dictionary(key) => match key {
            DictKeyType::Int8 => build_dict_kernel::<Int8Type>(ctx, inputs, program),
            DictKeyType::Int16 => build_dict_kernel::<Int16Type>(ctx, inputs, program),
            DictKeyType::Int32 => build_dict_kernel::<Int32Type>(ctx, inputs, program),
            DictKeyType::Int64 => build_dict_kernel::<Int64Type>(ctx, inputs, program),
            DictKeyType::UInt8 => build_dict_kernel::<UInt8Type>(ctx, inputs, program),
            DictKeyType::UInt16 => build_dict_kernel::<UInt16Type>(ctx, inputs, program),
            DictKeyType::UInt32 => build_dict_kernel::<UInt32Type>(ctx, inputs, program),
            DictKeyType::UInt64 => build_dict_kernel::<UInt64Type>(ctx, inputs, program),
        },
        KernelOutputType::RunEnd(key) => match key {
            RunEndType::Int16 => build_ree_kernel::<Int16Type>(ctx, inputs, program),
            RunEndType::Int32 => build_ree_kernel::<Int32Type>(ctx, inputs, program),
            RunEndType::Int64 => build_ree_kernel::<Int64Type>(ctx, inputs, program),
        },
    }
}

fn build_dict_kernel<'a, T: ArrowDictionaryKeyType>(
    ctx: &'a Context,
    inputs: &[&dyn Datum],
    program: SealedKernelProgram<'_>,
) -> Result<
    (
        Vec<KernelInputType>,
        JitFunction<'a, unsafe extern "C" fn(*mut c_void) -> u64>,
    ),
    DSLError,
> {
    if program.out_type().is_primitive() {
        build_kernel_with_writer::<DictWriter<T, PrimitiveArrayWriter>>(ctx, inputs, program)
    } else {
        match program.out_type() {
            DataType::Binary | DataType::Utf8 => build_kernel_with_writer::<
                DictWriter<T, StringArrayWriter<i32>>,
            >(ctx, inputs, program),
            DataType::LargeBinary | DataType::LargeUtf8 => build_kernel_with_writer::<
                DictWriter<T, StringArrayWriter<i64>>,
            >(ctx, inputs, program),
            _ => Err(DSLError::UnsupportedDictionaryValueType(
                program.out_type().clone(),
            )),
        }
    }
}

fn build_ree_kernel<'a, T: RunEndIndexType>(
    ctx: &'a Context,
    inputs: &[&dyn Datum],
    program: SealedKernelProgram<'_>,
) -> Result<
    (
        Vec<KernelInputType>,
        JitFunction<'a, unsafe extern "C" fn(*mut c_void) -> u64>,
    ),
    DSLError,
> {
    if program.out_type().is_primitive() {
        build_kernel_with_writer::<REEWriter<T, PrimitiveArrayWriter>>(ctx, inputs, program)
    } else {
        match program.out_type() {
            DataType::Binary | DataType::Utf8 => build_kernel_with_writer::<
                REEWriter<T, StringArrayWriter<i32>>,
            >(ctx, inputs, program),
            DataType::LargeBinary | DataType::LargeUtf8 => build_kernel_with_writer::<
                REEWriter<T, StringArrayWriter<i64>>,
            >(ctx, inputs, program),
            _ => Err(DSLError::UnsupportedDictionaryValueType(
                program.out_type().clone(),
            )),
        }
    }
}

fn build_kernel_with_writer<'a, W: ArrayWriter<'a>>(
    ctx: &'a Context,
    inputs: &[&dyn Datum],
    program: SealedKernelProgram<'_>,
) -> Result<
    (
        Vec<KernelInputType>,
        JitFunction<'a, unsafe extern "C" fn(*mut c_void) -> u64>,
    ),
    DSLError,
> {
    let llvm_mod = ctx.create_module("kernel");
    let builder = ctx.create_builder();
    let i64_type = ctx.i64_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());

    let num_total_inputs = inputs.len() + 1;
    let func_type = i64_type.fn_type(&[ptr_type.into()], false);
    let func_outer = llvm_mod.add_function("kernel", func_type, None);
    let func_inner = llvm_mod.add_function(
        "inner_kernel",
        i64_type.fn_type(
            &(0..num_total_inputs).map(|_| ptr_type.into()).collect_vec(),
            false,
        ),
        Some(Linkage::Private),
    );
    set_noalias_params(&func_inner);

    //
    // Outer function
    //
    declare_blocks!(ctx, func_outer, entry);
    builder.position_at_end(entry);
    let param_ptr = func_outer.get_nth_param(0).unwrap().into_pointer_value();
    let ptr_params = (0..num_total_inputs)
        .map(|idx| KernelParameters::llvm_get(ctx, &builder, param_ptr, idx).into())
        .collect_vec();
    let num_produced = builder
        .build_call(func_inner, &ptr_params, "num_produced")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left();
    builder.build_return(Some(&num_produced)).unwrap();

    //
    // Inner function
    //
    let out_type = program.out_type();
    let out_prim_type = PrimitiveType::for_arrow_type(out_type);

    let indexes_to_iter = program.iterated_indexes();
    if indexes_to_iter.is_empty() {
        return Err(DSLError::NoIteration);
    }

    let ihs: Vec<IteratorHolder> = {
        let idx_to_type: HashMap<usize, KernelInputType> = indexes_to_iter
            .iter()
            .map(|(ty, idx)| (*idx, *ty))
            .collect();

        inputs
            .iter()
            .enumerate()
            .map(|(idx, inp)| {
                if let Some(ty) = idx_to_type.get(&idx) {
                    match ty {
                        KernelInputType::Standard => datum_to_iter(*inp),
                        KernelInputType::SetBit => array_to_setbit_iter(inp.get().0.as_boolean()),
                    }
                } else {
                    datum_to_iter(*inp)
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    };

    let all_inputs_scalar = indexes_to_iter
        .iter()
        .map(|(_ty, idx)| idx)
        .all(|idx| inputs[*idx].get().1);

    let mut iter_llvm_types = HashMap::new();
    for (ty, idx) in indexes_to_iter.iter() {
        let ptype = match ty {
            KernelInputType::Standard => {
                PrimitiveType::for_arrow_type(inputs[*idx].get().0.data_type())
            }
            KernelInputType::SetBit => PrimitiveType::U64,
        };
        iter_llvm_types.insert(*idx, ptype.llvm_type(ctx));
    }
    let next_funcs: HashMap<usize, FunctionValue> = indexes_to_iter
        .iter()
        .map(|(_ty, idx)| {
            (
                *idx,
                generate_next(
                    ctx,
                    &llvm_mod,
                    &format!("next{}", idx),
                    inputs[*idx].get().0.data_type(),
                    &ihs[*idx],
                )
                .unwrap(),
            )
        })
        .collect();

    let next_block_funcs: HashMap<usize, Option<FunctionValue>> = indexes_to_iter
        .iter()
        .map(|(_ty, idx)| {
            (
                *idx,
                generate_next_block::<64>(
                    ctx,
                    &llvm_mod,
                    &format!("next_block{}", idx),
                    inputs[*idx].get().0.data_type(),
                    &ihs[*idx],
                ),
            )
        })
        .collect();

    let get_funcs: HashMap<usize, FunctionValue> = program
        .accessed_indexes()
        .iter()
        .map(|idx| {
            (
                *idx,
                generate_random_access(
                    ctx,
                    &llvm_mod,
                    &format!("get{}", idx),
                    inputs[*idx].get().0.data_type(),
                    &ihs[*idx],
                )
                .unwrap(),
            )
        })
        .collect();

    let mut blocked_access_funcs: HashMap<(usize, u32), FunctionValue> = HashMap::new();
    for &idx in program.accessed_indexes().iter() {
        let dt = inputs[idx].get().0.data_type();
        if let Some(func) = generate_blocked_random_access(
            ctx,
            &llvm_mod,
            &format!("blocked_get{}", idx),
            dt,
            &ihs[idx],
        ) {
            blocked_access_funcs.insert((idx, VEC_SIZE), func);
        }
    }

    declare_blocks!(
        ctx,
        func_inner,
        entry,
        loop_cond,
        filter_check,
        loop_body,
        exit
    );
    builder.position_at_end(entry);
    let mut iter_ptrs = Vec::new();
    for i in 0..inputs.len() {
        let ptr = func_inner
            .get_nth_param(i as u32)
            .unwrap()
            .into_pointer_value();
        iter_ptrs.push(ihs[i].localize_struct(ctx, &builder, ptr));
    }

    let out_ptr = func_inner
        .get_nth_param(inputs.len() as u32)
        .unwrap()
        .into_pointer_value();
    let writer = W::llvm_init(ctx, &llvm_mod, &builder, out_prim_type, out_ptr);

    let bufs: HashMap<usize, PointerValue> = indexes_to_iter
        .iter()
        .map(|(_ty, idx)| {
            (
                *idx,
                builder
                    .build_alloca(iter_llvm_types[idx], &format!("buf{}", idx))
                    .unwrap(),
            )
        })
        .collect();

    let produced_ptr = builder.build_alloca(i64_type, "out_count").unwrap();
    builder
        .build_store(produced_ptr, i64_type.const_zero())
        .unwrap();

    // possibly add a block-iteration fast path TODO
    let mut uses_blocks = false;
    if next_block_funcs.values().all(|f| f.is_some())
        && program.filter().is_none()
        && !all_inputs_scalar
    {
        // potentially use block iteration, allocate buffers
        let mut vec_bufs = HashMap::new();
        let mut vec_types = HashMap::new();
        for (ty, idx) in indexes_to_iter.iter() {
            let ptype = match ty {
                KernelInputType::Standard => {
                    PrimitiveType::for_arrow_type(inputs[*idx].get().0.data_type())
                }
                KernelInputType::SetBit => PrimitiveType::U64,
            };
            let vec_buf = ptype
                .llvm_vec_type(ctx, 64)
                .map(|t| builder.build_alloca(t, &format!("vbuf{}", t)).unwrap());
            vec_bufs.insert(idx, vec_buf);
            vec_types.insert(idx, ptype.llvm_vec_type(ctx, 64));
        }

        if vec_bufs.values().all(|x| x.is_some()) {
            let vec_bufs = vec_bufs
                .into_iter()
                .map(|(k, v)| (*k, v.unwrap()))
                .collect();
            let vec_types = vec_types
                .into_iter()
                .map(|(k, v)| (*k, v.unwrap().as_basic_type_enum()))
                .collect();

            // all our inputs support block iteration, see if our program does
            declare_blocks!(ctx, func_inner, block_loop_cond, block_loop_body);
            builder.position_at_end(block_loop_body);
            let block_context = CompilationContext {
                llvm_ctx: ctx,
                llvm_mod: &llvm_mod,
                builder: &builder,
                bufs: &bufs,
                accessors: &get_funcs,
                vec_bufs: &vec_bufs,
                blocked_access_funcs: &blocked_access_funcs,
                iter_ptrs: &iter_ptrs,
                iter_llvm_types: &vec_types,
            };
            let res = program.expr().compile_block(&block_context);
            match res {
                Ok(mut v) => {
                    // send `v` to the writer, loop back
                    if writer.llvm_ingest_type(ctx) == ctx.bool_type().as_basic_type_enum() {
                        v = builder
                            .build_int_compare(
                                IntPredicate::NE,
                                v,
                                v.get_type().const_zero(),
                                "to_bool",
                            )
                            .unwrap();
                    }
                    writer.llvm_ingest_block(ctx, &builder, v);

                    let curr_produced = builder
                        .build_load(i64_type, produced_ptr, "curr_produced")
                        .unwrap()
                        .into_int_value();
                    let new_produced = builder
                        .build_int_add(
                            curr_produced,
                            i64_type.const_int(v.get_type().get_size() as u64, false),
                            "new_produced",
                        )
                        .unwrap();
                    builder.build_store(produced_ptr, new_produced).unwrap();
                    builder.build_unconditional_branch(block_loop_cond).unwrap();

                    builder.position_at_end(block_loop_cond);
                    let mut had_nexts = Vec::new();
                    // call next on each of the iterators that we care about
                    for (_ty, param_idx) in indexes_to_iter.iter() {
                        let buf = vec_bufs[param_idx];
                        let next_func = next_block_funcs[param_idx].unwrap();
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
                        .build_conditional_branch(accum, block_loop_body, loop_cond)
                        .unwrap();

                    builder.position_at_end(entry);
                    builder.build_unconditional_branch(block_loop_cond).unwrap();

                    uses_blocks = true;
                }
                Err(e) => {
                    println!("Unable to compile blocked version of kernel: {}", e);
                    for buf in vec_bufs.into_values() {
                        buf.as_instruction().unwrap().remove_from_basic_block();
                    }
                    block_loop_cond.remove_from_function().unwrap();
                    block_loop_body.remove_from_function().unwrap();
                }
            }
        }
    }

    if !uses_blocks {
        builder.position_at_end(entry);
        builder.build_unconditional_branch(loop_cond).unwrap();
    }

    builder.position_at_end(loop_cond);
    let mut had_nexts = Vec::new();
    // call next on each of the iterators that we care about (note the
    // difference between the index of the iterator we care about and the index
    // of the parameter value)
    for (_ty, param_idx) in indexes_to_iter.iter() {
        let buf = bufs[param_idx];
        let next_func = next_funcs[param_idx];
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
        .build_conditional_branch(accum, filter_check, exit)
        .unwrap();

    builder.position_at_end(filter_check);
    let scalar_context = CompilationContext {
        llvm_ctx: ctx,
        llvm_mod: &llvm_mod,
        builder: &builder,
        bufs: &bufs,
        accessors: &get_funcs,
        vec_bufs: &bufs,
        blocked_access_funcs: &blocked_access_funcs,
        iter_ptrs: &iter_ptrs,
        iter_llvm_types: &iter_llvm_types,
    };
    match program.filter() {
        Some(cond) => {
            let result = cond.compile(&scalar_context)?;
            builder
                .build_conditional_branch(result.into_int_value(), loop_body, loop_cond)
                .unwrap();
        }
        None => {
            builder.build_unconditional_branch(loop_body).unwrap();
        }
    }

    builder.position_at_end(loop_body);
    let mut result = program.expr().compile(&scalar_context)?;

    if writer.llvm_ingest_type(ctx) == ctx.bool_type().as_basic_type_enum() {
        result = builder
            .build_int_compare(
                IntPredicate::NE,
                result.into_int_value(),
                result.get_type().const_zero().into_int_value(),
                "not_zero",
            )
            .unwrap()
            .as_basic_value_enum();
    }

    writer.llvm_ingest(ctx, &builder, result);
    let curr_produced = builder
        .build_load(i64_type, produced_ptr, "curr_produced")
        .unwrap()
        .into_int_value();
    let new_produced = builder
        .build_int_add(curr_produced, i64_type.const_int(1, false), "new_produced")
        .unwrap();
    builder.build_store(produced_ptr, new_produced).unwrap();

    // if all of our inputs are scalar, then the next function will return true
    // forever -- we need to jump to the exit after the first iteration.
    if all_inputs_scalar {
        builder.build_unconditional_branch(exit).unwrap();
    } else {
        builder.build_unconditional_branch(loop_cond).unwrap();
    }

    builder.position_at_end(exit);
    writer.llvm_flush(ctx, &builder);
    let produced = builder
        .build_load(i64_type, produced_ptr, "produced")
        .unwrap()
        .into_int_value();
    builder.build_return(Some(&produced)).unwrap();

    llvm_mod.verify().unwrap();
    optimize_module(&llvm_mod).unwrap();
    let ee = llvm_mod
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();
    link_req_helpers(&llvm_mod, &ee).unwrap();

    // build an access map for the caller -- for accessed indexes, assume
    // standard, but override that if we use the set bit iterator in the
    // expression
    let mut access_map = BTreeMap::new();
    for idx in program.accessed_indexes() {
        access_map.insert(idx, KernelInputType::Standard);
    }
    for (ty, idx) in indexes_to_iter {
        access_map.insert(idx, ty);
    }

    if access_map.len() != inputs.len() {
        return Err(DSLError::UnusedInput(format!(
            "{} inputs were used, but {} inputs were provided",
            access_map.len(),
            inputs.len()
        )));
    }
    let access_map = access_map.into_values().collect_vec();

    Ok((access_map, unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void) -> u64>(
            func_outer.get_name().to_str().unwrap(),
        )
        .unwrap()
    }))
}
