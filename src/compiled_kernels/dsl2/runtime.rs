use std::ffi::c_void;

use arrow_array::{cast::AsArray, ArrayRef};
use itertools::Itertools;

use crate::{
    compiled_iter::{array_to_setbit_iter, datum_to_iter, IteratorHolder},
    compiled_kernels::dsl2::{
        compiler::{CompiledDSLFunction, KernelReturnCode},
        resolver::{ResolveResult, Resolver, SizeTerm},
        two_d::TwoDArrayRuntime,
        DSLArgument, OutputSlot,
    },
    ArrowKernelError,
};

struct PreparedInputs {
    ptrs: Vec<*mut c_void>,
    ihs: Vec<IteratorHolder>,
    twods: Vec<TwoDArrayRuntime>,
    input_lengths: Vec<usize>,
}

pub struct RunnableDSLFunction {
    compiled: CompiledDSLFunction,
    resolver: Resolver,
    #[allow(dead_code)] // used in tests
    pub(crate) vectorized: bool,
}

impl RunnableDSLFunction {
    pub fn new(compiled: CompiledDSLFunction, vectorized: bool) -> Result<Self, ArrowKernelError> {
        let inputs = compiled
            .borrow_f()
            .params
            .iter()
            .filter_map(|p| p.ty.size_tag())
            .map(|t| SizeTerm::parse(t).unwrap())
            .collect_vec();

        let outputs = compiled
            .borrow_f()
            .ret
            .iter()
            .map(|r| r.length_tag())
            .map(|t| SizeTerm::parse(t).unwrap())
            .collect_vec();

        let resolver = Resolver::new(inputs, outputs).map_err(ArrowKernelError::ResolveError)?;

        Ok(Self {
            compiled,
            resolver,
            vectorized,
        })
    }

    pub fn run<'args>(
        &self,
        inputs: &[DSLArgument<'args>],
    ) -> Result<Vec<ArrayRef>, ArrowKernelError> {
        let mut prepared = self.prepare_inputs(inputs)?;
        let output_lengths = self
            .resolver
            .resolve(&prepared.input_lengths)
            .map_err(ArrowKernelError::ResolveError)?;

        assert_eq!(output_lengths.len(), self.compiled.borrow_f().ret.len());

        let mut outputs = Vec::new();
        for (os, &len) in self
            .compiled
            .borrow_f()
            .ret
            .iter()
            .zip(output_lengths.iter())
        {
            let len = resolved_len(len);
            outputs.push(os.allocate(len));
            prepared.ptrs.push(outputs.last_mut().unwrap().get_ptr());
        }

        self.execute(&mut prepared.ptrs)?;
        Ok(outputs
            .into_iter()
            .map(|o| o.into_array_ref(None))
            .collect_vec())
    }

    pub fn run_into<'args>(
        &self,
        inputs: &[DSLArgument<'args>],
        outputs: &mut [OutputSlot],
    ) -> Result<(), ArrowKernelError> {
        if outputs.len() != self.compiled.borrow_f().ret.len() {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "expected {} outputs, got {}",
                self.compiled.borrow_f().ret.len(),
                outputs.len()
            )));
        }

        let mut prepared = self.prepare_inputs(inputs)?;
        let output_lengths = self
            .resolver
            .resolve(&prepared.input_lengths)
            .map_err(ArrowKernelError::ResolveError)?;

        for (idx, ((expected, output), &len)) in self
            .compiled
            .borrow_f()
            .ret
            .iter()
            .zip(outputs.iter_mut())
            .zip(output_lengths.iter())
            .enumerate()
        {
            if output.spec() != expected.spec() {
                return Err(ArrowKernelError::ArgumentMismatch(format!(
                    "output {idx} expected {:?}, got {:?}",
                    expected.spec(),
                    output.spec()
                )));
            }

            output.reserve_for_additional(resolved_len(len));
            prepared.ptrs.push(output.get_ptr());
        }

        self.execute(&mut prepared.ptrs)
    }

    fn prepare_inputs<'args>(
        &self,
        inputs: &[DSLArgument<'args>],
    ) -> Result<PreparedInputs, ArrowKernelError> {
        let expected_inputs = self.compiled.borrow_arg_types().len();
        if inputs.len() != expected_inputs {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "expected {} inputs, got {}",
                expected_inputs,
                inputs.len()
            )));
        }

        let mut prepared = PreparedInputs {
            ptrs: Vec::new(),
            ihs: Vec::new(),
            twods: Vec::new(),
            input_lengths: Vec::new(),
        };

        for (idx, (input, arg_type)) in inputs
            .iter()
            .zip(self.compiled.borrow_arg_types().iter())
            .enumerate()
        {
            arg_type
                .matches(input)
                .map_err(|s| ArrowKernelError::RuntimeArgumentTypeMismatch(idx, s))?;

            match input {
                DSLArgument::Datum(datum) => {
                    if !arg_type.is_set_bit() {
                        let ih = datum_to_iter(*datum)?;
                        prepared.ihs.push(ih);
                    } else {
                        let ih = array_to_setbit_iter(datum.get().0.as_boolean())?;
                        prepared.ihs.push(ih);
                    }
                    prepared
                        .ptrs
                        .push(prepared.ihs.last().unwrap().get_ptr() as *mut c_void);
                    if !datum.get().1 {
                        prepared.input_lengths.push(datum.get().0.len());
                    }
                }
                DSLArgument::TwoDArray(datums) => {
                    let twod = TwoDArrayRuntime::new(datums)?;
                    prepared.twods.push(twod);
                    prepared
                        .ptrs
                        .push(prepared.twods.last().unwrap().get_ptr() as *mut c_void);
                }
                DSLArgument::Buffer { ptr, .. } => {
                    prepared.ptrs.push(*ptr);
                }
                DSLArgument::StringSaver(ptr) => {
                    prepared.ptrs.push(*ptr);
                }
            }
        }

        Ok(prepared)
    }

    fn execute(&self, ptrs: &mut Vec<*mut c_void>) -> Result<(), ArrowKernelError> {
        let result = KernelReturnCode::from(unsafe {
            self.compiled
                .borrow_compiled()
                .call(ptrs.as_mut_ptr() as *mut c_void)
        });

        match result {
            KernelReturnCode::Success => Ok(()),
            KernelReturnCode::InvalidEmitIndex => Err(ArrowKernelError::RuntimeInvalidEmitIndex),
            KernelReturnCode::Unknown(c) => Err(ArrowKernelError::RuntimeUnknownReturnCode(c)),
        }
    }
}

fn resolved_len(res: ResolveResult) -> usize {
    match res {
        ResolveResult::Exact(len) => len,
        ResolveResult::AtLeast(len) => len,
        ResolveResult::Unknown => {
            // We can get an unknown result if a kernel is only over scalar
            // inputs. In that case, allocate capacity for a single result.
            1
        }
    }
}
