use std::ffi::c_void;

use arrow_array::{cast::AsArray, ArrayRef, Datum};
use arrow_buffer::MutableBuffer;
use itertools::Itertools;

use crate::{
    compiled_iter::{array_to_setbit_iter, datum_to_iter},
    compiled_kernels::dsl2::{
        compiler::{CompiledDSLFunction, KernelReturnCode},
        resolver::{Resolver, SizeTerm},
        two_d::TwoDArrayRuntime,
        DSL2Error, DSLArgument,
    },
    PrimitiveType,
};

pub struct RunnableDSLFunction {
    compiled: CompiledDSLFunction,
    resolver: Resolver,
}

impl RunnableDSLFunction {
    pub fn new(compiled: CompiledDSLFunction) -> Result<Self, DSL2Error> {
        let inputs = compiled
            .borrow_f()
            .params
            .iter()
            .filter_map(|p| p.ty.size_tag().map(|t| t))
            .map(|t| SizeTerm::parse(t).unwrap())
            .collect_vec();

        let outputs = compiled
            .borrow_f()
            .ret
            .iter()
            .map(|r| r.length_tag())
            .map(|t| SizeTerm::parse(t).unwrap())
            .collect_vec();

        let resolver = Resolver::new(inputs, outputs).map_err(DSL2Error::ResolveError)?;

        Ok(Self { compiled, resolver })
    }

    pub fn run<'args>(&self, inputs: &[DSLArgument<'args>]) -> Result<Vec<ArrayRef>, DSL2Error> {
        let mut ptrs = Vec::new();
        let mut ihs = Vec::new();
        let mut twods = Vec::new();
        let mut buffers = Vec::new();
        let mut input_lengths = Vec::new();

        // allocate and check inputs
        for (idx, (input, arg_type)) in inputs
            .iter()
            .zip(self.compiled.borrow_arg_types().iter())
            .enumerate()
        {
            arg_type
                .matches(input)
                .map_err(|s| DSL2Error::RuntimeArgumentTypeMismatch(idx, s))?;

            match input {
                DSLArgument::Datum(datum) => {
                    if !arg_type.is_set_bit() {
                        let ih = datum_to_iter(*datum)?;
                        ihs.push(ih);
                    } else {
                        let ih = array_to_setbit_iter(datum.get().0.as_boolean())?;
                        ihs.push(ih);
                    }
                    ptrs.push(ihs.last().unwrap().get_ptr() as *const c_void);
                    input_lengths.push(datum.get().0.len());
                }
                DSLArgument::TwoDArray(datums) => {
                    let twod = TwoDArrayRuntime::new(datums)?;
                    twods.push(twod);
                    ptrs.push(twods.last().unwrap().get_ptr() as *const c_void);
                }
                DSLArgument::Buffer(buf, _pt) => {
                    buffers.push(buf);
                    ptrs.push(buffers.last().unwrap().as_ptr() as *const c_void);
                }
            }
        }

        // resolve lengths
        let output_lenghts = self
            .resolver
            .resolve(&input_lengths)
            .map_err(DSL2Error::ResolveError)?;

        // allocate and check outputs
        let mut outputs = Vec::new();
        for (os, len) in self
            .compiled
            .borrow_f()
            .ret
            .iter()
            .zip(output_lenghts.iter())
        {
            outputs.push(os.allocate(*len));
            ptrs.push(outputs.last_mut().unwrap().get_ptr());
        }

        let result = KernelReturnCode::from(unsafe {
            self.compiled
                .borrow_compiled()
                .call(ptrs.as_mut_ptr() as *mut c_void)
        });

        match result {
            KernelReturnCode::Success => Ok(outputs
                .into_iter()
                .map(|o| o.into_array_ref(None))
                .collect_vec()),
            KernelReturnCode::InvalidEmitIndex => Err(DSL2Error::RuntimeInvalidEmitIndex),
            KernelReturnCode::Unknown(c) => Err(DSL2Error::RuntimeUnknownReturnCode(c)),
        }
    }
}
