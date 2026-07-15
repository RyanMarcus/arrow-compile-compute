use std::{ffi::c_void, sync::Arc};

use arrow_array::{
    cast::AsArray,
    types::{Int16Type, Int32Type, Int64Type, RunEndIndexType},
    Array, ArrayRef, RunArray,
};
use inkwell::{values::PointerValue, AddressSpace, IntPredicate};
use repr_offset::ReprOffset;

use crate::{
    compiled_kernels::cmp::add_memcmp,
    compiled_writers::{
        AnyRuntime, AnyWriter, AnyWriterEmitter, PrimitiveWriter, RunEndType, Writer,
        WriterCodegen, WriterEmitter, WriterRuntime,
    },
    declare_blocks, increment_pointer, ArrowKernelError, ComparisonType, PrimitiveType,
};

pub struct RunEndWriter {
    run_end_type: RunEndType,
    value_type: PrimitiveType,
    run_ends: PrimitiveWriter,
    values: Box<AnyWriter>,
}

impl RunEndWriter {
    pub fn compile(
        run_end_type: RunEndType,
        value_type: PrimitiveType,
        values: AnyWriter,
    ) -> Result<Self, ArrowKernelError> {
        if matches!(value_type, PrimitiveType::List(_, _)) {
            return Err(ArrowKernelError::InternalError(
                "run-end writer does not support list values".into(),
            ));
        }
        Ok(Self {
            run_end_type,
            value_type,
            run_ends: PrimitiveWriter::compile(run_end_type.primitive_type())?,
            values: Box::new(values),
        })
    }

    fn get_child_ptr<'ctx>(
        codegen: WriterCodegen<'ctx, '_>,
        runtime_ptr: PointerValue<'ctx>,
        offset: usize,
        name: &str,
    ) -> PointerValue<'ctx> {
        codegen
            .builder
            .build_load(
                codegen.ctx.ptr_type(AddressSpace::default()),
                increment_pointer!(codegen.ctx, codegen.builder, runtime_ptr, offset),
                name,
            )
            .unwrap()
            .into_pointer_value()
    }

    fn emit_run_end<'ctx, 'borrow>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        run_end_runtime_ptr: PointerValue<'ctx>,
        run_end: inkwell::values::IntValue<'ctx>,
    ) -> Result<(), ArrowKernelError> {
        let converted = codegen
            .builder
            .build_int_truncate_or_bit_cast(
                run_end,
                self.run_end_type
                    .primitive_type()
                    .llvm_type(codegen.ctx)
                    .into_int_type(),
                "encoded_run_end",
            )
            .unwrap();
        self.run_ends
            .llvm_write(codegen, run_end_runtime_ptr, |e| e.emit(converted.into()))?;
        Ok(())
    }
}

impl Writer for RunEndWriter {
    fn allocate(&self, size: usize) -> AnyRuntime {
        let mut last_value =
            vec![0_u128; self.value_type.width().div_ceil(16).max(1)].into_boxed_slice();
        let mut runtime = RunEndWriterRuntime {
            run_end_runtime_ptr: std::ptr::null_mut(),
            value_runtime_ptr: std::ptr::null_mut(),
            last_value_ptr: last_value.as_mut_ptr().cast(),
            curr_run_end: 0,
            num_runs: 0,
            has_last_value: 0,
            run_ends: Box::new(self.run_ends.allocate(size)),
            values: Box::new(self.values.allocate(size)),
            last_value,
            run_end_type: self.run_end_type,
        };
        runtime.run_end_runtime_ptr = runtime.run_ends.as_ptr();
        runtime.value_runtime_ptr = runtime.values.as_ptr();
        runtime.into()
    }

    fn llvm_init<'ctx, 'borrow>(
        &self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
    ) {
        self.run_ends.llvm_init(
            codegen,
            Self::get_child_ptr(
                codegen,
                runtime_ptr,
                RunEndWriterRuntime::OFFSET_RUN_END_RUNTIME_PTR,
                "run_end_runtime",
            ),
        );
        self.values.llvm_init(
            codegen,
            Self::get_child_ptr(
                codegen,
                runtime_ptr,
                RunEndWriterRuntime::OFFSET_VALUE_RUNTIME_PTR,
                "run_value_runtime",
            ),
        );
    }

    fn llvm_write<'ctx, 'borrow, F>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        f: F,
    ) -> Result<(), ArrowKernelError>
    where
        F: Fn(&mut AnyWriterEmitter<'ctx, 'borrow>) -> Result<(), ArrowKernelError>,
    {
        let mut emitter = RunEndWriterEmitter {
            writer: self,
            codegen,
            runtime_ptr,
            run_end_runtime_ptr: Self::get_child_ptr(
                codegen,
                runtime_ptr,
                RunEndWriterRuntime::OFFSET_RUN_END_RUNTIME_PTR,
                "run_end_runtime",
            ),
            value_runtime_ptr: Self::get_child_ptr(
                codegen,
                runtime_ptr,
                RunEndWriterRuntime::OFFSET_VALUE_RUNTIME_PTR,
                "run_value_runtime",
            ),
            used: false,
        }
        .into();
        f(&mut emitter)
    }
}

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct RunEndWriterRuntime {
    run_end_runtime_ptr: *mut c_void,
    value_runtime_ptr: *mut c_void,
    last_value_ptr: *mut c_void,
    curr_run_end: u64,
    num_runs: u64,
    has_last_value: u8,
    run_ends: Box<AnyRuntime>,
    values: Box<AnyRuntime>,
    last_value: Box<[u128]>,
    run_end_type: RunEndType,
}

impl RunEndWriterRuntime {
    fn into_typed_array<K: RunEndIndexType>(
        self,
        len: usize,
    ) -> Result<ArrayRef, ArrowKernelError> {
        let run_ends = self.run_ends.to_array(self.num_runs as usize)?;
        let run_ends = run_ends.as_primitive::<K>();
        let values = self.values.to_array(self.num_runs as usize)?;
        let array = RunArray::<K>::try_new(run_ends, values.as_ref()).map_err(|error| {
            ArrowKernelError::InternalError(format!(
                "run-end writer produced an invalid array: {error}"
            ))
        })?;
        if array.len() != len {
            return Err(ArrowKernelError::InternalError(
                "run-end writer logical length mismatch".into(),
            ));
        }
        Ok(Arc::new(array))
    }
}

impl WriterRuntime for RunEndWriterRuntime {
    fn as_ptr(&mut self) -> *mut c_void {
        (self as *mut Self).cast()
    }

    fn reserve_for_additional(&mut self, count: usize) -> Result<(), ArrowKernelError> {
        if self.has_last_value != 0 {
            self.run_ends.append_integer(self.curr_run_end)?;
            self.has_last_value = 0;
        }
        self.run_ends.reserve_for_additional(count)?;
        self.values.reserve_for_additional(count)?;
        self.run_end_runtime_ptr = self.run_ends.as_ptr();
        self.value_runtime_ptr = self.values.as_ptr();
        self.last_value_ptr = self.last_value.as_mut_ptr().cast();
        Ok(())
    }

    fn len(&self) -> usize {
        self.curr_run_end as usize
    }

    fn to_array(mut self, len: usize) -> Result<ArrayRef, ArrowKernelError> {
        if len != self.curr_run_end as usize {
            return Err(ArrowKernelError::InternalError(format!(
                "run-end writer contains {} values but {len} were requested",
                self.curr_run_end
            )));
        }
        if self.has_last_value != 0 {
            self.run_ends.append_integer(self.curr_run_end)?;
            self.has_last_value = 0;
        }
        match self.run_end_type {
            RunEndType::Int16 => self.into_typed_array::<Int16Type>(len),
            RunEndType::Int32 => self.into_typed_array::<Int32Type>(len),
            RunEndType::Int64 => self.into_typed_array::<Int64Type>(len),
        }
    }
}

pub struct RunEndWriterEmitter<'ctx, 'borrow> {
    writer: &'borrow RunEndWriter,
    codegen: WriterCodegen<'ctx, 'borrow>,
    runtime_ptr: PointerValue<'ctx>,
    run_end_runtime_ptr: PointerValue<'ctx>,
    value_runtime_ptr: PointerValue<'ctx>,
    used: bool,
}

impl<'ctx, 'borrow> WriterEmitter<'ctx, 'borrow> for RunEndWriterEmitter<'ctx, 'borrow> {
    fn emit(
        &mut self,
        value: inkwell::values::BasicValueEnum<'ctx>,
    ) -> Result<(), ArrowKernelError> {
        if self.used {
            return Err(ArrowKernelError::InternalError(
                "emit called on non-empty run-end emitter".into(),
            ));
        }
        self.used = true;
        let i64_type = self.codegen.ctx.i64_type();
        let curr_run_end_ptr = increment_pointer!(
            self.codegen.ctx,
            self.codegen.builder,
            self.runtime_ptr,
            RunEndWriterRuntime::OFFSET_CURR_RUN_END
        );
        let curr_run_end = self
            .codegen
            .builder
            .build_load(i64_type, curr_run_end_ptr, "current_run_end")
            .unwrap()
            .into_int_value();
        let new_run_end = self
            .codegen
            .builder
            .build_int_add(curr_run_end, i64_type.const_int(1, false), "new_run_end")
            .unwrap();
        self.codegen
            .builder
            .build_store(curr_run_end_ptr, new_run_end)
            .unwrap();
        let last_value_ptr = self
            .codegen
            .builder
            .build_load(
                self.codegen.ctx.ptr_type(AddressSpace::default()),
                increment_pointer!(
                    self.codegen.ctx,
                    self.codegen.builder,
                    self.runtime_ptr,
                    RunEndWriterRuntime::OFFSET_LAST_VALUE_PTR
                ),
                "last_run_value_ptr",
            )
            .unwrap()
            .into_pointer_value();
        let func = self
            .codegen
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();
        declare_blocks!(
            self.codegen.ctx,
            func,
            run_end_compare,
            run_end_close,
            run_end_new,
            run_end_exit
        );
        let has_last_value_ptr = increment_pointer!(
            self.codegen.ctx,
            self.codegen.builder,
            self.runtime_ptr,
            RunEndWriterRuntime::OFFSET_HAS_LAST_VALUE
        );
        let has_last_value = self
            .codegen
            .builder
            .build_load(
                self.codegen.ctx.i8_type(),
                has_last_value_ptr,
                "run_end_has_last_value",
            )
            .unwrap()
            .into_int_value();
        let is_first = self
            .codegen
            .builder
            .build_int_compare(
                IntPredicate::EQ,
                has_last_value,
                self.codegen.ctx.i8_type().const_zero(),
                "run_end_is_first",
            )
            .unwrap();
        self.codegen
            .builder
            .build_conditional_branch(is_first, run_end_new, run_end_compare)
            .unwrap();

        self.codegen.builder.position_at_end(run_end_compare);
        let last_value = self
            .codegen
            .builder
            .build_load(
                self.writer.value_type.llvm_type(self.codegen.ctx),
                last_value_ptr,
                "last_run_value",
            )
            .unwrap();
        let matches = match self.writer.value_type.comparison_type() {
            ComparisonType::Int { .. } | ComparisonType::Float => {
                let int_type = PrimitiveType::int_with_width(self.writer.value_type.width())
                    .llvm_type(self.codegen.ctx);
                let lhs = self
                    .codegen
                    .builder
                    .build_bit_cast(last_value, int_type, "last_run_value_bits")
                    .unwrap()
                    .into_int_value();
                let rhs = self
                    .codegen
                    .builder
                    .build_bit_cast(value, int_type, "run_value_bits")
                    .unwrap()
                    .into_int_value();
                self.codegen
                    .builder
                    .build_int_compare(IntPredicate::EQ, lhs, rhs, "same_run_value")
                    .unwrap()
            }
            ComparisonType::String => {
                let memcmp = add_memcmp(self.codegen.ctx, self.codegen.module);
                let cmp = self
                    .codegen
                    .builder
                    .build_call(memcmp, &[last_value.into(), value.into()], "run_value_cmp")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_int_value();
                self.codegen
                    .builder
                    .build_int_compare(
                        IntPredicate::EQ,
                        cmp,
                        i64_type.const_zero(),
                        "same_run_value",
                    )
                    .unwrap()
            }
            ComparisonType::List(_) => unreachable!(),
        };
        self.codegen
            .builder
            .build_conditional_branch(matches, run_end_exit, run_end_close)
            .unwrap();

        self.codegen.builder.position_at_end(run_end_close);
        self.writer
            .emit_run_end(self.codegen, self.run_end_runtime_ptr, curr_run_end)?;
        self.codegen
            .builder
            .build_unconditional_branch(run_end_new)
            .unwrap();

        self.codegen.builder.position_at_end(run_end_new);
        self.writer
            .values
            .llvm_write(self.codegen, self.value_runtime_ptr, |emitter| {
                emitter.emit(value)
            })?;
        self.codegen
            .builder
            .build_store(last_value_ptr, value)
            .unwrap();
        self.codegen
            .builder
            .build_store(
                has_last_value_ptr,
                self.codegen.ctx.i8_type().const_int(1, false),
            )
            .unwrap();
        let num_runs_ptr = increment_pointer!(
            self.codegen.ctx,
            self.codegen.builder,
            self.runtime_ptr,
            RunEndWriterRuntime::OFFSET_NUM_RUNS
        );
        let num_runs = self
            .codegen
            .builder
            .build_load(i64_type, num_runs_ptr, "num_runs")
            .unwrap()
            .into_int_value();
        self.codegen
            .builder
            .build_store(
                num_runs_ptr,
                self.codegen
                    .builder
                    .build_int_add(num_runs, i64_type.const_int(1, false), "new_num_runs")
                    .unwrap(),
            )
            .unwrap();
        self.codegen
            .builder
            .build_unconditional_branch(run_end_exit)
            .unwrap();

        self.codegen.builder.position_at_end(run_end_exit);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int32Type, Array, Int32Array, RunArray};
    use inkwell::{context::Context, values::BasicValue, AddressSpace, OptimizationLevel};

    use crate::{
        compiled_writers::{
            RunEndType, Writer, WriterCodegen, WriterEmitter, WriterRuntime, WriterSpec,
        },
        declare_blocks, PrimitiveType,
    };

    #[test]
    fn run_end_writer_emits_runs_through_composed_child_writers() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers_run_end_writer");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let append_func = llvm_mod.add_function(
            "append",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );
        declare_blocks!(ctx, append_func, entry);
        build.position_at_end(entry);

        let writer = WriterSpec::RunEndEncoded(
            RunEndType::Int32,
            Box::new(WriterSpec::Primitive(PrimitiveType::I32)),
        )
        .compile()
        .unwrap();
        let runtime_ptr = append_func.get_nth_param(0).unwrap().into_pointer_value();
        let codegen = WriterCodegen {
            ctx: &ctx,
            module: &llvm_mod,
            builder: &build,
        };
        let input = [1_i32, 1, 1, 2, 3, 3, 4, 4, 4, 4, 50];

        writer.llvm_init(codegen, runtime_ptr);
        for value in input {
            writer
                .llvm_write(codegen, runtime_ptr, |emitter| {
                    emitter.emit(
                        ctx.i32_type()
                            .const_int(value as u64, true)
                            .as_basic_value_enum(),
                    )
                })
                .unwrap();
        }
        build.build_return(None).unwrap();

        llvm_mod.verify().unwrap();

        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let append = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(
                append_func.get_name().to_str().unwrap(),
            )
            .unwrap()
        };
        let mut runtime = writer.allocate(input.len());
        unsafe {
            append.call(runtime.as_ptr());
        }
        runtime.reserve_for_additional(input.len()).unwrap();
        unsafe {
            append.call(runtime.as_ptr());
        }

        let array = runtime.to_array_ref().unwrap();
        let run_array = array
            .as_any()
            .downcast_ref::<RunArray<Int32Type>>()
            .unwrap();
        assert_eq!(
            run_array.run_ends().values(),
            &[3, 4, 6, 10, 11, 14, 15, 17, 21, 22]
        );
        assert_eq!(
            run_array.values().as_primitive::<Int32Type>().values(),
            &[1, 2, 3, 4, 50, 1, 2, 3, 4, 50]
        );
        let decoded = run_array.downcast::<Int32Array>().unwrap();
        let decoded: Vec<i32> = decoded.into_iter().map(|value| value.unwrap()).collect();
        assert_eq!(decoded, input.into_iter().chain(input).collect::<Vec<_>>());
    }
}
