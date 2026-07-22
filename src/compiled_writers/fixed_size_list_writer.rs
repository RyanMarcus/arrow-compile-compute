use std::{ffi::c_void, sync::Arc};

use arrow_array::FixedSizeListArray;
use arrow_schema::Field;
use inkwell::{
    values::{BasicValueEnum, PointerValue, VectorValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::{
    compiled_writers::{
        AnyRuntime, AnyWriter, AnyWriterEmitter, Writer, WriterCodegen, WriterEmitter,
        WriterRuntime,
    },
    increment_pointer, ArrowKernelError, ListItemType, PrimitiveType,
};

pub struct FixedSizeListWriter {
    list_type: PrimitiveType,
    list_size: usize,
    values: Box<AnyWriter>,
}

impl FixedSizeListWriter {
    pub fn compile(list_type: PrimitiveType, values: AnyWriter) -> Result<Self, ArrowKernelError> {
        let PrimitiveType::List(_, list_size) = list_type else {
            return Err(ArrowKernelError::InternalError(format!(
                "fixed-size list writer requires a list type, got {list_type}"
            )));
        };
        Ok(Self {
            list_type,
            list_size,
            values: Box::new(values),
        })
    }

    fn get_value_ptr<'ctx>(
        codegen: WriterCodegen<'ctx, '_>,
        runtime_ptr: PointerValue<'ctx>,
    ) -> PointerValue<'ctx> {
        codegen
            .builder
            .build_load(
                codegen.ctx.ptr_type(AddressSpace::default()),
                increment_pointer!(
                    codegen.ctx,
                    codegen.builder,
                    runtime_ptr,
                    FixedSizeListWriterRuntime::OFFSET_VALUE_RUNTIME_PTR
                ),
                "fixed_size_list_value_runtime",
            )
            .unwrap()
            .into_pointer_value()
    }

    fn llvm_increment_num_written<'ctx>(
        codegen: WriterCodegen<'ctx, '_>,
        runtime_ptr: PointerValue<'ctx>,
        count: u64,
    ) {
        let i64_type = codegen.ctx.i64_type();
        let num_written_ptr = increment_pointer!(
            codegen.ctx,
            codegen.builder,
            runtime_ptr,
            FixedSizeListWriterRuntime::OFFSET_NUM_WRITTEN
        );
        let num_written = codegen
            .builder
            .build_load(i64_type, num_written_ptr, "fixed_size_lists_written")
            .unwrap()
            .into_int_value();
        let new_num_written = codegen
            .builder
            .build_int_add(
                num_written,
                i64_type.const_int(count, false),
                "new_fixed_size_lists_written",
            )
            .unwrap();
        codegen
            .builder
            .build_store(num_written_ptr, new_num_written)
            .unwrap();
    }
}

impl Writer for FixedSizeListWriter {
    fn allocate(&self, size: usize) -> AnyRuntime {
        let mut runtime = FixedSizeListWriterRuntime {
            value_runtime_ptr: std::ptr::null_mut(),
            num_written: 0,
            values: Box::new(self.values.allocate(size.saturating_mul(self.list_size))),
            list_size: self.list_size,
        };
        runtime.value_runtime_ptr = runtime.values.as_ptr();
        runtime.into()
    }

    fn llvm_init<'ctx, 'borrow>(
        &self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
    ) {
        self.values
            .llvm_init(codegen, Self::get_value_ptr(codegen, runtime_ptr));
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
        let mut emitter = AnyWriterEmitter::FixedSizeList(FixedSizeListWriterEmitter {
            codegen,
            value_runtime_ptr: Self::get_value_ptr(codegen, runtime_ptr),
            values: self.values.as_ref(),
            list_type: self.list_type,
            used: false,
        });
        f(&mut emitter)?;
        let AnyWriterEmitter::FixedSizeList(emitter) = emitter else {
            unreachable!()
        };
        if !emitter.used {
            return Err(ArrowKernelError::InternalError(
                "emit not called on fixed-size list emitter".into(),
            ));
        }

        Self::llvm_increment_num_written(codegen, runtime_ptr, 1);
        Ok(())
    }

    fn llvm_write_block<'ctx, 'borrow>(
        &'borrow self,
        codegen: WriterCodegen<'ctx, 'borrow>,
        runtime_ptr: PointerValue<'ctx>,
        values: VectorValue<'ctx>,
        logical_len: u32,
    ) -> Result<(), ArrowKernelError> {
        let expected_lanes = logical_len as usize * self.list_size;
        let actual_lanes = values.get_type().get_size();
        if actual_lanes as usize != expected_lanes {
            return Err(ArrowKernelError::InternalError(format!(
                "fixed-size-list block has {} lanes, expected {expected_lanes}",
                actual_lanes
            )));
        }
        self.values.llvm_write_block(
            codegen,
            Self::get_value_ptr(codegen, runtime_ptr),
            values,
            expected_lanes as u32,
        )?;
        Self::llvm_increment_num_written(codegen, runtime_ptr, logical_len as u64);
        Ok(())
    }
}

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct FixedSizeListWriterRuntime {
    value_runtime_ptr: *mut c_void,
    num_written: u64,
    values: Box<AnyRuntime>,
    list_size: usize,
}

impl WriterRuntime for FixedSizeListWriterRuntime {
    fn as_ptr(&mut self) -> *mut c_void {
        (self as *mut Self).cast()
    }

    fn reserve_for_additional(&mut self, count: usize) -> Result<(), ArrowKernelError> {
        self.values
            .reserve_for_additional(count.saturating_mul(self.list_size))?;
        self.value_runtime_ptr = self.values.as_ptr();
        Ok(())
    }

    fn len(&self) -> usize {
        self.num_written as usize
    }

    fn to_array(self, len: usize) -> Result<arrow_array::ArrayRef, ArrowKernelError> {
        let value_len = len.checked_mul(self.list_size).ok_or_else(|| {
            ArrowKernelError::InternalError("fixed-size list value count overflowed".into())
        })?;
        let values = self.values.to_array(value_len)?;
        let field = Arc::new(Field::new_list_field(values.data_type().clone(), false));

        let array = FixedSizeListArray::try_new(field, self.list_size as i32, values, None)
            .map_err(|error| {
                ArrowKernelError::InternalError(format!(
                    "fixed-size list writer produced an invalid array: {error}"
                ))
            })?;
        Ok(Arc::new(array))
    }
}

pub struct FixedSizeListWriterEmitter<'ctx, 'borrow> {
    codegen: WriterCodegen<'ctx, 'borrow>,
    value_runtime_ptr: PointerValue<'ctx>,
    values: &'borrow AnyWriter,
    list_type: PrimitiveType,
    used: bool,
}

impl<'ctx, 'borrow> WriterEmitter<'ctx, 'borrow> for FixedSizeListWriterEmitter<'ctx, 'borrow> {
    fn emit(&mut self, val: BasicValueEnum<'ctx>) -> Result<(), ArrowKernelError> {
        if self.used {
            return Err(ArrowKernelError::InternalError(
                "emit called on non-empty fixed-size list emitter".into(),
            ));
        }
        self.used = true;

        let expected_type = self.list_type.llvm_type(self.codegen.ctx);
        if val.get_type() != expected_type {
            return Err(ArrowKernelError::InternalError(format!(
                "fixed-size list writer expected LLVM type {expected_type}, got {}",
                val.get_type()
            )));
        }

        let PrimitiveType::List(item_type, list_size) = self.list_type else {
            unreachable!()
        };
        match item_type {
            ListItemType::P64x2 => {
                for idx in 0..list_size {
                    let value = self
                        .codegen
                        .builder
                        .build_extract_value(
                            val.into_array_value(),
                            idx as u32,
                            "fixed_size_list_string_value",
                        )
                        .unwrap();
                    self.values
                        .llvm_write(self.codegen, self.value_runtime_ptr, |e| e.emit(value))?;
                }
            }
            _ => self.values.llvm_write_block(
                self.codegen,
                self.value_runtime_ptr,
                val.into_vector_value(),
                list_size as u32,
            )?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, types::Int32Type, BinaryArray};
    use inkwell::{
        context::Context, types::VectorType, values::BasicValue, AddressSpace, OptimizationLevel,
    };

    use crate::{
        compiled_writers::{Writer, WriterCodegen, WriterEmitter, WriterRuntime, WriterSpec},
        declare_blocks, ListItemType, PrimitiveType,
    };

    #[test]
    fn fixed_size_list_writer_composes_primitive_and_reserves() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers_fixed_size_list_primitive");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );
        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);

        let runtime_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer =
            WriterSpec::FixedSizeList(Box::new(WriterSpec::Primitive(PrimitiveType::I32)), 3)
                .compile()
                .unwrap();
        let codegen = WriterCodegen {
            ctx: &ctx,
            module: &llvm_mod,
            builder: &build,
        };
        writer.llvm_init(codegen, runtime_ptr);
        for row in [[1_i32, 2, 3], [4, 5, 6]] {
            let row = row.map(|value| ctx.i32_type().const_int(value as u64, true));
            let row = VectorType::const_vector(&row);
            writer
                .llvm_write(codegen, runtime_ptr, |emitter| {
                    emitter.emit(row.as_basic_value_enum())
                })
                .unwrap();
        }
        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();

        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };
        let mut runtime = writer.allocate(2);
        unsafe { f.call(runtime.as_ptr()) };
        runtime.reserve_for_additional(2).unwrap();
        unsafe { f.call(runtime.as_ptr()) };

        assert_eq!(runtime.len(), 4);
        let array = runtime.to_array_ref().unwrap();
        let list = array.as_fixed_size_list();
        assert_eq!(list.value_length(), 3);
        assert_eq!(
            list.values().as_primitive::<Int32Type>().values(),
            &[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn fixed_size_list_writer_composes_boolean_writer() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers_fixed_size_list_boolean");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );
        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);

        let runtime_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = WriterSpec::for_primitive_type(PrimitiveType::List(ListItemType::Boolean, 3))
            .compile()
            .unwrap();
        let codegen = WriterCodegen {
            ctx: &ctx,
            module: &llvm_mod,
            builder: &build,
        };
        writer.llvm_init(codegen, runtime_ptr);
        let list_type = PrimitiveType::List(ListItemType::Boolean, 3)
            .llvm_type(&ctx)
            .into_vector_type();
        for row in [[true, false, true], [false, true, false]] {
            let row = row.map(|value| ctx.bool_type().const_int(u64::from(value), false));
            let row = VectorType::const_vector(&row);
            assert_eq!(row.get_type(), list_type);
            writer
                .llvm_write(codegen, runtime_ptr, |emitter| {
                    emitter.emit(row.as_basic_value_enum())
                })
                .unwrap();
        }
        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();

        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };
        let mut runtime = writer.allocate(2);
        unsafe { f.call(runtime.as_ptr()) };

        let array = runtime.to_array_ref().unwrap();
        let actual = array
            .as_fixed_size_list()
            .values()
            .as_boolean()
            .iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(actual, [true, false, true, false, true, false]);
    }

    #[test]
    fn fixed_size_list_writer_composes_string_writer() {
        let ctx = Context::create();
        let llvm_mod = ctx.create_module("compiled_writers_fixed_size_list_string");
        let build = ctx.create_builder();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let func = llvm_mod.add_function(
            "test",
            ctx.void_type().fn_type(&[ptr_type.into()], false),
            None,
        );
        declare_blocks!(ctx, func, entry);
        build.position_at_end(entry);

        let runtime_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let writer = WriterSpec::FixedSizeList(Box::new(WriterSpec::String), 2)
            .compile()
            .unwrap();
        let codegen = WriterCodegen {
            ctx: &ctx,
            module: &llvm_mod,
            builder: &build,
        };
        let string_type = PrimitiveType::P64x2.llvm_type(&ctx).into_struct_type();
        let list_type = PrimitiveType::List(ListItemType::P64x2, 2)
            .llvm_type(&ctx)
            .into_array_type();
        let strings = ["alpha", "", "beta", "gamma"];
        writer.llvm_init(codegen, runtime_ptr);
        for row in strings.chunks_exact(2) {
            let mut list = list_type.const_zero();
            for (idx, value) in row.iter().enumerate() {
                let start = value.as_ptr();
                let end = start.wrapping_add(value.len());
                let value = string_type.const_named_struct(&[
                    ctx.i64_type()
                        .const_int(start as usize as u64, false)
                        .const_to_pointer(ptr_type)
                        .into(),
                    ctx.i64_type()
                        .const_int(end as usize as u64, false)
                        .const_to_pointer(ptr_type)
                        .into(),
                ]);
                list = build
                    .build_insert_value(list, value, idx as u32, "fixed_size_list_string_row")
                    .unwrap()
                    .into_array_value();
            }
            writer
                .llvm_write(codegen, runtime_ptr, |emitter| {
                    emitter.emit(list.as_basic_value_enum())
                })
                .unwrap();
        }
        build.build_return(None).unwrap();
        llvm_mod.verify().unwrap();

        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        crate::compiled_kernels::link_req_helpers(&llvm_mod, &ee).unwrap();
        let f = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void)>(func.get_name().to_str().unwrap())
                .unwrap()
        };
        let mut runtime = writer.allocate(2);
        unsafe { f.call(runtime.as_ptr()) };

        let array = runtime.to_array_ref().unwrap();
        let values = array
            .as_fixed_size_list()
            .values()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert_eq!(
            values.iter().map(Option::unwrap).collect::<Vec<_>>(),
            strings.map(str::as_bytes)
        );
    }
}
