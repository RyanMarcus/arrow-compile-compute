use inkwell::{
    context::Context,
    module::{Linkage, Module},
    types::BasicType,
    values::FunctionValue,
};

use crate::PrimitiveType;

use super::IteratorHolder;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IteratorMapType {
    Boolean,
    Primitive(PrimitiveType),
    VariableList(Box<IteratorMapType>),
}

impl IteratorMapType {
    fn llvm_type<'ctx>(&self, ctx: &'ctx Context) -> inkwell::types::BasicTypeEnum<'ctx> {
        match self {
            Self::Boolean => ctx.bool_type().into(),
            Self::Primitive(primitive) => primitive.llvm_type(ctx),
            Self::VariableList(_) => super::list_value_llvm_type(ctx).into(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum IteratorSource<'iter, 'ctx> {
    Physical(&'iter IteratorHolder),
    Map {
        source: &'iter IteratorHolder,
        target: &'iter IteratorMapType,
        mapper: FunctionValue<'ctx>,
    },
}

impl<'iter, 'ctx> IteratorSource<'iter, 'ctx> {
    pub fn physical(source: &'iter IteratorHolder) -> Self {
        Self::Physical(source)
    }

    pub fn map(
        source: &'iter IteratorHolder,
        target: &'iter IteratorMapType,
        mapper: FunctionValue<'ctx>,
    ) -> Self {
        Self::Map {
            source,
            target,
            mapper,
        }
    }

    pub fn list_child(self) -> Option<Self> {
        match self {
            Self::Physical(IteratorHolder::Dictionary { values, .. })
            | Self::Physical(IteratorHolder::RunEnd { values, .. }) => {
                Self::Physical(values).list_child()
            }
            Self::Physical(IteratorHolder::List(source)) => Some(Self::Physical(source.child())),
            Self::Map {
                source: IteratorHolder::Dictionary { values, .. },
                target,
                mapper,
            }
            | Self::Map {
                source: IteratorHolder::RunEnd { values, .. },
                target,
                mapper,
            } => Self::Map {
                source: values,
                target,
                mapper,
            }
            .list_child(),
            Self::Map {
                source: IteratorHolder::List(source),
                target: IteratorMapType::VariableList(target),
                mapper,
            } => Some(Self::Map {
                source: source.child(),
                target,
                mapper,
            }),
            _ => None,
        }
    }

    pub fn description(self) -> String {
        match self {
            Self::Physical(source) => source.data_type().to_string(),
            Self::Map {
                source,
                target,
                mapper,
            } => {
                format!(
                    "map<{}, {:?}, {}>",
                    source.data_type(),
                    target,
                    mapper.get_name().to_string_lossy()
                )
            }
        }
    }

    pub fn generate_random_access(
        self,
        ctx: &'ctx Context,
        module: &Module<'ctx>,
    ) -> Option<FunctionValue<'ctx>> {
        let (source, target, mapper) = match self {
            IteratorSource::Physical(ih) => return ih.generate_random_access(ctx, module),
            IteratorSource::Map {
                source,
                target,
                mapper,
            } => (source, target, mapper),
        };

        let source_access = source.generate_random_access(ctx, module)?;
        if matches!(target, IteratorMapType::VariableList(_)) {
            return Some(source_access);
        }

        let target_llvm_type = target.llvm_type(ctx);
        let ptr_type = ctx.ptr_type(inkwell::AddressSpace::default());
        let fn_type = target_llvm_type.fn_type(&[ptr_type.into(), ctx.i64_type().into()], false);
        let mapper_name = mapper.get_name().to_string_lossy();
        let name = format!(
            "{}_map_c{}_{}_access",
            source.codegen_label(),
            mapper_name.len(),
            mapper_name
        );
        if let Some(existing) = module.get_function(&name) {
            debug_assert_eq!(existing.get_type(), fn_type);
            return Some(existing);
        }

        let access = module.add_function(&name, fn_type, Some(Linkage::Private));
        let iter_ptr = access.get_nth_param(0).unwrap();
        let index = access.get_nth_param(1).unwrap();
        let builder = ctx.create_builder();
        let entry = ctx.append_basic_block(access, "entry");
        builder.position_at_end(entry);
        let value = builder
            .build_call(
                source_access,
                &[iter_ptr.into(), index.into()],
                "map_source_value",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_basic();

        debug_assert_eq!(mapper.get_type().count_param_types(), 1);
        debug_assert_eq!(
            mapper.get_type().get_param_types()[0],
            source.codegen_info().value_type.llvm_type(ctx).into()
        );
        debug_assert_eq!(mapper.get_type().get_return_type(), Some(target_llvm_type));
        let value = builder
            .build_call(mapper, &[value.into()], "map_value")
            .unwrap()
            .try_as_basic_value()
            .unwrap_basic();
        debug_assert_eq!(value.get_type(), target_llvm_type);
        builder.build_return(Some(&value)).unwrap();
        Some(access)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::Int32Array;
    use inkwell::{context::Context, module::Linkage, OptimizationLevel};

    use super::{IteratorMapType, IteratorSource};
    use crate::{compiled_iter::array_to_iter, PrimitiveType};

    #[test]
    fn mapped_accessor_calls_an_arbitrary_element_mapper() {
        let values = Int32Array::from(vec![1, 2, 3]);
        let mut source = array_to_iter(&values);
        let context = Context::create();
        let module = context.create_module("mapped_iterator");
        let i32_type = context.i32_type();
        let mapper = module.add_function(
            "add_seven",
            i32_type.fn_type(&[i32_type.into()], false),
            Some(Linkage::Private),
        );
        let builder = context.create_builder();
        let entry = context.append_basic_block(mapper, "entry");
        builder.position_at_end(entry);
        let mapped = builder
            .build_int_add(
                mapper.get_nth_param(0).unwrap().into_int_value(),
                i32_type.const_int(7, false),
                "mapped",
            )
            .unwrap();
        builder.build_return(Some(&mapped)).unwrap();

        let target = IteratorMapType::Primitive(PrimitiveType::I32);
        let accessor = IteratorSource::map(&source, &target, mapper)
            .generate_random_access(&context, &module)
            .unwrap();
        let ptr_type = context.ptr_type(inkwell::AddressSpace::default());
        let test_accessor = module.add_function(
            "test_mapped_access",
            i32_type.fn_type(&[ptr_type.into(), context.i64_type().into()], false),
            None,
        );
        let builder = context.create_builder();
        let entry = context.append_basic_block(test_accessor, "entry");
        builder.position_at_end(entry);
        let mapped = builder
            .build_call(
                accessor,
                &[
                    test_accessor.get_nth_param(0).unwrap().into(),
                    test_accessor.get_nth_param(1).unwrap().into(),
                ],
                "mapped",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_basic();
        builder.build_return(Some(&mapped)).unwrap();
        module.verify().unwrap();
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let accessor = unsafe {
            execution_engine
                .get_function::<unsafe extern "C" fn(*mut c_void, u64) -> i32>("test_mapped_access")
                .unwrap()
        };

        unsafe {
            assert_eq!(accessor.call(source.get_mut_ptr(), 0), 8);
            assert_eq!(accessor.call(source.get_mut_ptr(), 1), 9);
            assert_eq!(accessor.call(source.get_mut_ptr(), 2), 10);
        }
    }
}
