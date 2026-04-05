use inkwell::{
    context::Context, module::Module, types::BasicType, values::FunctionValue, AddressSpace,
};

use crate::{increment_pointer, PrimitiveType};

pub fn generate_buffer_accessor<'a>(
    ctx: &'a Context,
    module: &Module<'a>,
    pt: PrimitiveType,
) -> FunctionValue<'a> {
    let func_name = format!("buffer_accessor_{}", pt);
    if let Some(func) = module.get_function(&func_name) {
        return func;
    }

    let ret_ty = pt.llvm_type(ctx);
    let ptr_ty = ctx.ptr_type(AddressSpace::default());
    let i64_ty = ctx.i64_type();
    let func_ty = ret_ty.fn_type(&[ptr_ty.into(), i64_ty.into()], false);

    let func = module.add_function(&func_name, func_ty, None);
    let entry = ctx.append_basic_block(func, "entry");
    let b = ctx.create_builder();
    b.position_at_end(entry);

    let ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let idx = func.get_nth_param(1).unwrap().into_int_value();

    let ptr_to_elem = increment_pointer!(ctx, b, ptr, pt.width(), idx);
    let el = b.build_load(ret_ty, ptr_to_elem, "load_el").unwrap();
    b.build_return(Some(&el)).unwrap();

    func
}

pub fn generate_buffer_writer<'a>(
    ctx: &'a Context,
    module: &Module<'a>,
    pt: PrimitiveType,
) -> FunctionValue<'a> {
    let func_name = format!("buffer_writer_{}", pt);
    if let Some(func) = module.get_function(&func_name) {
        return func;
    }

    let write_ty = pt.llvm_type(ctx);
    let ptr_ty = ctx.ptr_type(AddressSpace::default());
    let i64_ty = ctx.i64_type();
    let func_ty = ctx
        .void_type()
        .fn_type(&[ptr_ty.into(), i64_ty.into(), write_ty.into()], false);

    let func = module.add_function(&func_name, func_ty, None);
    let entry = ctx.append_basic_block(func, "entry");
    let b = ctx.create_builder();
    b.position_at_end(entry);

    let ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let idx = func.get_nth_param(1).unwrap().into_int_value();
    let value = func.get_nth_param(2).unwrap();

    let ptr_to_elem = increment_pointer!(ctx, b, ptr, pt.width(), idx);
    b.build_store(ptr_to_elem, value).unwrap();
    b.build_return(None).unwrap();

    func
}
