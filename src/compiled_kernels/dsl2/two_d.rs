use std::os::raw::c_void;

use arrow_array::Datum;
use inkwell::{
    context::Context, module::Module, types::BasicType, values::FunctionValue, AddressSpace,
};
use itertools::Itertools;

use crate::{
    compiled_iter::{datum_to_iter, generate_random_access, IteratorHolder},
    increment_pointer, mark_load_invariant, ArrowKernelError, PrimitiveType,
};

pub struct TwoDArrayRuntime {
    ptrs: Vec<*const c_void>,
    _ihs: Vec<IteratorHolder>,
}

impl TwoDArrayRuntime {
    pub fn new(arrs: &[&dyn Datum]) -> Result<Self, ArrowKernelError> {
        let ihs: Vec<IteratorHolder> = arrs.iter().map(|arr| datum_to_iter(*arr)).try_collect()?;
        let ptrs = ihs.iter().map(|x| x.get_ptr()).collect_vec();
        Ok(Self { ptrs, _ihs: ihs })
    }

    pub fn get_ptr(&self) -> *const c_void {
        self.ptrs.as_ptr() as *const c_void
    }
}

pub fn generate_two_d_access<'ctx, 'args>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    arrs: &[&'args dyn Datum],
) -> Result<FunctionValue<'ctx>, ArrowKernelError> {
    let witness_type = PrimitiveType::for_arrow_type(arrs[0].get().0.data_type());
    let mut funcs = Vec::new();
    for (idx, arr) in arrs.iter().copied().enumerate() {
        let ty = PrimitiveType::for_arrow_type(arr.get().0.data_type());
        if witness_type != ty {
            return Err(ArrowKernelError::Inconsistent2DArray(witness_type, ty));
        }

        let ih = datum_to_iter(arr)?;
        let func = generate_random_access(
            ctx,
            module,
            &format!("two_d_accessor_{}", idx),
            arr.get().0.data_type(),
            &ih,
        )
        .unwrap();
        funcs.push(func);
    }

    let ret_ty = witness_type.llvm_type(ctx);
    let ptr_ty = ctx.ptr_type(AddressSpace::default());
    let i64_ty = ctx.i64_type();
    let func_ty = ret_ty.fn_type(&[ptr_ty.into(), i64_ty.into(), i64_ty.into()], false);
    let func = module.add_function("access_twod", func_ty, None);

    let b = ctx.create_builder();
    let entry = ctx.append_basic_block(func, "entry");
    b.position_at_end(entry);
    let base_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let row = func.get_nth_param(1).unwrap().into_int_value();
    let col = func.get_nth_param(2).unwrap().into_int_value();

    let fail_arm = ctx.append_basic_block(func, "fail");
    b.position_at_end(fail_arm);
    b.build_return(Some(&ret_ty.const_zero())).unwrap();

    let switch_arms = funcs
        .iter()
        .enumerate()
        .map(|(idx, accessor)| {
            let block = ctx.append_basic_block(func, &format!("arm{}", idx));
            b.position_at_end(block);
            let arr_ptr_ptr = increment_pointer!(ctx, b, base_ptr, 8 * idx);
            let arr_ptr = b.build_load(ptr_ty, arr_ptr_ptr, "arr_ptr").unwrap();
            mark_load_invariant!(ctx, arr_ptr);

            let val = b
                .build_call(*accessor, &[arr_ptr.into(), col.into()], "access_call")
                .unwrap()
                .try_as_basic_value()
                .unwrap_basic();
            b.build_return(Some(&val)).unwrap();
            (ctx.i64_type().const_int(idx as u64, false), block)
        })
        .collect_vec();

    b.position_at_end(entry);
    b.build_switch(row, fail_arm, &switch_arms).unwrap();

    Ok(func)
}
