use arrow_array::{Array, BooleanArray, UInt32Array};
use arrow_schema::DataType;
use inkwell::{
    context::Context,
    intrinsics::Intrinsic,
    module::{Linkage, Module},
    types::BasicTypeEnum,
    values::FunctionValue,
    AddressSpace, IntPredicate, OptimizationLevel,
};
use itertools::Itertools;
use ouroboros::self_referencing;

use crate::{
    declare_blocks,
    dsl::KernelParameters,
    new_iter::{datum_to_iter, generate_random_access},
    new_kernels::{cmp::add_memcmp, link_req_helpers, optimize_module, Kernel},
    ArrowKernelError, PrimitiveType,
};

fn get_ucmp<'a>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    ty: BasicTypeEnum<'a>,
) -> FunctionValue<'a> {
    let i8_type = ctx.i8_type();
    if let Some(ucmp) = Intrinsic::find("llvm.ucmp") {
        return ucmp
            .get_declaration(llvm_mod, &[i8_type.into(), ty])
            .unwrap();
    }

    let func = llvm_mod.add_function(
        "ucmp",
        i8_type.fn_type(&[ty.into(), ty.into()], false),
        Some(Linkage::Private),
    );

    declare_blocks!(ctx, func, entry);
    let b = ctx.create_builder();
    b.position_at_end(entry);
    let p1 = func.get_nth_param(0).unwrap().into_int_value();
    let p2 = func.get_nth_param(1).unwrap().into_int_value();
    let is_lt = b
        .build_int_compare(IntPredicate::ULT, p1, p2, "is_lt")
        .unwrap();
    let is_gt = b
        .build_int_compare(IntPredicate::UGT, p1, p2, "is_gt")
        .unwrap();
    let val = b
        .build_select(
            is_lt,
            i8_type.const_all_ones(),
            b.build_select(
                is_gt,
                i8_type.const_int(1, false),
                i8_type.const_zero(),
                "gt_or_else",
            )
            .unwrap()
            .into_int_value(),
            "lt_or_else",
        )
        .unwrap();
    b.build_return(Some(&val)).unwrap();

    func
}

fn get_scmp<'a>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    ty: BasicTypeEnum<'a>,
) -> FunctionValue<'a> {
    let i8_type = ctx.i8_type();
    if let Some(scmp) = Intrinsic::find("llvm.scmp") {
        return scmp
            .get_declaration(llvm_mod, &[i8_type.into(), ty])
            .unwrap();
    }

    let func = llvm_mod.add_function(
        "scmp",
        i8_type.fn_type(&[ty.into(), ty.into()], false),
        Some(Linkage::Private),
    );

    declare_blocks!(ctx, func, entry);
    let b = ctx.create_builder();
    b.position_at_end(entry);
    let p1 = func.get_nth_param(0).unwrap().into_int_value();
    let p2 = func.get_nth_param(1).unwrap().into_int_value();
    let is_lt = b
        .build_int_compare(IntPredicate::SLT, p1, p2, "is_lt")
        .unwrap();
    let is_gt = b
        .build_int_compare(IntPredicate::SGT, p1, p2, "is_gt")
        .unwrap();
    let val = b
        .build_select(
            is_lt,
            i8_type.const_all_ones(),
            b.build_select(
                is_gt,
                i8_type.const_int(1, false),
                i8_type.const_zero(),
                "gt_or_else",
            )
            .unwrap()
            .into_int_value(),
            "lt_or_else",
        )
        .unwrap();
    b.build_return(Some(&val)).unwrap();

    func
}

fn generate_single_cmp<'a>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    pt: PrimitiveType,
    reverse: bool,
    access1: FunctionValue<'a>,
    access2: FunctionValue<'a>,
) -> FunctionValue<'a> {
    let fname = format!("sort_cmp_{}", pt);
    if let Some(f) = llvm_mod.get_function(&fname) {
        return f;
    }

    let i64_type = ctx.i64_type();
    let i8_type = ctx.i8_type();
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let func = llvm_mod.add_function(
        &fname,
        i8_type.fn_type(&[ptr_type.into(), i64_type.into(), i64_type.into()], false),
        Some(Linkage::Private),
    );

    let b = ctx.create_builder();
    let llvm_type = pt.llvm_type(ctx);

    declare_blocks!(ctx, func, entry);
    b.position_at_end(entry);
    let data_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let idx1 = func.get_nth_param(1).unwrap().into_int_value();
    let idx2 = func.get_nth_param(2).unwrap().into_int_value();

    let val1 = b
        .build_call(access1, &[data_ptr.into(), idx1.into()], "val1")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left();
    let val2 = b
        .build_call(access2, &[data_ptr.into(), idx2.into()], "val2")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left();

    let cmp_f = match pt {
        PrimitiveType::I8 | PrimitiveType::I16 | PrimitiveType::I32 | PrimitiveType::I64 => {
            get_scmp(ctx, llvm_mod, llvm_type)
        }
        PrimitiveType::U8 | PrimitiveType::U16 | PrimitiveType::U32 | PrimitiveType::U64 => {
            get_ucmp(ctx, llvm_mod, llvm_type)
        }
        PrimitiveType::P64x2 => {
            let llvm_type = pt.llvm_type(ctx);
            let memcmp = add_memcmp(ctx, llvm_mod);
            let cmp_f = llvm_mod.add_function(
                "cmp_str",
                i8_type.fn_type(&[llvm_type.into(), llvm_type.into()], false),
                Some(Linkage::Private),
            );
            declare_blocks!(ctx, cmp_f, entry);
            let b2 = ctx.create_builder();
            b2.position_at_end(entry);
            let v1 = cmp_f.get_nth_param(0).unwrap().into_struct_value();
            let v2 = cmp_f.get_nth_param(1).unwrap().into_struct_value();
            let res = b2
                .build_call(memcmp, &[v1.into(), v2.into()], "memcmp")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            let res = b2
                .build_call(
                    get_scmp(ctx, llvm_mod, i64_type.into()),
                    &[res.into(), i64_type.const_zero().into()],
                    "scmp_res",
                )
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            b2.build_return(Some(&res)).unwrap();
            cmp_f
        }
        PrimitiveType::F16 => todo!(),
        PrimitiveType::F32 => todo!(),
        PrimitiveType::F64 => todo!(),
    };

    let cmp = if reverse {
        b.build_call(cmp_f, &[val2.into(), val1.into()], "cmp")
    } else {
        b.build_call(cmp_f, &[val1.into(), val2.into()], "cmp")
    }
    .unwrap()
    .try_as_basic_value()
    .unwrap_left()
    .into_int_value();

    b.build_return(Some(&cmp)).unwrap();
    func
}

fn fill_in_cmp<'a>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    pts: &[(
        PrimitiveType,
        Option<FunctionValue<'a>>,
        SortOptions,
        FunctionValue<'a>,
        FunctionValue<'a>,
    )],
) {
    let mut all_cmps = Vec::new();
    for (pt, _nullable, opts, access1, access2) in pts {
        let cmp = generate_single_cmp(ctx, llvm_mod, *pt, opts.descending, *access1, *access2);
        all_cmps.push(cmp);
    }

    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let cmp = llvm_mod.add_function(
        "cmp",
        ctx.i8_type().fn_type(
            &[
                ptr_type.into(),
                ctx.i64_type().into(),
                ctx.i64_type().into(),
            ],
            false,
        ),
        None,
    );
    let i8_type = ctx.i8_type();
    let i8_one = i8_type.const_int(1, false);
    let i8_neg_one = i8_type.const_all_ones();
    let b = ctx.create_builder();
    declare_blocks!(ctx, cmp, entry, final_exit);
    b.position_at_end(entry);
    let data_ptr = cmp.get_nth_param(0).unwrap().into_pointer_value();
    let idx1 = cmp.get_nth_param(1).unwrap().into_int_value();
    let idx2 = cmp.get_nth_param(2).unwrap().into_int_value();

    let mut bitmap_ptrs = Vec::new();
    let mut data_ptrs = Vec::new();

    let mut idx = 0;
    for (_, nullable, _, _, _) in pts.iter() {
        if nullable.is_some() {
            let ptr = KernelParameters::llvm_get(ctx, &b, data_ptr, idx);
            bitmap_ptrs.push(Some(ptr));
            idx += 1;
        } else {
            bitmap_ptrs.push(None);
        }

        let ptr = KernelParameters::llvm_get(ctx, &b, data_ptr, idx);
        data_ptrs.push(ptr);
        idx += 1;
    }

    let all_entry_blocks = all_cmps
        .iter()
        .map(|_| ctx.append_basic_block(cmp, "entry"))
        .collect_vec();
    b.build_unconditional_branch(all_entry_blocks[0]).unwrap();

    for (idx, cmp_f) in all_cmps.into_iter().enumerate() {
        declare_blocks!(ctx, cmp, not_null, nonzero);
        b.position_at_end(all_entry_blocks[idx]);
        if let Some(null_access) = pts[idx].1 {
            declare_blocks!(ctx, cmp, lhs_null, rhs_null, both_null);
            let bitmap_ptr = bitmap_ptrs[idx].unwrap();
            let lhs_null_bit = b
                .build_call(null_access, &[bitmap_ptr.into(), idx1.into()], "lhs_null")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            let rhs_null_bit = b
                .build_call(null_access, &[bitmap_ptr.into(), idx2.into()], "rhs_null")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            let rhs_shifted = b
                .build_left_shift(rhs_null_bit, i8_type.const_int(1, false), "shifted")
                .unwrap();
            let sum = b.build_or(lhs_null_bit, rhs_shifted, "sum").unwrap();
            b.build_switch(
                sum,
                not_null,
                &[
                    (i8_type.const_int(0, false), not_null),
                    (i8_type.const_int(1, false), lhs_null),
                    (i8_type.const_int(2, false), rhs_null),
                    (i8_type.const_int(3, false), both_null),
                ],
            )
            .unwrap();

            b.position_at_end(lhs_null);
            if pts[idx].2.nulls_first {
                b.build_return(Some(&i8_one)).unwrap();
            } else {
                b.build_return(Some(&i8_neg_one)).unwrap();
            }

            b.position_at_end(rhs_null);
            if pts[idx].2.nulls_first {
                b.build_return(Some(&i8_neg_one)).unwrap();
            } else {
                b.build_return(Some(&i8_one)).unwrap();
            }

            b.position_at_end(both_null);
            b.build_return(Some(&i8_type.const_zero())).unwrap();
        } else {
            b.build_unconditional_branch(not_null).unwrap();
        }

        b.position_at_end(not_null);
        let res = b
            .build_call(
                cmp_f,
                &[data_ptrs[idx].into(), idx1.into(), idx2.into()],
                "cmp",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        let is_zero = b
            .build_int_compare(IntPredicate::EQ, res, i8_type.const_zero(), "is_zero")
            .unwrap();
        b.build_conditional_branch(
            is_zero,
            if idx + 1 < pts.len() {
                all_entry_blocks[idx + 1]
            } else {
                final_exit
            },
            nonzero,
        )
        .unwrap();

        b.position_at_end(nonzero);
        b.build_return(Some(&res)).unwrap();
    }

    b.position_at_end(final_exit);
    let f = get_ucmp(ctx, llvm_mod, ctx.i64_type().into());
    let res = b
        .build_call(f, &[idx1.into(), idx2.into()], "idx_cmp")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    b.build_return(Some(&res)).unwrap();
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy, Default)]
pub struct SortOptions {
    descending: bool,
    nulls_first: bool,
}

impl SortOptions {
    pub fn reverse(self) -> Self {
        Self {
            descending: !self.descending,
            nulls_first: !self.nulls_first,
        }
    }
}

use inkwell::execution_engine::JitFunction;
use std::ffi::c_void;

#[self_referencing]
pub struct SortKernel {
    context: Context,
    nullable: Vec<bool>,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, u64, u64) -> i8>,
}

unsafe impl Sync for SortKernel {}
unsafe impl Send for SortKernel {}

impl Kernel for SortKernel {
    type Key = Vec<(DataType, bool, SortOptions)>;

    type Input<'a> = Vec<&'a dyn Array>;

    type Params = Vec<SortOptions>;

    type Output = UInt32Array;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, super::ArrowKernelError> {
        let len = inp[0].len();
        assert!(
            inp.iter().all(|arr| arr.len() == len),
            "input arrays must have the same length"
        );

        let mut null_arrays = Vec::new();
        let mut ihs = Vec::new();
        let mut ptrs = Vec::new();

        for (arr, nullable) in inp.iter().zip(self.borrow_nullable().iter()) {
            if let Some(nulls) = arr.logical_nulls() {
                assert!(nullable, "kernel expected nullable input");
                let ba = BooleanArray::from(nulls.inner().clone());
                null_arrays.push(ba);
                ihs.push(datum_to_iter(&null_arrays.last().unwrap()).unwrap());
                ptrs.push(ihs.last_mut().unwrap().get_mut_ptr());
            } else {
                assert!(!nullable, "kernel expected non-nullable input");
            }

            ihs.push(datum_to_iter(arr).unwrap());
            ptrs.push(ihs.last_mut().unwrap().get_mut_ptr());
        }

        let mut kp = KernelParameters::new(ptrs);
        let mut perm = (0..len as u32).collect_vec();
        let kp_ptr = kp.get_mut_ptr();
        let func = self.borrow_func();
        perm.sort_unstable_by(|a, b| {
            let res = unsafe { func.call(kp_ptr, *a as u64, *b as u64) };
            debug_assert!(res == -1 || res == 0 || res == 1);
            unsafe { std::mem::transmute(res) }
        });

        let res = UInt32Array::from(perm);
        Ok(res)
    }

    fn compile(
        inp: &Self::Input<'_>,
        params: Self::Params,
    ) -> Result<Self, super::ArrowKernelError> {
        let ctx = Context::create();
        SortKernelTryBuilder {
            context: ctx,
            nullable: inp.iter().map(|x| x.is_nullable()).collect_vec(),
            func_builder: |ctx| generate_sort(ctx, inp, &params),
        }
        .try_build()
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        let mut keys = Vec::new();
        for (arr, opts) in i.iter().zip(p.iter()) {
            let nullable = arr.is_nullable();
            keys.push((arr.data_type().clone(), nullable, *opts));
        }
        Ok(keys)
    }
}

fn generate_sort<'a>(
    ctx: &'a Context,
    inp: &[&dyn Array],
    params: &[SortOptions],
) -> Result<JitFunction<'a, unsafe extern "C" fn(*mut c_void, u64, u64) -> i8>, ArrowKernelError> {
    let llvm_mod = ctx.create_module("test");

    let mut config = Vec::new();
    for (idx, (arr, opts)) in inp.iter().zip(params.iter()).enumerate() {
        let ptype = PrimitiveType::for_arrow_type(arr.data_type());
        let null_access = if let Some(nulls) = arr.logical_nulls() {
            let ba = BooleanArray::from(nulls.inner().clone());
            let ih = datum_to_iter(&ba)?;
            Some(
                generate_random_access(
                    ctx,
                    &llvm_mod,
                    &format!("null{}", idx),
                    &DataType::Boolean,
                    &ih,
                )
                .unwrap(),
            )
        } else {
            None
        };

        let ih = datum_to_iter(arr)?;
        let access = generate_random_access(
            ctx,
            &llvm_mod,
            &format!("access{}", idx),
            arr.data_type(),
            &ih,
        )
        .unwrap();

        config.push((ptype, null_access, *opts, access, access));
    }
    fill_in_cmp(ctx, &llvm_mod, &config);

    llvm_mod.verify().unwrap();
    optimize_module(&llvm_mod)?;
    let ee = llvm_mod
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();
    link_req_helpers(&llvm_mod, &ee)?;

    let cmp_func = unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, u64, u64) -> i8>("cmp")
            .unwrap()
    };

    Ok(cmp_func)
}

#[cfg(test)]
mod test {
    use crate::{
        new_kernels::sort::{SortKernel, SortOptions},
        Kernel,
    };
    use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, StringArray, UInt32Array};
    use itertools::Itertools;
    use std::sync::Arc;

    #[test]
    fn test_sort_i32_nonnull_fwd() {
        let data = vec![-10, 2, 7, 100, -274, 18, -22, 10000, -193, 18, 22, 12000];
        let mut perm = (0..data.len() as u32).collect_vec();
        perm.sort_by_key(|x| data[*x as usize]);
        let arr = Int32Array::from(data.clone());
        let input = vec![&arr as &dyn Array];

        let k = SortKernel::compile(&input, vec![SortOptions::default()]).unwrap();

        let our_res = k.call(input).unwrap();
        assert_eq!(our_res, UInt32Array::from(perm));
    }

    #[test]
    fn test_sort_i32_dups() {
        let data = (0..1000).map(|_| 1).collect_vec();
        let data = Arc::new(Int32Array::from(data)) as ArrayRef;
        let input = vec![&data as &dyn Array];

        let k = SortKernel::compile(&input, vec![SortOptions::default()]).unwrap();
        k.call(input).unwrap();
    }

    #[test]
    fn test_sort_i32_null() {
        let mut rng = fastrand::Rng::with_seed(42);
        let data = (0..32)
            .map(|_| {
                if rng.bool() {
                    Some(rng.i32(-1000..1000))
                } else {
                    None
                }
            })
            .collect_vec();

        let arr = Arc::new(Int32Array::from(data.clone())) as ArrayRef;
        let input = vec![&arr as &dyn Array];

        let k = SortKernel::compile(&input, vec![SortOptions::default()]).unwrap();
        let our_res = k
            .call(input.clone())
            .unwrap()
            .into_iter()
            .map(|x| x.unwrap())
            .collect_vec();

        let num_nulls = arr.null_count();
        for i in 0..num_nulls {
            assert!(
                arr.is_null(our_res[our_res.len() - i - 1] as usize),
                "index {} (null index {}) was not null",
                our_res[our_res.len() - i - 1],
                i
            );
        }

        let k = SortKernel::compile(&input, vec![SortOptions::default().reverse()]).unwrap();
        let our_res = k
            .call(input)
            .unwrap()
            .into_iter()
            .map(|x| x.unwrap())
            .collect_vec();

        let num_nulls = arr.null_count();
        for i in 0..num_nulls {
            assert!(
                arr.is_null(our_res[i] as usize),
                "index {} (null index {}) was not null",
                our_res[our_res.len() - i - 1],
                i
            );
        }
    }

    #[test]
    fn test_sort_i32_nonnull_rev() {
        let data = vec![-10, 2, 7, 100, -274, 18, -22, 10000, -193, 18, 22, 12000];
        let mut perm = (0..data.len() as u32).collect_vec();
        perm.sort_by_key(|x| -data[*x as usize]);
        let data = Arc::new(Int32Array::from(data)) as ArrayRef;
        let input = vec![&data as &dyn Array];

        let k = SortKernel::compile(&input, vec![SortOptions::default().reverse()]).unwrap();
        let our_res = k.call(input).unwrap();
        assert_eq!(our_res, UInt32Array::from(perm));
    }

    #[test]
    fn test_sort_multiarray_nonull() {
        let mut rng = fastrand::Rng::with_seed(42);
        let mut values = (0..1000).map(|_| rng.i32(..)).unique().collect_vec();

        rng.shuffle(&mut values);
        let arr1_data = values.clone();
        let arr1 = Int32Array::from(arr1_data.clone());

        rng.shuffle(&mut values);
        let arr2_data = values.clone();
        let arr2 = Int64Array::from(arr2_data.iter().cloned().map(|x| x as i64).collect_vec());

        let mut perm = (0..values.len() as u32).collect_vec();
        perm.sort_by_key(|x| (arr1_data[*x as usize], arr2_data[*x as usize]));

        let input = vec![&arr1 as &dyn Array, &arr2 as &dyn Array];
        let k = SortKernel::compile(&input, vec![SortOptions::default()]).unwrap();
        let our_res = k.call(input).unwrap();
        assert_eq!(our_res, UInt32Array::from(perm));
    }

    #[test]
    fn test_sort_string() {
        let data = StringArray::from(vec!["hello", "this", "is", "a", "test"]);

        let k =
            SortKernel::compile(&vec![&data as &dyn Array], vec![SortOptions::default()]).unwrap();
        let res = k.call(vec![&data]).unwrap();
        let res = res.iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(res, vec![3, 0, 2, 4, 1]);
    }
}
