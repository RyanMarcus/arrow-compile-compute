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
    compiled_iter::{datum_to_iter, generate_random_access},
    compiled_kernels::{
        cmp::{add_float_to_int, add_memcmp},
        dsl::KernelParameters,
        link_req_helpers, optimize_module, Kernel,
    },
    declare_blocks, logical_nulls, ArrowKernelError, PrimitiveType,
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
    sort_opts: SortOptions,
    access1: FunctionValue<'a>,
    nulls1: Option<FunctionValue<'a>>,
    access2: FunctionValue<'a>,
    nulls2: Option<FunctionValue<'a>>,
) -> FunctionValue<'a> {
    let fname = format!("sort_cmp_{}", pt);
    if let Some(f) = llvm_mod.get_function(&fname) {
        return f;
    }

    let i64_type = ctx.i64_type();
    let i8_type = ctx.i8_type();
    let i8_neg_one = i8_type.const_all_ones();
    let i8_one = i8_type.const_int(1, true);

    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let func = llvm_mod.add_function(
        &fname,
        i8_type.fn_type(
            &[
                ptr_type.into(),
                ptr_type.into(),
                ptr_type.into(),
                ptr_type.into(),
                i64_type.into(),
                i64_type.into(),
            ],
            false,
        ),
        Some(Linkage::Private),
    );

    let b = ctx.create_builder();
    let llvm_type = pt.llvm_type(ctx);

    declare_blocks!(
        ctx,
        func,
        entry,
        lhs_null,
        rhs_null,
        check_nulls,
        both_null,
        only_lhs_null,
        only_rhs_null,
        do_cmp
    );
    b.position_at_end(entry);
    let data_ptr1 = func.get_nth_param(0).unwrap().into_pointer_value();
    let null_ptr1 = func.get_nth_param(1).unwrap().into_pointer_value();
    let data_ptr2 = func.get_nth_param(2).unwrap().into_pointer_value();
    let null_ptr2 = func.get_nth_param(3).unwrap().into_pointer_value();
    let idx1 = func.get_nth_param(4).unwrap().into_int_value();
    let idx2 = func.get_nth_param(5).unwrap().into_int_value();
    let lhs_is_valid = b.build_alloca(ctx.bool_type(), "lhs_validity_ptr").unwrap();
    b.build_store(lhs_is_valid, ctx.bool_type().const_all_ones())
        .unwrap();
    let rhs_is_valid = b.build_alloca(ctx.bool_type(), "rhs_validity_ptr").unwrap();
    b.build_store(rhs_is_valid, ctx.bool_type().const_all_ones())
        .unwrap();
    b.build_unconditional_branch(lhs_null).unwrap();

    b.position_at_end(lhs_null);
    if let Some(null_accessor) = nulls1 {
        let validity_bit = b
            .build_call(
                null_accessor,
                &[null_ptr1.into(), idx1.into()],
                "lhs_null_call",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        b.build_store(lhs_is_valid, validity_bit).unwrap();
    }
    b.build_unconditional_branch(rhs_null).unwrap();

    b.position_at_end(rhs_null);
    if let Some(null_accessor) = nulls2 {
        let validity_bit = b
            .build_call(
                null_accessor,
                &[null_ptr2.into(), idx2.into()],
                "rhs_null_call",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        b.build_store(rhs_is_valid, validity_bit).unwrap();
    }
    b.build_unconditional_branch(check_nulls).unwrap();

    b.position_at_end(check_nulls);
    let lhs_null = b
        .build_load(ctx.bool_type(), lhs_is_valid, "lhs_null")
        .unwrap()
        .into_int_value();
    let rhs_null = b
        .build_load(ctx.bool_type(), rhs_is_valid, "rhs_null")
        .unwrap()
        .into_int_value();

    let lhs_as_i8 = b
        .build_int_z_extend(lhs_null, ctx.i8_type(), "lhs_as_i8")
        .unwrap();

    let combined = b
        .build_or(
            lhs_as_i8,
            b.build_select(
                rhs_null,
                ctx.i8_type().const_int(2, false),
                i8_type.const_zero(),
                "shifted",
            )
            .unwrap()
            .into_int_value(),
            "combined",
        )
        .unwrap();
    b.build_switch(
        combined,
        do_cmp,
        &[
            (ctx.i8_type().const_int(0, false), both_null),
            (ctx.i8_type().const_int(1, false), only_rhs_null),
            (ctx.i8_type().const_int(2, false), only_lhs_null),
            (ctx.i8_type().const_int(3, false), do_cmp),
        ],
    )
    .unwrap();

    b.position_at_end(only_rhs_null);
    let r = if sort_opts.nulls_first {
        i8_one
    } else {
        i8_neg_one
    };
    b.build_return(Some(&r)).unwrap();

    b.position_at_end(only_lhs_null);
    let r = if sort_opts.nulls_first {
        i8_neg_one
    } else {
        i8_one
    };
    b.build_return(Some(&r)).unwrap();

    b.position_at_end(both_null);
    b.build_return(Some(&ctx.i8_type().const_zero())).unwrap();

    b.position_at_end(do_cmp);
    let val1 = b
        .build_call(access1, &[data_ptr1.into(), idx1.into()], "val1")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left();
    let val2 = b
        .build_call(access2, &[data_ptr2.into(), idx2.into()], "val2")
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
        PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
            let f_to_i = add_float_to_int(ctx, llvm_mod, pt);
            let llvm_type = pt.llvm_type(ctx);
            let cmp_f = llvm_mod.add_function(
                &format!("cmp_{}", pt),
                i8_type.fn_type(&[llvm_type.into(), llvm_type.into()], false),
                Some(Linkage::Private),
            );
            declare_blocks!(ctx, cmp_f, entry);
            let b2 = ctx.create_builder();
            b2.position_at_end(entry);
            let v1 = cmp_f.get_nth_param(0).unwrap().into_float_value();
            let v2 = cmp_f.get_nth_param(1).unwrap().into_float_value();
            let v1_int = b2
                .build_call(f_to_i, &[v1.into()], "f_to_i1")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            let v2_int = b2
                .build_call(f_to_i, &[v2.into()], "f_to_i2")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            let res = b2
                .build_call(
                    get_scmp(
                        ctx,
                        llvm_mod,
                        PrimitiveType::int_with_width(pt.width()).llvm_type(ctx),
                    ),
                    &[v1_int.into(), v2_int.into()],
                    "scmp_res",
                )
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            b2.build_return(Some(&res)).unwrap();
            cmp_f
        }
        PrimitiveType::List(_, _) => todo!(),
    };

    let cmp = if sort_opts.descending {
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

struct ComparatorField<'a> {
    ptype: PrimitiveType,
    sort_options: SortOptions,
    lhs_nulls: Option<FunctionValue<'a>>,
    lhs_accessor: FunctionValue<'a>,
    rhs_nulls: Option<FunctionValue<'a>>,
    rhs_accessor: FunctionValue<'a>,
}

enum BreakTies {
    ByIndex,
    DoNotBreakTies,
}

fn fill_in_cmp<'a>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    pts: &[ComparatorField<'a>],
    break_ties: BreakTies,
) {
    let mut all_cmps = Vec::new();
    for cf in pts.iter() {
        let cmp = generate_single_cmp(
            ctx,
            llvm_mod,
            cf.ptype,
            cf.sort_options,
            cf.lhs_accessor,
            cf.lhs_nulls,
            cf.rhs_accessor,
            cf.rhs_nulls,
        );
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
    let b = ctx.create_builder();
    declare_blocks!(ctx, cmp, entry, final_exit);
    b.position_at_end(entry);
    let data_ptr = cmp.get_nth_param(0).unwrap().into_pointer_value();
    let idx1 = cmp.get_nth_param(1).unwrap().into_int_value();
    let idx2 = cmp.get_nth_param(2).unwrap().into_int_value();
    let mut cmp_ptrs = Vec::new();

    let mut idx = 0;
    for cf in pts.iter() {
        let lhs_data_ptr = KernelParameters::llvm_get(ctx, &b, data_ptr, idx);
        idx += 1;
        let lhs_null_ptr = if cf.lhs_nulls.is_some() {
            let ptr = KernelParameters::llvm_get(ctx, &b, data_ptr, idx);
            idx += 1;
            ptr
        } else {
            ptr_type.const_null()
        };

        let rhs_data_ptr = KernelParameters::llvm_get(ctx, &b, data_ptr, idx);
        idx += 1;
        let rhs_null_ptr = if cf.rhs_nulls.is_some() {
            let ptr = KernelParameters::llvm_get(ctx, &b, data_ptr, idx);
            idx += 1;
            ptr
        } else {
            ptr_type.const_null()
        };

        cmp_ptrs.push((lhs_data_ptr, lhs_null_ptr, rhs_data_ptr, rhs_null_ptr));
    }

    let all_entry_blocks = all_cmps
        .iter()
        .map(|_| ctx.append_basic_block(cmp, "entry"))
        .collect_vec();
    b.build_unconditional_branch(all_entry_blocks[0]).unwrap();

    for (idx, (cmp_f, cmp_ptr)) in all_cmps.into_iter().zip(cmp_ptrs.into_iter()).enumerate() {
        declare_blocks!(ctx, cmp, nonzero);
        b.position_at_end(all_entry_blocks[idx]);
        let res = b
            .build_call(
                cmp_f,
                &[
                    cmp_ptr.0.into(),
                    cmp_ptr.1.into(),
                    cmp_ptr.2.into(),
                    cmp_ptr.3.into(),
                    idx1.into(),
                    idx2.into(),
                ],
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
    match break_ties {
        BreakTies::ByIndex => {
            let f = get_ucmp(ctx, llvm_mod, ctx.i64_type().into());
            let res = b
                .build_call(f, &[idx1.into(), idx2.into()], "idx_cmp")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            b.build_return(Some(&res)).unwrap();
        }
        BreakTies::DoNotBreakTies => {
            b.build_return(Some(&i8_type.const_zero())).unwrap();
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy, Default)]
pub struct SortOptions {
    pub descending: bool,
    pub nulls_first: bool,
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
            ihs.push(datum_to_iter(arr).unwrap());
            let data_ptr = ihs.last_mut().unwrap().get_mut_ptr();
            if let Some(nulls) = logical_nulls(*arr)? {
                assert!(nullable, "kernel expected nullable input");
                let ba = BooleanArray::from(nulls.inner().clone());
                null_arrays.push(ba);
                ihs.push(datum_to_iter(&null_arrays.last().unwrap()).unwrap());
                let nulls_ptr = ihs.last_mut().unwrap().get_mut_ptr();
                ptrs.push(data_ptr);
                ptrs.push(nulls_ptr);
                ptrs.push(data_ptr);
                ptrs.push(nulls_ptr);
            } else {
                assert!(!nullable, "kernel expected non-nullable input");
                ptrs.push(data_ptr);
                ptrs.push(data_ptr);
            }
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

#[self_referencing]
pub struct TopKKernel {
    context: Context,
    nullable: Vec<bool>,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, u64, u64) -> i8>,
}

unsafe impl Sync for TopKKernel {}
unsafe impl Send for TopKKernel {}

impl Kernel for TopKKernel {
    type Key = Vec<(DataType, bool, SortOptions)>;

    type Input<'a> = (Vec<&'a dyn Array>, usize);

    type Params = Vec<SortOptions>;

    type Output = UInt32Array;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, super::ArrowKernelError> {
        let (arrs, k) = inp;
        let len = arrs[0].len();
        assert!(
            arrs.iter().all(|arr| arr.len() == len),
            "input arrays must have the same length"
        );

        let mut null_arrays = Vec::new();
        let mut ihs = Vec::new();
        let mut ptrs = Vec::new();

        for (arr, nullable) in arrs.iter().zip(self.borrow_nullable().iter()) {
            ihs.push(datum_to_iter(arr).unwrap());
            let data_ptr = ihs.last_mut().unwrap().get_mut_ptr();
            if let Some(nulls) = logical_nulls(*arr)? {
                assert!(nullable, "kernel expected nullable input");
                let ba = BooleanArray::from(nulls.inner().clone());
                null_arrays.push(ba);
                ihs.push(datum_to_iter(&null_arrays.last().unwrap()).unwrap());
                let nulls_ptr = ihs.last_mut().unwrap().get_mut_ptr();
                ptrs.push(data_ptr);
                ptrs.push(nulls_ptr);
                ptrs.push(data_ptr);
                ptrs.push(nulls_ptr);
            } else {
                assert!(!nullable, "kernel expected non-nullable input");
                ptrs.push(data_ptr);
                ptrs.push(data_ptr);
            }
        }

        let mut kp = KernelParameters::new(ptrs);
        let mut perm = (0..len as u32).collect_vec();
        let kp_ptr = kp.get_mut_ptr();
        let func = self.borrow_func();

        if k >= perm.len() {
            return Ok(UInt32Array::from(perm));
        }

        perm.select_nth_unstable_by(k, |a, b| {
            let res = unsafe { func.call(kp_ptr, *a as u64, *b as u64) };
            debug_assert!(res == -1 || res == 0 || res == 1);
            unsafe { std::mem::transmute(res) }
        });

        let mut perm = perm[..k].to_vec();
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
        TopKKernelTryBuilder {
            context: ctx,
            nullable: inp.0.iter().map(|x| x.is_nullable()).collect_vec(),
            func_builder: |ctx| generate_sort(ctx, &inp.0, &params),
        }
        .try_build()
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        let mut keys = Vec::new();
        for (arr, opts) in i.0.iter().zip(p.iter()) {
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
        let null_access = if let Some(nulls) = logical_nulls(*arr)? {
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

        config.push(ComparatorField {
            ptype,
            sort_options: *opts,
            lhs_nulls: null_access,
            lhs_accessor: access,
            rhs_nulls: null_access,
            rhs_accessor: access,
        });
    }
    fill_in_cmp(ctx, &llvm_mod, &config, BreakTies::ByIndex);

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

fn generate_lower_bound_cmp<'a>(
    ctx: &'a Context,
    keys: &[&dyn Array],
    sorted: &[&dyn Array],
    params: &[SortOptions],
) -> Result<JitFunction<'a, unsafe extern "C" fn(*mut c_void, u64, u64) -> i8>, ArrowKernelError> {
    let llvm_mod = ctx.create_module("lower_bound");

    let mut config = Vec::new();
    for (idx, ((key_arr, sorted_arr), opts)) in keys
        .iter()
        .zip(sorted.iter())
        .zip(params.iter())
        .enumerate()
    {
        let key_ptype = PrimitiveType::for_arrow_type(key_arr.data_type());
        let sorted_ptype = PrimitiveType::for_arrow_type(sorted_arr.data_type());
        if key_ptype != sorted_ptype {
            return Err(ArrowKernelError::TypeMismatch(key_ptype, sorted_ptype));
        }

        let key_null_access = if let Some(nulls) = logical_nulls(*key_arr)? {
            let ba = BooleanArray::from(nulls.inner().clone());
            let ih = datum_to_iter(&ba)?;
            Some(
                generate_random_access(
                    ctx,
                    &llvm_mod,
                    &format!("key_null{}", idx),
                    &DataType::Boolean,
                    &ih,
                )
                .unwrap(),
            )
        } else {
            None
        };

        let key_iter = datum_to_iter(&*key_arr)?;
        let key_access = generate_random_access(
            ctx,
            &llvm_mod,
            &format!("key_access{}", idx),
            key_arr.data_type(),
            &key_iter,
        )
        .unwrap();

        let sorted_null_access = if let Some(nulls) = logical_nulls(*sorted_arr)? {
            let ba = BooleanArray::from(nulls.inner().clone());
            let ih = datum_to_iter(&ba)?;
            Some(
                generate_random_access(
                    ctx,
                    &llvm_mod,
                    &format!("sorted_null{}", idx),
                    &DataType::Boolean,
                    &ih,
                )
                .unwrap(),
            )
        } else {
            None
        };

        let sorted_iter = datum_to_iter(&*sorted_arr)?;
        let sorted_access = generate_random_access(
            ctx,
            &llvm_mod,
            &format!("sorted_access{}", idx),
            sorted_arr.data_type(),
            &sorted_iter,
        )
        .unwrap();

        config.push(ComparatorField {
            ptype: key_ptype,
            sort_options: *opts,
            lhs_nulls: key_null_access,
            lhs_accessor: key_access,
            rhs_nulls: sorted_null_access,
            rhs_accessor: sorted_access,
        });
    }

    fill_in_cmp(ctx, &llvm_mod, &config, BreakTies::DoNotBreakTies);

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

#[self_referencing]
pub struct LowerBoundKernel {
    context: Context,
    key_nullable: Vec<bool>,
    sorted_nullable: Vec<bool>,
    options: Vec<SortOptions>,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, u64, u64) -> i8>,
}

unsafe impl Sync for LowerBoundKernel {}
unsafe impl Send for LowerBoundKernel {}

impl Kernel for LowerBoundKernel {
    type Key = Vec<(DataType, bool, DataType, bool, SortOptions)>;

    type Input<'a> = (Vec<&'a dyn Array>, Vec<&'a dyn Array>);

    type Params = Vec<SortOptions>;

    type Output = UInt32Array;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, super::ArrowKernelError> {
        let (keys, sorted) = inp;

        if keys.is_empty() {
            return Err(ArrowKernelError::ArgumentMismatch(
                "lower_bound requires at least one key column".to_string(),
            ));
        }

        if keys.len() != sorted.len() {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "expected the same number of key and sorted columns, got {} and {}",
                keys.len(),
                sorted.len()
            )));
        }

        let key_len = keys[0].len();
        for arr in keys.iter().skip(1) {
            if arr.len() != key_len {
                return Err(ArrowKernelError::SizeMismatch);
            }
        }

        let sorted_len = sorted.get(0).map(|arr| arr.len()).unwrap_or(0);
        for arr in sorted.iter().skip(1) {
            if arr.len() != sorted_len {
                return Err(ArrowKernelError::SizeMismatch);
            }
        }

        let mut null_arrays = Vec::new();
        let mut iters = Vec::new();
        let mut ptrs = Vec::new();

        for (idx, (key_arr, sorted_arr)) in keys.iter().zip(sorted.iter()).enumerate() {
            iters.push(datum_to_iter(&*key_arr)?);
            let key_data_ptr = iters.last_mut().unwrap().get_mut_ptr();
            if let Some(nulls) = logical_nulls(*key_arr)? {
                if !self.borrow_key_nullable()[idx] {
                    return Err(ArrowKernelError::ArgumentMismatch(format!(
                        "kernel expected non-nullable key column at index {}",
                        idx
                    )));
                }
                let ba = BooleanArray::from(nulls.inner().clone());
                null_arrays.push(ba);
                iters.push(datum_to_iter(&null_arrays.last().unwrap())?);
                let null_ptr = iters.last_mut().unwrap().get_mut_ptr();
                ptrs.push(key_data_ptr);
                ptrs.push(null_ptr);
            } else {
                if self.borrow_key_nullable()[idx] {
                    return Err(ArrowKernelError::ArgumentMismatch(format!(
                        "kernel expected nullable key column at index {}",
                        idx
                    )));
                }
                ptrs.push(key_data_ptr);
            }

            iters.push(datum_to_iter(&*sorted_arr)?);
            let sorted_data_ptr = iters.last_mut().unwrap().get_mut_ptr();
            if let Some(nulls) = logical_nulls(*sorted_arr)? {
                if !self.borrow_sorted_nullable()[idx] {
                    return Err(ArrowKernelError::ArgumentMismatch(format!(
                        "kernel expected non-nullable sorted column at index {}",
                        idx
                    )));
                }
                let ba = BooleanArray::from(nulls.inner().clone());
                null_arrays.push(ba);
                iters.push(datum_to_iter(&null_arrays.last().unwrap())?);
                let null_ptr = iters.last_mut().unwrap().get_mut_ptr();
                ptrs.push(sorted_data_ptr);
                ptrs.push(null_ptr);
            } else {
                if self.borrow_sorted_nullable()[idx] {
                    return Err(ArrowKernelError::ArgumentMismatch(format!(
                        "kernel expected nullable sorted column at index {}",
                        idx
                    )));
                }
                ptrs.push(sorted_data_ptr);
            }
        }

        let mut kp = KernelParameters::new(ptrs);
        let kp_ptr = kp.get_mut_ptr();
        let func = self.borrow_func();

        let mut out = Vec::with_capacity(key_len);
        for key_idx in 0..key_len {
            let mut lo = 0u64;
            let mut hi = sorted_len as u64;
            while lo < hi {
                let mid = (lo + hi) / 2;
                let cmp = unsafe { func.call(kp_ptr, key_idx as u64, mid) } as i8;
                if cmp <= 0 {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            out.push(lo as u32);
        }

        Ok(UInt32Array::from(out))
    }

    fn compile(
        inp: &Self::Input<'_>,
        params: Self::Params,
    ) -> Result<Self, super::ArrowKernelError> {
        let (keys, sorted) = inp;

        if keys.is_empty() {
            return Err(ArrowKernelError::ArgumentMismatch(
                "lower_bound requires at least one column".to_string(),
            ));
        }

        if keys.len() != sorted.len() || keys.len() != params.len() {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "expected the same number of key, sorted, and option columns, got {} keys, {} sorted, {} options",
                keys.len(),
                sorted.len(),
                params.len()
            )));
        }

        let key_nullable = keys.iter().map(|arr| arr.is_nullable()).collect_vec();
        let sorted_nullable = sorted.iter().map(|arr| arr.is_nullable()).collect_vec();

        let ctx = Context::create();
        LowerBoundKernelTryBuilder {
            context: ctx,
            key_nullable,
            sorted_nullable,
            options: params.clone(),
            func_builder: |ctx| generate_lower_bound_cmp(ctx, keys, sorted, &params),
        }
        .try_build()
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        let (keys, sorted) = i;

        if keys.is_empty() {
            return Err(ArrowKernelError::ArgumentMismatch(
                "lower_bound requires at least one column".to_string(),
            ));
        }

        let mut res = Vec::with_capacity(keys.len());
        for ((key_arr, sorted_arr), opts) in keys.iter().zip(sorted.iter()).zip(p.iter()) {
            res.push((
                key_arr.data_type().clone(),
                key_arr.is_nullable(),
                sorted_arr.data_type().clone(),
                sorted_arr.is_nullable(),
                *opts,
            ));
        }

        Ok(res)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        compiled_kernels::sort::{SortKernel, SortOptions, TopKKernel},
        Kernel,
    };
    use arrow_array::{
        cast::AsArray, types::Int32Type, Array, ArrayRef, Float32Array, Int32Array, Int64Array,
        StringArray, UInt32Array,
    };
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

        let sorted = data.iter().filter_map(|x| x.clone()).sorted().collect_vec();
        let perm_values = our_res
            .iter()
            .filter(|idx| !arr.is_null(**idx as usize))
            .map(|idx| arr.as_primitive::<Int32Type>().value(*idx as usize))
            .collect_vec();
        assert_eq!(perm_values, sorted);

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
    fn test_sort_i32_nulls_first() {
        let data = vec![Some(1), None, Some(3), None, Some(2)];
        let arr = Arc::new(Int32Array::from(data.clone())) as ArrayRef;
        let input = vec![&arr as &dyn Array];

        let k = SortKernel::compile(
            &input,
            vec![SortOptions {
                descending: false,
                nulls_first: true,
            }],
        )
        .unwrap();

        let perm = k
            .call(input.clone())
            .unwrap()
            .into_iter()
            .map(|x| x.unwrap())
            .collect_vec();

        assert_eq!(perm, vec![1, 3, 0, 4, 2]);
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

    #[test]
    fn test_sort_f32() {
        let data = Float32Array::from(vec![32.0, 16.0, f32::NAN, f32::INFINITY]);

        let k =
            SortKernel::compile(&vec![&data as &dyn Array], vec![SortOptions::default()]).unwrap();
        let res = k.call(vec![&data]).unwrap();
        let res = res.iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(res, vec![1, 0, 3, 2]);
    }

    #[test]
    fn test_topk_i32_nonnull_fwd() {
        let data = vec![-10, 2, 7, 100, -274, 18, -22, 10000, -193, 18, 22, 12000];
        let arr = Int32Array::from(data.clone());
        let input = vec![&arr as &dyn Array];

        let k = TopKKernel::compile(&(input.clone(), 3), vec![SortOptions::default()]).unwrap();

        let our_res = k.call((input.clone(), 3)).unwrap();
        let our_res = our_res.values().to_vec();
        assert_eq!(our_res, vec![4, 8, 6]);
    }
}
