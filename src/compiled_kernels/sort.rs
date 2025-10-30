use arrow_array::{Array, BooleanArray, Datum, UInt32Array};
use arrow_schema::DataType;
use inkwell::{
    context::Context,
    intrinsics::Intrinsic,
    module::{Linkage, Module},
    types::BasicTypeEnum,
    values::{FunctionValue, PointerValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};
use itertools::Itertools;
use ouroboros::self_referencing;

use crate::{
    compiled_iter::{datum_to_iter, generate_random_access, IteratorHolder},
    compiled_kernels::{
        cmp::{add_float_to_int, add_memcmp},
        dsl::KernelParameters,
        link_req_helpers, optimize_module, Kernel,
    },
    declare_blocks, logical_nulls, ArrowKernelError, PrimitiveType,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ComparatorSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PointerKind {
    Data,
    Null,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PointerLayoutEntry {
    field: usize,
    side: ComparatorSide,
    kind: PointerKind,
}

struct PointerIndexAllocator {
    entries: Vec<PointerLayoutEntry>,
}

impl PointerIndexAllocator {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn get_index(&mut self, field: usize, side: ComparatorSide, kind: PointerKind) -> usize {
        let entry = PointerLayoutEntry { field, side, kind };
        if let Some((idx, _)) = self
            .entries
            .iter()
            .enumerate()
            .find(|(_, existing)| **existing == entry)
        {
            idx
        } else {
            self.entries.push(entry);
            self.entries.len() - 1
        }
    }

    fn entries(&self) -> &[PointerLayoutEntry] {
        &self.entries
    }

    fn into_entries(self) -> Vec<PointerLayoutEntry> {
        self.entries
    }
}

#[derive(Clone, Copy)]
struct FieldAccessor<'a> {
    access: FunctionValue<'a>,
    null_access: Option<FunctionValue<'a>>,
    data_index: usize,
    null_index: Option<usize>,
}

struct ComparatorField<'a> {
    primitive: PrimitiveType,
    options: SortOptions,
    lhs: FieldAccessor<'a>,
    rhs: FieldAccessor<'a>,
}

#[derive(Clone)]
struct LowerBoundFieldMeta {
    array_nullable: bool,
    array_type: DataType,
}

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
        i8_type.fn_type(
            &[
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

    declare_blocks!(ctx, func, entry);
    b.position_at_end(entry);
    let lhs_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let rhs_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
    let idx1 = func.get_nth_param(2).unwrap().into_int_value();
    let idx2 = func.get_nth_param(3).unwrap().into_int_value();

    let val1 = b
        .build_call(access1, &[lhs_ptr.into(), idx1.into()], "val1")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left();
    let val2 = b
        .build_call(access2, &[rhs_ptr.into(), idx2.into()], "val2")
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
    fields: &[ComparatorField<'a>],
    pointer_layout: &[PointerLayoutEntry],
) {
    let mut cmp_funcs = Vec::new();
    for field in fields {
        let cmp = generate_single_cmp(
            ctx,
            llvm_mod,
            field.primitive,
            field.options.descending,
            field.lhs.access,
            field.rhs.access,
        );
        cmp_funcs.push(cmp);
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
    let params_ptr = cmp.get_nth_param(0).unwrap().into_pointer_value();
    let idx1 = cmp.get_nth_param(1).unwrap().into_int_value();
    let idx2 = cmp.get_nth_param(2).unwrap().into_int_value();

    let mut cached_ptrs: Vec<Option<PointerValue<'a>>> = vec![None; pointer_layout.len()];
    let mut get_ptr = |index: usize| {
        if let Some(ptr) = cached_ptrs[index] {
            ptr
        } else {
            let ptr = KernelParameters::llvm_get(ctx, &b, params_ptr, index);
            cached_ptrs[index] = Some(ptr);
            ptr
        }
    };

    if fields.is_empty() {
        let f = get_ucmp(ctx, llvm_mod, ctx.i64_type().into());
        let res = b
            .build_call(f, &[idx1.into(), idx2.into()], "idx_cmp_empty")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        b.build_return(Some(&res)).unwrap();
        return;
    }

    let all_entry_blocks = cmp_funcs
        .iter()
        .map(|_| ctx.append_basic_block(cmp, "entry"))
        .collect_vec();
    b.build_unconditional_branch(all_entry_blocks[0]).unwrap();

    for (idx, (field, cmp_f)) in fields.iter().zip(cmp_funcs.into_iter()).enumerate() {
        declare_blocks!(ctx, cmp, not_null, nonzero);
        b.position_at_end(all_entry_blocks[idx]);

        let lhs_data_ptr = get_ptr(field.lhs.data_index);
        let rhs_data_ptr = get_ptr(field.rhs.data_index);
        let lhs_null_ptr = field.lhs.null_index.map(|i| get_ptr(i));
        let rhs_null_ptr = field.rhs.null_index.map(|i| get_ptr(i));

        match (field.lhs.null_access, field.rhs.null_access) {
            (Some(lhs_null_fn), Some(rhs_null_fn)) => {
                declare_blocks!(ctx, cmp, lhs_null, rhs_null, both_null);
                let lhs_null_ptr = lhs_null_ptr.expect("missing lhs null pointer");
                let rhs_null_ptr = rhs_null_ptr.expect("missing rhs null pointer");

                let lhs_valid = b
                    .build_call(
                        lhs_null_fn,
                        &[lhs_null_ptr.into(), idx1.into()],
                        "lhs_valid",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let lhs_is_null = b.build_xor(lhs_valid, i8_one, "lhs_is_null").unwrap();

                let rhs_valid = b
                    .build_call(
                        rhs_null_fn,
                        &[rhs_null_ptr.into(), idx2.into()],
                        "rhs_valid",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let rhs_is_null = b.build_xor(rhs_valid, i8_one, "rhs_is_null").unwrap();

                let rhs_shifted = b
                    .build_left_shift(rhs_is_null, i8_type.const_int(1, false), "rhs_shifted")
                    .unwrap();
                let sum = b.build_or(lhs_is_null, rhs_shifted, "null_sum").unwrap();
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
                if field.options.nulls_first {
                    b.build_return(Some(&i8_neg_one)).unwrap();
                } else {
                    b.build_return(Some(&i8_one)).unwrap();
                }

                b.position_at_end(rhs_null);
                if field.options.nulls_first {
                    b.build_return(Some(&i8_one)).unwrap();
                } else {
                    b.build_return(Some(&i8_neg_one)).unwrap();
                }

                b.position_at_end(both_null);
                b.build_return(Some(&i8_type.const_zero())).unwrap();
            }
            (Some(lhs_null_fn), None) => {
                declare_blocks!(ctx, cmp, lhs_null);
                let lhs_null_ptr = lhs_null_ptr.expect("missing lhs null pointer");
                let lhs_valid = b
                    .build_call(
                        lhs_null_fn,
                        &[lhs_null_ptr.into(), idx1.into()],
                        "lhs_valid",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let lhs_is_null = b.build_xor(lhs_valid, i8_one, "lhs_is_null").unwrap();
                let lhs_null_cond = b
                    .build_int_compare(
                        IntPredicate::NE,
                        lhs_is_null,
                        i8_type.const_zero(),
                        "lhs_null_cond",
                    )
                    .unwrap();
                b.build_conditional_branch(lhs_null_cond, lhs_null, not_null)
                    .unwrap();

                b.position_at_end(lhs_null);
                if field.options.nulls_first {
                    b.build_return(Some(&i8_neg_one)).unwrap();
                } else {
                    b.build_return(Some(&i8_one)).unwrap();
                }
            }
            (None, Some(rhs_null_fn)) => {
                declare_blocks!(ctx, cmp, rhs_null);
                let rhs_null_ptr = rhs_null_ptr.expect("missing rhs null pointer");
                let rhs_valid = b
                    .build_call(
                        rhs_null_fn,
                        &[rhs_null_ptr.into(), idx2.into()],
                        "rhs_valid",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let rhs_is_null = b.build_xor(rhs_valid, i8_one, "rhs_is_null").unwrap();
                let rhs_null_cond = b
                    .build_int_compare(
                        IntPredicate::NE,
                        rhs_is_null,
                        i8_type.const_zero(),
                        "rhs_null_cond",
                    )
                    .unwrap();
                b.build_conditional_branch(rhs_null_cond, rhs_null, not_null)
                    .unwrap();

                b.position_at_end(rhs_null);
                if field.options.nulls_first {
                    b.build_return(Some(&i8_one)).unwrap();
                } else {
                    b.build_return(Some(&i8_neg_one)).unwrap();
                }
            }
            (None, None) => {
                b.build_unconditional_branch(not_null).unwrap();
            }
        }

        b.position_at_end(not_null);
        let res = b
            .build_call(
                cmp_f,
                &[
                    lhs_data_ptr.into(),
                    rhs_data_ptr.into(),
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
            if idx + 1 < fields.len() {
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
    pointer_layout: Vec<PointerLayoutEntry>,
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

        let pointer_layout = self.borrow_pointer_layout();
        let nullable = self.borrow_nullable();

        let mut _lhs_null_arrays: Vec<Option<BooleanArray>> = Vec::with_capacity(inp.len());
        let mut lhs_null_iters: Vec<Option<IteratorHolder>> = Vec::with_capacity(inp.len());
        let mut lhs_data_iters: Vec<Option<IteratorHolder>> = Vec::with_capacity(inp.len());

        for (arr, expected_nullable) in inp.iter().zip(nullable.iter()) {
            if let Some(nulls) = logical_nulls(*arr)? {
                assert!(*expected_nullable, "kernel expected nullable input");
                let ba = BooleanArray::from(nulls.inner().clone());
                let ba_datum: &dyn Datum = &ba;
                let ih = datum_to_iter(ba_datum)?;
                _lhs_null_arrays.push(Some(ba));
                lhs_null_iters.push(Some(ih));
            } else {
                assert!(!expected_nullable, "kernel expected non-nullable input");
                _lhs_null_arrays.push(None);
                lhs_null_iters.push(None);
            }

            let arr_datum = &*arr as &dyn Datum;
            let data_iter = datum_to_iter(arr_datum)?;
            lhs_data_iters.push(Some(data_iter));
        }

        let mut holders: Vec<IteratorHolder> = Vec::with_capacity(pointer_layout.len());
        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(pointer_layout.len());

        for entry in pointer_layout.iter() {
            match (entry.side, entry.kind) {
                (ComparatorSide::Left, PointerKind::Data) => {
                    let holder = lhs_data_iters[entry.field]
                        .take()
                        .expect("missing lhs data iterator");
                    let mut holder = holder;
                    let ptr = holder.get_mut_ptr();
                    ptrs.push(ptr);
                    holders.push(holder);
                }
                (ComparatorSide::Left, PointerKind::Null) => {
                    let holder = lhs_null_iters[entry.field]
                        .take()
                        .expect("missing lhs null iterator");
                    let mut holder = holder;
                    let ptr = holder.get_mut_ptr();
                    ptrs.push(ptr);
                    holders.push(holder);
                }
                (ComparatorSide::Right, _) => {
                    panic!("sort comparator unexpectedly referenced RHS iterator");
                }
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
        let pointer_layout_cell = std::cell::RefCell::new(Vec::new());
        let kernel = SortKernelTryBuilder {
            context: ctx,
            pointer_layout: Vec::new(),
            nullable: inp.iter().map(|x| x.is_nullable()).collect_vec(),
            func_builder: |ctx| {
                let (func, layout) = generate_sort(ctx, inp, &params)?;
                *pointer_layout_cell.borrow_mut() = layout;
                Ok::<_, ArrowKernelError>(func)
            },
        }
        .try_build()?;
        let mut kernel = kernel;
        kernel.with_pointer_layout_mut(|pl| *pl = pointer_layout_cell.into_inner());
        Ok(kernel)
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
) -> Result<
    (
        JitFunction<'a, unsafe extern "C" fn(*mut c_void, u64, u64) -> i8>,
        Vec<PointerLayoutEntry>,
    ),
    ArrowKernelError,
> {
    let llvm_mod = ctx.create_module("test");

    let mut allocator = PointerIndexAllocator::new();
    let mut fields = Vec::new();

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

        let arr_datum = &*arr as &dyn Datum;
        let ih = datum_to_iter(arr_datum)?;
        let access = generate_random_access(
            ctx,
            &llvm_mod,
            &format!("access{}", idx),
            arr.data_type(),
            &ih,
        )
        .unwrap();

        let lhs_data_index = allocator.get_index(idx, ComparatorSide::Left, PointerKind::Data);
        let lhs_null_index = if null_access.is_some() {
            Some(allocator.get_index(idx, ComparatorSide::Left, PointerKind::Null))
        } else {
            None
        };

        let lhs_accessor = FieldAccessor {
            access,
            null_access,
            data_index: lhs_data_index,
            null_index: lhs_null_index,
        };
        let rhs_accessor = FieldAccessor {
            access,
            null_access,
            data_index: lhs_data_index,
            null_index: lhs_null_index,
        };

        fields.push(ComparatorField {
            primitive: ptype,
            options: *opts,
            lhs: lhs_accessor,
            rhs: rhs_accessor,
        });
    }

    fill_in_cmp(ctx, &llvm_mod, &fields, allocator.entries());
    let pointer_layout = allocator.into_entries();

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

    Ok((cmp_func, pointer_layout))
}

#[self_referencing]
pub struct LowerBoundKernel {
    context: Context,
    pointer_layout: Vec<PointerLayoutEntry>,
    field_meta: Vec<LowerBoundFieldMeta>,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, u64, u64) -> i8>,
}

unsafe impl Sync for LowerBoundKernel {}
unsafe impl Send for LowerBoundKernel {}

impl Kernel for LowerBoundKernel {
    type Key = Vec<(DataType, bool, SortOptions)>;
    type Input<'a> = (Vec<&'a dyn Array>, Vec<&'a dyn Datum>);
    type Params = Vec<SortOptions>;
    type Output = u64;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (arrays, keys) = inp;
        if arrays.is_empty() {
            return Ok(0);
        }

        let len = arrays[0].len();
        if !arrays.iter().all(|arr| arr.len() == len) {
            return Err(ArrowKernelError::SizeMismatch);
        }
        if arrays.len() != keys.len() {
            return Err(ArrowKernelError::ArgumentMismatch(
                "lower_bound expects the same number of arrays and keys".to_string(),
            ));
        }

        let pointer_layout = self.borrow_pointer_layout();
        let field_meta = self.borrow_field_meta();

        let mut _lhs_null_arrays: Vec<Option<BooleanArray>> = Vec::with_capacity(arrays.len());
        let mut lhs_null_iters: Vec<Option<IteratorHolder>> = Vec::with_capacity(arrays.len());
        let mut lhs_data_iters: Vec<Option<IteratorHolder>> = Vec::with_capacity(arrays.len());

        for (arr, meta) in arrays.iter().zip(field_meta.iter()) {
            if arr.is_nullable() != meta.array_nullable {
                return Err(ArrowKernelError::ArgumentMismatch(format!(
                    "array nullability changed (expected {}, found {})",
                    meta.array_nullable,
                    arr.is_nullable()
                )));
            }
            if arr.data_type() != &meta.array_type {
                return Err(ArrowKernelError::ArgumentMismatch(format!(
                    "array type mismatch: expected {:?}, found {:?}",
                    meta.array_type,
                    arr.data_type()
                )));
            }

            if let Some(nulls) = logical_nulls(*arr)? {
                let ba = BooleanArray::from(nulls.inner().clone());
                let ba_datum: &dyn Datum = &ba;
                let ih = datum_to_iter(ba_datum)?;
                _lhs_null_arrays.push(Some(ba));
                lhs_null_iters.push(Some(ih));
            } else {
                _lhs_null_arrays.push(None);
                lhs_null_iters.push(None);
            }

            let arr_datum = &*arr as &dyn Datum;
            let data_iter = datum_to_iter(arr_datum)?;
            lhs_data_iters.push(Some(data_iter));
        }

        let mut rhs_data_iters: Vec<Option<IteratorHolder>> = Vec::with_capacity(keys.len());
        for (key, meta) in keys.iter().zip(field_meta.iter()) {
            let (key_arr, is_scalar) = key.get();
            if !is_scalar {
                return Err(ArrowKernelError::ArgumentMismatch(
                    "lower_bound requires scalar keys".to_string(),
                ));
            }
            if key_arr.data_type() != &meta.array_type {
                return Err(ArrowKernelError::ArgumentMismatch(format!(
                    "key type {:?} did not match array type {:?}",
                    key_arr.data_type(),
                    meta.array_type
                )));
            }
            if key_arr.is_null(0) {
                return Err(ArrowKernelError::UnsupportedArguments(
                    "lower_bound does not support null scalar keys".to_string(),
                ));
            }
            rhs_data_iters.push(Some(datum_to_iter(*key)?));
        }

        let mut holders: Vec<IteratorHolder> = Vec::with_capacity(pointer_layout.len());
        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(pointer_layout.len());

        for entry in pointer_layout.iter() {
            match (entry.side, entry.kind) {
                (ComparatorSide::Left, PointerKind::Data) => {
                    let holder = lhs_data_iters[entry.field]
                        .take()
                        .expect("missing lhs data iterator");
                    let mut holder = holder;
                    let ptr = holder.get_mut_ptr();
                    ptrs.push(ptr);
                    holders.push(holder);
                }
                (ComparatorSide::Left, PointerKind::Null) => {
                    let holder = lhs_null_iters[entry.field]
                        .take()
                        .expect("missing lhs null iterator");
                    let mut holder = holder;
                    let ptr = holder.get_mut_ptr();
                    ptrs.push(ptr);
                    holders.push(holder);
                }
                (ComparatorSide::Right, PointerKind::Data) => {
                    let holder = rhs_data_iters[entry.field]
                        .take()
                        .expect("missing rhs data iterator");
                    let mut holder = holder;
                    let ptr = holder.get_mut_ptr();
                    ptrs.push(ptr);
                    holders.push(holder);
                }
                (ComparatorSide::Right, PointerKind::Null) => {
                    panic!("lower_bound comparator unexpectedly referenced RHS null iterator");
                }
            }
        }

        let mut kp = KernelParameters::new(ptrs);
        let kp_ptr = kp.get_mut_ptr();
        let func = self.borrow_func();

        let mut lo = 0u64;
        let mut hi = len as u64;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let res = unsafe { func.call(kp_ptr, mid, 0) };
            debug_assert!(res == -1 || res == 0 || res == 1);
            if res == -1 {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        Ok(lo)
    }

    fn compile(inp: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arrays, keys) = inp;
        if arrays.len() != params.len() {
            return Err(ArrowKernelError::ArgumentMismatch(
                "lower_bound requires one SortOptions per array".to_string(),
            ));
        }
        if keys.len() != params.len() {
            return Err(ArrowKernelError::ArgumentMismatch(
                "lower_bound requires the same number of keys and arrays".to_string(),
            ));
        }

        for (arr, key) in arrays.iter().zip(keys.iter()) {
            let (key_arr, is_scalar) = key.get();
            if !is_scalar {
                return Err(ArrowKernelError::ArgumentMismatch(
                    "lower_bound requires scalar keys".to_string(),
                ));
            }
            if key_arr.data_type() != arr.data_type() {
                return Err(ArrowKernelError::ArgumentMismatch(format!(
                    "key type {:?} did not match array type {:?}",
                    key_arr.data_type(),
                    arr.data_type()
                )));
            }
            if key_arr.is_null(0) {
                return Err(ArrowKernelError::UnsupportedArguments(
                    "lower_bound does not support null scalar keys".to_string(),
                ));
            }
        }

        let ctx = Context::create();
        let field_meta = arrays
            .iter()
            .map(|arr| LowerBoundFieldMeta {
                array_nullable: arr.is_nullable(),
                array_type: arr.data_type().clone(),
            })
            .collect_vec();

        let pointer_layout_cell = std::cell::RefCell::new(Vec::new());
        let kernel = LowerBoundKernelTryBuilder {
            context: ctx,
            pointer_layout: Vec::new(),
            field_meta,
            func_builder: |ctx| {
                let (func, layout) = generate_lower_bound(
                    ctx,
                    arrays.as_slice(),
                    keys.as_slice(),
                    params.as_slice(),
                )?;
                *pointer_layout_cell.borrow_mut() = layout;
                Ok::<_, ArrowKernelError>(func)
            },
        }
        .try_build()?;
        let mut kernel = kernel;
        kernel.with_pointer_layout_mut(|pl| *pl = pointer_layout_cell.into_inner());
        Ok(kernel)
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        let (arrays, keys) = i;
        if arrays.len() != p.len() || keys.len() != p.len() {
            return Err(ArrowKernelError::ArgumentMismatch(
                "lower_bound requires matching numbers of arrays, keys, and options".to_string(),
            ));
        }

        let mut result = Vec::new();
        for ((arr, key), opts) in arrays.iter().zip(keys.iter()).zip(p.iter()) {
            let (key_arr, is_scalar) = key.get();
            if !is_scalar {
                return Err(ArrowKernelError::ArgumentMismatch(
                    "lower_bound requires scalar keys".to_string(),
                ));
            }
            if key_arr.data_type() != arr.data_type() {
                return Err(ArrowKernelError::ArgumentMismatch(format!(
                    "key type {:?} did not match array type {:?}",
                    key_arr.data_type(),
                    arr.data_type()
                )));
            }
            result.push((arr.data_type().clone(), arr.is_nullable(), *opts));
        }
        Ok(result)
    }
}

fn generate_lower_bound<'a>(
    ctx: &'a Context,
    arrays: &[&dyn Array],
    keys: &[&dyn Datum],
    params: &[SortOptions],
) -> Result<
    (
        JitFunction<'a, unsafe extern "C" fn(*mut c_void, u64, u64) -> i8>,
        Vec<PointerLayoutEntry>,
    ),
    ArrowKernelError,
> {
    let llvm_mod = ctx.create_module("lower_bound");

    let mut allocator = PointerIndexAllocator::new();
    let mut fields = Vec::new();

    for (idx, ((arr, key), opts)) in arrays
        .iter()
        .zip(keys.iter())
        .zip(params.iter())
        .enumerate()
    {
        let ptype = PrimitiveType::for_arrow_type(arr.data_type());
        let lhs_null_access = if let Some(nulls) = logical_nulls(*arr)? {
            let ba = BooleanArray::from(nulls.inner().clone());
            let ba_datum: &dyn Datum = &ba;
            let ih = datum_to_iter(ba_datum)?;
            Some(
                generate_random_access(
                    ctx,
                    &llvm_mod,
                    &format!("lhs_null{}", idx),
                    &DataType::Boolean,
                    &ih,
                )
                .unwrap(),
            )
        } else {
            None
        };

        let lhs_datum = &*arr as &dyn Datum;
        let lhs_iter = datum_to_iter(lhs_datum)?;
        let lhs_access = generate_random_access(
            ctx,
            &llvm_mod,
            &format!("lhs_access{}", idx),
            arr.data_type(),
            &lhs_iter,
        )
        .unwrap();

        let (key_arr, is_scalar) = key.get();
        if !is_scalar {
            return Err(ArrowKernelError::ArgumentMismatch(
                "lower_bound requires scalar keys".to_string(),
            ));
        }
        if key_arr.is_null(0) {
            return Err(ArrowKernelError::UnsupportedArguments(
                "lower_bound does not support null scalar keys".to_string(),
            ));
        }

        let key_iter = datum_to_iter(*key)?;
        let rhs_access = generate_random_access(
            ctx,
            &llvm_mod,
            &format!("rhs_access{}", idx),
            key_arr.data_type(),
            &key_iter,
        )
        .ok_or_else(|| ArrowKernelError::UnsupportedScalar(key_arr.data_type().clone()))?;

        let lhs_data_index = allocator.get_index(idx, ComparatorSide::Left, PointerKind::Data);
        let lhs_null_index = if lhs_null_access.is_some() {
            Some(allocator.get_index(idx, ComparatorSide::Left, PointerKind::Null))
        } else {
            None
        };
        let rhs_data_index = allocator.get_index(idx, ComparatorSide::Right, PointerKind::Data);

        let lhs_accessor = FieldAccessor {
            access: lhs_access,
            null_access: lhs_null_access,
            data_index: lhs_data_index,
            null_index: lhs_null_index,
        };
        let rhs_accessor = FieldAccessor {
            access: rhs_access,
            null_access: None,
            data_index: rhs_data_index,
            null_index: None,
        };

        fields.push(ComparatorField {
            primitive: ptype,
            options: *opts,
            lhs: lhs_accessor,
            rhs: rhs_accessor,
        });
    }

    fill_in_cmp(ctx, &llvm_mod, &fields, allocator.entries());
    let pointer_layout = allocator.into_entries();

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

    Ok((cmp_func, pointer_layout))
}

#[cfg(test)]
mod test {
    use crate::{
        compiled_kernels::sort::{LowerBoundKernel, SortKernel, SortOptions},
        Kernel,
    };
    use arrow_array::{
        cast::AsArray, types::Int32Type, Array, ArrayRef, Datum, Float32Array, Int32Array,
        Int64Array, Scalar, StringArray, UInt32Array,
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

        let k = SortKernel::compile(
            &input,
            vec![SortOptions {
                descending: false,
                nulls_first: true,
            }],
        )
        .unwrap();
        let our_res = k
            .call(input.clone())
            .unwrap()
            .into_iter()
            .map(|x| x.unwrap())
            .collect_vec();

        let num_nulls = arr.null_count();
        for i in 0..num_nulls {
            assert!(
                arr.is_null(our_res[i] as usize),
                "index {} (null index {}) was not null",
                our_res[i],
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
    fn test_lower_bound_basic() {
        let array = Int32Array::from(vec![1, 3, 5, 7, 9]);
        let arrays = vec![&array as &dyn Array];
        let options = vec![SortOptions::default()];
        let init_key = Scalar::new(Int32Array::from(vec![0]));
        let kernel = LowerBoundKernel::compile(
            &(arrays.clone(), vec![&init_key as &dyn Datum]),
            options.clone(),
        )
        .unwrap();

        let cases = vec![
            (-1, 0u64),
            (1, 0u64),
            (4, 2u64),
            (5, 2u64),
            (6, 3u64),
            (10, 5u64),
        ];
        for (value, expected) in cases {
            let scalar = Scalar::new(Int32Array::from(vec![value]));
            let result = kernel
                .call((arrays.clone(), vec![&scalar as &dyn Datum]))
                .unwrap();
            assert_eq!(result, expected, "value {value}");
        }
    }

    #[test]
    fn test_lower_bound_with_nulls() {
        let array = Int32Array::from(vec![Some(2), Some(4), None, None]);
        let arrays = vec![&array as &dyn Array];
        let options = vec![SortOptions::default()];
        let init_key = Scalar::new(Int32Array::from(vec![0]));
        let kernel = LowerBoundKernel::compile(
            &(arrays.clone(), vec![&init_key as &dyn Datum]),
            options.clone(),
        )
        .unwrap();

        let scalar = Scalar::new(Int32Array::from(vec![1]));
        let res = kernel
            .call((arrays.clone(), vec![&scalar as &dyn Datum]))
            .unwrap();
        assert_eq!(res, 0);

        let scalar = Scalar::new(Int32Array::from(vec![3]));
        let res = kernel
            .call((arrays.clone(), vec![&scalar as &dyn Datum]))
            .unwrap();
        assert_eq!(res, 1);

        let scalar = Scalar::new(Int32Array::from(vec![6]));
        let res = kernel
            .call((arrays.clone(), vec![&scalar as &dyn Datum]))
            .unwrap();
        assert_eq!(res, 2);
    }
}
