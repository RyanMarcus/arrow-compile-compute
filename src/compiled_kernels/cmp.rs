use arrow_array::cast::AsArray;
use arrow_array::{BooleanArray, Datum};
use arrow_schema::DataType;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::{Linkage, Module};

use inkwell::context::Context;
use inkwell::values::FunctionValue;
use inkwell::IntPredicate;

use crate::compiled_kernels::dsl2::{
    compile, DSLArgument, DSLBitwiseBinOp, DSLComparison, DSLContext, DSLFunction, DSLStmt,
    DSLType, RunnableDSLFunction,
};
use crate::compiled_writers::WriterSpec;
use crate::{
    declare_blocks, increment_pointer, intersect_and_copy_nulls, pointer_diff, Predicate,
    PrimitiveType,
};

use super::{ArrowKernelError, Kernel};

pub struct ComparisonKernel(RunnableDSLFunction);
unsafe impl Sync for ComparisonKernel {}
unsafe impl Send for ComparisonKernel {}

impl Kernel for ComparisonKernel {
    type Key = (DataType, bool, DataType, bool, Predicate);
    type Input<'a> = (&'a dyn Datum, &'a dyn Datum);
    type Params = Predicate;
    type Output = BooleanArray;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (a, b) = inp;

        let res = self
            .0
            .run(&[DSLArgument::Datum(a), DSLArgument::Datum(b)])?[0]
            .clone();

        intersect_and_copy_nulls(&[a, b], res).map(|x| x.as_boolean().clone())
    }

    fn compile(inp: &Self::Input<'_>, pred: Predicate) -> Result<Self, ArrowKernelError> {
        let (lhs, rhs) = *inp;
        let (lhs_arr, _lhs_scalar) = lhs.get();
        let (rhs_arr, _rhs_scalar) = rhs.get();

        let dom_type = PrimitiveType::dominant(
            PrimitiveType::for_arrow_type(lhs_arr.data_type()),
            PrimitiveType::for_arrow_type(rhs_arr.data_type()),
        )
        .ok_or_else(|| {
            ArrowKernelError::UnsupportedArguments(format!(
                "could not compare values of {:?} and {:?}",
                lhs_arr.data_type(),
                rhs_arr.data_type()
            ))
        })?;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("cmp");
        let arg_lhs = func.add_arg(&mut ctx, DSLType::array_like(lhs, "n"));
        let arg_rhs = func.add_arg(&mut ctx, DSLType::array_like(rhs, "n"));
        func.add_ret(WriterSpec::Boolean, "n");

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arg_lhs, arg_rhs], |loop_vars| {
                let lhs = loop_vars[0].expr().primitive_cast(dom_type)?;
                let rhs = loop_vars[1].expr().primitive_cast(dom_type)?;

                let cmp = lhs.cmp(&rhs, pred.into())?;
                DSLStmt::emit(0, cmp)
            })
            .unwrap(),
        );

        let func = compile(func, [DSLArgument::Datum(lhs), DSLArgument::Datum(rhs)])?;
        Ok(Self(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Predicate,
    ) -> Result<Self::Key, ArrowKernelError> {
        let (lhs, rhs) = *i;

        Ok((
            lhs.get().0.data_type().clone(),
            lhs.get().1,
            rhs.get().0.data_type().clone(),
            rhs.get().1,
            *p,
        ))
    }
}

pub struct BetweenKernel(RunnableDSLFunction);
unsafe impl Sync for BetweenKernel {}
unsafe impl Send for BetweenKernel {}

impl Kernel for BetweenKernel {
    type Key = (DataType, bool, DataType, bool, DataType, bool);
    type Input<'a>
        = (&'a dyn Datum, &'a dyn Datum, &'a dyn Datum)
    where
        Self: 'a;

    type Params = ();

    type Output = BooleanArray;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        self.0
            .run(&[
                DSLArgument::Datum(inp.0),
                DSLArgument::Datum(inp.1),
                DSLArgument::Datum(inp.2),
            ])
            .map(|x| x[0].as_boolean().clone())
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (inp, lb, ub) = inp;
        let inp_pt = PrimitiveType::for_arrow_type(inp.get().0.data_type());
        let lb_pt = PrimitiveType::for_arrow_type(lb.get().0.data_type());
        let ub_pt = PrimitiveType::for_arrow_type(ub.get().0.data_type());

        let dom_pt = PrimitiveType::dominant(inp_pt, lb_pt)
            .and_then(|pt| PrimitiveType::dominant(pt, ub_pt))
            .ok_or_else(|| {
                ArrowKernelError::UnsupportedArguments(format!(
                    "could not compare values of {}, {}, and {}",
                    inp_pt, lb_pt, ub_pt
                ))
            })?;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("between");
        let arg_inp = func.add_arg(&mut ctx, DSLType::array_like(*inp, "n"));
        let arg_lb = func.add_arg(&mut ctx, DSLType::array_like(*lb, "n"));
        let arg_ub = func.add_arg(&mut ctx, DSLType::array_like(*ub, "n"));
        func.add_ret(WriterSpec::Boolean, "n");

        func.add_body(
            DSLStmt::for_each(&mut ctx, &[arg_inp, arg_lb, arg_ub], |loop_vars| {
                let inp = loop_vars[0].expr().primitive_cast(dom_pt)?;
                let lb = loop_vars[1].expr().primitive_cast(dom_pt)?;
                let ub = loop_vars[2].expr().primitive_cast(dom_pt)?;

                let is_above_lb = inp.cmp(&lb, DSLComparison::Gte)?;
                let is_below_ub = inp.cmp(&ub, DSLComparison::Lt)?;
                let cmp = is_above_lb.bitwise(DSLBitwiseBinOp::And, is_below_ub)?;
                DSLStmt::emit(0, cmp)
            })
            .unwrap(),
        );
        let func = compile(
            func,
            [
                DSLArgument::Datum(*inp),
                DSLArgument::Datum(*lb),
                DSLArgument::Datum(*ub),
            ],
        )?;

        Ok(BetweenKernel(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((
            i.0.get().0.data_type().clone(),
            i.0.get().1,
            i.1.get().0.data_type().clone(),
            i.1.get().1,
            i.2.get().0.data_type().clone(),
            i.2.get().1,
        ))
    }
}

pub fn add_float_vec_to_int_vec<'a>(
    ctx: &'a Context,
    module: &Module<'a>,
    v_size: u32,
    ptype: PrimitiveType,
) -> FunctionValue<'a> {
    // apply the following algorithm to get a total float ordering
    // left ^= (((left >> 63) as u64) >> 1) as i64;
    // right ^= (((right >> 63) as u64) >> 1) as i64;
    // left.cmp(right)

    let inp_type = ptype.llvm_vec_type(ctx, v_size).unwrap();
    let int_type = PrimitiveType::int_with_width(ptype.width())
        .llvm_type(ctx)
        .into_int_type();
    let ret_type = int_type.vec_type(v_size);

    let func_type = ret_type.fn_type(&[inp_type.into()], false);
    let func = module.add_function("fvec_to_int", func_type, Some(Linkage::Private));
    let fvec = func.get_nth_param(0).unwrap().into_vector_value();

    declare_blocks!(ctx, func, entry);
    let builder = ctx.create_builder();
    builder.position_at_end(entry);

    let vminus1 = builder
        .build_insert_element(
            int_type.vec_type(v_size).const_zero(),
            int_type.const_int(ptype.width() as u64 * 8 - 1, false),
            int_type.const_zero(),
            "vminus1",
        )
        .unwrap();
    let vminus1 = builder
        .build_shuffle_vector(
            vminus1,
            int_type.vec_type(v_size).get_undef(),
            int_type.vec_type(v_size).const_zero(),
            "vminus1_bcast",
        )
        .unwrap();
    let v1 = builder
        .build_insert_element(
            int_type.vec_type(v_size).const_zero(),
            int_type.const_int(1, false),
            int_type.const_zero(),
            "v1",
        )
        .unwrap();
    let v1 = builder
        .build_shuffle_vector(
            v1,
            int_type.vec_type(v_size).get_undef(),
            int_type.vec_type(v_size).const_zero(),
            "v1_bcast",
        )
        .unwrap();

    let cleft = builder
        .build_bit_cast(fvec, int_type.vec_type(v_size), "cleft")
        .unwrap()
        .into_vector_value();
    let left = builder
        .build_right_shift(cleft, vminus1, true, "sleft")
        .unwrap();
    let left = builder.build_right_shift(left, v1, false, "sleft").unwrap();
    let res = builder.build_xor(cleft, left, "left").unwrap();
    builder.build_return(Some(&res)).unwrap();

    func
}

pub fn add_float_to_int<'a>(
    ctx: &'a Context,
    module: &Module<'a>,
    ptype: PrimitiveType,
) -> FunctionValue<'a> {
    // apply the following algorithm to get a total float ordering
    // left ^= (((left >> 63) as u64) >> 1) as i64;
    // right ^= (((right >> 63) as u64) >> 1) as i64;
    // left.cmp(right)

    let fn_name = format!("float_to_int_{}", ptype.width());
    if let Some(func) = module.get_function(&fn_name) {
        return func;
    }

    let inp_type = ptype.llvm_type(ctx);
    let int_type = PrimitiveType::int_with_width(ptype.width())
        .llvm_type(ctx)
        .into_int_type();

    let func_type = int_type.fn_type(&[inp_type.into()], false);
    let func = module.add_function(&fn_name, func_type, Some(Linkage::Private));
    let float = func.get_nth_param(0).unwrap().into_float_value();

    declare_blocks!(ctx, func, entry);
    let builder = ctx.create_builder();
    builder.position_at_end(entry);

    let vminus1 = int_type.const_int(ptype.width() as u64 * 8 - 1, false);
    let v1 = int_type.const_int(1, false);

    let cleft = builder
        .build_bit_cast(float, int_type, "cleft")
        .unwrap()
        .into_int_value();
    let left = builder
        .build_right_shift(cleft, vminus1, true, "sleft")
        .unwrap();
    let left = builder.build_right_shift(left, v1, false, "sleft").unwrap();
    let res = builder.build_xor(cleft, left, "left").unwrap();
    builder.build_return(Some(&res)).unwrap();

    func
}

/// Adds or returns the `memcmp` function to the module.
///
/// `memcmp` takes two `PrimitiveType::P64x2` values.
pub fn add_memcmp<'a>(ctx: &'a Context, module: &Module<'a>) -> FunctionValue<'a> {
    if let Some(func) = module.get_function("memcmp") {
        return func;
    }

    let i64_type = ctx.i64_type();
    let i8_type = ctx.i8_type();
    let str_type = PrimitiveType::P64x2.llvm_type(ctx);

    let fn_type = i64_type.fn_type(
        &[
            str_type.into(), // first string
            str_type.into(), // second string
        ],
        false,
    );

    let umin = Intrinsic::find("llvm.umin").unwrap();
    let umin_f = umin.get_declaration(module, &[i64_type.into()]).unwrap();

    let builder = ctx.create_builder();
    let function = module.add_function("memcmp", fn_type, Some(Linkage::Private));

    declare_blocks!(
        ctx,
        function,
        entry,
        for_cond,
        for_body,
        early_return,
        no_diff
    );

    builder.position_at_end(entry);
    let ptr1 = function.get_nth_param(0).unwrap().into_struct_value();
    let ptr2 = function.get_nth_param(1).unwrap().into_struct_value();

    let start_ptr1 = builder
        .build_extract_value(ptr1, 0, "start_ptr1")
        .unwrap()
        .into_pointer_value();
    let end_ptr1 = builder
        .build_extract_value(ptr1, 1, "end_ptr1")
        .unwrap()
        .into_pointer_value();
    let start_ptr2 = builder
        .build_extract_value(ptr2, 0, "start_ptr2")
        .unwrap()
        .into_pointer_value();
    let end_ptr2 = builder
        .build_extract_value(ptr2, 1, "end_ptr2")
        .unwrap()
        .into_pointer_value();

    let len1 = pointer_diff!(ctx, builder, start_ptr1, end_ptr1);
    let len2 = pointer_diff!(ctx, builder, start_ptr2, end_ptr2);

    let len = builder
        .build_call(umin_f, &[len1.into(), len2.into()], "len")
        .unwrap()
        .try_as_basic_value()
        .unwrap_basic()
        .into_int_value();

    let index_ptr = builder.build_alloca(i64_type, "index_ptr").unwrap();
    builder
        .build_store(index_ptr, i64_type.const_zero())
        .unwrap();

    builder.build_unconditional_branch(for_cond).unwrap();

    builder.position_at_end(for_cond);
    let index = builder
        .build_load(i64_type, index_ptr, "index")
        .unwrap()
        .into_int_value();
    let cmp = builder
        .build_int_compare(IntPredicate::ULT, index, len, "loop_cmp")
        .unwrap();
    builder
        .build_conditional_branch(cmp, for_body, no_diff)
        .unwrap();

    builder.position_at_end(for_body);
    let index = builder
        .build_load(i64_type, index_ptr, "index")
        .unwrap()
        .into_int_value();
    let elem1_ptr = increment_pointer!(ctx, builder, start_ptr1, 1, index);
    let elem2_ptr = increment_pointer!(ctx, builder, start_ptr2, 1, index);
    let elem1 = builder
        .build_load(i8_type, elem1_ptr, "elem1")
        .unwrap()
        .into_int_value();
    let elem2 = builder
        .build_load(i8_type, elem2_ptr, "elem2")
        .unwrap()
        .into_int_value();

    let elem1 = builder
        .build_int_z_extend_or_bit_cast(elem1, i64_type, "z_elem1")
        .unwrap();
    let elem2 = builder
        .build_int_z_extend_or_bit_cast(elem2, i64_type, "z_elem2")
        .unwrap();

    let diff = builder.build_int_sub(elem1, elem2, "sub").unwrap();

    let value_cmp = builder
        .build_int_compare(IntPredicate::EQ, diff, i64_type.const_zero(), "value_cmp")
        .unwrap();

    let inc_index = builder
        .build_int_add(index, i64_type.const_int(1, false), "inc_index")
        .unwrap();
    builder.build_store(index_ptr, inc_index).unwrap();

    builder
        .build_conditional_branch(value_cmp, for_cond, early_return)
        .unwrap();

    builder.position_at_end(early_return);
    builder.build_return(Some(&diff)).unwrap();

    builder.position_at_end(no_diff);
    let eq_len = builder
        .build_int_compare(IntPredicate::EQ, len1, len2, "is_eq")
        .unwrap();
    let arr1_longer = builder
        .build_int_compare(IntPredicate::UGT, len1, len2, "arr1_longer")
        .unwrap();

    let res = builder
        .build_select(
            eq_len,
            i64_type.const_zero(),
            builder
                .build_select(
                    arr1_longer,
                    i64_type.const_int(1, true),
                    i64_type.const_int(-1_i64 as u64, false),
                    "diff",
                )
                .unwrap()
                .into_int_value(),
            "same",
        )
        .unwrap()
        .into_int_value();
    builder.build_return(Some(&res)).unwrap();

    function
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use arrow_array::{
        BooleanArray, Float32Array, Int32Array, Int64Array, Scalar, StringArray, StringViewArray,
        UInt16Array, UInt32Array,
    };
    use arrow_schema::DataType;
    use itertools::Itertools;

    use crate::{
        compiled_kernels::{
            cmp::{BetweenKernel, ComparisonKernel},
            Kernel,
        },
        dictionary_data_type, Predicate,
    };

    #[test]
    fn test_num_num_cmp_nonnull() {
        let mut rng = fastrand::Rng::with_seed(42);
        let a = (0..100).map(|_| rng.u32(0..100)).collect_vec();
        let b = (0..100).map(|_| rng.u32(0..100)).collect_vec();
        let answer = a.iter().zip(b.iter()).map(|(a, b)| a < b).collect_vec();

        let a = UInt32Array::from(a);
        let b = UInt32Array::from(b);
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        assert!(k.0.vectorized);
        let r = k.call((&a, &b)).unwrap();
        assert_eq!(r, BooleanArray::from(answer));
    }

    #[test]
    fn test_num_num_cmp_nulls() {
        let a = UInt32Array::from(vec![Some(1), Some(2), None]);
        let b = UInt32Array::from(vec![Some(11), Some(0), Some(13)]);
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        assert!(k.0.vectorized);
        let r = k.call((&a, &b)).unwrap();
        assert_eq!(r, BooleanArray::from(vec![Some(true), Some(false), None]));
    }

    #[test]
    fn test_num_num_block_cmp() {
        let a = UInt32Array::from((0..100).collect_vec());
        let b = UInt32Array::from((1..101).collect_vec());
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        assert!(k.0.vectorized);
        let r = k.call((&a, &b)).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true; 100]))
    }

    #[test]
    fn test_i32_dict_ree_eq() {
        let n = 100_000usize;

        // Build a logical sequence of i32 values with small run lengths (1..=8)
        let mut values: Vec<i32> = Vec::with_capacity(n);
        let mut run_ends: Vec<i32> = Vec::new();
        let mut run_vals: Vec<i32> = Vec::new();

        let mut idx = 0usize;
        let mut cum_end = 0i32;
        while idx < n {
            let remaining = n - idx;
            let len = std::cmp::min(1 + (idx % 8), remaining); // run lengths in 1..=8
            let val = ((idx * 31) % 997) as i32; // deterministic varied values within a small domain

            values.extend(std::iter::repeat(val).take(len));
            cum_end += len as i32;
            run_ends.push(cum_end);
            run_vals.push(val);
            idx += len;
        }

        // Dictionary array from the full logical values
        let dict_logical = Int32Array::from(values.clone());
        let dict_arr = arrow_cast::cast(
            &dict_logical,
            &dictionary_data_type(DataType::Int16, DataType::Int32),
        )
        .unwrap();

        // Run-end encoded array from the runs
        let run_ends_arr = Int32Array::from(run_ends);
        let run_vals_arr = Int32Array::from(run_vals);
        let ree = arrow_array::RunArray::try_new(&run_ends_arr, &run_vals_arr).unwrap();
        let ree: arrow_array::ArrayRef = std::sync::Arc::new(ree);

        // Compare equality
        let k = ComparisonKernel::compile(&(&dict_arr, &ree), Predicate::Eq).unwrap();
        assert!(k.0.vectorized);
        let r = k.call((&dict_arr, &ree)).unwrap();

        assert_eq!(r, BooleanArray::from(vec![true; n]));
    }

    #[test]
    fn test_num_num_block_scalar_cmp() {
        let a = UInt32Array::from((0..100).collect_vec());
        let b = Scalar::new(UInt32Array::from(vec![5]));
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        assert!(k.0.vectorized);
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..100).map(|i| i < 5).collect::<Vec<bool>>();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_num_num_float_cmp() {
        let a = Float32Array::from((0..100).map(|i| i as f32).collect_vec());
        let b = Float32Array::from((1..101).map(|i| i as f32).collect_vec());
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        assert!(k.0.vectorized);
        let r = k.call((&a, &b)).unwrap();
        assert_eq!(r, BooleanArray::from(vec![true; 100]))
    }

    #[test]
    fn test_num_num_float_scalar_cmp() {
        let a = Float32Array::from((0..100).map(|i| i as f32).collect_vec());
        let b = Scalar::new(Float32Array::from(vec![5.0]));
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        assert!(k.0.vectorized);
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..100).map(|i| i < 5).collect::<Vec<bool>>();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_num_num_float_scalar_cmp_convert() {
        let a = Float32Array::from(vec![-93294.49]);
        let b = Float32Array::new_scalar(-205150180000.0);
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        assert!(k.0.vectorized);
        let r = k.call((&a, &b)).unwrap();

        let res = (-93294.49_f32).total_cmp(&-205150180000.0) == Ordering::Less;
        assert_eq!(r, BooleanArray::from(vec![res]));
    }

    #[test]
    fn test_int32_int64_cmp() {
        let a = Int32Array::from(vec![1, 2, 3, 4]);
        let b = Int64Array::from(vec![4, 3, 2, 1]);
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        assert!(k.0.vectorized);
        let r = k.call((&a, &b)).unwrap();

        let expected_result = vec![true, true, false, false];
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_string_string_cmp() {
        let a = StringArray::from(vec!["apple", "banana", "cherry"]);
        let b = StringArray::from(vec!["banana", "cherry", "apple"]);
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Lt).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = vec![true, true, false];
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_string_scalar_cmp() {
        let values = (0..100).map(|i| format!("value{}", i)).collect_vec();
        let a = StringArray::from(values);
        let b = Scalar::new(StringArray::from(vec!["value50"]));
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Eq).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..100).map(|i| i == 50).collect_vec();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_string_view_scalar_cmp() {
        let values = (0..100).map(|i| format!("value{}", i)).collect_vec();
        let a = StringViewArray::from(values);
        let b = Scalar::new(StringArray::from(vec!["value50"]));
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Eq).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..100).map(|i| i == 50).collect_vec();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_string_64_scalar_cmp() {
        let values = (0..64).map(|i| format!("value{}", i)).collect_vec();
        let a = StringArray::from(values);
        let b = Scalar::new(StringArray::from(vec!["value50"]));
        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Eq).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..64).map(|i| i == 50).collect_vec();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_dict_string_scalar_cmp() {
        let values = (0..100).map(|i| format!("value{}", i % 4)).collect_vec();
        let a = StringArray::from(values);
        let a =
            arrow_cast::cast(&a, &dictionary_data_type(DataType::Int8, DataType::Utf8)).unwrap();

        let b = Scalar::new(StringArray::from(vec!["value2"]));

        let k = ComparisonKernel::compile(&(&a, &b), Predicate::Eq).unwrap();
        let r = k.call((&a, &b)).unwrap();

        let expected_result = (0..100).map(|i| i % 4 == 2).collect_vec();
        assert_eq!(r, BooleanArray::from(expected_result));
    }

    #[test]
    fn test_u16_scalar_eq_on_sliced_input() {
        let base = UInt16Array::from((0..4096).map(|v| (v % 1024) as u16).collect_vec());
        let sliced = base.slice(3, 2048);
        let rhs = UInt16Array::new_scalar(127);

        let res = crate::cmp::eq(&sliced, &rhs).unwrap();
        assert_eq!(
            res.true_count(),
            (0..2048).filter(|i| ((i + 3) % 1024) == 127).count()
        );
    }

    #[test]
    fn test_between_kernel_scalar_bounds() {
        let a = Int32Array::from(vec![0, 1, 2, 3, 4, 5]);
        let lb = Scalar::new(Int32Array::from(vec![2]));
        let ub = Scalar::new(Int32Array::from(vec![5]));
        let k = BetweenKernel::compile(&(&a, &lb, &ub), ()).unwrap();
        assert!(k.0.vectorized);
        let r = k.call((&a, &lb, &ub)).unwrap();
        assert_eq!(
            r,
            BooleanArray::from(vec![false, false, true, true, true, false])
        );
    }
}
