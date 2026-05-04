use std::sync::Arc;

use arrow_array::{cast::AsArray, Array, ArrayRef, Datum, Float32Array, Scalar};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        dsl2::{
            compile, DSLArgument, DSLArithBinOp, DSLContext, DSLFunction, DSLReductionType,
            DSLStmt, DSLType, DSLValue, RunnableDSLFunction, WriterSpec,
        },
        null_utils::replace_nulls,
    },
    logical_nulls, ArrayDatum, ArrowKernelError, Kernel, PrimitiveType,
};

pub struct DotKernel(RunnableDSLFunction);
unsafe impl Sync for DotKernel {}
unsafe impl Send for DotKernel {}

impl Kernel for DotKernel {
    type Key = (DataType, DataType);

    type Input<'a>
        = (&'a dyn Datum, &'a dyn Array)
    where
        Self: 'a;

    type Params = ();

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (q, vecs) = inp;
        let mut res = self
            .0
            .run(&[DSLArgument::datum(&vecs), DSLArgument::Datum(q)])?[0]
            .clone();
        if let Some(nulls) = logical_nulls(vecs)? {
            res = replace_nulls(res, Some(nulls));
        }
        Ok(res)
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (q, vecs) = inp;
        let (q_val, q_is_scalar) = q.get();

        let vecs = vecs.as_fixed_size_list_opt().ok_or_else(|| {
            ArrowKernelError::UnsupportedArguments(format!(
                "Vectors must be fixed size list, got {}",
                vecs.data_type()
            ))
        })?;

        let q_val = q_val.as_fixed_size_list_opt().ok_or_else(|| {
            ArrowKernelError::UnsupportedArguments(format!(
                "Query must be fixed size list, got {}",
                q_val.data_type()
            ))
        })?;

        if !q_is_scalar {
            return Err(ArrowKernelError::UnsupportedArguments(
                "query vector must be scalar".to_string(),
            ));
        }

        if q_val.is_null(0) {
            return Err(ArrowKernelError::UnsupportedArguments(
                "query vector must not be null".to_string(),
            ));
        }

        if vecs.values().is_nullable() && vecs.values().null_count() > 0 {
            return Err(ArrowKernelError::UnsupportedArguments(
                "vector values must not contain null".to_string(),
            ));
        }

        if vecs.values().data_type() != q_val.values().data_type() {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "vector and query must have the same data type (got {} and {})",
                vecs.values().data_type(),
                q_val.values().data_type()
            )));
        }

        if vecs.value_length() != q_val.value_length() {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "vector and query must have the same length (got {} and {})",
                vecs.value_length(),
                q_val.value_length()
            )));
        }

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("dot_vec");
        let vecs_arg = func.add_arg(&mut ctx, DSLType::array_like(vecs, "n"));
        let q_arg = func.add_arg(&mut ctx, DSLType::array_like(*q, "n"));
        func.add_ret(
            WriterSpec::Primitive(PrimitiveType::for_arrow_type(vecs.values().data_type())),
            "n",
        );
        func.add_body(DSLStmt::for_each(
            &mut ctx,
            &[vecs_arg, q_arg],
            |loop_vars| {
                let lhs = loop_vars[0].expr();
                let rhs = loop_vars[1].expr();
                let dot = lhs.arith(DSLArithBinOp::Mul, rhs)?.vec_sum()?;
                DSLStmt::emit(0, dot)
            },
        )?);

        Ok(DotKernel(compile(
            func,
            [DSLArgument::datum(&vecs), DSLArgument::Datum(*q)],
        )?))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((i.0.get().0.data_type().clone(), i.1.data_type().clone()))
    }
}

pub struct NormVecKernel(RunnableDSLFunction);
unsafe impl Sync for NormVecKernel {}
unsafe impl Send for NormVecKernel {}

impl Kernel for NormVecKernel {
    type Key = DataType;

    type Input<'a>
        = &'a dyn Datum
    where
        Self: 'a;

    type Params = ();

    type Output = Arc<dyn Datum>;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (data, is_scalar) = inp.get();
        let mut res = self.0.run(&[DSLArgument::Datum(inp)])?[0].clone();
        if let Some(nulls) = logical_nulls(data)? {
            res = replace_nulls(res, Some(nulls));
        }

        if is_scalar {
            return Ok(Arc::new(Scalar::new(res)));
        }

        Ok(Arc::new(ArrayDatum(res)))
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, _is_scalar) = inp.get();

        let arr = arr.as_fixed_size_list_opt().ok_or_else(|| {
            ArrowKernelError::UnsupportedArguments(format!(
                "Vectors must be fixed size list, got {}",
                arr.data_type()
            ))
        })?;

        if arr.values().is_nullable() && arr.values().null_count() > 0 {
            return Err(ArrowKernelError::UnsupportedArguments(
                "vector values must not contain null".to_string(),
            ));
        }

        let value_length = arr.value_length() as usize;
        let item_type = PrimitiveType::for_arrow_type(arr.values().data_type());
        if !item_type.is_float() {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "vector normalization requires float values, got {}",
                arr.values().data_type()
            )));
        }

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("norm_vec");
        let vecs_arg = func.add_arg(&mut ctx, DSLType::array_like(*inp, "n"));
        func.add_ret(WriterSpec::for_base_type_of_datum(*inp), "n");
        func.add_body(DSLStmt::for_each(&mut ctx, &[vecs_arg], |loop_vars| {
            let vec = loop_vars[0].expr();
            let squared = vec.arith(DSLArithBinOp::Mul, vec.clone())?;
            let norm = squared.vec_sum()?.sqrt()?;
            let norm_vec = norm.splat(value_length)?;
            let normalized = vec.arith(DSLArithBinOp::Div, norm_vec)?;
            DSLStmt::emit(0, normalized)
        })?);

        Ok(NormVecKernel(compile(func, [DSLArgument::Datum(*inp)])?))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok(i.get().0.data_type().clone())
    }
}

pub struct NearestNeighborKernel(RunnableDSLFunction);
unsafe impl Sync for NearestNeighborKernel {}
unsafe impl Send for NearestNeighborKernel {}

impl Kernel for NearestNeighborKernel {
    type Key = (DataType, DataType, DataType);

    type Input<'a>
        = (&'a dyn Datum, &'a dyn Array, &'a Float32Array)
    where
        Self: 'a;

    type Params = ();

    type Output = Option<u64>;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        validate_nearest_neighbor_input(inp)?;

        match self.0.run(&[
            DSLArgument::Datum(inp.0),
            DSLArgument::datum(&inp.1),
            DSLArgument::Datum(inp.2),
        ]) {
            Ok(result) => Ok(result[0]
                .as_primitive::<arrow_array::types::UInt64Type>()
                .iter()
                .next()
                .unwrap()),
            Err(ArrowKernelError::RuntimeEmptyReduction) => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        validate_nearest_neighbor_input(*inp)?;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("nearest_neighbor");
        let query_arg = func.add_arg(&mut ctx, DSLType::array_like(inp.0, "n"));
        let values_arg = func.add_arg(&mut ctx, DSLType::array_like(&inp.1, "n"));
        let norms_arg = func.add_arg(&mut ctx, DSLType::array_like(inp.2, "n"));

        let reduced = DSLStmt::reduce(
            &mut ctx,
            DSLReductionType::ArgMin,
            &[values_arg, norms_arg, query_arg],
            |loop_vars| {
                let value = loop_vars[0].expr();
                let squared_norm = loop_vars[1].expr();
                let query = loop_vars[2].expr();
                let dot = query.arith(DSLArithBinOp::Mul, value)?.vec_sum()?;
                let scaled_dot = DSLValue::f32(-2.0)
                    .as_primitive_expr()?
                    .arith(DSLArithBinOp::Mul, dot)?;
                squared_norm.arith(DSLArithBinOp::Add, scaled_dot)
            },
        )?;

        func.add_body(reduced.stmt);
        func.add_body(DSLStmt::emit(0, reduced.value.expr())?);
        func.add_ret(WriterSpec::Primitive(PrimitiveType::U64), "<= n");

        Ok(Self(compile(
            func,
            [
                DSLArgument::Datum(inp.0),
                DSLArgument::datum(&inp.1),
                DSLArgument::Datum(inp.2),
            ],
        )?))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        validate_nearest_neighbor_input(*i)?;
        Ok((
            i.0.get().0.data_type().clone(),
            i.1.data_type().clone(),
            i.2.data_type().clone(),
        ))
    }
}

fn validate_nearest_neighbor_input(
    input: (&dyn Datum, &dyn Array, &Float32Array),
) -> Result<(), ArrowKernelError> {
    let (query, values, norms) = input;
    let (query_arr, query_is_scalar) = query.get();
    if !query_is_scalar {
        return Err(ArrowKernelError::UnsupportedArguments(
            "nearest neighbor query must be scalar".to_string(),
        ));
    }

    let query = query_arr.as_fixed_size_list_opt().ok_or_else(|| {
        ArrowKernelError::UnsupportedArguments(format!(
            "nearest neighbor query must be fixed size list, got {}",
            query_arr.data_type()
        ))
    })?;
    let values = values.as_fixed_size_list_opt().ok_or_else(|| {
        ArrowKernelError::UnsupportedArguments(format!(
            "nearest neighbor values must be fixed size list, got {}",
            values.data_type()
        ))
    })?;

    if query.value_type() != DataType::Float32 || values.value_type() != DataType::Float32 {
        return Err(ArrowKernelError::UnsupportedArguments(format!(
            "nearest neighbor requires Float32 vectors, got query {} and values {}",
            query.value_type(),
            values.value_type()
        )));
    }
    if query.value_length() != values.value_length() {
        return Err(ArrowKernelError::UnsupportedArguments(format!(
            "nearest neighbor query and values must have the same dimension, got {} and {}",
            query.value_length(),
            values.value_length()
        )));
    }
    if norms.len() != values.len() {
        return Err(ArrowKernelError::ArgumentMismatch(format!(
            "nearest neighbor norm length must match values length, got {} and {}",
            norms.len(),
            values.len()
        )));
    }
    if query.is_null(0) {
        return Err(ArrowKernelError::UnsupportedArguments(
            "nearest neighbor query must not be null".to_string(),
        ));
    }
    if logical_nulls(values)?.is_some() {
        return Err(ArrowKernelError::UnsupportedArguments(
            "nearest neighbor values must not contain null vectors".to_string(),
        ));
    }
    if query.values().is_nullable() && query.values().null_count() > 0 {
        return Err(ArrowKernelError::UnsupportedArguments(
            "nearest neighbor query vector values must not contain nulls".to_string(),
        ));
    }
    if values.values().is_nullable() && values.values().null_count() > 0 {
        return Err(ArrowKernelError::UnsupportedArguments(
            "nearest neighbor value vector values must not contain nulls".to_string(),
        ));
    }
    if norms.null_count() > 0 {
        return Err(ArrowKernelError::UnsupportedArguments(
            "nearest neighbor norms must not contain nulls".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use super::DotKernel;
    use crate::compiled_kernels::vec::NormVecKernel;
    use crate::Kernel;
    use arrow_array::{builder::FixedSizeListBuilder, cast::AsArray};
    use arrow_array::{builder::Float16Builder, types::Float16Type, Scalar};
    use arrow_array::{Array, Datum};
    use half::f16;

    #[test]
    fn test_dot_kernel() {
        // Build a FixedSizeList<f16, 3> array of vectors
        let mut list_builder = FixedSizeListBuilder::new(Float16Builder::new(), 3);

        // v0 = [1, 2, 3]
        list_builder.values().append_value(f16::from_f32(1.0));
        list_builder.values().append_value(f16::from_f32(2.0));
        list_builder.values().append_value(f16::from_f32(3.0));
        list_builder.append(true);

        // v1 = [4, 5, 6]
        list_builder.values().append_value(f16::from_f32(4.0));
        list_builder.values().append_value(f16::from_f32(5.0));
        list_builder.values().append_value(f16::from_f32(6.0));
        list_builder.append(true);

        // v2 = [-1, 0.5, 2]
        list_builder.values().append_value(f16::from_f32(-1.0));
        list_builder.values().append_value(f16::from_f32(0.5));
        list_builder.values().append_value(f16::from_f32(2.0));
        list_builder.append(true);
        let vecs = list_builder.finish();

        // Query scalar q = [1, -1, 0.5]
        let mut q = FixedSizeListBuilder::new(Float16Builder::new(), 3);
        q.values().append_value(f16::from_f32(1.0));
        q.values().append_value(f16::from_f32(-1.0));
        q.values().append_value(f16::from_f32(0.5));
        q.append(true);
        let q = q.finish();
        let q = Scalar::new(q);

        // Compile and run kernel
        let kernel = DotKernel::compile(&(&q, &vecs), ()).unwrap();
        let out = kernel.call((&q, &vecs)).unwrap();
        let out = out.as_primitive::<Float16Type>();

        // Expected: [0.5, 2.0, -0.5]
        assert_eq!(out.len(), 3);
        assert_eq!(out.value(0), f16::from_f32(0.5));
        assert_eq!(out.value(1), f16::from_f32(2.0));
        assert_eq!(out.value(2), f16::from_f32(-0.5));
    }

    #[test]
    fn test_vec_norm_kernel() {
        // Build a FixedSizeList<f16, 3> array of vectors
        let mut list_builder = FixedSizeListBuilder::new(Float16Builder::new(), 3);

        // v0 = [1, 2, 3]
        list_builder.values().append_value(f16::from_f32(1.0));
        list_builder.values().append_value(f16::from_f32(2.0));
        list_builder.values().append_value(f16::from_f32(3.0));
        list_builder.append(true);

        // v1 = [4, 5, 6]
        list_builder.values().append_value(f16::from_f32(4.0));
        list_builder.values().append_value(f16::from_f32(5.0));
        list_builder.values().append_value(f16::from_f32(6.0));
        list_builder.append(true);

        // v2 = [-1, 0.5, 2]
        list_builder.values().append_value(f16::from_f32(-1.0));
        list_builder.values().append_value(f16::from_f32(0.5));
        list_builder.values().append_value(f16::from_f32(2.0));
        list_builder.append(true);
        let vecs: Arc<dyn Datum> = Arc::new(list_builder.finish());

        // Compile and run kernel
        let kernel = NormVecKernel::compile(&vecs.as_ref(), ()).unwrap();
        let out = kernel.call(vecs.as_ref()).unwrap();
        let out = out.get().0.as_fixed_size_list();

        assert_eq!(out.len(), 3);
        assert_eq!(out.value(0).len(), 3);
        assert_eq!(out.value(1).len(), 3);
        assert_eq!(out.value(2).len(), 3);

        let mut list_builder = FixedSizeListBuilder::new(Float16Builder::new(), 3);
        list_builder.values().append_value(f16::from_f32(1.0));
        list_builder.values().append_value(f16::from_f32(2.0));
        list_builder.values().append_value(f16::from_f32(3.0));
        list_builder.append(true);
        let s: Arc<dyn Datum> = Arc::new(Scalar::new(list_builder.finish()));
        let kernel = NormVecKernel::compile(&s.as_ref(), ()).unwrap();
        let r = kernel.call(s.as_ref()).unwrap();

        let (r, is_scalar) = r.get();
        assert!(is_scalar);
        assert_eq!(r.as_fixed_size_list().value(0).len(), 3);
    }
}
