use std::sync::Arc;

use arrow_array::{cast::AsArray, Array, ArrayRef, Datum, Scalar};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        dsl::{DSLKernel, KernelOutputType},
        replace_nulls,
    },
    logical_nulls, ArrayDatum, ArrowKernelError, Kernel,
};

pub struct DotKernel(DSLKernel);
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
        let mut res = self.0.call(&[&vecs, q])?;
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

        if vecs.values().is_nullable() {
            if vecs.values().null_count() > 0 {
                return Err(ArrowKernelError::UnsupportedArguments(
                    "vector values must not contain null".to_string(),
                ));
            }
        }

        if vecs.values().data_type() != q_val.values().data_type() {
            return Err(ArrowKernelError::UnsupportedArguments(format!(
                "vector and query must have the same data type (got {} and {})",
                vecs.values().data_type(),
                q_val.values().data_type()
            )));
        }

        Ok(DotKernel(
            DSLKernel::compile(&[vecs, *q], |ctx| {
                let vecs = ctx.get_input(0)?;
                let q = ctx.get_input(1)?;
                ctx.iter_over(vec![vecs, q])
                    .map(|i| vec![i[0].mul(&i[1]).vec_sum()])
                    .collect(KernelOutputType::Array)
            })
            .map_err(ArrowKernelError::DSLError)?,
        ))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((i.0.get().0.data_type().clone(), i.1.data_type().clone()))
    }
}

pub struct NormVecKernel(DSLKernel);
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
        let mut res = self.0.call(&[&data])?;
        if let Some(nulls) = logical_nulls(data)? {
            res = replace_nulls(res, Some(nulls));
        }

        if is_scalar {
            return Ok(Arc::new(Scalar::new(res)));
        }

        return Ok(Arc::new(ArrayDatum(res)));
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, _is_scalar) = inp.get();

        let arr = arr.as_fixed_size_list_opt().ok_or_else(|| {
            ArrowKernelError::UnsupportedArguments(format!(
                "Vectors must be fixed size list, got {}",
                arr.data_type()
            ))
        })?;

        if arr.values().is_nullable() {
            if arr.values().null_count() > 0 {
                return Err(ArrowKernelError::UnsupportedArguments(
                    "vector values must not contain null".to_string(),
                ));
            }
        }

        Ok(NormVecKernel(
            DSLKernel::compile(&[arr], |ctx| {
                let vecs = ctx.get_input(0)?;
                ctx.iter_over(vec![vecs])
                    .map(|i| vec![i[0].powi(2).vec_sum().sqrt(), i[0].clone()])
                    .map(|i| vec![i[1].div(&i[0].splat(arr.value_length() as usize))])
                    .collect(KernelOutputType::FixedSizeList(arr.value_length() as usize))
            })
            .map_err(ArrowKernelError::DSLError)?,
        ))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok(i.get().0.data_type().clone())
    }
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
