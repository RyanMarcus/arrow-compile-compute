use std::sync::LazyLock;

use arrow_array::{Array, ArrayRef, Datum};
use arrow_buffer::{BooleanBufferBuilder, NullBuffer};
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        cast::coalesce_type,
        dsl2::{
            compile, DSLArgument, DSLContext, DSLFunction, DSLStmt, DSLType, OutputSlot,
            OutputSpec, RunnableDSLFunction,
        },
        KernelCache,
    },
    compiled_writers::WriterSpec,
    logical_arrow_type, logical_nulls, ArrowKernelError, Kernel, PrimitiveType,
};

pub fn concat_all(data: &[&dyn Array]) -> Result<ArrayRef, ArrowKernelError> {
    concat_with_spec(data, &concat_writer_spec(data[0].data_type()))
}

fn concat_with_spec(data: &[&dyn Array], spec: &WriterSpec) -> Result<ArrayRef, ArrowKernelError> {
    let total_els = data.iter().map(|x| x.len()).sum::<usize>();
    let mut alloc = OutputSpec::new(spec.clone(), "n").allocate(total_els);

    let nulls = if data.iter().any(|arr| arr.is_nullable()) {
        let mut bb = BooleanBufferBuilder::new(total_els);
        for el in data.iter() {
            match logical_nulls(*el)? {
                Some(nb) => bb.append_buffer(nb.inner()),
                None => bb.append_n(el.len(), true),
            }
        }
        Some(NullBuffer::new(bb.finish()))
    } else {
        None
    };

    for arr in data {
        CONCAT_PROGRAM_CACHE.get((*arr, &mut alloc), ())?;
    }

    assert_eq!(alloc.len(), total_els);
    let arr = alloc.into_array_ref(nulls);
    coalesce_type(
        arr,
        &match logical_arrow_type(data[0].data_type()) {
            DataType::Utf8 => DataType::Utf8View,
            DataType::LargeUtf8 => DataType::Utf8View,
            DataType::Binary => DataType::BinaryView,
            DataType::LargeBinary => DataType::BinaryView,
            dt => dt,
        },
    )
}

static CONCAT_PROGRAM_CACHE: LazyLock<KernelCache<ConcatKernel>> = LazyLock::new(KernelCache::new);

pub struct ConcatKernel(RunnableDSLFunction);

unsafe impl Sync for ConcatKernel {}
unsafe impl Send for ConcatKernel {}

impl Kernel for ConcatKernel {
    type Key = DataType;

    type Input<'a> = (&'a dyn Array, &'a mut OutputSlot);

    type Params = ();

    type Output = ();

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let (arr, output) = inp;
        let datum = &arr as &dyn Datum;
        self.0
            .run_into(&[DSLArgument::Datum(datum)], std::slice::from_mut(output))
    }

    fn compile(inp: &Self::Input<'_>, _params: Self::Params) -> Result<Self, ArrowKernelError> {
        let (arr, _output) = inp;
        let datum = arr as &dyn Datum;

        let mut ctx = DSLContext::new();
        let mut func = DSLFunction::new("concat");
        let arg = func.add_arg(&mut ctx, DSLType::array_like(datum, "n"));
        func.add_ret(concat_writer_spec(arr.data_type()), "n");
        func.add_body(DSLStmt::for_each(&mut ctx, &[arg], |loop_vars| {
            DSLStmt::emit(0, loop_vars[0].expr())
        })?);

        let func = compile(func, [DSLArgument::Datum(datum)])?;
        Ok(Self(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        _p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok(i.0.data_type().clone())
    }
}

fn concat_writer_spec(dt: &DataType) -> WriterSpec {
    match logical_arrow_type(dt) {
        DataType::Boolean => WriterSpec::Boolean,
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView => WriterSpec::StringView,
        DataType::FixedSizeList(field, len) => WriterSpec::FixedSizeList(
            PrimitiveType::for_arrow_type(field.data_type())
                .try_into()
                .unwrap(),
            len as usize,
        ),
        dt => WriterSpec::Primitive(PrimitiveType::for_arrow_type(&dt)),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray, types::Int32Type, Array, BooleanArray, FixedSizeListArray, Float32Array,
        Int32Array, StringArray,
    };

    use itertools::Itertools;

    use crate::{
        compiled_kernels::{concat::ConcatKernel, dsl2::OutputSpec},
        Kernel,
    };

    #[test]
    fn test_concat_i32() {
        let d1 = Int32Array::from(vec![1, 2, 3, 4]);
        let d2 = Int32Array::from(vec![5, 6, 7, 8]);
        let mut alloc = OutputSpec::new(super::concat_writer_spec(d1.data_type()), "n").allocate(8);

        let k = ConcatKernel::compile(&(&d1, &mut alloc), ()).unwrap();

        k.call((&d1, &mut alloc)).unwrap();
        k.call((&d2, &mut alloc)).unwrap();

        let res = alloc.into_array_ref(None);
        let res = res.as_primitive::<Int32Type>();
        let res: Vec<i32> = res.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_concat_strs() {
        let d1 = StringArray::from(vec!["hello", "world"]);
        let d2 = StringArray::from(vec![
            "!",
            "!",
            "this is a longer string that is more than 12 chars",
        ]);
        let mut alloc = OutputSpec::new(super::concat_writer_spec(d1.data_type()), "n").allocate(5);

        let k = ConcatKernel::compile(&(&d1, &mut alloc), ()).unwrap();
        k.call((&d1, &mut alloc)).unwrap();
        k.call((&d2, &mut alloc)).unwrap();

        let res = alloc.into_array_ref(None);
        let res =
            crate::compiled_kernels::cast::coalesce_type(res, &arrow_schema::DataType::Utf8View)
                .unwrap();
        let res: Vec<String> = res
            .as_string_view()
            .iter()
            .map(|x| x.unwrap().to_string())
            .collect_vec();
        assert_eq!(
            res,
            vec![
                "hello".to_string(),
                "world".to_string(),
                "!".to_string(),
                "!".to_string(),
                "this is a longer string that is more than 12 chars".to_string()
            ]
        );
    }

    #[test]
    fn test_concat_fixed_size_list_f32x4() {
        let d1_values = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let d2_values = Float32Array::from(vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);

        let field = Arc::new(arrow_schema::Field::new_list_field(
            arrow_schema::DataType::Float32,
            false,
        ));

        let d1 = FixedSizeListArray::try_new(field.clone(), 4, Arc::new(d1_values), None).unwrap();
        let d2 = FixedSizeListArray::try_new(field, 4, Arc::new(d2_values), None).unwrap();

        let mut alloc = OutputSpec::new(super::concat_writer_spec(d1.data_type()), "n").allocate(4);

        let k = ConcatKernel::compile(&(&d1, &mut alloc), ()).unwrap();
        k.call((&d1, &mut alloc)).unwrap();
        k.call((&d2, &mut alloc)).unwrap();

        let res = alloc.into_array_ref(None);
        let res = res.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

        let values = res.values();
        let values = values.as_any().downcast_ref::<Float32Array>().unwrap();
        let values: Vec<f32> = values.iter().map(|x| x.unwrap()).collect_vec();

        assert_eq!(
            values,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        );
    }

    #[test]
    fn test_concat_bools() {
        let d1 = BooleanArray::from(vec![true, false, true]);
        let d2 = BooleanArray::from(vec![false, true]);
        let mut alloc = OutputSpec::new(super::concat_writer_spec(d1.data_type()), "n").allocate(5);

        let k = ConcatKernel::compile(&(&d1, &mut alloc), ()).unwrap();
        k.call((&d1, &mut alloc)).unwrap();
        k.call((&d2, &mut alloc)).unwrap();

        let res = alloc.into_array_ref(None);
        let res = res.as_boolean();
        assert_eq!(
            res.iter().map(|x| x.unwrap()).collect_vec(),
            vec![true, false, true, false, true]
        );
    }
}
