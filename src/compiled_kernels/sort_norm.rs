use std::sync::{Arc, LazyLock};

use arrow_array::{
    builder::BinaryBuilder,
    cast::AsArray,
    types::{UInt16Type, UInt32Type, UInt64Type, UInt8Type},
    ArrayRef, BinaryArray, BooleanArray, Datum, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;

use crate::{
    compiled_kernels::{
        dsl2::{
            compile, DSLArgument, DSLBitwiseBinOp, DSLContext, DSLExpr, DSLFunction, DSLStmt,
            DSLType, DSLValue, RunnableDSLFunction,
        },
        KernelCache,
    },
    compiled_writers::{BooleanAllocation, WriterSpec},
    iter, ArrowKernelError, Kernel, PrimitiveType, SortOptions,
};

enum NormalizedColumn {
    UInt8(UInt8Array),
    UInt16(UInt16Array),
    UInt32(UInt32Array),
    UInt64(UInt64Array),
    Binary(BinaryArray),
}

impl TryFrom<ArrayRef> for NormalizedColumn {
    type Error = ArrowKernelError;

    fn try_from(value: ArrayRef) -> Result<Self, Self::Error> {
        match PrimitiveType::for_arrow_type(value.data_type()) {
            PrimitiveType::U8 => Ok(NormalizedColumn::UInt8(
                value.as_primitive::<UInt8Type>().clone(),
            )),
            PrimitiveType::U16 => Ok(NormalizedColumn::UInt16(
                value.as_primitive::<UInt16Type>().clone(),
            )),
            PrimitiveType::U32 => Ok(NormalizedColumn::UInt32(
                value.as_primitive::<UInt32Type>().clone(),
            )),
            PrimitiveType::U64 => Ok(NormalizedColumn::UInt64(
                value.as_primitive::<UInt64Type>().clone(),
            )),
            PrimitiveType::P64x2 => Ok(NormalizedColumn::Binary(value.as_binary::<i32>().clone())),
            _ => Err(ArrowKernelError::UnsupportedArguments(format!(
                "unknown type of normed array: {}",
                value.data_type()
            ))),
        }
    }
}

impl NormalizedColumn {
    pub fn use_bytes<F: FnMut(&[u8])>(&self, mut f: F) {
        match self {
            NormalizedColumn::UInt8(arr) => arr.values().iter().for_each(|v| f(&v.to_le_bytes())),
            NormalizedColumn::UInt16(arr) => arr.values().iter().for_each(|v| f(&v.to_le_bytes())),
            NormalizedColumn::UInt32(arr) => arr.values().iter().for_each(|v| f(&v.to_le_bytes())),
            NormalizedColumn::UInt64(arr) => arr.values().iter().for_each(|v| f(&v.to_le_bytes())),
            NormalizedColumn::Binary(arr) => arr.iter().for_each(|v| f(v.unwrap())),
        }
    }
}

pub fn normalize_columns(
    arrs: &[(&dyn Datum, SortOptions)],
) -> Result<BinaryArray, ArrowKernelError> {
    let witness_len = arrs[0].0.get().0.len();
    if !arrs.iter().all(|(arr, _)| arr.get().0.len() == witness_len) {
        return Err(ArrowKernelError::SizeMismatch);
    }

    let mut normed: Vec<ArrayRef> = Vec::new();
    for (arr, opts) in arrs.iter() {
        let arr = arr.get().0;
        let pt = PrimitiveType::for_arrow_type(arr.data_type());

        if let Some(nb) = arr.nulls() {
            normed.push(Arc::new(normalize_nulls(nb, opts.nulls_first)));
        }

        if pt.is_int() || pt.is_float() {
            let normalized = NORM_NUMERIC_CACHE.get(&arr, opts.descending)?;
            normed.push(normalized);
        } else if matches!(pt, PrimitiveType::P64x2) {
            normed.push(normalize_bytes(&arr, opts.descending)?);
        } else {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "non-sortable type: {}",
                arr.data_type()
            )));
        }
    }

    let mut normed_bytes: Vec<Vec<u8>> = vec![vec![]; witness_len];
    for arr in normed {
        let arr = NormalizedColumn::try_from(arr)?;
        let mut idx = 0;
        arr.use_bytes(|bytes| {
            normed_bytes[idx].extend_from_slice(bytes);
            idx += 1;
        });
    }

    let mut bb = BinaryBuilder::new();
    for bytes in normed_bytes {
        bb.append_value(bytes);
    }
    Ok(bb.finish())
}

static NORM_NUMERIC_CACHE: LazyLock<KernelCache<NormalizeNumeric>> =
    LazyLock::new(KernelCache::new);

struct NormalizeNumeric(RunnableDSLFunction);
unsafe impl Send for NormalizeNumeric {}
unsafe impl Sync for NormalizeNumeric {}

impl Kernel for NormalizeNumeric {
    type Key = (DataType, bool, bool);

    type Input<'a> = &'a dyn Datum;

    type Params = bool;

    type Output = ArrayRef;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let nulls = if let Some(nb) = inp.get().0.nulls() {
            let ba = BooleanArray::from(nb.inner().clone());
            Arc::new(ba) as Arc<dyn Datum>
        } else {
            Arc::new(BooleanArray::new_scalar(true))
        };

        let res = self
            .0
            .run(&[DSLArgument::Datum(inp), DSLArgument::Datum(nulls.as_ref())])?;
        let res = res[0].clone();
        Ok(res)
    }

    fn compile(inp: &Self::Input<'_>, invert: Self::Params) -> Result<Self, ArrowKernelError> {
        let pt = PrimitiveType::for_arrow_type((*inp).get().0.data_type());
        if !pt.is_float() && !pt.is_int() {
            return Err(ArrowKernelError::ArgumentMismatch(
                "non-numeric type".to_string(),
            ));
        }

        Ok(Self(normalize_numeric(*inp, invert)?))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        Ok((
            i.get().0.data_type().clone(),
            i.get().0.nulls().is_some(),
            *p,
        ))
    }
}

fn normalize_numeric(
    arr: &dyn Datum,
    invert: bool,
) -> Result<RunnableDSLFunction, ArrowKernelError> {
    let inp_pt = PrimitiveType::for_arrow_type(arr.get().0.data_type());
    let mut ctx = DSLContext::new();
    let mut func = DSLFunction::new("norm_sint");
    let arr_arg = func.add_arg(&mut ctx, DSLType::array_like(arr, "n"));

    let nul: Arc<dyn Datum> = if let Some(nb) = arr.get().0.nulls() {
        Arc::new(BooleanArray::from(nb.inner().clone()))
    } else {
        Arc::new(BooleanArray::new_scalar(true))
    };
    let nul_arg = func.add_arg(&mut ctx, DSLType::array_like(&*nul, "n"));

    let out_type = match inp_pt {
        PrimitiveType::I8 => PrimitiveType::U8,
        PrimitiveType::I16 => PrimitiveType::U16,
        PrimitiveType::I32 => PrimitiveType::U32,
        PrimitiveType::I64 => PrimitiveType::U64,
        PrimitiveType::F16 => PrimitiveType::U16,
        PrimitiveType::F32 => PrimitiveType::U32,
        PrimitiveType::F64 => PrimitiveType::U64,
        _ => inp_pt,
    };
    func.add_ret(WriterSpec::Primitive(out_type), "n");

    func.add_body(
        DSLStmt::for_each(&mut ctx, &[arr_arg, nul_arg], |loop_vars| {
            let mut itm = loop_vars[0].expr();
            let is_valid = loop_vars[1].expr();

            if inp_pt.is_float() {
                itm = itm.float_to_total_order_sint()?;
            }

            if inp_pt.is_signed() {
                let last_bit = match PrimitiveType::for_arrow_type(arr.get().0.data_type()).width()
                {
                    1 => DSLValue::u8(1 << 7),
                    2 => DSLValue::u16(1 << 15),
                    4 => DSLValue::u32(1 << 31),
                    8 => DSLValue::u64(1 << 63),
                    _ => unreachable!(),
                }
                .expr()
                .primitive_cast(itm.get_type().as_primitive().unwrap())?;
                itm = itm.bitwise(DSLBitwiseBinOp::Xor, last_bit)?;
            }
            #[cfg(target_endian = "little")]
            {
                itm = itm.bswap()?;
            }
            itm = itm.bit_cast(out_type)?;

            if invert {
                itm = itm.bit_not()?;
            }

            let zero = DSLValue::zero_like(&itm).as_primitive_expr()?;
            itm = is_valid.select(itm, zero)?;
            DSLStmt::emit(0, itm)
        })
        .unwrap(),
    );

    let func = compile(
        func,
        [DSLArgument::Datum(arr), DSLArgument::Datum(nul.as_ref())],
    )?;
    Ok(func)
}

fn normalize_bytes(arr: &dyn Datum, invert: bool) -> Result<ArrayRef, ArrowKernelError> {
    let mut bb = BinaryBuilder::new();
    let mut row_buf = Vec::new();
    for s in iter::iter_bytes(arr.get().0)? {
        row_buf.clear();
        if let Some(s) = s {
            for &b in s {
                let encoded = if invert { !b } else { b };
                row_buf.push(encoded);
                if (!invert && encoded == 0) || (invert && encoded == 255) {
                    row_buf.push(if invert { 0 } else { 255 });
                }
            }
        }
        row_buf.extend_from_slice(if invert { &[255, 255] } else { &[0, 0] });
        bb.append_value(&row_buf);
    }

    Ok(Arc::new(bb.finish()))
}

fn normalize_nulls(nb: &NullBuffer, nulls_first: bool) -> UInt8Array {
    nb.iter()
        .map(|b| {
            if nulls_first {
                if b {
                    1
                } else {
                    0
                }
            } else {
                if b {
                    0
                } else {
                    1
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray, types::UInt32Type, Array, ArrayRef, BinaryArray, BooleanArray, Datum,
        Float32Array, Int32Array, UInt32Array,
    };
    use arrow_buffer::NullBuffer;
    use itertools::Itertools;

    use super::{normalize_columns, normalize_numeric};
    use crate::{compiled_kernels::dsl2::DSLArgument, sort, SortOptions};

    fn lexicographic_sort_indices(arr: &BinaryArray) -> Vec<u32> {
        let mut idxes = (0..arr.len() as u32).collect_vec();
        idxes.sort_by(|lhs, rhs| {
            arr.value(*lhs as usize)
                .cmp(arr.value(*rhs as usize))
                .then(lhs.cmp(rhs))
        });
        idxes
    }

    fn assert_normalized_sort_matches(arrs: &[ArrayRef], options: &[SortOptions]) {
        let sort_inputs = arrs
            .iter()
            .map(|arr| arr.as_ref() as &dyn Array)
            .collect_vec();
        let datum_inputs = sort_inputs
            .iter()
            .zip(options.iter().copied())
            .map(|(arr, opts)| (arr as &dyn Datum, opts))
            .collect_vec();

        let normalized = normalize_columns(&datum_inputs).unwrap();
        let expected = sort::multicol_sort_to_indices(&sort_inputs, options).unwrap();
        let expected = expected.values().iter().copied().collect_vec();

        assert_eq!(normalized.len(), arrs[0].len());
        assert_eq!(lexicographic_sort_indices(&normalized), expected);
    }

    #[test]
    fn test_signed_int_normalization() {
        let data = Int32Array::from(vec![1, -5, 100, -1000]);
        let func = normalize_numeric(&data, false).unwrap();
        let valid = BooleanArray::new_scalar(true);
        let out = func
            .run(&[DSLArgument::datum(&data), DSLArgument::datum(&valid)])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].as_primitive::<UInt32Type>().values(),
            &[0x01000080, 0xFBFFFF7F, 0x64000080, 0x18FCFF7F]
        )
    }

    #[test]
    fn test_unsigned_int_normalization() {
        let data = UInt32Array::from(vec![1, 5, 100, 1000]);
        let func = normalize_numeric(&data, false).unwrap();
        let valid = BooleanArray::new_scalar(true);
        let out = func
            .run(&[DSLArgument::datum(&data), DSLArgument::datum(&valid)])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].as_primitive::<UInt32Type>().values(),
            &[0x01000000, 0x05000000, 0x64000000, 0xE8030000]
        )
    }

    #[test]
    fn test_unsigned_f32_normalization() {
        let data = Float32Array::from(vec![1.0, 5.5, 100.0, -1000.0]);
        let func = normalize_numeric(&data, false).unwrap();
        let valid = BooleanArray::new_scalar(true);
        let out = func
            .run(&[DSLArgument::datum(&data), DSLArgument::datum(&valid)])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].as_primitive::<UInt32Type>().values(),
            &[
                0x000080BF, // 1.0
                0x0000B0C0, // 5.5
                0x0000C8C2, // 100.0
                0xFFFF853B  // -1000.0
            ]
        )
    }

    #[test]
    fn test_normalize_columns_lexicographic_sort_matches_multicol_sort() {
        let bytes = Arc::new(BinaryArray::from(vec![
            Some(&b"a"[..]),
            Some(&b"a\0"[..]),
            Some(&b"a"[..]),
            Some(&b"a\0z"[..]),
            Some(&b"a\0"[..]),
            Some(&b""[..]),
        ])) as ArrayRef;
        let ints = Arc::new(UInt32Array::from(vec![1, 0, 0, 2, 1, 3])) as ArrayRef;

        assert_normalized_sort_matches(
            &[bytes, ints],
            &[SortOptions::default(), SortOptions::default()],
        );
    }

    #[test]
    fn test_normalize_columns_lexicographic_sort_matches_multicol_sort_descending() {
        let floats = Arc::new(Float32Array::from(vec![
            1.0,
            -0.0,
            2.5,
            f32::NEG_INFINITY,
            2.5,
            1.0,
        ])) as ArrayRef;
        let bytes = Arc::new(BinaryArray::from(vec![
            Some(&b"b"[..]),
            Some(&b"a\0"[..]),
            Some(&b"a"[..]),
            Some(&b"zz"[..]),
            Some(&b"a\0z"[..]),
            Some(&b"b"[..]),
        ])) as ArrayRef;

        let descending = SortOptions {
            descending: true,
            nulls_first: false,
        };
        assert_normalized_sort_matches(&[floats, bytes], &[descending, descending]);
    }

    #[test]
    fn test_normalize_columns_lexicographic_sort_matches_multicol_sort_nulls_first() {
        let ints = Arc::new(Int32Array::new(
            vec![50, 777, 10, -999, 10, 0].into(),
            Some(NullBuffer::from(vec![true, false, true, false, true, true])),
        )) as ArrayRef;
        let bytes = Arc::new(BinaryArray::from(vec![
            Some(&b"bb"[..]),
            Some(&b"zz"[..]),
            Some(&b"aa"[..]),
            None,
            Some(&b"aa\0"[..]),
            Some(&b""[..]),
        ])) as ArrayRef;

        assert_normalized_sort_matches(
            &[ints, bytes],
            &[
                SortOptions {
                    descending: false,
                    nulls_first: true,
                },
                SortOptions::default(),
            ],
        );
    }
}
