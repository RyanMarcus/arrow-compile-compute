use std::{ffi::c_void, marker::PhantomData, sync::Arc};

use arrow_array::{Array, BooleanArray};
use arrow_schema::DataType;
use inkwell::{
    context::Context, execution_engine::JitFunction, AddressSpace, IntPredicate, OptimizationLevel,
};
use ouroboros::self_referencing;

use crate::{
    compiled_iter::{
        array_to_setbit_iter, datum_to_iter, generate_next, generate_random_access, IteratorHolder,
    },
    compiled_kernels::{gen_convert_numeric_vec, link_req_helpers, optimize_module},
    declare_blocks, increment_pointer, Kernel, PrimitiveType,
};

use super::ArrowKernelError;

const BLOCK_SIZE: usize = 64;

pub trait ApplyType: Copy {
    fn primitive_type() -> PrimitiveType;

    unsafe fn from_byte_slice(data: &[u8]) -> Self {
        std::slice::from_raw_parts(data.as_ptr() as *const Self, 1)[0]
    }
}
impl ApplyType for i64 {
    fn primitive_type() -> PrimitiveType {
        PrimitiveType::I64
    }
}
impl ApplyType for f64 {
    fn primitive_type() -> PrimitiveType {
        PrimitiveType::F64
    }
}
impl ApplyType for u64 {
    fn primitive_type() -> PrimitiveType {
        PrimitiveType::U64
    }
}
impl ApplyType for u8 {
    fn primitive_type() -> PrimitiveType {
        PrimitiveType::U8
    }
}
impl ApplyType for &[u8] {
    fn primitive_type() -> PrimitiveType {
        PrimitiveType::P64x2
    }

    unsafe fn from_byte_slice(data: &[u8]) -> Self {
        let nums = std::slice::from_raw_parts(data.as_ptr() as *const u64, 2);
        let start = nums[0] as *const u8;
        let end = nums[1] as *const u8;
        let len = end.offset_from(start);
        debug_assert!(len >= 0);
        std::slice::from_raw_parts(start, len as usize)
    }
}

#[self_referencing]
pub struct IterFuncHolder {
    context: Context,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<
        'this,
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u8, *mut u64) -> u64,
    >,
}
unsafe impl Send for IterFuncHolder {}
unsafe impl Sync for IterFuncHolder {}

impl Kernel for Arc<IterFuncHolder> {
    type Key = (DataType, bool, PrimitiveType);

    type Input<'a> = &'a dyn Array;

    type Params = (PrimitiveType, bool);

    type Output = Self;

    fn call(&self, _inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        Ok(self.clone())
    }

    fn compile(inp: &Self::Input<'_>, params: Self::Params) -> Result<Self, ArrowKernelError> {
        let ih = datum_to_iter(inp)?;
        let (target_type, ignore_nulls) = params;
        let setbit_ih = (!ignore_nulls)
            .then(|| {
                inp.logical_nulls().map(|nulls| {
                    array_to_setbit_iter(&BooleanArray::from(nulls.clone().into_inner()))
                })
            })
            .flatten()
            .transpose()?;

        let func = IterFuncHolderTryBuilder {
            context: Context::create(),
            func_builder: |ctx| {
                generate_call(ctx, inp.data_type(), &ih, setbit_ih.as_ref(), target_type)
            },
        }
        .try_build()?;
        Ok(Arc::new(func))
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        p: &Self::Params,
    ) -> Result<Self::Key, ArrowKernelError> {
        let (target_dt, ignore_nulls) = p;
        Ok((
            i.data_type().clone(),
            i.logical_nulls().is_some() && !ignore_nulls,
            *target_dt,
        ))
    }
}

pub struct ArrowIter<T: ApplyType> {
    buffer: [u8; 64 * PrimitiveType::max_width()],
    ibuffer: [u64; 64],
    buffer_idx: usize,
    buffer_len: usize,
    iter_holder: IteratorHolder,
    setbit_iter_holder: Option<IteratorHolder>,

    next_func: Arc<IterFuncHolder>,
    pd: PhantomData<T>,
}

impl<T: ApplyType> ArrowIter<T> {
    /// Causes this iterator to additionally return the index of each element.
    /// Note this is different from `enumerate`, as this function returns the
    /// indexes of non-null elements only.
    /// # Example
    /// ```
    /// use arrow_array::Int32Array;
    /// use arrow_compile_compute::iter::iter_nonnull_i64;
    ///
    /// let arr = Int32Array::from(vec![Some(1), None, Some(3)]);
    /// let iter = iter_nonnull_i64(&arr).unwrap().indexed();
    /// assert_eq!(iter.collect::<Vec<_>>(), vec![(0, 1), (2, 3)]);
    /// ```
    pub fn indexed(self) -> IndexedArrowIter<T> {
        IndexedArrowIter { iter: self }
    }

    fn load_next(&mut self) -> bool {
        let num_returned = unsafe {
            self.next_func.borrow_func().call(
                self.iter_holder.get_mut_ptr(),
                self.setbit_iter_holder
                    .as_mut()
                    .map(|x| x.get_mut_ptr())
                    .unwrap_or(std::ptr::null_mut()),
                self.buffer.as_mut_ptr(),
                self.ibuffer.as_mut_ptr(),
            )
        };

        if num_returned == 0 {
            return false;
        }

        self.buffer_idx = 0;
        self.buffer_len = num_returned as usize;
        return true;
    }
}

impl<T: ApplyType> Iterator for ArrowIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer_idx < self.buffer_len {
            let width = T::primitive_type().width();
            let slice = unsafe {
                self.buffer
                    .get_unchecked(self.buffer_idx * width..(self.buffer_idx + 1) * width)
            };
            self.buffer_idx += 1;
            Some(unsafe { T::from_byte_slice(slice) })
        } else {
            if !self.load_next() {
                return None;
            }
            self.next()
        }
    }
}

impl<T: ApplyType> ArrowIter<T> {
    pub fn new(data: &dyn Array, func: Arc<IterFuncHolder>) -> Result<Self, ArrowKernelError> {
        let ih = datum_to_iter(&data)?;
        let setbit_ih = data
            .logical_nulls()
            .map(|nulls| array_to_setbit_iter(&BooleanArray::from(nulls.clone().into_inner())))
            .transpose()?;

        Ok(ArrowIter {
            buffer: [0; 1024],
            ibuffer: [0; 64],
            buffer_idx: 0,
            buffer_len: 0,
            iter_holder: ih,
            setbit_iter_holder: setbit_ih,
            next_func: func,
            pd: PhantomData,
        })
    }
}

pub struct IndexedArrowIter<T: ApplyType> {
    iter: ArrowIter<T>,
}

impl<T: ApplyType> Iterator for IndexedArrowIter<T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iter.buffer_idx < self.iter.buffer_len {
            let width = T::primitive_type().width();
            let slice = unsafe {
                self.iter
                    .buffer
                    .get_unchecked(self.iter.buffer_idx * width..(self.iter.buffer_idx + 1) * width)
            };
            let idx = self.iter.ibuffer[self.iter.buffer_idx];
            self.iter.buffer_idx += 1;
            Some((idx as usize, unsafe { T::from_byte_slice(slice) }))
        } else {
            if !self.iter.load_next() {
                return None;
            }
            self.next()
        }
    }
}

pub struct ArrowNullableIter<T: ApplyType> {
    data_iter: ArrowIter<T>,
    null_iter: Option<ArrowIter<u8>>,
}

impl<T: ApplyType> ArrowNullableIter<T> {
    pub fn new(data_iter: ArrowIter<T>, null_iter: Option<ArrowIter<u8>>) -> Self {
        ArrowNullableIter {
            data_iter,
            null_iter,
        }
    }
}

impl<T: ApplyType> Iterator for ArrowNullableIter<T> {
    type Item = Option<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match (
            self.data_iter.next(),
            self.null_iter
                .as_mut()
                .and_then(|iter| iter.next())
                .or(Some(1)),
        ) {
            (Some(data), Some(1)) => Some(Some(data)),
            (Some(_), Some(0)) => Some(None),
            _ => None,
        }
    }
}

fn generate_call<'a>(
    ctx: &'a Context,
    dt: &DataType,
    ih: &IteratorHolder,
    setbit_ih: Option<&IteratorHolder>,
    rust_expected_type: PrimitiveType,
) -> Result<
    JitFunction<'a, unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u8, *mut u64) -> u64>,
    ArrowKernelError,
> {
    let module = ctx.create_module("call");
    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let i64_type = ctx.i64_type();
    let input_prim_type = PrimitiveType::for_arrow_type(dt);
    let rust_expected_llvm_type = rust_expected_type.llvm_type(ctx);
    let input_type = input_prim_type.llvm_type(ctx);
    let func = module.add_function(
        "call_rust",
        i64_type.fn_type(
            &[
                ptr_type.into(),
                ptr_type.into(),
                ptr_type.into(),
                ptr_type.into(),
            ],
            false,
        ),
        None,
    );

    let next = generate_next(ctx, &module, "call", dt, ih).unwrap();
    let access = generate_random_access(ctx, &module, "call", dt, ih).unwrap();
    let next_bit = setbit_ih
        .map(|ih| generate_next(ctx, &module, "call_bit", &DataType::Boolean, ih).unwrap());

    let build = ctx.create_builder();
    declare_blocks!(ctx, func, entry, loop_cond, loop_body, exit);

    build.position_at_end(entry);
    let iter_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let bit_iter_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
    let rust_buf_ptr = func.get_nth_param(2).unwrap().into_pointer_value();
    let rust_idx_ptr = func.get_nth_param(3).unwrap().into_pointer_value();
    let offset_ptr = build.build_alloca(i64_type, "offset_ptr").unwrap();
    let buf = build.build_alloca(input_type, "buf").unwrap();
    build
        .build_store(offset_ptr, i64_type.const_zero())
        .unwrap();
    let next_bit_buf = next_bit.map(|_| {
        let buf = build.build_alloca(i64_type, "next_bit_buf").unwrap();
        build
            .build_store(offset_ptr, i64_type.const_zero())
            .unwrap();
        buf
    });
    build.build_unconditional_branch(loop_cond).unwrap();

    build.position_at_end(loop_cond);
    let res = if let Some(next_bit) = next_bit {
        let next_bit_buf = next_bit_buf.unwrap();
        build
            .build_call(
                next_bit,
                &[bit_iter_ptr.into(), next_bit_buf.into()],
                "had_next_setbit",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value()
    } else {
        build
            .build_call(next, &[iter_ptr.into(), buf.into()], "had_next")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value()
    };

    build
        .build_conditional_branch(res, loop_body, exit)
        .unwrap();

    build.position_at_end(loop_body);
    let val = if let Some(next_bit_buf) = next_bit_buf {
        let next_bit = build
            .build_load(i64_type, next_bit_buf, "next_bit")
            .unwrap()
            .into_int_value();

        let curr_offset = build
            .build_load(i64_type, offset_ptr, "curr_offset")
            .unwrap()
            .into_int_value();
        build
            .build_store(
                increment_pointer!(ctx, build, rust_idx_ptr, 8, curr_offset),
                next_bit,
            )
            .unwrap();
        build
            .build_call(access, &[iter_ptr.into(), next_bit.into()], "access_el")
            .unwrap()
            .try_as_basic_value()
            .unwrap_left()
    } else {
        build.build_load(input_type, buf, "val").unwrap()
    };

    let val = match rust_expected_type {
        PrimitiveType::P64x2 => val,
        _ => {
            let in_v_type = input_prim_type.llvm_vec_type(ctx, 1).unwrap();
            let val = build
                .build_bit_cast(val, in_v_type, "singleton_vec")
                .unwrap()
                .into_vector_value();
            let r = gen_convert_numeric_vec(ctx, &build, val, input_prim_type, rust_expected_type);
            build
                .build_bit_cast(r, rust_expected_llvm_type, "vec_to_single")
                .unwrap()
        }
    };

    let curr_offset = build
        .build_load(i64_type, offset_ptr, "curr_offset")
        .unwrap()
        .into_int_value();
    build
        .build_store(
            increment_pointer!(
                ctx,
                build,
                rust_buf_ptr,
                rust_expected_type.width(),
                curr_offset
            ),
            val,
        )
        .unwrap();
    let next_offset = build
        .build_int_add(curr_offset, i64_type.const_int(1, false), "next_offset")
        .unwrap();
    build.build_store(offset_ptr, next_offset).unwrap();

    let cmp = build
        .build_int_compare(
            IntPredicate::UGE,
            next_offset,
            i64_type.const_int(BLOCK_SIZE as u64, false),
            "buf_full",
        )
        .unwrap();

    build
        .build_conditional_branch(cmp, exit, loop_cond)
        .unwrap();

    build.position_at_end(exit);
    let curr_buf_len = build
        .build_load(i64_type, offset_ptr, "curr_buf_len")
        .unwrap()
        .into_int_value();
    build.build_return(Some(&curr_buf_len)).unwrap();

    module.verify().unwrap();
    optimize_module(&module)?;
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();
    link_req_helpers(&module, &ee)?;

    Ok(unsafe {
        ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u8, *mut u64) -> u64>(
            func.get_name().to_str().unwrap(),
        )
        .unwrap()
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        types::Int64Type, Array, BooleanArray, Float32Array, Int32Array, Int64Array, RunArray,
        StringArray, UInt32Array,
    };
    use itertools::Itertools;

    use crate::{
        compiled_kernels::{
            rust_iter::{ArrowIter, IterFuncHolder},
            ArrowNullableIter,
        },
        Kernel, PrimitiveType,
    };

    #[test]
    fn test_iter_i32() {
        let data = Int32Array::from((0..1000).collect_vec());
        let ifh =
            Arc::<IterFuncHolder>::compile(&(&data as &dyn Array), (PrimitiveType::I64, false))
                .unwrap();
        let res = ArrowIter::<i64>::new(&data, ifh).unwrap().collect_vec();
        assert_eq!(res, (0..1000).collect_vec());
    }

    #[test]
    fn test_iter_i32_nulls() {
        let data = Int32Array::from(
            (0..1000)
                .map(|x| if x % 2 == 0 { Some(x) } else { None })
                .collect_vec(),
        );
        let ifh =
            Arc::<IterFuncHolder>::compile(&(&data as &dyn Array), (PrimitiveType::I64, false))
                .unwrap();
        let res = ArrowIter::<i64>::new(&data, ifh).unwrap().collect_vec();
        assert_eq!(res, (0..1000).filter(|x| x % 2 == 0).collect_vec());
    }

    #[test]
    fn test_iter_i32_nulls_indexed() {
        let data = Int32Array::from(
            (0..1000)
                .map(|x| if x % 2 == 0 { Some(x) } else { None })
                .collect_vec(),
        );
        let ifh =
            Arc::<IterFuncHolder>::compile(&(&data as &dyn Array), (PrimitiveType::I64, false))
                .unwrap();
        let res = ArrowIter::<i64>::new(&data, ifh)
            .unwrap()
            .indexed()
            .collect_vec();
        assert_eq!(
            res,
            (0..1000)
                .enumerate()
                .filter(|(_idx, x)| x % 2 == 0)
                .collect_vec()
        );
    }

    #[test]
    fn test_iter_u32() {
        let data = UInt32Array::from((0..1000).collect_vec());
        let ifh =
            Arc::<IterFuncHolder>::compile(&(&data as &dyn Array), (PrimitiveType::U64, false))
                .unwrap();
        let res = ArrowIter::<u64>::new(&data, ifh).unwrap().collect_vec();
        assert_eq!(res, (0..1000).collect_vec());
    }

    #[test]
    fn test_iter_f32() {
        let data = Float32Array::from((0..1000).map(|x| x as f32).collect_vec());
        let ifh =
            Arc::<IterFuncHolder>::compile(&(&data as &dyn Array), (PrimitiveType::F64, false))
                .unwrap();
        let res = ArrowIter::<f64>::new(&data, ifh).unwrap().collect_vec();
        assert_eq!(res, (0..1000).map(|x| x as f64).collect_vec());
    }

    #[test]
    fn test_iter_str() {
        let vdata = (0..1000).map(|i| format!("string{}", i)).collect_vec();
        let data = StringArray::from(vdata.clone());
        let ifh =
            Arc::<IterFuncHolder>::compile(&(&data as &dyn Array), (PrimitiveType::P64x2, false))
                .unwrap();
        let res = ArrowIter::<&[u8]>::new(&data, ifh)
            .unwrap()
            .map(|x| String::from_utf8(x.to_vec()).unwrap())
            .collect_vec();
        assert_eq!(res, vdata);
    }

    #[test]
    fn test_iter_str_sliced() {
        let vdata = (0..1000).map(|i| format!("string{}", i)).collect_vec();
        let data = StringArray::from(vdata.clone());
        let data = data.slice(100, 200);
        let ifh =
            Arc::<IterFuncHolder>::compile(&(&data as &dyn Array), (PrimitiveType::P64x2, false))
                .unwrap();
        let res = ArrowIter::<&[u8]>::new(&data, ifh)
            .unwrap()
            .map(|x| String::from_utf8(x.to_vec()).unwrap())
            .collect_vec();
        assert_eq!(res, vdata[100..300]);
    }

    #[test]
    fn test_iter_str_sliced_null() {
        let vdata = (0..1000)
            .map(|i| {
                if i % 2 == 0 {
                    Some(format!("string{}", i))
                } else {
                    None
                }
            })
            .collect_vec();
        let data = StringArray::from(vdata.clone());
        let data = data.slice(100, 10);
        let ifh =
            Arc::<IterFuncHolder>::compile(&(&data as &dyn Array), (PrimitiveType::P64x2, false))
                .unwrap();
        let res = ArrowIter::<&[u8]>::new(&data, ifh)
            .unwrap()
            .map(|x| String::from_utf8(x.to_vec()).unwrap())
            .collect_vec();
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn test_iter_ree() {
        let arr = RunArray::<Int64Type>::try_new(
            &Int64Array::from(vec![5, 10, 15, 20]),
            &UInt32Array::from(vec![Some(1), Some(2), None, Some(4)]),
        )
        .unwrap();

        let ifh =
            Arc::<IterFuncHolder>::compile(&(&arr as &dyn Array), (PrimitiveType::U64, false))
                .unwrap();
        let res = ArrowIter::<u64>::new(&arr, ifh).unwrap().collect_vec();
        assert_eq!(res.len(), 15);
    }

    #[test]
    fn test_iter_bool() {
        let arr = BooleanArray::from(vec![true, false, true, false, true]);
        let ifh = Arc::<IterFuncHolder>::compile(&(&arr as &dyn Array), (PrimitiveType::U8, false))
            .unwrap();
        let res = ArrowIter::<u8>::new(&arr, ifh).unwrap().collect_vec();
        assert_eq!(res, vec![1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_nullable_iter_ree() {
        let arr = RunArray::<Int64Type>::try_new(
            &Int64Array::from(vec![2, 4, 6, 8]),
            &UInt32Array::from(vec![Some(1), Some(2), None, Some(4)]),
        )
        .unwrap();

        let data_ifh =
            Arc::<IterFuncHolder>::compile(&(&arr as &dyn Array), (PrimitiveType::U64, true))
                .unwrap();
        let data_iter = ArrowIter::<u64>::new(&arr, data_ifh).unwrap();

        let ba = BooleanArray::new(arr.logical_nulls().unwrap().inner().clone(), None);
        let null_ifh =
            Arc::<IterFuncHolder>::compile(&(&ba as &dyn Array), (PrimitiveType::U8, true))
                .unwrap();
        let null_iter = ArrowIter::<u8>::new(&ba, null_ifh).unwrap();

        let res = ArrowNullableIter::new(data_iter, Some(null_iter)).collect_vec();
        assert_eq!(
            res,
            vec![
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                None,
                None,
                Some(4),
                Some(4)
            ]
        );
    }
}
