use std::{ffi::c_void, sync::Arc};

use arrow_array::{builder::BinaryBuilder, make_array, ArrayRef};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_data::ArrayData;
use inkwell::AddressSpace;
use repr_offset::ReprOffset;

use crate::{
    compiled_kernels::aggregate::{AggAlloc, AggType, Aggregation, StringSaver},
    increment_pointer, mark_load_invariant, pointer_diff, PrimitiveType,
};

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct PtrHolder {
    data_ptr: *const c_void,
    used_ptr: *const c_void,
    ss_ptr: *const c_void,
}

#[repr(C)]
pub struct MostRecentAlloc {
    data: Vec<u8>,
    used: Vec<u8>,
    ptr_holder: Box<PtrHolder>,
    ss: Box<StringSaver>,
    width: usize,
}

impl MostRecentAlloc {
    pub fn new(pt: PrimitiveType) -> Self {
        let width = pt.width();
        let ss = Box::new(StringSaver::default());
        MostRecentAlloc {
            data: Vec::new(),
            used: Vec::new(),
            ptr_holder: Box::new(PtrHolder {
                data_ptr: std::ptr::null(),
                used_ptr: std::ptr::null(),
                ss_ptr: (ss.as_ref() as *const StringSaver) as *const c_void,
            }),
            ss: ss,
            width,
        }
    }

    fn update_ptrs(&mut self) {
        self.ptr_holder.data_ptr = self.data.as_ptr() as *const c_void;
        self.ptr_holder.used_ptr = self.used.as_ptr() as *const c_void;
    }
}

impl AggAlloc for MostRecentAlloc {
    fn get_ptr(&self) -> *const std::ffi::c_void {
        self.ptr_holder.as_ref() as *const PtrHolder as *const std::ffi::c_void
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        if self.data.len() < capacity * self.width {
            self.data.resize(capacity * self.width, 0);
            self.used.resize(capacity, 0);
            self.update_ptrs();
        }
    }

    fn current_capacity(&self) -> usize {
        self.data.len() / self.width
    }

    fn preallocate_capacity(&mut self, expected_unique: usize) {
        self.data.reserve(expected_unique * self.width);
        self.used.reserve(expected_unique);
        self.update_ptrs();
    }
}

pub struct MostRecentAgg {
    ptype: PrimitiveType,
}

impl Aggregation for MostRecentAgg {
    type Allocation = MostRecentAlloc;

    type Output = ArrayRef;

    fn new(pts: &[crate::PrimitiveType]) -> Self {
        assert_eq!(pts.len(), 1, "most recent agg only supports one input type");
        MostRecentAgg { ptype: pts[0] }
    }

    fn allocate(&self, num_tickets: usize) -> Self::Allocation {
        let mut alloc = MostRecentAlloc::new(self.ptype);
        alloc.ensure_capacity(num_tickets);
        alloc
    }

    fn ptype(&self) -> crate::PrimitiveType {
        self.ptype
    }

    fn merge_allocs(
        &self,
        mut alloc1: Self::Allocation,
        alloc2: Self::Allocation,
    ) -> Self::Allocation {
        for idx in 0..alloc1.used.len() {
            if alloc1.used[idx] > 0 {
                continue;
            }

            if alloc2.used[idx] > 0 {
                alloc1.used[idx] = 1;
                alloc1.data[idx * self.ptype.width()..(idx + 1) * self.ptype.width()]
                    .copy_from_slice(
                        &alloc2.data[idx * self.ptype.width()..(idx + 1) * self.ptype.width()],
                    );
            }
        }

        alloc1
    }

    fn agg_type() -> AggType {
        AggType::MostRecent
    }

    fn llvm_agg_one<'a>(
        &self,
        ctx: &'a inkwell::context::Context,
        llvm_mod: &inkwell::module::Module<'a>,
        b: &inkwell::builder::Builder<'a>,
        alloc_ptr: inkwell::values::PointerValue<'a>,
        ticket: inkwell::values::IntValue<'a>,
        value: inkwell::values::BasicValueEnum<'a>,
    ) {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();
        let i128_type = ctx.i128_type();

        let used_ptr = b
            .build_load(
                ptr_type,
                increment_pointer!(ctx, b, alloc_ptr, PtrHolder::OFFSET_USED_PTR),
                "used_ptr",
            )
            .unwrap()
            .into_pointer_value();
        mark_load_invariant!(ctx, used_ptr);

        let data_ptr = b
            .build_load(
                ptr_type,
                increment_pointer!(ctx, b, alloc_ptr, PtrHolder::OFFSET_DATA_PTR),
                "used_ptr",
            )
            .unwrap()
            .into_pointer_value();
        mark_load_invariant!(ctx, data_ptr);

        // mark this slot as used
        b.build_store(
            increment_pointer!(ctx, b, used_ptr, 1, ticket),
            ctx.i8_type().const_int(1, false),
        )
        .unwrap();

        // store the value
        match self.ptype {
            PrimitiveType::P64x2 => {
                let value = value.into_struct_value();
                let ss_ptr = b
                    .build_load(
                        ptr_type,
                        increment_pointer!(ctx, b, alloc_ptr, PtrHolder::OFFSET_SS_PTR),
                        "ss_ptr",
                    )
                    .unwrap()
                    .into_pointer_value();
                let save_str = llvm_mod.add_function(
                    "save_to_string_saver",
                    i128_type.fn_type(
                        &[
                            ptr_type.into(),
                            i64_type.into(),
                            ptr_type.into(),
                            ptr_type.into(),
                        ],
                        false,
                    ),
                    None,
                );

                let ptr1 = b
                    .build_extract_value(value, 0, "ptr1")
                    .unwrap()
                    .into_pointer_value();
                let ptr2 = b
                    .build_extract_value(value, 1, "ptr2")
                    .unwrap()
                    .into_pointer_value();
                let len = pointer_diff!(ctx, b, ptr1, ptr2);
                let out_ptr = b.build_alloca(i128_type, "out_ptr").unwrap();
                b.build_call(
                    save_str,
                    &[ptr1.into(), len.into(), ss_ptr.into(), out_ptr.into()],
                    "save_str",
                )
                .unwrap();

                let out_val = b.build_load(i128_type, out_ptr, "out_val").unwrap();
                b.build_store(
                    increment_pointer!(ctx, b, data_ptr, self.ptype.width(), ticket),
                    out_val,
                )
                .unwrap();
            }
            _ => {
                b.build_store(
                    increment_pointer!(ctx, b, data_ptr, self.ptype.width(), ticket),
                    value,
                )
                .unwrap();
            }
        };
    }

    fn finalize(&self, alloc: Self::Allocation) -> Self::Output {
        let MostRecentAlloc {
            data,
            used,
            ptr_holder: _,
            ss,
            width: _,
        } = alloc;
        let null_map = NullBuffer::from_iter(used.iter().copied().map(|x| x > 0));

        match self.ptype {
            PrimitiveType::P64x2 => {
                let mut builder = BinaryBuilder::new();
                for chunk in data.chunks_exact(16) {
                    let ptr1 = u64::from_le_bytes(chunk[..8].try_into().unwrap()) as *const u8;
                    let ptr2 = u64::from_le_bytes(chunk[8..].try_into().unwrap()) as *const u8;

                    if ptr1.is_null() || ptr2.is_null() {
                        assert!(ptr1.is_null() && ptr2.is_null());
                        builder.append_null();
                        continue;
                    }

                    let slc = unsafe {
                        std::slice::from_raw_parts(ptr1, ptr2.byte_offset_from(ptr1) as usize)
                    };
                    builder.append_value(slc);
                }

                let arr = builder.finish();
                std::mem::drop(ss); // ensure StringSaver stays alive until at least here
                Arc::new(arr)
            }
            PrimitiveType::List(item_type, _) => {
                let item_type = PrimitiveType::from(item_type);
                let child_data = ArrayData::builder(item_type.as_arrow_type())
                    .len(data.len() / item_type.width())
                    .add_buffer(Buffer::from_vec(data))
                    .align_buffers(true)
                    .build()
                    .unwrap();
                let ad = ArrayData::builder(self.ptype.as_arrow_type())
                    .len(used.len())
                    .nulls(Some(null_map))
                    .add_child_data(child_data)
                    .build()
                    .unwrap();

                make_array(ad)
            }
            _ => {
                let buf = Buffer::from_vec(data);
                let ad = ArrayData::builder(self.ptype.as_arrow_type())
                    .len(buf.len() / self.ptype.width())
                    .add_buffer(buf)
                    .align_buffers(true)
                    .nulls(Some(null_map))
                    .build()
                    .unwrap();
                make_array(ad)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{ffi::c_void, sync::Arc};

    use arrow_array::{
        cast::AsArray,
        types::{Float32Type, Int32Type},
        Array, FixedSizeListArray, Float32Array, Int32Array, StringArray,
    };
    use arrow_data::ArrayData;
    use arrow_schema::{DataType, Field};
    use inkwell::{context::Context, OptimizationLevel};
    use itertools::Itertools;

    use crate::{
        compiled_iter::{datum_to_iter, generate_next},
        compiled_kernels::{
            aggregate::{most_recent::MostRecentAgg, AggAlloc, Aggregation},
            link_req_helpers,
        },
        ListItemType, PrimitiveType,
    };

    #[test]
    fn test_most_recent_agg() {
        let tickets: Vec<u64> = vec![0, 1, 0, 1, 0, 1];
        let data = Int32Array::from(vec![5, 6, 7, 8, 1, 2]);

        let agg = MostRecentAgg::new(&[PrimitiveType::I32]);
        let mut alloc = agg.allocate(2);

        let ctx = Context::create();
        let llvm_mod = ctx.create_module("most_recent_agg");
        let mut ih = datum_to_iter(&data).unwrap();
        let next_func = generate_next(&ctx, &llvm_mod, "next", &DataType::Int32, &ih).unwrap();
        let func = agg
            .llvm_agg_func(&ctx, &llvm_mod, next_func, false)
            .unwrap();

        llvm_mod.verify().unwrap();
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&llvm_mod, &ee).unwrap();

        let agg_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void)>(
                func.get_name().to_str().unwrap(),
            )
            .unwrap()
        };

        unsafe {
            agg_func.call(
                alloc.get_mut_ptr(),
                tickets.as_ptr() as *const c_void,
                ih.get_mut_ptr(),
            );
        }

        let res = agg.finalize(alloc);
        let res = res.as_primitive::<Int32Type>();
        let res = res.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, vec![1, 2]);
    }

    #[test]
    fn test_most_recent_fixed_size_list_f32() {
        let tickets: Vec<u64> = vec![0, 1, 0, 1];
        let values = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        let array_data = ArrayData::builder(DataType::FixedSizeList(field, 2))
            .len(4)
            .add_child_data(values.into_data())
            .build()
            .unwrap();
        let data = FixedSizeListArray::from(array_data);

        let agg = MostRecentAgg::new(&[PrimitiveType::List(ListItemType::F32, 2)]);
        let mut alloc = agg.allocate(2);

        let ctx = Context::create();
        let llvm_mod = ctx.create_module("most_recent_fsl_agg");
        let mut ih = datum_to_iter(&data).unwrap();
        let next_func = generate_next(&ctx, &llvm_mod, "next", data.data_type(), &ih).unwrap();
        let func = agg
            .llvm_agg_func(&ctx, &llvm_mod, next_func, false)
            .unwrap();

        llvm_mod.verify().unwrap();
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&llvm_mod, &ee).unwrap();

        let agg_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void)>(
                func.get_name().to_str().unwrap(),
            )
            .unwrap()
        };

        unsafe {
            agg_func.call(
                alloc.get_mut_ptr(),
                tickets.as_ptr() as *const c_void,
                ih.get_mut_ptr(),
            );
        }

        let res = agg.finalize(alloc);
        let res = res.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res.value_length(), 2);

        let values = res
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .copied()
            .collect_vec();
        assert_eq!(values, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_most_recent_string() {
        let tickets: Vec<u64> = vec![0, 1, 0, 1, 0, 1];
        let data = StringArray::from(vec!["a", "b", "c", "d", "e", "f"]);

        let agg = MostRecentAgg::new(&[PrimitiveType::P64x2]);
        let mut alloc = agg.allocate(2);

        let ctx = Context::create();
        let llvm_mod = ctx.create_module("most_recent_string_agg");
        let mut ih = datum_to_iter(&data).unwrap();
        let next_func = generate_next(&ctx, &llvm_mod, "next", &DataType::Utf8, &ih).unwrap();
        let func = agg
            .llvm_agg_func(&ctx, &llvm_mod, next_func, false)
            .unwrap();

        llvm_mod.verify().unwrap();
        let ee = llvm_mod
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&llvm_mod, &ee).unwrap();

        let agg_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void)>(
                func.get_name().to_str().unwrap(),
            )
            .unwrap()
        };

        unsafe {
            agg_func.call(
                alloc.get_mut_ptr(),
                tickets.as_ptr() as *const c_void,
                ih.get_mut_ptr(),
            );
        }

        let res = agg.finalize(alloc);
        let res = res.as_binary::<i32>();
        let res = res.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, vec![b"e".as_slice(), b"f".as_slice()]);
    }
}
