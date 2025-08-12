use std::{ffi::c_void, sync::Arc};

use arrow_array::{
    builder::BinaryViewBuilder, make_array, ArrayRef, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow_schema::DataType;
use bytemuck::Pod;
use inkwell::{
    context::Context,
    module::{Linkage, Module},
    values::BasicValue,
    AddressSpace, IntPredicate,
};

use repr_offset::ReprOffset;

use crate::{
    compiled_kernels::{
        aggregate::{AggAlloc, AggType, Aggregation, StringSaver},
        cmp::{add_float_to_int, add_memcmp},
    },
    declare_blocks, increment_pointer, mark_load_invariant, pointer_diff, PrimitiveType,
};

#[repr(C)]
#[derive(ReprOffset, Debug, Default)]
#[roff(usize_offsets)]
pub struct COption<T: Copy> {
    used: u8,
    value: T,
}

#[repr(C)]
#[derive(ReprOffset)]
#[roff(usize_offsets)]
pub struct StringAlloc {
    data_ptr: *mut c_void,
    saver_ptr: *mut StringSaver,
}
unsafe impl Send for StringAlloc {}

pub enum MinMaxAlloc {
    W8(Vec<COption<u8>>),
    W16(Vec<COption<u16>>),
    W32(Vec<COption<u32>>),
    W64(Vec<COption<u64>>),
    W128(StringAlloc, Vec<u128>, Box<StringSaver>),
}

impl AggAlloc for MinMaxAlloc {
    fn get_mut_ptr(&mut self) -> *mut c_void {
        match self {
            MinMaxAlloc::W8(items) => items.as_mut_ptr() as *mut c_void,
            MinMaxAlloc::W16(items) => items.as_mut_ptr() as *mut c_void,
            MinMaxAlloc::W32(items) => items.as_mut_ptr() as *mut c_void,
            MinMaxAlloc::W64(items) => items.as_mut_ptr() as *mut c_void,
            MinMaxAlloc::W128(b, _, _) => b as *mut StringAlloc as *mut c_void,
        }
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        match self {
            MinMaxAlloc::W8(items) => {
                items.resize_with(capacity, Default::default);
            }
            MinMaxAlloc::W16(items) => {
                items.resize_with(capacity, Default::default);
            }
            MinMaxAlloc::W32(items) => {
                items.resize_with(capacity, Default::default);
            }
            MinMaxAlloc::W64(items) => {
                items.resize_with(capacity, Default::default);
            }
            MinMaxAlloc::W128(b, v, _ss) => {
                v.resize_with(capacity, Default::default);
                b.data_ptr = v.as_mut_ptr() as *mut c_void;
            }
        }
    }

    fn current_capacity(&self) -> usize {
        match self {
            MinMaxAlloc::W8(coptions) => coptions.len(),
            MinMaxAlloc::W16(coptions) => coptions.len(),
            MinMaxAlloc::W32(coptions) => coptions.len(),
            MinMaxAlloc::W64(coptions) => coptions.len(),
            MinMaxAlloc::W128(_, items, _) => items.len(),
        }
    }
}

impl MinMaxAlloc {
    fn finalize(self) -> ArrayRef {
        match self {
            MinMaxAlloc::W8(coptions) => {
                Arc::new(UInt8Array::from_iter(coptions.into_iter().map(|x| {
                    if x.used != 0 {
                        Some(x.value)
                    } else {
                        None
                    }
                })))
            }
            MinMaxAlloc::W16(coptions) => {
                Arc::new(UInt16Array::from_iter(coptions.into_iter().map(|x| {
                    if x.used != 0 {
                        Some(x.value)
                    } else {
                        None
                    }
                })))
            }
            MinMaxAlloc::W32(coptions) => {
                Arc::new(UInt32Array::from_iter(coptions.into_iter().map(|x| {
                    if x.used != 0 {
                        Some(x.value)
                    } else {
                        None
                    }
                })))
            }
            MinMaxAlloc::W64(coptions) => {
                Arc::new(UInt64Array::from_iter(coptions.into_iter().map(|x| {
                    if x.used != 0 {
                        Some(x.value)
                    } else {
                        None
                    }
                })))
            }
            MinMaxAlloc::W128(_string_alloc, items, _string_saver) => {
                let mut b = BinaryViewBuilder::with_capacity(items.len());
                for ptrs in items {
                    if ptrs == 0 {
                        b.append_null();
                    } else {
                        let start_ptr = (ptrs as u64) as *const u8;
                        let end_ptr = ((ptrs >> 64) as u64) as *const u8;
                        unsafe {
                            let len = end_ptr.offset_from(start_ptr);
                            let slc = std::slice::from_raw_parts(start_ptr, len as usize);
                            b.append_value(slc);
                        }
                    }
                }

                Arc::new(b.finish())
            }
        }
    }
}

pub struct MinMaxAgg<const MIN: bool> {
    pt: PrimitiveType,
}

impl<const MIN: bool> Aggregation for MinMaxAgg<MIN> {
    type Allocation = MinMaxAlloc;

    type Output = ArrayRef;

    fn new(pts: &[PrimitiveType]) -> Self {
        assert_eq!(
            pts.len(),
            1,
            "min/max aggregation takes exactly one input type"
        );
        Self { pt: pts[0] }
    }

    fn allocate(&self, num_tickets: usize) -> Self::Allocation {
        let mut alloc = match self.pt {
            PrimitiveType::U8 | PrimitiveType::I8 => MinMaxAlloc::W8(vec![]),
            PrimitiveType::F16 | PrimitiveType::U16 | PrimitiveType::I16 => {
                MinMaxAlloc::W16(vec![])
            }
            PrimitiveType::F32 | PrimitiveType::U32 | PrimitiveType::I32 => {
                MinMaxAlloc::W32(vec![])
            }
            PrimitiveType::F64 | PrimitiveType::U64 | PrimitiveType::I64 => {
                MinMaxAlloc::W64(vec![])
            }
            PrimitiveType::P64x2 => {
                let mut data = vec![];
                let mut ss = Box::new(StringSaver::default());
                let sa = StringAlloc {
                    data_ptr: data.as_mut_ptr() as *mut c_void,
                    saver_ptr: ss.as_mut() as *mut StringSaver,
                };
                MinMaxAlloc::W128(sa, data, ss)
            }
        };
        alloc.ensure_capacity(num_tickets);
        alloc
    }

    fn ptype(&self) -> PrimitiveType {
        self.pt
    }

    fn agg_type() -> AggType {
        if MIN {
            AggType::Min
        } else {
            AggType::Max
        }
    }

    fn merge_allocs(&self, alloc1: Self::Allocation, alloc2: Self::Allocation) -> Self::Allocation {
        match (alloc1, alloc2) {
            (MinMaxAlloc::W8(mut lhs), MinMaxAlloc::W8(rhs)) => {
                match self.pt {
                    PrimitiveType::I8 => merge_coptions::<u8, i8, MIN>(&mut lhs, rhs),
                    PrimitiveType::U8 => merge_coptions::<u8, u8, MIN>(&mut lhs, rhs),
                    _ => unreachable!(),
                }
                MinMaxAlloc::W8(lhs)
            }
            (MinMaxAlloc::W16(mut lhs), MinMaxAlloc::W16(rhs)) => {
                match self.pt {
                    PrimitiveType::I16 => merge_coptions::<u16, i16, MIN>(&mut lhs, rhs),
                    PrimitiveType::U16 => merge_coptions::<u16, u16, MIN>(&mut lhs, rhs),
                    PrimitiveType::F16 => merge_coptions::<u16, half::f16, MIN>(&mut lhs, rhs),
                    _ => unreachable!(),
                }
                MinMaxAlloc::W16(lhs)
            }
            (MinMaxAlloc::W32(mut lhs), MinMaxAlloc::W32(rhs)) => {
                match self.pt {
                    PrimitiveType::I32 => merge_coptions::<u32, i32, MIN>(&mut lhs, rhs),
                    PrimitiveType::U32 => merge_coptions::<u32, u32, MIN>(&mut lhs, rhs),
                    PrimitiveType::F32 => merge_coptions::<u32, f32, MIN>(&mut lhs, rhs),
                    _ => unreachable!(),
                }
                MinMaxAlloc::W32(lhs)
            }
            (MinMaxAlloc::W64(mut lhs), MinMaxAlloc::W64(rhs)) => {
                match self.pt {
                    PrimitiveType::I64 => merge_coptions::<u64, i64, MIN>(&mut lhs, rhs),
                    PrimitiveType::U64 => merge_coptions::<u64, u64, MIN>(&mut lhs, rhs),
                    PrimitiveType::F64 => merge_coptions::<u64, f64, MIN>(&mut lhs, rhs),
                    _ => unreachable!(),
                }
                MinMaxAlloc::W64(lhs)
            }
            (MinMaxAlloc::W128(_, lhs_i, _lhs_ss), MinMaxAlloc::W128(_, rhs_i, _rhs_ss)) => {
                let mut new_ss = StringSaver::default();
                let mut v = Vec::new();
                for (lhs, rhs) in lhs_i.into_iter().zip(rhs_i.into_iter()) {
                    let lhs_start_ptr = lhs as u64 as *const u8;
                    let rhs_start_ptr = rhs as u64 as *const u8;
                    let lhs_end_ptr = (lhs >> 64) as u64 as *const u8;
                    let rhs_end_ptr = (rhs >> 64) as u64 as *const u8;

                    match (lhs, rhs) {
                        (0, 0) => v.push(0),
                        (0, _) => unsafe {
                            let rhs_len = rhs_end_ptr.offset_from(rhs_start_ptr);
                            let rhs_slice =
                                std::slice::from_raw_parts(rhs_start_ptr, rhs_len as usize);
                            v.push(new_ss.insert(rhs_slice));
                        },
                        (_, 0) => unsafe {
                            let lhs_len = lhs_end_ptr.offset_from(lhs_start_ptr);
                            let lhs_slice =
                                std::slice::from_raw_parts(lhs_start_ptr, lhs_len as usize);
                            v.push(new_ss.insert(lhs_slice));
                        },
                        (_, _) => unsafe {
                            let lhs_len = lhs_end_ptr.offset_from(lhs_start_ptr);
                            let lhs_slice =
                                std::slice::from_raw_parts(lhs_start_ptr, lhs_len as usize);
                            let rhs_len = rhs_end_ptr.offset_from(rhs_start_ptr);
                            let rhs_slice =
                                std::slice::from_raw_parts(rhs_start_ptr, rhs_len as usize);
                            let res = if MIN {
                                lhs_slice.min(rhs_slice)
                            } else {
                                lhs_slice.max(rhs_slice)
                            };
                            v.push(new_ss.insert(res));
                        },
                    }
                }
                let mut ss = Box::new(new_ss);
                let sa = StringAlloc {
                    data_ptr: v.as_mut_ptr() as *mut c_void,
                    saver_ptr: ss.as_mut() as *mut StringSaver,
                };
                MinMaxAlloc::W128(sa, v, ss)
            }
            _ => unreachable!("cannot merge different types of minmax allocs"),
        }
    }

    fn finalize(&self, alloc: Self::Allocation) -> Self::Output {
        let res = alloc.finalize();
        match self.pt {
            PrimitiveType::I8 => make_array(
                res.to_data()
                    .into_builder()
                    .data_type(DataType::Int8)
                    .build()
                    .unwrap(),
            ),
            PrimitiveType::I16 => make_array(
                res.to_data()
                    .into_builder()
                    .data_type(DataType::Int16)
                    .build()
                    .unwrap(),
            ),
            PrimitiveType::I32 => make_array(
                res.to_data()
                    .into_builder()
                    .data_type(DataType::Int32)
                    .build()
                    .unwrap(),
            ),
            PrimitiveType::I64 => make_array(
                res.to_data()
                    .into_builder()
                    .data_type(DataType::Int64)
                    .build()
                    .unwrap(),
            ),
            PrimitiveType::F16 => make_array(
                res.to_data()
                    .into_builder()
                    .data_type(DataType::Float16)
                    .build()
                    .unwrap(),
            ),
            PrimitiveType::F32 => make_array(
                res.to_data()
                    .into_builder()
                    .data_type(DataType::Float32)
                    .build()
                    .unwrap(),
            ),
            PrimitiveType::F64 => make_array(
                res.to_data()
                    .into_builder()
                    .data_type(DataType::Float64)
                    .build()
                    .unwrap(),
            ),
            _ => res,
        }
    }

    fn llvm_agg_one<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        b: &inkwell::builder::Builder<'a>,
        alloc_ptr: inkwell::values::PointerValue<'a>,
        ticket: inkwell::values::IntValue<'a>,
        value: inkwell::values::BasicValueEnum<'a>,
    ) {
        let holder_width = match self.pt.width() {
            1 => std::mem::size_of::<COption<u8>>(),
            2 => std::mem::size_of::<COption<u16>>(),
            4 => std::mem::size_of::<COption<u32>>(),
            8 => std::mem::size_of::<COption<u64>>(),
            16 => 16,
            _ => unreachable!(),
        };
        let llvm_type = self.pt.llvm_type(ctx);
        let i8_type = ctx.i8_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();
        if self.pt == PrimitiveType::P64x2 {
            let do_str = {
                let i128_type = ctx.i128_type();
                let memcmp = add_memcmp(ctx, llvm_mod);
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
                let fn_type = ctx.void_type().fn_type(
                    &[
                        ptr_type.into(),  // ptr to holder
                        ptr_type.into(),  // string saver ptr
                        llvm_type.into(), // value to agg
                    ],
                    false,
                );
                let func = llvm_mod.add_function("do_str", fn_type, Some(Linkage::Private));
                declare_blocks!(ctx, func, entry, save_value, check_value, exit);
                let b = ctx.create_builder();
                b.position_at_end(entry);
                let holder_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
                let ss_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
                let value = func.get_nth_param(2).unwrap().into_struct_value();
                let cur_val = b
                    .build_load(i128_type, holder_ptr, "cur_val")
                    .unwrap()
                    .into_int_value();
                let cmp = b
                    .build_int_compare(IntPredicate::EQ, cur_val, i128_type.const_zero(), "is_null")
                    .unwrap();
                b.build_conditional_branch(cmp, save_value, check_value)
                    .unwrap();

                b.position_at_end(check_value);
                let cur_val = b
                    .build_load(llvm_type, holder_ptr, "cur_val")
                    .unwrap()
                    .into_struct_value();
                let res = b
                    .build_call(memcmp, &[cur_val.into(), value.into()], "cmp")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let cmp = b
                    .build_int_compare(
                        if MIN {
                            IntPredicate::SGT
                        } else {
                            IntPredicate::SLT
                        },
                        res,
                        i64_type.const_zero(),
                        "cmp",
                    )
                    .unwrap();
                b.build_conditional_branch(cmp, save_value, exit).unwrap();

                b.position_at_end(save_value);
                let out_ptr = b.build_alloca(llvm_type, "out_ptr").unwrap();

                let ptr1 = b
                    .build_extract_value(value, 0, "ptr1")
                    .unwrap()
                    .into_pointer_value();
                let ptr2 = b
                    .build_extract_value(value, 1, "ptr2")
                    .unwrap()
                    .into_pointer_value();
                let len = pointer_diff!(ctx, b, ptr1, ptr2);
                b.build_call(
                    save_str,
                    &[ptr1.into(), len.into(), ss_ptr.into(), out_ptr.into()],
                    "save_str",
                )
                .unwrap();
                let value = b.build_load(llvm_type, out_ptr, "value").unwrap();
                b.build_store(holder_ptr, value).unwrap();
                b.build_return(None).unwrap();

                b.position_at_end(exit);
                b.build_return(None).unwrap();

                func
            };
            let ptr_to_holders = b
                .build_load(
                    ptr_type,
                    increment_pointer!(ctx, b, alloc_ptr, StringAlloc::OFFSET_DATA_PTR),
                    "holders_ptr",
                )
                .unwrap()
                .into_pointer_value();
            mark_load_invariant!(ctx, ptr_to_holders);

            let holder_ptr = increment_pointer!(ctx, b, ptr_to_holders, holder_width, ticket);
            let ss_ptr = b
                .build_load(
                    ptr_type,
                    increment_pointer!(ctx, b, alloc_ptr, StringAlloc::OFFSET_SAVER_PTR),
                    "ss_ptr",
                )
                .unwrap()
                .into_pointer_value();
            mark_load_invariant!(ctx, ss_ptr);

            b.build_call(
                do_str,
                &[holder_ptr.into(), ss_ptr.into(), value.into()],
                "do_str",
            )
            .unwrap();
        } else {
            let holder_ptr = increment_pointer!(ctx, b, alloc_ptr, holder_width, ticket);
            let used_offset = match self.pt.width() {
                1 => COption::<u8>::OFFSET_USED,
                2 => COption::<u16>::OFFSET_USED,
                4 => COption::<u32>::OFFSET_USED,
                8 => COption::<u64>::OFFSET_USED,
                _ => unreachable!(),
            };
            let value_offset = match self.pt.width() {
                1 => COption::<u8>::OFFSET_VALUE,
                2 => COption::<u16>::OFFSET_VALUE,
                4 => COption::<u32>::OFFSET_VALUE,
                8 => COption::<u64>::OFFSET_VALUE,
                _ => unreachable!(),
            };

            let cur_val = b
                .build_load(
                    llvm_type,
                    increment_pointer!(ctx, b, holder_ptr, value_offset),
                    "cur_val",
                )
                .unwrap();

            let cmp = match self.pt {
                PrimitiveType::I8
                | PrimitiveType::I16
                | PrimitiveType::I32
                | PrimitiveType::I64 => b
                    .build_int_compare(
                        if MIN {
                            IntPredicate::SLT
                        } else {
                            IntPredicate::SGT
                        },
                        value.into_int_value(),
                        cur_val.into_int_value(),
                        "cmp",
                    )
                    .unwrap(),
                PrimitiveType::U8
                | PrimitiveType::U16
                | PrimitiveType::U32
                | PrimitiveType::U64 => b
                    .build_int_compare(
                        if MIN {
                            IntPredicate::ULT
                        } else {
                            IntPredicate::UGT
                        },
                        value.into_int_value(),
                        cur_val.into_int_value(),
                        "cmp",
                    )
                    .unwrap(),
                PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
                    let ftoi = add_float_to_int(ctx, llvm_mod, self.pt);
                    let tmp1 = b
                        .build_call(ftoi, &[value.into()], "val_to_i")
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value();
                    let tmp2 = b
                        .build_call(ftoi, &[cur_val.into()], "cur_to_i")
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value();
                    b.build_int_compare(
                        if MIN {
                            IntPredicate::SLT
                        } else {
                            IntPredicate::SGT
                        },
                        tmp1,
                        tmp2,
                        "cmp",
                    )
                    .unwrap()
                }
                PrimitiveType::P64x2 => unreachable!(),
            };

            let new_val_assume_used = b
                .build_select(cmp, value, cur_val, "new_val_assume_used")
                .unwrap();

            let is_used = b
                .build_load(
                    i8_type,
                    increment_pointer!(ctx, b, holder_ptr, used_offset),
                    "is_used",
                )
                .unwrap()
                .into_int_value();
            let is_used = b
                .build_int_compare(IntPredicate::NE, is_used, i8_type.const_zero(), "is_used")
                .unwrap();
            let new_val = b
                .build_select(is_used, new_val_assume_used, value, "new_value")
                .unwrap();
            b.build_store(
                increment_pointer!(ctx, b, holder_ptr, value_offset),
                new_val,
            )
            .unwrap();
            b.build_store(
                increment_pointer!(ctx, b, holder_ptr, used_offset),
                i8_type.const_int(1, false),
            )
            .unwrap();
        }
    }
}

fn merge_coptions<T: Copy + Pod, V: PartialOrd + Pod, const MIN: bool>(
    v1: &mut [COption<T>],
    v2: Vec<COption<T>>,
) {
    for (lhs, rhs) in v1.iter_mut().zip(v2.into_iter()) {
        if rhs.used == 0 {
            continue;
        }
        if lhs.used == 0 {
            lhs.value = rhs.value;
        }
        let lhs_v = bytemuck::cast::<T, V>(lhs.value);
        let rhs_v = bytemuck::cast::<T, V>(rhs.value);
        if (MIN && lhs_v > rhs_v) || ((!MIN) && lhs_v < rhs_v) {
            lhs.value = rhs.value;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{cast::AsArray, StringArray};
    use arrow_schema::DataType;
    use inkwell::{context::Context, OptimizationLevel};
    use itertools::Itertools;

    use crate::{
        compiled_iter::{datum_to_iter, generate_next},
        compiled_kernels::{
            aggregate::{minmax::MinMaxAgg, AggAlloc, Aggregation},
            link_req_helpers,
        },
        PrimitiveType,
    };

    #[test]
    fn test_min_str_agg() {
        let tickets: Vec<u64> = vec![0, 1, 0, 1, 0, 1];
        let data = StringArray::from(vec!["a", "c", "b", "b", "c", "a"]);

        let agg = MinMaxAgg::<true>::new(&[PrimitiveType::P64x2]);
        let mut alloc = agg.allocate(2);

        let ctx = Context::create();
        let llvm_mod = ctx.create_module("min_str_agg");
        let mut ih = datum_to_iter(&data).unwrap();
        let next_func = generate_next(&ctx, &llvm_mod, "next", &DataType::Utf8, &ih).unwrap();
        let func = agg.llvm_agg_func(&ctx, &llvm_mod, next_func);

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
        let res = res.as_binary_view();
        let res = res.iter().map(|x| x.unwrap()).collect_vec();
        assert_eq!(res, vec![b"a", b"a"]);
    }
}
