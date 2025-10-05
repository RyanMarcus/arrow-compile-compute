use arrow_array::{Datum, UInt64Array};
use arrow_schema::DataType;
use inkwell::attributes::AttributeLoc;
use inkwell::execution_engine::JitFunction;
use inkwell::values::BasicValue;
use inkwell::OptimizationLevel;
use inkwell::{
    builder::Builder,
    context::Context,
    intrinsics::Intrinsic,
    module::Module,
    values::{FunctionValue, IntValue, PointerValue},
    AddressSpace, IntPredicate,
};
use ouroboros::self_referencing;
use repr_offset::ReprOffset;
use std::ffi::c_void;

use crate::compiled_iter::{datum_to_iter, generate_next};
use crate::compiled_kernels::writers::{ArrayWriter, PrimitiveArrayWriter, WriterAllocation};
use crate::compiled_kernels::{link_req_helpers, optimize_module};
use crate::set_noalias_params;
use crate::{
    compiled_kernels::cmp::add_memcmp, declare_blocks, increment_pointer, pointer_diff,
    PrimitiveType,
};

use super::{ArrowKernelError, Kernel};

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct TicketTable {
    max_size: usize,
    keys: *mut u8,
    tickets: *mut u8,

    last_val: u64,
    keys_data: Box<[u8]>,
    pub tickets_data: Box<[u8]>,
    key_type: DataType,
    ticket_type: DataType,
}

impl TicketTable {
    pub fn new(max_size: usize, value_type: DataType, ticket_type: DataType) -> Self {
        let mut key_data = vec![0; PrimitiveType::for_arrow_type(&value_type).width() * max_size]
            .into_boxed_slice();
        let mut ticket_data =
            vec![0; PrimitiveType::for_arrow_type(&ticket_type).width() * max_size]
                .into_boxed_slice();
        Self {
            max_size,
            keys: key_data.as_mut_ptr(),
            tickets: ticket_data.as_mut_ptr(),
            keys_data: key_data,
            tickets_data: ticket_data,
            last_val: 0,
            key_type: value_type,
            ticket_type,
        }
    }

    /// Returns the number of elements in the table (not the maximum size of the
    /// table)
    pub fn len(&self) -> usize {
        self.last_val as usize
    }

    fn llvm_max_size<'a>(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let max = build
            .build_load(
                ctx.i64_type(),
                increment_pointer!(ctx, build, ptr, TicketTable::OFFSET_MAX_SIZE),
                "max_size",
            )
            .unwrap()
            .into_int_value();
        max.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        max
    }

    fn llvm_keys_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let keys = build
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                increment_pointer!(ctx, build, ptr, TicketTable::OFFSET_KEYS),
                "keys_ptr",
            )
            .unwrap()
            .into_pointer_value();
        keys.as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        keys
    }

    fn llvm_tickets_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        let tickets = build
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                increment_pointer!(ctx, build, ptr, TicketTable::OFFSET_TICKETS),
                "tickets_ptr",
            )
            .unwrap()
            .into_pointer_value();
        tickets
            .as_instruction_value()
            .unwrap()
            .set_metadata(ctx.metadata_node(&[]), ctx.get_kind_id("invariant.load"))
            .unwrap();
        tickets
    }

    fn llvm_last_val_ptr<'a>(
        &self,
        ctx: &'a Context,
        build: &Builder<'a>,
        ptr: PointerValue<'a>,
    ) -> PointerValue<'a> {
        increment_pointer!(ctx, build, ptr, TicketTable::OFFSET_LAST_VAL)
    }
}

/// Builds a function that uses a ticket table to assign each value a unique
/// ticket. The last/output parameter is `0` if an old value was read from the
/// table, `1` if a new value was inserted, and `2` if the table is full.
pub fn generate_lookup_or_insert<'a>(
    ctx: &'a Context,
    module: &Module<'a>,
    ht: &TicketTable,
) -> FunctionValue<'a> {
    let key_prim_type = PrimitiveType::for_arrow_type(&ht.key_type);
    let ticket_prim_type = PrimitiveType::for_arrow_type(&ht.ticket_type);

    let ptr_type = ctx.ptr_type(AddressSpace::default());
    let i64_type = ctx.i64_type();
    let i8_type = ctx.i8_type();
    let key_type = key_prim_type.llvm_type(ctx);
    let ticket_type = ticket_prim_type.llvm_type(ctx).into_int_type();

    let func = module.add_function(
        "lookup_or_insert",
        ticket_type.fn_type(
            &[
                ptr_type.into(), // pointer to the hash table
                key_type.into(),
                i64_type.into(), // hash value of the key
                ptr_type.into(), // output ptr to bool, true if new value was inserted
            ],
            false,
        ),
        None,
    );
    let build = ctx.create_builder();
    declare_blocks!(
        ctx,
        func,
        entry,
        loop_cond,
        table_full,
        loop_body,
        found_nonempty,
        found_empty,
        nonempty_match,
        nonempty_nomatch
    );
    build.position_at_end(entry);
    let ht_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
    let kvalue = func.get_nth_param(1).unwrap();
    let hvalue = func.get_nth_param(2).unwrap().into_int_value();
    let is_new_val_ptr = func.get_nth_param(3).unwrap().into_pointer_value();
    let table_size = ht.llvm_max_size(ctx, &build, ht_ptr);
    let start_pos = build
        .build_int_unsigned_rem(hvalue, table_size, "start_pos")
        .unwrap();
    let pos_ptr = build.build_alloca(i64_type, "pos_ptr").unwrap();
    build.build_store(pos_ptr, start_pos).unwrap();
    build.build_unconditional_branch(loop_body).unwrap();

    build.position_at_end(loop_cond);
    let pos = build
        .build_load(i64_type, pos_ptr, "pos")
        .unwrap()
        .into_int_value();
    let over_end = build
        .build_int_compare(IntPredicate::UGE, pos, table_size, "over_end")
        .unwrap();
    let new_pos = build
        .build_select(over_end, i64_type.const_zero(), pos, "new_pos")
        .unwrap()
        .into_int_value();
    let looped = build
        .build_int_compare(IntPredicate::EQ, new_pos, start_pos, "looped")
        .unwrap();
    build.build_store(pos_ptr, new_pos).unwrap();
    build
        .build_conditional_branch(looped, table_full, loop_body)
        .unwrap();

    build.position_at_end(table_full);
    build
        .build_store(is_new_val_ptr, i8_type.const_int(2, false))
        .unwrap();
    build.build_return(Some(&ticket_type.const_zero())).unwrap();

    build.position_at_end(loop_body);
    let pos = build
        .build_load(i64_type, pos_ptr, "pos")
        .unwrap()
        .into_int_value();
    let tickets = ht.llvm_tickets_ptr(ctx, &build, ht_ptr);
    let ticket_ptr = increment_pointer!(ctx, &build, tickets, ticket_prim_type.width(), pos);
    let curr_ticket = build
        .build_load(ticket_type, ticket_ptr, "curr_ticket")
        .unwrap()
        .into_int_value();
    let is_zero = build
        .build_int_compare(
            IntPredicate::EQ,
            curr_ticket,
            ticket_type.const_zero(),
            "is_zero",
        )
        .unwrap();
    build
        .build_conditional_branch(is_zero, found_empty, found_nonempty)
        .unwrap();

    // if we find an empty ticket, we should store our element here
    build.position_at_end(found_empty);
    let last_ticket_ptr = ht.llvm_last_val_ptr(ctx, &build, ht_ptr);
    let last_ticket = build
        .build_load(ticket_type, last_ticket_ptr, "last_ticket")
        .unwrap()
        .into_int_value();
    let my_ticket = build
        .build_int_add(last_ticket, ticket_type.const_int(1, false), "my_ticket")
        .unwrap();
    build.build_store(ticket_ptr, my_ticket).unwrap();
    build.build_store(last_ticket_ptr, my_ticket).unwrap();
    let keys_ptr = ht.llvm_keys_ptr(ctx, &build, ht_ptr);
    build
        .build_store(
            increment_pointer!(ctx, &build, keys_ptr, key_prim_type.width(), pos),
            kvalue,
        )
        .unwrap();
    // last_ticket instead of my_ticket because we want the value - 1
    build
        .build_store(is_new_val_ptr, i8_type.const_int(1, false))
        .unwrap();
    build.build_return(Some(&last_ticket)).unwrap();

    // if we found an occupied cell, check to see if our key matches
    build.position_at_end(found_nonempty);
    let keys_ptr = ht.llvm_keys_ptr(ctx, &build, ht_ptr);
    let key = build
        .build_load(
            key_type,
            increment_pointer!(ctx, &build, keys_ptr, key_prim_type.width(), pos),
            "key",
        )
        .unwrap();
    let matches = match key_prim_type {
        PrimitiveType::I16
        | PrimitiveType::I32
        | PrimitiveType::I64
        | PrimitiveType::U8
        | PrimitiveType::U16
        | PrimitiveType::U32
        | PrimitiveType::U64
        | PrimitiveType::I8 => {
            let key = key.into_int_value();
            build
                .build_int_compare(IntPredicate::EQ, key, kvalue.into_int_value(), "key_match")
                .unwrap()
        }
        PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
            let key = key.into_float_value();
            let int_type = PrimitiveType::int_with_width(key_prim_type.width()).llvm_type(ctx);
            let key = build
                .build_bit_cast(key, int_type, "key_as_int")
                .unwrap()
                .into_int_value();
            let kvalue = build
                .build_bit_cast(kvalue.into_float_value(), int_type, "param_as_int")
                .unwrap()
                .into_int_value();
            build
                .build_int_compare(IntPredicate::EQ, key, kvalue, "key_match")
                .unwrap()
        }
        PrimitiveType::P64x2 => {
            let memcmp = add_memcmp(ctx, module);
            let res = build
                .build_call(memcmp, &[key.into(), kvalue.into()], "memcmp")
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            build
                .build_int_compare(
                    IntPredicate::EQ,
                    res,
                    i64_type.const_int(0, false),
                    "key_match",
                )
                .unwrap()
        }
    };

    build
        .build_conditional_branch(matches, nonempty_match, nonempty_nomatch)
        .unwrap();

    build.position_at_end(nonempty_match);
    let ticket_m1 = build
        .build_int_sub(curr_ticket, ticket_type.const_int(1, false), "ticket_m1")
        .unwrap();
    build
        .build_store(is_new_val_ptr, i8_type.const_zero())
        .unwrap();
    build.build_return(Some(&ticket_m1)).unwrap();

    build.position_at_end(nonempty_nomatch);
    let next_pos = build
        .build_int_add(pos, i64_type.const_int(1, false), "next_pos")
        .unwrap();
    build.build_store(pos_ptr, next_pos).unwrap();
    build.build_unconditional_branch(loop_cond).unwrap();

    func
}

const MURMUR_C1: u64 = 0xff51afd7ed558ccd;
const MURMUR_C2: u64 = 0xc4ceb9fe1a85ec53;
const UPPER_MASK: u64 = 0x00000000FFFFFFFF;

/// Generate a murmur hash function.
///
/// The returned function takes a single parameter of type `pt` and returns an
/// `i64` hash value.
///
/// Strings are hashed by taking the first and last 32 bits.
pub fn generate_hash_func<'a>(
    ctx: &'a Context,
    llvm_mod: &Module<'a>,
    pt: PrimitiveType,
) -> FunctionValue<'a> {
    let i64_type = ctx.i64_type();
    let inp_type = pt.llvm_type(ctx);
    let ptr_type = ctx.ptr_type(AddressSpace::default());

    let func = llvm_mod.add_function("hash", i64_type.fn_type(&[inp_type.into()], false), None);

    let memcpy = Intrinsic::find("llvm.memcpy").unwrap();
    let memcpy_f = memcpy
        .get_declaration(
            llvm_mod,
            &[ptr_type.into(), ptr_type.into(), i64_type.into()],
        )
        .unwrap();

    declare_blocks!(ctx, func, entry, do_hash);
    let build = ctx.create_builder();
    build.position_at_end(entry);

    let v33 = i64_type.const_int(33, false);
    let v_upper = i64_type.const_int(UPPER_MASK, false);
    let vm_c1 = i64_type.const_int(MURMUR_C1, false);
    let vm_c2 = i64_type.const_int(MURMUR_C2, false);

    let buf_ptr = build.build_alloca(i64_type, "buf").unwrap();
    build.build_store(buf_ptr, i64_type.const_zero()).unwrap();
    match pt {
        PrimitiveType::P64x2 => {
            // extract first and last 4 chars
            let string = func.get_nth_param(0).unwrap().into_struct_value();
            let ptr1 = build
                .build_extract_value(string, 0, "ptr1")
                .unwrap()
                .into_pointer_value();
            let ptr2 = build
                .build_extract_value(string, 1, "ptr2")
                .unwrap()
                .into_pointer_value();

            // if the strlen < 8, use the whole string as a u64
            // otherwise, use the first and last 8 bytes
            let strlen = pointer_diff!(ctx, build, ptr1, ptr2);
            let cmp = build
                .build_int_compare(
                    IntPredicate::ULT,
                    strlen,
                    i64_type.const_int(8, false),
                    "cmp",
                )
                .unwrap();

            declare_blocks!(ctx, func, short_string, long_string);
            build
                .build_conditional_branch(cmp, short_string, long_string)
                .unwrap();

            build.position_at_end(short_string);

            build
                .build_call(
                    memcpy_f,
                    &[
                        buf_ptr.into(),
                        ptr1.into(),
                        strlen.into(),
                        ctx.bool_type().const_zero().into(),
                    ],
                    "memcpy_res",
                )
                .unwrap();
            build.build_unconditional_branch(do_hash).unwrap();

            build.position_at_end(long_string);
            let first_four = build
                .build_load(ctx.i32_type(), ptr1, "first_four")
                .unwrap()
                .into_int_value();

            let ptr2 = build.build_ptr_to_int(ptr2, i64_type, "ptr2").unwrap();
            let ptr2 = build
                .build_int_sub(ptr2, i64_type.const_int(4, false), "ptr2")
                .unwrap();
            let ptr2 = build.build_int_to_ptr(ptr2, ptr_type, "ptr2").unwrap();
            let last_four = build
                .build_load(ctx.i32_type(), ptr2, "last_four")
                .unwrap()
                .into_int_value();
            let first_four = build
                .build_int_z_extend(first_four, i64_type, "ex_first_four")
                .unwrap();
            let last_four = build
                .build_int_z_extend(last_four, i64_type, "ex_last_four")
                .unwrap();
            let last_four = build
                .build_left_shift(
                    last_four,
                    i64_type.const_int(32, false),
                    "shifted_last_four",
                )
                .unwrap();
            let combined = build.build_or(first_four, last_four, "combined").unwrap();

            build.build_store(buf_ptr, combined).unwrap();
            build.build_unconditional_branch(do_hash).unwrap();
        }
        PrimitiveType::I8
        | PrimitiveType::I16
        | PrimitiveType::I32
        | PrimitiveType::I64
        | PrimitiveType::U8
        | PrimitiveType::U16
        | PrimitiveType::U32
        | PrimitiveType::U64 => {
            let t = build
                .build_int_z_extend(
                    func.get_nth_param(0).unwrap().into_int_value(),
                    i64_type,
                    "t",
                )
                .unwrap();
            build.build_store(buf_ptr, t).unwrap();
            build.build_unconditional_branch(do_hash).unwrap();
        }
        PrimitiveType::F16 | PrimitiveType::F32 | PrimitiveType::F64 => {
            let t = build
                .build_float_cast(
                    func.get_nth_param(0).unwrap().into_float_value(),
                    ctx.f64_type(),
                    "t",
                )
                .unwrap();
            let t = build.build_bit_cast(t, i64_type, "t_int").unwrap();
            build.build_store(buf_ptr, t).unwrap();
            build.build_unconditional_branch(do_hash).unwrap();
        }
    };

    build.position_at_end(do_hash);
    let block = build
        .build_load(i64_type, buf_ptr, "block")
        .unwrap()
        .into_int_value();
    // v = v ^ (v >> 33);
    let block = build
        .build_xor(
            block,
            build.build_right_shift(block, v33, false, "shr").unwrap(),
            "xor",
        )
        .unwrap();

    // v = (v & UPPER_MASK).wrapping_mul(MURMUR_C1 & UPPER_MASK);
    let block = build
        .build_int_nuw_mul(
            build.build_and(block, v_upper, "and_outer").unwrap(),
            build.build_and(vm_c1, v_upper, "and_inner").unwrap(),
            "mul",
        )
        .unwrap();

    // v = v ^ (v >> 33);
    let block = build
        .build_xor(
            block,
            build.build_right_shift(block, v33, false, "shr").unwrap(),
            "xor",
        )
        .unwrap();

    // v = (v & UPPER_MASK).wrapping_mul(MURMUR_C2 & UPPER_MASK);
    let block = build
        .build_int_nuw_mul(
            build.build_and(block, v_upper, "and_outer").unwrap(),
            build.build_and(vm_c2, v_upper, "and_inner").unwrap(),
            "mul",
        )
        .unwrap();

    // v = v ^ (v >> 33);
    let block = build
        .build_xor(
            block,
            build.build_right_shift(block, v33, false, "shr").unwrap(),
            "xor",
        )
        .unwrap();

    build.build_return(Some(&block)).unwrap();

    func
}

fn generate_unchained_hash32<'a>(ctx: &'a Context, llvm_mod: &Module<'a>) -> FunctionValue<'a> {
    let i64_type = ctx.i64_type();
    let i32_type = ctx.i32_type();
    let crc32 = Intrinsic::find("llvm.x86.sse42.crc32.32.32").unwrap();

    let func = llvm_mod.add_function(
        "uc_hash32",
        i64_type.fn_type(&[i32_type.into()], false),
        None,
    );
    let feat = ctx.create_string_attribute("target-features", "+crc32");
    func.add_attribute(AttributeLoc::Function, feat);

    let input = func.get_nth_param(0).unwrap().into_int_value();

    let crc_f = crc32.get_declaration(llvm_mod, &[]).unwrap();

    declare_blocks!(ctx, func, entry);
    let build = ctx.create_builder();
    build.position_at_end(entry);
    let k = i64_type.const_int((0x8648DBDB << 32) + 1, false);
    let seed = i32_type.const_int(0x243F6A88, false);

    let crc = build
        .build_call(crc_f, &[seed.into(), input.into()], "crc")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    let crc = build
        .build_int_z_extend(crc, i64_type, "extended_crc")
        .unwrap();
    let res = build.build_int_mul(crc, k, "result").unwrap();
    build.build_return(Some(&res)).unwrap();
    func
}

fn generate_unchained_hash64<'a>(ctx: &'a Context, llvm_mod: &Module<'a>) -> FunctionValue<'a> {
    let i64_type = ctx.i64_type();
    let crc32 = Intrinsic::find("llvm.x86.sse42.crc32.64.64").unwrap();

    let func = llvm_mod.add_function(
        "uc_hash64",
        i64_type.fn_type(&[i64_type.into()], false),
        None,
    );
    let feat = ctx.create_string_attribute("target-features", "+crc32");
    func.add_attribute(AttributeLoc::Function, feat);
    let input = func.get_nth_param(0).unwrap().into_int_value();

    let crc_f = crc32.get_declaration(llvm_mod, &[]).unwrap();

    declare_blocks!(ctx, func, entry);
    let build = ctx.create_builder();
    build.position_at_end(entry);
    let k = i64_type.const_int(0x2545F4914F6CDD1D, false);
    let seed1 = i64_type.const_int(0xBF58476D1CE4E5B9, false);
    let seed2 = i64_type.const_int(0x94D049BB133111EB, false);

    let crc1 = build
        .build_call(crc_f, &[seed1.into(), input.into()], "crc1")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();

    let crc2 = build
        .build_call(crc_f, &[seed2.into(), input.into()], "crc2")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();

    let combined = build
        .build_or(
            crc1,
            build
                .build_left_shift(crc2, i64_type.const_int(32, false), "upper")
                .unwrap(),
            "combined",
        )
        .unwrap();

    let res = build.build_int_mul(combined, k, "result").unwrap();
    build.build_return(Some(&res)).unwrap();
    func
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashFunction {
    Murmur,
    Unchained,
}

impl HashFunction {
    pub fn generate_hf<'a>(
        &self,
        ctx: &'a Context,
        llvm_mod: &Module<'a>,
        pt: PrimitiveType,
    ) -> Result<FunctionValue<'a>, ArrowKernelError> {
        match self {
            HashFunction::Murmur => Ok(generate_hash_func(ctx, llvm_mod, pt)),
            HashFunction::Unchained => match pt {
                PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => {
                    Ok(generate_unchained_hash32(ctx, llvm_mod))
                }
                PrimitiveType::I64 | PrimitiveType::U64 => {
                    Ok(generate_unchained_hash64(ctx, llvm_mod))
                }
                _ => Err(ArrowKernelError::UnsupportedArguments(format!(
                    "unchained hash only available for 32 and 64 bit types (found {})",
                    pt
                ))),
            },
        }
    }
}

#[self_referencing]
pub struct HashKernel {
    dt: DataType,
    context: Context,

    #[borrows(context)]
    #[covariant]
    func: JitFunction<'this, unsafe extern "C" fn(*mut c_void, *mut c_void)>,
}
unsafe impl Send for HashKernel {}
unsafe impl Sync for HashKernel {}

impl Kernel for HashKernel {
    type Key = (DataType, HashFunction);

    type Input<'a> = &'a dyn Datum;

    type Params = HashFunction;

    type Output = UInt64Array;

    fn call(&self, inp: Self::Input<'_>) -> Result<Self::Output, ArrowKernelError> {
        let arr = inp.get().0;
        if arr.data_type() != self.borrow_dt() {
            return Err(ArrowKernelError::ArgumentMismatch(format!(
                "hash kernel expected {} but got {}",
                self.borrow_dt(),
                arr.data_type()
            )));
        }

        let mut iter = datum_to_iter(inp)?;
        let mut alloc = PrimitiveArrayWriter::allocate(arr.len(), PrimitiveType::U64);

        unsafe { self.borrow_func().call(iter.get_mut_ptr(), alloc.get_ptr()) };

        Ok(alloc.into_primitive_array(arr.len(), arr.logical_nulls()))
    }

    fn compile(inp: &Self::Input<'_>, hf: Self::Params) -> Result<Self, super::ArrowKernelError> {
        let ctx = Context::create();

        HashKernelTryBuilder {
            dt: inp.get().0.data_type().clone(),
            context: ctx,
            func_builder: |ctx| {
                let llvm_mod = ctx.create_module("hash");

                let ptr_t = ctx.ptr_type(AddressSpace::default());
                let p_type = PrimitiveType::for_arrow_type(inp.get().0.data_type());
                let p_llvm = p_type.llvm_type(ctx);

                let iter = datum_to_iter(*inp)?;
                let next_func =
                    generate_next(ctx, &llvm_mod, "hash_next", inp.get().0.data_type(), &iter)
                        .unwrap();
                let hash_func = hf.generate_hf(ctx, &llvm_mod, p_type)?;

                let func_ty = ctx
                    .void_type()
                    .fn_type(&[ptr_t.into(), ptr_t.into()], false);
                let func = llvm_mod.add_function("hash", func_ty, None);
                set_noalias_params(&func);
                declare_blocks!(ctx, func, entry, loop_cond, loop_body, exit);

                let b = ctx.create_builder();
                b.position_at_end(entry);
                let iter_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
                let out_ptr = func.get_nth_param(1).unwrap().into_pointer_value();
                let buf_ptr = b.build_alloca(p_llvm, "buf_ptr").unwrap();
                let writer = PrimitiveArrayWriter::llvm_init(
                    ctx,
                    &llvm_mod,
                    &b,
                    PrimitiveType::U64,
                    out_ptr,
                );
                b.build_unconditional_branch(loop_cond).unwrap();

                b.position_at_end(loop_cond);
                let res = b
                    .build_call(next_func, &[iter_ptr.into(), buf_ptr.into()], "next")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                b.build_conditional_branch(res, loop_body, exit).unwrap();

                b.position_at_end(loop_body);
                let item = b.build_load(p_llvm, buf_ptr, "item").unwrap();
                let hashed = b
                    .build_call(hash_func, &[item.into()], "hashed")
                    .unwrap()
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                writer.llvm_ingest(ctx, &b, hashed.into());
                b.build_unconditional_branch(loop_cond).unwrap();

                b.position_at_end(exit);
                b.build_return(None).unwrap();

                llvm_mod.verify().unwrap();
                optimize_module(&llvm_mod)?;

                let ee = llvm_mod
                    .create_jit_execution_engine(OptimizationLevel::Aggressive)
                    .unwrap();
                link_req_helpers(&llvm_mod, &ee)?;

                Ok(unsafe {
                    ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut c_void)>(
                        func.get_name().to_str().unwrap(),
                    )
                    .unwrap()
                })
            },
        }
        .try_build()
    }

    fn get_key_for_input(
        i: &Self::Input<'_>,
        hf: &Self::Params,
    ) -> Result<Self::Key, super::ArrowKernelError> {
        Ok((i.get().0.data_type().clone(), *hf))
    }
}

#[cfg(test)]
mod tests {

    use arrow_array::{Datum, Int16Array, Int32Array, Int64Array};
    use arrow_schema::DataType;
    use inkwell::{context::Context, OptimizationLevel};
    use itertools::Itertools;

    use crate::{
        compiled_kernels::{ht::HashFunction, link_req_helpers, Kernel},
        PrimitiveType,
    };

    use super::{generate_hash_func, generate_lookup_or_insert, HashKernel, TicketTable};

    #[test]
    fn test_ticket_overflow() {
        let mut tt = TicketTable::new(5, DataType::Int32, DataType::Int8);

        let ctx = Context::create();
        let module = ctx.create_module("test");
        let func = generate_lookup_or_insert(&ctx, &module, &tt);

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&module, &ee).unwrap();

        let test_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut TicketTable, i32, i64, *mut u8) -> i8>(
                func.get_name().to_str().unwrap(),
            )
            .unwrap()
        };

        let mut buf = 0;
        unsafe {
            test_func.call(&mut tt as *mut TicketTable, 50, 50, &mut buf);
            test_func.call(&mut tt as *mut TicketTable, 51, 51, &mut buf);
            test_func.call(&mut tt as *mut TicketTable, 52, 52, &mut buf);
            test_func.call(&mut tt as *mut TicketTable, 53, 53, &mut buf);
            test_func.call(&mut tt as *mut TicketTable, 54, 54, &mut buf);
            test_func.call(&mut tt as *mut TicketTable, 55, 55, &mut buf);
            assert_eq!(buf, 2);
        }
    }

    #[test]
    fn test_ticket_i32() {
        let mut tt = TicketTable::new(20, DataType::Int32, DataType::Int8);

        let ctx = Context::create();
        let module = ctx.create_module("test");
        let func = generate_lookup_or_insert(&ctx, &module, &tt);

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&module, &ee).unwrap();

        let test_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut TicketTable, i32, i64, *mut u8) -> i8>(
                func.get_name().to_str().unwrap(),
            )
            .unwrap()
        };

        let mut buf = 0;
        unsafe {
            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, 50, 5139532, &mut buf),
                0
            );
            assert!(buf != 0);
            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, 51, 5139532, &mut buf),
                1
            );
            assert!(buf != 0);
            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, 51, 5139532, &mut buf),
                1
            );
            assert!(buf == 0);
            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, 50, 5139532, &mut buf),
                0
            );
            assert!(buf == 0);
            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, 32, 51532, &mut buf),
                2
            );
            assert!(buf != 0);
            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, 50, 5139532, &mut buf),
                0
            );
            assert!(buf == 0);
        }
    }

    #[test]
    fn test_ticket_str() {
        let mut tt = TicketTable::new(20, DataType::Utf8, DataType::Int8);

        let ctx = Context::create();
        let module = ctx.create_module("test");
        let func = generate_lookup_or_insert(&ctx, &module, &tt);

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&module, &ee).unwrap();

        let test_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut TicketTable, u128, i64, *mut u8) -> i8>(
                func.get_name().to_str().unwrap(),
            )
            .unwrap()
        };

        let str1 = str_to_u128("this is a test");
        let str2 = str_to_u128("this is another test");
        let str3 = str_to_u128("hello");
        let str4 = str_to_u128("hello");

        let mut buf = 0;
        unsafe {
            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, str1, 1, &mut buf),
                0
            );
            assert!(buf != 0);
            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, str2, 1, &mut buf),
                1
            );
            assert!(buf != 0);
            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, str3, 1, &mut buf),
                2
            );
            assert!(buf != 0);

            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, str1, 1, &mut buf),
                0
            );
            assert!(buf == 0);
            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, str2, 1, &mut buf),
                1
            );
            assert!(buf == 0);
            assert_eq!(
                test_func.call(&mut tt as *mut TicketTable, str4, 1, &mut buf),
                2
            );
            assert!(buf == 0);
        }
    }

    #[test]
    fn test_hash_i32() {
        let ctx = Context::create();
        let module = ctx.create_module("test");
        let func = generate_hash_func(&ctx, &module, PrimitiveType::I32);
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&module, &ee).unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(i32) -> i64>(fname)
                .unwrap()
        };

        let data = vec![1, 2, 3, 4, 5];
        let num_unique = data
            .iter()
            .map(|&x| unsafe { next_block_func.call(x) })
            .sorted()
            .unique()
            .count();
        assert_eq!(num_unique, 5);
    }

    #[test]
    fn test_hash_f32() {
        let ctx = Context::create();
        let module = ctx.create_module("test");
        let func = generate_hash_func(&ctx, &module, PrimitiveType::F32);
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&module, &ee).unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(f32) -> i64>(fname)
                .unwrap()
        };

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let num_unique = data
            .iter()
            .map(|&x| unsafe { next_block_func.call(x) })
            .sorted()
            .unique()
            .count();
        assert_eq!(num_unique, 5);
    }

    fn str_to_u128(s: &str) -> u128 {
        let start_ptr = s.as_ptr();
        let end_ptr = start_ptr.wrapping_add(s.len());
        let combined: u128 = (start_ptr as u128) | (end_ptr as u128) << 64;
        combined
    }

    #[test]
    fn test_hash_str() {
        let ctx = Context::create();
        let module = ctx.create_module("test");
        let func = generate_hash_func(&ctx, &module, PrimitiveType::P64x2);
        let fname = func.get_name().to_str().unwrap();

        let short_str1 = "this";
        let short_str2 = "that&";
        let long_str1 = "this is a much longer string";
        let long_str2 = "this is a similar longer string";

        module.verify().unwrap();
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        link_req_helpers(&module, &ee).unwrap();

        let next_block_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(u128) -> i64>(fname)
                .unwrap()
        };

        let str1_hash = unsafe { next_block_func.call(str_to_u128(short_str1)) };
        let str2_hash = unsafe { next_block_func.call(str_to_u128(short_str2)) };
        assert_ne!(str1_hash, str2_hash);

        let str1_hash = unsafe { next_block_func.call(str_to_u128(long_str1)) };
        let str2_hash = unsafe { next_block_func.call(str_to_u128(long_str2)) };
        assert_eq!(str1_hash, str2_hash);
    }

    #[test]
    fn test_hash_murmur_kernel() {
        let data = Int32Array::from(vec![-1, -2, 0, 1, 2]);
        let k = HashKernel::compile(&(&data as &dyn Datum), HashFunction::Murmur).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(res.values().iter().unique().count(), 5);
    }

    #[test]
    fn test_hash_kernel_unchained32() {
        let data = Int32Array::from(vec![-1, -2, 0, 1, 2]);
        let k = HashKernel::compile(&(&data as &dyn Datum), HashFunction::Unchained).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(res.values().iter().unique().count(), 5);
    }

    #[test]
    fn test_hash_kernel_unchained64() {
        let data = Int64Array::from(vec![-1, -2, 0, 1, 2]);
        let k = HashKernel::compile(&(&data as &dyn Datum), HashFunction::Unchained).unwrap();
        let res = k.call(&data).unwrap();
        assert_eq!(res.values().iter().unique().count(), 5);
    }

    #[test]
    fn test_hash_kernel_unchained16_unsupported() {
        let data = Int16Array::from(vec![-1, -2, 0, 1, 2]);
        assert!(HashKernel::compile(&(&data as &dyn Datum), HashFunction::Unchained).is_err());
    }
}
