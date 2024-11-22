use std::mem;

use arrow_array::{cast::AsArray, types::Int64Type, Array, BooleanArray, Datum};
use arrow_buffer::{BooleanBuffer, Buffer};
use cranelift::{
    codegen::{
        self,
        ir::{types, Function, UserFuncName},
        verify_function,
    },
    prelude::{
        isa::CallConv, settings, AbiParam, Block, Configurable, EntityRef, FunctionBuilder,
        FunctionBuilderContext, InstBuilder, IntCC, MemFlags, Signature, Type, Variable,
    },
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;

#[derive(Default)]
struct VariableGenerator {
    next_var: usize,
}

impl VariableGenerator {
    pub fn new(&mut self) -> Variable {
        let var = Variable::new(self.next_var);
        self.next_var += 1;
        var
    }
}

struct Iterator {
    idx: Variable,
    init_block: Block,
    reenter_block: Block,
    iter_block: Block,
}

impl Iterator {
    pub fn declare_iter(fb: &mut FunctionBuilder, vg: &mut VariableGenerator) -> Iterator {
        Iterator {
            idx: vg.new(),
            init_block: fb.create_block(),
            reenter_block: fb.create_block(),
            iter_block: fb.create_block(),
        }
    }

    pub fn init_block(&self) -> Block {
        self.init_block
    }

    pub fn reenter_block(&self) -> Block {
        self.reenter_block
    }

    pub fn define_iter(self, fb: &mut FunctionBuilder, next_block: Block, final_block: Block) {
        fb.switch_to_block(self.init_block());
        let ptr = fb.append_block_param(self.init_block(), types::I64);
        let length = fb.append_block_param(self.init_block(), types::I64);
        fb.declare_var(self.idx, types::I64);
        let zero = fb.ins().iconst(types::I64, 0);
        fb.def_var(self.idx, zero);
        fb.ins().jump(self.reenter_block(), &[]);

        fb.switch_to_block(self.reenter_block());
        let curr_idx = fb.use_var(self.idx);
        let cmp_res = fb.ins().icmp(IntCC::UnsignedLessThan, curr_idx, length);
        fb.ins()
            .brif(cmp_res, self.iter_block, &[], final_block, &[]);

        fb.switch_to_block(self.iter_block);
        let curr_idx = fb.use_var(self.idx);
        let curr_idx_b = fb.ins().imul_imm(curr_idx, 8);
        let ptr_offset = fb.ins().iadd(ptr, curr_idx_b);
        let loaded_val = fb
            .ins()
            .load(types::I64, MemFlags::trusted(), ptr_offset, 0);
        let new_iter_var = fb.ins().iadd_imm(curr_idx, 1);
        fb.def_var(self.idx, new_iter_var);
        fb.ins().jump(next_block, &[loaded_val]);
        fb.seal_block(self.iter_block);
    }
}

struct Bitmap {
    init_block: Block,
    append_block: Block,
    store_block: Block,
    finish_block: Block,
    store_last_block: Block,
    curr_buf: Variable,
    curr_within_u32_idx: Variable,
    curr_between_u32_idx: Variable,
}

impl Bitmap {
    pub fn declare_bitmap(fb: &mut FunctionBuilder, vg: &mut VariableGenerator) -> Bitmap {
        Bitmap {
            init_block: fb.create_block(),
            append_block: fb.create_block(),
            store_block: fb.create_block(),
            finish_block: fb.create_block(),
            store_last_block: fb.create_block(),
            curr_buf: vg.new(),
            curr_within_u32_idx: vg.new(),
            curr_between_u32_idx: vg.new(),
        }
    }

    pub fn define_bitmap(
        self,
        fb: &mut FunctionBuilder,
        after_init_block: Block,
        return_block: Block,
        after_finish_block: Block,
    ) {
        fb.switch_to_block(self.init_block);
        let ptr = fb.append_block_param(self.init_block, types::I64);
        fb.declare_var(self.curr_within_u32_idx, types::I8);
        fb.declare_var(self.curr_between_u32_idx, types::I64);
        fb.declare_var(self.curr_buf, types::I32);
        let zero = fb.ins().iconst(types::I8, 0);
        fb.def_var(self.curr_within_u32_idx, zero);
        let zero = fb.ins().iconst(types::I64, 0);
        fb.def_var(self.curr_between_u32_idx, zero);
        let zero = fb.ins().iconst(types::I32, 0);
        fb.def_var(self.curr_buf, zero);
        fb.ins().jump(after_init_block, &[]);

        fb.switch_to_block(self.append_block);
        let idx = fb.append_block_param(self.append_block, types::I8);
        // precondition: curr_within_u32 < 32, so we can insert into the buffer
        let idx = fb.ins().uextend(types::I32, idx);
        let curr_buff_idx = fb.use_var(self.curr_within_u32_idx);
        let shifted_idx = fb.ins().ishl(idx, curr_buff_idx);
        let curr_buf = fb.use_var(self.curr_buf);
        let new_buf = fb.ins().bor(curr_buf, shifted_idx);
        fb.def_var(self.curr_buf, new_buf);
        let new_buff_idx = fb.ins().iadd_imm(curr_buff_idx, 1);
        fb.def_var(self.curr_within_u32_idx, new_buff_idx);

        // check to see if this was our last slot in the buffer
        let cmp = fb.ins().icmp_imm(IntCC::UnsignedLessThan, new_buff_idx, 32);
        fb.ins().brif(cmp, return_block, &[], self.store_block, &[]);

        fb.switch_to_block(self.store_block);
        let curr_idx = fb.use_var(self.curr_between_u32_idx);
        let ptr_offset_amt = fb.ins().imul_imm(curr_idx, 4);
        let offset_ptr = fb.ins().iadd(ptr, ptr_offset_amt);
        let curr_buf = fb.use_var(self.curr_buf);
        fb.ins().store(MemFlags::trusted(), curr_buf, offset_ptr, 0);
        let zero = fb.ins().iconst(types::I8, 0);
        fb.def_var(self.curr_within_u32_idx, zero);
        let zero = fb.ins().iconst(types::I32, 0);
        fb.def_var(self.curr_buf, zero);
        let new_idx = fb.ins().iadd_imm(curr_idx, 1);
        fb.def_var(self.curr_between_u32_idx, new_idx);
        fb.ins().jump(return_block, &[]);
        fb.seal_block(self.store_block);

        fb.switch_to_block(self.finish_block);
        let buf_idx = fb.use_var(self.curr_within_u32_idx);
        fb.ins()
            .brif(buf_idx, self.store_last_block, &[], after_finish_block, &[]);

        fb.switch_to_block(self.store_last_block);
        let curr_idx = fb.use_var(self.curr_between_u32_idx);
        let ptr_offset_amt = fb.ins().imul_imm(curr_idx, 4);
        let offset_ptr = fb.ins().iadd(ptr, ptr_offset_amt);
        let curr_buf = fb.use_var(self.curr_buf);
        fb.ins().store(MemFlags::trusted(), curr_buf, offset_ptr, 0);
        fb.ins().jump(after_finish_block, &[]);
        fb.seal_block(self.store_last_block);
    }

    pub fn init_block(&self) -> Block {
        self.init_block
    }

    pub fn append_block(&self) -> Block {
        self.append_block
    }

    fn finish_block(&self) -> Block {
        self.finish_block
    }
}

pub struct CompiledEqConst {
    _module: JITModule,
    dt: arrow_schema::DataType,
    func: Box<dyn Fn(&dyn Array, &dyn Datum) -> BooleanArray>,
}

impl CompiledEqConst {
    pub fn execute(&self, array: &dyn Array, datum: &dyn Datum) -> BooleanArray {
        (self.func)(array, datum)
    }
}

pub fn compile_eq_const() -> CompiledEqConst {
    let mut ctx = codegen::Context::new();
    let mut sig = Signature::new(CallConv::SystemV);
    let native = cranelift_native::builder().unwrap();
    let ptr_type = Type::triple_pointer_type(native.triple());

    // ptr
    sig.params.push(AbiParam::new(ptr_type));

    // length
    sig.params.push(AbiParam::new(types::I64));

    // value
    sig.params.push(AbiParam::new(types::I64));

    // bitmap pointer
    sig.params.push(AbiParam::new(ptr_type));

    ctx.func = Function::with_name_signature(UserFuncName::user(0, 0), sig);

    let mut fbctx = FunctionBuilderContext::new();
    let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
    let mut vg = VariableGenerator::default();

    let entry_block = fb.create_block();
    let preloop_block = fb.create_block();
    let loop_body = fb.create_block();
    let end_loop = fb.create_block();
    let end_block = fb.create_block();
    let iter = Iterator::declare_iter(&mut fb, &mut vg);
    let bitmap = Bitmap::declare_bitmap(&mut fb, &mut vg);

    fb.switch_to_block(entry_block);
    fb.append_block_params_for_function_params(entry_block);
    let data_ptr = fb.block_params(entry_block)[0];
    let len = fb.block_params(entry_block)[1];
    let const_val = fb.block_params(entry_block)[2];
    let bitmap_ptr = fb.block_params(entry_block)[3];
    fb.ins().jump(bitmap.init_block(), &[bitmap_ptr]);
    fb.seal_block(entry_block);

    fb.switch_to_block(preloop_block);
    fb.ins().jump(iter.init_block(), &[data_ptr, len]);

    fb.switch_to_block(loop_body);
    fb.append_block_param(loop_body, types::I64);
    let loaded_val = fb.block_params(loop_body)[0];
    let cmp_result = fb.ins().icmp(IntCC::Equal, loaded_val, const_val);
    fb.ins().jump(bitmap.append_block(), &[cmp_result]);

    fb.switch_to_block(end_loop);
    fb.ins().jump(bitmap.finish_block(), &[]);

    fb.switch_to_block(end_block);
    fb.ins().return_(&[]);

    bitmap.define_bitmap(&mut fb, preloop_block, iter.reenter_block(), end_block);
    fb.seal_block(preloop_block);
    iter.define_iter(&mut fb, loop_body, end_loop);

    fb.seal_all_blocks();
    fb.finalize();

    let settings_builder = settings::builder();
    let flags = settings::Flags::new(settings_builder);
    verify_function(&ctx.func, &flags).unwrap();

    let builder = JITBuilder::with_flags(
        &[("opt_level", "speed")],
        cranelift_module::default_libcall_names(),
    )
    .unwrap();
    let mut module = JITModule::new(builder);

    let count_eq_id = module
        .declare_function(
            "count_eq",
            cranelift_module::Linkage::Export,
            &ctx.func.signature,
        )
        .unwrap();

    module.define_function(count_eq_id, &mut ctx).unwrap();

    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let code = module.get_finalized_function(count_eq_id);

    let runnable_fn =
        unsafe { mem::transmute::<_, fn(*const i64, u64, i64, *mut u32) -> u64>(code) };

    let func = Box::new(move |arr: &dyn Array, c: &dyn Datum| {
        let arr = arr.as_primitive::<Int64Type>();
        let c = {
            let (c, is_scalar) = c.get();
            assert!(is_scalar);
            c.as_primitive::<Int64Type>().value(0)
        };
        let mut bitmap_buf = vec![0_u32; arr.len().div_ceil(32)];
        runnable_fn(
            arr.values().as_ptr(),
            arr.len() as u64,
            c,
            bitmap_buf.as_mut_ptr(),
        );
        let b = BooleanBuffer::new(Buffer::from_vec(bitmap_buf), 0, arr.len());
        BooleanArray::new(b, None)
    });

    CompiledEqConst {
        dt: arrow_schema::DataType::Int64,
        _module: module,
        func,
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;
    use arrow_ord::cmp;

    use super::*;

    #[test]
    fn test_eq_const() {
        let f = compile_eq_const();
        let data = Int64Array::from(vec![1, 2, 3, 3, 4, 5, 3]);
        let r = f.execute(&data, &Int64Array::new_scalar(3));
        assert_eq!(r.true_count(), 3);

        let arrow_result = cmp::eq(&data, &Int64Array::new_scalar(3)).unwrap();
        assert_eq!(arrow_result, r);

        let data: Vec<i64> = (0..200).collect();
        let data = Int64Array::from(data);
        let r = f.execute(&data, &Int64Array::new_scalar(199));
        assert_eq!(r.true_count(), 1);

        let arrow_result = cmp::eq(&data, &Int64Array::new_scalar(199)).unwrap();
        assert_eq!(arrow_result, r);
    }
}
