use arrow_array::{Array, BooleanArray};
use inkwell::{
    builder::Builder,
    context::Context,
    values::{IntValue, PointerValue},
    AddressSpace,
};
use repr_offset::ReprOffset;

use crate::{increment_pointer, mark_load_invariant};

use std::ops::Range;

/// A split of a bit-span into (optional) head/body/tail bit-index ranges.
/// All ranges are half-open: [start, end).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Segments {
    head: Option<Range<usize>>,
    body: Option<Range<usize>>,
    tail: Option<Range<usize>>,
}

/// Split a bit-span into head/body/tail relative to 64-bit alignment.
///
/// - `total_bits`: total number of bits in the underlying bit array (for bounds checking).
/// - `offset`: starting bit index into the array.
/// - `len`: length of the span in bits.
///
/// Returns `Err` if the (offset, len) span is out of bounds.
fn split_bit_span(offset: usize, len: usize) -> Result<Segments, &'static str> {
    // Basic bounds & overflow-safe end calculation
    let end = offset
        .checked_add(len)
        .ok_or("offset + len overflows usize")?;

    if len == 0 {
        return Ok(Segments {
            head: None,
            body: None,
            tail: None,
        });
    }

    fn align_up64(x: usize) -> usize {
        (x + 63) & !63
    }

    fn align_down64(x: usize) -> usize {
        x & !63
    }

    // Compute candidate boundaries
    let first_aligned = align_up64(offset);
    let last_aligned = align_down64(end);

    // Head: from offset up to min(end, first_aligned) if offset is not aligned.
    let head = if offset % 64 != 0 {
        let he = first_aligned.min(end);
        Some(offset..he).filter(|r| r.start < r.end)
    } else {
        None
    };

    // Body: fully aligned u64s between first_aligned and last_aligned. Note: If
    // offset was aligned, first_aligned == offset. If end was aligned,
    // last_aligned == end.
    let body_start = first_aligned.min(end);
    let body_end = last_aligned;
    let body = Some(body_start..body_end).filter(|r| r.start < r.end);

    // Tail: any remainder from max(body_end, body_start) to end (covers cases with no body).
    let tail_start = body_end.max(body_start);
    let tail = Some(tail_start..end).filter(|r| r.start < r.end);

    Ok(Segments { head, body, tail })
}

#[repr(C)]
#[derive(ReprOffset, Debug)]
#[roff(usize_offsets)]
pub struct SetBitIterator {
    bitmap: *const u8,
    header: [u64; 64],
    header_idx: u64,
    header_len: u64,
    tail: [u64; 64],
    tail_idx: u64,
    tail_len: u64,

    current_u64: u64,
    curr_setbit_idx: u64,
    segment_pos: u64,
    segment_len: u64,
    array_ref: BooleanArray,
}

impl From<&BooleanArray> for SetBitIterator {
    fn from(array: &BooleanArray) -> Self {
        assert!(
            array.nulls().is_none(),
            "cannot iterate over set bits of array with nulls"
        );

        let segments = split_bit_span(array.offset(), array.len()).unwrap();

        let mut header_buf = [0_u64; 64];
        let (header, header_len) = match &segments.head {
            Some(range) => {
                let header = array.slice(0, range.len());
                let mut header_len = 0;
                for (idx, dst) in header.values().set_indices().zip(header_buf.iter_mut()) {
                    *dst = idx as u64;
                    header_len += 1;
                }
                (header_buf, header_len)
            }
            None => (header_buf, 0),
        };

        let mut tail_buf = [0_u64; 64];
        let (tail, tail_len) = match &segments.tail {
            Some(range) => {
                let tail = array.slice(array.len() - range.len(), range.len());
                let mut tail_len = 0;
                for (idx, dst) in tail.values().set_indices().zip(tail_buf.iter_mut()) {
                    *dst = idx as u64;
                    tail_len += 1;
                }
                (tail_buf, tail_len)
            }
            None => (tail_buf, 0),
        };

        let (first_full_segment_idx, last_full_segment_idx) = match segments.body {
            Some(range) => (range.start / 64, range.end / 64),
            None => (0, 0),
        };

        SetBitIterator {
            bitmap: array.values().values().as_ptr(),
            header,
            header_len,
            tail,
            tail_len,
            header_idx: 0,
            tail_idx: 0,
            current_u64: 0,
            curr_setbit_idx: segments.head.map(|r| r.len()).unwrap_or(0) as u64,
            segment_pos: first_full_segment_idx as u64,
            segment_len: last_full_segment_idx as u64 - first_full_segment_idx as u64,
            array_ref: array.clone(),
        }
    }
}

impl SetBitIterator {
    /// returns a pointer to the header, along with the current index and length
    pub fn llvm_header_info<'a>(
        &self,
        ctx: &'a Context,
        b: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> (PointerValue<'a>, IntValue<'a>, IntValue<'a>) {
        let header_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_HEADER);
        let header_len_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_HEADER_LEN);
        let header_len = b
            .build_load(ctx.i64_type(), header_len_ptr, "header_len")
            .unwrap();
        mark_load_invariant!(ctx, header_len);

        let header_idx_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_HEADER_IDX);
        let header_idx = b
            .build_load(ctx.i64_type(), header_idx_ptr, "header_idx")
            .unwrap()
            .into_int_value();
        (header_ptr, header_idx, header_len.into_int_value())
    }

    pub fn llvm_inc_header_pos<'a>(&self, ctx: &'a Context, b: &'a Builder, ptr: PointerValue<'a>) {
        let header_idx_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_HEADER_IDX);
        let header_idx = b
            .build_load(ctx.i64_type(), header_idx_ptr, "header_idx")
            .unwrap()
            .into_int_value();
        let new_header_idx = b
            .build_int_add(
                header_idx,
                ctx.i64_type().const_int(1, false),
                "new_header_idx",
            )
            .unwrap();
        b.build_store(header_idx_ptr, new_header_idx).unwrap();
    }

    /// returns a pointer to the tail, along with the current index and length
    pub fn llvm_tail_info<'a>(
        &self,
        ctx: &'a Context,
        b: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> (PointerValue<'a>, IntValue<'a>, IntValue<'a>) {
        let tail_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_TAIL);
        let tail_len_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_TAIL_LEN);
        let tail_len = b
            .build_load(ctx.i64_type(), tail_len_ptr, "tail_len")
            .unwrap();
        mark_load_invariant!(ctx, tail_len);
        let tail_idx_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_TAIL_IDX);
        let tail_idx = b
            .build_load(ctx.i64_type(), tail_idx_ptr, "tail_idx")
            .unwrap()
            .into_int_value();
        (tail_ptr, tail_idx, tail_len.into_int_value())
    }

    pub fn llvm_inc_tail_pos<'a>(&self, ctx: &'a Context, b: &'a Builder, ptr: PointerValue<'a>) {
        let tail_idx_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_TAIL_IDX);
        let tail_idx = b
            .build_load(ctx.i64_type(), tail_idx_ptr, "tail_idx")
            .unwrap()
            .into_int_value();
        let new_tail_idx = b
            .build_int_add(tail_idx, ctx.i64_type().const_int(1, false), "new_tail_idx")
            .unwrap();
        b.build_store(tail_idx_ptr, new_tail_idx).unwrap();
    }

    /// extracts a 64-bit segment from the bitmap
    pub fn llvm_get_segment<'a>(
        &self,
        ctx: &'a Context,
        b: &'a Builder,
        idx: IntValue<'a>,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let bitmap_ptr_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_BITMAP);
        let bitmap_ptr = b
            .build_load(
                ctx.ptr_type(AddressSpace::default()),
                bitmap_ptr_ptr,
                "bitmap_ptr",
            )
            .unwrap();
        mark_load_invariant!(ctx, bitmap_ptr);
        let segment_ptr = increment_pointer!(ctx, b, bitmap_ptr.into_pointer_value(), 8, idx);
        b.build_load(ctx.i64_type(), segment_ptr, "segment")
            .unwrap()
            .into_int_value()
    }

    /// the total number of segments to process
    pub fn llvm_get_num_segments<'a>(
        &self,
        ctx: &'a Context,
        b: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let segment_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_SEGMENT_LEN);
        let segment_len = b
            .build_load(ctx.i64_type(), segment_ptr, "segment_len")
            .unwrap();
        mark_load_invariant!(ctx, segment_len);
        segment_len.into_int_value()
    }

    /// the current segment position
    pub fn llvm_get_curr_segment_pos<'a>(
        &self,
        ctx: &'a Context,
        b: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let segment_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_SEGMENT_POS);
        let segment_pos = b
            .build_load(ctx.i64_type(), segment_ptr, "segment_pos")
            .unwrap();
        segment_pos.into_int_value()
    }

    /// increment the current segment position, returns the new position
    pub fn llvm_inc_curr_segment<'a>(
        &self,
        ctx: &'a Context,
        b: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let segment_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_SEGMENT_POS);
        let segment_pos = b
            .build_load(ctx.i64_type(), segment_ptr, "segment_pos")
            .unwrap()
            .into_int_value();
        let new_segment_pos = b
            .build_int_add(
                segment_pos,
                ctx.i64_type().const_int(1, false),
                "new_segment_pos",
            )
            .unwrap();
        b.build_store(segment_ptr, new_segment_pos).unwrap();
        new_segment_pos
    }

    pub fn llvm_set_current_u64<'a>(
        &self,
        ctx: &'a Context,
        b: &'a Builder,
        val: IntValue<'a>,
        ptr: PointerValue<'a>,
    ) {
        let curr_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_CURRENT_U64);
        b.build_store(curr_ptr, val).unwrap();
    }

    pub fn llvm_get_current_u64<'a>(
        &self,
        ctx: &'a Context,
        b: &'a Builder,
        ptr: PointerValue<'a>,
    ) -> IntValue<'a> {
        let curr_ptr = increment_pointer!(ctx, b, ptr, SetBitIterator::OFFSET_CURRENT_U64);
        b.build_load(ctx.i64_type(), curr_ptr, "current_u64")
            .unwrap()
            .into_int_value()
    }

    /// clears the trailing bit of the current u64 value
    pub fn llvm_clear_last<'a>(&self, ctx: &'a Context, build: &'a Builder, ptr: PointerValue<'a>) {
        let curr_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_CURRENT_U64);
        let curr = build
            .build_load(ctx.i64_type(), curr_ptr, "load_curr")
            .unwrap()
            .into_int_value();
        // Use no unsigned wrap since we know we'll never call this function when curr == 0.
        let curr_minus_one = build
            .build_int_nuw_sub(curr, ctx.i64_type().const_int(1, false), "curr_minus_one")
            .unwrap();
        let curr_and = build.build_and(curr, curr_minus_one, "curr_and").unwrap();
        build.build_store(curr_ptr, curr_and).unwrap();
    }

    pub fn llvm_add_and_get_current_bit_idx<'a>(
        &self,
        ctx: &'a Context,
        build: &'a Builder,
        ptr: PointerValue<'a>,
        val: IntValue<'a>,
        store: bool,
    ) -> IntValue<'a> {
        let curr_ptr = increment_pointer!(ctx, build, ptr, SetBitIterator::OFFSET_CURR_SETBIT_IDX);
        let curr = build
            .build_load(ctx.i64_type(), curr_ptr, "curr_idx")
            .unwrap()
            .into_int_value();
        let next = build.build_int_add(curr, val, "next").unwrap();
        if store {
            build.build_store(curr_ptr, next).unwrap();
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use arrow_array::{Array, BooleanArray};
    use inkwell::{context::Context, execution_engine::JitFunction, OptimizationLevel};
    use itertools::Itertools;

    use crate::compiled_iter::{array_to_setbit_iter, generate_next, IteratorHolder};

    fn test_iter(
        arr: &BooleanArray,
        ih: &mut IteratorHolder,
        func: JitFunction<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>,
    ) {
        let arrow_iter = arr.values().set_indices();

        let mut buf: u64 = 0;
        for val in arrow_iter {
            unsafe {
                assert!(func.call(ih.get_mut_ptr(), &mut buf as *mut u64));
                assert_eq!(buf, val as u64, "expected {}, got {}", val, buf);
            }
        }

        unsafe {
            assert!(!func.call(ih.get_mut_ptr(), &mut buf as *mut u64));
        }
    }

    #[test]
    fn test_setbit_all_tail() {
        let data = BooleanArray::from(vec![
            true, true, false, true, false, false, false, false, true, true,
        ]);

        let mut iter = array_to_setbit_iter(&data).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("setbit_test");
        let func = generate_next(&ctx, &module, "setbit_iter", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>(fname)
                .unwrap()
        };

        test_iter(&data, &mut iter, next_func);
    }

    #[test]
    fn test_setbit_every_other() {
        let data = BooleanArray::from((0..1000).map(|i| i % 2 == 0).collect_vec());

        let mut iter = array_to_setbit_iter(&data).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("setbit_test");
        let func = generate_next(&ctx, &module, "setbit_iter", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>(fname)
                .unwrap()
        };

        test_iter(&data, &mut iter, next_func);
    }

    #[test]
    fn test_setbit_every_other_sliced_small() {
        let data = BooleanArray::from((0..1000).map(|i| i % 2 == 0).collect_vec());
        let data = data.slice(100, 10);

        let mut iter = array_to_setbit_iter(&data).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("setbit_test");
        let func = generate_next(&ctx, &module, "setbit_iter", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>(fname)
                .unwrap()
        };

        test_iter(&data, &mut iter, next_func);
    }

    #[test]
    fn test_setbit_every_other_sliced_large() {
        let data = BooleanArray::from((0..1000).map(|i| i % 2 == 0).collect_vec());
        let data = data.slice(10, 100);

        let mut iter = array_to_setbit_iter(&data).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("setbit_test");
        let func = generate_next(&ctx, &module, "setbit_iter", data.data_type(), &iter).unwrap();
        let fname = func.get_name().to_str().unwrap();

        module.verify().unwrap();

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let next_func = unsafe {
            ee.get_function::<unsafe extern "C" fn(*mut c_void, *mut u64) -> bool>(fname)
                .unwrap()
        };

        test_iter(&data, &mut iter, next_func);
    }

    use super::*;

    #[test]
    fn empty_span() {
        let s = split_bit_span(100, 0).unwrap();
        assert_eq!(s.head, None);
        assert_eq!(s.body, None);
        assert_eq!(s.tail, None);
    }

    #[test]
    fn fully_aligned_both_ends() {
        // offset and end both multiples of 64
        let s = split_bit_span(128, 128).unwrap(); // [128, 256)
        assert_eq!(s.head, None);
        assert_eq!(s.body, Some(128..256));
        assert_eq!(s.tail, None);
    }

    #[test]
    fn head_only_span() {
        // Span ends before reaching next 64 boundary
        // offset=5, next boundary=64, end=20  -> head=[5,20)
        let s = split_bit_span(5, 15).unwrap(); // [5,20)
        assert_eq!(s.head, Some(5..20));
        assert_eq!(s.body, None);
        assert_eq!(s.tail, None);
    }

    #[test]
    fn tail_only_span() {
        // offset aligned, but end not aligned and < offset+64
        // offset=64, end=90 -> tail=[64,90)
        let s = split_bit_span(64, 26).unwrap(); // [64,90)
        assert_eq!(s.head, None);
        assert_eq!(s.body, None);
        assert_eq!(s.tail, Some(64..90));
    }

    #[test]
    fn head_body_tail() {
        // offset=5 (unaligned) -> head up to 64
        // end=5+200=205; last_aligned=floor(205,64)=192
        // head=[5,64), body=[64,192), tail=[192,205)
        let s = split_bit_span(5, 200).unwrap();
        assert_eq!(s.head, Some(5..64));
        assert_eq!(s.body, Some(64..192));
        assert_eq!(s.tail, Some(192..205));
    }

    #[test]
    fn body_and_tail_only() {
        // offset aligned; end not aligned, span crosses >= one full u64
        // offset=128, end=128+100=228; last_aligned=192
        // body=[128,192), tail=[192,228)
        let s = split_bit_span(128, 100).unwrap();
        assert_eq!(s.head, None);
        assert_eq!(s.body, Some(128..192));
        assert_eq!(s.tail, Some(192..228));
    }

    #[test]
    fn test_setbit_repr_offsets() {
        use std::mem::offset_of;

        assert_eq!(
            SetBitIterator::OFFSET_HEADER,
            offset_of!(SetBitIterator, header)
        );
        assert_eq!(
            SetBitIterator::OFFSET_HEADER_IDX,
            offset_of!(SetBitIterator, header_idx)
        );
        assert_eq!(
            SetBitIterator::OFFSET_HEADER_LEN,
            offset_of!(SetBitIterator, header_len)
        );
        assert_eq!(
            SetBitIterator::OFFSET_TAIL,
            offset_of!(SetBitIterator, tail)
        );
        assert_eq!(
            SetBitIterator::OFFSET_TAIL_IDX,
            offset_of!(SetBitIterator, tail_idx)
        );
        assert_eq!(
            SetBitIterator::OFFSET_TAIL_LEN,
            offset_of!(SetBitIterator, tail_len)
        );
        assert_eq!(
            SetBitIterator::OFFSET_CURRENT_U64,
            offset_of!(SetBitIterator, current_u64)
        );
        assert_eq!(
            SetBitIterator::OFFSET_CURR_SETBIT_IDX,
            offset_of!(SetBitIterator, curr_setbit_idx)
        );
        assert_eq!(
            SetBitIterator::OFFSET_SEGMENT_POS,
            offset_of!(SetBitIterator, segment_pos)
        );
        assert_eq!(
            SetBitIterator::OFFSET_SEGMENT_LEN,
            offset_of!(SetBitIterator, segment_len)
        );
        assert_eq!(
            SetBitIterator::OFFSET_BITMAP,
            offset_of!(SetBitIterator, bitmap)
        );
    }
}
