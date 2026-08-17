# Fix rounds: closing the arrow gaps on Zen 5

Companion to `zen5-findings.md`. That document diagnosed why the arrow
baseline beat the JIT on 20 of 60 benchmarks; this one records the five
rounds of fixes that followed, what each one does in plain terms, and how
each affects the operators involved — including behavior changes that are
not visible in the timings.

Every round was verified the same way: the full test suite on both an
Apple M4 and the Zen 5 machine, then a complete benchmark run on Zen 5
(clean tree, `-C target-cpu=native`), compared row-by-row against the
previous round for regressions. No round introduced one.

| after round | llvm wins | arrow wins | equal |
|---|---|---|---|
| (baseline `6fa1f69`) | 38 | 20 | 2 |
| 1 — findings fixes | 41 | 12 | 8 |
| 2 — REE + compress-store | 45 | 9 | 7 |
| 3 — profiling round | 47 | 7 | 7 |
| 4 — parity round | 48 | 3 | 10 |
| 5 — offense round | *(verification in flight)* | | |

## Round 1 — the fixes the findings document prescribed (`276587c`)

**1. Guard empty string comparisons in LIKE.**
A pattern like `%abc%` compiles to an empty prefix, an infix, and an empty
suffix, and the kernel compared every row against the two empty strings.
Comparing against an empty slice still emits a real `bcmp` call whose second
pointer is the empty `Vec` sentinel `0x1`, and on Zen 5 the AVX-512 path in
glibc pays a ~250-cycle microcode assist for exactly that call. The fix is
two `is_empty()` guards so the comparison is never issued.
*Operators affected:* `cmp::like` with two-wildcard patterns only; results
identical. *Effect:* `%abc%` 108.6 → 13.9 ms (0.09× → 0.73×).

**2. Right-size reduction outputs.**
Reductions declared their output as "at most n elements", so `sum` over 10m
i32 allocated (and zeroed) 40 MB to return 4 bytes. The size-resolution
machinery gained numeric constants, and reductions now declare exactly one
element. *Operators affected:* `compute::sum/min/max/product/argmin/argmax`
— allocation drops from O(n) to O(1). *Effect:* all four measured rows went
from 0.11–0.12× to parity (2.5 → 0.29 ms, the memory-bandwidth floor).

**3. Stop zero-filling output buffers.**
Every kernel invocation memset its output to zero before the kernel
overwrote it — a full extra pass over memory, plus the page faults that
first write triggers. Outputs are now allocated uninitialized; only the
prefix the kernel actually wrote is ever exposed.
*Operators affected:* every kernel with a primitive output. One behavioral
note: a hypothetical kernel that wrote fewer elements than it claimed used
to produce silent zeros and would now produce garbage — the invariant
(write what you claim) is unchanged, but violations became visible.
*Effect:* `neg_wrapping(f64)` 0.61× → 1.00×, `cast(i32→i64)` 0.90× →
1.69× win, `concat` 0.65× → 0.92×.

**4. Hoist `descending` out of the sort comparator.**
The comparator re-tested the sort direction on every comparison. Two
specialized comparators (one per direction) fixed that; the deterministic
index tie-break was kept. *Operators affected:* `sort::sort_to_indices`,
identical output. *Effect:* 0.52× → 0.70× (u64), 0.43× → 0.55× (nullable).

**5. Route prefix/suffix LIKE to the JIT kernel.**
`LIKE 'abc%'` ran a per-row Rust closure; the crate already had a compiled
prefix/suffix kernel that loops in bulk, so the pattern compiler now
delegates to it (and a bare `%` short-circuits to constant-true).
*Operators affected:* `cmp::like` with single-edge wildcards; null slots'
values may differ under the hood (validity is unchanged, matching arrow's
own convention). *Effect:* `'abc%'` 0.25× → **1.19× win**, `'%xyz'`
0.28× → **1.41× win**.

## Round 2 — encoded arrays and the compress-store filter (`ee8bde0`, `6c6f858`)

**6. Decode-first for non-run-uniform REE work.**
Run-end encoding wins when one answer covers a whole run (`cmp::lt` on REE
is a 27× win); it loses when the operation varies inside runs. A filter
mask always varies inside runs, so `select::filter` on REE input now
decodes to a dense array first and runs the dense kernel. `select::take`
decodes only when the indices are dense enough that per-index binary
search costs more than materializing (`indices × log₂(runs) ≥ length`);
sparse takes still stream through the encoding.
*Operators affected:* `select::filter` and `select::take` on run-end
encoded inputs. Behavioral note: decode-first materializes a temporary
dense copy (O(logical length) memory) for the duration of the call.
*Effect:* `filter(ree)` 0.54× → **3.92× win**, `take(ree)` 0.70× →
**35.7× win**.

**7. Masked compress-store vectorization.**
The dense filter compiled to a scalar bit-test-and-branch loop because the
vectorizer could not handle conditional emits. It now lowers
`cond(mask, emit(x))` to AVX-512 `vpcompressd`: 16 elements filtered per
~9 instructions instead of 9 instructions per element. Blocks are capped
at one native 512-bit vector (larger blocks crash LLVM's x86 backend).
*Operators affected:* `select::filter` with primitive outputs, on AVX-512
hosts only — the M4 and Meteor Lake machines keep the previous set-bits
loop, unchanged. *Effect:* `filter(i32)` 0.78× → **1.83× win** (and the
decoded REE filter above rides the same loop).

## Round 3 — what per-row profiling turned up (`09b4450`)

This round came from `perf`-profiling both sides of every remaining loss
(the causes are written up at the end of `zen5-findings.md`'s story: an
uninlined function call, append granularity, comparator shape, and a
loop-carried pointer the optimizer could not keep in a register).

**8. `#[inline]` on `PrimitiveType::width`.** The per-row string iterator
called this tiny function — a match statement returning a constant — as a
real function call, 11–13% of the string benchmarks. One attribute.
*Operators affected:* everything using the Rust-side `ArrowIter` (string
predicates foremost).

**9. Batch REE validity expansion.** `logical_nulls` on run-end encoded
arrays appended one span per run; it now batches consecutive valid runs,
so the call count scales with the number of *null* runs (arrow's shape).
*Operators affected:* `logical_nulls(ree)` and everything that consults
it. *Effect (with 10):* 0.43× → 0.97× equal.

**10. Word-packed boolean building in `filter_bytes`.** String predicates
appended their result one bit at a time (a read-modify-write per row);
they now pack 64 results per word in a register (`collect_bool`).
*Operators affected:* `cmp::like`, `cmp::contains`, `starts/ends_with`
on the closure path.

**11. Packed sort keys.** Instead of sorting (index, value) tuples with a
comparator, values are bit-transformed so plain unsigned order matches
their semantic order (sign-flip for signed ints, the `total_cmp` transform
for floats) and packed into one integer with the row index in the low
bits. Sorting plain integers needs no comparator at all, and ties resolve
on the index bits — the deterministic output is *bit-identical* to before.
Descending order inverts the key bits, leaving ties ascending.
*Operators affected:* `sort::sort_to_indices` on all primitive types;
output unchanged. *Effect:* u64 sort 16.9 → 13.6 ms.

**12. Word-wise null partition in sort.** The nullable pre-pass tested
validity row by row; it now walks the validity words directly.
*Effect (with 11):* nullable sort 11.4 → 9.0 ms.

**13. Cache write heads in registers.** Vectorized loops advanced their
output pointer through a load/store round-trip on the runtime struct every
block, which LLVM could not hoist (the data stores might alias it). The
pointer now lives in a stack slot the optimizer promotes to a register,
synced at loop boundaries. *Operators affected:* all vectorized kernels
with primitive outputs; pure codegen, no semantic change.
*Effect:* `concat` 0.90× → 1.01× equal.

**14. Inline string appends.** Appending a string crossed from JIT code
into Rust (`extend_from_slice`) once per row. The byte buffer's
pointer/length/capacity are now mirrored in JIT-visible fields, so the
capacity check and `memcpy` happen inline, calling Rust only to grow.
*Operators affected:* every kernel producing utf8/binary output (casts,
filters, takes, concats of strings). *Effect:* `cast(dict→utf8)`
8.07 → 6.63 ms.

## Round 4 — the parity round (`94e9098`)

**15. Flat-bytes fast path for string predicates.** `filter_bytes` still
paid a buffered iterator plus closure indirection per row; for plain
utf8/binary arrays it now reads the offsets buffer directly, exactly
arrow's `from_unary` shape. Encoded layouts keep the iterator path.
*Operators affected:* `cmp::like`, `cmp::contains` on flat arrays; null
slots' values are computed and ignored (arrow's convention).
*Effect:* `like('%abc%')` 0.85× → **1.00× equal**, `contains` 0.90× →
**1.05× win**.

**16. `take_bits` route for boolean take.** Gathering bits through the
JIT serialized on a read-modify-write per output bit; plain boolean
arrays with non-null integer indices now route to a word-packed Rust
gather. *Operators affected:* `select::take` on plain booleans (encoded
layouts keep the JIT). *Effect:* 0.94× → **1.00× equal**.

**17. Radix sort for packed keys.** Above 50k rows, a stable LSD radix
sort over the packed keys' value digits replaces the comparison sort.
Stability plus the index bits makes the result provably identical to the
comparison sort (a tie-heavy unit test asserts it), so determinism
survives while the comparator disappears entirely.
*Operators affected:* large `sort::sort_to_indices`; adds one scratch
buffer the size of the key array during the sort. *Effect:* u64 sort
13.6 → 11.7 ms, 0.87× → **1.02× equal** — the determinism premium is gone.

## Round 5 — the offense round (`0be4bd6`, verification in flight)

**18. One-pass null partition.** The sort pre-pass now builds the valid
and null index lists in a single pass over the validity words (set bits
pack valids, cleared bits push nulls), with the null list pre-sized.
*Operators affected:* nullable `sort_to_indices`.

**19. 11-bit radix digits.** Six passes cover a 64-bit key instead of
eight, trading a 16 KB histogram for 25% less scatter traffic.
*Operators affected:* large sorts; also expected to push the non-null
sort from "equal" into a win.

**20. Non-temporal stores for large arithmetic outputs.** Ordinary stores
read each output cache line before overwriting it; for outputs too large
to stay in cache that read is pure waste. Kernels whose output is ≥32 MB
now compile a variant whose block stores bypass the cache (with a fence
before returning). The size class is part of the kernel cache key, so each
shape lazily holds a small and a large variant.
*Operators affected:* `arith::*` with large outputs — this is something
arrow's ahead-of-time kernels do not do, so it targets a win, not a tie.
First use of each size class pays one extra ~10 ms compile.

**21. Huge-page advice for large outputs.** Fresh multi-megabyte
allocations land on transparent huge pages only if the mmap happens to be
2 MB-aligned — luck that made `concat` bimodal (2.65 ms or 2.96 ms run to
run). Large writer allocations now `madvise(MADV_HUGEPAGE)` their
interior (Linux only, best effort).
*Operators affected:* every kernel with a multi-megabyte primitive
output; removes run-to-run timing variance rather than changing the
median.
