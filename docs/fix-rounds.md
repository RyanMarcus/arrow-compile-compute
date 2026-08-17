# Fix rounds: closing the arrow gaps on Zen 5

Companion to `zen5-findings.md`. That document diagnosed *why* stock arrow
beat our JIT-compiled kernels on 20 of 60 benchmarks. This one records the
five rounds of fixes that followed. Each fix is described three ways: what
was slow, what we changed, and what it means for the operators involved.

**How each round was verified:** full test suite on two machines (Apple M4
and the Zen 5 box), then a complete benchmark run on Zen 5, compared
row-by-row against the previous round. No round introduced a regression.

**The scoreboard over time** (out of ~60 paired benchmarks — higher "llvm
wins" is better):

| after round | llvm wins | arrow wins | equal |
|---|---|---|---|
| baseline | 38 | 20 | 2 |
| 1 — findings fixes | 41 | 12 | 8 |
| 2 — encoded arrays + AVX-512 filter | 45 | 9 | 7 |
| 3 — profiling round | 47 | 7 | 7 |
| 4 — parity round | 48 | 3 | 10 |
| 5 — offense round | 49 | 4 | 8 |

(The four remaining "arrow wins" after round 5 are all within a few percent
and oscillate around the equal margin between runs; every structural loss
is gone, and both single-column sorts became outright wins.)

---

## Round 1 — fix what the findings document already diagnosed

**1. Stop comparing strings against nothing.**
*What was slow:* `LIKE '%abc%'` splits into a prefix, middle, and suffix —
and for this pattern the prefix and suffix are *empty*. The kernel still
compared every row against those empty strings, and on Zen 5 each of those
"compare nothing" calls tripped a rare-case fallback inside the CPU that
costs ~250 cycles (the full story is Finding 1 in the findings doc).
*The fix:* two `is_empty()` checks, so the pointless comparison never runs.
*Who it affects:* `cmp::like` with `%...%`-style patterns. Same results.
*Result:* 108.6 ms → 13.9 ms.

**2. Don't allocate 40 MB to return 4 bytes.**
*What was slow:* reductions like `sum` declared "my output could be as big
as the input," so summing 10m integers allocated and zeroed a 40 MB buffer
to hold one number.
*The fix:* reductions now declare an output of exactly one element.
*Who it affects:* `compute::sum/min/max/product/argmin/argmax` — their
memory use drops from proportional-to-input to constant.
*Result:* all reduction rows went from ~8× slower than arrow to a dead tie
(2.5 ms → 0.29 ms, which is the speed of just reading the data).

**3. Stop writing every output twice.**
*What was slow:* every kernel first filled its output buffer with zeros,
then overwrote the zeros with real results — a full extra pass over memory.
*The fix:* outputs start uninitialized; only bytes the kernel actually
wrote are ever handed out.
*Who it affects:* every operator with a numeric output. One caveat worth
knowing: a buggy kernel that wrote less than it claimed used to produce
silent zeros, and would now produce visible garbage — the rule ("write
what you claim") is the same, but breaking it is now loud.
*Result:* `neg` on floats went from 0.61× to a tie; `cast(i32→i64)`
flipped to a 1.69× **win**.

**4. Ask "ascending or descending?" once, not a million times.**
*What was slow:* the sort re-checked the sort direction inside every single
comparison.
*The fix:* build two comparators (one per direction) and pick once.
*Who it affects:* `sort::sort_to_indices`; output identical.
*Result:* u64 sort 0.52× → 0.70×.

**5. Send simple LIKE patterns to the fast kernel that already existed.**
*What was slow:* `LIKE 'abc%'` ran a slow per-row closure even though the
crate already had a compiled starts-with kernel.
*The fix:* the pattern compiler now routes prefix/suffix patterns to that
kernel (and a bare `%` just returns all-true).
*Who it affects:* `cmp::like` with a wildcard at one end.
*Result:* `'abc%'` 0.25× → **1.19× win**; `'%xyz'` 0.28× → **1.41× win**.

## Round 2 — encoded arrays, and a filter that uses the hardware

**6. Decode run-encoded arrays before filtering or heavy takes.**
*What was slow:* run-end encoding stores "value X repeats N times." That's
great when one answer covers a whole run, but a filter mask differs on
every row, so the kernel was doing per-row work through the encoding —
the worst of both worlds.
*The fix:* `filter` on encoded input now decompresses first, then runs the
fast dense filter. `take` decodes only when there are enough indices to
justify it (sparse lookups still use the encoding).
*Who it affects:* `select::filter` / `select::take` on run-end encoded
input. Trade-off: decoding materializes a temporary dense copy of the
array during the call.
*Result:* `filter(ree)` 0.54× → **3.9× win**; `take(ree)` 0.70× →
**35.7× win**.

**7. Teach the compiler AVX-512's "filter instruction."**
*What was slow:* the dense filter checked one element at a time — about 9
instructions per element.
*The fix:* modern AVX-512 CPUs have an instruction (`vpcompressd`) that
takes 16 values plus a yes/no mask and writes just the selected ones,
packed together. The JIT now compiles filter loops down to it: 16 elements
per ~9 instructions instead of 9 instructions per element.
*Who it affects:* `select::filter` on AVX-512 machines. Machines without
AVX-512 (the M4, the Meteor Lake box) keep the old loop, unchanged.
*Result:* dense `filter(i32)` 0.78× → **1.83× win** — and the decoded
filters from fix 6 run through this same loop.

## Round 3 — what per-row profiling turned up

We profiled both sides of every remaining loss with `perf`. Four small
mechanical sins showed up, each worth real time.

**8. Let a tiny function be inlined.**
A one-line width lookup was being called — as an actual function call —
once per row inside the string iterator, eating 11–13% of the string
benchmarks. Adding `#[inline]` fixed it.

**9. Batch validity expansion for encoded arrays.**
`logical_nulls` on run-end data appended one little span per run; now
consecutive valid runs merge into one big append (arrow's trick).
*Result (with 10):* 0.43× → 0.97×, a tie.

**10. Build result bitmaps 64 answers at a time.**
String predicates wrote their true/false answers one bit at a time —
each write touching memory. Now 64 answers are collected in a CPU register
and stored as one word.

**11. Sort numbers, not comparisons.**
*What was slow:* the sort compared (row-number, value) pairs through a
comparator function, and keeping the output deterministic (equal values
stay in row order) cost 40% extra.
*The fix:* encode each value so that plain integer ordering matches its
semantic ordering (works for signed ints and even floats), then glue the
row number into the low bits of the same integer. Sorting plain integers
needs no comparator, and the row-number bits break ties automatically —
the output is bit-for-bit identical to before.
*Who it affects:* `sort::sort_to_indices`, all numeric types. Descending
order flips the encoded bits; determinism is preserved everywhere.
*Result:* u64 sort 16.9 → 13.6 ms.

**12–13. Two register tricks.**
The sort's null-handling pass now scans the validity bitmap a word at a
time instead of a row at a time (12). And vectorized loops used to bounce
their output pointer through memory on every block because the optimizer
couldn't prove it safe to keep in a register — a stack-slot trick makes it
safe (13). No semantic changes anywhere.
*Result:* nullable sort 11.4 → 9.0 ms; `concat` reached a tie.

**14. Copy strings without a detour.**
*What was slow:* appending each output string crossed from JIT code into a
Rust helper function — a per-row toll of function-call overhead and
bookkeeping.
*The fix:* the output buffer's pointer/length/capacity are visible to the
JIT now, so the capacity check and the byte copy happen inline; Rust is
only called when the buffer must grow.
*Who it affects:* every operator producing string output (casts, filters,
takes, concats of strings).
*Result:* `cast(dict→utf8)` 8.1 → 6.6 ms.

## Round 4 — the parity round

**15. Read string arrays the way arrow does.**
For plain string arrays, the predicate loop now indexes the offsets buffer
directly instead of going through a buffered iterator and two layers of
closures. (Encoded string layouts keep the iterator.)
*Result:* `like('%abc%')` reached a **tie**; `contains` became a
**1.05× win**.

**16. Gather bits into registers.**
`take` on a boolean array collected output bits one at a time, each one a
read-modify-write to memory that the CPU must finish before starting the
next. Plain boolean takes now gather 64 bits into a register word and
store once. *Result:* a tie.

**17. Replace comparison sorting with counting.**
*What was slow:* even with packed keys (fix 11), a comparison sort does
n·log(n) work; the last 15% vs arrow was the price of the extra tie-break
bits in every comparison.
*The fix:* a radix sort — sort a million keys by making a few passes that
*count* values into buckets instead of comparing pairs. Radix passes
preserve input order for equal keys, which is exactly what keeps the
deterministic tie-break intact (a dedicated test proves the output
identical to the comparison sort).
*Who it affects:* `sort_to_indices` above 50k rows; uses one temporary
buffer the size of the key array.
*Result:* u64 sort 13.6 → 11.7 ms — **a tie with arrow, determinism
kept**. The 40% determinism premium this story started with is now zero.

## Round 5 — the offense round

The last rows were near-parity; these fixes aimed past parity. Two hit,
one taught us something, and one turned out to be inert.

**18 + 19. Faster null separation, fewer radix passes.**
The sort now builds its valid-rows and null-rows lists in a single sweep
of the validity bitmap (valid bits pack keys, cleared bits record nulls,
list pre-sized), and the radix sort uses 11-bit digits instead of 8-bit —
6 counting passes over the data instead of 8, 25% less memory traffic.
*Who it affects:* `sort_to_indices`; output still bit-identical.
*Result:* the nullable sort went 8.2 → **4.70 ms, a 1.34× win**, and the
u64 sort went 11.7 → **9.51 ms, a 1.25× win**. Both sorts now beat arrow
outright while staying deterministic — the "determinism premium" this
story opened with ended as a determinism *discount*.

**20. Skip the cache for huge outputs — implemented, but inert for now.**
*The idea:* a normal store first *reads* the target cache line before
overwriting it; for a 40 MB output that read is pure waste, and
"non-temporal" stores skip it. Kernels with outputs over 32 MB now compile
a separate variant requesting non-temporal stores.
*What actually happened:* checking the emitted machine code showed the CPU
instructions never materialized — 512-bit non-temporal stores require
64-byte-aligned buffers, ours guarantee only 16, and LLVM silently falls
back to regular stores. The machinery is correct and stays in place;
activating it needs 64-byte-aligned output allocation (future work).
*Result:* `neg_wrapping(i32)` unchanged at 0.89× — which paired
measurements show is a statistical tie with arrow anyway.

**21. Huge pages: not a tweak, a load-bearing wall.**
`concat` was randomly 12% slower on some runs, which we suspected was
2 MB "huge page" luck on its fresh 40 MB output buffer. Large allocations
now explicitly request huge pages (Linux only, best effort). Testing the
opposite advice settled how much this matters: *forbidding* huge pages
made concat 2.2× slower (2.9 → 6.5 ms) and neg 2× slower — large-output
kernels on this machine live or die by them. With the advice in place,
concat runs a steady 2.90 ms instead of coin-flipping between 2.65 and
2.96; the residual few percent against arrow's 2.65–2.70 is within both
sides' run-to-run drift.
*Who it affects:* any operator with a multi-megabyte output.

## Where it ended

After five rounds: **49 wins, 8 equals, and 4 nominal losses** that are
all within a few percent and trade places with "equal" from run to run —
`concat` 0.93×, `cast(dict→utf8)` 0.93×, `logical_nulls(ree)` 0.96×
(a microseconds-scale benchmark), and `neg_wrapping(i32)` 0.89× (an exact
tie when measured back-to-back in one session). Of the twenty original
losses, none survive as a structural deficit, and the fixes brought new
headline wins along the way: `take(ree)` at 35×, `filter(ree)` at 3.9×,
the dense filter at 1.8×, and both sorts at 1.25–1.34× with bit-identical
deterministic output.
