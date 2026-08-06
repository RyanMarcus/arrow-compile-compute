# Cross-platform benchmark findings: Apple M4 vs Intel Meteor Lake

Comparison of two `benches/report.py` runs of the same benchmark suite on different
machines. The llvm-vs-arrow speedup ratios turn out to be strongly
architecture-dependent, in ways that are attributable to specific causes rather than
noise.

## Setup

| | Mac | Linux |
|---|---|---|
| CPU | Apple M4, 10 cores | Intel Core Ultra 7 155H (Meteor Lake), 22 threads |
| Memory | 24 GB unified | 62 GB DDR5 |
| OS | macOS (Darwin 25.5.0), arm64 | Arch Linux, x86_64 |
| SIMD | NEON (baseline for aarch64) | AVX2 available in hardware; **no** AVX-512 |
| Revision | `a9a2fc8` (dirty tree) | `ee09ec4` (clean tree) |

Caveat: the two runs are from nearby but not identical revisions, and the Mac run had
uncommitted changes. 54 benchmarks matched by name and are compared below; the
conclusions are directional until both machines re-run at one commit.

A fact that turns out to drive several findings: **the JIT and the arrow baseline are
not compiled the same way.** The JIT passes the real host CPU and feature set to LLVM
(`get_host_cpu_name()` / `get_host_cpu_features()`), so it emits AVX2 on the Intel
machine. The ahead-of-time Rust build does not: with `RUSTFLAGS` unset and no cargo
config, `rustc --print cfg` on the Linux box reports a baseline of **`sse, sse2`
only** — arrow-rs was never allowed to emit AVX2 there. On aarch64 this asymmetry does
not exist, because NEON is part of the baseline.

## Finding 1 — The JIT's reductions collapse on x86 (6× loss)

| benchmark (10m rows) | Mac llvm | Mac arrow | Linux llvm | Linux arrow |
|---|---|---|---|---|
| `compute::sum(i32)` | 1.11 ms | 572 µs | **6.49 ms** | 1.04 ms |
| `compute::min(i32)` | 1.01 ms | 569 µs | **6.65 ms** | 3.37 ms |
| `compute::product(i32)` | 1.00 ms | 560 µs | **6.96 ms** | 4.92 ms |

The loss goes from ~1.9× on the Mac to ~6× on Intel, and all three reductions land at
the same ~6.5–7 ms (≈2 cycles/element — the signature of a serial scalar loop).

**Root cause (proven — and it is NOT codegen).** The original hypothesis (scalar
reduce loop, auto-vectorized only on aarch64) was disproven by dumping the JIT's
output on both machines (`ACC_DUMP_IR=<dir>` hook in `optimize_module`): the reduce
loop auto-vectorizes *superbly* on both targets — four parallel NEON `add.4s`
accumulators on the M4, four parallel AVX2 `vpaddd` ymm accumulators (32 i32/iter)
on Intel, guarded only by trip-count checks.

The real culprit is the **output writer**: `ReductionKernel` declares its output as
`"<= n"`, so every call allocates a buffer sized to the *input* (40 MB for 10m rows)
to hold a **one-element result**. On Linux, glibc returns large freed buffers to the
OS, so each call re-faults the entire region and the kernel zero-fills 40 MB —
measured at **568 minor faults per call**. Evidence chain: pinning ruled out E-cores
(6.5 ms on a P-core); summing the same 40 MB as ten 1m slices ran 2.4× faster
(allocator recycles small buffers — 15 faults); arrow allocates nothing (0 faults,
1.05 ms). macOS masks the bug entirely because its allocator retains and reuses the
large region (one-shot ≈ sliced, 0 faults) — which is why Finding 1 originally looked
architecture-dependent.

**Action.** Give reduction kernels a constant-size output (the `SizeTerm` grammar
needs a numeric-constant variant; then `reduction.rs` uses capacity 1 instead of
`"<= n"`). After that fix a smaller (~2×) loop-throughput gap vs arrow's sum remains
on both machines (JIT streams ~14 GB/s on Intel / ~32 GB/s on M4 vs arrow's
38/61 GB/s) — worth a second look at the reduce loop's load pattern, but it is
second-order next to the allocation bug.

## Finding 2 — Plain comparisons win far bigger on Intel because *arrow* runs handicapped there

| benchmark | Mac llvm | Mac arrow | Mac ratio | Linux llvm | Linux arrow | Linux ratio |
|---|---|---|---|---|---|---|
| `cmp::lt(i32, i32 scalar)` 1m | 45.7 µs | 121 µs | 2.66× | 58.3 µs | **396 µs** | **6.80×** |
| `cmp::bounds(i32, scalars)` 1m | 66.2 µs | 231 µs | 3.49× | 79 µs | **805 µs** | **10.2×** |
| `cmp::lt(i32, i32)` 10m | — | — | 1.09× loss | — | — | **1.67× win** |

The JIT side barely moves between machines; the **arrow side slows ~3×** on Intel.

**Cause (confirmed).** The compilation asymmetry above: arrow's comparison loops are
compiled for generic x86-64 (SSE2, 128-bit) while the JIT emits AVX2 (256-bit) on the
same machine. On the Mac both sides get NEON, so the Mac ratio reflects codegen
quality alone, and the Linux ratio additionally reflects AOT-vs-JIT feature targeting.
This is, in fact, one of the crate's theses demonstrated: runtime compilation always
targets the actual machine; shipped binaries target the lowest common denominator.

**Action.** Re-run the Linux suite with `RUSTFLAGS='-C target-cpu=native'` to measure
arrow at its best. Both numbers are worth reporting — "arrow as shipped" vs "arrow
rebuilt for the host" answer different questions.

## Finding 3 — Encoded-input verdicts flip toward LLVM on Intel

| benchmark (10m rows) | Mac | Linux |
|---|---|---|
| `arith::neg_wrapping(dict i8→i32)` | 2.3× **loss** (11.2 vs 4.98 ms) | 1.76× **win** (8.86 vs 15.6 ms) |
| `cmp::lt(i32, i32)` | 1.09× loss | 1.67× win |
| `select::concat(i32 x5, dict x5)` | 1.15× loss | 1.25× win |

**Cause (confirmed, same root as Finding 2).** Arrow's path for encoded input is
decode-then-compute; the decode is a gather (`take`) loop that is hard to
autovectorize and, at the SSE2 baseline, runs ~3× slower on Intel (4.98 → 15.6 ms).
The JIT reads through the encoding and got AVX2, so it improved slightly on the same
hardware. The encoded-input argument for the JIT is therefore *stronger* on x86.

## Finding 4 — The Mac's memory bandwidth flatters the flagship JIT wins

| benchmark | Mac ratio | Linux ratio | what moved |
|---|---|---|---|
| `vec::norm(f32[768])` 16384 rows | 6.95× | 1.89× | llvm 2.99 → 12.3 ms; reference ~flat |
| `logical_nulls(dict i8→i32)` 100m | 7.37× | 2.36× | llvm 54 → 172 ms; arrow ~flat (399 → 407 ms) |
| `cmp::lt(utf8, utf8 scalar)` 1m | 2.86× | 1.55× | arrow *improved* 3.49 → 2.07 ms; llvm flat |

**Causes (one per row).**
- **norm** (hypothesis): the kernel is a per-row *horizontal* reduction (sum of squares
  across 768 lanes). aarch64 has fast across-lane reduce instructions; x86 legalizes
  wide horizontal reductions into slow shuffle chains. At 12.3 ms the Intel run is
  compute-stalled (~4 GB/s), consistent with shuffle-chain legalization rather than
  bandwidth. Same fix family as Finding 1: accumulate vertically, one final horizontal
  step.
- **logical_nulls** (hardware): 100M data-dependent lookups + bit writes — scalar on
  both machines and on both engines. The ratio collapses toward raw memory-subsystem
  quality, which is the M4's strength (~2× effective single-thread bandwidth, deeper
  load out-of-order). Arrow's absolute time barely moved; ours tripled. No code bug —
  the M4 was flattering us, not arrow.
- **utf8 lt** (medium confidence): Rust slice comparison bottoms out in `memcmp`, and
  glibc dispatches `memcmp` to a hand-tuned AVX2 implementation at *runtime* — its own
  CPU detection, unaffected by the rustc SSE2 baseline. Apple's short-string `memcmp`
  path evidently does less well at these 5–15-byte strings. The JIT byte-loop is flat
  on both.

## Cross-cutting caveat: hybrid cores

The 155H mixes fast P-cores with much slower E-cores, and an unpinned single-threaded
benchmark can migrate during a ~30-minute run. The near-bandwidth `cmp::lt` numbers
prove the JIT rows executed on P-cores, so no finding above depends on this — but
future Linux runs should be pinned (`taskset -c <p-core>`) to remove the variance.

## Update: native-build runs (`RUSTFLAGS='-C target-cpu=native'`)

Both machines re-ran the suite with arrow (and all AOT Rust code) rebuilt for the
host CPU, directly testing Finding 2's cause. Results:

- **Finding 2 confirmed.** On Intel, native-built arrow's comparison loop went
  396 → 77 µs (1m scalar) and 8.69 → 1.2 ms (10m) — the predicted ~5× AVX2
  unlock. The M4 control was flat (median shift 1.006), as predicted, since
  NEON is already the aarch64 baseline.
- **Sharper than expected: native arrow *overtakes* the JIT on plain dense
  comparisons at DRAM scale.** Four Linux rows flip to arrow wins: i32
  array-array and array-scalar at 10m rows, f32 array-array, and i64
  array-scalar. The overall Linux scoreboard evens out (34–25 for LLVM on the
  default build → 29–29 native). Cache-resident plain comparisons (1m i32)
  stay LLVM wins, barely (1.31×).
- **The encoded-layout wins are durable**: run-end rows barely move
  (6.8× → 6.2×; ree-of-dict 7.8× → 7.2×) — materialization cost, not
  instruction selection, and no build flag removes it.
- **Finding 1 worsens**: against native arrow, the scalar `Reduce` loop loses
  6–7× (min: 945 µs arrow vs 6.43 ms llvm at 10m).
- **New finding — JIT i64 comparison weakness on x86**: the JIT compares 1m
  i32 in 59 µs but 1m i64 in 322 µs (5.5× for 2× the data); on the M4 the
  same pair scales proportionally (46 → 87 µs). Native arrow does the i64 case
  in 114 µs. Something in the x86 lowering of the 64-bit comparison loop
  (likely the bit-packed output path) deserves a look.
- **String predicates** (benchmarked on all four datasets): the `like`
  prefix/suffix loss is universal and build-independent (3–6×), confirming a
  per-row-overhead cause rather than SIMD; meanwhile the JIT `ends_with`
  kernel *wins* on Linux and ties on M4. Routing prefix/suffix-shaped LIKE
  patterns to the existing `StringStartEndKernel` looks like a cheap, large
  win.

Framing for the paper: against arrow **as shipped** (the realistic default —
`target-cpu=native` is rarely set), the JIT wins broadly on x86. Against arrow
**at its best**, the JIT's durable advantages narrow to encoded layouts,
cache-resident predicates, and fused operations — which is the honest version
of the runtime-compilation argument: the JIT gets both the machine's full
instruction set *and* layout specialization for free, while AOT code must
choose portability or performance at build time.

## Recommended actions, in order

1. ~~Rebuild + re-run on Linux with `RUSTFLAGS='-C target-cpu=native'`~~ — **done**;
   see the update section above. Both build flavors are published on the results
   site per machine.
2. **Fix the reduction output allocation** (see the proven root cause under
   Finding 1): add a numeric-constant `SizeTerm` and allocate capacity 1, not
   `"<= n"`. A vectorized `Reduce` lowering is NOT needed — the auto-vectorizer
   already produces optimal loops on both targets.
3. **Investigate the x86 i64 comparison lowering** (new finding above).
4. **Route prefix/suffix LIKE patterns to `StringStartEndKernel`** instead of the
   per-row closure path.
5. **Pin benchmark runs to a P-core** on hybrid-core machines.
6. Re-run both machines at a single clean revision.
