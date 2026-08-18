# Zen 5 benchmark findings: why arrow wins on the AMD machine

Follow-up to `cross-platform-findings.md`, covering the third machine
(`benchmark-results-zen5-native.json`). Scope: the 22 benchmarks where the
arrow baseline beats the JIT on AMD, and why.

> **Status:** this document is the *diagnosis*, frozen at the revision it
> analyzes. Every recommended action below has since been implemented,
> along with a dozen further fixes the follow-up profiling uncovered — see
> `fix-rounds.md` for the full remediation story. After those rounds the
> Zen 5 scoreboard went from 38 wins / 20 losses / 2 equal to 49 / 4 / 8,
> with no loss remaining beyond a few percent. The loss table below
> describes the pre-fix code.

**How to read the ratios.** A ratio in this document is the JIT's speed as a
multiple of arrow's — `arrow ms ÷ llvm ms`, the same number the benchmark page
reports. `0.10×` means the JIT ran at a tenth of arrow's speed; anything above
`1.0×` is a JIT win.

## Setup

| | value |
|---|---|
| CPU | AMD Ryzen 7 9700X (Zen 5), 8 cores / 16 threads, 32 MB L3 |
| Memory | 123 GB |
| OS | Arch Linux (kernel 7.1.6-arch1-1), x86_64 |
| SIMD | full AVX-512 (F/DQ/BW/VL/VBMI/VBMI2/VNNI/BITALG/VPOPCNTDQ/VP2INTERSECT) |
| Revision | `107db50`, **dirty tree** |

Scoreboard: **37 llvm wins, 22 arrow wins, 1 equal** out of 60 paired
benchmarks.

Two caveats about the run itself:

- **The build really was AVX-512-native.** The run's manifest does not record
  `rustflags` — `report.py` only started writing that field after this run —
  so the build flavour was verified by disassembling the benchmark binary the
  run used: it contains 30,361 references to `zmm` (AVX-512) registers, 506
  `kmov` mask-register moves and 208 `vpternlog` instructions, while the
  sibling binary from the previous non-native build contains none.
- **The tree was dirty.** The manifest records `"dirty": true`, so the tree
  carried uncommitted changes on top of `107db50`. Nothing below depends on a
  specific working-tree state, but the run is not reproducible from the
  revision alone.
- **The published results have since been refreshed.** The
  `benchmark-results-zen5-native.json` on the site now comes from a clean
  re-run at `6fa1f69` (scoreboard 38 llvm / 20 arrow / 2 equal — the
  difference from the 37 / 22 / 1 above is drift on near-parity rows, e.g.
  `%abc%xyz%` moved from 0.96× to 1.07×). The analysis in this document is of
  the `107db50` run; its headline losses persist unchanged in the refreshed
  run (`%abc%` 0.09×, `sum` 0.12×, both `neg_wrapping`s within 0.02×).

## Every losing row, and what causes it

Six mechanisms — Findings 1 through 6 below — account for 19 of the 22 losing
rows. Only **one** of the six is specific to this machine (Finding 1). The
rest are fixed costs of ours that every machine pays; Zen 5 merely makes them
visible, because native arrow is much faster here than on the other two
machines, so those overheads stopped hiding behind other work.

| benchmark | ratio | cause | AMD-specific? |
|---|---|---|---|
| `cmp::like('%abc%')` | 0.10× | AVX-512 masked-load assist (F1) | **yes** |
| `compute::max/product/min/sum(i32)` | 0.11–0.12× | output zero-fill (F2) | no |
| `cmp::like('abc%')` | 0.23× | per-row callback overhead (F4) | no |
| `logical_nulls(ree)` | 0.26× | per-element REE walk (F5) | no |
| `cmp::like('%xyz')` | 0.28× | per-row callback overhead (F4) | no |
| `sort_to_indices(nullable u64)` | 0.44× | comparator (F3) + null pre-pass | no |
| `sort_to_indices(u64)` | 0.51× | comparator shape (F3) | no |
| `select::filter(ree, bool)` | 0.53× | branch mispredicts on REE (F5) | no |
| `arith::neg_wrapping(f64)` | 0.61× | output zero-fill (F2) | no |
| `arith::neg_wrapping(i32)` | 0.64× | output zero-fill (F2) | no |
| `select::concat(i32 x10)` | 0.67× | output zero-fill (F2) | no |
| `select::take(ree, u64)` | 0.70× | binary search per index (F5) | no |
| `select::filter(i32, bool)` | 0.74× | instruction count (F6) | no |
| `cast(dict(i32,utf8) → utf8)` | 0.79× | per-row callback overhead (F4) | no |
| `cmp::contains(utf8)` | 0.80× | per-row callback overhead (F4) | no |
| `cmp::lt(i32, i32 scalar)` 10m | 0.81× | **not investigated** | no |
| `cast(i32 → i64)` | 0.90× | output zero-fill (F2) | no |
| `select::take(bool, u64)` | 0.91× | **not investigated** | no |
| `cmp::like('%abc%xyz%')` | 0.96× | catch-all `match_like`, near parity | no |

The three uninvestigated rows are the tail: `cmp::lt(i32, scalar)` and
`select::take(bool, u64)` follow an x86-native pattern rather than an AMD one —
both are also arrow wins on Meteor Lake's native build (0.71× and 0.75×) while
staying LLVM wins on the M4 (1.78× and 1.11×), i.e. the "native arrow overtakes
on dense work" effect already recorded in `cross-platform-findings.md`. The
`%abc%xyz%` row is within 5% of parity on all three machines.

Fixing F2 turns four *measured* rows into ties or wins (`sum`, both
`neg_wrapping`s, and `cast(i32 → i64)`); `min`/`max`/`product` share `sum`'s
mechanism and should follow, which would be seven of the eight F2 rows.
`concat` is the exception — it stays a loss at 0.94× (see Finding 2).

## Finding 1 — `cmp::like('%abc%')` is 6.8× slower *only* on Zen 5 (AMD-specific)

**The short version:** for `%abc%`-shaped patterns the kernel compares every
row against an *empty* prefix and an *empty* suffix. Each comparison compiles
to a real `bcmp` call whose second pointer is unmapped, and on Zen 5 the
AVX-512 code inside glibc's `bcmp` pays a ~250-cycle microcode penalty for
exactly that call. A comparison that does no work at all becomes the most
expensive thing in the kernel.

| | llvm | arrow | ratio |
|---|---|---|---|
| Zen 5 | **104.69 ms** | 10.11 ms | **0.10×** |
| Meteor Lake | 17.84 ms | 12.25 ms | 0.69× |
| Apple M4 | 16.84 ms | 9.83 ms | 0.58× |

Our absolute time is 6× worse on AMD than on the other two machines, while
arrow's is *better* there. Two rows in the same bench file do the same work
with the same `memchr` `Finder` over the same data — `cmp::contains` takes
12.56 ms, `cmp::like('%abc%')` takes 104.69 ms.

**Root cause.** `perf` puts **71.4%** of the benchmark inside one libc
instruction, reached via `equal_same_length<u8,u8>` ← `[u8] == Vec<u8>` ← the
LIKE closure. The chain from SQL pattern to stalled pipeline:

1. `%abc%` lowers to the `(2, false)` branch of `compile_string_like`
   (`string.rs:359`) with `prefix = []`, `infix = "abc"`, `suffix = []` — the
   text before the first `%` and after the last one, both empty for this
   pattern.
2. The per-row closure evaluates `b[..prefix.len()] == prefix` and
   `b[b.len() - suffix.len()..] == suffix` (`string.rs:380-381`). Rust's
   `core` has no zero-length short circuit for slice equality, so comparing
   against an empty slice still emits a real library call: `bcmp(p, q, 0)`.
3. An empty `Vec<u8>` never allocates, so its `as_ptr()` is the dangling
   sentinel `0x1` — an address no page is mapped at.
4. glibc picks its `bcmp` implementation per CPU (IFUNC dispatch). On Zen 5 it
   selects `__memcmp_evex_movbe`, an AVX-512 implementation whose short-length
   (`len <= 32`) path is:

   <!-- raw: AVX-512 mask register syntax would otherwise parse as a Liquid tag -->
   {% raw %}
   ```asm
   bzhi     %edx,%eax,%eax              ; len -> mask (0 here)
   kmovd    %eax,%k2
   vmovdqu8 (%rsi),%ymm18{%k2}{z}       ; 32-byte masked load from 0x1
   vpcmpnequb (%rdi),%ymm18,%k1{%k2}    ; 71% of all samples land here
   ```
   {% endraw %}

   It turns the length into a per-byte lane mask and issues one *masked*
   32-byte load — masked-off lanes are guaranteed not to fault, which is what
   lets glibc read a 32-byte window it only partly owns. With `len = 0` the
   mask is all-zero: a 32-byte load from `0x1` with every lane masked off.
5. That load is architecturally legal — AVX-512 promises fault suppression for
   masked-off lanes — but on Zen 5, honouring the promise is expensive. The
   core resolves the suppressed fault with a **microcode assist**: it cancels
   the instruction mid-flight, flushes the pipeline, and completes the load in
   an on-chip microcode routine (measured below).

Measured cost of a zero-length `bcmp`, varying only the second pointer:

| second pointer | Zen 5 | Meteor Lake |
|---|---|---|
| resident page, load fits | **1.17 ns** | 4.74 ns |
| resident page, 8 B before page end | 47.59 ns | — |
| `PROT_NONE` page | 58.95 ns | — |
| unmapped `0x1000` | 59.14 ns | — |
| empty `Vec` dangling `0x1` | **59.21 ns** | 4.73 ns |
| mapped, len 3 / 16 / 64 | ~1.2 ns | 4.6–6.3 ns |

The second row is the informative one: that pointer is valid, resident and
readable — it merely sits 8 bytes before the end of its page, so the 32-byte
window runs off the mapping. The penalty is therefore a property of the
**page**, not the pointer value: any masked AVX-512 load whose 32-byte
footprint touches an unreadable page costs ~250 cycles, a 50× penalty, even
when the mask suppresses every element. Two such calls per row × 1m rows
≈ 90 ms of the 105 ms.

### That it is an assist, measured

Running 20m zero-length `bcmp` calls twice — identical code, the second
pointer mapped in one process and unmapped in the other. The two Zen counters
here: `ex_ret_ucode_ops` counts retired micro-ops that came out of the
microcode sequencer rather than the ordinary decoders, and
`bp_redirects.resync` counts retire-time pipeline restarts.

| counter | mapped | unmapped | per call |
|---|---|---|---|
| `instructions` | 540,291,958 | 540,293,675 | **identical** |
| `cycles` | 100,533,986 | 5,105,620,762 | 5.0 → **255** |
| `ex_ret_ucode_ops` | 41,583 | 320,791,099 | → **+16.0** |
| `bp_redirects.resync` | 100 | 59,996,861 | → **+3.0** |

The architectural instruction stream is the same to within 1,717 instructions
out of 540 million, so nothing about the *code* changed — only the page the
pointer names. Each slow call retires ~16 extra microcode ops and triggers ~3
retire-time pipeline resyncs. Aborting the instruction, flushing the pipeline,
and running a microcode routine to complete it is exactly what an assist is.

Two caveats on the numbers. `ex_ret_ucode_instr` barely moves (546 → 2,971)
while `ex_ret_ucode_ops` explodes, so the injected ops are not being attributed
to a retired *microcoded instruction* — consistent with an assist replaying the
instruction rather than the instruction itself decoding to microcode. And the
cycle figure fixes an earlier estimate: 255 cycles/call at the 4.30 GHz this
loop actually ran at, not the ~300 implied by assuming max boost.

Meteor Lake has no AVX-512, so glibc there selects `__memcmp_avx2_movbe`,
which never issues a masked load — hence no penalty on Intel, and none on the
M4.

**Confirmation.** Forcing glibc to skip its AVX-512 memcmp
(`GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX512VL`) takes `%abc%` from 102.37 ms to
**15.47 ms** while arrow is unchanged (10.25 → 10.03 ms). Sweeping every string
benchmark under both dispatch modes, `%abc%` is the *only* one affected
(6.82×); `'abc%'` and `'%xyz'` are 0.76×/0.83× — i.e. the EVEX memcmp is a
**win** at normal lengths. The pathology is exclusively the zero-length call
against an unmapped pointer.

**Fix.** Do not emit zero-length comparisons. Guarding both sides —

```rust
&& (prefix.is_empty() || b[..prefix.len()] == prefix)
&& (suffix.is_empty() || b[b.len() - suffix.len()..] == suffix)
```

— is semantics-preserving (comparing against an empty slice is always true) and
takes `%abc%` from **102.4 ms to 14.35 ms**, verified on the machine; the
bench's own `assert_eq!(arrow, llvm)` still passes. That moves the row from
0.10× to 0.70×.

### Which LIKE shapes the guard actually helps

Benchmarking four extra pattern shapes, paired before/after in one session
(1m rows, short-string dataset; times in ms):

| pattern | branch | before | guarded | arrow | effect |
|---|---|---|---|---|---|
| `%abc%` | `(2,false)`, both empty | 102.90 ms | **13.93 ms** | 10.48 ms | **7.2×** |
| `%abc%xyz` | `(2,false)`, prefix empty | 59.99 ms | **4.56 ms** | 10.86 ms | **12.4×** |
| `abc%xyz%` | `(2,false)`, suffix empty | 4.36 ms | 4.43 ms | 8.22 ms | unchanged |
| `a%bc%z` | `(2,false)`, neither empty | 4.93 ms | 4.94 ms | 8.84 ms | unchanged |
| `abc%` | `(1,false)` prefix | 3.46 ms | 3.49 ms | 0.86 ms | unchanged |
| `%xyz` | `(1,false)` suffix | 3.58 ms | 3.64 ms | 1.02 ms | unchanged |
| `%` | `(1,false)`, empty needle | 3.71 ms | 2.85 ms | 0.02 ms | 1.3× |
| `%abc%xyz%` | catch-all `match_like` | 17.73 ms | 17.80 ms | 18.83 ms | unchanged |

Two things this makes precise.

**The cost is exactly the number of dangling-pointer `bcmp` calls actually
executed per row.** `%abc%` evaluates both (≈102 ms), `%abc%xyz` evaluates one
(≈60 ms), and `abc%xyz%` evaluates *none* in practice — its non-empty prefix
`"abc"` is checked first and fails for all but ~1/17,576 of the rows, so `&&`
short-circuits before reaching the empty-suffix comparison. Ordering, not just
emptiness, decides whether a pattern is affected.

**`%abc%xyz` is the biggest single win in the suite** — 12.4×, flipping a
0.18× loss into a 2.4× win over arrow. It is not in the benchmark set today and
is worth adding, since `%infix%suffix` is a common shape.

Note that argument order does *not* matter: `bcmp(dangling, mapped, 0)` and
`bcmp(mapped, dangling, 0)` cost 57.8 ns and 61.0 ns respectively, so either
operand landing on an unreadable page is enough.

The `starts_with`/`ends_with` sites at `string.rs:237`/`246` were *expected* to
carry the same hazard for a bare `%` pattern, but do not: profiling `'%'` shows
the closure itself at 29.8% and **no libc memcmp frame at all** — LLVM inlines
that comparison instead of emitting a call. Guarding them is a 1.3× cleanup,
not an assist fix. That is an optimizer accident rather than a structural
guarantee, so the guard is still worth having, just not urgent.

More generally: on Zen 5, any slice compared against an empty `Vec` in a hot
loop is a ~60 ns landmine whenever it compiles to a real `bcmp` call, and so is
any AVX-512 masked load near the end of an allocation.

## Finding 2 — One `resize(.., 0)` accounts for *all* of the reduction and neg losses

**The short version:** every kernel invocation allocates its output buffer
with a `resize` that also writes a zero over every byte, before the kernel
writes its real output. A reduction that returns 4 bytes first allocates and
zeroes 40 MB. Deleting the zero-fill closes the entire gap — this is the
highest-value fix in the document, and it is not AMD-specific.

`PrimitiveWriter::allocate` (`compiled_writers/primitive_writer.rs:135`) does

```rust
pwr.alloc.resize((self.pt.width() * size).div_ceil(16), 0);
```

which allocates the output buffer **and memsets it to zero** on every call.
`RunnableDSLFunction::run` allocates fresh output buffers per call
(`dsl2/runtime.rs:69-79`), so this happens per invocation, not once.

(An earlier draft of this document blamed `DSLBuffer::new` in `dsl2/buffer.rs`.
That type has the same pattern but is not on this path; the kernels here reach
the writer through `WriterSpec::allocate`.)

Reductions declare their output as `"<= n"` (`reduction.rs:129`), so
`compute::sum` over 10m i32 allocates and zeroes **40 MB in order to return
4 bytes**.

**Proof: delete the memset and the entire gap disappears.** Patching that one
line to allocate without zeroing, rebuilding, and re-running (the benches' own
`assert_eq!` against arrow still passes):

| 10m rows | stock | zero-fill removed | arrow | before | after |
|---|---|---|---|---|---|
| `compute::sum(i32)` | 2.498 ms | **0.304 ms** | 0.295 ms | 0.12× | **0.97×** |
| `arith::neg_wrapping(i32)` | 4.114 ms | **2.441 ms** | 2.440 ms | 0.59× | **1.00×** |
| `arith::neg_wrapping(f64)` | 8.827 ms | **5.317 ms** | 5.334 ms | 0.60× | **1.00×** |

The `stock` column comes from this patch session, not from the published run;
`benchmark-results-zen5-native.json` has `sum` at 2.532 ms, `neg_wrapping(i32)`
at 4.096 ms and `neg_wrapping(f64)` at 8.799 ms. The LLVM side agrees to ~1%,
but arrow's `neg_wrapping(i32)` drifted more (2.440 ms here, 2.608 ms in the
published run), which is why the `before` ratio reads 0.59× here and 0.64× in
the table above. The paired before/after *within* a session is the meaningful
comparison.

All three land on top of arrow. There is no residual gap to explain, no
loop-throughput deficit, and nothing architecture-specific: the reduce loop and
the element-wise loops were already running at the machine's memory bandwidth.

**Why the two shapes differ in how much they gain.** The memset is not just a
memset — on a freshly `mmap`ed buffer, writing the zeroes is also what *faults
the pages in* (the first touch of each new page takes a page fault that maps
it).

- **Reductions never write their output** (one scalar, then truncate). Remove
  the memset and those 40 MB are never touched at all, so both the memset *and*
  the page faults vanish: 24,304 faults where the stock build took 870,122.
  Hence the 8.2× improvement.
- **Element-wise kernels write every output element**, so the pages fault
  regardless. Removing the memset only removes the redundant first pass:
  985,196 faults after the patch, against arrow's 1,041,596 for the same work.
  Hence the smaller 1.69× improvement, and hence the exact tie with arrow —
  both sides now do one allocation and one write.

Separating allocation from memset with `MALLOC_MMAP_THRESHOLD_=1G` (glibc
recycles the buffer instead of `mmap`/`munmap`ing it, so the pages stay
mapped between calls) gives the same picture from the other direction:

| 10m rows | stock, default | stock, recycled | no zero-fill |
|---|---|---|---|
| `compute::sum(i32)` | 2.498 ms | 1.029 ms | 0.304 ms |
| `neg_wrapping(i32)` | 4.114 ms | 2.773 ms | 2.441 ms |

A standalone 40 MB memset on this box costs 0.720 ms (55.5 GB/s) and a 40 MB
read-sum costs 0.300 ms (133.2 GB/s) — the latter being, to three digits,
arrow's sum time. Arrow is at the read-bandwidth limit, and so are we once the
redundant write is gone.

**Correction to an earlier reading.** A first pass measured the recycled
reduction at 1.83 ms and concluded a ~6× loop-throughput gap remained, to be
explained by L3 eviction. That number did not reproduce; the stable value is
1.03 ms, and the no-zero-fill result shows the loop was never the problem. The
L3-eviction story was fitted to a bad measurement and should be disregarded.

**Why the AMD ratio looks worse than Intel's (0.12× vs 0.14×):** our absolute
time *improved* on AMD (6.49 → 2.53 ms). Arrow improved more (0.94 → 0.29 ms),
because Zen 5's 32 MB L3 and AVX-512 make the scan bandwidth-bound. The ratio
worsened because arrow got faster, not because we got slower.

**It generalises.** Same patched build, same session:

| 10m rows | stock ratio | patched LLVM | patched arrow | patched ratio |
|---|---|---|---|---|
| `cast(i32 → i64)` | 0.91× | 4.797 ms | 7.863 ms | **1.64× win** |
| `select::concat(i32 x10)` | 0.65× | 2.906 ms | 2.728 ms | 0.94× |
| `select::filter(i32, bool)` | 0.74× | 4.765 ms | 3.730 ms | 0.78× |

`cast` flips from a loss to a 1.64× win and `concat` goes from 0.65× to roughly
even. `filter` barely moves (0.74 → 0.78), so its loss has a different cause
and still needs its own look (that cause is Finding 6).

**Caveat on the fix.** The patch used here (`reserve_exact` + `set_len`) was a
measurement instrument, not a shippable change: it exposes uninitialised
memory, which is fine when the kernel writes every element it hands back but
not in general — a writer that returns a buffer longer than it wrote would leak
uninitialised bytes into an Arrow array. A real fix wants either `MaybeUninit`
discipline in the writer, or a right-sized allocation. For reductions the
latter is the better answer anyway: give them capacity 1 instead of `"<= n"`,
which needs a numeric-constant variant in `SizeTerm` (`dsl2/resolver.rs:8`
currently has only `Term`, `Add` and `AtLeast`).

## Finding 3 — The sort comparator does two things arrow's doesn't

**The short version:** both sides run the identical compiled sort routine; the
whole gap lives in the comparison closure we hand it, which re-tests the
`descending` flag on every comparison and adds an index tie-break that arrow
doesn't have.

`sort::sort_to_indices(u64)` at 1m rows: 24.24 ms vs arrow 12.44 ms. Both
sides end up in the *same* monomorphization —
`core::slice::sort::unstable::quicksort::<(u32, u64)>`, one compiled instance
shared by both crates — so the algorithm and the tuple layout are identical.
The difference is the closure at `sort.rs:182`:

```rust
let cmp = T::Native::compare(*lhs_val, *rhs_val);
let cmp = if opts.descending { cmp.reverse() } else { cmp };   // runtime branch
cmp.then_with(|| lhs_idx.cmp(rhs_idx))                          // index tie-break
```

Sorting the same 1m `(u32, u64)` tuples with each variant in isolation:

| comparator | ms |
|---|---|
| this crate (branch + tie-break) | **23.68** |
| drop the tie-break only | 15.63 |
| drop the `descending` branch only | 16.54 |
| arrow's shape (branch hoisted, no tie-break) | **11.54** |
| `sort_unstable_by_key(\|x\| x.1)` | 11.03 |
| `(u64 value, u32 index)` + plain `sort_unstable()` | 16.32 |

23.68 / 11.54 reproduces the real 24.24 / 12.44 almost exactly. Arrow hoists
`descending` outside the sort into two monomorphizations and has no tie-break.

Note the tie-break is not free to delete: it makes our index output
deterministic, which arrow's `sort_unstable_by` does not guarantee. Keeping
determinism via a `(u64 value, u32 index)` layout and a plain `sort_unstable()`
costs 16.32 ms (1.45× win); giving it up gets the full 2.05×.

### Finding 3b — The nullable sort adds a null pre-pass on top of the comparator

**Not AMD-specific.** `sort_to_indices(nullable u64)` is 0.44×, slightly worse
than the non-null 0.51×. Profiling shows the same shared `quicksort::<(u32, …)>`
on both sides, so Finding 3's comparator problem applies unchanged. The extra
loss is a pre-pass: `sort::sort_primitive…` accounts for **13.6%** of our
runtime against **4.4%** for arrow's whole `sort_to_indices` entry point. That
pre-pass (`sort.rs:160-179`) tests validity one row at a time and pushes into
two separate vectors. Fixing the comparator helps this row too; the pre-pass is
a second, smaller item.

## Finding 4 — Prefix/suffix LIKE is all per-row overhead

**The short version:** these kernels call from JIT-compiled code back into
Rust once per row, and each crossing costs a few nanoseconds — closure
dispatch, an iterator call, a per-row append. Arrow runs one bulk vectorised
loop over the offsets instead. Against that, a few nanoseconds per row is the
whole loss.

`'abc%'` 3.89 ms vs arrow 0.91 ms; `'%xyz'` 3.56 ms vs 1.01 ms — universal
(0.16–0.35× on all three machines, worst on the M4), not AMD-specific, and
**not** related to Finding 1: these patterns have a 3-byte non-empty needle,
so their memcmp calls are ordinary ~1.2 ns ones.

The fixed cost can be isolated: a standalone loop does the entire `%abc%`
matching job in 9.49 ms, while the kernel's `contains` — same search, same
data — takes 12.93 ms. The ~3.4 ns/row difference is the fixed per-row cost of
the `filter_bytes` scaffolding, and at `'abc%'`'s ~3.9 ns/row total, that
scaffolding is essentially *all* of its time. The profile shows where it goes:
a libc memcmp call per row (~44%), `ArrowIter<&[u8]>` (4.5%), the boxed
closure (3.1%), `PrimitiveType::width` (2.1% — a function call in the hot
loop), plus a per-row `BooleanBufferBuilder::append`. Arrow uses a bulk
vectorized loop over the offsets.

This is the existing action item to route prefix/suffix-shaped LIKE patterns
to `StringStartEndKernel`, which already beats arrow on the long-string
dataset (5.67 ms vs 6.93 ms).

### The same pattern shows up in two more rows

`cmp::contains(utf8)` (0.80×) is the same `filter_bytes` path with a `memchr`
search in the closure, so it carries the same per-row toll — just diluted,
because the substring search itself is expensive enough to hide some of it.

`cast(dictionary(i32, utf8) → utf8)` (0.79×, 8.03 ms vs 6.34 ms) is the same
*shape* on the writer side. Our profile is led by `str_writer_append_bytes`
(9.4%) plus libc `memmove`, i.e. a Rust callback invoked once per string out of
JIT-compiled code. Arrow's `take_bytes` (19.6%) computes all the offsets in
bulk first and then copies. Over 1m rows the 1.69 ms difference is about
1.7 ns of per-row call overhead.

The general lesson across all three: **anything that crosses the JIT-to-Rust
boundary once per row costs a few nanoseconds per row**, and against a
bulk-vectorised arrow kernel that is enough to lose. The fix in each case is
to hoist the boundary out of the loop — do the work per batch, not per
element.

## Finding 5 — Run-end encoding only pays off when the work is uniform per run

**Not AMD-specific.** Three losing rows all take run-end-encoded (REE) inputs —
the layout that stores an array as a sequence of runs, so 20m logical elements
with 1m distinct runs occupy only 1m entries — and they share one explanation.

The whole point of reading through an encoding is that a 1m-run array standing
for 20m logical elements should cost 1m units of work, not 20m. That holds when
the operation gives the same answer for every element in a run — and when it
does, we win enormously: `cmp::lt(ree, scalar)` is a **27× win**, because we
compare once per run and stamp out the answer.

It stops holding the moment the operation varies *within* a run. Then we are
forced back to 20m units of work, and our per-element path is slower than
arrow's, which decodes the whole array in bulk and then operates densely.

**`select::filter(ree, bool)` — 0.53×, and it's branch mispredictions.** The
filter mask is random per element, so it varies inside every run. Measured
over ~20m logical elements (IPC = instructions per cycle):

| | instructions/el | cycles/el | IPC | branch misses/el | miss rate |
|---|---|---|---|---|---|
| ours | 26.4 | **21.2** | 1.24 | **0.552** | 9.9% |
| arrow | 31.4 | **10.7** | 2.94 | 0.088 | 2.7% |

Arrow executes *more* instructions than we do and still finishes in half the
cycles. That rules out "we generate too much code." At roughly 18 cycles per
mispredict on Zen 5, our 0.552 misses/element cost about **9.9 of our 21.2
cycles — around 47% of the entire benchmark**.

The reason is structural. We walk the array element by element, and every
element hits an unpredictable branch: does this one survive the mask? Arrow's
filter picks a bulk strategy instead — it converts the mask into a list of
selected indices and then copies, which is straight-line work with almost
nothing to mispredict.

**`logical_nulls(ree)` — 0.26×.** Same shape at small scale (1000 runs, ~5000
elements). Arrow takes the 1000-entry validity buffer and expands it per run
in bulk; we walk all ~5000 logical positions one at a time. The dictionary
version of the same operation at 100m rows is a **7.4× win**, which is the
contrast that makes the point: dictionaries don't force per-element work on
us, REE does.

**`select::take(ree, indices)` — 0.70×.** Random access into a run-end array
requires finding which run an index falls in, and `runend.rs:318` does that
with a binary search per index. Over 1m runs that's ~20 dependent,
cache-missing loads for every single index. This one is inherent to the layout
rather than a bug: arrow pays a different cost (decode everything once, then
take densely) and happens to come out ahead at this size.

**What to take from it.** The encoded-input thesis is sound and the numbers
elsewhere back it up — but it holds for *run-uniform* operations. For
filter/take/null-expansion over REE, decoding first is genuinely the better
plan, and the kernel could choose that path at runtime based on the operation
rather than always streaming through the encoding.

## Finding 6 — Dense `filter` simply emits more instructions per element

**Not AMD-specific.** `select::filter(array(i32), array(bool))` at 10m rows is
0.74×, and it is the one row that did *not* improve when the output zero-fill
was removed (0.74 → 0.78). So it has its own cause.

| | instructions/el | cycles/el | IPC | branch misses/el |
|---|---|---|---|---|
| ours | **9.23** | 2.81 | 3.28 | 0.017 |
| arrow | **5.81** | 2.06 | 2.82 | 0.016 |

Branch prediction is fine on both sides (under 2% miss rate), and our IPC is
actually *better* than arrow's. We are simply executing 1.6× more instructions
to move each element. Nine instructions to test a bit and conditionally copy
an i32 is a lot; arrow does it in under six.

This is a codegen-quality issue in the filter kernel rather than anything
architectural, and it is the least dramatic of the findings here — worth a
look at the emitted loop, but a long way behind F2 in value.

## Recommended actions, in order

*(All items below are now done — outcomes noted per item, details in
`fix-rounds.md`.)*

1. **Guard the zero-length comparisons in `compile_string_like`**
   (`string.rs:380-381`). Two lines, semantics-preserving, verified on Zen 5:
   `%abc%` 102.4 → 13.9 ms and `%abc%xyz` 60.0 → 4.6 ms. It fixes only the
   two-wildcard branch — no other LIKE shape changes — so it is not a
   substitute for item 4. Add `%abc%xyz` to the benchmark set while doing
   this. — **Done** (round 1): guarded, plus `%abc%xyz` added to the
   suite; the row later reached parity once the remaining per-row overhead
   fell (rounds 3–4).
2. **Stop zero-filling output buffers** in `PrimitiveWriter::allocate`
   (`primitive_writer.rs:135`), and give reductions a constant-size output
   instead of `"<= n"`. Measured on Zen 5: `compute::sum` 0.12× → 0.97×,
   `neg_wrapping` 0.59× → 1.00×, `cast(i32→i64)` 0.91× → 1.64×. This is the
   single highest-value change in this document — it is worth more than
   everything else here combined, and it is not AMD-specific. — **Done**
   (round 1): reductions now allocate exactly one element and sit at the
   read-bandwidth floor; `cast(i32→i64)` became a 1.69× win.
3. **Hoist `descending` out of the sort comparator** and decide explicitly
   whether the index tie-break is a guarantee worth 1.4×. Fixes both sort
   rows. — **Done and superseded** (rounds 1, 3–5): the comparator was
   first hoisted, then replaced entirely by packed integer sort keys and a
   stable radix sort. Both sort rows are now outright wins (1.25×/1.34×)
   with the deterministic tie-break preserved bit-for-bit.
4. **Get the per-row Rust callback out of the hot loop** for the string
   paths — route prefix/suffix LIKE to `StringStartEndKernel`, and batch
   `str_writer_append_bytes` in the dictionary-to-string cast. Covers four
   rows (F4), worth roughly 2–4× on the LIKE ones. — **Done** (rounds 1,
   3–4): prefix/suffix LIKE routes to the JIT kernel (1.2–1.4× wins),
   string appends are inlined, and flat string arrays skip the per-row
   iterator; `contains` became a win, `like('%abc%')` parity.
5. **Decode-then-operate for non-run-uniform REE work** (filter, take, null
   expansion). Streaming through the encoding is the right default and wins
   27× where the operation is run-uniform, but it is the wrong choice when the
   answer varies inside a run — the kernel could pick at runtime (F5). —
   **Done** (round 2): `filter(ree)` decodes first (3.9× win), `take(ree)`
   decodes when indices are dense (35.7× win), and REE null expansion is
   batched per span.
6. **Look at the dense filter loop's instruction count** — 9.2 instructions
   per element against arrow's 5.8, with no branch or memory problem to blame
   (F6). — **Done** (round 2): the vectorizer now lowers conditional emits
   to AVX-512 compress-stores; the dense filter processes 16 elements in ~9
   instructions and became a 1.83× win.
7. ~~Make `report.py` record `rustflags` in the manifest~~ — **done**; the
   field is written as of `report.py:416`. This run predates it, which is why
   its build flavour had to be recovered by disassembling the benchmark
   binary. Runs published before that change (all five currently on the site)
   still have no `rustflags` in their manifest.
8. **Investigate the three uninvestigated losing rows** — `cmp::lt(i32,
   scalar)` at 10m, `select::take(bool, u64)`, and the catch-all `%abc%xyz%`
   LIKE. The first two look like the same x86-native-arrow effect seen on
   Meteor Lake rather than anything of ours, but neither has been profiled.
   — **Resolved**: `take(bool, u64)` was root-caused (bit-serial output)
   and fixed with a word-packed gather (round 4); the other two drifted to
   wins/equal once the shared overheads above were removed.
