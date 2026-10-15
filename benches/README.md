# Benchmarks — adding a kernel pair

Every published result is a **pair**: one measurement of this crate's JIT kernel and
one of the baseline it is being compared against. The two are separate Criterion
benchmarks; `report.py` joins them by name. If the names don't line up the row is
silently dropped from the report into an "unpaired" list, so the naming rules below
are the part to get right.

For what happens to a pair after it runs — the report, the verdict thresholds, the
per-machine result files — see [`docs/benchmarking.md`](../docs/benchmarking.md).

## The name is the contract

```
c.bench_function("cmp::lt(array(i32), scalar(i32)) 1m rows/llvm warm", ...);
c.bench_function("cmp::lt(array(i32), scalar(i32)) 1m rows/arrow",     ...);
```

`report.py::phase_and_base` strips the tag after `/` and pairs the two entries on
what's left. A name has four parts:

| part | example | rules |
| --- | --- | --- |
| family | `cmp::` | one of `cmp` `arith` `boolean` `compute` `cast` `select` `sort` `vec` — this is the public-API module, and it decides the report section. Anything else lands under "Other"; see [Adding a family](#adding-a-family). |
| operator | `lt` | the public API function being measured. |
| inputs | `(array(i32), scalar(i32))` | constructor grammar, see below. |
| size note | ` 1m rows` | free text; shown as-is. Anything after the closing paren. |
| tag | `/llvm warm` | fixed vocabulary, see below. |

### Input grammar

`display_type` in `report.py` renders these; an unrecognized constructor is passed
through untouched, which is legal but reads badly in the report.

| written | rendered |
| --- | --- |
| `array(i32)` | `i32` |
| `scalar(i32)` | `i32 scalar` |
| `dictionary(i8, i32)` | `dict i8→i32` |
| `run_end_encoded(i32, dictionary(i8, i32))` | `ree i32→dict i8→i32` |
| `fixed_size_list(f32, 768)` | `f32[768]` |
| `array(i32) to array(i64)` | `i32 to i64` (used by cast benches) |

### Adding a family

Benchmarking a public-API module for the first time means the family does not exist
yet, and every new row silently lands in "Other". Two edits, in the same commit as the
benchmark:

1. add the module to `FAMILIES` in `report.py`, positioned where the section should
   appear on the page — dict order *is* section order;
2. add a case to `test_report.py` asserting the family assignment, so a later rename
   cannot quietly send the rows back to "Other".

`boolean::not` was added this way; `test_assigns_boolean_family` is the template.

### Tag vocabulary

| tag | meaning |
| --- | --- |
| `/llvm warm` | this crate, kernel already compiled and cached. The normal LLVM side. `llvm execute` and `llvm direct` are accepted synonyms. |
| `/arrow` | stock arrow-rs baseline. |
| `/rust reference` | baseline when arrow-rs has no equivalent kernel. |
| `/llvm compile` | compile-time measurement. Deliberately **never paired** — the suite reports warm execution only. |

Write the tags exactly. `/arrow-rs` is not `/arrow`: the regex strips `arrow` and
leaves `-rs` behind, so the base name no longer matches its partner and the pair
falls apart.

## Adding one, end to end

1. **Pick the file.** Group by operator family — `neg_bench.rs`, `string_bench.rs`,
   `matrix_bench.rs` (comparisons across encodings). Only add a new target when no
   existing file fits.

2. **Register a new target** in `Cargo.toml`:

   ```toml
   [[bench]]
   name = "my_bench"
   harness = false
   ```

3. **Build deterministic input.** Seed `fastrand` with 42, as every other bench does;
   1m rows is the default size, 10m for kernels cheap enough that 1m is noise.

   ```rust
   let mut rng = fastrand::Rng::with_seed(42);
   let data = Int32Array::from((0..N).map(|_| rng.i32(..)).collect_vec());
   ```

4. **Assert against the baseline before timing.** This is load-bearing twice: it is
   the correctness oracle, and it is what compiles and caches the kernel so the timed
   loop measures execution rather than the ~10 ms LLVM compile.

   ```rust
   let ours = arrow_compile_compute::arith::neg_wrapping(&data).unwrap();
   let theirs = arrow_arith::numeric::neg_wrapping(&data).unwrap();
   assert_eq!(ours, theirs);
   ```

   Use exact `assert_eq!` where the outputs are directly comparable. Where they are
   not — different output layouts, float encodings — assert something meaningful
   (`len` + `true_count` for boolean masks, as `matrix_bench.rs` does) rather than
   dropping the check.

5. **Emit both halves**, `black_box`-ing the inputs so the optimizer cannot hoist
   the call out of the loop:

   ```rust
   c.bench_function("arith::neg_wrapping(array(i32)) 10m rows/llvm warm", |b| {
       b.iter(|| arrow_compile_compute::arith::neg_wrapping(black_box(&data)).unwrap())
   });
   c.bench_function("arith::neg_wrapping(array(i32)) 10m rows/arrow", |b| {
       b.iter(|| arrow_arith::numeric::neg_wrapping(black_box(&data)).unwrap())
   });
   ```

6. **Always go through the public API** (`arrow_compile_compute::<module>::<fn>`),
   never an internal kernel type. The measurement should include cache lookup and
   dispatch, because that is what a user pays.

7. **Run it and check the pairing.** Send the export to a scratch file — **never omit
   `--output` on a `--bench` run.** `--output` defaults to this machine's published
   results file, and a partial run rewrites it with only the rows that ran, discarding
   the rest of the suite. The host guard does not catch this: it only compares CPUs,
   and the CPU matches.

   ```sh
   python3 benches/report.py --run --bench my_bench --output /tmp/check.json
   python3 -c "import json;d=json.load(open('/tmp/check.json'));print([(r['family'],r['name']) for r in d['results']]);print(d['unpaired'])"
   ```

   Your new rows must appear in `results`, under the expected family. If they show up
   in `unpaired`, the two names disagree after tag-stripping.

## Pitfalls

- **The base name must not contain the words `llvm` or `arrow`.** The regexes match
  anywhere in the string, not just in the tag, and the two sides are stripped
  differently: the LLVM side stops at the first `llvm warm`-style match, while the
  baseline side runs `re.sub` and removes *every* occurrence. So
  `arrow_cmp::not(...)` keeps its prefix on the `/llvm warm` entry but loses it on the
  `/arrow` entry, and the pair silently splits.
- **Duplicate base names hard-fail the export** with `duplicate {phase} result for ...`.
  Two benches that normalize to the same base — easy to hit when generating names in a
  loop — will abort the whole report, not just their own row.
- **A missing half is silent.** Add both `bench_function` calls in the same commit;
  a lone LLVM entry produces no report row and no error.
- **Missing family prefix** puts the row in "Other". A bench that names its rows
  `filter(i32) ...` rather than `select::filter(i32) ...` lands there — follow
  `neg_bench.rs` instead.
- **Baselines should be honest**, i.e. what an arrow-rs user would actually write.
  When stock arrow has no kernel for an encoded layout, decode first and include the
  decode in the timed closure (see the dictionary cases in `neg_bench.rs`), and say so
  in a comment.
- **Unpaired lower bounds are allowed** but must be tagged so they can't pair — the
  hand-written prototype in `string_bench.rs` is tagged `/custom` for this reason.

## Files

| file | role |
| --- | --- |
| `*_bench.rs` | the benchmarks; one Cargo target each, all registered in `Cargo.toml` |
| `report.py` | runs a suite in an isolated Criterion directory and exports the report JSON |
| `test_report.py` | unit tests for the pairing, naming, and verdict logic — `python3 -m unittest benches.test_report` from the repo root |
