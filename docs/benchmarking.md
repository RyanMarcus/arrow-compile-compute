# Benchmarking — running a suite and publishing the report

How a benchmark run becomes the page at `docs/index.html`. For writing the
benchmarks themselves, see [`benches/README.md`](../benches/README.md).

## The pipeline

```
cargo bench  →  isolated Criterion dir + manifest.json  →  report JSON  →  index.html
```

`benches/report.py` owns the middle two steps. It never reads the shared
`target/criterion` directory: each run gets its own directory under
`target/criterion-runs/<timestamp>-<revision>/`, so results left behind by older or
since-renamed benchmarks cannot leak into a report.

## Running

From the repository root (needs Rust, LLVM 21 via inkwell, and Python 3):

```sh
python3 benches/report.py --run                     # full suite → docs/benchmark-results-m4-native.json
python3 benches/report.py --results target/criterion-runs/<run-id>   # re-export, no re-benchmarking
python3 benches/report.py --run --output docs/benchmark-results-zen5-native.json

# one target (repeatable) — always with --output, see the warning below
python3 benches/report.py --run --bench neg_bench --output /tmp/check.json
```

> **`--bench` without `--output` destroys results.** The export always writes a
> *complete* report, so a single-target run rewrites the destination with only the rows
> that ran and drops the rest of the suite. `--output` defaults to the M4 file, and
> `guard_output_host` will not save you — it compares CPUs, and on that machine the CPU
> matches. Send partial runs to a scratch path; publish only from a full run.

View the result:

```sh
python3 -m http.server -d docs 8321   # → localhost:8321
```

`--results` is worth knowing about: exporting is pure post-processing, so a change to
the report logic can be applied to a run that already happened instead of spending
another hour of wall clock on it.

### Build flags

Runs inject `-C target-cpu=native` into `RUSTFLAGS` unless it already names a
`target-cpu`. This is deliberate: the arrow baseline is compiled ahead of time, and
without it the JIT would be comparing its host-specific code against a generic
x86-64 build. To measure arrow *as shipped* instead, set
`RUSTFLAGS='-C target-cpu=generic'` — the explicit value is respected.

The flags actually used are recorded in the manifest and carried into the report, so
a published result always says how it was built.

## The manifest

Each run directory holds a `manifest.json` recording `status`, git `revision`, whether
the tree was `dirty`, start/finish timestamps, the exact `cargo bench` commands, host
description (CPU, cores, memory, OS, arch), and `rustflags`. A run that fails is marked
`failed`, and export refuses anything whose status is not `complete`. The `dirty` flag
is surfaced on the page — results from a modified tree are labelled as such rather than
rejected.

## How results are classified

`report.py::verdict` compares the two halves of each pair using Criterion's **median**
point estimate and its confidence interval:

- ratio within **±3.5 %** (`EQUAL_MARGIN`) → `equal`, even if statistically separable.
  A 2–4 % gap is not a meaningful difference for this suite, and the margin is set at
  3.5 % specifically so the verdict can never contradict the two-decimal ratio the page
  prints: anything displaying as `1.03x` or less always reads "equal".
- beyond the margin, a win requires the two medians' confidence intervals **not to
  overlap** (`baseline.lower / llvm.upper > 1` for an LLVM win, and the mirror image
  for an arrow win). Overlapping intervals mean the measurement cannot separate the
  two, which is also reported as `equal`.

So every row is `llvm-win`, `arrow-win`, or `equal`, and "equal" covers both "the same"
and "too noisy to tell".

Rows are grouped into sections by their public-API prefix — Comparisons, Arithmetic,
Boolean, Aggregations, Casts, Selection, Sorting, Vector ops, then Other — and within a section
sorted ascending by speedup, so LLVM's losses appear first.

Benchmarks with no partner (compile-time measurements, hand-written prototypes) are
carried through in an `unpaired` list and rendered in a collapsed section at the bottom
of the page. That list is the first place to look when a benchmark you just added is
missing from the report.

## Per-machine result files

The suite is run on several machines and the page lets you switch between them.
`docs/runs.json` is the index:

```json
[
  { "label": "Apple M4 · macOS (arm64)", "file": "benchmark-results-m4-native.json" },
  { "label": "Intel Core Ultra 7 155H · Linux (x86_64)", "file": "benchmark-results-linux-native.json" },
  { "label": "AMD Ryzen 7 9700X · Linux (x86_64, AVX-512)", "file": "benchmark-results-zen5-native.json" }
]
```

`index.html` renders one button per entry and loads the first by default, falling back
to the M4 file if `runs.json` cannot be read.

`--output` defaults to the M4 file, so running the suite on any other machine would
otherwise overwrite that machine's numbers. `guard_output_host` prevents it: it compares
the CPU recorded in the destination file against the CPU just benchmarked and aborts
if they differ. When that fires, pass the right `--output`.

**Adding a machine:** run with an explicit `--output docs/benchmark-results-<name>.json`,
then add a `{label, file}` entry to `docs/runs.json`. Keep the label in the existing
shape — CPU · OS (arch, notable ISA).

## Comparing across machines

Ratios are comparable across machines; absolute times are not. The suite's cross-machine
findings live in `docs/cross-platform-findings.md`. Re-running one machine's file against
a different revision than the others makes the comparison unsound — the page shows each
run's revision for exactly this reason, so check it before drawing conclusions from a
difference between two columns.

## Report format

The exported JSON is `schema_version: 2`:

```
run             revision, dirty, started_at, completed_at, commands, host, rustflags
equal_margin    the ±ratio treated as "equal"
results[]       name, op, inputs, size, family, baseline_kind ("arrow" | "reference"),
                llvm {point, lower, upper}, baseline {…}, speedup, classification
unpaired[]      name, estimate
```

`index.html` reads this and nothing else — it is a static page with no build step.

## Tests

```sh
python3 -m unittest benches.test_report   # from the repo root
```

Covers name normalization, pairing, family assignment, verdict thresholds, and the
duplicate-detection failure. Run it after touching `report.py`; the naming and pairing
rules it locks down are documented in [`benches/README.md`](../benches/README.md).
