#!/usr/bin/env python3
"""Render Criterion benchmark results into a self-contained HTML page.

Reads every `target/criterion/*/new/estimates.json` and classifies each result
into a phase by its name:

  * `<base>/llvm compile` -> the one-time LLVM IR-gen + JIT cost
  * `<base>/llvm execute` -> warm, steady-state kernel execution
  * `<base>` containing a bare `llvm` token -> execution (older 2-phase benches)
  * `<base>` containing an `arrow` token -> the stock arrow-rs kernel

Benchmarks that carry a compile phase (the input-matrix benches) get the primary
"compile vs execute" table; the rest fall into a plain execute-vs-arrow table.
Run from the repo root:

    python3 benches/report.py     # writes docs/index.html

The numbers are baked into the page, so it publishes as a static file (e.g.
GitHub Pages -> /docs) without the gitignored target/ JSON.
"""

import glob
import html
import json
import os
import re
from datetime import date

CRITERION_DIR = "target/criterion"
OUTPUT = "docs/index.html"

IMPL_RE = re.compile(r"(?<![a-z])(llvm|arrow)(?![a-z])")
COMPILE_RE = re.compile(r"llvm[ _]compile")
EXECUTE_RE = re.compile(r"llvm[ _]execute")


def median_ns(path):
    with open(path) as fh:
        return json.load(fh)["median"]["point_estimate"]


def normalize(s):
    # criterion sanitizes "::" in a bench name to "__" in its directory name; restore it
    # before collapsing the remaining separators, so "cmp::lt" survives as "cmp::lt".
    s = s.replace("__", "::")
    return re.sub(r"[\s_/]+", " ", s).strip()


def phase_and_base(name):
    """Return (phase, base) where phase is compile/execute/arrow, or (None, None)."""
    if COMPILE_RE.search(name):
        return "compile", normalize(COMPILE_RE.sub("", name))
    if EXECUTE_RE.search(name):
        return "execute", normalize(EXECUTE_RE.sub("", name))
    m = IMPL_RE.search(name)
    if not m:
        return None, None
    phase = "execute" if m.group(1) == "llvm" else "arrow"
    return phase, normalize(IMPL_RE.sub("", name))


def fmt_time(ns):
    if ns is None:
        return "—"
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3), ("ns", 1.0)):
        if ns >= scale:
            return f"{ns / scale:.3g} {unit}"
    return f"{ns:.3g} ns"


def verdict(execute, arrow):
    speedup = arrow / execute
    faster = speedup >= 1.0
    factor = speedup if faster else 1.0 / speedup
    return speedup, ("win" if faster else "loss"), f"{factor:.2f}× {'faster' if faster else 'slower'}"


def collect():
    bench = {}  # base -> {compile, execute, arrow}
    unpaired = {}
    for path in glob.glob(f"{CRITERION_DIR}/*/new/estimates.json"):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        phase, base = phase_and_base(name)
        if phase is None:
            unpaired[name] = median_ns(path)
        else:
            bench.setdefault(base, {})[phase] = median_ns(path)

    matrix, other = [], []
    for base, ph in bench.items():
        if "execute" not in ph or "arrow" not in ph:
            for p, v in ph.items():
                unpaired[f"{base} ({p})"] = v
            continue
        speedup, cls, text = verdict(ph["execute"], ph["arrow"])
        row = {"name": base, "compile": ph.get("compile"), "execute": ph["execute"],
               "arrow": ph["arrow"], "speedup": speedup, "cls": cls, "verdict": text}
        (matrix if "compile" in ph else other).append(row)
    matrix.sort(key=lambda r: r["speedup"], reverse=True)
    other.sort(key=lambda r: r["speedup"], reverse=True)
    return matrix, other, unpaired


def render():
    matrix, other, unpaired = collect()

    def matrix_row(r):
        total = fmt_time((r["compile"] or 0) + r["execute"])
        return (
            f'<tr class="{r["cls"]}"><td class="name">{html.escape(r["name"])}</td>'
            f'<td class="num compile">{fmt_time(r["compile"])}</td>'
            f'<td class="num">{fmt_time(r["execute"])}</td>'
            f'<td class="num total">{total}</td>'
            f'<td class="num">{fmt_time(r["arrow"])}</td>'
            f'<td class="num badge">{r["verdict"]}</td></tr>'
        )

    def other_row(r):
        return (
            f'<tr class="{r["cls"]}"><td class="name">{html.escape(r["name"])}</td>'
            f'<td class="num">{fmt_time(r["execute"])}</td>'
            f'<td class="num">{fmt_time(r["arrow"])}</td>'
            f'<td class="num badge">{r["verdict"]}</td></tr>'
        )

    matrix_body = "\n".join(matrix_row(r) for r in matrix)
    other_body = "\n".join(other_row(r) for r in other)
    unpaired_body = "\n".join(
        f'<tr><td class="name">{html.escape(n)}</td><td class="num">{fmt_time(v)}</td></tr>'
        for n, v in sorted(unpaired.items())
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>arrow-compile-compute benchmarks</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; max-width: 1000px;
         margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ margin-bottom: .2rem; }}
  .meta {{ color: #888; margin-top: 0; }}
  h2 {{ margin-top: 2.4rem; border-bottom: 1px solid #8884; padding-bottom: .3rem; }}
  .count {{ font-size: .7em; color: #888; font-weight: normal; }}
  .sub {{ color: #888; margin-top: .2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: .5rem; }}
  th, td {{ text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #8883; }}
  th {{ font-size: .78em; text-transform: uppercase; letter-spacing: .03em; color: #888; }}
  .num {{ font-variant-numeric: tabular-nums; font-family: ui-monospace, monospace; text-align: right; }}
  td.name {{ font-family: ui-monospace, monospace; }}
  td.compile {{ color: #9a6700; }}
  td.total {{ color: #888; }}
  .badge {{ font-weight: 600; }}
  tr.win  td.badge {{ color: #1a7f37; }}
  tr.loss td.badge {{ color: #b3261e; }}
  tr.win  {{ background: #1a7f3712; }}
  tr.loss {{ background: #b3261e10; }}
  details {{ margin-top: 2rem; }}
  summary {{ cursor: pointer; color: #888; }}
</style>
</head>
<body>
<h1>arrow-compile-compute &mdash; JIT vs arrow-rs</h1>
<p class="meta">Criterion medians, host CPU. Generated {date.today().isoformat()}. Lower is better.</p>

<h2>Input matrix: compile vs execute <span class="count">{len(matrix)}</span></h2>
<p class="sub">The JIT pays <b>compile</b> once per kernel shape (LLVM IR-gen + JIT), then every
call is the warm <b>execute</b> cost. <b>total = compile + execute</b> is the cost of a single
cold call; amortized over many calls it tends to <b>execute</b>. The verdict compares
<b>execute</b> vs arrow. For dict inputs the JIT reads the encoding directly while arrow must
decode first.</p>
<table>
<thead><tr><th>benchmark</th><th>llvm compile</th><th>llvm execute</th><th>total (1 call)</th><th>arrow</th><th>execute vs arrow</th></tr></thead>
<tbody>
{matrix_body}
</tbody></table>

<h2>Other benchmarks: execute vs arrow <span class="count">{len(other)}</span></h2>
<p class="sub">Older two-phase benches (execution only; compile not isolated).</p>
<table>
<thead><tr><th>benchmark</th><th>llvm</th><th>arrow</th><th>result</th></tr></thead>
<tbody>
{other_body}
</tbody></table>

<details>
<summary>Unpaired benchmarks ({len(unpaired)}) &mdash; no arrow oracle / internal</summary>
<table><thead><tr><th>benchmark</th><th>median</th></tr></thead>
<tbody>
{unpaired_body}
</tbody></table>
</details>
</body>
</html>
"""


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as fh:
        fh.write(render())
    print(f"wrote {OUTPUT}")
