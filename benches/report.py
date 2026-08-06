#!/usr/bin/env python3
"""Run isolated Criterion benchmarks and export data for the static HTML report.

Create a fresh benchmark run and report:

    python3 benches/report.py --run

Export a previously completed run:

    python3 benches/report.py --results target/criterion-runs/<run-id>

Each fresh run gets its own Criterion output directory and manifest, preventing
results left by older or renamed benchmarks from entering the report.
"""

import argparse
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "target" / "criterion-runs"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "benchmark-results-ARM.json"
MANIFEST_NAME = "manifest.json"

IMPL_RE = re.compile(r"(?<![a-z])(llvm|arrow)(?![a-z])")
COMPILE_RE = re.compile(r"llvm[ _]compile")
EXECUTE_RE = re.compile(r"llvm[ _](?:execute|direct|warm)")
REFERENCE_RE = re.compile(r"rust[ _]reference")

# Ratios within this margin count as "equal": a 2-4% difference is treated as
# the same performance, per the reporting thresholds agreed for this suite.
# 3.5% specifically so the verdict can never contradict the page's two-decimal
# ratio display: anything rendering as "1.03x" or less is always "equal".
EQUAL_MARGIN = 1.035

# Section order and display names, keyed by the public-API module prefix.
FAMILIES = {
    "cmp": "Comparisons",
    "arith": "Arithmetic",
    "compute": "Aggregations",
    "cast": "Casts",
    "select": "Selection",
    "sort": "Sorting",
    "vec": "Vector ops",
}
OTHER_FAMILY = "Other"


def family_of(name):
    match = re.match(r"([a-z_]+)::", name)
    if match and match.group(1) in FAMILIES:
        return FAMILIES[match.group(1)]
    return OTHER_FAMILY


def split_top_level(text, separator=","):
    """Split on a separator, ignoring separators nested inside parentheses."""
    parts, depth, start = [], 0, 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and text.startswith(separator, i):
            parts.append(text[start:i])
            i += len(separator)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return [part.strip() for part in parts]


def display_type(text):
    """Render a benchmark type expression without constructor parentheses.

    array(i32) -> i32; scalar(i32) -> i32 scalar; dictionary(i8, i32) ->
    dict i8→i32; run_end_encoded(i32, dictionary(i8, i32)) -> ree i32→dict
    i8→i32; fixed_size_list(f32, 768) -> f32[768]. Unknown constructors are
    left untouched.
    """
    text = text.strip()
    match = re.match(r"^([a-z_]+)\((.*)\)(.*)$", text)
    if not match:
        return text
    constructor, inner, suffix = match.groups()
    if constructor == "array":
        rendered = display_type(inner)
    elif constructor == "scalar":
        rendered = f"{display_type(inner)} scalar"
    elif constructor in ("dictionary", "run_end_encoded"):
        parts = split_top_level(inner)
        if len(parts) != 2:
            return text
        short = "dict" if constructor == "dictionary" else "ree"
        rendered = f"{short} {display_type(parts[0])}→{display_type(parts[1])}"
    elif constructor == "fixed_size_list":
        parts = split_top_level(inner)
        if len(parts) != 2:
            return text
        rendered = f"{display_type(parts[0])}[{parts[1]}]"
    else:
        return text
    return rendered + suffix


def op_inputs_size(name):
    """Split a benchmark name into (operator, display inputs, size note)."""
    open_paren = name.find("(")
    if open_paren == -1:
        return name, "", ""
    depth = 0
    close_paren = None
    for index in range(open_paren, len(name)):
        if name[index] == "(":
            depth += 1
        elif name[index] == ")":
            depth -= 1
            if depth == 0:
                close_paren = index
                break
    if close_paren is None:
        return name, "", ""
    operator = name[:open_paren]
    arguments = name[open_paren + 1 : close_paren]
    size = name[close_paren + 1 :].strip()

    rendered = []
    for part in split_top_level(arguments):
        halves = split_top_level(part, " to ")
        rendered.append(" to ".join(display_type(half) for half in halves))
    return operator, ", ".join(rendered), size


def read_estimate(path):
    with path.open() as fh:
        median = json.load(fh)["median"]
    interval = median["confidence_interval"]
    return {
        "point": median["point_estimate"],
        "lower": interval["lower_bound"],
        "upper": interval["upper_bound"],
    }


def read_benchmark_name(estimates_path, criterion_dir):
    """Prefer Criterion's recorded full_id: directory names are truncated to
    ~64 characters, which mangles long benchmark ids and breaks pairing."""
    metadata_path = estimates_path.parent / "benchmark.json"
    if metadata_path.is_file():
        with metadata_path.open() as fh:
            return json.load(fh)["full_id"]
    relative = estimates_path.relative_to(criterion_dir)
    return "/".join(relative.parts[:-2])


def normalize(value):
    # Criterion sanitizes "::" in a bench name to "__" in its directory name.
    value = value.replace("__", "::")
    return re.sub(r"[\s/]+", " ", value).strip(" _")


def phase_and_base(name):
    """Return (phase, base, baseline_kind).

    phase is "llvm", "baseline", or None (unpaired). Compile-phase entries from
    older runs are deliberately unpaired: the suite now measures only warm
    public-API calls.
    """
    if COMPILE_RE.search(name):
        return None, None, None
    if EXECUTE_RE.search(name):
        return "llvm", normalize(EXECUTE_RE.sub("", name)), None
    if REFERENCE_RE.search(name):
        return "baseline", normalize(REFERENCE_RE.sub("", name)), "reference"
    match = IMPL_RE.search(name)
    if not match:
        return None, None, None
    if match.group(1) == "llvm":
        return "llvm", normalize(IMPL_RE.sub("", name)), None
    return "baseline", normalize(IMPL_RE.sub("", name)), "arrow"


def verdict(llvm, baseline):
    """Classify the ratio: every result is a win for one side or "equal".

    A point ratio within EQUAL_MARGIN counts as equal even when statistically
    distinguishable — a 2-4% gap is not a meaningful difference here. Beyond
    the margin, a win requires the median confidence intervals to not overlap;
    when they do overlap, the measurement cannot separate the two sides, so
    the result is also reported as equal.
    """
    values = (*llvm.values(), *baseline.values())
    if any(value <= 0 for value in values):
        raise ValueError("benchmark estimates must be positive")

    speedup = baseline["point"] / llvm["point"]
    if max(speedup, 1.0 / speedup) <= EQUAL_MARGIN:
        return speedup, "equal"

    lower = baseline["lower"] / llvm["upper"]
    upper = baseline["upper"] / llvm["lower"]
    if lower > 1.0:
        return speedup, "llvm-win"
    if upper < 1.0:
        return speedup, "arrow-win"
    return speedup, "equal"


def collect(criterion_dir):
    criterion_dir = Path(criterion_dir)
    bench = {}
    kinds = {}
    sources = {}
    unpaired = {}

    for path in criterion_dir.rglob("new/estimates.json"):
        name = read_benchmark_name(path, criterion_dir)
        phase, base, baseline_kind = phase_and_base(name)
        estimate = read_estimate(path)

        if phase is None:
            if name in unpaired:
                raise ValueError(f"duplicate unpaired benchmark: {name}")
            unpaired[name] = estimate
            continue

        key = (base, phase)
        if key in sources:
            raise ValueError(
                f"duplicate {phase} result for {base!r}: {sources[key]} and {path}"
            )
        sources[key] = path
        bench.setdefault(base, {})[phase] = estimate
        if baseline_kind is not None:
            kinds[base] = baseline_kind

    results = []
    for base, phases in bench.items():
        if "llvm" not in phases or "baseline" not in phases:
            for phase, estimate in phases.items():
                unpaired[f"{base} ({phase})"] = estimate
            continue

        speedup, result_class = verdict(phases["llvm"], phases["baseline"])
        operator, inputs, size = op_inputs_size(base)
        results.append(
            {
                "name": base,
                "op": operator,
                "inputs": inputs,
                "size": size,
                "family": family_of(base),
                "baseline_kind": kinds[base],
                "llvm": phases["llvm"],
                "baseline": phases["baseline"],
                "speedup": speedup,
                "classification": result_class,
            }
        )

    family_order = {name: index for index, name in enumerate(FAMILIES.values())}
    # Fixed group order within each family: the family's primary operator
    # first, remaining operators alphabetically — stable across runs. Rows
    # inside a group sort ascending by speedup, so the results that most need
    # attention (LLVM losses) come first.
    primary_ops = {"cmp::lt": 0}
    results.sort(
        key=lambda row: (
            family_order.get(row["family"], len(family_order)),
            primary_ops.get(row["op"], 1),
            row["op"],
            row["speedup"],
        )
    )
    return results, unpaired


def build_report_data(criterion_dir, manifest):
    results, unpaired = collect(criterion_dir)
    return {
        "schema_version": 2,
        "run": {
            "revision": manifest["revision"],
            "dirty": manifest["dirty"],
            "started_at": manifest["started_at"],
            "completed_at": manifest["completed_at"],
            "commands": manifest["commands"],
            "host": manifest.get("host"),
            "rustflags": manifest.get("rustflags"),
        },
        "equal_margin": EQUAL_MARGIN,
        "results": results,
        "unpaired": [
            {"name": name, "estimate": estimate}
            for name, estimate in sorted(unpaired.items())
        ],
    }


def load_manifest(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"not an isolated benchmark run (missing {manifest_path})")
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "complete":
        raise ValueError(
            f"benchmark run is not complete (status: {manifest.get('status', 'unknown')})"
        )
    criterion_dir = run_dir / manifest["criterion_dir"]
    if not criterion_dir.is_dir():
        raise ValueError(f"Criterion output directory does not exist: {criterion_dir}")
    return manifest, criterion_dir


def write_manifest(run_dir, manifest):
    with (run_dir / MANIFEST_NAME).open("w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


def with_native_target(rustflags):
    """Benchmark runs build for the host CPU by default, so the ahead-of-time
    baseline (arrow) competes with the full instruction set, like the JIT does.

    An explicit `-C target-cpu=...` in RUSTFLAGS is respected — e.g. pass
    `-C target-cpu=generic` (x86-64 baseline) to measure arrow as shipped.
    """
    if rustflags and "target-cpu" in rustflags:
        return rustflags
    native = "-C target-cpu=native"
    return f"{rustflags} {native}" if rustflags else native


def host_info():
    """Best-effort description of the machine the run executed on."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu": platform.processor() or None,
        "cores": os.cpu_count(),
        "memory_gb": None,
    }
    if platform.system() == "Darwin":
        for key, sysctl_name in [("cpu", "machdep.cpu.brand_string"), ("memory_gb", "hw.memsize")]:
            try:
                value = subprocess.run(
                    ["sysctl", "-n", sysctl_name], check=True, capture_output=True, text=True
                ).stdout.strip()
                info[key] = round(int(value) / 2**30) if key == "memory_gb" else value
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                pass
    elif platform.system() == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            if match:
                info["cpu"] = match.group(1).strip()
            meminfo = Path("/proc/meminfo").read_text()
            match = re.search(r"MemTotal:\s*(\d+) kB", meminfo)
            if match:
                info["memory_gb"] = round(int(match.group(1)) / 2**20)
        except OSError:
            pass
    return info


def run_benchmarks(bench_names, runs_dir=DEFAULT_RUNS_DIR):
    started = datetime.now(timezone.utc)
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )

    run_id = f"{started.strftime('%Y%m%dT%H%M%S.%fZ')}-{revision}"
    run_dir = Path(runs_dir).resolve() / run_id
    criterion_dir = run_dir / "criterion"
    run_dir.mkdir(parents=True)

    commands = (
        [["cargo", "bench", "--bench", name] for name in bench_names]
        if bench_names
        else [["cargo", "bench"]]
    )
    manifest = {
        "schema_version": 1,
        "status": "running",
        "revision": revision,
        "dirty": dirty,
        "started_at": started.isoformat(),
        "completed_at": None,
        "criterion_dir": "criterion",
        "commands": commands,
        "host": host_info(),
        "rustflags": with_native_target(os.environ.get("RUSTFLAGS")),
    }
    write_manifest(run_dir, manifest)

    environment = os.environ.copy()
    environment["CRITERION_HOME"] = str(criterion_dir)
    environment["RUSTFLAGS"] = with_native_target(environment.get("RUSTFLAGS"))
    try:
        for command in commands:
            subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    except subprocess.CalledProcessError:
        manifest["status"] = "failed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        raise

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(run_dir, manifest)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", action="store_true", help="run benchmarks in a fresh directory")
    source.add_argument("--results", type=Path, help="export a completed isolated run directory")
    parser.add_argument(
        "--bench",
        action="append",
        default=[],
        metavar="NAME",
        help="with --run, run only this Cargo benchmark target (repeatable)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()

    if args.bench and not args.run:
        parser.error("--bench requires --run")

    run_dir = run_benchmarks(args.bench, args.runs_dir) if args.run else args.results
    manifest, criterion_dir = load_manifest(run_dir)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        json.dump(build_report_data(criterion_dir, manifest), fh, indent=2)
        fh.write("\n")
    print(f"wrote {output} from {Path(run_dir).resolve()}")


if __name__ == "__main__":
    main()
