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
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "target" / "criterion-runs"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "benchmark-results.json"
MANIFEST_NAME = "manifest.json"

IMPL_RE = re.compile(r"(?<![a-z])(llvm|arrow)(?![a-z])")
COMPILE_RE = re.compile(r"llvm[ _]compile")
EXECUTE_RE = re.compile(r"llvm[ _](?:execute|direct|warm)")
REFERENCE_RE = re.compile(r"rust[ _]reference")

# Ratios within this margin count as "equal": a 2-4% difference is treated as
# the same performance, per the reporting thresholds agreed for this suite.
EQUAL_MARGIN = 1.03

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
    """Classify the ratio: near-parity is "equal", then conservative CI bounds.

    A point ratio within EQUAL_MARGIN counts as equal even when statistically
    distinguishable — a 2-4% gap is not a meaningful difference here. Beyond
    the margin, a win requires the median confidence intervals to not overlap.
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
    return speedup, "inconclusive"


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
        results.append(
            {
                "name": base,
                "family": family_of(base),
                "baseline_kind": kinds[base],
                "llvm": phases["llvm"],
                "baseline": phases["baseline"],
                "speedup": speedup,
                "classification": result_class,
            }
        )

    family_order = {name: index for index, name in enumerate(FAMILIES.values())}
    results.sort(
        key=lambda row: (
            family_order.get(row["family"], len(family_order)),
            -row["speedup"],
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
    }
    write_manifest(run_dir, manifest)

    environment = os.environ.copy()
    environment["CRITERION_HOME"] = str(criterion_dir)
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
