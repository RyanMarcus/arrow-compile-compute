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


def read_estimate(path):
    with path.open() as fh:
        median = json.load(fh)["median"]
    interval = median["confidence_interval"]
    return {
        "point": median["point_estimate"],
        "lower": interval["lower_bound"],
        "upper": interval["upper_bound"],
    }


def normalize(value):
    # Criterion sanitizes "::" in a bench name to "__" in its directory name.
    value = value.replace("__", "::")
    return re.sub(r"[\s/]+", " ", value).strip(" _")


def phase_and_base(name):
    """Return (phase, base) where execute includes direct and warm JIT calls."""
    if COMPILE_RE.search(name):
        return "compile", normalize(COMPILE_RE.sub("", name))
    if EXECUTE_RE.search(name):
        return "execute", normalize(EXECUTE_RE.sub("", name))
    match = IMPL_RE.search(name)
    if not match:
        return None, None
    phase = "execute" if match.group(1) == "llvm" else "arrow"
    return phase, normalize(IMPL_RE.sub("", name))


def verdict(execute, arrow):
    """Classify the ratio conservatively from the two median confidence intervals."""
    values = (*execute.values(), *arrow.values())
    if any(value <= 0 for value in values):
        raise ValueError("benchmark estimates must be positive")

    speedup = arrow["point"] / execute["point"]
    lower = arrow["lower"] / execute["upper"]
    upper = arrow["upper"] / execute["lower"]

    if lower > 1.0:
        return speedup, "llvm-win"
    if upper < 1.0:
        return speedup, "arrow-win"
    return speedup, "inconclusive"


def collect(criterion_dir):
    criterion_dir = Path(criterion_dir)
    bench = {}
    sources = {}
    unpaired = {}

    for path in criterion_dir.rglob("new/estimates.json"):
        relative = path.relative_to(criterion_dir)
        name = "/".join(relative.parts[:-2])
        phase, base = phase_and_base(name)
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

    matrix = []
    other = []
    for base, phases in bench.items():
        if "execute" not in phases or "arrow" not in phases:
            for phase, estimate in phases.items():
                unpaired[f"{base} ({phase})"] = estimate
            continue

        speedup, result_class = verdict(phases["execute"], phases["arrow"])
        row = {
            "name": base,
            "compile": phases.get("compile"),
            "execute": phases["execute"],
            "arrow": phases["arrow"],
            "speedup": speedup,
            "classification": result_class,
        }
        (matrix if "compile" in phases else other).append(row)

    matrix.sort(key=lambda row: row["speedup"], reverse=True)
    other.sort(key=lambda row: row["speedup"], reverse=True)
    return matrix, other, unpaired


def build_report_data(criterion_dir, manifest):
    matrix, other, unpaired = collect(criterion_dir)
    return {
        "schema_version": 1,
        "run": {
            "revision": manifest["revision"],
            "dirty": manifest["dirty"],
            "started_at": manifest["started_at"],
            "completed_at": manifest["completed_at"],
            "commands": manifest["commands"],
        },
        "matrix": matrix,
        "other": other,
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
