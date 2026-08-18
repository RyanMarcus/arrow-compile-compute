import json
import tempfile
import unittest
from pathlib import Path

from benches import report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.criterion_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_estimate(self, name, point, lower=None, upper=None, full_id=None):
        path = self.criterion_dir / name / "new" / "estimates.json"
        path.parent.mkdir(parents=True)
        with path.open("w") as fh:
            json.dump(
                {
                    "median": {
                        "confidence_interval": {
                            "lower_bound": lower if lower is not None else point * 0.95,
                            "upper_bound": upper if upper is not None else point * 1.05,
                        },
                        "point_estimate": point,
                    }
                },
                fh,
            )
        if full_id is not None:
            with (path.parent / "benchmark.json").open("w") as fh:
                json.dump({"full_id": full_id}, fh)

    def test_pairs_warm_llvm_with_arrow_and_assigns_family(self):
        self.write_estimate(
            "select__take(fixed_size_list(bool, 1024), array(u64))/llvm warm", 10
        )
        self.write_estimate(
            "select__take(fixed_size_list(bool, 1024), array(u64))/arrow", 20
        )

        results, unpaired = report.collect(self.criterion_dir)

        self.assertEqual(1, len(results))
        row = results[0]
        self.assertEqual(
            "select::take(fixed_size_list(bool, 1024), array(u64))", row["name"]
        )
        self.assertEqual("Selection", row["family"])
        self.assertEqual("arrow", row["baseline_kind"])
        self.assertEqual("llvm-win", row["classification"])
        self.assertEqual({}, unpaired)

    def test_pairs_rust_reference_baseline(self):
        self.write_estimate("vec__norm(fixed_size_list(f32, 768))/llvm warm", 10)
        self.write_estimate("vec__norm(fixed_size_list(f32, 768))_rust reference", 30)

        results, unpaired = report.collect(self.criterion_dir)

        self.assertEqual(1, len(results))
        row = results[0]
        self.assertEqual("vec::norm(fixed_size_list(f32, 768))", row["name"])
        self.assertEqual("Vector ops", row["family"])
        self.assertEqual("reference", row["baseline_kind"])
        self.assertEqual({}, unpaired)

    def test_pairs_truncated_directories_via_full_id(self):
        # Criterion truncates directory names to ~64 chars; the full benchmark
        # id must come from benchmark.json or long llvm/arrow pairs never match.
        long_name = "cmp::lt(run_end_encoded(i32, dictionary(i8, i64)), scalar(i64)) 1m rows"
        self.write_estimate(
            "cmp__lt(run_end_encoded(i32, dictionary(i8, i64)), scalar(i64)) ",
            10,
            full_id=f"{long_name}/llvm warm",
        )
        self.write_estimate(
            "cmp__lt(run_end_encoded(i32, dictionary(i8, i64)), scalar(i64))_2",
            20,
            full_id=f"{long_name}/arrow",
        )

        results, unpaired = report.collect(self.criterion_dir)

        self.assertEqual(1, len(results))
        self.assertEqual(long_name, results[0]["name"])
        self.assertEqual({}, unpaired)

    def test_compile_phase_entries_stay_unpaired(self):
        self.write_estimate("sort__sort_to_indices(array(u64))_llvm compile", 100)
        self.write_estimate("sort__sort_to_indices(array(u64))_llvm warm", 10)
        self.write_estimate("sort__sort_to_indices(array(u64))_arrow", 20)

        results, unpaired = report.collect(self.criterion_dir)

        self.assertEqual(1, len(results))
        self.assertEqual("sort::sort_to_indices(array(u64))", results[0]["name"])
        self.assertEqual(
            ["sort__sort_to_indices(array(u64))_llvm compile"], list(unpaired)
        )

    def test_verdict_equal_band_and_interval_bounds(self):
        equal = report.verdict(
            {"point": 100, "lower": 99, "upper": 101},
            {"point": 102, "lower": 101, "upper": 103},
        )
        # 3.3% apart with non-overlapping intervals still counts as equal: the
        # band is 3.5% so a ratio displayed as "1.03x" can never be a win.
        boundary = report.verdict(
            {"point": 100, "lower": 99.5, "upper": 100.5},
            {"point": 103.3, "lower": 102.8, "upper": 103.8},
        )
        self.assertEqual("equal", boundary[1])
        llvm_win = report.verdict(
            {"point": 10, "lower": 9, "upper": 11},
            {"point": 20, "lower": 18, "upper": 22},
        )
        arrow_win = report.verdict(
            {"point": 20, "lower": 18, "upper": 22},
            {"point": 10, "lower": 9, "upper": 11},
        )
        # Beyond the margin but with overlapping intervals: the measurement
        # cannot separate the sides, so it is reported as equal (there is no
        # separate "inconclusive" class).
        overlap = report.verdict(
            {"point": 10, "lower": 9, "upper": 11},
            {"point": 11, "lower": 9.5, "upper": 12.5},
        )

        self.assertEqual("equal", equal[1])
        self.assertEqual("llvm-win", llvm_win[1])
        self.assertEqual("arrow-win", arrow_win[1])
        self.assertEqual("equal", overlap[1])

    def test_sorts_by_family_then_fixed_group_then_speedup(self):
        self.write_estimate("vec__dot(a)/llvm warm", 10)
        self.write_estimate("vec__dot(a)_rust reference", 40)
        self.write_estimate("cmp__lt(slow)/llvm warm", 20)
        self.write_estimate("cmp__lt(slow)/arrow", 10)
        self.write_estimate("cmp__lt(fast)/llvm warm", 10)
        self.write_estimate("cmp__lt(fast)/arrow", 20)
        # bounds loses worse than any lt row, but group order is fixed:
        # the primary operator (cmp::lt) stays first regardless of results.
        self.write_estimate("cmp__bounds(b)/llvm warm", 100)
        self.write_estimate("cmp__bounds(b)/arrow", 10)

        results, _ = report.collect(self.criterion_dir)

        # Families in fixed order; groups fixed (primary first, then
        # alphabetical); rows inside a group worst-first (ascending speedup).
        self.assertEqual(
            ["cmp::lt(slow)", "cmp::lt(fast)", "cmp::bounds(b)", "vec::dot(a)"],
            [row["name"] for row in results],
        )

    def test_builds_serializable_report_data(self):
        self.write_estimate("cmp__lt(array(i32), scalar(i32))/llvm warm", 10)
        self.write_estimate("cmp__lt(array(i32), scalar(i32))/arrow", 20)
        manifest = {
            "revision": "abc1234",
            "dirty": False,
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:01:00+00:00",
            "commands": [["cargo", "bench"]],
        }

        data = report.build_report_data(self.criterion_dir, manifest)

        self.assertEqual(2, data["schema_version"])
        self.assertEqual("abc1234", data["run"]["revision"])
        self.assertEqual(report.EQUAL_MARGIN, data["equal_margin"])
        self.assertEqual("llvm-win", data["results"][0]["classification"])
        json.dumps(data)

    def test_op_inputs_size_rendering(self):
        cases = [
            (
                "cmp::lt(array(i32), scalar(i32)) 1m rows",
                ("cmp::lt", "i32, i32 scalar", "1m rows"),
            ),
            (
                "cmp::lt(run_end_encoded(i32, dictionary(i8, i32)), scalar(i32)) 1m rows",
                ("cmp::lt", "ree i32→dict i8→i32, i32 scalar", "1m rows"),
            ),
            (
                "cast::cast(array(u64) to dictionary(i16, u64)) 10m rows",
                ("cast::cast", "u64 to dict i16→u64", "10m rows"),
            ),
            (
                "vec::norm(fixed_size_list(f32, 768)) 16384 rows",
                ("vec::norm", "f32[768]", "16384 rows"),
            ),
            (
                "sort::multicol_sort_to_indices(array(u64) x7) 8 word key 1m rows",
                ("sort::multicol_sort_to_indices", "u64 x7", "8 word key 1m rows"),
            ),
            (
                "sort::sort_to_indices(array(nullable u64)) 1m rows",
                ("sort::sort_to_indices", "nullable u64", "1m rows"),
            ),
        ]
        for name, expected in cases:
            self.assertEqual(expected, report.op_inputs_size(name), name)

    def test_rejects_duplicate_normalized_phase(self):
        self.write_estimate("filter_llvm warm", 10)
        self.write_estimate("filter/llvm_warm", 11)

        with self.assertRaisesRegex(ValueError, "duplicate llvm result"):
            report.collect(self.criterion_dir)

    def test_rejects_incomplete_run_manifest(self):
        (self.criterion_dir / report.MANIFEST_NAME).write_text(
            json.dumps({"status": "failed", "criterion_dir": "criterion"})
        )

        with self.assertRaisesRegex(ValueError, "not complete"):
            report.load_manifest(self.criterion_dir)


if __name__ == "__main__":
    unittest.main()
