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

    def write_estimate(self, name, point, lower=None, upper=None):
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

    def test_collects_nested_criterion_groups(self):
        self.write_estimate(
            "select__take(fixed_size_list(bool, 1024), array(u64))/llvm warm", 10
        )
        self.write_estimate(
            "select__take(fixed_size_list(bool, 1024), array(u64))/arrow", 20
        )

        matrix, other, unpaired = report.collect(self.criterion_dir)

        self.assertEqual([], matrix)
        self.assertEqual(
            "select::take(fixed_size_list(bool, 1024), array(u64))",
            other[0]["name"],
        )
        self.assertEqual({}, unpaired)

    def test_preserves_function_underscores_and_recognizes_direct_calls(self):
        self.write_estimate("sort__sort_to_indices(array(u64))_llvm compile", 100)
        self.write_estimate("sort__sort_to_indices(array(u64))_llvm direct", 10)
        self.write_estimate("sort__sort_to_indices(array(u64))_arrow", 20)

        matrix, other, unpaired = report.collect(self.criterion_dir)

        self.assertEqual([], other)
        self.assertEqual("sort::sort_to_indices(array(u64))", matrix[0]["name"])
        self.assertEqual({}, unpaired)

    def test_verdict_requires_non_overlapping_intervals(self):
        llvm_win = report.verdict(
            {"point": 10, "lower": 9, "upper": 11},
            {"point": 20, "lower": 18, "upper": 22},
        )
        arrow_win = report.verdict(
            {"point": 20, "lower": 18, "upper": 22},
            {"point": 10, "lower": 9, "upper": 11},
        )
        overlap = report.verdict(
            {"point": 10, "lower": 9, "upper": 11},
            {"point": 10.5, "lower": 9.5, "upper": 11.5},
        )

        self.assertEqual("llvm-win", llvm_win[1])
        self.assertEqual("arrow-win", arrow_win[1])
        self.assertEqual("inconclusive", overlap[1])

    def test_builds_serializable_report_data(self):
        self.write_estimate("filter_llvm", 10)
        self.write_estimate("filter_arrow", 20)
        manifest = {
            "revision": "abc1234",
            "dirty": False,
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:01:00+00:00",
            "commands": [["cargo", "bench"]],
        }

        data = report.build_report_data(self.criterion_dir, manifest)

        self.assertEqual(1, data["schema_version"])
        self.assertEqual("abc1234", data["run"]["revision"])
        self.assertEqual("llvm-win", data["other"][0]["classification"])
        json.dumps(data)

    def test_rejects_duplicate_normalized_phase(self):
        self.write_estimate("filter_llvm execute", 10)
        self.write_estimate("filter/llvm_execute", 11)

        with self.assertRaisesRegex(ValueError, "duplicate execute result"):
            report.collect(self.criterion_dir)

    def test_rejects_incomplete_run_manifest(self):
        (self.criterion_dir / report.MANIFEST_NAME).write_text(
            json.dumps({"status": "failed", "criterion_dir": "criterion"})
        )

        with self.assertRaisesRegex(ValueError, "not complete"):
            report.load_manifest(self.criterion_dir)


if __name__ == "__main__":
    unittest.main()
