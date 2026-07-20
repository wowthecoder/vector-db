import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("generate_report.py")
SPEC = importlib.util.spec_from_file_location("generate_report", SCRIPT_PATH)
assert SPEC and SPEC.loader
report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report
SPEC.loader.exec_module(report)


def benchmark_row(name, real_time=1.0, cpu_time=1.1, unit="ms", **extra):
    row = {
        "name": name,
        "run_name": name,
        "run_type": "iteration",
        "repetitions": 1,
        "repetition_index": 0,
        "threads": 1,
        "iterations": 10,
        "real_time": real_time,
        "cpu_time": cpu_time,
        "time_unit": unit,
    }
    row.update(extra)
    return row


class ReportTestCase(unittest.TestCase):
    def write_document(self, document):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "results.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_parses_every_project_benchmark_name(self):
        cases = {
            "BM_CollectionInsert/1000/128": ("insert", None, {"count": 1000, "dimension": 128}),
            "BM_CollectionSave/10000/128/real_time": ("save", None, {"count": 10000, "dimension": 128}),
            "BM_CollectionLoad/1000/128/real_time": ("load", None, {"count": 1000, "dimension": 128}),
            "BM_CollectionSearch<vectordb::Metric::L2>/count:10000/dimension:128/top_k:10": (
                "search", "L2", {"count": 10000, "dimension": 128, "top_k": 10}
            ),
            "BM_CollectionBatchSearch<vectordb::Metric::Dot>/count:10000/dimension:128/query_count:8/top_k:10": (
                "batch_search", "Dot", {"count": 10000, "dimension": 128, "query_count": 8, "top_k": 10}
            ),
            "BM_RepeatedSingleSearch<vectordb::Metric::Cosine>/count:10000/dimension:128/query_count:8/top_k:10": (
                "repeated_search", "Cosine", {"count": 10000, "dimension": 128, "query_count": 8, "top_k": 10}
            ),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(report.parse_benchmark_name(name), expected)

    def test_normalizes_supported_time_units_to_milliseconds(self):
        rows = [
            benchmark_row("BM_ns/1", 2_000_000, 2_000_000, "ns"),
            benchmark_row("BM_us/1", 2_000, 2_000, "us"),
            benchmark_row("BM_ms/1", 2, 2, "ms"),
            benchmark_row("BM_s/1", 0.002, 0.002, "s"),
        ]
        _, results, _ = report.load_results(self.write_document({"benchmarks": rows}))
        self.assertEqual([result.real_ms for result in results], [2.0, 2.0, 2.0, 2.0])

    def test_prefers_aggregate_median_and_attaches_cv(self):
        run_name = "BM_CollectionInsert/1000/128"
        median = benchmark_row(run_name, 3.0, 4.0, "ms", items_per_second=500)
        median.update({"name": run_name + "_median", "run_type": "aggregate", "aggregate_name": "median", "aggregate_unit": "time", "repetitions": 5})
        cv = benchmark_row(run_name, 0.125, 0.1, "ms")
        cv.update({"name": run_name + "_cv", "run_type": "aggregate", "aggregate_name": "cv", "aggregate_unit": "percentage", "repetitions": 5})
        _, results, _ = report.load_results(self.write_document({"benchmarks": [median, cv]}))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].real_ms, 3.0)
        self.assertEqual(results[0].cv_percent, 12.5)
        self.assertEqual(results[0].source, "median")

    def test_computes_median_and_cv_from_raw_repetitions(self):
        name = "BM_CollectionInsert/1000/128"
        rows = [benchmark_row(name, value, value) for value in (1.0, 2.0, 3.0)]
        for index, row in enumerate(rows):
            row.update({"repetitions": 3, "repetition_index": index})
        _, results, _ = report.load_results(self.write_document({"benchmarks": rows}))
        self.assertEqual(results[0].real_ms, 2.0)
        self.assertEqual(results[0].cv_percent, 50.0)
        self.assertEqual(results[0].source, "computed median")

    def test_keeps_unknown_families_and_missing_optional_counters(self):
        row = benchmark_row("BM_CustomThing/value:7", 4.0, 5.0)
        _, results, warnings = report.load_results(self.write_document({"context": None, "benchmarks": [row]}))
        self.assertEqual(results[0].operation, "unknown")
        self.assertEqual(results[0].params, {"value": 7})
        self.assertIsNone(results[0].items_per_second)
        self.assertEqual(warnings, [])

    def test_skips_error_records_and_reports_warning(self):
        failed = benchmark_row("BM_Failed")
        failed.update({"error_occurred": True, "error_message": "boom"})
        good = benchmark_row("BM_Good")
        _, results, warnings = report.load_results(self.write_document({"benchmarks": [failed, good]}))
        self.assertEqual(len(results), 1)
        self.assertIn("boom", warnings[0])

    def test_rendered_report_is_escaped_self_contained_and_complete(self):
        rows = [
            benchmark_row("BM_CollectionInsert/10000/128", items_per_second=2_000_000),
            benchmark_row("BM_CollectionSave/10000/128/real_time", 5.0, 3.0),
            benchmark_row("BM_CollectionLoad/10000/128/real_time", 2.0, 1.5),
        ]
        for metric in ("L2", "Dot", "Cosine"):
            rows.extend([
                benchmark_row(f"BM_CollectionSearch<vectordb::Metric::{metric}>/count:10000/dimension:128/top_k:10", 2.0, 2.1, "ms", items_per_second=500),
                benchmark_row(f"BM_CollectionBatchSearch<vectordb::Metric::{metric}>/count:10000/dimension:128/query_count:8/top_k:10", 12.0, 13.0, "ms", items_per_second=600),
                benchmark_row(f"BM_RepeatedSingleSearch<vectordb::Metric::{metric}>/count:10000/dimension:128/query_count:8/top_k:10", 13.0, 14.0, "ms", items_per_second=550),
            ])
        context, results, warnings = report.load_results(self.write_document({"context": {"library_build_type": "release"}, "benchmarks": rows}))
        rendered = report.render_report("<Unsafe & title>", context, results, warnings)
        self.assertIn("&lt;Unsafe &amp; title&gt;", rendered)
        self.assertIn("Search scaling", rendered)
        self.assertIn("Batch search versus repeated searches", rendered)
        self.assertIn("Insertion and persistence", rendered)
        self.assertIn("Complete results", rendered)
        self.assertIn("only one repetition", rendered)
        self.assertNotIn("https://", rendered)
        self.assertNotIn("http://", rendered)
        self.assertGreaterEqual(rendered.count('data-search="'), len(results))

    def test_invalid_json_has_clear_cli_error(self):
        path = self.write_document({"not_benchmarks": []})
        error_output = io.StringIO()
        with contextlib.redirect_stderr(error_output):
            return_code = report.main([str(path)])
        self.assertEqual(return_code, 1)
        self.assertIn("benchmarks", error_output.getvalue())

    def test_malformed_json_has_clear_cli_error(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "broken.json"
        path.write_text("{not valid JSON", encoding="utf-8")
        error_output = io.StringIO()
        with contextlib.redirect_stderr(error_output):
            return_code = report.main([str(path)])
        self.assertEqual(return_code, 1)
        self.assertIn("invalid JSON", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
