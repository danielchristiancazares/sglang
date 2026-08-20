import importlib.util
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[4]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gap_analysis = _load(
    "analyze_graph_gap_timing",
    Path("scripts/windows/analyze_graph_gap_timing.py"),
)
trace_analysis = _load(
    "analyze_torch_trace_for_gap_test",
    Path("scripts/windows/analyze_torch_trace.py"),
)


def _record(category, value):
    return {
        "artifact_type": "speculative_graph_gap",
        "gap_before": category,
        "elapsed_ms": value,
    }


class TestGraphGapTiming(unittest.TestCase):
    def test_repeatable_p10_controls_the_0_75ms_admission_gate(self):
        records = [
            _record("draft", 0.80 + (index % 3) * 0.005) for index in range(28)
        ]
        records.extend(
            _record("target_verify", 0.50 + (index % 2) * 0.01)
            for index in range(28)
        )

        report = gap_analysis.summarize_gaps(records)

        draft = report["transitions"]["draft"]
        self.assertTrue(draft["repeatable"])
        self.assertGreaterEqual(draft["repeatable_recoverable_ms"], 0.75)
        self.assertTrue(draft["passes_0_75ms_gate"])
        self.assertFalse(
            report["transitions"]["target_verify"]["passes_0_75ms_gate"]
        )
        self.assertTrue(report["admission_gate"]["fund_graph_tail_work"])
        self.assertEqual(report["admission_gate"]["best_transition"], "draft")

    def test_insufficient_or_variable_samples_have_zero_recoverable_time(self):
        insufficient = [_record("draft_extend", 1.0) for _ in range(10)]
        variable = [
            _record("draft", 0.1 if index % 2 else 2.0) for index in range(30)
        ]

        report = gap_analysis.summarize_gaps(insufficient + variable)

        self.assertEqual(
            report["transitions"]["draft_extend"][
                "repeatable_recoverable_ms"
            ],
            0.0,
        )
        self.assertFalse(report["transitions"]["draft"]["repeatable"])
        self.assertFalse(report["admission_gate"]["fund_graph_tail_work"])

    def test_trace_full_cycle_uses_dominant_target_start_to_start(self):
        # Tuple: graph id, start_us, end_us, kernel_us, kernel_count.
        runs = [
            (2, 0.0, 15000.0, 14000.0, 100),
            (8, 16000.0, 17000.0, 900.0, 10),
            (5, 18000.0, 19000.0, 900.0, 10),
            (2, 20000.0, 35000.0, 14000.0, 100),
            (8, 36000.0, 37000.0, 900.0, 10),
            (5, 38000.0, 39000.0, 900.0, 10),
            (2, 41000.0, 56000.0, 14000.0, 100),
        ]

        samples = trace_analysis._cycle_samples_from_graph_runs(runs)

        self.assertEqual(samples["anchor_graph_id"], 2)
        self.assertEqual(samples["samples_ms"], [20.0, 21.0])
        self.assertAlmostEqual(samples["mean_ms"], 20.5)


if __name__ == "__main__":
    unittest.main()
