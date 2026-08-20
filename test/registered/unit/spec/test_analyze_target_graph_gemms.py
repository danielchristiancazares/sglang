import importlib.util
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


SCRIPT_DIR = Path(__file__).resolve().parents[4] / "scripts" / "windows"
SOURCE = SCRIPT_DIR / "analyze_target_graph_gemms.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("analyze_target_graph_gemms", SOURCE)
assert SPEC is not None and SPEC.loader is not None
attribution = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = attribution
SPEC.loader.exec_module(attribution)


def _event(
    name,
    ts,
    dur,
    *,
    graph_id=2,
    correlation=100,
    stream=1,
    grid=(1, 1, 1),
    block=(128, 1, 1),
):
    return {
        "cat": "kernel",
        "name": name,
        "ts": float(ts),
        "dur": float(dur),
        "args": {
            "graph id": graph_id,
            "correlation": correlation,
            "stream": stream,
            "grid": list(grid),
            "block": list(block),
            "shared memory": 0,
        },
    }


FP8 = "sm89_xmma_gemm_e4m3bf16_tilesize32x64x64_stage5"
NVFP4 = "cutlass_SM120_BLOCKSCALED_test"
BF16 = "cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_align8"


def _config():
    return {
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 16,
            "intermediate_size": 32,
            "vocab_size": 64,
            "head_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
        },
        "quantization_config": {
            "config_groups": {
                "fp8": {"weights": {"num_bits": 8}},
                "nvfp4": {"weights": {"num_bits": 4}},
            }
        },
    }


def _exact_run_events(correlation=100, offset=0.0):
    # Per-family ordering is the exact contract.  Cross-family timestamps can
    # interleave because the linear-attention BA projection uses an alt stream.
    events = []
    timestamp = offset
    for name in (FP8, FP8, FP8, FP8):
        events.append(_event(name, timestamp, 2, correlation=correlation))
        timestamp += 3
    events.append(
        _event(BF16, offset + 0.5, 1, correlation=correlation, stream=2)
    )
    timestamp = offset + 1
    for name in (NVFP4, NVFP4, NVFP4, NVFP4, NVFP4):
        events.append(_event(name, timestamp, 1, correlation=correlation))
        timestamp += 3
    return events


class TestAnalyzeTargetGraphGemms(unittest.TestCase):
    def test_attention_output_gate_has_a_dedicated_family(self):
        self.assertEqual(
            attribution.kernel_family("sglang_fused_sigmoid_mul_kernel"),
            "sigmoid_and_mul",
        )

    def test_interval_union_handles_overlap_and_gaps(self):
        self.assertEqual(
            attribution.interval_union_us([(0, 5), (2, 7), (10, 11)]), 8
        )

    def test_graph_replays_are_separated_by_launch_correlation(self):
        events = (
            _exact_run_events(100, 0)
            + _exact_run_events(101, 100)
            + [_event("ordinary", 200, 1, graph_id=0, correlation=102)]
        )

        runs = attribution.graph_runs_from_events(events)

        self.assertEqual(len(runs), 2)
        self.assertEqual([run.correlation for run in runs], [100, 101])

    def test_overlap_metrics_keep_aggregate_and_exposure_separate(self):
        events = [
            _event("main-a", 0, 10, stream=1),
            _event("alt", 2, 5, stream=2),
            _event("main-b", 10, 10, stream=1),
        ]
        run = attribution.graph_runs_from_events(events)[0]

        alt = attribution.summarize_event_group(
            [run], lambda _run_index, event: event["name"] == "alt"
        )
        main = attribution.summarize_event_group(
            [run], lambda _run_index, event: event["name"].startswith("main")
        )

        self.assertEqual(alt["aggregate_kernel_time"]["mean_ms"], 0.005)
        self.assertEqual(
            alt["critical_path_exposure"]["completion_stream_serialized"][
                "mean_ms"
            ],
            0.0,
        )
        self.assertEqual(
            alt["critical_path_exposure"]["exclusive_observed_wall"]["mean_ms"],
            0.0,
        )
        self.assertEqual(
            main["critical_path_exposure"]["completion_stream_serialized"][
                "mean_ms"
            ],
            0.02,
        )
        self.assertEqual(
            main["critical_path_exposure"]["exclusive_observed_wall"]["mean_ms"],
            0.015,
        )

    def test_qwen35_problem_shapes_and_exact_replay_match(self):
        roles = attribution.qwen35_target_roles(_config(), 3, 1)
        runs = attribution.graph_runs_from_events(_exact_run_events())

        report = attribution.qwen35_role_attribution(runs, roles)

        self.assertEqual(report["status"], "exact")
        self.assertEqual(
            report["expected_primary_gemms_per_replay"],
            {"bf16": 1, "fp8": 4, "nvfp4": 5},
        )
        gate_up = next(
            item
            for item in report["by_model_role"]
            if item["model_role"] == "mlp.gate_up_proj"
        )
        self.assertEqual(
            gate_up["problem_shapes_mnk"], [{"m": 3, "n": 64, "k": 16}]
        )
        linear_qkvz = next(
            item
            for item in report["by_model_role"]
            if item["model_role"] == "linear_attention.in_proj_qkvz"
        )
        self.assertEqual(
            linear_qkvz["problem_shapes_mnk"],
            [{"m": 3, "n": 24, "k": 16}],
        )

    def test_model_role_attribution_fails_closed_on_count_drift(self):
        roles = attribution.qwen35_target_roles(_config(), 3, 1)
        events = _exact_run_events()
        events.pop(next(index for index, event in enumerate(events) if event["name"] == FP8))
        runs = attribution.graph_runs_from_events(events)

        with self.assertRaisesRegex(attribution.AttributionError, "requires 4"):
            attribution.qwen35_role_attribution(runs, roles)


if __name__ == "__main__":
    unittest.main()
