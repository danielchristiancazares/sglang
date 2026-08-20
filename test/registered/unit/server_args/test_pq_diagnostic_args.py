from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_diagnostic_flags_are_default_inactive():
    args = ServerArgs(model_path="dummy")

    assert args.speculative_pq_capture_path is None
    assert args.speculative_pq_capture_max_cycles == 256
    assert args.speculative_graph_gap_timing_path is None
    assert args.speculative_graph_gap_timing_max_samples == 2048


def test_diagnostic_cli_flags_round_trip():
    args = prepare_server_args(
        [
            "--model-path",
            "dummy",
            "--speculative-pq-capture-path",
            "capture.jsonl",
            "--speculative-pq-capture-max-cycles",
            "17",
            "--speculative-graph-gap-timing-path",
            "gaps.jsonl",
            "--speculative-graph-gap-timing-max-samples",
            "31",
        ]
    )

    assert args.speculative_pq_capture_path == "capture.jsonl"
    assert args.speculative_pq_capture_max_cycles == 17
    assert args.speculative_graph_gap_timing_path == "gaps.jsonl"
    assert args.speculative_graph_gap_timing_max_samples == 31
