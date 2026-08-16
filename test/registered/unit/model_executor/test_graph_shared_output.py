from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_max_decode_logits_rows_uses_adaptive_draft_token_bound(monkeypatch):
    runner = object.__new__(ModelRunner)
    runner.server_args = SimpleNamespace(max_speculative_num_draft_tokens=4)
    runner.spec_algorithm = SimpleNamespace(is_speculative=lambda: True)
    runner.decode_num_tokens_per_req = Mock(return_value=4)

    monkeypatch.setattr(
        model_runner_module,
        "get_batch_sizes_to_capture",
        lambda _runner, _num_tokens_per_req: ([1], None),
    )

    assert runner.max_decode_logits_rows() == 4
    runner.decode_num_tokens_per_req.assert_called_once_with(num_draft_tokens=4)
