import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.model_runner_components import (
    attention_backend_setup,
)
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    ResolvedAttentionBackendStr,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        # Real backends always carry this (AttentionBackend class attribute).
        self.needs_cpu_seq_lens = True


def test_split_full_attention_applies_model_wrapper_once():
    runner = SimpleNamespace(
        server_args=SimpleNamespace(speculative_attention_mode="prefill"),
        model_config=SimpleNamespace(context_len=2048),
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        init_new_workspace=None,
    )
    wrapper_inputs = []
    wrapped_backend = object()

    def wrap_once(model_runner, backend):
        assert model_runner is runner
        wrapper_inputs.append(backend)
        return wrapped_backend

    constructors = {
        "decode-test": lambda model_runner: _FakeBackend("decode"),
        "prefill-test": lambda model_runner: _FakeBackend("prefill"),
    }
    resolved = ResolvedAttentionBackendStr(decode="decode-test", prefill="prefill-test")

    with (
        patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
        patch.object(
            attention_backend_setup,
            "attn_backend_wrapper",
            side_effect=wrap_once,
        ),
    ):
        result = attention_backend_setup._build_resolved_backend(
            model_runner=runner,
            resolved=resolved,
            init_new_workspace=True,
        )

    assert result is wrapped_backend
    assert len(wrapper_inputs) == 1
    split_backend = wrapper_inputs[0]
    assert isinstance(split_backend, HybridAttnBackend)
    assert split_backend.decode_backend.name == "decode"
    assert split_backend.prefill_backend.name == "prefill"
    assert runner.init_new_workspace is True


def test_sm120_trtllm_draft_splits_initial_prefill_and_routes_draft_extend():
    runner = SimpleNamespace(
        is_draft_worker=True,
        draft_attention_backend="trtllm_mha",
        server_args=SimpleNamespace(speculative_attention_mode="decode"),
        model_config=SimpleNamespace(context_len=2048),
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        init_new_workspace=None,
    )

    with (
        patch.object(attention_backend_setup, "is_sm120_supported", return_value=True),
        patch.object(
            attention_backend_setup,
            "attention_backends",
            return_value=("flashinfer", "trtllm_mha"),
        ),
    ):
        resolved = attention_backend_setup.resolve_attention_backend_strs(
            model_runner=runner
        )

    assert resolved == ResolvedAttentionBackendStr(
        prefill="flashinfer",
        decode="trtllm_mha",
        is_draft_override=True,
    )

    constructors = {
        "trtllm_mha": lambda model_runner: _FakeBackend("decode"),
        "flashinfer": lambda model_runner: _FakeBackend("prefill"),
    }
    with (
        patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
        patch.object(
            attention_backend_setup,
            "attn_backend_wrapper",
            side_effect=lambda model_runner, backend: backend,
        ),
    ):
        backend = attention_backend_setup._build_resolved_backend(
            model_runner=runner,
            resolved=resolved,
            init_new_workspace=False,
        )

    assert isinstance(backend, HybridAttnBackend)
    assert backend._select_backend(ForwardMode.EXTEND).name == "prefill"
    assert backend._select_backend(ForwardMode.DECODE).name == "decode"
    assert backend._select_backend(ForwardMode.DRAFT_EXTEND_V2).name == "decode"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
