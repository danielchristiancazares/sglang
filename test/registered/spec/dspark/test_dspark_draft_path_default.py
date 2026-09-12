import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _target_checkpoint_bundles_dspark_draft,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.models.dspark import DSparkDraftMixin, project_through_lm_head
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    _base_logits_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BUNDLED_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
_PLAIN_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _bundled_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )


def _plain_hf_config() -> SimpleNamespace:
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


def _make_dspark_server_args(
    *, model_path: str, hf_config: SimpleNamespace
) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = model_path
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = None
    server_args.speculative_dspark_block_size = 5
    server_args.model_config = SimpleNamespace(hf_config=hf_config)
    return server_args


class TestTargetCheckpointBundlesDsparkDraft(CustomTestCase):
    def test_bundled_dsv4_config_is_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        self.assertTrue(_target_checkpoint_bundles_dspark_draft(server_args))

    def test_plain_target_config_is_not_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        self.assertFalse(_target_checkpoint_bundles_dspark_draft(server_args))


class TestDsparkDraftPathDefaulting(CustomTestCase):
    def test_bundled_checkpoint_defaults_draft_path_to_model_path(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _handle_dspark(server_args)
        self.assertEqual(server_args.speculative_draft_model_path, _BUNDLED_MODEL_PATH)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_plain_target_without_draft_path_raises(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        with self.assertRaises(ValueError):
            _handle_dspark(server_args)

    def test_explicit_draft_path_is_not_overwritten(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_draft_model_path = "deepseek-ai/some-other-dspark-draft"
        _handle_dspark(server_args)
        self.assertEqual(
            server_args.speculative_draft_model_path,
            "deepseek-ai/some-other-dspark-draft",
        )


class TestDsparkDpAttentionMoeA2aGate(CustomTestCase):
    """Gate contract for DSpark + dp attention + MoE a2a backends."""

    def _dp_server_args(self, *, moe_a2a_backend: str) -> ServerArgs:
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.enable_dp_attention = True
        server_args.enable_dp_lm_head = True
        server_args.dp_size = 2
        server_args.tp_size = 2
        server_args.moe_a2a_backend = moe_a2a_backend
        return server_args

    def test_only_megamoe_is_admitted(self):
        """Both sides of the allowlist: megamoe passes, others raise by name."""
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            _handle_dspark(self._dp_server_args(moe_a2a_backend="megamoe"))
            for backend in ("deepep", "pplx"):
                with self.assertRaisesRegex(ValueError, backend):
                    _handle_dspark(self._dp_server_args(moe_a2a_backend=backend))

    def test_a2a_backend_with_compact_verify_mode_raises(self):
        server_args = self._dp_server_args(moe_a2a_backend="megamoe")
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            with self.assertRaisesRegex(ValueError, "static"):
                _handle_dspark(server_args)


class _PackedQuantMethod:
    def __init__(self, output: torch.Tensor):
        self.output = output
        self.calls = []

    def apply(self, layer, hidden, bias):
        self.calls.append((layer, hidden, bias))
        return self.output


class TestDsparkQuantizedTargetHead(CustomTestCase):
    def test_qwen_conditional_wrapper_exposes_aux_capture(self):
        for method_name in (
            "get_input_embeddings",
            "set_dflash_layers_to_capture",
        ):
            self.assertTrue(
                callable(
                    getattr(
                        Qwen3_5ForConditionalGeneration,
                        method_name,
                        None,
                    )
                ),
                method_name,
            )

    def test_packed_target_head_uses_its_quantized_projection(self):
        hidden = torch.randn(3, 4)
        expected = torch.randn(3, 7)
        quant_method = _PackedQuantMethod(expected)
        lm_head = SimpleNamespace(
            weight=torch.zeros((7, 2), dtype=torch.uint8),
            quant_method=quant_method,
        )

        actual = project_through_lm_head(hidden, lm_head)

        self.assertIs(actual, expected)
        self.assertEqual(quant_method.calls, [(lm_head, hidden, None)])

    def test_compute_base_logits_returns_the_quantized_projection(self):
        hidden = torch.randn(3, 4)
        expected = torch.randn(3, 7)
        quant_method = _PackedQuantMethod(expected)
        lm_head = SimpleNamespace(
            weight=torch.zeros((7, 2), dtype=torch.uint8),
            quant_method=quant_method,
        )
        model = SimpleNamespace(
            lm_head=lm_head,
            logits_mup_width_multiplier=None,
        )

        with patch(
            "sglang.srt.models.dspark.gather_and_crop_vocab",
            side_effect=lambda logits, _lm_head: logits,
        ):
            actual, confidence_tap = DSparkDraftMixin.compute_base_logits(
                model, hidden
            )

        self.assertIs(actual, expected)
        self.assertIsNone(confidence_tap)

    def test_dense_target_head_keeps_the_dense_projection(self):
        hidden = torch.randn(3, 4)
        weight = torch.randn(7, 4)
        lm_head = SimpleNamespace(weight=weight, quant_method=None)

        actual = project_through_lm_head(hidden, lm_head)

        self.assertTrue(torch.equal(actual, hidden @ weight.T))

    def test_packed_head_graph_buffers_follow_runtime_logits_dtype(self):
        markov_head = torch.nn.Linear(4, 7, bias=False, dtype=torch.bfloat16)
        model = SimpleNamespace(
            lm_head=SimpleNamespace(weight=torch.zeros((7, 2), dtype=torch.uint8)),
            markov_head=markov_head,
        )

        self.assertEqual(_base_logits_dtype(model), torch.bfloat16)


class TestDsparkReplaySsmCommit(CustomTestCase):
    def test_direct_dspark_commit_uses_the_gdn_fold_path(self):
        spec_state = object()
        state_batch_indices = torch.tensor([3, 5], dtype=torch.int32)
        track_indices = torch.tensor([7, 9], dtype=torch.int32)
        translated_track_indices = track_indices + 10
        steps_to_track = torch.tensor([1, -1], dtype=torch.int64)
        last_correct = torch.tensor([2, 4], dtype=torch.int64)
        req_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(
                replayssm_spec_fold=True,
                replayssm_is_kda=False,
            ),
            get_speculative_mamba2_params_all_layers=lambda: spec_state,
        )
        linear_backend = SimpleNamespace(
            req_to_token_pool=req_pool,
            forward_metadata=SimpleNamespace(
                mamba_cache_indices=state_batch_indices,
            ),
            _translate_mamba_indices=lambda value: value + 10,
        )
        backend = SimpleNamespace(linear_attn_backend=linear_backend)

        with patch(
            "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold.commit_gdn_replayssm_fold_after_verify"
        ) as commit:
            HybridLinearAttnBackend.update_mamba_state_after_mtp_verify(
                backend,
                last_correct_step_indices=last_correct,
                mamba_track_indices=track_indices,
                mamba_steps_to_track=steps_to_track,
                model=object(),
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            )

        commit.assert_called_once()
        kwargs = commit.call_args.kwargs
        self.assertIs(kwargs["spec_state"], spec_state)
        self.assertTrue(
            torch.equal(kwargs["state_batch_indices"], state_batch_indices)
        )
        self.assertTrue(torch.equal(kwargs["accept_lens"], last_correct + 1))
        self.assertTrue(
            torch.equal(kwargs["last_correct_step_indices"], last_correct)
        )
        self.assertTrue(
            torch.equal(kwargs["mamba_track_indices"], translated_track_indices)
        )
        self.assertTrue(torch.equal(kwargs["mamba_steps_to_track"], steps_to_track))


if __name__ == "__main__":
    unittest.main()
