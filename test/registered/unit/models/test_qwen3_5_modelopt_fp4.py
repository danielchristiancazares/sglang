"""Unit tests for modelopt_fp4 checkpoints that quantize Qwen3.5 attention.

Covers three things:
  1. ModelOptFp4Config.is_layer_excluded() decides per prefix whether attention is
     quantized or kept in BF16.
  2. RadixAttention registers k_scale/v_scale when built with a quant_config that
     declares kv_cache_quant_algo; without them, baked KV scales have nowhere to
     load into and silently fall back to 1.0.
  3. QWEN3_5_KV_SCALE_MAPPER remaps the checkpoint's baked KV-scale names onto the
     RadixAttention parameter names.
"""

import sys
import unittest
from unittest import mock

import torch
import torch.nn as nn

import sglang.srt.models.qwen3_5 as qwen3_5
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import QWEN3_5_KV_SCALE_MAPPER
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Projection(nn.Module):
    def forward(self, x):
        return x, None


class _RecordingActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + 1


class _RecordingDownProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_scale_inv = nn.Parameter(
            torch.tensor(1.0),
            requires_grad=False,
        )
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return x, None


@unittest.skipUnless(sys.platform == "win32", "Native Windows routing only")
class TestWindowsQwen3_5Nvfp4Routing(CustomTestCase):
    def _make_mlp(self):
        mlp = qwen3_5.Qwen2MoeMLP.__new__(qwen3_5.Qwen2MoeMLP)
        nn.Module.__init__(mlp)
        mlp.gate_up_proj = _Projection()
        mlp.down_proj = _RecordingDownProjection()
        mlp.act_fn = _RecordingActivation()
        mlp._use_silu_and_mul_nvfp4 = True
        return mlp

    def test_eager_forward_uses_prequantized_tuple(self):
        mlp = self._make_mlp()
        input = torch.tensor([[1.0]])
        packed = torch.tensor([[2]], dtype=torch.uint8)
        scales = torch.tensor([[3]], dtype=torch.uint8)

        with (
            mock.patch.object(torch.compiler, "is_compiling", return_value=False),
            mock.patch.object(
                qwen3_5,
                "silu_and_mul_nvfp4",
                return_value=(packed, scales),
            ) as fused,
        ):
            output = mlp(input)

        fused.assert_called_once_with(input, mlp.down_proj.input_scale_inv)
        self.assertIs(output[0], packed)
        self.assertIs(output[1], scales)
        self.assertEqual(mlp.act_fn.calls, 0)

    def test_compile_forward_preserves_native_activation(self):
        mlp = self._make_mlp()
        input = torch.tensor([[1.0]])

        with (
            mock.patch.object(torch.compiler, "is_compiling", return_value=True),
            mock.patch.object(qwen3_5, "silu_and_mul_nvfp4") as fused,
        ):
            output = mlp(input)

        fused.assert_not_called()
        torch.testing.assert_close(output, input + 1)
        self.assertEqual(mlp.act_fn.calls, 1)
        self.assertIsInstance(mlp.down_proj.last_input, torch.Tensor)


class TestModelOptFp4AttentionExclusion(CustomTestCase):
    def test_moe_only_checkpoint_excludes_attention(self):
        # NVIDIA's Qwen3.5 NVFP4 checkpoints: attention and lm_head are excluded,
        # MoE experts are not.
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo="FP8",
            group_size=16,
            exclude_modules=["*self_attn*", "lm_head"],
        )

        self.assertTrue(cfg.is_layer_excluded("model.layers.0.self_attn.qkv_proj"))
        self.assertTrue(cfg.is_layer_excluded("lm_head"))
        self.assertFalse(
            cfg.is_layer_excluded("model.layers.0.mlp.experts.3.gate_up_proj")
        )

    def test_uniform_w4a4_checkpoint_quantizes_attention(self):
        # Uniform W4A4 checkpoint: only lm_head is excluded, attention is quantized.
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo="FP8",
            group_size=16,
            exclude_modules=["lm_head"],
        )

        self.assertFalse(cfg.is_layer_excluded("model.layers.0.self_attn.qkv_proj"))
        self.assertFalse(
            cfg.is_layer_excluded("model.layers.0.linear_attn.in_proj_qkvz")
        )
        self.assertTrue(cfg.is_layer_excluded("lm_head"))


class TestRadixAttentionKvScaleRegistration(CustomTestCase):
    def _make_attn(self, quant_config):
        return RadixAttention(
            num_heads=2,
            head_dim=8,
            scaling=1.0,
            num_kv_heads=2,
            layer_id=0,
            quant_config=quant_config,
            prefix="model.layers.0.attn",
        )

    def test_with_fp8_kv_quant_config_registers_scale_params(self):
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo="FP8",
            group_size=16,
            exclude_modules=[],
        )
        attn = self._make_attn(cfg)

        self.assertIsInstance(attn.k_scale, torch.nn.Parameter)
        self.assertIsInstance(attn.v_scale, torch.nn.Parameter)
        # create_weights seeds -1.0, the sentinel for "checkpoint had no scale".
        self.assertEqual(attn.k_scale.item(), -1.0)
        self.assertEqual(attn.v_scale.item(), -1.0)

    def test_without_quant_config_has_no_scale_params(self):
        attn = self._make_attn(None)

        self.assertIsNone(attn.k_scale)
        self.assertIsNone(attn.v_scale)

    def test_quant_config_without_kv_cache_algo_has_no_scale_params(self):
        # Registration is gated on kv_cache_quant_algo, not on quant_config alone.
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            group_size=16,
            exclude_modules=[],
        )
        attn = self._make_attn(cfg)

        self.assertIsNone(attn.k_scale)
        self.assertIsNone(attn.v_scale)


class TestQwen3_5KvScaleMapper(CustomTestCase):
    def test_maps_baked_kv_scale_names_onto_radix_attention(self):
        # Source names come from ModelOpt's export format, target names from the
        # sglang module tree; a typo on either side silently zeroes the scales.
        weights = [
            ("model.layers.3.self_attn.k_proj.k_scale", torch.tensor(0.0347)),
            ("model.layers.3.self_attn.v_proj.v_scale", torch.tensor(0.0128)),
        ]

        mapped = list(QWEN3_5_KV_SCALE_MAPPER.apply(weights))

        self.assertEqual(
            [name for name, _ in mapped],
            ["model.layers.3.attn.k_scale", "model.layers.3.attn.v_scale"],
        )
        torch.testing.assert_close(mapped[0][1], torch.tensor(0.0347))
        torch.testing.assert_close(mapped[1][1], torch.tensor(0.0128))

    def test_all_other_names_pass_through_unchanged(self):
        # A mapping key that is too broad would corrupt regular weight loading.
        names = [
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.k_proj.input_scale",
            "model.layers.3.self_attn.k_proj.weight_scale",
            "model.layers.2.linear_attn.in_proj_qkvz.weight",
            "model.layers.0.mlp.experts.5.down_proj.weight",
            "lm_head.weight",
        ]
        weights = [(name, torch.zeros(1)) for name in names]

        mapped = list(QWEN3_5_KV_SCALE_MAPPER.apply(weights))

        self.assertEqual([name for name, _ in mapped], names)

    def test_mapped_scale_loads_via_default_weight_loader(self):
        # The scale params carry no weight_loader, so load_weights' fallback uses
        # default_weight_loader; its scalar path is what tolerates the 0-dim param
        # vs the shape-[1] checkpoint tensor.
        scale_param = torch.nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )
        loaded_weight = torch.tensor([0.0347], dtype=torch.float32)

        weight_loader = getattr(scale_param, "weight_loader", default_weight_loader)
        weight_loader(scale_param, loaded_weight)

        self.assertAlmostEqual(scale_param.item(), 0.0347, places=6)


if __name__ == "__main__":
    unittest.main()
