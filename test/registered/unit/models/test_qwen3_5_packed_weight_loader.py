"""
Unit tests for Qwen3_5GatedDeltaNet._make_packed_weight_loader.

Validates that per-tensor FP8 scales (scalar or single-element tensors)
are broadcast to every logical shard, while normal multi-element weights
are split correctly.

Regression test for https://github.com/sgl-project/sglang/issues/23051
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.models.qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    _reorder_grouped_v_heads_to_tiled,
    _restore_gguf_qwen35_linear_attention_tensor,
    _restore_gguf_tiled_v_heads,
)


def _make_mock_module(output_sizes):
    """Create a lightweight mock module with the attributes needed by the loader."""
    return SimpleNamespace(output_sizes=output_sizes)


def _make_per_tensor_scale_param(num_shards):
    """Create a PerTensorScaleParameter pre-allocated for `num_shards` scales.

    PerTensorScaleParameter requires a weight_loader callable;
    we supply a no-op since the packed loader wraps it anyway.
    """
    return PerTensorScaleParameter(
        data=torch.zeros(num_shards),
        weight_loader=lambda *args, **kwargs: None,
    )


class TestMakePackedWeightLoader(unittest.TestCase):
    """Tests for _make_packed_weight_loader broadcast / split logic."""

    # ------------------------------------------------------------------ #
    #  Per-tensor scale broadcast                                         #
    # ------------------------------------------------------------------ #

    def test_scalar_weight_broadcast(self):
        """A 0-d scalar should be broadcast (via .view(-1)) to every shard."""
        module = _make_mock_module(output_sizes=[128, 128, 64, 64])
        param = _make_per_tensor_scale_param(num_shards=4)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scalar = torch.tensor(0.5)  # shape=[]
        loader(param, scalar, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        for shard_id, chunk in calls:
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.5, places=5)

    def test_single_element_tensor_broadcast(self):
        """A [1]-shaped tensor (e.g. per-tensor weight_scale) should be
        broadcast to every logical shard."""
        module = _make_mock_module(output_sizes=[128, 128, 64, 64])
        param = _make_per_tensor_scale_param(num_shards=4)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scale = torch.tensor([0.25])  # shape=[1]
        loader(param, scale, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        for idx, (shard_id, chunk) in enumerate(calls):
            self.assertEqual(shard_id, idx)
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.25, places=5)

    def test_broadcast_with_two_shards(self):
        """Broadcast for in_proj_ba style (2 shards: b, a)."""
        module = _make_mock_module(output_sizes=[16, 16])
        param = _make_per_tensor_scale_param(num_shards=2)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scale = torch.tensor([0.1])
        loader(param, scale, loaded_shard_id=(0, 1))

        self.assertEqual(len(calls), 2)
        for shard_id, chunk in calls:
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.1, places=5)

    # ------------------------------------------------------------------ #
    #  Normal weight split                                                #
    # ------------------------------------------------------------------ #

    def test_normal_weight_split(self):
        """Multi-element weights should be split by output_sizes, not broadcast."""
        module = _make_mock_module(output_sizes=[128, 128, 64])
        param = MagicMock()
        param.output_dim = 0

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        # Simulate a checkpoint weight that covers shard 0, 1, 2
        weight = torch.randn(128 + 128 + 64, 256)
        loader(param, weight, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][1].shape[0], 128)
        self.assertEqual(calls[1][1].shape[0], 128)
        self.assertEqual(calls[2][1].shape[0], 64)

    # ------------------------------------------------------------------ #
    #  Passthrough for non-tuple shard_id                                 #
    # ------------------------------------------------------------------ #

    def test_int_shard_id_passthrough(self):
        """An int shard_id should bypass the tuple logic entirely."""
        module = _make_mock_module(output_sizes=[128, 128, 64, 64])

        calls = []

        def original_loader(p, loaded_weight, shard_id):
            calls.append(("original", shard_id))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        weight = torch.randn(128, 256)
        loader(MagicMock(), weight, loaded_shard_id=2)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ("original", 2))

    def test_none_shard_id_passthrough(self):
        """None shard_id should pass through to the original loader."""
        module = _make_mock_module(output_sizes=[128])

        calls = []

        def original_loader(p, loaded_weight, shard_id):
            calls.append(("original", shard_id))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        weight = torch.randn(128, 256)
        loader(MagicMock(), weight, loaded_shard_id=None)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ("original", None))

    # ------------------------------------------------------------------ #
    #  Edge case: nested single-element tensors                           #
    # ------------------------------------------------------------------ #

    def test_nested_single_element_tensor_broadcast(self):
        """A [[value]] shaped tensor (numel==1, ndim==2) should also broadcast."""
        module = _make_mock_module(output_sizes=[128, 128, 64])
        param = _make_per_tensor_scale_param(num_shards=3)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scale = torch.tensor([[0.75]])  # shape=[1,1], numel==1
        loader(param, scale, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        for shard_id, chunk in calls:
            # .view(-1) should flatten to [1]
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.75, places=5)


class TestQwen35GGUFVHeadLayout(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            linear_num_key_heads=2,
            linear_num_value_heads=6,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
        )

    def test_grouped_and_tiled_layouts_round_trip(self):
        grouped = torch.arange(2 * 6 * 2).reshape(2, 12)
        tiled = _reorder_grouped_v_heads_to_tiled(grouped, -1, 2, 6, 2)

        self.assertFalse(torch.equal(tiled, grouped))
        restored = _restore_gguf_tiled_v_heads(tiled, -1, 2, 6, 2)
        torch.testing.assert_close(restored, grouped)

    def test_restores_quantized_qkv_value_rows(self):
        q = torch.full((4, 3), 10, dtype=torch.uint8)
        k = torch.full((4, 3), 20, dtype=torch.uint8)
        grouped_v = torch.arange(12 * 3, dtype=torch.uint8).reshape(12, 3)
        tiled_v = _reorder_grouped_v_heads_to_tiled(grouped_v, 0, 2, 6, 2)
        converted = torch.cat((q, k, tiled_v), dim=0)

        restored = _restore_gguf_qwen35_linear_attention_tensor(
            "model.layers.0.linear_attn.in_proj_qkv.qweight",
            converted,
            self.config,
        )

        torch.testing.assert_close(restored[:4], q)
        torch.testing.assert_close(restored[4:8], k)
        torch.testing.assert_close(restored[8:], grouped_v)

    def test_restores_conv_value_channels_only(self):
        qk = torch.full((8, 4), -1.0)
        grouped_v = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)
        tiled_v = _reorder_grouped_v_heads_to_tiled(grouped_v, 0, 2, 6, 2)

        restored = _restore_gguf_qwen35_linear_attention_tensor(
            "model.layers.0.linear_attn.conv1d.weight",
            torch.cat((qk, tiled_v), dim=0),
            self.config,
        )

        torch.testing.assert_close(restored[:8], qk)
        torch.testing.assert_close(restored[8:], grouped_v)


if __name__ == "__main__":
    unittest.main()
