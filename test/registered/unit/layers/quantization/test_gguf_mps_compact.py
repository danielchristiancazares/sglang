import unittest
from unittest.mock import patch

import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization import gguf as gguf_quant


class TestGGUFMPSCompactShards(unittest.TestCase):
    @staticmethod
    def _make_layer(shards, shard_types):
        layer = torch.nn.Module()
        qweight = Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
        qweight.data_container = [data for _, data in shards]
        qweight.shard_id = [name for name, _ in shards]
        qweight.shard_id_map = {
            name: index for index, (name, _) in enumerate(shards)
        }
        qweight.tensor_shape = (
            sum(data.shape[0] for _, data in shards),
            32,
        )
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.zeros(len(shards), dtype=torch.uint8), requires_grad=False
        )
        qweight_type.weight_type = next(iter(shard_types.values()))
        qweight_type.shard_weight_type = shard_types
        layer.register_parameter("qweight_type", qweight_type)
        return layer

    def test_mixed_width_shards_share_compact_storage_in_logical_order(self):
        qkv = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.uint8)
        z = torch.tensor([[9, 10], [11, 12], [13, 14]], dtype=torch.uint8)
        layer = self._make_layer(
            [("z", z), ("qkv", qkv)],
            {"qkv": WeightType.Q4_K, "z": WeightType.IQ2_XXS},
        )
        method = gguf_quant.GGUFLinearMethod(gguf_quant.GGUFConfig())

        with patch.object(gguf_quant, "_is_mps", True):
            method.process_weights_after_loading(layer)

        self.assertEqual(layer.qweight.numel(), qkv.numel() + z.numel())
        self.assertEqual(
            layer.qweight.compact_shard_map,
            {"qkv": (0, 2, 4), "z": (8, 3, 2)},
        )
        torch.testing.assert_close(
            layer.qweight,
            torch.cat((qkv.reshape(-1), z.reshape(-1))),
        )

        calls = []

        def fake_matmul(x, weight, weight_type):
            calls.append(
                (
                    weight.shape,
                    weight.is_contiguous(),
                    weight.untyped_storage().data_ptr(),
                    weight_type,
                )
            )
            return weight.float().sum(dim=1).expand(x.shape[0], -1)

        with patch.object(gguf_quant, "fused_mul_mat_gguf", fake_matmul):
            output = method.apply(layer, torch.ones(1, 32))

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[1] for call in calls))
        self.assertTrue(
            all(
                call[2] == layer.qweight.untyped_storage().data_ptr()
                for call in calls
            )
        )
        torch.testing.assert_close(
            output,
            torch.tensor([[10.0, 26.0, 19.0, 23.0, 27.0]]),
        )

    def test_equal_width_and_type_shards_still_use_one_matmul(self):
        up = torch.tensor([[7, 8, 9]], dtype=torch.uint8)
        gate = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint8)
        layer = self._make_layer(
            [("up", up), ("gate", gate)],
            {"gate": WeightType.Q4_K, "up": WeightType.Q4_K},
        )
        method = gguf_quant.GGUFLinearMethod(gguf_quant.GGUFConfig())

        with patch.object(gguf_quant, "_is_mps", True):
            method.process_weights_after_loading(layer)

        calls = []

        def fake_matmul(x, weight, weight_type):
            calls.append((weight.shape, weight.untyped_storage().data_ptr()))
            return weight.float().sum(dim=1).expand(x.shape[0], -1)

        with patch.object(gguf_quant, "fused_mul_mat_gguf", fake_matmul):
            output = method.apply(layer, torch.ones(1, 32))

        self.assertEqual(
            calls,
            [(torch.Size([3, 3]), layer.qweight.untyped_storage().data_ptr())],
        )
        torch.testing.assert_close(output, torch.tensor([[6.0, 15.0, 24.0]]))

    def test_non_mps_path_keeps_padded_merged_parameter(self):
        narrow = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
        wide = torch.tensor([[5, 6, 7, 8]], dtype=torch.uint8)
        layer = self._make_layer(
            [("narrow", narrow), ("wide", wide)],
            {"narrow": WeightType.Q4_K, "wide": WeightType.Q4_K},
        )
        method = gguf_quant.GGUFLinearMethod(gguf_quant.GGUFConfig())

        with patch.object(gguf_quant, "_is_mps", False):
            method.process_weights_after_loading(layer)

        self.assertFalse(hasattr(layer.qweight, "compact_shard_map"))
        self.assertEqual(layer.qweight.shape, torch.Size([3, 4]))
        self.assertEqual(
            layer.qweight.shard_offset_map,
            {"narrow": (0, 2, 2), "wide": (2, 3, 4)},
        )


if __name__ == "__main__":
    unittest.main()
