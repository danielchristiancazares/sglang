import unittest
from types import SimpleNamespace

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.torch_native_backend import (
    TorchNativeAttnBackend,
    _native_mps_decode_gqa_inputs_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TensorMetadata:
    def __init__(
        self,
        shape,
        *,
        dtype=torch.float32,
        device="mps",
        contiguous=True,
    ):
        self.shape = torch.Size(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


class _TokenPool:
    kv_cache_layout = "nhd"

    def __init__(self, key_cache, value_cache):
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.full_kv_pool = self

    def get_key_buffer(self, layer_id):
        del layer_id
        return self.key_cache

    def get_value_buffer(self, layer_id):
        del layer_id
        return self.value_cache

    def set_kv_buffer(self, layer, loc_info, key, value):
        del layer
        locations = loc_info.loc.long()
        self.key_cache[locations] = key.reshape_as(self.key_cache[locations]).to(
            self.key_cache.dtype
        )
        self.value_cache[locations] = value.reshape_as(self.value_cache[locations]).to(
            self.value_cache.dtype
        )


class TestTorchNativeMPSDecodeDispatch(unittest.TestCase):
    @staticmethod
    def _supported(**overrides):
        metadata = {
            "query": _TensorMetadata((1, 24 * 256)),
            "key": _TensorMetadata((1, 4 * 256)),
            "value": _TensorMetadata((1, 4 * 256)),
            "key_cache": _TensorMetadata((7936, 4, 256)),
            "value_cache": _TensorMetadata((7936, 4, 256)),
            "head_dim": 256,
        }
        metadata.update(overrides)
        return _native_mps_decode_gqa_inputs_supported(**metadata)

    def test_native_capability_boundary(self):
        self.assertTrue(self._supported())
        self.assertFalse(self._supported(key_cache=_TensorMetadata((7937, 4, 256))))
        self.assertFalse(
            self._supported(
                key_cache=_TensorMetadata((7937, 4, 256)),
                value_cache=_TensorMetadata((7937, 4, 256)),
            )
        )
        self.assertFalse(self._supported(head_dim=257))

    def test_native_capability_rejects_incompatible_dtype_device_and_layout(self):
        self.assertFalse(
            self._supported(
                key_cache=_TensorMetadata((7936, 4, 256), dtype=torch.bfloat16)
            )
        )
        self.assertFalse(
            self._supported(query=_TensorMetadata((1, 24 * 256), dtype=torch.bfloat16))
        )
        self.assertFalse(
            self._supported(value=_TensorMetadata((1, 4 * 256), device="cpu"))
        )
        self.assertFalse(
            self._supported(key_cache=_TensorMetadata((7936, 4, 256), contiguous=False))
        )
        self.assertFalse(self._supported(value_cache=_TensorMetadata((7936, 4, 128))))

    def test_bf16_fallback_writes_current_kv_before_gqa(self):
        torch.manual_seed(53)
        query_heads, kv_heads, head_dim = 4, 2, 8
        scale = head_dim**-0.5
        key_cache = torch.randn(10, kv_heads, head_dim, dtype=torch.bfloat16)
        value_cache = torch.randn_like(key_cache)
        req_to_token = torch.tensor([[4, 1, 7]], dtype=torch.int32)
        cache_location = torch.tensor([7])
        query = torch.randn(1, query_heads * head_dim)
        key = torch.full((1, kv_heads * head_dim), 3.25)
        value = torch.full((1, kv_heads * head_dim), -2.5)

        token_pool = _TokenPool(key_cache, value_cache)
        backend = TorchNativeAttnBackend.__new__(TorchNativeAttnBackend)
        backend.token_to_kv_pool = token_pool
        backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        backend.swa_out_cache_loc = None
        backend.seq_lens_override = None
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=query_heads,
            tp_k_head_num=kv_heads,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            is_cross_attention=False,
            sliding_window_size=None,
            scaling=scale,
        )
        forward_batch = SimpleNamespace(
            out_cache_loc=cache_location,
            encoder_out_cache_loc=None,
            seq_lens=torch.tensor([3]),
            req_pool_indices=torch.tensor([0]),
            encoder_lens=None,
        )

        output = backend.forward_decode(query, key, value, layer, forward_batch)
        slots = req_to_token[0].long()
        expected = scaled_dot_product_attention(
            query.view(1, query_heads, 1, head_dim),
            key_cache[slots].movedim(0, 1).unsqueeze(0).float(),
            value_cache[slots].movedim(0, 1).unsqueeze(0).float(),
            is_causal=False,
            enable_gqa=True,
            scale=scale,
        ).view_as(output)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(
            key_cache[cache_location],
            key.view(1, kv_heads, head_dim).to(torch.bfloat16),
        )
        torch.testing.assert_close(
            value_cache[cache_location],
            value.view(1, kv_heads, head_dim).to(torch.bfloat16),
        )


if __name__ == "__main__":
    unittest.main()
