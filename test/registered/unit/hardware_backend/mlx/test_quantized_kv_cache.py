"""Tests for the geometric MLX long-context KV cache."""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None

if _HAS_MLX:
    import mlx.core as mx
    from sglang.srt.hardware_backend.mlx.kv_cache import (
        BatchedDecodeContext,
        ContiguousAttentionKVCache,
        MLXAttentionWrapper,
        QuantizedAttentionKVCache,
    )
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner


class TestQuantizedKVServerArgs(unittest.TestCase):
    def test_quantized_kv_requires_radix_cache_disabled(self):
        args = ServerArgs(model_path="dummy")
        args.mlx_kv_cache_bits = 4
        args.disable_radix_cache = False
        with self.assertRaisesRegex(ValueError, "requires --disable-radix-cache"):
            args._handle_other_validations()

    def test_quantized_kv_accepts_long_context_profile(self):
        args = ServerArgs(model_path="dummy")
        args.mlx_kv_cache_bits = 4
        args.mlx_kv_cache_group_size = 64
        args.disable_radix_cache = True
        args._handle_other_validations()


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestQuantizedAttentionKVCache(unittest.TestCase):
    def test_preallocated_round_trip_and_capacity(self):
        cache = QuantizedAttentionKVCache(max_seq_len=8, group_size=32, bits=4)
        keys = mx.arange(2 * 64, dtype=mx.float32).reshape(1, 1, 2, 64) / 64
        values = -keys

        q_keys, q_values = cache.update_and_fetch(keys, values)
        decoded_keys = mx.dequantize(
            *q_keys, group_size=cache.group_size, bits=cache.bits
        )
        decoded_values = mx.dequantize(
            *q_values, group_size=cache.group_size, bits=cache.bits
        )
        mx.eval(decoded_keys, decoded_values)

        self.assertEqual(cache.offset, 2)
        self.assertEqual(cache.keys[0].shape[2], 8)
        self.assertEqual(decoded_keys.shape, keys.shape)
        self.assertLess(mx.max(mx.abs(decoded_keys - keys)).item(), 0.04)
        self.assertLess(mx.max(mx.abs(decoded_values - values)).item(), 0.07)

        cache.write_token(
            mx.ones((1, 1, 1, 64), dtype=mx.float32),
            mx.ones((1, 1, 1, 64), dtype=mx.float32),
        )
        self.assertEqual(cache.offset, 3)
        self.assertEqual(cache.keys[0].shape[2], 8)

    def test_capacity_overflow_is_explicit(self):
        cache = QuantizedAttentionKVCache(max_seq_len=2, group_size=32, bits=4)
        item = mx.zeros((1, 1, 3, 64), dtype=mx.float32)
        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            cache.update_and_fetch(item, item)

    def test_native_context_starts_small_and_grows_geometrically(self):
        cache = QuantizedAttentionKVCache(max_seq_len=262144, group_size=64, bits=4)
        first = mx.zeros((1, 1, 1, 64), dtype=mx.float32)
        cache.update_and_fetch(first, first)
        self.assertEqual(cache.capacity, 4096)

        chunk = mx.ones((1, 1, 4096, 64), dtype=mx.float32)
        keys, _ = cache.update_and_fetch(chunk, chunk)
        decoded = mx.dequantize(*keys, group_size=cache.group_size, bits=cache.bits)
        mx.eval(decoded)
        self.assertEqual(cache.capacity, 8192)
        self.assertEqual(cache.offset, 4097)
        self.assertEqual(decoded[0, 0, 0, 0].item(), 0.0)

    def test_model_runner_wires_quantized_full_attention(self):
        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner._cache_layout = SimpleNamespace(
            has_auxiliary_state=False,
            num_layers=1,
            attention_layer_indices=(0,),
            window_size=lambda _: None,
        )
        runner._kv_cache_bits = 4
        runner._kv_cache_group_size = 64
        runner._pool_size = 1024
        runner._max_seq_len = 4096

        cache = runner._new_native_cache()

        self.assertIsInstance(cache[0], QuantizedAttentionKVCache)
        self.assertEqual(cache[0].max_seq_len, 1024)

    def test_batched_wrapper_dispatches_quantized_attention(self):
        width = 64

        class IdentityProjection:
            def __call__(self, x):
                return x

        class IdentityRope:
            def __call__(self, x, offset):
                return x

        inner = SimpleNamespace(
            q_proj=IdentityProjection(),
            k_proj=IdentityProjection(),
            v_proj=IdentityProjection(),
            o_proj=IdentityProjection(),
            rope=IdentityRope(),
            n_heads=1,
            n_kv_heads=1,
            head_dim=width,
            scale=width**-0.5,
        )
        wrapper = MLXAttentionWrapper(inner, layer_idx=0)
        x = mx.arange(width, dtype=mx.float32).reshape(1, 1, width) / width

        float_cache = ContiguousAttentionKVCache(
            n_kv_heads=1,
            head_dim=width,
            max_seq_len=4,
            dtype=mx.float32,
        )
        quant_cache = QuantizedAttentionKVCache(max_seq_len=4, group_size=32, bits=4)
        float_ctx = BatchedDecodeContext(
            batch_size=1,
            seq_lens=[0],
            attention_layer_caches=[[float_cache]],
        )
        quant_ctx = BatchedDecodeContext(
            batch_size=1,
            seq_lens=[0],
            attention_layer_caches=[[quant_cache]],
        )

        expected = wrapper._batched_decode(x, float_ctx)
        actual = wrapper._batched_decode(x, quant_ctx)
        mx.eval(expected, actual)

        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 0.04)


if __name__ == "__main__":
    unittest.main()
