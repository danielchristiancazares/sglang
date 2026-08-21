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
    from mlx_lm.models.qwen3_next import Qwen3NextAttention
    from sglang.srt.hardware_backend.mlx.kv_cache import (
        BatchedDecodeContext,
        ContiguousAttentionKVCache,
        MLXAttentionWrapper,
        QuantizedAttentionKVCache,
        tiled_quantized_scaled_dot_product_attention,
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

    def test_query_tiled_quantized_attention_matches_full_causal_result(self):
        def make_queries():
            queries = mx.arange(4 * 5 * 32, dtype=mx.float32).reshape(
                1, 4, 5, 32
            )
            return (queries % 37) / 37

        keys = mx.arange(2 * 8 * 32, dtype=mx.float32).reshape(1, 2, 8, 32)
        keys = ((keys * 3) % 41) / 41
        values = mx.arange(2 * 8 * 32, dtype=mx.float32).reshape(1, 2, 8, 32)
        values = ((values * 5) % 43) / 43
        q_keys = mx.quantize(keys, group_size=32, bits=4)
        q_values = mx.quantize(values, group_size=32, bits=4)

        expected = tiled_quantized_scaled_dot_product_attention(
            make_queries(),
            q_keys,
            q_values,
            scale=32**-0.5,
            mask="causal",
            group_size=32,
            bits=4,
            query_tile_size=0,
        )
        actual = tiled_quantized_scaled_dot_product_attention(
            make_queries(),
            q_keys,
            q_values,
            scale=32**-0.5,
            mask="causal",
            group_size=32,
            bits=4,
            query_tile_size=2,
        )
        mx.eval(expected, actual)

        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-6)

    def test_qwen35_quantized_prefill_matches_wrapped_attention(self):
        args = SimpleNamespace(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            attention_bias=False,
            rms_norm_eps=1e-6,
            partial_rotary_factor=1.0,
            rope_theta=1_000_000.0,
            rope_scaling=None,
            max_position_embeddings=4096,
        )
        inner = Qwen3NextAttention(args)
        wrapper = MLXAttentionWrapper(inner, layer_idx=0)
        object.__setattr__(wrapper, "_quantized_prefill_query_tile", 2)
        x = mx.arange(5 * args.hidden_size, dtype=mx.float32).reshape(
            1, 5, args.hidden_size
        )
        x = ((x * 7) % 101) / 101
        full_cache = QuantizedAttentionKVCache(
            max_seq_len=8, group_size=32, bits=4
        )
        tiled_cache = QuantizedAttentionKVCache(
            max_seq_len=8, group_size=32, bits=4
        )

        expected = inner(x, mask="causal", cache=full_cache)
        actual = wrapper._quantized_prefill(x, "causal", tiled_cache)
        mx.eval(expected, actual)

        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(tiled_cache.offset, full_cache.offset)
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-6)

    def test_quantized_prefill_score_estimate_tracks_context_growth(self):
        inner = SimpleNamespace(
            q_proj=None,
            k_proj=None,
            v_proj=None,
            o_proj=None,
            rope=None,
            n_heads=24,
            n_kv_heads=4,
            head_dim=256,
            scale=256**-0.5,
        )
        wrapper = MLXAttentionWrapper(inner, layer_idx=0)
        chunk = SimpleNamespace(
            shape=(1, 4096, 5120),
            dtype=SimpleNamespace(size=2),
        )

        first = wrapper._quantized_prefill_score_bytes(
            chunk, SimpleNamespace(offset=0)
        )
        at_observed_failure = wrapper._quantized_prefill_score_bytes(
            chunk, SimpleNamespace(offset=98304)
        )

        self.assertEqual(first, 4096 * 4096 * 50)
        self.assertEqual(at_observed_failure, 4096 * 102400 * 50)


if __name__ == "__main__":
    unittest.main()
