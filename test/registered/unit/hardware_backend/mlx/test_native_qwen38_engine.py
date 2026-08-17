"""Shipped-path tests for the compiled Qwen3.8 MLX C++ graph."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_mlx_ci(est_time=15, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_MODEL = Path.home() / (
    ".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/"
    "snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
)
_MTP = Path.home() / (
    ".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-MTP-4bit/"
    "snapshots/b643c01b6d3b094e325edb6ebd832e16c486c575"
)


class TestNativeQwen38Engine(CustomTestCase):
    def test_config_from_json_reads_shipped_text_config(self):
        """Parser must take the Qwen3.8 text_config numbers, not vision."""
        from sglang.srt.hardware_backend.mlx.native_engine import (
            NativeQwen38Engine,
        )

        cfg = NativeQwen38Engine.config_from_json(_MODEL.joinpath("config.json").read_text())
        self.assertEqual(cfg.hidden_size, 5120)
        self.assertEqual(cfg.num_hidden_layers, 64)
        self.assertEqual(cfg.vocab_size, 248320)
        self.assertEqual(cfg.num_attention_heads, 24)
        self.assertEqual(cfg.num_key_value_heads, 4)
        self.assertEqual(cfg.head_dim, 256)
        self.assertEqual(cfg.linear_num_value_heads, 48)
        self.assertEqual(cfg.linear_num_key_heads, 16)
        self.assertEqual(cfg.full_attention_interval, 4)
        self.assertEqual(cfg.quant_group_size, 64)
        self.assertEqual(cfg.quant_bits, 4)

    @unittest.skipUnless(_HAS_MLX, "requires mlx")
    def test_qlinear_matches_mlx_quantized_matmul(self):
        """C++ quantized_matmul must implement the same affine-q4 product
        mlx.core.quantize + quantized_matmul use. A drift here would
        silently change every 27B linear."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.native_engine import qlinear

        rng = np.random.default_rng(0)
        x = rng.standard_normal((3, 64), dtype=np.float32)
        w = rng.standard_normal((32, 64), dtype=np.float32)
        qw, scales, biases = mx.quantize(mx.array(w), group_size=64, bits=4)
        mx.eval(qw, scales, biases)
        expected = mx.quantized_matmul(
            mx.array(x), qw, scales, biases, transpose=True, group_size=64, bits=4
        )
        mx.eval(expected)
        got = qlinear(
            x,
            np.array(qw),
            np.array(scales),
            np.array(biases),
            group_size=64,
            bits=4,
        )
        np.testing.assert_allclose(got, np.array(expected), rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(_HAS_MLX, "requires mlx")
    def test_gated_delta_step_matches_mlx_lm_ops(self):
        """One recurrent Gated-DeltaNet step must match mlx-lm's ops path.
        The 48 linear-attention layers are this recurrence."""
        import mlx.core as mx
        from mlx_lm.models.gated_delta import _gated_delta_step_ops

        from sglang.srt.hardware_backend.mlx.native_engine import gated_delta_step

        rng = np.random.default_rng(1)
        h_k, h_v, d_k, d_v = 2, 4, 8, 8
        q = rng.standard_normal((h_k, d_k), dtype=np.float32)
        k = rng.standard_normal((h_k, d_k), dtype=np.float32)
        v = rng.standard_normal((h_v, d_v), dtype=np.float32)
        g = rng.random((h_v,), dtype=np.float32) * 0.5 + 0.4
        beta = rng.random((h_v,), dtype=np.float32)
        state = rng.standard_normal((h_v, d_v, d_k), dtype=np.float32)

        q_rep = np.repeat(q, h_v // h_k, axis=0)
        k_rep = np.repeat(k, h_v // h_k, axis=0)
        y_ref, st_ref = _gated_delta_step_ops(
            mx.array(q_rep)[None],
            mx.array(k_rep)[None],
            mx.array(v)[None],
            mx.array(g)[None],
            mx.array(beta)[None],
            mx.array(state)[None],
        )
        mx.eval(y_ref, st_ref)
        y, st = gated_delta_step(q, k, v, g, beta, state)
        np.testing.assert_allclose(y, np.array(y_ref[0]), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(st, np.array(st_ref[0]), rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(_HAS_MLX, "requires mlx")
    def test_gated_delta_metal_matches_mlx_lm_ops(self):
        """The shipped Metal recurrence must match mlx-lm's ops path over T>1.

        Dk=32 is the kernel's simd width. A grid/template mismatch would
        still compile and produce garbage the single-step ops test never
        sees.
        """
        import mlx.core as mx
        from mlx_lm.models.gated_delta import gated_delta_ops

        from sglang.srt.hardware_backend.mlx.native_engine import (
            gated_delta_update,
        )

        if not mx.metal.is_available():
            self.skipTest("requires Metal")

        rng = np.random.default_rng(2)
        t, h_k, h_v, d_k, d_v = 4, 2, 4, 32, 32
        q = rng.standard_normal((t, h_k, d_k), dtype=np.float32)
        k = rng.standard_normal((t, h_k, d_k), dtype=np.float32)
        v = rng.standard_normal((t, h_v, d_v), dtype=np.float32)
        g = rng.random((t, h_v), dtype=np.float32) * 0.5 + 0.4
        beta = rng.random((t, h_v), dtype=np.float32)
        state = rng.standard_normal((h_v, d_v, d_k), dtype=np.float32)

        y_ref, st_ref = gated_delta_ops(
            mx.array(q)[None],
            mx.array(k)[None],
            mx.array(v)[None],
            mx.array(g)[None],
            mx.array(beta)[None],
            mx.array(state)[None],
        )
        mx.eval(y_ref, st_ref)
        y, st = gated_delta_update(q, k, v, g, beta, state)
        np.testing.assert_allclose(y, np.array(y_ref[0]), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(st, np.array(st_ref[0]), rtol=1e-3, atol=1e-3)

    @unittest.skipUnless(_HAS_MLX, "requires mlx")
    def test_chained_native_decode_uses_prev_tokens_not_stale_req_ids(self):
        """A chained decode must feed the previous step's tokens into the
        compiled engine. ``req_token_ids`` is only appended in finalize,
        so reading it would replay the prefill token and desynchronize
        Gated-DeltaNet state.
        """
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.model_runner import (
            MlxModelRunner,
            MlxPendingDecode,
        )

        class Recorder:
            def __init__(self) -> None:
                self.seen: list[int] = []

            def decode(self, token: int) -> int:
                self.seen.append(int(token))
                return int(token) + 1

        runner = object.__new__(MlxModelRunner)
        engine = Recorder()
        runner._native_engine = engine
        runner._req_token_ids = {"r0": [10, 20]}
        runner._req_caches = {"r0": []}
        prev = MlxPendingDecode(
            lazy_tokens=mx.array([77], dtype=mx.int32),
            req_ids=["r0"],
            caches=[[]],
        )
        out = runner.decode_batch_start_chained(prev)
        self.assertEqual(engine.seen, [77])
        self.assertEqual(int(out.lazy_tokens.item()), 78)

    def test_native_engine_skips_python_auxiliary_snapshot(self):
        """The compiled engine keeps Gated-DeltaNet state in C++. Warmup
        finalize used to index an empty Python cache list and crash.
        """
        from types import SimpleNamespace

        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        class Boom:
            def store_cache(self, *args, **kwargs):
                raise AssertionError("native graph must not snapshot Python cache")

            def restore_cache(self, *args, **kwargs):
                raise AssertionError("native graph must not restore Python cache")

        runner = object.__new__(MlxModelRunner)
        runner._native_engine = object()
        runner._req_to_token_pool = SimpleNamespace(
            get_auxiliary_state_indices=lambda _idx: 0,
            auxiliary_state_pool=Boom(),
        )
        runner._store_auxiliary_state(0, [])
        self.assertFalse(runner._restore_auxiliary_state(0, []))

    @unittest.skipUnless(_HAS_MLX and _MTP.is_dir(), "requires local MTP snapshot")
    def test_mtp_load_rejects_missing_dir(self):
        """load_mtp must fail on a directory with no safetensors rather than
        silently running single-token decode."""
        from sglang.srt.hardware_backend.mlx.native_engine import NativeQwen38Engine

        if not _MODEL.is_dir():
            self.skipTest("requires local 27B snapshot")
        eng = NativeQwen38Engine.load(str(_MODEL))
        self.assertFalse(eng.has_mtp())
        with self.assertRaises(RuntimeError):
            eng.load_mtp("/tmp")
        self.assertFalse(eng.has_mtp())
        eng.close()

    @unittest.skipUnless(
        _HAS_MLX and _MODEL.is_dir() and _MTP.is_dir(),
        "requires local 27B and MTP snapshots",
    )
    def test_mtp_spec_step_emits_more_than_one_token_per_refill(self):
        """A working MTP drafter plus target verify must accept at least one
        draft on a short greedy prompt. Collapse to 1-token-per-forward
        would keep us bandwidth-capped at ~20 tok/s."""
        from sglang.srt.hardware_backend.mlx.native_engine import NativeQwen38Engine

        eng = NativeQwen38Engine.load(str(_MODEL))
        eng.load_mtp(str(_MTP))
        self.assertTrue(eng.has_mtp())
        tok = eng.prefill([1, 2, 3, 4], schedule_decode=False)
        seen = [tok]
        for _ in range(6):
            tok = eng.decode(tok)
            seen.append(tok)
        self.assertEqual(len(seen), 7)
        self.assertTrue(all(isinstance(t, int) for t in seen))
        eng.close()


if __name__ == "__main__":
    unittest.main()
