import functools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import flashinfer_autotune
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    _promote_flashinfer_file_cache_hits,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_runner(*, is_draft_worker: bool):
    spec_algorithm = Mock()
    spec_algorithm.is_speculative.return_value = True
    model_runner = SimpleNamespace(
        server_args=SimpleNamespace(max_prefill_tokens=16),
        is_generation=True,
        is_draft_worker=is_draft_worker,
        spec_algorithm=spec_algorithm,
        model_config=SimpleNamespace(is_multimodal=False),
        attn_backend=SimpleNamespace(extend_dummy_seqs_capped_by_req_pool=False),
        canary_manager=None,
    )
    runner = Mock()
    runner.model_runner = model_runner
    runner._alloc_dummy_decode_buffers.return_value = object()
    return runner


class TestFlashInferAutotuneExtend(CustomTestCase):
    def test_speculative_target_runs_plain_extend_autotune(self):
        runner = _make_runner(is_draft_worker=False)

        with (
            envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.override(True),
            patch.object(
                flashinfer_autotune,
                "run_flashinfer_autotune_forward",
            ) as run_autotune,
        ):
            flashinfer_autotune.maybe_flashinfer_autotune_extend(
                runner, decode_num_tokens=3
            )

        runner._alloc_dummy_decode_buffers.assert_called_once_with(
            16,
            num_tokens_per_req=1,
            allocate_logits_buffer=False,
        )
        forward_fn = run_autotune.call_args.args[1]
        self.assertIsInstance(forward_fn, functools.partial)
        self.assertEqual(forward_fn.keywords["batch_size"], 16)
        self.assertEqual(
            forward_fn.keywords["forward_mode_override"],
            ForwardMode.EXTEND,
        )
        self.assertIsNone(forward_fn.keywords["extend_num_tokens_per_req"])
        self.assertTrue(forward_fn.keywords["allow_speculative_target_extend"])
        run_autotune.assert_called_once_with(
            runner.model_runner,
            forward_fn,
            run_lm_head=False,
            promote_file_cache_hits=True,
        )

    def test_extend_autotune_defaults_off(self):
        runner = _make_runner(is_draft_worker=False)

        with (
            envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.override(False),
            patch.object(
                flashinfer_autotune,
                "run_flashinfer_autotune_forward",
            ) as run_autotune,
        ):
            flashinfer_autotune.maybe_flashinfer_autotune_extend(
                runner, decode_num_tokens=3
            )

        runner._alloc_dummy_decode_buffers.assert_not_called()
        run_autotune.assert_not_called()

    def test_file_cache_hits_are_promoted_and_method_is_restored(self):
        class CacheKey:
            file_key = "file-key"

        class Runner:
            @staticmethod
            def get_cache_key_extras(_inputs):
                return ()

        class Tuner:
            def __init__(self):
                import threading

                self._lock = threading.RLock()
                self._file_configs = {"file-key": ("Runner", 7)}
                self.profiling_cache = {}

            @staticmethod
            def _get_cache_key(*_args):
                return CacheKey()

            def search_cache(self, *_args, **_kwargs):
                return True, 0, 7, None

        tuner = Tuner()
        original_method = tuner.search_cache.__func__
        runner = Runner()

        with (
            patch(
                "flashinfer.autotuner.AutoTuner.get",
                return_value=tuner,
            ),
            _promote_flashinfer_file_cache_hits(),
        ):
            result = tuner.search_cache(
                "fp4_gemm",
                [runner],
                ((4096, 2560),),
                object(),
                inputs=[object()],
            )

        self.assertEqual(result, (True, 0, 7, None))
        self.assertEqual(list(tuner.profiling_cache.values()), [(7, None)])
        self.assertIs(tuner.search_cache.__func__, original_method)

        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with (
                patch(
                    "flashinfer.autotuner.AutoTuner.get",
                    return_value=tuner,
                ),
                _promote_flashinfer_file_cache_hits(),
            ):
                raise RuntimeError("synthetic failure")

        self.assertIs(tuner.search_cache.__func__, original_method)

    def test_extend_autotune_allocation_oom_falls_back(self):
        runner = _make_runner(is_draft_worker=False)
        runner._alloc_dummy_decode_buffers.side_effect = torch.OutOfMemoryError(
            "synthetic allocation failure"
        )

        with (
            envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.override(True),
            patch.object(
                flashinfer_autotune,
                "run_flashinfer_autotune_forward",
            ) as run_autotune,
            patch.object(torch.cuda, "empty_cache") as empty_cache,
        ):
            flashinfer_autotune.maybe_flashinfer_autotune_extend(
                runner, decode_num_tokens=3
            )

        run_autotune.assert_not_called()
        empty_cache.assert_called_once_with()

    def test_speculative_draft_skips_extend_autotune(self):
        runner = _make_runner(is_draft_worker=True)

        with (
            envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.override(True),
            patch.object(
                flashinfer_autotune,
                "run_flashinfer_autotune_forward",
            ) as run_autotune,
        ):
            flashinfer_autotune.maybe_flashinfer_autotune_extend(
                runner, decode_num_tokens=3
            )

        runner._alloc_dummy_decode_buffers.assert_not_called()
        run_autotune.assert_not_called()


if __name__ == "__main__":
    unittest.main()
