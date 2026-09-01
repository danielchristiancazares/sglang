import os
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestCheckTreeCacheGate(CustomTestCase):
    @contextmanager
    def _without_explicit_sanity_check_setting(self):
        with patch.dict(os.environ, {}, clear=False):
            envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.clear()
            yield

    def _make_checker(self, *, hybrid_swa=True):
        tree_cache = MagicMock()
        tree_cache.is_tree_cache.return_value = True
        tree_cache.supports_swa.return_value = True
        tree_cache.supports_mamba.return_value = True
        return SchedulerInvariantChecker(
            is_hybrid_swa=hybrid_swa,
            is_hybrid_ssm=not hybrid_swa,
            disaggregation_mode=DisaggregationMode.NULL,
            page_size=1,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=1024,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=MagicMock(),
            req_to_token_pool=MagicMock(),
            pool_stats_observer=MagicMock(),
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )

    def test_disabled_by_default(self):
        with (
            envs.SGLANG_IS_IN_CI.override(False),
            self._without_explicit_sanity_check_setting(),
        ):
            checker = self._make_checker()

            checker._check_tree_cache()

            checker.tree_cache.is_tree_cache.assert_not_called()
            checker.tree_cache.sanity_check.assert_not_called()

    def test_enabled_by_default_in_ci(self):
        with (
            envs.SGLANG_IS_IN_CI.override(True),
            self._without_explicit_sanity_check_setting(),
        ):
            checker = self._make_checker()

            checker._check_tree_cache()

            checker.tree_cache.sanity_check.assert_called_once_with()

    def test_explicitly_disabled_in_ci(self):
        with envs.SGLANG_IS_IN_CI.override(True):
            checker = self._make_checker()

            with envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(False):
                checker._check_tree_cache()

            checker.tree_cache.sanity_check.assert_not_called()

    def test_runs_when_enabled_for_hybrid_caches(self):
        with envs.SGLANG_IS_IN_CI.override(False):
            for hybrid_swa in (True, False):
                with self.subTest(hybrid_swa=hybrid_swa):
                    checker = self._make_checker(hybrid_swa=hybrid_swa)

                    with envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(True):
                        checker._check_tree_cache()

                    checker.tree_cache.sanity_check.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
