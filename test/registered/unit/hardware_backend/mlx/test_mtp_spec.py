"""Unit tests for the MLX MTP speculative-decode engine internals."""

import unittest

from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (
    ContiguousAttentionKVCache,
)
from sglang.srt.hardware_backend.mlx.mtp_spec import AdaptivePolicy


class TestAdaptivePolicy(unittest.TestCase):
    def _policy(self) -> AdaptivePolicy:
        return AdaptivePolicy(window=4, off_thresh=2.9, ar_run=8, ar_run_cap=32)

    def test_stays_on_while_winning(self):
        policy = self._policy()
        for _ in range(16):
            policy.note_round(4)
        self.assertFalse(policy.in_ar_mode())

    def test_trips_below_threshold_and_drains(self):
        policy = self._policy()
        for _ in range(4):
            policy.note_round(2)
        self.assertTrue(policy.in_ar_mode())
        self.assertEqual(policy.ar_budget, 8)
        policy.note_ar_tokens(8)
        self.assertFalse(policy.in_ar_mode())

    def test_short_window_never_trips(self):
        policy = self._policy()
        for _ in range(3):
            policy.note_round(1)
        self.assertFalse(policy.in_ar_mode())

    def test_backoff_doubles_then_caps(self):
        policy = self._policy()
        budgets = []
        for _ in range(4):
            for _ in range(4):
                policy.note_round(1)
            budgets.append(policy.ar_budget)
            policy.note_ar_tokens(policy.ar_budget)
        self.assertEqual(budgets, [8, 16, 32, 32])

    def test_winning_window_resets_backoff(self):
        policy = self._policy()
        for _ in range(4):
            policy.note_round(1)
        policy.note_ar_tokens(policy.ar_budget)      # one trip at stretch 8
        for _ in range(4):
            policy.note_round(4)                     # clearly winning again
        for _ in range(4):
            policy.note_round(1)
        self.assertEqual(policy.ar_budget, 8)        # backoff was reset

    def test_trip_clears_window(self):
        policy = self._policy()
        for _ in range(4):
            policy.note_round(1)
        policy.note_ar_tokens(policy.ar_budget)
        policy.note_round(1)                         # fresh window: one sample
        self.assertFalse(policy.in_ar_mode())        # needs a full window again


class TestAttentionTrim(unittest.TestCase):
    def test_trim_moves_offset_back(self):
        cache = ContiguousAttentionKVCache(max_seq_len=64)
        cache.offset = 10
        cache.trim(3)
        self.assertEqual(cache.offset, 7)

    def test_trim_rejects_more_than_offset(self):
        cache = ContiguousAttentionKVCache(max_seq_len=64)
        cache.offset = 2
        with self.assertRaises(ValueError):
            cache.trim(3)
        with self.assertRaises(ValueError):
            cache.trim(-1)


if __name__ == "__main__":
    unittest.main()
