from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_worker_common import (
    duplicate_prefix_tail_to_draft_branches,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker


class TestDeviceResidentEagleCycle(unittest.TestCase):
    def test_page_tail_copy_uses_fixed_capture_safe_shape(self):
        class FakePool:
            def move_kv_cache(self, target, source):
                self.target = target.clone()
                self.source = source.clone()

        pool = FakePool()
        rows = torch.arange(40, dtype=torch.int64).reshape(1, -1)
        duplicate_prefix_tail_to_draft_branches(
            pool,
            rows,
            prefix_base=torch.tensor([4]),
            last_page=torch.tensor([2]),
            num_new_pages=torch.tensor([2]),
            topk=4,
            page_size=4,
        )

        torch.testing.assert_close(
            pool.target,
            torch.tensor([12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]),
        )
        torch.testing.assert_close(
            pool.source,
            torch.tensor([4, 5, 14, 15, 4, 5, 22, 23, 4, 5, 30, 31]),
        )

    def test_matching_batch_consumes_precomputed_verify_input(self):
        worker = object.__new__(EagleDraftWorker)
        request = object()
        verify_input = object()
        worker.device_resident_cycle = True
        worker._precomputed_verify_input = verify_input
        worker._precomputed_verify_req_ids = (id(request),)

        actual = EagleDraftWorker.draft(
            worker, SimpleNamespace(reqs=[request])
        )

        self.assertIs(actual, verify_input)
        self.assertIsNone(worker._precomputed_verify_input)
        self.assertIsNone(worker._precomputed_verify_req_ids)

    def test_precompute_uses_device_new_lengths_on_a_structural_clone(self):
        worker = object.__new__(EagleDraftWorker)
        verify_input = object()
        worker.draft = Mock(return_value=verify_input)
        request = object()
        original_spec_info = object()
        next_draft_input = object()
        old_lengths = torch.tensor([101], dtype=torch.int64)
        new_lengths = torch.tensor([104], dtype=torch.int64)
        batch = SimpleNamespace(
            reqs=[request],
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            seq_lens=old_lengths,
            seq_lens_cpu=torch.tensor([101], dtype=torch.int64),
            seq_lens_sum=101,
            spec_info=original_spec_info,
        )
        batch_result = SimpleNamespace(new_seq_lens=new_lengths)

        EagleDraftWorker._precompute_device_cycle_verify_input(
            worker, batch, batch_result, next_draft_input
        )

        cycle_batch = worker.draft.call_args.args[0]
        self.assertIsNot(cycle_batch, batch)
        self.assertEqual(cycle_batch.forward_mode, ForwardMode.DECODE)
        self.assertIs(cycle_batch.seq_lens, new_lengths)
        self.assertIsNone(cycle_batch.seq_lens_cpu)
        self.assertIsNone(cycle_batch.seq_lens_sum)
        self.assertIs(cycle_batch.spec_info, next_draft_input)
        self.assertIs(batch.seq_lens, old_lengths)
        self.assertIs(batch.spec_info, original_spec_info)
        self.assertIs(worker._precomputed_verify_input, verify_input)
        self.assertEqual(worker._precomputed_verify_req_ids, (id(request),))


if __name__ == "__main__":
    unittest.main()
