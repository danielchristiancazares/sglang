from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_worker_common import (
    duplicate_prefix_tail_to_draft_branches,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_utils import sample_draft_proposal


class TestDeviceResidentEagleCycle(unittest.TestCase):
    @staticmethod
    def _sampling_info(
        *,
        temperatures,
        top_ps,
        additive=None,
        logit_bias=None,
    ):
        rows = temperatures.shape[0]
        return SamplingBatchInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=torch.full((rows,), -1, dtype=torch.int32),
            min_ps=torch.zeros((rows,), dtype=torch.float32),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=True,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=additive.shape[1] if additive is not None else 4,
            acc_additive_penalties=additive,
            logit_bias=logit_bias,
            device="cpu",
        )

    def test_graph_sampling_buffers_refresh_penalties_and_bias(self):
        runner = object.__new__(EAGLEDraftCudaGraphRunner)
        runner.temperatures = torch.zeros((2, 1), dtype=torch.float32)
        runner.draft_top_ps = torch.zeros((2,), dtype=torch.float32)
        runner.draft_additive_penalties = torch.zeros((2, 4), dtype=torch.float32)
        runner.model_runner = SimpleNamespace(
            model_config=SimpleNamespace(vocab_size=4)
        )
        source = self._sampling_info(
            temperatures=torch.tensor([[0.75], [1.25]]),
            top_ps=torch.tensor([0.9, 0.8]),
            additive=torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            ),
            logit_bias=torch.tensor(
                [[0.5, 0.0, -0.5, 1.0], [1.0, -1.0, 0.0, 0.5]]
            ),
        )

        runner.copy_sampling_info_to_graph(source, 2)
        stable = runner.sampling_info_for_graph(2)

        torch.testing.assert_close(stable.temperatures, source.temperatures)
        torch.testing.assert_close(stable.top_ps, source.top_ps)
        torch.testing.assert_close(
            stable.acc_additive_penalties,
            source.acc_additive_penalties + source.logit_bias,
        )
        self.assertIsNone(stable.logit_bias)

    def test_exact_proposal_consumes_caller_races(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
        sampling_info = self._sampling_info(
            temperatures=torch.ones((1, 1)),
            top_ps=torch.ones((1,)),
        )
        races = torch.tensor([[10.0, 0.1, 10.0, 10.0]], dtype=torch.float32)

        q, q_x, token = sample_draft_proposal(
            logits,
            sampling_info,
            draft_sampling_top_k=4,
            races=races,
        )

        self.assertEqual(int(token.item()), 1)
        torch.testing.assert_close(q_x, q.gather(1, token))

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
