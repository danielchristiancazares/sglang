from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from flashinfer.sampling import top_k_renorm_prob, top_p_renorm_prob

from sglang.kernels.ops.sampling.sparse_top_p_renorm import (
    sparse_top_p_renorm,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_composite import CudaGraphChildSequence
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.eagle_utils import _renorm_target_probs_top_p
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.spec_utils import (
    sample_draft_proposal,
    use_sparse_top_p_renorm,
)
from sglang.test.test_utils import CustomTestCase


def _argmax_sample(probs: torch.Tensor, num_samples: int = 1):
    assert num_samples == 1
    index = probs.argmax(dim=-1, keepdim=True)
    return probs.gather(1, index), index


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDraftProposalFlashInfer(CustomTestCase):
    def test_explicit_offset_refreshes_raw_composite_sampling(self):
        batch_size = 32
        vocab_size = 8192
        logits = torch.zeros(
            (batch_size, vocab_size), dtype=torch.float32, device="cuda"
        )
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((batch_size, 1), device="cuda"),
            top_ps=torch.full((batch_size,), 0.95, device="cuda"),
            acc_additive_penalties=None,
            logit_bias=None,
            need_top_p_sampling=True,
        )
        seed = torch.tensor([1701], dtype=torch.int64, device="cuda")
        offset = torch.tensor([0], dtype=torch.int64, device="cuda")

        # Warm the lazy FlashInfer sampling image before graph capture.
        sample_draft_proposal(
            logits,
            sampling_info,
            draft_sampling_top_k=20,
            sampling_seed=seed,
            sampling_offset=offset,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            q, q_x, token = sample_draft_proposal(
                logits,
                sampling_info,
                draft_sampling_top_k=20,
                sampling_seed=seed,
                sampling_offset=offset,
            )

        with CudaGraphChildSequence((graph,)) as sequence:
            offset.zero_()
            sequence.replay()
            torch.cuda.synchronize()
            first = token.clone()
            torch.testing.assert_close(q_x, q.gather(1, token), rtol=0, atol=0)

            offset.zero_()
            sequence.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(token, first, rtol=0, atol=0)

            offset.fill_(batch_size * vocab_size)
            sequence.replay()
            torch.cuda.synchronize()
            self.assertFalse(torch.equal(token, first))
            torch.testing.assert_close(q_x, q.gather(1, token), rtol=0, atol=0)

    @patch(
        "sglang.srt.speculative.spec_utils.fast_sample",
        side_effect=_argmax_sample,
    )
    def test_penalty_temperature_topk_topp_q_matches_sparse_reference(self, _sample):
        generator = torch.Generator(device="cuda").manual_seed(1701)
        logits = torch.randn(
            (2, 8192), dtype=torch.float32, device="cuda", generator=generator
        )
        penalties = torch.zeros_like(logits)
        penalties[:, :32] = -4.0
        bias = torch.zeros_like(logits)
        bias[:, 4096:4104] = 2.0
        sampling_info = SimpleNamespace(
            temperatures=torch.tensor([[0.8], [1.2]], device="cuda"),
            top_ps=torch.tensor([0.90, 0.95], device="cuda"),
            acc_additive_penalties=penalties,
            logit_bias=bias,
            need_top_p_sampling=True,
        )

        q, q_x, token = sample_draft_proposal(
            logits, sampling_info, draft_sampling_top_k=20
        )

        adjusted = (logits + penalties + bias) / sampling_info.temperatures
        values, indices = torch.topk(adjusted, 20, dim=-1, sorted=True)
        sparse = torch.softmax(values, dim=-1)
        ascending = torch.flip(sparse, dims=(-1,))
        cutoff = torch.sum(
            torch.cumsum(ascending, dim=-1)
            < (1.0 - sampling_info.top_ps[:, None]),
            dim=-1,
            keepdim=True,
        ).clamp_(max=19)
        pivots = ascending.gather(1, cutoff)
        sparse = torch.where(sparse >= pivots, sparse, torch.zeros_like(sparse))
        sparse = sparse / sparse.sum(dim=-1, keepdim=True)
        expected = torch.zeros_like(q).scatter_(1, indices, sparse)

        torch.testing.assert_close(q, expected, rtol=3e-5, atol=2e-7)
        torch.testing.assert_close(q.sum(dim=-1), torch.ones(2, device="cuda"))
        torch.testing.assert_close(q_x, q.gather(1, token), rtol=0, atol=0)

    def _check_aligned_q_is_captured_inside_single_graph(self):
        vocab_size = 8192
        generator = torch.Generator(device="cuda").manual_seed(2701)
        logits = torch.randn(
            (1, vocab_size), dtype=torch.float32, device="cuda", generator=generator
        )
        additive = torch.zeros_like(logits)
        additive[:, :64] = -3.0
        temperatures = torch.tensor([[1.1]], dtype=torch.float32, device="cuda")
        top_ps = torch.tensor([0.92], dtype=torch.float32, device="cuda")
        draft_probs = torch.empty(
            (1, 2, vocab_size), dtype=torch.float32, device="cuda"
        )

        runner = object.__new__(MultiLayerEagleDraftExtendCudaGraphRunner)
        runner.step = 0
        runner.prune_draft_extend_logits = True
        runner.eagle_worker = SimpleNamespace(draft_sampling_top_k=20)
        runner.buffers = SimpleNamespace(
            temperatures=temperatures,
            draft_probs=draft_probs,
            draft_top_ps=top_ps,
            draft_additive_penalties=additive,
        )
        ret = SimpleNamespace(next_token_logits=logits)

        # Warm every lazy FlashInfer image/cache before entering capture.
        runner._sample_draft_proposal(ret, bs=1)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner._sample_draft_proposal(ret, bs=1)
        graph.replay()
        torch.cuda.synchronize()

        def expected_q():
            probs = torch.softmax((logits + additive) / temperatures, dim=-1)
            probs = top_k_renorm_prob(probs, 20)
            return top_p_renorm_prob(probs, top_ps)

        expected = expected_q()
        torch.testing.assert_close(draft_probs[:, 0], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            ret.topk_p, draft_probs[:, 0].gather(1, ret.topk_index), rtol=0, atol=0
        )

        # Replay must consume current stable-buffer values, not capture-time q.
        logits.mul_(0.7)
        additive.zero_()
        additive[:, 400:464] = 2.5
        top_ps.fill_(0.85)
        graph.replay()
        torch.cuda.synchronize()
        expected = expected_q()
        torch.testing.assert_close(draft_probs[:, 0], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            ret.topk_p, draft_probs[:, 0].gather(1, ret.topk_index), rtol=0, atol=0
        )

    def test_aligned_q_is_captured_inside_single_graph(self):
        self._check_aligned_q_is_captured_inside_single_graph()

    def test_sparse_top_p_aligned_q_is_captured_inside_single_graph(self):
        with (
            patch(
                "sglang.srt.speculative.spec_utils.sys",
                SimpleNamespace(platform="win32"),
            ),
            patch(
                "sglang.srt.speculative.spec_utils.use_sparse_top_p_renorm",
                return_value=True,
            ),
            patch(
                "sglang.kernels.ops.sampling.sparse_top_p_renorm.sparse_top_p_renorm",
                wraps=sparse_top_p_renorm,
            ) as sparse_mock,
        ):
            self._check_aligned_q_is_captured_inside_single_graph()
        sparse_mock.assert_called()

    def test_sparse_top_p_cached_windows_gate(self):
        use_sparse_top_p_renorm.cache_clear()
        try:
            with (
                patch(
                    "sglang.srt.speculative.spec_utils._is_cuda",
                    True,
                ),
                patch(
                    "sglang.srt.speculative.spec_utils.sys",
                    SimpleNamespace(platform="win32"),
                ),
                envs.SGLANG_OPT_SPARSE_TOP_P_RENORM.override(True),
            ):
                self.assertTrue(use_sparse_top_p_renorm())
        finally:
            use_sparse_top_p_renorm.cache_clear()

    def test_draft_top_p_one_routes_to_both_proposal_owners(self):
        runner = EAGLEDraftCudaGraphRunner.__new__(EAGLEDraftCudaGraphRunner)
        runner.model_runner = SimpleNamespace(
            model_config=SimpleNamespace(vocab_size=16)
        )
        runner.max_bs = 1
        runner.draft_sampling_top_k = 20
        runner.draft_sampling_top_p = 1.0
        runner.temperatures = torch.ones((1, 1), device="cuda")
        runner.draft_top_ps = torch.zeros(1, device="cuda")
        runner.draft_additive_penalties = torch.zeros((1, 16), device="cuda")
        live_sampling_info = SimpleNamespace(
            temperatures=torch.full((1, 1), 0.8, device="cuda"),
            top_ps=torch.full((1,), 0.95, device="cuda"),
            acc_additive_penalties=None,
            logit_bias=None,
        )

        runner.copy_sampling_info_to_graph(live_sampling_info, raw_bs=1)
        graph_sampling_info = runner.sampling_info_for_graph(num_seqs=1)

        self.assertFalse(graph_sampling_info.need_top_p_sampling)
        torch.testing.assert_close(
            graph_sampling_info.top_ps,
            torch.ones(1, device="cuda"),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            live_sampling_info.top_ps,
            torch.full((1,), 0.95, device="cuda"),
            rtol=0,
            atol=0,
        )

        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.draft_sampling_top_p = 1.0
        post_extend_info = SimpleNamespace(
            need_top_p_sampling=True,
            top_ps=torch.full((1,), 0.95, device="cuda"),
            temperatures=torch.ones((1, 1), device="cuda"),
        )
        proposal_info = worker._proposal_sampling_info(post_extend_info)
        self.assertFalse(proposal_info.need_top_p_sampling)
        self.assertTrue(post_extend_info.need_top_p_sampling)

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.sys",
            SimpleNamespace(platform="linux"),
        ):
            generic_info = worker._proposal_sampling_info(post_extend_info)
        torch.testing.assert_close(
            generic_info.top_ps,
            torch.ones(1, device="cuda"),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            post_extend_info.top_ps,
            torch.full((1,), 0.95, device="cuda"),
            rtol=0,
            atol=0,
        )

        cpu_info = SimpleNamespace(
            need_top_p_sampling=True,
            top_ps=torch.full((1,), 0.95),
            temperatures=torch.ones((1, 1)),
        )
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.sys",
            SimpleNamespace(platform="win32"),
        ):
            cpu_proposal_info = worker._proposal_sampling_info(cpu_info)
        torch.testing.assert_close(
            cpu_proposal_info.top_ps,
            torch.ones(1),
            rtol=0,
            atol=0,
        )

        standalone = EagleDraftWorker.__new__(EagleDraftWorker)
        self.assertIs(
            standalone._proposal_sampling_info(post_extend_info),
            post_extend_info,
        )

    @patch(
        "sglang.srt.speculative.spec_utils.use_sparse_top_p_renorm",
        return_value=True,
    )
    def test_sparse_top_p_target_rows_expand_and_replay(self, _gate):
        generator = torch.Generator(device="cuda").manual_seed(3701)
        request_count = 2
        draft_token_num = 3
        vocab_size = 8192
        logits = torch.randn(
            (request_count * draft_token_num, vocab_size),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        initial_probs = top_k_renorm_prob(torch.softmax(logits, dim=-1), 20)
        static_probs = initial_probs.clone()
        sampling_info = SimpleNamespace(
            top_ps=torch.tensor([0.85, 0.95], device="cuda"),
            max_top_k=20,
        )

        def candidate():
            return _renorm_target_probs_top_p(
                static_probs,
                sampling_info,
                draft_token_num,
                top_p_renorm_prob,
            )

        candidate()
        static_probs.copy_(initial_probs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = candidate()

        for seed, top_ps in ((3702, (0.8, 0.9)), (3703, (0.9, 0.98))):
            generator.manual_seed(seed)
            logits = torch.randn(
                static_probs.shape,
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            probs = top_k_renorm_prob(torch.softmax(logits, dim=-1), 20)
            sampling_info.top_ps.copy_(
                torch.tensor(top_ps, dtype=torch.float32, device="cuda")
            )
            static_probs.copy_(probs)
            graph.replay()
            expected = top_p_renorm_prob(
                probs,
                torch.repeat_interleave(
                    sampling_info.top_ps,
                    draft_token_num,
                ),
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
