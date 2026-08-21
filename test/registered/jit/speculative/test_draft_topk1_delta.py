import unittest

import torch

from sglang.kernels.ops.speculative.draft_topk1_delta import draft_topk1_delta
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large"
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDraftTopK1Delta(CustomTestCase):
    def test_matches_stable_argmax_and_one_hot_q(self):
        for rows, vocab_size in ((1, 17), (3, 8193), (2, 248320)):
            with self.subTest(rows=rows, vocab_size=vocab_size):
                logits = torch.randn(
                    (rows, vocab_size), dtype=torch.float32, device="cuda"
                )
                logits[:, 1] = 10.0
                logits[:, 7] = 10.0
                if vocab_size > 9000:
                    logits[:, 9000] = 11.0

                q, topk_p, topk_index = draft_topk1_delta(logits)
                torch.cuda.synchronize()

                expected = torch.argmax(logits, dim=-1, keepdim=True)
                torch.testing.assert_close(topk_index, expected, rtol=0, atol=0)
                torch.testing.assert_close(topk_p, torch.ones_like(topk_p))
                torch.testing.assert_close(q.sum(dim=-1), torch.ones(rows, device="cuda"))
                torch.testing.assert_close(
                    q.gather(1, topk_index), torch.ones_like(topk_p)
                )
                self.assertEqual(torch.count_nonzero(q).item(), rows)

    def test_applies_additive_before_argmax(self):
        logits = torch.tensor([[5.0, 4.0, 3.0]], device="cuda")
        additive = torch.tensor([[-10.0, 0.0, 2.0]], device="cuda")

        q, topk_p, topk_index = draft_topk1_delta(logits, additive)
        torch.cuda.synchronize()

        self.assertEqual(topk_index.item(), 2)
        self.assertEqual(topk_p.item(), 1.0)
        self.assertEqual(q.tolist(), [[0.0, 0.0, 1.0]])

    def test_applies_additive_then_bias_without_reassociation(self):
        logits = torch.tensor([[1.0e8, 0.5]], device="cuda")
        additive = torch.tensor([[-1.0e8, 0.0]], device="cuda")
        logit_bias = torch.tensor([[1.0, 0.0]], device="cuda")

        q, _, topk_index = draft_topk1_delta(logits, additive, logit_bias)
        torch.cuda.synchronize()

        self.assertEqual(topk_index.item(), 0)
        self.assertEqual(q.tolist(), [[1.0, 0.0]])

    def test_nan_never_wins(self):
        logits = torch.tensor(
            [[float("nan"), torch.finfo(torch.float32).min, -1.0]],
            device="cuda",
        )

        _, _, topk_index = draft_topk1_delta(logits)
        torch.cuda.synchronize()

        self.assertEqual(topk_index.item(), 2)

    def test_all_negative_infinity_selects_stable_first_token(self):
        logits = torch.full((1, 17), -float("inf"), device="cuda")

        q, _, topk_index = draft_topk1_delta(logits)
        torch.cuda.synchronize()

        self.assertEqual(topk_index.item(), 0)
        self.assertEqual(q[0, 0].item(), 1.0)

    def test_all_nan_selects_stable_first_token_without_poisoning_context(self):
        logits = torch.full((1, 17), float("nan"), device="cuda")

        q, _, topk_index = draft_topk1_delta(logits)
        torch.cuda.synchronize()

        self.assertEqual(topk_index.item(), 0)
        self.assertEqual(q[0, 0].item(), 1.0)
        self.assertEqual(torch.count_nonzero(q).item(), 1)

    def test_row_count_beyond_cuda_grid_y_limit(self):
        logits = torch.zeros((65536, 1), device="cuda")

        q, _, topk_index = draft_topk1_delta(logits)
        torch.cuda.synchronize()

        self.assertEqual(q.shape, logits.shape)
        self.assertEqual(torch.count_nonzero(q).item(), logits.shape[0])
        self.assertEqual(torch.count_nonzero(topk_index).item(), 0)

    def test_rejects_invalid_public_inputs(self):
        with self.assertRaisesRegex(ValueError, "CUDA float32"):
            draft_topk1_delta(
                torch.zeros((1, 3), dtype=torch.bfloat16, device="cuda")
            )
        with self.assertRaisesRegex(ValueError, "contiguous rank two"):
            draft_topk1_delta(torch.zeros((3, 2), device="cuda").T)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            draft_topk1_delta(torch.empty((0, 3), device="cuda"))
        with self.assertRaisesRegex(ValueError, "additive"):
            draft_topk1_delta(
                torch.zeros((1, 3), device="cuda"),
                torch.zeros((1, 2), device="cuda"),
            )

    def test_mutable_cuda_graph_replay_overwrites_q(self):
        logits = torch.tensor([[4.0, 3.0, 2.0]], device="cuda")
        additive = torch.zeros_like(logits)
        draft_topk1_delta(logits, additive)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, topk_p, topk_index = draft_topk1_delta(logits, additive)

        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(topk_index.item(), 0)
        self.assertEqual(q.tolist(), [[1.0, 0.0, 0.0]])

        logits.copy_(torch.tensor([[1.0, 2.0, 5.0]], device="cuda"))
        additive.copy_(torch.tensor([[5.0, 0.0, -10.0]], device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(topk_index.item(), 0)
        self.assertEqual(topk_p.item(), 1.0)
        self.assertEqual(q.tolist(), [[1.0, 0.0, 0.0]])

        additive.zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(topk_index.item(), 2)
        self.assertEqual(q.tolist(), [[0.0, 0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
