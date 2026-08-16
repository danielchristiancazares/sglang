import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.speculative.eagle_utils import organize_tree_swor_probs

from sglang.srt.speculative.spec_utils import (
    discount_tree_node_scores_,
    renorm_draft_probs,
    sample_draft_proposal,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _argmax_sample(probs: torch.Tensor, num_samples: int = 1):
    assert num_samples == 1
    index = probs.argmax(dim=-1, keepdim=True)
    return probs.gather(1, index), index


class TestDraftProposalSampling(CustomTestCase):
    def test_swor_probs_follow_final_pruned_tree_node_order(self):
        # Source rows are root (-1), first-level candidates 0..3, and two
        # continuation frontiers. Final leaves 40/41 have no q row and safely
        # map to root because verification never reads q from a leaf.
        source_ids = torch.tensor(
            [[-1, 0, 1, 2, 3, 4, 8, 9, 10, 20, 24, 21, 28]],
            dtype=torch.long,
        )
        source_probs = torch.arange(13, dtype=torch.float32).view(1, 13, 1)
        selected = torch.tensor([[0, 4, 8, 20, 24, 40, 41]], dtype=torch.long)

        result = organize_tree_swor_probs(source_probs, source_ids, selected)

        self.assertEqual(result[:, :, 0].tolist(), [[0, 1, 5, 6, 9, 10, 0, 0]])

    def test_tree_depth_discount_only_changes_global_allocation_scores(self):
        scores = torch.tensor([[0.8, 0.4, 0.2]], dtype=torch.float32)

        self.assertIs(discount_tree_node_scores_(scores, 0, 0.5), scores)
        self.assertIs(discount_tree_node_scores_(scores, 3, 1.0), scores)
        discounted = scores.clone()
        self.assertIs(discount_tree_node_scores_(discounted, 2, 0.5), discounted)
        torch.testing.assert_close(
            discounted, scores * 0.25
        )
        torch.testing.assert_close(scores, torch.tensor([[0.8, 0.4, 0.2]]))

    @patch(
        "sglang.srt.speculative.spec_utils.fast_sample",
        side_effect=_argmax_sample,
    )
    def test_default_preserves_temperature_only_distribution(self, _sample):
        logits = torch.tensor([[2.0, 1.0, -1.0]], dtype=torch.float32)
        temperatures = torch.tensor([[2.0]], dtype=torch.float32)

        q, q_x, token = sample_draft_proposal(logits, temperatures)

        expected = torch.softmax(logits / temperatures, dim=-1)
        torch.testing.assert_close(q, expected)
        torch.testing.assert_close(q_x, q.gather(1, token))

    @patch(
        "sglang.srt.speculative.spec_utils.fast_sample",
        side_effect=_argmax_sample,
    )
    def test_sparse_q_applies_additive_bias_topk_and_topp_exactly(self, _sample):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((1, 1), dtype=torch.float32),
            top_ps=torch.tensor([0.8], dtype=torch.float32),
            acc_additive_penalties=torch.tensor(
                [[-10.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ),
            logit_bias=torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 3.0]], dtype=torch.float32
            ),
        )

        q, q_x, token = sample_draft_proposal(
            logits, sampling_info, draft_sampling_top_k=3
        )

        self.assertEqual(torch.count_nonzero(q).item(), 2)
        torch.testing.assert_close(q.sum(dim=-1), torch.ones(1))
        torch.testing.assert_close(q[0, [1, 4]], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(q[0, [0, 2, 3]], torch.zeros(3))
        torch.testing.assert_close(q_x, q.gather(1, token))
        self.assertIn(token.item(), (1, 4))

    def test_target_only_tree_can_use_same_aligned_distribution(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((1, 1), dtype=torch.float32),
            top_ps=torch.tensor([0.8], dtype=torch.float32),
            acc_additive_penalties=torch.tensor(
                [[-10.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ),
            logit_bias=torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 3.0]], dtype=torch.float32
            ),
        )

        scores = renorm_draft_probs(
            logits,
            sampling_info,
            use_rejection_sampling=False,
            draft_sampling_top_k=3,
        )

        self.assertEqual(torch.count_nonzero(scores).item(), 2)
        torch.testing.assert_close(scores.sum(dim=-1), torch.ones(1))
        torch.testing.assert_close(scores[0, [1, 4]], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(scores[0, [0, 2, 3]], torch.zeros(3))

    def test_sparse_q_rejects_out_of_range_width(self):
        logits = torch.zeros((1, 4), dtype=torch.float32)
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((1, 1), dtype=torch.float32),
            top_ps=torch.ones(1, dtype=torch.float32),
            acc_additive_penalties=None,
            logit_bias=None,
        )

        with self.assertRaisesRegex(ValueError, "draft_sampling_top_k"):
            sample_draft_proposal(logits, sampling_info, draft_sampling_top_k=5)


if __name__ == "__main__":
    unittest.main()
