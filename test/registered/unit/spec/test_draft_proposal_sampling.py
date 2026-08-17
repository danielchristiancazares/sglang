import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.speculative.eagle_utils import (
    build_swor_topology,
    default_swor_topology,
    organize_swor_draft_results,
    parse_swor_topology,
    select_swor_topology_step,
)

from sglang.srt.speculative.spec_utils import (
    discount_tree_node_scores_,
    fast_sample,
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
    def test_default_swor_topology_and_q_rows(self):
        topology = default_swor_topology("cpu")
        self.assertEqual(topology.parent_by_node, (-1, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 9))
        self.assertEqual(tuple(map(len, topology.nodes_by_depth[1:])), (4, 4, 2, 1))
        self.assertEqual(topology.selected_indices.tolist(), [[0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 36]])

        token_blocks = [
            torch.tensor([[10, 11, 12, 13]]),
            torch.tensor([[20, 21, 22, 23]]),
            torch.tensor([[30, 31]]),
            torch.tensor([[40]]),
        ]
        q_blocks = [
            torch.tensor([[[0.0]]]),
            torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1),
            torch.arange(5, 9, dtype=torch.float32).view(1, 4, 1),
            torch.arange(9, 13, dtype=torch.float32).view(1, 4, 1),
        ]
        _, _, tokens, q = organize_swor_draft_results(topology, token_blocks, q_blocks)
        self.assertEqual(tokens.tolist(), [[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 40]])
        self.assertEqual(q[..., 0].tolist(), [[0, 1, 2, 3, 4, 5, 6, 0, 0, 9, 0, 0]])

    def test_swor_organizer_keeps_low_q_early_sibling_prefix(self):
        topology = default_swor_topology("cpu")
        # The old global score pruning preferred token 91 and could omit token
        # 90. The fixed root prefix is drawn-order [90, 91, 92, 93].
        sampled = torch.tensor([[90, 91, 92, 93]])
        _, _, visible = select_swor_topology_step(topology, 0, sampled, None)
        self.assertEqual(visible.tolist(), [[90, 91, 92, 93]])

    def test_custom_swor_topology_json(self):
        topology = parse_swor_topology(
            "[-1,0,0,1,2,3,4]", draft_width=2, device="cpu"
        )
        self.assertEqual(topology.parent_by_node, (-1, 0, 0, 1, 2, 3, 4))
        self.assertEqual(tuple(map(len, topology.nodes_by_depth[1:])), (2, 2, 2))

    def test_custom_swor_topology_json_rejects_non_integer_parent(self):
        with self.assertRaisesRegex(ValueError, "array of integer parent IDs"):
            parse_swor_topology("[-1, true]", draft_width=2, device="cpu")

    def test_every_internal_node_requires_q_source(self):
        topology = default_swor_topology("cpu")
        internal = set(topology.parent_by_node[1:])
        for node in internal:
            self.assertIsNotNone(topology.q_sources[node], f"internal node {node}")
        self.assertIsNone(topology.q_sources[7])

    def test_topology_rejects_missing_fixed_width_frontier(self):
        # Eight internal depth-two nodes cannot fit a four-row draft forward.
        with self.assertRaisesRegex(ValueError, "depth 2 has 8 internal nodes"):
            build_swor_topology(
                (-1, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                 5, 6, 7, 8, 9, 10, 11, 12),
                4,
                "cpu",
            )

    def test_swor_support_exhaustion_orders_zero_mass_uniformly(self):
        torch.manual_seed(1234)
        for support_size in (0, 1, 3, 4):
            probs = torch.zeros((256, 8), dtype=torch.float32)
            if support_size:
                probs[:, :support_size] = 1.0 / support_size
            _, indices = fast_sample(probs, num_samples=4)
            self.assertTrue(torch.all(indices[:, :support_size] < support_size))
            ordered = indices.sort(dim=1).values
            self.assertTrue(torch.all(ordered[:, 1:] != ordered[:, :-1]))
            if support_size < 4:
                zero_first = indices[:, support_size]
                self.assertGreater(torch.unique(zero_first).numel(), 1)

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
