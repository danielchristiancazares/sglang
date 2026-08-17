from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.tree_sampling import (
    exact_tree_speculative_sampling,
    exact_tree_swor_sampling,
    swor_proposal_overlap_metrics,
)
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_swor_topology,
    build_tree_kernel_efficient,
    default_swor_topology,
    organize_swor_draft_results,
    select_swor_topology_step,
)
from sglang.srt.speculative.spec_utils import fast_sample, renorm_draft_probs
from sglang.test.test_utils import CustomTestCase


def _tree_metadata(batch_size: int, device: torch.device):
    #       0
    #    /  |  \
    #   1   2   3
    #  / \   \
    # 4   5   6
    candidates = torch.tensor(
        [[7, 1, 2, 3, 4, 5, 6]], dtype=torch.long, device=device
    ).repeat(batch_size, 1)
    retrieve_index = torch.arange(
        batch_size * 7, dtype=torch.long, device=device
    ).reshape(batch_size, 7)
    retrieve_next = torch.tensor(
        [[1, 4, 6, -1, -1, -1, -1]], dtype=torch.long, device=device
    ).repeat(batch_size, 1)
    retrieve_sibling = torch.tensor(
        [[-1, 2, 3, -1, 5, -1, -1]], dtype=torch.long, device=device
    ).repeat(batch_size, 1)
    return candidates, retrieve_index, retrieve_next, retrieve_sibling


def _target_probs(batch_size: int, device: torch.device):
    rows = torch.tensor(
        [
            [0.08, 0.20, 0.15, 0.10, 0.07, 0.09, 0.11, 0.20],
            [0.10, 0.05, 0.10, 0.05, 0.10, 0.20, 0.10, 0.30],
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.30, 0.10],
            [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
            [0.05, 0.10, 0.15, 0.10, 0.20, 0.10, 0.10, 0.20],
            [0.05, 0.05, 0.10, 0.10, 0.10, 0.15, 0.20, 0.25],
            [0.20, 0.10, 0.05, 0.05, 0.10, 0.20, 0.10, 0.20],
        ],
        dtype=torch.float32,
        device=device,
    )
    return rows.unsqueeze(0).repeat(batch_size, 1, 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestExactTreeSamplingJit(CustomTestCase):
    def test_sparse_proposal_overlap_grid(self):
        device = torch.device("cuda")
        target = torch.tensor([[[0.5, 0.3, 0.2, 0.0]]], device=device)
        draft = torch.tensor([[[0.4, 0.4, 0.2, 0.0]]], device=device)
        scales = torch.tensor([1.0], dtype=torch.float32, device=device)
        top_ks = torch.tensor([2, 3], dtype=torch.int32, device=device)

        metrics = swor_proposal_overlap_metrics(target, draft, scales, top_ks)

        torch.testing.assert_close(
            metrics[0, 0, 0, :, 0], torch.tensor([0.8, 0.9], device=device)
        )
        torch.testing.assert_close(
            metrics[0, 0, 0, :, 1], torch.zeros(2, device=device)
        )
        torch.testing.assert_close(
            metrics[0, 0, 0, :, 2], torch.full((2,), 3.0, device=device)
        )

    def _run(self, coins: torch.Tensor, final_coins: torch.Tensor):
        device = coins.device
        bs = coins.shape[0]
        candidates, retrieve_index, retrieve_next, retrieve_sibling = _tree_metadata(
            bs, device
        )
        probs = _target_probs(bs, device)
        predicts = torch.full((bs * 7,), -1, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, 3), -1, dtype=torch.int32, device=device)
        accept_num = torch.full((bs,), -1, dtype=torch.int32, device=device)
        exact_tree_speculative_sampling(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_num,
            candidates=candidates,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next,
            retrive_next_sibling=retrieve_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=final_coins,
            target_probs=probs,
        )
        torch.cuda.synchronize()
        return predicts, accept_index, accept_num

    def test_accepts_a_branch_and_samples_its_bonus(self):
        device = torch.device("cuda")
        coins = torch.zeros((1, 7), dtype=torch.float32, device=device)
        coins[0, 0] = 0.10  # root -> token 1
        coins[0, 1] = 0.25  # token 1 -> token 5 (after token 4's 0.10 mass)
        predicts, accept_index, accept_num = self._run(
            coins, torch.tensor([0.50], dtype=torch.float32, device=device)
        )
        self.assertEqual(accept_num.item(), 2)
        self.assertEqual(accept_index[0].tolist(), [0, 1, 5])
        self.assertEqual(predicts[0].item(), 1)
        self.assertEqual(predicts[1].item(), 5)
        # Row 5 CDF crosses 0.5 at token 5.
        self.assertEqual(predicts[5].item(), 5)

    def test_terminal_residual_removes_only_that_prefix_siblings(self):
        device = torch.device("cuda")
        coins = torch.zeros((1, 7), dtype=torch.float32, device=device)
        coins[0, 0] = 0.10  # root -> token 1
        coins[0, 1] = 0.90  # reject children 4 and 5
        predicts, accept_index, accept_num = self._run(
            coins, torch.tensor([0.50], dtype=torch.float32, device=device)
        )
        self.assertEqual(accept_num.item(), 1)
        self.assertEqual(accept_index[0, :2].tolist(), [0, 1])
        self.assertEqual(predicts[0].item(), 1)
        # Row 1 with tokens 4 and 5 removed has mass .7; .5*.7=.35, whose
        # token-ordered CDF crosses at token 6.
        self.assertEqual(predicts[1].item(), 6)

    def test_first_output_matches_target_distribution(self):
        device = torch.device("cuda")
        bs = 32768
        generator = torch.Generator(device=device).manual_seed(94721)
        coins = torch.rand((bs, 7), generator=generator, device=device)
        final_coins = torch.rand((bs,), generator=generator, device=device)
        predicts, _, _ = self._run(coins, final_coins)
        first_tokens = predicts.reshape(bs, 7)[:, 0].to(torch.long)
        observed = torch.bincount(first_tokens, minlength=8).float() / bs
        expected = _target_probs(1, device)[0, 0]
        torch.testing.assert_close(observed, expected, rtol=0, atol=0.012)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestExactTreeSworSamplingJit(CustomTestCase):
    def _run(
        self,
        candidates: torch.Tensor,
        probs: torch.Tensor,
        draft_probs: torch.Tensor,
        coins: torch.Tensor,
        final_coins: torch.Tensor,
    ):
        device = candidates.device
        bs = candidates.shape[0]
        retrieve_index = torch.arange(
            bs * 3, dtype=torch.long, device=device
        ).reshape(bs, 3)
        retrieve_next = torch.tensor(
            [[1, -1, -1]], dtype=torch.long, device=device
        ).repeat(bs, 1)
        retrieve_sibling = torch.tensor(
            [[-1, 2, -1]], dtype=torch.long, device=device
        ).repeat(bs, 1)
        predicts = torch.full((bs * 3,), -1, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, 2), -1, dtype=torch.int32, device=device)
        accept_num = torch.full((bs,), -1, dtype=torch.int32, device=device)
        exact_tree_swor_sampling(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_num,
            candidates=candidates,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next,
            retrive_next_sibling=retrieve_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=final_coins,
            target_probs=probs,
            draft_probs=draft_probs,
        )
        torch.cuda.synchronize()
        return predicts, accept_index, accept_num

    def test_accepts_second_ordered_sibling_after_residual_update(self):
        device = torch.device("cuda")
        p = torch.tensor([[[0.50, 0.30, 0.20]] * 3], device=device)
        q = torch.tensor([[[0.20, 0.60, 0.20]] * 3], device=device)
        candidates = torch.tensor([[2, 1, 0]], dtype=torch.long, device=device)
        coins = torch.zeros((1, 3), dtype=torch.float32, device=device)
        coins[0, 1] = 0.90  # reject token 1: p/q = 0.5
        coins[0, 2] = 0.10  # accept token 0 from the updated R/D
        predicts, accept_index, accept_num = self._run(
            candidates, p.clone(), q, coins, torch.tensor([0.0], device=device)
        )
        self.assertEqual(accept_num.item(), 1)
        self.assertEqual(accept_index[0].tolist(), [0, 2])
        self.assertEqual(predicts[0].item(), 0)

    def test_terminal_sample_uses_recursive_residual(self):
        device = torch.device("cuda")
        p = torch.tensor([[[0.50, 0.30, 0.20]] * 3], device=device)
        q = torch.tensor([[[0.20, 0.60, 0.20]] * 3], device=device)
        candidates = torch.tensor([[2, 1, 2]], dtype=torch.long, device=device)
        coins = torch.ones((1, 3), dtype=torch.float32, device=device)
        predicts, _, accept_num = self._run(
            candidates, p.clone(), q, coins, torch.tensor([0.25], device=device)
        )
        self.assertEqual(accept_num.item(), 0)
        self.assertEqual(predicts[0].item(), 0)

    def test_dense_target_support_fallback_remains_exact(self):
        device = torch.device("cuda")
        p = torch.zeros((1, 3, 80), device=device)
        p[..., 15:] = 1.0 / 65.0
        q = torch.zeros_like(p)
        q[..., 0] = 1.0
        candidates = torch.tensor([[0, 0, 1]], dtype=torch.long, device=device)
        coins = torch.ones((1, 3), dtype=torch.float32, device=device)

        predicts, _, accept_num = self._run(
            candidates, p, q, coins, torch.zeros(1, device=device)
        )

        self.assertEqual(accept_num.item(), 0)
        self.assertEqual(predicts[0].item(), 15)

    def test_first_output_matches_target_distribution(self):
        device = torch.device("cuda")
        bs = 32768
        generator = torch.Generator(device=device).manual_seed(73129)
        p_root = torch.tensor([0.08, 0.20, 0.15, 0.10, 0.07, 0.09, 0.11, 0.20], device=device)
        q_root = torch.tensor([0.16, 0.05, 0.10, 0.18, 0.08, 0.13, 0.20, 0.10], device=device)
        siblings = torch.multinomial(
            q_root.expand(bs, -1), 2, replacement=False, generator=generator
        )
        candidates = torch.cat(
            [torch.zeros((bs, 1), dtype=torch.long, device=device), siblings], dim=1
        )
        p = p_root.view(1, 1, -1).expand(bs, 3, -1).clone()
        q = q_root.view(1, 1, -1).expand(bs, 3, -1).contiguous()
        coins = torch.rand((bs, 3), generator=generator, device=device)
        final_coins = torch.rand((bs,), generator=generator, device=device)
        predicts, _, _ = self._run(candidates, p, q, coins, final_coins)
        observed = torch.bincount(
            predicts.reshape(bs, 3)[:, 0].to(torch.long), minlength=8
        ).float() / bs
        torch.testing.assert_close(observed, p_root, rtol=0, atol=0.012)

    def _run_production_topology(
        self,
        root_q: torch.Tensor,
        target_root: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        device = root_q.device
        bs, vocab = root_q.shape
        topology = default_swor_topology(device)
        q_blocks = [root_q.unsqueeze(1)]
        token_blocks = []
        sampled = fast_sample(root_q, 4)[1]
        torch.cuda.synchronize()
        for step in range(topology.num_steps):
            _, _, visible = select_swor_topology_step(topology, step, sampled, None)
            torch.cuda.synchronize()
            token_blocks.append(visible)
            if step + 1 == topology.num_steps:
                break
            q_block = root_q.unsqueeze(1).expand(-1, 4, -1).contiguous()
            q_blocks.append(q_block)
            sampled = fast_sample(q_block.reshape(bs * 4, vocab), 4)[1]
            torch.cuda.synchronize()

        parent_list, selected, draft_tokens, draft_probs = organize_swor_draft_results(
            topology, token_blocks, q_blocks
        )
        bonus = torch.zeros((bs,), dtype=torch.long, device=device)
        # The topology metadata is request-invariant. Build it through the
        # production organizer/kernel once, then tile its links for the large
        # Monte Carlo batch without allocating a quadratic tree mask per draw.
        _, _, _, retrieve_next_one, retrieve_sibling_one, _ = (
            build_tree_kernel_efficient(
                bonus[:1],
                parent_list[:1],
                selected[:1],
                draft_tokens[:1],
                torch.zeros((1,), dtype=torch.long, device=device),
                0,
                4,
                4,
                12,
                TreeMaskMode.QLEN_ONLY,
            )
        )
        torch.cuda.synchronize()
        candidates = torch.cat((bonus.unsqueeze(1), draft_tokens), dim=1)
        retrieve_index = torch.arange(
            bs * 12, dtype=torch.long, device=device
        ).reshape(bs, 12)
        retrieve_next = retrieve_next_one.expand(bs, -1).contiguous()
        retrieve_sibling = retrieve_sibling_one.expand(bs, -1).contiguous()
        target_probs = target_root.unsqueeze(1).expand(-1, 12, -1).contiguous()
        predicts = torch.full((bs * 12,), -1, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, 5), -1, dtype=torch.int32, device=device)
        accept_num = torch.full((bs,), -1, dtype=torch.int32, device=device)
        exact_tree_swor_sampling(
            predicts,
            accept_index,
            accept_num,
            candidates.reshape(bs, 12),
            retrieve_index,
            retrieve_next,
            retrieve_sibling,
            torch.rand((bs, 12), generator=generator, device=device),
            torch.rand((bs,), generator=generator, device=device),
            target_probs,
            draft_probs,
        )
        torch.cuda.synchronize()
        return predicts.reshape(bs, 12)[:, 0].long()

    def test_production_pipeline_first_token_exactness(self):
        device = torch.device("cuda")
        samples_per_case = 8192
        cases = [
            # overlapping support
            ([2.0, 1.0, 0.5, 0.0, -1.0, -2.0], 1.0, [0.30, 0.20, 0.10, 0.15, 0.15, 0.10]),
            # disjoint target emphasis
            ([8.0, 7.0, 6.0, 5.0, -8.0, -9.0], 1.0, [0.00, 0.00, 0.00, 0.00, 0.55, 0.45]),
            # repeated high-q mistakes
            ([9.0, 8.0, 7.0, 6.0, -4.0, -5.0], 1.0, [0.02, 0.02, 0.02, 0.02, 0.46, 0.46]),
            # top-p support collapse
            ([12.0, 0.0, -1.0, -2.0, -3.0, -4.0], 0.20, [0.10, 0.15, 0.20, 0.20, 0.20, 0.15]),
            # full sibling rejection and residual fallback
            ([7.0, 6.0, 5.0, 4.0, -7.0, -8.0], 1.0, [0.00, 0.00, 0.00, 0.00, 0.25, 0.75]),
        ]
        logits = torch.tensor(
            [case[0] for case in cases], dtype=torch.float32, device=device
        ).repeat_interleave(samples_per_case, dim=0)
        top_ps = torch.tensor(
            [case[1] for case in cases], dtype=torch.float32, device=device
        ).repeat_interleave(samples_per_case)
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((logits.shape[0], 1), device=device),
            top_ps=top_ps,
            acc_additive_penalties=None,
            logit_bias=None,
            need_top_p_sampling=True,
        )
        root_q = renorm_draft_probs(
            logits, sampling_info, False, draft_sampling_top_k=4
        )
        target = torch.tensor(
            [case[2] for case in cases], dtype=torch.float32, device=device
        ).repeat_interleave(samples_per_case, dim=0)
        generator = torch.Generator(device=device).manual_seed(99173)
        first = self._run_production_topology(root_q, target, generator)
        for case_index, case in enumerate(cases):
            segment = first[
                case_index * samples_per_case : (case_index + 1) * samples_per_case
            ]
            observed = torch.bincount(segment, minlength=6).float() / samples_per_case
            expected = torch.tensor(case[2], dtype=torch.float32, device=device)
            torch.testing.assert_close(observed, expected, rtol=0, atol=0.018)

    def test_support_exhaustion_sizes_are_exact(self):
        device = torch.device("cuda")
        bs = 4096
        generator = torch.Generator(device=device).manual_seed(8812)
        torch.cuda.manual_seed(8812)
        target = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05], device=device)
        for support_size in (0, 1, 3, 4):
            q = torch.zeros((bs, 8), device=device)
            if support_size:
                q[:, :support_size] = 1.0 / support_size
            first = self._run_production_topology(
                q, target.expand(bs, -1), generator
            )
            observed = torch.bincount(first, minlength=8).float() / bs
            torch.testing.assert_close(observed, target, rtol=0, atol=0.022)

    def test_branch_factor_one_matches_exact_chain_sampler(self):
        device = torch.device("cuda")
        bs, nodes, vocab = 2048, 4, 7
        generator = torch.Generator(device=device).manual_seed(4421)
        candidates = torch.randint(
            vocab, (bs, nodes), generator=generator, device=device
        )
        retrieve_index = torch.arange(bs * nodes, device=device).reshape(bs, nodes)
        retrieve_next = (
            torch.tensor([[1, 2, 3, -1]], device=device).expand(bs, -1).contiguous()
        )
        retrieve_sibling = torch.full_like(retrieve_next, -1)
        p = torch.rand((bs, nodes, vocab), generator=generator, device=device)
        p.div_(p.sum(dim=-1, keepdim=True))
        q = torch.rand((bs, nodes, vocab), generator=generator, device=device)
        q.div_(q.sum(dim=-1, keepdim=True))
        chain_coins = torch.rand((bs, nodes), generator=generator, device=device)
        tree_coins = torch.zeros_like(chain_coins)
        tree_coins[:, 1:] = chain_coins[:, :3]
        final_coins = torch.rand((bs,), generator=generator, device=device)

        def outputs():
            return (
                torch.full((bs * nodes,), -1, dtype=torch.int32, device=device),
                torch.full((bs, nodes), -1, dtype=torch.int32, device=device),
                torch.full((bs,), -1, dtype=torch.int32, device=device),
            )

        tree_predict, tree_index, tree_num = outputs()
        chain_predict, chain_index, chain_num = outputs()
        exact_tree_swor_sampling(
            tree_predict, tree_index, tree_num, candidates, retrieve_index,
            retrieve_next, retrieve_sibling, tree_coins, final_coins, p.clone(), q
        )
        chain_speculative_sampling_triton(
            chain_predict, chain_index, chain_num, candidates, retrieve_index,
            retrieve_next, retrieve_sibling, chain_coins, final_coins, p.clone(),
            q, 1.0, 1.0, True
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(tree_predict, chain_predict, rtol=0, atol=0)
        torch.testing.assert_close(tree_index, chain_index, rtol=0, atol=0)
        torch.testing.assert_close(tree_num, chain_num, rtol=0, atol=0)

    def test_cuda_graph_replay_reads_current_proposal_and_topology_buffers(self):
        device = torch.device("cuda")
        topology = default_swor_topology(device)
        token_blocks = [
            torch.tensor([[0, 1, 2, 3]], device=device),
            torch.tensor([[0, 1, 2, 3]], device=device),
            torch.tensor([[0, 1]], device=device),
            torch.tensor([[0]], device=device),
        ]
        q_placeholders = [
            torch.zeros((1, 1, 6), device=device),
            torch.zeros((1, 4, 6), device=device),
            torch.zeros((1, 4, 6), device=device),
            torch.zeros((1, 4, 6), device=device),
        ]
        parent_list, selected, draft_tokens, _ = organize_swor_draft_results(
            topology, token_blocks, q_placeholders
        )
        seq_lens = torch.zeros(1, dtype=torch.long, device=device)
        _, _, retrieve_index, retrieve_next, retrieve_sibling, candidates_flat = (
            build_tree_kernel_efficient(
                torch.zeros(1, dtype=torch.long, device=device), parent_list,
                selected, draft_tokens, seq_lens, 0, 4, 4, 12,
                TreeMaskMode.QLEN_ONLY
            )
        )
        candidates = candidates_flat.reshape(1, 12)
        logits = torch.tensor([[12.0, 0, 0, 0, 0, 0]], device=device)
        top_ps = torch.tensor([0.9], device=device)
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((1, 1), device=device), top_ps=top_ps,
            acc_additive_penalties=None, logit_bias=None, need_top_p_sampling=True
        )
        target_source = torch.zeros((1, 12, 6), device=device)
        target_source[..., 0] = 1.0
        target_scratch = torch.empty_like(target_source)
        q_current = torch.empty_like(target_source)
        predicts = torch.full((12,), -1, dtype=torch.int32, device=device)
        accept_index = torch.full((1, 5), -1, dtype=torch.int32, device=device)
        accept_num = torch.full((1,), -1, dtype=torch.int32, device=device)
        coins = torch.zeros((1, 12), device=device)
        final_coins = torch.zeros((1,), device=device)

        def captured_step():
            q_root = renorm_draft_probs(
                logits, sampling_info, False, draft_sampling_top_k=4
            )
            q_current.copy_(q_root.unsqueeze(1).expand_as(q_current))
            target_scratch.copy_(target_source)
            exact_tree_swor_sampling(
                predicts, accept_index, accept_num, candidates, retrieve_index,
                retrieve_next, retrieve_sibling, coins, final_coins,
                target_scratch, q_current
            )

        captured_step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_step()
        graph.replay()
        torch.cuda.synchronize()
        predict_storage = predicts.data_ptr()
        self.assertEqual(predicts[0].item(), 0)
        self.assertEqual(q_current[0, 0].argmax().item(), 0)

        logits.copy_(torch.tensor([[0, 0, 0, 0, 12.0, 0]], device=device))
        top_ps.fill_(0.2)
        candidates[0, 1:5].copy_(torch.tensor([4, 3, 2, 1], device=device))
        target_source.zero_()
        target_source[..., 4] = 1.0
        predicts.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(predicts.data_ptr(), predict_storage)
        self.assertEqual(predicts[0].item(), 4)
        self.assertEqual(q_current[0, 0].argmax().item(), 4)


if __name__ == "__main__":
    unittest.main()
