from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.speculative.tree_sampling import (
    exact_tree_speculative_sampling,
    exact_tree_swor_sampling,
)
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


if __name__ == "__main__":
    unittest.main()
