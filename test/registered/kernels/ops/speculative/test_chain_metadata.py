from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.speculative.chain_metadata import build_chain_metadata
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestChainMetadataJit(CustomTestCase):
    def test_fixed_topk1_chain(self):
        device = torch.device("cuda")
        for bs, num_steps in ((1, 1), (1, 2), (3, 2), (2, 7)):
            with self.subTest(bs=bs, num_steps=num_steps):
                num_slots = num_steps + 1
                bonus = torch.arange(100, 100 + bs, dtype=torch.long, device=device)
                drafts = torch.arange(
                    1000,
                    1000 + bs * num_steps,
                    dtype=torch.long,
                    device=device,
                ).reshape(bs, num_steps)
                seq_lens = torch.arange(
                    4096, 4096 + bs, dtype=torch.long, device=device
                )
                # The production XQA sink is capacity-sized; the active batch
                # may use only its prefix.
                mask = torch.full(
                    (bs * num_slots * num_slots + 17,),
                    True,
                    dtype=torch.bool,
                    device=device,
                )

                positions, retrieve_buf, tokens = build_chain_metadata(
                    bonus, drafts, seq_lens, mask
                )
                torch.cuda.synchronize()

                expected_tokens = torch.cat((bonus[:, None], drafts), dim=1).flatten()
                expected_positions = (
                    seq_lens[:, None]
                    + torch.arange(num_slots, dtype=torch.long, device=device)
                ).flatten()
                expected_index = torch.arange(
                    bs * num_slots, dtype=torch.long, device=device
                ).reshape(bs, num_slots)
                expected_next = torch.arange(
                    1, num_slots + 1, dtype=torch.long, device=device
                )
                expected_next[-1] = -1
                expected_next = expected_next.repeat(bs, 1)
                expected_sibling = torch.full_like(expected_next, -1)
                expected_mask = torch.tril(
                    torch.ones(
                        (num_slots, num_slots), dtype=torch.bool, device=device
                    )
                ).repeat(bs, 1, 1)

                torch.testing.assert_close(tokens, expected_tokens, rtol=0, atol=0)
                torch.testing.assert_close(
                    positions, expected_positions, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    retrieve_buf[0], expected_index, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    retrieve_buf[1], expected_next, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    retrieve_buf[2], expected_sibling, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    mask[: bs * num_slots * num_slots].reshape(
                        bs, num_slots, num_slots
                    ),
                    expected_mask,
                    rtol=0,
                    atol=0,
                )
                self.assertTrue(mask[bs * num_slots * num_slots :].all().item())


if __name__ == "__main__":
    unittest.main()
