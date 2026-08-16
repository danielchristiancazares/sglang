from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch
import triton

from sglang.kernels.ops.elementwise.elementwise import (
    _fused_sigmoid_mul_kernel,
    fused_sigmoid_mul,
)
from sglang.kernels.ops.elementwise.fused_sigmoid_mul import (
    fused_sigmoid_mul_native,
)
from sglang.test.test_utils import CustomTestCase


def _triton_reference(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    rows, hidden = attn_output.shape
    output = torch.empty_like(attn_output)
    block_h = 1024 if rows < 1024 else 2048
    _fused_sigmoid_mul_kernel[(rows, triton.cdiv(hidden, block_h))](
        output,
        attn_output,
        gate,
        gate.stride(0),
        gate.stride(0),
        hidden,
        HEAD_DIM=hidden,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestNativeFusedSigmoidMul(CustomTestCase):
    def test_qwen_bf16_shapes_match_triton_bit_exact(self):
        generator = torch.Generator(device="cuda").manual_seed(3801)
        for rows in (1, 3, 64):
            attn = torch.randn(
                (rows, 6144),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            gate = torch.randn(
                (rows, 6144),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            expected = _triton_reference(attn, gate)
            actual = fused_sigmoid_mul_native(attn, gate)
            self.assertTrue(torch.equal(actual, expected), f"rows={rows}")

    def test_inplace_dispatch_matches_triton_bit_exact(self):
        generator = torch.Generator(device="cuda").manual_seed(3802)
        attn = torch.randn(
            (3, 6144),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        gate = torch.randn(
            (3, 6144),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        expected = _triton_reference(attn, gate)
        data_ptr = attn.data_ptr()
        actual = fused_sigmoid_mul(attn, gate, inplace=True)
        self.assertEqual(actual.data_ptr(), data_ptr)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
