import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.quantization.gptq_marlin_repack import (
    gptq_marlin_repack,
)
from sglang.kernels.ops.quantization.nvfp4_marlin_relayout import (
    nvfp4_marlin_relayout_,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_nvfp4_layer_for_marlin,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.quant_ref_utils import quantize_nvfp4_shard
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=20,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestNvfp4MarlinRelayout(CustomTestCase):
    def test_native_nvfp4_wrapper_matches_dequantized_reference(self):
        torch.manual_seed(62064)
        size_m, size_n, size_k = 4, 192, 256
        x = torch.randn(
            (size_m, size_k),
            dtype=torch.bfloat16,
            device="cuda",
        ) / 10
        weight = torch.randn(
            (size_n, size_k),
            dtype=torch.bfloat16,
            device="cuda",
        ) / 10
        weight_fp4, weight_scale, quant_scale, weight_ref = quantize_nvfp4_shard(
            weight
        )
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(weight_fp4, requires_grad=False),
            weight_scale=torch.nn.Parameter(weight_scale, requires_grad=False),
            weight_global_scale=torch.nn.Parameter(
                (1.0 / quant_scale).to(torch.bfloat16).reshape(1),
                requires_grad=False,
            ),
            output_size_per_partition=size_n,
            input_size_per_partition=size_k,
            params_dtype=torch.bfloat16,
            quant_config=SimpleNamespace(group_size=16),
            bias=None,
        )
        prepare_nvfp4_layer_for_marlin(layer)

        output = apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=size_n,
            size_k=size_k,
        )
        output_ref = torch.matmul(x.float(), weight_ref.T).to(torch.bfloat16)
        torch.cuda.synchronize()

        torch.testing.assert_close(output, output_ref, rtol=0.04, atol=0.04)

    def test_matches_marlin_repack_and_round_trips(self):
        for size_n, size_k in ((64, 128), (256, 256), (512, 640)):
            with self.subTest(size_n=size_n, size_k=size_k):
                cutlass = torch.randint(
                    0,
                    256,
                    (size_n, size_k // 2),
                    dtype=torch.uint8,
                    device="cuda",
                )
                original = cutlass.clone()
                expected = gptq_marlin_repack(
                    b_q_weight=cutlass.view(torch.int32).T.contiguous(),
                    perm=torch.empty(0, dtype=torch.int32, device="cuda"),
                    size_k=size_k,
                    size_n=size_n,
                    num_bits=4,
                )
                storage = cutlass.reshape(-1)
                scratch = torch.empty_like(storage)

                nvfp4_marlin_relayout_(
                    storage,
                    scratch,
                    size_n=size_n,
                    size_k=size_k,
                    to_marlin=True,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    storage.view(torch.int32).view_as(expected),
                    expected,
                    rtol=0,
                    atol=0,
                )

                nvfp4_marlin_relayout_(
                    storage,
                    scratch,
                    size_n=size_n,
                    size_k=size_k,
                    to_marlin=False,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    storage.view_as(original),
                    original,
                    rtol=0,
                    atol=0,
                )

    def test_rejects_invalid_public_inputs(self):
        weight = torch.zeros(64 * 64, dtype=torch.uint8, device="cuda")
        scratch = torch.empty_like(weight)
        with self.assertRaisesRegex(ValueError, "weight"):
            nvfp4_marlin_relayout_(
                weight.cpu(),
                scratch,
                size_n=64,
                size_k=128,
                to_marlin=True,
            )
        with self.assertRaisesRegex(ValueError, "scratch"):
            nvfp4_marlin_relayout_(
                weight,
                scratch[:1],
                size_n=64,
                size_k=128,
                to_marlin=True,
            )


if __name__ == "__main__":
    unittest.main()
