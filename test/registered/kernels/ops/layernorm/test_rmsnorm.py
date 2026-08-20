import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=160, stage="nightly", runner_config="1-gpu-large")
register_amd_ci(est_time=45, suite="jit-kernel-unit-test-amd")


EPS = 1e-6
DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16]


def sglang_jit_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    eps: float = EPS,
) -> None:
    from sglang.kernels.ops.layernorm.norm import rmsnorm

    rmsnorm(input, weight, out=output, eps=eps)


def flashinfer_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor,
    eps: float = EPS,
) -> None:
    from flashinfer.norm import rmsnorm

    rmsnorm(input, weight, out=output, eps=eps)


def torch_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor,
    eps: float = EPS,
) -> None:
    x = input.float()
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    output.copy_((normed * weight.float()).to(output.dtype))


def reference_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor,
    eps: float = EPS,
) -> None:
    # NVIDIA uses flashinfer (the bitwise reference); flashinfer is CUDA-only,
    # so on ROCm fall back to the torch reference (matches flashinfer math).
    if is_hip():
        torch_rmsnorm(input, weight, output=output, eps=eps)
    else:
        flashinfer_rmsnorm(input, weight, output=output, eps=eps)


BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
SUPPORTED_HIDDEN_SIZE_LIST = [
    64,
    128,
    256,
    512,
    *range(1024, 8192 + 1, 1024),
    1536,
    2304,
    2560,
    8704,
    12288,
    16384,
]
RMSNORM_CASES = get_ci_test_range(
    list(itertools.product(BS_LIST, SUPPORTED_HIDDEN_SIZE_LIST)),
    [
        (1, 256),
        (18, 1024),
        (38, 4096),
        (1240, 1536),
        (2500, 1024),
        (4109, 1024),
        (7807, 128),
    ],
)


@pytest.mark.parametrize(
    "batch_size,hidden_size",
    RMSNORM_CASES,
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("specify_out", [True, False])
def test_rmsnorm(
    batch_size: int, hidden_size: int, dtype: torch.dtype, specify_out: bool
) -> None:
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)

    input_ref = input.clone()
    output_ref = torch.empty_like(input)
    reference_rmsnorm(input_ref, weight, output=output_ref)

    if specify_out:
        output_sglang = torch.empty_like(input)
        sglang_jit_rmsnorm(input, weight, output=output_sglang)
    else:
        output_sglang = input.clone()
        sglang_jit_rmsnorm(output_sglang, weight, output=output_sglang)

    torch.testing.assert_close(output_sglang, output_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512, 8192, 8704, 16384])
def test_rmsnorm_hidden_size_support(hidden_size: int) -> None:
    from sglang.kernels.ops.layernorm.norm import _is_supported_rmsnorm_hidden_size

    assert _is_supported_rmsnorm_hidden_size(hidden_size)


@pytest.mark.parametrize(
    ("hidden_size", "expected"),
    [
        (64, "RMSNormWarpKernel"),
        (128, "RMSNormWarpKernel"),
        (256, "RMSNormWarpKernel"),
        (512, "RMSNormHalfKernel"),
        (1536, "RMSNormKernel"),
        (2048, "RMSNormHalfKernel"),
        (2304, "RMSNormKernel"),  # NOTE: not 512 aligned
        (8192, "RMSNormHalfKernel"),
        (8704, "RMSNormHalfKernel"),
        (16384, "RMSNormHalfKernel"),
    ],
)
def test_rmsnorm_kernel_dispatch(hidden_size: int, expected: str) -> None:
    from sglang.kernels.ops.layernorm.norm import _rmsnorm_kernel_class

    assert _rmsnorm_kernel_class(hidden_size) == expected


@pytest.mark.parametrize("batch_size", [1, 3])
def test_qwen35_gemma_rmsnorm_weight_offset(batch_size: int) -> None:
    from sglang.kernels.ops.layernorm.norm import gemma_rmsnorm

    torch.manual_seed(1)
    hidden_size = 5120
    input = torch.randn(
        batch_size, hidden_size, device=DEVICE, dtype=torch.bfloat16
    )
    weight = torch.randn(hidden_size, device=DEVICE, dtype=torch.bfloat16)
    output = torch.empty_like(input)

    gemma_rmsnorm(input, weight, out=output, eps=EPS)
    normalized = input.float()
    normalized *= torch.rsqrt(
        normalized.square().mean(dim=-1, keepdim=True) + EPS
    )
    expected = (normalized * (weight.float() + 1.0)).to(input.dtype)

    if sys.platform == "win32" and torch.cuda.get_device_capability()[0] >= 12:
        torch.testing.assert_close(output, expected, atol=0.0, rtol=0.0)
    else:
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_qwen35_gemma_fused_add_rmsnorm_exact(batch_size: int) -> None:
    if sys.platform != "win32" or torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("Native Windows SM120 dispatch only")

    from sglang.kernels.ops.layernorm.norm import gemma_rmsnorm
    from sglang.srt.layers.layernorm import gemma_fused_add_rmsnorm

    torch.manual_seed(2)
    hidden_size = 5120
    input = torch.randn(
        batch_size, hidden_size, device=DEVICE, dtype=torch.bfloat16
    )
    residual = torch.randn_like(input)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=torch.bfloat16)

    expected_input = torch.empty_like(input)
    expected_residual = residual.clone()
    expected_residual.add_(input)
    gemma_rmsnorm(
        expected_residual,
        weight,
        out=expected_input,
        eps=EPS,
    )

    actual_input = input.clone()
    actual_residual = residual.clone()
    gemma_fused_add_rmsnorm(
        actual_input,
        actual_residual,
        weight,
        EPS,
    )

    torch.testing.assert_close(actual_input, expected_input, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        actual_residual, expected_residual, atol=0.0, rtol=0.0
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
