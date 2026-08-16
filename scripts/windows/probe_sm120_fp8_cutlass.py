"""Build and compare the isolated SM120 CUTLASS channelwise FP8 GEMM."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import torch
from torch.utils.cpp_extension import load


REPO_ROOT = Path(__file__).resolve().parents[2]
CUTLASS_ROOT = Path.home() / "cutlass-sglang"
SOURCE = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "kernels"
    / "jit"
    / "csrc"
    / "gemm"
    / "fp8_scaled_mm_sm120_windows.cu"
)


def build_extension() -> None:
    load(
        name="sgl_kernel_windows_fp8_sm120_aligned",
        sources=[str(SOURCE)],
        extra_include_paths=[
            str(REPO_ROOT / "python" / "sglang" / "kernels" / "aot" / "csrc"),
            str(REPO_ROOT / "python" / "sglang" / "kernels" / "aot" / "include"),
            str(CUTLASS_ROOT / "include"),
            str(CUTLASS_ROOT / "tools" / "util" / "include"),
        ],
        extra_cflags=["/std:c++20", "/O2"],
        extra_cuda_cflags=[
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-Xcompiler=/Zc:preprocessor",
            "-gencode=arch=compute_120a,code=sm_120a",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DCUTLASS_VERSIONS_GENERATED",
        ],
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )


def elapsed_ms(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark_pair(
    fallback_fn,
    candidate_fn,
    warmup: int = 32,
    iterations: int = 128,
    rounds: int = 9,
) -> dict[str, float]:
    for _ in range(warmup):
        fallback_fn()
        candidate_fn()
    torch.cuda.synchronize()

    fallback_samples = []
    candidate_samples = []
    paired_speedups = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            fallback_ms = elapsed_ms(fallback_fn, iterations)
            candidate_ms = elapsed_ms(candidate_fn, iterations)
        else:
            candidate_ms = elapsed_ms(candidate_fn, iterations)
            fallback_ms = elapsed_ms(fallback_fn, iterations)
        fallback_samples.append(fallback_ms)
        candidate_samples.append(candidate_ms)
        paired_speedups.append(fallback_ms / candidate_ms)

    return {
        "fallback_ms": median(fallback_samples),
        "fallback_ms_min": min(fallback_samples),
        "fallback_ms_max": max(fallback_samples),
        "cutlass_ms": median(candidate_samples),
        "cutlass_ms_min": min(candidate_samples),
        "cutlass_ms_max": max(candidate_samples),
        "speedup": median(paired_speedups),
        "speedup_min": min(paired_speedups),
        "speedup_max": max(paired_speedups),
    }


def make_fp8_operands(
    m: int, k: int, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    activation = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    activation_scale = (
        activation.float().abs().amax().clamp(min=1e-12) / 448.0
    ).reshape(1)
    activation_fp8 = (
        (activation.float() / activation_scale)
        .clamp(min=-448.0, max=448.0)
        .to(torch.float8_e4m3fn)
    )

    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    weight_scale = (
        weight.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 448.0
    )
    weight_fp8 = (
        (weight.float() / weight_scale)
        .clamp(min=-448.0, max=448.0)
        .to(torch.float8_e4m3fn)
        .t()
    )
    return activation_fp8, weight_fp8, activation_scale, weight_scale


def main() -> None:
    build_extension()
    from sglang.srt.layers.quantization.fp8_utils import (
        fp8_scaled_mm as fallback_fp8_scaled_mm,
    )

    cutlass_fp8_scaled_mm = torch.ops.sgl_kernel_windows.fp8_scaled_mm
    torch.manual_seed(1786880606)
    shapes = (
        (1, 5120, 10240),
        (2, 5120, 10240),
        (4, 5120, 10240),
        (1, 5120, 6144),
        (2, 5120, 6144),
        (4, 5120, 6144),
        (1, 6144, 5120),
        (2, 6144, 5120),
        (4, 6144, 5120),
        (1, 5120, 12288),
        (2, 5120, 12288),
        (4, 5120, 12288),
        (1, 5120, 1024),
        (2, 5120, 1024),
        (4, 5120, 1024),
    )
    results = []
    for m, k, n in shapes:
        activation, weight, activation_scale, weight_scale = make_fp8_operands(
            m, k, n
        )

        def run_fallback() -> torch.Tensor:
            return fallback_fp8_scaled_mm(
                activation,
                weight,
                activation_scale,
                weight_scale,
                torch.bfloat16,
            )

        def run_cutlass() -> torch.Tensor:
            return cutlass_fp8_scaled_mm(
                activation,
                weight,
                activation_scale,
                weight_scale,
                torch.bfloat16,
                None,
            )

        fallback = run_fallback()
        candidate = run_cutlass()
        torch.cuda.synchronize()
        denominator = fallback.float().abs().mean().clamp(min=1e-12)
        relative_mae = (
            (candidate.float() - fallback.float()).abs().mean() / denominator
        )
        timings = benchmark_pair(run_fallback, run_cutlass)
        results.append(
            {
                "shape_mkn": [m, k, n],
                "finite": bool(torch.isfinite(candidate).all()),
                "relative_mae_vs_fallback": round(float(relative_mae), 8),
                **{key: round(value, 6) for key, value in timings.items()},
            }
        )
        del activation, weight, activation_scale, weight_scale, fallback, candidate
        torch.cuda.empty_cache()

    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
