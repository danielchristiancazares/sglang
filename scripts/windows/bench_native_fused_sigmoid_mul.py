"""Compare native C++/CUDA and Triton Qwen attention-output gating."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import triton

from sglang.kernels.ops.elementwise.elementwise import _fused_sigmoid_mul_kernel
from sglang.kernels.ops.elementwise.fused_sigmoid_mul import (
    fused_sigmoid_mul_native,
)


def _triton(attn: torch.Tensor, gate: torch.Tensor, output: torch.Tensor) -> None:
    rows, hidden = attn.shape
    block_h = 1024 if rows < 1024 else 2048
    _fused_sigmoid_mul_kernel[(rows, triton.cdiv(hidden, block_h))](
        output,
        attn,
        gate,
        gate.stride(0),
        gate.stride(0),
        hidden,
        HEAD_DIM=hidden,
        BLOCK_H=block_h,
        num_warps=4,
    )


def _measure(fn, warmup: int, repeats: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / iterations / 1000.0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(20260816)
    attn = torch.randn(
        (args.rows, args.hidden),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    gate = torch.randn(
        (args.rows, args.hidden),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    native_out = torch.empty_like(attn)
    triton_out = torch.empty_like(attn)

    def native():
        fused_sigmoid_mul_native(attn, gate, output=native_out)

    def triton_impl():
        _triton(attn, gate, triton_out)

    native()
    triton_impl()
    torch.cuda.synchronize()
    if not torch.equal(native_out, triton_out):
        mismatches = torch.count_nonzero(native_out != triton_out).item()
        raise AssertionError(f"native/Triton mismatch count: {mismatches}")

    native_us = _measure(native, args.warmup, args.repeats, args.iterations)
    triton_us = _measure(triton_impl, args.warmup, args.repeats, args.iterations)
    native_median = statistics.median(native_us)
    triton_median = statistics.median(triton_us)
    print(f"shape: rows={args.rows}, hidden={args.hidden}, dtype=bf16")
    print(f"native median: {native_median:.3f} us")
    print(f"Triton median: {triton_median:.3f} us")
    print(f"speedup: {triton_median / native_median:.3f}x")
    print(f"saved: {triton_median - native_median:.3f} us/call")


if __name__ == "__main__":
    main()
