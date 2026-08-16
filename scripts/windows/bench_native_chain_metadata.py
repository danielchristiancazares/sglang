"""Microbenchmark the native fixed-chain metadata builder against Windows Triton."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from sglang.kernels.ops.speculative.chain_metadata import build_chain_metadata
from sglang.srt.speculative.eagle_utils import sgl_build_tree_kernel_triton
from sglang.srt.speculative.tree_mask import TreeMaskMode


def _measure(fn, *, warmup: int, repeats: int, iterations: int) -> list[float]:
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cuda")
    bs = args.batch_size
    steps = args.steps
    slots = steps + 1
    bonus = torch.arange(100, 100 + bs, dtype=torch.long, device=device)
    drafts = torch.arange(
        1000, 1000 + bs * steps, dtype=torch.long, device=device
    ).reshape(bs, steps)
    seq_lens = torch.full((bs,), 6213, dtype=torch.long, device=device)
    parents = torch.arange(-1, steps - 1, dtype=torch.long, device=device).repeat(
        bs, 1
    )
    selected = torch.arange(steps, dtype=torch.long, device=device).repeat(bs, 1)
    tree_mask = torch.empty(
        bs * slots * slots, dtype=torch.bool, device=device
    )

    def triton_path():
        output_tokens = torch.cat((bonus[:, None], drafts), dim=1).flatten()
        retrieve_buf = torch.full(
            (3, bs, slots), -1, dtype=torch.long, device=device
        )
        positions = torch.empty(bs * slots, dtype=torch.long, device=device)
        sgl_build_tree_kernel_triton(
            parents,
            selected,
            seq_lens,
            tree_mask,
            positions,
            retrieve_buf[0],
            retrieve_buf[1],
            retrieve_buf[2],
            1,
            steps,
            slots,
            TreeMaskMode.QLEN_ONLY,
        )
        return positions, retrieve_buf, output_tokens

    def native_path():
        return build_chain_metadata(bonus, drafts, seq_lens, tree_mask)

    triton_out = triton_path()
    native_out = native_path()
    torch.cuda.synchronize()
    for actual, expected in zip(native_out, triton_out):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    triton_us = _measure(
        triton_path,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    native_us = _measure(
        native_path,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    triton_median = statistics.median(triton_us)
    native_median = statistics.median(native_us)
    print(f"shape: bs={bs}, steps={steps}, slots={slots}")
    print(f"triton full path median: {triton_median:.3f} us")
    print(f"native full path median: {native_median:.3f} us")
    print(f"speedup: {triton_median / native_median:.3f}x")
    print(f"saved: {triton_median - native_median:.3f} us/cycle")


if __name__ == "__main__":
    main()
