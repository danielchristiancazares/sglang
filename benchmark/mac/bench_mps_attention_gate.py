"""Compare the PyTorch and native Metal Qwen attention output gates."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from sglang.srt.hardware_backend.mps.ops import sigmoid_mul_inplace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=6144)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(17)
    source = torch.randn(args.batch_size, args.width, generator=generator).to("mps")
    gate = torch.randn(args.batch_size, args.width, generator=generator).to("mps")
    output = torch.empty_like(source)

    def pytorch_gate() -> None:
        output.copy_(source)
        output.mul_(torch.sigmoid(gate))

    def native_gate() -> None:
        output.copy_(source)
        sigmoid_mul_inplace(output, gate)

    def measure(operation) -> list[float]:
        operation()
        torch.mps.synchronize()
        timings = []
        for _ in range(args.trials):
            start = time.perf_counter()
            for _ in range(args.iterations):
                operation()
            torch.mps.synchronize()
            timings.append((time.perf_counter() - start) / args.iterations)
        return timings

    pytorch_timings = measure(pytorch_gate)
    native_timings = measure(native_gate)
    print(
        f"shape=({args.batch_size}, {args.width}) "
        f"pytorch_median={statistics.median(pytorch_timings) * 1e6:.3f}us "
        f"native_median={statistics.median(native_timings) * 1e6:.3f}us"
    )
    print("pytorch_us=" + ",".join(f"{value * 1e6:.3f}" for value in pytorch_timings))
    print("native_us=" + ",".join(f"{value * 1e6:.3f}" for value in native_timings))


if __name__ == "__main__":
    main()
