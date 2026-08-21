"""Microbenchmark native Apple GGUF matrix-vector kernels."""

from __future__ import annotations

import argparse
import statistics
import time

import gguf
import numpy as np
import torch

from sglang.srt.hardware_backend.mps.ops import q4_0_matmul, quant_matmul


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument("tensor_name", nargs="?")
    parser.add_argument("--tensor", dest="tensor_override")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args()
    tensor_name = (
        args.tensor_override or args.tensor_name or "blk.0.ffn_gate.weight"
    )

    reader = gguf.GGUFReader(args.gguf_path)
    tensor = next(
        (candidate for candidate in reader.tensors if candidate.name == tensor_name),
        None,
    )
    if tensor is None:
        raise ValueError(f"tensor {tensor_name!r} is absent from the GGUF file")

    packed_np = np.array(tensor.data, copy=True)
    input_size = gguf.dequantize(
        np.array(tensor.data[:1], copy=True), tensor.tensor_type
    ).shape[1]
    output_size = packed_np.shape[0]
    packed = torch.from_numpy(packed_np).to("mps").view(torch.uint8)
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(
        args.batch_size, input_size, generator=generator, dtype=torch.float32
    ).to("mps")

    def run() -> torch.Tensor:
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0:
            return q4_0_matmul(packed, inputs, output_size, input_size)
        return quant_matmul(
            packed,
            inputs,
            output_size,
            input_size,
            int(tensor.tensor_type),
        )

    for _ in range(args.warmup):
        run()
    torch.mps.synchronize()

    samples_ms = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        run()
        torch.mps.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1000)

    median_ms = statistics.median(samples_ms)
    touched_bytes = (
        packed.numel()
        + inputs.numel() * inputs.element_size()
        + args.batch_size * output_size * torch.float32.itemsize
    )
    bandwidth_gib_s = touched_bytes / (median_ms / 1000) / (1024**3)
    print(
        f"tensor={tensor.name} type={tensor.tensor_type.name} "
        f"shape=({output_size},{input_size}) batch={args.batch_size}"
    )
    print("samples_ms=" + ",".join(f"{sample:.6f}" for sample in samples_ms))
    print(f"median_ms={median_ms:.6f} effective_gib_s={bandwidth_gib_s:.3f}")


if __name__ == "__main__":
    main()
