"""Benchmark one packed GGUF matrix through the native Metal extension."""

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
    parser.add_argument("tensor_name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.gguf_path)
    tensor = next(t for t in reader.tensors if t.name == args.tensor_name)
    output_size, encoded_size = tensor.data.shape
    block_size, type_size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
    input_size = encoded_size // type_size * block_size
    packed = torch.from_numpy(np.array(tensor.data, copy=True)).to("mps")
    x = torch.randn(args.batch_size, input_size, device="mps")

    def run():
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0:
            return q4_0_matmul(
                packed.view(torch.uint8), x, output_size, input_size
            )
        return quant_matmul(
            packed.view(torch.uint8),
            x,
            output_size,
            input_size,
            int(tensor.tensor_type),
        )

    run()
    torch.mps.synchronize()
    timings = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        run()
        torch.mps.synchronize()
        timings.append(time.perf_counter() - start)
    median = statistics.median(timings)
    gib = tensor.data.nbytes / (1024**3)
    print(
        f"{tensor.tensor_type.name} {tuple(tensor.shape)} batch={args.batch_size} "
        f"median={median * 1000:.3f}ms packed_bandwidth={gib / median:.1f}GiB/s"
    )


if __name__ == "__main__":
    main()
