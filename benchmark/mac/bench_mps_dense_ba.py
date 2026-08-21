"""Benchmark the native Apple batch-one GDN b/a F32 projection path."""

from __future__ import annotations

import argparse
import statistics
import time

import gguf
import numpy as np
import torch

from sglang.srt.hardware_backend.mps.ops import dense_matmul
from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf


F32 = int(gguf.GGMLQuantizationType.F32)


def _merged_ba_weights(reader: gguf.GGUFReader) -> list[torch.Tensor]:
    alpha = {
        int(tensor.name.split(".")[1]): tensor
        for tensor in reader.tensors
        if tensor.name.endswith("ssm_alpha.weight")
    }
    beta = {
        int(tensor.name.split(".")[1]): tensor
        for tensor in reader.tensors
        if tensor.name.endswith("ssm_beta.weight")
    }
    return [
        torch.from_numpy(
            np.concatenate(
                (
                    np.array(alpha[layer].data, copy=True),
                    np.array(beta[layer].data, copy=True),
                ),
                axis=0,
            )
        )
        for layer in sorted(set(alpha) & set(beta))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=9)
    args = parser.parse_args()

    weights = [
        weight.to("mps")
        for weight in _merged_ba_weights(gguf.GGUFReader(args.gguf_path))
    ]
    generator = torch.Generator().manual_seed(19)
    inputs = torch.randn(
        len(weights),
        weights[0].shape[1],
        generator=generator,
        dtype=torch.float32,
    ).to("mps")

    def candidate() -> list[torch.Tensor]:
        return [
            fused_mul_mat_gguf(inputs[index : index + 1], weight, F32)
            for index, weight in enumerate(weights)
        ]

    def control() -> list[torch.Tensor]:
        return [
            dense_matmul(weight, inputs[index : index + 1])
            for index, weight in enumerate(weights)
        ]

    for _ in range(args.warmup):
        candidate()
        control()
    torch.mps.synchronize()

    def measure(run) -> list[float]:
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            run()
            torch.mps.synchronize()
            samples.append((time.perf_counter() - started) * 1000)
        return samples

    candidate_a = measure(candidate)
    control_samples = measure(control)
    candidate_b = measure(candidate)

    candidate_outputs = candidate()
    control_outputs = control()
    torch.mps.synchronize()
    max_error = max(
        (actual - expected).abs().max().item()
        for actual, expected in zip(candidate_outputs, control_outputs)
    )

    print(
        f"layers={len(weights)} shape={tuple(weights[0].shape)} "
        f"bytes_per_sweep={sum(weight.numel() * 4 for weight in weights)}"
    )
    for name, samples in (
        ("candidate_a", candidate_a),
        ("control", control_samples),
        ("candidate_b", candidate_b),
    ):
        print(f"{name}_ms=" + ",".join(f"{sample:.6f}" for sample in samples))
        print(f"{name}_median_ms={statistics.median(samples):.6f}")
    print(f"candidate_control_max_abs={max_error:.9g}")


if __name__ == "__main__":
    main()
