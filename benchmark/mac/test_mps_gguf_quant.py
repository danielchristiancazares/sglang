"""Correctness smoke for the native Apple GGUF Metal kernels."""

from __future__ import annotations

import argparse

import gguf
import numpy as np
import torch

from sglang.srt.hardware_backend.mps.ops import (
    q4_0_matmul,
    quant_embedding,
    quant_matmul,
)


CASES = (
    gguf.GGMLQuantizationType.Q4_0,
    gguf.GGMLQuantizationType.Q4_1,
    gguf.GGMLQuantizationType.Q2_K,
    gguf.GGMLQuantizationType.Q4_K,
    gguf.GGMLQuantizationType.Q5_K,
    gguf.GGMLQuantizationType.Q6_K,
    gguf.GGMLQuantizationType.IQ2_XXS,
    gguf.GGMLQuantizationType.IQ1_M,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.gguf_path)
    tensors = list(reader.tensors)
    generator = torch.Generator().manual_seed(7)

    tested = set()
    for expected_type in CASES:
        tensor = next(
            (tensor for tensor in tensors if tensor.tensor_type == expected_type),
            None,
        )
        if tensor is None:
            continue
        tested.add(expected_type)
        packed_np = np.array(tensor.data[: args.rows], copy=True)
        dense_np = gguf.dequantize(packed_np, tensor.tensor_type)
        dense = torch.from_numpy(dense_np).to(torch.float32)
        input_size = dense.shape[1]
        x = torch.randn(args.batch_size, input_size, generator=generator)
        expected = x @ dense.T
        packed = torch.from_numpy(packed_np).to("mps").view(torch.uint8)
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0:
            actual = q4_0_matmul(
                packed,
                x.to("mps"),
                args.rows,
                input_size,
            ).cpu()
        else:
            actual = quant_matmul(
                packed,
                x.to("mps"),
                args.rows,
                input_size,
                int(tensor.tensor_type),
            ).cpu()
        max_error = (actual - expected).abs().max().item()
        max_reference = expected.abs().max().item()
        relative_error = max_error / max(max_reference, 1e-12)
        print(
            f"{tensor.tensor_type.name:8s} {tensor.name:30s} "
            f"max_error={max_error:.6g} relative={relative_error:.6g}"
        )
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-3)

    embedding = next(
        (
            tensor
            for tensor in tensors
            if tensor.name == "token_embd.weight"
            and tensor.tensor_type
            in {
                gguf.GGMLQuantizationType.Q2_K,
                gguf.GGMLQuantizationType.Q4_K,
                gguf.GGMLQuantizationType.IQ2_XXS,
                gguf.GGMLQuantizationType.IQ1_M,
            }
        ),
        None,
    )
    if embedding is not None:
        packed_np = np.array(embedding.data[: args.rows], copy=True)
        dense_np = gguf.dequantize(packed_np, embedding.tensor_type)
        token_ids = torch.tensor([0, args.rows - 1], dtype=torch.int64)
        actual = quant_embedding(
            torch.from_numpy(packed_np).to("mps").view(torch.uint8),
            token_ids.to("mps"),
            args.rows,
            dense_np.shape[1],
            int(embedding.tensor_type),
        ).cpu()
        expected = torch.from_numpy(dense_np)[token_ids]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        print(f"{embedding.tensor_type.name:8s} token embedding parity passed")

    print(
        "tested="
        + ",".join(sorted(quant_type.name for quant_type in tested))
    )


if __name__ == "__main__":
    main()
