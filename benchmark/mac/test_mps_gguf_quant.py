"""Correctness smoke for the native Intel-Mac GGUF Metal matmuls."""

from __future__ import annotations

import argparse

import gguf
import numpy as np
import torch

from sglang.srt.hardware_backend.mps.ops import q4_0_matmul, quant_matmul


CASES = (
    ("blk.8.ffn_gate.weight", gguf.GGMLQuantizationType.Q4_0),
    ("blk.0.ffn_down.weight", gguf.GGMLQuantizationType.Q4_1),
    ("blk.0.ssm_out.weight", gguf.GGMLQuantizationType.Q5_K),
    ("output.weight", gguf.GGMLQuantizationType.Q6_K),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.gguf_path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    generator = torch.Generator().manual_seed(7)

    for name, expected_type in CASES:
        tensor = tensors[name]
        assert tensor.tensor_type == expected_type
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
            f"{tensor.tensor_type.name:4s} {name:30s} "
            f"max_error={max_error:.6g} relative={relative_error:.6g}"
        )
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-3)


if __name__ == "__main__":
    main()
