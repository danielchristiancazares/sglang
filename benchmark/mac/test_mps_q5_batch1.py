"""Boundary parity for the native Apple Q5_K batch-one matvec."""

from __future__ import annotations

import argparse

import gguf
import numpy as np
import torch

from sglang.srt.hardware_backend.mps.ops import quant_matmul


Q5 = gguf.GGMLQuantizationType.Q5_K
Q5_BLOCK_BYTES = 176
Q5_BLOCK_SIZE = 256


def _mps_packed_view(packed_np: np.ndarray, storage_offset: int) -> torch.Tensor:
    packed = torch.from_numpy(packed_np).to("mps").view(torch.uint8)
    if storage_offset == 0:
        return packed

    storage = torch.empty(
        packed.numel() + storage_offset,
        dtype=torch.uint8,
        device="mps",
    )
    view = storage[storage_offset:].view(packed.shape)
    view.copy_(packed)
    assert view.storage_offset() == storage_offset
    return view


def _mps_input_view(input_cpu: torch.Tensor, storage_offset: int) -> torch.Tensor:
    input_mps = input_cpu.to("mps")
    if storage_offset == 0:
        return input_mps

    storage = torch.empty(
        input_mps.numel() + storage_offset,
        dtype=torch.float32,
        device="mps",
    )
    view = storage[storage_offset:].view(input_mps.shape)
    view.copy_(input_mps)
    assert view.storage_offset() == storage_offset
    return view


def _check_case(
    name: str,
    packed_np: np.ndarray,
    generator: torch.Generator,
    *,
    weight_offset: int = 0,
    input_offset: int = 0,
) -> None:
    dense_np = gguf.dequantize(packed_np, Q5)
    dense = torch.from_numpy(dense_np).to(torch.float32)
    rows, input_size = dense.shape
    input_cpu = torch.randn(1, input_size, generator=generator)
    expected = input_cpu @ dense.T
    actual = quant_matmul(
        _mps_packed_view(packed_np, weight_offset),
        _mps_input_view(input_cpu, input_offset),
        rows,
        input_size,
        int(Q5),
    ).cpu()
    error = (actual - expected).abs()
    max_error = error.max().item()
    max_reference = expected.abs().max().item()
    relative_error = max_error / max(max_reference, 1e-12)
    print(
        f"{name:22s} rows={rows:2d} k={input_size:4d} "
        f"woff={weight_offset:2d} xoff={input_offset:2d} "
        f"max_error={max_error:.6g} relative={relative_error:.6g}"
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-3)


def _synthetic_extrema(rows: int, input_size: int) -> np.ndarray:
    blocks_per_row = input_size // Q5_BLOCK_SIZE
    packed = np.zeros(
        (rows, blocks_per_row * Q5_BLOCK_BYTES),
        dtype=np.uint8,
    )
    scale_patterns = (
        np.zeros(12, dtype=np.uint8),
        np.full(12, 0xFF, dtype=np.uint8),
        np.array(
            [
                0x00,
                0x3F,
                0xC0,
                0xFF,
                0x15,
                0x2A,
                0x3F,
                0xC0,
                0x55,
                0xAA,
                0x0F,
                0xF0,
            ],
            dtype=np.uint8,
        ),
    )
    high_patterns = (0x00, 0xFF, 0x55, 0xAA)
    low_patterns = (0x00, 0xFF, 0x0F, 0xF0, 0x5A, 0xA5)
    for row in range(rows):
        for block_index in range(blocks_per_row):
            base = block_index * Q5_BLOCK_BYTES
            d = np.float16(0.03125 * (1 + (row + block_index) % 7))
            dmin = np.float16(0.015625 * (1 + (2 * row + block_index) % 5))
            packed[row, base : base + 2] = np.frombuffer(
                d.tobytes(), dtype=np.uint8
            )
            packed[row, base + 2 : base + 4] = np.frombuffer(
                dmin.tobytes(), dtype=np.uint8
            )
            packed[row, base + 4 : base + 16] = scale_patterns[
                (row + block_index) % len(scale_patterns)
            ]
            packed[row, base + 16 : base + 48] = high_patterns[
                (row + block_index) % len(high_patterns)
            ]
            packed[row, base + 48 : base + 176] = low_patterns[
                (row + 2 * block_index) % len(low_patterns)
            ]
    return packed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    args = parser.parse_args()

    tensor = next(
        tensor
        for tensor in gguf.GGUFReader(args.gguf_path).tensors
        if tensor.name == "output.weight" and tensor.tensor_type == Q5
    )
    generator = torch.Generator().manual_seed(23)

    for input_size in (256, 768):
        packed_bytes = input_size // Q5_BLOCK_SIZE * Q5_BLOCK_BYTES
        for rows in (1, 7, 8, 9, 15, 16, 17, 31, 32):
            _check_case(
                "actual-file prefix",
                np.array(tensor.data[:rows, :packed_bytes], copy=True),
                generator,
            )

    _check_case(
        "actual long/compact",
        np.array(tensor.data[:17], copy=True),
        generator,
        weight_offset=8,
        input_offset=4,
    )
    _check_case(
        "vector-align fallback",
        np.array(tensor.data[:17, : 3 * Q5_BLOCK_BYTES], copy=True),
        generator,
        weight_offset=2,
        input_offset=1,
    )
    _check_case(
        "synthetic extrema",
        _synthetic_extrema(17, 768),
        generator,
        weight_offset=8,
        input_offset=4,
    )


if __name__ == "__main__":
    main()
