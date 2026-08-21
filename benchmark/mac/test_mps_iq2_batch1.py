"""Boundary parity for the native Apple IQ2_XXS batch-one matvec."""

from __future__ import annotations

import argparse

import gguf
import numpy as np
import torch

from sglang.srt.hardware_backend.mps.ops import quant_matmul


IQ2 = gguf.GGMLQuantizationType.IQ2_XXS
IQ2_BLOCK_BYTES = 66
IQ2_BLOCK_SIZE = 256


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


def _check_case(
    name: str,
    packed_np: np.ndarray,
    generator: torch.Generator,
    *,
    storage_offset: int = 0,
) -> None:
    dense_np = gguf.dequantize(packed_np, IQ2)
    dense = torch.from_numpy(dense_np).to(torch.float32)
    rows, input_size = dense.shape
    x = torch.randn(1, input_size, generator=generator)
    expected = x @ dense.T
    actual = quant_matmul(
        _mps_packed_view(packed_np, storage_offset),
        x.to("mps"),
        rows,
        input_size,
        int(IQ2),
    ).cpu()
    error = (actual - expected).abs()
    max_error = error.max().item()
    max_reference = expected.abs().max().item()
    relative_error = max_error / max(max_reference, 1e-12)
    print(
        f"{name:24s} rows={rows:2d} k={input_size:4d} "
        f"offset={storage_offset:2d} max_error={max_error:.6g} "
        f"relative={relative_error:.6g}"
    )
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-3)


def _synthetic_extrema(rows: int) -> np.ndarray:
    packed = np.zeros((rows, IQ2_BLOCK_BYTES), dtype=np.uint8)
    grid_bytes = bytes((0, 255, 85, 170))
    sign_indices = (0, 1, 126, 127)
    for row in range(rows):
        scale = row % 16
        d = np.float16((row + 1) / rows)
        packed[row, :2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
        for atom in range(8):
            base = 2 + atom * 8
            rotation = atom % 4
            grids = grid_bytes[rotation:] + grid_bytes[:rotation]
            packed[row, base : base + 4] = np.frombuffer(
                grids,
                dtype=np.uint8,
            )
            signs = tuple(
                sign_indices[(index + atom + row) % 4] for index in range(4)
            )
            auxiliary = (
                signs[0]
                | (signs[1] << 7)
                | (signs[2] << 14)
                | (signs[3] << 21)
                | (scale << 28)
            )
            packed[row, base + 4 : base + 8] = np.frombuffer(
                auxiliary.to_bytes(4, "little"),
                dtype=np.uint8,
            )
    return packed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    args = parser.parse_args()

    tensor = next(
        tensor
        for tensor in gguf.GGUFReader(args.gguf_path).tensors
        if tensor.tensor_type == IQ2
    )
    generator = torch.Generator().manual_seed(17)

    for input_size in (256, 768):
        packed_bytes = input_size // IQ2_BLOCK_SIZE * IQ2_BLOCK_BYTES
        for rows in (1, 7, 8, 9, 17):
            packed_np = np.array(
                tensor.data[:rows, :packed_bytes],
                copy=True,
            )
            offset = 8 if input_size == 768 and rows == 17 else 0
            _check_case(
                "actual-file prefix",
                packed_np,
                generator,
                storage_offset=offset,
            )

    _check_case(
        "synthetic extrema",
        _synthetic_extrema(17),
        generator,
        storage_offset=8,
    )


if __name__ == "__main__":
    main()
