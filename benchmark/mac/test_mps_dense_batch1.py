"""Parity and dispatch coverage for Apple batch-one F32 GGUF matmul."""

from __future__ import annotations

import argparse

import gguf
import numpy as np
import torch

import sglang.srt.layers.quantization.gguf as gguf_quant


F32 = int(gguf.GGMLQuantizationType.F32)


def _check_case(
    name: str,
    weight: torch.Tensor,
    inputs: torch.Tensor,
) -> float:
    expected = inputs @ weight.T
    actual = gguf_quant.fused_mul_mat_gguf(
        inputs.to("mps"),
        weight.to("mps"),
        F32,
    ).cpu()
    error = (actual - expected).abs()
    max_error = error.max().item()
    max_reference = expected.abs().max().item()
    relative_error = max_error / max(max_reference, 1e-12)
    print(
        f"{name:22s} batch={inputs.shape[0]:2d} "
        f"shape={tuple(weight.shape)!s:12s} "
        f"max_error={max_error:.6g} relative={relative_error:.6g}"
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)
    return max_error


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
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(29)
    actual_weights = _merged_ba_weights(gguf.GGUFReader(args.gguf_path))
    assert len(actual_weights) == 48
    assert all(tuple(weight.shape) == (96, 5120) for weight in actual_weights)

    original_dense_matmul = gguf_quant.dense_matmul

    def reject_dense_matmul(*_args, **_kwargs):
        raise AssertionError("batch one unexpectedly selected dense_matmul")

    gguf_quant.dense_matmul = reject_dense_matmul
    try:
        _check_case(
            "batch-one selector",
            actual_weights[0],
            torch.randn(1, 5120, generator=generator),
        )
    finally:
        gguf_quant.dense_matmul = original_dense_matmul

    fallback_calls = []

    def record_dense_matmul(weight, inputs):
        fallback_calls.append((weight.shape, inputs.shape))
        return original_dense_matmul(weight, inputs)

    gguf_quant.dense_matmul = record_dense_matmul
    try:
        for batch in (2, 3, 4, 8):
            _check_case(
                "multi-batch fallback",
                actual_weights[0],
                torch.randn(batch, 5120, generator=generator),
            )
    finally:
        gguf_quant.dense_matmul = original_dense_matmul
    assert len(fallback_calls) == 4

    for rows in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 31, 32, 47, 48, 95, 96, 97):
        _check_case(
            "output-row boundary",
            torch.randn(rows, 33, generator=generator),
            torch.randn(1, 33, generator=generator),
        )
    for width in (1, 31, 32, 33, 63, 64, 65, 5120):
        _check_case(
            "input-width boundary",
            torch.randn(7, width, generator=generator),
            torch.randn(1, width, generator=generator),
        )

    actual_errors = []
    for layer, weight in enumerate(actual_weights):
        actual_errors.append(
            _check_case(
                f"actual layer {layer}",
                weight,
                torch.randn(1, 5120, generator=generator),
            )
        )
    print(
        f"actual_layers={len(actual_errors)} "
        f"max_actual_error={max(actual_errors):.9g}"
    )


if __name__ == "__main__":
    main()
