from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_nvfp4_marlin_relayout_module() -> Module:
    return load_jit(
        "nvfp4_marlin_relayout",
        cuda_files=["gemm/marlin/nvfp4_marlin_relayout.cuh"],
        cuda_wrappers=[
            ("nvfp4_marlin_relayout_inplace", "nvfp4_marlin_relayout_inplace")
        ],
    )


def preload_nvfp4_marlin_relayout() -> None:
    _jit_nvfp4_marlin_relayout_module()


def nvfp4_marlin_relayout_(
    weight: torch.Tensor,
    scratch: torch.Tensor,
    *,
    size_n: int,
    size_k: int,
    to_marlin: bool,
) -> None:
    expected_bytes = size_n * size_k // 2
    if (
        weight.device.type != "cuda"
        or weight.dtype != torch.uint8
        or weight.ndim != 1
        or not weight.is_contiguous()
        or weight.numel() != expected_bytes
    ):
        raise ValueError(
            "NVFP4 Marlin relayout weight must be a contiguous CUDA uint8 "
            f"vector with {expected_bytes} elements"
        )
    if (
        scratch.device != weight.device
        or scratch.dtype != torch.uint8
        or scratch.ndim != 1
        or not scratch.is_contiguous()
        or scratch.numel() < expected_bytes
    ):
        raise ValueError(
            "NVFP4 Marlin relayout scratch must be a contiguous CUDA uint8 "
            "vector on the weight device with sufficient capacity"
        )
    _jit_nvfp4_marlin_relayout_module().nvfp4_marlin_relayout_inplace(
        weight,
        scratch,
        size_n,
        size_k,
        to_marlin,
    )
