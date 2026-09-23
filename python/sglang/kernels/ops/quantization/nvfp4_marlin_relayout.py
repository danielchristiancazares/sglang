from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import tvm_ffi

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_nvfp4_marlin_relayout_module() -> Module:
    return load_jit(
        "nvfp4_marlin_relayout",
        cuda_files=["gemm/marlin/nvfp4_marlin_relayout.cuh"],
        cuda_wrappers=[
            ("nvfp4_marlin_relayout_inplace", "nvfp4_marlin_relayout_inplace"),
            (
                "nvfp4_marlin_scale_relayout_inplace",
                "nvfp4_marlin_scale_relayout_inplace",
            ),
            (
                "register_nvfp4_hybrid_marlin_state",
                "register_nvfp4_hybrid_marlin_state",
            ),
        ],
    )


@cache_once
def _nvfp4_hybrid_marlin_state_cls():
    _jit_nvfp4_marlin_relayout_module().register_nvfp4_hybrid_marlin_state()

    @tvm_ffi.register_object("sgl.Nvfp4HybridMarlinState")
    class Nvfp4HybridMarlinState(tvm_ffi.Object):
        __slots__ = ()

        def __init__(self, lazy_relayout: bool):
            self.__ffi_init__(lazy_relayout)

    return Nvfp4HybridMarlinState


def create_nvfp4_hybrid_marlin_state(layers, *, lazy_relayout: bool):
    state = _nvfp4_hybrid_marlin_state_cls()(lazy_relayout)
    for layer in layers:
        state.add_layer(
            tvm_ffi.from_dlpack(layer.weight.view(torch.uint8).view(-1)),
            tvm_ffi.from_dlpack(layer.weight_scale.view(torch.uint8).view(-1)),
            layer.output_size_per_partition,
            layer.input_size_per_partition,
        )
    return state


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


def nvfp4_marlin_scale_relayout_(
    scale: torch.Tensor,
    scratch: torch.Tensor,
    *,
    size_n: int,
    size_k: int,
    to_marlin: bool,
) -> None:
    expected_scales = size_n * size_k // 16
    scale_bytes = scale.view(torch.uint8)
    if (
        size_n <= 0
        or size_n % 128 != 0
        or size_k <= 0
        or size_k % 64 != 0
        or scale.device.type != "cuda"
        or scale.dtype != torch.float8_e4m3fn
        or scale.ndim != 2
        or not scale.is_contiguous()
        or scale.numel() != expected_scales
    ):
        raise ValueError(
            "NVFP4 Marlin relayout scale must use CUTLASS's 128-row/four-group "
            "tile alignment and be a contiguous CUDA E4M3 matrix with "
            f"{expected_scales} elements"
        )
    if (
        scratch.device != scale.device
        or scratch.dtype != torch.uint8
        or scratch.ndim != 1
        or not scratch.is_contiguous()
        or scratch.numel() < expected_scales
    ):
        raise ValueError(
            "NVFP4 Marlin scale relayout scratch must be a contiguous CUDA "
            "uint8 vector on the scale device with sufficient capacity"
        )
    _jit_nvfp4_marlin_relayout_module().nvfp4_marlin_scale_relayout_inplace(
        scale_bytes.reshape(-1),
        scratch,
        size_n,
        size_k,
        to_marlin,
    )
