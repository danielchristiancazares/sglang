from __future__ import annotations

import os
import pathlib
import sys
from typing import TYPE_CHECKING

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def is_supported_silu_and_mul_nvfp4(
    hidden_size: int,
    dtype: torch.dtype,
) -> bool:
    return (
        sys.platform == "win32"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (10, 0)
        and dtype == torch.bfloat16
        and hidden_size > 0
        and hidden_size % 16 == 0
        and not _env_enabled("FLASHINFER_NVFP4_4OVER6")
    )


@cache_once
def _jit_silu_and_mul_nvfp4_module(
    dtype: torch.dtype,
    disable_quant_fast_math: bool,
) -> Module:
    import flashinfer

    flashinfer_root = pathlib.Path(flashinfer.__file__).resolve().parent
    nv_internal_root = flashinfer_root / "data" / "csrc" / "nv_internal"
    args = make_cpp_args(
        dtype,
        is_arch_support_pdl(),
        disable_quant_fast_math,
    )
    return load_jit(
        "silu_and_mul_nvfp4",
        *args,
        cuda_files=["gemm/silu_and_mul_nvfp4.cuh"],
        cuda_wrappers=[
            (
                "silu_and_mul_nvfp4",
                f"SiluAndMulNVFP4Kernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["-DENABLE_BF16", "-DENABLE_FP8", "-DENABLE_FP4"],
        extra_include_paths=[
            str(nv_internal_root),
            str(nv_internal_root / "include"),
        ],
        extra_dependencies=["flashinfer"],
    )


@register_custom_op(
    op_name="silu_and_mul_nvfp4",
    mutates_args=["output", "output_scale"],
)
def _silu_and_mul_nvfp4_inplace(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> None:
    hidden_size = input.shape[-1] // 2
    if not is_supported_silu_and_mul_nvfp4(hidden_size, input.dtype):
        raise RuntimeError(
            "silu_and_mul_nvfp4 is unsupported for "
            f"hidden_size={hidden_size}, dtype={input.dtype}"
        )
    module = _jit_silu_and_mul_nvfp4_module(
        input.dtype,
        _env_enabled("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"),
    )
    module.silu_and_mul_nvfp4(input, global_scale, output, output_scale)


@debug_kernel_api
def silu_and_mul_nvfp4(
    input: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = input.shape[-1] // 2
    num_rows = input.numel() // input.shape[-1]
    input_2d = input.reshape(num_rows, hidden_size * 2)
    output = torch.empty(
        (num_rows, hidden_size // 2),
        dtype=torch.uint8,
        device=input.device,
    )
    scale_rows = ((num_rows + 127) // 128) * 128
    scale_cols = ((hidden_size // 16 + 3) // 4) * 4
    output_scale = torch.empty(
        (scale_rows, scale_cols),
        dtype=torch.uint8,
        device=input.device,
    )
    _silu_and_mul_nvfp4_inplace(
        input_2d,
        global_scale.reshape(1),
        output,
        output_scale,
    )
    return output, output_scale
