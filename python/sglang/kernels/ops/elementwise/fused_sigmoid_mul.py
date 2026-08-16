"""Native-Windows CUDA attention-output sigmoid gate."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MAX_VEC_ELEMS = 16


@cache_once
def _jit_fused_sigmoid_mul_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "fused_sigmoid_mul_bf16",
        *args,
        cuda_files=["elementwise/fused_sigmoid_mul.cuh"],
        cuda_wrappers=[("run", f"FusedSigmoidMulKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def covered(attn_output: torch.Tensor, gate: torch.Tensor) -> bool:
    """The native production path is the contiguous BF16 CUDA gate emitted by
    Qwen's fused QK-norm/RoPE preparation on native Windows."""
    return (
        sys.platform == "win32"
        and attn_output.is_cuda
        and gate.is_cuda
        and attn_output.dtype == gate.dtype == torch.bfloat16
        and attn_output.shape == gate.shape
        and attn_output.is_contiguous()
        and gate.is_contiguous()
        and attn_output.numel() > 0
        and attn_output.numel() % _MAX_VEC_ELEMS == 0
    )


@register_custom_op(mutates_args=["output"])
def _fused_sigmoid_mul_inplace(
    attn_output: torch.Tensor, gate: torch.Tensor, output: torch.Tensor
) -> None:
    _jit_fused_sigmoid_mul_module().run(
        attn_output.view(-1), gate.view(-1), output.view(-1)
    )


@register_custom_op(mutates_args=["attn_output"])
def _fused_sigmoid_mul_attn_inplace(
    attn_output: torch.Tensor, gate: torch.Tensor
) -> None:
    _jit_fused_sigmoid_mul_module().run(
        attn_output.view(-1), gate.view(-1), attn_output.view(-1)
    )


def fused_sigmoid_mul_native(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if output is None:
        output = torch.empty_like(attn_output)
    if output is attn_output:
        _fused_sigmoid_mul_attn_inplace(attn_output, gate)
    else:
        _fused_sigmoid_mul_inplace(attn_output, gate, output)
    return output


def preload_fused_sigmoid_mul() -> None:
    _jit_fused_sigmoid_mul_module()
