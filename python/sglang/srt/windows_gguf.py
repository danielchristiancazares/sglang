"""Native-Windows loader for SGLang's GGUF CUDA kernels."""

from __future__ import annotations

import functools
import os
import shutil
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _activate_msvc() -> None:
    if os.name != "nt" or shutil.which("cl") is not None:
        return
    from torch.utils.cpp_extension import _get_vc_env

    os.environ.update(
        {key.upper(): value for key, value in _get_vc_env("x86_amd64").items()}
    )


@functools.cache
def _ops():
    _activate_msvc()
    aot_root = Path(__file__).parents[1] / "kernels" / "aot"
    gguf_root = aot_root / "csrc" / "quantization" / "gguf"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    load(
        name="sglang_windows_gguf",
        sources=[
            str(gguf_root / "windows_extension.cpp"),
            str(gguf_root / "gguf_kernel.cu"),
        ],
        extra_include_paths=[str(aot_root / "include"), str(gguf_root)],
        extra_cflags=["/O2", "/std:c++20", "/Zc:preprocessor"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-Xcompiler=/Zc:preprocessor",
            "-DFLASHINFER_ENABLE_F16",
            "-DFLASHINFER_ENABLE_BF16",
        ],
        is_python_module=False,
        verbose=os.environ.get("SGLANG_JIT_KERNEL_VERBOSE", "0") == "1",
    )
    return torch.ops.sglang_windows_gguf


def ggml_dequantize(*args, **kwargs):
    return _ops().ggml_dequantize(*args, **kwargs)


def ggml_mul_mat_vec_a8(*args, **kwargs):
    return _ops().ggml_mul_mat_vec_a8(*args, **kwargs)


def ggml_mul_mat_a8(*args, **kwargs):
    return _ops().ggml_mul_mat_a8(*args, **kwargs)


def ggml_moe_a8(*args, **kwargs):
    return _ops().ggml_moe_a8(*args, **kwargs)


def ggml_moe_a8_vec(*args, **kwargs):
    return _ops().ggml_moe_a8_vec(*args, **kwargs)


def ggml_moe_get_block_size(*args, **kwargs):
    return _ops().ggml_moe_get_block_size(*args, **kwargs)
