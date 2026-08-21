from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_flashinfer_page_table_module(page_size: int) -> Module:
    if page_size <= 1 or page_size & (page_size - 1):
        raise ValueError(
            f"page_size must be a power of two greater than one, got {page_size}"
        )
    args = make_cpp_args(page_size)
    return load_jit(
        "flashinfer_page_table",
        *args,
        cuda_files=["kvcache/flashinfer_page_table.cuh"],
        cuda_wrappers=[
            (
                "build",
                f"FlashInferPageTableKernel<{args}>::run",
            )
        ],
    )


def preload_flashinfer_page_table(page_size: int) -> None:
    _jit_flashinfer_page_table_module(page_size)


@debug_kernel_api
def build_flashinfer_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_lens: torch.Tensor,
    page_indptr: torch.Tensor,
    total_pages: int,
    page_size: int,
    physical_page_count: int,
) -> torch.Tensor:
    page_indices = torch.empty(
        total_pages + 256,
        dtype=torch.int32,
        device=req_to_token.device,
    )
    _jit_flashinfer_page_table_module(page_size).build(
        page_indices,
        req_to_token,
        req_pool_indices,
        page_lens,
        page_indptr,
        total_pages,
        physical_page_count,
    )
    return page_indices
