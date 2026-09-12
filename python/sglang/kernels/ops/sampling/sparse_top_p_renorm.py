from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_sorted_top_p_renorm_module() -> Module:
    return load_jit(
        "sorted_top_p_renorm",
        cuda_files=["sampling/sorted_top_p_renorm.cuh"],
        cuda_wrappers=[
            ("sorted_top_p_renorm", "SortedTopPRenormKernel::run"),
            (
                "sorted_top_k_top_p_normalize",
                "SortedTopKTopPNormalizeKernel::run",
            ),
            ("sorted_top_k_top_p_sample", "SortedTopKTopPSampleKernel::run"),
        ],
    )


@debug_kernel_api
def sorted_top_p_renorm(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Renormalize sorted finite-support rows in one CUDA warp per row."""
    output = torch.empty_like(probs)
    _jit_sorted_top_p_renorm_module().sorted_top_p_renorm(
        probs, top_ps.reshape(-1), output
    )
    return output


@debug_kernel_api
def sorted_top_k_top_p_normalize(
    topk_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Fuse temperature softmax and top-p renormalization on sorted top-k rows."""
    probabilities = torch.empty_like(topk_logits, dtype=torch.float32)
    _jit_sorted_top_p_renorm_module().sorted_top_k_top_p_normalize(
        topk_logits,
        temperatures.reshape(-1),
        top_ks.reshape(-1),
        top_ps.reshape(-1),
        probabilities,
    )
    return probabilities


@debug_kernel_api
def sorted_top_k_top_p_sample(
    topk_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    exp_noise: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize a <=32-token sorted support and draw by exponential race."""
    probabilities = torch.empty_like(topk_logits, dtype=torch.float32)
    retained_indices = torch.empty_like(topk_indices)
    sampled_tokens = torch.empty(
        (topk_logits.shape[0],), dtype=torch.int64, device=topk_logits.device
    )
    _jit_sorted_top_p_renorm_module().sorted_top_k_top_p_sample(
        topk_logits,
        topk_indices,
        temperatures.reshape(-1),
        top_ks.reshape(-1),
        top_ps.reshape(-1),
        exp_noise,
        greedy_mask.reshape(-1),
        probabilities,
        retained_indices,
        sampled_tokens,
    )
    return sampled_tokens, probabilities, retained_indices


@cache_once
def _jit_sparse_top_p_renorm_module(max_nonzero: int) -> Module:
    if not 1 <= max_nonzero <= 1024:
        raise ValueError(
            f"max_nonzero must be in [1, 1024], got {max_nonzero}"
        )
    args = make_cpp_args(max_nonzero)
    return load_jit(
        "sparse_top_p_renorm",
        *args,
        cuda_files=["sampling/sparse_top_p_renorm.cuh"],
        cuda_wrappers=[
            (
                "sparse_top_p_renorm",
                f"SparseTopPRenormKernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["--use_fast_math"],
        extra_dependencies=["flashinfer"],
    )


@register_custom_op(
    op_name="sparse_top_p_renorm",
    mutates_args=["probs", "workspace"],
)
def _sparse_top_p_renorm_inplace(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
    workspace: torch.Tensor,
    max_nonzero: int,
) -> None:
    module = _jit_sparse_top_p_renorm_module(max_nonzero)
    module.sparse_top_p_renorm(probs, top_ps, workspace)


def _workspace_size(batch_size: int, vocab_size: int) -> int:
    def align256(value: int) -> int:
        return ((value + 255) // 256) * 256

    counter_size = 384
    num_buckets = 2048
    buf_len = max(align256(vocab_size // 32), 256)
    return (
        align256(counter_size * batch_size)
        + align256(4 * num_buckets * batch_size)
        + align256(4 * num_buckets * batch_size)
        + 2 * align256(4 * buf_len * batch_size)
    )


@debug_kernel_api
def sparse_top_p_renorm(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
    *,
    max_nonzero: int = 32,
) -> torch.Tensor:
    """Renormalize a finite-top-k probability tensor in place.

    FlashInfer AIR selects the exact pivot. Sparse rows use a one-pass apply;
    exact cutoff ties wider than ``max_nonzero`` use AIR's dense apply shape.
    """
    workspace = torch.empty(
        _workspace_size(probs.shape[0], probs.shape[1]),
        dtype=torch.uint8,
        device=probs.device,
    )
    _sparse_top_p_renorm_inplace(
        probs,
        top_ps.reshape(-1),
        workspace,
        max_nonzero,
    )
    return probs
