from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_draft_topk1_delta_module() -> Module:
    return load_jit(
        "draft_topk1_delta",
        cuda_files=["speculative/draft_topk1_delta.cuh"],
        cuda_wrappers=[
            ("build", "DraftTopK1DeltaKernel<false, false>::run"),
            ("build_additive", "DraftTopK1DeltaKernel<true, false>::run"),
            ("build_bias", "DraftTopK1DeltaKernel<false, true>::run"),
            ("build_additive_bias", "DraftTopK1DeltaKernel<true, true>::run"),
        ],
    )


def preload_draft_topk1_delta() -> None:
    _jit_draft_topk1_delta_module()


def draft_topk1_delta(
    logits: torch.Tensor,
    additive: torch.Tensor | None = None,
    logit_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or not logits.is_contiguous():
        raise ValueError("draft top-k1 delta logits must be contiguous rank two")
    if logits.dtype != torch.float32 or not logits.is_cuda:
        raise ValueError("draft top-k1 delta logits must be CUDA float32")
    if not logits.shape[0] or not logits.shape[1]:
        raise ValueError("draft top-k1 delta logits must be nonempty")
    for name, value in (("additive", additive), ("logit_bias", logit_bias)):
        if value is not None and (
            value.shape != logits.shape
            or value.dtype != torch.float32
            or value.device != logits.device
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"draft top-k1 delta {name} must match contiguous CUDA float32 logits"
            )
    rows, vocab_size = logits.shape
    num_splits = (vocab_size + 8191) // 8192
    q = torch.empty_like(logits, dtype=torch.float32)
    topk_p = torch.empty((rows, 1), dtype=torch.float32, device=logits.device)
    topk_index = torch.empty((rows, 1), dtype=torch.int64, device=logits.device)
    partial_values = torch.empty(
        (rows, num_splits), dtype=torch.float32, device=logits.device
    )
    partial_indices = torch.empty(
        (rows, num_splits), dtype=torch.int32, device=logits.device
    )
    module = _jit_draft_topk1_delta_module()
    if additive is None and logit_bias is None:
        module.build(
            logits,
            logits,
            logits,
            q,
            topk_p,
            topk_index,
            partial_values,
            partial_indices,
        )
    elif logit_bias is None:
        module.build_additive(
            logits,
            additive,
            logits,
            q,
            topk_p,
            topk_index,
            partial_values,
            partial_indices,
        )
    elif additive is None:
        module.build_bias(
            logits,
            logits,
            logit_bias,
            q,
            topk_p,
            topk_index,
            partial_values,
            partial_indices,
        )
    else:
        module.build_additive_bias(
            logits,
            additive,
            logit_bias,
            q,
            topk_p,
            topk_index,
            partial_values,
            partial_indices,
        )
    return q, topk_p, topk_index
