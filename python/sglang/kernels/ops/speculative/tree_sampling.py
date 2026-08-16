from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_exact_tree_sampling_module() -> Module:
    return load_jit(
        "exact_tree_speculative_sampling",
        cuda_files=["speculative/exact_tree_speculative_sampling.cuh"],
        cuda_wrappers=[
            ("sample", "ExactTreeSpeculativeSamplingKernel::run"),
            ("sample_swor", "ExactTreeSworSamplingKernel::run"),
        ],
    )


def preload_exact_tree_sampling() -> None:
    """Compile/load the verifier while model startup still has VRAM headroom."""
    _jit_exact_tree_sampling_module()


def exact_tree_speculative_sampling(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor | None = None,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """Sample exactly from target ``p`` while traversing a proposal tree.

    At each accepted prefix the proposal children partition their exact target
    mass on one uniform draw.  A draw outside that mass samples from ``p`` with
    only those child IDs removed.  The proposal distribution is therefore not
    part of the correction and no vocabulary-sized draft tensor is needed.

    The misspelled ``retrive_*`` names intentionally match the historical
    ``sgl_kernel`` operator schema used by the caller.
    """
    del draft_probs, deterministic
    if threshold_single != 1.0 or threshold_acc != 1.0:
        raise ValueError(
            "Exact tree sampling requires threshold_single == threshold_acc == 1.0"
        )
    _jit_exact_tree_sampling_module().sample(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
    )


def exact_tree_swor_sampling(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """Exact tree verification for ordered draft siblings sampled without replacement.

    ``target_probs`` is scratch for this operation: rejected siblings update the
    active parent row in place to the normalized residual distribution.
    """
    del deterministic
    if threshold_single != 1.0 or threshold_acc != 1.0:
        raise ValueError(
            "Exact tree SWOR sampling requires threshold_single == threshold_acc == 1.0"
        )
    _jit_exact_tree_sampling_module().sample_swor(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
    )
