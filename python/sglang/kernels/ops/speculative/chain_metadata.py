from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_chain_metadata_module() -> Module:
    return load_jit(
        "chain_metadata",
        cuda_files=["speculative/chain_metadata.cuh"],
        cuda_wrappers=[("build", "ChainMetadataKernel::run")],
    )


def preload_chain_metadata() -> None:
    """Compile/load the tiny module while startup still has VRAM headroom."""
    _jit_chain_metadata_module()


def build_chain_metadata(
    bonus_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    tree_mask: torch.Tensor,
):
    """Build fixed-top-k1 verification metadata and tokens in one CUDA launch.

    Outputs are newly allocated per cycle so their lifetime remains identical
    to the general tree path.  This matters because verification consumes them
    asynchronously after the builder returns.
    """
    bs, num_steps = draft_tokens.shape
    num_slots = num_steps + 1
    positions = torch.empty(bs * num_slots, dtype=torch.long, device=seq_lens.device)
    retrieve_buf = torch.empty(
        (3, bs, num_slots), dtype=torch.long, device=seq_lens.device
    )
    output_tokens = torch.empty(
        bs * num_slots, dtype=draft_tokens.dtype, device=draft_tokens.device
    )
    _jit_chain_metadata_module().build(
        bonus_tokens,
        draft_tokens,
        seq_lens,
        tree_mask,
        positions,
        retrieve_buf,
        output_tokens,
    )
    return positions, retrieve_buf, output_tokens
