from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gdn_tree_replay_module() -> Module:
    return load_jit(
        "gdn_tree_replay",
        cuda_files=["attention/gdn_tree_replay.cuh"],
        cuda_wrappers=[
            ("verify", "GdnTreeReplayVerifyKernel::run"),
            ("commit", "GdnTreeReplayCommitKernel::run"),
        ],
    )


def gdn_tree_replay_verify(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    checkpoint_state: torch.Tensor,
    state_indices: torch.Tensor,
    parent: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    g_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    scale: float,
    max_tree_depth: int | None = None,
) -> torch.Tensor:
    """Evaluate a GDN proposal tree from one persistent checkpoint.

    The CUDA kernel keeps every branch as a root-state scale plus ancestor
    rank-one delta updates.  It reads the large fp32 checkpoint once, never
    materializes a full recurrent state per tree node, and records raw node
    inputs for an exact accepted-path commit after sampling.
    """
    batch_size, num_nodes = parent.shape
    if max_tree_depth is None:
        max_tree_depth = num_nodes
    total_tokens = batch_size * num_nodes
    num_value_heads = A_log.numel()
    value_dim = checkpoint_state.shape[-2]
    key_dim = checkpoint_state.shape[-1]
    num_key_heads = q.numel() // (total_tokens * key_dim)

    q_view = q.reshape(total_tokens, num_key_heads, key_dim)
    k_view = k.reshape(total_tokens, num_key_heads, key_dim)
    v_view = v.reshape(total_tokens, num_value_heads, value_dim)
    a_view = a.reshape(total_tokens, num_value_heads)
    b_view = b.reshape(total_tokens, num_value_heads)
    output = torch.empty_like(v_view)
    inv_norms = torch.empty(
        (batch_size, num_key_heads, num_nodes, 2),
        dtype=torch.float32,
        device=q.device,
    )
    replay_params = torch.empty(
        (batch_size, num_value_heads, num_nodes, 2),
        dtype=torch.float32,
        device=q.device,
    )
    pair_dots = torch.empty(
        (batch_size, num_key_heads, num_nodes, max_tree_depth, 2),
        dtype=torch.float32,
        device=q.device,
    )
    _jit_gdn_tree_replay_module().verify(
        A_log,
        a_view,
        dt_bias,
        q_view,
        k_view,
        v_view,
        b_view,
        checkpoint_state,
        state_indices,
        parent,
        rawv_cache,
        rawk_cache,
        g_cache,
        beta_cache,
        inv_norms,
        replay_params,
        pair_dots,
        output,
        max_tree_depth,
        scale,
    )
    return output.reshape(v.shape)


def commit_gdn_tree_replay_all_layers(
    *,
    checkpoint_state: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    g_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    state_indices: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    num_tree_nodes: int,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_track_nodes: torch.Tensor | None = None,
) -> None:
    """Replay only the accepted tree path into every persistent GDN layer."""
    has_track = mamba_track_indices is not None and mamba_track_nodes is not None
    _jit_gdn_tree_replay_module().commit(
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        g_cache,
        beta_cache,
        state_indices,
        accept_index,
        accept_lens,
        mamba_track_indices if has_track else state_indices,
        mamba_track_nodes if has_track else accept_lens,
        num_tree_nodes,
        has_track,
    )


def commit_gdn_tree_replay_after_verify(
    *,
    spec_state,
    state_batch_indices: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    num_tree_nodes: int,
    last_correct_node_indices: torch.Tensor,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_track_nodes: torch.Tensor | None = None,
) -> None:
    """Commit recurrent and convolution state for one accepted tree path."""
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )

    commit_gdn_tree_replay_all_layers(
        checkpoint_state=spec_state.temporal,
        rawv_cache=spec_state.replayssm_rawv,
        rawk_cache=spec_state.replayssm_rawk,
        g_cache=spec_state.replayssm_g,
        beta_cache=spec_state.replayssm_beta,
        state_indices=state_batch_indices,
        accept_index=accept_index,
        accept_lens=accept_lens,
        num_tree_nodes=num_tree_nodes,
        mamba_track_indices=mamba_track_indices,
        mamba_track_nodes=mamba_track_nodes,
    )
    for conv_states, intermediate_conv in zip(
        spec_state.conv, spec_state.intermediate_conv_window
    ):
        fused_conv_window_scatter_with_mask(
            conv_states,
            intermediate_conv,
            state_batch_indices,
            last_correct_node_indices,
        )
        if mamba_track_indices is not None and mamba_track_nodes is not None:
            fused_conv_window_scatter_with_mask(
                conv_states,
                intermediate_conv,
                mamba_track_indices,
                mamba_track_nodes,
            )


def preload_gdn_tree_replay() -> None:
    _jit_gdn_tree_replay_module()
