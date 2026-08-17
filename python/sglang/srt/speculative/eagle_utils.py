from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch

from sglang.kernels.ops.speculative.spec_tree import (
    sgl_build_tree_kernel_efficient_triton,
    verify_tree_greedy_kernel_triton,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_build_dsv4_verify_bundle,
)
from sglang.srt.mem_cache.allocation import alloc_for_spec_decode
from sglang.srt.mem_cache.allocation_sizing import get_alloc_reserve_per_decode
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.tree_mask import TreeMaskMode, default_tree_mask_mode
from sglang.srt.utils import (
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)
from sglang.srt.utils.async_probe import maybe_detect_oob

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import GrammarMask
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()
_is_xpu = is_xpu()
_is_cpu = is_cpu()

logger = logging.getLogger(__name__)

MAX_SWOR_VERIFY_NODES = 32
SWOR_OVERLAP_TEMPERATURE_SCALES = (0.70, 0.85, 1.00, 1.15, 1.30)
SWOR_OVERLAP_TOP_KS = (4, 8, 12, 16, 20)
_swor_overlap_grid_cache = {}


def _get_swor_overlap_grid(device: torch.device):
    key = str(device)
    cached = _swor_overlap_grid_cache.get(key)
    if cached is None:
        cached = (
            torch.tensor(
                SWOR_OVERLAP_TEMPERATURE_SCALES,
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                SWOR_OVERLAP_TOP_KS,
                dtype=torch.int32,
                device=device,
            ),
        )
        _swor_overlap_grid_cache[key] = cached
    return cached


@dataclass(frozen=True)
class SworTopology:
    """Graph-static ordered-SWOR tree plan.

    Nodes are in final target-verify order; node zero is the bonus/root.  Every
    other node names a fixed child rank of a fixed parent, so each parent's
    visible children are an ordered prefix of the SWOR sequence drawn from its
    stored q.  ``q_sources`` maps every internal target node to the exact draft
    row that generated its children.  ``None`` marks explicit unused leaf
    storage and is never accepted for an internal node.

    Exactness relies on five facts: topology and child counts are fixed before
    draws; visible siblings are an ordered SWOR prefix; the stored q is the q
    used for those draws; generation and verification apply identical R/D
    rejection updates (including uniform D after support exhaustion); and an
    accepted child restarts descent from that child's target p and draft q.
    """

    draft_width: int
    parent_by_node: Tuple[int, ...]
    nodes_by_depth: Tuple[Tuple[int, ...], ...]
    source_ids: Tuple[int, ...]
    visible_local_indices: Tuple[Tuple[int, ...], ...]
    frontier_local_indices: Tuple[Tuple[int, ...], ...]
    q_sources: Tuple[Optional[Tuple[int, int]], ...]
    visible_indices: Tuple[torch.Tensor, ...]
    frontier_indices: Tuple[torch.Tensor, ...]
    frontier_parent_rows: Tuple[torch.Tensor, ...]
    parent_list: torch.Tensor
    selected_indices: torch.Tensor

    @property
    def num_verify_nodes(self) -> int:
        return len(self.parent_by_node)

    @property
    def num_steps(self) -> int:
        return len(self.nodes_by_depth) - 1


def build_swor_topology(
    parent_by_node: Sequence[int], draft_width: int, device: torch.device | str
) -> SworTopology:
    """Validate and bind an arbitrary fixed SWOR tree of at most 32 nodes."""
    parents = tuple(int(parent) for parent in parent_by_node)
    if not 1 < len(parents) <= MAX_SWOR_VERIFY_NODES:
        raise ValueError(
            f"SWOR topology must contain 2..{MAX_SWOR_VERIFY_NODES} verify nodes, "
            f"got {len(parents)}"
        )
    if draft_width < 1 or parents[0] != -1:
        raise ValueError("SWOR topology requires positive draft width and root parent -1")

    depths = [0]
    children: List[List[int]] = [[] for _ in parents]
    for node, parent in enumerate(parents[1:], start=1):
        if parent < 0 or parent >= node:
            raise ValueError(
                f"SWOR topology node {node} has invalid parent {parent}; "
                "parents must precede children"
            )
        children[parent].append(node)
        if len(children[parent]) > draft_width:
            raise ValueError(
                f"SWOR topology node {parent} has {len(children[parent])} children, "
                f"exceeding draft width {draft_width}"
            )
        depths.append(depths[parent] + 1)
    if depths != sorted(depths):
        raise ValueError("SWOR topology nodes must be grouped in nondecreasing depth order")

    max_depth = max(depths)
    nodes_by_depth = tuple(
        tuple(node for node, depth in enumerate(depths) if depth == level)
        for level in range(max_depth + 1)
    )
    internal_by_depth = tuple(
        tuple(node for node in level if children[node]) for level in nodes_by_depth
    )
    for depth, frontier in enumerate(internal_by_depth[1:], start=1):
        if len(frontier) > draft_width:
            raise ValueError(
                f"SWOR topology depth {depth} has {len(frontier)} internal nodes; "
                f"fixed draft width {draft_width} can advance at most {draft_width}"
            )

    source_ids = [-1] * len(parents)
    visible_local_indices: List[Tuple[int, ...]] = []
    frontier_local_indices: List[Tuple[int, ...]] = []
    for depth in range(1, max_depth + 1):
        parent_frontier = internal_by_depth[depth - 1]
        parent_slots = {node: slot for slot, node in enumerate(parent_frontier)}
        local_indices = []
        for node in nodes_by_depth[depth]:
            parent = parents[node]
            sibling_rank = children[parent].index(node)
            local = sibling_rank if depth == 1 else parent_slots[parent] * draft_width + sibling_rank
            local_indices.append(local)
            block_offset = 0 if depth == 1 else draft_width + (depth - 2) * draft_width**2
            source_ids[node] = block_offset + local
        visible_local_indices.append(tuple(local_indices))

        frontier = internal_by_depth[depth]
        if frontier:
            frontier_locals = [local_indices[nodes_by_depth[depth].index(node)] for node in frontier]
            frontier_locals.extend([frontier_locals[0]] * (draft_width - len(frontier_locals)))
        else:
            frontier_locals = [local_indices[0]] * draft_width
        frontier_local_indices.append(tuple(frontier_locals))

    q_sources: List[Optional[Tuple[int, int]]] = [None] * len(parents)
    q_sources[0] = (0, 0)
    for depth in range(1, max_depth):
        for row, node in enumerate(internal_by_depth[depth]):
            q_sources[node] = (depth, row)
    for node, node_children in enumerate(children):
        if node_children and q_sources[node] is None:
            raise ValueError(f"SWOR topology internal node {node} has no draft q row")

    # Legacy build-tree metadata addresses candidate-universe IDs in groups of
    # draft_width. These tensors are setup-time constants and are only expanded
    # during replay, so topology does not allocate in the CUDA graph pool.
    parent_ids = [-1]
    for depth in range(1, max_depth):
        frontier_ids = [source_ids[node] for node in internal_by_depth[depth]]
        frontier_ids.extend([frontier_ids[0]] * (draft_width - len(frontier_ids)))
        parent_ids.extend(frontier_ids)

    return SworTopology(
        draft_width=draft_width,
        parent_by_node=parents,
        nodes_by_depth=nodes_by_depth,
        source_ids=tuple(source_ids),
        visible_local_indices=tuple(visible_local_indices),
        frontier_local_indices=tuple(frontier_local_indices),
        q_sources=tuple(q_sources),
        visible_indices=tuple(
            torch.tensor(indices, dtype=torch.long, device=device)
            for indices in visible_local_indices
        ),
        frontier_indices=tuple(
            torch.tensor(indices, dtype=torch.long, device=device)
            for indices in frontier_local_indices
        ),
        frontier_parent_rows=tuple(
            torch.tensor(
                [index // draft_width for index in indices],
                dtype=torch.long,
                device=device,
            )
            for indices in frontier_local_indices
        ),
        parent_list=torch.tensor(parent_ids, dtype=torch.long, device=device).unsqueeze(0),
        selected_indices=torch.tensor(source_ids[1:], dtype=torch.long, device=device).unsqueeze(0),
    )


def default_swor_topology(device: torch.device | str) -> SworTopology:
    """The initial 12-node 4/4/2/1 production SWOR topology."""
    return build_swor_topology(
        (-1, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 9), draft_width=4, device=device
    )


def parse_swor_topology(
    topology_json: str, draft_width: int, device: torch.device | str
) -> SworTopology:
    """Parse the launcher-safe JSON representation of a fixed topology."""
    try:
        parents = json.loads(topology_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid SWOR topology JSON: {exc.msg}") from exc
    if not isinstance(parents, list) or any(
        isinstance(parent, bool) or not isinstance(parent, int) for parent in parents
    ):
        raise ValueError("SWOR topology JSON must be an array of integer parent IDs")
    return build_swor_topology(parents, draft_width=draft_width, device=device)


def select_swor_topology_step(
    topology: SworTopology,
    step: int,
    sampled_tokens: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Gather one fixed visible level and its padded next-forward frontier."""
    width = topology.draft_width
    bs = sampled_tokens.shape[0] if step == 0 else sampled_tokens.shape[0] // width
    flat = sampled_tokens.reshape(bs, -1)
    visible = flat.index_select(1, topology.visible_indices[step])
    input_ids = flat.index_select(1, topology.frontier_indices[step]).flatten()
    if hidden_states is not None:
        if step == 0:
            hidden_states = hidden_states.repeat_interleave(width, dim=0)
        else:
            hidden_states = hidden_states.reshape(bs, width, -1).index_select(
                1, topology.frontier_parent_rows[step]
            ).reshape(bs * width, -1)
    return input_ids, hidden_states, visible


def organize_swor_draft_results(
    topology: SworTopology,
    token_blocks: Sequence[torch.Tensor],
    q_blocks: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble fixed topology outputs without score-dependent selection."""
    if len(token_blocks) != topology.num_steps or len(q_blocks) != topology.num_steps:
        raise ValueError(
            f"SWOR topology expects {topology.num_steps} token/q blocks, got "
            f"{len(token_blocks)}/{len(q_blocks)}"
        )
    bs = token_blocks[0].shape[0]
    draft_tokens = torch.cat(tuple(token_blocks), dim=1)
    parent_list = topology.parent_list.expand(bs, -1)
    selected_indices = topology.selected_indices.expand(bs, -1)

    q_rows = []
    unused_leaf = q_blocks[0][:, :1]
    for source in topology.q_sources:
        q_rows.append(unused_leaf if source is None else q_blocks[source[0]][:, source[1] : source[1] + 1])
    draft_probs = torch.cat(q_rows, dim=1)
    return parent_list, selected_indices, draft_tokens, draft_probs

_use_triton_spec_fallback = False
_windows_top_k_renorm_prob = None
_windows_top_p_renorm_prob = None
if _is_cuda or _is_hip or _is_musa:
    try:
        from sgl_kernel import (
            build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
        )
    except ModuleNotFoundError:
        if not (_is_cuda and sys.platform == "win32"):
            raise
        _use_triton_spec_fallback = True
        try:
            # FlashInfer's CUDA sampling module is already part of the native
            # Windows serving stack. Resolve once here so the verify loop pays
            # no repeated Python import dispatch.
            from flashinfer.sampling import (
                top_k_renorm_prob as _windows_top_k_renorm_prob,
            )
            from flashinfer.sampling import (
                top_p_renorm_prob as _windows_top_p_renorm_prob,
            )
        except ImportError:
            from sglang.kernels.ops.sampling.renorm_triton import (
                top_k_renorm_probs_triton as _windows_top_k_renorm_prob,
            )
            from sglang.kernels.ops.sampling.renorm_triton import (
                top_p_renorm_probs_triton as _windows_top_p_renorm_prob,
            )
elif _is_cpu:
    from sgl_kernel import (
        build_tree_kernel_efficient_cpu as sgl_build_tree_kernel_efficient_cpu,
    )
    from sgl_kernel import verify_tree_greedy_cpu as sgl_verify_tree_greedy_cpu


def per_step_draft_out_cache_loc(
    out_cache_loc: torch.Tensor,
    batch_size: int,
    topk: int,
    num_steps: int,
) -> torch.Tensor:
    """Per-step slice of the multi-step EAGLE draft out_cache_loc buffer.

    Single source of truth for the layout shared by EagleWorkerV2.draft_forward
    (per-step write target) and DeepseekV4AttnBackend (per-step compression
    write target baked into metadata).
    """
    expected = batch_size * topk * num_steps
    assert out_cache_loc.shape[0] == expected, (
        f"out_cache_loc.shape[0]={out_cache_loc.shape[0]} != "
        f"batch_size * topk * num_steps = {batch_size}*{topk}*{num_steps}={expected}"
    )
    return (
        out_cache_loc.view(batch_size, topk, num_steps)
        .permute(2, 0, 1)
        .reshape(num_steps, -1)
    )


def _eagle_prefill_tail_tokens(
    batch: ScheduleBatch, next_token_ids: torch.Tensor
) -> torch.Tensor:
    """Per-seq tail token for EAGLE prefill rotation; uses next prompt token for
    non-final chunks (chunked-prefill chain consistency, see PR #26329)."""
    tail_tokens = next_token_ids.to(batch.input_ids.dtype)
    next_prompt_token = batch.chunked_req_next_prompt_token
    if next_prompt_token is not None:
        for i, r in enumerate(batch.reqs):
            if r is batch.chunked_req:
                tail_tokens = tail_tokens.clone()
                # Keep the scalar as a kernel argument. Assigning a Python scalar
                # through scalar indexing issues a pageable H2D copy and
                # synchronizes the current CUDA stream before draft extend.
                tail_tokens[i : i + 1].fill_(next_prompt_token)
                break
    return tail_tokens


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    # b, n, topk; n = 1 + (num_steps-1) * topk
    score_list = torch.cat(score_list, dim=1).flatten(1)
    # b, (topk + (num_steps-1) * topk)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    maybe_detect_oob(
        top_scores_index,
        0,
        ss_token_list.shape[1],
        "organize_draft_results: top_scores_index OOB for gather on ss_token_list",
    )
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(
            batch_size, 0, dtype=torch.long, device=parents_list[0].device
        )

    return parent_list, top_scores_index, draft_tokens


def build_tree_kernel_efficient(
    bonus_tokens: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    fill_prefix_mask: bool = True,
):
    if (
        _is_cuda
        and sys.platform == "win32"
        and _use_triton_spec_fallback
        and topk == 1
        and num_verify_tokens == spec_steps + 1
        and tree_mask_mode == TreeMaskMode.QLEN_ONLY
        and tree_mask_buf is not None
        and bonus_tokens.dtype == torch.long
        and draft_tokens.dtype == torch.long
        and seq_lens.dtype == torch.long
    ):
        # Native Windows has no sgl_kernel wheel.  For the production top-k1
        # chain, replace cat + fill + cumsum + general Triton tree construction
        # with one shape-specialized C++/CUDA JIT launch.  Keep per-cycle output
        # allocations: the verifier consumes these tensors asynchronously.
        from sglang.kernels.ops.speculative.chain_metadata import (
            build_chain_metadata,
        )

        positions, retrieve_buf, draft_tokens = build_chain_metadata(
            bonus_tokens, draft_tokens, seq_lens, tree_mask_buf
        )
        retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
        return (
            tree_mask_buf,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        )

    draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            if fill_prefix_mask:
                tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            # Only the [0, seq_len) prefix columns depend on this fill; the
            # kernel below writes every tree cell itself. Skip the (up to
            # 100s of MB) per-step memset when nothing reads the mask.
            if fill_prefix_mask:
                tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        mask_shape = (
            seq_lens_sum * num_verify_tokens
            + num_verify_tokens * num_verify_tokens * bs,
        )
        # Same reasoning as the preallocated branch above.
        tree_mask = (
            torch.full(mask_shape, True, dtype=torch.bool, device=device)
            if fill_prefix_mask
            else torch.empty(mask_shape, dtype=torch.bool, device=device)
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrieve_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    positions = torch.empty((bs * num_verify_tokens,), device=device, dtype=torch.long)

    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    elif _is_xpu or _use_triton_spec_fallback:
        sgl_build_tree_kernel_triton(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    elif _is_cpu:
        sgl_build_tree_kernel_efficient_cpu(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    )


def sgl_build_tree_kernel_triton(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
):
    """Triton-based implementation."""
    # TODO: Add support for QLEN_ONLY_BITPACKING mode
    if tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        raise NotImplementedError(
            "QLEN_ONLY_BITPACKING is not supported in Triton implementation"
        )

    batch_size = verified_seq_len.shape[0]
    seq_len_prefix_sum = torch.cumsum(verified_seq_len, dim=0) - verified_seq_len

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    sgl_build_tree_kernel_efficient_triton[grid](
        parent_list,
        selected_index,
        verified_seq_len,
        seq_len_prefix_sum,
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=int(tree_mask_mode),
        batch_size=batch_size,
        parent_list_stride=(
            parent_list.stride(0) if parent_list.dim() > 1 else parent_list.shape[0]
        ),
        selected_index_stride=selected_index.stride(0),
    )


def verify_tree_greedy_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    """Triton-based implementation."""
    batch_size = candidates.shape[0]
    num_speculative_tokens = accept_index.shape[1]
    num_draft_tokens = candidates.shape[1]

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    verify_tree_greedy_kernel_triton[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        batch_size=batch_size,
        num_speculative_tokens=num_speculative_tokens,
        num_draft_tokens=num_draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if (_is_cuda or _is_hip or _is_musa) and not _use_triton_spec_fallback:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_cpu:
        sgl_verify_tree_greedy_cpu(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match the CUDA op schema, so
            # the CPU/CUDA call sites stay grep-symmetric.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    elif _is_xpu or _use_triton_spec_fallback:
        verify_tree_greedy_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num


def get_draft_input_from_target_hidden_dim(model_runner: ModelRunner) -> int:
    """Width of the target hidden states fed into the draft model.

    This is the single source of truth and is derived entirely from config: for
    EAGLE3 aux mode the draft consumes `num_aux` concatenated target layers
    (each `target_hidden_size` wide); every other arch consumes the per-layer
    `spec_hidden_size`.

    Do NOT read this off a draft projection's `in_features` (e.g. an `fc`
    layer): that width is arch-specific.

    Note: read entirely from the *draft* `model_runner`'s config. The non-aux
    branch assumes the draft's `spec_hidden_size` equals the target hidden width
    fed to the draft (true for standard EAGLE, where the draft mirrors the
    target hidden size); aux mode reads the explicit `target_hidden_size`.
    """
    model_config = model_runner.model_config
    hf_config = model_config.hf_config
    eagle_config = getattr(hf_config, "eagle_config", None) or {}
    get_eagle_config = (
        eagle_config.get
        if isinstance(eagle_config, dict)
        else lambda key, default=None: getattr(eagle_config, key, default)
    )
    use_aux = get_eagle_config("use_aux_hidden_state", True)
    spec_algorithm = model_runner.spec_algorithm

    if not (spec_algorithm is not None and spec_algorithm.is_eagle3() and use_aux):
        return model_config.spec_hidden_size

    target_hidden = getattr(hf_config, "target_hidden_size", None)
    if target_hidden is None:
        target_hidden = model_config.hidden_size
    num_aux = getattr(hf_config, "num_aux_hidden_states", None)
    if num_aux is None:
        layer_ids = get_eagle_config("eagle_aux_hidden_state_layer_ids", None)
        if layer_ids is None:
            layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        num_aux = len(layer_ids) if layer_ids else 3
    return target_hidden * num_aux


def get_draft_recurrent_hidden_state_spec(
    model_runner: ModelRunner,
) -> tuple[Optional[int], Optional[torch.dtype]]:
    """Return hidden_states width/dtype carried between draft decode steps."""
    if model_runner.spec_algorithm.is_standalone():
        return None, None
    return model_runner.model_config.spec_hidden_size, model_runner.model_config.dtype


def eagle_prepare_for_verify(
    verify_input: EagleVerifyInput,
    req_to_token_pool: ReqToTokenPool,
    batch: ScheduleBatch,
    target_worker: TpModelWorker,
):
    from sglang.kernels.ops.speculative.cache_locs import (
        assign_extend_cache_locs_uniform_func,
    )
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify

    if not batch.forward_mode.is_idle():
        # Assign cache locations
        bs = len(batch.req_pool_indices)
        batch.input_ids = verify_input.draft_token
        maybe_detect_oob(
            batch.input_ids,
            0,
            batch.model_config.vocab_size,
            "v2 prepare_for_verify input_ids",
        )
        device = batch.device
        # Uniform variant: end offsets (= start + draft_token_num) are computed
        # inside the kernel, keeping the eager `seq_lens + N` add off the host
        # critical path (bs=1 MTP inter-phase seam).
        batch.out_cache_loc = assign_extend_cache_locs_uniform_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            batch_size=bs,
            draft_token_num=verify_input.draft_token_num,
            device=device,
        )

        batch.out_cache_loc_dsv4 = maybe_build_dsv4_verify_bundle(
            batch, verify_input.draft_token_num
        )

        prepare_mamba_track_for_verify(batch)

        # TBO's split_spec_info reads these; no-verify-sync leaves both None.
        verify_input.seq_lens_cpu = batch.seq_lens_cpu
        verify_input.seq_lens_sum = (
            int(batch.seq_lens_cpu.sum()) if batch.seq_lens_cpu is not None else None
        )

    # Get a forward batch
    batch.forward_mode = (
        ForwardMode.IDLE if batch.forward_mode.is_idle() else ForwardMode.TARGET_VERIFY
    )
    capture_mode = (
        CaptureHiddenMode.NULL
        if target_worker.model_runner.spec_algorithm.is_standalone()
        else CaptureHiddenMode.FULL
    )
    verify_forward_batch = ForwardBatch.init_new(
        batch,
        target_worker.model_runner,
        capture_hidden_mode=capture_mode,
        return_hidden_states_before_norm=False,
    )

    # Run attention backend plan and cuda graph preparation
    can_run_cuda_graph = bool(
        target_worker.model_runner.decode_cuda_graph_runner
        and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
            verify_forward_batch
        )
    )
    if can_run_cuda_graph:
        target_worker.model_runner.decode_cuda_graph_runner.load_batch(
            verify_forward_batch
        )
        verify_forward_batch.mark_forward_metadata_ready()
    # Non-cuda-graph: defer init to forward_extend, which runs after
    # `_forward_raw -> prepare_mlp_sync_batch` pads the batch. Initing
    # here would use pre-pad shapes and trip DSv4 indexer shape match.

    return verify_forward_batch, can_run_cuda_graph


def _seeded_verify_coins(
    *,
    sampling_seed: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive deterministic verify-side coins from per-request sampling seeds.

    Mirrors the main seeded-sampling path: murmur_hash32(seed, seq_lens,
    column) mapped to [0, 1). Columns [0, draft_token_num) drive the
    per-draft rejection coins; column draft_token_num drives the final
    fallback-sampling coin.

    Scope: this seeds only the verify-side RNG. With rejection sampling the
    draft workers still pick candidates via unseeded multinomial
    (fast_sample in eagle_worker_v2), so that mode stays non-deterministic
    until the draft RNG is seeded in a follow-up; top-k/greedy draft
    selection is already deterministic.
    """
    from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32

    cols = torch.arange(draft_token_num + 1, device=device, dtype=torch.int64)
    hashed = murmur_hash32(
        sampling_seed.to(torch.uint64), seq_lens.to(torch.uint64), cols
    )
    uniforms = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max
    # The float32 cast rounds the top 129 uint32 hashes to exactly 1.0, but
    # the sampling kernels expect half-open [0, 1) coins: a 1.0 coin walks
    # past the last CDF bucket and can return a zero-probability token.
    # Clamp to the largest float32 below one; every other coin value is
    # untouched, so previously verified bitwise baselines stay intact.
    max_coin = 1.0 - 2**-24
    coins = (
        uniforms[:, :draft_token_num].to(torch.float32).clamp_(max=max_coin)
    ).contiguous()
    coins_for_final_sampling = (
        uniforms[:, draft_token_num].to(torch.float32).clamp_(max=max_coin)
    ).contiguous()
    return coins, coins_for_final_sampling


def _verify_coins(
    *,
    sampling_info: SamplingBatchInfo,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    candidates: torch.Tensor,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rejection and final-sampling coins for verify: deterministic seeded
    coins when sampling_seed is set (see _seeded_verify_coins), torch.rand
    otherwise.
    """
    if sampling_info.sampling_seed is not None:
        return _seeded_verify_coins(
            sampling_seed=sampling_info.sampling_seed,
            seq_lens=seq_lens,
            draft_token_num=draft_token_num,
            device=device,
        )
    # coins for rejection sampling
    coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
    # coins for final sampling
    coins_for_final_sampling = torch.rand(
        (candidates.shape[0],), dtype=torch.float32, device=device
    )
    return coins, coins_for_final_sampling


def eagle_sample(
    verify_input: EagleVerifyInput,
    batch: ScheduleBatch,
    logits_output: LogitsProcessorOutput,
    grammar_mask: Optional[GrammarMask] = None,
):
    """
    Verify and find accepted tokens based on logits output and batch
    (which contains spec decoding information).
    """
    import torch.nn.functional as F

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.dp_attention import (
        is_dp_attention_enabled,
    )
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )
    from sglang.srt.speculative.spec_utils import (
        SIMULATE_ACC_LEN,
        SIMULATE_ACC_TOKEN_MODE,
        generate_simulated_accept_index,
    )
    from sglang.srt.utils.async_probe import maybe_detect_nan, sanitize_nan_logits

    device = batch.device
    if batch.forward_mode.is_idle():
        predict = torch.empty(0, dtype=torch.int32, device=device)
        num_correct_drafts = torch.empty(0, dtype=torch.int32, device=device)
        accept_index = torch.empty(0, dtype=torch.int32, device=device)
        return predict, num_correct_drafts, accept_index

    bs = len(batch.seq_lens)
    sampling_info = batch.sampling_info
    next_token_logits = logits_output.next_token_logits

    sanitize_nan_logits(next_token_logits, "verify: target model logits")

    # Apply penalty
    # This is a relaxed version of penalties for speculative decoding.
    if sampling_info.acc_additive_penalties is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.acc_additive_penalties,
                verify_input.draft_token_num,
                dim=0,
            )
        )
    if sampling_info.acc_scaling_penalties is not None:
        apply_scaling_penalties(
            next_token_logits,
            torch.repeat_interleave(
                sampling_info.acc_scaling_penalties, verify_input.draft_token_num, dim=0
            ),
        )
    if sampling_info.logit_bias is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.logit_bias, verify_input.draft_token_num, dim=0
            )
        )

    # Apply grammar mask if provided
    if grammar_mask is not None:
        grammar_mask.apply(next_token_logits)

    candidates = verify_input.draft_token.reshape(bs, verify_input.draft_token_num)
    predict_shape = list(next_token_logits.shape)[:-1]
    predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
    accept_index = torch.full(
        (bs, verify_input.max_tree_depth), -1, dtype=torch.int32, device=device
    )
    num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=device)

    # Sample tokens
    target_predict = None
    if sampling_info.is_all_greedy or _is_cpu or _is_npu or _is_hip or _is_xpu:
        target_predict = torch.argmax(next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, verify_input.draft_token_num)
        predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
            predicts=predict,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=num_correct_drafts,  # mutable
            candidates=candidates,
            retrieve_index=verify_input.retrieve_index,
            retrieve_next_token=verify_input.retrieve_next_token,
            retrieve_next_sibling=verify_input.retrieve_next_sibling,
            target_predict=target_predict,
            topk=verify_input.tree_topk,
        )
    else:
        from sglang.kernels.ops.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )

        use_rejection_sampling = get_spec().speculative_use_rejection_sampling
        use_tree_swor = (
            getattr(get_spec(), "speculative_tree_sampling_mode", "target_only")
            == "swor"
        )
        if _use_triton_spec_fallback:
            # Native Windows lacks the monolithic sgl_kernel wheel.  Its
            # FlashInfer 0.6.17 port already ships the CUDA renorm kernels used
            # by the ordinary Sampler, so speculative verification uses them as
            # well.  The old Triton fallback sorted the full 248K vocabulary
            # once for top-k and again for top-p.  Target-only trees use the
            # selective native CUDA verifier resolved below.
            top_k_renorm_prob = _windows_top_k_renorm_prob
            top_p_renorm_prob = _windows_top_p_renorm_prob

            if use_rejection_sampling:
                tree_speculative_sampling_target_only = None
            else:
                # Native Windows has no monolithic sgl_kernel wheel.  Its
                # selective CUDA JIT verifier implements the exact target-only
                # tree rule directly and carries only terminal sibling IDs,
                # avoiding the old vocabulary-sized zero draft tensor.
                from sglang.kernels.ops.speculative.tree_sampling import (
                    exact_tree_speculative_sampling,
                    exact_tree_swor_sampling,
                )

                tree_speculative_sampling_target_only = (
                    exact_tree_swor_sampling
                    if getattr(
                        get_spec(), "speculative_tree_sampling_mode", "target_only"
                    )
                    == "swor"
                    else exact_tree_speculative_sampling
                )
        else:
            from sgl_kernel import (
                top_k_renorm_prob,
                top_p_renorm_prob,
                tree_speculative_sampling_target_only,
            )

        # Apply temperature and get target probs
        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, verify_input.draft_token_num, dim=0
        )  # (bs * num_draft_tokens, 1)

        target_probs = F.softmax(
            next_token_logits / expanded_temperature, dim=-1
        )  # (bs * num_draft_tokens, vocab_size)
        maybe_detect_nan(target_probs, "v2 verify: target_probs after softmax")
        if sampling_info.need_top_k_sampling:
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, verify_input.draft_token_num, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            maybe_detect_nan(target_probs, "v2 verify: target_probs after top_k_renorm")
        if sampling_info.need_top_p_sampling:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, verify_input.draft_token_num, dim=0
                ),
            )
            maybe_detect_nan(target_probs, "v2 verify: target_probs after top_p_renorm")
        target_probs = target_probs.reshape(bs, verify_input.draft_token_num, -1)
        draft_probs = (
            verify_input.draft_probs
            if use_rejection_sampling or use_tree_swor
            else (
                None
                if _use_triton_spec_fallback
                else torch.zeros_like(target_probs)
            )
        )
        # Defense-in-depth behind the spec_hook startup allowlist: validate the
        # actual kernel inputs (catches draft_probs plumbing regressions or a
        # startup guard bypassed by a worker subclass) before the Triton kernel.
        if (use_rejection_sampling or use_tree_swor) and (
            draft_probs is None or draft_probs.shape[-1] != target_probs.shape[-1]
        ):
            raise ValueError(
                "Exact p/q verification requires a target-vocab draft proposal "
                "distribution; the current speculative algorithm/draft worker "
                "does not produce one (draft_probs missing or vocab-mismatched)."
            )

        if use_tree_swor and getattr(
            get_spec(), "speculative_swor_collect_overlap_stats", False
        ):
            from sglang.kernels.ops.speculative.tree_sampling import (
                swor_proposal_overlap_metrics,
            )

            temperature_scales, overlap_top_ks = _get_swor_overlap_grid(
                target_probs.device
            )
            verify_input.swor_overlap_metrics = swor_proposal_overlap_metrics(
                target_probs,
                draft_probs,
                temperature_scales,
                overlap_top_ks,
            )

        coins, coins_for_final_sampling = _verify_coins(
            sampling_info=sampling_info,
            seq_lens=batch.seq_lens,
            draft_token_num=verify_input.draft_token_num,
            candidates=candidates,
            device=device,
        )

        sampling_fn = (
            chain_speculative_sampling_triton
            if use_rejection_sampling
            else tree_speculative_sampling_target_only
        )
        sampling_fn(
            predicts=predict,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=num_correct_drafts,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=verify_input.retrieve_index,
            retrive_next_token=verify_input.retrieve_next_token,
            retrive_next_sibling=verify_input.retrieve_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=get_spec().speculative_accept_threshold_single,
            threshold_acc=get_spec().speculative_accept_threshold_acc,
            deterministic=True,
        )

    # Sync the verify decision across TP ranks: small per-rank differences in the
    # tensors feeding this point can make one rank accept a different number of
    # drafts, which desynchronizes the committed seq_lens and deadlocks the next
    # TP collective. Broadcast from rank 0 to ensure consistency.
    tp_group = (
        get_parallel().attn_tp_group if is_dp_attention_enabled() else get_tp_group()
    )
    if tp_group.world_size > 1:
        tp_group.broadcast(predict, src=0)
        tp_group.broadcast(accept_index, src=0)
        tp_group.broadcast(num_correct_drafts, src=0)

    if SIMULATE_ACC_LEN > 0:
        # Do simulation. The helper builds (and returns) a replacement
        # accept_index of width spec_steps + 1, so pass max_tree_depth - 1
        # to keep the simulated width identical to the real one.
        if SIMULATE_ACC_TOKEN_MODE not in ("fixed", "real-draft-token"):
            raise ValueError(
                "Invalid SGLANG_SIMULATE_ACC_TOKEN_MODE "
                f"{SIMULATE_ACC_TOKEN_MODE!r}; expected 'fixed' or "
                "'real-draft-token'."
            )

        if SIMULATE_ACC_TOKEN_MODE == "real-draft-token":
            if verify_input.tree_topk != 1:
                raise ValueError(
                    "SGLANG_SIMULATE_ACC_LEN with real draft tokens currently "
                    "requires speculative_eagle_topk=1."
                )

            # Use target argmax as the synthetic bonus for non-greedy requests.
            if target_predict is None:
                target_predict = torch.argmax(next_token_logits, dim=-1).reshape(
                    bs, verify_input.draft_token_num
                )
        accept_index = generate_simulated_accept_index(
            accept_index=accept_index,
            predict=predict,  # mutable
            num_correct_drafts=num_correct_drafts,  # mutable
            candidates=candidates,
            target_predict=target_predict,
            simulate_acc_len=SIMULATE_ACC_LEN,
            simulate_acc_token_mode=SIMULATE_ACC_TOKEN_MODE,
            bs=bs,
            spec_steps=verify_input.max_tree_depth - 1,
        )

    # `num_correct_drafts` stays drafts-only inside this function; the returned
    # tensor includes the trailing/bonus token via out-of-place +1 so the
    # name no longer flips semantics mid-function (naming doc C2).
    return predict, num_correct_drafts + 1, accept_index


def eagle_prepare_for_decode(batch: ScheduleBatch):
    batch.maybe_evict_swa()

    bs = batch.batch_size()

    # Accumulate penalty
    # This is a relaxed version of penalties for speculative decoding.
    if batch.sampling_info.penalizer_orchestrator.is_required:
        batch.cumulate_penalty_output_tokens()

    page_size = batch.token_to_kv_pool_allocator.page_size
    double_alloc = get_alloc_reserve_per_decode()

    cur_kv_lens = [0] * bs
    nxt_kv_lens = [0] * bs
    num_needed_tokens = 0
    for i, r in enumerate(batch.reqs):
        cur = r.kv.kv_allocated_len
        # max(cur, ...) clamps so adaptive downswitch cannot make nxt < cur.
        # kv_committed_len is honest (bonus committed in resolve, not here),
        # so it lags batch.seq_lens by ~1 verify in overlap; 2*alloc absorbs.
        # Whole-page accounting: the paged allocator hands out full pages, so
        # round nxt up to the page boundary or the unaligned tail is allocated
        # but never recorded — a stranded-tail leak at page_size > 1.
        nxt = max(
            cur,
            (r.kv_committed_len + double_alloc + page_size - 1)
            // page_size
            * page_size,
        )
        cur_kv_lens[i] = cur
        nxt_kv_lens[i] = nxt
        num_needed_tokens += nxt - cur
        r.decode_batch_idx += 1

    cur_kv_lens_cpu = torch.tensor(cur_kv_lens, dtype=torch.int32, device="cpu")
    nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens, dtype=torch.int32, device="cpu")

    # Fail fast if the page>1 + topk>1 draft over-allocation
    # (get_alloc_reserve_per_decode) outgrows the req_to_token row: the write below
    # would OOB and free would leak KV. The row is widened to hold it in _init_pools
    # (PR #26972); fail here with a clear error, not on a later cryptic CUDA assert.

    if page_size > 1 and (get_spec().speculative_eagle_topk or 1) > 1:
        max_alloc_len = int(nxt_kv_lens_cpu.max())
        row_width = batch.req_to_token_pool.req_to_token.shape[1]
        assert max_alloc_len <= row_width, (
            f"spec v2 page>1 topk>1 draft over-allocation ({max_alloc_len}) exceeds "
            f"req_to_token row width ({row_width}); page_size={page_size}. Widen the "
            f"row to hold committed + get_alloc_reserve_per_decode (PR #26972)."
        )

    # non_blocking H2D: a blocking .to() syncs the schedule stream, which the WAR
    # barrier has chained to the prev forward -> host stalls a full forward.
    cur_kv_lens_device = cur_kv_lens_cpu.to(device=batch.device, non_blocking=True)
    nxt_kv_lens_device = nxt_kv_lens_cpu.to(device=batch.device, non_blocking=True)
    tree_cache = batch.tree_cache
    req_to_token_pool = batch.req_to_token_pool
    req_pool_indices = batch.req_pool_indices
    reqs = batch.reqs
    cur_kv_lens = cur_kv_lens_device
    nxt_kv_lens = nxt_kv_lens_device
    alloc_for_spec_decode(
        tree_cache,
        req_to_token_pool,
        reqs=reqs,
        req_pool_indices=req_pool_indices,
        cur_kv_lens=cur_kv_lens,
        cur_kv_lens_cpu=cur_kv_lens_cpu,
        nxt_kv_lens=nxt_kv_lens,
        nxt_kv_lens_cpu=nxt_kv_lens_cpu,
        num_needed_tokens=num_needed_tokens,
        batch=batch,
    )
