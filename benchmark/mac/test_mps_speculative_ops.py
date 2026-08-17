from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_uniform_func,
)
from sglang.kernels.ops.speculative.eagle import (
    fill_accept_out_cache_loc_func,
    fill_bonus_tokens_func,
)
from sglang.srt.mem_cache.allocation import assign_req_to_token_pool_func
from sglang.srt.speculative.eagle_utils import (
    _sample_mps_chain_target_only,
    _verify_mps_chain,
    build_tree_kernel_efficient,
)
from sglang.srt.speculative.spec_utils import select_top_k_tokens
from sglang.srt.speculative.tree_mask import TreeMaskMode


device = torch.device("mps")

req_to_token = torch.arange(32, dtype=torch.int32, device=device).reshape(2, 16)
req_indices = torch.tensor([1, 0], dtype=torch.int64, device=device)
starts = torch.tensor([3, 5], dtype=torch.int64, device=device)
ends = torch.tensor([5, 8], dtype=torch.int64, device=device)
new_locs = torch.tensor([101, 102, 201, 202, 203], dtype=torch.int64, device=device)
assign_req_to_token_pool_func(
    req_indices, req_to_token, starts, ends, new_locs, batch_size=2
)

uniform_locs = assign_extend_cache_locs_uniform_func(
    req_indices,
    req_to_token,
    starts,
    batch_size=2,
    draft_token_num=2,
    device=device,
)

accept_tokens = torch.tensor(
    [31, 32, 33, 34, 41, 42, 43, 44], dtype=torch.int32, device=device
)
accept_lens = torch.tensor([2, 4], dtype=torch.int32, device=device)
bonus = torch.empty(2, dtype=torch.int32, device=device)
fill_bonus_tokens_func(accept_tokens.reshape(2, 4), accept_lens, bonus, 4, 2)

accept_index = torch.tensor(
    [0, 1, -1, -1, 4, 5, 6, -1], dtype=torch.int32, device=device
)
out_cache_loc = torch.arange(100, 108, dtype=torch.int64, device=device)
accepted_locs = torch.zeros(8, dtype=torch.int64, device=device)
fill_accept_out_cache_loc_func(accept_index, out_cache_loc, accepted_locs, 8)

candidates = torch.tensor(
    [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int64, device=device
)
target_predict = torch.tensor(
    [[11, 99, 12, 13], [21, 22, 23, 24]], dtype=torch.int64, device=device
)
retrieve_index = torch.arange(8, dtype=torch.int64, device=device).reshape(2, 4)
predicts = torch.zeros(8, dtype=torch.int32, device=device)
chain_accept_index = torch.full((2, 4), -1, dtype=torch.int32, device=device)
accepted_drafts = torch.empty(2, dtype=torch.int32, device=device)
_verify_mps_chain(
    predicts,
    chain_accept_index,
    accepted_drafts,
    candidates,
    retrieve_index,
    target_predict,
)

first = select_top_k_tokens(
    0,
    torch.ones((2, 1), dtype=torch.float32, device=device),
    candidates[:, :1],
    torch.zeros((2, 8), dtype=torch.float32, device=device),
    None,
    1,
)
later = select_top_k_tokens(
    1,
    torch.ones((2, 1), dtype=torch.float32, device=device),
    candidates[:, 1:2],
    first[1],
    first[2],
    1,
)

sampling_info = SimpleNamespace(
    max_top_k=5,
    top_ks=torch.tensor([[5]], dtype=torch.int32, device=device),
    top_ps=torch.tensor([[0.9]], dtype=torch.float32, device=device),
    need_top_p_sampling=True,
)
sampled = _sample_mps_chain_target_only(
    torch.softmax(torch.randn((1, 4, 32), device=device), dim=-1),
    sampling_info,
)

tree = build_tree_kernel_efficient(
    candidates[:, 0],
    torch.empty((2, 0), dtype=torch.int64, device=device),
    torch.arange(3, dtype=torch.int64, device=device).expand(2, -1),
    candidates[:, 1:],
    torch.tensor([7, 9], dtype=torch.int64, device=device),
    seq_lens_sum=16,
    topk=1,
    spec_steps=3,
    num_verify_tokens=4,
    tree_mask_mode=TreeMaskMode.FULL_MASK,
)

torch.mps.synchronize()
assert req_to_token.cpu().tolist() == [
    list(range(5)) + [201, 202, 203] + list(range(8, 16)),
    [16, 17, 18, 101, 102] + list(range(21, 32)),
]
assert uniform_locs.cpu().tolist() == [101, 102, 201, 202]
assert bonus.cpu().tolist() == [32, 44]
assert accepted_locs[:5].cpu().tolist() == [100, 101, 104, 105, 106]
assert accepted_drafts.cpu().tolist() == [1, 3]
assert later[0].cpu().tolist() == [11, 21]
assert sampled.shape == (1, 4)
assert chain_accept_index.cpu().tolist() == [[0, 1, -1, -1], [4, 5, 6, 7]]
assert tree[1].cpu().tolist() == [7, 8, 9, 10, 9, 10, 11, 12]
assert tree[2].cpu().tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
assert tree[3].cpu().tolist() == [[1, 2, 3, -1], [1, 2, 3, -1]]
print("MPS speculative primitives: OK")
