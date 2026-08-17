from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.mps.ops import (
    causal_conv1d_decode,
    gdn_decode,
)
from sglang.srt.layers.attention.linear.gdn_backend import causal_conv1d_update
from sglang.srt.layers.attention.linear.kernels.gdn_mps import MpsGDNKernel
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    scatter_mamba_states_after_mtp_verify,
)


torch.manual_seed(7)
device = torch.device("mps")
batch_size = 2
steps = 4
slots = 4
key_heads = 1
value_heads = 3
key_dim = 128
value_dim = 4

q = torch.randn(1, batch_size * steps, key_heads, key_dim, device=device)
k = torch.randn_like(q)
v = torch.randn(1, batch_size * steps, value_heads, value_dim, device=device)
a = torch.randn(batch_size * steps, value_heads, device=device)
b = torch.randn_like(a)
A_log = torch.randn(value_heads, device=device)
dt_bias = torch.randn(value_heads, device=device)
state = torch.randn(slots, value_heads, value_dim, key_dim, device=device)
state_before = state.clone()
cache_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
query_start_loc = torch.tensor([0, steps, 2 * steps], dtype=torch.int32, device=device)
scratch_rows = torch.tensor([0, 1], dtype=torch.int32, device=device)
intermediate = torch.zeros(
    batch_size, steps, value_heads, value_dim, key_dim, device=device
)

output = MpsGDNKernel().target_verify(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    ssm_states=state,
    cache_indices=cache_indices,
    query_start_loc=query_start_loc,
    intermediate_states_buffer=intermediate,
    intermediate_state_indices=scratch_rows,
    cache_steps=steps,
    retrieve_parent_token=None,
)

working = state_before.index_select(0, cache_indices.long()).clone()
local_slots = torch.arange(batch_size, dtype=torch.int32, device=device)
reference_outputs = []
reference_intermediate = torch.empty_like(intermediate)
q_view = q.reshape(1, batch_size, steps, key_heads, key_dim)
k_view = k.reshape_as(q_view)
v_view = v.reshape(1, batch_size, steps, value_heads, value_dim)
a_view = a.reshape(batch_size, steps, value_heads)
b_view = b.reshape_as(a_view)
for step in range(steps):
    reference_outputs.append(
        gdn_decode(
            q_view[:, :, step],
            k_view[:, :, step],
            v_view[:, :, step],
            a_view[:, step],
            b_view[:, step],
            A_log,
            dt_bias,
            working,
            local_slots,
        ).squeeze(0)
    )
    reference_intermediate[:, step].copy_(working)
reference_output = torch.stack(reference_outputs, dim=1).reshape_as(output)

channels = 48
conv_x = torch.randn(batch_size, channels, steps, device=device)
conv_weight = torch.randn(channels, 4, device=device)
conv_state = torch.randn(slots, channels, 3, device=device)
conv_state_before = conv_state.clone()
conv_intermediate = torch.empty(batch_size, steps, channels, 3, device=device)
conv_output = causal_conv1d_update(
    conv_x,
    conv_state,
    conv_weight,
    None,
    "silu",
    conv_state_indices=cache_indices,
    intermediate_conv_window=conv_intermediate,
    intermediate_state_indices=scratch_rows,
    retrieve_parent_token=None,
)
conv_working = conv_state_before.index_select(0, cache_indices.long()).clone()
conv_reference = []
conv_reference_intermediate = torch.empty_like(conv_intermediate)
for step in range(steps):
    conv_reference.append(
        causal_conv1d_decode(
            conv_x[:, :, step], conv_weight, conv_working, local_slots
        )
    )
    conv_reference_intermediate[:, step].copy_(conv_working)
conv_reference = torch.stack(conv_reference, dim=-1)

full_intermediate = torch.stack((intermediate, intermediate + 1), dim=0)
full_conv_intermediate = torch.stack(
    (conv_intermediate, conv_intermediate + 1), dim=0
)
cache = SimpleNamespace(
    temporal=[
        torch.zeros(slots, value_heads, value_dim, key_dim, device=device),
        torch.zeros(slots, value_heads, value_dim, key_dim, device=device),
    ],
    intermediate_ssm=full_intermediate,
    conv=[torch.zeros(2, slots, channels, 3, device=device)],
    intermediate_conv_window=[full_conv_intermediate],
)
accepted_steps = torch.tensor([1, 3], dtype=torch.int32, device=device)
scatter_mamba_states_after_mtp_verify(
    cache, cache_indices, accepted_steps, None, None
)

torch.mps.synchronize()
torch.testing.assert_close(state.cpu(), state_before.cpu(), rtol=0, atol=0)
torch.testing.assert_close(output.cpu(), reference_output.cpu(), rtol=0, atol=0)
torch.testing.assert_close(
    intermediate.cpu(), reference_intermediate.cpu(), rtol=0, atol=0
)
torch.testing.assert_close(conv_state.cpu(), conv_state_before.cpu(), rtol=0, atol=0)
torch.testing.assert_close(conv_output.cpu(), conv_reference.cpu(), rtol=0, atol=0)
torch.testing.assert_close(
    conv_intermediate.cpu(), conv_reference_intermediate.cpu(), rtol=0, atol=0
)
for layer in range(2):
    for request in range(batch_size):
        dst = int(cache_indices[request].cpu())
        step = int(accepted_steps[request].cpu())
        torch.testing.assert_close(
            cache.temporal[layer][dst].cpu(),
            full_intermediate[layer, request, step].cpu(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cache.conv[0][layer, dst].cpu(),
            full_conv_intermediate[layer, request, step].cpu(),
            rtol=0,
            atol=0,
        )
print("MPS speculative state verify/commit: OK")
