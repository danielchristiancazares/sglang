"""Correctness checks for native Metal decode glue kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.mps.ops import (
    dense_matmul,
    gdn_decode,
    gdn_gated_rmsnorm_reorder,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    pack_gdn_inputs,
    prepare_full_attention,
    sigmoid_mul_inplace,
    silu_and_mul,
)


def main() -> None:
    torch.manual_seed(23)
    batch, hidden = 8, 5120
    x = torch.randn(batch, hidden)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden) * 0.1
    epsilon = 1e-6

    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
    expected = expected * (1 + weight)
    actual = gemma_rmsnorm(x.to("mps"), weight.to("mps"), epsilon).cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    summed = x + residual
    expected_norm = summed * torch.rsqrt(
        summed.square().mean(-1, keepdim=True) + epsilon
    )
    expected_norm *= 1 + weight
    actual_norm, actual_residual = gemma_fused_add_rmsnorm(
        x.to("mps"), residual.to("mps"), weight.to("mps"), epsilon
    )
    torch.testing.assert_close(actual_norm.cpu(), expected_norm, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_residual.cpu(), summed, rtol=2e-6, atol=2e-6)

    activation_input = torch.randn(batch, 2 * 17408)
    expected_activation = F.silu(activation_input[:, :17408]) * activation_input[:, 17408:]
    actual_activation = silu_and_mul(activation_input.to("mps")).cpu()
    torch.testing.assert_close(
        actual_activation, expected_activation, rtol=2e-5, atol=2e-5
    )

    dense_weight = torch.randn(96, hidden)
    expected_dense = x @ dense_weight.T
    actual_dense = dense_matmul(dense_weight.to("mps"), x.to("mps")).cpu()
    torch.testing.assert_close(actual_dense, expected_dense, rtol=2e-5, atol=2e-4)

    key_dim, value_dim, value_heads, head_dim = 2048, 6144, 48, 128
    qkvz = torch.randn(batch, 2 * key_dim + 2 * value_dim)
    ba = torch.randn(batch, 2 * value_heads)
    mixed, gate, b, a = pack_gdn_inputs(
        qkvz.to("mps"), ba.to("mps"), key_dim, value_dim, value_heads
    )
    torch.testing.assert_close(mixed.cpu(), qkvz[:, : 2 * key_dim + value_dim])
    torch.testing.assert_close(gate.cpu(), qkvz[:, 2 * key_dim + value_dim :])
    torch.testing.assert_close(b.cpu(), ba[:, :value_heads])
    torch.testing.assert_close(a.cpu(), ba[:, value_heads:])

    norm_input = torch.randn(batch, value_heads, head_dim)
    norm_gate = torch.randn_like(norm_input)
    norm_weight = torch.randn(head_dim)
    normalized = norm_input * torch.rsqrt(
        norm_input.square().mean(-1, keepdim=True) + epsilon
    )
    normalized *= norm_weight * F.silu(norm_gate)
    expected_reorder = normalized.reshape(batch, 16, 3, head_dim)
    expected_reorder = expected_reorder.transpose(1, 2).contiguous().reshape(batch, -1)
    actual_reorder = gdn_gated_rmsnorm_reorder(
        norm_input.to("mps"),
        norm_gate.to("mps"),
        norm_weight.to("mps"),
        16,
        value_heads,
        head_dim,
        epsilon,
    ).cpu()
    torch.testing.assert_close(
        actual_reorder.reshape(batch, -1), expected_reorder, rtol=2e-5, atol=2e-5
    )

    gdn_batch, key_heads, value_heads, value_dim = 2, 1, 3, 4
    query = torch.randn(1, gdn_batch, key_heads, head_dim)
    key = torch.randn_like(query)
    value = torch.randn(1, gdn_batch, value_heads, value_dim)
    decay_a = torch.randn(gdn_batch, value_heads)
    update_b = torch.randn_like(decay_a)
    a_log = torch.randn(value_heads)
    dt_bias = torch.randn(value_heads)
    state = torch.randn(4, value_heads, value_dim, head_dim)
    indices = torch.tensor([1, 3], dtype=torch.int32)
    expected_state = state.clone()
    q_norm = query * torch.rsqrt(query.square().mean(-1, keepdim=True) + 1e-6) / head_dim
    k_norm = key * torch.rsqrt(key.square().mean(-1, keepdim=True) + 1e-6) / (head_dim**0.5)
    expected_gdn = torch.empty_like(value)
    for batch_index in range(gdn_batch):
        for value_head in range(value_heads):
            key_head = value_head // (value_heads // key_heads)
            decay = torch.exp(
                -torch.exp(a_log[value_head])
                * F.softplus(decay_a[batch_index, value_head] + dt_bias[value_head])
            )
            beta = torch.sigmoid(update_b[batch_index, value_head])
            slot = indices[batch_index]
            for dim in range(value_dim):
                state_row = expected_state[slot, value_head, dim] * decay
                remembered = (state_row * k_norm[0, batch_index, key_head]).sum()
                delta = (value[0, batch_index, value_head, dim] - remembered) * beta
                state_row += k_norm[0, batch_index, key_head] * delta
                expected_state[slot, value_head, dim] = state_row
                expected_gdn[0, batch_index, value_head, dim] = (
                    state_row * q_norm[0, batch_index, key_head]
                ).sum()
    actual_state = state.to("mps")
    actual_gdn = gdn_decode(
        query.to("mps"),
        key.to("mps"),
        value.to("mps"),
        decay_a.to("mps"),
        update_b.to("mps"),
        a_log.to("mps"),
        dt_bias.to("mps"),
        actual_state,
        indices.to("mps"),
    )
    torch.testing.assert_close(actual_gdn.cpu(), expected_gdn, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(actual_state.cpu(), expected_state, rtol=3e-5, atol=3e-5)

    tokens, query_heads, kv_heads, rotary_dim = 3, 6, 2, 64
    attention_qkv = torch.randn(
        tokens, 2 * query_heads * head_dim + 2 * kv_heads * head_dim
    )
    q_weight = torch.randn(head_dim) * 0.1
    k_weight = torch.randn(head_dim) * 0.1
    positions = torch.tensor([0, 3, 7], dtype=torch.int64)
    cos_sin = torch.randn(16, rotary_dim)
    q_gate = attention_qkv[:, : 2 * query_heads * head_dim].reshape(
        tokens, query_heads, 2 * head_dim
    )
    expected_q = q_gate[..., :head_dim]
    expected_gate = q_gate[..., head_dim:]
    expected_k = attention_qkv[
        :, 2 * query_heads * head_dim : 2 * query_heads * head_dim + kv_heads * head_dim
    ].reshape(tokens, kv_heads, head_dim)
    expected_v = attention_qkv[:, -kv_heads * head_dim :].reshape(
        tokens, kv_heads, head_dim
    )
    expected_q = expected_q * torch.rsqrt(
        expected_q.square().mean(-1, keepdim=True) + epsilon
    ) * (1 + q_weight)
    expected_k = expected_k * torch.rsqrt(
        expected_k.square().mean(-1, keepdim=True) + epsilon
    ) * (1 + k_weight)
    for tensor in (expected_q, expected_k):
        rotary = tensor[..., :rotary_dim].clone()
        first, second = rotary.chunk(2, dim=-1)
        selected = cos_sin[positions]
        cosine, sine = selected.chunk(2, dim=-1)
        tensor[..., : rotary_dim // 2] = first * cosine[:, None] - second * sine[:, None]
        tensor[..., rotary_dim // 2 : rotary_dim] = second * cosine[:, None] + first * sine[:, None]
    actual_q, actual_k, actual_v, actual_gate = prepare_full_attention(
        attention_qkv.to("mps"),
        q_weight.to("mps"),
        k_weight.to("mps"),
        cos_sin.to("mps"),
        positions.to("mps"),
        query_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        epsilon,
    )
    torch.testing.assert_close(actual_q.cpu(), expected_q, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(actual_k.cpu(), expected_k, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(actual_v.cpu(), expected_v)
    torch.testing.assert_close(actual_gate.cpu(), expected_gate)
    gate_input = torch.randn_like(expected_q)
    gated = expected_q.clone().to("mps")
    sigmoid_mul_inplace(gated, gate_input.to("mps"))
    torch.testing.assert_close(
        gated.cpu(), expected_q * torch.sigmoid(gate_input), rtol=3e-5, atol=3e-5
    )
    packed_gate = torch.randn(tokens, query_heads, 2 * head_dim)
    strided_gate = packed_gate[..., head_dim:]
    gate_value = strided_gate.reshape(tokens, -1)
    gated = expected_q.reshape(tokens, -1).clone().to("mps")
    sigmoid_mul_inplace(gated, gate_value.to("mps"))
    torch.testing.assert_close(
        gated.cpu(),
        expected_q.reshape(tokens, -1) * torch.sigmoid(gate_value),
        rtol=3e-5,
        atol=3e-5,
    )
    print("native Metal fused-op checks passed")


if __name__ == "__main__":
    main()
