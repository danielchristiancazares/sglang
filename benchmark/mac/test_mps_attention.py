"""Correctness smoke for native Metal grouped-query decode attention."""

from __future__ import annotations

import math

import torch

from sglang.srt.hardware_backend.mps.ops import (
    decode_gqa,
    prepare_full_attention,
)
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb


def test_prepare_full_attention() -> None:
    """Match the Qwen3.5 full-attention PyTorch preparation path."""
    torch.manual_seed(11)
    batch, q_heads, kv_heads = 2, 24, 4
    head_dim, rotary_dim = 256, 64
    epsilon = 1e-6
    q_gate_dim = 2 * q_heads * head_dim
    kv_dim = kv_heads * head_dim

    qkv = torch.randn(batch, q_gate_dim + 2 * kv_dim)
    q_weight = torch.randn(head_dim) * 0.02
    k_weight = torch.randn(head_dim) * 0.02
    positions = torch.tensor([3, 7], dtype=torch.int64)
    angles = torch.randn(16, rotary_dim // 2)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)

    q_gate, expected_key, expected_value = qkv.split(
        [q_gate_dim, kv_dim, kv_dim], dim=-1
    )
    q_gate = q_gate.view(batch, q_heads, 2 * head_dim)
    expected_query, expected_gate = q_gate.chunk(2, dim=-1)
    expected_key = expected_key.view(batch, kv_heads, head_dim)
    expected_value = expected_value.view(batch, kv_heads, head_dim)

    def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon)
        return x * inverse_rms * (1 + weight)

    expected_query = gemma_rmsnorm(expected_query, q_weight)
    expected_key = gemma_rmsnorm(expected_key, k_weight)
    cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
    expected_query = torch.cat(
        (
            apply_rotary_emb(
                expected_query[..., :rotary_dim], cos, sin, is_neox_style=True
            ),
            expected_query[..., rotary_dim:],
        ),
        dim=-1,
    )
    expected_key = torch.cat(
        (
            apply_rotary_emb(
                expected_key[..., :rotary_dim], cos, sin, is_neox_style=True
            ),
            expected_key[..., rotary_dim:],
        ),
        dim=-1,
    )

    actual_query, actual_key, actual_value, actual_gate = prepare_full_attention(
        qkv.to("mps"),
        q_weight.to("mps"),
        k_weight.to("mps"),
        cos_sin_cache.to("mps"),
        positions.to("mps"),
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        epsilon,
    )
    for actual, expected in (
        (actual_query, expected_query),
        (actual_key, expected_key),
        (actual_value, expected_value),
        (actual_gate, expected_gate),
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-5)


def main() -> None:
    test_prepare_full_attention()
    torch.manual_seed(17)
    batch, q_heads, kv_heads, head_dim = 2, 24, 4, 256
    cache_slots, req_stride = 64, 64
    scale = 1.0 / math.sqrt(head_dim)

    query = torch.randn(batch, q_heads, head_dim)
    key = torch.randn(batch, kv_heads, head_dim)
    value = torch.randn_like(key)
    key_cache = torch.randn(cache_slots, kv_heads, head_dim)
    value_cache = torch.randn_like(key_cache)
    cache_locations = torch.tensor([10, 11], dtype=torch.int64)
    req_pool_indices = torch.tensor([1, 2], dtype=torch.int64)
    seq_lens = torch.tensor([3, 5], dtype=torch.int64)
    req_to_token = torch.zeros(4, req_stride, dtype=torch.int32)
    req_to_token[1, :3] = torch.tensor([2, 3, 10], dtype=torch.int32)
    req_to_token[2, :5] = torch.tensor([4, 5, 6, 7, 11], dtype=torch.int32)

    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()
    expected_key_cache[cache_locations] = key
    expected_value_cache[cache_locations] = value
    expected = torch.empty_like(query)
    heads_per_kv = q_heads // kv_heads
    for batch_index in range(batch):
        slots = req_to_token[
            req_pool_indices[batch_index], : seq_lens[batch_index]
        ].long()
        for query_head in range(q_heads):
            kv_head = query_head // heads_per_kv
            scores = (
                expected_key_cache[slots, kv_head]
                @ query[batch_index, query_head]
            ) * scale
            expected[batch_index, query_head] = (
                torch.softmax(scores, dim=0).unsqueeze(0)
                @ expected_value_cache[slots, kv_head]
            ).squeeze(0)

    actual_key_cache = key_cache.to("mps")
    actual_value_cache = value_cache.to("mps")
    actual = decode_gqa(
        query.to("mps"),
        key.to("mps"),
        value.to("mps"),
        actual_key_cache,
        actual_value_cache,
        cache_locations.to("mps"),
        req_to_token.to("mps"),
        req_pool_indices.to("mps"),
        seq_lens.to("mps"),
        scale,
    ).cpu()
    torch.testing.assert_close(actual, expected.reshape(batch, -1), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_key_cache.cpu(), expected_key_cache)
    torch.testing.assert_close(actual_value_cache.cpu(), expected_value_cache)
    print(f"max_error={(actual - expected.reshape(batch, -1)).abs().max().item():.6g}")


if __name__ == "__main__":
    main()
