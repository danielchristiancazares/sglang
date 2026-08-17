"""Compare Qwen3.5 full-attention preparation on MPS."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from sglang.srt.hardware_backend.mps.ops import gemma_rmsnorm, prepare_full_attention
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(23)
    device = torch.device("mps")
    q_heads, kv_heads = 24, 4
    head_dim, rotary_dim = 256, 64
    epsilon = 1e-6
    q_gate_dim = 2 * q_heads * head_dim
    kv_dim = kv_heads * head_dim
    qkv = torch.randn(
        args.batch_size,
        q_gate_dim + 2 * kv_dim,
        device=device,
    )
    q_weight = (torch.randn(head_dim) * 0.02).to(device)
    k_weight = (torch.randn(head_dim) * 0.02).to(device)
    positions = torch.arange(args.batch_size, dtype=torch.int64, device=device)
    angles = torch.randn(4096, rotary_dim // 2, device=device)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)

    def current_prepare():
        q_gate, key, value = qkv.split([q_gate_dim, kv_dim, kv_dim], dim=-1)
        q_gate = q_gate.view(args.batch_size, q_heads, 2 * head_dim)
        query, gate = q_gate.chunk(2, dim=-1)
        query = gemma_rmsnorm(
            query.reshape(-1, head_dim), q_weight, epsilon
        ).view(args.batch_size, q_heads, head_dim)
        key = gemma_rmsnorm(
            key.reshape(-1, head_dim), k_weight, epsilon
        ).view(args.batch_size, kv_heads, head_dim)
        cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
        query = torch.cat(
            (
                apply_rotary_emb(
                    query[..., :rotary_dim], cos, sin, is_neox_style=True
                ),
                query[..., rotary_dim:],
            ),
            dim=-1,
        )
        key = torch.cat(
            (
                apply_rotary_emb(
                    key[..., :rotary_dim], cos, sin, is_neox_style=True
                ),
                key[..., rotary_dim:],
            ),
            dim=-1,
        )
        return (
            query.contiguous(),
            key.contiguous(),
            value.view(args.batch_size, kv_heads, head_dim).contiguous(),
            gate.reshape(args.batch_size, q_heads * head_dim).contiguous(),
        )

    def metal_prepare():
        return prepare_full_attention(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            epsilon,
        )

    for _ in range(args.warmup):
        current_prepare()
        metal_prepare()
    torch.mps.synchronize()

    timings = {"current": [], "metal": []}
    for _ in range(args.iterations):
        for name, operation in (
            ("current", current_prepare),
            ("metal", metal_prepare),
        ):
            start = time.perf_counter()
            operation()
            torch.mps.synchronize()
            timings[name].append((time.perf_counter() - start) * 1000)

    for name, values in timings.items():
        print(
            f"{name} batch={args.batch_size} median={statistics.median(values):.3f}ms "
            f"raw={','.join(f'{value:.3f}' for value in values)}"
        )


if __name__ == "__main__":
    main()
