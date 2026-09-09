"""Benchmark native Metal grouped-query decode attention at production shape."""

from __future__ import annotations

import argparse
import math
import statistics
import time

import torch

from sglang.srt.hardware_backend.mps.ops import decode_gqa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=6213)
    parser.add_argument("--cache-slots", type=int, default=7680)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    if args.sequence_length > args.cache_slots:
        raise ValueError("sequence length must fit in the cache")
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")

    device = torch.device("mps")
    query = torch.randn(
        args.batch_size, args.query_heads, args.head_dim, device=device
    )
    key = torch.randn(args.batch_size, args.kv_heads, args.head_dim, device=device)
    value = torch.randn_like(key)
    key_cache = torch.randn(
        args.cache_slots, args.kv_heads, args.head_dim, device=device
    )
    value_cache = torch.randn_like(key_cache)
    cache_locations = torch.full(
        (args.batch_size,), args.sequence_length - 1, dtype=torch.int64, device=device
    )
    req_to_token = torch.arange(
        args.cache_slots, dtype=torch.int32, device=device
    ).repeat(args.batch_size, 1)
    req_pool_indices = torch.arange(
        args.batch_size, dtype=torch.int64, device=device
    )
    seq_lens = torch.full(
        (args.batch_size,), args.sequence_length, dtype=torch.int64, device=device
    )
    scale = 1.0 / math.sqrt(args.head_dim)

    def run() -> None:
        decode_gqa(
            query,
            key,
            value,
            key_cache,
            value_cache,
            cache_locations,
            req_to_token,
            req_pool_indices,
            seq_lens,
            scale,
        )

    run()
    torch.mps.synchronize()
    trial_seconds = []
    for _ in range(args.trials):
        started = time.perf_counter()
        for _ in range(args.iterations):
            run()
        torch.mps.synchronize()
        trial_seconds.append((time.perf_counter() - started) / args.iterations)

    median = statistics.median(trial_seconds)
    print(
        f"batch={args.batch_size} q_heads={args.query_heads} "
        f"kv_heads={args.kv_heads} head_dim={args.head_dim} "
        f"sequence_length={args.sequence_length} median={median * 1000:.3f}ms "
        f"sixteen_layers={median * 16 * 1000:.3f}ms"
    )
    print("raw_ms=" + ",".join(f"{value * 1000:.3f}" for value in trial_seconds))


if __name__ == "__main__":
    main()
