"""Compare padded and lower-right-causal MPS SDPA extend attention."""

from __future__ import annotations

import argparse
import platform
import statistics
import time

import torch
from torch.nn.attention.bias import causal_lower_right
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend


def _sync() -> None:
    torch.mps.synchronize()


def _time_ms(fn, *, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        fn()
        _sync()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--extend-len", type=int, default=256)
    parser.add_argument("--query-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    if args.prefix_len < 0 or args.extend_len <= 0:
        raise ValueError("prefix length must be nonnegative and extend length positive")
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")

    torch.manual_seed(23)
    device = torch.device("mps")
    dtype = torch.float32
    seq_len = args.prefix_len + args.extend_len
    query = torch.randn(
        1,
        args.query_heads,
        args.extend_len,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        1,
        args.kv_heads,
        seq_len,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
    source_query = query.squeeze(0).movedim(0, 1).contiguous()
    source_key = key.squeeze(0).movedim(0, 1).contiguous()
    source_value = value.squeeze(0).movedim(0, 1).contiguous()
    source_output = torch.empty_like(source_query)
    req_to_token = torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(0)
    req_pool_indices = torch.tensor([0], device=device, dtype=torch.int64)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int64)
    prefix_lens = torch.tensor([args.prefix_len], device=device, dtype=torch.int64)
    extend_lens = torch.tensor([args.extend_len], device=device, dtype=torch.int64)
    source_backend = TorchNativeAttnBackend.__new__(TorchNativeAttnBackend)
    padded_query = torch.empty(
        1,
        args.query_heads,
        seq_len,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    padded_query[:, :, args.prefix_len :, :] = query
    lower_right_bias = causal_lower_right(args.extend_len, seq_len)

    def padded() -> torch.Tensor:
        return scaled_dot_product_attention(
            padded_query,
            key,
            value,
            is_causal=True,
            enable_gqa=args.query_heads != args.kv_heads,
        )[:, :, args.prefix_len :, :]

    def lower_right() -> torch.Tensor:
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=lower_right_bias,
            is_causal=False,
            enable_gqa=args.query_heads != args.kv_heads,
        )

    def explicit_lower_right() -> torch.Tensor:
        query_positions = torch.arange(
            args.prefix_len,
            seq_len,
            device=device,
        ).unsqueeze(1)
        key_positions = torch.arange(seq_len, device=device).unsqueeze(0)
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=key_positions <= query_positions,
            is_causal=False,
            enable_gqa=args.query_heads != args.kv_heads,
        )

    def source_lower_right() -> torch.Tensor:
        return source_backend._run_sdpa_forward_extend(
            source_query,
            source_output,
            source_key,
            source_value,
            req_to_token,
            req_pool_indices,
            seq_lens,
            prefix_lens,
            extend_lens,
            scaling=None,
            enable_gqa=args.query_heads != args.kv_heads,
            causal=True,
        )

    def source_padded() -> torch.Tensor:
        moved_query = source_query.movedim(0, 1)
        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = extend_lens[seq_idx]
            prefill_seq_len_q = prefix_lens[seq_idx]
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv
            per_req_query = moved_query[:, start_q:end_q, :]
            redundant_query = torch.empty(
                args.query_heads,
                seq_len_kv,
                args.head_dim,
                device=device,
                dtype=dtype,
            )
            redundant_query[:, prefill_seq_len_q:, :] = per_req_query
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, start_kv:end_kv]
            per_req_key = source_key[per_req_tokens].movedim(0, 1)
            per_req_value = source_value[per_req_tokens].movedim(0, 1)
            redundant_output = (
                scaled_dot_product_attention(
                    redundant_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    is_causal=True,
                    enable_gqa=args.query_heads != args.kv_heads,
                )
                .squeeze(0)
                .movedim(1, 0)
            )
            source_output[start_q:end_q] = redundant_output[prefill_seq_len_q:]
            start_q, start_kv = end_q, end_kv
        return source_output

    expected = padded()
    actual = lower_right()
    explicit = explicit_lower_right()
    source_actual = source_lower_right().movedim(0, 1).unsqueeze(0)
    source_expected = source_padded().movedim(0, 1).unsqueeze(0)
    _sync()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(explicit.cpu(), expected.cpu(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        source_actual.cpu(), expected.cpu(), rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        source_expected.cpu(), expected.cpu(), rtol=2e-5, atol=2e-5
    )
    max_error = (actual - expected).abs().max().item()
    explicit_max_error = (explicit - expected).abs().max().item()
    source_max_error = (source_actual - expected).abs().max().item()

    for _ in range(args.warmups):
        padded()
        lower_right()
        explicit_lower_right()
        source_lower_right()
        source_padded()
    _sync()

    padded_a = _time_ms(padded, repeats=args.repeats)
    lower_right_samples = _time_ms(lower_right, repeats=args.repeats)
    explicit_samples = _time_ms(explicit_lower_right, repeats=args.repeats)
    source_padded_a = _time_ms(source_padded, repeats=args.repeats)
    source_samples = _time_ms(source_lower_right, repeats=args.repeats)
    source_padded_b = _time_ms(source_padded, repeats=args.repeats)
    padded_b = _time_ms(padded, repeats=args.repeats)

    print(
        f"platform={platform.platform()} torch={torch.__version__} "
        f"device={device} dtype={dtype}"
    )
    print(
        f"shape=batch1,q_heads{args.query_heads},kv_heads{args.kv_heads},"
        f"prefix{args.prefix_len},extend{args.extend_len},head_dim{args.head_dim}"
    )
    print(f"max_error={max_error:.9g}")
    print(f"explicit_max_error={explicit_max_error:.9g}")
    print(f"source_max_error={source_max_error:.9g}")
    print(f"padded_a_ms={','.join(f'{sample:.6f}' for sample in padded_a)}")
    print(
        "lower_right_ms=" + ",".join(f"{sample:.6f}" for sample in lower_right_samples)
    )
    print("explicit_ms=" + ",".join(f"{sample:.6f}" for sample in explicit_samples))
    print(
        "source_padded_a_ms=" + ",".join(f"{sample:.6f}" for sample in source_padded_a)
    )
    print("source_ms=" + ",".join(f"{sample:.6f}" for sample in source_samples))
    print(
        "source_padded_b_ms=" + ",".join(f"{sample:.6f}" for sample in source_padded_b)
    )
    print(f"padded_b_ms={','.join(f'{sample:.6f}' for sample in padded_b)}")
    print(
        f"medians_ms={statistics.median(padded_a):.6f}/"
        f"{statistics.median(lower_right_samples):.6f}/"
        f"{statistics.median(explicit_samples):.6f}/"
        f"{statistics.median(source_padded_a):.6f}/"
        f"{statistics.median(source_samples):.6f}/"
        f"{statistics.median(source_padded_b):.6f}/"
        f"{statistics.median(padded_b):.6f}"
    )


if __name__ == "__main__":
    main()
