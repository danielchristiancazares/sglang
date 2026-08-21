"""Exercise torch-native MPS decode admission and SDPA fallback parity."""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend


class _TokenPool:
    kv_cache_layout = "nhd"

    def __init__(self, key_cache: torch.Tensor, value_cache: torch.Tensor):
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.full_kv_pool = self

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self.key_cache

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self.value_cache

    def set_kv_buffer(self, layer, loc_info, key, value) -> None:
        del layer
        locations = loc_info.loc.long()
        self.key_cache[locations] = key.reshape_as(self.key_cache[locations]).to(
            self.key_cache.dtype
        )
        self.value_cache[locations] = value.reshape_as(self.value_cache[locations]).to(
            self.value_cache.dtype
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-slots", type=int, default=32769)
    parser.add_argument(
        "--cache-dtype", choices=("float32", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--seq-len", type=int, default=257)
    parser.add_argument("--expect-error", action="store_true")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    if args.seq_len > args.cache_slots:
        raise ValueError("active sequence cannot exceed physical cache slots")

    torch.manual_seed(47)
    device = torch.device("mps")
    cache_dtype = getattr(torch, args.cache_dtype)
    batch_size, query_heads, kv_heads, head_dim = 1, 24, 4, 256
    scale = 1.0 / math.sqrt(head_dim)
    key_cache = torch.randn(
        args.cache_slots,
        kv_heads,
        head_dim,
        dtype=cache_dtype,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(batch_size, query_heads * head_dim, device=device)
    key = torch.randn(batch_size, kv_heads * head_dim, device=device)
    value = torch.randn_like(key)
    req_to_token = torch.arange(
        args.seq_len, dtype=torch.int32, device=device
    ).unsqueeze(0)
    cache_location = torch.tensor([args.seq_len - 1], dtype=torch.int64, device=device)

    token_pool = _TokenPool(key_cache, value_cache)
    backend = TorchNativeAttnBackend.__new__(TorchNativeAttnBackend)
    backend.token_to_kv_pool = token_pool
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    backend.swa_out_cache_loc = None
    backend.seq_lens_override = None
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=query_heads,
        tp_k_head_num=kv_heads,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        is_cross_attention=False,
        sliding_window_size=None,
        scaling=scale,
    )
    forward_batch = SimpleNamespace(
        out_cache_loc=cache_location,
        encoder_out_cache_loc=None,
        seq_lens=torch.tensor([args.seq_len], dtype=torch.int64, device=device),
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=device),
        encoder_lens=None,
    )

    try:
        output = backend.forward_decode(query, key, value, layer, forward_batch)
        torch.mps.synchronize()
    except RuntimeError as error:
        if not args.expect_error:
            raise
        print(f"status=expected_error error={error}")
        return

    if args.expect_error:
        raise AssertionError("decode completed while --expect-error was set")

    slots = req_to_token[0].long()
    reference = scaled_dot_product_attention(
        query.view(batch_size, query_heads, 1, head_dim),
        key_cache[slots].movedim(0, 1).unsqueeze(0).to(query.dtype),
        value_cache[slots].movedim(0, 1).unsqueeze(0).to(query.dtype),
        is_causal=False,
        enable_gqa=True,
        scale=scale,
    ).view_as(output)
    torch.mps.synchronize()
    torch.testing.assert_close(output.cpu(), reference.cpu(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        key_cache[cache_location].cpu(),
        key.reshape(batch_size, kv_heads, head_dim).to(cache_dtype).cpu(),
    )
    torch.testing.assert_close(
        value_cache[cache_location].cpu(),
        value.reshape(batch_size, kv_heads, head_dim).to(cache_dtype).cpu(),
    )
    max_error = (output - reference).abs().max().item()
    print(
        f"status=ok cache_slots={args.cache_slots} cache_dtype={cache_dtype} "
        f"seq_len={args.seq_len} max_error={max_error:.9g}"
    )


if __name__ == "__main__":
    main()
