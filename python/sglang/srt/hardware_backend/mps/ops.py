from __future__ import annotations

import importlib
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _extension():
    try:
        return importlib.import_module("sglang._metal_gguf")
    except ImportError:
        pass

    from torch.utils.cpp_extension import load

    venv_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

    source = (
        Path(__file__).resolve().parents[3]
        / "kernels"
        / "aot"
        / "csrc"
        / "metal"
        / "gguf_q4_0.mm"
    )
    logger.info("Building native Metal GGUF/GDN extension from %s", source)
    return load(
        name="sglang_metal_gguf",
        sources=[str(source)],
        extra_cflags=[
            "-O3",
            "-fobjc-arc",
            "-Wno-invalid-specialization",
            "-Wno-deprecated-literal-operator",
        ],
        extra_ldflags=[
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-framework",
            "MetalPerformanceShaders",
            "-framework",
            "MetalPerformanceShadersGraph",
        ],
        verbose=False,
    )


def q4_0_matmul(
    packed_weight: torch.Tensor,
    x: torch.Tensor,
    output_size: int,
    input_size: int,
) -> torch.Tensor:
    original_shape = x.shape[:-1]
    output = _extension().q4_0_matmul(
        packed_weight.contiguous(),
        x.reshape(-1, input_size).contiguous(),
        output_size,
        input_size,
    )
    return output.view(*original_shape, output_size)


def quant_matmul(
    packed_weight: torch.Tensor,
    x: torch.Tensor,
    output_size: int,
    input_size: int,
    weight_type: int,
) -> torch.Tensor:
    original_shape = x.shape[:-1]
    output = _extension().quant_matmul(
        packed_weight.contiguous(),
        x.reshape(-1, input_size).to(torch.float32).contiguous(),
        output_size,
        input_size,
        int(weight_type),
    )
    return output.view(*original_shape, output_size)


def dense_matmul(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape[:-1]
    output = _extension().dense_matmul(
        weight.contiguous(), x.reshape(-1, x.shape[-1]).contiguous()
    )
    return output.view(*original_shape, weight.shape[0])


def q4_0_embedding(
    packed_weight: torch.Tensor,
    token_ids: torch.Tensor,
    vocab_size: int,
    hidden_size: int,
) -> torch.Tensor:
    return _extension().q4_0_embedding(
        packed_weight.contiguous(),
        token_ids.contiguous(),
        vocab_size,
        hidden_size,
    )


def quant_embedding(
    packed_weight: torch.Tensor,
    token_ids: torch.Tensor,
    vocab_size: int,
    hidden_size: int,
    weight_type: int,
) -> torch.Tensor:
    return _extension().quant_embedding(
        packed_weight.contiguous(),
        token_ids.contiguous(),
        vocab_size,
        hidden_size,
        int(weight_type),
    )


def causal_conv1d_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    return _extension().causal_conv1d_decode(
        x.contiguous(),
        weight.contiguous(),
        state,
        cache_indices.to(torch.int32).contiguous(),
    )


def causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    return _extension().causal_conv1d_prefill(
        x.contiguous(),
        weight.contiguous(),
        state,
        cache_indices.to(torch.int32).contiguous(),
        query_start_loc.to(torch.int32).contiguous(),
        has_initial_state.to(torch.int32).contiguous(),
    )


def gdn_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    return _extension().gdn_decode(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        a.contiguous(),
        b.contiguous(),
        A_log.contiguous(),
        dt_bias.contiguous(),
        state,
        cache_indices.contiguous(),
    )


def gdn_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    return _extension().gdn_prefill(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        state,
        cache_indices.to(torch.int32).contiguous(),
        query_start_loc.to(torch.int32).contiguous(),
    )


def decode_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_locations: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fused native Metal KV write and grouped-query decode attention."""
    batch_size = query.shape[0]
    head_dim = key_cache.shape[-1]
    num_kv_heads = key_cache.shape[-2]
    num_q_heads = query.numel() // (batch_size * head_dim)
    output = _extension().decode_gqa(
        query.reshape(batch_size, num_q_heads, head_dim).contiguous(),
        key.reshape(batch_size, num_kv_heads, head_dim).contiguous(),
        value.reshape(batch_size, num_kv_heads, head_dim).contiguous(),
        key_cache,
        value_cache,
        cache_locations.contiguous(),
        req_to_token,
        req_pool_indices.contiguous(),
        seq_lens.contiguous(),
        float(scale),
    )
    return output.reshape(batch_size, num_q_heads * head_dim)


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return _extension().gemma_rmsnorm(
        x.contiguous(), weight.contiguous(), float(epsilon)
    )


def gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, residual_output = _extension().gemma_fused_add_rmsnorm(
        x.contiguous(),
        residual.contiguous(),
        weight.contiguous(),
        float(epsilon),
    )
    return output, residual_output


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return _extension().silu_and_mul(x.contiguous())


def pack_gdn_inputs(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_v_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        _extension().pack_gdn_inputs(
            qkvz.contiguous(),
            ba.contiguous(),
            int(key_dim),
            int(value_dim),
            int(num_v_heads),
        )
    )


def gdn_gated_rmsnorm_reorder(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    epsilon: float,
) -> torch.Tensor:
    return _extension().gdn_gated_rmsnorm_reorder(
        x.contiguous(),
        gate.contiguous(),
        weight.contiguous(),
        int(num_k_heads),
        int(num_v_heads),
        int(head_dim),
        float(epsilon),
    )


def prepare_full_attention(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        _extension().prepare_full_attention(
            qkv.contiguous(),
            q_weight.contiguous(),
            k_weight.contiguous(),
            cos_sin_cache.contiguous(),
            positions.to(torch.int64).contiguous(),
            int(num_q_heads),
            int(num_kv_heads),
            int(head_dim),
            int(rotary_dim),
            float(epsilon),
        )
    )


def sigmoid_mul_inplace(
    x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return _extension().sigmoid_mul_inplace(x, gate.contiguous())
