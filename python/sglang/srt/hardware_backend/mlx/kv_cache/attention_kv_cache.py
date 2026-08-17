"""Attention KV cache adapters for the MLX backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_pool import (
        MlxAttentionKVPool,
    )


def make_attention_mask(N, offset, return_array=False, window_size=None):
    """Mirror mlx_lm ``cache.create_attention_mask`` for cache shims.

    Containers delegate mask creation to ``cache.make_mask`` whenever the
    cache exposes it, so the shims must honor ``window_size`` (sliding-window
    layers pass it, including for N == 1) or windowed models silently fall
    back to full attention.
    """
    if window_size is not None and offset + N > window_size:
        return create_causal_mask(N, offset, window_size=window_size)
    # Either no window, or a window that cannot bind: the lowest query position
    # is ``offset``, so ``offset + N <= window_size`` puts every causally
    # visible key inside the band and the banded mask is elementwise identical
    # to a plain causal one (the same shortcut mlx_lm's RotatingKVCache takes).
    # Worth the branch because a materialised mask forces
    # mx.fast.scaled_dot_product_attention off its fused causal path: ~2x
    # slower per layer, plus an N x (offset + N) allocation.
    if N == 1:
        return None
    if return_array:
        return create_causal_mask(N, offset)
    return "causal"


class AttentionOffsetCache:
    """Data-free shim satisfying mlx-lm's cache protocol.

    Provides ``make_mask`` and ``state`` without storing actual K/V.
    """

    def __init__(self, offset: int = 0):
        self.offset = offset

    @property
    def state(self):
        return ()  # Empty — safe for mx.eval unpacking

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        return make_attention_mask(
            N, self.offset, return_array=return_array, window_size=window_size
        )

    def update_and_fetch(self, keys, values):
        raise RuntimeError("AttentionOffsetCache should not store data")


_DEFAULT_MAX_SEQ_LEN = 4096


class ContiguousAttentionKVCache:
    """Pre-allocated attention KV buffer for one request and one layer.

    Shape ``(1, n_kv_heads, max_seq_len, head_dim)``.  Slice assignment
    instead of ``mx.concatenate``.  Lazy-allocated on first write.
    """

    __slots__ = ("keys", "values", "offset", "max_seq_len")

    def __init__(
        self,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        max_seq_len: int = _DEFAULT_MAX_SEQ_LEN,
        dtype: mx.Dtype | None = None,
    ):
        if n_kv_heads is not None and head_dim is not None and dtype is not None:
            self.keys = mx.zeros((1, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
            self.values = mx.zeros((1, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
        else:
            self.keys = None
            self.values = None
        self.offset = 0
        self.max_seq_len = max_seq_len

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        return make_attention_mask(
            N, self.offset, return_array=return_array, window_size=window_size
        )

    def _allocate(self, keys: mx.array) -> None:
        """Allocate buffers matching the first key tensor's shape."""
        B, n_kv_heads, _, head_dim = keys.shape
        self.keys = mx.zeros(
            (B, n_kv_heads, self.max_seq_len, head_dim), dtype=keys.dtype
        )
        self.values = mx.zeros(
            (B, n_kv_heads, self.max_seq_len, head_dim), dtype=keys.dtype
        )

    @property
    def state(self):
        """Arrays for ``mx.eval`` unpacking."""
        if self.keys is None:
            return ()
        return (self.keys, self.values)

    def _grow(self, required: int) -> None:
        """Double the buffer until it can hold *required* tokens."""
        new_max = self.max_seq_len
        while new_max < required:
            new_max *= 2
        B, n_kv_heads, _, head_dim = self.keys.shape
        new_k = mx.zeros((B, n_kv_heads, new_max, head_dim), dtype=self.keys.dtype)
        new_v = mx.zeros((B, n_kv_heads, new_max, head_dim), dtype=self.values.dtype)
        if self.offset > 0:
            new_k[:, :, : self.offset, :] = self.keys[:, :, : self.offset, :]
            new_v[:, :, : self.offset, :] = self.values[:, :, : self.offset, :]
        self.keys = new_k
        self.values = new_v
        self.max_seq_len = new_max

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append K/V and return all valid K/V up to current offset."""
        if self.keys is None:
            self._allocate(keys)
        S = keys.shape[2]
        end = self.offset + S
        if end > self.max_seq_len:
            self._grow(end)
        self.keys[:, :, self.offset : end, :] = keys
        self.values[:, :, self.offset : end, :] = values
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def write_token(self, k: mx.array, v: mx.array) -> None:
        """Write one token. k, v shape: (1, n_kv_heads, 1, head_dim)."""
        end = self.offset + 1
        if end > self.max_seq_len:
            self._grow(end)
        self.keys[:, :, self.offset : end, :] = k
        self.values[:, :, self.offset : end, :] = v
        self.offset = end

    def get_kv(self, window: int | None = None) -> tuple[mx.array, mx.array]:
        """Return valid K/V: (1, n_kv_heads, min(offset, window), head_dim).

        ``window`` keeps only the trailing window a sliding-window layer can
        attend to; slicing it here costs one op instead of two.
        """
        start = 0 if window is None else max(0, self.offset - window)
        return (
            self.keys[:, :, start : self.offset, :],
            self.values[:, :, start : self.offset, :],
        )

    def trim(self, n: int) -> None:
        """Roll back the last ``n`` tokens (speculative-decode rejection).

        Rows beyond the new offset stay in the buffer; they are masked out
        by ``offset`` and overwritten by the next append.
        """
        if n < 0 or n > self.offset:
            raise ValueError(f"cannot trim {n} tokens from offset {self.offset}")
        self.offset -= n

    def reset(self) -> None:
        """Reset for reuse, keeping allocated buffers."""
        self.offset = 0


class QuantizedAttentionKVCache:
    """Geometrically grown affine-quantized attention KV for long contexts.

    MLX's stock ``QuantizedKVCache`` grows in 256-token increments. That is a
    good interactive default, though repeatedly concatenating the entire
    quantized history becomes costly at six-figure context lengths. This
    adapter starts with a 4K block and doubles at each boundary, requiring at
    most seven growths on the path to 262K while keeping short prompts free of
    a capacity-sized write target.

    Each of keys and values is the ``(packed, scales, biases)`` tuple returned
    by :func:`mx.quantize`. The cache protocol is intentionally identical to
    mlx-lm's quantized cache, including ``bits`` and ``group_size`` attributes,
    so model-native prefill dispatches to quantized SDPA automatically.
    """

    __slots__ = (
        "bits",
        "capacity",
        "group_size",
        "keys",
        "max_seq_len",
        "offset",
        "values",
    )

    def __init__(
        self,
        max_seq_len: int,
        group_size: int = 64,
        bits: int = 4,
    ):
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if group_size not in (32, 64, 128):
            raise ValueError(
                f"MLX affine KV group_size must be one of 32, 64, 128; got {group_size}"
            )
        if bits not in (4, 8):
            raise ValueError(f"MLX long-context KV supports 4 or 8 bits; got {bits}")
        self.keys = None
        self.values = None
        self.offset = 0
        self.capacity = 0
        self.max_seq_len = max_seq_len
        self.group_size = group_size
        self.bits = bits

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        return make_attention_mask(
            N, self.offset, return_array=return_array, window_size=window_size
        )

    def _next_capacity(self, required: int) -> int:
        capacity = self.capacity or min(4096, self.max_seq_len)
        while capacity < required:
            capacity = min(capacity * 2, self.max_seq_len)
        return capacity

    def _allocate(self, keys: mx.array, values: mx.array, capacity: int) -> None:
        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        elements_per_word = 32 // self.bits

        def allocate(dim: int, dtype: mx.Dtype):
            if dim % self.group_size:
                raise ValueError(
                    f"KV head dimension {dim} must be divisible by affine "
                    f"group_size {self.group_size}"
                )
            if dim % elements_per_word:
                raise ValueError(
                    f"KV head dimension {dim} cannot be packed at {self.bits} bits"
                )
            shape = (B, n_kv_heads, capacity)
            return (
                mx.zeros((*shape, dim // elements_per_word), dtype=mx.uint32),
                mx.zeros((*shape, dim // self.group_size), dtype=dtype),
                mx.zeros((*shape, dim // self.group_size), dtype=dtype),
            )

        self.keys = allocate(k_head_dim, keys.dtype)
        self.values = allocate(v_head_dim, values.dtype)
        self.capacity = capacity

    def _grow(self, keys: mx.array, values: mx.array, required: int) -> None:
        old_keys, old_values = self.keys, self.values
        self._allocate(keys, values, self._next_capacity(required))
        if self.offset:
            for target, source in zip(self.keys, old_keys):
                target[..., : self.offset, :] = source[..., : self.offset, :]
            for target, source in zip(self.values, old_values):
                target[..., : self.offset, :] = source[..., : self.offset, :]

    @staticmethod
    def _slice(quantized, end: int, start: int = 0):
        return tuple(value[..., start:end, :] for value in quantized)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        count = keys.shape[2]
        start = self.offset
        end = start + count
        if end > self.max_seq_len:
            raise ValueError(
                f"quantized KV capacity exceeded: need {end}, "
                f"configured for {self.max_seq_len} tokens"
            )
        if self.keys is None:
            self._allocate(keys, values, self._next_capacity(end))
        elif end > self.capacity:
            self._grow(keys, values, end)

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for target, source in zip(self.keys, q_keys):
            target[..., start:end, :] = source
        for target, source in zip(self.values, q_values):
            target[..., start:end, :] = source
        self.offset = end
        return self._slice(self.keys, end), self._slice(self.values, end)

    def write_token(self, keys: mx.array, values: mx.array) -> None:
        self.update_and_fetch(keys, values)

    def get_kv(self, window: int | None = None):
        start = 0 if window is None else max(0, self.offset - window)
        return (
            self._slice(self.keys, self.offset, start),
            self._slice(self.values, self.offset, start),
        )

    @property
    def state(self):
        if self.keys is None:
            return ()
        return self._slice(self.keys, self.offset), self._slice(
            self.values, self.offset
        )

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return sum(value.nbytes for value in (*self.keys, *self.values))

    def reset(self) -> None:
        self.offset = 0


class WindowedAttentionKVCache:
    """Sliding-window attention KV buffer for one request and one layer.

    Holds the trailing ``window`` tokens plus the in-flight chunk, in
    temporal order, instead of the full sequence.  ``offset`` stays
    absolute (RoPE positions, decode bookkeeping); the dropped prefix
    shows up only in the shorter arrays returned by
    ``update_and_fetch``/``get_kv`` and in the mask offset ``make_mask``
    clamps to, so mask width always equals returned key length.
    """

    __slots__ = ("keys", "values", "offset", "window", "_local")

    def __init__(self, window: int):
        self.window = window
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0  # absolute: every token ever written
        self._local = 0  # tokens currently in the buffer

    @property
    def state(self):
        """Arrays for ``mx.eval`` unpacking."""
        if self.keys is None:
            return ()
        return (self.keys, self.values)

    def reset(self) -> None:
        """Reset for reuse, keeping allocated buffers."""
        self.offset = 0
        self._local = 0

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        kept = min(self._local, self.window)
        if window_size is None and self.offset > kept:
            raise RuntimeError(
                "WindowedAttentionKVCache holds only the trailing window and "
                "cannot serve a full-context attention mask"
            )
        # No N == 1 shortcut here: mlx_lm's banded mask is
        # ``linds < rinds + window_size`` (strict), so a window of W admits
        # exactly W keys, while this buffer returns W + 1 once ``kept ==
        # window`` -- the trailing window plus the token just written.
        return make_attention_mask(
            N, kept, return_array=return_array, window_size=window_size
        )

    def _append(self, keys: mx.array, values: mx.array) -> tuple[int, int]:
        """Append a chunk in place; return the (start, end) span it serves.

        Split out of ``update_and_fetch`` so the decode path can skip building
        the two return slices, which it discards in favour of ``get_kv``.
        """
        S = keys.shape[2]
        kept = min(self._local, self.window)
        capacity = self.window + max(S, self.window)
        held = self.keys.shape[2] if self.keys is not None else 0
        if self._local + S > held or held > capacity:
            # Compact the trailing window into a right-sized buffer: this
            # allocates on the first write, drops history when the buffer
            # fills (amortised O(1) per decode token), and shrinks back to
            # 2 * window once an oversized prefill chunk is behind us.
            B, n_kv_heads, _, head_dim = keys.shape
            new_k = mx.zeros((B, n_kv_heads, capacity, head_dim), dtype=keys.dtype)
            new_v = mx.zeros((B, n_kv_heads, capacity, head_dim), dtype=keys.dtype)
            if kept:
                src = slice(self._local - kept, self._local)
                new_k[:, :, :kept, :] = self.keys[:, :, src, :]
                new_v[:, :, :kept, :] = self.values[:, :, src, :]
            self.keys, self.values, self._local = new_k, new_v, kept
        start, end = self._local - kept, self._local + S
        self.keys[:, :, self._local : end, :] = keys
        self.values[:, :, self._local : end, :] = values
        self._local = end
        self.offset += S
        return start, end

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append a chunk and return the kept trailing window plus the chunk.

        The kept prefix is ``min(local, window)``, matching what
        ``make_mask`` clamps to earlier in the same forward pass.
        """
        start, end = self._append(keys, values)
        return self.keys[:, :, start:end, :], self.values[:, :, start:end, :]

    def write_token(self, k: mx.array, v: mx.array) -> None:
        """Write one token. k, v shape: (1, n_kv_heads, 1, head_dim)."""
        self._append(k, v)

    def get_kv(self, window: int | None = None) -> tuple[mx.array, mx.array]:
        """Return buffered trailing K/V: (1, n_kv_heads, kept, head_dim).

        ``window`` mirrors :meth:`ContiguousAttentionKVCache.get_kv`, but the
        slice is buffer-relative: this buffer holds at most ``2 * window``
        tokens, so the trailing window starts from ``_local``, not ``offset``.
        """
        start = 0 if window is None else max(0, self._local - window)
        return (
            self.keys[:, :, start : self._local, :],
            self.values[:, :, start : self._local, :],
        )


class PoolBackedAttentionKVCache:
    """Lazily gathers cached attention KV from the shared pool during forward.

    Each ``update_and_fetch`` gathers this layer's prefix from the pool
    on demand, keeping operations in the lazy compute graph.  Convert to
    ``ContiguousAttentionKVCache`` via ``to_contiguous`` after the forward pass.
    """

    __slots__ = (
        "_pool",
        "_layer_idx",
        "_slots",
        "offset",
        "_full_keys",
        "_full_values",
        "_new_keys",
        "_new_values",
    )

    def __init__(
        self,
        pool: MlxAttentionKVPool,
        layer_idx: int,
        slots: mx.array,
        prefix_len: int,
    ):
        self._pool = pool
        self._layer_idx = layer_idx
        self._slots = slots
        self.offset = prefix_len
        self._full_keys: mx.array | None = None
        self._full_values: mx.array | None = None
        self._new_keys: mx.array | None = None
        self._new_values: mx.array | None = None

    @property
    def keys(self) -> mx.array | None:
        return self._full_keys

    @property
    def values(self) -> mx.array | None:
        return self._full_values

    @property
    def state(self):
        if self._full_keys is not None:
            return (self._full_keys, self._full_values)
        return ()

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        return make_attention_mask(
            N, self.offset, return_array=return_array, window_size=window_size
        )

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Gather cached prefix from pool, concatenate with new K/V."""
        S = keys.shape[2]

        if self.offset > 0:
            k_cached, v_cached = self._pool.get_kv(
                self._layer_idx, self._slots[: self.offset]
            )
            # Pool layout (S, n_kv_heads, head_dim) → cache (1, n_kv_heads, S, head_dim)
            k_cached = k_cached.transpose(1, 0, 2)[None]
            v_cached = v_cached.transpose(1, 0, 2)[None]
            k_all = mx.concatenate([k_cached, keys], axis=2)
            v_all = mx.concatenate([v_cached, values], axis=2)
        else:
            k_all = keys
            v_all = values

        self.offset += S
        self._full_keys = k_all
        self._full_values = v_all
        self._new_keys = keys
        self._new_values = values
        return k_all, v_all

    def to_contiguous(self, max_seq_len: int = 4096) -> ContiguousAttentionKVCache:
        """Convert to contiguous attention KV reusing forward-pass arrays."""
        cache = ContiguousAttentionKVCache(max_seq_len=max_seq_len)
        if self._full_keys is not None:
            cache.update_and_fetch(self._full_keys, self._full_values)
        return cache
