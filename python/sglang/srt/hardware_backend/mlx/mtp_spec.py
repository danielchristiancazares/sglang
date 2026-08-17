"""Multi-token-prediction speculative decoding for the MLX backend.

Qwen3.8 ships a one-block MTP head as an ``mtp.safetensors`` sidecar. Each
round drafts ``depth`` tokens with that head, then verifies all of them in a
single ``depth+1``-token trunk forward. A draft is kept only when it equals
the trunk's own greedy pick at that position, so emitted text is a greedy
stream of the trunk (verify logits run through the batched Metal matvec,
whose fp32 accumulation order differs from MLX's single-token kernel at
~1e-3 relative — the same numerics class as chunked prefill).

Engine-level wins on an M1 Max (paired in-process, August 2026): +1.4 to
+6.3 tok/s on coding-agent generation, ~breakeven on the adversarial dense-
identifier control, with the adaptive policy bounding hostile prompts.

Constraints (validated by the runner at startup):
  * greedy requests only, no logit edits/hooks/logprobs, one running request;
  * radix cache disabled — a prefix hit skips trunk computation, so there are
    no hidden states to teacher-force the head with;
  * the trunk must be a Qwen3.5-family hybrid (the head is one full-attention
    ``DecoderLayer`` and rollback assumes Gated DeltaNet auxiliary layers).

Rollback: attention layers trim their cache offsets; Gated DeltaNet layers
cannot slice their recurrent state, so the verify forward records each
layer's ``gated_delta_update`` inputs and rollback re-runs the recurrence
over the accepted prefix only (projections are not recomputed).

The recording forward mirrors mlx-lm's ``GatedDeltaNet.__call__`` (0.31.x);
``test_mtp_spec.py`` asserts it stays numerically identical to stock.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import time
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import msgspec
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_5 import DecoderLayer, GatedDeltaNet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batched affine-q4 matvec for the verify forward.
#
# MLX's qmv costs ~N x a single matvec for 2 <= N <= 5 (no weight-read
# amortization); this kernel reads each weight word once and applies it to
# all N columns, making a depth-3 verify ~2.4 AR steps instead of ~3.1.
# ---------------------------------------------------------------------------
_KERNEL_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
"""

KERNEL_MIN_N = 3
KERNEL_MAX_N = 16


@functools.lru_cache(maxsize=None)
def _build_batched_qmv(N: int, K: int, G: int, M: int, rsg: int, nsg: int):
    rows = nsg * rsg
    src = f"""
    constexpr uint N = {N};
    constexpr uint K = {K};
    constexpr uint G = {G};
    constexpr uint M = {M};
    constexpr uint RSG = {rsg};
    constexpr uint ROWS = {rows};
    constexpr uint KW = K / 8;
    constexpr uint KG = K / G;
    constexpr uint KT = 256;              // 32 lanes * 8 k per lane

    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint row_base = tgid * ROWS + sg * RSG;

    float acc[RSG][N];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < RSG; ++r)
        #pragma clang loop unroll(full)
        for (uint n = 0; n < N; ++n) acc[r][n] = 0.0f;

    for (uint kt0 = 0; kt0 < K; kt0 += KT) {{
        uint k0   = kt0 + lane * 8u;
        uint widx = k0 >> 3;
        uint g    = k0 / G;

        float xv[N][8];
        float sx[N];
        #pragma clang loop unroll(full)
        for (uint n = 0; n < N; ++n) {{
            const device T* xr = x + (size_t)n * K + k0;
            float s = 0.0f;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {{ float v = (float)xr[j]; xv[n][j] = v; s += v; }}
            sx[n] = s;
        }}

        #pragma clang loop unroll(full)
        for (uint r = 0; r < RSG; ++r) {{
            uint m = row_base + r;
            if (m >= M) continue;
            uint32_t word = w[(size_t)m * KW + widx];

            float nib[8];               // unpacked once, reused for every n
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) nib[j] = (float)((word >> (4u * j)) & 0xFu);

            float sc = (float)scales[(size_t)m * KG + g];
            float bi = (float)biases[(size_t)m * KG + g];
            #pragma clang loop unroll(full)
            for (uint n = 0; n < N; ++n) {{
                float sn = 0.0f;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j) sn = fma(nib[j], xv[n][j], sn);
                acc[r][n] = fma(sc, sn, fma(bi, sx[n], acc[r][n]));
            }}
        }}
    }}

    #pragma clang loop unroll(full)
    for (uint r = 0; r < RSG; ++r) {{
        uint m = row_base + r;
        #pragma clang loop unroll(full)
        for (uint n = 0; n < N; ++n) {{
            float v = simd_sum(acc[r][n]);
            if (lane == 0 && m < M) out[(size_t)n * M + m] = (T)v;
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"sglang_mtp_bqmv_{N}_{K}_{G}_{M}_{rsg}_{nsg}",
        input_names=["x", "w", "scales", "biases"],
        output_names=["out"],
        header=_KERNEL_HEADER,
        source=src,
    )


_ORIG_QLINEAR_CALL = nn.QuantizedLinear.__call__


def _tuned_tile(n: int) -> tuple[int, int]:
    return (4, 4) if n >= 6 else (4, 8)


def _batched_qlinear_call(self, x):
    shape = x.shape
    n = 1
    for d in shape[:-1]:
        n *= d
    if not (
        KERNEL_MIN_N <= n <= KERNEL_MAX_N
        and self.mode == "affine"
        and self.bits == 4
        and self.biases is not None
        and shape[-1] % 256 == 0
    ):
        return _ORIG_QLINEAR_CALL(self, x)

    rsg, nsg = _tuned_tile(n)
    K = shape[-1]
    M = self.weight.shape[0]
    rows = rsg * nsg
    ntg = (M + rows - 1) // rows
    kern = _build_batched_qmv(n, K, self.group_size, M, rsg, nsg)
    (out,) = kern(
        inputs=[x.reshape(n, K), self.weight, self.scales, self.biases],
        template=[("T", x.dtype)],
        grid=(ntg * nsg * 32, 1, 1),
        threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(n, M)],
        output_dtypes=[x.dtype],
    )
    out = out.reshape(*shape[:-1], M)
    if "bias" in self:
        out = out + self["bias"]
    return out


@contextlib.contextmanager
def batched_matmul(enabled: bool):
    if not enabled:
        yield
        return
    nn.QuantizedLinear.__call__ = _batched_qlinear_call
    try:
        yield
    finally:
        nn.QuantizedLinear.__call__ = _ORIG_QLINEAR_CALL


# ---------------------------------------------------------------------------
# Gated DeltaNet recording forward + rollback.
# ---------------------------------------------------------------------------
class GdnFrame(msgspec.Struct):
    cache: Any
    conv_input: mx.array
    q: mx.array
    k: mx.array
    v: mx.array
    a: mx.array
    b: mx.array
    state_in: Any
    mask: Any
    A_log: mx.array
    dt_bias: mx.array
    n_keep: int


_RECORD: list[GdnFrame] | None = None
_ORIG_GDN_CALL = GatedDeltaNet.__call__


def _recording_gdn_call(self, inputs, mask=None, cache=None):
    B, S, _ = inputs.shape
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
        )
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    n_keep = self.conv_kernel_size - 1
    if cache is not None:
        cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(self.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]

    state_in = cache[1] if cache else None
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    out, state = gated_delta_update(
        q, k, v, a, b, self.A_log, self.dt_bias, state_in, mask,
        use_kernel=not self.training,
    )
    if cache is not None:
        cache[1] = state
        cache.advance(S)

    if _RECORD is not None:
        _RECORD.append(GdnFrame(
            cache=cache, conv_input=conv_input, q=q, k=k, v=v, a=a, b=b,
            state_in=state_in, mask=mask, A_log=self.A_log, dt_bias=self.dt_bias,
            n_keep=n_keep,
        ))

    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


@contextlib.contextmanager
def recording_gdn():
    global _RECORD
    GatedDeltaNet.__call__ = _recording_gdn_call
    _RECORD = []
    try:
        yield _RECORD
    finally:
        _RECORD = None
        GatedDeltaNet.__call__ = _ORIG_GDN_CALL


def rollback_gdn(frames: list[GdnFrame], keep: int) -> None:
    """Restore every recorded GDN layer to its state after ``keep`` tokens."""
    for f in frames:
        mask = f.mask[:, :keep] if f.mask is not None else None
        _, state = gated_delta_update(
            f.q[:, :keep], f.k[:, :keep], f.v[:, :keep], f.a[:, :keep],
            f.b[:, :keep], f.A_log, f.dt_bias, f.state_in, mask,
        )
        f.cache[1] = state
        f.cache[0] = mx.contiguous(f.conv_input[:, keep:keep + f.n_keep, :])


# ---------------------------------------------------------------------------
# The MTP head.
# ---------------------------------------------------------------------------
class MtpHead(nn.Module):
    """One MTP block: fc over [embedding; hidden], one decoder layer, a norm.

    Contract (mtplx_runtime.json): the head consumes the trunk's post-norm
    hidden and the embedding of the NEXT token, concatenated embedding-first,
    and recurses on its own normed output. The sidecar's norm weights are
    already in mlx-lm's shifted convention — load with ``shift_norms=False``.
    """

    def __init__(self, args: Any, eps: float = 1e-6):
        super().__init__()
        d = args.hidden_size
        # layer_idx chosen so DecoderLayer builds the full-attention variant.
        self.layer = DecoderLayer(args, args.full_attention_interval - 1)
        if self.layer.is_linear:
            raise ValueError("MTP block must be a full-attention layer")
        self.fc = nn.Linear(2 * d, d, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(d, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(d, eps=eps)
        self.norm = nn.RMSNorm(d, eps=eps)

    def __call__(self, embeddings, hidden, cache=None, mask=None):
        z = mx.concatenate(
            [self.pre_fc_norm_embedding(embeddings), self.pre_fc_norm_hidden(hidden)],
            axis=-1,
        )
        z = self.fc(z)
        z = self.layer(z, mask=mask, cache=cache)
        return self.norm(z)


_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "mtp.norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def load_mtp_head(
    path: str, args: Any, *, shift_norms: bool = False, eps: float = 1e-6
) -> MtpHead:
    """Build an MtpHead from an ``mtp.safetensors`` sidecar."""
    raw = mx.load(str(path))
    weights = {}
    for key, value in raw.items():
        if shift_norms and value.ndim == 1 and any(
            key.endswith(s) for s in _NORM_SUFFIXES
        ):
            value = value + 1.0
        weights[key[len("mtp."):] if key.startswith("mtp.") else key] = value

    head = MtpHead(args, eps=eps)
    tree = {}
    for key, value in weights.items():
        if key.startswith("layers.0."):
            tree["layer." + key[len("layers.0."):]] = value
        else:
            tree[key] = value
    head.load_weights(list(tree.items()))
    head.eval()
    return head


# ---------------------------------------------------------------------------
# Adaptive policy: speculate only while it wins.
# ---------------------------------------------------------------------------
class MtpSpecConfig(msgspec.Struct, frozen=True, kw_only=True):
    head_path: str
    depth: int = 3
    quantize_head: bool = True
    window: int = 12
    # Breakeven is ~2.9 accepted tokens/round (round ~152 ms vs ~52 ms AR
    # steps). Trip only clearly below it: a marginal stretch costs a few
    # percent, while a false trip forfeits the high-acceptance regions that
    # follow (measured on tool-JSON: two trips locked half the generation
    # into AR fallback and erased a 1.3x win).
    off_thresh: float = 2.75
    # Short AR stretches: a losing stretch sits near breakeven, so probing
    # again after 32 tokens is nearly free, while late re-engagement forfeits
    # whole high-acceptance regions.
    ar_run: int = 32
    ar_run_cap: int = 128
    ar_chunk: int = 8


class AdaptivePolicy:
    """Trip to AR when the trailing mean tokens/round drops below breakeven.

    Breakeven for a depth-3 round is ~3.0 accepted tokens (the verify forward
    costs ~2.4 AR steps plus drafting); the threshold sits just below it
    because a false trip forfeits real upside while a true trip only saves a
    small deficit. Consecutive trips double the AR stretch (capped) so a
    hostile stream settles into long AR runs with rare probes; a winning
    window resets the backoff.
    """

    def __init__(self, *, window: int, off_thresh: float, ar_run: int,
                 ar_run_cap: int):
        self._window = window
        self._off_thresh = off_thresh
        self._ar_run = ar_run
        self._ar_run_cap = ar_run_cap
        self._recent: list[int] = []
        self._stretch = ar_run
        self.ar_budget = 0

    def note_round(self, accepted: int) -> None:
        self._recent.append(accepted)
        if len(self._recent) < self._window:
            return
        mean = sum(self._recent[-self._window:]) / self._window
        if mean >= self._off_thresh + 0.4:
            self._stretch = self._ar_run          # winning again: reset backoff
        elif mean < self._off_thresh:
            self.ar_budget = self._stretch
            self._stretch = min(self._stretch * 2, self._ar_run_cap)
            self._recent = []

    def in_ar_mode(self) -> bool:
        return self.ar_budget > 0

    def note_ar_tokens(self, n: int) -> None:
        self.ar_budget = max(0, self.ar_budget - n)


# ---------------------------------------------------------------------------
# Per-request state and the engine.
# ---------------------------------------------------------------------------
class _InflightRound(msgspec.Struct):
    """A launched-but-unsynced speculative round (graph already async_eval'd)."""

    fed: mx.array
    ys: mx.array
    vh: mx.array
    frames: list[GdnFrame]
    head_base: int


class _ReqSpecState:
    __slots__ = ("head_cache", "pend_h", "pend_t", "buffer", "policy", "inflight",
                 "rounds", "accepted", "ar_tokens", "round_seconds")

    def __init__(self, policy: AdaptivePolicy):
        self.head_cache = KVCache()
        self.pend_h: mx.array | None = None
        self.pend_t: mx.array | None = None
        self.buffer: list[int] = []
        self.policy = policy
        self.inflight: _InflightRound | None = None
        self.rounds = 0
        self.accepted = 0
        self.ar_tokens = 0
        self.round_seconds = 0.0


class MlxMtpSpecEngine:
    """Buffered MTP speculation behind the one-token-per-step decode contract.

    The runner calls :meth:`decode` once per scheduler step; internally the
    engine pops a buffered token, or refills the buffer with one speculative
    round (or a short AR chunk while the adaptive policy holds speculation
    off). All model access goes through the narrow callables supplied at
    construction, so the engine never reaches into the runner.
    """

    def __init__(
        self,
        *,
        trunk_step: Callable[[str, mx.array], mx.array],
        embed: Callable[[mx.array], mx.array],
        lm_head: Callable[[mx.array], mx.array],
        trim_attention: Callable[[str, int], None],
        model_args: Any,
        config: MtpSpecConfig,
    ):
        self._trunk_step = trunk_step
        self._embed = embed
        self._lm_head = lm_head
        self._trim_attention = trim_attention
        self._config = config
        self._head = load_mtp_head(config.head_path, model_args)
        if config.quantize_head:
            # The head is read in full every draft step; q4 cuts that ~4x.
            # Drafts are verified exactly, so this only moves acceptance.
            nn.quantize(self._head, group_size=64, bits=4)
        mx.eval(self._head.parameters())
        self._reqs: dict[str, _ReqSpecState] = {}
        logger.info(
            "MLX MTP speculation loaded: depth %d, head %s (q4=%s)",
            config.depth, config.head_path, config.quantize_head,
        )

    # -- request lifecycle ---------------------------------------------------
    def register(self, req_id: str) -> None:
        self._reqs[req_id] = _ReqSpecState(AdaptivePolicy(
            window=self._config.window,
            off_thresh=self._config.off_thresh,
            ar_run=self._config.ar_run,
            ar_run_cap=self._config.ar_run_cap,
        ))

    def has_request(self, req_id: str) -> bool:
        return req_id in self._reqs

    def release(self, req_id: str) -> None:
        self._reqs.pop(req_id, None)

    def clear(self) -> None:
        self._reqs.clear()

    # -- prefill -------------------------------------------------------------
    def observe_prefill_chunk(
        self, req_id: str, hidden: mx.array, token_ids: list[int]
    ) -> None:
        """Teacher-force the head over one prefill chunk.

        ``hidden``: post-norm trunk hiddens for the chunk's positions.
        ``token_ids``: the chunk's tokens (same positions). Head position p
        pairs hidden_p with token_{p+1}; the chunk's last hidden stays
        pending until the next chunk (or the first generated token) supplies
        its successor.
        """
        st = self._reqs[req_id]
        if st.pend_h is not None:
            st.pend_h = mx.concatenate([st.pend_h, hidden], axis=1)
            st.pend_t = mx.concatenate(
                [st.pend_t, mx.array([token_ids], dtype=mx.int32)], axis=1
            )
        else:
            st.pend_h = hidden
            st.pend_t = mx.array([token_ids[1:]], dtype=mx.int32)
        # Feed every completed (hidden, next-token) pair; keep the last
        # hidden pending — its next token is not known yet.
        n_pairs = st.pend_t.shape[1]
        if n_pairs >= 1:
            self._head_step(st, st.pend_h[:, :n_pairs], st.pend_t)
        st.pend_h = st.pend_h[:, n_pairs:]
        st.pend_t = mx.zeros((1, 0), dtype=mx.int32)
        if st.head_cache.keys is not None:
            # Head prefill belongs to the prefill phase; kick it off now so
            # the first decode round does not pay for it.
            mx.async_eval(st.head_cache.keys, st.head_cache.values)

    # -- decode --------------------------------------------------------------
    def decode(self, req_id: str, last_token: int) -> int:
        """Pop one buffered token, running a refill round when empty.

        Rounds run synchronously: launching the next round early (during
        buffer pops) was measured to LOSE ~1-6 ms/round to the extra Metal
        command-buffer split, both at the runner and through the server, so
        the round's single eval stays the only sync point.
        """
        st = self._reqs[req_id]
        if not st.buffer:
            if st.policy.in_ar_mode():
                self._ar_chunk(req_id, st, last_token)
            else:
                t0 = time.perf_counter()
                self._launch_round(req_id, st, last_token)
                self._finish_round(req_id, st)
                st.round_seconds += time.perf_counter() - t0
            if st.rounds > 0 and st.rounds % 24 == 0:
                logger.info(
                    "mtp[%s]: %d rounds, %.2f tok/round, %.1f ms/round, "
                    "%d AR-fallback tokens",
                    req_id[:8], st.rounds, st.accepted / st.rounds,
                    st.round_seconds / st.rounds * 1e3, st.ar_tokens,
                )
        return st.buffer.pop(0)

    # -- internals -----------------------------------------------------------
    def _head_step(
        self, st: _ReqSpecState, hiddens: mx.array, tokens: mx.array
    ) -> mx.array:
        return self._head(self._embed(tokens), hiddens, cache=st.head_cache)

    def _ar_chunk(self, req_id: str, st: _ReqSpecState, last_token: int) -> None:
        budget = min(self._config.ar_chunk, st.policy.ar_budget)
        y = mx.array([[last_token]], dtype=mx.int32)
        toks: list[mx.array] = []
        hs: list[mx.array] = []
        for _ in range(budget):
            h = self._trunk_step(req_id, y)
            hs.append(h[:, -1:])
            y = mx.argmax(self._lm_head(h[:, -1:]), -1).astype(mx.int32)
            toks.append(y)
            mx.async_eval(y)
            if len(toks) > 2:
                mx.eval(toks[-3])
        mx.eval(toks)
        st.pend_h = mx.concatenate([st.pend_h] + hs, axis=1)
        st.pend_t = mx.concatenate([st.pend_t] + toks, axis=1)
        st.buffer.extend(int(t.item()) for t in toks)
        st.ar_tokens += budget
        st.policy.note_ar_tokens(budget)

    def _launch_round(self, req_id: str, st: _ReqSpecState, last_token: int) -> None:
        """Build and async-launch one draft+verify round; sync happens later."""
        depth = self._config.depth
        y = mx.array([[last_token]], dtype=mx.int32)

        # Catch the head up on all pending pairs, then draft. pend_t runs one
        # short of pend_h exactly when the last hidden's successor was still
        # unknown at the previous step — that successor is `last_token`.
        if st.pend_t.shape[1] + 1 == st.pend_h.shape[1]:
            pend_t = mx.concatenate([st.pend_t, y], axis=1)
        else:
            pend_t = st.pend_t
        z = self._head_step(st, st.pend_h, pend_t)[:, -1:]
        head_base = st.head_cache.offset
        drafts: list[mx.array] = []
        for _ in range(depth):
            d = mx.argmax(self._lm_head(z), -1).astype(mx.int32)
            drafts.append(d)
            if len(drafts) < depth:
                z = self._head_step(st, z, d)
        fed = mx.concatenate([y] + drafts, axis=1)
        mx.async_eval(fed)   # drafts start on-GPU while the verify graph builds

        # Verify all depth+1 positions in one batched trunk forward. Both
        # patches act at graph-build time.
        with recording_gdn() as frames, batched_matmul(True):
            vh = self._trunk_step(req_id, fed)
            ys = mx.argmax(self._lm_head(vh), -1).astype(mx.int32)
        mx.async_eval(ys)
        st.inflight = _InflightRound(
            fed=fed, ys=ys, vh=vh, frames=frames, head_base=head_base
        )

    def _finish_round(self, req_id: str, st: _ReqSpecState) -> None:
        """Sync a launched round, accept/rollback, and refill the buffer."""
        depth = self._config.depth
        rnd = st.inflight
        st.inflight = None
        mx.eval(rnd.ys)   # usually already evaluated by the time pops drain

        fl = rnd.fed[0].tolist()
        yl = rnd.ys[0].tolist()
        n_acc = 0
        while n_acc < depth and fl[n_acc + 1] == yl[n_acc]:
            n_acc += 1
        keep = n_acc + 1

        if keep < len(fl):
            rollback_gdn(rnd.frames, keep)
            self._trim_attention(req_id, len(fl) - keep)
        st.head_cache.trim(st.head_cache.offset - rnd.head_base)

        st.buffer.extend(fl[1:keep])
        st.buffer.append(yl[n_acc])
        st.pend_h = rnd.vh[:, :keep]
        st.pend_t = mx.concatenate(
            [rnd.fed[:, 1:keep], rnd.ys[:, n_acc:n_acc + 1]], axis=1
        )
        st.rounds += 1
        st.accepted += keep
        st.policy.note_round(keep)
