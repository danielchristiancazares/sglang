# Brief: batched 4-bit dequant-matmul for Apple M1 Max

One kernel, one number. Everything below is measured on this machine, not
assumed. The prize is roughly **+8 to +12 tok/s** end to end; nothing else
found in this codebase's optimization sweep is in that range.

## The problem

Compute `y = x @ dequant(W).T` where

- `x`: `[N, K]` bfloat16, **N in 3..16, N=4 is the case that matters**
- `W`: affine-quantized 4-bit, group size 64, `[M, K]` logical
- `y`: `[N, M]`, same dtype as `x`

This is the verify step of MTP speculative decoding (`N = depth + 1`, depth 3).
Every quantized linear in the model runs at this N during a verify forward.

**Target: N=4 at ≤1.3× the cost of an N=1 matvec. Today it costs 1.8–2.6×.**

The weight read is *identical* at every N — the same 50 MB — so N=4 should
cost nearly the same as N=1. That it costs 1.8× is the entire opportunity.

## Weight format (verified against MLX, get this right first)

```
w:       uint32[M, K/8]      8 consecutive k per word, LITTLE-endian nibbles
scales:  bfloat16[M, K/64]
biases:  bfloat16[M, K/64]

W[m][k] = ((w[m][k/8] >> (4*(k%8))) & 0xF) * scales[m][k/64] + biases[m][k/64]
```

Group size 64 and 8 values per word means **a word never straddles a group**,
which licenses the per-word fold used by the current kernel:

```
contribution = scale * Σ_j(nibble_j * x_j) + bias * Σ_j(x_j)
```

Output is row-major `[N, M]`: `out[n * M + m]`. Reference is
`mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)`.

## Hardware constraints (this is not an H100)

`mx.device_info()` reports `applegpu_g13s` — Apple7 family, M1 Max, 32 GPU cores.

- **No `cp.async`.** No asynchronous global→threadgroup copy. Every
  double-buffering scheme has to be built out of ordinary loads and barriers.
- **32 KB threadgroup memory**, not 96–164 KB.
- `simdgroup_matrix<T,8,8>` exists (the `mma.sync` analogue) but is 8×8 and
  its fragment semantics differ from CUDA's. **Verify whether
  `simdgroup_bfloat8x8` is supported on Apple7** — M1 predates native bf16
  arithmetic. If it isn't, dequantize to `half` or `float` instead; bf16→float
  is a 16-bit left shift and free, but costs register/threadgroup space.
- Simdgroup width 32 (same as a CUDA warp — the one thing that maps cleanly).
- Threadgroup size is register-limited in practice: a ~34-register kernel here
  was capped at 384 threads, so 512-thread configs fail to launch.

Consequence: **Marlin/Machete cannot be ported, only re-derived.** Their three
load-bearing primitives (`cp.async`, `mma.sync`/`ldmatrix` fragments, large
shared memory) are all absent or different. There is also no CUDA source in
this tree to port from — the Apple path is MLX end to end.

## Baseline to beat

Cost as a multiple of the same shape's N=1 matvec, serially timed on real
weights. `v3` is the current production kernel (`mtp_spec.py`).

MLP gate `[17408, 5120]`, 128 instances/token, 50 MB, roofline 136 µs:

| N | MLX | v3 (current) | MLX GB/s |
|---:|---:|---:|---:|
| 1 | 1.01× | — | 213 |
| 4 | 2.53× | **1.82×** | 85 |
| 6 | 6.06× | **2.38×** | 35 |
| 8 | 6.08× | **3.23×** | 35 |
| 16 | 6.09× | — | 35 |

`lm_head` `[248320, 5120]`, 715 MB, roofline 1943 µs:

| N | MLX | v3 | MLX GB/s |
|---:|---:|---:|---:|
| 1 | 1.00× | — | **352** |
| 4 | 3.83× | **2.58×** | 92 |
| 8 | 9.23× | **4.82×** | 38 |
| 16 | 9.27× | — | 38 |

Two facts hide in that table.

1. **MLX's N=1 matvec is excellent** — 352 GB/s against a measured 368 GB/s
   streaming ceiling, 96%. Do not try to beat N=1; four attempts failed here.
2. **MLX's `qmm` path (N≥6) is a cliff upward** — flat 6.1–9.3× from N=6 to
   N=16, pinned at 33–38 GB/s on *every* shape from 29 MB to 715 MB. A fixed
   ~1/10-of-ceiling result that ignores problem size smells like structure, not
   tuning. It is tuned for large N and pays a fixed cost that dominates here.

## Why the current kernel is stuck at 1.8× (the actual diagnosis)

It is **ALU-bound, not bandwidth-bound**, and that is why "load wider" kept
failing. Per 32-bit weight word `v3` does ~16 unpack ops plus 10 ops per
column, so ~`16 + 10N` ops per 4 bytes loaded.

For the gate shape (11.1M words) at ~5.3e12 op/s:

| N | ALU time | Bandwidth floor | Binding |
|---:|---:|---:|---|
| 4 | ~118 µs | 136 µs | balanced — perfect overlap required just to hit roofline |
| 8 | ~202 µs | 136 µs | **ALU** |

Measured v3 at N=4 is 427 µs, i.e. 3.1× the 136 µs floor: neither pure-ALU nor
pure-bandwidth, but poor overlap between the two.

**The fix has to cut ops per byte, not add register residency.** Dequantize
each word *once* into a threadgroup tile (~16 ops/word, independent of N) and
do the multiply with `simdgroup_matrix`, which retires 512 MACs per
instruction. That drops the `10N` term entirely: at N=4 ALU time falls to
~34 µs against a 136 µs bandwidth floor, which is the only regime where
roofline is reachable.

Cost of that trade: a dequantized bf16/half tile is 4× the bytes of the packed
form in threadgroup memory (32 KB budget → e.g. 64 rows × 256 k), and N=4
wastes half of each 8-wide matrix — which is free, because the binding
constraint is bandwidth, not math.

## Four approaches already tried and beaten (do not repeat)

| | idea | result |
|---|---|---|
| v3 | scalar FMA, one word/lane/tile, nibbles unpacked once per row and reused across columns | **current winner**, 1.82× at N=4 |
| v6 | wide `uint4` weight loads, 32 k of x register-resident per row | wins at N≤2, **catastrophic register spill at N≥3** |
| v7 | wide loads with bounded live set, weight `uint4` held across four sub-steps | **still spills at N≥3** |
| v8 | threadgroup-staged *activations* with barriers | **4.2× worse** — barriers not amortized by enough math |

Pattern: at N≥3 registers are the binding constraint, and every scheme that
buys speed with register residency loses. Note v8 staged the *activations*;
staging *dequantized weights* to feed matrix instructions is a different
design whose barrier cost amortizes over 8× more math.

## Suggested order of attack

**Step 0, cheapest and possibly decisive:** read MLX's own kernel source
(`mlx/backend/metal/kernels/quantized.h`) and find out *why* `qmm` has a flat
6–9× floor. If that fixed cost is a tiling or launch artifact rather than
real work, the whole problem may reduce to fixing the dispatch, with no new
math. This is an hour of reading against days of kernel writing.

**Step 1:** `simdgroup_matrix` design above — dequant once into threadgroup
memory, pad N to 8, double-buffer the dequant against the matrix math using
ordinary loads plus barriers (no `cp.async` available).

**Step 2:** if that lands, re-tune speculation depth. Depth 3 is optimal
*given the current cost curve*; a flatter curve makes depth 5–7 profitable,
which multiplies the win. Expected tokens/round is ~3.1 at depth 3 and ~4.9 at
depth 5 in the high-acceptance regime.

## Acceptance criteria

1. **Correct:** relative error < 5e-3 vs `mx.quantized_matmul` on real weights.
   (Exact bit-match is not expected or required — fp32 accumulation order
   differs, and the greedy stream tolerates it; verify catches any drift by
   construction.)
2. **Fast:** N=4 ≤ 1.3× the N=1 column, on `gate`, `down`, and `lm_head`.
   Compare ratios only within a single harness run — freshly allocated weights
   measure ~270 GB/s at N=1 where the same shape inside the loaded 15 GB model
   measures ~213 GB/s, so the synthetic baseline is optimistic and its ratios
   read worse (MLX N=4 is 3.21× synthetic, 2.53× in-model). The table in this
   brief is in-model.
3. **Handles the awkward shapes:** M ranges 48 → 248320 and is not always a
   multiple of the row tile; K ∈ {5120, 6144, 17408}, all divisible by 1024.
4. **Real:** confirmed end to end, not just in the harness. A kernel that won
   every micro-benchmark here still lost 17% in the decode loop.

## Tooling and integration

Benchmark: `python scripts/bench_batched_qmv.py --candidate mymod:matmul`.
Read its docstring — the serial-chaining and min-of-iters protocol is load
bearing, not stylistic.

Drop-in point: `python/sglang/srt/hardware_backend/mlx/mtp_spec.py`

- `_build_batched_qmv(N, K, G, M, rsg, nsg)` — returns an
  `mx.fast.metal_kernel`, `lru_cache`d per shape.
- `_batched_qlinear_call` — dispatches for `3 ≤ n ≤ 16`, affine, 4-bit,
  `K % 256 == 0`; tiles come from `_tuned_tile(n)`.

Keep the interface, replace the source. **Use `mx.fast.metal_kernel`, not a
pybind11/ctypes binding** — a foreign boundary forces MLX arrays to
materialize and breaks the lazy graph, and a per-token sync was independently
measured to cost 16% here.

Validate:

1. `pytest test/registered/unit/hardware_backend/mlx/ -q` — currently 240 pass.
2. End to end: launch the server with and without `--mlx-mtp-path` per the
   recipe in `BENCHMARK.md` ("Speculative decoding profile"). Numbers to beat
   are **19.88 tok/s** on the 768-token coding generation (flag on) versus
   **19.05** (flag off).

## Reality check

The N=1 decode path — 96% of real-world time — is closed. MLX runs it at 86%
of roofline in-graph and 96% of streaming ceiling on `lm_head`. Only the
batched verify has visible slack. And even a perfect kernel is bounded by two
non-kernel costs already measured: scheduler overhead serializing with GPU
work (~4–5 ms/round) and draft acceptance sitting near breakeven on prose
stretches. Those set the ceiling on what any kernel can deliver.
