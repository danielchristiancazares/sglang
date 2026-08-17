# Performance Log

## Current Active Timings

| Benchmark | Baseline | Current | Delta | Command | Last Updated |
|---|---:|---:|---:|---|---|
| Qwen3.8-27B Q4_0, 8 concurrent requests, 32 output tokens each | 32.953 TPS | 38.016 TPS | +5.063 TPS | `.venv-mac-metal/bin/python benchmark/mac/bench_sglang_sampling.py --concurrency 8 --output-tokens 32` | 2026-08-16 22:26 PDT |
| Qwen3.8-27B Q4_0, batch 24, 128 output tokens each, real top-k/top-p sampling | 49.500 TPS | **62.034 TPS** | **+12.534 TPS** | `.venv-mac-metal/bin/python benchmark/mac/bench_sglang_batched_request.py --url http://127.0.0.1:30001/generate --batch-size 24 --output-tokens 128` | 2026-08-16 22:35 PDT |

Target: at least **60 TPS** with real sampling. Achieved with a three-run end-to-end median of **62.034 TPS**; warmed decode windows sustain **72.15–72.83 TPS**.

## Baseline

- Commit: `b270c6521ced7af70c6ff8d4740f89f752a3afd2` plus the existing dirty MPS/Metal port in this worktree.
- Hardware / OS: MacPro7,1; AMD Radeon Pro W6900X 32 GB; macOS 26.6 (25G72); x86_64.
- Runtime: Python 3.11 virtual environment `.venv-mac-metal`; PyTorch 2.2.2; MPS available.
- Model: `/Users/daniel/models/Qwen3.8-27B-Q4_0/Qwen3.8-27B-Q4_0.gguf` with tokenizer `/Users/daniel/models/Qwen3.8-27B-tokenizer`.
- Server: `python -m sglang.launch_server` with GGUF loading, float32 MPS execution, torch-native attention, PyTorch sampling, language-model-only mode, radix cache disabled, 24 maximum running requests, 4096 maximum total tokens, and 24 Mamba cache slots.
- Workload: eight barrier-synchronized `/generate` requests; each has a 16-token plain-text prompt, temperature 0.8, top-p 0.9, 32 forced output tokens, and EOS ignored. Aggregate TPS is 256 output tokens divided by client-observed wall time.
- Warmup policy: discard server/model/Metal first-use compilation run, then retain five consecutive samples from the already-loaded server.
- Discarded first-use run: `6.874 TPS` (`37.240 s`).
- Raw retained TPS: `32.094`, `32.931`, `32.953`, `32.977`, `33.009`.
- Raw retained wall times: `7.977 s`, `7.774 s`, `7.769 s`, `7.763 s`, `7.755 s`.
- Median: `32.953 TPS`; median wall time `7.769 s`.

## Deltas

### 2026-08-16 20:23 PDT - BASELINE

- Change: measurement only; no new optimization applied.
- Benchmark evidence: warmed five-sample median `32.953 TPS` from raw values above.
- Correctness evidence: all eight responses returned 32 output token IDs and successful HTTP status; generated samples were coherent continuations.
- Decision: baseline accepted. Required success threshold is `>=34.953 TPS` under the same warmed repeated protocol.
- Commit: pending with the surrounding MPS port.

## Candidate Inventory

| ID | Hypothesis | Scope | Status | Evidence |
|---|---|---|---|---|
| PERF-001 | Add a Q4_0 batch-8 Metal specialization so steady batch-8 decode reads each packed matrix once instead of twice. | `gguf_q4_0.mm` Q4_0 kernel and host dispatch | Rejected | Correct but regressed representative MLP Q4_0 median from `0.345 ms` to `0.778 ms`; removed. See `FAILED_PATHS.md`. |
| PERF-002 | Fuse or remove remaining GDN pack, normalization, and reorder launches. | Qwen3.5 GDN MPS path and native Metal extension | Pending profile | Production path launches native packing and gated-norm/reorder around native recurrent attention for most decoder layers. |
| PERF-003 | Reduce full-vocabulary PyTorch sampling overhead on MPS. | `sampler.py` / native sampling | Pending profile | Every decode step performs top-p sampling over the full vocabulary; impact relative to model kernels remains unmeasured. |
| PERF-004 | Remove proven-redundant `.contiguous()` conversions and metadata copies in native MPS wrappers. | `mps/ops.py`, attention and GDN callers | Pending trace | Calls are production-reachable, but views may already be contiguous and therefore free. |
| PERF-005 | Wire the dormant Metal Q/K norm + RoPE + QKV/gate preparation kernel into full-attention layers. | `qwen3_5.py`, `gguf_q4_0.mm` | Rejected | Isolated batch-8 preparation fell from `2.225 ms` to `0.162 ms`, but clean end-to-end median regressed from `32.309` to `30.680 TPS`; production wiring removed. |
| PERF-006 | Reduce Q5_K/Q6_K batch-8 accumulator pressure by reusing smaller or batch-24 tiles. | `gguf_q4_0.mm` quantized matmul dispatch | Rejected | Q5_K tile-4 regressed `0.809 -> 0.888 ms`; Q6_K tile-4 regressed `27.227 -> 32.446 ms`; batch-24 vec4 regressed Q6_K to `34.166 ms`. |
| PERF-007 | Vectorize Q6_K dequantization across four adjacent weights with four two-request SIMD subgroups. | `gguf_q4_0.mm` Q6_K batch-8 kernel | Retained | LM-head microbenchmark improved `27.227 -> 9.186 ms` (`35.7 -> 105.7 GiB/s`); Q6_K reference relative error `4.64191e-07`. |
| PERF-008 | Apply the same vec4/subgroup geometry to the repeated Q5_K GDN output projections. | `gguf_q4_0.mm` Q5_K batch-8 kernel | Retained | Representative projection improved `0.809 -> 0.356 ms` (`24.9 -> 56.6 GiB/s`); combined end-to-end median is `38.016 TPS`. |
| PERF-009 | Apply batch-subgroup reuse or alter unroll depth in the Q4_0 batch-eight kernel. | `gguf_q4_0.mm` Q4_0 kernels | Rejected | Four-subgroup vec4 regressed `0.350 -> 0.686 ms`; existing split kernel gave `0.390 ms`; unroll 2 and 8 gave `0.402` and `0.405 ms`. |
| PERF-010 | Fuse GDN projection packing with the decode causal convolution. | `gguf_q4_0.mm` GDN glue kernels | Rejected | Correct fused kernel measured `0.151 ms` versus `0.147 ms` for the separate chain; experimental kernel and benchmark removed. |
| PERF-011 | Coalesce near-simultaneous requests into one idle prefill batch. | MPS normal scheduler loop | Rejected | A 2 ms window produced `4 + 4` prefills and `37.876 TPS`; even a single prebatched size-eight request reached only `39.053 TPS`. |
| PERF-010 | Use the checkpoint's bundled NEXTN block for speculative decoding. | SGLang speculative control plane + MPS GDN state verify | Functional, rejected for throughput | Fully served coherent sampled output; batch-1 measured `4.872 TPS`, with accept length `2.80/4` and draft acceptance rate `0.60`. |
| PERF-011 | Vectorize exact-batch-24 Q6_K LM-head dequantization. | `gguf_q4_0.mm` Q6_K kernel | Retained | `57.343 -> 21.565 ms`; reference relative error `3.36184e-07`. |
| PERF-012 | Extend vectorized Q5_K projection to exact batch 24. | `gguf_q4_0.mm` Q5_K kernel | Retained | `2.081 -> 0.891 ms` per GDN projection; reference relative error `3.81316e-07`; final median `62.034 TPS`. |

### 2026-08-16 20:29 PDT - PERF-001

- Change: temporarily instantiated `q4_0_small_batch_impl<8>` and selected it for decode batches above six, reducing batch-eight Y groups from two to one.
- Benchmark evidence: representative `blk.8.ffn_gate.weight` Q4_0 shape `(5120, 17408)`, batch eight. Existing tile-four raw medians were `0.345`, `0.345`, `0.382 ms` (median `0.345 ms`). Tile-eight raw medians were `0.440`, `0.801`, `0.778`, `0.798`, `0.440 ms` (median `0.778 ms`), a `125.5%` regression.
- Correctness evidence: Q4_0 batch-eight output matched explicit GGUF dequantization and CPU F32 matmul with maximum absolute error `8.34465e-07` and relative error `4.46011e-07`.
- Decision: rejected and removed. The reduced weight traversal did not overcome tile-eight register/occupancy pressure on the W6900X.
- Commit: none; regressing kernel change removed. The new Q4_0 correctness coverage remains.

### 2026-08-16 22:11 PDT - PERF-005

- Change: temporarily routed Qwen3.5 full-attention preparation through the existing native Metal kernel that fuses Gemma Q/K normalization, partial NeoX RoPE, and QKV/gate unpacking.
- Benchmark evidence: exact production-shape batch-8 microbenchmark improved from `2.225 ms` to `0.162 ms`. Clean warmed end-to-end three-sample medians moved from `32.309 TPS` (`32.434`, `32.309`, `32.269`) to `30.680 TPS` (`30.486`, `30.814`, `30.680`), a `5.0%` regression. A 128-token comparison also favored the existing path (`40.470` versus `38.100 TPS`).
- Correctness evidence: fused Q/K, V, and gate outputs matched the PyTorch reference at `rtol=2e-5`, `atol=2e-5`; native grouped-query attention still passed with maximum error `4.76837e-07`.
- Decision: rejected and removed. Per-operation synchronization overstated the value of collapsing the asynchronous PyTorch/MPS command chain.
- Commit: none.

### 2026-08-16 22:17 PDT - PERF-006

- Change: separately tried the existing four-request Q5_K/Q6_K specialization and the batch-24 Q6_K vec4 specialization for batch eight.
- Benchmark evidence: representative Q5_K GDN output projection regressed from `0.809 ms` to `0.888 ms`; the Q6_K LM head regressed from `27.227 ms` to `32.446 ms` with the tile-four kernel and to `34.166 ms` with the batch-24 vec4 kernel.
- Correctness evidence: dispatch-only experiments retained the already-tested quantized kernels; no numerical mismatch was observed.
- Decision: rejected and removed. The alternative register geometries did not fit batch eight without a dedicated specialization.
- Commit: none.

### 2026-08-16 22:19 PDT - PERF-007

- Change: added a dedicated Q6_K batch-8 vec4 kernel. Four eight-lane subgroups share a SIMD group; each subgroup processes two requests while dequantizing and dotting four adjacent weights at once.
- Benchmark evidence: full `output.weight` Q6_K `(5120, 248320)` LM-head median improved from `27.227 ms` (`35.7 GiB/s`) to `9.186 ms` (`105.7 GiB/s`), a `66.3%` reduction.
- Correctness evidence: GGUF dequantized F32 reference passed at batch eight with maximum absolute error `1.19209e-06` and relative error `4.64191e-07`.
- Decision: retained for end-to-end server validation.
- Commit: pending.

### 2026-08-16 22:26 PDT - PERF-008 and end-to-end validation

- Change: added the analogous Q5_K batch-8 vec4 kernel and enabled the dedicated Q5_K/Q6_K paths only when the runtime batch is exactly eight.
- Benchmark evidence: representative Q5_K `(6144, 5120)` GDN output projection improved from `0.809 ms` (`24.9 GiB/s`) to `0.356 ms` (`56.6 GiB/s`), a `56.0%` reduction. After a discarded `34.978 TPS` first-use run, the real workload produced `37.965`, `38.027`, `37.887`, `38.016`, and `38.020 TPS`; median `38.016 TPS`, up `5.063 TPS` (`15.4%`) from baseline. Median wall time improved from `7.769 s` to `6.734 s`.
- Correctness evidence: all eight responses retained 32 output IDs. The combined quantized reference suite passed; Q5_K maximum absolute error was `1.54972e-06` and relative error `4.87504e-07`, while Q6_K retained `1.19209e-06` and `4.64191e-07`.
- Decision: retained. The remaining target gap is `4.937 TPS` to `42.953 TPS`.
- Commit: pending; the Metal extension file contains the surrounding uncommitted MPS port.

### 2026-08-16 22:29 PDT - PERF-009

- Change: separately tested a four-subgroup Q4_0 vec4 kernel, the already-present two-half batch-eight split kernel, and `chunks_per_thread` values two and eight around the existing value four.
- Benchmark evidence: representative Q4_0 `(5120, 17408)` baseline `0.350 ms`; four-subgroup vec4 `0.686 ms`, two-half split `0.390 ms`, unroll two `0.402 ms`, and unroll eight `0.405 ms`.
- Correctness evidence: no candidate was retained; the production dispatch and kernel source were restored after each microbenchmark.
- Decision: rejected. The current four-row-per-SIMD, four-chunk kernel remains the best measured Q4_0 geometry.
- Commit: none.

### 2026-08-16 22:34 PDT - PERF-010

- Change: temporarily fused projection unpacking, decode causal-convolution state update, SiLU, and gate/a/b extraction into one Metal dispatch.
- Benchmark evidence: alternating warmed batch-eight median was `0.147 ms` for the existing pack-plus-convolution chain and `0.151 ms` for the fused kernel; no measurable opportunity remained after asynchronous command submission.
- Correctness evidence: fused mixed QKV, gate, a/b outputs, and mutated convolution state matched the existing two-kernel path; the full native fused-op test passed.
- Decision: rejected and removed, including the temporary microbenchmark.
- Commit: none.

### 2026-08-16 22:45 PDT - PERF-011

- Change: temporarily added an opt-in idle-only MPS request-coalescing delay and tested a 2 ms window; separately submitted one HTTP request already containing all eight sequences as the perfect-coalescing upper bound.
- Benchmark evidence: the delay changed the observed prefill split from `1 + 7` to `4 + 4`, while a warmed concurrent-request sample measured `37.876 TPS`. The single size-eight batched request measured `39.053 TPS`, only `1.037 TPS` above the retained `38.016 TPS` median and still `3.900 TPS` below target.
- Correctness evidence: all paths returned the requested 256 output tokens; no scheduler state or admission rules beyond the bounded idle delay were changed.
- Decision: rejected and removed. The upper bound proves request coalescing cannot close the remaining gap, and retaining the delay would spend single-request TTFT for an unproven median gain.
- Commit: none.

### 2026-08-16 22:35 PDT - PERF-010 through PERF-012 and 60 TPS validation

- NEXTN: completed native-Metal target/draft loading, torch-native multi-step attention, top-k1 chain construction, real top-k/top-p verification sampling, causal-conv/GDN checkpoints, and accepted-state commit. It served coherent output end to end. Its batch-1 result was `4.872 TPS`; scheduler telemetry reported mean accept length `2.80` out of four verify tokens and draft acceptance rate `0.60`, so the experiment was retained as functionality and closed as the immediate throughput path.
- Q6_K change: exact batch 24 now dequantizes four adjacent weights per lane and processes six requests per eight-lane subgroup. Removing exact-24 bounds branches reduced the full `output.weight` median from `57.343 ms` to `21.565 ms` (`62.4%`).
- Q5_K change: the same vec4/subgroup layout now covers exact batch 24 for every GDN output projection, reducing representative `blk.0.ssm_out.weight` from `2.081 ms` to `0.891 ms` (`57.2%`).
- End-to-end evidence: three independent 24-request runs, each returning 128 sampled tokens per request (3,072 output tokens), measured `62.034`, `61.856`, and `62.556 TPS`; median `62.034 TPS`. Sampling used temperature `0.8`, top-p `0.9`, top-k `20`, and ignored EOS. Generated continuations were coherent.
- Steady evidence: warmed scheduler windows measured `72.80`, `72.31`, `72.83`, `72.51`, `72.49`, and `72.15 TPS`.
- Correctness evidence: the Q4_0/Q4_1/Q5_K/Q6_K GGUF reference suite passed at batch 24 and 17 output rows. Q5_K maximum absolute/relative error was `1.60933e-06` / `3.81316e-07`; Q6_K was `1.07288e-06` / `3.36184e-07`. Native fused-op, GDN, attention, speculative-control, and speculative-state tests also passed.
- Decision: target achieved with repeatable end-to-end margin; retain both exact-batch-24 kernels.
