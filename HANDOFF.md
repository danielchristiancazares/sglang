# Qwen3.8-27B Q5 SGLang performance handoff

**Snapshot:** 2026-09-01 17:06 PDT

**Repository:** `/Users/dcazares/sglang`

**Active goal ID:** `01a05d87-e5de-7bf0-bcef-7122ed2f6dd2`

**Goal status:** active

## Objective and present result

The objective is to serve Qwen3.8-27B at Q5-class quality through SGLang for
real Codex work with `xhigh` reasoning, a 131K context window, and at least
**20 generated tokens/s**. The Codex harness stays unchanged.

The current selected Apple lane is the immutable mixed 4.951-bpw checkpoint
with the A100 Q5 and A111 Q4 batch-one Metal kernels. Its qualified direct
result is:

| Metric | Selected result |
|---|---:|
| Generic mixed-checkpoint direct decode | **18.121698566 tok/s** |
| Selected A100 Q5 + A111 Q4 direct decode | **19.241981332 tok/s** |
| Gain over generic | **+1.120282766 / +6.181996%** |
| Remaining direct gap | **0.758018668 tok/s / 3.939400%** |
| Fixed-work digest | `d0193f6d413b68c1` |
| Last token | `11406` |

The selected number is the aggregate of two independent balanced five-pair
windows. Exact 131,072-token serving, sampled behavior, and Codex `xhigh`
qualification for this mixed checkpoint remain pending. An older GGUF Q5 lane
passed exact 131K capacity at far lower decode speed; that capacity result does
not qualify the current native affine-Q5 lane.

## Acceptance gates

A production promotion must pass every item below:

1. At least five consecutive direct samples and an independent reversed
   window at or above 20 tok/s with margin.
2. Stable fixed-work token count, digest, and last token.
3. Real `context_length=max_total_tokens=131072` capacity.
4. Ordinary sampled reasoning at temperature `1.0`, top-p `0.95`, top-k `20`,
   and presence penalty `1.5`.
5. Coherent `reasoning_content`, arithmetic answer `703`, and exactly one
   parsed `multiply({"a":37,"b":19})` call with
   `finish_reason=tool_calls`.
6. `/model_info` reporting image and audio understanding disabled.
7. A clean launcher-default relaunch and healthy listener.
8. A Codex 0.151.0 `xhigh` repository/tool turn through the Responses API at
   the real 131K window.

## Hard implementation and safety constraints

- Repository-wide `AGENTS.md` applies. Additions use C++ or Metal/CUDA; new
  Python code is outside the permitted implementation surface.
- Use `apply_patch` for source and documentation edits.
- Run one model, benchmark, Metal compilation test, or GPU workload at a time.
- Preserve Qwen reasoning, tool parsing, exact capacity, language-only
  metadata, and sampled behavior as performance requirements.
- Keep downloaded checkpoints immutable. Derived artifacts live at distinct
  paths with recorded provenance and hashes.
- A reboot, logout, sleep transition, WindowServer restart, or system-service
  restart can remove remote SSH access until an interactive login. IT already
  performed one recovery reboot. These operations require Daniel's explicit
  coordination.
- Leave Spotlight, FileProvider, `MTLCompilerService`, MCP, ChatGPT, Codex,
  and unrelated desktop processes under their existing ownership.

At this snapshot, port 30000 is free and the model/benchmark process set is
empty. Launchd owns the idle `MTLCompilerService` processes. A continuing
Spotlight/FileProvider wave has `fileproviderd`, `mds_stores`, and fresh
`mdworker_shared` cohorts active. Memory pressure reports 95% free, zero
throttled pages, and `pmset -g therm` reports normal thermal/performance
status. Wait for an ordinary idle host before qualification windows. Short
correctness probes and paired microbenchmarks remain useful while indexing is
active; label their performance as externally exposed to contention.

## Git and worktree state

### Main checkout

- Branch: `main`
- HEAD: `b1322134e51ebf301496ff53fc922aa24f9f0103`
  (`perf: record post-A111 Q4 screens`), signed with a verified good EDDSA
  signature.
- Tracking state at snapshot: `main...origin/main [ahead 84]`.
- Index: empty.
- Selected code commit: `22408c50c49dd10a84d73740617991a6bf9eba02`
  (`perf(mlx): tile stock-exact Q4 decode`).

Daniel owns these three existing main-checkout modifications. Preserve their
working-copy bytes and keep them outside optimization commits:

| Path | Working-copy Git blob |
|---|---|
| `python/sglang/srt/hardware_backend/mlx/native/qwen38_engine.cpp` | `0260ff714075fd6dc01619a544d7071870d75955` |
| `python/sglang/srt/hardware_backend/mlx/native/qwen38_engine.h` | `40e375e39ce8035c8777a46f4d9b45488f4d250e` |
| `test/registered/unit/hardware_backend/mlx/test_qwen38_affine_small_batch_qmm.cpp` | `bb42de39fbe8315b4a3a5819b0e498e6617fb739` |

### Active detached candidate worktree

Path:

```text
/Users/dcazares/.cache/sglang-qwen38/worktrees/perf-q4-packs4
```

The worktree is detached at exact base `22408c50c4`. It contains the active
A114 8-SIMD-by-4-paired-row fused Q4 gate/up/SwiGLU experiment:

| Modified path | Current Git blob |
|---|---|
| `python/sglang/srt/hardware_backend/mlx/native/qwen38_engine.cpp` | `48a4b550f8a81e19d31200ba8795685cbf7ac462` |
| `python/sglang/srt/hardware_backend/mlx/native/qwen38_engine.h` | `512335f1ae677f48ee76a77d2f097bef720e71ce` |
| `test/registered/unit/hardware_backend/mlx/test_qwen38_affine_q4_batch_one_qmv.cpp` | `b42ddac6a272e7db61282810bdcda8ba0664f4bd` |

`git diff --check` passes. The diff adds 302 lines across those three paths.
The earlier A113 removal of `eval(pending_tok_)` has been restored here, so
A114 isolates the fusion variable.

## Selected checkpoint and artifacts

### Model

```text
/Users/dcazares/.cache/huggingface/hub/models--maglun--Qwen3.8-27B-MLX-Mixed-4.95bpw/snapshots/596b8067f7cf429007bb668874ffee7e917c8340
```

This immutable language snapshot contains exactly 162 Q4 and 240 Q5 affine
linears plus 96 dense BF16 recurrent b/a linears. All 64 target MLP gate and
64 up projections are Q4/G64. Its reachable quantized-linear target stream is
15,877,570,560 bytes per token.

### Selected A100+A111 artifacts

| Artifact | SHA-256 |
|---|---|
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/libqwen38_q4_4x4_exact_a100.dylib` | `9b4452d6f635355f77cecb60e6281c08d71478e1892f66ef85006540aa00b01e` |
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/test_qwen38_affine_q4_batch_one_qmv_selected` | `2101c61d61ec2d09b1a933fc4e29379a5b65a6a51497b64d22a52e164c719609` |
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_affine_q4_qmv_exact` | `c105e7d38daf45515616a6ab152368432a3914eee75ec42bde962485e5d6b8ef` |

A selected Metal System Trace maps 48,156 of 48,831 target shader PCs. Stock
Q4 QMV owns 43.546%, custom Q5 QMV owns 48.613%, and all QMV kernels together
own 92.159%.

## A114 fused Q4 gate/up/SwiGLU experiment

### Mechanism

The candidate adds `SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU`. For batch-one Q4/G64
MLPs it consumes gate weights, up weights, and the shared BF16 activation in
one native Metal dispatch. Each 8-SIMD threadgroup emits 32 paired outputs,
with four gate and four up FP32 accumulators per SIMD. It preserves these BF16
rounding points explicitly:

1. gate QMV result;
2. up QMV result;
3. `gate * sigmoid(gate)`;
4. SwiGLU result multiplied by up.

Prefill rows and unsupported shapes retain the established separate path.
The candidate directly reuses MLX 0.32.2's Q4 load/dot expression structure
and its sigmoid expression.

### Artifacts

| Artifact | SHA-256 |
|---|---|
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/libqwen38_a114_q4_fused_swiglu_8x4.dylib` | `1733390d7e983762cd78fb64adaeb770e8fd18c0713fd32c0f2ccca68c454e3f` |
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/test_qwen38_a114_q4_fused_swiglu_8x4` | `9e94fd9cb0abdfd81c4aefc29474bc0833526ca1d4c7008538a809233a13ba6a` |
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_a114_q4_fused_swiglu_8x4` | `520b952915da56d73f89a3ee10e8e73dc51763fdcb9630d93b55cdca28f563c1` |
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_a114_q4_fused_swiglu.cpp` | `3f4ca53a61943a8bb17553426bb5868ccd19b5614d1dce4183bac7f330ad5f48` |

Strict C++20/O3 warnings-as-errors builds pass. The focused parity executable
reports zero mismatches for the selected Q4 path at K/N 512/64, 5120/128, and
5120/17408, and zero fused-chain mismatches at 512/64 and 5120/17408.

Long production-shape microbenchmarks use `1000` warmups and `5000` timed
iterations:

- separate gate/up/SwiGLU: **0.596468750, 0.600999883 ms**, mean
  **0.598734317 ms**;
- fused 8x4: **0.555415667, 0.560186117 ms**, mean
  **0.557800892 ms**;
- matched reduction: **0.040933425 ms / 6.837%**;
- direct 8x2 versus 8x4 head-to-head: **0.565323179 / 0.555322754 ms**,
  making 8x4 **1.769%** faster;
- 16x2 mean **0.562536738 ms**, slower than adjacent 8x4 mean
  **0.559433879 ms**;
- 8x8 mean **0.602943563 ms**, slower than adjacent 8x4 mean
  **0.560497392 ms**.

Every microbenchmark arm has digest `8a9031349585365a` and first value
`0.875`.

### Full-model result and correctness blocker

One full mixed-checkpoint screen completed at **19.406869588 tok/s** under
active Spotlight/FileProvider activity. It produced digest
`f2a59800c8d89f75` and last token `19`; the selected canonical values are
`d0193f6d413b68c1` and `11406`. Treat the throughput as a diagnostic only.
The current fused implementation fails the fixed-work trajectory gate.

Synthetic parity therefore covers too little of the real activation/parameter
space. Before another throughput run, locate the first real layer/token that
diverges and compare these boundaries independently:

1. fused gate QMV against selected A111 gate QMV;
2. fused up QMV against selected A111 up QMV;
3. BF16 sigmoid output;
4. the first BF16 multiply;
5. the second BF16 multiply.

An actual-checkpoint C++ fixture or a temporary diagnostic output from the
native engine can establish this without adding Python. Extend the focused C++
test with real tensors or a captured real hidden row. Preserve MLX's exact
expression structure and operation boundaries. A corrected implementation
still needs a canonical full-model digest and paired qualification windows.

### Exact A114 full-model command

```text
/usr/bin/env MLX_SDPA_BLOCKS=64 MLX_MAX_MB_PER_BUFFER=256 MLX_MAX_OPS_PER_BUFFER=100 MLX_METAL_FAST_SYNCH=1 SGLANG_MLX_NATIVE_TARGET_ONLY_PREFILL_CHUNK_SIZE=2048 SGLANG_MLX_NATIVE_SAMPLING=1 SGLANG_MLX_NATIVE_SAMPLING_SEED=42 SGLANG_MLX_NATIVE_MAX_REASONING_TOKENS=256 SGLANG_MLX_NATIVE_Q5_BATCH_ONE_QMV=1 SGLANG_MLX_NATIVE_Q4_BATCH_ONE_QMV=1 SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU=1 /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 240s /Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_native /Users/dcazares/.cache/sglang-qwen38/artifacts/libqwen38_a114_q4_fused_swiglu_8x4.dylib /Users/dcazares/.cache/huggingface/hub/models--maglun--Qwen3.8-27B-MLX-Mixed-4.95bpw/snapshots/596b8067f7cf429007bb668874ffee7e917c8340 128 32 128
```

## A113 duplicate scalar evaluation experiment

`Engine::emit_scheduled()` currently calls `eval(pending_tok_)` and then
`pending_tok_.item<int32_t>()`. Installed MLX 0.32.2's `array::item<T>()`
performs its own evaluation. Removing the explicit free `eval()` preserves
the two-token scheduling pipeline in completed screens.

Evidence retained in the logs:

- two preliminary reversed pairs: control **19.195765829**, candidate
  **19.219406606 tok/s**;
- four clean interleaved pairs: control **19.214768638**, candidate
  **19.235724277 tok/s**, a **+0.020955640 / +0.109065%** movement;
- two excluded indexing-contended candidates: **16.599721750** and
  **10.995856101 tok/s**.

The candidate source is currently absent from the detached worktree. Recreate
it by deleting only `eval(pending_tok_);` in `Engine::emit_scheduled()`. Finish
one clean replacement pair and an independent reversed five-pair window after
the host becomes idle. Keep A113 isolated from A114 during qualification.

## Important closed paths

Read `FAILED_PATHS.md` before reopening any branch. The most relevant closed
families are:

- A111 post-selection Q4 geometries: single-projection 8x4 is flat; 4x8 and
  2x8 regress; packed `ushort4` is full-model neutral; four packs per lane
  changes output; explicit locals and full unrolling regress.
- Hand-inlined Q4 dot arithmetic reached 19.413641543 tok/s and changed the
  digest. Its speed depends on unacceptable FP32 reassociation.
- Dense recurrent b/a fusion is aggregate-flat at 18.993221731 versus
  18.993383731 tok/s.
- Q5 paired-lane word sharing and 128-bit activation reads regress the
  production-shape matrix.
- The published Q4 MTP head and Q5 MTP/DFlash compositions remain below the
  target-only Q5 path under real sampling.
- Earlier full-Q4 and selective-Q2 lanes include results above 20; their
  precision/behavior contracts do not satisfy this Q5 objective.

## Reproduction commands

### Selected direct baseline

```text
/usr/bin/env MLX_SDPA_BLOCKS=64 MLX_MAX_MB_PER_BUFFER=256 MLX_MAX_OPS_PER_BUFFER=100 MLX_METAL_FAST_SYNCH=1 SGLANG_MLX_NATIVE_TARGET_ONLY_PREFILL_CHUNK_SIZE=2048 SGLANG_MLX_NATIVE_SAMPLING=1 SGLANG_MLX_NATIVE_SAMPLING_SEED=42 SGLANG_MLX_NATIVE_MAX_REASONING_TOKENS=256 SGLANG_MLX_NATIVE_Q5_BATCH_ONE_QMV=1 SGLANG_MLX_NATIVE_Q4_BATCH_ONE_QMV=1 /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 240s /Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_native /Users/dcazares/.cache/sglang-qwen38/artifacts/libqwen38_q4_4x4_exact_a100.dylib /Users/dcazares/.cache/huggingface/hub/models--maglun--Qwen3.8-27B-MLX-Mixed-4.95bpw/snapshots/596b8067f7cf429007bb668874ffee7e917c8340 128 32 128
```

### A114 focused parity

```text
/opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 180s /Users/dcazares/.cache/sglang-qwen38/artifacts/test_qwen38_a114_q4_fused_swiglu_8x4
```

Expected output has five `max_abs=0 mismatches=0` lines.

### A114 production-shape microbenchmark

```text
/opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 180s /Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_a114_q4_fused_swiglu_8x4 separate 5120 17408 1000 5000
/opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 180s /Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_a114_q4_fused_swiglu_8x4 fused 5120 17408 1000 5000
```

### Strict A114 dylib build

```text
clang++ -std=c++20 -O3 -fPIC -shared -Wall -Wextra -Werror -isystem /Users/dcazares/sglang/.venv/lib/python3.11/site-packages/mlx/include -I/Users/dcazares/.cache/sglang-qwen38/worktrees/perf-q4-packs4/python/sglang/srt/hardware_backend/mlx/native -L/Users/dcazares/sglang/.venv/lib/python3.11/site-packages/mlx/lib -Wl,-rpath,/Users/dcazares/sglang/.venv/lib/python3.11/site-packages/mlx/lib -lmlx -o /Users/dcazares/.cache/sglang-qwen38/artifacts/libqwen38_a114_q4_fused_swiglu_8x4.dylib /Users/dcazares/.cache/sglang-qwen38/worktrees/perf-q4-packs4/python/sglang/srt/hardware_backend/mlx/native/qwen38_engine.cpp /Users/dcazares/.cache/sglang-qwen38/worktrees/perf-q4-packs4/python/sglang/srt/hardware_backend/mlx/native/qwen38_c_api.cpp
```

The linker emits the known macOS 26.0 versus MLX 26.2 deployment warning.

## Recommended continuation order

1. Re-read root `AGENTS.md`, `notes/current-state.md`, this handoff, and every
   later `notes/experiment-log.md` entry.
2. Reconfirm main HEAD/status, the three user-owned blobs, detached-worktree
   status, port 30000, workload ancestry, Metal compiler ownership, GPU
   residency, memory pressure, and thermal state.
3. Let the indexing/FileProvider wave reach the ordinary idle state.
4. Diagnose A114 at the first real divergence. Retain the 8x4 geometry only
   after real-weight, real-hidden exactness passes.
5. Run one candidate/control direct screen. Require canonical digest
   `d0193f6d413b68c1` and last token `11406` before collecting a throughput
   window.
6. If A114 becomes exact, collect balanced five-pair and independent reversed
   windows. Record every raw sample and host state.
7. Complete A113's interrupted windows as a separate candidate.
8. Profile the latest exact winner and continue native C++/Metal hotspot work
   until direct performance clears 20 with margin.
9. Run exact 131K SGLang serving, behavior, Responses API, and Codex `xhigh`
   gates only after the direct floor is stable.
10. Update `PERFORMANCE_LOG.md`, `FAILED_PATHS.md`,
    `notes/experiment-log.md`, and compact state documents after every
    meaningful result. Commit signed, atomic wins and useful rejected-path
    evidence while preserving Daniel's three working-copy files.

## Recovery authorities

- `AGENTS.md`: repository-wide contract and process safety.
- `notes/current-state.md`: compact selected state.
- `notes/decisions.md`: durable selected and closed branches.
- `notes/benchmark-contract.md`: workload and qualification gates.
- `notes/experiment-log.md`: chronological commands, raw samples, failures,
  process state, and artifact provenance.
- `PERFORMANCE_LOG.md`: active timings and candidate deltas.
- `FAILED_PATHS.md`: measured rejected candidates and reopen conditions.

The objective remains open. The selected production-quality direct result is
19.241981332 tok/s; the current A114 implementation has useful microbenchmark
economics and a real-model correctness divergence that must be resolved before
its speed can count.
