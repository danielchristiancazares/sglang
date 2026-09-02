# Qwen3.8-27B Q5 SGLang performance handoff

**Snapshot:** 2026-09-01 19:09 PDT

**Repository:** `/Users/dcazares/sglang`

**Active goal ID:** `01a05f84-93f2-75b3-9fd1-f5fac3dfd8b4`

**Goal status:** active

## Objective and present result

The objective is to serve Qwen3.8-27B at Q5-class quality through SGLang for
real Codex work with `xhigh` reasoning, a 131K context window, and at least
**20 generated tokens/s**. The Codex harness stays unchanged.

The current selected Apple lane is the immutable mixed 4.951-bpw checkpoint
with the A100 Q5, A111 Q4, precise-exp A114 fused Q4 MLP Metal kernels, and
A113's duplicate token-evaluation cleanup, plus A117's lane-parallel precise
fused epilogue. Its qualified direct result is:

| Metric | Selected result |
|---|---:|
| Generic mixed-checkpoint direct decode | **18.121698566 tok/s** |
| Selected A100 + A111 + A114 + A113 + A117 direct decode | **19.453092367 tok/s** |
| Gain over generic | **+1.331393801 / +7.346959%** |
| A117 gain over matched serial epilogue | **+0.184616727 / +0.958128%** |
| Remaining direct gap | **0.546907633 tok/s / 2.811417%** |
| Fixed-work digest | `d0193f6d413b68c1` |
| Last token | `11406` |

The selected number is the aggregate of two independent A117 five-pair
windows, the second in reversed order and split by one deliberate cooldown to
avoid process-reload churn. Exact 131,072-token serving, sampled
behavior, and Codex `xhigh`
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
- Latest selected code commit: `00d09138ce`
  (`perf(mlx): parallelize fused Q4 SwiGLU epilogue`), signed with a verified
  good EDDSA signature. This documentation update follows it.
- Tracking state at the selected code commit:
  `main...origin/main [ahead 91]`.
- Index: empty.
- Selected code commit: `00d09138ce`.

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

The worktree is detached at exact base `22408c50c4`. It retains the promoted
A114 8-SIMD-by-4-paired-row fused Q4 gate/up/SwiGLU source:

| Modified path | Current Git blob |
|---|---|
| `python/sglang/srt/hardware_backend/mlx/native/qwen38_engine.cpp` | `5912bc2fe9b1e8f34bdace0b1a009f8aa1223d04` |
| `python/sglang/srt/hardware_backend/mlx/native/qwen38_engine.h` | `512335f1ae677f48ee76a77d2f097bef720e71ce` |
| `test/registered/unit/hardware_backend/mlx/test_qwen38_affine_q4_batch_one_qmv.cpp` | `a2137fa5f148c2d285ae68bd4852776049ed5aab` |

`git diff --check` passes. The diff adds 380 lines and removes three across
those three paths. It contains selected A114, A113, and A117; compare against
signed `ad11696f2e` to isolate A117's 18-addition/10-deletion epilogue diff.

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
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/libqwen38_a114_q4_fused_swiglu_8x4_precise.dylib` | `ce3693e7a3c7e10c12fa7fad8d4a6de1ad114801342069def712bbdf931eb208` |
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/test_qwen38_a114_q4_fused_swiglu_8x4_precise_boundary` | `dcbae745f0945d282565fb305750512968733dccf0876c000b0891ab78d08f47` |
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_a114_q4_fused_swiglu_8x4_precise` | `9dcb07560b177c5c920ddc3bede58479e5e659ee2a31f5e68d1980192d97ee9f` |
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

### Full-model qualification and correctness repair

A real-weight/real-hidden trace localized the old fast-exp failure to layer
62, element 36: gate and up were exact, while `metal::exp` rounded sigmoid to
BF16 `0x3a8c` instead of MLX's `0x3a8b`. `metal::precise::exp` makes gate, up,
sigmoid, SiLU, and output exact across all 64 layers. A dedicated C++ boundary
test at gate `-6.84375` passes the precise artifact in all 32 rows and fails
the preserved fast artifact in all 32.

Forward and reversed five-pair windows improve
**19.236012324 -> 19.264036375** and
**19.221428286 -> 19.272779458 tok/s**. Aggregate matched control/candidate is
**19.228720305 / 19.268407916 tok/s**, a
**+0.039687612 / +0.206398%** win. Every run retains digest
`d0193f6d413b68c1` and last token 11406.

### Exact A114 full-model command

```text
/usr/bin/env MLX_SDPA_BLOCKS=64 MLX_MAX_MB_PER_BUFFER=256 MLX_MAX_OPS_PER_BUFFER=100 MLX_METAL_FAST_SYNCH=1 SGLANG_MLX_NATIVE_TARGET_ONLY_PREFILL_CHUNK_SIZE=2048 SGLANG_MLX_NATIVE_SAMPLING=1 SGLANG_MLX_NATIVE_SAMPLING_SEED=42 SGLANG_MLX_NATIVE_MAX_REASONING_TOKENS=256 SGLANG_MLX_NATIVE_Q5_BATCH_ONE_QMV=1 SGLANG_MLX_NATIVE_Q4_BATCH_ONE_QMV=1 SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU=1 /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 240s /Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_native /Users/dcazares/.cache/sglang-qwen38/artifacts/libqwen38_a114_q4_fused_swiglu_8x4_precise.dylib /Users/dcazares/.cache/huggingface/hub/models--maglun--Qwen3.8-27B-MLX-Mixed-4.95bpw/snapshots/596b8067f7cf429007bb668874ffee7e917c8340 128 32 128
```

## A113 duplicate scalar evaluation experiment

Before signed `ad11696f2e`, `Engine::emit_scheduled()` called
`eval(pending_tok_)` and then `pending_tok_.item<int32_t>()`. Installed MLX
0.32.2's `array::item<T>()` performs its own evaluation. Removing the explicit
free `eval()` preserves the two-token scheduling pipeline.

Forward and reversed five-pair windows improve
**19.270942164 -> 19.277656005** and
**19.285820872 -> 19.301343552 tok/s**. Aggregate matched control/candidate is
**19.278381518 / 19.289499778**, a **+0.057672%** exact-output win.

## A117 lane-parallel precise fused epilogue

A selected A114+A113 Metal System Trace maps 36,768 of 37,168 sampled target
shader PCs with zero ambiguity. The fused Q4 gate/up/SwiGLU shader owns
**42.141%**, custom Q5 owns **44.533%**, and remaining ordinary Q4 owns
**6.804%**. A117 changes the largest single shader: after the unchanged four
gate/up reductions, lanes 0--3 each execute one precise sigmoid, SiLU, and
product chain. Compile-time-named ternaries select the lane's reduced values;
this avoids the dynamically indexed local arrays that regressed under A116.

Artifacts:

| Artifact | SHA-256 |
|---|---|
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/libqwen38_a117_q4_fused_swiglu_parallel_select.dylib` | `c8ed45122f36a700b588d93f7a227d31003a01f3b4370d9195367ee7c255f36e` |
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/test_qwen38_a117_q4_fused_swiglu_parallel_select` | `b1383f99fd2d73ae48753605aee3318652df489d75436da08dc85ae62166f0ef` |
| `/Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_a117_q4_fused_swiglu_parallel_select` | `b7829284af06cf2bc290ec8ab2659b69fbc9c1e0d3f024308b87160dff80fb15` |

The focused executable passes all three Q4 QMV shapes, both fused-chain
shapes, and the precise `-6.84375` sigmoid boundary with zero mismatches.
Corrected 10,000-iteration candidate/control micro means are
**0.556654523 / 0.564698075 ms**, a **1.4243%** latency reduction.

Forward control/candidate pairs are
**19.275163985/19.368736679**,
**19.255827337/19.482499777**,
**19.315313630/19.464807390**,
**19.244643242/19.468935480**, and
**19.250769815/19.469615480 tok/s**. Means are
**19.268343602 / 19.450918961**. Replacement reversed candidate/control pairs
are **19.449756511/19.249257165**,
**19.457791537/19.284351927**,
**19.446175454/19.238727679**,
**19.462623411/19.311370484**, and
**19.459981955/19.259331137 tok/s**. Control/candidate means are
**19.268607678 / 19.455265774**. Aggregate matched control/candidate is
**19.268475640 / 19.453092367 tok/s**, a
**+0.184616727 / +0.958128%** win. Every clean run is canonical.

An earlier reverse batch at 11--14 tok/s is excluded: repeated 20 GB process
reloads caused extreme disk reads, page-ins, and swapouts for both artifacts.
After a 60-second idle interval the selected control recovered to
**19.203472228 tok/s**. The replacement window used a second 60-second split
after two pairs. Do not use the contaminated samples as A/B evidence.

## Important closed paths

Read `FAILED_PATHS.md` before reopening any branch. The most relevant closed
families are:

- A111 post-selection Q4 geometries: single-projection 8x4 is flat; 4x8 and
  2x8 regress; packed `ushort4` is full-model neutral; four packs per lane
  changes output; explicit locals and full unrolling regress.
- Fused A114 geometry at four SIMD groups regresses the production-shape
  microbenchmark; dynamically indexed lane-parallel epilogues regress the
  longer reversed window. PERF-FA133/FA134 close those exact forms.
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
/usr/bin/env MLX_SDPA_BLOCKS=64 MLX_MAX_MB_PER_BUFFER=256 MLX_MAX_OPS_PER_BUFFER=100 MLX_METAL_FAST_SYNCH=1 SGLANG_MLX_NATIVE_TARGET_ONLY_PREFILL_CHUNK_SIZE=2048 SGLANG_MLX_NATIVE_SAMPLING=1 SGLANG_MLX_NATIVE_SAMPLING_SEED=42 SGLANG_MLX_NATIVE_MAX_REASONING_TOKENS=256 SGLANG_MLX_NATIVE_Q5_BATCH_ONE_QMV=1 SGLANG_MLX_NATIVE_Q4_BATCH_ONE_QMV=1 SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU=1 /opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 240s /Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_native /Users/dcazares/.cache/sglang-qwen38/artifacts/libqwen38_a117_q4_fused_swiglu_parallel_select.dylib /Users/dcazares/.cache/huggingface/hub/models--maglun--Qwen3.8-27B-MLX-Mixed-4.95bpw/snapshots/596b8067f7cf429007bb668874ffee7e917c8340 128 32 128
```

### A114 focused parity

```text
/opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 180s /Users/dcazares/.cache/sglang-qwen38/artifacts/test_qwen38_a114_q4_fused_swiglu_8x4_precise_boundary
```

Expected output has five general `max_abs=0 mismatches=0` lines plus the
`gate=-6.84375 ... mismatches=0` boundary line.

### A114 production-shape microbenchmark

```text
/opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 180s /Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_a114_q4_fused_swiglu_8x4_precise separate 5120 17408 1000 5000
/opt/homebrew/bin/gtimeout --signal=TERM --kill-after=10s 180s /Users/dcazares/.cache/sglang-qwen38/artifacts/bench_qwen38_a114_q4_fused_swiglu_8x4_precise fused 5120 17408 1000 5000
```

### Strict A114 dylib build

```text
clang++ -std=c++20 -O3 -fPIC -shared -Wall -Wextra -Werror -isystem /Users/dcazares/sglang/.venv/lib/python3.11/site-packages/mlx/include -I/Users/dcazares/.cache/sglang-qwen38/worktrees/perf-q4-packs4/python/sglang/srt/hardware_backend/mlx/native -L/Users/dcazares/sglang/.venv/lib/python3.11/site-packages/mlx/lib -Wl,-rpath,/Users/dcazares/sglang/.venv/lib/python3.11/site-packages/mlx/lib -lmlx -o /Users/dcazares/.cache/sglang-qwen38/artifacts/libqwen38_a114_q4_fused_swiglu_8x4_precise.dylib /Users/dcazares/.cache/sglang-qwen38/worktrees/perf-q4-packs4/python/sglang/srt/hardware_backend/mlx/native/qwen38_engine.cpp /Users/dcazares/.cache/sglang-qwen38/worktrees/perf-q4-packs4/python/sglang/srt/hardware_backend/mlx/native/qwen38_c_api.cpp
```

The linker emits the known macOS 26.0 versus MLX 26.2 deployment warning.

## Recommended continuation order

1. Re-read root `AGENTS.md`, `notes/current-state.md`, this handoff, and every
   later `notes/experiment-log.md` entry.
2. Reconfirm main HEAD/status, the three user-owned blobs, detached-worktree
   status, port 30000, workload ancestry, Metal compiler ownership, GPU
   residency, memory pressure, and thermal state.
3. Let the indexing/FileProvider wave reach the ordinary idle state.
4. Treat precise-exp A114 in signed `ca524c3282` as selected; keep the fast-exp
   artifact closed by PERF-FA132.
5. Treat signed A113 commit `ad11696f2e` as selected.
6. Treat A117 in signed `00d09138ce` as selected; keep four-SIMD and dynamic
   epilogue variants closed by PERF-FA133/FA134.
7. Keep A118's explicit fused-Q4 vector load and A119--A121's Q5 `qkv`/`z`
   launch-sharing forms closed by PERF-FA135--137. Copied/split, direct
   four-SIMD/two-output, and exact-ratio 5:3/eight-SIMD forms are all measured.
   The next distinct Q5 route must reduce weight-side bytes, instructions, or
   dependency cost rather than submission count alone.
8. Continue native C++/Metal hotspot work until direct performance clears 20
   with margin.
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
19.453092367 tok/s; A117 is qualified and committed on precise-exp A114+A113,
leaving 0.546907633 tok/s / 2.811417% to the direct floor before exact 131K
and Codex `xhigh` qualification.
