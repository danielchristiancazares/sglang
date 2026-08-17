# Qwen3.8-27B on 32 GB M1 Max: Engineering Handoff

## Mission

Make the full Qwen3.8-27B text model as capable, fast, and usable as the
hardware permits on this 32 GB M1 Max, with emphasis on coding agents. The
required serving surface is:

- the complete 27B text model;
- vision and audio disabled;
- reasoning enabled and disabled per request;
- parsed tool calls;
- preserved reasoning across tool-result turns;
- streaming OpenAI-compatible responses;
- the largest practical context window;
- measured, reproducible performance;
- Rust for the serving and control plane wherever it produces a real gain.

The inherited Windows-native Qwen3.8 work remains the behavioral and
benchmarking reference. Apple Silicon execution is implemented through MLX.

## Executive state

**MTP speculative decoding landed (August 16, 2026), flag-gated.**
`--mlx-mtp-path <mtp.safetensors>` (sidecar from
`Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`, head q4-quantized at load)
speculates depth-3 with batched verification inside
`hardware_backend/mlx/mtp_spec.py`, buffered behind the one-token-per-step
decode contract in `model_runner.py` (the native-engine pattern). Verified
through the production runner: **+2.72 tok/s (1.15x) on a coding prompt with
outputs identical to greedy decode**; engine-level coding-agent suite mean
**+2.87 tok/s** (write-code +1.37, refactor +4.41, tool-JSON +6.30). Through
the full server: **+0.83 tok/s on a 768-token coding generation (19.88 vs
19.05), +0.58 on 384-token tool-JSON**, adversarial control at breakeven.
The remaining engine-to-server gap has two quantified causes — scheduler
overhead serializing with GPU work during buffer pops (~4-5 ms/round;
early-launching the next round costs MORE via the Metal command-buffer
split), and think-prose stretches sitting at the acceptance breakeven and
tripping the adaptive policy (12-round window, off below 2.75 tok/round,
32-token AR stretches doubling to a 128 cap). Per-request round stats log as
`mtp[...]` lines. Constraints: greedy single-request decode, no logit edits,
`--disable-radix-cache` (prefix hits skip trunk computation, so there are no
hiddens to teacher-force the head), BF16 KV. The speed-winner control below
runs with the flag OFF and is unchanged. Details in BENCHMARK.md's
"Speculative decoding profile" section; measurement lessons and closed
optimization lanes (N=1 kernels lose to MLX's ~320 GB/s qmv, projection
fusion +0.45%, N=4 batched matvec floor ~2.2x N=1 across four kernel
architectures, memory config already optimal) are in the project memory.

Two production profiles have emerged:

1. **Fast 32K:** BF16 attention KV, unified radix/recurrent-state caching,
   native Rust HTTP/OpenAI control plane, one active request. This is the
   throughput winner: **19.58 tok/s steady decode** and **18.821 tok/s best
   end-to-end** on the fixed 256-token control.
2. **Maximum 262K:** affine q4 attention KV, geometrically grown buffers,
   native Rust control plane, one active request, radix cache disabled. It
   advertises the model's full **262,144-token** context. Verification has
   completed through 32,768 exact prompt tokens.

Rust now owns HTTP ingress, OpenAI request handling, tokenization/detokenization,
chat-template application, reasoning parsing, tool-call parsing, and streaming.
Python still owns the SGLang scheduler objects and the default mlx-lm graph.
A compiled C++ Qwen3.8 text graph (`SGLANG_USE_MLX_NATIVE_GRAPH=1`) now runs
affine-q4 linears, Gated DeltaNet (Metal kernel), full attention, and MLP
against vendored `libmlx.dylib`. Standalone native decode holds **19.57 tok/s**.
The Fast32K five-run on that path is **18.719 tok/s** with radix forced off
(C++ state is not a Python cache list), so `BENCHMARK.md` stays on the
mlx-lm incumbent. Live native Fast32K is on `:30000` (PID 12634) after a
warmup that exercises the compiled engine.

The repository is on `main` at base commit
`e577394a9bb7e86a0f8b34c3575ee02d899d2915`. All Apple-Silicon work is
uncommitted. Preserve the working tree exactly while continuing.

## Machine and toolchain

| Item | Value |
|---|---|
| Machine | Apple M1 Max MacBook Pro |
| CPU | 10 cores, 8 performance + 2 efficiency |
| GPU | 32 cores |
| Unified memory | 32 GB |
| OS | macOS 26.6.1 / Darwin 25.6.0 |
| Xcode | 26.6 |
| Metal compiler | Apple Metal 32023.883 |
| Installed Metal toolchain asset | 17F109 |
| Python environment | `.venv-mps`, Python 3.11.15 |
| MLX | 0.32.0 |
| mlx-lm | 0.31.3 |
| PyTorch | 2.11 |
| transformers | 5.12.1 |
| Rust used by repository | 1.92, pinned by `rust/rust-toolchain.toml` |
| Shell Rust | stable 1.97.1 |
| Added build/lint packages | `setuptools-rust==1.13.0`, `ruff==0.16.3` |

The Metal toolchain is healthy now:

```text
$ xcrun -sdk macosx metal --version
Apple metal version 32023.883 (metalfe-32023.883)
Target: air64-apple-darwin25.6.0

$ xcrun -sdk macosx --find metallib
/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.../Metal.xctoolchain/usr/bin/metallib
```

The successful installation sequence was:

```bash
sudo xcodebuild -runFirstLaunch
sudo xcodebuild -downloadComponent MetalToolchain
xcodebuild -downloadComponent MetalToolchain
```

The first privileged download mounted the asset; the following user invocation
made `xcrun metal` resolve it correctly.

## Model

```text
Repository: mlx-community/Qwen3.8-27B-4bit
Snapshot:   3e6447f082e89cc7f0bc6e5441afd38dfce760ff
Local path: /Users/dcazares/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff
```

- Full text weights are loaded with affine 4-bit, group-size-64 weight
  quantization.
- The architecture has 64 layers: 48 DeltaNet/recurrent layers and 16
  full-attention layers.
- Vision model construction is skipped by `--language-model-only`.
- The tokenizer and sibling `chat_template.jinja` remain available for agent
  semantics.
- Sampling uses the MLX device path with model sampling defaults; performance
  controls use greedy temperature 0.

## Current live processes

At this handoff:

| PID | Role | Purpose |
|---:|---|---|
| 11150 | launcher | Fast32K Rust+MLX server |
| 11152 | `sglang::scheduler` | listens on `127.0.0.1:30000` |

The previous 262K/q4 process (PID 5575) was already gone at the start of this
turn. A first Fast32K boot (PID 10886/10888) served `/model_info`, the four
capability probes, the decode-path `sample`, and one official five-run, then
died on Metal OOM while snapshotting recurrent state. The process above is
the post-fix relaunch.

Completed probe output:

```json
{
  "status": "HTTP/1.1 200 OK",
  "prompt_tokens": 32768,
  "max_new_tokens": 1,
  "elapsed_seconds": 425.299138,
  "prompt_throughput_tok_s": 77.046947,
  "output_ids": [248044]
}
```

## Qualified benchmark results

### Fast 32K serving profile

Fixed control: 12 prompt tokens, exactly 256 generated tokens,
`temperature=0`, `ignore_eos=true`, warmed server, five consecutive requests.

| Run | End-to-end time | Output throughput |
|---:|---:|---:|
| 1 | 13.644757 s | 18.761785 tok/s |
| 2 | 13.636761 s | 18.772786 tok/s |
| 3 | **13.602035 s** | **18.820713 tok/s** |
| 4 | 13.635554 s | 18.774448 tok/s |
| 5 | 13.627895 s | 18.784999 tok/s |
| Aggregate | **13.629400 s mean** | **18.782925 tok/s** |

- Peak logged steady decode: **19.58 tok/s**.
- Typical warmed steady decode: **19.42–19.58 tok/s**.
- Direct `mlx-lm` reference ceiling: **20.458 tok/s** on a shorter 58-prompt /
  27-output workload.
- Cold BF16-KV long-prefill record: **85.763 prompt tok/s**, 5,000 prompt
  tokens + 1 output in 58.300 seconds.

The previous Python HTTP/control-plane incumbent reached 18.798 tok/s best and
18.673 tok/s aggregate on the fixed control. The Rust path improved aggregate
request throughput by about 0.59% while moving the agent-facing serving work
into the compiled control plane.

### Maximum-context q4-KV profile

| Exact prompt tokens | Output tokens | End-to-end time | Prompt throughput | State |
|---:|---:|---:|---:|---|
| 5,000 | 1 | 59.176159 s | **84.493486 tok/s** | passed at native 262K cap |
| 16,384 | 1 | 204.230414 s | **80.223115 tok/s** | passed at native 262K cap |
| 32,768 | 1 | 425.299138 s | **77.046947 tok/s** | passed at native 262K cap |

Short-decode behavior under the 262K/q4 profile:

- one fixed end-to-end control: 13.710386 seconds, about 18.672 tok/s;
- warmed server logs: roughly 19.27–19.46 tok/s steady decode.

The 5K q4 run is an independent maximum-context result. The BF16 5K result
retains the pure prefill-speed record.

### KV memory model

Only the 16 full-attention layers require growing attention KV:

- BF16 KV: **65,536 bytes/token**, about 2.0 GiB at 32,768 tokens.
- Affine q4 KV with scale/bias metadata: **18,432 bytes/token**, exactly 4.5
  GiB at 262,144 tokens.

The 48 recurrent layers use bounded auxiliary state. q4 KV therefore makes the
native 262K model context feasible inside 32 GB while preserving the full
weights.

## Reproduction commands

Set the model path in the shell:

```bash
export MODEL=/Users/dcazares/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff
cd /Users/dcazares/sglang
```

### Fast 32K profile

```bash
env \
  SGLANG_USE_MLX=1 \
  SGLANG_RUST_SERVER=1 \
  SGLANG_MLX_CLEAR_CACHE_STEPS=0 \
  .venv-mps/bin/python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name qwen3.8-27b \
  --language-model-only \
  --context-length 32768 \
  --max-total-tokens 32768 \
  --max-running-requests 1 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 8192 \
  --mlx-enable-sampling \
  --sampling-defaults model \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --incremental-streaming-output \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --host 127.0.0.1 \
  --port 30000
```

This profile keeps unified FULL radix caching and recurrent auxiliary-state
caching enabled.

Compiled C++ graph (same Fast32K flags, plus the env and radix off):

```bash
env \
  SGLANG_USE_MLX=1 \
  SGLANG_USE_MLX_NATIVE_GRAPH=1 \
  SGLANG_RUST_SERVER=1 \
  SGLANG_MLX_CLEAR_CACHE_STEPS=0 \
  .venv-mps/bin/python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name qwen3.8-27b \
  --language-model-only \
  --context-length 32768 \
  --max-total-tokens 32768 \
  --max-running-requests 1 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 8192 \
  --disable-radix-cache \
  --mlx-enable-sampling \
  --sampling-defaults model \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --incremental-streaming-output \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --host 127.0.0.1 \
  --port 30000
```

`--disable-radix-cache` is required: the C++ engine owns conv / recurrent /
attention KV state and does not export it into the Python auxiliary pool.

### Maximum 262K profile

```bash
env \
  SGLANG_USE_MLX=1 \
  SGLANG_RUST_SERVER=1 \
  SGLANG_MLX_CLEAR_CACHE_STEPS=0 \
  .venv-mps/bin/python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name qwen3.8-27b \
  --language-model-only \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 8192 \
  --disable-radix-cache \
  --mlx-kv-cache-bits 4 \
  --mlx-kv-cache-group-size 64 \
  --mlx-enable-sampling \
  --sampling-defaults model \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --incremental-streaming-output \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --host 127.0.0.1 \
  --port 30000
```

The q4 profile deliberately uses one request and ChunkCache. Current radix
component semantics operate on floating cache arrays; the quantized tuple
cache uses a separate bounded path.

### Fixed 256-token throughput control

Run five times after warmup:

```bash
curl -sS -o /dev/null -w '%{time_total}\n' \
  -X POST http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Write a dense sequence of short Python identifiers separated by spaces.","sampling_params":{"temperature":0,"max_new_tokens":256,"ignore_eos":true}}'
```

### Native Rust exact-token context probe

The std-only probe bypasses Python request generation and tokenizer work. It
sends exact token IDs directly to `/generate`.

```bash
rustc +1.92 --edition 2024 -O \
  scripts/apple_silicon_context_probe.rs \
  -o /tmp/sglang-context-probe
/tmp/sglang-context-probe 32768 1
```

The first argument is exact prompt length; the second is output length.

## Rust control-plane work

### Request and chat-template path

- `rust/sglang-server/src/api_server/openai/template.rs`
  - discovers a sibling `chat_template.jinja`;
  - adapts and renders native `chat_template_kwargs` through MiniJinja.
- `rust/sglang-server/src/api_server/openai/chat.rs`
  - retains request `chat_template_kwargs` and its alias;
  - merges runtime `default_chat_template_kwargs`;
  - propagates request-specific “reasoning begins in the prompt” state through
    unary and streaming response paths.
- `rust/sglang-server/Cargo.toml` and `rust/Cargo.lock`
  - add the direct MiniJinja dependency required by the native template path.

### Reasoning and tool semantics

- `rust/sglang-server/src/api_server/openai/reasoning.rs`
  - teaches the Qwen parser when the template already injected a reasoning
    opener;
  - initializes the stream parser inside reasoning for thinking-enabled
    requests;
  - keeps thinking-disabled requests in final-content mode;
  - covers both modes with focused unit tests.

Live Rust-server gates already passed (re-verified on Fast32K after rebuild):

- streamed thinking: reasoning in `reasoning_content`, final `19 × 37 = 703`;
- streamed thinking-disabled request: `READY` in content with zero reasoning;
- streamed tool request: one exact
  `multiply({"a":37,"b":19})` call with `finish_reason=tool_calls`;
- preserved-thinking tool-result turn: final `37 × 19 = **703**`.

### Runtime/model metadata

- `rust/sglang-server/src/runtime/config.rs`
  - adds typed `default_chat_template_kwargs`, `language_model_only`,
    `weight_version`, capability flags, model type, and architectures.
- `rust/sglang-server/src/api_server/common.rs`
  - returns image/audio capability flags and model identity from
    `/model_info`.
- `python/sglang/srt/managers/rust_server.py`
  - stamps model type and architectures before releasing the heavyweight
    Python Hugging Face config.

The rebuilt Rust extension now serves `/model_info` with the language-only
capability bits. Two in-process hits on the first Fast32K boot, and a third
on the post-OOM relaunch, all reported:

```json
"has_image_understanding": false,
"has_audio_understanding": false,
"model_type": "qwen3_5",
"architectures": ["Qwen3_5ForConditionalGeneration"],
"weight_version": "default"
```

`/v1/models` advertised `max_model_len: 32768` on both hits.

## MLX execution and cache work

### Quantized long-context attention KV

- `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
  - adds `QuantizedAttentionKVCache`;
  - stores affine q4/q8 `(packed, scale, bias)` buffers;
  - starts at up to 4,096 tokens and doubles capacity to the configured cap;
  - preserves old tokens across growth;
  - supplies write, fetch, mask, reset, state, and overflow behavior.
- `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_wrapper.py`
  - recognizes quantized tuple KV during batched decode;
  - dispatches
    `mlx_lm.models.base.quantized_scaled_dot_product_attention`;
  - supports zero-padding by quantizing the pad;
  - validates homogeneous cache representations and sink settings.
- `python/sglang/srt/hardware_backend/mlx/model_runner.py`
  - wires q4/q8 cache only to full-attention layers;
  - keeps windowed/recurrent layers on their bounded paths;
  - accounts for packed data plus scale/bias metadata when sizing memory;
  - validates bit width, group size, head dimension, radix mode, and sinks.
- `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
  - propagates the new cache arguments.
- `python/sglang/srt/server_args.py`
  - adds `--mlx-kv-cache-bits {4,8}`;
  - adds `--mlx-kv-cache-group-size {32,64,128}`;
  - validates the ChunkCache requirement early.
- `test/registered/unit/hardware_backend/mlx/test_quantized_kv_cache.py`
  - covers arguments, quantized round-trip, overflow, geometric growth,
    preservation, runner wiring, and quantized-attention dispatch.

The initial full-capacity eager allocation increased short-decode latency. The
geometric allocator replaced it and restored near-incumbent decode while
retaining the 262K addressable cap. Failed trials stay outside `BENCHMARK.md`.

### Recurrent/radix correctness and language-only loading

The earlier Apple-Silicon changes remain part of this worktree:

- `python/sglang/srt/arg_groups/overrides.py`
- `python/sglang/srt/configs/model_config.py`
- `python/sglang/srt/hardware_backend/mlx/kv_cache/auxiliary_state.py`
- `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/tp_worker.py`
- `python/sglang/srt/mem_cache/allocation.py`
- `python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py`
- `python/sglang/srt/server_args.py`
- `test/registered/unit/hardware_backend/mlx/test_attention_patching.py`
- `test/registered/unit/test_model_overrides.py`

Together they provide tokenizer-preserving language-only model loading,
headless chunked prefill, per-request recurrent auxiliary state, radix
checkpoint integration, admission/eviction behavior, and MLX scheduler/worker
correctness.

### Existing MLX overlap path

`python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py` already queues two
lazy decode graphs. While the CPU finalizes step N, Metal can execute step N+1.
The chain remains safe for ordinary greedy/sampled decode and breaks for grammar
or custom-logit batches whose next mask depends on a materialized token.

This matters for future optimization: a generic “continuous decode steps”
tuning experiment may duplicate work already performed by the MLX-specific
two-graph chain. Profile the timeline before changing chain depth.

## Build and validation

Build the Rust extension from `python/` with the repository's pinned Rust:

```bash
cd /Users/dcazares/sglang/python
env RUSTUP_TOOLCHAIN=1.92 \
  SGLANG_BUILD_RUST_EXTS=server \
  ../.venv-mps/bin/python setup.py build_rust --inplace
```

This takes about 90 seconds on the M1 Max. Run it while the model server is
stopped so the result cannot contaminate a benchmark.

Focused test history:

- MLX attention/cache suite: **41 passed** (includes the Metal snapshot-clear
  regression), including 2 subtests.
- model override suite: **72 passed, 1 skipped**, including 2 subtests.
- q4 KV suite: **7 passed**.
- load-snapshot macOS `/dev/shm` parent-exists case: **passed**.
- Rust package: **247 passed** (`RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup"`
  required on macOS because workspace pyo3 enables `extension-module`).
- `cargo fmt --all -- --check` passed.
- `git diff --check` passed.

Recommended final validation after rebuilding:

```bash
cd /Users/dcazares/sglang
.venv-mps/bin/python -m pytest -q \
  test/registered/unit/hardware_backend/mlx/test_attention_patching.py \
  test/registered/unit/hardware_backend/mlx/test_quantized_kv_cache.py \
  test/registered/unit/test_model_overrides.py
```

```bash
cd /Users/dcazares/sglang/rust
env RUSTUP_TOOLCHAIN=1.92 cargo fmt --all -- --check
env RUSTUP_TOOLCHAIN=1.92 cargo test --release --package sglang-server
```

```bash
cd /Users/dcazares/sglang
git diff --check
.venv-mps/bin/ruff format --check \
  python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py \
  test/registered/unit/hardware_backend/mlx/test_quantized_kv_cache.py
```

The repository contains broader pre-existing Ruff findings in large legacy
files. Keep formatting changes scoped to the touched blocks.

## Current working tree

Modified tracked files at handoff:

```text
python/sglang/srt/arg_groups/overrides.py
python/sglang/srt/configs/model_config.py
python/sglang/srt/hardware_backend/mlx/kv_cache/__init__.py
python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py
python/sglang/srt/hardware_backend/mlx/kv_cache/attention_wrapper.py
python/sglang/srt/hardware_backend/mlx/kv_cache/auxiliary_state.py
python/sglang/srt/hardware_backend/mlx/model_runner.py
python/sglang/srt/hardware_backend/mlx/tp_worker.py
python/sglang/srt/managers/load_snapshot.py
python/sglang/srt/managers/rust_server.py
python/sglang/srt/managers/scheduler.py
python/sglang/srt/managers/tp_worker.py
python/sglang/srt/mem_cache/allocation.py
python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py
python/sglang/srt/server_args.py
rust/Cargo.lock
rust/sglang-server/Cargo.toml
rust/sglang-server/src/api_server/common.rs
rust/sglang-server/src/api_server/openai/chat.rs
rust/sglang-server/src/api_server/openai/reasoning.rs
rust/sglang-server/src/api_server/openai/template.rs
rust/sglang-server/src/mm.rs
rust/sglang-server/src/runtime/config.rs
test/registered/unit/hardware_backend/mlx/test_attention_patching.py
test/registered/unit/managers/test_load_snapshot_backends.py
test/registered/unit/test_model_overrides.py
```

Untracked work products:

```text
BENCHMARK.md
HANDOFF.md
scripts/apple_silicon_context_probe.rs
test/registered/unit/hardware_backend/mlx/test_quantized_kv_cache.py
```

Existing and new edits belong to the collaborator. Continue in place and use
targeted patches.

## Capability checks after the rebuild

Run these on the Rust server profile before promoting another benchmark:

1. `/v1/models` advertises `max_model_len` matching the active profile.
2. `/model_info` reports image and audio understanding disabled.
3. Thinking enabled streams parsed `reasoning_content` and a correct answer.
4. Thinking disabled streams final content and zero reasoning tokens.
5. `qwen3_coder` emits one parsed call with exact JSON arguments.
6. A tool-result continuation retains prior reasoning and returns 703.
7. Fast32K performs a 5,000-token two-chunk prefill and then records a
   4,096-token device-cache hit on the identical prefix.
8. Repeated unrelated requests survive recurrent-state pool pressure through
   checkpoint eviction/admission.
9. Max262K passes exact token-ID probes across geometric growth boundaries.

## Highest-value next work

1. **Cross the next geometric boundary.** A 32,769-token follow-up forces
   growth from 32K to 64K. A 65,536 actual prefill is the next practical
   capacity rung; it will take roughly 12–15 minutes at observed rates. A full
   262K prefill is expected to take well over 50 minutes and should be
   scheduled as a dedicated thermal and memory qualification run.
2. **Do not deepen lazy-decode chaining.** A 14 s `sample` of the scheduler
   during the 256-token control shows ~62% of the main thread in
   `async_eval` condvar wait (Metal still running) and ~17% in
   `gpu::eval` / quantized-matmul encode. Encode of step N+1 is already
   hidden. Extra in-flight graphs cannot retire token N earlier.
3. **A Rust/C++ MLX model port is the remaining decode lever, and it is
   ABI-blocked.** The six named Python/PyO3 hotspots are not the 51 ms/token
   bound. A compiled port needs stable bindings for MLX arrays, modules, lazy
   submission, quantization, and Metal sync, plus direct `mlx-lm` parity
   tests. Treat that as a separate project.
4. **Update `BENCHMARK.md` after every genuine win.** It is an incumbent
   snapshot. The post-rebuild Fast32K five-run was 18.743 tok/s aggregate /
   18.819 best and did not promote. Steady decode still hits 19.45–19.61 tok/s.
5. **Watch Metal residency on long sessions.** `store_cache` now calls
   `mx.clear_cache()` before cloning DeltaNet snapshots, and `remove_request`
   clears again. That survived seven consecutive 256-token jobs after the
   previous process died on `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
   Keep `SGLANG_MLX_CLEAR_CACHE_STEPS=0` for decode; do not put `clear_cache`
   on the per-token path.
6. **Graceful shutdown.** Ctrl-C still ends with Python `KeyboardInterrupt`
   and exit status 1; make the launcher return success after workers drain.
7. **Package the two launch profiles.** Fast32K is the live verified profile.
   Keep the control plane on `SGLANG_RUST_SERVER=1`.

## Performance discipline

- Run one model workload at a time.
- Let compilation, Spotlight activity, and other memory-bandwidth consumers
  settle before measurement.
- Warm the server before fixed-control samples.
- Use exactly five consecutive 256-token controls for decode promotion.
- Record both best and aggregate throughput.
- Treat prefill speed, decode speed, and maximum verified context as separate
  records.
- Keep request shape, cache type, radix mode, context allocation, and server
  mode attached to every result.
- Preserve output correctness and all reasoning/tool gates with every speed
  change.
- Keep failed and superseded trials out of `BENCHMARK.md`.

The inherited workload definitions and acceptance principles are in
[`notes/benchmark-contract.md`](notes/benchmark-contract.md). The current
incumbent-only record belongs in [`BENCHMARK.md`](BENCHMARK.md).

## Completion definition

This Apple-Silicon effort is ready to hand to a daily user when:

- Fast32K and Max262K have reproducible launch artifacts;
- the rebuilt Rust extension passes the full package and live capability gates;
- the maximum profile has a recorded, thermally stable capacity qualification;
- `BENCHMARK.md` reflects the highest qualified result for each independent
  workload;
- startup and graceful shutdown are clean;
- the remaining Python boundary is backed by a profile, with each feasible
  Rust migration either implemented and benchmarked or documented with its ABI
  prerequisite;
- coding-agent tool loops pass end to end from the intended client.
