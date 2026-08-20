# Qwen3.8-27B Apple Silicon Benchmark Incumbent

This file is the **current winning snapshot**, not a benchmark diary. Replace
an incumbent only when a reproducible run is faster for the same workload and
still passes the capability gates below. Keep failed and superseded trials out
of this file.

## Current winner

| Metric | Best hit | Five-run result | Workload |
|---|---:|---:|---|
| Steady server decode | **19.58 tok/s** | typically 19.42–19.58 tok/s | 12 prompt tokens, 256 generated tokens |
| End-to-end generation | **18.821 tok/s** | **18.783 tok/s aggregate** | 12 prompt tokens, 256 generated tokens, including request and prefill overhead |
| Cold long prefill | **85.763 tok/s** | one capacity-path probe | 5,000 prompt tokens, 1 generated token, 58.300 s end to end |
| Direct engine reference | **20.458 tok/s** | one short reference probe | `mlx-lm`, 58 prompt tokens, 27 generated tokens |

The production-serving number is **19.58 tok/s steady decode**. The
request-observed number is **18.821 tok/s best / 18.783 tok/s five-run
aggregate**.
The direct-engine result is a ceiling reference with a shorter workload and is
not a server promotion result.

### Five-run samples

Fixed output length: 256 tokens.

| Run | End-to-end time | Output throughput |
|---:|---:|---:|
| 1 | 13.644757 s | 18.761785 tok/s |
| 2 | 13.636761 s | 18.772786 tok/s |
| 3 | **13.602035 s** | **18.820713 tok/s** |
| 4 | 13.635554 s | 18.774448 tok/s |
| 5 | 13.627895 s | 18.784999 tok/s |
| Aggregate | **13.629400 s mean** | **18.782925 tok/s** |

## Winning configuration

- **Machine:** Apple M1 Max, 10 CPU cores, 32 GPU cores, 32 GB unified memory
- **OS:** macOS 26.6.1
- **Model:** full text side of Qwen3.8-27B, MLX affine 4-bit/group-size 64
- **Checkpoint:** `mlx-community/Qwen3.8-27B-4bit`
- **Snapshot:** `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`
- **Python:** 3.11.15
- **MLX:** 0.32.0
- **mlx-lm:** 0.31.3
- **Rust:** 1.92, repository-pinned toolchain
- **Control plane:** in-process Axum/Tokio Rust server with native OpenAI chat,
  template, reasoning, tool-call, and streaming paths
- **SGLang base:** `e577394a9bb7e86a0f8b34c3575ee02d899d2915` plus the current Apple-Silicon worktree
- **Context allocation:** 32,768 tokens
- **Attention KV:** BF16, 16 full-attention layers, approximately 2.0 GB at 32K
- **Concurrency:** one running request
- **Radix cache:** unified FULL + recurrent auxiliary-state components
- **Chunked prefill:** 4,096 tokens; headless text trunk enabled for discarded chunks
- **Sampling:** native MLX sampling enabled; benchmark uses greedy temperature 0
- **Optional Metal RoPE:** disabled in the winner
- **MLX buffer-cache clearing:** disabled (`SGLANG_MLX_CLEAR_CACHE_STEPS=0`)
- **Wired-memory limit:** 25.0 GB

Resolved launch shape:

```bash
env \
  SGLANG_USE_MLX=1 \
  SGLANG_RUST_SERVER=1 \
  SGLANG_MLX_CLEAR_CACHE_STEPS=0 \
  .venv-mps/bin/python -m sglang.launch_server \
  --model-path "$MODEL_SNAPSHOT" \
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

Fixed decode-control request, run five consecutive times after server warmup:

```bash
curl -sS -o /dev/null -w '%{time_total}\n' \
  -X POST http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Write a dense sequence of short Python identifiers separated by spaces.","sampling_params":{"temperature":0,"max_new_tokens":256,"ignore_eos":true}}'
```

## Maximum-context incumbent

The independent capacity winner advertises the full **262,144-token** native
context with affine q4 attention KV. Its cache grows geometrically from 4,096
tokens to the configured cap, avoiding repeated 256-token reallocations.

| Metric | Best qualified hit | Workload |
|---|---:|---|
| Verified q4 long prefill | **84.493486 prompt tok/s** | 5,000 exact prompt tokens + 1 output, 59.176159 s |
| Largest completed exact probe | **32,768 prompt tokens** | 1 output, 425.299138 s, 77.046947 prompt tok/s |
| Short steady decode | **19.46 tok/s** | fixed short control under native 262K allocation |

Maximum-context configuration differences:

- `--context-length 262144 --max-total-tokens 262144`
- `--disable-radix-cache`
- `--mlx-kv-cache-bits 4 --mlx-kv-cache-group-size 64`
- affine q4 KV size: 18,432 bytes/token across 16 full-attention layers,
  exactly 4.5 GiB at 262,144 tokens
- one running request, with the same Rust control plane, chunked prefill,
  reasoning, tool, and sampling settings as the speed winner

Verified exact-token capacity rungs under this profile:

| Prompt tokens | Output tokens | End-to-end time | Prompt throughput |
|---:|---:|---:|---:|
| 5,000 | 1 | 59.176159 s | **84.493486 tok/s** |
| 16,384 | 1 | 204.230414 s | **80.223115 tok/s** |
| 32,768 | 1 | 425.299138 s | **77.046947 tok/s** |

## Speculative decoding profile (flag-gated, off in the speed winner)

`--mlx-mtp-path <mtp.safetensors>` loads the Qwen3.8 multi-token-prediction
head (`Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`, q4-quantized at load)
and speculates depth-3 with single-forward batched verification. Output is a
greedy trunk stream: drafts are accepted only when they equal the trunk's own
argmax (verify logits come from a batched Metal matvec whose fp32 reduction
order differs from single-token decode at ~1e-3 relative). An adaptive policy
(trailing 12-round acceptance vs. the ~3.0 tokens/round breakeven, doubling
AR fallback stretches) bounds hostile prompts. Requires
`--disable-radix-cache`, BF16 attention KV, greedy single-request decode.

Measured through `MlxModelRunner` (paired in-process, 96 tokens, coding
prompt): **20.94 vs 18.22 tok/s (+2.72, 1.15x), outputs identical to the
runner's own greedy decode in all pairs.** Engine-level suite (160 tokens,
4 pairs each, adaptive on): write-code +1.37, explain -0.59, refactor +4.41,
tool-JSON +6.30, dense-identifier control -0.68 (breakeven +-0.7 across
runs); mean +2.16, coding-workload mean +2.87 tok/s.

Through the full server (Rust control plane, `/generate`, temperature 0,
`ignore_eos`, radix off in both arms, medians of 3):

| Workload | Flag off | Flag on | Delta |
|---|---:|---:|---:|
| LRU-cache prompt, 768 tokens | 19.05 tok/s | **19.88 tok/s** | +0.83 |
| Tool-JSON edit commands, 384 tokens | 18.68 tok/s | **19.26 tok/s** | +0.58 |
| LRU-cache prompt, 256 tokens | 18.32 tok/s | 18.86 best | +0.5 best, noisy |

The engine-to-server gap is quantified: scheduler/streaming overhead
serializes with GPU work during buffer pops (~4-5 ms/round; launching the
next round during pops was measured to cost MORE via the extra Metal
command-buffer split), and think-prose stretches sit at the acceptance
breakeven, tripping the adaptive policy. The engine logs per-request round
stats (`mtp[...]` lines) for further tuning.

The fixed decode control and both incumbents above run with the flag OFF and
are unaffected. Promotion of this profile into the speed winner would
require the five-run control with the flag ON to clear the rule below plus
all capability gates; the control prompt sits at breakeven, so the flag
stays opt-in for agent workloads.

## Capability gates passed by this winner

- Full 27B text model loaded; no layer or expert offload.
- Language-only loading disables the image and audio model paths. Live Rust
  `/model_info` reports `has_image_understanding` and
  `has_audio_understanding` as false.
- Thinking enabled returns parsed `reasoning_content` and a correct final answer.
- Thinking disabled returns `READY` with zero reasoning tokens.
- `qwen3_coder` emits exactly one parsed
  `multiply({"a": 37, "b": 19})` tool call.
- A preserved-thinking tool-result turn returns the correct final value, `703`.
- Unified radix FULL + recurrent-state caching remains enabled.
- Repeated unrelated requests survive recurrent-state pool pressure and evict
  retained checkpoints for admission.
- A 5,000-token, two-chunk prefill completes; an identical follow-up reports a
  4,096-token device-cache hit.

## Promotion rule

For this fixed decode control, promote only when all five consecutive runs
complete, aggregate output throughput exceeds **18.782925 tok/s**, the best hit
exceeds **18.820713 tok/s**, and every capability gate still passes. Record long
prefill and context-capacity winners independently because they measure a
different part of the system.

The broader inherited workload and acceptance definitions remain in
[`notes/benchmark-contract.md`](notes/benchmark-contract.md).
---

# Native-Windows 200K Context Benchmark

This is the primary performance scoreboard for the native-Windows
Qwen3.8-27B serving lane. The numbers that matter are measured at the exact
near-limit context workload: **199,000 prompt tokens plus 16 generated tokens
inside the real 200,000-token context and token pools**.

## Record to beat

| Metric | Current record |
|---|---:|
| Prompt processing | **3,016.444 tok/s** |
| Generation | **112.355 tok/s** |
| Time to first token | **65.971714 s** |
| End-to-end time | **66.105219 s** |
| Tokens completed | **199,016** |

This record was measured on the selective target-NVFP4 checkpoint
`C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4` with the
width-three NEXTN topology, chunk size 7680, and the bit-exact native-Windows
Gemma residual-norm direct-output path. The request completed successfully
with `finish_reason=length` and retained the established digest.

## Achieved target

| Metric | Milestone |
|---|---:|
| Prompt processing | **3,000 tok/s** |
| Generation | **110 tok/s** |
| Time to first token | **<= 66.33 s** |
| End-to-end time | **<= 66.5 s** |
| Tokens completed | **199,016** |

The current record clears every milestone value in one exact request. A second
independent server launch also cleared both throughput targets at
**3,013.736 prompt / 112.012 generation tok/s**, with 66.031008 s TTFT and
66.164923 s end to end.

## Qualification windows

Eight exact `199000+16` samples across two independent server launches all
cleared 3,000 prompt tok/s:

| Window/run | Prompt tok/s | Generation tok/s | TTFT | E2E |
|---|---:|---:|---:|---:|
| 1/1 | **3,016.444** | **112.355** | 65.971714 s | 66.105219 s |
| 1/2 | 3,013.834 | 97.506 | 66.028859 s | 66.182696 s |
| 1/3 | 3,013.975 | **112.534** | 66.025761 s | 66.159054 s |
| 2/1 | 3,014.657 | 96.531 | 66.010835 s | 66.166226 s |
| 2/2 | 3,009.496 | 86.114 | 66.124024 s | 66.298210 s |
| 2/3 | 3,012.204 | 98.100 | 66.064592 s | 66.217497 s |
| 2/4 | **3,013.736** | **112.012** | 66.031008 s | 66.164923 s |
| 2/5 | 3,011.489 | 79.442 | 66.080266 s | 66.269082 s |

Prompt averaged **3,013.229 tok/s** across all eight. The 16-token generation
metric remains cycle-quantized; three samples cleared 110 tok/s. Two exact
`199000+512` support runs averaged **3,013.443 prompt / 109.683 generation
tok/s** and peaked at 111.094 generation tok/s. Two sampled `6213/512`
windows averaged **144.535** and **138.621 tok/s**.

Launch this opt-in profile with:

```powershell
.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1 `
  -ModelPath C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4 `
  -ChunkedPrefillSize 7680
```

Do not make 7680 the global launcher default. Base RadixArk regressed to
2,226.770 prompt tok/s on exact `199000+16` and fell to 200 MiB free before
follow-up probes. Its production default remains 4096.

## Qualified production baseline

| Metric | Qualified baseline |
|---|---:|
| Prompt processing | **2,608.263 tok/s** |
| Generation | **102.358 tok/s** |
| End-to-end time | **76.442544 s** |
| Tokens completed | **199,016** |

The qualified baseline uses
`C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk` and the real 200K
production configuration.

## Benchmark command

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600
```

Run it against one deliberate native-Windows RTX 5090 server with the real
200K pools, ordinary inference, and every fixed-acceptance simulation disabled.
Record prompt throughput, generation throughput, TTFT, end-to-end time, token
counts, finish reason, resolved launcher arguments, GPU/process environment,
and cache treatment.

An overall record must complete exactly **199,016 tokens** and beat both
**3,016.444 prompt tok/s** and **112.355 generation tok/s** under the matched
contract. Lower TTFT and end-to-end time are supporting wins.

Detailed qualification rules and historical evidence remain in
[`notes/benchmark-contract.md`](notes/benchmark-contract.md) and
[`notes/experiment-log.md`](notes/experiment-log.md).
