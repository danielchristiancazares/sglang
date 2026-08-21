# Apple Silicon 32K Decode Benchmark

This is the primary performance scoreboard for the Apple Silicon
Qwen3.8-27B serving lane. The numbers that matter are measured on the warmed
fixed-decode workload: **12 prompt tokens plus 256 generated tokens** through
the Rust server with greedy sampling and one running request.

## Current record

| Metric | Current record |
|---|---:|
| Steady server decode | **19.58 tok/s** |
| End-to-end generation | **18.820713 tok/s** |
| Time to first token | **Unknown** |
| End-to-end time | **13.602035 s** |
| Five-run aggregate generation | **18.782925 tok/s** |

The five request-observed generation samples were
`18.761785, 18.772786, 18.820713, 18.774448, 18.784999 tok/s`, with a
**13.629400 s** mean end-to-end time. The direct `mlx-lm` engine result of
**20.458 tok/s** used a different `58+27` workload and remains a ceiling
reference rather than a server record.

This record was measured on an Apple M1 Max with 32 GPU cores and 32 GB of
unified memory using the full text side of Qwen3.8-27B.

| Area | Record profile |
|---|---|
| OS and hardware | macOS 26.6.1; M1 Max, 10 CPU cores, 32 GPU cores, 32 GB unified memory |
| Checkpoint | `mlx-community/Qwen3.8-27B-4bit` snapshot `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| Runtime | Python 3.11.15, MLX 0.32.0, mlx-lm 0.31.3, repository-pinned Rust 1.92 |
| Control plane | In-process Axum/Tokio Rust server with native OpenAI chat, template, reasoning, tool-call, and streaming paths |
| Source | SGLang base `e577394a9bb7e86a0f8b34c3575ee02d899d2915` plus the Apple Silicon worktree |
| Capacity and KV | 32,768-token context and pool; BF16 attention KV across 16 full-attention layers, approximately 2.0 GB at 32K |
| Cache and concurrency | Unified FULL radix plus recurrent auxiliary-state caching; one running request |
| Prefill | 4,096-token chunks; headless text trunk enabled for discarded chunks |
| Sampling and graphs | Native MLX sampling; benchmark temperature 0; optional Metal RoPE and CUDA graph backends disabled |
| Memory policy | `SGLANG_MLX_CLEAR_CACHE_STEPS=0`; 25.0 GB wired-memory limit |

Launch this profile with:

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

## Benchmark command

Run this fixed decode request five consecutive times after server warmup:

```bash
curl -sS -o /dev/null -w '%{time_total}\n' \
  -X POST http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Write a dense sequence of short Python identifiers separated by spaces.","sampling_params":{"temperature":0,"max_new_tokens":256,"ignore_eos":true}}'
```

A target result must complete all five fixed-length requests, exceed both
**18.782925 tok/s** aggregate generation and **18.820713 tok/s** best-hit
generation, and preserve every capability gate below. Long prefill,
maximum-context capacity, and direct-engine results remain independent records
because they measure different workloads.

## Maximum-context record

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

## Opt-in speculative profile

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
require the five-run control with the flag ON to clear the rule above plus
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

The broader inherited workload and acceptance definitions remain in
[`notes/benchmark-contract.md`](notes/benchmark-contract.md).

---

# Native-Windows 200K Context Benchmark

This is the primary performance scoreboard for the native-Windows
Qwen3.8-27B serving lane. The numbers that matter are measured at the exact
near-limit context workload: **199,000 prompt tokens plus 16 generated tokens
inside the real 200,000-token context and token pools**.

## Current record

| Metric | Current record |
|---|---:|
| Prompt processing | **3,078.058 tok/s** |
| Generation | **114.617 tok/s** |
| Time to first token | **64.651152 s** |
| End-to-end time | **64.782022 s** |

## Next target

| Metric | Next target |
|---|---:|
| Prompt processing | **>= 3,100 tok/s** |
| Generation | **>= 120 tok/s** |
| Time to first token | **<= 64.20 s** |
| End-to-end time | **<= 64.35 s** |

The two time targets are tied directly to the throughput targets on this exact
request: `199000 / 3100 = 64.1935` seconds to first token, followed by 15
measured decode intervals at 120 tok/s for approximately 64.32 seconds end to
end. Exact completion of `199016` tokens with `finish_reason=length` remains an
eligibility gate, not a fifth performance target.

This record was measured on the native-Windows launcher configuration with
benchmark seed `615388882`: the selective target-NVFP4 checkpoint
`C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4` with the
width-three NEXTN topology, chunk size 7680, native draft-k1 proposal
construction, and in-place Cutlass-prefill/Marlin-decode gate/up weights.
The target's ordinary 16,384-token EXTEND pass used the selected FlashInfer
FP4 tactics recorded in the detailed contract. The request completed with
`finish_reason=length`, exact `199000+16` usage, and output SHA-256
`9a0e20749e2930a697fefdd3bdd7863a067abe4d9860e6d1e7d9b80a62668b37`.

Launch the accepted profile with:

```powershell
.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1
```

An independent no-override relaunch with a process-selected seed reached
**3,052.437 prompt / 114.053 generation tok/s**, **65.193816 s TTFT**, and
**65.325334 s** end to end. It also beat all four prior record values in one
exact request. Base RadixArk and the older Cutlass/top-k20 route remain
available through explicit launcher overrides for controls.

## Benchmark command

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600
```

Run it against one deliberate native-Windows RTX 5090 server with the real
200K pools, ordinary inference, and every fixed-acceptance simulation disabled.
Record prompt throughput, generation throughput, TTFT, end-to-end time, token
counts, finish reason, resolved launcher arguments, GPU/process environment,
and cache treatment.

A target result must clear all four thresholds in the same exact request.
Because generation spans only 15 post-first-token intervals, it also requires
the repeated matched evidence defined by the detailed contract below.

Detailed qualification rules and historical evidence remain in
[`notes/benchmark-contract.md`](notes/benchmark-contract.md) and
[`notes/experiment-log.md`](notes/experiment-log.md).
