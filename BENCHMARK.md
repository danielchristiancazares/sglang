# Apple Silicon M1 Max Q2 32K Decode Benchmark

This is the primary performance scoreboard for the local Apple Silicon
Qwen3.8-27B lane. The fixed workload is **12 prompt tokens plus 256 generated
tokens**, greedy sampling, ignored EOS, and one running request.

The former affine-q4 entry came from a separate Mac Pro experiment and was
incorrectly attributed to this M1 Max. That cross-machine record and its
derived thresholds have been removed.

## Current measured Q2 record

| Metric | Current record |
|---|---:|
| Five-run aggregate generation | **14.661356 tok/s** |
| Best request-observed generation | **14.671473 tok/s** |
| Mean end-to-end time | **17.460868 s** |
| Best end-to-end time | **17.448827 s** |
| Mean internal decode | **14.784682 tok/s** |

The five request-observed samples were
`14.642054, 14.671473, 14.660470, 14.665758, 14.667059 tok/s`. Wall times were
`17.483886, 17.448827, 17.461923, 17.455627, 17.454079 s`. Every response
used the exact 12-token prompt, generated exactly 256 tokens, stopped at the
length limit, and retained FNV-1a-64 `6d4d220de481f54e`.

| Area | Record profile |
|---|---|
| OS and hardware | macOS 26.6.2; Apple M1 Max; 32 GPU cores; 32 GiB unified memory |
| Checkpoint | Bartowski `Qwen3.8-27B-IQ2_XXS.gguf`, revision `f0eec4a4bb4975114a030d048952d83c0a53c034` |
| Checkpoint SHA-256 | `b01f668356e5799fd76315bd6abc0e45234580409ebc5c8fb4b675e3c10dc2b9` |
| Runtime | Official llama.cpp build 10547 at commit `749f688fcaa4c472ec034b08cb8a907c45cfaa02` |
| Capacity and KV | 32,768-token context; one slot; FP16 K/V cache |
| Model surface | Full 27B text model; all layers on Metal; multimodal projector disabled |
| Sampling | Temperature zero; 256 forced output tokens; EOS ignored |

Launch the measured reference with:

```bash
/Users/dcazares/llama.cpp-749f688f/build-metal-release/bin/llama-server \
  --model /Users/dcazares/.cache/huggingface/hub/models--bartowski--Qwen3.8-27B-GGUF/blobs/b01f668356e5799fd76315bd6abc0e45234580409ebc5c8fb4b675e3c10dc2b9 \
  --alias qwen3.8-27b-iq2 \
  --ctx-size 32768 \
  --parallel 1 \
  --batch-size 4096 \
  --ubatch-size 512 \
  --n-gpu-layers all \
  --fit off \
  --flash-attn on \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --no-mmproj \
  --jinja \
  --reasoning-format deepseek \
  --reasoning on \
  --reasoning-preserve \
  --perf \
  --metrics \
  --offline \
  --no-webui \
  --host 127.0.0.1 \
  --port 30000 \
  --timeout 600
```

## Benchmark command

Warm the exact request once, then run it five consecutive times while
preserving the response body and full-precision wall time:

```bash
curl -sS -w '\n%{time_total}\n' \
  -X POST http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-iq2","prompt":"Write a dense sequence of short Python identifiers separated by spaces.","temperature":0,"max_tokens":256,"ignore_eos":true,"stream":false}'
```

For the repository-native SGLang route, send the same text through `/generate`
with `temperature=0`, `max_new_tokens=256`, and `ignore_eos=true`. Preserve
`meta_info`, `output_ids`, and the length finish; the Rust OpenAI completions
schema does not carry `ignore_eos`.

Compute aggregate throughput as `1280 / sum(the five wall times)`. A new
record completes every fixed-length request and strictly exceeds both
**14.661356 tok/s** aggregate generation and **14.671473 tok/s** best-hit
generation in the same five-run window.

### Repository-native SGLang baseline

The first fully matched Rust-ingress SGLang window at
`a35003d678b2363814a9c5e48d09e7abd3bd2a1a` used the same immutable IQ2_XXS
weights plus Qwen's official tokenizer snapshot
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`:

| Metric | Native SGLang baseline |
|---|---:|
| Five-run aggregate generation | **7.001584 tok/s** |
| Best request-observed generation | **7.015010 tok/s** |
| Mean end-to-end time | **36.563154 s** |
| Best end-to-end time | **36.493178 s** |
| llama.cpp record / native SGLang | **2.094006x** |

After one warmup, wall times were
`36.493178, 36.605944, 36.580286, 36.584450, 36.551911 s`; the corresponding
request rates were
`7.015010, 6.993400, 6.998305, 6.997509, 7.003738 tok/s`. Every response
reported 12 prompt and 256 completion tokens, stopped at the length limit,
returned the same 256 token IDs, and matched the record's 878-character
FNV-1a-64 `6d4d220de481f54e` output. The launch allocated the complete
32,768-token BF16 KV pool. The later selected Python-ingress route also passed
exact `32761+1` execution in that pool.

### Selected repository-native SGLang result

Signed commit `52b5326d8e5140b72a26a3909316fb1f665bbd3d` adds PERF-A016,
which reuses each activation fragment across two output rows for eligible
batch-one **Q4_K tensors inside this mixed-format IQ2_XXS/Q2 checkpoint**.
Record standing remains the Q2 checkpoint and M1 Max Q2 scoreboard; `Q4_K`
names the internal tensor family.

| Metric | Selected native SGLang |
|---|---:|
| Five-run aggregate generation | **8.586948 tok/s** |
| Best request-observed generation | **8.591773 tok/s** |
| Mean end-to-end time | **29.812688 s** |
| Best end-to-end time | **29.795946 s** |
| Gain over matched disabled-kernel control | **22.510241%** |
| llama.cpp Q2 record / selected SGLang | **1.707400x** |

After a 30.893266-second warmup, final-source Python-ingress wall times were
`29.801688, 29.820932, 29.824091, 29.795946, 29.820783 s`. The fresh matched
control aggregated **7.009167 tok/s** from
`36.534536, 36.515441, 36.531541, 36.523822, 36.512639 s`. An independent
candidate restart aggregated **8.578205 tok/s**. Every response retained the
same exact `12+256` usage, length finish, token IDs, text, and digest.

The selected route uses Qwen's official tokenizer, Python ingress, a 32,768-
token BF16 KV pool, one request, and 1,024-token prefill chunks. It passed
actual-file candidate/tail parity, sampled reasoning, thinking-disabled
behavior, parsed tool use and continuation, image/audio-disabled reporting,
exact `32761+1` capacity, and the named Codex profile gate below. The remaining
aggregate gap to the route-neutral llama.cpp Q2 record is **41.431420%**.

## Qualification gates

The measured reference loaded the full Q2 text model, reported image, video,
and audio disabled, preserved separate reasoning, and returned final `703` for
`37 * 19`. A promoted SGLang route must also retain:

- the exact 32,768-token context and token-pool allocation;
- sampled reasoning at temperature `1.0`, top-p `0.95`, top-k `20`, and
  presence penalty `1.5`;
- thinking-disabled exact `READY` with zero reasoning tokens;
- exactly one parsed `multiply({"a":37,"b":19})` tool call;
- preserved reasoning through the tool-result continuation;
- image and audio understanding disabled;
- an independent restart and second performance window;
- a 5,000-token two-chunk prefill and exact near-capacity evidence for changes
  that affect allocation, cache layout, or residency;
- standalone Codex CLI integration through the machine-local
  `qwen38-local` profile and `/v1/responses`, including one read-only
  `shell_command` round trip and its consumed result.

The selected client gate uses Codex CLI 0.149.0 with
`$CODEX_HOME/qwen38-local.config.toml` and
`$CODEX_HOME/qwen38-local.models.json`. The qualified task invoked `pwd`
exactly once, consumed `/Users/dcazares/sglang`, returned exact visible
`CODEX TOOL READY`, accounted for 62 reasoning-output tokens, and exited zero.
The earlier process-scoped OpenCode runs remain historical admission evidence.

Raw samples, exact process state, and the current native-SGLang Q2 handoff are
preserved in [`notes/experiment-log.md`](notes/experiment-log.md) and
[`notes/current-state.md`](notes/current-state.md).

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
