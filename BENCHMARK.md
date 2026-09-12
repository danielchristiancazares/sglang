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

### Current PERF-A021 native SGLang baseline

Signed commit `4dfa1ad3efdfe3f9236aa0ed0c841644ab513859` adds a
four-row batch-one Q2_K Metal matvec. This is the current Apple native-SGLang
baseline.

| Required baseline metric | Current PERF-A021 evidence | Workload |
|---|---:|---|
| Prompt processing | **22.945718 tok/s aggregate** | five exact `128+256` streaming requests |
| Generation | **9.156675 tok/s streaming**; **9.189086 tok/s fixed** | five exact `128+256` streams; five exact `12+256` fixed requests |
| Time to first token | **5.578383 s mean**; **5.521153 s best** | five exact `128+256` streaming requests |
| End-to-end time | **33.426973 s streaming mean**; **33.360443 s best**; **27.859136 s fixed mean** | exact `128+256` streams and `12+256` fixed requests |
| Capacity | **32,768-token pool; exact `32761+1` passed** | 32,762 total tokens, reasoning enabled |

The reasoning-enabled OpenAI chat template has a 52-token minimum, so the
streaming latency workload uses an exact 128-token prompt. After one internal
16-token warmup and one retained stabilization request, the five consecutive
cache-flushed streaming measurements were:

| Sample | Prompt tok/s | Generation tok/s | TTFT (s) | E2E (s) |
|---:|---:|---:|---:|---:|
| 1 | 23.184 | 9.144 | 5.521153 | 33.409791 |
| 2 | 22.836 | 9.175 | 5.605249 | 33.398367 |
| 3 | 22.977 | 9.130 | 5.570890 | 33.501759 |
| 4 | 22.940 | 9.145 | 5.579890 | 33.464504 |
| 5 | 22.797 | 9.191 | 5.614732 | 33.360443 |

Aggregate prompt throughput is `640 / sum(TTFT)`. Aggregate streaming
generation is `1275 / sum(last-token time - first-token time)`, measuring the
255 post-first-token intervals in each request. All five streams completed
exact `128+256` usage with `finish_reason=length`, 256 nonempty deltas, and
reasoning SHA-256
`3ea6b02f01fa96ad84bc9e8b3027a2af280c58ae68e14b8ea8408a18682b95a9`.

The cache-flushed current-source capacity request completed exact
`32761+1` usage with `finish_reason=length`, **18.942 observed prompt tok/s**,
**1729.565719 s TTFT**, and **1729.565822 s E2E**. It processed 31 full
1,024-token chunks plus a 1,017-token tail and returned reasoning SHA-256
`b344d80e24a3679999fa964450b34bc24d1578a35509f934c1418b0a20d21a67`.
Generation throughput is undefined for this one-token gate because it has
zero post-first-token intervals. The server cache was flushed immediately
afterward.

Reproduce the streaming and capacity rows against the launch above with:

```bash
.venv/bin/python scripts/windows/bench_openai_stream.py \
  --model qwen3.8-27b-iq2 --input-tokens 128 --output-tokens 256 \
  --temperature 0 --skip-warmup --timeout 600

.venv/bin/python scripts/windows/bench_openai_stream.py \
  --model qwen3.8-27b-iq2 --input-tokens 32761 --output-tokens 1 \
  --temperature 0 --skip-warmup --timeout 7200
```

The portable C++23 stream-client candidate is under `benchmark/native` and
uses the same server-owned tokenization, prompt bytes, request fields, SSE
metrics, and digest schema. Its native-Windows live parity window passes. The
Python command remains the Apple scoreboard authority through a matched live
parity window on the M1 Max. Build and promotion details are in
`notes/benchmark-contract.md`.

The independent five-request window used wall times
`27.842287, 27.856282, 27.867904, 27.868737, 27.860470 s`; its best
request-observed generation was **9.194647 tok/s**. The preceding candidate
window used `27.893382, 27.949060, 27.979015, 27.981408, 27.988928 s` and
aggregated **9.156475 tok/s**. Every response reported exact `12+256` usage,
length finish, and identical output IDs and text.

The matched generic-Q2_K control used wall times
`29.992152, 30.080830, 30.080217, 30.052984, 30.115632 s` and aggregated
**8.515065 tok/s**. PERF-A021's matched gain is **7.532647%**, and its gain
over the selected PERF-A016 aggregate is **7.012249%**.

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

# Native-Windows DSpark-v2 200K Serving Benchmark

This is the primary production scoreboard for the native-Windows Qwen3.8-27B
serving lane. Generation is measured with **6,213 uncached prompt tokens plus
512 generated tokens under ordinary Qwen sampling**: temperature 1.0, top-p
0.95, top-k 20, presence penalty 1.5, thinking enabled, and one admitted
request. Exact **199,000+16** execution inside the real 200,000-token target
and draft pools remains a separate mandatory capacity gate.

## Current production record

| Metric | Argument-free DSpark-v2 record |
|---|---:|
| Five-run mean generation | **162.500 tok/s** |
| Best request generation | **191.357 tok/s** |
| Mean prompt processing | **12,604.673 tok/s** |
| Mean time to first token | **0.492930 s** |
| Mean end-to-end time | **3.663612 s** |
| Capacity | **200,000 target + 200,000 draft tokens; exact `199000+16` passed** |

The five consecutive samples were:

| Sample | Prompt tok/s | Generation tok/s | TTFT (s) | E2E (s) |
|---:|---:|---:|---:|---:|
| 1 | 12,676.237 | **151.139** | 0.490130 | 3.871124 |
| 2 | 12,595.924 | **165.951** | 0.493255 | 3.572485 |
| 3 | 12,513.812 | **151.000** | 0.496491 | 3.880595 |
| 4 | 12,704.150 | **191.357** | 0.489053 | 3.159452 |
| 5 | 12,533.242 | **153.054** | 0.495722 | 3.834405 |

Every sample individually reached at least 150 tok/s and returned exact
`6213+512`, `finish_reason=length`, and preserved ordinary reasoning/content.
An independent exact-200K explicit-switch window measured
`157.176, 173.614, 135.338, 152.884, 160.793 tok/s`, mean **155.961 tok/s**.

The literal argument-free launcher capacity result was:

| Capacity metric | Result |
|---|---:|
| Prompt processing | **3,118.215 tok/s** |
| Time to first token | **63.818572 s** |
| End-to-end time | **64.236571 s** |
| Short 16-token decode | **35.885 tok/s** |
| Token/finish contract | **199,000+16=199,016; `length`** |

The 16-token rate spans only 15 post-first-token intervals and is reported as
capacity telemetry, not substituted for the 512-token generation scoreboard.
An independent explicit-switch capacity run reached 3,118.323 prompt tok/s,
63.816352 s TTFT, and 64.233185 s E2E with the same deterministic digest.

This record uses the selective target-NVFP4 checkpoint
`C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4`, immutable
trained draft `C:\Users\Daniel\models\Qwen3.8-27B-DSpark-v2`, online-FP8
draft weights, gamma seven/eight-row linear verification, 4,096-token prefill
chunks, five FP32 Mamba slots, FP8 target/draft KV, Triton draft attention,
TRT-LLM target verify/decode, folded draft proposal/sampling, static
target-graph draft-KV commit, and in-place Cutlass-prefill/Marlin-decode target
gate/up weights. The server keeps exact 200K pools, one request, Qwen3
reasoning/Qwen3-Coder tools, and a language-only model surface.

Launch the accepted profile with:

```powershell
.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1
```

The earlier NEXTN/chunk-7680 exact-16 record remains a historical long-context
control. Base RadixArk and the older Cutlass/top-k20 route remain available
through explicit launcher overrides. No new native-Windows performance target
is active after qualification of the requested 150 tok/s production gate.

## Benchmark command

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600
```

The portable CPU-only C++23 candidate is
`benchmark/native/bench_openai_stream.cpp`; its shared native support also
drives `bench_spec_acceptance.cpp`. Strict GCC and MSVC host suites pass. The
native-Windows adjacent live window passes greedy and sampled `6213+512`,
sampled acceptance, and exact-capacity `199000+16`, preserving workload,
schema, deterministic hashes, acceptance algebra, and timing boundaries. The
Python command remains the cross-platform record authority until the Apple
window passes. See `notes/benchmark-contract.md` for build commands and the
promotion gate.

Run it against one deliberate native-Windows RTX 5090 server with the real
200K pools, ordinary inference, and every fixed-acceptance simulation disabled.
Record prompt throughput, generation throughput, TTFT, end-to-end time, token
counts, finish reason, resolved launcher arguments, GPU/process environment,
and cache treatment.

A replacement production result must preserve exact 200K capacity and every
behavior/client gate, then clear its declared ordinary-sampling threshold in
at least five clean exact `6213+512` samples and an independent launcher
restart. A short exact-16 capacity result alone is not generation-promotion
evidence.

Detailed qualification rules and historical evidence remain in
[`notes/benchmark-contract.md`](notes/benchmark-contract.md) and
[`notes/experiment-log.md`](notes/experiment-log.md).

## NVIDIA stock checkpoint evaluation - 2026-09-08

`nvidia/Qwen3.8-27B-NVFP4` revision
`fed99d815f4e8c7c616dd3dd6780076e26d5fb61` is installed and works through
the existing native-Windows launcher without source or dependency changes.
It is a **mixed NVFP4/FP8** checkpoint with **21.922 GB / 20.416 GiB** of
weight files. All 19 upstream file sizes and all upstream LFS hashes were
verified. The production attention-selective RadixArk/DSpark default above
remains unchanged.

The checkpoint-only comparison below uses **target-only serving**, not
speculative decoding: real 200K context and target KV, five FP32 Mamba slots,
FP8 KV, chunk 4096, one request, full batch-one decode graphs, and the same
FlashInfer/hybrid-Marlin backends. Both checkpoints use the identical tokenizer
and chat template. The RTX 5090 used driver `616.56`; ordinary desktop clients
and unrelated CPU work were retained.

Each row averages five consecutive exact uncached **6213+512** requests with
one 16-token same-shape warmup per sample, temperature 1.0, top-p 0.95, top-k
20, presence penalty 1.5, and thinking enabled:

| Stock checkpoint | Prompt tok/s | Generation tok/s | TTFT (s) | E2E (s) |
|---|---:|---:|---:|---:|
| NVIDIA NVFP4/FP8 | **9,469.619** | **64.064** | **0.656153** | **8.633376** |
| Original RadixArk NVFP4/FP8 | **9,552.382** | **63.996** | **0.650634** | **8.635972** |

NVIDIA generation samples were
`63.234, 63.269, 64.686, 64.553, 64.578 tok/s`; original RadixArk samples
were `63.091, 63.849, 64.272, 64.238, 64.530 tok/s`. NVIDIA's **0.11%**
mean generation difference is within the observed run variation: there is
**no material throughput improvement** in this matched target-only comparison.
The original stock RadixArk control is not the attention-selective DSpark
production configuration.

NVIDIA also passed exact **199000+16=199016**, including a same-shape warmup,
at **2,720.158 prompt tok/s**, **73.157527 s TTFT**, and
**73.500594 s E2E**. The short 16-token decode rate was 43.723 tok/s.
Sampled arithmetic, preserved reasoning, exactly one parsed multiply call,
tool-result continuation, thinking-disabled output, and language-only
reporting passed. Cache flush after the capacity gate restored **2,660 MiB
free**. NVIDIA speculative serving, independent-restart/client promotion
gates, and the published accuracy benchmark suites were not run.

Reproduce the installed option without changing production defaults:

```powershell
.\.venv\Scripts\hf.exe download nvidia/Qwen3.8-27B-NVFP4 `
  --revision fed99d815f4e8c7c616dd3dd6780076e26d5fb61 `
  --local-dir C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-NVIDIA `
  --max-workers 3 --quiet

.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1 `
  -ModelPath C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-NVIDIA `
  -SpeculativeNumSteps 0
```

All benchmark servers were stopped after measurement; port 30000 is free.
Full samples, output digests, exact capacity data, source/checkpoint
provenance, and GPU brackets are preserved in
[`benchmark/windows/nvidia_qwen38_20260908.json`](benchmark/windows/nvidia_qwen38_20260908.json).
The initial higher-residency DSpark control is retained there but excluded
from the checkpoint-only comparison. See the September 8 entries in
[`notes/experiment-log.md`](notes/experiment-log.md) for the complete handoff.
