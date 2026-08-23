# Benchmark and acceptance contract

Use this contract for every comparison with the qualified native-Windows
Qwen3.8 production line. A comparable result records the request shape, server
mode, cache treatment, sampling profile, graph state, GPU environment, and
resolved launcher arguments.

**Native-Windows reconciled through:** 2026-08-21 13:48 PDT.

**Apple M1 Max Q2 addendum reconciled through:** 2026-08-23 13:45 PDT.

## Primary performance scoreboard

The user-selected headline workload is the exact near-limit `199000+16`
request in the real 200K context and token pools. The current record to beat is
the launcher-default selective target-NVFP4 checkpoint at **3078.058 prompt
tok/s** and **114.617 generation tok/s**, with **64.651152 s TTFT**,
**64.782022 s** end to end, exactly `199016` completed tokens, and
`finish_reason=length`.

The next compact-scoreboard target is **3100 prompt / 120 generation tok/s**,
with **TTFT <=64.20 s** and **end-to-end time <=64.35 s** in the same exact
request. The time limits are derived from the throughput targets rather than
being independent goals.

A new overall record completes the same workload and exceeds both headline
throughput values under a matched environment record. This single-run
scoreboard is distinct from production qualification, which still requires
the behavior, repeated-sampling, capacity, relaunch, and client gates below.
The compact scoreboard is [`../BENCHMARK.md`](../BENCHMARK.md).

Candidate ranking still requires repeated matched controls. The request
produces its first token during prefill, leaving only 15 post-first-token
decode intervals. Current M3/M4 batches showed roughly 8-15% generation CV
with identical outputs, so one favorable short-run generation hit alone is not
evidence of a regression or win. Pair this headline request with
verification-cycle counts, a longer decode/acceptance window, and an A-B-A
control.

The record profile is the Windows launcher default: selective checkpoint,
chunk 7680, native draft-k1 one-hot q, selected large-EXTEND tactics, and
in-place Cutlass-prefill/Marlin-decode gate/up weights. Base RadixArk/chunk
4096 remains an explicit comparison control.

## Qualified reference

| Gate | Current reference |
|---|---|
| Real sampled `6213/512` | **122.712 tok/s** ten-run mean; **122.371** median; **137.074** peak |
| Fixed accepted-length-3 `6213/512` | **171.263 tok/s** five-run mean |
| Fixed output digest | `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c` |
| Native two-step acceptance | **2.318174** five-probe mean |
| Exact capacity `199000+16` | `199016` total; **2608.263 prompt**, **102.358 generation tok/s** |
| Accepted exact record `199000+16` | `199016` total; **3078.058 prompt**, **114.617 generation tok/s**, **64.651152 s TTFT**, **64.782022 s E2E** |
| Independent default relaunch | `199016` total; **3052.437 prompt**, **114.053 generation tok/s** |
| Production pool | Context `200000`; target/draft token pools `200000` |

Remeasure the reference after changes to source, checkpoint, dependencies,
launcher defaults, graph topology, GPU residency, driver/toolchain, or client
workload.

## Reference environment

The qualified run used:

- native Windows and an RTX 5090 display GPU;
- PyTorch `2.13.0+cu130`;
- CUDA runtime 13.0 with CUDA toolkit 13.3.33;
- Triton Windows `3.7.1.post27`;
- the clean Windows FlashInfer `0.6.17` port;
- the attention-selective RadixArk Qwen3.8-27B NVFP4 checkpoint;
- `.venv` launchers from this checkout;
- one server request at a time.

These versions and machine conditions can drift. Record the live driver,
Python packages, toolchain, commit/worktree state, clocks, power, utilization,
temperature, free VRAM, listener, process tree, and competing WDDM clients.

## Workloads

| Name | Shape and mode | Purpose |
|---|---|---|
| Smoke | `256/16` sampled | API, tokenizer, SSE, finish reason, basic output, and post-launch health |
| Historical control | `6213/128`, temperature 0 | Compare early GGUF and base-NVFP4 results |
| Current fixed control | `6213/512`, temperature 0, simulated accepted length 3 | Attribute deterministic execution and dispatch cost on the selected linear topology |
| Current real control | `6213/512`, normal rejection sampling | Measure production generation throughput |
| Sampled profile | Temperature `1.0`, top-p `0.95`, top-k `20`, presence `1.5` | Match the selected Qwen reasoning workload |
| Native acceptance | `bench_spec_acceptance.py` under sampled production settings | Pair TPS with emitted/accepted length, proposal counts, histograms, and verify cycles |
| Long ladder | `32768/16`, `32768/512`, `65536/16` | Catch prefill, residency, repeated-request, and long-decode regressions |
| Primary scoreboard and capacity gate | `199000/16` | Rank near-limit prompt/generation throughput and prove exact total `199016` inside the selected 200K pool |
| Real client | Standalone OpenCode2 with fixed provider/workload | Final reasoning, tool continuity, queue, parser, and wall-time integration |

`6213` input tokens come from the calibrated local OpenCode-shaped fixture in
[`../benchmark/windows/qwen38_local_prompt.json`](../benchmark/windows/qwen38_local_prompt.json).

## Standard commands

Primary 200K scoreboard:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600
```

Greedy current-shape control:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512
```

Sampled-profile control:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5
```

Reasoning-disabled sampled control:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5 --disable-thinking
```

The 2026-08-16 exploratory result for that mode was **129.722 tok/s** over ten
runs, with **2.452943** mean accepted length. It is a separate behavior profile
from the reasoning-preserved production contract.

Native speculative acceptance counters:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_spec_acceptance.py
```

Add `--disable-thinking` to take the matching non-thinking acceptance probe.

For deterministic linear fixed-work attribution, launch
`serve_qwen38_27b_nvfp4_5090.ps1 -SimulateAcceptedLength 3`. That server is an
execution probe and carries no semantic or production qualification.

## Measurement procedure

1. Inspect `git status`, the relevant diff, launcher defaults, installed
   dependencies, and any retained experimental flags.
2. Establish exact process ancestry, port 30000 ownership, compiler workers,
   GPU ownership, memory, temperature, clocks/power, and WDDM clients. Run one
   server, CUDA compilation/test tree, or benchmark workload at a time.
3. Confirm resolved server arguments, expected graph-capture markers,
   `/health`, `/model_info`, and the presence or absence of fixed-acceptance or
   tree/oracle controls.
4. Require prompt calibration to equal the requested token count exactly
   before cache flush, warmup, or measurement; the benchmark fails closed
   otherwise. Warm the exact request shape and flush cache explicitly. The
   benchmark reports TTFT, end-to-end time, prompt rate, steady decode rate,
   token counts, finish reason, output length, complete/per-channel digests,
   SSE fragment counts and sizes, and time after the final output fragment.
   Preserve these fields so text-fragment coalescing or delayed response
   closure cannot masquerade as a model-speed change.
5. Run at least five consecutive samples and report the mean plus every sample.
   A production promotion requires an independent second real-sampling window.
6. Pair real TPS with native acceptance counters. Record accepted/emitted
   length, correct/proposed drafts, histogram, and target verification cycles.
7. Preserve deterministic output digests for fixed work. For stochastic work,
   treat digest changes as an investigation signal alongside semantics,
   acceptance, and request seed.
8. Flush after long or memory-heavy requests before a steady-state inference.
   Recheck VRAM and health after the flush.
9. Stop only the verified server process tree, leaf-first. Confirm every known
   PID is absent, port 30000 is free, compiler/CUDA workers are gone, and the
   GPU has returned to ordinary display residency.
10. Append commands, resolved settings, raw samples, process evidence,
    conclusion, and handoff to [`experiment-log.md`](experiment-log.md).

Keep labels explicit: cold startup, warmed graph, cached prefix, cache-flushed,
seeded, simulated, unsimulated, contended, and clean control.

## Promotion gates

Production selection requires all of these:

- coherent preserved reasoning at the sampled profile;
- correct arithmetic final answer `703` for `37 * 19`;
- exactly one parsed `multiply({"a":37,"b":19})` call and correct arguments;
- `reasoning_content` continuity across tool use and ordinary stop behavior;
- `/model_info` showing image and audio understanding disabled;
- exact `199000+16` capacity after any memory layout, graph coverage,
  workspace, cache dtype, sampling residency, or context change;
- focused unit tests for the touched dispatch and fallback paths;
- Python compilation, PowerShell parsing, and `git diff --check`;
- native CUDA parity and graph-replay tests for captured kernel changes;
- two real sampled windows plus native acceptance evidence;
- an unsimulated production relaunch using launcher defaults;
- standalone OpenCode2 integration with fixed workload/provider shape;
- clean post-run process, listener, GPU, memory, and thermal evidence.

Use `scripts/windows/invoke_cuda_pytest.ps1` for native CUDA pytest work and
`scripts/windows/invoke_cuda_python.ps1` for native scripts. They initialize
the intended MSVC/CUDA 13.3 environment and cap compilation at two jobs.

## Apple M1 Max Q2 client-gate addendum

[`../BENCHMARK.md`](../BENCHMARK.md) governs the separate Apple M1 Max Q2
scoreboard. Its selected repository-native route uses Python ingress, Qwen's
official tokenizer, a 32,768-token BF16 pool, one request, and 1,024-token
prefill chunks. Signed PERF-A016 commit
`52b5326d8e5140b72a26a3909316fb1f665bbd3d` specializes the Q4_K tensor
family inside the mixed-format IQ2_XXS/Q2 checkpoint; the checkpoint and
scoreboard remain Q2.

The Apple real-client gate is Codex CLI 0.149.0 with the machine-local
`qwen38-local` profile over `/v1/responses`. Pin and record these overlay
identities for each qualification:

```text
9706003ad8a43ad48e4260f282057c023214c9e66737eae3da88a49188079a1c  $CODEX_HOME/qwen38-local.config.toml
a67c491a1dd4d4df0f720fb966ac390bd20041d8ed29f02833dfca4424a013f0  $CODEX_HOME/qwen38-local.models.json
```

The fixed read-only sequential-tool gate is:

```bash
env SGLANG_API_KEY=local codex exec -p qwen38-local --ephemeral \
  --color never -C /Users/dcazares/sglang --json \
  'Use the shell_command tool exactly once to run pwd in the current workspace. After reading its output, reply with exactly CODEX TOOL READY. Do not use any other tool.'
```

A passing window records the Codex version, both hashes, the pinned catalog
and observed command-tool surface, exact server ingress and resolved arguments,
process-scoped API key, request usage, one successful `shell_command`, consumed
tool output, exact visible final marker, zero client exit, unchanged worktree,
post-request server health, cache flush, leaf-first cleanup, free listener, and
returned memory/thermal state. The qualified 2026-08-23 run used `pwd`,
consumed `/Users/dcazares/sglang`, returned `CODEX TOOL READY`, and accounted
for 17,871 input, 96 output, and 62 reasoning-output tokens. This gate
qualifies the profile's read-only sequential shell surface. Windows production
continues to use the standalone OpenCode2 provider/workload contract above.

Every Apple request expected to run longer than five minutes remains under an
active controller-side watchdog. Poll at intervals of at most 60 seconds and
record the last completed prefill/decode progress, exact process ancestry and
listener, free-memory percentage, swap use, page throttling, and thermal or
performance warnings. Declare the per-forward deadline before launch from the
preceding rung; it may not exceed eight minutes without a separately recorded
calibration. A missed deadline, lost process/listener, throttled pages, or a
thermal/performance warning triggers verified leaf-first cleanup. The client
timeout is only the terminal request bound. It never substitutes for polling.

## Tree and SWOR experiments

The retained tree machinery is opt-in experimental infrastructure. Its
measurement contract adds these requirements:

- record the exact parent topology, draft width, proposal steps, verify nodes,
  sampling mode, target/draft attention routes, compile state, and active pool;
- measure emitted tokens per target traversal, accepted-node histograms,
  sibling rank contribution, graph replay spans, and real sampled TPS;
- use `scripts/windows/bench_target_verify_width.py` with one already-running
  server for width profiles;
- use `scripts/windows/analyze_torch_trace.py` for graph replay spans;
- use `scripts/windows/analyze_swor_topology.py` for retained path-stat logs;
- use `scripts/windows/optimize_swor_topology.py` for cost/yield projections;
- qualify exact distribution, support-exhaustion fallback, tree ancestry,
  recurrent-state commit, and CUDA-graph replay before a model launch.
- before retaining any tree throughput rank, run at least three deterministic
  cycles that include a non-front accepted branch and compare against a serial
  linear path reference: request virtual-slot mapping and physical target KV,
  compacted accepted tokens and hidden rows, rejected-slot reclamation,
  recurrent/GDN state, terminal next-draft token/hidden state, and next-cycle
  proposal or logits;
- run that accepted-path comparison on both eager and captured/device-cycle
  execution. A plausible response or matching token count does not substitute
  for state parity.

Proposal-geometry replay uses an immutable branch-exact corpus. Record child
and parent IDs, token IDs, depth, branch rank, post-transform target `p`, draft
`q`, branch-local counts and penalties, active worker, compile mode, topology
hash, and raw full-cycle device samples. A selected-tree capture qualifies only
the observed current membership. Every aligned, calibrated, variable-fanout,
SWOR, confidence-gated, or target-aware counterfactual fails closed when its
required proposal-lattice node or support is absent.

Aggregate geometry throughput as:

```text
TPS = 1000 * sum(E[L] per cycle) / sum(full-cycle milliseconds)
```

Never average per-cycle TPS ratios. A geometry candidate's conservative lower
TPS must strictly exceed the explicitly measured frontier's best-case upper
TPS. Reject a family when its impossible target-aware upper bound cannot exceed
**200 TPS**. Fund a production implementation only when a complete-lattice,
implementable policy retains a conservative lower projection of at least
**215 TPS**.

Target-graph attribution records exact mathematical GEMM `M,N,K`, aggregate
kernel residency, all-stream wall coverage, terminal-stream serialized
residency, and exclusive observed-wall exposure per shape. Overlapping kernel
time may exceed graph wall time; every optimized graph requires device-cycle
remeasurement. Graph-tail implementation work additionally requires at least
**0.75 ms** of repeatable recoverable time from asynchronous CUDA-event
timestamps.

Linear `SimulateAcceptedLength` produces contiguous accepted indices and does
not represent tree ancestry. Tree fixed-work and recurrent-state validation
use tree-aware fixtures or real exact verification.

## Interpretation rules established by the experiments

- Fixed accepted length isolates selected execution cost; proposal quality and
  ordinary semantics come from real rejection sampling.
- Real speculative throughput moves with acceptance. Interpret TPS together
  with accepted length and verification cycles.
- Exact `199000+16` generation is quantized by a small integer number of
  speculative cycles and measures only 15 post-first-token intervals. Use
  repeated A-B-A controls and a longer acceptance/cycle window; never classify
  a 2-3% single-run generation change by itself.
- `observed_prompt_tps` and `decode_tps` are client-observed SSE metrics. Their
  timing boundary is the first nonempty reasoning/content delta, not a
  server-side token event. Require matching fragment-count/size and
  trailing-response telemetry when comparing small deltas, and use device
  cycles as supporting attribution rather than silently redefining the
  headline metric.
- Exact seeds can lock response sequences across restarts for attribution, yet
  graph capture and RNG lifecycle can still change speculative work. Preserve
  production randomness.
- Source inspection, CPU tests, and kernel microbenchmarks establish mechanism
  evidence. Full-model serving establishes VRAM, graph, quality, and E2E value.
- A short near-limit generation is a capacity/routing gate rather than a
  stable decode benchmark.
- Repeated low samples with identical output can come from WDDM contention or
  residency pressure. Chrome, ZCode, Epic Games Launcher, and other desktop
  traffic produced confirmed interference.
- Fixed-width tree breadth followed a near-flat cost/yield frontier at M8/M12.
  The current-q topology search exhausted rearrangement as a route to the
  historical 200 tok/s target.
- A mechanism can remain valuable after losing production throughput. The
  online quantizers and exact tree stack retain compatibility, capacity, and
  future research value behind opt-in controls.
- Operating headroom is a production criterion. The 232K pool passed exact
  capacity and was rejected after falling to 98 MiB free before cache flush.
