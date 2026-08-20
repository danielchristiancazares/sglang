# Benchmark and acceptance contract

Use this contract for every comparison with the qualified native-Windows
Qwen3.8 production line. A comparable result records the request shape, server
mode, cache treatment, sampling profile, graph state, GPU environment, and
resolved launcher arguments.

**Reconciled through:** 2026-08-20 12:29 PDT.

## Primary performance scoreboard

The user-selected headline workload is the exact near-limit `199000+16`
request in the real 200K context and token pools. The current record to beat is
the selective target-NVFP4 checkpoint at **3016.444 prompt tok/s** and
**112.355 generation tok/s**, with **65.971714 s TTFT**, **66.105219 s** end to end,
exactly `199016` completed tokens, and `finish_reason=length`.

The **3000 prompt / 110 generation tok/s** milestone is achieved. An
independent restart also reached **3013.736/112.012**, with 66.031008 s TTFT
and 66.164923 s end to end.

A new overall record completes the same workload and exceeds both headline
throughput values under a matched environment record. This single-run
scoreboard is distinct from production qualification, which still requires
the behavior, repeated-sampling, capacity, relaunch, and client gates below.
The compact scoreboard is [`../BENCHMARK.md`](../BENCHMARK.md).

Candidate ranking still requires repeated matched controls. The request
produces its first token during prefill, leaving only 15 post-first-token
decode intervals. Current M3/M4 batches showed roughly 8-15% generation CV
with identical outputs, so one generation hit above 110 tok/s is not evidence
of a regression or win. Pair this headline request with verification-cycle
counts, a longer decode/acceptance window, and an A-B-A control.

The record profile uses the selective checkpoint with explicit
`-ChunkedPrefillSize 7680` plus the bit-exact native-Windows Gemma
residual-norm direct-output path. Keep the production launcher chunk default
at 4096; base RadixArk regressed and lost operating headroom at 7680.

## Qualified reference

| Gate | Current reference |
|---|---|
| Real sampled `6213/512` | **122.712 tok/s** ten-run mean; **122.371** median; **137.074** peak |
| Fixed accepted-length-3 `6213/512` | **171.263 tok/s** five-run mean |
| Fixed output digest | `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c` |
| Native two-step acceptance | **2.318174** five-probe mean |
| Exact capacity `199000+16` | `199016` total; **2608.263 prompt**, **102.358 generation tok/s** |
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
- the RadixArk Qwen3.8-27B NVFP4 checkpoint;
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
4. Warm the exact request shape and flush cache explicitly. The benchmark
   reports TTFT, end-to-end time, prompt rate, steady decode rate, token counts,
   finish reason, output length, complete/per-channel digests, SSE fragment
   counts and sizes, and time after the final output fragment. Preserve these
   fields so text-fragment coalescing or delayed response closure cannot
   masquerade as a model-speed change.
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
