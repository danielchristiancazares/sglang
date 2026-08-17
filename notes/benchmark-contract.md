# Benchmark and acceptance contract

Use this contract for every comparison with the qualified native-Windows
Qwen3.8 production line. A comparable result records the request shape, server
mode, cache treatment, sampling profile, graph state, GPU environment, and
resolved launcher arguments.

**Reconciled through:** 2026-08-16 19:06 PDT.

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
| Capacity gate | `199000/16` | Prove exact total `199016` inside the selected 200K pool |
| Real client | Standalone OpenCode2 with fixed provider/workload | Final reasoning, tool continuity, queue, parser, and wall-time integration |

`6213` input tokens come from the calibrated local OpenCode-shaped fixture in
[`../benchmark/windows/qwen38_local_prompt.json`](../benchmark/windows/qwen38_local_prompt.json).

## Standard commands

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

Near-limit capacity:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600
```

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
   finish reason, output length, and digest.
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

Linear `SimulateAcceptedLength` produces contiguous accepted indices and does
not represent tree ancestry. Tree fixed-work and recurrent-state validation
use tree-aware fixtures or real exact verification.

## Interpretation rules established by the experiments

- Fixed accepted length isolates selected execution cost; proposal quality and
  ordinary semantics come from real rejection sampling.
- Real speculative throughput moves with acceptance. Interpret TPS together
  with accepted length and verification cycles.
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
