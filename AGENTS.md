# Agent guide for this checkout

This checkout carries a native-Windows Qwen3.8-27B performance and serving
lane alongside upstream SGLang. These instructions apply repository-wide.
[`docs/AGENTS.md`](docs/AGENTS.md) adds the documentation-site rules for work
under `docs/`.

Last reconciled with
[`notes/experiment-log.md`](notes/experiment-log.md) through
**2026-08-20 12:29 PDT**. A later experiment-log entry or fresh runtime
evidence supersedes every snapshot in this file.

## Recover context before acting

Start with the smallest source set that answers the task:

| Need | Read |
|---|---|
| Resume work or establish the handoff | [`notes/current-state.md`](notes/current-state.md), then any later entry in [`notes/experiment-log.md`](notes/experiment-log.md) |
| Change or revisit a selected default | [`notes/decisions.md`](notes/decisions.md) and any later experiment-log entries |
| Run or compare a benchmark | [`notes/benchmark-contract.md`](notes/benchmark-contract.md) |
| Understand experiment history | [`notes/timeline.md`](notes/timeline.md) |
| Recover exact samples, logs, failures, and intermediate state | [`notes/experiment-log.md`](notes/experiment-log.md) |

The compact documents and chronological experiment log are reconciled through
the qualified linear promotion, opt-in tree/SWOR experiments, rejected 232K
pool, and restored 200K production launch.

At the beginning of each task:

- inspect the branch, `HEAD`, full worktree status, and relevant diff;
- treat every existing modified or untracked path as user-owned work;
- verify listeners, process ancestry, GPU ownership, installed dependencies,
  and logs when they matter; recorded PIDs and process state are snapshots;
- inspect the resolved launcher arguments and live endpoint before describing
  a server as current or healthy.

## Preserve the recovery record

Append to [`notes/experiment-log.md`](notes/experiment-log.md) after meaningful
code changes, launches, measurements, failures, promotions, and cleanup.
Record the exact command or resolved arguments, commit/worktree state,
individual samples, environment, process ownership, result, and next handoff.
This is the recovery ledger; keep enough detail for a fresh agent to continue
after compaction or a crashed client.

Maintain the compact layer when conclusions change:

- update `notes/current-state.md` for a new qualified winner or handoff;
- add durable selections and closed candidates to `notes/decisions.md`;
- change `notes/benchmark-contract.md` when the workload or gates change;
- add a timeline phase for a material new direction;
- leave raw output, PIDs, incidents, and sample-by-sample narration in
  `notes/experiment-log.md`.

## Qualified production contract

The executable source of truth is
[`scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1`](scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1).
At the notebook cutoff, the accepted production configuration is:

- native Windows on the RTX 5090;
- checkpoint `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk`;
- endpoint `http://127.0.0.1:30000/v1`, served model `qwen3.8-27b`;
- real context and target/draft token pools of `200000`, one running request;
- language-only surface, Qwen3 reasoning parser, Qwen3 Coder tool parser;
- NEXTN linear rejection sampling, two speculative steps, three draft tokens,
  top-k-one chain, aligned draft sampling top-k 20 inside the single
  multi-step CUDA graph;
- FlashInfer prefill and sampling, TRT-LLM MHA/XQA target and draft decode,
  ReplaySSM linear speculation, FP8 E4M3 draft KV, checkpoint-selected target
  KV, 4096-token prefill chunks, and page size 64;
- FP32 Mamba state with four slots and `extra_buffer_lazy` caching;
- FP4 FlashInfer autotuning with FP8 GEMM autotuning skipped;
- torch compile mode `default`, batch-one full decode graphs, scheduler receive
  interval 4, stream interval 4, incremental output, and a 128 MiB FlashInfer
  workspace;
- every SWOR topology/oracle switch, adaptive depth, fixed-acceptance
  simulation, and explicit online draft quantizer left inactive.

The established results for that line are **122.712 tok/s** real sampled over
ten exact `6213/512` runs, **171.263 tok/s** for the safe fixed-work control,
and exact `199000+16` capacity. The final 200K relaunch captured all three
speculative graph phases with 1.84 GiB reported headroom. Treat these as the
comparison baseline and remeasure when code, dependencies, GPU environment,
or launcher defaults have moved.

The primary exact-200K scoreboard now also has a selective-checkpoint,
long-context-only profile:
`Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4` with
`-ChunkedPrefillSize 7680`. Combined with the bit-exact native-Windows Gemma
residual-norm direct-output path, it set the overall exact `199000+16` record
at **3016.444 prompt / 112.355 generation tok/s**, **65.971714 s TTFT**, and
**66.105219 s** end to end. An independent restart reached
**3013.736/112.012**. The 7680 chunk is not the production launcher default:
applying it globally to base RadixArk reduced its exact prompt result to
2226.770 tok/s and left only 200 MiB before follow-up probes, so production
remains at 4096. Use the explicit model/chunk override only for the selective
performance lane.

The exhaustive NVFP4 optimization run was marked complete. A new performance
branch begins from an explicit request or a newly measured gap. The historical
200 tok/s real-sampled objective remains experiment context rather than an
active authorization to resume GPU work.

## Behavior and capacity are part of performance

Every promoted candidate preserves all of these:

- exact 200,000 context and token-pool capacity, including a successful
  `199000+16` request when memory layout or residency changes;
- sampled reasoning at temperature `1.0`, top-p `0.95`, top-k `20`, and
  presence penalty `1.5`;
- coherent preserved `reasoning_content`, ordinary completion behavior, and
  the established arithmetic answer `703` for `37 * 19`;
- exactly one parsed `multiply({"a":37,"b":19})` call with
  `finish_reason=tool_calls`;
- `/model_info` reporting image and audio understanding disabled;
- an unsimulated launcher-default production relaunch with all intended CUDA
  graph captures;
- a standalone OpenCode2 integration check using the same provider and
  workload shape.

Keep OpenCode2's ordinary cloud-model configuration stable during server
tuning. Use a process-scoped model alias or wrapper for local title/workload
experiments, and preserve thinking continuity through the parser boundary.

## Measurement discipline

Use [`notes/benchmark-contract.md`](notes/benchmark-contract.md) as the full
contract. The standard controls are:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5
.\.venv\Scripts\python.exe .\scripts\windows\bench_spec_acceptance.py
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600
```

- Change one experimental variable at a time and compare with a fresh matched
  control.
- Capture at least five consecutive samples, their mean, and each individual
  result. Production promotion uses a second independent real-sampling window.
- Label cold, warmed, cached-prefix, cache-flushed, simulated, unsimulated,
  and externally contended measurements.
- Preserve token counts, finish reason, output length, and deterministic
  digests for fixed-work controls. Pair stochastic throughput with native
  acceptance counters and semantic output.
- Record clocks, power, utilization, free VRAM, and competing WDDM clients.
  Chrome, ZCode, Epic Games Launcher, and other desktop activity have produced
  measured contention.
- Flush the server cache after long or memory-heavy requests before drawing a
  steady-state conclusion.
- Treat fixed accepted length as an execution-cost probe. Semantic and
  production evidence comes from ordinary rejection sampling.
- Treat source inspection, CPU tests, microbenchmarks, and a single favorable
  stochastic window as intermediate evidence. Promotion requires the full
  behavior, capacity, real-client, and production-relaunch gates.

## Native-Windows GPU and process safety

Run one deliberate server, CUDA test/JIT build, compiler tree, or GPU benchmark
at a time. The 5090 is also the display GPU, and overlapping capture/compile
work has frozen the desktop.

- Establish the exact process ancestry, port owner, compiler workers, GPU
  owner, memory, utilization, and temperature before every GPU gate.
- Keep requests sequential and launches deliberate. Use one bounded readiness
  check after expected startup time.
- Stop only the verified server tree, leaf-first, using exact PIDs. Preserve
  unrelated MCP servers, OpenCode/ZCode processes, desktop clients, and every
  user-owned process.
- After shutdown, confirm the known PIDs are absent, port 30000 is free, CUDA
  and compiler workers are gone, and the GPU has returned to ordinary display
  residency.
- Keep compilation and extra CUDA contexts away from a resident production
  server; qualified launches can have only a few hundred MiB free after
  first-request JIT.
- Run native CUDA tests through
  `scripts/windows/invoke_cuda_pytest.ps1` and native scripts through
  `scripts/windows/invoke_cuda_python.ps1` so the intended MSVC/CUDA 13.3
  environment and two-job compiler limit are active.

## Implementation boundaries

- Performance hot-path implementations use C++/CUDA. Python is the thin
  binding, dispatch, configuration, test, and launch surface.
- Native-Windows Gemma residual normalization writes the existing bit-exact
  JIT result directly into caller-owned `x`; do not reintroduce its former
  temporary allocation and copy.
- Preserve upstream and non-Windows behavior behind narrow native-Windows
  dispatch gates. Keep experiments opt-in and launcher defaults production
  safe.
- Preserve asynchronous CUDA lifetimes. Stable graph inputs may be reused;
  per-cycle outputs that outlive a launch retain distinct storage.
- The linear `SimulateAcceptedLength` control models contiguous linear
  ancestry. Tree recurrent-state qualification uses real tree ancestry and
  accepted-path commit tests.
- Preserve the original FlashInfer checkout and the clean Windows 0.6.17 port
  as separate provenance lines.
- Leave the protected CUDA compatibility headers untouched. Their recorded
  SHA-256 is
  `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.
- Keep downloaded RadixArk and Gittensor source checkpoints immutable. Build
  any hybrid or converted artifact at a distinct path with provenance and
  checksums.
- Treat `sglang.bundle` as unrelated user-owned material.

For every touched surface, run focused tests, Python compilation, PowerShell
parsing where applicable, and `git diff --check`. Native kernel changes also
need isolated CUDA parity, CUDA-graph replay coverage when captured, then one
controlled full-model gate.

## Closed branches and retained experiments

Read [`notes/decisions.md`](notes/decisions.md) before reopening an older
candidate. The later notebook also closes these production branches under the
current topology:

- adaptive two/three-step depth, static three-step, and one-step MTP;
- reusable fused chain-metadata outputs with unsafe scheduling/lifetime cost;
- draft proposal top-k 8;
- full online FP8, MXFP8, and dense online NVFP4 draft weights for throughput;
- the stock Gittensor checkpoint as the production winner;
- 232K pools, which passed exact capacity yet fell to 98 MiB free before
  cache flush;
- the current target-only and SWOR tree proposal distributions, whose measured
  cost/yield remained below the qualified linear path.

The exact GPU tree verifier, low-rank GDN accepted-path commit, sparse SWOR
verifier, path/overlap oracles, topology analyzers, and their tests are retained
as opt-in experimental infrastructure. Preserve them. Reopening the tree route
requires measured proposal-overlap or draft-cost evidence that changes its
economics; topology rearrangement with the recorded q distribution has already
been exhausted.
