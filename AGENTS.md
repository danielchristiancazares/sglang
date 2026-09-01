# AGENTS.md

Do NOT add new Python code. Anything new must be in C++ or CUDA.

# Read First

This checkout carries a native-Windows Qwen3.8-27B performance and serving
lane alongside upstream SGLang. These instructions apply repository-wide.
[`docs/AGENTS.md`](docs/AGENTS.md) adds the documentation-site rules for work
under `docs/`.

Last reconciled with
[`notes/experiment-log.md`](notes/experiment-log.md) through
**2026-08-30 18:04 PDT**. A later experiment-log entry or fresh runtime
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

## Start Qwen3.8-27B for Codex

This is the local Codex setup for the qualified native-Windows lane. It uses
the repository launcher, its measured defaults, and one additional FP32 Mamba
cache slot for Codex's multi-chunk repository prompts. The derived checkpoint
is a local artifact assembled from the immutable RadixArk base and NVFP4
donor; its `selective-nvfp4-manifest.json` is part of the checkpoint
provenance.

### One-time setup

From PowerShell 7 at the repository root, confirm that the checked-in virtual
environment, checkpoint, and manifest are present:

```powershell
Test-Path .\.venv\Scripts\sglang.exe
Test-Path C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4
Test-Path C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4\selective-nvfp4-manifest.json
```

All three commands must return `True`. The launcher initializes the qualified
MSVC/CUDA 13.3 environment itself. Keep the model artifact at its recorded path
or pass a fully qualified `-ModelPath`; never modify the downloaded source
checkpoints in place.

Validate that the editable SGLang install resolves to this checkout:

```powershell
.\.venv\Scripts\sglang.exe --help
.\.venv\Scripts\python.exe -c "import sglang; print(sglang.__file__)"
```

The import must resolve below `C:\Users\Daniel\sglang\python`. Moving this
checkout can leave the uv-generated executable and editable-install metadata
pointing at its former location. Repair that state while holding dependencies
fixed, then rerun both checks:

```powershell
uv pip install --python C:\Users\Daniel\sglang\.venv\Scripts\python.exe `
  --editable C:\Users\Daniel\sglang\python `
  --no-deps `
  --reinstall-package sglang
```

This setup is validated with Codex CLI 0.151.0. Create or verify the dedicated
profile `C:\Users\Daniel\.codex\qwen38.config.toml` with this content:

```toml
model = "qwen3.8-27b"
model_provider = "sglang-qwen38"
model_context_window = 200000
model_auto_compact_token_limit = 180000
service_tier = "default"

[model_providers.sglang-qwen38]
name = "Local SGLang Qwen3.8"
base_url = "http://127.0.0.1:30000/v1"
wire_api = "responses"
requires_openai_auth = false
```

This profile layers over the regular Codex configuration only when selected;
it leaves the ordinary cloud model and authentication configuration unchanged.
The 180K compaction threshold reserves room inside the server's real 200K
token pool for instructions, tool results, and the next response. Codex appends
`/responses` to the provider base URL, so keep `/v1` in `base_url`.

### Preflight and start

The RTX 5090 is also the display GPU. Before launch, exit any earlier
`codex -p qwen38` session whose request is still reconnecting. Codex's
unbounded reconnect mode retains failed turns across a server restart; two
retained turns can arrive together when the listener returns. Then confirm
port 30000 is free, no SGLang/CUDA compiler tree is active, and the GPU has
returned to ordinary display residency. Follow the process-safety rules below
if anything is already running. Open a dedicated PowerShell 7 terminal and run
the Codex lane in the foreground:

```powershell
Set-Location C:\Users\Daniel\sglang
.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1 -MaxMambaCacheSize 5
```

This retains the selected checkpoint, reasoning and tool parsers, real 200K
context/token pools, one-request scheduler, CUDA graphs, and speculative-decode
settings. The fifth Mamba slot is the Codex-specific reserve for the transient
state donation that occurs when an unfinished 7,680-token prefill chunk enters
the radix cache. A real 10.5K-token Codex prompt exhausted the four-slot pool
at that boundary; the five-slot lane passed the same multi-chunk boundary,
retained-cache pressure, exact `199000+16` capacity, and a post-capacity Codex
request. The argument-free four-slot launcher remains the qualified production
benchmark baseline.

Leave the launcher terminal open. Startup loads the checkpoint, JIT-compiles
kernels, and captures three CUDA graph phases, so wait for the ready message
before sending traffic. The resolved `server_args` must include
`max_mamba_cache_size=5`, `max_running_requests=1`,
`max_total_tokens=200000`, and `context_length=200000`. Confirm target verify,
draft decode, and draft extend graph completion, then use one bounded readiness
check after the expected startup interval:

```powershell
Invoke-RestMethod http://127.0.0.1:30000/health
Invoke-RestMethod http://127.0.0.1:30000/v1/models | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:30000/model_info | ConvertTo-Json -Depth 8
```

The model list must contain `qwen3.8-27b`; `/model_info` must report image and
audio understanding disabled. A refused connection means the server is stopped
or still starting. Exit any reconnecting Qwen Codex session and inspect the
launcher terminal before a restart.

### Start and verify Codex

In another PowerShell terminal, start Codex with the local profile:

```powershell
Set-Location C:\Users\Daniel\sglang
codex -p qwen38
```

The startup banner must show model `qwen3.8-27b` and provider
`sglang-qwen38`. The disposable qualification gate uses a real Code Mode tool
round trip and the multi-chunk Tombstead prompt shape:

```powershell
codex exec -p qwen38 --ephemeral --color never `
  -C C:\Users\Daniel\tombstead --json `
  "Use the exec Code Mode tool exactly once to run git status --short in the current workspace. Read the tool output. Then reply with exactly CODEX TOOL READY and nothing else."
```

The JSON event stream must contain exactly one successful command execution,
its complete `git status --short` output, and visible final agent text exactly
`CODEX TOOL READY`. The server log must show a first 7,680-token prefill chunk
with a positive pending-token count, completion of the remaining chunk, a
second Responses turn that consumes the custom-tool output, and a healthy
scheduler. Confirm the Tombstead status is unchanged and recheck `/health`
afterward. This gate qualifies the multi-chunk request and Codex's free-form
`custom_tool_call` ABI together.

Codex 0.151.0 currently also emits an optional model-catalog refresh warning
because SGLang's OpenAI-compatible `/v1/models` response uses the
`object`/`data` schema while that catalog reader expects a `models` field. It
may also report fallback model metadata for this custom model ID. The explicit
profile supplies the real context and compaction limits; a successful expected
banner, completed Code Mode round trip, and healthy listener establish the
end-to-end gate.

Keep Qwen Codex requests sequential: this lane admits one running request. Exit
a failed Qwen TUI before relaunching the server so its retained retry cannot
overlap a new turn. Use `Ctrl+C` in the foreground launcher terminal for an
intentional shutdown. For a detached or orphaned launch, first resolve the
listener PID, its complete ancestry, CUDA workers, and unrelated processes;
stop only the verified server tree leaf-first and confirm port 30000, compiler
workers, and GPU residency are clear afterward.

## Qualified production contract

The executable source of truth is
[`scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1`](scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1).
At the notebook cutoff, the accepted production configuration is:

- native Windows on the RTX 5090;
- checkpoint
  `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4`;
- endpoint `http://127.0.0.1:30000/v1`, served model `qwen3.8-27b`;
- real context and target/draft token pools of `200000`, one running request;
- language-only surface, Qwen3 reasoning parser, Qwen3 Coder tool parser;
- NEXTN linear rejection sampling, two speculative steps, three draft tokens,
  top-k-one chain, draft top-k one, and native CUDA direct one-hot proposal
  construction inside the single multi-step CUDA graph;
- FlashInfer prefill and sampling, TRT-LLM MHA/XQA target and draft decode,
  ReplaySSM linear speculation, FP8 E4M3 draft KV, checkpoint-selected target
  KV, 7680-token prefill chunks, and page size 64;
- FP32 Mamba state with four slots and `extra_buffer_lazy` caching;
- FP4 FlashInfer autotuning, including large ordinary EXTEND, with FP8 GEMM
  autotuning skipped;
- Cutlass NVFP4 prefill plus in-place Marlin decode for all 64 target gate/up
  projections; the 85 MiB relayout scratch is reused across layers;
- torch compile mode `default`, batch-one full decode graphs, scheduler receive
  interval 4, stream interval 4, incremental output, and a 128 MiB FlashInfer
  workspace;
- every SWOR topology/oracle switch, adaptive depth, fixed-acceptance
  simulation, and explicit online draft quantizer left inactive.

The accepted exact `199000+16` record is **3078.058 prompt / 114.617
generation tok/s**, **64.651152 s TTFT**, and **64.782022 s** end to end.
An independent no-override launcher restart reached **3052.437/114.053**,
**65.193816 s TTFT**, and **65.325334 s** end to end, beating every prior
record metric in the same exact request. The default relaunch captured target
verify, draft decode, and draft extend graphs, preserved reasoning/tools and
OpenCode2, and left 4,338 MiB free after cache flush. Treat this as the
production comparison baseline and remeasure when source, dependencies, GPU
environment, or launcher defaults move.

Base RadixArk with chunk 4096 remains an explicit control. Its earlier
chunk-7680 regression does not contradict the new default because the
launcher now selects the attention-selective checkpoint rather than applying
7680 globally to base RadixArk.

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

`benchmark/native` contains the portable CPU-only C++23 replacements for the
stream and acceptance clients. Their strict host suites pass under GCC C++23
and MSVC's C++23-capable `c++latest` mode. The matched Windows
Python/native/Python stream and acceptance gates pass, including exact
`199000+16`; Apple remains pending. Keep the Python commands above as the
cross-platform scoreboard authority through the Apple gate, and record the
client implementation with every sample. Build instructions and the promotion
contract are in
[`notes/benchmark-contract.md`](notes/benchmark-contract.md).

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
- Hybrid Marlin keeps target gate/up weights in Cutlass layout for prefill,
  relayouts them in place after the final prefill forward, and uses Marlin only
  for at most four tokens. Preserve the canonical bit-exact relayout,
  single-scratch reuse, final-chunk handoff, and explicit Cutlass fallback.
- Preserve upstream and non-Windows behavior behind narrow native-Windows
  dispatch gates. Keep experiments opt-in and launcher defaults production
  safe.
- Preserve asynchronous CUDA lifetimes. Stable graph inputs may be reused;
  per-cycle outputs that outlive a launch retain distinct storage.
- Native-Windows TP or attention-TP size one makes sampler token
  synchronization an identity. Preserve the process-group size guard before
  grammar/env-driven CUDA collectives; the installed Gloo control plane does
  not provide the CUDA all-reduce implementation.
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
