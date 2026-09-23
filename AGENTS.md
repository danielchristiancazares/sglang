# AGENTS.md

Do NOT add new Python code. Anything new must be in C++ or CUDA.

# Read First

This checkout carries native-Windows CUDA and Apple-silicon MPS/MLX
performance and serving lanes alongside upstream SGLang. These instructions
apply repository-wide.
[`docs/AGENTS.md`](docs/AGENTS.md) adds the documentation-site rules for work
under `docs/`.

Durable guidance reconciled through **2026-09-20** against
[`notes/experiment-log.md`](notes/experiment-log.md) and the executable source.
Use entry dates and the matching platform, checkpoint, and workload when
resolving a conclusion. Cross-machine appends can appear out of date order;
fresh runtime evidence establishes live state.

## Recover context before acting

Start with the smallest source set that answers the task:

| Need | Read |
|---|---|
| Resume work or establish the handoff | [`notes/current-state.md`](notes/current-state.md), then any later entry in [`notes/experiment-log.md`](notes/experiment-log.md) |
| Change or revisit a selected default | [`notes/decisions.md`](notes/decisions.md) and any later experiment-log entries |
| Run or compare a benchmark | [`notes/benchmark-contract.md`](notes/benchmark-contract.md) |
| Understand experiment history | [`notes/timeline.md`](notes/timeline.md) |
| Recover exact samples, logs, failures, and intermediate state | [`notes/experiment-log.md`](notes/experiment-log.md) |

Read this file for recurring working rules, the decision ledger for specific
selections and rejected candidates, and the experiment log for exact evidence.
Historical performance targets and recorded running processes are scoped to
their entries. Continue the user's current task.

At the beginning of each task:

- inspect the branch, `HEAD`, full worktree status, and relevant diff;
- treat every existing modified or untracked path as user-owned work;
- verify listeners, process ancestry, GPU ownership, installed dependencies,
  and logs when they matter; recorded PIDs and process state are snapshots;
- inspect the resolved launcher arguments and live endpoint before describing
  a server as current or healthy;
- trace the loaded implementation and executed dispatch before tuning a knob.
  Confirm the active checkpoint, binary, backend, and graph path; an available
  flag or passing isolated kernel does not establish serving reachability.

## Preserve the recovery record

Append to [`notes/experiment-log.md`](notes/experiment-log.md) after meaningful
code changes, launches, measurements, failures, promotions, and cleanup, and
follow the writing rules at the top of that file. Record the exact command or
resolved arguments, the code revision by commit subject and branch, individual
samples, environment, and result. Leave out process IDs, hashes, and handoff
lines; the open handoff belongs in `notes/current-state.md`. This is the
recovery ledger; keep enough detail for a fresh agent to continue after
compaction or a crashed client.

Maintain the compact layer when conclusions change:

- put a recurring failure-prevention rule in this file when its cause and
  remedy are established; state the action, its scope, and a link to evidence;
- update the existing authoritative rule when later evidence changes it;
  retain superseded measurements and incident detail in the experiment log;
- update `notes/current-state.md` for a new qualified winner or handoff;
- add durable selections and closed candidates to `notes/decisions.md`;
- change `notes/benchmark-contract.md` when the workload or gates change;
- add a timeline phase for a material new direction;
- leave incidents and sample-by-sample results in `notes/experiment-log.md`.

## Start Qwen3.8-27B for Codex

This is the local Codex setup for the native-Windows lane. The repository
launcher includes five FP32 Mamba slots for multi-chunk repository prompts
in its default DSpark-v2 profile. The derived target checkpoint
is a local artifact assembled from the immutable RadixArk base and NVFP4
donor; its `selective-nvfp4-manifest.json` is part of the checkpoint
provenance.

### One-time setup

From PowerShell 7 at the repository root, confirm that the local virtual
environment, target checkpoint, manifest, and DSpark-v2 draft are present:

```powershell
Test-Path .\.venv\Scripts\sglang.exe
Test-Path C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4
Test-Path C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4\selective-nvfp4-manifest.json
Test-Path C:\Users\Daniel\models\Qwen3.8-27B-DSpark-v2
```

All four commands must return `True`. The launcher initializes the qualified
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
pointing at its former location. Inspect `.venv/pyvenv.cfg` if the interpreter
itself cannot start; a missing base interpreter requires environment repair
before editable-install repair. Keep dependency versions fixed. For stale
editable metadata, use the working environment interpreter and rerun both
checks after:

```powershell
uv pip install --python C:\Users\Daniel\sglang\.venv\Scripts\python.exe `
  --editable C:\Users\Daniel\sglang\python `
  --no-deps `
  --reinstall-package sglang
```

The recorded DSpark-v2 Code Mode gate used Codex CLI 0.152.0. Requalify the
installed client version with the gate below. Create or verify the dedicated
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
.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1
```

This retains the selected checkpoint, reasoning and tool parsers, real 200K
context/token pools, one-request scheduler, CUDA graphs, and speculative-decode
settings. Preserve the fifth Mamba slot: unfinished-prefill cache donation
needs transient state beyond the four-slot startup floor. Both the earlier
Codex lane and DSpark-v2 exposed this boundary. Five slots passed multi-chunk
clients, retained-cache pressure, exact `199000+16`, and post-capacity reuse.
See [the Codex failure](notes/experiment-log.md#2026-08-30-1342-pdt---five-slot-codex-lane-fixes-multi-chunk-stream-failure) and
[the DSpark OpenCode repair](notes/experiment-log.md#2026-09-01-0318-pdt---fifth-mamba-slot-closes-the-opencode2-failure-boundary).

Leave the launcher terminal open. Startup loads the checkpoint, JIT-compiles
kernels, and captures the selected topology's CUDA graphs. Wait for the ready
message before sending traffic. The resolved `server_args` must include
`max_mamba_cache_size=5`, `max_running_requests=1`,
`max_total_tokens=200000`, `context_length=200000`,
`speculative_algorithm='DSPARK'`, and `chunked_prefill_size=4096`. Confirm
target verification and the folded draft graph complete. NEXTN controls have
a separate draft-extend phase. Use one bounded readiness check after the
expected startup interval:

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
`CODEX TOOL READY`. The server log must show multi-chunk prefill at the resolved
chunk size with positive pending tokens, completion of the remaining prompt, a
second Responses turn that consumes the custom-tool output, and a healthy
scheduler. Confirm the Tombstead status is unchanged and recheck `/health`
afterward. This gate qualifies the multi-chunk request and Codex's free-form
`custom_tool_call` ABI together.

The recorded Codex 0.151.0 integration also emitted a model-catalog refresh warning
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
Its default profile at this source revision is:

- native Windows on the RTX 5090, attention-selective RadixArk NVFP4 target,
  served model `qwen3.8-27b` at `http://127.0.0.1:30000/v1`;
- real `200000` context and target/draft token pools, one running request,
  five FP32 Mamba slots, and `extra_buffer_lazy` caching;
- trained DSpark-v2 online-FP8 draft, block size seven/eight verification
  rows, ordinary sampled verification, and static target-graph draft-KV commit;
- FlashInfer prefill and sampling, TRT-LLM MHA/XQA target decode, Triton draft
  attention and GDN, ReplaySSM, FP8 E4M3 target/draft KV, chunk 4096, page 64;
- Cutlass prefill and in-place Marlin for eligible tensor-input projections
  through eight tokens, with shared relayout scratch and in-place block scales,
  and the native final-prefill Marlin handoff deferred to the first verify;
- full batch-one decode graphs, prefill graphs disabled, torch compile
  disabled, FP4 autotuning enabled, FP8 GEMM autotuning skipped, and the
  additional large-EXTEND autotune pass inactive;
- scheduler receive interval four, stream interval four, incremental output,
  a 128 MiB FlashInfer workspace, language-only loading, Qwen3 reasoning,
  and Qwen3 Coder tool parsing;
- the checkpoint served from its own path with model sampling defaults, so
  the FlashInfer FP4 tuning cache key is stable across launches; the current
  cache is the seeded September 22 tactic file, and a deliberate tactic change
  replaces that file and requalifies (see the
  [September 23 tuning identity entry](notes/experiment-log.md#2026-09-23---stable-fp4-tuning-identity-in-the-powershell-launcher)).

The [September 1 DSpark qualification](notes/experiment-log.md#2026-09-01-0624-pdt---dspark-v2-150-toks-objective-fully-qualified-and-promoted) includes
exact capacity, behavior, OpenCode2, Codex, and independent sampled windows.
Its **155.961/162.500 tok/s** means used presence penalty **1.5**.
The [September 2 sampling audit](notes/experiment-log.md#2026-09-02-0206-pdt---official-qwen38-coding-setting-audit) establishes
presence **0.0** for thinking/coding; fresh qualification uses that profile.
Keep each historical result attached to its actual sampling settings.

NEXTN/chunk-7680 and base RadixArk are explicit controls. Chunk size, draft
precision, compile policy, and graph count belong to the measured topology.
Recheck them after a source or checkpoint change. The
[September 20 TTFT trial](notes/experiment-log.md#2026-09-20-native-windows-qwen38-27b-ttft-optimization) again found that a historical
chunk-7680 result did not transfer to the DSpark/five-slot composition.

Remeasure after source, dependencies, GPU environment, client, or launcher
defaults change. Historical objectives supply context for the current task;
GPU experiments follow the user's current authorization.

## Behavior and capacity are part of performance

Every promoted native-Windows Qwen candidate preserves all of these:

- exact 200,000 context and token-pool capacity, including a successful
  `199000+16` request when memory layout or residency changes;
- sampled reasoning at temperature `1.0`, top-p `0.95`, top-k `20`, min-p
  `0.0`, presence penalty `0.0`, and repetition penalty `1.0`;
- coherent preserved `reasoning_content`, ordinary completion behavior, and
  the established arithmetic answer `703` for `37 * 19`;
- exactly one parsed `multiply({"a":37,"b":19})` call with
  `finish_reason=tool_calls`, followed by a successful tool-result continuation;
- `/model_info` reporting image and audio understanding disabled;
- an unsimulated launcher-default production relaunch with all intended CUDA
  graph captures;
- standalone OpenCode2 and Codex tool round trips using their established
  provider and workload shapes.

Keep OpenCode2's ordinary cloud-model configuration stable during server
tuning. Use a process-scoped model alias or wrapper for local title/workload
experiments, and preserve thinking continuity through the parser boundary.
Apple and DiffusionGemma work uses the capacity, workload, and client gates
defined for its own lane in the benchmark contract.

## Measurement discipline

Use [`notes/benchmark-contract.md`](notes/benchmark-contract.md) as the full
contract. The standard Windows controls are:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repetition-penalty 1.0
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
  control on the same machine, checkpoint revision, request shape, sampling
  profile, client implementation, and cache treatment. Keep Mac Pro, M1 Max,
  Windows, GGUF, affine-quantized, and diffusion results separately identified.
  The [cross-machine attribution correction](notes/experiment-log.md#2026-08-23-0746-pdt---cross-machine-q4-record-deleted-m1-max-q2-authority-restored) is binding.
- Capture at least five consecutive samples, their mean, and each individual
  result. Production promotion uses a second independent real-sampling window.
- Label cold, warmed, cached-prefix, cache-flushed, simulated, unsimulated,
  and externally contended measurements.
- Preserve token counts, finish reason, output length, and deterministic
  digests for fixed-work controls in the machine-written receipts. Compare
  digests with a tool and record the verdict, not the digest, in the
  experiment log. Pair stochastic throughput with native
  acceptance counters and semantic output. Record the executed target and
  draft distributions; independently changing their filters changes acceptance.
- Use `6213+512` for the Windows generation comparison. Exact `199000+16`
  establishes capacity and prefill; its 15 post-first-token intervals are too
  short for a stable decode ranking. Inspect SSE fragment/coalescing and final
  metadata timing when attributing small client-observed changes.
- Evaluate useful coding work through natural completion, retained reasoning
  and final text, real tool continuation, and independent checks of the result.
  Bounded token replays and a successful tool transport gate have narrower
  scope. The [completed Rust repair](notes/experiment-log.md#2026-09-14--naturally-completed-served-xhigh-rust-repair) passed its oracle
  while remaining below the requested sustained throughput target.
- On Windows, record clocks, power, utilization, free VRAM, and competing WDDM clients.
  Chrome, ZCode, Epic Games Launcher, and other desktop activity have produced
  measured contention.
- On macOS, record unified-memory use, pressure, swap, thermal state, and
  competing indexing or file-provider work. Preserve contended samples with
  their labels and repeat clean controls before drawing a performance conclusion.
- Flush the server cache after long or memory-heavy requests before drawing a
  steady-state conclusion.
- Treat fixed accepted length as an execution-cost probe. Semantic and
  production evidence comes from ordinary rejection sampling.
- Treat source inspection, CPU tests, microbenchmarks, and a single favorable
  stochastic window as intermediate evidence. Promotion requires the full
  behavior, capacity, real-client, and production-relaunch gates.
- Profile the complete critical path, including host gaps, prefill, cache reuse,
  draft work, verification, and client turns. Judge speculation by emitted
  tokens per complete cycle. A faster isolated kernel can leave the cycle
  unchanged, as [PERF-029 demonstrated](notes/experiment-log.md#2026-08-20-2231-pdt---perf-029-removed-after-full-cycle-attribution).
- Include loaded weights, populated KV, recurrent snapshots, temporary
  allocations, graph pools, and first-use JIT in the memory budget. Test
  near-capacity execution and a subsequent real request; allocation success
  and advertised context alone establish only part of usable capacity.

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

### Shared numerical and state contracts

- Performance hot-path implementations use C++/CUDA. Existing Python is the
  thin binding, dispatch, configuration, test, and launch surface; the
  repository-wide no-new-Python rule remains in force.
- Preserve the executed operation's rounding, accumulation order, dtype, and
  exponential behavior. Qualify eager/compiled and BF16/FP16 paths separately.
  The [eager/compiled NVFP4 split](notes/experiment-log.md#2026-08-20-2131-pdt---perf-027-repaired-as-an-eager-only-exact-producer) and
  [FP16 sigmoid correction](notes/experiment-log.md#2026-09-14---qualified-opt-in-fp16-activation-execution-and-arithmetic-repair) show why a mathematically
  equivalent expression can change the model's numerical contract.
- Test real post-load layouts and captured activations, boundary values,
  mutated graph inputs, and repeated replay. An inverse round trip against
  a synthetic representation can miss a wrong serving representation.
- Preserve asynchronous GPU lifetimes. Stable graph inputs may be reused;
  per-cycle outputs that outlive a launch retain distinct storage. Check target,
  draft, recurrent, and active KV state as well as emitted IDs across restore,
  continuation, and speculative commit.

### Windows CUDA

- Native-Windows Gemma residual normalization writes the existing bit-exact
  JIT result directly into caller-owned `x`; do not reintroduce its former
  temporary allocation and copy.
- Hybrid Marlin keeps eligible tensor-input weights in Cutlass layout for
  prefill and switches weights and block scales in place for at most eight
  tokens. Preserve canonical relayout parity, shared scratch, final-chunk
  handoff, and Cutlass fallback. Exclude prequantized FP4 tuple consumers;
  their Cutlass dispatch requires Cutlass-packed bytes. Convert scales from
  the actual post-load block-swizzled representation. See the
  [tuple-layout failure](notes/experiment-log.md#2026-09-01-0129-pdt---tuple-safe-broad-hybrid-reaches-152260-first-window-but-149393-over-ten) and
  [post-load scale failure](notes/experiment-log.md#2026-09-01-0249-pdt---first-in-place-scale-full-model-gate-rejected-on-acceptance).
- The native lazy handoff is a DSpark launcher default via
  `-EnableLazyMarlinRelayout`, which sets
  `SGLANG_ENABLE_NVFP4_MARLIN_LAZY_RELAYOUT`; the descriptor itself stays
  default-off elsewhere. Use `-EnableLazyMarlinRelayout:$false` as the matched
  eager control. Preserve real stream ordering and graph addresses, and reject
  online weight updates while its fixed post-load descriptors are active;
  restart the runner instead. The measured benefit is short TTFT, not a
  demonstrated E2E or long-prefill gain. Qualification is scoped to batch-one,
  compile-disabled DSpark-v2. See
  [the native qualification](notes/experiment-log.md#2026-09-22---native-lazy-relayout-opt-in-qualification) and
  [the default promotion](notes/experiment-log.md#2026-09-23---lazy-marlin-handoff-promoted-to-launcher-default).
- Preserve upstream and non-Windows behavior behind narrow native-Windows
  dispatch gates. Keep experiments opt-in and launcher defaults production
  safe.
- Native-Windows TP or attention-TP size one makes sampler token
  synchronization an identity. Preserve the process-group size guard before
  grammar/env-driven CUDA collectives; the installed Gloo control plane does
  not provide the CUDA all-reduce implementation.
- The linear `SimulateAcceptedLength` control models contiguous linear
  ancestry. Tree recurrent-state qualification uses real tree ancestry and
  accepted-path commit tests.

### Provenance and validation

- Preserve the original FlashInfer checkout and the clean Windows 0.6.17 port
  as separate provenance lines.
- Leave the protected CUDA compatibility headers untouched. Their recorded
  SHA-256 is
  `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.
- Keep downloaded RadixArk and Gittensor source checkpoints immutable. Build
  any hybrid or converted artifact at a distinct path with provenance and
  checksums. Keep required models, environments, and recovery artifacts outside
  automatically purged caches. Verify base, donor, and draft revisions before
  reconstruction; [byte-identical artifact recovery](notes/experiment-log.md#2026-09-12-1154-pdt--attnnvfp4-production-artifact-recreated-byte-for-byte) depends
  on those identities.
- Treat `sglang.bundle` as unrelated user-owned material.
- Commit messages carry only the subject and body. Do not add "Generated
  with" or `Co-Authored-By` trailers unless the user explicitly approves them.

For every touched surface, run focused tests, Python compilation, PowerShell
parsing where applicable, and `git diff --check`. Native kernel changes also
need isolated CUDA parity, CUDA-graph replay coverage when captured, then one
controlled full-model gate.

## macOS and Apple-silicon lessons

- Profile actual tensor families and runtime shapes. Mixed GGUF/affine
  checkpoints contain several precisions and outlier projections; a filename's
  nominal bit width is insufficient. Keep batch-one decode, small speculative
  verification, and large prefill measurements distinct.
- Budget unified-memory residency for the full model, populated cache,
  snapshots, and transients. Reducing the allocator cache limit or prefill
  chunk cannot make an oversized resident KV design fit. Use bounded query
  tiles with the correct prefix-offset causal boundary; preserve the
  distinction between isolated 131K attention and full-model long-context
  qualification. See [the dense-SDPA capacity failure](notes/experiment-log.md#2026-09-01-2119-pdt---exact-id-32k-rejects-the-cache-capped-dense-sdpa-path) and
  [the bounded-prefill gate](notes/experiment-log.md#2026-09-14---bounded-sdpa-for-full-context-prefill).
- Guard inactive cache entries before dequantization or value loads. Zero
  probability multiplied by a NaN still contaminates the result. Poison unused
  tails in tests and include all active keys, values, and affine coefficients
  in long-cache state checks. See [the default Q8 tail repair](notes/experiment-log.md#2026-09-14---guard-inactive-tails-in-the-default-q8-attention-kernel)
  and [active-cache verification](notes/experiment-log.md#2026-09-14---enable-long-context-q8-state-verification).
- Standard MTP history pairs target hidden rows with the next prompt tokens.
  Keep its committed history one token behind the target; snapshot/restore
  speculative work and append only committed pairs, including target-only
  fallbacks. Check long prompts, divergent prefixes, and identical-request
  reuse with sampled IDs and both state digests. See
  [the committed-history repair](notes/experiment-log.md#2026-09-02-0335-pdt---a164-committed-history-implementation-and-actual-work-selection) and
  [exact repeated-prompt reuse](notes/experiment-log.md#2026-09-13---exact-repeated-mtp-prompt-reuse).
- Verify prefix reuse through the actual client's continuation: inspect cached
  tokens, newly prefetched tokens, and end-to-end completion. A successful
  cache hit can still leave decode too slow, as the
  [eight-state real-work probe](notes/experiment-log.md#2026-08-31-0224-pdt---eight-no-buffer-states-retain-the-real-work-prefix) demonstrated.
- Keep BF16 as the native engine's default and treat FP16 activation execution
  as an explicitly audited numerical profile. Preserve packed weight codes
  and FP32 recurrence parameters. Follow the shared arithmetic contract above;
  a short matching token sequence does not replace boundary and state parity.
- If native and stock-MLX controls both stall, investigate the Metal session
  before further kernel changes. Preserve unrelated processes and launchd-owned
  compiler services. The [session-wide failure](notes/experiment-log.md#2026-09-01-1314-pdt---native-and-stock-mlx-fallbacks-confirm-a-session-wide-metal-boundary) and
  [fresh-session recovery](notes/experiment-log.md#2026-09-01-1505-pdt---fresh-boot-recovers-metal-and-mixed-q5-reaches-19032-toks) establish this diagnostic.
- Use the separate [Mac DiffusionGemma setup](native/diffusion_gemma/mac/README.md)
  and its explicit canvas/sampler request profile. These controls affect both
  latency and text quality. Preserve the Qwen environment and keep the
  advertised context limit distinct from the completed capacity gates.

## Closed branches and retained experiments

Read [`notes/decisions.md`](notes/decisions.md) before reopening an older
candidate. Name the changed assumption and a matched test that can change the
earlier conclusion. A rejection applies to its measured checkpoint, topology,
and workload:

- NEXTN adaptive two/three-step depth, static three-step, and one-step MTP;
- reusable fused chain-metadata outputs with unsafe scheduling/lifetime cost;
- NEXTN draft proposal top-k 8 and its full online FP8, MXFP8, and dense
  online NVFP4 draft-weight throughput trials; the trained DSpark-v2 online-FP8
  selection has its own qualification;
- automatic promotion of a checkpoint from throughput alone; stock Gittensor
  and the closed DavidAU/hyssra evaluation retain their recorded quality and
  integration limits;
- target NVFP4 KV, which failed the September 12 arithmetic retest, and
  the measured SM120 TurboQuant35 codec, which lost quality/latency admission;
  see [the KV semantic gate](notes/experiment-log.md#2026-09-12-1257-pdt--current-source-target-nvfp4-kv-repeats-the-semantic-failure) and
  [the codec gate](notes/experiment-log.md#2026-09-12-1307-pdt--native-sm120-turboquant35-admission-fails-quality-and-latency-gates);
- 232K pools, which passed exact capacity yet fell to 98 MiB free before
  cache flush;
- the current target-only and SWOR tree proposal distributions, whose measured
  cost/yield remained below the qualified linear path.

The exact GPU tree verifier, low-rank GDN accepted-path commit, sparse SWOR
verifier, path/overlap oracles, topology analyzers, and their tests are retained
as opt-in experimental infrastructure. Preserve them. Reopening the tree route
requires measured proposal-overlap or draft-cost evidence that changes its
economics; topology rearrangement with the recorded q distribution has already
been exhausted. Qualify multi-cycle, non-front accepted-path KV and recurrent
state parity on current source before assigning any production standing.
