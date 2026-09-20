# Benchmark and acceptance contract

Use this contract for every comparison with the qualified native-Windows
Qwen3.8 production line. A comparable result records the request shape, server
mode, cache treatment, sampling profile, graph state, GPU environment, and
resolved launcher arguments.

**Native-Windows reconciled through:** 2026-09-01 06:18 PDT.

**Apple M1 Max Q2 addendum reconciled through:** 2026-08-30 20:46 PDT.

**Apple M1 Max affine-Q5 addendum reconciled through:** 2026-09-01 14:33 PDT.

## Primary performance scoreboard

The user-selected headline workload is exact uncached `6213+512` under ordinary
Qwen sampling: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5,
thinking enabled, and one admitted request. The current argument-free
DSpark-v2 record is **162.500 tok/s** over five consecutive samples
`[151.139,165.951,151.000,191.357,153.054]`. An independent exact-200K
explicit-switch window averaged **155.961 tok/s**. Both exceed the requested
150 tok/s gate; every sample in the argument-free window individually clears
it.

Exact `199000+16` remains the mandatory capacity and near-limit prefill gate,
not the generation headline. Its 16-token response leaves only 15 measured
post-first-token intervals. The current launcher-default DSpark-v2 result is
exact `199016`, **3118.215 prompt tok/s**, **63.818572 s TTFT**, and
**64.236571 s E2E**. Report the short decode rate, but do not use it in place of
the 512-token ordinary-sampling window.

Production qualification still requires the behavior, two real-sampling
windows, exact capacity, independent launcher relaunch, native acceptance, and
client gates below. Candidate ranking requires repeated matched controls; a
single favorable stochastic or exact-16 hit is supporting evidence only. The
compact scoreboard is [`../BENCHMARK.md`](../BENCHMARK.md). No next
native-Windows target is active after qualification of 150 tok/s.

The record profile is the Windows launcher default: selective target
checkpoint, trained DSpark-v2 online-FP8 draft, gamma seven/eight verify rows,
chunk 4096, five FP32 Mamba slots, FP8 target/draft KV, Triton draft attention,
static target-graph draft-KV commit, selected target GEMM tactics, and in-place
Cutlass-prefill/Marlin-decode gate/up weights. The NEXTN/chunk-7680 and base
RadixArk routes remain explicit controls.

## Qualified reference

| Gate | Current reference |
|---|---|
| Argument-free real sampled `6213/512` | **162.500 tok/s** five-run mean; every sample >=150 |
| Independent full-pool real sampled `6213/512` | **155.961 tok/s** five-run mean |
| Native DSpark acceptance | **2.737968** accepted length, **0.249045** rate, 326/1309 correct/proposed over 187 verifies |
| Exact capacity `199000+16` | `199016` total; **3118.215 prompt tok/s**, **63.818572 s TTFT**, **64.236571 s E2E** |
| Independent exact capacity | `199016` total; **3118.323 prompt tok/s**, **63.816352 s TTFT**, **64.233185 s E2E** |
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
| Primary production scoreboard | `6213/512`, normal rejection sampling | Measure production generation throughput over repeated clean windows |
| Recorded Windows scoreboard profile | Temperature `1.0`, top-p `0.95`, top-k `20`, presence `1.5` | Reproduce the retained Windows qualification windows under their measured contract |
| Sampled thinking profile | Temperature `1.0`, top-p `0.95`, top-k `20`, min-p `0.0`, presence `0.0`, repetition `1.0` | Match Qwen3.8's official thinking/coding recommendation |
| Native acceptance | The qualified Python probe, with the C++23 candidate matched against it under sampled production settings | Pair TPS with emitted/accepted length, proposal counts, histograms, and verify cycles |
| Long ladder | `32768/16`, `32768/512`, `65536/16` | Catch prefill, residency, repeated-request, and long-decode regressions |
| Exact capacity and near-limit prefill gate | `199000/16` | Prove exact total `199016` inside the selected 200K pools and track prefill/TTFT; short decode is telemetry |
| Real client | Standalone OpenCode2 with fixed provider/workload | Final reasoning, tool continuity, queue, parser, and wall-time integration |

The streaming and acceptance clients synthesize their request from the exact
inline prompt unit
`Inspect this local program carefully, preserve its behavior, and identify the next useful correctness or performance change. `
and the filler unit ` x`, then calibrate that text through the live server
tokenizer. The `6213` label names the resulting exact token count. The separate
[`../benchmark/windows/qwen38_local_prompt.json`](../benchmark/windows/qwen38_local_prompt.json)
fixture belongs to the OpenCode-shaped integration workload and carries its own
provenance.

## Standard commands

Exact 200K capacity and near-limit prefill gate:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py `
  --base-url http://127.0.0.1:30000 `
  --model qwen3.8-27b `
  --backend sglang `
  --input-tokens 199000 `
  --output-tokens 16 `
  --warmup-output-tokens 16 `
  --warmup-runs 1 `
  --timeout 600 `
  --temperature 0
```

Greedy current-shape control:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py `
  --base-url http://127.0.0.1:30000 `
  --model qwen3.8-27b `
  --backend sglang `
  --input-tokens 6213 `
  --output-tokens 512 `
  --warmup-output-tokens 16 `
  --warmup-runs 1 `
  --timeout 600 `
  --temperature 0
```

Official thinking/coding control:

The retained Windows production scoreboard above used presence penalty `1.5`.
Use that value to reproduce those historical windows; the official coding
profile below uses `0.0`. Keep the profiles distinct when comparing results.

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py `
  --base-url http://127.0.0.1:30000 `
  --model qwen3.8-27b `
  --backend sglang `
  --input-tokens 6213 `
  --output-tokens 512 `
  --warmup-output-tokens 16 `
  --warmup-runs 1 `
  --timeout 600 `
  --temperature 1.0 `
  --top-p 0.95 `
  --top-k 20 `
  --min-p 0.0 `
  --presence-penalty 0.0 `
  --repetition-penalty 1.0
```

Reasoning-disabled sampled control:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py `
  --base-url http://127.0.0.1:30000 `
  --model qwen3.8-27b `
  --backend sglang `
  --input-tokens 6213 `
  --output-tokens 512 `
  --warmup-output-tokens 16 `
  --warmup-runs 1 `
  --timeout 600 `
  --temperature 0.7 `
  --top-p 0.80 `
  --top-k 20 `
  --min-p 0.0 `
  --presence-penalty 1.5 `
  --repetition-penalty 1.0 `
  --disable-thinking
```

The 2026-08-16 exploratory result for that mode was **129.722 tok/s** over ten
runs, with **2.452943** mean accepted length. It is a separate behavior profile
from the reasoning-preserved production contract.

Native speculative acceptance counters:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_spec_acceptance.py `
  --base-url http://127.0.0.1:30000 `
  --model qwen3.8-27b `
  --input-tokens 6213 `
  --output-tokens 512 `
  --warmup-output-tokens 16 `
  --timeout 600 `
  --temperature 1.0 `
  --top-p 0.95 `
  --top-k 20 `
  --presence-penalty 0.0
```

For a non-thinking acceptance probe, also pass `--temperature 0.7 --top-p
0.80 --top-k 20 --presence-penalty 1.5 --disable-thinking`. Qwen's official
model card assigns presence `1.5` to that non-thinking profile, not to thinking
mode. Measurements recorded before 2026-09-02 with thinking enabled and
presence `1.5` remain historical evidence under their stated contract and need
a fresh official-profile window before they can qualify a new default.

## Native C++23 client candidate

The portable, framework-free client implementation lives under
[`../benchmark/native`](../benchmark/native). It provides two CPU-only C++23
executables:

- `bench_openai_stream.cpp`, covering the OpenAI-compatible streaming,
  calibration, warmup, cache, timing, token, fragment, and digest contract;
- `bench_spec_acceptance.cpp`, covering exact raw-token `/generate` requests
  and fail-closed speculative-counter validation.

The client process owns host orchestration while the serving process owns GPU
execution. This keeps the load generator free of a second CUDA context and
preserves both native Windows and Apple use. The public headers enforce C++23
through `config.hpp`.

On Windows, enter the existing native toolchain environment and compile either
entry point with the shared source set. MSVC's `c++latest` mode is the current
C++23-capable switch on this toolchain; the source-level guard verifies the
resolved language version:

```powershell
. .\scripts\windows\initialize_cuda_build_env.ps1 -MaxJobs 2
cl.exe /nologo /std:c++latest /O2 /EHsc /W4 /WX /permissive- `
  /Zc:__cplusplus /I benchmark\native\include `
  benchmark\native\src\arguments.cpp `
  benchmark\native\src\http_client.cpp `
  benchmark\native\src\json.cpp `
  benchmark\native\src\openai_benchmark.cpp `
  benchmark\native\src\sha256.cpp `
  benchmark\native\src\sse_parser.cpp `
  benchmark\native\bench_openai_stream.cpp `
  /Fe:sglang_bench_openai_stream.exe /link ws2_32.lib
```

Replace the final source and output name with
`bench_spec_acceptance.cpp` and `sglang_bench_spec_acceptance.exe` for the
acceptance probe. Apple Clang uses the same sources with `-std=c++23` and omits
the Windows socket library:

```bash
c++ -std=c++23 -O2 -Wall -Wextra -Wpedantic -Werror \
  -I benchmark/native/include \
  benchmark/native/src/arguments.cpp \
  benchmark/native/src/http_client.cpp \
  benchmark/native/src/json.cpp \
  benchmark/native/src/openai_benchmark.cpp \
  benchmark/native/src/sha256.cpp \
  benchmark/native/src/sse_parser.cpp \
  benchmark/native/bench_openai_stream.cpp \
  -o sglang_bench_openai_stream
```

Use complete explicit arguments for each live pair. The native stream keeps the
legacy 128-token CLI default. The qualified controls set their 512- or 16-token
completion shape explicitly. These examples use the repository-root `/Fe`
outputs from the documented build; an isolated build substitutes its recorded
absolute executable path:

```powershell
.\sglang_bench_openai_stream.exe `
  --base-url http://127.0.0.1:30000 `
  --model qwen3.8-27b `
  --backend sglang `
  --input-tokens 6213 `
  --output-tokens 512 `
  --warmup-output-tokens 16 `
  --warmup-runs 1 `
  --timeout 600 `
  --temperature 0

.\sglang_bench_openai_stream.exe `
  --base-url http://127.0.0.1:30000 `
  --model qwen3.8-27b `
  --backend sglang `
  --input-tokens 6213 `
  --output-tokens 512 `
  --warmup-output-tokens 16 `
  --warmup-runs 1 `
  --timeout 600 `
  --temperature 1.0 `
  --top-p 0.95 `
  --top-k 20 `
  --presence-penalty 0.0

.\sglang_bench_openai_stream.exe `
  --base-url http://127.0.0.1:30000 `
  --model qwen3.8-27b `
  --backend sglang `
  --input-tokens 199000 `
  --output-tokens 16 `
  --warmup-output-tokens 16 `
  --warmup-runs 1 `
  --timeout 600 `
  --temperature 0

.\sglang_bench_spec_acceptance.exe `
  --base-url http://127.0.0.1:30000 `
  --model qwen3.8-27b `
  --input-tokens 6213 `
  --output-tokens 512 `
  --warmup-output-tokens 16 `
  --timeout 600 `
  --temperature 1.0 `
  --top-p 0.95 `
  --top-k 20 `
  --presence-penalty 0.0
```

Add `--disable-thinking` to both members of a paired profile when qualifying
the non-thinking route. Carry every explicit Python override into its native
partner.

The checked-in Python clients remain the measurement authority while the native
candidate completes matched live qualification on Windows and Apple. Every
pair resolves the same model, token-count, warmup, timeout, sampling,
seed-when-supported, stream backend, and thinking values. Stream promotion
requires matching calibrated counts, decoded request semantics, cache/warmup
order, exact usage and finish reason, and output field names and types.
Deterministic stream controls also require matching combined and per-channel
digests, character counts, and fragment counts; sampled stream profiles retain
those values as independent per-sample evidence.

Acceptance promotion requires the same 11 field names and compatible JSON
types, exact equal prompt/completion counts and thinking mode, finite latency,
valid counter ranges, and externally recomputed rate, accepted-length, cycle-
sum, and weighted-histogram algebra for every Python and native result. The
native companion additionally establishes exact calibration, warmup counts,
measured counts, and length finish through its fail-closed response validator.
Record each unseeded digest, latency, counter, and histogram independently, and
compare adjacent Python/native/Python windows for stable client-neutral timing
and acceptance distributions. Semantic behavior remains governed by the
sampled stream, reasoning, arithmetic, and tool gates. Record the client
implementation with every result.

**Windows live status (2026-08-30 17:55 PDT): passed.** One argument-free
four-slot production launch completed adjacent Python/native/Python windows
for deterministic and sampled `6213+512`, exact `199000+16`, and sampled
acceptance. Every stream result preserved the 38-field schema, exact usage and
length finish; deterministic windows also matched all hashes and
fragment/character counts. Every acceptance result preserved the 11-field
schema and exact counts, and all counter/histogram algebra passed. Native
timing stayed within or favorably adjacent to the Python observations. Apple
Clang/M1 Max live parity remains the final client-promotion gate, so Python
retains cross-platform scoreboard authority.

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
scoreboard. Its repository-native route uses Python ingress, Qwen's official
tokenizer, a 32,768-token BF16 pool, one request, and 1,024-token prefill
chunks. Signed PERF-A016 commit
`52b5326d8e5140b72a26a3909316fb1f665bbd3d` remains the first
named-client-qualified result. Signed PERF-A021 commit
`4dfa1ad3efdfe3f9236aa0ed0c841644ab513859` is the current performance,
capacity, and named-client baseline. It specializes the Q2_K tensor family
inside the same mixed-format IQ2_XXS/Q2 checkpoint; checkpoint and scoreboard
standing remain Q2.

The complete Apple baseline has three exact workloads:

1. Fixed decode uses `/generate` with the 12-token identifier prompt, 256
   forced greedy output tokens, ignored EOS, and five consecutive requests.
   Report aggregate generation as `1280 / sum(wall time)`, best request rate,
   mean E2E, and best E2E.
2. Reasoning-enabled OpenAI streaming uses exact `128+256` because the empty
   reasoning-enabled chat template occupies 52 tokens. Run one 16-token
   warmup, retain any stabilization measurement, then record five consecutive
   cache-flushed requests. Report prompt throughput as
   `640 / sum(TTFT)`, generation as 1,275 post-first-token intervals divided
   by their total time, mean/best TTFT, and mean/best E2E. Preserve exact
   token counts, length finish, nonempty-delta count, and output/reasoning
   digests.
3. Capacity uses one cache-flushed, reasoning-enabled exact `32761+1` stream
   inside the allocated 32,768-token pool. Report exact usage and finish,
   observed prompt throughput, TTFT, E2E, chunk/tail progress, digest,
   process/runtime state, and post-run cache flush. A one-token completion has
   zero post-first-token intervals, so generation throughput comes from the
   fixed and 128+256 workloads.

Prompt, Generation, TTFT, E2E, and Capacity must all appear together in root
`BENCHMARK.md` for the same source checkpoint. Any workload change receives a
fresh label and retains the preceding baseline.

The Windows real-client gate is Codex CLI 0.151.0 with the dedicated `qwen38`
profile over `/v1/responses`. Pin and record these overlay identities:

```text
FA1880D110D966D75423EF7C524CBCA21BCE38330A5C39A37562762DFDCA6510  C:\Users\Daniel\.codex\qwen38.config.toml
46FD2235681AD29571852F060BDC022A2E3ABE5E996A173F570C09EC861C9C87  C:\Users\Daniel\.codex\qwen38_models_cache.json
```

The fixed read-only sequential Code Mode gate is:

```powershell
C:\Users\Daniel\AppData\Roaming\npm\codex.cmd exec `
  -p qwen38 --ephemeral --color never `
  -C C:\Users\Daniel\tombstead --json `
  "Use the exec Code Mode tool exactly once to run git status --short in the current workspace. Read the tool output. Then reply with exactly CODEX TOOL READY and nothing else."
```

A passing window records the client/version and overlay hashes, exact five-slot
server ingress and resolved arguments, one successful command item and its
complete output, the consumed `custom_tool_call_output` turn, exact visible
final marker, usage, zero client exit, unchanged Tombstead status, multi-chunk
prefill/pending counts, post-request health, cache flush, verified foreground
cleanup, free listener, and returned GPU state. Session
`01a05519-7206-7d32-bd7c-1e573466e51f` qualified this contract with **25,010
input**, **12,544 cached input**, **201 output**, and **142 reasoning-output
tokens**.

The governing Apple workspace-write client gate is Codex CLI 0.151.0 with the
dedicated machine-local home
`/Users/dcazares/.codex/qwen38-local-hardened-home` over `/v1/responses`.
Record these three bundle identities before and after every qualification:

```text
9d7842bb47d15c5b7a63d1507b8e035784bf1ab768de36dbb131088493620409  $CODEX_HOME/config.toml
862339c156824879852dbdc9ebf096523d6312699fdd8723f13d81091de2ec71  $CODEX_HOME/models.json
5d59350d7a1568c3c458b05513e8b58ed50d70874f27fb80a06d131e09b9d096  $CODEX_HOME/instructions.md
```

The selected config uses sibling-relative artifact paths, default/exec low
reasoning, medium Plan-mode reasoning, 10,000-token tool-output truncation, a
2,400,000-ms idle timeout, zero request and
stream retries, no shell login startup, core-only secret-filtered initial shell
inheritance, disabled message-history append, analytics, feedback, and optional
tool/model-visible surfaces, an explicit trusted decision for the repository,
an untrusted decision for the fixed gate path, and workspace-write confinement
with network and implicit temporary roots disabled. Its configured
30,000-token Total-scope compaction limit resolves to 29,491. No system or
managed Codex config was present in the qualified window. The ordinary user
config and rules remained outside this home and byte-stable at
`97f15d75...5ca9c` / `63d2d91f...61e5` before and after the gate.

The shell policy sets `ZDOTDIR=/var/empty` after core inheritance and before
spawning unified-exec children. The qualified root-owned 0755 directory was
empty, and `/etc/zshenv` plus `/etc/zsh/zshenv` were absent. This prevents
spawned `zsh -c` tools from rereading mutable `~/.zshenv`; values already
present in the parent Codex environment remain subject to core filtering.

The earlier `qwen38-local*` profile pairs remain historical artifacts. The
19:46 profile-overlay write was behaviorally successful, while its lower user
config was not pinned at process start. Preserve its files and recorded hashes
without using it as the governing reproducibility identity.

After the PERF-A021 endpoint is ready, launch the ordinary interactive client
through the same selected home:

```bash
env CODEX_HOME=/Users/dcazares/.codex/qwen38-local-hardened-home \
  SGLANG_API_KEY=local /opt/homebrew/bin/codex --strict-config \
  -C /Users/dcazares/sglang
```

Every future non-interactive Qwen/Codex work or qualification attempt has a
120-second wall-clock gate around the complete client process tree:

```bash
env CODEX_HOME=/Users/dcazares/.codex/qwen38-local-hardened-home \
  SGLANG_API_KEY=local \
  /opt/homebrew/bin/timeout --signal=INT --kill-after=10s 120s \
  /opt/homebrew/bin/codex exec ... </dev/null
```

Leave GNU `timeout` in its process-group mode; `--foreground` exempts command
children from the timeout. Exit `124` or forced-cleanup exit `137` fails the
gate. After either result, record the elapsed time and exact thread, verify the
client and compiler descendants are absent, and recheck every protected input
and verifier hash. The configured 2,400,000-ms Codex idle timeout remains a
transport setting; this outer 120-second bound is the actual-work usability
contract.

The repository is trusted so root `AGENTS.md` reaches the interactive prompt.
At qualification, an ignore-independent scan found no project `.codex/**`,
`hooks.json`, `*.rules`, `.agents/skills/**`, or `AGENTS.override.md` path.
The dedicated home contained no `rules/`, `skills/`, `AGENTS.md`, or
`AGENTS.override.md`; `$HOME/.agents/skills` and `/etc/codex/skills` were also
absent. Future project config, hooks, policy, skills, override instructions, or
home sidecars can alter the interactive path without moving the three bundle
hashes, so record their continued absence or identities before launch. The
ordinary interactive TUI can persist thread rollout/state inside the dedicated
home; `[history] persistence="none"` specifically disables global message-
history append.

The fixed default workspace-write gate creates a fresh empty
`/private/tmp/qwen38-codex-isolated-gate`, records all pre-state, then executes
without a profile, `-c`, `--sandbox`, reasoning, or capacity override:

```bash
test ! -e /private/tmp/qwen38-codex-isolated-gate &&
mkdir -m 700 /private/tmp/qwen38-codex-isolated-gate &&
env CODEX_HOME=/Users/dcazares/.codex/qwen38-local-hardened-home \
  SGLANG_API_KEY=local /opt/homebrew/bin/codex exec \
  --strict-config --ephemeral --ignore-rules --skip-git-repo-check \
  --color never -C /private/tmp/qwen38-codex-isolated-gate --json \
  'Use exec_command exactly once. Set its cmd to this exact script:
apply_patch <<'"'"'PATCH'"'"'
*** Begin Patch
*** Add File: gate.txt
+QWEN38_ISOLATED_WRITE_GATE=passed
*** End Patch
PATCH
After the tool succeeds, reply exactly QWEN38 ISOLATED WRITE READY' </dev/null
```

The fixed path is explicitly untrusted, and `--ignore-rules` excludes policy
for this scratch exec. Repository `AGENTS.md` visibility is established by the
separate repository-CWD prompt-input diagnostic.

A passing window records the Codex version, bundle and ordinary-global hashes,
system/managed-layer state, strict-load result, exact server ingress and
arguments, process-scoped API key, complete JSONL or an immutable raw-log
path/hash, request usage, exactly one successful `file_change`, exact scratch
contents/size/hash, exact visible final marker, zero client exit, unchanged
repository worktree, post-request health, cache flush, scratch removal,
foreground shutdown or verified leaf-first cleanup, free listener, and returned
memory/thermal state. The 20:46 exact-bundle run added only the 34-byte
`gate.txt`, matched SHA-256
`f7ca43b4d2b9698e2f794c8bfffefe78836423c77bde38a7405f96ab12f6729a`,
returned `QWEN38 ISOLATED WRITE READY`, and used 2,670 input / 115 output / 39
reasoning-output tokens. Its catalog declares `shell_type=unified_exec`; the
exposed tool names are `exec_command` and `write_stdin`. All three bundle
hashes and both ordinary-global hashes were unchanged afterward.

Forced compaction uses process overrides
`model_auto_compact_token_limit=1000` and Total scope. The retained historical
recovery run belongs to the immediate medium-reasoning predecessor hashes
`39ad0f7c97ed30d36d41baf5d2b6ec3c127e2e44baf9aad2aa76f6bbf70c832b` /
`8fbb54a5407b9279c1abcc61a805bb15067fe27bf4bf6e8346bd80051572dfa5` /
`8a2fe9b979da48d5bc5a38ec22fb44f50b06de21216e3643b376ba330a4e2279`.
It recovered from one malformed patch, completed a retry across two observed
compaction boundaries, preserved its nonce, and exited zero. Its raw JSONL and
exact warning text were not retained, so a future exact-transcript
qualification records both warnings and every inference boundary. The
isolated low bundle owns the clean edit gate; production-threshold near-limit
compaction and concurrent tools remain separate qualification gates. Windows
production continues to use the standalone OpenCode2 provider/workload
contract above.

Every Apple request expected to run longer than five minutes remains under an
active controller-side watchdog. Poll at intervals of at most 60 seconds and
record the last completed prefill/decode progress, exact process ancestry and
listener, free-memory percentage, swap use, page throttling, and thermal or
performance warnings. Declare the per-forward deadline before launch from the
preceding rung; it may not exceed eight minutes without a separately recorded
calibration. A missed deadline, lost process/listener, throttled pages, or a
thermal/performance warning triggers verified leaf-first cleanup. The client
timeout is only the terminal request bound. It never substitutes for polling.

## Apple M1 Max affine-Q5 screening addendum

The active Apple Q5 objective is sampled Qwen3.8-27B generation at least
**20 tok/s**, a real **131,072-token** context/pool, Codex `xhigh` reasoning,
and preserved arithmetic, tool-call, reasoning-content, and workspace behavior.
Kernel microbenchmarks rank implementation candidates; only full-model and
served-client gates establish objective progress.

`benchmark/mac/bench_qwen38_affine_q5_qmv.cpp` is the shared direct-kernel
screen for native affine-Q5/G64 batch-one decode. Build one executable from
each candidate's `qwen38_engine.cpp`, then invoke it as:

```text
bench_qwen38_affine_q5_qmv K N WARMUP ITERATIONS
```

Use at least eight warm launches and fifty timed launches for each of these
production shapes: gate/up `(5120, 17408)`, down `(17408, 5120)`, attention
output `(6144, 5120)`, and value projection `(5120, 1024)`. Each launch is
evaluated and synchronized, so the measured mean is dependent decode latency.
Record every raw mean, effective streamed GB/s, first BF16 result, complete
BF16 FNV-1a digest, candidate source identity, binary hash, boot, thermals,
memory, process state, and ordering. Run at least five randomized candidate
orders, then five reversed orders per shape.

Standalone Metal parity runs before timing. Candidates that preserve FP32
accumulation order must match the selected control's complete output digest.
A candidate that deliberately regroups FP32 arithmetic requires bounded
numeric parity plus deterministic full-target digest, sampled semantics, and
served behavior. Rank only repeatable matched windows; preserve the uniform-Q5
control and follow the kernel screen with full uniform-Q5 and mixed-Q5 target
measurements. The final promotion still requires the real 131K pool, sampled
20 tok/s floor, arithmetic `703`, exactly one `multiply({"a":37,"b":19})`
tool call, reasoning continuity, language-only metadata, restart-default
launch, and Codex 0.151.0 `xhigh` integration.

## Mac DiffusionGemma interactive workload

This separate workload targets the M1 Max with 32 GiB unified memory using
MLX-VLM and the pinned MLX-community 4-bit DiffusionGemma checkpoint. It leaves
the Qwen comparison and capacity contracts unchanged. Keep its model,
environment, and results under `~/.local/share/sglang-diffusiongemma` because
this host periodically removes `~/.cache`.

The C++23 client `benchmark/mac/bench_diffusiongemma.cpp` measures from request
submission through the SSE terminator. TTFT ends at the first nonempty visible
content delta. Primary throughput is reported output tokens divided by full
request wall time, including the initial wait. Diffusion emits whole blocks;
chunk spacing and the server's post-first-block rate are supplementary metrics.
Output-token counts include generated special tokens according to server usage.

Use `benchmark/mac/prompts/diffusiongemma_explanation.txt` with a 512-token
budget for the throughput comparison and `diffusiongemma_arithmetic.txt` with
a 32-token budget for short-response correctness. The fixed-work comparison
uses temperature zero, seed 42, thinking disabled, natural EOS, and explicit
sampler and canvas size. Record actual prompt/output counts, finish reason,
complete output, peak memory, TTFT, and end-to-end time for every sample. A
budget-limited explanation qualifies timing only; check completed answers
separately. Changing canvas size or sampler can change the output and quality.

Separate cold first-request and first-shape samples from warmed five-request
windows. Select a setting using both latency and the 20 output tok/s objective,
then repeat a five-request window after an independent server restart. Also
run `sampled` mode: temperature one retains the checkpoint's 0.4–0.8
denoising schedule; per-sample seeds 42, 43, ... exercise varied outputs.
Keep
requests sequential, retain source checkpoint bytes, check swap/thermals and
GPU ownership, and verify coherent arithmetic, completed explanatory output,
HTTP/SSE accounting, and a longer prompt before recording the handoff. The
checkpoint's advertised context size remains unqualified unless measured.

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
