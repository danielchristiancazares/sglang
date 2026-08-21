# Condensed experiment timeline

This is an orientation map, not the evidence ledger. Read the matching region
of [`experiment-log.md`](experiment-log.md) when exact samples, commands, logs,
code changes, or process state matter.

## 2026-08-15

### 17:14–18:59 — recover and tune the native GGUF path

- Recovered the dirty native-Windows SGLang worktree and locked an exact local
  OpenCode-shaped workload.
- Fixed GGUF tokenizer `model_max_length` serialization and built the local SSE
  benchmark.
- Established the first `6213/128` SGLang baseline at `446.583` prompt and
  `38.673` decode tok/s.
- Selected 4096 prefill chunks, a smaller active pool, FP32 Mamba state, and
  Triton KV split 16. OpenCode title handling moved into a process-scoped
  wrapper.
- Identified movie/Chrome GPU use as external contention, preventing a false
  concurrency regression conclusion.

### 19:25–20:33 — enable bundled MTP and close the GGUF ceiling

- Repaired native-Windows optional-import, loader, and rejection-sampling gaps
  for the bundled Qwen MTP head.
- The trained MTP path worked, while one- and three-step forms initially lost
  to the selected non-MTP GGUF server.
- Incremental streaming in four-token intervals produced a small retained
  synthetic improvement.
- The investigation moved to native NVFP4 weights for a larger gain.

### 20:35–22:38 — bring up the first NVFP4 server

- Installed the Windows FlashInfer `0.6.11.post3` path and repaired SM120
  NVFP4, compressed-tensors, Windows JIT, and mixed FP8-linear boundaries.
- The first exact NVFP4 control reached `6072.305` prompt and `44.918` decode
  tok/s, versus GGUF `473.380/42.035`.
- A fused Triton FP8 epilogue and its single-row specialization improved the
  short path.
- FlashInfer attention was synthetically faster, then failed a real long-prompt
  coherence A/B. Triton stayed selected until a supported FlashInfer port.

### 23:00–00:47 — select RadixArk, MTP, scheduler, and compile mode

- The RadixArk checkpoint repaired reasoning quality and enabled a useful
  trained three-step MTP path.
- Fixed-work controls separated speculative execution cost from acceptance
  variance.
- Scheduler receive interval 4 and torch compile mode `default` won controlled
  comparisons. Fully compiling previously failing Triton kernels was correct
  but slower.
- The target expanded to a real 200,000-token contract with reasoning, tools,
  preserved thinking, and vision disabled.

## 2026-08-16

### 00:49–01:19 — make the 200K contract real

- Found that late MTP embedding/head sharing caused silent pool downsizing to
  about 66K tokens.
- Added an idempotent early-sharing hook before pool allocation. Target and
  draft pools then allocated all 200,000 tokens.
- Triton long prefill became the next wall. FlashInfer prefill raised 32K
  prompt throughput by an order of magnitude and completed exact `199016`.

### 01:23–03:19 — port FlashInfer 0.6.17 cleanly to Windows

- Built a separate clean 0.6.17 port with Windows JIT support, compact hashed
  artifact paths, PE/COFF tile coverage, and SM120 kernels.
- Reinstalled and live-qualified the port without altering the protected CUDA
  header boundary.
- Chunk 8192 and tight-residency variants lost; chunk 4096 remained selected.

### 03:33–04:50 — XQA, workspace, and autotune selection

- Native target NVFP4 KV recovered memory but corrupted reasoning; ordinary
  checkpoint-selected target KV with XQA was correct.
- FP4-only FlashInfer autotuning won decode. FP8 autotuning lost.
- A 128 MiB workspace was the first configuration to win both decode and long
  prefill; 64 MiB failed graph allocation.
- The result passed 32K, 64K, and exact 199K qualification plus OpenCode,
  reasoning, tools, and vision-disabled checks.

### 05:03–06:08 — move the draft to XQA

- Repaired the last Windows source collision for draft XQA and qualified the
  target/draft split.
- Controlled throughput reached `133.232 tok/s`; real recovery reached
  `96.110 tok/s`; exact `199016` remained valid.
- FlashInfer 0.6.17 source/install identity, JIT tests, focused SGLang tests,
  semantics, and OpenCode all passed.

### 06:16–06:54 — sampling, FP8 draft KV, and topology closure

- Enabled FlashInfer sampling on Windows for a small repeatable sampled-profile
  gain.
- FP8 draft KV reduced memory and raised the controlled result to `135.167`
  tok/s. Two real windows averaged `98.126 tok/s`; semantics and 200K passed.
- Two-step MTP lost. Top-k 2 had no correct sampled path. CUTLASS DSL GDN was
  confirmed unavailable on native Windows.

### 07:08–09:13 — trace the residual wall and remove environmental noise

- GPU traces showed NVFP4 and FP8 GEMM dominated kernel time.
- Repaired and measured a CUTLASS channelwise-FP8 candidate, then rejected it
  after robust paired medians showed no material gain.
- WDDM traffic from ZCode and other desktop clients explained large late-run
  variance. An uncontaminated window reproduced the `98.126 tok/s` result.
- The 09:13 hostile audit closed the then-current topology. Later dispatch work
  superseded its performance numbers.

### 09:21–09:54 — capture draft extend and route target verify through XQA

- Confirmed that a hybrid-backend type gate suppressed draft-extension graph
  capture and that target verification still used the prefill backend.
- Capturing `DRAFT_EXTEND_V2` raised fixed work from `135.167` to `145.941`
  tok/s.
- Added safe top-k-1 target-verify masks and routed verification through XQA.
- The combined fixed mean reached `156.968 tok/s`; two real five-run windows
  averaged `110.750 tok/s`; thinking, tools, language-only surface, and exact
  `199016` passed. XQA became the launcher default.

### 09:58–10:27 — close dense draft quantization and naive q alignment

- Stock `nvfp4_online` reached the draft routing boundary but did not quantize
  dense MTP linears, did not reduce memory, and lost fixed throughput.
- Added an opt-in exact aligned sparse draft-q sampler with graph-static
  buffers. Focused tests passed.
- Top-k-20 alignment imposed about a 10% fixed-work tax and sharply reduced the
  first real samples. It remains opt-in; the qualified default stays disabled.
- Restoring the default produced matched accepted length `2.3925` versus
  `2.4976` for top-k 20. The candidate's 4.39% acceptance-length gain could not
  repay its execution cost.
- The qualified unsimulated server was restored live and the work paused. No
  further candidate should launch until the effort is resumed.

### 10:32–10:37 — close the generic sampling-backend claim

- Work resumed. Source reachability confirmed that EAGLE target rejection and
  draft proposal sampling bypass the generic FlashInfer/PyTorch sampler choice.
- A full-200K fixed-work A/B produced overlapping medians and no reproducible
  sampler-memory recovery. FlashInfer remains selected for fallback sampling;
  adaptive-depth costs are next.

### 10:38–10:58 — retest and promote two-step MTP

- Reopened the old two-step result because draft-extension graph capture and
  target XQA materially changed the cost topology.
- Two steps / three draft tokens reached `159.973 tok/s` fixed and `117.794
  tok/s` across two real five-run windows, 6.36% above the three-step leader.
- Native accepted length was `2.3167` over 221 verification cycles. The cheaper
  width outweighed seven extra cycles versus the matched three-step control.
- One-step fixed work stayed near `102 tok/s` and was closed. Reasoning, tools,
  language-only surface, exact `199016`, tests, and launcher-default relaunch
  all passed; the default server was healthy with all intended graphs captured.

### 11:00–11:12 — close adaptive depth

- Added graph-resident adaptive two/three-step controls and repaired shared-
  logits sizing for the maximum adaptive width.
- Aggressive switching oscillated and reached only `100.739 tok/s`; a sparse
  controller reduced switching yet averaged `110.276 tok/s` over ten samples.
- Both policies lost to restored static two-step controls. Adaptive depth was
  closed for the measured proposal and cost topology.

### 11:21–11:52 — remove redundant chain work and establish GPU safety

- A compact unread XQA mask removed a context-sized generic mask path and
  raised deterministic fixed work to `162.726 tok/s`.
- Reusable fused chain-metadata buffers reached `169.767 tok/s` simulated, then
  exact-seed comparison exposed a real rejection-path scheduling/lifetime loss.
  The unsafe experiment was removed.
- Repeated server reset/capture saturated the display GPU and froze the
  desktop. The exact server tree was cleaned up, unrelated processes were
  preserved, and future work adopted one deliberate server/compile/CUDA job at
  a time with exact process-tree ownership.

### 11:59–13:29 — move hot paths to native code and promote captured alignment

- User direction narrowed new performance hot paths to C++/CUDA with Python as
  the binding and dispatch surface.
- Native-Windows SiLU, RMSNorm, Gemma RMSNorm, fixed-chain metadata, and CUDA
  toolchain support qualified in isolation. FlashInfer CUDA renormalization
  replaced an accidental full-vocabulary Windows sorting fallback.
- Aligned draft top-k 20 first improved real sampling while an eager per-depth
  topology collapsed fixed work. Moving proposal transforms and exact-q capture
  into the single multi-step CUDA graph restored fixed work to `167.776 tok/s`.
- The resulting unsimulated production line reached **122.712 tok/s** over ten
  real samples, median `122.371`, peak `137.074`, with acceptance mean `2.318174`.

### 13:27–14:51 — qualify native gates and close draft quantization

- A bit-exact native BF16 attention sigmoid gate contributed to a new safe
  fixed control of **171.263 tok/s**.
- Windows online FP8, MXFP8, and dense NVFP4 mechanisms were repaired through
  registry, loader, quantizer, backend, and graph-capture boundaries.
- Full MTP FP8 reached `167.023`, MXFP8 `163.457`, and dense NVFP4 `164.094
  tok/s` fixed. All remain opt-in compatibility/capacity mechanisms; BF16 MTP
  stayed selected for throughput.
- Three-step full acceptance produced one external `201.251 tok/s` sample,
  proving compute feasibility. Honest sampled three-step averaged `117.239`,
  and proposal top-k 8 also lost. Useful work per verification remained the
  binding problem.

### 14:56–15:18 — evaluate and close the Gittensor checkpoint branch

- Downloaded and verified the 25-file Gittensor RTX 5090 ModelOpt FP4
  checkpoint without mutating the source artifact.
- Windows `modelopt_fp4` registration enabled exact 200K serving. The checkpoint
  reduced target residency and improved acceptance/prefill, yet reached only
  `154.883 tok/s` fixed and `119.092 tok/s` real.
- The user closed the hybrid-lm-head branch and restored RadixArk as the
  production reference.

### 15:22–16:44 — build and measure an exact recurrent tree verifier

- Added an exact GPU target-only tree sampler, low-rank GDN tree replay, and
  accepted-path recurrent/conv state commit for native Windows. Distribution,
  production-stride, CUDA-graph replay, and commit tests passed.
- Root-heavy M12 served correctly and reasoned `37 * 19 = 703`, but produced
  about three tokens per traversal. Depth-only M16, aligned scoring, M8 width,
  and scalar depth discount all lost on yield or cycle cost.
- Trace analysis measured roughly `24.431 ms` of captured work per M12 cycle;
  the topology needed more useful output than its five-token maximum could
  deliver at the historical 200 tok/s target.

### 16:44–18:31 — qualify exact SWOR and exhaust current-q topology search

- Implemented exact sampling without replacement, fixed irregular topologies,
  uniform proposal fallback, and native distribution/graph-replay coverage.
- Replaced repeated dense residual scans with a shared-memory path for supports
  up to 64 entries, reducing verifier cost from `1.350 ms` to `0.359 ms` per
  cycle while retaining a dense fallback.
- The first M12 SWOR topology emitted `2.9653` tokens/traversal and averaged
  only `84.713 tok/s`. Path oracles measured a dominant rank-zero spine and
  declining deeper sibling value.
- Offline cost/yield search reached only `3.9800` expected outputs with measured
  decay and `4.0921` under an optimistic no-decay model. Topology rearrangement
  at the current q distribution was closed; overlap and per-depth cost became
  the conditions for a future reopening.

### 18:31–19:06 — preserve experiments, reject 232K, and complete the goal

- Committed the exact tree/SWOR verifier, recurrent commit, sparse CUDA path,
  path/overlap oracles, topology tools, tests, and trace evidence behind opt-in
  controls. Production defaults remained linear.
- Restored the qualified RadixArk linear server and passed a sampled smoke with
  reasoning/tools and language-only surface intact.
- A requested 232K pool captured and completed exact `231000+16`, yet left only
  98 MiB free before cache flush. The user rejected that operating margin.
- Restored both launcher defaults to 200K, captured all three speculative graph
  phases with 1.84 GiB reported headroom, and marked the exhaustive NVFP4
  optimization goal complete.

### 21:08–23:24 — remove the tree seam, sparsify GDN, then stop on accepted-path correctness

- Formed an opt-in two-graph steady cycle by capturing draft extend, the device
  bridge, and next draft decode in one composite graph. Replaced dense N-squared
  GDN pair state with strict sparse ancestry; M12 reductions fell from 288 to 56.
- Repeated M8/M12/M16 measurements still reached only 97.352, 94.685, and
  92.831 tok/s means. The offline p/q grid found scalar temperature/support
  changes essentially flat, with at most 0.000245 overlap improvement.
- A deterministic accepted path `[0,3,7]` then exposed a correctness defect:
  the unified hybrid pool treated virtual target-KV ids as physical ids, and
  the multi-layer worker could skip the front compaction its consumers assume.
- Added physical relocation translation distinct from MLA dense kernel ids,
  removed the compaction opt-out, and proved a captured four-cycle serial-path
  comparison with rejected-slot reclamation and virtual-id reuse. Prior tree
  throughput is mechanism-only until a corrected full-model gate passes.
- A fresh unchanged linear comparison retained every tree switch off. The
  first five samples averaged 112.253 tok/s during startup recovery; a second
  independent warmed window averaged 124.775 tok/s, confirming the production
  comparison path remained in its established range.

### 23:24–01:17 — close graph-tail work and expose the target/proposal frontier

- Exact linear composition preserved semantics and raised acceptance, yet its
  categorical form cost 21.132 ms/cycle and reached 120.075 tok/s. Ordinary
  scheduling remained selected.
- Added asynchronous CUDA-event timestamps at raw graph boundaries. Two
  independent windows produced 1,471 records; the best repeatable recoverable
  tail was 0.658355 ms, below the 0.75 ms admission floor.
- Added branch-exact p/q capture with branch-local additive and repetition
  state, explicit worker/compile provenance, and immutable multi-policy replay.
  Selected-tree capture now makes every unsupported counterfactual unavailable.
- Exact Qwen3.5 target attribution matched 305 primary GEMMs per M3 replay.
  The target graph spans 15.322 ms mean; primary GEMMs occupy 12.360 ms on the
  terminal stream. NVFP4 MLP gate/up and down expose 6.539 ms of that path.
- Measured M3/M8/M12/M16 geometries all fail the impossible path-length oracle.
  Candidate funding now requires complete lattice coverage, a conservative
  projection of at least 215 TPS, and strict clearance of the measured
  emitted-token/full-cycle-cost frontier.

## 2026-08-17–20

### Selective target NVFP4 sets the primary 200K scoreboard

- Converted the exposed FP8 target projection families into a distinct,
  provenance-tracked NVFP4 checkpoint. The measured M3 cycle fell from 19.446
  ms to 17.315 ms while reasoning and tool probes passed.
- The checkpoint completed the exact `199000+16` capacity run at **2838.980
  prompt tok/s**, **107.253 generation tok/s**, **70.096 s TTFT**, and **70.235
  s** end to end, with exact `199016` tokens.
- The user selected this exact near-limit workload as the primary performance
  scoreboard. Root `BENCHMARK.md` carries the record to beat; qualification
  continues to distinguish experimental records from production selection.

### Current-source M4 retest closes plain K+1

- Reproduced M3, tested M4, then restored M3 with the same selective
  checkpoint, seed, real 200K pools, and runtime backends.
- M4 acceptance rose **2.245614 -> 2.327273**, while measured full-cycle cost
  rose **16.058328 -> 18.419190 ms**. Projected throughput fell
  **139.841 -> 126.350 tok/s**.
- Warmed exact-200K prompt throughput was unchanged; 16-token generation
  remained too variable to support its isolated peaks. Plain SGLang M4 is
  rejected. The patched vLLM TurboQuant/full-graph K+1 architecture remains a
  separate information gate.
- The existing FlashInfer paged-only switch was then rejected: matched
  exact-200K prompt changed -0.135%, 512-token generation changed -2.207%, and
  its deterministic output diverged from the selected ragged/paged merge.

### Selective chunk 7680 clears the prompt milestone

- A 4096/5120/6144/6656/7168/7680/7808 sweep found a sharp selective
  long-context optimum at 7680. Eight exact prompt samples averaged 2997.744
  tok/s and peaked at **3002.344**, with best TTFT/E2E
  **66.281538/66.434400 s**.
- Exact `199000+512` support runs averaged **3001.742 prompt / 109.836
  generation tok/s** and peaked at **3004.324/110.693**. Two independent
  sampled `6213/512` windows averaged 138.537 and 139.885 tok/s; behavior,
  tools, model surface, and selective headroom passed.
- Global promotion was rejected after base RadixArk reached only 2226.770
  prompt tok/s and 200 MiB free before follow-up probes. Production stays at
  4096; selective 7680 is an explicit benchmark profile.
- Single-layer selected-row draft-extend logits then reduced graph memory but
  not time: graph span was 1.061 ms versus 1.059 control and full M3 cycle was
  16.066558 ms versus 16.058328. The patch was removed.

### Direct Gemma output clears the combined milestone

- Native Windows previously allocated a temporary Gemma-normalized tensor and
  copied it back after `residual.add_(x)`. Passing `x` directly as the existing
  JIT kernel's output preserves bit-exact arithmetic while removing the
  allocation/copy.
- The local Qwen shape fell 38.731 -> 29.254 us at M1 and 37.578 -> 29.184 us
  at M3. Two independent sampled windows averaged 144.535 and 138.621 tok/s.
- Exact `199000+16` set the new record at **3016.444 prompt / 112.355
  generation tok/s**, 65.971714 s TTFT, and 66.105219 s E2E. A fresh restart
  independently reached **3013.736/112.012**.
- Launcher-default base RadixArk also passed exact capacity, arithmetic,
  tools, model surface, OpenCode2, and post-flush headroom; production chunk
  remains 4096.

### A new branch starts from a fresh current-source baseline

- An explicit new optimization request reopened the performance lane from
  `adf3a620ef64` without touching the user-owned `BENCHMARK.md` edit or
  `HANDOFF.md` deletion.
- The selective chunk-7680 server resolved the intended M3 200K route and
  captured all three speculative graphs.
- After two full warmups, five exact `199000+16` scores averaged
  **2871.358 prompt / 90.459 legacy generation tok/s**. Prompt CV was 0.747%;
  generation CV was 15.633%; every result retained exact counts and digest.
- Active WDDM clients and accumulated software power-capping accompanied the
  4.810% prompt gap from the historical record. This is the immediate matched
  control, not a replacement for the qualified winner.

### Large-EXTEND FP4 tactics set a new exact-200K record

- Found that speculative startup rewrote target dummy forwards to
  TARGET_VERIFY, so ordinary large EXTEND FP4 buckets were never profiled.
  The existing expert opt-in now admits a narrowly asserted target EXTEND
  pass while draft workers and default behavior remain unchanged.
- An independent retune produced five exact prompts averaging **3046.912
  tok/s** and a same-request record of **3048.086 prompt / 112.499 generation
  tok/s**, TTFT **65.286869 s**, and E2E **65.420204 s**.
- Cache-only and dummy-only controls returned to the baseline tactic digest,
  proving the gain came from FP4 selection rather than stale Mamba/KV state.
  Fresh profiling was rejected after a different valid tactic set reduced
  long generation to **101.162 tok/s**.
- FlashInfer file hits were being cleared by later draft autotune contexts.
  Promoting only the 110 target entries exercised by the large EXTEND pass
  into the runner-keyed process cache reproduced the selected cache without
  re-profiling. Its five exact prompts averaged **3047.309 tok/s**; three
  exact long requests averaged **118.389 generation tok/s**.
- Reasoning, tools, language-only surface, sampled generation, native
  acceptance, standalone OpenCode2, exact capacity, and headroom passed. An
  unchanged base RadixArk/chunk-4096 relaunch ran no extra EXTEND pass,
  captured all three graphs, completed exact `199016`, and retained the
  production defaults.
- Retained in `7f5af878da7b8dc43063f31e554dfc69cee5d510`. The selected
  20,928-byte cache has SHA-256
  `8219484FA86EBB0E6DDA54F2D15447DBC502EBCEA9007B3E1BB917B9001F9ADF`.

### Fuse exact eager normalization and MLP activation boundaries

- Fused residual addition into the bit-exact native-Windows Gemma norm. The
  M1/M3 kernel boundary fell from about 16.5 to 9.5 us, and adjacent exact
  long generation improved **115.194 -> 116.583 tok/s**.
- Added a precise native SwiGLU-to-NVFP4 producer for Qwen MLP down
  projections. Random production shapes initially passed, but an exhaustive
  finite-BF16 sweep found 520 underflow-group packed-byte differences. The
  repair recreates FlashInfer's final FTZ quantizer boundary and passes compact
  plus TMA all-finite coverage.
- A second discriminator proved eager prefill and compiled target verification
  have different established arithmetic: Inductor removes the intermediate
  BF16 SiLU round. The producer is therefore eager-only; compiled M3 retains
  its former path and deterministic trajectory.
- Five repaired exact `199000+16` requests restored the established digest and
  averaged **2987.275 prompt tok/s**, **0.914%** above the adjacent PERF-028
  arm, with TTFT improved by **0.606649 s**. Three exact long requests restored
  their digest and averaged **3001.344 prompt / 115.225 generation tok/s**.
- A separately exact compiled-semantics producer reduced isolated M3 launch
  time from 70.848 to 25.152 us, but 233 full-cycle samples retained a
  **16.045 ms** median versus **16.058 ms** control. Its +0.839% long-client
  movement was noise; the experiment was removed.
- Retained in `5ea3b734b0`. The headline record remains PERF-024; these
  additive changes are the active source for the next optimization.

## Supersession map

Use these results when older “final” checkpoints conflict:

| Topic | Current value | Supersedes |
|---|---|---|
| Fixed `6213/512` | `171.263 tok/s` safe five-run mean | `167.776`, `162.726`, `159.973`, `156.968`, `135.167`, `86.016` |
| Real sampled `6213/512` | `122.712 tok/s` ten-run mean | `121.075`, `117.794`, `110.750`, `98.126`, `96.110` |
| Near-limit `199000/16` record | `3048.086` prompt, `112.499` generation tok/s on selective target NVFP4 + chunk 7680 + direct Gemma output + selected large-EXTEND FP4 tactics; qualified production remains `2608.263/102.358` | `3016.444/112.355`, `2838.980/107.253`, `2570.356`, `2429.153`, `2423.812`, `2200.563` |
| Production capacity | `200000` context and token pools | rejected `232000` operating-margin experiment |
| Speculation geometry | 2 steps / 3 draft tokens | 3 steps / 4 draft tokens |
| Target verification | TRT-LLM MHA/XQA | FlashInfer-prefill verify route |
| Draft extension | CUDA graph captured | eager draft extension |
| Draft-q alignment | top-k 20 inside the single multi-step CUDA graph | eager/per-depth aligned proposal path |
| Tree mode | opt-in exact target-only/SWOR infrastructure; linear production default | current-q M8/M12/depth/topology-only candidates |
