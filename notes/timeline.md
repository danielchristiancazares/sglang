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
- Fixed FlashInfer paged-prefix splits failed the 128 MiB workspace gate, and
  source inspection showed the proposed packed GDN verify path was already a
  zero-copy alias. Both routes closed without retained source.
- Coalescing the final two prefill passes preserved exact capacity but moved
  work into a 14,680-token ragged causal kernel, collapsing prompt throughput
  to 1,917.509 tok/s and changing output. The scheduler experiment was removed.
- An exact attention-gate-to-NVFP4 kernel improved isolated large shapes by
  37%, but only 16 layers use it; projected target and exact-prefill savings
  were 0.0068 ms and 20.9 ms. It closed before model wiring.
- Page 128 failed exact pool capacity and page 32 regressed long generation.
  FlashInfer prefill already uses per-token indices with logical page size 1,
  so global storage-page tuning cannot address the 78.1% paged-prefix wall.
- FlashInfer FP16 QK reduction initially moved adjacent server means by +0.679%,
  but a corrected exact 24Q/4KV/256-dimension prefix ladder measured it
  **163.705 ms slower** across 16 layers. The provisional opt-in was removed.
- Native FA2 tile screening then closed the practical dispatcher family for
  the same paged-prefix wall. CTA-Q 16 was much slower, CTA-Q 32/128 were
  invalid, and the correctly routed CTA-Q-64 `NUM_MMA_KV=2` kernel regressed
  the exact ladder **13.306%** and changed output/LSE digests. The dependency
  header and generated cache were restored before pivoting back to a
  repository-native SM120 decode fusion.
- An exact native residual-norm-to-NVFP4 producer improved its isolated M3
  launch but was neutral at the dependent projection boundary:
  **0.096704 ms staged vs 0.097152 ms fused** per layer. Existing PDL already
  hides activation quantization behind gate/up GEMM startup; the prototype
  closed before model wiring.
- A native 64-row M3 NVFP4 GEMM prototype then hit CUTLASS's architectural
  floor: cooperative scheduling requires CTA-M 128, while ping-pong cannot map
  the fixed 128-row scale TMA atom. FlashInfer already swaps A/B and uses the
  minimum supported CTA-N 32, closing plain tile specialization.
- An occupancy-preserving native MTP dual-norm/concat producer was exact
  through its BF16 FC but saved only 1.248 us at M1 and 2.080 us at M3, about
  0.0033 ms across both draft phases. It closed before routing.
- The proposed gate/up GEMM epilogue proved to require a custom swap-AB
  collective rather than a stock EVT functor and was deferred.
- Native sparse top-p retained AIR's exact radix pivot and replaced its dense
  apply after top-k 20. Fifteen CUDA plus six integration tests passed;
  A-B-A target-cycle medians improved 16.954/17.322/16.002 ms, and final
  reviewed source independently reached 16.001 ms with identical output and
  acceptance. Predecessor long generation still averaged only 111.559 tok/s
  under 2.194869 mean acceptance, so the kernel remains a default-off additive
  win rather than a new record. Signed commit `7cb4ed0796` retains the kernel,
  benchmark, and regression coverage.
- The earlier eager-fusion line remains in `5ea3b734b0`; the active source now
  includes `7cb4ed0796`. The headline record remains PERF-024.
- Page-aligned ordinary-prefix FlashInfer prefill then reused the physical
  64-token KV pages instead of a token-granular page table. The exact 25-shape
  ladder was bit-exact and 5.270% faster; five exact prompt scores averaged
  **3209.728 tok/s**, with every prompt/TTFT/E2E gate passing. The combined
  objective remains open because short generation averaged 98.029 tok/s.
  Signed commit `afd5606077` retains the default-off native route.
- Static draft top-k 32 reduced mean acceptance from 2.217279 to 2.173943 and
  was rejected; top-k 20 remained selected at that stage.
- Greedy draft top-k 1 also failed: exact short generation was 97.900 tok/s
  and greedy acceptance averaged 2.107020, so broad q support remains useful.
- Draft top-k 16 averaged 2.205710 acceptance versus k20's 2.217279. With
  k1/k8/k16/k32 all losing, further proposal work requires conditional or
  learned calibration rather than another scalar support size.
- The first proposal-only top-p 1.0 probe missed live `EAGLEWorkerV2`, so both
  profiled arms were controls. After routing through both actual proposal
  owners, AIR top-p fell from three to one launch/cycle and matched cycle
  mean/median/p90 improved by 0.194/0.185/0.149 ms; the default-off win was
  retained in signed commit `6b963eed05`.
- Correctly routed proposal penalty scales 0.75 and 0.0 reproduced identical
  proposal/output sequences; additive-penalty calibration was removed.
- Target ReplaySSM commit then moved to a dedicated side stream overlapping
  draft extend, with a forward-stream rejoin before scheduler return. Direct
  interval analysis showed 186.8 us hidden fold but 176.4 us graph expansion;
  the combined boundary was neutral/slower, so the source was removed.
- Two branch-exact p/q captures then ruled out static gamma, rank, and token
  calibration: all learned corrections overfit early chronology and regressed
  later states. The diagnostic queue was raised from eight to bounded 64 after
  the first writer backpressured; the independent 239-record capture passed,
  and signed commit `4d6782121e` retained the repair.
- A no-penalty greedy oracle then justified reopening draft k1. It improved
  exact199K+512 generation from 116.549 to 123.049 tok/s with identical output,
  but five exact16 samples stayed near 98.5 because seven 199K cycles averaged
  19.895 ms. M4 also required seven cycles and was rejected.
- XQA SM-count and PDL sweeps found no bit-exact material control; long-context
  target-graph reduction now requires a kernel change or a six-cycle proposal.
- The retained device-resident cycle was reopened under greedy k1 and measured
  97.730 tok/s on exact16 with the control digest, so graph composition remains
  rejected. Native XQA structural sweeps found only sub-microsecond valid
  changes; a faster single-K-buffer build was nondeterministic and removed.
- Exact-q/hidden capture then proved q20 contains the target at all six
  required exact16 oracle positions, but blocked PCA-linear rank heads failed
  minority validation. No learned proposal was retained.
- Native CUDA direct construction of greedy one-hot q became a real additive
  win: **122.352 -> 123.559 tok/s** matched long generation, **123.831** on an
  independent restart. Exact16 reached only **99.173**, so the original
  benchmark target remains open rather than being redefined.
- All 32 target FP4 tactics and CUTLASS PDL were then screened. PDL-off
  regressed; bit-exact qkvz/down tactic replacements moved real generation
  only **+0.114%** and were rejected as noise. Original source/cache returned.

### Hybrid Marlin becomes the Windows default

- Native SM120 Marlin initially improved exact16 generation to 114.820 tok/s
  but regressed full prefill to 1984.193 tok/s. A coalesced in-place layout
  converter then made Cutlass-prefill/Marlin-decode switching practical and
  matched the canonical repacker bit-for-bit.
- Projection and layer isolation found that all 64 target gate/up projections
  preserve the useful exact trajectory; full-target, draft-only, and partial
  layer masks were slower or required more verify rounds.
- The accepted exact result is **3078.058 prompt / 114.617 generation tok/s**,
  **64.651152 s TTFT**, and **64.782022 s E2E**. It completed exact `199016`
  with `finish_reason=length` and beat all four prior record values.
- A clean no-argument launcher restart independently reached
  **3052.437/114.053**, preserved exact capacity, captured all three graph
  phases, passed arithmetic/tools/tool continuation/model surface/OpenCode2,
  and left 4,338 MiB free after cache flush.
- The user accepted the all-four-metric improvement for normal use. Signed
  source commit `03ba3d2e27` promotes the selective checkpoint, chunk 7680,
  native draft-k1 q, and gate/up hybrid Marlin as Windows launcher defaults.
  The 3100/120 milestone remains the next record target, not a deployment gate.
### Apple Silicon Q2 lane restores behavior and removes mixed-weight copies

- Pinned Bartowski's conventional Qwen3.8-27B IQ2_XXS GGUF as the retained
  32 GiB playground and added native packed Q2_K/Q4_K/IQ2_XXS/IQ1_M Metal
  matmul/embedding support.
- Repaired GGUF USER_DEFINED vocabulary registration, restoring atomic Qwen
  reasoning/tool markers. Live gates now return preserved thinking, final
  `703`, exact thinking-off `READY`, and one parsed multiply call.
- A pinned current llama.cpp Metal build completed the exact `12+256`
  scoreboard at **14.661356 tok/s aggregate** with a **14.671473 tok/s** best
  hit, establishing the measured M1 Max Q2 reference.
- The repository path's mixed merged projections were rebuilding 478.125 MiB
  of packed weights per token. Compact MPS storage in `13bea403d6` removed all
  40 copies, reduced reported weight residency **10.03 -> 9.03 GB**, and
  improved five-run `128+32` generation **3.1858 -> 3.309 tok/s** with the
  identical digest.
- Native IQ2 kernel reuse is next. The former affine-q4 scoreboard came from a
  separate Mac Pro and has been deleted from the M1 Max record.

### Native IQ2 and Q5 batch-one kernels cross 8 tok/s

- Replaced generic IQ2_XXS batch-one scalar dequantization with a staged-table
  two-SIMD/four-row Metal mapping adapted from pinned ggml. Matched projection
  time fell `1.176875 -> 0.516000 ms`; exact served `128+32` generation rose
  `3.309 -> 7.1748 tok/s` with the deterministic digest intact.
- Constant-address lookup tables and a four-SIMD/two-row geometry both lost
  matched ablations. Signed `16b2bf7a06` retains the selected shader, focused
  boundary harness, and full MIT provenance.
- The retained artifact's vocabulary head is Q5_K, shape `248320x5120`, and
  read 0.814 GiB of packed weights per generated token. A four-cohort
  batch-one specialization reduced matched head time
  `19.659291 -> 3.754625 ms`.
- Five deterministic served runs averaged **8.0284 tok/s** and the committed
  restart reached **8.114 tok/s**. Required sampled windows averaged
  **7.9450** and **7.9552 tok/s**. Reasoning, thinking-disabled, parsed tool,
  preserved tool-result, and language-only gates passed. Signed
  `b19cf4acf3` retains the kernel and alignment/extrema test.
- The route-neutral M1 Max Q2 `12+256` reference remains **14.661356 tok/s**.
  The next native hotspot is the 48-call batch-one F32 GDN b/a path; exact
  SGLang-ingress qualification remains open.

### Native F32 projection reaches 8.3 sampled tok/s and exposes the client-capacity gate

- The compact loader presents every GDN layer's alpha/beta pair as one actual
  `96x5120` F32 b/a projection. Reusing input rows inside the custom Metal
  matvec reduced occupancy and lost three exact-shape ablations.
- Routing only the one-vector F32 case through native MPS matrix
  multiplication reduced the 48-layer sweep from **7.296667 ms** to
  **2.159000/2.051708 ms**. Multi-vector prefill retains the custom kernel.
- Five deterministic served runs averaged **8.4406 tok/s**. Required sampled
  windows on two independent launches averaged **8.3094** and **8.2942
  tok/s**, with arithmetic, thinking-disabled, tool parsing, preserved tool
  result, digest, and language-only behavior intact. Signed `4d1641fdcd`
  retains the change and its actual-weight benchmark and test.
- A process-scoped OpenCode 1.18.15 request reached the exact endpoint with a
  13,635-token main agent prompt. The 1,024-token diagnostic launch rejected
  it at admission, making native context enablement the next funded step. The
  exact M1 Max Q2 reference remains **14.661356 tok/s**.

### Torch-native partial prefill stops recomputing prefix queries

- The generic extend path padded every partial chunk's query back to the full
  KV length solely to obtain causal alignment. On MPS this also selected a
  two-pass temporary proportional to that manufactured query length.
- Signed `210a214c12` runs SDPA on only the new rows and supplies the required
  lower-right causal or offset sliding-window mask. Future-value sentinel,
  shuffled-cache GQA, ragged, noncausal, and empty-extend coverage passes.
- Exact `4096+256` source medians improved about 7.77x. Exact `4096+4096`
  five-sample medians improved 3.08-3.64x with zero observed output error.
- The remaining native context blocker is decode admission. A 32K/BF16 pool
  currently reaches a Metal binding restricted to FP32 and at most 7,936
  physical rows; the next narrow change routes incompatible tensors to the
  already established cache-write plus SDPA fallback.

### Torch-native decode admits long BF16 pools safely

- Signed `b2b8ab4af8` makes fused native MPS decode conditional on the
  binding's actual FP32, NHD, contiguous-cache, head-dimension, and 7,936-row
  contract. The dispatcher owns both native admission and the existing
  cache-write plus SDPA fallback.
- BF16 with 32,769 physical rows and FP32 with 7,937 rows failed before the
  change. Both now complete with zero observed error. FP32 at exactly 7,936
  rows stays fused and agrees within `2.5331974e-07`.
- Source-level prefill and decode blockers are cleared. The next rung is a
  controlled 32K BF16 full-model capacity launch followed by the measured
  13,635-token process-scoped OpenCode request. Fixed-memory native GQA remains
  funded afterward because the fallback enables capacity rather than solving
  long-history decode cost.

### Shared-dequant IQ2 prefill clears the 300-second watchdog

- PERF-A014 adds an Apple7-gated FP32 SIMD-matrix kernel for IQ2_XXS batches
  above eight. A 64x32 dequantized weight tile is shared across 32 input rows;
  the selected batch-one/four/eight kernels remain intact.
- On the actual `17408x5120` gate projection, batch 128 changed from a matched
  **70.074833 ms** control to **4.250250/4.277125 ms** candidate A/B/A arms.
  Batch 4096 changed from **1971.539875 -> 124.838125 ms**. Aligned and odd
  output/batch tails pass with maximum relative error `2.27121e-06`.
- The 32K-configured BF16 model now completes cache-flushed `4096+2` at **24.828
  prompt tok/s**, **164.975078 s TTFT**, after the former source crossed its
  300-second watchdog. The independent `5000+1` two-chunk rung reaches
  **24.845 prompt tok/s** and **201.251071 s E2E**.
- Required sampled reasoning, final arithmetic `703`, thinking-disabled
  `READY`, one parsed multiply call, preserved tool-result continuation, and
  image/audio-disabled surface all pass. Batch-one decode is unchanged, so the
  compact Q2 scoreboard remains **14.661356 tok/s** through pinned llama.cpp;
  exact SGLang `12+256` had not yet been measured in this phase.
- Five matched exact-`128+1` controls with the new path process-disabled
  averaged **7.0234 prompt tok/s**. Two independent default five-run windows
  averaged **22.9556** and **22.8072 tok/s**, directly attributing a 3.258x
  full-model prefill gain while leaving small-batch decode unchanged.

### Cross-machine q4 record removed; local Q2 frontier restored

- User-authoritative provenance established that the affine-q4 record came
  from a separate Mac Pro. Its decode, maximum-context, speculative, and
  capability claims were deleted from the M1 Max benchmark authority.
- Pinned llama.cpp build 10547 now owns the route-neutral M1 Max Q2
  `12+256` reference at **14.661356 tok/s aggregate** and **14.671473 tok/s**
  best hit. Exact SGLang ingress was still open at this point; the next phase
  resolves it.
- A mistaken partial q4 checkpoint restore was stopped and its isolated cache
  removed. The retained Bartowski Q2 artifact and every unrelated cache were
  preserved.

### Exact native SGLang Q2 baseline establishes the decode gap

- The Rust ingress requires a directory containing `tokenizer.json`; the
  GGUF-only tokenizer path failed cleanly before serving. Qwen's official
  tokenizer files were pinned at immutable revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` without downloading dense model
  weights.
- One warmup and five exact `12+256` native SGLang requests reached
  **7.001584 tok/s aggregate**, **7.015010 tok/s** best hit, and
  **36.563154 s** mean E2E. Every request length-finished with identical token
  IDs and the record's FNV `6d4d220de481f54e` output.
- This window qualified fixed decode and `/model_info` only. The earlier
  semantic, sampled, and restart evidence used Python ingress plus the GGUF
  tokenizer; those gates remain open for the official-tokenizer Rust route.
- The route-neutral llama.cpp Q2 record is **2.094006x** faster on the matched
  fixture. Batch-one decode profiling now governs the next native C++/Metal
  candidate; near-capacity and standalone OpenCode gates remain open.

### PERF-A016 accelerates the Q4_K tensor family inside the Q2 checkpoint

- Signed `52b5326d8e` adapts pinned llama.cpp's two-row Q4_K activation reuse
  into the native Metal `quant_matmul` owner. The default path requires batch
  one, four-row output alignment, complete four-block cohorts, aligned compact
  weight/input origins, and Apple7+ pipeline capability; the generic kernel
  owns every remaining shape.
- Production-view tracing showed that a temporary zero-offset guard excluded
  27.173913% of packed Q4_K traffic. The final `%2` packed-weight and `%4`
  float-input alignment rule restores 24 GDN QKV shards plus eight full-
  attention K projections. Candidate and output-tail actual-file parity pass
  with maximum Q4_K relative error `5.35063e-07`.
- The final official-tokenizer/Python-ingress candidate reaches
  **8.586948 tok/s** aggregate and **8.591773 tok/s** best hit on exact
  `12+256`. Its fresh disabled-kernel control is **7.009167 tok/s**, a matched
  **22.510241%** full-model gain. An independent candidate restart reaches
  **8.578205 tok/s**, and every arm reproduces the same token stream and
  digest.
- The selected 1K-chunk route completes exact `32761+1`, required sampled
  reasoning, arithmetic `703`, thinking-disabled `READY`, one parsed multiply
  call, preserved tool-result continuation, and image/audio-disabled reporting.
  `Q4_K` identifies internal tensors in the mixed-format Bartowski IQ2_XXS
  artifact; the machine, checkpoint, and record remain the M1 Max Q2 lane.

### Named Codex profile becomes the Apple real-client gate

- The machine-local `$CODEX_HOME/qwen38-local.config.toml` profile selects the
  local Q2 model, SGLang's Responses endpoint, 32,768 context, medium reasoning,
  a 900-second stream-idle bound, and read-only sandboxing. Its static catalog
  identifies a text-only `shell_command` surface and declares sequential tools;
  concurrency remained unqualified. Codex CLI is 0.149.0; profile/catalog
  SHA-256 values begin `9706003a` and `a67c491a`.
- A fixed no-tool run first admitted 8,839 input tokens, preserved 38 reasoning-
  output tokens, returned exact visible `CODEX READY`, and exited zero. The
  final read-only tool gate then issued `/bin/zsh -lc pwd` exactly once,
  consumed `/Users/dcazares/sglang`, returned exact visible
  `CODEX TOOL READY`, and exited zero with 17,871 input, 96 output, and 62
  reasoning-output tokens across the initial and follow-up Responses requests.
- The worktree remained unchanged across the Codex tool run, the server stayed
  healthy, cache flush succeeded, and verified leaf-first cleanup returned
  port 30000, 94% free memory, and normal thermal status. The earlier
  process-scoped OpenCode requests remain chronological admission evidence;
  the named Codex profile owned the Apple real-client decision at PERF-A016.

### Fixed-memory Metal EXTEND reaches the 131K isolated rung

- The generic Apple EXTEND route's source-visible working set reaches
  **10.1328125 GiB** per full-attention layer at `E=1024,L=65536` and
  **20.1953125 GiB** at `L=131072`, explaining the measured 64K swap and
  forward-progress collapse.
- PERF-A017 adds an Apple7+ BF16 paged-GQA Metal kernel for the production
  batch-one, 24-query-head, four-KV-head, dimension-256 geometry. Q8/C64
  online softmax uses 20,800 bytes of threadgroup memory and caller-owned
  output, giving zero history-dependent global auxiliary allocation.
- Shuffled mappings, nonzero storage offsets, Q8/C64 tails, query lengths
  through 1,024, strict invalid metadata, and host admission guards pass. The
  maximum observed dense-reference error is `1.3113022e-06`.
- At `E=17,L=131072`, final-source native samples have a **137.906625 ms** median and
  add `0 MiB` measured driver residency after inputs are resident. Dense MPS
  SDPA takes **424.528292 ms**, adds **8,088.515625 MiB** of driver residency,
  and matches the native output within `4.3120235e-07`.
- The shader and pipeline are isolated in a lazy Metal library; established
  native MPS operations keep their original shared-library initialization and
  pipeline cache.
- Production routing remains open under the no-new-Python constraint because
  the raw binding is outside the live backend call chain. Served context
  qualification remains 32,768 tokens while C++ dispatch ownership and the
  consecutive-cache-run fast path proceed.

### PERF-A018 loads consecutive BF16 cache runs directly

- Eight threads now classify the eight possible consecutive runs once per
  C64 tile. Direct BF16 SIMD-matrix loads serve eligible QK/PV operands, while
  the established FP32 staging route continues to own fragmented, invalid,
  and partial runs. The shared run table raises dynamic threadgroup storage by
  32 bytes to **20,832 bytes**.
- At `E=256,L=4352`, identical seeded inputs and an identical ascending map
  measured **59.526688 ms** with direct admission forced off and
  **26.801604/26.900646 ms** in surrounding direct-load builds. All arms
  produced one exact SHA-256 digest. A zero-eligible one-swap map measured
  **58.773563 ms**, clearing the 11.5% fragmented-map regression of the
  rejected per-consumer classifier.
- At `E=17,L=131072`, matched same-map medians are
  **148.002792 -> 66.553042 ms**, a **55.03%** reduction with exact digest
  parity and `0/0 MiB` measured current/driver allocation growth after inputs
  are resident.
- Storage offsets and physical starts `0..7`, direct/fallback and half-mixed
  maps, duplicates/gaps/invalid slots, Q8/C64 tails, every prefix residue,
  causal future sentinels, and physical row 131,072 pass. The raw mechanism
  remains outside the serving call chain, so the Apple production/capacity
  record is unchanged pending an approved backend dispatch seam.

### PERF-A019 publishes run flags from slot-loader registers

- Two slot-loader SIMDgroups now classify their own eight-lane cohorts with a
  first-slot broadcast and three XOR reductions. Cohort leaders publish run
  starts, and one threadgroup barrier makes slots plus run flags visible to all
  consumers. This removes one barrier and 64 shared classifier reads per C64
  tile without changing the 20,832-byte scratch contract.
- At `E=256,L=4352`, a restored signed-checkpoint arm measured
  **27.009396 ms**; candidate arms measured **26.398334/26.592563 ms**. The
  zero-eligible map moved only **58.832500 -> 58.874062 ms**.
- At `E=17,L=131072`, matched medians are
  **66.577542 -> 65.565041/65.493667 ms**, with exact digest parity and zero
  measured current/driver allocation growth. Every broken cohort position,
  duplicates, invalid slots, a run at the upper cache boundary, a run crossing
  physical row 64, and a repeated two-tile hybrid map pass.
- A full 32-query-matrix hoist was bitwise exact and regressed direct and
  fragmented maps by roughly 3.4-6.5%; its expanded live-register design is
  closed as `PERF-FA056`.

### PERF-A020 hoists direct cache-run address generation

- Each SIMDgroup now computes its two QK physical row offsets before the D256
  loop, and each PV key block computes one value-row base before eight output
  fragments. Direct/fallback predicates, BF16 matrix operands, FP32 arithmetic,
  and the 20,832-byte scratch contract stay unchanged.
- At `E=256,L=4352`, two matched source pairs improve
  **26.330084 -> 25.548479 ms** and **26.649625 -> 25.674125 ms**. The
  zero-eligible route also improves by **1.33-1.52%**.
- At `E=17,L=131072`, two matched pairs improve
  **65.647625 -> 63.767667 ms** and **65.687333 -> 63.886417 ms**, preserving
  the exact timing-fixture digest and zero measured allocation growth.
- Every cache storage offset and physical start `0..7`, repeated mixed
  direct/fallback execution, and physical row 131,072 pass. A uniform causal
  branch and cohort-leader online-softmax broadcast were exact and missed
  their timing gates; they remain closed as `PERF-FA057/058`.

### PERF-FA059 through PERF-FA063 close direct-load barrier elision

- PV-only, QK-only, combined, leading-only, and trailing-only removals of the
  direct BF16 matrix-load `mem_none` barriers all preserved exact output and
  zero measured current-allocation growth.
- Against an opening PERF-A020 control at **25.686792/63.915500 ms**, the best
  apparent candidate movement was only **0.720%**. A final independently
  rebuilt PERF-A020 control reached **25.520083/63.745709 ms**.
- Every candidate missed the predeclared reproduced 1% floor. Signed PERF-A020
  synchronization is restored, and this shader-local branch is closed on the
  M1 Max/Metal 32023.883 toolchain.

### PERF-A021 establishes the complete Apple five-metric baseline

- Signed `4dfa1ad3ef` adds a four-row batch-one Q2_K Metal matvec at the
  shared `quant_matmul` owner. Two SIMDgroups reuse each 32-value activation
  fragment across eight output rows per threadgroup; aligned complete cohorts
  admit the specialization, and every tail, offset, and multi-batch shape
  retains the generic path.
- Actual checkpoint gate/down medians move from about **1.07/1.09 ms** to
  **0.455/0.454 ms** over 50-sample A/B/A windows. Actual-file candidate,
  disabled-control, output-tail, and batch-eight parity pass.
- The matched generic control aggregates **8.515065 tok/s** on exact
  `12+256`; the first candidate window reaches **9.156475 tok/s** and an
  independent candidate restart reaches **9.189086 tok/s**, with
  **9.194647 tok/s** best hit and **27.859136 s** mean E2E. The matched gain is
  **7.532647%**, and current aggregate standing is **7.012249%** above
  PERF-A016.
- Five exact reasoning-enabled `128+256` streams establish **22.945718 prompt
  / 9.156675 generation tok/s**, **5.578383 s TTFT**, and **33.426973 s E2E**.
  Every request length-finishes with one reasoning digest.
- Current-source exact `32761+1` passes in the 32,768-token BF16 pool at
  **18.942 prompt tok/s**, **1729.565719 s TTFT**, and **1729.565822 s E2E**.
  Sampled reasoning, arithmetic, thinking-disabled, parsed tool call,
  continuation, and language-only gates pass.
- Codex CLI 0.151.0 then exercised the repaired, hash-pinned machine-local
  profile/catalog pair in one PERF-A021 shell-tool gate. One `pwd` shell call
  returned the workspace, the follow-up Responses request consumed it, and
  final visible output was exact
  `CODEX TOOL READY`; usage was 21,537 input, 413 output, and 379
  reasoning-output tokens. Health, cache flush, foreground shutdown,
  free-listener, memory, and thermal gates pass. The configured 30,000-token
  compaction limit resolves to 29,491. At that point, near-limit transport/
  compaction, concurrent tools, and a reversible workspace edit remained
  client-hardening work.

### Strict Codex scratch editing qualifies; compaction recovery remains historical

- A separate `qwen38-local-hardened` overlay passes real 0.151.0 strict config
  loading, removes model-visible plugin/agent/MCP surfaces, disables unbounded
  retries, bounds tool output, expands the idle timeout to 40 minutes, and
  confines workspace writes. Repository `AGENTS.md` and environment context
  remain visible in one
  observed 32,965-to-21,457-byte prompt render whose exact command was not
  retained.
- A forced 1,000-token compaction run on the immediate medium-reasoning
  predecessor hashes recovers from one malformed patch across two observed
  compaction boundaries, successfully writes a nonce-bearing file, preserves
  that nonce into final output, and exits zero. Its raw JSONL and exact warning
  text were not retained, so this is historical continuation evidence.
- An unmatched low-reasoning clean-edit trial completes one first-attempt patch
  at 4,136/141/70 tokens. Promoting it into the overlay is independently
  confirmed with no reasoning or capacity override at **4,111 input / 118
  output / 47 reasoning-output tokens**, exact `QWEN38 DEFAULT WRITE READY`,
  and the expected file hash; same-value sandbox/approval pins remained. The
  mutable lower user config and rules were not pinned at process start.
  All scratch paths were removed, and verified foreground shutdown returned the
  listener, processes, memory, and thermal state to the recorded idle snapshot.

### Isolated Codex home narrows the mutable-config boundary

- Exact-tag review found that `-p` composes a profile above the ordinary user
  config and loads exec-policy rules. It also found default-on login-shell,
  hooks, image generation, optional tools, and broad shell-environment
  inheritance that the profile hashes did not identify.
- At that stage, the dedicated `qwen38-local-hardened-home` held sibling-relative,
  hash-pinned config/catalog/instructions. It explicitly pinned the shell-only
  surface, low reasoning, 32K/30K Total capacity, core-only filtered shell
  inheritance, no login startup, untrusted exact paths, workspace confinement,
  disabled optional tool/model-visible features, zero retries, and a 40-minute
  idle timeout.
- The exact default-config gate used no profile, `-c`, `--sandbox`, reasoning,
  or capacity override. It emitted one first-attempt `file_change`, returned
  exact `QWEN38 ISOLATED WRITE READY`, and exited zero at **2,843 input / 260
  output / 184 reasoning-output tokens**. Bundle and ordinary-global hashes
  were stable pre/post. Cleanup returned port, processes, scratch, memory, and
  thermal state to idle.

### Trusted-repository unified exec became the Apple Codex handoff at 20:29

- The 20:29 repository trust decision admitted root `AGENTS.md`; an exact
  post-change prompt-input diagnostic contained its heading and C++/CUDA-only
  rule. The qualified tree had no project `.codex`, hook, or rule surface, and
  the dedicated home had no `rules/` directory. Those mutable inputs received
  a fresh absence/hash preflight before interactive work.
- The 20:29 catalog declared `shell_type=unified_exec`, and the instructions
  named `exec_command` plus its `cmd` argument. The 20:29 config/catalog/
  instruction hashes began `a1ce8b8e`, `862339c1`, and `5d59350d`.
- The 20:29 strict scratch gate emitted one first-attempt `file_change`,
  returned exact `QWEN38 ISOLATED WRITE READY`, and exited zero at **2,662
  input / 113 output / 37 reasoning-output tokens**. Stable hashes, health,
  cache flush, artifact cleanup, verified foreground shutdown, free memory,
  and thermal state completed the gate.

### Spawned zsh startup is isolated in the current Apple Codex handoff

- Exact-tag review found that non-login `zsh -c` still resolves a per-user
  `.zshenv` after core environment filtering. The selected config now sets
  `ZDOTDIR=/var/empty` before unified-exec child creation. The target is
  root-owned 0755 and empty, and both system zshenv paths are absent.
- Current config/catalog/instruction hashes begin `9d7842bb`, `862339c1`, and
  `5d59350d`. The final strict task uses explicit stdin EOF, emits one first-
  attempt `file_change`, returns exact `QWEN38 ISOLATED WRITE READY`, and exits
  zero at **2,670 input / 115 output / 39 reasoning-output tokens**. The exact
  34-byte artifact and stable pre/post identities pass.
- A separate repository-CWD prompt diagnostic confirms the trusted root
  `AGENTS.md` and C++/CUDA-only rule. Ignore-independent scans find no auxiliary
  project/home config, hook, rule, override-instruction, or skill surfaces.
  Cleanup returns the listener, process sets, scratch path, memory, and thermal
  state to the recorded idle snapshot.

### Actual-work Codex testing adds a two-minute process-tree watchdog

- A repository multi-file C++ control exposed full Responses-prefix replay
  under `ChunkCache`: 6,697-7,599 prompt tokens were recomputed at roughly
  24-25 tok/s on successive tool turns, taking about 4.7-5.2 minutes each.
- The matched four-slot hybrid `UnifiedRadixCache` candidate reused 6,785,
  7,140, 7,414, and 7,680 prefix tokens while admitting only 295, 219, 210,
  and 200 new tokens. Follow-up prefill fell to roughly 9-20 seconds. A later
  response decoded for more than four minutes before its next tool call, and
  the requested repository repair remained untouched.
- The usability contract now wraps every future non-interactive Qwen/Codex
  attempt with GNU `timeout --signal=INT --kill-after=10s 120s`. Both a cold
  repository-root minimal tool gate and a compact fully specified scratch C++
  repair reached exit `124`. The corrected process-group cutoff left the
  frozen scratch hashes exact, produced no binary, and left no client/compiler
  descendant. Wider actual-work qualification remains open.

### Parser-free structured output completes the supervised two-minute repair gate

- One-off prompt-elision diagnostics reduced Codex 0.151.0 ingress to 100
  tokens, and a compact scratch thread issued a real command and exited zero in
  about 52 seconds. The normal trusted-repository prompt and existing
  interactive configuration remain authoritative; elision flags are confined
  to disposable diagnostics.
- Default xgrammar mask application failed on MPS, and parser-wrapped Outlines
  exposed structural-tag/backend-mask incompatibilities. A controlled restart
  omitted both parsers and successfully served Outlines JSON-schema Responses
  with reasoning effort `none`.
- Four individually process-bounded structured requests generated the header,
  normalization algorithm, and authored-test intent. Host review corrected
  repository wrapper/API drift. Independent review then expanded the authored
  suite to the four required categories. The strict warning-as-error build and
  immutable verifier emitted exact `QWEN38_CPP_MULTI_FILE_GATE=passed`.
- This retains an opt-in supervised subworker for bounded actual work.
  Autonomous multi-file tool ownership and parser-enabled required tools remain
  open qualification targets.

## 2026-09-01

### Native affine-Q5 becomes the active Apple target

- Pinned and downloaded the 18.51 GB text-only affine-Q5/G64 checkpoint at
  revision `2568951b893b6427d0a8eb91cc7f4307154c2f05`; all four model shards
  verify by SHA-256 and the tensor inventory contains no vision tower.
- The existing compiled Qwen3.8 engine accepts five-bit affine tensors and
  reaches **16.322505765 tok/s** on the first complete sampled direct control,
  more than twice the selected GGUF-Q5 small-pool rate.
- The selected Q4 DFlash policy regresses on this target because five-bit M=8
  verification reaches generic MLX QMM. The retained affine-Q5 K-split kernel
  then cuts M=8 to about 228--229 ms and raises direct DFlash throughput
  **19.246%** to **13.721235888 tok/s** with representative parity and the
  full-Q4 regression gate intact. Target-only remains the active route before
  exact-131K serving and Codex qualification.

### Matched five-bit MTP exposes a 31.4 tok/s ceiling and a sampled-overlap gap

- Pinned the 292,018,299-byte five-bit MTP sidecar at revision
  `1faa5a803c972c57cfc1beed606184e726ad3d85` and verified its SHA-256.
- An opt-in eight-token block aligns verification with the native Q5 M=8
  kernel, reaches **31.380317186 tok/s**, mean width **8**, and preserves the
  deterministic target digest. This establishes sufficient execution
  capacity for the requested floor.
- Exact dense-q sampling and residual rejection complete, while the best
  calibrated arm reaches **6.380162135 tok/s** at width **2.285714286**.
  The matched head therefore remains a deterministic execution-cost probe;
  target-only Q5 and its batch-one weight traversal own production work.

### Direct affine-Q5 QMV moves target-only decode to 17.8 tok/s

- A guarded native Metal kernel decodes MLX affine-five-bit packs directly,
  reuses each 16-value activation fragment across four output rows per SIMD
  group, and performs FP32 accumulation. Three representative shapes pass
  parity at maximum absolute error at most **0.03125**; an unsupported K=256
  shape fails closed.
- The selected four-SIMD/four-row/two-pack geometry, fixed scalar FMAs,
  packed four-byte loads, and compile-time K reach **17.823163930 tok/s** in a
  complete sampled `128 / 32 warm / 128 timed` screen. This is **+9.194%**
  over the original **16.322505765 tok/s** affine-Q5 target result and leaves
  **2.176836070 tok/s** to the requested floor.
- Wider per-thread packing, additional result rows, a 16-lane cohort, and
  FP16 local accumulation all regress. A first gate/up multi-stream probe
  stalls on an unsignaled custom-Metal event and is removed. The QMV remains
  opt-in while repeated matched, exact-131K serving, and Codex gates stay
  active.

### An aggregate-4.951-bpw mixed checkpoint opens a smaller Q5-class lane

- Pinned `maglun/Qwen3.8-27B-MLX-Mixed-4.95bpw` at revision
  `596b8067f7cf429007bb668874ffee7e917c8340` and downloaded its four language
  shards while leaving the independently stored vision shard out of the
  language-only runtime snapshot. Every language-shard SHA-256 matches the
  published release manifest.
- The language payload is **16,645,209,088 tensor bytes**, **9.999%** below
  the uniform-five-bit checkpoint. It assigns Q4/G64 to embeddings, LM head,
  MLP gate/up, and attention Q/K, and Q5/G64 to MLP down, mixer projections,
  and attention V/O, for **4.9510 aggregate BPW**.
- The native engine's complete required-key and packed-shape audit passes:
  1,655 language tensors, 162 Q4 plus 240 Q5 affine matrices, zero missing or
  extra language tensors, and zero bit-width inference errors. A fresh Metal
  session is the remaining boundary before direct throughput measurement.

### Shader attribution selects a stock-exact Q4 cohort and reaches 19.24 tok/s

- A selected-A100 Metal System Trace maps 48,156 target-process GPU PCs.
  Stock affine-Q4 QMV owns **43.546%**, custom affine-Q5 QMV owns
  **48.613%**, and all QMV owns **92.159%** of mapped execution.
- A new Q4/G64 batch-one kernel preserves official MLX v0.32.2's helper and
  FP32 expression structure while doubling the output cohort from eight to
  sixteen rows. K/N `512/64`, `5120/128`, and `5120/17408` are bit-exact
  against stock MLX. A faster hand-inlined form is rejected because small
  BF16 differences change the seeded target trajectory.
- Two independent balanced five-versus-five windows improve paired A100-only
  **19.113936088 -> 19.241981332 tok/s**, **+0.669905%**, with the canonical
  digest and last token in all 20 qualified runs. The direct gap is now
  **0.758018668 tok/s / 3.939400%**; exact 131K serving and Codex `xhigh`
  remain gated on clearing 20 with margin.

### Fixed-memory native attention converts the mixed-Q5 exact 32K crash into a pass

- On-demand token-embedding dequantization first removes **1.702 GiB** of eager
  BF16 duplication. A follow-up contract repair admits the native sampler's
  valid MLX `uint32` feedback alongside prompt `int32`, with both forms
  bit-exact against gather from the complete dequantized table.
- A native Q8/C64 online-softmax Metal kernel consumes the existing contiguous
  BF16 GQA cache with fixed threadgroup storage. Five focused shapes, including
  prefix 8,191 and a 1,024-token query chunk, stay within maximum absolute
  error `0.000244141` and contain no non-finite values.
- Always-on dispatch changes the seeded short-prompt trajectory, so the
  selected policy keeps stock MLX SDPA through 8,192 active tokens. It restores
  the canonical digest at **19.744722973 tok/s** and uses fixed-memory
  attention only in the unsafe long-history region.
- Independent clean exact requests complete at **107.023 prompt tok/s** for
  `8192+16` and **87.807667 prompt tok/s** for `32768+16`. The latter takes
  **373.179259 s** and remains healthy where A131 plus stock SDPA fails near
  180 seconds. Exact 131K now moves to persistent BF16 KV and prompt-snapshot
  residency; the decode floor and Codex `xhigh` remain open.

### Spotlight isolation closes the full-reserve BF16 cache and selects affine Q8

- Spotlight indexing is disabled on `/`, `/System/Volumes/Data`, and
  `/System/Volumes/Preboot`; active shared metadata workers drained before the
  capacity launch, removing the earlier host-contention confounder.
- A metadata-only append-only snapshot retains only logical offset/length while
  the current cache owns prompt storage. Focused growth, rollback, overwrite,
  and five fixed-attention parity cases pass. A same-dylib switch pair is
  neutral at **19.685169420 -> 19.672600916 tok/s**, canonical in both arms.
- Reserving all 131,072 BF16 cache slots on the first long chunk fails exact
  `32768+16` after about 16 seconds with Metal insufficient memory. Scheduler
  RSS is **18,225,056 KiB** before the request, final BF16 K/V is 8 GiB, and
  MLX reports an approximately 25 GiB recommended working set; the final
  representation does not fit safely even after snapshot ownership is removed.
- PERF-A134 therefore owns exact 131K with affine-Q8/G64 K/V (approximately
  4.25 GiB including metadata), the retained append-only snapshot invariant,
  A130's fixed-memory prefill, and split-history long decode. The >=20 tok/s,
  sampled behavior, and Responses/Codex `xhigh` gates remain open.

### 03:07–06:18 — DSpark-v2 crosses 150 tok/s and becomes the Windows default

- The trained Qwen3.8-27B DSpark-v2 draft was integrated with online-FP8
  weights, gamma-seven linear proposals, eight-row target verification, Triton
  draft attention, FP8 target/draft KV, five FP32 Mamba slots, one admitted
  request, 4,096-token prefill chunks, and exact 200K target/draft pools.
- Trace attribution isolated roughly 1.67 ms between the captured target and
  draft graphs. Static verification had already reserved all eight physical
  draft-cache rows, making it exact to project/write every target-hidden row in
  the target graph and leave rejected rows unreachable. The target-to-draft gap
  fell to about 1.10 ms and full cycle wall moved 18.248 -> 17.98 ms without
  deleting device computation or changing deterministic acceptance.
- Reduced 32K ordinary sampling first averaged **151.694 tok/s**. The first
  exact-200K full-pool window averaged **155.961 tok/s**. After promoting the
  process-scoped switch, a literal argument-free restart averaged
  **162.500 tok/s** over `[151.139,165.951,151.000,191.357,153.054]`; every
  default-process sample individually reached 150.
- The argument-free launch repeated exact `199000+16`, coherent `703`, exactly
  one parsed multiply call plus clean continuation, thinking-disabled `READY`,
  image/audio false, native acceptance, standalone OpenCode2, and Codex 0.152.0
  Code Mode with exactly one successful command and exact `CODEX TOOL READY`.
  A first default Codex turn transported the output correctly but called the
  nonempty status empty in private narration; the bounded retained retry read
  it accurately. Both left Tombstead unchanged and no retained process.
- `SGLANG_DSPARK_STATIC_GRAPH_KV_COMMIT` is now a reversible DSpark-only
  launcher default. NEXTN, sparse/truncated DSpark experiments, tree/SWOR, and
  the independent native-controller lane remain preserved but unselected.

## 2026-09-02

### Official coding settings align the Apple qualification contract

- The pinned upstream Qwen3.8 sources select thinking and preserved thinking,
  `reasoning_effort=xhigh`, temperature **1.0**, top-p **0.95**, top-k
  **20**, min-p **0.0**, presence **0.0**, and repetition **1.0**. Native
  context is 262,144, so the 131,072-token lane uses no YaRN.
- The direct native sampler already implements that target distribution.
  Presence **1.5** is retained only for Qwen's separate non-thinking profile.
  Qwen publishes no recommended internal MTP proposal temperature.

### Exact shared Q4 head loads clear 20 tok/s directly

- A149/A150 first reduce standard-MTP M=2 verification from generic
  **15.204721403** to **19.795886878 tok/s** by sharing Q5 and fused-Q4 MLP
  loads across both verifier rows.
- A163 adds the remaining ordinary Q4 vocabulary head while preserving each
  row's scalar arithmetic order. The exact production micro improves
  **4.195437500 -> 3.440243000 ms**; the faster A162 vector form is rejected
  for 10,703 mismatches and changed acceptance.
- Two independent five-sample A163 windows average **20.109963672** and
  **20.107849480 tok/s**. Every sample clears 20 and preserves digest
  `6bd687fb75c4f5a9`, last token 1467, 75 refills, and width
  **1.706666667**. Signed `94ca4ff7fa` retains the opt-in win.
- The short direct floor is cleared. The 6,237-token actual-work window, exact
  served 131K composition, and a real Responses/Codex `xhigh` coding turn
  remain the active qualification sequence.

### Actual-work MTP collapse localizes to missing committed history

- A163 target-only reproduces **19.057906040 tok/s** after 6,237 prompt
  tokens. The official five-bit MTP reaches only **7.839819833 tok/s** and
  width **1.032258065**; the optimized Q4 MTP falls from
  **26.491661416 tok/s** short to **7.516953814 tok/s** and width
  **1.003921569** long.
- Native standard MTP clears its attention cache before every proposal and
  never constructs shifted prompt history. Official MTPLX keeps prompt hidden
  rows paired with following tokens and restores then appends only committed
  decode prefixes. PERF-A164 moves this invariant into the native MTP state
  owner before another checkpoint or kernel is ranked.

### Committed MTP history clears the actual-work decode floor

- Signed A164 commit `b92c21d69d` adds opt-in prompt-shifted and
  accepted-prefix history at the standard-MTP state owner. It streams prompt
  alignment in bounded chunks, restores speculative KV before commit, and
  keeps MTP cache length exactly one token behind the target.
- The optimized Q4 MTP improves from **7.516953814** empty-cache tok/s to a
  clean five-sample mean **23.821754359 tok/s** after 6,237 prompt tokens.
  All runs retain 131 refills, width **1.961832061**, digest
  `d468c7e1b0d274b3`, and last token 20.
- A post-commit rebuild reproduces **23.819226864 / 23.811460300 tok/s**.
  One FileProvider/indexing-contended sample is retained at
  **19.724760551 tok/s** with identical acceptance/output; the user accepted
  it as labeled host noise. Work now moves to exact served 131K and a real
  Responses/Codex `xhigh` coding turn.

## 2026-09-08

### 18:57-19:26 - install and benchmark NVIDIA's stock mixed-precision checkpoint

- Downloaded and hash-verified immutable NVIDIA revision
  `fed99d815f4e8c7c616dd3dd6780076e26d5fb61` into a separate local artifact.
  The existing Windows launcher and dependencies support it without source
  changes.
- Real-200K target-only serving passes sampled reasoning, arithmetic, tools,
  continuation, non-thinking output, and exact `199000+16`. Five sampled
  `6213+512` requests average **64.064 tok/s**; the identical original
  RadixArk target-only control averages **63.996 tok/s**, a practical tie.
- The earlier DSpark control was measured under substantially higher desktop
  residency and is excluded from this checkpoint-only comparison. No
  production default or capacity gate changes; NVIDIA speculative/client
  promotion and published accuracy remain unqualified. All task-owned
  servers are stopped, with port 30000 free and ordinary display residency
  restored. Full raw results are retained in
  `benchmark/windows/nvidia_qwen38_20260908.json`.

## 2026-09-12

### 12:51-13:07 - DeepSeek V4.1 cache ideas fail the no-training gate

- Qwen3.8's 48 GDN plus 16 global-attention topology leaves no persistent SWA
  state to remove. Retrofitting the causal encoder/decoder split or CSA2
  cross-layer K/V reuse would change learned residual flow or substitute
  distinct learned projections, so neither preserves the checkpoint without
  training.
- Current-source native target NVFP4 reached readiness when speculative target
  verification used the prefill backend, recovered the expected cache memory,
  then hallucinated unrelated conversations for a full 256-token deterministic
  arithmetic response instead of deriving `703`.
- A standalone native SM120 TurboQuant35 path implemented the mixed-bit MSE and
  residual codec, direct packed attention, and graph replay at the production
  shape. It saves 53.125% of FP8 cache bytes, but the independent encoder parity
  gate was followed by **0.199575** relative-L2 attention error and
  **11,540.973 us** at 199K, versus **3,310.600 us** for matched FP8 and the
  established **271.584 us** XQA FP8 authority.
- The codec remains an isolated benchmark. Serving cache layout and launcher
  defaults stay unchanged; the MTP-bearing AttnNVFP4 target and explicit
  DSpark-v2 draft remain selected.

## 2026-09-16 — screen DavidAU TURBO Fable Cold Fusion NVFP4

- Downloaded and verified the author's BF16 checkpoint and MTP GGUF. Standard
  local ModelOpt calibration exceeded the machine's memory-commit margin;
  no locally converted checkpoint was exported and no BF16 benchmark ran.
- At the user's direction, downloaded hyssra's existing NVFP4 W4A4 conversion.
  Target-only SGLang passes basic behavior and exact 200K capacity; five
  `6213+512` official-thinking samples average 61.9904 generation TPS.
- Two public Babouin coding responses fail: a non-Rust answer and a Rust
  answer with compiler-confirmed missing traits. The user closed the evaluation.
  Production defaults remain selected, all task processes are stopped, and
  downloaded artifacts and evidence are retained.

## 2026-09-19 — native-Windows DiffusionGemma compatibility trial

- Downloaded NVIDIA's pinned NVFP4 checkpoint and verified both weight shards
  and the tokenizer. Added only C++ program source for launch, loading, expert
  routing and process-local SGLang integration through the installed libraries.
- Expert parity and full-model text generation pass, along with health,
  chat/SSE, reasoning usage, multi-chunk requests, parsed tools and continuation.
  The local trial admits one request, 2048 total tokens and 1024 output tokens.
- Eager per-expert execution and buffered output remain compatibility limits.
  Individual functional samples are recorded; no performance branch or
  production promotion was started. Qwen defaults are unchanged, test processes
  are stopped, and the native executable, checkpoint and evidence are retained.

## 2026-09-20 — Mac DiffusionGemma interactive setup

- Established an isolated MLX-VLM runtime for the pinned MLX-community 4-bit
  checkpoint on the 32-GiB M1 Max, with a C++23 client measuring actual visible
  TTFT and whole-request throughput. No Python source was authored.
- Selected explicit 64-token blocks and confidence-threshold sampling. Two
  independent greedy windows average 39.88/39.90 tok/s and 1.40/1.39-second
  TTFT; one sampled five-seed window averages 38.44 tok/s and 1.88-second TTFT.
  Complete answers, arithmetic, reasoning, tool round trips, and 3935-token
  retrieval pass. Context capacity beyond this workload remains unmeasured.
- The user's `.cache` cron job removed the initial setup. Restored only this
  task's pinned files and recovered receipts under persistent `.local/share`.
  The user accepted the result, closed further benchmarking, and authorized
  publication. The checkpoint, isolated environment, and restart commands
  remain available; the task server is stopped at the publication handoff.

## Supersession map

Use these results when older “final” checkpoints conflict:

| Topic | Current value | Supersedes |
|---|---|---|
| Fixed `6213/512` | `171.263 tok/s` safe five-run mean | `167.776`, `162.726`, `159.973`, `156.968`, `135.167`, `86.016` |
| Real sampled `6213/512` | **`162.500 tok/s`** argument-free DSpark-v2 five-run mean; independent full-pool mean **155.961** | `122.712`, `121.075`, `117.794`, `110.750`, `98.126`, `96.110` |
| Near-limit linear record | `3190.815` prompt, `121.616` generation tok/s on the retained NEXTN/selective-checkpoint control | `3078.058/114.617`, `3048.086/112.499`, `3016.444/112.355`, `2838.980/107.253` |
| Current DSpark-v2 exact-capacity proof | Exact `199000+16` at `3118.215` prompt tok/s, `63.818572 s` TTFT, and `64.236571 s` E2E | explicit-switch proof `3118.323`, `63.816352 s`, `64.233185 s` |
| Production capacity | `200000` context and token pools | rejected `232000` operating-margin experiment |
| Speculation geometry | Trained DSpark-v2, one proposal step, block size 7 / eight verify rows | NEXTN 2 steps / 3 draft tokens; 3 steps / 4 draft tokens |
| Target verification | TRT-LLM MHA/XQA plus static graph-side draft-KV commit | eager accepted-row commit; FlashInfer-prefill verify route |
| Draft proposal | Captured DSpark proposal/sampling graph with Triton attention | separate captured/eager NEXTN draft extension |
| Draft-q source | Trained rank-256 DSpark-v2 Markov distribution with ordinary rejection | NEXTN top-k-one direct one-hot q and top-k-20 aligned proposals |
| Tree mode | opt-in exact target-only/SWOR infrastructure; linear production default | current-q M8/M12/depth/topology-only candidates |
