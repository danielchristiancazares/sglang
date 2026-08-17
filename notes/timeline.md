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

## Supersession map

Use these results when older “final” checkpoints conflict:

| Topic | Current value | Supersedes |
|---|---|---|
| Fixed `6213/512` | `171.263 tok/s` safe five-run mean | `167.776`, `162.726`, `159.973`, `156.968`, `135.167`, `86.016` |
| Real sampled `6213/512` | `122.712 tok/s` ten-run mean | `121.075`, `117.794`, `110.750`, `98.126`, `96.110` |
| Near-limit `199000/16` | `2608.263` prompt, `102.358` generation tok/s | `2570.356`, `2429.153`, `2423.812`, `2200.563` prompt results |
| Production capacity | `200000` context and token pools | rejected `232000` operating-margin experiment |
| Speculation geometry | 2 steps / 3 draft tokens | 3 steps / 4 draft tokens |
| Target verification | TRT-LLM MHA/XQA | FlashInfer-prefill verify route |
| Draft extension | CUDA graph captured | eager draft extension |
| Draft-q alignment | top-k 20 inside the single multi-step CUDA graph | eager/per-depth aligned proposal path |
| Tree mode | opt-in exact target-only/SWOR infrastructure; linear production default | current-q M8/M12/depth/topology-only candidates |
