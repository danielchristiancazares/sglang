# Condensed experiment timeline

This is an orientation map, not the evidence ledger. Read the matching region
of [`NOTES.md`](../NOTES.md) when exact samples, logs, code changes, or process
state matter.

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

## Supersession map

Use these results when older “final” checkpoints conflict:

| Topic | Current value | Supersedes |
|---|---|---|
| Fixed `6213/512` | `159.973 tok/s` | `156.968`, `135.167`, `133.232`, `126.615`, `86.016` |
| Real sampled `6213/512` | `117.794 tok/s` ten-run mean | `110.750`, `98.126`, `96.110`, `94.319` |
| Near-limit `199000/16` | `2608.263` prompt, `102.358` generation tok/s | `2570.356`, `2429.153`, `2423.812`, `2200.563` prompt results |
| Speculation geometry | 2 steps / 3 draft tokens | 3 steps / 4 draft tokens |
| Target verification | TRT-LLM MHA/XQA | FlashInfer-prefill verify route |
| Draft extension | CUDA graph captured | eager draft extension |
| Draft-q top-k alignment | disabled by default | rejected opt-in top-k 20 candidate |
