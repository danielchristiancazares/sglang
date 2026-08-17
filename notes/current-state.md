# Current state

**Reconciled through:** [`experiment-log.md`](experiment-log.md), 2026-08-16
23:55 PDT.

**Qualified source line:** commit `9681850bed660b9079ee1aee906cda819603da7a`
(`Add exact SWOR tree verification and topology analysis`), with the final
232K rejection and 200K restoration recorded after that commit. Launcher
defaults were restored to 200K and rechecked during the notes migration.

## Qualified production configuration

The accepted configuration is native-Windows SGLang serving
`C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk` on the RTX 5090. It
provides a real 200,000-token target/draft pool, preserved reasoning, parsed
tools, and a language-only model surface.

| Area | Selected value |
|---|---|
| Endpoint | `http://127.0.0.1:30000/v1`, model `qwen3.8-27b` |
| Capacity | Context `200000`; total-token pool `200000`; one running request |
| Model surface | `--language-model-only`; Qwen3 reasoning parser; Qwen3 Coder tool parser |
| Target attention | FlashInfer prefill; TRT-LLM MHA/XQA decode and target verification |
| Draft attention | TRT-LLM MHA/XQA; captured draft decode and `DRAFT_EXTEND_V2` graphs |
| Linear attention | Triton GDN with ReplaySSM speculative-state handling |
| Speculation | NEXTN linear rejection sampling; 2 steps; 3 draft tokens; EAGLE top-k 1 |
| Proposal distribution | Aligned draft top-k 20 inside the single multi-step CUDA graph |
| KV | Checkpoint-selected target KV; FP8 E4M3 draft KV; page size 64 |
| Sampling | FlashInfer, including native-Windows CUDA renormalization on the speculative path |
| Prefill | 4096-token chunks |
| Mamba | 4 slots; `extra_buffer_lazy`; FP32 state |
| GEMM tuning | FlashInfer CUTLASS FP4; autotune enabled; FP8 GEMM autotune skipped |
| Compile/graphs | Torch compile mode `default`; batch-one full decode graphs |
| Scheduling/streaming | Scheduler receive interval 4; stream interval 4; incremental output |
| Workspace | 128 MiB FlashInfer workspace, the measured functional floor |
| Draft quantization | Checkpoint-native BF16 MTP; explicit online FP8/MXFP8/NVFP4 inactive |
| Experimental controls | Adaptive depth, SWOR/tree topology, path/overlap oracles, and fixed-acceptance simulation inactive |

These values match the defaults in
[`../scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1`](../scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1)
at reconciliation time. The launcher and freshly resolved arguments remain the
executable source of truth.

## Qualified measurements

| Gate | Result |
|---|---|
| Real sampled, exact `6213/512` | **122.712 tok/s** ten-run mean; **122.371** median; **137.074** peak |
| Safe fixed work, exact `6213/512`, accepted length 3 | **171.263 tok/s** five-run mean; all runs retained the established digest |
| Native acceptance | Five-probe mean **2.318174** emitted/accepted tokens per verification |
| Near-limit capacity, exact `199000+16` | **2608.263 prompt tok/s**, **102.358 generation tok/s**, `199016` total |
| Behavior | Coherent preserved thinking; correct `703`; exactly one `multiply({"a":37,"b":19})` tool call |
| Surface | Image and audio understanding reported false |
| Final production relaunch | All three speculative graphs captured; **1.84 GiB** reported headroom |

The real promotion used seed `783025237` to make matched investigations easier,
while ordinary production semantics retain stochastic rejection sampling. The
single multi-step aligned-q graph recovered the fixed-work collapse caused by
Python execution between draft depths and became the measured winner.

## Active tree correctness hold

A deterministic non-front acceptance reproducer found that the unified hybrid
pool passed virtual target-KV slot ids directly to a physical backing pool. The
multi-layer EAGLE caller also allowed tree-path front compaction to be skipped
while draft extend indexed tokens and hidden rows as a compact front block.

The recorded M8/M12/M16 servers used `enable_unified_memory=False` and the
single-layer finalizer was already active. The unified-pool reproducer therefore
establishes a real optional-path defect, not proof that those exact requests
were corrupted. Their production hold remains because a full current-config
cross-cycle state comparison has not established target KV, recurrent state,
next-draft state, reclamation, and next-cycle proposal parity together.

The pending repair installs physical full-KV translation for relocation,
preserves MLA's separate dense kernel address space, and makes accepted-path
compaction mandatory for every top-k tree worker. Factory-created MHA/MLA tests
and a captured four-cycle serial-path comparison now cover target K/V,
token/hidden compaction, terminal next-draft state, rejected-slot reclamation,
and virtual-id reuse. The combined accepted-path, composite-graph, and GDN CUDA
suite passes eight tests plus two subtests.

Every earlier tree throughput result remains mechanism-only. The qualified
linear **122.712 tok/s** result is the production comparison authority until a
corrected full-model non-front path comparison and all ordinary promotion gates
pass.

A fresh production-linear comparison from correctness commit `3f276e8acda4`
retained all tree/device-cycle controls off. Its first five-run window averaged
**112.253 tok/s** while warming through an 84.130 tok/s first request; the
second independent window averaged **124.775 tok/s**. The ten-run combined mean
was **118.514 tok/s**, and five native acceptance probes averaged **2.204748**.
This confirms the top-k-one comparison path remains within the established
performance range while preserving the startup sample and lower stochastic
acceptance as real evidence.

An exact-q device-resident linear cycle was then qualified functionally and
closed for throughput. The dense-race form averaged **122.576 tok/s**; an
explicit-seed FlashInfer categorical refinement averaged **120.075 tok/s**
despite **2.277991** emitted tokens/cycle. Its normalized 21.132 ms/cycle
remained above the ordinary path's 20.771 ms/cycle. The architecture stays
opt-in, SWOR is rejected on raw-composite RNG grounds, and production defaults
remain unchanged.

## Final handoff

The exhaustive NVFP4 optimization goal was explicitly marked complete at
19:06 PDT. The accepted final action restored both launcher defaults from a
232K experiment to `ContextLength=200000` and `MaxTotalTokens=200000`, then
captured a fresh unsimulated production server.

At that timestamp, the hidden PowerShell parent was PID `30688`, the endpoint
was ready on port 30000, graph capture took 43.45 seconds for target verify,
1.39 seconds for draft decode, and 1.23 seconds for draft extend, and reported
headroom was 1.84 GiB. Every PID, listener, log path, and runtime health fact is
a snapshot; establish live state from the process tree, port, GPU, logs, and
endpoint before a lifecycle action.

The historical 200 tok/s real-sampled target remains evidence about the
performance frontier. The completed goal supplies no standing authorization
for another GPU experiment. A future branch starts from an explicit request or
a fresh measured gap.

## Behavior and capacity invariants

Every promoted candidate retains all of these:

- real `200000` context and token-pool capacity, including exact total `199016`
  when memory layout, graph coverage, cache dtype, workspace, or sampling
  residency changes;
- Qwen's selected sampled reasoning profile: temperature `1.0`, top-p `0.95`,
  top-k `20`, and presence penalty `1.5`;
- preserved `reasoning_content`, coherent thinking, and ordinary completion
  behavior;
- arithmetic probe result `703` for `37 * 19`;
- exactly one parsed `multiply({"a":37,"b":19})` call with
  `finish_reason=tool_calls`;
- image and audio understanding disabled;
- an unsimulated launcher-default relaunch with target verify, draft decode,
  and draft extend graphs captured;
- standalone OpenCode2 integration using a fixed provider/workload shape;
- protected CUDA compatibility headers outside the edit boundary.

## Native-Windows implementation retained in the source line

The production and experimental commits retain several durable Windows
capabilities:

- CUDA 13.3/MSVC JIT initialization with the conforming preprocessor and a
  bounded two-job compiler pool;
- native C++/CUDA SiLU-and-multiply, standard RMSNorm, Gemma RMSNorm, full-
  attention sigmoid-multiply, and fixed-chain metadata paths behind narrow
  native-Windows dispatch gates;
- FlashInfer CUDA top-k/top-p renormalization on the speculative target path;
- aligned draft proposal transforms and exact-q capture inside the single
  multi-step CUDA graph;
- Windows registration and isolated correctness coverage for optional online
  FP8, MXFP8, dense NVFP4, and pure ModelOpt FP4 mechanisms;
- an exact GPU target-only/SWOR tree verifier, sparse shared-memory residual
  path, low-rank tree-aware GDN verification, accepted-path recurrent/conv
  commit, path and overlap oracles, and offline topology analysis tools.

Production uses the linear path. Tree/SWOR machinery and online draft
quantizers remain opt-in infrastructure with preserved tests and evidence.

## Closed experimental frontier

The following branches are closed for the current checkpoint, proposal
distribution, and cost topology:

- adaptive depth over two and three steps;
- static one-step and three-step linear speculation;
- reusable fused metadata output buffers whose asynchronous lifetime changed
  real rejection-path scheduling;
- draft proposal top-k 8;
- full online FP8, MXFP8, and dense online NVFP4 MTP weights as performance
  defaults;
- stock Gittensor ModelOpt FP4 as the production checkpoint;
- current-q target-only and SWOR trees, fixed-width M8/M12 breadth, depth-only
  expansion, scalar depth discount, and measured topology rearrangement;
- a 232K production pool, which completed exact `231000+16` capacity yet fell
  to 98 MiB free VRAM before cache flush.

The tree implementation remains a future route only when proposal overlap or
per-depth draft cost changes enough to alter the measured economics. The
recorded topology optimizer reached fewer than 4.1 expected outputs even under
optimistic assumptions, while measured M12 SWOR throughput was 84.713 tok/s
against the 122.712 tok/s linear baseline.

## Workspace and provenance boundaries

- Preserve every pre-existing modified and untracked path as user-owned work.
- Treat `sglang.bundle` as unrelated user-owned material.
- Keep the original FlashInfer checkout and clean Windows 0.6.17 port as
  separate provenance lines.
- Keep RadixArk and Gittensor source checkpoints immutable; conversions or
  hybrids belong at separate paths with checksums and provenance.
- Preserve the protected CUDA compatibility headers. Their recorded SHA-256 is
  `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.
- Use C++/CUDA for new performance hot paths, with Python limited to bindings,
  dispatch, configuration, tests, and launch integration.
- Run one server, CUDA test/JIT build, compiler tree, or GPU benchmark at a
  time, and target lifecycle actions by verified process ancestry.
