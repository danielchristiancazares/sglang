# Current state

**Reconciled through:** [`experiment-log.md`](experiment-log.md), 2026-08-20
22:31 PDT.

**Qualified production source line:** commit
`7f5af878da7b8dc43063f31e554dfc69cee5d510`
(`perf: retain large-extend FlashInfer tactics`). The selected optimization is
expert-opt-in; the base RadixArk, 4096-token chunk, and exact 200K launcher
defaults were requalified unchanged after the commit's source was tested.

**Latest retained selective-performance source:** commit
`5ea3b734b0` (`perf: fuse eager Windows SwiGLU NVFP4`). It includes the
bit-exact fused Gemma residual norm from `e09e43171d` and adds an eager-only
SwiGLU-to-NVFP4 producer. The explicit selective checkpoint/chunk-7680 profile
passed exact capacity and deterministic-output gates. Production-default
behavior/client requalification is deferred until the active target is
cleared, so the qualified production source line above remains unchanged.

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
| Selective large-EXTEND tuning | Off by default; `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1` on the explicit AttnNVFP4/chunk-7680 profile |
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
| Primary exact `199000+16` record | **3048.086 prompt / 112.499 generation tok/s**, TTFT **65.286869 s**, E2E **65.420204 s** |
| Selected-cache exact prompt window | **3047.309 tok/s** five-run mean; every request exact `199016` |
| Selected-cache long generation | **118.389 tok/s** three-run mean at exact `199000+512` |
| Current eager-fusion prompt window | **2987.275 tok/s** five-run mean in the drifted current environment; **0.914%** above the adjacent PERF-028 arm with the established digest restored |
| Current eager-fusion long support | **3001.344 prompt / 115.225 generation tok/s** over three exact `199000+512` requests; established digest restored |
| Behavior | Coherent preserved thinking; correct `703`; exactly one `multiply({"a":37,"b":19})` tool call |
| Surface | Image and audio understanding reported false |
| Final production relaunch | Defaults unchanged; exact `199016`, all three graphs, OpenCode2, and **2,222 MiB** post-flush free |

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

Signed commit `3f276e8acda4` installs physical full-KV translation for relocation,
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

## Active performance handoff

The user designated the exact `199000+16` request in the real 200K pools as the
primary performance scoreboard. PERF-024 sets the record on the selective
target-NVFP4 checkpoint at **3048.086 prompt / 112.499 generation tok/s**,
with **65.286869 s TTFT**, **65.420204 s** end to end, exact `199016` tokens,
and `finish_reason=length`. Root [`../BENCHMARK.md`](../BENCHMARK.md) is the
compact authority.

The next target is **3100 prompt / 120 generation tok/s**, **<=64.20 s TTFT**,
and **<=64.35 s** end to end in one eligible exact request. The timing limits
are mathematically tied to the two throughput thresholds.

PERF-028 and PERF-027 are now retained additive changes on the active source
line. PERF-028 fuses residual-add plus Gemma norm and improved adjacent exact
long generation by **1.205%**. PERF-027 fuses eager Qwen SwiGLU directly into
the NVFP4 tuple consumed by `down_proj`; its repaired exact window improved
prompt by **0.914%** and TTFT by **0.606649 s** versus PERF-028 while restoring
both deterministic digests. PERF-027 deliberately bypasses
`torch.compiler.is_compiling()`: Inductor removes the eager BF16 SiLU rounding
boundary, so the compiled M3 target graph retains its former function until a
separately exact compiled-semantics producer qualifies.

The winning selective profile remains `AttnNVFP4`, chunk 7680, M3, and the
bit-exact Windows Gemma residual-norm direct-output path. PERF-024 additionally
runs an ordinary 16,384-token target EXTEND pass under the existing
`SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1` expert opt-in. An independent retune
produced exact prompt samples
`3051.345, 3048.538, 3048.086, 3042.488, 3044.105`, mean **3046.912**.

FlashInfer 0.6.17 stores file hits in process-global `_file_configs`, which
later draft autotune contexts replace. The retained adapter promotes only
target EXTEND file hits actually exercised by the pass into the runner-keyed
process cache. A clean relaunch promoted 110 entries from the selected
20,928-byte cache, SHA-256
`8219484FA86EBB0E6DDA54F2D15447DBC502EBCEA9007B3E1BB917B9001F9ADF`,
without re-profiling. Its five exact prompts averaged **3047.309 tok/s**.

Long generation is the stable generation authority because exact-16 measures
only 15 post-first-token intervals. Three selected-cache `199000+512` requests
averaged **3047.754 prompt / 118.389 generation tok/s**, while five real
sampled `6213/512` requests averaged **126.252 tok/s** and five native probes
averaged **2.217256** accepted tokens per verify. All deterministic windows
retained their selected-tactic digests.

Cache-only and dummy-only controls both returned to about 3009 prompt tok/s
and the baseline digest, proving tactics rather than stale recurrent/KV state
caused the gain. Profiling afresh on every launch was rejected: it retained a
3043.747 exact prompt mean but selected a long-generation tactic family that
averaged only 101.162 tok/s. Keep the selected cache and requalify any new
tactic selection.

The branch began from a five-run current-source control of **2871.358 prompt /
90.459 short generation tok/s** and an adjacent A2 control of
**2926.303/92.782**. The selected-cache prompt mean is 4.136% above A2 and
1.023% above the prior 3016.444 record. The benchmark client fails before
sending a request unless prompt calibration equals the requested count exactly.

The qualified RadixArk production baseline remains **2608.263 prompt tok/s**
and **102.358 generation tok/s** on the same exact workload. A current
launcher-default gate after PERF-024 reached **2648.283/88.187**, exact
capacity, arithmetic/tools, model surface, standalone OpenCode2, and
2,222 MiB post-flush free. Its short generation result is retained as
cycle-quantized gate evidence, not a replacement baseline. The selective
checkpoint remains the performance record profile rather than the production
default.

The earlier 200-TPS short-context objective and 215-TPS geometry funding floor
remain historical diagnostic context. Production defaults and the qualified
122.712 TPS `6213/512` line remain unchanged.

Asynchronous CUDA-event timing collected 1,471 transition records over two
independent real-sampling windows. The best repeatable graph-tail opportunity
was target-to-draft-extend at conservative p10 **0.658355 ms**, below the
required 0.75 ms. Graph-tail composition is closed.

The M3 target trace contains 61 exact graph-2 replays at **15.322 ms mean** and
**14.661 ms median**. Full target-start-to-target-start cycles averaged
**19.446 ms**. Primary GEMMs consume **13.086 ms aggregate** per replay,
**12.360 ms** on the terminal stream, and **11.821 ms** of exclusive observed
wall. Exact mathematical shape attribution ranks:

1. NVFP4 MLP gate/up, `M=3,N=34816,K=5120`: **4.211 ms** terminal-stream;
2. FP8 GDN qkvz, `3x16384x5120`: **2.851 ms** terminal-stream, **1.675 ms**
   exclusive wall because alternate-stream work overlaps it;
3. NVFP4 MLP down, `3x5120x17408`: **2.328 ms**;
4. FP8 output projections, `3x5120x6144`: **1.484 ms**;
5. FP8 full-attention qkv, `3x8192x5120`: **0.946 ms**;
6. NVFP4 lm-head, `3x248320x5120`: **0.539 ms**.

The BF16 GDN `in_proj_ba` consumes 0.726 ms of aggregate device work and has
only 0.000081 ms of exclusive observed wall at M3, so it is already hidden by
the qkvz stream. The draft-decode and draft-extend graphs span **1.217 ms** and
**1.063 ms**; together with inter-graph scheduling they account for the other
roughly 4.124 ms of the full cycle.

Two exact-shape, distinct-weight CUDA-graph windows funded selective conversion
of the exposed FP8 projections to NVFP4. Their overlap-adjusted cycle
projections were **1.976456 ms** and **1.865227 ms**. The derived checkpoint
then reduced the measured M3 cycle from 19.446 ms to **17.315 ms**, passed exact
`199000+16`, and established the earlier **2838.980/107.253 tok/s** scoreboard
record that the direct-output profile later superseded.

The funded plain M4 K+1 retest is now closed on current source. M4 raised
accepted length from **2.245614** to **2.327273** (+3.636%) while increasing
the matched full cycle from **16.058328** to **18.419190 ms** (+14.702%).
Measured projection fell **139.841 -> 126.350 tok/s**. Warmed exact-200K
prompt means were indistinguishable, and generation means overlapped their
8-11% variance. M3 remains selected.

FlashInfer paged-only prefill is also closed. Against the restored default,
exact-200K prompt changed **2789.036 -> 2785.260 tok/s** and 512-token
generation changed **106.467 -> 104.117 tok/s**. The default ragged-current
plus paged-prefix merge remains selected.

Chunk 7680 is selective-profile-only. Applying it to the launcher's default
base RadixArk checkpoint reduced exact prompt throughput to **2226.770 tok/s**
and left only 200 MiB free before follow-up probes. The production launcher
therefore remains at chunk 4096.

Current measured geometries fail the path-length oracle before proposal
quality is considered. M3's depth-two maximum is 154.270 TPS at mean cycle
cost. M8, corrected M12, and M16 depth-four best-sample impossible ceilings are
185.782, 179.547, and 166.666 TPS. Width/topology implementation remains
unfunded.

Branch-exact p/q capture, branch-local presence/frequency/repetition state,
startup worker/compile provenance, and deterministic replay tooling are now in
the worktree. The live six-cycle artifact is deliberately marked
`capture_scope=selected_tree`; later states have incomplete support. It can
replay the observed membership and cannot qualify aligned, irregular,
calibrated, SWOR, confidence-gated, or target-aware counterfactuals. The replay
gate requires complete lattice coverage and requires every geometry candidate's
conservative lower TPS to strictly clear the measured frontier's best-case
upper TPS before applying the 215-TPS funding floor.

MiaAI-Lab's single-5090 vLLM 0.27.1 recipe remains relevant architecture
evidence: it uses the same RadixArk checkpoint with MTP-3, TurboQuant 4-bit KV,
and a patched all-GPU K+1 verify route. Any reproduction or SGLang port now
ranks first on the exact `199000+16` scoreboard. Short-context acceptance and
device-cycle measurements remain supporting diagnostics.

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
- bit-exact native-Windows Gemma residual normalization that writes the JIT
  result directly into caller-owned `x`, removing a temporary tensor and copy;
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
- the selective-checkpoint plain M4 K+1 retest, whose 14.702% cycle-cost
  increase outweighed its 3.636% acceptance gain;
- FlashInfer paged-only prefill, which changed exact-200K prompt by -0.135%
  and long generation by -2.207%;
- selecting the eager-exact SwiGLU-to-NVFP4 producer inside the compiled M3
  target graph, which changed the deterministic output because Inductor's
  one-rounding function differs from eager prefill;
- a separately exact compiled-semantics producer, whose isolated M3 boundary
  improved 70.848 -> 25.152 us while the profiled full-cycle median remained
  neutral at 16.045 ms versus 16.058 ms control;
- global chunk-7680 promotion, which regressed the base checkpoint and
  transiently reduced headroom to 200 MiB;
- chunk 7808, which regressed the selective prompt mean to 2909.350 tok/s;
- single-layer selected-row draft-extend logits, which reduced graph memory
  but changed draft-extend 1.059 -> 1.061 ms and full cycle
  16.058328 -> 16.066558 ms;
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
