# Current state

**Reconciled through:** [`experiment-log.md`](experiment-log.md), 2026-08-23
14:15 PDT.

**Qualified production source line:** commit
`03ba3d2e27` (`perf: promote native Windows decode path`). The default
launcher now selects the attention-selective checkpoint, chunk 7680, native
draft-k1 proposal construction, and Cutlass-prefill/Marlin-decode gate/up
weights. It passed exact capacity, graph, reasoning, tools, language-only,
OpenCode2, and post-flush headroom gates.

## Qualified production configuration

The accepted configuration is native-Windows SGLang serving
`C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4` on the RTX
5090. It provides a real 200,000-token target/draft pool, preserved reasoning,
parsed tools, and a language-only model surface.

| Area | Selected value |
|---|---|
| Endpoint | `http://127.0.0.1:30000/v1`, model `qwen3.8-27b` |
| Capacity | Context `200000`; total-token pool `200000`; one running request |
| Model surface | `--language-model-only`; Qwen3 reasoning parser; Qwen3 Coder tool parser |
| Target attention | FlashInfer prefill; TRT-LLM MHA/XQA decode and target verification |
| Draft attention | TRT-LLM MHA/XQA; captured draft decode and `DRAFT_EXTEND_V2` graphs |
| Linear attention | Triton GDN with ReplaySSM speculative-state handling |
| Speculation | NEXTN linear rejection sampling; 2 steps; 3 draft tokens; EAGLE top-k 1 |
| Proposal distribution | Draft top-k one with native CUDA direct one-hot q inside the single multi-step CUDA graph |
| KV | Checkpoint-selected target KV; FP8 E4M3 draft KV; page size 64 |
| Sampling | FlashInfer, including native-Windows CUDA renormalization on the speculative path |
| Prefill | 7680-token chunks |
| Mamba | 4 slots; `extra_buffer_lazy`; FP32 state |
| GEMM tuning | FlashInfer CUTLASS FP4 prefill plus in-place Marlin gate/up decode; autotune and large-EXTEND tuning enabled; FP8 GEMM autotune skipped |
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
| Primary exact `199000+16` record | **3078.058 prompt / 114.617 generation tok/s**, TTFT **64.651152 s**, E2E **64.782022 s** |
| No-override default relaunch | **3052.437 prompt / 114.053 generation tok/s**, TTFT **65.193816 s**, E2E **65.325334 s** |
| Selected-cache exact prompt window | **3047.309 tok/s** five-run mean; every request exact `199016` |
| Selected-cache long generation | **118.389 tok/s** three-run mean at exact `199000+512` |
| Current eager-fusion prompt window | **2987.275 tok/s** five-run mean in the drifted current environment; **0.914%** above the adjacent PERF-028 arm with the established digest restored |
| Current eager-fusion long support | **3001.344 prompt / 115.225 generation tok/s** over three exact `199000+512` requests; established digest restored |
| Behavior | Coherent preserved thinking; correct `703`; exactly one `multiply({"a":37,"b":19})` tool call |
| Surface | Image and audio understanding reported false |
| Final production relaunch | Promoted defaults; exact `199016`, all three graphs, arithmetic/tools/OpenCode2, and **4,338 MiB** post-flush free |

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
primary performance scoreboard. PERF-062 sets the accepted record on the
promoted launcher defaults at **3078.058 prompt / 114.617 generation tok/s**,
with **64.651152 s TTFT**, **64.782022 s** end to end, exact `199016` tokens,
and `finish_reason=length`. Root [`../BENCHMARK.md`](../BENCHMARK.md) is the
compact authority.

The next target is **3100 prompt / 120 generation tok/s**, **<=64.20 s TTFT**,
and **<=64.35 s** end to end in one eligible exact request. The timing limits
are mathematically tied to the two throughput thresholds. The user explicitly
accepted the current all-four-metric record as production; the higher milestone
remains future work.

PERF-028 and PERF-027 are now retained additive changes on the active source
line. PERF-028 fuses residual-add plus Gemma norm and improved adjacent exact
long generation by **1.205%**. PERF-027 fuses eager Qwen SwiGLU directly into
the NVFP4 tuple consumed by `down_proj`; its repaired exact window improved
prompt by **0.914%** and TTFT by **0.606649 s** versus PERF-028 while restoring
both deterministic digests. PERF-027 deliberately bypasses
`torch.compiler.is_compiling()`: Inductor removes the eager BF16 SiLU rounding
boundary, so the compiled M3 target graph retains its former function until a
separately exact compiled-semantics producer qualifies.

PERF-035's provisional FP16 QK reduction was removed. Although the first
five-request A-B moved **2985.317 -> 3005.592 tok/s**, the corrected
24-query-head/4-KV-head/256-dimension exact prefix ladder measured FP16
**163.705 ms slower** across 16 layers. FP32 remains selected.

PERF-036 closes the practical native FA2 tile family for the same dominant
kernel. CTA-Q 16 regressed to **4154.807 ms/layer**, CTA-Q 32/128 are invalid,
and the correctly routed CTA-Q-64 `NUM_MMA_KV=2` candidate regressed the exact
ladder from **3013.932 to 3414.968 ms/layer** while changing output and LSE
digests. The maintained and installed FlashInfer headers are restored to
matching SHA-256
`2E5927BDC0D36DDB393CB4FAB68C2E958D65D5B4B0085C969F7CFA777ECDFB5B`;
CTA-Q 64 with `NUM_MMA_KV=4` remains selected.

PERF-037 closes standalone norm-to-NVFP4 fusion. Its native SM120 producer was
bit-exact at all production shapes, but the captured M3
norm+quant+gate/up-GEMM boundary moved **0.096704 -> 0.097152 ms/layer**.
FlashInfer's existing PDL chain already hides the quantizer behind GEMM
startup; the prototype and its exact JIT cache were removed before model
wiring.

PERF-038 closes plain sub-128-row SM120 NVFP4 tiling. Cooperative CUTLASS
requires CTA-M >=128; ping-pong accepts a 64-row MMA but cannot map NVFP4's
fixed 128-row scale-factor TMA atom. The selected M3 tactic already swaps A/B
and uses CTA-N 32, the minimum supported epilogue/LDSM width.

PERF-039 closes MTP dual-norm/concat fusion. An occupancy-preserving native
two-CTA producer was bit-exact through the dependent FC but saved only
**1.248 us at M1** and **2.080 us at M3**, about **0.0033 ms** over both draft
phases. It was removed before routing.

PERF-040 closes the gate/up epilogue as a small extension: selected tactics are
swap-AB DP, while stock CUTLASS EVT cannot pair accumulators or halve the
output. A future implementation needs a distinct custom collective.

PERF-041 is retained in signed commit `7cb4ed0796` as a default-off native
sparse top-p producer. It is exact under the selected finite top-k contract,
including AIR boundary and cutoff-tie semantics. The first A-B-A improved the
**17.322 ms** control median to
**16.954/16.002 ms**; final regression-reviewed source independently reached
**16.000558 ms** with identical output and acceptance. The predecessor
one-pass arm did not promote client throughput: exact long generation averaged
**111.559 tok/s** with **2.194869** mean acceptance. Continue stacking
acceptance-neutral native wins; AIR remains the production default.

PERF-042 is retained in signed commit `afd5606077` as the new exact-request
prompt and timing leader, pending full promotion gates. A native page-table
builder lets aligned ordinary
prefix prefill consume the existing physical 64-token pages directly. Five
exact `199000+16` requests averaged **3209.728 prompt tok/s** with
**61.999103 s TTFT** and **62.153173 s E2E**; the worst prompt was
**3205.270**, so every prompt/time threshold passed. All 25 isolated prefix
shapes matched page-1 output and LSE bit-for-bit. Short generation averaged
only **98.029 tok/s**, so the combined four-threshold objective remains open.
Eight focused/fast-plan tests and the final adversarial review pass after
repairing stale page metadata, ownership, MXFP8 admission, and mapping checks.
Static draft top-k 32 was immediately rejected after acceptance fell
**2.217279 -> 2.173943**. Greedy draft top-k 1 was also rejected: exact
generation remained **97.900 tok/s** and direct greedy acceptance averaged
only **2.107020**. Draft top-k 16 also lost at **2.205710** versus k20
**2.217279**; k1/k8/k16/k32 close the static support-size family.
After repairing all proposal-owner routes, proposal-only top-p 1.0 is retained
default-off in signed commit `6b963eed05`. It removes both q top-p transforms, improving matched
M3 mean/median/p90 by **0.194/0.185/0.149 ms** and raising five-probe
acceptance slightly to **2.229702**. Exact short generation still reached only
**87.402 tok/s**, so the target remains open. Proposal penalty scales 0.75 and
0.0 reproduced identical proposal/output sequences, closing that scalar.

PERF-048 ReplaySSM commit overlap is rejected. Although **186.819/189.478 us**
of fold time overlapped draft extend, the draft-extend graph expanded
**1.060552 -> 1.237001 ms** from bandwidth contention. Serial fold+extend was
about **1.234266 ms**, slightly below the overlapped span; the apparent
full-cycle gain was noise. PERF-046 remains the selected generation line.

PERF-049 closes static proposal calibration. Two branch-exact chronological
p/q corpora found a large support ceiling (**2.737586**) but no generalizing
gamma, rank, or token correction; the maximin expected-length gain was only
**0.000133** and learned weights regressed held-out chronology. The opt-in
capture queue was repaired from eight to bounded 64 entries after the first
capture backpressured; the second request completed 239 records.
Signed commit `4d6782121e` retains the bounded diagnostic queue repair.

PERF-050 reopens greedy draft top-k 1 on stronger evidence. It improves
exact199K+512 generation **116.549 -> 123.049 tok/s** with identical output
and reaches **2.426540** exact-context acceptance. However, five exact16
generation scores average only **98.478 tok/s**: the request uses seven
long-context cycles averaging **19.895 ms**, so the single-request generation
target remains open. M4 still used seven cycles and is rejected. XQA SM-count
and PDL controls provide no bit-exact material win.

PERF-053/054 close the retained device-resident cycle under greedy k1 and the
available XQA structural constants. Device composition produced only
**97.730 tok/s** on exact16 with the control digest. Valid native XQA buffering
and V-tile variants saved at most **0.960 us/call**; the apparently faster
single-K-buffer build was nondeterministic. Installed FlashInfer source and
the exact control JIT module are restored.

PERF-059 retains a default-off native CUDA draft-k1 delta producer. It reduces
proposal construction from **73-87 us to 3.7-3.9 us** and improved matched
exact199K+512 generation **122.352 -> 123.559 tok/s**; an independent restart
averaged **123.831 tok/s**, with identical output and all behavior gates.
Exact16 improved only to **99.173 tok/s** and remains seven-cycle limited, so
the root benchmark target is still open.

PERF-056 proves q20 support is not the exact16 blocker: every required target
token is present and a perfect rank oracle completes in six cycles. The first
hidden-conditioned PCA-linear rank heads failed locked minority validation and
are rejected. Target-hidden residual, q-tree, KNN, and RBF follow-ups also
failed the locked ranks despite an accurate target-hidden teacher. The
default-off diagnostic now preserves exact q, hidden
payloads, target ranks, and realized accept length with bounded backpressure.

PERF-061 closes stock target-FP4 tactic and PDL changes. Global PDL-off
regressed. Bit-exact qkvz/down tactic changes projected 0.361 ms synthetically
but moved real long generation only **123.831 -> 123.972 tok/s**, inside noise.

PERF-062 promotes the native-Windows gate/up hybrid. A coalesced native
Cutlass-to-Marlin relayout matches the canonical repacker bit-for-bit and
reuses one 85 MiB scratch buffer across 64 target gate/up projections. The
accepted exact request reached **3078.058 prompt / 114.617 generation tok/s**,
**64.651152 s TTFT**, and **64.782022 s E2E**, beating all four prior record
metrics. A no-override launcher restart independently beat the old record at
**3052.437/114.053**, preserved exact `199016`, and passed every behavior and
client gate. The user accepted this profile as the launcher default; the
3100/120 milestone remains a future target rather than a promotion blocker.

The selected default remains `AttnNVFP4`, chunk 7680, M3, and the bit-exact
Windows Gemma residual-norm direct-output path. Large ordinary EXTEND
autotuning is now enabled by the launcher. The earlier independent retune
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

Base RadixArk at chunk 4096 remains the historical production control at
**2608.263 prompt / 102.358 generation tok/s** on the same exact workload.
The earlier 122.712 tok/s `6213/512` and 200-TPS geometry objectives remain
historical diagnostic context, not the current launcher defaults.

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

Applying chunk 7680 to base RadixArk reduced exact prompt throughput to
**2226.770 tok/s** and left only 200 MiB free before follow-up probes. The new
production launcher avoids that failed combination by selecting the
attention-selective checkpoint; base RadixArk remains paired with chunk 4096
when used as a control.

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

## Apple-silicon experimental handoff

The Apple route is active as an experimental lane on an M1 Max with 32 GiB
unified memory. At the user's request, the pinned Q1 cache was deleted after
its experiment, reclaiming 7.9 GB; its immutable revision and checksum remain
in the 19:09 experiment-log entry. The current retained playground is
Bartowski's conventional `Qwen3.8-27B-IQ2_XXS.gguf`, pinned at revision
`f0eec4a4bb4975114a030d048952d83c0a53c034`. It is 9,393,043,040 bytes with
SHA-256 `b01f668356e5799fd76315bd6abc0e45234580409ebc5c8fb4b675e3c10dc2b9`.
All 866 tensors use formats supported by the native torch/MPS route.
Signed commit `7740cae691` owns that packed low-bit GGUF route and its MPS
convolution-state contract. Signed commit `1271610e0b` owns the separate,
opt-in MLX quantized-prefill query tiling mechanism. Signed follow-up
`ea983f3120` expresses its cross-tile lifetime ordering with `mx.depends`,
which preserves query values and avoids arithmetic dependency propagation.
Signed commit `8879ed3d01` registers GGUF USER_DEFINED vocabulary entries as
ordinary added tokens while keeping CONTROL entries special.
Signed commit `13bea403d6` stores heterogeneous merged GGUF shards in one
compact MPS allocation and passes storage-offset views directly to Metal,
while preserving the padded CUDA/non-MPS path.
Signed commit `16b2bf7a06` specializes aligned IQ2_XXS batch-one matvec with
four-row input/LUT reuse. Signed commit `b19cf4acf3` specializes the aligned
Q5_K batch-one vocabulary head with four eight-lane row cohorts per SIMDgroup
and retains generic alignment and multi-batch fallbacks. Signed commit
`4d1641fdcd` routes one-vector F32 GGUF projections through native MPS matrix
multiplication while retaining the custom dense kernel for multi-vector
prefill.
Signed commit `210a214c12` removes padded prefix query rows from the shared
torch-native extend path and uses lower-right causal alignment for partial
chunks.
Signed commit `b2b8ab4af8` gates fused native MPS decode on its actual dtype,
physical-pool, layout, and head-dimension contract so BF16 and long pools use
the established cache-write plus SDPA path.
Signed commit `52b5326d8e` retains PERF-A016, a batch-one Q4_K tensor-family
kernel for the mixed-format IQ2_XXS/Q2 checkpoint. It reuses each activation
fragment across two output rows and admits complete four-block cohorts with
safe compact-view alignment.
Signed commit `1ec20a0e87` widens the fixed-memory BF16 Metal decode fence to
131,073 physical rows. Isolated native admission, the 131,074-row fallback,
and active sequence length 131,072 all pass; served context qualification
remains the exact 32,768-token gate.
Host cleanup leaves this artifact as the only Hugging Face model cache and no
MTPLX model cache. A broader cache cleanup also removed the first retained
copy, so the same immutable revision was downloaded again and its byte size
and SHA-256 were reverified. SGLang, Codex-runtime, uv, and other rebuildable
user caches were cold at that cleanup checkpoint. The data volume had 267 GiB
free after restoration.

The selected native-IQ2 `128+32` deterministic window averages **7.0444
prompt / 8.4406 generation tok/s**, **18.170302 s TTFT**, and **21.842957 s
E2E** over five cache-flushed runs. This is **164.94%** more generation
throughput than the pre-kernel padded control's **3.1858 tok/s**, with the
same deterministic digest. An independent committed restart reached
**8.420 tok/s**. Required-sampling windows averaged **8.3094** and **8.2942
tok/s** across two restarts, 4.59% and 4.26% above the corresponding selected
Q5_K windows.

Packed weight loading reports **9.03 GB**, down from 10.03 GB; Mamba and KV
allocations add about 0.41 GB. Matched generic/candidate IQ2_XXS
`17408x5120` medians reached **1.176875 -> 0.516000 ms**. The Q5_K
`248320x5120` vocabulary head reached **19.659291 -> 3.754625 ms** in the
matched source window. Across all 48 actual F32 `96x5120` b/a projections,
the selected native MPS path changed **7.296667 -> 2.159000/2.051708 ms** in
an A/B/A sweep. Actual-file batch 1/3/4/8 parity, odd row boundaries, compact
offsets, and focused packed extrema pass.

The earlier missing-final and empty-thinking-disabled behavior was a native
GGUF tokenizer defect rather than checkpoint evidence. Qwen's `<think>`,
`</think>`, and tool markers are GGML USER_DEFINED entries; before
`8879ed3d01`, they fragmented into ordinary text pieces. A fresh explicit-parser
server now returns preserved `reasoning_content` plus visible final `703`,
exact thinking-disabled `READY`, exactly one parsed
`multiply({"a":37,"b":19})` call with `finish_reason=tool_calls`, and a
tool-result continuation ending in `37 × 19 = **703**`. `/model_info`
continues to report image/audio understanding false. The focused tokenizer,
reasoning-parser, and tool-parser suites passed 321 tests plus 64 subtests.

Pinned llama.cpp build 10547 owns the route-neutral M1 Max Q2 `12+256`
reference at **14.661356 tok/s** aggregate and **14.671473 tok/s** best hit.
The first native Rust `/generate` baseline remains **7.001584 tok/s** aggregate,
**7.015010 tok/s** best hit, and **36.563154 s** mean E2E. It established the
official-tokenizer fixed-output boundary with exact token IDs and FNV
`6d4d220de481f54e`.

PERF-A016 is the selected repository-native result on the tool-capable Python
ingress with the same official tokenizer. Its final-source five-run window is
**8.586948 tok/s** aggregate, **8.591773 tok/s** best hit, and **29.812688 s**
mean E2E. The fresh disabled-kernel control is **7.009167 tok/s**, attributing
a **22.510241%** full-model gain; an independent candidate restart reaches
**8.578205 tok/s**. Candidate, control, and restart all reproduce exact
`12+256` token IDs, text, length finish, and digest. Final parity covers the
enabled complete cohort and output-tail fallback. The safe host rule requires
Q4_K, batch one, four-row output alignment, four-block cohorts,
`weight_offset % 2 == 0`, `input_offset % 4 == 0`, and Apple7+ pipeline
capability. `Q4_K` names one internal tensor family among the checkpoint's 866
mixed-format tensors; benchmark and checkpoint standing remain Q2.

The selected 1,024-token-chunk route completed exact **32761+1** inside the
32,768-token BF16 pool at **19.242 prompt tok/s** and **1702.563753 s E2E**.
It also passes sampled reasoning, exact arithmetic `703`, thinking-disabled
`READY`, one parsed multiply call, tool-result reasoning continuity, and
image/audio-disabled reporting. Historical process-scoped OpenCode 1.18.15
runs admitted 13,635 and 13,691-token agent prompts. The governing Apple real-
client gate is now Codex CLI 0.149.0 through the machine-local
`qwen38-local` Responses profile: one read-only `pwd` tool call returned the
workspace, its result was consumed, visible final was exact
`CODEX TOOL READY`, and the client accounted for 17,871 input, 96 output, and
62 reasoning-output tokens before exiting zero. Profile/catalog SHA-256 values
are `9706003ad8a43ad48e4260f282057c023214c9e66737eae3da88a49188079a1c`
and `a67c491a1dd4d4df0f720fb966ac390bd20041d8ed29f02833dfca4424a013f0`.
Verified cleanup leaves every server/client PID absent, port 30000 free, 94%
memory free, and normal thermal/performance status.

A one-shot synchronized batch-one profile now supplies a candidate-selection
diagnostic.
After excluding layers 0-10 with visible first-use cost, full-attention layers
average **3.130000 ms** and GDN layers average **1.719026 ms**. Extrapolated
across the 16/48 topology from the raw sums, the stable profile projects to
**132.593 ms/token**. The separate exact request amortizes to **142.824820
ms/completion token**; their **10.232 ms/token** numerical difference crosses
runs and context distributions, so it does not isolate an outside-layer
budget. Nested layer-8 stages suggest that projection and MLP work outweigh
the recurrent core, while that layer is itself inside the first-use region
and nested synchronization perturbs its timing. These numbers guide a cheap
hypothesis test and do not replace end-to-end timing.

Both source-level context blockers are now retained. Exact source A/B at a
`4096+4096` partial extend reduced the padded-query controls from
**542.376416/641.256125 ms** to **176.066500 ms** with exact MPS output. A
`4096+256` rung improved **97.995583/97.847500 -> 12.608125 ms**. Pre-change
decode probes failed at BF16/32,769 rows and FP32/7,937 rows. Both now reach
the cache-write plus SDPA fallback with zero observed error, while the fused
FP32/7,936 boundary remains active with maximum error `2.5331974e-07`. The
first controlled 32K-configured BF16 launch reached ready state with an
allocated 32,768-token pool and exposed large-batch IQ2_XXS projection as the
remaining prefill blocker.

Signed commit `1676c71bed` retains PERF-A014, which routes Apple7+ IQ2_XXS
batches above eight through an FP32 SIMD-matrix kernel that shares each
dequantized 64x32 weight tile across 32 input rows. The actual
`blk.8.ffn_gate.weight` (`17408x5120`) improved from
matched medians **70.074833 -> 4.250250/4.277125 ms** at batch 128, from the
adjacent **249.418209 -> 16.009833 ms** at batch 512, and from
**1971.539875 -> 124.838125 ms** at batch 4096. Aligned, odd output/batch,
minimal-tail, and fallback-boundary parity pass with maximum IQ2 relative
error `2.27121e-06`; batch one/four/eight remain unchanged.

The same 32K-configured BF16 full model completed sampled `128+32` at
**23.093 prompt / 6.736 generation tok/s**. Cache-flushed `4096+2`, which
formerly crossed the
300-second watchdog, now completes at **24.828 prompt tok/s** with
**164.975078 s TTFT**. Exact `5000+1` completes two chunks at **24.845 prompt
tok/s** and **201.251071 s E2E**. Arithmetic `703`, thinking-disabled `READY`,
one parsed `multiply({"a":37,"b":19})`, preserved tool-result continuation,
and image/audio-disabled gates all pass. Verified cleanup leaves port 30000
free, no server or workload-owned compiler process, 90% free memory, and no
recorded thermal or performance warning.

A process-scoped served control then isolated the large-batch dispatch on
exact `128+1`. Disabling it produced **7.0234 prompt tok/s** and
**18.225055 s TTFT** over five cache-flushed samples. Two independent default
launches produced **22.9556** and **22.8072 prompt tok/s** five-sample means;
their combined mean is **22.8814**, with **5.594303 s TTFT**. This is 3.258x
the matched disabled prompt rate. Every request completed exact `128+1` with
`finish_reason=length`; the true batch-eight fallback also passes actual-file
parity.

The selected native route has cleared its semantic, sampled, independent-
restart, exact-capacity, and Codex-profile gates. Its **8.586948 tok/s**
aggregate leaves a **41.431420%** gap to pinned llama.cpp and makes the next
measured batch-one decode hotspot the compact-scoreboard handoff.

Long-context EXTEND now has a qualified native mechanism. PERF-A017 implements
batch-one BF16 paged GQA at 24 query heads, four KV heads, and dimension 256
with Q8/C64 Metal online softmax in 20,800 bytes of threadgroup storage. At
`E=17,L=131072`, its final-source three-sample median is **137.906625 ms**, its measured
post-input driver-residency delta is **0 MiB**, and dense MPS SDPA takes
**424.528292 ms** while adding **8,088.515625 MiB**. Maximum error is
`4.3120235e-07`; shuffled mappings, nonzero offsets, tile tails, query length
1,024, invalid metadata, and host safety guards pass. The shader lives in a
separate lazy Metal library, leaving the ordinary extension pipeline cache and
its failure boundary unchanged.

The raw native binding is outside `TorchNativeAttnBackend.forward_extend`.
The explicit no-new-Python rule makes clean C++ dispatch ownership the active
architecture gate. The measured 64K generic route remains closed by swap and
forward-progress limits, so served capacity stays qualified at exact
`32761+1`. The next native rung is direct BF16 matrix loading for consecutive
eight-slot cache runs, followed by fragmented-map fallback parity and the same
131K timing/residency probe.

The deleted affine-q4 scoreboard belonged to a separate Mac Pro experiment
and carries no M1 Max record standing. The retained MLX quantized-prefill query
tiler remains mechanism code behind
`SGLANG_MLX_QUANTIZED_PREFILL_QUERY_TILE`; any future M1 use requires fresh
dependency, correctness, performance, memory, and capacity qualification.

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
- explicit FlashInfer paged-prefix split sizes 4096/8192, which each required
  2.265 GB temporary storage from the qualified 128 MiB workspace;
- packed GDN target-verify split removal, because the selected Qwen width
  already takes zero-copy Q/K/V aliases and ReplaySSM accepts their strides;
- final-tail coalescing, whose 14,680-token ragged-current pass regressed prompt
  to 1,917.509 tok/s, TTFT to 103.781 s, and changed deterministic output;
- full-attention gate-to-NVFP4 fusion, which was exact but projected to only
  0.0068 ms per M3 replay and about 20.9 ms over the exact prompt;
- global page 128/32 tuning: 128 lost exact pool capacity, while 32 bypassed
  the page-size-1 prefill interface and regressed long generation;
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
