# Decision ledger

This ledger records choices that still govern the native-Windows Qwen3.8
system. Exact sample lists, commands, incident detail, and intermediate states
remain in [`experiment-log.md`](experiment-log.md).

**Reconciled through:** 2026-08-21 13:48 PDT.

## Selected production choices

| Decision | Selected choice | Durable evidence |
|---|---|---|
| Checkpoint | Attention-selective RadixArk Qwen3.8-27B NVFP4 | Launcher-default exact-200K record; preserved coherent reasoning, tools, OpenCode2, and 4,338 MiB post-flush free |
| Capacity | Real 200K target and draft pools | Exact `199016` passed repeatedly; 232K reached 98 MiB free before cache flush and was rejected for operating margin |
| Primary performance scoreboard | Exact `199000+16` near-limit request | Accepted record is **3078.058 prompt / 114.617 generation tok/s**, **64.651152 s TTFT**, and **64.782022 s E2E**; exact `199016` remains required |
| Next performance milestone | **3100 prompt / 120 generation tok/s**, TTFT **<=64.20 s**, E2E **<=64.35 s** | The two time limits are derived from the throughput thresholds on exact `199000+16`; all four must pass in one eligible request |
| Model surface | Language-only with Qwen3 reasoning and Qwen3 Coder tools | Preserves required behavior and VRAM; image/audio remain disabled |
| FlashInfer | Clean native-Windows port of 0.6.17 | Passed JIT/kernel tests, fixed long-prefill correctness, and satisfies the SGLang version contract |
| Prefill | FlashInfer, launcher-default chunk size 7680 | Qualified on the attention-selective checkpoint; base RadixArk/chunk 4096 remains an explicit control |
| Default long-context profile | `AttnNVFP4`, chunk size 7680 | Exact `199000+16` reached **3078.058/114.617**; an independent no-override launch reached **3052.437/114.053** |
| Target verify/decode | TRT-LLM MHA/XQA | Qualified real throughput and exact 200K capacity; compact unread mask removes redundant generic work |
| Draft decode | TRT-LLM MHA/XQA | Controlled gain over Triton draft decode; semantics and long ladder passed |
| Draft extend | Captured `DRAFT_EXTEND_V2` graph | Removed an eager dispatch wall and contributed to the later two-step winner |
| Linear attention | Triton GDN with ReplaySSM | Correct Qwen recurrent-state handling in the selected linear speculative topology |
| Draft KV | FP8 E4M3 | Reduced memory and improved the selected topology while preserving behavior |
| Speculation geometry | 2 steps, 3 draft tokens, EAGLE top-k 1 | Qualified **122.712 tok/s** real mean and **171.263 tok/s** fixed mean |
| Proposal alignment | Draft top-k one with native CUDA direct one-hot q | Exact q remains rejection-correct; removes softmax/dense renormalization/categorical proposal work and is enabled by the Windows launcher |
| Chain metadata | Native C++/CUDA fixed-chain path with distinct per-cycle outputs | 4.227x isolated metadata speedup while preserving asynchronous output lifetimes |
| Sampling | FlashInfer | Native CUDA renormalization controls the speculative target path; fallback sampling remains available |
| Native elementwise/norm | C++/CUDA SiLU, RMSNorm, Gemma RMSNorm, fused Gemma residual-add norm, direct Gemma residual output, and qualified sigmoid-multiply dispatch | Both Gemma paths are bit-exact; the fused residual-add norm improved adjacent exact long generation from 115.194 to 116.583 tok/s |
| Eager MLP activation quantization | Exact native SwiGLU-to-NVFP4 producer outside `torch.compile`; preserve the former compiled M3 path | All-finite-BF16, production-shape, graph, and tuple-consumer gates pass. Exact prompt improved **0.914%** versus PERF-028 with both deterministic digests restored |
| GEMM tuning | FP4 autotune plus large EXTEND; skip FP8 GEMM autotune | Selected target file hits improve long prefill; launcher enables the retained path |
| Gate/up decode | In-place Cutlass-prefill/Marlin-decode layout for all 64 target gate/up projections | Canonical repack parity and round-trip tests pass; exact record beats all prior metrics while reusing one 85 MiB scratch buffer |
| Selective tactic cache | Keep the independently selected 20,928-byte cache | SHA-256 `8219484FA86EBB0E6DDA54F2D15447DBC502EBCEA9007B3E1BB917B9001F9ADF`; fresh selection regressed long generation and requires requalification |
| Workspace | 128 MiB | Wins decode and long prefill; 64 MiB fails required graph allocation |
| Compile mode | `default`, with established partial fallbacks | Five-run fixed-work win over other compile/fallback arrangements |
| Scheduling | Receive interval 4; stream interval 4; incremental output | Measured fixed-work wins while retaining client streaming behavior |
| Implementation language | C++/CUDA hot paths with thin Python integration | Explicit user direction after the display-GPU incident; preserves graph capture and native dispatch |
| Tree/SWOR implementation | Retained as opt-in, production-ineligible infrastructure | A non-front unified-pool KV/compaction defect exists outside the measured static-pool route; current-config full cross-cycle parity is still required, and raw-composite SWOR RNG is invalid |
| Device-resident linear cycle | Retained opt-in; rejected for throughput | Exact-q dense-race and explicit-seed categorical forms reached 122.576 and 120.075 tok/s versus 124.775 matched control; ordinary scheduling remains selected |
| Geometry funding gate | Complete lattice; conservative lower >=215 TPS; strictly above measured frontier | Family rejection uses an impossible target-aware upper <=200 TPS; selected-tree gaps fail closed |
| Graph-tail work | Closed at the 0.75 ms admission gate | Two independent windows produced 1,471 CUDA-event records; best repeatable recoverable p10 was 0.658355 ms |
| Target attribution | Exact per-shape M/N/K plus overlap-aware exposure | M3 primary GEMMs occupy 12.360 ms on the terminal stream; aggregate residency alone overcounts alternate-stream overlap |
| Selective target NVFP4 | Launcher-default production checkpoint | User accepted the all-four-metric record; default relaunch and behavior/client gates passed |
| MiaAI-Lab vLLM recipe | Matched `199000+16` reproduction remains an information gate | Same checkpoint/GPU uses MTP-3, TurboQuant 4-bit KV, and patched full-graph K+1 verify; published ~160 TPS lacks raw workload evidence |

## Primary performance record

The root [`../BENCHMARK.md`](../BENCHMARK.md) scoreboard governs optimization
ranking. The selective target-NVFP4 checkpoint completed exact `199000+16` at
**3078.058 prompt tok/s**, **114.617 generation tok/s**, **64.651152 s TTFT**,
and **64.782022 s** end to end. An independent launcher-default relaunch
reached **3052.437/114.053**, **65.193816 s TTFT**, and **65.325334 s E2E**.
Both exact requests beat every prior record metric. This is the record to beat.

The next milestone is **3100 prompt / 120 generation tok/s**, with TTFT
**<=64.20 s** and end-to-end time **<=64.35 s** in the same eligible request.

## Qualified reference results

| Result | Accepted value |
|---|---|
| Real sampled `6213/512` | **122.712 tok/s** ten-run mean, **122.371** median, **137.074** peak |
| Fixed accepted-length-3 `6213/512` | **171.263 tok/s** five-run mean |
| Native two-step acceptance | **2.318174** mean emitted/accepted tokens per verification over five probes |
| Near-limit `199000/16` | **2608.263 prompt tok/s**, **102.358 generation tok/s**, exact `199016` total |
| Final production graph headroom | **1.84 GiB** reported after restored 200K capture |

## Closed or rejected candidates

### Checkpoints, quantization, and kernels

| Candidate | Status | Why |
|---|---|---|
| GGUF as production checkpoint | Superseded | Base NVFP4 improved prompt throughput by roughly 12.8x and E2E by 4.2x on `6213/128` |
| FlashInfer 0.6.11 attention | Rejected | Faster synthetic result produced degenerate repetition on the real long OpenCode prompt |
| Native target NVFP4 KV | Rejected | Recovered about 2.2 GiB but corrupted thinking and tool behavior |
| Stock `nvfp4_online` for draft | Superseded by dense experiment | The checkpoint is dense; the original MoE-only path left draft storage and lost fixed work |
| Full online FP8 MTP | Rejected for throughput | **167.023 tok/s** fixed versus the **171.263** BF16 control; activation quantization erased the GEMM saving |
| Full online MXFP8 MTP | Rejected for throughput | Mechanism qualified end to end but reached **163.457 tok/s** fixed |
| Dense online NVFP4 MTP | Rejected for throughput | Mechanism and graph replay qualified; **164.094 tok/s** fixed with about 0.46 GiB memory saving |
| Gittensor ModelOpt FP4 checkpoint | Rejected as production winner | Better acceptance and smaller residency, yet **119.092 tok/s** real and **154.883 tok/s** fixed lost to RadixArk |
| Gittensor/RadixArk hybrid `lm_head` | Closed by user direction | Source checkpoints remain immutable and RadixArk was restored as the active checkpoint |
| CUTLASS channelwise-FP8 dispatch | Rejected | Alignment and numerical tests passed; robust paired medians showed no material win |
| CUTLASS DSL / FlashInfer GDN on Windows | Unavailable | Required Windows DSL support/package is absent for this native path |
| Fully compiled repaired Triton kernels | Rejected | Correct yet slower, with very long startup compilation |
| Explicit compiler-disable boundaries | Rejected | Changed graph segmentation and lost throughput |
| Native fused-add RMSNorm on the target path | Gated | Residual is exact while output can move by one BF16 step; any draft-only use needs a separate controlled gate |
| Eager-exact SwiGLU-to-NVFP4 in compiled M3 | Rejected as a global route | Inductor removes the eager intermediate BF16 round; selecting the eager producer globally changed deterministic output. A compiled-semantics producer needs separate exact qualification |
| Compiled-semantics SwiGLU-to-NVFP4 | Rejected for throughput | Exact isolated latency improved 70.848 -> 25.152 us, but 233 full M3 cycles retained a 16.045 ms median versus 16.058 ms control; client movement stayed inside variance |
| FlashInfer fixed prefill split | Rejected at workspace gate | Split 4096 and 8192 both requested 2.265 GB temporary storage from the qualified 128 MiB workspace before the first exact warmup |
| Packed GDN target verify | Already selected through aliases | Qwen3.8 width 10,240 bypasses the materialized split and ReplaySSM accepts the existing strided Q/K/V views; no implementation is needed |
| Coalesced final prefill tail | Rejected | Merging 7680+7000 into one 14680-token ragged pass regressed prompt to 1917.509 tok/s, raised TTFT to 103.781 s, and changed deterministic output |
| Attention gate-to-NVFP4 fusion | Rejected below admission | Exact M3 saving projected to 0.0068 ms/replay and large-prefill saving to about 20.9 ms; no model wiring was retained |
| Global KV page size | Keep 64 | Page 128 floors exact pools to 199,936; page 32 does not reach prefill's token-index wrapper and reduced long generation to 112.576 tok/s |
| Paged-prefix QK reduction | Keep FP32 | Exact 25-prefix ladder measured FP16 163.705 ms slower across 16 layers; the provisional +0.679% server movement was noise |
| Paged-prefix FA2 tile | Keep CTA-Q 64 and `NUM_MMA_KV=4` | CTA-Q 16 was 37.9% slower, CTA-Q 32/128 are invalid, and `NUM_MMA_KV=2` regressed the exact ladder 13.3% while changing output/LSE digests |
| Gemma norm-to-NVFP4 | Rejected at dependent boundary | Bit-exact native fusion improved the isolated launch, but norm+quant+gate/up GEMM moved 0.096704 -> 0.097152 ms/layer because PDL already hides quantization |
| M3 NVFP4 tile geometry | Keep selected swap-AB CTA-N 32 family | CTA-M 64 violates the 128-row scale TMA atom; CTA-N 32 is the minimum supported epilogue/LDSM width |
| MTP dual norm/concat | Rejected below funding | Exact native two-CTA fusion saved only 1.248 us at M1 and 2.080 us at M3 through the dependent FC |
| Sparse top-p after finite top-k | Retain default-off native Windows opt-in in `7cb4ed0796`; keep AIR production default | Exact AIR pivot with 15 CUDA + 6 integration tests and repeatable cycle win; predecessor standalone long generation averaged only 111.559 tok/s |
| Page-aligned FlashInfer prefix prefill | Retain default-off in `afd5606077`; production promotion pending | Bit-exact 25-shape ladder improved 5.270%; five exact prompts averaged 3209.728 tok/s with every prompt/TTFT/E2E gate passing |
| Draft proposal top-k 32 | Rejected against the historical k20 route | Five-probe acceptance fell 2.217279 -> 2.173943 and latency worsened |
| Greedy draft proposal top-k 1 early screen | Superseded by PERF-050 | The single exact sample and three 6K probes understated the later exact199K+512 k1 gain |
| Draft proposal top-k 16 | Rejected against the historical k20 route | Five-probe acceptance averaged 2.205710 versus 2.217279 at k20 |
| Proposal-only top-p 1.0 | Retain default-off in `6b963eed05` | After fixing all proposal-owner routes, AIR top-p fell 3 -> 1 launch/cycle; matched mean/median/p90 improved 0.194/0.185/0.149 ms and acceptance rose slightly |
| Proposal additive-penalty scale | Rejected for current workload | Correctly routed scales 0.75 and 0.0 reproduced identical proposal/output sequences |
| ReplaySSM commit overlap | Rejected | 186.8 us fold overlap expanded draft graph 8 by 176.4 us; serial fold+extend ~1.234 ms beat overlapped ~1.237 ms |
| Static proposal gamma/rank/token calibration | Rejected | Two branch-exact corpora showed trajectory-dependent mismatch; maximin gain only 0.000133 and rank/token fits regressed held-out chronology |
| P/q diagnostic queue | Increase bounded capacity 8 -> 64 in `4d6782121e` | Eight entries crashed at 151 cycles; bounded 64 completed a 239-record request without ordinary-path impact |
| Greedy draft top-k 1 | Promoted with native delta q and hybrid target numerics | Exact199K+512 improved 116.549 -> 123.049 before native q; the final gate/up hybrid then set the accepted exact16 record |
| XQA SM count / PDL | Keep all SMs and default PDL | Lower SM counts changed output for negligible savings; PDL on/off was timing-neutral |
| M4 with greedy k1 | Rejected | Exact16 still required seven cycles; extra row/step adds cost without reducing the gate |
| Device-resident cycle with greedy k1 | Rejected | Exact16 reached 97.730 tok/s with the control digest, below the adjacent 98.478 tok/s mean |
| SM120 XQA structural constants | Keep restored FlashInfer control | Valid V buffering/tiling saved <=0.960 us; single-K-buffer output was nondeterministic and two buffers exceeded shared memory |
| Native draft-k1 delta q | Retain default-off additive win | Exact q construction fell to 3.7-3.9 us; matched long generation improved 122.352 -> 123.559 tok/s and independent restart reached 123.831 |
| Hidden rank classifier | Rejected; retain proof-bearing diagnostic only | q20 support gives a six-cycle oracle, but blocked minority-rank validation was 0-25% and locked exact16 corrections failed |
| Compressed target KV | Closed as exact16 solution | Native NVFP4 XQA ceiling is only 0.520 ms/cycle before overhead and the format fails semantics |
| Exact target FP4 tactic overrides / PDL | Keep selected cache and PDL | All-tactic sweep was bit-exact, but the best synthetic pair moved real generation only +0.114%; global PDL-off regressed |
| Gate/up custom epilogue | Closed as a small EVT change | Selected tactics are swap-AB DP and need a custom half-height collective; stock EVT cannot pair/halve coordinates |
| FlashInfer paged-only prefill | Rejected | Exact-200K prompt changed **2789.036 -> 2785.260 tok/s** and 512-token generation changed **106.467 -> 104.117**; deterministic output also changed |
| Global chunk-7680 default | Rejected | Base RadixArk exact prompt fell to **2226.770 tok/s** and only 200 MiB remained before follow-up probes |
| Selective chunk 7808 | Rejected | Exact-200K prompt averaged **2909.350 tok/s**, a stable cliff below the 7680 winner |
| Single-layer selected-row draft-extend logits | Rejected | Graph memory fell, but draft-extend stayed **1.059/1.061 ms** control/candidate and full cycle stayed **16.058328/16.066558 ms** |

The optional Windows quantization registrations, conversion repairs, backend
selection, and isolated tests remain valuable compatibility work. Their
production performance status stays closed unless the cost topology changes.

### Speculation, proposal, and scheduling

| Candidate | Status | Why |
|---|---|---|
| One-step/two-token MTP | Rejected | Fixed samples near 102 tok/s expose an insufficient emission ceiling |
| Static three-step/four-token MTP | Rejected for real production | Full acceptance crossed 200 tok/s once, while honest sampled mean was **117.239 tok/s** |
| Selective-checkpoint M4 K+1 retest | Rejected | Acceptance improved 3.636%, but the matched full cycle regressed 14.702% and projected TPS fell **139.841 -> 126.350**; exact-200K generation overlapped M3 noise |
| Adaptive 2/3 depth, aggressive policy | Rejected | Oscillation reduced acceptance and first real sample reached only 100.739 tok/s |
| Adaptive 2/3 depth, sparse policy | Rejected | Two real windows combined to **110.276 tok/s** |
| No MTP | Superseded | Useful control and slower than the trained RadixArk MTP path |
| Draft proposal top-k 8 | Rejected | Lower acceptance and **119.741 tok/s** three-run mean |
| Earlier eager aligned top-k 20 | Superseded by captured alignment | Eager/per-step graphs imposed a large fixed-work tax; single-CG alignment later made top-k 20 the selected path |
| Generic sampler swap to PyTorch | Rejected | Steady EAGLE bypassed the selector and paired fixed medians overlapped |
| Top-k 2 speculative tree in the old linear path | Rejected | No correct sampled path with native-Windows rejection, XQA, and ReplaySSM at that stage |
| Reusable fused metadata output buffers | Rejected | Simulated fixed work rose, while exact-seed real output showed a scheduling/aliasing regression |
| Continuous decode steps 4 | Rejected | Increased TTFT and E2E on the matched control |
| BF16 Mamba state | Rejected | Slower and changed deterministic output; FP32 remains selected |
| ReplaySSM on the original non-speculative GGUF route | Topology-specific rejection | It later became selected for the linear-chain MTP topology |

### Memory, context, and environment

| Candidate | Status | Why |
|---|---|---|
| Chunk size 8192 | Rejected | Lost on short work and collapsed after repeated 32K requests under VRAM pressure |
| Workspace 64 MiB | Rejected | Deterministic graph-capture buffer overflow; 128 MiB is the floor |
| FP8-only autotune | Rejected | Lost decode and long prefill while reducing headroom |
| Full FP4+FP8 autotune | Rejected | Inferior to FP4-only tuning and regressed large prefill |
| Fresh large-EXTEND profiling on every launch | Rejected | Exact prompt remained strong at **3043.747**, but independently selected tactics reduced long generation to **101.162 tok/s**; retain the qualified cache instead |
| 232K production context/pool | Rejected | Exact `231000+16` passed, but only 98 MiB remained before cache flush |
| NVML polling or keepalive | Rejected | Apparent gains failed to persist; WDDM client traffic explained the variance |

### Exact tree and SWOR experiments

| Candidate | Status | Why |
|---|---|---|
| Target-only M12 tree | Rejected as production topology | About **2.9436** emitted/traversal and one 104.145 tok/s stream; cycle cost required more output than the shape can emit |
| Target-only M8 tree | Rejected | Saved 4.81% captured work and lost about 4.52% yield; stream reached 94.080 tok/s |
| Six-step/depth-only tree | Rejected | Mean **3.1025** emitted/traversal and only 87.589 tok/s |
| Fully normalized aligned tree scoring | Rejected as default | Reduced M12 yield by about 3.45% versus plain scoring |
| Scalar depth discount 0.8 | Rejected | **2.9286** emitted/traversal versus plain M12 **2.9436** |
| Initial M12 exact SWOR topology | Rejected | **2.9653** emitted/traversal and **84.713 tok/s** real mean |
| Topology-only SWOR search at current q | Exhausted | Optimistic 32-node search reached only **4.0921** expected outputs and remained cost-limited |
| M3 depth-two geometry | Rejected by impossible oracle | Mean 19.446 ms full cycle caps perfect output at **154.270 TPS**; best observed cycle caps it at **167.480 TPS** |
| Post-change M8/M12/M16 depth-four geometries | Rejected by impossible oracle | Best-sample perfect-path ceilings are **185.782**, **179.547**, and **166.666 TPS**, below 200 before real sampling |
| Selected-tree p/q corpus for counterfactual topology ranking | Diagnostic only | The observed membership is replayable; incomplete descendant/support coverage makes every counterfactual policy unavailable |

The tree implementation itself is retained. It includes native target-only/SWOR
sampling, sparse support up to 64 entries with dense fallback, low-rank GDN
tree replay, accepted-path recurrent/conv commit, path and overlap oracles,
custom topology parsing, profiling, and offline search tools. Its earlier
throughput measurements are mechanism-only because a non-front accepted path
could address target KV in the wrong unified-pool space or bypass multi-layer
front compaction. Isolated repair coverage passes; full-model requalification
has not.

## Protected boundaries

- Preserve the user's worktree and unrelated `sglang.bundle`.
- Preserve the original FlashInfer checkout and the clean 0.6.17 Windows port
  as separate provenance lines.
- Preserve downloaded RadixArk and Gittensor checkpoints byte-for-byte; place
  derived artifacts at separate paths with provenance and checksums.
- Leave the protected CUDA compatibility headers untouched. Their recorded
  SHA-256 is
  `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.
- Keep OpenCode2's cloud-model configuration stable during local server tuning;
  use process-scoped aliases or wrappers.
- Use exact process ancestry for server lifecycle actions and preserve every
  unrelated user process.

## Reopening criteria

The exhaustive optimization goal is complete. A closed branch reopens when an
explicit user request or materially new evidence changes its governing
assumption:

- new kernels or hardware alter per-depth draft/target cost;
- proposal overlap improves enough to change tree yield;
- repaired tree acceptance passes deterministic multi-cycle non-front path
  parity against the serial linear reference before any throughput ranking;
- a new checkpoint passes the same preserved-thinking/tool/capacity contract;
- dependency support changes the native-Windows backend boundary;
- a newly measured production gap survives matched controls and environmental
  accounting.

A historical peak, simulated acceptance, microbenchmark, source inspection, or
single stochastic window remains supporting evidence within the full promotion
contract.
