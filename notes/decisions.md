# Decision ledger

This ledger records choices that still govern the native-Windows Qwen3.8
system. Exact sample lists, commands, incident detail, and intermediate states
remain in [`experiment-log.md`](experiment-log.md).

**Reconciled through:** 2026-09-01 15:39 PDT.

## Selected production choices

| Decision | Selected choice | Durable evidence |
|---|---|---|
| Checkpoint | Attention-selective RadixArk Qwen3.8-27B NVFP4 | Launcher-default exact-200K record; preserved coherent reasoning, tools, OpenCode2, and 4,338 MiB post-flush free |
| Serving architecture | Trained DSpark-v2, online-FP8 draft, gamma seven/eight-token linear verification, one request | Two clean exact-200K sampled windows average **155.961** and **162.500 tok/s**; the latter is an argument-free restart and every individual sample reaches at least 150 |
| Capacity | Real 200K target and draft pools | Argument-free DSpark-v2 repeated exact `199016` at **3118.215 prompt tok/s**, **63.818572 s TTFT**, and **64.236571 s E2E**; 232K remains rejected at 98 MiB margin |
| Primary performance scoreboard | Uncached exact `6213+512` ordinary sampling: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5 | Python authority windows **155.961** and **162.500 tok/s** independently clear the requested 150; exact counts, length finish, reasoning/content, process brackets, and client implementation are recorded |
| Next performance milestone | None active; reopen native-Windows throughput only by explicit request or a newly measured gap | The requested 150 tok/s objective is qualified on the literal launcher default, including capacity, behavior, OpenCode2, and Codex gates |
| Model surface | Language-only with Qwen3 reasoning and Qwen3 Coder tools | Preserves required behavior and VRAM; image/audio remain disabled |
| FlashInfer | Clean native-Windows port of 0.6.17 | Passed JIT/kernel tests, fixed long-prefill correctness, and satisfies the SGLang version contract |
| Prefill | FlashInfer, launcher-default chunk size 4096 | Exact 200K DSpark-v2 capacity and both sampled windows pass; chunk 7680 remains the historical NEXTN/selective-checkpoint control |
| Default long-context profile | `AttnNVFP4` target + DSpark-v2 online-FP8 draft + five FP32 Mamba slots | Literal argument-free launch captures the intended graphs, keeps exact 200K pools, clears 150 tok/s, and survives the real multi-chunk clients |
| Target verify/decode | TRT-LLM MHA/XQA with static target-graph draft-KV commit | All eight reserved verify rows are projected/written in graph; rejected rows stay unreachable, eager fallback remains, exact-capacity/behavior/acceptance parity passes |
| Draft decode | Triton inside the folded DSpark draft graph | Current trained-draft route is faster end to end with the graph-side commit despite the earlier NEXTN/XQA selection |
| Draft proposal | One captured DSpark proposal/sampling graph | Greedy and sampled proposal construction remain folded; the one-step DSpark topology has no separate production draft-extend phase |
| Linear attention | Triton GDN with ReplaySSM | Correct Qwen recurrent-state handling in the selected linear speculative topology |
| Draft KV | FP8 E4M3 | Reduced memory and improved the selected topology while preserving behavior |
| Speculation geometry | DSpark block size 7, eight target verification rows, one proposal step | Qualified ordinary sampled means **155.961/162.500 tok/s** with healthy native acceptance on two independent full-pool launches |
| Proposal distribution | Immutable trained rank-256 DSpark-v2 Markov head with ordinary rejection sampling | Controlled trained fixture parity, coherent serving behavior, native acceptance counters, and two real sampled windows pass; truncated/sparse variants remain default-off |
| Chain metadata | Native C++/CUDA fixed-chain path retained for NEXTN/control experiments | The selected DSpark path preserves its own graph-safe outputs and asynchronous lifetimes; do not reintroduce reusable per-cycle output aliasing |
| Sampling | FlashInfer | Native CUDA renormalization controls the speculative target path; fallback sampling remains available |
| Native elementwise/norm | C++/CUDA SiLU, RMSNorm, Gemma RMSNorm, fused Gemma residual-add norm, direct Gemma residual output, and qualified sigmoid-multiply dispatch | Both Gemma paths are bit-exact; the fused residual-add norm improved adjacent exact long generation from 115.194 to 116.583 tok/s |
| Eager MLP activation quantization | Exact native SwiGLU-to-NVFP4 producer outside `torch.compile`; preserve the former compiled M3 path | All-finite-BF16, production-shape, graph, and tuple-consumer gates pass. Exact prompt improved **0.914%** versus PERF-028 with both deterministic digests restored |
| GEMM tuning | FP4 autotune plus large EXTEND; skip FP8 GEMM autotune | Selected target file hits improve long prefill; launcher enables the retained path |
| Gate/up decode | In-place Cutlass-prefill/Marlin-decode layout for all 64 target gate/up projections | Canonical repack parity and round-trip tests pass; exact record beats all prior metrics while reusing one 85 MiB scratch buffer |
| Selective tactic cache | Keep the independently selected 20,928-byte cache | SHA-256 `8219484FA86EBB0E6DDA54F2D15447DBC502EBCEA9007B3E1BB917B9001F9ADF`; fresh selection regressed long generation and requires requalification |
| Workspace | 128 MiB | Wins decode and long prefill; 64 MiB fails required graph allocation |
| Compile mode | Disabled for the selected DSpark-v2 launcher | Current DSpark target/draft CUDA graphs capture directly and the independently restarted default clears all throughput and behavior gates |
| Scheduling | Receive interval 4; stream interval 4; incremental output | Measured fixed-work wins while retaining client streaming behavior |
| Codex client lane | Five FP32 Mamba slots are now the one-request launcher default; close failed Qwen sessions before restart | Four slots failed the unfinished multi-chunk donation boundary. Five passed exact 200K, OpenCode2, and repeated 9.7K-cached-token Codex Responses turns without weakening either token pool |
| Codex Code Mode ABI | Preserve the synthetic one-string Qwen adapter and custom call/output restoration | Codex 0.152.0 twice executed exactly one read-only command with complete output and exact `CODEX TOOL READY`; the retained retry accurately recognized the nonempty status, preserved Tombstead, and left no process |
| Local Codex audit instructions | Select bounded-evidence `C:\Users\Daniel\.codex\qwen38.md` only for the `qwen38` profile; retain `xhigh` reasoning and the full data-safety contract | Exact Tombstead audit baseline **472.504487 s**; independent selected-prompt samples **232.5230228/260.7988227 s**, mean **246.66092275 s** (**47.7971%** lower), with all changed files triaged, green static gate, independently supported verdict, explicit gaps, and byte-identical user work |
| Selected upstream hardening backports | Pin compressed-tensors 0.18.0; strip credentials from executable PR checkouts; reject NUL grammars; bound stop inputs; resolve Qwen tool properties across schema combinators | Explicitly authorized exception for the reviewed Python correctness changes. Consolidated host coverage passes 199 tests/28 subtests; live malformed requests fail closed, `oneOf` arguments retain integer/boolean types, ordinary multiply and multi-chunk Codex Code Mode remain qualified |
| Process-tree teardown | Wait up to 60 seconds for ordinary kill/reap; use explicit fire-and-forget only for the runtime/GC route; reap tokenizer descendants before tokenizer exit | Semantic adaptation of upstream `78d36f5f62`; focused race, ordering, and caller-policy coverage passes, and a production launch released every verified server PID, listener, and CUDA resource cleanly |
| Idle tree-cache diagnostics | Default the deep tree walk off in production and on in CI through `SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK`; preserve explicit override priority | Semantic adaptation of upstream `6afb5e1771`; ordinary idle pool/request checks remain active, SWA and Mamba opt-in coverage passes, and idle scheduler CPU stayed flat through a retained 199K cache |
| Single-rank token synchronization | Treat TP or attention-TP size one as an identity before the sampler's grammar/env collective | Native-Windows Gloo lacks the CUDA all-reduce registration in the installed PyTorch build. The guard preserves every multi-rank MIN reduction; isolated CUDA/Gloo and simultaneous parsed-tool/Codex Responses gates pass |
| Implementation language | C++/CUDA hot paths with thin Python integration | Daniel's explicit direction after the display-GPU incident; preserves graph capture and native dispatch |
| Benchmark client architecture | Portable framework-free C++23 on the CPU; serving process retains GPU ownership | HTTP, JSON, SSE, calibration, hashing, timing, and acceptance validation need host execution; a client CUDA context would perturb the display GPU and leave the Apple lane uncovered |
| Native benchmark promotion | Windows live parity passed; keep the qualified Python clients as scoreboard authority through the remaining Apple gate | C++23 host suites pass; Windows preserved explicit workloads, 38-/11-field schemas, deterministic artifacts, stochastic acceptance algebra, exact `199000+16`, and adjacent Python/native/Python timing |
| Tree/SWOR implementation | Retained as opt-in, production-ineligible infrastructure | A non-front unified-pool KV/compaction defect exists outside the measured static-pool route; current-config full cross-cycle parity is still required, and raw-composite SWOR RNG is invalid |
| Device-resident linear cycle | Retained opt-in; rejected for throughput | Exact-q dense-race and explicit-seed categorical forms reached 122.576 and 120.075 tok/s versus 124.775 matched control; ordinary scheduling remains selected |
| Geometry funding gate | Complete lattice; conservative lower >=215 TPS; strictly above measured frontier | Family rejection uses an impossible target-aware upper <=200 TPS; selected-tree gaps fail closed |
| Graph-tail work | Static target-graph DSpark K/V commit selected; remaining tail work closed absent a new measured gap | Target-to-draft gap fell **1.672 -> ~1.10 ms**, full cycle **18.248 -> 17.98 ms**, exact acceptance parity held, and production windows cleared 150 |
| Target attribution | Exact per-shape M/N/K plus overlap-aware exposure | M3 primary GEMMs occupy 12.360 ms on the terminal stream; aggregate residency alone overcounts alternate-stream overlap |
| Selective target NVFP4 | Launcher-default production checkpoint | User accepted the all-four-metric record; default relaunch and behavior/client gates passed |
| MiaAI-Lab vLLM recipe | Matched `199000+16` reproduction remains an information gate | Same checkpoint/GPU uses MTP-3, TurboQuant 4-bit KV, and patched full-graph K+1 verify; published ~160 TPS lacks raw workload evidence |

## Active Apple Q5 choices

| Decision | Selected choice | Durable evidence |
|---|---|---|
| Literal Q5 control | Pinned text-only affine-five-bit/group-64 target at revision `2568951b...c2f05` | Exact shard hashes and the native 1,847-tensor inventory pass; sampled direct baseline is **16.322505765 tok/s** |
| Q5 bandwidth arm | Pinned mixed 4.951-bpw target at revision `596b8067...8340` | Exact required-language inventory passes; selected A100+A111+A114+A113+A117 reaches **19.453092367 tok/s** across two clean paired windows, **+7.347%** over generic mixed execution |
| Batch-one target kernel | PERF-A100 aligned-16-bit-load Q5 QMV | Representative parity and every production-shape digest pass. Two independent balanced windows improve A094 **19.012629776 -> 19.134907634 tok/s** in aggregate, **+0.643140%** |
| Mixed-target Q4 kernel | PERF-A111 stock-exact four-SIMD Q4 QMV | Representative output is bit-exact against MLX. Two independent balanced windows improve paired A100-only **19.113936088 -> 19.241981332 tok/s**, **+0.669905%**; the hand-inlined faster arithmetic form is rejected for digest drift |
| Mixed-target Q4 MLP fusion | PERF-A114 precise-exp 8-SIMD/four-pair gate/up/SwiGLU | A real-boundary trace and dedicated `-6.84375` regression prove exact staged BF16 behavior. Forward/reversed five-pair windows improve matched A100+A111 **19.228720305 -> 19.268407916 tok/s**, **+0.206398%**; retain behind its explicit switch |
| Scheduled-token scalar evaluation | PERF-A113 direct `array::item<int32_t>()` | MLX evaluates inside `item()`, making the immediately preceding explicit `eval()` redundant. Forward/reversed five-pair windows improve matched A114 **19.278381518 -> 19.289499778 tok/s**, **+0.057672%**, with every fixed-work output unchanged; signed `ad11696f2e` retains the deletion |
| Fused-Q4 MLP epilogue | PERF-A117 lane-parallel compile-time register selection | Lanes 0--3 execute the four exact precise-exp/SiLU/product chains after the unchanged reductions. Two clean paired windows improve matched serial A114+A113 **19.268475640 -> 19.453092367 tok/s**, **+0.958128%**; signed `00d09138ce` retains the source. Four-SIMD geometry, dynamic selection, and gate/up row interleaving remain closed by PERF-FA133/134/138 |
| Fused-Q4 explicit vector load | Reject PERF-A118 and retain A117's scalar packed-word reads | Exact production-shape timing regresses **0.557832737 -> 0.560992548 ms**. Reopen only with changed compiler evidence or a load shared across more arithmetic; see PERF-FA135 |
| Linear-attention Q5 `qkv`/`z` launch sharing | Reject concatenated/split A119 and direct-output A120/A121 | A119's isolated **0.426645998 -> 0.420592346 ms** win loses its full-model gate. Split-free four-SIMD A120 regresses **1.341171%** and exact-ratio 5:3/eight-SIMD A121 regresses **0.616515%**. Reopen only with shared staging or reduced weight-side work; see PERF-FA136/137 |
| Affine-Q5 result-row read-ahead | Reject four-row A123 and two-row A124 | A123 regresses the two largest K families; A124 regresses decisive K=17,408 **0.428486222 -> 0.429494958 ms**. Exact A100 remains selected; see PERF-FA139 |
| MTP contract order | Retain the signed post-norm seed correction; keep target-only selected | Mixed-target Q4-head screens reach at most **11.341033334 tok/s**; accepted width and verification cost leave the composition below target-only |
| MTP history staging | Decode-only accepted-prefix history before prompt-seeded history | Exact source trace proves the `cycle_base + 1` rollback/accepted-prefix pairing; prompt history adds a 512 MiB exact-capacity cache and a one-layer causal prefill pass |
| Additional Q5 downloads | Keep the pinned uniform and mixed artifacts | Current published uniform alternatives repeat affine-five-bit/group-64 text layout or add BF16 vision; the OptiQ candidate reports a larger 5.50-BPW allocation |
| Metal recovery boundary | Keep one sequential Metal workload and preserve foreground remote access | IT's restart restored custom and stock MLX execution. A later indexing/file-provider incident contaminated one timing tail; matched clean windows resumed after host activity settled |

## NVIDIA stock checkpoint alternative

The September 8 NVIDIA checkpoint is retained as an immutable, explicitly
selected **target-only 200K** option:
`-ModelPath C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-NVIDIA -SpeculativeNumSteps 0`.
Its five-run sampled mean is **64.064 tok/s**, versus **63.996 tok/s** for
the original stock RadixArk checkpoint with identical serving flags; the
0.11% difference does not establish a performance win. Exact `199000+16`,
ordinary reasoning/arithmetic, parsed tools and continuation, non-thinking
output, and language-only reporting pass.

Do not replace the attention-selective DSpark production default on this
evidence. NVIDIA speculative serving, independent-restart/client promotion,
and the published accuracy scores were not qualified. This preserves an
additional usable checkpoint without weakening the production capacity or
promotion contract. See `benchmark/windows/nvidia_qwen38_20260908.json`
and the September 8 experiment-log entries for the paired samples and hashes.

## Native backend foundation

| Decision | Selected choice | Boundary |
|---|---|---|
| Tensor descriptor ABI | Native ABI 1.0: 184-byte metadata, 192-byte borrowed const/mutable views, rank at most eight | Windows x64 little-endian; fixed-width C record; process-local and non-owning |
| Validation boundary | Total `noexcept` metadata validation, checked bit arithmetic, mutable non-overlap proof, then dtype/rank typed narrowing | No allocation, tensor dereference, logging, CUDA call, framework dependency, or unchecked pointer access |
| Framework isolation | Keep Python, PyTorch, ATen, c10, TVM, DLPack, FlashInfer, and TRT-LLM outside the core ABI | Future adapters translate explicitly and retain storage/lifetime ownership |
| CUDA execution context | Owned nonblocking stream plus copyable device-affine context lease | Current-device mismatch fails closed; stream destruction is busy while any context or graph executable remains |
| Graph-stable storage | One fixed device allocation, aligned slices, explicit seal, retained leases, and owner-authored tensor provenance | Slices cannot move after sealing; storage destruction is busy while slices, typed views, or graph executables remain |
| Graph executable lifetime | Retain stream context and arena lease through recorded completion; synchronize before teardown | Stable-address replay passed on the RTX 5090 after all external view and lease handles were dropped |
| First operator consumer | Batch-one native linear rejection sampling, 2-64 slots, exact contiguous graph-arena views, authoritative vocabulary 248,320 | Standalone C++/CUDA only; strict layout/device/alias checks and captured replay at the qualified shape pass |
| Sampler content safety | Preflight every proposal token and unique output index on device before token/count/index writes | Malformed content publishes only a structured device status; no unchecked probability/output index access |
| Sampler semantics | Strict `coin * q < p`; NaN q becomes zero only for residual; positive `p - q` on rejection; target p on full correctness; strict CDF and last-token zero-mass fallback | Matches the active linear Triton rule; 256 randomized host-oracle cases, hand cases, and slot boundaries pass |
| Native AOT spine | Independent Windows x64 CMake/Ninja targets for tensor ABI, CUDA resources, kernels, aggregate linkage, probe, CTest, and isolated memcheck | C++20/CUDA C++20, MSVC, exact CUDA 13.3 family, SM120, static internal linkage, no framework discovery, downloads, registry, or dynamic plugin ABI |
| Verify RNG state | Versioned 32-byte `SGLRNGV1` state; Philox4x32-10 with counter/subsequence as counter words and seed as key | Each replay reserves `ceil((slots + 1) / 4)` blocks; known vectors, reset, disjoint ranges, 2/64 boundaries, and overflow preservation pass |
| Request-seeded verify RNG | Exact four-block MurmurHash32 seed/position/column mapping and established FP64-to-FP32 half-open conversion | Bit-exact native CUDA parity across known hashes, request pairs, and slot boundaries; position intentionally uses its low 32 bits |
| Device composition status | Gated rejection launch consumes coins only when upstream `device_status == 0` | RNG descriptor/counter failures retain their namespaced status and cannot be overwritten by proposal validation or publish accepted output |
| First native composite | Captured initialization -> stateful RNG -> gated linear rejection at batch 1, slots 3, vocabulary 248,320 | Stable-address replay advances one block, deterministic reset reproduces the full result, and overflow leaves initialized output unpublished |
| Integration state | Dormant cumulative native libraries, linked probe, two operators, and one composite graph | No framework adapter, SRT/Python registration, launcher route, endpoint, model plan, or production graph; full qualification is still required before promotion |

## Primary performance record

The root [`../BENCHMARK.md`](../BENCHMARK.md) scoreboard governs optimization
ranking. The selected DSpark-v2 production scoreboard is uncached ordinary
sampling at exact `6213+512`. The explicit-switch full-pool window averaged
**155.961 tok/s**; an independent literal argument-free restart averaged
**162.500 tok/s**, and all five of its individual samples reached at least 150.

The argument-free server also repeated exact `199000+16` at **3118.215 prompt
tok/s**, **63.818572 s TTFT**, and **64.236571 s E2E**. The prior NEXTN
selective-target exact-16 record remains a separate historical long-context
control; its short-completion generation rate is not substituted for the
512-token ordinary-sampling objective. There is no active next Windows
milestone after closing the requested 150 tok/s gate.

## Qualified reference results

| Result | Accepted value |
|---|---|
| Argument-free real sampled `6213/512` | **162.500 tok/s** five-run mean; samples `151.139/165.951/151.000/191.357/153.054` |
| Independent explicit-switch sampled `6213/512` | **155.961 tok/s** five-run mean |
| Current native DSpark acceptance | **2.737968** accepted length, **0.249045** rate, 326/1309 correct/proposed over 187 verifies |
| Current exact `199000/16` capacity | **3118.215 prompt tok/s**, **63.818572 s TTFT**, **64.236571 s E2E**, exact `199016` total |
| Current graph/headroom envelope | **1.71 GiB** at graph end; **253-255 MiB** with retained benchmark prefixes; **1,361 MiB** after final cache flush |
| Historical NEXTN fixed accepted-length-3 `6213/512` | **171.263 tok/s** five-run mean |

## Closed or rejected candidates

### Checkpoints, quantization, and kernels

| Candidate | Status | Why |
|---|---|---|
| GGUF as production checkpoint | Superseded | Base NVFP4 improved prompt throughput by roughly 12.8x and E2E by 4.2x on `6213/128` |
| FlashInfer 0.6.11 attention | Rejected | Faster synthetic result produced degenerate repetition on the real long OpenCode prompt |
| Native target NVFP4 KV | Rejected | Recovered about 2.2 GiB but corrupted thinking and tool behavior |
| Native TurboQuant35 target KV | Rejected before serving integration | The retained harness pins TurboQuant35 segment 512 at both lengths and FP8 segments 512/2,048 at 6,213/199K. The codec saves a projected 3.242 GiB across 16 global layers, but has **0.199575** relative-L2 error and its selected 199K sweep median is **11,540.973 us** versus **3,074.226 us** for the fastest FP8 layout and **271.584 us** for the established XQA FP8 authority |
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

### Apple Silicon experimental lane

| Candidate | Status | Why |
|---|---|---|
| Bartowski Qwen3.8-27B Q5_K_S with one F16 token embedding | Active Q5 serving base; exact 131K BF16 pool performance-rejected | The immutable pinned Q5_K_S source hashes to `b52fbc24...e569`. Pinned llama.cpp build 10547 copies all 865 remaining tensors and converts only `token_embd.weight`; the 21,349,656,160-byte derivative hashes to `c05a7778...fcfb`, passes Q4_0/Q5_K/Q6_K native-MPS parity, and loads at 21.37 GB. With the selected Q6_K and Q5_K batch-one kernels it averages **7.1646 tok/s** across five sampled exact `128+32` requests in a 1K pool. An exact 131,072-token BF16 pool starts and serves, while its 8.00 GB cache drives paging and yields **0.102 tok/s**. The 20 tok/s floor remains active. |
| Bartowski Qwen3.8-27B Q5_K_S with one Q4_K token embedding | Retained smaller exact-capacity artifact | Pinned llama.cpp COPY changes exactly the 833.59 MiB token embedding to 682.03 MiB Q4_K and preserves the other 865 tensors. The 19,522,020,960-byte artifact hashes to `8ed3117a...b06ae`, loads at 20.00 GB, passes direct embedding parity, reasoning, arithmetic `703`, and parsed tools. Its exact-131K sampled decode improves **0.102 -> 0.315 tok/s** (**3.088235x**) while paging remains active. Use this artifact for compressed-KV capacity work. |
| MPS FP8 KV cache | Retained exact-capacity selection; further residency reduction active | PyTorch 2.11.0 stock value conversion raises its unsupported-MPS-dtype error, while byte-backed float8 views and indexed gathers work. PERF-A089 adds exact contiguous/strided FP32/E4M3FN conversion in the existing Metal extension and preserves all other `_to_copy` behavior through the original composite implementation. The exact 131K pool occupies 4.00 GB, leaves 6.99 GB by server accounting, and averages **3.237 tok/s** across five sampled requests: **10.276x** its BF16 control and **55.419%** below the 1K FP8 lane. Reasoning, arithmetic, and parsed tools pass. |
| `lukaskremla/Qwen3.8-27B-5bit-MLX-TextOnly` revision `2568951b...c2f05` | Retained immutable native-Q5 candidate | The 18,514,909,284-byte Hub inventory is affine five-bit/group-64, contains 498 U32 packed plus 1,349 BF16 tensors and no vision tensors, and derives from the official Qwen base. The compiled engine loads it directly. Sampled direct `128 / 32 / 128` reaches **16.322505765 tok/s**, **2.248x** the GGUF Q5 1K-FP8 mean. Served exact-131K and behavior qualification remain active; the unchanged Q4-tuned DFlash composition is closed at **11.506669050 tok/s** pending a five-bit verifier kernel. |
| Affine-Q5 M=8 K-split verifier | Retained behind the existing native small-batch/M8 controls | Exact MLX five-bit packing is decoded eight values from five bytes inside the shared SG16/B32 tile. Representative parity passes at maximum error **0.107422**. The direct DFlash composition improves **11.506669050 -> 13.721235888 tok/s** while M=8 falls from roughly **363--368 to 228--229 ms**; full-Q4 digest and width remain exact. Target-only Q5 stays selected pending a better proposal/cycle. |
| `lukaskremla/Qwen3.8-27B-MTP-5bit-MLX` revision `1faa5a80...3d85` | Retained immutable draft artifact and execution-cost probe; sampled production rejected | The 292,018,299-byte sidecar hashes to `f63dd5c2...e6e`. Deterministic blocks three/eight reach **17.919995235 / 31.380317186 tok/s** at widths **3 / 8**, preserving the target digest. Exact dense-q sampling peaks at only **6.380162135 tok/s**, width **2.285714286** across calibrated arms. Preserve exact p/q semantics and keep target-only Q5 selected. |
| Affine-Q5 batch-one direct QMV | A100 aligned-word source retained behind the existing explicit Q5 switch | Four SIMD groups reuse one 16-value activation fragment across four output rows. Five aligned 16-bit loads reconstruct two 32-bit windows plus a 16-bit tail while preserving the original sixteen FP32 FMAs. Representative parity and exact digests pass. Two clean balanced mixed-target windows aggregate to **19.134907634 tok/s**, **+0.643140%** over paired A094 and **0.865092366 tok/s** below the floor. Exact 131K serving and Codex gates remain due. |
| Affine-Q4 batch-one direct QMV | A111 stock-helper source retained behind the explicit Q4 switch | Four SIMD groups produce sixteen rows per threadgroup while preserving MLX v0.32.2's Q4 load/dot expression structure and bit-exact BF16 output. Two clean balanced mixed-target windows aggregate to **19.241981332 tok/s**, **+0.669905%** over paired A100-only and **0.758018668 tok/s** below the floor. Exact 131K serving and Codex gates remain due. |
| Affine-Q4 paired gate/up/SwiGLU | A114 precise-exp source plus A117 lane-parallel epilogue retained behind `SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU` | Eight SIMD groups produce 32 paired rows while sharing each activation load and preserving MLX's BF16 gate, up, sigmoid, SiLU, and output boundaries. A117 assigns the four precise epilogues to lanes 0--3 through compile-time register selection. Boundary parity and every fixed-work digest pass. Two clean A117 paired windows aggregate to **19.453092367 tok/s**, **+0.958128%** over serial A114+A113 and **0.546907633 tok/s** below the floor. |
| Concurrent affine-Q5 gate/up MLX streams | Rejected and removed | Building the two dependent projections under separate persistent streams retained about 14 GB and waited indefinitely on an unsignaled Metal event during its first full-model run. The exact benchmark process was terminated, all temporary stream code was removed, and sequential `Engine::mlp` ownership is restored. Reopen only with explicit cross-stream event/lifetime ownership and an isolated minimal proof. |
| Q6_K batch-one two-row activation reuse | Retained at the common native Metal GGUF matmul owner | Matched QKV/head medians improve **0.748917 -> 0.448125 ms** and **12.009166 -> 3.324625 ms**. Five served samples average **7.0522 tok/s**, **+22.732%** over the committed baseline. Optimized 16-row, generic 17-row, and untouched batch-eight actual-file parity pass. `SGLANG_MPS_Q6_K_BATCH1_ROWS2=0` retains the matched generic control. |
| Q5_K batch-one 32-row cohort | Retained at the common native Metal GGUF matmul owner for output sizes at least 5,120 | Four lanes cooperate on each row and each consume eight adjacent weights, doubling each 128-thread group's output cohort. Reversed-order served comparisons improve `128+32` mean **7.1342 -> 7.1646 tok/s** (**+0.426%**) and `128+128` median **7.456 -> 7.500 tok/s** (**+0.590%**). The 5,120-row floor is the smallest measured winning shape; the measured 1,024-row attention K/V shapes retain the prior eight-lane mapping. Direct candidate, tails, alignment fallback, and served reasoning parity pass. `SGLANG_MPS_Q5_K_BATCH1_ROWS32=0` retains the matched control. |
| Q6_K exact-batch-four 16-row activation reuse | Retained at the common native Metal GGUF matmul owner | Eight lanes vector-decode one row and reuse it across four verifier activations; four SIMD groups produce sixteen rows. Matched head/QKV medians improve **29.166000 -> 6.121375 ms** and **1.442333 -> 0.588500 ms**. Same-GGUF NEXTN serving improves **3.3384 -> 3.7028 tok/s** while its matched control has slightly higher acceptance. Candidate/tail and batch-three/eight fallback parity pass. `SGLANG_MPS_Q6_K_BATCH4_ROWS16=0` retains the generic control. |
| Same-GGUF Q5 three-step NEXTN | Closed as a complete serving selection; explicit GGUF draft loading retained as provenance | `--speculative-draft-model-quantization gguf` reduces the trained draft from a failed 10.62 GB BF16 load to 1.00 GB and serves correctly. Five sampled `128+128` requests average **3.7028 tok/s**, **50.629%** below the selected **7.500 tok/s** target-only median. Reopen with a materially faster/smaller draft or target-verification topology; PERF-A085 independently retains the Q6_K batch-four win. |
| Unchanged Bartowski Q5_K_M and Q5_K_S artifacts | Closed on the current native Metal GGUF surface | Q5_K_M fails while processing mixed merged Q8_0 weights. Q5_K_S loads its 20.00 GB weights and reaches the unsupported Q5_K token-embedding path during warmup. Reopen an unchanged source artifact when its exact native execution boundary gains support. |
| Bartowski Qwen3.8-27B IQ2_XXS checkpoint | Retained native playground | The current official-tokenizer/Python-ingress baseline reaches **9.189086 tok/s** aggregate on exact `12+256`, passes sampled behavior and tool continuity, completes exact `32761+1` in the 32K BF16 pool, and passes a strict Codex 0.151.0 scratch edit from a dedicated default/exec-low-reasoning `CODEX_HOME`; the intermediate medium hashes separately retain forced-compaction continuation evidence |
| Two-minute Apple Qwen/Codex actual-work gate | Parser-enabled xhigh shell round trip qualified at real 131K; autonomous multi-file qualification pending | Keep the normal trusted-repository prompt and hash-pinned isolated home. Wrap every non-interactive attempt with GNU `timeout --signal=INT --kill-after=10s 120s` in process-group mode. With the unchanged 6,232-token prompt primed through a 5,952-token aligned checkpoint, Codex completed one exact `exec_command` plus final response in about 91.6 seconds at 12,678 input / 12,328 cached / 188 output / 131 reasoning-output tokens. The actual Responses body remained uncapped. Cold 6K prefill and automatic multi-file ownership remain open; prompt-elision flags remain diagnostic-only overrides |
| Native DFlash2 | Retained; signed `d57a6ac11c` plus later verifier optimizations | The exact affine-W4 draft, learned selector, exact p/q verifier, accepted-state replay, real 131K pools, and one Codex `xhigh` tool turn pass. Five identity-temperature controls average **15.4424 tok/s** on sampled `6237+128`; proposal acceptance remains the active floor gap |
| DFlash2 selector temperature | Retained opt-in at **1.15**; identity remains default | Five consecutive exact real-131K-pool `6237+128` samples average **15.8866 tok/s**, **+2.876%** over the adjacent identity mean, with exact token counts and stable coherent output. Exact rescaled q flows into rejection sampling. Temperature 0.95 is closed by its **13.739 tok/s** real-request regression |
| DFlash2 selected-q M=2/M=8 budget | Retained opt-in at mean-q6 threshold **0.62**; full M=8 remains default | Five exact real-131K-pool `6237+128` samples average **16.1776 tok/s**, **+1.763%** over the adjacent current-source **15.8974 tok/s** control. Each request selects 23 M=2 and 19 M=8 cycles and preserves exact sparse-q rejection. Thresholds 0.55/0.65 are closed by **15.191 / 16.056 tok/s** admission screens. The retained arm remains **3.8224 tok/s** below the floor |
| Full-Q4 target-only internal prefill | Retained opt-in at **2,048 tokens**; one-shot remains default | The one-shot 6,237-token request exhausts Metal residency. Internal target-only chunks complete direct prefill and exact real-131K serving at **19.300 tok/s**, while a single-chunk sampled control preserves its exact digest above 20. DFlash2/DSpark capture and ordinary MTP behavior remain unchanged; decode retains a **0.700 tok/s** served gap |
| One-SIMD-per-output affine-W4 batch-one QMV | Rejected and removed | Exact standalone BF16 parity passes, while direct full-Q4 target-only throughput falls **20.339874670 -> 10.456143330 tok/s** with independent parameter loads and to **7.773375044 tok/s** with eight-lane scale/bias broadcasts. A future reopening requires matrix tiling or a fused consumer with isolated real-tensor evidence |
| Native DSpark v2 | Retained execution base; signed `21cd561dfc` plus trained-confidence telemetry | Official BF16 and derived affine-W4 checkpoints run through five YaRN full-attention layers, rank-256 Markov correction, and the shared exact verifier. Affine-W4 reaches **10.050625 tok/s** direct and **11.242 tok/s** on the representative real-131K-pool request. Trace-only confidence preserves exact trajectories and supplies seven FP32 survival probabilities for cost-based scheduling |
| Fixed shortened DSpark verification | Rejected as production policy; bounded verifier retained | Counts one through six reach at most **11.920231 tok/s** direct. Target M=6/M=7 are especially slow through the generic affine-QMM route, while M=8 uses SG16/B32. Adaptive scheduling may choose only between measured M=2 and M=8 tiers |
| Trained-confidence DSpark M=2/M=8 budget | Retained opt-in at complete-cycle ratio **1.75**; fixed M=8 remains default | Five consecutive exact real-131K-pool `6237+128` samples average **13.6058 tok/s**, **+20.267%** over the adjacent **11.313 tok/s** control, with one shared reasoning/output digest. Ratio 1.4 is closed by its **10.916 tok/s** real-request regression. The selected adaptive result remains below DFlash2 and the 20 tok/s floor |
| DSpark low-budget probe cooldown | Retained opt-in at **16** target-only refills; default disabled | First-five real samples average **16.6396 tok/s**, including one retained transiently contended sample; five normal samples within six requests average **17.0514**. Every request preserves exact token counts and one reasoning/output digest. Accepted-width triggers and fixed cooldowns 4/8/32 are closed; a cheap pre-draft signal is required for the remaining floor gap |
| DSpark target-state score | Retained as trace-only telemetry; pre-draft threshold scheduler rejected | Threshold 0.525 reaches **33.222 / 32.788 tok/s** directly and only **13.652 tok/s** on the representative real request. On a no-policy natural trace, M=2/M=8 score ranges overlap and score versus accepted width has Pearson **0.141**. The threshold control and scheduling branch were removed |
| DFlash2 with the QKV-restored target | Rejected unchanged | QKV-only and complete recurrent Q4 overrides reach **10.130 / 7.841 tok/s** directly, with accepted widths **2.653 / 1.984** versus full-Q4 DFlash2's **31.318 tok/s** and width **6.684**. A target-matched draft is required |
| Two-phase 64-column M8 affine tile | Rejected and removed | Exact DFlash2 width and digest remain unchanged, while direct throughput falls **31.318 -> 8.961 tok/s** from register/occupancy pressure and reduced grid parallelism. SG16/B32 remains selected |
| M=8 gate/up fused or paired dispatch | Rejected and removed | Sequential exact SwiGLU fusion adds about **3.3 ms** to verification. A bit-exact two-plane launch changes five-sample direct mean only **26.943319961 -> 26.964966176 tok/s** (**+0.08034%**) with two flat/slower pairs; separate SG16/B32 products remain selected |
| Independently top-k/top-p-filtered DSpark q | Rejected | Direct throughput fell **10.050625 -> 6.997461 tok/s** and width **2.428571 -> 1.6** because independently filtered draft and target supports differ |
| BF16 DSpark Markov W2 inside affine-W4 draft | Rejected | Direct throughput fell **10.050625 -> 7.044992 tok/s** and width **2.428571 -> 1.65**. The real request moved **11.242 -> 11.294 tok/s** on a different trajectory while artifact size rose **87.148 MiB** |
| M1 Max split-history BF16 decode | Retained; signed `5f966ecb0d` | The bounded native Metal split/reduce path changes exact 131K attention decode from 148.078959 ms online and 65.117542 ms unsplit tiled to 4.338625 ms, preserves short-sequence cost, and passes long-context parity plus asynchronous lifetime coverage. `SGLANG_MPS_TILED_DECODE=0` and `SGLANG_MPS_SPLIT_DECODE=0` retain matched controls |
| Materialized native affine gate/up rows | Rejected | Adjacent deterministic `6237+128` serving changed **18.845 -> 18.782 tok/s** and startup-reported available unified memory **28.92 -> 22.28 GB** with identical output. Reopen only for a one-launch kernel that consumes the two original tensors without duplicated storage |
| Native linear-attention b/a row fusion | Rejected | Combining the two 48-row affine projections changed adjacent deterministic `6237+128` serving **18.845 -> 18.511 tok/s** with identical output and retained memory headroom; keep the separately scheduled MLX operations |
| Native recurrent convolution/state fusion | Retained; signed `6ad2c58921` | One decode launch now owns the exact BF16 four-tap causal convolution and distinct shifted next-state output across all 48 recurrent layers. Five exact direct long-history pairs improve **19.120031 -> 19.221589 tok/s** (+0.531%); matched five-sample 131K serving improves **18.7616 -> 18.8914 tok/s** (+0.692%) with identical output/reasoning SHA-256 |
| Native residual/RMSNorm fusion | Retained; signed `4905d68370` | A dual-output Metal kernel owns 127 single-token residual/normalization boundaries while preserving distinct residual storage. Direct long-history improves **19.222533 -> 19.310857 tok/s** (+0.459%); matched five-sample 131K serving improves **18.8286 -> 19.0484 tok/s** (+1.167%) with exact reasoning output and digest |
| Native recurrent q/k normalization fusion | Retained; signed `b851d3c9de` | One dual-output Metal launch owns the two RMS reductions, BF16 normalization rounding, and float32 scales repeated by all 48 gated-delta layers. Direct long-history improves **19.319062 -> 19.464738 tok/s** (+0.754%); matched five-sample 131K serving improves **19.0900 -> 19.1610 tok/s** (+0.372%) with exact reasoning output and digest |
| Native recurrent output norm/gate fusion | Retained; signed `28174b3da2` | One Metal launch owns recurrent-output RMS normalization, precise float32 SiLU gating, and final BF16 conversion in all 48 gated-delta layers. Direct long-history improves **19.469136 -> 19.515718 tok/s** (+0.239%); matched five-sample 131K serving improves **19.1260 -> 19.1548 tok/s** (+0.151%) with exact reasoning output and digest. Extreme-activation parity requires `metal::precise::exp` in the dynamic custom kernel |
| Native recurrent convolution/SiLU fusion | Retained; signed `4c1bc4c1e3` | The existing single-token convolution/state Metal owner now reproduces both BF16 boundaries of the following SiLU across all 48 gated-delta layers. Direct long-history improves **19.524068 -> 19.632483 tok/s** (+0.555%); matched five-sample 131K serving improves **19.1730 -> 19.2134 tok/s** (+0.211%) with exact reasoning output and digest |
| Full-attention affine q/k/v row concatenation | Rejected | Full q/k/v and k/v-only forms screened at **20.721378** and **20.672271 tok/s**, yet both changed the selected 256-token digest because production output geometry changes MLX accumulation. Reopen only with a multi-output implementation that preserves each product's accumulation order |
| Recurrent beta/decay inside q/k normalization | Rejected | Beta-only fusion changed matched long-history decode **19.602405 -> 19.600901 tok/s**; the complete exact owner changed **19.641655 -> 19.547612 tok/s**, with every final pair slower. Preserve MLX's independent scheduling unless a downstream consumer absorbs these transforms |
| Compact heterogeneous merged GGUF storage on MPS | Retained | Signed `13bea403d6` removes 40 packed copies / 478.125 MiB per forward, improves adjacent `128+32` generation **3.1858 -> 3.309 tok/s**, lowers reported weights **10.03 -> 9.03 GB**, and preserves the exact digest |
| IQ2_XXS batch-one four-row Metal kernel | Retained | Signed `16b2bf7a06` changes matched projection time **1.176875 -> 0.516000 ms** and served generation **3.309 -> 7.1748 tok/s** with exact behavior across two restarts |
| Q5_K batch-one four-cohort vocabulary head | Retained | Signed `b19cf4acf3` changes matched head time **19.659291 -> 3.754625 ms**; served deterministic generation reaches **8.0284 tok/s**, with an independent 8.114 tok/s confirmation and exact digest |
| F32 batch-one native MPS projection | Retained | Signed `4d1641fdcd` changes the 48-layer actual b/a sweep **7.296667 -> 2.159000/2.051708 ms**; deterministic generation reaches **8.4406 tok/s** and sampled restart windows average **8.3094/8.2942 tok/s** |
| Lower-right torch-native partial extend | Retained | Signed `210a214c12` submits only new query rows. Exact MPS `4096+4096` source medians change **542.376416/641.256125 -> 176.066500 ms**; focused causal, ragged, sliding, noncausal, and empty-extend tests pass |
| Native MPS decode capability gate | Retained | Signed `b2b8ab4af8` sends BF16 or more than 7,936 physical cache rows through the established cache-write plus SDPA fallback. BF16/32,769 and FP32/7,937 report zero observed error; eligible FP32/7,936 remains fused |
| IQ2_XXS Apple7 large-batch SIMD-matrix kernel | Retained | PERF-A014 changes actual `17408x5120` medians **70.074833 -> 4.250250/4.277125 ms** at batch 128 and **1971.539875 -> 124.838125 ms** at batch 4096. Served exact-`128+1` prompt improves **7.0234 -> 22.8814 tok/s** across a matched disabled control and two independent default windows. Exact `4096+2` completes inside the former watchdog; candidate/fallback tails and all behavior gates pass |
| Q4_K batch-one two-row reuse inside the retained IQ2_XXS checkpoint | Retained | Signed `52b5326d8e` specializes aligned complete-cohort Q4_K tensors within the mixed-format Q2 artifact. Final Python A/B is **7.009167 -> 8.586948 tok/s** (**+22.510241%**), the independent window is **8.578205 tok/s**, actual-file/tail parity passes, and record standing remains the M1 Max Q2 lane |
| Q2_K batch-one four-row reuse inside the retained IQ2_XXS checkpoint | Retained; current Apple baseline | Signed `4dfa1ad3ef` specializes aligned complete-cohort Q2_K tensors at the common Metal matmul owner. Actual gate/down medians fall from about **1.07/1.09 ms** to **0.455/0.454 ms**; matched exact-`12+256` generation changes **8.515065 -> 9.156475 tok/s**, and the independent window reaches **9.189086 tok/s**. Exact streaming Prompt/Generation/TTFT/E2E and current-source `32761+1` capacity are recorded in root `BENCHMARK.md` |
| Direct Metal BF16 matrix-load barrier elision | Rejected | PV-only, QK-only, combined, leading-only, and trailing-only `mem_none` removals were exact but moved the two primary raw-kernel medians by less than **0.720%**; the restored PERF-A020 control was faster at the diagnostic shape. Reopen only for materially different generated code on another compiler/GPU family |
| F32 custom-kernel cross-row reuse | Rejected | Exact-shape medians `0.484833`, `0.504208`, and `0.556667 ms` all trail the selected one-row-per-SIMD custom control at `0.390083 ms`; native MPS matrix multiplication is faster still |
| IQ2 constant-table and four-SIMD/two-row ablations | Rejected | Constant-table windows `0.546042/0.576833 ms` and alternate-geometry windows `0.560625/0.550667 ms` trail the selected staged two-SIMD/four-row path around `0.523625 ms` |
| Pinned llama.cpp build 10547 IQ2 route | Current M1 Max Q2 reference | Exact five-run `12+256` aggregate is **14.661356 tok/s**, with a **14.671473 tok/s** best hit. The original native SGLang baseline is **7.001584 tok/s**; current PERF-A021 reaches **9.189086 tok/s**, leaving a **1.595518x** route gap |
| Thresholded MLX quantized-query tiling | Retained source mechanism; Mac Pro evidence only | The cross-machine measurements carry no M1 Max record standing. Fresh dependency, parity, memory, and capacity gates are required before M1 use |
| Always-on MLX quantized-query tiling | Rejected on its measured machine | The process-wide policy regressed the measured 5K prompt while only larger score shapes benefited |

Large-batch native-IQ2 prefill is qualified through exact `5000+1` and
`32761+1` requests. The Apple workspace-write client selection is Codex CLI
0.151.0 via the dedicated machine-local `qwen38-local-hardened-home`.
Its exact default-config unified-exec run applied one scratch patch on the first
attempt and completed at 2,670 input / 115 output / 39 reasoning-output tokens.
The shell policy directs spawned zsh startup to root-owned empty `/var/empty`.
The trusted repository prompt contains root `AGENTS.md`; qualification also
records the absence of auxiliary project config/hook/rule/override/skill
sidecars. Dedicated-home rule/skill/override paths, `$HOME/.agents/skills`, and
`/etc/codex/skills` were absent. The immediate medium-reasoning predecessor
hashes own the
separate forced recovery gate at 11,093/5,120/3,943. That unmatched task
recovered from a malformed
patch across two observed compaction boundaries, preserved the written nonce,
and exited zero; its raw JSONL and exact warning text were not retained. The
production 30,000-token Total-scope limit resolves to 29,491. Autonomous
multi-file ownership, parser-enabled required tools, parallel-tool behavior,
and production-threshold near-limit compaction remain wider gates.
The earlier 13,635/13,691-token process-scoped OpenCode runs remain historical
admission evidence. The safe long-pool fallback is retained, and the next
measured batch-one decode hotspot governs funding. The deleted affine-q4
scoreboard belonged to a separate Mac Pro experiment; current record standing
is M1 Max Q2. Any future MLX long-context route on this M1 Max begins with
fresh baseline, dependency, parity, memory, and capacity evidence.

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
- Preserve the historical `$CODEX_HOME/qwen38-local*` profile artifacts and
  their recorded hashes. Qualify the selected Apple client through the pinned
  `qwen38-local-hardened-home` config/catalog/instruction bundle, its fixed
  scratch-write Responses gate, and stable pre/post bundle/global hashes.
- Use exact process ancestry for server lifecycle actions and preserve every
  unrelated user process.

## Reopening criteria

The exhaustive native-Windows NVFP4 goal is complete. The Apple Silicon lane
is active under the newer explicit request. A closed branch reopens when an
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
