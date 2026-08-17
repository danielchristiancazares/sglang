# Failed Paths

Read this ledger before choosing a performance candidate. Reopen an entry only
when the stated premise changes and record the new evidence that changed it.

## PERF-F001 - Full online FP8 MTP

- Hypothesis: quantizing every dense MTP projection, including the 10240 -> 5120 fusion projection, would lower draft cost and raise the fixed-work ceiling.
- Scope: explicit online FP8 draft quantization, Windows registration/routing, Qwen MTP fusion projection, and FlashInfer FP8 GEMM.
- Attempted change: made explicit draft FP8 survive mixed-checkpoint detection and converted the complete MTP projection set.
- Benchmark evidence: fixed warmup `161.164`; fresh `163.647, 167.825, 167.836, 168.026, 167.781 tok/s`; mean **167.023** from a **171.263** BF16 control.
- Correctness evidence: full 200K capture passed, the effective mechanism logged FP8, and the deterministic digest remained exact.
- Failure mode: dynamic activation quantization and small-M FP8 GEMMs cost more than the weight/GEMM saving.
- Why not to retry unchanged: the all-accepted ceiling fell about 2.0%.
- Reopen only if: activation quantization is fused/reused or a new exact-shape FP8 kernel wins the complete draft graph.
- Related commit or revert: optional mechanism retained; production default remains BF16 MTP.

## PERF-F002 - Full online MXFP8 MTP

- Hypothesis: Blackwell MXFP8 weights and native CUDA activation quantization would reduce MTP time and memory.
- Scope: FlashInfer native MXFP8 conversion, CUTLASS dispatch, complete MTP projections.
- Attempted change: repaired Windows weight conversion and exact dynamic-activation dispatch, then captured the full 200K server.
- Benchmark evidence: `162.788, 162.959, 169.168, 162.203, 160.169 tok/s`; mean **163.457** fixed.
- Correctness evidence: native quantization/GEMM tests passed, full capture succeeded, and the deterministic digest remained exact.
- Failure mode: conversion and small-M MXFP8 execution lowered the fixed ceiling by 4.56%.
- Why not to retry unchanged: a sampled window cannot repair a lower all-accepted ceiling enough to reach the goal.
- Reopen only if: a new backend removes the measured dynamic-quantization and small-M penalty.
- Related commit or revert: optional memory-saving mechanism retained; inactive by default.

## PERF-F003 - Dense online NVFP4 MTP

- Hypothesis: dense NVFP4 MTP weights would exploit Blackwell FP4 throughput and save enough residency for deeper speculation.
- Scope: dense ModelOpt FP4 apply path, online conversion, graph replay, complete MTP weights.
- Attempted change: repaired exclusion/routing and qualified dense online NVFP4 end to end.
- Benchmark evidence: warmup `165.254`; fresh `164.693, 161.478, 164.831, 164.902, 164.566 tok/s`; mean **164.094**.
- Correctness evidence: exact digest and full graph replay passed; approximately 0.46 GiB draft memory was saved.
- Failure mode: per-step dense NVFP4 execution was slower than checkpoint-native BF16 MTP.
- Why not to retry unchanged: deeper recursion multiplies the slower per-depth cost.
- Reopen only if: memory is the binding blocker for an oracle-qualified deeper topology or an exact-shape FP4 kernel changes the measured cost.
- Related commit or revert: mechanism retained as opt-in capacity infrastructure.

## PERF-F004 - Static honest three-step linear speculation

- Hypothesis: a fourth emitted slot would amortize target verification enough to cross 200 TPS.
- Scope: three MTP steps, four target rows, ordinary aligned-q rejection sampling.
- Attempted change: captured and measured the unsimulated three-step topology after a fixed full-accept control crossed 200 once.
- Benchmark evidence: sampled `111.626, 118.145, 123.111, 115.037, 118.276 tok/s`; mean **117.239**. Accepted length was **2.403756**, and the third draft survived on 59/213 cycles.
- Correctness evidence: full 200K capture and exact rejection sampling remained active.
- Failure mode: the third draft pass executes every cycle while useful third-token acceptance is sparse.
- Why not to retry unchanged: static cost exceeds its ordinary sampled yield.
- Reopen only if: proposal quality materially raises third-token survival or a graph-safe controller proves a profitable conditional schedule.
- Related commit or revert: topology left inactive.

## PERF-F005 - Adaptive two/three-step controller

- Hypothesis: select the extra draft depth only on cycles likely to accept it.
- Scope: adaptive graph selection and history/confidence policy.
- Attempted change: qualified aggressive and sparse 2/3 controllers after repairing adaptive shared-output sizing.
- Benchmark evidence: both measured adaptive policies regressed the selected static two-step line; exact samples remain in `notes/experiment-log.md` under the 11:00-11:12 entries.
- Correctness evidence: the mechanism ran after the graph-sizing repair.
- Failure mode: the available policy signal did not predict useful third-step work well enough to repay switching and extra execution.
- Why not to retry unchanged: history-only/sparse policies already lost under matched controls.
- Reopen only if: a new device-resident confidence signal is shown to predict third-draft success with a costed oracle.
- Related commit or revert: retained as inactive experimental infrastructure.

## PERF-F006 - Draft proposal top-k 8

- Hypothesis: narrower q support would reduce proposal work while retaining acceptance.
- Scope: two-step aligned-q production topology.
- Attempted change: changed draft proposal top-k from 20 to 8.
- Benchmark evidence: accepted length **2.235808**; real samples `128.028, 115.179, 116.016`; mean **119.741** from a 122.712 qualified baseline.
- Correctness evidence: ordinary real rejection sampling remained active.
- Failure mode: reduced q support lowered useful acceptance.
- Why not to retry unchanged: both acceptance and mean throughput lost.
- Reopen only if: a new q distribution changes support mass enough to alter the measured overlap.
- Related commit or revert: launcher default remains 20.

## PERF-F007 - Stock Gittensor RTX5090 checkpoint

- Hypothesis: a smaller pure ModelOpt FP4 target with better proposal overlap would beat RadixArk end to end.
- Scope: immutable Gittensor checkpoint, same 200K/two-step workload.
- Attempted change: registered the Windows ModelOpt FP4 loader and ran fixed plus real windows.
- Benchmark evidence: fixed mean **154.883**; real ten-run mean **119.092**, median **119.364**; accepted length **2.403756**.
- Correctness evidence: exact 200K graphs captured and bundled BF16 MTP loaded.
- Failure mode: BF16 lm_head and broader exclusions lowered target execution enough to erase the acceptance gain.
- Why not to retry unchanged: both fixed and real means trail RadixArk.
- Reopen only if: a derived immutable-provenance checkpoint selectively fixes the target head/exclusions and passes the full behavior contract.
- Related commit or revert: source checkpoint remains immutable; RadixArk restored.

## PERF-F008 - Full online FP8, MXFP8, or NVFP4 draft quantization as a class

- Hypothesis: weight dtype alone can carry 122.712 real TPS to 200.
- Scope: all three qualified online draft quantizers.
- Attempted change: measured every mechanism through full MTP conversion and fixed work.
- Benchmark evidence: FP8 **167.023**, MXFP8 **163.457**, NVFP4 **164.094**, each below BF16 **171.263**.
- Correctness evidence: all three mechanisms reached exact graph execution and retained deterministic output where measured.
- Failure mode: activation conversion and exact small-M shapes dominate weight-format savings.
- Why not to retry unchanged: the class has a complete fixed-work comparison.
- Reopen only if: a fused activation/weight kernel or new batch geometry changes the actual graph cost.
- Related commit or revert: optional mechanisms retained, all inactive by default.

## PERF-F009 - Target-only width sweep M8/M12/M16

- Hypothesis: fixed-width breadth would raise emitted tokens faster than target cost.
- Scope: four-step, top-k-four target-only trees after two-graph and sparse-GDN work.
- Attempted change: measured M8, M12, and M16 with five acceptance and five real samples per selected shape.
- Benchmark evidence: emitted tokens/cycle **2.737, 2.906, 3.061**; real means **97.352, 94.685, 92.831 tok/s**. M12 corrected raw values were `87.870, 101.393, 96.121, 98.484, 89.557`.
- Correctness evidence: every retained request returned 512 tokens; corrected graph capture remained stable.
- Failure mode: modest yield growth is outweighed by target width and WDDM-sensitive cycle cost.
- Why not to retry unchanged: the post-optimization width curve is complete and flat-to-regressive.
- Reopen only if: proposal overlap or per-node target cost changes materially.
- Related commit or revert: no topology promoted; traces retained as evidence.

## PERF-F010 - Fully normalized aligned target-tree scoring

- Hypothesis: applying target sampling transforms to tree allocation would spend nodes on candidates the verifier is likely to accept.
- Scope: M12 target-only tree scoring with temperature, penalties, top-k 20, and top-p.
- Attempted change: compared aligned and plain scoring over long acceptance windows.
- Benchmark evidence: aligned emitted **2.8456** per traversal; plain emitted **2.9436**, a roughly 3.45% yield loss.
- Correctness evidence: target-only verification remained exact under either candidate allocation.
- Failure mode: globally repeated penalty/normalization state distorted deeper branch allocation.
- Why not to retry unchanged: the matched acceptance window rejects global full normalization.
- Reopen only if: penalties are made branch-local and an offline p/q oracle projects a gain.
- Related commit or revert: aligned scoring retained only as an explicit experiment.

## PERF-F011 - Scalar tree depth discount 0.8

- Hypothesis: discounting deeper allocation scores would buy more root breadth.
- Scope: M12 target-only node ranking.
- Attempted change: multiplied only final global allocation scores by `0.8**depth`.
- Benchmark evidence: **2.9286** emitted/traversal from four windows, below plain **2.9436**.
- Correctness evidence: candidate tokens and exact verification were unchanged; an allocation-induced graph-memory issue was repaired by in-place score mutation.
- Failure mode: one scalar cannot represent branch/rank value.
- Why not to retry unchanged: yield was lower after a long window.
- Reopen only if: a measured branch-local oracle supplies non-scalar allocation weights.
- Related commit or revert: option remains inactive by default.

## PERF-F012 - Initial fixed M12 exact SWOR topology

- Hypothesis: ordered q proposals without replacement would improve sibling coverage enough to repay exact p/q verification.
- Scope: initial 12-node 4/4/2/1 topology and sparse exact verifier.
- Attempted change: implemented exact SWOR generation/verification and measured three real runs.
- Benchmark evidence: **2.9653 emitted/traversal**; `88.284, 82.811, 83.043 tok/s`; mean **84.713**.
- Correctness evidence: native sparse verifier, dense fallback, and path commit passed focused CUDA tests.
- Failure mode: node allocation over-spent root siblings, later branch continuation remained sparse, and exact verification added cost.
- Why not to retry unchanged: both yield and throughput miss the linear baseline materially.
- Reopen only if: a new topology and proposal distribution pass the offline 200 TPS plus margin gate.
- Related commit or revert: exact SWOR retained as opt-in infrastructure.

## PERF-F013 - Current-q topology-only SWOR search

- Hypothesis: an irregular deeper tree can reach 200 using the existing q.
- Scope: measured ordered sibling probabilities, fixed-tree beam search up to 32 nodes/depth nine.
- Attempted change: fit the topology optimizer to M12/M16 path data and searched optimistic decay assumptions.
- Benchmark evidence: realistic search reached **3.9800** expected outputs; deliberately optimistic no-decay reached **4.0921**. Cost-ranked winner remained near 100 predicted TPS.
- Correctness evidence: topology constraints enforce sibling prefixes, fixed frontiers, node/depth limits, and measured cycle costs.
- Failure mode: proposal overlap decays faster than extra nodes amortize target/draft work.
- Why not to retry unchanged: topology rearrangement with unchanged q is exhausted.
- Reopen only if: proposal overlap or per-depth cost changes enough to move the oracle projection above 200 plus explicit margin.
- Related commit or revert: optimizer and topologies retained.

## PERF-F014 - Scalar SWOR q temperature/support calibration

- Hypothesis: temperature scales `0.70..1.30` or retained q supports `4..20` would improve p/q overlap.
- Scope: native 16-node overlap grid over 669 real cycles.
- Attempted change: evaluated all 25 scale/support combinations at each target row without changing runtime q.
- Benchmark evidence: internal-node gains over scale 1/top-k20 were `0.000053, 0.000048, 0.000245, 0.000157, 0.000120, 0.000000`.
- Correctness evidence: exact SWOR remained active; request emitted **3.061286 tokens/cycle** with a retained accepted-node histogram.
- Failure mode: q mismatch is token/branch-conditional rather than a scalar sharpness/support error.
- Why not to retry unchanged: the exhaustive grid is flat.
- Reopen only if: a richer proposal transform is evaluated against sparse branch-local p/q rows.
- Related commit or revert: overlap oracle retained; no runtime calibration promoted.

## PERF-F015 - 232K production pool

- Hypothesis: larger context/pools could retain throughput while expanding useful capacity.
- Scope: real 232,000 target/draft pools and `231000+16` capacity request.
- Attempted change: captured and exercised the complete 232K production topology.
- Benchmark evidence: near-limit request passed, sampled 6213/512 measured 120.653, and only **98 MiB** remained before cache flush.
- Correctness evidence: exact 231016 total tokens completed.
- Failure mode: operating headroom was unsafe on the display GPU.
- Why not to retry unchanged: repeated serving/JIT activity needs more than 98 MiB margin.
- Reopen only if: model/graph residency falls enough to restore a measured safe margin.
- Related commit or revert: launcher defaults restored to 200K.

## PERF-F016 - Selective CUTLASS channelwise-FP8 dispatch

- Hypothesis: aligned channelwise-FP8 GEMM shapes would reduce dominant target/draft kernel time.
- Scope: repaired CUTLASS channelwise-FP8 dispatch and selected production shapes.
- Attempted change: qualified alignment/numerics, then ran robust paired measurements.
- Benchmark evidence: paired medians showed no material end-to-end win; exact samples are retained in `notes/experiment-log.md` around 07:49-08:07.
- Correctness evidence: alignment and numerical tests passed.
- Failure mode: dispatch/kernel savings did not survive full graph and scheduling cost.
- Why not to retry unchanged: robust matched reruns already rejected the same dispatch.
- Reopen only if: a new kernel schedule wins graph-specific attribution before server testing.
- Related commit or revert: candidate dispatch removed; tests/repairs retained where generally useful.

## PERF-F017 - FP8 GEMM autotuning

- Hypothesis: enabling FP8 tactics alongside FP4 autotuning would improve mixed-precision decode.
- Scope: FlashInfer FP8 autotuning and full mixed autotune.
- Attempted change: measured FP8-only and combined FP4+FP8 configurations.
- Benchmark evidence: FP8-only lost decode and long prefill while reducing memory headroom; full mixed tuning remained below FP4-only.
- Correctness evidence: serving remained functional during qualified runs.
- Failure mode: selected FP8 tactics and workspace pressure regressed the production workload.
- Why not to retry unchanged: FP4-only is the measured winner.
- Reopen only if: dependency tactics or exact hot shapes change.
- Related commit or revert: launcher skips FP8 GEMM autotuning.

## PERF-F018 - Unsafe reusable fixed-chain metadata outputs

- Hypothesis: reusing one captured metadata output buffer would remove allocation/copy cost.
- Scope: chain metadata CUDA graph outputs consumed asynchronously by later phases.
- Attempted change: reused captured outputs across cycles.
- Benchmark evidence: isolated/fixed work improved, then seeded real rejection sampling regressed and exposed changed scheduling/lifetime behavior.
- Correctness evidence: output content could be made exact while asynchronous ownership still differed.
- Failure mode: per-cycle outputs outlive launch replay; reusing storage violates the scheduler/consumer lifetime.
- Why not to retry unchanged: the hazard is architectural, not a missing synchronization in one call site.
- Reopen only if: ownership is redesigned with explicit ring-buffered per-cycle storage and a matched real-sampling proof.
- Related commit or revert: unsafe reuse removed; native metadata kernel retained with distinct outputs.

## PERF-F019 - FlashInfer 0.6.11 attention as production backend

- Hypothesis: the first native-Windows FlashInfer port would replace Triton attention with a faster path.
- Scope: FlashInfer 0.6.11 target attention on the real long OpenCode workload.
- Attempted change: enabled the synthetically faster backend and ran a long-prompt behavior A/B.
- Benchmark evidence: the synthetic timing improved, while the real long request produced degenerate repetition.
- Correctness evidence: long-prompt coherence failed, so timing did not qualify.
- Failure mode: the old port/backend combination was behaviorally incorrect for this Qwen workload.
- Why not to retry unchanged: production correctness is part of performance.
- Reopen only if: using the separately preserved clean 0.6.17 Windows port and its current qualification suite.
- Related commit or revert: 0.6.11 rejected; clean 0.6.17 became selected.

## PERF-F020 - Native target NVFP4 KV cache

- Hypothesis: target NVFP4 KV would recover roughly 2.2 GiB and allow more graph/speculative residency.
- Scope: target KV dtype and full reasoning/tool behavior.
- Attempted change: enabled native target NVFP4 KV and ran semantic probes.
- Benchmark evidence: memory was recovered as expected.
- Correctness evidence: thinking and tool behavior were corrupted.
- Failure mode: target KV quantization error crossed the functional boundary.
- Why not to retry unchanged: the selected model must retain coherent reasoning and tools.
- Reopen only if: a new KV format/kernel passes the complete semantic and exact-capacity gates.
- Related commit or revert: checkpoint-selected target KV restored.

## PERF-F021 - Chunked prefill size 8192

- Hypothesis: larger chunks would raise long-prompt ingestion throughput.
- Scope: 6213, 32K, 64K, and repeated memory-heavy prompts.
- Attempted change: raised the prefill chunk from 4096 to 8192.
- Benchmark evidence: short work lost, and repeated 32K requests collapsed under VRAM pressure.
- Correctness evidence: no durable production qualification survived the memory sequence.
- Failure mode: larger temporary/workspace residency erased the isolated prefill opportunity.
- Why not to retry unchanged: the production contract includes repeated work and 200K capacity.
- Reopen only if: graph/model residency falls and a complete context ladder shows safe headroom.
- Related commit or revert: 4096 remains selected.

## PERF-F022 - FlashInfer workspace 64 MiB

- Hypothesis: a smaller workspace would improve graph-end memory headroom.
- Scope: FlashInfer capture and the 200K production topology.
- Attempted change: lowered workspace below the selected 128 MiB value.
- Benchmark evidence: graph capture hit a deterministic buffer overflow at 64 MiB.
- Correctness evidence: startup could not complete the required graph set.
- Failure mode: 64 MiB is below the functional workspace floor.
- Why not to retry unchanged: the failure is deterministic for the captured shapes.
- Reopen only if: graph shapes/backends change and a fresh allocation proof establishes a lower floor.
- Related commit or revert: launcher default remains 128 MiB.

## PERF-F023 - One-step / two-row MTP

- Hypothesis: minimum draft cost would beat deeper speculation despite lower emission width.
- Scope: one MTP step and two target rows after the later XQA/graph improvements.
- Attempted change: reopened and remeasured the one-step topology under the evolved cost stack.
- Benchmark evidence: fixed samples remained near **102 tok/s**.
- Correctness evidence: the mechanism was functional.
- Failure mode: the two-token emission ceiling cannot amortize target execution.
- Why not to retry unchanged: even full acceptance stays far below the goal and the selected two-step line.
- Reopen only if: target cycle cost changes by a material fraction.
- Related commit or revert: two steps / three rows remains selected.

## PERF-F024 - Generic FlashInfer/PyTorch sampler swap

- Hypothesis: changing the generic sampling backend would reduce the active speculative sampling cost or memory.
- Scope: server sampling-backend selector under EAGLE rejection sampling.
- Attempted change: traced reachability and ran a full-200K fixed A/B.
- Benchmark evidence: paired fixed medians overlapped and no reproducible memory recovery appeared.
- Correctness evidence: both fallback samplers remained functional.
- Failure mode: steady EAGLE proposal/rejection bypassed the generic selector.
- Why not to retry unchanged: the requested selector does not own the measured hot path.
- Reopen only if: source reachability changes or the non-speculative fallback becomes the benchmark target.
- Related commit or revert: FlashInfer remains selected for the reachable fallback path.

## PERF-F025 - Continuous decode steps 4

- Hypothesis: more scheduler work per receive interval would lower host overhead.
- Scope: matched OpenAI streaming workload and scheduler cadence.
- Attempted change: set continuous decode steps to four.
- Benchmark evidence: TTFT and end-to-end latency increased on the matched control.
- Correctness evidence: responses remained functional.
- Failure mode: extra scheduler batching delayed visible progress without a compensating decode gain.
- Why not to retry unchanged: both client-facing latency metrics lost.
- Reopen only if: scheduler/streaming architecture changes enough to alter the cadence cost.
- Related commit or revert: selected receive/stream interval remains four without this continuous-step change.

## PERF-F026 - BF16 Mamba recurrent state

- Hypothesis: halving recurrent-state precision would lower residency and bandwidth.
- Scope: Qwen GDN/Mamba persistent state and deterministic output.
- Attempted change: changed persistent Mamba state from FP32 to BF16.
- Benchmark evidence: execution was slower.
- Correctness evidence: deterministic output changed.
- Failure mode: conversion/numerical effects erased memory benefits and crossed the behavior baseline.
- Why not to retry unchanged: it loses both speed and deterministic equivalence.
- Reopen only if: a dedicated BF16 state kernel plus semantic qualification changes both findings.
- Related commit or revert: FP32 state remains selected.

## PERF-F027 - Fully compiling repaired Triton kernels

- Hypothesis: repairing every compiler failure and forcing full compilation would remove eager/fallback overhead.
- Scope: target/draft Triton kernels, startup compilation, and steady decode.
- Attempted change: repaired compilation candidates and ran the fully compiled form.
- Benchmark evidence: steady execution was slower and startup compilation was very long.
- Correctness evidence: repaired kernels were correct.
- Failure mode: the compiler-selected segmentation/schedules were inferior to the established partial fallback mix.
- Why not to retry unchanged: correctness alone did not produce a runtime gain.
- Reopen only if: PyTorch/Triton versions or exact kernel schedules change.
- Related commit or revert: compile mode `default` with established fallbacks remains selected.

## PERF-F028 - Explicit compiler-disable boundaries

- Hypothesis: excluding known weak regions would improve graph segmentation and reduce compilation work.
- Scope: torch-compile boundaries around the hybrid Qwen model.
- Attempted change: introduced explicit disable regions and measured the resulting graphs.
- Benchmark evidence: throughput fell.
- Correctness evidence: execution remained functional.
- Failure mode: changed graph segmentation lost optimizations/fusion outside the intended region.
- Why not to retry unchanged: the boundary itself caused the regression.
- Reopen only if: a graph-specific trace identifies a new boundary with a costed mechanism.
- Related commit or revert: explicit boundaries removed; compile `default` retained.

## PERF-F029 - CUTLASS DSL / FlashInfer GDN on native Windows

- Hypothesis: Blackwell CuTe DSL GDN kernels would beat Triton recurrent attention.
- Scope: native Windows dependency and GDN backend selection.
- Attempted change: audited and attempted the available backend route.
- Benchmark evidence: no valid server benchmark could run because the required Windows DSL/package support was absent.
- Correctness evidence: unavailable dependency blocked qualification.
- Failure mode: `nvidia-cutlass-dsl`/required native support was outside the installed Windows stack.
- Why not to retry unchanged: the external condition is unchanged.
- Reopen only if: the dependency gains supported native-Windows installation and isolated parity passes.
- Related commit or revert: Triton GDN remains selected.

## PERF-F030 - NVML polling or GPU keepalive

- Hypothesis: periodic polling/work would retain higher clocks and stabilize late decode windows.
- Scope: display-GPU clock residency and WDDM environment.
- Attempted change: measured polling/keepalive behavior and investigated apparent gains.
- Benchmark evidence: gains failed to persist; competing WDDM clients explained the variance.
- Correctness evidence: model behavior was unaffected.
- Failure mode: the observed correlation came from desktop contention and residency, not a durable server optimization.
- Why not to retry unchanged: it adds background activity without a reproducible throughput win.
- Reopen only if: hardware clock control becomes available and matched uncontended windows prove causality.
- Related commit or revert: no keepalive retained.

## PERF-F031 - Old top-k-two tree route

- Hypothesis: two proposals per depth would raise acceptance over the linear chain.
- Scope: the pre-exact-tree native-Windows rejection/XQA/ReplaySSM path.
- Attempted change: enabled top-k two before exact recurrent tree verification existed.
- Benchmark evidence: no correct sampled production path was available in that architecture.
- Correctness evidence: rejection, XQA, and ReplaySSM contracts could not all be satisfied.
- Failure mode: the old implementation lacked exact tree sampling and recurrent-state commit.
- Why not to retry unchanged: later exact target-only/SWOR infrastructure supersedes this route.
- Reopen only if: evaluating the retained exact tree implementation under its oracle gate.
- Related commit or revert: old route rejected; exact tree machinery retained separately.

## PERF-F032 - Gittensor/RadixArk hybrid lm_head branch

- Hypothesis: combine Gittensor's smaller/better-overlap target with RadixArk's packed NVFP4 lm_head.
- Scope: a derived checkpoint with immutable source provenance.
- Attempted change: source layouts and the selective head opportunity were identified after Gittensor measurement.
- Benchmark evidence: stock Gittensor measured 119.092 real and 154.883 fixed; no hybrid benchmark was authorized in that branch.
- Correctness evidence: both source checkpoints remain immutable.
- Failure mode: the user closed the branch and restored RadixArk before a derived artifact was built.
- Why not to retry unchanged: this is a user-closed candidate, not an unmeasured default action.
- Reopen only if: the user explicitly reopens checkpoint derivation and the resulting artifact carries provenance/checksums plus full qualification.
- Related commit or revert: no hybrid artifact created.
