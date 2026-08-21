# Failed Paths

## PERF-001 - Q4_0 batch-eight accumulation tile

- Hypothesis: process all eight decode rows in one threadgroup Y tile so each packed Q4_0 row is loaded and dequantized once instead of twice.
- Scope: `q4_0_small_batch_impl` specialization, pipeline cache, and `q4_0_matmul` host selection in `python/sglang/kernels/aot/csrc/metal/gguf_q4_0.mm`.
- Attempted change: instantiated `RowsPerBatchTile=8` and selected it for batches above six.
- Benchmark evidence: representative production MLP weight `blk.8.ffn_gate.weight`, Q4_0 `(5120, 17408)`, batch eight. Existing tile-four median `0.345 ms` from raw `0.345`, `0.345`, `0.382 ms`. Tile-eight median `0.778 ms` from raw `0.440`, `0.801`, `0.778`, `0.798`, `0.440 ms`; `125.5%` slower.
- Correctness evidence: passed explicit dequantized CPU F32 reference at batch eight; maximum absolute error `8.34465e-07`, relative error `4.46011e-07`.
- Failure mode: eight per-thread accumulators and activation positions appear to reduce occupancy or spill enough to outweigh the eliminated weight traversal.
- Why not to retry unchanged: repeated warmed measurements showed no winning tile-eight sample; even its best `0.440 ms` result was `27.5%` slower than the existing median.
- Reopen only if: the shader is redesigned to reduce register/pointer pressure, such as a different rows-per-SIMD geometry or staged accumulations, and the same representative microbenchmark first beats `0.345 ms` repeatedly.
- Related commit or revert: no commit; candidate was removed before end-to-end server testing.

## Batch-24 Q4_0 wider reuse geometries

- The dormant eight-lane/four-subgroup batch-24 kernel regressed representative `blk.8.ffn_gate.weight` from `0.726 ms` to `1.630 ms`.
- A 16-lane/twelve-request subgroup variant regressed it to `2.916 ms`.
- Changing the production kernel's unroll from four chunks to two or eight produced `0.829 ms` and `0.897 ms`; using 16 or four lanes per output row produced `0.795 ms` and `1.186 ms`.
- All candidates were removed or restored. The existing eight-lane, four-row, four-chunk Q4_0 kernel remains the winner.

## Dense FP16 LM head through MPSGraph

- A dequantized FP16 `torch.mm` over 8,192 vocabulary rows took `18.908 ms`, extrapolating linearly to about `573 ms` for the 248,320-row head. FP32 was slower still.
- The retained custom Q6_K Metal kernel takes `21.565 ms` for the complete head, so dense MPSGraph execution is not viable on this Intel/AMD MPS stack.

## Q6_K exact-24 alternatives

- A 16-lane vec2 kernel took `23.753 ms`, slower than the retained eight-lane vec4 kernel at `21.565 ms`.
- Dequantizing once in one subgroup and distributing each `float4` with SIMD shuffles took `30.789 ms`; shuffle/divergence overhead outweighed the removed duplicate decode work.
- Both alternatives were removed.

## PERF-005 - Dormant full-attention preparation kernel

- Hypothesis: replacing Q/K Gemma normalization, partial RoPE, QKV/gate unpacking, and contiguity materialization with the existing single-dispatch Metal kernel would remove roughly two milliseconds from every full-attention layer at batch eight.
- Scope: `Qwen3_5AttentionDecoderLayer.self_attention`, `prepare_full_attention_f32`, and its MPS wrapper.
- Attempted change: routed float32, one-dimensional-position, gated MPS attention through the native preparation kernel while retaining the PyTorch path as an environment-controlled ablation.
- Benchmark evidence: isolated production-shape median improved from `2.225 ms` to `0.162 ms`, but clean warmed end-to-end median regressed from `32.309 TPS` (`32.434`, `32.309`, `32.269`) to `30.680 TPS` (`30.486`, `30.814`, `30.680`). A 128-token sample also regressed from `40.470` to `38.100 TPS`.
- Correctness evidence: native Q/K, V, and gate outputs matched the PyTorch reference at `rtol=2e-5`, `atol=2e-5`; downstream grouped-query attention retained maximum error `4.76837e-07`.
- Failure mode: synchronization around the isolated microbenchmark charged the asynchronous PyTorch/MPS chain more heavily than real serving does; the native dispatch reduced local synchronized latency while disrupting end-to-end command-stream throughput.
- Why not to retry unchanged: both the fixed 32-token acceptance workload and a longer 128-token workload favored the existing route.
- Reopen only if: the preparation work is fused directly into the grouped-query attention/cache-write kernel and an unsynchronized production trace shows a removable command-stream gap.
- Related commit or revert: no commit; production routing was removed.

## PERF-006 - Reusing Q5_K/Q6_K tiles at batch eight

- Hypothesis: lowering accumulator pressure with the existing four-request kernel, or borrowing the vectorized batch-24 Q6_K kernel, would improve batch-eight occupancy.
- Scope: quantized matmul dispatch in `gguf_q4_0.mm`.
- Attempted change: selected the tile-four Q5_K/Q6_K pipelines for batch eight, then separately selected the Q6_K batch-24 vec4 pipeline.
- Benchmark evidence: Q5_K regressed from `0.809 ms` to `0.888 ms`; Q6_K regressed from `27.227 ms` to `32.446 ms` with tile four and `34.166 ms` with the batch-24 vec4 kernel.
- Correctness evidence: the reused kernels retained existing quantized reference coverage.
- Failure mode: the tile-four path traversed weights twice, while the batch-24 geometry left unsuitable subgroup work and accumulator structure at batch eight.
- Why not to retry unchanged: all three representative medians were materially slower than the existing batch-eight specializations.
- Reopen only if: batch eight receives a dedicated vectorized subgroup geometry instead of reusing a kernel shaped for another batch.
- Related commit or revert: no commit; dispatch changes were removed.

## PERF-009 - Q4_0 batch-eight subgroup and unroll variants

- Hypothesis: the Q5_K/Q6_K two-request subgroup strategy, the existing batch-eight split kernel, or a different dequantization unroll depth would improve the heavily repeated Q4_0 projections.
- Scope: `q4_0_small_batch_impl`, `q4_0_batch_8_split`, an experimental four-subgroup vec4 kernel, and batch-eight host dispatch.
- Attempted change: measured four eight-lane subgroups with two requests each; selected the existing two-half split kernel; then changed `chunks_per_thread` from four to two and eight.
- Benchmark evidence: representative Q4_0 `(5120, 17408)` baseline `0.350 ms`; four-subgroup vec4 `0.686 ms`, two-half split `0.390 ms`, unroll two `0.402 ms`, and unroll eight `0.405 ms`.
- Correctness evidence: experimental code and dispatch changes were removed after the losing microbenchmarks; the previously validated production kernel remains selected.
- Failure mode: Q4_0's current kernel already vectorizes four weights, processes sixteen output rows per threadgroup, and balances four accumulators with four dequantized chunks. Subgroup variants duplicated packed-weight work or reduced row-level parallelism, while alternate unroll depths added loop overhead or register pressure.
- Why not to retry unchanged: every candidate was slower across repeated warmed measurements; the best alternative still regressed by `11.4%`.
- Reopen only if: a new kernel changes memory cooperation or uses matrix hardware instead of rearranging the same per-row SIMD work.
- Related commit or revert: no commit; all experimental Q4_0 changes were removed.

## PERF-010 - Fused GDN projection packing and decode convolution

- Hypothesis: merging the native GDN projection pack and causal-convolution decode kernels would remove one dispatch and the intermediate mixed-QKV write/read in every GDN layer.
- Scope: Metal GDN pack and causal-convolution kernels, MPS wrappers, and exact production-shape microbenchmark.
- Attempted change: one kernel copied z/b/a while directly applying the four-tap convolution, updating the indexed convolution state, applying SiLU, and writing the mixed QKV result.
- Benchmark evidence: alternating warmed batch-eight median was `0.147 ms` for the separate chain and `0.151 ms` for the fused kernel.
- Correctness evidence: mixed QKV, gate, a/b, and mutated convolution state matched the existing chain; the full native fused-op suite passed.
- Failure mode: the existing two dispatches are already small and asynchronous; combining them did not reduce completed GPU latency and was `2.7%` slower in the representative median.
- Why not to retry unchanged: the entire fused boundary was measured, including allocations and state mutation, and showed no local opportunity to carry into end-to-end serving.
- Reopen only if: packing can be eliminated across a larger boundary such as the input projection or recurrent core, with a directly measured reduction in completed GPU time.
- Related commit or revert: no commit; the experimental kernel, wrapper, test extension, and microbenchmark were removed.

## PERF-011 - Idle MPS request coalescing

- Hypothesis: a bounded idle delay would combine eight barrier-synchronized HTTP requests into one prefill and remove enough redundant prefill wall time to close the remaining throughput gap.
- Scope: MPS normal scheduler receive loop and one schedule CLI knob.
- Attempted change: after the first idle request, delayed 2 ms and drained the tokenizer queue once more; also measured a request that arrived prebatched with all eight sequences as an upper bound.
- Benchmark evidence: the delay changed the prefill batches from `1 + 7` to `4 + 4` and produced `37.876 TPS`. Perfect client-side batching measured `39.053 TPS`, versus the retained `38.016 TPS` median and `42.953 TPS` target.
- Correctness evidence: both cases returned all 256 requested output tokens.
- Failure mode: reducing the number of prefill launches has a small end-to-end opportunity; decode remains dominant after the Q5_K/Q6_K wins.
- Why not to retry unchanged: even perfect size-eight batching cannot close the target gap, while a delay adds TTFT to single idle requests.
- Reopen only if: the production workload values throughput over TTFT and combines this with a separate decode win, or prompt lengths make singleton prefill a materially larger fraction of wall time.
- Related commit or revert: no commit; scheduler and server-argument changes were removed.
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

## PERF-F033 - Production ranking from pre-fix tree benchmarks

- Hypothesis: measured M8/M12/M16 real-sampling throughput and emitted-token yield could rank a tree topology for production.
- Scope: top-k-greater-than-one EAGLE verification with the unified hybrid target pool, accepted target KV, compacted token/hidden rows, and the next draft cycle.
- Attempted change: exercised a deterministic non-front accepted path `[0,3,7]` under a nonidentity virtual-to-physical map, then repeated alternating non-front paths through one captured graph while reclaiming rejected slots.
- Benchmark evidence: the recorded M8/M12/M16 means of **97.352**, **94.685**, and **92.831 tok/s** remain useful execution-mechanism measurements only. They cannot support production promotion.
- Correctness evidence: before repair, `HybridLinearKVPool.move_kv_cache` forwarded virtual ids directly into a physical unified backing pool and the minimal test mismatched four of six sentinel values. The multi-layer caller also explicitly disabled accepted-path front compaction while its downstream draft-extend indexing assumed it. Later reachability inspection confirmed the recorded M8/M12/M16 launches used `enable_unified_memory=False` and the single-layer worker's existing finalizer, so that low-level failure is not proof those exact runs were corrupted. Their full cross-cycle state parity remains unproven.
- Failure mode: a non-front branch could copy sibling KV into the committed prefix or leave token/hidden state in tree order; a plausible completion and throughput number could therefore describe a state-corrupted trajectory.
- Why not to retry unchanged: width, topology, branch-local penalties, and target-kernel work cannot make a correctness-invalid tree benchmark promotable.
- Reopen only if: the repaired full model passes deterministic eager and captured multi-cycle non-front path parity, including target KV, request mapping, rejected-slot reclamation, recurrent state, terminal token/hidden state, and next-cycle proposal/logit comparison against a serial linear reference; then its ordinary sampled projection and two measured windows must exceed 200 tok/s with margin.
- Related commit or revert: repair pending in the accepted-path correctness commit; prior tree infrastructure is `d0116b54e5766932a46e06e0a66c3672370eaff8`.

## PERF-F034 - SWOR inside the raw child-graph composite

- Hypothesis: cloning the existing draft graph into the extend/bridge/draft parent preserves SWOR proposal randomness automatically.
- Scope: `CudaGraphChildSequence`, `torch.cuda.CUDAGraph` random kernels, and device-resident SWOR tree proposals.
- Attempted change: audited how the composed parent launches retained child graphs and added an external-race replay test.
- Benchmark evidence: prior device-cycle SWOR yield/TPS remains mechanism-only and cannot be used for a proposal-distribution claim.
- Correctness evidence: `CudaGraphChildSequence.replay` calls `cudaGraphLaunch` directly. It bypasses `torch.cuda.CUDAGraph.replay`, which owns PyTorch generator-offset advancement. A raw child containing captured random sampling can therefore reuse its captured RNG offset. External caller-refreshed races change samples correctly through the same raw parent; the new CUDA regression test passes.
- Failure mode: repeated or capture-stale SWOR random draws change the proposal law and can make acceptance/yield measurements look stable while sampling the wrong process.
- Why not to retry unchanged: graph composition alone cannot advance PyTorch's child-graph RNG bookkeeping.
- Reopen only if: every stochastic child consumes explicit caller-updated seed/offset or externally refreshed random inputs, and replay mutation/distribution tests pass before a server measurement.
- Related commit or revert: device-resident SWOR is now rejected at worker initialization in the active uncommitted experiment.

## PERF-F035 - Exact linear device-resident composite

- Hypothesis: composing draft extend, the exact-q bridge, and next draft decode into one raw parent would remove enough host seam cost to beat the production linear cycle.
- Scope: batch-one top-k-one EAGLE rejection sampling with the 200K production configuration.
- Attempted change: implemented stable live sampling inputs and exact q; first used caller-refreshed full-vocabulary exponential races, then replaced them with FlashInfer categorical sampling driven by explicit graph-stable seed/offset tensors.
- Benchmark evidence: dense-race form measured **122.576 tok/s** mean. Categorical form measured `115.058, 116.444, 120.530, 123.907, 124.434`, mean **120.075 tok/s**, versus the matched warmed control **124.775 tok/s**.
- Correctness evidence: all requests returned exactly 512 sampled tokens with thinking enabled; short exact acceptance smoke passed; q(X), offset replay, proposal transforms, and raw child graph tests passed. Five categorical acceptance probes averaged **2.277991**, above control.
- Failure mode: normalized execution cost remained higher. The categorical composite used **21.132 ms per verification cycle** over 1,124 cycles, while the ordinary linear path used **20.771 ms/cycle** over 1,163 cycles. Removing vocab-wide RNG recovered only part of the composite overhead.
- Why not to retry unchanged: both RNG strategies lost with higher acceptance, isolating the composition/bridge path rather than proposal yield.
- Reopen only if: a dependency/runtime change produces at least 0.75 ms of repeatable recoverable CUDA-event time under two independent windows, or CUDA/PyTorch gains a composition API that preserves runner bookkeeping without the current bridge work.
- Related commit or revert: retained opt-in in the pending exact linear device-cycle commit; launcher default remains off.

## PERF-F036 - Graph-tail scheduling recovery below the admission floor

- Hypothesis: work between target verify, draft extend, and the next draft graph contains enough repeatable idle device time to fund another graph-tail implementation.
- Scope: ordinary M3 linear rejection sampling, actual raw graph boundaries, asynchronous CUDA-event timestamps, active `EAGLEWorkerV2`, and torch compile mode `default`.
- Attempted change: added zero-default-overhead boundary probes, asynchronous event query, a bounded JSONL writer, and a robust admission analyzer using p10, median absolute deviation, and p10-to-p90 spread. Collected two independent 512-token windows and 1,471 transition records.
- Benchmark evidence: target-to-draft-extend was the best repeatable transition at conservative p10 **0.658355 ms**. Extend-to-next-draft had p10 0.474054 ms and failed the strict p80-span repeatability rule. Draft-to-target was roughly 0.09-0.10 ms.
- Correctness evidence: both startup logs and `/server_info` resolved the active worker and compile mode explicitly; the endpoint stayed healthy; three focused analyzer/probe tests passed; the exact process tree was stopped and port/GPU cleanup passed.
- Failure mode: the best repeatable recoverable tail is smaller than the **0.75 ms** minimum required to justify implementation and production requalification cost.
- Why not to retry unchanged: two independent windows agree on the same sub-threshold boundary, and the prior exact composite already regressed full-cycle cost.
- Reopen only if: code, CUDA, PyTorch, WDDM, or graph scheduling changes move the conservative repeatable p10 to at least 0.75 ms in two fresh windows.
- Related commit or revert: diagnostic probe retained; graph-tail production implementation remains closed.

## PERF-F037 - Selective-checkpoint M4 K+1 geometry

- Hypothesis: three speculative steps and four target rows would raise useful
  output per verification enough to exploit the selective target-NVFP4
  checkpoint's faster target cycle.
- Scope: unchanged single-layer NEXTN/EAGLE top-k-one rejection path, changing
  only `speculative_num_steps=3` and `speculative_num_draft_tokens=4`.
- Attempted change: launched the real 200K selective checkpoint with explicit
  seed `615388882`, profiled M4, measured five exact `199000+16` completions,
  then restored M3 and repeated the profile plus five exact completions.
- Benchmark evidence: M4 acceptance was **2.327273** over 55 cycles versus M3
  **2.245614** over 57 cycles. Full-cycle cost increased **16.058328 ->
  18.419190 ms**. Aggregate projected throughput therefore fell **139.841 ->
  126.350 tok/s** (-9.647%). Warmed exact prompt means were indistinguishable
  (**2790.258 M4**, **2789.288 M3**). Warmed generation means were
  **98.957 M4** and **100.982 M3** with 8-11% CV; isolated peaks above 110
  occurred in both arms.
- Correctness evidence: every exact request completed `199016`, returned
  `finish_reason=length`, kept thinking enabled, and retained digest
  `9a0e20749e2930a697fefdd3bdd7863a067abe4d9860e6d1e7d9b80a62668b37`.
  Both graph profiles resolved the intended width, worker, compile mode, and
  topology.
- Failure mode: the third draft step and fourth target row cost 14.702% more
  per cycle while acceptance improved only 3.636%. The 16-token generation
  metric is too cycle-quantized and variable to turn its isolated M4 peak into
  evidence against the device-cycle loss.
- Why not to retry unchanged: the A-B-A comparison directly measures the
  current selected checkpoint and exact production route. The unchanged shape
  needs either about **1.78 ms/cycle** less work or roughly **0.25** more
  accepted tokens/cycle merely to match the measured M3 projection.
- Reopen only if: selected-row draft-extend logits, a new proposal model, a
  quantized-KV/full-graph attention route, or another measured mechanism
  changes M4 cost or yield by at least that amount before another server run.
- Related commit or revert: configuration-only experiment; no runtime default
  changed. M3 remains selected.

## PERF-F038 - FlashInfer paged-only prefill

- Hypothesis: writing the current chunk to KV and running one paged attention
  would avoid the separate ragged-current attention, paged-prefix attention,
  and state merge on each long-prefill chunk.
- Scope: selective target-NVFP4 M3 server with only
  `SGLANG_FLASHINFER_USE_PAGED=1`.
- Attempted change: resolved the environment switch as true, used two full
  exact-shape warmups, measured three exact `199000+16` requests and two exact
  `199000+512` requests, then restored the default server for matched long
  generation and acceptance.
- Benchmark evidence: paged-only exact-200K prompt averaged **2785.260 tok/s**
  on the long pair versus **2789.036** control (-0.135%). Long generation
  averaged **104.117** versus **106.467 tok/s** (-2.207%). Acceptance improved
  only **1.961686 -> 1.976834** tokens/cycle.
- Correctness evidence: all requests completed exact token counts with
  `finish_reason=length`, and the hardened fragment/count telemetry passed.
  Paged-only selected a repeatable but different deterministic output for both
  16- and 512-token requests.
- Failure mode: the single paged operation did not make the 199K chunk path
  faster than the split ragged/paged calculation, and its numerical ordering
  changed the subsequent speculative trajectory without a throughput return.
- Why not to retry unchanged: matched prompt and long-generation results both
  fail to improve, while the apparent stable 16-token peak was contradicted by
  the 512-token comparison.
- Reopen only if: a new FlashInfer paged kernel or planner wins a direct
  per-chunk profile and preserves long-context logits closely enough to fund a
  fresh full-model comparison.
- Related commit or revert: environment-only experiment; default remains false.

## PERF-F039 - Global 7680-token chunk default

- Hypothesis: the selective checkpoint's exact-200K prompt win at chunk 7680
  would transfer safely to the launcher's default base RadixArk checkpoint.
- Scope: unchanged production launcher and base checkpoint, changing only the
  default chunk size from 4096 to 7680.
- Attempted change: temporarily changed the launcher default, relaunched base
  RadixArk without a model/chunk override, measured two five-run sampled
  windows, compared a fresh ten-run 4096 control, then ran exact
  `199000+16`, arithmetic, tools, and memory snapshots at 7680.
- Benchmark evidence: sampled generation was neutral: combined 7680 mean
  **121.054 tok/s** versus 4096 **121.027**. Exact base `199000+16` at 7680
  reached only **2226.770 prompt / 83.988 generation tok/s**, versus the
  qualified base exact prompt reference 2608.263.
- Correctness evidence: exact `199016` completed, arithmetic returned `703`,
  one correct tool call was parsed, and the server stayed healthy. Memory
  reached **31,988 MiB used / 200 MiB free** after the exact request; after
  follow-up probes and flush it recovered to 2,358 MiB free.
- Failure mode: the base checkpoint's larger residency and different target
  projection mix make the larger prefill chunks slower and leave unsafe
  transient operating margin, even though the selective checkpoint benefits.
- Why not to retry unchanged: the exact production checkpoint loses 14.6%
  prompt throughput against its qualified reference and approaches VRAM
  exhaustion.
- Reopen only if: base model/graph residency falls materially and a fresh
  exact-capacity A/B demonstrates both prompt improvement and safe pre-probe
  headroom.
- Related commit or revert: the temporary one-line default change was restored
  before commit. Use explicit `-ChunkedPrefillSize 7680` only with the
  selective performance checkpoint.

## PERF-F040 - Chunk 7808 refinement

- Hypothesis: keeping 26 chunks while increasing the main chunk from 7680 to
  7808 would move the selective exact prompt consistently above 3000 tok/s.
- Scope: selective M3 checkpoint with only chunk size changed.
- Attempted change: two full exact-shape warmups followed by three exact
  `199000+16` measurements.
- Benchmark evidence: prompt samples were `2912.697, 2909.720, 2905.634
  tok/s`, mean **2909.350**, versus the 7680 two-window mean **2997.744**.
- Correctness evidence: every request completed exact `199016` with one stable
  deterministic digest and valid fragment telemetry.
- Failure mode: the larger per-chunk shape hits a sharp kernel/planner
  efficiency cliff; preserving the same chunk count does not preserve
  throughput.
- Why not to retry unchanged: the regression is about 2.95% and stable across
  all three scored runs.
- Reopen only if: FlashInfer planning or kernel schedules change for this
  query length.
- Related commit or revert: configuration-only; 7680 remains the selective
  winner.

## PERF-F041 - Single-layer selected-row draft-extend logits

- Hypothesis: draft extend needs all three hidden/KV rows but only one row's
  vocabulary logits, so pruning `lm_head` input from three rows to one would
  reduce each speculative cycle.
- Scope: single-layer `EAGLEDraftExtendCudaGraphRunner` and
  `EagleDraftWorker._draft_extend_for_decode`, reusing the existing
  `EagleDraftExtendInput.select_index` and logits-processor contract.
- Attempted change: added graph-stable selection indices, reduced the logits
  output buffer to one row per request, preserved full hidden rows, and gated
  pruning off for gathered buffers, standalone drafting, and the retained
  device-resident composite.
- Benchmark evidence: matched traces measured draft-extend graph
  **1.059 ms control / 1.061 ms candidate** and full cycle **16.058328 /
  16.066558 ms**. Candidate kernel count increased from 28 to 29.
- Correctness evidence: the candidate captured all graphs and completed exact
  `6213+128` profiling at **2.370370** tokens/cycle. Focused CPU runner tests
  passed before launch.
- Failure mode: the 248K-vocabulary NVFP4 projection is weight-bandwidth-bound
  at these tiny row counts. Reducing M from three to one does not reduce the
  dominant weight read and adds one selection operation.
- Why not to retry unchanged: both the local graph span and end-to-end device
  cycle are unchanged-to-worse despite lower logits-buffer residency.
- Reopen only if: the lm-head kernel gains real cross-row weight reuse or a
  sparse-vocabulary head changes the amount of weight data read.
- Related commit or revert: implementation and test removed before commit;
  trace/manifest retained as evidence.

## PERF-F042 - FlashInfer TRT-LLM dense FP4 on SM120

- Hypothesis: the already-implemented FlashInfer TRT-LLM dense FP4 backend
  would reduce the selected checkpoint's dominant NVFP4 GEMM wall relative to
  the qualified CUTLASS backend.
- Scope: `ModelOptFp4LinearMethod`, FlashInfer `mm_fp4`, and the
  `flashinfer_trtllm` dense-linear backend on the native-Windows RTX 5090.
- Attempted change: no source change. Ran the real layer-path focused test
  `test_nvfp4_linear_backends.py::TestNvFp4LinearBackends::test_flashinfer_trtllm`
  through `scripts/windows/invoke_cuda_pytest.ps1`.
- Benchmark evidence: no valid kernel timing was possible. FlashInfer rejected
  all three test shapes, `(64,256,512)`, `(5,160,336)`, and
  `(128,1024,1024)`, before execution with
  `BackendSupportedError: mm_fp4 does not support backend 'trtllm' with
  capability 120`.
- Correctness evidence: the focused path reached real checkpoint-format weight
  loading, TRT-LLM weight shuffling, activation quantization, and
  `ModelOptFp4LinearMethod.apply`; it failed at FlashInfer's explicit backend
  capability check before producing output.
- Failure mode: FlashInfer `0.6.17` does not expose its dense TRT-LLM FP4 GEMM
  for SM120. Core SGLang support and B200 coverage do not make the backend
  available on the RTX 5090.
- Why not to retry unchanged: bypassing the explicit dependency capability
  gate would not establish a compiled or correct SM120 kernel, and a full
  server launch would fail at the same call.
- Reopen only if: a later FlashInfer build explicitly supports dense
  `mm_fp4(..., backend="trtllm")` on capability 120 and the focused numerics
  plus CUDA-graph replay test passes before a server launch.
- Related commit or revert: no implementation change; evidence-only update.

## PERF-F043 - FlashInfer CuTe-DSL fused SwiGLU-to-NVFP4 on native Windows

- Hypothesis: FlashInfer `0.6.17`'s public fused
  `silu_and_mul_nvfp4_quantize` API could directly replace the selected
  Windows activation plus NVFP4 quantization sequence.
- Scope: native Windows RTX 5090, production width 17408, real
  down-projection input scale, and `M={1,3,7000,7680}`.
- Attempted change: no runtime source change. Called the installed API through
  the native CUDA environment before any model wiring.
- Benchmark evidence: no kernel timing was possible. Importing
  `flashinfer.cute_dsl` failed with `ModuleNotFoundError: No module named
  'cutlass'`.
- Correctness evidence: unavailable because the dependency failed before
  compilation. The separate native CUDA expert producer was measurable but
  changed about 0.8% of packed values and is not a valid exact replacement.
- Failure mode: the public API is CuTe-DSL-only. NVIDIA's CUTLASS DSL binary
  dependency has no native-Windows wheel/source route in the documented
  environment.
- Why not to retry unchanged: installing the metadata package repeats the
  already-closed native-Windows CUTLASS-DSL failure and cannot supply its
  Linux-only compiled base.
- Reopen only if: NVIDIA publishes a supported native-Windows CUTLASS DSL
  runtime and the public API passes packed-value, scale-byte, graph, and
  latency parity.
- Related commit or revert: no dependency or site-package change. PERF-027
  continues through a separate exact native CUDA JIT producer.

## PERF-F044 - One eager-exact SwiGLU-to-NVFP4 producer for every phase

- Hypothesis: the byte-exact eager Windows producer could replace
  `SiluAndMul` plus activation quantization in both long prefill and the
  torch-compiled M3 target-verification graph.
- Scope: selective target-NVFP4 checkpoint, chunk 7680, compiled M3 target
  graph, eager 7680/7000 prefill, and exact `199000+16`/`199000+512`.
- Attempted change: selected the precise two-rounding native producer in every
  target MLP phase.
- Benchmark evidence: five short requests improved prompt to **2993.552
  tok/s**, but selected a different stable digest. Three long requests also
  changed trajectory and averaged **115.542 generation tok/s**.
- Correctness evidence: a direct discriminator found that eager native matched
  the explicit staged reference, while `torch.compile(fullgraph=True)` changed
  the quantized tuple by 63 packed/18 scale bytes at M1 and 216/51 at M3.
- Failure mode: Inductor fuses `F.silu(gate) * up` in FP32 and removes the
  eager path's intermediate BF16 rounding. The eager-exact producer therefore
  changed the established compiled target function even though it was exact
  for prefill.
- Why not to retry unchanged: production target verification is compiled, so
  selecting one eager arithmetic contract globally deterministically changes
  logits, rejection decisions, and output.
- Reopen only if: a separate compiled-semantics producer matches the prior M3
  packed values, scales, down-projection output, logits, RNG decisions, and
  outer CUDA-graph replay bit-for-bit.
- Related commit or revert: PERF-027 is retained only outside
  `torch.compiler.is_compiling()`; the former compiled path remains selected.

## PERF-F045 - Compiled-semantics SwiGLU-to-NVFP4 producer

- Hypothesis: matching Inductor's one-final-rounding SiLU function in a
  PDL-safe dense producer would remove the compiled activation and NVFP4
  quantization launch boundary and materially reduce the M3 target cycle.
- Scope: M1/M3 compiled target semantics, outer CUDA graph, selected M3 exact
  `199000+16` and `199000+512`, unchanged 200K pools.
- Attempted change: added a separately fast-math-compiled specialization using
  the FlashInfer expert arithmetic while preserving deterministic dense scale
  padding and PDL wait/trigger semantics.
- Benchmark evidence: isolated M3 improved **70.848 -> 25.152 us**. Three
  exact long requests moved **115.225 -> 116.192 tok/s**, only +0.839% and
  within launch/WDDM variance.
- Correctness evidence: exact packed/scale bytes, mutable graph replay, nested
  fullgraph, tuple-consumer graph, both deterministic output digests, and exact
  token counts all passed.
- Failure mode: 233 profiled full cycles measured **16.045 ms median** versus
  the existing **16.058 ms** control. The isolated launch saving was not
  serialized on the serving critical path.
- Why not to retry unchanged: standalone operator latency overstates value
  inside the compiled multi-stream target graph; the cycle-level admission
  result is neutral.
- Reopen only if: a new target trace shows at least 0.25 ms of repeatable
  serialized exposure at this boundary or a larger fusion removes adjacent
  graph work as well.
- Related commit or revert: all PERF-029 source and test changes were removed;
  PERF-027 eager fusion remains retained.

## PERF-F046 - FlashInfer fixed paged-prefix split under 128 MiB workspace

- Hypothesis: an explicit paged-prefix split size could improve the dominant
  long-prefix attention work while retaining ragged-current attention.
- Scope: selected checkpoint, chunk 7680, exact 200K pools, 128 MiB workspace,
  split sizes 4096 and 8192.
- Attempted change: honored the existing registered prefill-split descriptor
  when explicitly set outside deterministic mode; all other settings remained
  unchanged.
- Benchmark evidence: no score. Both arms failed on the first exact-shape
  warmup before inference output.
- Correctness evidence: model load, exact pool allocation, and graph capture
  passed; the request never reached a result.
- Failure mode: FlashInfer requested **2,264,924,160 bytes** for
  `batch_prefill_tmp_v`, but the qualified workspace contains 134,217,728
  bytes.
- Why not to retry unchanged: both tested split sizes hit the same allocation
  wall. Increasing workspace violates the selected 128 MiB contract and
  consumes limited exact-capacity headroom.
- Reopen only if: FlashInfer's fixed-split planner can bound temporary storage
  below 128 MiB for this exact ragged/paged geometry.
- Related commit or revert: the expert opt-in was removed; unset behavior is
  unchanged.

## PERF-F047 - Packed GDN target-verify split removal

- Hypothesis: target verification materialized Q, K, and V after convolution,
  and removing that copy would reduce the M3 cycle.
- Scope: Qwen3.8 M3, 48 GDN layers, ReplaySSM fold target verification.
- Attempted change: none; source reachability was the admission gate.
- Benchmark evidence: no candidate kernel exists on the selected route.
- Correctness evidence: Qwen3.8 has packed QKV width 10,240, above
  `MAX_FUSED_QKV_SPLIT_DIM=8192`, so the backend selects metadata-only
  `torch.split`/`view` aliases. ReplaySSM consumes their runtime token stride.
- Failure mode: the assumed materialization is already absent.
- Why not to retry unchanged: adding a packed-pointer API would replace an
  existing zero-copy alias with more code and potential register pressure.
- Reopen only if: a current exact M3 trace shows the split kernel running 48
  times with at least 0.05 ms exclusive wall per replay.
- Related commit or revert: no source change.

## PERF-F048 - Coalesced 14,680-token final prefill tail

- Hypothesis: merging the final 7,680 and 7,000-token passes would remove one
  complete 64-layer forward while retaining exact capacity and the 192K Mamba
  checkpoint.
- Scope: default-off selective profile, exact `199000+16`, one request,
  16,384-token tail ceiling.
- Attempted change: scheduler emitted `24 * 7680 + 14680`; the existing Mamba
  branching tracker preserved the 192,000-token checkpoint.
- Benchmark evidence: **1917.509 prompt tok/s**, **103.780505 s TTFT**, and
  **104.088783 s E2E**, versus roughly 2,987 prompt tok/s and 66.623 s TTFT on
  the retained source.
- Correctness evidence: exact `199016` and `finish_reason=length` passed, but
  the deterministic digest changed.
- Failure mode: coalescing moved the last chunk's interaction with the preceding
  7,680 tokens from paged-prefix attention into one much larger ragged-current
  causal pass, changing both kernel efficiency and reduction order.
- Why not to retry unchanged: the regression is about 36%, far outside noise;
  dispatch removal cannot repay the larger ragged kernel.
- Reopen only if: a fused/partitioned attention implementation preserves the
  selected ragged/paged kernel shapes while eliminating host/model dispatch.
- Related commit or revert: all tail option, sizing, scheduling, and tests were
  removed.

## PERF-F049 - Full-attention sigmoid gate to NVFP4 tuple

- Hypothesis: fusing the 16 attention-output gates with activation quantization
  would remove an exposed gate/quant boundary from target verify and prefill.
- Scope: BF16 width 4096, M1/M3/M7000/M7680, native Windows SM120.
- Attempted change: PDL-safe precise sigmoid-multiply plus native E4M3/E2M1
  packing with deterministic padding; no model wiring.
- Benchmark evidence: staged-to-fused medians were `2.731 -> 2.304 us` at M3
  and `135.402 -> 85.124 us` at M7680.
- Correctness evidence: **9 CUDA tests** passed, covering production shapes,
  all finite BF16 values, graph replay, fullgraph, and ModelOpt consumption.
- Failure mode: only 16 target layers use the boundary. Projected M3 saving is
  0.0068 ms/replay and exact-prefill saving is about 20.9 ms.
- Why not to retry unchanged: both projections are below the 0.05 ms target
  cycle admission floor and far below the roughly one-second prompt gap.
- Reopen only if: the fusion expands across an adjacent attention kernel or
  removes at least 0.05 ms of measured full-cycle exposure.
- Related commit or revert: all kernel, wrapper, test, and benchmark files were
  removed before model integration.

## PERF-F050 - Global KV page-size sweep for exact-200K prefill

- Hypothesis: page 128 or 32 could improve the FlashInfer paged-prefix kernel
  that dominates exact prefill.
- Scope: selective checkpoint, chunk 7680, page 64 control versus 128/32.
- Attempted change: launch-only page-size overrides; no source change.
- Benchmark evidence: page 32 short prompt mean was 3030.480 tok/s, but three
  long requests averaged only **2970.617 prompt / 112.576 generation tok/s**.
  Page 128 was not timed.
- Correctness evidence: page 32 retained exact counts and digests. Page 128
  allocated only 199,936 target/draft tokens and failed the 200K pool gate.
- Failure mode: SGLang's FlashInfer prefill wrapper plans with page size 1 and
  per-token slot IDs, independent of the global storage page size. The apparent
  page-32 short movement has no paged-prefill mechanism and decode regressed.
- Why not to retry unchanged: page 128 is capacity-ineligible; page 32 moves
  the wrong runtime surface and loses stable generation.
- Reopen only if: prefill receives a real page-layout specialization rather
  than the existing token-index interface.
- Related commit or revert: no source change; launcher default remains 64.

## PERF-F051 - FlashInfer FP16 QK reduction for paged prefill

- Hypothesis: FP16 QK reduction would accelerate the paged-prefix kernel while
  preserving BF16 output.
- Scope: all 25 exact-request prefix shapes, 24 Q heads, 4 KV heads, dimension
  256, BF16 Q and FP8-E4M3 KV.
- Attempted change: default-off ordinary-prefill precision switch; speculative
  graph planners retained FP32.
- Benchmark evidence: initial server A-B moved prompt +0.679%, but the exact
  prefix ladder summed to **2964.761 ms FP32 vs 2974.993 ms FP16 per layer**,
  or **163.705 ms slower** across 16 layers.
- Correctness evidence: every isolated shape and both server digests were
  bit-exact; exact counts and pools passed.
- Failure mode: shape-dependent variance produced mixed results. The initial
  server movement was environmental noise, not a stable kernel improvement.
- Why not to retry unchanged: the complete deterministic kernel ladder is a
  stronger attribution than the overlapping five-request means and favors
  FP32.
- Reopen only if: a later FlashInfer kernel changes the real
  24-head/4-head/256-dimension precision economics.
- Related commit or revert: the provisional PERF-035 code was removed in a
  corrective follow-up; FP32 remains selected.

## PERF-F052 - FlashInfer paged-prefix KV MMA tile reduction

- Hypothesis: reducing the traced FP8/head-dimension-256 paged-prefix kernel
  from `NUM_MMA_KV=4` to 2 would improve occupancy and shorten the dominant
  exact-prefill wall.
- Scope: all 25 exact-request paged-prefix shapes, BF16 Q, FP8-E4M3 K/V, 24
  query heads, 4 KV heads, dimension 256, logical page size 1.
- Attempted change: a narrow native dispatcher cap for CTA-Q 64 in
  `BatchPrefillWithPagedKVCacheDispatched`; CTA-Q 16/32/128 were screened
  separately.
- Benchmark evidence: the correctly routed candidate regressed the aggregate
  from **3013.932 to 3414.968 ms/layer** (**+13.306%**). CTA-Q 16 reached
  **4154.807 ms/layer**; CTA-Q 32 and 128 were invalid trait combinations.
- Correctness evidence: the candidate completed every shape but changed the
  aggregate output and LSE digests. The restored CTA-Q-64 control retained
  `d9ad4f3e...992d6` output and `2b20c9f2...ebc9` LSE.
- Failure mode: the smaller KV tile performs more iteration/reduction work and
  loses substantially on the active SM120 kernel; its different reduction
  order also changes BF16 output.
- Why not to retry unchanged: the deterministic full ladder shows a 13.3%
  kernel regression before server overlap or request variance.
- Reopen only if: a materially different FA2 implementation, mainloop, or
  accumulator schedule changes the active tile economics.
- Related commit or revert: both external header copies were restored to
  SHA-256
  `2E5927BDC0D36DDB393CB4FAB68C2E958D65D5B4B0085C969F7CFA777ECDFB5B`;
  the experimental generated module was deleted.

## PERF-F053 - Gemma residual norm to NVFP4 activation tuple

- Hypothesis: emitting exact NVFP4 values/scales from the selected fused
  residual-add/Gemma-RMSNorm kernel would remove the quantization launch before
  each target projection and save at least 0.30 ms over 64 layers.
- Scope: BF16 width 5120 at M1/M3/M7000/M7680, exact E4M3/E2M1 packing, then
  the real M3 `5120 -> 34816` NVFP4 gate/up GEMM.
- Attempted change: temporary repository-native SM120 CUDA dual-output
  producer; Python was only a thin custom-op binding. No model dispatch was
  changed.
- Benchmark evidence: isolated M3 staged/fused medians were
  **0.040000/0.027296 ms**. With the dependent gate/up GEMM, 51-sample medians
  were **0.096704/0.097152 ms**, a **0.000448 ms/layer regression**.
- Correctness evidence: normalized BF16 input, updated residual, packed values,
  and all scale bytes were bit-exact at every production shape.
- Failure mode: programmatic dependent launch already overlaps the standalone
  quantizer with GEMM startup, so isolated launch removal does not shorten the
  full boundary.
- Why not to retry unchanged: the exact dependent-boundary benchmark projects
  **-0.028672 ms** across 64 layers, below zero and far below admission.
- Reopen only if: the fusion expands across the GEMM mainloop/epilogue or a
  later dependency changes PDL overlap at this boundary.
- Related commit or revert: all prototype source was removed and the exact JIT
  cache directory was deleted before model wiring.

## PERF-F054 - Sub-128-row SM120 NVFP4 GEMM tile

- Hypothesis: reducing the dominant M3 NVFP4 CTA from `128x32x256` to
  `64x32x256` would cut padded token-row work and shorten the exposed GEMM
  family.
- Scope: repository-native CUTLASS, M=3, K=5120, N=34816, existing packed
  values/scales, static persistent scheduler, PDL enabled.
- Attempted change: temporary cooperative and ping-pong JIT specializations
  using the bundled FlashInfer/CUTLASS headers.
- Benchmark evidence: none; both variants failed compile-time architectural
  contracts before a kernel launch.
- Correctness evidence: no output was produced.
- Failure mode: cooperative SM120 GEMM requires CTA-M >=128. Ping-pong permits
  a 64-row MMA tile, but NVFP4's TMA scale layout is a fixed 128-row swizzled
  atom and cannot map onto CTA-M 64. The existing M3 tactic already swaps A/B
  and uses the minimum supported CTA-N 32.
- Why not to retry unchanged: no legal smaller tile exists in the current
  mainloop/epilogue family; adding another tactic cannot bypass its static
  layout requirements.
- Reopen only if: a new mainloop supports non-TMA scale loads, a smaller scale
  atom, or a CTA-N-16 epilogue/LDSM contract.
- Related commit or revert: all prototype source and exact failed-build cache
  directories were removed.

## PERF-F055 - MTP dual Gemma norm and concat fusion

- Hypothesis: combining both MTP pre-FC Gemma norms and `torch.cat` in one
  native producer would materially reduce draft-decode and draft-extend graph
  spans.
- Scope: BF16 hidden width 5120, M1/M3, dependent `10240 -> 5120` BF16 FC,
  mutable captured output.
- Attempted change: a temporary native SM120 kernel. The admitted design kept
  two 320-thread CTAs per row in one launch and wrote directly to the
  concatenated output.
- Benchmark evidence: 101-sample dependent-boundary medians improved
  **0.065824 -> 0.064576 ms** at M1 and **0.052128 -> 0.050048 ms** at M3.
- Correctness evidence: concatenated BF16 values and dependent FC outputs were
  bit-exact at both shapes.
- Failure mode: the existing norms, concat, and FC are already compact inside
  the captured graphs. Launch/copy removal exposes only 1-2 us per invocation.
- Why not to retry unchanged: one M1 plus one M3 use saves about
  **0.0033 ms/cycle**, two orders below the active decode funding floor.
- Reopen only if: the fusion expands across the FC mainloop or a materially
  wider draft topology multiplies this boundary.
- Related commit or revert: all prototype source and the exact JIT cache were
  removed before model routing.

## PERF-F056 - Stock-EVT gate/up GEMM-SwiGLU-NVFP4 fusion

- Hypothesis: replacing FlashInfer's BF16 linear-combination epilogue with an
  activation/block-scale EVT would fuse the dominant gate/up boundary.
- Scope: selected M3 SM120 NVFP4 gate/up tactics and exact compiled SwiGLU
  arithmetic.
- Attempted change: none; external CUTLASS source and selected tactic
  reachability were the admission gate.
- Benchmark evidence: selected `2560x34816` tactics are `12/12/4`; all map to
  swap-AB DP under FlashInfer's four-config-per-tile ordering.
- Correctness evidence: source inspection only.
- Failure mode: standard EVT is coordinate-preserving and cannot pair two
  accumulators or halve the output. Swap-AB moves gate/up pairing to GEMM M,
  requiring an independent half-height store design.
- Why not to retry unchanged: this is a custom collective epilogue project,
  not a fusion-functor or tactic edit.
- Reopen only if: a staged swap-AB collective first proves exact BF16 paired
  output and retains the selected mainloop economics.
- Related commit or revert: no source change.

## PERF-F057 - Sparse top-p as a standalone generation winner

- Hypothesis: removing dense AIR top-p would independently raise exact
  long-generation throughput above 120 tok/s.
- Scope: native-Windows top-k 20/top-p 0.95, exact `199000+512`, unchanged
  proposal and rejection RNG.
- Attempted change: exact native sparse-support top-p behind a default-off
  expert environment gate.
- Benchmark evidence: A-B-A device cycles improved, but long generation was
  `101.238, 125.757, 107.682` tok/s, mean **111.559** and worst **101.238**.
- Correctness evidence: 15 CUDA and 6 target/draft integration tests; final
  A2/control trace output, acceptance histogram, and cycle count matched
  exactly; exact capacity passed.
- Failure mode: the transform reduces cycle cost but cannot raise stochastic
  acceptance. The measured acceptance mean was only **2.194869**.
- Why not to retry unchanged: standalone client throughput remains below the
  milestone despite a real compute win.
- Reopen only if: stacked acceptance-neutral wins or a separately qualified
  proposal-quality improvement lifts the worst long window above 120.
- Related commit or revert: signed `7cb4ed0796` retains the kernel default-off
  as additive work.

## PERF-F058 - Draft proposal top-k 32

- Hypothesis: widening q from top-k 20 to 32 would recover target-top-20 tokens
  ranked 21-32 by the draft model and raise linear-chain acceptance.
- Scope: selected page-aligned M3 server, ordinary rejection sampling, changing
  only `--speculative-draft-sampling-top-k 20 -> 32`.
- Attempted change: ran five native 512-token acceptance probes per arm.
- Benchmark evidence: mean emitted length fell **2.217279 -> 2.173943**; the
  k32 arm also increased mean client latency.
- Correctness evidence: every probe completed 512 tokens with thinking enabled.
- Failure mode: extra proposal support mostly diluted q mass outside useful
  target overlap instead of recovering enough missing target mass.
- Why not to retry unchanged: matched five-probe evidence rejects static k32.
- Reopen only if: a measured root-only or confidence-gated policy identifies a
  repeatable subset where ranks 21-32 improve overlap.
- Related commit or revert: configuration-only experiment; top-k 20 remains
  selected.

## PERF-F059 - Greedy draft proposal top-k 1 single-sample conclusion

- Hypothesis: for the temperature-zero exact scoreboard, deterministic draft
  argmax would align more often with the one-hot greedy target than sampling
  from q top-k 20.
- Scope: page-aligned selective M3 server; only
  `--speculative-draft-sampling-top-k 20 -> 1` changed.
- Attempted change: one warmup plus one exact `199000+16` score, followed by
  three greedy 512-token acceptance probes.
- Benchmark evidence: exact short generation was **97.900 tok/s**. Greedy
  acceptance samples `2.115702, 2.106996, 2.098361` averaged **2.107020**.
- Correctness evidence: exact request completed `199016` with the established
  deterministic output digest; all probes completed 512 tokens.
- Failure mode: the draft argmax is not sufficiently aligned with target
  argmax; removing q support loses useful alternative-token overlap.
- Why not to retry unchanged: superseded by PERF-050's longer matched evidence.
- Reopen only if: use the PERF-050 page64/top-p1 stack and long-window contract.
- Related commit or revert: this early conclusion is superseded; k1 later
  reached 123.049 tok/s on exact199K+512 but still fails exact16.

## PERF-F060 - Draft proposal top-k 16

- Hypothesis: modestly concentrating q from top-k 20 to 16 would remove
  low-overlap draft-tail mass without the support loss observed at top-k 8.
- Scope: page-aligned selective M3 server; only draft sampling top-k changed.
- Attempted change: five sampled-profile 512-token acceptance probes.
- Benchmark evidence: samples
  `2.265487, 2.235808, 2.197425, 2.169492, 2.160338` averaged
  **2.205710**, below k20 **2.217279**; mean latency was also slightly worse.
- Correctness evidence: all probes completed 512 tokens with thinking enabled.
- Failure mode: any concentration gain was smaller than the lost support mass.
- Why not to retry unchanged: k1, k8, k16, and k32 all lose to k20 evidence.
- Reopen only if: a root-only, confidence-gated, or learned calibration policy
  demonstrates held-out overlap gain.
- Related commit or revert: configuration-only; top-k 20 remained selected at
  that stage. PERF-062 later promoted native top-k one with different target
  numerics.

## PERF-F061 - Vacuous proposal-only top-p 1.0 routing

- Hypothesis: retaining all q top-k-20 tokens would improve overlap and
  skipping two q top-p transforms would reduce the M3 cycle.
- Scope: page-aligned selective M3 server, target top-p unchanged at 0.95,
  temporary proposal-only q top-p override 0.95 -> 1.0.
- Attempted change: cached the override on `MultiLayerEagleWorkerV2`.
- Benchmark evidence: the apparent candidate/control cycle means were
  16.098454/16.108184 ms, but both arms ran the default live path.
- Correctness evidence: graph integration and exact capacity passed.
- Failure mode: the active worker is `EAGLEWorkerV2`; compatibility `getattr`
  in the runner resolved no override, making the experiment vacuous.
- Why not to retry unchanged: it does not reach production.
- Reopen only if: the override is cached on the actual worker and a trace proves
  q top-p kernels are absent.
- Related commit or revert: temporary source was removed; no commit.

## PERF-F062 - Proposal additive-penalty scalar

- Hypothesis: the draft head might implicitly over-apply presence/additive
  penalties, so scaling q-only penalties could improve p/q overlap.
- Scope: page-aligned M3, target penalties unchanged, proposal scales 0.75 and
  0.0.
- Attempted change: correctly cached the scalar on the live worker and scaled
  the graph-stable additive row before proposal q construction.
- Benchmark evidence: scale 0.75 reproduced all five control proposal/output
  sequences exactly. Scale 0.0 reproduced the identical first 512-token
  sequence and **2.216450** emitted length.
- Correctness evidence: every probe completed 512 tokens with thinking.
- Failure mode: this workload's proposal path has no decision-level leverage
  through the captured additive row.
- Why not to retry unchanged: even removing the row entirely changed nothing.
- Reopen only if: an authoritative p/q capture shows a nonzero additive row and
  coins near overlap boundaries for another workload.
- Related commit or revert: temporary source removed; no commit.

## PERF-F063 - ReplaySSM commit overlap

- Hypothesis: hide the target ReplaySSM fold/conv rollback under the independent
  draft-extend graph.
- Scope: page-aligned + proposal-top-p-one M3 line; default-off target-state
  side stream with a forward-stream rejoin before scheduler return.
- Attempted change: side stream waited on verify, consumed record-stream
  protected inputs, ran fold/conv commit, and overlapped draft extend.
- Benchmark evidence: production fold microbenchmark measured 222.6 us
  (334.2 us with tracking); live fold averaged 189.478 us. A/B/C cycle
  mean/median/p90 were
  `15.913862/15.859291/16.097856`,
  `15.755353/15.721147/15.889537`, and
  `15.922452/15.867868/16.123397 ms`.
- Correctness evidence: five output hashes, histograms, verify counts, and
  acceptance lengths matched exactly; exact `199016` digest/capacity passed.
- Failure mode: interval attribution showed **186.819 us** fold overlap but
  draft-extend graph 8 expanded **1.060552 -> 1.237001 ms**. Serial
  fold+extend was ~1.234266 ms versus ~1.237001 ms overlapped; bandwidth
  contention erased the hidden work.
- Why not to retry unchanged: the measured dependent boundary is neutral/slower
  despite complete fold overlap.
- Reopen only if: fold traffic is reduced materially or draft extend no longer
  competes for the same memory bandwidth.
- Related commit or revert: temporary source removed; no commit.

## PERF-F064 - Static proposal gamma/rank/token calibration

- Hypothesis: branch-exact p/q records would reveal a stable scalar, rank, or
  token bias that raises linear-chain overlap toward 2.34 emitted tokens.
- Scope: two chronological batch-one sampled-profile corpora from the active
  EAGLEWorkerV2 k20/top-p0.95 path (151 and 239 records).
- Attempted change: offline per-depth gamma grid, learned q-rank weights, and
  minimum-count token weights with chronological train/validation splits.
- Benchmark evidence: second corpus baseline expected length was **2.187060**
  with support ceiling **2.737586**. Best train gamma overfit; maximin gamma
  `(1.0,1.05)` improved the worse half only **0.000133**. Rank/token validation
  fell to **2.121759/2.124150** from **2.128776**.
- Correctness evidence: records are branch-exact post-transform p/q with
  complete finite supports; independent capture reproduced the conclusion.
- Failure mode: mismatch is context/trajectory-dependent; static corrections
  fit the early high-overlap phase and regress later states.
- Why not to retry unchanged: two corpora and held-out chronology reject every
  static family tested.
- Reopen only if: calibration consumes context/hidden features and clears a
  held-out conservative expected-length gain of at least 0.05.
- Related commit or revert: no serving calibration retained; diagnostic queue
  repair retained separately.

## PERF-F065 - XQA SM-count/PDL controls

- Hypothesis: reducing XQA SM residency or disabling PDL would shorten the
  199K target graph.
- Scope: exact SM120 B1/Q3/QH24/KVH4/D256/FP8-KV/page64 XQA microshape.
- Attempted change: swept 32,48,64,80,96,112,128,144,170 SMs and PDL on/off.
- Benchmark evidence: lower SM counts changed output and saved only a few us;
  all170 median was ~272 us. PDL true/false was 271.424/271.840 us.
- Correctness evidence: only all170 preserved the reference digest; both PDL
  modes were bit-exact.
- Failure mode: split/reduction geometry changes numerics before providing
  material value; PDL is neutral.
- Why not to retry unchanged: no admissible built-in control clears funding.
- Reopen only if: a CUDA kernel change preserves all170 reduction order.
- Related commit or revert: no source change.

## PERF-F066 - M4 under greedy k1

- Hypothesis: higher greedy k1 yield could let four-row verification complete
  exact16 in fewer cycles.
- Scope: page64/top-p1, k1, three speculative steps/four verify rows.
- Attempted change: launched full 200K M4 and measured exact16 acceptance.
- Benchmark evidence: still seven cycles, emitted length **2.285714**, same as
  M3 exact16; histogram `[2,1,2,2]`.
- Correctness evidence: exact `199016` and established digest passed.
- Failure mode: added row/step did not reduce discrete cycle count.
- Why not to retry unchanged: it necessarily costs more per cycle for no cycle
  reduction.
- Reopen only if: a new proposal head raises exact16 emitted length enough for
  six or fewer cycles.
- Related commit or revert: configuration-only; M3 remains selected.

## PERF-F067 - Device-resident cycle under greedy k1

- Hypothesis: composing draft extend and the next draft decode would remove
  enough host launch seam to improve the seven-cycle exact16 request now that
  greedy k1 avoids categorical proposal work.
- Scope: selected page64/top-p1/k1 M3 profile at exact `199000+16`.
- Attempted change: enabled the retained batch-one device-resident EAGLE cycle.
- Benchmark evidence: generation was **97.730 tok/s**, versus the adjacent
  five-run control mean **98.478 tok/s**; prompt was **3215.592 tok/s**.
- Correctness evidence: exact `199016`, `finish_reason=length`, and output
  SHA-256 `cdf5bb57...f647d9` matched the control.
- Failure mode: composing the two draft phases does not shorten the exposed
  long-context target/cycle wall; the movement is inside control variance.
- Why not to retry unchanged: the new greedy-k1 premise was tested directly.
- Reopen only if: the composite graph itself becomes materially cheaper or
  another change removes a measured exposed transition.
- Related commit or revert: configuration-only; the opt-in infrastructure
  remains retained.

## PERF-F068 - SM120 XQA structural constant sweep

- Hypothesis: deeper V buffering, wider K/V tiles, a no-hint row-max mode, or
  smaller CTA geometry could reduce the 199K XQA kernel.
- Scope: installed FlashInfer SM120 B1/Q3/QH24/KVH4/D256/FP8-KV/page64 source
  and its exact JIT module.
- Attempted change: independently screened V buffers 3, V tile 64, K partition
  128 with one/two buffers, row-max method 0, and CTA-x 2.
- Benchmark evidence: V buffers 3 moved **273.952 -> 273.824 us** and V tile 64
  moved **273.952 -> 272.992 us**, both below funding. Row-max 0 was
  **274.240 us**. K128/one-buffer appeared faster at **264.992 us** but was
  nondeterministic.
- Correctness evidence: V-buffer and V-tile candidates preserved digest
  `8a532a...034`; K128/one-buffer changed across repeated launches. K128/two
  buffers exceeded SM120 shared memory (`116352 > 101376` bytes), and CTA-x 2
  is structurally invalid for four GEMM1 warps per group.
- Failure mode: valid variants are noise-sized; the only faster build violates
  the kernel's double-buffer pipeline and output determinism.
- Why not to retry unchanged: every independent source constant was compiled
  or rejected by a precise architecture constraint.
- Reopen only if: a redesigned CUDA mainloop preserves buffering, reduction
  order, and all-170-SM output while reducing measured wall materially.
- Related commit or revert: all candidate modules were removed; installed
  `mha.cu` was restored to SHA-256
  `097203B6DCD37A04A2DC99F2174D397409E8F17D0AE0F3E16F4B754C8059218D`.

## PERF-F069 - Hidden-conditioned proposal rank head

- Hypothesis: proposal-aligned MTP hidden state could predict which q20 rank
  matches the greedy target and reduce exact16 from seven verify cycles to six.
- Scope: exact-q, draft-hidden, greedy-target labels at exact 199K context,
  separated by proposal role and chronological blocks.
- Attempted change: extended the default-off p/q diagnostic, captured 251
  exact-context cycles / 428 trainable rows, then fit PCA-linear rank heads
  across several ranks and class-weight schedules.
- Benchmark evidence: q20 contained **250/251** root target tokens and every
  observed correct-path inner target. The support oracle emits three tokens at
  exact16 committed positions `0,3,6,9,12,15`, proving a six-cycle ceiling.
- Correctness evidence: captures store exact verifier q, BF16 hidden payloads,
  greedy target rank, and realized device acceptance through the existing
  pinned asynchronous D2H lifetime.
- Failure mode: selected heads overfit training minority ranks. Role-zero
  validation/test minority accuracy was **0%**; the locked exact16 rank-1 and
  rank-2 roots remained misclassified. Role-one validation minority accuracy
  reached only **25%** and also missed the locked first inner correction.
- Follow-up target-hidden residual, q-shape tree, nearest-neighbor, and RBF
  kernel models all failed the same locked positions. The actual target-hidden
  teacher predicted them correctly (93-100% held-out rank accuracy), proving
  the LM-head reference was sound; learned draft-to-target mappings did not
  generalize.
- Why not to retry unchanged: linear hidden classification does not
  generalize beyond chronology despite perfect support headroom.
- Reopen only if: a target-hidden low-rank residual adapter or materially larger
  held-out corpus predicts the locked minority ranks without position leakage.
- Related commit or revert: no learned serving policy retained; the default-off
  proof-bearing diagnostic and backpressure repair remain useful.

## PERF-F070 - Compressed target KV as the remaining exact16 solution

- Hypothesis: a TurboQuant-style or rotated NVFP4 cache could cut enough XQA
  time to close the exact16 generation gap.
- Scope: exact SM120 XQA FP8 versus native NVFP4 at B1/Q3/QH24/KVH4/D256,
  page64, sequence 199000.
- Attempted change: measured nine alternating 101-call blocks and screened
  native Hadamard rotation before stock NVFP4 quantization.
- Benchmark evidence: FP8 median was **271.584 us**, NVFP4 **239.072 us**,
  only **0.520 ms/cycle** across 16 layers before transform/store overhead.
- Correctness evidence: stock NVFP4 previously corrupted reasoning/tools.
  Rotation left synthetic relative-L2 attention error effectively unchanged
  (**0.101824 -> 0.101624**).
- Failure mode: the full byte-reduction ceiling supplies only a fraction of
  the roughly 2 ms cycle need and fails the semantic boundary.
- Why not to retry unchanged: gentler FP8-K/4-bit-V saves fewer bytes; the
  stronger format already establishes the upper bound.
- Reopen only if: another candidate first removes a verify cycle and leaves a
  measured residual gap below the codec's independently proven net gain.
- Related commit or revert: no serving codec retained.

## PERF-F071 - Retune exact SM120 FP4 tactics and disable PDL

- Hypothesis: stale M4 file-cache tactics or unproductive CUTLASS PDL were
  stretching the exposed target NVFP4 family.
- Scope: all six exact target shapes, selected tactics, production occurrence
  counts, and the page64/delta-k1 selective server.
- Attempted change: swept all 32 precompiled tactics with bitwise-output
  filtering; separately rebuilt the SM120 module with PDL disabled; then
  changed only qkvz `12->4` and down `4->0` in a backed-up local tactic cache.
- Benchmark evidence: synthetic exact tactics projected **0.361 ms/cycle**.
  Global PDL-off regressed weighted shape time **6.373 -> 6.448 ms** and the
  flanking control was **6.469 ms**. The real tactic candidate averaged
  **123.972 tok/s** versus delta-only **123.831**, just +0.114%.
- Correctness evidence: all 32 tactics were bit-identical for the six saved
  shape outputs; every real request kept the established long digest.
- Failure mode: the isolated same-weight projection does not predict the
  captured distinct-layer target graph, and the real movement is noise-sized.
- Why not to retry unchanged: both PDL and the exact tactic pair were screened
  through flanking controls and real serving.
- Reopen only if: a current trace proves a specific layer-family span moves by
  at least 0.15 ms under a new mainloop, not merely a different stock tactic.
- Related commit or revert: installed FlashInfer header restored to SHA-256
  `A70A47370ED14EE8F88B4D93E54547DB6A23891AF2451F78CA1159DE0EDA312C`;
  original tactic cache restored to `BF50B56C...692E`.

## PERF-F072 - Full-target, draft-only, and partial-layer Marlin routes

- Hypothesis: weight-only Marlin would reduce enough small-M target or draft
  GEMM work to clear the exact16 generation target while Cutlass retained
  large prefill.
- Scope: selective checkpoint, exact `199000+16`, native draft-k1 proposal,
  in-place layout switching, and target projection/layer masks.
- Attempted change: screened full-target Marlin, draft-only Marlin, all target
  gate/up projections, layer halves/quarters, and cross-quarter masks. Added
  temporary CPU-side acceptance-length logging at the existing asynchronous
  result boundary; it was removed before promotion.
- Benchmark evidence: full-target Marlin reached **114.820 generation tok/s**
  but only **1984.193 prompt tok/s**. Draft-only stayed seven-cycle limited at
  **99.770**. Gate/up-only set the accepted **3078.058/114.617** record.
  Partial masks were non-monotonic: layers 32-63 returned the old 99.531 class;
  16-31 fell to 87.159; `0-7,16-23` used seven rounds at 114.503; other
  cross-quarter masks required eight or nine rounds.
- Correctness evidence: native relayout matches the canonical repacker
  bit-for-bit and round-trips exactly. Every full request completed `199016`
  with `finish_reason=length`; the promoted all-gate/up route passed reasoning,
  tools, tool continuation, model surface, and OpenCode2.
- Failure mode: full Marlin is unsuitable for large-M prefill, draft-only does
  not change the target acceptance trajectory, and partial target masks change
  floating-point reductions non-monotonically without reducing below the
  accepted route's seven verify rounds.
- Why not to retry unchanged: every projection family and representative layer
  mask was isolated with exact acceptance evidence. Narrowing did not produce a
  six-round request or exceed the all-gate/up record.
- Reopen only if: a new proposal law makes acceptance robust to target numeric
  perturbations or a Marlin mainloop improvement raises the accepted route
  above the 120 generation target without sacrificing prompt.
- Related commit or revert: only all 64 target gate/up projections remain in
  `03ba3d2e27`; draft-only, arbitrary shape/layer masks, and debug logging were
  removed.
## PERF-FA043 - Batch-one MLX ArraysCache merge/split bypass

- Hypothesis: a single-request auxiliary-state cache can avoid general
  merge/split handling and reduce the fixed decode round cost.
- Scope: MLX Qwen3.8 batch-one `ArraysCache` handling on the established
  Fast32K fixed-decode control.
- Attempted change: installed a batch-one bypass and compared five warmed
  end-to-end samples per arm with the same checkpoint and workload.
- Benchmark evidence: control times were `13.786914, 13.638193, 13.638745,
  13.647698, 13.625752 s` (mean **13.667460 s**); candidate times were
  `13.790976, 13.609490, 13.598677, 13.661740, 13.636127 s` (mean
  **13.659402 s**). The roughly 0.06% difference lies inside the window.
- Correctness evidence: the fixed workload completed with unchanged output
  behavior during the comparison.
- Failure mode: cache merge/split bookkeeping is not a material part of the
  batch-one end-to-end critical path.
- Why not to retry unchanged: the largest plausible effect is below the
  measurement noise and cannot clear the Apple record gate.
- Reopen only if: a trace on a materially different request topology shows
  cache composition on the serialized critical path.
- Related commit or revert: experimental code was removed before commit.

## PERF-FA044 - Always-on MLX quantized-prefill query tiling

- Hypothesis: dividing quantized attention by query rows would improve every
  prefill while reducing Metal score-matrix residency.
- Scope: Qwen3.8 quantized-KV prefill at a 64-row query tile, including the
  exact `5000+1` server control.
- Attempted change: forced query tiling without the later 1 GiB score-size
  admission threshold.
- Benchmark evidence: the established path completed exact `5000+1` in
  **59.078458 s / 84.633218 prompt tok/s**; always-tiled completed it in
  **59.271317 s / 84.357836 prompt tok/s**. Both returned token id 100.
  At the larger synthetic `Lq=1024,Lk=32768` shape, tiling improved
  **0.258692 -> 0.229053 s** and reduced peak allocation
  **1,732,382,776 -> 701,499,056 bytes**.
- Correctness evidence: causal helper parity and later complete Qwen3.5
  wrapper parity pass. The exact server control returned the same token.
- Failure mode: extra dispatch and concatenation cost has no small-prefill
  payoff, while large score matrices do benefit.
- Why not to retry unchanged: the process-wide always-on policy regresses the
  common small workload and offers no capacity benefit there.
- Reopen only if: dependency dispatch cost changes enough to make small score
  matrices measurably faster when tiled. The retained implementation instead
  gates tiling above a measured 1 GiB score estimate.
- Related commit or revert: superseded by the thresholded opt-in mechanism in
  `1271610e0b`.

## PERF-FA045 - Pinned llama.cpp IQ2 Metal server as the Apple record route

- Hypothesis: a current pinned llama.cpp Metal build could serve the retained
  IQ2 artifact fast enough to replace the unavailable MLX checkpoint and beat
  the Apple `12+256` scoreboard.
- Scope: official llama.cpp build 10547 at detached commit
  `749f688fcaa4c472ec034b08cb8a907c45cfaa02`, Release native ARM,
  Accelerate, embedded Metal, all layers on GPU, 32K context, one slot, and
  the OpenAI reasoning-preservation surface.
- Attempted change: no dependency source edits. Built the server in its
  separate checkout and ran one warmup plus five exact greedy, ignore-EOS
  scoreboard requests against the immutable Bartowski IQ2 checkpoint.
- Benchmark evidence: request-observed generation was `14.642054, 14.671473,
  14.660470, 14.665758, 14.667059 tok/s`, aggregate **14.661356**. This is
  **21.943%** below the 18.782925 aggregate record; its best sample is
  **22.046%** below the 18.820713 best-hit gate.
- Correctness evidence: every request generated exact 256 tokens with one
  stable digest. The OpenAI endpoint returned coherent preserved reasoning
  and final `703`, while its properties reported vision, video, and audio
  disabled.
- Failure mode: the pinned dependency is substantially faster than the
  repository's generic MPS route, yet its IQ2 Metal kernels still miss the
  existing MLX 4-bit record decisively.
- Why not to retry unchanged: five warmed samples were tightly clustered and
  the user closed further llama configuration work after seeing the result.
- Reopen only if: a dependency/kernel revision supplies a measured projection
  or decode improvement large enough to fund the roughly 28% throughput gain
  required from this baseline.
- Related commit or revert: no llama.cpp source change or commit; the detached
  checkout and binary remain as a reproducible supporting oracle.

## PERF-FA046 - Constant-address IQ2 lookup tables

- Hypothesis: leaving the 2 KiB IQ2 grid and 128-byte sign table in Metal
  constant address space would avoid the threadgroup copy/barrier and improve
  the selected four-row batch-one decoder.
- Scope: IQ2_XXS `17408x5120` batch-one matvec with identical two-SIMD,
  four-rows-per-SIMD geometry; only lookup-table residency changed.
- Attempted change: temporarily removed the threadgroup staging and read the
  immutable lookup arrays directly from constant memory. Ran two warmed
  matched windows around the retained staged implementation.
- Benchmark evidence: constant-table medians were **0.546042** and
  **0.576833 ms**. The staged path reached **0.523625 ms** in the matched
  window.
- Correctness evidence: both implementations retained actual-file tolerance;
  this was a residency-only ablation.
- Failure mode: repeated row-local random grid/sign accesses benefit from one
  cooperative staging pass; removing the copy/barrier increases steady lookup
  latency.
- Why not to retry unchanged: both independent constant-table windows trail
  the staged kernel.
- Reopen only if: the table access pattern, threadgroup geometry, or Metal
  constant-cache behavior changes materially.
- Related commit or revert: ablation removed; `16b2bf7a06` retains staging.

## PERF-FA047 - Four-SIMD two-row IQ2 batch-one geometry

- Hypothesis: four SIMDgroups producing two rows each would improve latency
  hiding while retaining the selected kernel's eight rows per threadgroup.
- Scope: IQ2_XXS batch-one matvec with the same staged tables, decode algebra,
  row count, and benchmark tensor.
- Attempted change: changed only the threadgroup geometry from two SIMDgroups
  × four rows to four SIMDgroups × two rows, retaining eight rows per group.
- Benchmark evidence: two warmed medians were **0.560625** and
  **0.550667 ms**, both above the matched selected two-by-four result around
  **0.523625 ms**.
- Correctness evidence: the row mapping retained actual-file tolerance and
  tail guards.
- Failure mode: the extra SIMDgroups add scheduling/resource pressure without
  creating more output work per threadgroup.
- Why not to retry unchanged: both windows lose consistently and the retained
  geometry already exposes ample grid parallelism.
- Reopen only if: the production matrix shapes or GPU SIMD occupancy change.
- Related commit or revert: ablation removed; `16b2bf7a06` retains two
  SIMDgroups × four rows.

## PERF-FA048 - Cross-row reuse in the F32 batch-one Metal matvec

- Hypothesis: share each F32 input tile across multiple output rows within a
  SIMDgroup so the `96x5120` GDN b/a projection reloads less input data.
- Scope: the custom native-MPS F32 dense matmul at the exact merged b/a shape,
  with only output-row/threadgroup geometry changed.
- Attempted change: measured two SIMDgroups × four rows, one SIMDgroup × four
  rows, and two SIMDgroups × two rows against the selected one-row-per-SIMD
  implementation. Every experimental shader change was removed afterward.
- Benchmark evidence: the selected control's 25-sample median was
  **0.390083 ms**. Two-SIMD/four-row measured **0.484833 ms**; one-SIMD/
  four-row measured **0.504208 ms**; two-SIMD/two-row measured
  **0.556667 ms**.
- Correctness evidence: each geometry retained CPU F32 tolerance before its
  timing decision. The restored source has no Metal diff.
- Failure mode: the matrix exposes only 96 output rows, and grouping rows
  reduces the threadgroup grid enough that lost occupancy outweighs shared
  input reads.
- Why not to retry unchanged: all three row-reuse geometries lose decisively
  on the exact production shape. The retained PERF-A009 route uses the faster
  system MPS matrix multiply instead.
- Reopen only if: a fused downstream consumer changes the matrix shape or
  removes enough launch/output traffic to offset the observed occupancy loss.
- Related commit or revert: no retained shader change.

## PERF-FA049 - OpenCode against the 1K native diagnostic launch

- Hypothesis: the short native-IQ2 server used for kernel qualification could
  also satisfy the required standalone OpenCode integration check.
- Scope: OpenCode 1.18.15 with a process-scoped OpenAI-compatible provider,
  tools advertised, thinking displayed, and the exact server on port 30000.
- Attempted change: left global OpenCode configuration untouched and invoked
  one bounded `opencode run --pure` request against the 1,024-token context
  and token pool.
- Benchmark evidence: OpenCode formed a **13,635-token** main agent prompt.
  SGLang rejected it with HTTP 400 because the live context limit was 1,024.
- Correctness evidence: the failure occurred at request admission before model
  execution. The same server had already passed direct reasoning, tools,
  preserved tool-result, and language-only checks.
- Failure mode: the real client's system/tool surface is more than thirteen
  times larger than the diagnostic launch's entire token pool.
- Why not to retry unchanged: prompt admission is mathematically impossible
  at 1,024 tokens, independent of decode speed or model output.
- Reopen only if: the native lane serves at least the measured 13,635-token
  prompt plus output headroom, then completes the same process-scoped request.
- Related commit or revert: no source change; PERF-A008 is the current native
  context-enabling candidate.

## PERF-FA050 - Unchanged 4K native-IQ2 prefill under the 300-second watchdog

- Hypothesis: the retained partial-extend correction made the existing generic
  quantized projection route fast enough to complete one 4,096-token outer
  chunk under the scheduler's default diagnostic watchdog.
- Scope: committed native-MPS IQ2 server, FP32 compute, BF16 KV, exact 32,768
  context/token pool, one request, 4,096-token prefill chunk, and ordinary
  rejection sampling.
- Attempted change: no source change. Started a clean exact-capacity launch,
  flushed its cache, and submitted exact `4096+2` with an 1,800-second client
  timeout.
- Benchmark evidence: the scheduler watchdog fired after the single forward
  remained active for 300 seconds. The request returned no token. A completed
  `128+32` prompt on the same launch measured 6.963 prompt tok/s, whose
  unchanged per-token rate predicts roughly 588 seconds for 4,096 tokens.
- Correctness evidence: the server reached ready state, reported image/audio
  false, and completed the sampled short gate with exact counts and separate
  reasoning. At timeout 28,672 of 32,768 token slots remained available;
  memory and thermal diagnostics were healthy.
- Failure mode: this request is censored by the watchdog, so it establishes
  only that the unchanged path is below 13.65 prompt tok/s at this shape.
  An independent actual-tensor sweep shows the current batch-eight IQ2 kernel
  traverses/dequantizes each packed matrix once per eight prompt rows.
- Why not to retry unchanged: completed short-prompt scaling already predicts
  a watchdog crossing, and the quantized projection sweep identifies a
  lower-level candidate with a direct matched benchmark.
- Reopen only if: PERF-A014 materially improves actual large-batch projections
  or a synchronized full-forward profile demonstrates a different dominant
  mechanism; then rerun with diagnostic ownership and an appropriate bound.
- Related commit or revert: no source change; full evidence is in the
  2026-08-21 00:36 and 00:41 experiment-log entries.
