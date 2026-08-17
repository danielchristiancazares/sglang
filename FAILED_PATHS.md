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
- Reopen only if: graph-specific profiling identifies a removable parent/bridge cost of at least 0.4 ms/cycle or CUDA/PyTorch gains a composition API that preserves runner bookkeeping without the current bridge work.
- Related commit or revert: retained opt-in in the pending exact linear device-cycle commit; launcher default remains off.
