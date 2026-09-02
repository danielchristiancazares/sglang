# Failed Paths

All Q4/Q4_0 benchmark evidence retained in this ledger came from separate Mac
Pro experiments. None of it was measured on the M1 Max or carries local record
standing.

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

The 2026-09-12 user-directed V4.1 follow-up also screened actual mixed-bit
residual coding rather than another E2M1 rotation. The retained native SM120
admission harness implements TurboQuant35's 4/3-bit groups, structured
Hadamard MSE projection, Lloyd-Max centroids, one-bit QJL residual, FP16
norms, fused quantize/store, packed-cache attention, and graph replay. Its
independent CPU encoder matched every CUDA payload byte and all norm fields.
The retained harness pins TurboQuant35 segment 512 at both lengths, FP8
segment 512 at 6,213, and FP8 segment 2,048 at 199K; segment 1,024 and the
other nonselected sweep call sites were removed. In the complete sweep, the
selected 199K layouts measured **11,540.973 us** for TurboQuant35 and
**3,074.226 us** for FP8, while relative-L2 error was **0.199575** with cosine
**0.980717**. The selected 6,213-token layouts measured **417.863 versus
119.514 us**, relative L2 **0.214638**. This is slower and less accurate than
the earlier admission bounds despite saving 53.125% of KV bytes. The isolated
benchmark remains available for future codec research; no cache ABI, launcher
option, or serving dispatch was added.

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
- Scope: MLX Qwen3.8 batch-one `ArraysCache` handling on a Mac Pro-only
  Fast32K fixed-decode control. This experiment supplies no M1 Max record.
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
- Why not to retry unchanged: the largest plausible effect was below the
  measurement noise in that cross-machine experiment.
- Reopen only if: a trace on a materially different request topology shows
  cache composition on the serialized critical path.
- Related commit or revert: experimental code was removed before commit.

## PERF-FA044 - Always-on MLX quantized-prefill query tiling

- Hypothesis: dividing quantized attention by query rows would improve every
  prefill while reducing Metal score-matrix residency.
- Scope: Mac Pro-only Qwen3.8 quantized-KV prefill at a 64-row query tile,
  including the exact `5000+1` server control. This experiment supplies no
  M1 Max record.
- Attempted change: forced query tiling without the later 1 GiB score-size
  admission threshold.
- Benchmark evidence: on the Mac Pro, the established path completed exact
  `5000+1` in **59.078458 s / 84.633218 prompt tok/s**; always-tiled completed
  it in
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

## PERF-FA051 - Simplified QK cache-staging lane map

- Hypothesis: assigning each lane one `(key_column, dimension)` pair with the
  simple `key_column=index/8`, `dimension=index&7` mapping would improve
  coalescing and reduce the address work in the Q8/C64 Metal EXTEND kernel.
- Scope: isolated BF16 paged-GQA attention at query length 256, prefix 4096,
  24 query heads, four KV heads, and dimension 256.
- Attempted change: replaced the selected four-consecutive-dimensions-per-lane
  QK staging map with the simpler eight-dimension lane mapping while retaining
  the same FP32 matrix arithmetic and online softmax.
- Benchmark evidence: the simplified mapping produced an approximately
  **63.08 ms** median in the matched isolated harness. The restored selected
  mapping produced approximately **51.57 ms** at the same shape.
- Correctness evidence: both mappings passed the dense MPS SDPA reference.
- Failure mode: the simpler mapping increases steady kernel time by about
  22%; its source-level coalescing intuition did not translate to the selected
  Apple7 matrix-stage geometry.
- Why not to retry unchanged: the same arithmetic and shape were measured
  directly, and the selected mapping won decisively.
- Reopen only if: a direct BF16 matrix-load path or a changed staging slab
  materially changes the cache-load transaction pattern.
- Related commit or revert: removed before the PERF-A017 checkpoint.

## PERF-FA052 - Paired PV matrix staging

- Hypothesis: staging two adjacent value matrices together would halve PV
  staging barriers in the Q8/C64 Metal EXTEND kernel.
- Scope: isolated BF16 paged-GQA attention with the same four-SIMD-group,
  dimension-256 production geometry.
- Attempted change: packed two 8x8 V fragments into each per-SIMD staging
  partition and consumed both before reusing the partition.
- Benchmark evidence: no timing score was retained because correctness failed
  at the first parity gate.
- Correctness evidence: output diverged materially from the dense causal SDPA
  reference, far outside the `atol=2e-5`, `rtol=2e-4` admission bound.
- Failure mode: the paired stage's row/stride interpretation did not preserve
  the required V matrix layout across both fragments.
- Why not to retry unchanged: a barrier reduction cannot fund a numerically
  invalid layout.
- Reopen only if: the fallback uses one explicit row-major `8x64` slab with
  stride 64 and independently validates every 8x8 matrix origin before timing.
- Related commit or revert: reverted before the PERF-A017 checkpoint; the
  selected kernel retains one 8x8 value stage at a time.

## PERF-FA053 - Import-time C++ hook for Metal EXTEND dispatch

- Hypothesis: a C++ `HookRegistry` or pybind monkeypatch installed during
  extension initialization could activate the direct-cache Metal kernel while
  satisfying the literal no-new-Python rule.
- Scope: `TorchNativeAttnBackend.forward_extend`, its private SDPA helper, and
  the lazy native Metal extension import.
- Attempted change: design review traced a paired outer/inner hook. The outer
  hook would carry `save_kv_cache=True` through thread-local state; the inner
  hook would gate the raw cache signature and delegate every other call.
- Benchmark evidence: no runtime score was taken because the architecture
  failed ownership and initialization review before implementation.
- Correctness evidence: shape-only inner hooking is insufficient because the
  private helper lacks `save_kv_cache`; direct callers and save-false calls can
  otherwise read stale cache state.
- Failure mode: extension bootstrap would own attention-backend policy through
  import-order-dependent mutation. HookRegistry late registration can conflict
  with its `_patched` set, class reload/subclass/compile behavior becomes
  implicit, and method-signature drift bypasses ordinary type checking.
- Why not to retry unchanged: implementing a Python dispatch mutation in C++
  satisfies the language restriction while placing policy outside its owning
  backend mechanism.
- Reopen only if: the backend owner explicitly approves a stable native hook
  contract with load-order, composition, save-false, exception, and fallback
  semantics.
- Related commit or revert: design rejected; no source change.

## PERF-FA054 - Global MPS aten SDPA override for EXTEND

- Hypothesis: registering a C++ MPS implementation of top-level aten scaled
  dot-product attention could activate bounded Metal attention without a new
  Python call site.
- Scope: every MPS SDPA caller in the process, including SGLang EXTEND and
  unrelated framework/model uses.
- Attempted change: source and dispatcher review confirmed an available MPS
  registration slot and a CompositeImplicitAutograd fallback route.
- Benchmark evidence: no kernel score was taken because the seam loses the
  direct BF16 cache and request-map provenance before dispatch.
- Correctness evidence: this boundary receives gathered FP32 dense K/V. It
  cannot call PERF-A017 and would require a second kernel plus complete
  autograd, compile, alias, mask, dropout, causality, and fallback coverage.
- Failure mode: the global operator owns a much broader semantic domain than
  the Qwen batch-one EXTEND rule. It retains roughly 0.5 GiB of BF16 gather
  plus 1 GiB of compact FP32 K/V at 128K and expands regression scope to every
  MPS attention caller.
- Why not to retry unchanged: the available schema lacks the metadata that
  makes the direct-cache mechanism safe and memory-flat.
- Reopen only if: a separately funded dense-input kernel implements the full
  aten contract and matched profiling shows that retained gather/cast traffic
  still clears memory and throughput gates.
- Related commit or revert: design rejected; no source change.

## PERF-FA055 - Per-consumer consecutive-run classification

- Hypothesis: each QK/PV consumer could recognize its own consecutive
  eight-slot cache run immediately before issuing a direct BF16 SIMD-matrix
  load, avoiding shared classifier state.
- Scope: PERF-A018's Q8/C64 native Metal BF16 paged-GQA EXTEND kernel.
- Attempted change: every SIMDgroup repeatedly loaded and compared the same
  eight-slot cohorts inside the dimension and value-fragment loops.
- Benchmark evidence: the direct ascending path improved, reaching a median
  of **44.449479 ms** at `E=256,L=4352`, while a zero-eligible shuffled map
  regressed from a matched staged **59.839625 ms** median to about
  **66.746 ms**, roughly **11.5%**.
- Correctness evidence: direct and staged outputs were bitwise equal for the
  exercised sequential and shuffled placements. This branch closed on its
  fragmented-map performance cost.
- Failure mode: identical eligibility work was repeated across QK dimension
  slices and PV output fragments, taxing every fallback run without changing
  the classification result.
- Why not to retry unchanged: classifier cost must be amortized per C64 tile,
  especially for recycled or fragmented paged-cache maps.
- Reopen only if: a new design proves one bounded classification pass per tile
  or a cheaper producer-register publication scheme, with a zero-eligible
  regression below 1%.
- Related commit or revert: superseded before the PERF-A018 checkpoint by one
  cooperative eight-thread classifier pass and a 32-byte shared run table.

## PERF-FA056 - Hoist all query matrix fragments across C64 tiles

- Hypothesis: loading the invariant Q8xD256 query into 32 FP32 SIMD-matrix
  fragments once per threadgroup would remove repeated threadgroup reads from
  every C64 key tile.
- Scope: PERF-A018's native Metal BF16 paged-GQA EXTEND kernel, with both
  consecutive direct loads and staged fragmented loads.
- Attempted change: a 32-entry `simdgroup_float8x8` array was populated before
  the key loop and reused at the existing QK multiply sites. Operand values
  and multiply-accumulate order were unchanged.
- Benchmark evidence: at `E=256,L=4352`, direct medians moved from the
  PERF-A018 **26.801604/26.900646 ms** windows to
  **27.710500/27.846354 ms**. The zero-eligible map moved from
  **58.773563** to **62.621042 ms**. These are roughly 3.4-6.5% regressions.
- Correctness evidence: sequential direct and one-swap-per-eight fallback
  outputs stayed bitwise equal and retained SHA-256
  `7209fefe46186dbbc09aa0ce1a675b5c3056d6add4091d1bf7622aff84236371`.
- Failure mode: the long-lived 32-matrix query array materially increases the
  shader's live register surface; measured timing is consistent with
  register-pressure or scheduling cost overwhelming the saved shared reads.
- Why not to retry unchanged: both the primary direct route and the fallback
  route miss their retention gates before a long-context score is needed.
- Reopen only if: query fragments can be tiled into a much smaller live set,
  compiler register allocation is inspected, and an interleaved short-window
  result clears the selected PERF-A018 medians by at least 2%.
- Related commit or revert: reverted before PERF-A019; no kernel source from
  this candidate remains.

## PERF-FA057 - Uniform fully-causal C64 branch

- Hypothesis: history tiles ending before the earliest query row's causal
  limit could bypass each lane's redundant logical-token comparison.
- Scope: PERF-A019's native Metal BF16 paged-GQA EXTEND softmax mask.
- Attempted change: one threadgroup-uniform predicate identified fully causal
  C64 tiles; diagonal and partial tiles retained the established comparison.
- Benchmark evidence: at `E=256,L=4352`, candidate direct medians were
  **26.358104/26.316271 ms** against an adjacent selected-source
  **26.330084 ms** median. At `E=17,L=131072`, candidate **65.734208 ms**
  trailed the matched **65.647625 ms** median.
- Correctness evidence: direct and fallback outputs stayed bitwise equal and
  retained the established small and timing-fixture SHA-256 digests.
- Failure mode: the uniform branch and predicate cost offset the eliminated
  comparisons; both diagnostic and long-context windows were neutral to
  slightly slower.
- Why not to retry unchanged: the candidate misses the 1% retention floor at
  its highest-coverage long-context shape.
- Reopen only if: compiler inspection proves the comparison remains in the
  current shader and a branch-free specialization can remove it without an
  added uniform predicate in the tile loop.
- Related commit or revert: reverted before PERF-A020.

## PERF-FA058 - Cohort-leader online-softmax state broadcast

- Hypothesis: one lane per 16-lane softmax cohort could load the prior
  maximum/sum, compute `next_max` and `old_scale`, then broadcast two FP32
  values, reducing repeated shared reads and exponent evaluations.
- Scope: PERF-A019's C64 online-softmax update.
- Attempted change: physical lanes 0 and 16 owned prior-state arithmetic;
  `simd_shuffle` distributed the exact next maximum and rescale to their
  cohorts. Accumulation order and stored state were unchanged.
- Benchmark evidence: `E=256,L=4352` direct medians were
  **26.354417/26.513146 ms** versus a nearby selected-source
  **26.330084 ms** median; the zero-eligible median was **58.963146 ms**
  versus **58.738667 ms**. At `E=17,L=131072`, candidate
  **65.733625 ms** trailed the matched **65.647625 ms** median.
- Correctness evidence: small direct/fallback parity remained bitwise exact,
  dense error stayed `5.3644180e-07`, and both timing digests were unchanged.
- Failure mode: two cohort shuffles and leader control do not repay the
  vector-issued transcendental/shared-state work on this Metal target.
- Why not to retry unchanged: every scored route stays inside noise or moves
  slightly slower.
- Reopen only if: profiling demonstrates active-lane transcendental pressure
  on a different Apple family and a matched window clears 0.75%.
- Related commit or revert: reverted before PERF-A020.

## PERF-FA059 through PERF-FA063 - Direct device-matrix-load barrier ablations

- Hypothesis: the `simdgroup_barrier(mem_flags::mem_none)` calls immediately
  before and after direct BF16 `simdgroup_load` operations could be removed
  because the QK/PV branches are SIMDgroup-uniform and each loaded fragment is
  consumed by a dependent SIMDgroup matrix multiply.
- Scope: PERF-A020's direct QK and PV routes on the M1 Max, with fallback
  `mem_threadgroup` synchronization, arithmetic, operands, tiling, scratch,
  and host dispatch unchanged.
- Attempted changes:
  - `PERF-FA059`: removed the PV leading/trailing pair;
  - `PERF-FA060`: removed both QK and PV pairs;
  - `PERF-FA061`: removed the QK leading/trailing pair;
  - `PERF-FA062`: removed only the QK/PV trailing consumer barriers;
  - `PERF-FA063`: removed only the QK/PV leading barriers.
- Benchmark evidence: opening PERF-A020 medians were
  **25.686792/63.915500 ms** at `E=256,L=4352` and `E=17,L=131072`.
  The five arms measured respectively
  **25.591584/63.552417**, **25.847500/63.455208**,
  **25.624625/63.901375**, **25.907875/63.845042**, and
  **25.575896/63.756833 ms**. The largest apparent reduction was only
  **0.720%**, while the final independently rebuilt PERF-A020 control reached
  **25.520083/63.745709 ms**.
- Correctness evidence: all arms retained exact SHA-256
  `7ec60bdc0d473fba047e79855bb9f95c54358cb1c281135797e2950756116a42`
  and `820241644b85aaa0fed24caf91cffef2254cd1bfd6cb93fab2aaf2b6727d1992`,
  unchanged checksums, and zero current-memory growth.
- Failure mode: the selected Metal compiler/GPU makes these execution-only
  barriers too cheap, already scheduled effectively, or beneficial enough to
  offset their source-visible count. None of the five isolations reached the
  predeclared reproduced 1% timing floor, and several diagnostic arms
  regressed.
- Why not to retry unchanged: leading, trailing, QK-only, PV-only, and combined
  removals exhaust the source-local synchronization combinations without a
  qualifying signal.
- Reopen only if: generated AIR/ISA on a different Metal compiler or Apple GPU
  proves materially different barrier code generation and a matched direct
  workload clears 1% in two windows.
- Related commit or revert: every shader edit was reverted; signed PERF-A020 at
  `5fe532b41c` is restored. The record-only closure commit follows this entry.

## PERF-FA064 - Apple MPS extra-buffer cache under non-overlap scheduling

- Hypothesis: eager `extra_buffer` would retain aligned recurrent checkpoints
  for divergent Codex task suffixes while preserving the established Apple
  non-overlap scheduler.
- Scope: Qwen3.8-27B IQ2_XXS on the M1 Max, exact 131,072-token context and
  BF16 KV pool, five FP32 Mamba slots, page size one, seed 67396869, and the
  qualified reasoning/tool parser pair.
- Attempted change: changed only the selected cache strategy from `no_buffer`
  to `extra_buffer`; retained `--disable-overlap-schedule`.
- Benchmark evidence: weight load, five-slot Mamba allocation, exact KV-pool
  allocation, and UnifiedRadixCache startup completed. The first six-token
  warmup prefetched successfully, then the first decode transition crashed
  before an external request could be scored.
- Correctness evidence: the fatal stack identifies
  `set_mamba_track_indices_from_reqs` and the pinned-CPU-to-MPS copy, ending in
  `at::native::mps::mps_copy_` and the AGX blit path. The scheduler exited
  `-11`; macOS retained the exact crash report. Cleanup left the listener and
  relevant process sets empty.
- Failure mode: current eager extra-buffer tracking takes an unsafe MPS device
  transfer path during decode preparation.
- Why not to retry unchanged: the configuration cannot pass its built-in
  startup generation gate and therefore cannot serve real client traffic.
- Reopen only if: the track-index construction gains an MPS-safe lifetime and
  device-transfer implementation with focused replay coverage, or a dependency
  change demonstrably removes the same crash.
- Related commit or revert: no source change; full evidence is in the
  2026-08-31 01:59 experiment-log entry.

## PERF-FA065 - Cap split-history Metal decode at sixteen partitions

- Hypothesis: a sixteen-way cap would avoid idle split threadgroups at the
  6.2K-token Codex history while still dividing the exact 131K history evenly.
- Scope: native Apple7 BF16 batch-one GQA decode on the M1 Max; current
  131,073-row physical cache, Qwen 24/4 heads, head dimension 256, and the
  retained two-kernel stable softmax merge.
- Attempted change: added a native `SGLANG_MPS_SPLIT_DECODE_MAX_SPLITS`
  control and selected sixteen by default; the 32-way value reproduced the
  committed control in separate processes.
- Benchmark evidence: focused medians changed **0.919091 -> 0.472529 ms** at
  sequence length 6,234 and **0.161865 -> 0.140490 ms** at length 16. Exact
  131,072-token history remained effectively flat at **4.066904 ->
  4.078974 ms**. A matched full-model six-Mamba-slot 131K window then measured
  warmed exact `128+256` decode at **9.190 tok/s** for the 32-way control and
  **9.122 tok/s** for the sixteen-way candidate. The identical output digest
  and all count/finish gates passed. The candidate therefore failed the served
  promotion gate despite its isolated long-history reduction.
- Correctness evidence: BF16 reference parity passed lengths 1, 257, 1,024,
  1,025, 6,234, and 131,072 with maximum error
  `5.066394805908203e-07`; a fragmented map and twelve unsynchronized outputs
  retained twelve distinct backing storages with maximum error
  `3.725290298461914e-07`. The native extension compiled, the existing MPS
  attention smoke passed at maximum error `5.96046e-07`, three dispatch unit
  tests passed, and `git diff --check` passed.
- Failure mode: the saved full-attention work is too small a fraction of the
  full token wall, and the warmed end-to-end window moved 0.74% lower.
- Why not to retry unchanged: the user-visible generation rate owns promotion;
  isolated attention speed alone cannot fund an end-to-end regression.
- Reopen only if: whole-token attribution shows a new long-history server path
  where the same cap clears a matched served window and the 20 tok/s floor.
- Related commit or revert: the Objective-C++ edit was fully removed before
  the next profile; no candidate source remains.

## PERF-FA066 - Full Qwen3.8-27B affine 3-bit MLX checkpoint

- Hypothesis: reducing the full 27B weight stream from affine q4 to affine
  q3 would save enough Metal memory bandwidth to clear 20 tok/s.
- Scope: immutable revision
  `c98bba5926f51fec1c8d8737e577221673f524d7` of
  `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`, real 131,072 context/token
  pools, q4 KV, one request, MLX 0.32.2, and the existing SGLang sampling and
  parser surface.
- Attempted change: launched the prequantized group-64, three-bit checkpoint
  through the MLX runner. The config's empty quantization method required the
  existing `--quantization mlx_q4` loader selection; model loading detected
  the stored affine-q3 tensors and preserved them.
- Benchmark evidence: three exact deterministic `128+256` server samples were
  **17.972, 17.961, and 17.941 tok/s**, mean **17.958**. The selected
  affine-q4/q4-KV endpoint averages **19.1432 tok/s** on the same dependency,
  so the smaller checkpoint is about 6.2% slower.
- Correctness evidence: all three requests completed 128 prompt and 256
  completion tokens, ended with `finish_reason=length`, preserved reasoning,
  and reproduced output SHA-256 `2f8a3468...212d`.
- Failure mode: the three-bit affine kernel's unpack/compute efficiency costs
  more than the reduced packed-weight traffic saves on this M1 Max/MLX build.
- Why not to retry unchanged: the immutable checkpoint and current MLX kernel
  miss the selected endpoint by more than 1.18 tok/s before sampling cost.
- Reopen only if: a native three-bit batch-one matvec specialization or an MLX
  dependency change demonstrates a direct target-loop gain over affine q4.
- Related commit or revert: no source change; checkpoint remains an immutable
  local cache artifact.

## PERF-FA067 - Official Qwen3.8-27B MLX MXFP4 checkpoint

- Hypothesis: MXFP4's group-32 block format would reduce affine metadata and
  improve batch-one quantized matvec throughput.
- Scope: immutable revision
  `97ab0819817ab1c61d7d39f9169fc71999915641` of
  `mlx-community/Qwen3.8-27B-mxfp4`, MLX 0.32.2, a 128-token input, 32 warm
  decode tokens, 256 timed tokens, and q4 attention KV.
- Attempted change: loaded the official full 27B MXFP4 checkpoint through
  `mlx_lm` and exercised the same direct `generate_step` loop as the selected
  affine-q4 control.
- Benchmark evidence: MXFP4 reached **18.581608 tok/s** over 13.777064 seconds.
  The matched affine-q4/q4-KV loop reached **19.513158 tok/s** over
  13.119353 seconds. MXFP4 is 4.774% slower.
- Correctness evidence: both arms completed the exact 32-token warmup and
  256-token timing interval and returned stable token-stream digests for their
  respective checkpoints.
- Failure mode: current MLX MXFP4 batch-one execution has higher per-token cost
  than affine q4 on this model and GPU.
- Why not to retry unchanged: the regression occurs inside the direct target
  loop, before SGLang scheduling, streaming, or sampling overhead.
- Reopen only if: MLX ships a changed MXFP4 Metal kernel or the checkpoint is
  paired with measured fused operators that reverse the direct-loop result.
- Related commit or revert: no source change; checkpoint remains an immutable
  local cache artifact.

## PERF-FA068 - Full Qwen3.8-27B affine 2-bit MLX checkpoint

- Hypothesis: reducing every quantized model projection to affine q2 would
  lower batch-one weight traffic enough to clear the 20 tok/s floor while
  retaining Qwen3.8 reasoning and tool behavior.
- Scope: immutable revision
  `33b90b60fd7ba16b668854e049bd65e22d6afddf` of
  `lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly`, MLX 0.32.2, BF16 attention KV,
  one running request, and real 131,072-token context and token pools.
- Attempted change: loaded the complete group-64 affine-q2 checkpoint through
  the existing MLX runner and exercised deterministic, production-sampled,
  arithmetic, and tool-call requests.
- Benchmark evidence: the direct target loop reached **21.134311 tok/s** with
  BF16 KV. Five deterministic exact `128+256` server samples were
  **21.161, 21.066, 21.054, 21.046, and 21.050 tok/s**, mean **21.0754**.
  Five production-sampled samples were
  **20.890, 20.902, 20.912, 20.900, and 20.912 tok/s**, mean **20.9032**.
- Correctness evidence: `/model_info` retained the language-only surface. The
  sampled arithmetic request ended with empty completion content instead of
  `703`; the tool request emitted repetitive text and no valid parsed call.
- Failure mode: whole-model q2 clears the throughput floor and loses the
  semantic behavior required for actual work.
- Why not to retry unchanged: both required behavior probes fail on the exact
  full-model checkpoint that produced the speed result.
- Reopen only if: a changed q2 checkpoint demonstrates the arithmetic and
  exact single-tool-call gates, or layer-level sensitivity evidence supports
  a distinct mixed-precision selection.
- Related commit or revert: no repository source change; checkpoint remains an
  immutable local cache artifact.

## PERF-FA069 - Group-128 affine requantization of the q4 checkpoint

- Hypothesis: doubling the affine group size would reduce scale/bias traffic
  while retaining the selected q4 model's behavior.
- Scope: in-memory MLX 0.32.2 requantization screens on the immutable affine-q4
  checkpoint, a 128-token input, 32 warm tokens, 256 timed tokens, and BF16
  attention KV.
- Attempted change: first requantized all 64 MLP down projections, then
  broadened the group-128 selection to 385 quantized linear modules while
  retaining the linear-attention `in_proj_z`, `out_proj`, and full-attention
  output projections at their original group size.
- Benchmark evidence: down-only group 128 reached **19.681944 tok/s** with
  BF16 KV. The broader 385-module selection reached **19.781749 tok/s**. The
  selected affine-q4/BF16 direct loop is **19.643293 tok/s** and the candidate
  remains below the direct-loop margin required for a 20 tok/s sampled server.
- Correctness evidence: both screens completed the exact warmup and timing
  interval. The candidate stopped at the throughput screen before promotion
  behavior gates.
- Failure mode: lower affine metadata traffic yields less than 0.71% over the
  direct q4/BF16 control, leaving scheduler and sampling overhead unfunded.
- Why not to retry unchanged: the broad selection already covers 385 modules
  and remains about 0.22 tok/s below the absolute floor in the direct loop.
- Reopen only if: an MLX kernel change materially increases group-128
  batch-one efficiency or a measured selection exceeds the served-workload
  promotion margin.
- Related commit or revert: no checkpoint was written and no repository source
  change remains.

## PERF-FA070 - Q2 linear-attention and gate/up mixed checkpoint

- Hypothesis: retain q4 embeddings, head, MLP down projections, and all full
  attention blocks while using q2 for linear-attention modules and MLP gate/up
  projections, combining full-q2 speed with q4 semantic anchors.
- Scope: derived immutable artifact
  `Qwen3.8-27B-MLX-Q2GDN-Q4Anchors-v1`, assembled from the affine-q4 base and
  PERF-FA068 donor with 368 per-module q2 overrides, MLX 0.32.2, BF16 KV, one
  request, and real 131,072-token context and token pools.
- Attempted change: substituted the donor's 128 MLP gate/up and 240
  linear-attention quantized modules, preserved every MLP down projection and
  full-attention block from q4, and wrote a provenance manifest with hashes for
  every artifact file.
- Benchmark evidence: the reloaded artifact reached **20.526347 tok/s** in the
  direct target loop. Five deterministic exact `128+256` server samples were
  **20.309, 20.287, 20.286, 20.291, and 20.299 tok/s**, mean **20.2944**.
  Five production-sampled samples were
  **20.144, 20.145, 20.148, 20.156, and 20.148 tok/s**, mean **20.1482**.
- Correctness evidence: sampled arithmetic returned `703` and `/model_info`
  retained the language-only surface. The required tool request emitted two
  malformed calls named `...` and ended by length instead of one parsed
  `multiply({"a":37,"b":19})` call with `finish_reason=tool_calls`.
- Failure mode: this precision boundary clears throughput and arithmetic while
  damaging structured tool-call behavior.
- Why not to retry unchanged: an exact server gate reproduces the malformed
  tool behavior on the reloaded, hashed artifact.
- Reopen only if: a narrower q2 linear-attention selection retains one exact
  multiply call and clears the sampled throughput floor.
- Related commit or revert: artifact v1 remains immutable and unqualified;
  repository source is unchanged.

## PERF-FA071 - YoozLabs quality-aware q3/q4/q6 MLX checkpoint

- Hypothesis: the checkpoint's long-context-aware mixed precision would retain
  Qwen3.8 reasoning and tool quality while its lower average weight width
  cleared the 20 tok/s floor.
- Scope: immutable revision
  `55c317fadb679431afef61ddd97a4ac2522ca420` of
  `YoozLabs/Qwen3.8-27B-lean-4bit-mlx`, MLX 0.32.2, a 128-token input,
  32 warm tokens, 256 timed tokens, and BF16 attention KV.
- Attempted change: loaded the published q3 gate/up, q6 self-attention value
  and language-head, and q4 remainder through the same direct target loop used
  for the active affine checkpoints.
- Benchmark evidence: the exact direct loop reached **18.553117 tok/s**.
- Correctness evidence: the immutable model loaded and completed the exact
  warmup and timed token counts. Its model card's quality claims remain
  external evidence; the local performance screen stopped before serving.
- Failure mode: current MLX affine-q3 batch-one execution makes this
  quality-aware layout about 7.2% slower than the absolute throughput floor
  before scheduler and sampling costs.
- Why not to retry unchanged: the deficit occurs in the direct model loop and
  leaves no server-overhead margin.
- Reopen only if: a native q3 matvec specialization or changed MLX q3 kernel
  demonstrates at least an 8% direct-loop gain on this exact artifact.
- Related commit or revert: no repository source change; the pinned checkpoint
  remains a redownloadable cache artifact.

## PERF-FA072 - PocketAiHub group-32 q2 AWQ checkpoint and selective mixing

- Hypothesis: group-32 AWQ q2 weights would provide the full-q2 bandwidth win
  with better quality than RTN q2, either unchanged or as selective overrides
  on the affine-q4 base.
- Scope: immutable revision
  `dcc3732f8c93ccf5580bf7a55e4ae639a40f194c` of
  `PocketAiHub/Qwen3.8-27B-MLX`, its `2bit` artifact, MLX 0.32.2, BF16 KV,
  and the exact direct Qwen tool prompt.
- Attempted change: measured the complete AWQ checkpoint, then substituted its
  gate/up and linear-attention modules into the q4 base; a broader arm also
  substituted the MLP down projections.
- Benchmark evidence: the complete checkpoint reached **20.796459 tok/s**.
  The gate/up plus linear-attention mix reached **20.298 tok/s**; adding down
  projections reached **20.758 tok/s**.
- Correctness evidence: the complete checkpoint emitted placeholder-example
  loops instead of one multiply call. The first selective mix produced empty
  output, and the broader mix produced only `</think>`.
- Failure mode: AWQ's transformed module weights are not independently
  interchangeable with the q4 model, while the complete artifact fails the
  required raw tool behavior.
- Why not to retry unchanged: all measured full and mixed forms fail before a
  parsed server tool gate despite clearing or approaching the speed floor.
- Reopen only if: the artifact's complete AWQ transform metadata can be
  applied coherently at a validated layer boundary and exact tool behavior
  passes before serving.
- Related commit or revert: no repository source change; the pinned checkpoint
  remains a redownloadable cache artifact.

## PERF-FA073 - Four-step MLX streaming and scheduler receive cadence

- Hypothesis: reducing output and receive bookkeeping from every token to
  every four tokens would raise the sampled mixed-checkpoint floor.
- Scope: `Qwen3.8-27B-MLX-Q2Expand-QKVZ-EarlyOut27-v2`, radix disabled,
  BF16 KV, real 131K pools, one request, and the required sampled `128+256`
  workload.
- Attempted change: changed only `--stream-interval` and
  `--scheduler-recv-interval` from one to four.
- Benchmark evidence: five candidate samples were
  **20.082, 20.074, 20.067, 20.082, and 20.076 tok/s**, mean **20.0762**.
  The matched one-step mean was **20.0626 tok/s**, a
  **0.0136 tok/s / 0.068%** difference.
- Correctness evidence: every exact timing request completed 128 prompt and
  256 sampled output tokens.
- Failure mode: the difference is below ordinary run noise, and source tracing
  shows the MLX overlap loop receives requests through its direct receiver
  call rather than the generic scheduler receive-interval path.
- Why not to retry unchanged: the five-sample server window produces no
  material user-visible gain.
- Reopen only if: scheduler attribution identifies cadence bookkeeping above
  0.25 ms/token on the reachable MLX path.
- Related commit or revert: no source change; the radix launch retains the
  four-step output cadence as a low-cost configuration choice.

## PERF-FA074 - Early-out27 selective RTN q2 artifact as an actual-work lane

- Hypothesis: q2 gate/up, qkv, z, and the first 27 linear-attention output
  projections would preserve the raw arithmetic and tool boundary while
  funding sampled serving above 20 tok/s.
- Scope: immutable derived artifact
  `Qwen3.8-27B-MLX-Q2Expand-QKVZ-EarlyOut27-v2`, MLX 0.32.2, BF16 KV,
  real 131,072 context and token pools, five auxiliary-state slots, radix
  prefix caching, and the frozen Codex 0.151.0 xhigh client.
- Attempted change: selected 251 RTN-q2 modules over the affine-q4 base and
  served them with 512-token prefill chunks and four-step output cadence.
- Benchmark evidence: five radix-enabled sampled `128+256` samples were
  **20.091, 20.092, 20.074, 20.068, and 20.081 tok/s**, mean
  **20.0812**. During the real 6.2K-token Codex turn, server telemetry fell to
  about **19.2 tok/s**.
- Correctness evidence: standalone sampled arithmetic returned `703` and the
  first tool probe produced exactly one parsed multiply call. Three sampled
  continuation cycles then produced one contradictory length-truncated answer,
  one clean `703`, and one duplicate multiply call. The frozen xhigh Codex
  turn emitted a blank message and an invalid exec request; its continuation
  ended in a Metal out-of-memory command-buffer failure.
- Failure mode: the short timing window clears the floor narrowly, while
  realistic context loses that floor, structured behavior is unstable, and
  the radix continuation exceeds available Metal residency.
- Why not to retry unchanged: the exact frozen-client gate reproduces all
  three actual-work failures on the hashed artifact.
- Reopen only if: a memory-residency change survives the same continuation,
  long-prefix decode stays at or above 20 tok/s, and repeated tool cycles are
  stable.
- Related commit or revert: artifact v2 remains immutable and unqualified;
  repository source is unchanged.

## PERF-FA075 - One-GiB MLX recycled-buffer cache cap

- Hypothesis: bounding MLX's recycled Metal buffers before model load would
  release enough transient residency for the 6.2K cached-prefix continuation.
- Scope: early-out27 v2, BF16 131K shared KV pool, five auxiliary slots,
  512-token prefill chunks, radix enabled, and
  `SGLANG_MLX_CACHE_LIMIT_GB=1` as the only memory change.
- Attempted change: applied the existing pre-load cache cap and replayed both
  the sampled short window and a deterministic 6,257-token two-request radix
  continuation.
- Benchmark evidence: five short sampled samples were
  **20.069, 20.069, 20.063, 20.068, and 20.054 tok/s**, mean
  **20.0646**, only 0.083% below the uncapped mean. The first long request
  completed; the immediate second request disconnected during its prefix-hit
  extend.
- Correctness evidence: server logs identify the same
  `kIOGPUCommandBufferCallbackErrorOutOfMemory` at
  `tp_worker._async_extend_batch -> mx.async_eval`.
- Failure mode: recycled-buffer residency is not the dominant peak; live
  model, pool, and fallback prefill state exceed Metal's working set.
- Why not to retry unchanged: the exact bounded reproducer reaches the same
  crash while the cap has already demonstrated throughput neutrality.
- Reopen only if: independent allocator telemetry shows more than 1 GiB of
  reclaimable cache remains live at the failing submission.
- Related commit or revert: no source change; the environment override is
  rejected as a standalone fix.

## PERF-FA076 - Halve cached-prefix prefill chunks to 256

- Hypothesis: halving the new-token extend graph would cut the transient peak
  enough for the cached-prefix continuation to complete.
- Scope: PERF-FA075's exact early-out27 v2 launch and bounded 6,257-token radix
  replay, changing only `--chunked-prefill-size 512 -> 256`.
- Attempted change: restarted cleanly with 256-token chunks and replayed the
  same two requests.
- Benchmark evidence: cold chunks held about 106--110 prompt tok/s. The first
  request completed, then the second request failed after its cache match.
- Correctness evidence: the failure again occurred in
  `_async_extend_batch -> mx.async_eval` with Metal insufficient memory.
- Failure mode: source tracing and the control show that the restore misses
  deferred auxiliary-state COW and reruns the entire cached prompt; the
  6,144-token fallback graph dominates the new 256-token chunk.
- Why not to retry unchanged: 512 and 256 produce the same failure at the same
  request boundary.
- Reopen only if: auxiliary restore succeeds and profiling then attributes a
  residual memory peak to the new-token chunk itself.
- Related commit or revert: no source change; the server was stopped and the
  512-token production-shaped baseline remains authoritative.

## PERF-FA077 - Native graph through the Python chunked-prefill handoff

- Hypothesis: selecting the native C++ graph with ordinary 4,096-token chunks
  would preserve the native path across a realistic multi-chunk Codex prompt.
- Scope: early-out27 v2, native graph enabled, one request, real 131,072
  context/token pools, and a deterministic 6,257-token prompt.
- Attempted change: launched with `--chunked-prefill-size 4096` and allowed the
  scheduler to hand the unfinished request to the next extend chunk.
- Benchmark evidence: the first 4,096-token native chunk ran at about
  **57.79 prompt tok/s**. The next chunk reached
  `MlxModelRunner.extend_start` and raised
  `TypeError: 'types.SimpleNamespace' object is not callable`.
- Correctness evidence: source tracing shows the native route installs a
  `SimpleNamespace` model surface while the later chunk invokes
  `self.model(...)`. A single 8,192-token chunk completes the same 6,257-token
  prompt natively.
- Failure mode: the Python chunk transition leaves the compiled engine path
  and calls a model object that is intentionally non-callable in native mode.
- Why not to retry unchanged: chunk size alone cannot make the second native
  chunk reachable through the current dispatch contract.
- Reopen only if: the compiled C++ engine gains a native multi-chunk prefill
  entry point, or the shared dispatch owner routes every chunk through the
  existing native C ABI without adding Python implementation code.
- Related commit or revert: no source change retained; the 8,192-token launch
  is an experiment-only bridge and does not establish 131K prompt capacity.

## PERF-FA078 - Online native MLX split-K decode attention

- Hypothesis: dividing a 6.2K BF16 attention history across independent Metal
  workgroups would overcome the serial decode cost left after reusable K/V
  storage.
- Scope: native early-out27 v2, exact 6,237-token deterministic history, 32
  warm tokens, 128 timed tokens, 16 or 32 history splits, and 256-dimensional
  24-query/4-KV-head GQA.
- Attempted change: first assigned four SIMD groups to each query-head/split;
  then grouped paired query heads over one shared K/V stream with 16 and 32
  splits. A second kernel merged numerically stable softmax partials.
- Benchmark evidence: per-query split-16 reached **15.931997 tok/s**. The
  paired-head split-16 and split-32 variants reached **13.068068** and
  **15.863200 tok/s**. The reusable-cache MLX SDPA control is
  **19.151623 tok/s**.
- Correctness evidence: every arm reproduced the exact control digest
  `382dd93cb724783226eae6ede000d6b62bbbc6439c8a39178cb9bb0ba8a27112`.
- Failure mode: per-query work overproduces Metal groups and duplicate cache
  reads; paired-head sharing leaves too little latency-hiding work per group.
- Why not to retry unchanged: both sides of the occupancy tradeoff were
  measured and each is materially slower than MLX SDPA.
- Reopen only if: one workgroup can reuse K/V across all six GQA heads through
  matrix tiles or a fused reduction eliminates the second dispatch.
- Related commit or revert: experimental C++/Metal source was removed.

## PERF-FA079 - Tiled simdgroup-matrix native MLX decode attention

- Hypothesis: porting the retained 8-query by 64-key tiled MPS kernel to MLX's
  custom Metal interface over contiguous BF16 caches would beat MLX SDPA.
- Scope: the PERF-FA078 exact workload, one workgroup per KV-head/history
  split, four SIMD groups, bounded 20.1-KiB threadgroup storage, fast math, and
  8, 16, or 32 splits.
- Attempted change: shared six GQA query heads per KV tile, used BF16
  simdgroup-matrix QK/PV operations, retained online softmax partials, and
  merged them in a 256-thread reduction kernel.
- Benchmark evidence: 8, 16, and 32 splits reached **18.475598**,
  **19.117317**, and **18.922351 tok/s**, respectively, against the
  **19.151623 tok/s** reusable-cache MLX SDPA control. Disabling row-contiguous
  normalization and selecting fast math changed the 8-split arm only from
  **18.460132** to **18.475598 tok/s**.
- Correctness evidence: every arm reproduced exact control digest
  `382dd93cb724783226eae6ede000d6b62bbbc6439c8a39178cb9bb0ba8a27112`.
- Failure mode: the extra partial-reduction dispatch and MLX custom-kernel
  scheduling cost consume the tiled attention gain at this history length.
- Why not to retry unchanged: the complete 8/16/32 occupancy sweep stayed at
  or below the selected MLX SDPA implementation.
- Reopen only if: attention is fused with adjacent gate/output work, partials
  are reduced inside one launch, or profiling shows a longer-history crossover
  that improves full-model throughput.
- Related commit or revert: experimental C++/Metal source was removed.

## PERF-FA080 - Greedy native C ABI as a frozen Codex xhigh lane

- Hypothesis: native mixed-width execution plus exact prefix reuse would make
  the frozen 131K/xhigh Codex request usable before sampled decoding landed.
- Scope: native early-out27 v2, one 8,192-token prefill chunk, real 131,072
  context/token pools, exact prefix reuse, and the frozen Codex 0.151.0 tool
  command.
- Attempted change: served the native C ABI, whose current output contract is
  greedy token IDs, and ran the strict ephemeral xhigh tool round trip for 180
  seconds.
- Benchmark evidence: the 6,236-token prompt completed in about **74.6 s** at
  **83.62 prompt tok/s**. Decode began around **18.51 tok/s** and declined to
  roughly **17.95--18.0 tok/s** before the bounded client ended.
- Correctness evidence: the request aborted cleanly and the endpoint remained
  healthy, but no valid Codex JSON tool event or exact final response appeared.
- Failure mode: long-history decode remained below 20 tok/s and greedy output
  rambled instead of satisfying the tool protocol.
- Why not to retry unchanged: the C ABI ignores request sampling parameters,
  and the exact frozen gate already exposed both speed and behavior failures.
- Reopen only if: native sampling preserves temperature 1.0, top-p 0.95,
  top-k 20, and presence penalty 1.5, while measured long-history decode clears
  20 tok/s.
- Related commit or revert: prefix reuse and cache-storage wins remain; the
  greedy actual-work configuration is unqualified.

## PERF-FA081 - Materialized native affine gate/up row fusion

- Hypothesis: one double-height affine quantized matmul per MLP would remove a
  launch from every target layer and the MTP layer while preserving independent
  row arithmetic.
- Scope: native early-out27 v2 target and MTP weight loading plus the shared
  `Engine::mlp` owner; process-isolated deterministic `6237+128` serving with
  real 131,072 context/token pools.
- Attempted change: concatenated packed weights, scales, and biases at load,
  released the layer-owned input handles, issued one quantized matmul, and
  split its output into gate/up halves.
- Benchmark evidence: the signed `14fd46b11a` control reached **18.845 tok/s**,
  **58.100626 s** TTFT, and **64.839787 s** end to end. The adjacent candidate
  reached **18.782 tok/s**, **58.605376 s** TTFT, and **65.367119 s** end to
  end, a **0.334%** decode regression. Startup-reported available unified
  memory fell from **28.92 GB** to **22.28 GB**.
- Correctness evidence: both arms produced exact `6237+128` counts,
  `finish_reason=length`, and output/reasoning SHA-256
  `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5`.
  The focused native suite passed all **8 tests**.
- Failure mode: materializing concatenated affine storage leaves large buffers
  resident in MLX's allocator and the larger quantized matmul does not reduce
  the measured long-history decode wall.
- Why not to retry unchanged: the exact end-to-end A/B is slower and the
  additional residency directly weakens the real 131K capacity margin.
- Reopen only if: a native quantized kernel can consume the original gate/up
  tensors in one dispatch without a concatenated copy, with separately measured
  MLP-boundary and served gains.
- Related commit or revert: the experimental engine diff was removed; only the
  evidence record remains.

## PERF-FA082 - Linear-attention b/a affine row fusion

- Hypothesis: combining the two tiny 48-row b/a projections would remove one
  affine launch from each of 48 recurrent layers with negligible duplicated
  storage.
- Scope: native early-out27 v2 `Engine::gated_delta`; process-isolated exact
  deterministic `6237+128` serving with real 131,072 context/token pools.
- Attempted change: concatenated only `in_proj_b` and `in_proj_a` packed
  weights, scales, and biases at load, then split one 96-row quantized-matmul
  result inside the shared recurrent-layer owner.
- Benchmark evidence: the adjacent separate-projection control reached
  **18.845 tok/s**. The candidate reached **18.511 tok/s**, **58.055139 s**
  TTFT, and **64.915750 s** end to end, a **1.772%** decode regression. It
  retained **28.89 GB** startup-reported available unified memory.
- Correctness evidence: exact `6237+128`, `finish_reason=length`, and
  output/reasoning SHA-256
  `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5`
  matched the control. The focused native suite passed all **8 tests**.
- Failure mode: MLX's separate 48-row operations schedule more efficiently at
  batch one than one 96-row operation; dispatch-count reduction alone does not
  reduce the asynchronous graph wall.
- Why not to retry unchanged: the exact served regression is well beyond the
  immediately observed run-to-run difference, with memory held constant.
- Reopen only if: a native fused kernel consumes both original tensors while
  also eliminating a downstream b/a transform, and its full boundary timing
  beats the two asynchronous MLX operations.
- Related commit or revert: the experimental engine diff was removed.

## PERF-FA083 - Native 4-bit MTP-2 draft and recurrent target verification

- Hypothesis: the existing Qwen3.8 MTP head would emit enough accepted tokens
  per target verification to carry the selected native-MLX lane beyond the
  20 tok/s floor.
- Scope: signed fused-convolution target engine, pinned
  `mlx-community/Qwen3.8-27B-MTP-4bit` revision
  `b643c01b6d3b094e325edb6ebd832e16c486c575`, deterministic direct
  `128 / 32 warm / 256 timed`, and the existing two-draft greedy verifier.
- Attempted change: loaded the native sidecar through the established C ABI and
  instrumented the C++ benchmark with exact refill counts and emitted widths.
- Benchmark evidence: target-only measured **20.187702923 tok/s**. MTP measured
  **9.649959984 tok/s**, a **52.199%** regression, across 135 timed refills with
  mean width **1.888888889**.
- Correctness evidence: both paths produced digest `8ea2430e3fa3d56e`, last
  token `198`, and 256 timed outputs. The sidecar loaded successfully and every
  refill emitted a nonempty block.
- Failure mode: two sequential MTP forwards followed by a multi-token target
  forward traverse the general recurrent sequence path. The accepted-token
  yield does not amortize that block cost.
- Why not to retry unchanged: sidecar representation alone cannot change the
  target recurrent verification owner that dominates this topology, and the
  measured gap is larger than the remaining target-only optimization gap.
- Reopen only if: an isolated recurrent verification kernel or materially new
  target batch implementation first demonstrates a block cost low enough for
  the measured acceptance distribution to exceed 20 tok/s.
- Related commit or revert: the C++ benchmark retains optional MTP/refill
  telemetry; engine behavior is unchanged.

## PERF-FA084 - Full-attention affine q/k/v row concatenation

- Hypothesis: one affine q4 product could emit q+gate, k, and v rows while
  removing two product launches from each of 16 full-attention layers.
- Scope: native early-out27 v2 full-attention loading and `Engine::full_attn`;
  exact direct `128 / 32 warm / 256 timed` full-model screens.
- Attempted change: concatenated packed q4 weights, scales, and biases at load,
  evaluated and detached the combined storage, then split one product output.
  A second form retained the established q+gate product and combined only the
  equal-shaped k/v rows.
- Benchmark evidence: the all-row form reached **20.721377944 tok/s** and the
  k/v-only form reached **20.672271140 tok/s**. The immediately selected short
  control record was **20.630307745 tok/s**.
- Correctness evidence: a synthetic C++20 parity test matched six-row separate
  and concatenated products bit-for-bit and confirmed detached packed storage.
  The full model exposed production-shape divergence: all-row digest
  `12bb3edf3d51feac` and k/v-only digest `af06cc7ce5e094be` differed from exact
  control `8ea2430e3fa3d56e`; all three ended at token `198`.
- Failure mode: changing the output-row geometry selects a different MLX
  affine accumulation path for production k/v shapes, and recurrent decoding
  amplifies those float differences into a different token trajectory.
- Why not to retry unchanged: both useful concatenation boundaries failed the
  first full-model digest gate, so a longer throughput window cannot qualify
  them as semantics-preserving wins.
- Reopen only if: MLX exposes a fixed accumulation-geometry control or a native
  multi-output kernel reproduces each separate product's bit order while
  sharing input work.
- Related commit or revert: every experimental C++ and test change was removed;
  the native dylib is rebuilt from the selected source before the next screen.

## PERF-FA085 - Recurrent beta and decay inside q/k normalization

- Hypothesis: the idle lanes in the dual-output q/k normalization dispatch
  could calculate the small `sigmoid(b)` and `compute_g` arrays, removing two
  MLX launches from each of 48 recurrent layers.
- Scope: native early-out27 v2 single-token `Engine::gated_delta`; direct exact
  `6237 / 32 warm / 256 timed` process-isolated adjacent pairs.
- Attempted change: added beta and decay inputs/outputs to the existing q/k
  Metal owner. The final exact form reproduced compiled MLX log-add-exp with
  fast `exp`/`log`, preserved precise outer exponentials, and used an
  independent decay input type.
- Benchmark evidence: beta-only controls/candidates averaged
  **19.602405132 / 19.600900659 tok/s**, a **0.007675%** regression. The full
  beta/decay controls averaged **19.641655398 tok/s** and candidates averaged
  **19.547612476 tok/s**, a **0.478793%** regression; every adjacent pair
  favored the control.
- Correctness evidence: widened production, nonaligned, extreme-value, float32
  q/k, and twelve-outstanding-output cases passed exact parity. All ten final
  model runs retained digest `faaecee6edebe116`, last token `19360`.
- Failure mode: scalar exponentials execute serially within the q/k dispatch,
  giving up asynchronous overlap already available between the independent MLX
  graphs. Saved dispatches fail to offset that serialized work.
- Why not to retry unchanged: both the beta-only isolation and complete exact
  fusion have five-pair evidence at the actual long-history decode shape.
- Reopen only if: one downstream recurrent-update kernel consumes raw beta and
  decay parameters directly, or profiling demonstrates genuine idle ALU work
  with preserved overlap.
- Related commit or revert: the experimental C++ and test diff was removed;
  selected source and its exact short digest were restored.

## PERF-FA086 - Global and decode-only MLX SDPA block override

- Hypothesis: halving MLX's 128-block long-history SDPA reduction to 64 blocks
  would retain the established output while removing enough attention overhead
  to carry client-observed serving beyond 20 tok/s.
- Scope: native early-out27 v2 full-attention prefill/decode; exact direct
  `6237 / 32 warm / 256 timed` screens and deterministic served `6237+128`
  requests with real 131,072 context/token pools.
- Attempted change: first installed `MLX_SDPA_BLOCKS=64` before every native
  attention call. A narrowed form restored MLX's adaptive prefill policy,
  fully materialized its first token, and selected 64 blocks only for decode.
  Explicit process environment values retained precedence in both forms.
- Benchmark evidence: 32/64/96/128 direct screens reached
  **19.116925846 / 20.419125823 / 19.541540885 / 20.139173026 tok/s**; a
  second 64-block screen reached **20.391260024**. Global 64-block serving
  reached **20.146 client tok/s** and decode-only reached **20.127**.
- Correctness evidence: every direct screen retained digest
  `faaecee6edebe116`, last token `19360`. Both served forms changed the
  established deterministic response digest from
  `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5` to
  `69f3577805ed5ae85d2f8253eb3ec10f89ac7246897d6d5de9061ee6f665715c`.
- Failure mode: changing SDPA partial count changes floating-point reduction
  grouping. The direct token sequence had sufficient logit margin; the served
  prompt exposed a changed greedy trajectory even after its adaptive prefill
  was preserved.
- Why not to retry unchanged: both broad and decode-only placements clear the
  speed target while failing the fixed-work digest gate on the authoritative
  real server path.
- Reopen only if: attention work around the reduction can be removed while
  retaining MLX's 128-partial arithmetic, or a fixed-order native kernel first
  proves exact logits and the established served digest.
- Related commit or revert: both experimental C++ forms were removed; the
  selected A050 dylib hash and short digest were restored.
- Sampled-lane disposition, 2026-08-31: the deterministic default rejection
  remains closed. The user-authorized xhigh lane now applies the dependency's
  process-scoped `MLX_SDPA_BLOCKS=64` override only with native stochastic
  sampling. One real Codex xhigh shell round trip passed, and five sampled
  real-131K client requests averaged **20.1722 tok/s** with every sample above
  20. The source default and its deterministic arithmetic remain unchanged.

## PERF-FA087 - Renormalize top-p over only the selected top-k candidates

- Hypothesis: discarding the full-vocabulary log-sum-exp after top-k would
  remove enough sampling overhead to clear 20 tok/s with MLX's default SDPA
  reduction topology.
- Scope: native early-out27 v2 sampler, seed 67396869, direct 6,237-history
  decode and the real 131K Codex xhigh shell-tool turn.
- Attempted change: converted and normalized only the 20 selected candidate
  logits before cumulative top-p filtering, preserving the asynchronous
  two-token pipeline and device-resident Gumbel selection.
- Benchmark evidence: two direct `6237 / 32 warm / 256 timed` samples reached
  **20.112478513** and **20.120420191 tok/s** with the same
  `48d911de593ab4fc` digest. Seed 42 reached **20.211157783 tok/s** with a
  different `d2b13675c7615ce6` digest.
- Correctness evidence: the real Codex turn issued the requested first
  `/bin/pwd`, then sampled a malformed extra tool call whose `session_id`
  string failed the harness schema; the bounded client timed out.
- Failure mode: changing the normalization support materially changed the
  low-bit model's tool trajectory and failed the authoritative xhigh behavior
  gate.
- Why not to retry unchanged: the performance gain has full-model evidence,
  while the required named-client continuation fails.
- Reopen only if: a broader fixed-seed behavior suite establishes equivalent
  or better tool reliability and a second independent throughput window keeps
  every sample above 20.
- Related commit or revert: the candidate normalization was removed before
  commit; full-vocabulary normalization is restored.

## PERF-FA088 - Greedy native xhigh lane with a reasoning bound

- Hypothesis: greedy selection plus a bounded reasoning section would retain
  the selected deterministic speed and make the xhigh tool turn terminate.
- Scope: native early-out27 v2, real 131K pools, 256-token reasoning bound,
  and the same Codex shell-tool request.
- Attempted change: launched with native stochastic sampling disabled while
  keeping prompt-boundary state reuse and the reasoning bound active.
- Benchmark evidence: server decode telemetry remained around 20.1--20.2
  tok/s.
- Correctness evidence: Codex invoked `/bin/pwd` three times and timed out,
  violating the request's exactly-once contract.
- Failure mode: the greedy trajectory repeats the tool after each continuation.
- Why not to retry unchanged: the failure reproduced across prompt-state
  continuations and is behavioral rather than a throughput shortfall.
- Reopen only if: model precision or tool-parser state changes enough to alter
  the repeated greedy trajectory.
- Related commit or revert: no source change; the selected interactive lane
  uses native stochastic sampling with seed 42.

## PERF-FA089 - Sample reasoning and switch to greedy structured output

- Hypothesis: retaining temperature/top-p/top-k sampling inside `<think>` and
  switching to argmax after `</think>` would preserve xhigh reasoning diversity
  while stabilizing tool syntax from the low-bit checkpoint.
- Scope: native early-out27 v2, real 131K pools, fixed request-local seeds,
  128- and 256-token reasoning bounds, and the exact Codex `/bin/pwd` gate.
- Attempted change: added an opt-in post-reasoning argmax policy. A follow-up
  also discarded the already sampled two-token-pipeline lookahead when the
  reasoning-end token was emitted and recomputed that boundary token greedily.
- Benchmark evidence: decode telemetry remained around **20.1--20.4 tok/s**.
  The five-request throughput workload stays wholly inside its reasoning
  section and retained the **20.1556 tok/s** request-reseed window.
- Correctness evidence: seed 42 with a 256-token cap completed one requested
  tool and final marker after first attempting a disallowed escalation. A
  128-token cap completed one `/bin/pwd` and the final marker while Codex's
  router reported trailing function-argument characters. Recomputing the
  reasoning-close lookahead preserved that same extra malformed tool tail.
  Seed 67396869 produced duplicate fields and a string session handle and
  timed out.
- Failure mode: argmax after the reasoning boundary does not supply the tool
  schema or the completed-command state needed to choose a valid structured
  continuation. The low-bit model still emits malformed or unnecessary tool
  calls.
- Why not to retry unchanged: both the precomputed-lookahead and boundary-
  replacement forms reached the same parser failure class across two seeds
  and two reasoning bounds.
- Reopen only if: checkpoint precision, schema-constrained native decoding, or
  a measured tool-state representation changes the structured logits.
- Related commit or revert: all greedy-after-reasoning source changes were
  removed; request-boundary RNG ownership was retained separately.

## PERF-FA090 - End each assistant response after its first tool call

- Hypothesis: forcing the assistant-end token immediately after one complete
  Qwen3-Coder tool block would remove a malformed parallel-call tail while
  retaining sequential tools on later Codex continuations.
- Scope: native early-out27 v2, sampled reasoning plus greedy structured
  output, 128-token reasoning bound, exact xhigh `/bin/pwd` gate, real 131K
  pools, and one running request.
- Attempted change: added an opt-in native transition from the tool-call-end
  token directly to the assistant-end token, discarding the pipeline's next
  proposal.
- Benchmark evidence: server decode telemetry remained around **20.0--20.4
  tok/s** throughout the bounded run.
- Correctness evidence: the first response emitted exactly one valid
  `/bin/pwd`, and Codex observed `/Users/dcazares/sglang`. Each following turn
  then tried `write_stdin` with fabricated alphanumeric handle `"85dfe4"`;
  Codex reported schema errors and the 120-second wrapper exited **124**.
- Failure mode: truncating the parallel tail removes the error result that had
  prompted the model to recover. The next assistant response reconstructs the
  same invalid `write_stdin` call, so turn serialization moves the defect
  across requests.
- Why not to retry unchanged: the policy worsened the authoritative behavior
  gate from exit zero with one router error to repeated router errors and a
  timeout.
- Reopen only if: a native grammar can validate tool arguments against the
  supplied schema or model quality removes the fabricated handle.
- Related commit or revert: the one-call transition was removed before
  commit; the native tool stream retains its original multi-call behavior.

## PERF-FA091 - Restore only recurrent output projections from Q4

- Hypothesis: higher-precision recurrent output anchors would repair the
  selective-Q2 checkpoint's structured-tool trajectory while preserving the
  established sampled throughput floor.
- Scope: all 48 `linear_attn.out_proj` quantized tensor triplets, loaded from
  immutable `mlx-community/Qwen3.8-27B-4bit` revision
  `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` into early-out27 v2; native
  sampled real-131K serving and the pinned Codex xhigh shell gate.
- Attempted change: added an opt-in checkpoint overlay and ran one complete
  five-sample `6237+128` production window before the exact client gate.
- Benchmark evidence: decode measured **20.111, 20.134, 20.115, 20.127, and
  20.117 tok/s**, mean **20.1208**, with every request above 20. Prompt
  throughput averaged **107.0072 tok/s** and every request completed exact
  6,365 tokens with `finish_reason=length`.
- Correctness evidence: thread `01a05bb4-576b-7a21-8cf0-2a84c94b6451`
  prefetched 6,214 tokens and decoded continuously until GNU timeout exit
  **124**, without a Codex tool or final event.
- Failure mode: recurrent output precision alone preserves speed and changes
  the model trajectory, yet it leaves the authoritative xhigh tool turn
  unusable.
- Why not to retry unchanged: the candidate has a full five-sample throughput
  window and an exact named-client failure under the selected request-local
  seed.
- Reopen only if: another precision family, native grammar mechanism, or
  materially different checkpoint changes the structured-output distribution.
- Related commit or revert: the generic precision-overlay diagnostic remains
  opt-in while `qkv`, `z`, and complete recurrent families are narrowed; the
  output-only candidate is closed.

## PERF-FA092 - Broaden the DFlash small-batch QMM to 5,120 outputs

- Hypothesis: routing the target and draft `17408 -> 5120` down projections
  through the custom batch-eight affine product would extend its isolated
  microbenchmark gain across the full verification cycle.
- Scope: full-Q4 target, affine-W4 DFlash2 draft, seven proposed tokens,
  selected recurrent tape commit, and the existing output-tiled Metal QMM.
- Attempted change: first admitted every affine projection with at least
  5,120 outputs, then narrowed the experiment to the exact
  `17408 -> 5120` down-projection shape.
- Benchmark evidence: the selected `N >= 6144` route measured steady draft
  **34.211--37.378 ms** and verify **305.130--305.604 ms**. The exact-down arm
  regressed to **34.906--38.626 ms** draft and **314.693--315.150 ms** verify.
  The broader 5,120-output route also regressed the full-Q4 cycle.
- Correctness evidence: the standalone `17408 -> 5120` comparison remained
  within BF16 accumulation tolerance; this rejection is based on reachable
  full-model cost.
- Failure mode: the broader predicate also reaches draft projection families
  whose shapes favor MLX's stock product, while the isolated down-projection
  saving does not repay the changed whole-cycle schedule.
- Why not to retry unchanged: both the broad and exact-shape dispatches lose
  against an adjacent selected full-model control.
- Reopen only if: a role-aware target-only dispatch or a different K-split
  kernel wins the complete verification cycle.
- Related commit or revert: both experimental predicates were removed; the
  retained opt-in dispatch requires at least 6,144 output features.

## PERF-FA093 - Fixed four-token DFlash2 block

- Hypothesis: drafting three tokens per verification would halve the current
  target pass and raise throughput when the seven-token block has modest
  acceptance.
- Scope: full-Q4 target, affine-W4 DFlash2 draft, block size four, three draft
  tokens, selected QMM and accepted-prefix tape commit.
- Attempted change: changed the fixed block from eight to four and screened
  the same direct random-token workload.
- Benchmark evidence: steady draft measured **18.735--19.078 ms**, verify
  **170.274--170.941 ms**, and total **191.727--192.747 ms**. Mean emitted
  width was **1.142857**. Perfect width four would provide only about
  **20.85 tok/s** before server and client overhead.
- Correctness evidence: the candidate completed the direct decode screen; the
  production constants were restored to block eight and seven draft tokens.
- Failure mode: the reduced target work also caps useful emission, leaving no
  operating margin at the 20 tok/s gate under ordinary acceptance.
- Why not to retry unchanged: measured acceptance is far below the width
  required to exploit the already narrow perfect-acceptance ceiling.
- Reopen only if: an adaptive policy predicts high-confidence short blocks
  from current logits and a real-client A/B window clears 20 with margin.
- Related commit or revert: the fixed block-four source change was removed.

## PERF-FA094 - Four-way M8 K split

- Hypothesis: halving the M8 affine kernel from eight to four K partitions
  would reduce threadgroup storage, final reduction work, and scheduling cost.
- Scope: the opt-in full-Q4 target plus affine-W4 DFlash2 M=8 verification
  path, with identical 32x16 BF16 weight tiles and FP32 accumulation.
- Attempted change: assigned one quarter of K to each of four SIMD groups and
  reduced the threadgroup from 256 to 128 threads.
- Benchmark evidence: gate/up `K=5120,N=17408` regressed from the SG8
  **0.978217 ms** sample to **1.027962 ms**. Down `K=17408,N=5120` improved
  from **1.026783** to **1.016329 ms**. The reachable whole-model verifier
  regressed from about **225.5--226.7 ms** to **233.3--234.5 ms**. One sampled
  direct run reached **11.592581 tok/s** through a changed mean emitted width
  of **3.097561**.
- Correctness evidence: checkpoint microbenchmarks stayed within the existing
  BF16 parity bound and the direct decode completed.
- Failure mode: the dominant gate/up family loses more execution time than
  the down projection saves. The one sampled throughput increase came from a
  changed stochastic acceptance trajectory while fixed cycle cost regressed.
  The sixteen-way candidate improves execution cost and throughput by a much
  larger margin.
- Why not to retry unchanged: it is dominated by the retained SG16 geometry
  on both verifier time and sampled direct throughput.
- Reopen only if: a future device has materially different occupancy limits
  and matched fixed-cycle measurements favor four groups.
- Related commit or revert: the SG4 constants were replaced during the same
  experiment; no repository commit contains the candidate.

## PERF-FA095 - Eight-way K split with 32-column output tiles

- Hypothesis: doubling the M8 affine output tile would reduce threadgroup
  count and amortize dequantization/launch cost enough to beat SG16/B16.
- Scope: full-Q4 target plus affine-W4 DFlash2 M=8 verification, using eight K
  partitions and one private 32x32 BF16 weight tile per SIMD group.
- Attempted change: used 256-thread groups, 32 output columns, four FP32 8x8
  accumulators per SIMD group, and the existing FP32 cross-group reduction.
- Benchmark evidence: gate/up measured **0.993133 ms** and down
  **0.957713 ms**, versus SG16/B16 **0.975192** and **0.972296 ms**. The full
  verifier improved to generally **203.5--205.8 ms**, yet one direct sampled
  run reached only **11.680308 tok/s**, 46 refills, and mean emitted width
  **2.913043**.
- Correctness evidence: real checkpoint tensor comparisons remained within
  the existing BF16 parity bound and full direct decode completed with digest
  `d3c38ed2009d8f81` and last token 735.
- Failure mode: returning from sixteen to eight K partitions changes BF16
  reduction grouping and the fixed-seed sampled trajectory. Its small fixed
  execution win does not offset the observed acceptance loss. SG16/B32 then
  lowers the verifier further to **185.4--186.7 ms** and raises the repeated
  sampled mean to **31.327334 tok/s**.
- Why not to retry unchanged: the retained SG16/B32 geometry dominates this
  candidate in both fixed verifier cost and sampled throughput.
- Reopen only if: another device cannot admit the SG16/B32 512-thread,
  32-KiB threadgroup and a device-specific matched server window favors SG8.
- Related commit or revert: the SG8/B32 constants were replaced during the
  same experiment; no repository commit contains the candidate.

## PERF-FA096 - Dense BF16 DFlash2 as the production draft

- Hypothesis: loading the official BF16 DFlash2 matrices directly would
  recover proposal accuracy lost during affine-W4 conversion and could reduce
  draft cost through optimized dense MLX products.
- Scope: the exact immutable 81-tensor
  `incoai/Qwen3.8-27B-DFlash2` checkpoint with the unchanged full-Q4 target,
  SG16/B32 target verifier, stochastic selector, and exact p/q rejection.
- Attempted change: extended the native linear owner and DFlash loader to
  accept BF16 `[output,input]` matrices directly, then compared the BF16
  checkpoint against affine-W4 through one candidate dylib and identical
  `128 / 32 warm / 128 timed` sampling settings.
- Benchmark evidence: the adjacent affine-W4 control reached **30.992508
  tok/s**, 19 refills, mean emitted width **6.684211**, and digest
  `46bd4bb035b72c2b`. Dense BF16 repeated at **9.005226** traced and
  **9.044043 tok/s** untraced, 61 refills, mean width **2.114754**, digest
  `408f99f917ffffcc`, and last token 96968. Its steady draft stage also rose
  from about **26--32 ms** affine to **41.5--42.1 ms** dense.
- Correctness evidence: the exact 81-tensor artifact loaded and completed both
  runs. Dense-QLinear bit parity, strict warning-as-error library/test builds,
  the standalone affine/dense suite, and the focused native suite passed.
- Failure mode: this BF16 proposal distribution takes a much lower-acceptance
  fixed-seed path and reads three times as much draft weight. The untraced
  result is **21.948464 tok/s / 70.818614%** below the adjacent affine control.
- Why not to retry unchanged: affine-W4 dominates both accepted width and
  draft execution cost on this device and workload.
- Reopen only if: proposal policy changes, another workload demonstrates a
  repeatable BF16 acceptance advantage, or a dense kernel removes the measured
  bandwidth cost while a matched production window clears the selected path.
- Related commit or revert: signed `6cf95442cc` retains dense loading as
  official-checkpoint compatibility; production continues to select the
  affine-W4 artifact.

## PERF-FA097 - Greedy DFlash2 proposals for sampled xhigh serving

- Hypothesis: matching the official SGLang DFlash worker's greedy target-head
  proposal rule would eliminate selector work and raise acceptance by choosing
  each draft position's most likely token.
- Scope: affine-W4 DFlash2 with the selected SG16/B32 verifier, exact
  temperature-1/top-p-0.95/top-k-20 target distribution, exact deterministic-q
  rejection, and unchanged real 131K pools.
- Attempted change: added an experimental C++ environment switch that replaced
  the learned top-16 selector distribution with LM-head argmax proposals. The
  residual sampler used a one-token proposal support with probability one.
- Benchmark evidence: five synthetic direct samples reached **37.537356,
  37.505550, 37.500870, 37.532174, and 37.517706 tok/s**, mean
  **37.518731**, with mean width **7.9375**. The exact real `6237+128` sampled
  request then fell to **9.512 tok/s**, versus learned-selector **15.328
  tok/s**, and took **70.236581 s** end to end.
- Correctness evidence: both direct and served runs completed exact requested
  token counts. The real request ended with `finish_reason=length` and output
  SHA-256 `62bc27d075d3d68fd4eb9fbbf8d4db312390c505bd36ddfe578086720c2b656e`.
- Failure mode: the synthetic repeated-token prompt makes later block tokens
  nearly deterministic and overstates greedy acceptance. Natural sampled
  reasoning frequently accepts zero or one greedy token, reducing real
  throughput by **5.816 tok/s / 37.943633%** from the selected learned
  selector.
- Why not to retry unchanged: the production-shaped request directly rejects
  the candidate, and its apparent **19.763562%** direct gain is a benchmark
  artifact.
- Reopen only if: the production contract changes to greedy target sampling,
  or a representative prompt corpus shows a matched learned-selector loss.
- Related commit or revert: the experimental switch was removed with
  `apply_patch`; no repository commit contains the candidate.

## PERF-FA098 - Independently top-k/top-p-filtered DSpark proposals

- Hypothesis: filtering each DSpark proposal row through the target's top-k 20
  and top-p 0.95 rule would remove proposal mass that the verifier's target
  distribution can never accept and raise overlap.
- Scope: native affine-W4 DSpark, the selected full-Q4 target and SG16/B32
  verifier, seed 42, exact dense-q rejection, and the direct
  `128 / 1 warm / 32 timed` sampled screen.
- Attempted change: added an opt-in C++ switch that replaced each full-vocab
  DSpark softmax with the existing target-side `sampling_probabilities`
  mechanism, then sampled and verified against that exact filtered q.
- Benchmark evidence: the unchanged full-softmax baseline reached **10.050625
  tok/s**, 14 refills, and mean emitted width **2.428571**. The aligned-filter
  candidate fell to **6.997461 tok/s**, 20 refills, and mean width **1.6**.
  Steady draft cost also rose from generally **36.55--40.07 ms** to
  **38.71--42.40 ms**.
- Correctness evidence: exact p/q rejection completed all 32 timed tokens;
  the candidate produced digest `b149ae20f95e9c7b` and last token 8420. The
  strict warning-as-error candidate library built successfully.
- Failure mode: the draft and target rank different top-20 supports.
  Independently truncating q removes lower-ranked draft tokens that overlap
  target support, reduces accepted width, and adds seven vocabulary
  partition/sort operations per refill.
- Why not to retry unchanged: both acceptance and fixed draft execution cost
  regress decisively on the first exact screen.
- Reopen only if: a shared target-informed support is available before draft
  sampling or measured proposal/target support overlap changes materially.
- Related commit or revert: the experimental switch was removed with
  `apply_patch`; no repository commit contains the candidate.

## PERF-FA099 - BF16 DSpark Markov output projection

- Hypothesis: retaining `markov_head.markov_w2` in BF16, matching the upstream
  CUDA lane's precision preference, would improve proposal/target overlap
  enough to offset its larger matrix and dense product.
- Scope: native DSpark with the selected full-Q4 target, five-layer affine-W4
  draft backbone, exact dense-q rejection, seed 42, SG16/B32 verifier, direct
  `128 / 1 warm / 32 timed`, and real-131K-pool sampled `6237+128` serving.
- Attempted change: taught the standalone C++ converter to retain only Markov
  W2 in source BF16 and the native linear loader to accept the resulting exact
  134-tensor hybrid contract. The distinct derived checkpoint carried full
  source provenance and SHA-256
  `73b829d7845a72ac34794e9dd74bd96eae2189a5bcd7b45c2099a2b45638674f`.
- Benchmark evidence: direct throughput fell from **10.050624654 to
  7.044992356 tok/s** (**-29.904930%**), refills rose from 14 to 20, and mean
  emitted width fell from **2.428571429 to 1.65**. The representative request
  reached **11.294 tok/s** versus the all-affine **11.242 tok/s**, a
  **+0.052 / +0.462551%** movement on a different sampled trajectory. The
  hybrid artifact was **87.148 MiB** larger.
- Correctness evidence: converter reload/provenance verification, strict
  warning-as-error native build, direct exact p/q execution, focused native
  pytest (**8 passed**), real exact 6,365-token completion, language-only
  `/model_info`, and post-request health all passed. Served output/reasoning
  SHA-256 was
  `47690f3aaf04561fa6abe2cd3205724c59204b43a4f3eb4a8e1c525d584da3b1`.
- Failure mode: this isolated precision change moves proposal sampling onto a
  sharply lower-acceptance direct trajectory, adds residency, and produces
  only a sub-percent served movement that is inseparable from trajectory
  variation. It remains **8.706 tok/s** below the required floor.
- Why not to retry unchanged: direct acceptance evidence is adverse and the
  representative result provides no material, repeatable margin.
- Reopen only if: fixed-context teacher-forced overlap analysis demonstrates
  a consistent BF16 Markov-W2 advantage across representative prompts, or a
  fused dense projection removes its residency/execution cost and a matched
  repeated served window clears the selected path.
- Related commit or revert: converter and loader changes were removed with
  `apply_patch`; the 1,227,639,900-byte derived artifact was deleted and is
  reproducible from the immutable source using the experiment record.

## PERF-FA100 - Fixed shortened DSpark verification

- Hypothesis: verifying fewer than seven DSpark proposals would reduce target
  work enough to offset the smaller maximum emitted width.
- Scope: affine-W4 DSpark, the selected full-Q4 target, exact dense-q
  rejection, seed 42, accepted-prefix tape commit, and direct
  `128 / 1 warm / 32 timed` sampling.
- Attempted change: generalized the shared verifier to checked one-through-seven
  draft prefixes and swept each fixed prefix while retaining the identical
  seven-position proposal graph.
- Benchmark evidence: draft counts one through seven reached
  **11.920230932 / 8.861865636 / 8.353200099 / 7.544333347 / 4.104265970 /
  4.139508352 / 10.025000236 tok/s**. The corresponding mean widths were
  **1.6 / 1.523809524 / 1.777777778 / 1.941176471 / 1.571428571 /
  1.571428571 / 2.428571429**. M=6/7 verification cost roughly **327--332
  ms**, while the selected M=8 kernel takes about **186--188 ms**.
- Correctness evidence: every prefix completed exact p/q execution. Default
  seven-token DSpark reproduced digest `5a38c7070d7badeb` and last token 16;
  DFlash reproduced its selected digest `46bd4bb035b72c2b` and last token 20.
- Failure mode: shortened blocks cap useful emission, and target matrices with
  six or seven rows fall into a particularly slow generic affine-QMM tier.
  The best fixed short block reaches 11.920231 tok/s, 8.079769 below the floor.
- Why not to retry unchanged: all fixed shortened widths are below both the
  required 20 tok/s and the retained target-only lane.
- Reopen only if: a faster M=2 verifier plus materially improved proposal
  survival clears the real-client floor, or an adaptive policy assigns M=2
  only where its measured expected throughput exceeds M=8.
- Related commit or revert: the checked bounded verifier is retained as
  opt-in profiling and adaptive-scheduling infrastructure; default remains
  seven drafts.

## PERF-FA101 - Underpriced DSpark full-width cost ratio

- Hypothesis: a **1.4** full-to-short cycle-cost ratio, selected by the best
  short direct screen, would assign M=8 often enough to maximize natural-prompt
  DSpark throughput.
- Scope: affine-W4 DSpark, selected full-Q4 target, trained current-block
  confidence, exact dense-q rejection, seed 42, real 131,072 context/token
  pools, and the exact sampled `6237+128` representative request.
- Attempted change: enabled the native M=2/M=8 confidence budget with
  `SGLANG_MLX_NATIVE_DSPARK_CONFIDENCE_COST_RATIO=1.4`; all server arguments,
  checkpoint paths, sampling controls, and request fields matched the adjacent
  fixed-M=8 control.
- Benchmark evidence: the fixed-M=8 control reached **11.313 tok/s**,
  **109.114 prompt tok/s**, **57.160628 s TTFT**, and **68.386711 s** end to
  end. Ratio 1.4 reached **10.916 tok/s**, **109.712 prompt tok/s**,
  **56.848776 s TTFT**, and **68.483064 s** end to end. Live warmed cycles
  measured about **146 ms** for M=2 and **259 ms** for M=8, a ratio near
  **1.77**.
- Correctness evidence: the candidate completed exact 6,365 tokens with
  `finish_reason=length`, coherent reasoning, and recorded SHA-256 prefix
  `095e73b6`. Server health and language-only metadata passed, followed by
  clean verified shutdown.
- Failure mode: the short direct trajectory underestimates the natural-history
  full-width cost. Ratio 1.4 selects M=8 for blocks whose expected survival
  does not repay the measured 1.77x complete-cycle cost.
- Why not to retry unchanged: the exact representative admission screen
  regresses the adjacent control by **0.397 tok/s / 3.509%**.
- Reopen only if: target kernels or history shape move the measured M=8/M=2
  complete-cycle ratio near 1.4, with a fresh adjacent real-prompt control.
- Related commit or revert: the generic budget mechanism is retained; the
  selected opt-in ratio is **1.75**, which averages **13.6058 tok/s** across
  five consecutive representative samples.

## PERF-FA102 - Accepted-width-triggered DSpark cooldown

- Hypothesis: scheduling target-only cooldown from the last block's actual
  emitted width would identify low-value M=8 verifications that the trained
  confidence budget misclassified.
- Scope: native affine-W4 DSpark, ratio 1.75, 16 bypass refills, exact sampled
  `128 / 32 warm / 256 timed`, selected full-Q4 target, and seed 42.
- Attempted change: temporarily triggered cooldown when the preceding block
  emitted fewer than three, five, or six tokens, independent of whether the
  confidence scheduler had selected M=2 or M=8.
- Benchmark evidence: minimum widths three/five/six reached respectively
  **16.160295443 / 19.319338815 / 18.686185376 tok/s**. The retained
  confidence-tier trigger reaches **24.332648695 tok/s** at the same cooldown
  and workload; its no-cooldown control reaches **24.494388127 tok/s**.
- Correctness evidence: every candidate completed exact sampled decoding.
  Width five and six shared digest `104b9dd15cebf891`; width three produced
  `8f2080982a3e9adb`. Strict warning-as-error builds passed throughout.
- Failure mode: actual sampled acceptance changes the following target token
  and random stream. Treating a low realized width as a stable predictor
  repeatedly enters cooldown on the resulting low-yield trajectory.
- Why not to retry unchanged: every threshold trails the confidence-tier
  trigger by at least **5.01331088 tok/s** on the admission screen.
- Reopen only if: a teacher-forced corpus demonstrates accepted-width
  autocorrelation under a fixed target-token trajectory, with a policy that
  does not feed its own sampling changes back into the predictor.
- Related commit or revert: the accepted-width trigger and its threshold
  control were removed; PERF-A074 retains only the trained-confidence M=2
  trigger.

## PERF-FA103 - Alternate fixed DSpark cooldown lengths

- Hypothesis: a shorter or longer target-only interval after each M=2 choice
  would find high-confidence regions sooner or amortize low-value probes more
  effectively than sixteen refills.
- Scope: native affine-W4 DSpark, ratio 1.75, exact sampled
  `128 / 32 warm / 256 timed`, selected full-Q4 target, and seed 42.
- Attempted change: screened fixed cooldowns **4, 8, 16, 32, 64, and 128** through the
  checked runtime control while preserving every other direct setting.
- Benchmark evidence: the six settings reached **19.198827552 /
  17.781301933 / 24.332648695 / 19.215717015 / 18.401520476 /
  19.663812056 tok/s**. Their respective refill counts were
  **151 / 175 / 104 / 173 / 249 / 235**, and every setting followed a
  different exact sampled trajectory. The no-cooldown ratio-1.75 control was
  **24.494388127 tok/s**. The longer settings approach target-only behavior
  with mean widths **1.028112450 / 1.089361702**.
- Correctness evidence: all settings completed exact 256-token sampling with
  finite nonempty output; strict warning-as-error candidate builds passed.
- Failure mode: fixed cooldown changes target sampling and therefore the
  future confidence/acceptance trajectory. Four and eight probe too often;
  thirty-two misses useful high-confidence regions on this screen.
- Why not to retry unchanged: 4/8/32/64/128 trail the selected 16-refill
  screen by **5.116931143 / 6.551346762 / 5.116931680 / 5.931128219 /
  4.668836639 tok/s** respectively.
- Reopen only if: a representative teacher-forced trace supplies a stable
  counterfactual trajectory or a cheap current-token predictor replaces
  periodic probing.
- Related commit or revert: the generic checked cooldown remains; PERF-A074
  selects **16** only as an opt-in measured setting.

## PERF-FA104 - DSpark target-state anchor bypass

- Hypothesis: the trained confidence projection evaluated on the current
  normalized target hidden state and current-token Markov embedding could
  identify low-value draft cycles before paying for the five-layer DSpark
  block.
- Scope: native affine-W4 DSpark, ratio 1.75, exact sampled direct
  `128 / 32 warm / 256 timed`, the representative real-131K-pool
  `6237+128` request, selected full-Q4 target, and seed 42.
- Attempted change: added a checked temporary threshold that executed exact
  target-only refill whenever the trace-only target-state score exceeded the
  configured value. Thresholds 0.475, 0.5, 0.525, 0.55, and 0.6 were screened.
- Benchmark evidence: direct rates were **16.393088197 / 30.299622883 /
  33.221518376 / 30.796768648 / 16.111013104 tok/s**. Threshold 0.525 repeated
  at **32.788326664** and **32.834047040 tok/s**. The real threshold-0.525
  request reached only **13.652 tok/s**, below cooldown-16's **16.6396 tok/s**
  mean. Its adjacent no-threshold trace reached **13.580 tok/s**.
- Correctness evidence: each direct screen and both real requests completed
  exact sampled decoding. The threshold request completed 6,365 tokens with
  `finish_reason=length` and recorded SHA-256 `c92e4510...`; health,
  language-only metadata, and verified cleanup passed.
- Failure mode: the synthetic trace's score relation does not transfer to the
  natural prompt. On 48 no-policy real cycles, M=2/M=8 score ranges overlap,
  score versus accepted width has Pearson **0.140841**, and threshold 0.525
  classifies 13 cycles from each budget tier as bypasses.
- Why not to retry unchanged: a one-sample **0.072 tok/s** movement over the
  adjacent baseline is far below the selected cooldown and the required floor,
  while the predictor has no useful real-prompt separation.
- Reopen only if: a calibrated current-token feature demonstrates stable
  held-out natural-prompt separation and improves cooldown-16 in a complete
  five-sample real window.
- Related commit or revert: the threshold parser, state, and scheduling branch
  were removed. PERF-A075 retains only target-state score telemetry under the
  existing speculative trace flag.

## PERF-FA105 - DFlash2 on the faster mixed-precision target

- Hypothesis: attaching DFlash2 to the QKV-restored target that already clears
  20 tok/s would preserve enough proposal overlap to add speculative margin.
- Scope: affine-W4 DFlash2, native exact p/q verification, selected early-out27
  v2 target, immutable Q4 recurrent-projection donor, QKV-only and complete
  recurrent override scopes, seed 42, and direct sampled
  `128 / 32 warm / 128 timed`.
- Attempted change: changed the target checkpoint and existing precision
  override only; the DFlash artifact, learned selector, verifier, sampler, and
  kernels stayed fixed.
- Benchmark evidence: QKV-only reached **10.130497494 tok/s**, 49 refills, and
  width **2.653061224**. Complete recurrent restoration reached
  **7.841259160 tok/s**, 64 refills, and width **1.984375**. The compatible
  full-Q4 target reproduces **31.317933897 tok/s**, 19 refills, and width
  **6.684210526**.
- Correctness evidence: both mixed-target screens completed exact rejection
  sampling with finite 128-token output and recorded deterministic digests.
- Failure mode: the draft is trained against the full-Q4 target distribution;
  the faster mixed target changes logits enough to collapse accepted width.
- Why not to retry unchanged: both precision scopes lose more than 21 tok/s
  directly before server overhead.
- Reopen only if: a DFlash2 checkpoint is trained or distilled against the
  selected mixed target, with measured natural-prompt proposal overlap.
- Related commit or revert: no source change was retained; both existing
  checkpoints remain immutable.

## PERF-FA106 - Two-phase 64-column M8 affine verifier tile

- Hypothesis: one threadgroup covering 64 output columns could reuse four M=8
  input fragments across two 32-column weight-staging phases and halve the
  output grid within the selected 32 KiB storage budget.
- Scope: SG16 affine-W4 M=8 verifier, full-Q4 target, affine-W4 DFlash2,
  selected direct `128 / 32 warm / 128 timed`, and exact p/q sampling.
- Attempted change: temporarily doubled the output tile to 64, retained a
  32-column staging tile, held eight FP32 SIMD-matrix accumulators, and reused
  four BF16 input fragments across both output halves.
- Benchmark evidence: direct throughput fell from the restored 32-column
  **31.317933897 tok/s** to **8.960931772 tok/s**, a
  **22.357002125 tok/s / 71.387219%** regression. Both arms used 19 refills and
  mean width **6.684210526**.
- Correctness evidence: warning-as-error library/test builds passed. Synthetic
  M8 parity passed K/N `512/256`, `5120/64`, and `512/6144`; the full-model
  candidate reproduced digest `46bd4bb035b72c2b` and last token 20.
- Failure mode: the eight accumulator fragments plus retained input fragments
  create severe register/occupancy pressure, while halving the threadgroup grid
  removes parallelism.
- Why not to retry unchanged: exact whole-model throughput regresses by more
  than 70% with unchanged acceptance.
- Reopen only if: a device or kernel representation can hold the wider tile
  without register pressure and an isolated real-tensor microbenchmark first
  beats SG16/B32.
- Related commit or revert: the 64-column source was removed; the signed
  SG16/B32 kernel remains exact and selected.

## PERF-FA107 - DFlash2 selector temperatures outside the retained 1.15 arm

- Hypothesis: rescaling the learned DFlash selector's unary-plus-transition
  logits can increase target overlap enough to improve sampled serving.
- Scope: full-Q4 target, affine-W4 DFlash2, seven-token learned proposal,
  exact p/q rejection, selected SG16/B32 verifier, and the representative
  real-131K-pool `6237+128` request.
- Attempted change: screened selector temperatures
  **0.7/0.85/0.95/1.05/1.15/1.3** while leaving request temperature, target
  probabilities, and exact verifier unchanged.
- Benchmark evidence: direct `128 / 32 warm / 128 timed` rates for
  0.7/0.85/0.95/1.05/1.15/1.3 were respectively
  **14.838952659 / 11.185534971 / 37.171244579 / 10.998405302 /
  26.936073462 / 9.563179041 tok/s**, versus identity
  **31.291292321**. The apparent 0.95 direct winner regressed the real request
  to **13.739 tok/s**. The retained 1.15 arm averaged **15.8866 tok/s** across
  five real requests versus the adjacent identity mean **15.4424**.
- Correctness evidence: the retained implementation forwards each selected
  token's exact rescaled q into the common rejection sampler. All real
  requests completed exact 6,365 tokens with `finish_reason=length` and a
  stable coherent digest within each setting. Invalid zero fails closed.
- Failure mode: repeated-token direct trajectories do not rank proposal
  calibration reliably for natural reasoning. Every screened arm apart from
  1.15 either lost directly or, for 0.95, failed the representative served
  gate.
- Why not to retry unchanged: the full sweep already isolates the selector
  scale, and 0.95's synthetic lead reverses on the admission workload.
- Reopen only if: a new target/draft pairing, selector checkpoint, request
  distribution, or adaptive calibration signal changes proposal overlap.
- Related commit or revert: PERF-A076 retains only the checked opt-in scale;
  identity remains default and 1.15 is the measured selected arm.

## PERF-FA108 - M=8 gate/up paired dispatch and sequential fused SwiGLU

- Hypothesis: consuming the original gate/up tensors in one native M=8
  dispatch can remove 64 projection submissions and intermediate elementwise
  work from each DFlash verification.
- Scope: full-Q4 target MLPs, eight-row SG16/B32 affine-W4 products, exact
  BF16 SiLU/multiply boundaries, affine-W4 DFlash2, selector temperature 1.15,
  and direct `128 / 32 warm / 128 timed` decoding.
- Attempted change: first ran gate and up sequentially inside one workgroup and
  emitted SwiGLU directly. Then retained independent workgroups on two z-grid
  planes in one Metal submission, emitted separate gate/up arrays, and left
  MLX SiLU/multiply unchanged.
- Benchmark evidence: the sequential fused form changed traced throughput
  **26.801071247 -> 26.413620964 tok/s** and steady verify about
  **186.2--187.3 -> 189.4--190.5 ms**. Five adjacent paired-grid controls and
  candidates averaged **26.943319961 / 26.964966176 tok/s**, only
  **+0.08034%**; two pairs were flat/slower.
- Correctness evidence: standalone K/N `512/256` products were bit-exact for
  both paired outputs, and the sequential SwiGLU result was bit-exact against
  separate selected products plus BF16 MLX SiLU/multiply. Every full-model arm
  reproduced 22 refills, width **6.090909091**, digest `6de63586df62ab2b`,
  and last token 220. Strict builds and tests passed.
- Failure mode: sequential fusion halves grid concurrency and adds about 3.3
  ms. Paired submission preserves concurrency, while Metal submission savings
  are only noise-scale beside two roughly one-millisecond products per layer.
- Why not to retry unchanged: both the execution-merging and launch-only
  limits were measured, with one regressing and one lacking a material margin.
- Reopen only if: a kernel can share weight or input work across gate/up while
  retaining grid-level concurrency, or it fuses the following down projection
  without changing BF16 boundaries.
- Related commit or revert: every candidate source/test change was removed;
  the selected separate SG16/B32 products remain unchanged.

## PERF-FA109 - DFlash2 mean-q thresholds outside the retained 0.62 arm

- Hypothesis: a lower or higher selected-q boundary may improve the balance
  between 133--135 ms M=2 cycles and 247--250 ms M=8 cycles on the natural
  reasoning request.
- Scope: full-Q4 target, affine-W4 DFlash2, selector temperature 1.15, exact
  sparse-q rejection sampling, real 131,072 context/token pools, and sampled
  `6237+128` serving.
- Attempted change: screened mean-q6 thresholds 0.55, 0.62, and 0.65 through
  the same checked native scheduler and otherwise identical foreground server
  and client commands.
- Benchmark evidence: the thresholds reached **15.191 / 16.151 / 16.056
  tok/s** respectively. The retained 0.62 arm then averaged **16.1776 tok/s**
  over five requests, while an adjacent threshold-disabled control averaged
  **15.8974 tok/s**.
- Correctness evidence: every arm completed exact 6,365 tokens with
  `finish_reason=length` and coherent reasoning. Each five-request setting
  reproduced one stable output digest. Exact selected sparse q continues into
  the prefix verifier and residual sampler.
- Failure mode: 0.55 leaves too many low-yield full-width cycles; 0.65
  shortens additional borderline cycles whose useful accepted prefixes repay
  M=8 cost. Both produce weaker exact sampled trajectories than 0.62 on the
  representative admission workload.
- Why not to retry unchanged: the bracketing screens isolate the useful local
  boundary, and the retained arm has a full matched five-sample window.
- Reopen only if: target M=2/M=8 cost, selector temperature, draft checkpoint,
  target checkpoint, or representative request distribution changes.
- Related commit or revert: PERF-A078 retains only the checked opt-in
  scheduler; threshold 0.62 is selected and the default remains full M=8.

## PERF-FA110 - One-shot full-Q4 target-only long prefill

- Hypothesis: the existing one-shot target-only prefill can admit the exact
  6,237-token representative request within M1 Max unified-memory residency.
- Scope: native full-Q4 target-only engine, one 6,237-token native prefill,
  BF16 KV, real 131,072 context/token pools, and one running request.
- Attempted change: first launched the target-only server with its unchanged
  one-shot native prefill. A stale `.venv-mps` build-prefix selection caused
  the initial launcher build to miss MLX headers; rebuilding with the active
  `.venv` prefix isolated the runtime result.
- Benchmark evidence: the rebuilt server reached the exact request and then
  exhausted Metal residency during the 6,237-token native prefill, before any
  generation sample. The same target with opt-in 2,048-token internal chunks
  completes directly and through the real 131K serving surface.
- Correctness evidence: the retained chunked arm completes exact 6,365 tokens
  with `finish_reason=length` and coherent reasoning. Single-chunk sampled
  controls preserve the exact 256-token digest and final token.
- Failure mode: the one-shot graph retains the whole-prompt working set across
  64 target layers and exceeds available Metal residency on this model and
  request shape.
- Why not to retry unchanged: the failure is deterministic under the recorded
  full-Q4, 6,237-token, 131K-pool contract, and internal chunking directly
  removes the residency lifetime.
- Reopen only if: model residency, MLX command-buffer lifetime, unified-memory
  capacity, or target prefill storage ownership changes materially.
- Related commit or revert: PERF-A079 retains checked opt-in target-only
  internal chunking at 2,048; the absent-value default preserves one-shot
  behavior.

## PERF-FA111 - One-SIMD-per-output affine-W4 batch-one QMV

- Hypothesis: a dedicated batch-one Metal QMV can beat MLX's generic affine
  quantized product across the full-Q4 target projections and close the final
  target-only decode gap.
- Scope: affine-W4/G64 BF16-input target products, full-Q4 target-only direct
  decoding, and exact short `128 / 32 warm / 128 timed` admission.
- Attempted change: assigned one SIMD group to each output row and eight output
  rows to each 256-thread group. The first form loaded scale and bias per
  packed word. A second form loaded each parameter pair once per quantization
  group and broadcast it across the corresponding eight lanes.
- Benchmark evidence: stock MLX reached **20.339874670 tok/s**. The independent
  parameter-load form reached **10.456143330 tok/s** (**-48.593%**), and the
  subgroup-broadcast form reached **7.773375044 tok/s** (**-61.783%**).
- Correctness evidence: strict library/test builds passed. Batch-one parity at
  K/N `128/128`, `5120/64`, and `512/6144` stayed within **0.03125** maximum
  absolute BF16 difference. Both full-model candidates completed 128 tokens
  with one shared digest and final token.
- Failure mode: scalar-output SIMD geometry leaves MLX's tuned matrix/vector
  memory and instruction schedule far ahead. `simd_shuffle` plus divergent
  parameter ownership further increases the full-model cost.
- Closure basis: two parameter-loading strategies lose by roughly twofold or
  more at the reachable whole-model path, leaving no admission margin for
  narrower shape tuning of this geometry.
- Reopen only if: a matrix-tiled batch-one kernel, dependency-level QMV
  primitive, or fused downstream consumer first beats stock MLX on real target
  tensors.
- Related commit or revert: all experimental C++/Metal/header/test changes
  were removed; PERF-A080 records the result.

## PERF-FA112 - Unchanged Bartowski Q5_K_M on native Metal GGUF

- Hypothesis: the balanced Q5_K_M source artifact can load and serve directly
  through the existing heterogeneous GGUF execution path.
- Scope: pinned Bartowski revision
  `f0eec4a4bb4975114a030d048952d83c0a53c034`, exact
  `Qwen3.8-27B-Q5_K_M.gguf`, M1 Max, native MPS GGUF quantization, float32
  model dtype, BF16 KV, one request, and a 1,024-token pool.
- Attempted change: downloaded and checksum-verified the immutable source,
  passed its actual-file Q4_0/Q5_K/Q6_K native arithmetic gate, then launched
  the ordinary SGLang GGUF loader with conservative caches.
- Benchmark evidence: server startup did not reach warmup or a generation
  sample. The loader raised `NotImplementedError` while processing mixed
  merged weights because that native Metal path does not support Q8_0 shards.
- Correctness evidence: actual-file batch-one parity passed Q4_0, Q5_K, and
  Q6_K before startup. The source is exactly 20,752,787,040 bytes and verifies
  as SHA-256 `e731e180...caa8`.
- Failure mode: Q5_K_M's heterogeneous merged tensors include Q8_0 members at
  a loader boundary whose supported native set excludes Q8_0.
- Why not to retry unchanged: the exception is deterministic during weight
  transformation and precedes all cache sizing and serving work.
- Reopen only if: the exact mixed-merge owner gains Q8_0 support or a
  provenance-preserving derived artifact converts the affected merged shards.
- Related commit or revert: PERF-A082 selects the narrower Q5_K_S source and a
  distinct token-embedding derivative. The immutable Q5_K_M source remains
  available as a control.

## PERF-FA113 - Unchanged Bartowski Q5_K_S token embedding

- Hypothesis: the smaller Q5_K_S source artifact can complete native Metal
  startup without artifact transformation.
- Scope: pinned Bartowski revision
  `f0eec4a4bb4975114a030d048952d83c0a53c034`, exact
  `Qwen3.8-27B-Q5_K_S.gguf`, M1 Max, native MPS GGUF quantization, float32
  model dtype, BF16 KV, one request, and a 1,024-token pool.
- Attempted change: downloaded and checksum-verified the immutable source,
  passed actual-file Q4_0/Q5_K/Q6_K native arithmetic, and launched the same
  conservative SGLang configuration used for Q5_K_M.
- Benchmark evidence: weight loading completed in **43.06 s** at **20.00 GB**
  residency with **11.99 GB** available. Mamba and 1,024-token BF16 KV caches
  allocated. Automatic warmup then raised `NotImplementedError` because the
  native Metal GGUF embedding path does not support Q5_K.
- Correctness evidence: the source is exactly 19,680,945,760 bytes, verifies
  as SHA-256 `b52fbc24...e569`, and passes representative packed-tensor parity.
- Failure mode: `token_embd.weight` is Q5_K; the current embedding dispatch
  supports a narrower set than the native quantized matrix owner.
- Why not to retry unchanged: every first-token forward reaches the same
  unsupported embedding dispatch after successful weight and cache setup.
- Reopen only if: native Q5_K embedding support reaches the shared execution
  owner. The derived F16-embedding artifact already supplies the active path.
- Related commit or revert: PERF-A082 converts only the source token embedding
  to F16 and retains all other source tensor encodings.

## PERF-FA114 - Q5_K four-lane row mapping on 1,024-row projections

- Hypothesis: doubling the Q5_K batch-one output cohort to 32 rows will also
  improve the compact full-attention K/V projections.
- Scope: native Metal Q5_K batch-one matvec, representative
  `blk.3.attn_k.weight` shape `(1024,5120)`, aligned compact storage, eight
  warmups, and 25 synchronized timed iterations.
- Attempted change: selected the four-lane-per-row, eight-weight-per-lane,
  32-row-per-threadgroup mapping across every aligned Q5_K batch-one shape.
- Benchmark evidence: the established eight-lane mapping measured
  **0.327208 ms / 10.329 GiB/s**; the wider-row candidate measured
  **0.337792 ms / 10.006 GiB/s**, a **3.23%** latency regression. Wider
  5,120--17,408-row target projections supplied positive served evidence under
  the thresholded route.
- Correctness evidence: actual-file prefixes, odd row tails, long-K compact
  views, alignment fallback, and synthetic packed extrema all passed.
- Failure mode: the compact 1,024-row grid supplies too little output work to
  amortize the reduced lanes per row and wider threadgroup cohort.
- Why not to retry unchanged: the production checkpoint has a stable 1,024-row
  K/V family and the established kernel is already faster on its exact shape.
- Reopen only if: GPU family, Metal compiler, row geometry, lane mapping, or a
  fused downstream consumer changes the compact-shape economics.
- Related commit or revert: PERF-A084 retains the four-lane mapping only for
  output sizes at least 5,120, the smallest measured winning shape, and
  preserves the prior 1,024-row owner.

## PERF-FA115 - Same-GGUF three-step NEXTN as the complete Q5 serving lane

- Hypothesis: the Qwen3.8 checkpoint's bundled NEXTN block, three speculative
  steps, four verify tokens, and a faster Q6_K batch-four target verifier would
  raise the derived Q5 lane above target-only throughput.
- Scope: the existing synchronous EAGLE/NEXTN worker v2, the same derived Q5
  GGUF as target and draft source, top-k-one linear proposals, and a 1K
  conservative MPS server pool.
- Attempted change: first launched the same GGUF draft without an explicit
  draft quantizer, then corrected the retained ordering boundary with
  `--speculative-draft-model-quantization gguf`; measured the retained
  batch-four kernel against `SGLANG_MPS_Q6_K_BATCH4_ROWS16=0`.
- Benchmark evidence: explicit GGUF draft loading reduced draft residency from
  all **10.62 GB** remaining memory to **1.00 GB** and served successfully.
  Five candidate `128+128` generation samples were
  `3.643,3.799,3.711,3.699,3.662 tok/s`, mean **3.7028**. The selected
  target-only median is **7.500 tok/s**. The batch-four kernel itself remains
  a win: its matched disabled mean was **3.3384 tok/s**.
- Correctness evidence: all ten matched served requests completed exact
  256-token length responses with reasoning preserved. Candidate/control mean
  accepted lengths were **2.960/3.024**. Direct candidate, tail, batch-three
  fallback, and existing batch-eight arithmetic checks pass.
- Failure mode: repeated draft forwards plus multi-token target verification
  consume more time than the accepted-token yield saves on this M1 Max
  topology. The first missing-quantizer attempt also exhausted memory before
  KV allocation because configuration propagation preceded target GGUF
  detection.
- Why not to retry unchanged: corrected same-GGUF NEXTN is **50.629%** below
  target-only serving and remains **16.2972 tok/s / 5.4013x** from the
  requested floor.
- Reopen only if: a materially smaller/faster trained draft, a lower-cost
  target verification topology, or measured acceptance/cycle evidence changes
  the complete-path economics.
- Related commit or revert: PERF-A085 retains the independent Q6_K exact-batch-
  four kernel win; the same-GGUF NEXTN launch remains unselected.

## PERF-FA116 - Exact 131K BF16 KV beside the derived Q5 artifact

- Hypothesis: the 32 GB unified-memory machine can retain the 21.37 GB derived
  model, exact 131,072-token BF16 attention cache, and runtime working set at
  interactive throughput.
- Scope: M1 Max 32 GB, derived Q5_K_S/F16-embedding artifact, float32 compute,
  exact 131,072 context and token pool, one request, one FP32 Mamba slot,
  page size one, and ordinary sampled `128+32` serving.
- Attempted change: launched the existing target-only server with exact
  `context_length=max_total_tokens=131072` and BF16 KV.
- Benchmark evidence: startup and warmup passed with 21.37 GB model residency,
  a 0.29 GB Mamba slot, and 4.00 GB each for K and V. One exact request reached
  **5.271 prompt tok/s**, **24.284058 s TTFT**, **0.102 generation tok/s**,
  and **329.689187 s E2E**. Memory pressure fell to 28--50% free and swap
  traffic rose materially.
- Correctness evidence: health, model-list length 131,072, language-only model
  metadata, exact 160-token length completion, and 32 preserved reasoning
  fragments all passed.
- Failure mode: model, cache, and transient working-set residency exceed the
  practical unified-memory budget and generation becomes page-fault bound.
- Why not to retry unchanged: the exact configuration is already capacity-
  functional and **98.576%** slower than the selected 1K-pool Q5 mean.
- Reopen only if: model or KV residency shrinks materially, the memory budget
  changes, or a measured residency control eliminates the paging boundary.
- Related commit or revert: PERF-A086 records this capacity result. PERF-A088
  changes only the token embedding to Q4_K, recovers 1.37 GB of runtime
  residency, and improves the matched exact-pool result to **0.315 tok/s**;
  the unchanged BF16 cache still pages heavily.

## PERF-FA117 - Stock float8 KV tensors on MPS

- Hypothesis: existing `--kv-cache-dtype fp8_e4m3` support can halve the exact
  Q5 attention-cache residency without source changes.
- Scope: PyTorch 2.11.0 MPS, `torch.float8_e4m3fn`, allocation, FP32
  conversion, indexed write, gather, and conversion back to FP32.
- Attempted change: ran an isolated capability probe before another server
  launch.
- Benchmark evidence: the first FP32-to-FP8 MPS conversion immediately raised
  `TypeError: Trying to convert Float8_e4m3fn to the MPS backend but it does
  not have support for that dtype.` No timing sample was admitted.
- Correctness evidence: byte-backed float8 views and indexed gathers were
  subsequently proven functional. The unchanged framework conversion itself
  still fails, and the isolated process exited cleanly.
- Failure mode: the generic KV pool maps `fp8_e4m3` to a torch float8 dtype
  whose value conversion is absent from the active stock MPS backend.
- Why not to retry unchanged: every generic pool allocation reaches the same
  framework dtype boundary.
- Reopen only if: PyTorch MPS adds native float8 conversion or a later runtime
  changes this exact boundary.
- Related commit or revert: PERF-A087 records the unchanged-framework screen.
  PERF-A089 retains an SGLang native Metal conversion over the already
  byte-backed pool and therefore changes the premise without changing the
  stock result.

## PERF-FA118 - Exact 131K E4M3FN cache as the complete Q5 speed solution

- Hypothesis: halving the exact attention cache from BF16 to E4M3FN restores
  the small-pool Q5 decode rate while preserving requested capacity.
- Scope: Q4_K-embedding Q5 artifact, one FP32 Mamba slot, exact 131,072-token
  FP8 K/V pool, native Metal conversion, and ordinary sampled `128+32` serving.
- Attempted change: launched signed PERF-A089 with
  `context_length=max_total_tokens=131072` and `kv_cache_dtype=fp8_e4m3`.
- Benchmark evidence: K/V residency fell **8.00 -> 4.00 GB** and reported
  headroom rose **0.99 -> 6.99 GB**. Five cache-flushed samples measured
  **3.098 / 3.299 / 3.264 / 3.273 / 3.251 tok/s**, mean **3.237**. This is
  **10.276x** the matched BF16 result and **55.419%** below the 1K FP8 mean.
- Correctness evidence: exact pool allocation, automatic warmup, health,
  maximum model length, language-only metadata, exact sampled token counts,
  reasoning, arithmetic `703`, and one parsed multiply call all pass.
- Failure mode: the 4.00 GB fully allocated pool still creates a material
  residency penalty on the 32 GB unified-memory machine. Long populated
  histories also retain generic FP8-to-FP32 gather materialization.
- Why not to retry unchanged: five warmed sequential samples establish a
  stable 3.237 tok/s region, **16.763 tok/s / 6.179x** from the floor.
- Reopen only if: another resident-byte reduction changes the full-pool
  boundary or fused compressed attention removes the long-history conversion
  path.
- Related commit or revert: PERF-A090 retains exact FP8 capacity and records
  this incomplete-speed boundary.

## PERF-FA119 - Unchanged Q4-tuned DFlash controls on affine Q5

- Hypothesis: the selected DFlash2 temperature and mean-q budget can lift the
  new native affine-Q5 target above 20 tok/s without another kernel change.
- Scope: immutable affine-Q5/G64 target, affine-W4 DFlash2 draft, native
  sampling seed 42, selector temperature 1.15, mean-q6 threshold 0.62, and
  direct `128 / 32 warm / 128 timed` decode.
- Attempted change: reused the complete selected full-Q4 DFlash environment
  while changing only the target checkpoint to affine Q5.
- Benchmark evidence: target-only reached **16.322505765 tok/s**. The unchanged
  DFlash composition reached **11.506669050 tok/s**, 56 refills, and mean
  emitted width **2.232142857**. M=2 verification was about 114 ms and M=8
  verification about 363--368 ms.
- Correctness evidence: the target and draft loaded, exact p/q rejection ran,
  128 timed tokens completed, and the process exited cleanly.
- Failure mode: the retained M=8 K-split Metal verifier accepts affine bits
  two/four only. Five-bit target matrices fall through to generic MLX QMM;
  the Q4-calibrated proposal controls also yield too few tokens on this direct
  trajectory to repay that cost.
- Why not to retry unchanged: it is **4.815836715 tok/s / 29.505%** slower
  than the adjacent target-only result and **8.493330950 tok/s** below the
  required floor.
- Reopen only if: a native affine-five-bit verifier materially reduces fixed
  M=8/M=2 cost, followed by selector/budget recalibration on the natural
  request.
- Related commit or revert: PERF-A091 retains the checkpoint and baseline;
  no source change was made by this failed composition.

## PERF-FA120 - Five-bit MTP as a sampled production accelerator

- Hypothesis: the official affine-five-bit MTP head can preserve native
  top-k/top-p sampling and lift the affine-Q5 target above 20 tok/s.
- Scope: pinned target revision `2568951b...c2f05`, pinned MTP revision
  `1faa5a80...3d85`, exact dense-q rejection sampling, blocks two through
  eight, and direct seed-42 decode.
- Attempted change: made the standard MTP block size opt-in through two to
  eight tokens, sampled every recurrent proposal from its recorded dense q,
  and reused the common exact p/q verifier.
- Benchmark evidence: matched deterministic block three reaches
  **17.919995235 tok/s** at width **3**; M=8 reaches
  **31.380317186 tok/s** at width **8**. Exact sampled calibration at
  temperatures 0.25/1.15/1.5/2.0
  reaches **3.990054799 / 4.821419896 / 6.380162135 / 5.062826830 tok/s**;
  the best width is **2.285714286**. Top-k four at temperature 1.5 reaches
  **3.780318328 tok/s** and width **1.26**.
- Correctness evidence: dense q flows into the established rejection sampler,
  target p/q acceptance and residual sampling complete, exact output counts
  return, and both deterministic probes preserve target digest
  `e446d211f2e2ff25` and last token 15.
- Failure mode: the trained head aligns strongly at argmax while its sampled
  distribution overlaps the target too weakly to amortize the roughly
  228--230 ms M=8 verification cycle.
- Why not to retry unchanged: the best sampled arm trails target-only
  **16.322505765 tok/s** by **9.942343630 tok/s** and the floor by
  **13.619837865 tok/s**.
- Reopen only if: a matched draft distribution materially increases exact
  sampled overlap or target verification cost falls enough for a mean width
  near 2.3 to win.
- Related commit or revert: PERF-A093 retains exact sampled semantics and the
  block-size probe; calibration-only temperature/top-k controls were removed.

## PERF-FA121 - Existing 64-column small-batch kernel for affine Q5 M=3

- Hypothesis: exact five-bit unpacking in the retained small-batch QMM can
  accelerate the default three-token MTP verification batch.
- Scope: native affine-Q5 target, matched MTP head, greedy block three, and
  the existing RowTile-8/OutputTile-64 Metal geometry.
- Attempted change: added exact eight-value/five-byte unpacking and admitted
  Q5 M=2 through M=8 to the small-batch kernel.
- Benchmark evidence: the adjacent greedy block-three result changes from
  **17.472384559 to 10.679521956 tok/s**, a **38.877%** regression.
- Correctness evidence: M=3 and M=4 K/N `512/256` parity both pass with
  maximum absolute error **0.03125**.
- Failure mode: the 64-column threadgroup staging geometry loses occupancy and
  scheduling efficiency at the three-row target batch.
- Why not to retry unchanged: full-model evidence is decisive even though the
  arithmetic is correct.
- Reopen only if: a measured M=3-specific geometry changes weight staging or
  split-K economics.
- Related commit or revert: the Q5 small-batch branch and its temporary tests
  were removed before PERF-A093.

## PERF-FA122 - Alternate affine-Q5 batch-one QMV geometries

- Hypothesis: increasing output-row reuse, pack depth, SIMD-group count, or a
  narrower K cohort can improve the direct affine-Q5 decode kernel.
- Scope: native affine-Q5/G64 target-only sampled `128 / 32 warm / 128 timed`
  screens with seed 42 and exact BF16/FP32 arithmetic.
- Attempted change: swept SIMD-groups/results/packs geometries `2/4/2`,
  `2/8/2`, `2/4/1`, `2/4/4`, `4/4/2`, and `8/4/2`, then a 16-K-lane/two-row
  cohort.
- Benchmark evidence: the corresponding full-model results were
  **17.466385085 / 16.016573901 / 16.956398080 / 16.837489631 /
  17.505461879 / 17.422819885 tok/s**. The 16-lane cohort passed parity and
  reached only **14.839742878 tok/s**. The selected four-SIMD/four-row/two-pack
  form later reached **17.823163930 tok/s** after fixed FMAs, packed loads,
  K specialization, and command-buffer controls.
- Correctness evidence: representative parity remained within the selected
  **0.25** BF16 acceptance bound. The one-pack form admitted K=256 and thus
  intentionally failed the selected-geometry fail-closed test.
- Failure mode: additional result rows and weight packs raise register
  pressure; narrower K cohorts reduce input reuse; fewer output rows underuse
  each activation fragment.
- Why not to retry unchanged: every alternate geometry trails the selected
  result by at least **0.350211221 tok/s**, and the 16-lane form trails it by
  **2.983421052 tok/s**.
- Reopen only if: shader attribution or occupancy counters identify a distinct
  bottleneck and a geometry changes both register pressure and weight-load
  coalescing.
- Related commit or revert: alternate source strings were removed before
  PERF-A094; only the selected four/four/two source remains.

## PERF-FA123 - FP16 local accumulation in affine-Q5 QMV

- Hypothesis: half-precision local dot products can reduce register bandwidth
  while a final FP32 accumulation preserves sufficient output accuracy.
- Scope: selected four-SIMD/four-row/two-pack Q5 kernel and the pinned
  affine-Q5 target.
- Attempted change: accumulated each unpacked weight/input product in FP16,
  converted each group dot to FP32, and retained FP32 cross-group/reduction
  arithmetic. The first explicit half-`fma` form failed Metal overload
  resolution; a half multiply/add form compiled.
- Benchmark evidence: the compiled form reached **16.743742934 tok/s**,
  **1.079420996 tok/s** below the selected FP32-local result.
- Correctness evidence: the representative synthetic maximum errors remained
  within the test bound, while the full-model sampled digest changed.
- Failure mode: conversions and half arithmetic cost exceed any register
  saving, and reduced precision changes the target sampling trajectory.
- Why not to retry unchanged: it is both slower and semantically less stable
  than fixed explicit FP32 FMAs.
- Reopen only if: a native packed-dot instruction can replace the scalar
  conversion sequence while preserving the selected BF16 output trajectory.
- Related commit or revert: the FP16-local branch was removed before
  PERF-A094.

## PERF-FA124 - Concurrent MLX streams for affine-Q5 gate/up projections

- Hypothesis: submitting independent gate and up affine-Q5 products on two
  persistent Metal streams can overlap their weight traversal.
- Scope: batch-one affine-Q5 `Engine::mlp`, two `mx::StreamContext` scopes,
  and the selected target-only full-model benchmark.
- Attempted change: created persistent gate/up GPU streams, built one
  projection on each stream, and consumed both through the ordinary SwiGLU
  and down projection.
- Benchmark evidence: the first full-model candidate retained about **14 GB**,
  consumed **0.0% CPU**, and made no progress for more than one minute. Stack
  inspection placed later custom-Metal probes in
  `IOSurfaceSharedEvent::waitUntilSignaledValue`; a stock MLX arithmetic probe
  completed in **0.388 seconds**.
- Correctness evidence: no candidate output completed. The exact benchmark PID
  was terminated with status 143. Temporary stream code was removed and the
  restored sequential source passes strict warning-as-error compilation.
- Failure mode: graph construction across independent MLX streams lacks an
  explicit event/dependency lifetime joining gate/up production to their
  default-stream consumer, leaving a custom-Metal event unsignaled.
- Why not to retry unchanged: the first execution stalls and contaminates
  subsequent custom-kernel validation until the Metal session recovers.
- Reopen only if: a minimal isolated C++ proof establishes explicit
  cross-stream event ownership, completion, and asynchronous array lifetime
  before any full-model launch.
- Related commit or revert: all parallel-stream source was removed before
  PERF-A094; sequential `Engine::mlp` remains authoritative.

## PERF-FA125 - Q5 aligned-load, vector-input, dot, and parameter-broadcast variants

- Hypothesis: fewer dynamic Q5 weight loads, wider activation transactions,
  grouped FP32 dot instructions, or shared scale/bias loads can raise the
  selected affine-Q5 batch-one kernel by the remaining five percent.
- Scope: selected A094 Q5/G64 Metal QMV at exact gate/up `5120x17408`, down
  `17408x5120`, attention-output `6144x5120`, and value `5120x1024` shapes;
  process-isolated order/reverse `100 / 1000` timing.
- Attempted change: evaluated A103 three-word overlapping weight loads, A104
  four `float4` dots, A106 four aligned 64-bit activation reads, A108
  four-lane parameter broadcast, and shape-specific `2/4/2` and `8/4/2`
  threadgroup mappings.
- Benchmark evidence: A103 was consistently slower than A094 across all four
  shapes. A104 and A106 exchanged sub-percent order effects and converged in
  reverse timing. A108 regressed representative latency by roughly **4--22%**.
  Shape-specific geometries converged with control and supplied no durable
  full-model funding signal. Raw means are retained under the corresponding
  PERFORMANCE_LOG entry.
- Correctness evidence: A103, A104, A106, and A108 pass representative parity
  at maximum errors **0.03125 / 0.03125 / 0.0234375**. Their four complete
  production-shape digests match A094 exactly.
- Failure mode: the Metal compiler/hardware already coalesces the ordinary
  input and parameter traffic effectively. Explicit funnels, vector
  conversions, leader branches, shuffles, and changed reduction grouping
  add instruction or scheduling cost without reducing streamed weight bytes.
- Why not to retry unchanged: exact production-shape timing resolves each
  mechanism, and none projects close to the remaining five-percent full-model
  gap.
- Reopen only if: a compiler/GPU change, shader-counter attribution, or a new
  mapping reduces total streamed Q5 bytes or proves a distinct occupancy
  bottleneck.
- Related commit or revert: candidates remain isolated in persistent detached
  worktrees; selected A094 source on `main` is unchanged.

## PERF-FA126 - Custom affine-Q4 QMV in the mixed Q5-class target

- Hypothesis: enabling the existing custom Q4 batch-one path for the mixed
  artifact's 162 Q4 linears will compound the selected Q5 QMV gain.
- Scope: pinned 4.951-bpw mixed target, seed 42, exact direct
  `128 / 32 warm / 128 timed`, with Q4 and Q5 QMV switches independently
  controlled.
- Attempted change: measured generic QMM, Q5-only, Q4-only, and combined QMV
  arms with all other command-buffer and sampling controls fixed.
- Benchmark evidence: generic reached **18.121698566 tok/s**, Q5-only
  **19.032187257**, Q4-only **17.221199280**, and both **17.987099210**.
- Correctness evidence: every arm completed the exact token contract. The Q5
  selected and adjacent clean-A094 arms share digest `d0193f6d413b68c1`.
- Failure mode: the custom Q4 geometry remains slower than MLX's generic
  batch-one owner on this model, erasing part or all of the Q5 gain.
- Why not to retry unchanged: both isolated and combined full-model arms
  directly measure the reachable mixed-target path.
- Reopen only if: a new Q4 kernel beats generic MLX at the exact mixed-model
  shapes under matched microbenchmarks before another full-model launch.
- Related commit or revert: the Q4 switch remains opt-in and disabled for the
  selected mixed-Q5 configuration.

## PERF-FA127 - Published Q4 MTP head on the mixed Q5-class target

- Hypothesis: the smaller published Q4 MTP head plus corrected post-norm seed
  can amortize target verification and lift mixed-Q5 sampled throughput.
- Scope: pinned mixed target, namespaced Q4/G64 MTP head, exact p/q sampling,
  block three, original/post-norm target seed, and optional Q4 QMV.
- Attempted change: ran three isolated direct smokes varying only seed and Q4
  draft execution around the retained compatibility loader.
- Benchmark evidence: original seed reached **9.122242179 tok/s**, width
  **1.882352941**; post-norm reached **9.507655940**, width **1.764705882**;
  Q4 QMV with original seed reached **11.341033334**, width **2.285714286**.
- Correctness evidence: the head loads through its `mtp.` namespace, exact p/q
  verification runs, and each arm completes its requested token count.
- Failure mode: accepted width remains too low to repay draft and multi-token
  target verification; the Q4 QMV improves draft cost while staying far below
  target-only execution.
- Why not to retry unchanged: the best arm trails the selected target-only
  screen by **7.691153923 tok/s** and the required floor by
  **8.658966666 tok/s**.
- Reopen only if: a measured proposal-distribution or target-verification
  breakthrough materially changes accepted tokens per refill or cycle cost.
- Related commit or revert: loader and post-norm controls remain retained for
  correctness research; the composition is unselected.

## PERF-FA128 - Dense BF16 recurrent b/a row fusion

- Hypothesis: combining the mixed artifact's two `[48,5120]` dense recurrent
  b/a projections removes 48 Metal matmul dispatches per generated token.
- Scope: pinned mixed target, selected A094 Q5 QMV, seed 42, exact
  `128 / 32 warm / 128 timed`, and two independent balanced five-versus-five
  process-isolated windows.
- Attempted change: concatenated each layer's dense b/a weights at load,
  issued one `[96,5120]` product, split its result at 48, and preserved the
  original quantized path as fallback.
- Benchmark evidence: window-one means were control **18.987806897** and
  fusion **19.008612939 tok/s**. Independent window-two means were control
  **18.998960565** and fusion **18.977830522**. Aggregate ten-sample means are
  control **18.993383731** and fusion **18.993221731**, a
  **-0.000162000 tok/s / -0.000853%** movement.
- Correctness evidence: strict C++20/O3 warnings-as-errors compilation and
  `git diff --check` pass. All 20 timed outputs reproduce digest
  `d0193f6d413b68c1` and last token `11406`.
- Failure mode: the removed tiny dense dispatches do not own measurable
  end-to-end time; the larger product and result split offset their encoding
  savings.
- Why not to retry unchanged: the independent balanced window cancels the
  first window's apparent 0.11% gain and the aggregate result is flat to four
  significant decimal places.
- Reopen only if: a fused native kernel also consumes b/a in the recurrent
  update or profiling attributes a larger dense-dispatch cost under a changed
  runtime.
- Related commit or revert: candidate remains outside `main` in persistent
  detached worktree `perf-ab-fusion`; no source revert is required.

## PERF-FA129 - Paired Q5 word loads and 128-bit activation reads

- Hypothesis: halving Q5 weight-load instructions with lane sharing or halving
  activation-load transactions can improve the selected A100 kernel further.
- Scope: exact gate/up, down, attention-output, and value-projection shapes;
  selected `100 / 1000` process-isolated order/reverse microbenchmark.
- Attempted change: A101 has even lanes load five 32-bit words for each
  adjacent lane pair and supplies odd lanes with three SIMD shuffles. A107
  reads each lane's 32 activation bytes through two `uint4` transactions and
  converts their four 64-bit halves as `bfloat4`.
- Benchmark evidence: A101 regresses gate/up, down, and attention-output by
  roughly **5--9%** across paired means and is also slower at value width.
  A107 regresses gate/up and down in both directions, has a mixed attention-
  output result, and is slower at value width. Exact raw means are retained in
  `PERFORMANCE_LOG.md`.
- Correctness evidence: both candidates compile under strict warnings, pass
  representative parity at **0.03125 / 0.03125 / 0.0234375**, and reproduce
  every control digest.
- Failure mode: A101 replaces coalesced lane-local loads with a masked branch
  and three shuffle dependencies. A107 changes transaction width while
  retaining all conversions and arithmetic, leaving the streamed Q5 weights
  and dominant dependency chain unchanged.
- Why not to retry unchanged: every high-byte production shape is flat or
  slower, so neither mechanism can fund the remaining 4.521% full-model gap.
- Reopen only if: a future compiler removes the masked/shuffle overhead, the
  activation base contract or cache hierarchy changes, or the wider reads fuse
  with a separate input consumer.
- Related commit or revert: selected A100 remains in signed commit
  `59a50653c4`; A101/A107 artifacts remain outside `main`.

## PERF-FA130 - Hand-inlined affine-Q4 helper arithmetic

- Hypothesis: an aggressively inlined Q4/G64 dot body can combine the new
  four-SIMD output cohort with compiler scheduling that beats stock MLX.
- Scope: mixed 4.951-bpw target, its 162 Q4 linears, selected A100 Q5 kernel,
  seed 42, and exact `128 / 32 warm / 128 timed` direct generation.
- Attempted change: loaded each four-nibble word into a local `ushort`, kept
  input and dot loops directly inside the dynamic Metal kernel body, and
  accumulated four output rows per each of four SIMD groups.
- Benchmark evidence: paired gate/up timing improved from stock
  **0.464814646 ms** to **0.434910417 ms**. One full-model candidate reached
  **19.413641543 tok/s** against **19.129950306** with the Q4 switch unset.
- Correctness evidence: the candidate completed all shapes, while parity
  reported **0 / 0.00195312 / 0.0078125** maximum BF16 error at K/N
  `512/64`, `5120/128`, and `5120/17408`. The full-model digest changed from
  `d0193f6d413b68c1` with last token 11406 to `70b8328e7074a21e` with last
  token 12.
- Failure mode: Apple Metal reassociates the hand-inlined FP32 expression
  enough to cross BF16 rounding boundaries. The resulting sampled trajectory
  violates the fixed-work digest gate.
- Why not to retry unchanged: the apparent speedup depends on arithmetic
  lowering that changes observable model output.
- Reopen only if: generated-code evidence identifies a load or scheduling
  change that retains MLX's helper/expression structure and bit-exact BF16
  output. PERF-A111 demonstrates that exact boundary.
- Related commit or revert: the hand-inlined source was replaced in the
  detached candidate worktree before promotion; no revert is required.

## PERF-FA131 - Remaining affine-Q4 output, load, lane-work, and unroll forms

- Hypothesis: more output rows, fewer threadgroups, wider weight transactions,
  more packs per lane, or compile-time unrolling can compound A111's Q4 gain.
- Scope: mixed 4.951-bpw target, its 162 Q4 linears, selected A100+A111,
  bit-exact synthetic parity, gate/up `100 / 1000` microbenchmarks, and exact
  `128 / 32 warm / 128 timed` full-model screens.
- Attempted change: evaluated exact `8x4`, `4x8`, and `2x8` output cohorts;
  one `packed_ushort4` load; four packs per lane; a helper-local `ushort`; and
  forced full unrolling of the K-specialized outer loop.
- Benchmark evidence: `8x4` is flat at control/candidate **19.229939742 /
  19.230535491 tok/s**. `4x8` and `2x8` fall to **18.840786855** and
  **18.851060160**. The vector load moves five-sample means only
  **19.222120223 -> 19.232188501**. Four packs per lane reaches
  **18.928908455** full-model tok/s. The local word regresses paired gate/up
  **0.454020313 -> 0.464569500 ms**, and full unrolling regresses one gate/up
  run **0.468931459 -> 1.799243958 ms**.
- Correctness evidence: every geometry, vector-load, local-word, and unroll
  form is bit-exact at the focused parity shapes. Four packs per lane has one
  `0.000488281` BF16 mismatch at `5120x17408` and changes the target digest to
  `92ae190a68ee376b`, last token 8.
- Failure mode: larger row cohorts lose occupancy; fewer threadgroups lose
  useful parallelism; the vector transaction yields no full-model margin;
  larger lane work changes FP32 grouping; explicit full unrolling sharply
  increases generated-kernel cost.
- Why not to retry unchanged: the complete cohort family and direct load/
  unroll variants are measured on the exact production shapes and target.
- Reopen only if: a compiler/GPU change or new shader counters prove a
  different occupancy/register regime, or a load-sharing mechanism reduces
  streamed bytes while retaining A111's FP32 reduction order.
- Related commit or revert: all candidates remain isolated outside `main`;
  signed A111 in `22408c50c4` remains selected.

## PERF-FA132 - Fast-exponent fused affine-Q4 gate/up/SwiGLU arithmetic

- Hypothesis: one batch-one Metal dispatch can consume both Q4 gate/up
  matrices and emit the BF16 SwiGLU product while eliminating intermediate
  arrays and elementwise kernels.
- Scope: mixed 4.951-bpw Q5-class target, its 128 Q4 MLP gate/up linears,
  selected A100+A111, production-shape microbenchmarks, synthetic parity, and
  one sampled direct `128 / 32 warm / 128 timed` screen.
- Attempted change: one 8-SIMD by four-paired-row kernel retains MLX's Q4
  load/dot expression and explicitly materializes the gate, up, SiLU, and
  final product BF16 rounding points. The rejected form evaluates sigmoid
  with `metal::exp`; the route is opt-in through
  `SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU`.
- Benchmark evidence: separate/fused production-shape means are
  **0.598734317 / 0.557800892 ms**, a **6.837%** isolated reduction. The
  full model reaches **19.406869588 tok/s** during active host indexing.
- Correctness evidence: focused synthetic tests report zero mismatches at all
  selected shapes. The real model changes digest/last token from
  `d0193f6d413b68c1` / 11406 to `f2a59800c8d89f75` / 19. A temporary
  real-weight/real-hidden trace localizes the first mismatch to layer 62,
  element 36: gate and up are exact, while sigmoid rounds to BF16 `0x3a8c`
  instead of MLX's `0x3a8b`. The difference propagates through SiLU
  (`0xbbf0` versus `0xbbee`) and the final product (`0x3c8f` versus
  `0x3c8e`).
- Failure mode: the separate MLX kernels use safe/precise exponential
  arithmetic. `metal::exp` inside this fused custom kernel changes one real
  sigmoid rounding boundary. Volatile BF16 locals and explicit
  BF16-to-`ushort`-to-BF16 round trips do not repair it.
- Why not to retry unchanged: the full-model fixed-work trajectory is part of
  the performance contract, and the fast-exponent source fails it.
- Reopen only if: a future Metal/MLX compiler change makes `metal::exp`
  demonstrably bit-exact to the safe separate path at every real boundary.
  Replacing it with `metal::precise::exp` is materially different evidence,
  not a retry of this failed arm.
- Related commit or revert: the corrected precise-exp A114 candidate remains
  materially distinct from this failed arm. Its 64-layer trace, dedicated
  boundary negative control, focused parity, and two paired throughput windows
  pass; precise-exp A114 is promoted while the fast-exp source stays closed.

## PERF-FA133 - Four-SIMD fused affine-Q4 gate/up/SwiGLU geometry

- Hypothesis: halving A114's threadgroup from eight to four SIMD groups may
  reduce per-threadgroup register pressure and improve occupancy enough to
  offset twice as many threadgroups.
- Scope: the selected precise-exp fused Q4/G64 gate/up/SwiGLU kernel at the
  production K/N shape `5120/17408`.
- Attempted change: change only shader and host `SimdGroups` from eight to
  four while retaining two packs per lane and four paired results per SIMD
  group.
- Benchmark evidence: balanced 5,000-iteration control samples are
  **0.567918008, 0.564163733 ms**, mean **0.566040871**. Candidate samples are
  **0.566923292, 0.567484408 ms**, mean **0.567203850**, about **0.205% slower**.
- Correctness evidence: the focused Q4 QMV, fused SwiGLU, and precise
  sigmoid-boundary tests all pass bit-exactly.
- Failure mode: doubling the output-grid threadgroup count outweighs any
  register/occupancy benefit at the measured production shape.
- Why not to retry unchanged: the exact geometry and production shape have a
  balanced microbenchmark regression; a full-model launch has no supporting
  mechanism or signal.
- Reopen only if: new shader counters or a materially different fused kernel
  changes register pressure, occupancy, or per-threadgroup work.
- Related commit or revert: candidate stayed outside `main`; selected A114
  eight-SIMD geometry remains in signed `ca524c3282`.

## PERF-FA134 - Dynamically indexed lane-parallel fused-Q4 epilogue

- Hypothesis: lanes 0--3 can execute A114's four precise sigmoid/SiLU/product
  chains concurrently after the SIMD reductions, removing four serial
  `metal::precise::exp` operations from lane zero.
- Scope: the selected eight-SIMD/four-result fused Q4/G64 gate/up/SwiGLU
  kernel at production shape `5120/17408`.
- Attempted change: reduce all four gate/up accumulators first, then use
  `thread_index_in_simdgroup` as a dynamic index into the two four-element
  thread-local result arrays for lanes 0--3.
- Benchmark evidence: an initial balanced 5,000-iteration window appeared to
  improve control/candidate **0.566223700 -> 0.564063788 ms**, but the longer
  reversed 10,000-iteration window averages candidate
  **0.564099544 ms** versus control **0.562567529 ms**, about **0.272% slower**.
- Correctness evidence: all focused production shapes and the `-6.84375`
  precise sigmoid boundary pass bit-exactly.
- Failure mode: dynamic thread-local array selection likely adds addressing,
  register-spill, or compiler-selection cost that exceeds the parallel
  epilogue benefit.
- Why not to retry unchanged: the longer order-reversed microbenchmark
  overturns the shorter apparent gain.
- Reopen only if: generated-shader evidence proves the dynamic arrays stay in
  registers under a new compiler, or the result storage/layout changes
  materially.
- Related commit or revert: candidate stayed outside `main`; PERF-A117's
  compile-time register selection is materially different and is retained in
  signed `00d09138ce`.

## PERF-FA135 - Explicit vector transaction in the fused-Q4 load helper

- Hypothesis: one `packed_ushort4` transaction can replace the fused Q4
  helper's two packed-word reads and reduce load-instruction cost across the
  gate and up streams.
- Scope: selected A117 exact fused affine-Q4/G64 gate/up/SwiGLU at production
  shape `5120/17408`; ordinary Q4 execution is unchanged.
- Attempted change: load the same eight packed bytes through one explicit
  four-element 16-bit vector, then preserve the existing MLX-compatible
  nibble unpack, FP32 reduction, and lane-parallel precise epilogue.
- Benchmark evidence: two 10,000-iteration candidate samples are
  **0.563729096 / 0.558256000 ms**, mean **0.560992548**. A117 controls are
  **0.558807537 / 0.556857937 ms**, mean **0.557832737**. Candidate latency
  increases **0.003159811 ms / about 0.566%**.
- Correctness evidence: all focused Q4 QMV shapes, both fused shapes, and the
  `-6.84375` precise sigmoid boundary pass exactly.
- Failure mode: the wider source transaction does not reduce the dominant
  paired weight stream or arithmetic and lowers less efficiently than the
  compiler-selected scalar form in this two-stream fused shader.
- Why not to retry unchanged: the exact selected production shape regresses
  in two long samples despite bit-exact output.
- Reopen only if: generated shader evidence shows a compiler or alignment
  change that removes the current vector-lowering cost, or the load is shared
  across materially more arithmetic.
- Related commit or revert: candidate remained outside `main`; exact A117 was
  restored before the next experiment.

## PERF-FA136 - Load-time row concatenation of Q5 `qkv` and `z`

- Hypothesis: the 48 linear-attention layers can replace two affine-Q5/G64
  input-projection launches with one `N=16384` launch because `qkv` and `z`
  consume the same BF16 hidden state.
- Scope: production `K=5120`, `Nqkv=10240`, `Nz=6144`; selected A100 Q5
  kernel, A111/A114/A113/A117 target, and exact sampled direct benchmark.
- Attempted change: concatenate each projection pair's packed weights,
  scales, and biases by output row during model load, execute the resulting
  QLinear once, and split its output at row 10240. The path is opt-in through
  `SGLANG_MLX_NATIVE_Q5_QKV_Z_FUSION`.
- Benchmark evidence: ten order/reverse 2,000-iteration micros improve
  two-launch/one-launch aggregate **0.426645998 -> 0.420592346 ms**
  (**1.418893%**). Full-model fusion then reaches
  **19.441971742 / 19.406059113 tok/s** around adjacent same-dylib disabled
  control **19.469521945**; both candidate comparisons lose.
- Correctness evidence: micro output is byte-exact with digest
  `555793dfc2cf896f`; the focused Q5 suite passes; every full-model arm
  retains digest `d0193f6d413b68c1` and last token 11406.
- Failure mode: the isolated launch saving does not survive the load-time
  copied tensor representation and runtime split graph in the full target.
- Why not to retry unchanged: the implementation is exact but negative in
  both adjacent full-model comparisons, and temporarily adds copied-weight
  residency during model load.
- Reopen only if: the implementation consumes the two original matrices and
  writes the original two outputs directly in one native dispatch, removing
  both concatenation and split, or new profiling proves those costs absent.
- Related commit or revert: candidate remained outside `main`; A117 source was
  restored byte-for-byte after the screen.
