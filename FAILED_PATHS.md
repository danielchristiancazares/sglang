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
