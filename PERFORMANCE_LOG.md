# Performance Log

## Current Active Timings

| Benchmark | Baseline | Current | Delta | Command | Last Updated |
|---|---:|---:|---:|---|---|
| Qwen3.8-27B Q4_0, 8 concurrent requests, 32 output tokens each | 32.953 TPS | 38.016 TPS | +5.063 TPS | `.venv-mac-metal/bin/python benchmark/mac/bench_sglang_sampling.py --concurrency 8 --output-tokens 32` | 2026-08-16 22:26 PDT |
| Qwen3.8-27B Q4_0, batch 24, 128 output tokens each, real top-k/top-p sampling | 49.500 TPS | **62.034 TPS** | **+12.534 TPS** | `.venv-mac-metal/bin/python benchmark/mac/bench_sglang_batched_request.py --url http://127.0.0.1:30001/generate --batch-size 24 --output-tokens 128` | 2026-08-16 22:35 PDT |
| Qwen3.8-27B RadixArk, real sampled `6213/512`, reasoning preserved | 122.712 tok/s | 122.712 tok/s | 0.000 | `.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5` | 2026-08-16 22:40 PDT |
| Post-correctness linear comparison, second warmed five-run window | 122.712 tok/s | 124.775 tok/s measured | +2.063 / +1.681% | same exact real-sampling command | 2026-08-16 23:24 PDT |
| Same production topology, fixed accepted length 3 | 171.263 tok/s | 171.263 tok/s | 0.000 | same client with launcher `-SimulateAcceptedLength 3` | 2026-08-16 22:40 PDT |
| Exact `199000+16` capacity | 199016 total tokens | 199016 total tokens | preserved | `.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600` | 2026-08-16 22:40 PDT |

Target: at least **60 TPS** with real sampling. Achieved with a three-run end-to-end median of **62.034 TPS**; warmed decode windows sustain **72.15–72.83 TPS**.

All M8/M12/M16 tree timings below are mechanism-only evidence. A deterministic
non-front path reproducer found that the unified hybrid pool moved virtual slot
ids as physical target-KV locations, and the multi-layer worker could skip
front compaction entirely. The repair passes isolated eager and captured
multi-cycle parity; a corrected full-model tree run remains required before any
tree throughput can be ranked for production.

## Baseline

- Commit: `b270c6521ced7af70c6ff8d4740f89f752a3afd2` plus the existing dirty MPS/Metal port in this worktree.
- Hardware / OS: MacPro7,1; AMD Radeon Pro W6900X 32 GB; macOS 26.6 (25G72); x86_64.
- Runtime: Python 3.11 virtual environment `.venv-mac-metal`; PyTorch 2.2.2; MPS available.
- Model: `/Users/daniel/models/Qwen3.8-27B-Q4_0/Qwen3.8-27B-Q4_0.gguf` with tokenizer `/Users/daniel/models/Qwen3.8-27B-tokenizer`.
- Server: `python -m sglang.launch_server` with GGUF loading, float32 MPS execution, torch-native attention, PyTorch sampling, language-model-only mode, radix cache disabled, 24 maximum running requests, 4096 maximum total tokens, and 24 Mamba cache slots.
- Workload: eight barrier-synchronized `/generate` requests; each has a 16-token plain-text prompt, temperature 0.8, top-p 0.9, 32 forced output tokens, and EOS ignored. Aggregate TPS is 256 output tokens divided by client-observed wall time.
- Warmup policy: discard server/model/Metal first-use compilation run, then retain five consecutive samples from the already-loaded server.
- Discarded first-use run: `6.874 TPS` (`37.240 s`).
- Raw retained TPS: `32.094`, `32.931`, `32.953`, `32.977`, `33.009`.
- Raw retained wall times: `7.977 s`, `7.774 s`, `7.769 s`, `7.763 s`, `7.755 s`.
- Median: `32.953 TPS`; median wall time `7.769 s`.

## Deltas

### 2026-08-16 20:23 PDT - BASELINE

- Change: measurement only; no new optimization applied.
- Benchmark evidence: warmed five-sample median `32.953 TPS` from raw values above.
- Correctness evidence: all eight responses returned 32 output token IDs and successful HTTP status; generated samples were coherent continuations.
- Decision: baseline accepted. Required success threshold is `>=34.953 TPS` under the same warmed repeated protocol.
- Commit: pending with the surrounding MPS port.
- Change: promoted aligned draft q into the single multi-step graph on the two-step linear topology.
- Benchmark evidence: fresh ten real samples listed above; **122.712 tok/s mean** and **122.371 median**. Five native acceptance probes averaged **2.31817**.
- Correctness evidence: exact q used for proposal and Leviathan rejection, mutable CUDA-graph replay coverage, preserved reasoning/tools, and full 200K capacity.
- Decision: qualified baseline and benchmark of record.
- Commit: retained in the qualified source line documented by `notes/current-state.md` and `notes/decisions.md`.

### 2026-08-16 13:33 PDT - PERF-BASELINE-FIXED all-accepted ceiling

- Change: native BF16 full-attention sigmoid gate active; fixed accepted length 3 on the production geometry.
- Benchmark evidence: `170.995, 171.291, 171.125, 171.541, 171.363 tok/s`; mean **171.263**.
- Correctness evidence: deterministic 512-token digest retained; native gate parity passed at production widths.
- Decision: fixed-work cost baseline. It proves the two-step geometry cannot reach 200 tok/s even with perfect acceptance.
- Commit: retained in the qualified source line.

### 2026-08-16 14:35 PDT - PERF-THREE-STEP fixed compute feasibility

- Change: three MTP steps, four target rows, forced accepted length 4.
- Benchmark evidence: fresh `194.466, 197.795, 197.314, 201.251, 183.687 tok/s`; conservative mean **194.903**, uncontended first-four mean **197.707**, externally observed peak **201.251**, and server windows up to 207.60.
- Correctness evidence: fixed-work digest retained and full 200K graphs captured.
- Decision: compute feasibility only. Ordinary sampling on this geometry later measured **117.239 tok/s mean** and 2.403756 emitted tokens/cycle, so no promotion occurred.
- Commit: evidence retained in `notes/experiment-log.md`.

### 2026-08-16 22:06 PDT - PERF-001 two-graph device-resident tree cycle

- Change: retained CUDA child graphs now form one parent containing draft extend, a device bridge, and the next draft decode. Target verification plus this composite parent are the only steady graph launches.
- Benchmark evidence: pre-change M12 extend -> next-draft gap was **3.228/2.475 ms mean/median**. Trace `target_width_m12-20260816-220558` contains target graph 6 at **18.378/18.357 ms** and composite graph 15 at **5.654/5.600 ms**; the former seam is now inside graph 15. Remaining median host gaps were 1.210 and 2.643 ms.
- Correctness evidence: generic shared-address child-graph CUDA test passed; fixed-width prefix-tail semantics and device-cycle unit tests passed. A shared-input-buffer lifetime defect found by M16 was repaired by reseeding extend inputs before warm/capture.
- Decision: retain as opt-in infrastructure. Production throughput has not crossed the baseline.
- Commit: `d0116b54e5766932a46e06e0a66c3672370eaff8`.

### 2026-08-16 22:06 PDT - PERF-002 sparse-ancestry GDN tree replay

- Change: pair state changed from `[B,H,N,N,2]` to `[B,H,N,max_tree_depth,2]`; parameters are built once per value head/node; pair reductions are warp-parallel.
- Benchmark evidence: M12 dot reductions fell from 288 to 56. Final pre-lifetime-fix trace measured per-layer main/pair/parameter kernels at `26.162/5.415/1.779 us`, about **1.60 ms per 48-layer target cycle** from the preceding approximately 1.656 ms path.
- Correctness evidence: reference parity, accepted-path state commit, and CUDA-graph replay passed **3 tests**.
- Decision: retain exact sparse implementation. The direct cycle saving is small and cannot carry the 200 TPS target alone.
- Commit: `d0116b54e5766932a46e06e0a66c3672370eaff8`.

### 2026-08-16 22:27 PDT - PERF-003 post-change width sweep

- Change: measured target-only M8, M12, and M16 after seam/GDN work; no shape was promoted.
- Benchmark evidence: emitted tokens/cycle were **2.737, 2.906, 3.061**. Five-run real means were **97.352, 94.685, 92.831 tok/s** in their respective WDDM windows. Corrected M12 raw values were `87.870, 101.393, 96.121, 98.484, 89.557`; M16 raw values were `98.158, 100.223, 93.694, 89.908, 82.173`.
- Correctness evidence: every retained request returned exactly 512 tokens with thinking enabled; M16 capture passed after the shared-buffer reseed repair.
- Decision: width-only tree changes are closed. Added width raises modest yield while increasing target cost.
- Commit: `d0116b54e5766932a46e06e0a66c3672370eaff8`.

### 2026-08-16 22:36 PDT - PERF-004 SWOR p/q calibration grid

- Change: collected native p/q overlap and path statistics for the 16-node topology `[-1,0,0,0,0,1,1,1,1,2,3,4,5,5,5,5]`; added an offline log analyzer.
- Benchmark evidence: exact `6213/2048` completed in 669 cycles at **3.061286 emitted/cycle**. Internal-node baseline overlaps were `0.75813, 0.70074, 0.51010, 0.47021, 0.41220, 0.66373`. The complete temperature/support grid improved them by at most **0.000245**.
- Correctness evidence: native exact SWOR path remained active; accepted-node histogram was `[0,514,57,18,9,364,36,16,9,40,11,6,249,31,15,3]`.
- Decision: scalar q temperature and retained support are closed. Branch-local proposal state or a stronger proposal model is required.
- Commit: `d0116b54e5766932a46e06e0a66c3672370eaff8`.

### 2026-08-16 23:11 PDT - PERF-C001 non-front accepted-path correctness

- Change: translated unified-pool accepted-path and prefix-tail moves from virtual token ids to physical full-KV ids, kept MLA dense kernel ids separate, and made tree-path compaction mandatory for both single- and multi-layer EAGLE workers.
- Benchmark evidence: no throughput number was retained. The earlier M8/M12/M16 values are now mechanism-only until a corrected full-model non-front-path gate passes.
- Correctness evidence: before the repair, the minimal nonidentity-map test copied four of six sentinel elements from the wrong physical rows. The strengthened factory test covers MHA and MLA page translation; a captured four-cycle `[0,3,7]`-style sequence covers alternating non-front paths, rejected-slot reclamation, virtual-id reuse, target K/V, compacted tokens/hidden rows, and terminal next-draft state. Focused native CUDA finished **4 passed plus 2 subtests**; the combined accepted-path/composite-graph/GDN suite finished **8 passed plus 2 subtests**. Allocator and move-gate CPU suites finished **65 passed** and **5 passed**.
- Decision: retain the repair and keep every tree topology production-ineligible pending a corrected full-model comparison with the qualified linear baseline.
- Commit: `3f276e8acda4db5911db9a69a689deb10bae8360`.

### 2026-08-16 23:24 PDT - PERF-C002 fresh qualified-linear comparison

- Change: launched the unchanged production-default linear topology from `3f276e8acda4` with the historical server seed `783025237`; every tree, SWOR, simulation, adaptive, and device-resident-cycle control remained inactive.
- Benchmark evidence: the first consecutive real window was `84.130, 114.807, 118.664, 119.385, 124.278 tok/s`, mean **112.253**, median **118.664**. The second independent warmed window was `123.237, 123.741, 125.001, 128.689, 123.207 tok/s`, mean **124.775**, median **123.741**. All ten combined mean **118.514** and median **123.222**; the low first request remains retained as startup/JIT evidence. Every request was exact `6213+512`, `finish_reason=length`, and thinking remained enabled.
- Acceptance evidence: five native probes were `2.381395, 2.169492, 2.160338, 2.124481, 2.188034` emitted tokens per verification, mean **2.204748**. This was 4.893% below the historical acceptance mean and explains much of the first-window TPS loss.
- Correctness/environment evidence: `/health` returned 200; `/model_info` reported image/audio understanding false; target verify, draft decode, and draft extend graphs captured in **42.42**, **1.56**, and **1.09 seconds** with 1.74 GiB initially reported free. The live server remained healthy after both windows. WDDM clients included Chrome, Edge WebView, Docker Desktop, and ordinary shell/display processes; post-request JIT residency left 222 MiB free.
- Decision: the accepted-path repair does not regress the qualified top-k-one production chain. Use the stable **124.775 tok/s** second window as the immediate matched control while preserving the complete ten-run **118.514 tok/s** evidence. Tree work remains blocked on full-model non-front parity.
- Commit: record update pending.

### 2026-08-16 23:47 PDT - PERF-005 exact linear device-resident cycle, dense-race form

- Change: extended the composed draft-extend/bridge/next-draft parent to ordinary top-k-one rejection sampling. The bridge now uses graph-stable temperature, top-p, accumulated additive penalties, logit bias, draft top-k 20, sampled token, and the exact q consumed by verification. Raw child-graph launch bypasses PyTorch's generator-offset replay hook, so this first form refreshes two full-vocabulary exponential-race rows before each parent launch.
- Benchmark evidence: exact real samples were `117.251, 121.340, 118.959, 131.667, 123.663 tok/s`, mean **122.576**, median **121.340**. This is **1.762% below** the fresh warmed linear control at 124.775 tok/s.
- Acceptance evidence: `2.188034, 2.216450, 2.188034, 2.275556, 2.359447`, mean **2.245504**, was **1.849% above** the matched control acceptance. Higher yield with lower TPS identifies an execution-cost regression.
- Correctness evidence: exact `6213+64` smoke completed in 27 cycles at 2.370370 emitted/cycle; all five full samples returned exactly 512 tokens with thinking enabled. Proposal/cycle CPU suites passed **12**, **5**, and **4** tests across the affected files; composite/FlashInfer/exact-tree CUDA suites passed **16 tests**.
- Decision: the architecture is retained for one smaller-randomness test. Replace the two 248K-wide exponential refreshes with FlashInfer categorical sampling driven by explicit graph-stable seed/offset tensors. Close the linear composite unchanged if that still does not beat the matched control.
- Commit: uncommitted experiment.

#### Explicit-seed categorical refinement

- Replaced vocab-wide races with FlashInfer categorical sampling and one graph-stable seed plus two explicit offset scalars. Fixed-offset raw replay is deterministic; advancing the offset changes sampled tokens while preserving exact `q(X)`.
- Five real samples were `115.058, 116.444, 120.530, 123.907, 124.434 tok/s`, mean **120.075**, median **120.530**, or **3.767% below** the 124.775 control.
- Five acceptance probes averaged **2.277991**, **3.322% above** control. Their 1,124 verification cycles took 23.753 seconds combined, **21.132 ms/cycle**, versus **20.771 ms/cycle** for the ordinary path. The categorical form improved the dense-race composite's 21.239 ms/cycle, while the composition itself stayed slower.
- Final decision: close exact linear device-cycle composition as a throughput candidate. Retain it opt-in as exact architectural infrastructure; keep production defaults unchanged.

## Candidate Inventory

| ID | Hypothesis | Scope | Status | Evidence |
|---|---|---|---|---|
| PERF-001 | Add a Q4_0 batch-8 Metal specialization so steady batch-8 decode reads each packed matrix once instead of twice. | `gguf_q4_0.mm` Q4_0 kernel and host dispatch | Rejected | Correct but regressed representative MLP Q4_0 median from `0.345 ms` to `0.778 ms`; removed. See `FAILED_PATHS.md`. |
| PERF-002 | Fuse or remove remaining GDN pack, normalization, and reorder launches. | Qwen3.5 GDN MPS path and native Metal extension | Pending profile | Production path launches native packing and gated-norm/reorder around native recurrent attention for most decoder layers. |
| PERF-003 | Reduce full-vocabulary PyTorch sampling overhead on MPS. | `sampler.py` / native sampling | Pending profile | Every decode step performs top-p sampling over the full vocabulary; impact relative to model kernels remains unmeasured. |
| PERF-004 | Remove proven-redundant `.contiguous()` conversions and metadata copies in native MPS wrappers. | `mps/ops.py`, attention and GDN callers | Pending trace | Calls are production-reachable, but views may already be contiguous and therefore free. |
| PERF-005 | Wire the dormant Metal Q/K norm + RoPE + QKV/gate preparation kernel into full-attention layers. | `qwen3_5.py`, `gguf_q4_0.mm` | Rejected | Isolated batch-8 preparation fell from `2.225 ms` to `0.162 ms`, but clean end-to-end median regressed from `32.309` to `30.680 TPS`; production wiring removed. |
| PERF-006 | Reduce Q5_K/Q6_K batch-8 accumulator pressure by reusing smaller or batch-24 tiles. | `gguf_q4_0.mm` quantized matmul dispatch | Rejected | Q5_K tile-4 regressed `0.809 -> 0.888 ms`; Q6_K tile-4 regressed `27.227 -> 32.446 ms`; batch-24 vec4 regressed Q6_K to `34.166 ms`. |
| PERF-007 | Vectorize Q6_K dequantization across four adjacent weights with four two-request SIMD subgroups. | `gguf_q4_0.mm` Q6_K batch-8 kernel | Retained | LM-head microbenchmark improved `27.227 -> 9.186 ms` (`35.7 -> 105.7 GiB/s`); Q6_K reference relative error `4.64191e-07`. |
| PERF-008 | Apply the same vec4/subgroup geometry to the repeated Q5_K GDN output projections. | `gguf_q4_0.mm` Q5_K batch-8 kernel | Retained | Representative projection improved `0.809 -> 0.356 ms` (`24.9 -> 56.6 GiB/s`); combined end-to-end median is `38.016 TPS`. |
| PERF-009 | Apply batch-subgroup reuse or alter unroll depth in the Q4_0 batch-eight kernel. | `gguf_q4_0.mm` Q4_0 kernels | Rejected | Four-subgroup vec4 regressed `0.350 -> 0.686 ms`; existing split kernel gave `0.390 ms`; unroll 2 and 8 gave `0.402` and `0.405 ms`. |
| PERF-010 | Fuse GDN projection packing with the decode causal convolution. | `gguf_q4_0.mm` GDN glue kernels | Rejected | Correct fused kernel measured `0.151 ms` versus `0.147 ms` for the separate chain; experimental kernel and benchmark removed. |
| PERF-011 | Coalesce near-simultaneous requests into one idle prefill batch. | MPS normal scheduler loop | Rejected | A 2 ms window produced `4 + 4` prefills and `37.876 TPS`; even a single prebatched size-eight request reached only `39.053 TPS`. |
| PERF-010 | Use the checkpoint's bundled NEXTN block for speculative decoding. | SGLang speculative control plane + MPS GDN state verify | Functional, rejected for throughput | Fully served coherent sampled output; batch-1 measured `4.872 TPS`, with accept length `2.80/4` and draft acceptance rate `0.60`. |
| PERF-011 | Vectorize exact-batch-24 Q6_K LM-head dequantization. | `gguf_q4_0.mm` Q6_K kernel | Retained | `57.343 -> 21.565 ms`; reference relative error `3.36184e-07`. |
| PERF-012 | Extend vectorized Q5_K projection to exact batch 24. | `gguf_q4_0.mm` Q5_K kernel | Retained | `2.081 -> 0.891 ms` per GDN projection; reference relative error `3.81316e-07`; final median `62.034 TPS`. |
| PERF-001 | Remove the cross-iteration speculative seam with a two-graph device-resident cycle. | CUDA graph backend, EAGLE draft/extend runners, worker bridge | Implemented; opt-in | Child graph test passed and steady M12 has two graph IDs. Committed in `d0116b54e5`; production relevance remains blocked by the tree correctness/full-model gate. |
| PERF-002 | Store and compute only strict GDN ancestry; remove value-tile parameter recomputation. | `gdn_tree_replay.cuh`, Python binding/backend | Implemented; opt-in tree path | Three native CUDA tests passed; measured direct saving is about 0.06 ms/cycle. Committed in `d0116b54e5`. |
| PERF-003 | Apply exact branch-local presence/frequency state to SWOR p and q. | sampling state, fixed topology metadata, draft graph buffers, target verifier | Paused at correctness gate | q mass outside p is 0.096-0.164 on dominant rows; offline oracle already encodes branch-local semantics. Resume only after corrected full-model non-front path parity and a fresh linear baseline. |
| PERF-004 | Attribute target/composite graph time by kernel family and exact graph ID before another kernel rewrite. | trace analyzer and Qwen3.5 target/draft hot paths | Ready for survey | Current whole-trace families mix prefill and graph work. The MacPro ledger confirms synchronized microbenchmarks can mis-rank async serving changes. |
| PERF-005 | Extend the device-resident cycle to exact linear rejection sampling. | proposal sampling, exact-q buffers, verification/extend bridge | Closed for throughput; retained opt-in | Dense races reached 122.576 tok/s; explicit-seed categorical reached 120.075 versus 124.775 control despite higher acceptance. Composite cycle cost remained 1.7% slower. |
| PERF-006 | Improve proposal quality with a distinct trained/calibrated proposal mechanism. | MTP adapter/training, standalone draft, or device-side mixture oracle | Survey | RadixArk and Gittensor embedded MTP tensors are byte-identical. Temperature/support calibration is flat. Any training path needs held-out behavior evidence. |
| PERF-007 | Fuse remaining target FP8/BF16 projection work only after graph-specific attribution. | Qwen3.5 GDN input/BA/output projections and ModelOpt linear kernels | Survey | Existing qkvz and BA projections are already merged separately; prior target/draft quantizers and broad GEMM autotuning lost. |
| PERF-008 | Build a deeper tree only after an oracle projection clears 200 TPS plus margin. | sparse p/q oracle and topology optimizer | Gated | Current-q optimistic 32-node search reached only 4.0921 expected outputs and stayed cost-limited. |

### 2026-08-16 20:29 PDT - PERF-001

- Change: temporarily instantiated `q4_0_small_batch_impl<8>` and selected it for decode batches above six, reducing batch-eight Y groups from two to one.
- Benchmark evidence: representative `blk.8.ffn_gate.weight` Q4_0 shape `(5120, 17408)`, batch eight. Existing tile-four raw medians were `0.345`, `0.345`, `0.382 ms` (median `0.345 ms`). Tile-eight raw medians were `0.440`, `0.801`, `0.778`, `0.798`, `0.440 ms` (median `0.778 ms`), a `125.5%` regression.
- Correctness evidence: Q4_0 batch-eight output matched explicit GGUF dequantization and CPU F32 matmul with maximum absolute error `8.34465e-07` and relative error `4.46011e-07`.
- Decision: rejected and removed. The reduced weight traversal did not overcome tile-eight register/occupancy pressure on the W6900X.
- Commit: none; regressing kernel change removed. The new Q4_0 correctness coverage remains.
- `2f49a60b46c62e728fb7db00a0d042248c27c8f4` — restored the continuing performance and failed-path ledgers.
- `d0116b54e5766932a46e06e0a66c3672370eaff8` — committed the device-resident cycle, sparse GDN replay, SWOR oracle/tooling, tests, and profiles behind opt-in controls.
- `3f276e8acda4db5911db9a69a689deb10bae8360` — fixed accepted-path virtual-to-physical relocation, made front compaction mandatory, and added captured multi-cycle serial parity.

### 2026-08-16 22:11 PDT - PERF-005

- Change: temporarily routed Qwen3.5 full-attention preparation through the existing native Metal kernel that fuses Gemma Q/K normalization, partial NeoX RoPE, and QKV/gate unpacking.
- Benchmark evidence: exact production-shape batch-8 microbenchmark improved from `2.225 ms` to `0.162 ms`. Clean warmed end-to-end three-sample medians moved from `32.309 TPS` (`32.434`, `32.309`, `32.269`) to `30.680 TPS` (`30.486`, `30.814`, `30.680`), a `5.0%` regression. A 128-token comparison also favored the existing path (`40.470` versus `38.100 TPS`).
- Correctness evidence: fused Q/K, V, and gate outputs matched the PyTorch reference at `rtol=2e-5`, `atol=2e-5`; native grouped-query attention still passed with maximum error `4.76837e-07`.
- Decision: rejected and removed. Per-operation synchronization overstated the value of collapsing the asynchronous PyTorch/MPS command chain.
- Commit: none.

### 2026-08-16 22:17 PDT - PERF-006

- Change: separately tried the existing four-request Q5_K/Q6_K specialization and the batch-24 Q6_K vec4 specialization for batch eight.
- Benchmark evidence: representative Q5_K GDN output projection regressed from `0.809 ms` to `0.888 ms`; the Q6_K LM head regressed from `27.227 ms` to `32.446 ms` with the tile-four kernel and to `34.166 ms` with the batch-24 vec4 kernel.
- Correctness evidence: dispatch-only experiments retained the already-tested quantized kernels; no numerical mismatch was observed.
- Decision: rejected and removed. The alternative register geometries did not fit batch eight without a dedicated specialization.
- Commit: none.

### 2026-08-16 22:19 PDT - PERF-007

- Change: added a dedicated Q6_K batch-8 vec4 kernel. Four eight-lane subgroups share a SIMD group; each subgroup processes two requests while dequantizing and dotting four adjacent weights at once.
- Benchmark evidence: full `output.weight` Q6_K `(5120, 248320)` LM-head median improved from `27.227 ms` (`35.7 GiB/s`) to `9.186 ms` (`105.7 GiB/s`), a `66.3%` reduction.
- Correctness evidence: GGUF dequantized F32 reference passed at batch eight with maximum absolute error `1.19209e-06` and relative error `4.64191e-07`.
- Decision: retained for end-to-end server validation.
- Commit: pending.

### 2026-08-16 22:26 PDT - PERF-008 and end-to-end validation

- Change: added the analogous Q5_K batch-8 vec4 kernel and enabled the dedicated Q5_K/Q6_K paths only when the runtime batch is exactly eight.
- Benchmark evidence: representative Q5_K `(6144, 5120)` GDN output projection improved from `0.809 ms` (`24.9 GiB/s`) to `0.356 ms` (`56.6 GiB/s`), a `56.0%` reduction. After a discarded `34.978 TPS` first-use run, the real workload produced `37.965`, `38.027`, `37.887`, `38.016`, and `38.020 TPS`; median `38.016 TPS`, up `5.063 TPS` (`15.4%`) from baseline. Median wall time improved from `7.769 s` to `6.734 s`.
- Correctness evidence: all eight responses retained 32 output IDs. The combined quantized reference suite passed; Q5_K maximum absolute error was `1.54972e-06` and relative error `4.87504e-07`, while Q6_K retained `1.19209e-06` and `4.64191e-07`.
- Decision: retained. The remaining target gap is `4.937 TPS` to `42.953 TPS`.
- Commit: pending; the Metal extension file contains the surrounding uncommitted MPS port.

### 2026-08-16 22:29 PDT - PERF-009

- Change: separately tested a four-subgroup Q4_0 vec4 kernel, the already-present two-half batch-eight split kernel, and `chunks_per_thread` values two and eight around the existing value four.
- Benchmark evidence: representative Q4_0 `(5120, 17408)` baseline `0.350 ms`; four-subgroup vec4 `0.686 ms`, two-half split `0.390 ms`, unroll two `0.402 ms`, and unroll eight `0.405 ms`.
- Correctness evidence: no candidate was retained; the production dispatch and kernel source were restored after each microbenchmark.
- Decision: rejected. The current four-row-per-SIMD, four-chunk kernel remains the best measured Q4_0 geometry.
- Commit: none.

### 2026-08-16 22:34 PDT - PERF-010

- Change: temporarily fused projection unpacking, decode causal-convolution state update, SiLU, and gate/a/b extraction into one Metal dispatch.
- Benchmark evidence: alternating warmed batch-eight median was `0.147 ms` for the existing pack-plus-convolution chain and `0.151 ms` for the fused kernel; no measurable opportunity remained after asynchronous command submission.
- Correctness evidence: fused mixed QKV, gate, a/b outputs, and mutated convolution state matched the existing two-kernel path; the full native fused-op test passed.
- Decision: rejected and removed, including the temporary microbenchmark.
- Commit: none.

### 2026-08-16 22:45 PDT - PERF-011

- Change: temporarily added an opt-in idle-only MPS request-coalescing delay and tested a 2 ms window; separately submitted one HTTP request already containing all eight sequences as the perfect-coalescing upper bound.
- Benchmark evidence: the delay changed the observed prefill split from `1 + 7` to `4 + 4`, while a warmed concurrent-request sample measured `37.876 TPS`. The single size-eight batched request measured `39.053 TPS`, only `1.037 TPS` above the retained `38.016 TPS` median and still `3.900 TPS` below target.
- Correctness evidence: all paths returned the requested 256 output tokens; no scheduler state or admission rules beyond the bounded idle delay were changed.
- Decision: rejected and removed. The upper bound proves request coalescing cannot close the remaining gap, and retaining the delay would spend single-request TTFT for an unproven median gain.
- Commit: none.

### 2026-08-16 22:35 PDT - PERF-010 through PERF-012 and 60 TPS validation

- NEXTN: completed native-Metal target/draft loading, torch-native multi-step attention, top-k1 chain construction, real top-k/top-p verification sampling, causal-conv/GDN checkpoints, and accepted-state commit. It served coherent output end to end. Its batch-1 result was `4.872 TPS`; scheduler telemetry reported mean accept length `2.80` out of four verify tokens and draft acceptance rate `0.60`, so the experiment was retained as functionality and closed as the immediate throughput path.
- Q6_K change: exact batch 24 now dequantizes four adjacent weights per lane and processes six requests per eight-lane subgroup. Removing exact-24 bounds branches reduced the full `output.weight` median from `57.343 ms` to `21.565 ms` (`62.4%`).
- Q5_K change: the same vec4/subgroup layout now covers exact batch 24 for every GDN output projection, reducing representative `blk.0.ssm_out.weight` from `2.081 ms` to `0.891 ms` (`57.2%`).
- End-to-end evidence: three independent 24-request runs, each returning 128 sampled tokens per request (3,072 output tokens), measured `62.034`, `61.856`, and `62.556 TPS`; median `62.034 TPS`. Sampling used temperature `0.8`, top-p `0.9`, top-k `20`, and ignored EOS. Generated continuations were coherent.
- Steady evidence: warmed scheduler windows measured `72.80`, `72.31`, `72.83`, `72.51`, `72.49`, and `72.15 TPS`.
- Correctness evidence: the Q4_0/Q4_1/Q5_K/Q6_K GGUF reference suite passed at batch 24 and 17 output rows. Q5_K maximum absolute/relative error was `1.60933e-06` / `3.81316e-07`; Q6_K was `1.07288e-06` / `3.36184e-07`. Native fused-op, GDN, attention, speculative-control, and speculative-state tests also passed.
- Decision: target achieved with repeatable end-to-end margin; retain both exact-batch-24 kernels.
