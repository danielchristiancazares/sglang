# Performance Log

## Current Active Timings

| Benchmark | Baseline | Current qualified | Delta | Command | Last updated |
|---|---:|---:|---:|---|---|
| Qwen3.8-27B RadixArk, real sampled `6213/512`, reasoning preserved | 122.712 tok/s | 122.712 tok/s | 0.000 | `.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5` | 2026-08-16 22:40 PDT |
| Same production topology, fixed accepted length 3 | 171.263 tok/s | 171.263 tok/s | 0.000 | same client with launcher `-SimulateAcceptedLength 3` | 2026-08-16 22:40 PDT |
| Exact `199000+16` capacity | 199016 total tokens | 199016 total tokens | preserved | `.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600` | 2026-08-16 22:40 PDT |

No unqualified tree result replaces the production row. The active optimization
target is **greater than or equal to 200 tok/s under ordinary unsimulated real
sampling**, with enough margin to survive a second independent window and the
complete behavior/capacity qualification.

## Baseline

- Commit at the start of the current branch: `b8426ebe7c05e4b24e7393f1f81f947fc5f79905` on `main`. The qualified source line recorded in the compact notes is `9681850bed660b9079ee1aee906cda819603da7a` plus later restored launcher defaults.
- Hardware and OS: native Windows, one NVIDIA RTX 5090 display GPU with 32,607 MiB VRAM, WDDM scheduling, CUDA 13.3/MSVC toolchain, and the repository virtual environment.
- Model: immutable `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk`, served as `qwen3.8-27b` at `http://127.0.0.1:30000/v1`.
- Qualified build/runtime profile: real 200,000-token target and draft pools; one running request; language-only surface; Qwen3 reasoning parser; Qwen3 Coder tool parser; NEXTN linear rejection sampling; two MTP steps and three target rows; top-k-one chain; draft top-k 20/top-p aligned inside one multi-step CUDA graph; target and draft TRT-LLM MHA/XQA; FlashInfer prefill and sampling; Triton GDN/ReplaySSM; FP8 E4M3 draft KV; page size 64; 4096-token prefill chunks; four FP32 Mamba slots; FP4 autotuning with FP8 GEMM autotuning skipped; torch compile `default`; batch-one full decode graphs; scheduler receive interval 4; stream interval 4; incremental output; 128 MiB FlashInfer workspace.
- Real workload: exact 6213 prompt tokens and 512 completion tokens, temperature `1.0`, top-p `0.95`, top-k `20`, presence penalty `1.5`, preserved thinking, cache flushes between controlled runs, and SSE client timing. The decode metric excludes TTFT; end-to-end latency and TTFT remain recorded qualification evidence.
- Warmup policy: exact-shape warmup before retained windows. Production promotion requires at least five consecutive samples and a second independent window. Fixed-work probes retain token count and deterministic digest.
- Qualified real samples: `122.002, 122.739, 113.058, 118.119, 118.948, 119.239, 137.074, 124.047, 125.909, 125.980 tok/s`; mean **122.712**, median **122.371**, peak **137.074**.
- Qualified acceptance probes: `2.29596, 2.46154, 2.33790, 2.17872, 2.31674` emitted tokens per verification; mean **2.31817**.
- Qualified fixed samples: `170.995, 171.291, 171.125, 171.541, 171.363 tok/s`; mean **171.263**. Every run retained SHA-256 `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`.
- Behavior/capacity baseline: coherent preserved reasoning, arithmetic result `703`, exactly one parsed `multiply({"a":37,"b":19})` call, image/audio understanding disabled, exact `199000+16`, standalone OpenCode2 integration, all intended CUDA graph captures, and 1.84 GiB reported graph-end headroom on the final 200K relaunch.
- Environmental note: Chrome, ZCode, desktop clients, and WDDM residency have shifted retained windows materially. GPU clocks, power, temperature, utilization, free memory, process ancestry, and competing clients belong with every promotion window.

## Deltas

### 2026-08-16 13:29 PDT - PERF-BASELINE qualified real production

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
- Commit: pending.

### 2026-08-16 22:06 PDT - PERF-002 sparse-ancestry GDN tree replay

- Change: pair state changed from `[B,H,N,N,2]` to `[B,H,N,max_tree_depth,2]`; parameters are built once per value head/node; pair reductions are warp-parallel.
- Benchmark evidence: M12 dot reductions fell from 288 to 56. Final pre-lifetime-fix trace measured per-layer main/pair/parameter kernels at `26.162/5.415/1.779 us`, about **1.60 ms per 48-layer target cycle** from the preceding approximately 1.656 ms path.
- Correctness evidence: reference parity, accepted-path state commit, and CUDA-graph replay passed **3 tests**.
- Decision: retain exact sparse implementation. The direct cycle saving is small and cannot carry the 200 TPS target alone.
- Commit: pending.

### 2026-08-16 22:27 PDT - PERF-003 post-change width sweep

- Change: measured target-only M8, M12, and M16 after seam/GDN work; no shape was promoted.
- Benchmark evidence: emitted tokens/cycle were **2.737, 2.906, 3.061**. Five-run real means were **97.352, 94.685, 92.831 tok/s** in their respective WDDM windows. Corrected M12 raw values were `87.870, 101.393, 96.121, 98.484, 89.557`; M16 raw values were `98.158, 100.223, 93.694, 89.908, 82.173`.
- Correctness evidence: every retained request returned exactly 512 tokens with thinking enabled; M16 capture passed after the shared-buffer reseed repair.
- Decision: width-only tree changes are closed. Added width raises modest yield while increasing target cost.
- Commit: trace/tool/code evidence pending reconciliation.

### 2026-08-16 22:36 PDT - PERF-004 SWOR p/q calibration grid

- Change: collected native p/q overlap and path statistics for the 16-node topology `[-1,0,0,0,0,1,1,1,1,2,3,4,5,5,5,5]`; added an offline log analyzer.
- Benchmark evidence: exact `6213/2048` completed in 669 cycles at **3.061286 emitted/cycle**. Internal-node baseline overlaps were `0.75813, 0.70074, 0.51010, 0.47021, 0.41220, 0.66373`. The complete temperature/support grid improved them by at most **0.000245**.
- Correctness evidence: native exact SWOR path remained active; accepted-node histogram was `[0,514,57,18,9,364,36,16,9,40,11,6,249,31,15,3]`.
- Decision: scalar q temperature and retained support are closed. Branch-local proposal state or a stronger proposal model is required.
- Commit: pending.

## Candidate Inventory

| ID | Hypothesis | Scope | Status | Evidence / next gate |
|---|---|---|---|---|
| PERF-001 | Remove the cross-iteration speculative seam with a two-graph device-resident cycle. | CUDA graph backend, EAGLE draft/extend runners, worker bridge | Implemented; opt-in | Child graph test passed and steady M12 has two graph IDs. Needs coherent commit and production-relevant topology evidence. |
| PERF-002 | Store and compute only strict GDN ancestry; remove value-tile parameter recomputation. | `gdn_tree_replay.cuh`, Python binding/backend | Implemented; opt-in tree path | Three native CUDA tests passed; measured direct saving is about 0.06 ms/cycle. |
| PERF-003 | Apply exact branch-local presence/frequency state to SWOR p and q. | sampling state, fixed topology metadata, draft graph buffers, target verifier | Next implementation candidate | q mass outside p is 0.096-0.164 on dominant rows; offline oracle already encodes branch-local semantics. Must remain graph-safe and exact. |
| PERF-004 | Attribute target/composite graph time by kernel family and exact graph ID before another kernel rewrite. | trace analyzer and Qwen3.5 target/draft hot paths | Ready for survey | Current whole-trace families mix prefill and graph work. The MacPro ledger confirms synchronized microbenchmarks can mis-rank async serving changes. |
| PERF-005 | Extend the device-resident cycle to exact linear rejection sampling. | proposal sampling, exact-q buffers, verification/extend bridge | Survey | Production baseline still pays graph/scheduler boundaries. Requires exact RNG/q/residual semantics and a measured cycle projection. |
| PERF-006 | Improve proposal quality with a distinct trained/calibrated proposal mechanism. | MTP adapter/training, standalone draft, or device-side mixture oracle | Survey | RadixArk and Gittensor embedded MTP tensors are byte-identical. Temperature/support calibration is flat. Any training path needs held-out behavior evidence. |
| PERF-007 | Fuse remaining target FP8/BF16 projection work only after graph-specific attribution. | Qwen3.5 GDN input/BA/output projections and ModelOpt linear kernels | Survey | Existing qkvz and BA projections are already merged separately; prior target/draft quantizers and broad GEMM autotuning lost. |
| PERF-008 | Build a deeper tree only after an oracle projection clears 200 TPS plus margin. | sparse p/q oracle and topology optimizer | Gated | Current-q optimistic 32-node search reached only 4.0921 expected outputs and stayed cost-limited. |

## Commit History

- Current branch work after `b8426ebe7c05e4b24e7393f1f81f947fc5f79905` is uncommitted. Atomic commits will separate validated graph scheduling, sparse GDN, offline oracle/tooling, and record-only evidence while leaving unrelated worktree paths untouched.

## Historical Supersession

The original root `NOTES.md` was deleted by commit
`b8426ebe7c05e4b24e7393f1f81f947fc5f79905` when its recovery record was split
into the compact `notes/` layer. It remains recoverable as
`b8426ebe7c05e4b24e7393f1f81f947fc5f79905^:NOTES.md` and contains 2,428
lines through the 18:35 production restoration. These are the governing
supersessions extracted from that original ledger:

| Topic | Selected result | Superseded results |
|---|---:|---|
| Fixed `6213/512` | **171.263 tok/s** | 167.776, 162.726, 159.973, 156.968, 135.167, 86.016 |
| Real sampled `6213/512` | **122.712 tok/s** | 121.075, 117.794, 110.750, 98.126, 96.110 |
| Near-limit `199000+16` prompt/decode | **2608.263 / 102.358 tok/s** | 2570.356, 2429.153, 2423.812, 2200.563 prompt results |
| Production capacity | **200000** context and pools | 232000 operating-margin experiment |
| Speculation geometry | **2 steps / 3 rows** | 3 steps / 4 rows and 1 step / 2 rows |
| Target verification | **TRT-LLM MHA/XQA** | FlashInfer-prefill verification |
| Draft extension | **captured CUDA graph** | eager draft extension |
| Draft q | **top-k 20 in one multi-step graph** | eager/per-depth aligned proposal |
| Production topology | **linear exact rejection** | current-q tree-width/depth/topology candidates |
