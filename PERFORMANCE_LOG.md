# Performance Log

## Current Active Timings

| Benchmark | Baseline | Current | Delta | Command | Last Updated |
|---|---:|---:|---:|---|---|
| Qwen3.8-27B Q4_0, 8 concurrent requests, 32 output tokens each | 32.953 TPS | 38.016 TPS | +5.063 TPS | `.venv-mac-metal/bin/python benchmark/mac/bench_sglang_sampling.py --concurrency 8 --output-tokens 32` | 2026-08-16 22:26 PDT |
| Qwen3.8-27B Q4_0, batch 24, 128 output tokens each, real top-k/top-p sampling | 49.500 TPS | **62.034 TPS** | **+12.534 TPS** | `.venv-mac-metal/bin/python benchmark/mac/bench_sglang_batched_request.py --url http://127.0.0.1:30001/generate --batch-size 24 --output-tokens 128` | 2026-08-16 22:35 PDT |
| Qwen3.8-27B RadixArk, real sampled `6213/512`, reasoning preserved | 122.712 tok/s | 122.712 tok/s | 0.000 | `.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5` | 2026-08-16 22:40 PDT |
| Post-correctness linear comparison, second warmed five-run window | 122.712 tok/s | 124.775 tok/s measured | +2.063 / +1.681% | same exact real-sampling command | 2026-08-16 23:24 PDT |
| Selective target NVFP4 (`AttnNVFP4`) candidate, real sampled `6213/512`, admission window 1 | 124.775 tok/s | 131.707 tok/s mean / 130.824 median (unqualified) | +6.932 / +5.556% | same exact real-sampling command against `-ModelPath C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4` | 2026-08-17 02:00 PDT |
| Same production topology, fixed accepted length 3 | 171.263 tok/s | 171.263 tok/s | 0.000 | same client with launcher `-SimulateAcceptedLength 3` | 2026-08-16 22:40 PDT |
| Exact `199000+16` prompt processing, selective target NVFP4 M3 | 2838.980 tok/s record | 2654.502 mean / 2653.105 median / 2733.249 best | -184.478 / -6.498% mean | `.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600` | 2026-08-20 08:57 PDT |
| Exact `199000+16` generation, selective target NVFP4 M3 | 107.253 tok/s record | 96.682 mean / 91.627 median / 114.847 best | -10.571 / -9.856% mean; 14.850% CV | same exact command | 2026-08-20 08:57 PDT |
| Matched A2 exact `199000+16` prompt, selective target NVFP4 M3 | 2838.980 tok/s record | 2791.022 mean / 2789.956 median | -47.958 / -1.689% mean | same exact command, explicit seed `615388882` | 2026-08-20 09:23 PDT |
| M4 versus M3 warmed exact `199000+16` prompt | 2789.288 tok/s M3 | 2790.258 tok/s M4 | +0.970 / +0.035%; no material change | same exact command; only steps/width changed from `2/3` to `3/4` | 2026-08-20 09:23 PDT |
| M4 versus M3 warmed exact `199000+16` generation | 100.982 tok/s M3 | 98.957 tok/s M4 | -2.025 / -2.005%; noisy | same exact command; four warmed runs per arm | 2026-08-20 09:23 PDT |
| M4 versus M3 measured full-cycle projection | 139.841 tok/s M3 | 126.350 tok/s M4 | -13.491 / -9.647% | `bench_target_verify_width.py --width {3,4}` | 2026-08-20 09:23 PDT |
| FlashInfer paged-only exact-200K prompt | 2789.036 tok/s matched default | 2785.260 tok/s | -3.776 / -0.135% | exact `199000+512`, `SGLANG_FLASHINFER_USE_PAGED=1` | 2026-08-20 10:03 PDT |
| FlashInfer paged-only exact-200K long generation | 106.467 tok/s matched default | 104.117 tok/s | -2.350 / -2.207% | same exact `199000+512` pair | 2026-08-20 10:03 PDT |
| Exact-200K prompt, chunk 4096/5120/6144 | 2792.988 tok/s matched 4096 | 2892.671 / **2940.905 tok/s** | +3.569% / **+5.296%** | exact `199000+16`, two full warmups + three scored runs per arm | 2026-08-20 10:31 PDT |
| Exact-200K TTFT, chunk 4096/5120/6144 | 71.249895 s matched 4096 | 68.794554 / **67.666275 s** | -3.446% / **-5.030%** | same matched sweep | 2026-08-20 10:31 PDT |
| Selective exact-200K prompt, chunk 7680 | 2792.988 tok/s matched 4096 | 2997.744 mean / **3002.344 best** over 8 | +7.330% mean / **+7.497% best** | exact `199000+16`, two independent windows | 2026-08-20 11:14 PDT |
| Selective exact-200K long generation, chunk 7680 | 106.467 tok/s matched 4096 | 109.836 mean / **110.693 best** | +3.164% mean / **+3.970% best** | exact `199000+512`, two runs | 2026-08-20 10:02 PDT |
| Selective real sampled `6213/512`, chunk 7680 | 131.707 tok/s prior selective window | 138.537 / 139.885 two five-run means | +5.186% / +6.208% | sampled production profile, two independent windows | 2026-08-20 11:27 PDT |
| Base RadixArk real sampled `6213/512`, chunk 7680 vs 4096 | 121.027 tok/s matched 4096 | 121.054 tok/s combined 7680 | +0.027 / +0.022%; neutral | ten runs per geometry | 2026-08-20 11:36 PDT |
| Single-layer selected-row draft-extend logits | 16.058328 ms M3 cycle / 1.059 ms extend graph | 16.066558 ms / 1.061 ms | +0.008230 ms cycle; no-op | matched width-3 GPU traces | 2026-08-20 11:54 PDT |
| PERF-022 former exact `199000+16` record | 2838.980 prompt / 107.253 generation tok/s | **3016.444 / 112.355 tok/s** | +177.464 / +6.251% prompt; +5.102 / +4.757% generation | selective checkpoint, chunk 7680, direct Gemma output | 2026-08-20 12:22 PDT |
| Exact winner prompt, two independent windows | 2997.744 tok/s pre-change mean | **3014.751 / 3012.316 tok/s** | every sample >3000 | three plus five exact runs | 2026-08-20 12:22 PDT |
| Selective real sampled `6213/512`, direct Gemma output | 138.537 / 139.885 prior 7680 means | **144.535 / 138.621 tok/s** | combined 141.578 | two independent five-run windows | 2026-08-20 12:23 PDT |
| Production base sampled `6213/512`, direct Gemma output | 121.027 matched pre-change mean | **124.208 tok/s** | +3.181 / +2.628% | launcher-default base RadixArk | 2026-08-20 12:29 PDT |
| Exact `199000+16` capacity | 199016 total tokens | 199016 total tokens | preserved | `.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600` | 2026-08-16 22:40 PDT |
| Fresh current-source exact `199000+16` prompt baseline | 3016.444 tok/s record | 2871.358 mean / 2873.846 median / 2897.795 best | -145.086 / -4.810% mean | same exact command; selective checkpoint, chunk 7680, seed `615388882`, two full warmups then five cache-flushed scored requests | 2026-08-20 15:36 PDT |
| Fresh current-source exact `199000+16` legacy generation baseline | 112.355 tok/s record | 90.459 arithmetic mean / 88.746 aggregate / 111.926 best | -21.896 / -19.488% arithmetic mean; 15.633% CV | same exact command and launch | 2026-08-20 15:36 PDT |
| Adjacent pre-candidate exact `199000+16` prompt control A | 3016.444 tok/s record | 2867.286 mean / 2858.962 median / 2909.109 best | -149.158 / -4.945% mean | same launch, five cache-flushed `--skip-warmup` requests after long/sampled support probes | 2026-08-20 15:55 PDT |
| Adjacent pre-candidate exact `199000+512` support control | 3013.443 tok/s prior qualified mean | 2956.842 prompt / 108.738 legacy generation tok/s | -56.601 prompt / -0.945 generation | same launch, three exact cache-flushed requests; stable output digest | 2026-08-20 15:44 PDT |
| PERF-024 exact `199000+16` same-request record | 3016.444 prompt / 112.355 generation tok/s | **3048.086 / 112.499 tok/s** | **+31.642 / +1.049% prompt; +0.144 / +0.128% generation** | selective checkpoint, chunk 7680, target ordinary-EXTEND FP4 tactics | 2026-08-20 18:17 PDT |
| PERF-024 deterministic exact prompt window | 3016.444 tok/s record | **3047.309 tok/s five-run mean** | **+30.865 / +1.023%** | restored 20,928-byte selected cache; 110 target configs promoted into process cache | 2026-08-20 18:17 PDT |
| PERF-024 exact `199000+512` long generation | 109.683 tok/s prior qualified mean | **118.389 tok/s three-run mean** | **+8.706 / +7.937%** | persisted-cache relaunch; exact counts and stable digest | 2026-08-20 18:17 PDT |
| PERF-024 real sampled `6213/512` support | 117.940 tok/s fresh matched baseline | **126.252 tok/s five-run mean** | **+8.312 / +7.048%** | sampled production profile on the persisted-cache relaunch | 2026-08-20 18:17 PDT |
| Current-source exact `199000+16` prompt baseline | 3048.086 tok/s record | **2937.410 tok/s five-run mean** | -110.676 / -3.631% | selected cache, chunk 7680, seed 615388882, two exact warmups then five cache-flushed scores | 2026-08-20 19:21 PDT |
| Current-source exact `199000+16` generation baseline | 112.499 tok/s record | **93.539 tok/s five-run mean** | -18.960 / -16.854% | same exact requests; only 15 post-first-token intervals | 2026-08-20 19:21 PDT |
| Current-source exact `199000+16` TTFT baseline | 65.286869 s record | **67.749929 s five-run mean** | +2.463060 s / +3.772% slower | same exact requests | 2026-08-20 19:21 PDT |
| Current-source exact `199000+16` E2E baseline | 65.420204 s record | **67.910954 s five-run mean** | +2.490750 s / +3.807% slower | same exact requests | 2026-08-20 19:21 PDT |
| PERF-028 adjacent exact `199000+16` arm | 2967.386 prompt / 102.302 generation tok/s staged control | 2960.228 / 98.817 tok/s fused | -0.241% prompt; short generation inconclusive | three staged and five fused exact requests; identical digest; only 15 decode intervals | 2026-08-20 20:04 PDT |
| PERF-028 adjacent exact `199000+512` generation | 115.194 tok/s staged control | **116.583 tok/s fused** | **+1.388 / +1.205%** | three exact requests per arm; identical `199512` count and digest | 2026-08-20 20:04 PDT |
| PERF-027 first full exact `199000+16` arm | 2960.228 prompt tok/s PERF-028 fused arm | **2993.552 prompt tok/s** | **+33.324 / +1.126%** | five exact requests; TTFT improved 67.229581 -> 66.485194 s, but deterministic output changed | 2026-08-20 20:37 PDT |
| PERF-027 first full exact `199000+512` arm | 116.583 tok/s PERF-028 fused arm | 115.542 tok/s | -1.041 / -0.893% | three exact requests; changed deterministic trajectory, so not a valid decode attribution | 2026-08-20 20:42 PDT |
| PERF-027 repaired exact `199000+16` prompt | 2960.228 tok/s PERF-028 fused arm | **2987.275 tok/s** | **+27.047 / +0.914%** | five exact requests after two warmups; eager-only fusion restored the established digest | 2026-08-20 21:58 PDT |
| PERF-027 repaired exact `199000+16` TTFT | 67.229581 s PERF-028 fused arm | **66.622932 s** | **-0.606649 s / -0.902%** | same exact five-request window; all `199016`, `finish_reason=length` | 2026-08-20 21:58 PDT |
| PERF-027 repaired exact `199000+512` support | 2974.600 prompt / 116.583 generation tok/s PERF-028 fused arm | **3001.344 prompt / 115.225 generation tok/s** | +0.899% prompt; decode within current variance | three exact requests; restored `cac0c6...a2092` digest | 2026-08-20 22:03 PDT |
| PERF-029 compiled-semantics exact `199000+512` | 115.225 tok/s adjacent PERF-027 window | 116.192 tok/s | +0.966 / +0.839% client-observed | three exact requests, but 233 profiled device cycles retained a 16.045 ms median versus 16.058 ms control | 2026-08-20 22:30 PDT |

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

### 2026-08-17 00:42 PDT - PERF-009 asynchronous graph-tail admission

- Change: placed CUDA events at the actual raw draft, target-verify, and draft-extend graph boundaries. Event completion is queried asynchronously and written through a bounded background JSONL sink; the disabled path allocates no timing state.
- Benchmark evidence: two independent ordinary real-sampling windows emitted 512 tokens in 221 and 246 cycles, with mean emitted lengths **2.316742** and **2.081301**. The retained artifact contains **1,471** transition records. Target-to-draft-extend was repeatable with conservative p10 **0.658355 ms**; extend-to-next-draft p10 was 0.474054 ms and failed the strict p80-span repeatability rule; draft-to-target was about 0.09-0.10 ms.
- Correctness evidence: active worker was `EAGLEWorkerV2`, torch compile was enabled in mode `default`, both `/server_info` and startup logs recorded the same provenance, and three focused CPU tests passed.
- Decision: **close graph-tail work**. The best repeatable recoverable time is below the required **0.75 ms** admission threshold.
- Artifact: `benchmark/windows/profiles/m3_graph_gaps_20260817_0042.jsonl`, SHA-256 `4c7797ae1cf70694994b10fb2d9936543f3e415c1a5ecb2a96174dddf2b7c819`.

### 2026-08-17 00:46 PDT - PERF-010 branch-exact diagnostic and replay boundary

- Change: added opt-in post-transform p/q capture with exact child/parent IDs, depth, branch rank, token IDs, topology membership, branch-local presence/frequency/repetition counts, explicit transform order, active worker, and actual compile mode. Added immutable schema-v2 replay for current/aligned/irregular/calibrated/SWOR/confidence-gated/target-aware policies.
- Correctness evidence: the sequential small-vocabulary reference covers repeated-token branches and the active additive-then-sign-aware-repetition transform order. The live six-cycle JSONL capture preserved exact selected edges and full probability mass accounting. Runtime provenance resolved `EAGLEWorkerV2` with torch compile `default`.
- Coverage boundary: the artifact is explicitly `capture_scope=selected_tree`. Its observed current membership is replayable; descendant and alternate support becomes incomplete. Every counterfactual policy now fails closed until a declared complete proposal lattice is present.
- Frontier gate: an explicitly measured current membership defines the frontier. Every geometry candidate's conservative lower TPS must be strictly greater than the frontier's best-case upper TPS. Funding additionally requires complete lattice coverage and at least **215 TPS**. Family rejection requires a complete impossible target-aware upper at or below **200 TPS**.
- Artifact: `benchmark/windows/profiles/m3_pq_capture_20260817_0046.jsonl`, six records, SHA-256 `f87c0bf9b0d91c920dba3735823c05ee86cbdb3b30f724d9d4014a4ce629f588`.

### 2026-08-17 01:17 PDT - PERF-011 exact target-GEMM attribution and measured frontier

- Change: added `analyze_target_graph_gemms.py`. It groups replays by CUDA launch correlation, matches all **305** primary GEMMs per target replay against the Qwen3.5 projection contract, and fails model-role attribution closed on any count drift. Each launch and mathematical problem shape retains aggregate kernel time, all-stream wall coverage, terminal-stream serialized residency, and exclusive observed-wall exposure.
- M3 graph evidence: 61 graph-2 replays span **15.321986 ms mean / 14.660981 ms median**. Target-start-to-target-start cycles average **19.446434 ms** over 60 samples. Primary GEMMs are **13.086192 ms aggregate**, **12.360049 ms terminal-stream**, and **11.821001 ms exclusive observed wall** per replay.

| M3 target problem shape | Role | Aggregate ms/replay | Terminal-stream ms | Exclusive-wall ms |
|---|---|---:|---:|---:|
| NVFP4 `M3 x N34816 x K5120` | 64 MLP gate/up projections | 4.211372 | 4.211372 | 4.184174 |
| FP8 `M3 x N16384 x K5120` | 48 GDN qkvz projections | 2.851188 | 2.851188 | 1.675006 |
| NVFP4 `M3 x N5120 x K17408` | 64 MLP down projections | 2.328160 | 2.328160 | 2.293793 |
| FP8 `M3 x N5120 x K6144` | 48 GDN + 16 full-attention output projections | 1.483579 | 1.483579 | 1.483579 |
| FP8 `M3 x N8192 x K5120` | 16 full-attention qkv projections | 0.946352 | 0.946352 | 0.946352 |
| BF16 `M3 x N96 x K5120` | 48 GDN BA projections | 0.726143 | 0.000000 | 0.000081 |
| NVFP4 `M3 x N248320 x K5120` | lm-head | 0.539398 | 0.539398 | 0.539212 |

- Proposal execution evidence: graph 5 draft decode spans **1.216837 ms** and is led by five BF16 GEMVs at 0.515453 ms plus one NVFP4 GEMM at 0.441165 ms. Graph 8 draft extend spans **1.062720 ms** and is led by five BF16 GEMMs at 0.561803 ms plus one NVFP4 GEMM at 0.432317 ms. Full cycle minus target span is about **4.124 ms**, including proposal graphs and scheduling transitions.
- Width frontier: trace-local M3 emitted 2.133333/cycle for **109.703 projected TPS**. Its impossible depth-two ceiling is **154.270 TPS** at mean cost and **167.480 TPS** at the best observed cycle. Post-change M8, corrected M12, and M16 depth-four best-sample impossible ceilings are **185.782**, **179.547**, and **166.666 TPS**. All four measured geometries are rejected before proposal quality.
- Decision: no new topology is funded. Target work begins with the exposed NVFP4 gate/up and down shapes; the BF16 BA path is already hidden at M3. Geometry work waits for complete lattice capture and must clear both the measured frontier and the 215-TPS conservative floor.
- Artifact: M3 trace SHA-256 `01a113fa2e8aed1bee57a15fd3b02a718afafd712504722dd295233a1a694e92`; generated attribution SHA-256 `fb27a0ab703711a4629e1bff0d75f02d4fa33049a79d5a69eab60d72a8333d06`.

### 2026-08-17 01:26 PDT - PERF-012 external vLLM MTP-3/TurboQuant architecture

- Evidence: `MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090` serves the same RadixArk checkpoint on one RTX 5090 through vLLM 0.27.1 and claims approximately **160 tok/s** single-stream at full 262K context.
- Architecture: three speculative MTP tokens, four-row K+1 verification, TurboQuant 4-bit KV pinned to 5.5 GiB, Flash Attention v2, one sequence, full CUDA graphs, and a backport of vLLM PR #40914 that routes uniform K+1 verify through the TurboQuant decode kernel with GPU-only synthetic metadata.
- Ceiling implication: at our 19.446 ms measured cycle, a four-token maximum is **205.693 TPS**, while the current three-token maximum is 154.270 TPS. The external result escapes the current topology's mathematical ceiling before any target GEMM improvement.
- Qualification gap: the external repository publishes no prompt length, output length, raw samples, sampling parameters, acceptance, or cycle data. Its examples use temperature zero and thinking disabled. The 160 claim is an architecture lead rather than a production comparison.
- Decision: matched vLLM reproduction under exact `6213/512` real sampling becomes the next gate. Porting or selecting this lane requires the 200/215 frontier plus reasoning/tool/200K behavior qualification.

### 2026-08-17 01:36 PDT - PERF-013 selective target NVFP4 admission probe

- Compared production-shaped static-FP8 cuBLAS BMM against CUTLASS NVFP4 for QKVZ `3x16384x5120`, output `3x5120x6144`, and full-attention QKV `3x8192x5120`. Both sides include activation quantization and replay family graphs with one distinct weight per production layer.
- Two paired family-streaming windows projected **1.976456 ms** and **1.865227 ms** overlap-adjusted full-cycle savings. The second assigned 0.724913 ms to QKVZ, 0.819841 ms to output projections, and 0.320473 ms to full-attention QKV.
- At the lower 1.865227 ms projection, M3 cost is 17.580773 ms: **170.641 TPS** at the three-token perfect ceiling and **227.521 TPS** at a four-token K+1 perfect ceiling.
- Decision: fund selective FP8-to-NVFP4 checkpoint construction. It clears the 0.75 ms implementation gate and the 215-TPS geometry floor when paired with K+1. The exact-shape probe remains as diagnostic infrastructure.

### 2026-08-17 02:00 PDT - PERF-014 selective target NVFP4 admission window 1

- Change: loaded the derived `Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4` checkpoint (208 target attention projection bases converted FP8 -> NVFP4) through the unchanged production launcher with only `-ModelPath` changed.
- Benchmark evidence: five consecutive real samples `130.403, 134.384, 130.824, 136.749, 126.173 tok/s`, mean **131.707**, median **130.824**, versus the 124.775 matched control and 122.712 qualified baseline. Five acceptance probes `2.216450, 2.275556, 2.178723, 2.226087, 2.188034`, mean **2.216970** over 1,155 cycles, aggregate histogram `[308, 292, 555]`. Device cycle previously measured **17.314950 ms** (from 19.446434 ms, -10.96%).
- Correctness evidence: tool gate passed (one `multiply({"a":37,"b":19})` call, `finish_reason=tool_calls`), preserved coherent reasoning, exact 512-token completions, `/model_info` language-only, all three graphs captured.
- Decision: admission window 1 passes with the largest measured single-window gain so far. Width-three remains capped near 173.260 TPS; K+1 geometry is required for 200. Remaining gates: exact `199000+16` capacity, second independent window, OpenCode2 integration, unsimulated relaunch, and removal of the temporary loader diagnostic.
- Commit: uncommitted experiment.

### 2026-08-20 08:57 PDT - PERF-015 current exact-200K M3 baseline

- Change: measurement only from clean `main` at
  `2eddaf4e8fd13911be3937df0d1f5f40583e4b4d`. Launched the selective
  target-NVFP4 checkpoint with explicit seed `615388882` and otherwise
  unchanged production settings: real 200K pools, 4096-token prefill chunks,
  M3 width, ordinary rejection sampling, draft top-k 20, FlashInfer prefill,
  TRT-LLM MHA/XQA target and draft decode, FP8 draft KV, ReplaySSM, lazy
  extra-buffer Mamba state, FP4-only autotuning, and all tree/device-cycle
  experiments inactive.
- Environment: native Windows RTX 5090, driver `610.88`, Python `3.13.14`,
  PyTorch `2.13.0+cu130`, CUDA runtime `13.0`, Triton `3.7.1`, and FlashInfer
  `0.6.17`. Graph capture completed for target verify, draft decode, and draft
  extend in `31.04`, `1.38`, and `0.88` seconds. The measured runs held
  2.962-3.015 GHz SM and 13.801 GHz memory clocks at 496.79-525.90 W and
  60-68 C. Visible WDDM clients included Edge WebView, Windows shell/display,
  iCloud, PC Manager, and OpenCode2.
- Warmup/cache policy: run one used the benchmark's exact-shape internal warmup;
  the following four skipped the redundant internal warmup on the already
  loaded/captured server. Every measured request was cache-flushed, and the
  server reported `cached-token: 0` on every long-prefill chunk.
- Benchmark evidence: prompt samples were `2603.510, 2610.132, 2733.249,
  2672.513, 2653.105 tok/s`, mean **2654.502**, median **2653.105**, standard
  deviation **52.670**, and CV **1.984%**. Generation samples were `91.627,
  108.879, 114.847, 83.791, 84.268 tok/s`, mean **96.682**, median **91.627**,
  standard deviation **14.358**, and CV **14.850%**. TTFT samples were
  `76.435263, 76.241344, 72.807133, 74.461748, 75.006454 s`.
- Correctness evidence: all five requests completed exactly `199000+16`,
  returned `finish_reason=length`, kept thinking enabled, and produced the
  identical digest
  `9a0e20749e2930a697fefdd3bdd7863a067abe4d9860e6d1e7d9b80a62668b37`.
- Decision: accept this as the current reproducible M3 baseline, not a
  replacement for the historical record. The historical `2838.980/107.253`
  hit was not reproduced. A 16-token request has only 15 post-first-token
  decode intervals, so generation is quantized by verification-cycle count
  and shows far more variance than the 2.6% target gap. Candidate decisions
  require matched interleaved controls and repeated exact completions; a
  single favorable generation sample is insufficient.
- Artifacts:
  `baseline-m3-selective-199k.jsonl`,
  `baseline-m3-selective-199k-environment.jsonl`, and the server logs under
  the active Copilot session-state `files` directory.

### 2026-08-20 09:23 PDT - PERF-016 selective-checkpoint M4/K+1 retest

- Change: changed only the selective-checkpoint speculative shape from two
  steps / three target rows to three steps / four target rows. Both arms used
  seed `615388882`, real 200K pools, the same checkpoint and backends, ordinary
  rejection sampling, draft top-k 20, and complete target/draft/extend graph
  capture. A fresh M3 server followed the M4 server for an A-B-A comparison.
- Device evidence: M4 accepted **2.327273** tokens/cycle over 55 cycles versus
  M3 **2.245614** over 57 cycles, a 3.636% gain. M4 full-cycle cost rose from
  **16.058328 ms** to **18.419190 ms**, or 14.702%. The resulting measured
  projection fell from **139.841** to **126.350 tok/s** (-9.647%). M4's
  perfect-four ceiling is **217.165 tok/s**, barely above the 215-TPS funding
  floor and dependent on unattained full acceptance.
- Exact-200K evidence: M4 samples were prompt `2653.695, 2792.130, 2788.118,
  2790.292, 2790.491` and generation `117.545, 91.572, 106.251, 92.061,
  105.943 tok/s`. The matched M3 A2 samples were prompt `2797.957, 2789.956,
  2787.745, 2787.968, 2791.484` and generation `99.306, 115.665, 100.016,
  100.035, 88.214 tok/s`. Excluding each arm's first internally warmed run,
  prompt means were **2790.258 M4** and **2789.288 M3** (+0.035%), while
  generation means were **98.957 M4** and **100.982 M3** (-2.005%). Both
  generation arms had 8-11% CV and peaks above 110, so neither peak is a
  decision-capable win.
- Correctness evidence: all ten exact requests completed `199000+16`, returned
  `finish_reason=length`, kept thinking enabled, and produced the same digest
  `9a0e20749e2930a697fefdd3bdd7863a067abe4d9860e6d1e7d9b80a62668b37`.
  Both profiles resolved `EAGLEWorkerV2`, torch compile `default`, the intended
  chain topology, and exact width.
- Decision: reject plain M4/K+1 under the current selected checkpoint and
  execution path. Its acceptance gain does not repay the extra draft and
  target work, prompt throughput is unchanged, and exact generation does not
  improve outside noise. The external vLLM TurboQuant/full-graph K+1
  architecture remains a distinct information gate rather than evidence for
  this SGLang shape.
- Artifacts: tracked M3/M4 traces and manifests under
  `benchmark/windows/profiles/target_width_m{3,4}-20260820-*`; raw exact
  results, environment snapshots, and logs remain in the active session-state
  `files` directory.

### 2026-08-20 09:35 PDT - PERF-017 exact-benchmark validity telemetry

- Change: retained the user-selected prompt/generation formulas while making
  benchmark validity explicit. `bench_openai_stream.py` now hashes both output
  channels in stream order, emits separate reasoning/content hashes, records
  nonempty SSE delta counts, first/max delta size, per-channel fragment counts,
  and response time after the final output delta. It rejects prompt,
  completion, total-token, and finish-reason mismatches. `--warmup-runs`
  records fixed repeated warmups while `--skip-warmup` remains compatible.
- Benchmark evidence: a live `256+16` request on the restored M3 server
  completed exact `272`, `finish_reason=length`, with three nonempty reasoning
  deltas, first/max delta sizes `2/39` characters, and 0.000168 seconds after
  the final output delta. The new full-output and reasoning hashes matched;
  the empty content hash was explicit.
- Correctness evidence: the new CPU-only unit suite passed **3 tests plus 4
  subtests**, covering dual-channel deltas, strict result validation, and an
  impossible prompt target. Python compilation, CLI parsing, and
  `git diff --check` passed.
- Decision: retain as measurement infrastructure. The headline metrics remain
  client-observed SSE timings, not pure device prefill/decode timers. Future
  1-3% claims must retain the new fragment/trailing telemetry and exact-count
  validity fields.

### 2026-08-20 10:03 PDT - PERF-018 FlashInfer paged-only prefill

- Change: set only `SGLANG_FLASHINFER_USE_PAGED=1` for the selective M3
  checkpoint. This writes the current chunk into KV first and runs one paged
  attention over prefix plus current tokens instead of separate ragged-current
  and paged-prefix calls followed by state merge. The process resolved the
  environment switch as true; all launcher arguments and pools remained
  matched.
- Exact-200K prompt evidence: after two full-shape warmups, three
  `199000+16` prompt samples were `2790.384, 2782.369, 2781.207 tok/s`, mean
  **2784.653**. Two `199000+512` samples were `2786.844, 2783.676`, mean
  **2785.260**. The restored default control measured `2789.332, 2788.740`,
  mean **2789.036**. Paged-only therefore changed prompt throughput by
  **-0.135%** in the long matched pair.
- Generation evidence: paged-only's three 16-token samples clustered at
  `114.675, 114.644, 114.877 tok/s`, but the result did not survive the longer
  validation. Exact `199000+512` generation was `104.514, 103.720`, mean
  **104.117**, versus restored-control `108.022, 104.912`, mean **106.467**
  (-2.207%). Short acceptance moved only `1.961686 -> 1.976834` tokens/cycle.
- Correctness evidence: every request completed its exact token count and
  `finish_reason=length`; fragment/trailing telemetry was valid. Paged-only
  deterministically changed the output digest (`35dc...dabd1` for 16 tokens,
  `d2f8...cdf73` for 512) from the default path (`9a0e...8b37` and
  `9ca9...25ee8`), consistent with attention-order numerical differences.
- Decision: reject paged-only for this workload. It provides no prompt gain,
  loses on the longer generation comparison, and changes the deterministic
  trajectory. Keep the default ragged-current plus paged-prefix merge.

### 2026-08-20 10:31 PDT - PERF-019 chunk-size sweep, 4096/5120/6144

- Change: changed only `chunked_prefill_size` on fresh selective M3 servers,
  using seed `615388882`, two full exact-shape warmups, and three scored exact
  `199000+16` requests per arm. All other launcher values, pools, backends, and
  graph settings remained matched.
- Benchmark evidence: chunk 4096 prompt samples were `2795.255, 2790.685,
  2793.024 tok/s`, mean **2792.988**, with mean TTFT **71.249895 s**. Chunk
  5120 produced `2894.440, 2892.438, 2891.136`, mean **2892.671**, TTFT
  **68.794554 s**. Chunk 6144 produced `2943.285, 2939.119, 2940.310`, mean
  **2940.905**, TTFT **67.666275 s**. The 6144 candidate improves matched
  prompt throughput **5.296%**, beats the historical 2838.980 prompt record by
  3.590%, and remains 1.970% below the active 3000 target.
- Correctness evidence: all nine scored requests completed exact `199016`,
  returned `finish_reason=length`, and passed strict token/fragment telemetry.
  Each chunk geometry selected a stable deterministic digest: `9a0e...8b37`
  for 4096, `a6bc...19ec` for 5120, and `3e01...2417` for 6144.
- Decision: retain 6144 as the leading prompt candidate, not yet as a launcher
  default. Complete the nearby 6656/7168 sweep, then run long generation,
  reasoning/tool, capacity/headroom, and restored-control gates on the final
  winner.

### 2026-08-20 11:44 PDT - PERF-020 selective long-context chunk profile

- Change: completed the chunk refinement at 6656, 7168, 7680, and 7808. The
  production launcher continues to default to 4096; the winner is invoked
  explicitly with the selective checkpoint and `-ChunkedPrefillSize 7680`.
- Sweep evidence: exact-200K prompt means were **2965.411** at 6656,
  **2980.383** at 7168, **2998.342** in the first 7680 window, **2997.386** in
  the second 7680 window, and **2909.350** at 7808. The 7808 cliff closes
  upward refinement without reopening the rejected 8192 branch.
- Independent prompt record: eight 7680 exact `199000+16` samples averaged
  **2997.744 tok/s**, with best **3002.344**, best TTFT **66.281538 s**, and
  best E2E **66.434400 s**. Every request completed exact `199016`; all eight
  retained the established 4096 digest `9a0e...8b37`.
- Long decode: exact `199000+512` samples reached
  `3004.324/110.693` and `2999.159/108.978` prompt/generation tok/s, mean
  **3001.742/109.836**. Both completed exact `199512` with a stable digest.
- Production-profile evidence: selective 7680 real sampled `6213/512` windows
  averaged **138.537** and **139.885 tok/s**. Five acceptance probes averaged
  **2.245332** tokens/cycle. Arithmetic returned `703`, the tool gate emitted
  exactly one `multiply({"a":37,"b":19})`, image/audio remained false, and
  post-flush headroom was 4.63 GiB.
- Default-safety evidence: base RadixArk sampled generation was neutral across
  ten-run geometry windows (**121.054** at 7680 versus **121.027 tok/s** at
  4096), but exact base `199000+16` at 7680 fell to **2226.770 prompt tok/s**
  and only 200 MiB free before follow-up probes. Capacity and semantic gates
  passed, but prompt performance and operating margin did not.
- Decision: retain 7680 as an explicit selective-checkpoint long-context
  profile and restore the global launcher default to 4096. This establishes an
  independent prompt milestone, not the combined exact-16 winner; generation
  work continues.

### 2026-08-20 11:54 PDT - PERF-021 selected-row draft-extend logits

- Change: ported the multi-layer EAGLE selected-row `lm_head` pruning contract
  to the single-layer graph behind gathered-buffer, standalone, and
  device-resident-cycle guards. The graph kept full hidden rows and computed
  vocabulary logits only for each request's selected accepted row. The patch
  and its new white-box tests were removed after measurement.
- Benchmark evidence: draft-extend graph span was **1.061 ms** versus the
  matched unpruned trace's **1.059 ms**. Full M3 cycle was **16.066558 ms**
  versus **16.058328 ms** control. Kernel count was 29 versus 28. The
  candidate's 147.534 projected tok/s came entirely from a favorable
  **2.370370** acceptance sample, not execution savings.
- Correctness evidence: graph capture completed; exact `6213+128` profiling
  completed in 54 verification cycles. Focused runner tests passed **6** before
  the full-model gate. The candidate trace hash is
  `3d431c6142df0037fcf2180729d65ca1a6f1626b070083832e2f92ca693230cc`.
- Decision: reject and remove. The NVFP4 vocabulary projection is
  weight-bandwidth-bound at one to three rows, so pruning rows saves about
  0.02 GiB of graph residency but no device time.

### 2026-08-20 12:29 PDT - PERF-022 direct native Gemma residual-norm output

- Change: on native Windows, preserve `residual.add_(x)` and write the existing
  bit-exact JIT Gemma RMSNorm result directly into `x`. This removes the
  temporary normalized tensor and subsequent `x.copy_()` without changing
  arithmetic, dtype, dispatch, or output ownership.
- Isolated evidence: Qwen hidden-size 5120 measured **38.731 -> 29.254 us** at
  one row and **37.578 -> 29.184 us** at three rows, reductions of 24.47% and
  22.34%. Input and residual were bit-exact. Four targeted Qwen Gemma tests
  passed; the native hot-path smoke retained fullgraph parity.
- Exact scoreboard evidence: first independent window was prompt
  `3016.444, 3013.834, 3013.975` and generation `112.355, 97.506, 112.534`.
  Second window was prompt `3014.657, 3009.496, 3012.204, 3013.736, 3011.489`
  and generation `96.531, 86.114, 98.100, 112.012, 79.442`. All eight
  completed exact `199016` with the established digest. The new overall record
  is **3016.444/112.355 tok/s**, TTFT **65.971714 s**, E2E **66.105219 s**.
- Supporting evidence: exact `199000+512` averaged **3013.443 prompt /
  109.683 generation tok/s**. Selective sampled `6213/512` windows averaged
  **144.535** and **138.621 tok/s** with five-probe acceptance **2.249107**.
  Arithmetic, tools, language-only surface, and standalone OpenCode2 `READY`
  passed.
- Production-default evidence: base RadixArk exact `199000+16` completed at
  **2643.254 prompt / 101.980 generation tok/s** with 698 MiB free, recovering
  to 1.91 GiB after probes/flush. Its five-run sampled mean was **124.208
  tok/s**. Arithmetic, tools, model surface, and all three graph captures
  passed.
- Decision: retain and promote. This is a bit-exact Windows hot-path
  simplification and the combined selective chunk-7680 profile clears the
  complete 3000/110 milestone.

### 2026-08-20 15:36 PDT - PERF-023 fresh current-source exact baseline

- Change: measurement only from `main` at
  `adf3a620ef64e11aea6159643f560c790327c57f`, with the pre-existing
  user-owned `BENCHMARK.md` edit and `HANDOFF.md` deletion left untouched.
  Launched the selective `AttnNVFP4` checkpoint with chunk 7680 and server
  seed `615388882`; all tree, SWOR, adaptive, simulation, and
  device-resident-cycle controls remained inactive.
- Runtime evidence: the listener was PID `41904` under
  `44500 -> 37588 -> 16276 -> 41904`. Resolved arguments retained exact
  200K context and pools, page size 64, one request, FlashInfer prefill and
  sampling, TRT-LLM MHA target/draft decode, M3 linear rejection sampling,
  draft top-k 20, FP8 draft KV, FP32 ReplaySSM state, torch compile
  `default`, and the 128 MiB workspace. Target, draft-decode, and
  draft-extend graph captures completed in **33.49, 1.43, and 0.88 s** with
  4.29 GiB reported after capture. `/health` returned 200 and `/model_info`
  reported image/audio understanding false.
- Environment: native Windows RTX 5090, driver `610.88`, Python `3.13.14`,
  PyTorch `2.13.0+cu130`, CUDA runtime `13.0`, toolkit `13.3.33`, Triton
  `3.7.1`, and FlashInfer `0.6.17`. WDDM clients included Chrome, Edge
  WebView, iCloud, Windows shell/display processes, and an unrelated Python
  process. Scored-run snapshots reached P1, 2.947-2.977 GHz SM,
  13.801 GHz memory, 59-69 C, and 515-559 W. NVIDIA reported accumulated
  software power-capping time, so this window is not an uncontended
  replacement for the historical record.
- Warmup/cache policy: two complete exact-shape warmups preceded the first
  score. All five scored requests were cache-flushed and subsequent
  invocations used `--skip-warmup`.
- Prompt evidence: `2897.795, 2875.047, 2837.904, 2873.846, 2872.198
  tok/s`; mean **2871.358**, median **2873.846**, standard deviation
  **21.439**, CV **0.747%**, and aggregate fixed-token rate **2871.229**.
  TTFT was `68.672916, 69.216270, 70.122180, 69.245186, 69.284914 s`.
- Legacy generation evidence: `90.816, 85.650, 91.199, 111.926, 72.704
  tok/s`; arithmetic mean **90.459**, aggregate
  `75 / sum(E2E-TTFT)` rate **88.746**, standard deviation **14.141**, and
  CV **15.633%**. E2E was `68.838085, 69.391402, 70.286654, 69.379203,
  69.491229 s`. Nonempty SSE fragment counts varied `4,4,4,4,3`, reinforcing
  that the 16-token legacy generation metric is not a stable small-effect
  estimator.
- Correctness evidence: every request completed exact `199000+16`, returned
  `finish_reason=length`, kept thinking enabled, and retained output digest
  `9a0e20749e2930a697fefdd3bdd7863a067abe4d9860e6d1e7d9b80a62668b37`.
- Decision: retain **2871.358 prompt / 90.459 legacy generation tok/s** as
  the immediate current-environment baseline. It does not supersede the
  **3016.444/112.355** record. Candidate decisions require fresh matched
  controls in the same launch block; recovering or explaining the 4.810%
  prompt gap is part of the active optimization branch.
- Artifact:
  `C:\Users\Daniel\.copilot\session-state\df1c744a-8e2f-4823-bd37-18b450ed10d1\files\baseline-200k-20260820-1527.log`.

### 2026-08-20 15:55 PDT - PERF-023 supporting controls and exact-client guard

- Long-generation support: three cache-flushed exact `199000+512` requests
  measured prompt `2983.007, 2942.383, 2945.135 tok/s` and legacy generation
  `107.385, 107.491, 111.337 tok/s`. Means were **2956.842 prompt /
  108.738 generation tok/s**, prompt CV was 0.768%, generation CV was 2.071%,
  and aggregate generation was **108.707 tok/s**. All three completed exact
  `199512` with digest
  `1e90cc8fad3e1b1802db4cdc2af762790bcd392c062a14f0afc334df8b5e97f9`.
- Real sampled support: five `6213/512` requests measured
  `122.714, 122.917, 111.596, 119.056, 113.418 tok/s`, mean **117.940** and
  CV 4.436%. Five independent native counter probes averaged accepted length
  **2.155292**, acceptance rate **0.577233**, and **237.6** target
  verifications.
- Adjacent control A: after those probes, five cache-flushed exact
  `199000+16` requests measured prompt
  `2909.109, 2827.344, 2908.788, 2832.229, 2858.962 tok/s`, mean
  **2867.286**, median **2858.962**, CV 1.391%, and aggregate
  **2866.843**. Legacy generation was
  `92.718, 92.717, 112.714, 103.474, 107.168 tok/s`, mean **101.758**,
  CV 8.731%, and aggregate **101.136**. Exact counts, `finish_reason=length`,
  and the established digest held.
- Interpretation: prompt throughput moved materially between the three-run
  `+512` window and the immediately following `+16` window despite identical
  prompt shape and cache flushing. Candidate attribution therefore requires
  A-B-A windows and cannot use the historical record or either standalone
  baseline as its sole control.
- Measurement hardening: `bench_openai_stream.py` now rejects any calibrated
  prompt below the requested count before cache flush, warmup, or measurement,
  and validates server usage against the requested count. A CPU regression
  test proves an inexact `198999` calibration sends no request.
- Artifacts:
  `C:\Users\Daniel\.copilot\session-state\df1c744a-8e2f-4823-bd37-18b450ed10d1\files\baseline-support-20260820-1542.log`
  and
  `C:\Users\Daniel\.copilot\session-state\df1c744a-8e2f-4823-bd37-18b450ed10d1\files\control-a-exact16-20260820-1554.log`.

### 2026-08-20 18:17 PDT - PERF-024 large-EXTEND FlashInfer FP4 tactics

- Change: allowed the existing opt-in FlashInfer EXTEND autotuner to run one
  ordinary 16,384-token EXTEND forward on a speculative target worker while
  keeping draft workers and ordinary speculative dummy callers on their prior
  paths. FlashInfer file hits exercised by that pass are promoted into the
  runner-keyed process cache so later draft autotune contexts cannot discard
  them.
- Causality: the initial candidate beat the record, and an independent retune
  produced a distinct 20,928-byte cache with SHA-256
  `8219484FA86EBB0E6DDA54F2D15447DBC502EBCEA9007B3E1BB917B9001F9ADF`.
  Cache-only and dummy-only controls both returned to about 3009 prompt tok/s
  and the baseline output digest. The gain therefore comes from FP4 tactics,
  not stale state or the extra forward.
- Record evidence: the independent exact prompt window was
  `3051.345, 3048.538, 3048.086, 3042.488, 3044.105 tok/s`, mean
  **3046.912**. The third request set the qualified same-request record at
  **3048.086 prompt / 112.499 generation tok/s**, TTFT **65.286869 s**,
  and E2E **65.420204 s**.
- Persistence evidence: a clean relaunch promoted exactly 110 selected target
  configs without re-profiling. Its exact prompt window was
  `3050.570, 3048.607, 3044.288, 3045.422, 3047.659 tok/s`, mean
  **3047.309**. Three exact `199000+512` requests averaged
  **3047.754 prompt / 118.389 generation tok/s**. Five real sampled requests
  averaged **126.252 tok/s**, and five native probes averaged
  **2.217256** accepted tokens per verify.
- Correctness/capacity: every exact request completed its requested
  `199016` or `199512` tokens with stable selected-tactic digests. Preserved
  reasoning returned `703`; exactly one multiply tool call parsed; image and
  audio remained false; standalone OpenCode2 returned visible `READY`; and
  cache flush left 5,386 MiB free.
- Rejected branch: profiling afresh on every launch kept exact prompt mean at
  **3043.747 tok/s** but selected tactics whose long generation averaged only
  **101.162 tok/s**. Fresh profiling remains diagnostic, not the promotion
  policy.
- Decision: retain as an expert opt-in for the selective chunk-7680 profile.
  The base RadixArk/chunk-4096 launcher defaults remain unchanged. A hostile
  review also moved large-buffer allocation inside the OOM fallback and added
  CPU coverage for allocation failure and exception-safe method restoration.

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
| PERF-003 | Apply exact branch-local presence/frequency/repetition state to SWOR p and q. | sampling state, topology metadata, draft graph buffers, target verifier | Diagnostic implemented | Live selected-tree p/q capture is exact for the observed membership; counterfactual policy coverage fails closed. |
| PERF-004 | Attribute target/composite graph time by kernel family, exact M/N/K, and graph ID before another kernel rewrite. | trace analyzer and Qwen3.5 target/draft hot paths | Complete for M3/M8/M12/M16 | All 305 primary target GEMMs/replay match exactly; M3 gate/up and down expose 6.539 ms on the terminal stream. |
| PERF-005 | Extend the device-resident cycle to exact linear rejection sampling. | proposal sampling, exact-q buffers, verification/extend bridge | Closed for throughput; retained opt-in | Dense races reached 122.576 tok/s; explicit-seed categorical reached 120.075 versus 124.775 control despite higher acceptance. Composite cycle cost remained 1.7% slower. |
| PERF-006 | Improve proposal quality with a distinct trained/calibrated proposal mechanism. | MTP adapter/training, standalone draft, or device-side mixture oracle | Survey | RadixArk and Gittensor embedded MTP tensors are byte-identical. Temperature/support calibration is flat. Any training path needs held-out behavior evidence. |
| PERF-007 | Reduce the exposed target GEMM critical path. | Qwen3.5 MLP gate/up/down and FP8 qkvz/output projections | Measured: admission window 1 passed | Derived `AttnNVFP4` checkpoint cut the cycle 10.96% and raised real TPS to 131.707 mean. Remaining: capacity, second window, OpenCode2, relaunch. |
| PERF-014 | Raise the emitted-token path length from three to four (K+1) on the selective checkpoint. | launcher speculative shape only (three steps / four rows), EAGLE worker/graph code path | Rejected | Acceptance rose 3.636% while measured cycle cost rose 14.702%; projected TPS fell 139.841 -> 126.350 and matched exact-200K generation did not improve outside noise. |
| PERF-017 | Make exact-200K prompt/generation comparisons fail closed and expose SSE timing boundaries. | `bench_openai_stream.py` and CPU unit tests | Retained | Headline formulas unchanged; exact calibration now fails before a request, and exact token/finish validation, complete output hashes, fragment coalescing, trailing time, and repeated warmup metadata accompany every run. |
| PERF-018 | Replace ragged-current plus paged-prefix merge with one paged FlashInfer prefill. | Existing `SGLANG_FLASHINFER_USE_PAGED` path | Rejected | Exact-200K prompt changed -0.135%; 512-token generation changed -2.207% and deterministic output changed. |
| PERF-019 | Increase prefill chunks below the rejected 8192 geometry. | Selective-checkpoint chunk sweep through 7808 | Retained as explicit 7680 profile | Eight exact-200K prompt samples averaged 2997.744 with 3002.344 best; long decode reached 110.693. Global default rejected on base checkpoint. |
| PERF-021 | Run single-layer draft-extend `lm_head` only on each selected accepted row. | EAGLE worker/graph runner using the existing multi-layer selection contract | Rejected | Draft-extend span changed 1.059 -> 1.061 ms and full cycle 16.058328 -> 16.066558 ms; memory fell but runtime did not. |
| PERF-022 | Remove the temporary/copy from native-Windows Gemma residual normalization. | Windows `GemmaRMSNorm` dispatch using existing JIT output buffer | Retained | Bit-exact; 22-24% isolated reduction; former exact `199000+16` record **3016.444/112.355**, independent confirmation **3013.736/112.012**. |
| PERF-023 | Re-establish the current-source exact-200K baseline before changing code. | Selective checkpoint, chunk 7680, exact benchmark and live environment | Complete | Five exact scores averaged **2871.358/90.459** with exact digest; current environment did not reproduce the historical record. |
| PERF-024 | Autotune target ordinary-EXTEND FP4 tactics at the real 7680/7000 prefill shapes without losing them to later draft contexts. | FlashInfer autotune runner and speculative target prefill | Retained expert opt-in; qualified record | Same-request record **3048.086/112.499**; persisted-cache exact prompt mean **3047.309** and long-generation mean **118.389**. Defaults remain unchanged. |
| PERF-025 | Evaluate the already-implemented FlashInfer TRT-LLM dense FP4 backend on native-Windows SM120. | `ModelOptFp4LinearMethod`, FP4 backend selector, launcher | Blocked by installed backend | The real layer-path test reaches FlashInfer and fails all three shapes with `mm_fp4 does not support backend 'trtllm' with capability 120`; see `PERF-F042`. |
| PERF-026 | Specialize greedy EAGLE draft proposals and retain a sparse exact sampled p/q path. | EAGLE draft graphs, proposal buffers, rejection sampling | Survey | Temperature-zero target verification is greedy while the draft still samples stochastic top-k 20. Any change must preserve exact q(X), RNG, graph replay, and asynchronous output lifetimes. |
| PERF-027 | Fuse Qwen SwiGLU output directly into byte-identical NVFP4 activation/scales for `down_proj`. | Native CUDA activation/quant producer and FP4 linear tuple input | Retained eager-prefill win | Exact across every finite BF16 gate value, production shapes, mutable graphs, and the ModelOpt consumer. Eager-only selection preserves the former compiled M3 function and restored both deterministic digests; exact short prompt improved 0.914% versus PERF-028. |
| PERF-028 | Fuse the native-Windows BF16 residual add into the bit-exact Gemma RMSNorm direct-output kernel. | JIT CUDA half-width RMSNorm and Windows Gemma dispatch | Retained additive decode win | Exact at M1/M3/M7000/M7680 and under mutable CUDA-graph replay. Stable M1/M3 kernel-only A-B-A improved about `16.5 -> 9.5 us`; adjacent exact `199000+512` generation improved `115.194 -> 116.583 tok/s` (+1.205%) with identical output. Prefill was neutral. |
| PERF-029 | Match the compiled M3 SiLU arithmetic while fusing activation and NVFP4 packing. | Separate fast-math native producer inside target `torch.compile` | Rejected; graph-neutral | Byte-exact and 70.848 -> 25.152 us in isolated launch timing, but full-cycle median was 16.045 ms versus the 16.058 ms control and long generation remained inside variance. The experiment was removed. |
| PERF-030 | Tune FlashInfer paged-prefix fixed split size without changing ragged-current attention. | Existing prefill split descriptor outside deterministic mode | Rejected; workspace overflow | Split sizes 4096 and 8192 each requested 2,264,924,160 bytes from the qualified 128 MiB workspace on the first exact warmup. No score was produced; the opt-in was removed. |
| PERF-031 | Remove target-verify GDN Q/K/V split materialization. | Post-convolution QKV handoff into ReplaySSM | Closed by source gate | Qwen3.8 `qkv_dim=10240` already exceeds the 8192 materialization threshold and uses zero-copy strided aliases accepted by ReplaySSM. There is no split kernel to remove. |
| PERF-032 | Coalesce the final 7680+7000 prefill pair into one 14680-token forward. | Scheduler tail geometry and Mamba branching checkpoint | Rejected | Exact completion fell to 1917.509 prompt tok/s and 103.780505 s TTFT with a changed digest. The larger ragged-current pass erased the saved dispatch. |
| PERF-033 | Fuse full-attention sigmoid gating directly into the NVFP4 `o_proj` tuple. | PDL-safe native gate/quant producer | Rejected at isolated admission | Exact and 37% faster at M7680, but M3 saved only 0.427 us/layer (0.007 ms/replay) and total exact-prefill projection was about 21 ms. No model wiring was retained. |
| PERF-034 | Tune global KV page size for the paged-prefix attention wall. | Page sizes 128 and 32 | Rejected | Page 128 floored pools to 199,936 tokens. Page 32 retained exact pools but does not reach prefill's page-size-1 token-index wrapper and reduced long generation to 112.576 tok/s. |
| PERF-035 | Use FlashInfer FP16 QK reduction only for ordinary paged prefill. | Paged-prefix plan precision mode | Rejected as noise | Initial server A-B moved +0.679%, but the exact 25-prefix ladder was 163.705 ms slower across 16 layers. The opt-in and tests were removed. |
| PERF-036 | Reduce the native FA2 paged-prefix KV MMA tile for FP8 KV/head dimension 256. | FlashInfer `BatchPrefillWithPagedKVCacheDispatched` | Rejected | The correctly routed `NUM_MMA_KV=2` kernel regressed the exact ladder from 3013.932 to 3414.968 ms/layer (+13.306%) and changed output/LSE digests. CTA-Q 16 also lost; CTA-Q 32/128 are invalid. Restored CTA-Q 64 and `NUM_MMA_KV=4`. |
| PERF-008 | Build a deeper tree only after an oracle projection clears 200 TPS plus margin. | sparse p/q replay and topology optimizer | Fail-closed | Current capture is selected-tree only; measured D2/D4 shapes fail the impossible oracle. Funding requires complete lattice and conservative >=215 TPS. |
| PERF-009 | Recover graph-tail scheduling time. | async CUDA event probe and graph boundaries | Closed | Best repeatable conservative p10 is 0.658355 ms, below the 0.75 ms admission gate. |
| PERF-010 | Reproduce vLLM MTP-3 with TurboQuant K+1 verification. | isolated vLLM 0.27.1 lane, same checkpoint/GPU, exact client contract | Highest-priority comparison | External ~160 TPS claim lifts the path ceiling above 200 but lacks comparable workload evidence. |

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

### 2026-08-20 19:21 PDT - PERF-BASELINE-025 exact-200K current-source control

- Change: measurement only from clean `main` at
  `cb11475a4e0c68cfe542f66a919b468f205392f0`. Launched the selective
  `AttnNVFP4` checkpoint with chunk 7680, seed `615388882`, real 200K pools,
  ordinary M3 rejection sampling, and
  `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`. Startup promoted the selected 110
  target FP4 file-cache configs and captured target verify, draft decode, and
  draft extend graphs.
- Environment: native Windows RTX 5090, driver `610.88`, Python `3.13.14`,
  PyTorch `2.13.0+cu130`, CUDA runtime `13.0`, Triton `3.7.1`, and FlashInfer
  `0.6.17`. Before measurement the server occupied 27,224 MiB with 4,964 MiB
  free. Chrome, Edge WebView, iCloud, Windows shell/display clients, and the
  server Python process were resident WDDM clients.
- Warmup/cache policy: two full exact-shape warmups preceded five scored
  requests. Every scored invocation flushed the cache and completed exact
  `199000+16` with `finish_reason=length`.
- Prompt samples were
  `2905.351, 2927.990, 2957.401, 2936.653, 2959.654 tok/s`; mean
  **2937.410**, median **2936.653**, CV **0.683%**.
- Generation samples were
  `94.098, 90.935, 105.232, 87.540, 89.888 tok/s`; mean **93.539**, median
  **90.935**, CV **6.644%**. TTFT samples were
  `68.494311, 67.964704, 67.288805, 67.764230, 67.237593 s`; mean
  **67.749929 s**. E2E samples were
  `68.653720, 68.129658, 67.431347, 67.935580, 67.404467 s`; mean
  **67.910954 s**.
- Correctness evidence: all five requests returned `199000` prompt tokens,
  `16` completion tokens, `199016` total, thinking enabled, and stable output
  digest
  `cdf5bb57b88deaa7515abaedf36406d10494599fce2e23eeaa400461d9f647d9`.
- Decision: accept this as the reproducible current-environment control. It
  does not supersede the qualified `3048.086/112.499` record. Candidate
  attribution must use an adjacent A-B-A comparison because this environment
  is 3.631% below the prompt record.
- Artifact:
  `C:\Users\Daniel\.copilot\session-state\fd2e8d01-e225-4b48-9ab3-4d118100a4a9\files\baseline-exact200k-20260820-1910.log`.

### 2026-08-20 19:24 PDT - PERF-025 TRT-LLM dense FP4 capability gate

- Change: no source change. Exercised the existing real dense-linear
  `flashinfer_trtllm` path on the RTX 5090 before exposing it through the
  Windows launcher.
- Benchmark evidence: no timing qualified. FlashInfer `0.6.17` rejected
  shapes `(64,256,512)`, `(5,160,336)`, and `(128,1024,1024)` with
  `BackendSupportedError: mm_fp4 does not support backend 'trtllm' with
  capability 120`.
- Correctness evidence: the focused test reached checkpoint-format loading,
  TRT-LLM weight shuffle, activation quantization, and the real
  `ModelOptFp4LinearMethod.apply` call. The dependency rejected the backend
  before producing output.
- Decision: close on the installed native-Windows stack. Do not add the
  launcher choice or attempt a server launch until FlashInfer explicitly
  supports dense TRT-LLM FP4 on SM120.
- Commit: `538be003dd` (`docs: record exact-200k baseline and blocked backend`).

### 2026-08-20 19:30 PDT - PERF-028 bit-exact fused Gemma residual norm

- Change: added a JIT CUDA half-width Gemma residual-norm specialization that
  rounds `residual + input` to the output dtype before applying the existing
  RMSNorm vector ownership and reduction order. It stores the rounded residual
  and normalized input in place in one launch. Native Windows dispatch uses
  the fused path only where the current exact half-width kernel applies and
  retains the former two-launch fallback for every unsupported shape/dtype.
- Correctness evidence: focused native-Windows CUDA coverage passed exact
  equality for both mutated tensors at BF16 `H=5120`,
  `M={1,3,7000,7680}`. A separate CUDA-graph test captured the direct JIT op,
  changed both inputs before each replay, and passed exact equality at M1/M3.
  The existing fullgraph smoke also remained exact.
- Benchmark evidence: the established 5,000-iteration smoke measured the fused
  path versus the staged residual-add plus direct-output norm at
  **24.406 vs 40.812 us** (M1) and **25.280 vs 41.419 us** (M3), including
  matched input resets. Kernel-only stable A-B-A windows measured
  `9.554 / 16.705 / 9.505 us` at M1 and
  `9.760 / 16.478 / 9.450 us` at M3. Large-row A-B-A measured
  `195.403 / 193.696 / 195.809 us` at M7000 and
  `217.110 / 245.719 / 213.329 us` at M7680.
- Full-model fused evidence: five exact `199000+16` requests measured
  `2976.028, 2988.295, 2947.764, 2916.429, 2972.626` prompt tok/s
  (mean **2960.228**) and
  `88.022, 98.132, 112.992, 93.347, 101.591` generation tok/s
  (mean **98.817**). Mean TTFT/E2E were
  **67.229581/67.382454 s**. Three exact `199000+512` requests measured
  `116.100, 116.486, 117.162` generation tok/s (mean **116.583**) and
  **2974.600 prompt tok/s**.
- Adjacent staged control: three exact short requests averaged
  **2967.386/102.302 tok/s** with **67.064823/67.212277 s** TTFT/E2E.
  Three exact long requests measured
  `116.226, 113.749, 115.608` generation tok/s (mean **115.194**) and
  **2983.424 prompt tok/s**. Every fused and staged request preserved the
  exact total, finish reason, and established deterministic digest.
- Decision: retain as an additive decode win. The long adjacent arm improved
  **1.205%**, consistent with the isolated M1/M3 launch reduction. Do not
  attribute a prefill gain: the adjacent short and long prompt arms differed
  by only -0.241% and -0.296%, respectively.

### 2026-08-20 20:26 PDT - PERF-027 exact fused SwiGLU-to-NVFP4 producer

- Admission probe: the installed
  `flashinfer.silu_and_mul_nvfp4_quantize` API cannot run on this native
  Windows stack because it imports the unavailable `cutlass` Python module.
  The already-compiled native expert variant does run on SM120 and is fast,
  but changed about 0.8% of packed values because it uses fast SiLU and only
  one BF16 rounding boundary.
- Change: added a native-Windows JIT producer that computes precise FP32 SiLU,
  rounds the activation to BF16, multiplies by the FP32-converted up value,
  rounds the product to BF16, then reuses FlashInfer's native NVFP4
  E4M3-scale/E2M1 packing helpers. It writes caller-owned packed values and
  every 128x4 scale-layout padding byte in one launch.
- Exactness evidence: with production hidden width 17408 and the real layer-0
  down-projection input scale `0.0025692894123494625`, both packed values and
  every scale byte matched the current staged Windows sequence at
  `M={1,3,7000,7680}`.
- Isolated latency medians, staged versus exact fused:
  - M1: `50.528 -> 20.224 us` (**2.498x**)
  - M3: `49.568 -> 20.256 us` (**2.447x**)
  - M7000: `664.112 -> 366.176 us` (**1.814x**)
  - M7680: `730.416 -> 402.704 us` (**1.814x**)
- Decision: isolated numerical and latency admission passed. The kernel
  remains unwired until mutable CUDA-graph replay, fullgraph compilation, the
  real ModelOpt tuple consumer, and a whole-model adjacent comparison pass.
- Focused follow-up passed **8 tests**: mutable graph replay, fullgraph
  compilation, and a captured exact producer-to-ModelOpt CUTLASS tuple chain
  are all bit-exact. The producer was then wired behind the narrow native
  Windows TP1 serialized non-AWQ per-tensor ModelOpt/CUTLASS gate.
- First full-model arm: both 200K pools, 110 selected target tactics, and all
  three graphs passed. Five exact `199000+16` requests measured prompt
  `3044.589, 3014.218, 2984.494, 2984.251, 2940.207` tok/s (mean
  **2993.552**), TTFT mean **66.485194 s**, and E2E mean
  **66.660452 s**. This is +1.126% prompt and -0.744387 s TTFT versus the
  prior PERF-028 fused arm under the current environment.
- The arm is not promotable: every short request selected a new stable digest
  `9db488...21375`. Three exact `199000+512` requests selected another changed
  trajectory and measured `114.263, 119.534, 112.829` generation tok/s
  (mean **115.542**), below PERF-028's 116.583. Next step is real-activation
  numerical localization; no speed result can be retained until the digest
  divergence is eliminated.

### 2026-08-20 22:04 PDT - PERF-027 repaired and retained for eager prefill

- Exhaustive finite-BF16 localization found 520 packed-byte differences in
  final-product underflow groups. FlashInfer's separately compiled quantizer
  flushes BF16 subnormals at its module boundary; the fused precise module
  preserved them and could encode `0x77` under a zero E4M3 scale. The retained
  kernel canonicalizes only the final rounded BF16 subnormal to signed zero.
- A separate deployment-equivalent probe proved that the established compiled
  target graph has a different contract from eager prefill. Inductor removes
  the intermediate BF16 SiLU round: compiled native differed from eager staged
  by 63 packed/18 scale bytes at M1 and 216/51 at M3. PERF-027 therefore runs
  only outside `torch.compile`; compiled M3 keeps its original activation and
  quantizer.
- Focused coverage now passes **10 native CUDA tests**: exact
  M1/M3/M7000/M7680 values, compact and TMA all-finite-BF16 sweeps, mutable
  graph replay, fullgraph compilation, and a captured ModelOpt tuple chain.
  The Qwen3.5 ModelOpt CPU suite passes **10 tests**, including explicit eager
  tuple routing and compile-time preservation of the former activation path.
- The repaired selective server restored the short digest
  `cdf5bb57...f647d9` across five exact `199000+16` scores. Prompt samples
  were `2977.888, 3008.041, 2946.967, 2968.875, 3034.603` tok/s, mean
  **2987.275**; mean TTFT/E2E were **66.622932/66.776008 s**. Versus the
  PERF-028 fused arm, prompt improved **0.914%** and TTFT improved
  **0.606649 s**.
- Three exact `199000+512` requests restored digest `cac0c6...a2092`, measured
  prompt `3040.821, 2982.656, 2980.554` tok/s (mean **3001.344**) and
  generation `117.174, 114.334, 114.168` tok/s (mean **115.225**). The prompt
  gain persisted; decode remained within the existing 1-3% launch/WDDM range
  and is not attributed to this eager-only change.
- Decision: retain the exact eager-prefill producer as an additive prompt/TTFT
  win. A compiled-semantics producer is a separate candidate and must match
  the old M3 tuple and outer graph exactly before it can replace the compile
  guard.

### 2026-08-20 22:31 PDT - PERF-029 compiled producer rejected as graph-neutral

- FlashInfer's raw expert producer exactly matched the compiled M1/M3 packed
  values and scales when given caller-owned zeroed padding. A separate
  PDL-safe dense specialization reproduced that one-rounding fast-math
  contract and wrote deterministic padding.
- The custom producer matched every byte and improved isolated M3 median
  latency from **70.848 to 25.152 us**. Compiled graph replay, nested
  fullgraph, and a captured producer-to-ModelOpt tuple chain passed; the
  focused suites reached **16 CUDA / 10 CPU tests**.
- Five exact short requests retained digest `cdf5bb57...f647d9` but averaged
  only **2951.844 prompt / 97.653 short-generation tok/s** with
  **67.419972/67.573812 s** TTFT/E2E. Three exact long requests retained
  `cac0c6...a2092` and averaged **2983.961 prompt / 116.192 generation
  tok/s**.
- Device-cycle attribution closed the candidate. A 233-cycle M3 profile
  measured **16.389 ms mean / 16.045 ms median**, against the current
  **16.058 ms** control. The isolated launch removal was absorbed by the
  compiled graph's existing overlap/cost topology. The +0.839% long client
  movement is inside the observed variance and is not a win.
- Decision: remove PERF-029 and preserve the compile guard. Reopen only if a
  graph trace identifies serialized exposure rather than standalone launch
  latency.

### 2026-08-20 22:49 PDT - PERF-030/031 prefill split and packed-GDN routes closed

- Exposed the already-registered FlashInfer prefill split descriptor outside
  deterministic mode without changing ragged attention or workspace. Both
  fixed sizes 4096 and 8192 loaded 200K pools and captured all graphs, then
  failed on the first exact warmup: `batch_prefill_tmp_v` required
  **2,264,924,160 bytes** from the 128 MiB allocator. The code was removed.
- A separate implementation review found that Qwen3.8 target verification
  already avoids the proposed GDN split. Its packed width is 10,240, above the
  8,192 fused-materialization limit; `torch.split`/`view` create zero-copy
  aliases with token stride 10,240, which ReplaySSM already accepts. No
  production change is funded.

### 2026-08-20 23:01 PDT - PERF-032 tail coalescing rejected

- Added a default-off tail ceiling that planned exact 199K as
  `24 * 7680 + 14680`, preserving the eliminated 192,000-token Mamba
  checkpoint through the existing branching tracker. CPU scheduling and
  derived-buffer tests passed.
- Both 200K pools and all graphs passed. After one exact warmup, the scored
  request completed exact `199016` but reached only **1917.509 prompt tok/s**,
  **48.657 short generation tok/s**, **103.780505 s TTFT**, and
  **104.088783 s E2E**. Digest changed to `8e1d884c...43d42a`.
- The saved forward moved the former paged-prefix interaction into a
  14,680-token ragged-current causal pass. Its much worse kernel regime and
  changed reduction boundary overwhelmed dispatch savings. The full
  implementation and tests were removed.

### 2026-08-20 23:07 PDT - PERF-033 exact gate-to-NVFP4 fusion below admission

- Built a precise native producer matching the selected BF16 sigmoid-gate
  rounding and FlashInfer NVFP4 bytes, including final-subnormal FTZ
  canonicalization and deterministic scale padding. Production shapes,
  all-finite BF16, graph replay, fullgraph, and ModelOpt tuple consumption
  passed **9 CUDA tests**.
- Registered benchmark results, staged versus fused:
  M1 `2.650 -> 1.974 us`, M3 `2.731 -> 2.304 us`,
  M7000 `117.552 -> 77.945 us`, and M7680 `135.402 -> 85.124 us`.
- The target graph has only 16 full-attention layers, so the M3 saving projects
  to **0.0068 ms per replay**. Across 26 exact-prefill passes, the large-shape
  projection is only about **20.9 ms**. Both are below the admission scale
  needed to separate from current noise. No model code was wired; all
  experimental files were removed.

### 2026-08-20 23:34 PDT - PERF-034 page-size sweep rejected

- Page 128 failed the exact-pool contract during startup: both pools were
  floored to 199,936 tokens because 200,000 is not divisible by 128.
- Page 32 retained exact pools and the deterministic digests. Five warmed
  short prompts averaged **3030.480 tok/s** with **65.668465 s TTFT**; the
  isolated screening hit was 3071.156 tok/s. Three long requests averaged
  **2970.617 prompt / 112.576 generation tok/s**, below the retained
  3001.344/115.225 adjacent window.
- Source reachability explains the inconsistency: FlashInfer prefill is planned
  with page size 1 and per-token slot IDs, so the storage-pool page setting
  does not change the dominant paged-prefix kernel. Page 32 mainly changes
  XQA/cache geometry and regressed decode. Page 64 remains selected.

### 2026-08-21 00:17 PDT - PERF-035 FP16 QK reduction initially appeared positive

- At the exact Q=7000/KV=192000 paged-prefix shape, FP16 QK reduction preserved
  BF16 output and LSE bit-for-bit while improving median kernel time
  **109.321 -> 108.172 ms**.
- Added a default-off expert environment gate reaching only ordinary paged
  prefill. Speculative target verification and the graph-stable fast planner
  retain their captured FP32 reduction module.
- Candidate five-run exact prompt mean was **3005.592 tok/s** with
  **66.212604/66.379962 s** TTFT/E2E. The adjacent env-off control averaged
  **2985.317 tok/s** and **66.661888/66.834958 s**. This is a repeatable
  **+20.275 tok/s / +0.679%** and **-0.449285 s / -0.674%** TTFT movement,
  aligned with the isolated kernel delta.
- Three candidate/control exact `199000+512` windows averaged
  **3003.053/113.231** versus **2958.955/113.407** prompt/generation tok/s.
  Decode is neutral as expected; both short and long digests remained exact.
- This was provisionally retained pending a full prefix-ladder attribution.

### 2026-08-21 00:24 PDT - PERF-035 rejected after exact ladder attribution

- Corrected the isolated benchmark to the checkpoint's real full-attention
  dimensions: 24 query heads, 4 KV heads, and head dimension 256. The earlier
  32/16/128 probe was not representative.
- Measured all 25 paged-prefix calls in exact request order:
  24 Q=7680 prefixes from 7,680 through 184,320, then Q=7000/KV=192,000.
  Every FP16/FP32 output and LSE pair was bit-exact.
- Summed per-shape medians were **2964.761 ms FP32** versus
  **2974.993 ms FP16 per layer**. Across 16 full-attention layers, FP16 was
  **163.705 ms slower**, with mixed signs and large individual-shape variance.
- The initial five-request +0.679% movement did not clear server variance and
  lacks a kernel mechanism on the true workload. Removed the environment
  descriptor, backend routing, and test change. FP32 QK reduction remains
  selected.

### 2026-08-21 01:11 PDT - PERF-036 native paged-prefix tile family rejected

- Source-provenance review found that the first `NUM_MMA_KV=2` experiment
  changed the single-prefill dispatcher, not the batch-paged dispatcher that
  owns the 78.1%-share exact-prefix kernel. Its apparent
  **3013.932 -> 2977.011 ms/layer** movement is noise and is not credited.
- After moving the cap exclusively to
  `BatchPrefillWithPagedKVCacheDispatched`, the same 25-shape ladder regressed
  to **3414.968 ms/layer** (**+13.306%**) and changed both output and LSE
  digests. CTA-Q 16 measured **4154.807 ms/layer**; CTA-Q 32/128 are invalid
  for the active FP8/head-dimension-256 traits.
- Restored the maintained and installed FlashInfer headers byte-for-byte,
  removed the generated candidate module, and retained CTA-Q 64 with
  `NUM_MMA_KV=4`. No full server gate was warranted.
