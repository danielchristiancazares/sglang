# Performance Log

## Current Active Timings

| Benchmark | Baseline | Current | Delta | Command | Last Updated |
|---|---:|---:|---:|---|---|
| M1 Max mixed 4.951-bpw Q5-class target, sampled direct `128 / 32 warm / 128 timed` | generic MLX QMM **18.121698566 tok/s** | selected A100+A111+A114+A113+A117 ten-sample mean **19.453092367 tok/s** | **+1.331393801 tok/s / +7.347%** over generic and **+0.184616727 / +0.958%** over matched serial-epilogue control **19.268475640**; exact digest stable; **0.546907633 tok/s / 2.811%** remains to the floor | pinned mixed revision `596b8067...8340`, seed 42, selected command-buffer controls, Q5/Q4/fused-MLP switches | 2026-09-01 18:36 PDT |
| M1 Max fused affine-Q4 gate/up/SwiGLU source-order scheduling micro | selected A117 gate rows then up rows **0.558719448 ms** | A122 interleaved gate/up rows **0.569066813 ms** | **+0.010347364 ms / +1.851979%** regression; loses all ten order/reverse pairs while retaining exact digest `8a9031349585365a` | production `K=5120`, `N=17408`, `1000 / 10000`; preserved A117 and dedicated A122 executables | 2026-09-01 19:18 PDT |
| M1 Max linear-attention affine-Q5 `qkv` plus `z`, deterministic batch-one direct micro and sampled full model | separate A100 launches **0.421745535--0.426645998 ms** aggregate by harness/window | row-concatenated A119 reaches **0.420592346 ms** but loses full-model; direct two-output A120/A121 reach **0.427401865 / 0.426008156 ms** | A119's isolated launch win does not survive its split graph; split-free 4-SIMD and exact-ratio 5:3 kernels regress **1.341171% / 0.616515%**; all exact; launch-only family closed | production K/N `5120/(10240+6144)`, order/reverse 2,000-iteration windows; PERF-FA136/137 | 2026-09-01 19:09 PDT |
| M1 Max fused affine-Q4 gate/up/SwiGLU epilogue, sampled direct `128 / 32 warm / 128 timed` | A114+A113 serial four-result epilogue **19.268475640 tok/s** | A117 four-lane register-selected epilogue **19.453092367 tok/s** | forward/reversed windows improve **+0.947541% / +0.968716%**, aggregate **+0.184616727 tok/s / +0.958128%**; all 20 clean runs retain canonical digest/last token; a separate reload-churn window is excluded | same exact direct contract and artifacts differing only in the fused Metal epilogue | 2026-09-01 18:36 PDT |
| M1 Max mixed-target Q4 gate/up/SwiGLU, sampled direct `128 / 32 warm / 128 timed` | matched A100+A111 control **19.228720305 tok/s** | precise-exp A114 **19.268407916 tok/s** | two independent five-pair windows improve **+0.145685% / +0.267156%**, aggregate **+0.039687612 tok/s / +0.206398%**; all 20 runs retain canonical digest/last token; boundary regression rejects the old fast-exp artifact in all 32 rows | same direct contract with `SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU=0/1` as the only variable | 2026-09-01 17:56 PDT |
| M1 Max affine-Q5/G64 target-only batch-one decode, sampled direct `128 / 32 warm / 128 timed` | native affine-Q5 target **16.322505765 tok/s** | opt-in direct Q5 QMV **17.823163930 tok/s** | **+1.500658165 tok/s / +9.194%** in the first complete screen; representative parity passes; **2.176836070 tok/s** remains to the floor and a repeated matched window is pending | pinned Q5 target, selected command-buffer controls, and `SGLANG_MLX_NATIVE_Q5_BATCH_ONE_QMV=1`; PERF-A094/FA122/FA123/FA124 | 2026-09-01 12:43 PDT |
| M1 Max affine-Q5 plus matched 5-bit MTP, deterministic direct `128 / 32 warm / 128 timed` | three-token block **17.919995235 tok/s**, width **3.0** | opt-in eight-token block **31.380317186 tok/s**, width **8.0**; exact sampled p/q arms peak at **6.380162135 tok/s**, width **2.285714286** | **+13.460321951 tok/s / +75.113%** for the deterministic execution-cost probe, clearing 20 by **11.380317186 tok/s**; native sampling preserves its configured distribution and rejects this proposal route for production | pinned Q5 target/MTP snapshots plus `SGLANG_MLX_NATIVE_MTP_BLOCK_SIZE=8`; PERF-A093/FA120/FA121 | 2026-09-01 12:08 PDT |
| M1 Max affine-Q5 target plus DFlash2, sampled direct `128 / 32 warm / 128 timed` | generic affine-Q5 verifier **11.506669050 tok/s**, M=8 about **363--368 ms** | native Q5 M=8 K-split **13.721235888 tok/s**, M=8 about **228--229 ms** | **+2.214566838 tok/s / +19.246%**; representative five-bit unpack parity passes; target-only remains faster at **16.322505765 tok/s** | PERF-A092 candidate dylib with the exact PERF-A091 DFlash command | 2026-09-01 11:49 PDT |
| M1 Max Qwen3.8-27B affine-Q5/G64 native target, sampled direct `128 / 32 warm / 128 timed` | GGUF Q5 1K-FP8 five-run mean **7.2614 tok/s** | first native affine-Q5 sample **16.322505765 tok/s** | **+9.061105765 tok/s / +124.781%**; immutable 18.51 GB five-bit target loads without source changes; served 131K and behavior gates pending | `bench_qwen38_native` with native sampling seed 42 and the pinned affine-Q5 snapshot | 2026-09-01 11:43 PDT |
| M1 Max Qwen3.8-27B Q5_K_S, Q4_K token embedding, exact 131K FP8 KV pool, sampled served `128+32` | same-artifact BF16: **0.315 tok/s**, **5.616 prompt tok/s**, **22.790664 s TTFT**, **121.240667 s E2E** | FP8 five-run mean **3.237 tok/s**, **5.8974 prompt tok/s**, **21.705304 s TTFT**, **31.286811 s E2E** | cache **8.00 -> 4.00 GB**, reported headroom **0.99 -> 6.99 GB**; **10.276x / +927.619%** generation; exact capacity, reasoning, arithmetic, and tools pass | exact PERF-A090 server contract; five cache-flushed ordinary sampled requests | 2026-09-01 11:29 PDT |
| M1 Max Qwen3.8-27B Q5_K_S, Q4_K token embedding, 1K FP8 KV pool, sampled served `128+32` | same-artifact BF16 smoke: **7.086 tok/s**, **5.809 prompt tok/s**, **22.033405 s TTFT**, **26.408174 s E2E** | native FP8 five-run mean **7.2614 tok/s**, **5.8668 prompt tok/s**, **21.818682 s TTFT**, **26.087973 s E2E** | first native FP8 capacity lane; **+0.1754 tok/s / +2.475%** versus the prior same-artifact smoke; exact lengths, reasoning, arithmetic, and tools pass | PERF-A089 server contract; five cache-flushed ordinary sampled requests | 2026-09-01 11:17 PDT |
| M1 Max Qwen3.8-27B Q5_K_S, Q4_K token embedding, exact 131K BF16 pool, sampled served `128+32` | F16 token embedding: **0.102 tok/s**, **5.271 prompt tok/s**, **24.284058 s TTFT**, **329.689187 s E2E** | Q4_K token embedding: **0.315 tok/s**, **5.616 prompt tok/s**, **22.790664 s TTFT**, **121.240667 s E2E** | runtime residency **21.37 -> 20.00 GB**; **3.088235x / +208.824%** generation; exact capacity, reasoning, arithmetic, and tools pass; paging remains active | exact PERF-A088 server contract, changing only model artifact; ordinary sampled client with `--skip-warmup` | 2026-09-01 10:55 PDT |
| M1 Max Qwen3.8-27B Q5_K_S-derived, exact 131K BF16 pool, sampled served `128+32` | selected 1K-pool mean **7.1646 tok/s**, **5.809 prompt tok/s**, **22.035143 s TTFT** | exact 131K pool: **0.102 tok/s**, **5.271 prompt tok/s**, **24.284058 s TTFT**, **329.689187 s E2E** | exact capacity, health, language-only metadata, and reasoning pass; generation is **98.576%** slower under unified-memory paging; stock MPS float8 allocation is unsupported | exact PERF-A086 server contract; ordinary sampled client with `--skip-warmup` | 2026-09-01 10:39 PDT |
| M1 Max Qwen3.8-27B Q5_K_S-derived, Q6_K exact-batch-four verifier projections | matched `SGLANG_MPS_Q6_K_BATCH4_ROWS16=0`: head **29.166000 ms**, QKV **1.442333 ms** | 16-row/eight-lane cohort: head **6.121375 ms**, QKV **0.588500 ms** | **-79.012% / -59.198%** latency; sampled same-GGUF NEXTN `128+128` improves **3.3384 -> 3.7028 tok/s** (**+10.915%**) even though control acceptance is higher | `.venv/bin/python benchmark/mac/bench_mps_gguf_quant.py $Q5_DERIVED --tensor {output.weight,blk.0.attn_qkv.weight} --batch-size 4 --warmup 8 --iterations 25`; matched sampled server window | 2026-09-01 10:20 PDT |
| M1 Max Qwen3.8-27B Q5_K_S-derived target, sampled served `128+32`, 1K pool | generic Q6_K batch-one path: **5.746 tok/s**, **5.8 prompt tok/s**, **22.070851 s TTFT**, **27.466190 s E2E** | Q6_K two-row plus Q5_K 32-row reuse: five-run **7.1646 tok/s** mean, warmed-four **7.2075**; five-run **5.809 prompt tok/s**, **22.035143 s TTFT**, **26.362669 s E2E** | **+1.4186 / +24.688%** generation from the original baseline; Q5_K change adds **+0.426%** against its matched disabled mean; exact 160 tokens and reasoning preserved; **12.8354 tok/s / 2.7915x** remains to the floor | `bench_openai_stream.py --model qwen3.8-27b-q5 --input-tokens 128 --output-tokens 32 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5 --skip-warmup --timeout 600` | 2026-09-01 09:45 PDT |
| M1 Max Q5_K batch-one 32-row cohort, sampled served `128+128`, 1K pool | matched `SGLANG_MPS_Q5_K_BATCH1_ROWS32=0`: **7.456 tok/s median**, **7.41675** first-four mean | default candidate: **7.500 tok/s median**, **7.44675** first-four mean, **7.5065** warmed-four mean | **+0.044 / +0.590%** median and **+0.404%** first-four mean; every request exact at 256 tokens; one control tail sample is retained as externally contended | same sampled command with `--output-tokens 128` | 2026-09-01 09:45 PDT |
| M1 Max full-Q4 target-only long-history GPU shader attribution, sampled `6237 / 1 warm / 128 timed` | prior Metal System Trace exposed command-buffer cadence without Shader Timeline labels | **88.470%** affine-W4 `qmv_fast`; **3.904%** two-pass SDPA; **2.358%** recurrent state update; **1.405%** full-attention q/k norm plus RoPE | 47,676 of 48,343 sampled shader PCs map to the target process; the remaining throughput branch is the stock-compatible matrix-tiled QMV/dependency owner | `xctrace record --template 'Metal System Trace' --instrument 'Metal GPU Counters' ...` plus exported shader-PC attribution | 2026-09-01 07:41 PDT |
| M1 Max native full-Q4 target-only prefill, sampled served `6237+128`, real 131K pools | one-shot native prefill: Metal OOM before generation | internal 2,048-token prefill: **19.300 tok/s**, **109.988 prompt tok/s**, **56.706198 s TTFT**, **63.286374 s E2E** | exact 6,365-token request now completes with coherent reasoning; direct long-history decode is **19.586705 tok/s** and short decode remains above 20 with an exact digest; **0.700 tok/s** remains to the served floor | exact target-only server/client contract plus `SGLANG_MLX_NATIVE_TARGET_ONLY_PREFILL_CHUNK_SIZE=2048` | 2026-09-01 07:09 PDT |
| M1 Max full-Q4 affine-W4 batch-one QMV, direct `128 / 32 warm / 128 timed` | stock MLX **20.339874670 tok/s** | one-SIMD-per-output **10.456143330 tok/s**; quant-parameter subgroup broadcast **7.773375044 tok/s** | **-48.593% / -61.783%**; both native forms removed after exact standalone parity and full-model screens | candidate dylib plus `SGLANG_MLX_NATIVE_BATCH_ONE_QMV=1` under the target-only direct contract | 2026-09-01 07:28 PDT |
| M1 Max DFlash2 selected-q M=2/M=8 budget, sampled served `6237+128`, real 131K pools | adjacent current-source threshold-disabled **15.883 / 15.908 / 15.900 / 15.899 / 15.897 tok/s**, mean **15.8974** | mean-q6 threshold 0.62 **16.167 / 16.181 / 16.178 / 16.179 / 16.183 tok/s**, mean **16.1776** | **+0.2802 / +1.763%**; all requests exact with a stable coherent digest; 23 M=2 and 19 M=8 cycles per request; retained opt-in with **3.8224 tok/s** to the floor | exact PERF-A076 server/client contract plus `SGLANG_MLX_NATIVE_DFLASH_MEAN_Q_THRESHOLD=0.62` | 2026-09-01 06:54 PDT |
| M1 Max DFlash2 M=8 gate/up dispatch fusion, sampled direct `128 / 32 warm / 128 timed` | separate SG16/B32 gate/up five-sample mean **26.943319961 tok/s** | paired two-plane dispatch five-sample mean **26.964966176 tok/s** | **+0.021646215 / +0.08034%**, with two of five adjacent pairs flat/slower; exact trajectory preserved and complexity rejected. Sequential fused-SwiGLU arm regressed trace throughput **26.801071247 -> 26.413620964** | PERF-A077 exact PERF-A076 direct contract at selector temperature 1.15 | 2026-09-01 06:18 PDT |
| M1 Max DFlash2 selector calibration, sampled served `6237+128`, real 131K pools | adjacent selector temperature 1.0: **15.434 / 15.449 / 15.443 / 15.443 / 15.443 tok/s**, mean **15.4424** | selector temperature 1.15: **15.870 / 15.884 / 15.890 / 15.896 / 15.893 tok/s**, mean **15.8866** | **+0.4442 / +2.876%**; exact 6,365 tokens in every sample, stable coherent digest within each setting, mean emitted width **3.878788 -> 4.031250**; retained opt-in with **4.1134 tok/s** to the floor | exact PERF-A065 contract plus `SGLANG_MLX_NATIVE_DFLASH_SELECTOR_TEMPERATURE=1.15` | 2026-09-01 06:04 PDT |
| M1 Max DSpark target-state score, sampled served `6237+128`, real 131K pools | ratio-1.75 baseline **13.580 tok/s** | temporary score-0.525 bypass **13.652 tok/s** | one-sample **+0.072 / +0.530%** versus its no-bypass trace, while trailing selected cooldown-16 by **2.9876 tok/s**; real score/accepted-width Pearson **0.141** closes the scheduler and retains trace-only telemetry | PERF-A075 exact PERF-A073 contract with trace-only target score; temporary threshold source removed | 2026-09-01 05:33 PDT |
| M1 Max native DSpark low-budget probe cooldown, sampled served `6237+128`, real 131K pools | confidence-budget ratio 1.75 mean **13.6058 tok/s** | 16 target-only refills after each M=2 choice: first five **17.028 / 17.080 / 17.036 / 17.102 / 14.952 tok/s**, mean **16.6396**; recovery **17.011** | all-sample **+3.0338 / +22.298%**; five normal samples within the six-request window mean **17.0514**; one transiently contended sample retained; exact tokens and one shared reasoning/output SHA-256 | PERF-A074 exact PERF-A073 contract plus `SGLANG_MLX_NATIVE_DSPARK_BYPASS_REFILLS=16` | 2026-09-01 05:10 PDT |
| M1 Max native DSpark trained-confidence M=2/M=8 budget, sampled served `6237+128`, real 131K pools | adjacent fixed-M=8 control **11.313 tok/s** | ratio **1.75**: **13.595 / 13.609 / 13.603 / 13.607 / 13.615 tok/s**, mean **13.6058** | **+2.2928 / +20.267%**; every sample completed exact 6,365 tokens with one shared reasoning/output SHA-256; retained opt-in while DFlash2 holds **15.328** and target-only holds above **20** | exact PERF-A068 server/client contract plus `SGLANG_MLX_NATIVE_DSPARK_CONFIDENCE_COST_RATIO=1.75` | 2026-09-01 04:43 PDT |
| M1 Max native DSpark fixed verification-width sweep, sampled direct `128 / 1 warm / 32 timed` | seven drafts / target M=8 **10.025000 tok/s**, width **2.428571** | one draft / target M=2 **11.920231 tok/s**, width **1.6**; M=3/4/5/6/7 **8.861866 / 8.353200 / 7.544333 / 4.104266 / 4.139508 tok/s** | M=2 is the only useful short tier; M=6/7 hit a slow generic affine-QMM tier, while selected SG16/B32 makes M=8 faster; fixed shortening remains below 20 | PERF-A072 candidate dylib, exact PERF-A068 environment plus `SGLANG_MLX_NATIVE_DSPARK_VERIFY_DRAFT_TOKENS=1..7` | 2026-09-01 04:18 PDT |
| M1 Max native DSpark trained-confidence telemetry, sampled direct `128 / 1 warm / 32 timed` | affine-W4 **10.050625 tok/s**, width **2.428571** | trace-enabled confidence **10.071235 tok/s**, width **2.428571** | exact digest/last token retained; steady draft **36.58--36.88 ms** includes confidence; timing movement is diagnostic noise | PERF-A071 candidate dylib with the existing trace flag and otherwise exact PERF-A068 environment | 2026-09-01 04:11 PDT |
| M1 Max native Qwen3.8 DSpark v2, sampled served `6237+128`, real 131K pools | selected DFlash2 **15.328 tok/s** | first affine-W4 DSpark **11.242 tok/s** | **-4.086 / -26.657%**; DSpark live draft **42--45 ms**, verify **212--213 ms**; exact tokens/reasoning retained | exact PERF-A065 server/client contract, changing only `SGLANG_MLX_MTP_DIR` to the DSpark affine-W4 artifact | 2026-09-01 03:48 PDT |
| M1 Max native Qwen3.8 DSpark v2, sampled direct `128 / 1 warm / 32 timed` | official BF16 **5.746863 tok/s**, width **2.285714**, steady draft **44.54--44.75 ms** | affine-W4 **10.050625 tok/s**, width **2.428571**, steady draft **36.55--40.07 ms** | first functional screen; trajectories differ, so throughput is admission evidence rather than a precision ranking; both remain below 20 | PERF-A068 direct command with the same target, seed, sampler, verifier, and only the draft directory changed | 2026-09-01 03:44 PDT |
| M1 Max DFlash2 proposal policy, sampled served `6237+128`, real 131K pools | learned selector **15.328 tok/s** | greedy target-head proposal **9.512 tok/s** | **-5.816 / -37.944%**; synthetic direct mean **37.518731 tok/s** was nonrepresentative | PERF-A065 server/control with temporary `SGLANG_MLX_NATIVE_DFLASH_GREEDY_DRAFT=1` | 2026-09-01 03:17 PDT |
| M1 Max native DFlash2 draft precision, sampled direct `128 / 32 warm / 128 timed` | affine-W4 **30.992508 tok/s**, width **6.684211** | official dense BF16 **9.044043 tok/s**, width **2.114754** | **-21.948464 / -70.819%**; dense loader retained for compatibility, BF16 production selection rejected | PERF-A062 direct command, one candidate dylib, changing only the final draft directory | 2026-09-01 03:10 PDT |
| M1 Max native full-Q4 plus affine-W4 DFlash2, sampled direct `128 / 32 warm / 128 timed` | SG16/B16 mean **17.651701 tok/s**, mean emitted width **4.3** | SG16/B32 **31.347246 / 31.354397 / 31.268351 / 31.330814 / 31.335863 tok/s**, mean **31.327334**, mean emitted width **6.684211** | **+13.675633 / +77.475%**; direct throughput clears 20 by **11.327334 tok/s**; steady cycle is about **214--217 ms** | add `SGLANG_MLX_NATIVE_M8_KSPLIT_QMM=1` to the signed PERF-A059 command; SG16/B32 is in `3bae8a5e67` | 2026-09-01 02:54 PDT |
| M1 Max full-Q4 plus affine-W4 DFlash2, Codex 0.151.0 xhigh tool turn, real 131K pools | SG16/B16 exact `6237+128` **15.336 tok/s**, live verify about **236--239 ms** | SG16/B32 exact `6237+128` **15.328 tok/s**, live verify about **211--214 ms**; exact xhigh tool turn passes | fixed cycle improves about **10%** while this sampled trajectory shifts; sustained result leaves **4.672 tok/s** to the floor | exact server, sampled control, and bounded Codex commands recorded under PERF-A065 | 2026-09-01 03:03 PDT |
| M1 Max full-Q4 DFlash2 steady verification cycle | stock MLX affine path: draft **41.288--41.686 ms**, verify **352.531--356.089 ms**; partial restore/re-forward **83--191 ms** | selected affine QMM: draft **34.211--37.378 ms**, verify **305.130--305.604 ms**; accepted-prefix replay about **2--6 ms** | draft and verify each improve about **15%** at the measured bounds; recurrent partial commit becomes a low-single-digit-ms stage | `SGLANG_MLX_NATIVE_TRACE_SPEC=1` matched direct full-Q4 screens | 2026-09-01 02:03 PDT |
| M1 Max native early-out27 v2 with recurrent QKV restored from Q4, sampled served `6237+128`, real 131K pools | selected Q2 checkpoint **20.1556 tok/s** mean with reproducible malformed extra tool call | two independent QKV windows **20.0222 / 20.0292 tok/s** mean; every sample above 20 | **-0.1334 / -0.662%** from selected mean; one clean xhigh tool turn, one post-window turn recovered after a malformed extra call | `SGLANG_MLX_NATIVE_LINEAR_ATTN_OVERRIDE_PATH=<immutable-q4> SGLANG_MLX_NATIVE_LINEAR_ATTN_OVERRIDE_SCOPE=qkv ... bench_openai_stream.py --input-tokens 6237 --output-tokens 128 ...`; then the pinned xhigh Codex shell gate | 2026-09-01 00:26 PDT |
| M1 Max native early-out27 v2 with recurrent QKV+Z restored from Q4, sampled served `6237+128`, real 131K pools | selected Q2 checkpoint **20.1556 tok/s** mean with reproducible malformed extra tool call | first QKV+Z sample **19.948 tok/s** with a clean xhigh tool turn | **-0.2076 / -1.030%** from selected mean; **0.052 tok/s** below floor; one `/bin/pwd`, correct result, exact final marker, exit 0 | same precision-overlay command with scope `qkvz`; then the pinned xhigh Codex shell gate | 2026-09-01 00:26 PDT |
| M1 Max native early-out27 v2, request-local sampled served `6237+128`, real 131K pools | signed `883da94` restart **20.1356 tok/s** mean | request-boundary reseed **20.1556 tok/s** mean | **+0.0200 / +0.099%**; five samples **20.155/20.148/20.157/20.152/20.166**, every sample clears 20 and shares one digest | `MLX_SDPA_BLOCKS=64 SGLANG_MLX_NATIVE_SAMPLING=1 ... bench_openai_stream.py --input-tokens 6237 --output-tokens 128 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5 --skip-warmup`, five sequential samples | 2026-08-31 17:43 PDT |
| M1 Max native early-out27 v2, first sampled served `6237+128` window, real 131K pools | selected 128-partial sampled screen **19.941 tok/s** | opt-in 64-partial first window **20.1722 tok/s** mean | **+0.2312 / +1.159%**; all five samples clear 20; the first xhigh shell round trip passed and later order-replay exposed process-global RNG drift | same production-sampled command, five sequential samples | 2026-08-31 16:46 PDT |
| M1 Max native early-out27 v2, Codex 0.151.0 xhigh shell-tool turn, real 131K pools | first sampled turn: one `/bin/pwd`, final marker, exit 0 in about 81.8 s | fixed-seed request replay reaches the valid `/bin/pwd` and then emits a malformed `write_stdin` handle | speed and prompt-snapshot reuse pass; broader structured-output qualification remains active | hash-pinned isolated `CODEX_HOME`, strict config, ephemeral 120-second command recorded in the experiment log | 2026-08-31 17:43 PDT |
| M1 Max native early-out27 v2, deterministic served `6237+128`, real 131K pools | fused full-attention q/k norm/RoPE **19.9886 tok/s** mean | reduced RMS barriers **20.0173 tok/s** combined ten-request mean | **+0.0287 / +0.144%**; first and independent windows clear 20 in aggregate | `MLX_MAX_MB_PER_BUFFER=128 ... bench_openai_stream.py --input-tokens 6237 --output-tokens 128 --temperature 0 --skip-warmup`, two five-sample restarts | 2026-08-31 15:21 PDT |
| M1 Max native early-out27 v2, deterministic served `6237+128`, real 131K pools | MLX 50 MiB command-buffer budget **19.2260 tok/s** mean | 128 MiB plus fused full-attention q/k norm/RoPE **19.9886 tok/s** mean | **+0.7626 / +3.966%**; all measured requests exact; **0.0114 tok/s** remains to the client floor | `MLX_MAX_MB_PER_BUFFER=128 ... bench_openai_stream.py --input-tokens 6237 --output-tokens 128 --temperature 0 --skip-warmup`, five-sample candidate window against qualified controls | 2026-08-31 14:18 PDT |
| M1 Max native early-out27 v2, direct deterministic 6,237-history decode, 32 warm + 256 timed | MLX 50 MiB command-buffer budget **19.623300 tok/s** mean | native-engine 128 MiB default **20.153966 tok/s** mean | **+0.530666 / +2.704%**; every candidate clears 20 and all five pairs are exact | `bench_qwen38_native ... 6237 32 256`, process-isolated adjacent control/candidate pairs | 2026-08-31 13:21 PDT |
| M1 Max native early-out27 v2, direct deterministic 6,237-history decode, 32 warm + 256 timed | separate recurrent beta/decay graphs **19.641655 tok/s** mean | beta/decay inside q/k normalization owner **19.547612 tok/s** mean | **-0.094043 / -0.479%**; all five adjacent pairs exact and slower; rejected | `bench_qwen38_native ... 6237 32 256`, process-isolated adjacent control/candidate pairs | 2026-08-31 12:43 PDT |
| M1 Max native early-out27 v2, deterministic served `6237+128`, real 131K pools | separate BF16 convolution and SiLU **19.1730 tok/s** mean | fused convolution/SiLU owner **19.2134 tok/s** mean | **+0.0404 / +0.211%**; all ten requests exact | `bench_openai_stream.py --input-tokens 6237 --output-tokens 128 --temperature 0 --skip-warmup`, process-isolated five-sample control/candidate windows | 2026-08-31 11:41 PDT |
| M1 Max native early-out27 v2, direct deterministic 6,237-history decode, 32 warm + 256 timed | separate BF16 convolution and SiLU **19.524068 tok/s** mean | fused convolution/SiLU owner **19.632483 tok/s** mean | **+0.108415 / +0.555%**; all five adjacent pairs exact and positive | `bench_qwen38_native ... 6237 32 256`, process-isolated adjacent control/candidate pairs | 2026-08-31 11:25 PDT |
| M1 Max native early-out27 v2, deterministic served `6237+128`, real 131K pools | separate recurrent output RMSNorm and SiLU gate **19.1260 tok/s** mean | fused norm/gate owner **19.1548 tok/s** mean | **+0.0288 / +0.151%**; all ten requests exact | `bench_openai_stream.py --input-tokens 6237 --output-tokens 128 --temperature 0 --skip-warmup`, process-isolated five-sample control/candidate windows | 2026-08-31 11:05 PDT |
| M1 Max native early-out27 v2, direct deterministic 6,237-history decode, 32 warm + 256 timed | separate recurrent output RMSNorm and SiLU gate **19.469136 tok/s** mean | fused norm/gate owner **19.515718 tok/s** mean | **+0.046582 / +0.239%**; all five adjacent pairs exact and positive | `bench_qwen38_native ... 6237 32 256`, process-isolated adjacent control/candidate pairs | 2026-08-31 10:50 PDT |
| M1 Max native early-out27 v2, deterministic served `6237+128`, real 131K pools | separate recurrent q/k RMSNorm and scaling **19.0900 tok/s** mean | fused q/k normalization owner **19.1610 tok/s** mean | **+0.0710 / +0.372%**; all ten requests exact | `bench_openai_stream.py --input-tokens 6237 --output-tokens 128 --temperature 0 --skip-warmup`, process-isolated five-sample control/candidate windows | 2026-08-31 10:19 PDT |
| M1 Max native early-out27 v2, direct deterministic 6,237-history decode, 32 warm + 256 timed | separate recurrent q/k RMSNorm and scaling **19.319062 tok/s** mean | fused q/k normalization owner **19.464738 tok/s** mean | **+0.145676 / +0.754%**; all five adjacent pairs exact and positive | `bench_qwen38_native ... 6237 32 256`, process-isolated adjacent control/candidate pairs | 2026-08-31 10:04 PDT |
| M1 Max native early-out27 v2, deterministic served `6237+128`, real 131K pools | separate BF16 residual addition and RMSNorm **18.8286 tok/s** mean | fused residual/RMS owner **19.0484 tok/s** mean | **+0.2198 / +1.167%**, all ten requests exact | `bench_openai_stream.py --input-tokens 6237 --output-tokens 128 --temperature 0 --skip-warmup`, process-isolated five-sample control/candidate windows | 2026-08-31 09:42 PDT |
| M1 Max native early-out27 v2, direct deterministic 6,237-history decode, 32 warm + 256 timed | separate BF16 residual addition and RMSNorm **19.222533 tok/s** uncontended mean | fused residual/RMS owner **19.310857 tok/s** mean | **+0.088324 / +0.459%**; all six adjacent pairs exact and positive | `bench_qwen38_native ... 6237 32 256`, process-isolated adjacent control/candidate pairs | 2026-08-31 09:26 PDT |
| M1 Max native early-out27 v2, direct deterministic `128 / 32 warm / 256 timed` | target-only fused-convolution **20.187703 tok/s** | native 4-bit MTP-2 **9.649960 tok/s** | **-10.537743 / -52.199%** despite exact output and **1.888889** mean block width; rejected | `bench_qwen38_native ... 128 32 256 [MTP_DIR]` | 2026-08-31 09:01 PDT |
| M1 Max native early-out27 v2, deterministic served `6237+128`, real 131K pools | general depthwise convolution plus concatenated state **18.7616 tok/s** mean | fused decode convolution/state owner **18.8914 tok/s** mean | **+0.1298 / +0.692%**, all ten requests exact | `bench_openai_stream.py --input-tokens 6237 --output-tokens 128 --temperature 0 --skip-warmup`, process-isolated five-sample control/candidate windows | 2026-08-31 08:53 PDT |
| M1 Max native early-out27 v2, direct deterministic 6,237-history decode, 32 warm + 256 timed | general depthwise convolution plus concatenated state **19.120031 tok/s** mean | fused decode convolution/state owner **19.221589 tok/s** mean | **+0.101558 / +0.531%**, all five pairs exact | `bench_qwen38_native ... 6237 32 256`, process-isolated adjacent control/candidate pairs | 2026-08-31 08:37 PDT |
| M1 Max native early-out27 v2, direct deterministic decode after a 6,237-token history | historical direct control **19.151623 tok/s** | standalone C++ harness **19.116578 tok/s** | **-0.035045 / -0.183%** with identical `13eb9a7159a2612f` digest; harness retained | `/private/tmp/bench_qwen38_native ... 6237 32 128` | 2026-08-31 08:09 PDT |
| M1 Max native early-out27 v2, deterministic served `6237+128`, real 131K pools | separate linear-attention b/a projections **18.845 tok/s** | separate b/a projections **18.845 tok/s** | fused 96-row projection reached **18.511 tok/s** (**-1.772%**) with identical output; rejected | same process-isolated exact served A/B as PERF-A039 | 2026-08-31 07:52 PDT |
| M1 Max native early-out27 v2, deterministic served `6237+128`, real 131K pools | separate affine gate/up **18.845 tok/s** | separate affine gate/up **18.845 tok/s** | materialized fused rows reached **18.782 tok/s** (**-0.334%**) with identical output; rejected | process-isolated `bench_openai_stream.py --input-tokens 6237 --output-tokens 128 --temperature 0 --skip-warmup` A/B | 2026-08-31 07:46 PDT |
| M1 Max native early-out27 v2, deterministic decode after a 6,237-token history | growing concatenated BF16 K/V **17.923409 tok/s** | reusable power-of-two BF16 K/V **19.151623 tok/s** | **+1.228215 / +6.853%**; exact 128-token digest retained | direct native engine, 32 warm tokens plus 128 timed tokens | 2026-08-31 06:10 PDT |
| M1 Max native early-out27 v2, exact-prefix continuation with 16 new prompt tokens and 32 generated tokens | fresh full prefill **3.063133 s / 10.4468 tok/s** | retained native state **1.927515 s / 16.6017 tok/s** | **-37.074% latency / +58.916% throughput**; output lists identical | direct native engine exact-prefix/fresh A/B | 2026-08-31 05:58 PDT |
| M1 Max Qwen3.8-27B early-out27 selective q2, required sampled `128+256`, BF16 KV, radix enabled, real 131K pools | affine-q4/BF16 **19.2786 tok/s** | **20.0812 tok/s** | **+0.8026 / +4.163%**; all five samples clear 20; actual-work gate rejected | production-sampled stream command on `Qwen3.8-27B-MLX-Q2Expand-QKVZ-EarlyOut27-v2` | 2026-08-31 05:06 PDT |
| M1 Max Qwen3.8-27B early-out27 selective q2, frozen Codex xhigh turn at about 6.2K tokens | short sampled **20.0812 tok/s** | server telemetry **~19.2 tok/s** | **~-0.88 / -4.4%**; continuation Metal OOM | Codex 0.151.0 strict-config ephemeral tool round trip with explicit 131,072 window and xhigh reasoning | 2026-08-31 05:18 PDT |
| M1 Max Qwen3.8-27B radix continuation after a 6,257-token first turn, real 131K pools | early-out27 v2 **Metal OOM** | full affine-q2 **2.04 new tok/s** on the prefix-hit request | q2 residency survives; missing MLX auxiliary-state COW forces a 6,144-token recompute | bounded two-request OpenAI replay with `SGLANG_MLX_CACHE_LIMIT_GB=1`, 512-token chunks, and five auxiliary slots | 2026-08-31 05:39 PDT |
| M1 Max Qwen3.8-27B full affine-q2, required sampled `128+256`, BF16 KV, real 131K pools | affine-q4/BF16 **19.2786 tok/s** | **20.9032 tok/s** | **+1.6246 / +8.427%**; arithmetic and tool behavior rejected | same production-sampled stream command on immutable q2 revision `33b90b6...` | 2026-08-31 04:04 PDT |
| M1 Max Qwen3.8-27B q2-linear-attention plus q2-gate/up mixed artifact, required sampled `128+256`, BF16 KV, real 131K pools | affine-q4/BF16 **19.2786 tok/s** | **20.1482 tok/s** | **+0.8696 / +4.511%**; exact tool gate rejected | same production-sampled stream command on `Qwen3.8-27B-MLX-Q2GDN-Q4Anchors-v1` | 2026-08-31 04:18 PDT |
| M1 Max Qwen3.8-27B affine-q4, exact `128+256` deterministic MLX dependency gate, real 131K pools, q4 KV | MLX 0.32.0 **19.0468 tok/s** | MLX 0.32.2 **19.1432 tok/s** | **+0.0964 / +0.506%** | `.venv/bin/python scripts/windows/bench_openai_stream.py --model qwen3.8-27b-iq2 --input-tokens 128 --output-tokens 256 --temperature 0 --skip-warmup --timeout 600` | 2026-08-31 03:23 PDT |
| M1 Max Qwen3.8-27B affine-q4, exact `128+256` deterministic MLX lane, real 131K pools | q4 KV **19.1432 tok/s** | BF16 KV **19.4274 tok/s** | **+0.2842 / +1.485%** | same exact deterministic stream command; only `--mlx-kv-cache-bits 4` omitted | 2026-08-31 03:49 PDT |
| M1 Max Qwen3.8-27B affine-q4, required sampled `128+256` MLX lane, real 131K pools | q4 KV **18.9904 tok/s** | BF16 KV **19.2786 tok/s** | **+0.2882 / +1.518%** | same stream command with `--temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5` | 2026-08-31 03:49 PDT |
| M1 Max Qwen3.8-27B affine-q4, direct 32-warmup plus 256-token target loop | q4 KV **19.513158 tok/s** | BF16 KV **19.643293 tok/s** | **+0.130135 / +0.667%** | inline `mlx_lm.generate_step`, 128-token input, 32 warm tokens, 256 timed tokens | 2026-08-31 03:46 PDT |
| M1 Max Qwen3.8-27B affine-q4 native C++ graph, exact `128+256` deterministic server candidate | selected Python MLX q4-KV **19.1432 tok/s** | **19.2990 tok/s** | **+0.1558 / +0.814%**; sampled semantics gate open | same exact deterministic stream command under `SGLANG_USE_MLX_NATIVE_GRAPH=1` | 2026-08-31 03:44 PDT |
| M1 Max Qwen3.8-27B IQ2_XXS, exact `12+256` fixed-decode scoreboard | unqualified cross-machine q4 entry deleted | **14.661356 tok/s aggregate; 14.671473 best hit** | route-neutral local Q2 authority | pinned llama.cpp build 10547 command in `BENCHMARK.md` | 2026-08-23 07:46 PDT |
| M1 Max Qwen3.8-27B IQ2_XXS, historical native SGLang Rust `/generate` baseline, exact `12+256` | llama.cpp **14.661356 tok/s** aggregate | **7.001584 tok/s aggregate; 7.015010 best hit** | **-52.2446%; reference is 2.094006x faster** | exact launch and request in the 2026-08-23 08:05 experiment-log entry | 2026-08-23 08:05 PDT |
| M1 Max Qwen3.8-27B IQ2_XXS, historical PERF-A016 native SGLang Python `/generate`, exact `12+256` | **7.009167 tok/s** matched disabled-kernel control | **8.586948 tok/s aggregate; 8.591773 best hit** | **+22.510241%; llama.cpp reference is 1.707400x faster** | final PERF-A016 launch and request in the 2026-08-23 11:31 experiment-log entry | 2026-08-23 11:31 PDT |
| M1 Max Qwen3.8-27B IQ2_XXS, current PERF-A021 native SGLang `/generate`, exact `12+256` | **8.515065 tok/s** matched generic-Q2_K control | **9.189086 tok/s aggregate; 9.194647 best hit** | **+7.532647% matched; +7.012249% over PERF-A016** | fixed command and raw windows in `BENCHMARK.md` and the 2026-08-23 16:25 experiment-log entry | 2026-08-23 16:25 PDT |
| M1 Max Qwen3.8-27B IQ2_XXS, current PERF-A021 reasoning-enabled stream, exact `128+256` | new explicit four-metric Apple baseline | **22.945718 prompt / 9.156675 generation tok/s; 5.578383 s TTFT; 33.426973 s E2E** | five cache-flushed requests; exact counts and one reasoning digest | `.venv/bin/python scripts/windows/bench_openai_stream.py --model qwen3.8-27b-iq2 --input-tokens 128 --output-tokens 256 --temperature 0 --skip-warmup --timeout 600` | 2026-08-23 16:25 PDT |
| M1 Max Qwen3.8-27B IQ2_XXS, current PERF-A021 capacity | historical PERF-A016 exact `32761+1` pass | **32,768-token BF16 pool; exact `32761+1` passed; 18.942 prompt tok/s; 1729.565719 s TTFT; 1729.565822 s E2E** | capacity preserved on current source | same client with `--input-tokens 32761 --output-tokens 1 --temperature 0 --skip-warmup --timeout 7200` | 2026-08-23 16:25 PDT |
| M1 Max Qwen3.8-27B IQ2_XXS, historical PERF-A021 read-only Codex Responses tool gate | historical PERF-A016 gate on Codex 0.149.0 | **Codex 0.151.0; one `pwd`; consumed result; exact `CODEX TOOL READY`; 21,537 input / 413 output / 379 reasoning tokens** | repaired profile/catalog hashes exercised by one read-only shell round trip | executed command and raw events in the 2026-08-30 18:29 experiment-log entry | 2026-08-30 18:29 PDT |
| M1 Max Qwen3.8-27B IQ2_XXS, isolated-home strict Codex workspace-write gate | profile-overlay scratch gate with mutable lower config | **one first-attempt `file_change`; exact `QWEN38 ISOLATED WRITE READY`; 2,670 input / 115 output / 39 reasoning tokens** | dedicated pinned `CODEX_HOME`; unified-exec catalog; root-owned empty `/var/empty` target; separate trusted-repository AGENTS prompt diagnostic; exact 34-byte file/hash; no `-p`, `-c`, `--sandbox`, reasoning, or capacity override; bundle/global hashes stable pre/post | fixed command in `notes/benchmark-contract.md`; exact events in the 2026-08-30 20:46 experiment-log entry | 2026-08-30 20:46 PDT |
| M1 Max Qwen3.8-27B IQ2_XXS, synchronized batch-one decoder-layer profile | separate exact-request wall time **142.824820 ms/completion token** | **132.593 ms/token** topology projection from stable profiled layers | **10.232 ms/token cross-run numerical difference; no outside-layer attribution** | one-shot `SGLANG_MPS_PROFILE_LAYERS=1 SGLANG_MPS_PROFILE_STAGES=1` launch in the 08:14 experiment-log entry | 2026-08-23 08:14 PDT |
| Qwen3.8-27B IQ2_XXS, native-MPS `17408x5120` batch-one projection | 1.176875 ms matched generic | **0.516000 ms** | **-0.660875 ms / -56.16%** | `.venv/bin/python benchmark/mac/bench_mps_gguf_quant.py $IQ2_GGUF --tensor blk.8.ffn_gate.weight --batch-size 1 --warmup 8 --iterations 25` | 2026-08-20 22:25 PDT |
| Qwen3.8-27B IQ2_XXS, native-MPS Q5_K `248320x5120` head at batch one | 19.659291 ms matched generic | **3.754625 ms** | **-15.904666 ms / -80.90%; 5.24x** | `.venv/bin/python benchmark/mac/bench_mps_gguf_quant.py $IQ2_GGUF --tensor output.weight --batch-size 1 --warmup 8 --iterations 25` | 2026-08-20 22:52 PDT |
| Qwen3.8-27B IQ2_XXS, native-MPS 48-layer F32 b/a projection sweep | 7.296667 ms selected custom Metal | **2.159000 / 2.051708 ms** native `torch.mm` A/B arms | **-70.41% / -71.88%; 3.38-3.56x** | `.venv/bin/python benchmark/mac/bench_mps_dense_ba.py $IQ2_GGUF --warmup 4 --iterations 9` | 2026-08-20 23:43 PDT |
| Torch-native MPS GQA partial extend, `4096+4096`, FP32 | 542.376416 / 641.256125 ms padded-query controls | **176.066500 ms** lower-right-causal source path | **-67.54% / -72.54%; 3.08-3.64x** | `.venv/bin/python benchmark/mac/bench_mps_sdpa_extend.py --prefix-len 4096 --extend-len 4096 --warmups 1 --repeats 5` | 2026-08-21 00:15 PDT |
| Native Metal BF16 GQA EXTEND, isolated `E=17,L=131072` | dense MPS SDPA **424.528292 ms**, **+8,088.515625 MiB** driver residency | final-source **137.906625 ms** median, **+0 MiB** measured driver residency | **-67.52% latency; -8,088.515625 MiB residency** | raw `_extension().extend_gqa_bf16` harness recorded in the 2026-08-23 14:15 experiment-log entry | 2026-08-23 14:15 PDT |
| Native Metal BF16 GQA EXTEND, consecutive-cache direct load, `E=17,L=131072` | **148.002792 ms** matched same-map forced-staged median | **66.553042 ms** median, **+0 MiB** measured current/driver residency | **-81.449750 ms / -55.03%** | raw `_extension().extend_gqa_bf16` A/B/A harness recorded in the 2026-08-23 14:38 experiment-log entry | 2026-08-23 14:38 PDT |
| Native Metal BF16 GQA EXTEND, register-published run classifier, `E=17,L=131072` | **66.577542 ms** matched PERF-A018 median | **65.493667 ms** median, **+0 MiB** measured current/driver residency | **-1.083875 ms / -1.628%** | raw `_extension().extend_gqa_bf16` candidate/checkpoint/candidate harness recorded in the 2026-08-23 14:47 experiment-log entry | 2026-08-23 14:47 PDT |
| Native Metal BF16 GQA EXTEND, hoisted cache-run bases, `E=17,L=131072` | **65.687333 ms** matched PERF-A019 median | **63.886417 ms** final-source median, **+0 MiB** measured current/driver residency | **-1.800916 ms / -2.742%** | raw `_extension().extend_gqa_bf16` candidate/checkpoint/candidate harness recorded in the 2026-08-23 14:56 experiment-log entry | 2026-08-23 14:56 PDT |
| Torch-native MPS decode at unsupported physical pool/dtype boundaries | Runtime error for BF16 or more than 7,936 cache rows | **SDPA fallback, max error 0** at BF16/32,769 and FP32/7,937; fused FP32/7,936 preserved at `2.5331974e-07` | long-pool decode admitted without widening the native kernel contract | `.venv/bin/python benchmark/mac/test_mps_decode_fallback.py --cache-slots {32769,7937,7936} --cache-dtype {bfloat16,float32,float32} --seq-len 257` | 2026-08-21 00:24 PDT |
| Qwen3.8-27B IQ2_XXS, native-MPS `17408x5120` large-batch projection | 65.578125 / 1971.539875 ms at batch 128 / 4096 | **4.277125 / 124.838125 ms** | **-93.48% / -93.67%; 15.33x / 15.79x** | `.venv/bin/python benchmark/mac/bench_mps_gguf_quant.py $IQ2_GGUF --tensor blk.8.ffn_gate.weight --batch-size {128,4096} --warmup 1 --iterations 5` | 2026-08-23 07:06 PDT |
| Qwen3.8-27B IQ2_XXS, 32K/BF16 required sampled `128+32` served workload | 8.2942 generation tok/s selected FP32/fused restart mean | **6.963 prompt / 6.772 generation tok/s** | BF16 fallback generation is 18.35% below the selected short-pool mean; long-pool execution is functional | same sampled command on the 32,768 context/token-pool launch with BF16 KV | 2026-08-21 00:36 PDT |
| Qwen3.8-27B IQ2_XXS, deterministic `128+32` served workload | 6.979 prompt / 3.1858 generation tok/s | **7.0444 prompt / 8.4406 generation tok/s** | **+0.937% / +164.94%** | `.venv/bin/python scripts/windows/bench_openai_stream.py --model qwen3.8-27b-iq2 --input-tokens 128 --output-tokens 32` | 2026-08-20 23:42 PDT |
| Qwen3.8-27B IQ2_XXS, required sampled `128+32` served workload | 7.0562 generation tok/s after PERF-A002 | **8.3094 / 8.2942 tok/s** in two PERF-A009 restart windows | **+4.59% / +4.26% over matched PERF-A011 windows** | same command with `--temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5` | 2026-08-20 23:59 PDT |
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
| PERF-042 exact `199000+16` prompt window | 3048.086 tok/s record | **3209.728 mean / 3205.270 worst / 3216.299 best** | **+161.642 / +5.303% mean** | page-aligned FlashInfer prefix prefill; five cache-flushed exact scores | 2026-08-21 03:50 PDT |
| PERF-042 exact `199000+16` TTFT window | 65.286869 s record | **61.999103 s mean / 62.085254 s worst** | **-3.287766 s / -5.036% mean** | same five exact requests; every request below 64.20 s | 2026-08-21 03:50 PDT |
| PERF-042 exact `199000+16` E2E window | 65.420204 s record | **62.153173 s mean / 62.240542 s worst** | **-3.267031 s / -4.994% mean** | same five exact requests; every request below 64.35 s | 2026-08-21 03:50 PDT |
| PERF-042 exact `199000+16` short generation | 112.499 tok/s record | 98.029 mean / 86.371 worst / 112.151 best | -14.470 / -12.862% mean; noisy 15-interval metric | same five exact requests; generation target still open | 2026-08-21 03:50 PDT |
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
| PERF-062 accepted exact `199000+16` record | 3048.086 prompt / 112.499 generation tok/s | **3078.058 / 114.617 tok/s** | **+29.972 / +0.983% prompt; +2.118 / +1.883% generation** | launcher-default selective checkpoint, chunk 7680, native draft-k1 q, gate/up hybrid Marlin | 2026-08-21 13:48 PDT |
| PERF-062 accepted TTFT/E2E | 65.286869 / 65.420204 s | **64.651152 / 64.782022 s** | **-0.635717 / -0.638182 s** | same exact request, exact `199016`, `finish_reason=length` | 2026-08-21 13:48 PDT |
| PERF-062 independent no-override launch | 3048.086 / 112.499 tok/s prior record | **3052.437 / 114.053 tok/s** | all four prior metrics beaten again | `.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1`, process-selected seed | 2026-08-21 13:48 PDT |

The former Mac Pro Q4 target and record are retired from the active scoreboard.
Their detailed entries below remain chronological experiment evidence.

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
| PERF-001 | On Mac Pro/W6900X, add a Q4_0 batch-8 Metal specialization so steady batch-8 decode reads each packed matrix once instead of twice. | `gguf_q4_0.mm` Q4_0 kernel and host dispatch | Rejected | Correct but regressed representative MLP Q4_0 median from `0.345 ms` to `0.778 ms`; removed. See `FAILED_PATHS.md`. |
| PERF-002 | Fuse or remove remaining GDN pack, normalization, and reorder launches. | Qwen3.5 GDN MPS path and native Metal extension | Pending profile | Production path launches native packing and gated-norm/reorder around native recurrent attention for most decoder layers. |
| PERF-003 | Reduce full-vocabulary PyTorch sampling overhead on MPS. | `sampler.py` / native sampling | Pending profile | Every decode step performs top-p sampling over the full vocabulary; impact relative to model kernels remains unmeasured. |
| PERF-004 | Remove proven-redundant `.contiguous()` conversions and metadata copies in native MPS wrappers. | `mps/ops.py`, attention and GDN callers | Pending trace | Calls are production-reachable, but views may already be contiguous and therefore free. |
| PERF-005 | Wire the dormant Metal Q/K norm + RoPE + QKV/gate preparation kernel into full-attention layers. | `qwen3_5.py`, `gguf_q4_0.mm` | Rejected | Isolated batch-8 preparation fell from `2.225 ms` to `0.162 ms`, but clean end-to-end median regressed from `32.309` to `30.680 TPS`; production wiring removed. |
| PERF-006 | Reduce Q5_K/Q6_K batch-8 accumulator pressure by reusing smaller or batch-24 tiles. | `gguf_q4_0.mm` quantized matmul dispatch | Rejected | Q5_K tile-4 regressed `0.809 -> 0.888 ms`; Q6_K tile-4 regressed `27.227 -> 32.446 ms`; batch-24 vec4 regressed Q6_K to `34.166 ms`. |
| PERF-007 | Vectorize Q6_K dequantization across four adjacent weights with four two-request SIMD subgroups. | `gguf_q4_0.mm` Q6_K batch-8 kernel | Retained | LM-head microbenchmark improved `27.227 -> 9.186 ms` (`35.7 -> 105.7 GiB/s`); Q6_K reference relative error `4.64191e-07`. |
| PERF-008 | Apply the same vec4/subgroup geometry to the repeated Q5_K GDN output projections. | `gguf_q4_0.mm` Q5_K batch-8 kernel | Retained | Representative projection improved `0.809 -> 0.356 ms` (`24.9 -> 56.6 GiB/s`); combined end-to-end median is `38.016 TPS`. |
| PERF-009 | On Mac Pro/W6900X, apply batch-subgroup reuse or alter unroll depth in the Q4_0 batch-eight kernel. | `gguf_q4_0.mm` Q4_0 kernels | Rejected | Four-subgroup vec4 regressed `0.350 -> 0.686 ms`; existing split kernel gave `0.390 ms`; unroll 2 and 8 gave `0.402` and `0.405 ms`. |
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
| PERF-037 | Fuse residual-add/Gemma RMSNorm directly into the following NVFP4 activation tuple. | Native SM120 dual-output norm/quant producer | Rejected; boundary-neutral | Bit-exact at M1/M3/M7000/M7680. The real captured norm+quant+gate/up-GEMM boundary moved 0.096704 -> 0.097152 ms/layer, projecting -0.0287 ms over 64 layers. PDL already hides the separate quantizer. The prototype was removed. |
| PERF-038 | Specialize the dominant M3 NVFP4 GEMM below CTA-M 128. | Native SM120 CUTLASS 64x32x256 cooperative/ping-pong schedules | Closed at compile-time architecture gate | Cooperative GEMM requires CTA-M >=128; ping-pong still requires the fixed 128-row NVFP4 scale TMA atom. Qualified M3 tactics already swap A/B and use the minimum supported CTA-N 32. No kernel launched. |
| PERF-039 | Fuse the two MTP Gemma norms and concatenation before the BF16 fusion projection. | Native SM120 two-CTA producer | Rejected below funding | Bit-exact through the dependent FC. M1 saved 1.248 us and M3 saved 2.080 us; combined draft-decode/draft-extend value is only about 0.0033 ms/cycle. The prototype was removed. |
| PERF-040 | Fuse the SM120 gate/up GEMM epilogue with compiled SwiGLU and NVFP4 packing. | Custom CUTLASS collective epilogue | Closed as a small change | Stock EVT cannot pair/halve output coordinates. Selected tactics are swap-AB DP, so even the proposed non-swap staged prototype cannot replace production; a distinct half-height collective is required. |
| PERF-041 | Replace dense AIR apply after top-k 20 with an exact-pivot sparse-support apply. | Native CUDA sampling transform; default-off Windows gate | Retained in `7cb4ed0796`; client gate failed | 15 CUDA plus 6 integration tests pass. Top-k+top-p fell 109.12 -> 78.12 us at M1 and 121.17 -> 86.72 us at M3. Final-source cycle median was 16.001 ms with control-identical output/acceptance; predecessor long generation averaged only 111.559 tok/s. |
| PERF-042 | Plan aligned ordinary-prefix FlashInfer prefill with the physical page size instead of page 1. | Native CUDA page-table builder plus FlashInfer page-64 dispatch | Retained in `afd5606077`; generation target open | All 25 exact-request shapes were bit-exact; eight focused/fast-plan tests and adversarial review pass. Five exact prompts averaged 3209.728 tok/s with every prompt/TTFT/E2E gate passing. |
| PERF-043 | Widen aligned draft proposal support from top-k 20 to 32. | Existing launcher/configuration | Rejected | Five-probe acceptance fell 2.217279 -> 2.173943 tokens/verify and mean latency increased; added q support diluted useful target overlap. |
| PERF-044 | Collapse draft q to top-k 1 for the greedy exact scoreboard. | Existing launcher/configuration | Rejected | First exact score remained 97.900 generation tok/s; three greedy acceptance probes averaged only 2.107020 tokens/verify. |
| PERF-045 | Narrow aligned draft proposal support from top-k 20 to 16. | Existing launcher/configuration | Rejected | Five-probe acceptance averaged 2.205710, below k20's 2.217279, with slightly worse mean latency. Static proposal-support sizing is closed. |
| PERF-046 | Override proposal-only top-p to 1.0 and skip q top-p renormalization. | Default-off graph and post-extend dispatch | Retained in `6b963eed05`; generation target open | After repairing routing across all proposal owners, AIR top-p fell from three to one launch/cycle. M3 mean/median/p90 improved 0.194/0.185/0.149 ms; acceptance rose slightly to 2.229702. |
| PERF-047 | Scale proposal-only additive penalties. | Temporary default-off graph dispatch | Rejected as no-op on workload | Correctly routed scales 0.75 and 0.0 reproduced the exact same proposal/output trajectory; the active workload exposed no leverage through this additive row. |
| PERF-048 | Overlap target ReplaySSM fold/conv commit with independent draft extend. | Default-off CUDA side stream with forward-stream rejoin | Rejected after interval attribution | 186.8/189.5 us fold overlapped graph 8, but graph 8 expanded 1.061 -> 1.237 ms from contention. Serial fold+extend was ~1.234 ms versus ~1.237 ms overlapped; apparent cycle gain was noise. |
| PERF-049 | Learn static root/depth proposal calibration from branch-exact p/q capture. | Existing diagnostic plus offline gamma/rank/token fitting | Static calibration rejected; queue fixed in `4d6782121e` | Two chronological corpora (151 and 239 records) found maximin gamma at identity/negligible +0.00013 expected length; rank/token fits regressed held-out data. Queue capacity 8 -> bounded 64 prevents diagnostic backpressure. |
| PERF-050 | Use greedy draft top-k 1 on the temperature-zero exact profile. | Existing proposal configuration plus page64/top-p1 stack | Retained greedy long-window candidate; exact16 target open | Exact199K+512 improved 116.549 -> 123.049 tok/s with identical digest; exact-context acceptance reached 2.426540. Five exact16 scores still averaged 98.478 tok/s because seven long-context cycles cost ~19.90ms each. |
| PERF-051 | Tune XQA SM count and PDL at exact199K shape. | Existing FlashInfer XQA controls | Rejected | Counts below all170 changed output and saved only a few us/call; PDL true/false was bit-exact and timing-neutral. |
| PERF-052 | Reopen M4 under greedy k1 proposal law. | Existing 3-step/4-row configuration | Rejected | Exact16 still required seven verify cycles and emitted 2.285714 tokens/cycle; the extra target row/draft step cannot improve the discrete gate. |
| PERF-053 | Reopen the retained device-resident cycle under greedy k1. | Composite draft-extend/next-draft CUDA graph | Rejected | Exact199K+16 generation was 97.730 tok/s with the control digest, inside and slightly below the 98.478 tok/s control window. |
| PERF-054 | Tune SM120 XQA structural constants. | FlashInfer native CUDA XQA mainloop | Rejected | Valid V-buffer/V-tile variants saved at most 0.960 us/call; wider K with one buffer was nondeterministic, two buffers exceeded shared memory, row-max 0 was slower, and CTA-x 2 was invalid. |
| PERF-055 | Rotate stock NVFP4 KV with the native Hadamard kernel. | Native Hadamard plus existing NVFP4/XQA | Rejected | Rotation left synthetic attention error effectively unchanged (~0.1018 relative L2), while native NVFP4 XQA's measured ceiling was only 0.520 ms/cycle before transforms and failed semantics. |
| PERF-056 | Learn a hidden-conditioned rank correction for greedy q20. | Default-off exact-q/hidden diagnostic and offline blocked fitting | Diagnostic retained; model rejected | Exact q20 support contained every required exact16 target token and a perfect oracle finishes in six cycles, but the first PCA-linear rank head had 0% locked minority accuracy and did not beat draft argmax. |
| PERF-059 | Emit greedy draft-k1 q directly with native CUDA. | Native Windows argmax/one-hot q producer | Promoted additive win in `03ba3d2e27` | Proposal construction fell 73-87 -> 3.7-3.9 us. Matched long generation improved 122.352 -> 123.559 tok/s (+0.987%); independent restart averaged 123.831. |
| PERF-061 | Retune all exact target FP4 tactics and disable PDL. | Six SM120 target GEMM families | Rejected | PDL-off regressed weighted shape timing; exact tactic pair projected 0.361 ms but moved real long generation only 123.831 -> 123.972 tok/s (+0.114%), inside noise. |
| PERF-062 | Use Cutlass for prefill and in-place Marlin for target gate/up decode. | 64 target gate/up projections plus native draft-k1 q | Promoted launcher default in `03ba3d2e27` | Accepted exact score **3078.058/114.617**, TTFT **64.651152 s**, E2E **64.782022 s**; independent no-override launch **3052.437/114.053**. Both beat all four prior record metrics. |
| PERF-A001 | Keep Q2_K, Q4_K, IQ2_XXS, and IQ1_M GGUF weights packed through native Metal matmul and embedding kernels. | MPS GGUF loader, Metal kernels, convolution-state selection | Retained in `7740cae691` | Actual Bartowski IQ2 rows pass batch 1/3/4/8 parity; prior immutable IQ1 evidence is retained in the experiment log. Packed serving reduces model residency and makes the retained Q2 checkpoint runnable. |
| PERF-A002 | Reuse each IQ2_XXS unpack and input load across multiple output rows at batch one. | `gguf_q4_0.mm` IQ2_XXS kernel and host dispatch | Retained in `16b2bf7a06` | Matched generic/candidate medians reached `1.176875 -> 0.516000 ms`; complementary long-K projection reached `1.234667 -> 0.540791 ms`. Two served restart windows preserved exact behavior while deterministic generation rose `3.309 -> 7.1748 tok/s`. Multi-batch prefill remains on its established kernels. |
| PERF-A003 | Bound quantized-KV prefill score residency by tiling independent query rows above a measured threshold. | MLX quantized attention wrapper and environment surface | Retained source mechanism in `1271610e0b`; safety follow-up `ea983f3120`; Mac Pro evidence only | Helper and full Qwen3.5 wrapper parity pass. The cross-machine timing and allocation measurements carry no M1 Max record standing; fresh dependency, parity, memory, and capacity gates are required before local use. |
| PERF-A004 | Separate GGUF checkpoint behavior from SGLang formatting before qualifying the retained IQ2 route. | native GGUF tokenizer, reasoning parser, Qwen3 Coder tool parser | Correctness fix retained in `8879ed3d01`; later selected route qualified | GGUF USER_DEFINED reasoning/tool markers now encode atomically without becoming skippable specials. Live gates return final `703`, exact thinking-off `READY`, one parsed multiply call, and a preserved tool-result continuation. PERF-A016 later closes performance, capacity, and client qualification. |
| PERF-A005 | Bound MLX's recycled Metal buffer cache during monotonically growing long-context prefill. | `SGLANG_MLX_CACHE_LIMIT_GB`, outer prefill boundary, former 262K profile | Mac Pro mechanism survey only | The source supports a pre-load cache cap. The deleted cross-machine profile supplies no M1 Max capacity or performance record. |
| PERF-A006 | Compare the retained IQ2 GGUF through a pinned current llama.cpp Metal server. | Supporting dependency, exact Apple benchmark and behavior gates | Current route-neutral M1 Max Q2 reference | Official build 10547 at `749f688f` completed exact `12+256` at **14.661356 tok/s aggregate** with a **14.671473 tok/s** best hit. The matched native SGLang baseline is **7.001584 tok/s**, leaving this reference **2.094006x** faster. |
| PERF-A007 | Eliminate per-forward materialization of heterogeneous merged GGUF projection shards. | `GGUFLinearMethod` storage/apply and packed Metal storage offsets | Retained in `13bea403d6` | One compact MPS backing allocation removes 40 copies / 478.125 MiB of packed-weight materialization per full forward. Five exact `128+32` samples improved generation **3.1858 -> 3.309 tok/s** (+3.867%), prompt **6.979 -> 7.0224** (+0.622%), and reported weight residency **10.03 -> 9.03 GB**, with the identical output digest. |
| PERF-A008 | Replace native MPS score-array GQA with fixed-memory online or split-K softmax. | `decode_gqa` Metal kernel, KV indirection, cache write | Capacity-critical native candidate | Current native decode allocates `(cache_slots + 256) * 4` bytes of threadgroup scratch and rejects `cache_slots > 7936`. Fixed-size online state removes the configured-pool occupancy cost and this absolute context ceiling; split-K adds long-history parallelism. |
| PERF-A009 | Specialize the 96x5120 batch-one F32 b/a projection, then remove GDN input packing only if it remains funded. | GGUF F32 dispatch and Qwen3.5 GDN producer/consumer boundary | Retained in `4d1641fdcd`; two windows passed; later route qualified | Native `torch.mm` reduced the actual 48-layer sweep from `7.296667` to `2.159000/2.051708 ms`. Deterministic served generation improved `8.0284 -> 8.4406 tok/s`; sampled restart windows reached `8.3094/8.2942 tok/s`, with behavior and multi-batch fallback intact. PERF-A016 later passes exact context and Codex gates. |
| PERF-A010 | Add native affine-q4 quantized SDPA and GQA-aware KV reuse to MLX. | Supporting MLX C++/Metal dependency | Mac Pro-only design; outside the M1 Q2 lane | The source design remains recoverable. It carries no local performance, capacity, or record authority. |
| PERF-A011 | Vectorize the Q5_K vocabulary head across four eight-lane row cohorts per SIMDgroup at batch one. | `gguf_q4_0.mm` Q5_K kernel and aligned host dispatch | Retained in `b19cf4acf3` | Matched candidate/control/candidate medians were `3.737000 / 19.659291 / 3.754625 ms`. Five-run deterministic served generation rose `7.1748 -> 8.0284 tok/s` (+11.90%) with the exact digest; a committed restart and two required-sampling windows passed. |
| PERF-A012 | Run torch-native partial extend on only the new query rows with an offset causal mask. | Shared torch-native SDPA extend mechanism | Retained in `210a214c12`; exact long-context gate passed | Exact source A/B at `4096+256` changed `97.995583/97.847500 -> 12.608125 ms`; at `4096+4096` it changed `542.376416/641.256125 -> 176.066500 ms`. Outputs were exact on MPS, six focused CPU cases cover causal/ragged/fallback behavior, and the selected route later completes exact `32761+1`. |
| PERF-A013 | Admit BF16 and long physical pools through the established torch-native decode fallback while preserving eligible fused Metal decode. | Torch-native MPS decode dispatch | Retained in `b2b8ab4af8`; full-model 32K gate passed | The pre-change BF16/32,769 and FP32/7,937 probes raised at the native binding. Both now reach pool-write plus SDPA with zero observed error; the FP32/7,936 boundary remains fused with maximum error `2.5331974e-07`. Nine focused CPU tests pass, and the selected route later completes exact `32761+1`. |
| PERF-A014 | Reuse each quantized weight tile across large activation batches with native simdgroup matrix multiplication. | `quant_matmul` Metal kernels and host dispatch | Qualified in signed `1676c71bed` | The FP32 64-output by 32-batch path changes the actual IQ2_XXS `17408x5120` median **70.074833 -> 4.250250/4.277125 ms** at batch 128. Matched served exact-`128+1` prompt changes **7.0234 -> 22.8814 tok/s** across a control and two independent default windows; multi-chunk prefill, parity, behavior, and cleanup gates pass. |
| PERF-A016 | Reuse each activation fragment across two eligible Q4_K output rows inside the retained mixed-format IQ2_XXS/Q2 checkpoint. | `quant_matmul` Metal kernel and aligned compact-view dispatch | Qualified in signed `52b5326d8e` | Final Python A/B changes exact-`12+256` generation **7.009167 -> 8.586948 tok/s** (**+22.510241%**); an independent restart reaches **8.578205 tok/s**. Candidate/tail parity, exact `32761+1`, behavior, Codex Responses tool integration, and cleanup pass. `Q4_K` names the tensor family; record standing remains M1 Max Q2. |
| PERF-A021 | Reuse one activation fragment across four eligible Q2_K output rows. | `quant_matmul` Metal kernel, aligned batch-one dispatch, and generic environment control | Retained in signed `4dfa1ad3ef`; current Apple baseline | Actual Q2_K gate/down medians fall from about **1.07/1.09 ms** to **0.455/0.454 ms**. Matched exact-`12+256` generation changes **8.515065 -> 9.156475 tok/s**; the independent current window reaches **9.189086 tok/s**. Streaming Prompt/Generation/TTFT/E2E and current-source exact `32761+1` capacity are recorded in `BENCHMARK.md`. |
| PERF-A022 | Pick up current MLX batch-one decode improvements while holding model, pools, cache format, and output trajectory fixed. | Apple MLX dependency floor | Retained in signed `45b50cc4c3` | MLX 0.32.0 -> 0.32.2 changes five deterministic exact `128+256` q4-KV samples from **19.0468** to **19.1432 tok/s** (+0.506%) with identical output and reasoning digests. |
| PERF-A023 | Reduce decode attention cost by retaining BF16 K/V. | Qwen3.8-27B affine-q4 MLX attention cache | Current sampled baseline candidate | Five deterministic samples average **19.4274 tok/s** and five production-sampled samples average **19.2786 tok/s**, respective gains of 1.485% and 1.518% over q4 KV. Exact 131K allocation and actual-work gates remain open. |
| PERF-A024 | Use a smaller full-model affine 3-bit checkpoint to reduce weight bandwidth. | `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | Rejected | Three deterministic exact `128+256` samples average **17.958 tok/s**, about 6.2% below the selected affine-q4/q4-KV endpoint. See PERF-FA066. |
| PERF-A025 | Use MLX MXFP4 storage to reduce affine metadata and decode traffic. | `mlx-community/Qwen3.8-27B-mxfp4` | Rejected | Matched direct target-loop throughput is **18.581608 tok/s** versus **19.513158** for affine q4/q4 KV, a 4.774% loss. See PERF-FA067. |
| PERF-A026 | Execute the complete token graph through the repository C++ MLX engine. | `hardware_backend/mlx/native/qwen38_engine.cpp` | Deterministic candidate; sampled gate open | Five deterministic exact `128+256` server samples average **19.2990 tok/s**, +0.814% over selected Python q4 KV. The current C ABI returns argmax token ids, so sampled Codex behavior requires a native logits/sampling seam before promotion. |
| PERF-A027 | Use full-model affine q2 to reduce batch-one weight bandwidth. | Immutable q2 revision `33b90b60fd7ba16b668854e049bd65e22d6afddf` | Rejected | Five sampled server samples average **20.9032 tok/s**, while arithmetic and tool behavior fail. See PERF-FA068. |
| PERF-A028 | Reduce affine metadata with group-128 requantization. | In-memory q4 checkpoint module selections | Rejected | The broad 385-module screen reaches **19.781749 tok/s** in the direct BF16-KV loop, leaving too little margin for sampled server overhead. See PERF-FA069. |
| PERF-A029 | Preserve q4 semantic anchors around q2 linear-attention and MLP expansion projections. | Derived `Qwen3.8-27B-MLX-Q2GDN-Q4Anchors-v1` artifact | Rejected; narrower boundary active | Five sampled server samples average **20.1482 tok/s** and sampled arithmetic returns `703`; the exact tool gate emits two malformed calls. See PERF-FA070. |
| PERF-A030 | Quantize the measured raw-tool-safe boundary through the first 27 linear-attention output projections. | Derived `Qwen3.8-27B-MLX-Q2Expand-QKVZ-EarlyOut27-v2` artifact | Rejected unchanged; residency and long-prefix work active | Radix-enabled sampled `128+256` averages **20.0812 tok/s**, while real 6.2K decode is about 19.2 tok/s, tool continuations are unstable, and the frozen Codex continuation OOMs. See PERF-FA074. |
| PERF-A031 | Use a published long-context quality-aware q3/q4/q6 precision map. | Immutable YoozLabs revision `55c317fadb679431afef61ddd97a4ac2522ca420` | Rejected | Direct throughput is **18.553117 tok/s**, below the server floor before sampling overhead. See PERF-FA071. |
| PERF-A032 | Use group-32 q2 AWQ unchanged or as selective q4 overrides. | Immutable PocketAiHub revision `dcc3732f8c93ccf5580bf7a55e4ae639a40f194c` | Rejected | Full AWQ reaches **20.796459 tok/s** and fails raw tools; mixed forms reach **20.298/20.758** and produce empty or terminator-only output. See PERF-FA072. |
| PERF-A033 | Bound MLX's recycled Metal buffer cache before model load and radix continuation. | Existing `SGLANG_MLX_CACHE_LIMIT_GB` control | Rejected unchanged | A one-GiB cap keeps five short samples at **20.0646 tok/s** and still OOMs on the immediate 6,257-token continuation. See PERF-FA075. |
| PERF-A034 | Restore the radix-matched recurrent state before MLX continuation prefill. | Deferred Mamba COW handoff into `MlxAuxiliaryStatePool` | Active | The scheduler emits source/destination indices and the generic runner consumes them; the reachable MLX generation path does neither. Full q2 proves residency can survive and exposes a **2.04 new tok/s** full-prefix fallback. |
| PERF-A035 | Infer each stored affine projection width in the native checkpoint loader. | Native Qwen3.8 C++ weight loading | Retained in signed `42ee99493e` | Early-out27 v2 reaches **20.219228 tok/s** directly and **20.1186 tok/s** across five deterministic server samples with the Python-MLX token digest, arithmetic, and parsed tool call preserved. |
| PERF-A036 | Reuse exact native prompt state across strict-prefix request continuations. | Native attention/recurrent state and request boundary | Retained in signed `24686a37b1` | A 16-token suffix plus 32 generated tokens changes **3.063133 -> 1.927515 s** with identical output; a served 6,257-token continuation takes **0.416196 s**. |
| PERF-A037 | Replace per-token full-attention K/V concatenation with reusable power-of-two storage. | Native full-attention cache owner | Retained in signed `5ac91e2f22` | Exact 6,237-history decode changes **17.923409 -> 19.151623 tok/s** (+6.853%) with the same 128-token digest. |
| PERF-A038 | Split long-history native attention across custom Metal workgroups. | Native MLX decode attention | Rejected | The best tiled split arm reaches **19.117317 tok/s**, below the **19.151623** MLX SDPA control. See PERF-FA078/PERF-FA079. |
| PERF-A039 | Materialize gate/up affine rows and issue one quantized matmul per MLP. | Native Qwen3.8 target and MTP MLP owner | Rejected and removed | Adjacent deterministic `6237+128` serving changes **18.845 -> 18.782 tok/s** and reported available unified memory falls **28.92 -> 22.28 GB**, while output SHA-256 remains exact. See PERF-FA081. |
| PERF-A040 | Combine the two 48-row linear-attention b/a affine projections. | Native Qwen3.8 `Engine::gated_delta` owner | Rejected and removed | Adjacent deterministic `6237+128` serving changes **18.845 -> 18.511 tok/s** (-1.772%) with the same output SHA-256 and **28.89 GB** startup headroom. See PERF-FA082. |
| PERF-A041 | Add a standalone direct native Qwen3.8 benchmark with deterministic tokens and a stable digest. | C++ benchmark infrastructure | Retained | The synchronized `6237 / 32 warm / 128 timed` run reaches **19.116578 tok/s**, within **0.183%** of the signed **19.151623 tok/s** direct control, with matching digest `13eb9a7159a2612f`. |
| PERF-A042 | Fuse single-token causal convolution with its next-state window. | Native Qwen3.8 recurrent decode owner | Qualified and retained in signed `6ad2c58921` | Five adjacent exact long-history direct pairs improve mean decode **19.120031 -> 19.221589 tok/s** (+0.531%); matched five-sample serving improves **18.7616 -> 18.8914 tok/s** (+0.692%). An isolated BF16 C++ parity test matches MLX convolution and state output exactly at production width. |
| PERF-A043 | Reopen the existing native MTP-2 draft/verify route with exact acceptance telemetry. | Native Qwen3.8 C ABI and direct C++ benchmark | Rejected; telemetry retained | The 4-bit sidecar emits **1.888889 tokens/refill** and preserves digest `8ea2430e3fa3d56e`, while throughput changes **20.187703 -> 9.649960 tok/s** (-52.199%). The target's multi-token recurrent verification topology requires a separate execution-cost breakthrough before another sidecar/depth screen. |
| PERF-A044 | Fuse single-token residual addition with the following RMSNorm. | Native Qwen3.8 decoder-layer boundary | Qualified and retained in signed `4905d68370` | The dual-output Metal owner replaces 127 add/normalization pairs and preserves distinct residual storage. Direct long-history changes **19.222533 -> 19.310857 tok/s** (+0.459%); matched five-sample 131K serving changes **18.8286 -> 19.0484 tok/s** (+1.167%). Production-width, nonaligned, and 12-outstanding-output parity pass. |
| PERF-A045 | Fuse recurrent q/k RMS normalization and float scaling. | Native Qwen3.8 gated-delta decode owner | Qualified and retained in signed `b851d3c9de` | One dual-output Metal launch replaces four operations in each of 48 recurrent layers. Direct long-history improves **19.319062 -> 19.464738 tok/s** (+0.754%); matched five-sample 131K serving improves **19.0900 -> 19.1610 tok/s** (+0.372%). Production `16x128`, multi-simdgroup width 257, and 12-outstanding-output parity pass. |
| PERF-A046 | Fuse recurrent output RMS normalization with the SiLU gate. | Native Qwen3.8 gated-delta decode owner | Qualified and retained in signed `28174b3da2` | One exact Metal launch replaces the single-token RMSNorm, sigmoid, and elementwise gate chain in each of 48 recurrent layers. Direct long-history improves **19.469136 -> 19.515718 tok/s** (+0.239%); matched five-sample 131K serving improves **19.1260 -> 19.1548 tok/s** (+0.151%). Production `48x128`, nonaligned width 257, extreme activations, and 12-outstanding-output parity pass. |
| PERF-A047 | Fuse recurrent causal convolution with its BF16 SiLU. | Native Qwen3.8 gated-delta decode owner | Qualified and retained in signed `4c1bc4c1e3` | The existing convolution/state launch now reproduces both BF16 boundaries of the following sigmoid and multiply, removing two launches in each of 48 recurrent layers. Five adjacent long-history pairs improve **19.524068 -> 19.632483 tok/s** (+0.555%); matched five-sample 131K serving improves **19.1730 -> 19.2134 tok/s** (+0.211%). Production width 10,240, nonaligned width 257, extreme activation parity, and exact served reasoning output pass. |
| PERF-A048 | Concatenate full-attention q/k/v affine rows at load and issue fewer quantized products. | Native Qwen3.8 full-attention projection owner | Rejected and removed | Full q/k/v fusion screened at **20.721377944 tok/s** with digest `12bb3edf3d51feac`; q plus fused k/v screened at **20.672271140 tok/s** with digest `af06cc7ce5e094be`. Both diverge from exact control digest `8ea2430e3fa3d56e` because production row geometry changes MLX accumulation. See PERF-FA084. |
| PERF-A054 | Sample the full supported distribution and reuse exact prompt-boundary native state for Codex continuations. | Native Qwen3.8 sampler, reasoning bound, and recurrent/KV snapshots | Retained in signed `883da94`; behavior qualification active | The independent committed restart averages **20.1356 tok/s** with every sample above 20. A first xhigh tool turn passed; later request-order replay exposed process-global RNG ownership and low-bit structured-output instability. |
| PERF-A055 | Reinitialize the configured native sampling stream at each unrelated full request reset. | Native Qwen3.8 engine request boundary | Validated reproducibility fix; retained in this change | Five `6237+128` samples average **20.1556 tok/s**, all exceed 20, and all share SHA-256 `91dbc7056abfdc989aaee9e1d0f1fa3abc410637aabe144ac2ab0ea9bd97df3e`. Strict-prefix and prompt-snapshot continuations retain their ongoing stream. |
| PERF-A056 | Restore only the precision-critical recurrent projections from immutable Q4 weights. | Native Qwen3.8 checkpoint loader and gated-delta projections | Speed floor passed; behavior narrowing active | QKV-only restoration sustains **20.0222 / 20.0292 tok/s** across independent five-sample windows with every sample above 20. One xhigh turn passed cleanly; a post-window turn completed after one malformed extra call. QKV+Z passes a clean xhigh turn at **19.948 tok/s**, **0.052 tok/s** below the floor. |
| PERF-A057 | Integrate the Qwen3.8-specific DFlash2 draft into the native MLX C++ lane. | Native draft checkpoint loader, block proposal/selector, target verify, and accepted-state commit | Runtime and real-client behavior qualified; throughput active | The exact 175-tensor affine-W4 artifact loads through `Engine::load_mtp`; native capture, five draft layers, selector, exact p/q verification, and accepted-path commit complete one exact Codex xhigh tool turn. Direct sampled throughput is **9.300145 tok/s**, and real-client telemetry remains below 20. |
| PERF-A059 | Bound DFlash target/capture prefill inside the native engine and retain verified accepted-prefix state. | Native `Engine::prefill`, target capture, recurrent tape, full-attention logical commit, and affine small-batch QMM | Retained in signed `d57a6ac11c` | Internal 2,048-token units complete the 6.2K Codex prompt without changing the Python ABI. W2/W4 QMM parity passes long-K and wide-N cases; recurrent prefix replay is FP32 bit-exact for lengths 1--7; focused native engine suite passes 8 tests. |
| PERF-A060 | Partition M=8 affine-W4 verification products across SIMD groups. | Native Metal quantized-matmul owner for DFlash target verification | Retained in signed `b853514b5c`; real-client behavior passed | Two process-isolated samples average **11.242643 tok/s** versus adjacent control **9.304876**, a **20.825%** gain. Verify falls from about **305--306 ms** to **225.5--226.7 ms**; parity passes long-K, wide-N, and invalid-contract cases. |
| PERF-A061 | Qualify the M8 K-split path through a real Codex `xhigh` tool turn with real 131K pools. | Native DFlash server, Responses API, reasoning parser, tool parser, and Codex client | Behavior qualified; served-throughput work active | Thread `01a05c45-6b01-7a21-8a69-065170fd3402` executed exactly one requested command and returned exact `QWEN38_DFLASH2_READY`, exit zero. First prefill reached **95.05 tok/s** and reported generation intervals reached **17.20 tok/s**; live verify remains about **251--253 ms**. |
| PERF-A062 | Increase the exact M=8 affine-W4 K split from eight to sixteen SIMD groups. | Native Metal quantized-matmul owner for every reachable target and DFlash verification projection | Retained in signed `1e21aece56`; real-client behavior passed | Five no-trace samples average **17.651701 tok/s** versus the SG8 three-sample mean **11.242437**, a **57.010%** gain. Verify falls to **210.6--212.1 ms**. All five samples reproduce width 4.3, digest `3e40b8569af9555f`, and last token 735. |
| PERF-A063 | Qualify SG16 through sampled serving and a Codex `xhigh` tool turn with real 131K pools. | Native DFlash server, OpenAI streaming endpoint, Responses API, and Codex client | Behavior qualified; sustained-throughput work active | Exact `6237+128` reaches **15.336 tok/s** and exact token counts. Codex thread `01a05c5a-28a6-79f0-a526-efa19d961645` executes one command and returns exact final marker; one interval reaches **20.85 tok/s**. Live verify is **236--239 ms**. |
| PERF-A064 | Double the SG16 M8 affine-W4 output tile and reuse threadgroup storage across disjoint phases. | Native Metal quantized-matmul owner for every reachable target and DFlash verification projection | Retained in signed `3bae8a5e67`; real-client behavior passed | Five no-trace samples average **31.327334 tok/s**, **77.475%** above SG16/B16. Verify falls to **185.4--186.7 ms** direct and **211--214 ms** at 6.2K live history. The exact served sample reaches **15.328 tok/s** because its sampled acceptance trajectory differs. |
| PERF-A065 | Qualify SG16/B32 through sampled serving and a Codex `xhigh` tool turn with real 131K pools. | Native DFlash server, OpenAI streaming endpoint, Responses API, and Codex client | Behavior qualified; acceptance optimization active | Exact `6237+128` completes at **15.328 tok/s**, **109.106 prompt tok/s**, and exact 6,365 total tokens. Codex thread `01a05c69-215f-7fb0-a7f8-1425c9b2ae5a` executes one command, returns the exact final marker, and exits zero. |
| PERF-A066 | Load the official 81-tensor BF16 DFlash2 checkpoint directly. | Shared native QLinear execution and exact DFlash checkpoint loader | Compatibility retained in signed `6cf95442cc`; BF16 performance choice rejected | Dense BF16 loads and completes exact direct decoding. It reaches **9.044043 tok/s** versus adjacent affine-W4 **30.992508**, with width **2.114754** versus **6.684211** and about **42 ms** versus **26--32 ms** draft work. |
| PERF-A067 | Replace learned selector sampling with the official worker's greedy LM-head proposal rule. | Native DFlash proposal and exact rejection/residual sampler | Rejected and removed | Five direct samples misleadingly average **37.518731 tok/s** and width **7.9375**. The representative real `6237+128` request reaches only **9.512 tok/s**, **37.944%** below the learned selector. |
| PERF-A068 | Integrate the official Qwen3.8 DSpark v2 draft into the native MLX C++ lane. | Immutable draft artifact, native full-attention backbone, rank-256 Markov proposal, target verifier, and accepted-state commit | Native BF16/affine-W4 execution and correctness gates pass; sampled fidelity optimization active | The exact loaders, five-layer full-attention YaRN draft, sequential Markov proposal, exact dense-q verifier, and accepted-state commit run end to end. Affine-W4 reaches **10.050625 tok/s** direct and **11.242 tok/s** on the exact sampled served `6237+128` admission screen. |
| PERF-A069 | Apply the target top-k/top-p policy independently to every DSpark proposal row. | Native DSpark proposal distribution and exact dense-q verifier | Rejected and removed | Direct throughput fell **10.050625 -> 6.997461 tok/s** and width **2.428571 -> 1.6** because the independently filtered draft and target supports differ. See PERF-FA098. |
| PERF-A070 | Preserve DSpark `markov_w2` in BF16 inside the affine-W4 draft artifact. | C++ checkpoint converter, exact native linear loader, and immutable derived artifact | Rejected and removed | Direct throughput fell **10.050625 -> 7.044992 tok/s**, width **2.428571 -> 1.65**. Served throughput moved **11.242 -> 11.294 tok/s** on a different trajectory while artifact size rose **87.148 MiB**; this supplies no robust promotion signal. See PERF-FA099. |
| PERF-A071 | Execute the official trained DSpark confidence head and expose exact per-position survival telemetry. | Native DSpark loader, BF16 hidden/Markov feature projection, FP32 sigmoid, and existing speculative trace | Retained; cost-based scheduler work active | Direct trace preserves width **2.428571**, digest `5a38c7070d7badeb`, and last token 16 at **10.071235 tok/s**. The natural served request preserves exact baseline reasoning SHA-256 at **11.303 tok/s**. Confidence spans useful regimes and remains probabilistic under exact p/q sampling. |
| PERF-A072 | Generalize exact speculative verification to checked one-through-seven-token prefixes and measure every DSpark target batch. | Shared native p/q verifier, accepted-state commit, DSpark prefix selection, and trace telemetry | Retained as opt-in profiling/scheduling infrastructure; fixed short widths rejected | Default DSpark and DFlash reproduce their exact selected trajectories. DSpark target M=2 reaches **11.920231 tok/s**; M=3/4/5/6/7 reach **8.861866 / 8.353200 / 7.544333 / 4.104266 / 4.139508**, and selected M=8 reaches **10.025000**. The useful adaptive geometry is therefore binary M=2/M=8 under the current kernels. See PERF-FA100. |
| PERF-A073 | Budget each DSpark target verification from the trained current-block survival probabilities and measured M=2/M=8 cycle-cost ratio. | Native DSpark confidence head, shared exact verifier, dense-q prefix slicing, and opt-in scheduler | Retained opt-in at ratio **1.75**; fixed M=8 remains the default | Five real-131K-pool sampled `6237+128` requests average **13.6058 tok/s**, **+20.267%** over the adjacent **11.313 tok/s** control, with exact token counts and one shared reasoning/output SHA-256. Ratio 1.4 regresses to **10.916 tok/s** because live full/short cycle cost is about 1.77. See PERF-FA101. |
| PERF-A074 | Skip a bounded number of DSpark draft/verify probes after the confidence scheduler selects the low-value M=2 tier. | Native DSpark refill scheduler, exact target-only target/draft-context advance, and trace telemetry | Retained opt-in at **16** bypass refills; default remains disabled | The first five real samples average **16.6396 tok/s**, including a retained transiently contended **14.952** sample; five normal samples within six requests average **17.0514**. Every request completes exact 6,365 tokens with shared SHA-256 `1d1398eb...`. The selected setting remains **3.3604 tok/s** below the floor. See PERF-FA102/103. |
| PERF-A075 | Evaluate the trained DSpark confidence projection on the current normalized target hidden state before drafting. | Trace-only native target-state/Markov feature projection at the DSpark refill owner | Telemetry retained; threshold scheduler rejected and removed | Direct score-0.525 screens reached **33.222 / 32.788 tok/s**, yet the real request reached only **13.652 tok/s**. On the no-policy real trace, M=2/M=8 score ranges overlap and score versus accepted width has Pearson **0.141**. Faster-target DFlash mixing and a 64-column verifier tile also fail their gates. See PERF-FA104/105/106. |
| PERF-A076 | Calibrate the learned DFlash2 selector distribution while forwarding exact proposal q to rejection sampling. | Native DFlash selector score owner and shared exact p/q verifier | Retained opt-in at temperature **1.15**; identity remains default | Five consecutive real-131K-pool sampled `6237+128` requests average **15.8866 tok/s**, **+2.876%** over the adjacent identity-temperature mean **15.4424**. Every sample completes exact 6,365 tokens; candidate mean emitted width rises **3.878788 -> 4.031250**. Temperature 0.95 regresses the real request to **13.739 tok/s**; see PERF-FA107. |
| PERF-A077 | Submit the two exact M=8 affine gate/up projections together and test direct SwiGLU production. | Native SG16/B32 Metal QMM and shared target MLP owner | Rejected and removed | A sequential one-grid SwiGLU kernel regressed steady verify about **3.3 ms**. A two-plane paired grid preserved both outputs bit-exactly and moved five-sample direct mean only **26.943319961 -> 26.964966176 tok/s** (**+0.08034%**), with two adjacent pairs flat/slower. See PERF-FA108. |
| PERF-A078 | Budget DFlash2 verification from the current block's exact selected proposal probabilities. | Native sparse-q owner, common exact rejection verifier, checked prefix slicing, and trace telemetry | Retained opt-in at mean-q6 threshold **0.62**; full M=8 remains default | Five real-131K-pool sampled `6237+128` requests average **16.1776 tok/s**, **+1.763%** over the adjacent current-source **15.8974 tok/s** control. Each request uses 23 M=2 and 19 M=8 cycles, returns exact 6,365 tokens, and shares output SHA-256 `bdf9428e...`. Thresholds 0.55/0.65 reach **15.191 / 16.056 tok/s** on their admission screens; see PERF-FA109. |
| PERF-A079 | Bound target-only prefill residency with internal native chunks. | Native target-only prefill owner and full-Q4 long-prompt path | Retained opt-in at 2,048 tokens | The representative real-131K-pool request now completes at **19.300 tok/s** with exact token count and coherent reasoning; direct long-history decode reaches **19.586705 tok/s**. The absent setting preserves one-shot behavior. |
| PERF-A080 | Replace MLX batch-one QMV with a scalar-output native Metal geometry. | Full-Q4 affine-W4/G64 target projections | Rejected and removed | One-SIMD-per-output and subgroup-broadcast forms reach **10.456143330 / 7.773375044 tok/s** versus stock **20.339874670**. See PERF-FA111. |
| PERF-A081 | Attribute full-Q4 long-history decode at the shader-PC owner before another kernel change. | Metal GPU Counters Shader Timeline and native target process | Complete; matrix-tiled QMV branch active | 47,676 mapped samples place **88.470%** in MLX `affine_qmv_fast`, **3.904%** in two-pass SDPA, and **2.358%** in the recurrent update. Installed MLX 0.32.2 matches current official QMV source; upstream HEAD adds no QMV optimization after that tag. |
| PERF-A082 | Establish a runnable provenance-pinned Q5 checkpoint and served baseline. | Bartowski Q5_K_M/Q5_K_S source artifacts, pinned llama.cpp COPY conversion, native Metal GGUF execution, and sampled OpenAI serving | Derived Q5_K_S base retained; throughput optimization active | Converting only the unsupported 833.59 MiB Q5_K token embedding to F16 yields a 21,349,656,160-byte artifact with SHA-256 `c05a7778...fcfb`. It loads at 21.37 GB, warms, and completes exact sampled `128+32` at **5.746 tok/s** with reasoning preserved. Unchanged Q5_K_M and Q5_K_S fail at distinct current native boundaries; see PERF-FA112/113. |
| PERF-A083 | Reuse one activation fragment across two Q6_K output rows during batch-one decode. | Native Metal GGUF Q6_K matrix-vector kernel and guarded common quantized-matmul dispatch | Retained; first Q5 kernel win | Matched QKV/head medians improve **0.748917 -> 0.448125 ms** and **12.009166 -> 3.324625 ms**. Five sampled served requests average **7.052 tok/s**, **22.732%** above the 5.746 baseline; warmed-four average **7.144**. Actual-file optimized, odd-row fallback, and batch-eight parity pass. |
| PERF-A084 | Double the batch-one Q5_K output-row cohort while each lane decodes eight adjacent weights. | Native Metal GGUF Q5_K matrix-vector kernel and guarded common quantized-matmul dispatch | Retained; second Q5 kernel win | Reversed-order matched served windows improve `128+32` mean **7.1342 -> 7.1646 tok/s** and `128+128` median **7.456 -> 7.500 tok/s**. The 1,024-row K/V projections retain the prior mapping; direct candidate, tail, alignment, fallback, and full-model behavior gates pass. |
| PERF-A085 | Reuse each Q6_K row across all four verifier activations and produce sixteen rows per threadgroup. | Native Metal GGUF Q6_K exact-batch-four kernel and guarded common quantized-matmul dispatch | Retained; first served NEXTN verifier-kernel win | Matched head/QKV medians improve **29.166000 -> 6.121375 ms** and **1.442333 -> 0.588500 ms**. Five sampled same-GGUF NEXTN requests improve **3.3384 -> 3.7028 tok/s** while disabled-control acceptance is slightly higher. Candidate, 17-row tail, batch-three fallback, and batch-eight preservation checks pass. The unchanged three-step same-GGUF NEXTN configuration remains below target-only serving; see PERF-FA115. |
| PERF-A086 | Establish exact 131K capacity and residency behavior for the derived Q5 target. | M1 Max unified memory, 21.37 GB derived model, one FP32 Mamba slot, and exact 131,072-token BF16 KV pool | Capacity passed; unchanged residency rejected | Server startup, warmup, health, language-only metadata, and exact request pass. The 8.00 GB KV allocation leaves no reported headroom and one sampled `128+32` request falls to **0.102 tok/s** with active paging; see PERF-FA116. |
| PERF-A087 | Screen the existing FP8 KV configuration before native implementation. | PyTorch 2.11.0 MPS float8 storage/conversion and generic SGLang KV pool dtype | Framework route unavailable | Direct `torch.float8_e4m3fn` conversion raises the unsupported-MPS-dtype `TypeError`. Byte-backed float8 views and indexed gathers work, which enabled PERF-A089 to supply the missing native conversion; see PERF-FA117. |
| PERF-A088 | Reduce the only transformed Q5 artifact tensor while preserving the Q5_K_S body. | Pinned llama.cpp COPY conversion of `token_embd.weight` Q5_K to Q4_K, native embedding parity, behavior, and exact 131K residency | Retained smaller capacity artifact | The 19,522,020,960-byte artifact preserves 865 source tensors, loads at 20.00 GB, and leaves 0.99 GB after the exact 8.00 GB BF16 cache. Matched 131K generation improves **0.102 -> 0.315 tok/s** (**3.088235x**). A 1K smoke reaches **7.086 tok/s** and arithmetic/tools pass. |
| PERF-A089 | Supply the missing MPS E4M3FN value conversion while preserving SGLang's byte-backed generic KV pool. | Existing native Metal extension, narrow MPS `aten::_to_copy` specialization, contiguous/strided FP32-to-FP8 and FP8-to-FP32 kernels | Retained; exact 131K qualification active | 400,006 CPU-reference encodes and all 256 raw decodes match bit-exactly; offset/strided/empty and ordinary BF16 fallback checks pass. A 1K FP8 server warms and five sampled requests average **7.2614 tok/s** with arithmetic/tools preserved. No Python source changed. |
| PERF-A090 | Qualify the native FP8 conversion at the requested exact token pool. | Q4_K-embedding Q5 artifact, one FP32 Mamba slot, 131,072-token byte-backed FP8 K/V pool, sampled behavior | Retained exact-capacity selection; additional residency reduction active | Exact K/V allocation falls **8.00 -> 4.00 GB** and reported headroom rises **0.99 -> 6.99 GB**. Five sampled requests average **3.237 tok/s**, **10.276x** the matched BF16 result, while the full pool remains **55.42%** below the 1K FP8 mean. See PERF-FA118. |
| PERF-A091 | Move the Q5 target onto the compiled native MLX engine with a genuine affine five-bit checkpoint. | Pinned text-only Qwen3.8-27B affine-Q5/G64 snapshot, native target loader, and sampled direct decode | Artifact retained; native five-bit verifier optimization active | Revision `2568951b...c2f05` contains 498 U32 packed tensors, 1,349 BF16 tensors, and no vision tensors. Direct `128 / 32 / 128` reaches **16.322505765 tok/s**, **2.248x** the GGUF Q5 1K-FP8 mean. The unchanged Q4-tuned DFlash path reaches only **11.506669050 tok/s** because affine-Q5 M=8 falls through the generic verifier; see PERF-FA119. |
| PERF-A092 | Decode affine five-bit weights inside the shared M=8 K-split verifier. | Native Metal SG16/B32 target QMM, affine Q5/G64 bitstream, and exact p/q DFlash decode | Retained opt-in verifier win; target-only remains selected | Representative K/N parity passes at maximum error **0.03125 / 0.0625 / 0.107422**. M=8 falls from about **363--368 to 228--229 ms** and direct throughput rises **11.506669050 -> 13.721235888 tok/s** (**+19.246%**). The unchanged full-Q4 path preserves exact digest/width and improves in the adjacent sample. |
| PERF-A093 | Align the official five-bit MTP head with the selected M=8 verifier and preserve native sampling through exact p/q rejection. | Pinned Q5 MTP sidecar, recurrent one-layer draft, configurable two-through-eight-token block, and shared verifier | Exact sampled semantics retained; deterministic M=8 is an execution-cost probe | Matched deterministic blocks three/eight reach **17.919995235 / 31.380317186 tok/s** at widths **3 / 8**, preserving the target digest. Exact sampling completes through dense-q rejection, while the best calibrated screen reaches **6.380162135 tok/s**, width **2.285714286**. The existing Q5 target-only **16.322505765 tok/s** remains production-selected. See PERF-FA120/121. |
| PERF-A094 | Decode affine-five-bit batch-one target projections directly and reuse each activation fragment across output rows. | Native MLX Metal custom kernel, affine Q5/G64 packed weights, and guarded `QLinear` dispatch | Retained opt-in; repeated matched and production gates active | K/N `512/64`, `5120/128`, and `17408/32` parity pass at maximum error **0.03125 / 0.03125 / 0.0234375**. The selected four-SIMD/four-row/two-pack K-specialized kernel reaches **17.823163930 tok/s**, **+9.194%** over the original Q5 target result. It remains **2.176836070 tok/s** below the floor. See PERF-FA122/123/124. |
| PERF-A095 | Collapse each lane's two five-byte Q5 packs from four weight-load instructions to three exact packed loads. | Selected affine-Q5 batch-one Metal kernel only | Runtime-correct and neutral; unselected | Fresh-session parity passes at maximum error **0.03125 / 0.03125 / 0.0234375**. Representative A094/A095 microbenchmarks converge, and mixed full-model screens reach **19.010697650 / 19.032187257 tok/s** with identical digest `d0193f6d413b68c1`; the **0.113%** movement carries no promotion claim. |
| PERF-A096 | Reduce the aggregate Q5 weight stream with a quality-oriented mixed affine-Q4/Q5 policy. | Immutable `maglun/Qwen3.8-27B-MLX-Mixed-4.95bpw` text shards and the existing native per-tensor bit-width inference | Active Q5-class target; speed floor still open | Revision `596b8067...c8340` contains **16,645,209,088** text tensor bytes at **4.9510 aggregate BPW**, with 162 Q4 plus 240 Q5 affine matrices. The dense BF16 loader fix admits all recurrent b/a projections. Selected A094 Q5 QMV reaches a ten-sample mean **18.993383731 tok/s** with stable digest, leaving **1.006616269 tok/s** to 20. |
| PERF-A097 | Restore the Qwen3.8 MTP hidden-state contract and screen the published smaller draft head. | Native target/MTP handoff, matched one-layer sidecar, exact p/q sampling, and accepted-cache lifecycle | Loader retained; published Q4 head rejected for throughput | Signed `0da5c5a135` accepts the head's `mtp.` namespace and signed `1b328149c7` provides the post-norm seed. Mixed-target block-three sampled screens reach **9.122242179 tok/s**, post-norm **9.507655940**, and Q4 QMV **11.341033334**, all below target-only. See PERF-FA127. |
| PERF-A098 | Replace ten scalarized Q5 weight-byte loads with five aligned 16-bit loads. | Selected PERF-A095 Metal Q5 batch-one kernel | Offline AIR candidate; evaluate after PERF-A095 | Apple Metal 32023.883 lowers the three packed-byte ranges to ten `i8` loads. A `packed_ushort4` plus scalar `ushort` form lowers to five aligned-two `i16` loads. K-multiple-512 row strides and ten-byte lane offsets prove two-byte alignment. Runtime parity and throughput remain pending. |
| PERF-A100 | Reconstruct each Q5 lane as three continuous bit windows after five aligned 16-bit loads. | Selected affine-Q5 batch-one Metal kernel | Qualified and retained | Representative parity passes at maximum error **0.03125 / 0.03125 / 0.0234375** and every production-shape digest matches A094. Two independent balanced full-model windows improve **19.004038228 -> 19.122257628** and **19.021221323 -> 19.147557640 tok/s**; aggregate gain is **+0.122277858 / +0.643%**. |
| PERF-A101 | Pair adjacent SIMD lanes so one lane loads both ten-byte Q5 packs through aligned 32-bit words. | Selected affine-Q5 batch-one Metal kernel | Runtime-correct and rejected | Representative parity and all four production-shape digests match A100. The masked leader loads plus three shuffles regress the high-byte gate/down/attention shapes by roughly **5--9%** in paired order/reverse timing. See PERF-FA129. |
| PERF-A102 | Preserve committed decode-only MTP attention history across sampled speculative cycles. | Native Qwen3.8 MTP cache lifecycle, target-hidden pairing, and exact p/q verification | Opt-in source candidate prebuilt; fresh-session acceptance and throughput pending | The candidate retains the current pending-token entry, discards later provisional entries, and appends accepted draft tokens paired with the verifier's committed hidden prefix. Absolute target/MTP offset checks reset history after a target-only fallback. Prompt history and position-origin changes stay outside this arm. Strict C++20/O3 compilation and `git diff --check` pass from signed `711214b27c`; the candidate dylib is `b4f3b222...a5d`. |
| PERF-A103 | Replace each lane's five aligned 16-bit Q5 loads with three overlapping aligned 32-bit loads. | Selected affine-Q5 batch-one Metal kernel and unchanged 320-byte standard packing | Runtime-correct and rejected | Production-shape digests match A094. Order/reverse `100 / 1000` screens are consistently slower across gate/up, down, attention-output, and value shapes. See PERF-FA125. |
| PERF-A104 | Replace sixteen scalar FP32 weight/input FMAs with four `float4` dot products. | PERF-A103 aligned-overlap Q5 load geometry and per-lane FP32 reduction | Runtime-correct and rejected | Representative parity and all four production-shape digests match A094. Order/reverse timings are neutral and do not fund changed FP32 reduction grouping. See PERF-FA125. |
| PERF-A105 | Measure every prebuilt affine-Q5 batch-one kernel through one deterministic source-identical harness. | Direct native Q5/G64 QMV at representative Qwen3.8 gate/up, down, attention-output, and value-projection shapes | Retained benchmark; fresh-session measurements pending | The C++20 harness generates fixed packed weights, scale/bias, and input data; synchronizes each timed launch; and reports mean latency, effective streamed GB/s, first output, and an FNV-1a digest over the complete BF16 result. Strict, formatted builds are pinned for A094/A095/A100/A101/A103/A104/A106/A107/A108. Matching digests gate load-only arms; A104 additionally requires numeric and full-model behavior qualification. |
| PERF-A106 | Read each lane's sixteen BF16 activations through four aligned 64-bit words and convert them four at a time. | Selected A094 affine-Q5 batch-one kernel input owner; weight loads and FP32 order unchanged | Runtime-correct and rejected | Representative parity and every production-shape digest match A094. Order/reverse timings are neutral across the four exact shapes, with no durable margin. See PERF-FA125. |
| PERF-A107 | Fetch each lane's sixteen BF16 activations through two aligned 128-bit vectors while retaining four `bfloat4` conversions. | Selected A094 affine-Q5 batch-one kernel input owner; A106 transaction-width comparison | Runtime-correct and rejected | Representative parity and all production-shape digests match A094. Gate/up and down regress in both directions, attention-output is mixed, and value is slower; no full-model admission remains. See PERF-FA129. |
| PERF-A108 | Let one lane in each four-lane quantization group load BF16 scale/bias and broadcast the pair. | Selected A094 affine-Q5 batch-one parameter owner; group-64 four-lane sharing | Runtime-correct and rejected | Representative parity and all four production-shape digests match A094. Leader branching plus shuffles regress gate/up, down, attention-output, and value latency by roughly **4--22%**. See PERF-FA125. |
| PERF-A109 | Combine each dense BF16 recurrent b/a projection pair at load and issue one 96-row matmul. | Mixed-Q5 native loader and `Engine::gated_delta` | Runtime-correct and rejected as neutral | Two independent balanced five-versus-five windows aggregate to control **18.993383731** and fusion **18.993221731 tok/s**. All 20 runs reproduce digest `d0193f6d413b68c1`; the **-0.000162000 tok/s** aggregate movement closes the unchanged fusion. See PERF-FA128. |
| PERF-A110 | Attribute the selected mixed-target decode to concrete Metal shaders. | Selected A100 target under Metal System Trace plus GPU shader counters | Retained diagnostic | A 128-token sampled trace maps **48,156 / 48,831** target-process PCs with zero ambiguity. Stock affine-Q4 QMV owns **43.546%**, custom Q5 QMV owns **48.613%**, and all QMV owns **92.159%**, funding a stock-compatible Q4 kernel. |
| PERF-A111 | Double the stock affine-Q4/G64 batch-one output cohort while preserving MLX's helper and FP32 expression structure. | Mixed target's 162 Q4 linears, explicit native Q4 switch, and selected A100 Q5 path | Qualified and retained | K/N `512/64`, `5120/128`, and `5120/17408` are bit-exact against MLX. Two balanced five-versus-five windows improve **19.103271206 -> 19.230598293** and **19.124600969 -> 19.253364371 tok/s**; aggregate gain is **+0.128045244 / +0.669905%** with canonical digest `d0193f6d413b68c1`. |
| PERF-A112 | Screen the remaining Q4 output cohorts, load widths, lane work, and compile-time unrolling around A111. | Exact affine-Q4/G64 batch-one Metal kernel and mixed target | Rejected | Exact `8x4`, `4x8`, and `2x8` cohorts are flat or materially slower. A 128-bit weight load moves a five-sample full-model mean only **19.222120223 -> 19.232188501 tok/s**. Four packs per lane changes BF16 output and the seeded digest; forced full unrolling regresses gate/up to **1.799243958 ms**. See PERF-FA131. |
| PERF-A113 | Remove the explicit scalar `eval()` immediately before `array::item()`. | Native target-only two-token decode pipeline, selected A114, and MLX 0.32.2 scalar completion | Qualified and retained | MLX `array::item()` calls `array::eval()` itself, so output and pipeline ownership remain unchanged. Two independent five-pair windows improve **19.270942164 -> 19.277656005** and **19.285820872 -> 19.301343552 tok/s**; aggregate gain is **+0.011118260 / +0.057672%**, with all 20 outputs canonical. |
| PERF-A114 | Fuse paired affine-Q4 gate/up projections and SwiGLU while retaining every staged BF16 boundary. | Mixed target's 64 Q4/G64 MLP pairs, selected A100+A111, and an opt-in 8-SIMD/four-pair Metal kernel | Qualified and retained | The original `metal::exp` arm fails one real sigmoid boundary and is closed by PERF-FA132. `metal::precise::exp` is exact across all 64 traced layers and a dedicated `-6.84375` boundary regression rejects the old artifact. Two independent five-pair windows improve **19.236012324 -> 19.264036375** and **19.221428286 -> 19.272779458 tok/s**; aggregate gain is **+0.039687612 / +0.206398%** with canonical digest `d0193f6d413b68c1`. |
| PERF-A115 | Halve the fused-Q4 threadgroup from eight to four SIMD groups. | A114 fused gate/up/SwiGLU production shape | Rejected | The exact 4-SIMD/four-result form averages **0.567203850 ms** against **0.566040871 ms** for selected 8-SIMD, about **0.205% slower**. See PERF-FA133. |
| PERF-A116 | Assign each fused-Q4 epilogue row to one lane through dynamically indexed result arrays. | A114 fused gate/up/SwiGLU precise epilogue | Rejected | Focused parity passes, but a reversed 10,000-iteration window averages **0.564099544 ms** against **0.562567529 ms** for the serial control, about **0.272% slower**. See PERF-FA134. |
| PERF-A117 | Execute the four precise fused-Q4 epilogues on lanes 0--3 with compile-time register selection. | A114+A113 fused gate/up/SwiGLU decode path | Qualified and retained | Exact general-shape and `-6.84375` boundary tests pass. Two clean five-pair windows improve **19.268343602 -> 19.450918961** and **19.268607678 -> 19.455265774 tok/s**; aggregate gain is **+0.184616727 / +0.958128%**, with every output canonical. Signed `00d09138ce` retains the source. |
| PERF-A118 | Replace the fused-Q4 helper's two scalar packed-word reads with one explicit `packed_ushort4` transaction. | Selected A117 fused gate/up/SwiGLU kernel | Runtime-correct and rejected | Focused Q4, fused-chain, and precise-boundary parity pass. Two 10,000-iteration candidate samples average **0.560992548 ms** versus **0.557832737 ms** for A117, about **0.566% slower**. See PERF-FA135. |
| PERF-A119 | Share one affine-Q5 input launch across linear-attention `qkv` and `z` by concatenating their output rows at load time. | Forty-eight selected Q5/G64 linear-attention layer pairs | Runtime-correct and rejected | Ten order/reverse micros improve **0.426645998 -> 0.420592346 ms** with exact BF16 output. Full-model fusion produces **19.441971742 / 19.406059113 tok/s** around an adjacent same-dylib control at **19.469521945**. The copied tensor plus split-graph form is rejected; only a direct two-output kernel is materially different. See PERF-FA136. |
| PERF-A120 | Consume the two original affine-Q5 projections in one four-SIMD custom kernel and emit two direct outputs. | Production `qkv`/`z` pair, original packed tensors, no concatenation or split | Runtime-correct and rejected | Ten exact order/reverse samples regress **0.421745535 -> 0.427401865 ms**, **1.341171%**. Every pair is slower; no model wiring or reload ran. See PERF-FA137. |
| PERF-A121 | Map the exact `qkv:z = 5:3` row ratio to five plus three SIMD groups in each eight-SIMD threadgroup. | Production `10240/6144` two-output Q5 pair | Runtime-correct and rejected | Ten exact order/reverse samples regress **0.423397846 -> 0.426008156 ms**, **0.616515%**. The topology reduces threadgroups but does not overcome eight-SIMD and per-output selection cost. See PERF-FA137. |
| PERF-A122 | Interleave each fused-Q4 gate row with its matching up row to expose two independent weight streams sooner. | Selected A117 production `5120/17408` fused Metal kernel | Runtime-correct and rejected | Ten order/reverse pairs regress **0.558719448 -> 0.569066813 ms**, **1.851979%**, with every pair slower and every digest exact. Exact A117 restored; see PERF-FA138. |
| PERF-A017 | Replace shape-growing BF16-cache gather/GQA-repeat/score materialization with fixed-memory native Metal EXTEND attention. | `gguf_q4_0.mm` Q8/C64 BF16 paged GQA kernel and caller-owned pybind surface | Native mechanism qualified; production dispatch pending | At `E=17,L=131072`, the final-source native median is **137.906625 ms** with **0 MiB** measured driver-residency growth; dense MPS SDPA is **424.528292 ms** with **+8,088.515625 MiB**. Maximum error is `4.3120235e-07`. A lazy isolated Metal library keeps the new shader outside ordinary extension initialization. The raw binding is outside `TorchNativeAttnBackend`; the no-new-Python boundary requires an owner-approved dispatch seam before served gates. |
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

### 2026-08-21 01:40 PDT - PERF-037 exact norm-to-NVFP4 producer rejected

- A native SM120 kernel fused the selected residual-add/Gemma RMSNorm with
  exact NVFP4 packing while retaining both BF16 outputs. M1/M3/M7000/M7680
  normalized input, residual, packed values, and scale bytes matched exactly.
- Isolated M3 improved **0.040000 -> 0.027296 ms**, but the real dependent
  `3x5120 -> 34816` gate/up-GEMM boundary measured **0.096704 ms staged**
  versus **0.097152 ms fused** across 51 captured samples.
- The candidate projects to **-0.028672 ms** across 64 target layers. PDL
  already overlaps the standalone quantizer with GEMM startup, so the isolated
  launch saving is not serialized. Removed the native prototype and its exact
  JIT cache without changing model routing.

### 2026-08-21 02:05 PDT - PERF-038 narrow M3 NVFP4 CTA closed

- A native SM120 CUTLASS `64x32x256` prototype first failed because cooperative
  kernels require CTA-M at least 128. The ping-pong schedule passed that
  constraint but failed the fixed 128-row NVFP4 scale-factor TMA layout.
- The existing selected tactic already swaps A/B for M3 and uses CTA-N 32, the
  minimum supported epilogue/LDSM width. A plain-tile specialization therefore
  has no smaller legal geometry in the bundled CUTLASS implementation.
- No kernel executed. Removed the prototype and its exact JIT cache; plain
  GEMM work now requires a different mainloop/scale-loader architecture, not
  another tactic entry.

### 2026-08-21 02:25 PDT - PERF-039 MTP norm/concat fusion below admission

- The occupancy-preserving native design launched two 320-thread norm CTAs in
  one graph node and wrote the concatenated MTP FC input directly. Concatenated
  BF16 values and the dependent FC output were bit-exact.
- Through the FC, 101-sample medians improved by **1.248 us at M1** and
  **2.080 us at M3**. The combined speculative-cycle ceiling is about
  **0.0033 ms**, far below the 0.25 ms funding floor.
- Removed the prototype and exact JIT cache before model wiring.

### 2026-08-21 02:40 PDT - PERF-040 selected swap-AB epilogue blocks a small fusion

- CUTLASS's stock EVT cannot access paired accumulator coordinates or store an
  N/2 result. The selected gate/up tactics are all swap-AB DP, moving the
  paired dimension to GEMM M and invalidating the proposed non-swap prototype.
- A full solution is a custom collective epilogue with tactic-specific weight
  interleave, half-height output mapping, exact BF16 staging, and exact
  TensorRT-LLM NVFP4 conversion. No implementation was started.

### 2026-08-21 05:00 PDT - PERF-041 sparse top-p device win retained

- Final exact-pivot top-k+top-p medians improved
  **109.122 -> 78.115 us** at M1 and **121.170 -> 86.722 us** at M3.
  Fifteen CUDA tests plus six target/draft graph integration tests cover
  graph replay, AIR prefix boundaries, tie overflow, and q ownership.
- Full-cycle A-B-A means were **16.910 / 17.332 / 16.302 ms** and medians
  **16.954 / 17.322 / 16.002 ms** for candidate/control/candidate. Both
  candidate arms improved p90; the independent A2 arm exactly matched control
  output and acceptance.
- Final-source follow-up measured **16.090642 mean / 16.000558 median /
  16.265734 p90 ms** over 234 cycles with the same output/acceptance evidence.
  It retains AIR's exact pivot passes and removes only the dense apply.
- Exact short prompt mean was **3000.454 tok/s**. Exact long generation was
  `101.238, 125.757, 107.682` tok/s, mean **111.559**; native acceptance mean
  was **2.194869**. Retain default-off as additive compute work, not a promoted
  client record.
- Commit: signed `7cb4ed0796` (`perf: add exact sparse top-p renormalization`).

### 2026-08-21 03:50 PDT - PERF-042 page-aligned prefix prefill

- Native page-table construction plus FlashInfer page-64 planning was bit-exact
  at all 25 exact-request prefix shapes. Aggregate paged attention improved
  **2940.132 -> 2785.184 ms/layer**, with another **77.074 -> 3.263 us**
  metadata reduction at the 192K prefix.
- Five exact `199000+16` prompt scores were
  `3216.299, 3209.808, 3208.212, 3209.050, 3205.270 tok/s`, mean
  **3209.728**. TTFT averaged **61.999103 s** and E2E **62.153173 s**;
  every sample cleared all prompt and wall-time thresholds.
- All requests completed exact `199016` with the established deterministic
  output digest. Short generation averaged only **98.029 tok/s**, so the
  generation target remains open and PERF-042 is not a complete scoreboard
  promotion yet.
- Regression review found and repaired stale page metadata, decode-child
  ownership, MXFP8 sidecar admission, and fail-loud mapping gaps. Eight
  focused/fast-plan tests and fresh re-disproofs found no remaining material
  behavioral regression risk.
- Commit: signed `afd5606077` (`perf: use physical pages for aligned prefill`).

### 2026-08-21 03:55 PDT - PERF-043 draft top-k 32 rejected

- With page-aligned prefill held fixed, changed only aligned draft sampling
  top-k from 20 to 32.
- Five 512-token probes moved acceptance **2.217279 -> 2.173943** and worsened
  mean client latency. Static support widening admits more q mass outside the
  useful target overlap and is closed unchanged.

### 2026-08-21 04:03 PDT - PERF-044 greedy draft top-k 1 rejected

- Held page-aligned prefill fixed and changed only draft q top-k from 20 to 1.
- Exact `199000+16` generation remained **97.900 tok/s**. Three 512-token
  greedy-mode probes averaged only **2.107020** emitted tokens/verify, so
  collapsing q did not improve target overlap and is closed.

### 2026-08-21 04:12 PDT - PERF-045 draft top-k 16 rejected

- Held page-aligned prefill fixed and changed only draft q top-k from 20 to 16.
- Five sampled-profile acceptance probes averaged **2.205710**, below the
  adjacent k20 mean **2.217279**, with slightly worse mean latency. Together
  with the earlier k1/k8/k32 losses, this closes static support sizing.

### 2026-08-21 04:42 PDT - PERF-046 proposal top-p 1.0 probe was vacuous

- The override was cached only on `MultiLayerEagleWorkerV2`; the live worker
  was `EAGLEWorkerV2`, so the runner's compatibility default retained normal
  top-p. The alleged candidate and matched control were both controls.
- Their cycle difference (**0.010/0.002 ms mean/median**) is noise evidence,
  not a mechanism result. Proposal-only top-p 1.0 remains unmeasured until
  routed through the live worker. Removed all temporary source changes.

### 2026-08-21 05:12 PDT - PERF-046 fully routed proposal top-p 1.0 retained

- Routed the override through both actual owners: captured
  `EAGLEDraftCudaGraphRunner` proposals and worker-owned post-extend proposals.
  The final trace reduced AIR top-p from about three to one launch/cycle.
- Matched M3 control/candidate cycle:
  - mean **16.108184 -> 15.913862 ms** (-0.194322 ms);
  - median **16.044640 -> 15.859291 ms** (-0.185350 ms);
  - p90 **16.247111 -> 16.097856 ms** (-0.149255 ms).
- Five acceptance probes averaged **2.229702** versus k20/top-p0.95
  **2.217279**, while mean 512-token latency improved about 1.6%. Exact
  capacity/digest passed at **3214.278 prompt / 87.402 short-generation
  tok/s**, **61.911255 s TTFT**, and **62.082877 s E2E**.
- Retain default-off as an additive generation-cycle win; it does not solve the
  120 tok/s exact short-generation target alone.
- Seventeen focused graph/device-cycle/runner tests pass. Claim-by-claim
  re-disproof found no material exactness, routing, RNG, lifetime, ownership,
  mutation, or synchronization regression.
- Commit: signed `6b963eed05` (`perf: skip full-support draft top-p`).

### 2026-08-21 05:02 PDT - PERF-047 proposal penalty scale has no workload effect

- Added one cached proposal-only additive-penalty scalar to both worker
  implementations and scaled the graph-stable additive row before q.
- Correctly routed scale 0.75 reproduced the exact same five sampled proposal,
  output, acceptance, and histogram sequences as the preceding control.
  Scale 0.0 reproduced the same first 512-token sequence:
  **2.216450** emitted length and identical output SHA-256.
- The workload therefore exposes no acceptance leverage through this row;
  temporary source was removed.

### 2026-08-21 06:18 PDT - PERF-048 ReplaySSM commit overlap rejected

- Production-shape isolated fold measured **222.6 us** without tracking and
  **334.2 us** with tracking. The live traced fold averaged **189.478 us**.
- Enqueued target ReplaySSM fold/conv rollback on a dedicated side stream after
  verify, overlapped it with draft extend, and rejoined the forward stream
  before returning to the scheduler.
- Versus PERF-046, M3 cycle mean/median/p90 improved
  **15.913862/15.859291/16.097856 ->
  15.755353/15.721147/15.889537 ms**, saving
  **0.158508/0.138144/0.208319 ms**.
- Flanking C control reproduced A at **15.922452/15.867868/16.123397 ms**.
  Interval analysis found the candidate hid **186.819 of 189.478 us** fold
  under graph 8, but graph 8 expanded **1.060552 -> 1.237001 ms**. The serial
  fold+graph span was about **1.234266 ms**, versus **1.237001 ms** overlapped.
  The side stream is neutral/slightly worse; full-cycle B movement is noise.
- All five output/acceptance trajectories and exact capacity matched, ruling
  out gross skipped work but not changing the performance conclusion.
- Removed the overlap source. PERF-046 remains the selected additive cycle line.

### 2026-08-21 06:50 PDT - PERF-049 static p/q calibration closed

- First branch-exact capture produced 151 complete records before the
  diagnostic's eight-item writer queue filled and crashed the opt-in server.
- Raised the bounded queue to `min(max_cycles, 64)`; eight unit tests pass.
  A second independent request completed and sealed 239 records without
  backpressure.
- Chronological train/validation analysis found:
  - support ceiling **2.737586**, proving proposal-model improvement remains
    theoretically valuable;
  - maximin gamma `(1.0, 1.05)` improves worst-half expected length only
    **0.000133**;
  - rank calibration validation **2.121759** versus baseline **2.128776**;
  - token calibration validation **2.124150**, also below baseline.
- Static gamma/rank/token corrections are rejected. Future proposal work needs
  context-dependent calibration or a learned hidden adapter.
- Commit: signed `4d6782121e` (`fix: prevent p-q capture backpressure`) retains
  only the bounded diagnostic queue repair and test.

### 2026-08-21 08:10 PDT - PERF-050 greedy k1 and long-context gate

- A no-penalty/top-p1 branch-exact corpus predicted that sharpening q toward
  argmax materially improves greedy target overlap. Live greedy k20 acceptance
  averaged **1.883967**; k1 evidence averaged **2.107020** at 6K and
  **2.426540** over exact199K+512.
- Matched exact199K+512 improved **116.549 -> 123.049 generation tok/s** with
  identical `cac0c6...a2092` output digest. K1 is retained as a specialized
  greedy long-window candidate.
- Five exact199K+16 scores kept prompt/time gates but generation remained
  `97.424, 99.245, 98.992, 98.476, 98.253`, mean **98.478**. Direct metadata
  shows seven verify cycles and emitted length **2.285714**.
- Exact-context profiling measured **19.895310 ms** mean cycle and
  **16.250653 ms** target graph. The 15 post-first tokens plus one-time
  transition cost therefore cannot clear 120 tok/s without either six cycles
  or a material long-context target-graph reduction.

### 2026-08-21 08:12 PDT - PERF-051 XQA controls closed

- Exact XQA shape: SM120, B1/Q3, QH24/KVH4, D256, FP8 KV, BF16 Q/O,
  page64, sequence 199000. Counts 32-144 all changed bitwise output and saved
  at most a few microseconds versus all 170 SMs. Only 170 matched.
- At 170 SMs, PDL true/false was bit-exact and timing-neutral
  (**271.424/271.840 us median**). No built-in XQA launch control is retained.

### 2026-08-21 08:15 PDT - PERF-052 M4 greedy k1 rejected

- Reopened M4 because greedy k1 changed the prior q20 acceptance premise.
  Exact199K+16 still used seven cycles and emitted **2.285714** tokens/cycle
  (histogram `[2,1,2,2]`). The extra row/step cannot reduce the discrete cycle
  count and is rejected without another long window.

### 2026-08-21 07:05 PDT - PERF-053 device-resident greedy cycle rejected

- Reopened the retained two-launch cycle because greedy k1 removes the
  categorical proposal cost that informed its earlier rejection.
- The exact request measured **3215.592 prompt / 97.730 generation tok/s**,
  **61.885956 s TTFT**, and **62.039440 s E2E**. Exact `199016`,
  `finish_reason=length`, and output SHA-256 `cdf5bb57...f647d9` matched.
- The candidate remains inside and slightly below the five-run **98.478 tok/s**
  control window. Graph composition therefore does not fund the exact16 gate.

### 2026-08-21 07:06 PDT - PERF-054 XQA source constants closed

- At the exact SM120 XQA shape, three V buffers changed the matched all-SM
  median **273.952 -> 273.824 us**; a 64-token V tile reached **272.992 us**.
  Both were bit-exact but too small to fund a cycle change.
- A 128-byte K partition with the required two buffers exceeded shared memory.
  Reducing to one K buffer compiled and appeared faster (**264.992 us**) but
  produced different outputs across repeated launches, proving the mainloop
  depends on double buffering.
- Deterministic row-max method 0 was slower and changed numerics. CTA-x 2 cannot
  represent the four-warps-per-GEMM1 group. Every candidate cache was removed;
  installed source and the control module were restored.

### 2026-08-21 09:13 PDT - PERF-059 native draft-k1 delta proposal

- Replaced the aligned draft-k1 softmax, dense top-k renormalization, and
  categorical draw with a default-off two-launch CUDA argmax plus exact
  caller-owned one-hot q producer. The native boundary measured
  **3.659/3.901 us** for M1/M3 versus **86.635/73.155 us** for aligned q.
- Four CUDA cases plus three shape subtests cover stable ties, NaNs, additive
  state, full Qwen vocabulary, and mutable graph replay. Thirteen proposal
  dispatch tests pass.
- Adjacent A-B-A long-context windows:
  - initial control was externally contended: **119.274 tok/s** mean;
  - candidate B: **123.559 tok/s** mean over
    `123.406,123.428,123.418,123.769,123.776`;
  - closing control A2: **122.352 tok/s** mean over
    `122.028,122.382,122.610,122.564,122.175`.
  Candidate B improves the matched A2 mean by **1.208 tok/s / 0.987%** with
  identical `cac0c6...a2092` output.
- An independent candidate restart measured
  `123.711,123.711,123.874,123.934,123.924`, mean **123.831 tok/s**.
  Every candidate request completed exact `199512`; prompt means were
  **3191.592/3196.061 tok/s**, TTFT **62.351/62.264 s**, and E2E
  **66.487/66.391 s** across the two windows.
- Exact16 remained a seven-cycle diagnostic at **99.173 tok/s** with exact
  `199016` and the established digest. The original BENCHMARK.md target remains
  open; the temporary idea of redefining the scoreboard around long generation
  was rejected as goalpost movement.
- Preserved reasoning returned `703`; one exact multiply tool call parsed;
  image/audio remained false; standalone OpenCode2 returned `READY`; cache
  flush reported 4,775 MiB free.

### 2026-08-21 13:48 PDT - PERF-062 promoted gate/up hybrid

- Enabled the repository's Marlin W4A16 kernel on SM120 by replacing the
  MSVC-hostile nested `else-if` template chain with independent predicates and
  adding a native NVFP4 wrapper that does not depend on the optional AOT
  `sgl_kernel` Python package.
- Added a coalesced native in-place Cutlass-to-Marlin relayout. It matches the
  canonical Marlin repacker bit-for-bit, round-trips to the original Cutlass
  bytes, and projects **26.056 ms** to relayout all measured target families.
  The promoted route narrows this to all 64 gate/up projections and reuses one
  **85 MiB** scratch buffer.
- Full-target Marlin improved exact16 generation to **114.820 tok/s** but
  collapsed prompt to **1984.193 tok/s**. Draft-only Marlin stayed seven-cycle
  limited at **99.770**. All-target gate/up Marlin retained the useful output
  trajectory without the other projection costs and set the accepted record:
  **3078.058 prompt / 114.617 generation tok/s**, **64.651152 s TTFT**,
  **64.782022 s E2E**, exact `199016`, `finish_reason=length`.
- A clean no-argument launcher restart independently reached **3052.437 prompt
  / 114.053 generation tok/s**, **65.193816 s TTFT**, and **65.325334 s E2E**
  with a process-selected seed. It again beat every prior record metric in one
  exact request and reproduced output SHA-256
  `9a0e20749e2930a697fefdd3bdd7863a067abe4d9860e6d1e7d9b80a62668b37`.
- The default server returned sampled arithmetic `703`, exactly one parsed
  `multiply({"a":37,"b":19})` call, preserved reasoning through the tool-result
  continuation, non-thinking `READY`, image/audio false, and visible
  standalone OpenCode2 `READY`. All target/draft graph phases captured; cache
  flush left **4,338 MiB** free.
- The user accepted the all-four-metric improvement as production. Commit
  `03ba3d2e27` makes the selective checkpoint, chunk 7680, native draft-k1 q,
  large-EXTEND tuning, and gate/up hybrid Marlin the normal Windows launcher
  path. The higher 3100/120 milestone remains future work.
### 2026-08-20 21:10 PDT - PERF-A-BASELINE-001 retained IQ2 Metal baseline

- Change: measurement only from `main` at
  `e09e43171d46d4f5bdf35d3c8b3b56a628bafc0d` plus the recovered Apple
  worktree. No source was changed before this baseline.
- Environment: MacBookPro18,2 with an Apple M1 Max, 32 GPU cores, 32 GiB
  unified memory, macOS 26.6.2 (25G83), Python 3.11.15, PyTorch 2.11.0,
  MLX 0.32.0, mlx-lm 0.31.3, AC power, sleeping display, 94% system memory
  free, and no thermal or performance warning. Port 30000 and the SGLang,
  Metal-compiler, and model-runtime process set were empty.
- Input: immutable Bartowski `Qwen3.8-27B-IQ2_XXS.gguf` revision
  `f0eec4a4bb4975114a030d048952d83c0a53c034`, SHA-256
  `b01f668356e5799fd76315bd6abc0e45234580409ebc5c8fb4b675e3c10dc2b9`.
  The measured tensor was `blk.8.ffn_gate.weight`, IQ2_XXS,
  `17408x5120`, batch one.
- Command:
  `.venv/bin/python benchmark/mac/bench_mps_gguf_quant.py <immutable-blob> --tensor blk.8.ffn_gate.weight --batch-size 1 --warmup 8 --iterations 25`.
- Raw milliseconds:
  `1.255834, 1.215417, 1.204625, 1.227542, 1.184541, 1.191250,
  1.215709, 1.201458, 1.185042, 1.189375, 1.208000, 1.189250,
  1.179875, 1.190500, 1.200875, 1.186125, 1.183500, 1.168500,
  1.179125, 1.187875, 1.187375, 1.194792, 1.175834, 1.189125,
  1.214000`; median **1.189375 ms**, effective packed bandwidth
  **18.064 GiB/s**.
- Correctness evidence: actual-file CPU-dequantized parity passed at 17 odd
  output rows for every packed type present in this checkpoint at batch sizes
  1, 3, 4, and 8, plus Q2_K embedding lookup. Worst observed absolute error
  was `2.86102e-06`; worst relative error was `9.45619e-07`. The focused MPS
  convolution-state tests passed 2/2 and the MLX quantized-KV suite passed
  9/9.
- Decision: accept as the fresh pre-candidate IQ2 kernel baseline. It agrees
  within 0.4% with the retained 1.193958 ms window and does not qualify the
  checkpoint's served behavior.

### 2026-08-20 21:14 PDT - PERF-A001/A003 recovered Apple mechanisms committed

- Change: committed the recovered packed Q2_K/Q4_K/IQ2_XXS/IQ1_M Metal
  matmul and embedding route, the MPS FP32 convolution-state default, and the
  focused microbenchmark/parity tools as signed commit `7740cae691`
  (`perf: serve packed low-bit GGUF on MPS`). Committed thresholded,
  process-opt-in quantized-KV query tiling as signed commit `1271610e0b`
  (`perf: tile long-context MLX quantized attention`).
- Benchmark evidence: the packed Bartowski IQ2 lane retains the fresh M1 Max
  **1.189375 ms / 18.064 GiB/s** batch-one projection baseline and the prior
  **6.956 prompt / 3.189 generation tok/s** model-level window. Separately,
  the Mac Pro-only MLX helper evidence retains the measured `1024x32768`
  reduction from **0.258692 s / 1.732 GB** to **0.229053 s / 0.701 GB**;
  its exact 5K behavior remains cross-machine history under the 1 GiB
  threshold and carries no M1 Max record standing.
- Correctness evidence: actual-file packed parity passed at batch 1/3/4/8,
  MPS convolution-state tests passed 2/2, GGUF metadata/name-map tests passed
  3/3, and the MLX quantized-KV suite passed 10/10. The added Qwen3.5 test
  compares the complete gated attention wrapper against mlx-lm's ordinary
  quantized-cache forward, including projections, Q/K normalization, RoPE,
  causal attention, gate, output projection, and cache offset.
- Decision: retain both mechanisms. The MPS path is the available Q2
  playground. MLX tiling is capacity infrastructure rather than a general
  speed claim; its long-context profile still needs a cache policy and an
  exact rung beyond 32K.

### 2026-08-20 21:30 PDT - PERF-A004 native GGUF marker repair

- Change: registered GGML USER_DEFINED vocabulary entries as ordinary added
  tokens in the native GGUF tokenizer while retaining GGML CONTROL entries as
  special tokens. Qwen reasoning and tool boundary tokens are now atomic and
  remain visible when `skip_special_tokens=True`. The change is isolated to
  native GGUF tokenizer construction and is signed in `8879ed3d01`.
- Benchmark evidence: no throughput number is attributed to this correctness
  change. Weight/cache residency stayed at the retained 10.03/0.29/0.12 GB
  values during the live gate.
- Correctness evidence: the real retained artifact changed `<think>\n` from
  four token pieces to `[248068, 198]` and `<tool_call>\n` from five pieces to
  `[248058, 198]`. An explicit Qwen3/Qwen3-Coder parser launch returned
  coherent separate reasoning plus visible `703`, exact thinking-disabled
  `READY`, exactly one `multiply({"a":37,"b":19})` call with
  `finish_reason=tool_calls`, and a preserved tool-result continuation ending
  in `703`. Image/audio remained disabled. Focused validation passed 321 tests
  plus 64 subtests.
- Decision: retain. The previous apparent checkpoint formatting failure was
  tokenizer-induced. The IQ2 lane now has valid local behavior evidence, while
  sampled performance, independent restart, maximum context, and OpenCode2
  gates remain required before promotion.

### 2026-08-20 21:30 PDT - PERF-A003 dependency-safe tile ordering

- Change: replaced the quantized-prefill tiler's arithmetic
  `sum(previous) * 0` lifetime edge with MLX's `mx.depends` scheduling
  primitive. Signed commit `ea983f3120` preserves the intended serialized
  allocation lifetime without modifying query values.
- Benchmark evidence: no new speed claim; the retained Mac Pro-only
  long-shape evidence is **0.258692 -> 0.229053 s** with peak allocation
  **1,732,382,776 -> 701,499,056 bytes**. The ordinary path remains default,
  and these measurements carry no M1 Max record standing.
- Correctness evidence: the full quantized-KV helper and Qwen wrapper suite
  passed **10/10** after the change; Python compilation and whitespace checks
  passed.
- Decision: retain as a promotion prerequisite. The former arithmetic edge
  could propagate NaN/infinity across independent tiles and added avoidable
  operations. Exact long-context qualification still depends on a pinned MLX
  build, cache limit, explicit profile, and capacity ladder.

### 2026-08-20 21:45 PDT - PERF-A006 pinned llama.cpp Metal comparison

- Change: built the official llama.cpp source at detached commit
  `749f688fcaa4c472ec034b08cb8a907c45cfaa02`, build 10547, with AppleClang
  21, Release, native ARM, Accelerate, and embedded Metal. The server binary's
  SHA-256 is
  `261f0de1320e14c4512751639998f4ab46f54ddd20e48bc239d8dc80b5ccb178`.
- Benchmark evidence: one warmup followed by five exact `12+256`, greedy,
  ignore-EOS requests produced request-observed generation
  `14.642054, 14.671473, 14.660470, 14.665758, 14.667059 tok/s`;
  aggregate **14.661356**, mean **14.661363**. Internal decode samples were
  `14.802667, 14.793395, 14.771764, 14.777205, 14.778380 tok/s`, mean
  **14.784682**. All requests produced 256 tokens, stopped at the length
  limit, and retained FNV-1a-64 `6d4d220de481f54e`.
- Correctness evidence: the live OpenAI endpoint returned separate coherent
  `reasoning_content` and visible final `703`; `/props` reported vision,
  video, and audio false. The exact tool/preserved-result gate was already
  established on the corrected SGLang path and was not repeated after the
  decisive benchmark window.
- Decision: this window establishes the route-neutral M1 Max Q2 frontier at
  **14.661356 tok/s** aggregate and **14.671473 tok/s** best hit. Retain the
  pinned checkout as both the benchmark reference and a supporting dependency
  oracle. Its 4.60x advantage over the contemporaneous native-MPS decode
  confirms substantial kernel and dispatch headroom in the repository path.

### 2026-08-20 21:45 PDT - PERF-A007 adjacent native-MPS control

- Change: measurement only on signed `9b3debddee`, before compact mixed-shard
  storage. The explicit parser server used the retained IQ2 artifact, float32
  MPS, context/pool 1024, one request, chunk 256, and disabled radix, overlap,
  and graphs.
- Benchmark evidence: five cache-flushed exact `128+32` samples produced
  prompt `6.988, 6.972, 6.963, 6.987, 6.985 tok/s` and generation
  `3.185, 3.186, 3.186, 3.185, 3.187 tok/s`; means **6.979/3.1858**.
  TTFT was `18.316205, 18.359352, 18.382821, 18.319858, 18.323902 s`, mean
  **18.340428 s**. E2E mean was **28.070683 s**.
- Correctness evidence: every run completed exact 128+32,
  `finish_reason=length`, and SHA-256
  `37f5512bd18c1962bd2e170f543fae806ca48b70495c2761ae162df3cac56299`.
- Decision: use this adjacent window as control A for eliminating mixed packed
  weight materializations.

### 2026-08-20 21:58 PDT - PERF-A007 compact mixed-shard storage

- Change: materialize heterogeneous merged GGUF shards once into a compact,
  logical-output-order MPS parameter. Mixed native-Metal calls receive
  contiguous narrow views into that shared allocation, whose storage offsets
  the native binding already honors. Equal-format/equal-width merged shards
  retain their single matmul. CUDA and every non-MPS platform retain the
  established padded parameter path. Signed commit `13bea403d6` owns the
  implementation and focused tests.
- Loader diagnostic: the first attempt concatenated CPU loader shards and the
  startup warmup failed immediately with `packed_weight must be on MPS` after
  reporting 4.03 GB residency. The corrected owner allocates directly on
  `qweight.device` and copies each shard into its recorded range. The clean
  launch loaded in 19.88 seconds, reported **9.03 GB** weights versus the
  adjacent control's **10.03 GB**, and completed startup normally.
- Benchmark evidence: five cache-flushed exact `128+32` samples produced
  prompt `7.025, 6.965, 7.031, 7.155, 6.936 tok/s` and generation
  `3.306, 3.312, 3.322, 3.311, 3.294 tok/s`; means were
  **7.0224/3.309**. Relative to control A, prompt improved **0.622%** and
  generation improved **3.867%**. Mean TTFT changed
  **18.340428 -> 18.229377 s** (-0.605%); mean E2E changed
  **28.070683 -> 27.597721 s** (-1.685%).
- Correctness evidence: every benchmark completed exact 128+32 with
  `finish_reason=length` and unchanged SHA-256
  `37f5512bd18c1962bd2e170f543fae806ca48b70495c2761ae162df3cac56299`.
  The live endpoint returned coherent preserved reasoning plus final `703`,
  exact thinking-disabled `READY`, and exactly one parsed
  `multiply({"a":37,"b":19})` call with `finish_reason=tool_calls`.
  `/model_info` reports image/audio false. Focused unit coverage passed 3/3;
  Python compilation and cached whitespace checks passed.
- Decision: retain. This removes all measured mixed packed-weight copies,
  creates additional unified-memory headroom for the maximum-context lane,
  and establishes the faster native-IQ2 control for the next kernel change.

### 2026-08-20 22:36 PDT - PERF-A002 IQ2_XXS batch-one specialization

- Change: replaced the generic IQ2_XXS batch-one scalar dequantizer with a
  pinned-ggml-derived Metal mapping that reuses each 32-value input atom across
  four output rows. Two SIMDgroups produce eight rows per threadgroup; the
  compact grid/sign tables are staged once in 2,176 bytes of threadgroup
  memory. Generic batches four and eight remain unchanged. Signed commit
  `16b2bf7a06` contains the shader, focused boundary test, and complete MIT
  notice/provenance.
- Microbenchmark evidence: adjacent generic controls on
  `blk.8.ffn_gate.weight` were `1.187792/1.176875 ms`; restored candidates
  were `0.584083/0.516000 ms`. The complementary long-K
  `blk.8.ffn_down.weight` changed **1.234667 -> 0.540791 ms**. Actual-file
  batches one, three, four, and eight passed, as did K=256/768 row tails,
  compact offset eight, grid indices 0/255, sign indices 0/1/126/127, and
  scale extrema.
- Served evidence: five deterministic exact `128+32` runs averaged
  **7.1748 generation tok/s**, up **116.77%** from compact-storage control,
  with the exact digest. Required sampling averaged **7.0562 tok/s**; a
  committed restart independently averaged **7.0638 tok/s**. Arithmetic,
  thinking-disabled, parsed-tool, preserved-tool-result, and language-only
  gates passed.
- Decision: retain. Constant-address-space LUTs and a four-SIMD/two-row
  geometry lost their matched warmed ablations and remain closed.

### 2026-08-20 23:09 PDT - PERF-A011 Q5_K batch-one vocabulary head

- Change: added an aligned Q5_K batch-one vector-four path. Each SIMDgroup's
  four eight-lane cohorts produce four independent rows, giving 16 rows per
  128-thread group. The generic pipeline remains available when packed-weight
  or input offsets do not satisfy vector alignment; every multi-batch route is
  unchanged. Signed commit `b19cf4acf3` owns the shader, shared dispatch, and
  focused boundary/extrema harness.
- Microbenchmark evidence: the retained `248320x5120` head's matched
  candidate/control/candidate medians were **3.737000 / 19.659291 /
  3.754625 ms**, sustaining approximately 217-218 GiB/s. Actual-file
  M=`1,7,8,9,15,16,17,31,32`, K=256/768, compact K=5120, vector-alignment
  fallback, and packed scale/min/high/low extrema all passed. Largest
  actual-file error was `5.96046e-07`; synthetic absolute/relative error was
  `0.00146484/3.02919e-07`.
- Served evidence: five deterministic exact `128+32` requests averaged
  **8.0284 generation tok/s**, up **11.90%** from PERF-A002, with exact
  digest and counts. Required sampling averaged **7.9450 tok/s**. The
  committed restart reached **8.114 tok/s** deterministically and averaged
  **7.9552 tok/s** across five sampled requests. All behavior/model-surface
  gates passed.
- Decision: retain as the selected native-IQ2 Q5_K head. The pinned llama
  one-row-per-SIMD decoder remains an oracle; its much larger threadgroup grid
  is unfunded after the cohort path's 5.24x isolated result. The next measured
  native target is PERF-A009's batch-one F32 GDN b/a projection.

### 2026-08-20 23:43 PDT - PERF-A009 native F32 batch-one projection

- Change: select PyTorch's native MPS matrix multiplication for F32 GGUF
  projections containing exactly one input vector. The existing custom Metal
  dense kernel remains selected for every multi-vector prefill. The live
  production path reaches this branch 48 times per decoded token through the
  compact `96x5120` GDN b/a projection.
- Microbenchmark evidence: a fresh actual-weight A/B/A sweep across all 48
  layers measured candidate/control/candidate medians **2.159000 /
  7.296667 / 2.051708 ms** for 90 MiB of F32 weights. Raw candidate A was
  `2.216708, 2.180458, 2.174042, 2.183375, 2.144958, 2.158209, 2.140250,
  2.148500, 2.159000 ms`; control was `7.218208, 7.243125, 7.296667,
  7.308958, 7.343000, 7.296208, 7.266750, 7.647750, 7.864792 ms`;
  candidate B was `2.154292, 2.071875, 2.025875, 2.039833, 2.051708,
  2.024833, 2.045709, 2.064416, 2.070833 ms`. Candidate and control differed
  by at most `1.78813934e-06`.
- Served evidence: five clean deterministic exact `128+32` requests reached
  generation `8.447, 8.450, 8.444, 8.431, 8.431 tok/s`, mean **8.4406**,
  or **5.13%** above PERF-A011. Prompt mean was **7.0444 tok/s**, TTFT mean
  **18.170302 s**, and E2E mean **21.842957 s**. The required sampled profile
  reached `8.309, 8.304, 8.295, 8.325, 8.314 tok/s`, mean **8.3094**, or
  **4.59%** above PERF-A011's first window. Every deterministic response kept
  exact counts, length finish, and digest
  `37f5512bd18c1962bd2e170f543fae806ca48b70495c2761ae162df3cac56299`;
  every sampled response preserved nonempty reasoning.
- Correctness evidence: focused coverage proves batch one bypasses the custom
  kernel, batches 2/3/4/8 retain it, output-row and input-width boundaries
  pass, and all 48 actual b/a matrices agree with CPU F32 within tolerance;
  maximum actual-layer error was `2.86102295e-06`. Arithmetic returned final
  `703`, thinking-disabled returned exact `READY`, the parser emitted exactly
  one `multiply({"a":37,"b":19})` call, the tool-result continuation kept
  reasoning and returned `703`, and `/model_info` retained image/audio false.
- Decision: retain provisionally. The first isolated and served windows pass;
  the independent committed restart also passes. This is the selected native
  IQ2 diagnostic path. Full Apple promotion remains gated by the exact Rust
  scoreboard, long-context capacity, and a real OpenCode request.
- Commit: signed `4d1641fdcd` (`perf: accelerate F32 GGUF decode on MPS`).

#### Independent restart

- From committed `HEAD`, one deterministic confirmation reached **8.420
  generation / 7.010 prompt tok/s**, **18.259251 s TTFT**, and
  **21.940782 s E2E** with exact counts and the established digest.
- Five required-sampling requests measured generation
  `8.313, 8.294, 8.238, 8.313, 8.313 tok/s` (mean **8.2942**) and prompt
  `7.029, 7.052, 7.044, 7.025, 7.026 tok/s` (mean **7.0352**). TTFT mean was
  **18.194274 s** and E2E mean **21.931784 s**. This is within 0.19% of the
  first candidate window and 4.26% above PERF-A011's independent mean.
- Arithmetic, thinking-disabled, parsed-tool, preserved-tool-result, and
  language-only gates passed again. Process-scoped OpenCode 1.18.15 reached
  this endpoint with a real **13,635-token** agent prompt; the 1,024-token
  diagnostic launch rejected it before execution. Context-enabling work is
  required before the standalone-client gate can pass.

### 2026-08-21 00:15 PDT - PERF-A012 lower-right causal partial extend

- Change: the shared torch-native SDPA extend mechanism now submits only the
  actual new query rows. Causal chunks with an existing prefix use an explicit
  lower-right-aligned mask derived from the gathered Q/KV shapes. One-shot
  causal, noncausal, cross-attention, and sliding-window paths retain their
  domain semantics without manufacturing unused prefix query rows.
- Baseline: signed `f697e40a10` on macOS 26.6.2, M1 Max 32-core GPU, PyTorch
  2.11.0 MPS, FP32 Qwen geometry `24Q/4KV/D256`, one process, synchronized
  samples, no server/client/compiler, 91-93% free memory, and no recorded
  thermal or performance warning. The benchmark carries an exact copy of the
  former padded source path as both matched controls.
- Short-chunk evidence: `prefix=4096, extend=256`, two warmups and five
  synchronized samples, produced padded A
  `98.142292,97.995583,97.975167,98.056584,97.830542 ms`, candidate
  `12.608125,12.673916,12.595750,12.723875,12.597375 ms`, and padded B
  `97.847500,97.742375,97.852083,97.806250,98.268000 ms`. Medians were
  **97.995583 / 12.608125 / 97.847500 ms**, a 7.77x source-level reduction.
- Production-chunk evidence: `prefix=4096, extend=4096`, one warmup and five
  samples, produced padded A
  `503.558875,556.033500,541.891708,542.376416,618.237542 ms`, candidate
  `508.756750,176.682250,176.066500,175.942334,176.000667 ms`, and padded B
  `788.375209,641.256125,617.944750,689.572459,582.157333 ms`. The first
  candidate sample carried a cold delayed dispatch; five-sample medians were
  **542.376416 / 176.066500 / 641.256125 ms**, a 3.08-3.64x reduction.
- Correctness: every MPS benchmark reported `source_max_error=0`. Six focused
  CPU tests pass for partial-prefix GQA with shuffled physical slots, an
  explicit future-value sentinel, lower-right sliding windows, ragged
  multi-request GQA, noncausal partial extend, and zero-length extend. Python
  compilation, Black 26.1.0, Ruff 0.15.1, and diff checks pass.
- Decision: retain in signed commit `210a214c12`. The governing rule belongs
  in `_run_sdpa_forward_extend`, the shared mechanism that gathers KV and owns
  SDPA masking for every torch-native extend call. Full-model long-context and
  OpenCode qualification remain gated by a separate decode admission defect:
  BF16 or more than 7,936 physical cache rows currently enters an incompatible
  FP32 score-array Metal kernel instead of the established SDPA fallback.

### 2026-08-21 00:24 PDT - PERF-A013 MPS decode capability gate

- Change: the shared torch-native MPS decode dispatcher now selects the fused
  native GQA kernel only when the actual query, current K/V, and physical NHD
  cache satisfy its FP32, contiguous-cache, head-dimension, and 7,936-row
  contract. Unsupported configurations continue through the existing cache
  write and PyTorch SDPA path.
- Baseline: with port 30000 free, no server/client/compiler process, 92% free
  memory, and no recorded thermal or performance warning, BF16 with 32,769
  physical rows failed with `native Metal decode attention requires float32
  tensors`; FP32 with 7,937 rows failed with `native Metal decode attention
  supports at most 7936 cache slots`.
- Boundary evidence: the same BF16/32,769 and FP32/7,937 probes now complete
  through SDPA with `max_error=0`. The eligible FP32/7,936 boundary remains on
  the fused native kernel and reports `max_error=2.5331974e-07`. Each process
  was isolated; memory remained 92% free and macOS recorded no warning.
- Correctness: the harness uses production-equivalent cache dtype casting and
  verifies current-token K/V mutation before comparing grouped-query output.
  Nine focused CPU cases pass across this dispatch and the retained partial
  extend behavior. Python compilation, Black 26.1.0, Ruff 0.15.1, and diff
  checks pass.
- Decision: retain in signed commit `b2b8ab4af8`. The rule lives in
  `TorchNativeAttnBackend.forward_decode`, the lowest shared layer that owns
  both native admission and the established fallback. Full-model 32K capacity
  and the 13,635-token OpenCode request remain the next gates; PERF-A008's
  bounded native kernel remains funded as the throughput path after capacity.

### 2026-08-21 00:41 PDT - PERF-A014 large-batch quantized projection baseline

- Change: measurement and source attribution only. No kernel or dispatch was
  changed.
- Benchmark evidence: fresh-process medians for the actual IQ2_XXS
  `blk.8.ffn_gate.weight` (`17408x5120`) were **65.359958, 250.314041,
  493.479750, 983.834750, and 1967.899583 ms** at batch sizes 128, 512, 1024,
  2048, and 4096. Three raw synchronized samples per size are preserved in
  `notes/experiment-log.md`.
- Production evidence: the committed 32K/BF16 server became ready with exact
  pool capacity and completed required-sampling `128+32` at **6.963 prompt /
  6.772 generation tok/s**. Its exact `4096+2` request crossed the scheduler's
  300-second watchdog before returning a token. This is censored timing;
  linear projection of the completed short prompt already predicts about
  583-588 seconds for 4,096 tokens.
- Mechanism: `quant_matmul` selects `BatchTile=8` for IQ2_XXS above batch four,
  so batch 4,096 launches 512 groups that each traverse and dequantize the
  complete packed matrix. A shared-dequant 64-output by 32-input-row matrix
  tile reduces packed-weight traversals sixteenfold relative to that path.
- Correctness evidence: no new arithmetic path exists yet. The current kernel
  remains covered by actual-file quantized references; candidate admission
  requires format parity, odd output/batch tails, stable decode dispatch, and
  served behavior gates.
- Decision: fund the narrow native Metal matrix-matrix candidate. Keep the
  selected batch-one/four/eight paths intact and require an actual-tensor
  matched A/B before another long server request.
- Commit: pending evidence commit.

### 2026-08-23 06:57 PDT - PERF-A014 baseline reproduced before implementation

- Change: measurement only from clean `11b2b33613`; no source or dependency
  changed before this window.
- Benchmark evidence: the actual IQ2_XXS `blk.8.ffn_gate.weight`
  (`17408x5120`) measured **65.578125 ms** at batch 128 (five synchronized
  samples), **249.418209 ms** at batch 512, and **1971.539875 ms** at batch
  4096 (three samples each). Raw samples and the exact commands are preserved
  in `notes/experiment-log.md`.
- Environment: Apple M1 Max, 32 GPU cores, Metal 4, PyTorch 2.11.0, GGUF
  0.19.0. Port 30000 was free, no SGLang/llama/benchmark/compiler process was
  live, memory pressure was 93% free, and macOS reported no thermal or
  performance warning.
- Decision: continue PERF-A014. The fresh medians reproduce the prior
  repeated-weight-traversal curve closely enough to serve as the adjacent
  implementation baseline.
- Commit: pending candidate evidence.

### 2026-08-23 07:06 PDT - PERF-A014 FP32 simdgroup kernel clears isolated gate

- Change: added an Apple7-gated IQ2_XXS matrix-matrix path for batches above
  eight. Four SIMD groups share FP32 dequantized `64x32` weights and FP32
  `32x32` activations from 12 KiB of threadgroup memory, accumulate and store
  FP32, and preserve the existing batch-one/four/eight paths. A process-scoped
  `SGLANG_MPS_IQ2_LARGE_BATCH=0` control selects the prior implementation.
- Benchmark evidence: on the actual `blk.8.ffn_gate.weight` (`17408x5120`),
  batch 128 changed from a matched disabled median of **70.074833 ms** to
  **4.250250/4.277125 ms** in A/B/A candidate arms. Batch 512 changed from the
  adjacent **249.418209 ms** baseline to **16.009833 ms**; batch 4096 changed
  from **1971.539875 ms** to **124.838125 ms**. Candidate medians at batch
  9/16 were **2.160166/1.700792 ms**, versus matched disabled medians
  **6.439458/9.281000 ms**. Batch eight remains on the selected path and
  measured **4.671791 ms**.
- Correctness evidence: the existing actual-file suite passes unchanged at
  aligned `64x32`, output/batch tails `63x33`, `65x33`, and `65x31`, the
  minimal one-output tail at batch 32, and the true batch-eight fallback at
  `65x8`. IQ2_XXS maximum relative error was at most `2.27121e-06`; the
  aligned `64x32` case
  was exact. All other exercised quant formats and token embedding parity
  remained green.
- Decision: retain for full-forward and serving qualification. The isolated
  win is large enough to fund one controlled 32K full-model gate while the
  implementation remains uncommitted.
- Commit: pending full-model evidence.

### 2026-08-23 07:19 PDT - PERF-A014 full-model prefill qualification

- Launch: the uncommitted candidate loaded the immutable 9.393 GB IQ2_XXS
  artifact into the native torch/MPS 32K-configured profile with FP32 compute,
  BF16 KV, one request, 4,096-token chunks, reasoning and Qwen3 Coder tool parsing,
  and every graph backend disabled. Weight load used 9.03 GB; the exact
  32,768-token KV pool used 2.00 GB; the server reached ready with 19.97 GB
  available and no thermal or performance warning.
- Short sampled evidence: exact `128+32` completed at **23.093 prompt / 6.736
  generation tok/s**, **5.542854 s TTFT**, and **10.144946 s E2E**. Decode is
  expectedly unchanged because batch one retains the selected kernel.
- Long sampled evidence: cache-flushed exact `4096+2` completed at **24.828
  prompt tok/s**, **164.975078 s TTFT**, and **165.438140 s E2E**. The former
  source crossed the 300-second scheduler watchdog on the same shape. Exact
  `5000+1` then completed two chunks at **24.845 prompt tok/s** and
  **201.251071 s TTFT/E2E**.
- Behavior: sampled reasoning remained in `reasoning_content`; arithmetic
  stopped normally with final `703`; thinking-disabled output was exactly
  `READY` with zero reasoning tokens; the parser returned one
  `multiply({"a":37,"b":19})` call; and the preserved tool-result turn
  stopped with `37 × 19 = 703`. `/model_info` reported image and audio
  understanding false.
- Cleanup: after an explicit cache flush, the verified `3807/3806/3805/3800`
  process tree was stopped leaf-first. All PIDs disappeared, port 30000 and
  Metal compiler workers were clear, memory returned to 90% free, and macOS
  still reported no thermal or performance warning.
- Decision: retain. This removes the native IQ2 prefill watchdog blocker and
  qualifies the governing large-batch dispatch end to end. Batch-one decode is
  unchanged; a bounded native long-history GQA path and the exact SGLang
  `12+256` scoreboard remain separate funded work.
- Commit: signed `1676c71bed` (`perf: accelerate large IQ2 prefills on Metal`).

### 2026-08-23 07:30 PDT - PERF-A014 repeated full-model control closes review

- Scope correction: `65x31` is a guarded candidate batch tail under the
  batch-greater-than-eight rule. A fresh `65x8` actual-file suite exercised
  the retained fallback and passed every quant format; IQ2_XXS maximum
  absolute/relative error was `8.58307e-06/2.27121e-06`.
- Matched control: a fresh 32K-configured server with only
  `SGLANG_MPS_IQ2_LARGE_BATCH=0` changed measured exact `128+1` required-
  sampling prompt rates to `7.018,7.013,7.021,7.029,7.036 tok/s`, mean
  **7.0234**. TTFT samples were
  `18.238450,18.252279,18.231962,18.211360,18.191226 s`, mean
  **18.225055 s**. Every cache-flushed request completed exact `128+1` with
  `finish_reason=length`.
- First default window after an independent launch produced prompt
  `22.675,23.191,23.060,22.916,22.936 tok/s`, mean **22.9556**, and mean TTFT
  **5.576266 s**. A second independent default launch produced
  `22.621,22.857,22.795,22.901,22.862 tok/s`, mean **22.8072**, and mean TTFT
  **5.612339 s**.
- The two default windows combine to **22.8814 prompt tok/s** and
  **5.594303 s TTFT**: 3.258x the matched disabled prompt rate and 69.30%
  less TTFT. The process-scoped switch is the only server variable, directly
  tying the full-model delta to the new shared-dequant dispatch. The earlier
  exact `4096+2` and `5000+1` requests retain the long/multi-chunk gates.
- Each verified server tree was cleaned leaf-first between arms. Final state:
  no server, client, or workload-owned Metal compiler process; port 30000
  free; memory 90% free; no macOS thermal or performance warning.
- Decision: the default-enabled native IQ2 dispatch now satisfies repeated
  full-model measurement and independent-restart evidence. Retain and commit.

### 2026-08-23 07:46 PDT - invalid cross-machine q4 scoreboard removed

- Provenance correction: the affine-q4 results came from a separate Mac Pro
  experiment and were incorrectly attributed to this M1 Max. The q4 decode,
  maximum-context, speculative, and winner-capability entries have been
  removed from the local Apple benchmark authority.
- The measured M1 Max Q2 `12+256` frontier is pinned llama.cpp build 10547 at
  **14.661356 tok/s aggregate**, **14.671473 tok/s** best hit, **17.460868 s**
  mean E2E, and **17.448827 s** best E2E. Every sample completed exact
  `12+256` with stable digest.
- Decision: use that route-neutral Q2 result as the benchmark. Exact SGLang
  ingress remains open; PERF-A014 leaves batch-one decode unchanged and keeps
  fixed-memory GQA plus batch-one IQ2 optimization funded.

### 2026-08-23 08:05 PDT - exact native SGLang Q2 baseline established

- First Rust-ingress launch failed before serving because the GGUF-only
  tokenizer path contains no `tokenizer.json`. Qwen's official tokenizer-only
  files were then pinned at immutable revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
  the model weights remained the same Bartowski IQ2_XXS blob.
- One warmup followed by five exact `/generate` requests produced wall times
  `36.493178,36.605944,36.580286,36.584450,36.551911 s` and request rates
  `7.015010,6.993400,6.998305,6.997509,7.003738 tok/s`. Aggregate throughput
  is **7.001584 tok/s**, mean E2E **36.563154 s**, and best hit
  **7.015010 tok/s**.
- Every response completed exact `12+256`, stopped at length, returned the
  same 256 token IDs, and matched FNV-1a-64 `6d4d220de481f54e`. `/model_info`
  reported the text architecture with image/audio understanding false. The
  32,768-token BF16 KV pool allocated successfully.
- This window did not repeat the reasoning, tool, required-sampling, or
  independent-restart gates through the official-tokenizer Rust boundary;
  earlier Python/GGUF-tokenizer evidence does not qualify that combination.
- Decision: the native route is **52.2446%** below the Q2 record, which is
  **2.094006x** faster. Profile batch-one decode before selecting the next
  native C++/Metal kernel. Verified cleanup left no server or workload-owned
  compiler process; four system `MTLCompilerService` XPC processes remained
  idle at 0.0% CPU. Port 30000 was free, memory 90% free, and no
  thermal/performance warning was recorded.

### 2026-08-23 08:14 PDT - native Q2 batch-one decode diagnostic captured

- A clean 32K Rust/official-tokenizer launch enabled the existing one-shot
  synchronized layer and stage profilers at batch one. The built-in six-token
  warmup consumed the profile after the same 9.03 GB weight, 0.29 GB Mamba,
  and 2.00 GB BF16 KV allocations as the exact baseline.
- The 64 raw layer timings summed to **164.266 ms**. Excluding layers 0-10
  with clear first-use overhead, 14 full-attention samples averaged exactly
  **3.130000 ms/layer** and 39 GDN samples averaged **1.719026 ms/layer**.
  Using the raw sums, their model-topology projection is **50.080 + 82.513 =
  132.593 ms/token**. The separate exact-request wall time is **142.824820
  ms/completion token**; its **10.232 ms/token** numerical difference crosses
  runs and context distributions and therefore does not isolate outside-layer
  work.
- The nested GDN layer-8 view measured 0.522 ms input projection, 0.354 ms
  recurrent core plus convolution, 0.360 ms output projection, and 0.863 ms
  MLP, with additional pack/norm/reorder stages. Nested synchronization
  inflates that sum, and layer 8 belongs to the excluded first-use region. Its
  broad proportions support a cheap batch-one IQ2 projection experiment; they
  do not establish route-wide attribution.
- Full-attention layers carry a **1.411 ms/layer** premium over the stable GDN
  mean. This bounds the complete layer-family difference; fixed-memory BF16
  GQA remains a separately measurable follow-on.
- Verified leaf-first cleanup removed PIDs 7925 and 7921. Port 30000 is free,
  no workload-owned compiler or server process remains, the four system Metal
  compiler services are idle, memory is 90% free, and macOS records no thermal
  or performance warning.

### 2026-08-23 11:31 PDT - PERF-A016 final Q2 decode and Codex-profile qualification

- Signed commit `52b5326d8e` adapts pinned llama.cpp's Q4_K two-row activation
  reuse to the native Metal `quant_matmul` owner. Dispatch covers complete
  four-block, four-row-aligned batch-one tensors with safe compact-view
  `half`/`ushort` and float alignment. The specialization applies to the Q4_K
  tensor family inside the mixed-format Bartowski IQ2_XXS/Q2 checkpoint; the
  benchmark remains the M1 Max Q2 lane.
- Final-source Python-ingress candidate walls were
  `29.801688,29.820932,29.824091,29.795946,29.820783 s`: aggregate
  **8.586947946 tok/s**, best **8.591772854 tok/s**. The fresh matched disabled-
  kernel control aggregated **7.009167482 tok/s**, establishing a
  **22.510241%** full-model gain. An independent candidate restart aggregated
  **8.578204721 tok/s**. Every arm returned exact `12+256`, length finish, and
  the same token IDs, text, and digest.
- Actual-file enabled/tail parity passes. Exact `32761+1` completed inside the
  32,768-token BF16 pool at **19.242 prompt tok/s** and **1702.563753 s E2E**.
  Sampled reasoning, arithmetic `703`, thinking-disabled `READY`, one parsed
  multiply call, reasoning continuity through its result, and image/audio-
  disabled reporting all pass.
- Codex CLI 0.149.0 used the machine-local `qwen38-local` profile over
  `/v1/responses`. Its fixed read-only gate invoked `pwd` once, consumed the
  returned workspace, emitted exact visible `CODEX TOOL READY`, and exited
  zero with 17,871 input, 96 output, and 62 reasoning-output tokens. The
  worktree stayed unchanged and the server remained healthy.
- Final leaf-first cleanup removed the verified server tree, freed port 30000,
  returned memory to 94%, and left normal thermal/performance status. The
  selected SGLang aggregate remains **41.431420%** below pinned llama.cpp's
  **14.661356 tok/s** Q2 reference, so the next measured batch-one decode
  hotspot owns the continuing performance handoff.

### 2026-08-23 14:15 PDT - PERF-A017 bounded native Metal EXTEND mechanism

- Change: added an Apple7+ BF16 paged-GQA EXTEND kernel specialized for batch
  one, 24 query heads, four KV heads, dimension 256, and query lengths through
  1,024. The Q8/C64, four-SIMD-group design keeps FP32 online-softmax state in
  exactly 20,800 bytes of threadgroup memory and writes caller-owned FP32
  output. The host binding preserves tensor storage offsets, validates every
  dimension and narrowed scale, accounts for static plus dynamic threadgroup
  memory, and rejects output aliasing. Its material tiling provenance is the
  pinned llama.cpp `kernel_flash_attn_ext_impl`; the source comment and MIT
  notice record that adaptation.
- Correctness evidence: seven shuffled-cache cases spanning `E=1,7,8,9,17,
  256,1024`, prefixes `0,1,63,64,65,257`, partial Q8/C64 tiles, and invalid
  metadata all passed the dense lower-right-causal MPS reference. Maximum
  absolute error was `1.1064112e-06`. A separate nonzero-offset case reached
  `1.3113022e-06`. Existing torch-native EXTEND coverage passed six tests.
  Invalid sequence metadata produced exact zeros. A nondefault scale of
  `0.0375` matched at `5.3644180e-07`; zero, negative, nonfinite,
  FP32-overflow/underflow scales and direct/offset-view output aliasing were
  rejected before dispatch.
- Adversarial arithmetic evidence: invalid physical slots `[-1,0,4,2]`
  matched the filtered-key oracle exactly. A forced new maximum in the third
  C64 tile at `L=129` matched an analytic online-rescale oracle within
  `1.9073486e-06`. Equal-logit lengths `63,64,65,127,128,129` matched causal
  means within `2.9802322e-08`. Negative/short/oversized sequence metadata and
  negative/out-of-range/uint32-overflow request rows each produced exact zero
  output.
- Maximum-address evidence: analytic `E=5,L=131073` placed the sole marked
  K/V at physical row 131,072. The first four rows excluded that future value
  exactly; the fifth matched the closed-form softmax at maximum error
  `1.1920929e-07`. The call took **137.202250 ms** with zero measured driver
  growth. A 131,074-row view was rejected at the host fence.
- Long-context evidence: at physical pool 131,073 and active `L=131072`, an
  `E=1` five-sample window was
  `139.039291,138.983250,139.011958,138.997958,139.037042 ms` with median
  **139.011958 ms** and maximum error `1.7136335e-07` against the qualified
  fixed-memory decode kernel. SHA-256 was
  `81f55f2e2fe19a1c4e6dc697e79757a43519bfcb0f017a916fef15914311f508`.
- Independent dense-oracle evidence: `E=17,L=131072` native samples were
  `137.940500,137.931875,137.871750 ms`, median **137.931875 ms**. Dense MPS
  SDPA took **424.528292 ms**. Maximum/mean error was
  `4.3120235e-07/2.9860754e-08`. Native current/driver allocation deltas were
  exactly `0/0 MiB` after the caller-owned tensors were resident; the dense
  path added `1,027.125/8,088.515625 MiB`.
- Regression hardening: the new shader now compiles from a separate lazy Metal
  library. Ordinary extension initialization keeps its original compile
  options, leaves `SGLANG_EXTEND_GQA` undefined, preprocesses the experimental
  block away, and creates only the established pipelines. The hardware
  capability query returned `True` without creating
  the new pipeline; an isolated first call compiled it in **259.783416 ms**
  and retained parity. The ordinary BF16 decode capability/pipeline still
  returned `True` in a fresh process. Decode output remained bitwise identical
  before and after lazy EXTEND initialization; five-sample medians were
  `0.831250/0.838167 ms`. Final-source `E=17,L=131072` samples
  were `137.968875,137.906625,137.869250 ms`, median **137.906625 ms**, with
  exact `0/0 MiB` current/driver deltas.
- Decision: retain the bounded native mechanism as a capacity-critical
  checkpoint. Its raw pybind surface remains outside the live
  `TorchNativeAttnBackend.forward_extend` call chain. Production activation,
  served capacity, reasoning/tools, real-client behavior, and an independent
  restart remain open. An adversarial location review rejects import-time
  C++ monkeypatching and global aten override as hidden or overly broad policy
  ownership. The explicit repository rule against adding Python code makes an
  owner-approved dispatch seam the next architectural gate.
- Commit: signed checkpoint `0d1d0ea643` (`perf: add bounded Metal extend
  attention`).

### 2026-08-23 14:38 PDT - PERF-A018 direct BF16 cache-run loads

- Starting point: signed commit `0d1d0ea643` (`perf: add bounded Metal extend
  attention`), clean except for this Metal candidate. Eight-lane cache runs
  are classified once per C64 tile, published in 32 bytes of threadgroup
  storage, and consumed by direct BF16 SIMD-matrix loads for both QK and PV.
  Ordered adjacency, nonnegative slots, and the upper cache bound govern
  admission; the established staged FP32 path owns every other run.
- Resource change: dynamic threadgroup storage increases from **20,800** to
  **20,832 bytes**. Global auxiliary allocation remains zero, query and
  output accumulation remain FP32, and the operation still writes the
  caller-owned output.
- Attribution at `E=256,L=4352`: an identical ascending map and identical
  seeded inputs produced SHA-256
  `7209fefe46186dbbc09aa0ce1a675b5c3056d6add4091d1bf7622aff84236371`
  in every build. With direct admission forced off at runtime, ten samples
  were `59.514333,59.850917,59.306000,59.403542,59.781000,59.539042,
  59.571667,59.404750,59.627458,59.482666 ms`, median **59.526688 ms**.
  Direct-load windows around that control reached medians **26.801604** and
  **26.900646 ms**, a reproduced **54.81-54.97%** same-map reduction.
- Fragmentation: the first design recomputed eight-slot eligibility inside
  every QK/PV consumer and moved a one-swap-per-eight map from a staged
  **59.839625 ms** median to about **66.746 ms**. The selected cooperative
  classifier removes that repeated work. Its zero-eligible one-swap map
  measured `58.792583,58.860792,58.660459,58.608000,58.772959,58.774167,
  58.576333,58.781916,58.815083,58.412500 ms`, median **58.773563 ms**,
  with output bitwise equal to the direct ascending placement.
- Long-context same-map attribution used `E=17,L=131072`. Forced staged
  samples were `148.002792,148.063042,147.958209,148.010125,147.939625 ms`,
  median **148.002792 ms**. Direct samples were
  `66.655875,66.553042,66.553000,66.457125,66.630042 ms`, median
  **66.553042 ms**, a **55.03%** reduction. Both produced SHA-256
  `0a550d0baacf2a6bbc67c252a7b849db9ae0a227206ccdf725c0b4ecc68306df`;
  post-input current/driver allocation deltas were exactly `0/0 MiB`.
- Correctness: direct and staged paths were bitwise equal at `E=17,L=129`;
  both matched dense attention over the exact BF16 cache values within
  `5.3644180e-07`. Cache storage offsets `0..7`, independent K/V offsets,
  physical run starts `0..7`, and a final run at the allocation boundary all
  produced one identical digest and maximum dense error `1.0728836e-06`.
  Ascending, reverse, interior-swap, duplicate, cross-run-gap, half-mixed,
  and invalid-slot maps were finite and stayed within `6.5565109e-07` of
  dense attention.
- Causal/tile coverage swept `E=1,7,8,9,15,16,17`, total lengths around
  `63/64/65`, and every prefix residue modulo eight. All cases were finite;
  maximum error was `2.2649765e-06`. Replacing later future-cache rows with
  large finite sentinels left the earliest query row bitwise unchanged.
  The analytic `E=5,L=131073` maximum-address case kept the first four rows
  exact zero and matched the final marked physical row within
  `4.7683716e-07`; its three-sample median was **66.350709 ms**, and a
  131,074-row view remained rejected.
- Focused validation rebuilt the native extension with `MAX_JOBS=2`, reported
  capability `True`, passed all six generic torch-native EXTEND tests,
  compiled the two existing Python dispatch modules, and passed
  `git diff --check`.
- Decision: retain PERF-A018 as the selected isolated mechanism. The raw
  binding remains outside `TorchNativeAttnBackend.forward_extend`; production
  serving and context standing remain unchanged while the explicit
  no-new-Python rule keeps activation at an owner-approved dispatch gate.

### 2026-08-23 14:47 PDT - PERF-A019 register-published run classification

- Change: the first two SIMDgroups retain each sanitized mapped slot in its
  loader register, broadcast the first slot within each eight-lane cohort,
  reduce ordered-adjacency predicates with three XOR shuffles, and let cohort
  leaders publish the eight run starts. One threadgroup barrier now publishes
  both `shared_slots` and `shared_runs`; the prior barrier plus 64 shared-int
  classifier reads are removed. Scratch remains **20,832 bytes**.
- Matched `E=256,L=4352` evidence used the same seed, buffers, placement, and
  SHA-256 as PERF-A018. The restored checkpoint median was **27.009396 ms**.
  Candidate windows were **26.398334** and **26.592563 ms**, reproduced
  reductions of **2.26%** and **1.54%**. The zero-eligible checkpoint/candidate
  medians were **58.832500/58.874062 ms**, a **0.071%** movement.
- Matched `E=17,L=131072` checkpoint samples were
  `66.547625,66.595417,66.577542,66.560625,66.599000 ms`, median
  **66.577542 ms**. Candidate windows reached medians **65.565041** and
  **65.493667 ms**; the latter samples were
  `65.493667,65.725375,65.074916,65.593500,65.377875 ms`. Exact output
  SHA-256 and `0/0 MiB` current/driver allocation growth were preserved.
- Correctness: direct and fallback outputs remain bitwise equal at
  `E=17,L=129` and match dense attention within `5.3644180e-07`. Breaking
  every individual lane position in an eight-slot run, reverse order,
  duplicates, invalid slots, and half-mixed maps all remain finite and within
  `7.1525574e-07` of dense attention. Runs at `cache_slots-8` and across
  physical rows `60..67` pass. An alternating direct/fallback two-C64-tile
  map reproduced one digest over 20 calls and stayed within `4.7683716e-07`.
- A full-query-fragment hoist was measured first and rejected. It preserved
  exact output while regressing direct maps to **27.710500/27.846354 ms** and
  the zero-eligible map to **62.621042 ms**. `PERF-FA056` records that closed
  live-register design.
- Focused generic torch-native EXTEND coverage passes all six tests after the
  retained classifier change.
- Decision: retain PERF-A019. The change is internal to the already-isolated
  raw native mechanism; serving reachability and qualified context remain
  unchanged.

### 2026-08-23 14:56 PDT - PERF-A020 hoisted cache-run bases

- Change: each SIMDgroup loads its two QK run starts once per C64 tile and
  computes their 64-bit cache-row offsets once before the D256 loop. Each PV
  key block likewise computes one safe value-row base before its eight output
  fragments. The existing direct/fallback predicates, BF16 matrix loads,
  FP32 arithmetic, scratch, and global-residency contracts remain unchanged.
- At `E=256,L=4352`, matched PERF-A019 baselines were **26.330084** and
  **26.649625 ms** in two source-restored arms. Candidate medians were
  **25.548479** and **25.674125 ms**, reproduced reductions of
  **2.97%** and **3.66%**. Zero-eligible baselines/candidates moved
  `58.738667 -> 57.844396` and `58.786396 -> 58.004959 ms`, improvements of
  **1.52%** and **1.33%**.
- At `E=17,L=131072`, checkpoint medians were **65.647625** and
  **65.687333 ms**. Candidate medians were **63.767667** and
  **63.886417 ms**, reproduced reductions of **2.86%** and **2.74%**.
  Exact SHA-256
  `0a550d0baacf2a6bbc67c252a7b849db9ae0a227206ccdf725c0b4ecc68306df`
  and `0/0 MiB` current/driver allocation growth were preserved.
- Direct/fallback small parity remains bitwise exact at dense maximum error
  `5.3644180e-07`. Cache storage offsets and physical starts `0..7` all retain
  one digest and `1.0728836e-06` maximum error. A 20-call alternating
  direct/fallback two-tile map retains one digest. The analytic physical-row
  131,072 result remains within `4.7683716e-07`; its median improves to
  **63.905375 ms**, and the 131,074-row host fence remains active.
- Two adjacent softmax candidates were rejected before this selection. A
  fully-causal C64 branch was neutral at the diagnostic shape and slightly
  slower at 131K. A cohort-leader prior-state broadcast was likewise neutral
  to slightly slower. `PERF-FA057/058` retain their exact evidence.
- Focused generic torch-native EXTEND coverage passes all six tests.
- Decision: retain PERF-A020. Serving reachability and qualified Apple context
  remain unchanged because this is still an isolated raw native mechanism.

### 2026-08-23 15:11 PDT - PERF-FA059 through PERF-FA063 direct-load barrier sweep

- Baseline: clean signed PERF-A020 at `5fe532b41c`. A fresh rebuild measured
  `E=256,L=4352` at **25.686792 ms** and `E=17,L=131072` at
  **63.915500 ms**. An independent final source-restored rebuild measured
  **25.520083/63.745709 ms**. Both controls retained SHA-256
  `7ec60bdc0d473fba047e79855bb9f95c54358cb1c281135797e2950756116a42`
  and `820241644b85aaa0fed24caf91cffef2254cd1bfd6cb93fab2aaf2b6727d1992`.
- PERF-FA059 removed both leading and trailing `mem_none` barriers around the
  direct PV matrix load. Medians were **25.591584/63.552417 ms**, apparent
  reductions of **0.371%/0.568%** against the opening control.
- PERF-FA060 also removed the QK pair, leaving all direct matrix loads without
  explicit `mem_none` barriers. Medians were **25.847500/63.455208 ms**,
  a **0.626% diagnostic regression** and **0.720%** apparent long reduction.
- PERF-FA061 restored PV synchronization and removed only the QK pair.
  Medians were **25.624625/63.901375 ms**, apparent reductions of
  **0.242%/0.022%**.
- PERF-FA062 retained the leading QK/PV barriers and removed both consumer-side
  barriers. Medians were **25.907875/63.845042 ms**, a **0.861% diagnostic
  regression** and **0.110%** apparent long reduction.
- PERF-FA063 retained the trailing QK/PV barriers and removed both leading
  barriers. Medians were **25.575896/63.756833 ms**, apparent reductions of
  **0.432%/0.248%**.
- Every arm produced the exact opening-control digests and checksum, and each
  reported zero current-memory growth with `0` or `-0.015625 MiB` driver
  movement. The restored final control was faster than every diagnostic arm;
  no arm reached the predeclared reproduced **1%** floor at either shape.
- Decision: reject and revert PERF-FA059 through PERF-FA063. The signed
  PERF-A020 synchronization remains selected, and the direct-load barrier
  branch is closed on this Apple M1 Max/Metal toolchain.

### 2026-08-23 16:25 PDT - PERF-A021 four-row Q2_K batch-one matvec

- Baseline: the live generic-Q2_K control produced actual gate/down medians
  **1.067353/1.085771 ms** before the candidate and
  **1.065125/1.089750 ms** after it. Its five exact `12+256` wall times were
  `29.992152,30.080830,30.080217,30.052984,30.115632 s`, aggregating
  **8.515065 tok/s**.
- Change: two 32-wide SIMDgroups each compute four Q2_K output rows while
  reusing the same 32 activation values. The default-on dispatch is confined
  to aligned complete batch-one cohorts and retains the generic path through
  `SGLANG_MPS_Q2_K_BATCH1_ROWS4=0`.
- Kernel evidence: actual gate/down medians are **0.454833/0.453563 ms** over
  50 synchronized samples per arm. The experiment ledger retains all 300
  individual A/B/A timings.
- Full-model evidence: candidate window one aggregates **9.156475 tok/s**;
  an independent candidate restart aggregates **9.189086 tok/s**, reaches
  **9.194647 tok/s** best hit, and averages **27.859136 s E2E**. The matched
  full-model gain is **7.532647%**; current aggregate standing is
  **7.012249%** above PERF-A016.
- Complete baseline: five exact reasoning-enabled `128+256` streams aggregate
  **22.945718 prompt / 9.156675 generation tok/s**, with **5.578383 s TTFT**
  and **33.426973 s E2E**. Current-source exact `32761+1` passes inside the
  32,768-token BF16 pool at **18.942 prompt tok/s**, **1729.565719 s TTFT**,
  and **1729.565822 s E2E**.
- Correctness: actual-file candidate/control/tail/multi-batch parity, three
  compact-view tests, exact fixed and streaming token counts/digests, sampled
  reasoning, arithmetic `703`, thinking-disabled `READY`, one parsed multiply
  call, tool continuation, and image/audio-disabled reporting pass. The full
  launch rebuilt the exact shader source successfully.
- Decision: retain. Signed commit
  `4dfa1ad3efdfe3f9236aa0ed0c841644ab513859` owns the Metal kernel and
  third-party notice. Root `BENCHMARK.md` owns the current Prompt, Generation,
  TTFT, E2E, and Capacity baseline; the full commands, samples, process state,
  and cleanup are in `notes/experiment-log.md`.

### 2026-08-30 18:29 PDT - PERF-A021 Codex tool gate and profile-capacity repair

- Corrected the machine-local `qwen38-local` profile and catalog from
  unsupported 131,072-token declarations to the served 32,768-token context.
  The configured compaction limit is 30,000; Codex 0.151.0 applies an effective
  29,491-token threshold. Final profile/catalog SHA-256 values begin
  `9706003a` and `680e762e`.
- A fresh independent PERF-A021 restart used the exact 32K launch and the
  repaired artifacts with no capacity override. Codex issued one
  `/bin/zsh -lc pwd`, received `/Users/dcazares/sglang`, consumed the result,
  returned exact visible `CODEX TOOL READY`, and exited zero with 21,537 input,
  413 output, and 379 reasoning-output tokens.
- Post-tool health and language-only reporting passed, cache flush succeeded,
  foreground shutdown exited zero, all four verified PIDs disappeared, port
  30000 became free, memory returned to 92% free, and thermal status remained
  normal.
- The exact-tag client audit found that the 900-second stream-idle bound is
  shorter than near-capacity TTFT, the catalog's sequential-tool flag is
  ignored, and no native `apply_patch` tool is registered. These remain the
  next client-hardening gates; the current evidence covers one read-only
  shell-command/result-consumption round trip.

### 2026-08-30 19:46 PDT - strict Codex editing and compaction continuation

- Added a separate `qwen38-local-hardened` three-artifact overlay while
  preserving the earlier profile/catalog hashes. Strict Codex 0.151.0 loading
  passes; model-visible plugin, agent, skill-instruction, goal, memory, web,
  and MCP tool surfaces are absent or disabled. One observed prompt render
  changed from 32,965 to
  21,457 serialized bytes while retaining repository `AGENTS.md`, environment,
  and user context; its exact render command was not retained.
- A medium-reasoning forced-compaction gate on the immediate predecessor
  hashes recovered from one malformed patch across two observed compaction
  boundaries, successfully added a nonce-bearing file, preserved that nonce in
  final output, and exited zero with 11,093 input, 5,120 output, and 3,943
  reasoning-output tokens. The raw JSONL and warning text were not retained.
- An unmatched low-reasoning clean-edit trial completed the patch on its first
  attempt and exited zero with 4,136 input, 141 output, and 70 reasoning-output
  tokens. Promoting low reasoning into the overlay produced an independent run
  with no reasoning or capacity override at 4,111 input, 118 output, and 47
  reasoning-output tokens; same-value sandbox/approval pins remained on the
  command. The lower user config and rules were not pinned at process start.
- All scratch files were verified and removed, cache flush passed, foreground
  shutdown exited zero, PIDs 27939/28026/28027/28028 disappeared, port 30000
  became free, memory returned to 93% free, and thermal status remained normal.

### 2026-08-30 20:11 PDT - isolated Codex home established the scratch-write gate

- Moved the selected client into the dedicated
  `/Users/dcazares/.codex/qwen38-local-hardened-home`. Its sibling-relative
  config/catalog/instruction hashes were `a764dc28...b9984`,
  `eb15e828...51db1`, and `8a2fe9b...2279`. Strict loading was independent of
  the ordinary user config, `--ignore-rules` excluded user/project exec policy,
  and exact-path untrusted decisions excluded project config and hooks.
- The fixed clean gate used no `-p`, `-c`, `--sandbox`, reasoning, or capacity
  override. Thread `01a055c9-f5e8-7621-9822-acfeaa1a9c45` emitted one
  first-attempt `file_change`, returned exact
  `QWEN38 ISOLATED WRITE READY`, and exited zero at 2,843 input / 260 output /
  184 reasoning-output tokens. The only file was 34 bytes with SHA-256
  `f7ca43b4...6729a`.
- All three bundle hashes and ordinary config/rules hashes remained unchanged
  pre/post. Post-gate health, language-only reporting, cache flush, scratch
  removal, and foreground shutdown passed. PIDs 31773/31778/31779/31780 were
  absent, port 30000 was free, memory was 92% free, and thermal status was
  normal.

### 2026-08-30 20:29 PDT - trusted-repository unified-exec Codex gate

- Exact-tag review aligned the catalog with Codex's exposed unified tools:
  `shell_type=unified_exec`, instruction names `exec_command` and `cmd`, and
  sequential use at task/instruction scope. It also made the repository
  trusted so root `AGENTS.md` reached actual-work prompts while retaining the
  fixed scratch path as untrusted.
- The 20:29 config/catalog/instruction hashes were
  `a1ce8b8e...2981bf`, `862339c1...2ec71`, and `5d59350...9d096`.
  A post-change prompt-input diagnostic rendered the root AGENTS heading and
  exact C++/CUDA-only rule. The qualified repository contained no `.codex`
  project surface, hook file, or rule file, and the dedicated home contained
  no `rules/` directory.
- Thread `01a055da-e3ba-7b03-9384-be480e8ce20c` emitted one first-attempt
  `file_change`, returned exact `QWEN38 ISOLATED WRITE READY`, and exited zero
  at 2,662 input / 113 output / 37 reasoning-output tokens. The only file was
  34 bytes with SHA-256 `f7ca43b4...6729a`; all bundle and ordinary-global
  hashes remained stable.
- Post-gate health, language-only reporting, cache flush, scratch removal, and
  foreground shutdown passed. PIDs 33801/33842/33843/33844 were absent, port
  30000 was free, memory was 92% free with zero throttled pages, and thermal and
  performance status were normal.

### 2026-08-30 20:46 PDT - zsh-startup-isolated Codex gate

- Pinned `ZDOTDIR=/var/empty` in the dedicated shell-environment policy. The
  root-owned 0755 target was empty, `/etc/zshenv` and `/etc/zsh/zshenv` were
  absent, and exact-tag review confirms the setting reaches spawned unified-
  exec shells before `zsh -c` startup.
- Final config/catalog/instruction hashes are `9d7842bb...0409`,
  `862339c1...2ec71`, and `5d59350...9d096`. The exact strict task used
  explicit stdin EOF, emitted one first-attempt `file_change`, returned
  `QWEN38 ISOLATED WRITE READY`, and exited zero at 2,670 input / 115 output /
  39 reasoning-output tokens. The verified 34-byte file retained SHA-256
  `f7ca43b4...6729a`; every governing and ordinary-global hash was stable.
- A separate final prompt-input diagnostic under the stored trusted-repository
  decision rendered root `AGENTS.md` and its exact C++/CUDA-only rule. No
  project `.codex`, hook, rule, override-instruction, or skill sidecar was
  present; dedicated-home rule/skill/override and global agent/system skill
  roots were also absent.
- Health, language-only reporting, cache flush, host patch cleanup, and
  foreground shutdown passed. PIDs 36097/36112/36113/36114 disappeared, port
  30000 and matching process sets were empty, memory returned to 92% free with
  zero throttled pages, and thermal/performance status was normal.

### 2026-08-31 03:23 PDT - PERF-A022 MLX 0.32.2 dependency win

- Change: upgraded the isolated `.venv` from MLX/`mlx-metal` 0.32.0 to 0.32.2
  and raised the existing optional-dependency floor in
  `python/pyproject_other.toml` to `mlx>=0.32.2`.
- Benchmark evidence: the exact affine-q4/q4-KV 131K launch produced
  `19.190, 19.133, 19.131, 19.119, 19.143 tok/s`, mean **19.1432**. The
  matched 0.32.0 baseline was
  `19.046, 19.058, 19.006, 19.090, 19.034`, mean **19.0468**. This is a
  **0.506%** retained gain.
- Correctness evidence: all ten requests completed exact `128+256`, preserved
  reasoning, ended at `finish_reason=length`, and retained output SHA-256
  `df00bc9380ea6fcc5b0c4aea8694a3a01f29ee8e483afe19c9e8202e9c7c146c`
  and reasoning SHA-256
  `1b45e66ba0245377a41c4da996a3a96cc0d7e0957fbaba2d7fd8683a4011939c`.
  The MLX sampling suite passed 26 tests and 11 subtests; TOML parsing and
  `git diff --check` passed.
- Decision: retain. Signed commit `45b50cc4c3` owns the dependency floor.

### 2026-08-31 03:40 PDT - PERF-A024/A025 compact-weight checkpoint screen

- PERF-A024 loaded immutable revision
  `c98bba5926f51fec1c8d8737e577221673f524d7` of the full 27B affine-3-bit
  text checkpoint. Three exact deterministic 131K server samples were
  `17.972, 17.961, 17.941 tok/s`, mean **17.958**. All completed exact counts
  and reproduced output digest
  `2f8a3468...212d`; throughput was about 6.2% below affine q4.
- PERF-A025 loaded immutable revision
  `97ab0819817ab1c61d7d39f9169fc71999915641` of the official MLX MXFP4
  checkpoint. A matched 32-warmup/256-token direct target loop reached
  **18.581608 tok/s**. The affine-q4/q4-KV control reached
  **19.513158 tok/s**, leaving MXFP4 4.774% lower.
- Decision: reject both checkpoint substitutions. PERF-FA066 and PERF-FA067
  retain the provenance and reopening conditions.

### 2026-08-31 03:44 PDT - PERF-A026 native C++ graph candidate

- Change: rebuilt the ignored native MLX engine dylib against MLX 0.32.2 and
  selected the existing `SGLANG_USE_MLX_NATIVE_GRAPH=1` route with affine-q4
  weights, BF16 attention KV, real 131,072 context/token pools, and one request.
- Benchmark evidence: five exact deterministic `128+256` samples were
  `19.359, 19.278, 19.262, 19.286, 19.310 tok/s`, mean **19.2990**. This is
  +0.814% over the selected Python q4-KV endpoint and remains 3.63% below the
  20 tok/s floor.
- Correctness evidence: every request completed exact counts, preserved
  reasoning, ended by length, and retained one deterministic digest. The
  current native C ABI returns argmax token ids directly, while sampled Codex
  requests require logits edits and stochastic top-k/top-p selection.
- Decision: keep as a profiling candidate. Promotion waits for preserved
  sampled semantics and a measured gain above the current BF16 Python route.

### 2026-08-31 03:49 PDT - PERF-A023 BF16 attention KV

- Change: omitted `--mlx-kv-cache-bits 4` from the otherwise exact affine-q4
  131K MLX launch, selecting BF16 attention K/V.
- Benchmark evidence: the direct target loop moved
  **19.513158 -> 19.643293 tok/s**. Five deterministic server samples were
  `19.449, 19.430, 19.418, 19.439, 19.401`, mean **19.4274**. Five ordinary
  sampled samples were `19.283, 19.287, 19.253, 19.284, 19.286`, mean
  **19.2786**. The server gains over q4 KV are 1.485% deterministic and
  1.518% sampled.
- Correctness evidence: every request completed exact `128+256`, preserved
  reasoning, and ended by length. The deterministic digest was stable; all
  five sampled outputs were distinct under temperature 1.0, top-p 0.95,
  top-k 20, and presence penalty 1.5.
- Decision: select as the current real-sampling baseline candidate. The
  remaining measured gap is **0.7214 tok/s / 3.74%**; exact-capacity and
  actual-work promotion gates remain open.

### 2026-08-31 04:18 PDT - PERF-A027/A028/A029 precision-boundary screens

- PERF-A027 loaded immutable full affine-q2 revision
  `33b90b60fd7ba16b668854e049bd65e22d6afddf`. Five deterministic 131K/BF16
  server samples averaged **21.0754 tok/s** and five production-sampled
  samples averaged **20.9032 tok/s**. Sampled arithmetic returned empty
  completion content and the tool request produced no valid parsed call.
- PERF-A028 tested group-128 affine requantization in memory. The 64-module
  down-projection selection reached **19.681944 tok/s** and the broader
  385-module selection reached **19.781749 tok/s** in matched direct BF16-KV
  target loops, leaving insufficient served-workload margin.
- PERF-A029 combined q2 MLP gate/up and linear-attention modules with q4
  embeddings, head, MLP down projections, and full-attention blocks. The
  reloaded, hashed v1 artifact reached **20.526347 tok/s** directly; exact
  deterministic and production-sampled server means were **20.2944** and
  **20.1482 tok/s**. Sampled arithmetic returned `703`. The required tool
  request emitted two malformed calls named `...` and ended by length.
- Decision: reject all three unchanged candidates. The active search narrows
  PERF-A029's q2 linear-attention boundary while retaining its q2 gate/up
  modules, whose raw greedy tool form produced one exact `multiply` call.

### 2026-08-31 05:18 PDT - PERF-A030/A031/A032 actual-work boundary screens

- PERF-A030 retained q2 for all 128 MLP gate/up projections, all 48
  linear-attention qkv projections, all 48 z projections, and the first 27
  linear-attention output projections. The hashed reloaded artifact reached
  **20.412561 tok/s** directly. Radix-disabled deterministic and sampled
  server means were **20.1998** and **20.0626 tok/s**. Four-step cadence moved
  the sampled mean only to **20.0762 tok/s**. Enabling the five-slot unified
  radix cache produced five sampled scores of
  `20.091, 20.092, 20.074, 20.068, 20.081`, mean **20.0812**.
- Standalone arithmetic returned `703`, and the first parsed tool request was
  one exact multiply call. Three sampled continuations produced one
  contradictory truncated answer, one clean answer, and one duplicate tool
  call. The frozen Codex xhigh request prefetched about 6.2K tokens at roughly
  107--111 tok/s and decoded at about **19.2 tok/s**. Its tool attempt was
  invalid for the harness schema, and the next Responses continuation failed
  with `kIOGPUCommandBufferCallbackErrorOutOfMemory` in Metal.
- PERF-A031 measured YoozLabs' q3/q4/q6 precision map at
  **18.553117 tok/s** directly. PERF-A032 measured PocketAiHub group-32 q2 AWQ
  at **20.796459 tok/s** directly, while its full and selectively mixed forms
  all failed the raw tool/output gate.
- Decision: reject all three unchanged candidates. PERF-A033 now isolates the
  existing pre-load MLX buffer-cache cap against the frozen continuation OOM;
  long-prefix decode and repeated tool behavior remain coequal gates.

### 2026-08-31 05:43 PDT - PERF-A033 cache cap and PERF-A034 deferred-state handoff

- Changed only `SGLANG_MLX_CACHE_LIMIT_GB=1` on the radix-enabled early-out27
  v2 launch. Five required sampled short scores were
  `20.069, 20.069, 20.063, 20.068, 20.054 tok/s`, mean **20.0646**. A frozen
  Codex xhigh retry completed its 6,237-token prefill and decoded around
  **19.02--19.28 tok/s** until the bounded client timed out. A deterministic
  two-request replay completed the first 6,257-token request and reproduced a
  Metal OOM on the immediate continuation. Halving prefill chunks from 512 to
  256 reproduced the same request-boundary failure. PERF-FA075 and PERF-FA076
  close both unchanged controls.
- A matched full affine-q2 memory control started with about **24.12 GB**
  available GPU memory versus **21.65 GB** for v2 and completed both bounded
  requests in **57.387437 s** and **64.289053 s**. The second request reported
  6,144 cached tokens and processed 131 new tokens at only **2.04 tok/s**,
  proving that lower weight residency survives while the matched recurrent
  state is still recomputed.
- Reachability tracing found that
  `ScheduleBatch._collect_deferred_mamba_cow_and_clear` publishes source and
  destination auxiliary-state indices. The generic model runner executes the
  deferred copy, while `MlxTPWorker.async_forward_batch_generation_mlx` never
  consumes those fields. Its restore therefore sees an empty destination and
  `prefill_start` falls back to the complete prompt. PERF-A034 owns the missing
  state handoff; the implementation must remain within the repository's native
  C++/CUDA boundary.

### 2026-08-31 06:04 PDT - PERF-A035 mixed-width native checkpoint loading

- Change: inferred every affine projection's stored bit width from its packed
  weight and scale shapes inside the native C++ loader, including the embedding
  table, while rejecting malformed or unsupported layouts.
- Benchmark evidence: the early-out27 v2 artifact reached **20.219228 tok/s**
  in the direct native loop. Five deterministic server samples were
  `20.147, 20.125, 20.121, 20.105, 20.095 tok/s`, mean **20.1186 tok/s**.
- Correctness evidence: the direct token digest exactly matched the Python MLX
  v2 digest beginning `90c684d7`; server output retained one deterministic
  digest, arithmetic returned `703`, and a 256-token probe produced exactly one
  parsed `multiply({"a":37,"b":19})` call with `finish_reason=tool_calls`.
- Decision: retain the native loading capability. Signed commit `42ee99493e`
  owns the change. Native sampling and realistic-context speed remain separate
  production gates.

### 2026-08-31 06:14 PDT - PERF-A036 exact native prompt-state reuse

- Change: retained C++ attention and recurrent state only when the next full
  prompt has exact token history as a strict prefix; reset the decode pipeline
  and process only the nonempty suffix. MTP continues to take the hard-reset
  path.
- Benchmark evidence: a direct continuation with 16 new prompt tokens and 32
  generated tokens took **1.927515 s**, versus **3.063133 s** for a fresh full
  prefill, reducing latency **37.074%**. A served 6,257-token first request took
  **58.459993 s**; its exact-prefix continuation with 16 new tokens took
  **0.416196 s** and returned the same token.
- Correctness evidence: direct retained-state and fresh-prefill output lists
  were identical, with SHA-256
  `9ab2e8830abbe71df5999d71a2b6a90eff4b1c3dcb11eee7ca67b81304255413`.
  The focused native suite passed **8 tests**.
- Decision: retain. Signed commit `24686a37b1` owns the exact-prefix state
  contract.

### 2026-08-31 06:24 PDT - PERF-A037 reusable native attention storage

- Change: replaced per-token full-attention K/V concatenation with growable
  power-of-two BF16 storage and in-place slice updates. Rope offset, logical
  cache length, and physical capacity are tracked independently so MTP draft
  position and target snapshot/restore semantics remain exact.
- Benchmark evidence: at a deterministic 6,237-token history, the committed
  concatenation control reached **17.923409 tok/s** and the candidate reached
  **19.151623 tok/s**, a **6.853%** gain. Both used 32 warm tokens and 128 timed
  tokens on the same process-isolated workload.
- Correctness evidence: both arms produced exact SHA-256
  `382dd93cb724783226eae6ede000d6b62bbbc6439c8a39178cb9bb0ba8a27112`.
  Exact-prefix retained-state output still matched a fresh prefill, and the
  focused native suite passed **8 tests**.
- Decision: retain. Signed commit `5ac91e2f22` owns the reusable cache.

### 2026-08-31 06:40 PDT - PERF-A038 native split-attention topology screen

- Change: screened opt-in custom MLX Metal decode attention over the committed
  contiguous cache: per-query online split-K, paired-head shared-K/V split-K,
  and the retained 8-by-64 tiled simdgroup-matrix algorithm with 8, 16, and 32
  history splits. Experimental source was removed after measurement.
- Benchmark evidence: per-query split-16 reached **15.931997 tok/s**;
  paired-head split-16 and split-32 reached **13.068068** and
  **15.863200 tok/s**. Tiled 8/16/32 reached **18.475598**,
  **19.117317**, and **18.922351 tok/s**. The matched MLX SDPA control remains
  **19.151623 tok/s**.
- Correctness evidence: every completed arm reproduced exact 128-token SHA-256
  `382dd93cb724783226eae6ede000d6b62bbbc6439c8a39178cb9bb0ba8a27112`.
- Decision: reject these unchanged custom topologies. PERF-FA078 and
  PERF-FA079 retain their reopening criteria.

### 2026-08-31 07:46 PDT - PERF-A039 materialized affine gate/up rows

- Change: concatenated each native target and MTP MLP's packed affine gate/up
  weights, scales, and biases during load, then replaced two quantized matmuls
  with one double-height product and a result split.
- Benchmark evidence: a process-isolated committed control at `14fd46b11a`
  served exact deterministic `6237+128` at **18.845 tok/s**, **58.100626 s**
  TTFT, and **64.839787 s** end to end. The adjacent fused candidate served
  the same request at **18.782 tok/s**, **58.605376 s** TTFT, and
  **65.367119 s** end to end. This is **-0.334%** decode throughput. Startup's
  reported available unified memory changed **28.92 -> 22.28 GB**.
- Correctness evidence: both arms completed exact `6365` total tokens with
  `finish_reason=length` and identical output/reasoning SHA-256
  `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5`.
  The native suite passed **8 tests** before the served A/B.
- Decision: reject and remove the materialized concatenation. It consumes
  long-context residency without reducing the measured decode wall. Reopen
  only for a native kernel that reads the two original affine tensors directly
  in one launch without duplicating their storage.

### 2026-08-31 07:52 PDT - PERF-A040 linear-attention b/a row fusion

- Change: concatenated only the two 48-row affine `in_proj_b` and `in_proj_a`
  tensors in each of 48 linear-attention layers, replacing their two
  quantized-matmul calls with one 96-row call and a split.
- Benchmark evidence: the immediately preceding process-isolated control
  reached **18.845 tok/s** on exact deterministic `6237+128`. The candidate
  reached **18.511 tok/s**, **58.055139 s** TTFT, and **64.915750 s** end to
  end, a **1.772%** decode regression. Startup retained **28.89 GB** reported
  available unified memory, isolating execution topology from PERF-A039's
  large materialization peak.
- Correctness evidence: the candidate completed exact `6365` total tokens with
  `finish_reason=length` and the control's output/reasoning SHA-256
  `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5`.
  The focused native suite passed **8 tests** before serving.
- Decision: reject and remove. Independent MLX projection scheduling is faster
  than the combined 96-row quantized matmul at this batch-one shape. PERF-FA082
  retains the distinct evidence and reopening condition.

### 2026-08-31 08:09 PDT - PERF-A041 direct native benchmark harness

- Added `benchmark/mac/bench_qwen38_native.cpp`, a C++20 executable that loads
  the native engine C ABI, generates deterministic prompt IDs, performs an
  explicit warmup, synchronizes MLX around the measured decode interval, and
  reports a stable little-endian FNV-1a token digest.
- The synchronized exact `6237 / 32 warm / 128 timed` validation measured
  **19.116578 tok/s** and digest `13eb9a7159a2612f`. The signed historical
  direct control is **19.151623 tok/s** with the same digest, a **0.183%**
  difference. A `128 / 8 / 16` smoke run measured **20.270547 tok/s** and
  digest `a3cb71cd75683b5f` before the final synchronization placement.
- Strict compilation passed with `-std=c++20 -O3 -Wall -Wextra -Werror`; the
  existing MLX macOS 26.2 versus local 26.0 link warning remained. Missing
  arguments return status 2 with usage text, and all eight focused native
  engine tests pass.
- MLX's process-lifetime compile cache retains primitives implemented by the
  engine dynamic library. The benchmark therefore keeps its `dlopen` handle
  alive through process teardown. This removed the exit-time compiled-cache
  destructor crash seen when the library was unloaded first.
- A Metal System Trace capture succeeded for a short direct run. Its default
  encoder tables expose generic command-buffer and compute-command labels;
  shader timeline data is disabled in that template, so the trace supplies
  dispatch-density evidence while stage attribution still needs an engine-side
  measurement seam.

### 2026-08-31 08:37 PDT - PERF-A042 fused decode convolution state

- Added a single-token custom Metal owner for the 48 recurrent layers. It reads
  the existing three-row causal-convolution state and new QKV row, accumulates
  the same four BF16 taps in float, rounds the convolution result to BF16, and
  emits a distinct shifted next-state allocation from one launch. Multi-token
  prefill retains MLX's general concatenate, slice, and depthwise-convolution
  path.
- Five interleaved process-isolated short-history `128 / 32 warm / 256 timed`
  controls measured **20.205545, 20.045465, 20.168646, 20.161979,
  20.152875 tok/s**, mean **20.146902**. Candidates measured **20.240219,
  20.216071, 20.211578, 20.222553, 20.220025**, mean **20.222089 tok/s**,
  a **0.373%** increase. Every run produced digest `8ea2430e3fa3d56e` and
  last token `198`.
- Five interleaved process-isolated 6,237-history controls measured
  **19.142256, 19.143054, 19.069096, 19.125504, 19.120243 tok/s**, mean
  **19.120031**. Candidates measured **19.189544, 19.229916, 19.237295,
  19.223291, 19.227897**, mean **19.221589 tok/s**, a **0.531%** increase.
  All ten long runs produced digest `faaecee6edebe116` and last token `19360`.
- The new C++ parity executable compares the production custom kernel with
  MLX's BF16 concatenate, depthwise `conv1d`, and state slice. Production
  width `(B=1,K=4,D=10240)` and a non-tile-aligned `(B=2,K=3,D=257)` case
  match exactly for convolution and next-state output. Strict compilation
  passes with MLX treated as a system include; all eight focused native engine
  tests pass.
- The direct full-model gate clears the candidate for a signed recovery point.
  Process-isolated served qualification with real 131,072 context/token pools
  remains the next promotion gate.

### 2026-08-31 08:53 PDT - PERF-A042 served qualification

- Qualified signed candidate `6ad2c58921` against a detached signed
  `bd52acb255` control. Both native libraries were built from their respective
  source trees against MLX 0.32.2. The control dylib SHA-256 was
  `e0ae523b64663a156224c1030fae9fc99477de0ad292b5618caa1f02ccaed1ca`;
  the candidate dylib SHA-256 was
  `42471b53aa2c7df08fa35a9f4bbe0024ef3c60741e00b21fd822f854377df9ba`.
- Both process-isolated servers resolved `context_length=131072`,
  `max_total_tokens=131072`, `max_running_requests=1`,
  `max_mamba_cache_size=5`, and 8,192-token prefill chunks. They retained the
  Qwen3 reasoning parser, Qwen3 Coder tool parser, language-only model surface,
  and startup-reported 28.92 GB available unified memory. `/model_info`
  reported image and audio understanding disabled.
- Five control decode samples were **18.796, 18.688, 18.719, 18.844,
  18.761 tok/s**, mean **18.7616**. Five candidate samples were **18.806,
  19.003, 18.847, 18.918, 18.883 tok/s**, mean **18.8914**, a
  **0.1298 tok/s / 0.692%** increase. Mean prompt throughput changed
  **107.3162 -> 107.4536 tok/s**, mean TTFT **58.118316 -> 58.043591 s**,
  and mean end-to-end latency **64.887544 -> 64.766309 s**.
- All ten requests completed exact `6237+128=6365` tokens with
  `finish_reason=length`, 585 reasoning characters, 33 stream fragments, and
  identical output/reasoning SHA-256
  `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5`.
  Both foreground server trees exited cleanly through `Ctrl+C`; their exact
  PIDs were absent afterward, port 30000 was free, compiler/model process scans
  were empty, throttled pages remained zero, and thermal/performance status was
  normal. The detached control worktree was clean and removed, and the signed
  candidate library was restored.

### 2026-08-31 09:01 PDT - PERF-A043 native MTP telemetry

- Extended the standalone C++ benchmark with an optional MTP sidecar argument.
  It resolves the existing load/availability/refill-width C ABI, selects the
  engine's MTP prefill contract, and reports refill count plus mean emitted
  block width. The original five-argument path retains its scheduled target-
  only decode pipeline.
- The selected fused-convolution target-only control measured
  **12.680987083 s / 20.187702923 tok/s** on `128 / 32 warm / 256 timed`, with
  digest `8ea2430e3fa3d56e` and last token `198`. The locally pinned
  `mlx-community/Qwen3.8-27B-MTP-4bit` sidecar at revision
  `b643c01b6d3b094e325edb6ebd832e16c486c575` measured
  **26.528607417 s / 9.649959984 tok/s** on the same shape. It issued 135
  target refills with mean emitted width **1.888888889** and reproduced the
  exact control digest and final token.
- The acceptance signal is healthy while the recurrent multi-token target
  verification plus two sequential draft forwards more than doubles the
  per-output wall. PERF-FA083 closes the unchanged MTP-2 topology. A later MTP
  branch first needs a native recurrent verification mechanism whose measured
  execution cost changes this block-level result.

### 2026-08-31 09:26 PDT - PERF-A044 fused residual RMSNorm

- Added a single-token dual-output Metal owner for decoder residual addition
  followed by RMS normalization. It computes the BF16 residual with the same
  addition type, accumulates the exact MLX v0.32.2 four-value looped reduction,
  uses `metal::precise::rsqrt`, and emits separately allocated residual and
  normalized outputs. Each thread retains its residual fragments across the
  reduction barrier. Multi-token prefill and target verification retain the
  established MLX operations.
- The decode loop carries each fused input-normalized row into the following
  layer. This covers all 64 attention-to-MLP boundaries and 63 inter-layer MLP
  boundaries; the initial embedding normalization and terminal residual/final
  normalization remain separate.
- A strict C++20 parity executable matches MLX exactly at production BF16
  width 5,120 and a three-row width-257 case. Twelve unevaluated calls with
  distinct residual inputs remain exact after delayed evaluation, covering
  asynchronous output lifetime. The focused native engine suite passes all
  eight tests.
- A detached `c2f1cc780d` control and the candidate used six interleaved,
  process-isolated `6237 / 32 warm / 256 timed` pairs. Controls were
  **19.240067, 19.231638, 19.227116, 19.189970, 18.417971, 19.223874 tok/s**;
  candidates were **19.285124, 19.320820, 19.330695, 19.309766, 19.318736,
  19.307879 tok/s**. Every candidate exceeded its adjacent control and all
  twelve runs produced digest `faaecee6edebe116`, last token `19360`.
- The fifth control is an isolated unclassified outlier outside the otherwise
  narrow control band; its adjacent candidate immediately returned to the
  candidate band. All six means are **19.088439 -> 19.312170 tok/s**
  (+1.172%). The five controls inside the repeated band average **19.222533**;
  all six candidates average **19.312170**, while the corresponding five
  candidates average **19.310857 tok/s**, a conservative **0.459%** increase.
  Served qualification with real 131,072 context/token pools is next.

### 2026-08-31 09:42 PDT - PERF-A044 served qualification

- Qualified signed `4905d68370` against detached signed control
  `c2f1cc780d`. The control dylib SHA-256 was
  `62d1b10fb8423c0ccc090f5f8ef61a949678131d6b70c016e80a4cff84581762`;
  the candidate was
  `593367be85f60377ee2ede15b7da8f14c1f77931e9593d71e0fa141e746879ff`.
  Both isolated servers resolved real 131,072 context/token pools, one request,
  five Mamba slots, 8,192-token prefill chunks, both Qwen parsers, language-
  only mode, and 28.92 GB startup-reported available unified memory.
- Five control decode samples were **18.737, 18.826, 18.816, 18.933,
  18.831 tok/s**, mean **18.8286**. Candidate samples were **19.041, 19.082,
  18.993, 19.032, 19.094 tok/s**, mean **19.0484**, a **0.2198 tok/s /
  1.167%** increase. Mean prompt throughput changed **107.0234 -> 107.0572
  tok/s**, mean TTFT **58.276943 -> 58.258711 s**, and mean end-to-end latency
  **65.022026 -> 64.925940 s**.
- All ten requests completed exact `6237+128=6365`, `finish_reason=length`,
  585 reasoning characters, 33 response fragments, and output/reasoning
  SHA-256 `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5`.
  Both foreground servers stopped through `Ctrl+C`; port 30000 and matching
  model processes were absent, memory returned to 93% free with zero throttled
  pages, and thermal/performance status remained normal. The clean detached
  control worktree was removed and is recoverable from `c2f1cc780d`.

### 2026-08-31 10:04 PDT - PERF-A045 fused recurrent q/k normalization

- Added one dual-output single-token Metal owner for the q/k RMS normalization
  and float scaling repeated by all 48 gated-delta layers. It uses MLX v0.32.2's
  four-value BF16 RMS reduction order and `metal::precise::rsqrt`, rounds each
  normalized value to BF16, then emits the established float32 scaled q/k
  values. Multi-token forwards continue through the general MLX operations.
- The strict C++20 parity executable matches the established operations exactly
  at production `(rows=16,width=128)` and at `(rows=3,width=257)`. Twelve
  unevaluated dual-output calls remain exact after delayed evaluation. The
  focused native engine suite remains **8 passed** with its existing warnings.
  The first build invocation found the retired `.venv-mps` default absent; the
  explicit installed MLX prefix build passed with the known macOS 26.0/26.2
  link warning.
- A clean detached signed `80ef84925e` control produced dylib SHA-256
  `eb18f32aa07e1e9507c8da8c213137857b96a06d5e4e639625217a67d72c0f2c`;
  the candidate was
  `b987bd6df0b87e6f5a55a1bb33365bb7376ac04e4cff9fea8b0358903b4ab947`.
  Five adjacent process-isolated control samples were **19.295295, 19.321204,
  19.330269, 19.311036, 19.337509 tok/s**, mean **19.319062**. Candidate
  samples were **19.436754, 19.473608, 19.475633, 19.464942, 19.472754
  tok/s**, mean **19.464738**, a **0.145676 tok/s / 0.754052%** increase.
  Every run produced digest `faaecee6edebe116`, last token `19360`.
- Port 30000 remained free throughout the direct window. Post-run memory had
  zero throttled pages and thermal/performance status remained normal. The
  detached control stays available for matched served qualification under real
  131,072 context/token pools.

### 2026-08-31 10:19 PDT - PERF-A045 served qualification

- Qualified signed `b851d3c9de` against detached signed control `80ef84925e`.
  Served dylib SHA-256 values were
  `20fbcab3321afcc0ac73eabc4573bbcd306148da548c8df9eccb60b25456e234`
  for the control and
  `b987bd6df0b87e6f5a55a1bb33365bb7376ac04e4cff9fea8b0358903b4ab947`
  for the candidate. Both isolated launches resolved real 131,072 context and
  token pools, one running request, five Mamba slots, 8,192-token chunks, both
  Qwen parsers, language-only mode, and 28.92 GB startup-reported available
  unified memory.
- Control decode samples were **19.128, 19.082, 19.062, 19.094, 19.084
  tok/s**, mean **19.0900**. Candidate samples were **19.152, 19.144, 19.159,
  19.177, 19.173 tok/s**, mean **19.1610**, a **0.0710 tok/s / 0.371922%**
  increase. Mean prompt throughput changed **107.4084 -> 107.4140 tok/s**,
  mean TTFT **58.068098 -> 58.065245 s**, and mean end-to-end latency
  **64.720796 -> 64.693245 s**.
- Every paired candidate sample exceeded its control. All ten requests completed
  exact `6237+128=6365`, `finish_reason=length`, 585 reasoning characters, 33
  response fragments, and output/reasoning SHA-256
  `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5`.
  Both foreground servers exited zero after `Ctrl+C`; port 30000 and matching
  model processes were absent, memory returned with zero throttled pages, and
  thermal/performance status remained normal. The detached control worktree was
  removed and remains reproducible from signed `80ef84925e`.

### 2026-08-31 10:50 PDT - PERF-A046 fused recurrent norm/gate

- Added a single-token Metal owner for the recurrent output RMS normalization
  and `z * sigmoid(z)` gate repeated by all 48 gated-delta layers. It retains
  MLX v0.32.2's four-value float32 RMS reduction and
  `metal::precise::rsqrt`; multi-token forwards keep the general MLX graph.
- The first custom exponential reproduced ordinary activations exactly while an
  extreme queued case differed by one BF16 ULP and changed the short full-model
  digest from `8ea2430e3fa3d56e` to `a6562984e670e10e`. Widened `[-113,113]`
  parity inputs isolated the exponential. `metal::precise::exp` restored exact
  parity and the established token stream. The strict C++20 test passes at
  production `(rows=48,width=128)`, nonaligned `(rows=3,width=257)`, and twelve
  delayed outputs. The focused native engine suite remains **8 passed** with
  its existing 16 warnings.
- The corrected short full-model gate reached **20.590855470 tok/s**, digest
  `8ea2430e3fa3d56e`, last token `198`. A clean detached signed `068f9ca072`
  control produced dylib SHA-256
  `746e230fcee3f328fd2cb54d329210bb8dd8db07ff4a5617270f4c865870f7c2`;
  the candidate was
  `5865de3c7e3956986076ebc3a57aadcb84e2058cfad3fa42a0a6eb244bf31a1d`.
- Five adjacent process-isolated long-history control samples were
  **19.478588247, 19.453995931, 19.478673901, 19.480679024,
  19.453745231 tok/s**, mean **19.469136467**. Candidate samples were
  **19.516119108, 19.516004982, 19.516870482, 19.532189316,
  19.497406122 tok/s**, mean **19.515718002**, a **0.046581535 tok/s /
  0.239258%** increase. Every candidate exceeded its adjacent control; all ten
  runs produced digest `faaecee6edebe116`, last token `19360`.
- Port 30000 remained free throughout the direct window. Post-run memory was
  93% free, no matching workload remained, and thermal/performance status was
  normal. The detached control remains available for matched served
  qualification under real 131,072 context/token pools.

### 2026-08-31 11:05 PDT - PERF-A046 served qualification

- Qualified signed candidate `28174b3da2` against detached signed
  `068f9ca072`. The control dylib SHA-256 was
  `746e230fcee3f328fd2cb54d329210bb8dd8db07ff4a5617270f4c865870f7c2`;
  the candidate was
  `5865de3c7e3956986076ebc3a57aadcb84e2058cfad3fa42a0a6eb244bf31a1d`.
- Both isolated servers resolved real 131,072 context and token pools, one
  running request, five Mamba slots, 8,192-token prefill chunks, the Qwen3 and
  Qwen3 Coder parsers, language-only mode, and 28.92 GB startup-reported
  available unified memory. `/health`, `/v1/models`, and `/model_info` passed;
  image and audio understanding remained disabled.
- Control decode samples were **19.055, 19.045, 19.200, 19.185, 19.145
  tok/s**, mean **19.1260**. Candidate samples were **19.109, 19.117, 19.254,
  19.177, 19.117 tok/s**, mean **19.1548**, a **0.0288 tok/s / 0.150580%**
  increase. Mean prompt throughput was **107.4042 -> 107.4040 tok/s**, mean
  TTFT **58.070235 -> 58.070435 s**, and mean end-to-end latency
  **64.710486 -> 64.700673 s**.
- All ten requests completed exact `6237+128=6365`, `finish_reason=length`,
  585 reasoning characters, 33 response fragments, and output/reasoning
  SHA-256 `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5`.
  Both foreground server trees exited zero through `Ctrl+C`; port 30000 and
  matching workloads were absent afterward, memory returned to 93% free, and
  thermal/performance status remained normal. The clean detached control
  worktree was removed and is reproducible from signed `068f9ca072`.

### 2026-08-31 11:25 PDT - PERF-A047 fused recurrent convolution/SiLU

- Extended the existing single-token causal-convolution/state Metal owner to
  emit the activated convolution row. It first rounds the four-tap accumulator
  to BF16, computes the stable BF16 sigmoid with `metal::precise::exp`, rounds
  that result to BF16, and performs the established BF16 multiply. The shifted
  next-state output remains distinct. Multi-token forwards retain MLX `conv1d`
  and SiLU.
- The first float32 sigmoid formulation failed strict production-shape parity
  at element 3 (`0.155273` versus `0.15625`). Matching MLX's templated BF16
  sigmoid arithmetic restored exact output. The strict C++20 test passes at
  production `(B=1,K=4,D=10240)`, nonaligned `(B=2,K=3,D=257)`, and exact
  convolution outputs `[-23,23,-113,113]`. The focused native engine suite
  remains **8 passed** with its existing 16 warnings.
- The corrected short full-model gate reached **20.630307745 tok/s**, digest
  `8ea2430e3fa3d56e`, last token `198`. A clean detached signed `8dcf68177c`
  control produced dylib SHA-256
  `2ae6591c347f5ff1665d20c510bdd2e22788407d58fe844834c96f6a01af76ea`;
  the candidate was
  `16b58ce3056617b49582a180078181fe62442b404085c27910d0aed9fdeca703`.
- Five adjacent process-isolated long-history control samples were
  **19.551438201, 19.531679410, 19.531585717, 19.471752586,
  19.533882708 tok/s**, mean **19.524067724**. Candidate samples were
  **19.646667601, 19.610726755, 19.638978864, 19.622965103,
  19.643077544 tok/s**, mean **19.632483173**, a **0.108415449 tok/s /
  0.555291%** increase. Every candidate exceeded its adjacent control; all ten
  runs produced digest `faaecee6edebe116`, last token `19360`.
- Port 30000 remained free throughout the direct window. Post-run memory was
  93% free, no matching workload remained, and thermal/performance status was
  normal. The detached control remains available for matched served
  qualification under real 131,072 context/token pools.

### 2026-08-31 11:41 PDT - PERF-A047 served qualification

- Qualified signed candidate `4c1bc4c1e3` against detached signed control
  `8dcf68177c`. The control dylib SHA-256 was
  `2ae6591c347f5ff1665d20c510bdd2e22788407d58fe844834c96f6a01af76ea`;
  the candidate was
  `16b58ce3056617b49582a180078181fe62442b404085c27910d0aed9fdeca703`.
- Both isolated servers resolved real 131,072 context and token pools, one
  running request, five Mamba slots, 8,192-token prefill chunks, the Qwen3 and
  Qwen3 Coder parsers, language-only mode, inactive MTP, disabled graph
  capture, and 28.92 GB startup-reported available unified memory. `/health`,
  `/v1/models`, and `/model_info` passed; image and audio understanding
  remained disabled.
- Control decode samples were **19.119, 19.233, 19.116, 19.122, 19.275
  tok/s**, mean **19.1730**. Candidate samples were **19.199, 19.224, 19.215,
  19.217, 19.212 tok/s**, mean **19.2134**, a **0.0404 tok/s / 0.210713%**
  increase. Mean prompt throughput was **107.4068 -> 107.4000 tok/s**, mean
  TTFT **58.069039 -> 58.072705 s**, and mean end-to-end latency
  **64.692994 -> 64.682682 s**.
- All ten requests completed exact `6237+128=6365`, `finish_reason=length`,
  585 reasoning characters, 33 response fragments, and output/reasoning
  SHA-256 `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5`.
  Control root process `72627` and candidate root process `72769` exited zero
  after foreground `Ctrl+C`; expected child `KeyboardInterrupt` traces
  accompanied shutdown. Port 30000 and matching workloads were absent
  afterward, memory returned to 92% free with zero throttled pages, and
  thermal/performance status remained normal. The clean detached control
  worktree was removed and is reproducible from signed `8dcf68177c`. The
  selected served mean is **19.2134 tok/s**, leaving **0.7866 tok/s** to the
  required floor.

### 2026-08-31 11:51 PDT - PERF-A048 full-attention affine row fusion rejected

- Materialized q+gate, k, and v q4 rows into one detached affine projection at
  load time, replacing three products in each of 16 full-attention layers. A
  strict synthetic C++20 test reproduced separate-product float32 outputs
  bit-for-bit across six input rows and verified that the realized packed
  tensors detached from their source graphs.
- The exact short full-model screen reached **12.354390750 s /
  20.721377944 tok/s**, last token `198`, while digest changed from control
  `8ea2430e3fa3d56e` to `12bb3edf3d51feac`. Keeping q+gate separate and
  combining only the equal-shaped k/v products reached **12.383738500 s /
  20.672271140 tok/s** with digest `af06cc7ce5e094be`.
- Production output geometry therefore changes MLX's affine accumulation path
  even when packed rows, scales, biases, and per-row quantization remain
  identical. Both variants failed the first exactness gate. The engine and test
  diffs were removed before any long-history or served timing window.

### 2026-08-31 12:43 PDT - PERF-A049 recurrent beta/decay fusion rejected

- Extended the exact single-token recurrent q/k normalization Metal owner to
  emit `sigmoid(b)` and the compiled `compute_g` decay values for all 48
  gated-delta layers. The retained multi-token path continued through the
  general MLX graph.
- A beta-only five-pair window was neutral: controls averaged
  **19.602405132 tok/s** and candidates averaged **19.600900659 tok/s**, a
  **-0.001504472 tok/s / -0.007675%** change. The full beta/decay owner was
  exact after matching MLX compiled softplus/log-add-exp arithmetic and
  separating q/k and decay input types.
- Five adjacent full-candidate controls measured **19.644892037,
  19.594597286, 19.665576690, 19.652227855, 19.650983123 tok/s**, mean
  **19.641655398**. Candidates measured **19.494418723, 19.583726912,
  19.555902888, 19.570241386, 19.533772471 tok/s**, mean **19.547612476**.
  Every pair favored the separate MLX graphs; the candidate changed throughput
  by **-0.094042922 tok/s / -0.478793%**. All ten outputs retained digest
  `faaecee6edebe116` and last token `19360`.
- Serializing the scalar exponentials inside the q/k normalization dispatch
  removes MLX scheduling overlap and costs more wall time than its saved
  launches. The source and expanded test diff were removed. A selected-source
  rebuild restored digest `8ea2430e3fa3d56e`, last token `198`, at
  **20.689765463 tok/s** on the short screen.

### 2026-08-31 13:21 PDT - PERF-A050 native command-buffer byte budget

- Raised the native Qwen3.8 engine's default MLX command-buffer byte budget
  from the M1 Max architecture default of 50 MiB to 128 MiB. The constructor
  installs the default while its first member initializes, before any MLX array
  can create the Metal device. A process-level `MLX_MAX_MB_PER_BUFFER` value
  keeps precedence.
- The short exact candidate measured **21.025485404 tok/s**, digest
  `8ea2430e3fa3d56e`, last token `198`. An explicit 50 MiB override measured
  **20.663320571 tok/s** with the same output, confirming the compatibility
  control. The focused native suite passed **8 tests** with 16 existing
  warnings.
- Five adjacent long-history controls measured **19.645876112,
  19.578931372, 19.640621068, 19.602503035, 19.648569033 tok/s**, mean
  **19.623300124**. Candidates measured **20.174844979, 20.120396667,
  20.187648928, 20.157111892, 20.129828219 tok/s**, mean **20.153966137**.
  Pair deltas were **+0.528968867, +0.541465295, +0.547027860,
  +0.554608857, +0.481259186 tok/s**. The mean gain is
  **+0.530666013 tok/s / +2.704265%**. Every candidate clears 20; all ten runs
  retained digest `faaecee6edebe116`, last token `19360`.

### 2026-08-31 13:42 PDT - PERF-A050 served process-start qualification

- A no-environment server sample established that MLX creates its Metal device
  before the native `Engine` constructor on the SGLang process path: the
  internal default reached **19.257 tok/s**, inside the selected 50 MiB served
  range. The native constructor remains effective for standalone engine users.
- Supplying the same 128 MiB setting at process start produced five exact
  client decode samples of **19.917, 19.945, 19.943, 19.938, and 19.940
  tok/s**, mean **19.9366**. The matched 50 MiB window averaged **19.2260
  tok/s**, making the served gain **0.7106 tok/s / 3.696%**.
- Mean prompt throughput was **107.0978 tok/s**, mean TTFT was
  **58.236469 s**, and mean end-to-end latency was **64.606635 s**. Every
  candidate request completed exact `6237+128=6365`, `finish_reason=length`,
  585 reasoning characters, 33 fragments, and output/reasoning SHA-256
  `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5`.
- The selected process-start serving configuration retains the 3.696% win and
  leaves **0.0634 tok/s** to the required client-observed floor. Server
  telemetry settled around 20.11--20.20 tok/s.

### 2026-08-31 13:59 PDT - PERF-A051 SDPA block override rejected

- Official MLX v0.32.2 source selects 128 SDPA blocks for this M1 Max decode
  shape. Exact direct screens at 32/64/96/128 blocks measured
  **19.116925846 / 20.419125823 / 19.541540885 / 20.139173026 tok/s**; a
  second 64-block screen reached **20.391260024 tok/s**. The direct token
  digest remained exact in every screen.
- A global native 64-block default reached **20.146 client tok/s** on the real
  131K server. Restricting the override to decode after fully materializing
  the adaptive-block prefill reached **20.127 client tok/s**. Both responses
  completed exact token counts and `finish_reason=length` while changing the
  established output/reasoning digest from
  `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5` to
  `69f3577805ed5ae85d2f8253eb3ec10f89ac7246897d6d5de9061ee6f665715c`.
- The reduction-topology override is therefore closed under the deterministic
  exactness gate. Both experimental forms were removed. Rebuilding selected
  source restored dylib SHA-256
  `6c6df982252d170426ee3a1d505c9157bd4ebf5f0e17cd8d2d3a4193d8671688`;
  its short confirmation retained digest `8ea2430e3fa3d56e`, last token `198`,
  at **21.135133773 tok/s**.

### 2026-08-31 14:18 PDT - PERF-A052 full-attention q/k norm and RoPE fusion

- Added a single-token Metal owner for the 16 full-attention layers' q/k
  RMS normalization, head transpose, and partial rotary embedding. One
  threadgroup owns each head, reproduces MLX's four-value reduction order,
  precise reciprocal square root, BF16 normalization boundary, and fast
  rotary trigonometry. Multi-token prefill retains the established MLX graph.
- The installed-MLX build passed with the known macOS 26.0/26.2 link warning.
  The candidate dylib SHA-256 is
  `5bbb6d04bb3d905c209f98758e773e35a9bac88a5b2f11346571cccf67bf34b4`.
  The focused native engine suite passed **8 tests** with its existing 16
  warnings, and `git diff --check` passed.
- The exact short direct screen reached **21.228266681 tok/s**, digest
  `8ea2430e3fa3d56e`, last token `198`. The exact 6,237-history direct screen
  reached **20.201697377 tok/s**, digest `faaecee6edebe116`, last token
  `19360`; the qualified A050 direct mean is **20.153966137 tok/s**.
- Five sequential real-131K client samples measured **19.997, 19.989,
  19.990, 19.984, and 19.983 tok/s**, mean **19.9886 tok/s**. This is a
  **+0.0520 tok/s / +0.260824%** gain over the qualified A050 served mean.
  Mean prompt throughput was **107.1000 tok/s**, mean TTFT was
  **58.235501 s**, and mean end-to-end latency was **64.589099 s**.
- Every request completed exact `6237+128=6365`, `finish_reason=length`, 585
  reasoning characters, 33 fragments, empty visible content, and established
  output/reasoning SHA-256
  `e56e48a5587cc7b4d9981bc58ff1bdb227266ba83c2062fea8737d356f0955e5`.
  Server telemetry settled around 20.15--20.27 tok/s. Port 30000, matching
  workloads, and compiler processes were absent after shutdown; memory
  returned to ordinary residency and thermal/performance status was normal.
  The selected client mean now leaves **0.0114 tok/s** to the required floor.

### 2026-08-31 15:21 PDT - PERF-A053 reduced native RMS barriers

- Removed the initialization barrier from the native residual, recurrent q/k,
  recurrent output, and full-attention q/k reductions. The batch-one
  recurrent widths now use register-only single-SIMD reductions; wider shapes
  retain the shared partial reduction and both required barriers.
- Five exact short direct control/candidate pairs changed
  **21.195638542 -> 21.235273528 tok/s**, a **+0.186996%** gain with every
  pair positive. Four clean long-history pairs changed
  **20.211797451 -> 20.217209606 tok/s**; the movement is positive and below
  classification at that shape.
- Two process-isolated real-131K serving windows averaged **20.0200** and
  **20.0146 tok/s**. Their combined ten-request mean is **20.0173 tok/s**,
  clearing the required client floor while retaining exact counts, reasoning,
  stream shape, and digest. Signed commit `173cd4bc719229f0a3caf805f0cfed7d3e27eb51`
  owns the source and expanded full-attention parity case.

### 2026-08-31 16:46 PDT - PERF-A054 native sampled xhigh continuation lane

- Added an opt-in native Qwen sampler for the model contract at temperature
  1.0, top-k 20, and top-p 0.95. The graph keeps partitioning,
  full-vocabulary normalization, filtering, random Gumbel selection, and token
  selection on Metal. The default native path remains greedy when
  `SGLANG_MLX_NATIVE_SAMPLING` is absent.
- Added an opt-in reasoning safety bound and exact prompt-boundary state
  snapshot. A Codex tool continuation whose full incoming prompt grew from
  6,211 to 6,683 tokens now restores the prior prompt state and evaluates only
  the appended suffix. A synthetic `[100,200,300] ->
  [100,200,300,400]` trace independently reported `reuse_snapshot=1`.
- The unmodified Codex 0.151.0 harness, explicit 131,072 context metadata,
  `xhigh` effort, seed 42, and 256-token reasoning bound completed one
  `/bin/pwd` tool call and the final `QWEN38_TOOL_READY` response inside the
  120-second wrapper. The selected 64-partial run exited zero in about 81.8
  seconds at **12,894 input / 502 output / 332 reasoning-output tokens**.
- A selected 128-partial sampled client screen reached **19.941 tok/s**. The
  opt-in sampled lane reopens MLX's supported `MLX_SDPA_BLOCKS=64` setting
  under a different premise: stochastic xhigh behavior is authoritative and
  the exact default remains unchanged. Five sequential real-131K samples were
  **20.171, 20.170, 20.178, 20.166, and 20.176 tok/s**, mean
  **20.1722 tok/s**; every sample clears the requested floor. Prompt rates
  were **107.132, 107.210, 107.138, 107.028, and 107.100 tok/s**. TTFTs were
  **58.217857, 58.175599, 58.214610, 58.274212, and 58.235432 s**.
- The final rebuilt dylib SHA-256 is
  `550702cd5a66f134ef0f98d446f07ff6f7d78efeb24403f8934bf6b16f216d5b`.
  Its direct sampled `6237 / 32 warm / 256 timed` confirmation under the
  selected 64-partial launch reached **20.326297821 tok/s**, digest
  `bd6b79adfcbc3125`, last token `28322`. The three focused C++ parity
  binaries pass, the native pytest suite passes **8 tests** with 16 existing
  warnings, and `git diff --check` passes.

### 2026-08-31 17:43 PDT - PERF-A055 request-boundary sampling ownership

- The independent signed-`883da94` restart produced sampled decode results
  **20.163, 20.165, 20.123, 20.073, and 20.154 tok/s**, mean **20.1356**.
  Every request completed exact `6237+128=6365` with
  `finish_reason=length`. Running the xhigh shell-tool gate after this window
  changed its trajectory because MLX's process-global random stream had been
  advanced by the five unrelated benchmark requests.
- The engine now stores the configured native seed and reapplies it from the
  full `Engine::reset()` owner. Exact terminal-prefix and prompt-snapshot
  continuations enter through `begin_request()` and preserve the active stream;
  unrelated prompts perform the full reset and receive the configured stream
  origin. This makes request outcome independent of earlier unrelated traffic.
- Five sequential real-131K sampled requests after the change measured
  **20.155, 20.148, 20.157, 20.152, and 20.166 tok/s**, mean **20.1556**.
  Mean prompt throughput was **107.12 tok/s**, mean TTFT **58.224555 s**, and
  mean end-to-end latency **64.525519 s**. Every request completed 6,365
  tokens with `finish_reason=length` and identical output/reasoning SHA-256
  `91dbc7056abfdc989aaee9e1d0f1fa3abc410637aabe144ac2ab0ea9bd97df3e`.
- Fixed-seed replay also made the remaining behavior issue reproducible: the
  low-bit model completes the requested `/bin/pwd`, then can construct an
  alphanumeric `write_stdin.session_id`. Greedy-after-reasoning and one-call
  turn truncation were screened and removed; PERF-FA089 and PERF-FA090 preserve
  those results.
- The installed-MLX native rebuild retains the established macOS 26.0/26.2
  link warning. Q/k normalization plus full-attention RoPE, recurrent
  norm/gate, and residual RMSNorm C++ parity pass. The focused native engine
  suite passes **8 tests** with 16 existing warnings, and `git diff --check`
  passes. This change is selected as a request-semantics/reproducibility win;
  xhigh structured-output qualification continues independently.

### 2026-08-31 23:58 PDT - PERF-A056 recurrent precision boundary and PERF-A057 DFlash2 track

- The selected early-out27-v2 checkpoint plus every recurrent output
  projection from immutable Q4 retained five-sample served throughput at
  **20.111, 20.134, 20.115, 20.127, and 20.117 tok/s**, mean **20.1208**,
  while the exact xhigh shell gate decoded until the 120-second timeout without
  emitting a usable tool call.
- Restoring every quantized recurrent-attention projection from Q4 screened at
  **20.107157009 tok/s** in the direct sampled long-history harness. Its first
  real `6237+128` server sample reached **19.913 tok/s**, completed exact
  6,365 tokens, and fell **0.087 tok/s** below the hard floor. The pinned
  Codex xhigh gate passed cleanly with one `/bin/pwd`, the expected working
  directory, final `QWEN38_TOOL_READY`, and exit zero. This establishes a
  narrow precision/performance boundary suitable for projection-family
  bisection.
- A `qkvz`-only Q4 restore reached **21.056891733 tok/s** in the direct short
  harness. The loader remains opt-in and fails closed on missing or surplus
  projection tensors. Real-server speed and behavior qualification remain
  active.
- A current upstream artifact check found the Qwen3.8-specific DFlash2 draft,
  including a 1.14 GiB Q4 checkpoint. The existing SGLang v2 worker and model
  card define a concrete native MLX integration target. Exact lossless
  sampling verification, recurrent accepted-path state commit, 131K
  residency, and the established xhigh shell gate govern admission.

### 2026-09-01 00:26 PDT - PERF-A056 QKV precision crosses the served floor

- QKV+Z restoration completed an exact sampled `6237+128` request at
  **19.948 tok/s** generation, **107.113 tok/s** prompt, **58.228452 s**
  TTFT, and **64.595021 s** end to end. The pinned xhigh gate then passed
  cleanly with one `/bin/pwd`, the expected working directory, exact final
  marker, and exit zero. This arm remains **0.052 tok/s** below the hard
  served floor.
- QKV-only restoration produced a first five-sample window of **20.025,
  20.024, 20.023, 20.023, and 20.016 tok/s**, mean **20.0222**, then an
  independent restart window of **20.029, 20.024, 20.029, 20.034, and 20.030
  tok/s**, mean **20.0292**. All ten samples completed exact total 6,365 with
  `finish_reason=length`; each window was internally digest-stable and every
  sample cleared 20.
- The first QKV xhigh turn passed cleanly with one requested shell tool and
  exit zero. The post-window independent turn issued the valid `/bin/pwd`,
  then a malformed `write_stdin` with a string session id, consumed the
  router error, and recovered to final `QWEN38_TOOL_READY`. Throughput is now
  qualified above the floor across restarts; structured-output reliability
  remains active.

### 2026-09-01 00:43 PDT - PERF-A058 native DFlash2 draft artifact

- Added a standalone C++20 MLX converter for the exact
  `incoai/Qwen3.8-27B-DFlash2` BF16 checkpoint. It validates the 81-tensor
  source contract, converts all 47 eligible two-dimensional linear weights
  to affine W4/G64, preserves the selector codebooks, normalization weights,
  and convolution bases in BF16, and saves through an atomic temporary file.
- The converter's in-memory self-test validates the exact 81 -> 175 tensor
  expansion and bounds round-trip affine dequantization error. A strict
  `-Wall -Wextra -Werror` C++ build passed, with only the established external
  macOS 26.0 / MLX 26.2 link warning; `clang-format --dry-run --Werror` and
  `git diff --check` passed.
- The immutable 3.6 GiB source has SHA-256
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
  Full conversion produced the distinct 1.2 GiB artifact
  `Qwen3.8-27B-DFlash2-MLX-AffineQ4/model.safetensors`. A second complete
  conversion verified all 175 names, shapes, dtypes, and provenance fields
  and reproduced the file byte-for-byte, with SHA-256
  `33bf2ddd0d46c27d6f383b6822ab897eb87261d34ca9c6eeb6a0f232026f723c`.
  Its copied upstream config has SHA-256
  `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`.
- This is a validated draft-residency prerequisite. Throughput qualification
  follows runtime integration. Native target hidden capture, DFlash block
  execution, exact p/q verification, and accepted-path recurrent-state commit
  remain the active integration gates.

### 2026-09-01 02:03 PDT - PERF-A059 native DFlash2 runtime and bounded prefill

- Added exact loading for the 175-tensor affine-W4 DFlash2 artifact through
  the existing C++ `Engine::load_mtp` entry. The native path captures target
  hidden states after layers 5, 19, 33, 47, and 61; executes the five draft
  layers over one anchor plus seven mask tokens; applies the rank-256,
  top-16 selector; verifies through the target; and uses exact lossless p/q
  rejection with residual target sampling.
- A recurrent commit tape retains convolution inputs, normalized keys, FP32
  decay, and FP32 innovation deltas from the verified target pass. Replaying
  accepted prefixes one through seven from the saved state is bit-exact with
  a direct prefix forward at the production 16-key-head, 48-value-head,
  128-dimension shape. Partial commit time falls from **83--191 ms** for
  restore/re-forward to about **2--6 ms** in current full-model cycles.
- The opt-in small-batch affine Metal product reduces matched steady full-Q4
  draft time from **41.288--41.686 ms** to **34.211--37.378 ms** and target
  verification from **352.531--356.089 ms** to **305.130--305.604 ms**. Its
  W2/W4 parity test passes rows 2, 7, and 8, a 5,120-element K loop, a
  6,144-output grid, and fail-closed unsupported parameters. A broader 5,120-
  output dispatch and the exact `17408 -> 5120` arm were removed after matched
  full-model regressions.
- The native prefill owner now processes DFlash target/capture work in
  internal 2,048-token units and synchronizes each unit. Direct 4,096- and
  6,237-token prompts complete. This retains one external SGLang native
  prefill call, which covers the model runner path that cannot continue an
  externally chunked native request.
- One full-Q4, real-131K server completed the exact pinned Codex 0.151.0
  `xhigh` request. Thread `01a05c29-42b4-7ee3-ab20-a4454e592c06` issued
  exactly `/bin/zsh -c '/usr/bin/printf QWEN38_DFLASH2_TOOL=passed'`, observed
  the exact stdout and exit zero, then returned exact
  `QWEN38_DFLASH2_READY`. The first 6,228-token prefill ran at **90.75 prompt
  tok/s**. Generation telemetry settled around **7.6--11.85 tok/s** and the
  tool-result continuation reached **15.20 tok/s** during high acceptance.
- Current exact direct sampled `128 / 32 warm / 128 timed` output is
  **9.300145465 tok/s**, mean emitted width **3.175**, digest
  `41bf022415fb9842`, and last token 21. Steady refill cycles are about
  **343--346 ms**. The behavior and first-request residency gates pass; the
  hard 20 tok/s generation gate remains active.
- A four-token draft block reduces steady draft to **18.735--19.078 ms** and
  verify to **170.274--170.941 ms**, or about **191.727--192.747 ms** total.
  Its random-token screen emits only **1.142857** tokens per cycle, and even
  perfect four-token emission provides about 20.85 tok/s before server
  overhead. The fixed block-four candidate is rejected; adaptive depth stays
  available only with measured acceptance evidence.
- Strict C++20 builds, both new native parity executables, the existing native
  engine suite (**8 passed**, 16 existing warnings), and `git diff --check`
  pass. The first focused-suite invocation lacked the explicit installed-MLX
  prefix and failed during native compilation; the rerun with
  `MLX_PREFIX=.venv/lib/python3.11/site-packages/mlx` passed.
- Commit: signed `d57a6ac11c8317943a94a7092e8242f2bb730aed`
  (`feat(mps): add native DFlash2 runtime`).

### 2026-09-01 02:17 PDT - PERF-A060 M8 K-split affine verification

- Change: added an opt-in affine-W4/G64 Metal product for exactly eight BF16
  rows. Each of eight SIMD groups owns one contiguous eighth of K, stages one
  32x16 dequantized tile, accumulates two 8x8 FP32 matrices, and reduces the
  eight partial 8x16 outputs once. The existing selected product remains the
  fallback. Enable with both `SGLANG_MLX_NATIVE_SMALL_BATCH_QMM=1` and
  `SGLANG_MLX_NATIVE_M8_KSPLIT_QMM=1`.
- Isolated draft-artifact measurements, reported as stock MLX / prior custom /
  M8 K-split milliseconds, were: gate/up `5120x17408`
  **1.733475 / 1.198400 / 0.978217**; down `17408x5120`
  **1.707825 / 1.550488 / 1.026783**; q `5120x4096`
  **0.939500 / 1.079846 / 0.626283**; o `4096x5120`
  **1.197017 / 1.100367 / 0.649871**; dynamic convolution
  `5120x1280` **0.607971 / 1.259121 / 0.378050**; capture FC
  `25600x5120` **2.027963 / 2.332121 / 1.407288**; selector
  `5120x256` **0.313808 / 0.771321 / 0.297050**. Full-target gate/up measured
  **1.706962 / 1.185346 / 1.039687** and target down measured
  **1.782333 / 1.643600 / 1.019617**.
- An adjacent full-Q4 control from the candidate library reproduced
  **9.304875518 tok/s**, mean emitted width **3.175**, digest
  `41bf022415fb9842`, and last token 21. Two process-isolated K-split samples
  measured **11.241518417** and **11.243767395 tok/s**, mean
  **11.242642906**, a **1.937767388 tok/s / 20.825291%** gain. Both candidate
  samples reproduced mean emitted width **2.909090909**, digest
  `13442dfecdbbbcda`, and last token 4973.
- Steady candidate draft time is generally **28.5--33.5 ms**, target verify
  **225.5--226.7 ms**, and total **257--263 ms**. The adjacent control remains
  near **34--37 / 305--306 / 342--346 ms**. The arithmetic schedule changes
  the sampled trajectory and lowers this synthetic sample's acceptance, while
  the execution-cost reduction still produces the repeatable throughput win.
- Correctness: affine parity passes W2/W4 rows 2, 7, and 8; M8-specific
  K/N `256/256`, long-K `5120/64`, and wide-N `256/6144`; and fail-closed
  group-size rejection. Maximum observed M8 difference from MLX is 0.0625 in
  BF16 output. Strict `-Wall -Wextra -Werror` engine/test builds, the focused
  native engine suite (**8 passed**, 16 existing warnings), test formatting,
  and `git diff --check` pass.
- Decision: retain as opt-in signed commit
  `b853514b5c8e18fe073a30d54103b2149e70257d`. The real 131K-configured Codex
  xhigh behavior and served-throughput gate is next.

### 2026-09-01 02:28 PDT - PERF-A061 M8 real Codex qualification

- Launched the committed full-Q4 target plus affine-W4 DFlash2 draft with real
  131,072 context and token pools, one request, five auxiliary slots, internal
  2,048-token native DFlash prefill units, and both selected affine-QMM
  switches. The server resolved the Qwen3 reasoning and Qwen3 Coder tool
  parsers, incremental output, language-only mode, radix disabled, and native
  sampling seed 42. `/health`, `/v1/models`, and `/model_info` passed; the
  model list reported 131,072 maximum length and image/audio understanding
  remained disabled.
- The bounded Codex 0.151.0 request used the isolated pinned configuration,
  explicit `model_context_window=131072`, compaction threshold 117,964, and
  `model_reasoning_effort="xhigh"`. Thread
  `01a05c45-6b01-7a21-8a69-065170fd3402` called exactly
  `/bin/zsh -c '/usr/bin/printf QWEN38_DFLASH2_TOOL=passed'`, observed exact
  stdout and exit zero, then returned exact `QWEN38_DFLASH2_READY`. Codex
  exited zero with 12,822 input, 349 output, and 296 reasoning-output tokens.
- The first request prefetched 6,228 tokens at **95.05 prompt tok/s**. After
  the initial decode interval, server telemetry reported **10.54, 12.46,
  13.66, 13.72, 14.16, and 17.20 tok/s**. Natural accepted counts spanned
  zero through seven. The 6,594-token tool-result continuation prefetched at
  **80.48 prompt tok/s** and completed the final marker.
- Live first-request cycles generally spent **30.7--35.9 ms** drafting and
  **251.2--253.5 ms** verifying, with sampling about **0.6--0.8 ms** and
  ordinary accepted-prefix commits about **2.5--4.9 ms**. This preserves the
  direct M8 execution win while identifying roughly 25--27 ms of additional
  target verification cost at a 6.2K full-attention history.
- Post-turn `/health` passed. The foreground server exited cleanly on
  `Ctrl+C`; port 30000 and matching server/compiler processes are absent,
  memory is 94% free, and macOS reports normal thermal/performance status.
  The verifier remains the next measured optimization owner, and the hard
  real served generation floor remains 20 tok/s.

### 2026-09-01 02:39 PDT - PERF-A062 sixteen-way M8 K split

- Expanded the committed affine-W4 M=8 kernel from eight to sixteen
  32-thread SIMD groups. Each group now owns one sixteenth of K; the 512-thread
  group retains the same private 32x16 BF16 weight staging, FP32 accumulation,
  and final reduction. Dispatch and the public helper now require K divisible
  by 512, which guarantees every K partition contains an integral number of
  32-element tiles. Every reachable Qwen3.8 target and DFlash projection using
  this path satisfies the narrower contract.
- Exact checkpoint microbenchmarks measured gate/up `K=5120,N=17408` at
  **0.975192 ms** and down `K=17408,N=5120` at **0.972296 ms**. The SG8 values
  were **0.978217** and **1.026783 ms**, respectively. The down projection
  improves about 5.3% while gate/up remains within 0.4% of the prior sample.
- One traced whole-model sample measured steady draft **27.4--27.9 ms** and
  verify **210.6--212.1 ms**, down from SG8's **28.5--33.5 / 225.5--226.7
  ms**. A separate 31.56 ms draft and 216.35 ms verify outlier were observed.
- Five process-isolated no-trace samples measured **17.635926855,
  17.647986134, 17.645350442, 17.654348952, and 17.674891803 tok/s**, mean
  **17.651700837**. The three available SG8 samples measure **11.241518417,
  11.243767395, and 11.242024298**, mean **11.242436703**. SG16 gains
  **6.409264134 tok/s / 57.009564%**. All candidate samples reproduce 30
  refills, mean emitted width **4.3**, digest `3e40b8569af9555f`, and last token
  735. The traced sample was excluded from this no-trace window.
- Permanent parity covers M8 K/N `512/256`, `5120/64`, and `512/6144`, with
  maximum absolute BF16 error **0.0625**. A valid affine K=256 case now proves
  the narrower M8 shape contract fails closed. The strict
  `-Wall -Wextra -Werror` library and test builds pass, the standalone parity
  executable passes, the focused native engine suite passes **8 tests** with
  16 existing warnings, test formatting passes, and `git diff --check` passes.
- Decision: retain as signed commit
  `1e21aece563191fc2413c769aaeea644a7e49b68`. Its EDDSA signature verifies as
  good. The SG16 real 131K-pool served and Codex gates follow.

### 2026-09-01 02:46 PDT - PERF-A063 SG16 sampled serving and Codex gate

- Launched the committed full-Q4 target plus affine-W4 DFlash2 draft with
  SG16 enabled, real 131,072 context and token pools, one request, five
  auxiliary slots, native sampling seed 42, both Qwen parsers, and the same
  bounded prefill/streaming configuration used by PERF-A061. Readiness,
  model-list, and model-info checks passed; maximum model length remained
  131,072 and image/audio understanding remained disabled.
- The exact sampled `6237+128` control completed at **15.336 generation tok/s**
  and **109.087 observed prompt tok/s**, with **57.174631 s TTFT** and
  **65.455554 s** end to end. It completed exact 6,365 total tokens with
  `finish_reason=length`; output SHA-256 was
  `87938038e62e3f8911acfac664f6d17336c159abdf3fa716efbc5d8db01766fb`.
- At 6.2K history, SG16 cycles generally spent **29.5--31.8 ms** drafting and
  **236.5--238.9 ms** verifying. This cuts about 14--16 ms from SG8's
  **251--253 ms** live verifier while retaining the same M=8 block contract.
- Codex thread `01a05c5a-28a6-79f0-a526-efa19d961645` issued exactly
  `/bin/zsh -c '/usr/bin/printf QWEN38_DFLASH2_TOOL=passed'`, observed exact
  stdout and exit zero, and returned exact `QWEN38_DFLASH2_READY`. Codex exited
  zero with 12,820 input, 347 output, and 294 reasoning-output tokens. The
  first request prefetched 6,228 tokens at **81.49 tok/s** and the 6,592-token
  continuation prefetched at **82.45 tok/s**. Reported decode intervals
  included **12.22, 13.30, 13.33, 13.54, 16.31, and 20.85 tok/s**.
- Post-turn health passed. The server exited through its foreground `Ctrl+C`
  handler; port 30000 and matching server/compiler processes are absent,
  memory is 94% free, and macOS thermal/performance status is normal. SG16
  passes behavior and improves fixed verifier cost. The exact sampled control
  leaves **4.664 tok/s** to the sustained served floor.

### 2026-09-01 02:54 PDT - PERF-A064 SG16/B32 verifier

- Doubled the M8 affine-W4 output tile from 16 to 32 columns while retaining
  sixteen 32-thread K partitions. Each SIMD group dequantizes a private 32x32
  BF16 tile and accumulates four FP32 8x8 outputs. The weight-staging and
  cross-group-reduction phases are separated by a threadgroup barrier and
  reuse one 32 KiB allocation, keeping the wider tile inside the measured
  Apple threadgroup-memory limit.
- Dispatch and direct-helper contracts now require N divisible by 32 in
  addition to the selected K-divisible-by-512 condition. Every reachable
  Qwen3.8 target and DFlash projection satisfies both. A valid affine
  `K=512,N=16` case proves the new N contract fails closed.
- Exact checkpoint gate/up `K=5120,N=17408` improves from SG16/B16
  **0.975192 ms** to **0.829204 ms**. Down `K=17408,N=5120` improves from
  **0.972296** to **0.855483 ms**. Maximum real-tensor errors from stock MLX
  remain **0.03125** and **0.125**, respectively.
- The traced whole-model screen measured steady draft **26.0--26.4 ms** and
  verify **185.4--186.7 ms**, versus SG16/B16's **27.4--27.9 /
  210.6--212.1 ms**. One 191.16 ms verify and several 30.1--30.3 ms draft
  outliers occurred.
- Five process-isolated no-trace samples measured **31.347246490,
  31.354396960, 31.268350560, 31.330813634, and 31.335863464 tok/s**, mean
  **31.327334222**. Every sample reproduced 19 refills, mean emitted width
  **6.684210526**, digest `46bd4bb035b72c2b`, and last token 20. The gain over
  the SG16/B16 five-sample mean is **13.675633384 tok/s / 77.474876%**. A
  separate traced sample reached 30.500698761 tok/s and was excluded from the
  no-trace window.
- Permanent M8 parity still passes K/N `512/256`, `5120/64`, and `512/6144`,
  with maximum absolute BF16 error **0.0625**. Valid K=256 and N=16 contracts
  fail closed. Strict library/test builds, the standalone parity executable,
  focused native engine suite (**8 passed**, 16 existing warnings), test
  formatting, and `git diff --check` pass.
- Decision: retain as signed commit
  `3bae8a5e67acf6b71f5f1b86498a5f3ee82146e6`. Its EDDSA signature verifies as
  good. The real sampled serving and Codex gates follow.

### 2026-09-01 03:03 PDT - PERF-A065 SG16/B32 real-client gate

- Launched signed HEAD `96bc05a6a2` with the exact 131,072 context/token
  pools, one running request, affine-W4 DFlash2 draft, SG16/B32 M8 kernel,
  native sampling seed 42, model sampling defaults, reasoning parser, and
  Qwen3 Coder tool parser. Resolved arguments and `/v1/models` reported
  131,072 tokens; `/model_info` reported image and audio understanding
  disabled.
- The exact sampled `6237+128` request completed 6,365 total tokens with
  `finish_reason=length`, **15.328 generation tok/s**, **109.106 prompt
  tok/s**, **57.164599 s TTFT**, and **65.450058 s** end to end. The reasoning
  SHA-256 is
  `51bb9afea1e151fccdd1a6b2fd39ef1be4c9062e7d6065b247cb1a7993161d66`.
- Live 6.2K-history cycles generally measured **28--32 ms** draft and
  **211--214 ms** verify. This confirms the wider verifier reaches serving;
  this prompt's accepted-width trajectory leaves **4.672 tok/s** to the hard
  sustained floor.
- Codex thread `01a05c69-215f-7fb0-a7f8-1425c9b2ae5a` executed exactly one
  `/usr/bin/printf QWEN38_DFLASH2_TOOL=passed`, observed exact stdout and exit
  zero, then returned exact `QWEN38_DFLASH2_READY`. Codex exited zero with
  12,769 input, 317 output, and 264 reasoning-output tokens.
- Post-turn health passed. PID 10684 and its three verified children exited
  through foreground `Ctrl+C`; port 30000 and matching server/benchmark
  processes are clear, memory returned to 94% free, and thermal/performance
  status is normal. Continue with proposal-acceptance and fixed-cycle work.

### 2026-09-01 03:10 PDT - PERF-A066 official dense BF16 DFlash2

- Generalized the native `QLinear` execution owner to dispatch exact BF16
  matrices through MLX matmul with validated dtype and feature dimensions.
  The DFlash loader now distinguishes the official 81-tensor dense contract
  from the derived 175-tensor affine contract and validates every matrix at
  its exact `[output,input]` shape.
- The official 3.6-GiB checkpoint loaded and completed the direct sampled
  harness. One traced run reached **9.005226 tok/s** with steady draft
  **41.5--42.1 ms**, verify generally **185--187 ms**, 61 refills, width
  **2.114754**, digest `408f99f917ffffcc`, and last token 96968. A no-trace
  repeat reached **9.044043 tok/s** with the same refill/width/digest result.
- An adjacent affine-W4 run from the same binary reached **30.992508 tok/s**,
  19 refills, width **6.684211**, digest `46bd4bb035b72c2b`, and last token 20.
  Dense BF16 is **21.948464 tok/s / 70.818614%** slower in this matched screen.
- Dense bit parity passed. Strict warning-as-error library and standalone-test
  builds passed after classifying the external MLX headers as system headers;
  the initial `-I` command correctly surfaced four warnings inside MLX under
  `-Werror`. The combined affine/dense executable passed, the focused native
  suite passed **8 tests** with 16 existing warnings, test formatting passed,
  and `git diff --check` passed.
- Retained official-checkpoint compatibility in signed commit
  `6cf95442ccde5a2df2696553ad33ad217bff60da`
  (`feat(mps): load native dense DFlash2 drafts`). Its EDDSA signature
  verifies as good. PERF-FA096 closes dense BF16 as the unchanged production
  selection; affine-W4 remains active while proposal policy is investigated.

### 2026-09-01 03:17 PDT - PERF-A067 greedy proposal policy

- Added a temporary native switch matching the official DFlash worker's LM
  head argmax proposal. Its exact verifier represented the one-token proposal
  support with probability one and used the existing rejection/residual path.
- Five process-isolated direct samples reached **37.537356, 37.505550,
  37.500870, 37.532174, and 37.517706 tok/s**, mean **37.518731**. Every
  sample used 16 refills, width **7.9375**, digest `1c33d03ba961ff25`, and last
  token 198. A traced screen reached **37.310474 tok/s** with steady draft
  **24.5--24.8 ms** and verify **185--187 ms**.
- The exact real sampled `6237+128` gate completed at **9.512 tok/s**,
  **109.642 prompt tok/s**, **56.885039 s TTFT**, and **70.236581 s** end to
  end. It produced exact 6,365 total tokens and `finish_reason=length`, with
  reasoning SHA-256
  `62bc27d075d3d68fd4eb9fbbf8d4db312390c505bd36ddfe578086720c2b656e`.
  Natural accepted counts were mostly zero through three despite occasional
  full blocks. The learned selector's matched real result is **15.328 tok/s**.
- Decision: reject. The repeated-token direct prompt overstates greedy
  acceptance and cannot admit proposal-policy changes. Removed the switch and
  generic one-token-support scaffolding with `apply_patch`; the source
  worktree returned exactly to signed HEAD. PERF-FA097 records the result.

### 2026-09-01 03:27 PDT - PERF-A068 pinned DSpark v2 artifact

- Downloaded the official `RadixArk/Qwen3.8-27B-DSpark` revision
  `b9a5dbdf03bc999c6c73c426b19c2d9041cea393` into the distinct immutable
  directory
  `/Users/dcazares/.cache/sglang/checkpoints/Qwen3.8-27B-DSpark-BF16`.
  The exact command used `hf download --revision ... --local-dir ...
  --max-workers 4`; the repository target and every prior draft artifact were
  untouched.
- `model.safetensors` is **3,714,723,322 bytes** with SHA-256
  `2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7`.
  `config.json` has SHA-256
  `dd65fb1b01c2adea69512ff2990a79d58eb7fe2c7ea97375aa66f657a29a5bfd`.
- The safetensors header contains exactly 62 BF16 tensors: a 25,600-to-5,120
  context projection, five full-attention Qwen3 draft layers, final and input
  norms, rank-256 `markov_w1`/`markov_w2` tables over the 248,320-token
  vocabulary, and the 5,376-input confidence projection. Configuration fixes
  gamma at seven, target verification width at eight, target captures at
  layers 5/19/33/47/61, full YaRN RoPE through 262,144 positions, and static
  verification remains available without consulting the confidence head.
- Rebuilt the ignored repository native dylib from signed source after the
  removed greedy experiment. The strict warning-as-error build passed with
  only the established macOS 26.0 / MLX 26.2 linker warning. Port 30000 and
  matching server/compiler processes were clear before acquisition; the
  volume had 161 GiB available.
- Next: create a provenance-bearing affine-W4 derivative, then implement the
  native DSpark backbone/Markov proposal at the shared target-verifier owner.

### 2026-09-01 03:31 PDT - PERF-A068 affine-W4 DSpark derivative

- Added the standalone C++20 converter
  `benchmark/mac/convert_dspark_mlx.cpp`. It fails closed unless the source is
  the exact 62-tensor BF16 Qwen3.8 DSpark contract, including every tensor's
  shape and dtype.
- The converter applies MLX affine W4/group-64 quantization to the context
  projection, all 35 transformer matrices, and Markov output projection. It
  retains the Markov embedding table for direct indexed lookup and retains the
  confidence tensors for exact static/dynamic checkpoint compatibility.
- Strict warning-as-error compilation and `clang-format --dry-run --Werror`
  passed. The synthetic selection/retention and dequantization self-test
  passed with the established macOS 26.0 / MLX 26.2 linker warning.
- Converted the immutable BF16 source into
  `/Users/dcazares/.cache/sglang/checkpoints/Qwen3.8-27B-DSpark-MLX-AffineQ4/model.safetensors`.
  Reload verification passed all 136 output names/shapes/dtypes and embedded
  format, quantization, matrix-count, source-revision, and source-SHA metadata.
  The final artifact is **1,136,258,326 bytes** with SHA-256
  `a69938fff75fecb662da3f8c5acfb6e2d4e147f776d6e6709966455d54584033`.
- Next: add exact native loading and DSpark full-attention/Markov execution,
  reusing the already-qualified target verification and recurrent commit
  owners.

### 2026-09-01 03:44 PDT - PERF-A068 native DSpark execution

- Added exact native loading for the official 62-tensor BF16 and derived
  136-tensor affine-W4 DSpark contracts. The five full-attention draft layers
  use their shared target-hidden projection, full-context per-layer K/V,
  Qwen3 RMS normalization, and the checkpoint's YaRN parameters. Seven
  semi-autoregressive rows use the shared target embedding/head and sequential
  rank-256 vanilla Markov correction.
- Generalized the existing exact target verifier and accepted-prefix commit
  owner across DFlash2 sparse proposals and DSpark dense proposals. The
  established DFlash sampled regression reproduced **31.283141 tok/s**, 19
  refills, mean width **6.684211**, digest `46bd4bb035b72c2b`, and last token
  20, matching the selected trajectory exactly.
- The first matched sampled `128 / 1 warm / 32 timed` DSpark screens reached
  **10.050625 tok/s**, width **2.428571**, with affine-W4 and **5.746863
  tok/s**, width **2.285714**, with BF16. Steady target verification remained
  about **186 ms**; affine-W4 draft work was generally **36.55--40.07 ms**
  and BF16 was **44.54--44.75 ms**. Different generated trajectories make
  this a functional admission screen. The affine-W4 greedy control reached
  **35.825160 tok/s**, width **8.0**, after warmup.
- The independent C++ YaRN reference passes at offsets 0, 9,000, and 131,071;
  maximum absolute errors are `5.96046e-08`, `0.000168275`, and `0.0015974`.
  Affine QMM, recurrent commit lengths 1--7, q/k normalization and RoPE,
  recurrent norm/gate, residual RMSNorm, and causal-convolution parity all
  pass. Strict warning-as-error library/tests, test formatting, the focused
  native suite (**8 passed**, 16 existing warnings), and `git diff --check`
  pass.
- Decision: retain native execution as the DSpark optimization base. The
  representative sampled serving and exact 131K capacity gates remain open;
  proposal-distribution fidelity is the immediate performance owner.

### 2026-09-01 03:48 PDT - PERF-A068 representative sampled baseline

- Launched signed `21cd561dfc` with the real 131,072 context/token pools,
  one running request, five auxiliary slots, native sampling seed 42, the
  Qwen3 reasoning/parser surface, and the affine-W4 DSpark artifact. Resolved
  arguments, `/health`, `/v1/models`, and `/model_info` passed; image and
  audio understanding remained disabled.
- The exact sampled `6237+128` request completed all 6,365 tokens with
  `finish_reason=length`, **11.242 generation tok/s**, **109.106 observed
  prompt tok/s**, **57.164421 s TTFT**, and **68.461186 s** end to end.
  Reasoning SHA-256 was
  `555ba1da1dc6fc0f7969f2a67261a05b33c41136d8fdbfa607e8653a05a88fe2`.
- Live steady cycles generally used **41.77--45.45 ms** for DSpark draft,
  **211.97--213.05 ms** for target verification, about **0.58--0.77 ms**
  for exact sampling, and **1.89--4.43 ms** for ordinary commit. Natural
  acceptance ranged from zero through seven.
- Decision: use **11.242 tok/s** as the representative DSpark baseline. It is
  **4.086 tok/s / 26.657%** below the selected DFlash sample and **8.758
  tok/s** below the hard floor. Draft/proposal optimization proceeds before
  the exact 131K capacity and Codex gates.

### 2026-09-01 03:50 PDT - PERF-A069 aligned DSpark sampling filter

- Added a temporary C++ switch that sampled each Markov-corrected DSpark row
  from the same top-k 20/top-p 0.95 filtering mechanism used by the target and
  supplied that exact dense q to rejection sampling.
- The direct baseline was **10.050625 tok/s**, 14 refills, width **2.428571**.
  The candidate reached only **6.997461 tok/s**, 20 refills, width **1.6**;
  steady draft also rose by roughly 2 ms from seven extra vocabulary
  partition/sort operations.
- Decision: reject and remove. Independently truncating draft and target
  supports discards useful overlap because their top-20 rankings differ.
  PERF-FA098 records the closed route.

### 2026-09-01 03:58 PDT - PERF-A070 BF16 DSpark Markov output projection

- Derived a distinct 134-tensor checkpoint that kept only
  `markov_head.markov_w2` in source BF16 while quantizing the other 36
  eligible draft matrices to affine W4/G64. The artifact was
  **1,227,639,900 bytes**, **87.148 MiB** above the selected 136-tensor
  all-affine artifact, with SHA-256
  `73b829d7845a72ac34794e9dd74bd96eae2189a5bcd7b45c2099a2b45638674f`.
- The fixed-seed direct `128 / 1 warm / 32 timed` screen fell from
  **10.050624654 to 7.044992356 tok/s** (**-29.904930%**), from 14 to 20
  refills, and from width **2.428571429 to 1.65**. Candidate digest was
  `a9eed7de4b198270`, last token 9. Steady draft remained about **37--41 ms**
  and target verify remained about **186 ms**.
- The governing real-131K-pool sampled `6237+128` request completed exact
  6,365 tokens at **11.294 tok/s**, **109.107 prompt tok/s**, **57.164173 s
  TTFT**, and **68.409090 s** end to end. Its reasoning/output SHA-256 was
  `47690f3aaf04561fa6abe2cd3205724c59204b43a4f3eb4a8e1c525d584da3b1`.
  The **+0.052 tok/s / +0.462551%** movement from 11.242 occurs on a different
  sampled trajectory and leaves the candidate 8.706 tok/s below the floor.
- Decision: reject and remove. The direct acceptance regression, added
  residency, and sub-percent trajectory-level served movement do not support
  promotion. The derived artifact was deleted; immutable source and selected
  affine artifacts remain intact. PERF-FA099 records the closed route.

### 2026-09-01 04:11 PDT - PERF-A071 native DSpark confidence telemetry

- Retained the official BF16 confidence projection and bias in the native
  DSpark owner. The checked helper concatenates final draft hidden state with
  the previous-token rank-256 Markov embedding, applies the checkpoint's
  5,376-to-1 projection, and converts its raw score through an FP32 sigmoid,
  matching upstream survival semantics.
- Confidence executes only under the existing speculative trace flag. It is
  materialized in the same draft evaluation barrier as proposal tokens and q,
  preserving `draft_ms` meaning and keeping the ordinary untraced path free of
  projection work. Trace output reports all seven per-position probabilities
  immediately before the corresponding accepted-width record.
- Final exact direct `128 / 1 warm / 32 timed` evidence reaches
  **10.071235235 tok/s**, 14 refills, mean width **2.428571429**, digest
  `5a38c7070d7badeb`, and last token 16. The PERF-A068 control was
  **10.050624654 tok/s** with the same trajectory. Steady draft stages remain
  **36.58--36.88 ms** and verify remains about **186 ms**; the small throughput
  movement is diagnostic noise.
- The exact natural sampled `6237+128` request with real 131K pools completed
  all 6,365 tokens at **11.303 tok/s**, **109.711 prompt tok/s**, **56.849497 s
  TTFT**, and **68.085129 s** end to end. It reproduced the all-affine baseline
  reasoning/output SHA-256
  `555ba1da1dc6fc0f7969f2a67261a05b33c41136d8fdbfa607e8653a05a88fe2`.
- Natural-prompt confidence ranged from about **0.1895 to 0.9998**. Several
  near-one blocks accepted all seven tokens, while exact stochastic p/q also
  rejected some high-confidence blocks. Scheduling must maximize expected
  cumulative survival against measured per-width cost; a deterministic cutoff
  would misclassify observed blocks.
- Strict warning-as-error library and standalone-test builds pass. The C++
  test covers exact confidence values, FP32 output shape/dtype, and invalid
  shape rejection alongside YaRN parity. Focused native pytest passes **8
  tests** with 16 existing warnings. The real server passed health and
  language-only gates, then its verified process tree exited cleanly.
- Decision: retain as the measured input to native dynamic scheduling. Next,
  generalize the shared verifier to bounded widths and profile its M1 Max cost
  curve before selecting a survival budget.

### 2026-09-01 04:18 PDT - PERF-A072 bounded DSpark verification

- Generalized the common native exact verifier from a fixed seven-token draft
  contract to checked prefixes of one through seven tokens. Target input,
  p/q slicing, rejection and residual sampling, accepted-prefix recurrent
  commit, output buffering, and trace telemetry now derive from the checked
  prefix length. DFlash continues to supply seven drafts. DSpark reads an
  opt-in `SGLANG_MLX_NATIVE_DSPARK_VERIFY_DRAFT_TOKENS`; an absent value keeps
  seven, and malformed or out-of-range values fail closed.
- The DSpark proposal remains the same seven-position semi-autoregressive
  graph, with only a prefix passed into target verification. This isolates the
  target cost curve needed by confidence scheduling. The exact sampled
  `128 / 1 warm / 32 timed` results for draft counts one through seven were
  **11.920230932 / 8.861865636 / 8.353200099 / 7.544333347 / 4.104265970 /
  4.139508352 / 10.025000236 tok/s**. Mean emitted widths were respectively
  **1.6 / 1.523809524 / 1.777777778 / 1.941176471 / 1.571428571 /
  1.571428571 / 2.428571429**.
- Target verification cost rises from about **92 ms** at M=2 through **210
  ms** at M=5. M=6 and M=7 take roughly **327--332 ms** through the generic
  affine small-batch route. M=8 returns to about **186--188 ms** through the
  selected SG16/B32 K-split kernel. Draft cost remains roughly **37--40 ms**.
  This leaves M=2 and M=8 as the useful current target tiers.
- Default DSpark reproduced 14 refills, width **2.428571429**, digest
  `5a38c7070d7badeb`, and last token 16. The unchanged DFlash path reproduced
  **31.295166030 tok/s**, 19 refills, width **6.684210526**, digest
  `46bd4bb035b72c2b`, and last token 20 on its established
  `128 / 32 warm / 128 timed` workload.
- Decision: retain the checked bounded verifier as exact opt-in scheduling
  infrastructure. Close every fixed shortened width as a production policy;
  the best, M=2, remains **8.079769 tok/s** below the floor. Next, compare a
  trained-confidence M=2/M=8 budget against fixed M=8 on representative
  natural reasoning.

### 2026-09-01 04:43 PDT - PERF-A073 DSpark confidence cost budget

- Added a native current-block scheduler behind
  `SGLANG_MLX_NATIVE_DSPARK_CONFIDENCE_COST_RATIO`. It evaluates the official
  seven survival probabilities with proposal tokens and dense q, computes the
  cumulative expected emitted width for M=2 and M=8, and selects M=8 exactly
  when its expected width exceeds M=2 multiplied by the configured complete
  cycle-cost ratio. The shared verifier owns the selected token/q prefix and
  preserves exact rejection, residual sampling, and accepted-state commit.
- The switch accepts only a positive finite ratio, requires native sampling
  and seven proposal tokens, and otherwise fails closed. An absent value keeps
  the fixed seven-token default. DFlash supplies a zero ratio through the same
  verifier and reproduces its selected direct trajectory at
  **31.279682028 tok/s**, width **6.684210526**, digest
  `46bd4bb035b72c2b`, and last token 20. Default DSpark reproduces
  **10.043170988 tok/s**, width **2.428571429**, digest
  `5a38c7070d7badeb`, and last token 16.
- Direct screening at ratios **1.694 / 1.4 / 1.3 / 1.5** reached respectively
  **11.231309029 / 13.400887301 / 12.945884584 / 11.398039610 tok/s**.
  Natural 6.2K-history traces then measured roughly **144--147 ms** for M=2
  and **257--261 ms** for M=8, a complete-cycle ratio near **1.77**. The
  ratio-1.4 real request reached **10.916 tok/s** versus its adjacent
  fixed-M=8 control at **11.313 tok/s**, closing the underpriced threshold in
  PERF-FA101.
- Ratio **1.75** produced five consecutive exact sampled `6237+128` results
  of **13.595 / 13.609 / 13.603 / 13.607 / 13.615 tok/s**, mean
  **13.6058**. This is **+2.2928 tok/s / +20.267%** over the adjacent control.
  Prompt rates were **109.706 / 109.486 / 109.927 / 109.930 / 109.932
  tok/s**; TTFTs were **56.851961 / 56.966250 / 56.737835 / 56.735866 /
  56.735048 s**; end-to-end times were **66.193580 / 66.298033 / 66.074309 /
  66.069194 / 66.062970 s**.
- Every candidate sample completed exact 6,365 tokens with
  `finish_reason=length` and shared reasoning/output SHA-256
  `7af3ef829ce6717a226bb3d06ad62dfa18bcf44b15684ac9cc1c614b3b7349bd`.
  Strict warning-as-error native library and standalone-test builds pass, as
  do the YaRN/confidence/budget cases and focused native pytest (**8 passed**,
  16 existing warnings). Health, language-only model metadata, shutdown,
  listener/process cleanup, 93% free memory, and normal thermal status pass.
- Decision: retain ratio 1.75 as opt-in DSpark scheduling infrastructure. Its
  **13.6058 tok/s** remains **1.7222 tok/s** below selected DFlash2 and
  **6.3942 tok/s** below the requested floor. The next DSpark cost branch is a
  bounded target-only bypass for low-value draft cycles.

### 2026-09-01 05:10 PDT - PERF-A074 DSpark low-budget probe cooldown

- Added checked opt-in `SGLANG_MLX_NATIVE_DSPARK_BYPASS_REFILLS=0..1024`.
  After a confidence-budgeted block selects the measured M=2 tier, the next
  configured number of refills execute exact target-only decode while
  advancing the DSpark target-hidden cache from the same captured target
  states. The countdown then forces a fresh DSpark probe. Request reset clears
  the countdown; a positive value requires confidence budgeting; absent or
  zero keeps the existing default path.
- Extracted the existing reasoning-cap target-only block into the shared
  speculative refill owner. DFlash and ordinary DSpark cap paths retain their
  previous behavior. A bypassed DSpark refill additionally observes a sampled
  reasoning-close token at its immediate emission boundary.
- On sampled direct `128 / 32 warm / 256 timed`, ratio-1.75 without cooldown
  reached **24.494388127 tok/s**, 47 refills, width **5.553191489**, digest
  `75fac01a0ef1f7d8`. Cooldowns 4/8/16/32 reached respectively
  **19.198827552 / 17.781301933 / 24.332648695 / 19.215717015 tok/s** on
  different exact trajectories. Sixteen was the only cooldown admitted to
  the real workload; PERF-FA103 closes the others unchanged.
- A preliminary exact real-131K-pool `6237+128` request at cooldown 16 reached
  **17.031 tok/s**, **109.202 prompt tok/s**, **57.114227 s TTFT**, and
  **64.571419 s** end to end. A fresh repeated launch produced first-five
  decode rates **17.028 / 17.080 / 17.036 / 17.102 / 14.952 tok/s**, mean
  **16.6396**, followed by a recovered sixth sample at **17.011 tok/s**.
  Five normal samples within the six-request window average **17.0514 tok/s**.
- The fifth repeated sample retained the identical output while live draft,
  verify, sample, and commit stages inflated together; it is retained as a
  transiently contended **14.952 tok/s** observation. macOS reported no
  thermal or performance warning, the following sample recovered, and no
  specific competing process was established. All six raw results remain in
  the experiment ledger.
- Every repeated request completed exact 6,365 tokens with
  `finish_reason=length`, coherent reasoning, and shared output/reasoning
  SHA-256
  `1d1398eb3dfe3ae4813c1e53c12258506484e0821e19361a2286495c685a8851`.
  The first-five mean improves PERF-A073 by **3.0338 tok/s / 22.297844%** and
  exceeds the selected DFlash2 sample by **1.3116 tok/s**. It remains
  **3.3604 tok/s** below the requested floor, so this is retained opt-in
  infrastructure with an independent promotion window still open.
- Warning-as-error native library and standalone-test builds pass; focused
  native pytest passes **8 tests** with 16 existing warnings. Default DSpark
  reproduces width **2.428571429** and digest `5a38c7070d7badeb`; DFlash
  reproduces width **6.684210526** and digest `46bd4bb035b72c2b`. Invalid
  bypass-without-budget configuration fails closed. Health, language-only
  metadata, shutdown, listener/process cleanup, 93% free memory, and normal
  thermal status pass.
- Decision: retain cooldown 16 as the best measured speculative serving lane,
  default disabled. The next scheduler candidate requires a cheap pre-draft
  signal that suppresses probe cost while preserving access to high-yield
  regions; periodic widening alone cannot supply the remaining floor margin.

### 2026-09-01 05:33 PDT - PERF-A075 DSpark target-state score telemetry

- Added trace-only evaluation of the existing trained DSpark confidence
  projection over the normalized current target hidden state and current-token
  Markov embedding. With speculative tracing disabled, the new path is
  unreachable. The final code makes no scheduling decision from this score.
- A ratio-1.75 direct trace completed at **24.321012666 tok/s**, 47 timed
  refills, width **5.553191489**, digest `75fac01a0ef1f7d8`, and last token 19.
  Across warmup and timed cycles, M=2 and M=8 target-state scores averaged
  **0.476591** and **0.407225**; score versus accepted length had Pearson
  **-0.291199**. A score above 0.5 identified six M=2 and zero M=8 cycles in
  this synthetic trajectory.
- Temporarily made the score a checked pre-draft bypass. Direct thresholds
  0.475/0.5/0.525/0.55/0.6 reached **16.393088197 / 30.299622883 /
  33.221518376 / 30.796768648 / 16.111013104 tok/s** on different exact
  sampled trajectories. A fresh 0.525 repeat reached **32.788326664 tok/s**;
  its traced repeat reached **32.834047040 tok/s**, with 18 bypasses, nine M=2
  verifies, and 33 M=8 verifies across warmup and timing.
- The real 131K-pool `6237+128` threshold-0.525 request reached **13.652
  tok/s**, **109.133 prompt tok/s**, **57.150588 s TTFT**, and **66.453264 s**
  end to end. It completed exact 6,365 tokens with `finish_reason=length` and
  reasoning/output SHA-256
  `c92e4510efb86ca08711fe296f67632ac1bc79684811c24d19ceca40c042f70b`.
  A fresh no-threshold trace reached **13.580 tok/s** and reproduced the
  selected ratio-1.75 digest `7af3ef82...`. In its 48 natural cycles, M=2/M=8
  target-state scores averaged **0.538717 / 0.585506**, their ranges overlapped
  **0.355775--0.777300 / 0.348645--0.827828**, and score versus accepted width
  had Pearson **0.140841**. Threshold 0.525 would suppress 13 cycles from each
  tier. The scheduler source and environment control were removed; see
  PERF-FA104.
- Extended the fixed-cooldown direct screen to 64 and 128 refills. They reached
  **18.401520476 / 19.663812056 tok/s**, with **249 / 235** timed refills and
  mean widths **1.028112450 / 1.089361702**. Both trail the selected 16-refill
  direct result; PERF-FA103 now includes them.
- DFlash2 against the faster QKV-restored target reached only
  **10.130497494 tok/s** and width **2.653061224**. Restoring every recurrent
  projection reached **7.841259160 tok/s** and width **1.984375**. The full-Q4
  target remains the compatible DFlash2 target; see PERF-FA105.
- A two-phase 64-column M8 affine verifier reused input fragments inside the
  existing 32 KiB threadgroup allocation. Strict builds and parity passed with
  the exact selected digest and width, while direct throughput fell to
  **8.960931772 tok/s** from the restored 32-column **31.317933897 tok/s**.
  The 64-column kernel was removed; see PERF-FA106.
- Final warning-as-error library and standalone DSpark test builds pass. YaRN
  offsets 0/9,000/131,071, confidence/budget parity, and the final test marker
  pass; focused native pytest passes **8 tests** with 16 existing warnings.
  Default DSpark reproduces **10.036881881 tok/s**, width **2.428571429**, and
  digest `5a38c7070d7badeb`. DFlash reproduces **31.317933897 tok/s**, width
  **6.684210526**, and digest `46bd4bb035b72c2b`.
- Decision: retain only the target-state score under the existing trace flag.
  The natural-prompt evidence closes it as a scheduler, target mixing closes
  the faster-target DFlash route, and the next branch returns to shared target
  decode cost.

### 2026-09-01 06:04 PDT - PERF-A076 DFlash2 selector calibration

- Added checked opt-in
  `SGLANG_MLX_NATIVE_DFLASH_SELECTOR_TEMPERATURE`. The native DFlash loader
  accepts a positive finite FP32 value, defaults to exact identity, and
  reports the selected value under speculative tracing. `dflash_select`
  scales the learned unary-plus-transition scores immediately before its
  FP32 softmax; the sampled token's exact q continues through the shared
  rejection verifier.
- Direct `128 / 32 warm / 128 timed` screening at temperatures
  1.0/0.7/0.85/1.15/1.3/0.95/1.05 reached respectively
  **31.291292321 / 14.838952659 / 11.185534971 / 26.936073462 /
  9.563179041 / 37.171244579 / 10.998405302 tok/s** on distinct seeded
  trajectories. This repeated-token screen ranked 0.95 first, so the exact
  natural request remained the admission owner.
- The real temperature-0.95 request reached **13.739 tok/s**, **109.654
  prompt tok/s**, **56.879050 s TTFT**, and **66.122635 s** end to end,
  closing that direct-screen winner. Temperature 1.15 then produced five
  consecutive exact results of **15.870 / 15.884 / 15.890 / 15.896 / 15.893
  tok/s**, mean **15.8866**. The adjacent identity control produced
  **15.434 / 15.449 / 15.443 / 15.443 / 15.443 tok/s**, mean **15.4424**.
  The candidate gains **0.4442 tok/s / 2.876%**.
- Every candidate sample completed exact 6,365 tokens with
  `finish_reason=length` and shared coherent reasoning/output SHA-256
  `3bc2d4a6534c3938d24d5cb2ea2f512a6a868ec80159f5310b3e962b6ecb3606`.
  Every control sample completed the same count and finish reason with shared
  SHA-256
  `51bb9afea1e151fccdd1a6b2fd39ef1be4c9062e7d6065b247cb1a7993161d66`.
  Candidate traces used 32 refills to emit 129 verifier outputs, mean
  **4.031250**; controls used 33 refills to emit 128, mean **3.878787879**.
- A zero selector temperature fails closed with the exact positive-finite
  diagnostic. The strict warning-as-error library and repository dylib builds
  pass, focused native pytest passes **8 tests** with 16 existing warnings,
  health/language-only metadata pass, and every foreground server exits
  cleanly. Identity remains the default while the 1.15 setting stays opt-in;
  its selected real mean retains a **4.1134 tok/s** floor gap. The next branch
  measures DFlash verification-width economics at the natural request.

### 2026-09-01 06:18 PDT - PERF-A077 M=8 gate/up fusion lacks an execution margin

- Began from signed `095b4ba664479dbb1e44053a248040df4f9a209a`
  (`perf(mps): calibrate DFlash selector temperature`), 57 commits ahead of
  `origin/main`, with a clean worktree, clear listener/process preflight, 92%
  free memory, zero throttled pages, and normal thermal/performance status.
- Reopened PERF-FA081 only at its recorded boundary: a native M=8 kernel
  consumed the original gate/up affine tensors without concatenated storage.
  The first form ran gate then up in one threadgroup, kept only four
  accumulators live per phase, retained the gate reduction across phases, and
  emitted the BF16 SwiGLU product directly. A permanent C++ probe established
  bit-exact output against separate SG16/B32 products plus MLX BF16
  SiLU/multiply.
- The exact traced control at selector temperature 1.15 reached
  **26.801071247 tok/s**, 22 timed refills, width **6.090909091**, digest
  `6de63586df62ab2b`, and last token 220. Steady verify generally measured
  **186.215--187.345 ms**. The sequential fused-SwiGLU form preserved the
  trajectory while reaching **26.413620964 tok/s** and steady verify
  **189.425--190.547 ms**. Serializing the two formerly independent output
  grids added roughly **3.3 ms** per verification and closed this form.
- The second form narrowed the candidate to launch submission only. One Metal
  grid used its z plane to select gate or up tensors and wrote two independent
  outputs; each workgroup retained the selected SG16/B32 arithmetic and
  occupancy, and the existing MLX SiLU/multiply remained the consumer.
  Standalone K/N `512/256` gate and up outputs were both bit-exact against two
  independent selected-kernel calls.
- Five process-isolated adjacent no-trace pairs on exact `128 / 32 warm / 128
  timed` produced:

  | Pair | Separate tok/s | Paired tok/s | Delta tok/s |
  |---:|---:|---:|---:|
  | 1 | 26.928227174 | 26.988569313 | +0.060342139 |
  | 2 | 26.969434153 | 26.966388240 | -0.003045913 |
  | 3 | 26.932138773 | 26.973712039 | +0.041573266 |
  | 4 | 26.950369132 | 26.960561359 | +0.010192227 |
  | 5 | 26.936430574 | 26.935599929 | -0.000830645 |

  Means are **26.943319961 / 26.964966176 tok/s**, a candidate movement of
  **+0.021646215 tok/s / +0.08034%**. Every arm reproduced 22 refills, width
  **6.090909091**, digest `6de63586df62ab2b`, and last token 220. The movement
  is smaller than pair noise and does not justify a new kernel, public helper,
  test surface, environment switch, or production server window.
- Both candidate libraries and standalone tests built under
  `-Wall -Wextra -Werror` with only the established macOS 26.0 / MLX 26.2
  linker warning. All experimental source and tests were removed through
  `apply_patch`; `git diff --check` returned the worktree to clean before this
  evidence update. PERF-FA108 closes both forms. The next branch requires a
  material target-verifier cost mechanism or a proposal signal with natural
  acceptance correlation.

### 2026-09-01 06:54 PDT - PERF-A078 selected-q budgeting improves DFlash2 serving 1.763%

- A traced natural no-policy request established the current-block signal.
  Mean selected proposal q over positions one through six has Pearson
  **0.801577** with accepted length across 32 cycles; accepted-length lag-one
  autocorrelation is only **0.324698**. This places the decision after exact q
  production and before target verification.
- Added checked opt-in
  `SGLANG_MLX_NATIVE_DFLASH_MEAN_Q_THRESHOLD`. An absent value is zero and
  leaves the prior path untouched. A present value must be finite, positive,
  and at most one. The common exact verifier reads the already-evaluated sparse
  support, averages the selected q for the first six positions, and verifies
  one proposal when the mean falls below the threshold. Existing exact prefix
  slicing forwards the matching sparse indices and q to rejection/residual
  sampling. Trace now reports selected q, mean-q6, target p, and per-position
  acceptance probability.
- The exact traced direct `128 / 32 warm / 128 timed` threshold-0.62 screen
  reached **22.891918473 tok/s**, 32 refills, mean emitted width **3.843750**,
  digest `79bf856f14024a3e`, and last token 20. Steady M=2 cycles cost about
  **120 ms**, versus about **216 ms** for M=8. The unchanged fixed-screen
  trajectory remains faster, so the representative natural request owns
  selection.
- Real-131K-pool one-request admission screens at thresholds 0.55/0.62/0.65
  reached **15.191 / 16.151 / 16.056 tok/s**. The retained 0.62 arm then ran
  five consecutive exact requests:

  | Sample | Generation tok/s | Prompt tok/s | TTFT s | E2E s |
  |---:|---:|---:|---:|---:|
  | 1 | 16.167 | 109.774 | 56.816503 | 64.672015 |
  | 2 | 16.181 | 109.308 | 57.058949 | 64.907745 |
  | 3 | 16.178 | 109.511 | 56.953346 | 64.803690 |
  | 4 | 16.179 | 109.113 | 57.160957 | 65.010610 |
  | 5 | 16.183 | 109.452 | 56.984009 | 64.831852 |

  Mean generation is **16.1776 tok/s**. Every request completed exact 6,365
  tokens with `finish_reason=length`, coherent reasoning, and shared
  output/reasoning SHA-256
  `bdf9428e30eca513e416f3dc69973a6d28ecf9ec864abdd3e72442eec8fda653`.
  Each trace contains 23 M=2 and 19 M=8 cycles; their mean complete costs are
  **133.648 / 248.447 ms** and their mean emitted widths are **1.608696 /
  4.947368**. The 42 cycles emit 131 server-side tokens before the client's
  exact 128-token limit.
- The adjacent current-source threshold-disabled control was **15.883 /
  15.908 / 15.900 / 15.899 / 15.897 tok/s**, mean **15.8974**. Every control
  completed exact 6,365 tokens with shared SHA-256
  `3bc2d4a6534c3938d24d5cb2ea2f512a6a868ec80159f5310b3e962b6ecb3606`.
  Its 32 M=8 cycles emit 129 tokens, mean width **4.031250**, at mean complete
  cost **248.092 ms**. Candidate delta is **+0.2802 tok/s / +1.763%**.
- Strict warning-as-error native compilation passes with the established MLX
  linker warning. Present values 0 and 1.01 fail closed with the exact range
  diagnostic. Focused native pytest passes **8 tests** with 16 existing
  warnings. Unset-path direct regressions reproduce DFlash width
  **6.684210526**, digest `46bd4bb035b72c2b`, and last token 20 at
  **31.302904851 tok/s**, plus DSpark width **2.428571429**, digest
  `5a38c7070d7badeb`, and last token 16 at **10.117263162 tok/s**.
- Retain threshold 0.62 as an opt-in DFlash2 scheduler. The default remains
  full M=8. The measured candidate is **3.8224 tok/s** below the required
  served floor, so target-cycle arithmetic and proposal quality remain active.

### 2026-09-01 07:09 PDT - PERF-A079 target-only internal prefill chunking

- Generalized the native prefill owner with a checked target-only chunk size.
  `SGLANG_MLX_NATIVE_TARGET_ONLY_PREFILL_CHUNK_SIZE` accepts integers from 1
  through 8,192 and defaults to zero, preserving the one-shot path. DFlash2
  and DSpark continue to use their fixed 2,048-token target-capture chunks;
  the ordinary MTP lane remains unchanged.
- Current-source full-Q4 target-only sampled `128 / 32 warm / 256 timed`
  baseline was **20.306236488 tok/s**, digest `35b0d203b5a2b19d`, last token
  71093. The candidate measured **20.339874670 tok/s** with the option absent
  and **20.301552601 tok/s** at chunk 2,048, preserving that digest and final
  token in both arms. Values 0 and 8,193 fail before model loading with the
  exact checked-range diagnostic.
- Chunk 2,048 completed direct `6237 / 32 warm / 128 timed` at
  **19.586704594 tok/s**, digest `c7d6e4c732907ca5`, last token 83. The exact
  real-131K-pool served request then completed at **19.300 generation tok/s**,
  **109.988 prompt tok/s**, **56.706198 s TTFT**, and **63.286374 s E2E**.
  It produced exact `6237+128=6365` tokens, `finish_reason=length`, coherent
  reasoning, and SHA-256
  `17f13ded7c4f2bbae58af6de1e825a6e5b5d0055a3589e3b2c6e048c6f9f5cd7`.
- Strict `-Wall -Wextra -Werror` compilation passed with only the established
  external MLX linker warning. Focused native pytest passed **8 tests** with
  16 existing warnings. Refactored-path regressions reproduced DFlash2
  **31.353044621 tok/s**, width **6.684210526**, digest
  `46bd4bb035b72c2b`, and DSpark **10.048528637 tok/s**, width
  **2.428571429**, digest `5a38c7070d7badeb`.
- Retain chunk 2,048 as the target-only long-prefill setting. It converts the
  representative one-shot OOM into an exact served completion while leaving
  the default path unchanged. Target-only long-history decode remains
  **0.700 tok/s** below the client floor, making single-token target arithmetic
  and serving-loop cost the next active branch. See PERF-FA110.

### 2026-09-01 07:28 PDT - PERF-A080 batch-one affine QMV screen

- Reproduced current signed source at the representative deterministic direct
  shape: `6237 / 32 warm / 128 timed` reached **6.496580792 s /
  19.702671928 tok/s**, digest `e339db7c4c119325`, and last token 95726.
  This remains consistent with the preceding **19.586704594 tok/s** sampled
  long-history result and isolates native sampling to a small fraction of the
  remaining served gap.
- Added a temporary affine-W4/G64 Metal QMV with eight SIMD groups per
  threadgroup and one SIMD group per output row. The first form dequantized
  each packed word independently. The second loaded each scale and bias once
  per eight-lane quantization group and broadcast them with `simd_shuffle`.
  Both used FP32 accumulation and BF16 output behind one opt-in switch.
- Strict library/test builds passed. Standalone parity covered batch one at
  K/N `128/128`, `5120/64`, and `512/6144`; maximum absolute difference from
  MLX was **0.03125**. The original small-batch and M8 parity matrix also
  passed unchanged.
- Against the current short target-only control of **20.339874670 tok/s**, the
  independent-load form reached **10.456143330 tok/s** and the subgroup-
  broadcast form reached **7.773375044 tok/s**, regressions of **48.593%** and
  **61.783%**. Both full-model screens produced the same candidate digest
  `e446d211f2e2ff25` and last token 15.
- The native QMV, public helper, test additions, and environment switch were
  removed through `apply_patch`; `git diff --check` returned clean before this
  record update. PERF-FA111 closes this scalar-output geometry. A future
  affine route requires a stock-competitive matrix/SIMD tile or a fused
  downstream consumer that eliminates measured work.

### 2026-09-01 07:41 PDT - PERF-A081 long-history shader attribution

- Recorded the full-Q4 target-only sampled `6237 / 1 warm / 128 timed` shape
  with Metal GPU Counters and Shader Timeline enabled. The instrumented target
  completed in **6.686056458 s / 19.144319346 tok/s**, digest
  `b9b90c038014cbe7`, and last token 40218. Profiling overhead changes timing
  and the sampled trajectory, so this run supplies attribution only.
- Exported the target shader address ranges and GPU program-counter samples.
  A temporary strict C++20 aggregator mapped **47,676 / 48,343** sampled PCs:
  **42,769 / 88.470%** are MLX
  `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0`; **1,887 / 3.904%** are the
  two long-history SDPA passes; **1,140 / 2.358%** are the fused recurrent
  state update; and **679 / 1.405%** are the fused full-attention q/k norm plus
  RoPE kernel. Every remaining individually mapped owner is below 0.71%.
- The installed dependency is MLX **0.32.2**, official tag
  `1f8e74e3f12f31365464a6867c6579f0e9b29d85`. Official HEAD
  `117188cd735299f396a08f7697a81759b1e0550b` has no post-tag QMV change; its
  relevant Metal attention change adds GQA kernels at histories of at least
  8,192 and does not cover the 6,237-token admission shape.
- Removed the temporary aggregator source after strict compilation and use.
  Retained the reproducible trace and XML exports under `/private/tmp`. The
  next candidate must preserve MLX's multi-output SIMD accumulation geometry
  while improving its output tile, parameter loads, or dependency dispatch.

### 2026-09-01 08:44 PDT - PERF-A082 runnable Q5 baseline

- Pinned both Bartowski source artifacts at revision
  `f0eec4a4bb4975114a030d048952d83c0a53c034`. Q5_K_M is exactly
  **20,752,787,040 bytes** with SHA-256 `e731e180...caa8`; Q5_K_S is exactly
  **19,680,945,760 bytes** with SHA-256 `b52fbc24...e569`. Actual-file native
  MPS parity passes every representative packed family in each source.
- The unchanged Q5_K_M server reaches mixed merged-weight processing and fails
  on Q8_0 shards unsupported by that native Metal path. The unchanged Q5_K_S
  loads as `Qwen3_5ForCausalLM`, reports **20.00 GB** of weights and
  **11.99 GB** available, allocates its small caches, then warmup reaches the
  unsupported Q5_K embedding boundary. PERF-FA112/113 retain these closed
  source-artifact routes.
- Built official llama.cpp build 10547 at pinned commit
  `749f688fcaa4c472ec034b08cb8a907c45cfaa02` with Metal disabled. A narrow
  external C++ converter patch permits the explicit token-embedding override
  during `COPY`; all unspecified tensors retain their source encoding. The
  exact conversion command was:

  ```bash
  /private/tmp/llama.cpp-q5/build-q5/bin/llama-quantize \
    --allow-requantize --token-embedding-type F16 \
    /Users/dcazares/.cache/huggingface/hub/models--bartowski--Qwen3.8-27B-GGUF/snapshots/f0eec4a4bb4975114a030d048952d83c0a53c034/Qwen3.8-27B-Q5_K_S.gguf \
    /Users/dcazares/.cache/sglang/checkpoints/Qwen3.8-27B-Q5_K_S-TokenF16.gguf \
    COPY 4
  ```

  Conversion changes only `token_embd.weight` from Q5_K **833.59 MiB** to F16
  **2,425.00 MiB**. The other 865 tensors remain copied. The artifact is
  exactly **21,349,656,160 bytes**, SHA-256
  `c05a777870159b0779a441e2f58b543a0660af466d833d5b550a8aab9c17fcfb`.
  Actual-file parity passes Q4_0, Q5_K, and Q6_K with maximum absolute errors
  `4.76837e-07`, `9.53674e-07`, and `4.76837e-07`.
- A conservative 1,024-token foreground launch loads the derived checkpoint
  in **68.21 s**, reports **21.37 GB** of weights and **10.62 GB** available,
  allocates one 0.29 GB Mamba slot plus 0.06 GB BF16 KV, completes native Metal
  warmup, and serves `qwen3.8-27b-q5`. `/model_info` reports Qwen3.5 text and
  image/audio understanding disabled.
- The first ordinary sampled `128+32` request completed exact 160 tokens with
  `finish_reason=length`, all 32 output tokens in `reasoning_content`, and
  output/reasoning SHA-256
  `70977c61cdccc1f820d5a48cfe8621c2a82d7c94e1e94b76691a525c362e5c05`.
  Generation was **5.746 tok/s**, prompt **5.8 tok/s**, TTFT
  **22.070851 s**, and end to end **27.466190 s**. The active decode gap is
  **14.254 tok/s / 3.481x**. Profiling the reachable Q5_K/Q6_K projection path
  owns the next candidate.
- Listener PID 20456 and children 20459/20460/20461 were resolved before a
  foreground `Ctrl+C`. All four PIDs, matching workloads, and port 30000 were
  absent afterward. Memory returned to 95% free with zero throttled pages and
  normal thermal/performance status.

### 2026-09-01 09:04 PDT - PERF-A083 Q6_K batch-one two-row reuse

- Change: adapted pinned llama.cpp build 10547's Q6_K matrix-vector mapping at
  commit `749f688fcaa4c472ec034b08cb8a907c45cfaa02`. Two 32-thread SIMD groups
  each reuse one sixteen-value activation fragment across two contiguous output
  rows. The common native Metal quantized-matmul owner selects the new path only
  for batch one, complete four-row output cohorts, aligned compact views, and a
  32-wide pipeline capable of 64 threads. The generic kernel remains available
  through `SGLANG_MPS_Q6_K_BATCH1_ROWS2=0` and owns every failed guard.
- Actual-file parity on the derived Q5 checkpoint passed the optimized 16-row
  domain at Q6_K maximum absolute/relative error
  **2.98023e-07 / 1.84679e-07**, the 17-row fallback at
  **4.76837e-07 / 2.95486e-07**, and untouched batch-eight execution at
  **1.90735e-06 / 5.17281e-07**. The same commands passed all Q4_0 and Q5_K
  representatives.
- Representative Q6_K QKV matched windows used eight warmups and 25 timed
  synchronized iterations. Candidate raw milliseconds were
  `0.505667,0.480250,0.472250,0.472292,0.467542,0.470875,0.420750,0.436709,0.426916,0.440166,0.430208,0.431083,0.408542,0.411167,0.420458,0.464000,0.454042,0.562750,0.461916,0.450500,0.455083,0.443375,0.446542,0.443875,0.448125`;
  control raw milliseconds were
  `0.934292,0.895042,0.820542,0.765208,0.739458,0.727042,0.761625,0.762875,0.753666,0.756208,0.759292,0.744917,0.752209,0.756458,0.738875,0.725667,0.729625,0.737000,0.748917,0.741458,0.738041,0.751417,0.748125,0.738916,0.748166`.
  Medians improve **0.748917 -> 0.448125 ms**, a **40.2%** reduction, and
  effective bandwidth rises **53.559 -> 89.510 GiB/s**.
- Full Q6_K vocabulary-head candidate raw milliseconds were
  `3.409916,3.369167,3.323542,3.343833,3.398375,3.371625,3.348250,3.316583,3.321625,3.310583,3.324625,3.318875,3.368708,3.358375,3.337458,3.342958,3.321375,3.309542,3.294125,3.282834,3.290000,3.283791,3.357209,3.371958,3.320000`;
  matched control raw milliseconds were
  `12.169000,12.017916,11.969959,12.002459,12.026000,11.976916,11.952083,12.032125,11.994584,12.051875,12.047541,11.966500,11.994291,12.009959,11.947250,11.970917,11.989958,11.959583,12.001833,12.009166,12.019875,12.035583,12.090417,12.115042,12.035375`.
  Medians improve **12.009166 -> 3.324625 ms**, a **72.3%** reduction, and
  effective bandwidth rises **80.960 -> 292.442 GiB/s**.
- A clean foreground server loaded the same 21.37 GB artifact and completed
  five cache-flushed ordinary-sampling `128+32` requests. Generation samples
  were **6.685 / 7.176 / 7.134 / 7.135 / 7.131 tok/s**, mean **7.0522** and
  warmed-four mean **7.1440**. Prompt samples were
  **5.336 / 5.822 / 5.794 / 5.806 / 5.805 tok/s**; TTFT samples were
  **23.985920 / 21.984879 / 22.092834 / 22.046234 / 22.048684 s**; E2E
  samples were **28.622898 / 26.304740 / 26.438155 / 26.391247 / 26.395664
  s**. Five-run generation improves **22.732%** over the committed
  **5.746 tok/s** baseline. Every request returned exact `128+32=160` tokens,
  `finish_reason=length`, and all 32 output tokens as preserved reasoning.
- Root/listener PID 20985 and children 20992/20993/20994 were resolved before
  foreground `Ctrl+C`. All PIDs, matching workloads, and port 30000 were absent
  after cleanup. Memory returned to 94% free with zero throttled pages and
  normal thermal/performance status. The active Q5 gap is now
  **12.9478 tok/s / 2.8360x**; remaining Q5_K projections, model execution
  overhead, 131K capacity, and speculative economics own the next iterations.

### 2026-09-01 09:45 PDT - PERF-A084 Q5_K batch-one 32-row cohorts

- Change: added a Q5_K batch-one Metal matvec in which four lanes cooperate on
  each output row, each lane decodes two adjacent `float4` fragments, and four
  SIMD groups cover 32 rows per 128-thread group. The shared native quantized
  matmul owner selects it for aligned batch-one Q5_K projections with at least
  5,120 output rows, the smallest measured winning shape. The 1,024-row
  attention K/V shapes retain the
  prior eight-lane path. `SGLANG_MPS_Q5_K_BATCH1_ROWS32=0` selects the matched
  control for the wider shapes.
- The initial eight-warmup/25-timed microbench screen measured control to
  candidate medians of **0.648083 -> 0.625667 ms** on
  `blk.0.ffn_gate.weight`, **0.670209 -> 0.664542 ms** on the FFN down
  projection, and **0.549750 -> 0.388458 ms** on
  `blk.0.attn_gate.weight`. The 1,024-row attention-K shape moved
  **0.327208 -> 0.337792 ms**, establishing that compact shape's exclusion.
  A later independent cache-resident pair measured attention gate
  **0.554417 -> 0.546000 ms** and FFN gate
  **0.619875 -> 0.632416 ms**. Whole-model serving therefore owns the
  promotion decision; the direct synchronized microbench varies with cache
  residency on this 21.37 GB model.
- The first reversed-control served comparison used five cache-flushed sampled
  `128+32` requests per arm. Candidate generation samples were
  **6.993 / 7.214 / 7.206 / 7.218 / 7.192 tok/s**, mean **7.1646** and
  warmed-four mean **7.2075**. Disabled-control samples were
  **6.956 / 7.187 / 7.141 / 7.216 / 7.171 tok/s**, mean **7.1342** and
  warmed-four mean **7.17875**. The candidate gains
  **0.0304 tok/s / 0.426%** by all-sample mean and **0.400%** by warmed mean.
  Candidate prompt/TTFT/E2E samples were
  `5.786/22.120818/26.554005`, `5.819/21.998563/26.295831`,
  `5.820/21.994362/26.296606`, `5.808/22.039649/26.334344`, and
  `5.812/22.022324/26.332560`; their means are **5.809 tok/s**,
  **22.035143 s**, and **26.362669 s**. Every arm returned exact 160-token
  length completions with all output carried as reasoning.
- A higher-resolution reversed-order comparison used five sampled `128+128`
  requests per arm. Disabled-control decode samples were
  **7.277 / 7.474 / 7.460 / 7.456 / 6.677 tok/s**; the fifth coincided with
  39.4% `launchd`, 11.9% FileProvider, active Spotlight workers, and server-log
  decode intervals of 6.58--6.59 tok/s. It remains in the all-sample mean
  **7.2688**, while the median is **7.456** and first-four mean is
  **7.41675**. Candidate samples were
  **7.261 / 7.521 / 7.491 / 7.514 / 7.500 tok/s**, mean **7.4574**,
  median **7.500**, first-four mean **7.44675**, and warmed-four mean
  **7.5065**. Candidate versus control improves the median by
  **0.044 tok/s / 0.590%**, first-four mean by **0.404%**, and matched
  positions two through four by **0.607%**. All ten requests completed exact
  256-token length responses with 128 streamed reasoning fragments.
- Direct candidate execution before the production size threshold passed
  actual-file Q5_K rows `1,7,8,9,15,16,17,31,32`, long-K compact views,
  unaligned fallback, and synthetic packed extrema. The long-K maximum
  absolute/relative error was **7.15256e-07 / 3.69632e-07**; synthetic
  extrema remained **0.00146484 / 3.02919e-07**. The final checked source
  rebuilt during two complete server launches. The post-threshold boundary
  suite and derived-artifact Q4_0/Q5_K/Q6_K smoke both pass; final Q5_K
  representative error is **9.53674e-07 / 4.60024e-07**. A 2,049-row
  all-family smoke request reached a smaller Q4_0 tensor first and triggered
  its packed-shape guard, so it contributes only a recorded harness-boundary
  failure.
- Decision: retain the narrow default-on Q5_K path. Two matched serving
  windows in opposite launch order agree on a small decode gain, and the
  wider 128-output window supplies the promotion signal. The original generic
  baseline to current `128+32` mean improves **5.746 -> 7.1646 tok/s**
  (**+24.688%**) after PERF-A083/A084. The active gap is
  **12.8354 tok/s / 2.7915x**. Root/listener trees
  21640/21644--21646, 21785/21793--21795, and 21884/21889--21891 were each
  resolved before foreground shutdown; every PID, matching workload, and port
  30000 was clear afterward, with normal reported thermal state.

### 2026-09-01 10:20 PDT - PERF-A085 Q6_K exact-batch-four 16-row cohorts

- Change: added an exact-batch-four Q6_K Metal kernel in which each eight-lane
  cohort owns one output row, vector-decodes four adjacent weights per lane,
  reuses the row across all four activations, and produces sixteen rows per
  128-thread group. The common native quantized-matmul owner selects it only
  for aligned Q6_K batch four with a 32-wide pipeline and 128-thread capacity.
  `SGLANG_MPS_Q6_K_BATCH4_ROWS16=0` preserves the matched generic route.
- Eight-warmup/25-timed synchronized head samples moved from disabled-control
  median **29.166000 ms / 33.433 GiB/s** to candidate
  **6.121375 ms / 159.293 GiB/s**, a **79.012%** latency reduction. The
  representative `blk.0.attn_qkv.weight` projection moved
  **1.442333 ms / 27.929 GiB/s -> 0.588500 ms / 68.451 GiB/s**, a
  **59.198%** reduction. These are matched final-source processes.
- Direct derived-artifact correctness at 17 rows and exact batch four passed
  Q4_0/Q5_K/Q6_K with maximum absolute/relative errors
  `1.90735e-06/6.27034e-07`, `2.6226e-06/7.75595e-07`, and
  `1.66893e-06/8.06965e-07`. Q6_K batch-three fallback passed at
  `1.54972e-06/4.32145e-07`; batch eight passed at
  `1.90735e-06/5.17281e-07` through its established specialization.
- The first same-GGUF NEXTN launch omitted explicit draft quantization. The
  draft resolved as BF16, consumed the remaining **10.62 GB**, and exited
  before KV allocation. Adding
  `--speculative-draft-model-quantization gguf` resolved the trained draft as
  GGUF, loaded it in **1.00 GB**, and left **8.50 GB** after target/draft
  state and 1K KV allocation. This ordering requirement matches the retained
  earlier Qwen GGUF provenance.
- Five exact sampled `128+128` candidate requests produced generation samples
  **3.643 / 3.799 / 3.711 / 3.699 / 3.662 tok/s**, mean **3.7028**. The
  matched environment-disabled restart produced
  **3.215 / 3.619 / 3.269 / 3.373 / 3.216 tok/s**, mean **3.3384**. The
  kernel improves served generation by **0.3644 tok/s / 10.915%**. Candidate
  accepted-length telemetry averaged **2.960** versus control **3.024**, so
  the measured speedup is independent of proposal-window favorability. Every
  request returned exact 256-token length output as preserved reasoning.
- Decision: retain the narrow default-on kernel. The complete same-GGUF
  three-step NEXTN lane remains an experimental control at **3.7028 tok/s**,
  **50.629%** below the selected target-only `128+128` median and
  **16.2972 tok/s / 5.4013x** from the requested floor. A smaller or faster
  draft topology owns any reopening of that complete configuration. Candidate
  root/listener 22210 owned 22213--22215; disabled-control root/listener 22332
  owned 22341--22343. Both trees exited through foreground `Ctrl+C`; all
  known PIDs, port 30000, compiler workers, and matching workloads were clear,
  memory returned to 94% free with zero throttled pages, and reported thermal
  state was normal.

### 2026-09-01 10:39 PDT - PERF-A086/A087 exact 131K residency boundary

- The derived 21.37 GB target successfully allocated one 0.29 GB FP32 Mamba
  slot and an exact 131,072-token BF16 KV pool at 4.00 GB K plus 4.00 GB V.
  Startup, warmup, health, model length, and language-only metadata passed.
- One ordinary sampled exact `128+32` request completed with preserved
  reasoning at **5.271 prompt tok/s**, **24.284058 s TTFT**,
  **0.102 generation tok/s**, and **329.689187 s E2E**. Unified-memory
  pressure reached 28--50% free and swap traffic rose. This is a capacity pass
  and a residency-performance rejection under PERF-FA116.
- A direct `torch.float8_e4m3fn` MPS conversion/allocation screen failed at
  the framework dtype boundary before a cache tensor existed. PERF-FA117
  closes the unchanged generic FP8 route.
- The verified server tree exited cleanly. Port 30000, all known PIDs,
  matching workloads, and compiler work were absent; memory returned to 94%
  free with zero throttled pages and normal reported thermal state. The next
  resident-byte lever is the immutable source artifact's Q5_K token embedding,
  followed by a native uint8-backed compressed KV cache.

### 2026-09-01 10:55 PDT - PERF-A088 Q4_K token-embedding residency win

- Pinned llama.cpp build 10547 converted exactly `token_embd.weight` from
  833.59 MiB Q5_K to 682.03 MiB Q4_K while COPY preserved all other 865
  source tensors. The distinct artifact is **19,522,020,960 bytes**,
  **1.702118 GiB** smaller than the F16 derivative, and hashes to
  `8ed3117aff80d105a302da708221364579d483964c4c07d56eb6091a774b06ae`.
- Direct actual-file Q4_0/Q4_K/Q5_K/Q6_K and token embedding parity pass. A
  1K-pool sampled smoke reaches **7.086 tok/s** with exact reasoning. The
  arithmetic result is **703**, and the parser emits exactly one
  `multiply({"a":37,"b":19})` call with `finish_reason=tool_calls`.
- Whole-model runtime residency falls **21.37 -> 20.00 GB**. With the exact
  131,072-token BF16 pool, reported post-allocation headroom rises
  **0.00 -> 0.99 GB**. The matched sampled `128+32` result improves from
  **0.102 to 0.315 tok/s**, a **3.088235x / +208.823529%** gain, while prompt
  throughput remains **5.616 tok/s** and TTFT is **22.790664 s**.
- Paging remains visible at 27--36% free memory and up to 1,819.81 MiB swap
  used during the request. Retain the smaller artifact for the next capacity
  iteration; a native uint8-backed KV cache is now the active 4 GB residency
  lever. Both server trees exited cleanly, all PIDs/listeners/workloads were
  absent, memory recovered to 95% free, and reported thermals were normal.

### 2026-09-01 11:17 PDT - PERF-A089 native MPS E4M3FN conversion

- PyTorch **2.11.0** can allocate a `uint8` MPS tensor, reinterpret it as
  `float8_e4m3fn`, and gather indexed rows. Its stock FP32/FP8 value
  conversion remains unavailable. SGLang's generic KV pool already owns that
  exact byte-backed storage layout, so the missing operation was isolated to
  conversion at cache write and attention read boundaries.
- Added a narrow native MPS `_to_copy` implementation to the existing Metal
  extension. It handles contiguous and positive-stride FP32/E4M3FN pairs,
  uses PyTorch's exact E4M3FN round-to-nearest-even bit contract, and forwards
  every other dtype/device/layout case to the original composite
  implementation. The change adds no Python source.
- Exact validation passed **400,006** reference FP32 encodes, every one of the
  **256** raw FP8 decodes at the FP32 bit level, storage-offset and strided
  encode/decode cases, empty tensors, and the ordinary MPS BF16 fallback.
  Extension compilation and `git diff --check` pass.
- The first live implementation admitted the contiguous key tensor and then
  exposed a strided value view. Generalizing the native index calculation to
  eight dimensions closed that boundary. The successful 1,024-token launch
  loaded the 20.00 GB Q5 artifact in **76.38 s**, allocated one 0.29 GB Mamba
  slot and **0.02+0.02 GB** FP8 K/V pools, completed automatic warmup, and
  exposed healthy language-only metadata.
- After an initial **7.044 tok/s** smoke, five cache-flushed sampled
  `128+32` requests measured generation
  **7.248 / 7.277 / 7.268 / 7.251 / 7.263 tok/s**, mean **7.2614**. Prompt
  throughput was **5.835 / 5.866 / 5.886 / 5.865 / 5.882**, mean
  **5.8668 tok/s**. TTFT mean was **21.818682 s** and E2E mean was
  **26.087973 s**. Every request returned exact 160-token length output with
  32 preserved reasoning fragments.
- Deterministic behavior returned final arithmetic answer **703** and exactly
  one parsed `multiply` call with arguments `a=37, b=19` and
  `finish_reason=tool_calls`. Root/listener 24186 owned tracker/scheduler/
  detokenizer 24198/24199/24200. Foreground shutdown cleared every PID, port
  30000, matching workload, and compiler process; memory recovered with zero
  throttled pages and normal reported thermals. Retain the native conversion
  and qualify the exact 131K FP8 pool next.

### 2026-09-01 11:29 PDT - PERF-A090 exact 131K FP8 capacity

- The signed PERF-A089 source relaunched with exact
  `context_length=max_total_tokens=131072`, one request, one Mamba slot,
  memory fraction 0.95, page one, chunk 2,048, and prefill cap 8,192. The
  20.00 GB model loaded in **71.93 s**. Its exact FP8 pool occupied
  **2.00 GB K + 2.00 GB V** and left **6.99 GB** by server accounting,
  compared with 4.00+4.00 GB and 0.99 GB for the same artifact's BF16 pool.
  Automatic warmup, health, maximum model length 131,072, and language-only
  metadata passed.
- One cold-residency sampled request reached **2.638 tok/s** and remained
  exact. A later orchestration call yielded its client session before stdout
  collection; process inspection confirmed that client ended before the
  admitted window, and the sample is excluded. Five subsequent sequential,
  cache-flushed samples measured:

  | Sample | Gen tok/s | Prompt tok/s | TTFT s | E2E s | Output SHA-256 |
  |---:|---:|---:|---:|---:|---|
  | 1 | 3.098 | 5.839 | 21.922251 | 31.929734 | `b56fe5dc...646cc` |
  | 2 | 3.299 | 5.915 | 21.639897 | 31.035651 | `e6d202d3...32353` |
  | 3 | 3.264 | 5.887 | 21.741808 | 31.238994 | `bfb3adf4...ad125` |
  | 4 | 3.273 | 5.928 | 21.592876 | 31.064729 | `2fd793aa...20f17` |
  | 5 | 3.251 | 5.918 | 21.629686 | 31.164948 | `2fd793aa...20f17` |

  Means are **3.237 generation tok/s**, **5.8974 prompt tok/s**,
  **21.705304 s TTFT**, and **31.286811 s E2E**. Every request completed exact
  160-token usage with 32 streamed reasoning fragments. This is
  **10.276190x / +927.619%** over the same artifact's 0.315 tok/s BF16
  result and **31.735294x** over the original F16-embedding/BF16 result.
- The full FP8 pool remains **55.419%** below the 1K FP8 mean of 7.2614 tok/s.
  `memory_pressure` reported 89% system-wide free capacity and zero throttled
  pages while the compressor remained active; encrypted swap usage moved from
  823.75 MiB after the first sample to 857.19 MiB after the full window and
  behavior gates. This isolates a material full-pool residency cost even
  after removing four cache gigabytes.
- Exact-pool deterministic behavior returned arithmetic **703** and exactly
  one parsed `multiply({"a": 37, "b": 19})` call with
  `finish_reason=tool_calls`. Root/listener 24369 owned tracker/scheduler/
  detokenizer 24372/24373/24374. Foreground shutdown cleared every PID, port,
  matching workload, and compiler process. Memory recovered to 95% free,
  swap usage settled at 540.75 MiB, pages throttled remained zero, and
  reported thermals stayed normal.
- Decision: retain FP8 as the exact-capacity cache selection. Another cache or
  model-residency reduction owns the short-context throughput recovery; a
  fused native FP8 attention owner remains required for long populated
  histories where generic SDPA materializes FP32 K/V.

### 2026-09-01 11:43 PDT - PERF-A091 native affine-Q5 target baseline

- Downloaded the text-only affine-Q5/G64 checkpoint
  `lukaskremla/Qwen3.8-27B-5bit-MLX-TextOnly` at immutable revision
  `2568951b893b6427d0a8eb91cc7f4307154c2f05`. The Hub inventory is
  **18,514,909,284 bytes**. Its four model-shard SHA-256 values are
  `21b2bb76...317d8`, `10b2b7d8...22add`, `586db398...8314`, and
  `47b1db6c...222dd`. Local headers expose 498 U32 packed tensors and 1,349
  BF16 tensors, with no `vision_tower` tensors.
- MLX 0.32.2 successfully quantized and executed an isolated affine five-bit
  matrix before the model launch. The existing native engine inferred bit
  width five from the checkpoint and loaded the full target without a source
  change. A `128 / 1 warm / 16 timed` smoke reached **16.225609671 tok/s**.
- The complete sampled direct `128 / 32 warm / 128 timed` control used seed
  42, the 256-token reasoning cap, 128 MiB command-buffer budget, 64 SDPA
  blocks, and internal target-only prefill chunk 2,048. It reached
  **16.322505765 tok/s** in **7.841933208 s**, digest
  `dba01798d97b3f4e`, and last token 15.
- Reusing the selected Q4 DFlash controls unchanged reached only
  **11.506669050 tok/s**, 56 refills, and mean width **2.232142857**. Trace
  showed M=2 verification near 114 ms and M=8 verification near 363--368 ms;
  the native M=8 K-split route currently admits only affine bits two/four, so
  the five-bit target uses generic MLX QMM. PERF-FA119 closes that unchanged
  composition. Next: add exact affine-five-bit unpacking to the common native
  M=8 verifier, validate parity, and rerun fixed-cycle plus natural serving.

### 2026-09-01 11:49 PDT - PERF-A092 affine-Q5 M=8 K-split verifier

- Generalized the selected SG16/B32 M=8 K-split Metal kernel over affine bits
  four/five. Five-bit rows use MLX's exact eight-values/five-bytes packed
  order; each SIMD lane decodes one output column into the existing BF16
  32x32 tile. Four-bit rows keep their aligned 32-bit decode branch.
- A new strict C++ test compares the native result with MLX
  `quantized_matmul` at Q5 K/N `512/256`, `5120/64`, and `17408/32`.
  Maximum absolute errors are **0.03125 / 0.0625 / 0.107422**. The rejected
  K=256 shape fails closed. The complete pre-existing 2/4-bit small-batch,
  M=8, dense, and user-owned batch-one suite also passes.
- The exact PERF-A091 DFlash trace improves from **11.506669050** to
  **13.721235888 tok/s**, a **+19.246%** gain. M=8 verification falls from
  about **363--368 ms** to **228--229 ms**. The candidate completes 128 timed
  tokens with 41 refills, mean width **3.170731707**, digest
  `142b1268bcf49768`, and last token 271. The changed BF16 reduction grouping
  changes the sampled trajectory, so fixed cycle cost and exact p/q execution
  are retained alongside throughput.
- A full-Q4 regression through the generalized kernel preserves exact digest
  `79bf856f14024a3e`, last token 20, 32 refills, and mean width 3.84375. The
  candidate reached **22.286223210 tok/s** versus the adjacent original-kernel
  sample at **20.581517098 tok/s**. Retain the Q5 verifier under the existing
  opt-in QMM controls. Target-only affine Q5 at **16.322505765 tok/s** remains
  faster, so proposal quality and target-cycle bytes remain active.

### 2026-09-01 12:04 PDT - PERF-A093 matched five-bit MTP topology

- Downloaded `lukaskremla/Qwen3.8-27B-MTP-5bit-MLX` at immutable revision
  `1faa5a803c972c57cfc1beed606184e726ad3d85`. Its single safetensor is
  **292,018,299 bytes**, SHA-256
  `f63dd5c230035c032037d7215195cad41e7cc936cd7a1cbff14f5fc590632e6e`,
  and config declares affine five-bit/group-64 with one full-attention MTP
  layer.
- Matched deterministic three/eight-token blocks reach
  **17.919995235 / 31.380317186 tok/s** in
  **7.142859042 / 4.078990000 s**, with mean widths **3 / 8** and identical
  digest `e446d211f2e2ff25`/last token 15. This is a
  **+13.460321951 tok/s / +75.113%** execution-cost win. The earlier
  sampling-enabled legacy-greedy M=8 probe reached **29.818370889 tok/s** at
  width **7.75**; block four reached **16.969662795 tok/s** at width
  **3.969696970**.
- Native sampling now draws every MTP proposal from its recorded dense q and
  routes the block through the common exact p/q rejection sampler. Default
  sampling completed, and temperature screens at 0.25/1.15/1.5/2.0 reached
  **3.990054799 / 4.821419896 / 6.380162135 / 5.062826830 tok/s** with mean
  widths **1.28 / 1.702703 / 2.285714 / 1.8**. Temperature 1.5 plus proposal
  top-k four reached **3.780318328 tok/s**, width **1.26**. Calibration-only
  controls were removed. The final default-q source recheck reached
  **4.397922404 tok/s**, 42 refills, and width **1.523809524**.
- Extending the existing 64-column small-batch affine kernel to Q5 passed M=3
  and M=4 parity at maximum error **0.03125**, yet the greedy block-three
  full-model result fell to **10.679521956 tok/s**. That kernel branch was
  removed. Retain the block-size experiment and exact sampled semantics; move
  the production optimization owner to batch-one target Q5.

### 2026-09-01 12:43 PDT - PERF-A094 affine-Q5 batch-one direct QMV

- Added an opt-in C++/Metal affine-Q5/G64 batch-one owner. Four SIMD groups
  each compute four output rows while every lane loads 16 BF16 activations,
  decodes two eight-value/five-byte packs, and accumulates in FP32. Packed
  four-byte weight loads and compile-time K specialization retain the selected
  source. Dispatch requires batch one, BF16 input/parameters, U32 packed
  weights, five bits, group 64, K divisible by 512, and N divisible by 16.
- Strict warning-as-error dylib and standalone-test compilation pass with the
  established macOS 26.0 / MLX 26.2 link warning. Direct parity against MLX
  `quantized_matmul` passes K/N `512/64`, `5120/128`, and `17408/32` at
  maximum absolute error **0.03125 / 0.03125 / 0.0234375**; K=256 fails
  closed with the exact unsupported-shape diagnostic.
- The first complete selected-path screen uses the pinned affine-Q5 snapshot,
  native seed-42 sampling, reasoning cap 256, prefill chunk 2,048,
  `MLX_SDPA_BLOCKS=64`, a 256 MiB command-buffer byte budget, 100 operations
  per buffer, and fast synchronization. It completes 128 timed tokens in the
  established `128 / 32 warm / 128 timed` shape at
  **17.823163930 tok/s**, digest `c7d64fcd3a0e3eb2`, last token 24. This is
  **+1.500658165 tok/s / +9.193798%** over the original native affine-Q5
  **16.322505765 tok/s** control and leaves **2.176836070 tok/s** to the
  requested floor. The result remains a one-window screen pending matched
  repetition and served qualification.
- Geometry and runtime screens are retained in PERF-FA122/123. A parallel
  gate/up MLX-stream probe waited indefinitely on a custom-Metal event while
  retaining about 14 GB; exact process cleanup restored the source to the
  sequential MLP owner. The post-cleanup source compiles strictly. The same
  custom-Metal event incident prevented an immediate fresh parity/benchmark
  replay, while a stock MLX arithmetic probe completed in 0.388 seconds.
  PERF-FA124 records the incident and the one-stream rule.

### 2026-09-01 13:29 PDT - PERF-A095/A096/A097/A098 fresh-session queue

- PERF-A095 preserves the selected Q5 arithmetic and geometry while replacing
  four source-level weight reads with three packed ranges. Strict host
  compilation passes; runtime parity remains blocked by the inherited Metal
  event state and carries no throughput claim.
- PERF-A096 reduces the actually traversed quantized-linear bytes per target
  token from **17,615,093,760** to **15,877,570,560**, a
  **1,737,523,200-byte / 9.863831687%** reduction. Pure bandwidth scaling of
  the first PERF-A094 sample yields **19.773598394 tok/s**, leaving
  **0.226401606 tok/s / 1.144969%**; the measured mixed result remains the
  gate because full-Q4/Q5 interpolation predicts less.
- PERF-A097 traces the native standard-MTP producer and consumer. Target
  forward returns the residual before `final_norm_`; `select_token` stores
  that raw row in `last_hidden_`; sampled `spec_refill` and greedy `mtp_draft`
  pass it directly to `mtp_forward`. Upstream `Qwen3_5ForCausalLM` applies its
  final norm before returning hidden states, and standard draft extend asks
  for the post-norm variant. Native embedding-hidden concatenation and
  recurrent `mtp_norm_` already match the published contract.
- Pinned MTPLX v2.9.0 at commit
  `76b52bec6fb22856260355a8f723add67100bb1d` for independent source
  inspection. Its Qwen3.8 artifact revision
  `123db8bcc7101455b00d9aad36c0e760c6e7de02` records post-norm target and
  recurrent hidden states, embedding-before-hidden concat, and local/cache
  positions. The production server requests committed MTP history; native
  `mtp_reset` currently clears history on every refill. Hidden seeding,
  history, and position origin therefore remain separate ablations.
- Downloaded only `mtp.safetensors`, `config.json`, and `mtplx_runtime.json`
  from that exact artifact. The head is **238,934,249 bytes**, SHA-256
  `c58feddc584f37971c72af1f0da95e0099478487009936b3c26ddd88844fab10`,
  and contains 31 tensors: eight affine-Q4/G64 matrices plus seven BF16 norms.
  Every norm is bit-identical to the existing official Q5 sidecar. Its
  published exact-sampling acceptance by depth is
  **0.958762887 / 0.872852234 / 0.759450172**. Native loading needs to accept
  the artifact's `mtp.` key namespace before its fresh-session screen.
- PERF-A098 compiles the candidate Q5 loads offline with Apple Metal
  32023.883. PERF-A095 lowers to ten aligned-one `i8` loads. An equivalent
  `packed_ushort4` plus scalar-`ushort` mapping lowers to five aligned-two
  `i16` loads; K-multiple-512 row strides and ten-byte lane offsets prove
  alignment. It stays behind PERF-A095 in the runtime queue.
- A mandatory analysis-only subagent batch could not start because the single
  available collaboration slot is occupied by the primary agent. Direct
  source inspection supplied the evidence above. No GPU submission occurred;
  the boot remains the 2026-08-31 18:06:52 PDT session and a full machine
  restart is still required.

### 2026-09-01 13:36 PDT - PERF-A097 loader and post-norm seed arms

- Signed `0da5c5a135` makes `Engine::load_mtp` select the common optional
  `mtp.` namespace once, then applies it to every affine matrix and norm in the
  one-layer head. Existing unprefixed Q5 sidecars retain precedence and their
  prior keys. The pinned Q4 head contains all eight required weight/scale/bias
  triplets, every inferred width is exactly four bits at group 64, and all 31
  tensors are accounted for.
- Signed `1b328149c7` adds
  `SGLANG_MLX_NATIVE_MTP_POST_NORM_SEED=1`. The common `mtp_seed_hidden`
  owner applies the target's existing `final_norm_` once before either sampled
  `spec_refill` or greedy `mtp_draft`; the absent setting returns the original
  residual stream. Proposal sampling, recurrence, target verification, and
  exact rejection remain unchanged.
- The final combined source compiled with `clang++ -std=c++20 -O3 -fPIC
  -shared -Wall -Wextra -Werror`, MLX headers marked external, and only the
  established macOS 26.0 / MLX 26.2 linker warning. `git diff --check` passes.
  Runtime acceptance and throughput await the required full restart.

### 2026-09-01 13:45 PDT - PERF-A099 committed-history alignment and Q5 inventory

- Traced the pinned MTPLX 2.9.0 production path through prompt history,
  per-cycle drafting, rollback, and accepted-prefix commit. Prompt entry
  `i` pairs post-norm target hidden `H[i]` with the following token `T[i+1]`.
  A speculative cycle keeps the MTP entry produced for its current pending
  token, rolls the provisional tail back to `cycle_base + 1`, and appends
  accepted tokens using the matching committed target-hidden prefix. The
  sampled correction or bonus remains pending until the next cycle.
- This proves two independently measurable candidates: decode-only committed
  history and prompt-seeded committed history. At exact 131,072 capacity, the
  latter requires **512 MiB** of BF16 MTP K/V storage
  (`4 KV heads * 256 dimensions * K/V * 2 bytes * 131072`) plus a one-layer
  causal history pass. The committed post-norm seed remains the first arm;
  history is admitted only if its measured acceptance premise changes.
- Checked the current published Qwen3.8-27B MLX five-bit inventory. Additional
  uniform artifacts retain affine five-bit/group-64 text weights and either
  add the BF16 vision tower or duplicate the existing text layout. The OptiQ
  artifact reports 5.50 average BPW. No additional download offers a smaller
  uniform-Q5 language stream than the pinned text-only control, while the
  pinned 4.951-bpw mixed target remains the smallest compatible Q5-class arm.
- No Metal work ran. Boot time remains 2026-08-31 18:06:52 PDT; port 30000 and
  matching Qwen, benchmark, MLX, and compiler processes are clear. Runtime
  results still require a full machine restart.

### 2026-09-01 13:52 PDT - PERF-A100 continuous Q5 bitstream lowering

- Refined PERF-A098 with a standalone two-kernel Metal 3.2 source that emits
  the same sixteen unsigned five-bit values from each ten-byte lane pack. The
  control uses PERF-A095's `packed_uchar4`, `packed_uchar4`, and
  `packed_uchar2` byte windows. The candidate loads `packed_ushort4` plus one
  trailing `ushort`, constructs two 32-bit windows and one 16-bit tail, and
  extracts fields at continuous five-bit offsets. Only fields 6 and 12 cross
  a window boundary.
- Apple Metal 32023.883 AIR confirms **10 aligned-one `i8` loads plus 10 byte
  extensions** for the control and **5 aligned-two `i16` loads plus 5 word
  extensions** for the candidate. The selected kernel admits K only in
  multiples of 512, making each Q5 row stride a multiple of 320 bytes; each
  SIMD lane advances 10 bytes. Every 16-bit read is therefore aligned.
- Reproducibility artifacts are
  `/private/tmp/q5_unpack_compare.metal` SHA-256
  `b23bfe1f8918490c4864559c2c94e0abe1141d9383352deb0c5ea94dcfa98c66`,
  AIR SHA-256
  `d28e95f6b32943194070e1e142a3d12852d433fbf6644a8001963f514742b9a3`,
  and LLVM-disassembled AIR SHA-256
  `3bbe495e134222ca2a40f741b95a75e3a766c7e607e171750e1cb216d03ed892`.
  This is compiler evidence only. PERF-A095 keeps first place in the runtime
  queue; PERF-A100 follows only after its fresh-session parity and matched
  benchmark.
- `kern.boottime` remained **2026-08-31 18:06:52 PDT** at 13:52 PDT.
  No Metal workload was submitted and the repository retained exactly the
  three user-owned dirty paths present before this documentation update.

### 2026-09-01 13:56 PDT - PERF-A101 paired-lane Q5 word loads

- Grouped adjacent SIMD lanes into 20-byte Q5 pack pairs. Each even lane reads
  `packed_uint4` plus one trailing `uint`; odd lanes execute no weight load.
  Three `simd_shuffle_up` operations transfer the shared words needed by the
  odd lane, after which each lane reconstructs its own two 32-bit windows and
  16-bit tail. Even lane offsets advance by 20 bytes, and every admitted row
  stride and K-block step is a multiple of 320 bytes, proving four-byte load
  alignment and an in-row 20-byte tail for lanes 30/31.
- Apple Metal 32023.883 AIR retains a masked even-lane branch, five aligned-four
  `i32` loads, and three `air.simd_shuffle_up.u.i32` operations. Relative to
  PERF-A100's five aligned 16-bit loads in all 32 lanes, the pair mapping
  changes dynamic memory operations from **160 to 80 per SIMD group per
  output row** while preserving the exact **320-byte** weight stream. The
  tradeoff is three shuffle operations and lane selection.
- A standalone C++20 parity harness independently packs two lanes at bit
  offsets, reconstructs their pair words, and compares all 32 values. It
  passes all-zero, all-31, every one-hot position/value combination, and one
  million fixed-seed random pairs: **1,001,026 total cases**. Metal runtime
  parity remains required.
- Reproducibility artifacts:
  `/private/tmp/q5_pair_load_compare.metal` SHA-256
  `9afbe674e066fa75e36be952f50a9e15fb2b0e2938e0a492bee63045a271b13b`,
  AIR SHA-256
  `bae87f371b46e519dbdbedfd1de357c354839d26ce74fddd72cea7387f9c65e5`,
  LLVM IR SHA-256
  `2f56f0c158c5cb1df416af10eb42fe34dcb721d036130f253e3dce2854979425`,
  and test source SHA-256
  `944b879bb106404111d007a112d602de021008a8adc1b399b3b015d9e3ecb5f2`.
  No Metal workload ran; the boot remained 2026-08-31 18:06:52 PDT.

### 2026-09-01 14:04 PDT - Current-source Q5 restart matrix prebuilt

- Created four detached clean worktrees from signed `e06fc9c83c` and its
  documentation-only descendant `5e61f10095`: the unchanged PERF-A094
  control, PERF-A095 byte windows, PERF-A100 aligned-ushort continuous stream,
  and PERF-A101 paired-uint stream. Their surrounding engine source is
  identical; only each candidate's Q5 Metal source string differs. The main
  worktree and its user-owned changes remain untouched.
- Compiled each full native dylib and standalone Q5 parity executable with
  C++20/O3, repository warnings as errors, and the active MLX include/library.
  All eight builds pass; the established macOS 26.0 versus MLX 26.2 linker
  warning is the only output. `git diff --check` passes in all three candidate
  worktrees.
- Exact fresh-boot artifacts:

  | Arm | Dylib SHA-256 | Standalone parity SHA-256 |
  |---|---|---|
  | PERF-A094 HEAD control | `b01d2712a3fc810538ff1efc349fc57d2f29380b89ed97d4ba10f9073cbfb242` | `a5e477f819861f91abcef9c1cc367c526235a8fe17940f2f6b32bb5e13dc8c07` |
  | PERF-A095 byte windows | `3638e361b346e15fba6985d155b457f9c43ccfc39a8672231efb776aa935dd27` | `9b30c71ef8e0083220d05ae2f2f32b7a36f3b2a0fca703a88e64f4c86527310b` |
  | PERF-A100 ushort stream | `88c28fc043ffd5a0be6d577b1eccaf2488fe13c2af2abd8bf1b30d519f6ae85f` | `c634186be8ce92dbb29919c475f450ff950238c6b0aed2d22421371edcc5071a` |
  | PERF-A101 paired uint stream | `7ff5004e668115ee53326a140dffcd1407a332968fa8cbdf2824e4af0e61a347` | `90a669dfe6da1bbdab148bc123b2133fc443b1e09d041358d2224ee0ea64e662` |

- After a fresh boot, run the four standalone parity executables first, then
  use their dylibs in a randomized/reversed matched kernel and full-model
  matrix. No candidate receives throughput credit before those measurements.
  No Metal submission ran in this session.

### 2026-09-01 14:15 PDT - PERF-A102 decode-only committed MTP history

- Implemented the PERF-A099 committed-history contract behind
  `SGLANG_MLX_NATIVE_MTP_DECODE_HISTORY=1` in detached worktree
  `/private/tmp/sglang-perf-a102-mtp-history`, based on signed
  `711214b27c517c5cd6012c9e1687498d77b377b3`. The main worktree remains
  Daniel's three user-owned Q4 paths plus PERF-A095's isolated Q5 load hunk.
- At the start of each sampled MTP cycle, the candidate records the logical
  cache length and absolute position. Drafting appends the current pending
  token followed by provisional predecessors. After target verification, the
  candidate restores the MTP boundary to `cycle_base + 1`, then appends each
  accepted draft token with the matching `final_norm(verify_hidden)` prefix.
  The sampled correction or full-acceptance bonus remains pending for the next
  cycle, matching pinned MTPLX 2.9.0 commit
  `76b52bec6fb22856260355a8f723add67100bb1d`.
- A target/MTP absolute-offset mismatch clears the decode-only cache and
  resynchronizes its RoPE origin. This covers target-only reasoning-cap
  fallbacks. Prompt history, MTP position-origin alternatives, greedy MTP,
  DFlash, DSpark, proposal sampling, and target verification remain unchanged.
- Rebuilt the complete dylib with `clang++ -std=c++20 -O3 -fPIC -shared
  -Wall -Wextra -Werror`, MLX as a system include, and the active MLX
  library/rpath. It exits zero with only the established macOS 26.0 / MLX
  26.2 linker warning. `git diff --check` passes. The repository files do not
  conform wholesale to the installed `clang-format`, so that broad check is
  outside the candidate's evidence.
- Exact SHA-256 values are
  `b4f3b222659b5461ed77b737335671e7eedd614411b5228e7ef9ab66febc1a5d`
  for `/private/tmp/libqwen38_mtp_decode_history.dylib`,
  `e14f89e9d260d3590538811ac71ed691cb949e5ce0aa13f614ccae7a4fd0c5ee`
  for the candidate `.cpp`, and
  `b09b8b4fc4ba61152c3a63a29e50f346720a1f9bc881ef3ccc2f2b4105925e5b`
  for the candidate header.
- This is source, contract-trace, and strict-build evidence. Run the published
  Q4 MTP head with post-norm seed off/on first, then add PERF-A102 only when
  its persistent history premise can be paired with per-depth q, acceptance,
  cycle width, and throughput. The boot remains 2026-08-31 18:06:52 PDT and
  no Metal workload ran.

### 2026-09-01 14:20 PDT - PERF-A103 aligned-overlap Q5 word loads

- Built a third continuous-bitstream load geometry from signed
  `92a979bce7fa5d4ab3df77f19abb84cee51944bb` in detached worktree
  `/private/tmp/sglang-perf-a103`. Each lane aligns backward to its adjacent
  lane pair's 20-byte base, loads three `packed_uint3` words, and selects its
  own ten-byte half. All Q5 dispatches require K divisible by 512, making row
  strides and 320-byte loop steps divisible by four. The pair bases advance
  20 bytes, so every load is aligned. Lane 31 reads bytes 308--319 and reaches
  the exact block boundary.
- Per pair, the even lane reads words 0/1/2 and the odd lane reads words 2/3/4.
  Dynamic load operations are therefore **96 per SIMD group/output row**,
  between PERF-A100's 160 aligned 16-bit loads and PERF-A101's 80 aligned
  32-bit loads plus 96 SIMD shuffles. Although each pair requests 24 bytes,
  its address union remains the original five words and exact 20-byte stream.
- Apple Metal 32023.883 lowers the isolated candidate to three
  `load i32 ... align 4` operations per lane, two `llvm.fshl.i32` calls, and
  selects. The isolated PERF-A100 control emits five aligned-two `i16` loads.
  Source/AIR/LLVM SHA-256 values are respectively
  `5e3a670ccc2eb219ca933550f55aacdf84450e862fd0ecc7ee8d53a6cb0673db`,
  `10522229748a51f977b79670e0b0c40d888c879e28f37e2e4c73c617129ef38a`,
  and `7a5417cec6b357a0567a9b2ef8adfff652e2c4ccfa58595e5f6d3cc70f86539c`.
- A strict C++20 reference packs the complete 512-value/320-byte block, reads
  every lane through the aligned-overlap mapping, and compares all values. It
  passes zero, all-31, every one-hot position/value combination, and one
  million fixed-seed random blocks: **1,016,386 cases**. Source/executable
  SHA-256 values are
  `2f2c655032a2544df96c849d32ee5aa3d55856d24341957bc119114166287b4c`
  and `319d86f09d22517a79cb22e0e0b70cb061a498176785fe03016461b2baaaba7f`.
- Strict C++20/O3 warning-as-error builds pass for the complete native dylib
  and standalone Metal parity executable. Their SHA-256 values are
  `71defea337930e17a5102119198d976349575e7185e264a1d34d87c7781863ef`
  and `935ab16e642b0a0674ff4c8c73e89d2617ae8560c0b28b14ef952ac63429b9d6`;
  the candidate engine source is
  `cf3fdadc71dee602301b61083f7c6aedfccc1d1c56d38322af129de86e2f55aa`.
  `git diff --check` passes. The established macOS 26.0 / MLX 26.2 linker
  warning is the only build output.
- Keep the executable idle until a fresh boot. Then run parity before a
  randomized/reversed A100/A101/A103 microbenchmark and full uniform-Q5
  window. No throughput is attributed from compiler evidence. The boot still
  reports 2026-08-31 18:06:52 PDT and no Metal workload ran.

### 2026-09-01 14:25 PDT - PERF-A104 four-way FP32 dot screen

- Compiled an explicit Metal 3.2 arithmetic comparison using the same sixteen
  five-bit values. The control spells the selected sequential FP32 FMA chain
  exactly; AIR retains **16** `air.fma.f32` calls. Four `float4` groups in the
  candidate lower to **4** `air.dot.v4f32` calls plus **3** FP32 additions.
  This establishes a distinct compiler path while leaving the final hardware
  cost for runtime measurement.
- Built the full candidate atop PERF-A103's aligned-overlap loads in detached
  worktree `/private/tmp/sglang-perf-a104` at signed
  `6e181e68d2096f85a990c311c5b1ef4b83d0b9d1`. It changes only the embedded Q5
  batch-one Metal source. The complete dylib and repository standalone parity
  executable compile under C++20/O3 with warnings as errors; `git diff
  --check` passes. The established macOS 26.0 / MLX 26.2 linker warning is the
  only build output.
- The grouped dot changes FP32 accumulation order. It therefore retains no
  correctness, digest, behavior, or throughput credit from AIR inspection.
  Run A103/A104 numeric parity first, then require the selected target digest,
  exact sampled semantics, and randomized/reversed timing before admission.
- Exact SHA-256 values are
  `1d73b5d47ef0c54b388164b25a33d43d299b2a30cb1d669d337fa358ef122385`
  for `/private/tmp/q5_dot_compare.metal`,
  `186653233a00981b0c60d79dae130d6133974fc96c7e5d3377d26516adafbab9`
  for its AIR,
  `fa63bfcf4e63397ffdff7eea0019dafb70b021c9e187e5af42c79249e3be2668`
  for its LLVM IR,
  `d9c79dc464f3148d71d940569d5a549dc81ac70e844f09219722c5a282115fd7`
  for the candidate engine,
  `a4dbb21000af631505938883c35d6561987a54530694b9ed936cbd216a4b964b`
  for `/private/tmp/libqwen38_affine_q5_qmv_overlap_dot4.dylib`, and
  `a5e9ac1192b6ccc9a3f73b228dd089d286e1de8f9df56c73e0b5e5b90dd69e83`
  for `/private/tmp/test_qwen38_affine_q5_batch_one_qmv_overlap_dot4`.
- Both runtime artifacts remain idle. The machine still reports boot time
  2026-08-31 18:06:52 PDT and no Metal submission ran.

### 2026-09-01 14:33 PDT - PERF-A105 deterministic affine-Q5 QMV matrix

- Added `benchmark/mac/bench_qwen38_affine_q5_qmv.cpp`, a C++20-only direct
  harness for the production-reachable `affine_q5_qmv_batch_one` owner. It
  accepts `K N WARMUP ITERATIONS`, enforces the kernel's K-multiple-512 and
  N-multiple-16 contract, and generates fixed Q5 packed weights, BF16/G64
  scale and bias, and BF16 activations without a checkpoint dependency.
- Each warm and timed iteration evaluates and synchronizes exactly one QMV.
  The output records mean milliseconds, effective streamed bandwidth from
  packed weights plus scale/bias bytes, the first BF16 value, and an FNV-1a
  digest over every BF16 output byte. That shared digest makes load-layout
  equivalence observable across PERF-A094/A095/A100/A101/A103; PERF-A104's
  changed FP32 grouping remains subject to numeric parity and full-model
  semantics even when its digest differs.
- Strict C++20/O3 warning-as-error builds pass against every staged candidate
  worktree. `clang-format --dry-run --Werror` and `git diff --check` pass. The
  only compiler output is the established macOS 26.0 / MLX 26.2 linker
  warning. Harness source SHA-256 is
  `6f7e336ea83c6eeec0ffc36b5edb509f6b8edfa10f4c2a5b2556be36f38d6da3`.
- Final benchmark executables and SHA-256 values are A094
  `dbd52a85fc39eb495b655e93e4df9cbda55d274033b5400537b3c9d8fbb39ebc`,
  A095 `7e93fb10dcf5bf3780b23845d13bca250e5077d641a2fa69d3c5f4446d52cc8e`,
  A100 `021b06087296a0ea6efc6a88099d6a5879a3f8de97eed2fb2522255176fcff17`,
  A101 `1e596108a080c6fe2be4f06f1b04049c2949cb6d6d7b1c6e9b4b5475b10b1f27`,
  A103 `dc9fd39a4a59aeb1e8c3673425ddc591bf82f0d8f6a156c525dfad874bf7c2e9`,
  and A104
  `2c88876c9a49eb0f7aec01b9ff4bf70d6975c0657c1a572b5cf1213a3995c223`.
- Runtime remains intentionally pending. The boot is still 2026-08-31
  18:06:52 PDT. After restart, run parity first, then randomized and reversed
  matrices at `5120x17408`, `17408x5120`, `6144x5120`, and `5120x1024`
  before full uniform-Q5 and mixed-Q5 windows. Compiler evidence carries no
  throughput claim.

### 2026-09-01 14:40 PDT - PERF-A106 aligned 64-bit BF16 input reads

- Traced the selected Q5 QMV's live activation owner. Every SIMD lane reads
  sixteen contiguous BF16 values per 512-value K block and reuses them across
  four output rows. The scalar source therefore performs sixteen dynamic BF16
  loads and conversions per lane/block; all four SIMD groups read the same
  activation block for their separate output cohorts.
- An isolated Metal 3.2 comparison replaces those reads with four aligned
  64-bit words, reinterprets each word as `bfloat4`, and uses four vector
  BF16-to-FP32 conversions. AIR contains four aligned-eight `<4 x bfloat>`
  loads and four `air.convert.f.v4f32.f.v4bf16` calls. The candidate spells
  the original sequential input sum and sixteen FP32 FMAs in the same order.
  Lane and K-block offsets advance 32 and 1,024 bytes, so an eight-byte-aligned
  input base remains aligned. Live parity gates that base-alignment premise.
- Built this single-variable arm on the selected PERF-A094 weight path in
  detached worktree `/private/tmp/sglang-perf-a106` at signed
  `83c2731a3d8c908dc21c9fd6554e423f10b34c89`. Strict C++20/O3
  warning-as-error builds pass for the complete dylib, standalone Metal parity
  executable, and PERF-A105 benchmark. `git diff --check` passes. Only the
  established macOS 26.0 / MLX 26.2 linker warning appears.
- Candidate engine, dylib, parity executable, and benchmark SHA-256 values are
  `a8ef76d3a338fd1451dda2d1179a0e80aac1f850b842f539624b0f64e65d60f6`,
  `868223066aa54e250c26d97c64782931c311cf73411d5efe19e7c9d2b1669bbe`,
  `ee67d521efacf7f19ce129c670f386e56784ae7cb515466adc183e55972dbbfc`,
  and `27a8f5daf2efc86e5ac9df1a28a3515e8f0924903cf25ead36815158d0706c78`.
  The final multi-form comparison source/AIR/LLVM hashes are
  `6a6dffb1c404d5dd8839c0f0973ad909ebccbc00106b99cb05c5e9e4ed789f23`,
  `a324bcfb0cbc5d4bceb81bd61c30197a43d153aa625468d37ef551e951b5bc24`,
  and `530be58b11ba128d87a8b89bef19531855886b65e505ab70221a38cff565f185`.
- No runtime artifact ran; the boot remains 2026-08-31 18:06:52 PDT. After a
  fresh boot, require standalone parity and exact A094 digest equality, then
  add A106 to randomized/reversed PERF-A105 timing before any combination with
  a weight-load winner.

### 2026-09-01 14:42 PDT - PERF-A107 aligned 128-bit BF16 input reads

- Tightened only A106's activation transaction width. Two `uint4` reads cover
  the same 32-byte per-lane activation fragment; their four 64-bit halves are
  reinterpreted as `bfloat4` and converted exactly as in A106. AIR lowers this
  to two aligned-sixteen `<2 x i64>` loads plus four vector BF16-to-FP32
  conversions. The input sum, weight loads, unpack, FP32 FMA order, reduction,
  and output conversion remain unchanged from A094.
- The form requires a sixteen-byte-aligned input base. Lane starts advance 32
  bytes and K blocks advance 1,024 bytes, so those offsets preserve the
  premise. Standalone parity on live MLX allocations is the first runtime gate;
  no source promotion can rely on compiler evidence alone.
- Built in detached worktree `/private/tmp/sglang-perf-a107` at signed
  `414198a15c2cb92814f422d5173171042c8d8d1d`. The complete dylib,
  standalone parity executable, and PERF-A105 benchmark compile under
  C++20/O3 with warnings as errors. `git diff --check` passes. The established
  macOS 26.0 / MLX 26.2 linker warning is the only output.
- Candidate engine, dylib, parity executable, and benchmark SHA-256 values are
  `f7fd5eca635598d33562434265b891dcd7cf05fa9b1d581fbefc107efdd70c43`,
  `9ab436a8eb8d11d98b8d500dd614b44db32d99847be27d1bc84d5d2b079f133a`,
  `66dc45ec16350f0e63e92c9b84238007035e62bc1cefee5aca156d75ed7d76f8`,
  and `0530a4a1d36b470c7319f92fd3b9e2cfe7d355a25ecb74c44d5f91318330ed8f`.
- No runtime artifact ran; boot time remains 2026-08-31 18:06:52 PDT. After
  restart, parity and exact A094 digest equality precede randomized/reversed
  A094/A106/A107 timing. The better activation-load width then combines with
  the independently selected weight-load arm.

### 2026-09-01 14:45 PDT - PERF-A108 four-lane scale/bias broadcast

- Traced the group-64 parameter mapping in the selected kernel. Each lane owns
  sixteen consecutive K values, so adjacent groups of four lanes use the same
  BF16 scale and bias. The control issues 32 scale and 32 bias
  loads/conversions per SIMD/output row/K block even though only eight values
  of each are distinct.
- In an isolated Metal 3.2 comparison, only lanes `0,4,...,28` load their
  pair. One `simd_shuffle(float2, leader)` broadcasts it within each four-lane
  group. AIR retains the leader branch, two BF16 loads/conversions, and one
  `air.simd_shuffle.v2f32` call. Dynamic scale/bias loads and conversions fall
  fourfold to 8+8 per SIMD/output row; each lane then executes the unchanged
  `scale * quantized_dot + bias * input_sum` expression.
- Built the isolated source arm in detached worktree
  `/private/tmp/sglang-perf-a108` at signed
  `b1eeaf9f11d4d2e1a526ae0d2a3ff7e9549bbf04`. Strict C++20/O3
  warning-as-error builds pass for the full dylib, standalone parity
  executable, and PERF-A105 benchmark. Candidate `git diff --check` passes;
  the established macOS 26.0 / MLX 26.2 linker warning is the only output.
- Comparison source/AIR/LLVM SHA-256 values are
  `0eb75b04eab73d8137f7e3f0ac6c1f4151d6f02341f568adececb7f492c76859`,
  `7d3029e7bd092bff74592eeef53c90222a54067ace0fa4a4c7316aeecdfa6102`,
  and `fd6a87bfab6162497063088d12083a1016399e90815b7e42025b828e78d49316`.
  Candidate engine, dylib, parity executable, and benchmark hashes are
  `7bf44461fd03d4e93bfd5468e24106f710a7f32ba4fa7eb1b7d6407d098c7dc7`,
  `293b4a97b0a7d678ce5d4d223ec511f148b03e240a3b7cd497ca3f2b12f085d5`,
  `a0d6390f6c1629f47f245cf35bfd853e5e69e65495cde8563ce3f05b349d5ac5`,
  and `779665728148db6aeff9af071e8d06bf1ddfcfe6874fbfc162a58f957daf2a1a`.
- No runtime artifact ran; the boot remains 2026-08-31 18:06:52 PDT. Exact
  A094 digest equality and matched A094/A108 timing decide whether explicit
  broadcast beats the hardware's ordinary duplicate-address coalescing. Only
  a measured win proceeds to combination with input/weight load candidates.

### 2026-09-01 15:17 PDT - PERF-A103/A104/A106/A108 fresh-session resolution

- Change: rebuilt the detached Q5 load, vector-input, dot-product, parameter-
  broadcast, and geometry candidates from the signed post-rewrite source line.
  No candidate source entered `main`.
- Correctness evidence: A103, A104, A106, and A108 pass K/N `512/64`,
  `5120/128`, and `17408/32` parity with maximum errors
  **0.03125 / 0.03125 / 0.0234375**. Their complete gate/up, down,
  attention-output, and value-projection digests match A094:
  `245b954f045f18cc`, `d05378cc8066dc41`, `46a6c240b685fce9`, and
  `2e6fc0972644b8d2`.
- Benchmark evidence: process-isolated order/reverse `100 / 1000` mean
  latencies in milliseconds were:
  - A103 versus A094: gate/up **0.48196/0.43744 vs 0.47755/0.43195**;
    down **0.44199/0.43844 vs 0.43641/0.42623**; attention-output
    **0.33037/0.33116 vs 0.32052/0.32444**; value
    **0.30889/0.30264 vs 0.29867/0.30034**.
  - A106 versus A094: gate/up **0.47823/0.42967 vs 0.48474/0.42895**;
    down **0.43564/0.44247 vs 0.44257/0.43328**; attention-output
    **0.33184/0.33994 vs 0.33382/0.33134**; value
    **0.29964/0.29922 vs 0.30004/0.30191**.
  - A104 versus A094: gate/up **0.48029/0.43486 vs 0.48825/0.43295**;
    down **0.44167/0.43766 vs 0.43445/0.43707**; attention-output
    **0.32671/0.32127 vs 0.32530/0.32782**; value
    **0.29846/0.29838 vs 0.29339/0.30381**.
  - A108 versus A094: gate/up **0.53967/0.49802 vs 0.48134/0.42829**;
    down **0.50928/0.51664 vs 0.42254/0.42942**; attention-output
    **0.36516/0.36891 vs 0.32969/0.32940**; value
    **0.30812/0.31230 vs 0.29600/0.29582**.
- A shape-specific `2/4/2` and `8/4/2` geometry check converged with A094 in
  order/reverse timing and supplied no durable five-percent route. The earlier
  complete full-model geometry sweep already rejects global selection.
- Decision: close A103, A104, A106, and A108 under the current compiler/GPU
  topology. A107 remains a distinct unmeasured 128-bit activation-load arm.
  PERF-FA125 records the shared reopening condition.

### 2026-09-01 15:18 PDT - PERF-A096 mixed-target Q4 and MTP ablations

- Change: varied only the existing Q4 batch-one QMV switch, Q5 batch-one QMV
  switch, published Q4 MTP head, and post-norm seed around the pinned mixed
  target. No source changed.
- Benchmark evidence: exact direct `128 / 32 warm / 128 timed` results were
  generic **18.121698566**, Q5 QMV only **19.032187257**, Q4 QMV only
  **17.221199280**, and both QMV paths **17.987099210 tok/s**. Q4 custom
  execution is therefore excluded from the selected mixed lane.
- Published Q4 MTP block-three sampled smokes reached **9.122242179 tok/s**
  at width **1.882352941**; post-norm seed reached **9.507655940** at width
  **1.764705882**; Q4 QMV with the original seed reached **11.341033334** at
  width **2.285714286**. All completed exact requested token counts.
- Decision: retain the compatibility loader and opt-in seed correction. Close
  the unchanged Q4 QMV and published-Q4-MTP compositions through
  PERF-FA126/127.

### 2026-09-01 15:23 PDT - PERF-A109 dense recurrent b/a fusion

- Change: for the mixed artifact's 48 dense BF16 recurrent layers, materialized
  each `[48,5120]` b/a weight pair as one `[96,5120]` tensor during load,
  replaced two dense matmuls with one, split its result at row 48, and released
  the original array references. Quantized checkpoints retain their existing
  separate path.
- Build evidence: the detached candidate dylib builds under C++20/O3 with
  `-Wall -Wextra -Werror`; `git diff --check` passes. Candidate dylib SHA-256
  is `5980d553005547de38564c640076df414baec1e717c1b1d123143439a10329d3`.
- Benchmark contract: pinned mixed revision `596b8067...8340`, A094 Q5 QMV,
  seed 42, `MLX_SDPA_BLOCKS=64`, 256 MiB/100-op command-buffer limits,
  `MLX_METAL_FAST_SYNCH=1`, and exact `128 / 32 warm / 128 timed`. Every
  process loaded the model independently.
- First balanced window raw tok/s:
  - control: **19.008018889, 19.018764842, 18.967816850, 18.975859871,
    18.968574033**, mean **18.987806897**;
  - fusion: **19.053691587, 18.981565248, 18.988260834, 19.011803229,
    19.007743799**, mean **19.008612939**.
- Independent balanced window raw tok/s:
  - control: **18.998121449, 19.026949772, 18.971042050, 18.988260834,
    19.010428720**, mean **18.998960565**;
  - fusion: **18.982331387, 18.966904100, 18.995294950, 18.931278663,
    19.013343509**, mean **18.977830522**.
- Aggregate control is **18.993383731 tok/s** and fusion is
  **18.993221731 tok/s**, a **-0.000162000 tok/s / -0.000853%** movement.
  Every run produced digest `d0193f6d413b68c1` and last token `11406`.
- Decision: reject and keep the candidate outside `main`. The second window
  cancels the first window's small apparent gain; the dispatch reduction has
  no measurable end-to-end value on this dense shape. PERF-FA128 records the
  closure.

### 2026-09-01 15:39 PDT - PERF-A100 aligned Q5 word loads

- Change: replaced ten scalarized byte loads for each lane's two Q5 packs with
  one aligned `packed_ushort4` plus one aligned trailing `ushort`. Two 32-bit
  windows and one 16-bit tail feed the original sixteen sequential FP32 FMAs;
  only five-bit fields 6 and 12 cross a window boundary. The guard already
  requires K divisible by 512, so every ten-byte lane segment and 320-byte
  block step is two-byte aligned.
- Correctness evidence: standalone K/N `512/64`, `5120/128`, and `17408/32`
  parity passes at maximum absolute error
  **0.03125 / 0.03125 / 0.0234375**. Gate/up, down, attention-output, and
  value-projection digests match A094 exactly:
  `245b954f045f18cc`, `d05378cc8066dc41`, `46a6c240b685fce9`, and
  `2e6fc0972644b8d2`. Every full-model run preserves token digest
  `d0193f6d413b68c1` and last token 11406.
- Balanced production-shape `100 / 1000` microbenchmarks keep gate/up and down
  effectively flat while improving the compact shapes. Order/reverse means in
  milliseconds were A094/A100 gate/up **0.477073/0.478874** and
  **0.428166/0.426401**; down **0.428398/0.429761** and
  **0.434184/0.431101**; attention-output **0.333244/0.321322** and
  **0.335367/0.333845**; value **0.305514/0.300372** and
  **0.309724/0.304864**.
- Full-model contract: pinned mixed revision `596b8067...8340`, native
  sampling seed 42, A094/A100 Q5 QMV, Q4 QMV unset, exact
  `128 / 32 warm / 128 timed`, and the selected command-buffer controls.
- First balanced five-versus-five window:
  - A094 **18.979970071, 18.964319729, 19.010746827, 19.036241095,
    19.028913416**, mean **19.004038228 tok/s**;
  - A100 **19.110513585, 19.128907179, 19.081414615, 19.116677570,
    19.173775190**, mean **19.122257628 tok/s**;
  - gain **+0.118219400 tok/s / +0.622075%**.
- A first independent attempt became externally contended after two clean
  pairs. The discarded tail was control **15.322072049**, A100
  **19.164288188 / 18.566956374**, control **14.285164044 / 14.438592893**,
  and A100 **14.382498251 tok/s**. Immediate diagnostics found Spotlight
  `mds_stores`/`mdworker_shared` and `fileproviderd` processing the newly
  materialized worktree, with no thermal warning, model process, or listener.
  This tail carries no candidate attribution.
- After the host returned to its ordinary idle state, the independent clean
  window was:
  - A094 **18.963286862, 19.023466393, 19.047192232, 19.026645144,
    19.045515986**, mean **19.021221323 tok/s**;
  - A100 **19.087183885, 19.161381911, 19.185076409, 19.145774023,
    19.158371971**, mean **19.147557640 tok/s**;
  - gain **+0.126336316 tok/s / +0.664186%**.
- Aggregate clean means are A094 **19.012629776** and A100
  **19.134907634 tok/s**, a **+0.122277858 / +0.643140%** gain. Against the
  generic mixed target, A100 is **+1.013209068 tok/s / +5.591137%**.
- Build evidence: candidate full dylib, standalone parity, and deterministic
  microbenchmark compile under C++20/O3 with `-Wall -Wextra -Werror`.
  Their SHA-256 values are
  `ce1229795f806a0ffd6bc66225e06e77a14f23baf2e13405b179723d36e07b36`,
  `ce589315db40dd79508a0a2fd4bddf04c8948cbb92c71e37695bdb0141168119`,
  and `45e6e6f2116d66139bc77a2bd8f51fc18af679426b5cc9caaf670f864efe792e`.
  Candidate `git diff --check` passes.
- Decision: promote A100. The change preserves arithmetic order and output,
  repeats across independent process-isolated windows, and lowers the direct
  gap to **0.865092366 tok/s / 4.521017%**.

### 2026-09-01 15:46 PDT - PERF-A101/A107 remaining load forms

- Change: evaluated the two remaining prebuilt load mechanisms against their
  direct parent controls. A101 lets each even lane read an adjacent 20-byte
  Q5 pair through five aligned 32-bit words and uses three shuffles for its odd
  neighbor. A107 replaces A106's four 64-bit activation reads with two 128-bit
  reads while retaining four `bfloat4` conversions and sequential sums.
- Correctness evidence: both strict C++20/O3 dylib, standalone parity, and
  microbenchmark builds pass with warnings as errors. Both return maximum
  parity errors **0.03125 / 0.03125 / 0.0234375**, and all four production-
  shape digests match their controls.
- A101 versus selected A100 order/reverse milliseconds:
  - gate/up **0.505742/0.469187 vs 0.483396/0.433035**;
  - down **0.467180/0.464544 vs 0.431774/0.436414**;
  - attention-output **0.345742/0.353793 vs 0.320258/0.330694**;
  - value **0.309436/0.304927 vs 0.301531/0.303991**.
- A107 versus A094 order/reverse milliseconds:
  - gate/up **0.486450/0.438125 vs 0.475459/0.434990**;
  - down **0.444962/0.441823 vs 0.426812/0.435401**;
  - attention-output **0.333475/0.337715 vs 0.338058/0.330822**;
  - value **0.303557/0.306522 vs 0.299531/0.301201**.
- Decision: reject both before a full-model launch. A101's halved load-
  instruction count is outweighed by masked leadership and shuffle cost;
  A107's wider transactions do not improve the high-byte projections. A100
  remains selected. PERF-FA129 records the reopening condition.

### 2026-09-01 16:21 PDT - PERF-A110/A111 mixed shader attribution and exact Q4 QMV

- A selected-A100 Metal System Trace with GPU shader counters completed at
  **19.171907764 tok/s** under instrumentation with digest
  `d0193f6d413b68c1`. A strict C++20 analyzer maps **48,156 / 48,831**
  target-process shader PCs, with zero ambiguous mappings. Stock
  `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0` owns **43.546%**; custom Q5
  owns **48.613%** across K=17,408/5,120/6,144; all QMV owns **92.159%**.
- PERF-A111 starts from official MLX tag `v0.32.2`, commit
  `1f8e74e3f12f31365464a6867c6579f0e9b29d85`. It retains MLX's Q4 load and
  dot helper expression structure, changes only the batch-one output cohort
  from two to four SIMD groups, specializes K, and remains behind
  `SGLANG_MLX_NATIVE_Q4_BATCH_ONE_QMV`.
- Strict full-dylib and focused-test builds pass. Direct parity against MLX
  is bit-exact at K/N **512/64**, **5120/128**, and **5120/17408**; the
  unsupported K=256 shape fails closed. The selected dylib/test/benchmark
  SHA-256 values are `9b4452d6...b01e`, `2101c61d...9609`, and
  `c105e7d3...8ef`.
- A hand-inlined precursor reduced gate/up QMV from paired stock
  **0.464814646 ms** to **0.434910417 ms**, and one full-model screen reached
  **19.413641543 tok/s**. It produced up to **0.0078125** BF16 error and
  changed the seeded target digest, so PERF-FA130 rejects that arithmetic
  form. Restoring MLX's helper/expression structure makes all measured output
  bit-exact; its isolated gate/up timing is flat, while halving threadgroup
  count still wins end to end.
- First balanced five-versus-five full-model window:
  - A100-only control **19.129950306, 19.114006087, 19.107881382,
    19.132208843, 19.032309413**, mean **19.103271206 tok/s**;
  - A100+A111 **19.311764663, 19.201559289, 19.158388459,
    19.223084040, 19.258195016**, mean **19.230598293 tok/s**;
  - gain **+0.127327087 / +0.666520%**.
  The first listed control used the precursor dylib with its Q4 switch unset;
  the reachable A100 path and canonical output are identical. All remaining
  controls and every candidate use the selected exact-source dylib.
- Independent reversed window:
  - A100-only control **19.140532994, 19.126232751, 19.097367169,
    19.153648409, 19.105223523**, mean **19.124600969 tok/s**;
  - A100+A111 **19.231720441, 19.263637787, 19.256509432,
    19.256183163, 19.258771030**, mean **19.253364371 tok/s**;
  - gain **+0.128763401 / +0.673287%**.
- All 20 qualified outputs reproduce digest `d0193f6d413b68c1` and last token
  11406. Aggregate control/candidate means are **19.113936088 / 19.241981332
  tok/s**, a **+0.128045244 / +0.669905%** win. Promote A111 beside A100.
  The direct gap is now **0.758018668 tok/s / 3.939400%**; exact 131K serving
  and Codex `xhigh` qualification remain pending until the direct floor is
  cleared with margin.

### 2026-09-01 16:45 PDT - PERF-A112 remaining Q4 geometry and load screens

- Scope: signed A111 plus A100, pinned mixed revision `596b8067...8340`,
  native sampling seed 42, and the exact `128 / 32 warm / 128 timed` direct
  contract. Every completed qualified output used the canonical digest
  `d0193f6d413b68c1` and last token 11406.
- Exact geometry screens:
  - eight SIMD groups by four rows is flat across reversed full-model pairs:
    control **19.229939742** and candidate **19.230535491 tok/s**;
  - four SIMD groups by eight rows reaches **18.840786855** beside selected
    **19.240688283 tok/s**;
  - two SIMD groups by eight rows reaches **18.851060160** beside selected
    **19.217529511 tok/s**.
- Replacing four scalar `ushort` reads with one `packed_ushort4` remains
  bit-exact at all three parity shapes. Gate/up falls from **0.464261542** to
  **0.452577875 ms**, while the balanced full-model means are control
  **19.222120223** and candidate **19.232188501 tok/s**, a
  **+0.010068277 / +0.052379%** neutral movement.
- Four packs per lane reaches paired gate/up means **0.463281000 ->
  0.454654105 ms**. It introduces one `0.000488281` BF16 mismatch at
  `5120x17408`; the full model falls to **18.928908455 tok/s**, digest
  `92ae190a68ee376b`, and last token 8.
- Keeping the selected helper while materializing each `ushort` local is
  bit-exact and regresses paired gate/up means **0.454020313 ->
  0.464569500 ms**. Forcing full unrolling remains bit-exact and regresses the
  same projection from **0.468931459** to **1.799243958 ms**.
- Decision: retain A111 unchanged. PERF-FA131 closes these source forms.

### 2026-09-01 16:52 PDT - PERF-A113 duplicate scalar evaluation screen

- Production reachability: `Engine::emit_scheduled()` is the target-only
  scalar return path for prefill and every steady-state decode. The selected
  engine schedules one following token before emitting the token queued by
  the prior call. Installed MLX 0.32.2 `array::item<T>()` calls `eval()` before
  reading data, so the preceding free `eval(pending_tok_)` performs the same
  completion twice at the host API layer.
- Removing only the explicit free `eval()` preserves the two-token pipeline,
  state/history ownership, canonical digest, and last token. Two preliminary
  reversed pairs measure control **19.195765829** and candidate
  **19.219406606 tok/s**. The first four clean interleaved qualification pairs
  measure control **19.214768638** and candidate **19.235724277 tok/s**, a
  **+0.020955640 / +0.109065%** movement.
- A fifth candidate fell to **16.599721750 tok/s**, followed by a replacement
  at **10.995856101**. Immediate diagnostics found newly spawned
  `mdworker_shared` cohorts plus `mds_stores`, `fileproviderd`, and
  `CGPDFService` using substantial CPU and storage bandwidth. Memory remained
  95% free, pages throttled remained zero, AC/high-power policy remained
  selected, and `pmset -g therm` reported no thermal or performance warning.
- Decision: exclude the two externally contended samples and keep A113 in
  qualification. Resume only after the background indexing wave returns to
  the ordinary idle state; require a complete five-pair window and an
  independent reversed window before promotion.

### 2026-09-01 17:06 PDT - PERF-A114 fused Q4 gate/up/SwiGLU screen

- Production reachability: all 64 target MLP gate and 64 up projections in
  the mixed 4.951-bpw checkpoint are affine-Q4/G64. Batch-one decode reaches
  `Engine::mlp()` on every target layer. The candidate replaces the two A111
  QMV submissions and the BF16 sigmoid/multiply chain with one 8-SIMD by
  four-paired-row Metal kernel behind
  `SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU`.
- Strict C++20/O3 warnings-as-errors builds pass. Synthetic focused parity is
  bit-exact for the selected Q4 QMV at K/N **512/64, 5120/128, and
  5120/17408**, and for the complete fused chain at **512/64** and
  **5120/17408**.
- Production-shape `1000 / 5000` microbenchmarks measure separate
  **0.596468750, 0.600999883 ms** (mean **0.598734317**) and fused 8x4
  **0.555415667, 0.560186117 ms** (mean **0.557800892**), a
  **0.040933425 ms / 6.837%** reduction with one shared digest. Direct fused
  geometry screens select 8x4 over 8x2 by **1.769%**; 16x2 and 8x8 are
  slower.
- One full mixed-target screen under continuing Spotlight/FileProvider
  activity reaches **19.406869588 tok/s**. It changes the canonical digest
  and last token from `d0193f6d413b68c1` / 11406 to
  `f2a59800c8d89f75` / 19.
- Decision: reject the current fused implementation at the real-model
  correctness gate. Preserve the 8x4 artifact and candidate worktree for a
  first-diverging-layer diagnostic. Throughput remains diagnostic until a
  real-weight/real-hidden fixture proves gate, up, sigmoid, and both BF16
  multiply boundaries independently exact. The selected timing stays
  **19.241981332 tok/s**.

### 2026-09-01 17:42 PDT - PERF-A114 precise-exp correctness repair

- A temporary opt-in real-weight/real-hidden trace compares the fused gate,
  up, BF16 sigmoid, BF16 SiLU, and BF16 final product against the selected
  separate MLX route at every target layer. Layers 0--61 match. At layer 62,
  element 36, gate and up remain exact but `metal::exp` produces sigmoid
  BF16 `0x3a8c` (`0.00106811523`) instead of `0x3a8b`
  (`0.00106048584`). SiLU then differs as `0xbbf0` versus `0xbbee`, and the
  output as `0x3c8f` versus `0x3c8e`.
- Volatile BF16 locals and explicit BF16/`ushort` round trips leave the same
  mismatch. MLX v0.32.2 precompiled kernels use `-fno-fast-math`, custom
  kernels default to safe math, and the existing exact recurrent SiLU kernel
  already uses `metal::precise::exp`. Changing only this fused helper to
  `metal::precise::exp` makes all five boundaries exact for all 64 traced
  layers.
- The trace plumbing and unsuccessful experiments were removed. The clean
  candidate remains a 302-line, three-file diff over signed `22408c50c4`;
  its engine/header/test blobs are `a22c844a1c`, `512335f1ae`, and
  `b42ddac6a2`. Strict C++20/O3 warnings-as-errors builds and focused Q4/fused
  parity pass with zero mismatches.
- Precise production-shape `1000 / 5000` microbenchmarks measure separate
  **0.603388567, 0.605122308 ms** (mean **0.604255438**) and fused
  **0.565522233, 0.562880642 ms** (mean **0.564201438**), a
  **0.040054000 ms / 6.628659%** reduction. Every arm retains digest
  `8a9031349585365a` and first value `0.875`.
- One full-model correctness screen during the continuing
  Spotlight/FileProvider indexing wave reaches **19.356657788 tok/s** and
  restores canonical digest `d0193f6d413b68c1`, last token 11406. This is
  correctness evidence only. Retain A100+A111 at **19.241981332 tok/s** until
  a balanced five-pair window and an independent reversed five-pair window
  pass on an ordinary-idle host.

### 2026-09-01 17:56 PDT - PERF-A114 paired qualification and regression guard

- Forward control/fused pairs are:
  - **19.324730384 / 19.238353460**;
  - **19.251111098 / 19.252900231**;
  - **19.192408442 / 19.305214020**;
  - **19.209043376 / 19.234817304**;
  - **19.202768320 / 19.288896860 tok/s**.
  Means are **19.236012324 / 19.264036375**, a
  **+0.028024051 / +0.145685%** candidate movement.
- After a cooldown, the independent reversed fused/control pairs are:
  - **19.289244945 / 19.245002515**;
  - **19.224734784 / 19.212625936**;
  - **19.296968595 / 19.183495840**;
  - **19.256385706 / 19.211986477**;
  - **19.296563258 / 19.254030660 tok/s**.
  Control/fused means are **19.221428286 / 19.272779458**, a
  **+0.051351172 / +0.267156%** candidate movement.
- All 20 requests preserve 128 timed tokens, digest `d0193f6d413b68c1`, and
  last token 11406. Aggregate control/fused means are
  **19.228720305 / 19.268407916 tok/s**, a
  **+0.039687612 / +0.206398%** qualified win. The remaining direct gap is
  **0.731592084 tok/s / 3.796848%**.
- A new C++ boundary fixture constructs gate BF16 `-6.84375` through valid
  Q4 scale/bias arithmetic. The precise artifact matches MLX at
  `-0.00726318` in all 32 rows; the preserved fast-exp artifact returns
  `-0.00732422` and fails all 32. This regression would have caught the
  original real-model defect.
- Decision: promote precise-exp A114 behind its explicit switch and retain
  A100+A111 as the disabled control. Exact 131K serving and Codex `xhigh`
  remain gated on clearing 20 tok/s with margin.

### 2026-09-01 18:10 PDT - PERF-A113 duplicate token evaluation promotion

- On top of selected A114, the candidate removes only `eval(pending_tok_)`
  immediately before `pending_tok_.item<int32_t>()`; installed MLX 0.32.2's
  `array::item<T>()` invokes `array::eval()` itself.
- Forward A114-control/A113-candidate pairs are
  **19.347148243/19.329479119**, **19.242627231/19.215573021**,
  **19.218382143/19.307212105**, **19.306167755/19.295138296**, and
  **19.240385449/19.240877485 tok/s**. Means are
  **19.270942164 / 19.277656005**, a
  **+0.006713841 / +0.034839%** movement.
- After cooldown, reversed candidate/control pairs are
  **19.317661340/19.264135243**, **19.320699549/19.331620301**,
  **19.256770404/19.268522238**, **19.296717561/19.304819499**, and
  **19.314868904/19.260007079 tok/s**. Control/candidate means are
  **19.285820872 / 19.301343552**, a
  **+0.015522680 / +0.080488%** movement.
- Aggregate matched control/candidate is
  **19.278381518 / 19.289499778 tok/s**, a
  **+0.011118260 / +0.057672%** win. All 20 runs retain 128 timed tokens,
  digest `d0193f6d413b68c1`, and last token 11406.
- Strict C++20/O3 compilation and the complete focused A114 exact suite pass.
  Decision: retain the one-line simplification in signed `ad11696f2e`; the
  selected direct gap is now **0.710500222 tok/s / 3.683352%**.

### 2026-09-01 18:36 PDT - PERF-A115/A116/A117 fused-Q4 epilogue search

- A fresh selected A114+A113 Metal System Trace maps **36,768 / 37,168**
  sampled target-process shader PCs with zero ambiguous mappings. The fused
  Q4 gate/up/SwiGLU shader owns **42.141%** of all PCs, custom Q5 kernels own
  **44.533%**, and the remaining ordinary Q4 kernels own **6.804%**. This
  makes the fused kernel the largest single shader and establishes production
  reachability for the epilogue candidates.
- PERF-A115 halves the fused threadgroup from eight to four SIMD groups while
  preserving four results per group. Its balanced 5,000-iteration means are
  control **0.566040871 ms** and candidate **0.567203850 ms**, about
  **0.205% slower**. It is rejected before a full-model run.
- PERF-A116 computes all four reductions, then assigns rows to lanes 0--3 by
  dynamically indexing the thread-local result arrays. Exact parity and the
  precise sigmoid-boundary test pass. A 10,000-iteration reversed window
  averages candidate **0.564099544 ms** and control **0.562567529 ms**, about
  **0.272% slower**; dynamic selection is rejected.
- PERF-A117 retains the selected eight-SIMD geometry and selects each lane's
  already-reduced gate/up result through compile-time-named ternaries. This
  keeps the four exact `metal::precise::exp` chains independent without a
  dynamically indexed local array. An initial timing build accidentally still
  carried A115's geometry and is excluded. The corrected diff against
  `ad11696f2e` contains only the epilogue ownership change.
- Strict C++20/O3 warnings-as-errors builds pass with only the established
  macOS 26.0 / MLX 26.2 linker warning. Candidate dylib, focused test, and
  microbenchmark SHA-256 values are
  `c8ed45122f36a700b588d93f7a227d31003a01f3b4370d9195367ee7c255f36e`,
  `b1383f99fd2d73ae48753605aee3318652df489d75436da08dc85ae62166f0ef`,
  and `b7829284af06cf2bc290ec8ab2659b69fbc9c1e0d3f024308b87160dff80fb15`.
  Q4 QMV `512/64`, `5120/128`, and `5120/17408`, fused SwiGLU `512/64` and
  `5120/17408`, and the `-6.84375` sigmoid boundary all report zero mismatch.
- Corrected 10,000-iteration production-shape micro results are candidate
  **0.556409946, 0.556899100 ms** and serial control
  **0.566113429, 0.563282721 ms**. Means are
  **0.556654523 / 0.564698075 ms**, a **1.4243%** candidate latency reduction;
  every arm reports digest `8a9031349585365a` and first value `0.875`.
- One full-model screen improves **19.276941020 -> 19.410146375 tok/s** with
  canonical output. The first five-pair control/candidate window is
  **19.275163985/19.368736679**,
  **19.255827337/19.482499777**,
  **19.315313630/19.464807390**,
  **19.244643242/19.468935480**, and
  **19.250769815/19.469615480 tok/s**. Means are
  **19.268343602 / 19.450918961**, a
  **+0.182575359 / +0.947541%** gain.
- An immediately following reverse batch fell outside the valid host band at
  **11.590838117--14.360129325 tok/s** for both artifacts while disk reads,
  page-ins, and swapouts surged. It is retained as externally contaminated
  evidence and excluded. After a 60-second idle interval, a selected-control
  probe recovered to **19.203472228 tok/s**. The replacement reverse window
  was split by another 60-second idle interval to preserve residency.
- Replacement reversed candidate/control pairs are
  **19.449756511/19.249257165**,
  **19.457791537/19.284351927**,
  **19.446175454/19.238727679**,
  **19.462623411/19.311370484**, and
  **19.459981955/19.259331137 tok/s**. Control/candidate means are
  **19.268607678 / 19.455265774**, a
  **+0.186658095 / +0.968716%** gain. All 20 clean qualification runs preserve
  128 timed tokens, digest `d0193f6d413b68c1`, and last token 11406.
- Aggregate clean control/candidate is
  **19.268475640 / 19.453092367 tok/s**, a
  **+0.184616727 / +0.958128%** win. The direct gap is now
  **0.546907633 tok/s / 2.811417%**. Signed commit `00d09138ce`
  (`perf(mlx): parallelize fused Q4 SwiGLU epilogue`) contains only the Metal
  epilogue change; user-owned overlapping working-copy hashes remain intact.

### 2026-09-01 18:48 PDT - PERF-A118 fused-Q4 vector-load rejection

- Change: replaced only the selected A117 fused-Q4 helper's two scalar
  packed-word reads with one explicit `packed_ushort4` load. Ordinary Q4
  execution and the lane-parallel precise epilogue remained unchanged.
- Benchmark evidence: candidate 10,000-iteration samples are
  **0.563729096 / 0.558256000 ms**, mean **0.560992548**. Selected A117
  control samples are **0.558807537 / 0.556857937 ms**, mean
  **0.557832737**. The vector form is **0.003159811 ms / about 0.566%**
  slower.
- Correctness evidence: the strict focused executable passes all selected Q4
  QMV shapes, both fused-chain shapes, and the `-6.84375` precise sigmoid
  boundary exactly.
- Decision: reject and restore A117 before full-model work. Preserve the
  artifacts and close this exact transaction-width change as PERF-FA135.

### 2026-09-01 18:57 PDT - PERF-A119 row-concatenated Q5 `qkv`/`z` rejection

- Change: admitted one Q5/G64 launch for production dimensions
  `K=5120`, `Nqkv=10240`, and `Nz=6144`, then implemented opt-in load-time
  row concatenation for the 48 linear-attention `in_proj_qkv`/`in_proj_z`
  pairs. Runtime split the combined result at the original output boundary.
- Benchmark evidence: forward two-launch/one-launch means are
  **0.428439779 / 0.419288358 ms**; reversed means are
  **0.424852217 / 0.421896333 ms**. Aggregate is
  **0.426645998 / 0.420592346 ms**, a **1.418893%** isolated reduction.
  Full-model candidate samples are **19.441971742** and
  **19.406059113 tok/s** around an adjacent same-dylib disabled control at
  **19.469521945 tok/s**. Both candidate comparisons are negative.
- Correctness evidence: all 20 micro arms report exact BF16 parity and digest
  `555793dfc2cf896f`; the existing Q5 focused suite passes; all three full
  runs retain target digest `d0193f6d413b68c1` and last token 11406.
- Decision: reject the copied-weight plus output-split implementation and
  restore exact A117. A native one-dispatch kernel that consumes the two
  original matrices and emits two outputs directly is materially different
  and is the next admission candidate.

### 2026-09-01 19:06 PDT - PERF-A120 direct two-output Q5 rejection

- Change: copied A100's exact Q5 unpack and sixteen-FMA sequence into one
  custom kernel over the original `qkv` and `z` tensors. A threadgroup-uniform
  output-range selection writes two original output arrays directly, with no
  concatenated weights and no split node.
- Benchmark evidence: forward separate/paired means are
  **0.423242262 / 0.426657187 ms**; reversed means are
  **0.420248808 / 0.428146542 ms**. Aggregate is
  **0.421745535 / 0.427401865 ms**, a **1.341171%** regression; every pair is
  negative.
- Correctness evidence: the first Metal compile and all 20 timed arms pass
  byte-exact `qkv` and `z` parity with digest `555793dfc2cf896f`.
- Decision: reject before engine wiring or full-model reload. The two-output
  ABI and dynamic parameter/output selection cost more than one saved launch.

### 2026-09-01 19:09 PDT - PERF-A121 exact-ratio 5:3 Q5 rejection

- Change: remapped the same direct two-output kernel to eight SIMD groups per
  threadgroup. Five groups always produce 20 `qkv` rows and three always
  produce 12 `z` rows, exactly matching `10240:6144`; total threadgroups fall
  from 1,024 to 512 while total SIMD work remains fixed.
- Benchmark evidence: forward separate/paired means are
  **0.425085434 / 0.426956033 ms**; reversed means are
  **0.421710258 / 0.425060279 ms**. Aggregate is
  **0.423397846 / 0.426008156 ms**, a **0.616515%** regression.
- Correctness evidence: every order/reverse arm remains byte-exact with
  digest `555793dfc2cf896f`.
- Decision: reject and restore exact A117. Further Q5 work must reduce
  weight-side work or dependencies rather than submission count alone.

### 2026-09-01 19:18 PDT - PERF-A122 fused-Q4 row-interleaving rejection

- Change: combined A117's separate four-row gate and up loops into one loop
  that issues each gate dot immediately before the matching up dot. Geometry,
  packed loads, reductions, per-accumulator update order, and the lane-parallel
  precise BF16 epilogue were unchanged.
- Benchmark evidence: forward A117/A122 means are
  **0.561355594 / 0.570393451 ms**; reversed A122/A117 means are
  **0.567740174 / 0.556083303 ms**. Aggregate control/candidate is
  **0.558719448 / 0.569066813 ms**, a **0.010347364 ms / 1.851979%**
  regression. The candidate loses all ten pairs.
- Correctness evidence: the strict focused suite passes every ordinary and
  fused Q4 shape plus the `-6.84375` sigmoid boundary bit-exactly. All twenty
  timed arms report digest `8a9031349585365a` and first value `0.875`.
- Decision: reject before a dylib or model reload and restore exact A117.
  Source-order alternation does not reduce work and disrupts the compiler's
  faster grouped-stream schedule; close it as PERF-FA138.
