# Performance Log

## Current Active Timings

| Benchmark | Baseline | Current | Delta | Command | Last Updated |
|---|---:|---:|---:|---|---|
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
