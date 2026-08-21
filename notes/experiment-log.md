# Qwen3.8 27B native-Windows SGLang experiment log

> **Location note:** This complete chronological record was migrated from the
> repository-root `NOTES.md` into `notes/experiment-log.md`. Historical entries
> retain their original path and state descriptions. Append future recovery
> checkpoints here.

## Recovery checkpoint - 2026-08-15 17:14 PDT

- Goal: minimize OpenCode2 time to first response and maximize sustained single-stream generation speed on the RTX 5090.
- Client: OpenCode2. Model: `C:\Users\Daniel\models\Qwen3.8-27B-UD-Q4_K_XL\Qwen3.8-27B-UD-Q4_K_XL.gguf`.
- Companion private fork: `C:\Users\Daniel\llama.cpp`, branch `work/dynamic-kv-qwen`.
- This checkout is at `dd458f3212` on `main`; its substantial native-Windows/GGUF/Qwen3.8 work is currently uncommitted and must be preserved.
- Live inventory at recovery: 46 tracked files modified (1056 insertions, 273 deletions) plus native-Windows support files, GGUF tests, event-loop support, and `scripts/windows/serve_qwen38_27b_5090.ps1`.
- The launch script currently targets one request, Triton attention, PyTorch sampling, decode CUDA graphs, 131072 context, 2048-token chunked prefill, and 0.88 static-memory fraction.
- The older `C:\Users\Daniel\llama.cpp\NOTES.md` understates SGLang progress. Reconstruct current behavior from this worktree and fresh measurements.
- No local benchmark artifact was found during the first narrow recovery scan. Fresh results must be recorded below as they are produced.

## Recovery queue

1. Validate the modified SGLang unit tests and launcher without disturbing the current worktree.
2. Start the server and capture a repeatable OpenAI-compatible baseline matching OpenCode2's request shape.
3. Compare SGLang with the current llama.cpp server on prompt throughput, generation throughput, TTFT, wall time, VRAM, and correctness.
4. Profile the winning path, make one controlled change at a time, and retain only measured improvements.

## Measurements

Pending fresh baseline.

## Validation log

### 2026-08-15 17:18 PDT - recovery validation

- Installed editable launcher reports `sglang 0.5.18.dev601+gdd458f321` at revision `dd458f3`.
- Native GGUF CUDA extension already exists at `%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py313_cu130\sglang_windows_gguf\sglang_windows_gguf.pyd`; its last link time is 2026-08-15 15:31 PDT.
- Targeted recovery suite passed:

```text
python -m pytest test/registered/unit/utils/test_event_loop.py test/registered/unit/utils/test_gguf_native.py test/registered/unit/models/test_qwen3_5_packed_weight_loader.py -q
13 passed, 14 warnings in 7.74s
```

- The passing tests cover Windows selector-loop selection, per-process GGUF metadata caching, packed Qwen3.5 weight loading, and GGUF V-head layout restoration.
- PowerShell history confirms repeated native SGLang launcher attempts through `scripts/windows/serve_qwen38_27b_5090.ps1`, after the earlier llama.cpp/OpenCode2 work. Command history contains no performance result, so runtime success and speed still require a fresh controlled check.
- OpenCode2 is `v0.0.0-next-17444`. Its sanitized local provider configuration already points `llama-cpp/qwen3.8-27b` at `http://127.0.0.1:30000/v1` using the OpenAI-compatible provider, enables/preserves thinking, exposes tools, and advertises a 32768-token client context with 8192 output tokens. Despite the historical provider id, its display name is `SGLang local`.

### 2026-08-15 17:22 PDT - live-test preflight

- Python stack: PyTorch `2.13.0+cu130`, CUDA runtime `13.0`; CUDA is available and identifies the RTX 5090.
- Pre-launch GPU state: 32607 MiB total, 1941 MiB used, 30247 MiB free, P8, 48.93 W, 41 C.
- Pre-launch system memory: 49,452,764 KiB visible, 37,883,140 KiB free.
- The next action is the first fresh full-model launch through the existing script. Standard output and error will be retained in temp logs and inspected with narrow readiness/error/performance patterns.

### 2026-08-15 17:20 PDT - fresh SGLang launch succeeds

- Launch command: existing `scripts/windows/serve_qwen38_27b_5090.ps1` defaults.
- Recovery logs: `%TEMP%\sglang-qwen-20260815-1722.stdout.log` and `%TEMP%\sglang-qwen-20260815-1722.stderr.log`.
- Weight load: 30.65 s, 17.78 GB device-memory use, 12.36 GB available afterward.
- Memory-pool setup left 2.75 GB available.
- Batch-1 full decode CUDA-graph capture: 3.04 s; 2.81 GB available afterward.
- OpenAI-compatible server became ready at `http://127.0.0.1:30000` and the port probe succeeds.
- Idle post-start GPU state: 29926 MiB used, 2262 MiB free, P8, 44.46 W, 41 C.
- This confirms the current native-Windows SGLang/GGUF/Qwen3.8 path is operational. No request-performance claim yet.

### 2026-08-15 17:24 PDT - benchmark workload locked

- Added `benchmark/windows/qwen38_local_prompt.json`, a tiny local two-turn fixture used only as a safe token source by SGLang's existing random-request benchmark.
- The benchmark repeats/truncates this fixture to an exact requested token length. It avoids remote dataset access and avoids arbitrary random token IDs that can produce invalid model states.
- Controlled profile: one request, concurrency one, 6213 input tokens, 128 output tokens, deterministic seed 42, warmup capped to 32 output tokens, then server cache flush before measurement.
- The stock serving benchmark's first smoke attempt was stopped after more than 60 seconds because its client had not reached the server; it was spending time in local GGUF/tokenizer setup. No model request or measurement occurred.
- Added `scripts/windows/bench_openai_stream.py`. It uses only the standard library, calibrates prompt length through the local `/v1/tokenize` endpoint, warms the exact request shape, flushes server cache, consumes the OpenAI SSE stream without printing generated text, and reports TTFT, end-to-end time, observed prompt rate, and steady decode rate as one JSON object.

### 2026-08-15 17:27 PDT - first live API defect isolated

- The benchmark smoke request reached `/v1/tokenize`, which returned HTTP 500 before any model work.
- Server traceback ends at `orjson.dumps(...): TypeError: Integer exceeds 64-bit range`.
- Root cause: the GGUF-built `PreTrainedTokenizerFast` inherited Transformers' enormous unknown-length sentinel as `model_max_length`; the tokenize endpoint includes that value in its response.
- Fix: derive a finite `model_max_length` from `<architecture>.context_length` in GGUF metadata (262144 for this model), preserve an explicit caller override, and ignore missing/out-of-range metadata.
- Added regression coverage for the finite, missing, and out-of-range cases. A server restart is required because the current tokenizer object predates the fix.
- Post-fix targeted suite: `14 passed, 14 warnings in 14.19s`; `git diff --check` is clean.

### 2026-08-15 17:30 PDT - tokenize fix live and smoke benchmark passes

- Restarted server with the tokenizer fix. Second launch: weight load 38.41 s, memory pool left 2.76 GB, decode CUDA graph captured in 2.98 s, then the server became ready.
- OpenAI streaming smoke profile, with exact-shape warmup and cache flush: 256 prompt tokens, 16 completion tokens, 0.625466 s TTFT, 1.008610 s end to end, 409.295 observed prompt tok/s, 39.150 steady decode tok/s, finish reason `length`.
- `/v1/tokenize` now returns a serialization-safe response and calibrated the prompt to exactly 256 tokens. The OpenAI SSE path, usage accounting, thinking template, forced length, and cache flush all work end to end.

### 2026-08-15 17:31 PDT - 6213-token OpenCode-shaped baseline

```text
prompt tokens:          6213
completion tokens:       128
TTFT:                 13.912322 s
end to end:           17.196236 s
observed prompt rate:   446.583 tok/s
steady decode rate:      38.673 tok/s
finish reason:           length
```

- The profile includes an exact-shape warmup, cache flush, local OpenAI chat templating, HTTP/SSE overhead, and a forced 128-token completion.
- Server-side measured decode settled at 38.95 tok/s with the batch-1 CUDA graph active, corroborating the client-side 38.673 tok/s.
- The measured prompt was chunked as 2048 + 2048 + 2048 + 69. After warmup, server chunk rates were 240.71, 449.87, 1675.95, and 1502.36 tok/s; end-to-first-token observed rate was 446.583 tok/s.
- Post-request GPU state was 30731 MiB used and 1457 MiB free. The unified cache retains state after requests, accounting for the increase from idle.
- Compared with the older llama.cpp OpenCode checkpoint (278.17 prompt tok/s, 31.47 decode tok/s, 26.085 s total for 6213 prompt and 119 completion tokens), this SGLang baseline is about 60.5% faster in prompt rate, 22.9% faster in decode rate, and 34.1% shorter end to end despite producing nine more tokens. Re-run llama.cpp with the same harness for a current apples-to-apples comparison.

### 2026-08-15 17:34 PDT - first tuning axis

- SGLang is already using its Triton chunkwise GDN prefill kernel. llama.cpp's fused CUDA GDN file still contains `TODO: Add chunked kernel for even faster pre-fill`, explaining the architectural source of the current prompt-rate lead.
- The Blackwell CuTe DSL GDN backend exists in this checkout, but its required `cutlass` Python package is absent and the project dependency excludes `nvidia-cutlass-dsl` on Windows. Keep Triton as the viable native-Windows backend for now.
- Parameterized `ChunkedPrefillSize` in the Windows launcher, retaining 2048 as the baseline default. This permits controlled 4096/8192 tests without editing the command between runs.

### 2026-08-15 17:37 PDT - 4096-token prefill chunks

- Same server configuration and 6213/128 workload, changing only `ChunkedPrefillSize` from 2048 to 4096.
- Three cache-flushed measured runs after exact-shape warmup:

```text
run  TTFT (s)   E2E (s)   prompt tok/s   decode tok/s
1    13.659356  16.927516 454.853        38.860
2    13.637822  16.931294 455.571        38.561
3    13.742506  17.038411 452.101        38.533
mean 13.679895  16.965740 454.175        38.651
```

- Relative to the initial 2048 run, the 4096 mean reduces TTFT by 1.67% and E2E by 1.34%; decode is effectively unchanged. This is a small, consistent prefill win. A repeated 2048 control remains necessary before final selection.
- The benchmark now records only the generated text length and SHA-256 digest. This provides a deterministic-output guard for aggressive backend/state experiments while keeping generated content out of command output.

### 2026-08-15 17:40 PDT - VRAM headroom and repeated-run flaw found

- The first 8192-chunk run produced the same deterministic output digest and measured 14.384421 s TTFT / 39.398 decode tok/s. It was slower in prefill than 4096.
- On the next invocation, the warmup reused 6208 cached prompt tokens from the preceding measured run. After the benchmark flushed that cache, the measured run had a cold prompt and an incompletely representative warmup.
- More importantly, compiled kernels and retained cache state drove GPU use to 31756 MiB, leaving only 432 MiB free. The server logged late Triton device loads at 0.57-0.75 GiB free, and the second run's first decode interval collapsed before recovering to ~41 tok/s. Output remained identical (`df852e3f...e9c30`), so this is performance instability under VRAM pressure rather than model divergence.
- Fixed the harness to flush before and after exact-shape warmup on every invocation.
- Parameterized `MaxTotalTokens` separately from logical `ContextLength`. The 131072-token pool allocates 8.00 GB of full-attention KV even though OpenCode2 currently advertises 32768 context. Testing a 32768-token active pool will free about 6 GB while retaining the server's 131072 logical context setting.

### 2026-08-15 17:45 PDT - 32768 active pool restores stable speed

- Configuration: logical context 131072, active token pool 32768, prefill chunk 8192, all other launcher settings unchanged.
- KV allocation fell from 8.00 GB to 2.00 GB. Post-start available device memory rose from roughly 2.8 GB to 9.47 GB; after three full warmup/measurement cycles, 6.61 GB remained free.
- Three corrected, cache-flushed 6213/128 runs:

```text
run  TTFT (s)   E2E (s)   prompt tok/s   decode tok/s
1    13.675127  16.764820 454.329        41.104
2    13.329190  16.399793 466.120        41.360
3    13.330023  16.483607 466.091        40.272
mean 13.444780  16.549407 462.180        40.912
```

- All runs produced 128 tokens, finish reason `length`, 654 output characters, and identical SHA-256 `df852e3f6a6e5bacc9de7023b01bd031c4e0fe2ecc408bc19c59164cf40e9c30`.
- Against the original 131072-pool/2048-chunk baseline, this mean improves observed prompt rate by 3.49%, decode rate by 5.79%, TTFT by 3.36%, and end-to-end time by 3.76%. The main benefit is stable headroom that prevents late JIT/cache growth from pushing WDDM into a pathological low-memory regime.
- Parameterized `ContinuousDecodeSteps` in the launcher for a later scheduling-overhead sweep; the retained default is 1 until measured.

### 2026-08-15 17:49 PDT - 4096 wins with the 32768 pool

- Configuration: logical context 131072, active token pool 32768, prefill chunk 4096.
- Three corrected, cache-flushed 6213/128 runs:

```text
run  TTFT (s)   E2E (s)   prompt tok/s   decode tok/s
1    12.959925  16.033958 479.401        41.314
2    13.132193  16.204395 473.112        41.338
3    13.356186  16.472012 465.178        40.760
mean 13.149435  16.236788 472.564        41.137
```

- Output digest and length match every 8192 run exactly.
- Against 8192 chunks at the same active-pool size, 4096 improves mean TTFT by 2.20%, end-to-end time by 1.89%, and prompt rate by 2.25%; decode is within normal variance. Retain 4096.
- Against the original 131072-pool/2048 baseline, the current mean improves prompt rate by 5.82%, decode rate by 6.37%, TTFT by 5.48%, and end-to-end time by 5.58%.

### 2026-08-15 17:53 PDT - continuous decode steps rejected

- `ContinuousDecodeSteps=4`, with the retained 4096/32768 settings, produced means of 13.496591 s TTFT, 16.607805 s E2E, 460.340 prompt tok/s, and 40.828 decode tok/s across three identical-output runs.
- It did not reduce decode overhead and increased TTFT/E2E. Retain `ContinuousDecodeSteps=1`.
- Parameterized the Mamba radix strategy and ReplaySSM switch for the next decode-path experiment. ReplaySSM requires `no_buffer`; defaults remain `extra_buffer_lazy` with ReplaySSM off until measured.

### 2026-08-15 17:57 PDT - ReplaySSM rejected

- Configuration: logical context 131072, active token pool 32768, prefill chunk 4096, `MambaRadixCacheStrategy=no_buffer`, and ReplaySSM enabled.
- Three corrected, cache-flushed 6213/128 runs:

```text
run  TTFT (s)   E2E (s)   prompt tok/s   decode tok/s
1    13.227247  16.430439 469.712        39.648
2    13.455009  16.641209 461.761        39.859
3    13.434659  16.618970 462.461        39.883
mean 13.372305  16.563539 464.645        39.797
```

- All runs retained the 654-character, 128-token deterministic output with SHA-256 `df852e3f6a6e5bacc9de7023b01bd031c4e0fe2ecc408bc19c59164cf40e9c30`.
- Relative to the current 4096/32768 winner, ReplaySSM reduced decode throughput by 3.26% and increased end-to-end latency by 2.01%. Retain `extra_buffer_lazy` with ReplaySSM disabled.
- Parameterized Mamba state dtype and pinned both linear-attention modes to Triton so the native-Windows SM100 path can test BF16 without the unavailable FlashInfer backend being auto-selected.

### 2026-08-15 18:00:54 PDT - explicit recovery checkpoint

- ReplaySSM is measured and rejected; its complete three-run result is immediately above.
- Current retained control: logical context 131072, active pool 32768, prefill chunk 4096, continuous decode steps 1, `extra_buffer_lazy`, ReplaySSM off, FP32 Mamba state.
- A single-axis BF16 Mamba-state experiment was launched as PowerShell PID 26460, using the retained control settings and explicit Triton linear-attention prefill/decode backends.
- BF16 logs: `%TEMP%\sglang-qwen-bf16.stdout.log` and `%TEMP%\sglang-qwen-bf16.stderr.log`.
- Next recovery action: inspect only readiness/error markers in the BF16 stderr log; if ready, run three corrected 6213/128 benchmark cycles and compare their digest and means with the FP32 control.

### 2026-08-15 18:02:13 PDT - post-compaction checkpoint

- `date` confirmed `Sat Aug 15 18:02:13 PDT 2026` immediately after context recovery.
- BF16 experiment remains active under launcher PID 26460. At the last selective log read it had accepted `mamba_ssm_dtype='bfloat16'`, pinned both linear-attention modes to Triton, and reached `Load weight begin` with 30.14 GB available.
- No BF16 benchmark run has been recorded yet. Resume by checking server readiness, then run three corrected 6213/128 cycles before making any retention decision.
- Standing procedure requested by the user: after every future context compaction, run `date` and write a new `notes/experiment-log.md` recovery checkpoint before continuing.

### 2026-08-15 18:04:27 PDT - BF16 Mamba state rejected

- Configuration: the retained 4096/32768 control with `MambaSsmDtype=bfloat16`; both linear-attention modes remained explicitly pinned to Triton.
- Three corrected, cache-flushed 6213/128 runs:

```text
run  TTFT (s)   E2E (s)   prompt tok/s   decode tok/s
1    13.022886  16.083950 477.083        41.489
2    13.895463  17.160843 447.124        38.893
3    13.437499  16.504162 462.363        41.413
mean 13.451949  16.582985 462.190        40.598
```

- BF16 was deterministic across its own runs: 128 tokens, 644 characters, SHA-256 `f2dc18d4830d0cd9120041cf9ecacb8669c11316c009d33ef0339a3d101f573f`.
- Relative to the FP32 control mean, BF16 increased TTFT by 2.30% and end-to-end time by 2.13%, while reducing prompt throughput by 2.19% and decode throughput by 1.31%.
- BF16 also diverged from the FP32 output guard (654 characters, SHA-256 `df852e3f...e9c30`). Retain FP32 Mamba state.
- Next action: stop BF16, restore the winning FP32 server configuration, then perform the current llama.cpp apples-to-apples benchmark.

### 2026-08-15 18:06:59 PDT - llama.cpp harness smoke checkpoint

- BF16 SGLang was stopped cleanly. llama.cpp MTP/dynamic-KV server is live on port 8080 under launcher PID 21040.
- The harness's llama.cpp `/apply-template` plus `/tokenize` calibration path completed successfully.
- The first small smoke stopped before inference because `/slots/0?action=erase` returned HTTP 501: slot actions require llama.cpp to start with `--slot-save-path`.
- No performance sample was accepted from this attempt.
- Recovery action: restart the otherwise identical llama.cpp command with `--slot-save-path C:\Users\Daniel\AppData\Local\Temp`, repeat the 256/16 smoke, then collect three 6213/128 runs.

### 2026-08-15 18:10:58 PDT - fresh llama.cpp comparison complete

- llama.cpp was restarted with slot actions enabled; the 256/16 smoke passed with exact token accounting and coherent OpenAI streaming usage.
- Retained llama.cpp MTP/dynamic-KV configuration, three cache-erased 6213/128 runs:

```text
run  TTFT (s)   E2E (s)   prompt tok/s   decode tok/s
1    25.985273  29.836042 239.097        32.980
2    25.562760  29.517353 243.049        32.115
3    25.760814  29.671996 241.180        32.471
mean 25.769616  29.675130 241.109        32.522
```

- All llama.cpp runs were deterministic: 128 tokens, 630 characters, SHA-256 `080fb9eb633200dc7940ae499f91cd3fbef6120b10a6b17c338b7375c995d11b`.
- The current SGLang winner is 96.00% faster in observed prompt throughput and 26.49% faster in steady decode throughput. It reduces TTFT by 48.97% and end-to-end latency by 45.29%.
- This is now an apples-to-apples HTTP/SSE comparison: same GGUF, exact prompt and completion counts, local deterministic prompt construction, thinking template, forced length, single stream, exact-shape warmup, and cache reset before measurement.
- llama.cpp remains live on port 8080 under launcher PID 3392. Next high-value control is the already-planned llama.cpp no-MTP run; afterward restore SGLang FP32 and make its winning pool/chunk settings the launcher defaults.

### 2026-08-15 18:14:56 PDT - llama.cpp no-MTP control complete

- Same llama.cpp configuration and harness, changing only speculative decoding to `--spec-type none`.

```text
run  TTFT (s)   E2E (s)   prompt tok/s   decode tok/s
1    24.260230  29.319567 256.098        25.102
2    24.154200  29.184422 257.222        25.247
3    24.520069  29.703282 253.384        24.502
mean 24.311500  29.402424 255.568        24.950
```

- Output remained identical to llama.cpp MTP across all runs: 128 tokens, 630 characters, SHA-256 `080fb9eb633200dc7940ae499f91cd3fbef6120b10a6b17c338b7375c995d11b`.
- Removing MTP improves llama.cpp prompt throughput by 6.00% and TTFT by 5.66%, but reduces decode throughput by 23.28%. At 128 output tokens it improves end-to-end time by only 0.92%.
- From the measured client timings, MTP repays its prefill overhead at roughly 157 generated tokens. Retain MTP for normal OpenCode2 responses; use no-MTP only for predictably short replies.
- SGLang still beats the better no-MTP 128-token total by 44.78%, with 84.91% higher prompt throughput and 64.88% higher decode throughput.
- Current server on port 8080 is the no-MTP control under launcher PID 5900. Next action: stop it, make the SGLang winning values launcher defaults, restore SGLang FP32, and verify OpenCode2 end to end.

### 2026-08-15 18:18:08 PDT - selected defaults restored; final decode sweep

- Stopped the llama.cpp no-MTP control. Ports 8080 and 30000 were clear before restoring SGLang.
- Updated the SGLang Windows launcher's defaults to the measured winner: `MaxTotalTokens=32768`, `ChunkedPrefillSize=4096`, `MambaSsmDtype=float32`, continuous decode steps 1, `extra_buffer_lazy`, ReplaySSM off, and explicit Triton linear-attention backends.
- Exposed `TritonAttentionNumKvSplits` with the existing SGLang default of 8 for a final low-cost decode sweep.
- Fresh split-8/default control: 13.100961 s TTFT, 16.229136 s E2E, 474.240 prompt tok/s, 40.599 decode tok/s, with the retained 654-character output digest. This corroborates the prior three-run winning mean.
- Split 4 is loading under launcher PID 8424. Recovery action: wait for readiness, run a corrected 6213/128 sample, and retain only if it clearly improves decode without regressing total latency or output.

### 2026-08-15 18:20:18 PDT - Triton KV split 4 rejected

- Corrected 6213/128 result with `TritonAttentionNumKvSplits=4`: 13.230808 s TTFT, 16.513970 s E2E, 469.586 prompt tok/s, and 38.682 decode tok/s.
- Versus the fresh split-8 control, split 4 reduced prompt throughput by 0.98%, reduced decode throughput by 4.72%, and increased E2E latency by 1.76%.
- It also changed the deterministic output to 657 characters with SHA-256 `72afad8e879e7e399e14d67d09dbdda4aa7348e0b16516622fc8cfb74b994bee`.
- Reject split 4 after this clear loss. Test split 16 once; retain split 8 unless split 16 produces a clear, output-safe win.

### 2026-08-15 18:23:58 PDT - Triton KV split 16 wins

- Three corrected, cache-flushed 6213/128 runs with `TritonAttentionNumKvSplits=16`:

```text
run  TTFT (s)   E2E (s)   prompt tok/s   decode tok/s
1    13.208662  16.250590 470.373        41.750
2    13.050497  16.031002 476.074        42.610
3    13.079629  16.060752 475.013        42.601
mean 13.112929  16.114115 473.820        42.320
```

- All split-16 runs were deterministic: 128 tokens, 646 characters, SHA-256 `0e78a5597df6c2f67d4b5cf22c8aff046d23e8991123f72e38d7f62c1e9ae7b5`.
- Versus the prior split-8 three-run mean, split 16 improves prompt throughput by 0.27%, decode throughput by 2.88%, TTFT by 0.28%, and end-to-end latency by 0.76%.
- Promoted split 16 to the launcher default. The selected live server is already running this exact configuration on port 30000 under launcher PID 17392.
- Final selected mean versus the original SGLang baseline: +6.10% prompt throughput, +9.43% decode throughput, -5.75% TTFT, and -6.29% end-to-end latency.
- Final selected mean versus fresh llama.cpp MTP: +96.52% prompt throughput, +30.13% decode throughput, -49.11% TTFT, and -45.70% end-to-end latency.

### 2026-08-15 18:27:26 PDT - real OpenCode2 integration exposes queue latency

- `opencode2 run --standalone --model llama-cpp/qwen3.8-27b --format json 'Reply with exactly READY. Do not call tools.'` completed successfully with exit code 0 and returned the requested text through the configured SGLang endpoint.
- Total CLI wall time was 28.74 s. The server log shows why: OpenCode2 submitted a 589-token auxiliary request and an 8719-token agent request during the same turn.
- With `max-running-requests=1`, the agent request sat queued while the auxiliary request decoded. Its prefill did not begin until roughly 20 seconds later. This integration queue dominates the otherwise much faster model path.
- The 8719-token agent request then prefetched as 4096 + 4096 + 527 and entered the batch-1 decode CUDA graph successfully.
- A global OpenCode title-agent override could alter non-local provider sessions, so leave the user's global OpenCode behavior intact.
- Parameterized `MaxRunningRequests` and decode CUDA-graph max batch size. Next recovery action: test both at 2 with the same real OpenCode2 command; retain only if it removes the auxiliary-request queue without harming stability.

### 2026-08-15 18:29:52 PDT - concurrency attempt capped by Mamba state pool

- Launched with requested max-running requests 2 and decode CUDA graph max batch size 2 under launcher PID 15988.
- SGLang resolved the scheduler back to `max_running_requests=1` and captured only batch size 1.
- Root cause from `kv_cache_configurator.py`: this hybrid model needs four Mamba state slots per active request. `MaxMambaCacheSize=4` therefore caps concurrency to one even when the CLI requests two.
- No OpenCode timing was taken from this knowingly equivalent server.
- There is ample VRAM headroom. Next recovery action: relaunch with `MaxMambaCacheSize=8`, max-running requests 2, and graph batch size 2; confirm the resolved scheduler and captured batch sizes before running OpenCode2.

### 2026-08-15 18:34:16 PDT - concurrency works, but FP32 state-pool tradeoff found

- With `MaxMambaCacheSize=8`, SGLang resolved `max_running_requests=2`, captured decode CUDA graphs for batch sizes 1 and 2, and retained 8.89 GB free after capture.
- Exact real OpenCode2 repeat passed and reduced CLI wall time from 28.74 s to 25.03 s (12.92%). The 8719-token main request prefetched while the 589-token auxiliary request remained active, so the prior hard queue was removed.
- OpenCode cancelled the auxiliary request after the main response. SGLang then emitted repeated `state was deleted in TokenizerManager` messages for its late output; this is cancellation/logging cleanup debt, not a failed user request.
- Two corrected synthetic single-stream controls on this server:

```text
run  TTFT (s)   E2E (s)   prompt tok/s   decode tok/s
1    13.886771  17.080778 447.404        39.762
2    13.971319  17.170538 444.697        39.697
mean 13.929045  17.125658 446.051        39.730
```

- Relative to the selected one-request split-16 mean, the larger FP32 state pool regresses single-stream prompt throughput by 5.86%, decode throughput by 6.12%, and end-to-end latency by 6.28%.
- Real OpenCode2 still benefits overall, but this is not yet the clean fastest configuration. Next experiment: use `no_buffer` without ReplaySSM. Its three-state-per-request ratio needs only six FP32 slots for two requests, potentially retaining concurrency with less state-pool overhead.

### 2026-08-15 18:37:17 PDT - no-buffer concurrency rejected

- `no_buffer`, FP32 state pool 6, max-running requests 2, graph batch sizes 1 and 2 resolved successfully with 9.18 GB free after capture.
- Corrected 6213/128 control: 14.039951 s TTFT, 17.188727 s E2E, 442.523 prompt tok/s, and 40.333 decode tok/s. Output matched the selected split-16 digest.
- This is slower overall than both the one-request selected configuration and the two-request `extra_buffer_lazy`/pool-8 configuration. The smaller state pool did not remove the concurrency-path prefill penalty.
- Reject `no_buffer` without spending another real OpenCode turn. Next isolation: restore `extra_buffer_lazy`, pool 8, max-running requests 2, while capturing only the batch-1 CUDA graph. This distinguishes scheduler/state-pool cost from batch-2 graph cost.

### 2026-08-15 18:42:09 PDT - scoped OpenCode title model removes the real bottleneck

- `extra_buffer_lazy`, pool 8, max-running requests 2, graph batch size 1 still measured only 449.974 prompt tok/s, 39.101 decode tok/s, and 17.055447 s E2E. The single-stream penalty comes from the larger state/scheduler configuration, not capturing batch 2.
- Added `scripts/windows/opencode_qwen.ps1`. It uses OpenCode2's process-scoped `OPENCODE_CONFIG_CONTENT`, restoring the previous environment afterward, so no global provider or user configuration is changed.
- A title-agent options override alone lost to the local model definition's `body` precedence. The wrapper now supplies a process-scoped `qwen3.8-27b-title` model alias whose `modelID` remains `qwen3.8-27b` while its chat-template body disables thinking. The main `qwen3.8-27b` model retains thinking.
- With the alias and the two-request server, the same exact real OpenCode2 turn completed in 23.79 s versus 28.74 s unoptimized (17.24% faster). The auxiliary prompt fell from 589 to 555 tokens and no late-output warning stream followed it.
- Stopped the concurrency server. The selected fast single-stream defaults are loading on port 30000 under launcher PID 6384.
- Next recovery action: run the exact wrapper-based OpenCode2 turn against the restored one-request server. If its short non-thinking title clears before the main request, this combines the fastest model path with the fastest integration path.

### 2026-08-15 18:48:27 PDT - final OpenCode2 paths measured

- The restored selected server resolved exactly as intended: max-running requests 1, active pool 32768, Mamba pool 4, split 16, batch-1 full CUDA graph, and 9.47 GB free after capture. It remains live on port 30000 under launcher PID 6384.
- A sanitized OpenCode export confirmed the process-scoped title alias used 555 input tokens and only 2 output tokens. The main build-agent request used 8721 input and 33 output tokens; main thinking remained enabled.
- One initial one-request wrapper trial ended after `step_start` with the main prefill complete but no decode. The immediate repeat and both subsequent controlled trials returned complete text events with exit code 0. Keep this transient in mind if the preview CLI changes its standalone lifecycle again.
- A 13.07 s trial reused 4800 cached main-prompt tokens and is recorded only as a warm-prefix observation.
- After an explicit SGLang cache flush, the snapshot-preserving wrapper completed the exact real OpenCode2 turn in 23.94 s. This is 16.71% faster than the original 28.74 s cold turn, with global OpenCode configuration untouched.
- Added wrapper switch `-DisableSnapshots`. On a flushed cache it completed in 22.17 s, 22.86% faster than the original turn. The server scheduled the 8721-token main request first and handled the 555-token title afterward.
- `-DisableSnapshots` deliberately removes snapshot-backed undo/revert for that invocation. The wrapper default preserves snapshots. Both paths restore any pre-existing `OPENCODE_CONFIG_CONTENT` environment value on exit.
- The two-request server is rejected as a default: it improves auxiliary overlap but costs about 6% in steady single-stream throughput. The title wrapper achieves a larger real-workflow win while retaining the fastest model server.

### 2026-08-15 18:52:18 PDT - final-check warm-lifetime regression under investigation

- Focused regression suite remains green: 14 passed, 14 warnings in 11.04 s. Python compilation, both PowerShell parse checks, and `git diff --check` all pass.
- After several real OpenCode turns and additional prompt shapes, the live default server's next three corrected 6213/128 controls averaged 13.963498 s TTFT, 17.152857 s E2E, 444.953 prompt tok/s, and 39.820 decode tok/s. Output stayed identical to the split-16 guard.
- GPU telemetry during a full run showed 100% SM utilization, roughly 599-600 W, 2857-2910 MHz during prefill, no thermal violation, and 6.59 GB free. This is not an idle-clock or low-memory failure.
- The late mean is about 6% below the fresh split-16 mean, so do not silently present the fresh peak as permanent steady state.
- Recovery action: restart the exact default server and benchmark immediately. This distinguishes per-process accumulated kernel/cache state from external thermal/power variance; record both peak and steady observations in the final result.

### 2026-08-15 18:55:35 PDT - external GPU contention identified; concurrency promoted

- Restarted the exact one-request selected server. Its immediate corrected control was still 13.862773 s TTFT, 17.047025 s E2E, 448.179 prompt tok/s, and 39.884 decode tok/s. The slowdown therefore was not accumulated per-process SGLang state.
- During a full run the RTX 5090 held 100% SM utilization and 599-600 W without thermal violation. `nvidia-smi pmon -c 1` then identified Chrome PID 22312 consuming about 5% SM and 2% memory continuously. That external contention closely matches the gap from the earlier clean split-16 peak.
- Under the same contended environment, the two-request `extra_buffer_lazy`/pool-8 mean (446.051 prompt tok/s, 39.730 decode tok/s, 17.125658 s E2E) and the one-request/pool-4 late mean (444.953, 39.820, 17.152857) are statistically equivalent.
- Correction to the earlier provisional interpretation: the post-18:27 variants were measured after external GPU contention appeared, so their roughly 6% difference from the clean peak cannot be assigned to concurrency. The direct same-environment control removes that apparent penalty.
- Promoted the OpenCode-oriented defaults to Mamba pool 8, max-running requests 2, and decode CUDA graph batch size 2. This configuration previously reduced the unwrapped real OpenCode2 cold wall time from 28.74 s to 25.03 s by overlapping its auxiliary request, while preserving single-stream speed under a controlled environment comparison.
- Clean, uncontended peak remains the split-16 three-run mean of 473.820 prompt tok/s and 42.320 decode tok/s. Current measurements with Chrome's GPU workload are about 445 prompt tok/s and 39.8 decode tok/s; closing or idling that GPU-heavy Chrome workload should recover the missing margin.

### 2026-08-15 18:59:44 PDT - final live checkpoint

- Final defaults launched successfully under PowerShell PID 28392. Resolved server state: max-running requests 2, Mamba state pool 8, decode CUDA graph batches `[1, 2]`, active token pool 32768, chunk 4096, split 16, FP32 state, and 8.89 GB free after capture.
- Corrected final live 6213/128 check under the current Chrome GPU workload: 13.828596 s TTFT, 17.013191 s E2E, 449.286 prompt tok/s, 39.879 decode tok/s, 646 output characters, and the retained split-16 SHA-256.
- Final combined `opencode_qwen.ps1 -DisableSnapshots` check returned a complete text event with exit code 0 in 24.02 s. Its sanitized export confirms the auxiliary title used 555 input / 1 output token; the main build request used 8720 input / 21 output tokens with its reasoning part preserved.
- The original unoptimized sanitized export was 589 input / 159 output tokens for the auxiliary title and 8719 input / 40 output for the main request. The scoped alias removes the title's long generation without changing the global config file.
- Verification remains green after implementation: 14 focused tests pass; both PowerShell scripts parse; the Python benchmark compiles; `git diff --check` passes. Re-run the small static checks after any future launcher edit.
- Live endpoint: `http://127.0.0.1:30000/v1`. Port 8080 is clear. Recovery logs: `%TEMP%\sglang-qwen-selected.stdout.log` and `%TEMP%\sglang-qwen-selected.stderr.log`.

### 2026-08-15 19:19:11 PDT - post-compaction clarification

- Post-compaction checkpoint time confirmed with `date`: Sat Aug 15 19:19:11 PDT 2026.
- The user confirmed Chrome's approximately 5% SM activity was intentional movie playback. Treat the final live result of 449.286 prompt tok/s and 39.879 decode tok/s as a movie-contended measurement, not a server regression or defect.
- The clean, uncontended three-run peak remains 473.820 prompt tok/s and 42.320 decode tok/s. The selected server configuration and live endpoint require no corrective restart.
### 2026-08-15 19:22:02 PDT - post-compaction checkpoint

- Ran `date` immediately after compaction: `Sat Aug 15 19:22:02 PDT 2026`.
- Resuming from the selected live SGLang configuration and clean peak of 473.820 prompt / 42.320 decode tok/s.
- New investigation: remaining performance outside the already-tuned server flags, including model-specific kernels, graph coverage, scheduling, tokenizer/HTTP overhead, and OpenCode prompt-path overhead.

### 2026-08-15 19:25 PDT - remaining high-value levers identified

- The GGUF contains Qwen3.8's bundled MTP head, and this checkout has a `Qwen3_5ForCausalLMMTP` draft implementation plus GGUF draft-path resolution. SGLang can therefore plausibly load the same GGUF as a one-layer `NEXTN` draft instead of requiring another model file.
- The fork also contains the hybrid GDN `--enable-linear-replayssm-spec` path specifically intended to avoid per-draft full recurrent-state snapshots during linear-chain MTP verification. This is a materially larger possible decode win than further HTTP/tokenizer tuning.
- Prefill CUDA graphs and `torch.compile` remain untested, but they carry longer compilation/startup and compatibility risk on the native-Windows GGUF path. Test cheap runtime axes and bundled MTP first; preserve the known-good launcher defaults until a candidate passes output and repeated-run guards.
- Triton KV split values above the current winner of 16 were never swept. Split 32 is the first low-risk control, measured under the same current Chrome/movie load as its immediate split-16 control.

### 2026-08-15 19:29:46 PDT - Triton KV split 32 rejected

- Immediate split-16 control under the current movie load: 13.767226 s TTFT, 16.948115 s E2E, 451.289 prompt tok/s, 39.926 decode tok/s, 646 characters, retained SHA-256 `0e78a559...e9ae7b5`.
- Two split-32 runs measured 13.917028 / 13.925455 s TTFT, 17.136491 / 17.044715 s E2E, 446.432 / 446.161 prompt tok/s, and 39.448 / 40.715 decode tok/s.
- Split 32 consistently loses about 1.1% prompt throughput and 0.8% E2E latency while changing the deterministic output to 654 characters and SHA-256 `df852e3f...e9c30`. Reject it; split 16 remains selected.
- The split-32 experiment is currently live only as a temporary server. Next action is bundled Qwen3.8 `NEXTN`/MTP startup, then correctness and performance measurements before any launcher-default change.

### 2026-08-15 19:36:40 PDT - bundled MTP native-Windows startup gaps repaired

- Added opt-in launcher switches for the bundled same-GGUF `NEXTN` draft and ReplaySSM spec verification. Default launch behavior remains unchanged while the experiment is unproven.
- First MTP startup loaded the 17.78 GB target successfully, then the EAGLE worker failed because `eagle_utils.py` imported `sgl_kernel`'s build-tree helper unconditionally on CUDA. Added a native-Windows fallback to the existing Triton build-tree and greedy-verification kernels.
- Native-Windows stochastic EAGLE uses exact rejection sampling plus the existing Triton top-k/top-p renormalizers; the launcher enables that mode for bundled NextN. This avoids depending on unavailable `sgl_kernel` sampling ops while retaining the target distribution.
- Second startup exposed an existing incomplete refactor: EAGLE workers imported `default_tree_mask_mode` from `eagle_utils`, while the implementation had moved into the untracked `speculative/tree_mask.py` without being re-exported. Re-exported it; a direct `eagle_worker_v2` import now passes.
- Third startup is running under launcher PID 2588 with one request, four target draft tokens, target split 16, FP32 state, and `--enable-linear-replayssm-spec`. Logs: `%TEMP%\sglang-qwen-nextn-replay3.{stdout,stderr}.log`.

### 2026-08-15 19:44:28 PDT - bundled MTP loader reaches expert weights

- Third startup reached the same-GGUF draft loader and exposed that `qwen3_5_mtp.py` could not register on Windows because its module-level `FusedMoE` import transitively required `sgl_kernel`. Made the import unnecessary for dense module registration.
- The GGUF-derived Qwen3.8 config reports `num_experts=512`, so its one-layer MTP head does need expert-name mappings. A later lazy `FusedMoE.make_expert_params_mapping` call hit the same unavailable extension.
- Replaced that dependency with the identical small mapping comprehension locally in `qwen3_5_mtp.py`; model construction already uses the working GGUF MoE path. No inference arithmetic changed.
- Sixth startup is active under launcher PID 29408. If it passes weight loading, the next checks are ReplaySSM allocation size, graph capture, deterministic greedy output, then sampled OpenCode behavior.

## Checkpoint — Sat Aug 15 19:52:42 PDT 2026

- Post-compaction checkpoint. The active experiment is bundled Qwen3.8 MTP/NextN with linear ReplaySSM speculation on native Windows; attempt 9 logs are `%TEMP%\sglang-qwen-nextn-replay9.{stdout,stderr}.log` and launch success still needs verification.
- The preceding attempt loaded both target and MTP, allocated the small ReplaySSM state, captured target-verify and draft-decode CUDA graphs, then failed when an optional DeepSeek backend imported unavailable `deep_gemm`. `eagle_worker_v2.py` now skips those unused CUDA-only optional imports on Windows.
- Baseline remains split 16, 32K active pool, max-running 2, Mamba pool 8, decode CUDA graphs [1,2], with the exact 6213/128 clean peak at 473.820 prompt tok/s and 42.320 decode tok/s. Split 32 was measured and rejected.
- Next: inspect attempt 9 narrowly, finish startup, run 256/16 correctness smoke and exact 6213/128 guarded benchmark, then either validate sampled OpenCode behavior or restore the known-good non-MTP server before stopping.

### 19:55 — MTP initialization now clears every CUDA graph phase

- Attempt 9 loaded the target in 30.62s (17.78GB) and bundled MTP in 22.92s (6.38GB), allocated 0.70GB FP32 recurrent state plus ~0.015GB ReplaySSM raw rings, and captured target-verify, draft-decode, and draft-extend graphs.
- Its auto-profiled active token pool is 25,113 tokens with one request and four Mamba slots. The final startup stop was the scheduler's blanket Windows stub for `get_draft_recurrent_hidden_state_spec`, whose real implementation only returns two model-config fields for this non-standalone EAGLE path.
- `scheduler.py` now imports that narrow helper on Windows while retaining the Windows guard for unrelated DFlash validation. Attempt 10 should establish whether any runtime-only gap remains.

### 19:57 — First native-Windows bundled MTP server is live

- Attempt 10 reached Uvicorn on port 30000 after all target and draft graph captures. Auto-profiled active token capacity is 24,282 in this run; target and draft weights plus one-request speculative state fit with ~5.6GB available before the final token-pool allocation.
- Launcher PID is 29816. Next gate is generated-output correctness under a 256/16 smoke, followed by the exact 6213/128 workload and acceptance/throughput inspection.

### 19:58 — Runtime works, but the draft currently accepts zero tokens

- The 256/16 smoke completed without an exception (256 prompt, 16 completion, SHA-256 `f15166cb...`, 20.424 decode tok/s).
- The exact 6213/128 workload also ran, but scheduler telemetry reports `accept len: 1.00, accept rate: 0.00` and only ~16.7 generated tok/s after warmup. The target path is therefore functional while the MTP predictions are misaligned or misloaded; enabling it in this state is a severe regression.
- Investigate target-hidden-state alignment and GGUF MTP expert/head mapping before spending time tuning draft step count. Keep the known-good non-MTP configuration as the recovery state.

### 20:02 — Root cause of zero acceptance found and repaired

- The custom Qwen3.5 GGUF name map enumerated only target layers 0-63. llama.cpp stores the bundled MTP as appended `blk.64.*` plus `blk.64.nextn.*`, so none of the trained MTP tensors entered `Qwen3_5ForCausalLMMTP.load_weights`; the runtime had successfully graphed an uninitialized draft.
- `build_qwen3_5_name_map` now maps appended MTP full-attention/MLP tensors into `mtp.layers.0.*`, and maps `nextn.eh_proj/enorm/hnorm/shared_head_norm` into the SGLang MTP fusion and norm parameters. Target loading remains safe because its loader already discards every `mtp` name.
- GGUF MTP now uses a quantization-aware `ColumnParallelLinear` for the quantized `eh_proj`/`fc`, and restores llama.cpp RMSNorm effective scales before loading. Focused name-map tests pass (3 tests), Python compilation passes, and the scoped diff check is clean. Attempt 11 is the trained-weight runtime test.

### 20:08 — Attempt 11 exposed draft quantization auto-detection ordering

- All 15 actual `blk.64.*` tensors are now present in the name map, but attempt 11 still created an unquantized draft: server args showed `speculative_draft_model_quantization=None`, so every quantized MTP `qweight` was correctly offered and then rejected because its module had BF16 `weight` parameters.
- Target quantization is inferred as GGUF after the server's default draft-quantization propagation point. The bundled-MTP launcher now passes `--speculative-draft-model-quantization gguf` explicitly. This should both load trained draft weights and greatly reduce its current 6.44GB footprint.

### 20:12 — Trained bundled MTP works; three steps are slower

- Attempt 12 explicitly quantized the draft as GGUF. All mapped weights loaded with no missing-parameter warnings; the draft consumes only 0.29GB instead of 6.44GB, and the requested 32,768-token active pool now fits with 8.69GB available before its allocation.
- Correctness smoke stayed deterministic at the prior 256/16 digest (`f15166cb...`). Exact 6213/128: TTFT 13.929530s, E2E 17.552118s, prompt 446.031 tok/s, decode 35.058 tok/s, 628 chars, SHA-256 `f2cbb4de...`.
- Scheduler telemetry shows useful draft quality (`accept len 2.58`, `accept rate 0.53`), proving trained MTP inference is aligned. Three draft steps still lose to the movie-contended non-MTP control (~39.9 decode tok/s), so benchmark shorter speculation before deciding.

### 20:15 — One-step MTP still loses despite 80% acceptance

- User confirmed the movie has been off for a while; treat the current short-step sweep as clean and refresh the non-MTP control afterward.
- One-step exact 6213/128: TTFT 13.673759s, E2E 17.255928s, prompt 454.374 tok/s, decode 35.453 tok/s, 646 chars, SHA-256 `e9a21f2c...`.
- First-token draft acceptance is strong (`accept len 1.80`, `accept rate 0.80`), yet quantized draft plus two-token target verification costs more than ordinary target decode. One step marginally beats three steps but remains far below the prior clean 42.320 tok/s control. Test the middle two-step point once, then restore and refresh non-MTP.

### 20:18 — Bundled MTP rejected for SGLang throughput

- Two-step exact 6213/128: TTFT 13.935107s, E2E 17.688034s, prompt 445.852 tok/s, decode 33.840 tok/s, accept length 2.17 / rate 0.59.
- Clean decode results across steps: one 35.453, two 33.840, three 35.058 tok/s. Even an 80% first draft-token acceptance rate cannot repay SGLang's GGUF draft and multi-token verification overhead on this RTX 5090 path.
- Keep the complete native-Windows MTP support opt-in for future kernel work, but do not enable it in the selected launcher defaults. Restore the retained non-MTP max-running-2 / graph-[1,2] / Mamba-pool-8 configuration and take a fresh no-movie control.

### 20:23 — Fresh no-movie control confirms the selected non-MTP path

- Restored default launcher state: active pool 32,768, chunk 4,096, split 16, FP32 Mamba, `extra_buffer_lazy`, Mamba pool 8, max-running 2, decode graphs `[1,2]`, speculation off. Server is live on port 30000 under launcher PID 29536 (Uvicorn PID in the selected-clean log).
- Three exact 6213/128 runs: TTFT 13.031638 / 13.150062 / 13.477358s; E2E 16.042335 / 16.143740 / 16.560371s; prompt 476.763 / 472.469 / 460.995 tok/s; decode 42.183 / 42.423 / 41.193 tok/s.
- Means: TTFT 13.219686s, E2E 16.248815s, prompt 470.076 tok/s, decode 41.933 tok/s. All retained the 646-character SHA-256 `0e78a559...e9ae7b5`. Best clean prompt throughput is now 476.763 tok/s and best decode 42.423 tok/s.
- Clean one-step MTP is 15.5% slower in decode than this mean; two-step is 19.3% slower; three-step is 16.4% slower. The selected server remains the fastest measured configuration.

### 20:24 — Testing the remaining local streaming overhead

- Parameterized the launcher's official `stream_interval` and `incremental_streaming_output` controls without changing their defaults. SGLang emits token 1 immediately even when the interval exceeds 1, then batches later output; interval 4 is about a 95ms UI cadence at ~42 tok/s.
- Candidate server is starting with four-token streaming and disjoint incremental output. Compare exact TTFT/E2E/decode and output digest, then run the real OpenCode path only if the synthetic result is favorable.

### 20:28 — Four-token incremental streaming is a small synthetic win

- Three exact 6213/128 runs: TTFT 13.021622 / 13.212315 / 13.141735s; E2E 16.031131 / 16.251051 / 16.157528s; prompt 477.129 / 470.243 / 472.769 tok/s; decode 42.200 / 41.794 / 42.112 tok/s. All outputs retained SHA-256 `0e78a559...e9ae7b5`.
- Means: TTFT 13.125224s, E2E 16.146570s, prompt 473.380 tok/s, decode 42.035 tok/s. Against the immediately preceding three-run interval-1 control, mean E2E improves 0.63%, prompt 0.70%, and decode 0.24%; token-one TTFT is preserved.
- The effect is small but directionally consistent. Validate the exact snapshot-disabled OpenCode wrapper turn before promotion.

### 20:29 — Streaming-path optimization promoted

- Exact `opencode_qwen.ps1 -DisableSnapshots` turn completed with exit code 0 in 23.35s. The server overlapped the 555-token title request with the 8,719-token main request and logged no scheduler, cancellation, parser, or streaming exception.
- Promoted stream interval 4 and incremental disjoint output to launcher defaults. Token 1 still streams immediately; later batches arrive roughly every 95ms at current decode speed. `-DisableIncrementalStreamingOutput` and `-StreamInterval 1` preserve an explicit compatibility escape hatch.
- This is the only newly measured default win after the MTP sweep: about 0.6% mean end-to-end on the exact synthetic workload, with larger expected serialization savings on long OpenCode responses because cumulative output is no longer recopied between processes on every token.

### 20:33 — Final validation checkpoint

- Live endpoint remains the promoted non-MTP configuration at `http://127.0.0.1:30000/v1`, with resolved `stream_interval=4` and `incremental_streaming_output=True`.
- Focused regression suite: 19 tests passed plus 4 subtests (14 known Torch JIT deprecation warnings). All touched Python files compile, native-Windows EAGLE and scheduler imports pass directly, both PowerShell scripts parse, and `git diff --check` is clean.
- Default behavior stays non-speculative. Bundled NextN and linear ReplaySSM speculation remain explicit switches; their trained GGUF path is functional and documented, while the measured loser stays out of normal OpenCode service.

## Checkpoint — Sat Aug 15 20:32:50 PDT 2026 — NVFP4 direction

- User identified `unsloth/Qwen3.8-27B-NVFP4` as the intended Blackwell checkpoint. Preserve the live GGUF server and its fresh measurements as the apples-to-apples control while verifying the checkpoint contents, SGLang loader/kernel requirements, local availability, and disk/VRAM fit.
- Current promoted GGUF control remains live on port 30000: exact 6213/128 mean 473.380 prompt tok/s, 42.035 decode tok/s, 16.146570s E2E with four-token incremental streaming. Next: inspect the referenced model, choose the native NVFP4 load path, acquire it locally if absent, then benchmark the same harness and OpenCode wrapper.

## Post-compaction checkpoint — Sat Aug 15 20:35:09 PDT 2026

- Resumed on the NVFP4 pivot. The clean GGUF control remains live and must stay available until the NVFP4 checkpoint is fully downloaded and ready to launch.
- Metadata already acquired at `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4`: the target is a compressed-tensors `nvfp4-pack-quantized` checkpoint with one separate 849.4 MB MTP safetensors file; total download is about 23.5 GB. RTX 5090 is SM120 with 32,607 MiB VRAM, PyTorch 2.13.0+cu130, CUDA 13.0, and driver 610.88.
- Next: establish the exact SGLang compressed-tensors NVFP4 loader and usable native-Windows FP4 runner, start the full download, add an isolated NVFP4 launcher, then run the same 6213/128 and OpenCode measurements against the GGUF mean of 473.380 prompt tok/s, 42.035 decode tok/s, and 16.146570s E2E.

### 20:36 — NVFP4 acquisition started

- Full public checkpoint download is running into `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4` under exec session 95219. The CLI is unauthenticated and warned only about lower Hub rate limits.
- Config confirms a mixed compressed-tensors export: MLPs are `nvfp4-pack-quantized` W4A4 with 16-value groups; attention, linear-attention projections, lm_head, and the final eight layers' MLPs use float8 W8A8; MTP is explicitly excluded and stored separately.

### 20:39 — NVFP4 checkpoint acquired; kernel dependency isolated

- Download session 95219 completed successfully. `model.safetensors` is 22,568,192,096 bytes and `model_mtp.safetensors` is 849,400,392 bytes; the local checkpoint is complete.
- This SGLang checkout recognizes the checkpoint automatically through `CompressedTensorsW4A4Fp4` on SM120. Its dense FP4 path currently requires FlashInfer's `fp4_quantize` and `mm_fp4` symbols.
- The active native-Windows environment has no `flashinfer` or `sgl_kernel`; PyTorch is CUDA 13.0/SM120 and exposes native `float4_e2m1fn_x2` plus the new NV block-scaled `torch._scaled_mm_v2`, while cuDNN is 9.2.0. The immediate task is to supply the Windows FlashInfer FP4 subset or wire a native-Torch fallback, then launch without changing the GGUF control script.

### 20:56 — Native-Windows SM120 NVFP4 kernel path passes

- Acquired `SystemPanic/flashinfer-windows` at commit `713358284345314df4f40ddc352f4e981f5bb03e` and installed FlashInfer 0.6.11.post3 into the SGLang venv without replacing Torch or other dependencies.
- Added a source-local CUDA 13.3/MSVC workaround in the FlashInfer checkout: its JIT now shadows only `cuda.h` with a Windows wrapper that gives `CUtensorMap` the 64-byte alignment MSVC can pass by value. NVIDIA's system headers remain untouched. Added the nv_internal wrapper include to all FlashInfer JIT modules.
- Added `scripts/windows/initialize_cuda_build_env.ps1` to initialize VS 2026, CUDA 13.3, and SM120-only JIT variables, plus `smoke_flashinfer_nvfp4.ps1` for a reproducible numerical check.
- The FlashInfer activation quantizer and SM120 CUTLASS dense FP4 GEMM both compiled. Smoke result: finite 4x128 BF16 output, relative MAE 0.148438 versus BF16 reference. Kernel acquisition is complete; next is an isolated NVFP4 server launcher and real checkpoint startup.

## Post-compaction checkpoint — Sat Aug 15 21:07:46 PDT 2026

- The full `unsloth/Qwen3.8-27B-NVFP4` checkpoint and native-Windows SM120 FlashInfer FP4 kernels are ready. The reproducible quantize-plus-GEMM smoke passes; the prior GGUF server was intentionally stopped and port 30000 is free for the NVFP4 launch.
- Added an isolated launcher at `scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1` and native-Windows text-only Qwen3.5/Qwen3.8 wrapper support. Direct model-class import and launcher help parsing passed.
- First server attempt exited before loading weights because the Windows quantization registry exposed only `gguf`: `ValueError: Unknown quantization method: compressed-tensors`. The latest unvalidated edits add `CompressedTensorsConfig` to the Windows registry and give the unused INT8 scheme a `torch._scaled_mm` fallback when `sgl_kernel` is absent.
- Immediate next step: validate the compressed-tensors registry/import and modified-file compilation, repair only any proven optional-import boundary, then relaunch and inspect startup logs selectively. Once live, run smoke, three exact 6213/128 measurements, and compare with the clean GGUF control of 473.380 prompt tok/s, 42.035 decode tok/s, and 16.146570s E2E.

### 21:09 — Compressed-tensors registry validation passes

- Native-Windows quantization discovery now reports exactly `gguf` and `compressed-tensors`. The modified quantization registry, INT8 import fallback, Qwen text-only wrapper, and server arguments all pass `py_compile`; `git diff --check` is clean.
- The registry failure from the first launch is resolved. Relaunching the NVFP4 checkpoint is now the active step.

### 21:11 — Dense loader passes the optional MoE boundary

- The second launch reached Qwen model construction, then an unconditional `FusedMoE` type import pulled in the optional Triton MoE runner and failed on missing `sgl_kernel`. This checkpoint is dense; no MoE execution is involved.
- `CompressedTensorsConfig.get_quant_method` now gates the lazy `FusedMoE` import behind an MRO-name check. Dense layers return without importing the unavailable runner, while actual `FusedMoE` instances and subclasses retain the existing path. The edit compiles and `git diff --check` remains clean.

### 21:13 — Weights load; FP8 JIT reaches a Windows command-generation fault

- The next launch completed model construction and loaded the two checkpoint shards far enough to begin CUDA-graph capture. The mixed checkpoint then requested `sgl_kernel_jit_per_token_quant_fp8_precise_math_bf16_t` for its W8A8 layers.
- SGLang's native JIT emitted a Ninja command with POSIX single quotes around `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc` and include paths, plus `-Xcompiler -fPIC`. Windows failed at process creation before compilation: `CreateProcess failed: The system cannot find the file specified.`
- The failed process exited cleanly and GPU memory returned to the baseline (~2.75 GiB used). Next: repair Windows command emission in the SGLang JIT builder, validate the one kernel directly, then relaunch from the cached model files.

### 21:19 — Native-Windows SGLang FP8 JIT passes

- Made the content-addressed SGLang JIT toolchain platform-aware: Windows command quoting, `cl` flags/link syntax, CUDA `lib\x64`/`cudart.lib`, NVCC dependency emission, and an MSVC version fingerprint now work without weakening the existing Unix path.
- Added the missing standard `<bit>` include for `std::has_single_bit`; MSVC exposed the previously transitive include dependency.
- Added `scripts/windows/smoke_sglang_fp8_jit.ps1`. The actual kernel that blocked graph capture now compiles, links, loads, and numerically passes on SM120: FP8 output `(4, 128)`, scale `(4, 1)`, finite reconstruction, relative MAE `0.022174`. A second run also passed from the published cache with no compiler warning.
- Next: relaunch the checkpoint. Further first-use SGLang JIT kernels may compile during warmup; each will now use the native-Windows toolchain.

### 21:24 — Mixed-checkpoint FP8 linear fallback passes

- The relaunch loaded all weights in 15.92s: compressed-tensors model memory 23.32 GB, 6.82 GB then free; a 4-entry FP32 Mamba pool and 16,384-token BF16 KV pool left about 5.03 GB before graph capture.
- Graph warmup reached the first channelwise W8A8 projection and found `fp8_scaled_mm` undefined because the Windows fork had skipped the AOT `sgl_kernel` import without supplying its call-site fallback.
- Added a Windows implementation using SM120's native scalar-scale `torch._scaled_mm` with FP32 output, followed by rowwise/channelwise dequant scaling and BF16 conversion. This mirrors the existing generic fallback while retaining the fast FP8 tensor-core multiplication; bias is applied after scaling.
- Expanded the FP8 smoke to cover quantization plus channelwise GEMM. It passes with finite `(4, 256)` output and relative MAE `0.037598` against BF16 (quantizer-only MAE `0.022174`). Relaunching again is the active step.

### 21:23 — NVFP4 server is live

- The base checkpoint is now serving at `http://127.0.0.1:30000` as `qwen3.8-27b`. Decode CUDA-graph capture completed in 36.29s, used 0.11 GB, and left about 5.12 GB available; the server reported ready.
- A calibrated 256-input/16-output correctness smoke completed with `finish_reason=length`: observed prompt 2027.581 tok/s, decode 44.936 tok/s, 0.460069s E2E. This is a short functional smoke, not the comparison result.
- Next: exact 6213-input/128-output run, followed by clean repeats and tuning against the GGUF control.

### 21:24 — Exact NVFP4 control beats GGUF decisively

- Three clean `--skip-warmup` runs at the exact calibrated 6213-input/128-output shape produced the same output hash and `finish_reason=length` every time.
- Clean NVFP4 mean: prompt `6072.305 tok/s`, decode `44.918 tok/s`, E2E `3.851374s`, TTFT `1.023292s`, and end-to-end output `33.238 tok/s`.
- Clean GGUF mean: prompt `473.380 tok/s`, decode `42.035 tok/s`, E2E `16.146570s`.
- NVFP4 is therefore about `12.83x` GGUF prompt throughput, `6.86%` faster decode, and `4.19x` faster end-to-end (`76.15%` less wall time) on the controlled single-stream workload. The initial warmed run was slightly faster again at 6576.812 prompt / 45.169 decode / 3.756352s E2E.
- The performance goal has a strong winning baseline. Next: isolate remaining decode overhead, especially the Windows channelwise-FP8 fallback's separate dequant-scale/cast work, then tune conservatively without losing this validated configuration.

### 21:43 — Fused FP8 epilogue candidate passes

- A direct port of SGLang's fused SM120 CUTLASS rowwise/channelwise FP8 GEMM compiled as a native-Windows Torch extension, including against the exact pinned CUTLASS commit `57e3cfb47a2d9e0d46eb6335c3dc411498efa198`, but every tested launch ended in `cudaErrorMisalignedAddress`. Input, weight, and both scale pointers were all 128-byte aligned. The live validated server was untouched throughout this experiment; the fused CUTLASS candidate is not enabled.
- Replaced the fallback's three eager post-GEMM operations (row scale, channel scale, BF16 cast) with one Triton epilogue kernel. The existing Windows FP8 quantization+GEMM smoke still matches exactly: finite `(4, 256)` output, relative MAE `0.037598`.
- Next: restart once onto the fused epilogue and repeat the exact benchmark. The prior live configuration and its 6072.305 prompt / 44.918 decode / 3.851374s E2E mean are the rollback control.

### 21:50 — Post-compaction checkpoint

- Checkpoint time: `Sat Aug 15 21:50:42 PDT 2026`.
- The validated base NVFP4 control remains `6072.305 tok/s` prompt, `44.918 tok/s` decode, and `3.851374s` E2E over three exact 6213-input/128-output clean runs, versus the GGUF control at `473.380 / 42.035 / 16.146570s`.
- The replacement server carrying the fused Triton FP8 scale/cast epilogue is parent PowerShell PID `19400`; logs are `%LOCALAPPDATA%\Temp\sglang-qwen-nvfp4-epilogue.stdout.log` and `sglang-qwen-nvfp4-epilogue.stderr.log`. At compaction it had reached target decode CUDA-graph capture and was still alive.
- Immediate continuation: confirm capture/readiness, run the 256/16 smoke, then one warmed and three clean exact 6213/128 measurements. Retain the epilogue only if the controlled mean improves without changing output correctness.
- The direct CUTLASS fused experiment remains disabled after repeatable `cudaErrorMisalignedAddress`; its experimental files require careful cleanup after the live Triton comparison.

### 21:52 — Triton FP8 epilogue improves end-to-end latency

- Restart completed cleanly: decode CUDA-graph capture ended in 10.10s and the server became ready at 21:50:00. The 256/16 smoke completed with finite output and `finish_reason=length`.
- Exact warmed 6213/128 run: prompt `7417.724 tok/s`, decode `44.624 tok/s`, E2E `3.683581s`, TTFT `0.837588s`; output hash remained `fd160d318c65695aa00fe9e2868e5053df785ed2d2f730dc0afc987fd5493da3`.
- Three exact clean runs: `(6549.323, 44.652, 3.792835s)`, `(7206.070, 44.904, 3.690418s)`, `(7440.897, 45.089, 3.651626s)` for prompt/decode/E2E. Mean: prompt `7065.430 tok/s`, decode `44.882 tok/s`, E2E `3.711626s`, TTFT `0.881939s`.
- Against the prior NVFP4 control mean, this is `16.36%` more prompt throughput and `3.63%` less E2E wall time; decode is effectively flat (`-0.08%`). All three outputs are identical and length-terminated.
- The rising prompt sequence indicates residual warm-state variance, so retain the epilogue provisionally and gather a longer stable sample before declaring its exact gain.

### 21:53 — Extended epilogue sample exposes host-side variance

- Five additional exact runs produced mean prompt `7098.376 tok/s`, decode `42.726 tok/s`, and E2E `3.850584s`. Four decode samples clustered at 41.756-42.567 tok/s, while the fifth returned to `45.089 tok/s`; output remained byte-identical in every run.
- A monitored follow-up held the GPU at 98-99% SM utilization and about 3.0 GHz during inference and measured `5903.516 / 44.445 / 3.909867s`. The short request's host/clock startup changes TTFT substantially; the CUDA work itself saturates the GPU.
- The epilogue's prompt-side advantage is repeatable around 7.1K tok/s once hot, but its decode path remains statistically indistinguishable from the earlier fallback. Keep it because it reduces kernels and improves the hot path, while treating `3.65s` as the observed best and the longer mean as the honest variance-inclusive result.

### 21:55 — Decode-specialized epilogue restart

- Added a compile-time single-row Triton branch that removes per-element integer row/column division during batch-one decode; the multi-row prefill branch is unchanged.
- Expanded the FP8 smoke to exercise both shapes. Multi-row remains finite at relative MAE `0.037598`; single-row is finite at relative MAE `0.036133`.
- Restarted the server under parent PID `35416`. Logs: `%LOCALAPPDATA%\Temp\sglang-qwen-nvfp4-single-row.stdout.log` and `sglang-qwen-nvfp4-single-row.stderr.log`. Next: wait for capture, then repeat identical correctness and performance guards.

### 21:57 — Single-row specialization is the new retained winner

- Restart and graph capture passed, followed by the 256/16 smoke and exact warmed guard. Output hashes and length termination remain unchanged.
- Five clean exact 6213/128 runs were stable: prompt `7028.426, 7540.274, 7468.722, 7407.050, 7367.059`; decode `44.846, 44.610, 44.743, 44.888, 44.764`; E2E `3.715915, 3.670873, 3.670324, 3.668052, 3.680435s`.
- Five-run mean: prompt `7362.306 tok/s`, decode `44.770 tok/s`, E2E `3.681120s`, TTFT `0.844394s`, E2E output `34.773 tok/s`.
- Versus the original NVFP4 control, this is `21.25%` higher prompt throughput and `4.42%` less E2E wall time, with decode within `0.33%`. Versus GGUF, prompt is now `15.55x` and total wall time is `4.39x` faster (`77.20%` less time).
- Retain the fused epilogue and its batch-one specialization. Live selected server: parent PID `35416`, port 30000.

### 21:59 — NVFP4 chunk-8192 control started

- The retained chunk-4096 control is the stable five-run mean `7362.306 / 44.770 / 3.681120s` for prompt/decode/E2E.
- Started an otherwise identical `ChunkedPrefillSize=8192` server under parent PID `33760`; logs are `%LOCALAPPDATA%\Temp\sglang-qwen-nvfp4-chunk8192.stdout.log` and `sglang-qwen-nvfp4-chunk8192.stderr.log`.
- This tests whether placing the 6213-token prompt in one NVFP4 prefill chunk now beats the two-chunk schedule. Restore 4096 unless repeat measurements win.

### 22:02 — Chunk 8192 rejected again

- Five clean 8192-chunk runs mean prompt `6847.911 tok/s`, decode `43.278 tok/s`, E2E `3.846870s`, and TTFT `0.911138s`.
- Relative to the retained 4096 control, prompt falls `6.99%`, decode falls `3.33%`, and E2E wall time rises `4.50%`. The one-chunk schedule does less scheduler work but feeds the model kernels less efficiently.
- The 8192 candidate is deterministic within its run set but follows a different greedy path (SHA-256 `19ad91d0...42541b`, 832 chars) than 4096 (`fd160d31...49da3`, 835 chars), consistent with chunk-boundary numerical differences.
- Reject 8192 and retain 4096. Next isolated axis: FlashInfer paged attention versus the current Triton attention backend now that the Windows FlashInfer port is installed.

### 22:04 — FlashInfer attention control started

- Parameterized the NVFP4 launcher attention backend while retaining Triton as its default. PowerShell help parsing passes.
- Started the isolated `AttentionBackend=flashinfer`, chunk-4096 candidate under parent PID `14720`; logs are `%LOCALAPPDATA%\Temp\sglang-qwen-nvfp4-flashinfer-attn.stdout.log` and `sglang-qwen-nvfp4-flashinfer-attn.stderr.log`.
- Compare against the retained Triton mean `7362.306 / 44.770 / 3.681120s`; output and startup guards apply before timing.

### 22:07 — FlashInfer paged attention wins, with a version-gate caveat

- The installed Windows port reports FlashInfer `0.6.11.post3`, while this SGLang checkout requires `>=0.6.17` whenever FlashInfer attention is selected. A first attempt stopped at that explicit gate. A process-local `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` control then loaded, captured graphs, and served without kernel/runtime errors; the pinned Windows fork's main branch still reports 0.6.11.post3.
- Five clean exact runs mean prompt `7630.280 tok/s`, decode `45.517 tok/s`, E2E `3.605032s`, TTFT `0.814445s`, and E2E output `35.512 tok/s`.
- Versus the retained Triton attention mean, FlashInfer raises prompt `3.64%`, raises decode `1.67%`, and cuts E2E wall time `2.07%`. All five outputs are deterministic (`8d2ad329...405cff`, 918 chars, length-terminated), though numerical differences choose a different greedy path than Triton.
- FlashInfer attention is the measured performance winner. Keep it provisional until a real chat/OpenCode correctness probe passes; production launch must make the exact version-check exception explicit because silently relabeling the package would be inaccurate.

### 22:13 — Real OpenCode probe exposes runaway hidden reasoning

- A standalone OpenCode2 request for a two-word answer reached the NVFP4 server with about 8.7K prompt tokens and sustained roughly 43-46 generated tok/s, proving the full OpenCode transport and long-context kernel path are live.
- The request generated only hidden reasoning until the 16,384 active-token pool filled, then exited with no visible final answer. A second run selecting the existing non-thinking title alias behaved the same way and was interrupted after more than 4K generated tokens; its intended `enable_thinking=false` / 32-token body settings were evidently not reaching the server on this run path.
- This is now a request-configuration/parser issue, not a throughput or CUDA failure. Inspect OpenCode2's resolved provider body and SGLang request fields, then make a bounded non-thinking probe work before promoting NVFP4 for daily use.

### 22:20 — Bounded direct chat passes; OpenCode request controls need capture

- A new bounded local chat probe sent `enable_thinking=false`, returned `NVFP4 READY` exactly in 5 completion tokens, stopped normally, and reported zero reasoning tokens. This confirms the NVFP4 checkpoint, template, reasoning parser, and FlashInfer-attention output path are coherent.
- Found and fixed one wrapper defect: `ConvertTo-Json -Depth 6` was too shallow for nested `chat_template_kwargs`; it is now depth 12. Added a late-applied `#nothink` variant as an additional guard. OpenCode still ran away, so another control is overriding or omitting those fields.
- Added a process-gated `SGLANG_LOG_CHAT_CONTROLS=1` diagnostic that logs only max-token and reasoning controls, never messages. Restart once with it, capture the actual OpenCode request controls, fix the proven boundary, then remove the diagnostic.

### 22:25 — OpenCode controls fixed; FlashInfer long-prompt quality is suspect

- The narrow server diagnostic confirms the repaired overlay reaches SGLang exactly: title request has `max_completion_tokens=32` and `chat_template_kwargs={'enable_thinking': False, 'preserve_thinking': False}`. The earlier unbounded request had no max-token field, explaining why it could fill the active pool.
- An interrupted client did not cancel the already-scheduled generation: the orphan consumed the GPU to 16,384 tokens while its output state was deleted, and queued the bounded probe behind it. This is a separate cancellation-efficiency defect to address.
- Once scheduled, the bounded 8,673-prompt-token OpenCode probe completed in 32 tokens but returned repetitive text (`...you could have been told...`) instead of the requested two words. A 23-token direct prompt on the same server remains exact and coherent.
- The next decisive control is the identical bounded OpenCode request on Triton attention. If Triton is coherent, reject the faster FlashInfer 0.6.11 attention path as numerically incompatible at real context lengths; if both fail, the checkpoint/template itself needs comparison with another NVFP4 export.

### 22:27 — FlashInfer 0.6.11 attention rejected on quality

- Identical bounded, non-thinking OpenCode controls now reach SGLang with `max_completion_tokens=32`; the wrapper depth defect is fully repaired.
- FlashInfer attention at the real 8.7K prompt produced a degenerate repetition loop. Triton attention produced a coherent response immediately on the same prompt and controls (`Let me verify that claim...`), even though the agent's system instructions made it decline the requested exact wording.
- This clean A/B localizes the quality defect to using the below-minimum FlashInfer 0.6.11 attention implementation at real context length. Reject that path despite its 2.07% synthetic speed advantage. Keep Triton attention as the production backend and keep SGLang's FlashInfer >=0.6.17 gate intact.
- The valid retained performance mean remains the Triton result `7362.306 prompt / 44.770 decode / 3.681120s E2E`; it is 4.39x faster end-to-end than GGUF and passes both bounded direct and real OpenCode long-prompt coherence guards.

### 22:31 — Thinking-only acceptance path resumed

- Removed the temporary main-model `#nothink` diagnostic variant. The direct probe now defaults to thinking enabled; future acceptance measurements use the user's real thinking/tool path.
- The first real thinking request had about 8.7K prompt tokens plus an advertised 8,192-token output budget, which cannot fit inside the conservative 16,384-token active pool. The server stopped at pool exhaustion before a final answer.
- Started an otherwise identical Triton server with a 32,768-token active pool under parent PID `34364`; logs are `%LOCALAPPDATA%\Temp\sglang-qwen-nvfp4-pool32768.stdout.log` and `sglang-qwen-nvfp4-pool32768.stderr.log`. This costs only one additional GiB of KV cache and should still leave over 4 GiB free.
- Next: verify startup/memory and unchanged synthetic speed, then run the actual OpenCode thinking stream with reasoning visible and its full configured output allowance.

### 22:32 — Post-compaction checkpoint

- Checkpoint time: `Sat Aug 15 22:32:42 PDT 2026`.
- Active candidate: native NVFP4, Triton attention, `4096` chunked prefill, `32768` total-token pool, server parent PID `34364` on port `30000`.
- Current 32K-pool exact 6213/128 five-run mean: prompt `6999.371 tok/s`, decode `42.336 tok/s`, E2E `3.887727 s`, TTFT `0.887928 s`.
- Relative to the retained 16K-pool Triton result: prompt `-4.93%`, decode `-5.44%`, E2E time `+5.61%`.
- A monitored follow-up held `98-99%` SM utilization, `13801 MHz` memory, `3000 MHz` core, and roughly `381-447 W`; the regression occurred while the GPU was saturated and fully clocked.
- FlashInfer attention remains rejected: its synthetic mean was faster, but the installed Windows `0.6.11.post3` path produced degenerate repetition on the real 8.7K OpenCode prompt. Triton produced coherent output for the same bounded prompt.
- Acceptance remains thinking-only with visible reasoning and tools intact. The next measured control is a smaller active pool (`20480`, then `24576` only if needed) that still fits the observed 8.7K prompt plus the configured 8192-token output allowance.
- After choosing the smallest speed-preserving pool, run the actual OpenCode thinking stream, investigate abort propagation for orphaned generation, remove temporary diagnostics and the failed direct-CUTLASS experiment, and revalidate the focused diff.

### 22:36 — A 20K active pool does not recover the earlier decode rate

- Restarted the same retained Triton-attention/4096-chunk candidate with `--max-total-tokens 20480`. Startup was clean: weights ended in `18.71 s`, the memory pool left `5.45 GB` available, decode graph capture ended in `8.98 s`, and the server became ready.
- The 256/16 smoke passed. Five exact 6213/128 runs retained the deterministic hash `fd160d318c65695aa00fe9e2868e5053df785ed2d2f730dc0afc987fd5493da3`.
- Five-run mean: prompt `6819.370 tok/s`, decode `41.730 tok/s`, E2E `3.955196 s`, TTFT `0.911462 s`.
- Per-run decode was `42.038, 40.947, 41.441, 42.083, 42.143 tok/s`; the slowdown is repeatable enough that a larger-pool-only explanation is unproven.
- Next control: immediately restart at the original `16384` pool and repeat the same five-run measurement. If it also remains near 42 tok/s, treat the earlier 44.770 mean as a different runtime/power state and choose the smallest pool that safely fits thinking by capacity and current matched-A/B behavior.

### 22:38 — Matched 16K A/B rules out a material pool-size regression

- Restarted at the original `16384` pool and repeated five exact 6213/128 runs immediately. Mean: prompt `6960.166 tok/s`, decode `42.243 tok/s`, E2E `3.899293 s`, TTFT `0.892860 s`; all five hashes matched the retained deterministic output.
- Against the immediately preceding 20K mean, 16K was only `1.23%` faster in prompt and decode and `1.41%` faster E2E. The current 32K mean was also within this band (`42.336 tok/s` decode, `3.887727 s` E2E).
- Therefore active-pool size is not the source of the roughly 5% difference from the earlier `44.770 tok/s` measurement. The prior rate is currently unreproduced runtime variance; current matched controls cluster near `42.2-42.3 tok/s`.
- Retain a `32768` active pool for the configured OpenCode contract: it supports the observed 8.7K thinking prompt plus 8192 output tokens and avoids a capacity failure as the real workspace prompt grows. The matched measurements show no meaningful speed cost versus 16K.
- Next: restore 32K, run a real thinking-enabled OpenCode stream with visible reasoning, then test the native separate MTP head as the remaining high-value decode lever.

### 22:45 — Real thinking OpenCode run exposes an unbounded-generation failure

- Restored the retained 32K Triton server; startup completed with `4.71 GB` available after pool allocation and `4.56 GB` after decode-graph capture.
- Ran OpenCode2 standalone with `--thinking`, the main `llama-cpp/qwen3.8-27b` model, tools enabled, and a real request to read the NVFP4 launcher. The prompt was roughly 8.7K tokens and decode CUDA graphs stayed active around `38-40 tok/s` as context grew.
- OpenCode emitted only `step_start`; its reasoning part remained an empty buffered field in the persisted session while generation was active. At 300 seconds the client returned `Transport` and deleted its TokenizerManager state.
- SGLang then continued the request as an orphan, repeatedly logging `Received output ... but the state was deleted in TokenizerManager`. At the latest sample the full sequence had reached `20,973` tokens, meaning well over 12K generated tokens for a trivial file read.
- This confirms the main OpenCode request did not supply an effective completion cap; its configured provider `limit.output = 8192` is metadata rather than an OpenAI request field on this path. The exact reasoning text cannot be recovered because OpenCode never committed the buffered part and SGLang does not log generated text.
- Immediate remediation: terminate the orphan by restarting the local server, add an opt-in explicit `max_completion_tokens` overlay for the main thinking model, rerun the same request with thinking and visible reasoning, then fix disconnect-to-abort propagation separately.

### 22:51 — Bounded reasoning proves deterministic repetition before any tool call

- Added a process-scoped `qwen3.8-27b-thinking-capped` OpenCode alias; it preserves thinking and tools while supplying an explicit `max_completion_tokens`. A 512-token run now ends cleanly and exposes the buffered reasoning.
- Greedy output began coherently, then repeated `where you can verify it step by step while you're looking through it one more time before moving into analysis mode` for the remainder of the cap. The model never emitted a file-read tool call.
- Replayed with Qwen's official sampled profiles: precise coding (`temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=0.0`) and general thinking (`temperature=1.0`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`). Both OpenCode runs repeated `right now` through the full 512-token cap; the latter two outputs were byte-identical.
- A separate direct 256-token thinking probe forced greedy decoding and independently repeated `analysis should include answer content` for nearly the entire response. This confirms the failure is visible outside OpenCode's large tool prompt as well.
- Because standard OpenCode sampling fields may override model-body defaults, the SGLang request-control diagnostic now records effective temperature, top-p, top-k, min-p, presence penalty, and repetition penalty. Restart and inspect those values before attributing the sampled failures to the checkpoint/runtime.

### 23:00 — RadixArk checkpoint fixes quality and raises speed to 49.67 tok/s

- Found the complete alternate checkpoint at `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk`. Its included qualification reports a source-audited ModelOpt mixed NVFP4 conversion, 97.27% GSM8K thinking accuracy, 1,319/1,319 clean stops, zero request errors/truncations, and BF16 source-identical MTP.
- Added narrow Windows registration for `ModelOptMixedPrecisionConfig`. The module already imported successfully on Windows and has Windows-safe dense-model boundaries; registry validation now reports `compressed-tensors`, `gguf`, and `modelopt_mixed`.
- FlashInfer `0.6.11.post3` exposes an older `autotune()` signature without `skip_ops`; added a launcher switch to pass `--disable-flashinfer-autotune`. With it, RadixArk loaded successfully: `19.25 GB` model usage, 32K FP8 KV cache using `1.00 GB`, `9.12 GB` available after pool allocation, and `8.94 GB` after decode graph capture.
- Direct thinking isolation with the official general profile (`temperature=1.0`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`) stopped after 28 tokens and returned coherent reasoning plus final answer `4`. This conclusively isolates the repetition defect to the Unsloth checkpoint rather than OpenCode or the shared SGLang/kernel path.
- Five exact 6213/128 runs on RadixArk: prompt `8363.351, 8341.303, 8283.999, 8354.172, 8283.892 tok/s`; decode `49.519, 49.645, 49.724, 49.629, 49.818 tok/s`; E2E `3.307574, 3.303033, 3.304102, 3.302709, 3.299284 s`.
- RadixArk five-run mean: prompt `8325.343 tok/s`, decode `49.667 tok/s`, E2E `3.303340 s`, TTFT `0.746288 s`. Against the matched Unsloth 32K mean, this is `+18.94%` prompt, `+17.32%` decode, and `15.03%` less E2E time.
- Real OpenCode2 standalone acceptance passed with thinking visible and tools intact. It reasoned coherently, called the `read` tool, cited the correct launcher lines, and answered the requested defaults. The SGLang boundary confirmed the effective sampled settings and explicit 1024-token cap on both agent turns.
- Next: make RadixArk/32K/old-FlashInfer-autotune-disable the launcher defaults, benchmark the checkpoint's bundled BF16 MTP on one GPU, then remove diagnostics and fix or bound the uncapped OpenCode production path.

### 23:06 — Three-step bundled MTP reaches 62.64 tok/s and passes OpenCode

- Promoted launcher capacity/checkpoint defaults to RadixArk and 32K, default-disabled unsupported FlashInfer autotuning, and added optional NEXTN controls with Triton draft attention plus linear ReplaySSM verify.
- RadixArk's bundled MTP loaded as `Qwen3_5ForCausalLMMTP` in `1.76 s` and used about `6.20 GB` on this single-GPU topology. Target verify, draft decode, and draft extend CUDA graphs all captured; the server remained within the 32 GB card.
- Native Windows sampled EAGLE correctly required `--speculative-use-rejection-sampling`; after adding it, the direct official-profile thinking smoke remained coherent and stopped normally with final answer `4`.
- Five exact 6213/128 MTP runs: prompt `7378.918, 7864.456, 7714.023, 7826.654, 7855.241 tok/s`; decode `62.092, 56.537, 60.410, 67.409, 66.745 tok/s`; E2E `2.887336, 3.036308, 2.907734, 2.677849, 2.693705 s`.
- MTP mean: prompt `7727.858 tok/s`, decode `62.639 tok/s`, E2E `2.840586 s`, TTFT `0.804436 s`. Against native RadixArk without MTP, decode improved `26.12%` and E2E time fell `14.01%`; prompt prefill fell `7.18%` because the speculative worker adds overhead around extend.
- Logged acceptance length ranged `2.27-2.62` with acceptance rate `0.42-0.54`, closely tracking the checkpoint's included qualification evidence (`2.775` final acceptance length on TP4).
- Real OpenCode thinking/tool acceptance also passed under three-step MTP: coherent visible reasoning, successful file-read call, a correct final answer, and clean stops on both agent turns.
- Next: measure one-step and, only if competitive, two-step MTP to choose the fastest single-GPU setting; then make it default and bound the production OpenCode model request.

### 23:09 — One-step MTP rejected; three-step is the selected decode path

- Repeated the same five-run 6213/128 control with one MTP step and two draft tokens. Mean: prompt `7766.049 tok/s`, decode `48.812 tok/s`, E2E `3.404566 s`, TTFT `0.800883 s`.
- One-step decode was `22.07%` slower than the three-step MTP mean and `1.72%` slower than the non-speculative RadixArk baseline. Draft-model overhead exceeds the value of its one-token proposal on this single GPU.
- One-step is decisively noncompetitive, so the planned two-step follow-up is skipped. Select the checkpoint-qualified three-step/four-token chain at `62.639 tok/s` decode.
- Next: set three steps as the launcher default, make the wrapper's normal main model send the recommended sampling profile and an explicit 8192-token cap, remove temporary diagnostics/failed experiments, relaunch from defaults, and run final thinking plus benchmark validation.

### 23:18 — Post-compaction checkpoint

- Checkpoint time: `Sat Aug 15 23:18:23 PDT 2026`.
- Selected serving path: RadixArk ModelOpt NVFP4 at 32K with bundled BF16 MTP, three speculative steps, four draft tokens, top-k 1, rejection sampling, Triton draft/verify attention, and linear ReplaySSM speculative verification.
- Selected five-run 6213/128 mean: prompt `7727.858 tok/s`, decode `62.639 tok/s`, E2E `2.840586 s`, TTFT `0.804436 s`. This is `+26.12%` decode and `14.01%` less E2E time than RadixArk without MTP; versus matched Unsloth it is about `+48.0%` decode and `26.9%` less E2E time.
- Checkpoint quality: RadixArk passes coherent direct thinking and real OpenCode thinking/tool tests. The Unsloth checkpoint repeats in thinking mode under the exact official sampling profile, including outside OpenCode.
- Launcher defaults now select `models\Qwen3.8-27B-NVFP4-RadixArk`, 32K, and the selected three-step/four-token MTP path. The OpenCode wrapper's normal main model sends the official thinking sampling controls with an explicit `8192`-token production cap.
- Cancellation cleanup is patched in `tokenizer_manager.py`: cancelled or closed streaming requests now abort scheduler work before local request-state cleanup. The focused cleanup test file passes all `19` tests.
- Final default server was launched as PowerShell PID `23864`; at compaction it was still loading from defaults into `sglang-qwen-nvfp4-radixark-final.stderr.log`.
- Remaining acceptance: confirm final server readiness, run direct thinking smoke, live client-cancellation integration, normal-main OpenCode thinking/tool acceptance, a clean final default benchmark, focused static/diff checks, then leave the validated server live.

### 23:26 — Final RadixArk/MTP production acceptance complete

- The untouched default launcher completed at `23:17:27`: target ModelOpt NVFP4 used `19.25 GB`, bundled MTP used `6.37 GB`, the 32K memory pools initialized, and target-verify/draft-decode/draft-extend CUDA graphs all captured. PowerShell parent PID `23864` remains live.
- Direct official-profile thinking passed repeatedly after the final restart (`2+2 -> 4`, `3+3 -> 6`, final health `5+5 -> 10`), each with coherent `reasoning_content`, a normal stop, and a separate final answer.
- Live client-cancellation integration passed. A deliberately long 8192-token stream was interrupted after ten seconds. One already-in-flight scheduler packet crossed the teardown boundary and was ignored, no repeated orphan-output storm followed, and an immediate new thinking request completed normally. This validates the scheduler-abort-before-local-cleanup fix against the original failure mode.
- Normal production OpenCode model acceptance passed through the standalone wrapper with `MainOutputCap=512`: visible coherent reasoning, a successful real `read` tool call, retained tool output, and the correct one-sentence final answer. The ordinary `llama-cpp/qwen3.8-27b` alias was used, rather than the diagnostic alias.
- Final restart's first five exact 6213/128 runs averaged prompt `6329.643 tok/s`, decode `58.409 tok/s`, E2E `3.170793 s`, TTFT `0.989530 s`. After the cold restart settled, runs 6-10 averaged prompt `7661.368 tok/s`, decode `59.656 tok/s`, E2E `2.953115 s`, TTFT `0.815722 s`; prefill therefore reproduced the earlier selected `7727.858 tok/s` result closely.
- Short MTP decode varies with proposal acceptance: the final ten-run range was `53.208-63.192 tok/s`. A sustained exact 6213/1024 run reached `70.357 tok/s` decode, `15.578063 s` E2E, and `65.733 output tok/s` including prefill. Preserve the controlled earlier five-run `62.639 tok/s` mean for matched 128-token comparisons, while using `70.357 tok/s` as the measured sustained decode result.
- Final validation: `git diff --check` clean; all relevant Python files compile; both PowerShell launchers parse and render help; request-state cleanup tests `19/19` pass in the repo venv; native FP8 JIT smoke passes with finite results (`0.022174` quant relative MAE, `0.037598` batched GEMM, `0.036133` single-row GEMM); failed experimental kernels and temporary diagnostics are absent.
- Selected end state: RadixArk ModelOpt NVFP4, 32K active pool, Triton attention, FlashInfer CUTLASS FP4 GEMM, bundled three-step/four-token MTP, thinking enabled for the main model, explicit 8192-token production cap, and the validated server left running.

### 23:29 — Residual performance audit resumed

- Prior evidence rules out the ordinary high-value knobs: continuous decode steps 4, BF16 Mamba state, non-spec ReplaySSM, chunk 8192, KV splits 4/32, and FlashInfer attention either lost throughput or failed the thinking-quality guard on the earlier controlled path. Stream interval 4 and incremental disjoint output are already promoted.
- Remaining candidates specific to the newer RadixArk/MTP path are FP4 GEMM backend selection and scheduler amortization under speculation. The installed FlashInfer API exposes `cudnn`, `cutlass`, `cute-dsl`, and `auto` FP4 backends.
- Parameterized `smoke_flashinfer_nvfp4.ps1` with a backend selector while preserving `cutlass` as its default. A standalone smoke while the final server was live caused Ninja to attempt relinking FlashInfer's already-loaded `fp4_quantization_120f.dll`; Windows correctly rejected overwriting the mapped DLL with `LNK1104`. This is a diagnostic-process collision, not a production-server failure.
- Next: verify the live endpoint, stop it cleanly, exercise the alternative FP4 backend through an actual SGLang launch, benchmark only if it reaches readiness, then restore/promote the measured winner.

### 23:35 — cuDNN FP4 ruled out; seeded benchmark path added

- The cutlass endpoint remained coherent after the standalone DLL-lock collision (`7+8 -> 15` with normal thinking). Stopping the main Python process cleanly reaped the complete SGLang process tree and released GPU memory.
- A real `flashinfer_cudnn` RadixArk/MTP launch loaded the full target and draft weights and reached CUDA graph capture, then failed with FlashInfer's explicit `RuntimeError: cuDNN is not available`; the environment lacks the required cuDNN frontend/runtime. The failed tree exited and returned GPU use to the desktop baseline.
- Restored the default cutlass server under parent PID `3488`; it reached ready at `23:33:40` with all target/draft graphs captured.
- Added optional `--seed` support to `bench_openai_stream.py`. Three identical 6213/512 requests with `seed=42` still produced different hashes and `59.780-71.263 tok/s`. Code inspection explains this: per-request `sampling_seed` reaches the EAGLE verifier only with `--enable-deterministic-inference`; otherwise rejection sampling intentionally uses process-global CUDA RNG.
- `num_continuous_decode_steps` is a dead compatibility argument in this checkout: repository search finds no runtime consumer. Git history shows its scheduler loop was removed by `ffd20fcd03` while the CLI field remained. Further value sweeps would measure noise.
- The live scheduler-amortization mechanism is `scheduler_recv_interval`; its skipper explicitly handles speculative `TARGET_VERIFY`. Defer that small-overhead axis until FP4 backend capability is exhausted. Next candidate: FlashInfer CUTE DSL, which SGLang advertises and the launcher does not yet expose.

### 23:42 — FP4 backend sweep complete; scheduler polling does not win

- Exposed `flashinfer_cutedsl` through the Windows launcher and performed a full RadixArk/MTP startup attempt. Target and draft weights loaded, then graph capture rejected the backend explicitly: `mm_fp4 does not support backend 'cute-dsl' with capability 120`. Together with the cuDNN runtime failure and SM120's auto selection, this proves FlashInfer CUTLASS is the viable dense FP4 backend on this machine.
- Exposed `SchedulerRecvInterval` and launched the real speculative server at interval 4. Five 6213/512 runs averaged decode `62.631 tok/s` and E2E `9.016143 s`; the immediate interval-1 three-run control averaged decode `65.287 tok/s` and E2E `8.823775 s`. Proposal acceptance varied, but matched acceptance windows showed effectively identical generation rates. Interval 4 provides no reproducible win; retain interval 1.
- A deterministic A/B cannot be enabled with the selected distribution-correct rejection sampler. SGLang explicitly rejects this combination because draft proposals still use unseeded multinomial/Gumbel sampling, even though verify-side coins already have a seeded implementation. Removed the temporary deterministic launcher switch rather than expose an option that fails with production defaults.
- Remaining potentially material server lever is `torch.compile`; CUDA graphs already remove launch overhead, so its only plausible gain is fused pointwise/model arithmetic. Test it once with batch-size-1 capture, reject on startup incompatibility or absent repeated-run gain, then restore the final default server.

### 23:54 — Fixed-acceptance benchmark control established

- Partial `torch.compile` reached readiness after `75.68 s` target-verify capture, but Inductor repeatedly failed FP8 fusion fragments with `Unsupported conversion from f64 to f8E4M3FN` / `LLVM ERROR: Unsupported rounding mode for conversion` and fell back around them. Its noisy five-run 6213/512 mean was decode `64.516 tok/s`, E2E `8.742304 s`; an ordinary default five-run set later averaged `60.722 tok/s`, proving real acceptance randomness can reverse the apparent result.
- Switched the tuning A/B to SGLang's purpose-built `SGLANG_SIMULATE_ACC_LEN=3` benchmark mode. This holds speculative work and accepted length constant without changing production defaults. The baseline produced the identical 512-token digest on all five runs.
- Fixed-acceptance interval-1/CUTLASS/uncompiled baseline: decode `78.593, 73.245, 72.510, 73.972, 77.787 tok/s`; mean `75.221 tok/s`. E2E `7.596365, 7.808489, 7.938820, 7.805822, 7.380334 s`; mean `7.705966 s`. Mean TTFT `0.905364 s`.
- Next: repeat the identical simulated-acceptance workload at scheduler receive interval 4, then with partial torch compilation. Promote only a repeated fixed-work winner; afterward restart without the simulation environment and rerun thinking/tool acceptance.

### 23:58 — Scheduler receive interval 4 is a fixed-work winner

- Repeated the same five-run 6213/512 workload under `SGLANG_SIMULATE_ACC_LEN=3`, changing only `scheduler_recv_interval` from 1 to 4. All outputs retained the exact baseline digest.
- Interval-4 decode: `76.437, 78.086, 78.299, 78.826, 78.487 tok/s`; mean `78.027 tok/s`. E2E: `7.809897, 7.368444, 7.332937, 7.291395, 7.309253 s`; mean `7.422385 s`.
- Against interval 1, interval 4 improves mean decode by `3.73%` and reduces E2E time by `3.68%`. Excluding each server's first cold request, decode improves from `74.379` to `78.425 tok/s` (`+5.44%`). This cleanly isolates ZMQ receive-poll overhead from stochastic MTP acceptance.
- Interval 4 has earned promotion pending the plateau sweep. Next: test interval 8 under the identical fixed-acceptance guard; test 16 only if 8 produces another material gain. Then compare partial torch compilation against the winning interval, remove diagnostics, and run production thinking/tool acceptance without simulation.

### 00:01 — Receive-poll plateau found; select interval 4

- Fixed-acceptance interval-8 decode: `79.131, 78.511, 78.418, 78.409, 78.131 tok/s`; mean `78.520 tok/s`. E2E mean `7.357560 s`. The output digest remained identical to intervals 1 and 4.
- The apparent all-run edge over interval 4 is cold-start placement. Warm means are interval 4 `78.425 tok/s` / `7.325507 s` and interval 8 `78.367 tok/s` / `7.337442 s`. Interval 8 is flat on decode and fractionally slower E2E.
- Select `SchedulerRecvInterval=4` as the first plateau point. Skip interval 16 by the predeclared rule; it can only add request/abort polling latency after the throughput gain has saturated.
- Final controlled candidate: partial torch compilation at interval 4 under the same fixed acceptance. Then remove its opt-in if it does not beat the selected eager/CUDA-graph path, promote interval 4 as the launcher default, and relaunch without simulation for real thinking/tool validation.

### 00:04 — Account-swap pause checkpoint

- Pause time from `date`: `Sun Aug 16 00:04:48 PDT 2026`.
- Work stopped immediately at the user's request. No benchmark was started after the pause request.
- Selected result already proven: fixed-acceptance `SchedulerRecvInterval=4` mean decode `78.027 tok/s`, `+3.73%` over interval 1; interval 8 is the plateau and does not improve warm throughput.
- The only live experiment at pause was fixed-acceptance (`SGLANG_SIMULATE_ACC_LEN=3`) interval-4 partial torch compilation, PowerShell parent PID `13380`, logs `%LOCALAPPDATA%\Temp\sglang-qwen-nvfp4-sim3-recv4-compile.{stdout,stderr}.log`.
- At pause, target and draft weights had loaded. Torch Inductor was again emitting the known `f64` to `f8E4M3FN` unsupported-conversion failures during target-verify graph compilation; readiness and performance were not yet checked.
- Resume exactly by checking readiness/error tail for that log. If ready, run the same five 6213/512 fixed-acceptance requests and compare against eager interval 4 (`78.027 tok/s`, `7.422385 s`). Then remove the torch-compile opt-in unless it wins, set interval 4 as the launcher default, restart without the simulation environment, and rerun thinking/OpenCode acceptance.

### 00:07 — Post-compaction checkpoint

- Checkpoint time from `date`: `Sun Aug 16 00:07:13 PDT 2026`.
- The account-swap pause checkpoint above is intact. Work resumes from its exact boundary.
- The selected scheduler result remains `SchedulerRecvInterval=4`: fixed-acceptance mean decode `78.027 tok/s`, mean E2E `7.422385 s`; interval 8 is the warm-throughput plateau.
- The pending live experiment is interval-4 partial torch compilation under `SGLANG_SIMULATE_ACC_LEN=3`, PowerShell parent PID `13380`. Its readiness/process state must be inspected before any benchmark or cleanup.
- Next: inspect the compile candidate narrowly, benchmark it only if ready, retain it only for a material repeated fixed-work win, then promote interval 4 and restore a normal production launch for thinking/tool acceptance.

### 00:09 — Partial torch compilation wins fixed-work comparison

- The interval-4 compile server reached readiness at `00:04:44` after target-verify capture. Inductor emitted repeated unsupported `f64` to `f8E4M3FN` conversion errors and fell back for those graphs, yet the server remained healthy.

- Five identical fixed-acceptance 6213/512 runs produced decode `80.950, 81.214, 80.037, 81.880, 81.063 tok/s`; mean `81.029 tok/s`. E2E was `7.292835, 7.137911, 7.253272, 7.094331, 7.147561 s`; mean `7.185182 s`.
- Every run retained digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`, so the work and output are identical to the eager scheduler comparisons.
- Versus eager interval 4, partial compile improves mean decode from `78.027` to `81.029 tok/s` (`+3.85%`) and reduces mean E2E from `7.422385` to `7.185182 s` (`-3.20%`). Warm-only decode improves from `78.425` to `81.049 tok/s` (`+3.35%`).
- This is a repeated material win. Promote both `SchedulerRecvInterval=4` and partial torch compilation for the speed-first production launcher, then validate real thinking, tool use, cancellation, and unsimulated acceptance.

### 00:13 — Winning production defaults pass thinking, tools, and cancellation

- Promoted `SchedulerRecvInterval=4` and partial torch compilation to the launcher defaults. `-DisableTorchCompile` remains as the explicit fallback. Removed the dead `ContinuousDecodeSteps` launcher surface because the upstream runtime no longer consumes that compatibility argument.
- Clean unsimulated default launch is live under PowerShell PID `11804`; target verify and both draft graphs captured, and the server reached readiness at `00:11:19`.
- Direct thinking passed with separate coherent `reasoning_content`, a normal stop, and final answer `323` for `17 * 19`.
- Normal OpenCode2 acceptance passed through `opencode_qwen.ps1 -MainOutputCap 512`: visible coherent reasoning, a completed real `read` tool call against the launcher, retained tool output, and the correct final statement that receive interval 4 and torch compilation are defaults.
- Live cancellation passed after the polling change. An 8192-token thinking stream was interrupted after ten seconds; exactly three already-in-flight output packets crossed the teardown boundary, the count stopped at three, and an immediate thinking request completed normally with final answer `10`.
- Remaining: measure unsimulated production throughput, run focused static/tests/help validation, inspect the exact diff, update the final checkpoint, and leave PID `11804` live.

### 00:15 — Unsimulated production measurement

- Five real-acceptance 6213/512 thinking runs on the promoted defaults produced decode `66.318, 69.418, 60.457, 71.517, 61.886 tok/s`; mean `65.919 tok/s`. Mean prompt throughput was `7373.696 tok/s`, mean E2E `8.627941 s`, and mean TTFT `0.843718 s`.
- Output hashes varied because production rejection sampling has stochastic proposal/acceptance paths. Server logs tracked the rate swings with acceptance length/rate while all decode batches stayed on CUDA graphs.
- One 6213/1024 sustained production run measured `62.550 tok/s` decode, `17.269187 s` E2E, and `0.914245 s` TTFT; acceptance varied from roughly `1.68` to `3.00` across log windows.
- The fixed-work result (`81.029 tok/s`) is the valid isolated proof for the scheduler/compile optimization; the unsimulated numbers describe real production variance rather than a deterministic A/B.

### 00:17 — Final residual-performance checkpoint

- Checkpoint time from `date`: `Sun Aug 16 00:17:33 PDT 2026`.
- Final speed-first defaults are RadixArk ModelOpt NVFP4, 32K pool, Triton attention, FlashInfer CUTLASS FP4 GEMM, three-step/four-token bundled MTP, scheduler receive interval 4, stream interval 4, and partial torch compilation at batch size 1.
- Isolated cumulative residual gain: interval 4 improved fixed-work decode `+3.73%` over interval 1; partial compile then improved another `+3.85%` over eager interval 4. The selected five-run fixed-work mean is `81.029 tok/s` with identical output digests.
- Production acceptance is complete: coherent thinking, normal stop, real OpenCode read-tool round trip, clean interrupted-stream cleanup, and immediate post-cancel recovery all passed.
- Final validation is clean: `git diff --check`; relevant Python `py_compile`; launcher, OpenCode wrapper, FlashInfer NVFP4 smoke, and benchmark help/parse checks; tokenizer-manager request cleanup `19/19` tests. The benchmark seed help now explicitly states that speculative proposals may remain nondeterministic.
- The known Inductor FP8 conversion failures affect only unsupported graph compilations and fall back during startup; the remaining compiled subset produced the measured repeated throughput win. `-DisableTorchCompile` is the documented operational escape hatch.
- Validated production server remains live: PowerShell PID `11804` -> `sglang.exe` PID `5096` -> Python PID `31908`, with no simulation environment and effective `scheduler_recv_interval=4`, `enable_torch_compile=True`, `torch_compile_max_bs=1`.

### 00:18 — Compiler-fallback optimization continuation

- Continuation time from `date`: `Sun Aug 16 00:18:31 PDT 2026`.
- The fastest validated production server above stays live while investigation begins.
- Next evidence-backed target: trace Inductor's repeated unsupported `f64` to `f8E4M3FN` conversion during target-verify compilation. Partial compilation already adds `3.85%`; compiling the fallback graphs may expose further throughput and startup gains.
- Preserve the current `81.029 tok/s` fixed-work result as the control. Any compiler repair must retain the identical fixed-acceptance digest plus thinking/tool/cancellation behavior before promotion.

### 00:22 — Two fp64 promotion roots repaired locally

- The first Inductor failure was `_static_quant_fp8`: `fp8_min` and `fp8_max` appeared in metadata as `fp64`, upcast the clamp, and produced an unsupported SM120 `f64 -> f8E4M3FN` conversion. The kernel now explicitly casts both bounds to `tl.float32`.
- Six other failures were the fused sigmoid-gating recurrent kernel: `softplus_beta`, `softplus_threshold`, `lower_bound`, and `scale` appeared as `fp64`, promoted loop-carried `b_h`, and violated Triton's stable loop-type requirement. The kernel now casts these runtime scalars once to `tl.float32` before arithmetic.
- Focused GDN target-verify parity passes `12/12` across `N={1,8,16}` and `T={1,4,8}`. The native Windows FP8 JIT smoke now also covers static quantization and reports `static_exact=True`; existing quant/GEMM metrics remain unchanged.
- The broad manual FP8 test module cannot collect in this Windows environment because its unrelated MoE import requires absent optional `sgl_kernel`. The direct static-quant smoke provides the narrow executable coverage, and the authoritative next test is a clean compile-enabled server capture.
- Production PID `11804` still runs the pre-patch modules as the retained control. Next: restart under fixed acceptance, confirm all 14 prior Triton compile failures disappear, then measure against `81.029 tok/s`.

### 00:28 — Full compile capture still progressing cleanly

- Patched fixed-acceptance server launched under PowerShell PID `16560` with `SGLANG_SIMULATE_ACC_LEN=3`; target and draft weights loaded successfully.
- At `Sun Aug 16 00:28:09 PDT 2026`, target-verify compilation has emitted none of the prior `Triton compilation failed`, unsupported FP8 conversion, loop-type, traceback, or runtime-error markers.
- Successful compilation is materially longer than the fallback startup because the repaired graphs now proceed through code generation. Worker PID `34132` remains CPU-active and its accumulated CPU time continues rising; the log remains at the capture boundary rather than reporting a failure.
- Continue waiting for readiness. Benchmark only after capture completes; preserve the pre-patch `81.029 tok/s` control and identical digest gate.

### 00:31 — Fully compiled repair is correct but slower

- The patched target-verify capture completed with zero recurrence of all 14 prior Triton compile failures. First-time successful code generation took `403.54 s`; the server reached readiness normally.
- Five fixed-acceptance 6213/512 runs retained the exact control digest and produced decode `74.878, 76.237, 76.282, 76.042, 76.200 tok/s`; mean `75.928 tok/s`. Mean E2E was `7.603428 s` and TTFT `0.873020 s`.
- This is `6.30%` slower than the selected partial/fallback-compile control (`81.029 tok/s`). Warm-only decode is `76.190 tok/s`, still `5.99%` below the control, so cold placement does not explain the result.
- Correctness is preserved, but fully compiling the repaired kernels is a performance regression. Do not promote both casts together.
- Next isolation: retain the static-FP8 cast and remove the recurrent-gating cast, then repeat fixed work. This distinguishes transformer static-quant compilation from recurrent GDN compilation; test the inverse only if needed.

### 00:34 — Static quant compilation also loses throughput

- With static FP8 quant repaired/compiled and recurrent GDN left on its original fallback, target capture completed in `23.63 s`; only the seven expected recurrent-kernel loop-type failures remained.
- Five fixed-acceptance runs retained the control digest and produced decode `79.978, 79.509, 79.984, 80.081, 79.946 tok/s`; mean `79.900 tok/s`. Mean E2E was `7.267232 s`; warm decode was `79.880 tok/s`.
- This remains `1.39%` below the `81.029 tok/s` both-fallback control. The full repair's additional loss therefore comes mainly from compiled recurrent GDN, while compiled static quant contributes a smaller repeatable loss.
- Fastest proven behavior keeps both Triton calls eager. Next: replace the accidental compile exceptions with explicit `torch.compiler.disable` boundaries on the two public wrappers, restoring their eager CUDA-graph capture while allowing surrounding compilation to proceed cleanly.

### 00:42 — Explicit graph boundaries rejected

- Added explicit `torch.compiler.disable` boundaries around static FP8 quant and fused recurrent GDN, with both kernels restored to their original eager arithmetic. Focused GDN parity remained `12/12`; the FP8 smoke remained exact and finite.
- Compile capture became clean with no Triton failures. A stale process-cleanup race killed the first launch after readiness; a fresh independent launch stayed healthy and provided the measurement.
- Five fixed-acceptance runs retained the control digest and produced decode `77.420, 77.443, 77.007, 77.354, 77.415 tok/s`; mean `77.328 tok/s`. Mean E2E was `7.483167 s`.
- Explicit boundaries change Dynamo segmentation and are `4.57%` slower than the selected accidental-fallback topology. Reject them. Restore both kernel source files to their original state; retain only the expanded static-FP8 correctness smoke.
- The compiler investigation is conclusive: `81.029 tok/s` partial compilation with these two natural fallbacks remains fastest among fully compiled, one-class compiled, explicit-boundary, and original-fallback variants.

### 00:45 — Torch compile `default` mode reaches 86.016 tok/s

- Confirmed the serving invariant requested by the user: effective server args report `language_model_only=True` and the launcher passes `--language-model-only`; the vision encoder is disabled and never loaded. OpenCode advertises text-only input for the main model.
- SGLang's decode compile wrapper defaults to environment mode `max-autotune-no-cudagraphs`. Tested `SGLANG_TORCH_COMPILE_MODE=default` under the identical fixed-acceptance, interval-4, partial-fallback topology.
- Five 6213/512 runs retained digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c` and produced decode `85.779, 85.456, 86.278, 86.378, 86.190 tok/s`; mean `86.016 tok/s`. Mean E2E was `6.743358 s`, TTFT `0.802521 s`, and prompt throughput `7752.565 tok/s`.
- This adds `6.16%` decode over the prior `81.029 tok/s` winner and cuts E2E by `6.15%`. From the original interval-1 compile control (`75.221 tok/s`), the combined residual gain is `14.35%`.
- Promote compile mode `default` unless `reduce-overhead` provides another repeated fixed-work gain. Preserve the original kernel sources; their partial fallback remains part of the selected fast topology.

### 00:47 — Compile mode winner promoted; vision remains disabled

- `reduce-overhead` is incompatible with SGLang's outer full CUDA-graph capture on this topology. Startup aborted with `Detected 1 tensor(s) in the cudagraph pool not tracked as outputs`; reject nested compiler CUDA graphs.
- Promoted `TorchCompileMode='default'` in the Windows NVFP4 launcher. The mode is process-scoped through `SGLANG_TORCH_COMPILE_MODE` only while the server child runs, with the caller's prior environment restored on exit.
- The launcher ValidateSet retains the measured winner and the previous operational fallback (`max-autotune-no-cudagraphs`); the known-incompatible nested-CUDA-graph mode is excluded.
- Vision stays disabled through the existing explicit `--language-model-only` argument. The speed change affects only decode compilation.
- Next: launch from untouched defaults without simulation, confirm the effective mode and language-only state, rerun thinking/OpenCode/cancellation acceptance, measure real production throughput, and leave the winner live.

### 00:49 — Expanded exhaustive target locked

- Requirement checkpoint from `date`: `Sun Aug 16 00:48:50 PDT 2026`.
- Final target is Qwen3.8-27B NVFP4 with reasoning enabled, vision disabled, tool calling enabled, thinking preserved, and a real `200000`-token context contract.
- Exhaust the SGLang runtime, kernels, launch configuration, native Windows build, PyTorch/Triton/FlashInfer and other supporting dependencies. CUDA headers are explicitly out of scope and must remain untouched.
- The current 32K-pool / 131072-context launch remains a throughput baseline only. Completion now requires proving 200K request capacity or implementing the necessary memory strategy, updating OpenCode's advertised context, and rerunning long-context plus reasoning/tool acceptance.
- Preserve the new fixed-work speed control: compile mode `default`, scheduler interval 4, five-run mean `86.016 tok/s`, identical digest. Every capacity or dependency candidate must be measured against a matched control.

### 00:51 — Post-compaction checkpoint

- Checkpoint from `date`: `Sun Aug 16 00:51:41 PDT 2026`.
- Selected speed control remains compile mode `default` at a five-run mean of **86.016 tok/s**.
- Clean pre-200K production baseline was live at compaction under PowerShell parent PID **6384**.
- Required end state remains Qwen3.8-27B NVFP4 with reasoning enabled, vision disabled, tool calling enabled, thinking preserved, and a real **200000-token** context contract.
- Current investigation: Qwen3.5 MTP embedding/lm-head sharing occurs after target/draft KV-pool allocation, despite releasing enough duplicate draft storage to potentially make the 200K pools feasible.
- Exact next action: inspect `Scheduler.init_model_worker` ordering, then move sharing ahead of target-pool allocation only if its token-map and pool dependencies prove safe.

### 00:55 — MTP allocation-order root confirmed

- `Scheduler.init_model_worker()` constructs target and draft workers, then immediately calls `init_memory_pools()`. That routine allocates the target KV pool first and the draft KV pool second.
- `EagleDraftWorker.alloc_memory_pool()` only calls `init_token_map()` and `init_lm_head()` after the draft pool exists. For the Qwen3.5 MTP path, those two initializers have no KV-pool dependency: the token-map initializer only resolves optional vocabulary mapping, and `init_lm_head()` only aliases target embedding/head storage into the draft model.
- Qwen3.5 MTP `set_embed_and_head()` deletes the duplicate draft parameters, aliases the target tensors, empties the CUDA allocator cache, and synchronizes. Moving this preparation ahead of the target allocation can expose the reclaimed storage when the 200K pool size is chosen.
- Guard the optimization around deferred startup-weight loading: that mode holds sentinel target weights until finalization, so generic early sharing is unsafe there. The selected local launch does not use startup weight-load overlap.
- Next: run an untouched-code 200K startup control, preserve its exact failure/memory evidence, then add an idempotent pre-pool preparation hook with focused ordering tests.

### 00:56 — Untouched 200K control proves the contract is currently false

- Launched untouched code with `-ContextLength 200000 -MaxTotalTokens 200000` and all selected speed defaults. The server became ready, but SGLang silently sized the usable pool to only **66459 tokens**; startup reported `context_len=200000` alongside `max_total_num_tokens=66459`.
- Exact memory sequence: target weights 19.25 GB left 10.89 GB; duplicate MTP weights 6.22 GB left 4.67 GB; target FP8 KV for 66459 tokens used 1.01 GB K + 1.01 GB V and left 2.07 GB; draft BF16 KV used 0.13 GB K + 0.13 GB V and left 1.87 GB.
- After late embedding/head sharing, graph capture began with 6.33 GB available and ended with 5.92 GB available. This confirms roughly 4.46 GB becomes available only after pool sizing, exactly where it cannot help the requested capacity.
- Evidence logs: `%LOCALAPPDATA%\\Temp\\sglang-qwen-nvfp4-200k-control.stderr.log`; control parent PID 35044. The 200K request is therefore configuration-deep but not capacity-real.
- Next: implement early, idempotent draft weight sharing before target pool sizing, while preserving the old deferred-weight-load path.

### 01:00 — Early-sharing hook corrected through the outer worker

- First implementation placed the idempotent hook on the inner `EagleDraftWorker`, while the scheduler holds the outer `EAGLEWorkerV2`. The scheduler therefore reached the base no-op and the first experimental launch remained effectively unchanged at 66601 tokens.
- Corrected `BaseSpecWorker.prepare_memory_pool_allocation()` to delegate to an inner draft worker that implements the hook. The outer EAGLE worker now reaches the Qwen MTP aliasing path before target KV sizing.
- Four focused unit tests pass: outer delegation, inner idempotency, serial preparation-before-target ordering, and deferred-weight-load preservation. `py_compile` and targeted `git diff --check` also pass.
- Next: repeat the identical 200K launch with delegation active and inspect the pre-pool available-memory transition.

### 01:01 — Real 200K target and draft pools fit on the RTX 5090

- Corrected early sharing succeeded under the identical selected launch: target FP8 KV allocated exactly **200000 tokens** (3.05 GB K + 3.05 GB V), and draft BF16 KV allocated exactly **200000 tokens** (0.38 GB K + 0.38 GB V).
- Target pool ended with 2.84 GB available; draft pool ended with 2.07 GB. Full target graph capture ended with 1.48 GB, draft decode with 1.49 GB, and draft extend with 1.22 GB.
- Startup completed cleanly and reported `max_total_num_tokens=200000`, `context_len=200000`, `max_running_requests=1`, `available_gpu_mem=1.22 GB`.
- Evidence: `%LOCALAPPDATA%\\Temp\\sglang-qwen-nvfp4-200k-earlyshare2.stderr.log`; live experimental parent PID 28016.
- This converts the prior false 200K configuration into a capacity-real server without changing quantization, attention, compilation, speculative settings, or CUDA headers.
- Next: send a calibrated near-limit prompt, validate reasoning/tool behavior, then rerun the fixed 6213/512 throughput control against the 86.016 tok/s winner.

### 01:07 — Near-limit request accepted; Triton long-prefill is the next wall

- A calibrated 199000-prompt-token plus 16-output-token request passed tokenization, cache flush, admission, and began chunked prefill against the real 200K pool.
- With the selected all-Triton target attention path and 4096-token chunks, per-chunk input throughput decayed as the prefix grew: 332.16, 217.74, 171.26, 138.57, 116.91, 102.34, 88.16, then 81.12 tok/s through the first 36864 prompt tokens. This makes a completed 199K proof impractically slow and exposes a large user-visible long-context bottleneck.
- Cancelled the client, then stopped the experimental server because an in-flight chunked prefill does not promptly observe the disconnected client. No CUDA-header changes.
- The runtime exposes independent `prefill_attention_backend` and `decode_attention_backend` settings. On SM120 hybrid GDN models, this fork allows Triton, FlashInfer, and TRT-LLM MHA. Preserve Triton decode while benchmarking FlashInfer and TRT-LLM prefill at controlled long prefixes.
- Next: add launcher controls for split target attention backends, compare fixed 32K/64K long-prefill workloads, and retain the 6213/512 decode control for every viable split.

### 01:10 — FlashInfer prefill blocked by an outdated supporting dependency

- Added explicit split target-backend controls to the launcher: `PrefillAttentionBackend` and `DecodeAttentionBackend`, each limited to the SM120 hybrid-GDN-supported Triton, FlashInfer, and TRT-LLM MHA set. Help parsing and targeted diff checks pass.
- The first FlashInfer-prefill / Triton-decode launch failed before weight loading because local `flashinfer_python==0.6.11.post3` is below this fork's required **0.6.17**.
- This is a concrete supporting-dependency optimization gap, directly within the user's requested scope. Preserve CUDA headers; inspect the local PyTorch/CUDA ABI and official FlashInfer Windows package matrix before upgrading.
- Failure evidence: `%LOCALAPPDATA%\\Temp\\sglang-qwen-nvfp4-200k-fi-prefill.stderr.log`; attempted parent PID 36252.
- Next: inventory exact local dependency/build versions and install the newest compatible official FlashInfer build, then rerun its smoke tests before repeating the split-backend benchmark.

### 01:15 — Dependency ABI and FlashInfer provenance locked

- Local runtime: PyTorch `2.13.0+cu130`, PyTorch CUDA runtime 13.0, Triton Windows `3.7.1.post27`, FlashInfer `0.6.11.post3`, Transformers `5.12.1`, RTX 5090 SM120, NVIDIA driver 610.88, and CUDA toolkit 13.3.33.
- Installed FlashInfer came from the local Windows fork at `C:\Users\Daniel\flashinfer-windows`, tag `v0.6.11.post3`, origin `SystemPanic/flashinfer-windows`. Its worktree already contains user changes, including an untracked `csrc/nv_internal/cuda.h`; that CUDA header and its compatibility work remain explicitly untouched.
- Official FlashInfer 0.6.17 exists and is the SGLang fork's pinned version. Its PyPI wheel is universal and supports CUDA 13, but normal resolution is blocked on Windows by mandatory `nccl4py>=0.3.1`, for which no win_amd64 wheel exists. A `--no-deps` install resolves cleanly.
- The official 0.6.17 release includes refreshed SM12x FP4 kernels and fixes, making it both a compatibility requirement and a plausible performance/correctness gain. The personal Windows fork origin has not advanced beyond v0.6.11.post3.
- Next: inspect the official v0.6.17 source against the Windows fork and carry forward only the required Windows adaptations in a separate clean update path, preserving every existing dirty file and CUDA header.

### 01:16 — Existing Windows fork already has the fast FlashInfer prefill path

- Capability-tested the installed Windows FlashInfer fork with SGLang's version gate bypassed only for the experiment. It loaded the full 200K target/draft pools and initialized `decode_backend=triton, prefill_backend=flashinfer` successfully; the dependency gap is at least partly metadata/version alignment rather than a missing attention API.
- Cold calibrated 32768-token prefill plus 16 output tokens completed with TTFT **8.809964 s** and observed prompt throughput **3719.425 tok/s**. Decode remained **80.339 tok/s** on the short 16-token tail.
- The all-Triton near-limit trace took about 268 seconds merely to complete its first 36864 prompt tokens and had fallen to 81.12 tok/s per chunk. FlashInfer therefore removes the dominant long-context latency wall while leaving the selected Triton decode path available.
- Experimental server: parent PID 7428; logs `%LOCALAPPDATA%\\Temp\\sglang-qwen-nvfp4-fi-old-bypass.*.log`; capacity remained exactly 200000 tokens.
- Do not permanently bypass the version check. Qualify the local backport against required 0.6.17 APIs/smokes, then version/reinstall the personal fork truthfully or port it onto 0.6.17 without touching the user's CUDA header.
- Next: benchmark 6213/512 decode equivalence, 64K prefill, chunk sizes, and TRT-LLM MHA prefill; then settle the safe dependency packaging path.

### 01:19 — Full near-limit 200K request proven with FlashInfer prefill

- The calibrated **199000 prompt + 16 completion = 199016 total token** request completed successfully on the 200000-token server. This is the end-to-end capacity proof the earlier all-Triton run could not finish.
- Near-limit result: TTFT **102.042461 s**, observed prefill **1950.169 tok/s**, E2E **102.376826 s**, finish reason `length`, and all 16 requested completion tokens returned.
- Intermediate cold checks: 32768 prompt tokens at 3719.425 tok/s / 8.809964 s TTFT; 65536 at 4716.828 tok/s / 13.894084 s TTFT. A normal 6213/512 request retained 85.383 decode tok/s and 7472.838 prompt tok/s.
- FlashInfer prefill therefore makes the real 200K contract operational in about 102 seconds on the RTX 5090 while preserving the fast Triton decode path.
- User directive: port the personal Windows FlashInfer adaptation onto official **0.6.17**. Use a clean sibling worktree, preserve the existing CUDA header byte-for-byte and leave its original dirty worktree untouched, then install and rerun the same qualification.

### 01:23 — FlashInfer 0.6.17 port worktree established safely

- Fetched official signed tag v0.6.17 (`a0a6b01`) and created clean branch/worktree `windows-v0.6.17` at `C:\Users\Daniel\flashinfer-windows-0.6.17`. The original `C:\Users\Daniel\flashinfer-windows` worktree and all five dirty paths remain untouched.
- Replayed the old Windows snapshot once as a three-way compatibility audit. It touched 80 files and conflicted with evolved 0.6.17 FP4/GDN/XQA code; carrying that whole snapshot would risk replacing newer upstream kernels with older backports. The audit cherry-pick was cleanly aborted and the new worktree is clean again.
- Port strategy: carry the platform/build/JIT adaptations onto 0.6.17 surgically, allow current upstream kernel implementations to stand, then discover any remaining MSVC issues through focused imports/JIT smokes. Preserve the existing CUDA compatibility header exactly and keep it outside the edit set.

### 01:29 — 0.6.17 Windows JIT foundation ported; first smoke reached NVCC

- Ported the platform-neutral 0.6.17 JIT generator onto native Windows: MSVC compile/link rules, CUDA/MSVC flags, `.dll` artifacts, Ninja drive-letter escaping, Windows library search paths, `shutil.which` CUDA discovery, and the short `C:\\_fij` cache root.
- The first isolated 0.6.17 package smoke advanced past import and Ninja parsing, launched seven MSVC/NVCC compilation units, then stopped because the clean worktree's CUTLASS submodule had not been initialized. This validates the new JIT plumbing itself.
- Initialized the exact official 0.6.17 CCCL, CUTLASS, NIXL, and spdlog submodule revisions. CUDA 13.3 is in use; the user's compatibility header remains untouched and was not copied into the port.
- Next: rebuild the isolated wheel with populated submodules and continue the NVFP4 smoke from the first genuine compiler incompatibility.
## 2026-08-16 01:30:29 PDT — post-compaction checkpoint

- Checkpoint time: `Sun Aug 16 01:30:29 PDT 2026`.
- Real 200K capacity is established: target and draft KV pools each allocate 200,000 tokens, and a request completed with 199,000 prompt tokens + 16 generated tokens = 199,016 total tokens.
- FlashInfer prefill at 199K measured TTFT 102.042461 s and 1950.169 prompt tok/s.
- User explicitly ordered: **“Port it onto 0.6.17”**.
- Clean FlashInfer port worktree/branch: `C:\Users\Daniel\flashinfer-windows-0.6.17`, `windows-v0.6.17`, based on official `v0.6.17` (`a0a6b01`).
- Current exact blocker: after the native-alignment fix, official 0.6.17 `csrc/nv_internal/cpp/kernels/quantization.cu` hits an NVCC/MSVC internal compiler error in the nested generic-lambda dispatch near line 131: `could not lookup variable in map!`.
- Next action: rewrite that dispatch with named helper templates while preserving every dispatch combination and leaving the user's CUDA header untouched.
### 2026-08-16 01:32 PDT — FlashInfer 0.6.17 NVCC-ICE workaround applied

- In the clean `windows-v0.6.17` worktree, rewrote only `dispatchNVFP44Over6Config` and its enabled/disabled tag selection in `csrc/nv_internal/cpp/kernels/quantization.cu`.
- The replacement uses named helper templates and direct runtime branches instead of the NVCC/MSVC-crashing nested generic lambdas.
- Preserved both disable-fast-math tags, both E4M3 maxima (256/448), both error modes (MAE/MSE), both error-fast-math tags, and the non-4-over-6 `std::false_type` path.
- No CUDA header was edited. Next: inspect the exact diff, rebuild the isolated 0.6.17 target, and run the NVFP4 smoke test.
### 2026-08-16 01:34 PDT — FlashInfer 0.6.17 isolated build passed

- Built the surgical port into `%LOCALAPPDATA%\Temp\flashinfer-0.6.17-port5` with `uv pip install --target ... --no-deps .`.
- Build completed successfully as `flashinfer-python==0.6.17`; the prior NVCC/MSVC internal compiler error in `quantization.cu` is cleared by the flat named-template dispatch.
- Active SGLang environment remains unchanged. Next gate is the RTX 5090 NVFP4 smoke test against this isolated package.
### 2026-08-16 01:36 PDT — 0.6.17 JIT reached a second NVCC backend defect

- The isolated wheel/package build passed, but the real SM120 JIT smoke compile still fails in `kernels/quantization.cu`.
- The original `could not lookup variable in map!` ICE is gone. NVCC now emits malformed LLVM IR diagnostics after compiling the same translation unit: repeated `Terminator found in the middle of a basic block! label %2`, followed by `<unnamed>: parse Invalid instruction with no BB (Producer: 'LLVM23.0.0' Reader: 'LLVM 23.0.0')`.
- This is a child-process failure despite `Start-Process` itself returning success; smoke correctness has **not** passed.
- Next: flatten the remaining generic-lambda dispatch layers in the FP4 selector/call path, then rebuild into a new isolated target and rerun from a fresh JIT cache key. User CUDA headers remain untouched.
### 2026-08-16 01:39 PDT — malformed NVCC IR root candidate corrected

- In official 0.6.17 `quantization_utils.cuh`, the 4-over-6 specialization returned for every runtime path but still structurally compiled the ordinary tail, whose `fp8SFVal` and `outputScale` state is intentionally unset for 4-over-6.
- Wrapped that ordinary tail in the compile-time `else` of the 4-over-6 branch. Runtime semantics are unchanged; NVCC can now discard the unreachable tail cleanly for each 4-over-6 specialization.
- This edits the clean port's official kernel source only. The user's custom CUDA header in the original dirty Windows fork remains untouched.
- Next: isolated port6 build and real SM120 JIT smoke.
### 2026-08-16 01:42 PDT — compiler failures separated

- Diagnostic-only edit in disposable `flashinfer-0.6.17-port6` removed 4-over-6 template instantiations, then a fresh `C:\_fij7` JIT compile no longer produced malformed LLVM IR. This confirms the LLVM defect is inside the newly instantiated 4-over-6 kernel family, not the ordinary NVFP4 path.
- Compilation then advanced to a distinct MSVC host-stub failure: repeated `C2719: formal parameter with requested alignment of 128 won't be aligned` for by-value `CUtensorMap` kernel parameters.
- The original dirty Windows fork's user-owned, untracked `csrc/nv_internal/cuda.h` already works around exactly this CUDA 13.3/MSVC issue by preserving the toolkit header and reducing that declaration to 64-byte alignment. It was inspected read-only and remains unmodified.
- Next diagnostic: instantiate one 4-over-6 configuration to determine whether every new configuration breaks CUDA 13.3 or only part of the configuration matrix; handle the tensor-map alignment separately without editing the user's original header.
### 2026-08-16 01:51 PDT — ordinary 0.6.17 quantization JIT passed; GEMM exposed include-order gap

- With 4-over-6 gated and an unchanged copy of the proven CUDA 13.3 tensor-map shim in the disposable port6 package, `fp4_quantization_120f` compiled successfully. This is the first successful real SM120 compile of the 0.6.17 ordinary NVFP4 quantizer.
- The smoke advanced into the full `fp4_gemm_cutlass_sm120` JIT (18 compilation units, roughly 8 minutes here), then failed with the same MSVC `C2719` 128-byte formal-parameter alignment error across generated CUTLASS kernels.
- Cause is include topology: the quantization module includes `data/csrc/nv_internal` before the toolkit and sees the shim; the CUTLASS GEMM JIT does not include that directory and therefore sees the raw CUDA 13.3 `CUtensorMap` declaration.
- Next: port the existing header unchanged and add its directory as the first Windows CUDA include for every JIT spec, then reuse the completed object cache so only failed GEMM units rebuild.
### 2026-08-16 01:54 PDT — permanent 0.6.17 Windows compatibility choices staged

- Restored official non-Windows 4-over-6 dispatch exactly. On native Windows only, requesting the optional 4-over-6 quantizer now raises an explicit compiler-compatibility error so CUDA 13.3 does not instantiate malformed kernels; the ordinary/default NVFP4 path is preserved.
- Ported the existing `csrc/nv_internal/cuda.h` into the clean 0.6.17 worktree **byte-for-byte** and added `nv_internal` as the first Windows system include for every JIT module, including generated CUTLASS GEMM sources.
- SHA-256 of both the original user header and clean-port copy: `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`. The original user-owned file was not edited.
- Reverted the ineffective `quantization_utils.cuh` experiment. Non-Windows kernel source behavior remains official 0.6.17.
- Next: build port7, run quantization + CUTLASS GEMM smoke from a fresh cache, then test active SGLang integration.
### 2026-08-16 02:07 PDT — FlashInfer 0.6.17 NVFP4 smoke passed on RTX 5090

- Clean port7 package plus fresh `C:\_fij10` cache completed the full native SM120 JIT successfully.
- Real smoke result: `flashinfer=0.6.17`, `backend=cutlass`, `output_shape=(4, 128)`, `finite=True`, `relative_mae=0.148438`, child exit code 0.
- This validates both 0.6.17 ordinary NVFP4 quantization and the generated 0.6.17 CUTLASS SM120 FP4 GEMM kernels on Windows/CUDA 13.3.
- First cold JIT took roughly 9 minutes; the cache is now populated. Optional 4-over-6 remains explicitly unavailable on this compiler because even one configuration generates malformed LLVM IR. The Qwen serving path does not enable it.
- Next: install this exact clean-worktree package into `sglang\.venv`, verify version/import provenance, and launch the 200K FlashInfer-prefill + Triton-decode production topology for direct acceptance and throughput requalification.
### 2026-08-16 02:10 PDT — SGLang venv upgraded to the clean 0.6.17 port

- Replaced `flashinfer-python==0.6.11.post3` from the old dirty fork with `flashinfer-python==0.6.17` built from `C:\Users\Daniel\flashinfer-windows-0.6.17`.
- Verified live import version `0.6.17` from `sglang\.venv\Lib\site-packages\flashinfer\__init__.py`.
- Verified installed `direct_url.json` points to `file:///C:/Users/Daniel/flashinfer-windows-0.6.17`.
- No SGLang server was live during replacement. Next: reuse/prime the SGLang cache, launch the real 200K production topology, and requalify prefill, decode, thinking, and tools.
### 2026-08-16 02:20 PDT — installed 0.6.17 package recompiled and passed

- Re-ran the full NVFP4 smoke through the actual `sglang\.venv` installation with `FLASHINFER_CACHE_DIR=C:\_fij10` and no `PYTHONPATH` override.
- The cache rebuilt against stable site-packages paths, then passed identically: version 0.6.17, CUTLASS backend, finite `(4, 128)` output, relative MAE 0.148438, exit 0.
- `C:\_fij10` now holds production-usable SM120 objects and DLLs built from the installed clean port. Next: launch SGLang with this cache and real 200K limits.
### 2026-08-16 02:21 PDT — 0.6.17 production requalification server launched

- Hidden parent PowerShell PID `21980` launched the real target topology: context 200000, max total tokens 200000, FlashInfer prefill, Triton decode, 4096-token chunks, MTP/speculation enabled, reasoning parser enabled, Qwen tool parser enabled, language-model-only vision disable, preserved-thinking request support.
- Server inherits `FLASHINFER_CACHE_DIR=C:\_fij10` containing the installed-package 0.6.17 SM120 cache.
- Logs: `%LOCALAPPDATA%\Temp\sglang-qwen-nvfp4-fi0617-200k.stdout.log` and `.stderr.log`.
- Next: wait for readiness while selectively checking version/backend/pool/graph milestones; then direct thinking, tools, 6213/512 control, and long-prefill validation.
### 2026-08-16 02:23 PDT — real server found a 0.6.17 long-object-name regression

- 0.6.17 model load and real 200K allocation path succeeded through target/draft loading and hybrid backend initialization: FlashInfer prefill + Triton decode, with 1.40 GB available at target verify graph capture.
- Graph capture triggered the 0.6.17 paged-prefill JIT and failed before compilation because NVCC could not open its generated dependency output path. Exact error: `nvcc fatal: Could not open output file ...batch_prefill_with_kv_cache_..._batch_prefill*.cuda.o.d`.
- Root `C:\_fij10` is already short; the regression is the duplicated, parameter-rich 0.6.17 module URI in each object/dependency filename. This is a Windows path-length issue, not a kernel/source failure and not memory pressure despite SGLang's generic suggestions.
- Parent PID `21980` exited with graph-capture failure. Next: port/extend Windows object-name hashing in FlashInfer's Ninja generator, rebuild/reinstall, and relaunch against the same cache directory.
### 2026-08-16 02:28 PDT — Windows JIT artifact hashing implemented and tested

- FlashInfer 0.6.17 now retains descriptive module/cache directories while naming Windows objects as `obj_<index>_<12-hex-source-hash>.cuda.o` (or `.o`) and the JIT DLL as `module.dll`.
- Non-Windows artifact naming is unchanged. `JitSpecNvcc` library paths, object paths, compile-command output, and Ninja generation share the same naming helpers.
- Added a focused regression test using a 170-character module name. Pycompile passed; focused pytest passed `1 passed, 20 deselected`; `git diff --check` passed.
- This reduces the failing real module's object/dependency path from well beyond Win32 limits to roughly 230 characters while preserving deterministic incremental builds.
- Next: rebuild/reinstall port, relaunch the exact 200K topology, and let the failed attention JIT resume with short artifacts.
### 2026-08-16 02:33 PDT — path-safe 0.6.17 server relaunched

- Refined Windows naming to shorten artifacts only when the resolved output/dependency path would reach 240 characters; short modules retain readable official filenames and their warm cache.
- Added short-path preservation coverage. Pycompile, `2 passed`, and diff check passed; rebuilt and reinstalled the same 0.6.17 port.
- Relaunched exact 200K FlashInfer-prefill/Triton-decode topology under hidden parent PID `8464`.
- Logs: `sglang-qwen-nvfp4-fi0617-200k-short.stdout.log` and `.stderr.log`. The failed attention module should now emit 27-character object filenames and `module.dll` while reusing short-module FP4 caches.
### 2026-08-16 02:35 PDT — hashed paths worked; next omitted Windows source patch identified

- Relaunch emitted the intended short artifacts (`obj_000_...cuda.o` through `obj_009_...cuda.o`), so the path-length failure is cleared.
- All attention units then failed at `include/flashinfer/vec_dtypes.cuh:114` because official 0.6.17 defines `FLASHINFER_INLINE` with GNU `__attribute__((always_inline))`, which native Windows NVCC/MSVC rejects.
- The old Windows fork already carries the exact platform branch: Windows uses `inline [[msvc::forceinline]] __device__`; other platforms retain the official GNU attribute.
- This shows the initial surgical JIT port omitted at least one committed Windows source adaptation. Next: inventory every committed Windows delta from the old fork, port the still-relevant set systematically, then rebuild once instead of chasing them one at a time.
### 2026-08-16 02:42 PDT — committed Windows attention layer ported forward

- Ported the old working fork's still-relevant MSVC adaptations onto evolved 0.6.17: `FLASHINFER_INLINE`, Windows shared-memory syntax, fixed-width `std::max` operands, `ushort`, `M_PI`, and the Llama-3.1 rotary host signature.
- Extended the fixed-width scheduler corrections to new 0.6.17 call sites absent in 0.6.11.
- Ported attention Jinja `static constexpr` linkage for batch/single prefill and decode.
- Added Windows-only compact attention URIs via a decorator, preserving official URI strings on other platforms. The real long module now maps to `bpkvcd_q_bf16_kv_e4m3_o_bf16_idx_i32_qk_256_vo_256_pe_0_swa_False_lc_False_f16qk_False`; generic artifact hashing remains as a final safety net.
- Python compile, focused Windows tests (`3 passed`), and diff check pass. Next: reinstall once, relaunch, and compile the compact real prefill module.
### 2026-08-16 02:37:54 PDT — post-compaction checkpoint: 0.6.17 port reached final prefill link gap

- Clean FlashInfer 0.6.17 port: `C:\Users\Daniel\flashinfer-windows-0.6.17`, branch `windows-v0.6.17`, based on official tag `v0.6.17` / commit `a0a6b01`.
- The full isolated NVFP4 quantization + CUTLASS GEMM smoke passed, including after installation through SGLang's actual virtual environment. SGLang now imports FlashInfer 0.6.17 from the clean port.
- A real SGLang FlashInfer-prefill launch compiled every CUDA object in the compact batch-prefill JIT module. Linking now fails on exactly eight unresolved template externals.
- All eight unresolved symbols are `CTA_TILE_Q=32` batch-prefill ragged/paged variants across mask modes 0, 1, 2, and 3. Generated explicit instantiations contain CTA tile sizes 16, 64, and 128; none contains 32.
- Next action: repair the generator/dispatch coverage mismatch while preserving official Linux behavior and native-Windows compatibility, then resume the real 200K server launch.
- No server is currently live. The last hidden parent PID `23796` exited after the link failure.
### 2026-08-16 02:41:29 PDT — PE/COFF batch-prefill tile coverage repaired

- Root cause confirmed in official 0.6.17: `DISPATCH_CTA_TILE_Q` references 16/32/64/128, while the FA2 Jinja generator deliberately emits only the planner-reachable set (`16/64/128` for VO head dimension 256, `16/32` for 512+). ELF shared-library linking tolerates the unreachable undefined symbols; native Windows PE/COFF DLL linking rejects them.
- Added a Windows-only generator tile set of `16/32/64/128`; non-Windows generation retains the exact official planner-specific sets. Invalid tile/head pairs compile as guarded error stubs and remain unreachable through `FA2DetermineCtaTileQ`.
- Updated both paged and ragged explicit-instantiation templates and added platform behavior tests.
- Python compile, focused tests (`5 passed`), and `git diff --check` pass. Next: reinstall the clean port, regenerate the real module, and verify the DLL links and loads.
### 2026-08-16 02:48:34 PDT — real 0.6.17 FlashInfer prefill DLL linked, loaded, and served

- Reinstalled the clean FlashInfer port into SGLang's pip-less uv-managed virtual environment with `uv pip install --python ... --no-deps --reinstall .`; installed version remains `flashinfer-python==0.6.17` from the clean local port.
- Relaunched the exact 200K topology under hidden parent PID `29236`, with `FLASHINFER_CACHE_DIR=C:\_fij10`, FlashInfer prefill, Triton decode, 4096-token chunks, language-model-only loading, Qwen3 reasoning, and Qwen3 tool parsing.
- The regenerated real batch-prefill sources now contain all eight previously missing `CTA_TILE_Q=32` paged/ragged mask variants.
- Native linking succeeded. The real loaded DLL is `C:\_fij10\0.6.17\120f\cached_ops\bpkvcd_q_bf16_kv_e4m3_o_bf16_idx_i32_qk_256_vo_256_pe_0_swa_False_lc_False_f16qk_False\bpkvcd_q_bf16_kv_e4m3_o_bf16_idx_i32_qk_256_vo_256_pe_0_swa_False_lc_False_f16qk_False.dll`, 2,660,864 bytes, built at 02:45.
- Server startup completed with target and draft pools each at 200,000 tokens and 200,000 context; it is live on `127.0.0.1:30000`.
- Direct thinking-only API probe passed: 69 prompt tokens + 64 reasoning tokens, `reasoning_tokens=64`, non-empty preserved `reasoning_content`, empty final content because the bounded response ended during thinking. The server accepted model defaults temperature 1.0, top-k 20, top-p 0.95.
- TorchInductor's speculative target graph precompile logged several Triton `PassManager::run failed` candidate errors on SM120, then completed capture successfully in 158.59 s. This is a later dependency/performance cleanup opportunity, not a startup failure.
- Next: tool-call acceptance, longer thinking completion, vision-disabled/API verification, then controlled 0.6.17 throughput and long-context measurements.
### 2026-08-16 02:53:01 PDT — 0.6.17 behavior acceptance and unsimulated production control

- Longer thinking probe stopped normally: 69 reasoning tokens, coherent preserved `reasoning_content`, and final answer `703` for `37 * 19`.
- Auto tool selection passed with thinking preserved. The model emitted one parsed `multiply` call with JSON arguments `{"a":37,"b":19}` and `finish_reason=tool_calls`; added an optional bounded `--tool-probe` mode to the local probe utility.
- The standalone encoder-free runtime already rejects multimodal input in `TokenizerManager`, but `/model_info` still advertised the checkpoint's static image capability. Fixed it to advertise image/audio as false under `--language-model-only` while preserving VLM advertisement for disaggregated `--language-only` mode. Focused endpoint tests pass (`3 passed`), Python compile and diff check pass.
- Five normal-production 6213/512 runs on the 200K/FlashInfer-prefill server produced decode `79.294, 77.697, 73.050, 85.398, 74.951 tok/s`; mean **78.078 tok/s**. E2E mean **7.903030 s**. This unsimulated set varies with real MTP acceptance and is not the matched comparison against the old `SGLANG_SIMULATE_ACC_LEN=3` mean of 86.016 tok/s.
- FlashInfer 0.6.17 short-prefill TTFT is visibly below the prior Triton control on this shape: prompt rates were `5621.979, 4907.067, 4115.113, 4332.300, 4495.246 tok/s`. Preserve the hybrid candidate for long-context testing before selecting it as the global prefill default.
- Next: 32K/64K/199K FlashInfer 0.6.17 prefill qualification, then matched fixed-acceptance and alternate-prefill controls.
### 2026-08-16 02:57:09 PDT — FlashInfer 0.6.17 long-context qualification complete at chunk 4096

- Exact 32,000 + 16 warmed request: TTFT `7.367024 s`, observed prompt throughput **4343.681 tok/s**, E2E `7.619770 s`.
- Exact 64,000 + 16 warmed request: TTFT `20.151566 s`, observed prompt throughput **3175.932 tok/s**, E2E `20.447340 s`.
- Exact 199,000 + 16 request after prior kernel warmup: completed successfully at 199,016 total tokens, TTFT `109.872243 s`, observed prompt throughput **1811.194 tok/s**, E2E `110.320727 s`.
- This proves the full 200K request contract on FlashInfer 0.6.17, but the 199K result is slower than the prior 0.6.11 measurement (`102.042461 s`, `1950.169 tok/s`) by roughly 7.7% TTFT / 7.1% prompt rate. It is still radically faster than the unusable all-Triton long-context decay, while also trailing Triton on the 6213-token production shape.
- Current global 4096-token hybrid is therefore a capacity-valid intermediate, not yet the performance winner. Next axes: FlashInfer autotune, 8192/16384 chunks, TRT-LLM MHA feasibility, and a dynamic short/long prefill routing opportunity if fixed backends cannot win both regions.
### 2026-08-16 03:04:44 PDT — FlashInfer autotune exposes a 117 tok/s decode path and a memory-pressure tradeoff

- Enabled FlashInfer 0.6.17 autotuning under the fixed-work `SGLANG_SIMULATE_ACC_LEN=3` guard. It selected concrete CUTLASS FP4 tactics for target/draft token shapes 1/2/4 and cached them under SGLang's version/SM-specific autotune directory.
- Five exact 6213/512 runs retained the historical fixed-work digest and produced decode `115.202, 115.879, 118.607, 118.926, 117.872 tok/s`; mean **117.297 tok/s**. Mean E2E was **5.777052 s**. This is **36.37% faster** than the prior 86.016 tok/s selected fixed-work control.
- The gain is real and stable, but the tuned server retained substantially less VRAM headroom. A warmed 32K prefill fell from 4343.681 to 2543.428 tok/s; a repeat after further requests fell to 1624.045 tok/s. `nvidia-smi` showed only 189 MiB free at idle, consistent with WDDM/allocator pressure rather than an attention-kernel regression alone.
- Startup evidence: the autotuned 200K server ended pools with 1.94 GB free, began target capture with 1.31 GB, and finished all graphs with 0.89 GB; the non-autotuned server retained 1.16 GB after graphs. The four-entry Mamba cache itself uses about 0.72 GB while live requests peak at three entries.
- Added a post-autotune `torch.cuda.empty_cache()` after the synchronized profiling context so short-lived candidate workspaces are returned before graph capture. Syntax and diff checks pass.
- Next controlled launch: retain autotuning, reduce the Mamba pool from 4 to the observed required 3 entries, measure recovered free memory, recheck fixed decode and 32K prefill, then decide whether this combined path safely preserves 200K.
### 2026-08-16 03:10:31 PDT — autotune cleanup validated; Mamba pool floor is four

- A three-entry Mamba pool is invalid for this one-request, three-step MTP topology: the configurator requires four state slots per request and correctly refused startup. The observed steady usage of three still needs a fourth transient slot. Retain `MaxMambaCacheSize=4`.
- Relaunched autotuned/fixed-work with four slots after adding the post-autotune allocator flush. Cached autotuning completed immediately; target graph capture began with 1.28 GB free and all graphs ended with 1.12 GB free. Idle `nvidia-smi` showed 1,388 MiB free, materially recovering the prior launch's headroom.
- Fixed-work performance stayed intact: one exact 6213/512 run produced **117.506 tok/s** decode with the historical digest. Prompt throughput also recovered to 6886.849 tok/s.
- Warmed 32K FlashInfer prefill improved from the pressured 2543.428 tok/s to **3780.629 tok/s**, though it still trails the non-autotuned 4096-chunk measurement of 4343.681 tok/s. `/flush_cache` returns allocator blocks and restores about 1,457 MiB free before the next request; post-prefill reserved memory can leave only ~117 MiB free until that flush.
- Keep post-autotune `empty_cache`; it preserves the 36% decode gain and materially improves capacity headroom. Next: test 8192 then 16384 FlashInfer prefill chunks with the tuned decode graph, using short and long prompt controls.

### 2026-08-16 03:15:05 PDT — post-compaction checkpoint: clean 0.6.17 port live; chunk-8192 experiment in flight

- The clean FlashInfer `0.6.17` port in `C:\Users\Daniel\flashinfer-windows-0.6.17` now compiles, links, loads, and serves Qwen3.8-27B NVFP4 on native Windows/SM120. The user's original CUDA header remains untouched; the port uses a byte-identical local shim.
- Fixed the PE/COFF prefill link failure by emitting planner-reachable CTA tile-Q coverage `16/32/64/128` on Windows for both paged and ragged templates; focused tests pass and the real DLL linked/loaded.
- Thinking-path acceptance passes with reasoning preserved, parsed tool calls enabled, and vision disabled both operationally and in `/model_info`. A longer thinking probe stopped normally with 69 coherent reasoning tokens; the tool probe selected `multiply({"a":37,"b":19})`.
- Non-autotuned chunk-4096 capacity results: 32K prompt `4343.681 tok/s`, 64K `3175.932 tok/s`, and 199K `1811.194 tok/s` with all requests completing inside the exact 200K contract.
- FlashInfer autotune raised matched fixed-work decode from the historical `86.016 tok/s` to a five-run mean of **117.297 tok/s**. Retain the post-autotune allocator flush; it preserves the gain while recovering graph-capture and request headroom.
- `MaxMambaCacheSize=3` is invalid for this topology; four state slots are the proven minimum. Retain `4`.
- Current experiment: autotuned, fixed-work, chunk-8192 server on port 30000 under hidden parent PID `6024`. Its exact 6213/512 result was **111.738 tok/s** decode, already below chunk 4096 on the short shape.
- A warmed 32K/chunk-8192 benchmark is or was running in unified exec session `53025`; the last poll overflowed the available output context. Recover its result from that session or the narrow benchmark/server logs before restarting anything.

### 2026-08-16 03:19:03 PDT — reject chunk 8192; repeated 32K request collapses under tight VRAM

- The pre-compaction benchmark session could not be recovered, so the exact warmed 32K/chunk-8192 benchmark was repeated against the same live server and configuration.
- The first pre-compaction 32K sequence completed both warmup and measurement in the server log with roughly 5–6 second middle chunks. The repeated controlled run degraded sharply after allocator pressure accumulated: exact result TTFT **119.798301 s**, observed prompt throughput **267.116 tok/s**, E2E **121.202198 s**, decode **10.685 tok/s**.
- Immediately after that request, `nvidia-smi` reported only **78 MiB** free. A manual successful `/flush_cache` raised free VRAM to **2,128 MiB**, proving the severe slowdown is pressure/residency-sensitive and recoverable rather than inherent steady-state chunk throughput.
- Chunk 8192 also lost on the short fixed-work shape (`111.738 tok/s` decode versus chunk-4096's ~117 tok/s). Reject it as the production chunk size. Do not spend a 64K or 199K qualification run on this candidate.
- Next: return to chunk 4096, isolate the allocator-retention path around request completion/flush, then test backend routing and TRT-LLM MHA feasibility with headroom protected.

### 2026-08-16 03:28 PDT — first production-autotune relaunch did not reach readiness

- Returned to the selected 4096 chunk with exact 200K pools, FlashInfer prefill, Triton full-attention decode, autotune enabled, and real speculative acceptance (no fixed-acceptance benchmark guard).
- Target/draft allocation remained healthy: target FP8 KV and draft BF16 KV both held exactly 200,000 tokens; post-pool headroom was `1.85 GB` before graph capture.
- Startup remained inside the first cached FlashInfer autotune forward from `03:21:09` through `03:28`, with no completion or graph-capture marker. The process tree had spawned dozens of nested compiler/helper descendants, so it was terminated as a non-ready experimental launch rather than accepted as a server.
- This does not invalidate the measured tuned tactics. It exposes a startup/replay problem specific to the first real-acceptance relaunch that must be isolated before autotune becomes a production default.
- High-value remaining 0.6.17 capabilities found in local source: native NVFP4 KV storage with FlashInfer prefill plus TRT-LLM/XQA decode on SM120, and explicit SM120 FlashInfer GDN prefill/decode kernels. Both can recover several GB or accelerate the 48/64 GDN layers; they require controlled capability tests and the missing CUTLASS DSL Python dependency.

### 2026-08-16 03:33 PDT — native NVFP4 KV/XQA capability launch recovers over 2 GB

- Exposed controlled launcher parameters for KV-cache dtype, page size, speculative draft attention, and per-phase linear-attention backends. PowerShell help parsing and targeted diff whitespace checks pass.
- NVIDIA CUTLASS DSL `4.6.2` cannot resolve on native Windows because `nvidia-cutlass-dsl-libs-base==4.6.2` publishes no `win_amd64` wheel. Installing the metadata package alone produced a broken `.pth`; it was immediately uninstalled and the venv was verified clean. NVIDIA's current CUTLASS repository also states that CUTLASS 4.x builds are known down on Windows, making the required MLIR/CuTe runtime an upstream platform boundary rather than a missing ordinary Python package.
- Launched the independent 0.6.17 path that does not need CUTLASS DSL: target/draft KV dtype `nvfp4`, FlashInfer prefill, TRT-LLM/XQA target decode on SM120. SGLang correctly snapped requested page 1 to XQA-supported page 64.
- Exact 200K pools allocated successfully. Target KV fell from `3.05 + 3.05 GB` to `1.91 + 1.91 GB`; draft KV fell from `0.38 + 0.38 GB` to `0.30 + 0.30 GB`. Post-pool headroom rose from the prior `1.85 GB` to **4.03 GB**.
- Launch remains in CUDA-graph/kernel preparation. It is not accepted until target and draft KV-access compatibility, serving correctness, thinking/tool behavior, and measured throughput pass.

### 2026-08-16 03:42:56 PDT — target NVFP4/XQA graph builds; draft KV must stay BF16 or move to XQA

- The one-time FlashInfer SM120 CUTLASS build completed and is now cached under `C:\_fij\0.6.17\120f\cached_ops\fp4_gemm_cutlass_sm120`. Target-verify graph capture completed with **2.98 GB** still available; its first capture took `610.55 s` because it built the full native module.
- The launch then failed exactly at draft-decode capture: global `--kv-cache-dtype nvfp4` also converted the MTP draft pool, while the retained Triton draft backend cannot canonicalize `float4_e2m1fn_x2` (`KeyError`). This is a clean backend-contract failure, not a target XQA failure.
- Added optional `SpeculativeDraftKvCacheDtype` launcher control. Next launch keeps the target pool NVFP4/XQA and restores only the small draft KV pool to BF16/Triton. The expensive target module should now reuse its completed cache.

### 2026-08-16 03:46 PDT — native NVFP4 KV/XQA serves but fails the thinking quality guard

- Relaunched with target `nvfp4` KV + TRT-LLM/XQA decode, draft BF16 KV + Triton decode, FlashInfer prefill, page 64, and exact 200K pools. The cached native build reduced target graph capture to `41.50 s`; all graphs completed and the server became ready with **2.82 GB** available.
- Thinking-path requests streamed with exact token accounting, but content was corrupted. The calculator/tool probe exhausted 256 reasoning tokens in punctuation/multilingual gibberish and emitted no tool call. Two direct `37 * 19` thinking probes wandered into unrelated fabricated conversation instead of calculating, including at temperature 0.
- Reject **native NVFP4 KV storage** for production on this model despite its ~2.2 GB headroom gain. The user requires preserved useful thinking and tools, so capacity cannot outrank this acceptance failure.
- Next isolation: retain ordinary checkpoint-selected FP8 target KV and test TRT-LLM/XQA decode alone. This determines whether the quality failure belongs to native KV scale/layout handling or the SM120 XQA backend generally, while preserving the possibility of an attention-decode speedup.

### 2026-08-16 03:53 PDT — XQA itself is correct; its default workspace hurts prefill headroom

- Ordinary checkpoint-selected target FP8 KV + TRT-LLM/XQA decode + FlashInfer prefill passed preserved-thinking and tool acceptance. With an explicit calculator request it emitted coherent reasoning and the parsed call `multiply({"a":37,"b":19})` with `finish_reason=tool_calls`. This isolates the prior corruption to native NVFP4 KV storage/scale/layout, not XQA generally.
- Default-workspace XQA production measurements: 6213/512 decode **79.239 tok/s**, prompt **4797.899 tok/s**; warmed 32K/16 prompt **2636.254 tok/s**; warmed 32K/512 decode **68.517 tok/s**, prompt **2036.304 tok/s**. The 32K prefill is materially below the selected Triton-decode server's 4096-chunk control (`4343.681 tok/s`).
- XQA adds a dedicated default 512 MiB workspace on top of FlashInfer prefill's workspace. Graph completion retained only `0.73 GB`, consistent with the same WDDM/residency pressure seen in the rejected 8192-chunk experiment.
- Exposed an optional general FlashInfer workspace override and a benchmark-only fixed accepted-length launcher control, both disabled by default and restored on exit. Current experiment is XQA with a 256 MiB workspace and fixed acceptance 3; it will test whether reclaimed headroom restores prefill while providing a matched decode comparison.

### 2026-08-16 03:56 PDT — XQA + 256 MiB workspace is a new untuned winner

- Configuration: ordinary target FP8 KV, draft BF16 KV, FlashInfer prefill, TRT-LLM/XQA target decode, Triton draft decode, page 64, chunk 4096, exact 200K pools, fixed acceptance 3, FlashInfer workspace override 256 MiB, autotune disabled.
- Three exact 6213/512 runs preserved the historical digest and produced decode **110.367, 110.094, 109.923 tok/s**; mean **110.128 tok/s**. Prompt rates were `6134.682, 6114.476, 6214.135 tok/s`.
- This is already **28.03%** above the old selected fixed-work mean of `86.016 tok/s` without FlashInfer GEMM autotuning, and within 6.1% of the 117.297 tok/s autotuned-Triton mean.
- The workspace reduction also fixed long-prefill pressure: warmed exact 32K/16 completed at TTFT `6.764973 s`, prompt **4730.248 tok/s**, E2E `6.944565 s`, beating the prior selected chunk-4096 result of 4343.681 tok/s. The fixed-work output digest remained exact.
- Graph completion retained `0.92 GB`, up from default-workspace XQA's `0.73 GB`. Next: combine this XQA/workspace path with cached FlashInfer autotuned GEMM tactics, then sweep 128 MiB only if the combined path remains capacity-safe.

## 2026-08-16 04:02:05 PDT — post-compaction checkpoint (FlashInfer 0.6.17 XQA/autotune investigation)

- Clean 0.6.17 port remains in `C:\Users\Daniel\flashinfer-windows-0.6.17` on `windows-v0.6.17`, based on official v0.6.17 commit `a0a6b01`; the original fork and original CUDA headers remain untouched. The copied compatibility shim remains byte-identical to the original header (`SHA256 304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`).
- Ported Windows/MSVC/NVCC JIT support, compact cache/artifact paths, include-shim ordering, CUDA 13.3 alignment compatibility, SM120 attention compatibility, PE/COFF-safe 16/32/64/128 CTA tile-Q instantiation, NVFP4 quantization, and CUTLASS GEMM all remain validated. SGLang imports `flashinfer-python==0.6.17` from this clean port.
- Native NVFP4 KV/XQA triggered and completed the one-time full SM120 CUTLASS module build/cache in `C:\_fij\0.6.17\120f\cached_ops\fp4_gemm_cutlass_sm120`; first target graph capture took `610.55 s` and subsequent launches reuse the cache.
- Native target NVFP4 KV reduced the exact-200K target K/V pools from FP8 `3.05 + 3.05 GB` to `1.91 + 1.91 GB`, increasing post-pool headroom from about `1.85 GB` to `4.03 GB`. With draft BF16, post-pool headroom was `4.13 GB` and graph-end headroom `2.82 GB`.
- Native NVFP4 KV is rejected for production correctness: thinking/tool probes became multilingual/punctuation gibberish or unrelated hallucinated conversations and failed to emit the expected tool call. Ordinary target FP8 KV plus XQA produced coherent preserved thinking and `multiply({"a":37,"b":19})` with `finish_reason=tool_calls`, isolating the corruption to native NVFP4 KV storage/scale/layout rather than XQA.
- Default XQA workspace was functionally correct but memory-pressure limited: graph-end headroom `0.73 GB`; production 6213/512 decode `79.239 tok/s`, and warmed 32K/16 prefill `2636.254 tok/s`.
- XQA with a 256 MiB FlashInfer workspace, autotune disabled, and fixed accepted length 3 is correct and capacity-safe. Three exact 6213/512 runs were `110.367`, `110.094`, and `109.923 tok/s` (mean `110.128 tok/s`) with the expected digest. Warmed 32K/16 reached `4730.248 prompt tok/s`, TTFT `6.764973 s`, E2E `6.944565 s`; graph-end headroom `0.92 GB`.
- Current live server parent PowerShell PID is `34752`. It uses RadixArk Qwen3.8-27B NVFP4 weights; exact 200,000 context/total pools; target checkpoint FP8 KV; draft BF16 KV; FlashInfer prefill; TRT-LLM/XQA target decode; Triton draft decode; page 64; chunk 4096; 256 MiB workspace; FlashInfer autotune enabled; fixed accepted length 3; reasoning/tool parsing enabled; vision disabled/language-model-only.
- On the live autotuned XQA server, five exact 6213/512 runs were `120.130`, `121.487`, `121.349`, `120.224`, and `118.313 tok/s` (mean `120.301 tok/s`). E2E times were `5.273659`, `5.409831`, `5.436464`, `5.445719`, and `5.541230 s` (mean `5.421381 s`), each with exact digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`.
- Autotuned XQA regresses large prefill: warmed 32K/16 was `3902.091 prompt tok/s`; after manual cache flush, a no-warmup repeat was `3934.026 prompt tok/s`, both below non-autotuned XQA/workspace256 `4730.248`. The first 32K request left only `121 MiB` free VRAM; manual `/flush_cache` restored `1461 MiB`, yet the repeated rate remained about 3934, so residency alone does not explain it.
- Immediate investigation: inspect FlashInfer 0.6.17 autotuner dispatch/cache behavior and retain cached decode tactics for M=1/2/4 while bypassing any slower autotuned/fallback path for large-M prefill. Keep PID 34752 live until its evidence is fully captured.

### 2026-08-16 04:04:59 PDT — autotuner dispatch trace

- `AutoTuner.choose_one()` only applies a loaded tactic on an exact mapped cache-key hit. Outside tuning mode, a missing large-M shape returns runner 0 with tactic `-1`; 32K chunked prefill maps each FP4/FP8 GEMM's M=4096 to bucket 4096, while the persisted target/draft files contain only decode/verify M buckets 1, 2, and 4. The nominal Python dispatch therefore already bypasses decode tactics for prefill.
- SGLang's autotune contexts leave loaded configs in the singleton after the context exits. Later draft-cache loads clear and replace `_file_configs`, but the draft cache duplicates every target decode/verify FP4 and FP8 key/tactic and adds the draft-only expanded projection, so target decode remains accelerated.
- Both autotuned cache files were inspected. Target cache `3dd6f7b9700937e8` selects FP4 tactics 12/0 at M=1 and tactic 4 at M=2/4 for the two target shapes; draft cache `fd2c82615631e23d` preserves those entries and adds the 248320-wide projection. FP8 tactics likewise cover M=1/2/4 only.
- Since large-M dispatch already uses tactic `-1`, the observed 32K prefill difference is below the cache-key layer (persistent CUDA/module/workspace state, GPU run state, or control-run variance). Next decisive step is a freshly matched non-autotuned workspace256/XQA control on the current code and clocks, followed by a minimally isolated cache-replay test if the regression reproduces.

### 2026-08-16 04:08:53 PDT — matched non-autotuned XQA control confirms the prefill delta

- Stopped captured autotuned parent PID `34752` and launched the exact current-code non-autotuned control under parent PID `20032`: ordinary target FP8 KV, draft BF16 KV, FlashInfer prefill, TRT-LLM/XQA target decode, Triton draft decode, page64, chunk4096, exact200K, workspace256, fixed acceptance3. Resolved `disable_flashinfer_autotune=True`; graph-end headroom was `0.86 GB`.
- Warmed exact 32K/16: TTFT `7.151875 s`, prompt **4581.736 tok/s**, E2E `7.341123 s`. Immediate flush/no-warmup repeat: TTFT `7.167771 s`, prompt **4571.574 tok/s**, E2E `7.354395 s`. A third flushed no-warmup run under concurrent 1 Hz GPU telemetry was **4559.492 prompt tok/s**. Mean of the three current matched control rates: **4570.934 tok/s**.
- This current control is slightly below the earlier non-autotuned `4730.248` peak but remains **16.19%** above the autotuned XQA repeat at `3934.026`; the prefill regression is real and reproducible.
- GPU telemetry excludes throttling: during the third prefill, SM utilization stayed 96–99%, memory clock `13801 MHz`, core clock `2902–2925 MHz`, power `454–460 W`, temperature `54–57 C`, with zero reported power/thermal violation. After `/flush_cache`, non-autotuned free VRAM was `1486 MiB` versus autotuned `1461 MiB`, only a 25 MiB difference. The regression is a persistent kernel/runtime-state effect rather than clock or gross capacity pressure.

### 2026-08-16 04:20:28 PDT — live 0.6.17 tactic trace proves large-M fallback

- Added a temporary opt-in `FLASHINFER_AUTOTUNE_TRACE_CHOICES=1` first-seen dispatch trace to the clean 0.6.17 port, reinstalled it, and added launcher controls for trace activation plus `--flashinfer-autotune-skip-ops`. Python compilation, reinstall, installed-source verification, and PowerShell command syntax all pass.
- The real 32K request conclusively shows large-M behavior: FP4 `(4096,2560)x(2560,34816)` and `(4096,8704)x(8704,5120)` both dispatch `CutlassFp4GemmRunner tactic=-1`; all three FP8 M=4096 shapes dispatch `CublasFp8GemmRunner tactic=-1`. The cached tactics remain confined to target/draft M=1/2/4. This rules out an accidental large-M decode-tactic hit.
- Startup/runtime also exposed M=6 fallback shapes and the expected cached M=4 target-verify paths. The draft-only 248320-wide FP4 projection uses cached tactic14 at M=1 and tactic6 at M=4.
- A meta-tensor microbenchmark of 20,000 large-M `choose_one()` misses measured `11.703 us/call` with no file configs, `13.011 us/call` with the real loaded cache, and `11.998 us/call` after clearing it. The ~1.3 us lookup delta is too small to explain the multi-second prefill gap.
- The first traced 32K request included cold post-reinstall/JIT effects (`1567.454 prompt tok/s`) and is excluded. Repeats reached `4164.518` then `3455.587` as request residency accumulated; manual flush restored `1471 MiB`, matching earlier autotuned headroom. Trace itself remains diagnostic-only. Next isolate FP4 versus FP8 autotune with per-op skip launches, keeping tracing disabled for measured runs.

### 2026-08-16 04:24:51 PDT — FP4-only autotune finds a new 126 tok/s decode peak

- Launched autotuning with `flashinfer_autotune_skip_ops=['fp8_gemm']`, so FP4 CUTLASS tactics are tuned/cached while FP8 GEMMs retain heuristics. Parent PID `14908`; exact production topology otherwise unchanged. Fresh target tuning took about 6 seconds and graphs ended with `0.97 GB` headroom.
- Five warmed exact 6213/512 fixed-work runs preserved digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c` and produced decode **123.798, 126.120, 126.576, 127.401, 126.569 tok/s**; mean **126.093 tok/s**. E2E mean **5.448144 s**. This is a new decode peak: `+4.81%` over full FP4+FP8 autotune (`120.301`) and `+14.50%` over untuned XQA (`110.128`).
- The new FP4-only cache selected target FP4 tactics: 2560x34816 uses `12/12/4` at M=1/2/4; 8704x5120 uses `4/4/4`. This differs from the older full cache (`12/4/4` and `0/4/4`) and likely explains part of the decode gain.
- Long-prefill latency is unacceptable on this raw combination: repeated flushed exact 32K/16 runs stabilized at only **2875.708** and **2852.016 prompt tok/s**, despite better graph headroom. Full autotune was about 3934 and untuned about 4571 on the matched current controls. FP4-only is therefore a decode winner but not yet the production combination.
- Next isolate FP8-only autotune (`skip_ops=['fp4_gemm']`). Then combine the fastest decode tactics with an explicit prefill-time heuristic policy if operation/forward-mode isolation confirms it preserves large-M throughput.

### 2026-08-16 04:28:37 PDT — FP8-only autotune rejected; workspace is the next combined lever

- Launched FP8-only autotune with `flashinfer_autotune_skip_ops=['fp4_gemm']` under parent PID `33952`. Fresh target tuning completed in about one second; graph-end headroom was only `0.74 GB`.
- Three warmed exact 6213/512 fixed-work runs preserved the exact digest and produced decode **106.096, 108.003, 109.311 tok/s**, mean **107.803 tok/s**. This loses to untuned XQA (`110.128`) and is far below FP4-only (`126.093`), proving that the selected FP8 tactics are a net decode regression for this model/SM120 topology.
- Repeated flushed exact 32K/16 runs reached only **2613.812** and **2583.481 prompt tok/s** (mean `2598.647`). A manual flush left only `1297 MiB` free, roughly 170–190 MiB below the full/FP4-only launches. FP8-only is rejected on decode, prefill, and capacity.
- Per-op conclusion: FP4 autotune is the sole decode win; FP8 autotune should remain skipped. The raw FP4-only/workspace256 path gives the best decode but poor prefill. Next sweep workspace128 (then64 only if safe) on FP4-only to recover enough prefill headroom while retaining the 126 tok/s graph.

### 2026-08-16 04:31:55 PDT — FP4-only autotune + workspace128 is the new combined winner

- Launched the FP4-only cached autotune path (`skip_ops=['fp8_gemm']`) with the FlashInfer workspace reduced from 256 to **128 MiB**, parent PID `20624`. All other exact-200K/XQA/page64/chunk4096/fixed-work controls remained unchanged.
- Graph-end headroom increased from FP4-only/workspace256 `0.97 GB` to **1.14 GB**. The selected decode graph remained fully valid.
- Five warmed exact 6213/512 runs retained the exact historical digest and produced decode **127.690, 126.919, 127.371, 125.796, 125.298 tok/s**; mean **126.615 tok/s**. E2E times were `4.823261, 4.846588, 4.833330, 4.895535, 4.912177 s`; mean **4.862178 s**. Prompt rates stayed `7450–7573 tok/s` instead of decaying under workspace256.
- Two flushed exact 32K/16 no-warmup runs produced prompt **5740.299** and **5660.561 tok/s**, TTFT `5.708414` and `5.788825 s`, E2E `5.876050` and `5.959567 s`, with exact output digest. This beats current matched non-autotuned workspace256 (`4570.934`) by about **24.71%**, full autotune workspace256 (`3934.026`) by about **44.91%**, and the earlier best non-autotuned workspace256 peak (`4730.248`) by about **20.51%**.
- This is the first combination that wins both decode and long prefill. Next: test workspace64 once. Retain workspace128 immediately if 64 fails, regresses, or destabilizes; then qualify 32K/512, 64K, and near-limit 199K on the winner.

### 2026-08-16 04:33:36 PDT — workspace128 is the exact functional floor

- The otherwise identical FP4-only/workspace64 launch failed deterministically during target graph capture. FlashInfer requested `133,693,440` bytes for `batch_prefill_tmp_v`, while the 64 MiB aligned workspace provided only `67,108,864` bytes; it raised the intended buffer-overflow error and the server exited cleanly.
- A 128 MiB workspace is `134,217,728` bytes, leaving only `524,288` bytes above this real maximum request. It is therefore the smallest whole-MiB power-of-two setting that satisfies the current target graph, and the prior 128 MiB winner is retained. Do not test 64 again.

### 2026-08-16 04:38:29 PDT — winner qualifies 32K decode, 64K prefill, and the exact 199K request

- Relaunched the cached FP4-only/workspace128 fixed-work winner under parent PID `32728`. Graph-end headroom improved further to `1.29 GB`; startup and all target/draft captures completed normally.
- Warmed exact 32K/512 completed at **5777.726 prompt tok/s**, TTFT `5.671435 s`, **102.905 decode tok/s**, E2E `10.637201 s`, with the exact fixed-work digest. This is dramatically above the default-workspace XQA 32K/512 result (`2036.304` prompt, `68.517` decode).
- Exact 64K/16 completed at **4399.968 prompt tok/s**, TTFT `14.894653 s`, E2E `15.095667 s`, exact digest. This is `38.54%` above the prior non-autotuned FlashInfer 0.6.17 64K result (`3175.932`).
- Exact 199K/16 completed inside the real 200,000-token pool at **2200.563 prompt tok/s**, TTFT `90.431423 s`, E2E `90.782955 s`, total `199,016` tokens, exact digest. This is `21.50%` above the prior 0.6.17 result (`1811.194`) and `12.84%` above the old 0.6.11 result (`1950.169`).
- Capacity, correctness digest, short decode, long-context decode, and the near-limit context contract all pass on the selected fixed-work topology. Next remove the temporary trace instrumentation, promote clean launcher/OpenCode defaults, then relaunch without simulated acceptance for reasoning/tool/cancellation/vision/near-limit production acceptance.

### 2026-08-16 04:50:00 PDT — clean production defaults and OpenCode2 acceptance pass

- Removed the temporary tactic trace completely, restored `flashinfer/autotuner/autotuner.py` byte-for-byte to its clean 0.6.17 source hash, and reinstalled the clean port. Source and installed autotuner SHA256 both equal `D1E9EBC15F3F55E81167A7D1BE3FCEE7BE8122D901A00ACAA635E6FBDE2900F3`.
- Promoted Windows launcher defaults: exact context/active pool `200000`, page64, chunk4096, FlashInfer prefill, TRT-LLM/XQA target decode, Triton draft, FP4 CUTLASS, FlashInfer autotune enabled with `fp8_gemm` skipped, 128 MiB FlashInfer workspace, three steps/four draft tokens, scheduler interval4, torch compile, language-model-only reasoning/tool parsing, and no simulation.
- Promoted both OpenCode declarations to a `200000` context limit with text-only/tool-capable main-model capabilities, preserved thinking, reasoning-field compatibility, 8192 output cap, and the selected Qwen sampling profile. Global `opencode.json` parses; both PowerShell wrappers render syntax.
- Untouched default production launch is live under parent PID **4996**. Resolved server arguments exactly match the promoted topology; target and both draft graphs captured, graph-end headroom `1.17 GB`, and the server reached ready normally.
- Production thinking passed: coherent preserved reasoning, `reasoning_tokens=111`, normal stop, final `703`. Production tool parsing passed with preserved reasoning, `finish_reason=tool_calls`, and `multiply({"a":37,"b":19})`.
- `/model_info` reports `has_image_understanding=false`, `has_audio_understanding=false`, and generation architecture `Qwen3_5ForConditionalGeneration`, confirming operational vision disablement.
- Live cancellation passed: an 8192-token thinking stream was disconnected after 10 seconds (`31,968` bytes received); exactly two already-in-flight packets arrived after local state deletion, the count stopped at two, and an immediate thinking health request completed normally with final `10`.
- Real standalone OpenCode2 acceptance passed through the ordinary `llama-cpp/qwen3.8-27b` alias: visible coherent thinking, a successful `read` tool call against the launcher, retained tool output, and the correct concise statement of `200K` / FlashInfer / TRT-LLM-XQA / `128 MiB`.
- Five unsimulated 6213/512 production runs produced decode `86.717, 98.369, 95.990, 95.383, 95.134 tok/s`; mean **94.319 tok/s**. Mean prompt throughput **7583.006 tok/s**, mean E2E **6.247350 s**. Real proposal acceptance varies; fixed-work qualification remains the controlled peak at `126.615 tok/s`.
- Remaining credible opportunities: repair the two local Triton AOT scalar-typing failures currently falling back during torch compilation, then test TRT-LLM/XQA for the BF16 draft attention path once. Preserve PID 4996 until this production evidence is checkpointed (now complete).
## Post-compaction checkpoint — Sun Aug 16 04:59:10 PDT 2026

- FlashInfer 0.6.17 port remains clean in `C:\Users\Daniel\flashinfer-windows-0.6.17` (`windows-v0.6.17`, official v0.6.17 base `a0a6b01`). The user's original `C:\Users\Daniel\flashinfer-windows\csrc\nv_internal\cuda.h` remains protected and untouched; clean-port copy previously matched SHA256 `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.
- Selected winner: FP4-only FlashInfer autotune (`fp8_gemm` skipped), TRT-LLM/XQA target decode, FlashInfer prefill, page size 64, chunk 4096, and the exact functional 128 MiB workspace floor. Fixed accepted-length-3 five-run decode mean is **126.615 tok/s**, mean E2E **4.862178 s**.
- Long-context measurements: 32K/16 prompt rates **5740.299** and **5660.561 tok/s**; 32K/512 **5777.726 prompt / 102.905 decode tok/s**; 64K/16 **4399.968 prompt tok/s**; 199K/16 **2200.563 prompt tok/s** at exact total **199016 tokens**.
- Clean production defaults are promoted. Global OpenCode config and wrapper expose a real **200000-token** context. Production acceptance passed thinking preservation, real tool use, vision disabled, cancellation cleanup, standalone OpenCode2, and unsimulated fixed-work decode mean **94.319 tok/s**.
- Temporary FlashInfer tactic tracing was removed from source and installed package; clean source/install autotuner SHA256 previously matched `D1E9EBC15F3F55E81167A7D1BE3FCEE7BE8122D901A00ACAA635E6FBDE2900F3`.
- Experimental SGLang scalar-`tl.constexpr` compiler patch currently touches `fp8_kernel.py` and `fused_sigmoid_gating_recurrent.py`; focused smoke and tests passed. Compiler-fix server parent PID **29648** had target graph capture running for more than five minutes (prior baseline about 35 seconds), with no earlier specific f64-conversion or loop-carried-variable failures before the last output overflow. Current process/log status is unknown at this checkpoint.
- Immediate next action: inspect PID 29648 and narrow capture/error markers. If capture is still pathological, stop its tree and revert only the four experimental `tl.constexpr` annotations, validate, and record rejection. The remaining optional draft TRT-LLM attention test is worthwhile only after the compiler path is clean or reverted. Finish with focused validations and leave the fastest validated unsimulated production server live.

### 2026-08-16 05:00 PDT — scalar-constexpr compiler repair is rejected

- Compiler-fix PID `29648` eventually completed target-verify graph capture in **265.37 s** and reached ready at `04:57:14`. The specific prior f64-to-FP8 and loop-carried-variable failures were absent, so the annotations did make those Triton fragments compile.
- Three exact fixed-acceptance 6213/512 runs preserved digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c` and produced **124.530, 124.127, 121.568 tok/s**, mean **123.408 tok/s**; mean E2E **4.925112 s**.
- This loses about **2.53%** against the selected five-run `126.615 tok/s` mean while increasing target capture from roughly 35 seconds to more than four minutes. Reject the scalar-constexpr patch on startup and steady-state performance. Stop this server and revert only the four experimental annotations before any remaining backend test.
- PID `29648` and its process tree were stopped. All six experimental annotations (the two FP8 bounds plus four GDN scalars) were reverted with a surgical patch; both files pass `py_compile` and `git diff --exit-code` confirms they are byte-clean against the branch.

### 2026-08-16 05:03 PDT — draft XQA uncovers one final Windows source collision

- Tested the remaining draft-attention axis with the selected topology plus `--speculative-draft-attention-backend trtllm_mha` and fixed acceptance 3, parent PID `3440`.
- Target XQA captured normally in `35.90 s`. Draft XQA reached its first BF16-KV/head-dim-256/group-ratio-6 FlashInfer JIT build, then failed in `csrc/xqa/mha_stdheaders.cuh`: Windows SAL defines `__in` as an annotation macro, erasing the tuple implementation's local parameter named `__in` and producing repeated `expected an expression` diagnostics such as `._M_tail()`.
- This is a narrow FlashInfer 0.6.17 Windows-port gap rather than a backend incompatibility. The failed parent exited. Next rename the private tuple parameter throughout that header to a collision-free identifier, reinstall the clean port, rerun the focused FlashInfer tests, and retry the exact draft-XQA launch.
- Renamed every private `__in` tuple parameter in `csrc/xqa/mha_stdheaders.cuh` to `other`, avoiding global preprocessor state and leaving the user's CUDA shim untouched. Reinstalled local `flashinfer-python==0.6.17`; the installed header contains the collision-free source.
- Whole `tests/jit/test_jit_cpp_ext.py` currently reports **22 passed / 3 failed**. The failures are existing Windows debug/release flag-expectation gaps (`--device-debug` and POSIX `-DNDEBUG` assertions), unrelated to the XQA header change; retain them for final port-hardening review after the draft backend result.
- The next BF16 draft-XQA compile exposed `std::stoi`, which CUDA's reduced XQA standard-header surface does not provide on this Windows path. Replaced all three equivalent XQA host-side environment parses (`mha.cu`, `mha_sm90.cu`, `mla_sm120.cu`) with `std::strtol`, reinstalled 0.6.17, and confirmed draft-decode XQA now builds and captures in **6.28 s**.
- Draft-extend then reached XQA and correctly asserted that speculative `q_len=4` needs an explicit packed mask. Added a backend-owned uint16 causal mask to SGLang's TRT-LLM MHA draft-extend CUDA-graph metadata and passed it only on draft-extend. The static graph-stride test now also locks the exact two-word rows `[1,3,7,15]`; targeted pytest passes and diff check is clean. Next launch determines full serving correctness and performance.
- Draft XQA graphs now all capture, but first eager draft-prefill progressively exposed TRT-LLM-Gen Windows loader gaps. Gated CUDA 13.3 Rubin-only private oversized-shared-memory enums off on Windows (irrelevant to this SM120), replaced host-only `__builtin_ctz` with a portable power-of-two shift count, and reached a successfully linked `fmha_gen` DLL.
- The linked Windows DLL did not export its two ctypes cubin callbacks. Added a narrow `_WIN32` `__declspec(dllexport)` C-API macro in `include/flashinfer/cubin_loader.h` for `FlashInferSetCubinCallback` and `FlashInferSetCurrentCubin`; source and installed header are aligned for the next incremental rebuild. No CUDA header was edited.
- User-visible `step1-T5_1786880606` / `step2` repetition occurred while the candidate was launched with benchmark-only `SGLANG_SIMULATE_ACC_LEN=3`. Forced acceptance intentionally bypasses semantic rejection and can corrupt interactive output; use it only for matched throughput/digest measurement. Final semantic acceptance and live production must run with simulation disabled.
- Forced the generated `fmha_gen.dll` and its launcher object to rebuild; both were disposable/recoverable JIT cache artifacts. The callbacks then exported and loaded correctly, and the next true boundary surfaced: TRTLLM-Gen context kernels support SM100/103/107, not SM120. This is architectural, so do not weaken the runtime guard.
- Added an SM120 draft split: the draft runner resolves initial prefill to the configured FlashInfer backend while retaining TRT-LLM/XQA for draft decode and `DRAFT_EXTEND_V2`. Eagle now reuses the runner's already model-wrapped split backend instead of constructing a second all-TRT draft-extend backend. Resolver/routing tests pass (`2 passed`), and the Eagle backend lifecycle file passes (`9 passed`, `7 subtests`).

### 2026-08-16 05:43 PDT — split draft XQA is a new fixed-work decode winner

- The SM120 split candidate reached ready under parent PID `16180`: FlashInfer initial draft prefill, TRT-LLM/XQA draft decode, and the selected target topology. Target verify captured in `33.62 s`; draft decode captured in `1.14 s`; the split deliberately leaves draft-extend eager (`draft_extend=0.00`) while routing that mode to XQA.
- First three exact 6213/512 runs were `133.294`, `132.747`, and `132.770 tok/s`. Two transient overlapped/residency-pressured runs fell to `108.249` and `109.246`, then performance immediately recovered.
- The subsequent clean five-run window produced **133.019, 133.529, 133.627, 132.410, 133.576 tok/s**, mean **133.232 tok/s**; mean E2E **4.643008 s** and mean prompt throughput **7694.395 tok/s**. Every run retained exact digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`.
- This is **5.23%** above the prior selected fixed-work mean `126.615 tok/s`. Qualify 32K/16, 32K/512, 64K, and near-limit 199K before promotion; then run unsimulated semantic/tool acceptance because fixed acceptance cannot establish quality.

### 2026-08-16 05:45:03 PDT — post-compaction checkpoint

- FlashInfer `0.6.17` is ported in the clean `C:\Users\Daniel\flashinfer-windows-0.6.17` worktree. The user's CUDA header and its clean-copy shim remain protected; last known SHA-256 for both was `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`. Recheck before handoff.
- Prior production winner: FP4-only autotune (`fp8_gemm` skipped), target TRT-LLM/XQA decode, FlashInfer prefill, page size 64, chunk 4096, and the exact 128 MiB workspace floor. Fixed-work mean `126.615 tok/s`; long-context rates and thinking/tool/vision-disabled/OpenCode acceptance passed.
- Rejected compiler experiment: adding scalar `tl.constexpr` annotations made target capture take `265.37 s` and averaged only `123.408 tok/s` (`124.530/124.127/121.568`); it was reverted byte-clean.
- Draft-XQA Windows/supporting-dependency gaps fixed: renamed the SAL-colliding XQA tuple parameter `__in` to `other`; replaced `std::stoi` with `std::strtol` in three XQA CUDA sources; added SGLang's packed causal `uint16` XQA mask for `DRAFT_EXTEND_V2` plus graph-metadata coverage; gated CUDA 13.3 Rubin-only private FMHA enums off Windows; replaced `__builtin_ctz` with a portable shift loop; exported cubin-loader callbacks with `__declspec(dllexport)` on Windows. Generated `fmha_gen.dll` and one launcher object were deleted and rebuilt as recoverable JIT artifacts.
- TRTLLM-Gen context remains architecturally unsupported on SM120. The SGLang split uses FlashInfer for initial draft prefill and TRT-LLM/XQA for draft decode and `DRAFT_EXTEND_V2`; Eagle reuses the split runner. Passing coverage: attention setup `2`, Eagle `9` plus `7` subtests, graph mask `1`.
- The repeated `step1-T5_1786880606` / `step2` output occurred with `SGLANG_SIMULATE_ACC_LEN=3`. Forced acceptance corrupts semantics and must never be used for semantic qualification.
- Candidate parent PID `16180` was likely live at compaction. It reached ready with target capture `33.62 s`, draft decode capture `1.14 s`, and deliberately eager XQA draft-extend (`0.00 s` graph capture).
- Clean consecutive five-run window: `133.019`, `133.529`, `133.627`, `132.410`, `133.576 tok/s`; mean **`133.232 tok/s`**, mean E2E **`4.643008 s`**, prompt mean **`7694.395 tok/s`**, exact digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`. This is **5.23%** above `126.615`. Earlier candidate runs were `133.294/132.747/132.770`, followed by transient `108.249/109.246`, then immediate recovery to the clean five.
- Next: confirm PID `16180`; qualify `32K/16` twice, `32K/512`, `64K/16`, and `199K/16`; then reinstall from clean FlashInfer source, run unsimulated thinking/tool/vision-disabled acceptance, measure unsimulated fixed work, promote only if sound, complete focused tests and hash checks, and leave the fastest validated production server live.

### 2026-08-16 05:48 PDT — draft-XQA long-context qualification in progress

- PID `16180` and its server process tree remain live; port `30000` is listening.
- Exact cold `32K/16` runs passed with identical digest `9a67b36acb65b4e41a889858333e74da5b4949fa96e24a21b5a8edcf0be66fe4`: prompt/decode `5967.343/128.107` and `6052.577/124.687 tok/s`.
- Warmed `32K/512` passed with exact digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`: prompt `4304.162`, decode `103.820 tok/s`. Decode is slightly above the prior winner's `102.905`, while this single prompt rate is lower than its prior `5777.726`; keep investigating rather than infer from one run.
- First `64K/16` attempt timed out while opening a tokenizer calibration connection. The process tree survived and the socket remained listening. Immediate retry passed exact `65536 + 16` with prompt `3131.863`, decode `90.351 tok/s`, same short-output digest. This prompt rate is below the prior `4399.968`; repeat after near-limit qualification and inspect logs for the transient stall.

### 2026-08-16 05:50 PDT — draft-XQA context ladder passed

- Exact near-limit `199000 + 16 = 199016` passed: prompt **`2429.153 tok/s`**, decode `98.425 tok/s`, TTFT `81.922 s`, same deterministic short-output digest. This exceeds the prior selected topology's near-limit prompt rate `2200.563 tok/s`.
- A fresh exact `64K/16` immediately afterward passed at prompt **`4656.795 tok/s`** and decode `115.872 tok/s`, above the prior `4399.968` prompt rate. The earlier `3131.863` result and one connection timeout were transient; the endpoint recovered without restart.
- Context qualification is complete: two `32K/16`, one `32K/512`, two successful `64K/16`, and one exact `199K/16`. Candidate remains eligible for unsimulated semantic/tool acceptance and production promotion.

### 2026-08-16 05:55 PDT — clean reinstall and unsimulated candidate ready

- Stopped fixed-work candidate tree rooted at PID `16180` after qualification.
- Reinstalled FlashInfer from `C:\Users\Daniel\flashinfer-windows-0.6.17` with `uv pip install --no-deps --reinstall .`; install completed successfully and the SGLang venv reports `flashinfer 0.6.17` from its site-packages.
- Launched unsimulated (`SGLANG_SIMULATE_ACC_LEN` absent, launcher value `0`) draft-XQA candidate rooted at PID `13084`, logs `sglang-qwen-fi0617-draft-trtllm11.*.log`.
- Clean-source launch resolved the intended split and became ready at `05:55:08`: target verify capture `33.52 s`, draft decode `6.14 s` on the first post-reinstall build, draft extend eager `0.00 s`, available GPU memory `1.17 GB`, exact `200000` token pool. Proceeding to real thinking/tool/vision-disabled acceptance before changing launcher defaults.

### 2026-08-16 05:57 PDT — unsimulated semantics pass; performance needs another window

- Thinking at the recommended Qwen sampling profile (`temperature=1.0`, thinking enabled/preserved) stopped normally with `118` reasoning tokens and a coherent correct `12 * 17 = 204` final. The model emitted Unicode replacement glyphs where punctuation was expected; arithmetic and reasoning were intact, and this should be tracked separately from draft-XQA correctness.
- Real tool parsing passed: normal preserved reasoning and exactly one `multiply` call with `{"a":37,"b":19}`, `finish_reason=tool_calls`.
- `/model_info` reports `has_image_understanding=False` and `has_audio_understanding=False`; vision remains disabled by the language-model-only launch.
- First five unsimulated exact `6213/512` runs: `99.166`, `97.608`, `108.475`, `82.336`, `72.482 tok/s`; mean **`92.013 tok/s`**, mean E2E **`6.587488 s`**, prompt mean **`6915.069 tok/s`**. This is `2.44%` below the prior Triton-draft production mean `94.319`, with the first three strong and the last two dragged down by lower real proposal acceptance. Temperature-zero output also changed digest on the fifth run, which can occur with nondeterministic draft proposal sampling but warrants a recovery window and semantic comparison.
- GPU is at `32035/32607 MiB` used with `153 MiB` WDDM free after lazy Triton kernels loaded. Run another clean consecutive window before deciding promotion; fixed accepted-length performance alone is insufficient.

### 2026-08-16 05:59 PDT — unsimulated recovery window wins

- Second consecutive unsimulated `6213/512` window: `101.834`, `105.244`, `100.471`, `89.973`, `83.029 tok/s`; mean **`96.110 tok/s`**, mean E2E **`6.215101 s`**, prompt mean **`7333.339 tok/s`**. This is **1.90% above** the prior Triton-draft production mean `94.319`.
- Combined ten-run draft-XQA mean is `94.062 tok/s`, only `0.27%` below the old five-run mean and dominated by nondeterministic real acceptance paths. Controlled fixed-acceptance work remains a clear `5.23%` kernel win, while the second real window confirms the gain can survive rejection sampling.
- The old Triton-draft production log shows the same wide acceptance-rate swings (`~0.28–0.63`) and corresponding throughput variation. Draft XQA's observed variability is therefore consistent with proposal acceptance rather than an XQA-only stability failure. Candidate remains promotable after final code/test/hash checks.

### 2026-08-16 06:01 PDT — promotion and focused validation pass

- Promoted `SpeculativeDraftAttentionBackend = 'trtllm_mha'` in `scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1`; PowerShell resolves the full parameter syntax successfully.
- Closed the remaining FlashInfer JIT flag gaps: Windows debug JIT now honors `FLASHINFER_JIT_DEBUG=1`, uses MSVC `/Od /Zi`, passes NVCC's sccache-compatible `--device-debug`, and omits both host/device `NDEBUG`; release mode keeps `/DNDEBUG` for MSVC and `-DNDEBUG` for NVCC. Parameterized Linux/Windows tests cover both modes.
- Whole `tests/jit/test_jit_cpp_ext.py`: **28 passed**.
- SGLang focused suite covering XQA graph metadata/mask, attention-backend setup and SM120 split, Eagle split reuse, HTTP warmup, and tokenizer RID cleanup: **184 passed, 7 subtests passed**.
- `git diff --check` passes in both SGLang and clean FlashInfer 0.6.17 worktrees.
- Protected original and clean-copy CUDA headers still both hash `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.
- Final sequence: stop PID `13084`, reinstall the now-final FlashInfer source once more, verify source/install identity, relaunch the untouched promoted defaults, confirm ready/backend/no simulation, run a final thinking/tool health probe, and leave that server live.

## Final FlashInfer 0.6.17 port checkpoint — Sun Aug 16 06:08:31 PDT 2026

- Final FlashInfer source was rebuilt and reinstalled successfully as `flashinfer 0.6.17`. Source/install `flashinfer/jit/core.py` both hash `4D760AA002D8EE594291536CAD8F078D88F308FEB05C4FBF290788966BC7DACC`.
- Source/install header pairs match: XQA tuple header `2215B3A4959D78255680AA44D9E83185680A5FE725049FF82CC40A0AA8CF0106`; TRTLLM-Gen FMHA header `1A97573111EE5348B7B8E9AE37A93790F25EF49E4FE01B69BCD435F204E58F50`; cubin loader `650E60D427E3DC7EFEABC446A90BCB418E8300DB5B6D444DBFDCDD3534E011F0`.
- The user's original CUDA header and the clean-port copy remain untouched and identical at `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.
- Untouched promoted defaults are live under hidden PowerShell PID **`34364`**, logs `sglang-qwen-fi0617-production-xqa.*.log`. Resolved draft backend is `trtllm_mha` with the SM120 FlashInfer-prefill/XQA-decode split. Exact token pool/context are `200000`; target capture `33.63 s`, draft decode `6.00 s`, draft extend eager `0.00 s`, graph-end GPU headroom `1.27 GB`; ready at `06:06:52`.
- Production is genuinely unsimulated: a live request logged variable `accept len=3.02`, `accept rate=0.68`, rather than forced `3.00` behavior.
- Final default acceptance: thinking preserved with `57` coherent reasoning tokens and final `703`; tool parser emitted exactly `multiply({"a":37,"b":19})`; `/model_info` reports image/audio understanding false; standalone OpenCode2 exited `0` in `4.51 s` with `XQA READY`.
- Selected performance: controlled five-run mean **`133.232 tok/s`**, **5.23%** above the prior `126.615`; real recovery-window mean **`96.110 tok/s`**, **1.90%** above prior production `94.319`; ten-run real mean `94.062` reflects the same wide proposal-acceptance variation present in the prior backend. Long ladder passed through exact total `199016`, including `2429.153` prompt tok/s near the limit and recovered `4656.795` at 64K.
- Final validation: FlashInfer JIT file **28 passed**; SGLang focused suite **184 passed plus 7 subtests**; both worktrees pass `git diff --check`; launcher syntax resolves; OpenCode wrapper executed successfully. PID `34364` is the fastest validated reasoning/tool/vision-disabled 200K NVFP4 production server and is intentionally left live.

## Exhaustive-goal continuation checkpoint — Sun Aug 16 06:10:55 PDT 2026

- The 0.6.17/XQA production handoff is valid, but the broader request is to exhaust runtime/kernel/supporting-dependency opportunities. Reopened the completion audit instead of treating the port alone as the end state.
- Current untouched winner remains healthy under PID `34364`; preserve it until the next isolated candidate is ready to launch.
- Newly confirmed high-value Windows gap: `python/sglang/srt/layers/sampler.py` excludes **all** FlashInfer sampling imports on `win32` because upstream grouped them with unavailable `sgl_kernel` renormalizers. Installed FlashInfer 0.6.17 actually exposes `top_k_top_p_sampling_from_probs`, `min_p_sampling_from_probs`, `top_k_renorm_prob`, and `top_p_renorm_prob` on Windows. The launcher therefore forces the slower PyTorch sampler even though the accelerated backend is locally available.
- This sampler runs in ordinary sampled reasoning and throughout the speculative draft/verification path, so it is a credible per-step decode lever. Next: split the import boundary so Windows uses FlashInfer's own renormalizers, add a launcher sampling-backend control, test the JIT kernel directly, then run matched fixed-acceptance and real-acceptance windows. Retain only a measured semantic-safe win.
- Other remaining audit surfaces to resolve after sampling: draft FP8 KV under XQA, the skipped two-step MTP control, and the Windows-unavailable CUTLASS DSL/FlashInfer GDN path. Existing evidence already rejects native target NVFP4 KV, chunk 8192/16384, cuDNN FP4, alternative torch-compile modes, workspace below 128 MiB, and the scalar-constexpr compiler experiment.

### 2026-08-16 06:16 PDT — FlashInfer sampling works; first real window is inconclusive

- Split `sampler.py`'s CUDA import boundary: Windows imports FlashInfer sampling plus FlashInfer's own top-k/top-p renormalizers, while non-Windows retains the established `sgl_kernel` renormalizers. Import smoke confirms all three resolved callables come from `flashinfer.sampling` on this host.
- Added a measured launcher control `-SamplingBackend pytorch|flashinfer`, leaving the production default at `pytorch` during qualification.
- FlashInfer sampling candidate parent PID `34928` resolved `sampling_backend='flashinfer'`, captured the selected graphs, and reached ready. Its first real top-k/top-p thinking request triggered and successfully built `C:\_fij\0.6.17\120f\cached_ops\sampling\sampling.dll` in about 46 seconds; coherent preserved reasoning stopped normally with final `43`.
- First unsimulated exact `6213/512` window: `95.936`, `99.206`, `105.354`, `80.470`, `79.738 tok/s`; mean **`92.141 tok/s`**, mean E2E **`6.531587 s`**, prompt mean **`6909.429 tok/s`**. This closely matches the earlier draft-XQA first window (`92.013`) and is below its recovery window (`96.110`); real proposal-acceptance variance hides any sampler delta.
- Next isolate raw overhead with fixed accepted length 3 and identical digest, then add a sampled-profile benchmark surface because the existing throughput harness hardcodes greedy temperature zero and cannot directly measure the user's normal top-k-20/top-p-0.95 path.

### 2026-08-16 06:26 PDT — FlashInfer sampling is a measured sampled-decode win

- Extended `bench_openai_stream.py` with opt-in temperature/top-p/top-k/min-p/presence/repetition controls while preserving the historical temperature-zero defaults. This creates a controlled throughput surface for the user's actual Qwen general-thinking profile.
- Greedy fixed-work is unchanged, as expected: FlashInfer sampling clean runs cluster around the existing `133.232 tok/s` winner (`132.837/133.145/133.497/133.431`; mean `133.228`). Sampling does not perturb the temperature-zero fast path or digest.
- Fixed accepted-length-3, sampled-profile (`temperature=1.0`, top-k `20`, top-p `0.95`, min-p `0`, presence `1.5`, repetition `1.0`) clean short windows: FlashInfer `128.553/128.643/129.421/128.774`, mean **`128.848 tok/s`**; PyTorch `128.150/128.182/128.329`, mean **`128.220 tok/s`**. FlashInfer gains about `0.49%`.
- Stronger 4096-output sampled-profile control averages residency cycles inside one request with identical forced-work digest: PyTorch **`128.684 tok/s`**, E2E `32.629860 s`; FlashInfer **`129.382 tok/s`**, E2E `32.479891 s`. FlashInfer is **`0.542%` faster** and saves `0.150 s` over 4096 generated tokens.
- Candidate graph-end headroom is `1.17 GB` after the sampling module is resident, versus about `1.27 GB` for PyTorch. This remains above the proven functional floor, but 199K capacity must be rechecked after promotion.
- Promote FlashInfer sampling only after focused import/launcher tests, then validate real unsimulated thinking/tools and exact near-limit capacity. Current fixed candidate PID is `24892` and must not be used for semantic acceptance.

## Post-compaction checkpoint — Sun Aug 16 06:30:02 PDT 2026

- Active goal remains open: exhaust credible runtime, kernel, and supporting-dependency performance opportunities for reasoning-enabled, vision-disabled, tool-enabled, thinking-preserved, real-200000-context Qwen3.8-27B NVFP4. The FlashInfer `0.6.17` port alone is not the completion boundary.
- Clean `0.6.17`/XQA winner before this continuation: target+draft XQA split with FlashInfer prefill, controlled **`133.232 tok/s`**, real recovery **`96.110 tok/s`**, and long-context validation through exact total `199016`. Prior production PID `34364` was intentionally stopped to qualify further candidates.
- FlashInfer sampling Windows gap is fixed: `sampler.py` imports FlashInfer sampling and its own top-k/top-p renormalizers on Windows while non-Windows retains `sgl_kernel` renormalizers. Launcher `-SamplingBackend` is present and promoted to default `flashinfer`; the benchmark now accepts opt-in temperature/top-p/top-k/min-p/presence/repetition flags while preserving temperature-zero defaults.
- Sampling kernel built successfully at `C:\_fij\0.6.17\120f\cached_ops\sampling\sampling.dll`; sampled thinking smoke preserved coherent reasoning and final `43`. First unsimulated window mean `92.141 tok/s` was inconclusive. Fixed greedy remained `133.228`; fixed sampled short was FlashInfer `128.848` versus PyTorch `128.220` (**+0.49%**); fixed sampled 4096 was FlashInfer `129.382`/E2E `32.479891 s` versus PyTorch `128.684`/E2E `32.629860 s` (**+0.542%**, `0.150 s` saved). FlashInfer sampling is promoted, with unsimulated and exact-199K requalification still due after remaining candidates.
- Current candidate is draft FP8 KV under parent PID `34796`, launched with `-SpeculativeDraftKvCacheDtype fp8_e4m3 -SimulateAcceptedLength 3`. Target FP8 K/V allocation is `3.05+3.05 GB`; draft FP8 K/V is `0.19+0.19 GB` versus BF16 `0.38+0.38 GB`. Target capture was `33.06 s`, draft capture `11.36 s`, and graph-end headroom `1.73 GB`.
- First warmup failed at `trtllm_mha_backend.py:1374` -> FlashInfer `xqa.py:375` with `AssertionError: Output and query must have the same dtype`. Immediate task: inspect `q_data_type`/`out_dtype` plumbing and determine whether a narrow, correctness-preserving BF16-query/output plus FP8-KV fix can enable draft XQA; test it only if the contract supports it.
- Remaining credible surfaces after draft FP8: current-topology two-step MTP control, CUTLASS DSL/FlashInfer GDN Windows feasibility, top-k-2 speculative-tree source audit, and profiling-guided hotspot review. Final production still requires unsimulated thinking/tool/OpenCode checks and exact `199016` capacity with the winning defaults.
- Protect both CUDA headers. Original and clean-port copy last verified identical at `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.

### 2026-08-16 06:37 PDT — draft FP8 KV is functional and wins the fixed control

- Root cause of the FP8-draft warmup failure was the new XQA `DRAFT_EXTEND_V2` route inheriting TRTLLM-Gen's query cast: FP8 KV made `forward_extend` cast Q to FP8 while passing model/BF16 as `out_dtype`, violating XQA's same-Q/O-dtype contract. Normal XQA decode already keeps Q/O in model dtype.
- Narrow fix in `trtllm_mha_backend.py`: every XQA extend mode now preserves the model-dtype query/output while FP8 remains confined to the KV cache. Non-XQA TRTLLM-Gen retains its FP8 query cast. Python compilation passed.
- Fixed-acceptance relaunch is healthy under parent PID `31352`. Target and draft both allocate FP8 KV at `3.05+3.05 GB` and `0.19+0.19 GB`; target capture `33.45 s`, draft decode capture `1.15 s`, final graph headroom **`1.49 GB`**. Health returns HTTP 200 and the prior dtype assertion is absent.
- Five exact `6213/512`, temperature-zero, forced-acceptance-3 runs preserved digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c` and produced **`135.364`, `135.826`, `135.074`, `134.968`, `134.601 tok/s`**; mean **`135.167 tok/s`**. This is **`1.45%` above** the prior XQA/BF16-draft winner `133.232`, with mean E2E about `4.568 s`.
- This candidate is now high priority, but forced acceptance cannot establish proposal quality. Next: stop the fixed server, relaunch unsimulated, run thinking/tool semantic probes and two real-performance windows; retain only if FP8 draft KV does not materially damage acceptance or reasoning. Then requalify the long-context ladder, including exact `199016`.

### 2026-08-16 06:41 PDT — draft FP8 KV wins real rejection sampling too

- Fixed server PID `31352` was stopped intentionally and the same FP8-draft candidate relaunched unsimulated under parent PID **`31224`**. It reached ready with target capture `33.38 s`, draft capture `1.13 s`, and `1.56 GB` graph-end headroom.
- Thinking semantic probe passed with `67` reasoning tokens, explicit correct arithmetic, and final `703`. Tool probe passed with preserved reasoning and exactly one `multiply({"a":37,"b":19})` call, `finish_reason=tool_calls`.
- First unsimulated exact `6213/512` window: **`97.866`, `94.768`, `98.211`, `102.846`, `95.033 tok/s`**; mean **`97.745 tok/s`**, mean E2E `6.035496 s`.
- Second consecutive window: **`106.110`, `97.169`, `101.735`, `91.762`, `95.762 tok/s`**; mean **`98.508 tok/s`**, mean E2E `5.995323 s`.
- Combined ten-run real mean is **`98.126 tok/s`**, mean E2E **`6.015409 s`**, mean observed prompt throughput about `7777.608 tok/s`. This is **`4.32%` above** the prior BF16-draft XQA combined mean `94.062`, **`2.10%` above** its recovery-window mean `96.110`, and **`4.04%` above** the earlier Triton-draft production mean `94.319`.
- Proposal acceptance still varies normally (logged windows span roughly `1.62–2.85` accepted tokens and `0.21–0.62` rate), but FP8 draft KV did not lower realized throughput; it improved it. Candidate is promoted provisionally. Next: long-context capacity/prefill ladder and exact `199016`, then add launcher default and regression coverage after the remaining isolated controls.

### 2026-08-16 06:44 PDT — FP8 draft passes the real 200K context boundary

- Unsimulated long-context qualification passed without restart: exact `32768+16` twice at prompt `5994.673` and `6114.117 tok/s`; warmed `32768+512` at prompt/decode `6165.872/96.244 tok/s`; exact `65536+16` at `4740.959/100.952 tok/s`.
- Exact near-limit **`199000+16 = 199016`** passed at prompt **`2423.812 tok/s`**, decode `83.976 tok/s`, TTFT `82.102060 s`, E2E `82.280684 s`. This matches the selected BF16-draft/XQA near-limit capability and is close to its `2429.153 tok/s` prompt result while preserving the extra FP8-draft performance/headroom.
- `/model_info` confirms `has_image_understanding=False` and `has_audio_understanding=False`; vision remains disabled. Parent PID `31224` is the current fastest provisionally promoted unsimulated candidate and remains live for the next isolated control decision.

### 2026-08-16 06:49 PDT — current-topology two-step MTP control rejected

- Stopped the promoted three-step candidate only after its full checkpoint, then launched the same FP8-draft/XQA/FlashInfer-sampling configuration with only MTP geometry changed to two steps and three draft tokens. Candidate parent PID `36260`; target verify capture `48.17 s`, draft decode `1.18 s`, graph-end headroom `1.52 GB`.
- First real exact `6213/512` window: `90.844/97.374/98.087/96.690/96.842 tok/s`, mean **`95.967`**. Second window: `97.741/91.281/94.866/96.052/95.118`, mean **`95.012`**.
- Combined two-step mean **`95.490 tok/s`** is **`2.69%` below** the selected three-step FP8-draft mean `98.126`. All two-step outputs happened to share one digest, but realized throughput is the production selection criterion after semantic safety. Reject two-step and retain three steps/four draft tokens.

### 2026-08-16 06:52 PDT — CUTLASS DSL / FlashInfer GDN is unavailable on native Windows

- Rechecked the current dependency surface rather than relying on the old note. SGLang pins `nvidia-cutlass-dsl[cu13]==4.6.2` only when `sys_platform != 'win32'`; the active Windows venv has no `cutlass` module. FlashInfer `0.6.17` GDN decode is a genuine CuTe-DSL implementation and imports `cutlass`, `cutlass.cute`, and `cuda.bindings.driver` at module import.
- Official PyPI metadata for `nvidia-cutlass-dsl 4.6.2` declares POSIX/Linux, has only a tiny platform-independent metapackage, and requires `nvidia-cutlass-dsl-libs-base==4.6.2`. The base package publishes only manylinux x86-64/aarch64 wheels, with **no source distribution and no Windows wheel**. The current `4.7.0` base release has the same Linux-only artifact matrix.
- NVIDIA's current CUTLASS README explicitly states CUTLASS 4.x builds are down on Windows for all CUDA toolkits. With the proprietary compiled DSL base absent and no source archive to port, FlashInfer GDN/CuTe-DSL kernels cannot be enabled honestly in this native-Windows stack. Keep Triton GDN plus ReplaySSM; revisit only when NVIDIA publishes Windows DSL binaries/source or marks Windows fixed.
- Evidence: https://pypi.org/project/nvidia-cutlass-dsl/4.6.2/ ; https://pypi.org/project/nvidia-cutlass-dsl-libs-base/4.6.2/ ; https://pypi.org/project/nvidia-cutlass-dsl-libs-base/4.7.0/ ; https://github.com/NVIDIA/cutlass/blob/main/README.md

### 2026-08-16 06:54 PDT — top-k-2 tree speculation is incompatible with the required sampled profile

- Source audit established that TRT-LLM/XQA speculative metadata supports only `topk <= 1`, ReplaySSM spec verify is a linear-chain kernel and explicitly rejects `topk > 1`, and SGLang's rejection-sampling implementation requires `speculative_eagle_topk=1`.
- Gave top-k 2 one live compatible launch anyway by using FlashInfer target+draft attention, disabling ReplaySSM, and disabling rejection sampling. Parent PID `30544` reached ready, but lost substantial memory/headroom: pool end `1.42 GB`, target graph begin `0.84 GB`, final graph end only **`0.56 GB`**. It also loaded the recurrent/tree kernel set that the selected linear-chain topology avoids.
- The first real Qwen sampled-thinking request failed before producing output: scheduler raised `RuntimeError: Native-Windows EAGLE sampling requires --speculative-use-rejection-sampling`. Enabling that flag is rejected at argument validation because rejection sampling supports top-k 1 only. This leaves no sound top-k-2 route for the required temperature-1/top-k-20 reasoning workload; greedy-only numbers are irrelevant to production and were not accepted.
- Reject top-k 2. Restore the launcher after removing the two temporary test-only disable switches, then return to three-step/top-k-1 XQA + ReplaySSM.

## Post-compaction checkpoint — Sun Aug 16 07:02:23 PDT 2026

- Active goal remains exhaustive optimization of reasoning-enabled, vision-disabled, tool-enabled, preserved-thinking Qwen3.8-27B NVFP4 at a real 200,000-token context on native Windows/RTX 5090, ported to FlashInfer `0.6.17`. The current winner is three MTP steps/four draft tokens, top-k 1, target+draft XQA, FlashInfer prefill/sampling, ReplaySSM, and **FP8 E4M3 draft KV**.
- The narrow XQA dtype fix in `python/sglang/srt/layers/attention/trtllm_mha_backend.py` keeps XQA query/output in model dtype while FP8 remains confined to KV; non-XQA TRTLLM-Gen retains FP8-Q behavior. Pure-helper regression coverage in `test/registered/attention/test_trtllm_mha_graph_metadata.py` passed **3 tests**. Fixed work improved to **135.167 tok/s**; real unsimulated ten-run mean improved to **98.126 tok/s**, with valid thinking and tool calls.
- The winner passed `32768+512`, `65536+16`, and exact **`199000+16=199016`** at prompt/decode `2423.812/83.976 tok/s`; `/model_info` reports image and audio understanding false. Launcher default now uses `SpeculativeDraftKvCacheDtype='fp8_e4m3'` and keeps required rejection sampling plus ReplaySSM.
- Isolated controls are closed: two-step MTP produced **95.490 tok/s** and lost; top-k 2 has no correct sampled path with native-Windows rejection sampling/XQA/ReplaySSM and crashed its first temperature-1 request; FlashInfer GDN/CuTe DSL cannot run because NVIDIA publishes no Windows CUTLASS DSL base and documents CUTLASS 4.x Windows builds as down. Temporary top-k-2 launcher switches were removed.
- A first full-200K PyTorch GPU profile exhausted profiling headroom and ended in an asynchronous CUDA launch failure without flushing a trace. This was profiler overhead, after the same candidate had already passed exact `199016` and ordinary `6213/512` workloads.
- A reduced-pool **profiling-only** winner server is currently live under parent PID **`29540`**, with `-MaxTotalTokens 32768` and `SGLANG_TORCH_PROFILER_DIR=C:\Users\Daniel\sglang\benchmark\windows\profiles`. It passed warmup and a profiled `6213/512` request. The successful trace is `benchmark/windows/profiles/fp8_draft_32k_pool-1786888805.248822-TP-0.trace.json.gz` with `1,154,146` events (`440,841` kernels). This reduced-pool process is not the final production server.
- New intentional analyzer `scripts/windows/analyze_torch_trace.py` aggregates category time and exact kernel names. Immediate next action: compile/run it against the successful trace, normalize kernel families if needed, and use measured shares to close or test only credible remaining hotspots. Then stop PID `29540`, perform focused syntax/regression/diff/header-integrity checks, relaunch the full-200K default winner, and finish live thinking/tool/OpenCode2/vision-health acceptance before the hostile completion audit.

### 2026-08-16 07:08 PDT — GPU trace reveals one credible fused-FP8 dependency fix

- The reduced-pool GPU trace analyzed successfully. It records `5681.962 ms` of kernel time over the profiled request. Dominant families are native CUTLASS NVFP4 GEMMs **40.180%**, cuBLAS FP8 GEMMs **26.759%**, CUTLASS BF16 GEMMs **6.350%**, and BF16 GEMV **4.449%**. FlashInfer prefill/verify attention is only `3.059%`, ReplaySSM `0.746%`, GDN `0.379%`, and XQA `0.316%`; attention/recurrent tuning is no longer the material wall. Core quantized linear algebra is roughly three quarters of GPU time.
- The 26.8% FP8 family is the model's mixed-precision attention/linear projection path. Windows currently uses `torch._scaled_mm` into FP32 plus the already-fused Triton row/channel-scale conversion because the SGL CUTLASS channelwise FP8 op is unavailable. FlashInfer FP8 autotuning was already measured and rejected; it selected slower tactics.
- Revisited the earlier standalone SGL SM120 channelwise CUTLASS extension, which compiled but raised `cudaErrorMisalignedAddress`. NVIDIA CUTLASS issue `#2905` now documents the exact native-Windows RTX 5090 failure in the same pinned CUTLASS headers: embedded TMA descriptors in mainloop and epilogue `Params` lack required 64-byte alignment. The proposed `alignas(64)` fix removes the crash. Local clean clone `C:\Users\Daniel\cutlass-sglang` is still pinned at `57e3cfb47a2d9e0d46eb6335c3dc411498efa198` and contains both unaligned definitions; the old compiled test extension also remains cached.
- This is the sole new profile-backed high-value candidate. Next: patch only the local CUTLASS dependency clone, rebuild the isolated SM120 FP8 op, require numerical parity across real ModelOpt shapes, microbenchmark it against the selected cuBLAS+Triton fallback, and launch a controlled fixed-work server only if it wins. The protected CUDA header and its clean copy remain out of scope and untouched. Evidence: https://github.com/NVIDIA/cutlass/issues/2905

### 2026-08-16 07:49 PDT — aligned SGL CUTLASS FP8 now runs; default tile is not a clear win

- The first dependency rebuild still faulted because issue `#2905` names the block-scaled mainloop, while this channelwise kernel selects the ordinary SM120 mainloop in `sm120_mma_tma.hpp`. Applying the same required 64-byte alignment to that actual TMA `Params`/A/B pair, together with the epilogue pair, removed `cudaErrorMisalignedAddress`. The isolated extension now compiles, links, loads, and runs on the RTX 5090.
- Correctness against the selected Windows cuBLAS-FP32 + fused-Triton-scale fallback is excellent across real high-frequency ModelOpt shapes and M=1/4: five shapes were bit-identical in BF16, and `(M,K,N)=(4,6144,5120)` differed by only `4.3e-7` relative MAE. Every output was finite.
- Default fixed `128x128x128` SGL CUTLASS timings are mixed: `(1,5120,10240)` `0.062340 ms` vs fallback `0.061989`; `(4,5120,10240)` `0.061998` vs `0.062212`; `(1,5120,6144)` `0.062049` vs `0.025596` (large loss); `(4,5120,6144)` `0.059742` vs `0.062666` (+4.89%); `(1,6144,5120)` `0.069933` vs `0.071851` (+2.74%); `(4,6144,5120)` `0.069921` vs `0.069790`. This untuned fused kernel is not promotable as a blanket backend.
- The result confirms the dependency defect and unlocks valid SM120 experimentation, but the trace wall is now kernel geometry rather than scaling overhead. Next narrow audit: test valid low-M SM120 CUTLASS tile/schedule shapes plus the missing full-attention widths and M=2. Continue only if a shape-dispatch policy shows a material weighted microbenchmark win; otherwise reject the fused backend without disturbing production.

## Post-compaction checkpoint — Sun Aug 16 08:01:46 PDT 2026

- Active goal remains exhaustive performance optimization of reasoning-enabled, vision-disabled, tool-enabled, preserved-thinking Qwen3.8-27B NVFP4 at a real 200,000-token context on native Windows/RTX 5090, using the FlashInfer `0.6.17` port. The validated production winner remains three MTP steps/four draft tokens, top-k 1, target+draft XQA, FlashInfer prefill/sampling, ReplaySSM, and FP8 E4M3 draft KV.
- Winner evidence remains: fixed-work mean **135.167 tok/s**; real unsimulated ten-run mean **98.126 tok/s**; correct thinking answer `703`; exact tool call `multiply({"a":37,"b":19})`; exact **199000+16=199016** passed at prompt/decode `2423.812/83.976 tok/s`; `/model_info` reports image/audio understanding false. The XQA FP8-draft dtype regression tests passed `3 passed, 151 deselected`.
- The successful reduced-pool trace is `benchmark/windows/profiles/fp8_draft_32k_pool-1786888805.248822-TP-0.trace.json.gz`: `5681.962 ms`, `440841` kernels. Dominant measured families are CUTLASS NVFP4 GEMM **40.180%**, cuBLAS FP8 GEMM **26.759%**, CUTLASS BF16 GEMM **6.350%**, PyTorch elementwise **6.306%**, BF16 GEMV **4.449%**, FlashInfer prefill/verify attention **3.059%**, ReplaySSM **0.746%**, GDN **0.379%**, and XQA **0.316%**. Quantized GEMMs are the material remaining wall.
- Profiling-only reduced-pool server parent PID **29540** is still expected live with `-MaxTotalTokens 32768`; it must be stopped before final production relaunch. It is not the final server.
- The trace-backed SGL CUTLASS channelwise-FP8 experiment found and repaired the native-Windows SM120 TMA descriptor alignment defect in local dependency clone `C:\Users\Daniel\cutlass-sglang` (base commit `57e3cfb47a2d9e0d46eb6335c3dc411498efa198`). The actually selected mainloop is `include/cutlass/gemm/collective/sm120_mma_tma.hpp`; its `Params`, `TMA_A`, and `TMA_B`, plus epilogue `Params`, `TMA_C`, and `TMA_D`, require `alignas(64)`. Earlier exploratory changes in the block-scaled and SM90 mainloop headers may be unnecessary and need a narrow live-state review before cleanup. The protected CUDA header and its clean copy remain untouched.
- SGLang experiment artifacts are `python/sglang/kernels/jit/csrc/gemm/fp8_scaled_mm_sm120_windows.cu`, `scripts/windows/probe_sm120_fp8_cutlass.py`, and `scripts/windows/probe_sm120_fp8_cutlass.ps1`; `python/sglang/kernels/aot/csrc/gemm/math.hpp` has the MSVC-portable `next_pow_2` implementation. The extension cache is `torch_extensions\Cache\py313_cu130\sgl_kernel_windows_fp8_sm120_aligned`.
- Stable 200-iteration `128x128x128` measurements showed the aligned candidate broadly about 2x slower than the selected cuBLAS-plus-fused-Triton fallback, so it is rejected as a blanket backend. Numerical parity remained excellent: all outputs finite and maximum observed relative MAE about `8.3e-06` on the narrow `(1,5120,1024)` case.
- The source currently uses the CUTLASS-unit-tested SM120 tile `CTAShapeDefault = Shape<_128, _64, _64>` in `python/sglang/kernels/aot/csrc/gemm/fp8_gemm_kernel.cu`. Its latest 200-iteration pass is shape-selective: wide projections lose badly (`5120x10240` about `0.55x`; `5120x12288` about `0.26-0.42x`), while narrow `N=1024` wins at M=2/4 by about **36%**, `(M,K,N)=(4,6144,5120)` wins about **4.7%**, and M=1 narrow is a tie. This is not safe for blanket routing.
- Immediate next action is a cached matched rerun of `scripts\windows\probe_sm120_fp8_cutlass.ps1` to rule out compile/cold-state artifacts. Then calculate actual runtime shape frequencies and a weighted benefit for a very narrow dispatch (`N==1024 && M in {2,4}`, possibly `(4,6144,5120)`). Integrate only if the expected end-to-end gain is credible; otherwise restore the original `128x128x128` tile and remove/revert experimental artifacts narrowly with `apply_patch` after inspecting live state.
- Final work after this candidate decision: stop PID `29540`; run Python, PowerShell, targeted regression, `git diff --check`, and CUDA-header integrity checks; relaunch the full-200K unsimulated winner; revalidate health, thinking, tools, model-info/vision-off, OpenCode2, and leave that production server live; then perform the hostile requirement-by-requirement audit.
- The user is committing concurrently and explicitly warned that Git state may change underneath this work. Treat those changes as intentional user activity, inspect live files/status before every edit or cleanup, and never undo or overwrite their commit state.

### 2026-08-16 08:05 PDT — robust matched rerun rejects selective CUTLASS FP8 dispatch

- The first cached `128x64x64` rerun again showed the fallback timing swinging enough to reverse apparent wins, while CUTLASS remained stable. The probe was tightened to nine alternating-order matched blocks per shape, 128 iterations per block, with median paired speedup and min/max ranges; Python compilation passed.
- Robust medians eliminate the earlier apparent selective wins. Wide shapes remain decisive losses: `5120x10240` is only `0.235-0.241x` fallback speed and `5120x12288` only `0.181-0.252x`; `5120x6144` is `0.388-0.431x`; `6144x5120` is `0.455-0.492x`.
- The only plausible narrow family, `5120x1024`, is a statistical tie: paired median speedups are M1 **1.031**, M2 **1.024**, and M4 **0.970**, with every range crossing 1.0 (`0.942-1.262`, `0.901-1.512`, and `0.927-1.074`). Median absolute times are about `0.031-0.032 ms` for both paths. The prior 36-52% apparent gains were fallback timing transients, not a reproducible kernel advantage.
- Reject the SGL CUTLASS channelwise-FP8 kernel for production, including shape-selective dispatch. It offers no robust material win and would add JIT/dependency/dispatch complexity. Preserve the alignment-port finding in notes, restore the AOT tile to its original `128x128x128`, and clean only the rejected SGLang experiment artifacts/source changes after checking the user's live Git state. Keep the useful trace analyzer and production FP8-draft/XQA work.

### 2026-08-16 08:07 PDT — rejected experiment closed; focused validation clean

- The user committed the accumulated SGLang work as `abe8be35fc` (`WIP: preserve Windows Qwen 3.5 enablement work`) and explicitly authorized continued edits. The untracked `sglang.bundle` is user-owned and remains untouched.
- Restored production `fp8_gemm_kernel.cu` to its original `128x128x128` tile, leaving no diff in that runtime source. Kept the improved alternating/median probe for reproducibility. In `C:\Users\Daniel\cutlass-sglang`, removed the two exploratory alignments from the unused block-scaled and SM90 mainloops; retained only the actually required ordinary SM120 mainloop and shared epilogue `alignas(64)` fixes that make the native-Windows probe function.
- Stopped profiling-only server tree rooted at PowerShell PID `29540` by terminating its main Python PID `1860`; parent/launcher/main processes all exited and no SGLang compute process remains. The full GPU is available for final production startup.
- Validation passed: Python compilation for TRTLLM-MHA dtype routing, FlashInfer sampler, stream benchmark, trace analyzer, and robust CUTLASS probe; PowerShell parsing for the production launcher, OpenCode2 wrapper, CUTLASS probe, CUDA environment initializer, and NVFP4 smoke; `git diff --check` in SGLang, FlashInfer `0.6.17`, and CUTLASS.
- Both protected CUDA headers still match SHA-256 `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.
- Focused XQA/FP8 regression is exactly **3 passed, 151 deselected**. The broader SGLang focused suite covering XQA metadata, attention-backend setup, Eagle preparation/top-k1, HTTP warmup, and tokenizer RID cleanup is now **191 passed plus 7 subtests**. Whole FlashInfer `tests/jit/test_jit_cpp_ext.py` is **28 passed**.
- Next: verify installed FlashInfer source identity, launch untouched full-200K unsimulated defaults, inspect resolved backends/no simulation/readiness, and run final health, preserved-thinking, tool, model-info, OpenCode2, and representative performance acceptance. Leave the winning server live.

### 2026-08-16 08:15 PDT — final server is correct; NVML polling exposes a clock-residency opportunity

- Installed FlashInfer is `0.6.17` and matches the clean source: `jit/core.py` hash `4D760AA...DACC`; XQA standard header `2215B3...0106`; TRTLLM-Gen FMHA header `1A9757...8F50`; cubin loader `650E60...11F0`. Every source/install pair is identical.
- Untouched full-200K unsimulated defaults are live under hidden PowerShell PID **`37220`**, logs `sglang-qwen-fi0617-fp8-final.*.log`. Resolved target split is FlashInfer prefill plus TRTLLM-MHA/XQA decode; draft is TRTLLM-MHA/XQA with FlashInfer initial prefill; FP8 E4M3 draft KV, FlashInfer sampling, top-k1 rejection sampling, three steps/four draft tokens, ReplaySSM, and exact `200000` context/pool are present. Target graph capture ended in `39.49 s`, draft capture in `1.34 s`, final internal headroom `1.40 GB`; server ready at `08:09:26` and health is HTTP 200.
- `/model_info` confirms image/audio understanding false. Preserved-thinking temperature-1/top-k20/top-p0.95 semantic probe stopped normally with `138` reasoning tokens and correct `703`. Tool probe stopped with `finish_reason=tool_calls`, preserved `66` reasoning tokens, and exactly one `multiply` call with integer arguments `37` and `19`. The first display attempt only hit the local CP1252 console on Unicode `−`; rerunning the same probe with Python UTF-8 printed it correctly and the server remained healthy.
- Initial 6213/512 production checks unexpectedly alternated between roughly `49-56 tok/s` and the expected high band. Server logs showed this was not merely proposal acceptance: accepted-token throughput per speculative iteration was about half the earlier validated server on slow runs.
- A live `nvidia-smi dmon` control revealed the cause/opportunity. Without the NVML poller, repeated short CUDA-graph bursts sometimes ran around half-rate. With one-second NVML polling active, two matched unsimulated runs recovered to **90.597** and **87.081 tok/s**; logs recovered from about `21-27` to `45-48` speculative iterations/s at comparable acceptance. During those runs GPU clocks held around `2.96-3.01 GHz`, memory at `13.801 GHz`, SM utilization `86-100%`, power `276-385 W`, and no power/thermal violation was reported. Stopping the poller reproduced the slow band.
- This is now the last credible system-level performance surface: test a reversible NVIDIA CUDA-clock override/locked-clock route if supported, or a lightweight hidden NVML keepalive if consumer-WDDM rejects clock control. Require a repeated matched window and leave only the fastest safe state live. PID `37220` remains healthy during this audit.

### 2026-08-16 08:22 PDT — clock control unavailable; background WDDM traffic explains final-window variance

- NVIDIA's supported CUDA-clock override (`nvidia-smi -cc 1`) reports unsupported on this GeForce. A reversible `3000-3090 MHz` GPU clock lock is exposed but the current user lacks permission; no clock state changed.
- Tested a silent Python NVML keepalive at 1.0 s and 0.1 s intervals and hidden `nvidia-smi dmon`. Each initially coincided with restored runs as high as **101.027/99.262/97.290 tok/s**, but longer sequences still fell to `46-65 tok/s`; the polling mechanism is therefore not a reliable production optimization by itself and remains unintegrated.
- Continuous device/process monitoring found the actual changing condition: while SGLang was idle, `ZCode.exe` PID `29576` sustained roughly **2-6% SM / 1-3% memory** and repeatedly moved about **1.1 GB/s over PCIe**, with observed spikes near `5.9 GB/s`. `EpicGamesLauncher.exe` PID `31256` also appeared at `08:18:44`. Under WDDM, this graphics/transfer traffic preempts the very short speculative CUDA graphs and cleanly explains why identical output digests and comparable acceptance sometimes ran at half the prior iteration rate. No monitor/keepalive helper is currently left running.
- Asked the user to minimize/close ZCode and Epic for one uncontaminated final window; neither process was terminated or altered. The server remains live and correct.
- Standalone OpenCode2 acceptance passed with the production overlay, thinking visible, main model `llama-cpp/qwen3.8-27b`, snapshots disabled, and exact output `XQA READY`; exit code `0`. The run took `26.64 s` under the same active GPU contention, so it is correctness evidence rather than a clean latency measurement.
- Final acceptance is complete except for the clean throughput window and last hostile audit. Keep parent PID `37220` live throughout.

### 2026-08-16 09:12 PDT — uncontaminated final production window reproduces the winner

- After Chrome/Epic activity stopped and ZCode went GPU-idle, `nvidia-smi pmon` showed no competing process with measurable SM/memory/video activity. No NVML poller, clock override, or keepalive helper was running.
- Five consecutive unsimulated exact `6213/512` production runs measured **`90.967`, `98.274`, `98.778`, `101.994`, `101.018 tok/s`**; mean **`98.206 tok/s`**. This reproduces the prior independent ten-run winner mean **`98.126 tok/s`** to within `0.08%`.
- Mean E2E was **`6.124864 s`**, mean TTFT `0.913063 s`, and mean observed prompt throughput `6875.009 tok/s`. Excluding the first post-idle/cold request, the warm four averaged **`100.016 tok/s`** and `5.973826 s` E2E.
- Four runs shared output digest `de8f4d9a...c4bd9`; the fifth followed a nearby stochastic path with digest `245a7c36...af78`. Every request returned exactly 512 completion tokens with `finish_reason=length`. The variance is expected from unseeded speculative proposal/rejection sampling, and throughput remained in the selected production band.
- The earlier low-band results were external WDDM contention, not a regression in the committed runtime or the final server. Reject the unintegrated NVML keepalive experiment; its untracked script was removed and no helper process remains.
- Production parent PID **`37220`** remains live, healthy, unsimulated, vision/audio disabled, thinking preserved, tools enabled, and OpenCode2-compatible. Proceed to the final hostile requirement audit and leave this server running.

## Final hostile completion audit — Sun Aug 16 09:13:48 PDT 2026

- **Required model/runtime:** live endpoint loads `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk` as ModelOpt mixed-precision/NVFP4 Qwen3.8-27B on native Windows RTX 5090. FlashInfer reports `0.6.17`; installed Python and critical XQA/TRTLLM/cubin-loader files match the clean port source byte-for-byte.
- **Reasoning enabled and preserved:** launcher selects `--reasoning-parser qwen3`; direct production requests send Qwen's recommended temperature `1.0`, top-p `0.95`, top-k `20`, presence penalty `1.5`, and both `enable_thinking`/`preserve_thinking=true`. Final probe returned coherent preserved `reasoning_content`, normal stop, and correct `703`.
- **Tools enabled:** launcher selects `--tool-call-parser qwen3_coder`; final temperature-1 thinking probe emitted exactly one parsed `multiply` call with integer arguments `37` and `19`, `finish_reason=tool_calls`. Standalone OpenCode2 passed with thinking visible and exact `XQA READY`.
- **Vision disabled:** launch uses `--language-model-only`, avoids the vision encoder, and final `/model_info` reports `has_image_understanding=False`, `has_audio_understanding=False`, and embedding multimodal support false.
- **Real 200K context:** live resolved args contain `context_length=200000` and `max_total_tokens=200000`; the selected configuration previously passed exact `199000+16=199016` at prompt/decode `2423.812/83.976 tok/s`. No simulation variable is present; live acceptance lengths/rates vary naturally.
- **Selected topology:** three MTP steps/four draft tokens, top-k1 distribution-correct rejection sampling, target and draft TRTLLM-MHA/XQA decode, FlashInfer target/draft initial prefill and sampling, ReplaySSM, Triton GDN, FP8 E4M3 draft KV, checkpoint-selected target KV, FlashInfer CUTLASS FP4 GEMM, page 64, 128 MiB workspace, 4096 prefill chunks, scheduler/stream interval 4, batch-one CUDA graphs, and partial torch compilation.
- **Final performance:** fixed-work mean **135.167 tok/s**. Independent real unsimulated windows are **98.126 tok/s over ten runs** and **98.206 tok/s over the final five clean runs**; final warm-four mean is **100.016 tok/s**. The two independent real means agree within `0.08%`. Exact near-limit prompt performance and correct semantics both remain qualified.
- **Exhaustive candidate closure:** measured/rejected alternate GGUF/Unsloth/RadixArk paths as applicable, non-MTP and MTP geometry controls, one/two/three-step speculation, top-k2 tree mode, Triton/FlashInfer/TRTLLM attention splits, page/KV/cache/workspace/chunk/polling/stream axes, target native-NVFP4 KV, draft BF16 versus FP8 KV, PyTorch versus FlashInfer sampling, FlashInfer FP4 CUTLASS/cuDNN/CuTe DSL, FP8 autotuning, torch-compile modes/repair experiments, ReplaySSM/GDN backends, and SGL CUTLASS channelwise-FP8 tiles. Trace evidence closes the residual wall: NVFP4 GEMM `40.180%`, FP8 GEMM `26.759%`, while attention/GDN/ReplaySSM/XQA together are small. CUTLASS's Windows TMA alignment defect was repaired and proven numerically, then its kernel was rejected after robust paired medians showed no material win.
- **Validation:** Python compilation passed for production routing/sampler/bench/analyzer/probe; PowerShell parsing passed for launcher/OpenCode/environment/smoke/probe scripts; focused XQA FP8 tests **3 passed, 151 deselected**; broader SGLang suite **191 passed plus 7 subtests**; FlashInfer JIT suite **28 passed**; `git diff --check` passes in SGLang, FlashInfer `0.6.17`, and CUTLASS.
- **Protected dependency boundary:** original and clean-port CUDA headers are untouched and still identical at SHA-256 `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`. CUTLASS retains only the two proven alignment edits in the actual ordinary SM120 mainloop and shared epilogue; exploratory unused-header edits were removed.
- **Workspace/live state:** user checkpoint commit is `abe8be35fc`. Post-commit SGLang changes are only this updated `NOTES.md` and the more rigorous paired-median CUTLASS probe; untracked `sglang.bundle` is user-owned and untouched. No profiler, dmon, clock override, or keepalive helper remains. Final server tree is PowerShell PID **`37220`** -> `sglang.exe` PID `32352` -> Python PID **`35692`**; health remains HTTP 200 and this fastest validated server is intentionally left live.
- All requested requirements and trace-backed remaining opportunities are closed with measured evidence. Future material gains require a faster vendor NVFP4/FP8 GEMM implementation, a Windows CUTLASS DSL release, or removal of unrelated WDDM GPU contention rather than another SGLang flag change.

### 2026-08-16 09:19 PDT — orphan cleanup verification

- User authorized termination of ZCode orphans. Enumerated the complete Electron tree rooted at PID `31264`, including GPU PID `29576`, crashpad, renderer, utility, three app-server processes, plugin host, and audio service; terminated every `ZCode.exe` process and verified none remained. SGLang health stayed HTTP 200.
- User observed an ephemeral `pcctl` PID repeatedly appearing and then killed it. Immediate and delayed process-table checks found no process name or non-shell command line containing `pcctl`; no matching service or scheduled task was present in the prior checks.
- Collaboration registry contains only `/root`; there are no child/sub-agents registered for this session. The only `codex.exe` process is this root session, PID `35908`. Eight-second delayed verification showed no `pcctl` respawn and no ZCode process.

### 2026-08-16 09:21:46 PDT — post-compaction checkpoint; dispatch audit reopened

- Final selected production result remains the live RadixArk Qwen3.8-27B NVFP4 server at exact `200000` context: vision/audio disabled, thinking preserved, tools enabled, three MTP steps/four draft tokens, top-k1 rejection sampling, target/draft TRTLLM-MHA/XQA decode, FlashInfer prefill/sampling, ReplaySSM, Triton GDN, FP8 draft KV, page64, 128 MiB workspace, and batch-one CUDA graphs. Qualified performance is fixed-work **135.167 tok/s**, real ten-run mean **98.126 tok/s**, and clean five-run mean **98.206 tok/s** (warm four **100.016 tok/s**); exact `199000+16` passed at `2423.812/83.976 tok/s`.
- The selected server is intentionally still live as PowerShell PID `37220` -> `sglang.exe` PID `32352` -> Python PID `35692` on port `30000`; its last verified health was HTTP 200. Preserve it until an isolated candidate is ready for measurement.
- A new GPT Pro review reopens the performance audit around dispatch-level opportunities: (1) draft extension may miss CUDA-graph capture, (2) target verification may be routed through FlashInfer prefill instead of XQA, (3) repeated BF16 MTP work may be avoidable, and (4) fixed top-k1/chain decoding may still pay general tree-construction/bookkeeping overhead. Secondary leads are fused Qwen MLP and reduced-vocabulary sampling.
- These are hypotheses pending live source and trace attribution. Immediate order: map actual target-verify and draft-extend dispatch; quantify per-iteration CPU/GPU cost and BF16 parents in the successful trace; inspect the existing top-k1 fast path for residual general tree work; then implement and benchmark one behavior-locked candidate at a time against the qualified baselines.
- Recent cleanup remains stable: all ZCode processes were terminated, the user killed the transient `pcctl` process they observed, delayed checks found no respawn, and the collaboration registry contained only `/root`. No profiler, keepalive, clock override, or monitoring helper was left running.

### 2026-08-16 09:23:57 PDT — two dispatch gaps confirmed in the selected run

- The live startup log captured target verify (`39.49 s`) and draft decode (`1.34 s`) CUDA graphs, but contains **no draft-extend capture**. This is a real per-iteration eager gap, not absent infrastructure: `_draft_extend_for_decode` tests and replays `cuda_graph_runner_for_draft_extend` every speculative loop, while the runner was never constructed.
- Root cause is the selected split attention topology. The draft runner wraps FlashInfer prefill plus TRTLLM-MHA decode in `HybridAttnBackend`; `DRAFT_EXTEND_V2` correctly selects its TRTLLM decode child, but `_capture_cuda_graphs` permits concrete Triton/TRTLLM/FlashInfer backend types and rejects the `HybridAttnBackend` wrapper. Its graph delegation methods already exist, so the support gate—not the model—is suppressing capture.
- Target verification is also confirmed to use the FlashInfer prefill child: live args resolve `speculative_attention_mode='prefill'`, and `HybridAttnBackend._select_backend(TARGET_VERIFY)` follows that mode. TRTLLM-MHA already implements uniform target-verify metadata and the SM120 XQA `q_len_per_req` path, so `speculative_attention_mode=decode` is a valid isolated candidate to benchmark.
- Next: prove Hybrid draft-extend capture with a focused type/behavior test, add a launcher-selectable verify mode, then profile/benchmark the combined dispatch correction. Preserve the current server until the code and tests are ready.

### 2026-08-16 09:30:29 PDT — draft-extend graph dispatch fix implemented and unit-proven

- Implemented `_resolve_draft_extend_graph_backend`: for the Qwen hybrid stack it unwraps `HybridLinearAttnBackend`, follows `HybridAttnBackend`'s `DRAFT_EXTEND_V2` routing, and hands the concrete TRTLLM-MHA child directly to `EAGLEDraftExtendCudaGraphRunner`. The eager wrapper remains unchanged; graph capture avoids both the incorrect wrapper-type rejection and unused GDN graph-state setup.
- Added focused unit coverage for nested hybrid resolution and concrete-backend identity. The full focused file passes: **11 passed, 7 subtests passed**. Python compilation and `git diff --check` pass.
- Added launcher-selectable `SpeculativeAttentionMode` (`prefill`/`decode`) and optional draft quantization (`nvfp4_online`/`unquant`) so target-verify XQA and online-quantized MTP can be isolated without ad-hoc command changes. Defaults remain behavior-preserving until measurement selects a winner.
- The GPU-only successful trace contains exactly **245** general tree-build kernel calls totaling only **0.555 ms** (`2.266 us` each); kernel execution itself is negligible, though host allocation/dispatch remains unmeasured. It also contains 245 verify-tree kernels totaling `0.491 ms`.
- The same trace gives a strong MTP quantization signal: BF16 CUTLASS GEMM contributes **360.829 ms / 13,091 calls / 6.350%**, and BF16 GEMV contributes **252.816 ms / 2,450 calls / 4.449%**. The checkpoint explicitly ignores every `mtp*` weight, while `nvfp4_online` supports online conversion of BF16 source weights; this is a concrete candidate, subject to acceptance-quality and throughput measurement.

### 2026-08-16 09:34:47 PDT — draft-extend graph live proof; unsimulated window requires fixed-work attribution

- Replaced the prior production server intentionally and launched the graph-only candidate at full exact 200K under parent PID `20364`, keeping `speculative_attention_mode=prefill`. Startup now proves all three graphs: target verify `52.04 s`, draft decode `1.57 s`, and the previously missing **draft extend `1.01 s`**. Draft-extend capture cost `0.08 GB`; final graph headroom is `1.77 GB`; health is HTTP 200.
- First five real `6213/512` results were generation **`82.920, 80.387, 84.933, 82.319, 77.576 tok/s`**, mean **`81.627 tok/s`**. Prompt mean is `4777.549 tok/s`; excluding the first cold request it is `5124.135 tok/s`. Mean E2E is `7.602519 s`. Four outputs retained the production digest; the fifth followed another valid stochastic proposal path.
- This window is below the qualified production mean `98.206 tok/s`, but acceptance varied heavily (`~1.8-2.6`) and the prior production itself exhibited WDDM/power-state bands. The graph candidate is **not promoted or rejected from this stochastic window**. Next required discriminator is the exact forced-acceptance-3 fixed workload versus `135.167 tok/s`; only then decide whether replay overhead outweighs captured eager launch savings.
- GPU process inspection found only the SGLang compute process plus normal desktop C+G clients (Chrome, terminal, shell, Snipping Tool, WebView); ZCode remains absent. No separate compute contender appeared.

### 2026-08-16 09:37:21 PDT — forced-work control proves draft-extend graph is a large win

- Restarted the same full-200K graph-only topology with `SGLANG_SIMULATE_ACC_LEN=3`; startup again captured draft extend successfully (`0.96 s`, `0.08 GB`) after target verify and draft decode.
- Five exact `6213/512` forced-acceptance runs preserved the established digest and measured **`147.928, 147.799, 147.607, 148.639, 137.731 tok/s`**; mean **`145.941 tok/s`**. The tightly grouped first four average **`147.993 tok/s`**; the fifth is another short WDDM/power-state dip.
- Against the prior qualified fixed-work mean **`135.167 tok/s`**, the conservative five-run result is **+7.97%** generation throughput. The first-four steady band is `+9.49%`. Draft-extend graph capture is therefore retained; its prior low real window reflected proposal-acceptance/runtime variance rather than graph cost.
- Current fixed server parent PID is `32252` and is simulation-only. Next isolate target verification on the TRTLLM-MHA/XQA decode child (`speculative_attention_mode=decode`) while retaining this graph fix, then compare the same forced workload.

### 2026-08-16 09:40:00 PDT — target-verify XQA startup gap repaired safely

- First `speculative_attention_mode=decode` launch reached target graph warmup and failed loudly in FlashInfer XQA: `AssertionError: Mask is required for speculative decoding`. This confirms dispatch reached XQA and identified an incomplete backend path rather than a model incompatibility.
- Root cause: TRTLLM-MHA already created an explicit packed causal mask for multi-token draft extension, but its target-verify metadata omitted one and the forward call passed `None` for every target verify. XQA requires the packed in-window mask whenever query length exceeds one.
- Added the same static packed causal mask to target-verify graph state **only when `speculative_eagle_topk == 1`**. The selected four-token chain gets rows `[1, 3, 7, 15]`; wider tree modes retain `None` and fail loudly instead of receiving incorrect chain ancestry.
- Focused XQA/draft-extend coverage passes **5 passed, 151 deselected**; Python compilation and `git diff --check` pass. Relaunch the same XQA fixed-work candidate next.

### 2026-08-16 09:43:27 PDT — post-compaction checkpoint; XQA eager warmup repair pending validation

- The prior qualified production reference remains exact `200000` context with **6875.009 prompt tok/s**, **98.206 tok/s** real five-run mean, and **100.016 tok/s** warm-four generation. Exact `199000+16` remains **2423.812 prompt / 83.976 generation tok/s**. The literal untouched GGUF reference was `32768` usable context, `446.583` prompt tok/s, and `38.673` generation tok/s; untouched native-Windows NVFP4 did not serve before this port's enablement work.
- Draft-extension CUDA-graph capture is now a proven retained win. At fixed accepted length 3, the full-200K candidate reached **145.941 tok/s** over five runs and **147.993 tok/s** over the steady first four, versus the earlier qualified fixed-work mean of **135.167 tok/s** (`+7.97%` conservatively, `+9.49%` in the steady band).
- Target verification successfully dispatched into TRTLLM-MHA/XQA with `speculative_attention_mode=decode`, then exposed a missing explicit speculative causal mask. Graph metadata now carries the safe top-k1 four-token chain mask. A second launch showed that eager warmup reaches the same XQA path before graph state exists, so the identical top-k1 mask has now also been added to eager target-verify metadata.
- The newest eager-mask edit and its focused unit test are present but **not yet validated after compaction**. The most recent XQA candidate parent (`35044`) exited on the warmup assertion; assume no live server until process and port checks prove otherwise. No subagents are active.
- Immediate next action: run the focused TRTLLM-MHA XQA tests, Python compilation, and diff hygiene; relaunch the full-200K fixed-work XQA candidate; require all three CUDA-graph captures and HTTP readiness; then benchmark the same five `6213/512` requests against **145.941 / 147.993 tok/s**.

### 2026-08-16 09:44 PDT — eager target-XQA mask repair unit-proven

- The eager target-verify XQA warmup repair passes its focused coverage: **6 passed, 151 deselected**. This includes both static CUDA-graph metadata and eager warmup metadata for the top-k1 four-token causal chain, plus the wider-tree safety case.
- `py_compile` for `trtllm_mha_backend.py` and `git diff --check` both pass. The next step is a clean full-200K fixed-work XQA launch and live graph/readiness proof.

### 2026-08-16 09:47 PDT — target-verification XQA produces another large fixed-work win

- Full-200K fixed candidate launched under parent PID `18944` with `speculative_attention_mode=decode` and forced accepted length 3. Target verify XQA now survives eager warmup and capture. All three graphs completed: target verify `39.85 s`, draft decode `1.27 s`, and draft extend `0.94 s`; final graph headroom is **1.85 GB** and `/health` returns HTTP 200.
- Five matched exact `6213/512` runs preserved digest `9d850fbf...af9c` and measured generation **`152.127, 158.793, 158.459, 160.640, 154.819 tok/s`**. Five-run mean is **`156.968 tok/s`**; warm-four mean is **`158.178 tok/s`**; observed peak is **`160.640 tok/s`**.
- Mean prompt throughput was `7405.744 tok/s` including the cold first request and `7838.476 tok/s` over the warm four. Mean E2E was `4.109840 s` (`4.023834 s` warm four).
- Versus graph-only target-prefill fixed mean `145.941`, target XQA adds **7.56%**. Comparing steady bands, `158.178` versus `147.993` is **+6.88%**. Versus the earlier production fixed reference `135.167`, the combined draft-extend graph plus target-XQA dispatch work is **+16.13%**.
- This establishes a new fixed-work ceiling but does not yet qualify stochastic proposal acceptance or reasoning/tool behavior. Next: stop the simulation server, relaunch the identical XQA topology unsimulated, run real `6213/512` windows plus semantic/tool checks, and retain XQA only if real throughput and behavior hold.

### 2026-08-16 09:50 PDT — target-verification XQA wins the first real rejection-sampling window

- Stopped the simulation process tree and relaunched the identical full-200K topology without `SGLANG_SIMULATE_ACC_LEN`, under parent PID `13560`. Startup again completed all graphs: target verify `38.15 s`, draft decode `1.47 s`, draft extend `1.02 s`; graph-end headroom is `1.81 GB` and the server is ready on port 30000.
- Five real exact `6213/512` runs measured generation **`106.747, 112.461, 111.869, 114.629, 111.769 tok/s`**. Mean is **`111.495 tok/s`**; warm-four mean is **`112.682 tok/s`**. All five produced the same 512-token digest `951615d5...14ec2` with `finish_reason=length`.
- Versus the prior final real five-run mean `98.206 tok/s`, this first target-XQA window is **+13.53%**. Versus the independent ten-run prior mean `98.126`, it is **+13.62%**. Mean prompt throughput is about `7241.467 tok/s` including the first cold request and `7805.787 tok/s` over the warm four.
- XQA is now a real-work winner pending semantic/tool checks and an independent recovery window. Keep the current unsimulated server live while validating those gates.

### 2026-08-16 09:51 PDT — target-XQA real win reproduces and semantic gates pass

- Independent second exact `6213/512` window measured **`105.938, 111.477, 109.324, 110.200, 113.089 tok/s`**, mean **`110.006 tok/s`** and warm-four mean **`111.023 tok/s`**. All five again returned the identical `951615d5...14ec2` digest.
- Combined target-XQA real ten-run mean is **`110.750 tok/s`**. This is **+12.87%** versus the prior independent ten-run production mean `98.126`, and **+12.77%** versus the prior final clean mean `98.206`. The two new five-run windows agree within `1.35%`, so the improvement reproduces beyond one favorable acceptance window.
- Recommended Qwen thinking profile passed: temperature `1.0`, top-p `0.95`, top-k `20`, presence penalty `1.5`, preserved coherent `reasoning_content`, normal stop, and correct final answer **703**. Tool parsing passed with preserved reasoning and exactly one `multiply({"a":37,"b":19})` call, `finish_reason=tool_calls`.
- `/model_info` still reports `has_image_understanding=false`, `has_audio_understanding=false`, and multimodal embeddings disabled. The unsimulated full-200K XQA winner remains live under parent PID `13560`.
- Target verification XQA is promoted provisionally. Next gates are exact near-limit `199000+16`, a launcher-default update to `SpeculativeAttentionMode='decode'`, focused regression tests, and then the remaining isolated MTP `nvfp4_online` candidate.

### 2026-08-16 09:53 PDT — target-XQA passes the exact 199016-token boundary

- The live unsimulated target-XQA winner completed exact `199000+16=199016` at **`2570.356 prompt tok/s`**, **`80.593 generation tok/s`**, and `77.607312 s` E2E with `finish_reason=length`.
- Capacity is therefore preserved at the required exact 200K pool. Prompt throughput is `6.05%` above the prior qualified near-limit sample (`2423.812 tok/s`). The 16-token generation sample is `4.03%` below the old single sample (`83.976 tok/s`), too short and acceptance-sensitive to override the replicated `+12.87%` normal-work win; retain this as a long-context routing lead rather than claiming a near-limit generation improvement.
- Next: make target-XQA the launcher default with focused validation, then isolate online NVFP4 MTP quantization. A future matched long-context prefill-vs-XQA control may justify length-adaptive verify routing if the near-limit generation delta reproduces.

### 2026-08-16 09:54 PDT — target-XQA promoted to the measured launcher default

- Changed the Windows Qwen launcher default `SpeculativeAttentionMode` from `prefill` to **`decode`**, selecting the newly qualified TRTLLM-MHA/XQA target-verification path while retaining FlashInfer for ordinary prefill.
- Focused validation passes after promotion: TRTLLM-MHA graph/eager XQA coverage **6 passed, 151 deselected**; EAGLE draft-extend graph/top-k1 coverage **11 passed, 7 subtests passed**; both modified Python modules compile; `git diff --check` passes.
- The current live server already uses the promoted value explicitly and remains the fastest qualified unsimulated candidate. Next isolate online NVFP4 quantization of the BF16 MTP draft model under fixed work.

### 2026-08-16 09:58 PDT — first MTP NVFP4 trial exposed two Windows/config routing gaps

- The first `-SpeculativeDraftModelQuantization nvfp4_online` launch did **not** quantize the draft. Logs still showed `quant=modelopt_mixed`, the ModelOpt loader, and about `6.17 GB` draft load memory. The CLI value reached `ServerArgs`, then checkpoint auto-detection replaced it.
- Root causes were concrete: Windows omitted `NvFp4OnlineConfig` from `QUANTIZATION_METHODS`, and draft quantization preservation only recognized `modelopt_fp4` while this checkpoint parses as `modelopt_mixed`. The checkpoint explicitly excludes `mtp*` and `mtp.layers.0*`, proving its embedded MTP weights are BF16.
- Added the importable NVFP4-online config to the Windows registry and preserved an **explicit** draft `nvfp4_online` request only when a ModelOpt FP4/mixed checkpoint explicitly excludes MTP weights. Packed MTP checkpoints retain detected serialization and cannot be accidentally requantized.
- Lightweight routing coverage passes **3 passed, 3 deselected**; modified modules compile and `git diff --check` passes. Direct Windows import of `NvFp4OnlineConfig` also succeeds.
- Important limitation before interpreting the live retry: current `nvfp4_online` converts MoE experts and leaves ordinary dense linears unquantized; this 27B checkpoint is dense. The retry will establish whether any runtime portion changes, but a true dense-MTP FP4 path would require a separate load-time dense-weight converter and per-token activation method.

### 2026-08-16 10:02 PDT — NVFP4-online routing now reaches the draft; Windows MoE import repaired

- The second live retry proved the routing fix: startup logged `Using CLI-specified quantization (nvfp4_online) ... modelopt_mixed` before constructing the draft.
- It then exposed a Windows-only optional dependency leak: `NvFp4OnlineConfig.get_quant_method()` imported the Triton `FusedMoE` package unconditionally even while inspecting a vocabulary embedding, which pulled absent `sgl_kernel` and aborted startup.
- Reused the existing Windows-safe `_fused_moe_type()` resolver so dense/non-MoE layer inspection no longer imports the Triton MoE stack. A direct Windows config/layer probe now returns normally, focused routing tests remain **3 passed**, compilation and diff hygiene pass.
- Relaunch the same fixed candidate once more. Expect functional parity and unchanged BF16 memory unless another layer type opts into conversion; measure rather than assume.

### 2026-08-16 10:06 PDT — stock NVFP4-online is rejected for this dense MTP; audit priorities refreshed

- The repaired `nvfp4_online` candidate served, but draft load memory remained about **6.29 GB** and no online-quantization message appeared. Source confirms why: stock `NvFp4OnlineConfig` converts `FusedMoE` expert weights and returns ordinary unquantized methods for dense linears; Qwen3.8-27B and its MTP layer are dense.
- Ten forced-work samples were extremely unstable (`72.044` to `145.502 tok/s`, mean `112.300`) and even the peak remained below the qualified unquantized-MTP target-XQA mean `156.968`. A monitored run reached full GPU clocks (`3052 MHz`, about `466 W`, `97%` utilization) yet measured only `142.270 tok/s`. This candidate is rejected; it neither reduced BF16 storage nor improved throughput.
- GPT Pro's final audit arrived after the graph/XQA work. Correct its stale status and baselines: draft-extend graph and target-verify XQA are implemented, correctness-qualified, and jointly lift fixed work from `135.167` to `156.968 tok/s`; real ten-run production rises from `98.126` to `110.750 tok/s`; exact `199016` passes at `2570.356/80.593 tok/s`.
- The audit's strongest remaining open lead is now proposal-distribution alignment: verify whether draft `q` omits target top-k/presence/token transformations, instrument acceptance overlap, and make any correctness implementation preserve the exact post-transform `q` used by rejection sampling. Sampling-backend reachability, adaptive depth, shape-specific GEMM dispatch, and long-context routing follow.

### 2026-08-16 10:10:43 PDT — post-compaction checkpoint; proposal-distribution alignment is next

- Current qualified winner: full-200K target verification through XQA plus captured `DRAFT_EXTEND_V2`. Fixed-work exact `6213/512` is **156.968 tok/s mean**, **158.178 tok/s warm-four**, and **160.640 tok/s peak**. Real sampled decode is **110.750 tok/s** across ten runs, up **12.87%** from the prior `98.126 tok/s` mean. Exact `199000+16` remains valid at **2570.356 prompt / 80.593 generation tok/s**.
- The stock `nvfp4_online` dense-MTP experiment is rejected. This checkpoint is dense, while the shipped path quantizes MoE experts; draft memory did not shrink and the forced-work mean fell to `112.300 tok/s`.
- GPT Pro's audit needs its status and baseline corrected: draft-extension graph capture and target-verification XQA are already implemented and qualified, and the suggested `0–3%` XQA ceiling was materially low. Its proposal-distribution mismatch remains the strongest open finding and moves to priority one.
- Source inspection confirms draft `q` currently receives temperature only, whereas target `p` also receives additive presence/frequency penalties, logit bias or masks when present, top-k, and top-p. Any repair must sample from and retain the exact transformed `q` used by rejection sampling, including inside captured draft-decode graphs.
- Commit `edc03bc22f` contains the work through the user's latest commit; further edits are explicitly allowed. The currently running server is the **simulation-only rejected NVFP4-online candidate**, not the production winner. Next source action is a behavior-preserving, opt-in sparse draft-`q` alignment path with graph-static sampling buffers, followed by fixed-work overhead and real sampled acceptance/throughput tests.

### 2026-08-16 10:17:24 PDT — sparse aligned draft-q candidate is unit-proven

- Added opt-in `--speculative-draft-sampling-top-k` (launcher value `0` keeps the qualified behavior byte-for-byte). The candidate samples directly from a fixed sparse support after applying additive presence/frequency/min-token penalties, logit bias, temperature, and request top-p, then scatters that exact post-transform `q` into the dense tensor required by the current rejection kernel.
- The top-p implementation matches the target fallback's threshold/tie semantics on the selected support. The sampled token probability is gathered from the same sparse `q`; rejection therefore receives the exact distribution used to draw the proposal.
- Draft-decode CUDA graphs now own stable top-p and combined additive-penalty/logit-bias buffers and refresh them on every replay. Prefill seeding and graph/eager draft extension use the same proposal function. Default temperature-only behavior remains unchanged when the option is absent.
- Focused proposal and graph-buffer tests pass **7/7**; ServerArgs namespace/CLI metadata tests pass **7/7 with 2 subtests**; modified Python modules compile and `git diff --check` passes.
- The rejected simulated `nvfp4_online` server is still the listener on port 30000 (`pwsh` parent `32780`, listener process `25512`). Next: stop that exact process tree, launch top-k-20 aligned `q` with the normal BF16 MTP under forced acceptance to measure proposal overhead, then relaunch unsimulated for acceptance histogram and real TPS if startup/graphs pass.

### 2026-08-16 10:25:05 PDT — naïve top-k-20 draft-q alignment is a measured loss

- The opt-in candidate captured and replayed all three graphs successfully with normal BF16 MTP storage and target XQA. Forced-acceptance exact `6213/512` samples were `125.359, 141.188, 142.769, 142.825, 99.766 tok/s`; the clean middle band is about **142.26 tok/s**, roughly 10% below the qualified **156.968 tok/s** fixed-work mean. Sparse Torch top-k/top-p plus dense-q scatter and replay-time penalty copies therefore impose a substantial tax.
- Honest rejection sampling did not repay it. The first two real exact `6213/512` samples fell to **78.507** and **62.269 tok/s**, versus the qualified ten-run **110.750 tok/s**. A matched native acceptance probe reported `spec_accept_length=2.4976`, `spec_accept_rate=0.4992`, `307/615` correct/proposed drafts, and histogram `[57,53,31,64]` across 205 verifies.
- Added `scripts/windows/bench_spec_acceptance.py` to expose the server's built-in speculative counters on the exact calibrated benchmark shape; it avoids inferring proposal quality from TPS alone.
- This refutes the audit's implied assumption that mechanically mirroring target top-k/presence/top-p will improve this MTP checkpoint. The draft's ranking/calibration error and the added sampler work dominate. Keep the implementation opt-in for narrower/per-depth calibration experiments; do **not** promote top-k 20.
- The live listener is currently the losing unsimulated top-k-20 candidate under PowerShell parent `35316`. Next: restore the qualified default q topology, take one matched native acceptance control, then search cheaper/narrower q calibrations only if the control confirms the expected acceptance gap.

### 2026-08-16 10:27:15 PDT — matched q control complete; paused on qualified production topology

- Restored the qualified unsimulated full-200K XQA + draft-extend-graph topology under PowerShell parent **`2508`**. Startup resolved `speculative_draft_sampling_top_k=None`, ordinary BF16 MTP weights, `speculative_attention_mode='decode'`, and no simulated-acceptance setting. All three graphs captured, `/health` returns HTTP 200, and this server remains live on port 30000.
- The matched native default-q acceptance control measured `spec_accept_length=2.3925`, `spec_accept_rate=0.4642`, `298/642` correct/proposed drafts, and histogram `[68,55,30,61]` across 214 verifies. The top-k-20 candidate's single matched probe improved accepted length only from `2.3925` to `2.4976` (`+4.39%`) and reduced verify cycles from 214 to 205 (`-4.21%`).
- That modest work-efficiency gain cannot repay the candidate's roughly 10% clean fixed-work cost, and the real OpenAI samples regressed sharply. Mechanical q/p transform matching is therefore rejected in its current Torch implementation. The audit should describe proposal calibration as a measured search direction, not an unqualified largest win.
- Worktree intentionally retains the opt-in implementation, focused tests, acceptance probe, and these notes for a future fused or per-depth calibration pass. `sglang.bundle` remains user-owned and untouched. Pause requested by the user; do not launch another candidate until resumed.

### 2026-08-16 10:32:31 PDT — work resumed; inactive sampling-backend claim is first

- The user supplied and verified the compact `notes/` working set through the 10:27 checkpoint, then explicitly resumed the work. The restored qualified server remains healthy on port 30000 under PowerShell parent `2508`; the worktree still contains the proposal-alignment stream and user-owned `sglang.bundle` is untouched.
- Source reconfirmation supports GPT Pro's reachability concern: steady EAGLE rejection sampling builds target probabilities and calls `chain_speculative_sampling_triton` directly, while draft proposals use `fast_sample`; neither reaches the generic `Sampler._sample_from_probs` branch selected by `--sampling-backend`. The generic sampler can still serve non-speculative/fallback work, so live attribution is required before changing the selected launcher.
- No separate SGLang sampling workspace allocation was found; FlashInfer attention already owns the measured 128 MiB shared workspace. The audit's claimed roughly 100 MiB recovery is therefore unproven in this tree.
- Next controlled branch: relaunch the exact qualified topology with only `SamplingBackend='pytorch'`, compare graph-end headroom and forced-acceptance `6213/512` cost, then take sampled work only if the fixed path differs materially. Restore the qualified server afterward unless the isolated candidate wins.

### 2026-08-16 10:37:52 PDT — generic sampling-backend branch closed; no hot-path or memory win

- Source reachability is conclusive for the selected EAGLE rejection topology: target verification directly invokes `chain_speculative_sampling_triton`, and draft proposals directly invoke `fast_sample`. `Sampler._sample_from_probs`, where `sampling_backend` chooses FlashInfer versus PyTorch, is outside steady speculation.
- Isolated full-200K forced-work A/B confirms the same execution topology. `sampling_backend='pytorch'` produced `148.961, 137.123, 148.903, 152.878, 153.307 tok/s` (mean `148.234`, median `148.961`). The immediately following FlashInfer control produced `151.877, 152.222, 148.603, 147.185, 147.547 tok/s` (mean `149.487`, median `148.603`). Identical output digest and overlapping medians show no backend-controlled steady-state effect; mean movement follows the known WDDM band.
- The memory claim also fails to reproduce. Graph-end headroom was `1.84 GB` for PyTorch and `1.80 GB` for FlashInfer, while pre-target-capture headroom moved in the opposite direction (`1.92` versus `2.04 GB`). No dedicated SGLang sampler workspace exists, so these allocator-scale differences do not establish a reclaimable ~100 MiB buffer.
- Retain the qualified FlashInfer launcher default because the branch offers no measured speed or capacity gain and still governs ordinary/fallback sampling. The currently live server is the **simulation-only FlashInfer fixed-work control** under PowerShell parent `24924`; restore an unsimulated winner after the next controlled candidate or before any user-facing handoff.

### 2026-08-16 10:44:47 PDT — two-step MTP becomes the new real-throughput leader

- Reopened the previously rejected two-step branch because captured draft extension and target-verification XQA materially changed its relative cost. Full-200K graph capture succeeds at a three-token verify width and leaves about `1.84 GB` headroom.
- With forced accepted length 3, two-step exact `6213/512` measured **`158.524, 158.364, 159.518, 164.063, 159.397 tok/s`**, mean **`159.973 tok/s`**. It emits the same forced three-token run as the established simulation while avoiding the unused third proposal, exceeding the prior qualified three-step fixed mean `156.968` by `1.91%`.
- Honest sampled reasoning reproduced across two five-run windows. Window one: `116.744, 114.194, 120.419, 121.118, 113.112` (mean `117.117`). Window two: `114.524, 117.837, 114.086, 120.082, 125.822` (mean `118.470`). Combined ten-run mean is **`117.794 tok/s`**, **`+6.36%`** over the three-step XQA winner `110.750`.
- Matched native counters report acceptance length `2.3167`, rate `0.6561`, `290/442` correct/proposed drafts, histogram `[52,48,121]`, and 221 verify cycles for 512 tokens. The prior three-step control needed 214 verifies; seven extra cycles are more than repaid by the narrower draft and verify work.
- This is the current performance leader pending reasoning/tool/long-capacity qualification and launcher promotion. The live server is unsimulated two-step under PowerShell parent `12536`. Before promotion, test one step under the same topology; its first-draft acceptance inferred from the two-step histogram is about `76.5%`, so it remains a plausible cost/acceptance point.

### 2026-08-16 10:47:54 PDT — one-step MTP is decisively closed again

- Full-200K one-step capture succeeds with a two-token target-verify graph and a two-token captured draft-extend graph; there is no multi-step draft-decode graph because the single proposal is seeded by the prior extend output.
- Even under the ideal forced accepted length 2, exact `6213/512` measured only **`102.142, 102.107, 101.282 tok/s`**. Each cycle still costs about `19.6 ms`, essentially the same order as the two-step cycle, while emitting at most two tokens instead of three. The fixed loss is roughly 36% versus the two-step `159.973 tok/s` result and cannot be repaired by real acceptance.
- Stop after three samples because the effect is large, stable, and mechanistically explained; an unsimulated window would only add acceptance loss. Any adaptive controller for this batch-one topology must exclude one step. The useful choice set is two versus three, with static two currently leading.
- The live server is the **simulation-only one-step rejection candidate** under PowerShell parent `27988`. Next restore unsimulated two-step, run the reasoning/tool/surface gates, and then exact `199000+16` before changing launcher defaults.

### 2026-08-16 10:53:39 PDT — two-step passes qualification and becomes the launcher default

- Restored unsimulated two-step under parent `25660`. Recommended Qwen sampling preserved coherent `reasoning_content`, normal stop, and exact final answer **703**. Tool parsing preserved reasoning and emitted exactly one `multiply` call with arguments `{"a":37,"b":19}` and `finish_reason=tool_calls`.
- `/model_info` continues to report `has_image_understanding=false`, `has_audio_understanding=false`, multimodal support disabled, and the language-only Qwen architecture.
- Exact `199000+16=199016` completed at **`2608.263 prompt tok/s`**, **`102.358 generation tok/s`**, and `76.442544 s` E2E. This preserves the real 200K contract; the short generation figure is an encouraging sample rather than a stable long-context mean.
- Promoted launcher defaults from three steps/four draft tokens to **two steps/three draft tokens** and updated its description. Focused speculative/XQA coverage passes **175 tests plus 7 subtests**; ServerArgs metadata/namespace coverage passes **7 tests plus 2 subtests**; modified Python and acceptance-probe modules compile; PowerShell parsing and `git diff --check` pass.
- Relaunch once from defaults with no explicit speculative geometry or simulation, prove resolved args/readiness, and leave that server live while exploring adaptive two-versus-three policies read-only. The ten-run **117.794 tok/s** result is now the qualified performance reference.

### 2026-08-16 10:56:30 PDT - post-compaction checkpoint

- Qualified leader is the **two-step / three-draft-token** MTP topology: ten real sampled runs average **117.794 tok/s** (two five-run windows **117.117** and **118.470**, peak **125.822**), **+6.36%** versus the superseded 110.750 tok/s three-step leader.
- Fixed exact `6213/512`, forced accepted length 3 averages **159.973 tok/s**. Native acceptance: length **2.3167**, rate **0.6561**, `290/442`, histogram `[52,48,121]`, 221 verifies.
- Semantic gates pass: preserved reasoning with exact answer 703; one exact `multiply` tool call with `{"a":37,"b":19}` and `finish_reason=tool_calls`; `/model_info` reports vision/audio disabled.
- Exact `199000+16` passes at **2608.263 prompt tok/s** and **102.358 generation tok/s** in 76.442544 s.
- Launcher defaults are now 2 steps / 3 draft tokens. Validation: 175 focused tests plus 7 subtests; ServerArgs 7 tests plus 2 subtests; Python compile, PowerShell parse, and `git diff --check` pass.
- One-step is rejected by fixed-work samples `102.142, 102.107, 101.282`. Generic PyTorch-versus-FlashInfer sampling is closed as effectively equal in the speculative hot path, with no demonstrated 100 MiB recovery.
- A default-only, unsimulated two-step launch was started under PowerShell parent PID **32156**; resolved args already show the intended topology. Readiness and graph-completion verification remain the immediate continuation task.
- Next performance branch is adaptive depth over **2 and 3 only**; step 1 is excluded. Preserve exact-200K capacity and all semantic gates for any promoted candidate.

### 2026-08-16 10:58 PDT - default promotion launch verified

- Default-only unsimulated launch is healthy: PowerShell parent PID **32156**, worker PID **15352**, `/health` HTTP **200**.
- Resolved default topology completed target-verify, draft-decode, and draft-extend CUDA-graph capture; Uvicorn became ready at 10:55:10 with **1.70 GB** reported post-capture headroom.
- The qualified **117.794 tok/s** two-step configuration is therefore both the launcher default and the live production server before adaptive experiments.

### 2026-08-16 11:00 PDT - adaptive 2/3 candidate prepared

- Added launcher-only opt-in controls for `--speculative-adaptive` and its config path; production defaults remain the qualified static two-step topology.
- Added `scripts/windows/qwen38_adaptive_2_3.json`: candidates `[2,3]`, EMA alpha 0.5, per-cycle decisions after eight warmup batches, rise threshold 1.6 accepted drafts, drop threshold 1.5.
- The JSON parses and resolves exactly to candidate steps `[2,3]`; PowerShell parsing and `git diff --check` pass.
- The combined adaptive/worker unit selection reached **31 passed + 7 subtests**. Nine config-file tests hit the known Windows `NamedTemporaryFile` sharing lock (`PermissionError` while reopening the still-open file); every failure has that test-harness cause, and the real repository config loads successfully.
- Next: replace the healthy static server temporarily, measure adaptive graph residency and switching on the real sampled workload, and restore static two-step unless adaptive clears the two-window promotion bar.

### 2026-08-16 11:02:37 PDT - adaptive startup exposed shared-logits sizing defect

- Adaptive `[2,3]` captured the initial target-verify graph and both draft decode/extend state shapes, retaining **1.66 GB** reported headroom, then failed before readiness.
- Exact failure: `GraphSharedOutput.get_logits_buffer` asserted that its shared logits buffer holds **3 rows** while the three-step adaptive target graph requires **4 rows** (`vocab_size=248320`).
- This is a real adaptive runtime defect rather than capacity exhaustion: adaptive-aware max draft-token sizing exists elsewhere, but the target graph shared-output allocation still used the initial static three-token width.
- No candidate result exists yet. Next: trace that allocation boundary, size it from the resolved adaptive maximum, add a regression test, and relaunch.

### 2026-08-16 11:07 PDT - adaptive runs; aggressive controller rejected

- Fixed the shared-logits allocation to use `ServerArgs.max_speculative_num_draft_tokens`; added a focused regression test. The test pair passes **2/2**, Python compilation and diff checks pass, and adaptive `[2,3]` now reaches readiness.
- Both adaptive graph states captured while preserving the exact 200K pool. The extra three-step runtime state costs about **0.55 GB**, leaving **1.31 GB** reported startup headroom.
- The alpha-0.5/per-cycle policy oscillates almost every decision between steps 2 and 3, including synchronous info logging for each transition. First real sample was only **100.739 tok/s**.
- Matched acceptance probe: length **2.0729**, rate **0.5364**, `265/494`, histogram `[91,62,79,15]`, **247** verifies. This is decisively worse than static two-step length 2.3167 / 221 verifies.
- Reject this aggressive policy. Next candidate must switch sparsely: slower EMA, a higher rise threshold, and a multi-cycle decision interval so its floor behavior approaches the 117.794 tok/s static leader.

### 2026-08-16 11:11:50 PDT - sparse adaptive 2/3 also rejected

- Retuned the opt-in controller to alpha 0.2, decisions every 8 cycles after 16 warmup cycles, rise threshold 1.7, drop threshold 1.5. It switched only 16 times across the measured activity instead of oscillating each cycle.
- Real sampled windows: `107.210, 108.305, 109.027, 117.738, 113.295` (**111.115 mean**) and `103.454, 108.219, 113.375, 110.797, 111.340` (**109.437 mean**). Combined ten-run mean: **110.276 tok/s**.
- This loses **6.38%** to the qualified static-two-step mean of 117.794 tok/s. A matched acceptance probe with no three-step acceptances in its histogram still measured only length **2.1070**, `270/486`, and 243 verifies, showing substantial stochastic acceptance variance but no adaptive win.
- Adaptive depth is closed for this topology: one-step is structurally too slow, static three-step is slower, and both aggressive and sparse two/three policies lose. Restore static two-step and use it as the baseline for the next bottleneck.

### 2026-08-16 11:21:39 PDT - compact unread XQA mask is a fixed-work win

- Fresh restored static-two-step real control before this change averaged **120.091 tok/s** over `123.203, 118.597, 121.759, 118.255, 118.643`, confirming the adaptive loss was real and the machine was healthy.
- Root cause: target XQA already uses its own capture-time packed causal chain mask, but the generic EAGLE tree builder saw no backend mask and allocated/initialized a context-sized `FULL_MASK` every decode cycle.
- TRT-LLM MHA now advertises a compact fixed-capacity `QLEN_ONLY` sink marked unread; the tree builder also skips its redundant prefill when the sink is unread. Focused mask/top-k1 tests pass **24 + 11 subtests**; Python compile and diff checks pass.
- Fixed exact `6213/512`, forced accepted length 3 samples: `162.517, 162.268, 163.131, 163.863, 161.851`; mean **162.726 tok/s**, **+1.72%** versus the qualified 159.973 fixed mean. Digest is unchanged.
- The first unsimulated five samples before the fixed gate averaged only **109.882 tok/s** because of a low-acceptance stochastic window; a matched later probe recovered acceptance length **2.2857**, `288/448`, 224 verifies. Relaunch unsimulated and require two fresh real windows before promotion.

### 2026-08-16 11:27:56 PDT - compact-mask real sampling remains acceptance-limited

- Fresh unsimulated windows after the fixed win averaged **118.582** and **115.018 tok/s**; ten more samples averaged **117.166**. Across those 20 fresh samples the mean is **116.983 tok/s**, 0.69% below the earlier qualified 117.794 mean and well inside the observed acceptance-driven spread.
- Five additional native probes averaged accepted length about **2.132** (individual `2.160, 2.090, 2.048, 2.107, 2.256`) and 240.4 verify cycles, materially worse proposal luck than the earlier qualified 2.3167 / 221 control. The compact mask is not read by TRT-LLM MHA/XQA and cannot alter q, p, or rejection semantics.
- Keep the source change as a deterministic execution-path win (**162.726** fixed versus **159.973**) while leaving **117.794 tok/s** as the qualified real-sampled headline until a later cumulative window actually exceeds it.
- Next: specialize the remaining fixed top-k1 chain metadata itself; the current path still allocates retrieval/position tensors and launches the general ancestry tree builder even though XQA owns an unread fixed-chain mask.

### 2026-08-16 11:33:27 PDT - fused fixed-chain metadata is a major fixed-work win

- Added a one-launch Triton top-k1 chain metadata kernel that fuses bonus/draft token assembly with position construction. Preallocated invariant retrieval links and siblings replace per-cycle allocation/fill and the general ancestry tree walk when the selected backend advertises an unread mask.
- The specialization is gated to top-k1, a fitting fixed-width buffer, and an unread backend mask; all other backends/tree shapes retain the general path.
- Validation: focused worker/mask tests **24 passed + 11 subtests**; CUDA top-k1 kernel tests **6 passed + 7 subtests**; Python compile and `git diff --check` pass.
- Fixed exact `6213/512`, forced accepted length 3 samples: `171.252, 171.899, 169.961, 168.229, 167.495`; mean **169.767 tok/s**. A sixth sample reached **171.811 tok/s** with the exact established digest.
- This is **+4.33%** over compact-mask-only 162.726 and **+6.12%** over the previously qualified 159.973 fixed mean. Next: relaunch unsimulated, collect two real windows plus acceptance, then run semantic and exact-200K gates if the real result promotes.

### 2026-08-16 11:40:23 PDT - fixed-chain real regression isolated with server seed

- Unseeded fused-metadata real sampling remained acceptance-poor: first ten mean **115.491 tok/s**; five acceptance probes averaged length about **2.083**.
- Added an optional launcher `RandomSeed` control (default omitted) and relaunched with the exact seed **783025237** from the fresh pre-change static control.
- This locks the five target outputs across restarts: all five output SHA-256 values exactly match the pre-change run in order. The fused-metadata TPS was `121.908, 110.904, 113.649, 110.858, 114.501` (**114.364 mean**) versus the exact-output pre-change `123.203, 118.597, 121.759, 118.255, 118.643` (**120.091 mean**).
- Therefore the fused reusable metadata buffers have a real rejection-path scheduling/aliasing cost despite their **169.767 tok/s** simulated fixed-work result. Do not promote them on the simulation number.
- Next controlled split: disable only the reusable/fused metadata handoff while retaining the compact unread XQA mask, relaunch the same server seed, and compare the identical output sequence. This will attribute the safe compact-mask change separately.

### 2026-08-16 11:43:45 PDT - post-compaction checkpoint: exact-output split in flight

- Qualified production leader remains static two-step / three draft tokens: real ten-run **117.794 tok/s**, deterministic fixed-work **159.973 tok/s**, and exact `199000+16` **2608.263 prompt / 102.358 generation tok/s** at the full 200K pool.
- Adaptive speculation is closed for this topology. Aggressive alpha 0.5/per-cycle sampling reached **100.739 tok/s** (accepted length 2.0729, 247 verifies); sparse alpha 0.2/every-eight sampling reached **110.276 tok/s**, 6.38% below static two-step. The adaptive startup shared-logits allocation defect is fixed by sizing from the adaptive maximum, with a focused regression test.
- Compact unread XQA mask specialization removed the redundant context-sized generic FULL_MASK construction/fill and raised deterministic fixed work to **162.726 tok/s** (**+1.72%**). Its unseeded real 20-run mean was acceptance-poor at **116.983 tok/s**, so it is undergoing a same-seed exact-output split.
- Fused/reusable top-k1 metadata reached **169.767 tok/s** fixed (sixth sample **171.811**) but regressed exact-output real sampling: seed 783025237 produced the same five SHA sequence yet averaged **114.364 tok/s**, versus the pre-change **120.091 tok/s**. This experiment is rejected and must be removed completely after the split.
- Current server: PowerShell parent PID **30536**, stderr `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-compactmask-seed783025237.stderr.log`, seed **783025237**. Worker calls pass `topk1_chain_buffers=None`, disabling only fused/reusable metadata while retaining the compact unread mask. The server reached HTTP 200 and an exact five-run benchmark was launched just before compaction; recover its result or rerun it.
- Expected exact output hashes, in order: `03063ef3330b630698a5de96af886012d65402ec20ec9c24f1a78aedd5311e70`, `9bbf0df86fafa97330735139d684e425e822ce4475c17bb3eadcf84159ef38e9`, `0025d4c81d1f4708340d196ac6d87c1f62826efca4478ce8196a9970755927b4`, `ba29eb8288bfa5c905333b6d2325e04790951ae683472d2bd23dafe96c2c5fca`, `3900543fd39fccf65ff459b0215790b44ee0ce5fda5c1c56a7c72245c8f8266c`.
- Immediate next action: resolve compact-mask-only TPS against the identical-output **120.091** control, then remove the rejected fused metadata code/tests/preallocation, validate, and continue into dense MTP GEMM/fusion and long-context branches.

### 2026-08-16 11:49:10 PDT - GPU-load safety pause during seeded reset

- A second five-request pass on the already-advanced seeded server averaged **116.243 tok/s** and produced the next RNG sequence, so it is not the required first-five exact-output comparison.
- Resetting seed 783025237 required a full server restart. Startup spent **38.48 s** capturing target-verify, **1.22 s** capturing draft-decode, and **1.03 s** capturing draft-extend CUDA graphs; total tokenizer startup was **65.74 s**. This saturated the display GPU and temporarily froze the desktop.
- Replacement parent PID is **31968**; the server became ready at 11:46:57 and is now idle. No benchmark is running. Pause all GPU-heavy benchmarks, restarts, compilation, and capture until Daniel confirms the desktop is responsive.
- Safety adjustment: avoid repeated seed-reset restarts. The next exact first-five pass can run from this already-reset idle server once approved, after which resolve the split and return to source-only work between deliberately scheduled GPU gates.

### 2026-08-16 11:51:58 PDT - emergency cleanup verified

- Daniel observed 11 processes during the startup freeze and N/A GPU reporting. Stopped only the exact seven-process replacement server tree, leaf-first: PIDs `35968, 16900, 7208, 26740, 10764, 13680, 31968`. A follow-up PID query returned empty.
- Verified there are no remaining `sglang.exe`, `curl.exe`, `ptxas.exe`, `nvcc.exe`, `ninja.exe`, `cmake.exe`, or `cl.exe` processes. The only remaining Python pair is the unrelated pre-existing Quasimorph MCP server (`37620 -> 15832`) and was left untouched.
- NVIDIA driver telemetry is responsive again: RTX 5090 / driver 610.88, 30 C, 1191 MiB of 32607 MiB after settling, and no SGLang/Python compute process. Full `nvidia-smi` reports only ordinary Windows WDDM graphics clients.
- The N/A fields still shown by full `nvidia-smi` are WDDM per-process memory, ECC, and MIG fields; live GPU temperature, utilization, memory, power, and performance-state readings are populated. Keep all inference/server work stopped until Daniel explicitly resumes it.

### 2026-08-16 11:59:36 PDT - native C++/CUDA track and rejected-code cleanup

- Daniel confirmed the RTX 5090 returned to 29 C / 0% idle and authorized work to resume. Future server starts remain single, deliberate GPU gates with exact process-tree cleanup and telemetry checks.
- Completely removed the rejected fused/reusable top-k1 metadata experiment: custom Triton kernel, buffer dataclass/allocation, worker handoff, and its tests. Preserved the earlier qualified top-k1 parent/score fast path and the compact unread XQA mask candidate.
- Cleanup verification: `git diff --check` passes; focused mask, shared-logits, and top-k1 worker suite passes **25 tests + 11 subtests**. No model or CUDA graph was loaded for this validation.
- Daniel explicitly expanded the optimization objective to convert measured steady-state Python hot paths into native code, then narrowed the implementation language to **C++/CUDA only**. Use Python only as the required thin SGLang binding/integration surface. Promote every conversion on real sampled end-to-end TPS plus exact semantic gates.

### 2026-08-16 12:12:23 PDT - first C++/CUDA hot paths enabled on native Windows

- Found a broad native-Windows gap: `sgl_kernel` is absent, so SiLU-and-multiply and standard/Gemma RMSNorm were routed through multi-op PyTorch implementations even though this tree already contains native C++/CUDA JIT kernels.
- Unblocked CUDA 13.3/MSVC JIT compilation centrally: Windows host/device builds now pass `/Zc:preprocessor`, required by current CCCL. Fixed a second MSVC portability failure in `fused_add_rmsnorm.cuh` by replacing non-standard `uint` with `uint32_t`. The conforming-preprocessor unit test passes.
- Extended the native RMSNorm CUDA kernel with a runtime weight offset and exposed `gemma_rmsnorm`; this directly covers Qwen3.5 MTP's two 5120-wide `GemmaRMSNorm` inputs. The custom op is opaque to Dynamo and a fullgraph integration check passes exactly.
- Native-Windows production dispatch now uses rounding-preserving C++/CUDA SiLU-and-multiply plus native standard/Gemma RMSNorm. Fused add-RMSNorm remains on the old path for now because its output differs by up to one BF16 step; its residual is exact and it remains a later controlled candidate.
- Exact Qwen-shape microbench (`M=1/3`, BF16): SiLU-and-multiply **~10-11 us vs 24-28 us** (bit-exact); RMSNorm **~9-10 us vs 78-85 us** (bit-exact); Gemma RMSNorm **~9-11 us vs 89-90 us** (bit-exact). Native fused add-RMSNorm is **~19-23 us vs 96-108 us**, with max output delta 0.015625.
- Added `smoke_native_qwen35_hotpaths.ps1` and a CUDA-environment pytest wrapper. Smoke/fullgraph checks pass; the two Qwen 5120 Gemma cases pass exactly. The broader JIT-cache suite is 40 passed / 3 pre-existing Windows path-format assumptions failed; the new toolchain test passes independently.
- Post-probe cleanup is clean: no SGLang, NVCC, CL, Ninja, or curl processes remain. Next gate is a single controlled model start to measure whether torch compilation had already hidden part of these eager microbench wins and to verify exact sampled semantics.

### 2026-08-16 12:21:41 PDT - post-compaction checkpoint: C++/CUDA-only hot-path track

- User direction is explicit: keep performance implementation work in C++/CUDA, with Python limited to thin bindings. Do not pursue Rust or Python/Triton rewrites for new hot paths.
- One intentional qualified server is live under parent PowerShell PID 36396. Its exact seven-process tree at the last check was 36396, 32992, 25776, 27664, 7580, 22272, and 28080. Do not broad-kill it; verify the exact tree before any lifecycle action.
- Current log is `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-nativehotpaths-seed783025237.stderr.log`; configured server seed 783025237 was verified. Startup captured target verify in 43.71 s, draft decode in 1.34 s, and draft extend in 1.21 s. Weight load was 19.68 s and scheduler startup 76.66 s.
- New native Windows paths are implemented for rounding-preserving SiLU-and-multiply, standard RMSNorm, and Gemma RMSNorm with runtime weight offset. CUDA 13.3/MSVC JIT support was repaired with `/Zc:preprocessor`, `<cstdint>`, and `uint32_t`. Isolated Qwen-shape checks are bit-exact and approximately 2.4-9x faster than the former eager PyTorch compositions.
- Native fused-add RMSNorm remains gated because output can differ by one BF16 step even though the residual is exact. Treat draft-only/MTP use as the first permissible evaluation; do not silently move the target path.
- Current native-server sampled observations: first five mean 123.132 tok/s; later warm ten mean 116.844 tok/s, median 117.983. Five acceptance probes average 2.188 accepted length. These do not yet promote the 117.794 tok/s qualified production baseline because speculative RNG differs across changed capture/compile topology.
- Immediate analysis item is CUDA-graph RNG lifecycle: inspect seed placement, captured generator/state handling, and whether a semantically safe post-capture reseed exists. Request-level `seed=42` did not make speculative sampling repeat. Preserve production randomness and do not patch merely for benchmark cosmetics.
- Highest-value later C++/CUDA candidates remain a selective native Windows speculative tree/sampling extension from the existing AOT CUDA sources, draft-only native fused-add RMSNorm, first-request kernel preloading/registration, the five dense BF16 MTP GEMMs, and exact-shape CUTLASS/cuBLAS dispatch.
- Safety remains one deliberate GPU job at a time, sequential requests, no readiness loops, exact process-tree checks, and compiler-worker cleanup checks. Protected CUDA headers in both FlashInfer trees remain out of scope and must retain SHA256 `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.

### 2026-08-16 12:28:05 PDT - native fixed-chain metadata candidate ready for compile gate

- Added a selective C++/CUDA JIT path for native-Windows top-k1 QLEN_ONLY verification metadata. One CUDA launch now constructs the causal chain mask, positions, retrieve index/next/sibling tables, and concatenated bonus/draft tokens.
- The specialization removes the active path's token `cat`, `full` initialization, prefix `cumsum`, and general Triton tree kernel. It retains newly allocated per-cycle outputs to preserve the asynchronous verification lifetime; the earlier unsafe reusable-output behavior is not reintroduced.
- Dispatch is deliberately narrow: CUDA + native Windows sgl-kernel fallback + top-k1 + `num_verify_tokens == spec_steps + 1` + preallocated QLEN_ONLY mask + int64 token/length inputs. Every other topology stays on the established general path.
- Added `test_chain_metadata.py` covering 1/2/7-step chains, multi-request indexing, capacity-sized mask tails, token order, positions, and retrieve links. Python syntax and `git diff --check` pass. CUDA/MSVC compilation and parity remain pending and require taking down the one intentional model tree first.

### 2026-08-16 12:42:29 PDT - native metadata passes; FlashInfer sampling dispatch defect fixed

- Stopped the one intentional server by its verified seven-PID tree. Killing leaf PIDs 22272 and 28080 caused the remaining tree to exit cleanly. Post-stop checks found no SGLang, Python-SGLang, NVCC, CL, Ninja, or CMake worker. GPU returned to 0% utilization and approximately 28-30 C.
- The new C++/CUDA chain-metadata JIT compiles under CUDA 13.3/MSVC and passes 1 test + 4 CUDA subtests. Production-shape microbenchmark (`bs=1`, two MTP steps, three verify slots): general Windows Triton path 64.745 us versus native 15.316 us, **4.227x**, saving 49.429 us per speculative cycle.
- Found a concrete dependency dispatch defect: native-Windows speculative target verification ignored the already-installed FlashInfer 0.6.17 CUDA renormalization kernels and used a fallback that sorts the entire 248,320-token vocabulary for top-k, then sorts it again for top-p.
- Production-shape microbench medians across repeated runs: old dense Windows renorm about 700-1,017 us; FlashInfer dense renorm about 175-185 us; existing dense rejection kernel about 70-75 us. FlashInfer saves roughly 0.52-0.84 ms per verify before end-to-end effects.
- `eagle_utils.py` now resolves FlashInfer CUDA top-k/top-p renorm once on native Windows, retaining the Triton implementation only as an import fallback. This makes the configured FlashInfer dependency control the actual speculative target path.
- Draft/target proposal alignment is now wired into the multi-layer Qwen MTP worker. With launcher draft top-k 20, q applies additive penalties/logit bias, temperature, top-k 20, and top-p before sampling; the exact resulting dense q is retained for Leviathan rejection. Native Windows uses FlashInfer CUDA renorm because measured dense-aligned q (about 183 us/step) beats FlashInfer sparse top-k q (about 305-334 us/step). The former full-vocab q was about 64 us/step, so the extra draft cost is more than covered by target-renorm savings even before higher acceptance.
- Existing proposal and draft-graph unit tests pass (7 total). Single-CG draft capture now declines sparse/aligned q because that graph currently bakes the temperature-only proposal; the qualified per-step draft graphs remain active.
- Added reusable Windows CUDA Python/pytest launch wrappers and retained microbench scripts for chain metadata and sampling dispatch. No model server is currently running.

### 2026-08-16 12:53:12 PDT - first promoted real-sampled result with native chain + aligned FlashInfer q/p

- Controlled server parent PID 29880 is healthy with exact seven-process tree 29880, 9588, 36840, 36256, 11740, 13512, 29564. Log: `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-flashinfer-qalign-seed783025237.stderr.log`.
- Resolved runtime is exact 200,000 context, vision/language-only, Qwen3 reasoning parser, Qwen3 Coder tool parser, two MTP steps/three draft tokens, target and draft TRT-LLM/XQA verification/decode, rejection sampling, and aligned draft top-k 20.
- First startup after the sampling-image/topology change was cold: load 23.47 s, target verify capture 54.53 s, draft decode 50.51 s because FlashInfer autotune ran for about 49 s, draft extend 1.04 s, scheduler 135.00 s. Available GPU memory after init is 1.70 GiB.
- First request safely loaded four remaining prefill/rejection Triton modules with a 0.50 GiB low-water mark: chunk gated-delta, chunk output, layer norm, fused sigmoid-mul, and the rejection kernel. No OOM or runtime correctness error. These are the next preload/native-port targets.
- Fresh warm real-sampled ten-run TPS: `129.757, 117.145, 124.366, 101.657, 112.796, 117.329, 122.303, 131.613, 127.563, 126.224`; mean **121.075 tok/s**, median **123.335 tok/s**. This is +2.79% over the qualified 117.794 real mean. The low 101.657 outlier remains acceptance-dependent.
- Earlier five-run window on the same server was `122.813, 117.285, 117.560, 116.798, 121.590`, mean **119.209 tok/s**.
- Five aligned-q acceptance probes: `2.4265, 2.1974, 2.2358, 2.2358, 2.2960`, mean **2.2783** accepted tokens/verify, versus 2.188 from the preceding native temperature-only sample. The aligned distribution improves work efficiency while the CUDA renorm dispatch lowers per-verify cost.
- Ten-run observed prompt processing mean is approximately **8,501 tok/s** and median approximately **9,447 tok/s** on the 6,213-token contract, with wide 4,970-9,682 variation. This is a strong short-prompt signal from the native SiLU/RMSNorm path but still needs the exact long-context gate.
- GPU remained cool at 38 C after the sequential window. Next gate: exact fixed-work restart (`SimulateAcceptedLength=3`) to isolate execution speed from acceptance, followed by return to the real-sampled winner.

### 2026-08-16 12:57:32 PDT - post-compaction checkpoint; C++/CUDA-only continuation

- User narrowed all further optimization implementation to **C++/CUDA**. Python remains only a thin binding, dispatch, test, or launch layer; do not pursue Python hot-path rewrites. Supporting dependencies remain in scope, while the two protected CUDA headers remain untouched.
- One intentional fixed-work simulation server is live under parent PowerShell PID 2760 (`-RandomSeed 783025237 -SimulateAcceptedLength 3`), with observed direct children conhost 5304 and sglang 23832. Its stderr log is `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-flashinfer-qalign-fixed3.stderr.log`. This is a simulation-only topology and must not be used for semantic or production promotion.
- Fixed all-accepted warm five: `130.275, 128.752, 125.085, 130.504, 129.601 tok/s`, mean **128.843 tok/s**; preceding first run 130.255 tok/s. Logs report accept length 3.00, acceptance 1.00, and CUDA graph true.
- The 128.843 result is not yet classified as an execution regression. Fixed acceptance=3 may force the expensive all-accepted/double-draft-extend topology every cycle, whereas real sampling exercises that path only some of the time. Inspect simulation index generation and extend control flow before restarting or drawing a ceiling comparison.
- Current real-sampled promotion remains mean **121.075 tok/s**, median **123.335 tok/s**, acceptance mean **2.2783**, on exact 200,000-context, reasoning-preserved, tool-enabled, vision-disabled Qwen 27B NVFP4. Restore and leave this fastest qualified production server live after the fixed-work investigation.
- Next source work is restricted to native hot paths: explain/specialize the all-accepted extend topology; port fused sigmoid-multiply; preload or port first-request GDN/layer-norm work; then consider native rejection sampling, grouped/fused or low-precision MTP GEMMs, per-shape CUTLASS dispatch, and long-context CUDA execution.
- Safety discipline remains one server or compile/capture workload at a time, sequential benchmarks, exact process-tree verification, exact leaf-to-root stop, and post-gate orphan/toolchain/GPU checks. The prior eleven-process incident must not recur.

### 2026-08-16 13:08 PDT - fixed-work topology resolved; simulation server stopped cleanly

- Source tracing confirms the selected EAGLE v2 loop performs exactly one `_draft_extend_for_decode` after every target verification. `SGLANG_SIMULATE_ACC_LEN=3` replaces the accepted indices/tokens but does not introduce a second outer draft-extend pass in this topology.
- The current forced-acceptance result therefore isolates a real execution cost: aligned draft q remains eager outside the captured model forward and reduces the native-chain fixed mean to **128.843 tok/s**. The earlier 15% double-extend audit observation does not explain this run.
- The fixed server's exact seven-process tree was `2760, 5304, 23832, 31196, 19804, 23804, 20284`. Stopping only leaf PIDs 23804 and 20284 caused the entire tree to exit. A follow-up query found no SGLang, SGLang Python, NVCC, CL, Ninja, or CMake process.
- GPU returned to 29 C with 30,957 MiB free. The next optimization target is a C++/CUDA proposal path that removes full-vocabulary eager Torch allocation/dispatch and preserves the exact q used by rejection sampling.

### 2026-08-16 13:18 PDT - aligned q restored to the single multi-step CUDA graph

- Root cause of the fixed-work collapse is the execution topology: setting draft top-k 20 had deliberately disabled `SGLANG_ENABLE_SINGLE_CG_DRAFT`, forcing Python to run between the two captured MTP step graphs. The measured 128.843 tok/s therefore included a much larger graph-boundary/dispatch penalty than the CUDA renormalization kernels alone.
- The one-graph multi-layer runner now stages runtime top-p plus a combined penalties/logit-bias row in stable CUDA buffers. During capture, each MTP depth performs adjustment, temperature scaling, FlashInfer CUDA top-k/top-p renormalization, Gumbel sampling, and an exact-q snapshot inside the single multi-step graph. No Python executes between depths in steady replay.
- The temperature-only single-CG path is preserved unchanged. Aligned q uses the identical tensor for sampling and later Leviathan rejection, retaining correctness. Runtime `prepare()` updates current request values before replay, including changing penalties, bias, temperature, and top-p.
- Added a direct CUDA-graph replay test. It captures the aligned proposal, verifies exact q and sampled q(X), mutates logits/penalties/top-p after capture, replays, and verifies the graph consumed the new stable-buffer values. Focused result: **2 passed** in 64.84 s.
- Post-test checks found no SGLang/Python server or compiler worker. GPU is idle at 28 C with 30,848 MiB free. Next gate is a single controlled full-model capture; successful startup must explicitly report the single-CG draft graph before fixed and real measurements.

### 2026-08-16 13:23 PDT - new safe fixed-work record: 167.776 tok/s

- Controlled full-200K fixed launch uses seed 783025237 and simulated accepted length 3 under PowerShell parent PID 2764. Log: `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-singlecg-qalign-fixed3.stderr.log`.
- Cold startup completed normally: target verification graph 37.74 s; aligned draft decode graph 53.29 s (cold FlashInfer sampling/autotune image); draft extend graph 1.00 s; 1.84 GiB remained after graph capture. CUDA graph replay is active with accept length 3.00 / rate 1.00.
- Warmup fixed result: **164.151 tok/s**. Fresh five: `167.239, 167.279, 168.025, 168.157, 168.178`; mean **167.776 tok/s**. Every run retained the established fixed digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`.
- This is **+30.2%** over the broken aligned-q/per-step mean 128.843 and **+3.10%** over the previous safe compact-mask fixed mean 162.726. Server-log warm intervals reached **170.13 tok/s**. The rejected unsafe reusable-buffer experiment's 171.811 singleton remains only a ceiling datapoint; 167.776 is the new safe deterministic record.
- First-request JIT images reduce remaining free VRAM to about 350 MiB, while the GPU stays cool at 34 C and idle between requests. Do not add another concurrent CUDA process. Next: stop this exact simulation tree, restore unsimulated seeded production, and measure real throughput/acceptance before any further native kernel work.

### 2026-08-16 13:29 PDT - real production promotion: 122.712 tok/s mean, 137.074 peak

- The fixed server's verified seven-PID tree `2764, 21844, 36420, 33576, 26428, 18808, 24880` exited after stopping only leaf PIDs 18808 and 24880. Post-stop checks found no orphan server/compiler process and 30,893 MiB GPU memory free.
- Restored unsimulated full-200K production with seed 783025237 under PowerShell parent PID **8252**. Log: `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-singlecg-qalign-real-seed783025237.stderr.log`. Startup: target verify 39.34 s, draft decode 1.40 s, draft extend 1.03 s, 1.83 GiB graph-end headroom, HTTP ready.
- One initial sampled run measured 127.563 tok/s. Fresh ten: `122.002, 122.739, 113.058, 118.119, 118.948, 119.239, 137.074, 124.047, 125.909, 125.980`; mean **122.712 tok/s**, median **122.371 tok/s**. The 137.074 sample is the new real-production single-run peak.
- This is **+1.35%** over the preceding aligned-q production mean 121.075. Split windows remain acceptance-sensitive: first five 118.973, second five 126.450.
- Five native acceptance probes: `2.29596, 2.46154, 2.33790, 2.17872, 2.31674`; mean **2.31817** accepted tokens/verify, versus the preceding five-probe mean 2.2783. The deterministic fixed gain is the clean execution proof; part of the sampled-window gain is the improved acceptance window.
- Current server is unsimulated and is the fastest measured production topology. It remains live and idle. Next native work should preserve this server while source-only analysis targets C++/CUDA fused proposal/verification sampling, full-attention sigmoid gating, and the remaining BF16 MTP GEMMs.

### 2026-08-16 13:36 PDT - C++/CUDA full-attention gate candidate staged

- Added a native-Windows BF16 C++/CUDA sigmoid-multiply kernel for Qwen's contiguous full-attention gate (`attn_output * sigmoid(gate)`). It uses vectorized Blackwell-width loads/stores, FP32 math, optional PDL, supports true input in-place mutation, and leaves every strided/non-BF16/non-Windows case on the established Triton fallback.
- Dispatch is intentionally limited to the active fused QK-norm/RoPE output shape: same-shape contiguous CUDA BF16 tensors with vector-safe length. The module compiles on first model capture, moving this path out of first-request Triton JIT loading.
- Added focused exact-parity coverage at Qwen production width 6144 for 1/3/64 rows, including in-place dispatch, plus a reusable native-versus-Triton microbenchmark. Python syntax and `git diff --check` pass.
- CUDA/MSVC compilation, bit-exact parity, and timing are pending. The qualified production server under parent PID 8252 remains live and idle; do not compile beside its roughly 350 MiB post-JIT headroom. Stop its exact tree only when ready for the next controlled native gate.

### 2026-08-16 13:24:10 PDT - post-compaction continuation checkpoint

- `date` immediately after context rollover reported `Sun Aug 16 13:24:10 PDT 2026`. The existing notebook contains later-stamped entries, so preserve both the observed clock value and chronology rather than rewriting either.
- At rollover the user had paused to swap reasoning. The persistent optimization goal has now been continued, but no GPU workload, compilation, server stop, or source edit beyond this checkpoint has been started after compaction.
- Last-known qualified production server is the unsimulated seed-783025237 full-200K topology under hidden PowerShell parent PID **8252**, using `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-singlecg-qalign-real-seed783025237.stderr.log`. It was live and idle before rollover. Its exact current process tree and GPU ownership still require a fresh read-only check before any further work; never place a compile or second CUDA process beside it.
- Current production record: fresh-ten real mean **122.712 tok/s**, median **122.371 tok/s**, and single-run peak **137.074 tok/s**. Samples: `122.002, 122.739, 113.058, 118.119, 118.948, 119.239, 137.074, 124.047, 125.909, 125.980`. Five-probe mean accepted length: **2.318174**. The deterministic safe fixed-work record is **167.776 tok/s** from `167.239, 167.279, 168.025, 168.157, 168.178`, retaining digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`.
- Current production semantics passed before rollover. Reasoning probe for `37*19`: thinking preserved, stop finish, 74 prompt / 68 completion / 63 reasoning tokens, coherent reasoning, final answer `703`. Tool probe for the same multiplication: thinking preserved, `tool_calls` finish, 350 prompt / 84 completion / 45 reasoning tokens, exactly one `multiply` call with `{"a": 37, "b": 19}`.
- The aligned-q single multi-step draft CUDA graph is the active measured winner: proposal transforms and exact-q capture execute inside graph replay, recovering fixed throughput from 128.843 to 167.776 tok/s. The default temperature-only path remains intact, and direct mutable-buffer replay coverage passed **2 tests**.
- The next staged candidate is the native BF16 full-attention sigmoid-multiply gate in `fused_sigmoid_mul.cuh` plus its binding, dispatch, focused parity test, and benchmark. Python syntax and diff checks passed. CUDA/MSVC compilation, exact-bit comparison against the established Triton path, and production-shape timing remain pending. Retain it only if exact and faster.
- After that gate, the next controlled branch is explicit draft `fp8` quantization, followed only if warranted by a narrow `fc` conversion for the remaining raw BF16 MTP projection. Promotion still requires real sampled throughput, acceptance, reasoning/tool semantics, full 200K exactness, VRAM, process, and thermal evidence.

### 2026-08-16 13:26:06 PDT - live-winner ownership verified before native gate

- The qualified server is healthy and its complete seven-process tree is exactly `8252 -> {29924, 1952 -> 25320 -> 2144 -> {32392, 10152}}`. No NVCC, CL, Ninja, or CMake process is present. PID 32392 is the expected SGLang multiprocessing CUDA worker, not an orphan.
- GPU aggregate state is 30 C, 19% momentary utilization, 31,826 MiB used / 362 MiB free, 55.68 W. WDDM per-process memory remains reported as `N/A`; aggregate accounting and the exact process ancestry identify the single expected owner.
- `/health` succeeded and `/model_info` confirms `has_image_understanding=false`, `has_audio_understanding=false`, generation enabled, and the RadixArk NVFP4 model path. The launch command still resolves 200000 context/total tokens, Qwen3 reasoning and Qwen3 Coder tools, two MTP steps, target/draft TRT-LLM MHA, FlashInfer prefill, draft top-k 20, and FP8 draft KV.
- Worktree status was inspected before proceeding. All modified and untracked files are treated as user work; the native sigmoid candidate is the only next compile/test target. The exact two scheduler/tokenizer leaves above will be stopped deliberately before CUDA/MSVC compilation, then the tree and GPU will be rechecked.

### 2026-08-16 13:27:52 PDT - native BF16 sigmoid gate qualifies in isolation

- Stopping only CUDA-worker leaf PID 32392 caused the expected complete seven-process server tree to exit. Post-stop state: no SGLang or compiler worker, 28 C, 0% utilization, 1,287 MiB used / 30,901 MiB free. The unrelated Quasimorph MCP Python tree remains CPU-side and was left untouched.
- Native CUDA/MSVC JIT compilation and the exact-parity suite passed: **2 passed** in 13.78 s. Random BF16 Qwen-width tensors at 1, 3, and 64 rows matched the established Triton gate bit-for-bit; the true in-place dispatch also retained its data pointer and matched exactly.
- Isolated 6144-wide BF16 medians, 11 repeats x 500 launches: rows 1 native **10.704 us** vs Triton **11.164 us** (1.043x, 0.460 us saved); rows 3 native **10.778 us** vs Triton **11.819 us** (1.097x, 1.041 us saved); rows 64 native **11.169 us** vs Triton **11.558 us** (1.035x, 0.389 us saved).
- This is a qualified, behavior-preserving micro-optimization and remains staged. Its end-to-end effect should be modest because both paths are graph-captured; final promotion depends on paired fixed and real-server measurements after larger MTP work is evaluated.

### 2026-08-16 13:33:04 PDT - fixed server reaches 171.263 tok/s; explicit FP8 exposed as inactive

- A single fixed-work server launched under parent PID **36020** with the native gate, seed 783025237, simulated accepted length 3, and an explicit `--speculative-draft-model-quantization fp8`. Capture completed with 1.77 GiB headroom and the server became ready normally.
- Warm fixed control: **171.339 tok/s**. Fresh five: `170.995, 171.291, 171.125, 171.541, 171.363`; mean **171.263 tok/s**. All retained the established digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`. Server-internal warm intervals were consistently about 172.6-173.6 tok/s. This is **+2.08%** over the previous safe 167.776 mean and nearly reaches the rejected unsafe-buffer singleton without its lifetime hazard.
- The run does **not** test FP8 draft weights yet. Although server args retain the explicit `fp8` value, draft load reports `Qwen3_5ForCausalLMMTP, quant=modelopt_mixed`; `ModelConfig._verify_quantization()` silently replaces explicit FP8 with checkpoint-wide ModelOpt mixed detection. Because the checkpoint excludes embedded `mtp*` weights from its packed target format, this is a real dispatch/configuration hole.
- The launcher now admits explicit `fp8` and `mxfp8` draft candidates. Next source repair must preserve an explicitly requested online draft quantizer only when the embedded MTP is excluded/unpacked, with focused config tests. The resulting load log must say FP8 before any throughput number is attributed to FP8.
- The 171.263 fixed result is presently attributable to the newly active native-gate topology plus run-state effects; causality remains to be separated with a narrow native-gate on/off A/B after the larger draft-quantization branch is made real.

### 2026-08-16 13:44:26 PDT - explicit target is 200 tok/s real; Windows FP8 draft path now executes

- The user set the concrete target: **200 tok/s real sampled** while retaining reasoning, tools, preserved thinking, disabled vision, and the exact 200K pool. The present ~171 tok/s all-accepted ceiling cannot meet that target, so topology and draft work per emitted token take priority over isolated sub-microsecond kernels.
- Ported dense online FP8/MXFP8 quantization registration to native Windows. Explicit draft quantization now survives mixed-checkpoint auto-detection only when `mtp*` is excluded/unpacked; the loader constructs a clean online FP8 config instead of parsing the packed target's ModelOpt metadata. Dense FP8 model construction also avoids importing the unavailable Windows `sgl_kernel` MoE extension. Focused config/loader routing: **7 passed + 12 subtests**; a direct dense-config smoke also passes.
- The bring-up found and repaired three independent dispatch blockers in sequence: Windows registry omission, target metadata missing FP8 `activation_scheme`, and unconditional `FusedMoE`/`sgl_kernel` import. Each failed server tree exited on its own; no orphan or competing CUDA process remained.
- The third launch reaches full readiness under parent PID **3368**. It loads the embedded MTP in 5.83 GiB versus 6.10 GiB for the BF16 control, leaves 2.13 GiB after graphs versus 1.77 GiB in the preceding control, and graph tracing enters FlashInfer `fp8_gemm_sm100`, proving active dense FP8 execution even though the generic load-summary string still reflects the shared checkpoint's `modelopt_mixed` metadata.
- Fixed FP8 warmup was cold at 163.785 tok/s. Five subsequent samples: `171.562, 172.563, 171.667, 171.946, 161.004`; median **171.667**, all digest-stable. The four uncontended samples average **171.935**, essentially flat/slightly above the native-gate control; online FP8 saves memory but its dynamic-quant overhead cancels the four converted small-M GEMMs.
- The remaining raw BF16 MTP `fc` is the largest unconverted draft projection (10240 -> 5120). Next change will construct it as an FP8-capable `ColumnParallelLinear` only for explicit FP8/MXFP8/GGUF draft configs, preserving the established raw BF16 baseline, and will then remeasure the all-accepted ceiling before spending a real-sampling window.

### 2026-08-16 13:48:48 PDT - full MTP FP8 loses fixed-work throughput

- Converted the 10240 -> 5120 MTP fusion projection to SGLang `ColumnParallelLinear` only for explicit online FP8/MXFP8 or GGUF configs; the production BF16/modelopt-mixed path remains the original `nn.Linear`. Added focused routing coverage; the exact Qwen MTP unit passes.
- Startup now states the resolved mechanism directly: `Qwen3.5 MTP effective quantization=fp8, fusion projection=ColumnParallelLinear`. Full-200K capture succeeded under parent PID **13724**, with 2.14 GiB reported graph-end headroom.
- Fixed warmup: 161.164 tok/s. Fresh five: `163.647, 167.825, 167.836, 168.026, 167.781`; mean **167.023**, warm-four mean **167.867**. Digest remained identical. This is about **2.0% below** the 171.263 native-gate/BF16 control; dynamic activation quantization plus the FP8 small-M GEMMs cost more than the large fusion projection saves.
- Do not promote full online FP8. The code remains isolated behind an explicit draft quantization option while MXFP8 is checked once. The production default stays checkpoint-native BF16 MTP. Reaching 200 real requires changing emitted-tokens-per-verification or eliminating larger target/draft work, not merely casting these five projections.

### 2026-08-16 13:51:34 PDT - post-compaction checkpoint; 200 real target retained

- The explicit promotion target is **200 tok/s real sampled** with reasoning and thinking preserved, tools enabled, vision/audio disabled, and the exact 200,000-token pool. The latest qualified production mean remains **122.712 tok/s real**; the current safe fixed-work record is **171.263 tok/s**. A topology/work-efficiency improvement is therefore mandatory even if dense draft kernels improve.
- The one isolated MXFP8 launch selected `Qwen3.5 MTP effective quantization=mxfp8, fusion projection=ColumnParallelLinear`, loaded the target, then exited cleanly during online weight postprocessing. Root error: `ModuleNotFoundError: No module named 'triton_kernels'`, surfaced as `RuntimeError: MXFP8 quantization requires triton_kernels with MXFP8 support.` No live server is expected after that self-terminating failure; exact process/GPU state must be verified before another workload.
- The native dependency already exposes FlashInfer `mxfp8_quantize` and `mm_mxfp8`; only online weight conversion is stranded on `triton_kernels`. Next candidate is a thin-dispatch repair that quantizes weight rows through FlashInfer's native CUDA path with the exact operand/scale layout expected by `mm_mxfp8`, retains the existing Triton implementation where available, and proves numerical/layout correctness in isolation before one server launch.
- Full online FP8 including the 10240 -> 5120 fusion projection remains rejected at **167.023 tok/s mean** versus the **171.263 tok/s** BF16 control. The production default remains BF16 MTP. After the single MXFP8 branch closes, work returns to accepted/emitted tokens per verification and fused sparse proposal/rejection paths, where the fixed-versus-real gap resides.

### 2026-08-16 14:03:02 PDT - native FlashInfer MXFP8 weight conversion qualifies

- Replaced the Blackwell CUDA online-weight dependency on optional `triton_kernels` with FlashInfer 0.6.17's existing native CUDA MXFP8 quantizer. It emits the canonical contiguous FP8 weight and reshapes its linear UE8M0 scale buffer to `[rows, K/32]`; the existing backend-specific post-load code still owns CUTLASS/CuTe/TRT-LLM swizzling. Non-Blackwell and non-CUDA callers retain the established Triton implementation.
- Added bit-exact tests against an independent torch UE8M0/E4M3 reference at `(3,64)`, the Qwen fusion width `(3,10240)`, and the padding boundary `(129,128)`. Added a real native quantize -> scale-interleave -> CUTLASS MXFP8 GEMM test against dequantized matmul.
- Result: **4 passed** in 448.44 s. The long first run was FlashInfer CUDA JIT: exactly one pytest tree, Ninja, and the configured maximum of two NVCC/CICC workers. It completed cleanly; no compiler or SGLang process remains. GPU after completion: 28 C, display-only 1,217 MiB, no compute owner. The eight visible Python processes are four exact parent/child Quasimorph MCP pairs, unrelated and CPU-side.
- The worktree was committed underneath the test run, as the user had warned could happen. The native production repair is present in `HEAD`; the new registered MXFP8 test was still untracked at this checkpoint. Before further edits, inspect the new commit and retain the test deliberately.

### 2026-08-16 14:06:18 PDT - MXFP8 second dispatch blocker repaired

- The first post-repair full server loaded the 27B target and the online MXFP8 MTP successfully: target load 19.25 GiB, draft load 5.80 GiB, and 2.56 GiB available before graph capture. Target verification graph capture completed in 38.21 s with 2.47 GiB left.
- Draft decode capture then failed at its first dynamic activation. SGLang's MXFP8 wrapper defaulted to FlashInfer CuTe DSL and auto-resolution selected `flashinfer_cutedsl` on SM120, while the optional `nvidia-cutlass-dsl` package is absent (`ModuleNotFoundError: cutlass`). The exact server tree self-terminated; post-failure GPU state returned to 28 C, 0% utilization, 1,217 MiB display residency, with no SGLang/compiler owner.
- The RTX 5090 launcher now explicitly resolves `--fp8-gemm-backend flashinfer_cutlass`. FlashInfer MXFP8 activation quantization defaults to its stable native CUDA backend and switches to CuTe DSL only when the dense GEMM backend is explicitly CuTe DSL. This keeps the production path native CUDA/CUTLASS and leaves the optional dependency as a later controlled benchmark branch.
- Extended the registered test through the exact dynamic-activation -> CUTLASS call that failed during capture. Result after cached native JIT: **4 passed** in 10.05 s. The next and only GPU workload is one relaunch of the same fixed 200K MXFP8 topology.

### 2026-08-16 14:10:40 PDT - native MXFP8 works end to end and is rejected on throughput

- The repaired native CUDA/CUTLASS MXFP8 server reached full readiness at the exact 200,000-token pool. Target load remained 19.25 GiB; draft load fell to 5.59 GiB; target verify, draft decode, and draft extend graphs all captured, leaving 2.14 GiB at pool initialization. FlashInfer autotuning exercised the real `mxfp8_gemm` shapes.
- Fixed-work warm result: **161.621 tok/s**. Five fresh exact `6213/512` samples: `162.788, 162.959, 169.168, 162.203, 160.169`; mean **163.457 tok/s**. Every sample returned 512 tokens with the established deterministic SHA-256 `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`.
- This is **4.56% below** the 171.263 tok/s BF16 fixed control. Full MTP MXFP8 is therefore a valid optional memory-saving mechanism, but it is rejected for the production performance topology. A sampled window cannot repair a lower all-accepted ceiling sufficiently to advance the explicit 200 tok/s real target.
- Retain the native FlashInfer weight-conversion repair, backend-explicit activation dispatch, launcher backend control, and registered test. Restore BF16 MTP for performance work. Next priority is reducing verification/draft work per emitted sampled token or increasing useful accepted depth; the present 122.712 real versus 171.263 fixed gap remains the dominant opportunity.

### 2026-08-16 14:18:40 PDT - post-compaction checkpoint; topology search widened

- The target remains **200 tok/s real sampled** with exact 200,000-token capacity, reasoning and preserved thinking, tools enabled, and vision/audio disabled. The 171.263 tok/s fixed result describes only the current BF16 two-step topology; it is a benchmark boundary to break, not a physical ceiling.
- The active branch extends `nvfp4_online` from MoE-only conversion to dense MTP linears. It adds a dense `ModelOptNvFp4OnlineLinearMethod`, routes the Qwen3.5 MTP 10240 -> 5120 fusion projection through `ColumnParallelLinear`, and selects FlashInfer's stable native CUDA quantizer. These edits are unqualified and must pass syntax, focused routing, isolated CUDA numerical/graph tests, and then one exact-process full-200K fixed launch.
- Dense online NVFP4 is being tested because it can reduce every recursive draft step enough to reopen deeper speculative chains. If it qualifies, speculative depth, graph buckets, and proposal alignment must be retuned together; old two-step rankings no longer describe the new cost topology.
- Larger routes remain open behind it: native fused sparse top-k20 target/draft rejection over the <=40-token support union, fused target vocabulary projection plus top-k selection, shape-specific FP4 GEMM dispatch, and deeper or branched speculation. Promotion remains governed by real sampled throughput plus acceptance, exact target-distribution semantics, reasoning/tool behavior, 200K exactness, VRAM, thermal, and process cleanliness.
- No CUDA workload should be presumed live after rollover. Before any compilation or GPU test, re-establish the exact process tree and GPU owner state; run only one CUDA/compiler/server workload at a time.

### 2026-08-16 14:22:52 PDT - dense NVFP4 CPU contract passes; CUDA waits for user build

- The dense online NVFP4 implementation compiles cleanly. Focused CPU coverage passes: **7/7** ModelOpt/NVFP4 tests plus the exact Qwen3.5 MTP routing test. Coverage now proves `nvfp4_online` chooses the dense conversion method, allocates a source-shaped BF16 parameter compatible with ordinary loaders, and routes the MTP fusion projection through `ColumnParallelLinear`.
- Added an isolated SM100+ CUTLASS test for floating-source load -> packed NVFP4 postprocessing -> M=1/M=3 numerical comparison -> CUDA graph capture/replay. It has not run yet.
- A separate user-owned Rust extension build began under exact tree `23868 -> 25468 -> 37628 -> 23776` (`setup.py build_rust --inplace` -> `cargo rustc`). The GPU is idle at 28 C / 0%, but the one-workload safety contract applies to compilers too. The build is left untouched and no CUDA test or server will start until the tree finishes and process/GPU cleanliness is reverified.

### 2026-08-16 14:27:13 PDT - dense online NVFP4 qualifies in isolation

- Found and closed two dispatch/contract gaps before the GPU run. The explicit draft config was still inheriting the target checkpoint's `mtp*` exclusion, which would have silently left every draft linear in BF16; `get_quant_config()` now constructs a clean online config for explicit draft `nvfp4_online`, as it already did for FP8/MXFP8. The config also now supplies the non-AWQ marker required by the reused dense ModelOpt FP4 apply path.
- Updated focused config, source-shaped loader, and MTP fusion-routing coverage passes **3/3**. The clean explicit config retains the packed-module mapping, dynamic activation contract, no inherited target exclusions, and non-serialized source state.
- The isolated SM120/CUTLASS dense test passes floating BF16 load -> native CUDA NVFP4 weight conversion -> ModelOpt packing/interleaving -> M=1 and M=3 execution -> CUDA graph capture and repeated replay. Both eager and graphed outputs remain within relative-MAE 0.15 of the BF16 source matmul; consecutive graph replays are bit-identical.
- The initial graph assertion compared capture-time storage before the first explicit replay and failed; the production-relevant replay/replay contract plus BF16 numerical reference passes. No correctness exception was weakened: the final test checks both deterministic replay and numerical fidelity.
- The user paused the separate Rust test stream. Before the CUDA test, its process tree was confirmed gone; the test ran as the only SGLang/CUDA process. Next gate is one exact full-200K fixed-work server using explicit dense `nvfp4_online`, with startup mechanism, draft load memory, graph coverage, digest, thermals, and exact tree monitored before any real-sampling run.

### 2026-08-16 14:31:19 PDT - dense NVFP4 is real, exact, and slower

- One fixed-work full-200K server launched with explicit `nvfp4_online`. Startup resolves `Qwen3.5 MTP effective quantization=nvfp4_online, fusion projection=ColumnParallelLinear`; draft load memory is **5.64 GiB** versus roughly 6.10 GiB BF16. Target verify, draft decode, and draft extend graphs all captured; graph-end available memory was **2.36 GiB** and WDDM free memory after readiness was about 1.97 GiB.
- Target graph compilation re-exercised cached/fallback Triton candidates and logged multiple Windows LLVM `Unsupported rounding mode for conversion` / loop-carried fp64 failures. These were rejected compile candidates rather than serving failures: target capture finished in 42.01 s, both draft graphs captured, `/health` returned 200, and execution remained stable.
- Fixed warmup: **165.254 tok/s**. Fresh five: `164.693, 161.478, 164.831, 164.902, 164.566`; mean **164.094 tok/s**. Every sample produced 512 tokens with the established exact digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`.
- Dense online NVFP4 is **4.19% below** the 171.263 tok/s BF16 fixed record. It is rejected for the performance topology and receives no sampled window. Its ~0.46 GiB draft-memory saving is retained as an optional capacity mechanism, but deeper draft recursion would multiply its slower per-step cost unless BF16 graph residency becomes the only blocker.
- Stopped only exact CUDA-worker leaf PID 15224; the expected server tree cascaded out. Port 30000 is free, all seven known tree PIDs are absent, and the GPU returned to display-only 1,527 MiB. The next performance branch returns to BF16 MTP and changes accepted/emitted tokens per target verification rather than draft weight dtype.

### 2026-08-16 14:35:23 PDT - three-step full acceptance breaks 200 tok/s

- Reopened the current aligned-q topology at three MTP steps / four draft tokens and forced accepted length 4. This is a different work-efficiency measurement from the historical three-step control forced to emit only three tokens. The full exact-200K BF16 server captured four-token target verify, draft decode, and draft extend graphs; reported graph-end headroom was **1.72 GiB**.
- Fixed warmup: **197.414 tok/s**. Fresh five: `194.466, 197.795, 197.314, 201.251, 183.687`; all retained the exact established digest. Conservative five-run mean is **194.903 tok/s**; the uncontended first-four mean is **197.707 tok/s**. A sixth contended sample was 188.531.
- The fourth fresh sample is the first externally measured **201.251 tok/s** result. Server-internal steady windows reached 204.40 and 207.60 tok/s. Physics is duly offended: the present kernels can exceed 200 when four useful tokens are emitted per verification.
- The late two samples coincided with lower prompt rates and WDDM residency rising to 31,826 MiB used / only 362 MiB free after repeated runs. Treat them as a real capacity/contention warning, not grounds to discard the topology. A production form needs more residency margin or narrower graph/state allocation before long-run promotion.
- This remains simulation-only. It proves compute feasibility and moves the binding problem to sampled fourth-token acceptance plus memory. Exact CUDA-worker leaf PID 24404 was stopped; the complete server tree cascaded out, port 30000 is free, and the GPU returned to display-only 1,387 MiB. Next: measure unsimulated three-step acceptance under aligned q, then build a confidence/cost-aware 2/3-step policy or improve the fourth proposal rather than statically paying for it everywhere.

### 2026-08-16 14:39:25 PDT - honest third step loses statically; fourth-token signal isolated

- The unsimulated aligned-q three-step/four-token server captured all exact-200K graphs with 1.64 GiB reported headroom and passed readiness. Sampled-profile warm decode was **129.591 tok/s**. Five fresh runs were `111.626, 118.145, 123.111, 115.037, 118.276`; mean **117.239 tok/s**, below the qualified static-two-step mean 122.712.
- A matched native acceptance probe measured emitted/accepted length **2.403756**, acceptance rate **0.469484**, `300/639` correct/proposed drafts, 213 target verifications, and histogram `[65,55,34,59]`. The third draft was accepted on **59/213 = 27.70%** of cycles. It raises useful tokens per verification above the two-step mean 2.318, but static execution pays its cost on every cycle.
- This closes static three-step under the current proposal distribution while identifying a concrete control target. A perfect oracle that runs the third step only on those 27.7% successful cycles would retain most of the extra emitted tokens with a fraction of the cost. The next branch should classify that event from already-device-resident draft-q confidence (top-1 probability, margin, entropy, penalty status, recent rejection) and choose a separately captured 3- or 4-token verification graph without a host synchronization.
- Exact CUDA-worker leaf PID 32388 was stopped and the server tree cascaded out. Port 30000 is free; GPU state returned to 1,294 MiB display residency with no SGLang/compiler process. Any adaptive experiment must first audit the existing policy and graph selection so the earlier history-only adaptive rejection is not mistaken for this confidence/cost policy.

### 2026-08-16 14:46:10 PDT - pause checkpoint; draft-q top-k 8 loses its first gate

- The latest exact-200K server returned to the qualified static two-step BF16 topology while changing only draft proposal top-k from 20 to 8. Its acceptance probe measured emitted/accepted length **2.235808**, acceptance rate **0.620087**, `284/458` correct/proposed drafts, 229 target verifications, and histogram `[63,48,118]`. This is worse than the qualified top-k-20 accepted-length mean **2.318174**.
- Three real sampled runs were `128.028, 115.179, 116.016`; mean **119.741 tok/s**. The isolated peak does not offset the lower acceptance or the mean below the qualified **122.712 tok/s** production result. Treat top-k 8 as rejected under the current proposal distribution.
- At the requested pause, no benchmark was running, but the server remained resident and idle: exact launch parent PID 12628, CUDA worker PID 24872, listener PID 29392 on `127.0.0.1:30000`; GPU utilization 0%, 30 C, and 31,958 MiB WDDM residency. Stop only the exact CUDA-worker leaf and verify the expected tree cascade before any further compiler or CUDA workload.
- The compute-feasibility result remains intact: three-step/full-accept fixed work reached **201.251 tok/s externally** and 207.60 tok/s in a server window. The active optimization target is therefore useful sampled work per verification, led by graph-safe native sparse rejection / fused top-k and a device-side confidence-gated third proposal. No new workload starts while paused.
- Cleanup completed at 14:47 PDT by stopping only CUDA-worker leaf PID 24872. The full known server tree cascaded out, port 30000 is free, and the GPU returned to display-only residency: 1,160 MiB, 0% utilization, 28 C. The machine is clean and the work remains paused.

### 2026-08-16 14:51:00 PDT - resumed; refreshed sampled-path cost boundary

- Resumed from a clean GPU and reran the exact production-shape sampling probe at 3 target rows, 2 draft rows, vocabulary 248,320, target/draft top-k 20, and top-p 0.95. Eleven repeated medians: FlashInfer dense target softmax+renorm **191.722 us**, each FlashInfer dense aligned draft proposal **193.532 us**, and dense Triton Leviathan rejection **76.699 us**. The current total is approximately **655.5 us per two-step verification cycle**.
- Existing FlashInfer sparse top-k target construction is slower at **246.772 us**; its sparse draft construction is **343.307 us**. A fused native sparse implementation remains valid engineering work because it can remove scatter/dense residual traffic, but erasing the entire current sampling stack would save only about **3.7%** of the measured 17.5 ms fixed-work cycle. It cannot by itself close 122.712 -> 200 real tok/s.
- Priority therefore moves to proposal quality and conditional topology while retaining native sparse sampling as a parallelizable later cut: measure per-depth `sum(min(p,q))`, support mismatch, confidence, and third-proposal success; use that evidence to calibrate q and select the already-proven 4-slot topology only on profitable cycles. The three-step full-accept path has already demonstrated 201.251 external tok/s, so selection/acceptance is the binding route.

### 2026-08-16 14:56:12 PDT - user-directed Gittensor RTX 5090 checkpoint branch

- User supplied `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`. Hub dry-run succeeds publicly and reports three safetensor shards totaling approximately **20.6 GB**. Small metadata is staged at `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RTX5090`; the GPU/server/toolchain state was clean before inspection.
- This is a distinct NVIDIA ModelOpt NVFP4 W4A4/group-16 export with FP8 KV and an advertised approximately **18.8 GB** runtime weight footprint. The card targets the RTX 5090 and reports 80.6 tok/s non-thinking/non-speculative vLLM concurrency-one decode plus native 262,144 context. Those figures are informative only; they are not comparable to this project's thinking-on, presence-penalized, real-sampled MTP contract.
- The checkpoint does contain the bundled BF16 MTP tensors (`mtp.fc`, layer 0 attention/MLP/norm tensors) and config `mtp_num_hidden_layers=1`; its quantization exclusions deliberately retain MTP, embeddings, lm_head, vision, and Gated-DeltaNet input/conv projections in BF16. It is therefore eligible for the exact same SGLang speculative topology rather than a non-MTP fallback.
- The model card claims a 45/60 smoke tie with Unsloth, coherent tools, and a later higher-accuracy/faster weight update. Treat quality as unqualified until direct preserved-thinking, math, repetition, and tool probes pass locally. Next: complete the three-shard download with two workers, verify checksums/config layout, then launch one exact-200K fixed and sampled A/B against RadixArk.

### 2026-08-16 15:04:28 PDT - Gittensor download verified; pure ModelOpt FP4 Windows gate opened

- The complete checkpoint downloaded successfully with two workers. Hub verification checked all **25 remote files** and passed hashes/content; its only strict-extra complaint was the expected local `.cache/huggingface` download metadata. Shards are 9,972,777,720 + 9,875,839,416 + 744,532,384 bytes.
- First exact launcher attempt exited before GPU allocation with `Unknown quantization method: modelopt_fp4`. This is a Windows registry omission, not an unsupported tensor layout: the fork already carries the Windows-safe `ModelOptFp4Config`, native FlashInfer FP4 GEMM path, per-tensor activation-scale loader, exclusion handling, and the same packed `weight` / `input_scale` / `weight_scale` / `weight_scale_2` families present in this checkpoint.
- Registered `ModelOptFp4Config` as `modelopt_fp4` in the deliberately narrowed Windows quantization registry and imported it from the already-loaded ModelOpt module. Added a Windows registry contract test. Focused config suite passes **9 tests / 12 subtests**; both files compile; direct local `ModelConfig` construction now resolves `modelopt_fp4` and `Qwen3_5ForConditionalGeneration`.
- The failed parent PID 27840 and child exited naturally; no server/compiler process or port listener remained and GPU residency returned to desktop-only. Next gate is a second fixed-work launch, where actual packed-weight loading, FP4 GEMM execution, embedded BF16 MTP routing, full 200K allocation, and graph capture must all qualify before throughput is measured.

### 2026-08-16 15:08:08 PDT - Gittensor loads at 200K; fixed work is slower

- The second launch passed every mechanism gate. Target resolved as `modelopt_fp4/NVFP4`, loaded in 11.87 s, and used **17.91 GiB**, 1.34 GiB below RadixArk's approximately 19.25 GiB. Embedded MTP correctly stayed BF16, loaded in 0.49 s, and used **6.40 GiB**. Exact 200,000 target and draft pools allocated; target verify, draft decode, and draft extend graphs all captured; final reported graph headroom was **2.92 GiB**.
- Fixed two-step/full-accept warm result was **162.097 tok/s**. Five fresh exact `6213/512` results: `152.303, 152.279, 158.916, 158.185, 152.733`; mean **154.883 tok/s**. All retained the established 512-token digest `9d850fbf7217c585190b3eff9003bf2223907f0d4b59c5b11ddbaf56bc70af9c`.
- This is **9.57% below** the RadixArk BF16-MTP fixed record of 171.263 tok/s. The likely direct cost is this recipe's deliberately BF16 `lm_head` and broader BF16 exclusion set; RadixArk's index carries packed/scaled lm_head tensors. Gittensor's smaller residency is real, while its SGLang single-stream fixed execution is currently slower.
- Do not reject the checkpoint before measuring proposal quality: a materially better target/draft overlap could still repay the fixed tax. Stop the simulation server by exact CUDA-worker leaf PID 4716, relaunch unsimulated with the identical two-step/top-k20 contract, then measure acceptance and real sampled throughput before semantic qualification.

### 2026-08-16 15:11:30 PDT - stock Gittensor real decode loses narrowly; hybrid lm_head route opened

- Unsimulated exact two-step/top-k20 acceptance probe: emitted length **2.403756**, acceptance rate **0.699531**, `298/426` correct/proposed drafts, 213 verifications, histogram `[47,34,132]`. This is a real work-efficiency improvement over RadixArk's accepted-length mean 2.318174.
- Ten fresh real sampled results: `119.700, 112.690, 129.119, 119.256, 120.416, 119.472, 117.992, 121.861, 112.087, 118.328`; mean **119.092 tok/s**, median **119.364 tok/s**. Stock Gittensor is **2.95% below** RadixArk's qualified 122.712 mean despite better acceptance. Its observed prompt rates were mostly 11.7-12.3K tok/s, a promising prefill signal.
- The loss is small enough that this checkpoint exposes a stronger hybrid opportunity. Gittensor deliberately leaves the 248,320 x 5,120 `lm_head` and embedding in BF16, while RadixArk carries packed/scaled NVFP4 lm_head tensors and reaches a 10.6% higher fixed ceiling. Selectively quantizing or substituting only Gittensor's lm_head can plausibly recover the decode gap while retaining its smaller target, larger headroom, better proposal overlap, and faster prefill.
- Next: stop the exact unsimulated tree, inspect Gittensor and RadixArk lm_head tensor shapes/layout/scales and base-weight compatibility, then implement a load-time selective ModelOpt NVFP4 lm_head path or a reproducible hybrid checkpoint. Require fixed speed, real acceptance, preserved-thinking quality, and checksum/provenance evidence; do not mutate either downloaded source checkpoint in place.

### 2026-08-16 15:12:48 PDT - post-compaction continuation checkpoint

- Resumed from the completed stock Gittensor evaluation. Last verified runtime state was clean: no SGLang/server/compiler process, port 30000 free, and the RTX 5090 at display-only residency and idle temperature.
- Stock Gittensor remains a useful component rather than the production winner: **119.092 tok/s** real mean versus RadixArk **122.712**, while improving accepted length to **2.403756**, reducing target residency to **17.91 GiB**, leaving about **2.92 GiB** after graph capture, and showing substantially faster short-prompt ingestion.
- Immediate branch: inspect the two checkpoints' exact `lm_head` tensor layouts and provenance, then qualify an explicit opt-in selective online NVFP4 target-head path. Keep both source checkpoints immutable; validate the isolated CUDA conversion/graph path before starting exactly one full 200K server.

### 2026-08-16 15:18:35 PDT - RadixArk restored as active checkpoint

- User ended the Gittensor hybrid branch and directed the work back to RadixArk. No runtime teardown was required: the GPU is clean at **0%**, **28 C**, **1,406 MiB** display-only residency; the only permission-hidden NVIDIA entry is confirmed `dwm.exe`.
- The qualified Windows launcher already defaults to `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk`, so no launcher edit or source-checkpoint mutation was needed. RadixArk remains the production reference at **122.712 tok/s real sampled mean** and **171.263 tok/s fixed two-step mean**.
- Retain the Gittensor Windows `modelopt_fp4` registry port and its measurements as supporting evidence. Resume optimization against RadixArk's measured bottleneck: improve proposal/work efficiency and conditional speculative topology while preserving the exact 200K, thinking, tool, and quality contract.

### 2026-08-16 15:22:21 PDT - topology pivot: exact root-heavy GPU tree

- User set the next architecture directly: replace linear speculation with an **exact, GPU-resident, root-heavy tree verifier**, preserve Qwen's recurrent state, and raise emitted output above four tokens per target traversal. This supersedes the confidence-gated linear two/three-step branch.
- The requirement matches the measured boundary. Static two-step tops out at **171.263 tok/s** even when every three-token traversal is useful; static three-step only grazes 200 under forced four-token acceptance and loses honestly. A wider/deeper tree is the remaining topology with enough useful-output headroom.
- Immediate audit: existing general EAGLE tree construction and target-only verifier semantics, stochastic exactness under branching, CUDA-graph/static-shape constraints, and Qwen Gated-DeltaNet/ReplaySSM state commit. Production implementation remains C++/CUDA with Python limited to binding and dispatch.

### 2026-08-16 15:35:45 PDT - native exact tree-sampling primitive passes

- The shipped non-chain algorithm contains an exact target-only construction: at each prefix, proposal siblings partition their true target probability mass on one uniform; a miss samples from target `p` with only that sibling set removed. Candidate generation may be deterministic and no proposal `q` enters the correction.
- Added a native Windows C++/CUDA JIT verifier for trees up to 32 nodes. It traverses entirely on GPU, retains only terminal rejected token IDs, and performs a vectorized two-pass residual CDF without allocating or scanning a dense draft-probability tensor. The existing non-Windows `sgl_kernel` path is unchanged.
- CUDA qualification passes **3/3**: a two-level accepted branch, a terminal sibling-residual case, and 32,768 parallel trials whose first emitted-token distribution matches target `p` within 0.012 absolute tolerance. Compile completed as the sole CUDA workload.
- Native Windows target-only EAGLE now dispatches to this verifier; module preload occurs before model weights for top-k greater than one. Backend validation was narrowed so XQA may remain ordinary target decode while tree verification uses the FlashInfer prefill backend and draft tree attention uses a tree-capable backend.
- Remaining blocker is the important one: current ReplaySSM fold asserts linear ancestry. Implement a tree-aware low-rank GDN verify that reads the fp32 checkpoint once, represents branch states as ancestor rank updates, writes raw per-node commit records, and folds only the accepted `accept_index` path. This avoids the otherwise prohibitive full recurrent-state snapshot per tree node.

### 2026-08-16 15:48:06 PDT - post-compaction root-heavy tree checkpoint

- The active direction remains the user-specified **exact, GPU-resident, root-heavy tree verifier** for RadixArk, with recurrent-state preservation and a promotion requirement above four emitted tokens per target traversal. Linear speculation is no longer the target architecture.
- Native exact target-only tree sampling is isolated and qualified at **3/3 tests**. Native low-rank GDN tree verification plus accepted-path recurrent-state commit is isolated and qualified at **2/2 tests**, including persistent fp32 state and track-slot behavior.
- A final four-file integration patch was in flight when context compacted and its tool output was truncated. Treat `gdn_backend.py`, `gdn_tree_replay.py`, `spec_utils.py`, and `eagle_worker_v2.py` as unverified until each intended hunk is inspected and repaired independently.
- Runtime remains intentionally stopped. Before any server launch: compile the modified Python, run focused dispatch/config tests, rerun both CUDA primitive suites serially, add CUDA-graph replay coverage for the GDN tree path, and confirm a clean GPU/process state.

### 2026-08-16 15:55:34 PDT - tree recurrent integration survives CUDA-graph replay

- Verified all four previously uncertain integration hunks landed: tree target verify dispatches into the native low-rank GDN kernel, accepted-path commit is wired through `commit_mamba_states_after_verify`, convolution rollback follows the selected tree node, and both native modules preload before model allocation.
- Found and repaired a production-only dtype mismatch hidden by the original isolated test: prefix-cache track destinations are int64, so the native all-layer commit now reads int64 track slots while keeping state/node/count inputs int32. The no-track path remains allocation-free and does not inspect its dummy pointer.
- Added a graph-replay test that captures fixed kernel pointers, mutates q/k/v/gates, recurrent slot, and tree ancestry, replays, compares against materialized fp32 recurrence, proves verify leaves the persistent checkpoint untouched, then commits only root -> node 2 -> node 5.
- One serial native-Windows CUDA process now passes the complete primitive gate: **6/6 tests** (three exact tree-sampling tests plus verify, accepted-path commit/track, and CUDA-graph replay/commit). The first invocation used the wrong global Python/MSVC environment and failed before compilation; rerunning through `scripts/windows/invoke_cuda_pytest.ps1` loaded the intended toolchain and passed.

### 2026-08-16 16:00:14 PDT - first live root-tree capture reached native GDN

- A single 16K-capacity RadixArk smoke server resolved the intended topology: top-k 4, four proposal steps, 12 target-verify nodes, rejection sampling off, FlashInfer target verification through speculative prefill mode, Triton draft tree attention, and ReplaySSM fold enabled.
- Target graph capture reached the new native GDN tree kernel on the real model, proving the backend and forward-mode routing. It stopped safely at a strict tensor contract: production `dt_bias` is bf16 while the first isolated fixture had supplied fp32. No inference ran and the process tree exited naturally; port 30000 and compiler/server processes are clean.
- Split `A_log` and `dt_bias` validation and changed the CUDA bias load to bf16-to-fp32, matching Qwen's actual parameter dtypes and the existing Triton gating semantics. Both affected fixtures now use production bf16 bias. The complete serial CUDA gate passes **6/6** again after recompilation.

### 2026-08-16 16:07:05 PDT - root-heavy tree runs end to end

- Repaired the remaining production tensor-layout mismatch: Qwen's q/k/v/a/b and output are padded split views with distinct token strides. The native GDN verifier now accepts explicit strides for every input and output, including pair-dot/raw-k and main replay, without packing or copying. The first fixture deliberately uses different padded strides to guard the production contract.
- The serial native-Windows CUDA gate passes **6/6** after the stride repair. A single 16K RadixArk smoke server then captured all intended shapes successfully: target verify `N=12`, draft decode `M=4`, and draft extend `N=12`; about **7.93 GiB** remained after small-context graph capture.
- A real reasoning request completed 64 tokens through repeated tree verification and accepted-path recurrent-state commits and correctly reasoned that `37 * 19 = 703`; the final answer was only truncated by the requested 64-token cap.
- A 512-token acceptance probe measured **3.160494 emitted tokens per target traversal** (`512/162`), **0.196409** node acceptance, and histogram `[21, 35, 43, 23, 40]`. Root coverage is already strong: first-level acceptance is **87.0%**, while the four-draft path completes on **24.7%** of traversals.
- Matching warm smoke decode measured **105.683 tok/s** (`93.083 tok/s` end-to-end including TTFT). This is mechanism qualification, not a promotion result: it used a 16K pool and disabled model torch compile. The 12-node tree raises useful output about **36%** over the 2.318 linear reference but remains below the required **greater than four**.
- Next topology experiment should allocate budget deeper before widening the root: start with top-k 4, six proposal steps, and 16-20 verify nodes, then measure acceptance histogram, emitted tokens per traversal, target traversal time, and real sampled decode. Do not use the existing linear `SimulateAcceptedLength`; its contiguous indices are not valid tree ancestry and would corrupt recurrent-state commit.

### 2026-08-16 16:11:58 PDT - depth alone does not raise tree yield

- Cleanly stopped the 12-node server and verified port 30000 free, no SGLang/CUDA compiler process, and the RTX 5090 back at display-only residency. Two remaining Python processes belong to the user's Quasimorph MCP parent/child pair and were left untouched.
- Launched exactly one deeper root-tree candidate at 16K: top-k 4, six proposal steps, 16 target-verify nodes, rejection sampling off, FlashInfer target verify, Triton draft tree attention, and torch model compile disabled. All three graphs captured successfully (`target N=16`, `draft M=4`, `extend N=16`) with **7.89 GiB** free.
- Five exact 6213/512 probes measured emitted tokens per target traversal `2.8927, 3.2611, 3.0476, 3.3684, 2.9425`; mean **3.1025**. Aggregated histogram is `[129, 209, 221, 121, 64, 36, 48]` over 828 traversals. Survival through draft depths 1-6 is approximately `84.4%, 59.2%, 32.5%, 17.9%, 10.1%, 5.8%`.
- The extra depth therefore fails the greater-than-four gate and spends two additional four-token draft passes for little tail yield. A matching streaming sample measured only **87.589 tok/s** decode, substantially below both the 12-node smoke sample and the qualified linear reference.
- Source inspection exposed an actionable allocation mismatch: target-only tree construction currently ranks every draft level from plain full-vocabulary softmax. It ignores the request's temperature, additive presence/frequency penalties, logit bias, top-k, and top-p even though the exact target verifier samples after those transformations. Exactness is preserved, but the 15-node budget is assigned using the wrong distribution. Align tree scoring with the target transformations first; then design a tapered root-heavy frontier that avoids paying width four at every deep step.

### 2026-08-16 16:20:15 PDT - Sequoia audit folded into the live tree branch

- Reviewed the supplied Sequoia audit against current code and the primary paper. Its former implementation blockers are now materially closed here: native exact target-only tree sampling, low-rank tree-aware GDN verification, accepted-path recurrent/conv commit, and one captured target traversal all run. The remaining useful directions are target-aligned candidate allocation, branch-local penalty semantics, a measured width/yield oracle, and a hardware-aware irregular topology.
- The current exact verifier is Sequoia's documented top-k target-sampling baseline: it samples directly from target `p` and descends when that token is among the proposed children, so tree generation does not affect the output distribution. Sequoia's without-replacement `p/q` verifier remains a later acceptance upgrade, especially if aligned deterministic top-k stalls below four emitted tokens.
- Implemented opt-in target-aligned scoring for target-only trees by sharing the sparse draft distribution builder: committed-prefix additive penalties and logit bias, temperature, fixed top-k, and top-p now define cumulative candidate scores. Native Windows keeps the transforms on GPU through FlashInfer renormalization; Python only dispatches captured CUDA operations.
- Broadened graph replay sampling-buffer refresh and launcher/hook semantics so `-SpeculativeDraftSamplingTopK 20` applies to rejection chains and target-only trees. Legacy behavior remains available when the option is absent. Added focused CPU coverage; proposal/scoring tests pass **4/4**, and the broader focused suite passes **144 tests plus 15 subtests** with only the same two unrelated Windows fixture failures (IPC path assumption and open NamedTemporaryFile).
- Next isolated A/B is the original top-k4 / four-step / 12-node smoke topology with aligned scoring enabled. Measure five acceptance windows and real sampled decode before changing tree shape.

### 2026-08-16 16:22:52 PDT - fully normalized tree scores allocate too deep

- The aligned 12-node graph captured and served correctly, including dynamic replay updates for temperature, top-p, and committed-prefix additive penalties. Graph-end headroom was **7.91 GiB** at 16K; aligned draft decode consumed about 0.03 GiB more graph memory than the plain scorer.
- Five 6213/512 acceptance probes measured emitted lengths `3.1030, 2.7676, 2.6806, 2.8927, 2.8132`; unweighted mean **2.8514**. Aggregated histogram `[189, 219, 196, 134, 162]` over 900 traversals gives **2.8456** emitted/traversal, 79.0% first-depth survival, and 18.0% full four-draft survival.
- One matching streaming sample measured **105.722 tok/s**, essentially equal to the earlier single plain-scoring smoke result despite worse useful work. Full top-k20/top-p renormalization appears to make cumulative draft path scores overconfident, shifting the fixed 11-node budget deeper and away from the root-heavy coverage this topology needs.
- Keep the implementation opt-in while measuring a five-window plain-softmax control on the identical server shape (`-SpeculativeDraftSamplingTopK 0`). If confirmed, split token ranking from path calibration: apply target penalties/bias for candidate identity while retain a learned depth/branch discount or the original untruncated probability mass for root-heavy allocation.

### 2026-08-16 16:26:26 PDT - plain 12-node control confirms aligned scorer loses yield

- Repeated the identical 16K, 12-node, four-step tree topology with the original plain scorer (`-SpeculativeDraftSamplingTopK 0`) for five 6213/512 acceptance probes. Emitted lengths were `2.8603, 2.9595, 2.8287, 2.9595, 3.1411`; unweighted mean **2.9498**.
- Aggregated histogram `[153, 235, 169, 132, 180]` over 869 traversals gives **2.9436 emitted/traversal**, **82.39%** first-depth survival, and **20.71%** full four-draft survival. The fully normalized aligned scorer produced 2.8456 emitted/traversal, so alignment reduced useful yield by about **3.45%** on this topology.
- One plain streaming sample measured **104.145 tok/s** (91.203 tok/s end-to-end, 8784.653 prompt tok/s). The single aligned/plain throughput samples are too noisy to distinguish, while the multi-window acceptance result rejects full target top-k20/top-p normalization as the default tree scorer.
- Preserve aligned scoring as an explicit experiment only. Restore plain tree scoring as the launcher default, then move to a target-width curve and a topology/yield oracle. Depth-only expansion already lost; the next candidate must reallocate nodes toward root breadth or use exact sampling without replacement.
- The test server was stopped cleanly after the run. No benchmark/server workload was intentionally left active.

### 2026-08-16 16:31:33 PDT - width-curve tooling ready

- Restored plain tree scoring as the launcher default. `-SpeculativeAlignTreeScoring` now explicitly gates passing draft top-k into a non-rejection tree run; rejection sampling retains its qualified top-k20 default. PowerShell parsing and `git diff --check` pass.
- Extended `scripts/windows/analyze_torch_trace.py` to group contiguous CUDA-graph kernels by graph ID and report replay wall spans. Against the qualified older trace it recovers 245 target replays at **15.516 ms mean / 14.905 ms median** and 245 recursive-draft replays at **2.489 ms mean / 2.319 ms median**, matching the independent audit.
- Added `scripts/windows/bench_target_verify_width.py`: it warms one already-running server, profiles a single local real-sampling request with GPU activity only, records acceptance beside the trace path, and never launches another server. This is the safe one-process-at-a-time harness for the M-width curve.
- GPU inspection shows only ordinary WDDM display processes with N/A accounting; port 30000 has no listener (only expired TIME_WAIT entries). The next action is one 12-node server, one short profile, clean stop, then selected smaller/larger widths only if the measured curve warrants them.

### 2026-08-16 16:33:59 PDT - M12 cost curve exposes the full-tree tax

- Profiled the plain-scoring 12-node/four-step tree at 16K with one 6213/128 real-sampling request. The request produced **2.9091 emitted/traversal** over 44 verifications, histogram `[10, 9, 10, 6, 9]`. Trace: `benchmark/windows/profiles/target_width_m12-20260816-163326/target_width_m12-1786923206.8860037-TP-0.trace.json.gz`.
- CUDA-graph replay spans are: target verify graph ID 2 **19.081 ms mean / 19.124 median** (45 replays), draft decode graph ID 5 **3.744 / 3.603 ms**, and draft extend graph ID 8 **1.606 / 1.605 ms**. The captured spans sum to **24.431 ms/cycle**. The stale audit's predicted 18-20 ms M12 target cost is confirmed, while its simplified economic model omitted roughly 5.35 ms of captured tree construction.
- With at most five emitted tokens, the graph-only perfect-accept bound at this shape is about **204.7 tok/s**. The earlier unprofiled plain run implies about **28.26 ms per real cycle** (`2.9436 / 104.145`), which would require 5.65 emitted tokens/cycle for 200 and therefore cannot reach the target at this topology. The root-heavy path must reduce tree-construction/eager cost, add useful depth, or both; improving breadth/yield alone is insufficient at current cycle time.
- The first harness attempt exposed that `/start_profile` and `/stop_profile` intentionally return plain text. It started no inference, was stopped immediately, and left a harmless empty profiling directory. The client now handles those endpoints as text; the successful trace flushed cleanly. Server session 76811 was then stopped by Ctrl+C. Port 30000 has no listener, known server PIDs are gone, and the GPU is back to WDDM display-only processes.

### 2026-08-16 16:36:57 PDT - M8 saves cost and loses essentially the same amount of yield

- Profiled the same four-step/top-k4 tree with eight verify nodes. Trace: `benchmark/windows/profiles/target_width_m8-20260816-163532/target_width_m8-1786923332.9323876-TP-0.trace.json.gz`.
- M8 replay spans are target **18.004 ms**, draft decode **3.647 ms**, and draft extend **1.605 ms**, totaling **23.256 ms/cycle**. Relative to M12, it saves **1.175 ms / 4.81%** of captured work; almost all savings are in the wider target pass, while the two tree-construction graphs remain effectively fixed-cost.
- The profiled 128-token window happened to emit 3.3684 tokens/traversal, but four longer 512-token controls resolved the noise: emitted lengths `2.6392, 2.7978, 2.7826, 3.0476`; aggregate histogram `[168, 185, 139, 91, 146]` over 729 traversals gives **2.8107 emitted/traversal**. That is about **4.52% less yield** than M12's 2.9436, almost exactly canceling the captured-cost reduction.
- A matching production streaming sample measured **94.080 tok/s**, below M12's earlier 104.145 sample. M8 is rejected as the production topology. The data now says ordinary fixed-width beam allocation traces a near-flat cost/yield frontier; the next gain requires changing node placement/proposal semantics rather than merely selecting 8 versus 12 nodes.
- Server session 60733 was stopped by Ctrl+C. Port 30000 and the known process IDs are clear; GPU returned to **34 C, 1% utilization, 1438 MiB display allocation**.

### 2026-08-16 16:42:22 PDT - scalar root-breadth discount does not improve M12

- Added an opt-in `--speculative-tree-depth-discount` / `-SpeculativeTreeDepthDiscount` topology lever. It multiplies only the scores used by final global node allocation by `discount**draft_step`; undiscounted cumulative scores still choose draft branches for continuation, candidate tokens are unchanged, and exact target-only verification is unchanged. Default is 1.0. Focused unit coverage passes **5 tests**.
- Tested M12 with discount **0.8** across four 6213/512 windows. Emitted lengths were `2.9595, 3.0118, 3.0118, 2.7380`; aggregate histogram `[119, 174, 175, 102, 130]` over 700 traversals gives **2.9286 emitted/traversal**. Plain M12 remains slightly better at 2.9436. A scalar depth discount does not expose the needed gain.
- The first out-of-place score multiplication caused the draft-decode CUDA graph pool to reserve **1.03 GiB** instead of the normal ~0.08 GiB, leaving 6.99 GiB at 16K and making the experiment structurally unsuitable for 200K. The helper now performs the final-allocation-only multiplication in place; tests and `git diff --check` pass. This memory fix must be confirmed on the next captured experimental server before any 200K use.
- Server session 53340 was stopped by Ctrl+C and port 30000 is clear. The next topology work needs branch/depth observability or exact without-replacement proposals; broad scalar reweighting is already flat.

## Checkpoint — 2026-08-16 16:44:49 PDT — exact sparse SWOR is the next tree lever

- Re-read the supplied Sequoia/tree-verifier audit against the live implementation. Its central direction still applies, while several old blockers are already closed here: native exact target-only tree sampling, compact tree ReplaySSM, selected-path commit, and a captured tree graph all exist.
- The exact Sequoia per-parent sibling algorithm is pinned for implementation: initialize residual `R = P` and draft `D = Q`; draw ordered siblings without replacement from `D`; accept sibling `s` when `u < R[s] / D[s]`; after rejection update `R = normalize(max(R - D, 0))`, remove `s` from `D`, renormalize `D` (uniform over unrejected support if exhausted), and sample the terminal token from `R` when no sibling is accepted.
- Current target-only exact tree verification is the paper's target-top-k baseline. The next credible yield experiment is an opt-in exact sparse sampling-without-replacement verifier plus matching SWOR candidate generation, initially correctness-first at M8/M12, then sparse top-20 support and an irregular root-heavy fixed topology if yield clears the measured cost curve.
- M12 depth discount `0.8` lost (`2.9286` emitted/traversal versus plain `2.9436`). Its first out-of-place score multiplication also inflated draft-decode graph reservation to about 1.03 GiB; the helper now mutates scores in place with `mul_()`. Confirm graph memory has returned to normal before any 200K experiment.
- No server is intentionally running. Session 53340 was stopped and port 30000 was clear at the preceding check; re-verify process, port, and GPU state before the next launch.

### 2026-08-16 17:04 PDT — SWOR M12 server captures all speculative graphs

- The opt-in exact sampling-without-replacement candidate is live at `http://127.0.0.1:30000` with `topk=4`, four speculative steps, 12 draft tokens, a 16,384-token active pool, and 200,000 logical context. Default target-only behavior remains unchanged.
- All three full CUDA-graph phases completed: target verify in `2.01 s` (`12` tokens/request), draft decode in `15.80 s` (`4` tokens/request), and draft extend in `0.84 s` (`12` tokens/request). Available device memory after capture was `7.92 GB`.
- Uvicorn completed startup at `16:59:18 PDT`. Logs are `%LOCALAPPDATA%\Temp\sglang-qwen-swor-m12.{stdout,stderr}.log`.
- Startup and graph coverage are established; generation correctness, thinking/tool behavior, acceptance statistics, and controlled performance versus target-only remain pending.

## Checkpoint — 2026-08-16 17:09:24 PDT — incoming SWOR stream recovered and runtime cleaned

- Re-reading the recovery set found a concurrently launched experimental SGLang tree from the user's other GPT-5.6 Pro process. It was the intended RadixArk M12/four-step SWOR server (`--speculative-tree-sampling-mode swor`, aligned draft top-k 20), listening on port 30000 with about 25.1 GiB resident.
- With explicit user authorization, stopped only its identified CUDA-worker leaf PID 27920. The exact tree `27736 -> 37440 -> 2484 -> 3260 -> {27920, 5060}` cascaded out. The listener is gone; all six known PIDs are absent; GPU state returned to 1.72 GiB display residency / 30.47 GiB free at 29 C.
- The other stream left an uncommitted SWOR implementation across the native exact-tree kernel, tree sampling binding, server/hook arguments, draft graph buffers, EAGLE info/worker/utils, launcher, and focused tests. Preserve and audit those edits as incoming work; do not independently reimplement over them.
- Immediate continuation: inspect the full diff and native algorithm against Sequoia Algorithm 2, run CPU/static gates, then run the CUDA distribution suite as the only CUDA workload. Only after correctness and graph-memory checks should the already-running M12 SWOR experiment be relaunched and measured.

## Checkpoint — 2026-08-16 17:41:09 PDT — crashed OpenCode session recovered

- Recovered OpenCode session `ses_ff2e38896ffeoNOG8wnIytY3y5`, titled `Repairing exact SWOR proposal topology`, from `C:\Users\Daniel\.local\share\opencode\opencode.db`. Its last recorded action was reading `scripts/windows/invoke_cuda_python.ps1`; it stopped before rerunning native tests through the initialized MSVC/CUDA environment.
- The live worktree contains the fixed root-heavy 12-node SWOR topology rewrite in `eagle_utils.py`, `eagle_worker_v2.py`, and `spec_utils.py`. The focused draft-proposal unit test and Python bytecode compilation passed. Direct native test invocations were invalid on this host because they lacked `cl.exe`; the CPU kernel suite also requires the unavailable Windows `sgl_kernel` wheel, and Ruff is absent from the venv.
- The first recovery snapshot showed only the Python topology rewrite, but that snapshot raced the still-running OpenCode process. A subsequent live refresh found additional uncommitted edits in the native CUDA verifier and both focused test files: explicit uniform fallback after finite-q support exhaustion plus topology/exhaustion/distribution coverage. Treat the files as incoming completed work until the diff and native Windows CUDA tests establish correctness.
- A stale mid-edit M12 SWOR server is still healthy on port 30000. Its process lineage is `31876, 9136, 26216, 12988, 30948, 37384`; CUDA worker `37384` holds the active model. At inspection it used about 24,992 MiB with the GPU at 28 C and 0 percent utilization. Stop this exact tree before compiling or testing; it predates the latest source edits and cannot qualify the live diff.
- Stopped only CUDA-worker PID `37384` under the user's standing authorization for this other OpenCode process. The identified six-process server tree cascaded out, port 30000 is clear, and the GPU returned to 1,785 MiB display residency with 30,403 MiB free at 29 C. The machine is clean for one native CUDA workload.

## Checkpoint — 2026-08-16 17:50:05 PDT — hidden OpenCode continuation fully recovered

- The OpenCode UI had crashed while its headless `opencode2.exe` PID `32452` continued the same session in the background. Its database sequence advanced from `379` to `557` and it continued editing the shared worktree while this recovery was underway. This explains why the first snapshot appeared to lack the native verifier and test fixes.
- The completed incoming work includes uniform proposal fallback after positive q support exhaustion in the CUDA verifier, fixed root-heavy topology construction, production-organizer Monte Carlo coverage, support sizes zero/one/three/four, branch-factor-one parity, and CUDA-graph replay mutation coverage. A misplaced `draft_support_remaining` declaration was moved into the SWOR kernel before native compilation.
- The background session repaired its large Monte Carlo test to build production topology metadata once and tile only the fixed links, avoiding a display-GPU timeout from launching the general Triton tree builder for 16K to 40K synthetic batch rows. It also made the expanded chain metadata contiguous.
- Its last command completed the full native exact-tree test file with all **10 tests passing in 0.595 seconds**. After that completion, stopped only orphan `opencode2.exe` PID `32452`; its transient shell child was already gone. ZCode was left untouched.
- Repeated the same native suite from the now-stable worktree through `scripts/windows/invoke_cuda_python.ps1`: all **10 tests passed in 0.652 seconds**. The focused Python proposal/topology suite also passes all **10 tests**, Python compilation passes, and `git diff --check` passes. Next gate is a fresh M12 SWOR server capture and real sampled yield/throughput measurement.
- Re-ran the existing top-k-one fast-path regression file through the initialized Windows CUDA toolchain as well: all **11 tests passed in 0.308 seconds**. This closes the earlier invalid direct invocation that lacked `cl.exe` and confirms the opt-in SWOR rewrite has not disturbed the qualified linear path's focused contracts.

### 2026-08-16 17:58:52 PDT — exact fixed SWOR is correct but the first M12 topology loses badly

- Launched the stable fixed-topology implementation at 16K active tokens and 200K logical context with four draft steps, top-k four, 12 verify nodes, SWOR, aligned top-k20 q, Triton draft attention, FlashInfer target verification, torch compile disabled, and FlashInfer autotune disabled. All three graphs captured and health returned 200.
- Graph capture exposed a memory regression: target verify used 0.21 GiB, draft decode reserved **1.08 GiB**, and draft extend used 0.07 GiB, leaving **6.95 GiB**. The prior plain tree draft-decode graph used roughly 0.08 GiB. The likely new owner is dense PyTorch exponential-race/top-k proposal sampling and its graph-pool workspace; this needs a fused CUDA implementation before any exact-200K attempt.
- Three native acceptance windows emitted 512 tokens in 174, 182, and 162 verifier cycles. Aggregate histogram is `[56,180,112,66,104]` over 518 cycles, or **2.9653 emitted tokens per target traversal**. That is only about 0.74 percent above the plain target-only M12 result of 2.9436 and far below the greater-than-four requirement.
- Three real sampled streaming runs measured **88.284, 82.811, and 83.043 tok/s**, mean **84.713 tok/s**. This decisively loses to the 122.712 tok/s qualified linear production path and the earlier 104.145 tok/s plain M12 tree.
- A fresh trace at `benchmark/windows/profiles/target_width_m12-20260816-175748/target_width_m12-1786928268.2689219-TP-0.trace.json.gz` measured graph replay spans: target **22.630 ms**, draft decode **4.671 ms**, draft extend **1.763 ms**, sum **29.064 ms/cycle**. The native exact SWOR verifier alone costs about **1.350 ms/cycle** because it repeatedly scans the full 248,320-token vocabulary. The initial 12-node 4/4/2/1 topology is rejected; the active tree route now requires fused sparse support sampling/verification and a materially deeper/wider fixed topology whose measured mean exceeds four.
- Replaced the verifier's repeated dense residual scans with an exact shared-memory path for target supports up to 64 entries. It compacts and vocabulary-sorts p once per visited parent, performs each recursive residual update over that support, samples the terminal residual in shared memory, and retains the original dense-vocabulary algorithm when runtime support exceeds 64. Added an explicit 65-token-support fallback test; the complete native exact-tree suite now passes **11 tests in 0.591 seconds** from the warm JIT cache.
- Runtime recapture confirmed the proposal-graph allocation issue was transient compilation/cache state rather than the fixed topology: draft-decode graph reservation returned from 1.08 GiB to **0.10 GiB**, leaving **7.93 GiB** after all three graphs. A fresh 128-token trace reduced exact SWOR verification from **1.350 ms to 0.359 ms per cycle** (73.4 percent), while target graph span remained about 22.79 ms because the saved millisecond is small relative to model GEMMs and run variance. The optimized experimental server was stopped by its exact CUDA-worker leaf; its process tree cascaded out, port 30000 is clear, and the GPU returned to display-only residency.

## Checkpoint — 2026-08-16 18:11:58 PDT — restart recovery and topology handoff

- Recovered the crashed OpenCode continuation in full. The fixed SWOR candidate generation, exact verifier, uniform support-exhaustion fallback, topology organizer, and graph-replay coverage are present in the worktree. Focused Python tests pass 10 cases; native exact-tree tests pass 11 cases; the top-k-one native fast-path regression passes 11 cases.
- The first fixed M12 SWOR topology is rejected. Across 518 verifier cycles it emitted **2.9653 tokens per target traversal** with histogram `[56,180,112,66,104]`; three real sampled runs measured **88.284, 82.811, and 83.043 tok/s**, mean **84.713 tok/s**. Its node allocation gives four siblings at the root and almost no sibling coverage deeper in the tree, where conditional continuation is only about 0.61.
- The native verifier now compacts target support up to 64 tokens into shared memory, preserves vocabulary-ordered residual sampling, and falls back to the dense exact path for larger supports. The measured verifier cost fell from **1.350 ms to 0.359 ms per cycle**. Draft-decode graph reservation recaptured at **0.10 GiB**, confirming the earlier 1.08 GiB reservation was transient.
- Evidence is retained under `benchmark/windows/profiles/target_width_m12-20260816-175748/` and `benchmark/windows/profiles/target_width_m12-20260816-180829/`. The latest experimental server is stopped, port 30000 is clear, and the CUDA runtime was returned to a clean display-only state.
- Next action is to collect accepted node and sibling-rank statistics, build a topology/yield oracle, and use it to choose a materially better fixed irregular tree. Only narrow, deeper candidates with useful sibling coverage beyond the root warrant another benchmark.

### 2026-08-16 18:15:36 PDT — asynchronous accepted-path oracle instrumentation ready

- Added opt-in `--speculative-swor-collect-path-stats` plumbing and the matching launcher switch `-SpeculativeSworCollectPathStats`. Ordinary inference remains unchanged when the switch is absent.
- Oracle runs retain the verifier's accepted global node rows, copy the tiny tensor asynchronously on the existing result-copy stream, convert each request to local node IDs, and aggregate an accepted-node histogram. A completed request writes one machine-readable `SWOR_ACCEPT_PATH_STATS` log line with verifier count and histogram.
- This histogram supplies the useful topology quantities directly: node count divided by its parent's accepted count is branch contribution conditional on reaching that parent; sibling counts sum to the continuation probability bought by that sibling group. It adds no per-cycle copy to production because collection is opt-in.
- Static Python compilation, PowerShell parsing, `git diff --check`, the 10-test proposal/topology suite, the two-test decode-bookkeeping contract, and a four-test result-processing suite all pass. The latter now covers batch-local conversion of global accepted tree rows.
- Next run: relaunch only the fixed M12 SWOR topology with path collection, gather a long histogram, then use those node/rank probabilities to choose the smallest informative wider oracle topology.

### 2026-08-16 18:19:36 PDT — M12 path oracle identifies the dominant spine

- Relaunched the same fixed 12-node SWOR topology with accepted-path collection. Capture remained healthy: target verify reserved 0.21 GiB and draft decode 0.10 GiB. Draft extend showed a fresh-process 0.82 GiB reservation instead of the preceding warm-cache 0.07 GiB, leaving 7.19 GiB; treat that as a cold-capture allocation anomaly until a recapture separates cache state from instrumentation.
- Three production-parameter requests generated 2048, 1024, and 1024 tokens in 572, 337, and 339 verifier cycles. Their emitted lengths were **3.5804, 3.0386, and 3.0206**; decode rates were **106.834, 90.516, and 91.436 tok/s**. The first continuation became unusually easy late in the request, so its faster rate is workload acceptance variance rather than a promoted throughput result.
- Aggregate oracle data across 1248 cycles is `node_histogram=[0,978,105,34,13,704,67,22,7,512,36,374]`. The histogram implies 2852 accepted drafts and **3.2853 topology outputs per traversal** including one bonus. The response-visible aggregate is 4096 divided by 1248, or 3.2821, due to final-cycle truncation.
- Root sibling ranks contribute **78.37%, 8.41%, 2.72%, and 1.04%** of all cycles, for 90.54% root continuation. The one-child second level continues 70.80% of accepted roots. Conditional first-child acceptance on the dominant path is stable near 0.72: node 5 over node 1 is 71.98%, node 9 over node 5 is 72.73%, and node 11 over node 9 is 73.05%.
- This rejects equal treatment of sibling branches. Most value sits on the rank-zero spine; root ranks two and three together buy only 3.77 accepted drafts per hundred traversals, and their next-level nodes buy only 2.32. The next oracle topology should spend those low-value nodes on ordered sibling prefixes deeper along the dominant branch, while retaining just enough root breadth to avoid an early hard rejection.
- Stopped only CUDA-worker PID 30872. Its identified server tree cascaded out, port 30000 is clear, and the GPU returned to 1.78 GiB display residency at 30 C.

### 2026-08-16 18:21:40 PDT — arbitrary fixed topologies and offline path analysis ready

- Added opt-in `--speculative-swor-topology` and launcher `-SpeculativeSworTopology`. The value is a JSON parent-node array, parsed and validated through the same fixed-topology builder; omitted input preserves the existing M12 default exactly.
- Added `scripts/windows/analyze_swor_topology.py`. It aggregates any number of `SWOR_ACCEPT_PATH_STATS` records, skips warmups explicitly, and reports depth yield plus per-node unconditional, parent-conditional, and sibling-group continuation probabilities.
- Re-analysis of the three M12 oracle samples reproduces 1248 cycles, histogram `[0,978,105,34,13,704,67,22,7,512,36,374]`, and 3.285256 expected outputs per traversal. This makes topology experiments reproducible from retained logs rather than hand arithmetic.
- Twelve focused proposal/topology tests pass, including custom JSON topology validation. Python compilation, PowerShell parsing, and `git diff --check` pass.
- Next controlled topology is an M16 information oracle: four root siblings; four depth-two siblings under the dominant root candidate plus one child under each remaining root candidate; then four depth-three siblings under the dominant depth-two candidate. It measures the currently unknown deeper sibling-rank distribution with three draft steps.

### 2026-08-16 18:24:03 PDT — M16 oracle measures deeper sibling value

- Ran topology `[-1,0,0,0,0,1,1,1,1,2,3,4,5,5,5,5]`: four root siblings, four siblings under root rank zero, one child under each other root rank, and four siblings under the dominant depth-two node. It uses 16 target rows and three draft steps.
- Graph capture was clean: target verify 0.21 GiB, draft decode 0.10 GiB, draft extend 0.10 GiB, and 7.88 GiB free. This confirms the preceding 0.82 GiB draft-extend reservation was transient cold-process state rather than path instrumentation.
- Three production-parameter requests produced 4096 tokens in 673, 331, and 332 cycles. Decode rates were **94.160, 99.041, and 99.084 tok/s**. Aggregate path data across 1336 cycles is `node_histogram=[0,988,122,62,27,701,98,33,15,75,41,14,471,76,29,11]`, giving **3.068114 topology outputs per traversal**.
- The measured ordered sibling distributions conditional on reaching each dominant parent are:
  - depth one: `[0.7395, 0.0913, 0.0464, 0.0202]`, group continuation **0.8975**;
  - depth two: `[0.7095, 0.0992, 0.0334, 0.0152]`, group continuation **0.8573**;
  - depth three: `[0.6719, 0.1084, 0.0414, 0.0157]`, group continuation **0.8374**.
- This is the missing yield model. Later sibling prefixes remain valuable, but each level's rank-zero proposal declines while the second sibling becomes slightly more important. A single deep spine saturates too low; a full four-way tree grows too quickly. The next topology must use two-child prefixes on the most probable parents and prune the low-probability tail with a cost-aware fixed-tree optimizer.
- Stopped only CUDA-worker PID 5600. Its exact server tree cascaded out, port 30000 is clear, and the GPU returned to display-only residency.

### 2026-08-16 18:30:12 PDT — measured topology search moves the bottleneck to q quality

- Added `scripts/windows/optimize_swor_topology.py`, a fixed-tree beam search using measured ordered sibling probabilities. It enforces sibling prefixes, four-row draft frontiers, node/depth budgets, and emits launcher-ready parent arrays. Its default cycle model is fitted to the measured M12 and M16 oracle cycles.
- With the measured depth-one through depth-three rank vectors, a 32-node/depth-eight search with 0.97 unmeasured-depth decay reaches only **3.9800 expected outputs**. Even the deliberately optimistic no-decay search tops out at **4.0921 outputs** with 32 nodes and depth nine. The cost-ranked winner is a 12-node/depth-three topology at roughly 100 predicted tok/s, consistent with the M16 measurements.
- This closes the current-q topology-only branch: rearranging the same proposals cannot reach 200 tok/s. Tree verification remains the required amortization topology, while proposal overlap must improve materially and per-depth draft cost must fall before greater width/depth pays.
- Added an opt-in CUDA proposal-overlap oracle behind `--speculative-swor-collect-overlap-stats` and launcher `-SpeculativeSworCollectOverlapStats`. A dedicated one-block-per-target-row CUDA kernel compacts q support and evaluates exact `sum(min(p,q))` plus q mass outside target support for temperature scales `[0.70,0.85,1.00,1.15,1.30]` and retained q widths `[4,8,12,16,20]`. Production SWOR uses the unchanged kernel when the flag is absent.
- Added request-level overlap accumulation and one final `SWOR_OVERLAP_STATS` record. The native exact-tree suite, including a hand-checked sparse overlap grid, passes **12 tests in 71.92 seconds** after JIT rebuild. Sixteen focused CPU tests, Python compilation, PowerShell parsing, and `git diff --check` pass.
- Next run: one M16 overlap-oracle server, one long request, select the best q scale/support by depth, then implement only the measured calibration and remeasure path yield.

### 2026-08-16 18:31:26 PDT — tree experiment closed and preservation commit requested

- The measured fixed-tree search has answered the current question: with the present proposal distribution and draft cost, wider SWOR topologies regress real sampled throughput and topology rearrangement alone cannot reach the 200 tok/s target.
- Preserve the completed exact-tree implementation, CUDA sparse verifier and overlap oracle, runtime plumbing, topology analysis tools, focused tests, and retained M12 profile evidence in one comprehensive experimental commit.
- Keep every tree and oracle path opt-in. The qualified linear rejection-sampling configuration remains the production default and is the configuration to restore after the commit; its established real sampled mean is **122.712 tok/s**.
- At the last clean runtime check there was no SGLang listener or CUDA worker and the RTX 5090 had returned to display-only residency. Reverify process, port, and GPU state before relaunching the production server.

### 2026-08-16 18:35:16 PDT — preservation committed and qualified linear server restored

- Preserved the exact SWOR tree verifier, sparse CUDA verifier and overlap oracle, runtime diagnostics, topology tooling, tests, notes, and two M12 trace artifacts in one comprehensive commit. The production defaults remain unchanged by the opt-in experiment.
- Relaunched the qualified unsimulated RadixArk linear configuration with seed `783025237`. Resolved arguments confirm exact `200000` context and token pool, two speculative steps, three draft tokens, top-k-one linear rejection sampling, aligned draft top-k 20, target and draft TRT-LLM MHA/XQA, FlashInfer prefill, FP8 E4M3 draft KV, Qwen3 reasoning, Qwen3 Coder tools, and all SWOR collection/topology switches disabled.
- Target verification, draft decode, and draft extend CUDA graphs captured successfully in `43.95 s`, `51.58 s`, and `1.21 s`. The endpoint is healthy on `127.0.0.1:30000`; model information reports generation enabled with image and audio understanding disabled.
- A real sampled `256+16` smoke request completed at **120.862 tok/s** decode with temperature `1.0`, top-p `0.95`, top-k `20`, and presence penalty `1.5`. Post-request health remains HTTP 200, the GPU is idle at 29 C with 1186 MiB free, and the server owns the expected seven-process tree rooted at hidden PowerShell PID `22160`.
- Runtime logs are `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-linear-b7f4a0b005-seed783025237.stdout.log` and `.stderr.log`. Leave this qualified linear server live for OpenCode2.

### 2026-08-16 18:53:21 PDT — 232K production pool requested

- User requested raising the qualified linear server from a real 200,000-token context and pool to **232,000**.
- The checkpoint maximum is 262,144. Scaling the measured target and draft FP8 KV pools from 200K to 232K adds approximately 1.04 GiB, projecting roughly 0.88 GiB of graph-end headroom from the latest 1.92 GiB capture result.
- Change both launcher defaults together, retain every reasoning/tool/vision-disabled and speculative-performance setting, stop only the identified live server tree, then require clean graph capture, a sampled smoke request, and a near-limit capacity request before leaving the replacement server live.

### 2026-08-16 19:04:07 PDT — 232K rejected on operating headroom; restore 200K

- The real 232,000-token target and draft pools allocated and all three CUDA graphs captured, ending with 0.67 GiB reported headroom. A sampled `256+16` smoke request passed at 114.339 tok/s.
- The decisive `231000+16` request also completed correctly: **1317.527 prompt tok/s**, **82.342 decode tok/s**, and exactly 231,016 total tokens. A standard sampled `6213+512` control remained healthy at **120.653 tok/s**.
- The near-limit request left only **98 MiB** free VRAM before cache flush. Although `/flush_cache` restored roughly 1.1 GiB, the user correctly rejected that operating margin as too tight.
- Restore both launcher defaults to the proven **200,000** context and token pool, stop only the identified 232K server tree, and leave a freshly captured 200K production server live.

### 2026-08-16 19:06:12 PDT — 200K production restored; optimization goal complete

- Stopped only the two CUDA-worker leaves of the identified 232K server; its exact seven-process tree cascaded out, port 30000 cleared, and the RTX 5090 returned to display-only residency.
- Restored the launcher defaults byte-for-byte to `ContextLength=200000` and `MaxTotalTokens=200000`, then relaunched the seeded unsimulated linear winner under hidden PowerShell PID `30688`.
- Runtime arguments confirm the real 200K context and pool. Target verify, draft decode, and draft extend graphs captured in `43.45 s`, `1.39 s`, and `1.23 s`, ending with **1.84 GiB** reported headroom. The endpoint is ready on port 30000.
- User accepted the restored 200K operating margin and explicitly marked the exhaustive NVFP4 optimization goal complete.

### 2026-08-16 21:18:33 PDT — real sampling with reasoning disabled

- The existing qualified server remained live throughout: listener PID `30960`, CUDA worker PID `30436`, exact 200K RadixArk linear topology, seed `783025237`, two speculative steps / three draft tokens, aligned draft top-k 20, TRT-LLM MHA/XQA, FlashInfer prefill/sampling, and FP8 draft KV. No restart or server edit occurred.
- Added a default-preserving `--disable-thinking` benchmark flag. It sets both `enable_thinking=false` and `preserve_thinking=false`, calibrates the prompt through that exact chat template, and records the selected mode. The acceptance harness gained the same opt-in flag. Python compilation and CLI help passed.
- Mode integrity check at `256/64`: disabled thinking returned `reasoning_chars=0`, `content_chars=295`; enabled thinking returned `reasoning_chars=281`, `content_chars=0`.
- Workload for all full measurements: exact `6213/512`, temperature `1.0`, top-p `0.95`, top-k `20`, presence penalty `1.5`, ordinary unsimulated rejection sampling, exact-shape warmup, cache flush, SSE client timing.
- Fresh thinking-enabled control decode TPS: `115.184, 122.593, 120.839, 116.862, 116.393`; mean **118.374**, median **116.862**, mean TTFT `0.698222 s`, mean E2E `5.017495 s`.
- Reasoning-disabled decode TPS: `128.232, 124.343, 123.311, 121.721, 138.824, 137.832, 136.199, 134.703, 129.760, 122.291`; mean **129.722**, median **128.996**. Window means were **127.286** and **132.157**. Mean TTFT was `0.708832 s`; mean E2E was `4.657528 s`.
- Matched change: **+9.59% decode TPS** and **-7.17% E2E time**. Prompt tokens and completion tokens were exactly 6213 and 512 in every sample.
- Benchmark-of-record comparison: reasoning-disabled **129.722 tok/s** versus the qualified reasoning-enabled **122.712 tok/s** is **+7.010 tok/s / +5.71%**. The fresh 118.374 tok/s control above is a same-session diagnostic showing stochastic/WDDM variance; it does not replace the qualified benchmark.
- Five disabled-thinking acceptance probes measured lengths `2.485437, 2.381395, 2.461538, 2.426540, 2.509804`, mean **2.452943**, with mean `208.8` target verifications. Five matched thinking-enabled probes measured `2.306306, 2.188034, 2.275556, 2.235808, 2.295964`, mean **2.260334**, with mean `226.6` verifications.
- The throughput gain tracks **+8.52% accepted length** and **-7.86% target verification cycles**. Reasoning-disabled output is a separate behavior profile and does not replace the reasoning-preserved production qualification.
- WDDM conditions remained visible in the evidence: Chrome and ZCode were active, and the two disabled windows moved materially. The endpoint stayed healthy after the run.

### 2026-08-16 22:22 PDT — two-graph tree cycle, sparse GDN replay, and post-change width checkpoint

- The user opened a new explicit performance branch with a hard completion gate of **at least 200 tok/s under ordinary real sampling**, authorizing model requantization, PyTorch rewrites, and native kernels. Reasoning/tool continuity, the real 200K capacity contract, and the established production behavior gates remain part of qualification. Work began on `main` at `b8426ebe7c05e4b24e7393f1f81f947fc5f79905`; all changes in this checkpoint remain uncommitted and opt-in.
- Added `--speculative-device-resident-cycle` and launcher `-SpeculativeDeviceResidentCycle`, currently restricted to batch-one, single-layer EAGLE/NEXTN, top-k tree decode, target-only scoring, no adaptive depth, no aligned draft sampler, and full CUDA graphs. A retained-raw-graph backend plus `CudaGraphChildSequence` creates one CUDA parent containing **draft extend -> device bridge -> next draft decode**. Steady state is now two launches per speculative cycle: target verify and the composite parent. The bridge keeps selected logits/hidden state, renormalization/top-k, next sequence lengths, cache metadata, and next-draft graph preparation on device-resident static addresses.
- Extended `scripts/windows/analyze_torch_trace.py` with graph-transition gaps. The old M12 schedule measured draft-decode -> target **1.309/1.195 ms mean/median**, target -> extend **1.474/1.423 ms**, and extend -> next draft-decode **3.228/2.475 ms**. The post-composition M12 trace `benchmark/windows/profiles/target_width_m12-20260816-220558/target_width_m12-1786943158.6831138-TP-0.trace.json.gz` contains only target graph 6 (**18.378/18.357 ms**) and composite graph 15 (**5.654/5.600 ms**) in steady state; the former extend-to-next-draft launch seam is inside graph 15. Remaining median host gaps were target -> composite **1.210 ms** and composite -> target **2.643 ms**.
- Added a generic CUDA child-graph parity test and graph-safe fixed-width prefix-tail duplication. The first bridge capture failed because boolean indexing lowered through data-dependent `nonzero`; replacing it with target-row self-copies on invalid lanes preserved semantics and capture shape. `powershell -NoProfile -File scripts\windows\invoke_cuda_pytest.ps1 test\registered\kernels\test_cuda_graph_composite.py -q` passed **1 test**. `test/registered/unit/spec/test_device_resident_eagle_cycle.py` passed **3 tests**.
- Replaced dense GDN pair state `[B,H,N,N,2]` with strict-ancestry packed state `[B,H,N,max_tree_depth,2]`. M12 needs **56** dot reductions (self kq plus ancestor kk/kq) instead of **288**, a 5.14x reduction. Added a replay-parameter kernel so gate/alpha/beta are computed once per `(batch,value-head,node)` instead of redundantly in every one of 16 value tiles, then made pair reductions warp-parallel. `powershell -NoProfile -File scripts\windows\invoke_cuda_pytest.ps1 test\registered\kernels\ops\attention\test_gdn_tree_replay.py -q` passed **3 tests**, covering reference parity, accepted-path commit, and CUDA-graph replay. In the final pre-lifetime-fix M12 trace, per-layer means were main **26.162 us**, sparse pair **5.415 us**, and parameter build **1.779 us**, about **1.60 ms per 48-layer target cycle** versus the preceding approximately 1.656 ms path.
- Added `scripts/windows/sparse_pq_swor_oracle.py` and eight focused CPU tests. The offline oracle exactly enumerates finite sparse proposal draws and verifier coins, carries branch-local presence/frequency state as initial counts plus root plus accepted suffix, handles proposal-support exhaustion through the exact uniform unrejected-vocabulary fallback, accepts componentized measured cycle costs, and refuses promotion unless projected throughput strictly exceeds the configured target plus margin. The focused oracle suite passed **8 tests**. Runtime p/q capture has not yet supplied a promotion candidate.
- Post-change M12 acceptance lengths across five exact `6213/512` probes were `2.752688, 3.047619, 2.652850, 3.029586, 2.639175`, mean **2.824384**. Matching real-sampling decode rates were `94.269, 106.585, 100.297, 104.620, 110.115 tok/s`, mean **103.177 tok/s**. M12 is far below the 200 tok/s gate and is not promotable.
- Post-change M8 trace `benchmark/windows/profiles/target_width_m8-20260816-220916/target_width_m8-1786943356.5113451-TP-0.trace.json.gz` measured target **18.609/18.568 ms**, composite **5.664/5.584 ms**, and median target -> composite / composite -> target gaps of **1.171/2.022 ms**. Five `6213/512` acceptance lengths were `2.666667, 2.737968, 2.813187, 2.828729, 2.639175`, mean **2.737145**. Five real-sampling decode rates were `93.577, 104.793, 94.714, 98.226, 95.452 tok/s`, mean **97.352 tok/s**. M8 remains below M12 and is rejected.
- The first M16 composite capture exposed a cross-runner lifetime defect. Draft-decode and draft-extend runners intentionally alias equal-sized `ForwardInputBuffers`; the eager next-draft precompute overwrote the extend `seq_lens` before one-time bridge warm/capture. M8/M12 remained in bounds accidentally, while M16 produced a `ScatterGatherKernel` out-of-bounds assertion in `prepare_for_draft`. The bridge capture now reseeds the extend runner's complete static inputs immediately before both warmup and capture. A `CUDA_LAUNCH_BLOCKING=1` reproduction pinpointed the gather, and the fixed M16 server captured all three component graphs and completed its startup request without an assertion.
- The interrupted M16 profiling client nevertheless completed cleanly and stopped the profiler. Trace `benchmark/windows/profiles/target_width_m16-20260816-221953/target_width_m16-1786943993.4730976-TP-0.trace.json.gz` has 37 steady target/composite cycles: target **21.404/21.448 ms**, composite **6.043/5.871 ms**, and median gaps **1.314 ms** target -> composite and **2.522 ms** composite -> target. Its 128 emitted tokens imply approximately **3.46 emitted tokens/cycle** for this short window; a long acceptance window is still required. This shape remains mathematically below 200 tok/s at measured cycle cost and is not a promotion candidate.
- The completed M16 long window resolved the favorable short-profile noise. Five exact `6213/512` acceptance lengths were `2.892655, 3.103030, 3.103030, 3.121951, 3.084337`, mean **3.061001**. Five matching real-sampling decode rates were `98.158, 100.223, 93.694, 89.908, 82.173 tok/s`, mean **92.831 tok/s**. Throughput declined during the window under visible WDDM variance, yet even its best sample remained about half the promotion gate. M16 is rejected.
- Current runtime at this checkpoint is the fixed opt-in M16 server: listener PID `17344`, root hidden PowerShell PID `21016`, 200K logical context, 16K active pool, four speculative steps, 16 draft tokens, top-k four, plain target-only scoring, rejection sampling disabled, device-resident cycle enabled, and torch compile disabled. Logs are `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-m16-composite-fixed.stdout.log` and `.stderr.log`. Stop only its verified process tree, then repeat the selected M12 control after the alias fix before using the completed width curve to choose the next 200 tok/s path.

### 2026-08-16 22:27 PDT — corrected-capture M12 closes the width-only branch

- Stopped the verified M16 server by terminating CUDA-worker leaf PID `10160`; its remaining identified tree cascaded out. Port 30000 cleared, no matching compiler/CUDA workers remained, and the GPU returned to ordinary WDDM residency before the replacement launch.
- Relaunched M12 with the same seed and experimental controls after the shared-buffer reseed fix. Resolved shape is four speculative steps, 12 target rows, top-k four, target-only scoring, no rejection sampling, the two-graph device-resident cycle, 200K logical context, and a 16K active pool. All component graphs captured and the internal startup generation completed without an indexing fault.
- Corrected trace `benchmark/windows/profiles/target_width_m12-20260816-222434/target_width_m12-1786944274.1622381-TP-0.trace.json.gz` emitted 128 tokens in 42 verifications (**3.047619/cycle**). Target graph 6 measured **19.148/18.934 ms mean/median**, composite graph 15 **5.764/5.757 ms**, with median target -> composite / composite -> target gaps **1.303/2.712 ms**. The corrected graph structure and cost agree with the earlier capture within WDDM variance.
- Five corrected-capture `6213/512` acceptance lengths were `2.942529, 3.047619, 2.942529, 2.828729, 2.767568`, mean **2.905795**. Five matching real-sampling decode rates were `87.870, 101.393, 96.121, 98.484, 89.557 tok/s`, mean **94.685 tok/s**. Prompt throughput in this window was only about 8.0-8.3K tok/s, lower than the preceding M8/M12 windows and consistent with additional WDDM contention; the economic conclusion is unchanged by the absolute slowdown.
- The corrected post-change width results are M8 **2.737 emitted/cycle**, M12 **2.906**, and M16 **3.061**, while measured mean real-sampling rates are **97.352, 94.685, and 92.831 tok/s** in their respective windows. Added width continues to buy modest yield while raising target cost; every shape is far below 200 tok/s. Width-only topology changes remain closed, and no tree shape is promoted.
- The active M12 process at this checkpoint listens under PID `3928`, rooted at hidden PowerShell PID `17552`; logs are `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-m12-composite-fixed.stdout.log` and `.stderr.log`. Stop its verified tree before any compiler, conversion, or next server experiment.

### 2026-08-16 22:36 PDT — p/q calibration grid is flat; remote performance ledgers reviewed

- Stopped the corrected M12 server by its two verified CUDA-worker leaves, PIDs `13516` and `37784`; the remaining identified process tree cascaded out. Port 30000 cleared and the GPU returned to ordinary display residency before the next launch.
- Added `scripts/windows/compare_mtp_tensors.py` and compared every embedded `mtp.*` tensor in the immutable RadixArk and Gittensor RTX5090 checkpoints. All 15 tensors are exactly equal, including the 5120x10240 fusion projection, attention projections, three MLP matrices, norms, and pre-fusion norms. The previously observed acceptance difference comes from target execution/quantization or sampling variance; transplanting the MTP head cannot improve q.
- Ran the retained native proposal-overlap oracle on the 16-node topology `[-1,0,0,0,0,1,1,1,1,2,3,4,5,5,5,5]` with three draft steps, SWOR, draft top-k 20/top-p 0.95, path collection, and the exact real-sampling parameters. One `6213/2048` request completed in 669 verifications at **3.061286 emitted tokens/cycle**, with histogram `[0,514,57,18,9,364,36,16,9,40,11,6,249,31,15,3]` and **90.59 tok/s** request-level generation throughput from the acceptance harness latency.
- Added `scripts/windows/analyze_swor_overlap.py` to parse the machine-readable grid. Across all 669 cycles, baseline `scale=1.0/top-k=20` p/q overlaps on internal nodes 0-5 were `0.75813, 0.70074, 0.51010, 0.47021, 0.41220, 0.66373`. Exhaustive scales `[0.70,0.85,1.00,1.15,1.30]` and retained supports `[4,8,12,16,20]` improved those nodes by only `0.000053, 0.000048, 0.000245, 0.000157, 0.000120, 0.000000`. Temperature/support calibration is decisively flat and is closed.
- The request's q mass outside target support was still material on the dominant rows: **0.0965** at root, **0.1224** on the rank-zero depth-one branch, and **0.1639** on its depth-two continuation. This points specifically at branch-local proposal state or a stronger proposal model; scalar temperature and support width cannot recover it. The offline sparse p/q oracle already models `initial counts + root + accepted suffix` independently for each branch and remains the gate for any candidate.
- Read `/Users/daniel/sglang/PERFORMANCE_LOG.md` and `FAILED_PATHS.md` from `macpro.local` at the user's request. The transferable findings are methodological: specialize the exact hot batch shape, retain correctness references, and require unsynchronized end-to-end evidence because a fused full-attention preparation kernel improved its synchronized microbenchmark by 13.7x yet regressed real serving by 5%. The retained Q5_K/Q6_K batch-eight Metal vec4 kernels are W6900X/GGUF-specific and do not transfer directly to the RTX 5090 ModelOpt path.
- Stopped the overlap server by exact CUDA-worker leaves `30332` and `30344`; its identified process tree cascaded out. Port 30000 is free, the known PIDs are absent, no matching compiler/CUDA worker remains, and the RTX 5090 is back at display-only residency. Logs remain at `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-m16-overlap.stderr.log` and `.stdout.log`.

### 2026-08-16 22:47 PDT — continuing optimization ledgers restored from the original notebook

- The user invoked `$performance-optimization`, making the benchmark-driven loop, root performance ledger, rejected-path ledger, analysis-only review batches, and regular atomic commits the active workflow.
- Created root `PERFORMANCE_LOG.md` with the qualified 122.712 tok/s real baseline, 171.263 tok/s fixed ceiling, exact 199016 capacity contract, raw samples, environment, benchmark commands, current seam/GDN/width/overlap deltas, candidate inventory, commit status, and historical supersession map.
- Created root `FAILED_PATHS.md` with 32 evidence-backed closed/blocked candidates and explicit reopening conditions. It covers the measured draft quantizers, static/adaptive speculation, checkpoint branch, width/depth/SWOR searches, q calibration, old dependency/backend failures, unsafe asynchronous buffer reuse, memory/context experiments, and environment false leads.
- Recovered the original root notebook directly from Git history without changing the worktree. `NOTES.md` was deleted by `b8426ebe7c05e4b24e7393f1f81f947fc5f79905` during the notes migration; `b8426ebe7c05e4b24e7393f1f81f947fc5f79905^:NOTES.md` contains 2,428 lines through the 18:35 production restoration. Its supersession map and closed-path evidence were folded into the new ledgers; the compact `notes/` files remain the current handoff layer.
- Both root files are untracked and intentionally visible to Git, with 124 and 388 lines respectively. `git diff --no-index --check -- NUL <file>` emitted no whitespace diagnostics for either file; status code 1 is the expected NUL comparison result. No server, CUDA compiler, or benchmark was started for this record-only checkpoint.

### 2026-08-16 22:51 PDT — device-resident SWOR experiment prepared for delivery

- The user requested that the accumulated experiment be committed and pushed. The delivery checkpoint began on `main` at `2f49a60b46c62e728fb7db00a0d042248c27c8f4`, exactly synchronized with `origin/main`, with the 26 implementation, test, analyzer, launcher, oracle, and trace paths already staged.
- The first `git diff --cached --check` found one extra blank line at EOF in `python/sglang/srt/model_executor/cuda_graph_composite.py`. Removed that blank line, restaged the file, and confirmed both `git diff --cached --check` and `git diff --check` are clean.
- Recompiled every touched Python source and test with the repository venv, parsed `scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1` through `ScriptBlock.Create`, and reran the two CPU-focused suites: `test_device_resident_eagle_cycle.py` plus `test_sparse_pq_swor_oracle.py` passed **11 tests** with 14 known Torch JIT deprecation warnings in 10.98 seconds. The native CUDA parity/replay gates remain the successful recorded runs above; no new server, CUDA compiler, GPU test, or benchmark was started for this commit-only checkpoint.
- The intended atomic commit is `Add device-resident SWOR tree experiments`. It preserves the qualified launcher defaults: the device-resident cycle and every SWOR/tree experiment remain opt-in.

### 2026-08-16 23:27 PDT — upstream rebase completed with macOS and Windows portability repair

- Resumed the in-progress `main` rebase of upstream tip `eafbe2cb6fa3e73d150dde1971e262de4c03d8ac` onto private-fork merge `b231e8b378b04ac1e88fe813d9905c29bcc5dcd9`. The histories diverged after `dd458f3212dd4ddf0e1a7907bbf539b660e70d21` by 63 upstream commits and seven private-fork commits. The rebase completed at `b26729f224`; `origin/main..HEAD` is exactly 63 commits before the portability follow-up below.
- Resolved three conflicts by preserving both platform contracts: kept the native-MPS exclusion around Triton/FLA imports while accepting the AMD HIP empty-grid guards; retained the MPS GGUF gated-norm/reorder path while accepting both upstream empty-DP-attention layer guards; and retained the native-Windows dependency exclusions while adding the Python-before-3.11 `tomli` dependency required by the new Rust-extension loader.
- `git range-diff dd458f3212..eafbe2cb6f b231e8b378..b26729f224` reports 60 patch-equivalent commits and only the three expected conflict-context differences above. No upstream patch was dropped, reordered, or otherwise rewritten.
- Rebase validation exposed a new cross-platform issue in upstream's source-build loader: unconditional `fcntl` import prevented Windows import, directory-handle `fsync` was not a Windows operation, the alternate macOS/Windows dependency surface omitted `tomli` on Python 3.10, and the Rust build test hard-coded a Linux `.so` artifact. The follow-up uses `msvcrt.locking` on Windows with blocking retry, retains `fcntl.flock` on POSIX, skips only Windows directory `fsync`, adds `tomli` to `runtime_base`, makes the artifact test platform-derived, and runs the lock test with a spawn context on Windows.
- On this Intel Mac Pro (`x86_64`, macOS 26.6, Python 3.14.6), isolated Rust-loader coverage passed **12 tests** including process lock serialization, Darwin library naming, atomic staging, cache reuse, and the new platform checks. Python source compilation passed for the two conflict files, setup hook, loader, and test; both pyprojects parsed as TOML; target-environment PEP 508 checks passed for Intel macOS, arm64 macOS, and Windows; a simulated `win32` import exercised the `msvcrt` lock/unlock path, DLL naming, and directory-sync bypass; and `git diff --check` passed.
- The first marker-check attempt through `uv --with packaging` could not reach PyPI after three retries. The same check then passed with pip's locally installed vendored `packaging`, so validation did not depend on network recovery. A bounded SSH probe reached the configured Windows desktop account, but that account exposes only the Microsoft Store Python alias and is not the recorded RTX 5090 server checkout; no remote file, process, service, or GPU state was changed. Native arm64-MPS execution and native Windows execution remain unrun in this checkpoint; their evidence is source compilation, target-marker evaluation, retained MPS tests, and the simulated Windows loader path.
- No server, CUDA compiler, GPU benchmark, dependency checkout, protected CUDA compatibility header, launcher default, or production process was touched. The portability follow-up and this ledger entry are the only post-rebase worktree changes pending final validation and delivery.
### 2026-08-16 23:11 PDT — non-front accepted path reproduced and repaired before throughput work

- The user identified a production-disqualifying invariant: a non-front accepted tree branch can leave target KV and the next-draft token/hidden state in tree order while downstream code consumes a compact front block. Throughput optimization was paused immediately. The live branch was `main` at `d0116b54e5766932a46e06e0a66c3672370eaff8`, synchronized with `origin/main`; only the focused correctness worktree paths were modified or untracked.
- Added `test/registered/kernels/test_tree_accept_path_compaction.py`. The minimal path is root/node 3/node 7 under a deliberately nonidentity virtual-to-physical map. Before the repair, the hybrid-pool translation test failed with four of six sentinel values read from the wrong physical rows, while the serial front-compaction boundary test passed. This established the storage-address bug independently from model quality or stochastic acceptance.
- Root cause: `req_to_token` and speculative `out_cache_loc` hold virtual token ids under the unified pool, while `UnifiedMHATokenToKVPool.move_kv_cache` and `UnifiedMLATokenToKVPool.move_kv_cache` accept physical storage ids. `HybridLinearKVPool.move_kv_cache` delegated the virtual ids unchanged. A front/self move could hide the defect; a non-front branch copied unrelated physical rows. MLA's `translate_kv_loc_dense` is a separate kernel-facing address space and cannot identify whole page envelopes for relocation.
- Added a dedicated identity-by-default `_full_move_translate` hook to `HybridLinearKVPool`; `init_unified_mamba_pools` installs `allocator.translate_kv_loc` for both MHA and MLA unified pools. Accepted-path and prefix-tail moves now delegate physical ids while ordinary static pools retain identity behavior. Removed the `finalize_tree_path` switch from `run_eagle_verify`, so single- and multi-layer top-k tree workers must both compact accepted KV, token rows, and hidden rows before draft extend.
- Strengthened the regression to instantiate the real unified-pool factory for MHA and MLA with page-size translation, then capture one graph and replay four alternating non-front paths. It compares physical target K/V, compacted tokens/hidden rows, and the terminal next-draft row against a serial path ledger after every cycle; rejected slots are freed, physical compaction runs, and the test requires a virtual id to be reused. The first strengthened run reported one failure and three passes because graph-pool outputs were inspected immediately after capture; changing the checked contract to an explicit first replay resolved that test-harness issue without a production-code change.
- Exact validation: `powershell -NoProfile -File scripts\windows\invoke_cuda_pytest.ps1 test\registered\kernels\test_tree_accept_path_compaction.py -q` finished **4 passed, 2 subtests passed**. The combined native command over that file, `test_cuda_graph_composite.py`, and `ops/attention/test_gdn_tree_replay.py` finished **8 passed, 2 subtests passed**. CPU commands for `test_multi_ended_allocator.py` and `test_unified_memory_move_gate.py` finished **65 passed** and **5 passed**. Python compilation of all five affected sources plus the test passed, and `git diff --check` is clean.
- No server or compiler tree was active for the CUDA gates. Port 30000 was free; the RTX 5090 snapshot before testing was 27 C, 40.10 W, 652 MHz, 1 percent utilization, 1,645 MiB used, and 30,543 MiB free. Unrelated Quasimorph Python processes and ordinary WDDM display clients were left untouched.
- Decision: every prior M8/M12/M16 tree timing and SWOR yield result is mechanism-only and cannot support production promotion. The qualified linear **122.712 tok/s** result remains the comparison authority. Next: commit this correctness unit, launch the exact production linear default, collect a fresh five-sample `6213/512` real-sampling baseline, then require corrected full-model non-front path parity before any tree throughput or branch-local penalty work resumes.

### 2026-08-16 23:24 PDT — fresh linear baseline compared before resuming optimization

- Committed the accepted-path correctness unit as signed commit `3f276e8acda4db5911db9a69a689deb10bae8360` (`Fix non-front tree accepted-path state`); the worktree was clean and one commit ahead of `origin/main` before launch.
- Started one hidden production-default server through `scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1 -RandomSeed 783025237`, rooted at PowerShell PID `25396`. Its live ancestry was `25396 -> 29236 (sglang.exe) -> 2068 (python.exe) -> 32456 (API)` with scheduler children `37612` and `25308`; listener PID `32456` owned `127.0.0.1:30000`. Logs are `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-linear-baseline-20260816-2318.stdout.log` and `.stderr.log`.
- Resolved arguments confirmed the comparison contract: RadixArk checkpoint, context and pool 200000, page 64, one request, seed 783025237, language-only surface, rejection sampling true, two speculative steps, three draft tokens, EAGLE top-k one, draft proposal top-k 20, device-resident cycle false, target-only tree mode with no topology/stats, FP8 draft KV, TRT-LLM MHA target/draft decode, FlashInfer prefill/sampling, ReplaySSM spec, FP32 Mamba state, torch compile default, and all experimental controls inactive.
- The first readiness request returned 503 during the launcher's internal warm generation. Startup then completed normally at 23:19:26. Target verify, draft decode, and draft extend graphs captured in **42.42**, **1.56**, and **1.09 seconds**; the scheduler reported **1.74 GiB** free at graph end. Subsequent `/health` returned 200 and `/model_info` reported image/audio understanding false.
- Ran ten exact real-sampling commands: `.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5`. Window one was `84.130, 114.807, 118.664, 119.385, 124.278 tok/s`, mean **112.253**, median **118.664**. Window two was `123.237, 123.741, 125.001, 128.689, 123.207 tok/s`, mean **124.775**, median **123.741**. Combined mean was **118.514**, median **123.222**. Every response had exactly 6213 prompt and 512 completion tokens, `finish_reason=length`, and thinking enabled. The first 84.130 result is retained and labeled startup/JIT-affected rather than discarded.
- Five exact native acceptance probes returned `2.381395, 2.169492, 2.160338, 2.124481, 2.188034` emitted tokens per verification, mean **2.204748**, median **2.169492**. This is 4.893% below the historical 2.318174 mean and accounts for much of the lower first-window TPS.
- The server stayed healthy. The post-window GPU snapshot was 32 C, 36.62 W, 300/405 MHz core/memory, zero utilization, 31,966 MiB used, and 222 MiB free after first-request/JIT residency. WDDM clients included Chrome, Edge WebView, Docker Desktop, Snipping Tool, Windows shell clients, and the display compositor; no second SGLang server or compiler tree existed.
- Decision: the top-k-one production comparison path bypasses the repaired top-k tree compaction and shows no matched stable-window regression; its fresh second window is 1.681% above the historical 122.712 tok/s mean. Keep the complete combined result visible. The minimal reproducer and baseline comparison gates are complete; full-model tree state parity remains required before tree throughput, while target-path profiling may now resume against this linear control.

### 2026-08-16 23:47 PDT — exact linear composite is functional and throughput-neutral

- Reachability correction: the recorded M8/M12/M16 logs resolve `enable_unified_memory=False`, and their single-layer EAGLE worker already invoked `_finalize_accept_tree_path`. The virtual-to-physical unified-pool reproducer is a real optional-path bug and the multi-layer compaction bypass is real, while neither proves those exact static-pool measurements corrupted. They remain production-ineligible because the user-required full current-config cross-cycle state comparison has not passed.
- Audited `CudaGraphChildSequence`: its direct `cudaGraphLaunch` bypasses `torch.cuda.CUDAGraph.replay`, including PyTorch's generator-offset update. Existing stochastic SWOR inside the raw draft child can therefore replay capture-stale random offsets. Added an external-race composite replay regression and rejected device-resident SWOR at worker initialization pending explicit RNG state. This further disqualifies earlier SWOR composite yield measurements.
- Implemented an exact linear bridge. It reuses stable draft-runner sampling buffers, refreshes live temperature/top-p/accumulated presence penalties/logit bias, applies draft top-k 20, and carries the exact sampled q into Leviathan verification. The first implementation refreshes two caller-owned 248K-vocabulary Exp(1) race rows before each raw parent launch, so its stochastic inputs advance without relying on the bypassed PyTorch replay hook.
- The first launch at 23:38 failed before model load because the server-argument hook still required top-k greater than one. Relaxed only that obsolete gate for the top-k-one exact bridge and retained the batch-one, single-layer, non-adaptive, target-only constraints. The second launch used the full 200K production defaults plus only `-SpeculativeDeviceResidentCycle`; logs are `C:\Users\Daniel\AppData\Local\Temp\sglang-qwen-linear-device-cycle-20260816-2341.stdout.log` and `.stderr.log`.
- Startup completed with target/draft/draft-extend captures of **40.92**, **52.30**, and **0.99 seconds** and 1.57 GiB reported graph-end headroom. The long one-time draft capture included exact-q FlashInfer autotune. A `6213+64` acceptance smoke completed at 2.370370 emitted/cycle. `/health` stayed 200 and `/model_info` retained the language-only surface.
- Five exact `6213/512` real samples were `117.251, 121.340, 118.959, 131.667, 123.663 tok/s`, mean **122.576**, median **121.340**. Five acceptance probes were `2.188034, 2.216450, 2.188034, 2.275556, 2.359447`, mean **2.245504**. Acceptance improved 1.849% versus the fresh linear probes while TPS fell 1.762% versus the warmed 124.775 control, locating the loss in execution cost.
- Validation before launch: `test_draft_proposal_sampling.py` passed 12, `test_device_resident_eagle_cycle.py` passed 5, `test_eagle_draft_cuda_graph_runner.py` passed 4, and the combined composite/FlashInfer proposal/exact-tree native CUDA command passed all 16 tests. All five server samples returned exactly 512 tokens with thinking enabled.
- Stopped the verified device-cycle server by CUDA-worker PID `21644`; its ancestry `11244 -> 26752 -> 12568 -> 11364 -> {21644,18100}` cascaded out. Port 30000 is free, the known PIDs are absent, no compiler/CUDA worker remains, and the GPU returned to ordinary display residency.
- Decision: the two-graph architecture is real and the exact bridge is functional, but the dense-race form yields no TPS gain. One narrower refinement remains: use FlashInfer categorical sampling with explicit graph-stable seed/offset tensors instead of refreshing 496K exponential values per cycle. If that does not beat the warmed control, close exact linear composition unchanged and move to target-kernel cost.

### 2026-08-16 23:55 PDT — categorical RNG removes dense refresh cost; composite still loses

- Replaced the two full-vocabulary exponential-race buffers with FlashInfer categorical sampling. The raw parent reads one stable int64 seed and a per-depth offset tensor; one tiny offset add before replay advances both proposal streams. A new raw-child CUDA test proves same seed/offset replay is deterministic, advancing the offset changes the 32-row sample, and every returned probability equals the exact captured q gathered at the sampled token.
- CPU proposal/runner/device-cycle suites passed **21 tests**. The focused FlashInfer proposal plus composite native CUDA command passed **5 tests**. Python compilation and `git diff --check` passed before launch.
- Relaunched the same full 200K configuration with only `-SpeculativeDeviceResidentCycle`. Startup target/draft/draft-extend captures were **43.87**, **56.31**, and **1.05 seconds**, with 1.72 GiB reported graph-end headroom. A short exact `6213+64` smoke completed in 25 cycles at 2.56 emitted/cycle.
- Five exact real samples were `115.058, 116.444, 120.530, 123.907, 124.434 tok/s`, mean **120.075**, median **120.530**, **3.767% below** the matched 124.775 control. Five acceptance probes were `2.255507, 2.265487, 2.255507, 2.275556, 2.337900`, mean **2.277991**, **3.322% above** control.
- Normalized request evidence closes the path: categorical probes spent 23.753 seconds over 1,124 verify cycles (**21.132 ms/cycle**); dense races spent 24.235 seconds over 1,141 (**21.239 ms/cycle**); the ordinary linear probes spent 24.157 seconds over 1,163 (**20.771 ms/cycle**). The small-RNG refinement worked, and the composed execution path remained slower.
- Stopped the verified server by CUDA-worker PID `32516`; ancestry `34084 -> 37812 -> 27832 -> 18100 -> {32516,16864}` cascaded out. Port 30000 is free, all known PIDs are absent, and the GPU returned to display-only residency.
- Decision: close exact linear device-cycle composition for throughput and preserve it only as opt-in exact architecture. Production defaults stay unchanged. The next active uncertainty is graph-specific target execution cost; no topology qualifies for promotion.

### 2026-08-17 00:03 PDT — M3 target profile establishes the current device-cycle ceiling

- Began from local `main` at signed `746f135a9dda50dceaae9f67b56b6320958186ae`, three commits ahead and 67 commits behind the freshly fetched `origin/main`. The measurement deliberately remains on this exact local source line; no upstream merge or rebase occurred. Production defaults stayed unchanged. The profile server used the immutable RadixArk checkpoint, real 200K context, 16K active pool for headroom, one request, seed `783025237`, ordinary two-step/three-row top-k-one rejection sampling, aligned draft top-k 20, torch compile `default`, and the device-resident cycle disabled.
- Ran `scripts/windows/bench_target_verify_width.py --width 3` for exact `6213/128` sampled work after warmup/cache flush. The request emitted 128 tokens in 60 verification cycles, **2.133333 tokens/cycle**, histogram `[20,13,27]`, 1.86046 seconds end to end, digest `b6e0...`. Trace `benchmark/windows/profiles/target_width_m3-20260817-000307/target_width_m3-1786950187.0744395-TP-0.trace.json.gz` has SHA-256 `01a113fa2e8aed1bee57a15fd3b02a718afafd712504722dd295233a1a694e92`.
- Graph IDs are stable over 61 replays: target graph 2 **15.322/14.661 ms mean/median**, draft graph 5 **1.217/1.216 ms**, and draft-extend graph 8 **1.063/1.058 ms**. Profiler transition means/medians are target->extend 0.972/0.750 ms, draft->target 0.223/0.137 ms, and extend->draft 0.684/0.579 ms.
- Added raw target-start-to-target-start extraction. Sixty full device cycles are `17.912555..21.900988 ms`, mean **19.446434**, median **19.432881**. A depth-two chain emits at most three tokens, so even a perfect target-aware policy reaches only **154.270 TPS** at mean cost and **167.480 TPS** at the single best observed cycle. The current M3 geometry is mathematically closed as a 200-TPS topology.

### 2026-08-17 00:42 PDT — asynchronous timestamps close graph-tail work

- Added opt-in `--speculative-graph-gap-timing-path` / `--speculative-graph-gap-timing-max-samples` plumbing and `graph_gap_probe.py`. The probe records CUDA events immediately around the actual raw target, draft, and draft-extend graph boundaries, queries completion asynchronously, and writes through a bounded background JSONL queue. The default-disabled path creates no probe/writer. `analyze_graph_gap_timing.py` requires at least 20 samples, bounded median absolute deviation and p10-to-p90 spread, and conservative p10 >=0.75 ms.
- Launched one exact M3 diagnostic server rooted at PID `29568`; listener/CUDA leaf PID `31140` owned port 30000. Resolved runtime remained real 200K context, 16K active pool, seed `783025237`, ordinary rejection sampling, compile `default`, active `sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2`, and no device-resident cycle. Startup provenance appeared at stderr line 272 and `/server_info` returned the same active worker and compile mode.
- Two independent 512-token acceptance windows produced 221 cycles at **2.316742** emitted/cycle with histogram `[55,42,124]`, then 246 cycles at **2.081301** with histogram `[82,61,103]`. Artifact `benchmark/windows/profiles/m3_graph_gaps_20260817_0042.jsonl` contains **1,471** records and has SHA-256 `4c7797ae1cf70694994b10fb2d9936543f3e415c1a5ecb2a96174dddf2b7c819`.
- The best repeatable transition is target->draft-extend: conservative p10 **0.658355 ms**. Extend->draft has median 0.504928 ms and p10 0.474054 ms, with excessive p80 spread; draft->target is approximately 0.09-0.10 ms. The 0.75 ms funding gate fails. The user accepted the two-window 0.658 ms result as closing graph-tail work.
- Stopped the verified listener/CUDA leaf `31140`; the root tree exited. Confirmed the known PIDs absent, port 30000 free, no `cl`/`nvcc`/`ninja` process, and only ordinary WDDM/display clients on the RTX 5090.

### 2026-08-17 00:46 PDT — branch-exact p/q capture and fail-closed replay boundary

- Extended the offline sparse p/q oracle with repetition penalty state. The active speculative/overlap transform contract is accumulated additive presence/frequency first, then one sign-aware repetition scaling for every token with a positive branch-local count, followed by temperature, top-k, and top-p. An independent sequential small-vocabulary reference covers repeated-token paths and traversal-order independence.
- Added immutable schema-v2 replay for current deterministic, aligned deterministic, irregular variable-fanout, scalar/learned depth calibration, integrated SWOR, confidence-gated two/three-step chain, and the impossible target-aware oracle. Throughput is `1000 * sum(E[L]) / sum(full-cycle ms)`. The explicit measured current membership defines a frontier; a geometry candidate must have complete lattice coverage, a conservative lower TPS strictly above the frontier's best-case upper, and at least 215 TPS before implementation funding. Counterfactual policies fail closed on missing lattice nodes/support.
- Added opt-in runtime `pq_diagnostic.py`. Raw target and draft logits survive graph output only while capture is enabled; the existing pinned async D2H result fence carries them to a bounded JSONL writer. Each record contains exact edge IDs, child/parent IDs, token, depth, branch rank, current score, post-transform p/q, support masses, initial and branch-local counts, penalty configuration/order, topology membership, request/cycle IDs, active worker, and actual compile mode.
- Launched one M3 diagnostic server rooted at PID `25228`, with scheduler process `26436` and listener/CUDA leaf `33780`. Resolved runtime was the same 200K/16K, seeded, compile-default production geometry with only p/q capture enabled. `bench_spec_acceptance.py --output-tokens 16` emitted 16 tokens in six verify cycles (**2.666667/cycle**, histogram `[0,3,3]`, 0.77138 seconds).
- Artifact `benchmark/windows/profiles/m3_pq_capture_20260817_0046.jsonl` contains exactly six records, SHA-256 `f87c0bf9b0d91c920dba3735823c05ee86cbdb3b30f724d9d4014a4ce629f588`. The first record has edges `0: 0->1` and `1: 1->2`, depths 1/2 and rank zero, exact transformed support maps, and branch-local counts. It explicitly declares `capture_scope=selected_tree`; alternate descendants and later support are incomplete. Only the observed membership is decision-capable. Every counterfactual geometry remains unavailable.
- Stopped exact listener/CUDA leaf `33780`; PIDs `25228`, `26436`, and `33780` are absent. Port 30000 is free, compiler workers are absent, and the GPU is back to display-only residency.

### 2026-08-17 01:17 PDT — exact M/N/K target attribution and width frontier

- Added `scripts/windows/analyze_target_graph_gemms.py` and focused synthetic tests. Replays are grouped by positive CUDA graph ID plus launch correlation. Ordinary graph-ID-zero kernels are excluded. The analyzer builds the dense Qwen3.5 projection contract from the immutable model config, matches every primary GEMM independently within FP8/NVFP4/BF16 families on every replay, and marks model-role attribution unavailable on any count drift.
- Each launch and mathematical problem shape reports summed residency, all-stream interval union, serialized residency on the stream containing the terminal graph kernel, and exclusive observed-wall exposure. This preserves overlap: aggregate kernel time may exceed graph wall time and does not become a projected saving without full-cycle remeasurement.
- M3 matches exactly **305 primary GEMMs/replay** across all 61 target replays. They total **13.086192 ms aggregate**, **12.360049 ms terminal-stream**, and **11.821001 ms exclusive observed wall** inside the 15.321986 ms mean target span. Exact shape ranking is: NVFP4 MLP gate/up `3x34816x5120` **4.211372 ms** terminal-stream; FP8 GDN qkvz `3x16384x5120` **2.851188 ms** terminal-stream / **1.675006 ms** exclusive; NVFP4 down `3x5120x17408` **2.328160 ms**; FP8 output projections `3x5120x6144` **1.483579 ms**; FP8 full-attention qkv `3x8192x5120` **0.946352 ms**; NVFP4 lm-head `3x248320x5120` **0.539398 ms**. BF16 GDN BA `3x96x5120` uses 0.726143 ms aggregate and only **0.000081 ms** exclusive wall because it overlaps the qkvz stream.
- Proposal graph attribution is also exact at graph boundaries. Draft graph 5 spans **1.216837 ms** and is dominated by five BF16 GEMVs (0.515453 ms) plus one NVFP4 GEMM (0.441165 ms). Extend graph 8 spans **1.062720 ms** and is dominated by five BF16 GEMMs (0.561803 ms) plus one NVFP4 GEMM (0.432317 ms). Full cycle minus target span is roughly **4.124 ms**, including both proposal graphs and transitions.
- Replayed the analyzer over retained post-change width traces. M8 full cycles average **27.745758 ms**; corrected M12 averages **29.416496 ms**; M16 averages **31.877243 ms**. Their long-window emitted lengths are 2.737145, 2.905795, and 3.061001, yielding trace-cost projections of **98.651**, **98.781**, and **96.025 TPS**. All use four speculative steps, so the perfect five-token best-sample ceilings are **185.782**, **179.547**, and **166.666 TPS**. Each measured width is rejected by the impossible oracle before stochastic proposal uncertainty.
- Generated attribution reports beside the M3/M8/M12/M16 traces. The current M3 report has SHA-256 `fb27a0ab703711a4629e1bff0d75f02d4fa33049a79d5a69eab60d72a8333d06`. Inspection also confirms the MLP activation/quantization work is already represented by tiny compiled Triton fusion kernels, FP4 CUTLASS tactics are present in the retained FlashInfer autotune cache, and the GDN BA projection is already hidden. The next kernel candidate must attack the exposed NVFP4 gate/up/down or FP8 projection path and must change the measured full-cycle frontier.
- Focused validation at this checkpoint: target attribution tests **6 passed** after adding the dedicated attention-gate family; replay tests **11 passed** with conservative frontier and incomplete-lattice cases. No server or CUDA compiler was started for the offline trace analysis.

### 2026-08-17 01:26 PDT — MiaAI-Lab vLLM 160-TPS claim traced to a different K+1 architecture

- Located the exact external repository: `https://github.com/MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090`, initial commit `18c55e20f6d23c897085906215e41aa2bf276960`. It uses the same `RadixArk/Qwen3.8-27B-NVFP4` checkpoint class and a single RTX 5090, and claims approximately **160 tok/s** single-stream generation at full 262,144 context.
- Exact serve architecture is vLLM `0.27.1`, V1 engine, Flash Attention v2, `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`, `--kv-cache-dtype turboquant_4bit_nc`, a fixed 5.5 GiB KV pool, `--max-num-seqs 1`, and `--max-num-batched-tokens 512`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required for MTP-3 residency. The FlashInfer sampler is disabled in favor of vLLM's native sampler.
- The repository backports open vLLM PR `#40914`. Stock vLLM 0.27.1 captures TurboQuant K+1 verification through a context-blind first-chunk FULL graph, so replay ignores prior KV and garbles 13/15 tests. The patch builds synthetic sequence metadata entirely on GPU and routes uniform K+1 verification through `triton_turboquant_decode_attention`; MiaAI reports 0/15 failures after the patch. The upstream PR's separate A5000/TP2 test measured 57.2 -> 75.6 tok/s (+32%) when full graph execution was restored.
- The crucial geometry difference is now explicit. MiaAI proposes three speculative tokens and verifies four target rows. The qualified SGLang path proposes two steps and emits at most three tokens/cycle. At the measured 19.446 ms M3 cycle, raising only the path-length ceiling from three to four moves the impossible limit from **154.270 to 205.693 TPS**. A 160-TPS result therefore requires about 3.11 emitted tokens/cycle at the same cycle cost, which is plausible for an easier/greedy acceptance profile.
- Benchmark comparability remains unresolved. The MiaAI repository publishes no benchmark command, prompt/output lengths, raw samples, sampling parameters, emitted length, or cycle count. Its documented examples and 26K warmup use temperature zero with thinking disabled. The ~160 claim is architecture evidence and cannot replace the exact `6213/512`, temperature 1.0, top-p 0.95, top-k 20, presence 1.5, preserved-thinking production contract.
- Decision: this is material enough to supersede another speculative SGLang kernel rewrite as the immediate information gate. Reproduce the vLLM MTP-3/TurboQuant/K+1 recipe against the exact client workload, then choose a vLLM production lane or a narrow SGLang architectural port only from matched cycle, acceptance, behavior, and capacity evidence.

### 2026-08-17 01:36 PDT — distinct-weight projection graphs fund selective target NVFP4

- Added `scripts/windows/bench_target_projection_quantization.py`, an exact-shape admission probe for the three exposed target FP8 projection families. It compares the checkpoint's static-FP8 quantization plus FlashInfer cuBLAS BMM with runtime NVFP4 quantization plus CUTLASS FP4 GEMM. QKVZ savings are discounted by its measured M3 exclusive/serialized exposure ratio.
- The initial eager-loop projection of 4.887768 ms was rejected because Python launch gaps dominated. Two one-operation CUDA-graph windows then projected 0.619317/0.624054 ms, but those were also rejected: replaying one weight keeps it resident in the RTX 5090's large L2 and does not model production's distinct per-layer weight stream.
- Corrected each family graph to hold one distinct FP8 and NVFP4 weight per production occurrence: 48 QKVZ, 64 output, and 16 full-attention QKV matrices. This reproduces the production family residency closely: first-window FP8 totals were 2.873040, 1.892228, and 0.649521 ms versus trace totals 2.851188, 1.483579, and 0.946352 ms.
- First distinct-weight window (`--iterations 64 --rounds 9`) measured FP8/FP4 medians of 2.873040/1.628330 ms for QKVZ, 1.892228/0.996203 ms for output, and 0.649521/0.300328 ms for full-attention QKV. Its overlap-adjusted saving was **1.976456 ms**.
- Independent window (`--iterations 128 --rounds 9`) measured 2.862755/1.628811 ms, 1.788045/0.968205 ms, and 0.598108/0.277635 ms. Savings were 0.724913, 0.819841, and 0.320473 ms after overlap adjustment, totaling **1.865227 ms**.
- The lower projection gives a 17.580773 ms M3 cycle: **170.641 TPS** at a perfect three-token ceiling and **227.521 TPS** at a perfect four-token K+1 ceiling. This clears the 0.75 ms implementation gate and 215-TPS geometry funding floor. Decision: fund a distinct derived checkpoint that converts all three FP8 projection families, then require full-cycle and quality evidence.
- Both source checkpoints remain untouched. The server stayed stopped; port 30000 was free; no compiler workers overlapped the probe.

### 2026-08-17 01:47 PDT - selective target NVFP4 artifact built with bounded resharding

- Added `scripts/windows/build_selective_target_nvfp4_checkpoint.py`. It derives the selection directly from RadixArk's 208 `FP8` `quantized_layers`, requires the donor to expose exactly `input_scale`, `weight`, `weight_scale`, and `weight_scale_2` for every base, preserves every other RadixArk tensor, and rewrites the selected quantization declarations to NVFP4 group size 16.
- The first whole-shard build was terminated under host-memory pressure before publishing any shard. A bounded 2-GiB reshard attempt wrote two reproducible partial shards, then exposed a Windows mapping lifetime bug: tensors returned from `safe_open` did not own storage after the handle closed. Those two generated shards were removed; no unique data was lost. The loader now clones each tensor before closing its source mapping.
- Built `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4` from immutable RadixArk base plus immutable Gittensor NVFP4 donor. The output has 10 shards, 2,402 tensors, 208 transplanted projection bases / 832 selected tensors, and `selective-nvfp4-manifest.json` with per-shard SHA-256. Index audit found exactly 2,402 indexed and stored tensors, zero missing/extra tensors, and zero wrong-shard mappings.
- Output index SHA-256 is `f694aa7216ee4adf9895326ea706e40dd2426c83372f45c05416b48230aaa4ae`; quant config SHA-256 is `302d028778a8da5954458de596d48fff8b5beadfc36ba14f4808ed669b71999b`. Source checkpoints were not mutated.
- Began the first full 200K launcher-default load with only `-ModelPath C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4` changed. Load/capture result is pending.

### 2026-08-17 01:55 PDT - selective target NVFP4 loads and clears the measured cycle gate

- The first full load failed because the derived artifact updated `hf_quant_config.json` while retaining the base checkpoint's complete `config.json.quantization_config`. SGLang correctly preferred the complete runtime map and allocated fused `in_proj_qkvz` as FP8 `(16384, 5120)`, then rejected the donor's correctly packed NVFP4 shard `(2048, 2560)`. Shape-rich diagnostics identified `param_attr='weight'`, logical shard 0, source `(10240, 2560)`, and logical split sizes `[2048, 2048, 6144]`; no serialized tensor defect was found.
- Extended `build_selective_target_nvfp4_checkpoint.py` so `config.json` and `hf_quant_config.json` are rewritten together. The selected 208 projection bases move from the FP8 config group to the NVFP4 group and become `NVFP4/group_size=16` in both per-layer maps. A bounded `--resume-metadata-only` path validates the existing tensor map and every shard before repairing metadata.
- Repaired `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4` without rewriting any tensor shard or the index. All ten recorded shard hashes, the index hash `f694aa7216ee4adf9895326ea706e40dd2426c83372f45c05416b48230aaa4ae`, and the standalone quant hash `302d028778a8da5954458de596d48fff8b5beadfc36ba14f4808ed669b71999b` remain unchanged. The synchronized `config.json` hash is `88cb667373c8556f13fef4813cc9ba32c15a061d7867a6ca2b04cc082fa03ce1`.
- Focused CPU validation passed **11 tests**: the ten existing Qwen3.5 packed-loader cases plus a new runtime-manifest synchronization regression. Python compilation and `git diff --check` passed.
- Relaunched the exact 200K launcher configuration with only the derived model path changed. The live endpoint reports ready, exact `context_length=max_total_tokens=max_total_num_tokens=200000`, target/draft/extend graph captures of **24.686/1.603/1.748 s**, weight residency **16.207 GiB**, startup available memory **4.314 GiB**, and all experimental topology/device-cycle controls inactive. `/model_info` reports image/audio false. Listener PID snapshot is `34172`; re-establish the full ancestry before stopping it.
- Short preserved-thinking arithmetic smoke returned coherent reasoning and final content `703` with `finish_reason=stop`.
- Exact M3 profile is `benchmark/windows/profiles/target_width_m3-20260817-015217/target_width_m3-1786956737.5260205-TP-0.trace.json.gz`. It emitted 128 tokens in 53 verifications (**2.415094/cycle**) and recorded 53 target-start cycles: **17.314950 ms mean**, **17.197624 ms median**, **15.492377 ms minimum**, **19.894875 ms maximum**. Target graph span fell to **13.126 ms mean** from the prior 15.322 ms; full cycle fell by **2.131484 ms / 10.96%** from 19.446434 ms. Perfect three-token and four-token ceilings are approximately **173.260** and **231.014 TPS**. The selective conversion therefore beats its 1.865 ms admission projection and keeps the K+1 route above the 215-TPS funding floor.
- `target_graph_gemm_attribution.json` was generated beside the trace. Its old role contract fails closed because it requires 128 FP8 primary GEMMs; the converted graph correctly has zero. Update that analyzer contract before using role-level post-conversion attribution.
- Trace-integrity anomaly remains open: the trace is 7,656,253 bytes, starts with a valid gzip header, and parses fully, yet Python hashlib, PowerShell `Get-FileHash`, CertUtil, Node crypto, and the generated manifest all report the empty-input SHA-256 `e3b0...`. Small-buffer SHA-256 works. Treat the recorded trace hash as invalid until this cumulative-hash/environment anomaly is understood; retain the trace and do not delete or rewrite it.
- Current server is deliberately left live for the ordinary sampled window. No sampled `6213/512` throughput window, tool-call check, exact `199000+16`, or OpenCode2 check has yet run on this checkpoint. Production defaults and the qualified RadixArk line remain unchanged.

### 2026-08-17 02:00 PDT - selective target NVFP4 admission window one: 131.707 TPS mean, tool gate passed

- Handoff continuation began on local `main` at signed `746f135a9dda50dceaae9f67b56b6320958186ae` (ahead 3, behind 67). The selective-NVFP4 server was confirmed live: port 30000 owner remains listener PID `34172`; full ancestry is `10464 opencode2.exe -> 33680 pwsh launcher -> 25264 sglang.exe -> 9280 python.exe -> 34172 listener`, with children `26020` CUDA multiprocessing worker, `13296` detokenizer, and `9584` conhost. `/server_info` confirmed ready, exact 200000 context/pools/capacity, graph captures 24.686/1.603/1.748 s, weight residency 16.207 GiB, startup available 4.314 GiB, seed `15962589`, all topology/device-cycle experiment controls inactive, `/model_info` image/audio false. No compiler workers were present.
- GPU at window start: 31 C, 74.34 W, 13% utilization, SM 2715 MHz, memory clock 13801 MHz, 29287 MiB used, 2901 MiB free. Active WDDM clients included Chrome, Edge WebView2, Docker Desktop, Claude, SnippingTool, and ordinary shell/display processes; the server tree owns the CUDA compute residency.
- Ran one `flush_cache` POST, then five consecutive exact sampled `6213/512` runs (temperature 1.0, top-p 0.95, top-k 20, presence 1.5, thinking enabled). Decode TPS samples: `130.403, 134.384, 130.824, 136.749, 126.173`; mean **131.7066**, median **130.824**. E2E output TPS: `114.865, 117.567, 115.176, 119.726, 111.143`. TTFT s: `0.538774, 0.552435, 0.539370, 0.539656, 0.556681`. E2E s: `4.457401, 4.354974, 4.445387, 4.276428, 4.606670`. Every request returned prompt 6213, completion 512, total 6725, `finish_reason=length`. Reasoning/content chars: `1072/1063, 2370/0, 2410/0, 1814/529, 2401/0`; reasoning remained coherent and preserved. Sample output SHA-256s: `0aa5af82…, 117cd936…, 1c93c781…, a280997e…, 8584d47e…`.
- Interpretation: mean 131.707 is **+5.556% over the 124.775 matched fresh control** and +7.330% over the qualified 122.712. This is an experimental admission window only; promotion requires a second independent sampled window. The measured decode rate agrees with the measured acceptance x cycle-cost projection: at 2.217 emitted/cycle and 17.315 ms/cycle the width-three geometry caps near 128 TPS ordinary / 173.260 TPS perfect, so K+1 remains required for 200.
- Five consecutive `bench_spec_acceptance.py` probes recorded emitted/cycle `2.216450, 2.275556, 2.178723, 2.226087, 2.188034`; mean **2.216970** over 1,155 verification cycles. Proposed/correct drafts were 2310/1402; aggregate histogram `[308, 292, 555]`. Acceptance is slightly below the historical 2.318174 reference but in line with the matched control's 2.204748 window, so TPS tracks execution cost as expected. Probe digests: `3809eef7…, 14332f35…, 2adafd10…, 4a9d77c9…, a957169b…`.
- Tool-call gate passed on the selective checkpoint: exactly one parsed call, function `multiply`, arguments `{"a": 37, "b": 19}`, `finish_reason=tool_calls`, preserved 128-char reasoning, 75 completion tokens, prompt 344 / total 419. Combined with the earlier `703` arithmetic smoke, both semantic parser gates are green.
- Post-window GPU snapshot: 34 C, 79.98 W, 9% utilization, SM 2347 MHz, 29287 MiB used, 2901 MiB free; after the tool probe, idle SM fell to 847 MHz with 29630 MiB used / 2558 MiB free. Free VRAM pressure matches the qualified production profile (a few hundred MiB of steady free), so the capacity gate must recheck memory before/after/flush.
- Remaining before this candidate can advance: exact `199000+16` capacity with memory snapshots, a second independent sampled window, standalone OpenCode2 integration, and unsimulated production-style relaunch evidence. The attribution analyzer's mixed-precision role contract update remains secondary. K+1 (three proposal steps / four target rows) is the funded next geometry from the measured 17.315 ms cycle and 231.014 TPS perfect-four ceiling.
- Production defaults, the qualified RadixArk checkpoint, and the qualified 122.712 TPS line remain unchanged. The live selective server stays up for the capacity gate.

### 2026-08-17 02:11 PDT - INCIDENT selective checkpoint crashes the scheduler on long prefill

- First crash (server instance 1, handoff tree): the live server died between a successful `flush_cache` at 02:05:03 and the next request at 02:05:20. Port 30000 was free, PIDs `34172/9280/25264/33680/26020/13296` absent, GPU back to display residency (2054 MiB used). No Windows Application-log error event was recorded; the parent harness (opencode2) survived. Death cause unresolved because the first launch left no captured stdout.
- Second instance relaunched at the user's request with the identical command. New tree: `10464 opencode2.exe -> 34300 pwsh -> 16872 sglang.exe -> 16184 python.exe -> 10880 listener`, CUDA workers `33932` and `12700`. Startup graph captures: target verify 15.46 s, draft decode 1.09 s, draft extend 0.81 s; startup available 3.73 GiB. Warmup `6213/64` returned 156.979 decode TPS.
- Startup emitted one Triton CompilationError in the ReplaySSM GDN prefill kernel variant (`MAX_CACHE_LEN=3, CACHE_RING=True, DISABLE_STATE_UPDATE=True`): `AssertionError: Loop-carried variable b_h has initial type <[128,32], fp32> but is re-assigned to <[128,32], fp64> in loop`. The server nevertheless completed capture and served decode correctly; retain this as suspect evidence, not an explanation.
- Exact `199000+16` capacity attempt crashed the server: chunked prefill advanced cleanly through 4096-token chunks to about 55% (`pending-token: 92504`, full utilization 0.55, prefill throughput degrading normally from 10215 to 2626 tok/s), then at 02:10:03 `Subprocess scheduler_0 (pid=33932) crashed with exit code 4294967295` (native -1, no Python traceback). The client saw `ConnectionResetError`. Port freed; GPU returned to 2206 MiB display residency.
- Interpretation: decode, short prefill (6213), tools, and graphs are all verified good on this checkpoint; long-context prefill is a hard crasher. The qualified RadixArk checkpoint passes this exact capacity gate, so the selective NVFP4 conversion is implicated. Next: restart and localize with the contract long ladder (`32768/16`, `65536/16`) before repeating the 199K gate; if reproducible, relaunch under `CUDA_LAUNCH_BLOCKING=1` at the failing length to capture the offending kernel. Capacity gate status: FAILED (crash), selective checkpoint promotion blocked on this.
- **Correction (2026-08-17 12:56 PDT):** the "crash" was an external kill, not a server fault. A concurrent supervised session stopped exact CUDA leaf `33932` on the user's `kill it` request at the same instant; killing scheduler_0 produces the observed `crashed with exit code 4294967295` and SIGQUIT cascade. Instance 1's idle exit is likewise attributable to external action rather than a checkpoint defect. The capacity gate subsequently passed cleanly (next entry).

### 2026-08-17 02:05 PDT - harness-owned selective server exits before capacity request

- Prepared the exact capacity gate from the still-live width-three selective server. The pre-gate snapshots were 29,619 MiB used / 2,569 MiB free at 28 C, then 29,615 MiB used / 2,573 MiB free after an eight-second idle wait. A `POST /flush_cache` returned the normal success message.
- The subsequent exact command, `.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600`, failed during its first `/v1/tokenize` connection with `ConnectionRefusedError [WinError 10061]`. No prompt or generation work began, so this is **not** a capacity result.
- Immediate inspection found no listener on port 30000 and every known SGLang PID absent: launcher `33680`, shim `25264`, parent Python `9280`, listener `34172`, CUDA worker `26020`, and detokenizer `13296`. GPU residency had returned to 2,054 MiB used / 30,134 MiB free at 29 C. No matching Python/SGLang application-error event appeared in the recent Windows Application log.
- The launch had been attached to an earlier background harness shell and did not retain a dedicated stdout/stderr file. Evidence therefore establishes a complete process-tree exit between the successful 02:05:02 GPU snapshot and the client's connection attempt; it does not distinguish a harness-lifetime termination from a server fault. Preserve that uncertainty. Relaunch through a dedicated hidden process with explicit stdout/stderr paths before retrying capacity.
- Removed the temporary shape-rich wrapper from `qwen3_5.py`, restoring direct packed-loader delegation and eliminating the temporary `param_attr` plumbing. The combined builder plus packed-loader suite passed **11 tests**; all three touched Python files compiled; the focused `git diff --check` passed; and `git diff -- python/sglang/srt/models/qwen3_5.py` is empty. The artifact-builder metadata synchronization remains the sole loader repair.

### 2026-08-17 02:10 PDT - unexpected supervised relaunch identified and stopped on request

- Before starting the planned explicit-log relaunch, port 30000 unexpectedly became occupied again. The replacement process had been created at 02:07:18-02:07:19 by the same external OpenCode2 service parent, not by the planned command: `10464 opencode2.exe -> 34300 pwsh.exe -> 16872 sglang.exe -> 16184 python.exe -> 10880 listener`, with leaf children `12700` and CUDA worker `33932` plus console host `29224`.
- The live command was the unchanged selective width-three launch with model `Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4`, 200K context/pools, two speculative steps, three target rows, ordinary rejection sampling, and no device-resident cycle. `/server_info` reported ready with a new implicit seed `615388882`; `/model_info` remained language-only. During the launcher's warm work the RTX 5090 reached 520-533 W, 96-100% utilization, and 29,594 MiB residency.
- On the user's explicit `kill it`, re-read listener PID `10880`, re-resolved the complete ancestry, and stopped exact CUDA leaf `33932` first followed by exact leaf `12700`; the remaining identified tree cascaded out. All known PIDs are absent, port 30000 remained free after a five-second restart check, `cl`/`nvcc`/`ninja` are absent, and the GPU returned to 2,110 MiB used / 30,078 MiB free at 35 C. No unrelated Python, PowerShell, OpenCode2, MCP, WSL, or desktop process was stopped. Leave the server stopped.

### 2026-08-17 12:58 PDT - reconciliation: capacity gate passed cleanly; K+1 lane opens

- Cross-session reconciliation: the 02:10:03 `scheduler_0 crashed with exit code 4294967295` during the first capacity attempt was the concurrent session stopping exact CUDA leaf `33932` on the user's `kill it` request, not a selective-checkpoint fault. Instance 1's idle exit after a flush is likewise externally attributed. The long-prefill crash hypothesis is closed; no `CUDA_LAUNCH_BLOCKING` diagnostic is warranted.
- A third instance (relaunched with the identical selective width-three command) passed the full localization ladder and the exact capacity gate before its harness shell ended: `32768/16` 8887.996 prompt TPS; `65536/16` 6181.925; `131072/16` 3746.628; then exact `199000+16` returned **199016 total tokens**, `finish_reason=length`, TTFT 70.096 s, prompt **2838.980 tok/s** (qualified reference 2608.263), generation **107.253 tok/s** (reference 102.358), e2e 70.235 s. No crash, no retraction.
- Capacity memory snapshots: immediately after the 199016 request 29,248 MiB used / 2,940 MiB free at 39 C; after `flush_cache` 28,058 MiB used / 4,130 MiB free. Operating headroom is consistent with the qualified profile.
- The temporary `qwen3_5.py` loader diagnostic was removed (parallel-session evidence, verified here: `git diff -- python/sglang/srt/models/qwen3_5.py` is empty, `git diff --check` clean); the builder metadata synchronization remains the sole loader repair and its 11-test suite passed.
- Candidate status: the selective width-three checkpoint has now passed loading, graphs, model surface, arithmetic, tool parsing, cycle funding, ordinary window one (131.707 TPS), acceptance, and exact capacity. Still required for any promotion: second independent sampled window, OpenCode2 integration, unsimulated production-style relaunch evidence, and — decisively — a geometry that can reach 200 TPS.
- K+1 semantics verified from source: for the top-k-one NEXTN chain, `speculative_num_draft_tokens` is the target verify width and draft-extend fills that width per request (`eagle_worker_v2.py` lines 1155-1164), so launcher args `-SpeculativeNumSteps 3 -SpeculativeNumDraftTokens 4` produce three proposals plus one bonus row: four target rows, maximum four emitted tokens per cycle. Perfect four-token ceiling at the measured M3 17.315 ms cycle is 231.014 TPS; the actual M4 cycle must be measured because the fourth target row and third draft step add cost. Reject immediately if the measured four-token ceiling falls below 215 TPS.
- Next: relaunch the selective checkpoint with only the two shape arguments changed, warm, capture the exact M4 device cycle with `bench_target_verify_width.py --width 4`, then measure ordinary acceptance and TPS.

### 2026-08-20 08:17 PDT - exact 200K capacity throughput made the primary scoreboard

- At the user's direction, created root `BENCHMARK.md` and made the exact `199000+16` near-limit workload the primary performance scoreboard for this lane.
- The record to beat is the selective target-NVFP4 result: **2838.980 prompt tok/s**, **107.253 generation tok/s**, **70.096 s TTFT**, **70.235 s end to end**, and exact **199016** tokens with `finish_reason=length`.
- The qualified RadixArk production reference remains **2608.263 prompt tok/s**, **102.358 generation tok/s**, **76.442544 s** end to end, and exact **199016** tokens.
- A new overall record must complete the same real 200K-pool workload and exceed both headline throughput figures under the matched command and environment record. The fuller behavior and promotion gates remain in `notes/benchmark-contract.md`.
- Reconciled the new primary scoreboard into `notes/current-state.md`, `notes/decisions.md`, `notes/benchmark-contract.md`, and `notes/timeline.md` so future handoffs rank candidates on the same user-selected workload.

### 2026-08-20 08:21 PDT - first active 200K performance milestone selected

- At the user's direction, set the current active target in root `BENCHMARK.md` to **3000 prompt tok/s** and **110 generation tok/s** on the exact `199000+16` workload.
- The equivalent supporting latency targets are **TTFT <=66.33 s** and **end-to-end <=66.5 s**. A successful milestone run must complete exact `199016` and meet both throughput targets together.
- This asks for approximately **5.7%** more prompt throughput and **2.6%** more generation throughput than the current selective target-NVFP4 record. Reconciled the target into the compact benchmark contract, current-state handoff, and decision ledger.

### 2026-08-20 08:57 PDT - clean current-source M3 baseline exposes material variance

- Began the resumed optimization run from clean `main` at
  `2eddaf4e8fd13911be3937df0d1f5f40583e4b4d`, synchronized with
  `origin/main`. No pre-existing modified or untracked paths were present.
  Port 30000 was free, no SGLang/CUDA/compiler tree existed, and the RTX 5090
  initially reported 31 C, 69.77 W, 1,777 MHz SM, 810 MHz memory, 24 percent
  utilization, 1,464 MiB used, and 30,724 MiB free. Installed versions are
  Python 3.13.14, PyTorch 2.13.0+cu130, CUDA runtime 13.0, Triton 3.7.1, and
  FlashInfer 0.6.17; the NVIDIA driver is 610.88.
- Launched one deliberate selective-checkpoint server through
  `scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1 -ModelPath
  C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4 -RandomSeed
  615388882`. The verified lineage was detached PowerShell `21704 -> 28808`,
  SGLang shim `43340`, Python parents/listener `47012 -> 26572`, scheduler/CUDA
  worker `16760`, and detokenizer `34624`; listener `26572` owned
  `127.0.0.1:30000`.
- Resolved runtime matched the intended baseline: exact
  `context_length=max_total_tokens=max_total_num_tokens=200000`, one request,
  chunk size 4096, page size 64, checkpoint-selected target KV, FP8 E4M3 draft
  KV, FlashInfer prefill/sampling, TRT-LLM MHA/XQA target/draft decode,
  ReplaySSM Triton linear attention, M3 top-k-one rejection sampling, draft
  top-k 20, FP32 Mamba with four slots and `extra_buffer_lazy`, torch compile
  default, batch-one full decode graphs, prefill graphs disabled, FP4
  autotuning with FP8 GEMM skipped, and every tree/adaptive/device-cycle
  control inactive. `/model_info` reported image/audio understanding false.
  Target, draft, and draft-extend captures took 31.04, 1.38, and 0.88 seconds;
  4.59 GiB was available at graph end.
- Ran five exact measured completions with
  `.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py
  --input-tokens 199000 --output-tokens 16 --timeout 600`. Run one retained
  the benchmark's exact-shape internal warmup; runs two through five used
  `--skip-warmup` on the already loaded/captured server. Every measured request
  flushed the cache, and every long-prefill server line reported
  `#cached-token: 0`.
- Prompt throughput was `2603.510, 2610.132, 2733.249, 2672.513, 2653.105
  tok/s`: mean **2654.502**, median **2653.105**, standard deviation 52.670,
  and CV 1.984 percent. TTFT was `76.435263, 76.241344, 72.807133, 74.461748,
  75.006454 s`; end-to-end time was `76.598970, 76.379112, 72.937741,
  74.640765, 75.184459 s`.
- Generation throughput was `91.627, 108.879, 114.847, 83.791, 84.268
  tok/s`: mean **96.682**, median **91.627**, standard deviation 14.358, and
  CV 14.850 percent. The post-first-token interval contains only 15 decoded
  tokens and an integer number of speculative verification cycles, so this
  spread is much larger than the active 2.6-percent generation target.
- All five requests returned exact `199016`, `finish_reason=length`, thinking
  enabled, and digest
  `9a0e20749e2930a697fefdd3bdd7863a067abe4d9860e6d1e7d9b80a62668b37`.
  During retained runs the GPU held 2.962-3.015 GHz SM and 13.801 GHz memory
  clocks at 496.79-525.90 W, 60-68 C, and 98-99 percent utilization. WDDM
  clients included Edge WebView, Windows shell/display, iCloud, PC Manager,
  and OpenCode2.
- Decision: the historical **2838.980 prompt / 107.253 generation tok/s**
  record remains the scoreboard but is not today's reproducible baseline.
  Use **2654.502/96.682 mean** and the complete raw distribution for immediate
  matched A/B work. Interleave controls with candidates; do not promote a
  single favorable 16-token generation hit. The server remains live for the
  first M4/K+1 comparison. Raw result, environment, stdout, and stderr files
  are under
  `C:\Users\Daniel\.copilot\session-state\f539b6f3-61df-4654-ab5a-6cb8c5c40957\files`.

### 2026-08-20 09:23 PDT - selective M4 K+1 rejected by matched cycle economics

- Stopped the verified M3 tree leaf-first and confirmed port 30000 free,
  compiler workers absent, and the GPU back at 1,324 MiB display residency.
  Launched the same selective checkpoint and seed `615388882` with only
  `-SpeculativeNumSteps 3 -SpeculativeNumDraftTokens 4` changed. The live
  endpoint resolved exact 200K pools, width four, top-k one, draft top-k 20,
  ordinary rejection sampling, full target/draft/extend graphs, and no
  device-resident/tree controls. M4 target/draft/extend captures took
  17.79/1.03/0.81 seconds with 4.58 GiB available at graph end.
- `bench_target_verify_width.py --width 4` produced trace
  `benchmark/windows/profiles/target_width_m4-20260820-090229/target_width_m4-1787241749.4595737-TP-0.trace.json.gz`,
  SHA-256
  `c2ea02706b7fdf90792f82163f537c2c2a2b78fb8cd92a3de608e014237d512f`.
  The exact chain accepted **2.327273** tokens/cycle over 55 cycles. Full
  target-start cycles averaged **18.419190 ms**, median 18.394597, minimum
  17.140989, and maximum 20.765297. Actual measured projection was
  **126.350 tok/s**; perfect-four mean ceiling was **217.165 tok/s**.
- Five exact M4 `199000+16` runs returned prompt `2653.695, 2792.130,
  2788.118, 2790.292, 2790.491 tok/s` and generation `117.545, 91.572,
  106.251, 92.061, 105.943 tok/s`. Excluding the first exact-shape warmup
  recovery, prompt mean/median were **2790.258/2790.491** with 0.059% CV;
  generation mean/median were **98.957/105.943** with 8.335% CV. All five
  completed exact `199016`, `finish_reason=length`, and the established
  `9a0e...8b37` digest.
- Stopped the M4 tree leaf-first and restored the same M3 control. M3 captures
  took 14.39/0.96/0.76 seconds with 4.61 GiB available. The matched trace
  `benchmark/windows/profiles/target_width_m3-20260820-091451/target_width_m3-1787242491.6518054-TP-0.trace.json.gz`,
  SHA-256
  `170ab49f24dce5ea69e45130b0133d92c42291c7303fcc23c3857deec5dbd7b6`,
  accepted **2.245614** tokens/cycle over 57 cycles. Full cycles averaged
  **16.058328 ms**, minimum 15.655271, for **139.841 tok/s** actual projection
  and 186.819 tok/s perfect-three ceiling.
- Five exact M3 A2 runs returned prompt `2797.957, 2789.956, 2787.745,
  2787.968, 2791.484 tok/s` and generation `99.306, 115.665, 100.016,
  100.035, 88.214 tok/s`. Warmed prompt mean was **2789.288**, while warmed
  generation mean was **100.982** with 11.152% CV. All five retained exact
  completion and the same digest.
- A-B-A conclusion: M4 acceptance improved only 3.636% while cycle cost rose
  14.702%, reducing projected throughput 9.647%. Warmed prompt changed only
  +0.035%, and exact generation did not improve outside noise. Reject plain
  SGLang M4 K+1 and keep M3 selected. The external vLLM TurboQuant/full-graph
  K+1 lane is architecturally distinct and remains an information gate.

### 2026-08-20 09:35 PDT - exact benchmark now fails closed and exposes SSE boundaries

- A skeptical read of the exact client found that its headline formulas are
  deliberately client-observed: TTFT ends at the first nonempty
  `reasoning_content` or `content` SSE fragment, and generation ends at
  response completion. Speculative token coalescing or delayed final metadata
  can therefore move the reported split without changing model work. It also
  found that `reasoning_content or content` omitted content from the digest
  when both fields appeared in one delta, and measured token counts were not
  enforced.
- Updated `scripts/windows/bench_openai_stream.py` without changing either
  headline formula. It now appends both output channels in event order,
  reports full/reasoning/content SHA-256 values, nonempty delta and per-channel
  fragment counts, first/max delta character counts, and time from the final
  output delta to response completion. Warmup and measured responses must
  exactly match prompt, completion, total-token, and `finish_reason=length`
  expectations. Added `--warmup-runs`; `--skip-warmup` still forces zero.
  Prompt calibration now rejects a requested target below the empty templated
  prompt length.
- Added
  `test/registered/unit/scripts/test_bench_openai_stream.py` using
  `CustomTestCase` and CPU CI registration. It covers a single SSE delta that
  contains both reasoning and content, all count/finish failure modes, and the
  below-template calibration boundary. The focused run passed **3 tests, 4
  subtests**; Python compilation, CLI parsing, and `git diff --check` passed.
- Live M3 smoke:
  `bench_openai_stream.py --input-tokens 256 --output-tokens 16 --timeout 120`
  completed exact `272`, returned three nonempty reasoning deltas, first/max
  delta sizes `2/39`, 0.000168 seconds of trailing response time, and matching
  full/reasoning hashes. Decision: retain this measurement guardrail before
  screening further small prompt or generation candidates.

### 2026-08-20 10:03 PDT - paged-only FlashInfer prefill rejected

- Took two instrumented exact M3 controls on the aged restored server before
  the candidate. Prompt was `2802.045, 2792.021 tok/s`; both completed exact
  `199016` with the established digest. Their 16-token generation rates
  `87.343, 88.725` accompanied three/four SSE output fragments, demonstrating
  why fragment telemetry and longer decode evidence are required.
- Stopped the verified M3 tree and relaunched the identical selective
  checkpoint and seed with process-scoped
  `SGLANG_FLASHINFER_USE_PAGED=1`. Startup provenance printed
  `SGLANG_FLASHINFER_USE_PAGED=True`; `/server_info` retained 200K pools, M3,
  FlashInfer prefill, XQA decode, and all other defaults. The server had 4.61
  GiB available after all three graph captures.
- After two full exact-shape warmups, paged-only `199000+16` prompt samples
  were `2790.384, 2782.369, 2781.207 tok/s`, mean **2784.653**. Generation
  samples were `114.675, 114.644, 114.877 tok/s`, but two/three-fragment SSE
  output and the short 15-interval window made that apparent gain
  non-decision-capable. The candidate's deterministic digest was
  `35dc6596...7dabd1`, different from default.
- Extended the same prompt to 512 output tokens. Paged-only prompt measured
  `2786.844, 2783.676 tok/s`; generation measured `104.514, 103.720 tok/s`;
  both runs produced digest `d2f8ad71...51cdf73`. A temperature-zero
  acceptance probe returned **1.976834** tokens/cycle over 259 verifications.
- Stopped paged-only and restored the default M3 server with the same seed.
  After the same two full warmups, the control `199000+16` prompt was
  **2796.116 tok/s**. Two `199000+512` runs measured prompt `2789.332,
  2788.740` and generation `108.022, 104.912 tok/s`; both produced digest
  `9ca9ea3b...25ee8`. Control acceptance was **1.961686** over 261
  verifications.
- Decision: paged-only changes long prompt by **-0.135%** and long generation
  by **-2.207%** despite a 0.772% acceptance increase. Reject it and retain the
  default ragged-current plus paged-prefix merge. Raw candidate/control JSONL,
  environment, stdout, and stderr files remain in the active session-state
  `files` directory.

### 2026-08-20 10:31 PDT - 6144-token chunks set a new matched prompt leader

- Began a fresh-server 4096/5120/6144 sweep on the selective M3 checkpoint.
  Every arm used seed `615388882`, exact 200K pools, two complete
  `199000+16` warmups, then three cache-flushed scored requests. GPU clocks
  held around 2.96-3.01 GHz SM and 13.801 GHz memory under full prefill load;
  every server reported 4.61-4.65 GiB after graph capture.
- The 6144 server resolved only `chunked_prefill_size=6144`; all M3,
  rejection, draft top-k 20, FlashInfer/XQA/ReplaySSM, quantization, compile,
  scheduling, workspace, and cache settings remained matched. Its prompt
  samples were `2943.285, 2939.119, 2940.310 tok/s`, mean **2940.905**. TTFT
  was `67.611530, 67.707359, 67.679937 s`, mean **67.666275**. E2E was
  `67.784737, 67.859864, 67.832433 s`.
- The 5120 server resolved only `chunked_prefill_size=5120`. Prompt was
  `2894.440, 2892.438, 2891.136 tok/s`, mean **2892.671**; mean TTFT was
  **68.794554 s**.
- Restored 4096 control prompt was `2795.255, 2790.685, 2793.024 tok/s`, mean
  **2792.988**; mean TTFT was **71.249895 s**. This final control reproduces
  the stable later M3 regime and confirms the earlier low 2654.502 window was
  startup/environment history rather than the comparison authority.
- All nine scored requests completed exact `199016`,
  `finish_reason=length`, and valid fragment/trailing telemetry. Chunk
  boundaries changed the deterministic trajectory: digest `9a0e...8b37` at
  4096, `a6bc...19ec` at 5120, and `3e01...2417` at 6144.
- Interim decision: 6144 improves prompt throughput **5.296%** over matched
  4096 and **3.590%** over the historical 2838.980 record, while remaining
  **1.970%** below 3000. Keep it as the leading candidate, not yet the
  launcher default. Test nearby 6656 and 7168 before long-generation,
  reasoning/tool, headroom, and production-relaunch qualification.

### 2026-08-20 11:44 PDT - selective 7680 profile clears prompt target; global default rejected

- Continued the selective-checkpoint sweep with identical seed, pools,
  backends, two full warmups, and three scored exact requests per fresh server.
  Chunk 6656 produced `2969.057, 2963.184, 2963.991 prompt tok/s`, mean
  **2965.411** and mean TTFT 67.107117 s. Chunk 7168 produced `2983.084,
  2975.322, 2982.743`, mean **2980.383** and mean TTFT 66.770039 s.
- Chunk 7680 first-window prompt was `3001.487, 2996.399, 2997.141`, mean
  **2998.342**. A launcher-default-style second independent window produced
  `3002.344, 2995.936, 2995.271, 2996.713, 2996.665`, mean **2997.386**.
  Across all eight, mean was **2997.744** and range **2995.271..3002.344**.
  Best TTFT/E2E were **66.281538/66.434400 s**. Every request completed exact
  `199016` and retained the original 4096 digest `9a0e...8b37`.
- A same-count 7808 refinement regressed sharply: `2912.697, 2909.720,
  2905.634`, mean **2909.350**. This closes upward refinement without
  reopening the previously rejected 8192 branch.
- Two exact selective `199000+512` requests at 7680 measured prompt
  `3004.324, 2999.159` and generation `110.693, 108.978 tok/s`, with exact
  `199512` completion and stable digest `1e90...e97f9`. This is supporting
  evidence, not the exact-16 scoreboard result.
- Selective 7680 behavior and short-work qualification passed. Arithmetic
  returned `703`; the tool parser emitted exactly one
  `multiply({"a":37,"b":19})` call with `finish_reason=tool_calls`;
  `/model_info` kept image/audio false. Post-long-request/headroom was 2,336
  MiB before explicit flush and 4,634 MiB after. Two independent real sampled
  `6213/512` windows were `137.173, 139.825, 135.042, 137.258, 143.387`
  (mean **138.537**) and `140.353, 139.883, 139.233, 142.658, 137.297`
  (mean **139.885**). Five acceptance probes averaged **2.245332**.
- Temporarily changed the launcher's chunk default to 7680 and checked its
  actual default checkpoint before committing. Base RadixArk sampled
  `6213/512` was neutral: two 7680 windows averaged **121.221** and
  **120.887**, combined **121.054**; a fresh 4096 ten-run control averaged
  **121.027**. Acceptance means were 2.268015 at 7680 and 2.218692 at 4096.
- The exact base-checkpoint capacity gate rejected global promotion. Base
  7680 completed exact `199000+16`, but reached only **2226.770 prompt /
  83.988 generation tok/s**, TTFT 89.367121 s, E2E 89.545717 s, and fell to
  **200 MiB free** before arithmetic/tool probes. Arithmetic and tools still
  passed; after flush, free VRAM recovered to 2,358 MiB.
- Decision: restore the production launcher default to 4096. Retain selective
  `AttnNVFP4 + chunk 7680` as an explicit long-context performance profile.
  It sets the independent prompt record and demonstrates long decode above
  110, but no exact `199000+16` run has yet met both headline targets
  together. Raw outputs and server logs remain in the active session-state
  `files` directory.

### 2026-08-20 11:54 PDT - selected-row draft-extend logits save memory, not time

- Implemented a narrow single-layer port of the multi-layer EAGLE
  `select_index` contract. Non-gathered, non-standalone, non-device-resident
  draft-extend graphs kept all hidden/KV rows but passed only the last accepted
  row into `lm_head`; the worker avoided a second logits gather. A graph-stable
  select-index buffer changed on replay. Device-resident and gathered paths
  retained full logits.
- Added a CPU white-box replay test for selection-index copying, pruned logits
  shape, and full hidden output, plus direct logits-processor row-selection
  coverage. The focused new/existing runner command passed **6 tests**.
  A broader unrelated top-k-one file was not usable in the plain shell because
  its native JIT lacked MSVC in `PATH` and several pre-existing `__new__`
  fixtures omit current worker fields; those failures did not enter the
  candidate decision.
- The selective chunk-7680 server captured successfully. Draft-extend graph
  memory fell from roughly 0.05 to 0.03 GiB and graph-end available memory rose
  to 4.63 GiB. Exact `6213+128` profile completed in 54 cycles at
  **2.370370** tokens/cycle.
- Candidate trace:
  `benchmark/windows/profiles/target_width_m3-20260820-115319/target_width_m3-1787251999.850064-TP-0.trace.json.gz`,
  SHA-256
  `3d431c6142df0037fcf2180729d65ca1a6f1626b070083832e2f92ca693230cc`.
  Full cycles averaged **16.066558 ms**, median 16.047987, versus matched
  unpruned **16.058328 ms**. Draft-extend graph 8 averaged **1.061 ms** versus
  **1.059 ms** control; kernel count was 29 versus 28.
- Decision: the apparent 147.534 tok/s projection was acceptance-only. Row
  pruning saves memory but not execution because the NVFP4 lm-head reads the
  same weights at M=1 and M=3. Removed all code and test changes, stopped the
  server, and retained only the trace/manifest plus documentation.

### 2026-08-20 12:29 PDT - bit-exact Gemma direct output clears 3000/110

- Found a lower-risk native-Windows norm opportunity before writing a new
  kernel. The Windows `gemma_fused_add_rmsnorm` wrapper performed
  `residual.add_(x)`, allocated a temporary through `gemma_rmsnorm`, then
  copied the result back into `x`. The existing JIT API already accepts an
  output tensor. Changed the wrapper to pass `x` directly, preserving the
  exact residual update and exact JIT Gemma arithmetic.
- Added native-Windows SM120 Qwen-shape coverage at rows 1/3 and hidden 5120.
  Input and residual are bit-exact against the former staged sequence. The
  four targeted Qwen Gemma tests passed. The updated native hot-path smoke
  retained fullgraph exactness and measured **38.731 -> 29.254 us** at M1 and
  **37.578 -> 29.184 us** at M3. A broader FP8 norm-fusion test still hits an
  unrelated FlashInfer 0.6.17 source compile error (`uint` undefined); it is
  outside this JIT path.
- The first selective chunk-7680 exact window was:
  prompt `3016.444, 3013.834, 3013.975`, generation
  `112.355, 97.506, 112.534`, TTFT `65.971714, 66.028859, 66.025761 s`, E2E
  `66.105219, 66.182696, 66.159054 s`. Two of three runs cleared every
  milestone gate.
- A fresh independent server produced prompt `3014.657, 3009.496, 3012.204,
  3013.736, 3011.489` and generation `96.531, 86.114, 98.100, 112.012,
  79.442`. Run four independently cleared every gate at
  **3013.736/112.012**, TTFT 66.031008 s, E2E 66.164923 s. All eight exact
  requests completed `199016`, `finish_reason=length`, and digest
  `9a0e2074...62668b37`; prompt mean was **3013.229**.
- Two exact `199000+512` support runs measured
  `3015.106/108.271` and `3011.779/111.094`, mean
  **3013.443/109.683**. Two real sampled `6213/512` windows averaged
  **144.535** and **138.621 tok/s**; combined mean **141.578**. Five
  acceptance probes averaged **2.249107**, showing the sampled gain did not
  come from a materially different acceptance regime.
- Selective behavior passed: coherent reasoning and final `703`, exactly one
  `multiply({"a":37,"b":19})` call, language-only model surface, and 4,994 MiB
  free after flush. The candidate full-cycle trace is
  `benchmark/windows/profiles/target_width_m3-20260820-115928/target_width_m3-1787252368.4682736-TP-0.trace.json.gz`,
  SHA-256
  `e9f2e09a78a85f2656bc02b9379cbf3b401cb26c455f41ba634379cbfe8009b2`.
- Relaunched launcher-default base RadixArk at production chunk 4096. All
  target/draft/extend graphs captured; exact `199000+16` completed at
  **2643.254 prompt / 101.980 generation tok/s**, with 698 MiB free after the
  request and 1,910 MiB after arithmetic/tool probes plus flush. Arithmetic,
  one parsed tool call, image/audio false, and five-run sampled mean
  **124.208 tok/s** passed.
- Standalone process-scoped OpenCode2 integration completed with exit code 0
  and visible `READY`; the wrapper restored its environment overlay and did
  not change global user configuration. The exact server tree was stopped
  leaf-first afterward; port 30000, compiler workers, and CUDA residency
  returned to clean state.
- Decision: promote the bit-exact direct-output change and the combined
  selective chunk-7680 record. The user's **3000 prompt / 110 generation
  tok/s** milestone is fully achieved and independently verified.

### 2026-08-20 15:36 PDT - new optimization branch establishes a fresh exact baseline

- Began a new explicitly authorized performance branch on `main` at
  `adf3a620ef64e11aea6159643f560c790327c57f`. Initial worktree state was
  `BENCHMARK.md` modified and `HANDOFF.md` deleted; both are pre-existing
  user-owned changes and remain untouched. Root `PERFORMANCE_LOG.md` and
  `FAILED_PATHS.md` were present and retained.
- Read the compact benchmark authorities and full latest notebook, then
  dispatched read-only documentation and source surveys covering repository
  rules, all root/notes records, all human-authored `docs/` pages and rendered
  cookbook prose, benchmark/test/script documentation, kernel/cache/runtime
  documentation, multimodal documentation, and the remaining peripheral
  trees. The surveys used no shell commands, processes, or edits.
- Pre-launch state: port 30000 free; no `cl`, `link`, `nvcc`, `ninja`, or
  `ptxas` worker; RTX 5090 driver `610.88`, P8, 29 C, 1,716 MiB used and
  30,472 MiB free. Installed versions were Python `3.13.14`, PyTorch
  `2.13.0+cu130`, CUDA runtime `13.0`, CUDA toolkit `13.3.33`, Triton `3.7.1`,
  and FlashInfer `0.6.17`. Both base RadixArk and selective AttnNVFP4 model
  paths existed.
- Launched exactly:

  ```powershell
  .\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1 `
    -ModelPath C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4 `
    -ChunkedPrefillSize 7680 `
    -RandomSeed 615388882
  ```

  The verified listener lineage was
  `44500 (pwsh) -> 37588 (sglang) -> 16276 (python) -> 41904 (listener)`.
  The command line resolved exact 200K context/pools, page 64, one request,
  FlashInfer prefill/sampling, TRT-LLM MHA target/draft decode, M3 linear
  rejection sampling, draft top-k 20, FP8 draft KV, FP32 ReplaySSM state,
  `extra_buffer_lazy`, torch compile `default`, and no tree, SWOR, adaptive,
  simulation, or device-resident-cycle control.
- Target, draft-decode, and draft-extend graphs captured in **33.49, 1.43,
  and 0.88 s** with 4.29 GiB reported after capture. `/health` returned 200;
  `/model_info` reported image/audio understanding false. Startup weight,
  KV allocation, and scheduler timings were 24.82, 0.82, and 69.99 seconds.
- Ran two complete exact-shape warmups, then five consecutive cache-flushed
  scores. The first invocation used `--warmup-runs 2`; the following four
  used `--skip-warmup`:

  ```powershell
  .\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py `
    --input-tokens 199000 --output-tokens 16 --timeout 600
  ```

  Prompt samples were `2897.795, 2875.047, 2837.904, 2873.846, 2872.198
  tok/s`; mean **2871.358**, median **2873.846**, standard deviation
  **21.439**, CV **0.747%**, and fixed-token aggregate **2871.229**.
  TTFT was `68.672916, 69.216270, 70.122180, 69.245186, 69.284914 s`.
- Legacy generation samples were `90.816, 85.650, 91.199, 111.926, 72.704
  tok/s`; arithmetic mean **90.459**, aggregate
  `75 / sum(E2E-TTFT)` rate **88.746**, standard deviation **14.141**, and
  CV **15.633%**. E2E was `68.838085, 69.391402, 70.286654, 69.379203,
  69.491229 s`. Nonempty SSE fragments varied `4,4,4,4,3`, so this
  short-generation result remains cycle- and transport-quantized.
- Every score completed exact `199000+16`, returned `finish_reason=length`,
  kept thinking enabled, and retained digest
  `9a0e20749e2930a697fefdd3bdd7863a067abe4d9860e6d1e7d9b80a62668b37`.
  Scored snapshots reached P1, 2.947-2.977 GHz SM, 13.801 GHz memory,
  59-69 C, and 515-559 W. Chrome, Edge WebView, iCloud, shell/display
  clients, and an unrelated Python process remained present. NVIDIA reported
  accumulated software power-capping time.
- Decision: the current environment did not reproduce the historical
  **3016.444/112.355** record. Retain **2871.358/90.459** as the immediate
  matched baseline without replacing the qualified record. The prompt gap is
  4.810%; every candidate now needs an adjacent same-environment control.
  Longer generation and device-cycle evidence will determine whether an
  apparent generation change is real.
- Raw artifact:
  `C:\Users\Daniel\.copilot\session-state\df1c744a-8e2f-4823-bd37-18b450ed10d1\files\baseline-200k-20260820-1527.log`.

### 2026-08-20 15:55 PDT - supporting baseline, adjacent control A, and exact calibration guard

- On the same verified selective chunk-7680 server, ran three cache-flushed
  exact long-generation requests:

  ```powershell
  .\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py `
    --input-tokens 199000 --output-tokens 512 --timeout 600
  ```

  The first invocation included one same-shape warmup; the next two used
  `--skip-warmup`. Prompt was `2983.007, 2942.383, 2945.135 tok/s`, mean
  **2956.842**, median **2945.135**, and CV 0.768%. Legacy generation was
  `107.385, 107.491, 111.337 tok/s`, mean **108.738**, median **107.491**,
  CV 2.071%, and aggregate **108.707**. All completed exact `199512` with
  stable digest
  `1e90cc8fad3e1b1802db4cdc2af762790bcd392c062a14f0afc334df8b5e97f9`.
- Five real sampled `6213/512` requests then measured
  `122.714, 122.917, 111.596, 119.056, 113.418 tok/s`, mean **117.940**,
  median **119.056**, and CV 4.436%. Five native acceptance probes measured
  accepted length `2.133333, 2.169492, 2.124481, 2.142259, 2.206897`, mean
  **2.155292**; mean acceptance rate was **0.577233** and mean verify count
  was **237.6**.
- Took the adjacent exact-shape control A after those probes, with five
  cache-flushed `--skip-warmup` requests. Prompt was
  `2909.109, 2827.344, 2908.788, 2832.229, 2858.962 tok/s`, mean
  **2867.286**, median **2858.962**, CV 1.391%, and aggregate **2866.843**.
  Legacy generation was `92.718, 92.717, 112.714, 103.474, 107.168 tok/s`,
  mean **101.758**, median **103.474**, CV 8.731%, and aggregate
  **101.136**. Every request completed exact `199016`, returned
  `finish_reason=length`, and retained digest
  `9a0e20749e2930a697fefdd3bdd7863a067abe4d9860e6d1e7d9b80a62668b37`.
- The long and short windows had the same exact 199000-token prompt,
  explicit cache flushes, server, and source but differed materially in TTFT.
  This confirms the environment is non-stationary enough that every candidate
  requires adjacent A-B-A evidence.
- Hardened `scripts/windows/bench_openai_stream.py`: if local calibration
  returns anything other than `args.input_tokens`, it raises before cache
  flush or generation. Warmup and measurement usage validation now compares
  against the requested count. Added a CPU regression test showing a
  `198999` result for a `199000` request invokes neither `flush_cache` nor
  `stream_request`. All four focused tests and Python compilation passed.
- Raw artifacts:
  `C:\Users\Daniel\.copilot\session-state\df1c744a-8e2f-4823-bd37-18b450ed10d1\files\baseline-support-20260820-1542.log`
  and
  `C:\Users\Daniel\.copilot\session-state\df1c744a-8e2f-4823-bd37-18b450ed10d1\files\control-a-exact16-20260820-1554.log`.

### 2026-08-20 16:43 PDT - PERF-024 large-EXTEND tactics set a provisional exact-200K record

- On committed `HEAD=69d88c6912863aa89141af13e94c8081ef4439c8`, changed
  the FlashInfer startup autotune experimentally so an explicitly requested
  ordinary EXTEND dummy remains EXTEND on a speculative target worker,
  speculative verify metadata is created only for TARGET_VERIFY, and the
  existing `SGLANG_FLASHINFER_AUTOTUNE_EXTEND` gate excludes the draft worker
  rather than every speculative runner. Multimodal and non-generation skips
  remained intact.
- Snapshotted the original FlashInfer tactic cache under
  `C:\Users\Daniel\.copilot\session-state\df1c744a-8e2f-4823-bd37-18b450ed10d1\files\flashinfer-autotune-pre-perf024`.
  It was 3,029 bytes with SHA-256
  `54C9520E83722FA0670C5334520ADC228E22B9F8E2A5BC2304CFA185BE938B0A`.
  Launched the selective profile with the opt-in set only in the launch
  process:

  ```powershell
  $env:SGLANG_FLASHINFER_AUTOTUNE_EXTEND = "1"
  .\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1 `
    -ModelPath C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4 `
    -ChunkedPrefillSize 7680 `
    -RandomSeed 615388882
  ```

  The extra ordinary-EXTEND pass ran at 16,384 tokens and populated 22 FP4
  M-buckets through 16,384. The cache grew to 20,926 bytes, SHA-256
  `20E745B0B328437F4C5DC0360575541D0A2B1AABD29676393CFBD70B33603150`.
  Target, draft-decode, and draft-extend graph phases all captured; startup
  reported 4.65 GiB free.
- The first five exact `199000+16` scores had prompt
  `3051.759, 3033.324, 3039.994, 3034.686, 2862.891 tok/s`, mean
  **3004.531**, median **3034.686**, and best **3051.759**. Legacy generation
  was `131.591, 130.960, 130.178, 129.702, 129.927 tok/s`, mean **130.472**.
  The first request set a same-request provisional record at
  **3051.759 prompt / 131.591 generation tok/s**, TTFT **65.208286 s** and
  E2E **65.322275 s**. All requests completed exact `199016`, returned
  `finish_reason=length`, and had digest
  `a6bcf2394ef3cdd140bf100ab6b4ee5fa90f3e77c3fefd993b9acc63fe6d19ec`.
- After the support probes and a 60-second idle, the second five-score window
  was stable: prompt `3047.409, 3034.755, 3042.432, 3041.355, 3041.038`,
  mean **3041.398**, median **3041.355**, and CV **0.148%**. Legacy
  generation was `111.725, 135.797, 132.677, 111.717, 96.922`, mean
  **117.768**; the fixed 16-token interval remained cycle-quantized. The
  stable prompt mean was **6.073%** above adjacent control A,
  **3.934%** above restored control A2, and **0.827%** above the historical
  3016.444 record.
- Three exact `199000+512` requests measured prompt
  `2881.222, 3019.790, 3048.056 tok/s`, mean **2983.023**, and generation
  `110.670, 118.180, 116.470 tok/s`, mean **115.107**, aggregate
  **115.016**. Each completed exact `199512`, returned
  `finish_reason=length`, and retained digest
  `c29e8dff98df577b8abf24c8373ca3e5ab7451ce14aeec43018c74eaf5abe2ea`.
  Five real sampled `6213/512` requests measured
  `125.251, 121.666, 125.182, 120.795, 129.973 tok/s`, mean **124.573**,
  a **5.624%** improvement over the fresh baseline support window.
- Five native acceptance probes measured accepted length
  `2.226087, 2.188034, 2.370370, 2.255507, 2.188034`, mean **2.245606**;
  mean acceptance rate was **0.621928** and mean verify count **228.2**.
  Behavior retained coherent `reasoning_content`, arithmetic result `703`,
  exactly one parsed `multiply({"a":37,"b":19})` call with
  `finish_reason=tool_calls`, and image/audio understanding false. Cache flush
  left 5.301 GiB free.
- Stopped the candidate tree leaf-first
  (`35596 -> 54872 -> 20684 -> 45040`). Port 30000 and compiler workers were
  clear; GPU residency returned to 650 MiB. Reverted the experimental source
  and restored the original tactic cache byte-for-byte before control A2.
- Relaunched the identical selective model/chunk/seed with baseline source and
  the restored 3,029-byte cache. No large EXTEND autotune ran and the cache
  remained unchanged. Five exact A2 scores measured prompt
  `2997.970, 2996.980, 2883.162, 2893.345, 2860.058 tok/s`, mean
  **2926.303**, and generation
  `98.152, 86.027, 94.325, 105.532, 79.873 tok/s`, mean **92.782**.
  Every request returned the baseline
  `9a0e20749e2930a697fefdd3bdd7863a067abe4d9860e6d1e7d9b80a62668b37`
  digest. This A-B-A result attributes the speed and deterministic-output
  change to selected FP4 tactics rather than the ordinary launch arguments.
- Stopped A2 leaf-first
  (`55984 -> 44644 -> 46188 -> 37276`) at 16:42 PDT. Port 30000 and compiler
  workers were clear; GPU residency was 824 MiB.
- Reapplied the candidate as a narrower retained implementation:
  `_dummy_run` now requires an explicit `allow_speculative_target_extend`
  capability in addition to target-worker and EXTEND assertions; only the
  gated FlashInfer helper supplies it. Draft workers still skip the pass and
  ordinary speculative dummy callers still become TARGET_VERIFY. Added CPU
  coverage for opt-in target admission, draft rejection, and default-off
  behavior. Seven focused tests, four subtests, Python compilation, and
  whitespace checks passed. The system virtualenv lacks Ruff/Black, so the
  repository-pinned Ruff 0.15.1 and Black 26.1.0 were run through `uvx`;
  lint and formatting passed.
- Raw artifacts:
  `perf024-exact16-20260820-1604.log`,
  `perf024-support-20260820-1611.log`,
  `perf024-exact16-window2-20260820-1618.log`,
  `perf024-behavior-20260820-1627.log`,
  `perf024-shutdown-20260820-1629.log`,
  `control-a2-exact16-20260820-1632.log`, and
  `control-a2-shutdown-20260820-1643.log`, all under the session `files`
  directory above.
- Status: PERF-024 is a provisional winner, not yet promoted. Required next
  evidence is a clean independent restart with the retained source, a cache-
  only restart to separate tactic persistence from startup-state effects, and
  final production-default/OpenCode2 gates.

### 2026-08-20 17:12 PDT - independent retune reproduced PERF-024 and isolated tactic causality

- Kept `HEAD=69d88c6912863aa89141af13e94c8081ef4439c8` and the
  retained speculative-target EXTEND implementation. Restored the original
  3,029-byte cache before launch, then independently ran the same opt-in
  selective profile:

  ```powershell
  $env:SGLANG_FLASHINFER_AUTOTUNE_EXTEND = "1"
  .\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1 `
    -ModelPath C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4 `
    -ChunkedPrefillSize 7680 `
    -RandomSeed 615388882
  ```

  The new cache was 20,928 bytes with SHA-256
  `8219484FA86EBB0E6DDA54F2D15447DBC502EBCEA9007B3E1BB917B9001F9ADF`.
  Its preserved snapshot is
  `C:\Users\Daniel\.copilot\session-state\df1c744a-8e2f-4823-bd37-18b450ed10d1\files\flashinfer-autotune-perf024-restart\rank_tp0_pp0_dp0.json`.
- Five exact `199000+16` scores measured prompt
  `3051.345, 3048.538, 3048.086, 3042.488, 3044.105 tok/s`, mean
  **3046.912**. The third request independently beat both published
  same-request records at **3048.086 prompt / 112.499 generation tok/s**,
  TTFT **65.286869 s**, and E2E **65.420204 s**. Every request completed
  exact `199016`, returned `finish_reason=length`, and used digest
  `cdf5bb57b88deaa7515abaedf36406d10494599fce2e23eeaa400461d9f647d9`.
- Three exact `199000+512` requests measured prompt
  `3045.851, 3049.214, 3044.966 tok/s`, mean **3046.677**, and generation
  `117.632, 117.686, 118.241 tok/s`, mean **117.853**. All completed exact
  `199512` with digest
  `cac0c6e4fab3115102a9a0c4163e4465068fba30cb09f0bb5556c7021e4a2092`.
  Five real sampled `6213/512` requests averaged **125.517 tok/s**.
  Five native probes averaged about **2.2173** accepted tokens per verify,
  **0.6069** acceptance rate, and **231** verifies.
- Arithmetic/reasoning returned `703`; the tool check emitted exactly one
  parsed `multiply({"a":37,"b":19})` call with
  `finish_reason=tool_calls`; `/model_info` kept image/audio false. A cache
  flush left about 5.4 GiB free.
- Ran two causality controls after clean leaf-first restarts:
  - **Cache-only:** launched with the 20,928-byte cache but without the
    ordinary EXTEND pass. Exact prompt returned to about **3009.716 tok/s**,
    long generation to about **111.016 tok/s**, and both outputs returned to
    the baseline digest family. The file entries alone were not retained for
    runtime use.
  - **Dummy-only:** ran the same 16,384-token ordinary EXTEND forward while
    skipping FP4 autotuning. Three exact prompts averaged about
    **3008.942 tok/s** and retained the baseline digest. The dummy forward's
    Mamba/KV state was therefore not the source of the gain.
- These controls attribute both the throughput and deterministic-output
  change to the selected FP4 tactics. Raw artifacts include
  `perf024-restart-server-20260820-1645.log`,
  `perf024-restart-exact16-20260820-1650.log`,
  `perf024-restart-support-20260820-1659.log`,
  `perf024-restart-behavior-20260820-1704.log`,
  `perf024-cacheonly-causality-20260820-1707.log`, and
  `perf024-dummyonly-causality-20260820-1718.log` under the session
  `files` directory.

### 2026-08-20 17:57 PDT - rejected fresh profiling and fixed FlashInfer file-cache lifetime

- A direct attempt to reactivate the saved cache after later startup phases
  did not recover the selected tactics. Source tracing found the precise
  lifetime mismatch in FlashInfer 0.6.17:
  - file hits loaded by `autotune(cache=...)` live in process-global
    `_file_configs`;
  - each later speculative-draft autotune context clears and replaces that
    table;
  - live profiling writes runner-keyed entries to `profiling_cache`, which
    survives later contexts.
- Tested always profiling the large target EXTEND pass afresh. Five exact
  prompts remained strong at mean **3043.747 tok/s**, and two short requests
  beat both historical same-request values. However, the independently
  selected tactics reduced three long-generation results to
  `101.469, 98.969, 103.049 tok/s`, mean **101.162**, despite sampled
  generation averaging about **125.006 tok/s**. Behavior and standalone
  OpenCode2 still passed. Rejected fresh profiling as a production policy
  because startup-selected tactics introduced material long-generation
  variance.
- Added an opt-in file-hit promotion adapter around only the target EXTEND
  forward. It wraps `AutoTuner.search_cache`, copies only matching file
  entries actually exercised by that forward into the exact runner-keyed
  `profiling_cache`, and restores the original method in `finally`. Ordinary
  target-decode and speculative-draft autotune paths remain unchanged.
- Restored the independently selected 20,928-byte cache and launched the
  selective profile again. Startup promoted exactly **110** target FP4
  configs without re-profiling. Target, draft-decode, and draft-extend graph
  phases all captured. The first exact smoke measured
  **3050.570 prompt / 97.844 short generation tok/s** with digest
  `cdf5bb57b88deaa7515abaedf36406d10494599fce2e23eeaa400461d9f647d9`;
  the first long smoke measured **3048.094 prompt / 118.184 generation
  tok/s** with digest
  `cac0c6e4fab3115102a9a0c4163e4465068fba30cb09f0bb5556c7021e4a2092`.
  Those digests exactly matched the independent retune, proving the adapter
  recreated the selected in-process tactic state.
- Raw artifacts include `perf024-final-gates-20260820-1728.log`,
  `perf024-freshfinal-exact16-20260820-1737.log`,
  `perf024-freshfinal-fullgates-20260820-1747.log`,
  `perf024-promoted-server-20260820-1758.log`, and
  `perf024-promoted-smoke-20260820-1800.log`.

### 2026-08-20 18:17 PDT - deterministic promotion qualified; hostile review closed OOM fallback

- Continued the promoted-cache smoke into a five-request exact window. Prompt
  was `3050.570, 3048.607, 3044.288, 3045.422, 3047.659 tok/s`, mean
  **3047.309**, all exact `199000+16` with identical `cdf5bb...f647d9`
  digests. Short-generation samples were
  `97.844, 97.326, 77.929, 86.769, 111.849 tok/s`, mean **94.343**;
  this 15-token interval remains quantized by speculative-cycle and SSE
  boundaries and is not the primary generation comparison.
- Three exact `199000+512` requests measured prompt
  `3048.094, 3047.287, 3047.882 tok/s`, mean **3047.754**, and generation
  `118.184, 118.764, 118.219 tok/s`, mean **118.389**. Every request
  completed exact `199512` with identical `cac0c6...a2092` digests.
- Five sampled `6213/512` requests measured end-to-end generation
  `123.488, 122.142, 128.497, 126.584, 130.551 tok/s`, mean **126.252**;
  decode-only mean was **143.329 tok/s**. Five native probes measured
  accepted length
  `2.285714, 2.245614, 2.169492, 2.197425, 2.188034`, mean **2.217256**;
  acceptance-rate mean was **0.606901** and verify-count mean **231.0**.
- On this exact launch, arithmetic/reasoning returned `703`, exactly one
  `multiply({"a":37,"b":19})` tool call parsed with
  `finish_reason=tool_calls`, and `/model_info` kept image/audio false. An
  intentionally reduced OpenCode2 output cap of 128 exhausted its budget
  after hidden reasoning and emitted no visible text; this harness probe was
  rejected. The established standalone wrapper at its normal 8192-token cap
  returned visible `READY` and restored the process-scoped configuration
  overlay. Cache flush left **5,386 MiB** free.
- Verified and stopped the launch leaf-first:
  `36608 (listener) -> 23124 (python) -> 45672 (sglang) -> 4284 (pwsh)`.
  Port 30000 and compiler workers were clear; GPU residency returned to
  573 MiB. Artifacts are
  `perf024-promoted-window-20260820-1805.log`,
  `perf024-promoted-generation-20260820-1814.log`,
  `perf024-promoted-final-gates-20260820-1816.log`, and
  `perf024-promoted-shutdown-20260820-1816.log`.
- Hostile review found one startup failure-path gap: the optional EXTEND pass
  allocated its large dummy buffers before entering the OOM fallback, so an
  allocation OOM could still abort startup. Moved allocation inside the
  guarded region, retained fail-fast behavior for non-OOM errors, narrowed
  file-hit promotion to speculative targets, and added CPU regression
  coverage for allocation OOM plus exception-safe method restoration. All
  five focused tests passed.
- Selected PERF-024 result:
  - qualified same-request record: **3048.086 prompt / 112.499 generation
    tok/s** from the independent retune;
  - deterministic five-run exact prompt mean: **3047.309 tok/s**;
  - deterministic three-run long generation mean: **118.389 tok/s**;
  - best observed same-request result: **3051.759 / 131.591 tok/s**;
  - selected cache: 20,928 bytes,
    `8219484FA86EBB0E6DDA54F2D15447DBC502EBCEA9007B3E1BB917B9001F9ADF`.
  The next gate is an unchanged launcher-default base-model/chunk-4096
  production relaunch with the EXTEND opt-in absent.

### 2026-08-20 18:26 PDT - unchanged production defaults passed after PERF-024

- Launched
  `scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1` with no arguments and
  `SGLANG_FLASHINFER_AUTOTUNE_EXTEND` absent. The resolved server used base
  `Qwen3.8-27B-NVFP4-RadixArk`, chunk 4096, generated seed `10981265`, exact
  200,000 context and target/draft token pools, one request, page 64, M3
  linear rejection sampling, draft top-k 20, FP8 draft KV, FP32 Mamba state,
  and all experimental tree/SWOR/simulation controls inactive.
- Startup loaded both exact 200,000-token KV pools. No extra EXTEND pass and
  no file-hit promotion appeared in the log. Target verify, draft decode, and
  draft extend graphs captured in **41.18, 1.29, and 0.98 s**; startup
  reported 1.56 GiB free and `/health` returned 200.
- One cache-flushed exact `199000+16` request completed exact `199016` with
  `finish_reason=length`, **2648.283 prompt / 88.187 short generation tok/s**,
  TTFT **75.143026 s**, and E2E **75.313120 s**. The short generation field
  remains cycle-quantized; this gate was for unchanged behavior and capacity,
  not a new base-profile performance window.
- Preserved reasoning returned `703`; exactly one
  `multiply({"a":37,"b":19})` call parsed with
  `finish_reason=tool_calls`; `/model_info` reported the base checkpoint and
  image/audio false. Standalone OpenCode2 returned visible `READY` and
  restored its process-scoped overlay. Post-flush free VRAM was **2,222 MiB**.
- Verified and stopped the tree leaf-first:
  `35148 (listener) -> 31028 (python) -> 22888 (sglang) -> 34068 (pwsh)`.
  Port 30000 and compiler workers were clear; GPU residency returned to
  585 MiB.
- The selected live selective-profile cache remains 20,928 bytes with SHA-256
  `8219484FA86EBB0E6DDA54F2D15447DBC502EBCEA9007B3E1BB917B9001F9ADF`.
  The original 3,029-byte cache remains separately preserved with SHA-256
  `54C9520E83722FA0670C5334520ADC228E22B9F8E2A5BC2304CFA185BE938B0A`.
  Artifacts are `perf024-production-default-server-20260820-1820.log`,
  `perf024-production-default-ready-20260820-1822.log`,
  `perf024-production-default-gates-20260820-1824.log`, and
  `perf024-production-default-shutdown-20260820-1825.log`.
- Committed the implementation, focused CPU tests, performance ledger, and
  raw recovery record as
  `7f5af878da7b8dc43063f31e554dfc69cee5d510`
  (`perf: retain large-extend FlashInfer tactics`). The pre-existing
  `BENCHMARK.md` edits and `HANDOFF.md` deletion were not staged.

### 2026-08-20 18:45 PDT - compact Windows scoreboard and next target selected

- Simplified the root Windows scoreboard to the four user-selected outputs:
  prompt throughput, generation throughput, TTFT, and end-to-end time. Removed
  surrounding qualification-window, cache, digest, and secondary-workload
  result numbers from the compact section; detailed evidence remains in this
  ledger and `notes/benchmark-contract.md`.
- Set the next exact `199000+16` target to **3100 prompt / 120 generation
  tok/s**, **<=64.20 s TTFT**, and **<=64.35 s** end to end in the same
  eligible request. The time limits are derived from
  `199000 / 3100 = 64.1935` plus 15 post-first-token intervals at 120 tok/s.
  Exact `199016` completion and `finish_reason=length` remain pass/fail gates,
  not a fifth performance target.

### 2026-08-20 19:21 PDT - new optimization lane baseline established

- Began the explicit autonomous target of clearing all four root-scoreboard
  thresholds in one exact request: prompt **>=3100 tok/s**, generation
  **>=120 tok/s**, TTFT **<=64.20 s**, and E2E **<=64.35 s**, with exact
  `199016` usage and `finish_reason=length`. The validation loop is a matched
  current-source baseline, one reachable variable at a time, five consecutive
  scores plus supporting correctness/capacity evidence, and an independent
  promotion window.
- Worktree before launch was clean on `main` at
  `cb11475a4e0c68cfe542f66a919b468f205392f0`, one commit ahead of
  `origin/main`. Port 30000 was free, no SGLang/CUDA/compiler worker was
  active, and the RTX 5090 was at ordinary display residency: driver `610.88`,
  786 MiB used, 31,402 MiB free, 30 C.
- Exact launch:
  `$env:SGLANG_FLASHINFER_AUTOTUNE_EXTEND='1';`
  `.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1`
  `-ModelPath C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4`
  `-ChunkedPrefillSize 7680 -RandomSeed 615388882`.
  Resolved arguments retained real 200,000 context and target/draft token
  pools, one request, page 64, FlashInfer prefill/sampling, TRT-LLM MHA target
  and draft decode, M3 linear rejection sampling, aligned draft top-k 20,
  FP8 E4M3 draft KV, FP32 ReplaySSM state, torch compile `default`, scheduler
  and stream intervals 4, and every tree/SWOR/adaptive/simulation/device-cycle
  control inactive.
- Startup promoted exactly **110** selected target FP4 configs into the
  process cache. Target verify, draft decode, and draft extend graph captures
  completed in **20.09, 1.25, and 0.96 s**. `/health` returned 200 and
  `/model_info` reported image/audio understanding false. Listener PID
  `44656` descended through Python `49796`, `sglang.exe` `53928`, and launcher
  PowerShell `38844`; re-resolve the live tree before stopping it.
- Runtime versions were Python `3.13.14`, PyTorch `2.13.0+cu130`, CUDA runtime
  `13.0`, Triton `3.7.1`, and FlashInfer `0.6.17`. Before the benchmark the
  server occupied 27,224 MiB with 4,964 MiB free. WDDM clients included
  Chrome, Edge WebView, iCloud, Windows shell/display processes, and the server
  Python process.
- The first collection wrapper failed before any request because it attempted
  to assign PowerShell's read-only `$PID` variable. The corrected wrapper
  immediately retried without changing server state.
- Benchmark command for the first score was
  `.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py`
  `--input-tokens 199000 --output-tokens 16 --warmup-runs 2`
  `--warmup-output-tokens 16 --timeout 600`; the next four added
  `--skip-warmup`. This produced two full exact-shape warmups followed by five
  cache-flushed scored requests.
- Prompt samples:
  `2905.351, 2927.990, 2957.401, 2936.653, 2959.654 tok/s`; mean
  **2937.410**, median **2936.653**, CV **0.683%**. Generation samples:
  `94.098, 90.935, 105.232, 87.540, 89.888 tok/s`; mean **93.539**, median
  **90.935**, CV **6.644%**.
- TTFT samples:
  `68.494311, 67.964704, 67.288805, 67.764230, 67.237593 s`; mean
  **67.749929 s**. E2E samples:
  `68.653720, 68.129658, 67.431347, 67.935580, 67.404467 s`; mean
  **67.910954 s**.
- Every score completed exact `199000+16`, returned
  `finish_reason=length`, kept thinking enabled, and retained digest
  `cdf5bb57b88deaa7515abaedf36406d10494599fce2e23eeaa400461d9f647d9`.
  This is the reproducible current-environment baseline, not a replacement for
  the qualified **3048.086/112.499** record.
- Raw artifact:
  `C:\Users\Daniel\.copilot\session-state\fd2e8d01-e225-4b48-9ab3-4d118100a4a9\files\baseline-exact200k-20260820-1910.log`.

### 2026-08-20 19:24 PDT - dense TRT-LLM FP4 rejected by the SM120 capability gate

- Stopped the baseline server leaf-first after re-resolving its exact tree:
  CUDA/detokenizer leaves `24316/31188`, listener `44656`, Python parent
  `49796`, launcher shim `53928`, console `56548`, and PowerShell `38844`.
  All known PIDs exited, port 30000 was free, compiler workers were absent,
  and the GPU returned to 1,126 MiB display residency with 31,062 MiB free.
- Inspected the core route before changing the launcher. SGLang's
  `Fp4GemmRunnerBackend`, `ModelOptFp4LinearMethod` weight shuffling, and
  `fp4_gemm` dispatch all support the user-facing `flashinfer_trtllm` value.
  The Windows launcher omits it from the FP4 `ValidateSet`, and existing dense
  backend CI runs on B200 rather than SM120.
- Ran:
  `.\scripts\windows\invoke_cuda_pytest.ps1`
  `test\registered\unit\layers\quantization\test_nvfp4_linear_backends.py::`
  `TestNvFp4LinearBackends::test_flashinfer_trtllm -q`.
- All three real-layer subtest shapes reached FlashInfer and failed before
  kernel execution with
  `BackendSupportedError: mm_fp4 does not support backend 'trtllm' with
  capability 120`. The shapes were `(64,256,512)`, `(5,160,336)`, and
  `(128,1024,1024)`.
- No launcher or runtime source was changed. Close this candidate for
  FlashInfer `0.6.17` on the RTX 5090; reopen only when the dependency
  explicitly supports dense TRT-LLM FP4 on capability 120 and the focused
  numerics plus graph-replay gate passes.

### 2026-08-20 19:30 PDT - fused Gemma residual norm passed isolated exactness and latency gates

- Began from clean commit `538be003dd` with no server, CUDA/compiler worker,
  or port-30000 listener. The RTX 5090 was at 1,135 MiB display residency,
  31,053 MiB free, 11% transient utilization, and 30 C.
- Added `GemmaFusedAddRMSNormHalfKernel` beside the existing exact JIT
  `RMSNormHalfKernel`. The new kernel loads BF16 input and residual, performs
  the FP32 add and BF16 rounding that `residual.add_(input)` exposed at the
  former kernel boundary, stores that exact residual, then uses the same
  thread/vector ownership, warp/CTA reduction, `rsqrt`, and
  `input * norm * (weight + 1)` order as the current direct-output Gemma norm.
  Both the pre-Blackwell 16-byte/two-vector and Blackwell
  32-byte/one-vector geometries are present.
- Added a mutating custom-op wrapper and narrowly selected it in the native
  Windows Gemma path only for supported two-byte half-width shapes. The old
  `residual.add_` plus direct-output JIT norm remains the explicit fallback.
  No non-Windows dispatch changed.
- Ran
  `.\scripts\windows\invoke_cuda_pytest.ps1`
  `test\registered\kernels\ops\layernorm\test_rmsnorm.py`
  `-k qwen35_gemma -v -s`: **6 passed**, including exact BF16 equality for
  both output buffers at `M={1,3,7000,7680}`, `H=5120`.
  Ran the new graph-replay node directly: **2 passed** at M1/M3 after changing
  both captured input buffers twice. Python compilation and
  `git diff --check` also passed.
- Ran
  `.\scripts\windows\smoke_native_qwen35_hotpaths.ps1 -Iterations 5000`.
  The fused candidate versus the staged sequence measured
  **24.406 vs 40.812 us** at M1 and **25.280 vs 41.419 us** at M3, with
  exact input/residual outputs; the fullgraph integration remained exact.
- A kernel-only A-B-A without reset cost measured
  `9.554 / 16.705 / 9.505 us` at M1,
  `9.760 / 16.478 / 9.450 us` at M3,
  `195.403 / 193.696 / 195.809 us` at M7000, and
  `217.110 / 245.719 / 213.329 us` at M7680
  (fused A / staged / fused B). M1/M3 have large repeatable launch and memory
  savings; M7680 improved; M7000 was neutral within about 1%.
- The source change is not yet retained or committed. Next gate is the
  selected-checkpoint/chunk-7680 full-model exact-200K adjacent comparison
  under the same seed and selected EXTEND cache.

### 2026-08-20 20:08 PDT - PERF-028 retained after adjacent exact-200K attribution

- Launched the fused worktree with:
  `$env:SGLANG_FLASHINFER_AUTOTUNE_EXTEND='1';`
  `.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1`
  `-ModelPath`
  `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4`
  `-ChunkedPrefillSize 7680 -RandomSeed 615388882`.
  Both target and draft pools resolved to 200,000 tokens and all three
  speculative graph phases captured. The target-verify capture took 15.39 s;
  draft decode and draft extend took 1.04 and 0.81 s.
- After two exact warmups, five fused `199000+16` scores produced prompt
  `2976.028, 2988.295, 2947.764, 2916.429, 2972.626` tok/s (mean
  **2960.228**), generation
  `88.022, 98.132, 112.992, 93.347, 101.591` tok/s (mean **98.817**),
  TTFT mean **67.229581 s**, and E2E mean **67.382454 s**. Every result
  completed exact `199016`, `finish_reason=length`, with digest
  `cdf5bb57b88deaa7515abaedf36406d10494599fce2e23eeaa400461d9f647d9`.
- Three fused exact `199000+512` scores measured
  `116.100, 116.486, 117.162` generation tok/s (mean **116.583**) and
  `2966.772, 2966.291, 2990.736` prompt tok/s (mean **2974.600**).
  All completed exact `199512` with digest
  `cac0c6e4fab3115102a9a0c4163e4465068fba30cb09f0bb5556c7021e4a2092`.
- Stopped that server, temporarily restored only the former Windows staged
  dispatch, and relaunched the identical checkpoint, chunk, seed, and selected
  cache. Three staged `199000+16` scores averaged
  **2967.386 prompt / 102.302 generation tok/s**,
  **67.064823 s TTFT**, and **67.212277 s E2E**. Three staged
  `199000+512` scores measured
  `116.226, 113.749, 115.608` generation tok/s (mean **115.194**) and
  `2975.865, 2974.860, 2999.546` prompt tok/s (mean **2983.424**).
  Counts, finish reasons, and digests matched the fused arm exactly.
- Attribution: retain the fused kernel as a narrow additive decode win.
  Adjacent exact long generation improved **1.388 tok/s / 1.205%**, in the
  direction predicted by the isolated M1/M3 kernel A-B-A. Prefill remained
  neutral within current WDDM variation: fused versus staged was -0.241% for
  the short prompt arm and -0.296% for the long prompt arm. The 16-token
  generation values remain too quantized and variable to attribute.
- Artifacts:
  `perf028-fused-server-20260820-1931.log`,
  `perf028-fused-exact200k-20260820-1936.log`,
  `perf028-fused-generation-20260820-1946.log`,
  `perf028-fused-shutdown-20260820-1951.log`,
  `perf028-control-server-20260820-1952.log`, and
  `perf028-control-window-20260820-1955.log` under the session `files`
  directory.
- Re-resolved the control tree from port 30000 and stopped only verified PIDs
  `49272, 38408, 44688, 39228, 36348`, leaf-first. Port 30000 was free,
  no compiler worker remained, and the RTX 5090 returned to 1,047 MiB display
  residency with 31,141 MiB free. Restored the fused Windows dispatch before
  validation and retention.
- With the fused dispatch restored, ran
  `.\scripts\windows\invoke_cuda_pytest.ps1`
  `test\registered\kernels\ops\layernorm\test_rmsnorm.py`
  `-k "qwen35_gemma or qwen35_jit_gemma" -q`: **8 passed**. Python
  compilation for both runtime modules and the test, plus
  `git diff --check`, also passed.

### 2026-08-20 20:26 PDT - PERF-027 exact native producer passed isolated byte and latency gates

- Recovered the documented chunk-size boundary before proceeding: 7680 is
  selective-checkpoint-only, exact 199K is `25 * 7680 + 7000`, base RadixArk
  remains at 4096, and 7808 is a closed planner/kernel cliff. Every isolated
  and later full-model gate therefore covers both M7680 and M7000 and must not
  alter the launcher default.
- Read the selected checkpoint metadata directly: hidden size 5120,
  intermediate size 17408, 64 layers. All 64 down projections carry ModelOpt
  input scales; layer 0 is `0.0025692894123494625`, used as the real-scale
  isolated reference.
- The installed public
  `flashinfer.silu_and_mul_nvfp4_quantize` failed before compilation with
  `ModuleNotFoundError: No module named 'cutlass'`. This is the same
  Linux-only CUTLASS-DSL dependency boundary already recorded for native
  Windows; no package was installed or modified. Artifact:
  `perf027-flashinfer-fused-probe-20260820-2010.log`.
- FlashInfer's native CUDA expert producer was then exercised with one expert.
  It improved median producer latency by 1.24-1.79x, but changed about 0.8% of
  packed FP4 bytes. Source inspection confirmed the cause: `__expf` plus one
  BF16 conversion after the product, versus the selected precise `expf`,
  activation-round, multiply, product-round contract. It was rejected as the
  exact producer. Artifact:
  `perf027-flashinfer-native-fused-probe-20260820-2014.log`.
- Added `silu_and_mul_nvfp4.cuh` plus a thin mutating custom-op wrapper. The
  kernel preserves the two BF16 rounding boundaries and calls the same
  FlashInfer native E4M3-scale/E2M1 conversion helper as the current
  quantizer. Its grid also writes all padded 128x4 scale bytes, avoiding
  uninitialized padding.
- The first compile exposed one missing FlashInfer internal include root; after
  adding the dependency's `nv_internal/include` path, compilation succeeded.
  No server was running, no compiler tree overlapped the probes, and post-run
  GPU residency returned to 1,193 MiB with 30,995 MiB free.
- Exact producer versus the current staged activation plus
  `fp4_quantize`, using BF16 width 17408 and the real layer-0 scale:
  - M1: zero packed/scale mismatches; `50.528 -> 20.224 us` median.
  - M3: zero packed/scale mismatches; `49.568 -> 20.256 us` median.
  - M7000: zero mismatches across 60,928,000 packed bytes and 7,659,520
    scale bytes; `664.112 -> 366.176 us`.
  - M7680: zero mismatches across 66,846,720 packed bytes and 8,355,840
    scale bytes; `730.416 -> 402.704 us`.
- Artifact:
  `perf027-exact-jit-probe-20260820-2026.log`.
  Next gates are mutable CUDA-graph replay, torch fullgraph compilation, and
  exact consumption through `ModelOptFp4LinearMethod` before any server launch.
- Added focused registered coverage. Exact BF16 production shapes
  M1/M3/M7000/M7680, mutable M1/M3 CUDA-graph replay with both input and scale
  changed, `torch.compile(fullgraph=True)`, and a captured
  producer-to-`ModelOptFp4LinearMethod` CUTLASS tuple chain all passed:
  **8 tests passed**. The existing Qwen3.5 ModelOpt CPU file also passed
  **8 tests**; Python compilation, module import, and `git diff --check`
  passed.
- Wired the producer only in native-Windows `Qwen2MoeMLP` when the down
  projection is TP1, serialized per-tensor non-AWQ ModelOpt FP4 using
  FlashInfer CUTLASS, the hidden width is supported, and 4over6 is inactive.
  The down projection's existing explicit prequantized-tuple contract is
  enabled only in that branch. Every other quantization, topology, platform,
  and Qwen path retains the staged activation and quantizer.
- Pre-launch state: commit `e09e43171d`, only the PERF-027 source, tests, and
  ledgers dirty; port 30000 free; no compiler worker; RTX 5090 at 1,157 MiB
  used / 31,031 MiB free, 21% transient utilization, 30 C.
- Launched the explicit selective profile (not the production default):
  `$env:SGLANG_FLASHINFER_AUTOTUNE_EXTEND='1';`
  `.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1`
  `-ModelPath`
  `C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4`
  `-ChunkedPrefillSize 7680 -RandomSeed 615388882`.
  Startup artifact:
  `perf027-exact-fused-server-20260820-2033.log`.
- Readiness passed on listener PID 12320. Both target and draft KV pools
  resolved to exactly 200,000 tokens; 110 selected target FP4 tactics were
  promoted. Target verify, draft decode, and draft extend graph captures
  completed in 15.65, 1.06, and 0.86 s with 4.58 GiB reported free.
- Two exact warmups plus five cache-flushed `199000+16` scores produced:
  - prompt `3044.589, 3014.218, 2984.494, 2984.251, 2940.207` tok/s,
    mean **2993.552**;
  - legacy generation `77.939, 96.132, 76.080, 86.495, 95.414` tok/s,
    mean **86.412**;
  - TTFT mean **66.485194 s** and E2E mean **66.660452 s**.
  All completed exact `199016` with `finish_reason=length`, but selected
  digest `9db488121a2e5f6b7a64dfb69ba000c62910b06e09ef7da340854481f6621375`
  instead of PERF-028's `cdf5bb57...f647d9`.
- Three exact `199000+512` scores measured prompt
  `3030.638, 3013.477, 2957.484` tok/s (mean **3000.533**) and generation
  `114.263, 119.534, 112.829` tok/s (mean **115.542**). TTFT/E2E means were
  **66.328776/70.754119 s**. All completed exact `199512`, but the changed
  output digest `deb15a60...bf40a0` confirms a different trajectory.
- Interpretation: the +1.126% prompt and -0.744387 s TTFT movement versus the
  prior PERF-028 fused arm is large enough to continue, but neither output nor
  long decode qualifies. Isolated random inputs, graph replay, fullgraph, and
  a small real tuple consumer were exact; the next gate must locate a
  real-layer divergence, including exhaustive BF16 edge values and an
  FP4-GEMM -> fused producer -> FP4-GEMM chain. Artifacts:
  `perf027-exact-fused-window-20260820-2038.log` and
  `perf027-exact-fused-long-20260820-2040.log`.

### 2026-08-20 21:08 PDT - PERF-027 finite-BF16 sweep reproduces the numerical defect

- Resumed from session `fd2e8d01-e225-4b48-9ab3-4d118100a4a9` on commit
  `e09e43171d` with only the PERF-027 source, tests, and ledgers dirty. Port
  30000 was free, no server or CUDA/compiler worker was active, and the RTX
  5090 was at ordinary display residency: 1,341 MiB used, 30,847 MiB free,
  30 C, and 5% sampled utilization.
- Reused the prior session's numerical-localization script after fixing its
  diagnostic-only two-dimensional activation index. The first invocation
  failed before CUDA initialization because `PYTHONPATH` was absent; the
  corrected command was:
  `$env:PYTHONPATH=(Resolve-Path .\python).Path;`
  `.\scripts\windows\invoke_cuda_python.ps1 -Script`
  `C:\Users\Daniel\.copilot\session-state\fd2e8d01-e225-4b48-9ab3-4d118100a4a9\files\perf027_numerical_localization.py`.
- The sweep paired all **65,280 finite BF16 bit patterns** as gate values with
  a 7,919-position rotation as up values. At each tested global scale
  (`389.212673, 286.720003, 86.015997, 40, 20, 10`), the fused producer
  differed from staged activation plus `fp4_quantize` in exactly **520 packed
  bytes** and **zero scale bytes**.
- The first mismatches are finite underflow cases: gate values near
  `1.82e-11`, up values near `4.36e-30`, and staged BF16 products equal to
  zero or the minimum subnormal. The staged path packs `0x00`; the fused path
  packs `0x77` while retaining the same scale byte. This is a concrete
  numerical defect that random-normal production-shape tests did not cover.
- The separate FP4 gate-projection chain still matched exactly in eager,
  `torch.compile(fullgraph=True)`, and three mutable captured replays. The
  finite underflow mismatch, rather than generic graph capture, is now the
  first repair target.
- Artifact:
  `C:\Users\Daniel\.copilot\session-state\93a94358-a792-4795-bcc3-a02f1f278ae6\files\perf027-edge-detail-20260820-2108.log`.
  Do not retain or benchmark PERF-027 again until every finite BF16 packed and
  scale byte matches the staged path and the full-model deterministic digest
  is restored.

### 2026-08-20 21:31 PDT - PERF-027 repaired as an eager-only exact producer

- A deployment-equivalent arithmetic probe exposed a second, independent
  contract. The former eager Windows `SiluAndMul.forward_native` call matched
  the explicit two-rounding staged reference exactly, but
  `torch.compile(fullgraph=True)` fused the native expression and changed the
  quantized tuple:
  - M1: **63 packed / 18 scale-byte** mismatches versus staged.
  - M3: **216 packed / 51 scale-byte** mismatches versus staged.
  - The exact fused producer still had zero ordinary-random mismatches versus
    eager staged at both shapes.
- This proves that replacing every phase with the eager-exact producer changed
  the established compiled M3 target-verification function. The candidate now
  runs only when `torch.compiler.is_compiling()` is false. Eager 7680/7000
  prefill uses the fused exact producer; the compiled M3 target graph retains
  its former `SiluAndMul.forward_native` plus `fp4_quantize` path and RNG/logit
  trajectory.
- The probe initially attempted to instantiate `SiluAndMul` without a
  published runtime-context `exec` namespace and failed before measurement.
  The corrected probe called the class's exact native expression directly.
  Artifact:
  `C:\Users\Daniel\.copilot\session-state\93a94358-a792-4795-bcc3-a02f1f278ae6\files\perf027-compiled-baseline-20260820-2122.log`.
- Repaired the finite-domain mismatch by canonicalizing only the final rounded
  BF16 subnormal product to sign-preserving zero before NVFP4 conversion. This
  recreates the staged global-memory boundary into FlashInfer's FTZ-compiled
  quantizer without enabling fast SiLU math or flushing a subnormal rounded
  activation before multiplication.
- Added fail-closed SM100+ host and Python gates plus explicit 32-byte input
  and 8-byte output alignment checks. Added vectorized all-finite-BF16
  coverage in both the compact non-TMA shape `(4080,16)` and the native TMA
  shape `(1024,512)`.
- Focused native CUDA results: **10 passed**, including production
  M1/M3/M7000/M7680 equality, both finite-domain sweeps, mutable graph replay,
  fullgraph compilation, and the captured ModelOpt tuple consumer. The
  existing Qwen3.5 ModelOpt CPU suite passed **8 tests**; Python compilation
  and `git diff --check` passed.
- The next gate is a clean selective-profile relaunch. It must restore the
  established deterministic short and long digests while retaining a prompt
  gain. Decode is intentionally the PERF-028 control path until a separately
  exact compiled-semantics producer is proven.

### 2026-08-20 22:04 PDT - repaired PERF-027 restores exact output and retains the prefill win

- Two attempts to launch through `.venv\Scripts\sglang.exe` failed immediately
  with `uv trampoline failed to canonicalize script path`; the session's
  relocated virtual environment was created after the earlier successful
  launches. No model process or CUDA context survived either failure.
- Launched the same resolved launcher arguments through the still-supported
  `D:\sglang\.venv\Scripts\python.exe -m sglang.launch_server` entry point
  after sourcing the native CUDA/MSVC environment. The full resolved argument
  list is the first line of
  `perf027-repaired-server-direct-20260820-2152.log`; all simulation, tree,
  SWOR, adaptive, and device-cycle controls remained absent.
- Both target and draft KV pools resolved to exactly 200,000 tokens. The
  selected 16,384-token target EXTEND pass ran, and target verify, draft
  decode, and draft extend graphs captured in **18.17, 61.18, and 0.92 s**.
  The long draft-decode capture included a one-time missing-cache autotune.
  `/health` returned 200; `/model_info` reported image/audio understanding
  false. Listener PID 21336 descended from Python 33140 and detached launcher
  PowerShell 22096.
- After two exact warmups, five cache-flushed `199000+16` scores produced:
  - prompt `2977.888, 3008.041, 2946.967, 2968.875, 3034.603` tok/s,
    mean **2987.275**, median **2977.888**, CV **1.150%**;
  - short generation `95.687, 90.710, 113.181, 95.859, 97.245` tok/s,
    mean **98.536**, retained only as 15-interval evidence;
  - TTFT mean **66.622932 s** and E2E mean **66.776008 s**.
- Every short request completed exact `199016`, returned
  `finish_reason=length`, and restored digest
  `cdf5bb57b88deaa7515abaedf36406d10494599fce2e23eeaa400461d9f647d9`.
  Relative to the PERF-028 fused arm, prompt improved
  **27.047 tok/s / 0.914%** and TTFT improved **0.606649 s / 0.902%**.
  Relative to the fresh current-source baseline, prompt improved **1.698%**
  and TTFT improved **1.126997 s**.
- Three exact `199000+512` requests measured:
  - prompt `3040.821, 2982.656, 2980.554` tok/s, mean **3001.344**;
  - generation `117.174, 114.334, 114.168` tok/s, mean **115.225**, CV
    **1.466%**;
  - TTFT/E2E means **66.309344/70.744768 s**.
  All completed exact `199512` and restored digest
  `cac0c6e4fab3115102a9a0c4163e4465068fba30cb09f0bb5556c7021e4a2092`.
  The prompt gain persisted; decode is statistically unchanged and remains
  governed by PERF-028 plus environment variance.
- Retain PERF-027 only for eager execution. Its mechanism and full-model
  signal agree: the staged producer changed `664.112 -> 366.176 us` at M7000
  and `730.416 -> 402.704 us` at M7680, while the repaired exact request
  improved prompt/TTFT without changing output. The compile guard deliberately
  leaves M3 decode unchanged.
- Re-resolved and stopped only the verified tree rooted at PID 22096:
  worker leaves `11992/52320`, listener `21336`, parent Python `33140`,
  console `45832`, then launcher PowerShell `22096`. All known PIDs exited,
  port 30000 was free, compiler workers were absent, and the RTX 5090 returned
  to 1,945 MiB display residency with 30,243 MiB free.
- Artifacts:
  `perf027-repaired-server-direct-20260820-2152.log` and
  `perf027-repaired-long-20260820-2159.log` under session
  `93a94358-a792-4795-bcc3-a02f1f278ae6`.
- Added CPU routing ratchets after shutdown: eager Qwen MLP execution must pass
  the prequantized tuple, while compile tracing must retain the native
  activation path. The expanded Qwen3.5 ModelOpt suite passed **10 tests**;
  Python compilation and `git diff --check` passed.

### 2026-08-20 22:13 PDT - PERF-029 compiled-semantics producer passes isolated gates

- Probed FlashInfer's native expert SwiGLU-to-NVFP4 kernel as a candidate for
  the established Inductor M3 function. With `B=1`, a valid-row-count mask,
  and caller-owned zeroed scale storage, its packed and complete swizzled scale
  bytes matched the compiled `F.silu(gate) * up` plus `fp4_quantize` oracle
  exactly at M1 and M3.
- The public expert API is not directly usable: it exposes grouped-GEMM views
  and leaves padded scale positions uninitialized. Its M3 packed bytes were
  exact, but the returned scale storage differed outside the written offsets.
  The raw caller-owned-buffer entry proved the arithmetic itself is exact.
- Added a separate fast-math specialization to the retained dense native
  producer. It uses the expert kernel's one-final-rounding `__expf` contract,
  writes every padding byte deterministically, and retains the existing PDL
  wait/trigger boundary. The eager two-rounding module remains separately
  compiled without fast math.
- Isolated M1 medians were **67.904 us compiled staged**, **15.200 us raw
  expert**, and **25.184 us PDL-safe dense custom**. M3 medians were
  **70.848, 15.104, and 25.152 us**, respectively. The retained custom form
  removes about **45.7 us per target MLP** while preserving the dense tuple.
  Artifact:
  `perf027-compiled-custom-20260820-2235.log`.
- Wired only compile tracing to the compiled-semantics producer; eager prefill
  retains PERF-027. Focused coverage now includes compiled M1/M3 byte equality,
  mutable graph replay, nested fullgraph compilation, and a compiled captured
  producer-to-ModelOpt tuple chain. Results: **16 native CUDA tests** and
  **10 Qwen3.5 ModelOpt CPU tests** passed; Python compilation and
  `git diff --check` passed.
- This is not yet retained. The next gate is a selective full-model launch
  requiring the old short and long deterministic digests, unchanged 200K
  pools, successful graph capture, a shorter target graph/cycle, and a
  repeatable generation gain.

### 2026-08-20 22:31 PDT - PERF-029 removed after full-cycle attribution

- The selective server resolved both 200K pools, promoted the same 110 target
  tactic entries, and captured target verify, draft decode, and draft extend
  in **17.37, 1.14, and 0.90 s**. Target graph memory usage fell to 0.04 GB;
  `/health` and the language-only model surface passed.
- After two exact warmups, five cache-flushed `199000+16` scores produced:
  - prompt `2962.354, 2913.028, 2987.034, 2951.224, 2945.581` tok/s,
    mean **2951.844**, CV **0.911%**;
  - short generation `94.156, 92.580, 102.402, 98.039, 101.088` tok/s,
    mean **97.653**;
  - TTFT/E2E means **67.419972/67.573812 s**.
  Every request completed exact `199016` and retained digest
  `cdf5bb57b88deaa7515abaedf36406d10494599fce2e23eeaa400461d9f647d9`.
- Three exact `199000+512` requests measured prompt
  `2961.809, 3017.342, 2972.733` tok/s (mean **2983.961**) and generation
  `116.146, 113.944, 118.485` tok/s (mean **116.192**, CV **1.954%**).
  All completed exact `199512` and retained digest
  `cac0c6e4fab3115102a9a0c4163e4465068fba30cb09f0bb5556c7021e4a2092`.
- Profiled the ordinary M3 server with `bench_target_verify_width.py` over 512
  sampled output tokens. Acceptance was 2.197425 over 233 verification cycles.
  Full target-start-to-target-start samples measured **16.389343 ms mean,
  16.044576 ms median, 15.628654 ms minimum**, and 20.934177 ms maximum.
  The median is effectively identical to the existing **16.058328 ms** M3
  control, so the isolated 45.7 us-per-layer launch result did not become
  serialized full-cycle work.
- The adjacent long client mean moved +0.839%, while short prompt/TTFT moved in
  the opposite direction. With a neutral device cycle and current WDDM
  variation, this is statistical noise rather than a retained gain.
- Re-resolved and stopped only the verified tree rooted at PID 12000:
  worker leaves `5068/45972`, listener `33244`, parent Python `26544`, console
  `56132`, then launcher PowerShell `12000`. All known PIDs exited, port 30000
  was free, compiler workers were absent, and the GPU returned to 1,748 MiB
  display residency with 30,440 MiB free.
- Removed the compiled-semantics module, Qwen routing, and added tests. PERF-027
  eager fusion remains the active source. Artifacts:
  `perf029-server-20260820-2215.log`,
  `perf029-exact200k-20260820-2216.log`,
  `perf029-long-20260820-2225.log`, and
  `target_width_m3-20260820-223006`.
