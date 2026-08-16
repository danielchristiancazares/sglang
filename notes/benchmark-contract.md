# Benchmark and acceptance contract

Use this contract for comparisons with the qualified winner. A result is
comparable only when the request shape, server mode, cache treatment, sampling
profile, and GPU environment are identified.

## Reference environment

The notebook's qualified run used:

- native Windows and an RTX 5090;
- PyTorch `2.13.0+cu130`;
- CUDA runtime 13.0 with CUDA toolkit 13.3.33;
- Triton Windows `3.7.1.post27`;
- the clean Windows FlashInfer `0.6.17` port;
- the RadixArk Qwen3.8-27B NVFP4 checkpoint.

These versions can drift. Record the live environment when remeasuring.

## Workloads

| Name | Shape and mode | Purpose |
|---|---|---|
| Smoke | `256/16` | API, tokenizer, SSE, finish reason, and basic output health |
| Historical control | `6213/128`, temperature 0 | Compare with the early GGUF and base-NVFP4 results |
| Current fixed control | `6213/512`, temperature 0, accepted length 3 | Attribute kernel and dispatch cost with fixed speculative work |
| Current real control | `6213/512`, normal rejection sampling | Measure production generation throughput |
| Sampled profile | temperature `1.0`, top-p `0.95`, top-k `20`, presence `1.5` | Match the user's Qwen reasoning profile |
| Long ladder | `32768/16`, `32768/512`, `65536/16` | Catch prefill, residency, and long-decode regressions |
| Capacity gate | `199000/16` | Prove exact total `199016` inside the real 200K pool |

`6213` input tokens come from the local calibrated OpenCode-shaped fixture in
[`benchmark/windows/qwen38_local_prompt.json`](../benchmark/windows/qwen38_local_prompt.json).

## Commands

Greedy current-shape control:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512
```

Sampled-profile control:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 6213 --output-tokens 512 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5
```

Native speculative acceptance counters at the sampled profile:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_spec_acceptance.py
```

Near-limit capacity:

```powershell
.\.venv\Scripts\python.exe .\scripts\windows\bench_openai_stream.py --input-tokens 199000 --output-tokens 16 --timeout 600
```

For fixed-work attribution, launch with
`serve_qwen38_27b_nvfp4_5090.ps1 -SimulateAcceptedLength 3`. A server started
this way is simulation-only and cannot satisfy production or semantic gates.

## Measurement procedure

1. Confirm resolved server arguments, all expected graph-capture markers,
   `/health`, and absence or presence of fixed-acceptance simulation.
2. Use an exact-shape warmup and explicit cache flushes. The benchmark handles
   its calibrated request and reports TTFT, end-to-end time, prompt rate,
   steady decode rate, token counts, finish reason, output length, and digest.
3. Run at least five consecutive samples. For a production promotion, take an
   independent second real-sampling window.
4. Preserve output digests for deterministic fixed-work comparisons. Treat
   stochastic digest changes as an investigation signal alongside semantics
   and acceptance, rather than as an automatic failure.
5. Record GPU clocks, power, utilization, free VRAM, and competing WDDM clients
   when results move materially. Chrome, ZCode, and Epic Games Launcher
   produced proven contention during the notebook run.
6. Flush the server cache after long or memory-heavy requests before inferring
   a steady-state regression.

Report means and individual samples. Label cold, warm, contended, simulated,
and unsimulated results explicitly.

## Promotion gates

A faster number alone is incomplete. Before selection, require:

- coherent preserved reasoning with the sampled profile;
- correct arithmetic final answer (`703` is the established probe);
- exactly one parsed `multiply({"a":37,"b":19})` call;
- `/model_info` showing image and audio understanding disabled;
- exact `199000+16` capacity when memory layout, graph coverage, workspace,
  cache dtype, or sampling residency changes;
- focused tests, Python compilation, PowerShell parsing, and
  `git diff --check` for the touched surface;
- an unsimulated production relaunch using launcher defaults.

Standalone OpenCode2 is the final client integration check. Keep its workload
and provider shape fixed when comparing another server implementation.

## Interpretation rules learned from the notebook

- Fixed accepted length isolates execution cost; it says nothing about proposal
  quality.
- Real speculative throughput varies with acceptance. Use native acceptance
  counters alongside TPS.
- Source inspection and CPU tests do not establish GPU serving, quality, or
  performance.
- A short near-limit generation sample is a capacity and routing gate, not a
  stable decode benchmark.
- Repeated low samples with unchanged output can come from WDDM contention or
  residency pressure. Measure those conditions before assigning a code cause.
