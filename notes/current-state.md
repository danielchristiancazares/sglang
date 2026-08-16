# Current state

**Source cutoff:** `NOTES.md` through 2026-08-16 10:58 PDT.

**Source revision:** `edc03bc22f02707bf06b0aff3f6feaf5eaa2365b`, plus the
user-owned uncommitted work described below.

## Qualified winner

The qualified production configuration is native-Windows SGLang serving
`C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk` on the RTX 5090.
It provides a real 200,000-token pool, preserved reasoning, tool calls, and a
language-only model surface.

| Area | Selected value |
|---|---|
| Endpoint | `http://127.0.0.1:30000/v1`, model `qwen3.8-27b` |
| Capacity | context `200000`; total-token pool `200000`; one running request |
| Model surface | `--language-model-only`; Qwen3 reasoning parser; Qwen3 Coder tool parser |
| Target attention | FlashInfer prefill; TRT-LLM MHA/XQA decode and target verification |
| Draft attention | TRT-LLM MHA/XQA; captured `DRAFT_EXTEND_V2` graph |
| Linear attention | Triton GDN with ReplaySSM speculation |
| Speculation | NEXTN; 2 steps; 3 draft tokens; top-k 1; rejection sampling |
| KV | checkpoint-selected target KV; FP8 E4M3 draft KV; page size 64 |
| Sampling | FlashInfer |
| Prefill | 4096-token chunks |
| Mamba | 4 slots; `extra_buffer_lazy`; FP32 state |
| GEMM tuning | FlashInfer CUTLASS FP4; autotune enabled; FP8 GEMM autotune skipped |
| Compile/graphs | torch compile mode `default`; batch-one full decode graphs |
| Scheduling/streaming | scheduler receive interval 4; stream interval 4; incremental output |
| Workspace | 128 MiB FlashInfer workspace, the measured functional floor |
| Draft-q alignment | disabled (`SpeculativeDraftSamplingTopK=0`) |

These values match the defaults in
[`serve_qwen38_27b_nvfp4_5090.ps1`](../scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1)
at the source cutoff.

## Qualified measurements

| Gate | Result |
|---|---|
| Fixed work, exact `6213/512`, accepted length 3 | **159.973 tok/s** five-run mean; `164.063` peak |
| Real rejection sampling, exact `6213/512` | **117.794 tok/s** across two five-run windows (`117.117`, `118.470`) |
| Improvement over the prior qualified real mean | **+6.36%** versus the `110.750 tok/s` three-step leader |
| Near-limit capacity, exact `199000+16` | **2608.263 prompt tok/s**, `102.358` generation tok/s, `199016` total |
| Behavior | coherent preserved thinking; correct `703`; exactly one `multiply({"a":37,"b":19})` tool call |
| Surface | image and audio understanding reported false |

The newer dispatch topology changed the cost balance enough for two-step MTP to
overtake three-step. Native two-step acceptance measured length `2.3167`, rate
`0.6561`, `290/442` accepted proposals, histogram `[52,48,121]`, and 221
verification cycles. It needed only seven more cycles than the matched
three-step control while making each cycle materially cheaper.

## Latest experiment and handoff

The old `95.490 tok/s` two-step rejection belonged to the pre-dispatch
topology. Retesting after draft-extension graph capture and target XQA produced
the new qualified leader. One-step remains closed: forced-acceptance samples
`102.142`, `102.107`, and `101.282 tok/s` show that its two-token emission
ceiling cannot repay cycle cost.

Launcher defaults now resolve to two steps and three draft tokens. At the
source cutoff, the default-only unsimulated server was healthy on port 30000
under PowerShell parent PID `32156` and worker PID `15352`; `/health` returned
HTTP 200. Target verify, draft decode, and draft extend graphs all captured,
with 1.70 GB reported post-capture headroom.

The next measured branch is adaptive depth over **2 and 3 only**. Any adaptive
candidate must beat two independent real windows and retain exact-200K
capacity. The generic sampling-backend branch is closed; steady EAGLE bypasses
that selector and its PyTorch/FlashInfer medians overlapped.

## Behavior and capacity invariants

Every promoted candidate must retain all of these:

- real `200000` context and token-pool capacity, including exact total `199016`;
- Qwen recommended sampled reasoning: temperature `1.0`, top-p `0.95`, top-k
  `20`, presence penalty `1.5`;
- preserved `reasoning_content` and normal completion behavior;
- parsed tools with correct arguments and `finish_reason=tool_calls`;
- vision and audio disabled;
- no fixed-acceptance simulation in production;
- CUDA headers outside the edit boundary.

## Workspace boundary at the cutoff

The active uncommitted stream includes speculative proposal alignment, graph
buffer handling, launcher controls, focused tests, and the acceptance probe.
Important paths are:

- `python/sglang/srt/arg_groups/speculative_hook.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`
- `python/sglang/srt/speculative/eagle_worker_v2.py`
- `python/sglang/srt/speculative/spec_utils.py`
- `scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1`
- `scripts/windows/bench_spec_acceptance.py`
- `test/registered/unit/spec/test_eagle_draft_cuda_graph_runner.py`
- `test/registered/unit/spec/test_draft_proposal_sampling.py`

`NOTES.md` is also modified. `sglang.bundle` is unrelated and user-owned.
Preserve all of these boundaries during recovery or cleanup.
