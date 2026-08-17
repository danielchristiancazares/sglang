# Decision ledger

This ledger records choices that still govern the native-Windows Qwen3.8
system. Exact sample lists, commands, incident detail, and intermediate states
remain in [`experiment-log.md`](experiment-log.md).

**Reconciled through:** 2026-08-16 19:06 PDT.

## Selected production choices

| Decision | Selected choice | Durable evidence |
|---|---|---|
| Checkpoint | RadixArk Qwen3.8-27B NVFP4 | Restored coherent reasoning and tools; remained the fastest qualified real-sampled line |
| Capacity | Real 200K target and draft pools | Exact `199016` passed repeatedly; 232K reached 98 MiB free before cache flush and was rejected for operating margin |
| Model surface | Language-only with Qwen3 reasoning and Qwen3 Coder tools | Preserves required behavior and VRAM; image/audio remain disabled |
| FlashInfer | Clean native-Windows port of 0.6.17 | Passed JIT/kernel tests, fixed long-prefill correctness, and satisfies the SGLang version contract |
| Prefill | FlashInfer, chunk size 4096 | Strong short and long prefill without the pressure and throughput losses of 8192 |
| Target verify/decode | TRT-LLM MHA/XQA | Qualified real throughput and exact 200K capacity; compact unread mask removes redundant generic work |
| Draft decode | TRT-LLM MHA/XQA | Controlled gain over Triton draft decode; semantics and long ladder passed |
| Draft extend | Captured `DRAFT_EXTEND_V2` graph | Removed an eager dispatch wall and contributed to the later two-step winner |
| Linear attention | Triton GDN with ReplaySSM | Correct Qwen recurrent-state handling in the selected linear speculative topology |
| Draft KV | FP8 E4M3 | Reduced memory and improved the selected topology while preserving behavior |
| Speculation geometry | 2 steps, 3 draft tokens, EAGLE top-k 1 | Qualified **122.712 tok/s** real mean and **171.263 tok/s** fixed mean |
| Proposal alignment | Draft top-k 20, captured inside one multi-step CUDA graph | Preserves exact q for rejection, improves acceptance, and avoids Python between draft depths |
| Chain metadata | Native C++/CUDA fixed-chain path with distinct per-cycle outputs | 4.227x isolated metadata speedup while preserving asynchronous output lifetimes |
| Sampling | FlashInfer | Native CUDA renormalization controls the speculative target path; fallback sampling remains available |
| Native elementwise/norm | C++/CUDA SiLU, RMSNorm, Gemma RMSNorm, and qualified sigmoid-multiply dispatch | Exact or explicitly gated parity with large isolated reductions in eager launch cost |
| GEMM tuning | FP4 autotune; skip FP8 GEMM autotune | FP4 tactics improved decode; FP8 tactics regressed decode and prefill |
| Workspace | 128 MiB | Wins decode and long prefill; 64 MiB fails required graph allocation |
| Compile mode | `default`, with established partial fallbacks | Five-run fixed-work win over other compile/fallback arrangements |
| Scheduling | Receive interval 4; stream interval 4; incremental output | Measured fixed-work wins while retaining client streaming behavior |
| Implementation language | C++/CUDA hot paths with thin Python integration | Explicit user direction after the display-GPU incident; preserves graph capture and native dispatch |
| Tree/SWOR implementation | Retained as opt-in experimental infrastructure | Exactness, recurrent commit, graph replay, diagnostics, and analyzers are proven; current economics lose to linear production |

## Qualified reference results

| Result | Accepted value |
|---|---|
| Real sampled `6213/512` | **122.712 tok/s** ten-run mean, **122.371** median, **137.074** peak |
| Fixed accepted-length-3 `6213/512` | **171.263 tok/s** five-run mean |
| Native two-step acceptance | **2.318174** mean emitted/accepted tokens per verification over five probes |
| Near-limit `199000/16` | **2608.263 prompt tok/s**, **102.358 generation tok/s**, exact `199016` total |
| Final production graph headroom | **1.84 GiB** reported after restored 200K capture |

## Closed or rejected candidates

### Checkpoints, quantization, and kernels

| Candidate | Status | Why |
|---|---|---|
| GGUF as production checkpoint | Superseded | Base NVFP4 improved prompt throughput by roughly 12.8x and E2E by 4.2x on `6213/128` |
| FlashInfer 0.6.11 attention | Rejected | Faster synthetic result produced degenerate repetition on the real long OpenCode prompt |
| Native target NVFP4 KV | Rejected | Recovered about 2.2 GiB but corrupted thinking and tool behavior |
| Stock `nvfp4_online` for draft | Superseded by dense experiment | The checkpoint is dense; the original MoE-only path left draft storage and lost fixed work |
| Full online FP8 MTP | Rejected for throughput | **167.023 tok/s** fixed versus the **171.263** BF16 control; activation quantization erased the GEMM saving |
| Full online MXFP8 MTP | Rejected for throughput | Mechanism qualified end to end but reached **163.457 tok/s** fixed |
| Dense online NVFP4 MTP | Rejected for throughput | Mechanism and graph replay qualified; **164.094 tok/s** fixed with about 0.46 GiB memory saving |
| Gittensor ModelOpt FP4 checkpoint | Rejected as production winner | Better acceptance and smaller residency, yet **119.092 tok/s** real and **154.883 tok/s** fixed lost to RadixArk |
| Gittensor/RadixArk hybrid `lm_head` | Closed by user direction | Source checkpoints remain immutable and RadixArk was restored as the active checkpoint |
| CUTLASS channelwise-FP8 dispatch | Rejected | Alignment and numerical tests passed; robust paired medians showed no material win |
| CUTLASS DSL / FlashInfer GDN on Windows | Unavailable | Required Windows DSL support/package is absent for this native path |
| Fully compiled repaired Triton kernels | Rejected | Correct yet slower, with very long startup compilation |
| Explicit compiler-disable boundaries | Rejected | Changed graph segmentation and lost throughput |
| Native fused-add RMSNorm on the target path | Gated | Residual is exact while output can move by one BF16 step; any draft-only use needs a separate controlled gate |

The optional Windows quantization registrations, conversion repairs, backend
selection, and isolated tests remain valuable compatibility work. Their
production performance status stays closed unless the cost topology changes.

### Speculation, proposal, and scheduling

| Candidate | Status | Why |
|---|---|---|
| One-step/two-token MTP | Rejected | Fixed samples near 102 tok/s expose an insufficient emission ceiling |
| Static three-step/four-token MTP | Rejected for real production | Full acceptance crossed 200 tok/s once, while honest sampled mean was **117.239 tok/s** |
| Adaptive 2/3 depth, aggressive policy | Rejected | Oscillation reduced acceptance and first real sample reached only 100.739 tok/s |
| Adaptive 2/3 depth, sparse policy | Rejected | Two real windows combined to **110.276 tok/s** |
| No MTP | Superseded | Useful control and slower than the trained RadixArk MTP path |
| Draft proposal top-k 8 | Rejected | Lower acceptance and **119.741 tok/s** three-run mean |
| Earlier eager aligned top-k 20 | Superseded by captured alignment | Eager/per-step graphs imposed a large fixed-work tax; single-CG alignment later made top-k 20 the selected path |
| Generic sampler swap to PyTorch | Rejected | Steady EAGLE bypassed the selector and paired fixed medians overlapped |
| Top-k 2 speculative tree in the old linear path | Rejected | No correct sampled path with native-Windows rejection, XQA, and ReplaySSM at that stage |
| Reusable fused metadata output buffers | Rejected | Simulated fixed work rose, while exact-seed real output showed a scheduling/aliasing regression |
| Continuous decode steps 4 | Rejected | Increased TTFT and E2E on the matched control |
| BF16 Mamba state | Rejected | Slower and changed deterministic output; FP32 remains selected |
| ReplaySSM on the original non-speculative GGUF route | Topology-specific rejection | It later became selected for the linear-chain MTP topology |

### Memory, context, and environment

| Candidate | Status | Why |
|---|---|---|
| Chunk size 8192 | Rejected | Lost on short work and collapsed after repeated 32K requests under VRAM pressure |
| Workspace 64 MiB | Rejected | Deterministic graph-capture buffer overflow; 128 MiB is the floor |
| FP8-only autotune | Rejected | Lost decode and long prefill while reducing headroom |
| Full FP4+FP8 autotune | Rejected | Inferior to FP4-only tuning and regressed large prefill |
| 232K production context/pool | Rejected | Exact `231000+16` passed, but only 98 MiB remained before cache flush |
| NVML polling or keepalive | Rejected | Apparent gains failed to persist; WDDM client traffic explained the variance |

### Exact tree and SWOR experiments

| Candidate | Status | Why |
|---|---|---|
| Target-only M12 tree | Rejected as production topology | About **2.9436** emitted/traversal and one 104.145 tok/s stream; cycle cost required more output than the shape can emit |
| Target-only M8 tree | Rejected | Saved 4.81% captured work and lost about 4.52% yield; stream reached 94.080 tok/s |
| Six-step/depth-only tree | Rejected | Mean **3.1025** emitted/traversal and only 87.589 tok/s |
| Fully normalized aligned tree scoring | Rejected as default | Reduced M12 yield by about 3.45% versus plain scoring |
| Scalar depth discount 0.8 | Rejected | **2.9286** emitted/traversal versus plain M12 **2.9436** |
| Initial M12 exact SWOR topology | Rejected | **2.9653** emitted/traversal and **84.713 tok/s** real mean |
| Topology-only SWOR search at current q | Exhausted | Optimistic 32-node search reached only **4.0921** expected outputs and remained cost-limited |

The exact tree implementation itself is retained. It includes native target-
only/SWOR sampling, sparse support up to 64 entries with dense fallback,
low-rank GDN tree replay, accepted-path recurrent/conv commit, path and overlap
oracles, custom topology parsing, profiling, and offline search tools.

## Protected boundaries

- Preserve the user's worktree and unrelated `sglang.bundle`.
- Preserve the original FlashInfer checkout and the clean 0.6.17 Windows port
  as separate provenance lines.
- Preserve downloaded RadixArk and Gittensor checkpoints byte-for-byte; place
  derived artifacts at separate paths with provenance and checksums.
- Leave the protected CUDA compatibility headers untouched. Their recorded
  SHA-256 is
  `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.
- Keep OpenCode2's cloud-model configuration stable during local server tuning;
  use process-scoped aliases or wrappers.
- Use exact process ancestry for server lifecycle actions and preserve every
  unrelated user process.

## Reopening criteria

The exhaustive optimization goal is complete. A closed branch reopens when an
explicit user request or materially new evidence changes its governing
assumption:

- new kernels or hardware alter per-depth draft/target cost;
- proposal overlap improves enough to change tree yield;
- a new checkpoint passes the same preserved-thinking/tool/capacity contract;
- dependency support changes the native-Windows backend boundary;
- a newly measured production gap survives matched controls and environmental
  accounting.

A historical peak, simulated acceptance, microbenchmark, source inspection, or
single stochastic window remains supporting evidence within the full promotion
contract.
