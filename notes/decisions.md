# Decision ledger

This ledger records the choices that still matter to the current system. Full
sample lists and incident detail remain in [`NOTES.md`](../NOTES.md).

## Selected production choices

| Decision | Selected choice | Evidence |
|---|---|---|
| Checkpoint | RadixArk Qwen3.8-27B NVFP4 | Restored coherent reasoning and tools; became the basis of all later qualified work |
| Capacity | Real 200K target and draft pools | Early MTP embedding/head sharing moved before pool sizing; exact `199016` repeatedly passed |
| FlashInfer | Clean Windows port of `0.6.17` | Passed JIT/kernel tests, fixed long-prefill correctness, and met SGLang's version contract |
| Prefill | FlashInfer, chunk size 4096 | Strong short and long prefill without the pressure and throughput losses of 8192 |
| Target verify/decode | TRT-LLM MHA/XQA | Qualified real ten-run mean `110.750 tok/s`; exact 200K capacity retained |
| Draft decode | TRT-LLM MHA/XQA | Controlled gain over Triton draft decode; semantics and long ladder passed |
| Draft extend | Captured `DRAFT_EXTEND_V2` graph | Raised fixed work from `135.167` to `145.941 tok/s` before the XQA verify gain |
| Draft KV | FP8 E4M3 | Raised real mean to `98.126 tok/s` in the pre-dispatch topology and reduced memory |
| Speculation geometry | 2 steps, 3 draft tokens, top-k 1 | Qualified real mean `117.794 tok/s`; cheaper width outweighed its small acceptance-length loss |
| Proposal alignment | Established default path; opt-in top-k value 0 | Top-k-20 raised accepted length only 4.39%, then lost fixed and real throughput |
| Sampling | FlashInfer | Steady EAGLE bypasses the generic selector; paired fixed-work medians overlap PyTorch. Retained for ordinary/fallback sampling |
| GEMM tuning | FP4 autotune; skip FP8 GEMM autotune | FP4 tactics improved decode; FP8 tactics regressed decode and prefill |
| Workspace | 128 MiB | Wins decode and long prefill; 64 MiB is below the graph's required allocation |
| Compile mode | `default`, with the established partial fallbacks | Five-run fixed-work win over other compile/fallback arrangements |
| Scheduling | Receive interval 4; stream interval 4 | Measured fixed-work wins; incremental streaming retained |
| Model surface | Language-only, reasoning and tools enabled | Preserves required behavior and VRAM; vision/audio remain disabled |

## Closed or rejected candidates

| Candidate | Status | Why |
|---|---|---|
| GGUF as the production checkpoint | Superseded | Base NVFP4 improved prompt throughput by roughly 12.8x and E2E by 4.2x on `6213/128` |
| FlashInfer `0.6.11` attention | Rejected | Faster synthetic result produced degenerate repetition on the real 8.7K OpenCode prompt |
| Native target NVFP4 KV storage | Rejected | Recovered about 2.2 GB but corrupted thinking and tool behavior |
| Stock `nvfp4_online` for the draft | Rejected | The checkpoint is dense; the shipped path quantizes MoE experts, left draft storage near 6.29 GB, and averaged `112.300 tok/s` fixed work |
| Aligned draft-q top-k 20 | Rejected as a default | Fixed work fell near `142.26`; first real samples were `78.507` and `62.269 tok/s` |
| Generic sampler swap to PyTorch | Rejected | Steady EAGLE bypasses both backends; fixed medians were `148.961` versus `148.603`, with no reproducible memory recovery |
| Top-k 2 speculative tree | Rejected | No correct sampled path with native-Windows rejection sampling, XQA, and ReplaySSM |
| Three-step/four-token MTP | Superseded | `110.750 tok/s` after dispatch fixes; the retested two-step topology reached `117.794` |
| One-step/two-token MTP | Rejected | New fixed-work samples `102.142`, `102.107`, and `101.282` confirm its emission ceiling is too costly |
| No MTP | Superseded | Useful control; slower than the final trained three-step RadixArk MTP path |
| Chunk size 8192 | Rejected | Lost on short work and collapsed after repeated 32K requests under VRAM pressure |
| Workspace 64 MiB | Rejected | Deterministic graph-capture buffer overflow; 128 MiB is the floor |
| FP8-only autotune | Rejected | Lost decode and long prefill; reduced headroom |
| Full FP4+FP8 autotune | Rejected | Inferior to FP4-only tuning and regressed large prefill |
| Fully compiled repaired Triton kernels | Rejected | Correct but slower; startup compilation was also very long |
| Explicit compiler-disable boundaries | Rejected | Changed graph segmentation and lost throughput |
| Continuous decode steps 4 | Rejected | Increased TTFT/E2E on the GGUF control |
| BF16 Mamba state | Rejected | Slower and changed the deterministic output on the GGUF control |
| CUTLASS channelwise-FP8 dispatch candidate | Rejected | Alignment was repaired and numerical tests passed; robust paired medians showed no material win |
| CUTLASS DSL / FlashInfer GDN on Windows | Unavailable | NVIDIA publishes no Windows DSL base and documents CUTLASS 4.x Windows builds as down |
| NVML polling or keepalive | Rejected | Apparent gains did not persist; WDDM client traffic explained the variance |

Some early decisions were topology-specific. In particular, ReplaySSM lost on
the original non-speculative GGUF path and later became part of the selected
linear-chain MTP topology. Use the current selected table when an old entry
appears to conflict.

## Protected boundaries

- Preserve the user's dirty SGLang worktree and unrelated `sglang.bundle`.
- Preserve the original FlashInfer checkout and the clean 0.6.17 port as
  separate provenance lines.
- CUDA headers were explicitly outside the edit boundary. The original and
  clean-port compatibility header were last recorded byte-identical at
  `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.
- Do not promote from source inspection, unit tests, fixed acceptance, or a
  single favorable stochastic window.

## Open leads

The matched default-q acceptance control is complete and the work is paused.
Further work needs a measured gap before implementation:

- narrower or depth-specific proposal calibration;
- adaptive speculative depth over steps 2 and 3 only;
- shape-specific GEMM dispatch;
- length-adaptive target-verification routing if the near-limit generation
  difference reproduces;
- vendor improvements to NVFP4/FP8 GEMM or a supported Windows CUTLASS DSL.
