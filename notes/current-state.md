# Current state

**Reconciled through:** [`experiment-log.md`](experiment-log.md), 2026-09-02
02:20 PDT.

**Live runtime at reconciliation:** no SGLang server is running and port 30000
is free. All verified SGLang, benchmark, and CUDA compiler processes are
absent. After the pinned TurboQuant35 replay, the RTX 5090 reported **1,329
MiB used / 30,859 MiB free**, 18% sampled display utilization, 32 C, and 78.90
W on driver `616.92`; 51,294 MiB of host RAM was available and disk traffic
was 0.524 MiB/s. Unrelated desktop and CPU-work processes were preserved. This
supersedes the earlier September 12 live-runtime snapshot, not its qualified
production selection. Recheck ownership before every GPU action.

## DeepSeek V4.1 ideas without retraining

Qwen3.8-27B has 48 recurrent GDN layers and 16 ordinary global-attention
layers. It has no sliding-window-attention stack, and the production launcher
does not enable HiCache or SSD persistence. A causal encoder/decoder split
would change learned residual flow, while CSA2 cache reuse would substitute
one layer's learned K/V projections for another. Neither preserves this
checkpoint without architectural training. Removing persistent SWA state has
no active storage to remove on this lane.

The directly applicable lower-bit global KV experiment is closed. Current
source reached readiness with native target NVFP4 only after routing
multi-token target verification through the prefill backend, then failed the
first deterministic arithmetic gate by hallucinating unrelated conversations
for all 256 tokens instead of deriving `703`. A separate native SM120
TurboQuant35 admission benchmark then implemented mixed 4/3-bit groups,
Hadamard MSE coding, a one-bit residual, FP16 norms, fused cache writes, packed
attention, and CUDA graph replay. It would save **53.125%** of FP8 KV bytes,
or a projected **3.242 GiB** across the 16 global layers at 200K. The validated
codec nevertheless produced **0.199575** relative-L2 error. The retained
benchmark now pins its fastest sweep layouts: TurboQuant35 segment 512 at both
lengths, FP8 segment 512 at 6,213, and FP8 segment 2,048 at 199K. Their
historical 199K sweep medians were **11,540.973 us** and **3,074.226 us**;
the established XQA FP8 authority remains **271.584 us**. Segment 1,024 and
the other nonselected sweep call sites are gone. No serving cache ABI or
launcher option was added. The MTP-bearing AttnNVFP4 target and explicit
DSpark-v2 draft path remain selected, with no fallback route introduced.

## NVIDIA stock checkpoint option

The immutable `nvidia/Qwen3.8-27B-NVFP4` revision
`fed99d815f4e8c7c616dd3dd6780076e26d5fb61` is installed at
`C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-NVIDIA`. All upstream sizes and LFS
hashes pass. It works with the existing dependencies and launcher:

```powershell
.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1 `
  -ModelPath C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-NVIDIA `
  -SpeculativeNumSteps 0
```

This is an explicit **target-only, real-200K** option, not a new production
default. Five warmed uncached ordinary-sampling `6213+512` requests average
**64.064 generation tok/s**, **9469.619 prompt tok/s**, **0.656153 s TTFT**,
and **8.633376 s E2E**. An identically configured original RadixArk control
averages **63.996 generation tok/s**: no material speed difference. Exact
`199000+16` passes at **2720.158 prompt tok/s**, **73.157527 s TTFT**, and
**73.500594 s E2E**, with 2,660 MiB free after cache flush. Reasoning,
arithmetic, parsed tool use and continuation, non-thinking output, and the
language-only surface pass.

The mixed NVFP4/FP8 stock weights occupy **21.922 GB**, versus **18.766 GB**
for the selected attention-NVFP4 derivative. NVIDIA DSpark serving, published
accuracy suites, and independent-restart/OpenCode2/Codex promotion gates are
not qualified. A preliminary full-200K production control ran at 95.899 tok/s
under substantially higher desktop residency; it is preserved as contended
evidence, not used to replace the historical production record. Full paired
samples and provenance are in
[`../benchmark/windows/nvidia_qwen38_20260908.json`](../benchmark/windows/nvidia_qwen38_20260908.json)
and [`../BENCHMARK.md`](../BENCHMARK.md). No launcher or dependency changed.

## Qualified native-Windows DSpark-v2 production winner

The literal argument-free command
`.\scripts\windows\serve_qwen38_27b_nvfp4_5090.ps1` now selects the
attention-selective RadixArk target checkpoint, the immutable trained
`Qwen3.8-27B-DSpark-v2` draft, online-FP8 draft weights, gamma seven/eight-token
linear verification, Triton draft attention, FP8 target/draft KV, five FP32
Mamba cache slots, one admitted request, 4,096-token prefill chunks, exact
200,000 context and target/draft pools, and the existing hybrid-Marlin target
path. Qwen3 reasoning, Qwen3-Coder tools, and the language-only surface remain
enabled.

`SGLANG_DSPARK_STATIC_GRAPH_KV_COMMIT` is the final qualifying mechanism. A
static width-eight verify already reserves all eight physical draft-cache
slots, so the target CUDA-graph tail now projects and writes K/V for all eight
target-hidden rows. Accepted rows are immediately addressable; rejected rows
remain unreachable through the shorter committed sequence length and are
overwritten on slot reuse. Eager fallback still owns the commit whenever the
target verify does not replay its graph. The launcher enables this only for
DSpark, restores the caller's prior environment in `finally`, and exposes
`-EnableDSparkStaticGraphKvCommit:$false` as the matched control.

The paired trace reduced the target-to-draft gap from 1.672 to about 1.10 ms.
It moved the projection/cache-write device work into the target graph rather
than deleting it; full cycle wall fell from 18.248 to 17.98 ms. A reduced 32K
sampled window averaged **151.694 tok/s**. The first exact-200K window measured
**`[157.176,173.614,135.338,152.884,160.793]`**, mean **155.961 tok/s**. The
independent argument-free production restart measured
**`[151.139,165.951,151.000,191.357,153.054]`**, mean **162.500 tok/s**; every
individual default-process sample reached at least 150 tok/s. Each score used
the Python authority client, uncached exact `6213+512`, one 16-token warmup,
temperature 1.0, top-p 0.95, top-k 20, and presence penalty 1.5, with exact
counts, length finish, and preserved ordinary reasoning/content.

The argument-free server repeated exact `199000+16` at 3,118.215 prompt tok/s,
63.818572 seconds TTFT, and 64.236571 seconds E2E. It coherently derived `703`,
emitted exactly one parsed `multiply({"a":37,"b":19})` call, carried that
call through a tool-result continuation without a second call, and returned
thinking-disabled exact `READY` with zero reasoning. `/model_info` reports
image/audio false. Standalone OpenCode2 passed its auxiliary-plus-main
multi-chunk boundary. Codex CLI 0.152.0 twice completed the real Responses Code
Mode round trip; the retained clean run executed exactly one command, consumed
its complete nonempty output accurately, returned exact `CODEX TOOL READY`,
and preserved Tombstead byte-for-byte.

## Qualified local Codex audit prompt

`C:\Users\Daniel\.codex\qwen38.config.toml` selects the Qwen-specific
`C:\Users\Daniel\.codex\qwen38.md` while retaining
`model_reasoning_effort = "xhigh"`. The selected instruction SHA-256 is
`20B9DE8E98BA5D0B0106B5EAC4CC43131C3805F7D154574836563235C91810D8`.
Its audit path requires every changed file to be triaged, bounds individual
evidence results and live hypotheses, runs one early non-writing check, follows
only concrete leads, names coverage gaps, and stops when the live hypotheses
are proved or closed. Data ownership and destructive/process safety remain
unchanged.

On the exact literal Tombstead prompt `Look through this game for bugs.`, the
complete pre-budget xhigh baseline took **472.504487 seconds**, 38 commands,
2,672,390 cumulative input tokens, and 43,104 output tokens. Two independent
cache-flushed samples with the selected prompt took **232.5230228** and
**260.7988227 seconds**, mean **246.66092275 seconds**: a repeatable
**47.7971%** start-to-finish reduction. They used 26 and 14 commands, covered
all 11 changed files, passed `npm run check`, reached the same independently
supported no-current-bug verdict, disclosed generated-regression/browser/
balance gaps, preserved the exact Tombstead diff and untracked hashes, and did
not repeat the baseline's incorrect seven-survivor safe-integer claim. The
26-command first sample shows the budget is guidance rather than a hard tool
gate; do not tighten it merely for latency while this quality result holds.
Exact commands, usage, answer audits, and hashes are in the 08:37-08:51
experiment-log entries.

## Selected semantic upstream backports

Upstream `78d36f5f62` and `6afb5e1771` are adapted to the local control
surfaces. `kill_process_tree` now waits up to 60 seconds for ordinary teardown;
the runtime/GC route explicitly opts into fire-and-forget cleanup, and graceful
tokenizer shutdown reaps its descendants before exiting. The parent-discovery
race is covered by the existing `psutil.NoSuchProcess` guard.

The expensive idle tree-cache reconstruction is controlled by
`SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK`. Its unset value follows
`SGLANG_IS_IN_CI`, so production defaults off and CI defaults on; an explicit
environment value has priority. Pool accounting and request-pool leak checks
remain active on the idle path. Focused host coverage passes **8 tests and 2
subtests**, including the local hybrid-SSM/Mamba branch.

The semantic backports remain present in the qualified DSpark-v2 source line.
The earlier `03ba3d2e27` NEXTN/chunk-7680 launch is retained as a historical
control rather than the current production selection. Full commands, resolved
arguments, individual samples, process ancestry, behavior/client results, and
the one explicitly retained stochastic Codex narration retry are in the latest
experiment-log entries.

## Active Apple handoff

The current speed target is the immutable 4.951-bpw mixed-Q5 checkpoint at
revision `596b8067f7cf429007bb668874ffee7e917c8340`. The selected direct path
uses A100 aligned-word affine-Q5, stock-exact A111 affine-Q4, A114/A117 fused
Q4 gate/up/SwiGLU with lane-parallel epilogues, A113's redundant token-
evaluation removal, A128's aligned raw parameter bundles, A130 fixed-memory
long-prefill attention, A131 on-demand embedding, and now A137's exact fast
sigmoid. A137's two independent five-pair direct windows improve matched
precise-exp control **19.640965505 -> 19.667930618 tok/s**, a
**+0.026965113 / +0.137290%** gain. Every arm retains digest
`d0193f6d413b68c1`, last token 11406, and exact length; the strict final-source
smoke reaches **19.729345617 tok/s**. The current qualified mean remains
**0.332069382 tok/s / 1.660347%** below the requested floor.

The representative selected direct `6237 / 32 warm / 256 timed` window is
lower: five clean processes measure **19.049906088, 19.052370326,
19.013470487, 19.046250807, and 19.032662806 tok/s**, mean
**19.038932103 tok/s**. Every process has identical digest
`9ec00ec01f8781e1` and last token 20. This actual-work shape needs
**0.961067897 tok/s / 5.047909%** over current execution to reach 20. A
bounded A137 long-history trace maps **58,739 / 59,372** sampled PCs:
affine-Q5 owns **49.061847%**, fused raw-parameter Q4 SwiGLU **35.227%**,
ordinary Q4 **6.051674%**, and two-pass SDPA **3.863774%**. Streamed Q5
bytes/instructions remain the first optimization owner; attention alone
cannot close the measured gap at this history.

PERF-A139/A140 close lossless preassembled Q5 windows. A139's four-row form
regresses the dominant `K=17408, N=5120` micro
**0.435539514 -> 0.441254344 ms** (**1.312127%**) across six samples per arm.
A140 reduces live weight state to one row and regresses its adjacent exact
pair **0.418530771 -> 0.453039271 ms** (**8.245152%**). Both preserve digest
`d05378cc8066dc41`; neither reached the full-model gate. The next exact Q5
candidate must change extraction cost, not only rearrange the same continuous
bit windows.

PERF-A141--A144 then close the remaining byte-neutral representation family
on the same dominant shape. Low-nibble/high-plane extraction regresses
**5.800641%**; sixteen-row and four-row unchanged-block interleaves regress
**1.508948% / 0.515803%**; a combined weight plus raw-parameter stream
regresses **3.662772%**. Every arm is bit-exact. A143's first 2.28% apparent
win reverses across six samples per arm, so it receives no promotion credit.
Further Q5 work must reduce actual bytes or remove arithmetic rather than
repack the same stream.

PERF-A145 bounds that byte-reduction route with the real checkpoint. Across
all 240 Q5 tensors and **62,914,560** sampled codes, aggregate entropy is
**4.722514 Shannon / 4.748775 ideal-Huffman bits/code**; practical local
range, palette, and sparse-high-plane forms expose effectively no reduction.
Even a fictional zero-cost ideal decoder projects to only about **2.5%** end
to end at Q5's measured **49.061847%** owner, below the **5.047909%** gap.
Exact-kernel Q4 is **7.648307%** faster only for the down-projection shape;
the other Q5 families are nearly flat or slower, so down-only substitution
projects to about **1.6658%** and changes precision. Lossless compression and
selective Q4 are closed as standalone solutions. The active candidate is now
algorithmic: remove full-vocabulary dense proposal construction and retention
from the sampled standard-MTP path while preserving exact p/q verification.

PERF-A146/A147 close that proposal-only route. On the selected mixed target
with the official five-bit MTP head and block two, dense q reaches
**15.230953312 tok/s** at width **1.802816901**. Sampling probability-sorted
sparse support changes MLX's seeded inverse-CDF order, collapses width to
**1.196261682**, and reaches only **6.983233096 tok/s**. Reordering the
support by vocabulary ID restores the exact digest, last token, width, refill
count, target probabilities, and acceptance path, but measures
**15.222551699 tok/s**, a **0.055161%** regression. A traced steady cycle
spends approximately **103 ms** in two-token target verification; dense q
construction/retention is not the owner.

PERF-A149/A150 specialize that M=2 owner and are now retained behind opt-in
switches in signed `6b6d0d15ea`. Q5 loads each packed weight once while
accumulating both verifier rows; fused Q4 shares gate/up weights and raw
parameters across the same two rows. Production micros improve **19.8--29.3%**
for Q5 and **33.803365%** for fused Q4. The full sampled block-two path
improves **15.204721403 -> 19.795886878 tok/s**; a clean committed-source
rebuild reaches **19.785495602 tok/s**, 75 refills, width **1.706666667**,
digest `6bd687fb75c4f5a9`, and last token 1467. Focused two-row tests plus the
existing batch-one and exhaustive fused-Q4 tests pass. The durable 20 tok/s
floor remains open by **0.204113122 tok/s / 1.031089%** at the best full
window; exact 131K capacity and the real Responses/Codex `xhigh` gate remain
separate unresolved requirements.

A128's 680 MiB duplicate parameter stream remains opt-in. A130 plus A131
completes an independent exact `32768+16` server request at
**87.807667 prompt tok/s / 373.179259 s**, but final BF16 K/V cannot fit at
131K. A134 proves that a final approximately 4.25 GiB affine-Q8/G64 K/V
allocation fits; its current long-history composition reaches only
**3.515167775 tok/s** and is rejected for promotion. Exact served 131K,
steady sampled serving, and Responses/Codex `xhigh` qualification remain
pending behind a faster compressed-cache design and a direct result above
20 tok/s with margin.

The fresh-session candidate matrix is resolved through A111. A100 is promoted:
five aligned 16-bit loads reconstruct the same three bit windows and retain the
original sixteen sequential FP32 FMAs. A111 follows official MLX v0.32.2's
Q4 load/dot helper structure exactly while doubling the output cohort from two
to four SIMD groups. Representative parity is bit-exact, and two independent
full-model windows qualify the 0.670% increment. A095 packed-byte loads are
correct and neutral. A103 aligned overlapping loads, A104 vector
dots, A106 vector input reads, A108 parameter broadcast, alternate threadgroup
geometry, the older two-SIMD/two-result mixed-target Q4 custom execution, and
the published Q4 MTP head are closed by current measurements. The faster
hand-inlined Q4 precursor is also closed because its small BF16 differences
changed the seeded target trajectory. Dense BF16 recurrent b/a row fusion is
exact and aggregate-flat at **18.993221731 vs 18.993383731 tok/s** across ten
samples per arm, so its source remains outside `main`. A101 paired-lane word
sharing and A107 128-bit activation reads are exact and slower across the production-
shape matrix; both are closed. Exact 131K serving, sampled behavior, and Codex
`xhigh` qualification follow only after a direct candidate clears 20 with
margin.

A114 is qualified and retained. It fuses each batch-one affine-Q4 gate/up
pair and the SwiGLU chain into one 8-SIMD/four-paired-row Metal dispatch. A
temporary real-weight/real-hidden trace localized the original trajectory
failure to layer 62, element 36: gate and up were exact, but `metal::exp`
rounded sigmoid to BF16 `0x3a8c` rather than MLX's safe-math `0x3a8b`.
Volatile locals and explicit BF16 bit round trips did not change it. Replacing
only that operation with `metal::precise::exp` makes gate, up, sigmoid, SiLU,
and final output exact across all 64 traced layers and restores the canonical
full-model digest/last token. The corrected production-shape microbenchmark
moves **0.604255438 -> 0.564201438 ms** (**-6.628659%**). Forward and reversed
five-pair windows improve **19.236012324 -> 19.264036375** and
**19.221428286 -> 19.272779458 tok/s**; all 20 outputs are canonical. A new
Q4 fixture constructs the failing gate value `-6.84375`: the precise kernel
matches MLX in all 32 rows while the preserved fast-exp artifact fails all 32.
The selected result and direct gap are now **19.268407916 tok/s** and
**0.731592084 tok/s / 3.796848%**. Exact 131K serving and Codex `xhigh` stay
deferred until the direct lane clears 20 with margin.

A113 is also qualified and retained. `Engine::emit_scheduled()` previously
evaluated `pending_tok_` and immediately called `array::item<int32_t>()`, which
performs the same evaluation internally. Removing the explicit call preserves
the two-token pipeline and every fixed-work output. Forward and reversed
five-pair windows improve **19.270942164 -> 19.277656005** and
**19.285820872 -> 19.301343552 tok/s**. Aggregate matched control/candidate is
**19.278381518 / 19.289499778**, a **+0.057672%** win. Signed
`ad11696f2e` contains the one-line deletion. The selected gap is now
superseded by A117 below.

A117 is qualified and retained in signed `00d09138ce`. A selected-path Metal
trace attributes **42.141%** of sampled target shader PCs to the fused Q4
gate/up/SwiGLU kernel, making it the largest single shader. A117 keeps A114's
eight-SIMD geometry and exact BF16 boundaries, reduces all four rows as before,
then lets lanes 0--3 execute the four `metal::precise::exp`/SiLU/product chains
concurrently. Compile-time-named ternaries select each lane's already-reduced
register values; the dynamically indexed precursor is slower and closed by
PERF-FA134. General-shape and `-6.84375` boundary parity remain bit-exact.
Forward and reversed five-pair windows improve
**19.268343602 -> 19.450918961** and
**19.268607678 -> 19.455265774 tok/s**. Aggregate matched
control/candidate is **19.268475640 / 19.453092367**, a
**+0.184616727 / +0.958128%** win. The selected direct gap is now
**0.546907633 tok/s / 2.811417%**. One separate process-reload window at
11--14 tok/s is explicitly excluded after extreme page-in/swap churn; a
60-second idle interval restored the selected control to **19.203472228**
before the replacement reverse window.

A128 is qualified and retained as an opt-in in signed `f2fcce0c73`. The fused
Q4 kernel previously read gate scale, gate bias, up scale, and up bias as four
independent BF16 planes. A128 preserves the raw BF16 bits but interleaves four
gate row pairs followed by four up row pairs in one group-local stream. Each
iteration therefore issues two aligned `uint4` loads while keeping quantized
weights, FP32 dot order, reductions, the precise sigmoid, and every BF16
boundary unchanged. Ten order/reverse production-shape micro pairs improve
**0.5601321469 -> 0.5542401462 ms** (**1.051895%**) with every pair faster and
digest `8a9031349585365a`. Forward and cooled reversed complete-model windows
improve **19.439523730 -> 19.639348219** and
**19.454291289 -> 19.611393731 tok/s**. Aggregate matched throughput is
**19.446907510 -> 19.625370975**, a **+0.917696%** win; all 20 clean outputs
are canonical. Three second-position controls at **15.202523971**,
**17.620645469**, and **14.934890513 tok/s** coincided with a Spotlight worker
wave and repeated 18.5 GB reloads, are excluded, and were replaced after
enforced idle intervals. The final hardened source smoke reaches
**19.662460455 tok/s**. The extra decode stream duplicates
**713,031,680 bytes / 680 MiB**. PERF-A129's one-GiB MLX cache cap plus
1,024-token internal prefill converts the repeatable 8K Metal OOM into a
complete sampled request, but an exact-ID `32768+16` request still fails with
Metal insufficient memory after about 66 seconds. Startup advertises the real
131,072-token context/admission surface and language-only metadata; the stock
shape-growing BF16 KV plus dense-SDPA prefill path is not a capacity solution.

A137 is qualified and retained in signed `24d745ff38`. The shared fused-Q4
sigmoid helper now uses fast Metal exp and directly supplies precise BF16
sigmoid bits `0x3a8b` for input `0xc0db` (`-6.84375`), the sole mismatch found
by an exhaustive 65,536-pattern comparison. Corrected sigmoid and SiLU digests
match precise evaluation over the complete BF16 domain. The production fused
kernel also matches the separate QMV plus MLX SiLU path for all 65,280 finite
input patterns. Two independent direct windows improve
**19.640965505 -> 19.667930618 tok/s** (**+0.137290%**) with every output
canonical. Keeping precise exp as a conditional fallback regresses the fused
micro by **0.200153%** and is closed by PERF-FA149.

PERF-A130 is now the selected long-prefill mechanism behind
`SGLANG_MLX_NATIVE_FIXED_PREFILL_ATTENTION=1`. Its Q8/C64 native Metal
online-softmax path starts only above 8,192 active tokens, retaining canonical
stock SDPA for ordinary prompts. The thresholded direct smoke reaches
**19.744722973 tok/s**, digest `d0193f6d413b68c1`, and last token 11406. An
independent clean exact-ID `32768+16` server request completes at
**87.807667 prompt tok/s / 373.179259 s** where A131 plus stock SDPA fails
around 180 seconds. Signed `ee0bf40711` owns this win. Exact 131K still needs
persistent KV/snapshot residency reduction; steady sampled serving and Codex
`xhigh` also remain gates before default selection.

PERF-A131 retains on-demand quantized embedding behind
`SGLANG_MLX_NATIVE_QUANTIZED_EMBEDDING=1`. It replaces the eager
**2,542,796,800-byte** BF16 vocabulary table with **715,161,600 bytes** of
checkpoint-resident affine weight/scale/bias tensors and dequantizes only
selected rows, removing **1,827,635,200 bytes / 1.702 GiB**. Focused output is
bit-exact, matched sampled decode is **19.713595204 -> 19.690129490 tok/s**
(**-0.119033%**), and canonical full-model output is unchanged. The exact 32K
request now runs about 180 rather than 66 seconds before the same stock-path
Metal OOM; this is a qualified residency win, not a capacity pass. Signed
`e7643c904d` additionally admits the native sampler's valid MLX `uint32`
indices alongside prompt `int32` indices; both gathered forms are bit-exact
and floating indices remain rejected.

PERF-A133 proves that snapshot ownership is not the remaining capacity owner
by itself. Its metadata-only append-only snapshot passes exact logical
rollback, suffix overwrite, growth-boundary, and fixed-attention parity tests;
changing only that switch is neutral in one short direct pair at
**19.685169420 -> 19.672600916 tok/s**, with canonical output in both arms.
However, eagerly reserving all 131,072 BF16 slots makes the exact `32768+16`
server probe fail after about 16 seconds. Scheduler RSS is 18,225,056 KiB
(approximately 17.38 GiB) before the request; adding the final 8 GiB BF16 K/V
cache exceeds MLX's approximately 25 GiB recommended Metal working set before
safe OS/display headroom. That unchanged representation is closed by
PERF-FA145. A134's affine-Q8/G64 representation reduces final storage to
approximately 4.25 GiB and its complete 131,072-slot direct allocation fits,
but `8193 / 1 warm / 8 timed` reaches only **3.515167775 tok/s**. Fixed split
counts from one through 32 do not repair it. Preserve the capacity
infrastructure, but do not promote the current Q8 composition; exact 131K,
steady sampled serving, Responses/Codex `xhigh`, and decode above 20 tok/s
with margin remain open gates.

A118 and A119 are closed in their measured forms. Replacing A117's fused-Q4
packed-word reads with one explicit `packed_ushort4` transaction is exact but
regresses **0.557832737 -> 0.560992548 ms**. Row-concatenating each affine-Q5
linear-attention `qkv`/`z` pair reduces the isolated two-launch aggregate
**0.426645998 -> 0.420592346 ms**, but the copied-weight plus runtime-split
implementation reaches only **19.441971742 / 19.406059113 tok/s** around an
adjacent disabled control at **19.469521945**. Exact A117 is restored. The
split-free A120 and A121 follow-ups are also exact and slower. A120's direct
four-SIMD/two-output aggregate regresses **0.421745535 -> 0.427401865 ms**;
A121's exact-ratio five-plus-three/eight-SIMD topology regresses
**0.423397846 -> 0.426008156 ms**. The launch-sharing family is closed under
copied/split, direct four-SIMD, and 5:3 eight-SIMD representations. Exact A117
is restored. Further Q5 work must reduce weight-side bytes, instructions, or
dependency cost rather than submission count alone.

A122 also closes source-order alternation inside the selected fused-Q4 body.
Issuing each gate row immediately before its matching up row preserves every
arithmetic and BF16 boundary but regresses production-shape aggregate latency
**0.558719448 -> 0.569066813 ms** (**1.851979%**) and loses all ten
order/reverse pairs. Exact A117 is restored. Further fused-Q4 work must remove
instructions or bytes, not merely reorder the same eight row dots.

A123/A124 close Q5 result-row read-ahead. Prefetching all four rows regresses
the two largest K families by **0.431270% / 0.727663%**; prefetching only two
rows still regresses the decisive K=17,408 aggregate
**0.428486222 -> 0.429494958 ms** and wins only four of ten long pairs. Every
shape is exact and A100/A117 is restored. Further Q5 kernel work must remove
streamed bytes or unpack/arithmetic instructions rather than merely extend
the number of live row loads.

A139/A140 also close preassembled continuous-Q5 window layouts in four-row and
row-local forms. The balanced A139 production mean regresses
**0.435539514 -> 0.441254344 ms**; A140's first matched exact pair regresses
**0.418530771 -> 0.453039271 ms**. Preserve A100's selected aligned-word
stream. A low-nibble/high-bit-plane representation remains distinct because
it changes extraction instructions rather than only address layout.

The requested Q5 lane now serves through a provenance-pinned derived artifact.
Bartowski's immutable `Qwen3.8-27B-Q5_K_S.gguf` source is pinned at revision
`f0eec4a4bb4975114a030d048952d83c0a53c034`, occupies exactly
19,680,945,760 bytes, and verifies as SHA-256
`b52fbc242bde75a8e8f1dd2ec9ef9da4a1ce074d2513d3087a4f15003c11e569`.
Pinned llama.cpp build 10547 copied all source tensors and converted only the
833.59 MiB Q5_K `token_embd.weight` to F16. The distinct
`Qwen3.8-27B-Q5_K_S-TokenF16.gguf` artifact is exactly 21,349,656,160 bytes
with SHA-256
`c05a777870159b0779a441e2f58b543a0660af466d833d5b550a8aab9c17fcfb`.

A second provenance-preserving derivative changes only that same token
embedding to Q4_K. `Qwen3.8-27B-Q5_K_S-TokenQ4_K.gguf` is exactly
19,522,020,960 bytes with SHA-256
`8ed3117aff80d105a302da708221364579d483964c4c07d56eb6091a774b06ae`.
All other 865 tensors retain COPY from the immutable Q5_K_S source. Direct
Q4_0/Q4_K/Q5_K/Q6_K arithmetic and token-id embedding parity pass.

Actual-file native-MPS parity passes for Q4_0, Q5_K, and Q6_K. The derived
checkpoint loads as `Qwen3_5ForCausalLM`, occupies 21.37 GB at runtime, warms
the native Metal path, and exposes the language-only `qwen3.8-27b-q5` serving
surface. The first retained Q5 optimization adapts pinned llama.cpp's Q6_K
batch-one matrix-vector mapping so each SIMD group reuses its activation
fragment across two output rows. Matched QKV and vocabulary-head medians fall
**0.748917 -> 0.448125 ms** and **12.009166 -> 3.324625 ms**. Optimized,
odd-row fallback, and batch-eight actual-file parity all pass.

The second retained Q5 optimization doubles the Q5_K batch-one cohort to 32
rows for aligned projections with at least 5,120 outputs, the smallest measured
winning shape. Four lanes cooperate
on each row and each lane consumes two adjacent `float4` fragments. The
1,024-row attention K/V shapes retain the prior mapping. Reversed-order matched
serving moves the `128+32` five-run mean **7.1342 -> 7.1646 tok/s**
(**+0.426%**) and the higher-resolution `128+128` median
**7.456 -> 7.500 tok/s** (**+0.590%**). Direct candidate, tail, alignment,
fallback, and complete served-behavior gates pass;
`SGLANG_MPS_Q5_K_BATCH1_ROWS32=0` selects the matched control.

The first retained NEXTN prerequisite specializes exact-batch-four Q6_K. Each
eight-lane cohort dequantizes one row once, reuses it across all four verifier
activations, and four SIMD groups produce sixteen output rows. Matched final-
source medians move `output.weight` **29.166000 -> 6.121375 ms** and the
representative QKV projection **1.442333 -> 0.588500 ms**. In the real
synchronous NEXTN path, five sampled `128+128` requests improve from the
environment-disabled mean **3.3384** to **3.7028 tok/s** (**+10.915%**) even
though control mean accepted length is slightly higher (**3.024 vs 2.960**).
Direct batch-four, 17-row tail, batch-three fallback, and batch-eight
preservation checks pass; `SGLANG_MPS_Q6_K_BATCH4_ROWS16=0` selects the
matched generic route.

The complete same-GGUF three-step NEXTN configuration remains below the Q5
target-only lane. Explicit `--speculative-draft-model-quantization gguf` is
required because draft propagation precedes target GGUF inference; with it,
the trained draft occupies **1.00 GB** and leaves **8.50 GB** after the 1K
state/cache allocation. Its **3.7028 tok/s** mean trails the selected
target-only `128+128` median **7.500 tok/s** by **50.629%**, leaving
**16.2972 tok/s / 5.4013x** to the floor. PERF-FA115 closes this unchanged
full configuration while retaining the independent verifier-kernel gain.

Five ordinary sampled exact `128+32` requests reach
**6.993 / 7.214 / 7.206 / 7.218 / 7.192 generation tok/s**, mean **7.1646**
and warmed-four mean **7.2075**, versus the original generic-kernel baseline
of **5.746 tok/s**. Every request preserves 32 reasoning tokens and exact
length. The five-run prompt/TTFT/E2E means are **5.809 tok/s**,
**22.035143 s**, and **26.362669 s**. This leaves a
**12.8354 tok/s / 2.7915x** decode gap.

Both derivatives pass server startup, warmup, health, language-only metadata,
and an exact request with real `context_length=max_total_tokens=131072`. The
F16-embedding artifact occupies **21.37 GB** at runtime; its **8.00 GB** BF16
attention cache leaves no reported headroom, and one sampled `128+32`
diagnostic reaches only **0.102 generation tok/s / 329.689187 s E2E** under
heavy paging. The Q4_K-embedding artifact occupies **20.00 GB**, leaves
**0.99 GB** after the same cache, and improves the matched exact-capacity
result to **0.315 tok/s / 121.240667 s E2E**, a **3.088235x** residency win.
Its 1K-pool smoke reaches **7.086 tok/s**, returns arithmetic answer **703**,
and emits exactly one parsed `multiply({"a":37,"b":19})` call.

PERF-A089 supplies PyTorch 2.11.0's missing MPS E4M3FN value conversion in
the existing C++/Metal extension. SGLang's generic pool already stores float8
pages in `uint8`; the native path encodes FP32 cache writes and decodes both
contiguous and moved-dimension gathers while every other `_to_copy` case uses
PyTorch's original composite implementation. **400,006** reference encodes,
all **256** raw decodes, offsets, strides, empty tensors, and BF16 fallback
pass. A 1K FP8 server completes warmup and five sampled `128+32` requests at
**7.248 / 7.277 / 7.268 / 7.251 / 7.263 tok/s**, mean **7.2614**, with
arithmetic and parsed tools preserved.

The exact 131K FP8 pool now passes. It occupies **2.00 GB K + 2.00 GB V** and
leaves **6.99 GB** by server accounting, versus 8.00 GB and 0.99 GB for the
same artifact's BF16 pool. Five cache-flushed sampled requests average
**3.237 tok/s**, **10.276x** the BF16 result, with exact reasoning,
arithmetic, and tools. The fully allocated pool still trails the 1K FP8 mean
by **55.419%**. Another resident-byte reduction owns short-context recovery;
a fused native FP8 attention owner owns long-history decode without FP32 K/V
materialization. The sustained 20 tok/s admission window and Codex `xhigh`
work gate remain due.

A native affine-Q5 route now supersedes the GGUF lane as the throughput
candidate. The immutable text-only group-64 five-bit checkpoint at revision
`2568951b893b6427d0a8eb91cc7f4307154c2f05` contains 498 U32 packed and
1,349 BF16 tensors with no vision tensors. Its four model shards verify against
their SHA-256-addressed Hub blobs. The existing compiled engine loads it
directly and reaches **16.322505765 tok/s** on sampled direct
`128 / 32 warm / 128 timed`, **2.248x** the GGUF Q5 1K-FP8 mean. Exact served
131K, reasoning/tools, and Codex gates remain pending. The unchanged selected
DFlash controls regress to **11.506669050 tok/s** because five-bit M=8 target
verification falls through to generic MLX QMM. PERF-A092 adds exact
five-byte/eight-value affine-Q5 unpacking to the selected SG16/B32 K-split
kernel. Representative parity passes, M=8 verification falls to about
**228--229 ms**, and the same direct composition rises to
**13.721235888 tok/s** (**+19.246%**). Target-only remains faster at
**16.322505765 tok/s**; proposal quality and target-cycle bytes own the next
gap.

The pinned matching five-bit MTP head at revision
`1faa5a803c972c57cfc1beed606184e726ad3d85` establishes a high execution-cost
ceiling. Its default three-token deterministic path reaches
**17.919995235 tok/s**;
an opt-in eight-token block aligns with the Q5 M=8 kernel and reaches
**31.380317186 tok/s**, mean width **8**, with the target digest preserved.
Native sampling now uses the head's dense proposal probabilities through exact
p/q rejection. The best calibration screen reaches only **6.380162135 tok/s**
at mean width **2.285714286**, so this MTP head remains an experimental probe.
Target-only affine Q5 stays selected and batch-one target weight traversal owns
the active speed gap.

PERF-A094 now supplies an opt-in batch-one affine-Q5/G64 Metal QMV that decodes
each five-byte pack directly, reuses a 16-value BF16 activation fragment across
four output rows per SIMD group, and accumulates in FP32. Representative K/N
parity against MLX `quantized_matmul` passes at maximum absolute error
**0.03125 / 0.03125 / 0.0234375**. With the selected command-buffer controls,
one complete sampled `128 / 32 warm / 128 timed` screen reaches
**17.823163930 tok/s**, **+1.500658165 / +9.194%** over the original affine-Q5
target baseline and **2.176836070 tok/s** below the floor. The target trajectory
is preserved against the adjacent native-kernel screens. This remains an
opt-in candidate pending a repeated matched window, exact 131K serving, and
Codex qualification. A concurrent gate/up stream experiment caused a bounded
custom-Metal event stall and is closed; use one sequential MLX stream.

PERF-A095 replaces the selected kernel's two packed-four-byte plus two scalar
weight reads with three packed reads over the same ten bytes. Fresh-session
parity passes at maximum errors **0.03125 / 0.03125 / 0.0234375**. The A094
and A095 production-shape microbenchmarks converge, and mixed full-model
screens reach **19.010697650 / 19.032187257 tok/s** with the same digest.
This **0.113%** movement is neutral and A094 remains selected.

PERF-A096 adds a smaller aggregate-Q5 checkpoint candidate. The immutable
`maglun/Qwen3.8-27B-MLX-Mixed-4.95bpw` revision
`596b8067f7cf429007bb668874ffee7e917c8340` is now cached without its separate
vision shard. Its four language shards contain exactly **16,645,209,088 tensor
bytes**, **9.999%** fewer than the uniform-Q5 checkpoint, and all four match
the release SHA-256 manifest. The inventory is exactly the native engine's
1,655 required language tensors: 1,253 BF16 and 402 U32 tensors, with 162
affine-Q4 and 240 affine-Q5 matrices, every one group 64, zero missing or
extra language keys, and zero packed-shape/bit-width errors. The text aggregate
is **4.9510 bits per weight**. The committed dense-BF16 loader branch admits
all 96 recurrent b/a tensors. Selected A094 Q5 QMV reaches
**18.993383731 tok/s** across ten sampled direct controls; exact 131K capacity,
behavior, and Codex qualification remain open.

The reachable quantized-linear stream refines the aggregate artifact estimate.
Excluding the embedding table, uniform Q5 traverses **17,615,093,760 bytes**
per target token and the mixed checkpoint traverses **15,877,570,560 bytes**,
a **1,737,523,200-byte / 9.863831687%** reduction. Pure byte scaling of the
PERF-A094 result projects **19.773598394 tok/s**, only **0.226401606 tok/s**
below the floor; measured full-Q4/Q5 interpolation is more conservative, so
the fresh-session result remains the admission evidence.

PERF-A097 identifies a likely correctness defect in the native matched-MTP
seed. `Engine::select_token` retains the target's raw pre-final-norm residual,
while the target logits apply `final_norm_` separately; both native sampled
and greedy MTP drafting feed that raw value into `mtp_forward`. Upstream
Qwen3.5 returns the final-normalized target hidden state to the standard MTP
worker. The pinned MTPLX 2.9.0 Qwen3.8 contract independently selects
`base_hidden_variant=post_norm`, embedding-before-hidden concatenation,
post-norm recurrent hidden, and local/cache positions. Native concatenation
and recurrent normalization already match; the initial target seed does not.

The corresponding published INT4/G64 head is cached independently at exact
revision `123db8bcc7101455b00d9aad36c0e760c6e7de02`. Its
**238,934,249-byte** `mtp.safetensors` has SHA-256
`c58feddc584f37971c72af1f0da95e0099478487009936b3c26ddd88844fab10`,
contains the exact 31-tensor one-layer Qwen3.8 MTP shape, and preserves all
seven BF16 norm tensors bit-for-bit against the existing five-bit sidecar.
Its eight affine matrices are Q4/G64 and carry an `mtp.` namespace. Signed
commit `0da5c5a135` teaches the shared native MTP loader to accept that namespace
while preserving unprefixed sidecars. Signed commit `1b328149c7` adds the
opt-in `SGLANG_MLX_NATIVE_MTP_POST_NORM_SEED=1` correction at the common
sampled/greedy seed owner. Both compile under strict warnings. Published
depth-three exact-sampling acceptance is
**0.958762887 / 0.872852234 / 0.759450172**. Fresh-session mixed-target
block-three screens reach **9.122242179 tok/s**, post-norm seed
**9.507655940**, and Q4 QMV **11.341033334**. The unchanged composition is
closed for throughput. Committed MTP history and position origin remain
isolated research variables; the native engine currently clears the MTP
attention cache at each refill, while the published server path uses committed
history.

Offline Metal AIR inspection also queues PERF-A098 after PERF-A095. Apple
Metal 32023.883 scalarizes PERF-A095's packed byte vectors to ten aligned-one
byte loads. An alignment-safe `packed_ushort4` plus scalar `ushort` mapping
generates five aligned-two 16-bit loads. Every admitted Q5 row and lane begins
at an even address because K is a multiple of 512 and each lane advances ten
bytes. This remains source-only evidence until PERF-A095 has a fresh-session
runtime result.

PERF-A100 refines that queued 16-bit form into a continuous Q5 bitstream. A
standalone Metal 3.2 control reproduces PERF-A095's ten aligned-one `i8` loads
and ten byte extensions. The candidate emits five aligned-two `i16` loads and
five word extensions, forms two 32-bit windows plus a 16-bit tail, and needs
cross-window reconstruction only for fields 6 and 12. The alignment proof is
unchanged. Fresh-session parity and all production-shape digests match A094.
Two balanced full-model windows improve **19.004038228 -> 19.122257628** and
**19.021221323 -> 19.147557640 tok/s**. The aggregate
**+0.122277858 tok/s / +0.643140%** gain qualifies A100 as the selected Q5
kernel source.

PERF-A101 adds a paired-lane mapping above that continuous-bitstream form.
Each even SIMD lane reads both adjacent ten-byte packs through five aligned
32-bit loads; three native SIMD shuffles deliver the words needed by its odd
neighbor. AIR preserves that masked mapping. Dynamic weight-load operations
fall from 160 to 80 per SIMD group and output row while bytes remain 320. A
standalone C++ pack/unpack check passes **1,001,026** deterministic cases.
Fresh-session parity and complete digests pass, while gate/up, down, and
attention-output regress roughly **5--9%** in the matched matrix. A101 is
closed under the current compiler/GPU topology.

PERF-A103 supplies a branchless middle point. Each lane rounds its ten-byte
segment back to the pair's aligned 20-byte base and reads three aligned 32-bit
words. The two lanes overlap one word, so the SIMD group issues **96** dynamic
loads while the union of addresses remains the exact 320-byte Q5 block. Metal
AIR lowers reconstruction to two funnel shifts and selects, with no SIMD
shuffles. A standalone C++ pack/unpack check passes **1,016,386** cases; its
strict full dylib and standalone Metal parity executable are prebuilt from
signed `92a979bce7`. Runtime ranking belongs beside PERF-A100/A101 after the
fresh-session parity gate.

PERF-A104 layers a four-way FP32 dot formulation over PERF-A103. In an
explicit isolated control, Metal AIR retains sixteen scalar `air.fma.f32`
calls; the candidate emits four `air.dot.v4f32` calls and three FP32 adds.
This changes the within-lane reduction order, so compiler lowering alone
carries no correctness or speed standing. A strict full dylib and standalone
parity executable are ready from signed `6e181e68d2`; fresh-session numeric
parity, deterministic digest, sampled semantics, and matched timing determine
whether the vector intrinsic is useful.

PERF-A105 adds a single C++20 direct-QMV benchmark for all nine prebuilt Q5
arms. It constructs deterministic Q5/G64 weights and BF16 inputs, synchronizes
each launch, and reports latency, effective streamed bandwidth, and a complete
BF16 output digest. Strict final executables for
A094/A095/A100/A101/A103/A104/A106/A107/A108 are ready in `/private/tmp`; the
harness source hash is `6f7e336e...d6da3`.
After restart, parity precedes randomized and reversed gate/up, down,
attention-output, and value-projection matrices. Equal digests are required
through A103; A104 receives separate numeric and behavior gates because its
FP32 reduction grouping changes.

PERF-A106 isolates the activation-load side on the A094 weight path. Four
aligned 64-bit reads replace sixteen scalar BF16 reads per lane and K block;
AIR keeps four vector BF16-to-FP32 conversions while the sequential input sum
and every FP32 FMA retain their order. The candidate engine, strict full
dylib, parity executable, and PERF-A105 binary are prebuilt from signed
`83c2731a3d`. Fresh-session parity, exact control digest equality, and matched
matrix timing determine whether it combines with the selected weight-load arm.

PERF-A107 tests the wider aligned form independently. AIR replaces A106's four
64-bit reads with two aligned 128-bit `<2 x i64>` reads while retaining its
four vector conversions and the original arithmetic order. The candidate is
prebuilt from signed `414198a15c`; a sixteen-byte-aligned base is required and
the 32-byte lane/1,024-byte block offsets preserve it. Live parity gates that
alignment premise before exact digest and A094/A106/A107 matrix timing.

PERF-A108 isolates quantization-parameter sharing. Four adjacent lanes address
the same group-64 scale/bias pair; AIR keeps a leader-only load branch and one
vector SIMD shuffle, reducing dynamic BF16 parameter loads/conversions
fourfold from 32+32 to 8+8 per SIMD/output row. Arithmetic order after the
broadcast is unchanged. The strict candidate matrix is prebuilt from signed
`b1eeaf9f11`; parity and matched A094/A108 timing determine whether explicit
broadcast improves on the hardware's duplicate-address handling.

The fresh-boot comparison is prebuilt from source-identical signed revisions
in four detached clean worktrees. PERF-A094 control and PERF-A095/A100/A101
each have a strict full native dylib plus standalone Metal parity executable
in `/private/tmp`. Their dylib SHA-256 values are respectively
`b01d2712...fb242`, `3638e361...dd27`, `88c28fc0...ae85f`, and
`7ff5004e...a347`. Run all parity executables before the randomized matched
timing matrix; the current session has executed none of them.

PERF-A099 pins the remaining committed-history alignment before any source
implementation. Published MTPLX prefill pairs post-norm target hidden row
`H[i]` with token `T[i+1]`. Each decode cycle retains the provisional MTP
cache entry for the current pending token, rolls the later speculative entries
back to `cycle_base + 1`, then appends only accepted target tokens paired with
the committed target-hidden prefix. The sampled correction or full-acceptance
bonus stays pending and enters history in the following cycle. Decode-only
history and prompt history are therefore separable experiments. A full
131,072-token prompt history adds one 4-KV-head, 256-head-dimension BF16 MTP
cache, exactly **512 MiB** at capacity, plus a one-layer causal history pass.
Run the already-committed post-norm seed arm first; add decode-only history
only if acceptance remains insufficient, then qualify prompt history as a
separate TTFT/capacity candidate.

PERF-A102 prebuilds that decode-only history arm in a detached clean worktree
at signed `711214b27c`. The opt-in
`SGLANG_MLX_NATIVE_MTP_DECODE_HISTORY=1` path begins each sampled MTP cycle at
the target's absolute sequence offset, retains the cache entry produced for
the current pending token, rolls the provisional tail back to that boundary,
and appends accepted draft tokens with the verifier's corresponding post-norm
target-hidden prefix. An offset mismatch clears the decode-only history and
resynchronizes it, covering target-only reasoning-cap fallbacks. Prompt MTP
history, position-origin experiments, greedy MTP, DFlash, and DSpark remain
unchanged. Strict C++20/O3 compilation and `git diff --check` pass; the dylib
SHA-256 is `b4f3b222...a5d`. Fresh-session acceptance and throughput remain
the admission evidence.

A current Hugging Face inventory found no smaller distinct uniform-five-bit
MLX language layout to download. The additional published uniform candidates
use the same affine five-bit/group-64 text quantization and either carry the
BF16 vision tower or duplicate the existing text layout. The published OptiQ
candidate averages 5.50 BPW and is larger. Keep the already-pinned 18.51 GB
uniform text-only target for the literal Q5 control and the already-pinned
4.951-bpw mixed target for the bandwidth arm.

Two bounded fallbacks confirm that the current-session fault reaches the full
model path. The committed pre-PERF-A094 native library with every Q5 custom
dispatch disabled produced no output before a **120-second** cutoff. A pure
stock-MLX `mlx_lm.generate_step` load of PERF-A096 produced no output before a
**180-second** cutoff. Both complete process trees exited with status 124;
port 30000 and Qwen, benchmark, MLX, and compiler process searches are clear.
The system still reports **1,207,315 wired pages** and memory pressure remains
at **39% free capacity** after 30 seconds. End GPU submissions for this
session. Resume after a full machine restart, then verify ordinary MLX
arithmetic, PERF-A095
parity, the PERF-A094 matched control, and PERF-A096 in that order.

The unchanged Q5_K_M artifact is closed on this loader because mixed merged
weights contain Q8_0 shards unsupported by the native Metal merge path. The
unchanged Q5_K_S artifact loads its 20.00 GB weights and then reaches the
unsupported Q5_K embedding boundary. The derived artifact owns the active Q5
route. The affine-Q4 lane below remains optimization substrate and comparison
evidence.

The native M1 Max Qwen3.8 lane now clears the requested 20 tok/s served floor
in two independent real-131K five-sample windows when recurrent QKV
projections are restored from the immutable Q4 checkpoint. The windows average
**20.0222** and **20.0292 tok/s**, and every individual sample exceeds 20.
One pinned Codex 0.151.0 `xhigh` turn completed the requested shell tool
cleanly. A second completed the work after sampling a malformed extra
`write_stdin` call, so strict structured-output reliability remains active.

DFlash2 now runs end to end through the native C++ MLX engine in signed commit
`d57a6ac11c` (`feat(mps): add native DFlash2 runtime`). The exact
`incoai/Qwen3.8-27B-DFlash2` BF16 source remains immutable, and the distinct
175-tensor affine-W4/G64 draft artifact remains at
`/Users/dcazares/.cache/sglang/checkpoints/Qwen3.8-27B-DFlash2-MLX-AffineQ4`
with model SHA-256
`33bf2ddd0d46c27d6f383b6822ab897eb87261d34ca9c6eeb6a0f232026f723c`.
The existing `Engine::load_mtp` entry autodetects its tensor contract. Native
target-layer capture, five-layer block drafting, selector sampling, exact p/q
rejection, and accepted-prefix recurrent-state replay are implemented behind
the existing opt-in draft path.

Signed commit `6cf95442cc` also loads the official 81-tensor BF16 checkpoint
directly through the shared native linear owner. It passes dense bit parity
and complete direct decoding. BF16 is a compatibility lane: its matched
sample reaches **9.044043 tok/s**, 61 refills, and mean width **2.114754**,
versus adjacent affine-W4 **30.992508 tok/s**, 19 refills, and width
**6.684211**. The derived affine artifact remains the performance selection.

The native prefill owner divides DFlash target/capture work into internal
2,048-token units while preserving one external SGLang prefill call. This
completed 4,096- and 6,237-token direct prompts and the real 6.2K-token Codex
shape that previously exhausted Metal residency. Two real 131K-configured
Codex 0.151.0 `xhigh` turns have each issued exactly one requested shell tool
and returned the exact final marker.

Generation remains the active gap. Signed commit `3bae8a5e67`
(`perf(mps): widen DFlash verifier output tiles`) expands the opt-in M=8
affine-W4 kernel to 32 output columns while retaining sixteen independent K
partitions. The dequantized-weight staging and FP32-partial phases have
disjoint lifetimes and share one 32 KiB threadgroup allocation. Five
process-isolated no-trace `128 / 32 warm / 128 timed` samples measure
**31.347246 / 31.354397 / 31.268351 / 31.330814 / 31.335863 tok/s**, mean
**31.327334**, versus the preceding SG16/B16 mean **17.651701 tok/s**. Every
sample reproduces mean emitted width **6.684211**, digest
`46bd4bb035b72c2b`, and last token 20. Steady short-harness cycles now spend
about **26.0--26.4 ms** drafting and **185.4--186.7 ms** verifying. The shape
owner requires K divisible by 512 and N divisible by 32, covering every
reachable projection and failing closed elsewhere.

SG16/B32 also passes the real server and Codex behavior gates. The exact
sampled `6237+128` control reaches **15.328 generation tok/s**, **109.106
prompt tok/s**, **57.164599 s TTFT**, and exact `6365` total tokens. Live
6.2K-history verification falls from SG16/B16's roughly **236--239 ms** to
**211--214 ms**. Codex thread `01a05c69-215f-7fb0-a7f8-1425c9b2ae5a`
issues one exact command, observes exact stdout, returns exact
`QWEN38_DFLASH2_READY`, and exits zero with 264 reasoning tokens. The fixed
cycle win is production-reachable; this prompt's sampled proposal acceptance
leaves **4.672 tok/s** to the sustained admission requirement. Proposal
precision/selection and further verifier work are the next active branch.

The learned selector now has a checked opt-in temperature at its native score
owner. Identity remains the default and preserves the prior operation graph.
Temperature **1.15** raises five consecutive exact real-131K-pool samples from
the adjacent identity mean **15.4424** to **15.8866 tok/s**, a **2.876%**
gain; mean emitted verifier width rises **3.878788 -> 4.031250**. Every sample
completes exact 6,365 tokens with a stable coherent digest within its setting,
and exact rescaled q continues into rejection sampling. Temperature 0.95,
which appeared strongest on the repeated-token direct screen, reaches only
**13.739 tok/s** on the natural request and is closed. The selected opt-in
establishes the proposal distribution used by the selected verifier budget.

The common exact verifier now also supports a checked DFlash-only current-block
q budget. Mean selected q over proposal positions one through six has Pearson
**0.801577** with accepted length on the natural trace, while accepted-length
lag-one autocorrelation is only **0.324698**. With selector temperature 1.15,
the opt-in mean-q6 threshold **0.62** chooses target M=2 below the boundary and
M=8 otherwise. Five exact real-131K-pool `6237+128` requests reach **16.167 /
16.181 / 16.178 / 16.179 / 16.183 tok/s**, mean **16.1776**, versus adjacent
current-source full-M=8 **15.883 / 15.908 / 15.900 / 15.899 / 15.897**, mean
**15.8974**. The gain is **0.2802 tok/s / 1.763%**. Each candidate request
uses 23 M=2 cycles averaging **133.648 ms** and 19 M=8 cycles averaging
**248.447 ms**, completes exact 6,365 tokens, and reproduces coherent SHA-256
`bdf9428e...`. Thresholds 0.55 and 0.65 are closed at **15.191 / 16.056
tok/s**. Full M=8 remains the default; threshold 0.62 is retained opt-in with
**3.8224 tok/s** to the requested floor. Target-cycle arithmetic and proposal
quality are the next active branches.

Native M=8 gate/up fusion is also closed under the current kernel geometry. A
single workgroup that computes both products and exact BF16 SwiGLU serializes
the output grids and adds roughly **3.3 ms** to verification. A launch-only
two-plane grid preserves both products bit-exactly and changes five-sample
direct mean only **26.943319961 -> 26.964966176 tok/s** (**+0.08034%**), with
two adjacent pairs flat or slower. Separate SG16/B32 products remain selected;
a future reopening needs shared arithmetic or a fused down-projection boundary.

The official worker's greedy LM-head proposal rule is closed for sampled
production. It reached a misleading five-sample direct mean of **37.518731
tok/s** and width **7.9375** on the repeated-token harness, then only **9.512
tok/s** on the exact natural `6237+128` request, **37.944%** below the learned
selector. The temporary switch was removed. Proposal-policy candidates now
require the real 6.2K request as their admission screen.

DSpark v2 also runs end to end through signed commit `21cd561dfc`. The native
lane accepts the exact official 62-tensor BF16 checkpoint and the distinct
136-tensor affine-W4 derivative, executes all five full-attention YaRN draft
layers plus sequential rank-256 Markov correction, and shares the exact target
verifier/accepted-state commit owner with DFlash2. The affine artifact reaches
**10.050625 tok/s** in the first direct sampled screen and **11.242 tok/s** on
the exact real-131K-pool sampled `6237+128` request. Full reasoning, exact token
counts, health, and the language-only model surface pass.

The native DSpark lane now also evaluates the checkpoint's trained confidence
head under the existing trace flag. It combines each final draft hidden row
with its previous-token Markov embedding and emits the exact seven FP32-sigmoid
survival probabilities while preserving proposal width and the direct/served
baseline digests. The natural prompt spans roughly **0.1895--0.9998** and
confirms the signal is probabilistic. Bounded-width verifier profiling and a
cost-based cumulative-survival budget are the active follow-up.

The common exact verifier now accepts checked prefixes of one through seven
draft tokens behind a DSpark-only opt-in width probe; DFlash and default
DSpark remain at seven. The complete fixed-width direct sweep reaches
**11.920231 / 8.861866 / 8.353200 / 7.544333 / 4.104266 / 4.139508 /
10.025000 tok/s**. Target M=6/M=7 fall onto a slow generic affine-QMM tier,
while M=8 uses the selected SG16/B32 kernel. M=2 and M=8 are the useful
measured tiers for the next confidence scheduler; every fixed short-width
policy is closed below the required floor.

The native trained-confidence scheduler now chooses between those M=2 and M=8
tiers from cumulative expected survival and a measured complete-cycle cost
ratio. At the selected opt-in ratio **1.75**, five consecutive real-131K-pool
sampled `6237+128` requests reach **13.595 / 13.609 / 13.603 / 13.607 /
13.615 tok/s**, mean **13.6058**, versus the adjacent fixed-M=8 control at
**11.313 tok/s**. Every sample completed exact 6,365 tokens with one shared
reasoning/output digest. Fixed M=8 remains the DSpark default; the adaptive
lane is **1.7222 tok/s** behind selected DFlash2 and **6.3942 tok/s** behind
the requested floor. A bounded target-only bypass for predicted low-value
draft cycles is the next cost candidate.

An additional opt-in cooldown now skips sixteen DSpark probes after the
confidence budget selects M=2, advancing target and draft context through
exact target-only refills before forcing a fresh probe. The first five real
samples reach **17.028 / 17.080 / 17.036 / 17.102 / 14.952 tok/s**, mean
**16.6396**; the fifth is retained as transiently contended, and the recovery
sample reaches **17.011**. Five normal samples within the six-request window
average **17.0514 tok/s**. Every request completes exact 6,365 tokens with one
shared reasoning/output digest. Cooldown sixteen is the strongest measured
speculative serving lane and remains opt-in; it retains a **3.3604 tok/s** gap
to the floor on the all-sample mean.

The cheap target-state predictor branch is now closed as a scheduler. Its
score-0.525 direct screens reached **33.222 / 32.788 tok/s**, while the exact
natural request reached only **13.652 tok/s**. On a no-policy real trace, M=2
and M=8 score ranges overlap and score versus accepted width has Pearson
**0.141**. The target-state score remains available under the existing trace
flag as diagnostic telemetry; its parser and scheduling branch were removed.
Longer fixed cooldowns 64/128 reach only **18.402 / 19.664 tok/s** directly.

DFlash2 also cannot reuse the faster mixed-precision target unchanged:
QKV-only and complete recurrent restoration collapse direct accepted width to
**2.653 / 1.984** and throughput to **10.130 / 7.841 tok/s**. A two-phase
64-column M8 verifier preserves the selected digest and width yet falls to
**8.961 tok/s**; it was removed. The full-Q4 target and SG16/B32 verifier
remain the compatible selections. Further progress returns to shared target
decode cost while preserving the qualified target-only lane above 20.

The full-Q4 target-only lane now has a memory-safe representative prefill.
Checked opt-in `SGLANG_MLX_NATIVE_TARGET_ONLY_PREFILL_CHUNK_SIZE=2048`
evaluates and releases 2,048-token target units inside the native owner. It
completes direct 6,237-token prefill and the exact real-131K-pool served
`6237+128` request that exhausted Metal residency on the one-shot path. The
served sample reaches **19.300 generation tok/s**, **109.988 prompt tok/s**,
**56.706198 s TTFT**, and **63.286374 s E2E**, with exact token count and
coherent reasoning. Direct long-history decode reaches **19.586705 tok/s**;
single-chunk sampled decode retains its exact digest above 20. DFlash2 and
DSpark retain their existing capture chunks, ordinary MTP is unchanged, and
the default target-only path remains one-shot. The target-only served gap is
now **0.700 tok/s**.

A dedicated affine-W4 batch-one Metal QMV is closed at its scalar-output
geometry. Standalone BF16 parity passed, while the one-SIMD-per-output form
reached **10.456143330 tok/s** and an eight-lane quant-parameter broadcast
reached **7.773375044 tok/s**, against stock MLX **20.339874670 tok/s** on the
same short full-Q4 target-only shape. Every experimental source and test
change was removed. Future batch-one affine work requires matrix tiling,
dependency-level evidence, or fusion with a downstream consumer.

Metal GPU Counters now attribute the representative long-history target-only
decode. Of 48,343 sampled shader PCs, 47,676 map to the target process:
**88.470%** land in MLX's affine-W4 `qmv_fast`, **3.904%** in the two SDPA
passes, **2.358%** in the recurrent state update, and **1.405%** in the fused
full-attention q/k norm plus RoPE kernel. The trace-instrumented run reaches
**19.144319346 tok/s** with its stable 128-token completion. Installed MLX is
0.32.2 at official tag `1f8e74e3`; official HEAD `117188cd` contains no later
QMV change. A stock-compatible multi-output SIMD tile or dependency-level QMV
change owns the next measured branch.

Two DSpark proposal-fidelity candidates are closed. Independently applying
target top-k/top-p filtering to each draft row fell to **6.997461 tok/s** and
width **1.6** direct. Retaining only Markov W2 in BF16 fell to **7.044992
tok/s** and width **1.65** direct; its representative served request reached
**11.294 tok/s**, a **0.462551%** different-trajectory movement, while adding
**87.148 MiB**. Both experimental source changes were removed. The selected
affine checkpoint, DFlash2 path, and production runtime remain intact.

## Native backend roadmap handoff

Roadmap milestone 1, the native tensor-view ABI, is complete in the current
worktree. The dormant C++/CUDA slice fixes ABI 1.0 as 184-byte metadata and
192-byte const/mutable views with rank at most eight, closed dtype/device
vocabularies, checked bit-span validation, conservative mutable non-overlap,
typed witnesses, and allocation-free structured diagnostics. It now has an
independent framework-free CMake/Ninja AOT target family and linked capability
probe. It has no framework adapter, Python binding, server route, or launcher
change. Its standalone CUDA consumers are described below.

The final contract audit corrected `ValidationOutcome::match` so its error
branch passes the copied `TensorValidationError` value category promised by
its constraints and `noexcept` expression. An rvalue-only error-handler
regression raises the standalone suite to 35 cases. Default and `/Zp1` MSVC
warning-as-error builds, VS 2022 AddressSanitizer, MSVC static analysis, and
the NVCC `sm_120` compile-only probe pass. The dependency and raw-type
confinement scans, Mintlify validation, and worktree checks pass. Repository
baselines remain independently red: `base-a-test-cpu` cannot enter execution
because the unchanged `test_cuda_graph_composite.py` lacks CI registration,
and Mintlify reports 392 pre-existing broken links in 103 files with no
finding for the ABI page.

At the user's explicit request, the earlier qualified production server was
stopped leaf-first through exact PIDs `15140`, `15824`, `56364`, `20092`,
`49872`, `49852`, and `39816`. A later Codex setup task launched the supported
five-slot client lane described below; the argument-free launcher remains the
production benchmark source of truth.

The remaining roadmap item 1 substrate is now implemented beside the ABI.
`CudaStream` and copyable `CudaExecutionContext` establish explicit
device-affine nonblocking stream ownership. `GraphMemoryArena` allocates one
fixed CUDA address range, issues aligned slices until sealing, and then
creates retained leases. Owner-backed binding overwrites device provenance,
capacity, and base address before validation and produces only typed
`GraphStableTensorView` values. `CudaGraphExecutable` retains both its stream
context and arena lease, records completion after launch, and refuses storage
or stream cleanup while an asynchronous dependency remains.

Strict host tests, MSVC static analysis, and a real `sm_120` RTX 5090 CUDA
suite pass. The CUDA test captures an increment kernel, destroys the source
graph, drops every external slice/view/lease, proves arena and stream cleanup
remain busy, replays the executable three times at the same pointer, and
observes every output as `3.0`. The two-device mismatch branch is present and
fails closed when two GPUs exist; this single-GPU host skips that branch.
This substrate now backs two standalone operators and a captured composite,
but remains dormant with no SRT adapter, launcher route, or production CUDA
graph.

Roadmap item 2 now has its first bounded kernel port:
`launch_linear_rejection_sampling`. The framework-free C++/CUDA operator
implements the qualified batch-one linear p/q rejection rule for 2 through 64
slots. It accepts only exact-size, contiguous, non-aliasing
`GraphStableTensorView` buffers on the execution context's device. The
contract uses `int64` proposal tokens/indices, FP32 uniforms and p/q rows,
`int32` output tokens/accept indices/`num_correct_drafts`, and a `uint32`
device-status slot.

Device code preflights every proposal token and output index, including index
uniqueness, before changing any token/count/index output. Valid input preserves
strict `coin * q < p` acceptance, NaN-q residual fallback, positive `p - q`
sampling after rejection, pure target sampling after full draft correctness,
strict CDF crossing, and the existing last-token zero-residual fallback.

The strict host contract suite passes 2 cases. The live CUDA 13.3 `sm_120`
suite passes 5 cases: malformed layout/content, hand-oracle parity, 256
deterministic randomized comparisons, 2/64-slot boundaries, and captured
stable-address replay at the checkpoint's authoritative 248,320-token
vocabulary with changed inputs. CUDA memcheck reports zero errors. MSVC static
analysis, the existing ABI suites (35/35 default and `/Zp1`), resource host
suite (2/2), resource CUDA suite (5/5 with the two-GPU body skipped), the ABI
CUDA compile probe, dependency/raw-view/integration scans, and Mintlify
validation pass.

Roadmap item 2 now also has a graph-safe verify RNG producer. Request-seeded
mode reproduces the established four-block MurmurHash32 mapping, FP64 division,
FP32 conversion, and half-open clamp. Stateful mode freezes a 32-byte
`SGLRNGV1` record and Philox4x32-10 mapping over `(counter, subsequence, seed)`;
each replay reserves exactly `ceil((num_slots + 1) / 4)` counter blocks. Invalid
descriptors and counter exhaustion preserve state and coin outputs while
publishing namespaced device status.

The sampler now exposes a composition-only ready-status launch. A nonzero
upstream status returns before proposal validation or output writes, preserving
the originating device failure. The production-shape native graph captures
result initialization, stateful RNG, and gated rejection at batch one, three
verify slots, and vocabulary 248,320. Three successive replays use stable
addresses and disjoint counter blocks; reset reproduces the complete first
result; an overflow replay preserves coins, publishes the RNG status, and
leaves initialized sampler output unpublished.

`native/CMakeLists.txt` builds cumulative tensor, CUDA-resource, and kernel
static libraries, an aggregate native target, a linked capability probe, and
the existing/new host and CUDA tests without framework discovery or downloads.
The clean MSVC 19.51/CUDA 13.3.33/Ninja build passes six host-labelled tests and
three serialized CUDA tests. Three separate Compute Sanitizer memcheck targets
report zero errors; MSVC static analysis passes for the new host source and
probe. The linked probe contains SM120 fatbins and imports only Windows/MSVC
runtime libraries. It reports tensor ABI 1.0, RNG descriptor
`0x3156474e524c4753`, CUDA headers 13.3, and runtime-reported 13.4.

No SRT, Python, kernel registry, launcher, endpoint, or production graph uses
the native capsule. The later live Codex server retains that boundary.
Full behavior, capacity, production-relaunch, and OpenCode2 gates remain
mandatory before any adapter is promoted. Four ignored root object files
created before this task at 16:10-16:14 remain Daniel's and untouched; every
artifact created for this milestone was removed with its isolated build
directories.

The later native continuation now covers the production batch-one, gamma-seven,
vocabulary-248,320 DSpark proposal; BF16/FP16 corrected logits and FP32 log
normalizers; trained rank-256 Markov sampling with explicit Philox reservation;
corrected-logit rejection; AOT ReplaySSM fold plus convolution rollback/scatter;
and an eight-stage native cycle controller with one 24-byte D2H result. Real
graph factories exist for proposal, verify RNG, rejection, ReplaySSM, and
compact publication. Target verify, draft extend, and KV write remain typed
placeholders, not model implementations.

A controlled fixture against the immutable trained DSpark-v2 checkpoint now
matches the Python and Triton oracle exactly for all seven dependent sampled
tokens and the final Philox counter. Across all `7 * 248320` BF16 corrected
logits, the scalar native rank reduction differs from PyTorch's GEMM reduction
by at most `0.0625`; the maximum log-normalizer difference is
`0.00276756287`. Full isolated CTest remains 16/16, and both the direct trained
fixture plus all seven ordinary native Compute Sanitizer targets report zero
errors. Exact artifacts, hashes, commands, and the numerical rationale are in
the 2026-09-01 05:48 PDT experiment-log entry. This establishes controlled
trained proposal parity only; it is not full-model or serving parity.

The next native ownership boundary is the real target/draft graph seam: replace
the target-verify, draft-extend, and KV-write placeholders with model-owned AOT
graphs and prove their state/KV semantics in the unified controller. Scheduler,
request/pool, model-descriptor, and serving work remains explicitly later.

## Native benchmark client handoff

The benchmark-script review covers all nine `bench*` Python files beneath
`scripts/`. The production contract centers on
`bench_openai_stream.py`, `bench_spec_acceptance.py`, and the retained
`bench_target_verify_width.py` profiler. The first two now have portable,
framework-free, CPU-only C++23 candidates under `benchmark/native`. Their
shared layer owns strict CLI parsing, JSON and UTF-8, SHA-256, plain local
HTTP/1.1, SSE framing, server-owned prompt calibration, cache control,
streaming telemetry, exact result validation, and speculative-counter
validation. CUDA stays with the serving process.

The native stream path preserves the exact prompt/filler bytes, SGLang and
llama tokenization protocols, request-field omission rules, reasoning-before-
content ordering, Unicode scalar counts, complete/per-channel digests,
warmup/flush sequence, and historical throughput formulas. Its event clock is
sampled after JSON decoding and fragment accounting, matching the Python
client boundary. The native acceptance path closes the existing script's
exactness gaps by requiring exact calibration before cache activity, exact
warmup and measurement counts, length finish, complete speculative metrics,
and internally consistent counters and histograms.

The source enforces C++23. GCC C++23 warning-as-error builds pass the JSON/SHA,
argument, HTTP/SSE loopback, and full orchestration suites; their results are
`PASS`, **8/8**, **10/10**, and **11/11**. MSVC `/std:c++latest /W4 /WX`
also links and passes the full **11/11** suite, establishing the current
toolchain's C++23 mode. MSVC and Clang static analysis pass for the transport
and benchmark core; the analyzer-led 64 KiB receive-buffer finding moved that
storage to the heap. The transport includes Windows `FD_CONNECT` event
handling, POSIX `FD_SETSIZE` guards, bounded framing, and macOS `SIGPIPE`
coverage in the early-close test path.

The Windows live gate passes. Adjacent Python/native/Python windows preserved
the exact 38-field stream schema, request metadata, cache sequence, usage, and
length finish for greedy and sampled `6213+512` plus exact `199000+16`.
Deterministic windows matched every combined/per-channel hash,
fragment/character count, and delta count. The sampled acceptance triplet
preserved the 11-field schema and exact `6213/512`; every externally recomputed
rate, accepted-length, cycle-sum, and weighted-histogram invariant passed. The
native results stayed within or favorably adjacent to their Python timing and
acceptance windows.

| Windows client gate | Python before | Native | Python after |
|---|---:|---:|---:|
| Greedy `6213+512` decode tok/s | 138.992 | **139.809** | 139.926 |
| Greedy E2E s | 4.210606 | **4.175476** | 4.155960 |
| Sampled `6213+512` decode tok/s | 118.510 | **131.753** | 127.890 |
| Sampled acceptance length | 1.848375 | **1.896296** | 1.917603 |
| Exact `199000+16` prompt tok/s | 3145.208 | **3146.939** | 3092.176 |
| Exact `199000+16` TTFT s | 63.270853 | **63.236045** | 64.355971 |

Apple Clang/M1 Max live parity remains pending, so the qualified Python
commands remain the cross-platform scoreboard authority. Direct compiler
commands, complete paired invocations, and the full parity gate are in
`benchmark-contract.md`.

Daniel stopped the earlier four-slot server before the first integrated
GCC/MSVC gates. A separate five-slot Code Mode qualification later acquired
port 30000; this task left that process tree untouched, waited for its owner to
clean up, then completed the distinct four-slot native-client gate after
Daniel's explicit approval.

The remaining review findings define later work:

- `bench_target_verify_width.py` shares the acceptance client's historical
  below-target/result-validation gaps and gains the native request layer after
  gzip trace ownership is selected;
- `bench_native_fused_sigmoid_mul.py` and
  `bench_native_chain_metadata.py` are the first CUDA microbenchmark ports;
  the latter's Python comparison currently overwrites one shared tree mask and
  omits mask parity;
- sparse sampling should split into the retained native sparse top-p and
  framework-free rejection operators;
- projection quantization needs explicit FlashInfer/CUTLASS AOT tactics and a
  manifest-backed trace baseline;
- `scripts/playground/bench_speculative.py` currently carries stale serving
  namespace fields and a deprecated import, so its default text sweep requires
  an upstream product decision before a native orchestration rewrite;
- `scripts/bench_batched_qmv.py` remains an Apple MLX/Metal affine-Q4 tool;
  a Windows CUDA counterpart would measure a distinct NVFP4/Marlin workload.

Immediately before the live-launch approval boundary, a concurrent user-owned
change set appeared in grammar initialization, Qwen3 Coder schema conversion,
sampling-parameter limits, their focused tests, CI checkout credentials, and
the `compressed-tensors` dependency declaration. The benchmark stream and
acceptance requests omit grammars, tools, stop strings, and stop regexes, and
the installed dependency is already the newly pinned 0.18.0, so their paired
client paths remain stable. Adversarial follow-up repaired the concurrent
schema helper so direct and combinator-defined properties merge with direct
declarations taking precedence; its mixed-schema regression passes. The
four-slot launch pinned `d802d75bcc` plus those visible edits; their owner
committed the same audited branch as `efac2c9a21` during the resident window.
Every paired client request used the unchanged stream/acceptance paths in one
stable server process.

## Qualified production configuration

The accepted configuration is native-Windows SGLang serving
`C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4` on the RTX
5090. It provides a real 200,000-token target/draft pool, preserved reasoning,
parsed tools, and a language-only model surface.

| Area | Selected value |
|---|---|
| Endpoint | `http://127.0.0.1:30000/v1`, model `qwen3.8-27b` |
| Capacity | Context `200000`; total-token pool `200000`; one running request |
| Model surface | `--language-model-only`; Qwen3 reasoning parser; Qwen3 Coder tool parser |
| Target attention | FlashInfer prefill; TRT-LLM MHA/XQA decode and target verification |
| Draft attention | TRT-LLM MHA/XQA; captured draft decode and `DRAFT_EXTEND_V2` graphs |
| Linear attention | Triton GDN with ReplaySSM speculative-state handling |
| Speculation | NEXTN linear rejection sampling; 2 steps; 3 draft tokens; EAGLE top-k 1 |
| Proposal distribution | Draft top-k one with native CUDA direct one-hot q inside the single multi-step CUDA graph |
| KV | Checkpoint-selected target KV; FP8 E4M3 draft KV; page size 64 |
| Sampling | FlashInfer, including native-Windows CUDA renormalization on the speculative path |
| Prefill | 7680-token chunks |
| Mamba | 4 slots; `extra_buffer_lazy`; FP32 state |
| GEMM tuning | FlashInfer CUTLASS FP4 prefill plus in-place Marlin gate/up decode; autotune and large-EXTEND tuning enabled; FP8 GEMM autotune skipped |
| Compile/graphs | Torch compile mode `default`; batch-one full decode graphs |
| Scheduling/streaming | Scheduler receive interval 4; stream interval 4; incremental output |
| Workspace | 128 MiB FlashInfer workspace, the measured functional floor |
| Draft quantization | Checkpoint-native BF16 MTP; explicit online FP8/MXFP8/NVFP4 inactive |
| Experimental controls | Adaptive depth, SWOR/tree topology, path/overlap oracles, and fixed-acceptance simulation inactive |

These values match the defaults in
[`../scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1`](../scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1)
at reconciliation time. The launcher and freshly resolved arguments remain the
executable source of truth.

## Codex client lane

Codex CLI 0.151.0 uses `C:\Users\Daniel\.codex\qwen38.config.toml`. Start
its server with the production launcher plus `-MaxMambaCacheSize 5`. The fifth
FP32 state slot is a client-specific transient reserve; the benchmarked
production default remains four. A real `hauberk` Codex prompt crossed the
7,680-token prefill boundary and exhausted the four-slot pool while the
unfinished first chunk was being donated to the radix cache. Five slots passed
the same multi-chunk boundary, sequential retained-prefix pressure, exact
`199000+16`, a post-capacity Codex request, sampled reasoning, parsed tools,
and standalone OpenCode2 while preserving one-request admission.

Codex Code Mode custom tools are now qualified on the non-Harmony Responses
path. SGLang preserves the custom-tool grammar contract, presents Qwen with a
one-string synthetic function, accepts its observed `functions.exec` alias or
direct-shell `cmd` prior, and restores the exact `custom_tool_call` input and
stream events Codex expects. The following turn replays
`custom_tool_call_output` through the Qwen chat template, including Codex's
lazy array-form output representation. A real Tombstead gate crossed the
7,680-token boundary, executed exactly one `git status --short`, consumed its
complete output, returned exact `CODEX TOOL READY`, and left both repository
status and server health intact. The validation server was cache-flushed and
stopped; port 30000 is free and the GPU is at ordinary display residency.

Five selected upstream hardening backports are qualified on top of that lane:
`compressed-tensors` is pinned to the already-installed 0.18.0; executable PR
checkouts keep the GitHub token out of `.git/config`; grammar initialization
rejects raw or decoded NULs before XGrammar; stop inputs are bounded to 32
strings, 32 regexes, and 256 UTF-8 bytes per regex; and Qwen3 Coder resolves
typed parameters across direct plus top-level `anyOf`/`oneOf`/`allOf` schema
properties. Consolidated host coverage passes 199 tests and 28 subtests. Live
gates returned ordinary HTTP 400 responses for malformed grammar and excess
stops, emitted integer/boolean arguments from a `oneOf` tool schema, preserved
the multiply call, and repeated the exact multi-chunk `CODEX TOOL READY` round
trip.

Exit an earlier Qwen Codex TUI before restarting this listener. Codex's
unbounded reconnect mode can retain multiple failed turns and submit them
together when the endpoint returns; that violates this lane's sequential-use
contract. The executable setup, real multi-chunk gate, and recovery procedure
are in [`../AGENTS.md`](../AGENTS.md).

The native-Windows grammar sampler access violation is fixed in the current
worktree. The selected TP or attention-TP process group now has to contain
more than one rank before `_sync_token_ids_across_tp` enters its existing
CUDA token-ID collective. This restores the same one-rank identity invariant
used by `GroupCoordinator`: the qualified lane has `tp_size=1`, so grammar
sampling keeps its local token and never submits it to Gloo.

An isolated unguarded CUDA/Gloo reproducer reached the reported
`ProcessGroupGloo::allreduce -> enqueue` access violation immediately. The
patched sampler completed the same real one-rank process-group path. Four
focused policy cases passed, covering grammar and forced-sync bypass at size
one, the preserved MIN reduction at size two, and the inactive size-two path.
A five-slot full-model launch then captured target verify, draft decode, and
draft extend and served two simultaneous parsed-tool chat requests plus two
simultaneous Codex `/v1/responses` requests. Both request pairs produced one
queued request, every response completed exactly, and `/health` remained
green. That validation server was stopped through its foreground launcher;
port 30000 is currently free and the GPU is at ordinary display residency.

## Qualified measurements

| Gate | Result |
|---|---|
| Current argument-free sampled, exact `6213/512` | **162.500 tok/s** five-run mean; every individual sample >=150 |
| Independent explicit-switch sampled, exact `6213/512` | **155.961 tok/s** five-run mean |
| Current native acceptance | **2.737968** accepted length, **0.249045** rate, 326/1309 correct/proposed, 187 verifies |
| Current exact capacity, `199000+16` | **3118.215 prompt tok/s**, **63.818572 s TTFT**, **64.236571 s E2E**, `199016` total |
| Historical NEXTN sampled, exact `6213/512` | **122.712 tok/s** ten-run mean; **122.371** median; **137.074** peak |
| Historical NEXTN fixed work, accepted length 3 | **171.263 tok/s** five-run mean; all runs retained the established digest |
| Primary exact `199000+16` record | **3078.058 prompt / 114.617 generation tok/s**, TTFT **64.651152 s**, E2E **64.782022 s** |
| No-override default relaunch | **3052.437 prompt / 114.053 generation tok/s**, TTFT **65.193816 s**, E2E **65.325334 s** |
| Selected-cache exact prompt window | **3047.309 tok/s** five-run mean; every request exact `199016` |
| Selected-cache long generation | **118.389 tok/s** three-run mean at exact `199000+512` |
| Current eager-fusion prompt window | **2987.275 tok/s** five-run mean in the drifted current environment; **0.914%** above the adjacent PERF-028 arm with the established digest restored |
| Current eager-fusion long support | **3001.344 prompt / 115.225 generation tok/s** over three exact `199000+512` requests; established digest restored |
| Behavior | Coherent preserved thinking; correct `703`; exactly one `multiply({"a":37,"b":19})` tool call |
| Surface | Image and audio understanding reported false |
| Final production relaunch | Argument-free DSpark-v2; exact `199016`, intended target/draft graphs, arithmetic/tool continuation, OpenCode2, Codex, and image/audio false |

The earlier NEXTN promotion used seed `783025237` to make matched investigations
easier. It remains a historical control. The selected DSpark-v2 lane uses
ordinary stochastic rejection sampling and the trained Markov proposal; its
argument-free qualifying restart used launcher seed `929684324`.

## Active tree correctness hold

A deterministic non-front acceptance reproducer found that the unified hybrid
pool passed virtual target-KV slot ids directly to a physical backing pool. The
multi-layer EAGLE caller also allowed tree-path front compaction to be skipped
while draft extend indexed tokens and hidden rows as a compact front block.

The recorded M8/M12/M16 servers used `enable_unified_memory=False` and the
single-layer finalizer was already active. The unified-pool reproducer therefore
establishes a real optional-path defect, not proof that those exact requests
were corrupted. Their production hold remains because a full current-config
cross-cycle state comparison has not established target KV, recurrent state,
next-draft state, reclamation, and next-cycle proposal parity together.

Signed commit `3f276e8acda4` installs physical full-KV translation for relocation,
preserves MLA's separate dense kernel address space, and makes accepted-path
compaction mandatory for every top-k tree worker. Factory-created MHA/MLA tests
and a captured four-cycle serial-path comparison now cover target K/V,
token/hidden compaction, terminal next-draft state, rejected-slot reclamation,
and virtual-id reuse. The combined accepted-path, composite-graph, and GDN CUDA
suite passes eight tests plus two subtests.

Every earlier tree throughput result remains mechanism-only. The qualified
linear DSpark-v2 **162.500 tok/s** default result is the production comparison
authority until a corrected full-model non-front path comparison and all
ordinary promotion gates pass.

A fresh production-linear comparison from correctness commit `3f276e8acda4`
retained all tree/device-cycle controls off. Its first five-run window averaged
**112.253 tok/s** while warming through an 84.130 tok/s first request; the
second independent window averaged **124.775 tok/s**. The ten-run combined mean
was **118.514 tok/s**, and five native acceptance probes averaged **2.204748**.
This confirms the top-k-one comparison path remains within the established
performance range while preserving the startup sample and lower stochastic
acceptance as real evidence.

An exact-q device-resident linear cycle was then qualified functionally and
closed for throughput. The dense-race form averaged **122.576 tok/s**; an
explicit-seed FlashInfer categorical refinement averaged **120.075 tok/s**
despite **2.277991** emitted tokens/cycle. Its normalized 21.132 ms/cycle
remained above the ordinary path's 20.771 ms/cycle. The architecture stays
opt-in, SWOR is rejected on raw-composite RNG grounds, and production defaults
remain unchanged.

## Active performance handoff

The user designated the exact `199000+16` request in the real 200K pools as the
primary performance scoreboard. PERF-062 sets the accepted record on the
promoted launcher defaults at **3078.058 prompt / 114.617 generation tok/s**,
with **64.651152 s TTFT**, **64.782022 s** end to end, exact `199016` tokens,
and `finish_reason=length`. Root [`../BENCHMARK.md`](../BENCHMARK.md) is the
compact authority.

The next target is **3100 prompt / 120 generation tok/s**, **<=64.20 s TTFT**,
and **<=64.35 s** end to end in one eligible exact request. The timing limits
are mathematically tied to the two throughput thresholds. The user explicitly
accepted the current all-four-metric record as production; the higher milestone
remains future work.

PERF-028 and PERF-027 are now retained additive changes on the active source
line. PERF-028 fuses residual-add plus Gemma norm and improved adjacent exact
long generation by **1.205%**. PERF-027 fuses eager Qwen SwiGLU directly into
the NVFP4 tuple consumed by `down_proj`; its repaired exact window improved
prompt by **0.914%** and TTFT by **0.606649 s** versus PERF-028 while restoring
both deterministic digests. PERF-027 deliberately bypasses
`torch.compiler.is_compiling()`: Inductor removes the eager BF16 SiLU rounding
boundary, so the compiled M3 target graph retains its former function until a
separately exact compiled-semantics producer qualifies.

PERF-035's provisional FP16 QK reduction was removed. Although the first
five-request A-B moved **2985.317 -> 3005.592 tok/s**, the corrected
24-query-head/4-KV-head/256-dimension exact prefix ladder measured FP16
**163.705 ms slower** across 16 layers. FP32 remains selected.

PERF-036 closes the practical native FA2 tile family for the same dominant
kernel. CTA-Q 16 regressed to **4154.807 ms/layer**, CTA-Q 32/128 are invalid,
and the correctly routed CTA-Q-64 `NUM_MMA_KV=2` candidate regressed the exact
ladder from **3013.932 to 3414.968 ms/layer** while changing output and LSE
digests. The maintained and installed FlashInfer headers are restored to
matching SHA-256
`2E5927BDC0D36DDB393CB4FAB68C2E958D65D5B4B0085C969F7CFA777ECDFB5B`;
CTA-Q 64 with `NUM_MMA_KV=4` remains selected.

PERF-037 closes standalone norm-to-NVFP4 fusion. Its native SM120 producer was
bit-exact at all production shapes, but the captured M3
norm+quant+gate/up-GEMM boundary moved **0.096704 -> 0.097152 ms/layer**.
FlashInfer's existing PDL chain already hides the quantizer behind GEMM
startup; the prototype and its exact JIT cache were removed before model
wiring.

PERF-038 closes plain sub-128-row SM120 NVFP4 tiling. Cooperative CUTLASS
requires CTA-M >=128; ping-pong accepts a 64-row MMA but cannot map NVFP4's
fixed 128-row scale-factor TMA atom. The selected M3 tactic already swaps A/B
and uses CTA-N 32, the minimum supported epilogue/LDSM width.

PERF-039 closes MTP dual-norm/concat fusion. An occupancy-preserving native
two-CTA producer was bit-exact through the dependent FC but saved only
**1.248 us at M1** and **2.080 us at M3**, about **0.0033 ms** over both draft
phases. It was removed before routing.

PERF-040 closes the gate/up epilogue as a small extension: selected tactics are
swap-AB DP, while stock CUTLASS EVT cannot pair accumulators or halve the
output. A future implementation needs a distinct custom collective.

PERF-041 is retained in signed commit `7cb4ed0796` as a default-off native
sparse top-p producer. It is exact under the selected finite top-k contract,
including AIR boundary and cutoff-tie semantics. The first A-B-A improved the
**17.322 ms** control median to
**16.954/16.002 ms**; final regression-reviewed source independently reached
**16.000558 ms** with identical output and acceptance. The predecessor
one-pass arm did not promote client throughput: exact long generation averaged
**111.559 tok/s** with **2.194869** mean acceptance. Continue stacking
acceptance-neutral native wins; AIR remains the production default.

PERF-042 is retained in signed commit `afd5606077` as the new exact-request
prompt and timing leader, pending full promotion gates. A native page-table
builder lets aligned ordinary
prefix prefill consume the existing physical 64-token pages directly. Five
exact `199000+16` requests averaged **3209.728 prompt tok/s** with
**61.999103 s TTFT** and **62.153173 s E2E**; the worst prompt was
**3205.270**, so every prompt/time threshold passed. All 25 isolated prefix
shapes matched page-1 output and LSE bit-for-bit. Short generation averaged
only **98.029 tok/s**, so the combined four-threshold objective remains open.
Eight focused/fast-plan tests and the final adversarial review pass after
repairing stale page metadata, ownership, MXFP8 admission, and mapping checks.
Static draft top-k 32 was immediately rejected after acceptance fell
**2.217279 -> 2.173943**. Greedy draft top-k 1 was also rejected: exact
generation remained **97.900 tok/s** and direct greedy acceptance averaged
only **2.107020**. Draft top-k 16 also lost at **2.205710** versus k20
**2.217279**; k1/k8/k16/k32 close the static support-size family.
After repairing all proposal-owner routes, proposal-only top-p 1.0 is retained
default-off in signed commit `6b963eed05`. It removes both q top-p transforms, improving matched
M3 mean/median/p90 by **0.194/0.185/0.149 ms** and raising five-probe
acceptance slightly to **2.229702**. Exact short generation still reached only
**87.402 tok/s**, so the target remains open. Proposal penalty scales 0.75 and
0.0 reproduced identical proposal/output sequences, closing that scalar.

PERF-048 ReplaySSM commit overlap is rejected. Although **186.819/189.478 us**
of fold time overlapped draft extend, the draft-extend graph expanded
**1.060552 -> 1.237001 ms** from bandwidth contention. Serial fold+extend was
about **1.234266 ms**, slightly below the overlapped span; the apparent
full-cycle gain was noise. PERF-046 remains the selected generation line.

PERF-049 closes static proposal calibration. Two branch-exact chronological
p/q corpora found a large support ceiling (**2.737586**) but no generalizing
gamma, rank, or token correction; the maximin expected-length gain was only
**0.000133** and learned weights regressed held-out chronology. The opt-in
capture queue was repaired from eight to bounded 64 entries after the first
capture backpressured; the second request completed 239 records.
Signed commit `4d6782121e` retains the bounded diagnostic queue repair.

PERF-050 reopens greedy draft top-k 1 on stronger evidence. It improves
exact199K+512 generation **116.549 -> 123.049 tok/s** with identical output
and reaches **2.426540** exact-context acceptance. However, five exact16
generation scores average only **98.478 tok/s**: the request uses seven
long-context cycles averaging **19.895 ms**, so the single-request generation
target remains open. M4 still used seven cycles and is rejected. XQA SM-count
and PDL controls provide no bit-exact material win.

PERF-053/054 close the retained device-resident cycle under greedy k1 and the
available XQA structural constants. Device composition produced only
**97.730 tok/s** on exact16 with the control digest. Valid native XQA buffering
and V-tile variants saved at most **0.960 us/call**; the apparently faster
single-K-buffer build was nondeterministic. Installed FlashInfer source and
the exact control JIT module are restored.

PERF-059 retains a default-off native CUDA draft-k1 delta producer. It reduces
proposal construction from **73-87 us to 3.7-3.9 us** and improved matched
exact199K+512 generation **122.352 -> 123.559 tok/s**; an independent restart
averaged **123.831 tok/s**, with identical output and all behavior gates.
Exact16 improved only to **99.173 tok/s** and remains seven-cycle limited, so
the root benchmark target is still open.

PERF-056 proves q20 support is not the exact16 blocker: every required target
token is present and a perfect rank oracle completes in six cycles. The first
hidden-conditioned PCA-linear rank heads failed locked minority validation and
are rejected. Target-hidden residual, q-tree, KNN, and RBF follow-ups also
failed the locked ranks despite an accurate target-hidden teacher. The
default-off diagnostic now preserves exact q, hidden
payloads, target ranks, and realized accept length with bounded backpressure.

PERF-061 closes stock target-FP4 tactic and PDL changes. Global PDL-off
regressed. Bit-exact qkvz/down tactic changes projected 0.361 ms synthetically
but moved real long generation only **123.831 -> 123.972 tok/s**, inside noise.

PERF-062 historically promoted the native-Windows gate/up hybrid. A coalesced native
Cutlass-to-Marlin relayout matches the canonical repacker bit-for-bit and
reuses one 85 MiB scratch buffer across 64 target gate/up projections. The
accepted exact request reached **3078.058 prompt / 114.617 generation tok/s**,
**64.651152 s TTFT**, and **64.782022 s E2E**, beating all four prior record
metrics. A no-override launcher restart independently beat the old record at
**3052.437/114.053**, preserved exact `199016`, and passed every behavior and
client gate. That NEXTN profile remains a qualified control; the later
DSpark-v2 promotion at the top of this document supersedes it as the launcher
default.

The selected target remains `AttnNVFP4` with the bit-exact Windows Gemma
residual-norm direct-output path, while production speculation is now DSpark-v2
with chunk 4096 and five Mamba slots. The NEXTN M3/chunk-7680 configuration is
retained as a control. The earlier independent retune
produced exact prompt samples
`3051.345, 3048.538, 3048.086, 3042.488, 3044.105`, mean **3046.912**.

FlashInfer 0.6.17 stores file hits in process-global `_file_configs`, which
later draft autotune contexts replace. The retained adapter promotes only
target EXTEND file hits actually exercised by the pass into the runner-keyed
process cache. A clean relaunch promoted 110 entries from the selected
20,928-byte cache, SHA-256
`8219484FA86EBB0E6DDA54F2D15447DBC502EBCEA9007B3E1BB917B9001F9ADF`,
without re-profiling. Its five exact prompts averaged **3047.309 tok/s**.

Long generation is the stable generation authority because exact-16 measures
only 15 post-first-token intervals. Three selected-cache `199000+512` requests
averaged **3047.754 prompt / 118.389 generation tok/s**, while five real
sampled `6213/512` requests averaged **126.252 tok/s** and five native probes
averaged **2.217256** accepted tokens per verify. All deterministic windows
retained their selected-tactic digests.

Cache-only and dummy-only controls both returned to about 3009 prompt tok/s
and the baseline digest, proving tactics rather than stale recurrent/KV state
caused the gain. Profiling afresh on every launch was rejected: it retained a
3043.747 exact prompt mean but selected a long-generation tactic family that
averaged only 101.162 tok/s. Keep the selected cache and requalify any new
tactic selection.

The branch began from a five-run current-source control of **2871.358 prompt /
90.459 short generation tok/s** and an adjacent A2 control of
**2926.303/92.782**. The selected-cache prompt mean is 4.136% above A2 and
1.023% above the prior 3016.444 record. The benchmark client fails before
sending a request unless prompt calibration equals the requested count exactly.

Base RadixArk at chunk 4096 remains the historical production control at
**2608.263 prompt / 102.358 generation tok/s** on the same exact workload.
The earlier 122.712 tok/s `6213/512` NEXTN result and 200-TPS geometry objective
remain historical diagnostic context, not the current launcher defaults.

The older NEXTN trace collected 1,471 transition records and closed its best
repeatable graph-tail opportunity at conservative p10 **0.658355 ms**. DSpark-v2
later exposed a different roughly 1.67 ms target-to-draft boundary. Static
all-row K/V commit safely moved its projection/cache writes into the target
graph, reduced that gap to about 1.10 ms, and is now selected. Further graph-tail
work is closed absent a newly measured production gap.

The M3 target trace contains 61 exact graph-2 replays at **15.322 ms mean** and
**14.661 ms median**. Full target-start-to-target-start cycles averaged
**19.446 ms**. Primary GEMMs consume **13.086 ms aggregate** per replay,
**12.360 ms** on the terminal stream, and **11.821 ms** of exclusive observed
wall. Exact mathematical shape attribution ranks:

1. NVFP4 MLP gate/up, `M=3,N=34816,K=5120`: **4.211 ms** terminal-stream;
2. FP8 GDN qkvz, `3x16384x5120`: **2.851 ms** terminal-stream, **1.675 ms**
   exclusive wall because alternate-stream work overlaps it;
3. NVFP4 MLP down, `3x5120x17408`: **2.328 ms**;
4. FP8 output projections, `3x5120x6144`: **1.484 ms**;
5. FP8 full-attention qkv, `3x8192x5120`: **0.946 ms**;
6. NVFP4 lm-head, `3x248320x5120`: **0.539 ms**.

The BF16 GDN `in_proj_ba` consumes 0.726 ms of aggregate device work and has
only 0.000081 ms of exclusive observed wall at M3, so it is already hidden by
the qkvz stream. The draft-decode and draft-extend graphs span **1.217 ms** and
**1.063 ms**; together with inter-graph scheduling they account for the other
roughly 4.124 ms of the full cycle.

Two exact-shape, distinct-weight CUDA-graph windows funded selective conversion
of the exposed FP8 projections to NVFP4. Their overlap-adjusted cycle
projections were **1.976456 ms** and **1.865227 ms**. The derived checkpoint
then reduced the measured M3 cycle from 19.446 ms to **17.315 ms**, passed exact
`199000+16`, and established the earlier **2838.980/107.253 tok/s** scoreboard
record that the direct-output profile later superseded.

The funded plain M4 K+1 retest is now closed on current source. M4 raised
accepted length from **2.245614** to **2.327273** (+3.636%) while increasing
the matched full cycle from **16.058328** to **18.419190 ms** (+14.702%).
Measured projection fell **139.841 -> 126.350 tok/s**. Warmed exact-200K
prompt means were indistinguishable, and generation means overlapped their
8-11% variance. M3 remains selected.

FlashInfer paged-only prefill is also closed. Against the restored default,
exact-200K prompt changed **2789.036 -> 2785.260 tok/s** and 512-token
generation changed **106.467 -> 104.117 tok/s**. The default ragged-current
plus paged-prefix merge remains selected.

Applying chunk 7680 to base RadixArk reduced exact prompt throughput to
**2226.770 tok/s** and left only 200 MiB free before follow-up probes. The new
production launcher avoids that failed combination by selecting the
attention-selective checkpoint; base RadixArk remains paired with chunk 4096
when used as a control.

Current measured geometries fail the path-length oracle before proposal
quality is considered. M3's depth-two maximum is 154.270 TPS at mean cycle
cost. M8, corrected M12, and M16 depth-four best-sample impossible ceilings are
185.782, 179.547, and 166.666 TPS. Width/topology implementation remains
unfunded.

Branch-exact p/q capture, branch-local presence/frequency/repetition state,
startup worker/compile provenance, and deterministic replay tooling are now in
the worktree. The live six-cycle artifact is deliberately marked
`capture_scope=selected_tree`; later states have incomplete support. It can
replay the observed membership and cannot qualify aligned, irregular,
calibrated, SWOR, confidence-gated, or target-aware counterfactuals. The replay
gate requires complete lattice coverage and requires every geometry candidate's
conservative lower TPS to strictly clear the measured frontier's best-case
upper TPS before applying the 215-TPS funding floor.

MiaAI-Lab's single-5090 vLLM 0.27.1 recipe remains relevant architecture
evidence: it uses the same RadixArk checkpoint with MTP-3, TurboQuant 4-bit KV,
and a patched all-GPU K+1 verify route. Any reproduction or SGLang port now
ranks first on the exact `199000+16` scoreboard. Short-context acceptance and
device-cycle measurements remain supporting diagnostics.

## Apple-silicon experimental handoff

The Apple route is active as an experimental lane on an M1 Max with 32 GiB
unified memory. At the user's request, the pinned Q1 cache was deleted after
its experiment, reclaiming 7.9 GB; its immutable revision and checksum remain
in the 19:09 experiment-log entry. The current retained playground is
Bartowski's conventional `Qwen3.8-27B-IQ2_XXS.gguf`, pinned at revision
`f0eec4a4bb4975114a030d048952d83c0a53c034`. It is 9,393,043,040 bytes with
SHA-256 `b01f668356e5799fd76315bd6abc0e45234580409ebc5c8fb4b675e3c10dc2b9`.
All 866 tensors use formats supported by the native torch/MPS route.
Signed commit `7740cae691` owns that packed low-bit GGUF route and its MPS
convolution-state contract. Signed commit `1271610e0b` owns the separate,
opt-in MLX quantized-prefill query tiling mechanism. Signed follow-up
`ea983f3120` expresses its cross-tile lifetime ordering with `mx.depends`,
which preserves query values and avoids arithmetic dependency propagation.
Signed commit `8879ed3d01` registers GGUF USER_DEFINED vocabulary entries as
ordinary added tokens while keeping CONTROL entries special.
Signed commit `13bea403d6` stores heterogeneous merged GGUF shards in one
compact MPS allocation and passes storage-offset views directly to Metal,
while preserving the padded CUDA/non-MPS path.
Signed commit `16b2bf7a06` specializes aligned IQ2_XXS batch-one matvec with
four-row input/LUT reuse. Signed commit `b19cf4acf3` specializes the aligned
Q5_K batch-one vocabulary head with four eight-lane row cohorts per SIMDgroup
and retains generic alignment and multi-batch fallbacks. Signed commit
`4d1641fdcd` routes one-vector F32 GGUF projections through native MPS matrix
multiplication while retaining the custom dense kernel for multi-vector
prefill.
Signed commit `210a214c12` removes padded prefix query rows from the shared
torch-native extend path and uses lower-right causal alignment for partial
chunks.
Signed commit `b2b8ab4af8` gates fused native MPS decode on its actual dtype,
physical-pool, layout, and head-dimension contract so BF16 and long pools use
the established cache-write plus SDPA path.
Signed commit `52b5326d8e` retains PERF-A016, a batch-one Q4_K tensor-family
kernel for the mixed-format IQ2_XXS/Q2 checkpoint. It reuses each activation
fragment across two output rows and admits complete four-block cohorts with
safe compact-view alignment.
Signed commit `4dfa1ad3ef` retains PERF-A021, a batch-one Q2_K tensor-family
kernel that reuses each activation fragment across four output rows. It is the
current Apple benchmark baseline and preserves generic aligned, tail, and
multi-batch fallbacks.
Signed commit `1ec20a0e87` widens the fixed-memory BF16 Metal decode fence to
131,073 physical rows. Isolated native admission, the 131,074-row fallback,
and active sequence length 131,072 all pass. Signed commit `5f966ecb0d`
splits long BF16 tiled decode history and stably reduces the exact 131,072-row
attention median from **148.078959 ms** online and **65.117542 ms** unsplit to
**4.338625 ms**. Focused parity, fragmented maps, nonzero storage offsets, and
12 outstanding asynchronous output lifetimes pass.
Signed commit `0d1d0ea643` owns PERF-A017's fixed-memory BF16 paged-GQA
EXTEND mechanism and its separate lazy Metal pipeline.
Host cleanup leaves this artifact as the only Hugging Face model cache and no
MTPLX model cache. A broader cache cleanup also removed the first retained
copy, so the same immutable revision was downloaded again and its byte size
and SHA-256 were reverified. SGLang, Codex-runtime, uv, and other rebuildable
user caches were cold at that cleanup checkpoint. The data volume had 267 GiB
free after restoration.

The selected native-IQ2 `128+32` deterministic window averages **7.0444
prompt / 8.4406 generation tok/s**, **18.170302 s TTFT**, and **21.842957 s
E2E** over five cache-flushed runs. This is **164.94%** more generation
throughput than the pre-kernel padded control's **3.1858 tok/s**, with the
same deterministic digest. An independent committed restart reached
**8.420 tok/s**. Required-sampling windows averaged **8.3094** and **8.2942
tok/s** across two restarts, 4.59% and 4.26% above the corresponding selected
Q5_K windows.

Packed weight loading reports **9.03 GB**, down from 10.03 GB; Mamba and KV
allocations add about 0.41 GB. Matched generic/candidate IQ2_XXS
`17408x5120` medians reached **1.176875 -> 0.516000 ms**. The Q5_K
`248320x5120` vocabulary head reached **19.659291 -> 3.754625 ms** in the
matched source window. Across all 48 actual F32 `96x5120` b/a projections,
the selected native MPS path changed **7.296667 -> 2.159000/2.051708 ms** in
an A/B/A sweep. Actual-file batch 1/3/4/8 parity, odd row boundaries, compact
offsets, and focused packed extrema pass.

The earlier missing-final and empty-thinking-disabled behavior was a native
GGUF tokenizer defect rather than checkpoint evidence. Qwen's `<think>`,
`</think>`, and tool markers are GGML USER_DEFINED entries; before
`8879ed3d01`, they fragmented into ordinary text pieces. A fresh explicit-parser
server now returns preserved `reasoning_content` plus visible final `703`,
exact thinking-disabled `READY`, exactly one parsed
`multiply({"a":37,"b":19})` call with `finish_reason=tool_calls`, and a
tool-result continuation ending in `37 × 19 = **703**`. `/model_info`
continues to report image/audio understanding false. The focused tokenizer,
reasoning-parser, and tool-parser suites passed 321 tests plus 64 subtests.

Pinned llama.cpp build 10547 owns the route-neutral M1 Max Q2 `12+256`
reference at **14.661356 tok/s** aggregate and **14.671473 tok/s** best hit.
The first native Rust `/generate` baseline remains **7.001584 tok/s** aggregate,
**7.015010 tok/s** best hit, and **36.563154 s** mean E2E. It established the
official-tokenizer fixed-output boundary with exact token IDs and FNV
`6d4d220de481f54e`.

PERF-A016 remains the historical first named-client-qualified
repository-native result on the tool-capable Python ingress with the same
official tokenizer. Its final-source five-run window is
**8.586948 tok/s** aggregate, **8.591773 tok/s** best hit, and **29.812688 s**
mean E2E. The fresh disabled-kernel control is **7.009167 tok/s**, attributing
a **22.510241%** full-model gain; an independent candidate restart reaches
**8.578205 tok/s**. Candidate, control, and restart all reproduce exact
`12+256` token IDs, text, length finish, and digest. Final parity covers the
enabled complete cohort and output-tail fallback. The safe host rule requires
Q4_K, batch one, four-row output alignment, four-block cohorts,
`weight_offset % 2 == 0`, `input_offset % 4 == 0`, and Apple7+ pipeline
capability. `Q4_K` names one internal tensor family among the checkpoint's 866
mixed-format tensors; benchmark and checkpoint standing remain Q2.

PERF-A021 is the current Apple native-SGLang benchmark baseline. Its
independent exact-`12+256` window reaches **9.189086 tok/s** aggregate,
**9.194647 tok/s** best hit, and **27.859136 s** mean E2E. The fresh generic-
Q2_K control is **8.515065 tok/s**, giving a **7.532647%** matched gain and a
**7.012249%** gain over PERF-A016. Actual Q2_K gate/down medians move from
about **1.07/1.09 ms** to **0.455/0.454 ms**. The safe host rule requires Q2_K,
batch one, eight-row output alignment, four-block cohorts,
`weight_offset % 2 == 0`, `input_offset % 4 == 0`, and Apple7+ pipeline
capability.

The current five-sample reasoning-enabled `128+256` stream establishes all
four latency metrics together: **22.945718 prompt / 9.156675 generation
tok/s**, **5.578383 s TTFT**, and **33.426973 s E2E**. The current-source
1,024-token-chunk route completed exact **32761+1** inside the 32,768-token
BF16 pool at **18.942 observed prompt tok/s**, **1729.565719 s TTFT**, and
**1729.565822 s E2E**. It passes sampled reasoning, exact arithmetic `703`,
thinking-disabled `READY`, one parsed multiply call, tool-result reasoning
continuity, and image/audio-disabled reporting.

The experimental long-context server now allocates real
`context_length=max_total_tokens=131072` with BF16 KV and one request. Five
no-buffer Mamba slots use 0.87 GB, the K/V pool uses 8.00 GB, packed weights use
9.03 GB, and **10.97 GB remains** after allocation. Health and language-only
reporting pass. The fifth slot is retained as Codex transient-state headroom;
four- and five-slot experiments both proved that a late prompt-token change can
miss a compressed terminal recurrent checkpoint, so slot count does not solve
that radix-path issue.

The current native-MLX early-out27 v2 decode line is signed at `4c1bc4c1e3`.
Its reusable power-of-two full-attention K/V storage first changed exact
6,237-history direct decode **17.923409 -> 19.151623 tok/s**. The latest retained
kernels make each single-token recurrent causal-convolution/state transition
one Metal launch, fuse the following BF16 SiLU into that same owner, fuse 127
residual/RMSNorm boundaries into dual-output launches with distinct residual
storage, combine q/k normalization with float scaling, and combine recurrent-
output RMS normalization with the SiLU gate across all 48 recurrent layers.
The convolution/state step improved matched serving **18.7616 -> 18.8914
tok/s**; residual/RMS fusion improved its matched control **18.8286 -> 19.0484
tok/s**; recurrent q/k fusion improved the next matched control **19.0900 ->
19.1610 tok/s**; recurrent output norm/gate fusion improved its fresh matched
control **19.1260 -> 19.1548 tok/s**; convolution/SiLU fusion improved its
fresh matched control **19.1730 -> 19.2134 tok/s**. All served requests retained
exact `6237+128` counts, reasoning output, stream shape, and SHA-256. The latest
actual-work generation mean leaves a **0.7866 tok/s** gap to the required floor.
Full-attention q/k/v affine row concatenation is closed under current MLX
semantics: both the all-row and k/v-only forms changed the production-shape
deterministic digest despite exact synthetic row arithmetic. PERF-FA084 retains
their bounded screens. Recurrent beta/decay work inside the q/k normalization
owner is also closed: the beta-only isolation was neutral, while the complete
exact owner changed matched long-history decode **19.641655 -> 19.547612
tok/s**. PERF-FA085 retains the five-pair evidence; selected source and digest
are restored.

The native engine's command-buffer byte default is now under qualification at
128 MiB, with explicit `MLX_MAX_MB_PER_BUFFER` values retaining precedence.
Five exact adjacent direct pairs improve **19.623300 -> 20.153966 tok/s**
(**+2.704%**), and every candidate run clears 20. Its real 131K served gate is
the remaining promotion step.

The normal hash-pinned client prompt at `xhigh` renders 6,232 tokens. A midnight
date update changed exactly one token at index 6,003. Infrastructure priming at
aligned endpoints 2,048, 4,160, and 5,952 completed in separate bounded windows
at **81.027809 / 85.022104 / 72.424614 seconds**; the last endpoint sits before
the mutable date token. Actual Codex thread
`01a056b0-54e5-7a60-921a-c7a05ef0843a` then completed in about **91.6 seconds**
under `/opt/homebrew/bin/timeout --signal=INT --kill-after=10s 120s`. It ran
exactly one `exec_command`, observed exact stdout
`QWEN38_XHIGH_SPLIT_DECODE_TOOL=passed`, and returned the requested
`QWEN38_XHIGH_SPLIT_DECODE_READY`. Usage was **12,678 input / 12,328 cached /
188 output / 131 reasoning-output tokens**. The two server turns reused
5,952+280 and 6,376+70 prompt tokens. The actual Responses body omitted
`max_output_tokens`; the default instructions and repository prompt remained
byte-identical. This qualifies parser-enabled, ordinary-prompt, uncapped-output
tool use at `xhigh` with real 131K client/server metadata inside the two-minute
contract. Cold 6K prefill remains the leading interactivity gap and autonomous
multi-file ownership remains open.

Historical process-scoped OpenCode 1.18.15 runs admitted 13,635 and
13,691-token agent prompts. The 18:29 Codex 0.151.0 `qwen38-local` read-only
gate remains historical evidence with hashes `9706003a...9a1c` and
`680e762e...e970`. The 19:46 strict profile-overlay write also remains
historical behavioral evidence at `d347c93e...f0ba` / `eb15e828...1db1` /
`8a2fe9b9...2279`; it used same-value sandbox/approval pins, and its mutable
lower user config and rules were not hashed at process start.

The governing Apple workspace-write client gate now uses the dedicated
`CODEX_HOME=/Users/dcazares/.codex/qwen38-local-hardened-home`. Final
config/catalog/instruction SHA-256 values are
`9d7842bb47d15c5b7a63d1507b8e035784bf1ab768de36dbb131088493620409`,
`862339c156824879852dbdc9ebf096523d6312699fdd8723f13d81091de2ec71`,
and `5d59350d7a1568c3c458b05513e8b58ed50d70874f27fb80a06d131e09b9d096`.
The exact strict task used no profile layer, `-c`, `--sandbox`, reasoning, or
capacity override. It issued one successful `file_change` on the first attempt,
added only `gate.txt` with exact content
`QWEN38_ISOLATED_WRITE_GATE=passed`, returned visible
`QWEN38 ISOLATED WRITE READY`, and exited zero with **2,670 input**, **115
output**, and **39 reasoning-output tokens**. The 34-byte file SHA-256 was
`f7ca43b4d2b9698e2f794c8bfffefe78836423c77bde38a7405f96ab12f6729a`.

After the PERF-A021 server is ready, the normal interactive entry is:

```bash
env CODEX_HOME=/Users/dcazares/.codex/qwen38-local-hardened-home \
  SGLANG_API_KEY=local /opt/homebrew/bin/codex --strict-config \
  -C /Users/dcazares/sglang
```

The repository is trusted so root `AGENTS.md` reaches the interactive prompt;
the final separate prompt-input diagnostic rendered 21,741 characters and
contained its exact C++/CUDA-only rule. Qualification scans that did not rely on
ignore rules found no project `.codex`, hook, rule, override-instruction, or
skill sidecar; no dedicated-home rule, skill, or override-instruction path; and
no `$HOME/.agents/skills` or `/etc/codex/skills` directory.
Recheck those mutable interactive inputs before launch. The fixed scratch gate
is untrusted and supplies `--ignore-rules` explicitly.

The normal trusted-repository prompt and existing interactive configuration
remain the authoritative working lane. A one-off Codex 0.151.0 diagnostic with
`project_doc_max_bytes=0`, `include_environment_context=false`, and reasoning
effort `none` reduced debug prompt input to 100 tokens; those overrides are
confined to diagnostics and disposable scratch gates. Compact scratch thread
`01a05638-3921-7411-9564-f680af8ea1b8` issued a real `printf` tool command,
received `QWEN38_COMPACT_TOOL=passed`, returned the requested final marker, and
exited zero in about 52 seconds. A following automatic header-edit prompt
returned only `ack`, so tool selection for edits remains a supervision point.

Parser-free `--grammar-backend outlines` is retained as an opt-in supervised
structured subworker on MPS, outside the normal Codex prompt. Four JSON-schema
Responses requests completed inside their individual 120-second process
deadlines and generated a header, validation/sort/merge implementation, and
authored-test intent. Host review preserved repository paths and namespaces and
corrected generated API/type drift. The strict three-file C++20 build emitted
exact
`QWEN38_CPP_MULTI_FILE_GATE=passed`, while the immutable verifier retained SHA-
256 `37726f73...c9ec0`. This qualifies supervised multi-file repair within the
two-minute request contract. Autonomous multi-file editing remains open.

Every future non-interactive Qwen/Codex work attempt runs inside
`/opt/homebrew/bin/timeout --signal=INT --kill-after=10s 120s`. Hybrid
`UnifiedRadixCache` remains established for delta-sized continuation prefill.
Default xgrammar vocabulary-mask application currently raises `Unsupported
device: mps`; retaining Qwen reasoning/tool parsers with Outlines also fails at
the structural-tag/backend-mask boundary. The selected local structured lane
therefore omits both parsers and uses reasoning effort `none`. The ordinary
Qwen-parser Codex lane now owns a successful auto-selected shell tool round
trip at `xhigh`; required named-tool grammar remains a separate qualification
target.

A separate 1,000-token Total-scope gate on the immediate medium-reasoning
predecessor hashes
`39ad0f7c97ed30d36d41baf5d2b6ec3c127e2e44baf9aad2aa76f6bbf70c832b` /
`8fbb54a5407b9279c1abcc61a805bb15067fe27bf4bf6e8346bd80051572dfa5` /
`8a2fe9b979da48d5bc5a38ec22fb44f50b06de21216e3643b376ba330a4e2279`
supplies historical compaction evidence. The task recovered from a malformed
patch across two observed compaction boundaries, preserved its nonce, and
exited zero with 11,093/5,120/3,943 tokens. It is unmatched to the clean low
task, and its raw JSONL and exact warning text were not retained. Production
remains configured at 30,000 Total-scope tokens, resolving to 29,491 under
Codex's 90% clamp; near-limit runtime continuation at that effective threshold
remains a separate qualification gate.

The isolated lane passes strict config loading and explicitly disables hooks,
plugins, agents, goals, memories, message-history append, analytics, feedback,
login-shell startup, web/network tools, and other optional tool/model-visible
surfaces. It has zero configured MCP servers, and all effective skill-root
paths were absent; skill-instruction injection and bundled skills are disabled.
It caps tool output at 10,000 tokens, disables unbounded/request/stream
retries, uses a 2,400,000-ms idle bound, filters a core-only initial shell
environment, and sets `ZDOTDIR=/var/empty` before spawned shell startup. The
root-owned target
was empty and system zshenv files were absent, so spawned non-login zsh tools
do not reread `~/.zshenv`. The config trusts the repository, leaves the fixed
scratch path untrusted, and confines workspace writes with network and implicit
temporary roots disabled. The catalog declares `shell_type=unified_exec`; its
exposed tools are `exec_command` and `write_stdin`. The client sends
`parallel_tool_calls=true` while the instruction bundle requests sequential
use. Complex multi-file editing, concurrent-tool behavior, and production-
threshold near-limit compaction remain wider client gates.

Pre/post bundle hashes and the ordinary user config/rules hashes
`97f15d75...5ca9c` / `63d2d91f...61e5` were stable. System and managed Codex
layers were absent. Post-tool health, language-only reporting, cache flush,
and scratch cleanup passed. Foreground shutdown removed root/listener `36097`,
tracker `36112`, scheduler `36113`, and detokenizer `36114`. In the 20:46
post-shutdown snapshot, port 30000 and model/compiler/client process sets were
empty, the fixed scratch path was absent, memory had returned to 92% free with
zero throttled pages, and thermal/performance status was normal.

A one-shot synchronized batch-one profile now supplies a candidate-selection
diagnostic.
After excluding layers 0-10 with visible first-use cost, full-attention layers
average **3.130000 ms** and GDN layers average **1.719026 ms**. Extrapolated
across the 16/48 topology from the raw sums, the stable profile projects to
**132.593 ms/token**. The separate exact request amortizes to **142.824820
ms/completion token**; their **10.232 ms/token** numerical difference crosses
runs and context distributions, so it does not isolate an outside-layer
budget. Nested layer-8 stages suggest that projection and MLP work outweigh
the recurrent core, while that layer is itself inside the first-use region
and nested synchronization perturbs its timing. These numbers guide a cheap
hypothesis test and do not replace end-to-end timing.

Both source-level context blockers are now retained. Exact source A/B at a
`4096+4096` partial extend reduced the padded-query controls from
**542.376416/641.256125 ms** to **176.066500 ms** with exact MPS output. A
`4096+256` rung improved **97.995583/97.847500 -> 12.608125 ms**. Pre-change
decode probes failed at BF16/32,769 rows and FP32/7,937 rows. Both now reach
the cache-write plus SDPA fallback with zero observed error, while the fused
FP32/7,936 boundary remains active with maximum error `2.5331974e-07`. The
first controlled 32K-configured BF16 launch reached ready state with an
allocated 32,768-token pool and exposed large-batch IQ2_XXS projection as the
remaining prefill blocker.

Signed commit `1676c71bed` retains PERF-A014, which routes Apple7+ IQ2_XXS
batches above eight through an FP32 SIMD-matrix kernel that shares each
dequantized 64x32 weight tile across 32 input rows. The actual
`blk.8.ffn_gate.weight` (`17408x5120`) improved from
matched medians **70.074833 -> 4.250250/4.277125 ms** at batch 128, from the
adjacent **249.418209 -> 16.009833 ms** at batch 512, and from
**1971.539875 -> 124.838125 ms** at batch 4096. Aligned, odd output/batch,
minimal-tail, and fallback-boundary parity pass with maximum IQ2 relative
error `2.27121e-06`; batch one/four/eight remain unchanged.

The same 32K-configured BF16 full model completed sampled `128+32` at
**23.093 prompt / 6.736 generation tok/s**. Cache-flushed `4096+2`, which
formerly crossed the
300-second watchdog, now completes at **24.828 prompt tok/s** with
**164.975078 s TTFT**. Exact `5000+1` completes two chunks at **24.845 prompt
tok/s** and **201.251071 s E2E**. Arithmetic `703`, thinking-disabled `READY`,
one parsed `multiply({"a":37,"b":19})`, preserved tool-result continuation,
and image/audio-disabled gates all pass. Verified cleanup leaves port 30000
free, no server or workload-owned compiler process, 90% free memory, and no
recorded thermal or performance warning.

A process-scoped served control then isolated the large-batch dispatch on
exact `128+1`. Disabling it produced **7.0234 prompt tok/s** and
**18.225055 s TTFT** over five cache-flushed samples. Two independent default
launches produced **22.9556** and **22.8072 prompt tok/s** five-sample means;
their combined mean is **22.8814**, with **5.594303 s TTFT**. This is 3.258x
the matched disabled prompt rate. Every request completed exact `128+1` with
`finish_reason=length`; the true batch-eight fallback also passes actual-file
parity.

The current PERF-A021 baseline has cleared its semantic, sampled, independent-
restart, exact-capacity, and isolated-home Codex gates. Its **9.189086 tok/s**
aggregate leaves a **37.324447%** gap to pinned llama.cpp. The next measured
batch-one decode hotspot remains the compact-scoreboard handoff.

Long-context EXTEND now has a qualified native mechanism. PERF-A017 implements
batch-one BF16 paged GQA at 24 query heads, four KV heads, and dimension 256
with Q8/C64 Metal online softmax. PERF-A018 classifies consecutive eight-slot
cache runs once per C64 tile and loads their BF16 K/V matrices directly while
retaining the staged path for fragmented, invalid, and partial runs. The run
table raises threadgroup storage from 20,800 to **20,832 bytes**.

At `E=17,L=131072`, PERF-A018's median is **66.553042 ms** against a matched
same-map forced-staged **148.002792 ms**, a **55.03%** reduction, with exact
output SHA-256 parity and `0/0 MiB` post-input current/driver allocation
growth. The prior dense MPS oracle took **424.528292 ms** and added
**8,088.515625 MiB** of driver residency. Direct/fallback bitwise parity,
independent K/V storage offsets `0..7`, every prefix residue modulo eight,
mixed and malformed maps, causal sentinels, and the physical row 131,072 gate
pass. The maximum ordinary random dense-reference error in the expanded sweep
is `2.2649765e-06`. The shader lives in a separate lazy Metal library, leaving
the ordinary extension pipeline cache and its failure boundary unchanged.

PERF-A019 then derives the eight run flags from the slot-loader registers and
publishes slots plus flags through one threadgroup barrier. At
`E=256,L=4352`, matched PERF-A018/candidate medians are
**27.009396 -> 26.398334/26.592563 ms**; the zero-eligible path moves only
**58.832500 -> 58.874062 ms**. At `E=17,L=131072`, two candidate windows
reach **65.565041/65.493667 ms** against the matched **66.577542 ms**
checkpoint. Exact digests, `0/0 MiB` allocation growth, every broken cohort
position, cross-boundary runs, and repeated two-tile hybrid execution pass.

PERF-A020 hoists each SIMDgroup's two QK cache-row offsets before the D256
loop and each PV row base before its eight output fragments. Matched
`E=256,L=4352` windows improve **26.330084 -> 25.548479 ms** and
**26.649625 -> 25.674125 ms**; zero-eligible maps also improve by
**1.33-1.52%**. At `E=17,L=131072`, matched medians improve
**65.647625 -> 63.767667 ms** and **65.687333 -> 63.886417 ms**, with exact
digests and `0/0 MiB` allocation growth. Offset views, a repeated mixed map,
and physical row 131,072 retain their established arithmetic results.

The raw native binding is outside `TorchNativeAttnBackend.forward_extend`.
The explicit no-new-Python rule makes clean C++ dispatch ownership the active
architecture gate. The measured 64K generic route remains closed by swap and
forward-progress limits, so served capacity stays qualified at exact
`32761+1`. Hoisting all query matrices, a fully-causal history branch, and a
cohort-leader softmax broadcast were bitwise exact and missed their timing
gates; those designs are closed.

Direct device-matrix-load synchronization is now closed as well. PV-only,
QK-only, combined, leading-only, and trailing-only `mem_none` barrier
ablations all preserved exact output, but their best apparent movement was
only **0.720%** and the independently rebuilt source-restored control reached
**25.520083/63.745709 ms** at the diagnostic/131K shapes. Signed PERF-A020 is
restored. Because the raw binding has no clean C++-only executable edge from
the policy-owning backend under the no-new-Python constraint, further shader
work first requires a distinct reachable consumer or materially new compiler
evidence.

The deleted affine-q4 scoreboard belonged to a separate Mac Pro experiment
and carries no M1 Max record standing. The retained MLX quantized-prefill query
tiler remains mechanism code behind
`SGLANG_MLX_QUANTIZED_PREFILL_QUERY_TILE`; any future M1 use requires fresh
dependency, correctness, performance, memory, and capacity qualification.

## Behavior and capacity invariants

Every promoted candidate retains all of these:

- real `200000` context and token-pool capacity, including exact total `199016`
  when memory layout, graph coverage, cache dtype, workspace, or sampling
  residency changes;
- Qwen's selected sampled reasoning profile: temperature `1.0`, top-p `0.95`,
  top-k `20`, min-p `0.0`, presence penalty `0.0`, and repetition penalty
  `1.0`;
- preserved `reasoning_content`, coherent thinking, and ordinary completion
  behavior;
- arithmetic probe result `703` for `37 * 19`;
- exactly one parsed `multiply({"a":37,"b":19})` call with
  `finish_reason=tool_calls`;
- image and audio understanding disabled;
- an unsimulated launcher-default relaunch with every intended target-verify
  and draft-proposal/decode graph captured for the selected topology;
- standalone OpenCode2 integration using a fixed provider/workload shape;
- protected CUDA compatibility headers outside the edit boundary.

## Native-Windows implementation retained in the source line

The production and experimental commits retain several durable Windows
capabilities:

- CUDA 13.3/MSVC JIT initialization with the conforming preprocessor and a
  bounded two-job compiler pool;
- native C++/CUDA SiLU-and-multiply, standard RMSNorm, Gemma RMSNorm, full-
  attention sigmoid-multiply, and fixed-chain metadata paths behind narrow
  native-Windows dispatch gates;
- bit-exact native-Windows Gemma residual normalization that writes the JIT
  result directly into caller-owned `x`, removing a temporary tensor and copy;
- FlashInfer CUDA top-k/top-p renormalization on the speculative target path;
- aligned draft proposal transforms and exact-q capture inside the single
  multi-step CUDA graph;
- Windows registration and isolated correctness coverage for optional online
  FP8, MXFP8, dense NVFP4, and pure ModelOpt FP4 mechanisms;
- an exact GPU target-only/SWOR tree verifier, sparse shared-memory residual
  path, low-rank tree-aware GDN verification, accepted-path recurrent/conv
  commit, path and overlap oracles, and offline topology analysis tools.

Production uses the linear path. Tree/SWOR machinery and online draft
quantizers remain opt-in infrastructure with preserved tests and evidence.

## Closed experimental frontier

The following branches are closed for the current checkpoint, proposal
distribution, and cost topology:

- adaptive depth over two and three steps;
- static one-step and three-step linear speculation;
- the selective-checkpoint plain M4 K+1 retest, whose 14.702% cycle-cost
  increase outweighed its 3.636% acceptance gain;
- FlashInfer paged-only prefill, which changed exact-200K prompt by -0.135%
  and long generation by -2.207%;
- selecting the eager-exact SwiGLU-to-NVFP4 producer inside the compiled M3
  target graph, which changed the deterministic output because Inductor's
  one-rounding function differs from eager prefill;
- a separately exact compiled-semantics producer, whose isolated M3 boundary
  improved 70.848 -> 25.152 us while the profiled full-cycle median remained
  neutral at 16.045 ms versus 16.058 ms control;
- explicit FlashInfer paged-prefix split sizes 4096/8192, which each required
  2.265 GB temporary storage from the qualified 128 MiB workspace;
- packed GDN target-verify split removal, because the selected Qwen width
  already takes zero-copy Q/K/V aliases and ReplaySSM accepts their strides;
- final-tail coalescing, whose 14,680-token ragged-current pass regressed prompt
  to 1,917.509 tok/s, TTFT to 103.781 s, and changed deterministic output;
- full-attention gate-to-NVFP4 fusion, which was exact but projected to only
  0.0068 ms per M3 replay and about 20.9 ms over the exact prompt;
- global page 128/32 tuning: 128 lost exact pool capacity, while 32 bypassed
  the page-size-1 prefill interface and regressed long generation;
- global chunk-7680 promotion, which regressed the base checkpoint and
  transiently reduced headroom to 200 MiB;
- chunk 7808, which regressed the selective prompt mean to 2909.350 tok/s;
- single-layer selected-row draft-extend logits, which reduced graph memory
  but changed draft-extend 1.059 -> 1.061 ms and full cycle
  16.058328 -> 16.066558 ms;
- reusable fused metadata output buffers whose asynchronous lifetime changed
  real rejection-path scheduling;
- draft proposal top-k 8;
- full online FP8, MXFP8, and dense online NVFP4 MTP weights as performance
  defaults;
- stock Gittensor ModelOpt FP4 as the production checkpoint;
- current-q target-only and SWOR trees, fixed-width M8/M12 breadth, depth-only
  expansion, scalar depth discount, and measured topology rearrangement;
- a 232K production pool, which completed exact `231000+16` capacity yet fell
  to 98 MiB free VRAM before cache flush.

The tree implementation remains a future route only when proposal overlap or
per-depth draft cost changes enough to alter the measured economics. The
recorded topology optimizer reached fewer than 4.1 expected outputs even under
optimistic assumptions, while measured M12 SWOR throughput was 84.713 tok/s
against the 122.712 tok/s linear baseline.

## Workspace and provenance boundaries

- Preserve every pre-existing modified and untracked path as user-owned work.
- Treat `sglang.bundle` as unrelated user-owned material.
- Keep the original FlashInfer checkout and clean Windows 0.6.17 port as
  separate provenance lines.
- Keep RadixArk and Gittensor source checkpoints immutable; conversions or
  hybrids belong at separate paths with checksums and provenance.
- Preserve the protected CUDA compatibility headers. Their recorded SHA-256 is
  `304C9CDDB08FA69E680E6ABE46C02C17F992F904A4AF20B978E4CC4B767EADBD`.
- Use C++/CUDA for new performance hot paths, with Python limited to bindings,
  dispatch, configuration, tests, and launch integration.
- Run one server, CUDA test/JIT build, compiler tree, or GPU benchmark at a
  time, and target lifecycle actions by verified process ancestry.
