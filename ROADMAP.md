At the endpoint, this becomes a **native Windows inference and serving engine housed in the SGLang repository**.

There are two distinct boundaries:

1. **Python-free, PyTorch-backed:** C++ owns the server and scheduling while LibTorch/ATen still owns tensors, dispatch, allocation, CUDA graphs, and kernels.
2. **Python-free and PyTorch-free:** C++ owns tensor metadata, memory, scheduling, model execution, graph capture, RNG, loading, tokenization, HTTP, and parsing; CUDA/CUTLASS/FlashInfer/TRT-LLM own GPU computation.

The second boundary is where it genuinely stops being PyTorch. It can continue to be called **SGLang** through its API, behavior, benchmark contract, and repository integration. Architecturally, it is a new native backend alongside SRT.

The current lane remains Python-first orchestration around increasingly large native compute islands.

---

# Audit scope

This inventory is against:

- `main` at `0c755bf8243a50da112674c2a6dbc8d3784eda74`;
- clean worktree;
- the qualified defaults in [`serve_qwen38_27b_nvfp4_5090.ps1`](scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1);
- single RTX 5090, one request, language-only Qwen3.8-27B;
- 64 target layers: **48 GDN linear-attention layers and 16 full-attention layers**;
- NEXTN M3: two draft steps, three draft tokens, top-k-one chain, rejection sampling;
- FlashInfer prefill/sampling, TRT-LLM MHA/XQA target/draft decode, ReplaySSM, ModelOpt NVFP4;
- default 4096-token chunks and the explicit selective 7680-token profile.

The two commits that arrived during the audit were documentation-only. Runtime source stayed unchanged.

This covers every Python ownership class in the qualified Windows serving lane, including request-gated behavior required by the production contract. Benchmark clients, tests, documentation tooling, and unrelated upstream hardware/model backends sit outside this serving-loop inventory.

### Cadence legend

| Mark | Python executes |
|---|---|
| **S** | Startup, loading, compilation, tuning, graph capture |
| **R** | Once or several times per request |
| **P** | Every prefill chunk |
| **C** | Every speculative generation cycle |
| **O** | Every emitted output chunk; launcher interval is four scheduler cycles |
| **F** | Feature/request-gated path |

---

# The current execution path

```text
PowerShell launcher
  └─ .venv\Scripts\sglang.exe                         Python entry point
       ├─ FastAPI / Uvicorn / OpenAI serving          Python main process
       ├─ TokenizerManager                            Python main process
       ├─ Scheduler                                   Python subprocess
       │    ├─ prefill planning and allocation
       │    ├─ PyTorch model execution
       │    └─ speculative cycle
       └─ DetokenizerManager                          Python subprocess
```

The steady generation cycle is effectively:

```text
Python Scheduler
  → Python prepares draft inputs and buffers
  → replay one captured two-step draft CUDA graph
  → Python builds/wraps chain verification metadata
  → replay target-verify CUDA graph
  → Python/PyTorch penalties + softmax + RNG
  → FlashInfer renormalization
  → Python-authored Triton rejection sampler
  → Python invokes ReplaySSM verification commit
  → replay draft-extend CUDA graph
  → Python/PyTorch samples the next root draft proposal
  → Python initiates asynchronous device-to-host result copy
  → Python updates Req/cache/penalty/finish/metrics state
  → Python sends output to detokenizer
```

During captured graph replay, the recorded GPU nodes execute without Python intervention. Python resumes at the graph boundaries and owns the uncaptured sampling, commit, scheduling, lifetime, and output work.

The two draft depths are already inside a single CUDA graph, so that former inter-depth Python seam has been removed. The remaining large seams are:

1. draft graph → target verification;
2. target graph → rejection sampling and accepted-state commit;
3. commit → draft-extend graph;
4. draft-extend graph → next root proposal;
5. completed cycle → scheduler bookkeeping and D2H result materialization.

---

# Complete runtime inventory

## 1. Launcher, CLI, configuration, and process lifecycle — **S**

The PowerShell launcher itself is already outside Python. Its executable target is Python:

- [`scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1`](scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1#L125) resolves `.venv\Scripts\sglang.exe`.
- [`python/sglang/cli/main.py`](python/sglang/cli/main.py) owns top-level CLI dispatch.
- [`python/sglang/cli/serve.py`](python/sglang/cli/serve.py#L166) selects the LLM backend and constructs `ServerArgs`.
- [`python/sglang/launch_server.py`](python/sglang/launch_server.py) enters SRT.
- [`python/sglang/srt/server_args.py`](python/sglang/srt/server_args.py) owns argument parsing, compatibility checks, backend resolution, defaults, and derived configuration.
- [`python/sglang/srt/runtime_context.py`](python/sglang/srt/runtime_context.py), [`environ.py`](python/sglang/srt/environ.py), and the configuration registries publish process-global runtime state.
- [`python/sglang/srt/entrypoints/engine.py`](python/sglang/srt/entrypoints/engine.py#L849) creates scheduler/detokenizer processes, ports, ZMQ endpoints, watchdogs, and readiness state.
- [`python/sglang/srt/entrypoints/http_server.py`](python/sglang/srt/entrypoints/http_server.py#L2784) assembles the whole server topology.

### Native port unit

A C++ `main` needs to own:

- argument parsing and validation;
- resolved immutable runtime configuration;
- environment handling;
- Windows process/job lifecycle or a native threaded topology;
- readiness, shutdown, signals, watchdog behavior, and logging;
- model/server information;
- dynamic-control operations such as abort, pause, continue, cache flush, and weight/version queries.

The existing PowerShell environment/bootstrap scripts can continue to launch the resulting `.exe`.

---

## 2. HTTP, OpenAI protocol, JSON, and SSE — **R/O/F**

The qualified `/v1` surface is Python:

- [`http_server.py`](python/sglang/srt/entrypoints/http_server.py) owns FastAPI routes, validation, middleware, health, model information, control APIs, error translation, and Uvicorn.
- [`serving_chat.py`](python/sglang/srt/entrypoints/openai/serving_chat.py#L195) owns `/v1/chat/completions`, request preparation, stream generation, usage chunks, finish reasons, and OpenAI response assembly.
- [`protocol.py`](python/sglang/srt/entrypoints/openai/protocol.py) defines Python/msgspec/Pydantic-style request and response schemas.
- [`serving_base.py`](python/sglang/srt/entrypoints/openai/serving_base.py), `serving_completion.py`, usage processors, response handlers, and shared helpers provide the rest of the OpenAI behavior.
- Streaming uses Python `StreamingResponse`, async generators, ORJSON/msgspec conversion, and disconnect/abort handling.

### Native port unit

The C++ front end needs:

- HTTP/1.1 keep-alive and streaming;
- OpenAI-compatible JSON decoding and validation;
- `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/model_info`, health, abort, and required control endpoints;
- SSE framing and incremental usage;
- exact error/status/finish-reason behavior;
- disconnect propagation into the scheduler;
- request IDs, accounting, and cancellation.

There is an opt-in Rust server behind `SGLANG_RUST_SERVER`. The qualified launcher leaves it unset. Under the C++/CUDA-only rule, that code is useful as a behavioral oracle.

---

## 3. Chat rendering, reasoning, and tool-call parsing — **R/O/F**

The behavior gates specifically require these Python paths:

- [`template_manager.py`](python/sglang/srt/parser/template_manager.py) selects and loads the checkpoint chat template.
- [`conversation.py`](python/sglang/srt/conversation.py), [`jinja_template_utils.py`](python/sglang/srt/parser/jinja_template_utils.py), and associated template-detection modules render messages, tools, thinking controls, and assistant prefixes.
- [`serving_chat.py`](python/sglang/srt/entrypoints/openai/serving_chat.py#L2106) processes reasoning incrementally.
- [`reasoning_parser.py`](python/sglang/srt/parser/reasoning_parser.py#L355) contains the configured `Qwen3Detector` and its streaming state.
- [`qwen3_coder_detector.py`](python/sglang/srt/function_call/qwen3_coder_detector.py#L21) is a fully Python state machine using strings, regex, JSON, schema inspection, and `literal_eval`.
- [`function_call_parser.py`](python/sglang/srt/function_call/function_call_parser.py) dispatches tool parsers and manages parser lifecycle.

These paths execute on each streamed text delta when reasoning or tool parsing is engaged.

### Native port unit

C++ needs exact implementations of:

- checkpoint Jinja/chat-template semantics;
- thinking-mode and continuation behavior;
- reasoning-content separation across arbitrary chunk boundaries;
- Qwen3 Coder structural tag parsing;
- JSON-schema value conversion;
- multiple/partial tool calls and stable tool-call IDs;
- the established single `multiply({"a":37,"b":19})` result;
- final OpenAI finish-reason assembly.

---

## 4. Tokenization and incremental detokenization — **R/O**

Python owns the tokenizer process contract:

- [`tokenizer_manager.py`](python/sglang/srt/managers/tokenizer_manager.py#L375) validates requests, renders/tokenizes inputs, creates request state, performs ZMQ dispatch, manages async request streams, and coalesces responses.
- `_tokenize_texts` and `_tokenize_one_request` live at approximately lines 889 and 993.
- [`hf_transformers/tokenizer.py`](python/sglang/srt/utils/hf_transformers/tokenizer.py#L470) loads the tokenizer through Hugging Face `AutoTokenizer`.
- The checkpoint declares `backend: "tokenizers"` and supplies `tokenizer.json`, so the primitive BPE implementation is native Rust underneath Python. Python still owns every call and all surrounding semantics.
- [`detokenizer_manager.py`](python/sglang/srt/managers/detokenizer_manager.py#L95) owns incremental decode state, batched `decode`/`batch_decode`, Unicode recovery, stop trimming, offsets, and message construction.
- [`io_struct.py`](python/sglang/srt/managers/io_struct.py) defines messages and serialization.
- [`utils/network.py`](python/sglang/srt/utils/network.py) and the managers own ZMQ IPC.

### Native port unit

C++ needs:

- tokenizer JSON/vocab/merge loading;
- byte-level BPE encode/decode;
- special-token policy;
- chat-template integration;
- incremental UTF-8-safe decode with stable offsets;
- stop-token/string trimming;
- request/response state and native IPC or in-process queues.

The native engine can call an existing tokenizer library through a stable native ABI, or own the Qwen tokenizer directly.

---

## 5. Scheduler and request state — **C/P/R**

Python still is the scheduler:

- [`scheduler.py`](python/sglang/srt/managers/scheduler.py#L417) owns global engine state.
- [`event_loop_overlap`](python/sglang/srt/managers/scheduler.py#L1814) runs every batch/cycle under the launcher defaults.
- [`get_next_batch_to_run`](python/sglang/srt/managers/scheduler.py#L3075), `get_new_batch_prefill`, `update_running_batch`, `run_batch`, and `process_batch_result` execute continuously.
- [`schedule_batch.py`](python/sglang/srt/managers/schedule_batch.py#L803) owns `Req`, stop conditions, token histories, cache ownership, finish state, sampling parameters, and batch mutation.
- [`schedule_policy.py`](python/sglang/srt/managers/schedule_policy.py) and [`prefill_adder.py`](python/sglang/srt/managers/prefill_adder.py) own priority, prefix matching, capacity estimates, chunking, and admission.
- [`overlap_utils.py`](python/sglang/srt/managers/overlap_utils.py) owns future/relay buffers and cross-cycle lifetimes.
- [`scheduler_components/request_receiver.py`](python/sglang/srt/managers/scheduler_components/request_receiver.py) polls and decodes incoming messages.
- [`scheduler_components/batch_result_processor.py`](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L826) synchronizes D2H completion, converts tensors to Python lists, commits accepted tokens, advances grammar state, checks stops, updates reasoning counters, and frees cache state.
- [`scheduler_components/output_streamer.py`](python/sglang/srt/managers/scheduler_components/output_streamer.py#L104) determines emission and constructs detokenizer payloads.
- Metrics, load reporting, health signaling, timers, cache events, and invariant checking remain Python scheduler components.

Even with one running request, the Python scheduler still executes the full planning and result machinery every speculative cycle.

### Native port unit

C++ needs native equivalents for:

- `Req`, `ScheduleBatch`, and finish-state state machines;
- prefill/decode admission;
- exact capacity accounting and retraction;
- overlap scheduling and cross-stream lifetimes;
- token/sequence-length progression;
- stop token/string/regex semantics;
- accepted-token commit;
- asynchronous result-copy management;
- streaming cadence;
- metrics and health.

---

## 6. Device-to-host result path — **C**

Every generation cycle produces a host-visible result through Python/PyTorch:

- [`managers/utils.py`](python/sglang/srt/managers/utils.py#L27) implements `_async_d2h`.
- `GenerationBatchResult.copy_to_cpu` allocates pinned PyTorch tensors, launches copies, records streams, and records a PyTorch CUDA event.
- The scheduler synchronizes `copy_done`.
- `_resolve_spec_v2_tokens` converts `next_token_ids` and `accept_lens` to Python lists.
- Python then updates `Req.output_ids`, acceptance histograms, committed lengths, penalties, stop state, and output payloads.

The overlap schedule hides some host time behind the next forward. It still requires Python every cycle.

### Native port unit

The native implementation needs:

- preallocated pinned result rings;
- `cudaMemcpyAsync` on a dedicated stream;
- native CUDA events;
- explicit source-buffer lifetime ownership;
- compact accepted-token/result structs;
- native request-state updates once the event completes.

---

## 7. Prefix cache, token pools, KV allocation, and Mamba state — **R/P/C**

The qualified hybrid model selects Python `UnifiedRadixCache`:

- [`mem_cache/registry.py`](python/sglang/srt/mem_cache/registry.py#L80) selects it for hybrid SSM.
- [`unified_radix_cache.py`](python/sglang/srt/mem_cache/unified_radix_cache.py#L133) owns prefix matching, insertion, caching finished/unfinished requests, eviction, and locks.
- [`unified_cache/unified_tree_core.py`](python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py) owns nodes, hashes, LRU lists, traversal, and component state.
- [`unified_cache/components/full_component.py`](python/sglang/srt/mem_cache/unified_cache/components/full_component.py) and [`mamba_component.py`](python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py) coordinate MHA and recurrent components.
- [`memory_pool.py`](python/sglang/srt/mem_cache/memory_pool.py) owns request-to-token maps, the 200K KV pools, Mamba slots, FP32 recurrent-state tensors, ReplaySSM rings, and `extra_buffer_lazy` ping-pong state.
- [`allocator/paged.py`](python/sglang/srt/mem_cache/allocator/paged.py), [`allocator/mamba.py`](python/sglang/srt/mem_cache/allocator/mamba.py), and [`allocation.py`](python/sglang/srt/mem_cache/allocation.py) own allocation/free policy and batch cache-location construction.
- PyTorch’s CUDA caching allocator supplies the actual tensor storage and lifetime machinery.

### Native port unit

C++ needs:

- radix nodes, hashes, LRU ownership, and locks;
- virtual-to-physical token/page mapping;
- page-64 allocation/free lists;
- target/draft KV descriptors and layout;
- Mamba slot and lazy checkpoint management;
- ReplaySSM raw rings and accepted-state commits;
- transactional allocation and exact 200K accounting;
- native GPU memory arenas with stream/event-aware lifetime rules.

---

## 8. Model construction and eager prefill — **S/P**

The target is still a Python `nn.Module` tree:

- [`models/qwen3_5.py`](python/sglang/srt/models/qwen3_5.py) defines the complete target architecture.
- [`Qwen3_5GatedDeltaNet`](python/sglang/srt/models/qwen3_5.py#L545) implements 48 linear-attention layers.
- [`Qwen3_5AttentionDecoderLayer`](python/sglang/srt/models/qwen3_5.py#L1300) implements 16 full-attention layers.
- [`Qwen3_5ForCausalLM`](python/sglang/srt/models/qwen3_5.py#L1728) owns the model.
- Its forward path iterates the 64 layers in a Python `for` loop for every eager prefill chunk.
- [`models/qwen3_5_mtp.py`](python/sglang/srt/models/qwen3_5_mtp.py#L96) defines the one-layer MTP model, including BF16 `nn.Linear` fusion, norms, embedding/head sharing, and logits processing.
- [`layers/linear.py`](python/sglang/srt/layers/linear.py), [`layernorm.py`](python/sglang/srt/layers/layernorm.py), [`activation.py`](python/sglang/srt/layers/activation.py), [`radix_attention.py`](python/sglang/srt/layers/radix_attention.py), and layer communicators dispatch every suboperation.
- [`forward_batch_info.py`](python/sglang/srt/model_executor/forward_batch_info.py#L739) constructs the Python model’s runtime view.

The launcher explicitly disables prefill CUDA graphs. Consequently:

- each 4096-token target chunk traverses the 64-layer Python loop;
- each attention/MLP/norm call passes through Python/PyTorch dispatch;
- the draft-prefill extension traverses its one-layer MTP model in Python;
- first-token/root-q sampling returns to Python.

The GPU arithmetic is largely native or compiled. The launch graph and tensor orchestration remain Python.

### Native port unit

C++ needs an explicit Qwen3.8 model plan containing:

- embedding and LM head;
- 48 GDN blocks;
- 16 MHA blocks;
- residual/norm rules;
- ModelOpt NVFP4 linear descriptors;
- BF16 MTP fusion and one full-attention MTP block;
- target-to-draft hidden-state handoff;
- eager/graph execution plans for 4096 and 7680 chunks;
- logits pruning and final-row selection.

---

## 9. PyTorch model runner, tensors, streams, and CUDA graphs — **S/P/C**

These files make PyTorch the engine substrate:

- [`tp_worker.py`](python/sglang/srt/managers/tp_worker.py#L575) dispatches generation.
- [`model_runner.py`](python/sglang/srt/model_executor/model_runner.py#L1522) owns forward execution and sampling.
- [`eager_runner.py`](python/sglang/srt/model_executor/runner/eager_runner.py) owns eager prefill.
- [`decode_cuda_graph_runner.py`](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py) prepares/captures/replays decode graphs.
- [`full_cuda_graph_backend.py`](python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py#L47) stores one `torch.cuda.CUDAGraph` per shape and calls `.replay()` from Python.
- [`cuda_graph_buffer_registry.py`](python/sglang/srt/model_executor/cuda_graph_buffer_registry.py) owns graph-stable tensor buffers and aliases.
- [`forward_context.py`](python/sglang/srt/model_executor/forward_context.py) publishes Python-global dispatch context.
- [`fused_op.py`](python/sglang/kernels/fused_op.py) selects platform/compile/native methods.
- [`torch_compile_decoration.py`](python/sglang/srt/compilation/torch_compile_decoration.py#L51) wraps target `model.forward` in `torch.compile`.
- The rest of [`srt/compilation`](python/sglang/srt/compilation) reaches into Dynamo and Inductor internals.

PyTorch currently owns:

- `torch.Tensor` shape/stride/storage semantics;
- parameters and module traversal;
- ATen dispatch;
- CUDA streams and events;
- the caching allocator;
- CUDA graph pools and replay;
- random generator state/offsets;
- custom-op registration;
- Dynamo/Inductor tracing and generated Triton/CUDA kernels.

### Native port unit

A Python-free C++ implementation can initially use LibTorch. A PyTorch-free implementation requires:

- a native tensor/view descriptor;
- explicit dtype/layout/stride contracts;
- a stream-aware CUDA allocator;
- CUDA stream/event wrappers;
- direct `cudaGraph*` capture/instantiate/launch;
- explicit operator dispatch;
- native RNG and graph-replay offset management;
- cuBLAS/cuDNN/CUTLASS/FlashInfer/TRT-LLM C++ entry points;
- replacement kernels for every ATen/Inductor expression.

This substrate decision determines the actual “stops being PyTorch” point.

---

## 10. Attention backend control and metadata — **P/C**

Core attention compute is native; its ownership and metadata remain Python:

- [`flashinfer_backend.py`](python/sglang/srt/layers/attention/flashinfer_backend.py) constructs FlashInfer wrappers, indices, plans, workspaces, and prefill calls.
- `fast_prefill_plan` is a Python implementation tailored to graph-stable Windows behavior.
- [`trtllm_mha_backend.py`](python/sglang/srt/layers/attention/trtllm_mha_backend.py) owns TRT-LLM MHA/XQA metadata and calls `flashinfer.decode.trtllm_batch_decode_with_kv_cache`.
- [`hybrid_attn_backend.py`](python/sglang/srt/layers/attention/hybrid_attn_backend.py) chooses the correct full-attention backend per mode/layer.
- [`hybrid_linear_attn_backend.py`](python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py) coordinates full and recurrent attention.
- [`linear/gdn_backend.py`](python/sglang/srt/layers/attention/linear/gdn_backend.py) owns GDN dispatch, convolution, prefill, verify, and state handling.
- [`trtllm_mha_graph_metadata.py`](python/sglang/kernels/ops/kvcache/trtllm_mha_graph_metadata.py) and [`trtllm_mha_page_table.py`](python/sglang/kernels/ops/kvcache/trtllm_mha_page_table.py) contain Python-authored Triton metadata/page-table kernels.

### Native port unit

C++ needs:

- mode/layer backend selection resolved at startup;
- native page-table and metadata construction;
- direct FlashInfer prefill plan/run;
- direct TRT-LLM MHA/XQA invocation;
- target/draft KV format management;
- graph-stable workspace and metadata buffers.

---

## 11. NVFP4 quantization, GEMM dispatch, and autotuning — **S/P/C**

SGLang-side Python:

- [`modelopt_quant.py`](python/sglang/srt/layers/quantization/modelopt_quant.py#L1614) defines `ModelOptFp4LinearMethod`.
- It constructs PyTorch parameters, pads and swizzles checkpoint data, quantizes activations, calls FP4 GEMM, slices output, and handles tuple handoff.
- [`fp4_utils.py`](python/sglang/srt/layers/quantization/fp4_utils.py) selects FP4 backends and wraps FlashInfer quantization as a Torch custom op.
- On SM100+, activation FP4 quantization explicitly selects FlashInfer’s `"cute-dsl"` backend.
- [`flashinfer_autotune.py`](python/sglang/srt/model_executor/runner/flashinfer_autotune.py) constructs cache paths, synthesizes forwards, manages tuning contexts, and promotes selected file-cache tactics.

The external FlashInfer checkout contributes active Python ownership:

- `C:\Users\Daniel\flashinfer-windows-0.6.17\flashinfer\gemm\gemm_base.py`
  - `mm_fp4`;
  - runner creation;
  - workspace lookup;
  - tactic selection and launch.
- `...\flashinfer\quantization\fp4_quantization.py`
  - activation-quantization dispatch and output allocation.
- `...\flashinfer\quantization\kernels\nvfp4_quantize.py`
  - Python CuTe DSL implementation/code generation.
- `...\flashinfer\autotuner\autotuner.py`
  - cache keys, JSON persistence, candidate profiling, selection, and process caches.
- `...\flashinfer\jit\*`
  - extension generation, compilation, and loading.

The CUTLASS FP4 GEMM itself is native CUDA. Python still chooses, allocates, and invokes it.

### Native port unit

C++ needs:

- ModelOpt checkpoint interpretation;
- packed weight and scale descriptors;
- weight padding/swizzle/interleave;
- native activation quantization;
- direct CUTLASS/FlashInfer FP4 GEMM calls;
- persistent workspace management;
- native tactic registry, profiling, cache validation, and serialization;
- fixed eager and graph execution tactics.

---

## 12. Speculative decoding controller — **C/P/S**

The production controller is Python:

- [`eagle_worker_v2.py`](python/sglang/srt/speculative/eagle_worker_v2.py#L1576) sequences target prefill or draft → verify → draft extend.
- [`eagle_worker_common.py`](python/sglang/srt/speculative/eagle_worker_common.py#L482) prepares target verification, launches it, samples, commits state, and constructs results.
- [`eagle_utils.py`](python/sglang/srt/speculative/eagle_utils.py#L1102) implements target probability construction and acceptance.
- [`spec_utils.py`](python/sglang/srt/speculative/spec_utils.py) implements proposal distributions, random sampling, cache-location helpers, and ReplaySSM commit selection.
- [`eagle_info.py`](python/sglang/srt/speculative/eagle_info.py) contains Python dataclasses for cross-phase state.
- [`eagle_draft_cuda_graph_runner.py`](python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py) owns the captured two-step draft graph.
- [`eagle_draft_extend_cuda_graph_runner.py`](python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py) owns the captured draft-extend graph.

### Remaining critical seams

#### Draft

`EagleDraftWorker.draft`:

1. builds `ForwardBatch` and attention metadata in Python;
2. copies inputs into graph-stable PyTorch buffers;
3. replays the captured graph from Python;
4. passes outputs to Python `build_eagle_verify_input`.

The two draft depths and their aligned q sampling are captured together. That part is already device-bundled.

#### Verification and acceptance

`run_eagle_verify`:

1. prepares cache locations and target attention metadata in Python;
2. replays the target graph;
3. constructs grammar masks when required;
4. calls Python `eagle_sample`;
5. applies presence/repetition penalties with PyTorch;
6. computes softmax with PyTorch;
7. calls FlashInfer top-k/top-p renormalization;
8. creates random coins using PyTorch;
9. launches the Python-authored Triton rejection kernel;
10. invokes ReplaySSM accepted-state commit;
11. fills bonus tokens and constructs `GenerationBatchResult`.

#### Draft extend and next root

`_draft_extend_for_decode`:

1. builds `EagleDraftExtendInput`;
2. calculates selection indices with PyTorch;
3. prepares the graph on a Python-managed plan stream;
4. replays the draft-extend graph;
5. gathers selected logits/hidden states;
6. calls `_sample_next_draft_proposal` outside the graph.

That final proposal path executes PyTorch softmax, FlashInfer renormalization, and `torch.multinomial` through Python every cycle.

### Native port unit

The native controller needs a fixed batch-one state machine:

```text
prepare draft
→ draft graph
→ chain metadata
→ target verify graph
→ p/q acceptance
→ ReplaySSM accepted-path commit
→ draft-extend graph
→ next-root q
→ publish compact host result
```

The entire device portion can become one native CUDA-graph executable or a small fixed sequence of native graph launches. The scheduler-facing result should contain only accepted token IDs, accepted length, next sequence length, and completion flags.

---

## 13. Sampling, penalties, logits, and RNG — **P/C**

Python/PyTorch owns:

- [`layers/logits_processor.py`](python/sglang/srt/layers/logits_processor.py#L285): last-token selection, LM-head invocation, output reshaping, hidden-state handling, and logprobs.
- [`layers/sampler.py`](python/sglang/srt/layers/sampler.py#L79): temperature scaling, softmax, backend dispatch, logprob calculation, and sampling.
- [`sampling_batch_info.py`](python/sglang/srt/sampling/sampling_batch_info.py#L29): temperatures, top-k/top-p, bias, grammar masks, and accumulated penalty tensors.
- [`sampling/penaltylib`](python/sglang/srt/sampling/penaltylib): presence, frequency, repetition, min-new-token, and orchestrator state.
- [`spec_utils.py`](python/sglang/srt/speculative/spec_utils.py#L337): exact draft q construction and sampling.
- [`eagle_utils.py`](python/sglang/srt/speculative/eagle_utils.py#L1102): target p construction, rejection coins, and acceptance invocation.

Presence penalty `1.5` makes the penalty state an active production path. PyTorch tensor scatter/update operations maintain that state each cycle.

### Native port unit

CUDA/C++ needs:

- persistent token-count/presence state;
- additive and multiplicative penalty application;
- temperature scaling;
- top-k 20 and top-p 0.95 normalization;
- categorical sampling;
- exact Leviathan p/q rejection;
- Philox-compatible or explicitly qualified RNG state;
- graph-replay-safe RNG advancement;
- logits/hidden selection and optional logprobs.

---

## 14. Structured-output grammar — **F/C/O**

The default grammar backend resolves to XGrammar, with Python coordination:

- [`constrained/grammar_manager.py`](python/sglang/srt/constrained/grammar_manager.py#L26) owns request queues, asynchronous compilation, cache lookup, and ready-state transitions.
- [`base_grammar_backend.py`](python/sglang/srt/constrained/base_grammar_backend.py) owns the Python backend abstraction and vocabulary-mask buffers.
- [`xgrammar_backend.py`](python/sglang/srt/constrained/xgrammar_backend.py) wraps XGrammar, advances its FSM, fills masks, and applies them to PyTorch logits.
- [`batch_result_processor.py`](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L725) advances grammar state over every accepted speculative run.
- [`eagle_worker_common.py`](python/sglang/srt/speculative/eagle_worker_common.py) builds speculative grammar masks around target verification.

Tool requests and response-format requests can activate this surface, so it belongs to the required server contract.

### Native port unit

C++ needs:

- grammar compilation/cache ownership;
- native XGrammar FFI or a C++ grammar implementation;
- FSM cloning and accepted-token advancement;
- speculative-chain mask generation;
- CUDA vocabulary-mask application;
- grammar termination and suffix trimming.

---

# Python-authored GPU code still in the active lane

These files are especially important because they produce GPU binaries today while retaining Python/Triton/CuTe/Inductor as the source and compilation environment.

## Full-attention path

- [`fused_qk_rmsnorm_rope_gate.py`](python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py): Triton fused Q/K Gemma norm, RoPE, and gate deinterleave.
- [`trtllm_mha_graph_metadata.py`](python/sglang/kernels/ops/kvcache/trtllm_mha_graph_metadata.py): Triton graph metadata update.
- [`trtllm_mha_page_table.py`](python/sglang/kernels/ops/kvcache/trtllm_mha_page_table.py): Triton page-table construction.

## GDN projection, convolution, and output

- [`triton_gdn_fused_proj.py`](python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py): QKVZBA packing and eligible QKV splitting.
- [`causal_conv1d_triton.py`](python/sglang/kernels/ops/mamba/causal_conv1d_triton.py): prefill and update convolution.
- [`fused_gdn_gating.py`](python/sglang/kernels/ops/attention/fla/fused_gdn_gating.py): GDN gate calculation.
- [`layernorm_gated.py`](python/sglang/kernels/ops/attention/fla/layernorm_gated.py): gated RMS normalization.

## GDN prefill FLA stack

- [`chunk.py`](python/sglang/kernels/ops/attention/fla/chunk.py): Python orchestration.
- [`cumsum.py`](python/sglang/kernels/ops/attention/fla/cumsum.py)
- [`chunk_fwd.py`](python/sglang/kernels/ops/attention/fla/chunk_fwd.py)
- [`wy_fast.py`](python/sglang/kernels/ops/attention/fla/wy_fast.py)
- [`chunk_delta_h.py`](python/sglang/kernels/ops/attention/fla/chunk_delta_h.py)
- [`chunk_o.py`](python/sglang/kernels/ops/attention/fla/chunk_o.py)
- [`l2norm.py`](python/sglang/kernels/ops/attention/fla/l2norm.py)
- [`index.py`](python/sglang/kernels/ops/attention/fla/index.py)
- [`op.py`](python/sglang/kernels/ops/attention/fla/op.py)

## ReplaySSM production verify/commit

The launcher’s `--enable-linear-replayssm-spec` selects fold-every-commit:

- [`fused_sigmoid_gating_recurrent.py`](python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py): Triton verify recurrent update and raw-ring write.
- [`gdn_replayssm_spec_fold.py`](python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_fold.py): Triton accepted-prefix fold into FP32 state.

`gdn_replayssm_spec_decode.py` remains a retained alternative path. The production fold path above is the current default.

## Speculative metadata, allocation, and acceptance

- [`reject_sampling.py`](python/sglang/kernels/ops/speculative/reject_sampling.py): active linear rejection sampler in Triton.
- [`cache_locs.py`](python/sglang/kernels/ops/speculative/cache_locs.py): several Triton kernels plus a `torch.compile` helper.
- [`eagle.py`](python/sglang/kernels/ops/speculative/eagle.py): Triton bonus-token and accepted-cache-location kernels.
- [`memory/allocator.py`](python/sglang/kernels/ops/memory/allocator.py): Triton extend/decode allocation.
- [`memory/common.py`](python/sglang/kernels/ops/memory/common.py): Triton request-to-token and last-location operations.
- [`mem_cache/allocation.py`](python/sglang/srt/mem_cache/allocation.py#L610): Triton request-pool assignment.

## Inductor-generated code

The target graph wraps the Python model in `torch.compile`. Any operation expressed as ordinary PyTorch may become an Inductor-generated kernel:

- residual and view operations;
- pure-Torch SiLU/multiply inside compiled M3;
- softmax and tensor transforms where graph capture includes them;
- indexing, concatenation, scatter, repeat-interleave, and arithmetic;
- fake/custom-op metadata used to build the compiled graph.

There is no durable C++/CUDA source file for many of these generated kernels. A PyTorch-free port needs explicit native implementations or library calls for each one.

## External Python CuTe DSL

FlashInfer’s active FP4 activation quantizer is Python CuTe DSL:

- `flashinfer\quantization\kernels\nvfp4_quantize.py`

A durable C++/CUDA port needs this as AOT CUDA/CUTLASS code or a native JIT whose compiler and launch controller live in C++.

---

# Compute islands that are already native

These reduce the kernel-writing portion of the job. Their Python wrappers, tensor ABI, JIT loading, and launch ownership remain port scope.

| Area | Existing native implementation |
|---|---|
| Target/draft MHA | FlashInfer TRT-LLM MHA/XQA CUDA |
| Prefill MHA | FlashInfer CUDA |
| FP4 GEMM | FlashInfer/CUTLASS CUDA |
| Speculative top-k/top-p normalization | FlashInfer CUDA |
| Top-k-one chain metadata | [`chain_metadata.cuh`](python/sglang/kernels/jit/csrc/speculative/chain_metadata.cuh) |
| Eager SwiGLU → NVFP4 | [`silu_and_mul_nvfp4.cuh`](python/sglang/kernels/jit/csrc/gemm/silu_and_mul_nvfp4.cuh) |
| Gemma/RMS normalization | C++/CUDA JIT bodies under [`kernels/jit`](python/sglang/kernels/jit) |
| Attention sigmoid gate | [`fused_sigmoid_mul.cuh`](python/sglang/kernels/jit/csrc/elementwise/fused_sigmoid_mul.cuh) |
| Optional exact tree verifier | C++/CUDA JIT path |
| PowerShell CUDA/MSVC environment | Existing Windows launch scripts |

The qualified full-attention gate hits the native C++/CUDA sigmoid-multiply implementation for its supported contiguous BF16 shape. [`elementwise.py`](python/sglang/kernels/ops/elementwise/elementwise.py#L413) still contains the Python dispatcher and Triton fallback.

The eager target prefill MLP already uses the native fused SwiGLU-to-NVFP4 producer. During compiled target graph construction, `SiluAndMul.forward_native` is a pure PyTorch expression traced by Inductor, so the compiled M3 version still belongs to the PyTorch-removal inventory.

---

# Startup-only Python that still counts toward “all Python”

A native steady-state loop can arrive before these pieces. A zero-Python executable also needs all of them ported.

## Checkpoint/configuration/loading

- [`configs/model_config.py`](python/sglang/srt/configs/model_config.py#L248)
- [`configs/qwen3_5.py`](python/sglang/srt/configs/qwen3_5.py)
- [`configs/load_config.py`](python/sglang/srt/configs/load_config.py)
- [`model_loader/loader.py`](python/sglang/srt/model_loader/loader.py#L379)
- [`model_loader/weight_utils.py`](python/sglang/srt/model_loader/weight_utils.py#L1108)
- [`model_loader/utils.py`](python/sglang/srt/model_loader/utils.py#L197)
- target and MTP `load_weights` implementations in `qwen3_5.py` and `qwen3_5_mtp.py`.

These use Transformers configuration classes, Python Safetensors bindings, PyTorch parameters, Python weight-name mapping, and ModelOpt post-load transforms.

Native replacements need:

- JSON config parsing;
- Safetensors header/data mapping;
- tensor-name-to-native-descriptor mapping;
- target/MTP sharing;
- NVFP4 scale and weight transforms;
- provenance/checksum validation;
- model and cache memory sizing.

## Kernel compilation and loading

- [`kernels/jit/utils/compile`](python/sglang/kernels/jit/utils/compile) owns Ninja generation, NVCC/MSVC invocation, cache keys, dependency scans, TVM-FFI module loading, and runtime registration.
- [`srt/utils/custom_op.py`](python/sglang/srt/utils/custom_op.py) registers Torch custom ops and fake implementations.
- FlashInfer’s Python JIT layer performs similar work externally.
- Triton compiles Python kernels.
- Dynamo/Inductor compiles Python model expressions.
- Python captures all three required CUDA graphs.

A native executable can use AOT binaries for the qualified checkpoint/shapes, or carry a C++ JIT/cache manager.

## Autotuning

SGLang and FlashInfer Python perform:

- shape synthesis;
- candidate enumeration;
- timing/profiling;
- cache lookup;
- metadata validation;
- JSON persistence;
- selected-tactic promotion;
- target/draft cache separation.

This needs a native tactic database and controlled CUDA-event measurement.

---

# Conditional Windows paths included in the eventual port

These are outside the ordinary default request shape while remaining part of retained Windows behavior or the production gates:

| Surface | Python ownership |
|---|---|
| Structured output and tool grammar | `constrained/*`, grammar barriers, vocabulary masks |
| Logprobs/top-logprobs | `logits_processor.py`, `logprob_processor.py`, batch result conversion |
| Stop strings and regex | `Req.update_finish_state`, tokenizer decode, Python regex |
| Request abort/pause/control | HTTP server, TokenizerManager, Scheduler |
| Reasoning/tool parsing | OpenAI stream processor and parser state machines |
| Device-resident speculative bridge | `eagle_worker_v2.py` and Python CUDA composite-graph construction |
| Adaptive depth | Python controller and host acceptance feedback |
| Tree/SWOR infrastructure | Python topology/controller/state code plus a mixture of native and Triton kernels |
| Diagnostic p/q and graph-gap capture | Python artifact assembly, file I/O, metrics |
| Selective AttnNVFP4/chunk-7680 profile | Same Python architecture with alternate checkpoint/chunk/tactic selection |

The selective long-context lane does not add a new runtime category. It exercises the same Python owners with a different model artifact, chunk size, and FlashInfer tactic cache.

---

# The practical cut lines

## Cut 1: Native speculative hot loop

C++ owns:

- graph buffers;
- draft/verify/accept/commit/draft-extend sequencing;
- p/q sampling and RNG;
- ReplaySSM state;
- compact result publication.

Python can still perform startup, scheduling, and HTTP at this stage.

This removes Python from the highest-frequency GPU seam while preserving SRT around it.

## Cut 2: Native model engine

C++ owns:

- scheduler;
- request/KV/Mamba pools;
- target and draft model plans;
- prefill and decode execution;
- CUDA graph capture/replay;
- sampling;
- output token state.

Python can still perform model loading and the API front end temporarily.

At this cut, the Windows lane is a new native SGLang engine.

## Cut 3: Python-free executable using LibTorch

C++ additionally owns:

- loading;
- tokenization;
- HTTP/SSE;
- parsing;
- lifecycle.

LibTorch/ATen supplies tensors, allocation, CUDA graphs, dispatch, and selected operations. This is a Python-free server and remains PyTorch-backed.

## Cut 4: PyTorch-free native runtime

The implementation replaces:

- `torch.Tensor`/ATen;
- the Torch dispatcher;
- `CUDACachingAllocator`;
- `torch.cuda.CUDAGraph`;
- PyTorch streams/events;
- Dynamo/Inductor;
- Torch custom ops;
- Torch RNG;
- Python FlashInfer wrappers.

C++/CUDA and native libraries own those responsibilities directly.

**This is the point where it stops being PyTorch.**

## Cut 5: SGLang identity

With the existing OpenAI surface, reasoning/tool semantics, benchmark contract, capacity guarantees, and launcher integration preserved, the result remains a **native SGLang Windows backend**.

If those compatibility contracts are eventually removed, it becomes a standalone Qwen3.8 inference server that happens to live in the SGLang checkout.

Your stated goal reaches **Cut 4**, with Cut 5 preserving the useful part of the SGLang identity.

---

# Sensible implementation order

The dependency order I would use is:

1. **Freeze the native ABI and tensor descriptors**
   - dtypes, shapes, strides, ownership, streams, errors, graph-stable storage.

2. **Port active Triton/CuTe/Inductor kernels**
   - full-attention preparation;
   - GDN prefill and ReplaySSM;
   - speculative acceptance/cache metadata;
   - allocation;
   - activation quantization;
   - compiled Torch expressions.

3. **Build the native Qwen execution plan**
   - 48 GDN + 16 MHA target;
   - one-layer MTP;
   - NVFP4 and BF16 linears;
   - logits/head.

4. **Build the native speculative executor**
   - draft → verify → acceptance/commit → extend → next q;
   - explicit graph capture;
   - graph-safe RNG.

5. **Port memory, radix cache, and scheduler**
   - exact 200K pools;
   - page-64 allocator;
   - `extra_buffer_lazy`;
   - overlap and D2H result rings.

6. **Port loading, JIT/AOT selection, and autotuning**
   - config/Safetensors/ModelOpt;
   - tactic cache;
   - graph capture and warmup.

7. **Port tokenization, templates, reasoning, tools, and grammar**

8. **Port HTTP/OpenAI/SSE and process lifecycle**

9. **Retire the Python runtime and PyTorch runtime dependencies**

10. **Run the complete qualification contract**
    - exact `199000+16`;
    - sampled reasoning and arithmetic;
    - exactly one parsed multiply call;
    - language-only `/model_info`;
    - all three graph phases;
    - standalone OpenCode2;
    - real-sampling and fixed-work benchmark windows.
