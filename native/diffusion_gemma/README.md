# Native-Windows DiffusionGemma trial

This directory contains a C++ executable and C++ runtime adapters for the pinned
`nvidia/diffusiongemma-26B-A4B-it-NVFP4` checkpoint. All authored launch,
loading, routing, and SGLang integration code is C++. The executable embeds the
installed CPython runtime to call the existing Transformers, Torch, FlashInfer,
and SGLang libraries. It adds no Python or PowerShell source files.

The checkpoint lives at
`C:\Users\Daniel\models\diffusiongemma-26B-A4B-it-NVFP4`, at revision
`ec4ff3df205028f4e81c954c2227f9312b3ec2ea`. Both weight shards and the tokenizer
were verified against the published SHA-256 values. Downloaded files remain
immutable. This trial uses the text-generation portion of the checkpoint.

## Build

The current build uses MSVC, CUDA 13.3, standalone CPython 3.13.14, and the
checkout's installed dependencies: Torch 2.13.0+cu130, Transformers 5.12.1,
and FlashInfer 0.6.17. The CMake configuration records the checkout and Python
base paths, so rebuild after relocating either directory. The native executable
loads the existing virtual environment's packages without changing that
environment's interpreter metadata.
Qualification used this checkout's pre-existing native-Windows source repairs;
those user-owned changes are preserved separately from this integration.

From PowerShell 7 in `C:\Users\Daniel\sglang`, initialize the existing compiler
environment and build with two jobs:

```powershell
& .\scripts\windows\initialize_cuda_build_env.ps1 -MaxJobs 2
cmake -S native/diffusion_gemma -B build/diffusiongemma -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=C:/Users/Daniel/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe -DPython3_INCLUDE_DIR=C:/Users/Daniel/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/include -DPython3_LIBRARY=C:/Users/Daniel/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/libs/python313.lib
cmake --build build/diffusiongemma --parallel 2
```

Keep the executable and `diffusiongemma_runtime.dll` together. The dependency
check does not load the model onto the GPU:

```powershell
.\build\diffusiongemma\bin\diffusiongemma.exe inspect
```

## Trial commands

Run one GPU workload at a time, following the repository's process-ownership
preflight. Use the existing compiler-environment initialization above in the
launcher terminal; FlashInfer can compile a missing kernel on first use.

```powershell
.\build\diffusiongemma\bin\diffusiongemma.exe verify-moe
.\build\diffusiongemma\bin\diffusiongemma.exe prompt "What is 37 times 19? Give a short answer."
.\build\diffusiongemma\bin\diffusiongemma.exe serve
```

`verify-moe` loads one expert layer and compares native routing and the combined
gate/up projection against independent per-expert NVFP4 matrix multiplications.
The `prompt` command loads the language model, applies its chat template, and
generates up to 256 tokens using the checkpoint's entropy-bound denoising
schedule. Each invocation releases its model when the process exits.

The serving adapter registers the model in the native process's SGLang registry
and uses SGLang's synchronous diffusion scheduler. Its configured address is
`http://127.0.0.1:30001/v1`, with model ID `diffusiongemma-26b-a4b-nvfp4`.
The trial admits one running request, a 2048-token prompt-plus-output limit,
and 1 through 1024 requested output tokens. It keeps Qwen's launcher and port
30000 unchanged. Keep this foreground terminal open, and use `Ctrl+C` to stop
the server before starting another GPU workload. Launch through this executable
to install the compiled adapters; stock `sglang.exe serve` does not register
them.

Submit chat requests to `http://127.0.0.1:30001/v1/chat/completions`. For a short
answer, use this JSON body with `Content-Type: application/json`:

```json
{
  "model": "diffusiongemma-26b-a4b-nvfp4",
  "messages": [{"role": "user", "content": "What is 37 times 19?"}],
  "max_tokens": 128,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Set `enable_thinking` to `true` and allow a larger output budget, such as 512,
when requesting reasoning. A small budget can be exhausted before the final
answer. Responses preserve `reasoning_content` and Gemma4 tool calls. SSE
delivery is available with `stream: true`.

`/health` performs real inference. Its blank-prompt probe took about 26 seconds
in one functional check; the native launcher sets the server's health deadline
to 90 seconds. Allow that deadline for a bounded readiness request.

## Implementation limits

The installed Transformers implementation owns encoder/decoder attention,
self-conditioning, and entropy-bound generation. The C++ loader preserves
shared encoder/decoder weights and replaces the expert modules with native
NVFP4 adapters. Expert weights remain packed; the adapter routes selected rows
through FlashInfer's CUDA NVFP4 matrix-multiplication kernels. The initial fused
MoE path failed compilation against the installed CUTLASS epilogue ABI and is
not part of this implementation.

This eager compatibility path prioritizes an interactive model trial. It has no
CUDA graph capture, prefix reuse, batching, speculative decoding, or throughput
qualification. Generation completes before SGLang receives output slices;
streamed HTTP delivery therefore does not provide incremental generation
latency. SGLang's KV pool serves scheduler accounting; Transformers owns the
actual attention cache.

The checkpoint's diffusion schedule determines sampling. Autoregressive
sampling overrides, seed overrides, beam search, custom sampling parameters,
constrained decoding, log probabilities, and ignoring EOS are rejected.
Unknown sampling keys also fail validation. Omit `temperature`, `top_p`, and similar knobs
when submitting a trial request. Image and audio inputs are outside this
text-only entry point.

## Functional evidence and observed latency

The native expert verifier passed heterogeneous routing and weighted reduction
with relative-L2 error 0.00167094 against independent NVFP4 expert GEMMs. The
full model returned the arithmetic answer 703. The API passed clean chat
rendering, SSE completion and usage, a 1089-token prompt with 320 output tokens
across scheduler chunks, and an automatic `multiply({"a":37,"b":19})` call
followed by the tool-result continuation. A thinking-enabled arithmetic reply
returned final answer 703 and 290 reasoning tokens out of 294 generated tokens,
with reasoning accounting preserved across output slices. Invalid sampling
overrides, unknown sampling keys, fractional/excessive output budgets, beam
search, and out-of-vocabulary inputs returned HTTP 400.

Individual warmed functional requests on the RTX 5090 took:

| Request | Prompt / output tokens | Client end-to-end time |
|---|---:|---:|
| Short arithmetic answer | 24 / 17 | 3.869 s |
| Exact `NATIVE WINDOWS READY` response | 21 / 10 | 2.240 s |
| Multi-chunk numbered response | 1089 / 320 | 8.205 s |
| Automatic multiply tool call | 79 / 21 | 2.417 s |
| Thinking-enabled arithmetic | 31 / 294 | 17.148 s |

These single samples measure client-visible behavior and include generation
and delivery. The 320-token sample corresponds to about 39 output tokens per
second end to end. No repeated benchmark or performance promotion was run.
The eager per-expert launch and routing overhead is a candidate for future
measurement and optimization. SGLang's prefill-throughput log includes
scheduler-only bookkeeping chunks and does not measure actual model prefill
speed in this adapter.

Exact commands, failures, runtime checks, and retained response artifacts are
recorded in [the experiment log](../../notes/experiment-log.md).
