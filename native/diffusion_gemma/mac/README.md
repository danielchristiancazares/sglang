# DiffusionGemma on Apple silicon

This setup runs the pinned 4-bit DiffusionGemma checkpoint through MLX-VLM's
OpenAI-compatible server on port 30001. It uses a separate dependency
environment and preserves the repository's Qwen and Windows serving paths.
The timing client is C++23; inference uses the installed MLX-VLM package.

## Install

The measured host is a 32-GiB M1 Max. From the repository root, use persistent
storage outside `~/.cache`; this Mac's cron job periodically removes that
directory. The model occupies approximately 16.6 GB.

```sh
diffusiongemma_root="$HOME/.local/share/sglang-diffusiongemma"
uv venv --python .venv/bin/python "$diffusiongemma_root/venv"
UV_CACHE_DIR="$diffusiongemma_root/install-cache" uv pip install \
  --python "$diffusiongemma_root/venv/bin/python" \
  -r native/diffusion_gemma/mac/requirements.lock
HF_HOME="$diffusiongemma_root/huggingface" HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
  "$diffusiongemma_root/venv/bin/hf" download \
  mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --revision a7a81407613811e8ba63af92ac0d852b809e191f \
  --local-dir "$diffusiongemma_root/model" --max-workers 1
HF_HOME="$diffusiongemma_root/huggingface" \
  "$diffusiongemma_root/venv/bin/hf" cache verify \
  mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --revision a7a81407613811e8ba63af92ac0d852b809e191f \
  --local-dir "$diffusiongemma_root/model" \
  --fail-on-missing-files --json
```

The lock pins MLX-VLM 0.7.1, MLX/MLX-Metal 0.32.2, Transformers 5.14.0,
and their dependencies. The qualified interpreter is CPython 3.11.15. Leave
the downloaded checkpoint unchanged.
The local-directory downloader also creates `.cache/huggingface` metadata
inside the model directory. Verification reports those as extra local files;
all 13 checkpoint files must pass their checksum checks.

## Serve

Run one model server at a time. Check existing listeners and process ownership
before starting. Keep this command in a foreground terminal; `Ctrl+C` stops
its server and releases the model.

```sh
diffusiongemma_root="$HOME/.local/share/sglang-diffusiongemma"
HF_HOME="$diffusiongemma_root/huggingface" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  "$diffusiongemma_root/venv/bin/python" -m mlx_vlm.server \
  --model "$diffusiongemma_root/model" \
  --host 127.0.0.1 --port 30001 --model-discovery served \
  --max-num-seqs 1 --max-tokens 1024 --prefill-step-size 512 \
  --vision-cache-size 0 --log-progress-interval 32
```

After startup reports completion, check readiness and the exact model ID:

```sh
curl --fail http://127.0.0.1:30001/health
curl --fail http://127.0.0.1:30001/v1/models
```

The model ID is the absolute local model directory. Health checks runtime
readiness; a chat completion verifies inference. The diffusion worker handles
requests sequentially and emits completed generation blocks through SSE.
Thinking can be selected with the request's `enable_thinking` field. This lane
has no automatic prefix cache enabled.

Use explicit 64-token blocks and the confidence-threshold sampler for the
interactive setting. These are per-request controls; the unconfigured server
uses its library defaults. From the same shell:

```sh
jq -n --arg model "$diffusiongemma_root/model" \
  --arg prompt 'Explain how a hash table handles collisions.' \
  '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:512,
    temperature:0,seed:42,enable_thinking:false,stream:true,
    stream_options:{include_usage:true},diffusion_sampler:"confidence-threshold",
    diffusion_min_canvas_length:64,diffusion_max_canvas_length:64}' |
  curl --fail --no-buffer http://127.0.0.1:30001/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @-
```

## Measure

Build from the repository root with a C++23-capable compiler:

```sh
clang++ -std=c++23 -O2 -Wall -Wextra -Wpedantic -Werror \
  -I benchmark/native/include benchmark/mac/bench_diffusiongemma.cpp \
  benchmark/native/src/http_client.cpp benchmark/native/src/json.cpp \
  benchmark/native/src/sse_parser.cpp benchmark/native/src/sha256.cpp \
  -o "$diffusiongemma_root/bench_diffusiongemma"
```

The client takes `MODEL PROMPT_FILE MAX_TOKENS CANVAS SAMPLER greedy|sampled
SAMPLES RESULT_JSONL`. It uses port 30001, thinking disabled, natural EOS,
and equal minimum/maximum canvas lengths. Greedy mode uses temperature zero
and seed 42. Sampled mode uses temperature one, preserving the checkpoint's
0.4–0.8 denoising schedule, and seeds 42, 43, ... across samples. Choose a new result
filename for each invocation; existing files are protected from replacement.

```sh
"$diffusiongemma_root/bench_diffusiongemma" "$diffusiongemma_root/model" \
  benchmark/mac/prompts/diffusiongemma_explanation.txt \
  512 64 confidence-threshold greedy 5 "$diffusiongemma_root/explanation-64.jsonl"
```

TTFT measures time to first visible text. End-to-end throughput divides server
output-token usage by the entire request duration. Usage includes generated
special tokens. The server's decode-rate field is also retained; its timing
boundary differs from the client measurement. Peak memory is the server
process's high-water measurement. Complete text and its SHA-256
digest accompany each sample. See the separate
[Mac benchmark contract](../../../notes/benchmark-contract.md#mac-diffusiongemma-interactive-workload)
for repeatability and quality gates.

The checkpoint advertises a 262144-token context. That metadata alone does
not establish usable capacity on this 32-GiB host. Smaller diffusion canvases
and alternate samplers can change output quality as well as latency.

## Measured behavior

On the M1 Max, the explicit confidence-threshold sampler (threshold 0.9) and
64-token canvas passed two independent five-request greedy windows:

| Workload | Output tok/s, end to end | Time to first visible text |
|---|---:|---:|
| 62-token prompt / 512-token response, first warm window | 39.88 mean | 1.40 s mean |
| Same workload after restart, warm window | 39.90 mean | 1.39 s mean |
| Sampled, temperature one, seeds 42–46, five requests | 38.44 mean; 34.72–43.15 range | 1.88 s mean |
| First request after that restart | 37.27 | 2.36 s |
| Short arithmetic, five requests | 8 generated tokens per reply | 0.36 s mean; 0.30–0.62 s range |
| Complete collision/resizing explanation, one request | 38.43 | 2.57 s |
| 3935-token retrieval prompt, one request | 11 generated tokens | 6.68 s |

Both fixed-work windows produced identical text and hit the 512-token budget.
The separate explanation completed naturally at 356 tokens, with correct
worked-example positions. Arithmetic, SSE accounting, parsed multiply calls,
tool-result continuation, and preserved reasoning with the correct final
answer pass. These checks establish local functionality; broader model
accuracy and large-context capacity remain unmeasured.

The selected profile's restarted process reached 17.02 GB of MLX peak memory
on the short-prompt workload. Prefill dominates the longer prompt's latency:
6.17 seconds of its 6.68-second initial wait. Prefix reuse is disabled in these
measurements. Model loading took 3.9–4.7 seconds, separately from request TTFT.

The 256-token entropy-bound control reached 35.16 tok/s with 6.67-second TTFT.
A 32-token entropy-bound candidate produced malformed text and 14.25 tok/s;
it is excluded from the selected configuration. Full commands, individual
samples, and the sampling-profile limits are recorded in the
[experiment log](../../../notes/experiment-log.md).
The sampled result is one five-request window; the independent restart
comparison covers greedy decoding. Broader sampled accuracy was not evaluated.
