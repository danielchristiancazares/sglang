#include <sglang/benchmark/ttft_benchmark.hpp>
#include <sglang/benchmark/sha256.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sglang::benchmark {
namespace {
using Object = JsonValue::object;

void validate(const TtftOptions &o) {
  if (o.base_url.empty() || o.model.empty() || o.label.empty() ||
      o.input_tokens < 1 || o.input_tokens > 199000 ||
      o.output_tokens < 1 || o.output_tokens > 200000 - o.input_tokens ||
      o.samples < 1 || o.samples > 100 ||
      !std::isfinite(o.timeout_seconds) || o.timeout_seconds <= 0 ||
      o.timeout_seconds > 3600) {
    throw std::invalid_argument("invalid TTFT options; see --help");
  }
  if (o.continuation_tokens != 0 &&
      (o.continuation_tokens <= o.input_tokens ||
       o.continuation_tokens > 200000 - o.output_tokens)) {
    throw std::invalid_argument(
        "continuation must be longer than the seed and fit the 200K pool");
  }
}

std::string base_url(std::string value) {
  while (!value.empty() && value.back() == '/') {
    value.pop_back();
  }
  if (value.ends_with("/v1")) {
    throw std::invalid_argument("--base-url is the server root, without /v1");
  }
  return value;
}

JsonValue get_json(HttpTransport &transport, const std::string &url,
                   double timeout) {
  HttpRequest request;
  request.url = url;
  request.connect_timeout = std::chrono::milliseconds(
      static_cast<std::int64_t>(std::ceil(timeout * 1000.0)));
  request.io_timeout = request.connect_timeout;
  request.max_body_bytes = 4U * 1024U * 1024U;
  auto response = transport.perform(request);
  if (!response.ok() || response.response.status_code != 200) {
    throw std::runtime_error("GET " + url + " failed: " +
                             response.error.message + " HTTP " +
                             std::to_string(response.response.status_code));
  }
  return JsonValue::parse(response.response.body);
}

std::vector<std::int64_t> tokenize(HttpTransport &transport,
                                    const TtftOptions &o,
                                    const CalibratedPrompt &prompt) {
  const auto response = request_json(
      transport, base_url(o.base_url) + "/v1/tokenize",
      Object{{"model", o.model}, {"messages", messages_for(prompt.content)},
             {"chat_template_kwargs", chat_template_kwargs(true)}},
      o.timeout_seconds);
  const auto &tokens = response.at("tokens").as_array();
  if (response.at("count").as_int() != prompt.token_count ||
      tokens.size() != static_cast<std::size_t>(prompt.token_count)) {
    throw std::runtime_error("tokenization changed after exact calibration");
  }
  std::vector<std::int64_t> ids;
  ids.reserve(tokens.size());
  for (const auto &token : tokens) {
    if (!token.is_int() || token.as_int() < 0) {
      throw std::runtime_error("token IDs must be nonnegative integers");
    }
    ids.push_back(token.as_int());
  }
  return ids;
}

std::int64_t common_prefix(const std::vector<std::int64_t> &a,
                           const std::vector<std::int64_t> &b) {
  const auto mismatch = std::mismatch(a.begin(), a.end(), b.begin(), b.end());
  return static_cast<std::int64_t>(mismatch.first - a.begin());
}

JsonValue sampled_controls() {
  return Object{{"temperature", 1.0}, {"top_p", 0.95}, {"top_k", 20},
                {"min_p", 0.0}, {"presence_penalty", 0.0},
                {"repetition_penalty", 1.0}, {"enable_thinking", true},
                {"ignore_eos", true}, {"seed", nullptr}};
}

// Keep configuration evidence small and do not print keys, auth, or full paths.
JsonValue server_profile(const JsonValue &server) {
  Object result;
  for (const auto key :
       {"version", "served_model_name", "context_length", "max_total_tokens",
        "max_running_requests", "chunked_prefill_size", "page_size",
        "max_mamba_cache_size", "mamba_ssm_dtype", "mamba_radix_cache_strategy",
        "kv_cache_dtype", "speculative_draft_kv_cache_dtype",
        "speculative_algorithm", "speculative_dspark_block_size",
        "speculative_draft_model_quantization", "attention_backend",
        "prefill_attention_backend", "decode_attention_backend",
        "sampling_backend", "fp4_gemm_backend", "fp8_gemm_backend",
        "disable_radix_cache", "enable_cache_report", "stream_interval",
        "scheduler_recv_interval", "disable_flashinfer_autotune",
        "flashinfer_autotune_skip_ops", "language_model_only",
        "reasoning_parser", "tool_call_parser"}) {
    if (const auto *value = server.find(key)) {
      result.emplace(key, *value);
    }
  }
  return result;
}

void check_server(const JsonValue &server, const TtftOptions &o) {
  const auto *startup = server.find("startup_time");
  if (startup == nullptr || !startup->is_object() || startup->size() == 0) {
    throw std::runtime_error("server startup identity is unavailable");
  }
  if (!server.at("enable_cache_report").as_bool()) {
    throw std::runtime_error("TTFT prefix probe requires --enable-cache-report");
  }
  if (server.at("disable_radix_cache").as_bool()) {
    throw std::runtime_error("prefix reuse cannot be measured with radix cache disabled");
  }
  if (server.at("served_model_name").as_string() != o.model ||
      server.at("max_running_requests").as_int() != 1) {
    throw std::runtime_error("expected selected model and one-request scheduler");
  }
  const auto needed = std::max(o.input_tokens, o.continuation_tokens) + o.output_tokens;
  if (server.at("context_length").as_int() < needed ||
      server.at("max_total_tokens").as_int() < needed) {
    throw std::runtime_error("request does not fit the resolved context/token pool");
  }
}

JsonValue model_identity(const JsonValue &model, const TtftOptions &o) {
  if (model.at("served_model_name").as_string() != o.model ||
      !model.at("is_generation").as_bool() ||
      model.at("has_image_understanding").as_bool() ||
      model.at("has_audio_understanding").as_bool() ||
      model.at("reasoning_parser").as_string() != "qwen3" ||
      model.at("tool_call_parser").as_string() != "qwen3_coder") {
    throw std::runtime_error("expected language-only Qwen reasoning/tool model");
  }
  Object result;
  for (const auto key :
       {"served_model_name", "is_generation", "weight_version", "load_format",
        "reasoning_parser", "tool_call_parser", "has_image_understanding",
        "has_audio_understanding", "model_type", "architectures",
        "preferred_sampling_params"}) {
    // Require the live identity surface; null weight_version is legitimate.
    result.emplace(key, model.at(key));
  }
  for (const auto key : {"model_path", "tokenizer_path"}) {
    const auto &value = model.at(key).as_string();
    if (value.empty()) throw std::runtime_error("empty live model/tokenizer path");
    result.emplace(std::string(key) + "_sha256", sha256_hex(value));
  }
  return result;
}
} // namespace

std::string_view ttft_help() noexcept {
  return R"(usage: bench_ttft [options]

Sequential SGLang TTFT/prefix-reuse probe (CPU-only, native JSONL receipts).
Uses official thinking controls: temperature 1, top-p .95, top-k 20,
min-p 0, presence penalty 0, repetition penalty 1; natural sampling, no seed.
Requires a dedicated, idle one-request server with --enable-cache-report.
Flushes its cache before warmup, between sequences, and after completion.

  --base-url URL            server root (default http://127.0.0.1:30000)
  --model ID                served alias (default qwen3.8-27b)
  --label TEXT              control/candidate identity (default control)
  --input-tokens N          exact seed prompt (default 6213; at most 199000)
  --output-tokens N         exact output budget (default 512)
  --samples N               consecutive sequences (default 5; range 1..100)
  --continuation-tokens N   optional longer exact prompt; zero disables
  --timeout SECONDS         per-request timeout (default 600; at most 3600)
  --help                    show this help

Each sequence: flush -> seed -> identical replay -> optional longer prompt.
The longer prompt is a shared-prefix workload, not a multi-turn tool chat.
Real token-prefix overlap is measured after chat templating. Shared tokens
can require recomputation at Mamba checkpoints; this is not itself a defect.
HTTP/SSE timing does not separate GPU stages. Do not compare these receipts
to cached or Python scoreboard samples without labeling the protocol.
)";
}

TtftOptions parse_ttft_arguments(std::span<const std::string_view> args) {
  TtftOptions options;
  std::set<std::string_view> seen;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const auto key = args[i];
    if (!seen.insert(key).second || i + 1 == args.size() ||
        args[i + 1].starts_with("--")) {
      throw std::invalid_argument("duplicate option or missing value: " + std::string(key));
    }
    const auto value = args[++i];
    const auto integer = [&]() {
      std::int64_t result{};
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("invalid integer for " + std::string(key));
      }
      return result;
    };
    if (key == "--base-url") options.base_url = value;
    else if (key == "--model") options.model = value;
    else if (key == "--label") options.label = value;
    else if (key == "--input-tokens") options.input_tokens = integer();
    else if (key == "--output-tokens") options.output_tokens = integer();
    else if (key == "--samples") options.samples = integer();
    else if (key == "--continuation-tokens") options.continuation_tokens = integer();
    else if (key == "--timeout") {
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(),
                                           options.timeout_seconds);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("invalid timeout");
      }
    } else {
      throw std::invalid_argument("unknown option: " + std::string(key));
    }
  }
  validate(options);
  options.base_url = base_url(options.base_url);
  return options;
}

void run_ttft_benchmark(HttpTransport &transport, const TtftOptions &o,
                         const std::function<void(const JsonValue &)> &emit,
                         BenchmarkNow now) {
  validate(o);
  if (!emit || !now) throw std::invalid_argument("TTFT callbacks must be set");
  const auto base = base_url(o.base_url);
  const auto server = get_json(transport, base + "/server_info", o.timeout_seconds);
  check_server(server, o);
  const auto identity = model_identity(
      get_json(transport, base + "/model_info", o.timeout_seconds), o);
  const auto calibrate = [&](std::int64_t target) {
    auto prompt = calibrate_prompt(transport, base, o.model, target,
                                     o.timeout_seconds, Backend::kSglang, true);
    if (prompt.token_count != target) {
      throw std::runtime_error("exact TTFT prompt calibration failed");
    }
    return prompt;
  };
  const auto seed = calibrate(o.input_tokens);
  const auto seed_ids = tokenize(transport, o, seed);
  std::optional<CalibratedPrompt> continuation;
  std::int64_t overlap = 0;
  if (o.continuation_tokens != 0) {
    continuation = calibrate(o.continuation_tokens);
    overlap = common_prefix(seed_ids, tokenize(transport, o, *continuation));
  }
  const auto timestamp_ms = [] {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
  };
  emit(Object{
      {"kind", "manifest"}, {"protocol", "native-ttft-prefix-v1"},
      {"timestamp_unix_ms", timestamp_ms()}, {"label", o.label},
      {"base_url", base}, {"model", o.model}, {"samples", o.samples},
      {"input_tokens", o.input_tokens}, {"output_tokens", o.output_tokens},
      {"continuation_tokens", o.continuation_tokens},
      {"continuation_common_prefix_tokens", overlap},
      {"seed_prompt_sha256", sha256_hex(seed.content)},
      {"continuation_prompt_sha256",
       continuation ? JsonValue(sha256_hex(continuation->content)) : JsonValue(nullptr)},
      {"sampling", sampled_controls()}, {"server_profile", server_profile(server)},
      {"model_identity", identity},
      {"startup_identity_sha256", sha256_hex(server.at("startup_time").dump())},
      {"ttft_boundary", "dispatch_to_post_json_first_nonempty_reasoning_or_content"},
      {"warmup_sequences", 1},
  });

  std::map<std::string, std::vector<double>> timings;
  const auto flush = [&] {
    flush_cache(transport, base, o.timeout_seconds, Backend::kSglang, 0);
  };
  const auto measure = [&](const CalibratedPrompt &prompt, std::string phase,
                            std::int64_t sample, std::int64_t shared) {
    StreamRequestOptions request;
    request.base_url = base;
    request.model = o.model;
    request.content = prompt.content;
    request.output_tokens = o.output_tokens;
    request.timeout_seconds = o.timeout_seconds;
    request.temperature = 1.0;
    request.top_p = 0.95;
    request.top_k = 20;
    request.min_p = 0.0;
    request.presence_penalty = 0.0;
    request.repetition_penalty = 1.0;
    request.diagnostics = true;
    request.now = now;
    auto result = stream_request(transport, request);
    validate_result_counts(result, prompt.token_count, o.output_tokens, phase);
    if (!result.at("diagnostics").at("sse_done").as_bool() ||
        result.at("nonempty_delta_count").as_int() == 0) {
      throw std::runtime_error("incomplete TTFT output stream");
    }
    const auto &reported = result.at("diagnostics").at("cached_prompt_tokens_reported");
    // This server's usage serializer omits zero. check_server established that
    // reporting is enabled; preserve the original nullable value as evidence.
    const auto cached = reported.is_null() ? 0 : reported.as_int();
    if (phase == "cache_flushed" && cached != 0) {
      throw std::runtime_error("cache-flushed request reports prefix hits; server not isolated");
    }
    auto &record = result.as_object();
    record["kind"] = sample == 0 ? "warmup" : "sample";
    record["timestamp_unix_ms"] = timestamp_ms();
    record["label"] = o.label;
    record["sample"] = sample;
    record["phase"] = phase;
    record["common_prefix_tokens"] = shared;
    record["cached_prompt_tokens"] = cached;
    record["uncached_prompt_tokens"] = prompt.token_count - cached;
    record["shared_prefix_not_reported_cached"] =
        std::max<std::int64_t>(0, shared - cached);
    emit(result);
    if (sample != 0) timings[phase].push_back(result.at("ttft_s").as_double());
  };

  // Full-shape warmup includes retained-prefix paths; its receipts are excluded.
  // Each measured window then has exactly the same bounded cache sequence.
  for (std::int64_t sample = 0; sample <= o.samples; ++sample) {
    flush();
    measure(seed, "cache_flushed", sample, 0);
    measure(seed, "identical_replay", sample, seed.token_count);
    if (continuation) {
      measure(*continuation, "shared_prefix_continuation", sample, overlap);
    }
  }
  flush();
  // Detect configuration replacement during a run before publishing summaries.
  const auto final_server = get_json(transport, base + "/server_info", o.timeout_seconds);
  check_server(final_server, o);
  if (server_profile(final_server) != server_profile(server) ||
      final_server.at("startup_time") != server.at("startup_time") ||
      model_identity(get_json(transport, base + "/model_info", o.timeout_seconds), o) !=
          identity) {
    throw std::runtime_error("server changed during TTFT probe; no summary emitted");
  }
  for (const auto &[phase, values] : timings) {
    double sum = 0;
    JsonValue::array samples;
    for (const double value : values) {
      sum += value;
      samples.emplace_back(value);
    }
    emit(Object{{"kind", "summary"}, {"label", o.label}, {"phase", phase},
                {"samples", values.size()}, {"ttft_s_samples", std::move(samples)},
                {"ttft_s_mean", sum / static_cast<double>(values.size())},
                {"ttft_s_min", *std::min_element(values.begin(), values.end())},
                {"ttft_s_max", *std::max_element(values.begin(), values.end())}});
  }
}
} // namespace sglang::benchmark
