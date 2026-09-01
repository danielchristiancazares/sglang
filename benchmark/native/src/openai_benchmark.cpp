#include "sglang/benchmark/openai_benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "sglang/benchmark/sha256.hpp"
#include "sglang/benchmark/sse_parser.hpp"

namespace sglang::benchmark {
namespace {

using JsonArray = JsonValue::array;
using JsonObject = JsonValue::object;

[[nodiscard]] std::string normalize_base_url(std::string_view value) {
  while (!value.empty() && value.back() == '/') {
    value.remove_suffix(1);
  }
  if (value.empty()) {
    throw std::invalid_argument("base URL must contain a scheme and host");
  }
  return std::string(value);
}

[[nodiscard]] std::chrono::milliseconds timeout_duration(double seconds) {
  if (!std::isfinite(seconds) || seconds <= 0.0) {
    throw std::invalid_argument("timeout must be a positive finite number");
  }
  constexpr double kMillisecondsPerSecond = 1000.0;
  const double milliseconds = std::ceil(seconds * kMillisecondsPerSecond);
  if (milliseconds >
      static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("timeout is outside the supported range");
  }
  return std::chrono::milliseconds(static_cast<std::int64_t>(milliseconds));
}

[[nodiscard]] std::string format_general(double value) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(6) << std::defaultfloat << value;
  return stream.str();
}

[[nodiscard]] std::string http_detail(std::string_view body) {
  constexpr std::size_t kMaximumDetailBytes = 4096;
  return std::string(body.substr(0, kMaximumDetailBytes));
}

[[nodiscard]] HttpResponse
perform_checked(HttpTransport &transport, const HttpRequest &request,
                const HttpBodyChunkCallback &on_body_chunk = {}) {
  HttpResult result = transport.perform(request, on_body_chunk);
  if (!result.ok()) {
    throw std::runtime_error(
        "HTTP transport " +
        std::string(http_error_code_name(result.error.code)) + " from " +
        request.url + ": " + result.error.message);
  }
  if (result.response.status_code < 200 || result.response.status_code >= 300) {
    throw std::runtime_error(
        "HTTP " + std::to_string(result.response.status_code) + " from " +
        request.url + ": " + http_detail(result.response.body));
  }
  return std::move(result.response);
}

[[nodiscard]] HttpRequest make_json_request(std::string_view url,
                                            const JsonValue &payload,
                                            double timeout_seconds,
                                            std::string_view accept) {
  HttpRequest request;
  request.method = "POST";
  request.url = std::string(url);
  request.headers = {
      HttpHeader{"Accept", std::string(accept)},
      HttpHeader{"Content-Type", "application/json"},
      HttpHeader{"User-Agent", "sglang-local-benchmark/1"},
  };
  request.body = payload.dump();
  request.connect_timeout = timeout_duration(timeout_seconds);
  request.io_timeout = request.connect_timeout;
  return request;
}

[[nodiscard]] const JsonValue &require_member(const JsonValue &object,
                                              std::string_view key,
                                              std::string_view context) {
  if (!object.is_object()) {
    throw std::runtime_error(std::string(context) + " must be a JSON object");
  }
  const JsonValue *value = object.find(key);
  if (value == nullptr) {
    throw std::runtime_error(std::string(context) + " is missing " +
                             std::string(key));
  }
  return *value;
}

[[nodiscard]] std::int64_t require_integer(const JsonValue &object,
                                           std::string_view key,
                                           std::string_view context) {
  const JsonValue &value = require_member(object, key, context);
  if (!value.is_int()) {
    throw std::runtime_error(std::string(context) + "." + std::string(key) +
                             " must be an integer");
  }
  return value.as_int();
}

[[nodiscard]] double require_finite_number(const JsonValue &object,
                                           std::string_view key,
                                           std::string_view context) {
  const JsonValue &value = require_member(object, key, context);
  if (!value.is_number()) {
    throw std::runtime_error(std::string(context) + "." + std::string(key) +
                             " must be a number");
  }
  const double number = value.as_double();
  if (!std::isfinite(number)) {
    throw std::runtime_error(std::string(context) + "." + std::string(key) +
                             " must be finite");
  }
  return number;
}

[[nodiscard]] std::int64_t optional_usage_integer(const JsonValue &usage,
                                                  std::string_view key) {
  const JsonValue *value = usage.find(key);
  if (value == nullptr || value->is_null()) {
    return 0;
  }
  if (!value->is_int()) {
    throw std::runtime_error("stream usage." + std::string(key) +
                             " must be an integer");
  }
  return value->as_int();
}

[[nodiscard]] std::int64_t checked_code_point_count(std::string_view text) {
  const std::size_t count = utf8_code_point_count(text);
  if (count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("Unicode character count exceeds int64 range");
  }
  return static_cast<std::int64_t>(count);
}

void checked_add(std::int64_t amount, std::int64_t &value,
                 std::string_view label) {
  if (amount < 0 || value > std::numeric_limits<std::int64_t>::max() - amount) {
    throw std::overflow_error(std::string(label) + " exceeds int64 range");
  }
  value += amount;
}

[[nodiscard]] double seconds_between(HttpTimePoint start, HttpTimePoint end) {
  return std::chrono::duration<double>(end - start).count();
}

[[nodiscard]] double round_decimal(double value, int digits) {
  double scale = 1.0;
  for (int index = 0; index < digits; ++index) {
    scale *= 10.0;
  }
  return std::nearbyint(value * scale) / scale;
}

[[nodiscard]] std::string repeated(std::string_view unit, std::int64_t count) {
  if (count < 0) {
    throw std::invalid_argument("repeat count must be non-negative");
  }
  const auto repeats = static_cast<std::uint64_t>(count);
  if (!unit.empty() &&
      repeats > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max() / unit.size())) {
    throw std::length_error("calibration prompt exceeds addressable memory");
  }
  std::string result;
  result.reserve(static_cast<std::size_t>(repeats) * unit.size());
  for (std::uint64_t index = 0; index < repeats; ++index) {
    result.append(unit);
  }
  return result;
}

[[nodiscard]] std::string current_timestamp_utc() {
  using SystemClock = std::chrono::system_clock;
  const auto now = SystemClock::now();
  const auto whole_seconds =
      std::chrono::floor<std::chrono::seconds>(now.time_since_epoch());
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                          now.time_since_epoch() - whole_seconds)
                          .count();
  const std::time_t epoch_seconds =
      SystemClock::to_time_t(SystemClock::time_point(whole_seconds));
  std::tm utc{};
#if defined(_WIN32)
  if (gmtime_s(&utc, &epoch_seconds) != 0) {
    throw std::runtime_error("failed to convert current time to UTC");
  }
#else
  if (gmtime_r(&epoch_seconds, &utc) == nullptr) {
    throw std::runtime_error("failed to convert current time to UTC");
  }
#endif
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(6)
         << std::setfill('0') << micros << "+00:00";
  return stream.str();
}

void validate_common_arguments(std::string_view base_url,
                               std::string_view model,
                               std::int64_t input_tokens,
                               std::int64_t output_tokens,
                               std::int64_t warmup_output_tokens,
                               double timeout_seconds) {
  static_cast<void>(normalize_base_url(base_url));
  if (model.empty()) {
    throw std::invalid_argument("model must be nonempty");
  }
  if (input_tokens < 1 || output_tokens < 1) {
    throw std::invalid_argument(
        "input and output token counts must be positive");
  }
  if (warmup_output_tokens < 1) {
    throw std::invalid_argument("warmup output token count must be positive");
  }
  static_cast<void>(timeout_duration(timeout_seconds));
}

void validate_temperature(double temperature) {
  if (!std::isfinite(temperature) || temperature < 0.0) {
    throw std::invalid_argument(
        "temperature must be a finite non-negative number");
  }
}

void validate_optional_finite(const std::optional<double> &value,
                              std::string_view name) {
  if (value.has_value() && !std::isfinite(*value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

} // namespace

JsonValue chat_template_kwargs(bool enable_thinking) {
  return JsonValue(JsonObject{
      {"enable_thinking", JsonValue(enable_thinking)},
      {"preserve_thinking", JsonValue(enable_thinking)},
  });
}

JsonValue messages_for(std::string_view content) {
  return JsonValue(JsonArray{JsonValue(JsonObject{
      {"role", JsonValue("user")},
      {"content", JsonValue(content)},
  })});
}

JsonValue request_json(HttpTransport &transport, std::string_view url,
                       const JsonValue &payload, double timeout_seconds) {
  HttpRequest request =
      make_json_request(url, payload, timeout_seconds, "application/json");
  HttpResponse response = perform_checked(transport, request);
  try {
    return JsonValue::parse(response.body);
  } catch (const std::exception &error) {
    throw std::runtime_error("Invalid JSON from " + std::string(url) + ": " +
                             error.what());
  }
}

std::int64_t token_count(HttpTransport &transport, std::string_view base_url,
                         std::string_view model, std::string_view content,
                         double timeout_seconds, Backend backend,
                         bool enable_thinking) {
  const std::string base = normalize_base_url(base_url);
  if (model.empty()) {
    throw std::invalid_argument("model must be nonempty");
  }

  if (backend == Backend::kLlama) {
    const JsonValue templated = request_json(
        transport, base + "/apply-template",
        JsonValue(JsonObject{
            {"model", JsonValue(model)},
            {"messages", messages_for(content)},
            {"chat_template_kwargs", chat_template_kwargs(enable_thinking)},
        }),
        timeout_seconds);
    const JsonValue &prompt =
        require_member(templated, "prompt", "apply-template response");
    if (!prompt.is_string()) {
      throw std::runtime_error(
          "apply-template response.prompt must be a string");
    }

    const JsonValue tokenized =
        request_json(transport, base + "/tokenize",
                     JsonValue(JsonObject{
                         {"content", JsonValue(prompt.as_string())},
                         {"add_special", JsonValue(false)},
                         {"parse_special", JsonValue(true)},
                         {"with_pieces", JsonValue(false)},
                     }),
                     timeout_seconds);
    const JsonValue &tokens =
        require_member(tokenized, "tokens", "tokenize response");
    if (!tokens.is_array()) {
      throw std::runtime_error("tokenize response.tokens must be an array");
    }
    if (tokens.size() >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::overflow_error("token count exceeds int64 range");
    }
    return static_cast<std::int64_t>(tokens.size());
  }

  if (backend != Backend::kSglang) {
    throw std::invalid_argument("unknown tokenization backend");
  }
  const JsonValue tokenized = request_json(
      transport, base + "/v1/tokenize",
      JsonValue(JsonObject{
          {"model", JsonValue(model)},
          {"messages", messages_for(content)},
          {"chat_template_kwargs", chat_template_kwargs(enable_thinking)},
      }),
      timeout_seconds);
  const std::int64_t count =
      require_integer(tokenized, "count", "tokenize response");
  if (count < 0) {
    throw std::runtime_error("tokenize response.count must be non-negative");
  }
  return count;
}

CalibratedPrompt calibrate_prompt(HttpTransport &transport,
                                  std::string_view base_url,
                                  std::string_view model,
                                  std::int64_t target_tokens,
                                  double timeout_seconds, Backend backend,
                                  bool enable_thinking) {
  if (target_tokens < 0) {
    throw std::invalid_argument("target token count must be non-negative");
  }
  static_cast<void>(timeout_duration(timeout_seconds));

  std::int64_t low = 0;
  std::int64_t high = target_tokens;
  std::int64_t best_repeats = 0;
  std::int64_t best_count =
      token_count(transport, base_url, model, "", timeout_seconds, backend,
                  enable_thinking);
  if (best_count > target_tokens) {
    throw std::invalid_argument("Target " + std::to_string(target_tokens) +
                                " is below the empty templated prompt length " +
                                std::to_string(best_count));
  }

  while (low <= high) {
    const std::int64_t middle = low + ((high - low) / 2);
    const std::string candidate = repeated(kPromptUnit, middle);
    const std::int64_t count =
        token_count(transport, base_url, model, candidate, timeout_seconds,
                    backend, enable_thinking);
    if (count <= target_tokens) {
      best_repeats = middle;
      best_count = count;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }

  std::string content = repeated(kPromptUnit, best_repeats);
  const std::int64_t remaining = target_tokens - best_count;
  if (remaining <= 0) {
    return CalibratedPrompt{std::move(content), best_count};
  }

  low = 0;
  if (remaining > std::numeric_limits<std::int64_t>::max() / 2) {
    throw std::length_error("filler calibration range exceeds int64");
  }
  high = std::max<std::int64_t>(16, remaining * 2);
  std::int64_t best_filler = 0;
  while (low <= high) {
    const std::int64_t middle = low + ((high - low) / 2);
    std::string candidate = content;
    candidate.append(repeated(kFillerUnit, middle));
    const std::int64_t count =
        token_count(transport, base_url, model, candidate, timeout_seconds,
                    backend, enable_thinking);
    if (count <= target_tokens) {
      best_filler = middle;
      best_count = count;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  content.append(repeated(kFillerUnit, best_filler));
  return CalibratedPrompt{std::move(content), best_count};
}

void flush_cache(HttpTransport &transport, std::string_view base_url,
                 double timeout_seconds, Backend backend,
                 std::int64_t slot_id) {
  const std::string base = normalize_base_url(base_url);
  if (backend == Backend::kLlama) {
    if (slot_id < 0) {
      throw std::invalid_argument("slot ID must be non-negative");
    }
    static_cast<void>(request_json(
        transport, base + "/slots/" + std::to_string(slot_id) + "?action=erase",
        JsonValue(JsonObject{}), timeout_seconds));
    return;
  }
  if (backend != Backend::kSglang) {
    throw std::invalid_argument("unknown cache backend");
  }

  HttpRequest request;
  request.method = "GET";
  request.url =
      base + "/flush_cache?timeout=" + format_general(timeout_seconds);
  request.connect_timeout = timeout_duration(timeout_seconds);
  request.io_timeout = request.connect_timeout;
  static_cast<void>(perform_checked(transport, request));
}

bool StreamAccumulator::consume_data(std::string_view data) {
  const BenchmarkNow now = [] { return HttpClock::now(); };
  return consume_data_impl(data, now);
}

bool StreamAccumulator::consume_data_with_clock(std::string_view data,
                                                const BenchmarkNow &now) {
  if (!now) {
    throw std::invalid_argument("stream clock callback must be set");
  }
  return consume_data_impl(data, now);
}

bool StreamAccumulator::consume_data_at(std::string_view data,
                                        HttpTimePoint processed_at) {
  const BenchmarkNow now = [processed_at] { return processed_at; };
  return consume_data_impl(data, now);
}

bool StreamAccumulator::consume_data_impl(std::string_view data,
                                          const BenchmarkNow &now) {
  if (done_) {
    return true;
  }
  while (!data.empty() && (data.front() == ' ' || data.front() == '\t' ||
                           data.front() == '\r' || data.front() == '\n')) {
    data.remove_prefix(1);
  }
  while (!data.empty() && (data.back() == ' ' || data.back() == '\t' ||
                           data.back() == '\r' || data.back() == '\n')) {
    data.remove_suffix(1);
  }
  if (data == "[DONE]") {
    done_ = true;
    return true;
  }

  JsonValue event;
  try {
    event = JsonValue::parse(data);
  } catch (const std::exception &error) {
    throw std::runtime_error("invalid JSON in SSE data: " +
                             std::string(error.what()));
  }
  if (!event.is_object()) {
    throw std::runtime_error("SSE event must be a JSON object");
  }

  if (const JsonValue *usage = event.find("usage");
      usage != nullptr && usage->is_object()) {
    usage_ = *usage;
  }

  const JsonValue *choices = event.find("choices");
  if (choices == nullptr || choices->is_null()) {
    return false;
  }
  if (!choices->is_array()) {
    throw std::runtime_error("SSE event.choices must be an array");
  }
  if (choices->as_array().empty()) {
    return false;
  }
  const JsonValue &choice = choices->at(0);
  if (!choice.is_object()) {
    throw std::runtime_error("SSE event.choices[0] must be an object");
  }

  if (const JsonValue *finish_reason = choice.find("finish_reason");
      finish_reason != nullptr && !finish_reason->is_null()) {
    if (!finish_reason->is_string()) {
      throw std::runtime_error(
          "SSE event.choices[0].finish_reason must be a string or null");
    }
    finish_reason_ = finish_reason->as_string();
  }

  const JsonValue *delta = choice.find("delta");
  if (delta == nullptr || delta->is_null()) {
    return false;
  }
  if (!delta->is_object()) {
    throw std::runtime_error("SSE event.choices[0].delta must be an object");
  }

  std::int64_t delta_chars = 0;
  const JsonValue *reasoning = delta->find("reasoning_content");
  if (reasoning != nullptr && !reasoning->is_null()) {
    if (!reasoning->is_string()) {
      throw std::runtime_error(
          "SSE reasoning_content must be a string or null");
    }
    if (!reasoning->as_string().empty()) {
      const std::int64_t count =
          checked_code_point_count(reasoning->as_string());
      checked_add(count, reasoning_chars_, "reasoning character count");
      checked_add(count, delta_chars, "output delta character count");
      checked_add(1, reasoning_fragment_count_, "reasoning fragment count");
      reasoning_text_.append(reasoning->as_string());
      output_text_.append(reasoning->as_string());
    }
  }

  const JsonValue *content = delta->find("content");
  if (content != nullptr && !content->is_null()) {
    if (!content->is_string()) {
      throw std::runtime_error("SSE content must be a string or null");
    }
    if (!content->as_string().empty()) {
      const std::int64_t count = checked_code_point_count(content->as_string());
      checked_add(count, content_chars_, "content character count");
      checked_add(count, delta_chars, "output delta character count");
      checked_add(1, content_fragment_count_, "content fragment count");
      content_text_.append(content->as_string());
      output_text_.append(content->as_string());
    }
  }

  if (delta_chars > 0) {
    const HttpTimePoint processed_at = now();
    checked_add(1, nonempty_delta_count_, "nonempty delta count");
    if (!first_output_delta_chars_.has_value()) {
      first_output_delta_chars_ = delta_chars;
    }
    if (!first_output_at_.has_value()) {
      first_output_at_ = processed_at;
    }
    last_output_at_ = processed_at;
    max_output_delta_chars_ = std::max(max_output_delta_chars_, delta_chars);
  }
  return false;
}

JsonValue StreamAccumulator::finalize(HttpTimePoint started_at,
                                      HttpTimePoint ended_at) const {
  if (ended_at < started_at) {
    throw std::runtime_error("stream ended before it started");
  }
  const HttpTimePoint first_output = first_output_at_.value_or(ended_at);
  const HttpTimePoint last_output = last_output_at_.value_or(ended_at);
  if (first_output < started_at || first_output > ended_at ||
      last_output < first_output || last_output > ended_at) {
    throw std::runtime_error("stream event timestamps are out of order");
  }

  const std::int64_t prompt_tokens =
      optional_usage_integer(usage_, "prompt_tokens");
  const std::int64_t completion_tokens =
      optional_usage_integer(usage_, "completion_tokens");
  const std::int64_t total_tokens =
      optional_usage_integer(usage_, "total_tokens");
  const double ttft = seconds_between(started_at, first_output);
  const double elapsed = seconds_between(started_at, ended_at);
  const double decode_elapsed =
      std::max(0.0, seconds_between(first_output, ended_at));
  const std::int64_t decode_tokens =
      std::max<std::int64_t>(0, completion_tokens - 1);

  JsonObject result{
      {"prompt_tokens", JsonValue(prompt_tokens)},
      {"completion_tokens", JsonValue(completion_tokens)},
      {"total_tokens", JsonValue(total_tokens)},
      {"ttft_s", JsonValue(round_decimal(ttft, 6))},
      {"e2e_s", JsonValue(round_decimal(elapsed, 6))},
      {"observed_prompt_tps", JsonValue(nullptr)},
      {"decode_tps", JsonValue(nullptr)},
      {"output_tps_e2e", JsonValue(nullptr)},
      {"finish_reason", finish_reason_.has_value() ? JsonValue(*finish_reason_)
                                                   : JsonValue(nullptr)},
      {"output_chars", JsonValue(checked_code_point_count(output_text_))},
      {"reasoning_chars", JsonValue(reasoning_chars_)},
      {"content_chars", JsonValue(content_chars_)},
      {"nonempty_delta_count", JsonValue(nonempty_delta_count_)},
      {"reasoning_fragment_count", JsonValue(reasoning_fragment_count_)},
      {"content_fragment_count", JsonValue(content_fragment_count_)},
      {"first_output_delta_chars",
       JsonValue(first_output_delta_chars_.value_or(0))},
      {"max_output_delta_chars", JsonValue(max_output_delta_chars_)},
      {"trailing_after_last_delta_s",
       JsonValue(round_decimal(seconds_between(last_output, ended_at), 6))},
      {"output_sha256", JsonValue(sha256_hex(output_text_))},
      {"reasoning_sha256", JsonValue(sha256_hex(reasoning_text_))},
      {"content_sha256", JsonValue(sha256_hex(content_text_))},
  };
  if (ttft != 0.0) {
    result["observed_prompt_tps"] =
        JsonValue(round_decimal(static_cast<double>(prompt_tokens) / ttft, 3));
  }
  if (decode_tokens != 0 && decode_elapsed != 0.0) {
    result["decode_tps"] = JsonValue(
        round_decimal(static_cast<double>(decode_tokens) / decode_elapsed, 3));
  }
  if (completion_tokens != 0 && elapsed != 0.0) {
    result["output_tps_e2e"] = JsonValue(
        round_decimal(static_cast<double>(completion_tokens) / elapsed, 3));
  }
  return JsonValue(std::move(result));
}

JsonValue stream_request(HttpTransport &transport,
                         const StreamRequestOptions &options) {
  const std::string base = normalize_base_url(options.base_url);
  if (options.model.empty()) {
    throw std::invalid_argument("model must be nonempty");
  }
  if (options.output_tokens < 1) {
    throw std::invalid_argument("output token count must be positive");
  }
  static_cast<void>(timeout_duration(options.timeout_seconds));
  validate_temperature(options.temperature);
  validate_optional_finite(options.top_p, "top-p");
  validate_optional_finite(options.min_p, "min-p");
  validate_optional_finite(options.presence_penalty, "presence penalty");
  validate_optional_finite(options.repetition_penalty, "repetition penalty");
  if (options.top_p.has_value() &&
      (*options.top_p < 0.0 || *options.top_p > 1.0)) {
    throw std::invalid_argument("top-p must be between zero and one");
  }
  if (options.min_p.has_value() &&
      (*options.min_p < 0.0 || *options.min_p > 1.0)) {
    throw std::invalid_argument("min-p must be between zero and one");
  }
  if (options.top_k.has_value() && *options.top_k < 1) {
    throw std::invalid_argument("top-k must be positive");
  }
  if (options.repetition_penalty.has_value() &&
      *options.repetition_penalty <= 0.0) {
    throw std::invalid_argument("repetition penalty must be positive");
  }
  if (!options.now) {
    throw std::invalid_argument("stream clock callback must be set");
  }

  JsonObject payload{
      {"model", JsonValue(options.model)},
      {"messages", messages_for(options.content)},
      {"max_completion_tokens", JsonValue(options.output_tokens)},
      {"temperature", JsonValue(options.temperature)},
      {"stream", JsonValue(true)},
      {"stream_options",
       JsonValue(JsonObject{{"include_usage", JsonValue(true)}})},
      {"ignore_eos", JsonValue(true)},
      {"chat_template_kwargs", chat_template_kwargs(options.enable_thinking)},
  };
  if (options.top_p.has_value()) {
    payload.emplace("top_p", JsonValue(*options.top_p));
  }
  if (options.top_k.has_value()) {
    payload.emplace("top_k", JsonValue(*options.top_k));
  }
  if (options.min_p.has_value()) {
    payload.emplace("min_p", JsonValue(*options.min_p));
  }
  if (options.presence_penalty.has_value()) {
    payload.emplace("presence_penalty", JsonValue(*options.presence_penalty));
  }
  if (options.repetition_penalty.has_value()) {
    payload.emplace("repetition_penalty",
                    JsonValue(*options.repetition_penalty));
  }
  if (options.seed.has_value()) {
    payload.emplace("seed", JsonValue(*options.seed));
  }

  const std::string url = base + "/v1/chat/completions";
  HttpRequest request =
      make_json_request(url, JsonValue(std::move(payload)),
                        options.timeout_seconds, "text/event-stream");
  StreamAccumulator accumulator;
  SseParser parser;
  std::exception_ptr callback_error;

  const SseEventCallback on_event = [&](const SseEvent &event) {
    if (callback_error != nullptr) {
      return false;
    }
    try {
      if (event.kind == SseEventKind::kDone) {
        static_cast<void>(
            accumulator.consume_data_with_clock("[DONE]", options.now));
      } else {
        static_cast<void>(
            accumulator.consume_data_with_clock(event.data, options.now));
      }
      return true;
    } catch (...) {
      callback_error = std::current_exception();
      return false;
    }
  };

  const HttpBodyChunkCallback on_chunk = [&](std::string_view bytes,
                                             HttpTimePoint received_at) {
    if (callback_error != nullptr) {
      return false;
    }
    const SseParseStatus status = parser.feed(bytes, received_at, on_event);
    if (status == SseParseStatus::kError) {
      callback_error = std::make_exception_ptr(std::runtime_error(
          "SSE parse error: " + std::string(parser.error())));
      return false;
    }
    return status == SseParseStatus::kContinue;
  };

  const HttpResponse response = perform_checked(transport, request, on_chunk);
  if (callback_error != nullptr) {
    std::rethrow_exception(callback_error);
  }
  if (!parser.done()) {
    const SseParseStatus status =
        parser.finish(response.completed_at, on_event);
    if (callback_error != nullptr) {
      std::rethrow_exception(callback_error);
    }
    if (status == SseParseStatus::kError) {
      throw std::runtime_error("SSE parse error: " +
                               std::string(parser.error()));
    }
  }
  return accumulator.finalize(response.request_started_at,
                              response.completed_at);
}

void validate_result_counts(const JsonValue &result,
                            std::int64_t expected_prompt_tokens,
                            std::int64_t expected_completion_tokens,
                            std::string_view label) {
  if (expected_prompt_tokens < 0 || expected_completion_tokens < 0) {
    throw std::invalid_argument("expected token counts must be non-negative");
  }
  const std::int64_t prompt_tokens =
      require_integer(result, "prompt_tokens", label);
  const std::int64_t completion_tokens =
      require_integer(result, "completion_tokens", label);
  const std::int64_t total_tokens =
      require_integer(result, "total_tokens", label);
  if (prompt_tokens != expected_prompt_tokens) {
    throw std::runtime_error(std::string(label) +
                             " prompt token mismatch: expected=" +
                             std::to_string(expected_prompt_tokens) +
                             ", actual=" + std::to_string(prompt_tokens));
  }
  if (completion_tokens != expected_completion_tokens) {
    throw std::runtime_error(std::string(label) +
                             " completion token mismatch: expected=" +
                             std::to_string(expected_completion_tokens) +
                             ", actual=" + std::to_string(completion_tokens));
  }
  if (prompt_tokens >
      std::numeric_limits<std::int64_t>::max() - completion_tokens) {
    throw std::runtime_error(std::string(label) +
                             " token total exceeds int64 range");
  }
  if (total_tokens != prompt_tokens + completion_tokens) {
    throw std::runtime_error(
        std::string(label) +
        " total token mismatch: total=" + std::to_string(total_tokens) +
        ", prompt=" + std::to_string(prompt_tokens) +
        ", completion=" + std::to_string(completion_tokens));
  }
  const JsonValue &finish_reason =
      require_member(result, "finish_reason", label);
  if (!finish_reason.is_string() || finish_reason.as_string() != "length") {
    throw std::runtime_error(std::string(label) + " finish reason mismatch");
  }
}

JsonValue run_stream_benchmark(HttpTransport &transport,
                               const StreamOptions &options,
                               std::string timestamp_utc, BenchmarkNow now) {
  validate_common_arguments(options.base_url, options.model,
                            options.input_tokens, options.output_tokens,
                            options.warmup_output_tokens,
                            options.timeout_seconds);
  if (options.slot_id < 0) {
    throw std::invalid_argument("slot ID must be non-negative");
  }
  if (options.warmup_runs < 0) {
    throw std::invalid_argument("warmup run count must be non-negative");
  }
  validate_temperature(options.temperature);
  validate_optional_finite(options.top_p, "top-p");
  validate_optional_finite(options.min_p, "min-p");
  validate_optional_finite(options.presence_penalty, "presence penalty");
  validate_optional_finite(options.repetition_penalty, "repetition penalty");
  if (options.top_p.has_value() &&
      (*options.top_p < 0.0 || *options.top_p > 1.0)) {
    throw std::invalid_argument("top-p must be between zero and one");
  }
  if (options.min_p.has_value() &&
      (*options.min_p < 0.0 || *options.min_p > 1.0)) {
    throw std::invalid_argument("min-p must be between zero and one");
  }
  if (options.top_k.has_value() && *options.top_k < 1) {
    throw std::invalid_argument("top-k must be positive");
  }
  if (options.repetition_penalty.has_value() &&
      *options.repetition_penalty <= 0.0) {
    throw std::invalid_argument("repetition penalty must be positive");
  }
  if (!now) {
    throw std::invalid_argument("stream clock callback must be set");
  }

  const std::string base = normalize_base_url(options.base_url);
  const bool enable_thinking = !options.disable_thinking;
  CalibratedPrompt calibrated = calibrate_prompt(
      transport, base, options.model, options.input_tokens,
      options.timeout_seconds, options.backend, enable_thinking);
  if (calibrated.token_count != options.input_tokens) {
    throw std::runtime_error(
        "Prompt calibration did not reach the exact requested token count: "
        "requested=" +
        std::to_string(options.input_tokens) +
        ", calibrated=" + std::to_string(calibrated.token_count));
  }

  const auto request_options =
      [&](std::int64_t output_tokens) -> StreamRequestOptions {
    return StreamRequestOptions{
        base,
        options.model,
        calibrated.content,
        output_tokens,
        options.timeout_seconds,
        options.seed,
        options.temperature,
        options.top_p,
        options.top_k,
        options.min_p,
        options.presence_penalty,
        options.repetition_penalty,
        enable_thinking,
        now,
    };
  };

  flush_cache(transport, base, options.timeout_seconds, options.backend,
              options.slot_id);
  const std::int64_t warmup_runs =
      options.skip_warmup ? 0 : options.warmup_runs;
  for (std::int64_t index = 0; index < warmup_runs; ++index) {
    const JsonValue warmup = stream_request(
        transport, request_options(options.warmup_output_tokens));
    validate_result_counts(warmup, options.input_tokens,
                           options.warmup_output_tokens,
                           "warmup " + std::to_string(index + 1));
    if (index + 1 < warmup_runs) {
      flush_cache(transport, base, options.timeout_seconds, options.backend,
                  options.slot_id);
    }
  }
  flush_cache(transport, base, options.timeout_seconds, options.backend,
              options.slot_id);
  JsonValue result =
      stream_request(transport, request_options(options.output_tokens));
  validate_result_counts(result, options.input_tokens, options.output_tokens,
                         "measurement");

  JsonObject &object = result.as_object();
  if (timestamp_utc.empty()) {
    timestamp_utc = current_timestamp_utc();
  }
  object["timestamp"] = JsonValue(std::move(timestamp_utc));
  object["base_url"] = JsonValue(base);
  object["model"] = JsonValue(options.model);
  object["backend"] = JsonValue(backend_name(options.backend));
  object["requested_prompt_tokens"] = JsonValue(options.input_tokens);
  object["calibrated_prompt_tokens"] = JsonValue(calibrated.token_count);
  object["requested_completion_tokens"] = JsonValue(options.output_tokens);
  object["warmup"] = JsonValue(warmup_runs > 0);
  object["warmup_runs"] = JsonValue(warmup_runs);
  object["seed"] =
      options.seed.has_value() ? JsonValue(*options.seed) : JsonValue(nullptr);
  object["temperature"] = JsonValue(options.temperature);
  object["top_p"] = options.top_p.has_value() ? JsonValue(*options.top_p)
                                              : JsonValue(nullptr);
  object["top_k"] = options.top_k.has_value() ? JsonValue(*options.top_k)
                                              : JsonValue(nullptr);
  object["min_p"] = options.min_p.has_value() ? JsonValue(*options.min_p)
                                              : JsonValue(nullptr);
  object["presence_penalty"] = options.presence_penalty.has_value()
                                   ? JsonValue(*options.presence_penalty)
                                   : JsonValue(nullptr);
  object["repetition_penalty"] = options.repetition_penalty.has_value()
                                     ? JsonValue(*options.repetition_penalty)
                                     : JsonValue(nullptr);
  object["enable_thinking"] = JsonValue(enable_thinking);
  return result;
}

namespace {

struct GenerateResultView final {
  const JsonValue *meta{nullptr};
  const std::string *text{nullptr};
  std::int64_t prompt_tokens{0};
  std::int64_t completion_tokens{0};
};

[[nodiscard]] JsonArray
tokenize_input_ids(HttpTransport &transport, std::string_view base_url,
                   std::string_view model, std::string_view content,
                   double timeout_seconds, bool enable_thinking,
                   std::int64_t expected_tokens) {
  const JsonValue tokenized = request_json(
      transport, std::string(base_url) + "/v1/tokenize",
      JsonValue(JsonObject{
          {"model", JsonValue(model)},
          {"messages", messages_for(content)},
          {"chat_template_kwargs", chat_template_kwargs(enable_thinking)},
      }),
      timeout_seconds);
  const JsonValue &tokens =
      require_member(tokenized, "tokens", "tokenize response");
  if (!tokens.is_array()) {
    throw std::runtime_error("tokenize response.tokens must be an array");
  }
  if (tokens.size() != static_cast<std::size_t>(expected_tokens)) {
    throw std::runtime_error(
        "tokenization changed: calibrated=" + std::to_string(expected_tokens) +
        ", actual=" + std::to_string(tokens.size()));
  }
  JsonArray input_ids = tokens.as_array();
  for (std::size_t index = 0; index < input_ids.size(); ++index) {
    if (!input_ids[index].is_int() || input_ids[index].as_int() < 0) {
      throw std::runtime_error("tokenize response.tokens[" +
                               std::to_string(index) +
                               "] must be a non-negative integer");
    }
  }
  if (const JsonValue *count = tokenized.find("count"); count != nullptr) {
    if (!count->is_int() || count->as_int() != expected_tokens) {
      throw std::runtime_error(
          "tokenize response.count disagrees with the exact token array");
    }
  }
  return input_ids;
}

[[nodiscard]] JsonValue generate_acceptance(HttpTransport &transport,
                                            std::string_view base_url,
                                            const JsonArray &input_ids,
                                            std::int64_t output_tokens,
                                            const AcceptanceOptions &options) {
  return request_json(
      transport, std::string(base_url) + "/generate",
      JsonValue(JsonObject{
          {"input_ids", JsonValue(input_ids)},
          {"sampling_params",
           JsonValue(JsonObject{
               {"max_new_tokens", JsonValue(output_tokens)},
               {"temperature", JsonValue(options.temperature)},
               {"top_p", JsonValue(options.top_p)},
               {"top_k", JsonValue(options.top_k)},
               {"presence_penalty", JsonValue(options.presence_penalty)},
               {"ignore_eos", JsonValue(true)},
           })},
      }),
      options.timeout_seconds);
}

[[nodiscard]] GenerateResultView validate_generate_result(
    const JsonValue &response, std::int64_t expected_prompt_tokens,
    std::int64_t expected_completion_tokens, std::string_view label) {
  const JsonValue &meta = require_member(response, "meta_info", label);
  if (!meta.is_object()) {
    throw std::runtime_error(std::string(label) +
                             ".meta_info must be an object");
  }
  const std::int64_t prompt_tokens =
      require_integer(meta, "prompt_tokens", "generate meta_info");
  const std::int64_t completion_tokens =
      require_integer(meta, "completion_tokens", "generate meta_info");
  if (prompt_tokens != expected_prompt_tokens) {
    throw std::runtime_error(std::string(label) +
                             " prompt token mismatch: expected=" +
                             std::to_string(expected_prompt_tokens) +
                             ", actual=" + std::to_string(prompt_tokens));
  }
  if (completion_tokens != expected_completion_tokens) {
    throw std::runtime_error(std::string(label) +
                             " completion token mismatch: expected=" +
                             std::to_string(expected_completion_tokens) +
                             ", actual=" + std::to_string(completion_tokens));
  }

  const JsonValue &finish_reason =
      require_member(meta, "finish_reason", "generate meta_info");
  if (!finish_reason.is_object()) {
    throw std::runtime_error(std::string(label) +
                             " finish_reason must be an object");
  }
  const JsonValue &finish_type =
      require_member(finish_reason, "type", "generate finish_reason");
  if (!finish_type.is_string() || finish_type.as_string() != "length") {
    throw std::runtime_error(std::string(label) +
                             " finish reason type mismatch");
  }

  const JsonValue &text = require_member(response, "text", label);
  if (!text.is_string()) {
    throw std::runtime_error(std::string(label) + ".text must be a string");
  }
  return GenerateResultView{&meta, &text.as_string(), prompt_tokens,
                            completion_tokens};
}

[[nodiscard]] bool nearly_equal(double left, double right) {
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return std::abs(left - right) <=
         (16.0 * std::numeric_limits<double>::epsilon() * scale);
}

void validate_acceptance_metrics(const GenerateResultView &result) {
  const JsonValue &meta = *result.meta;
  const double e2e_latency =
      require_finite_number(meta, "e2e_latency", "generate meta_info");
  const double accept_rate =
      require_finite_number(meta, "spec_accept_rate", "generate meta_info");
  const double accept_length =
      require_finite_number(meta, "spec_accept_length", "generate meta_info");
  const std::int64_t correct =
      require_integer(meta, "spec_num_correct_drafts", "generate meta_info");
  const std::int64_t proposed =
      require_integer(meta, "spec_num_proposed_drafts", "generate meta_info");
  const std::int64_t verify_count =
      require_integer(meta, "spec_verify_ct", "generate meta_info");
  if (e2e_latency < 0.0) {
    throw std::runtime_error(
        "generate meta_info.e2e_latency must be non-negative");
  }
  if (accept_rate < 0.0 || accept_rate > 1.0) {
    throw std::runtime_error(
        "generate meta_info.spec_accept_rate must be between zero and one");
  }
  if (accept_length < 1.0) {
    throw std::runtime_error(
        "generate meta_info.spec_accept_length must be at least one");
  }
  if (correct < 0 || proposed <= 0 || verify_count <= 0 || correct > proposed) {
    throw std::runtime_error(
        "generate speculative counters are outside their valid ranges");
  }
  if (verify_count > result.completion_tokens) {
    throw std::runtime_error(
        "generate spec_verify_ct exceeds completion_tokens");
  }
  const double expected_rate =
      static_cast<double>(correct) / static_cast<double>(proposed);
  const double expected_length = static_cast<double>(result.completion_tokens) /
                                 static_cast<double>(verify_count);
  if (!nearly_equal(accept_rate, expected_rate)) {
    throw std::runtime_error(
        "generate spec_accept_rate disagrees with draft counters");
  }
  if (!nearly_equal(accept_length, expected_length)) {
    throw std::runtime_error(
        "generate spec_accept_length disagrees with completion/verify counts");
  }

  const JsonValue &histogram = require_member(
      meta, "spec_correct_drafts_histogram", "generate meta_info");
  if (!histogram.is_array() || histogram.as_array().empty()) {
    throw std::runtime_error(
        "generate spec_correct_drafts_histogram must be a nonempty array");
  }
  std::int64_t histogram_cycles = 0;
  std::int64_t histogram_correct = 0;
  for (std::size_t index = 0; index < histogram.as_array().size(); ++index) {
    const JsonValue &bucket = histogram.at(index);
    if (!bucket.is_int() || bucket.as_int() < 0) {
      throw std::runtime_error(
          "generate spec_correct_drafts_histogram buckets must be "
          "non-negative integers");
    }
    checked_add(bucket.as_int(), histogram_cycles,
                "acceptance histogram cycle count");
    if (bucket.as_int() != 0 &&
        index >
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() /
                                     bucket.as_int())) {
      throw std::overflow_error(
          "acceptance histogram draft count exceeds int64 range");
    }
    const std::int64_t weighted =
        static_cast<std::int64_t>(index) * bucket.as_int();
    checked_add(weighted, histogram_correct,
                "acceptance histogram draft count");
  }
  if (histogram_cycles != verify_count || histogram_correct != correct) {
    throw std::runtime_error(
        "generate acceptance histogram disagrees with speculative counters");
  }
}

} // namespace

JsonValue run_acceptance_benchmark(HttpTransport &transport,
                                   const AcceptanceOptions &options) {
  validate_common_arguments(options.base_url, options.model,
                            options.input_tokens, options.output_tokens,
                            options.warmup_output_tokens,
                            options.timeout_seconds);
  validate_temperature(options.temperature);
  if (!std::isfinite(options.top_p) || options.top_p < 0.0 ||
      options.top_p > 1.0) {
    throw std::invalid_argument("top-p must be between zero and one");
  }
  if (options.top_k < 1) {
    throw std::invalid_argument("top-k must be positive");
  }
  if (!std::isfinite(options.presence_penalty)) {
    throw std::invalid_argument("presence penalty must be finite");
  }

  const std::string base = normalize_base_url(options.base_url);
  const bool enable_thinking = !options.disable_thinking;
  const CalibratedPrompt calibrated = calibrate_prompt(
      transport, base, options.model, options.input_tokens,
      options.timeout_seconds, Backend::kSglang, enable_thinking);
  if (calibrated.token_count != options.input_tokens) {
    throw std::runtime_error(
        "Prompt calibration did not reach the exact requested token count: "
        "requested=" +
        std::to_string(options.input_tokens) +
        ", calibrated=" + std::to_string(calibrated.token_count));
  }

  const JsonArray input_ids = tokenize_input_ids(
      transport, base, options.model, calibrated.content,
      options.timeout_seconds, enable_thinking, calibrated.token_count);
  if (input_ids.empty()) {
    throw std::runtime_error("exact acceptance prompt produced no token IDs");
  }

  flush_cache(transport, base, options.timeout_seconds, Backend::kSglang, 0);
  const JsonValue warmup = generate_acceptance(
      transport, base, input_ids, options.warmup_output_tokens, options);
  static_cast<void>(validate_generate_result(
      warmup, options.input_tokens, options.warmup_output_tokens, "warmup"));
  flush_cache(transport, base, options.timeout_seconds, Backend::kSglang, 0);
  const JsonValue response = generate_acceptance(
      transport, base, input_ids, options.output_tokens, options);
  const GenerateResultView measurement = validate_generate_result(
      response, options.input_tokens, options.output_tokens, "measurement");
  validate_acceptance_metrics(measurement);

  const JsonValue &meta = *measurement.meta;
  return JsonValue(JsonObject{
      {"prompt_tokens", JsonValue(measurement.prompt_tokens)},
      {"completion_tokens", JsonValue(measurement.completion_tokens)},
      {"enable_thinking", JsonValue(enable_thinking)},
      {"e2e_latency", meta.at("e2e_latency")},
      {"spec_accept_rate", meta.at("spec_accept_rate")},
      {"spec_accept_length", meta.at("spec_accept_length")},
      {"spec_num_correct_drafts", meta.at("spec_num_correct_drafts")},
      {"spec_num_proposed_drafts", meta.at("spec_num_proposed_drafts")},
      {"spec_verify_ct", meta.at("spec_verify_ct")},
      {"spec_correct_drafts_histogram",
       meta.at("spec_correct_drafts_histogram")},
      {"output_sha256", JsonValue(sha256_hex(*measurement.text))},
  });
}

} // namespace sglang::benchmark
