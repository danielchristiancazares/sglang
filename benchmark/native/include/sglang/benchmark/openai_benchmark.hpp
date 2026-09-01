#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <sglang/benchmark/arguments.hpp>
#include <sglang/benchmark/http_client.hpp>
#include <sglang/benchmark/json.hpp>

namespace sglang::benchmark {

using BenchmarkNow = std::function<HttpTimePoint()>;

inline constexpr std::string_view kPromptUnit =
    "Inspect this local program carefully, preserve its behavior, and identify "
    "the next useful correctness or performance change. ";
inline constexpr std::string_view kFillerUnit = " x";

struct CalibratedPrompt final {
  std::string content;
  std::int64_t token_count{0};
};

struct StreamRequestOptions final {
  std::string base_url;
  std::string model;
  std::string content;
  std::int64_t output_tokens{0};
  double timeout_seconds{0.0};
  std::optional<std::int64_t> seed;
  double temperature{0.0};
  std::optional<double> top_p;
  std::optional<std::int64_t> top_k;
  std::optional<double> min_p;
  std::optional<double> presence_penalty;
  std::optional<double> repetition_penalty;
  bool enable_thinking{true};
  BenchmarkNow now{[] { return HttpClock::now(); }};
};

// Accumulates the JSON payloads of SSE data lines. The caller supplies the
// receive timestamp associated with each completed line, which makes timing
// behavior independently testable without sleeping or a live server.
class StreamAccumulator final {
public:
  // Returns true after a [DONE] marker has been consumed. Production timing
  // is sampled after JSON decoding and fragment accounting, matching the
  // Python benchmark's perf_counter boundary.
  [[nodiscard]] bool consume_data(std::string_view data);

  [[nodiscard]] bool consume_data_with_clock(std::string_view data,
                                             const BenchmarkNow &now);

  // Deterministic test surface for the same post-processing boundary.
  [[nodiscard]] bool consume_data_at(std::string_view data,
                                     HttpTimePoint processed_at);

  [[nodiscard]] JsonValue finalize(HttpTimePoint started_at,
                                   HttpTimePoint ended_at) const;
  [[nodiscard]] bool done() const noexcept { return done_; }

private:
  [[nodiscard]] bool consume_data_impl(std::string_view data,
                                       const BenchmarkNow &now);

  JsonValue usage_{JsonValue::object{}};
  std::optional<std::string> finish_reason_;
  std::string output_text_;
  std::string reasoning_text_;
  std::string content_text_;
  std::int64_t reasoning_chars_{0};
  std::int64_t content_chars_{0};
  std::int64_t nonempty_delta_count_{0};
  std::int64_t reasoning_fragment_count_{0};
  std::int64_t content_fragment_count_{0};
  std::optional<std::int64_t> first_output_delta_chars_;
  std::int64_t max_output_delta_chars_{0};
  std::optional<HttpTimePoint> first_output_at_;
  std::optional<HttpTimePoint> last_output_at_;
  bool done_{false};
};

[[nodiscard]] JsonValue chat_template_kwargs(bool enable_thinking);
[[nodiscard]] JsonValue messages_for(std::string_view content);

[[nodiscard]] JsonValue request_json(HttpTransport &transport,
                                     std::string_view url,
                                     const JsonValue &payload,
                                     double timeout_seconds);

[[nodiscard]] std::int64_t
token_count(HttpTransport &transport, std::string_view base_url,
            std::string_view model, std::string_view content,
            double timeout_seconds, Backend backend, bool enable_thinking);

[[nodiscard]] CalibratedPrompt
calibrate_prompt(HttpTransport &transport, std::string_view base_url,
                 std::string_view model, std::int64_t target_tokens,
                 double timeout_seconds, Backend backend,
                 bool enable_thinking = true);

void flush_cache(HttpTransport &transport, std::string_view base_url,
                 double timeout_seconds, Backend backend, std::int64_t slot_id);

[[nodiscard]] JsonValue stream_request(HttpTransport &transport,
                                       const StreamRequestOptions &options);

void validate_result_counts(const JsonValue &result,
                            std::int64_t expected_prompt_tokens,
                            std::int64_t expected_completion_tokens,
                            std::string_view label);

// Executes calibration, cache flushing, optional warmup requests, and the
// measured stream. timestamp_utc is injectable for deterministic callers; an
// empty value selects the current UTC time.
[[nodiscard]] JsonValue run_stream_benchmark(
    HttpTransport &transport, const StreamOptions &options,
    std::string timestamp_utc = {},
    BenchmarkNow now = [] { return HttpClock::now(); });

// Executes exact SGLang tokenization, a warmup /generate, and the measured
// speculative-acceptance /generate request with fail-closed result checks.
[[nodiscard]] JsonValue
run_acceptance_benchmark(HttpTransport &transport,
                         const AcceptanceOptions &options);

} // namespace sglang::benchmark
