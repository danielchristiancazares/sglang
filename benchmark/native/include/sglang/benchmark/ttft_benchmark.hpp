#pragma once

#include <functional>
#include <span>
#include <string>
#include <string_view>

#include <sglang/benchmark/openai_benchmark.hpp>

namespace sglang::benchmark {

struct TtftOptions final {
  std::string base_url{"http://127.0.0.1:30000"};
  std::string model{"qwen3.8-27b"};
  std::string label{"control"};
  std::int64_t input_tokens{6213};
  std::int64_t output_tokens{512};
  std::int64_t samples{5};
  // A second exact calibrated prompt. Zero disables the continuation.
  std::int64_t continuation_tokens{0};
  double timeout_seconds{600.0};
};

[[nodiscard]] TtftOptions
parse_ttft_arguments(std::span<const std::string_view> arguments);
[[nodiscard]] std::string_view ttft_help() noexcept;

// Emits a manifest, each individual warmup/measurement, and phase summaries.
// Calibration is not inference. Only completed requests enter summaries.
// Each sequence is flush -> seed -> identical replay -> optional longer prompt;
// there is no cache flush inside a retained-prefix sequence.
void run_ttft_benchmark(
    HttpTransport &transport, const TtftOptions &options,
    const std::function<void(const JsonValue &)> &emit,
    BenchmarkNow now = [] { return HttpClock::now(); });

} // namespace sglang::benchmark
