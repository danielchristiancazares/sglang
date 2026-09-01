#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <sglang/benchmark/config.hpp>

namespace sglang::benchmark {

enum class Backend {
  kSglang,
  kLlama,
};

struct StreamOptions final {
  std::string base_url{"http://127.0.0.1:30000"};
  std::string model{"qwen3.8-27b"};
  Backend backend{Backend::kSglang};
  std::int64_t slot_id{0};
  std::int64_t input_tokens{6213};
  std::int64_t output_tokens{128};
  std::int64_t warmup_output_tokens{16};
  std::int64_t warmup_runs{1};
  double timeout_seconds{600.0};
  double temperature{0.0};
  std::optional<double> top_p;
  std::optional<std::int64_t> top_k;
  std::optional<double> min_p;
  std::optional<double> presence_penalty;
  std::optional<double> repetition_penalty;
  std::optional<std::int64_t> seed;
  bool skip_warmup{false};
  bool disable_thinking{false};
};

struct AcceptanceOptions final {
  std::string base_url{"http://127.0.0.1:30000"};
  std::string model{"qwen3.8-27b"};
  std::int64_t input_tokens{6213};
  std::int64_t output_tokens{512};
  std::int64_t warmup_output_tokens{16};
  double timeout_seconds{600.0};
  double temperature{1.0};
  double top_p{0.95};
  std::int64_t top_k{20};
  double presence_penalty{1.5};
  bool disable_thinking{false};
};

enum class ParseStatus {
  kRun,
  kHelp,
  kError,
};

template <typename Options> struct ParseResult final {
  ParseStatus status{ParseStatus::kError};
  Options options{};
  std::string message;

  [[nodiscard]] bool should_run() const noexcept {
    return status == ParseStatus::kRun;
  }
};

[[nodiscard]] ParseResult<StreamOptions>
parse_stream_arguments(std::span<const std::string_view> arguments);
[[nodiscard]] ParseResult<AcceptanceOptions>
parse_acceptance_arguments(std::span<const std::string_view> arguments);

[[nodiscard]] std::string_view stream_help() noexcept;
[[nodiscard]] std::string_view acceptance_help() noexcept;
[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;

} // namespace sglang::benchmark
