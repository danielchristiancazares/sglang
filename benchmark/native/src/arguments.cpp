#include "sglang/benchmark/arguments.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>

namespace sglang::benchmark {
namespace {

constexpr std::string_view kStreamHelp = R"(usage: bench_openai_stream [options]

Benchmark one local OpenAI-compatible streaming request.

options:
  -h, --help                    show this help message and exit
  --base-url URL                server root (default: http://127.0.0.1:30000)
  --model MODEL                 served model (default: qwen3.8-27b)
  --backend {sglang,llama}      backend protocol (default: sglang)
  --slot-id N                   llama.cpp slot to erase (default: 0)
  --input-tokens N              exact prompt tokens (default: 6213)
  --output-tokens N             completion tokens (default: 128)
  --warmup-output-tokens N      tokens per warmup (default: 16)
  --warmup-runs N               warmup count (default: 1)
  --timeout SECONDS             request timeout (default: 600.0)
  --temperature VALUE           sampling temperature (default: 0.0)
  --top-p VALUE                 optional nucleus probability
  --top-k N                     optional top-k limit
  --min-p VALUE                 optional minimum probability
  --presence-penalty VALUE      optional presence penalty
  --repetition-penalty VALUE    optional repetition penalty
  --seed N                      optional request seed forwarded to the backend;
                                speculative proposal sampling may remain nondeterministic
  --skip-warmup                 skip every configured warmup
  --disable-thinking            set enable_thinking and preserve_thinking false
)";

constexpr std::string_view kAcceptanceHelp =
    R"(usage: bench_spec_acceptance [options]

options:
  -h, --help                    show this help message and exit
  --base-url URL                server root (default: http://127.0.0.1:30000)
  --model MODEL                 served model (default: qwen3.8-27b)
  --input-tokens N              exact prompt tokens (default: 6213)
  --output-tokens N             generated tokens (default: 512)
  --warmup-output-tokens N      warmup tokens (default: 16)
  --timeout SECONDS             request timeout (default: 600.0)
  --temperature VALUE           sampling temperature (default: 1.0)
  --top-p VALUE                 nucleus probability (default: 0.95)
  --top-k N                     top-k limit (default: 20)
  --presence-penalty VALUE      presence penalty (default: 1.5)
  --disable-thinking            set enable_thinking and preserve_thinking false
)";

struct ParsedToken final {
  std::string_view name;
  std::optional<std::string_view> inline_value;
};

[[nodiscard]] ParsedToken split_option(std::string_view token) noexcept {
  const auto equals = token.find('=');
  if (equals == std::string_view::npos) {
    return ParsedToken{token, std::nullopt};
  }
  return ParsedToken{token.substr(0, equals), token.substr(equals + 1)};
}

template <typename Options>
[[nodiscard]] ParseResult<Options> error(std::string message) {
  return ParseResult<Options>{ParseStatus::kError, {}, std::move(message)};
}

template <typename Options>
[[nodiscard]] ParseResult<Options> help(Options options) {
  return ParseResult<Options>{ParseStatus::kHelp, std::move(options), {}};
}

template <typename Options>
[[nodiscard]] ParseResult<Options> run(Options options) {
  return ParseResult<Options>{ParseStatus::kRun, std::move(options), {}};
}

[[nodiscard]] std::string quoted(std::string_view value) {
  std::string result{"'"};
  result.append(value);
  result.push_back('\'');
  return result;
}

[[nodiscard]] std::optional<std::string>
mark_once(std::unordered_set<std::string_view> &seen, std::string_view name) {
  if (!seen.insert(name).second) {
    return "duplicate option: " + quoted(name);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view>
value_for(const ParsedToken &option,
          std::span<const std::string_view> arguments, std::size_t &index,
          std::string &failure) {
  if (option.inline_value.has_value()) {
    if (option.inline_value->empty()) {
      failure = "missing value for " + quoted(option.name);
      return std::nullopt;
    }
    return option.inline_value;
  }
  if (index + 1 >= arguments.size() || arguments[index + 1].starts_with("--")) {
    failure = "missing value for " + quoted(option.name);
    return std::nullopt;
  }
  ++index;
  return arguments[index];
}

template <typename Number>
[[nodiscard]] bool parse_number(std::string_view text,
                                Number &output) noexcept {
  const char *const begin = text.data();
  const char *const end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, output);
  return parsed.ec == std::errc{} && parsed.ptr == end;
}

[[nodiscard]] bool parse_real(std::string_view text, double &output) noexcept {
  if (!parse_number(text, output)) {
    return false;
  }
  return std::isfinite(output);
}

template <typename Options, typename Number>
[[nodiscard]] std::optional<ParseResult<Options>>
set_number(const ParsedToken &option,
           std::span<const std::string_view> arguments, std::size_t &index,
           Number &destination) {
  std::string failure;
  const auto value = value_for(option, arguments, index, failure);
  if (!value.has_value()) {
    return error<Options>(std::move(failure));
  }
  if (!parse_number(*value, destination)) {
    return error<Options>("invalid value for " + quoted(option.name) + ": " +
                          quoted(*value));
  }
  return std::nullopt;
}

template <typename Options>
[[nodiscard]] std::optional<ParseResult<Options>>
set_real(const ParsedToken &option, std::span<const std::string_view> arguments,
         std::size_t &index, double &destination) {
  std::string failure;
  const auto value = value_for(option, arguments, index, failure);
  if (!value.has_value()) {
    return error<Options>(std::move(failure));
  }
  if (!parse_real(*value, destination)) {
    return error<Options>("invalid value for " + quoted(option.name) + ": " +
                          quoted(*value));
  }
  return std::nullopt;
}

template <typename Options>
[[nodiscard]] std::optional<ParseResult<Options>>
set_string(const ParsedToken &option,
           std::span<const std::string_view> arguments, std::size_t &index,
           std::string &destination) {
  std::string failure;
  const auto value = value_for(option, arguments, index, failure);
  if (!value.has_value()) {
    return error<Options>(std::move(failure));
  }
  destination.assign(*value);
  return std::nullopt;
}

template <typename Options>
[[nodiscard]] std::optional<ParseResult<Options>>
reject_flag_value(const ParsedToken &option) {
  if (option.inline_value.has_value()) {
    return error<Options>("option does not take a value: " +
                          quoted(option.name));
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string>
validate_common(std::string_view base_url, std::string_view model,
                std::int64_t input_tokens, std::int64_t output_tokens,
                std::int64_t warmup_output_tokens, double timeout_seconds,
                double temperature) {
  if (base_url.empty())
    return "base URL must be non-empty";
  if (model.empty())
    return "model must be non-empty";
  if (input_tokens <= 0)
    return "input token count must be positive";
  if (output_tokens <= 0)
    return "output token count must be positive";
  if (warmup_output_tokens <= 0) {
    return "warmup output token count must be positive";
  }
  if (!(timeout_seconds > 0.0))
    return "timeout must be positive";
  if (temperature < 0.0)
    return "temperature must be non-negative";
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string>
validate_probability(std::string_view name,
                     const std::optional<double> &value) {
  if (value.has_value() && !(*value >= 0.0 && *value <= 1.0)) {
    return std::string{name} + " must be within [0, 1]";
  }
  return std::nullopt;
}

} // namespace

std::string_view stream_help() noexcept { return kStreamHelp; }

std::string_view acceptance_help() noexcept { return kAcceptanceHelp; }

std::string_view backend_name(Backend backend) noexcept {
  switch (backend) {
  case Backend::kSglang:
    return "sglang";
  case Backend::kLlama:
    return "llama";
  }
  return "unknown";
}

ParseResult<StreamOptions>
parse_stream_arguments(std::span<const std::string_view> arguments) {
  StreamOptions options;
  std::unordered_set<std::string_view> seen;
  bool requested_help = false;

  for (std::size_t index = 0; index < arguments.size(); ++index) {
    ParsedToken option = split_option(arguments[index]);
    if (option.name == "-h")
      option.name = "--help";
    if (!option.name.starts_with("--")) {
      return error<StreamOptions>("unexpected argument: " +
                                  quoted(arguments[index]));
    }
    if (const auto failure = mark_once(seen, option.name)) {
      return error<StreamOptions>(*failure);
    }

    std::optional<ParseResult<StreamOptions>> failure;
    if (option.name == "--help") {
      failure = reject_flag_value<StreamOptions>(option);
      requested_help = !failure.has_value();
    } else if (option.name == "--base-url") {
      failure =
          set_string<StreamOptions>(option, arguments, index, options.base_url);
    } else if (option.name == "--model") {
      failure =
          set_string<StreamOptions>(option, arguments, index, options.model);
    } else if (option.name == "--backend") {
      std::string value;
      failure = set_string<StreamOptions>(option, arguments, index, value);
      if (!failure.has_value()) {
        if (value == "sglang") {
          options.backend = Backend::kSglang;
        } else if (value == "llama") {
          options.backend = Backend::kLlama;
        } else {
          failure = error<StreamOptions>(
              "invalid value for '--backend': " + quoted(value) +
              " (expected 'sglang' or 'llama')");
        }
      }
    } else if (option.name == "--slot-id") {
      failure =
          set_number<StreamOptions>(option, arguments, index, options.slot_id);
    } else if (option.name == "--input-tokens") {
      failure = set_number<StreamOptions>(option, arguments, index,
                                          options.input_tokens);
    } else if (option.name == "--output-tokens") {
      failure = set_number<StreamOptions>(option, arguments, index,
                                          options.output_tokens);
    } else if (option.name == "--warmup-output-tokens") {
      failure = set_number<StreamOptions>(option, arguments, index,
                                          options.warmup_output_tokens);
    } else if (option.name == "--warmup-runs") {
      failure = set_number<StreamOptions>(option, arguments, index,
                                          options.warmup_runs);
    } else if (option.name == "--timeout") {
      failure = set_real<StreamOptions>(option, arguments, index,
                                        options.timeout_seconds);
    } else if (option.name == "--temperature") {
      failure = set_real<StreamOptions>(option, arguments, index,
                                        options.temperature);
    } else if (option.name == "--top-p") {
      double value = 0.0;
      failure = set_real<StreamOptions>(option, arguments, index, value);
      if (!failure.has_value())
        options.top_p = value;
    } else if (option.name == "--top-k") {
      std::int64_t value = 0;
      failure = set_number<StreamOptions>(option, arguments, index, value);
      if (!failure.has_value())
        options.top_k = value;
    } else if (option.name == "--min-p") {
      double value = 0.0;
      failure = set_real<StreamOptions>(option, arguments, index, value);
      if (!failure.has_value())
        options.min_p = value;
    } else if (option.name == "--presence-penalty") {
      double value = 0.0;
      failure = set_real<StreamOptions>(option, arguments, index, value);
      if (!failure.has_value())
        options.presence_penalty = value;
    } else if (option.name == "--repetition-penalty") {
      double value = 0.0;
      failure = set_real<StreamOptions>(option, arguments, index, value);
      if (!failure.has_value())
        options.repetition_penalty = value;
    } else if (option.name == "--seed") {
      std::int64_t value = 0;
      failure = set_number<StreamOptions>(option, arguments, index, value);
      if (!failure.has_value())
        options.seed = value;
    } else if (option.name == "--skip-warmup") {
      failure = reject_flag_value<StreamOptions>(option);
      if (!failure.has_value())
        options.skip_warmup = true;
    } else if (option.name == "--disable-thinking") {
      failure = reject_flag_value<StreamOptions>(option);
      if (!failure.has_value())
        options.disable_thinking = true;
    } else {
      return error<StreamOptions>("unknown option: " + quoted(option.name));
    }
    if (failure.has_value())
      return std::move(*failure);
  }

  if (requested_help)
    return help(std::move(options));
  if (const auto failure =
          validate_common(options.base_url, options.model, options.input_tokens,
                          options.output_tokens, options.warmup_output_tokens,
                          options.timeout_seconds, options.temperature)) {
    return error<StreamOptions>(*failure);
  }
  if (options.slot_id < 0)
    return error<StreamOptions>("slot ID must be non-negative");
  if (options.warmup_runs < 0) {
    return error<StreamOptions>("warmup run count must be non-negative");
  }
  if (const auto failure = validate_probability("top-p", options.top_p)) {
    return error<StreamOptions>(*failure);
  }
  if (const auto failure = validate_probability("min-p", options.min_p)) {
    return error<StreamOptions>(*failure);
  }
  if (options.top_k.has_value() && *options.top_k <= 0) {
    return error<StreamOptions>("top-k must be positive");
  }
  if (options.repetition_penalty.has_value() &&
      !(*options.repetition_penalty > 0.0)) {
    return error<StreamOptions>("repetition penalty must be positive");
  }
  return run(std::move(options));
}

ParseResult<AcceptanceOptions>
parse_acceptance_arguments(std::span<const std::string_view> arguments) {
  AcceptanceOptions options;
  std::unordered_set<std::string_view> seen;
  bool requested_help = false;

  for (std::size_t index = 0; index < arguments.size(); ++index) {
    ParsedToken option = split_option(arguments[index]);
    if (option.name == "-h")
      option.name = "--help";
    if (!option.name.starts_with("--")) {
      return error<AcceptanceOptions>("unexpected argument: " +
                                      quoted(arguments[index]));
    }
    if (const auto failure = mark_once(seen, option.name)) {
      return error<AcceptanceOptions>(*failure);
    }

    std::optional<ParseResult<AcceptanceOptions>> failure;
    if (option.name == "--help") {
      failure = reject_flag_value<AcceptanceOptions>(option);
      requested_help = !failure.has_value();
    } else if (option.name == "--base-url") {
      failure = set_string<AcceptanceOptions>(option, arguments, index,
                                              options.base_url);
    } else if (option.name == "--model") {
      failure = set_string<AcceptanceOptions>(option, arguments, index,
                                              options.model);
    } else if (option.name == "--input-tokens") {
      failure = set_number<AcceptanceOptions>(option, arguments, index,
                                              options.input_tokens);
    } else if (option.name == "--output-tokens") {
      failure = set_number<AcceptanceOptions>(option, arguments, index,
                                              options.output_tokens);
    } else if (option.name == "--warmup-output-tokens") {
      failure = set_number<AcceptanceOptions>(option, arguments, index,
                                              options.warmup_output_tokens);
    } else if (option.name == "--timeout") {
      failure = set_real<AcceptanceOptions>(option, arguments, index,
                                            options.timeout_seconds);
    } else if (option.name == "--temperature") {
      failure = set_real<AcceptanceOptions>(option, arguments, index,
                                            options.temperature);
    } else if (option.name == "--top-p") {
      failure =
          set_real<AcceptanceOptions>(option, arguments, index, options.top_p);
    } else if (option.name == "--top-k") {
      failure = set_number<AcceptanceOptions>(option, arguments, index,
                                              options.top_k);
    } else if (option.name == "--presence-penalty") {
      failure = set_real<AcceptanceOptions>(option, arguments, index,
                                            options.presence_penalty);
    } else if (option.name == "--disable-thinking") {
      failure = reject_flag_value<AcceptanceOptions>(option);
      if (!failure.has_value())
        options.disable_thinking = true;
    } else {
      return error<AcceptanceOptions>("unknown option: " + quoted(option.name));
    }
    if (failure.has_value())
      return std::move(*failure);
  }

  if (requested_help)
    return help(std::move(options));
  if (const auto failure =
          validate_common(options.base_url, options.model, options.input_tokens,
                          options.output_tokens, options.warmup_output_tokens,
                          options.timeout_seconds, options.temperature)) {
    return error<AcceptanceOptions>(*failure);
  }
  if (!(options.top_p >= 0.0 && options.top_p <= 1.0)) {
    return error<AcceptanceOptions>("top-p must be within [0, 1]");
  }
  if (options.top_k <= 0) {
    return error<AcceptanceOptions>("top-k must be positive");
  }
  return run(std::move(options));
}

} // namespace sglang::benchmark
