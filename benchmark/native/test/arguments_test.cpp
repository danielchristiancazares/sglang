#include "sglang/benchmark/arguments.hpp"

#include <array>
#include <cstdio>
#include <string_view>
#include <utility>

namespace {

using sglang::benchmark::AcceptanceOptions;
using sglang::benchmark::Backend;
using sglang::benchmark::parse_acceptance_arguments;
using sglang::benchmark::parse_stream_arguments;
using sglang::benchmark::ParseStatus;
using sglang::benchmark::StreamOptions;

[[nodiscard]] bool record_check(bool passed, const char *expression,
                                int line) noexcept {
  if (!passed) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return passed;
}

#define CHECK(condition)                                                       \
  if (!record_check(static_cast<bool>(condition), #condition, __LINE__)) {     \
    return false;                                                              \
  }

template <typename... Values>
[[nodiscard]] auto parse_stream(Values... values) {
  const std::array<std::string_view, sizeof...(Values)> arguments{
      std::string_view{values}...};
  return parse_stream_arguments(arguments);
}

template <typename... Values>
[[nodiscard]] auto parse_acceptance(Values... values) {
  const std::array<std::string_view, sizeof...(Values)> arguments{
      std::string_view{values}...};
  return parse_acceptance_arguments(arguments);
}

[[nodiscard]] bool StreamDefaultsMatchLegacyTool() {
  const auto parsed = parse_stream();
  CHECK(parsed.status == ParseStatus::kRun);
  const StreamOptions &value = parsed.options;
  CHECK(value.base_url == "http://127.0.0.1:30000");
  CHECK(value.model == "qwen3.8-27b");
  CHECK(value.backend == Backend::kSglang);
  CHECK(value.slot_id == 0);
  CHECK(value.input_tokens == 6213);
  CHECK(value.output_tokens == 128);
  CHECK(value.warmup_output_tokens == 16);
  CHECK(value.warmup_runs == 1);
  CHECK(value.timeout_seconds == 600.0);
  CHECK(value.temperature == 0.0);
  CHECK(!value.top_p.has_value());
  CHECK(!value.top_k.has_value());
  CHECK(!value.min_p.has_value());
  CHECK(!value.presence_penalty.has_value());
  CHECK(!value.repetition_penalty.has_value());
  CHECK(!value.seed.has_value());
  CHECK(!value.skip_warmup);
  CHECK(!value.disable_thinking);
  return true;
}

[[nodiscard]] bool AcceptanceDefaultsMatchLegacyTool() {
  const auto parsed = parse_acceptance();
  CHECK(parsed.status == ParseStatus::kRun);
  const AcceptanceOptions &value = parsed.options;
  CHECK(value.base_url == "http://127.0.0.1:30000");
  CHECK(value.model == "qwen3.8-27b");
  CHECK(value.input_tokens == 6213);
  CHECK(value.output_tokens == 512);
  CHECK(value.warmup_output_tokens == 16);
  CHECK(value.timeout_seconds == 600.0);
  CHECK(value.temperature == 1.0);
  CHECK(value.top_p == 0.95);
  CHECK(value.top_k == 20);
  CHECK(value.presence_penalty == 1.5);
  CHECK(!value.disable_thinking);
  return true;
}

[[nodiscard]] bool StreamAcceptsEveryLegacyOption() {
  const auto parsed = parse_stream(
      "--base-url=http://localhost:8080/", "--model", "local", "--backend",
      "llama", "--slot-id", "4", "--input-tokens", "17", "--output-tokens=19",
      "--warmup-output-tokens", "3", "--warmup-runs", "2", "--timeout", "0.25",
      "--temperature", "1.125", "--top-p", "0.9", "--top-k", "8", "--min-p",
      "0.1", "--presence-penalty", "-0.5", "--repetition-penalty", "1.2",
      "--seed", "-7", "--skip-warmup", "--disable-thinking");
  CHECK(parsed.status == ParseStatus::kRun);
  const StreamOptions &value = parsed.options;
  CHECK(value.base_url == "http://localhost:8080/");
  CHECK(value.model == "local");
  CHECK(value.backend == Backend::kLlama);
  CHECK(value.slot_id == 4);
  CHECK(value.input_tokens == 17);
  CHECK(value.output_tokens == 19);
  CHECK(value.warmup_output_tokens == 3);
  CHECK(value.warmup_runs == 2);
  CHECK(value.timeout_seconds == 0.25);
  CHECK(value.temperature == 1.125);
  CHECK(value.top_p == 0.9);
  CHECK(value.top_k == 8);
  CHECK(value.min_p == 0.1);
  CHECK(value.presence_penalty == -0.5);
  CHECK(value.repetition_penalty == 1.2);
  CHECK(value.seed == -7);
  CHECK(value.skip_warmup);
  CHECK(value.disable_thinking);
  return true;
}

[[nodiscard]] bool AcceptanceAcceptsEveryLegacyOption() {
  const auto parsed = parse_acceptance(
      "--base-url", "http://localhost:8080", "--model=x", "--input-tokens", "7",
      "--output-tokens", "9", "--warmup-output-tokens", "2", "--timeout", "5.5",
      "--temperature", "0.75", "--top-p", "0.8", "--top-k", "4",
      "--presence-penalty", "-1.25", "--disable-thinking");
  CHECK(parsed.status == ParseStatus::kRun);
  const AcceptanceOptions &value = parsed.options;
  CHECK(value.base_url == "http://localhost:8080");
  CHECK(value.model == "x");
  CHECK(value.input_tokens == 7);
  CHECK(value.output_tokens == 9);
  CHECK(value.warmup_output_tokens == 2);
  CHECK(value.timeout_seconds == 5.5);
  CHECK(value.temperature == 0.75);
  CHECK(value.top_p == 0.8);
  CHECK(value.top_k == 4);
  CHECK(value.presence_penalty == -1.25);
  CHECK(value.disable_thinking);
  return true;
}

[[nodiscard]] bool HelpIsASeparateSuccessfulDisposition() {
  const auto stream = parse_stream("-h");
  const auto acceptance = parse_acceptance("--help");
  CHECK(stream.status == ParseStatus::kHelp);
  CHECK(acceptance.status == ParseStatus::kHelp);
  CHECK(sglang::benchmark::stream_help().find("--repetition-penalty") !=
        std::string_view::npos);
  CHECK(sglang::benchmark::acceptance_help().find("--top-k") !=
        std::string_view::npos);
  return true;
}

[[nodiscard]] bool UnknownDuplicateAndMissingValuesFailClosed() {
  CHECK(parse_stream("position").status == ParseStatus::kError);
  CHECK(parse_stream("--unknown").status == ParseStatus::kError);
  CHECK(parse_stream("--model", "x", "--model=y").status ==
        ParseStatus::kError);
  CHECK(parse_stream("--skip-warmup", "--skip-warmup").status ==
        ParseStatus::kError);
  CHECK(parse_stream("-h", "--help").status == ParseStatus::kError);
  CHECK(parse_stream("--model").status == ParseStatus::kError);
  CHECK(parse_stream("--model", "--top-k", "4").status == ParseStatus::kError);
  CHECK(parse_stream("--model=").status == ParseStatus::kError);
  CHECK(parse_stream("--skip-warmup=true").status == ParseStatus::kError);
  CHECK(parse_acceptance("--timeout", "--top-k", "4").status ==
        ParseStatus::kError);
  return true;
}

[[nodiscard]] bool MalformedAndOverflowedNumbersFailClosed() {
  CHECK(parse_stream("--input-tokens", "12x").status == ParseStatus::kError);
  CHECK(parse_stream("--input-tokens", "999999999999999999999999999").status ==
        ParseStatus::kError);
  CHECK(parse_stream("--timeout", "1e9999").status == ParseStatus::kError);
  CHECK(parse_stream("--temperature", "nan").status == ParseStatus::kError);
  CHECK(parse_acceptance("--top-k", "3.5").status == ParseStatus::kError);
  return true;
}

[[nodiscard]] bool InvalidDomainsFailBeforeNetworkWork() {
  CHECK(parse_stream("--input-tokens", "0").status == ParseStatus::kError);
  CHECK(parse_stream("--output-tokens", "-1").status == ParseStatus::kError);
  CHECK(parse_stream("--warmup-output-tokens", "0").status ==
        ParseStatus::kError);
  CHECK(parse_stream("--warmup-runs", "-1").status == ParseStatus::kError);
  CHECK(parse_stream("--timeout", "0").status == ParseStatus::kError);
  CHECK(parse_stream("--slot-id", "-1").status == ParseStatus::kError);
  CHECK(parse_stream("--temperature", "-0.1").status == ParseStatus::kError);
  CHECK(parse_stream("--top-p", "1.1").status == ParseStatus::kError);
  CHECK(parse_stream("--min-p", "-0.1").status == ParseStatus::kError);
  CHECK(parse_stream("--top-k", "0").status == ParseStatus::kError);
  CHECK(parse_stream("--repetition-penalty", "0").status ==
        ParseStatus::kError);
  CHECK(parse_acceptance("--input-tokens", "0").status == ParseStatus::kError);
  CHECK(parse_acceptance("--top-p", "1.5").status == ParseStatus::kError);
  CHECK(parse_acceptance("--top-k", "0").status == ParseStatus::kError);
  return true;
}

using Test = bool (*)();

constexpr std::array<std::pair<std::string_view, Test>, 8> kTests{{
    {"StreamDefaultsMatchLegacyTool", StreamDefaultsMatchLegacyTool},
    {"AcceptanceDefaultsMatchLegacyTool", AcceptanceDefaultsMatchLegacyTool},
    {"StreamAcceptsEveryLegacyOption", StreamAcceptsEveryLegacyOption},
    {"AcceptanceAcceptsEveryLegacyOption", AcceptanceAcceptsEveryLegacyOption},
    {"HelpIsASeparateSuccessfulDisposition",
     HelpIsASeparateSuccessfulDisposition},
    {"UnknownDuplicateAndMissingValuesFailClosed",
     UnknownDuplicateAndMissingValuesFailClosed},
    {"MalformedAndOverflowedNumbersFailClosed",
     MalformedAndOverflowedNumbersFailClosed},
    {"InvalidDomainsFailBeforeNetworkWork",
     InvalidDomainsFailBeforeNetworkWork},
}};

} // namespace

int main() {
  for (const auto &[name, test] : kTests) {
    if (!test()) {
      std::printf("FAILED: %.*s\n", static_cast<int>(name.size()), name.data());
      return 1;
    }
  }
  std::printf("arguments_test: %zu/%zu passed\n", kTests.size(), kTests.size());
  return 0;
}
