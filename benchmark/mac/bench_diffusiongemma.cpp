// SPDX-License-Identifier: Apache-2.0
// Client-visible diffusion timing: never infer token rate from chunk spacing.
#include "sglang/benchmark/http_client.hpp"
#include "sglang/benchmark/json.hpp"
#include "sglang/benchmark/sha256.hpp"
#include "sglang/benchmark/sse_parser.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

namespace dg {
namespace wire = sglang::benchmark;
using Clock = std::chrono::steady_clock;
using Instant = Clock::time_point;

class InputFailure final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class ProtocolFailure final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class NonEmptyString final {
  std::string value_;
 public:
  explicit NonEmptyString(std::string value) : value_(std::move(value)) {
    if (value_.empty()) throw InputFailure("A nonempty string is required.");
  }
  const std::string& text() const { return value_; }
  void append(const NonEmptyString& tail) { value_ += tail.value_; }
};

template <int Maximum> class PositiveCount final {
  int value_;
  static int validate(std::int64_t value) {
    if (value < 1 || value > Maximum) throw InputFailure("Count exceeds its permitted range.");
    return static_cast<int>(value);
  }
 public:
  explicit PositiveCount(std::int64_t value) : value_(validate(value)) {}
  static PositiveCount parse(std::string_view text) {
    int value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
      throw InputFailure("Expected a decimal integer.");
    return PositiveCount(value);
  }
  int value() const { return value_; }
};
using Budget = PositiveCount<4096>;
using Canvas = PositiveCount<256>;
using SampleCount = PositiveCount<20>;

class Sampler final {
  enum class Algorithm { EntropyBound, ConfidenceThreshold };
  Algorithm algorithm_;
  explicit Sampler(Algorithm algorithm) : algorithm_(algorithm) {}
 public:
  static Sampler parse(std::string_view value) {
    if (value == "entropy-bound") return Sampler(Algorithm::EntropyBound);
    if (value == "confidence-threshold") return Sampler(Algorithm::ConfidenceThreshold);
    throw InputFailure("Sampler must be entropy-bound or confidence-threshold.");
  }
  NonEmptyString name() const {
    switch (algorithm_) {
      case Algorithm::EntropyBound: return NonEmptyString("entropy-bound");
      case Algorithm::ConfidenceThreshold: return NonEmptyString("confidence-threshold");
    }
    throw ProtocolFailure("Sampler dispatch failed.");
  }
};

class Sampling final {
  enum class Process { Argmax, CheckpointSchedule };
  Process process_;
  explicit Sampling(Process process) : process_(process) {}
 public:
  static Sampling parse(std::string_view value) {
    if (value == "greedy") return Sampling(Process::Argmax);
    if (value == "sampled") return Sampling(Process::CheckpointSchedule);
    throw InputFailure("Sampling must be greedy or sampled.");
  }
  void write_request(wire::JsonValue::object& request, SampleCount ordinal) const {
    switch (process_) {
      case Process::Argmax:
        request.emplace("temperature", 0.0);
        request.emplace("seed", 42);
        return;
      case Process::CheckpointSchedule:
        request.emplace("temperature", 1.0);
        request.emplace("seed", 41 + ordinal.value());
        return;
    }
    throw ProtocolFailure("Sampling dispatch failed.");
  }
};

class FinishReason final {
  enum class Cause { ModelStop, BudgetExhausted };
  Cause cause_;
  explicit FinishReason(Cause cause) : cause_(cause) {}
 public:
  static FinishReason parse(std::string_view value) {
    if (value == "stop") return FinishReason(Cause::ModelStop);
    if (value == "length") return FinishReason(Cause::BudgetExhausted);
    throw ProtocolFailure("Unexpected completion finish reason.");
  }
  NonEmptyString name() const {
    switch (cause_) {
      case Cause::ModelStop: return NonEmptyString("stop");
      case Cause::BudgetExhausted: return NonEmptyString("length");
    }
    throw ProtocolFailure("Completion dispatch failed.");
  }
};

class Measurement final {
  double value_;
 public:
  explicit Measurement(double value) : value_(value) {
    if (!std::isfinite(value) || value <= 0)
      throw ProtocolFailure("Expected a finite positive timing or memory measurement.");
  }
  double value() const { return value_; }
};

struct Totals final {
  PositiveCount<1048576> prompt;
  Budget output;
  Measurement decode_rate;
  Measurement peak_gb;
};

// The stream owner controls every transition. A terminal report can only be
// obtained after visible text, a model finish event, usage, and the SSE terminator.
class StreamMeasurement final {
  struct WaitingForText { Instant start; };
  struct Rendering { Instant start; Instant first; NonEmptyString text; };
  struct ModelFinished { Rendering rendering; FinishReason reason; };
  struct UsageReceived { ModelFinished finished; Totals totals; };
  struct Complete { UsageReceived usage; Instant end; };
  using State = std::variant<WaitingForText, Rendering, ModelFinished, UsageReceived, Complete>;
  State state_;

  void append(const NonEmptyString& text, Instant now) {
    state_ = std::visit([&](auto&& state) -> State {
      using T = std::decay_t<decltype(state)>;
      if constexpr (std::is_same_v<T, WaitingForText>)
        return Rendering{state.start, now, text};
      else if constexpr (std::is_same_v<T, Rendering>) {
        state.text.append(text);
        return std::move(state);
      } else
        throw ProtocolFailure("Text arrived after completion.");
    }, std::move(state_));
  }
  void finish(FinishReason reason) {
    state_ = std::visit([&](auto&& state) -> State {
      using T = std::decay_t<decltype(state)>;
      if constexpr (std::is_same_v<T, Rendering>)
        return ModelFinished{std::move(state), reason};
      else
        throw ProtocolFailure("Completion did not follow visible text.");
    }, std::move(state_));
  }
  void account(Totals totals) {
    state_ = std::visit([&](auto&& state) -> State {
      using T = std::decay_t<decltype(state)>;
      if constexpr (std::is_same_v<T, ModelFinished>)
        return UsageReceived{std::move(state), totals};
      else
        throw ProtocolFailure("Usage did not follow the model finish event.");
    }, std::move(state_));
  }
  void complete(Instant now) {
    state_ = std::visit([&](auto&& state) -> State {
      using T = std::decay_t<decltype(state)>;
      if constexpr (std::is_same_v<T, UsageReceived>)
        return Complete{std::move(state), now};
      else
        throw ProtocolFailure("SSE ended before complete usage accounting.");
    }, std::move(state_));
  }

 public:
  explicit StreamMeasurement(Instant start) : state_(WaitingForText{start}) {}

  // JSON null/optional fields and SSE control flags stay in this wire adapter.
  void receive(const wire::SseEvent& event) {
    switch (event.kind) {
      case wire::SseEventKind::kDone:
        complete(event.line_completed_at);
        return;
      case wire::SseEventKind::kData:
        break;
    }
    const auto json = wire::JsonValue::parse(event.data);
    if (json.contains("error")) throw ProtocolFailure(json.at("error").dump());
    for (const auto& choice : json.at("choices").as_array()) {
      if (choice.at("index").as_int() != 0)
        throw ProtocolFailure("Expected one completion choice.");
      const auto& delta = choice.at("delta");
      if (delta.contains("reasoning_content") && !delta.at("reasoning_content").is_null() &&
          !delta.at("reasoning_content").as_string().empty())
        throw ProtocolFailure("The direct-answer benchmark received reasoning content.");
      if (delta.contains("content") && !delta.at("content").is_null() &&
          !delta.at("content").as_string().empty())
        append(NonEmptyString(delta.at("content").as_string()), event.line_completed_at);
      if (!choice.at("finish_reason").is_null())
        finish(FinishReason::parse(choice.at("finish_reason").as_string()));
    }
    if (json.contains("usage") && !json.at("usage").is_null()) {
      const auto& usage = json.at("usage");
      const auto& timings = json.at("timings");
      if (usage.at("total_tokens").as_int() != usage.at("prompt_tokens").as_int() +
          usage.at("completion_tokens").as_int())
        throw ProtocolFailure("Usage token counts disagree.");
      account(Totals{PositiveCount<1048576>(usage.at("prompt_tokens").as_int()),
                     Budget(usage.at("completion_tokens").as_int()),
                     Measurement(timings.at("predicted_per_second").as_double()),
                     Measurement(timings.at("peak_memory").as_double())});
    }
  }

  NonEmptyString report() const {
    return std::visit([](const auto& state) -> NonEmptyString {
      using T = std::decay_t<decltype(state)>;
      if constexpr (std::is_same_v<T, Complete>) {
        const auto& rendering = state.usage.finished.rendering;
        const auto& totals = state.usage.totals;
        const Measurement ttft(std::chrono::duration<double>(rendering.first - rendering.start).count());
        const Measurement elapsed(std::chrono::duration<double>(state.end - rendering.start).count());
        return NonEmptyString(wire::JsonValue(wire::JsonValue::object{
          {"prompt_tokens", totals.prompt.value()}, {"output_tokens", totals.output.value()},
          {"ttft_seconds", ttft.value()}, {"elapsed_seconds", elapsed.value()},
          {"end_to_end_tokens_per_second", totals.output.value() / elapsed.value()},
          {"decode_tokens_per_second", totals.decode_rate.value()},
          {"peak_memory_gb", totals.peak_gb.value()},
          {"finish_reason", state.usage.finished.reason.name().text()},
          {"text_sha256", wire::sha256_hex(rendering.text.text())},
          {"text", rendering.text.text()}}).dump());
      } else
        throw ProtocolFailure("The response stream is incomplete.");
    }, state_);
  }
};

NonEmptyString read_prompt(const NonEmptyString& filename) {
  if (std::filesystem::file_size(filename.text()) > 1048576)
    throw InputFailure("Prompt file exceeds one MiB.");
  std::ifstream input(filename.text());
  if (!input) throw InputFailure("Cannot open prompt file.");
  const std::string contents(std::istreambuf_iterator<char>(input), {});
  if (input.bad()) throw InputFailure("Cannot read the complete prompt file.");
  return NonEmptyString(contents);
}

NonEmptyString measure(const NonEmptyString& model, const NonEmptyString& prompt,
                       Budget budget, Canvas canvas, const Sampler& sampler,
                       const Sampling& sampling, SampleCount ordinal) {
  // External HTTP/JSON booleans are serialized only in this boundary adapter.
  wire::HttpRequest request;
  request.method = "POST";
  request.url = "http://127.0.0.1:30001/v1/chat/completions";
  request.headers = {{"Content-Type", "application/json"}};
  request.connect_timeout = std::chrono::seconds(5);
  request.io_timeout = std::chrono::seconds(120);
  wire::JsonValue::object body{
    {"model", model.text()},
    {"messages", wire::JsonValue::array{wire::JsonValue::object{{"role", "user"}, {"content", prompt.text()}}}},
    {"max_tokens", budget.value()},
    {"enable_thinking", false}, {"stream", true},
    {"stream_options", wire::JsonValue::object{{"include_usage", true}}},
    {"diffusion_sampler", sampler.name().text()},
    {"diffusion_min_canvas_length", canvas.value()},
    {"diffusion_max_canvas_length", canvas.value()}};
  sampling.write_request(body, ordinal);
  request.body = wire::JsonValue(std::move(body)).dump();
  StreamMeasurement measurement(Clock::now());
  wire::SseParser parser;
  const auto accept = [&](const wire::SseEvent& event) {
    measurement.receive(event);
    return true;
  };
  wire::SocketHttpTransport transport;
  const auto response = transport.perform(request, [&](std::string_view bytes, Instant at) {
    switch (parser.feed(bytes, at, accept)) {
      case wire::SseParseStatus::kContinue:
      case wire::SseParseStatus::kDone: return true;
      case wire::SseParseStatus::kStopped: throw ProtocolFailure("Unexpected stopped SSE parser.");
      case wire::SseParseStatus::kError: throw ProtocolFailure(std::string(parser.error()));
    }
    throw ProtocolFailure("Unexpected SSE parser state.");
  });
  if (!response.ok()) throw ProtocolFailure(response.error.message);
  if (response.response.status_code != 200)
    throw ProtocolFailure("HTTP " + std::to_string(response.response.status_code) + ": " + response.response.body);
  return measurement.report();
}
}  // namespace dg

int main(int argc, char** argv) {
  try {
    if (argc != 9) {
      std::cerr << "Usage: bench_diffusiongemma MODEL PROMPT_FILE MAX_TOKENS CANVAS SAMPLER greedy|sampled SAMPLES RESULT_JSONL\n";
      return 2;
    }
    const dg::NonEmptyString model(argv[1]);
    const auto prompt = dg::read_prompt(dg::NonEmptyString(argv[2]));
    const auto budget = dg::Budget::parse(argv[3]);
    const auto canvas = dg::Canvas::parse(argv[4]);
    const auto sampler = dg::Sampler::parse(argv[5]);
    const auto sampling = dg::Sampling::parse(argv[6]);
    const auto samples = dg::SampleCount::parse(argv[7]);
    const dg::NonEmptyString destination(argv[8]);
    std::ofstream output(destination.text(), std::ios::out | std::ios::noreplace);
    if (!output) throw dg::InputFailure("Cannot create a new result file.");
    for (int sample = 0; sample < samples.value(); ++sample) {
      std::cerr << "Starting sample " << sample + 1 << '/' << samples.value() << std::endl;
      const auto result = dg::measure(model, prompt, budget, canvas, sampler,
                                      sampling, dg::SampleCount(sample + 1));
      output << result.text() << '\n' << std::flush;
      if (!output) throw dg::InputFailure("Cannot save benchmark result.");
      std::cout << result.text() << std::endl;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "DiffusionGemma benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
