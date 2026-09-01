#include "sglang/benchmark/openai_benchmark.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sglang/benchmark/sha256.hpp"

namespace {

using namespace std::chrono_literals;
using sglang::benchmark::AcceptanceOptions;
using sglang::benchmark::Backend;
using sglang::benchmark::calibrate_prompt;
using sglang::benchmark::CalibratedPrompt;
using sglang::benchmark::HttpBodyChunkCallback;
using sglang::benchmark::HttpRequest;
using sglang::benchmark::HttpResponse;
using sglang::benchmark::HttpResult;
using sglang::benchmark::HttpTimePoint;
using sglang::benchmark::HttpTransport;
using sglang::benchmark::JsonValue;
using sglang::benchmark::kFillerUnit;
using sglang::benchmark::kPromptUnit;
using sglang::benchmark::run_acceptance_benchmark;
using sglang::benchmark::run_stream_benchmark;
using sglang::benchmark::sha256_hex;
using sglang::benchmark::stream_request;
using sglang::benchmark::StreamAccumulator;
using sglang::benchmark::StreamOptions;
using sglang::benchmark::StreamRequestOptions;
using sglang::benchmark::token_count;
using sglang::benchmark::validate_result_counts;

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

template <typename Callable> [[nodiscard]] bool throws(Callable &&callable) {
  try {
    std::invoke(std::forward<Callable>(callable));
  } catch (const std::exception &) {
    return true;
  }
  return false;
}

[[nodiscard]] HttpTimePoint at(std::chrono::milliseconds elapsed) {
  return HttpTimePoint{} + elapsed;
}

class StreamTransport final : public HttpTransport {
public:
  std::vector<std::pair<std::string, HttpTimePoint>> chunks;
  HttpRequest observed;

  [[nodiscard]] HttpResult
  perform(const HttpRequest &request,
          const HttpBodyChunkCallback &on_body_chunk) override {
    observed = request;
    HttpResult result;
    result.response.status_code = 200;
    result.response.request_started_at = at(0ms);
    result.response.headers_completed_at = at(100ms);
    for (const auto &[chunk, received_at] : chunks) {
      result.response.body.append(chunk);
      result.response.body_bytes += chunk.size();
      result.response.completed_at = received_at;
      if (on_body_chunk && !on_body_chunk(chunk, received_at)) {
        result.response.stopped_early = true;
        break;
      }
    }
    result.response.body_complete = !result.response.stopped_early;
    return result;
  }
};

class LocalProtocolTransport final : public HttpTransport {
public:
  bool inconsistent_histogram{false};
  bool unreachable_exact_target{false};
  std::vector<HttpRequest> requests;

  [[nodiscard]] HttpResult
  perform(const HttpRequest &request,
          const HttpBodyChunkCallback &on_body_chunk) override {
    requests.push_back(request);
    HttpResult result;
    HttpResponse &response = result.response;
    response.status_code = 200;
    response.request_started_at = at(0ms);
    response.headers_completed_at = at(1ms);
    response.completed_at = at(2ms);
    response.body_complete = true;

    if (request.url.find("/apply-template") != std::string::npos) {
      const JsonValue body = JsonValue::parse(request.body);
      const std::string &content =
          body.at("messages").at(0).at("content").as_string();
      response.body =
          JsonValue(JsonValue::object{{"prompt", JsonValue("T:" + content)}})
              .dump();
      return result;
    }
    if (request.url.find("/v1/tokenize") != std::string::npos) {
      const JsonValue body = JsonValue::parse(request.body);
      const std::string &content =
          body.at("messages").at(0).at("content").as_string();
      const std::int64_t count = calibrated_count(content);
      JsonValue::array tokens;
      for (std::int64_t token = 0; token < count; ++token) {
        tokens.emplace_back(token);
      }
      response.body =
          JsonValue(JsonValue::object{{"count", JsonValue(count)},
                                      {"tokens", JsonValue(std::move(tokens))}})
              .dump();
      return result;
    }
    if (request.url.find("/tokenize") != std::string::npos) {
      const JsonValue body = JsonValue::parse(request.body);
      const std::string &prompt = body.at("content").as_string();
      if (!prompt.starts_with("T:")) {
        throw std::runtime_error("llama template was not passed to tokenize");
      }
      const std::int64_t count =
          calibrated_count(std::string_view(prompt).substr(2));
      JsonValue::array tokens(static_cast<std::size_t>(count), JsonValue(7));
      response.body =
          JsonValue(JsonValue::object{{"tokens", JsonValue(std::move(tokens))}})
              .dump();
      return result;
    }
    if (request.url.find("/flush_cache") != std::string::npos ||
        request.url.find("/slots/") != std::string::npos) {
      response.body = "{}";
      return result;
    }
    if (request.url.ends_with("/v1/chat/completions")) {
      if (!on_body_chunk) {
        throw std::runtime_error("stream request omitted its body callback");
      }
      const JsonValue body = JsonValue::parse(request.body);
      const std::string &content =
          body.at("messages").at(0).at("content").as_string();
      const std::int64_t prompt_tokens = calibrated_count(content);
      const std::int64_t completion_tokens =
          body.at("max_completion_tokens").as_int();
      const std::string stream =
          "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"x\"},"
          "\"finish_reason\":null}]}\n"
          "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],"
          "\"usage\":{\"prompt_tokens\":" +
          std::to_string(prompt_tokens) +
          ",\"completion_tokens\":" + std::to_string(completion_tokens) +
          ",\"total_tokens\":" +
          std::to_string(prompt_tokens + completion_tokens) +
          "}}\n"
          "data: [DONE]\n";
      response.body = stream;
      response.body_bytes = stream.size();
      response.stopped_early = !on_body_chunk(stream, at(0ms));
      response.body_complete = !response.stopped_early;
      return result;
    }
    if (request.url.ends_with("/generate")) {
      const JsonValue body = JsonValue::parse(request.body);
      const std::int64_t prompt_tokens =
          static_cast<std::int64_t>(body.at("input_ids").size());
      const std::int64_t completion_tokens =
          body.at("sampling_params").at("max_new_tokens").as_int();
      JsonValue::object meta{
          {"prompt_tokens", JsonValue(prompt_tokens)},
          {"completion_tokens", JsonValue(completion_tokens)},
          {"finish_reason",
           JsonValue(JsonValue::object{{"type", JsonValue("length")}})},
      };
      if (completion_tokens == 4) {
        meta.emplace("e2e_latency", JsonValue(0.5));
        meta.emplace("spec_accept_rate", JsonValue(0.5));
        meta.emplace("spec_accept_length", JsonValue(2.0));
        meta.emplace("spec_num_correct_drafts", JsonValue(2));
        meta.emplace("spec_num_proposed_drafts", JsonValue(4));
        meta.emplace("spec_verify_ct", JsonValue(2));
        meta.emplace("spec_correct_drafts_histogram",
                     JsonValue(inconsistent_histogram
                                   ? JsonValue::array{2, 0, 0}
                                   : JsonValue::array{1, 0, 1}));
      }
      response.body =
          JsonValue(JsonValue::object{{"meta_info", JsonValue(std::move(meta))},
                                      {"text", JsonValue("answer")}})
              .dump();
      return result;
    }
    throw std::runtime_error("unexpected test URL: " + request.url);
  }

private:
  [[nodiscard]] std::int64_t calibrated_count(std::string_view content) const {
    std::int64_t count = 2;
    std::size_t position = 0;
    while (content.substr(position).starts_with(kPromptUnit)) {
      position += kPromptUnit.size();
      count += 3;
    }
    while (content.substr(position).starts_with(kFillerUnit)) {
      position += kFillerUnit.size();
      count += unreachable_exact_target ? 2 : 1;
    }
    if (position != content.size()) {
      throw std::runtime_error("unexpected calibration prompt composition");
    }
    return count;
  }
};

[[nodiscard]] bool ConstantsAndPayloadHelpersMatchLegacyTool() {
  CHECK(kPromptUnit ==
        "Inspect this local program carefully, preserve its behavior, and "
        "identify the next useful correctness or performance change. ");
  CHECK(kFillerUnit == " x");
  const JsonValue messages = sglang::benchmark::messages_for("hello");
  CHECK(messages.is_array());
  CHECK(messages.size() == 1);
  CHECK(messages.at(0).at("role").as_string() == "user");
  CHECK(messages.at(0).at("content").as_string() == "hello");
  const JsonValue enabled = sglang::benchmark::chat_template_kwargs(true);
  CHECK(enabled.at("enable_thinking").as_bool());
  CHECK(enabled.at("preserve_thinking").as_bool());
  return true;
}

[[nodiscard]] bool AccumulatorPreservesUnicodeOrderHashesAndTiming() {
  StreamAccumulator accumulator;
  CHECK(!accumulator.consume_data_at(
      R"({"choices":[{"delta":{"reasoning_content":"\ud83d\ude00","content":"\u00e9"},"finish_reason":null}]})",
      at(1s)));
  CHECK(!accumulator.consume_data_at(
      R"({"choices":[{"delta":{"reasoning_content":"\u601d"},"finish_reason":null}]})",
      at(2s)));
  CHECK(!accumulator.consume_data_at(
      R"({"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}})",
      at(2500ms)));
  CHECK(accumulator.consume_data_at("[DONE]", at(3s)));

  const JsonValue result = accumulator.finalize(at(0ms), at(3s));
  CHECK(result.at("prompt_tokens").as_int() == 10);
  CHECK(result.at("completion_tokens").as_int() == 3);
  CHECK(result.at("ttft_s").as_double() == 1.0);
  CHECK(result.at("e2e_s").as_double() == 3.0);
  CHECK(result.at("observed_prompt_tps").as_double() == 10.0);
  CHECK(result.at("decode_tps").as_double() == 1.0);
  CHECK(result.at("output_tps_e2e").as_double() == 1.0);
  CHECK(result.at("output_chars").as_int() == 3);
  CHECK(result.at("reasoning_chars").as_int() == 2);
  CHECK(result.at("content_chars").as_int() == 1);
  CHECK(result.at("nonempty_delta_count").as_int() == 2);
  CHECK(result.at("reasoning_fragment_count").as_int() == 2);
  CHECK(result.at("content_fragment_count").as_int() == 1);
  CHECK(result.at("first_output_delta_chars").as_int() == 2);
  CHECK(result.at("max_output_delta_chars").as_int() == 2);
  CHECK(result.at("trailing_after_last_delta_s").as_double() == 1.0);
  CHECK(result.at("output_sha256").as_string() == sha256_hex("😀é思"));
  CHECK(result.at("reasoning_sha256").as_string() == sha256_hex("😀思"));
  CHECK(result.at("content_sha256").as_string() == sha256_hex("é"));
  return true;
}

[[nodiscard]] bool StreamRequestUsesSseTimestampsAndOptionalOmission() {
  StreamTransport transport;
  transport.chunks = {
      {"event: ignored\n", at(500ms)},
      {R"(data: {"choices":[{"delta":{"reasoning_content":"a"},"finish_reason":null}]})",
       at(750ms)},
      {"\n", at(1s)},
      {R"(data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}})"
       "\n",
       at(2s)},
      {"data: [DONE]\n", at(3s)},
  };
  StreamRequestOptions options;
  options.base_url = "http://localhost:30000/";
  options.model = "model";
  options.content = "prompt";
  options.output_tokens = 2;
  options.timeout_seconds = 10.0;
  options.temperature = 0.0;
  options.top_p = 0.95;
  options.enable_thinking = false;
  options.now = [] { return at(1s); };
  const JsonValue result = stream_request(transport, options);
  CHECK(result.at("ttft_s").as_double() == 1.0);
  CHECK(result.at("decode_tps").as_double() == 0.5);
  CHECK(result.at("trailing_after_last_delta_s").as_double() == 2.0);
  CHECK(transport.observed.url == "http://localhost:30000/v1/chat/completions");
  const JsonValue body = JsonValue::parse(transport.observed.body);
  CHECK(body.at("max_completion_tokens").as_int() == 2);
  CHECK(body.at("top_p").as_double() == 0.95);
  CHECK(!body.contains("top_k"));
  CHECK(!body.contains("seed"));
  CHECK(!body.contains("min_p"));
  CHECK(!body.contains("presence_penalty"));
  CHECK(!body.contains("repetition_penalty"));
  CHECK(!body.at("chat_template_kwargs").at("enable_thinking").as_bool());
  return true;
}

[[nodiscard]] bool CalibrationPreservesBinarySearchAndFillerShape() {
  LocalProtocolTransport transport;
  const CalibratedPrompt calibrated =
      calibrate_prompt(transport, "http://localhost:30000", "model", 10, 30.0,
                       Backend::kSglang, true);
  CHECK(calibrated.token_count == 10);
  CHECK(calibrated.content ==
        std::string(kPromptUnit) + std::string(kPromptUnit) + " x x");
  CHECK(!transport.requests.empty());
  for (const HttpRequest &request : transport.requests) {
    CHECK(request.url == "http://localhost:30000/v1/tokenize");
    const JsonValue body = JsonValue::parse(request.body);
    CHECK(body.at("model").as_string() == "model");
    CHECK(body.at("chat_template_kwargs").at("enable_thinking").as_bool());
  }
  return true;
}

[[nodiscard]] bool LlamaTemplateTokenizeProtocolIsPreserved() {
  LocalProtocolTransport transport;
  const std::int64_t count =
      token_count(transport, "http://localhost:8080/", "llama-model",
                  kPromptUnit, 5.0, Backend::kLlama, false);
  CHECK(count == 5);
  CHECK(transport.requests.size() == 2);
  CHECK(transport.requests[0].url == "http://localhost:8080/apply-template");
  CHECK(transport.requests[1].url == "http://localhost:8080/tokenize");
  const JsonValue apply = JsonValue::parse(transport.requests[0].body);
  CHECK(!apply.at("chat_template_kwargs").at("enable_thinking").as_bool());
  const JsonValue tokenize = JsonValue::parse(transport.requests[1].body);
  CHECK(tokenize.at("content").as_string().starts_with("T:"));
  CHECK(!tokenize.at("add_special").as_bool());
  CHECK(tokenize.at("parse_special").as_bool());
  CHECK(!tokenize.at("with_pieces").as_bool());
  return true;
}

[[nodiscard]] bool ExactStreamCountValidationFailsClosed() {
  const JsonValue valid(JsonValue::object{
      {"prompt_tokens", JsonValue(10)},
      {"completion_tokens", JsonValue(2)},
      {"total_tokens", JsonValue(12)},
      {"finish_reason", JsonValue("length")},
  });
  validate_result_counts(valid, 10, 2, "measurement");
  JsonValue invalid = valid;
  invalid.as_object()["prompt_tokens"] = JsonValue(9);
  CHECK(throws([&] { validate_result_counts(invalid, 10, 2, "measurement"); }));
  invalid = valid;
  invalid.as_object()["completion_tokens"] = JsonValue(1);
  CHECK(throws([&] { validate_result_counts(invalid, 10, 2, "measurement"); }));
  invalid = valid;
  invalid.as_object()["total_tokens"] = JsonValue(11);
  CHECK(throws([&] { validate_result_counts(invalid, 10, 2, "measurement"); }));
  invalid = valid;
  invalid.as_object()["finish_reason"] = JsonValue("stop");
  CHECK(throws([&] { validate_result_counts(invalid, 10, 2, "measurement"); }));
  return true;
}

[[nodiscard]] bool StreamRunnerPreservesFlushWarmupAndMetadata() {
  LocalProtocolTransport transport;
  StreamOptions options;
  options.base_url = "http://localhost:30000/";
  options.model = "model";
  options.input_tokens = 10;
  options.output_tokens = 4;
  options.warmup_output_tokens = 2;
  options.warmup_runs = 2;
  options.timeout_seconds = 10.0;
  options.temperature = 0.0;

  const JsonValue result = run_stream_benchmark(
      transport, options, "2026-08-30T16:00:00+00:00", [] { return at(1ms); });
  CHECK(result.size() == 38);
  CHECK(result.at("prompt_tokens").as_int() == 10);
  CHECK(result.at("completion_tokens").as_int() == 4);
  CHECK(result.at("total_tokens").as_int() == 14);
  CHECK(result.at("timestamp").as_string() == "2026-08-30T16:00:00+00:00");
  CHECK(result.at("base_url").as_string() == "http://localhost:30000");
  CHECK(result.at("model").as_string() == "model");
  CHECK(result.at("backend").as_string() == "sglang");
  CHECK(result.at("requested_prompt_tokens").as_int() == 10);
  CHECK(result.at("calibrated_prompt_tokens").as_int() == 10);
  CHECK(result.at("requested_completion_tokens").as_int() == 4);
  CHECK(result.at("warmup").as_bool());
  CHECK(result.at("warmup_runs").as_int() == 2);
  CHECK(result.at("enable_thinking").as_bool());
  CHECK(result.at("seed").is_null());

  std::vector<std::string> tail_urls;
  std::vector<std::int64_t> stream_lengths;
  for (const HttpRequest &request : transport.requests) {
    if (request.url.find("/flush_cache") != std::string::npos ||
        request.url.ends_with("/v1/chat/completions")) {
      tail_urls.push_back(request.url);
    }
    if (request.url.ends_with("/v1/chat/completions")) {
      stream_lengths.push_back(
          JsonValue::parse(request.body).at("max_completion_tokens").as_int());
    }
  }
  CHECK(tail_urls.size() == 6);
  CHECK(tail_urls[0].find("/flush_cache") != std::string::npos);
  CHECK(tail_urls[1].ends_with("/v1/chat/completions"));
  CHECK(tail_urls[2].find("/flush_cache") != std::string::npos);
  CHECK(tail_urls[3].ends_with("/v1/chat/completions"));
  CHECK(tail_urls[4].find("/flush_cache") != std::string::npos);
  CHECK(tail_urls[5].ends_with("/v1/chat/completions"));
  CHECK(stream_lengths == std::vector<std::int64_t>({2, 2, 4}));
  return true;
}

[[nodiscard]] AcceptanceOptions acceptance_options() {
  AcceptanceOptions options;
  options.base_url = "http://localhost:30000/";
  options.model = "model";
  options.input_tokens = 5;
  options.output_tokens = 4;
  options.warmup_output_tokens = 2;
  options.timeout_seconds = 600.0;
  options.temperature = 1.0;
  options.top_p = 0.95;
  options.top_k = 20;
  options.presence_penalty = 1.5;
  return options;
}

[[nodiscard]] bool AcceptanceRunnerValidatesAndExtractsExactResult() {
  LocalProtocolTransport transport;
  const JsonValue result =
      run_acceptance_benchmark(transport, acceptance_options());
  CHECK(result.size() == 11);
  CHECK(result.at("prompt_tokens").as_int() == 5);
  CHECK(result.at("completion_tokens").as_int() == 4);
  CHECK(result.at("enable_thinking").as_bool());
  CHECK(result.at("e2e_latency").as_double() == 0.5);
  CHECK(result.at("spec_accept_rate").as_double() == 0.5);
  CHECK(result.at("spec_accept_length").as_double() == 2.0);
  CHECK(result.at("spec_num_correct_drafts").as_int() == 2);
  CHECK(result.at("spec_num_proposed_drafts").as_int() == 4);
  CHECK(result.at("spec_verify_ct").as_int() == 2);
  CHECK(result.at("spec_correct_drafts_histogram").as_array() ==
        JsonValue::array({1, 0, 1}));
  CHECK(result.at("output_sha256").as_string() == sha256_hex("answer"));

  CHECK(transport.requests.size() >= 5);
  const std::size_t count = transport.requests.size();
  CHECK(transport.requests[count - 4].url.find("/flush_cache?timeout=600") !=
        std::string::npos);
  CHECK(transport.requests[count - 3].url.ends_with("/generate"));
  CHECK(transport.requests[count - 2].url.find("/flush_cache?timeout=600") !=
        std::string::npos);
  CHECK(transport.requests[count - 1].url.ends_with("/generate"));
  const JsonValue measured =
      JsonValue::parse(transport.requests[count - 1].body);
  CHECK(measured.at("input_ids").size() == 5);
  CHECK(measured.at("sampling_params").at("max_new_tokens").as_int() == 4);
  CHECK(measured.at("sampling_params").at("temperature").as_double() == 1.0);
  CHECK(measured.at("sampling_params").at("top_p").as_double() == 0.95);
  CHECK(measured.at("sampling_params").at("top_k").as_int() == 20);
  CHECK(measured.at("sampling_params").at("presence_penalty").as_double() ==
        1.5);
  CHECK(measured.at("sampling_params").at("ignore_eos").as_bool());
  return true;
}

[[nodiscard]] bool AcceptanceRejectsInvalidArgumentsBeforeNetworkWork() {
  LocalProtocolTransport transport;
  AcceptanceOptions options = acceptance_options();
  options.input_tokens = 0;
  CHECK(throws([&] {
    static_cast<void>(run_acceptance_benchmark(transport, options));
  }));
  CHECK(transport.requests.empty());
  return true;
}

[[nodiscard]] bool AcceptanceRejectsInexactCalibrationBeforeGeneration() {
  LocalProtocolTransport transport;
  transport.unreachable_exact_target = true;
  AcceptanceOptions options = acceptance_options();
  options.input_tokens = 3;
  CHECK(throws([&] {
    static_cast<void>(run_acceptance_benchmark(transport, options));
  }));
  for (const HttpRequest &request : transport.requests) {
    CHECK(!request.url.ends_with("/generate"));
    CHECK(request.url.find("/flush_cache") == std::string::npos);
  }
  return true;
}

[[nodiscard]] bool AcceptanceRejectsInconsistentMeasurementCounters() {
  LocalProtocolTransport transport;
  transport.inconsistent_histogram = true;
  const AcceptanceOptions options = acceptance_options();
  CHECK(throws([&] {
    static_cast<void>(run_acceptance_benchmark(transport, options));
  }));
  return true;
}

using Test = bool (*)();

constexpr std::array<std::pair<std::string_view, Test>, 11> kTests{{
    {"ConstantsAndPayloadHelpersMatchLegacyTool",
     ConstantsAndPayloadHelpersMatchLegacyTool},
    {"AccumulatorPreservesUnicodeOrderHashesAndTiming",
     AccumulatorPreservesUnicodeOrderHashesAndTiming},
    {"StreamRequestUsesSseTimestampsAndOptionalOmission",
     StreamRequestUsesSseTimestampsAndOptionalOmission},
    {"CalibrationPreservesBinarySearchAndFillerShape",
     CalibrationPreservesBinarySearchAndFillerShape},
    {"LlamaTemplateTokenizeProtocolIsPreserved",
     LlamaTemplateTokenizeProtocolIsPreserved},
    {"ExactStreamCountValidationFailsClosed",
     ExactStreamCountValidationFailsClosed},
    {"StreamRunnerPreservesFlushWarmupAndMetadata",
     StreamRunnerPreservesFlushWarmupAndMetadata},
    {"AcceptanceRunnerValidatesAndExtractsExactResult",
     AcceptanceRunnerValidatesAndExtractsExactResult},
    {"AcceptanceRejectsInvalidArgumentsBeforeNetworkWork",
     AcceptanceRejectsInvalidArgumentsBeforeNetworkWork},
    {"AcceptanceRejectsInexactCalibrationBeforeGeneration",
     AcceptanceRejectsInexactCalibrationBeforeGeneration},
    {"AcceptanceRejectsInconsistentMeasurementCounters",
     AcceptanceRejectsInconsistentMeasurementCounters},
}};

} // namespace

int main() {
  for (const auto &[name, test] : kTests) {
    try {
      if (test()) {
        continue;
      }
      std::printf("FAILED: %.*s\n", static_cast<int>(name.size()), name.data());
      return 1;
    } catch (const std::exception &error) {
      std::printf("FAILED: %.*s: %s\n", static_cast<int>(name.size()),
                  name.data(), error.what());
      return 1;
    } catch (...) {
      std::printf("FAILED: %.*s: unknown exception\n",
                  static_cast<int>(name.size()), name.data());
      return 1;
    }
  }
  std::printf("openai_benchmark_test: %zu/%zu passed\n", kTests.size(),
              kTests.size());
  return 0;
}
