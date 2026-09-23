#include <sglang/benchmark/ttft_benchmark.hpp>

#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace {
using namespace sglang::benchmark;
using namespace std::chrono_literals;
using Object = Json::object;
using Array = Json::array;

HttpTimePoint at(std::chrono::milliseconds value) {
  return HttpTimePoint{} + value;
}

class Protocol final : public HttpTransport {
public:
  Json server = Object{{"enable_cache_report", true},
                       {"disable_radix_cache", false},
                       {"served_model_name", "qwen3.8-27b"},
                       {"max_running_requests", 1},
                       {"context_length", 200000},
                       {"max_total_tokens", 200000},
                       {"chunked_prefill_size", 4096},
                       {"max_mamba_cache_size", 5},
                       {"startup_time", Object{{"unix", 123456}}}};
  Json model =
      Object{{"served_model_name", "qwen3.8-27b"},
             {"is_generation", true},
             {"model_path", "private/model"},
             {"tokenizer_path", "private/tokenizer"},
             {"weight_version", nullptr},
             {"load_format", "auto"},
             {"reasoning_parser", "qwen3"},
             {"tool_call_parser", "qwen3_coder"},
             {"has_image_understanding", false},
             {"has_audio_understanding", false},
             {"model_type", "qwen"},
             {"architectures", Array{"Qwen"}},
             {"preferred_sampling_params", Object{{"temperature", 1.0}}}};
  std::vector<std::string> operations;
  std::vector<Json> payloads;
  int server_reads{0};
  int model_reads{0};
  int stream_count{0};
  int fail_stream{0};
  bool truncate_done{false};
  bool no_output{false};
  bool wrong_counts{false};
  bool wrong_finish{false};
  bool flush_hit{false};
  bool restart{false};
  bool change_config{false};
  bool change_model{false};
  bool bad_token_ids{false};
  bool unreachable_count{false};
  bool omit_cached{false};
  bool explicit_zero{false};
  std::optional<Json> malformed_cached;

  HttpResult perform(const HttpRequest &request,
                     const HttpBodyChunkCallback &callback) override {
    HttpResult result;
    auto &response = result.response;
    response.status_code = 200;
    response.request_started_at = at(0ms);
    response.headers_completed_at = at(1ms);
    response.completed_at = at(10ms);
    response.body_complete = true;
    if (request.url.ends_with("/server_info")) {
      operations.emplace_back("server");
      auto returned = server;
      if (++server_reads > 1) {
        if (restart)
          returned.as_object()["startup_time"] = Object{{"unix", 999999}};
        if (change_config)
          returned.as_object()["chunked_prefill_size"] = 7680;
      }
      response.body = returned.dump();
      return result;
    }
    if (request.url.ends_with("/model_info")) {
      operations.emplace_back("model");
      auto returned = model;
      if (++model_reads > 1 && change_model)
        returned.as_object()["weight_version"] = "changed";
      response.body = returned.dump();
      return result;
    }
    if (request.url.ends_with("/v1/tokenize")) {
      operations.emplace_back("tokenize");
      const auto payload = Json::parse(request.body);
      auto ids = tokens(payload.at("messages").at(0).at("content").as_string());
      if (bad_token_ids && !ids.empty())
        ids.back() = -1;
      response.body =
          Json(Object{{"count", ids.size()}, {"tokens", ids}}).dump();
      return result;
    }
    if (request.url.find("/flush_cache") != std::string::npos) {
      operations.emplace_back("flush");
      cached_sequence_ = false;
      response.body = "{}";
      return result;
    }
    if (!request.url.ends_with("/v1/chat/completions")) {
      throw std::runtime_error("unexpected URL " + request.url);
    }
    operations.emplace_back("stream");
    ++stream_count;
    if (fail_stream == stream_count) {
      result.error = {HttpErrorCode::kReceiveTimeout, "injected timeout"};
      return result;
    }
    const auto payload = Json::parse(request.body);
    payloads.push_back(payload);
    const auto count =
        tokens(payload.at("messages").at(0).at("content").as_string()).size();
    const auto completion = payload.at("max_completion_tokens").as_int();
    Object usage{{"prompt_tokens", count + (wrong_counts ? 1 : 0)},
                 {"completion_tokens", completion},
                 {"total_tokens", count + completion}};
    if (malformed_cached) {
      usage["prompt_tokens_details"] =
          Object{{"cached_tokens", *malformed_cached}};
    } else if (explicit_zero) {
      usage["prompt_tokens_details"] = Object{{"cached_tokens", 0}};
    } else if (!omit_cached && (cached_sequence_ || flush_hit)) {
      usage["prompt_tokens_details"] = Object{{"cached_tokens", 6}};
    }
    cached_sequence_ = true;
    const auto send = [&](const std::string &bytes,
                          std::chrono::milliseconds received) {
      response.body += bytes;
      response.body_bytes += bytes.size();
      if (!callback(bytes, at(received)))
        response.stopped_early = true;
    };
    send("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n", 2ms);
    if (!no_output) {
      send("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"x\"}}]}\n",
           3ms);
    }
    send("data: " +
             Json(Object{{"choices",
                          Array{Object{{"delta", Object{}},
                                       {"finish_reason",
                                        wrong_finish ? "stop" : "length"}}}},
                         {"usage", usage}})
                 .dump() +
             "\n",
         7ms);
    if (!truncate_done)
      send("data: [DONE]\n", 10ms);
    return result;
  }

private:
  bool cached_sequence_{false};
  Array tokens(std::string_view content) const {
    Array ids{1};
    std::size_t position = 0;
    while (content.substr(position).starts_with(kPromptUnit)) {
      ids.insert(ids.end(), {2, 3, 4});
      position += kPromptUnit.size();
    }
    while (content.substr(position).starts_with(kFillerUnit)) {
      ids.emplace_back(5);
      if (unreachable_count)
        ids.emplace_back(6);
      position += kFillerUnit.size();
    }
    if (position != content.size())
      throw std::runtime_error("unexpected prompt shape");
    ids.emplace_back(9);
    return ids;
  }
};

TtftOptions options() {
  TtftOptions result;
  result.input_tokens = 10;
  result.output_tokens = 4;
  result.continuation_tokens = 14;
  result.samples = 5;
  return result;
}

void run(Protocol &transport, const TtftOptions &o,
         std::vector<Json> &records) {
  run_ttft_benchmark(
      transport, o, [&](const Json &value) { records.push_back(value); },
      [] { return at(5ms); });
}

void SequenceWarmupAndControls() {
  Protocol transport;
  std::vector<Json> records;
  run(transport, options(), records);
  REQUIRE(records.size() == 1 + 18 + 3);
  const auto &manifest = records[0];
  REQUIRE(manifest.at("kind").as_string() == "manifest");
  REQUIRE(manifest.at("continuation_common_prefix_tokens").as_int() == 7);
  REQUIRE(manifest.at("sampling").at("presence_penalty").as_double() == 0.0);
  REQUIRE(manifest.at("model_identity").at("weight_version").is_null());
  REQUIRE(manifest.dump().find("private/model") == std::string::npos);
  REQUIRE(manifest.dump().find("private/tokenizer") == std::string::npos);
  REQUIRE(transport.stream_count == 18);
  std::vector<std::string> actions;
  for (const auto &action : transport.operations) {
    if (action == "flush" || action == "stream")
      actions.push_back(action);
  }
  std::vector<std::string> expected;
  for (int i = 0; i < 6; ++i) {
    expected.insert(expected.end(), {"flush", "stream", "stream", "stream"});
  }
  expected.emplace_back("flush");
  REQUIRE(actions == expected);
  for (const auto &payload : transport.payloads) {
    REQUIRE(payload.at("max_completion_tokens").as_int() == 4);
    REQUIRE(payload.at("temperature").as_double() == 1.0);
    REQUIRE(payload.at("top_p").as_double() == 0.95);
    REQUIRE(payload.at("top_k").as_int() == 20);
    REQUIRE(payload.at("min_p").as_double() == 0.0);
    REQUIRE(payload.at("presence_penalty").as_double() == 0.0);
    REQUIRE(payload.at("repetition_penalty").as_double() == 1.0);
    REQUIRE(payload.at("ignore_eos").as_bool());
    REQUIRE(
        payload.at("chat_template_kwargs").at("preserve_thinking").as_bool());
    REQUIRE(payload.at("chat_template_kwargs").at("enable_thinking").as_bool());
    REQUIRE(!payload.contains("seed"));
  }
  REQUIRE(records[1].at("kind").as_string() == "warmup");
  REQUIRE(records[1].at("cached_prompt_tokens").as_int() == 0);
  REQUIRE(records[1]
              .at("diagnostics")
              .at("cached_prompt_tokens_reported")
              .is_null());
  REQUIRE(records[2].at("cached_prompt_tokens").as_int() == 6);
  REQUIRE(records[2].at("shared_prefix_not_reported_cached").as_int() == 4);
  REQUIRE(records[3].at("common_prefix_tokens").as_int() == 7);
  REQUIRE(records[4].at("kind").as_string() == "sample");
  for (std::size_t i = 19; i < records.size(); ++i) {
    REQUIRE(records[i].at("kind").as_string() == "summary");
    REQUIRE(records[i].at("samples").as_int() == 5);
    REQUIRE(records[i].at("ttft_s_samples").size() == 5);
    REQUIRE(records[i].at("ttft_s_mean").as_double() == 0.005);
  }
}

void NoContinuationAndZeroReporting() {
  for (const bool explicit_zero : {false, true}) {
    Protocol transport;
    transport.explicit_zero = explicit_zero;
    transport.omit_cached = true;
    auto o = options();
    o.samples = 1;
    o.continuation_tokens = 0;
    std::vector<Json> records;
    run(transport, o, records);
    REQUIRE(records.size() == 7);
    REQUIRE(transport.stream_count == 4);
    for (std::size_t i = 1; i <= 4; ++i) {
      REQUIRE(records[i].at("cached_prompt_tokens").as_int() == 0);
      REQUIRE(records[i]
                  .at("diagnostics")
                  .at("cached_prompt_tokens_reported")
                  .is_null() == !explicit_zero);
    }
  }
}

void InvalidServerStopsBeforeCalibrationOrFlush() {
  const std::vector<std::pair<std::string, Json>> changes{
      {"enable_cache_report", false}, {"disable_radix_cache", true},
      {"served_model_name", "other"}, {"max_running_requests", 2},
      {"context_length", 16},         {"max_total_tokens", 16},
      {"startup_time", nullptr},      {"startup_time", Object{}}};
  for (const auto &[field, value] : changes) {
    Protocol transport;
    transport.server.as_object()[field] = value;
    std::vector<Json> records;
    REQUIRE(native_test::throws([&] { run(transport, options(), records); }));
    REQUIRE(transport.operations == std::vector<std::string>{"server"});
    REQUIRE(records.empty());
  }
  Protocol transport;
  transport.model.as_object()["has_image_understanding"] = true;
  std::vector<Json> records;
  REQUIRE(native_test::throws([&] { run(transport, options(), records); }));
  REQUIRE(transport.operations ==
          std::vector<std::string>({"server", "model"}));
}

void FailedOrIncompleteStreamsNeverFlushAgainOrSummarize() {
  for (int failure = 0; failure < 7; ++failure) {
    Protocol transport;
    if (failure == 0)
      transport.fail_stream = 2;
    if (failure == 1)
      transport.truncate_done = true;
    if (failure == 2)
      transport.no_output = true;
    if (failure == 3)
      transport.wrong_counts = true;
    if (failure == 4)
      transport.wrong_finish = true;
    if (failure == 5)
      transport.flush_hit = true;
    if (failure == 6)
      transport.malformed_cached = 11;
    std::vector<Json> records;
    REQUIRE(native_test::throws([&] { run(transport, options(), records); }));
    REQUIRE(transport.operations.back() == "stream");
    REQUIRE(std::count(transport.operations.begin(), transport.operations.end(),
                       "flush") == 1);
    REQUIRE(
        std::none_of(records.begin(), records.end(), [](const Json &record) {
          return record.at("kind").as_string() == "summary";
        }));
    REQUIRE(records.size() == (failure == 0 ? 2 : 1));
  }
}

void ConfigurationRestartAndLiveModelChangeWithholdSummaries() {
  for (int change = 0; change < 3; ++change) {
    Protocol transport;
    transport.restart = change == 0;
    transport.change_config = change == 1;
    transport.change_model = change == 2;
    std::vector<Json> records;
    REQUIRE(native_test::throws([&] { run(transport, options(), records); }));
    REQUIRE(records.size() == 19);
    REQUIRE(transport.stream_count == 18);
  }
}

void InvalidCalibrationCannotStartInference() {
  for (const bool invalid_ids : {true, false}) {
    Protocol transport;
    transport.bad_token_ids = invalid_ids;
    transport.unreachable_count = !invalid_ids;
    auto o = options();
    if (!invalid_ids) {
      o.input_tokens = 3;
      o.continuation_tokens = 0;
    }
    std::vector<Json> records;
    REQUIRE(native_test::throws([&] { run(transport, o, records); }));
    REQUIRE(records.empty());
    REQUIRE(transport.stream_count == 0);
    REQUIRE(std::count(transport.operations.begin(), transport.operations.end(),
                       "flush") == 0);
  }
}

void OptionsFailBeforeNetwork() {
  const std::vector<std::vector<std::string_view>> invalid{
      {"--samples", "0"},
      {"--samples", "101"},
      {"--samples", "2", "--samples", "2"},
      {"--timeout", "nan"},
      {"--timeout", "0"},
      {"--timeout", "3601"},
      {"--input-tokens", "199001"},
      {"--input-tokens", "199000", "--output-tokens", "1001"},
      {"--continuation-tokens", "6213"},
      {"--continuation-tokens", "-1"},
      {"--base-url", "http://localhost:30000/v1"},
      {"--unknown", "1"},
      {"--model"}};
  for (const auto &args : invalid) {
    REQUIRE(native_test::throws(
        [&] { static_cast<void>(parse_ttft_arguments(args)); }));
  }
  REQUIRE(parse_ttft_arguments({}).samples == 5);
  auto o = options();
  o.output_tokens = 200000;
  Protocol transport;
  std::vector<Json> records;
  REQUIRE(native_test::throws([&] { run(transport, o, records); }));
  REQUIRE(transport.operations.empty());
}

void ReceiptWriteFailureStopsFurtherWork() {
  Protocol transport;
  int receipts = 0;
  REQUIRE(native_test::throws([&] {
    run_ttft_benchmark(
        transport, options(),
        [&](const Json &) {
          if (++receipts == 2)
            throw std::runtime_error("output failed");
        },
        [] { return at(5ms); });
  }));
  REQUIRE(transport.stream_count == 1);
  REQUIRE(transport.operations.back() == "stream");
}
} // namespace

int main() {
  using Test = void (*)();
  const std::array<std::pair<std::string_view, Test>, 8> tests{
      {{"SequenceWarmupAndControls", SequenceWarmupAndControls},
       {"NoContinuationAndZeroReporting", NoContinuationAndZeroReporting},
       {"InvalidServerStopsBeforeCalibrationOrFlush",
        InvalidServerStopsBeforeCalibrationOrFlush},
       {"FailedOrIncompleteStreamsNeverFlushAgainOrSummarize",
        FailedOrIncompleteStreamsNeverFlushAgainOrSummarize},
       {"ConfigurationRestartAndLiveModelChangeWithholdSummaries",
        ConfigurationRestartAndLiveModelChangeWithholdSummaries},
       {"InvalidCalibrationCannotStartInference",
        InvalidCalibrationCannotStartInference},
       {"OptionsFailBeforeNetwork", OptionsFailBeforeNetwork},
       {"ReceiptWriteFailureStopsFurtherWork",
        ReceiptWriteFailureStopsFurtherWork}}};
  return native_test::run("ttft_benchmark_test", tests);
}
