#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include "sglang/benchmark/openai_benchmark.hpp"

namespace sb = sglang::benchmark;

int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      std::cerr << "usage: bench_qwen38_workload BASE_URL MODEL PROMPT_FILE RESULT_JSON MAX_TOKENS\n";
      return 2;
    }
    if (std::filesystem::file_size(argv[3]) > 32U * 1024U * 1024U)
      throw std::runtime_error("prompt exceeds 32 MiB");
    std::ifstream input(argv[3], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open prompt");
    sb::StreamRequestOptions options;
    options.base_url = argv[1];
    options.model = argv[2];
    options.content.assign(std::istreambuf_iterator<char>(input), {});
    if (options.content.empty()) throw std::runtime_error("prompt is empty");
    const std::string count(argv[5]);
    const auto [end, error] = std::from_chars(count.data(), count.data() + count.size(), options.output_tokens);
    if (error != std::errc{} || end != count.data() + count.size() ||
        options.output_tokens < 1 || options.output_tokens >= 131072)
      throw std::runtime_error("invalid output token budget");
    options.timeout_seconds = 14400.0;
    options.temperature = 1.0;
    options.top_p = 0.95;
    options.top_k = 20;
    options.min_p = 0.0;
    options.presence_penalty = 0.0;
    options.repetition_penalty = 1.0;
    options.seed = 42;
    options.enable_thinking = true;
    options.ignore_eos = false;
    options.include_output_text = true;
    options.reasoning_effort = "xhigh";
    sb::SocketHttpTransport transport;
    const auto tokenized = sb::request_json(transport,
        options.base_url + "/v1/tokenize", sb::JsonValue::object{
          {"model", options.model}, {"messages", sb::messages_for(options.content)},
          {"chat_template_kwargs", sb::chat_template_kwargs(true)},
          {"reasoning_effort", "xhigh"}}, 600.0);
    const auto& tokens = tokenized.at("tokens").as_array();
    if (tokens.empty() || tokenized.at("count").as_int() != static_cast<std::int64_t>(tokens.size()))
      throw std::runtime_error("inconsistent tokenization response");
    if (tokens.size() + static_cast<std::size_t>(options.output_tokens) > 131072)
      throw std::runtime_error("request exceeds the 131072-token context");
    const std::filesystem::path result_path(argv[4]);
    if (std::filesystem::exists(result_path))
      throw std::runtime_error("result path already exists");
    std::ofstream ids(result_path.string() + ".prompt.ids", std::ios::out | std::ios::noreplace);
    if (!ids) throw std::runtime_error("cannot create prompt ID receipt");
    for (const auto& token : tokens) ids << token.as_int() << '\n';
    ids.close();
    if (!ids) throw std::runtime_error("cannot finish prompt ID receipt");
    sb::flush_cache(transport, options.base_url, 600.0, sb::Backend::kSglang, 0);
    auto result = sb::stream_request(transport, options);
    const auto prompt_count = result.at("prompt_tokens").as_int();
    const auto completion_count = result.at("completion_tokens").as_int();
    if (prompt_count != static_cast<std::int64_t>(tokens.size()) ||
        completion_count < 1 || completion_count > options.output_tokens ||
        result.at("total_tokens").as_int() != prompt_count + completion_count)
      throw std::runtime_error("inconsistent measured token counts");
    auto& receipt = result.as_object();
    receipt.emplace("model", options.model);
    receipt.emplace("prompt_file", std::filesystem::absolute(argv[3]).string());
    receipt.emplace("reasoning_effort", "xhigh");
    receipt.emplace("ignore_eos", false);
    receipt.emplace("requested_max_tokens", options.output_tokens);
    receipt.emplace("temperature", options.temperature);
    receipt.emplace("top_p", *options.top_p);
    receipt.emplace("top_k", *options.top_k);
    receipt.emplace("presence_penalty", *options.presence_penalty);
    receipt.emplace("seed", *options.seed);
    const std::string encoded = result.dump();
    std::ofstream output(result_path, std::ios::out | std::ios::noreplace);
    if (!output) throw std::runtime_error("cannot create result receipt");
    output << encoded << '\n';
    output.close();
    if (!output) throw std::runtime_error("cannot finish result receipt");
    std::cout << encoded << '\n';
    // Preserve bounded screens, but distinguish them from naturally completed work.
    if (result.at("finish_reason").as_string() != "stop") return 3;
    if (result.at("reasoning_chars").as_int() == 0 ||
        result.at("content_chars").as_int() == 0) return 4;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "workload benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
