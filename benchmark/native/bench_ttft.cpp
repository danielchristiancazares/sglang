#include <sglang/benchmark/http_client.hpp>
#include <sglang/benchmark/ttft_benchmark.hpp>

#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char *argv[]) {
  try {
    std::vector<std::string_view> arguments;
    for (int i = 1; i < argc; ++i) {
      arguments.emplace_back(argv[i]);
    }
    if (arguments.size() == 1 &&
        (arguments[0] == "--help" || arguments[0] == "-h")) {
      std::cout << sglang::benchmark::ttft_help();
      return std::cout ? 0 : 1;
    }
    const auto options = sglang::benchmark::parse_ttft_arguments(arguments);
    sglang::benchmark::SocketHttpTransport transport;
    sglang::benchmark::run_ttft_benchmark(
        transport, options, [](const sglang::benchmark::JsonValue &record) {
          std::cout << record.dump() << '\n' << std::flush;
          if (!std::cout) {
            throw std::runtime_error("failed to write JSONL receipt");
          }
        });
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "bench_ttft: " << error.what() << '\n';
    return 1;
  }
}
