#include "sglang/benchmark/arguments.hpp"
#include "sglang/benchmark/http_client.hpp"
#include "sglang/benchmark/openai_benchmark.hpp"

#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

[[nodiscard]] std::vector<std::string_view>
arguments_after_program(int argc, char *argv[]) {
  std::vector<std::string_view> arguments;
  if (argc > 1)
    arguments.reserve(static_cast<std::size_t>(argc - 1));
  for (int index = 1; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }
  return arguments;
}

} // namespace

int main(int argc, char *argv[]) {
  using sglang::benchmark::ParseStatus;

  try {
    const auto arguments = arguments_after_program(argc, argv);
    const auto parsed =
        sglang::benchmark::parse_acceptance_arguments(arguments);
    if (parsed.status == ParseStatus::kHelp) {
      std::cout << sglang::benchmark::acceptance_help();
      if (!std::cout) {
        std::cerr << "bench_spec_acceptance: failed to write help to stdout\n";
        return 1;
      }
      return 0;
    }
    if (parsed.status == ParseStatus::kError) {
      std::cerr << "bench_spec_acceptance: error: " << parsed.message << '\n';
      return 2;
    }

    sglang::benchmark::SocketHttpTransport transport;
    const auto result =
        sglang::benchmark::run_acceptance_benchmark(transport, parsed.options);
    std::cout << result.dump() << '\n';
    if (!std::cout) {
      std::cerr << "bench_spec_acceptance: failed to write result to stdout\n";
      return 1;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "bench_spec_acceptance: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "bench_spec_acceptance: unknown failure\n";
    return 1;
  }
}
