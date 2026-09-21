#include <sglang/windows_ttft/launcher.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {
int run(const std::vector<std::string> &owned) {
  using namespace sglang::windows_ttft;
  std::vector<std::string_view> args(owned.begin(), owned.end());
  if (args.size() == 1 && (args[0] == "--help" || args[0] == "-h")) {
    std::cout << help();
    return std::cout ? 0 : 1;
  }
  const auto options = parse_arguments(args);
  const auto environment = current_environment();
  if (options.command == "serve") check_windows_environment(environment);
  const auto plan = make_plan(options, environment);
  if (options.command == "plan") {
    std::cout << receipt(plan).dump() << '\n';
    return std::cout ? 0 : 1;
  }
  WorkspaceLock lock(plan.state_root);
  prepare_views(plan);
  std::cout << receipt(plan).dump() << '\n' << std::flush;
  if (!std::cout) throw std::runtime_error("failed to write launch receipt");
  return options.command == "serve" ? serve(plan) : 0;
}
} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t *argv[]) {
#else
int main(int argc, char *argv[]) {
#endif
  try {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
#ifdef _WIN32
      args.push_back(sglang::windows_ttft::wide_to_utf8(argv[i]));
#else
      args.emplace_back(argv[i]);
#endif
    }
    return run(args);
  } catch (const std::exception &error) {
    std::cerr << "serve_windows_ttft: " << error.what() << '\n';
    return 1;
  }
}
