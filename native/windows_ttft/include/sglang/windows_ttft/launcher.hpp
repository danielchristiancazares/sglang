#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <sglang/benchmark/json.hpp>

namespace sglang::windows_ttft {

namespace fs = std::filesystem;
using Json = benchmark::JsonValue;
using Environment = std::map<std::string, std::string, std::less<>>;

struct Options {
  std::string command;
  fs::path repo{"."};
  fs::path target;
  fs::path draft;
  std::string runtime_tag;
  std::vector<fs::path> dependency_roots;
  bool autotune_extend{false};
};

struct Plan {
  Options options;
  std::string fingerprint;
  fs::path state_root;
  fs::path view_root;
  fs::path executable;
  Environment environment;
  Json manifest;
  std::vector<std::string> arguments;
};

[[nodiscard]] Options parse_arguments(std::span<const std::string_view> args);
[[nodiscard]] std::string_view help() noexcept;
[[nodiscard]] std::string path_text(const fs::path &path);
[[nodiscard]] fs::path text_path(std::string_view text);
[[nodiscard]] Environment selected_environment(const Options &options,
                                                const Environment &inherited);
// Reads inputs only. No files, processes, or CUDA contexts are created.
[[nodiscard]] Plan make_plan(Options options, const Environment &inherited);
[[nodiscard]] Json receipt(const Plan &plan);

// OS-backed, checkout-wide nonblocking lock. Hold across preparation AND the
// complete foreground child lifetime. A stale filename is not a stale lock.
class WorkspaceLock {
public:
  explicit WorkspaceLock(const fs::path &state_root);
  ~WorkspaceLock();
  WorkspaceLock(const WorkspaceLock &) = delete;
  WorkspaceLock &operator=(const WorkspaceLock &) = delete;

private:
  std::intptr_t handle_{-1};
};

// Caller owns WorkspaceLock. Publishes atomically, never repairs/replaces an
// existing view. Reuse requires matching manifest, copies and hard-link identity.
void prepare_views(const Plan &plan);

// Platform helpers are exposed for CPU-only contract tests.
[[nodiscard]] std::string file_identity(const fs::path &path);
[[nodiscard]] bool is_link(const fs::path &path);
[[nodiscard]] std::wstring quote_windows_argument(std::wstring_view argument);
[[nodiscard]] std::wstring utf8_to_wide(std::string_view text);
[[nodiscard]] std::string wide_to_utf8(std::wstring_view text);
[[nodiscard]] Environment current_environment();
void check_windows_environment(const Environment &environment);
// Native Windows only; inherits the foreground console, owns only its child
// job, and waits for exit. No shell, Python program, or dependency repair.
[[nodiscard]] int serve(const Plan &plan);

} // namespace sglang::windows_ttft
