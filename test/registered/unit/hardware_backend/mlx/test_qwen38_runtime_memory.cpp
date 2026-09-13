#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "mlx/memory.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;

namespace {

constexpr size_t kGiB = size_t{1024} * 1024 * 1024;

void Check(const char* setting, bool valid, size_t expected) {
  if (setting == nullptr) {
    if (unsetenv("SGLANG_MLX_CACHE_LIMIT_GB") != 0) {
      throw std::runtime_error("unsetenv failed");
    }
  } else if (setenv("SGLANG_MLX_CACHE_LIMIT_GB", setting, 1) != 0) {
    throw std::runtime_error("setenv failed");
  }
  mx::set_cache_limit(2 * kGiB);
  std::string error;
  try {
    // The invalid model config stops construction before checkpoint I/O.
    // Runtime settings must already be applied at that boundary.
    sglang::mlx_qwen38::Engine engine(MlxQwen38Config{}, "");
  } catch (const std::runtime_error& caught) {
    error = caught.what();
  }
  const size_t observed = mx::set_cache_limit(2 * kGiB);
  const std::string_view expected_error = valid
      ? "invalid Qwen3.8 config"
      : "SGLANG_MLX_CACHE_LIMIT_GB must be a nonnegative finite size";
  if (error != expected_error || observed != expected) {
    throw std::runtime_error(
        "runtime memory setting failed: " +
        std::string(setting == nullptr ? "<unset>" : setting));
  }
}

}  // namespace

int main() {
  try {
    Check(nullptr, true, 2 * kGiB);
    Check("0", true, 0);
    Check("0.25", true, kGiB / 4);
    Check("1", true, kGiB);
    Check("1e0", true, kGiB);
    for (const char* invalid :
         {"", "-1", "nan", "inf", "1GB", "1e300", "18446744073709551616"}) {
      Check(invalid, false, 2 * kGiB);
    }
    std::cout << "native runtime cache configuration passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
