#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/memory.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"
namespace mx = mlx::core;
namespace {
int Positive(const char* value) {
  const std::string_view text(value);
  int result = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
  if (error != std::errc{} || end != text.data() + text.size() || result <= 0)
    throw std::runtime_error("expected a positive integer");
  return result;
}
mx::array Values(const mx::Shape& shape, int seed) {
  std::size_t count = 1;
  for (int dimension : shape) count *= static_cast<std::size_t>(dimension);
  std::vector<float> values(count);
  std::uint32_t state = static_cast<std::uint32_t>(seed);
  for (auto& value : values) {
    state = state * 1664525U + 1013904223U;
    value = (static_cast<float>(state >> 16) / 32768.0f - 1.0f) * 0.5f;
  }
  mx::array result = mx::astype(mx::array(values.data(), shape, mx::float32), mx::bfloat16);
  mx::eval(result);
  return result;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      std::cerr << "usage: bench_qwen38_q8_attention QUERIES ACTIVE_LENGTH CAPACITY ITERATIONS\n";
      return 2;
    }
    const int queries = Positive(argv[1]), length = Positive(argv[2]);
    const int capacity = Positive(argv[3]), iterations = Positive(argv[4]);
    if (queries > 1024 || length < queries || length > capacity ||
        capacity > 262144 || capacity % 64 != 0 || iterations > 10000)
      throw std::runtime_error("unsupported benchmark shape");
    mx::array q = Values({1, 24, queries, 256}, 17);
    auto k = mx::quantize(Values({1, 4, capacity, 256}, 31), 64, 8, "affine");
    auto v = mx::quantize(Values({1, 4, capacity, 256}, 71), 64, 8, "affine");
    mx::eval(k); mx::eval(v); mx::synchronize(); mx::clear_cache();
    const auto execute = [&] {
      return sglang::mlx_qwen38::fixed_q8_attention(q, k[0], k[1], k[2],
          v[0], v[1], v[2], length - queries, length);
    };
    if (setenv("SGLANG_MLX_NATIVE_Q8_SPLIT_VERIFY", "0", 1) != 0)
      throw std::runtime_error("cannot select control");
    mx::array control = mx::astype(execute(), mx::float32);
    mx::eval(control);
    if (setenv("SGLANG_MLX_NATIVE_Q8_SPLIT_VERIFY", "1", 1) != 0)
      throw std::runtime_error("cannot select split verifier");
    mx::array candidate = mx::astype(execute(), mx::float32);
    mx::eval(candidate);
    float maximum_error = 0;
    for (std::size_t i = 0; i < control.size(); ++i) {
      if (!std::isfinite(candidate.data<float>()[i]))
        throw std::runtime_error("nonfinite split output");
      maximum_error = std::max(maximum_error,
          std::abs(control.data<float>()[i] - candidate.data<float>()[i]));
    }
    std::cout << std::fixed << std::setprecision(9)
              << "queries=" << queries << " length=" << length
              << " capacity=" << capacity << " max_abs=" << maximum_error << '\n';
    if (maximum_error > 0.00390625f)
      throw std::runtime_error("split output exceeds numerical bound");
    for (int sample = 0; sample < 6; ++sample) {
      const int split = sample % 4 == 0 || sample % 4 == 3 ? 0 : 1;
      if (setenv("SGLANG_MLX_NATIVE_Q8_SPLIT_VERIFY", split ? "1" : "0", 1) != 0)
        throw std::runtime_error("cannot set benchmark arm");
      for (int warmup = 0; warmup < 3; ++warmup) {
        mx::eval(execute()); mx::synchronize();
      }
      const auto started = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < iterations; ++iteration) {
        mx::eval(execute()); mx::synchronize();
      }
      const double milliseconds = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count() / iterations;
      std::cout << "sample=" << sample << " split=" << split
                << " mean_ms=" << milliseconds << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Q8 attention benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
