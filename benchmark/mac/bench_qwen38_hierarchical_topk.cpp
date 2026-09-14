#include "mlx/mlx.h"
#include "qwen38_engine.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace mx = mlx::core;
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("usage: topk ROWS BLOCK");
    const int rows = std::stoi(argv[1]), block = std::stoi(argv[2]),
              vocab = 248320;
    if (rows < 1 || rows > 8 || block < 1)
      throw std::runtime_error("invalid shape");
    std::vector<float> values(static_cast<std::size_t>(rows) * vocab);
    std::uint32_t state = 17;
    for (auto &value : values) {
      state = state * 1664525U + 1013904223U;
      value = float(state >> 8) / 16777216.0f * 64.0f - 32.0f;
    }
    const auto logits =
        mx::astype(mx::array(values.data(), {rows, vocab}, mx::float32),
                   sglang::mlx_qwen38::activation_dtype());
    mx::eval(logits);
    std::vector<float> expected;
    std::cout << std::fixed << std::setprecision(9);
    for (int sample = 0; sample < 10; ++sample) {
      const bool candidate = sample % 2;
      setenv("SGLANG_MLX_NATIVE_HIERARCHICAL_TOPK", candidate ? argv[2] : "0",
             1);
      const auto run = [&] {
        const auto ids = sglang::mlx_qwen38::sampling_topk_indices(logits, 20);
        return mx::sort(mx::take_along_axis(logits, ids, -1), -1);
      };
      auto result = mx::astype(run(), mx::float32);
      mx::eval(result);
      mx::synchronize();
      std::vector<float> current(result.data<float>(),
                                 result.data<float>() + result.size());
      if (sample == 0)
        expected = current;
      if (current != expected)
        throw std::runtime_error("top-k scores changed");
      for (int i = 0; i < 10; ++i) {
        mx::eval(run());
        mx::synchronize();
      }
      const auto started = std::chrono::steady_clock::now();
      for (int i = 0; i < 100; ++i) {
        mx::eval(run());
        mx::synchronize();
      }
      std::cout << "rows=" << rows << " block=" << block << " sample=" << sample
                << " candidate=" << candidate << " milliseconds="
                << std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - started)
                           .count() /
                       100
                << " topk_values_equal=1\n";
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
