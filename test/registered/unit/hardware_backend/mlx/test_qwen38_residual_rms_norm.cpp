#include <cstdint>
#include <cstddef>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;

namespace {

std::vector<float> MakeValues(std::size_t size, int multiplier) {
  std::vector<float> values(size);
  for (std::size_t i = 0; i < size; ++i) {
    const int centered =
        (static_cast<int>(i % 509) * multiplier + 23) % 211 - 105;
    values[i] = static_cast<float>(centered) / 128.0f;
  }
  return values;
}

bool ExactEqual(
    std::string_view label,
    const mx::array& actual,
    const mx::array& expected) {
  mx::array actual_float = mx::astype(actual, mx::float32);
  mx::array expected_float = mx::astype(expected, mx::float32);
  mx::eval(actual_float, expected_float);
  if (actual_float.shape() != expected_float.shape()) {
    std::cerr << label << " shape mismatch\n";
    return false;
  }
  const float* actual_data = actual_float.data<float>();
  const float* expected_data = expected_float.data<float>();
  for (std::size_t i = 0; i < actual_float.size(); ++i) {
    if (actual_data[i] != expected_data[i]) {
      std::cerr << label << " mismatch at " << i << ": " << actual_data[i]
                << " != " << expected_data[i] << '\n';
      return false;
    }
  }
  return true;
}

bool CheckCase(int rows, int width) {
  const std::size_t size =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(width);
  const auto x_values = MakeValues(size, 37);
  const auto residual_values = MakeValues(size, 71);
  const auto weight_values = MakeValues(static_cast<std::size_t>(width), 97);
  mx::array x = mx::astype(
      mx::array(x_values.data(), {rows, 1, width}, mx::float32),
      sglang::mlx_qwen38::activation_dtype());
  mx::array residual = mx::astype(
      mx::array(residual_values.data(), {rows, 1, width}, mx::float32),
      sglang::mlx_qwen38::activation_dtype());
  mx::array weight = mx::astype(
      mx::array(weight_values.data(), {width}, mx::float32), sglang::mlx_qwen38::activation_dtype());

  auto actual =
      sglang::mlx_qwen38::residual_rms_norm(x, residual, weight, 1e-6f);
  mx::array expected_residual = x + residual;
  mx::array expected_norm =
      mx::fast::rms_norm(expected_residual, weight, 1e-6f);
  return ExactEqual("residual", actual.first, expected_residual) &&
      ExactEqual("RMSNorm", actual.second, expected_norm);
}

bool CheckOutstandingOutputs() {
  constexpr int kWidth = 5120;
  constexpr int kOutstanding = 12;
  const auto x_values = MakeValues(kWidth, 31);
  const auto weight_values = MakeValues(kWidth, 89);
  mx::array x = mx::astype(
      mx::array(x_values.data(), {1, 1, kWidth}, mx::float32), sglang::mlx_qwen38::activation_dtype());
  mx::array weight = mx::astype(
      mx::array(weight_values.data(), {kWidth}, mx::float32), sglang::mlx_qwen38::activation_dtype());

  using ArrayPair = std::pair<mx::array, mx::array>;
  std::vector<std::pair<ArrayPair, ArrayPair>> pending;
  pending.reserve(kOutstanding);
  for (int i = 0; i < kOutstanding; ++i) {
    const auto residual_values = MakeValues(kWidth, 41 + i * 2);
    mx::array residual = mx::astype(
        mx::array(
            residual_values.data(), {1, 1, kWidth}, mx::float32),
        sglang::mlx_qwen38::activation_dtype());
    auto actual =
        sglang::mlx_qwen38::residual_rms_norm(x, residual, weight, 1e-6f);
    mx::array expected_residual = x + residual;
    mx::array expected_norm =
        mx::fast::rms_norm(expected_residual, weight, 1e-6f);
    pending.emplace_back(
        std::move(actual),
        ArrayPair{std::move(expected_residual), std::move(expected_norm)});
  }

  for (auto& [actual, expected] : pending) {
    if (!ExactEqual("outstanding residual", actual.first, expected.first) ||
        !ExactEqual("outstanding RMSNorm", actual.second, expected.second)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckCase(1, 5120) || !CheckCase(3, 257) ||
      !CheckOutstandingOutputs()) {
    return 1;
  }
  std::cout << "qwen38 residual RMSNorm parity passed\n";
  return 0;
}
