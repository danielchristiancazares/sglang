#include <cmath>
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
        (static_cast<int>(i % 521) * multiplier + 19) % 223 - 111;
    values[i] = static_cast<float>(centered) / 128.0f;
  }
  return values;
}

bool ExactEqual(
    std::string_view label,
    const mx::array& actual,
    const mx::array& expected) {
  mx::eval(actual, expected);
  if (actual.shape() != expected.shape() || actual.dtype() != expected.dtype()) {
    std::cerr << label << " metadata mismatch\n";
    return false;
  }
  const float* actual_data = actual.data<float>();
  const float* expected_data = expected.data<float>();
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual_data[i] != expected_data[i]) {
      std::cerr << label << " mismatch at " << i << ": " << actual_data[i]
                << " != " << expected_data[i] << '\n';
      return false;
    }
  }
  return true;
}

using ArrayPair = std::pair<mx::array, mx::array>;

ArrayPair Reference(
    const mx::array& q,
    const mx::array& k,
    float q_scale,
    float k_scale) {
  return {
      q_scale * mx::fast::rms_norm(q, std::nullopt, 1e-6f),
      k_scale * mx::fast::rms_norm(k, std::nullopt, 1e-6f)};
}

bool CheckCase(int rows, int width, float q_scale, float k_scale) {
  const std::size_t size =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(width);
  const auto q_values = MakeValues(size, 37);
  const auto k_values = MakeValues(size, 71);
  mx::array q = mx::astype(
      mx::array(q_values.data(), {rows, 1, width}, mx::float32),
      mx::bfloat16);
  mx::array k = mx::astype(
      mx::array(k_values.data(), {rows, 1, width}, mx::float32),
      mx::bfloat16);

  auto actual = sglang::mlx_qwen38::normalize_gated_delta_qk(
      q, k, q_scale, k_scale, 1e-6f);
  auto expected = Reference(q, k, q_scale, k_scale);
  return ExactEqual("q", actual.first, expected.first) &&
      ExactEqual("k", actual.second, expected.second);
}

bool CheckOutstandingOutputs() {
  constexpr int kRows = 16;
  constexpr int kWidth = 128;
  constexpr int kOutstanding = 12;
  const auto q_values = MakeValues(kRows * kWidth, 43);
  mx::array q = mx::astype(
      mx::array(q_values.data(), {1, 1, kRows, kWidth}, mx::float32),
      mx::bfloat16);

  std::vector<std::pair<ArrayPair, ArrayPair>> pending;
  pending.reserve(kOutstanding);
  for (int i = 0; i < kOutstanding; ++i) {
    const auto k_values = MakeValues(kRows * kWidth, 47 + i * 2);
    mx::array k = mx::astype(
        mx::array(k_values.data(), {1, 1, kRows, kWidth}, mx::float32),
        mx::bfloat16);
    const float inv = 1.0f / std::sqrt(static_cast<float>(kWidth));
    auto actual = sglang::mlx_qwen38::normalize_gated_delta_qk(
        q, k, inv * inv, inv, 1e-6f);
    pending.emplace_back(std::move(actual), Reference(q, k, inv * inv, inv));
  }

  for (auto& [actual, expected] : pending) {
    if (!ExactEqual("outstanding q", actual.first, expected.first) ||
        !ExactEqual("outstanding k", actual.second, expected.second)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  const float inv = 1.0f / std::sqrt(128.0f);
  if (!CheckCase(16, 128, inv * inv, inv) ||
      !CheckCase(3, 257, 0.03125f, 0.176776692f) ||
      !CheckOutstandingOutputs()) {
    return 1;
  }
  std::cout << "qwen38 gated-delta q/k normalization parity passed\n";
  return 0;
}
