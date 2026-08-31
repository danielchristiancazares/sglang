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
        (static_cast<int>(i % 523) * multiplier + 29) % 227 - 113;
    values[i] = static_cast<float>(centered);
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

mx::array Reference(
    const mx::array& recurrent_out,
    const mx::array& z,
    const mx::array& weight) {
  mx::array normed = mx::fast::rms_norm(recurrent_out, weight, 1e-6f);
  mx::array gated =
      sglang::mlx_qwen38::silu(mx::astype(z, mx::float32)) *
      mx::astype(normed, mx::float32);
  return mx::astype(gated, z.dtype());
}

bool CheckCase(int rows, int width) {
  const std::size_t size =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(width);
  const auto out_values = MakeValues(size, 37);
  const auto z_values = MakeValues(size, 71);
  const auto weight_values = MakeValues(static_cast<std::size_t>(width), 97);
  mx::array recurrent_out(
      out_values.data(), {rows, 1, width}, mx::float32);
  mx::array z = mx::astype(
      mx::array(z_values.data(), {rows, 1, width}, mx::float32),
      mx::bfloat16);
  mx::array weight = mx::astype(
      mx::array(weight_values.data(), {width}, mx::float32), mx::bfloat16);

  mx::array actual = sglang::mlx_qwen38::gated_delta_norm_gate(
      recurrent_out, z, weight, 1e-6f);
  return ExactEqual("gated output", actual, Reference(recurrent_out, z, weight));
}

bool CheckOutstandingOutputs() {
  constexpr int kRows = 48;
  constexpr int kWidth = 128;
  constexpr int kOutstanding = 12;
  const auto out_values = MakeValues(kRows * kWidth, 41);
  const auto weight_values = MakeValues(kWidth, 83);
  mx::array recurrent_out(
      out_values.data(), {1, 1, kRows, kWidth}, mx::float32);
  mx::array weight = mx::astype(
      mx::array(weight_values.data(), {kWidth}, mx::float32), mx::bfloat16);

  std::vector<std::pair<mx::array, mx::array>> pending;
  pending.reserve(kOutstanding);
  for (int i = 0; i < kOutstanding; ++i) {
    const auto z_values = MakeValues(kRows * kWidth, 47 + i * 2);
    mx::array z = mx::astype(
        mx::array(z_values.data(), {1, 1, kRows, kWidth}, mx::float32),
        mx::bfloat16);
    mx::array actual = sglang::mlx_qwen38::gated_delta_norm_gate(
        recurrent_out, z, weight, 1e-6f);
    pending.emplace_back(std::move(actual), Reference(recurrent_out, z, weight));
  }

  for (std::size_t i = 0; i < pending.size(); ++i) {
    auto& [actual, expected] = pending[i];
    if (!ExactEqual("outstanding gated output", actual, expected)) {
      std::cerr << "outstanding output " << i << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckCase(48, 128) || !CheckCase(3, 257) ||
      !CheckOutstandingOutputs()) {
    return 1;
  }
  std::cout << "qwen38 gated-delta norm/gate parity passed\n";
  return 0;
}
