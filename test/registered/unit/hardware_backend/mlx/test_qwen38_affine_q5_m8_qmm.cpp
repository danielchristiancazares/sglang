#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;

namespace {

std::vector<float> MakeValues(std::size_t size, float frequency, float scale) {
  std::vector<float> values(size);
  for (std::size_t index = 0; index < size; ++index) {
    values[index] = std::sin(static_cast<float>(index) * frequency) * scale;
  }
  return values;
}

bool CheckQ5M8Parity(int input_features, int output_features) {
  constexpr int kRows = 8;
  const auto input_values = MakeValues(
      static_cast<std::size_t>(kRows) * input_features, 0.017f, 0.25f);
  const auto weight_values = MakeValues(
      static_cast<std::size_t>(output_features) * input_features,
      0.013f,
      0.125f);
  mx::array input = mx::astype(
      mx::array(
          input_values.data(), {1, kRows, input_features}, mx::float32),
      mx::bfloat16);
  mx::array dense_weight = mx::astype(
      mx::array(
          weight_values.data(),
          {output_features, input_features},
          mx::float32),
      mx::bfloat16);
  std::vector<mx::array> quantized =
      mx::quantize(dense_weight, 64, 5, "affine");
  sglang::mlx_qwen38::QLinear linear{
      quantized[0], quantized[1], quantized[2], 64, 5, true};

  mx::array expected = mx::astype(
      mx::quantized_matmul(
          input,
          linear.w,
          linear.scales,
          linear.biases,
          true,
          linear.group_size,
          linear.bits,
          "affine"),
      mx::float32);
  mx::array actual = mx::astype(
      sglang::mlx_qwen38::affine_qmm_m8_ksplit(linear, input),
      mx::float32);
  mx::eval(expected, actual);

  float maximum_absolute_error = 0.0f;
  const float* expected_data = expected.data<float>();
  const float* actual_data = actual.data<float>();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    maximum_absolute_error = std::max(
        maximum_absolute_error,
        std::abs(expected_data[index] - actual_data[index]));
  }
  std::cout << "Q5 M8 K=" << input_features << " N=" << output_features
            << " max_abs=" << maximum_absolute_error << '\n';
  return std::isfinite(maximum_absolute_error) &&
      maximum_absolute_error <= 0.25f;
}

bool RejectsUnsupportedQ5M8Shape() {
  constexpr int kRows = 8;
  constexpr int kInputFeatures = 256;
  constexpr int kOutputFeatures = 32;
  const auto input_values = MakeValues(
      static_cast<std::size_t>(kRows) * kInputFeatures, 0.017f, 0.25f);
  const auto weight_values = MakeValues(
      static_cast<std::size_t>(kOutputFeatures) * kInputFeatures,
      0.013f,
      0.125f);
  mx::array input = mx::astype(
      mx::array(
          input_values.data(), {1, kRows, kInputFeatures}, mx::float32),
      mx::bfloat16);
  mx::array dense_weight = mx::astype(
      mx::array(
          weight_values.data(),
          {kOutputFeatures, kInputFeatures},
          mx::float32),
      mx::bfloat16);
  std::vector<mx::array> quantized =
      mx::quantize(dense_weight, 64, 5, "affine");
  sglang::mlx_qwen38::QLinear linear{
      quantized[0], quantized[1], quantized[2], 64, 5, true};
  try {
    (void)sglang::mlx_qwen38::affine_qmm_m8_ksplit(linear, input);
  } catch (const std::runtime_error& error) {
    return std::string_view(error.what()) ==
        "unsupported M8 K-split affine QMM shape";
  }
  return false;
}

}  // namespace

int main() {
  if (!CheckQ5M8Parity(512, 256) || !CheckQ5M8Parity(5120, 64) ||
      !CheckQ5M8Parity(17408, 32) || !RejectsUnsupportedQ5M8Shape()) {
    return 1;
  }
  return 0;
}
