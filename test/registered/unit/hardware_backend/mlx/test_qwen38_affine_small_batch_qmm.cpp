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

bool CheckParity(int rows, int input_features, int output_features, int bits) {
  const auto input_values = MakeValues(
      static_cast<std::size_t>(rows) * input_features, 0.017f, 0.25f);
  const auto weight_values =
      MakeValues(static_cast<std::size_t>(output_features) * input_features,
                 0.013f, 0.125f);
  mx::array input = mx::astype(
      mx::array(input_values.data(), {1, rows, input_features}, mx::float32),
      mx::bfloat16);
  mx::array dense_weight =
      mx::astype(mx::array(weight_values.data(),
                           {output_features, input_features}, mx::float32),
                 mx::bfloat16);
  std::vector<mx::array> quantized =
      mx::quantize(dense_weight, 64, bits, "affine");
  sglang::mlx_qwen38::QLinear linear{quantized[0], quantized[1], quantized[2],
                                     64,           bits,         true};

  mx::array expected = mx::astype(
      mx::quantized_matmul(input, linear.w, linear.scales, linear.biases, true,
                           linear.group_size, linear.bits, "affine"),
      mx::float32);
  mx::array actual = mx::astype(
      sglang::mlx_qwen38::affine_qmm_small_batch(linear, input), mx::float32);
  const bool check_m8 = rows == 8 && bits == 4 && input_features % 256 == 0 &&
                        output_features % 16 == 0;
  mx::array m8 =
      check_m8
          ? mx::astype(sglang::mlx_qwen38::affine_qmm_m8_ksplit(linear, input),
                       mx::float32)
          : actual;
  mx::eval(expected, actual, m8);

  float maximum_absolute_error = 0.0f;
  float m8_maximum_absolute_error = 0.0f;
  const float *expected_data = expected.data<float>();
  const float *actual_data = actual.data<float>();
  const float *m8_data = m8.data<float>();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    maximum_absolute_error =
        std::max(maximum_absolute_error,
                 std::abs(expected_data[index] - actual_data[index]));
    m8_maximum_absolute_error =
        std::max(m8_maximum_absolute_error,
                 std::abs(expected_data[index] - m8_data[index]));
  }
  std::cout << "rows=" << rows << " K=" << input_features
            << " N=" << output_features << " bits=" << bits
            << " max_abs=" << maximum_absolute_error;
  if (check_m8) {
    std::cout << " m8_max_abs=" << m8_maximum_absolute_error;
  }
  std::cout << '\n';
  const float error_limit =
      input_features <= 128 ? 0.02f : (input_features <= 256 ? 0.04f : 0.25f);
  return std::isfinite(maximum_absolute_error) &&
         maximum_absolute_error <= error_limit &&
         std::isfinite(m8_maximum_absolute_error) &&
         m8_maximum_absolute_error <= error_limit;
}

bool RejectsInvalidShape() {
  constexpr int kRows = 8;
  constexpr int kInputFeatures = 96;
  constexpr int kOutputFeatures = 64;
  const auto input_values = MakeValues(
      static_cast<std::size_t>(kRows) * kInputFeatures, 0.017f, 0.25f);
  const auto weight_values =
      MakeValues(static_cast<std::size_t>(kOutputFeatures) * kInputFeatures,
                 0.013f, 0.125f);
  mx::array input = mx::astype(
      mx::array(input_values.data(), {1, kRows, kInputFeatures}, mx::float32),
      mx::bfloat16);
  mx::array dense_weight =
      mx::astype(mx::array(weight_values.data(),
                           {kOutputFeatures, kInputFeatures}, mx::float32),
                 mx::bfloat16);
  std::vector<mx::array> quantized =
      mx::quantize(dense_weight, 32, 4, "affine");
  sglang::mlx_qwen38::QLinear linear{
      quantized[0], quantized[1], quantized[2], 32, 4, true};
  try {
    (void)sglang::mlx_qwen38::affine_qmm_small_batch(linear, input);
  } catch (const std::runtime_error &error) {
    if (std::string_view(error.what()) !=
        "invalid small-batch affine QMM inputs") {
      return false;
    }
    try {
      (void)sglang::mlx_qwen38::affine_qmm_m8_ksplit(linear, input);
    } catch (const std::runtime_error &m8_error) {
      return std::string_view(m8_error.what()) ==
             "invalid M8 K-split affine QMM inputs";
    }
  }
  return false;
}

} // namespace

int main() {
  for (int bits : {2, 4}) {
    for (int rows : {2, 7, 8}) {
      if (!CheckParity(rows, 128, 128, bits)) {
        return 1;
      }
    }
  }
  if (!CheckParity(8, 256, 256, 4) || !CheckParity(8, 5120, 64, 4) ||
      !CheckParity(8, 256, 6144, 4) || !CheckParity(6, 64, 6144, 4) ||
      !RejectsInvalidShape()) {
    return 1;
  }
  std::cout << "qwen38 affine small-batch QMM parity passed\n";
  return 0;
}
