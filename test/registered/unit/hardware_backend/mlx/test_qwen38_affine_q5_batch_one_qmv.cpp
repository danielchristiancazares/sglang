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

bool CheckQ5BatchOneParity(int input_features, int output_features) {
  const auto input_values = MakeValues(input_features, 0.017f, 0.25f);
  const auto weight_values =
      MakeValues(static_cast<std::size_t>(output_features) * input_features,
                 0.013f, 0.125f);
  mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 1, input_features}, mx::float32),
      mx::bfloat16);
  mx::array dense_weight =
      mx::astype(mx::array(weight_values.data(),
                           {output_features, input_features}, mx::float32),
                 mx::bfloat16);
  std::vector<mx::array> quantized =
      mx::quantize(dense_weight, 64, 5, "affine");
  sglang::mlx_qwen38::QLinear linear{
      quantized[0], quantized[1], quantized[2], 64, 5, true};

  mx::array expected = mx::astype(
      mx::quantized_matmul(input, linear.w, linear.scales, linear.biases, true,
                           linear.group_size, linear.bits, "affine"),
      mx::float32);
  mx::array actual = mx::astype(
      sglang::mlx_qwen38::affine_q5_qmv_batch_one(linear, input), mx::float32);
  mx::eval(expected, actual);

  float maximum_absolute_error = 0.0f;
  const float *expected_data = expected.data<float>();
  const float *actual_data = actual.data<float>();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    maximum_absolute_error =
        std::max(maximum_absolute_error,
                 std::abs(expected_data[index] - actual_data[index]));
  }
  std::cout << "Q5 batch-one K=" << input_features << " N=" << output_features
            << " max_abs=" << maximum_absolute_error << '\n';
  return std::isfinite(maximum_absolute_error) &&
         maximum_absolute_error <= 0.25f;
}

bool RejectsUnsupportedQ5BatchOneShape() {
  constexpr int kInputFeatures = 256;
  constexpr int kOutputFeatures = 32;
  const auto input_values = MakeValues(kInputFeatures, 0.017f, 0.25f);
  const auto weight_values =
      MakeValues(static_cast<std::size_t>(kOutputFeatures) * kInputFeatures,
                 0.013f, 0.125f);
  mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 1, kInputFeatures}, mx::float32),
      mx::bfloat16);
  mx::array dense_weight =
      mx::astype(mx::array(weight_values.data(),
                           {kOutputFeatures, kInputFeatures}, mx::float32),
                 mx::bfloat16);
  std::vector<mx::array> quantized =
      mx::quantize(dense_weight, 64, 5, "affine");
  sglang::mlx_qwen38::QLinear linear{
      quantized[0], quantized[1], quantized[2], 64, 5, true};
  try {
    (void)sglang::mlx_qwen38::affine_q5_qmv_batch_one(linear, input);
  } catch (const std::runtime_error &error) {
    return std::string_view(error.what()) ==
           "unsupported batch-one affine Q5 QMV shape";
  }
  return false;
}

} // namespace

int main() {
  if (!CheckQ5BatchOneParity(512, 64) || !CheckQ5BatchOneParity(5120, 128) ||
      !CheckQ5BatchOneParity(17408, 32) ||
      !RejectsUnsupportedQ5BatchOneShape()) {
    return 1;
  }
  return 0;
}
