#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;

namespace {

std::uint32_t Next(std::uint64_t &state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return static_cast<std::uint32_t>(state);
}

bool CheckQ4BatchOneParity(int input_features, int output_features) {
  const std::size_t packed_columns =
      static_cast<std::size_t>(input_features) / 8;
  const std::size_t packed_elements =
      static_cast<std::size_t>(output_features) * packed_columns;
  const std::size_t parameter_columns =
      static_cast<std::size_t>(input_features) / 64;
  const std::size_t parameter_elements =
      static_cast<std::size_t>(output_features) * parameter_columns;

  std::vector<std::uint32_t> packed(packed_elements);
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (auto &value : packed) {
    value = Next(state);
  }
  std::vector<float> scale_values(parameter_elements);
  std::vector<float> bias_values(parameter_elements);
  for (std::size_t index = 0; index < parameter_elements; ++index) {
    scale_values[index] = 0.0025f + static_cast<float>(index % 17) * 0.000125f;
    bias_values[index] = (static_cast<float>(index % 11) - 5.0f) * 0.00025f;
  }
  std::vector<float> input_values(input_features);
  for (int index = 0; index < input_features; ++index) {
    input_values[static_cast<std::size_t>(index)] =
        std::sin(static_cast<float>(index) * 0.017f) * 0.25f;
  }

  mx::array weights(packed.data(),
                    {output_features, static_cast<int>(packed_columns)},
                    mx::uint32);
  mx::array scales = mx::astype(
      mx::array(scale_values.data(),
                {output_features, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  mx::array biases = mx::astype(
      mx::array(bias_values.data(),
                {output_features, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 1, input_features}, mx::float32),
      mx::bfloat16);
  sglang::mlx_qwen38::QLinear linear{weights, scales, biases, 64, 4, true};

  mx::array expected = mx::astype(
      mx::quantized_matmul(input, linear.w, linear.scales, linear.biases, true,
                           linear.group_size, linear.bits, "affine"),
      mx::float32);
  mx::array actual = mx::astype(
      sglang::mlx_qwen38::affine_q4_qmv_batch_one(linear, input), mx::float32);
  mx::eval(expected, actual);

  float maximum_absolute_error = 0.0f;
  std::size_t mismatches = 0;
  const float *expected_data = expected.data<float>();
  const float *actual_data = actual.data<float>();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const float absolute_error =
        std::abs(expected_data[index] - actual_data[index]);
    maximum_absolute_error = std::max(maximum_absolute_error, absolute_error);
    mismatches += absolute_error != 0.0f;
  }
  std::cout << "Q4 batch-one K=" << input_features << " N=" << output_features
            << " max_abs=" << maximum_absolute_error
            << " mismatches=" << mismatches << '\n';
  return maximum_absolute_error == 0.0f && mismatches == 0;
}

bool RejectsUnsupportedQ4BatchOneShape() {
  constexpr int kInputFeatures = 256;
  constexpr int kOutputFeatures = 32;
  std::vector<std::uint32_t> packed(static_cast<std::size_t>(kOutputFeatures) *
                                    kInputFeatures / 8);
  std::vector<float> parameters(
      static_cast<std::size_t>(kOutputFeatures) * kInputFeatures / 64, 0.125f);
  std::vector<float> input_values(kInputFeatures, 0.25f);
  mx::array weights(packed.data(), {kOutputFeatures, kInputFeatures / 8},
                    mx::uint32);
  mx::array scales =
      mx::astype(mx::array(parameters.data(),
                           {kOutputFeatures, kInputFeatures / 64}, mx::float32),
                 mx::bfloat16);
  mx::array biases =
      mx::astype(mx::array(parameters.data(),
                           {kOutputFeatures, kInputFeatures / 64}, mx::float32),
                 mx::bfloat16);
  mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 1, kInputFeatures}, mx::float32),
      mx::bfloat16);
  sglang::mlx_qwen38::QLinear linear{weights, scales, biases, 64, 4, true};
  try {
    (void)sglang::mlx_qwen38::affine_q4_qmv_batch_one(linear, input);
  } catch (const std::runtime_error &error) {
    return std::string_view(error.what()) ==
           "unsupported batch-one affine Q4 QMV shape";
  }
  return false;
}

} // namespace

int main() {
  if (!CheckQ4BatchOneParity(512, 64) || !CheckQ4BatchOneParity(5120, 128) ||
      !CheckQ4BatchOneParity(5120, 17408) ||
      !RejectsUnsupportedQ4BatchOneShape()) {
    return 1;
  }
  return 0;
}
