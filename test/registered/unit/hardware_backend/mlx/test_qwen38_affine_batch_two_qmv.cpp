#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

float MaximumAbsoluteError(const mx::array &expected, const mx::array &actual) {
  const mx::array expected_float = mx::astype(expected, mx::float32);
  const mx::array actual_float = mx::astype(actual, mx::float32);
  mx::eval(expected_float, actual_float);
  const float *expected_data = expected_float.data<float>();
  const float *actual_data = actual_float.data<float>();
  float maximum = 0.0f;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    maximum =
        std::max(maximum, std::abs(expected_data[index] - actual_data[index]));
  }
  return maximum;
}

bool SameBfloat16(const mx::array &lhs, const mx::array &rhs) {
  mx::eval(lhs, rhs);
  if (lhs.shape() != rhs.shape() || lhs.dtype() != mx::bfloat16 ||
      rhs.dtype() != mx::bfloat16) {
    return false;
  }
  const auto *lhs_data = lhs.data<mx::bfloat16_t>();
  const auto *rhs_data = rhs.data<mx::bfloat16_t>();
  return std::equal(lhs_data, lhs_data + lhs.size(), rhs_data);
}

bool CheckQ5BatchTwoParity(int input_features, int output_features) {
  const std::size_t packed_columns =
      static_cast<std::size_t>(input_features) * 5 / 32;
  const std::size_t packed_elements =
      static_cast<std::size_t>(output_features) * packed_columns;
  const std::size_t parameter_columns =
      static_cast<std::size_t>(input_features) / 64;
  const std::size_t parameter_elements =
      static_cast<std::size_t>(output_features) * parameter_columns;

  std::vector<std::uint32_t> packed(packed_elements);
  std::uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
  for (auto &value : packed) {
    value = Next(state);
  }
  std::vector<float> scales(parameter_elements);
  std::vector<float> biases(parameter_elements);
  for (std::size_t index = 0; index < parameter_elements; ++index) {
    scales[index] = 0.0025f + static_cast<float>(index % 17) * 0.000125f;
    biases[index] = (static_cast<float>(index % 11) - 5.0f) * 0.00025f;
  }
  std::vector<float> input_values(static_cast<std::size_t>(2) * input_features);
  for (std::size_t index = 0; index < input_values.size(); ++index) {
    input_values[index] = std::sin(static_cast<float>(index) * 0.017f) * 0.25f;
  }

  const mx::array weights(packed.data(),
                          {output_features, static_cast<int>(packed_columns)},
                          mx::uint32);
  const mx::array scale_array = mx::astype(
      mx::array(scales.data(),
                {output_features, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  const mx::array bias_array = mx::astype(
      mx::array(biases.data(),
                {output_features, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  const mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 2, input_features}, mx::float32),
      mx::bfloat16);
  const sglang::mlx_qwen38::QLinear linear{weights, scale_array, bias_array,
                                           64,      5,           true};

  const mx::array expected =
      mx::quantized_matmul(input, linear.w, linear.scales, linear.biases, true,
                           linear.group_size, linear.bits, "affine");
  const mx::array actual =
      sglang::mlx_qwen38::affine_q5_qmv_batch_two(linear, input);

  if (setenv("SGLANG_MLX_NATIVE_Q5_BATCH_TWO_QMV", "1", 1) != 0) {
    throw std::runtime_error("failed to enable Q5 batch-two dispatch");
  }
  const mx::array dispatched = linear(input);
  if (unsetenv("SGLANG_MLX_NATIVE_Q5_BATCH_TWO_QMV") != 0) {
    throw std::runtime_error("failed to disable Q5 batch-two dispatch");
  }

  const float maximum_absolute_error = MaximumAbsoluteError(expected, actual);
  const bool dispatch_matches = SameBfloat16(actual, dispatched);
  std::cout << "Q5 batch-two K=" << input_features << " N=" << output_features
            << " max_abs=" << maximum_absolute_error
            << " dispatch_matches=" << dispatch_matches << '\n';
  return expected.shape() == mx::Shape{1, 2, output_features} &&
         actual.shape() == expected.shape() &&
         std::isfinite(maximum_absolute_error) &&
         maximum_absolute_error <= 0.015625f && dispatch_matches;
}

bool CheckQ4BatchTwoParity(int input_features, int output_features) {
  const std::size_t packed_columns =
      static_cast<std::size_t>(input_features) / 8;
  const std::size_t packed_elements =
      static_cast<std::size_t>(output_features) * packed_columns;
  const std::size_t parameter_columns =
      static_cast<std::size_t>(input_features) / 64;
  const std::size_t parameter_elements =
      static_cast<std::size_t>(output_features) * parameter_columns;

  std::vector<std::uint32_t> packed(packed_elements);
  std::uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
  for (auto &value : packed) {
    value = Next(state);
  }
  std::vector<float> scales(parameter_elements);
  std::vector<float> biases(parameter_elements);
  for (std::size_t index = 0; index < parameter_elements; ++index) {
    scales[index] = 0.0025f + static_cast<float>(index % 17) * 0.000125f;
    biases[index] = (static_cast<float>(index % 11) - 5.0f) * 0.00025f;
  }
  std::vector<float> input_values(static_cast<std::size_t>(2) * input_features);
  for (std::size_t index = 0; index < input_values.size(); ++index) {
    input_values[index] = std::sin(static_cast<float>(index) * 0.017f) * 0.25f;
  }

  const mx::array weights(packed.data(),
                          {output_features, static_cast<int>(packed_columns)},
                          mx::uint32);
  const mx::array scale_array = mx::astype(
      mx::array(scales.data(),
                {output_features, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  const mx::array bias_array = mx::astype(
      mx::array(biases.data(),
                {output_features, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  const mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 2, input_features}, mx::float32),
      mx::bfloat16);
  const sglang::mlx_qwen38::QLinear linear{weights, scale_array, bias_array,
                                           64,      4,           true};

  const mx::array expected =
      mx::quantized_matmul(input, linear.w, linear.scales, linear.biases, true,
                           linear.group_size, linear.bits, "affine");
  const mx::array actual =
      sglang::mlx_qwen38::affine_q4_qmv_batch_two(linear, input);

  if (setenv("SGLANG_MLX_NATIVE_Q4_BATCH_TWO_QMV", "1", 1) != 0) {
    throw std::runtime_error("failed to enable Q4 batch-two dispatch");
  }
  const mx::array dispatched = linear(input);
  if (unsetenv("SGLANG_MLX_NATIVE_Q4_BATCH_TWO_QMV") != 0) {
    throw std::runtime_error("failed to disable Q4 batch-two dispatch");
  }

  const float maximum_absolute_error = MaximumAbsoluteError(expected, actual);
  const bool exact = SameBfloat16(expected, actual);
  const bool dispatch_matches = SameBfloat16(actual, dispatched);
  std::cout << "Q4 batch-two K=" << input_features << " N=" << output_features
            << " max_abs=" << maximum_absolute_error << " exact=" << exact
            << " dispatch_matches=" << dispatch_matches << '\n';
  return expected.shape() == mx::Shape{1, 2, output_features} &&
         actual.shape() == expected.shape() && exact && dispatch_matches;
}

bool CheckQ4BatchTwoFusedParity(int input_features, int output_features) {
  const std::size_t packed_columns =
      static_cast<std::size_t>(input_features) / 8;
  const std::size_t packed_elements =
      static_cast<std::size_t>(output_features) * packed_columns;
  const std::size_t parameter_columns =
      static_cast<std::size_t>(input_features) / 64;
  const std::size_t parameter_elements =
      static_cast<std::size_t>(output_features) * parameter_columns;

  std::vector<std::uint32_t> gate_packed(packed_elements);
  std::vector<std::uint32_t> up_packed(packed_elements);
  std::uint64_t state = UINT64_C(0xd1b54a32d192ed03);
  for (std::size_t index = 0; index < packed_elements; ++index) {
    gate_packed[index] = Next(state);
    up_packed[index] = Next(state);
  }
  std::vector<float> gate_scales(parameter_elements);
  std::vector<float> gate_biases(parameter_elements);
  std::vector<float> up_scales(parameter_elements);
  std::vector<float> up_biases(parameter_elements);
  for (std::size_t index = 0; index < parameter_elements; ++index) {
    gate_scales[index] = 0.0025f + static_cast<float>(index % 17) * 0.000125f;
    gate_biases[index] = (static_cast<float>(index % 11) - 5.0f) * 0.00025f;
    up_scales[index] = 0.003f + static_cast<float>(index % 13) * 0.0001f;
    up_biases[index] = (static_cast<float>(index % 7) - 3.0f) * 0.000375f;
  }
  std::vector<float> input_values(static_cast<std::size_t>(2) * input_features);
  for (std::size_t index = 0; index < input_values.size(); ++index) {
    input_values[index] =
        std::sin(static_cast<float>(index) * 0.013f) * 0.3125f;
  }

  const mx::array gate_weights(
      gate_packed.data(), {output_features, static_cast<int>(packed_columns)},
      mx::uint32);
  const mx::array up_weights(
      up_packed.data(), {output_features, static_cast<int>(packed_columns)},
      mx::uint32);
  const mx::Shape parameter_shape{output_features,
                                  static_cast<int>(parameter_columns)};
  const mx::array gate_scale_array =
      mx::astype(mx::array(gate_scales.data(), parameter_shape, mx::float32),
                 mx::bfloat16);
  const mx::array gate_bias_array =
      mx::astype(mx::array(gate_biases.data(), parameter_shape, mx::float32),
                 mx::bfloat16);
  const mx::array up_scale_array = mx::astype(
      mx::array(up_scales.data(), parameter_shape, mx::float32), mx::bfloat16);
  const mx::array up_bias_array = mx::astype(
      mx::array(up_biases.data(), parameter_shape, mx::float32), mx::bfloat16);
  const mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 2, input_features}, mx::float32),
      mx::bfloat16);
  sglang::mlx_qwen38::QLinear gate{
      gate_weights, gate_scale_array, gate_bias_array, 64, 4, true};
  const sglang::mlx_qwen38::QLinear up{
      up_weights, up_scale_array, up_bias_array, 64, 4, true};
  if (!sglang::mlx_qwen38::prepare_fused_q4_raw_decode_parameters(gate, up)) {
    return false;
  }

  const mx::array gate_output =
      mx::quantized_matmul(input, gate.w, gate.scales, gate.biases, true,
                           gate.group_size, gate.bits, "affine");
  const mx::array expected =
      sglang::mlx_qwen38::silu(gate_output) *
      mx::quantized_matmul(input, up.w, up.scales, up.biases, true,
                           up.group_size, up.bits, "affine");
  const mx::array actual =
      sglang::mlx_qwen38::affine_q4_fused_swiglu_batch_two(gate, up, input);
  const float maximum_absolute_error = MaximumAbsoluteError(expected, actual);
  std::cout << "Q4 fused batch-two K=" << input_features
            << " N=" << output_features << " max_abs=" << maximum_absolute_error
            << '\n';
  return expected.shape() == mx::Shape{1, 2, output_features} &&
         actual.shape() == expected.shape() &&
         std::isfinite(maximum_absolute_error) &&
         maximum_absolute_error <= 0.015625f;
}

bool CheckQ4BatchTwoSigmoidBoundary() {
  constexpr int kInputFeatures = 512;
  constexpr int kOutputFeatures = 32;
  constexpr float kGateValue = -6.84375f;
  constexpr int kParameterColumns = kInputFeatures / 64;
  const std::size_t packed_elements =
      static_cast<std::size_t>(kOutputFeatures) * kInputFeatures / 8;
  const std::size_t parameter_elements =
      static_cast<std::size_t>(kOutputFeatures) * kParameterColumns;

  std::vector<std::uint32_t> packed(packed_elements, 0);
  std::vector<float> zero_scales(parameter_elements, 0.0f);
  std::vector<float> gate_biases(parameter_elements, kGateValue);
  std::vector<float> up_biases(parameter_elements, 1.0f);
  std::vector<float> input_values(static_cast<std::size_t>(2) * kInputFeatures,
                                  0.0f);
  input_values[0] = 1.0f;
  input_values[kInputFeatures] = 1.0f;

  const mx::array weights(packed.data(), {kOutputFeatures, kInputFeatures / 8},
                          mx::uint32);
  const mx::Shape parameter_shape{kOutputFeatures, kParameterColumns};
  const mx::array scales =
      mx::astype(mx::array(zero_scales.data(), parameter_shape, mx::float32),
                 mx::bfloat16);
  const mx::array gate_bias =
      mx::astype(mx::array(gate_biases.data(), parameter_shape, mx::float32),
                 mx::bfloat16);
  const mx::array up_bias = mx::astype(
      mx::array(up_biases.data(), parameter_shape, mx::float32), mx::bfloat16);
  const mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 2, kInputFeatures}, mx::float32),
      mx::bfloat16);
  sglang::mlx_qwen38::QLinear gate{weights, scales, gate_bias, 64, 4, true};
  const sglang::mlx_qwen38::QLinear up{weights, scales, up_bias, 64, 4, true};
  if (!sglang::mlx_qwen38::prepare_fused_q4_raw_decode_parameters(gate, up)) {
    return false;
  }

  const mx::array gate_output =
      mx::quantized_matmul(input, gate.w, gate.scales, gate.biases, true,
                           gate.group_size, gate.bits, "affine");
  const mx::array expected = sglang::mlx_qwen38::silu(gate_output);
  const mx::array actual =
      sglang::mlx_qwen38::affine_q4_fused_swiglu_batch_two(gate, up, input);
  const bool matches = SameBfloat16(expected, actual);
  std::cout << "Q4 fused batch-two sigmoid boundary gate=" << kGateValue
            << " matches=" << matches << '\n';
  return matches;
}

bool RejectsInvalidBatchTwoInputs() {
  constexpr int kInputFeatures = 512;
  constexpr int kOutputFeatures = 16;
  std::vector<std::uint32_t> q5_packed(
      static_cast<std::size_t>(kOutputFeatures) * kInputFeatures * 5 / 32);
  std::vector<std::uint32_t> q4_packed(
      static_cast<std::size_t>(kOutputFeatures) * kInputFeatures / 8);
  std::vector<float> parameters(
      static_cast<std::size_t>(kOutputFeatures) * kInputFeatures / 64, 0.125f);
  std::vector<float> input_values(static_cast<std::size_t>(2) * kInputFeatures,
                                  0.25f);
  const mx::array q5_weights(
      q5_packed.data(), {kOutputFeatures, kInputFeatures * 5 / 32}, mx::uint32);
  const mx::array q4_weights(q4_packed.data(),
                             {kOutputFeatures, kInputFeatures / 8}, mx::uint32);
  const mx::array scales =
      mx::astype(mx::array(parameters.data(),
                           {kOutputFeatures, kInputFeatures / 64}, mx::float32),
                 mx::bfloat16);
  const mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 2, kInputFeatures}, mx::float32),
      mx::bfloat16);
  const sglang::mlx_qwen38::QLinear q5{q5_weights, scales, scales, 64, 5, true};
  sglang::mlx_qwen38::QLinear q4{q4_weights, scales, scales, 64, 4, true};
  const sglang::mlx_qwen38::QLinear q4_up{q4_weights, scales, scales,
                                          64,         4,      true};

  const mx::array one_row = mx::slice(input, {0, 0, 0}, {1, 1, kInputFeatures});
  bool q4_qmv_invalid = false;
  try {
    (void)sglang::mlx_qwen38::affine_q4_qmv_batch_two(q4, one_row);
  } catch (const std::runtime_error &error) {
    q4_qmv_invalid = std::string_view(error.what()) ==
                     "invalid batch-two affine Q4 QMV inputs";
  }
  const mx::array short_input =
      mx::slice(input, {0, 0, 0}, {1, 2, kInputFeatures / 2});
  bool q4_qmv_unsupported = false;
  try {
    (void)sglang::mlx_qwen38::affine_q4_qmv_batch_two(q4, short_input);
  } catch (const std::runtime_error &error) {
    q4_qmv_unsupported = std::string_view(error.what()) ==
                         "unsupported batch-two affine Q4 QMV shape";
  }
  if (!q4_qmv_invalid || !q4_qmv_unsupported) {
    return false;
  }

  bool q4_invalid = false;
  try {
    (void)sglang::mlx_qwen38::affine_q4_fused_swiglu_batch_two(q4, q4_up,
                                                               input);
  } catch (const std::runtime_error &error) {
    q4_invalid = std::string_view(error.what()) ==
                 "invalid batch-two affine Q4 fused SwiGLU inputs";
  }
  if (!q4_invalid ||
      !sglang::mlx_qwen38::prepare_fused_q4_raw_decode_parameters(q4, q4_up)) {
    return false;
  }

  bool q4_unsupported = false;
  try {
    (void)sglang::mlx_qwen38::affine_q4_fused_swiglu_batch_two(q4, q4_up,
                                                               input);
  } catch (const std::runtime_error &error) {
    q4_unsupported = std::string_view(error.what()) ==
                     "unsupported batch-two affine Q4 fused SwiGLU shape";
  }

  bool q5_invalid = false;
  try {
    (void)sglang::mlx_qwen38::affine_q5_qmv_batch_two(q5, one_row);
  } catch (const std::runtime_error &error) {
    q5_invalid = std::string_view(error.what()) ==
                 "invalid batch-two affine Q5 QMV inputs";
  }
  return q4_unsupported && q5_invalid;
}

} // namespace

int main() {
  if (!CheckQ5BatchTwoParity(512, 64) || !CheckQ5BatchTwoParity(5120, 128) ||
      !CheckQ4BatchTwoParity(512, 64) || !CheckQ4BatchTwoParity(5120, 128) ||
      !CheckQ4BatchTwoFusedParity(512, 64) ||
      !CheckQ4BatchTwoFusedParity(5120, 128) ||
      !CheckQ4BatchTwoSigmoidBoundary() || !RejectsInvalidBatchTwoInputs()) {
    return 1;
  }
  std::cout << "qwen38 affine batch-two QMV parity passed\n";
  return 0;
}
