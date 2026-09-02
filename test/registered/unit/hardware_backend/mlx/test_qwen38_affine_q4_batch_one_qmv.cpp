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

bool CheckQ4FusedSwiGluParity(int input_features, int output_features) {
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
  std::uint64_t state = 0xd1b54a32d192ed03ULL;
  for (std::size_t index = 0; index < packed_elements; ++index) {
    gate_packed[index] = Next(state);
    up_packed[index] = Next(state);
  }
  std::vector<float> gate_scale_values(parameter_elements);
  std::vector<float> gate_bias_values(parameter_elements);
  std::vector<float> up_scale_values(parameter_elements);
  std::vector<float> up_bias_values(parameter_elements);
  for (std::size_t index = 0; index < parameter_elements; ++index) {
    gate_scale_values[index] =
        0.0025f + static_cast<float>(index % 17) * 0.000125f;
    gate_bias_values[index] =
        (static_cast<float>(index % 11) - 5.0f) * 0.00025f;
    up_scale_values[index] =
        0.003f + static_cast<float>(index % 13) * 0.0001f;
    up_bias_values[index] =
        (static_cast<float>(index % 7) - 3.0f) * 0.000375f;
  }
  std::vector<float> input_values(input_features);
  for (int index = 0; index < input_features; ++index) {
    input_values[static_cast<std::size_t>(index)] =
        std::sin(static_cast<float>(index) * 0.013f) * 0.3125f;
  }

  mx::array gate_weights(
      gate_packed.data(),
      {output_features, static_cast<int>(packed_columns)},
      mx::uint32);
  mx::array up_weights(
      up_packed.data(),
      {output_features, static_cast<int>(packed_columns)},
      mx::uint32);
  mx::array gate_scales = mx::astype(
      mx::array(gate_scale_values.data(),
                {output_features, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  mx::array gate_biases = mx::astype(
      mx::array(gate_bias_values.data(),
                {output_features, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  mx::array up_scales = mx::astype(
      mx::array(up_scale_values.data(),
                {output_features, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  mx::array up_biases = mx::astype(
      mx::array(up_bias_values.data(),
                {output_features, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 1, input_features}, mx::float32),
      mx::bfloat16);
  sglang::mlx_qwen38::QLinear gate{
      gate_weights, gate_scales, gate_biases, 64, 4, true};
  sglang::mlx_qwen38::QLinear up{
      up_weights, up_scales, up_biases, 64, 4, true};
  if (!sglang::mlx_qwen38::prepare_fused_q4_raw_decode_parameters(gate, up)) {
    std::cerr << "failed to prepare raw fused Q4 parameters\n";
    return false;
  }

  mx::array expected = mx::astype(
      sglang::mlx_qwen38::silu(
          sglang::mlx_qwen38::affine_q4_qmv_batch_one(gate, input)) *
          sglang::mlx_qwen38::affine_q4_qmv_batch_one(up, input),
      mx::float32);
  mx::array actual = mx::astype(
      sglang::mlx_qwen38::affine_q4_fused_swiglu_batch_one(gate, up, input),
      mx::float32);
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
  std::cout << "Q4 fused SwiGLU K=" << input_features
            << " N=" << output_features
            << " max_abs=" << maximum_absolute_error
            << " mismatches=" << mismatches << '\n';
  return maximum_absolute_error == 0.0f && mismatches == 0;
}

bool CheckQ4FusedSwiGluSigmoidBoundary() {
  constexpr int kInputFeatures = 512;
  constexpr int kOutputFeatures = 32;
  constexpr float kGateValue = -6.84375f;
  const std::size_t packed_columns = kInputFeatures / 8;
  const std::size_t parameter_columns = kInputFeatures / 64;
  const std::size_t packed_elements = kOutputFeatures * packed_columns;
  const std::size_t parameter_elements =
      kOutputFeatures * parameter_columns;

  std::vector<std::uint32_t> packed(packed_elements, 0);
  std::vector<float> zero_scales(parameter_elements, 0.0f);
  std::vector<float> gate_biases(parameter_elements, kGateValue);
  std::vector<float> up_biases(parameter_elements, 1.0f);
  std::vector<float> input_values(kInputFeatures, 0.0f);
  input_values[0] = 1.0f;

  mx::array weights(
      packed.data(),
      {kOutputFeatures, static_cast<int>(packed_columns)},
      mx::uint32);
  mx::array scales = mx::astype(
      mx::array(zero_scales.data(),
                {kOutputFeatures, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  mx::array gate_bias = mx::astype(
      mx::array(gate_biases.data(),
                {kOutputFeatures, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  mx::array up_bias = mx::astype(
      mx::array(up_biases.data(),
                {kOutputFeatures, static_cast<int>(parameter_columns)},
                mx::float32),
      mx::bfloat16);
  mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 1, kInputFeatures}, mx::float32),
      mx::bfloat16);
  sglang::mlx_qwen38::QLinear gate{
      weights, scales, gate_bias, 64, 4, true};
  sglang::mlx_qwen38::QLinear up{
      weights, scales, up_bias, 64, 4, true};
  if (!sglang::mlx_qwen38::prepare_fused_q4_raw_decode_parameters(gate, up)) {
    std::cerr << "failed to prepare raw fused Q4 boundary parameters\n";
    return false;
  }

  mx::array expected = mx::astype(
      sglang::mlx_qwen38::silu(
          sglang::mlx_qwen38::affine_q4_qmv_batch_one(gate, input)) *
          sglang::mlx_qwen38::affine_q4_qmv_batch_one(up, input),
      mx::float32);
  mx::array actual = mx::astype(
      sglang::mlx_qwen38::affine_q4_fused_swiglu_batch_one(gate, up, input),
      mx::float32);
  mx::eval(expected, actual);

  std::size_t mismatches = 0;
  const float *expected_data = expected.data<float>();
  const float *actual_data = actual.data<float>();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    mismatches += expected_data[index] != actual_data[index];
  }
  std::cout << "Q4 fused SwiGLU sigmoid boundary gate=" << kGateValue
            << " expected=" << expected_data[0]
            << " actual=" << actual_data[0]
            << " mismatches=" << mismatches << '\n';
  return mismatches == 0;
}

bool CheckQ4FusedSwiGluFiniteBfloatDomain() {
  constexpr int kInputFeatures = 512;
  constexpr int kOutputFeatures = 65280;
  constexpr int kParameterColumns = kInputFeatures / 64;
  constexpr std::uint16_t kBfloatOne = UINT16_C(0x3f80);
  const std::size_t packed_columns = kInputFeatures / 8;
  const std::size_t packed_elements =
      static_cast<std::size_t>(kOutputFeatures) * packed_columns;
  const std::size_t parameter_elements =
      static_cast<std::size_t>(kOutputFeatures) * kParameterColumns;

  std::vector<std::uint16_t> finite_patterns;
  finite_patterns.reserve(kOutputFeatures);
  for (unsigned int pattern = 0; pattern <= UINT16_MAX; ++pattern) {
    if ((pattern & 0x7f80U) != 0x7f80U) {
      finite_patterns.push_back(static_cast<std::uint16_t>(pattern));
    }
  }
  if (finite_patterns.size() != kOutputFeatures) {
    std::cerr << "unexpected finite BF16 pattern count\n";
    return false;
  }

  std::vector<std::uint32_t> packed(packed_elements, 0);
  std::vector<std::uint16_t> zero_scale_bits(parameter_elements, 0);
  std::vector<std::uint16_t> gate_bias_bits(parameter_elements);
  std::vector<std::uint16_t> up_bias_bits(parameter_elements, kBfloatOne);
  for (int row = 0; row < kOutputFeatures; ++row) {
    for (int column = 0; column < kParameterColumns; ++column) {
      gate_bias_bits[static_cast<std::size_t>(row) * kParameterColumns +
                     column] = finite_patterns[static_cast<std::size_t>(row)];
    }
  }
  std::vector<float> input_values(kInputFeatures, 0.0f);
  input_values[0] = 1.0f;

  const mx::array weights(
      packed.data(),
      {kOutputFeatures, static_cast<int>(packed_columns)},
      mx::uint32);
  const mx::Shape parameter_shape{kOutputFeatures, kParameterColumns};
  const mx::array scales = mx::view(
      mx::array(zero_scale_bits.data(), parameter_shape, mx::uint16),
      mx::bfloat16);
  const mx::array gate_bias = mx::view(
      mx::array(gate_bias_bits.data(), parameter_shape, mx::uint16),
      mx::bfloat16);
  const mx::array up_bias = mx::view(
      mx::array(up_bias_bits.data(), parameter_shape, mx::uint16),
      mx::bfloat16);
  const mx::array input = mx::astype(
      mx::array(input_values.data(), {1, 1, kInputFeatures}, mx::float32),
      mx::bfloat16);
  sglang::mlx_qwen38::QLinear gate{
      weights, scales, gate_bias, 64, 4, true};
  sglang::mlx_qwen38::QLinear up{
      weights, scales, up_bias, 64, 4, true};
  if (!sglang::mlx_qwen38::prepare_fused_q4_raw_decode_parameters(gate, up)) {
    std::cerr << "failed to prepare exhaustive fused Q4 parameters\n";
    return false;
  }

  const mx::array gate_output =
      sglang::mlx_qwen38::affine_q4_qmv_batch_one(gate, input);
  const mx::array expected = sglang::mlx_qwen38::silu(gate_output);
  const mx::array actual =
      sglang::mlx_qwen38::affine_q4_fused_swiglu_batch_one(gate, up, input);
  mx::eval(gate_output, expected, actual);

  const auto *gate_output_bits = reinterpret_cast<const std::uint16_t *>(
      gate_output.data<mx::bfloat16_t>());
  const auto *expected_bits = reinterpret_cast<const std::uint16_t *>(
      expected.data<mx::bfloat16_t>());
  const auto *actual_bits = reinterpret_cast<const std::uint16_t *>(
      actual.data<mx::bfloat16_t>());
  std::size_t changed_gate_patterns = 0;
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < finite_patterns.size(); ++index) {
    changed_gate_patterns += gate_output_bits[index] != finite_patterns[index];
    mismatches += expected_bits[index] != actual_bits[index];
  }
  std::cout << "Q4 fused SwiGLU finite BF16 patterns="
            << finite_patterns.size()
            << " changed_by_qmv=" << changed_gate_patterns
            << " mismatches=" << mismatches << '\n';
  return mismatches == 0;
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
      !CheckQ4FusedSwiGluParity(512, 64) ||
      !CheckQ4FusedSwiGluParity(5120, 17408) ||
      !CheckQ4FusedSwiGluSigmoidBoundary() ||
      !CheckQ4FusedSwiGluFiniteBfloatDomain() ||
      !RejectsUnsupportedQ4BatchOneShape()) {
    return 1;
  }
  return 0;
}
